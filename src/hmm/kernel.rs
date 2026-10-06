//! Per-site HMM arithmetic: one forward step, one backward step, and the
//! blocked accumulation of expected transition counts.
//!
//! The state dimension is padded to a multiple of [`TILE`] with zero
//! probabilities, so every loop runs over whole tiles. Padded states never
//! receive mass: their emission, prior and transition entries are zero.
//!
//! PSMC transition matrices are semiseparable (see [`Semiseparable`]), which
//! makes the forward and backward steps O(n) instead of O(n²); other
//! matrices use the dense kernels.

use ndarray::Array2;

use super::obs::ObsTable;

/// Width (in f64) of the column tiles the matrix-vector products are blocked into.
pub(crate) const TILE: usize = 8;

/// Rows of the expected-count matrix handled per micro-tile in [`Kernel::accumulate_transitions`].
const ROW_TILE: usize = 4;
const _: () = assert!(TILE.is_multiple_of(ROW_TILE));

/// `y[j] = sum_r x[r] * m[r][j]` for a row-major `np x np` matrix `m`
/// (`np` a multiple of [`TILE`]).
///
/// Each output tile is accumulated in registers over all rows, an axpy per
/// row, so there is no loop-carried reduction across `j`.
#[inline(always)]
fn vec_mat(x: &[f64], m: &[f64], np: usize, y: &mut [f64]) {
    assert!(np.is_multiple_of(TILE) && x.len() == np && m.len() == np * np && y.len() == np);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the lengths were checked above.
    unsafe {
        neon::vec_mat(x.as_ptr(), m.as_ptr(), np, y.as_mut_ptr())
    }
    #[cfg(not(target_arch = "aarch64"))]
    portable::vec_mat(x, m, np, y)
}

/// `m[i][j] += sum_b u[b][i] * w[b][j]` for `i < n_rows_used` (rounded up to
/// [`ROW_TILE`]) and all `j`; `u`, `w` hold `rows` rows of `np` values.
#[inline(always)]
fn outer_accumulate(u: &[f64], w: &[f64], rows: usize, np: usize, n: usize, m: &mut [f64]) {
    assert!(np.is_multiple_of(TILE) && n <= np);
    assert!(u.len() >= rows * np && w.len() >= rows * np && m.len() == np * np);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the lengths were checked above.
    unsafe {
        neon::outer_accumulate(u.as_ptr(), w.as_ptr(), rows, np, n, m.as_mut_ptr())
    }
    #[cfg(not(target_arch = "aarch64"))]
    portable::outer_accumulate(u, w, rows, np, n, m)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    //! NEON kernels. Fused multiply-adds are explicit, and every product has
    //! enough independent accumulators (16) to cover the FMA latency.

    use super::ROW_TILE;
    use std::arch::aarch64::*;

    /// Columns `col..col + 2V` of `y = xᵀ m`, with rows interleaved over `P`
    /// accumulator sets that are added at the end.
    #[inline(always)]
    unsafe fn vec_mat_tile<const V: usize, const P: usize>(
        x: *const f64,
        m: *const f64,
        np: usize,
        col: usize,
        y: *mut f64,
    ) {
        unsafe {
            let mut acc = [[vdupq_n_f64(0.0); V]; P];
            let mut r = 0;
            while r < np {
                for (p, acc_p) in acc.iter_mut().enumerate() {
                    let s = vld1q_dup_f64(x.add(r + p));
                    let row = m.add((r + p) * np + col);
                    for (v, a) in acc_p.iter_mut().enumerate() {
                        *a = vfmaq_f64(*a, s, vld1q_f64(row.add(2 * v)));
                    }
                }
                r += P;
            }
            for v in 0..V {
                let mut t = acc[0][v];
                for acc_p in &acc[1..] {
                    t = vaddq_f64(t, acc_p[v]);
                }
                vst1q_f64(y.add(col + 2 * v), t);
            }
        }
    }

    /// `np` must be a multiple of 8; `x`, `y` hold `np` values, `m` `np * np`.
    #[inline(always)]
    pub(super) unsafe fn vec_mat(x: *const f64, m: *const f64, np: usize, y: *mut f64) {
        unsafe {
            let mut col = 0;
            while col + 16 <= np {
                vec_mat_tile::<8, 2>(x, m, np, col, y);
                col += 16;
            }
            if col < np {
                vec_mat_tile::<4, 4>(x, m, np, col, y);
            }
        }
    }

    /// See [`super::outer_accumulate`]; micro-tiles of 4 rows by 8 columns.
    #[inline(always)]
    pub(super) unsafe fn outer_accumulate(
        u: *const f64,
        w: *const f64,
        rows: usize,
        np: usize,
        n: usize,
        m: *mut f64,
    ) {
        const _: () = assert!(ROW_TILE == 4);
        unsafe {
            let mut i0 = 0;
            while i0 < n {
                let mut j0 = 0;
                while j0 < np {
                    let mut acc = [[vdupq_n_f64(0.0); 4]; 4];
                    for b in 0..rows {
                        let up = u.add(b * np + i0);
                        let u01 = vld1q_f64(up);
                        let u23 = vld1q_f64(up.add(2));
                        let wp = w.add(b * np + j0);
                        let wv = [
                            vld1q_f64(wp),
                            vld1q_f64(wp.add(2)),
                            vld1q_f64(wp.add(4)),
                            vld1q_f64(wp.add(6)),
                        ];
                        for c in 0..4 {
                            acc[0][c] = vfmaq_laneq_f64::<0>(acc[0][c], wv[c], u01);
                            acc[1][c] = vfmaq_laneq_f64::<1>(acc[1][c], wv[c], u01);
                            acc[2][c] = vfmaq_laneq_f64::<0>(acc[2][c], wv[c], u23);
                            acc[3][c] = vfmaq_laneq_f64::<1>(acc[3][c], wv[c], u23);
                        }
                    }
                    for (r, acc_r) in acc.iter().enumerate() {
                        let mp = m.add((i0 + r) * np + j0);
                        for (c, a) in acc_r.iter().enumerate() {
                            let p = mp.add(2 * c);
                            vst1q_f64(p, vaddq_f64(vld1q_f64(p), *a));
                        }
                    }
                    j0 += 8;
                }
                i0 += ROW_TILE;
            }
        }
    }
}

// Compiled everywhere so that the aarch64 build tests it against NEON.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod portable {
    //! Plain Rust kernels for other targets (no FMA: `mul_add` would be a
    //! libm call on x86-64 builds without the `fma` target feature).

    use super::{ROW_TILE, TILE};

    pub(super) fn vec_mat(x: &[f64], m: &[f64], np: usize, y: &mut [f64]) {
        for (jt, y_tile) in y.as_chunks_mut::<TILE>().0.iter_mut().enumerate() {
            let col = jt * TILE;
            let mut acc = [0.0f64; TILE];
            for (row, &s) in m.chunks_exact(np).zip(x) {
                let r: &[f64; TILE] = row[col..col + TILE].try_into().unwrap();
                for c in 0..TILE {
                    acc[c] += s * r[c];
                }
            }
            *y_tile = acc;
        }
    }

    pub(super) fn outer_accumulate(
        u: &[f64],
        w: &[f64],
        rows: usize,
        np: usize,
        n: usize,
        m: &mut [f64],
    ) {
        for i0 in (0..n).step_by(ROW_TILE) {
            for j0 in (0..np).step_by(TILE) {
                let mut acc = [[0.0f64; TILE]; ROW_TILE];
                for b in 0..rows {
                    let ub: &[f64; ROW_TILE] =
                        u[b * np + i0..b * np + i0 + ROW_TILE].try_into().unwrap();
                    let wb: &[f64; TILE] = w[b * np + j0..b * np + j0 + TILE].try_into().unwrap();
                    for r in 0..ROW_TILE {
                        for c in 0..TILE {
                            acc[r][c] += ub[r] * wb[c];
                        }
                    }
                }
                for r in 0..ROW_TILE {
                    let m_row = &mut m[(i0 + r) * np + j0..(i0 + r) * np + j0 + TILE];
                    for c in 0..TILE {
                        m_row[c] += acc[r][c];
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn sum(v: &[f64]) -> f64 {
    let mut acc = [0.0f64; TILE];
    for chunk in v.as_chunks::<TILE>().0 {
        for c in 0..TILE {
            acc[c] += chunk[c];
        }
    }
    let mut s = 0.0;
    for a in acc {
        s += a;
    }
    s
}

/// Blocks of [`TILE`] states covered by the semiseparable kernel's stack
/// scratch (up to 256 states; larger models use the dense kernels).
const MAX_SEMISEP_BLOCKS: usize = 32;

/// Largest relative difference between a transition probability and its
/// semiseparable reconstruction for the structure to be used. PSMC matrices
/// reconstruct to within a few ulp (measured: < 1e-15).
const SEMISEP_TOL: f64 = 1e-13;

/// A transition matrix whose strictly lower and strictly upper triangles
/// are each rank one: `a[k][m] = f[k] g[m]` for `m < k`, `h[k] j[m]` for
/// `m > k`, and `d[k]` on the diagonal.
///
/// PSMC's matrix has this form (in `PsmcModel::param_recalculate`, row `k`
/// is `tmp3_k * q[m]` with `q[m] = tmp_k * q_aux[m]` below the diagonal and
/// `(alpha[m] - alpha[m + 1]) * tmp2_k` above it), so a product with a
/// vector needs only one prefix sum and one suffix sum: the linear-time
/// forward and backward steps of Harris et al. (2014, *Decoding coalescent
/// hidden Markov models in linear time*, RECOMB). The factors are
/// recovered from the matrix itself and accepted only if every entry is
/// reproduced to [`SEMISEP_TOL`].
#[derive(Debug, Clone)]
struct Semiseparable {
    f: Vec<f64>,
    g: Vec<f64>,
    h: Vec<f64>,
    j: Vec<f64>,
    d: Vec<f64>,
}

impl Semiseparable {
    /// The factors of `a` (padded to `np` with zeros), or `None` if `a` is
    /// not semiseparable to within [`SEMISEP_TOL`] (or has negative,
    /// non-finite or underflowed reference entries).
    fn detect(a: &Array2<f64>, np: usize) -> Option<Self> {
        let n = a.nrows();
        if np / TILE > MAX_SEMISEP_BLOCKS || a.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return None;
        }
        let mut s = Self {
            f: vec![0.0; np],
            g: vec![0.0; np],
            h: vec![0.0; np],
            j: vec![0.0; np],
            d: vec![0.0; np],
        };
        for k in 0..n {
            s.d[k] = a[(k, k)];
        }
        if n < 2 {
            return Some(s);
        }
        // Reference row and column of each triangle: the last row and first
        // column span the lower triangle, the first row and last column the upper.
        let lower_ref = a[(n - 1, 0)];
        let upper_ref = a[(0, n - 1)];
        if !(lower_ref.is_normal() && upper_ref.is_normal()) {
            return None;
        }
        for m in 0..n - 1 {
            s.g[m] = a[(n - 1, m)];
            s.j[m + 1] = a[(0, m + 1)];
        }
        for k in 1..n {
            s.f[k] = a[(k, 0)] / lower_ref;
            s.h[k - 1] = a[(k - 1, n - 1)] / upper_ref;
        }
        for k in 0..n {
            for m in 0..n {
                let approx = match m.cmp(&k) {
                    std::cmp::Ordering::Less => s.f[k] * s.g[m],
                    std::cmp::Ordering::Greater => s.h[k] * s.j[m],
                    std::cmp::Ordering::Equal => continue,
                };
                let exact = a[(k, m)];
                // Entries are finite and non-negative here, so no NaN arises.
                if (approx - exact).abs() > SEMISEP_TOL * exact {
                    return None;
                }
            }
        }
        Some(s)
    }
}

#[inline(always)]
fn fmadd(a: f64, b: f64, c: f64) -> f64 {
    // One instruction on aarch64 and on x86-64 builds with FMA; elsewhere
    // `mul_add` would be a libm call.
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// `y[i] = p[i] * sum_{l > i} x[l] q[l] + r[i] * sum_{l < i} x[l] s[l] + d[i] x[i]`.
///
/// Each block of [`TILE`] entries first gets the terms of its own entries
/// (two short scans, independent across blocks); the scanned block totals
/// are then added in one elementwise pass. All terms are non-negative, so
/// the order of summation costs no accuracy.
#[inline(always)]
fn semisep_mul(x: &[f64], p: &[f64], q: &[f64], r: &[f64], s: &[f64], d: &[f64], y: &mut [f64]) {
    let n = x.len();
    let nb = n / TILE;
    assert!(nb <= MAX_SEMISEP_BLOCKS && n == nb * TILE);
    let (p, q, r, s, d, y) = (&p[..n], &q[..n], &r[..n], &s[..n], &d[..n], &mut y[..n]);
    let tile =
        |v: &[f64], b: usize| -> [f64; TILE] { v[b * TILE..(b + 1) * TILE].try_into().unwrap() };
    let mut suffix = [0.0f64; MAX_SEMISEP_BLOCKS];
    let mut prefix = [0.0f64; MAX_SEMISEP_BLOCKS];
    for b in 0..nb {
        let (xb, pb, qb, rb, sb, db) = (
            tile(x, b),
            tile(p, b),
            tile(q, b),
            tile(r, b),
            tile(s, b),
            tile(d, b),
        );
        let yb: &mut [f64; TILE] = (&mut y[b * TILE..(b + 1) * TILE]).try_into().unwrap();
        let mut acc = 0.0;
        for t in (0..TILE).rev() {
            yb[t] = pb[t] * acc;
            acc = fmadd(xb[t], qb[t], acc);
        }
        suffix[b] = acc;
        let mut acc = 0.0;
        for t in 0..TILE {
            yb[t] = fmadd(rb[t], acc, fmadd(db[t], xb[t], yb[t]));
            acc = fmadd(xb[t], sb[t], acc);
        }
        prefix[b] = acc;
    }
    // Exclusive scans of the block totals: what the later (earlier) blocks
    // contribute to every entry of a block.
    let mut carry = 0.0;
    for v in suffix[..nb].iter_mut().rev() {
        let t = *v;
        *v = carry;
        carry += t;
    }
    let mut carry = 0.0;
    for v in prefix[..nb].iter_mut() {
        let t = *v;
        *v = carry;
        carry += t;
    }
    for (b, yb) in y.as_chunks_mut::<TILE>().0.iter_mut().enumerate() {
        let (pb, rb) = (tile(p, b), tile(r, b));
        let (cs, cp) = (suffix[b], prefix[b]);
        for t in 0..TILE {
            yb[t] = fmadd(pb[t], cs, fmadd(rb[t], cp, yb[t]));
        }
    }
}

/// Model parameters laid out for the per-site recursions.
#[derive(Debug, Clone)]
pub(crate) struct Kernel {
    n: usize,
    np: usize,
    /// `a[i * np + j]`: transition probability from state `i` to state `j`.
    a: Vec<f64>,
    /// `at[j * np + i] = a[i * np + j]`.
    at: Vec<f64>,
    /// `em[o * np + k]`: the emission probability of observation index `o`
    /// in state `k` ([`ObsTable::emissions`]).
    em: Vec<f64>,
    pi: Vec<f64>,
    semi: Option<Semiseparable>,
}

/// The padded state dimension of `n` states: a multiple of [`TILE`].
pub(crate) fn padded(n: usize) -> usize {
    n.div_ceil(TILE).max(1) * TILE
}

impl Kernel {
    /// Copy `pi` and `a` into padded buffers, with the emission row of every
    /// observation index of `table` built from the rows of `em` (`T`, `K`,
    /// `N`). Shapes must have been validated by the caller.
    pub(crate) fn new(pi: &[f64], a: &Array2<f64>, em: &Array2<f64>, table: &ObsTable) -> Self {
        let n = pi.len();
        let np = padded(n);
        let mut a_pad = vec![0.0; np * np];
        let mut at_pad = vec![0.0; np * np];
        for i in 0..n {
            for j in 0..n {
                a_pad[i * np + j] = a[(i, j)];
                at_pad[j * np + i] = a[(i, j)];
            }
        }
        let mut pi_pad = vec![0.0; np];
        pi_pad[..n].copy_from_slice(pi);
        Self {
            n,
            np,
            a: a_pad,
            at: at_pad,
            em: table.emissions(em, n, np),
            pi: pi_pad,
            semi: Semiseparable::detect(a, np),
        }
    }

    /// The emission row of observation index `o`.
    #[inline(always)]
    fn em(&self, o: u16) -> &[f64] {
        let at = usize::from(o) * self.np;
        &self.em[at..at + self.np]
    }

    /// Whether the O(n) semiseparable kernel is in use.
    #[cfg(test)]
    fn is_semiseparable(&self) -> bool {
        self.semi.is_some()
    }

    /// Whether a forward step is cheap enough that caching full forward rows
    /// would not pay for its memory: true for the O(n) semiseparable kernel.
    pub(crate) fn cheap_forward(&self) -> bool {
        self.semi.is_some()
    }

    /// `out = Aᵀ x`.
    #[inline(always)]
    fn transpose_mul(&self, x: &[f64], out: &mut [f64]) {
        match &self.semi {
            Some(s) => semisep_mul(x, &s.g, &s.f, &s.j, &s.h, &s.d, out),
            None => vec_mat(x, &self.a, self.np, out),
        }
    }

    /// `out = A x`.
    #[inline(always)]
    fn mul(&self, x: &[f64], out: &mut [f64]) {
        match &self.semi {
            Some(s) => semisep_mul(x, &s.h, &s.j, &s.f, &s.g, &s.d, out),
            None => vec_mat(x, &self.at, self.np, out),
        }
    }

    /// Number of model states.
    pub(crate) fn n_states(&self) -> usize {
        self.n
    }

    /// Padded state dimension: the stride of every state vector.
    pub(crate) fn np(&self) -> usize {
        self.np
    }

    /// `out = pi ∘ e(o)`, normalised; returns the normaliser.
    #[inline]
    pub(crate) fn first(&self, o: u16, out: &mut [f64]) -> f64 {
        let em = self.em(o);
        for ((d, &p), &e) in out.iter_mut().zip(&self.pi).zip(em) {
            *d = p * e;
        }
        normalise(out)
    }

    /// `out = e(o) ∘ (Aᵀ prev)`, normalised; returns the normaliser.
    #[inline]
    pub(crate) fn forward(&self, prev: &[f64], o: u16, out: &mut [f64]) -> f64 {
        self.transpose_mul(prev, out);
        let em = self.em(o);
        for (d, &e) in out.iter_mut().zip(em) {
            *d *= e;
        }
        normalise(out)
    }

    /// One backward step from site `t` (observation `o`, scaled backward
    /// vector `beta`): `w = e(o) ∘ beta * scale` and `beta_prev = A w`.
    #[inline]
    pub(crate) fn backward(
        &self,
        beta: &[f64],
        o: u16,
        scale: f64,
        w: &mut [f64],
        beta_prev: &mut [f64],
    ) {
        let em = self.em(o);
        for ((d, &b), &e) in w.iter_mut().zip(beta).zip(em) {
            *d = e * b * scale;
        }
        self.mul(w, beta_prev);
    }

    /// Like [`Kernel::backward`], with `scale = 1 / sum(e(o) ∘ beta)`.
    /// Used where only the direction of beta matters (posterior decoding).
    #[inline]
    pub(crate) fn backward_normalised(
        &self,
        beta: &[f64],
        o: u16,
        w: &mut [f64],
        beta_prev: &mut [f64],
    ) {
        let em = self.em(o);
        for ((d, &b), &e) in w.iter_mut().zip(beta).zip(em) {
            *d = e * b;
        }
        normalise(w);
        self.mul(w, beta_prev);
    }

    /// `m[i][j] += sum_b u[b][i] * w[b][j]` over `rows` rows of `u` and `w`
    /// (each row `np` long). With `u` the forward vectors at sites `t - 1`
    /// and `w` the backward weights at sites `t`, `a[i][j] * m[i][j]` is the
    /// expected number of `i -> j` transitions (see [`Kernel::transition_counts`]).
    pub(crate) fn accumulate_transitions(&self, u: &[f64], w: &[f64], rows: usize, m: &mut [f64]) {
        // Entries of u beyond the real states are zero, so only the real rows
        // of m are updated (rounded up to whole row tiles, which np allows).
        outer_accumulate(u, w, rows, self.np, self.n, m);
    }

    /// Expected transition counts `xi = A ∘ m` (unpadded `n x n`).
    pub(crate) fn transition_counts(&self, m: &[f64]) -> Array2<f64> {
        let (n, np) = (self.n, self.np);
        Array2::from_shape_fn((n, n), |(i, j)| self.a[i * np + j] * m[i * np + j])
    }
}

#[inline(always)]
fn normalise(v: &mut [f64]) -> f64 {
    let s = sum(v);
    let inv = 1.0 / s;
    for d in v.iter_mut() {
        *d *= inv;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PsmcModel;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn psmc_kernel(rng: &mut SmallRng, n_steps: usize, pattern: Option<&str>) -> Kernel {
        // Large rho makes param_recalculate produce negative diagonal
        // entries (not a probability matrix); those fall back to dense.
        let theta = 10f64.powf(rng.gen_range(-5.0..-1.0));
        let rho = theta * 10f64.powf(rng.gen_range(-2.0..0.3));
        let mut m = PsmcModel::new(
            rng.gen_range(5.0..30.0),
            n_steps,
            theta,
            rho,
            2.5e-8,
            pattern.map(String::from),
        )
        .unwrap();
        for l in m.lam.iter_mut() {
            *l = 10f64.powf(rng.gen_range(-1.0..1.0));
        }
        m.param_recalculate().unwrap();
        Kernel::new(
            m.prior_matrix(),
            m.transition_matrix(),
            m.emission_matrix(),
            &ObsTable::plain(),
        )
    }

    fn max_rel(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| if x == y { 0.0 } else { ((x - y) / y).abs() })
            .fold(0.0, f64::max)
    }

    #[test]
    fn psmc_matrices_use_the_semiseparable_kernel() {
        let mut rng = SmallRng::seed_from_u64(7);
        for (n_steps, pattern) in [
            (1, None),
            (22, Some("4+5*3+4")),
            (63, Some("4+25*2+4+6")),
            (99, None),
        ] {
            for _ in 0..20 {
                let k = psmc_kernel(&mut rng, n_steps, pattern);
                assert!(k.is_semiseparable(), "n_steps {n_steps}");
                let np = k.np();
                let mut x = vec![0.0; np];
                for v in &mut x[..k.n_states()] {
                    *v = rng.gen_range(0.0..1.0);
                }
                let (mut fast, mut dense) = (vec![0.0; np], vec![0.0; np]);
                k.transpose_mul(&x, &mut fast);
                vec_mat(&x, &k.a, np, &mut dense);
                assert!(max_rel(&fast, &dense) < 1e-13, "Aᵀx, n_steps {n_steps}");
                k.mul(&x, &mut fast);
                vec_mat(&x, &k.at, np, &mut dense);
                assert!(max_rel(&fast, &dense) < 1e-13, "Ax, n_steps {n_steps}");
                assert!(fast[k.n_states()..].iter().all(|v| *v == 0.0));
            }
        }
    }

    /// The portable kernels (used on targets other than aarch64) agree with
    /// the dispatched ones.
    #[test]
    fn portable_kernels_match() {
        let mut rng = SmallRng::seed_from_u64(9);
        for np in [8, 16, 24, 64] {
            let n = np - 3;
            let x: Vec<f64> = (0..np)
                .map(|i| if i < n { rng.gen_range(0.0..1.0) } else { 0.0 })
                .collect();
            let m: Vec<f64> = (0..np * np).map(|_| rng.gen_range(0.0..1.0)).collect();
            let (mut fast, mut plain) = (vec![0.0; np], vec![0.0; np]);
            vec_mat(&x, &m, np, &mut fast);
            portable::vec_mat(&x, &m, np, &mut plain);
            assert!(max_rel(&fast, &plain) < 1e-13, "vec_mat np={np}");

            let rows = 37;
            let u: Vec<f64> = (0..rows * np)
                .map(|i| {
                    if i % np < n {
                        rng.gen_range(0.0..1.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            let w: Vec<f64> = (0..rows * np).map(|_| rng.gen_range(0.0..1.0)).collect();
            let (mut fast, mut plain) = (vec![1.0; np * np], vec![1.0; np * np]);
            outer_accumulate(&u, &w, rows, np, n, &mut fast);
            portable::outer_accumulate(&u, &w, rows, np, n, &mut plain);
            assert!(max_rel(&fast, &plain) < 1e-13, "outer_accumulate np={np}");
        }
    }

    #[test]
    fn other_matrices_use_the_dense_kernel() {
        let mut rng = SmallRng::seed_from_u64(8);
        let n = 12;
        let pi = vec![1.0 / n as f64; n];
        let em = Array2::from_elem((3, n), 0.5);
        let mut a = Array2::from_shape_fn((n, n), |_| rng.gen_range(0.01..1.0));
        for mut row in a.rows_mut() {
            let s: f64 = row.sum();
            row.mapv_inplace(|v| v / s);
        }
        assert!(!Kernel::new(&pi, &a, &em, &ObsTable::plain()).is_semiseparable());

        // A PSMC matrix with one entry perturbed beyond rounding.
        let mut k = psmc_kernel(&mut rng, 11, None);
        let mut a = Array2::from_shape_fn((12, 12), |(i, j)| k.a[i * k.np() + j]);
        a[(5, 2)] *= 1.0 + 1e-10;
        k = Kernel::new(&pi, &a, &em, &ObsTable::plain());
        assert!(!k.is_semiseparable());

        // An underflowed reference entry falls back too.
        a[(11, 0)] = 0.0;
        assert!(!Kernel::new(&pi, &a, &em, &ObsTable::plain()).is_semiseparable());
    }
}
