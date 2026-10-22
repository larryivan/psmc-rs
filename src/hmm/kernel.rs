//! Per-site HMM arithmetic: one forward step, one backward step, and the
//! sums of expected transition counts the M-step needs.
//!
//! The state dimension is padded to a multiple of [`TILE`] with zero
//! probabilities, so every loop runs over whole tiles. Padded states never
//! receive mass: their emission, prior and transition entries are zero.
//!
//! PSMC transition matrices are semiseparable (see [`Semiseparable`]), which
//! makes the forward and backward steps O(n) instead of O(n²) for models of
//! up to 256 states, and the expected transition counts' sums O(n) for
//! models of any size ([`Kernel::transition_sums`]); other matrices use the
//! dense kernels.

use ndarray::Array2;

use super::TransitionSums;
use super::obs::ObsTable;

/// Width (in f64) of the column tiles the matrix-vector products are blocked into.
pub(crate) const TILE: usize = 8;

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

#[cfg(target_arch = "aarch64")]
mod neon {
    //! NEON kernels. Fused multiply-adds are explicit, and every product has
    //! enough independent accumulators (16) to cover the FMA latency.

    use super::{CHUNK, Semiseparable};
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

    /// [`super::semisep_sums`] over `groups` pairs of sites of `u` and `w`
    /// (`np` values each), the two sites of a pair in the lanes of a vector,
    /// four states at a time; adds to the [`N_SUMS`](super::N_SUMS) vectors
    /// of `out`.
    #[inline(always)]
    pub(super) unsafe fn semisep_sums(
        s: &Semiseparable,
        u: *const f64,
        w: *const f64,
        groups: usize,
        out: *mut f64,
    ) {
        const _: () = assert!(super::LANES == 2);
        unsafe {
            let np = s.d.len();
            let sums = |i: usize| out.add(i * np);
            let [lower_rows, lower_cols, upper_rows, upper_cols, diagonal] =
                [0, 1, 2, 3, 4].map(sums);
            // States k..k + 4 of the pair of sites `p`: one vector per state.
            let lanes = |x: *const f64, p: usize, k: usize| -> [float64x2_t; 4] {
                let (a, b) = (x.add(2 * p * np + k), x.add((2 * p + 1) * np + k));
                let (a01, a23) = (vld1q_f64(a), vld1q_f64(a.add(2)));
                let (b01, b23) = (vld1q_f64(b), vld1q_f64(b.add(2)));
                [
                    vzip1q_f64(a01, b01),
                    vzip2q_f64(a01, b01),
                    vzip1q_f64(a23, b23),
                    vzip2q_f64(a23, b23),
                ]
            };
            let splat = |v: &[f64], k: usize| -> [float64x2_t; 4] {
                std::array::from_fn(|q| vdupq_n_f64(v[k + q]))
            };
            // Adds the lanes of `acc[q]` to `sum[k + q]`.
            let add = |sum: *mut f64, k: usize, acc: &[float64x2_t; 4]| {
                for (q, pair) in [
                    (0, vpaddq_f64(acc[0], acc[1])),
                    (2, vpaddq_f64(acc[2], acc[3])),
                ] {
                    let p = sum.add(k + q);
                    vst1q_f64(p, vaddq_f64(vld1q_f64(p), pair));
                }
            };
            // The scans of each pair of sites: P and V, then U and S.
            let mut scans = [[vdupq_n_f64(0.0); 2]; CHUNK / 2];
            let scans = &mut scans[..groups];
            let zero = [vdupq_n_f64(0.0); 4];
            for k in (0..np).step_by(4) {
                let (g, h) = (splat(&s.g, k), splat(&s.h, k));
                let (mut lr, mut uc, mut d) = (zero, zero, zero);
                for (p, [sp, sv]) in scans.iter_mut().enumerate() {
                    let (uk, wk) = (lanes(u, p, k), lanes(w, p, k));
                    for q in 0..4 {
                        lr[q] = vfmaq_f64(lr[q], uk[q], *sp);
                        *sp = vfmaq_f64(*sp, wk[q], g[q]);
                        uc[q] = vfmaq_f64(uc[q], wk[q], *sv);
                        *sv = vfmaq_f64(*sv, uk[q], h[q]);
                        d[q] = vfmaq_f64(d[q], uk[q], wk[q]);
                    }
                }
                add(lower_rows, k, &lr);
                add(upper_cols, k, &uc);
                add(diagonal, k, &d);
            }
            scans.fill([vdupq_n_f64(0.0); 2]);
            for k in (0..np).step_by(4).rev() {
                let (j, f) = (splat(&s.j, k), splat(&s.f, k));
                let (mut ur, mut lc) = (zero, zero);
                for (p, [su, ss]) in scans.iter_mut().enumerate() {
                    let (uk, wk) = (lanes(u, p, k), lanes(w, p, k));
                    for q in (0..4).rev() {
                        ur[q] = vfmaq_f64(ur[q], uk[q], *su);
                        *su = vfmaq_f64(*su, wk[q], j[q]);
                        lc[q] = vfmaq_f64(lc[q], wk[q], *ss);
                        *ss = vfmaq_f64(*ss, uk[q], f[q]);
                    }
                }
                add(upper_rows, k, &ur);
                add(lower_cols, k, &lc);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! AVX2 and FMA kernels, for the passes built fused with these
    //! features ([`super::super::simd`]): they give the bits of the
    //! portable kernels instantiated fused.

    use super::{CHUNK, Semiseparable};
    use std::arch::x86_64::*;

    /// States `k` and `k + 1` of the four sites from `x`, `np` values
    /// apart: one vector per state, one lane per site.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn lanes(x: *const f64, np: usize) -> [__m256d; 2] {
        unsafe {
            let site = |i: usize| _mm_loadu_pd(x.add(i * np));
            let even = _mm256_insertf128_pd(_mm256_castpd128_pd256(site(0)), site(2), 1);
            let odd = _mm256_insertf128_pd(_mm256_castpd128_pd256(site(1)), site(3), 1);
            [_mm256_unpacklo_pd(even, odd), _mm256_unpackhi_pd(even, odd)]
        }
    }

    /// Adds the lanes of `acc[q]` to `sum[k + q]`: the lanes `(0, 2)` and
    /// `(1, 3)` first, then those two sums.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn add(sum: *mut f64, k: usize, acc: [__m256d; 2]) {
        let halves = |a| _mm_add_pd(_mm256_castpd256_pd128(a), _mm256_extractf128_pd::<1>(a));
        let pair = _mm_hadd_pd(halves(acc[0]), halves(acc[1]));
        unsafe {
            let p = sum.add(k);
            _mm_storeu_pd(p, _mm_add_pd(_mm_loadu_pd(p), pair));
        }
    }

    /// The portable [`semisep_sums`](super::portable::semisep_sums) of four
    /// lanes over `groups` groups of four sites of `u` and `w` (`np` values
    /// each), two states at a time; adds to the [`N_SUMS`](super::N_SUMS)
    /// vectors of `out`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn semisep_sums(
        s: &Semiseparable,
        u: *const f64,
        w: *const f64,
        groups: usize,
        out: *mut f64,
    ) {
        const _: () = assert!(super::LANES == 4);
        let np = s.d.len();
        let splat = |v: &[f64], k: usize| [_mm256_set1_pd(v[k]), _mm256_set1_pd(v[k + 1])];
        // The scans of each group of sites: P and V, then U and S.
        let zero = [_mm256_setzero_pd(); 2];
        let mut scans = [zero; CHUNK / 4];
        let scans = &mut scans[..groups];
        unsafe {
            let [lower_rows, lower_cols, upper_rows, upper_cols, diagonal] =
                [0, 1, 2, 3, 4].map(|i| out.add(i * np));
            for k in (0..np).step_by(2) {
                let (g, h) = (splat(&s.g, k), splat(&s.h, k));
                let (mut lr, mut uc, mut d) = (zero, zero, zero);
                for (p, [sp, sv]) in scans.iter_mut().enumerate() {
                    let at = 4 * p * np + k;
                    let (uk, wk) = (lanes(u.add(at), np), lanes(w.add(at), np));
                    for q in 0..2 {
                        lr[q] = _mm256_fmadd_pd(uk[q], *sp, lr[q]);
                        *sp = _mm256_fmadd_pd(wk[q], g[q], *sp);
                        uc[q] = _mm256_fmadd_pd(wk[q], *sv, uc[q]);
                        *sv = _mm256_fmadd_pd(uk[q], h[q], *sv);
                        d[q] = _mm256_fmadd_pd(uk[q], wk[q], d[q]);
                    }
                }
                add(lower_rows, k, lr);
                add(upper_cols, k, uc);
                add(diagonal, k, d);
            }
            scans.fill(zero);
            for k in (0..np).step_by(2).rev() {
                let (j, f) = (splat(&s.j, k), splat(&s.f, k));
                let (mut ur, mut lc) = (zero, zero);
                for (p, [su, ss]) in scans.iter_mut().enumerate() {
                    let at = 4 * p * np + k;
                    let (uk, wk) = (lanes(u.add(at), np), lanes(w.add(at), np));
                    for q in (0..2).rev() {
                        ur[q] = _mm256_fmadd_pd(uk[q], *su, ur[q]);
                        *su = _mm256_fmadd_pd(wk[q], j[q], *su);
                        lc[q] = _mm256_fmadd_pd(wk[q], *ss, lc[q]);
                        *ss = _mm256_fmadd_pd(uk[q], f[q], *ss);
                    }
                }
                add(upper_rows, k, ur);
                add(lower_cols, k, lc);
            }
        }
    }
}

// Compiled everywhere so that the aarch64 build tests it against NEON.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
mod portable {
    //! Plain Rust kernels for other targets. Their products are not fused
    //! (Rust never contracts `a * b + c`; [`semisep_sums`] fuses where
    //! `FUSED` says so), so they give the same bits with or without FMA in
    //! the target features, at any vector width.

    use super::{CHUNK, Semiseparable, TILE, fmadd, sums_mut};

    #[inline(always)]
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

    /// [`super::semisep_sums`] over the sites of `u` and `w` (`np` values
    /// each), `L` sites in the lanes of a vector, two states at a time;
    /// adds to the [`N_SUMS`](super::N_SUMS) vectors of `out`.
    #[inline(always)]
    pub(super) fn semisep_sums<const FUSED: bool, const L: usize>(
        s: &Semiseparable,
        u: &[f64],
        w: &[f64],
        out: &mut [f64],
    ) {
        let np = s.d.len();
        let [lower_rows, lower_cols, upper_rows, upper_cols, diagonal] = sums_mut(out, np);
        // States k, k + 1 of the `L` sites of `x`: one lane per site.
        let lanes = |x: &[f64], k: usize| -> [[f64; L]; 2] {
            let sites: [[f64; 2]; L] =
                std::array::from_fn(|i| x[i * np + k..i * np + k + 2].try_into().unwrap());
            std::array::from_fn(|q| std::array::from_fn(|i| sites[i][q]))
        };
        let fma = |a: [f64; L], b: [f64; L], c: [f64; L]| -> [f64; L] {
            std::array::from_fn(|i| fmadd::<FUSED>(a[i], b[i], c[i]))
        };
        // The lanes' sum, halves first: `(v[0] + v[2]) + (v[1] + v[3])`.
        let lane_sum = |mut v: [f64; L]| {
            let mut half = L;
            while half > 1 {
                half /= 2;
                for i in 0..half {
                    v[i] += v[i + half];
                }
            }
            v[0]
        };
        // The scans of each group of sites: P and V, then U and S.
        let mut scans = [[[0.0; L]; 2]; CHUNK];
        let scans = &mut scans[..u.len() / (L * np)];
        let sites = || u.chunks_exact(L * np).zip(w.chunks_exact(L * np));
        for k in (0..np).step_by(2) {
            let (g, h) = ([s.g[k]; L], [s.h[k]; L]);
            let (g1, h1) = ([s.g[k + 1]; L], [s.h[k + 1]; L]);
            let [mut lr, mut uc, mut d] = [[[0.0; L]; 2]; 3];
            for ((ut, wt), [p, v]) in sites().zip(scans.iter_mut()) {
                let (uk, wk) = (lanes(ut, k), lanes(wt, k));
                for (q, (gq, hq)) in [(g, h), (g1, h1)].into_iter().enumerate() {
                    lr[q] = fma(uk[q], *p, lr[q]);
                    *p = fma(wk[q], gq, *p);
                    uc[q] = fma(wk[q], *v, uc[q]);
                    *v = fma(uk[q], hq, *v);
                    d[q] = fma(uk[q], wk[q], d[q]);
                }
            }
            for q in 0..2 {
                lower_rows[k + q] += lane_sum(lr[q]);
                upper_cols[k + q] += lane_sum(uc[q]);
                diagonal[k + q] += lane_sum(d[q]);
            }
        }
        scans.fill([[0.0; L]; 2]);
        for k in (0..np).step_by(2).rev() {
            let (j, f) = ([s.j[k]; L], [s.f[k]; L]);
            let (j1, f1) = ([s.j[k + 1]; L], [s.f[k + 1]; L]);
            let [mut ur, mut lc] = [[[0.0; L]; 2]; 2];
            for ((ut, wt), [su, ss]) in sites().zip(scans.iter_mut()) {
                let (uk, wk) = (lanes(ut, k), lanes(wt, k));
                for (q, (jq, fq)) in [(j, f), (j1, f1)].into_iter().enumerate().rev() {
                    ur[q] = fma(uk[q], *su, ur[q]);
                    *su = fma(wk[q], jq, *su);
                    lc[q] = fma(wk[q], *ss, lc[q]);
                    *ss = fma(uk[q], fq, *ss);
                }
            }
            for q in 0..2 {
                upper_rows[k + q] += lane_sum(ur[q]);
                lower_cols[k + q] += lane_sum(lc[q]);
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

/// Blocks of [`TILE`] states covered by the stack scratch of the
/// semiseparable products (up to 256 states; larger models multiply by the
/// dense matrix).
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
        if a.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
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

/// `a * b + c`, rounded once when `FUSED` and twice otherwise. Each build
/// of the passes fixes `FUSED` at compile time ([`super::simd`]), never
/// fusing where `mul_add` would call libm's `fma` in this latency-bound scan.
#[inline(always)]
fn fmadd<const FUSED: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FUSED { a.mul_add(b, c) } else { a * b + c }
}

/// `y[i] = p[i] * sum_{l > i} x[l] q[l] + r[i] * sum_{l < i} x[l] s[l] + d[i] x[i]`.
///
/// Each block of [`TILE`] entries first gets the terms of its own entries
/// (two short scans, independent across blocks); the scanned block totals
/// are then added in one elementwise pass. All terms are non-negative, so
/// the order of summation costs no accuracy.
#[inline(always)]
fn semisep_mul<const FUSED: bool>(
    x: &[f64],
    p: &[f64],
    q: &[f64],
    r: &[f64],
    s: &[f64],
    d: &[f64],
    y: &mut [f64],
) {
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
            acc = fmadd::<FUSED>(xb[t], qb[t], acc);
        }
        suffix[b] = acc;
        let mut acc = 0.0;
        for t in 0..TILE {
            yb[t] = fmadd::<FUSED>(rb[t], acc, fmadd::<FUSED>(db[t], xb[t], yb[t]));
            acc = fmadd::<FUSED>(xb[t], sb[t], acc);
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
            yb[t] = fmadd::<FUSED>(pb[t], cs, fmadd::<FUSED>(rb[t], cp, yb[t]));
        }
    }
}

/// The sums of [`Kernel::transition_sums`], one vector of `np` values
/// each, in this order: the rows and the columns of the lower triangle,
/// those of the upper triangle, and the diagonal.
pub(crate) const N_SUMS: usize = 5;

/// The [`N_SUMS`] vectors of `out`, each `np` long.
#[inline(always)]
fn sums_mut(out: &mut [f64], np: usize) -> [&mut [f64]; N_SUMS] {
    let (lower_rows, rest) = out.split_at_mut(np);
    let (lower_cols, rest) = rest.split_at_mut(np);
    let (upper_rows, rest) = rest.split_at_mut(np);
    let (upper_cols, diagonal) = rest.split_at_mut(np);
    [
        lower_rows,
        lower_cols,
        upper_rows,
        upper_cols,
        &mut diagonal[..np],
    ]
}

/// Sites in the lanes of a vector of [`semisep_sums`]'s kernels.
const LANES: usize = if cfg!(target_arch = "aarch64") { 2 } else { 4 };

/// Sites that [`semisep_sums`]'s kernels take at once. Their scans go over
/// every state of each of them, reading the same columns of their rows
/// state after state, and with more sites the rows of a model of 64 states
/// or more (512 bytes apart) evict each other from the L1 cache: the
/// fastest on an Apple M1 Pro (128 KB of L1 data cache) and on an AMD EPYC
/// 7702 (32 KB) with 23 to 128 states.
const CHUNK: usize = if cfg!(target_arch = "aarch64") {
    32
} else {
    16
};
const _: () = assert!(CHUNK.is_multiple_of(LANES));

/// Adds to the sums `out` ([`N_SUMS`] vectors of `np` values) the
/// transitions into the `rows` sites of `w` from those of `u`, without the
/// factors of `s` that do not depend on the site. The expected number of
/// `k -> l` transitions at a site is `u[k] a[k][l] w[l]` (see
/// [`Kernel::transition_sums`]); below the diagonal `a[k][l] = f[k] g[l]`,
/// so its row `k` sums to `f[k] u[k] P[k]` and its column `l` to
/// `g[l] w[l] S[l]`, with the prefix and suffix sums
///
/// ```text
/// P[k] = sum_{l < k} g[l] w[l],   S[l] = sum_{k > l} f[k] u[k],
/// ```
///
/// and likewise above it, where `a[k][l] = h[k] j[l]`:
///
/// ```text
/// row k:     h[k] u[k] U[k],   U[k] = sum_{l > k} j[l] w[l],
/// column l:  j[l] w[l] V[l],   V[l] = sum_{k < l} h[k] u[k].
/// ```
///
/// So this adds `u P`, `w S`, `u U`, `w V` and, for the diagonal `d[k]`,
/// `u w`: four scans and five multiply-adds per state and site. The scans
/// run along the states, so the kernels put [`LANES`] sites in the lanes
/// of a vector: for every [`CHUNK`] sites, each scan goes state by state
/// (up for `P` and `V`, down for `U` and `S`), and each sum over the
/// chunk's sites is added up lane by lane, then the lanes in halves. The
/// last sites, fewer than [`LANES`], follow one at a time.
#[inline(always)]
fn semisep_sums<const FUSED: bool>(
    s: &Semiseparable,
    u: &[f64],
    w: &[f64],
    rows: usize,
    out: &mut [f64],
) {
    let np = s.d.len();
    assert!(out.len() == N_SUMS * np);
    let (u, w) = (&u[..rows * np], &w[..rows * np]);
    let full = rows / LANES * LANES;
    for c in (0..full).step_by(CHUNK) {
        let sites = c * np..full.min(c + CHUNK) * np;
        let (u, w) = (&u[sites.clone()], &w[sites]);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `u` and `w` hold the same whole number of pairs of sites,
        // at most `CHUNK`, and `out` the sums of `np` states.
        unsafe {
            neon::semisep_sums(
                s,
                u.as_ptr(),
                w.as_ptr(),
                u.len() / (2 * np),
                out.as_mut_ptr(),
            )
        }
        #[cfg(target_arch = "x86_64")]
        if FUSED && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: as for NEON, with groups of four sites; the CPU has
            // AVX2 and FMA.
            unsafe {
                avx2::semisep_sums(
                    s,
                    u.as_ptr(),
                    w.as_ptr(),
                    u.len() / (4 * np),
                    out.as_mut_ptr(),
                )
            }
            continue;
        }
        #[cfg(not(target_arch = "aarch64"))]
        portable::semisep_sums::<FUSED, LANES>(s, u, w, out);
    }
    portable::semisep_sums::<FUSED, 1>(s, &u[full * np..], &w[full * np..], out);
}

/// [`semisep_sums`] for any `n x n` matrix `a` (row-major, padded to `np`,
/// with its transpose `at`), factors included: each site's sums are its
/// rows' and columns' products with `w` and `u`, O(n²) per site.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dense_sums(
    a: &[f64],
    at: &[f64],
    n: usize,
    np: usize,
    u: &[f64],
    w: &[f64],
    rows: usize,
    out: &mut [f64],
) {
    let [lower_rows, lower_cols, upper_rows, upper_cols, diagonal] = sums_mut(out, np);
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).fold(0.0, |acc, (p, q)| acc + p * q);
    for (ut, wt) in u.chunks_exact(np).zip(w.chunks_exact(np)).take(rows) {
        for k in 0..n {
            let (row, col) = (&a[k * np..k * np + n], &at[k * np..k * np + n]);
            lower_rows[k] += ut[k] * dot(&row[..k], &wt[..k]);
            upper_rows[k] += ut[k] * dot(&row[k + 1..], &wt[k + 1..n]);
            lower_cols[k] += wt[k] * dot(&col[k + 1..], &ut[k + 1..n]);
            upper_cols[k] += wt[k] * dot(&col[..k], &ut[..k]);
            diagonal[k] += ut[k] * row[k] * wt[k];
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
    /// The factors of `a`'s triangles, when it is semiseparable.
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

    /// The factors with which the products with `a` take O(n): those of a
    /// semiseparable matrix of up to [`MAX_SEMISEP_BLOCKS`] blocks.
    #[inline(always)]
    fn semi_products(&self) -> Option<&Semiseparable> {
        self.semi
            .as_ref()
            .filter(|_| self.np / TILE <= MAX_SEMISEP_BLOCKS)
    }

    /// Whether a forward step is cheap enough that caching full forward rows
    /// would not pay for its memory: true for the O(n) semiseparable kernel.
    pub(crate) fn cheap_forward(&self) -> bool {
        self.semi_products().is_some()
    }

    /// `out = Aᵀ x`.
    #[inline(always)]
    fn transpose_mul<const FUSED: bool>(&self, x: &[f64], out: &mut [f64]) {
        match self.semi_products() {
            Some(s) => semisep_mul::<FUSED>(x, &s.g, &s.f, &s.j, &s.h, &s.d, out),
            None => vec_mat(x, &self.a, self.np, out),
        }
    }

    /// `out = A x`.
    #[inline(always)]
    fn mul<const FUSED: bool>(&self, x: &[f64], out: &mut [f64]) {
        match self.semi_products() {
            Some(s) => semisep_mul::<FUSED>(x, &s.h, &s.j, &s.f, &s.g, &s.d, out),
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
    #[inline(always)]
    pub(crate) fn first(&self, o: u16, out: &mut [f64]) -> f64 {
        let em = self.em(o);
        for ((d, &p), &e) in out.iter_mut().zip(&self.pi).zip(em) {
            *d = p * e;
        }
        normalise(out)
    }

    /// `out = e(o) ∘ (Aᵀ prev)`, normalised; returns the normaliser.
    #[inline(always)]
    pub(crate) fn forward<const FUSED: bool>(&self, prev: &[f64], o: u16, out: &mut [f64]) -> f64 {
        self.transpose_mul::<FUSED>(prev, out);
        let em = self.em(o);
        for (d, &e) in out.iter_mut().zip(em) {
            *d *= e;
        }
        normalise(out)
    }

    /// One backward step from site `t` (observation `o`, scaled backward
    /// vector `beta`): `w = e(o) ∘ beta * scale` and `beta_prev = A w`.
    #[inline(always)]
    pub(crate) fn backward<const FUSED: bool>(
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
        self.mul::<FUSED>(w, beta_prev);
    }

    /// Like [`Kernel::backward`], with `scale = 1 / sum(e(o) ∘ beta)`.
    /// Used where only the direction of beta matters (posterior decoding).
    #[inline(always)]
    pub(crate) fn backward_normalised<const FUSED: bool>(
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
        self.mul::<FUSED>(w, beta_prev);
    }

    /// The sums of the expected transitions into `rows` sites into `out`
    /// ([`N_SUMS`] vectors of `np` values, overwritten), from the forward
    /// vectors `u` at sites `t - 1` and the weights `w` of
    /// [`Kernel::backward`] at sites `t` (`rows` rows of `np` values each):
    /// `u[k] a[k][l] w[l]` is the expected number of `k -> l` transitions at
    /// site `t`. For a semiseparable matrix, in O(n) per site and without
    /// the factors of its triangles ([`semisep_sums`]); otherwise in O(n²).
    /// [`Kernel::transitions`] completes the sums of a whole E-step.
    #[inline(always)]
    pub(crate) fn transition_sums<const FUSED: bool>(
        &self,
        u: &[f64],
        w: &[f64],
        rows: usize,
        out: &mut [f64],
    ) {
        let np = self.np;
        assert!(u.len() >= rows * np && w.len() >= rows * np && out.len() == N_SUMS * np);
        out.fill(0.0);
        match &self.semi {
            Some(s) => semisep_sums::<FUSED>(s, u, w, rows, out),
            None => dense_sums(&self.a, &self.at, self.n, np, u, w, rows, out),
        }
    }

    /// The expected transition counts' sums of an E-step, from the sums of
    /// [`Kernel::transition_sums`] added over its segments: times the
    /// triangles' factors of each state for a semiseparable matrix.
    pub(crate) fn transitions(&self, total: &[f64]) -> TransitionSums {
        let n = self.n;
        let semi = self.semi.as_ref();
        let sum = |i: usize, factor: Option<&Vec<f64>>| -> Vec<f64> {
            let v = &total[i * self.np..i * self.np + n];
            match factor {
                Some(f) => v.iter().zip(f).map(|(x, f)| f * x).collect(),
                None => v.to_vec(),
            }
        };
        TransitionSums {
            lower_rows: sum(0, semi.map(|s| &s.f)),
            lower_cols: sum(1, semi.map(|s| &s.g)),
            upper_rows: sum(2, semi.map(|s| &s.h)),
            upper_cols: sum(3, semi.map(|s| &s.j)),
            diagonal: sum(4, semi.map(|s| &s.d)),
        }
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
                assert!(k.cheap_forward(), "n_steps {n_steps}");
                let np = k.np();
                let mut x = vec![0.0; np];
                for v in &mut x[..k.n_states()] {
                    *v = rng.gen_range(0.0..1.0);
                }
                check_semiseparable::<true>(&k, &x, n_steps);
                check_semiseparable::<false>(&k, &x, n_steps);
            }
        }
    }

    /// The semiseparable products of `k` with `x` against the dense ones.
    fn check_semiseparable<const FUSED: bool>(k: &Kernel, x: &[f64], n_steps: usize) {
        let np = k.np();
        let (mut fast, mut dense) = (vec![0.0; np], vec![0.0; np]);
        k.transpose_mul::<FUSED>(x, &mut fast);
        vec_mat(x, &k.a, np, &mut dense);
        assert!(max_rel(&fast, &dense) < 1e-13, "Aᵀx, n_steps {n_steps}");
        k.mul::<FUSED>(x, &mut fast);
        vec_mat(x, &k.at, np, &mut dense);
        assert!(max_rel(&fast, &dense) < 1e-13, "Ax, n_steps {n_steps}");
        assert!(fast[k.n_states()..].iter().all(|v| *v == 0.0));
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
        }
    }

    /// `rows` random forward vectors of `k` (zero in the padding) and as
    /// many random backward weights, and the transition sums of `k` over
    /// them as the E-step adds them up ([`Kernel::transitions`]).
    fn sums_of<const FUSED: bool>(
        k: &Kernel,
        rng: &mut SmallRng,
        rows: usize,
    ) -> (Vec<f64>, Vec<f64>, TransitionSums) {
        let (n, np) = (k.n_states(), k.np());
        let mut draw = || -> Vec<f64> {
            (0..rows * np)
                .map(|i| {
                    if i % np < n {
                        rng.gen_range(0.0..1.0)
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let (u, w) = (draw(), draw());
        let mut sums = vec![0.0; N_SUMS * np];
        k.transition_sums::<FUSED>(&u, &w, rows, &mut sums);
        (u, w, k.transitions(&sums))
    }

    /// The sums over each triangle's rows and columns and the diagonal of
    /// the expected transitions `u[k] a[k][l] w[l]` of every site, from the
    /// whole matrix of them.
    fn full_matrix_sums(k: &Kernel, u: &[f64], w: &[f64]) -> TransitionSums {
        let (n, np) = (k.n_states(), k.np());
        let mut xi = Array2::zeros((n, n));
        for (ut, wt) in u.chunks(np).zip(w.chunks(np)) {
            for ((i, j), x) in xi.indexed_iter_mut() {
                *x += ut[i] * k.a[i * np + j] * wt[j];
            }
        }
        TransitionSums::of(&xi)
    }

    /// The largest relative difference between two transition sums.
    fn sums_rel(a: &TransitionSums, b: &TransitionSums) -> f64 {
        let parts = |t: &TransitionSums| {
            [
                t.lower_rows.clone(),
                t.lower_cols.clone(),
                t.upper_rows.clone(),
                t.upper_cols.clone(),
                t.diagonal.clone(),
            ]
        };
        let (pa, pb) = (parts(a), parts(b));
        pa.iter()
            .zip(&pb)
            .map(|(x, y)| max_rel(x, y))
            .fold(0.0, f64::max)
    }

    /// The transition sums of semiseparable matrices (with the O(n)
    /// products up to 256 states, and the dense ones beyond) and of other
    /// matrices are those of the whole matrix of expected counts, on any
    /// number of sites, fused and not. Measured: 2.8e-15 at most.
    #[test]
    fn transition_sums_are_those_of_the_whole_matrix() {
        fn check<const FUSED: bool>(k: &Kernel, rng: &mut SmallRng, what: &str) -> f64 {
            let mut worst = 0.0f64;
            for rows in [1, 2, 3, 5, 31, 32, 33, 127, 128] {
                let (u, w, got) = sums_of::<FUSED>(k, rng, rows);
                let rel = sums_rel(&got, &full_matrix_sums(k, &u, &w));
                assert!(rel < 1e-13, "{what}, {rows} sites, fused {FUSED}: {rel:e}");
                worst = worst.max(rel);
            }
            worst
        }
        let mut rng = SmallRng::seed_from_u64(10);
        let mut worst = 0.0f64;
        for (n_steps, pattern) in [
            (1, None),
            (22, Some("4+5*3+4")),
            (63, Some("4+25*2+4+6")),
            (99, None),
            (300, Some("1+30*10")),
        ] {
            let k = psmc_kernel(&mut rng, n_steps, pattern);
            assert!(k.semi.is_some() && k.cheap_forward() == (n_steps < 256));
            let what = format!("PSMC, n_steps {n_steps}");
            worst = worst.max(check::<true>(&k, &mut rng, &what));
            worst = worst.max(check::<false>(&k, &mut rng, &what));
        }
        let n = 12;
        let mut a = Array2::from_shape_fn((n, n), |_| rng.gen_range(0.01..1.0));
        for mut row in a.rows_mut() {
            let s: f64 = row.sum();
            row.mapv_inplace(|v| v / s);
        }
        let k = Kernel::new(
            &[1.0 / n as f64; 12],
            &a,
            &Array2::from_elem((3, n), 0.5),
            &ObsTable::plain(),
        );
        assert!(k.semi.is_none());
        worst = worst.max(check::<true>(&k, &mut rng, "dense"));
        eprintln!("worst relative difference {worst:e}");
    }

    /// The SIMD kernels of the transition sums give the bits of the portable
    /// code instantiated fused, with as many sites in the lanes (NEON on
    /// aarch64, AVX2 and FMA on x86-64 CPUs that have them).
    #[test]
    fn transition_sum_kernels_give_the_portable_bits() {
        let mut rng = SmallRng::seed_from_u64(11);
        for (n_steps, pattern) in [(22, Some("4+5*3+4")), (63, Some("4+25*2+4+6")), (299, None)] {
            let k = psmc_kernel(&mut rng, n_steps, pattern);
            let (s, np) = (k.semi.as_ref().unwrap(), k.np());
            for rows in [1, 3, 4, 5, 32, 33, 37, 127, 128] {
                let (u, w, _) = sums_of::<true>(&k, &mut rng, rows);
                let mut fast = vec![0.0; N_SUMS * np];
                semisep_sums::<true>(s, &u, &w, rows, &mut fast);
                let mut plain = vec![0.0; N_SUMS * np];
                let full = rows / LANES * LANES;
                for c in (0..full).step_by(CHUNK) {
                    let sites = c * np..full.min(c + CHUNK) * np;
                    let (u, w) = (&u[sites.clone()], &w[sites]);
                    portable::semisep_sums::<true, LANES>(s, u, w, &mut plain);
                }
                portable::semisep_sums::<true, 1>(s, &u[full * np..], &w[full * np..], &mut plain);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&fast), bits(&plain), "n_steps {n_steps}, {rows} sites");
            }
        }
    }

    /// `fmadd::<true>` rounds once: `(1 + e)(1 - e) - 1 = -e²` exactly;
    /// `fmadd::<false>` rounds the product `1 - e²` to 1 and gives 0. The
    /// passes compiled for the target as given use the unfused instance
    /// except on aarch64: on x86-64, fusing there would call libm's `fma`.
    #[test]
    fn fmadd_fuses_when_asked() {
        let e = 2f64.powi(-30);
        let (a, b) = (std::hint::black_box(1.0 + e), std::hint::black_box(1.0 - e));
        assert_eq!(fmadd::<true>(a, b, -1.0), -(e * e));
        assert_eq!(fmadd::<false>(a, b, -1.0), 0.0);
        let default = fmadd::<{ crate::hmm::simd::DEFAULT_FUSED }>(a, b, -1.0);
        let want = if cfg!(target_arch = "aarch64") {
            -(e * e)
        } else {
            0.0
        };
        assert_eq!(default, want);
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
        assert!(Kernel::new(&pi, &a, &em, &ObsTable::plain()).semi.is_none());

        // A PSMC matrix with one entry perturbed beyond rounding.
        let mut k = psmc_kernel(&mut rng, 11, None);
        let mut a = Array2::from_shape_fn((12, 12), |(i, j)| k.a[i * k.np() + j]);
        a[(5, 2)] *= 1.0 + 1e-10;
        k = Kernel::new(&pi, &a, &em, &ObsTable::plain());
        assert!(k.semi.is_none());

        // An underflowed reference entry falls back too.
        a[(11, 0)] = 0.0;
        assert!(Kernel::new(&pi, &a, &em, &ObsTable::plain()).semi.is_none());
    }
}
