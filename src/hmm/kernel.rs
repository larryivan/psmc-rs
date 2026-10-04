//! Per-site HMM arithmetic: one forward step, one backward step, and the
//! blocked accumulation of expected transition counts.
//!
//! The state dimension is padded to a multiple of [`TILE`] with zero
//! probabilities, so every loop runs over whole tiles. Padded states never
//! receive mass: their emission, prior and transition entries are zero.

use ndarray::Array2;

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
    portable::vec_mat(x, m, np, y)
}

/// `m[i][j] += sum_b u[b][i] * w[b][j]` for `i < n_rows_used` (rounded up to
/// [`ROW_TILE`]) and all `j`; `u`, `w` hold `rows` rows of `np` values.
#[inline(always)]
fn outer_accumulate(u: &[f64], w: &[f64], rows: usize, np: usize, n: usize, m: &mut [f64]) {
    assert!(np.is_multiple_of(TILE) && n <= np);
    assert!(u.len() >= rows * np && w.len() >= rows * np && m.len() == np * np);
    portable::outer_accumulate(u, w, rows, np, n, m)
}

mod portable {
    //! Plain Rust kernels (no FMA: `mul_add` would be a libm call on x86-64
    //! builds without the `fma` target feature).

    use super::{ROW_TILE, TILE};

    pub(super) fn vec_mat(x: &[f64], m: &[f64], np: usize, y: &mut [f64]) {
        for (jt, y_tile) in y.chunks_exact_mut(TILE).enumerate() {
            let col = jt * TILE;
            let mut acc = [0.0f64; TILE];
            for (row, &s) in m.chunks_exact(np).zip(x) {
                let r: &[f64; TILE] = row[col..col + TILE].try_into().unwrap();
                for c in 0..TILE {
                    acc[c] += s * r[c];
                }
            }
            y_tile.copy_from_slice(&acc);
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
    for chunk in v.chunks_exact(TILE) {
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

/// Model parameters laid out for the per-site recursions.
#[derive(Debug, Clone)]
pub(crate) struct Kernel {
    n: usize,
    np: usize,
    /// `a[i * np + j]`: transition probability from state `i` to state `j`.
    a: Vec<f64>,
    /// `at[j * np + i] = a[i * np + j]`.
    at: Vec<f64>,
    em: [Vec<f64>; 3],
    pi: Vec<f64>,
}

impl Kernel {
    /// Copy `pi`, `a` and the first three rows of `em` into padded buffers.
    /// Shapes must have been validated by the caller.
    pub(crate) fn new(pi: &[f64], a: &Array2<f64>, em: &Array2<f64>) -> Self {
        let n = pi.len();
        let np = n.div_ceil(TILE).max(1) * TILE;
        let mut a_pad = vec![0.0; np * np];
        let mut at_pad = vec![0.0; np * np];
        for i in 0..n {
            for j in 0..n {
                a_pad[i * np + j] = a[(i, j)];
                at_pad[j * np + i] = a[(i, j)];
            }
        }
        let em_row = |r: usize| {
            let mut v = vec![0.0; np];
            for k in 0..n {
                v[k] = em[(r, k)];
            }
            v
        };
        let mut pi_pad = vec![0.0; np];
        pi_pad[..n].copy_from_slice(pi);
        Self {
            n,
            np,
            a: a_pad,
            at: at_pad,
            em: [em_row(0), em_row(1), em_row(2)],
            pi: pi_pad,
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
    pub(crate) fn first(&self, o: u8, out: &mut [f64]) -> f64 {
        let em = &self.em[o as usize];
        for ((d, &p), &e) in out.iter_mut().zip(&self.pi).zip(em) {
            *d = p * e;
        }
        normalise(out)
    }

    /// `out = e(o) ∘ (Aᵀ prev)`, normalised; returns the normaliser.
    #[inline]
    pub(crate) fn forward(&self, prev: &[f64], o: u8, out: &mut [f64]) -> f64 {
        vec_mat(prev, &self.a, self.np, out);
        let em = &self.em[o as usize];
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
        o: u8,
        scale: f64,
        w: &mut [f64],
        beta_prev: &mut [f64],
    ) {
        let em = &self.em[o as usize];
        for ((d, &b), &e) in w.iter_mut().zip(beta).zip(em) {
            *d = e * b * scale;
        }
        vec_mat(w, &self.at, self.np, beta_prev);
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
