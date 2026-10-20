//! The forward pass of psmc-rs's straightforward f64 forward-backward
//! before the checkpointed engine (`src/hmm.rs` at commit ac53c53), on
//! which `tests/estep_oracle.rs` and `tests/estep_tmrca.rs` build their
//! reference E-step and TMRCA decoding.

/// The emission row of a bin: 0 (T), 1 (K), and 2 for any other symbol (N).
pub fn obs_to_idx(v: u8) -> usize {
    match v {
        0 | 1 => usize::from(v),
        _ => 2,
    }
}

/// The transition matrix `a` row after row, as the passes read it.
pub fn row_major(a: &ndarray::Array2<f64>) -> Vec<f64> {
    a.iter().copied().collect()
}

/// `dst = src A`, with `a_row` the row-major `A`.
pub fn matvec_row_major(src: &[f64], a_row: &[f64], dst: &mut [f64]) {
    let n_states = src.len();
    dst.fill(0.0);
    for (i, &s) in src.iter().enumerate() {
        let row = &a_row[i * n_states..(i + 1) * n_states];
        for (d, &a) in dst.iter_mut().zip(row) {
            *d += s * a;
        }
    }
}

/// The scaled forward variables (`alpha_row`, bin after bin) and their
/// normalisers (`c_norm`) of a row of `len` bins whose bin `t` emits with
/// `em_at(t)`, from `pi`, or from the last forward variables of the row
/// before (`start_prev`) when the row continues a sequence.
#[allow(clippy::too_many_arguments)]
pub fn forward_fill_row<'e>(
    pi: &[f64],
    a_row: &[f64],
    len: usize,
    em_at: impl Fn(usize) -> &'e [f64],
    start_prev: Option<&[f64]>,
    alpha_row: &mut [f64],
    c_norm: &mut [f64],
    tmp: &mut [f64],
) {
    let n_states = pi.len();
    for t in 0..len {
        let em_obs = em_at(t);
        let mut norm = 0.0f64;
        if t == 0 && start_prev.is_none() {
            for k in 0..n_states {
                let v = pi[k] * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        } else {
            let src = if t == 0 {
                start_prev.unwrap()
            } else {
                &alpha_row[(t - 1) * n_states..t * n_states]
            };
            matvec_row_major(src, a_row, tmp);
            for k in 0..n_states {
                let v = tmp[k] * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        }
        assert!(norm > 0.0, "normalization factor is zero in forward pass");
        c_norm[t] = norm;
        for k in 0..n_states {
            alpha_row[t * n_states + k] = tmp[k] / norm;
        }
    }
}
