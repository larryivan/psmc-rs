use anyhow::{bail, Result};
use faer::linalg::matmul::matmul;
use faer::mat::{from_row_major_slice, from_row_major_slice_mut};
use faer::{Mat, Parallelism};
use indicatif::{MultiProgress, ProgressDrawTarget};
use ndarray::Array2;

use crate::progress;

#[derive(Debug, Clone)]
pub struct SufficientStats {
    pub g0: Vec<f64>,
    pub gobs: [Vec<f64>; 3],
    pub xi: Array2<f64>,
}

#[derive(Debug)]
pub struct EStepResult {
    pub stats: SufficientStats,
    pub loglike: f64,
}

pub fn e_step_streaming(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let (batch, s_max) = x.dim();
    let n_states = pi.len();
    if a.nrows() != n_states || a.ncols() != n_states {
        bail!("transition matrix shape mismatch");
    }
    if em.nrows() < 3 || em.ncols() != n_states {
        bail!("emission matrix shape mismatch");
    }

    let mut g0 = vec![0.0; n_states];
    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut xi = Array2::<f64>::zeros((n_states, n_states));
    let mut loglike = 0.0;

    let a_faer = Mat::<f64>::from_fn(n_states, n_states, |i, j| a[(i, j)]);
    let par = Parallelism::None;

    let total_fwd = batch.saturating_mul(s_max.saturating_sub(1)) as u64;
    let total_bwd = batch.saturating_mul(s_max) as u64;

    let mp = if progress_enabled {
        Some(MultiProgress::with_draw_target(
            ProgressDrawTarget::stderr_with_hz(15),
        ))
    } else {
        None
    };
    let pb_fwd = if total_fwd > 0 {
        mp.as_ref()
            .map(|mp| mp.add(progress::bar_raw(total_fwd, label, "forward")))
    } else {
        None
    };
    let pb_bwd = if total_bwd > 0 {
        mp.as_ref()
            .map(|mp| mp.add(progress::bar_raw(total_bwd, label, "backward")))
    } else {
        None
    };
    let stride_fwd = (total_fwd / 200).max(1) as usize;
    let stride_bwd = (total_bwd / 200).max(1) as usize;
    let mut pending_fwd = 0usize;
    let mut pending_bwd = 0usize;

    for b in 0..batch {
        let mut alpha = vec![0.0f64; s_max * n_states];
        let mut c_norm = vec![0.0f64; s_max];
        let mut tmp = vec![0.0f64; n_states];

        let obs0 = x[(b, 0)] as usize;
        let mut norm = 0.0;
        for k in 0..n_states {
            let v = pi[k] * em[(obs0, k)];
            alpha[k] = v;
            norm += v;
        }
        if norm <= 0.0 {
            bail!("normalization factor is zero at t=0");
        }
        c_norm[0] = norm;
        for k in 0..n_states {
            alpha[k] /= norm;
        }

        for t in 1..s_max {
            let obs = x[(b, t)] as usize;
            let prev = &alpha[(t - 1) * n_states..t * n_states];
            let lhs = from_row_major_slice(prev, 1, n_states);
            let rhs = a_faer.as_ref();
            let mut out = from_row_major_slice_mut(&mut tmp, 1, n_states);
            matmul(out.as_mut(), lhs, rhs, None, 1.0, par);

            norm = 0.0;
            for k in 0..n_states {
                tmp[k] *= em[(obs, k)];
                norm += tmp[k];
            }
            if norm <= 0.0 {
                bail!("normalization factor is zero at t={}", t);
            }
            c_norm[t] = norm;
            let alpha_t = &mut alpha[t * n_states..(t + 1) * n_states];
            for k in 0..n_states {
                alpha_t[k] = tmp[k] / norm;
            }

            if let Some(pb) = &pb_fwd {
                pending_fwd += 1;
                if pending_fwd == stride_fwd {
                    pb.inc(pending_fwd as u64);
                    pending_fwd = 0;
                }
            }
        }

        for v in c_norm.iter() {
            loglike += v.ln();
        }

        let mut beta = vec![1.0f64; n_states];
        let mut beta_new = vec![0.0f64; n_states];

        for t in (0..s_max).rev() {
            let obs = x[(b, t)] as usize;
            let alpha_t = &alpha[t * n_states..(t + 1) * n_states];
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                if t == 0 {
                    g0[k] += g;
                }
                match obs {
                    0 => gobs0[k] += g,
                    1 => gobs1[k] += g,
                    _ => gobs2[k] += g,
                }
            }

            if t == 0 {
                break;
            }

            let norm_t = c_norm[t];
            let alpha_prev = &alpha[(t - 1) * n_states..t * n_states];
            let a_ref = a_faer.as_ref();
            for i in 0..n_states {
                let a_i = a_ref.row(i);
                for j in 0..n_states {
                    let a_ij = a_i.read(j);
                    let val = alpha_prev[i] * a_ij * em[(obs, j)] * beta[j] / norm_t;
                    xi[(i, j)] += val;
                }
            }

            for i in 0..n_states {
                let a_i = a_ref.row(i);
                let mut acc = 0.0;
                for j in 0..n_states {
                    acc += a_i.read(j) * em[(obs, j)] * beta[j];
                }
                beta_new[i] = acc / norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
            if let Some(pb) = &pb_bwd {
                pending_bwd += 1;
                if pending_bwd == stride_bwd {
                    pb.inc(pending_bwd as u64);
                    pending_bwd = 0;
                }
            }
        }
    }

    if let Some(pb) = pb_fwd {
        if pending_fwd > 0 {
            pb.inc(pending_fwd as u64);
        }
        pb.finish_with_message(format!("{label} forward done"));
    }

    if let Some(pb) = pb_bwd {
        if pending_bwd > 0 {
            pb.inc(pending_bwd as u64);
        }
        pb.finish_with_message(format!("{label} backward done"));
    }

    Ok(EStepResult {
        stats: SufficientStats {
            g0,
            gobs: [gobs0, gobs1, gobs2],
            xi,
        },
        loglike,
    })
}
