// HMM forward-backward and Viterbi will live here.
// This module is intentionally minimal for the first iteration.

use anyhow::{bail, Result};
use ndarray::{Array2, Array3};

use crate::progress;

#[derive(Debug)]
pub struct ForwardBackwardResult {
    pub gamma: Array3<f64>,
    pub xi: Array2<f64>,
    pub c_norm: Vec<f64>,
    pub loglike: f64,
}

pub fn forward_backward(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    progress_enabled: bool,
    label: &str,
) -> Result<ForwardBackwardResult> {
    let (batch, s_max) = x.dim();
    let n_states = pi.len();
    if a.nrows() != n_states || a.ncols() != n_states {
        bail!("transition matrix shape mismatch");
    }
    if em.nrows() < 3 || em.ncols() != n_states {
        bail!("emission matrix shape mismatch");
    }

    let mut alpha = Array3::<f64>::zeros((batch, s_max, n_states));
    let mut beta = Array3::<f64>::zeros((batch, s_max, n_states));
    let mut c_norm = vec![0.0f64; batch * s_max];

    for b in 0..batch {
        let obs0 = x[(b, 0)] as usize;
        let mut sum = 0.0;
        for k in 0..n_states {
            let v = em[(obs0, k)] * pi[k];
            alpha[(b, 0, k)] = v;
            sum += v;
        }
        if sum <= 0.0 {
            bail!("normalization factor is zero at t=0");
        }
        c_norm[b * s_max] = sum;
        for k in 0..n_states {
            alpha[(b, 0, k)] /= sum;
        }
    }

    let span = s_max.saturating_sub(1) as u64;
    let pb_alpha = if progress_enabled && span > 0 {
        Some(progress::bar(span, label, "alpha"))
    } else {
        None
    };
    let stride = (span as usize / 200).max(1);
    let mut pending = 0usize;

    for t in 1..s_max {
        for b in 0..batch {
            let obs = x[(b, t)] as usize;
            let mut norm = 0.0;
            for k in 0..n_states {
                let mut acc = 0.0;
                for j in 0..n_states {
                    acc += alpha[(b, t - 1, j)] * a[(j, k)];
                }
                let v = em[(obs, k)] * acc;
                alpha[(b, t, k)] = v;
                norm += v;
            }
            if norm <= 0.0 {
                bail!("normalization factor is zero at t={}", t);
            }
            c_norm[b * s_max + t] = norm;
            for k in 0..n_states {
                alpha[(b, t, k)] /= norm;
            }
        }
        if let Some(pb) = &pb_alpha {
            pending += 1;
            if pending == stride {
                pb.inc(pending as u64);
                pending = 0;
            }
        }
    }
    if let Some(pb) = pb_alpha {
        if pending > 0 {
            pb.inc(pending as u64);
        }
        pb.finish_with_message(format!("{label} alpha done"));
    }

    for b in 0..batch {
        for k in 0..n_states {
            beta[(b, s_max - 1, k)] = 1.0;
        }
    }

    let pb_beta = if progress_enabled && span > 0 {
        Some(progress::bar(span, label, "beta"))
    } else {
        None
    };
    let mut pending = 0usize;
    for t in (0..(s_max - 1)).rev() {
        for b in 0..batch {
            let obs_next = x[(b, t + 1)] as usize;
            let norm = c_norm[b * s_max + t + 1];
            for k in 0..n_states {
                let mut acc = 0.0;
                for j in 0..n_states {
                    acc += beta[(b, t + 1, j)] * em[(obs_next, j)] * a[(k, j)];
                }
                beta[(b, t, k)] = acc / norm;
            }
        }
        if let Some(pb) = &pb_beta {
            pending += 1;
            if pending == stride {
                pb.inc(pending as u64);
                pending = 0;
            }
        }
    }
    if let Some(pb) = pb_beta {
        if pending > 0 {
            pb.inc(pending as u64);
        }
        pb.finish_with_message(format!("{label} beta done"));
    }

    let mut gamma = Array3::<f64>::zeros((batch, s_max, n_states));
    for b in 0..batch {
        for t in 0..s_max {
            for k in 0..n_states {
                gamma[(b, t, k)] = alpha[(b, t, k)] * beta[(b, t, k)];
            }
        }
    }

    let mut xi = Array2::<f64>::zeros((n_states, n_states));
    let pb_xi = if progress_enabled && span > 0 {
        Some(progress::bar(span, label, "xi/gamma"))
    } else {
        None
    };
    let mut pending = 0usize;
    for t in 1..s_max {
        for b in 0..batch {
            let obs = x[(b, t)] as usize;
            let denom = c_norm[b * s_max + t];
            for i in 0..n_states {
                let alpha_prev = alpha[(b, t - 1, i)];
                for j in 0..n_states {
                    xi[(i, j)] += alpha_prev * a[(i, j)] * em[(obs, j)] * beta[(b, t, j)] / denom;
                }
            }
        }
        if let Some(pb) = &pb_xi {
            pending += 1;
            if pending == stride {
                pb.inc(pending as u64);
                pending = 0;
            }
        }
    }
    if let Some(pb) = pb_xi {
        if pending > 0 {
            pb.inc(pending as u64);
        }
        pb.finish_with_message(format!("{label} xi/gamma done"));
    }

    let mut loglike = 0.0;
    for v in c_norm.iter() {
        loglike += v.ln();
    }

    Ok(ForwardBackwardResult {
        gamma,
        xi,
        c_norm,
        loglike,
    })
}
