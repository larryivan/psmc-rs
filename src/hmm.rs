// HMM forward-backward and Viterbi will live here.
// This module is intentionally minimal for the first iteration.

use anyhow::{bail, Result};
use ndarray::{Array2, Array3};

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
    }

    for b in 0..batch {
        for k in 0..n_states {
            beta[(b, s_max - 1, k)] = 1.0;
        }
    }

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
