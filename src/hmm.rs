use anyhow::{Result, anyhow, bail};
use indicatif::{MultiProgress, ProgressDrawTarget};
use ndarray::Array2;
use rayon::prelude::*;

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

#[derive(Debug, Clone)]
struct BatchAccum {
    g0: Vec<f64>,
    gobs: [Vec<f64>; 3],
    xi: Vec<f64>,
    loglike: f64,
}

impl BatchAccum {
    fn zeros(n_states: usize) -> Self {
        Self {
            g0: vec![0.0; n_states],
            gobs: [
                vec![0.0; n_states],
                vec![0.0; n_states],
                vec![0.0; n_states],
            ],
            xi: vec![0.0; n_states * n_states],
            loglike: 0.0,
        }
    }

    fn add_assign(&mut self, rhs: &Self) {
        for k in 0..self.g0.len() {
            self.g0[k] += rhs.g0[k];
            self.gobs[0][k] += rhs.gobs[0][k];
            self.gobs[1][k] += rhs.gobs[1][k];
            self.gobs[2][k] += rhs.gobs[2][k];
        }
        for i in 0..self.xi.len() {
            self.xi[i] += rhs.xi[i];
        }
        self.loglike += rhs.loglike;
    }

    fn into_estep(self, n_states: usize) -> Result<EStepResult> {
        let xi = Array2::from_shape_vec((n_states, n_states), self.xi)
            .map_err(|e| anyhow!("failed to build xi matrix from flat buffer: {e}"))?;
        Ok(EStepResult {
            stats: SufficientStats {
                g0: self.g0,
                gobs: self.gobs,
                xi,
            },
            loglike: self.loglike,
        })
    }
}

#[inline]
fn obs_to_idx(v: u8) -> usize {
    if v == 0 {
        0
    } else if v == 1 {
        1
    } else {
        2
    }
}

fn forward_fill_row(
    b: usize,
    pi: &[f64],
    a_col: &[f64],
    em_rows: &[Vec<f64>; 3],
    x: &Array2<u8>,
    start_prev: Option<&[f64]>,
    alpha_row: &mut [f64],
    c_norm: &mut [f64],
    tmp: &mut [f64],
) -> Result<()> {
    let s_max = x.ncols();
    let n_states = pi.len();
    for t in 0..s_max {
        let obs = obs_to_idx(x[(b, t)]);
        let em_obs = &em_rows[obs];

        let mut norm = 0.0;
        if t == 0 {
            match start_prev {
                Some(prev) => {
                    for k in 0..n_states {
                        let a_col_k = &a_col[k * n_states..(k + 1) * n_states];
                        let mut dot = 0.0;
                        for i in 0..n_states {
                            dot += prev[i] * a_col_k[i];
                        }
                        let v = dot * em_obs[k];
                        tmp[k] = v;
                        norm += v;
                    }
                }
                None => {
                    for k in 0..n_states {
                        let v = pi[k] * em_obs[k];
                        tmp[k] = v;
                        norm += v;
                    }
                }
            }
        } else {
            let prev = &alpha_row[(t - 1) * n_states..t * n_states];
            for k in 0..n_states {
                let a_col_k = &a_col[k * n_states..(k + 1) * n_states];
                let mut dot = 0.0;
                for i in 0..n_states {
                    dot += prev[i] * a_col_k[i];
                }
                let v = dot * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        }

        if norm <= 0.0 {
            bail!("normalization factor is zero at batch={}, t={}", b, t);
        }
        c_norm[t] = norm;
        let alpha_t = &mut alpha_row[t * n_states..(t + 1) * n_states];
        for k in 0..n_states {
            alpha_t[k] = tmp[k] / norm;
        }
    }
    Ok(())
}

fn accumulate_one_batch(
    acc: &mut BatchAccum,
    b: usize,
    pi: &[f64],
    a_row: &[f64],
    a_col: &[f64],
    em_rows: &[Vec<f64>; 3],
    x: &Array2<u8>,
) -> Result<()> {
    let s_max = x.ncols();
    let n_states = pi.len();

    let mut alpha = vec![0.0f64; s_max * n_states];
    let mut c_norm = vec![0.0f64; s_max];
    let mut tmp = vec![0.0f64; n_states];

    let obs0 = obs_to_idx(x[(b, 0)]);
    let em_obs0 = &em_rows[obs0];
    let mut norm = 0.0;
    for k in 0..n_states {
        let v = pi[k] * em_obs0[k];
        alpha[k] = v;
        norm += v;
    }
    if norm <= 0.0 {
        bail!("normalization factor is zero at batch={}, t=0", b);
    }
    c_norm[0] = norm;
    for k in 0..n_states {
        alpha[k] /= norm;
    }

    for t in 1..s_max {
        let obs = obs_to_idx(x[(b, t)]);
        let em_obs = &em_rows[obs];
        let prev = &alpha[(t - 1) * n_states..t * n_states];

        norm = 0.0;
        for k in 0..n_states {
            let a_col_k = &a_col[k * n_states..(k + 1) * n_states];
            let mut dot = 0.0;
            for i in 0..n_states {
                dot += prev[i] * a_col_k[i];
            }
            let v = dot * em_obs[k];
            tmp[k] = v;
            norm += v;
        }
        if norm <= 0.0 {
            bail!("normalization factor is zero at batch={}, t={}", b, t);
        }
        c_norm[t] = norm;
        let alpha_t = &mut alpha[t * n_states..(t + 1) * n_states];
        for k in 0..n_states {
            alpha_t[k] = tmp[k] / norm;
        }
    }

    for v in &c_norm {
        acc.loglike += v.ln();
    }

    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];

    for t in (0..s_max).rev() {
        let obs = obs_to_idx(x[(b, t)]);
        let alpha_t = &alpha[t * n_states..(t + 1) * n_states];
        for k in 0..n_states {
            let g = alpha_t[k] * beta[k];
            if t == 0 {
                acc.g0[k] += g;
            }
            acc.gobs[obs][k] += g;
        }

        if t == 0 {
            break;
        }

        let norm_t = c_norm[t];
        let em_obs = &em_rows[obs];
        let alpha_prev = &alpha[(t - 1) * n_states..t * n_states];

        for i in 0..n_states {
            let a_i = &a_row[i * n_states..(i + 1) * n_states];
            let alpha_prev_i = alpha_prev[i];
            let mut acc_beta = 0.0;
            for j in 0..n_states {
                let emit_beta = em_obs[j] * beta[j];
                let trans = a_i[j];
                acc.xi[i * n_states + j] += alpha_prev_i * trans * emit_beta / norm_t;
                acc_beta += trans * emit_beta;
            }
            beta_new[i] = acc_beta / norm_t;
        }
        std::mem::swap(&mut beta, &mut beta_new);
    }

    Ok(())
}

fn e_step_streaming_contiguous_rows(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let (batch, s_max) = x.dim();
    let n_states = pi.len();
    if batch == 0 || s_max == 0 {
        bail!("empty input matrix for E-step");
    }

    let mut g0 = vec![0.0; n_states];
    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut xi = Array2::<f64>::zeros((n_states, n_states));
    let mut loglike = 0.0;

    let mut a_row = vec![0.0f64; n_states * n_states];
    let mut a_col = vec![0.0f64; n_states * n_states];
    for i in 0..n_states {
        for j in 0..n_states {
            let v = a[(i, j)];
            a_row[i * n_states + j] = v;
            a_col[j * n_states + i] = v;
        }
    }
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let mut alpha_row = vec![0.0f64; s_max * n_states];
    let mut c_norm = vec![0.0f64; s_max];
    let mut tmp = vec![0.0f64; n_states];

    // alpha at global position immediately before each row start (row 0 unused).
    let mut row_prev_alpha = vec![0.0f64; batch * n_states];
    let mut prev_end = vec![0.0f64; n_states];

    let total_sites = (batch as u64).saturating_mul(s_max as u64);
    let pb_fwd = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites, label, "forward"))
    } else {
        None
    };

    for b in 0..batch {
        let start_prev = if b == 0 {
            None
        } else {
            let dst = &mut row_prev_alpha[b * n_states..(b + 1) * n_states];
            dst.copy_from_slice(&prev_end);
            Some(&row_prev_alpha[b * n_states..(b + 1) * n_states])
        };
        forward_fill_row(
            b,
            pi,
            &a_col,
            &em_rows,
            x,
            start_prev,
            &mut alpha_row,
            &mut c_norm,
            &mut tmp,
        )?;
        for v in &c_norm {
            loglike += v.ln();
        }
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);
        if let Some(pb) = &pb_fwd {
            pb.inc(s_max as u64);
        }
    }
    if let Some(pb) = pb_fwd {
        pb.finish_with_message(format!("{label} forward done"));
    }

    let pb_bwd = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites, label, "backward"))
    } else {
        None
    };

    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];
    for b_rev in 0..batch {
        let b = batch - 1 - b_rev;
        let start_prev = if b == 0 {
            None
        } else {
            Some(&row_prev_alpha[b * n_states..(b + 1) * n_states])
        };
        forward_fill_row(
            b,
            pi,
            &a_col,
            &em_rows,
            x,
            start_prev,
            &mut alpha_row,
            &mut c_norm,
            &mut tmp,
        )?;

        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            let obs = obs_to_idx(x[(b, t)]);
            let alpha_t = &alpha_row[t * n_states..(t + 1) * n_states];
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                if b == 0 && t == 0 {
                    g0[k] += g;
                }
                match obs {
                    0 => gobs0[k] += g,
                    1 => gobs1[k] += g,
                    _ => gobs2[k] += g,
                }
            }

            if b == 0 && t == 0 {
                continue;
            }

            let norm_t = c_norm[t];
            let em_obs = &em_rows[obs];
            let alpha_prev = if t > 0 {
                &alpha_row[(t - 1) * n_states..t * n_states]
            } else {
                &row_prev_alpha[b * n_states..(b + 1) * n_states]
            };

            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let alpha_prev_i = alpha_prev[i];
                let mut acc = 0.0;
                for j in 0..n_states {
                    let emit_beta = em_obs[j] * beta[j];
                    let trans = a_i[j];
                    xi[(i, j)] += alpha_prev_i * trans * emit_beta / norm_t;
                    acc += trans * emit_beta;
                }
                beta_new[i] = acc / norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }
        if let Some(pb) = &pb_bwd {
            pb.inc(s_max as u64);
        }
    }

    if let Some(pb) = pb_bwd {
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

fn e_step_streaming_parallel(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let (batch, _s_max) = x.dim();
    let n_states = pi.len();

    let mut a_row = vec![0.0f64; n_states * n_states];
    let mut a_col = vec![0.0f64; n_states * n_states];
    for i in 0..n_states {
        for j in 0..n_states {
            let v = a[(i, j)];
            a_row[i * n_states + j] = v;
            a_col[j * n_states + i] = v;
        }
    }
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let pb_batches = if progress_enabled && batch > 0 {
        Some(progress::bar_raw(batch as u64, label, "parallel batches"))
    } else {
        None
    };
    let pb_worker = pb_batches.clone();

    let acc = (0..batch)
        .into_par_iter()
        .try_fold(
            || BatchAccum::zeros(n_states),
            |mut local, b| {
                accumulate_one_batch(&mut local, b, pi, &a_row, &a_col, &em_rows, x)?;
                if let Some(pb) = &pb_worker {
                    pb.inc(1);
                }
                Ok::<BatchAccum, anyhow::Error>(local)
            },
        )
        .try_reduce(
            || BatchAccum::zeros(n_states),
            |mut left, right| {
                left.add_assign(&right);
                Ok::<BatchAccum, anyhow::Error>(left)
            },
        )?;

    if let Some(pb) = pb_batches {
        pb.finish_with_message(format!("{label} parallel done"));
    }

    acc.into_estep(n_states)
}

fn e_step_streaming_sequential(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let (batch, s_max) = x.dim();
    let n_states = pi.len();

    let mut g0 = vec![0.0; n_states];
    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut xi = Array2::<f64>::zeros((n_states, n_states));
    let mut loglike = 0.0;

    // Keep both row-major and column-major views to avoid tiny GEMM call overhead
    // and preserve contiguous reads in forward/backward passes.
    let mut a_row = vec![0.0f64; n_states * n_states];
    let mut a_col = vec![0.0f64; n_states * n_states];
    for i in 0..n_states {
        for j in 0..n_states {
            let v = a[(i, j)];
            a_row[i * n_states + j] = v;
            a_col[j * n_states + i] = v;
        }
    }

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
        let em_obs0 = em.row(obs0);
        let mut norm = 0.0;
        for k in 0..n_states {
            let v = pi[k] * em_obs0[k];
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
            let em_obs = em.row(obs);
            let prev = &alpha[(t - 1) * n_states..t * n_states];

            norm = 0.0;
            for k in 0..n_states {
                let a_col_k = &a_col[k * n_states..(k + 1) * n_states];
                let mut acc = 0.0;
                for i in 0..n_states {
                    acc += prev[i] * a_col_k[i];
                }
                let v = acc * em_obs[k];
                tmp[k] = v;
                norm += v;
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

        for v in &c_norm {
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
            let em_obs = em.row(obs);
            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let alpha_prev_i = alpha_prev[i];
                let mut acc = 0.0;
                for j in 0..n_states {
                    let emit_beta = em_obs[j] * beta[j];
                    let trans = a_i[j];
                    xi[(i, j)] += alpha_prev_i * trans * emit_beta / norm_t;
                    acc += trans * emit_beta;
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

pub fn e_step_streaming(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    x: &Array2<u8>,
    rows_independent: bool,
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let (batch, _s_max) = x.dim();
    let n_states = pi.len();
    if a.nrows() != n_states || a.ncols() != n_states {
        bail!("transition matrix shape mismatch");
    }
    if em.nrows() < 3 || em.ncols() != n_states {
        bail!("emission matrix shape mismatch");
    }

    if !rows_independent {
        e_step_streaming_contiguous_rows(pi, a, em, x, progress_enabled, label)
    } else if batch > 1 && rayon::current_num_threads() > 1 {
        e_step_streaming_parallel(pi, a, em, x, progress_enabled, label)
    } else {
        e_step_streaming_sequential(pi, a, em, x, progress_enabled, label)
    }
}
