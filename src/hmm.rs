use anyhow::{Result, bail};
use ndarray::Array2;

use crate::progress;

#[derive(Debug, Clone)]
pub struct SufficientStats {
    pub gobs: [Vec<f64>; 3],
    pub xi: Array2<f64>,
}

#[derive(Debug)]
pub struct EStepResult {
    pub stats: SufficientStats,
    pub loglike: f64,
}

#[derive(Debug, Clone)]
struct RowForwardCache {
    alpha: Vec<f64>,
    c_norm: Vec<f64>,
}

fn alpha_cache_budget_bytes() -> usize {
    // Allow tuning without changing CLI: PSMC_ALPHA_CACHE_MB=0 disables caching.
    const DEFAULT_MB: usize = 1024;
    let mb = std::env::var("PSMC_ALPHA_CACHE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MB);
    mb.saturating_mul(1024).saturating_mul(1024)
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
    pi: &[f64],
    a_col: &[f64],
    em_rows: &[Vec<f64>; 3],
    obs_row: &[u8],
    start_prev: Option<&[f64]>,
    alpha_row: &mut [f64],
    c_norm: &mut [f64],
    tmp: &mut [f64],
) -> Result<()> {
    let s_max = obs_row.len();
    let n_states = pi.len();
    for t in 0..s_max {
        let obs = obs_to_idx(obs_row[t]);
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
            bail!("normalization factor is zero in forward pass");
        }
        c_norm[t] = norm;
        let alpha_t = &mut alpha_row[t * n_states..(t + 1) * n_states];
        for k in 0..n_states {
            alpha_t[k] = tmp[k] / norm;
        }
    }
    Ok(())
}

pub fn e_step_streaming(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    let n_rows = rows.len();
    let n_states = pi.len();
    if n_rows == 0 {
        bail!("empty input rows for E-step");
    }
    if row_starts.len() != n_rows {
        bail!("row_starts length mismatch");
    }
    if !row_starts[0] {
        bail!("row_starts[0] must be true");
    }
    if a.nrows() != n_states || a.ncols() != n_states {
        bail!("transition matrix shape mismatch");
    }
    if em.nrows() < 3 || em.ncols() != n_states {
        bail!("emission matrix shape mismatch");
    }
    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() {
            bail!("empty observation row at index {i}");
        }
        if i == 0 && !row_starts[i] {
            bail!("first row must be sequence start");
        }
    }

    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut xi_flat = vec![0.0f64; n_states * n_states];
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

    let total_sites: u64 = rows.iter().map(|r| r.len() as u64).sum();
    let max_row_len = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut alpha_row_buf = vec![0.0f64; max_row_len * n_states];
    let mut c_norm_buf = vec![0.0f64; max_row_len];
    let mut tmp = vec![0.0f64; n_states];

    let pb_fwd = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites, label, "forward"))
    } else {
        None
    };

    let mut row_prev_alpha: Vec<Option<Vec<f64>>> = vec![None; n_rows];
    let mut row_cache: Vec<Option<RowForwardCache>> = vec![None; n_rows];
    let mut prev_end = vec![0.0f64; n_states];
    let cache_budget = alpha_cache_budget_bytes();
    let mut total_cache_bytes = 0usize;
    for row in rows {
        let row_bytes = (row.len() * n_states + row.len()) * std::mem::size_of::<f64>();
        total_cache_bytes = total_cache_bytes.saturating_add(row_bytes);
    }
    let enable_cache = cache_budget > 0 && total_cache_bytes <= cache_budget;

    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];

        if !row_starts[r] {
            row_prev_alpha[r] = Some(prev_end.clone());
        }
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi, &a_col, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
        )?;
        for v in c_norm.iter() {
            loglike += v.ln();
        }
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);

        if enable_cache {
            row_cache[r] = Some(RowForwardCache {
                alpha: alpha_row.to_vec(),
                c_norm: c_norm.to_vec(),
            });
        }

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
    for r_rev in 0..n_rows {
        let r = n_rows - 1 - r_rev;
        let obs_row = &rows[r];
        let s_max = obs_row.len();

        let (alpha_row, c_norm): (&[f64], &[f64]) = if let Some(cache) = row_cache[r].as_ref() {
            (&cache.alpha, &cache.c_norm)
        } else {
            let alpha_row = &mut alpha_row_buf[..s_max * n_states];
            let c_norm = &mut c_norm_buf[..s_max];
            let start_prev = row_prev_alpha[r].as_deref();
            forward_fill_row(
                pi, &a_col, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
            )?;
            (alpha_row, c_norm)
        };

        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            let obs = obs_to_idx(obs_row[t]);
            let alpha_t = &alpha_row[t * n_states..(t + 1) * n_states];
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                match obs {
                    0 => gobs0[k] += g,
                    1 => gobs1[k] += g,
                    _ => gobs2[k] += g,
                }
            }

            if t == 0 && row_starts[r] {
                continue;
            }

            let norm_t = c_norm[t];
            let inv_norm_t = 1.0 / norm_t;
            let em_obs = &em_rows[obs];
            let alpha_prev = if t > 0 {
                &alpha_row[(t - 1) * n_states..t * n_states]
            } else {
                row_prev_alpha[r]
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("missing row boundary alpha state"))?
            };

            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let xi_i = &mut xi_flat[i * n_states..(i + 1) * n_states];
                let alpha_prev_i = alpha_prev[i];
                let xi_scale = alpha_prev_i * inv_norm_t;
                let mut acc = 0.0;
                for j in 0..n_states {
                    let emit_beta = em_obs[j] * beta[j];
                    let trans = a_i[j];
                    xi_i[j] += xi_scale * trans * emit_beta;
                    acc += trans * emit_beta;
                }
                beta_new[i] = acc * inv_norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }

        if row_starts[r] {
            beta.fill(1.0);
        }
        if let Some(pb) = &pb_bwd {
            pb.inc(s_max as u64);
        }
    }

    if let Some(pb) = pb_bwd {
        pb.finish_with_message(format!("{label} backward done"));
    }

    let xi = Array2::from_shape_vec((n_states, n_states), xi_flat)
        .expect("internal error: xi shape mismatch");

    Ok(EStepResult {
        stats: SufficientStats {
            gobs: [gobs0, gobs1, gobs2],
            xi,
        },
        loglike,
    })
}
