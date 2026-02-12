use anyhow::{Result, bail};
use ndarray::Array2;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EStepPhase {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy)]
pub struct EStepProgress {
    pub phase: EStepPhase,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, Clone)]
pub struct TmrcaReportData {
    pub sampled_mean: Vec<[f64; 2]>,
    pub sampled_map: Vec<[f64; 2]>,
    pub sampled_lo: Vec<[f64; 2]>,
    pub sampled_hi: Vec<[f64; 2]>,
    pub state_mass: Vec<f64>,
    pub state_years: Vec<f64>,
    pub total_sites: usize,
    pub seq_count: usize,
}

#[derive(Debug, Clone)]
struct RowForwardCache {
    // Alpha is cache-heavy: store as f32 to reduce memory footprint and
    // keep full-row caching enabled on long inputs (faster than recomputing forward).
    alpha: Vec<f32>,
    // c_norm is also cached as f32 to further reduce memory pressure.
    c_norm: Vec<f32>,
}

const DEFAULT_ALPHA_CACHE_BUDGET_BYTES: usize = 2048 * 1024 * 1024; // 2 GiB

fn alpha_cache_budget_bytes() -> usize {
    match std::env::var("PSMC_ALPHA_CACHE_MB") {
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(mb) if mb > 0 => mb.saturating_mul(1024 * 1024),
            _ => DEFAULT_ALPHA_CACHE_BUDGET_BYTES,
        },
        Err(_) => DEFAULT_ALPHA_CACHE_BUDGET_BYTES,
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

#[inline]
fn obs_to_char(v: u8) -> char {
    if v == 0 {
        'T'
    } else if v == 1 {
        'K'
    } else {
        'N'
    }
}

#[inline]
fn quantile_from_probs(probs: &[f64], state_years: &[f64], q: f64) -> f64 {
    debug_assert_eq!(probs.len(), state_years.len());
    if probs.is_empty() {
        return 0.0;
    }
    let target = q.clamp(0.0, 1.0);
    let mut cdf = 0.0;
    for (k, p) in probs.iter().enumerate() {
        cdf += *p;
        if cdf >= target {
            return state_years[k];
        }
    }
    state_years.last().copied().unwrap_or(0.0)
}

#[inline]
fn matvec_row_major(src: &[f64], a_row: &[f64], dst: &mut [f64]) {
    let n_states = src.len();
    debug_assert_eq!(dst.len(), n_states);
    dst.fill(0.0);
    for i in 0..n_states {
        let s = src[i];
        let row = &a_row[i * n_states..(i + 1) * n_states];
        let mut k = 0usize;
        while k + 4 <= n_states {
            dst[k] += s * row[k];
            dst[k + 1] += s * row[k + 1];
            dst[k + 2] += s * row[k + 2];
            dst[k + 3] += s * row[k + 3];
            k += 4;
        }
        while k < n_states {
            dst[k] += s * row[k];
            k += 1;
        }
    }
}

#[inline(always)]
fn matvec_row_major_fast(src: &[f64], a_row: &[f64], dst: &mut [f64]) {
    let n_states = src.len();
    debug_assert_eq!(dst.len(), n_states);
    debug_assert_eq!(a_row.len(), n_states * n_states);
    // Experimental lower-level kernel for A/B testing.
    unsafe {
        std::ptr::write_bytes(dst.as_mut_ptr(), 0, n_states);
        let src_ptr = src.as_ptr();
        let a_ptr = a_row.as_ptr();
        let dst_ptr = dst.as_mut_ptr();
        let mut i = 0usize;
        while i < n_states {
            let s = *src_ptr.add(i);
            let row_ptr = a_ptr.add(i * n_states);
            let mut k = 0usize;
            while k + 8 <= n_states {
                *dst_ptr.add(k) += s * *row_ptr.add(k);
                *dst_ptr.add(k + 1) += s * *row_ptr.add(k + 1);
                *dst_ptr.add(k + 2) += s * *row_ptr.add(k + 2);
                *dst_ptr.add(k + 3) += s * *row_ptr.add(k + 3);
                *dst_ptr.add(k + 4) += s * *row_ptr.add(k + 4);
                *dst_ptr.add(k + 5) += s * *row_ptr.add(k + 5);
                *dst_ptr.add(k + 6) += s * *row_ptr.add(k + 6);
                *dst_ptr.add(k + 7) += s * *row_ptr.add(k + 7);
                k += 8;
            }
            while k + 4 <= n_states {
                *dst_ptr.add(k) += s * *row_ptr.add(k);
                *dst_ptr.add(k + 1) += s * *row_ptr.add(k + 1);
                *dst_ptr.add(k + 2) += s * *row_ptr.add(k + 2);
                *dst_ptr.add(k + 3) += s * *row_ptr.add(k + 3);
                k += 4;
            }
            while k < n_states {
                *dst_ptr.add(k) += s * *row_ptr.add(k);
                k += 1;
            }
            i += 1;
        }
    }
}

fn forward_fill_row(
    pi: &[f64],
    a_row: &[f64],
    em_rows: &[Vec<f64>; 3],
    obs_row: &[u8],
    start_prev: Option<&[f64]>,
    alpha_row: &mut [f64],
    c_norm: &mut [f64],
    tmp: &mut [f64],
    use_fast_kernel: bool,
) -> Result<()> {
    let s_max = obs_row.len();
    let n_states = pi.len();
    for t in 0..s_max {
        let obs = obs_to_idx(obs_row[t]);
        let em_obs = &em_rows[obs];

        let mut norm = 0.0f64;
        if t == 0 && start_prev.is_none() {
            for k in 0..n_states {
                let v = pi[k] * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        } else {
            let src = if t == 0 {
                start_prev.expect("checked is_some above")
            } else {
                &alpha_row[(t - 1) * n_states..t * n_states]
            };
            if use_fast_kernel {
                matvec_row_major_fast(src, a_row, tmp);
            } else {
                matvec_row_major(src, a_row, tmp);
            }
            for k in 0..n_states {
                let v = tmp[k] * em_obs[k];
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

fn sequence_ranges(rows: &[Vec<u8>], row_starts: &[bool]) -> Vec<(usize, usize, u64)> {
    let mut out = Vec::new();
    if rows.is_empty() {
        return out;
    }
    let mut start = 0usize;
    let mut sites = rows[0].len() as u64;
    for i in 1..rows.len() {
        if row_starts[i] {
            out.push((start, i, sites));
            start = i;
            sites = 0;
        }
        sites = sites.saturating_add(rows[i].len() as u64);
    }
    out.push((start, rows.len(), sites));
    out
}

fn e_step_sequence_no_cache(
    pi: &[f64],
    a_row: &[f64],
    em_rows: &[Vec<f64>; 3],
    rows: &[Vec<u8>],
    row_starts: &[bool],
    use_fast_kernel: bool,
) -> Result<EStepResult> {
    let n_rows = rows.len();
    let n_states = pi.len();

    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut xi_flat = vec![0.0f64; n_states * n_states];
    let mut loglike = 0.0;

    let max_row_len = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut alpha_row_buf = vec![0.0f64; max_row_len * n_states];
    let mut c_norm_buf = vec![0.0f64; max_row_len];
    let mut tmp = vec![0.0f64; n_states];

    let mut row_prev_alpha: Vec<Option<Vec<f64>>> = vec![None; n_rows];
    let mut prev_end = vec![0.0f64; n_states];
    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        if !row_starts[r] {
            row_prev_alpha[r] = Some(prev_end.clone());
        }
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi,
            a_row,
            em_rows,
            obs_row,
            start_prev,
            alpha_row,
            c_norm,
            &mut tmp,
            use_fast_kernel,
        )?;
        for v in c_norm.iter() {
            loglike += v.ln();
        }
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);
    }

    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];
    let mut emit_beta = vec![0.0f64; n_states];
    for r_rev in 0..n_rows {
        let r = n_rows - 1 - r_rev;
        let obs_row = &rows[r];
        let s_max = obs_row.len();

        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi,
            a_row,
            em_rows,
            obs_row,
            start_prev,
            alpha_row,
            c_norm,
            &mut tmp,
            use_fast_kernel,
        )?;

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
            for j in 0..n_states {
                emit_beta[j] = em_obs[j] * beta[j];
            }
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
                let xi_scale = alpha_prev[i] * inv_norm_t;
                let mut acc = 0.0;
                for j in 0..n_states {
                    let trans = a_i[j];
                    xi_i[j] += xi_scale * trans * emit_beta[j];
                    acc += trans * emit_beta[j];
                }
                beta_new[i] = acc * inv_norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }

        if row_starts[r] {
            beta.fill(1.0);
        }
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

pub fn e_step_streaming(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    e_step_streaming_with_progress(pi, a, em, rows, row_starts, progress_enabled, label, None)
}

pub fn e_step_streaming_with_progress(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    progress_enabled: bool,
    label: &str,
    mut on_progress: Option<&mut dyn FnMut(EStepProgress)>,
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
    for i in 0..n_states {
        for j in 0..n_states {
            let v = a[(i, j)];
            a_row[i * n_states + j] = v;
        }
    }
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let total_sites: u64 = rows.iter().map(|r| r.len() as u64).sum();
    let seq_ranges = sequence_ranges(rows, row_starts);
    // Avoid nested Rayon parallelism: only parallelize sequences at top-level E-step calls.
    let can_parallel_sequences = seq_ranges.len() > 1
        && rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none();
    if can_parallel_sequences {
        let use_fast_kernel = true;
        let pb_fwd = if progress_enabled && total_sites > 0 {
            Some(progress::bar_raw(total_sites, label, "forward"))
        } else {
            None
        };
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EStepProgress {
                phase: EStepPhase::Forward,
                done: 0,
                total: total_sites,
            });
        }

        let seq_results = seq_ranges
            .par_iter()
            .map(|(start, end, sites)| {
                e_step_sequence_no_cache(
                    pi,
                    &a_row,
                    &em_rows,
                    &rows[*start..*end],
                    &row_starts[*start..*end],
                    use_fast_kernel,
                )
                .map(|res| (*sites, res))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut fwd_done = 0u64;
        for (sites, _) in &seq_results {
            if let Some(pb) = &pb_fwd {
                pb.inc(*sites);
            }
            fwd_done = fwd_done.saturating_add(*sites);
            if let Some(cb) = on_progress.as_mut() {
                (*cb)(EStepProgress {
                    phase: EStepPhase::Forward,
                    done: fwd_done,
                    total: total_sites,
                });
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
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EStepProgress {
                phase: EStepPhase::Backward,
                done: 0,
                total: total_sites,
            });
        }

        let mut bwd_done = 0u64;
        for (sites, _) in &seq_results {
            if let Some(pb) = &pb_bwd {
                pb.inc(*sites);
            }
            bwd_done = bwd_done.saturating_add(*sites);
            if let Some(cb) = on_progress.as_mut() {
                (*cb)(EStepProgress {
                    phase: EStepPhase::Backward,
                    done: bwd_done,
                    total: total_sites,
                });
            }
        }
        if let Some(pb) = pb_bwd {
            pb.finish_with_message(format!("{label} backward done"));
        }

        for (_, res) in seq_results {
            loglike += res.loglike;
            for k in 0..n_states {
                gobs0[k] += res.stats.gobs[0][k];
                gobs1[k] += res.stats.gobs[1][k];
                gobs2[k] += res.stats.gobs[2][k];
            }
            for (dst, src) in xi_flat.iter_mut().zip(res.stats.xi.iter()) {
                *dst += *src;
            }
        }

        let xi = Array2::from_shape_vec((n_states, n_states), xi_flat)
            .expect("internal error: xi shape mismatch");
        return Ok(EStepResult {
            stats: SufficientStats {
                gobs: [gobs0, gobs1, gobs2],
                xi,
            },
            loglike,
        });
    }

    let max_row_len = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut alpha_row_buf = vec![0.0f64; max_row_len * n_states];
    let mut c_norm_buf = vec![0.0f64; max_row_len];
    let mut tmp = vec![0.0f64; n_states];
    // E-step always uses the lower-level hot kernel.
    let use_fast_kernel = true;

    let pb_fwd = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites, label, "forward"))
    } else {
        None
    };
    if let Some(cb) = on_progress.as_mut() {
        (*cb)(EStepProgress {
            phase: EStepPhase::Forward,
            done: 0,
            total: total_sites,
        });
    }

    let mut row_prev_alpha: Vec<Option<Vec<f64>>> = vec![None; n_rows];
    let mut row_cache: Vec<Option<RowForwardCache>> = vec![None; n_rows];
    let mut prev_end = vec![0.0f64; n_states];
    let mut row_cache_bytes = vec![0usize; n_rows];
    let mut total_cache_bytes = 0usize;
    for (r, row) in rows.iter().enumerate() {
        let row_bytes = row
            .len()
            .saturating_mul(n_states)
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_add(row.len().saturating_mul(std::mem::size_of::<f32>()));
        row_cache_bytes[r] = row_bytes;
        total_cache_bytes = total_cache_bytes.saturating_add(row_bytes);
    }
    let cache_budget = alpha_cache_budget_bytes();
    let mut cache_mask = vec![false; n_rows];
    let mut used_cache_bytes = 0usize;
    if total_cache_bytes <= cache_budget {
        cache_mask.fill(true);
        used_cache_bytes = total_cache_bytes;
    } else {
        // Cache the most expensive rows first to reduce backward-pass recompute.
        let mut row_order: Vec<usize> = (0..n_rows).collect();
        row_order.sort_unstable_by_key(|&r| std::cmp::Reverse(row_cache_bytes[r]));
        for r in row_order {
            let row_bytes = row_cache_bytes[r];
            let remain = cache_budget.saturating_sub(used_cache_bytes);
            if row_bytes <= remain {
                cache_mask[r] = true;
                used_cache_bytes = used_cache_bytes.saturating_add(row_bytes);
            }
        }
    }

    let cached_rows = cache_mask.iter().filter(|&&x| x).count();
    if progress_enabled && cached_rows == 0 {
        let need_mb = (total_cache_bytes as f64) / (1024.0 * 1024.0);
        let have_mb = (cache_budget as f64) / (1024.0 * 1024.0);
        eprintln!(
            "Warning: E-step alpha cache disabled (need ~{need_mb:.1} MB, budget {have_mb:.1} MB). Set PSMC_ALPHA_CACHE_MB higher for speed."
        );
    } else if progress_enabled && cached_rows < n_rows {
        let need_mb = (total_cache_bytes as f64) / (1024.0 * 1024.0);
        let using_mb = (used_cache_bytes as f64) / (1024.0 * 1024.0);
        let pct = 100.0 * (cached_rows as f64) / (n_rows as f64);
        eprintln!(
            "Info: E-step alpha cache partially enabled: {cached_rows}/{n_rows} rows ({pct:.1}%), using ~{using_mb:.1} MB (need ~{need_mb:.1} MB for full cache)."
        );
    }

    let mut fwd_done = 0u64;
    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];

        if !row_starts[r] {
            row_prev_alpha[r] = Some(prev_end.clone());
        }
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi,
            &a_row,
            &em_rows,
            obs_row,
            start_prev,
            alpha_row,
            c_norm,
            &mut tmp,
            use_fast_kernel,
        )?;
        for v in c_norm.iter() {
            loglike += v.ln();
        }
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);

        if cache_mask[r] {
            row_cache[r] = Some(RowForwardCache {
                alpha: alpha_row.iter().map(|v| *v as f32).collect(),
                c_norm: c_norm.iter().map(|v| *v as f32).collect(),
            });
        }

        if let Some(pb) = &pb_fwd {
            pb.inc(s_max as u64);
        }
        fwd_done = fwd_done.saturating_add(s_max as u64);
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EStepProgress {
                phase: EStepPhase::Forward,
                done: fwd_done,
                total: total_sites,
            });
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
    if let Some(cb) = on_progress.as_mut() {
        (*cb)(EStepProgress {
            phase: EStepPhase::Backward,
            done: 0,
            total: total_sites,
        });
    }

    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];
    let mut emit_beta = vec![0.0f64; n_states];
    let mut bwd_done = 0u64;
    for r_rev in 0..n_rows {
        let r = n_rows - 1 - r_rev;
        let obs_row = &rows[r];
        let s_max = obs_row.len();
        if let Some(cache) = row_cache[r].as_ref() {
            let alpha_row = &cache.alpha;
            let c_norm = &cache.c_norm;
            for t_rev in 0..s_max {
                let t = s_max - 1 - t_rev;
                let obs = obs_to_idx(obs_row[t]);
                let alpha_t_off = t * n_states;
                for k in 0..n_states {
                    let g = (alpha_row[alpha_t_off + k] as f64) * beta[k];
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
                let inv_norm_t = 1.0 / (norm_t as f64);
                let em_obs = &em_rows[obs];
                for j in 0..n_states {
                    emit_beta[j] = em_obs[j] * beta[j];
                }
                let alpha_prev_off = if t > 0 {
                    Some((t - 1) * n_states)
                } else {
                    None
                };
                let row_prev = if alpha_prev_off.is_none() {
                    Some(
                        row_prev_alpha[r]
                            .as_deref()
                            .ok_or_else(|| anyhow::anyhow!("missing row boundary alpha state"))?,
                    )
                } else {
                    None
                };
                for i in 0..n_states {
                    let a_i = &a_row[i * n_states..(i + 1) * n_states];
                    let xi_i = &mut xi_flat[i * n_states..(i + 1) * n_states];
                    let alpha_prev_i = if let Some(off) = alpha_prev_off {
                        alpha_row[off + i] as f64
                    } else {
                        row_prev.expect("row_prev set above")[i]
                    };
                    let xi_scale = alpha_prev_i * inv_norm_t;
                    let mut acc = 0.0;
                    for j in 0..n_states {
                        let trans = a_i[j];
                        xi_i[j] += xi_scale * trans * emit_beta[j];
                        acc += trans * emit_beta[j];
                    }
                    beta_new[i] = acc * inv_norm_t;
                }
                std::mem::swap(&mut beta, &mut beta_new);
            }
        } else {
            let alpha_row = &mut alpha_row_buf[..s_max * n_states];
            let c_norm = &mut c_norm_buf[..s_max];
            let start_prev = row_prev_alpha[r].as_deref();
            forward_fill_row(
                pi,
                &a_row,
                &em_rows,
                obs_row,
                start_prev,
                alpha_row,
                c_norm,
                &mut tmp,
                use_fast_kernel,
            )?;

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
                for j in 0..n_states {
                    emit_beta[j] = em_obs[j] * beta[j];
                }
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
                        let trans = a_i[j];
                        xi_i[j] += xi_scale * trans * emit_beta[j];
                        acc += trans * emit_beta[j];
                    }
                    beta_new[i] = acc * inv_norm_t;
                }
                std::mem::swap(&mut beta, &mut beta_new);
            }
        }

        if row_starts[r] {
            beta.fill(1.0);
        }
        if let Some(pb) = &pb_bwd {
            pb.inc(s_max as u64);
        }
        bwd_done = bwd_done.saturating_add(s_max as u64);
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EStepProgress {
                phase: EStepPhase::Backward,
                done: bwd_done,
                total: total_sites,
            });
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

pub fn write_tmrca_posterior_tsv(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    tmrca_years: &[f64],
    out_path: &Path,
    progress_enabled: bool,
    max_points: usize,
) -> Result<TmrcaReportData> {
    let n_rows = rows.len();
    let n_states = pi.len();
    if n_rows == 0 {
        bail!("empty input rows for TMRCA posterior");
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
    if tmrca_years.len() != n_states {
        bail!(
            "tmrca_years length {} != n_states {}",
            tmrca_years.len(),
            n_states
        );
    }
    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() {
            bail!("empty observation row at index {i}");
        }
    }

    let total_sites: usize = rows.iter().map(Vec::len).sum();
    let max_row_len = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut alpha_row_buf = vec![0.0f64; max_row_len * n_states];
    let mut c_norm_buf = vec![0.0f64; max_row_len];
    let mut tmp = vec![0.0f64; n_states];
    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];
    let mut emit_beta = vec![0.0f64; n_states];

    let mut a_row = vec![0.0f64; n_states * n_states];
    for i in 0..n_states {
        for j in 0..n_states {
            a_row[i * n_states + j] = a[(i, j)];
        }
    }
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let pb_prep = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites as u64, "TMRCA", "prep"))
    } else {
        None
    };

    let mut row_prev_alpha: Vec<Option<Vec<f64>>> = vec![None; n_rows];
    let mut prev_end = vec![0.0f64; n_states];
    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        if !row_starts[r] {
            row_prev_alpha[r] = Some(prev_end.clone());
        }
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp, false,
        )?;
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);
        if let Some(pb) = &pb_prep {
            pb.inc(s_max as u64);
        }
    }
    if let Some(pb) = pb_prep {
        pb.finish_with_message("TMRCA prep done");
    }

    let pb_back = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites as u64, "TMRCA", "backward"))
    } else {
        None
    };

    let mut row_end_beta: Vec<Vec<f64>> = vec![vec![1.0f64; n_states]; n_rows];
    beta.fill(1.0);
    for r_rev in 0..n_rows {
        let r = n_rows - 1 - r_rev;
        row_end_beta[r].copy_from_slice(&beta);

        let obs_row = &rows[r];
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp, false,
        )?;

        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            if t == 0 && row_starts[r] {
                continue;
            }
            let obs = obs_to_idx(obs_row[t]);
            let norm_t = c_norm[t];
            let inv_norm_t = 1.0 / norm_t;
            let em_obs = &em_rows[obs];
            for j in 0..n_states {
                emit_beta[j] = em_obs[j] * beta[j];
            }
            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let mut acc = 0.0;
                for j in 0..n_states {
                    acc += a_i[j] * emit_beta[j];
                }
                beta_new[i] = acc * inv_norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }

        if row_starts[r] {
            beta.fill(1.0);
        }
        if let Some(pb) = &pb_back {
            pb.inc(s_max as u64);
        }
    }
    if let Some(pb) = pb_back {
        pb.finish_with_message("TMRCA backward done");
    }

    let mut row_seq_id = vec![0usize; n_rows];
    let mut row_seq_offset = vec![0usize; n_rows];
    let mut seq_id = 0usize;
    let mut seq_offset = 0usize;
    let mut seq_count = 0usize;
    for r in 0..n_rows {
        if row_starts[r] {
            if r == 0 {
                seq_id = 0;
                seq_count = 1;
            } else {
                seq_id += 1;
                seq_count += 1;
            }
            seq_offset = 0;
        }
        row_seq_id[r] = seq_id;
        row_seq_offset[r] = seq_offset;
        seq_offset += rows[r].len();
    }

    let pb_out = if progress_enabled && total_sites > 0 {
        Some(progress::bar_raw(total_sites as u64, "TMRCA", "write"))
    } else {
        None
    };

    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(
        b"seq_id\tseq_bin\tglobal_bin\tobs\tmap_state\ttmrca_map_years\ttmrca_mean_years\ttmrca_q025_years\ttmrca_q975_years\tpmax\tentropy\n",
    )?;

    let mut state_mass = vec![0.0f64; n_states];
    let mut global_bin = 0usize;
    let stride = if max_points == 0 {
        1
    } else {
        total_sites.div_ceil(max_points).max(1)
    };
    let mut sampled_mean = Vec::<[f64; 2]>::new();
    let mut sampled_map = Vec::<[f64; 2]>::new();
    let mut sampled_lo = Vec::<[f64; 2]>::new();
    let mut sampled_hi = Vec::<[f64; 2]>::new();

    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        let start_prev = row_prev_alpha[r].as_deref();
        forward_fill_row(
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp, false,
        )?;
        beta.copy_from_slice(&row_end_beta[r]);

        let mut mean_years = vec![0.0f64; s_max];
        let mut map_years = vec![0.0f64; s_max];
        let mut q025_years = vec![0.0f64; s_max];
        let mut q975_years = vec![0.0f64; s_max];
        let mut map_state = vec![0usize; s_max];
        let mut pmax_vals = vec![0.0f64; s_max];
        let mut entropy_vals = vec![0.0f64; s_max];

        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            let obs = obs_to_idx(obs_row[t]);
            let alpha_t = &alpha_row[t * n_states..(t + 1) * n_states];

            let mut gamma_norm = 0.0f64;
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                tmp[k] = g;
                gamma_norm += g;
            }
            if gamma_norm <= 0.0 {
                bail!("normalization factor is zero in TMRCA posterior");
            }
            let inv_gamma_norm = 1.0 / gamma_norm;

            let mut mean = 0.0f64;
            let mut max_p = -1.0f64;
            let mut max_k = 0usize;
            let mut entropy = 0.0f64;
            for k in 0..n_states {
                let p = tmp[k] * inv_gamma_norm;
                tmp[k] = p;
                state_mass[k] += p;
                mean += p * tmrca_years[k];
                if p > max_p {
                    max_p = p;
                    max_k = k;
                }
                if p > 0.0 {
                    entropy -= p * p.ln();
                }
            }
            mean_years[t] = mean;
            map_state[t] = max_k;
            map_years[t] = tmrca_years[max_k];
            q025_years[t] = quantile_from_probs(&tmp[..n_states], tmrca_years, 0.025);
            q975_years[t] = quantile_from_probs(&tmp[..n_states], tmrca_years, 0.975);
            pmax_vals[t] = max_p;
            entropy_vals[t] = entropy;

            if t == 0 && row_starts[r] {
                continue;
            }
            let norm_t = c_norm[t];
            let inv_norm_t = 1.0 / norm_t;
            let em_obs = &em_rows[obs];
            for j in 0..n_states {
                emit_beta[j] = em_obs[j] * beta[j];
            }
            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let mut acc = 0.0;
                for j in 0..n_states {
                    acc += a_i[j] * emit_beta[j];
                }
                beta_new[i] = acc * inv_norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }

        for t in 0..s_max {
            let seq_bin = row_seq_offset[r] + t;
            let obs_ch = obs_to_char(obs_row[t]);
            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
                row_seq_id[r],
                seq_bin,
                global_bin,
                obs_ch,
                map_state[t],
                map_years[t],
                mean_years[t],
                q025_years[t],
                q975_years[t],
                pmax_vals[t],
                entropy_vals[t],
            )?;

            if global_bin % stride == 0 {
                sampled_mean.push([global_bin as f64, mean_years[t]]);
                sampled_map.push([global_bin as f64, map_years[t]]);
                sampled_lo.push([global_bin as f64, q025_years[t]]);
                sampled_hi.push([global_bin as f64, q975_years[t]]);
            }
            global_bin += 1;
        }

        if let Some(pb) = &pb_out {
            pb.inc(s_max as u64);
        }
    }
    writer.flush()?;

    if let Some(pb) = pb_out {
        pb.finish_with_message("TMRCA write done");
    }

    Ok(TmrcaReportData {
        sampled_mean,
        sampled_map,
        sampled_lo,
        sampled_hi,
        state_mass,
        state_years: tmrca_years.to_vec(),
        total_sites,
        seq_count,
    })
}
