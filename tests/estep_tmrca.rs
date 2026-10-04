//! Posterior TMRCA decoding: the bounded-memory writer must agree with the
//! previous full-row implementation (copied below as the reference), write
//! gzip when the path ends in `.gz`, and not depend on the thread count.

use anyhow::{Result, bail};
use flate2::read::MultiGzDecoder;
use ndarray::Array2;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::{CHECKPOINT_STRIDE, TmrcaReportData, write_tmrca_posterior_tsv};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

// ---- reference implementation (ac53c53, src/hmm.rs) ------------------------

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

#[allow(clippy::too_many_arguments)]
fn forward_fill_row(
    pi: &[f64],
    a_row: &[f64],
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
            matvec_row_major(src, a_row, tmp);
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

#[allow(clippy::too_many_arguments)]
fn reference_tmrca(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    tmrca_years: &[f64],
    out_path: &Path,
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

    let pb_prep: Option<indicatif::ProgressBar> = None;

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
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
        )?;
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);
        if let Some(pb) = &pb_prep {
            pb.inc(s_max as u64);
        }
    }
    if let Some(pb) = pb_prep {
        pb.finish_with_message("TMRCA prep done");
    }

    let pb_back: Option<indicatif::ProgressBar> = None;

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
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
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

    let pb_out: Option<indicatif::ProgressBar> = None;

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
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
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

            if global_bin.is_multiple_of(stride) {
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

// ---- inputs -----------------------------------------------------------------

const K: usize = CHECKPOINT_STRIDE;

struct Input {
    model: PsmcModel,
    rows: Vec<Vec<u8>>,
    row_starts: Vec<bool>,
    years: Vec<f64>,
}

fn input(seed: u64) -> Input {
    let mut rng = SmallRng::seed_from_u64(seed);
    let theta = rng.gen_range(1e-3..2e-2);
    let mut model = PsmcModel::new(
        rng.gen_range(10.0..20.0),
        22,
        theta,
        theta / rng.gen_range(2.0..6.0),
        2.5e-8,
        Some("4+5*3+4".to_string()),
    )
    .unwrap();
    for l in model.lam.iter_mut() {
        *l = rng.gen_range(0.3..3.0);
    }
    model.param_recalculate().unwrap();
    let t = model.compute_t(0.1);
    let n0 = model.theta / (4.0 * model.mu * 100.0);
    let years = (0..=model.n_steps)
        .map(|k| t[k] * 2.0 * n0 * 25.0)
        .collect();

    let mut rows = Vec::new();
    let mut row_starts = Vec::new();
    for (i, len) in [3 * K + 5, 1, K - 1, K, K + 1, 2000]
        .into_iter()
        .enumerate()
    {
        let mut seq: Vec<u8> = (0..len)
            .map(|_| u8::from(rng.gen_bool(theta * 2.0)))
            .collect();
        if len > 100 {
            let start = rng.gen_range(0..len - 50);
            seq[start..start + 50].fill(2);
        }
        if i % 2 == 0 && len > K + 2 {
            // Chained rows, one cut just past a segment boundary.
            let cut = K + 1;
            rows.push(seq[..cut].to_vec());
            row_starts.push(true);
            rows.push(seq[cut..].to_vec());
            row_starts.push(false);
        } else {
            rows.push(seq);
            row_starts.push(true);
        }
    }
    Input {
        model,
        rows,
        row_starts,
        years,
    }
}

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("psmc_rs_estep_tmrca_{}_{name}", std::process::id()))
}

fn write_new(inp: &Input, path: &Path) -> TmrcaReportData {
    write_tmrca_posterior_tsv(
        inp.model.prior_matrix(),
        inp.model.transition_matrix(),
        inp.model.emission_matrix(),
        &inp.rows,
        &inp.row_starts,
        &inp.years,
        path,
        false,
        500,
    )
    .unwrap()
}

fn read_text(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap();
    let mut text = String::new();
    if bytes.starts_with(&[0x1f, 0x8b]) {
        MultiGzDecoder::new(&bytes[..])
            .read_to_string(&mut text)
            .unwrap();
    } else {
        text = String::from_utf8(bytes).unwrap();
    }
    text
}

fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        0.0
    } else {
        (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
    }
}

fn assert_points_close(got: &[[f64; 2]], want: &[[f64; 2]], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g[0], w[0], "{what}: bin");
        assert!(rel(g[1], w[1]) <= tol, "{what}: {g:?} vs {w:?}");
    }
}

// ---- tests ------------------------------------------------------------------

#[test]
fn matches_reference_implementation() {
    for seed in 0..3 {
        let inp = input(seed);
        let (p_new, p_ref) = (tmp_path("new.tsv"), tmp_path("ref.tsv"));
        let got = write_new(&inp, &p_new);
        let want = reference_tmrca(
            inp.model.prior_matrix(),
            inp.model.transition_matrix(),
            inp.model.emission_matrix(),
            &inp.rows,
            &inp.row_starts,
            &inp.years,
            &p_ref,
            500,
        )
        .unwrap();
        let (t_new, t_ref) = (read_text(&p_new), read_text(&p_ref));
        let _ = std::fs::remove_file(&p_new);
        let _ = std::fs::remove_file(&p_ref);

        let (l_new, l_ref): (Vec<&str>, Vec<&str>) =
            (t_new.lines().collect(), t_ref.lines().collect());
        assert_eq!(l_new.len(), l_ref.len(), "seed {seed}: line count");
        assert_eq!(l_new[0], l_ref[0], "header");
        for (a, b) in l_new.iter().zip(&l_ref).skip(1) {
            let (ca, cb): (Vec<&str>, Vec<&str>) =
                (a.split('\t').collect(), b.split('\t').collect());
            assert_eq!(ca.len(), 11);
            // seq_id, seq_bin, global_bin, obs, map_state, MAP and quantile
            // years are discrete; mean, pmax and entropy are printed to 9
            // significant digits, so allow a difference in the last digit.
            for col in [0, 1, 2, 3, 4, 5, 7, 8] {
                assert_eq!(ca[col], cb[col], "seed {seed}: column {col}\n{a}\n{b}");
            }
            for col in [6, 9, 10] {
                let (x, y): (f64, f64) = (ca[col].parse().unwrap(), cb[col].parse().unwrap());
                assert!(
                    rel(x, y) <= 1e-7 || (x - y).abs() < 1e-12,
                    "seed {seed}: column {col}\n{a}\n{b}"
                );
            }
        }

        assert_eq!(got.total_sites, want.total_sites);
        assert_eq!(got.seq_count, want.seq_count);
        assert_eq!(got.state_years, want.state_years);
        assert_points_close(&got.sampled_mean, &want.sampled_mean, 1e-12, "mean");
        assert_points_close(&got.sampled_map, &want.sampled_map, 0.0, "map");
        assert_points_close(&got.sampled_lo, &want.sampled_lo, 0.0, "q025");
        assert_points_close(&got.sampled_hi, &want.sampled_hi, 0.0, "q975");
        for (g, w) in got.state_mass.iter().zip(&want.state_mass) {
            assert!(rel(*g, *w) <= 1e-10, "state mass {g} vs {w}");
        }
    }
}

#[test]
fn gz_path_writes_gzip_with_the_same_content() {
    let inp = input(7);
    let (plain, gz) = (tmp_path("out.tsv"), tmp_path("out.tsv.gz"));
    let a = write_new(&inp, &plain);
    let b = write_new(&inp, &gz);
    let gz_bytes = std::fs::read(&gz).unwrap();
    let (t_plain, t_gz) = (read_text(&plain), read_text(&gz));
    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&gz);
    assert!(gz_bytes.starts_with(&[0x1f, 0x8b]), "not gzip");
    assert_eq!(t_gz, t_plain);
    assert_eq!(a.state_mass, b.state_mass);
    assert_eq!(a.sampled_mean, b.sampled_mean);
}

#[test]
fn output_does_not_depend_on_thread_count() {
    let inp = input(11);
    let mut outputs = Vec::new();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let path = tmp_path(&format!("threads{threads}.tsv"));
        let data = pool.install(|| write_new(&inp, &path));
        outputs.push((std::fs::read(&path).unwrap(), data));
        let _ = std::fs::remove_file(&path);
    }
    assert_eq!(outputs[0].0, outputs[1].0);
    assert_eq!(outputs[0].1.state_mass, outputs[1].1.state_mass);
    assert_eq!(outputs[0].1.sampled_mean, outputs[1].1.sampled_mean);
}
