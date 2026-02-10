use anyhow::{Context, Result, bail};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::io::Observations;
use crate::model::PsmcModel;

#[derive(Debug, Clone, Serialize)]
pub struct BootstrapCurvePoint {
    pub x_years: f64,
    pub ne_main: f64,
    pub ne_q025: f64,
    pub ne_q500: f64,
    pub ne_q975: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BootstrapSummary {
    pub n_replicates: usize,
    pub block_size: usize,
    pub seed: u64,
    pub points: Vec<BootstrapCurvePoint>,
}

pub fn default_bootstrap_dir(output_json: &Path) -> PathBuf {
    let stem = output_json
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("psmc");
    output_json.with_file_name(format!("{stem}.bootstrap"))
}

pub fn replicate_json_path(dir: &Path, replicate_index_1based: usize) -> PathBuf {
    dir.join(format!("replicate_{replicate_index_1based:03}.json"))
}

pub fn new_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

pub fn rows_to_sequences(rows: &[Vec<u8>], row_starts: &[bool]) -> Result<Vec<Vec<u8>>> {
    if rows.is_empty() {
        bail!("cannot build sequences from empty rows");
    }
    if row_starts.len() != rows.len() {
        bail!(
            "row_starts length {} does not match rows {}",
            row_starts.len(),
            rows.len()
        );
    }
    if !row_starts[0] {
        bail!("row_starts[0] must be true");
    }

    let mut seqs = Vec::<Vec<u8>>::new();
    let mut current = Vec::<u8>::new();
    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() {
            bail!("row {i} is empty");
        }
        if row_starts[i] {
            if !current.is_empty() {
                seqs.push(current);
                current = Vec::new();
            }
        } else if current.is_empty() {
            bail!("row {i} continues a sequence but no previous sequence exists");
        }
        current.extend_from_slice(row);
    }
    if !current.is_empty() {
        seqs.push(current);
    }
    if seqs.is_empty() {
        bail!("no sequences reconstructed from rows");
    }
    Ok(seqs)
}

pub fn sequences_to_observations(
    seqs: &[Vec<u8>],
    batch_size: Option<usize>,
) -> Result<Observations> {
    if seqs.is_empty() {
        bail!("sequences are empty");
    }
    if let Some(batch) = batch_size
        && batch == 0
    {
        bail!("batch_size must be > 0");
    }

    let mut rows = Vec::<Vec<u8>>::new();
    let mut row_starts = Vec::<bool>::new();
    for seq in seqs {
        if seq.is_empty() {
            continue;
        }
        match batch_size {
            None => {
                rows.push(seq.clone());
                row_starts.push(true);
            }
            Some(batch) => {
                for (i, chunk) in seq.chunks(batch).enumerate() {
                    rows.push(chunk.to_vec());
                    row_starts.push(i == 0);
                }
            }
        }
    }
    if rows.is_empty() {
        bail!("no rows produced from sequences");
    }
    Ok(Observations { rows, row_starts })
}

pub fn sample_sequences_block_bootstrap(
    seqs: &[Vec<u8>],
    block_size: usize,
    rng: &mut SmallRng,
) -> Result<Vec<Vec<u8>>> {
    if block_size == 0 {
        bail!("bootstrap block_size must be > 0");
    }
    if seqs.is_empty() {
        bail!("input sequences are empty");
    }

    let mut out = Vec::<Vec<u8>>::with_capacity(seqs.len());
    for (seq_id, seq) in seqs.iter().enumerate() {
        if seq.is_empty() {
            bail!("sequence {seq_id} is empty");
        }
        let blocks: Vec<&[u8]> = seq.chunks(block_size).collect();
        if blocks.is_empty() {
            bail!("sequence {seq_id} produced no bootstrap blocks");
        }
        let draw_n = seq.len().div_ceil(block_size);
        let mut sampled = Vec::<u8>::with_capacity(seq.len() + block_size);
        for _ in 0..draw_n {
            let idx = rng.gen_range(0..blocks.len());
            sampled.extend_from_slice(blocks[idx]);
        }
        sampled.truncate(seq.len());
        out.push(sampled);
    }
    Ok(out)
}

fn model_curve(model: &PsmcModel, bin_size: f64, gen_years: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    let lam_full = model.map_lam(&model.lam)?;
    let t = model.compute_t(0.1);
    let n0 = model.theta / (4.0 * model.mu * bin_size);

    let mut xs = Vec::<f64>::with_capacity(model.n_steps + 1);
    let mut ys = Vec::<f64>::with_capacity(model.n_steps + 1);
    for k in 0..=model.n_steps {
        let x = t[k] * 2.0 * n0 * gen_years;
        let y = lam_full[k] * n0;
        if x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.is_empty() {
        bail!("model curve has no valid points");
    }
    Ok((xs, ys))
}

#[inline]
fn step_lookup(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert!(!xs.is_empty() && xs.len() == ys.len());
    let pos = xs.partition_point(|v| *v <= x);
    if pos == 0 {
        ys[0]
    } else {
        ys[(pos - 1).min(ys.len() - 1)]
    }
}

fn quantile(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let qq = q.clamp(0.0, 1.0);
    let n = vals.len();
    if n == 1 {
        return vals[0];
    }
    let pos = qq * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        vals[lo]
    } else {
        let w = pos - lo as f64;
        vals[lo] * (1.0 - w) + vals[hi] * w
    }
}

pub fn summarize_bootstrap_models(
    main_model: &PsmcModel,
    bootstrap_models: &[PsmcModel],
    bin_size: f64,
    gen_years: f64,
    block_size: usize,
    seed: u64,
) -> Result<BootstrapSummary> {
    if bootstrap_models.is_empty() {
        bail!("bootstrap model list is empty");
    }
    let (x_main, y_main) = model_curve(main_model, bin_size, gen_years)?;
    let n_points = x_main.len();
    let n_boot = bootstrap_models.len();

    let mut sample_mat = vec![vec![0.0f64; n_boot]; n_points];
    for (r, model) in bootstrap_models.iter().enumerate() {
        let (x_rep, y_rep) = model_curve(model, bin_size, gen_years)?;
        for i in 0..n_points {
            sample_mat[i][r] = step_lookup(&x_rep, &y_rep, x_main[i]);
        }
    }

    let mut points = Vec::<BootstrapCurvePoint>::with_capacity(n_points);
    for i in 0..n_points {
        let mut qbuf = sample_mat[i].clone();
        let q025 = quantile(&mut qbuf, 0.025);
        let q500 = quantile(&mut qbuf, 0.5);
        let q975 = quantile(&mut qbuf, 0.975);
        points.push(BootstrapCurvePoint {
            x_years: x_main[i],
            ne_main: y_main[i],
            ne_q025: q025,
            ne_q500: q500,
            ne_q975: q975,
        });
    }

    Ok(BootstrapSummary {
        n_replicates: n_boot,
        block_size,
        seed,
        points,
    })
}

pub fn write_bootstrap_summary_tsv(path: &Path, summary: &BootstrapSummary) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create bootstrap summary dir {parent:?}"))?;
    }
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let mut w = BufWriter::new(file);
    w.write_all(b"x_years\tne_main\tne_q025\tne_q500\tne_q975\n")?;
    for p in &summary.points {
        writeln!(
            w,
            "{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
            p.x_years, p.ne_main, p.ne_q025, p.ne_q500, p.ne_q975
        )?;
    }
    w.flush()?;
    Ok(())
}

pub fn write_bootstrap_summary_json(path: &Path, summary: &BootstrapSummary) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create bootstrap summary dir {parent:?}"))?;
    }
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, summary)
        .with_context(|| format!("failed to write {path:?}"))?;
    Ok(())
}
