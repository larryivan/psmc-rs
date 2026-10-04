//! Posterior TMRCA decoding written as a per-site TSV.

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use ndarray::Array2;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::fb::{self, Ctx, Progress, SeqObs, Workspace};
use super::kernel::Kernel;
use super::sched::Plan;
use super::{AlphaCacheBudget, CHECKPOINT_STRIDE, TmrcaReportData, validate};
use crate::progress;

/// Segments decoded and formatted per parallel task (and per gzip member).
const SEGMENTS_PER_BLOCK: usize = 8;
/// Blocks held in memory at once while writing in order.
const BLOCKS_PER_WINDOW: usize = 64;

const HEADER: &str = "seq_id\tseq_bin\tglobal_bin\tobs\tmap_state\ttmrca_map_years\ttmrca_mean_years\ttmrca_q025_years\ttmrca_q975_years\tpmax\tentropy\n";

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

fn obs_char(class: u8) -> char {
    match class {
        0 => 'T',
        1 => 'K',
        _ => 'N',
    }
}

/// Output sink: plain text, or gzip when the path ends in `.gz`.
enum Sink {
    Plain(BufWriter<File>),
    /// Blocks are compressed in parallel into independent gzip members and
    /// concatenated (a valid multi-member gzip file, like bgzip output).
    Gzip(BufWriter<File>),
}

impl Sink {
    fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
        let w = BufWriter::new(file);
        let gz = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gz"));
        Ok(if gz { Self::Gzip(w) } else { Self::Plain(w) })
    }

    fn is_gzip(&self) -> bool {
        matches!(self, Self::Gzip(_))
    }

    /// Bytes ready to append: the text itself, or a gzip member of it.
    fn encode(gzip: bool, text: String) -> Result<Vec<u8>> {
        if !gzip {
            return Ok(text.into_bytes());
        }
        let mut enc = GzEncoder::new(Vec::with_capacity(text.len() / 4), Compression::default());
        enc.write_all(text.as_bytes())?;
        Ok(enc.finish()?)
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            Self::Plain(w) | Self::Gzip(w) => w.write_all(bytes)?,
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        match self {
            Self::Plain(mut w) | Self::Gzip(mut w) => w.flush()?,
        }
        Ok(())
    }
}

struct BlockOut {
    bytes: Vec<u8>,
    state_mass: Vec<f64>,
    samples: Vec<[f64; 5]>,
}

struct SeqInfo<'a> {
    seq: SeqObs<'a>,
    seq_id: usize,
    global_offset: usize,
}

/// Decode segments `segs` of one sequence: posterior summaries per site,
/// formatted as TSV lines.
#[allow(clippy::too_many_arguments)]
fn decode_block(
    k: &Kernel,
    info: &SeqInfo<'_>,
    ckpt: &fb::DecodeCheckpoints,
    segs: std::ops::Range<usize>,
    years: &[f64],
    stride: usize,
    gzip: bool,
    scratch: &mut (Workspace, Vec<f64>, Vec<u8>),
) -> Result<BlockOut> {
    let (ws, post, obs) = scratch;
    let n = k.n_states();
    let np = k.np();
    let mut text = String::with_capacity(segs.len() * CHECKPOINT_STRIDE * 140);
    let mut state_mass = vec![0.0f64; n];
    let mut samples = Vec::new();
    for s in segs {
        fb::segment_posteriors(k, &info.seq, ckpt, s, ws, post)?;
        fb::segment_obs(&info.seq, s, obs);
        let (start, _) = info.seq.segment(s);
        for (t, &o) in obs.iter().enumerate() {
            let p = &post[t * np..t * np + n];
            let mut mean = 0.0f64;
            let mut max_p = -1.0f64;
            let mut max_k = 0usize;
            let mut entropy = 0.0f64;
            for (kk, &pk) in p.iter().enumerate() {
                state_mass[kk] += pk;
                mean += pk * years[kk];
                if pk > max_p {
                    max_p = pk;
                    max_k = kk;
                }
                if pk > 0.0 {
                    entropy -= pk * pk.ln();
                }
            }
            let map_years = years[max_k];
            let q025 = quantile_from_probs(p, years, 0.025);
            let q975 = quantile_from_probs(p, years, 0.975);
            let seq_bin = start + t;
            let global_bin = info.global_offset + seq_bin;
            writeln!(
                text,
                "{}\t{}\t{}\t{}\t{}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
                info.seq_id,
                seq_bin,
                global_bin,
                obs_char(o),
                max_k,
                map_years,
                mean,
                q025,
                q975,
                max_p,
                entropy,
            )
            .expect("writing to a String cannot fail");
            if global_bin.is_multiple_of(stride) {
                samples.push([global_bin as f64, mean, map_years, q025, q975]);
            }
        }
    }
    Ok(BlockOut {
        bytes: Sink::encode(gzip, text)?,
        state_mass,
        samples,
    })
}

/// Posterior decoding of the time to the most recent common ancestor, one
/// TSV line per bin (gzip-compressed when `out_path` ends in `.gz`).
///
/// `tmrca_years[k]` is the time assigned to state `k`. Memory is bounded:
/// each sequence keeps forward and backward checkpoints every
/// [`CHECKPOINT_STRIDE`] sites, and blocks of segments are decoded in parallel
/// and written in order. Returns summaries for the HTML report (at most about
/// `max_points` sampled bins; 0 samples every bin).
#[allow(clippy::too_many_arguments)]
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
    validate(pi, a, em, rows, row_starts)?;
    let n_states = pi.len();
    if tmrca_years.len() != n_states {
        bail!(
            "tmrca_years length {} != n_states {}",
            tmrca_years.len(),
            n_states
        );
    }
    let kernel = Kernel::new(pi, a, em);
    let np = kernel.np();
    let plan = Plan::new(rows, row_starts);
    let total_sites = plan.total_sites as usize;
    let stride = if max_points == 0 {
        1
    } else {
        total_sites.div_ceil(max_points).max(1)
    };
    let progress = Progress::default();
    let budget = AlphaCacheBudget::new(0);
    let ctx = Ctx {
        cancel: None,
        progress: &progress,
        budget: &budget,
    };
    let pb = (progress_enabled && total_sites > 0)
        .then(|| progress::bar_raw(total_sites as u64, "TMRCA", "posterior"));

    let mut sink = Sink::create(out_path)?;
    let gzip = sink.is_gzip();
    sink.write(&Sink::encode(gzip, HEADER.to_string())?)?;

    let mut state_mass = vec![0.0f64; n_states];
    let mut samples: Vec<[f64; 5]> = Vec::new();
    let mut global_offset = 0usize;
    for (seq_id, seq_rows) in plan.seqs.iter().enumerate() {
        let info = SeqInfo {
            seq: SeqObs::new(&rows[seq_rows.clone()]),
            seq_id,
            global_offset,
        };
        let ckpt = fb::decode_checkpoints(&kernel, &info.seq, &ctx)?;
        let n_seg = info.seq.n_segments();
        let n_blocks = n_seg.div_ceil(SEGMENTS_PER_BLOCK);
        for w0 in (0..n_blocks).step_by(BLOCKS_PER_WINDOW) {
            let w1 = (w0 + BLOCKS_PER_WINDOW).min(n_blocks);
            let outs = (w0..w1)
                .into_par_iter()
                .map_init(
                    || {
                        (
                            Workspace::new(np),
                            vec![0.0; CHECKPOINT_STRIDE * np],
                            Vec::new(),
                        )
                    },
                    |scratch, b| {
                        let segs =
                            b * SEGMENTS_PER_BLOCK..((b + 1) * SEGMENTS_PER_BLOCK).min(n_seg);
                        decode_block(
                            &kernel,
                            &info,
                            &ckpt,
                            segs,
                            tmrca_years,
                            stride,
                            gzip,
                            scratch,
                        )
                    },
                )
                .collect::<Result<Vec<_>>>()?;
            for out in outs {
                sink.write(&out.bytes)?;
                for (d, s) in state_mass.iter_mut().zip(&out.state_mass) {
                    *d += *s;
                }
                samples.extend_from_slice(&out.samples);
            }
            if let Some(pb) = &pb {
                let done = (w1 * SEGMENTS_PER_BLOCK * CHECKPOINT_STRIDE).min(info.seq.len());
                pb.set_position((global_offset + done) as u64);
            }
        }
        global_offset += info.seq.len();
    }
    sink.finish()?;
    if let Some(pb) = pb {
        pb.finish_with_message("TMRCA done");
    }

    Ok(TmrcaReportData {
        sampled_mean: samples.iter().map(|s| [s[0], s[1]]).collect(),
        sampled_map: samples.iter().map(|s| [s[0], s[2]]).collect(),
        sampled_lo: samples.iter().map(|s| [s[0], s[3]]).collect(),
        sampled_hi: samples.iter().map(|s| [s[0], s[4]]).collect(),
        state_mass,
        state_years: tmrca_years.to_vec(),
        total_sites,
        seq_count: plan.seqs.len(),
    })
}
