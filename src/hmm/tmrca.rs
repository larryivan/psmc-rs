//! Posterior TMRCA decoding written as a per-site TSV.
//!
//! The output is produced in windows of work units, in input order. A unit
//! is either a block of segments of a long sequence, decoded from that
//! sequence's checkpoints, or a group of consecutive short sequences, each
//! decoded whole. Within a window, the checkpoint passes of the sequences
//! that need them run in parallel, then all units are decoded in parallel;
//! the previous window is written meanwhile. Units depend only on the input,
//! and every site's line is computed the same way in either kind of unit, so
//! the output does not depend on the thread count.

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use indicatif::ProgressBar;
use ndarray::Array2;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;

use super::fb::{self, Ctx, DecodeCheckpoints, Progress, SeqObs, Workspace};
use super::kernel::Kernel;
use super::sched::Plan;
use super::{AlphaCacheBudget, CHECKPOINT_STRIDE, TmrcaReportData, validate};
use crate::progress;

/// Segments of a long sequence decoded per work unit.
const SEGMENTS_PER_BLOCK: usize = 8;
/// Most sites per work unit (and per gzip member). Sequences no longer than
/// this are short: decoded whole from their full forward rows, several to a unit.
const UNIT_SITES: usize = SEGMENTS_PER_BLOCK * CHECKPOINT_STRIDE;
/// Work units per window: the units decoded in parallel before being written.
const UNITS_PER_WINDOW: usize = 64;

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
    /// Units are compressed in parallel into independent gzip members and
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

/// A piece of the output decoded by one task.
enum Unit {
    /// Consecutive short sequences, at most [`UNIT_SITES`] sites in total.
    Short { seqs: Range<usize> },
    /// Segments `segs` of a longer sequence, decoded from its checkpoints
    /// (entry `ckpt` of the window's checkpoint list).
    Long {
        seq: usize,
        ckpt: usize,
        segs: Range<usize>,
    },
}

/// Where the next window starts: segment `seg` of sequence `seq`.
#[derive(Debug, Default)]
struct Cursor {
    seq: usize,
    seg: usize,
}

struct Window {
    units: Vec<Unit>,
    /// The long sequences the `Long` units belong to.
    ckpt_seqs: Vec<usize>,
    /// Whether the last of `ckpt_seqs` continues into the next window.
    continues: bool,
    /// Sites output once this window is written.
    end_site: usize,
}

/// The next window of at most [`UNITS_PER_WINDOW`] units, or `None` at the end.
/// `seq_start[i]` is the first site of sequence `i`; the last entry is the total.
fn next_window(seq_start: &[usize], cur: &mut Cursor) -> Option<Window> {
    let n_seqs = seq_start.len() - 1;
    let mut units = Vec::new();
    let mut ckpt_seqs = Vec::new();
    let mut group_sites = 0;
    while cur.seq < n_seqs {
        let len = seq_start[cur.seq + 1] - seq_start[cur.seq];
        if len <= UNIT_SITES {
            if let Some(Unit::Short { seqs }) = units.last_mut()
                && group_sites + len <= UNIT_SITES
            {
                seqs.end += 1;
                group_sites += len;
            } else if units.len() < UNITS_PER_WINDOW {
                units.push(Unit::Short {
                    seqs: cur.seq..cur.seq + 1,
                });
                group_sites = len;
            } else {
                break;
            }
            cur.seq += 1;
        } else {
            if units.len() == UNITS_PER_WINDOW {
                break;
            }
            if ckpt_seqs.last() != Some(&cur.seq) {
                ckpt_seqs.push(cur.seq);
            }
            let n_seg = len.div_ceil(CHECKPOINT_STRIDE);
            let end = (cur.seg + SEGMENTS_PER_BLOCK).min(n_seg);
            units.push(Unit::Long {
                seq: cur.seq,
                ckpt: ckpt_seqs.len() - 1,
                segs: cur.seg..end,
            });
            if end == n_seg {
                cur.seq += 1;
                cur.seg = 0;
            } else {
                cur.seg = end;
            }
        }
    }
    if units.is_empty() {
        return None;
    }
    let end_site = seq_start[cur.seq] + cur.seg * CHECKPOINT_STRIDE;
    Some(Window {
        units,
        ckpt_seqs,
        continues: cur.seg > 0,
        end_site,
    })
}

/// The TSV lines and report summaries of one unit.
struct UnitOut {
    bytes: Vec<u8>,
    state_mass: Vec<f64>,
    samples: Vec<[f64; 5]>,
}

/// Formats decoded sites as TSV lines and accumulates the summaries of one unit.
struct UnitWriter<'a> {
    years: &'a [f64],
    stride: usize,
    text: String,
    state_mass: Vec<f64>,
    samples: Vec<[f64; 5]>,
}

impl<'a> UnitWriter<'a> {
    fn new(years: &'a [f64], stride: usize, sites: usize) -> Self {
        Self {
            years,
            stride,
            text: String::with_capacity(sites * 140),
            state_mass: vec![0.0; years.len()],
            samples: Vec::new(),
        }
    }

    /// Lines for consecutive sites of sequence `seq_id`, the first of which
    /// is bin `seq_bin` of the sequence and `global_bin` of the input, with
    /// observation classes `obs` and posteriors `post` (`np` values per site).
    fn sites(
        &mut self,
        seq_id: usize,
        seq_bin: usize,
        global_bin: usize,
        obs: &[u8],
        post: &[f64],
        np: usize,
    ) {
        let years = self.years;
        let n = years.len();
        for (t, &o) in obs.iter().enumerate() {
            let p = &post[t * np..t * np + n];
            let mut mean = 0.0f64;
            let mut max_p = -1.0f64;
            let mut max_k = 0usize;
            let mut entropy = 0.0f64;
            for (kk, &pk) in p.iter().enumerate() {
                self.state_mass[kk] += pk;
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
            let global_bin = global_bin + t;
            writeln!(
                self.text,
                "{}\t{}\t{}\t{}\t{}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
                seq_id,
                seq_bin + t,
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
            if global_bin.is_multiple_of(self.stride) {
                self.samples
                    .push([global_bin as f64, mean, map_years, q025, q975]);
            }
        }
    }

    fn finish(self, gzip: bool) -> Result<UnitOut> {
        Ok(UnitOut {
            bytes: Sink::encode(gzip, self.text)?,
            state_mass: self.state_mass,
            samples: self.samples,
        })
    }
}

/// Everything the parallel decoding tasks share.
struct Decoder<'a> {
    kernel: Kernel,
    rows: &'a [Vec<u8>],
    plan: Plan,
    /// First site of each sequence in the input; the last entry is the total.
    seq_start: Vec<usize>,
    years: &'a [f64],
    stride: usize,
    gzip: bool,
    ctx: Ctx<'a>,
    /// Workspaces of finished tasks, reused by later ones.
    free: Mutex<Vec<Workspace>>,
}

impl Decoder<'_> {
    fn seq(&self, i: usize) -> SeqObs<'_> {
        SeqObs::new(&self.rows[self.plan.seqs[i].clone()])
    }

    fn with_workspace<R>(&self, f: impl FnOnce(&mut Workspace) -> R) -> R {
        let pop = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let mut ws = pop.unwrap_or_else(|| Workspace::new(self.kernel.np()));
        let r = f(&mut ws);
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
        r
    }

    fn decode_unit(&self, unit: &Unit, ckpts: &[DecodeCheckpoints]) -> Result<UnitOut> {
        let (k, np) = (&self.kernel, self.kernel.np());
        let sites = match unit {
            Unit::Short { seqs } => self.seq_start[seqs.end] - self.seq_start[seqs.start],
            Unit::Long { segs, .. } => segs.len() * CHECKPOINT_STRIDE,
        };
        let mut out = UnitWriter::new(self.years, self.stride, sites);
        self.with_workspace(|ws| -> Result<()> {
            match unit {
                Unit::Short { seqs } => {
                    for i in seqs.clone() {
                        let (obs, post) = fb::sequence_posteriors(k, &self.seq(i), ws)?;
                        out.sites(i, 0, self.seq_start[i], obs, post, np);
                    }
                }
                Unit::Long { seq, ckpt, segs } => {
                    let seq_obs = self.seq(*seq);
                    for s in segs.clone() {
                        let (obs, post) =
                            fb::segment_posteriors(k, &seq_obs, &ckpts[*ckpt], s, ws)?;
                        let start = s * CHECKPOINT_STRIDE;
                        out.sites(*seq, start, self.seq_start[*seq] + start, obs, post, np);
                    }
                }
            }
            Ok(())
        })?;
        out.finish(self.gzip)
    }

    /// Decode the units of `window` in parallel, after computing the
    /// checkpoints they need (in parallel too). `carried` holds the
    /// checkpoints of a sequence continued from the previous window. Returns
    /// the units' output and the checkpoints of the sequence that continues
    /// into the next window, if any.
    #[allow(clippy::type_complexity)]
    fn decode_window(
        &self,
        window: &Window,
        carried: Option<(usize, DecodeCheckpoints)>,
    ) -> Result<(Vec<UnitOut>, Option<(usize, DecodeCheckpoints)>)> {
        let mut ckpts: Vec<Option<DecodeCheckpoints>> =
            window.ckpt_seqs.iter().map(|_| None).collect();
        if let Some((seq, c)) = carried
            && window.ckpt_seqs.first() == Some(&seq)
        {
            ckpts[0] = Some(c);
        }
        ckpts
            .par_iter_mut()
            .zip(&window.ckpt_seqs)
            .try_for_each(|(slot, &seq)| -> Result<()> {
                if slot.is_none() {
                    *slot = Some(fb::decode_checkpoints(
                        &self.kernel,
                        &self.seq(seq),
                        &self.ctx,
                    )?);
                }
                Ok(())
            })?;
        let mut ckpts: Vec<DecodeCheckpoints> = ckpts
            .into_iter()
            .map(|c| c.expect("every checkpoint was computed"))
            .collect();
        let outs = window
            .units
            .par_iter()
            .map(|u| self.decode_unit(u, &ckpts))
            .collect::<Result<Vec<_>>>()?;
        let next = match (window.continues, ckpts.pop()) {
            (true, Some(c)) => Some((*window.ckpt_seqs.last().expect("non-empty"), c)),
            _ => None,
        };
        Ok((outs, next))
    }
}

/// The file being written and the report summaries, in output order.
struct Output {
    sink: Sink,
    state_mass: Vec<f64>,
    samples: Vec<[f64; 5]>,
    pb: Option<ProgressBar>,
}

impl Output {
    fn append(&mut self, outs: Vec<UnitOut>, end_site: usize) -> Result<()> {
        for out in outs {
            self.sink.write(&out.bytes)?;
            for (d, s) in self.state_mass.iter_mut().zip(&out.state_mass) {
                *d += *s;
            }
            self.samples.extend_from_slice(&out.samples);
        }
        if let Some(pb) = &self.pb {
            pb.set_position(end_site as u64);
        }
        Ok(())
    }
}

/// Posterior decoding of the time to the most recent common ancestor, one
/// TSV line per bin (gzip-compressed when `out_path` ends in `.gz`).
///
/// `tmrca_years[k]` is the time assigned to state `k`. Memory is bounded:
/// sequences longer than a work unit keep forward and backward checkpoints
/// every [`CHECKPOINT_STRIDE`] sites, and units are decoded in parallel,
/// across sequences as well as within them, and written in order. Returns
/// summaries for the HTML report (at most about `max_points` sampled bins;
/// 0 samples every bin).
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
    let plan = Plan::new(rows, row_starts);
    let total_sites = plan.total_sites as usize;
    let mut seq_start = Vec::with_capacity(plan.seqs.len() + 1);
    seq_start.push(0);
    for r in &plan.seqs {
        let len: usize = rows[r.clone()].iter().map(Vec::len).sum();
        seq_start.push(seq_start.last().copied().unwrap_or(0) + len);
    }
    let stride = if max_points == 0 {
        1
    } else {
        total_sites.div_ceil(max_points).max(1)
    };
    let progress = Progress::default();
    let budget = AlphaCacheBudget::new(0);
    let mut sink = Sink::create(out_path)?;
    let gzip = sink.is_gzip();
    sink.write(&Sink::encode(gzip, HEADER.to_string())?)?;
    let dec = Decoder {
        kernel: Kernel::new(pi, a, em),
        rows,
        plan,
        seq_start,
        years: tmrca_years,
        stride,
        gzip,
        ctx: Ctx {
            cancel: None,
            progress: &progress,
            budget: &budget,
        },
        free: Mutex::new(Vec::new()),
    };
    let mut output = Output {
        sink,
        state_mass: vec![0.0; n_states],
        samples: Vec::new(),
        pb: (progress_enabled && total_sites > 0)
            .then(|| progress::bar_raw(total_sites as u64, "TMRCA", "posterior")),
    };

    let mut cursor = Cursor::default();
    let mut carried = None;
    let mut decoded: Option<(Vec<UnitOut>, usize)> = None;
    loop {
        let window = next_window(&dec.seq_start, &mut cursor);
        let (written, next) = rayon::join(
            || match decoded.take() {
                Some((outs, end_site)) => output.append(outs, end_site),
                None => Ok(()),
            },
            || -> Result<_> {
                let Some(w) = window else {
                    return Ok(None);
                };
                let (outs, keep) = dec.decode_window(&w, carried.take())?;
                carried = keep;
                Ok(Some((outs, w.end_site)))
            },
        );
        written?;
        match next? {
            Some(d) => decoded = Some(d),
            None => break,
        }
    }
    let Output {
        sink,
        state_mass,
        samples,
        pb,
    } = output;
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
        seq_count: dec.plan.seqs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: usize = CHECKPOINT_STRIDE;

    fn starts(lens: &[usize]) -> Vec<usize> {
        let mut s = vec![0];
        for &l in lens {
            s.push(s.last().unwrap() + l);
        }
        s
    }

    /// Walks every window and checks that the units cover each site once, in
    /// order, within the size limits. Returns the number of units per window.
    fn check_windows(lens: &[usize]) -> Vec<usize> {
        let seq_start = starts(lens);
        let mut cur = Cursor::default();
        let mut next_site = 0;
        let mut continued: Option<usize> = None;
        let mut sizes = Vec::new();
        while let Some(w) = next_window(&seq_start, &mut cur) {
            assert!(w.units.len() <= UNITS_PER_WINDOW);
            if let Some(seq) = continued {
                assert_eq!(w.ckpt_seqs.first(), Some(&seq), "carried checkpoints");
            }
            for u in &w.units {
                match u {
                    Unit::Short { seqs } => {
                        assert!(!seqs.is_empty());
                        assert_eq!(seq_start[seqs.start], next_site);
                        assert!(seq_start[seqs.end] - seq_start[seqs.start] <= UNIT_SITES);
                        next_site = seq_start[seqs.end];
                    }
                    Unit::Long { seq, ckpt, segs } => {
                        assert!(lens[*seq] > UNIT_SITES);
                        assert_eq!(w.ckpt_seqs[*ckpt], *seq);
                        assert!(!segs.is_empty() && segs.len() <= SEGMENTS_PER_BLOCK);
                        assert_eq!(seq_start[*seq] + segs.start * K, next_site);
                        next_site = (seq_start[*seq] + segs.end * K).min(seq_start[*seq + 1]);
                    }
                }
            }
            assert_eq!(w.end_site, next_site);
            continued = w.continues.then(|| *w.ckpt_seqs.last().unwrap());
            sizes.push(w.units.len());
        }
        assert_eq!(next_site, *seq_start.last().unwrap());
        assert_eq!(continued, None);
        sizes
    }

    #[test]
    fn short_sequences_share_units() {
        // Ten 100-site sequences fit in a unit.
        assert_eq!(check_windows(&[100; 25]), vec![3]);
        // Exactly one unit's worth fits; one more site starts a new unit.
        assert_eq!(check_windows(&[UNIT_SITES / 2; 2]), vec![1]);
        assert_eq!(
            check_windows(&[UNIT_SITES / 2, UNIT_SITES / 2 + 1]),
            vec![2]
        );
        assert_eq!(check_windows(&[1; 3 * UNIT_SITES]), vec![3]);
        // Many short sequences fill several windows.
        let lens: Vec<usize> = (0..5000).map(|i| 50 + (i * 37) % 251).collect();
        let sizes = check_windows(&lens);
        assert!(sizes.len() > 1);
        assert!(
            sizes[..sizes.len() - 1]
                .iter()
                .all(|&s| s == UNITS_PER_WINDOW)
        );
    }

    #[test]
    fn long_sequences_are_split_into_blocks_across_windows() {
        let long = UNITS_PER_WINDOW * UNIT_SITES + 3 * K + 5;
        // 65 blocks: the sequence continues into a second window.
        assert_eq!(check_windows(&[long]), vec![UNITS_PER_WINDOW, 1]);
        assert_eq!(check_windows(&[UNIT_SITES + 1]), vec![2]);
        check_windows(&[
            3,
            UNIT_SITES,
            UNIT_SITES + 1,
            7,
            long,
            1,
            2 * long,
            UNIT_SITES - 1,
            2,
        ]);
    }
}
