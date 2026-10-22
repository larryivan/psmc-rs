//! Posterior TMRCA decoding written as a per-site TSV.
//!
//! The output is produced in windows of work units, in input order. A unit
//! is either a block of segments of a long sequence, decoded from that
//! sequence's checkpoints, or a group of consecutive short sequences, each
//! decoded whole. Within a window, the checkpoint passes of the sequences
//! that need them run in parallel, then all units are decoded in parallel;
//! the previous window is written meanwhile. A sequence longer than a window
//! has its checkpoint passes run earlier instead, alongside the decoding of
//! the output before it. Units depend only on the input, and every site's
//! line is computed the same way in either kind of unit, so the output does
//! not depend on the thread count.

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

use super::fb::{self, Ctx, DecodeCheckpoints, Progress, Workspace};
use super::{AlphaCacheBudget, CHECKPOINT_STRIDE, Pass, TmrcaReportData};
use crate::io::CallableBases;
use crate::progress;

/// Segments of a long sequence decoded per work unit.
const SEGMENTS_PER_BLOCK: usize = 8;
/// Most sites per work unit (and per gzip member). Sequences no longer than
/// this are short: decoded whole from their full forward rows, several to a unit.
const UNIT_SITES: usize = SEGMENTS_PER_BLOCK * CHECKPOINT_STRIDE;
/// Work units per window: the units decoded in parallel before being written.
const UNITS_PER_WINDOW: usize = 64;
/// Sites in a full window. The checkpoint passes of a sequence longer than
/// this run while the output before it is decoded, instead of in its first
/// window, where only two threads would have work.
const WINDOW_SITES: usize = UNITS_PER_WINDOW * UNIT_SITES;

const HEADER: &str = "seq_id\tseq_bin\tglobal_bin\tobs\tmap_state\ttmrca_map_years\ttmrca_mean_years\ttmrca_q025_years\ttmrca_q975_years\tpmax\tentropy\n";

/// The first state at which the cumulative posterior reaches `q` (the last
/// state if rounding keeps the total below `q`).
#[inline]
fn quantile_state(probs: &[f64], q: f64) -> usize {
    let target = q.clamp(0.0, 1.0);
    let mut cdf = 0.0;
    for (k, p) in probs.iter().enumerate() {
        cdf += *p;
        if cdf >= target {
            return k;
        }
    }
    probs.len().saturating_sub(1)
}

fn obs_char(class: u8) -> char {
    match class {
        0 => 'T',
        1 => 'K',
        _ => 'N',
    }
}

/// Appends the decimal digits of `n`.
fn push_usize(out: &mut String, mut n: usize) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    out.push_str(std::str::from_utf8(&buf[i..]).expect("ASCII digits"));
}

/// Powers of ten that are exact in f64.
const POW10: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// The nine significant digits and the decimal exponent of `v` as
/// `{:.8e}` prints them, or `None` where this cheap method cannot be sure.
///
/// `v` is scaled into `[1e8, 1e9)` by one multiplication or division by an
/// exact power of ten. That single rounding is at most 2^-53 relative, less
/// than 1.2e-7 absolute, so rounding the scaled value to an integer gives the
/// correctly rounded digits unless it lies within 1e-6 of a tie. Ties, zero,
/// subnormal, negative and non-finite values, and exponents beyond the exact
/// powers return `None`.
#[inline]
fn sci8_digits(v: f64) -> Option<(u64, i32)> {
    if !(v.is_normal() && v > 0.0) {
        return None;
    }
    let scale = |e10: i32| -> Option<f64> {
        let k = 8 - e10;
        let p = *POW10.get(k.unsigned_abs() as usize)?;
        Some(if k >= 0 { v * p } else { v / p })
    };
    // floor(log10(v)) to within one, from the binary exponent (78913 / 2^18
    // is log10(2) to 6 digits).
    let e2 = ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut e10 = (e2 * 78913) >> 18;
    let mut m = scale(e10)?;
    if m < 1e8 {
        e10 -= 1;
        m = scale(e10)?;
    } else if m >= 1e9 {
        e10 += 1;
        m = scale(e10)?;
    }
    if !(1e8..1e9).contains(&m) {
        return None;
    }
    let floor = m.floor();
    let frac = m - floor;
    if (frac - 0.5).abs() < 1e-6 {
        return None;
    }
    let mut digits = floor as u64 + u64::from(frac > 0.5);
    if digits == 1_000_000_000 {
        digits = 100_000_000;
        e10 += 1;
    }
    Some((digits, e10))
}

/// Appends `v` exactly as `write!(out, "{v:.8e}")` would, usually without
/// the standard formatter (which costs a few hundred ns per value and
/// dominated the cost of a TSV line).
fn push_sci8(out: &mut String, v: f64) {
    let Some((digits, e10)) = sci8_digits(v) else {
        write!(out, "{v:.8e}").expect("writing to a String cannot fail");
        return;
    };
    let mut buf = [0u8; 10];
    let mut d = digits;
    for b in buf[2..].iter_mut().rev() {
        *b = b'0' + (d % 10) as u8;
        d /= 10;
    }
    buf[0] = b'0' + d as u8;
    buf[1] = b'.';
    out.push_str(std::str::from_utf8(&buf).expect("ASCII digits"));
    out.push('e');
    if e10 < 0 {
        out.push('-');
    }
    push_usize(out, e10.unsigned_abs() as usize);
}

/// Output sink: plain text, or gzip when the path ends in `.gz`, in which
/// case units are compressed in parallel into independent gzip members and
/// concatenated (a valid multi-member gzip file, like bgzip output).
struct Sink {
    file: BufWriter<File>,
    gzip: bool,
}

impl Sink {
    fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
        Ok(Self {
            file: BufWriter::new(file),
            gzip: path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gz")),
        })
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
        Ok(self.file.write_all(bytes)?)
    }

    fn finish(mut self) -> Result<()> {
        Ok(self.file.flush()?)
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

/// The first index of each of consecutive pieces of lengths `lens`, then
/// the total.
fn starts(lens: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut s = vec![0];
    for len in lens {
        s.push(s[s.len() - 1] + len);
    }
    s
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

/// The next window of at most [`UNITS_PER_WINDOW`] units from sequences
/// before `stop`, or `None` once the cursor reaches `stop`. `seq_start[i]`
/// is the first site of sequence `i`; the last entry is the total.
fn next_window(seq_start: &[usize], cur: &mut Cursor, stop: usize) -> Option<Window> {
    let mut units = Vec::new();
    let mut ckpt_seqs = Vec::new();
    let mut group_sites = 0;
    while cur.seq < stop {
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

/// The time of each state, as a number and as printed in the TSV.
struct StateYears<'a> {
    values: &'a [f64],
    /// `values[k]` formatted as `{:.8e}`. The MAP and quantile columns are
    /// always one of these, so they are formatted once rather than per line
    /// (float formatting dominates the cost of writing a line).
    text: Vec<String>,
}

impl<'a> StateYears<'a> {
    fn new(values: &'a [f64]) -> Self {
        Self {
            values,
            text: values.iter().map(|y| format!("{y:.8e}")).collect(),
        }
    }
}

/// Formats decoded sites as TSV lines and accumulates the summaries of one unit.
struct UnitWriter<'a> {
    years: &'a StateYears<'a>,
    /// The symbol class of each observation index.
    classes: &'a [u8],
    stride: usize,
    text: String,
    state_mass: Vec<f64>,
    samples: Vec<[f64; 5]>,
}

impl<'a> UnitWriter<'a> {
    fn new(years: &'a StateYears<'a>, classes: &'a [u8], stride: usize, sites: usize) -> Self {
        Self {
            years,
            classes,
            stride,
            text: String::with_capacity(sites * 140),
            state_mass: vec![0.0; years.values.len()],
            samples: Vec::new(),
        }
    }

    /// Lines for consecutive sites of sequence `seq_id`, the first of which
    /// is bin `seq_bin` of the sequence and `global_bin` of the input, with
    /// observation indices `obs` and posteriors `post` (`np` values per
    /// site).
    fn sites(
        &mut self,
        seq_id: usize,
        seq_bin: usize,
        global_bin: usize,
        obs: &[u16],
        post: &[f64],
        np: usize,
    ) {
        let (years, years_text) = (self.years.values, &self.years.text);
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
            let k025 = quantile_state(p, 0.025);
            let k975 = quantile_state(p, 0.975);
            let global_bin = global_bin + t;
            // Same text as "{}\t{}\t{}\t{}\t{}\t{:.8e}\t..." (see `push_sci8`).
            let line = &mut self.text;
            for v in [seq_id, seq_bin + t, global_bin] {
                push_usize(line, v);
                line.push('\t');
            }
            line.push(obs_char(self.classes[usize::from(o)]));
            line.push('\t');
            push_usize(line, max_k);
            line.push('\t');
            line.push_str(&years_text[max_k]);
            line.push('\t');
            push_sci8(line, mean);
            line.push('\t');
            line.push_str(&years_text[k025]);
            line.push('\t');
            line.push_str(&years_text[k975]);
            line.push('\t');
            push_sci8(line, max_p);
            line.push('\t');
            push_sci8(line, entropy);
            line.push('\n');
            if global_bin.is_multiple_of(self.stride) {
                self.samples.push([
                    global_bin as f64,
                    mean,
                    years[max_k],
                    years[k025],
                    years[k975],
                ]);
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

/// Checkpoints of a long sequence that continues into the next window.
type Carried = Option<(usize, DecodeCheckpoints)>;

/// Everything the parallel decoding tasks share.
struct Decoder<'a> {
    pass: Pass<'a>,
    /// First site of each sequence in the input; the last entry is the total.
    seq_start: Vec<usize>,
    years: StateYears<'a>,
    stride: usize,
    gzip: bool,
    ctx: Ctx<'a>,
    /// Workspaces of finished tasks, reused by later ones.
    free: Mutex<Vec<Workspace>>,
}

impl Decoder<'_> {
    fn checkpoints(&self, seq: usize) -> Result<DecodeCheckpoints> {
        fb::decode_checkpoints(&self.pass.kernel, &self.pass.seq(seq), &self.ctx)
    }

    fn with_workspace<R>(&self, f: impl FnOnce(&mut Workspace) -> R) -> R {
        let pop = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let mut ws = pop.unwrap_or_else(|| Workspace::new(self.pass.kernel.np()));
        let r = f(&mut ws);
        self.free.lock().unwrap_or_else(|e| e.into_inner()).push(ws);
        r
    }

    fn decode_unit(&self, unit: &Unit, ckpts: &[DecodeCheckpoints]) -> Result<UnitOut> {
        let k = &self.pass.kernel;
        let np = k.np();
        let sites = match unit {
            Unit::Short { seqs } => self.seq_start[seqs.end] - self.seq_start[seqs.start],
            Unit::Long { segs, .. } => segs.len() * CHECKPOINT_STRIDE,
        };
        let classes = self.pass.table.classes();
        let mut out = UnitWriter::new(&self.years, classes, self.stride, sites);
        self.with_workspace(|ws| -> Result<()> {
            match unit {
                Unit::Short { seqs } => {
                    for i in seqs.clone() {
                        let (obs, post) = fb::sequence_posteriors(k, &self.pass.seq(i), ws)?;
                        out.sites(i, 0, self.seq_start[i], obs, post, np);
                    }
                }
                Unit::Long { seq, ckpt, segs } => {
                    let seq_obs = self.pass.seq(*seq);
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
    fn decode_window(&self, window: &Window, carried: Carried) -> Result<(Vec<UnitOut>, Carried)> {
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
                    *slot = Some(self.checkpoints(seq)?);
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

    /// The first sequence from `from` on with more than [`WINDOW_SITES`]
    /// sites, or the number of sequences.
    fn next_big(&self, from: usize) -> usize {
        let n_seqs = self.seq_start.len() - 1;
        (from..n_seqs)
            .find(|&i| self.seq_start[i + 1] - self.seq_start[i] > WINDOW_SITES)
            .unwrap_or(n_seqs)
    }

    /// Decode the windows from `cursor` up to sequence `stop`, writing each
    /// one while the next is decoded. The last window's output is left in
    /// `decoded`, to be written by the caller.
    fn run_span(
        &self,
        cursor: &mut Cursor,
        stop: usize,
        carried: &mut Carried,
        output: &mut Output,
        decoded: &mut Option<(Vec<UnitOut>, usize)>,
    ) -> Result<()> {
        while let Some(w) = next_window(&self.seq_start, cursor, stop) {
            let (written, next) = rayon::join(
                || match decoded.take() {
                    Some((outs, end_site)) => output.append(outs, end_site),
                    None => Ok(()),
                },
                || self.decode_window(&w, carried.take()),
            );
            written?;
            let (outs, keep) = next?;
            *carried = keep;
            *decoded = Some((outs, w.end_site));
        }
        Ok(())
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

/// Options for [`write_tmrca_posterior_tsv_with_options`].
#[derive(Debug, Clone, Copy, Default)]
pub struct TmrcaOptions<'a> {
    /// Draw a progress bar on stderr.
    pub progress_bars: bool,
    /// Sample at most about this many bins for the report's summaries (0
    /// samples every bin).
    pub max_points: usize,
    /// The callable bases of each bin, row for row: decode with the
    /// callable-length emissions the fit used ([`super::EStepOptions::callable`]).
    /// `None` is C psmc's model.
    pub callable: Option<&'a CallableBases>,
}

/// Posterior decoding of the time to the most recent common ancestor, one
/// TSV line per bin (gzip-compressed when `out_path` ends in `.gz`), without
/// callable bases; see [`write_tmrca_posterior_tsv_with_options`].
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
    let opts = TmrcaOptions {
        progress_bars: progress_enabled,
        max_points,
        callable: None,
    };
    write_tmrca_posterior_tsv_with_options(pi, a, em, rows, row_starts, tmrca_years, out_path, opts)
}

/// Posterior decoding of the time to the most recent common ancestor, one
/// TSV line per bin (gzip-compressed when `out_path` ends in `.gz`). The
/// `obs` column is the bin's symbol; with callable bases
/// ([`TmrcaOptions::callable`]) each bin is decoded with the emissions of
/// its count, as the E-step fits them.
///
/// `tmrca_years[k]` is the time assigned to state `k`. Memory is bounded:
/// sequences longer than a work unit keep forward and backward checkpoints
/// every [`CHECKPOINT_STRIDE`] sites, and units are decoded in parallel,
/// across sequences as well as within them, and written in order. Returns
/// summaries for the HTML report (at most about [`TmrcaOptions::max_points`]
/// sampled bins).
#[allow(clippy::too_many_arguments)]
pub fn write_tmrca_posterior_tsv_with_options(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    tmrca_years: &[f64],
    out_path: &Path,
    opts: TmrcaOptions<'_>,
) -> Result<TmrcaReportData> {
    let TmrcaOptions {
        progress_bars: progress_enabled,
        max_points,
        callable,
    } = opts;
    let pass = Pass::new(pi, a, em, rows, row_starts, callable, None)?;
    let n_states = pi.len();
    if tmrca_years.len() != n_states {
        bail!(
            "tmrca_years length {} != n_states {n_states}",
            tmrca_years.len()
        );
    }
    let total_sites = pass.plan.total_sites as usize;
    let n_seqs = pass.plan.seqs.len();
    let seq_len = |r: &Range<usize>| rows[r.clone()].iter().map(Vec::len).sum();
    let seq_start = starts(pass.plan.seqs.iter().map(seq_len));
    let stride = if max_points == 0 {
        1
    } else {
        total_sites.div_ceil(max_points).max(1)
    };
    let progress = Progress::default();
    let budget = AlphaCacheBudget::new(0);
    let mut sink = Sink::create(out_path)?;
    let gzip = sink.gzip;
    sink.write(&Sink::encode(gzip, HEADER.to_string())?)?;
    let dec = Decoder {
        pass,
        seq_start,
        years: StateYears::new(tmrca_years),
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
            .then(|| progress::bar(total_sites as u64, "TMRCA", "posterior")),
    };

    // The output is decoded in spans that end at each sequence longer than a
    // window; the checkpoint passes of that sequence run alongside the span.
    // At most two such sequences have checkpoints in memory.
    let mut cursor = Cursor::default();
    let mut carried: Carried = None;
    let mut decoded = None;
    loop {
        let ready = carried.as_ref().is_some_and(|(s, _)| *s == cursor.seq);
        let stop = dec.next_big(cursor.seq + usize::from(ready));
        let (span, ahead) = rayon::join(
            || dec.run_span(&mut cursor, stop, &mut carried, &mut output, &mut decoded),
            || (stop < n_seqs).then(|| dec.checkpoints(stop)).transpose(),
        );
        span?;
        match ahead? {
            Some(c) => carried = Some((stop, c)),
            None => break,
        }
    }
    if let Some((outs, end_site)) = decoded.take() {
        output.append(outs, end_site)?;
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

    // Each sample is the bin, then the mean, MAP and 2.5% and 97.5% quantiles.
    let sampled = |col: usize| samples.iter().map(|s| [s[0], s[col]]).collect();
    Ok(TmrcaReportData {
        sampled_mean: sampled(1),
        sampled_map: sampled(2),
        sampled_lo: sampled(3),
        sampled_hi: sampled(4),
        state_mass,
        state_years: tmrca_years.to_vec(),
        total_sites,
        seq_count: n_seqs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: usize = CHECKPOINT_STRIDE;

    /// Walks every window, in spans ending at each sequence longer than a
    /// window as `write_tmrca_posterior_tsv` does, and checks that the units
    /// cover each site once, in order, within the size limits. Returns the
    /// number of units per window.
    fn check_windows(lens: &[usize]) -> Vec<usize> {
        let seq_start = starts(lens.iter().copied());
        let n = lens.len();
        let stops = (0..n).filter(|&i| lens[i] > WINDOW_SITES).chain([n]);
        let mut cur = Cursor::default();
        let mut next_site = 0;
        let mut continued: Option<usize>;
        let mut sizes = Vec::new();
        for stop in stops {
            // A span starting at a long sequence uses its prefetched checkpoints.
            continued = (cur.seq < stop && lens[cur.seq] > WINDOW_SITES).then_some(cur.seq);
            while let Some(w) = next_window(&seq_start, &mut cur, stop) {
                assert!(w.units.len() <= UNITS_PER_WINDOW);
                if let Some(seq) = continued {
                    assert_eq!(w.ckpt_seqs.first(), Some(&seq), "carried checkpoints");
                }
                for u in &w.units {
                    match u {
                        Unit::Short { seqs } => {
                            assert!(!seqs.is_empty() && seqs.end <= stop);
                            assert_eq!(seq_start[seqs.start], next_site);
                            assert!(seq_start[seqs.end] - seq_start[seqs.start] <= UNIT_SITES);
                            next_site = seq_start[seqs.end];
                        }
                        Unit::Long { seq, ckpt, segs } => {
                            assert!(lens[*seq] > UNIT_SITES && *seq < stop);
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
            assert_eq!((cur.seq, cur.seg), (stop, 0), "a span ends at its stop");
            assert_eq!(next_site, seq_start[stop]);
            assert_eq!(continued, None);
        }
        sizes
    }

    fn assert_sci8(v: f64) {
        let mut fast = String::new();
        push_sci8(&mut fast, v);
        assert_eq!(fast, format!("{v:.8e}"), "bits {:016x}", v.to_bits());
    }

    /// `push_sci8` must print exactly what `{:.8e}` prints, including where
    /// rounding is closest to a tie and around powers of ten.
    #[test]
    fn push_sci8_matches_std_formatting() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(1);
        let (mut in_range, mut fast_path) = (0, 0);
        for _ in 0..300_000 {
            // Any finite double, then the ranges the TSV uses.
            let v = f64::from_bits(rng.r#gen::<u64>());
            if v.is_finite() {
                assert_sci8(v);
            }
            let v = 10f64.powf(rng.gen_range(-30.0..30.0));
            assert_sci8(v);
            // Exact powers of ten reach decimal exponents -14..=30 (and the
            // first estimate of the exponent may be one too small).
            if (1e-13..1e30).contains(&v) {
                in_range += 1;
                fast_path += usize::from(sci8_digits(v).is_some());
            }
            assert_sci8(rng.gen_range(0.0..1.0));
        }
        assert!(
            fast_path + in_range / 1000 >= in_range,
            "fast path taken for {fast_path} of {in_range} values"
        );
        for k in -25..=25 {
            let p = 10f64.powi(k);
            for _ in 0..500 {
                // Halfway between two 9-digit values, and its neighbours.
                let n: u64 = rng.gen_range(100_000_000..1_000_000_000);
                let tie = (n as f64 + 0.5) * 10f64.powi(k - 8);
                for d in 0..4u64 {
                    assert_sci8(f64::from_bits(tie.to_bits() + d));
                    assert_sci8(f64::from_bits(tie.to_bits() - d));
                }
            }
            for d in 0..40u64 {
                assert_sci8(f64::from_bits(p.to_bits() + d));
                assert_sci8(f64::from_bits(p.to_bits() - d));
                // Rounds up to the next power of ten.
                let q = 9.999_999_995 * p;
                assert_sci8(f64::from_bits(q.to_bits() + d));
                assert_sci8(f64::from_bits(q.to_bits() - d));
            }
        }
        // Exact ties among integers (1000000005 has ten digits).
        for i in 0..20_000u64 {
            assert_sci8((1_000_000_000 + 5 * i) as f64);
            assert_sci8((999_990_000 + i) as f64);
        }
        for v in [
            0.0,
            -0.0,
            1.0,
            0.5,
            -2.5e-17,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert_sci8(v);
        }
    }

    #[test]
    fn push_usize_matches_display() {
        for n in [0, 1, 9, 10, 99, 100, 12_345, usize::MAX] {
            let mut s = String::new();
            push_usize(&mut s, n);
            assert_eq!(s, n.to_string());
        }
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
        // Spans end at every sequence longer than a window, also when such
        // sequences are adjacent or first; shorter ones are not stops.
        assert_eq!(
            check_windows(&[long, long, 5, WINDOW_SITES, long]),
            vec![64, 1, 64, 64, 2, 64, 1]
        );
    }
}
