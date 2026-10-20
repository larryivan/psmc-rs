//! The callable bases of each bin, for the callable-length emission model
//! of `docs/design/bin-callable-model.md`.
//!
//! A bin of `s` bases is `N` when more than 90% of them are missing
//! (`fq2psmcfa`'s rule), and C psmc treats every other bin as `s` callable
//! bases. The readers that bin per-site data (VCF in both modes and
//! multihetsep, through [`crate::io::common::Binner`]) also record each
//! bin's count `c`: `s` minus the positions of the bin classified missing.
//! The last bin of a sequence may be shorter than `s`; as `fq2psmcfa`
//! judges it against the full `s`, its positions past the end of the
//! sequence are not missing and count as callable. So the count and the
//! symbol always agree: a bin is `N` exactly when `10 * c < s`, and a `T` or
//! `K` bin has `0 < c <= s`. Complete data give `c = s` for every bin.
//! psmcfa input has no counts.
//!
//! # Storage
//!
//! Counts are 16-bit, so they are recorded only for bins of at most
//! [`MAX_COUNTED_BIN_SIZE`] bases (PSMC bins are normally 100); input
//! binned into larger bins has none, as psmcfa input.
//!
//! The counts of a row, sequence or trunk ([`BinCounts`]) are kept in the
//! smaller of two forms:
//!
//! - runs: the bins whose count is not `s`, as runs of consecutive bins
//!   with the same count, at 10 bytes per run. Full bins cost nothing, so
//!   complete data take no memory, and a stretch of missing data costs at
//!   most three runs whatever its length: the partly called bins at its two
//!   ends and the empty bins (count 0) between them;
//! - dense: one count per bin, at 1 byte when every count fits in a byte
//!   (always for bins of at most 255 bases, so for the usual 100) and 2
//!   bytes otherwise. It is the smaller when there are more runs than a
//!   tenth of the bins (a fifth for 2-byte counts), as when missing data
//!   come in stretches a few bins long or shorter, which leave many bins
//!   partly called.
//!
//! So the counts never take more than 2 bytes per bin, nor more than the
//! symbols themselves (1 byte per bin) for bins of at most 255 bases.

use std::borrow::Cow;
use std::ops::Range;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::io::{Observations, Sequences};

/// The largest bin size whose callable bases are recorded: counts are
/// 16-bit. The readers bin larger bins as before, without counts.
pub const MAX_COUNTED_BIN_SIZE: usize = u16::MAX as usize;

/// `bin_size` as the count of a full bin, when bins of that size are
/// counted (between 1 and [`MAX_COUNTED_BIN_SIZE`]).
pub(crate) fn counted_bin_size(bin_size: usize) -> Option<u16> {
    u16::try_from(bin_size).ok().filter(|&s| s > 0)
}

fn checked_bin_size(bin_size: usize) -> Result<u16> {
    match counted_bin_size(bin_size) {
        Some(s) => Ok(s),
        None => bail!(
            "the bin size of callable bases must be between 1 and {MAX_COUNTED_BIN_SIZE}, got \
             {bin_size}"
        ),
    }
}

/// Bytes a run of bins with the same count takes: its start, its length
/// (both 32-bit) and its count.
const RUN_BYTES: usize = 10;

/// Whether `len` bins, those that are not full forming `n_runs` runs, take
/// less memory as one count per bin (of 1 byte when `narrow`, else 2) than
/// as runs, or are too many to index in 32 bits.
fn prefers_dense(len: usize, n_runs: usize, narrow: bool) -> bool {
    let width = if narrow { 1 } else { 2 };
    u32::try_from(len).is_err() || n_runs.saturating_mul(RUN_BYTES) > len.saturating_mul(width)
}

/// Whether a count fits in a byte.
fn is_narrow(c: u16) -> bool {
    c <= u16::from(u8::MAX)
}

/// The maximal runs of consecutive bins with the same count, other than
/// `full`, in `counts`.
fn count_runs(full: u16, counts: impl Iterator<Item = u16>) -> usize {
    let mut n = 0;
    let mut prev = full;
    for c in counts {
        if c != full && c != prev {
            n += 1;
        }
        prev = c;
    }
    n
}

/// The callable bases of the bins of one row, sequence or trunk of bins of
/// `s` bases ([`BinCounts::bin_size`]); see the module docs for what a
/// count is and how the counts are stored.
///
/// Two `BinCounts` are equal when they have the same bin size and the same
/// count for every bin.
#[derive(Debug, Clone)]
pub struct BinCounts {
    /// `s`, the count of a full bin.
    full: u16,
    repr: Repr,
}

#[derive(Debug, Clone)]
enum Repr {
    /// `len` bins, those whose count is not `full` given as runs.
    Runs { len: usize, runs: Runs },
    /// One count per bin.
    Dense(Dense),
}

impl BinCounts {
    /// The counts of bins of `bin_size` bases, one per bin. The bin size
    /// must be between 1 and [`MAX_COUNTED_BIN_SIZE`]; the counts are checked
    /// with their bins ([`Observations::validate`]).
    pub fn new(bin_size: usize, counts: Vec<u16>) -> Result<Self> {
        Ok(Self::from_dense(
            checked_bin_size(bin_size)?,
            Dense::Wide(counts),
        ))
    }

    /// Store one count per bin in the smaller form.
    fn from_dense(full: u16, dense: Dense) -> Self {
        let len = dense.len();
        let n_runs = count_runs(full, dense.iter());
        let repr = if prefers_dense(len, n_runs, dense.iter().all(is_narrow)) {
            Repr::Dense(dense.into_narrowest())
        } else {
            let mut runs = Runs::with_capacity(n_runs);
            for (i, c) in dense.iter().enumerate() {
                if c != full {
                    runs.add(i, 1, c);
                }
            }
            Repr::Runs { len, runs }
        };
        Self { full, repr }
    }

    /// Store `len` bins, those that are not `full` given by `runs`, in the
    /// smaller form.
    fn from_runs(full: u16, len: usize, mut runs: Runs) -> Self {
        let n_full = len - runs.run.iter().map(|&r| r as usize).sum::<usize>();
        let narrow = (n_full == 0 || is_narrow(full)) && runs.count.iter().all(|&c| is_narrow(c));
        let repr = if prefers_dense(len, runs.len(), narrow) {
            let mut dense = Dense::full(full, len, narrow);
            runs.fill(&mut dense);
            Repr::Dense(dense)
        } else {
            runs.shrink_to_fit();
            Repr::Runs { len, runs }
        };
        Self { full, repr }
    }

    /// Bases per bin, `s`: the count of a full bin.
    pub fn bin_size(&self) -> usize {
        usize::from(self.full)
    }

    /// Number of bins.
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Runs { len, .. } => *len,
            Repr::Dense(dense) => dense.len(),
        }
    }

    /// Whether there are no bins.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The count of every bin, in order.
    pub fn iter(&self) -> Iter<'_> {
        Iter(match &self.repr {
            Repr::Runs { len, runs } => IterRepr::Runs {
                full: self.full,
                next: 0,
                len: *len,
                start: &runs.start,
                run: &runs.run,
                count: &runs.count,
            },
            Repr::Dense(Dense::Narrow(counts)) => IterRepr::Narrow(counts.iter()),
            Repr::Dense(Dense::Wide(counts)) => IterRepr::Wide(counts.iter()),
        })
    }

    /// One count per bin, in order.
    pub fn to_vec(&self) -> Vec<u16> {
        self.iter().collect()
    }

    /// The counts of bins `start..start + out.len()`, into `out`: a copy
    /// for one count per bin, a binary search and fills for runs, so that
    /// any stretch of bins is read in time proportional to its length (the
    /// E-step reads its input a segment at a time, forward and backward).
    ///
    /// # Panics
    ///
    /// If the bins are not within `0..self.len()`.
    pub(crate) fn read_into(&self, start: usize, out: &mut [u16]) {
        let end = start + out.len();
        assert!(
            end <= self.len(),
            "bins {start}..{end} out of {}",
            self.len()
        );
        match &self.repr {
            Repr::Dense(Dense::Narrow(counts)) => {
                for (o, &c) in out.iter_mut().zip(&counts[start..end]) {
                    *o = u16::from(c);
                }
            }
            Repr::Dense(Dense::Wide(counts)) => out.copy_from_slice(&counts[start..end]),
            Repr::Runs { runs, .. } => {
                out.fill(self.full);
                for k in runs.overlapping(start..end) {
                    let from = start.max(runs.start[k] as usize);
                    let to = end.min(runs.end(k));
                    out[from - start..to - start].fill(runs.count[k]);
                }
            }
        }
    }

    /// Bytes of memory the counts hold, besides the struct itself.
    pub fn heap_bytes(&self) -> usize {
        match &self.repr {
            Repr::Runs { runs, .. } => runs.heap_bytes(),
            Repr::Dense(dense) => dense.heap_bytes(),
        }
    }

    /// The counts of the bins in `range`.
    ///
    /// # Panics
    ///
    /// If `range` is not within `0..self.len()`.
    pub(crate) fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len(),
            "bins {range:?} out of {}",
            self.len()
        );
        match &self.repr {
            Repr::Runs { runs, .. } => {
                let overlapping = runs.overlapping(range.clone());
                let mut cut = Runs::with_capacity(overlapping.len());
                for k in overlapping {
                    let from = range.start.max(runs.start[k] as usize);
                    let to = range.end.min(runs.end(k));
                    cut.add(from - range.start, to - from, runs.count[k]);
                }
                Self::from_runs(self.full, range.len(), cut)
            }
            Repr::Dense(dense) => Self::from_dense(self.full, dense.slice(range)),
        }
    }

    /// The counts of `parts`, one after the other.
    ///
    /// # Panics
    ///
    /// If `parts` is empty or its bin sizes differ.
    pub(crate) fn concat(parts: &[BinCounts]) -> Self {
        let mut builder = CountsBuilder::new(parts[0].full);
        for part in parts {
            builder.append(part);
        }
        builder.finish()
    }
}

impl PartialEq for BinCounts {
    fn eq(&self, other: &Self) -> bool {
        self.full == other.full && self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl Eq for BinCounts {}

impl<'a> IntoIterator for &'a BinCounts {
    type Item = u16;
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

/// Runs of consecutive bins with the same count, in increasing order: run
/// `k` is the `run[k]` bins from `start[k]` on, each of `count[k]` bases.
/// [`Repr::Runs`] keeps them maximal, one per count change.
#[derive(Debug, Clone, Default)]
struct Runs {
    start: Vec<u32>,
    run: Vec<u32>,
    count: Vec<u16>,
}

impl Runs {
    fn with_capacity(n: usize) -> Self {
        Self {
            start: Vec::with_capacity(n),
            run: Vec::with_capacity(n),
            count: Vec::with_capacity(n),
        }
    }

    fn len(&self) -> usize {
        self.start.len()
    }

    /// The bin after the last of run `k`.
    fn end(&self, k: usize) -> usize {
        self.start[k] as usize + self.run[k] as usize
    }

    /// The runs that hold a bin of `bins`, found by binary search.
    fn overlapping(&self, bins: Range<usize>) -> Range<usize> {
        let mut first = self.start.partition_point(|&s| (s as usize) < bins.start);
        if first > 0 && self.end(first - 1) > bins.start {
            first -= 1;
        }
        let last = self.start.partition_point(|&s| (s as usize) < bins.end);
        first..last
    }

    fn heap_bytes(&self) -> usize {
        (self.start.capacity() + self.run.capacity()) * size_of::<u32>()
            + self.count.capacity() * size_of::<u16>()
    }

    /// Whether bins of `c` bases from bin `at` on would extend the last run.
    fn extends_last(&self, at: usize, c: u16) -> bool {
        self.count.last() == Some(&c) && self.end(self.len() - 1) == at
    }

    /// Add `n` bins of `c` bases from bin `at` on, after every run so far;
    /// `at + n` must fit in 32 bits.
    fn add(&mut self, at: usize, n: usize, c: u16) {
        if n == 0 {
            return;
        }
        let in_32_bits = |v: usize| u32::try_from(v).expect("runs within 32 bits");
        if self.extends_last(at, c) {
            let last = self.run.last_mut().expect("a last run");
            *last = in_32_bits(*last as usize + n);
        } else {
            self.start.push(in_32_bits(at));
            self.run.push(in_32_bits(n));
            self.count.push(c);
        }
    }

    /// Write the runs' counts into `dense`.
    fn fill(&self, dense: &mut Dense) {
        for k in 0..self.len() {
            dense.fill(self.start[k] as usize..self.end(k), self.count[k]);
        }
    }

    fn shrink_to_fit(&mut self) {
        self.start.shrink_to_fit();
        self.run.shrink_to_fit();
        self.count.shrink_to_fit();
    }
}

/// One count per bin.
#[derive(Debug, Clone)]
enum Dense {
    /// Every count at most 255.
    Narrow(Vec<u8>),
    Wide(Vec<u16>),
}

impl Dense {
    /// `len` full bins, narrow when `narrow` (the value of the bins is then
    /// 0 when `full` does not fit in a byte, for bins all overwritten).
    fn full(full: u16, len: usize, narrow: bool) -> Self {
        if narrow {
            Dense::Narrow(vec![u8::try_from(full).unwrap_or(0); len])
        } else {
            Dense::Wide(vec![full; len])
        }
    }

    fn len(&self) -> usize {
        match self {
            Dense::Narrow(counts) => counts.len(),
            Dense::Wide(counts) => counts.len(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Dense::Narrow(counts) => counts.capacity(),
            Dense::Wide(counts) => counts.capacity() * size_of::<u16>(),
        }
    }

    fn iter(&self) -> Iter<'_> {
        Iter(match self {
            Dense::Narrow(counts) => IterRepr::Narrow(counts.iter()),
            Dense::Wide(counts) => IterRepr::Wide(counts.iter()),
        })
    }

    fn slice(&self, range: Range<usize>) -> Self {
        match self {
            Dense::Narrow(counts) => Dense::Narrow(counts[range].to_vec()),
            Dense::Wide(counts) => Dense::Wide(counts[range].to_vec()),
        }
    }

    /// Hold counts of more than 255 (no change when wide).
    fn widen(&mut self) {
        if let Dense::Narrow(counts) = self {
            *self = Dense::Wide(counts.iter().map(|&c| u16::from(c)).collect());
        }
    }

    /// Set the bins in `range` to `c`.
    fn fill(&mut self, range: Range<usize>, c: u16) {
        if !is_narrow(c) {
            self.widen();
        }
        match self {
            // Lossless: `c` fits in a byte.
            Dense::Narrow(counts) => counts[range].fill(c as u8),
            Dense::Wide(counts) => counts[range].fill(c),
        }
    }

    /// Add `n` bins of `c` bases.
    fn push_run(&mut self, c: u16, n: usize) {
        if !is_narrow(c) {
            self.widen();
        }
        match self {
            // Lossless: `c` fits in a byte.
            Dense::Narrow(counts) => counts.extend(std::iter::repeat_n(c as u8, n)),
            Dense::Wide(counts) => counts.extend(std::iter::repeat_n(c, n)),
        }
    }

    /// The same counts, narrow when they all fit in a byte, without spare
    /// capacity.
    fn into_narrowest(self) -> Self {
        match self {
            Dense::Wide(counts) if counts.iter().all(|&c| is_narrow(c)) => {
                // Lossless: every count fits in a byte.
                Dense::Narrow(counts.iter().map(|&c| c as u8).collect())
            }
            Dense::Narrow(mut counts) => {
                counts.shrink_to_fit();
                Dense::Narrow(counts)
            }
            Dense::Wide(mut counts) => {
                counts.shrink_to_fit();
                Dense::Wide(counts)
            }
        }
    }
}

/// The count of every bin of a [`BinCounts`], in order
/// ([`BinCounts::iter`]).
#[derive(Debug, Clone)]
pub struct Iter<'a>(IterRepr<'a>);

#[derive(Debug, Clone)]
enum IterRepr<'a> {
    Runs {
        full: u16,
        next: usize,
        len: usize,
        start: &'a [u32],
        run: &'a [u32],
        count: &'a [u16],
    },
    Narrow(std::slice::Iter<'a, u8>),
    Wide(std::slice::Iter<'a, u16>),
}

impl<'a> Iterator for Iter<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        match &mut self.0 {
            IterRepr::Narrow(counts) => counts.next().map(|&c| u16::from(c)),
            IterRepr::Wide(counts) => counts.next().copied(),
            IterRepr::Runs {
                full,
                next,
                len,
                start,
                run,
                count,
            } => {
                if *next == *len {
                    return None;
                }
                let runs: (&'a [u32], &'a [u32], &'a [u16]) = (*start, *run, *count);
                let c = match runs {
                    ([s, ..], [n, ..], [c, ..]) if *s as usize <= *next => {
                        if *next + 1 == *s as usize + *n as usize {
                            (*start, *run, *count) = (&runs.0[1..], &runs.1[1..], &runs.2[1..]);
                        }
                        *c
                    }
                    _ => *full,
                };
                *next += 1;
                Some(c)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.0 {
            IterRepr::Narrow(counts) => counts.len(),
            IterRepr::Wide(counts) => counts.len(),
            IterRepr::Runs { next, len, .. } => len - next,
        };
        (n, Some(n))
    }
}

impl ExactSizeIterator for Iter<'_> {}

/// Builds the [`BinCounts`] of bins added in order, as a binner emits them,
/// in the smaller form. It keeps runs until they take more memory than one
/// count per bin would (with some slack, so that the first bins do not
/// decide), then one count per bin, so that it holds about as much as its
/// result.
#[derive(Debug, Clone)]
pub(crate) struct CountsBuilder {
    full: u16,
    len: usize,
    /// Whether every count so far fits in a byte.
    narrow: bool,
    runs: Runs,
    /// One count per bin, once runs take too much memory.
    dense: Option<Dense>,
}

/// Bytes of runs a builder may hold beyond one count per bin.
const RUNS_SLACK_BYTES: usize = 16 * 1024;

impl CountsBuilder {
    pub(crate) fn new(full: u16) -> Self {
        Self {
            full,
            len: 0,
            narrow: true,
            runs: Runs::default(),
            dense: None,
        }
    }

    /// Add a bin of `c` callable bases.
    pub(crate) fn push(&mut self, c: u16) {
        self.push_run(c, 1);
    }

    /// Add `n` bins of `c` callable bases each.
    fn push_run(&mut self, c: u16, n: usize) {
        if n == 0 {
            return;
        }
        let end = self.len + n;
        if self.dense.is_none() && c != self.full {
            let n_runs = self.runs.len() + usize::from(!self.runs.extends_last(self.len, c));
            let width = if self.narrow && is_narrow(c) { 1 } else { 2 };
            let limit = end.saturating_mul(width).saturating_add(RUNS_SLACK_BYTES);
            if u32::try_from(end).is_err() || n_runs.saturating_mul(RUN_BYTES) > limit {
                let mut dense = Dense::full(self.full, self.len, self.narrow);
                std::mem::take(&mut self.runs).fill(&mut dense);
                self.dense = Some(dense);
            } else {
                self.runs.add(self.len, n, c);
            }
        }
        if let Some(dense) = &mut self.dense {
            dense.push_run(c, n);
        }
        self.narrow &= is_narrow(c);
        self.len = end;
    }

    /// Add the bins of `other`, which must be of the same size.
    fn append(&mut self, other: &BinCounts) {
        assert_eq!(other.full, self.full, "counts of bins of different sizes");
        match &other.repr {
            Repr::Runs { len, runs } => {
                let mut done = 0;
                for ((&s, &n), &c) in runs.start.iter().zip(&runs.run).zip(&runs.count) {
                    self.push_run(self.full, s as usize - done);
                    self.push_run(c, n as usize);
                    done = s as usize + n as usize;
                }
                self.push_run(self.full, len - done);
            }
            Repr::Dense(dense) => dense.iter().for_each(|c| self.push(c)),
        }
    }

    /// The counts of the bins added since the last call, leaving the
    /// builder empty.
    pub(crate) fn finish(&mut self) -> BinCounts {
        let len = std::mem::take(&mut self.len);
        let runs = std::mem::take(&mut self.runs);
        self.narrow = true;
        match self.dense.take() {
            Some(dense) => BinCounts::from_dense(self.full, dense),
            None => BinCounts::from_runs(self.full, len, runs),
        }
    }
}

/// The callable bases of each bin of a list of rows or sequences of bins,
/// one [`BinCounts`] per row or sequence, all for bins of the same size.
/// See the module docs for what a count is; [`Observations::validate`] and
/// [`Sequences::validate`] check the counts against their bins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallableBases {
    bin_size: u16,
    rows: Vec<BinCounts>,
}

impl CallableBases {
    /// Counts for bins of `bin_size` bases, which must be between 1 and
    /// [`MAX_COUNTED_BIN_SIZE`], one [`BinCounts`] per row or sequence. The
    /// counts themselves are checked with their bins
    /// ([`Observations::validate`]).
    pub fn new(bin_size: usize, rows: Vec<BinCounts>) -> Result<Self> {
        Ok(Self {
            bin_size: checked_bin_size(bin_size)?,
            rows,
        })
    }

    /// [`CallableBases::new`] from one count per bin, row for row.
    pub fn from_counts(bin_size: usize, counts: Vec<Vec<u16>>) -> Result<Self> {
        let full = checked_bin_size(bin_size)?;
        Ok(Self {
            bin_size: full,
            rows: counts
                .into_iter()
                .map(|c| BinCounts::from_dense(full, Dense::Wide(c)))
                .collect(),
        })
    }

    /// The counts of sequences a [`crate::io::common::Binner`] of bins of
    /// `bin_size` bases binned, one entry per sequence: `None` when bins of
    /// that size are not counted, and otherwise every entry has counts.
    pub(crate) fn of_binned(
        bin_size: usize,
        rows: impl IntoIterator<Item = Option<BinCounts>>,
    ) -> Option<Self> {
        let bin_size = counted_bin_size(bin_size)?;
        let rows = rows
            .into_iter()
            .map(|r| r.expect("a binner of counted bins counts every sequence"))
            .collect();
        Some(Self { bin_size, rows })
    }

    /// Bases per bin, `s`.
    pub fn bin_size(&self) -> usize {
        usize::from(self.bin_size)
    }

    /// The counts, one entry per row or sequence.
    pub fn rows(&self) -> &[BinCounts] {
        &self.rows
    }

    /// The counts, one entry per row or sequence.
    pub fn into_rows(self) -> Vec<BinCounts> {
        self.rows
    }

    /// Other counts for bins of the same size, such as the rows or trunks
    /// these counts are cut into.
    pub fn with_rows(&self, rows: Vec<BinCounts>) -> Self {
        Self {
            bin_size: self.bin_size,
            rows,
        }
    }

    /// One count per bin, row for row.
    pub fn to_counts(&self) -> Vec<Vec<u16>> {
        self.rows.iter().map(BinCounts::to_vec).collect()
    }

    /// Bytes of memory the counts hold, besides the structs themselves
    /// ([`BinCounts::heap_bytes`] of every row).
    pub fn heap_bytes(&self) -> usize {
        self.rows.capacity() * size_of::<BinCounts>()
            + self.rows.iter().map(BinCounts::heap_bytes).sum::<usize>()
    }
}

/// How much of its called bins an input's counts say is callable: the bias
/// that treating every called bin as full would cause (theta, times and Ne
/// low by about `1 - mean_fraction`), and what the callable-length model
/// corrects for. See [`CallableSummary::of`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallableSummary {
    /// Bases per bin, `s`.
    pub bin_size: usize,
    /// Called (`T` or `K`) bins.
    pub called_bins: u64,
    /// Called bins with fewer than `s` callable bases.
    pub partial_bins: u64,
    /// The mean share of callable bases over the called bins, `sum c / (s
    /// * called_bins)`: the effective bin length over `s` (1 for complete
    ///   data; 0 without called bins).
    pub mean_fraction: f64,
    /// Called bins by their share of callable bases, in tenths: entry `i`
    /// for `i` from 0 to 8 counts the partly callable bins with `c / s` in
    /// `[(i + 1) / 10, (i + 2) / 10)`, and entry 9 the full bins. A called
    /// bin holds at least a tenth of its bases callable.
    pub histogram: [u64; 10],
}

impl CallableSummary {
    /// The summary of `rows` (symbols 0 `T`, 1 `K`, 2 `N`) with their
    /// callable bases, row for row. Bins whose count is above the bin size,
    /// or called with less than a tenth of it, count in the nearest class.
    pub fn of(rows: &[Vec<u8>], callable: &CallableBases) -> Self {
        let s = callable.bin_size();
        let mut histogram = [0u64; 10];
        let (mut called, mut partial, mut sum) = (0u64, 0u64, 0u64);
        for (row, counts) in rows.iter().zip(callable.rows()) {
            for (&v, c) in row.iter().zip(counts) {
                if v >= 2 {
                    continue;
                }
                let c = usize::from(c).min(s);
                called += 1;
                sum += c as u64;
                if c == s {
                    histogram[9] += 1;
                } else {
                    partial += 1;
                    histogram[((10 * c) / s).clamp(1, 9) - 1] += 1;
                }
            }
        }
        Self {
            bin_size: s,
            called_bins: called,
            partial_bins: partial,
            mean_fraction: if called == 0 {
                0.0
            } else {
                sum as f64 / (s as f64 * called as f64)
            },
            histogram,
        }
    }

    /// The share of called bins that are partly callable (0 without called
    /// bins).
    pub fn partial_fraction(&self) -> f64 {
        if self.called_bins == 0 {
            0.0
        } else {
            self.partial_bins as f64 / self.called_bins as f64
        }
    }
}

/// The counts of the rows `rows` of `counts` as one, borrowed when it is
/// one row.
pub(crate) fn join_counts(counts: &[BinCounts], rows: Range<usize>) -> Cow<'_, BinCounts> {
    if rows.len() == 1 {
        Cow::Borrowed(&counts[rows.start])
    } else {
        Cow::Owned(BinCounts::concat(&counts[rows]))
    }
}

/// The checks of [`Observations::validate`] and [`Sequences::validate`]:
/// `bins` are rows chained by `row_starts`, or sequences of their own when it
/// is `None`.
pub(crate) fn validate(
    bins: &[Vec<u8>],
    row_starts: Option<&[bool]>,
    callable: Option<&CallableBases>,
) -> Result<()> {
    let entry = if row_starts.is_some() {
        "row"
    } else {
        "sequence"
    };
    if let Some(starts) = row_starts {
        if starts.len() != bins.len() {
            bail!(
                "{} row starts for {} rows of bins",
                starts.len(),
                bins.len()
            );
        }
        if starts.first() == Some(&false) {
            bail!("the first row must start a sequence");
        }
    }
    let Some(callable) = callable else {
        return Ok(());
    };
    let rows = callable.rows();
    if rows.len() != bins.len() {
        bail!(
            "callable bases for {} {entry}s, but there are {} {entry}s of bins",
            rows.len(),
            bins.len()
        );
    }
    let s = u32::from(callable.bin_size);
    for (i, (row, row_counts)) in bins.iter().zip(rows).enumerate() {
        if row_counts.full != callable.bin_size {
            bail!(
                "{entry} {i} has callable bases of bins of {} bases, in counts of bins of {s}",
                row_counts.full
            );
        }
        if row_counts.len() != row.len() {
            bail!(
                "{entry} {i} has {} bins but callable bases for {}",
                row.len(),
                row_counts.len()
            );
        }
        for (j, (&symbol, c)) in row.iter().zip(row_counts).enumerate() {
            let c = u32::from(c);
            let at = || format!("{entry} {i}, bin {j}");
            if c > s {
                bail!("{}: {c} callable bases in a bin of {s}", at());
            }
            // fq2psmcfa's rule: N when more than 90% of the bases are missing.
            let missing = 10 * c < s;
            match symbol {
                0 | 1 if missing => bail!(
                    "{}: a called bin ({}) with {c} callable bases of {s}, fewer than 10%",
                    at(),
                    if symbol == 0 { "T" } else { "K" }
                ),
                2 if !missing => bail!(
                    "{}: a missing bin (N) with {c} callable bases of {s}, at least 10%",
                    at()
                ),
                0..=2 => {}
                _ => bail!("{}: invalid bin symbol {symbol} (expected 0, 1 or 2)", at()),
            }
        }
    }
    Ok(())
}

impl Observations {
    /// Check that the rows, their starts and their callable bases fit
    /// together: one start per row, the first row starting a sequence; and,
    /// with callable bases, one count per bin, for bins of the same size,
    /// none above the bin size, and every symbol T (0), K (1) or N (2) as
    /// its count says (N exactly when fewer than 10% of its bases are
    /// callable).
    pub fn validate(&self) -> Result<()> {
        validate(&self.rows, Some(&self.row_starts), self.callable.as_ref())
    }
}

impl Sequences {
    /// [`Observations::validate`] for whole sequences: with callable bases,
    /// one count per bin, for bins of the same size, none above the bin
    /// size, and every symbol as its count says.
    pub fn validate(&self) -> Result<()> {
        validate(&self.bins, None, self.callable.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64: deterministic and dependency-free.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in `lo..=hi`.
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
        }
    }

    /// `len` counts of bins of `full` bases, in stretches: full bins, empty
    /// ones (count 0, as in a long run of missing data), one partial count
    /// repeated, or a different partial count per bin. `other_per_mille` of
    /// the stretches are not full.
    fn random_counts(rng: &mut Rng, full: u16, len: usize, other_per_mille: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let n = rng.range(1, 40).min(len - out.len());
            let partial = |rng: &mut Rng| rng.range(0, usize::from(full) - 1) as u16;
            if rng.range(0, 999) >= other_per_mille {
                out.extend(std::iter::repeat_n(full, n));
                continue;
            }
            match rng.range(0, 2) {
                0 => out.extend(std::iter::repeat_n(0, n)),
                1 => {
                    let c = partial(rng);
                    out.extend(std::iter::repeat_n(c, n));
                }
                _ => out.extend((0..n).map(|_| partial(rng))),
            }
        }
        out
    }

    fn is_dense(c: &BinCounts) -> bool {
        !matches!(c.repr, Repr::Runs { .. })
    }

    /// The form [`BinCounts`] promises for `counts`: runs of 10 bytes, or one
    /// count per bin of 1 byte when all fit in a byte, else 2, whichever is
    /// smaller.
    fn check_form(c: &BinCounts, counts: &[u16], ctx: &str) {
        let full = c.full;
        // Maximal runs of equal counts that are not full, counted directly.
        let n_runs = (0..counts.len())
            .filter(|&i| counts[i] != full && (i == 0 || counts[i - 1] != counts[i]))
            .count();
        let width = if counts.iter().all(|&x| x <= 255) {
            1
        } else {
            2
        };
        let dense = 10 * n_runs > width * counts.len();
        assert_eq!(is_dense(c), dense, "{ctx}: {n_runs} runs, {}", counts.len());
        let want = if dense {
            width * counts.len()
        } else {
            10 * n_runs
        };
        assert_eq!(c.heap_bytes(), want, "{ctx}");
        assert!(c.heap_bytes() <= width * counts.len(), "{ctx}");
    }

    #[test]
    fn counts_come_back_in_the_smaller_form() {
        let mut rng = Rng(5);
        for case in 0..600 {
            let full = [1, 7, 100, 255, 256, 1000, u16::MAX][case % 7];
            let len = rng.range(0, 3000);
            let per_mille = [0, 10, 100, 300, 600, 1000][case % 6];
            let counts = random_counts(&mut rng, full, len, per_mille);
            let ctx = format!("case {case}, s {full}, len {len}");
            let c = BinCounts::new(usize::from(full), counts.clone()).unwrap();
            assert_eq!((c.len(), c.bin_size()), (len, usize::from(full)), "{ctx}");
            assert_eq!(c.to_vec(), counts, "{ctx}");
            assert_eq!(c.iter().len(), len, "{ctx}");
            check_form(&c, &counts, &ctx);

            // Built a bin at a time, the counts are the same and in the same
            // form.
            let mut b = CountsBuilder::new(full);
            for &x in &counts {
                b.push(x);
            }
            let built = b.finish();
            assert_eq!(built.to_vec(), counts, "{ctx}");
            check_form(&built, &counts, &ctx);
            // The builder is empty again.
            assert!(b.finish().is_empty(), "{ctx}");

            // Slices, and their concatenation.
            let mut cuts = vec![0, len];
            for _ in 0..rng.range(0, 6) {
                cuts.push(rng.range(0, len));
            }
            cuts.sort_unstable();
            let parts: Vec<BinCounts> = cuts.windows(2).map(|w| c.slice(w[0]..w[1])).collect();
            for (w, part) in cuts.windows(2).zip(&parts) {
                let ctx = format!("{ctx}, bins {w:?}");
                assert_eq!(part.to_vec(), counts[w[0]..w[1]], "{ctx}");
                check_form(part, &counts[w[0]..w[1]], &ctx);
                // A slice of a slice.
                let mid = w[0] + (w[1] - w[0]) / 3;
                let sub = part.slice(mid - w[0]..w[1] - w[0]);
                assert_eq!(sub.to_vec(), counts[mid..w[1]], "{ctx}");
            }
            let joined = BinCounts::concat(&parts);
            assert_eq!(joined.to_vec(), counts, "{ctx}");
            check_form(&joined, &counts, &ctx);

            // Any stretch reads back as the same counts, in either form.
            for _ in 0..4 {
                let from = rng.range(0, len);
                let to = rng.range(from, len);
                let mut out = vec![u16::MAX; to - from];
                c.read_into(from, &mut out);
                assert_eq!(out, counts[from..to], "{ctx}, read {from}..{to}");
            }
        }
    }

    #[test]
    fn full_bins_cost_nothing_and_runs_little() {
        let c = BinCounts::new(100, vec![100; 100_000]).unwrap();
        assert_eq!(c.heap_bytes(), 0);
        // A run of empty bins between two partly called ones: three runs,
        // whatever the length of the run.
        let mut b = CountsBuilder::new(100);
        for _ in 0..50_000 {
            b.push(100);
        }
        b.push(40);
        for _ in 0..20_000 {
            b.push(0);
        }
        b.push(70);
        for _ in 0..50_000 {
            b.push(100);
        }
        let c = b.finish();
        assert_eq!((c.len(), c.heap_bytes()), (120_002, 30));
        assert_eq!(
            c.slice(49_999..50_003).to_vec(),
            [100, 40, 0, 0],
            "a slice cuts the runs"
        );
        // Every bin partly called, each differently from the last: one byte
        // per bin for bins of up to 255 bases, two above.
        let partial: Vec<u16> = (0..1000).map(|i| 50 + i % 2).collect();
        let c = BinCounts::new(100, partial.clone()).unwrap();
        assert_eq!(c.heap_bytes(), 1000);
        let c = BinCounts::new(1000, partial.iter().map(|&x| x + 500).collect()).unwrap();
        assert_eq!(c.heap_bytes(), 2000);
        let cb = CallableBases::new(1000, vec![c.clone(), c.slice(0..0)]).unwrap();
        assert_eq!(cb.heap_bytes(), 2 * size_of::<BinCounts>() + 2000);
    }

    #[test]
    fn the_builder_holds_little_more_than_its_result() {
        // Counts that alternate, so that runs would take 10 bytes per bin:
        // the builder switches to one byte per bin early on.
        let mut b = CountsBuilder::new(100);
        for i in 0..100_000u32 {
            b.push(50 + (i % 2) as u16);
            let held = b.dense.as_ref().map_or(0, Dense::heap_bytes) + b.runs.heap_bytes();
            assert!(held <= 2 * b.len + 2 * RUNS_SLACK_BYTES, "bin {i}: {held}");
        }
        assert_eq!(b.finish().heap_bytes(), 100_000);
        // A count above 255 widens it.
        for i in 0..10_000u32 {
            b.push(if i == 5000 { 300 } else { 50 + (i % 2) as u16 });
        }
        let c = b.finish();
        assert_eq!(c.heap_bytes(), 20_000);
        assert_eq!(c.iter().nth(5000), Some(300));
    }

    #[test]
    fn equal_counts_are_equal_in_any_form() {
        let mut counts = vec![100; 100];
        counts[2..4].fill(30);
        counts[4] = 0;
        let runs = BinCounts::new(100, counts.clone()).unwrap();
        let narrow = BinCounts {
            full: 100,
            repr: Repr::Dense(Dense::Narrow(counts.iter().map(|&x| x as u8).collect())),
        };
        let wide = BinCounts {
            full: 100,
            repr: Repr::Dense(Dense::Wide(counts.clone())),
        };
        assert!(!is_dense(&runs) && is_dense(&narrow) && is_dense(&wide));
        assert_eq!(runs, narrow);
        assert_eq!(runs, wide);
        assert_ne!(runs, runs.slice(0..99));
        let other_size = BinCounts {
            full: 99,
            repr: Repr::Dense(Dense::Wide(counts)),
        };
        assert_ne!(other_size, wide);
    }
}
