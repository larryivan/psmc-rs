//! VCF input for one diploid sample.
//!
//! The reader produces exactly the binned sequence that `fq2psmcfa -s <bin>`
//! produces from a per-base consensus of the selected sample: every position
//! of a chromosome is classified as a called homozygous site, a heterozygous
//! SNP or missing, and the positions are binned with the `fq2psmcfa` rule
//! (see [`Binner`](crate::io::common::Binner)).
//!
//! A position is missing if it lies outside the callable set (see
//! [`VcfOptions::masks`]), if one of its records leaves the genotype unknown,
//! or if it lies in the REF span of a heterozygous non-SNP. Otherwise it is a
//! heterozygous SNP if one of its records is, and a called homozygous site in
//! every other case, including positions without any record. Per record, for
//! the selected sample:
//!
//! - FILTER other than `PASS` or `.`: POS is missing, unless
//!   [`VcfOptions::keep_filtered`] (then classified like any other record);
//! - missing or partial `GT` (`.`, `./.`, `./1`), or no `GT`: POS is missing;
//! - heterozygous (the allele strings differ) with every allele a single
//!   `A`/`C`/`G`/`T` (any case): heterozygous SNP;
//! - heterozygous with another allele (indel, `*`, symbolic `<...>`): the
//!   whole REF span `POS..POS+len(REF)-1` is missing (`END` and `SVLEN` are
//!   not consulted, except for gVCF blocks, below);
//! - homozygous (identical allele strings, including hom-alt and homozygous
//!   indels) or haploid: called homozygous.
//!
//! A filtered or missing genotype does not reach past POS: the record carries
//! no usable call, so the bases after POS keep their own records (in an
//! all-sites VCF, possibly PASS heterozygous SNPs) or the mask's verdict.
//!
//! The exception is a gVCF block: a record whose ALT alleles all stand for
//! "no other allele" (`.`, `<*>`, `<NON_REF>` or `<X>`) and whose INFO/END is
//! at or after POS states its genotype for every position of `POS..=END`
//! (END is clamped to the `##contig` length). When that record makes POS
//! missing (filtered, missing genotype, heterozygous), the whole block is
//! missing; a reference block is read like positions without records, which
//! the mask decides, and a sequence that ends at its last record ends at the
//! END of a block. [`VcfSummary::warnings`] points out that reference blocks
//! need a callable mask made from their depth and quality.
//!
//! In a gVCF the records and blocks cover every position the caller looked
//! at; `bcftools call -g` writes nothing where there are no reads. Read as a
//! gVCF ([`VcfOptions::gvcf`]), a position that no record, REF span or block
//! covers is therefore missing, whatever the masks say.
//!
//! Each bin also gets its callable bases: the bin size minus its positions
//! classified missing ([`crate::io::callable`]).
//!
//! The file is streamed, and the work is proportional to the number of records
//! and mask intervals plus the number of output bins, never to the genome
//! length: runs of positions between records are emitted straight from the
//! mask intervals.

use anyhow::{Context, Result, bail};
use std::borrow::Cow;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use crate::io::common::{
    BinnedSeq, DepthCounter, DepthStats, SeqBuilder, chunk_sequences, open_reader,
};
use crate::io::consensus::utf8_chrom;
use crate::io::{Observations, Sequences};
use crate::qc::fmt::{count, counted};

type Interval = (u64, u64); // inclusive [start, end], 1-based

/// Largest accepted POS and `##contig` length: 2^36, about 68.7 Gb, several
/// times longer than any known chromosome. The work and memory of a sequence
/// grow with its length (687M bins at the default bin size for this one), so
/// without a bound a single bad POS or header line makes the reader allocate
/// bins until memory runs out.
pub(crate) const MAX_POS: u64 = 1 << 36;

/// Sort `xs` and merge its overlapping and adjacent intervals, in place.
fn merge_in_place(xs: &mut Vec<Interval>) {
    if xs.len() < 2 {
        return;
    }
    xs.sort_unstable();
    let mut w = 0;
    for r in 1..xs.len() {
        let (s, e) = xs[r];
        if s <= xs[w].1.saturating_add(1) {
            xs[w].1 = max(xs[w].1, e);
        } else {
            w += 1;
            xs[w] = (s, e);
        }
    }
    xs.truncate(w + 1);
}

fn merge_intervals(mut xs: Vec<Interval>) -> Vec<Interval> {
    merge_in_place(&mut xs);
    xs
}

/// The intervals of one chromosome (or the chromosome-less rows) of a mask,
/// merged as they are added, so that memory follows the merged intervals
/// rather than the rows: a mask of one row per callable site, as an
/// all-sites VCF gives, is billions of rows for a human genome but a few
/// million merged intervals.
///
/// A row that overlaps or touches the last interval extends it; in a sorted
/// mask that merges every row that can be merged, and the list stays sorted
/// and merged. Rows out of order are appended, and the list is sorted and
/// merged again whenever it has doubled since the last time, which keeps the
/// work at O(n log n) and the memory within twice the merged size (plus
/// [`IntervalAcc::MIN_COMPACT`] rows).
#[derive(Debug, Clone)]
struct IntervalAcc {
    xs: Vec<Interval>,
    /// Whether `xs` is sorted and merged.
    merged: bool,
    /// Length at which an unmerged `xs` is merged again.
    compact_at: usize,
}

impl Default for IntervalAcc {
    fn default() -> Self {
        Self {
            xs: Vec::new(),
            merged: true,
            compact_at: Self::MIN_COMPACT,
        }
    }
}

impl IntervalAcc {
    const MIN_COMPACT: usize = 1 << 16;

    fn push(&mut self, (s, e): Interval) {
        if let Some(last) = self.xs.last_mut() {
            if s >= last.0 && s <= last.1.saturating_add(1) {
                last.1 = max(last.1, e);
                return;
            }
            if s < last.0 {
                self.merged = false;
            }
        }
        self.xs.push((s, e));
        if !self.merged && self.xs.len() >= self.compact_at {
            self.compact();
        }
    }

    fn compact(&mut self) {
        merge_in_place(&mut self.xs);
        self.merged = true;
        self.compact_at = max(2 * self.xs.len(), Self::MIN_COMPACT);
    }

    /// The sorted, merged intervals.
    fn finish(mut self) -> Vec<Interval> {
        if !self.merged {
            self.compact();
        }
        self.xs.shrink_to_fit();
        self.xs
    }
}

fn intersect_intervals(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let (as_, ae) = a[i];
        let (bs, be) = b[j];
        let s = max(as_, bs);
        let e = min(ae, be);
        if s <= e {
            out.push((s, e));
        }
        if ae < be {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn subtract_intervals(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    let mut j = 0;
    for &(as_, ae) in a {
        let mut cur_s = as_;
        while j < b.len() && b[j].1 < cur_s {
            j += 1;
        }
        let mut k = j;
        while k < b.len() && b[k].0 <= ae {
            let (bs, be) = b[k];
            if bs > cur_s {
                out.push((cur_s, bs - 1));
            }
            if be >= ae {
                cur_s = ae.saturating_add(1);
                break;
            }
            cur_s = be.saturating_add(1);
            k += 1;
        }
        if cur_s <= ae {
            out.push((cur_s, ae));
        }
    }
    out
}

/// Positions `>= 1` not covered by the sorted, merged `intervals`.
fn complement_intervals(intervals: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::with_capacity(intervals.len() + 1);
    let mut next = 1u64;
    for &(s, e) in intervals {
        if s > next {
            out.push((next, s - 1));
        }
        if e == u64::MAX {
            return out;
        }
        next = max(next, e + 1);
    }
    out.push((next, u64::MAX));
    out
}

/// The rows of one or more mask files, merged as they are read.
#[derive(Debug, Default)]
struct MaskAcc {
    by_chrom: HashMap<String, IntervalAcc>,
    global: IntervalAcc,
}

impl MaskAcc {
    /// Add the rows of `path`. Rows of 3 or more columns are BED (`chrom
    /// start end`, 0-based half-open) and apply to their chromosome; rows of
    /// exactly 2 columns are `start end`, 1-based inclusive, and apply to
    /// every chromosome. Blank lines and lines starting with `#` are skipped.
    fn add_file(&mut self, path: &Path) -> Result<()> {
        let mut reader = open_reader(path)?;
        let mut line = Vec::<u8>::new();
        let mut row_no = 0usize;
        // The chromosome of the last BED row and its intervals, out of the
        // map: consecutive rows of one chromosome need no lookup.
        let mut cur: Option<(String, IntervalAcc)> = None;
        let bad_row = |row_no: usize, what: &str| {
            anyhow::anyhow!("invalid mask row {row_no} in {}: {what}", path.display())
        };
        let number = |field: &[u8], row_no: usize, what: &str| -> Result<u64> {
            std::str::from_utf8(field)
                .ok()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| bad_row(row_no, &format!("bad {what} '{}'", lossy(field))))
        };
        loop {
            line.clear();
            let n = reader
                .read_until(b'\n', &mut line)
                .with_context(|| format!("failed to read mask {path:?}"))?;
            if n == 0 {
                break;
            }
            row_no += 1;
            let s = line.trim_ascii();
            if s.is_empty() || s[0] == b'#' {
                continue;
            }
            let mut cols = s.split(u8::is_ascii_whitespace).filter(|c| !c.is_empty());
            let (Some(c0), Some(c1)) = (cols.next(), cols.next()) else {
                return Err(bad_row(row_no, "expected 2 or 3+ columns"));
            };
            let Some(c2) = cols.next() else {
                // Two-column mask format: start end (1-based inclusive).
                let start = number(c0, row_no, "start")?;
                let end = number(c1, row_no, "end")?;
                if start == 0 || end < start {
                    return Err(bad_row(row_no, "start/end out of range"));
                }
                self.global.push((start, end));
                continue;
            };
            // BED: chrom start end, start 0-based and end 1-based exclusive.
            let start0 = number(c1, row_no, "start")?;
            let end_exclusive = number(c2, row_no, "end")?;
            if end_exclusive <= start0 {
                return Err(bad_row(row_no, "end must be > start"));
            }
            let interval = (start0 + 1, end_exclusive);
            match &mut cur {
                Some((name, acc)) if name.as_bytes() == c0 => acc.push(interval),
                _ => {
                    let name = std::str::from_utf8(c0).map_err(|_| {
                        let what = format!(
                            "the chromosome name '{}' is not valid UTF-8",
                            c0.escape_ascii()
                        );
                        bad_row(row_no, &what)
                    })?;
                    if let Some((done, acc)) = cur.take() {
                        self.by_chrom.insert(done, acc);
                    }
                    let mut acc = self.by_chrom.remove(name).unwrap_or_default();
                    acc.push(interval);
                    cur = Some((name.to_string(), acc));
                }
            }
        }
        if let Some((done, acc)) = cur {
            self.by_chrom.insert(done, acc);
        }
        Ok(())
    }

    fn finish(self) -> MaskFile {
        MaskFile {
            by_chrom: self
                .by_chrom
                .into_iter()
                .map(|(chrom, acc)| (chrom, acc.finish()))
                .collect(),
            global: self.global.finish(),
        }
    }
}

/// One mask (or the union of several), as sorted, merged intervals.
#[derive(Debug, Clone, Default)]
struct MaskFile {
    by_chrom: HashMap<String, Vec<Interval>>,
    global: Vec<Interval>, // 2-col mask lines apply globally
}

impl MaskFile {
    fn union_from(paths: &[PathBuf]) -> Result<Self> {
        let mut acc = MaskAcc::default();
        for p in paths {
            acc.add_file(p)?;
        }
        Ok(acc.finish())
    }

    /// The intervals that apply to `chrom`: its own rows and the
    /// chromosome-less ones, borrowed unless there are both.
    fn intervals_for_chrom(&self, chrom: &str) -> Cow<'_, [Interval]> {
        let own = self.by_chrom.get(chrom).map_or(&[][..], Vec::as_slice);
        if self.global.is_empty() {
            Cow::Borrowed(own)
        } else if own.is_empty() {
            Cow::Borrowed(&self.global)
        } else {
            Cow::Owned(merge_intervals([own, &self.global].concat()))
        }
    }
}

/// The sorted, merged positions of one chromosome outside the callable set
/// ([`MaskEngine::uncallable_for_chrom`]), visited in non-decreasing order.
#[derive(Debug)]
pub(crate) struct Uncallable {
    xs: Vec<Interval>,
    /// First interval that can still reach the positions visited.
    next: usize,
}

impl Uncallable {
    /// The first interval that ends at or after `pos`, if any.
    fn seek(&mut self, pos: u64) -> Option<Interval> {
        while self.xs.get(self.next).is_some_and(|&(_, e)| e < pos) {
            self.next += 1;
        }
        self.xs.get(self.next).copied()
    }

    /// Whether `pos` is callable.
    pub(crate) fn is_callable(&mut self, pos: u64) -> bool {
        !self.seek(pos).is_some_and(|(s, _)| s <= pos)
    }
}

/// The warning for VCF chromosomes that no positive mask has a row for.
pub(crate) fn outside_mask_hint(names: &[String]) -> Option<String> {
    (!names.is_empty()).then(|| {
        format!(
            "the callable mask has no interval on {} ({} of the VCF's chromosomes), so every \
             position there is missing; check that the chromosome names in the mask match \
             the VCF (e.g. 'chr1' vs '1')",
            list_names(names),
            names.len()
        )
    })
}

/// The warning for VCF chromosomes that some positive masks have rows for
/// and others do not.
pub(crate) fn outside_some_masks_hint(names: &[String]) -> Option<String> {
    (!names.is_empty()).then(|| {
        format!(
            "some of the callable masks have intervals on {} ({} of the VCF's chromosomes) and \
             the others do not, so every position there is missing: the masks given with \
             --vcf-mask are intersected (a position must be in every one). To combine \
             per-chromosome masks, concatenate them into one file (e.g. cat mask_chr*.bed > \
             mask.bed) and give that file once",
            list_names(names),
            names.len()
        )
    })
}

/// Up to five names, comma-separated, then `...` if there are more.
pub(crate) fn list_names(names: &[String]) -> String {
    let shown = names[..names.len().min(5)].join(", ");
    if names.len() > 5 {
        format!("{shown}, ...")
    } else {
        shown
    }
}

/// The callable set: the intersection of the positive masks (every position
/// when there is none) minus the union of the negative masks.
#[derive(Debug, Clone)]
pub(crate) struct MaskEngine {
    positives: Vec<MaskFile>,
    negatives: MaskFile,
}

impl MaskEngine {
    pub(crate) fn new(mask_paths: &[PathBuf], negative_mask_paths: &[PathBuf]) -> Result<Self> {
        Ok(Self {
            positives: mask_paths
                .iter()
                .map(|p| MaskFile::union_from(std::slice::from_ref(p)))
                .collect::<Result<_>>()?,
            negatives: MaskFile::union_from(negative_mask_paths)?,
        })
    }

    pub(crate) fn has_positive(&self) -> bool {
        !self.positives.is_empty()
    }

    /// Unless every positive mask has a row that applies to `chrom`, the
    /// whole chromosome is uncallable: add it to `no_mask` when none has
    /// (usually a naming mismatch, `chr1` against `1`), or to `some_masks`
    /// when some have (the masks are intersected, as when per-chromosome
    /// masks are given as separate masks).
    pub(crate) fn note_coverage(
        &self,
        chrom: &str,
        no_mask: &mut Vec<String>,
        some_masks: &mut Vec<String>,
    ) {
        let with_rows = self
            .positives
            .iter()
            .filter(|m| !m.global.is_empty() || m.by_chrom.contains_key(chrom))
            .count();
        if with_rows == self.positives.len() {
            return;
        }
        let list = if with_rows == 0 { no_mask } else { some_masks };
        list.push(chrom.to_string());
    }

    /// The positions of `chrom` that are outside the callable set. Only
    /// this list is built: the masks' own intervals are not copied unless
    /// several of them have to be combined.
    pub(crate) fn uncallable_for_chrom(&self, chrom: &str) -> Uncallable {
        Uncallable {
            xs: self.uncallable_intervals(chrom),
            next: 0,
        }
    }

    fn uncallable_intervals(&self, chrom: &str) -> Vec<Interval> {
        let neg = self.negatives.intervals_for_chrom(chrom);
        let Some((first, rest)) = self.positives.split_first() else {
            return neg.into_owned();
        };
        let mut callable = first.intervals_for_chrom(chrom);
        for m in rest {
            if callable.is_empty() {
                break;
            }
            callable = Cow::Owned(intersect_intervals(
                &callable,
                &m.intervals_for_chrom(chrom),
            ));
        }
        if !neg.is_empty() {
            callable = Cow::Owned(subtract_intervals(&callable, &neg));
        }
        complement_intervals(&callable)
    }
}

fn resolve_sample_index(header_fields: &[&str], sample_name: Option<&str>) -> Result<usize> {
    if header_fields.len() < 10 {
        bail!("VCF must contain at least one sample column");
    }
    let samples = &header_fields[9..];
    if let Some(name) = sample_name {
        return samples
            .iter()
            .position(|&s| s == name)
            .with_context(|| format!("VCF sample '{name}' not found in header"));
    }
    if samples.len() == 1 {
        return Ok(0);
    }
    bail!(
        "VCF has multiple samples ({}) - please specify --vcf-sample",
        samples.len()
    )
}

/// Options for [`read_vcf_with`].
#[derive(Debug, Clone)]
pub struct VcfOptions {
    /// Number of bases per output bin (`fq2psmcfa -s`).
    pub bin_size: usize,
    /// Split each sequence into HMM rows of at most this many bins.
    pub batch_size: Option<usize>,
    /// Sample column to read; required when the VCF has several samples.
    pub sample: Option<String>,
    /// Callable-site masks. A position is callable when every positive mask
    /// contains it (every position is callable when there is none) and no
    /// negative mask does. Rows with 3 or more columns are BED
    /// (`chrom start end`, 0-based half-open) and apply to that chromosome;
    /// rows with exactly 2 columns are `start end`, 1-based inclusive, and
    /// apply to every chromosome. Rows may come in any order, and
    /// overlapping or adjacent rows are merged as they are read, so a mask
    /// of one row per callable site takes the memory of its merged
    /// intervals, not of its rows.
    pub masks: Vec<PathBuf>,
    /// Masks of positions to exclude, in the same formats as `masks`.
    pub negative_masks: Vec<PathBuf>,
    /// Use records whose FILTER is neither `PASS` nor `.`; by default they
    /// make their position missing.
    pub keep_filtered: bool,
    /// End each sequence at its chromosome's `##contig` length, when the
    /// header gives one, instead of at the last record. The positions after
    /// the last record are then classified like any other (callable ->
    /// homozygous, else missing). A record beyond the length, or a length
    /// above 2^36 (no real chromosome comes close), is an error.
    pub to_contig_end: bool,
    /// Read the VCF as a gVCF: a position that no record covers (by its
    /// POS, its REF span or, for a gVCF block, its INFO/END) is missing,
    /// since the caller did not look at it, rather than called homozygous
    /// when the masks allow it. [`crate::io::input::load`] sets it when the
    /// records it inspects include gVCF blocks
    /// ([`crate::io::input::VcfModeChoice::gvcf_blocks`]).
    pub gvcf: bool,
}

impl Default for VcfOptions {
    fn default() -> Self {
        Self {
            bin_size: 100,
            batch_size: None,
            sample: None,
            masks: Vec::new(),
            negative_masks: Vec::new(),
            keep_filtered: false,
            to_contig_end: true,
            gvcf: false,
        }
    }
}

/// What [`read_vcf_with`] read, for reporting and sanity warnings.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct VcfSummary {
    /// Data records (non-header lines).
    pub records: u64,
    /// Heterozygous SNP positions that reached the output as heterozygous
    /// sites (callable and not overridden by a missing record).
    pub het_snps: u64,
    /// Records with a heterozygous SNP genotype, used or not.
    pub het_snp_records: u64,
    /// Records with a homozygous (including hom-alt) or haploid genotype.
    pub hom_calls: u64,
    /// Records with a missing or partial genotype (`.`, `./.`, `./1`, no GT).
    pub missing_genotypes: u64,
    /// Records whose FILTER is neither `PASS` nor `.`. Unless
    /// [`VcfOptions::keep_filtered`] is set they are missing and not counted
    /// in the genotype counts.
    pub filtered_records: u64,
    /// Heterozygous records with an allele that is not a single base (indels,
    /// `*`, symbolic alleles); their REF span is missing.
    pub het_non_snp_records: u64,
    /// Records whose POS lies outside the callable set.
    pub records_outside_mask: u64,
    /// Positions carrying more than one record.
    pub multi_record_positions: u64,
    /// Output sequences (chromosomes with at least one record).
    pub sequences: usize,
    /// Output bins over all sequences.
    pub bins: usize,
    /// Positions covered by the output sequences.
    pub sites: u64,
    /// Of `sites`, the positions classified as missing.
    pub missing_sites: u64,
    /// Whether a positive (callable) mask was given. Without one, every
    /// position without a record counts as a called homozygous site.
    pub positive_mask: bool,
    /// Chromosomes with records that no positive mask has a row for, so
    /// that they are entirely missing (usually a naming mismatch such as
    /// `chr1` against `1`).
    pub chroms_outside_mask: Vec<String>,
    /// Chromosomes with records that some positive masks have rows for and
    /// others do not, so that they are entirely missing too: the masks are
    /// intersected (as when per-chromosome masks are given separately).
    pub chroms_outside_some_masks: Vec<String>,
    /// Chromosomes whose sequence was extended to their `##contig` length.
    pub contig_lengths_used: Vec<String>,
    /// Read depth of the sample over all data records, for QC (the bins do
    /// not depend on it): FORMAT/DP of the sample, or INFO/DP when FORMAT
    /// has no DP key and the VCF has this one sample only (INFO/DP sums over
    /// the samples). A DP of `.` or a FORMAT with no DP in a multi-sample
    /// VCF counts as a record without DP.
    pub depth: DepthStats,
    /// Of the records counted in `depth`, those whose DP is INFO/DP.
    pub depth_from_info: u64,
    /// gVCF reference and no-call blocks: records whose ALT alleles are all
    /// `.`, `<*>`, `<NON_REF>` or `<X>` and whose INFO/END is at or after
    /// POS (clamped to the `##contig` length). The genotype of such a record
    /// holds for its whole span.
    pub gvcf_blocks: u64,
    /// Of `gvcf_blocks`, those without a usable call (filtered, a missing
    /// genotype, or heterozygous for the symbolic allele), all of whose
    /// positions are missing.
    pub gvcf_unusable_blocks: u64,
    /// Whether the VCF was read as a gVCF ([`VcfOptions::gvcf`]).
    pub read_as_gvcf: bool,
    /// Read as a gVCF, the positions of `sites` that no record covers,
    /// which are missing; included in `missing_sites`.
    pub uncovered_sites: u64,
}

impl VcfSummary {
    /// Warnings worth showing to the user about how the input was read.
    /// Without a positive mask, the first one says so.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.positive_mask && !self.read_as_gvcf {
            out.push(
                "no callable mask (--vcf-mask) was given, so every position without a VCF \
                 record counts as a called homozygous site; for a variants-only VCF (the \
                 GATK/bcftools default) this hides all missing data, which pulls theta and \
                 Ne down, while false heterozygous calls that a mask would remove push them \
                 up. Pass a callable-sites BED with --vcf-mask, or use an all-sites VCF."
                    .to_string(),
            );
        }
        if self.gvcf_blocks > 0 || self.read_as_gvcf {
            let uncovered = if self.read_as_gvcf {
                format!(
                    ", and so are the positions that no record or block covers ({} here; \
                     bcftools writes nothing where there are no reads)",
                    self.uncovered_sites
                )
            } else {
                "; positions that no record or block covers count as called, as in a VCF \
                 without blocks"
                    .to_string()
            };
            out.push(format!(
                "the VCF is a gVCF: {} records are reference or no-call blocks (INFO/END). \
                 Blocks without a usable call (filtered or no genotype; {} here) are missing \
                 over their whole span{uncovered}, but a reference block counts as called \
                 homozygous over its span whatever its depth and genotype quality (GATK writes \
                 stretches without reads as 0/0 blocks). Unless the callable mask (--vcf-mask) \
                 was made from this gVCF's blocks with depth and quality bounds, make it that \
                 way and give the variant records alone (bcftools view -v snps,indels), as \
                 docs/preprocessing.md describes",
                self.gvcf_blocks, self.gvcf_unusable_blocks
            ));
        }
        out.extend(self.mask_coverage_hints());
        out
    }

    /// The warnings about chromosomes the callable masks leave out entirely.
    fn mask_coverage_hints(&self) -> impl Iterator<Item = String> {
        [
            outside_mask_hint(&self.chroms_outside_mask),
            outside_some_masks_hint(&self.chroms_outside_some_masks),
        ]
        .into_iter()
        .flatten()
    }
}

/// The result of [`read_vcf_with`].
#[derive(Debug, Clone)]
pub struct VcfRead {
    pub obs: Observations,
    pub summary: VcfSummary,
}

/// Per-position class, in precedence order: a position takes the largest
/// class among its records, its mask state and any covering REF span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Site {
    Hom,
    Het,
    Missing,
}

/// The columns of a data line that the reader needs, borrowed from the line.
struct Record<'a> {
    chrom: &'a [u8],
    pos: u64,
    ref_allele: &'a [u8],
    alt: &'a [u8],
    filter: &'a [u8],
    info: &'a [u8],
    format: &'a [u8],
    sample: &'a [u8],
}

impl Record<'_> {
    /// The last position of the REF allele.
    fn ref_end(&self) -> u64 {
        self.pos + max(self.ref_allele.len(), 1) as u64 - 1
    }
}

/// Finds the sample's read depth for [`VcfSummary::depth`]. The FORMAT
/// column rarely changes, so the position of its DP key is kept for the last
/// FORMAT seen.
#[derive(Default)]
struct DepthReader {
    format: Vec<u8>,
    dp_index: Option<usize>,
}

impl DepthReader {
    /// The sample's depth, and whether it is INFO/DP.
    fn depth(&mut self, rec: &Record<'_>, single_sample: bool) -> (Option<u64>, bool) {
        if rec.format != self.format.as_slice() {
            self.format.clear();
            self.format.extend_from_slice(rec.format);
            self.dp_index = rec.format.split(|&b| b == b':').position(|k| k == b"DP");
        }
        if let Some(i) = self.dp_index {
            let value = rec.sample.split(|&b| b == b':').nth(i);
            return (value.and_then(parse_u64), false);
        }
        if !single_sample {
            return (None, false);
        }
        (info_value(rec.info, b"DP").and_then(parse_u64), true)
    }
}

fn lossy(b: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(b)
}

fn parse_u64(b: &[u8]) -> Option<u64> {
    // 19 digits always fit in a u64.
    if b.is_empty() || b.len() > 19 {
        return None;
    }
    let mut v = 0u64;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + u64::from(c - b'0');
    }
    Some(v)
}

/// The POS column of data row `row_no`: a number of at most [`MAX_POS`].
pub(crate) fn parse_pos(b: &[u8], row_no: usize) -> Result<u64> {
    let pos = parse_u64(b)
        .with_context(|| format!("invalid VCF row {row_no}: bad POS '{}'", lossy(b)))?;
    if pos > MAX_POS {
        bail!("invalid VCF row {row_no}: POS {pos} is beyond the supported maximum {MAX_POS}");
    }
    Ok(pos)
}

fn parse_record(line: &[u8], sample_idx: usize, row_no: usize) -> Result<Record<'_>> {
    let mut it = line.split(|&b| b == b'\t');
    let mut cols: [&[u8]; 10] = [&[]; 10];
    for slot in cols.iter_mut() {
        *slot = it
            .next()
            .with_context(|| format!("invalid VCF row {row_no}: expected >= 10 columns"))?;
    }
    let sample = if sample_idx == 0 {
        cols[9]
    } else {
        it.nth(sample_idx - 1).with_context(|| {
            format!(
                "invalid VCF row {}: missing sample column index {}",
                row_no,
                9 + sample_idx
            )
        })?
    };
    let pos = parse_pos(cols[1], row_no)?;
    if pos == 0 {
        bail!("invalid VCF row {}: POS must be >= 1", row_no);
    }
    Ok(Record {
        chrom: cols[0],
        pos,
        ref_allele: cols[3],
        alt: cols[4],
        filter: cols[6],
        info: cols[7],
        format: cols[8],
        sample,
    })
}

enum Genotype {
    /// Missing or partial (`.`, `./.`, `./1`), or no GT field.
    Missing,
    /// Identical allele strings, or a haploid call.
    Hom,
    /// Allele strings differ and each is a single A/C/G/T: the first allele
    /// of the GT and the first one that differs from it, in GT order.
    HetSnp([u8; 2]),
    /// Allele strings differ and one is an indel, `*` or symbolic allele.
    HetOther,
}

fn is_base(allele: &[u8]) -> bool {
    matches!(
        allele,
        [b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't']
    )
}

fn allele<'a>(rec: &Record<'a>, idx: usize) -> Option<&'a [u8]> {
    if idx == 0 {
        return Some(rec.ref_allele);
    }
    if rec.alt == b"." {
        return None;
    }
    rec.alt.split(|&b| b == b',').nth(idx - 1)
}

fn genotype(rec: &Record<'_>, row_no: usize) -> Result<Genotype> {
    let Some(gt_idx) = rec.format.split(|&b| b == b':').position(|k| k == b"GT") else {
        return Ok(Genotype::Missing);
    };
    let Some(gt) = rec.sample.split(|&b| b == b':').nth(gt_idx) else {
        return Ok(Genotype::Missing);
    };
    // VCF 4.4 allows an explicit phasing prefix on the first allele.
    let gt = gt
        .strip_prefix(b"|")
        .or(gt.strip_prefix(b"/"))
        .unwrap_or(gt);
    if gt.is_empty() {
        return Ok(Genotype::Missing);
    }
    let mut first: Option<&[u8]> = None;
    let mut second: Option<&[u8]> = None;
    let mut all_bases = true;
    for a in gt.split(|&b| b == b'/' || b == b'|') {
        if a == b"." {
            return Ok(Genotype::Missing);
        }
        let idx = parse_u64(a)
            .with_context(|| format!("invalid VCF row {}: bad GT '{}'", row_no, lossy(gt)))?;
        let al = usize::try_from(idx)
            .ok()
            .and_then(|i| allele(rec, i))
            .with_context(|| {
                format!(
                    "invalid VCF row {}: GT '{}' refers to allele {} but ALT is '{}'",
                    row_no,
                    lossy(gt),
                    idx,
                    lossy(rec.alt)
                )
            })?;
        all_bases &= is_base(al);
        match first {
            None => first = Some(al),
            Some(f) if second.is_none() && !f.eq_ignore_ascii_case(al) => second = Some(al),
            Some(_) => {}
        }
    }
    // Comparing allele strings rather than indices keeps e.g. `1/2` with
    // ALT `C,C` homozygous.
    Ok(match (first, second) {
        (Some(a), Some(b)) if all_bases => Genotype::HetSnp([a[0], b[0]]),
        (_, Some(_)) => Genotype::HetOther,
        _ => Genotype::Hom,
    })
}

/// What one record says about its position.
struct Class {
    site: Site,
    /// The last position of the span the record makes missing, beyond POS:
    /// the REF span of a heterozygous non-SNP, or a gVCF block without a
    /// usable call.
    span_end: Option<u64>,
    /// For a heterozygous SNP, its two alleles in GT order.
    alleles: Option<[u8; 2]>,
}

/// The value of the INFO key `key`, if INFO has it.
fn info_value<'a>(info: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    info.split(|&b| b == b';')
        .find_map(|kv| kv.strip_prefix(key)?.strip_prefix(b"="))
}

/// The END of a gVCF reference or no-call block, from a record's POS, ALT
/// and INFO: a record whose ALT alleles all stand for "no other allele"
/// (`.`, or the symbolic `<*>` of bcftools and DeepVariant, `<NON_REF>` of
/// GATK, `<X>` of old bcftools) and whose INFO/END is at or after POS. Such
/// a record states one genotype for every position of `POS..=END`.
pub(crate) fn gvcf_block_end(pos: u64, alt: &[u8], info: &[u8]) -> Option<u64> {
    let reference_only = alt
        .split(|&b| b == b',')
        .all(|a| matches!(a, b"." | b"<*>" | b"<NON_REF>" | b"<X>"));
    if !reference_only {
        return None;
    }
    let end = info_value(info, b"END").and_then(parse_u64)?;
    (end >= pos).then_some(end)
}

/// Classify one record. `block_end` is the END of a gVCF block
/// ([`gvcf_block_end`]), already clamped to the sequence.
fn classify(
    rec: &Record<'_>,
    keep_filtered: bool,
    block_end: Option<u64>,
    row_no: usize,
    s: &mut VcfSummary,
) -> Result<Class> {
    let class = |site, span_end, alleles| Class {
        site,
        span_end,
        alleles,
    };
    s.gvcf_blocks += u64::from(block_end.is_some());
    // A block without a usable call leaves its whole span uncalled.
    let unusable_block = |s: &mut VcfSummary| {
        s.gvcf_unusable_blocks += u64::from(block_end.is_some());
        block_end
    };
    if rec.filter != b"PASS" && rec.filter != b"." {
        s.filtered_records += 1;
        if !keep_filtered {
            return Ok(class(Site::Missing, unusable_block(s), None));
        }
    }
    Ok(match genotype(rec, row_no)? {
        Genotype::Missing => {
            s.missing_genotypes += 1;
            class(Site::Missing, unusable_block(s), None)
        }
        Genotype::Hom => {
            s.hom_calls += 1;
            class(Site::Hom, None, None)
        }
        Genotype::HetSnp(alleles) => {
            s.het_snp_records += 1;
            class(Site::Het, None, Some(alleles))
        }
        // Not a SNP: which bases differ is unknown, so the whole REF allele
        // is unreliable for a model of point mutations.
        Genotype::HetOther => {
            s.het_non_snp_records += 1;
            let span_end = max(rec.ref_end(), unusable_block(s).unwrap_or(0));
            class(Site::Missing, Some(span_end), None)
        }
    })
}

struct Pending {
    pos: u64,
    site: Site,
    n_records: u64,
    /// Alleles of the first heterozygous SNP record at `pos`.
    alleles: Option<[u8; 2]>,
}

/// Streams the per-base states of one chromosome, in position order, into a
/// [`SeqBuilder`]. Positions before `cursor` are binned; the records at
/// `pending` may still be joined by further records at the same POS.
struct ChromBuilder {
    name: Vec<u8>,
    seq: SeqBuilder,
    uncallable: Uncallable,
    cursor: u64,
    /// Positions up to here are missing because of an earlier REF span or
    /// gVCF block.
    miss_until: u64,
    /// Positions up to here are covered by a record: its POS, its REF span
    /// or its gVCF block.
    covered_until: u64,
    /// Whether positions that no record covers are missing (a gVCF).
    uncovered_missing: bool,
    pending: Option<Pending>,
    last_pos: u64,
    /// The last END of a gVCF block: a sequence that does not end at its
    /// contig length ends at the last record or the last block end.
    last_block_end: u64,
    /// Where the sequence ends when it does not end at the last record.
    contig_len: Option<u64>,
}

impl ChromBuilder {
    fn new(
        name: &[u8],
        uncallable: Uncallable,
        contig_len: Option<u64>,
        opts: &VcfOptions,
        collect_het: bool,
    ) -> Self {
        Self {
            name: name.to_vec(),
            seq: SeqBuilder::new(opts.bin_size, collect_het),
            uncallable,
            cursor: 1,
            miss_until: 0,
            covered_until: 0,
            uncovered_missing: opts.gvcf,
            pending: None,
            last_pos: 0,
            last_block_end: 0,
            contig_len,
        }
    }

    fn missing(&mut self, n: u64, s: &mut VcfSummary) {
        self.seq.missing(n);
        s.missing_sites += n;
    }

    /// Emit the positions `cursor..=to`, none of which has a record.
    fn fill_gap(&mut self, to: u64, s: &mut VcfSummary) {
        if to < self.cursor {
            return;
        }
        let mut a = self.cursor;
        self.cursor = to + 1;
        if self.miss_until >= a {
            let e = min(to, self.miss_until);
            self.missing(e - a + 1, s);
            a = e + 1;
        }
        // In a gVCF, the positions after the last record's coverage were
        // not looked at.
        let callable_to = if self.uncovered_missing {
            min(to, self.covered_until)
        } else {
            to
        };
        if callable_to < to {
            let from = max(a, callable_to + 1);
            if from <= to {
                let n = to - from + 1;
                s.uncovered_sites += n;
                self.fill_mask(a, callable_to, s);
                self.missing(n, s);
                return;
            }
        }
        self.fill_mask(a, to, s);
    }

    /// Emit the positions `a..=to`, covered by records but without one of
    /// their own, as the masks say: called where callable, else missing.
    fn fill_mask(&mut self, mut a: u64, to: u64, s: &mut VcfSummary) {
        while a <= to {
            match self.uncallable.seek(a) {
                Some((us, ue)) if us <= to => {
                    if us > a {
                        self.seq.called(us - a);
                        a = us;
                    }
                    let e = min(ue, to);
                    self.missing(e - a + 1, s);
                    a = e + 1;
                }
                _ => {
                    self.seq.called(to - a + 1);
                    break;
                }
            }
        }
    }

    fn flush_pending(&mut self, s: &mut VcfSummary) {
        let Some(p) = self.pending.take() else {
            return;
        };
        debug_assert_eq!(self.cursor, p.pos);
        let callable = self.uncallable.is_callable(p.pos);
        if !callable {
            s.records_outside_mask += p.n_records;
        }
        if p.n_records > 1 {
            s.multi_record_positions += 1;
        }
        let site = if !callable || p.pos <= self.miss_until {
            Site::Missing
        } else {
            p.site
        };
        match site {
            Site::Missing => self.missing(1, s),
            Site::Het => {
                let alleles = p.alleles.expect("a heterozygous site has SNP alleles");
                self.seq.het(p.pos, alleles);
                s.het_snps += 1;
            }
            Site::Hom => self.seq.called(1),
        }
        self.cursor = p.pos + 1;
    }

    /// Add a record; `pos` must not be smaller than any earlier record's.
    /// `block_end` is the END of a gVCF block, `ref_end` the last position
    /// of its REF allele.
    fn add_record(
        &mut self,
        pos: u64,
        class: Class,
        (ref_end, block_end): (u64, Option<u64>),
        s: &mut VcfSummary,
    ) {
        self.last_block_end = max(self.last_block_end, block_end.unwrap_or(0));
        match self.pending.as_mut() {
            Some(p) if p.pos == pos => {
                p.site = max(p.site, class.site);
                p.n_records += 1;
                p.alleles = p.alleles.or(class.alleles);
            }
            _ => {
                self.flush_pending(s);
                self.fill_gap(pos - 1, s);
                self.pending = Some(Pending {
                    pos,
                    site: class.site,
                    n_records: 1,
                    alleles: class.alleles,
                });
            }
        }
        if let Some(e) = class.span_end {
            self.miss_until = max(self.miss_until, e);
        }
        self.covered_until = max(self.covered_until, max(ref_end, block_end.unwrap_or(0)));
        self.last_pos = pos;
    }

    /// Emit everything up to the end of the sequence and return it, with
    /// the callable bases of each bin: the bin size minus the bin's
    /// positions classified missing ([`crate::io::callable`]).
    fn finish(mut self, s: &mut VcfSummary) -> BinnedSeq {
        let end = self
            .contig_len
            .unwrap_or(max(self.last_pos, self.last_block_end));
        self.flush_pending(s);
        self.fill_gap(end, s);
        s.sites += end;
        self.seq.finish(lossy(&self.name).into_owned(), end)
    }
}

/// `(ID, length)` of a `##contig=<ID=...,length=...,...>` header line. A
/// length of 20 or more digits comes back as `u64::MAX`; the caller rejects
/// lengths above [`MAX_POS`] for the chromosomes it extends.
fn parse_contig_header(line: &[u8]) -> Option<(&[u8], u64)> {
    let body = line.strip_prefix(b"##contig=<")?;
    let body = body.strip_suffix(b">").unwrap_or(body);
    let (mut id, mut len) = (None, None);
    let mut in_quotes = false;
    // Split on the commas outside quoted values such as description="a, b".
    for field in body.split(|&b| {
        if b == b'"' {
            in_quotes = !in_quotes;
        }
        b == b',' && !in_quotes
    }) {
        if let Some(v) = field.strip_prefix(b"ID=") {
            id = Some(v);
        } else if let Some(v) = field.strip_prefix(b"length=") {
            len = if !v.is_empty() && v.iter().all(u8::is_ascii_digit) {
                Some(parse_u64(v).unwrap_or(u64::MAX))
            } else {
                None
            };
        }
    }
    let len = len.filter(|&l| l > 0)?;
    Some((id?, len))
}

/// Read one sample of a VCF into binned observations; see the module docs for
/// how positions are classified.
///
/// Each chromosome with at least one record becomes one sequence covering
/// positions `1..=L`, where `L` is the chromosome's `##contig` length when
/// [`VcfOptions::to_contig_end`] is set and the header gives one, and the last
/// record's POS otherwise. Chromosomes without records produce no sequence.
/// Records must be sorted by position within a chromosome (several records at
/// one POS are combined), and each chromosome must form one contiguous block,
/// as in any `bcftools sort`ed or indexed VCF. POS, and the `##contig` lengths
/// that are used, must not exceed 2^36, and chromosome names must be valid
/// UTF-8, as VCF requires.
///
/// The observations carry the callable bases of each bin
/// ([`Observations::callable`]): the bin size minus the bin's positions
/// classified missing. Bins of more than
/// [`MAX_COUNTED_BIN_SIZE`](crate::io::MAX_COUNTED_BIN_SIZE) bases have
/// none.
pub fn read_vcf_with(path: &Path, opts: &VcfOptions) -> Result<VcfRead> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    let (seqs, summary) = read_vcf_seqs_from(open_reader(path)?, path, opts, false)?;
    if summary.het_snps == 0 {
        bail!("{}", no_het_snp_message(&summary, opts.keep_filtered));
    }
    let obs = chunk_sequences(
        Sequences::of_binned(opts.bin_size, seqs),
        opts.batch_size,
        "no valid segregating sites found in VCF input",
    )?;
    Ok(VcfRead { obs, summary })
}

/// The error text for a VCF without any usable heterozygous SNP.
pub(crate) fn no_het_snp_message(summary: &VcfSummary, keep_filtered: bool) -> String {
    let mut hint = String::new();
    // Filtered records are not classified, so `het_snp_records` cannot
    // show the heterozygous calls among them.
    if summary.filtered_records > 0 && !keep_filtered {
        hint.push_str(&format!(
            "; {} a FILTER other than PASS or '.' and are treated as missing: keep \
             them with --vcf-keep-filtered if the filter does not apply to this analysis",
            counted(
                summary.filtered_records as usize,
                "record has",
                "records have"
            )
        ));
    }
    for h in summary.mask_coverage_hints() {
        hint.push_str(&format!("; {h}"));
    }
    format!(
        "no heterozygous SNP at a callable position found in VCF input ({}, {}, {} \
         outside the callable set){hint}",
        counted(summary.records as usize, "record", "records"),
        counted(
            summary.het_snp_records as usize,
            "heterozygous SNP record",
            "heterozygous SNP records"
        ),
        count(summary.records_outside_mask)
    )
}

/// Read every chromosome of one sample of a VCF from `reader`, as
/// [`read_vcf_with`] does, without chunking and without the check for
/// heterozygous SNPs; `path` names the input in messages. With
/// `collect_het`, each sequence carries its heterozygous sites.
pub(crate) fn read_vcf_seqs_from(
    mut reader: impl BufRead,
    path: &Path,
    opts: &VcfOptions,
    collect_het: bool,
) -> Result<(Vec<BinnedSeq>, VcfSummary)> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    let masks = MaskEngine::new(&opts.masks, &opts.negative_masks)?;
    let mut summary = VcfSummary {
        positive_mask: masks.has_positive(),
        read_as_gvcf: opts.gvcf,
        ..VcfSummary::default()
    };

    let mut line = Vec::<u8>::new();
    let mut row_no = 0usize;
    let mut sample_idx: Option<usize> = None;
    let mut single_sample = false;
    let mut depth = DepthCounter::default();
    let mut depth_reader = DepthReader::default();
    let mut contig_lens = HashMap::<Vec<u8>, u64>::new();
    let mut finished_chroms = HashSet::<Vec<u8>>::new();
    let mut cur: Option<ChromBuilder> = None;
    let mut seqs = Vec::<BinnedSeq>::new();

    loop {
        line.clear();
        let n = reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read VCF {path:?}"))?;
        if n == 0 {
            break;
        }
        row_no += 1;
        let s = line.trim_ascii();
        if s.is_empty() {
            continue;
        }
        if s[0] == b'#' {
            if s.starts_with(b"#CHROM") {
                let header = lossy(s);
                let fields: Vec<&str> = header.split('\t').collect();
                sample_idx = Some(resolve_sample_index(&fields, opts.sample.as_deref())?);
                single_sample = fields.len() == 10;
            } else if let Some((id, len)) = parse_contig_header(s) {
                contig_lens.insert(id.to_vec(), len);
            }
            continue;
        }
        let idx = sample_idx.with_context(|| "VCF header '#CHROM' not found before records")?;
        let rec = parse_record(s, idx, row_no)?;
        summary.records += 1;
        let (dp, from_info) = depth_reader.depth(&rec, single_sample);
        depth.add(dp);
        summary.depth_from_info += u64::from(from_info && dp.is_some());

        if cur.as_ref().is_none_or(|c| c.name != rec.chrom) {
            if let Some(done) = cur.take() {
                finished_chroms.insert(done.name.clone());
                seqs.push(done.finish(&mut summary));
            }
            if finished_chroms.contains(rec.chrom) {
                bail!(
                    "invalid VCF row {row_no}: chromosome '{}' appears again after another \
                     chromosome; the VCF must be sorted (e.g. with bcftools sort)",
                    lossy(rec.chrom)
                );
            }
            let contig_len = opts
                .to_contig_end
                .then(|| contig_lens.get(rec.chrom).copied())
                .flatten();
            cur = Some(start_chrom(
                rec.chrom,
                contig_len,
                row_no,
                &masks,
                opts,
                collect_het,
                &mut summary,
            )?);
        }
        let chrom = cur.as_mut().expect("chromosome state was just set");
        if rec.pos < chrom.last_pos {
            bail!(
                "invalid VCF row {row_no}: POS {} on '{}' comes after POS {}; the VCF must be \
                 sorted by position (e.g. with bcftools sort)",
                rec.pos,
                lossy(rec.chrom),
                chrom.last_pos
            );
        }
        if let Some(len) = chrom.contig_len.filter(|&len| rec.pos > len) {
            bail!(
                "invalid VCF row {row_no}: POS {} on '{}' is beyond the ##contig length {len}",
                rec.pos,
                lossy(rec.chrom),
            );
        }
        // A block reaching past the contig (or past any real chromosome)
        // ends there.
        let block_end = gvcf_block_end(rec.pos, rec.alt, rec.info)
            .map(|end| end.min(chrom.contig_len.unwrap_or(MAX_POS)));
        let class = classify(&rec, opts.keep_filtered, block_end, row_no, &mut summary)?;
        chrom.add_record(rec.pos, class, (rec.ref_end(), block_end), &mut summary);
    }
    if let Some(done) = cur.take() {
        seqs.push(done.finish(&mut summary));
    }
    summary.depth = depth.stats();
    summary.sequences = seqs.iter().filter(|s| !s.bins.is_empty()).count();
    summary.bins = seqs.iter().map(|s| s.bins.len()).sum();
    Ok((seqs, summary))
}

/// The builder of chromosome `chrom`, first met at row `row_no`, whose
/// sequence ends at `contig_len` when given; notes in `summary` how the masks
/// cover it and whether its contig length is used.
fn start_chrom(
    chrom: &[u8],
    contig_len: Option<u64>,
    row_no: usize,
    masks: &MaskEngine,
    opts: &VcfOptions,
    collect_het: bool,
    summary: &mut VcfSummary,
) -> Result<ChromBuilder> {
    let name = utf8_chrom(chrom, row_no)?;
    masks.note_coverage(
        name,
        &mut summary.chroms_outside_mask,
        &mut summary.chroms_outside_some_masks,
    );
    if let Some(len) = contig_len {
        if len > MAX_POS {
            let len = if len == u64::MAX {
                "a number of 20 or more digits".to_string()
            } else {
                len.to_string()
            };
            bail!(
                "invalid VCF header: the ##contig length of '{name}' ({len}) is beyond the \
                 supported maximum {MAX_POS}; correct the header, or end each sequence at its \
                 last record instead of the contig length"
            );
        }
        summary.contig_lengths_used.push(name.to_string());
    }
    let uncallable = masks.uncallable_for_chrom(name);
    Ok(ChromBuilder::new(
        chrom,
        uncallable,
        contig_len,
        opts,
        collect_het,
    ))
}

/// [`read_vcf_with`] with the default options for everything not given here.
pub fn read_vcf(
    path: &Path,
    batch_size: Option<usize>,
    bin_size: usize,
    sample_name: Option<&str>,
    mask_paths: &[PathBuf],
    negative_mask_paths: &[PathBuf],
) -> Result<Observations> {
    let opts = VcfOptions {
        bin_size,
        batch_size,
        sample: sample_name.map(str::to_string),
        masks: mask_paths.to_vec(),
        negative_masks: negative_mask_paths.to_vec(),
        ..VcfOptions::default()
    };
    Ok(read_vcf_with(path, &opts)?.obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*, enough for shuffling test rows.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
        }
    }

    #[test]
    fn interval_acc_merges_like_sorting_everything_first() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        // Sorted, nearly sorted and shuffled rows; with more rows than
        // MIN_COMPACT the shuffled ones are compacted on the way.
        for (n, swaps) in [(1_000, 0), (1_000, 50), (300_000, 300_000), (5, 5)] {
            let mut rows: Vec<Interval> = (0..n)
                .map(|i| {
                    let s = 1 + 3 * i + rng.below(4);
                    (s, s + rng.below(5))
                })
                .collect();
            for _ in 0..swaps {
                let (a, b) = (rng.below(n) as usize, rng.below(n) as usize);
                rows.swap(a, b);
            }
            let mut acc = IntervalAcc::default();
            for &r in &rows {
                acc.push(r);
            }
            assert_eq!(acc.finish(), merge_intervals(rows), "n {n}, swaps {swaps}");
        }
        assert!(IntervalAcc::default().finish().is_empty());
    }

    #[test]
    fn interval_acc_keeps_adjacent_rows_as_one_interval() {
        let mut acc = IntervalAcc::default();
        for s in 1..=1_000_000u64 {
            acc.push((s, s));
        }
        assert_eq!(acc.xs, [(1, 1_000_000)]);
        // Reversed rows are merged whenever the list doubles.
        let mut acc = IntervalAcc::default();
        for s in (1..=1_000_000u64).rev() {
            acc.push((s, s));
            assert!(acc.xs.len() <= IntervalAcc::MIN_COMPACT);
        }
        assert_eq!(acc.finish(), [(1, 1_000_000)]);
    }

    #[test]
    fn mask_rows_may_come_in_any_order() {
        let path = std::env::temp_dir().join(format!(
            "psmc_rs_mask_rows_any_order_{}.bed",
            std::process::id()
        ));
        // chr2 comes back after chr1, rows within a chromosome are out of
        // order, and two-column rows apply to every chromosome.
        let text = "# comment\nchr2\t10\t20\nchr1\t5\t6\nchr1\t0\t3\n\n\
                    chr2\t20\t30\n100\t100\nchr1\t3\t5\nchr2\t0\t1\n";
        std::fs::write(&path, text).unwrap();
        let m = MaskFile::union_from(std::slice::from_ref(&path)).unwrap();
        assert_eq!(m.by_chrom["chr1"], [(1, 6)]);
        assert_eq!(m.by_chrom["chr2"], [(1, 1), (11, 30)]);
        assert_eq!(m.global, [(100, 100)]);
        assert_eq!(*m.intervals_for_chrom("chr1"), [(1, 6), (100, 100)]);
        assert_eq!(*m.intervals_for_chrom("chr3"), [(100, 100)]);

        for (text, want) in [
            ("chr1\t5\t6\nchr1\tx\t9\n", "invalid mask row 2"),
            ("chr1\tx\t9\n", "bad start 'x'"),
            ("10\t5\n", "start/end out of range"),
            ("chr1\n", "expected 2 or 3+ columns"),
            ("chr1\t9\t9\n", "end must be > start"),
        ] {
            std::fs::write(&path, text).unwrap();
            let err = MaskFile::union_from(std::slice::from_ref(&path))
                .unwrap_err()
                .to_string();
            assert!(err.contains(want), "{text:?}: {err}");
        }
        let _ = std::fs::remove_file(path);
    }
}
