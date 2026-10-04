//! VCF input for one diploid sample.
//!
//! The reader produces exactly the binned sequence that `fq2psmcfa -s <bin>`
//! produces from a per-base consensus of the selected sample: every position
//! of a chromosome is classified as a called homozygous site, a heterozygous
//! SNP or missing, and the positions are binned with the `fq2psmcfa` rule
//! (see [`Binner`]).
//!
//! A position is missing if it lies outside the callable set (see
//! [`VcfOptions::masks`]) or if one of its records leaves the genotype unknown.
//! Otherwise it is a heterozygous SNP if one of its records is, and a called
//! homozygous site in every other case, including positions without any
//! record. Per record, for the selected sample:
//!
//! - FILTER other than `PASS` or `.`: unknown, unless
//!   [`VcfOptions::keep_filtered`] (then classified like any other record);
//! - missing or partial `GT` (`.`, `./.`, `./1`), or no `GT`: unknown;
//! - heterozygous (the allele strings differ) with every allele a single
//!   `A`/`C`/`G`/`T` (any case): heterozygous SNP;
//! - heterozygous with another allele (indel, `*`, symbolic `<...>`): unknown;
//! - homozygous (identical allele strings, including hom-alt and homozygous
//!   indels) or haploid: called homozygous.
//!
//! An unknown record makes its whole REF span `POS..POS+len(REF)-1` missing
//! (`END` and `SVLEN` are not consulted).
//!
//! The file is streamed, and the work is proportional to the number of records
//! and mask intervals plus the number of output bins, never to the genome
//! length: runs of positions between records are emitted straight from the
//! mask intervals.

use anyhow::{Context, Result, bail};
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::io::Observations;
use crate::io::common::{Binner, chunk_sequences, open_reader};

type Interval = (u64, u64); // inclusive [start, end], 1-based

/// Largest accepted POS. Far beyond any real chromosome, it keeps
/// `POS + len(REF)` and run lengths clear of `u64` overflow.
const MAX_POS: u64 = 1 << 60;

fn merge_intervals(mut xs: Vec<Interval>) -> Vec<Interval> {
    if xs.is_empty() {
        return xs;
    }
    xs.sort_unstable_by_key(|x| (x.0, x.1));
    let mut out = Vec::with_capacity(xs.len());
    let mut cur = xs[0];
    for &(s, e) in &xs[1..] {
        if s <= cur.1.saturating_add(1) {
            cur.1 = max(cur.1, e);
        } else {
            out.push(cur);
            cur = (s, e);
        }
    }
    out.push(cur);
    out
}

fn intersect_intervals(a: &[Interval], b: &[Interval]) -> Vec<Interval> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut out = Vec::<Interval>::new();
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
    if a.is_empty() {
        return Vec::new();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut out = Vec::<Interval>::new();
    let mut j = 0usize;
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

#[derive(Debug, Clone, Default)]
struct MaskFile {
    by_chrom: HashMap<String, Vec<Interval>>,
    global: Vec<Interval>, // 2-col mask lines apply globally
}

impl MaskFile {
    fn is_empty(&self) -> bool {
        self.by_chrom.is_empty() && self.global.is_empty()
    }

    fn merge_all(&mut self) {
        for v in self.by_chrom.values_mut() {
            let merged = merge_intervals(std::mem::take(v));
            *v = merged;
        }
        self.global = merge_intervals(std::mem::take(&mut self.global));
    }

    fn union_from(paths: &[PathBuf]) -> Result<Self> {
        let mut out = MaskFile::default();
        for p in paths {
            let one = parse_mask_file(p)?;
            out.global.extend(one.global);
            for (chrom, intervals) in one.by_chrom {
                out.by_chrom.entry(chrom).or_default().extend(intervals);
            }
        }
        out.merge_all();
        Ok(out)
    }

    fn intervals_for_chrom(&self, chrom: &str) -> Vec<Interval> {
        let mut out = Vec::<Interval>::new();
        if let Some(v) = self.by_chrom.get(chrom) {
            out.extend(v.iter().copied());
        }
        if !self.global.is_empty() {
            out.extend(self.global.iter().copied());
        }
        merge_intervals(out)
    }
}

fn parse_mask_file(path: &Path) -> Result<MaskFile> {
    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut out = MaskFile::default();
    let mut row_no = 0usize;

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read mask {path:?}"))?;
        if n == 0 {
            break;
        }
        row_no += 1;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = s.split_whitespace().collect();
        if cols.len() < 2 {
            bail!(
                "invalid mask row {} in {}: expected 2 or 3+ columns",
                row_no,
                path.display()
            );
        }

        if cols.len() == 2 {
            // Two-column mask format: start end (1-based inclusive).
            let start: u64 = cols[0].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad start '{}'",
                    row_no,
                    path.display(),
                    cols[0]
                )
            })?;
            let end: u64 = cols[1].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad end '{}'",
                    row_no,
                    path.display(),
                    cols[1]
                )
            })?;
            if start == 0 || end < start {
                bail!(
                    "invalid mask row {} in {}: start/end out of range",
                    row_no,
                    path.display()
                );
            }
            out.global.push((start, end));
        } else {
            // BED-like format: chrom start end where start is 0-based and end is 1-based exclusive.
            let chrom = cols[0].to_string();
            let start0: u64 = cols[1].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad start '{}'",
                    row_no,
                    path.display(),
                    cols[1]
                )
            })?;
            let end_exclusive: u64 = cols[2].parse().with_context(|| {
                format!(
                    "invalid mask row {} in {}: bad end '{}'",
                    row_no,
                    path.display(),
                    cols[2]
                )
            })?;
            if end_exclusive <= start0 {
                bail!(
                    "invalid mask row {} in {}: end must be > start",
                    row_no,
                    path.display()
                );
            }
            let start = start0 + 1;
            let end = end_exclusive;
            out.by_chrom.entry(chrom).or_default().push((start, end));
        }
    }

    out.merge_all();
    Ok(out)
}

/// The callable set: the intersection of the positive masks (every position
/// when there is none) minus the union of the negative masks.
#[derive(Debug, Clone)]
struct MaskEngine {
    positives: Vec<MaskFile>,
    negatives: MaskFile,
}

impl MaskEngine {
    fn new(mask_paths: &[PathBuf], negative_mask_paths: &[PathBuf]) -> Result<Self> {
        let mut positives = Vec::<MaskFile>::new();
        for p in mask_paths {
            positives.push(parse_mask_file(p)?);
        }
        let negatives = MaskFile::union_from(negative_mask_paths)?;
        Ok(Self {
            positives,
            negatives,
        })
    }

    fn has_positive(&self) -> bool {
        !self.positives.is_empty()
    }

    /// False when some positive mask has no row that applies to `chrom`,
    /// which leaves the whole chromosome uncallable.
    fn mentions_chrom(&self, chrom: &str) -> bool {
        self.positives
            .iter()
            .all(|m| !m.global.is_empty() || m.by_chrom.contains_key(chrom))
    }

    /// Sorted, merged positions of `chrom` that are outside the callable set.
    fn uncallable_for_chrom(&self, chrom: &str) -> Vec<Interval> {
        let neg = if self.negatives.is_empty() {
            Vec::new()
        } else {
            self.negatives.intervals_for_chrom(chrom)
        };
        let Some((first, rest)) = self.positives.split_first() else {
            return neg;
        };
        let mut callable = first.intervals_for_chrom(chrom);
        for m in rest {
            if callable.is_empty() {
                break;
            }
            callable = intersect_intervals(&callable, &m.intervals_for_chrom(chrom));
        }
        complement_intervals(&subtract_intervals(&callable, &neg))
    }
}

fn resolve_sample_index(header_fields: &[&str], sample_name: Option<&str>) -> Result<usize> {
    if header_fields.len() < 10 {
        bail!("VCF must contain at least one sample column");
    }
    let samples = &header_fields[9..];
    if let Some(name) = sample_name {
        for (i, s) in samples.iter().enumerate() {
            if *s == name {
                return Ok(i);
            }
        }
        bail!("VCF sample '{}' not found in header", name);
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
    /// apply to every chromosome.
    pub masks: Vec<PathBuf>,
    /// Masks of positions to exclude, in the same formats as `masks`.
    pub negative_masks: Vec<PathBuf>,
    /// Use records whose FILTER is neither `PASS` nor `.`; by default they
    /// make their REF span missing.
    pub keep_filtered: bool,
    /// End each sequence at its chromosome's `##contig` length, when the
    /// header gives one, instead of at the last record. The positions after
    /// the last record are then classified like any other (callable ->
    /// homozygous, else missing), and a record beyond the length is an error.
    pub to_contig_end: bool,
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
        }
    }
}

/// What [`read_vcf_with`] read, for reporting and sanity warnings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
    /// Chromosomes with records that some positive mask has no row for, so
    /// that they are entirely missing (usually a naming mismatch such as
    /// `chr1` against `1`).
    pub chroms_outside_mask: Vec<String>,
    /// Chromosomes whose sequence was extended to their `##contig` length.
    pub contig_lengths_used: Vec<String>,
}

impl VcfSummary {
    /// Warnings worth showing to the user about how the input was read.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.positive_mask {
            out.push(
                "no callable mask (--vcf-mask) was given, so every position without a VCF \
                 record counts as a called homozygous site; for a variants-only VCF (the \
                 GATK/bcftools default) this hides all missing data and biases theta and \
                 the Ne estimates low. Pass a callable-sites BED with --vcf-mask, or use an \
                 all-sites VCF."
                    .to_string(),
            );
        }
        if let Some(hint) = self.chroms_outside_mask_hint() {
            out.push(hint);
        }
        out
    }

    fn chroms_outside_mask_hint(&self) -> Option<String> {
        let names = &self.chroms_outside_mask;
        if names.is_empty() {
            return None;
        }
        let shown = names.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
        let more = if names.len() > 5 { ", ..." } else { "" };
        Some(format!(
            "the callable mask has no interval on {} VCF chromosome(s) ({shown}{more}), \
             which are therefore entirely missing; check that the chromosome names in \
             the mask match the VCF (e.g. 'chr1' vs '1')",
            names.len()
        ))
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
    format: &'a [u8],
    sample: &'a [u8],
}

fn lossy(b: &[u8]) -> std::borrow::Cow<'_, str> {
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
    let pos = parse_u64(cols[1])
        .with_context(|| format!("invalid VCF row {}: bad POS '{}'", row_no, lossy(cols[1])))?;
    if pos == 0 {
        bail!("invalid VCF row {}: POS must be >= 1", row_no);
    }
    if pos > MAX_POS {
        bail!("invalid VCF row {}: POS {} is out of range", row_no, pos);
    }
    Ok(Record {
        chrom: cols[0],
        pos,
        ref_allele: cols[3],
        alt: cols[4],
        filter: cols[6],
        format: cols[8],
        sample,
    })
}

enum Genotype {
    /// Missing or partial (`.`, `./.`, `./1`), or no GT field.
    Missing,
    /// Identical allele strings, or a haploid call.
    Hom,
    /// Allele strings differ and each is a single A/C/G/T.
    HetSnp,
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
    let mut all_same = true;
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
            Some(f) => all_same &= f.eq_ignore_ascii_case(al),
        }
    }
    // Comparing allele strings rather than indices keeps e.g. `1/2` with
    // ALT `C,C` homozygous.
    Ok(if all_same {
        Genotype::Hom
    } else if all_bases {
        Genotype::HetSnp
    } else {
        Genotype::HetOther
    })
}

/// Classify one record: its class at POS and, when it leaves the genotype
/// unknown, the last position of its REF span (all of which become missing).
fn classify(
    rec: &Record<'_>,
    keep_filtered: bool,
    row_no: usize,
    s: &mut VcfSummary,
) -> Result<(Site, Option<u64>)> {
    let unknown = (
        Site::Missing,
        Some(rec.pos + max(rec.ref_allele.len(), 1) as u64 - 1),
    );
    if rec.filter != b"PASS" && rec.filter != b"." {
        s.filtered_records += 1;
        if !keep_filtered {
            return Ok(unknown);
        }
    }
    Ok(match genotype(rec, row_no)? {
        Genotype::Missing => {
            s.missing_genotypes += 1;
            unknown
        }
        Genotype::Hom => {
            s.hom_calls += 1;
            (Site::Hom, None)
        }
        Genotype::HetSnp => {
            s.het_snp_records += 1;
            (Site::Het, None)
        }
        // Not a SNP: which bases differ is unknown, so the whole REF allele
        // is unreliable for a model of point mutations.
        Genotype::HetOther => {
            s.het_non_snp_records += 1;
            unknown
        }
    })
}

struct Pending {
    pos: u64,
    site: Site,
    n_records: u64,
}

/// Streams the per-base states of one chromosome, in position order, into a
/// [`Binner`]. Positions before `cursor` are binned; the records at `pending`
/// may still be joined by further records at the same POS.
struct ChromBuilder {
    name: Vec<u8>,
    binner: Binner,
    /// Sorted, merged positions outside the callable set.
    uncallable: Vec<Interval>,
    /// First entry of `uncallable` that can still reach `cursor` or beyond.
    next_unc: usize,
    cursor: u64,
    /// Positions up to here are missing because of an earlier REF span.
    miss_until: u64,
    pending: Option<Pending>,
    last_pos: u64,
    /// Where the sequence ends when it does not end at the last record.
    contig_len: Option<u64>,
}

impl ChromBuilder {
    fn new(
        name: &[u8],
        uncallable: Vec<Interval>,
        contig_len: Option<u64>,
        bin_size: usize,
    ) -> Self {
        Self {
            name: name.to_vec(),
            binner: Binner::new(bin_size),
            uncallable,
            next_unc: 0,
            cursor: 1,
            miss_until: 0,
            pending: None,
            last_pos: 0,
            contig_len,
        }
    }

    fn missing(&mut self, n: u64, s: &mut VcfSummary) {
        self.binner.add_missing(n);
        s.missing_sites += n;
    }

    /// Whether `pos` is callable. Calls must come with non-decreasing `pos`.
    fn is_callable(&mut self, pos: u64) -> bool {
        while self
            .uncallable
            .get(self.next_unc)
            .is_some_and(|&(_, e)| e < pos)
        {
            self.next_unc += 1;
        }
        !self
            .uncallable
            .get(self.next_unc)
            .is_some_and(|&(start, _)| start <= pos)
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
        while a <= to {
            while self
                .uncallable
                .get(self.next_unc)
                .is_some_and(|&(_, e)| e < a)
            {
                self.next_unc += 1;
            }
            match self.uncallable.get(self.next_unc) {
                Some(&(us, ue)) if us <= to => {
                    if us > a {
                        self.binner.add_called(us - a);
                        a = us;
                    }
                    let e = min(ue, to);
                    self.missing(e - a + 1, s);
                    a = e + 1;
                }
                _ => {
                    self.binner.add_called(to - a + 1);
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
        let callable = self.is_callable(p.pos);
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
                self.binner.add_het();
                s.het_snps += 1;
            }
            Site::Hom => self.binner.add_called(1),
        }
        self.cursor = p.pos + 1;
    }

    /// Add a record; `pos` must not be smaller than any earlier record's.
    fn add_record(&mut self, pos: u64, site: Site, span_end: Option<u64>, s: &mut VcfSummary) {
        match self.pending.as_mut() {
            Some(p) if p.pos == pos => {
                p.site = max(p.site, site);
                p.n_records += 1;
            }
            _ => {
                self.flush_pending(s);
                self.fill_gap(pos - 1, s);
                self.pending = Some(Pending {
                    pos,
                    site,
                    n_records: 1,
                });
            }
        }
        if let Some(e) = span_end {
            self.miss_until = max(self.miss_until, e);
        }
        self.last_pos = pos;
    }

    /// Emit everything up to the end of the sequence and return its bins.
    fn finish(&mut self, s: &mut VcfSummary) -> Vec<u8> {
        let end = self.contig_len.unwrap_or(self.last_pos);
        self.flush_pending(s);
        self.fill_gap(end, s);
        s.sites += end;
        self.binner.finish()
    }
}

/// `(ID, length)` of a `##contig=<ID=...,length=...,...>` header line.
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
            len = parse_u64(v);
        }
    }
    let len = len.filter(|&l| (1..=MAX_POS).contains(&l))?;
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
/// as in any `bcftools sort`ed or indexed VCF.
pub fn read_vcf_with(path: &Path, opts: &VcfOptions) -> Result<VcfRead> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    let masks = MaskEngine::new(&opts.masks, &opts.negative_masks)?;
    let mut summary = VcfSummary {
        positive_mask: masks.has_positive(),
        ..VcfSummary::default()
    };

    let mut reader = open_reader(path)?;
    let mut line = Vec::<u8>::new();
    let mut row_no = 0usize;
    let mut sample_idx: Option<usize> = None;
    let mut contig_lens = HashMap::<Vec<u8>, u64>::new();
    let mut finished_chroms = HashSet::<Vec<u8>>::new();
    let mut cur: Option<ChromBuilder> = None;
    let mut seqs = Vec::<Vec<u8>>::new();

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
            } else if let Some((id, len)) = parse_contig_header(s) {
                contig_lens.insert(id.to_vec(), len);
            }
            continue;
        }
        let idx = sample_idx.with_context(|| "VCF header '#CHROM' not found before records")?;
        let rec = parse_record(s, idx, row_no)?;
        summary.records += 1;

        if cur.as_ref().is_none_or(|c| c.name != rec.chrom) {
            if let Some(mut done) = cur.take() {
                seqs.push(done.finish(&mut summary));
                finished_chroms.insert(done.name);
            }
            if finished_chroms.contains(rec.chrom) {
                bail!(
                    "invalid VCF row {}: chromosome '{}' appears again after another \
                     chromosome; the VCF must be sorted (e.g. with bcftools sort)",
                    row_no,
                    lossy(rec.chrom)
                );
            }
            let name = lossy(rec.chrom);
            if !masks.mentions_chrom(&name) {
                summary.chroms_outside_mask.push(name.to_string());
            }
            let contig_len = if opts.to_contig_end {
                contig_lens.get(rec.chrom).copied()
            } else {
                None
            };
            if contig_len.is_some() {
                summary.contig_lengths_used.push(name.to_string());
            }
            let uncallable = masks.uncallable_for_chrom(&name);
            cur = Some(ChromBuilder::new(
                rec.chrom,
                uncallable,
                contig_len,
                opts.bin_size,
            ));
        }
        let chrom = cur.as_mut().expect("chromosome state was just set");
        if rec.pos < chrom.last_pos {
            bail!(
                "invalid VCF row {}: POS {} on '{}' comes after POS {}; the VCF must be \
                 sorted by position (e.g. with bcftools sort)",
                row_no,
                rec.pos,
                lossy(rec.chrom),
                chrom.last_pos
            );
        }
        if let Some(len) = chrom.contig_len.filter(|&len| rec.pos > len) {
            bail!(
                "invalid VCF row {}: POS {} on '{}' is beyond the ##contig length {}",
                row_no,
                rec.pos,
                lossy(rec.chrom),
                len
            );
        }
        let (site, span_end) = classify(&rec, opts.keep_filtered, row_no, &mut summary)?;
        chrom.add_record(rec.pos, site, span_end, &mut summary);
    }
    if let Some(mut done) = cur.take() {
        seqs.push(done.finish(&mut summary));
    }

    if summary.het_snps == 0 {
        let hint = summary
            .chroms_outside_mask_hint()
            .map(|h| format!("; {h}"))
            .unwrap_or_default();
        bail!(
            "no heterozygous SNP at a callable position found in VCF input ({} records, \
             {} heterozygous SNP records, {} outside the callable set){hint}",
            summary.records,
            summary.het_snp_records,
            summary.records_outside_mask
        );
    }
    summary.sequences = seqs.iter().filter(|s| !s.is_empty()).count();
    summary.bins = seqs.iter().map(Vec::len).sum();
    let obs = chunk_sequences(
        seqs,
        opts.batch_size,
        "no valid segregating sites found in VCF input",
    )?;
    Ok(VcfRead { obs, summary })
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
