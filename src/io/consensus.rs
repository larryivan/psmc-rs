//! VCF consensus mode, which reproduces the classic PSMC input pipeline
//!
//! ```text
//! bcftools mpileup -C50 -q20 -Q20 -f ref.fa aln.bam | bcftools call -c \
//!   | vcfutils.pl vcf2fq -d D -D DD -Q Q -l L | fq2psmcfa -q q -s s
//! ```
//!
//! in one streaming pass over the all-sites VCF of `bcftools call -c`, without
//! the intermediate FASTQ. The bins are exactly those `fq2psmcfa` writes from
//! the FASTQ of `vcfutils.pl vcf2fq` (bcftools 1.24) with the same options;
//! only `fq2psmcfa`'s per-sequence output filter (at least 10,000 called
//! bases, making up at least 20% of the sequence) is left to the caller,
//! which gets the counts it needs in [`ConsensusSeq`].
//! [`crate::io::input::load`] applies it ([`crate::io::input::SeqFilter`]).
//!
//! # vcf2fq
//!
//! Lines starting with `#` are skipped; the others are split on whitespace like
//! Perl's `split`. Each chromosome is a string that grows as records are read:
//!
//! - a record whose POS is more than one past the previous record's appends
//!   `n` (no data) for the positions in between;
//! - a record with a one-character REF, no `INDEL` anywhere in INFO and an ALT
//!   matching `^([A-Za-z.])(,[A-Za-z])*$` (the "SNP or reference" branch)
//!   appends one base. With `FQ < 0` it is homozygous: REF if `AF1 < 0.5` or
//!   ALT is `.`, else the first ALT. Otherwise (including a missing FQ, which
//!   Perl reads as 0) it is the IUPAC code of REF and the first ALT, or `N`
//!   when there is none (ALT `.`, equal or non-ACGT bases, lowercase bases).
//!   The base is uppercase when `MQ >= Q` and `d <= DP <= D` (INFO), else
//!   lowercase. Its quality is `int(|FQ| + 33.499)`, at most 126;
//! - any other record with ALT other than `.` is an INDEL: once the chromosome
//!   is read, the string positions `POS-l+1 ..= POS+len(REF)+l` (1-based,
//!   clipped) are lowercased;
//! - other records (ALT `.` with a longer REF, or with an `INDEL` flag) add
//!   nothing.
//!
//! The string grows by appending, not by POS: several base records at one POS
//! append several bases, and a POS that only has an INDEL record appends
//! none, which shifts the rest of the chromosome by one base each way.
//! [`ConsensusSummary::extra_bases_at_pos`] and
//! [`ConsensusSummary::positions_without_base`] count these; `bcftools call -c`
//! output has neither. Each INFO value is found the way vcf2fq's regular
//! expressions find it: the leftmost `FQ=` followed by a number
//! (`-?[0-9.]+`, read like Perl's numeric conversion), `AF1=` by `[0-9.]+`,
//! and `MQ=` and `DP=` by digits. On bcftools output, these are exactly the
//! `FQ`, `AF1`, `MQ` and `DP` keys.
//!
//! # fq2psmcfa
//!
//! A base whose quality minus 33 is below `-q` is lowercased. Lowercase bases
//! and bases outside the IUPAC table (`N`, `-`, ...) are missing; two-base
//! IUPAC codes (`K`, `M`, `R`, `S`, `W`, `Y`) are heterozygous; `A`, `C`, `G`,
//! `T` and the three-base codes are called homozygous. Bins follow
//! [`Binner`]: missing when more than 90% of the bases are, else heterozygous
//! when one base is.
//!
//! # Differences from the Perl and C tools
//!
//! None of these arise from `bcftools call -c` output.
//!
//! - Malformed lines are errors instead of being read as Perl reads them: a
//!   data line with fewer than 8 fields, a non-numeric POS, or a POS above
//!   2^36. Blank lines are skipped.
//! - An FQ value with two decimal points (`-1.2.3`) is read as its leading
//!   number. vcf2fq negates such a string as text (`"+1.2.3"`), and the
//!   number Perl later takes from it depends on interpreter state (in one
//!   loop, 34 the first time and 32 afterwards), so it cannot be mirrored.
//! - A chromosome without any base (only INDEL or ignored records at its
//!   first positions) is dropped. vcf2fq writes it as an empty FASTQ record,
//!   which fq2psmcfa cannot read: kseq then misreads the next record from
//!   inside its quality string, or crashes when it is the first record.
//! - REF and ALT bytes are taken as they are. A REF of `>`, `+` or `@`, or of
//!   a non-ASCII byte, would change how `fq2psmcfa` splits the FASTQ; such REF
//!   alleles are not valid VCF and are not mirrored.
//! - As in vcf2fq, a chromosome named `0` (false in Perl) is discarded unless
//!   it is the last one, a chromosome that appears in two blocks becomes two
//!   sequences, and an INDEL window starting beyond the end of its chromosome
//!   is an error (Perl dies with "substr outside of string"). The first two
//!   are reported in the summary.
//! - Optional callable masks (an extension) make bases outside the callable
//!   set missing; they are positioned by string index, which is POS unless
//!   the string was shifted as described above.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::{HashSet, VecDeque};
use std::io::BufRead;
use std::path::{Path, PathBuf};

use crate::io::common::{Binner, HetCall, HetTracker, is_regular_file, open_reader, reopen_reader};
use crate::io::vcf::{Interval, MAX_POS, MaskEngine};

/// Depth thresholds of `vcfutils.pl vcf2fq` (`-d` and `-D`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DepthThresholds {
    /// Fixed bounds on INFO/DP: bases are kept when `min <= DP <= max`.
    Fixed { min: u64, max: u64 },
    /// Chosen from the data in a first pass over the file: with `mean` the
    /// mean INFO/DP of the records in vcf2fq's SNP/reference branch,
    /// `min = max(1, round(mean / 3))` and `max = round(2 * mean)`, as Li and
    /// Durbin (2011) recommend.
    Auto,
}

impl Default for DepthThresholds {
    /// vcf2fq's defaults, `-d 3 -D 100000`.
    fn default() -> Self {
        Self::Fixed {
            min: 3,
            max: 100_000,
        }
    }
}

/// The thresholds of `vcfutils.pl vcf2fq` and `fq2psmcfa`, with their
/// defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConsensusFilters {
    /// INFO/DP bounds (`vcf2fq -d`, `-D`; default 3 and 100000).
    pub depth: DepthThresholds,
    /// Minimum INFO/MQ (`vcf2fq -Q`, default 10).
    pub min_mq: u64,
    /// Bases masked on each side of an INDEL (`vcf2fq -l`, default 5).
    pub indel_window: u64,
    /// Minimum base quality, i.e. `|FQ|` rounded (`fq2psmcfa -q`, default 10).
    pub min_qual: u64,
}

impl Default for ConsensusFilters {
    fn default() -> Self {
        Self {
            depth: DepthThresholds::default(),
            min_mq: 10,
            indel_window: 5,
            min_qual: 10,
        }
    }
}

/// Options for [`read_vcf_consensus`].
#[derive(Debug, Clone)]
pub struct ConsensusOptions {
    /// Bases per bin (`fq2psmcfa -s`, default 100).
    pub bin_size: usize,
    pub filters: ConsensusFilters,
    /// Callable masks, in the formats of [`crate::io::vcf::VcfOptions::masks`].
    /// Not part of the classic pipeline: without masks the output is that of
    /// vcf2fq and fq2psmcfa.
    pub masks: Vec<PathBuf>,
    /// Masks of positions to exclude.
    pub negative_masks: Vec<PathBuf>,
    /// Record every heterozygous base (for multihetsep output).
    pub collect_het_sites: bool,
}

impl Default for ConsensusOptions {
    fn default() -> Self {
        Self {
            bin_size: 100,
            filters: ConsensusFilters::default(),
            masks: Vec::new(),
            negative_masks: Vec::new(),
            collect_het_sites: false,
        }
    }
}

/// Why a base of the consensus is missing. Each missing base is counted once,
/// under the first reason in declaration order that applies to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LossReason {
    /// No record at this position (`n` from vcf2fq).
    NoRecord,
    /// INFO/MQ missing or below `-Q`.
    LowMq,
    /// INFO/DP missing or below `-d`.
    LowDepth,
    /// INFO/DP above `-D`.
    HighDepth,
    /// `|FQ|` (rounded) below fq2psmcfa's `-q`.
    LowQuality,
    /// A heterozygous call (`FQ >= 0` or no FQ) without an IUPAC code: `N`.
    NoHetCode,
    /// A homozygous base that fq2psmcfa reads as missing (e.g. REF `N`).
    NotABase,
    /// Within `-l` bases of an INDEL.
    IndelWindow,
    /// Outside the callable masks.
    OutsideMask,
}

impl LossReason {
    /// Every reason, in attribution order.
    pub const ALL: [LossReason; 9] = [
        LossReason::NoRecord,
        LossReason::LowMq,
        LossReason::LowDepth,
        LossReason::HighDepth,
        LossReason::LowQuality,
        LossReason::NoHetCode,
        LossReason::NotABase,
        LossReason::IndelWindow,
        LossReason::OutsideMask,
    ];

    /// Short label for reports.
    pub fn label(self) -> &'static str {
        match self {
            LossReason::NoRecord => "no record",
            LossReason::LowMq => "low MQ",
            LossReason::LowDepth => "low depth",
            LossReason::HighDepth => "high depth",
            LossReason::LowQuality => "low quality",
            LossReason::NoHetCode => "no het code",
            LossReason::NotABase => "not a base",
            LossReason::IndelWindow => "indel window",
            LossReason::OutsideMask => "outside mask",
        }
    }
}

/// Missing bases by [`LossReason`] (one field per reason, in the same order).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct LostBases {
    /// [`LossReason::NoRecord`].
    pub no_record: u64,
    /// [`LossReason::LowMq`].
    pub low_mq: u64,
    /// [`LossReason::LowDepth`].
    pub low_depth: u64,
    /// [`LossReason::HighDepth`].
    pub high_depth: u64,
    /// [`LossReason::LowQuality`].
    pub low_quality: u64,
    /// [`LossReason::NoHetCode`].
    pub no_het_code: u64,
    /// [`LossReason::NotABase`].
    pub not_a_base: u64,
    /// [`LossReason::IndelWindow`].
    pub indel_window: u64,
    /// [`LossReason::OutsideMask`].
    pub outside_mask: u64,
}

impl LostBases {
    fn slot(&mut self, r: LossReason) -> &mut u64 {
        match r {
            LossReason::NoRecord => &mut self.no_record,
            LossReason::LowMq => &mut self.low_mq,
            LossReason::LowDepth => &mut self.low_depth,
            LossReason::HighDepth => &mut self.high_depth,
            LossReason::LowQuality => &mut self.low_quality,
            LossReason::NoHetCode => &mut self.no_het_code,
            LossReason::NotABase => &mut self.not_a_base,
            LossReason::IndelWindow => &mut self.indel_window,
            LossReason::OutsideMask => &mut self.outside_mask,
        }
    }

    /// Bases lost to `r`.
    pub fn get(&self, r: LossReason) -> u64 {
        match r {
            LossReason::NoRecord => self.no_record,
            LossReason::LowMq => self.low_mq,
            LossReason::LowDepth => self.low_depth,
            LossReason::HighDepth => self.high_depth,
            LossReason::LowQuality => self.low_quality,
            LossReason::NoHetCode => self.no_het_code,
            LossReason::NotABase => self.not_a_base,
            LossReason::IndelWindow => self.indel_window,
            LossReason::OutsideMask => self.outside_mask,
        }
    }

    /// All missing bases.
    pub fn total(&self) -> u64 {
        LossReason::ALL.iter().map(|&r| self.get(r)).sum()
    }

    fn add(&mut self, other: &LostBases) {
        for r in LossReason::ALL {
            *self.slot(r) += other.get(r);
        }
    }
}

/// Distribution of INFO/DP over the records of vcf2fq's SNP/reference branch
/// that have one.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DepthStats {
    /// Records counted.
    pub records: u64,
    /// SNP/reference records without INFO/DP (always lowercase in vcf2fq).
    pub records_without_dp: u64,
    /// The mean depth (0 when no record was counted).
    pub mean: f64,
    /// The middle value (the mean of the two middle values for an even count).
    pub median: f64,
    /// `histogram[d]` records have `DP == d`, for `d <` [`DP_HISTOGRAM_CAP`];
    /// the last entry counts `DP >= DP_HISTOGRAM_CAP`.
    pub histogram: Vec<u64>,
}

/// Depths from this value on share the last bin of [`DepthStats::histogram`].
pub const DP_HISTOGRAM_CAP: usize = 1000;

/// Exact counts of every DP value, compact for the usual small values.
#[derive(Debug, Clone, Default)]
struct DepthCounter {
    small: Vec<u64>,
    large: std::collections::BTreeMap<u64, u64>,
    without_dp: u64,
}

impl DepthCounter {
    const SMALL: u64 = 1 << 16;

    fn add(&mut self, dp: Option<u64>) {
        let Some(dp) = dp else {
            self.without_dp += 1;
            return;
        };
        if dp < Self::SMALL {
            let i = dp as usize;
            if i >= self.small.len() {
                self.small.resize(i + 1, 0);
            }
            self.small[i] += 1;
        } else {
            *self.large.entry(dp).or_insert(0) += 1;
        }
    }

    fn counts(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let small = self
            .small
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(d, &n)| (d as u64, n));
        small.chain(self.large.iter().map(|(&d, &n)| (d, n)))
    }

    fn stats(&self) -> DepthStats {
        let n: u64 = self.counts().map(|(_, c)| c).sum();
        let sum: u128 = self.counts().map(|(d, c)| d as u128 * c as u128).sum();
        let mut histogram = vec![0u64; DP_HISTOGRAM_CAP + 1];
        for (d, c) in self.counts() {
            histogram[(d as usize).min(DP_HISTOGRAM_CAP)] += c;
        }
        // The values of rank n/2 and (n-1)/2 (0-based) are the middle ones.
        let value_at = |rank: u64| {
            let mut seen = 0;
            for (d, c) in self.counts() {
                seen += c;
                if seen > rank {
                    return d;
                }
            }
            0
        };
        let median = if n == 0 {
            0.0
        } else {
            (value_at((n - 1) / 2) as f64 + value_at(n / 2) as f64) / 2.0
        };
        DepthStats {
            records: n,
            records_without_dp: self.without_dp,
            mean: if n == 0 { 0.0 } else { sum as f64 / n as f64 },
            median,
            histogram,
        }
    }
}

/// What [`read_vcf_consensus`] read.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ConsensusSummary {
    /// Data records.
    pub records: u64,
    /// Records of the SNP/reference branch (one base each).
    pub base_records: u64,
    /// Records read as INDELs (masking windows).
    pub indel_records: u64,
    /// Records that add nothing (ALT `.` with a longer REF or an INDEL flag).
    pub ignored_records: u64,
    /// Base records without INFO/FQ: vcf2fq takes them as heterozygous with
    /// quality 0, so they are missing.
    pub records_without_fq: u64,
    /// Bases appended at a POS that already had one; each shifts the rest of
    /// its chromosome one base to the right.
    pub extra_bases_at_pos: u64,
    /// Positions with records but no base; each shifts the rest of its
    /// chromosome one base to the left.
    pub positions_without_base: u64,
    /// Sample columns in the header. vcf2fq reads only INFO, which bcftools
    /// computes over all samples together.
    pub samples: usize,
    /// Sequences produced (before any output filter).
    pub sequences: usize,
    /// Bases over all sequences (vcf2fq's sequence lengths).
    pub bases: u64,
    /// Bases fq2psmcfa reads as called (heterozygous or not).
    pub good_bases: u64,
    /// Bases fq2psmcfa reads as heterozygous.
    pub het_bases: u64,
    /// The other bases, by reason.
    pub lost: LostBases,
    /// The lower INFO/DP bound used (`-d`).
    pub min_depth: u64,
    /// The upper INFO/DP bound used (`-D`).
    pub max_depth: u64,
    /// Whether they were chosen from the data ([`DepthThresholds::Auto`]).
    pub auto_depth: bool,
    /// INFO/DP of the base records.
    pub depth: DepthStats,
    /// Whether callable masks were applied.
    pub masked: bool,
    /// Chromosomes some positive mask has no row for.
    pub chroms_outside_mask: Vec<String>,
    /// Chromosomes that appear in more than one block and so give several
    /// sequences of the same name.
    pub repeated_chroms: Vec<String>,
    /// Chromosomes vcf2fq drops because their name is false in Perl (`0`)
    /// and another chromosome follows.
    pub discarded_chroms: Vec<String>,
}

impl ConsensusSummary {
    /// Warnings worth showing to the user about how the input was read.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.samples > 1 {
            out.push(format!(
                "the VCF has {} samples, but consensus mode (like vcfutils.pl vcf2fq) reads the \
                 INFO fields FQ, AF1, DP and MQ, which bcftools computes over all samples \
                 together; call one sample at a time for PSMC",
                self.samples
            ));
        }
        if self.records_without_fq > 0 {
            out.push(format!(
                "{} SNP/reference records have no INFO/FQ; vcf2fq reads them as heterozygous \
                 with quality 0, so they are missing. Consensus mode needs the output of \
                 'bcftools call -c'",
                self.records_without_fq
            ));
        }
        if self.extra_bases_at_pos > 0 || self.positions_without_base > 0 {
            out.push(format!(
                "{} SNP/reference records share a POS with an earlier one and {} positions have \
                 only INDEL or ignored records; vcf2fq appends bases instead of placing them, so \
                 every such case shifts the rest of its chromosome by one base (reproduced here). \
                 Normalised or merged VCFs (bcftools norm -m) are not the all-sites output \
                 vcf2fq expects",
                self.extra_bases_at_pos, self.positions_without_base
            ));
        }
        if !self.repeated_chroms.is_empty() {
            out.push(format!(
                "chromosome(s) {} appear in more than one block of the VCF and give several \
                 sequences each, as in vcf2fq; sort the VCF (bcftools sort)",
                list_names(&self.repeated_chroms)
            ));
        }
        if !self.discarded_chroms.is_empty() {
            out.push(format!(
                "chromosome(s) {} were dropped because vcf2fq drops a chromosome named '0' when \
                 another one follows",
                list_names(&self.discarded_chroms)
            ));
        }
        if !self.chroms_outside_mask.is_empty() {
            out.push(format!(
                "the callable mask has no interval on {} VCF chromosome(s) ({}), which are \
                 therefore entirely missing; check that the chromosome names in the mask match \
                 the VCF (e.g. 'chr1' vs '1')",
                self.chroms_outside_mask.len(),
                list_names(&self.chroms_outside_mask)
            ));
        }
        out
    }
}

fn list_names(names: &[String]) -> String {
    let shown = names.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    if names.len() > 5 {
        format!("{shown}, ...")
    } else {
        shown
    }
}

/// One sequence of the consensus, as fq2psmcfa bins it.
#[derive(Debug, Clone)]
pub struct ConsensusSeq {
    /// The chromosome (vcf2fq's FASTQ record name).
    pub name: String,
    /// One symbol per bin: 0 = `T`, 1 = `K`, 2 = `N`.
    pub bins: Vec<u8>,
    /// Length of vcf2fq's sequence (fq2psmcfa's `len`).
    pub bases: u64,
    /// Bases read as called (fq2psmcfa's `n_good_bases`).
    pub good_bases: u64,
    /// Bases read as heterozygous.
    pub het_bases: u64,
    /// Missing bases by reason.
    pub lost: LostBases,
    /// Heterozygous bases (1-based string position, `nr_called`, and REF then
    /// ALT), when [`ConsensusOptions::collect_het_sites`] is set.
    pub(crate) het_sites: Vec<HetCall>,
}

impl ConsensusSeq {
    /// Heterozygous bases as `(position, nr_called, alleles)`; empty unless
    /// [`ConsensusOptions::collect_het_sites`] was set. The position is the
    /// 1-based index in vcf2fq's string (the POS on `bcftools call -c` output),
    /// and `nr_called` counts the called bases since the previous
    /// heterozygous one (exclusive) up to and including this one.
    pub fn het_sites(&self) -> impl Iterator<Item = (u64, u64, [u8; 2])> + '_ {
        self.het_sites
            .iter()
            .map(|h| (h.pos, h.nr_called, h.alleles))
    }
}

/// The result of [`read_vcf_consensus`].
#[derive(Debug, Clone)]
pub struct ConsensusRead {
    /// Every chromosome vcf2fq prints, in file order, before fq2psmcfa's
    /// output filter.
    pub seqs: Vec<ConsensusSeq>,
    pub summary: ConsensusSummary,
}

/// `seq_nt16_table` of fq2psmcfa.c, copied verbatim: 15 is missing, and a
/// code with two bits set (`bitcnt_table`) is heterozygous. Note `X` (and
/// `x`) map to 0, which fq2psmcfa reads as a called base.
#[rustfmt::skip]
const NT16: [u8; 256] = [
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15, 1,14, 2, 13,15,15, 4, 11,15,15,12, 15, 3,15,15,
    15,15, 5, 6,  8,15, 7, 9,  0,10,15,15, 15,15,15,15,
    15, 1,14, 2, 13,15,15, 4, 11,15,15,12, 15, 3,15,15,
    15,15, 5, 6,  8,15, 7, 9,  0,10,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
    15,15,15,15, 15,15,15,15, 15,15,15,15, 15,15,15,15,
];

/// `bitcnt_table` of fq2psmcfa.c: bases a code stands for.
const BITCNT: [u8; 16] = [4, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// How fq2psmcfa reads an uppercase base: missing, heterozygous (with the
/// two bases of its IUPAC code, in ACGT order) or called homozygous.
fn read_base(b: u8) -> Base {
    match NT16[b as usize] {
        15 => Base::Lost(LossReason::NotABase),
        code if BITCNT[code as usize] == 2 => {
            let mut pair = [0u8; 2];
            let bases = (0..4).filter(|k| code & (1 << k) != 0).map(|k| b"ACGT"[k]);
            for (slot, base) in pair.iter_mut().zip(bases) {
                *slot = base;
            }
            Base::Het(pair)
        }
        _ => Base::Hom,
    }
}

/// vcf2fq's `%het`: the IUPAC code of two different uppercase bases.
fn het_code(a: u8, b: u8) -> Option<u8> {
    Some(match (a, b) {
        (b'A', b'C') | (b'C', b'A') => b'M',
        (b'A', b'G') | (b'G', b'A') => b'R',
        (b'A', b'T') | (b'T', b'A') => b'W',
        (b'C', b'G') | (b'G', b'C') => b'S',
        (b'C', b'T') | (b'T', b'C') => b'Y',
        (b'G', b'T') | (b'T', b'G') => b'K',
        _ => return None,
    })
}

/// Perl's `\s`: the separators of `split` without arguments.
fn is_perl_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// The first eight whitespace-separated fields of a line, as Perl's `split`.
fn split8(line: &[u8]) -> Option<[&[u8]; 8]> {
    let mut out: [&[u8]; 8] = [&[]; 8];
    let mut i = 0;
    let n = line.len();
    for slot in out.iter_mut() {
        while i < n && is_perl_space(line[i]) {
            i += 1;
        }
        if i == n {
            return None;
        }
        let start = i;
        while i < n && !is_perl_space(line[i]) {
            i += 1;
        }
        *slot = &line[start..i];
    }
    Some(out)
}

/// `$t[4] =~ /^([A-Za-z.])(,[A-Za-z])*$/`.
fn alt_is_snp_or_ref(alt: &[u8]) -> bool {
    let Some((&first, rest)) = alt.split_first() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'.')
        && rest.len() % 2 == 0
        && rest
            .chunks_exact(2)
            .all(|c| c[0] == b',' && c[1].is_ascii_alphabetic())
}

fn is_num_char(b: u8) -> bool {
    b.is_ascii_digit() || b == b'.'
}

fn run_len(s: &[u8], ok: impl Fn(u8) -> bool) -> usize {
    s.iter().position(|&b| !ok(b)).unwrap_or(s.len())
}

/// The INFO values vcf2fq's regular expressions capture, as the leftmost
/// matches of `/INDEL/`, `/FQ=(-?[\d\.]+)/`, `/AF1=([\d\.]+)/`, `/MQ=(\d+)/`
/// and `/DP=(\d+)/`.
#[derive(Debug, Default, Clone, Copy)]
struct Info<'a> {
    indel: bool,
    fq: Option<&'a [u8]>,
    af1: Option<&'a [u8]>,
    mq: Option<&'a [u8]>,
    dp: Option<&'a [u8]>,
}

fn scan_info(info: &[u8]) -> Info<'_> {
    let mut out = Info::default();
    let n = info.len();
    let mut i = 0;
    while i + 3 <= n {
        let rest = &info[i..];
        match rest[0] {
            b'D' if out.dp.is_none() && rest.starts_with(b"DP=") => {
                let k = run_len(&rest[3..], |b| b.is_ascii_digit());
                if k > 0 {
                    out.dp = Some(&rest[3..3 + k]);
                }
            }
            b'M' if out.mq.is_none() && rest.starts_with(b"MQ=") => {
                let k = run_len(&rest[3..], |b| b.is_ascii_digit());
                if k > 0 {
                    out.mq = Some(&rest[3..3 + k]);
                }
            }
            b'F' if out.fq.is_none() && rest.starts_with(b"FQ=") => {
                let v = &rest[3..];
                // `-?` takes a '-' when a number follows it. Otherwise there
                // is no match here: `[\d.]+` cannot start at the '-' either.
                let sign = usize::from(v.first() == Some(&b'-'));
                let k = run_len(&v[sign..], is_num_char);
                if k > 0 {
                    out.fq = Some(&v[..sign + k]);
                }
            }
            b'A' if out.af1.is_none() && rest.starts_with(b"AF1=") => {
                let k = run_len(&rest[4..], is_num_char);
                if k > 0 {
                    out.af1 = Some(&rest[4..4 + k]);
                }
            }
            b'I' if !out.indel && rest.starts_with(b"INDEL") => out.indel = true,
            _ => {}
        }
        i += 1;
    }
    out
}

/// Perl's numeric value of a string matching `-?[\d.]+`: the longest prefix
/// of the form `-?\d*(\.\d*)?`, or 0 when it has no digit.
fn perl_num(s: &[u8]) -> f64 {
    let sign = usize::from(s.first() == Some(&b'-'));
    let int = run_len(&s[sign..], |b| b.is_ascii_digit());
    let mut end = sign + int;
    let mut digits = int;
    if s.get(end) == Some(&b'.') {
        let frac = run_len(&s[end + 1..], |b| b.is_ascii_digit());
        end += 1 + frac;
        digits += frac;
    }
    if digits == 0 {
        return 0.0;
    }
    // Only ASCII digits, '-' and '.', so this is valid UTF-8 and a valid
    // float literal once a lone trailing '.' is accepted (Rust accepts it).
    std::str::from_utf8(&s[..end])
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Digits as an integer, saturating (the value is only compared with
/// thresholds).
fn digits_u64(s: &[u8]) -> u64 {
    s.iter().fold(0u64, |v, &c| {
        v.saturating_mul(10).saturating_add(u64::from(c - b'0'))
    })
}

fn parse_pos(b: &[u8], row_no: usize) -> Result<u64> {
    if b.is_empty() || b.len() > 19 || !b.iter().all(u8::is_ascii_digit) {
        bail!(
            "invalid VCF row {row_no}: bad POS '{}'",
            String::from_utf8_lossy(b)
        );
    }
    let pos = digits_u64(b);
    if pos > MAX_POS {
        bail!("invalid VCF row {row_no}: POS {pos} is beyond the supported maximum {MAX_POS}");
    }
    Ok(pos)
}

/// What the SNP/reference branch makes of one record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Base {
    Hom,
    Het([u8; 2]),
    Lost(LossReason),
}

impl Base {
    fn is_called(self) -> bool {
        !matches!(self, Base::Lost(_))
    }
}

/// Thresholds in effect for one read.
#[derive(Debug, Clone, Copy)]
struct Thresholds {
    min_depth: u64,
    max_depth: u64,
    min_mq: u64,
    min_qual: u64,
}

/// One base of vcf2fq's output as fq2psmcfa reads it. `ref_b` is REF (one
/// byte) and `alt_b` the first byte of ALT.
fn consensus_base(ref_b: u8, alt_b: u8, info: &Info<'_>, t: &Thresholds) -> Base {
    let fq = info.fq.map(perl_num);
    // vcf2fq's two branches give the base, read here as fq2psmcfa reads it
    // in uppercase, and the quality before the filters.
    let (base, q) = match fq {
        Some(fq) if fq < 0.0 => {
            let af1 = info.af1.map_or(0.0, perl_num);
            let b = if af1 < 0.5 || alt_b == b'.' {
                ref_b
            } else {
                alt_b
            };
            (read_base(b.to_ascii_uppercase()), -fq)
        }
        // A missing FQ is Perl's undef: not below 0, and 0 as a number.
        _ => {
            let base = match het_code(ref_b, alt_b) {
                Some(_) => Base::Het([ref_b, alt_b]),
                None => Base::Lost(LossReason::NoHetCode),
            };
            (base, fq.unwrap_or(0.0))
        }
    };
    // Lowercase unless `MQ >= Q && d <= DP <= D`, in Perl's evaluation order.
    if !info.mq.is_some_and(|m| digits_u64(m) >= t.min_mq) {
        return Base::Lost(LossReason::LowMq);
    }
    match info.dp.map(digits_u64) {
        None => return Base::Lost(LossReason::LowDepth),
        Some(d) if d < t.min_depth => return Base::Lost(LossReason::LowDepth),
        Some(d) if d > t.max_depth => return Base::Lost(LossReason::HighDepth),
        Some(_) => {}
    }
    // vcf2fq's quality character is `int($q + 33 + .499)`, at most 126;
    // fq2psmcfa lowercases bases whose quality minus 33 is below `-q`.
    let qchar = ((q + 33.0) + 0.499).trunc().min(126.0) as i64;
    if qchar - 33 < t.min_qual as i64 {
        return Base::Lost(LossReason::LowQuality);
    }
    base
}

/// A run of the vcf2fq string waiting for INDEL windows that may still cover
/// it. Only `NoRecord` runs are longer than one base.
#[derive(Debug, Clone, Copy)]
struct Run {
    len: u64,
    base: Base,
}

/// One chromosome of vcf2fq's string, streamed into fq2psmcfa's binner.
///
/// Bases stay in `pending` until no later INDEL window can reach them: the
/// window of an INDEL at POS p starts at string index `p - l` (0-based), and
/// POS never decreases, so indices below `last_pos - l` are final.
struct Chrom {
    name: Vec<u8>,
    /// Length of the vcf2fq string so far.
    len: u64,
    last_pos: u64,
    /// Bases appended at `last_pos`.
    bases_at_last_pos: u64,
    pending: VecDeque<Run>,
    /// String index of the first pending base.
    committed: u64,
    /// Parts of INDEL windows beyond `len` (sorted, disjoint, half-open).
    future: VecDeque<(u64, u64)>,
    /// POS and start of the last INDEL window (window starts never
    /// decrease).
    last_window: Option<(u64, u64)>,
    /// Whether a record has been read, i.e. `last_pos` is a real position.
    started: bool,
    uncallable: Vec<Interval>,
    next_unc: usize,
    binner: Binner,
    good: u64,
    het_bases: u64,
    lost: LostBases,
    het: HetTracker,
}

impl Chrom {
    fn new(name: &[u8], bin_size: usize, uncallable: Vec<Interval>, collect_het: bool) -> Self {
        Self {
            name: name.to_vec(),
            len: 0,
            last_pos: 0,
            bases_at_last_pos: 0,
            pending: VecDeque::new(),
            committed: 0,
            future: VecDeque::new(),
            last_window: None,
            started: false,
            uncallable,
            next_unc: 0,
            binner: Binner::new(bin_size),
            good: 0,
            het_bases: 0,
            lost: LostBases::default(),
            het: HetTracker::new(collect_het),
        }
    }

    fn push_gap(&mut self, n: u64) {
        if n == 0 {
            return;
        }
        match self.pending.back_mut() {
            Some(r) if r.base == Base::Lost(LossReason::NoRecord) => r.len += n,
            _ => self.pending.push_back(Run {
                len: n,
                base: Base::Lost(LossReason::NoRecord),
            }),
        }
        self.len += n;
    }

    fn push_base(&mut self, base: Base) {
        let i = self.len;
        while self.future.front().is_some_and(|w| w.1 <= i) {
            self.future.pop_front();
        }
        let base = if base.is_called() && self.future.front().is_some_and(|w| w.0 <= i) {
            Base::Lost(LossReason::IndelWindow)
        } else {
            base
        };
        self.pending.push_back(Run { len: 1, base });
        self.len += 1;
        self.bases_at_last_pos += 1;
    }

    /// Lowercase string indices `beg..end` (0-based, half-open) for the
    /// INDEL at `pos`, now for the bases already appended and later for the
    /// others.
    fn add_window(&mut self, pos: u64, beg: u64, end: u64) {
        debug_assert!(beg >= self.committed);
        self.last_window = Some((pos, beg));
        if beg < self.len {
            let mut run_end = self.len;
            for r in self.pending.iter_mut().rev() {
                let run_start = run_end - r.len;
                if run_start < end && beg < run_end && r.base.is_called() {
                    r.base = Base::Lost(LossReason::IndelWindow);
                }
                if run_start <= beg {
                    break;
                }
                run_end = run_start;
            }
        }
        if end > self.len {
            let start = beg.max(self.len);
            match self.future.back_mut() {
                Some(w) if start <= w.1 => w.1 = w.1.max(end),
                _ => self.future.push_back((start, end)),
            }
        }
    }

    /// Whether 1-based `pos` is callable; `pos` must not decrease.
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
            .is_some_and(|&(s, _)| s <= pos)
    }

    /// Bin every pending base with a string index below `limit`.
    fn commit(&mut self, limit: u64) {
        while self.committed < limit {
            let Some(run) = self.pending.front_mut() else {
                break;
            };
            let take = run.len.min(limit - self.committed);
            let mut base = run.base;
            run.len -= take;
            if run.len == 0 {
                self.pending.pop_front();
            }
            let pos = self.committed + 1;
            self.committed += take;
            if base.is_called() && !self.uncallable.is_empty() && !self.is_callable(pos) {
                base = Base::Lost(LossReason::OutsideMask);
            }
            match base {
                Base::Hom => {
                    self.binner.add_called(take);
                    self.het.called(take);
                    self.good += take;
                }
                Base::Het(alleles) => {
                    debug_assert_eq!(take, 1);
                    self.binner.add_het();
                    self.het.het(pos, alleles);
                    self.good += 1;
                    self.het_bases += 1;
                }
                Base::Lost(r) => {
                    self.binner.add_missing(take);
                    *self.lost.slot(r) += take;
                }
            }
        }
    }

    fn finish(mut self) -> Result<ConsensusSeq> {
        if let Some((pos, beg)) = self.last_window.filter(|&(_, b)| b > self.len) {
            bail!(
                "the INDEL at POS {pos} of chromosome '{}' masks bases from {} on, but the \
                 chromosome ends at base {} (vcf2fq dies here with 'substr outside of string'); \
                 vcf2fq places bases by appending them, so records at positions without a \
                 SNP/reference record shorten the chromosome",
                String::from_utf8_lossy(&self.name),
                beg + 1,
                self.len
            );
        }
        self.commit(self.len);
        Ok(ConsensusSeq {
            name: String::from_utf8_lossy(&self.name).into_owned(),
            bins: self.binner.finish(),
            bases: self.len,
            good_bases: self.good,
            het_bases: self.het_bases,
            lost: self.lost,
            het_sites: self.het.take(),
        })
    }
}

/// Whether INFO has a value vcf2fq reads as FQ.
pub(crate) fn info_has_fq(info: &[u8]) -> bool {
    scan_info(info).fq.is_some()
}

/// The SNP/reference test of vcf2fq.
fn is_base_record(f: &[&[u8]; 8], info: &Info<'_>) -> bool {
    f[3].len() == 1 && !info.indel && alt_is_snp_or_ref(f[4])
}

/// Mean, median and histogram of INFO/DP over the SNP/reference records of
/// a VCF (one pass over the file), as used by [`DepthThresholds::Auto`].
pub fn scan_depth(path: &Path) -> Result<DepthStats> {
    scan_depth_from(open_reader(path)?, path)
}

/// [`scan_depth`] over an opened input; `path` names it in messages.
pub(crate) fn scan_depth_from(mut reader: impl BufRead, path: &Path) -> Result<DepthStats> {
    let mut line = Vec::new();
    let mut counter = DepthCounter::default();
    let mut row_no = 0usize;
    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read VCF {path:?}"))?
            == 0
        {
            break;
        }
        row_no += 1;
        if line.first() == Some(&b'#') {
            continue;
        }
        let Some(f) = split8(&line) else {
            if line.iter().all(|&b| is_perl_space(b)) {
                continue;
            }
            bail!("invalid VCF row {row_no}: expected at least 8 fields");
        };
        let info = scan_info(f[7]);
        if is_base_record(&f, &info) {
            counter.add(info.dp.map(digits_u64));
        }
    }
    Ok(counter.stats())
}

/// The depth thresholds [`DepthThresholds::Auto`] derives from a mean depth:
/// `(max(1, round(mean / 3)), round(2 * mean))`.
pub fn auto_depth_thresholds(mean: f64) -> (u64, u64) {
    (
        ((mean / 3.0).round() as u64).max(1),
        (2.0 * mean).round() as u64,
    )
}

/// The INFO/DP bounds of one read, fixed or chosen from the data.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DepthBounds {
    pub min: u64,
    pub max: u64,
    /// Whether they come from [`DepthThresholds::Auto`].
    pub auto: bool,
}

impl DepthBounds {
    /// [`DepthThresholds::Auto`]'s bounds from the depths of a first pass.
    pub(crate) fn from_stats(stats: &DepthStats, path: &Path) -> Result<Self> {
        if stats.records == 0 {
            bail!(
                "cannot choose depth thresholds automatically: no SNP/reference record in \
                 {path:?} has INFO/DP; give the thresholds explicitly"
            );
        }
        let (min, max) = auto_depth_thresholds(stats.mean);
        Ok(Self {
            min,
            max,
            auto: true,
        })
    }
}

/// Fail unless `path` can be read twice, as [`DepthThresholds::Auto`] needs.
pub(crate) fn ensure_rereadable(path: &Path) -> Result<()> {
    if !is_regular_file(path) {
        bail!(
            "automatic depth thresholds need two passes over the VCF (the first one measures \
             the mean depth), but {path:?} is not a regular file and can be read only once \
             (a pipe or FIFO); give the depth thresholds explicitly, or write the VCF to a \
             file first"
        );
    }
    Ok(())
}

/// Read an all-sites VCF from `bcftools call -c` the way `vcfutils.pl vcf2fq`
/// and `fq2psmcfa` read it; see the module docs.
///
/// [`DepthThresholds::Auto`] reads the file twice, so `path` must then be a
/// regular file. Otherwise it is read once and may be a pipe or FIFO.
pub fn read_vcf_consensus(path: &Path, opts: &ConsensusOptions) -> Result<ConsensusRead> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    match opts.filters.depth {
        DepthThresholds::Fixed { min, max } => {
            let bounds = DepthBounds {
                min,
                max,
                auto: false,
            };
            read_consensus_from(open_reader(path)?, path, opts, bounds)
        }
        DepthThresholds::Auto => {
            ensure_rereadable(path)?;
            let bounds = DepthBounds::from_stats(&scan_depth(path)?, path)?;
            read_consensus_from(reopen_reader(path)?, path, opts, bounds)
        }
    }
}

/// [`read_vcf_consensus`] over an opened input with the depth bounds already
/// chosen (`opts.filters.depth` is not used); `path` names the input in
/// messages.
pub(crate) fn read_consensus_from(
    mut reader: impl BufRead,
    path: &Path,
    opts: &ConsensusOptions,
    bounds: DepthBounds,
) -> Result<ConsensusRead> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    let filters = &opts.filters;
    let mut summary = ConsensusSummary {
        min_depth: bounds.min,
        max_depth: bounds.max,
        auto_depth: bounds.auto,
        ..ConsensusSummary::default()
    };
    let t = Thresholds {
        min_depth: bounds.min,
        max_depth: bounds.max,
        min_mq: filters.min_mq,
        min_qual: filters.min_qual,
    };
    let masks = MaskEngine::new(&opts.masks, &opts.negative_masks)?;
    summary.masked = !opts.masks.is_empty() || !opts.negative_masks.is_empty();
    let l = filters.indel_window;

    let mut line = Vec::new();
    let mut row_no = 0usize;
    let mut depth = DepthCounter::default();
    let mut seen = HashSet::<Vec<u8>>::new();
    let mut seqs = Vec::new();
    let mut cur: Option<Chrom> = None;

    loop {
        line.clear();
        if reader
            .read_until(b'\n', &mut line)
            .with_context(|| format!("failed to read VCF {path:?}"))?
            == 0
        {
            break;
        }
        row_no += 1;
        if line.first() == Some(&b'#') {
            if line.starts_with(b"#CHROM") {
                summary.samples = line.split(|&b| b == b'\t').count().saturating_sub(9);
            }
            continue;
        }
        let Some(f) = split8(&line) else {
            if line.iter().all(|&b| is_perl_space(b)) {
                continue;
            }
            bail!("invalid VCF row {row_no}: expected at least 8 fields");
        };
        summary.records += 1;
        let pos = parse_pos(f[1], row_no)?;

        if cur.as_ref().is_none_or(|c| c.name != f[0]) {
            if let Some(done) = cur.take() {
                end_chrom(done, false, &mut summary, &mut seqs)?;
            }
            if !seen.insert(f[0].to_vec()) {
                let name = String::from_utf8_lossy(f[0]).into_owned();
                if !summary.repeated_chroms.contains(&name) {
                    summary.repeated_chroms.push(name);
                }
            }
            let name = String::from_utf8_lossy(f[0]);
            let uncallable = if summary.masked {
                if !masks.mentions_chrom(&name)
                    && !summary.chroms_outside_mask.iter().any(|n| *n == name)
                {
                    summary.chroms_outside_mask.push(name.to_string());
                }
                masks.uncallable_for_chrom(&name)
            } else {
                Vec::new()
            };
            cur = Some(Chrom::new(
                f[0],
                opts.bin_size,
                uncallable,
                opts.collect_het_sites,
            ));
        }
        let c = cur.as_mut().expect("chromosome state was just set");
        if pos < c.last_pos {
            bail!(
                "invalid VCF row {row_no}: POS {pos} on '{}' comes after POS {} (vcf2fq: \
                 unsorted input); sort the VCF with bcftools sort",
                String::from_utf8_lossy(f[0]),
                c.last_pos
            );
        }
        if pos > c.last_pos || !c.started {
            count_position(c, &mut summary);
            c.bases_at_last_pos = 0;
            c.push_gap(pos.saturating_sub(c.last_pos + 1));
            c.started = true;
        }

        let info = scan_info(f[7]);
        if is_base_record(&f, &info) {
            summary.base_records += 1;
            if info.fq.is_none() {
                summary.records_without_fq += 1;
            }
            depth.add(info.dp.map(digits_u64));
            c.push_base(consensus_base(f[3][0], f[4][0], &info, &t));
        } else if f[4] != b"." {
            summary.indel_records += 1;
            let beg = pos.saturating_sub(l);
            let end = pos + f[3].len() as u64 + l;
            c.add_window(pos, beg, end);
        } else {
            summary.ignored_records += 1;
        }
        c.last_pos = pos;
        let limit = c.len.min(pos.saturating_sub(l));
        c.commit(limit);
    }
    if let Some(done) = cur.take() {
        end_chrom(done, true, &mut summary, &mut seqs)?;
    }

    summary.depth = depth.stats();
    summary.sequences = seqs.len();
    for s in &seqs {
        summary.bases += s.bases;
        summary.good_bases += s.good_bases;
        summary.het_bases += s.het_bases;
        summary.lost.add(&s.lost);
    }
    Ok(ConsensusRead { seqs, summary })
}

/// Account for the bases appended at the last POS of `c` before moving on.
fn count_position(c: &Chrom, summary: &mut ConsensusSummary) {
    if !c.started {
        return;
    }
    match c.bases_at_last_pos {
        0 => summary.positions_without_base += 1,
        n => summary.extra_bases_at_pos += n - 1,
    }
}

/// vcf2fq's `v2q_post_process` for a finished chromosome; `last` tells
/// whether it is the last one of the file.
fn end_chrom(
    c: Chrom,
    last: bool,
    summary: &mut ConsensusSummary,
    seqs: &mut Vec<ConsensusSeq>,
) -> Result<()> {
    count_position(&c, summary);
    // `&v2q_post_process(...) if ($last_chr)`: the name '0' is false in Perl.
    if c.name == b"0" && !last {
        summary.discarded_chroms.push("0".to_string());
        return Ok(());
    }
    seqs.push(c.finish()?);
    Ok(())
}
