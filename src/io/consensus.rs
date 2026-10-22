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
//! [`Binner`](crate::io::common::Binner): missing when more than 90% of the
//! bases are, else heterozygous when one base is. Each bin's callable bases
//! ([`ConsensusSeq::callable`]) are the bin size minus its missing bases: the
//! good bases of a full bin.
//!
//! # Differences from the Perl and C tools
//!
//! None of these arise from `bcftools call -c` output.
//!
//! - Malformed lines are errors instead of being read as Perl reads them: a
//!   data line with fewer than 8 fields, a non-numeric POS, or a POS above
//!   2^36. Blank lines are skipped.
//! - A chromosome name that is not valid UTF-8 is an error. VCF is UTF-8,
//!   and names are kept as text ([`ConsensusSeq::name`]); vcf2fq and
//!   fq2psmcfa copy the bytes.
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
use foldhash::{HashSet, HashSetExt};
use std::collections::VecDeque;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use crate::io::BinCounts;
pub use crate::io::common::DepthStats;
use crate::io::common::{
    DepthCounter, HetCall, LineReader, SeqBuilder, is_regular_file, open_reader, reopen_reader,
};
use crate::io::vcf::{MaskEngine, Uncallable, list_names, mask_coverage_hints, parse_pos};

/// Depth thresholds of `vcfutils.pl vcf2fq` (`-d` and `-D`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthThresholds {
    /// Fixed bounds on INFO/DP: bases are kept when `min <= DP <= max`.
    Fixed { min: u64, max: u64 },
    /// Chosen from the data in a first pass over the file: with `mean` the
    /// mean INFO/DP of the records in vcf2fq's SNP/reference branch,
    /// `min = max(1, round(mean / 3))` and `max = round(2 * mean)`, as the
    /// psmc README recommends for `vcf2fq -d` and `-D`. (Li & Durbin 2011
    /// masked sites below half or above twice the mean depth.)
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Every reason, in attribution (declaration) order: `ALL[r as usize]` is `r`.
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

/// Missing bases by [`LossReason`] ([`LostBases::get`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LostBases([u64; LossReason::ALL.len()]);

impl LostBases {
    /// Bases lost to `r`.
    pub fn get(&self, r: LossReason) -> u64 {
        self.0[r as usize]
    }

    /// All missing bases.
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }

    fn add(&mut self, r: LossReason, n: u64) {
        self.0[r as usize] += n;
    }

    fn add_all(&mut self, other: &LostBases) {
        for (n, m) in self.0.iter_mut().zip(other.0) {
            *n += m;
        }
    }
}

/// What [`read_vcf_consensus`] read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConsensusSummary {
    /// Data records.
    pub records: u64,
    /// Records of the SNP/reference branch (one base each).
    pub base_records: u64,
    /// Records read as INDELs (masking windows).
    pub indel_records: u64,
    /// Records that add nothing (ALT `.` with a longer REF or an INDEL flag).
    pub ignored_records: u64,
    /// Base records without INFO/FQ. vcf2fq takes each as a heterozygous call
    /// of quality 0, which is missing unless `fq2psmcfa -q` is 0
    /// ([`ConsensusFilters::min_qual`]): then a record whose REF and first ALT
    /// have an IUPAC code and that passes the MQ and depth bounds is a called
    /// heterozygous base.
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
    /// The minimum INFO/MQ used (`vcf2fq -Q`).
    pub min_mq: u64,
    /// The bases masked on each side of an INDEL (`vcf2fq -l`).
    pub indel_window: u64,
    /// The minimum base quality used (`fq2psmcfa -q`).
    pub min_qual: u64,
    /// INFO/DP of the base records.
    pub depth: DepthStats,
    /// Whether callable masks were applied.
    pub masked: bool,
    /// Chromosomes no positive mask has a row for.
    pub chroms_outside_mask: Vec<String>,
    /// Chromosomes some positive masks have rows for and others do not
    /// (the masks are intersected, so they are entirely missing too).
    pub chroms_outside_some_masks: Vec<String>,
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
            let outcome = if self.min_qual > 0 {
                format!("which fq2psmcfa -q {} makes missing", self.min_qual)
            } else {
                "and with fq2psmcfa -q 0 the ones whose REF and ALT have an IUPAC code and that \
                 pass the MQ and depth bounds are called heterozygous"
                    .to_string()
            };
            out.push(format!(
                "{} SNP/reference records have no INFO/FQ; vcf2fq reads each as a heterozygous \
                 call of quality 0, {outcome}. Consensus mode needs the output of 'bcftools \
                 call -c'",
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
                "the VCF has more than one block of {}, and each block gives a sequence of its \
                 own, as in vcf2fq; sort the VCF (bcftools sort)",
                list_names(&self.repeated_chroms)
            ));
        }
        if !self.discarded_chroms.is_empty() {
            out.push(format!(
                "dropped {}, as vcf2fq drops a chromosome named '0' when another one follows",
                list_names(&self.discarded_chroms)
            ));
        }
        out.extend(mask_coverage_hints(
            &self.chroms_outside_mask,
            &self.chroms_outside_some_masks,
        ));
        out
    }
}

/// One sequence of the consensus, as fq2psmcfa bins it.
#[derive(Debug, Clone)]
pub struct ConsensusSeq {
    /// The chromosome (vcf2fq's FASTQ record name).
    pub name: String,
    /// One symbol per bin: 0 = `T`, 1 = `K`, 2 = `N`.
    pub bins: Vec<u8>,
    /// The callable bases of each bin: the bin size minus the bases
    /// fq2psmcfa reads as missing, i.e. the good bases of a full bin (the
    /// bases past the end of a shorter last bin count as callable, see
    /// [`crate::io::callable`]); `None` for bins of more than
    /// [`MAX_COUNTED_BIN_SIZE`](crate::io::MAX_COUNTED_BIN_SIZE) bases.
    pub callable: Option<BinCounts>,
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
        // The code's lowest and highest set bits, in ACGT order.
        code if BITCNT[code as usize] == 2 => Base::Het([
            b"ACGT"[code.trailing_zeros() as usize],
            b"ACGT"[(u8::BITS - 1 - code.leading_zeros()) as usize],
        ]),
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
    let mut rest = line;
    for slot in out.iter_mut() {
        let start = rest.iter().position(|&b| !is_perl_space(b))?;
        rest = &rest[start..];
        (*slot, rest) = rest.split_at(run_len(rest, |b| !is_perl_space(b)));
    }
    Some(out)
}

/// A line of a VCF as vcf2fq reads it ([`Lines`]).
enum Line<'a> {
    /// A line starting with `#`.
    Header(&'a [u8]),
    /// The first eight fields of a data line ([`split8`]), and its number.
    Record {
        row_no: usize,
        fields: [&'a [u8]; 8],
    },
}

/// The lines of a VCF.
struct Lines<'p, R> {
    reader: LineReader<R>,
    /// Names the input in messages.
    path: &'p Path,
    row_no: usize,
}

impl<'p, R: BufRead> Lines<'p, R> {
    fn new(reader: R, path: &'p Path) -> Self {
        Self {
            reader: LineReader::new(reader),
            path,
            row_no: 0,
        }
    }

    /// The next line that is not blank, or `None` at the end of the input.
    /// A data line with fewer than 8 fields is an error.
    fn next(&mut self) -> Result<Option<Line<'_>>> {
        loop {
            let path = self.path;
            let more = self
                .reader
                .advance()
                .with_context(|| format!("failed to read VCF {path:?}"))?;
            if !more {
                return Ok(None);
            }
            self.row_no += 1;
            if !self.reader.line().iter().all(|&b| is_perl_space(b)) {
                break;
            }
        }
        let row_no = self.row_no;
        let line = self.reader.line();
        if line[0] == b'#' {
            return Ok(Some(Line::Header(line)));
        }
        match split8(line) {
            Some(fields) => Ok(Some(Line::Record { row_no, fields })),
            None => bail!("invalid VCF row {row_no}: expected at least 8 fields"),
        }
    }
}

/// `$t[4] =~ /^([A-Za-z.])(,[A-Za-z])*$/`.
fn alt_is_snp_or_ref(alt: &[u8]) -> bool {
    let Some((&first, rest)) = alt.split_first() else {
        return false;
    };
    let (pairs, odd) = rest.as_chunks::<2>();
    (first.is_ascii_alphabetic() || first == b'.')
        && odd.is_empty()
        && pairs
            .iter()
            .all(|&[comma, base]| comma == b',' && base.is_ascii_alphabetic())
}

fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

fn is_num_char(b: u8) -> bool {
    is_digit(b) || b == b'.'
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

/// [`Info`] of an INFO column. Every key but INDEL ends with an `=`, so the
/// leftmost match of each is found at the first `=` it ends at. The `=` and
/// `I` bytes are found eight at a time, and visited in order.
fn scan_info(info: &[u8]) -> Info<'_> {
    const EQ: u64 = u64::from_ne_bytes([b'='; 8]);
    const I: u64 = u64::from_ne_bytes([b'I'; 8]);
    let mut out = Info::default();
    let (words, _) = info.as_chunks::<8>();
    for (k, &word) in words.iter().enumerate() {
        let x = u64::from_le_bytes(word);
        let mut found = has_zero(x ^ EQ) | has_zero(x ^ I);
        while found != 0 {
            info_at(info, 8 * k + found.trailing_zeros() as usize / 8, &mut out);
            found &= found - 1;
        }
    }
    for i in 8 * words.len()..info.len() {
        info_at(info, i, &mut out);
    }
    out
}

/// The high bit of every zero byte of `x`, and perhaps of bytes after one
/// (which [`info_at`] looks at and passes over).
fn has_zero(x: u64) -> u64 {
    x.wrapping_sub(u64::from_ne_bytes([1; 8])) & !x & u64::from_ne_bytes([0x80; 8])
}

/// What `info[i]` adds to `out`: an INDEL that starts there, or the value
/// of a key whose `=` it is. (Inlined: [`scan_info`] calls it for every
/// `=`.)
#[inline(always)]
fn info_at<'a>(info: &'a [u8], i: usize, out: &mut Info<'a>) {
    if info[i] == b'I' {
        out.indel |= info[i..].starts_with(b"INDEL");
    }
    if info[i] != b'=' {
        return;
    }
    // The value after the key: its leading run of `ok` bytes, if not
    // empty.
    let (key, v) = info.split_at(i + 1);
    let value = |ok: fn(u8) -> bool| Some(&v[..run_len(v, ok)]).filter(|v| !v.is_empty());
    if key.ends_with(b"DP=") {
        out.dp = out.dp.or_else(|| value(is_digit));
    } else if key.ends_with(b"MQ=") {
        out.mq = out.mq.or_else(|| value(is_digit));
    } else if key.ends_with(b"AF1=") {
        out.af1 = out.af1.or_else(|| value(is_num_char));
    } else if key.ends_with(b"FQ=") && out.fq.is_none() {
        // `-?` takes a '-' when a number follows it. Otherwise there
        // is no match here: `[\d.]+` cannot start at the '-' either.
        let sign = usize::from(v.first() == Some(&b'-'));
        let k = run_len(&v[sign..], is_num_char);
        if k > 0 {
            out.fq = Some(&v[..sign + k]);
        }
    }
}

/// `10^k` for `k` up to 15, each exact.
const POW10: [f64; 16] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
];

/// Perl's numeric value of a string matching `-?[\d.]+`: the longest prefix
/// of the form `-?\d*(\.\d*)?`, or 0 when it has no digit.
fn perl_num(s: &[u8]) -> f64 {
    let sign = usize::from(s.first() == Some(&b'-'));
    let int = run_len(&s[sign..], is_digit);
    let mut end = sign + int;
    let mut digits = int;
    if s.get(end) == Some(&b'.') {
        let frac = run_len(&s[end + 1..], is_digit);
        end += 1 + frac;
        digits += frac;
    }
    if digits == 0 {
        return 0.0;
    }
    // With at most 15 digits, the digits as an integer and the power of ten
    // they are divided by are exact doubles, so the one rounding of the
    // division gives the correctly rounded value, as the parser below does
    // (Clinger's fast path), in a fraction of its time: FQ is read for
    // every record.
    if digits <= 15 {
        let m = s[sign..end]
            .iter()
            .filter(|&&b| b != b'.')
            .fold(0u64, |m, &b| 10 * m + u64::from(b - b'0'));
        let v = m as f64 / POW10[digits - int];
        return if sign == 1 { -v } else { v };
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
    name: String,
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
    uncallable: Uncallable,
    seq: SeqBuilder,
    lost: LostBases,
}

impl Chrom {
    fn new(name: &str, bin_size: usize, uncallable: Uncallable, collect_het: bool) -> Self {
        Self {
            name: name.to_string(),
            len: 0,
            last_pos: 0,
            bases_at_last_pos: 0,
            pending: VecDeque::new(),
            committed: 0,
            future: VecDeque::new(),
            last_window: None,
            started: false,
            uncallable,
            seq: SeqBuilder::new(bin_size, collect_het),
            lost: LostBases::default(),
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
            if base.is_called() && !self.uncallable.is_callable(pos) {
                base = Base::Lost(LossReason::OutsideMask);
            }
            match base {
                Base::Hom => self.seq.called(take),
                Base::Het(alleles) => {
                    debug_assert_eq!(take, 1);
                    self.seq.het(pos, alleles);
                }
                Base::Lost(r) => {
                    self.seq.missing(take);
                    self.lost.add(r, take);
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
                self.name,
                beg + 1,
                self.len
            );
        }
        self.commit(self.len);
        let seq = self.seq.finish(self.name, self.len);
        Ok(ConsensusSeq {
            name: seq.name,
            bins: seq.bins,
            callable: seq.callable,
            bases: seq.extent,
            good_bases: seq.called,
            het_bases: seq.n_het,
            lost: self.lost,
            het_sites: seq.het_sites,
        })
    }
}

/// A chromosome name as text. VCF is UTF-8; another encoding is an error
/// rather than a lossy conversion, which could give two chromosomes one name.
pub(crate) fn utf8_chrom(name: &[u8], row_no: usize) -> Result<&str> {
    std::str::from_utf8(name).map_err(|_| {
        anyhow::anyhow!(
            "invalid VCF row {row_no}: the chromosome name '{}' is not valid UTF-8, which VCF \
             requires",
            name.escape_ascii()
        )
    })
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
fn scan_depth_from(reader: impl BufRead, path: &Path) -> Result<DepthStats> {
    let mut counter = DepthCounter::default();
    let mut lines = Lines::new(reader, path);
    while let Some(line) = lines.next()? {
        if let Line::Record { fields: f, .. } = line {
            let info = scan_info(f[7]);
            if is_base_record(&f, &info) {
                counter.add(info.dp.map(digits_u64));
            }
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

/// Read an all-sites VCF from `bcftools call -c` the way `vcfutils.pl vcf2fq`
/// and `fq2psmcfa` read it; see the module docs.
///
/// [`DepthThresholds::Auto`] reads the file twice, so `path` must then be a
/// regular file. Otherwise it is read once and may be a pipe or FIFO.
pub fn read_vcf_consensus(path: &Path, opts: &ConsensusOptions) -> Result<ConsensusRead> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    read_consensus(path, opts, || open_reader(path))
}

/// [`read_vcf_consensus`] of `path` as `open` opens it (the caller checks
/// the bin size); [`DepthThresholds::Auto`] opens `path` again for its
/// second pass.
pub(crate) fn read_consensus(
    path: &Path,
    opts: &ConsensusOptions,
    open: impl FnOnce() -> Result<Box<dyn BufRead>>,
) -> Result<ConsensusRead> {
    let context = || format!("failed to read VCF {path:?} in consensus mode");
    let (reader, depth) = match opts.filters.depth {
        DepthThresholds::Fixed { min, max } => (open()?, (min, max)),
        DepthThresholds::Auto => {
            if !is_regular_file(path) {
                bail!(
                    "automatic depth thresholds need two passes over the VCF (the first one \
                     measures the mean depth), but {path:?} is not a regular file and can be \
                     read only once (a pipe or FIFO); give the depth thresholds explicitly, or \
                     write the VCF to a file first"
                );
            }
            let stats = scan_depth_from(open()?, path).with_context(context)?;
            if stats.records == 0 {
                bail!(
                    "cannot choose depth thresholds automatically: no SNP/reference record in \
                     {path:?} has INFO/DP; give the thresholds explicitly"
                );
            }
            (reopen_reader(path)?, auto_depth_thresholds(stats.mean))
        }
    };
    read_consensus_from(reader, path, opts, depth).with_context(context)
}

/// [`read_vcf_consensus`] over an opened input, with the INFO/DP bounds
/// `(min_depth, max_depth)` that `opts.filters.depth` chose; `path` names the
/// input in messages.
fn read_consensus_from(
    reader: impl BufRead,
    path: &Path,
    opts: &ConsensusOptions,
    (min_depth, max_depth): (u64, u64),
) -> Result<ConsensusRead> {
    let filters = &opts.filters;
    let t = Thresholds {
        min_depth,
        max_depth,
        min_mq: filters.min_mq,
        min_qual: filters.min_qual,
    };
    let mut summary = ConsensusSummary {
        min_depth,
        max_depth,
        auto_depth: filters.depth == DepthThresholds::Auto,
        min_mq: filters.min_mq,
        indel_window: filters.indel_window,
        min_qual: filters.min_qual,
        ..ConsensusSummary::default()
    };
    let masks = MaskEngine::new(&opts.masks, &opts.negative_masks)?;
    summary.masked = !opts.masks.is_empty() || !opts.negative_masks.is_empty();
    let l = filters.indel_window;

    let mut depth = DepthCounter::default();
    let mut seen = HashSet::<Vec<u8>>::new();
    let mut repeated = HashSet::<Vec<u8>>::new();
    let mut seqs = Vec::new();
    let mut cur: Option<Chrom> = None;

    let mut lines = Lines::new(reader, path);
    while let Some(line) = lines.next()? {
        let (row_no, f) = match line {
            Line::Header(header) => {
                if header.starts_with(b"#CHROM") {
                    summary.samples = header.split(|&b| b == b'\t').count().saturating_sub(9);
                }
                continue;
            }
            Line::Record { row_no, fields } => (row_no, fields),
        };
        summary.records += 1;
        let pos = parse_pos(f[1], row_no)?;

        if cur.as_ref().is_none_or(|c| c.name.as_bytes() != f[0]) {
            if let Some(done) = cur.take() {
                end_chrom(done, false, &mut summary, &mut seqs)?;
            }
            let name = utf8_chrom(f[0], row_no)?;
            // Each name is listed once, from its first or second block, so
            // no list is searched (assemblies can have a million scaffolds).
            let first_block = seen.insert(f[0].to_vec());
            if !first_block && repeated.insert(f[0].to_vec()) {
                summary.repeated_chroms.push(name.to_string());
            }
            if first_block {
                masks.note_coverage(
                    name,
                    &mut summary.chroms_outside_mask,
                    &mut summary.chroms_outside_some_masks,
                );
            }
            let uncallable = masks.uncallable_for_chrom(name);
            cur = Some(Chrom::new(
                name,
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
                c.name,
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
        summary.lost.add_all(&s.lost);
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
    if c.name == "0" && !last {
        summary.discarded_chroms.push("0".to_string());
        return Ok(());
    }
    seqs.push(c.finish()?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn perl_num_reads_as_the_float_parser_does() {
        // Digit strings of up to 17 digits (the short ones take the fast
        // path), with a decimal point anywhere or none, and either sign.
        let mut rng = SmallRng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
        for len in 1..=17 {
            for _ in 0..2000 {
                let mut s: String = (0..len)
                    .map(|_| char::from(b'0' + rng.gen_range(0..10) as u8))
                    .collect();
                let dot = rng.gen_range(0..len + 2);
                if dot <= len {
                    s.insert(dot, '.');
                }
                if rng.gen_range(0..2) == 1 {
                    s.insert(0, '-');
                }
                let want: f64 = s.parse().unwrap();
                assert_eq!(perl_num(s.as_bytes()).to_bits(), want.to_bits(), "{s}");
            }
        }
        assert_eq!(perl_num(b"-29.9905"), -29.9905);
        assert_eq!(perl_num(b"-0").to_bits(), (-0.0f64).to_bits());
        assert_eq!(perl_num(b"1.2.3"), 1.2);
        assert_eq!(perl_num(b"-."), 0.0);
    }

    /// What a regular expression `key(value)` captures in `info`: the
    /// value at the leftmost `key` where `value` matches `len` bytes, more
    /// than 0.
    fn leftmost<'a>(info: &'a [u8], key: &[u8], len: fn(&[u8]) -> usize) -> Option<&'a [u8]> {
        (0..info.len()).find_map(|i| {
            let v = info[i..].strip_prefix(key)?;
            Some(&v[..len(v)]).filter(|v| !v.is_empty())
        })
    }

    #[test]
    fn info_values_are_the_leftmost_matches_of_vcf2fq_patterns() {
        let digits: fn(&[u8]) -> usize = |v| run_len(v, is_digit);
        let number: fn(&[u8]) -> usize = |v| run_len(v, is_num_char);
        // `-?[\d.]+`: the '-' only before a number.
        let signed: fn(&[u8]) -> usize = |v| {
            let sign = usize::from(v.first() == Some(&b'-'));
            match run_len(&v[sign..], is_num_char) {
                0 => 0,
                k => sign + k,
            }
        };
        let tokens: [&[u8]; 16] = [
            b"DP=", b"MQ=", b"FQ=", b"AF1=", b"INDEL", b"IND", b"EL", b"A", b"D", b"F", b"=", b"-",
            b".", b"7", b"42", b";",
        ];
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..50_000 {
            let n = rng.gen_range(0..12);
            let info: Vec<u8> = (0..n)
                .flat_map(|_| tokens[rng.gen_range(0..tokens.len())])
                .copied()
                .collect();
            let got = scan_info(&info);
            let text = String::from_utf8_lossy(&info);
            assert_eq!(got.dp, leftmost(&info, b"DP=", digits), "{text}");
            assert_eq!(got.mq, leftmost(&info, b"MQ=", digits), "{text}");
            assert_eq!(got.af1, leftmost(&info, b"AF1=", number), "{text}");
            assert_eq!(got.fq, leftmost(&info, b"FQ=", signed), "{text}");
            let indel = info.windows(5).any(|w| w == b"INDEL");
            assert_eq!(got.indel, indel, "{text}");
        }
    }
}
