//! One loader for every input format, shared by inference, conversion and QC.
//!
//! [`load`] reads a psmcfa, multihetsep or VCF file into binned
//! [`Observations`] (with the callable bases of each bin, except for psmcfa
//! input and bins too large to count; see [`crate::io::callable`]) plus what
//! QC and the writers need:
//! per-sequence counts, the sequences the output filter dropped and why, a
//! per-format summary and, on request, the heterozygous sites with their
//! multihetsep `nr_called`.
//!
//! A VCF is read in one of two modes ([`VcfMode`]): `Consensus` reproduces the
//! classic `bcftools call -c | vcfutils.pl vcf2fq | fq2psmcfa` pipeline from an
//! all-sites VCF ([`crate::io::consensus`]), `Mask` classifies the genotypes
//! of one sample against callable masks ([`crate::io::vcf`]). `Auto` picks one
//! from the first records ([`detect_vcf_mode`]).
//!
//! Every VCF mode and multihetsep input then go through `fq2psmcfa`'s
//! per-sequence output filter ([`SeqFilter`]). psmcfa input is not filtered:
//! the tool that wrote it already did.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::{BufRead, Cursor, Read};
use std::path::{Path, PathBuf};

use crate::io::callable::join_counts;
use crate::io::common::{BinnedSeq, HetCall, chunk_sequences, open_reader};
use crate::io::consensus::{
    ConsensusFilters, ConsensusOptions, ConsensusSeq, ConsensusSummary, info_has_fq, read_consensus,
};
use crate::io::mhs::{MhsOptions, MhsSummary, read_mhs_seqs_from};
use crate::io::psmcfa::{PsmcfaRecord, parse_psmcfa_with_stats};
use crate::io::vcf::{
    VcfOptions, VcfSummary, gvcf_block_end, list_names, no_het_snp_message, read_vcf_seqs_from,
};
use crate::io::{BinCounts, CallableBases, Observations, Sequences, sequence_ranges};
use crate::qc::fmt::{count, counted, plural};

/// Input file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InputFormat {
    /// PSMC's binned FASTA (`fq2psmcfa` output).
    Psmcfa,
    /// MSMC's multihetsep.
    Mhs,
    /// VCF (plain, gzip or bgzip).
    Vcf,
}

impl InputFormat {
    /// Lowercase name, as on the command line.
    pub fn name(self) -> &'static str {
        match self {
            InputFormat::Psmcfa => "psmcfa",
            InputFormat::Mhs => "mhs",
            InputFormat::Vcf => "vcf",
        }
    }

    /// Guess the format from the content (gzip is decoded first, a UTF-8
    /// byte order mark dropped and BCF refused, see
    /// [`crate::io::common::open_reader`]), skipping blank lines:
    ///
    /// - a line starting with `##` (VCF meta-information such as
    ///   `##fileformat`, `##contig` or `##source`) or `#CHROM` makes it a VCF;
    /// - other lines starting with `#` are skipped as comments;
    /// - the first other line decides: `>` or `@` starts a psmcfa file, a line
    ///   of at least 8 tab-separated fields is a VCF record (of a VCF without
    ///   header, which `vcfutils.pl vcf2fq` reads too), and anything else is
    ///   multihetsep, as is a file of comments only.
    pub fn sniff(path: &Path) -> Result<InputFormat> {
        sniff_format(&mut Input::open(path)?, path)
    }
}

/// [`InputFormat::sniff`] on the first lines of `input`.
fn sniff_format(input: &mut Input, path: &Path) -> Result<InputFormat> {
    let mut comments = false;
    loop {
        let Some(line) = input.next_line(path)? else {
            if comments {
                return Ok(InputFormat::Mhs);
            }
            bail!("{path:?} is empty, so its format cannot be recognised");
        };
        let s = line.trim_ascii();
        if s.is_empty() {
            continue;
        }
        if s[0] == b'#' {
            if s.starts_with(b"##") || s.starts_with(b"#CHROM") {
                return Ok(InputFormat::Vcf);
            }
            comments = true;
            continue;
        }
        return Ok(if s[0] == b'>' || s[0] == b'@' {
            InputFormat::Psmcfa
        } else if s.split(|&b| b == b'\t').nth(7).is_some() {
            InputFormat::Vcf
        } else {
            InputFormat::Mhs
        });
    }
}

/// The input of [`load`], opened once.
///
/// Recognising the format and the VCF mode reads the first lines of the
/// input. They are kept, and [`Input::into_reader`] reads them again in front
/// of the rest, so that the input is read only once: a pipe or FIFO (such as
/// `/dev/stdin` or `<(bcftools call ...)`) cannot be opened a second time.
struct Input {
    reader: Box<dyn BufRead>,
    /// Every byte read so far.
    kept: Vec<u8>,
}

impl Input {
    fn open(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: open_reader(path)?,
            kept: Vec::new(),
        })
    }

    /// The next line, with its line end, or `None` at the end of the input.
    fn next_line(&mut self, path: &Path) -> Result<Option<&[u8]>> {
        let start = self.kept.len();
        let n = self
            .reader
            .read_until(b'\n', &mut self.kept)
            .with_context(|| format!("failed to read {path:?}"))?;
        Ok((n > 0).then(|| &self.kept[start..]))
    }

    /// The sample names of the `#CHROM` line of a VCF header, or `None` when
    /// a data line, a blank line or the end of the input comes first.
    fn vcf_samples(&mut self, path: &Path) -> Result<Option<Vec<String>>> {
        let mut start = 0;
        loop {
            if start >= self.kept.len() && self.next_line(path)?.is_none() {
                return Ok(None);
            }
            let end = self.kept[start..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(self.kept.len(), |i| start + i);
            let line = &self.kept[start..end];
            if line.first() != Some(&b'#') {
                return Ok(None);
            }
            if line.starts_with(b"#CHROM") {
                let text = String::from_utf8_lossy(line.trim_ascii_end());
                return Ok(Some(text.split('\t').skip(9).map(str::to_string).collect()));
            }
            start = end + 1;
        }
    }

    /// The whole input, from its first byte.
    fn into_reader(self) -> Box<dyn BufRead> {
        if self.kept.is_empty() {
            self.reader
        } else {
            Box::new(Cursor::new(self.kept).chain(self.reader))
        }
    }
}

/// How a VCF is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub enum VcfMode {
    /// Pick `Consensus` or `Mask` from the first records ([`detect_vcf_mode`]).
    #[default]
    Auto,
    /// An all-sites VCF from `bcftools call -c`, read like `vcfutils.pl
    /// vcf2fq` and `fq2psmcfa` ([`crate::io::consensus`]).
    Consensus,
    /// The genotypes of one sample plus callable masks ([`crate::io::vcf`]).
    Mask,
}

impl VcfMode {
    /// Lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            VcfMode::Auto => "auto",
            VcfMode::Consensus => "consensus",
            VcfMode::Mask => "mask",
        }
    }
}

/// Data records [`detect_vcf_mode`] inspects at most.
pub const MODE_DETECTION_RECORDS: u64 = 10_000;

/// Bytes of data records [`detect_vcf_mode`] reads at most. [`load`] keeps
/// the lines it inspects in memory, so for VCFs with very many samples it
/// inspects fewer than [`MODE_DETECTION_RECORDS`] records (at least one).
pub const MODE_DETECTION_BYTES: usize = 64 << 20;

/// The mode a VCF is read in, and why.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct VcfModeChoice {
    /// `Consensus` or `Mask`, never `Auto`.
    pub mode: VcfMode,
    /// Whether the mode was requested rather than detected.
    pub forced: bool,
    /// One sentence for the user.
    pub reason: String,
    /// Data records inspected: by the detection, or, when mask mode is
    /// forced, for gVCF blocks (0 when consensus mode is forced).
    pub inspected: u64,
    /// Of those, records with ALT `.`.
    pub alt_dot: u64,
    /// Of those, records with INFO/FQ.
    pub alt_dot_with_fq: u64,
    /// Inspected records with INFO/FQ.
    pub records_with_fq: u64,
    /// Positions from the first to the last inspected record of each
    /// chromosome, summed. An all-sites VCF has about one record per
    /// position, a VCF of variant sites far fewer.
    pub span: u64,
    /// Inspected records that are gVCF reference or no-call blocks (ALT
    /// `.`, `<*>`, `<NON_REF>` or `<X>` with INFO/END). With any, the VCF is
    /// a gVCF: it is read in mask mode, and positions that no record covers
    /// are missing ([`crate::io::vcf::VcfOptions::gvcf`]).
    pub gvcf_blocks: u64,
}

/// The [`VcfMode::Auto`] rule, applied to the first [`MODE_DETECTION_RECORDS`]
/// data records:
///
/// - If any is a gVCF reference or no-call block (ALT `.`, `<*>`,
///   `<NON_REF>` or `<X>` with INFO/END, as `bcftools call -g`, GATK,
///   DeepVariant and Strelka2 write them), the file is a gVCF and is read in
///   mask mode ([`VcfModeChoice::gvcf_blocks`]). Consensus mode, like
///   vcf2fq, would read a block as its first position alone.
/// - If at least 50% have ALT `.`, the file is an all-sites VCF. It is read
///   in consensus mode when at least 90% of those carry INFO/FQ (as
///   `bcftools call -c` writes), and is an error otherwise, because only the
///   consensus caller's FQ tells vcf2fq how reliable a reference call is.
/// - Otherwise it is also read in consensus mode when the records are at
///   least half of the positions they [`span`](VcfModeChoice::span) (an
///   all-sites VCF) and at least 90% of them carry INFO/FQ. This is
///   `bcftools call -c -A` at high depth, which gives most reference sites an
///   ALT allele seen in some read. A `bcftools call -cv` VCF of variant sites
///   also carries FQ, but is far sparser.
/// - Any other VCF is read in mask mode.
pub fn detect_vcf_mode(path: &Path) -> Result<VcfModeChoice> {
    detect_mode(&mut Input::open(path)?, path)
}

/// [`detect_vcf_mode`] on the next lines of `input`.
fn detect_mode(input: &mut Input, path: &Path) -> Result<VcfModeChoice> {
    choose_mode(inspect_records(input, path)?, path)
}

/// The counts of [`VcfModeChoice`] over the next data records of `input`,
/// at most [`MODE_DETECTION_RECORDS`] of them in [`MODE_DETECTION_BYTES`];
/// the mode is left to [`choose_mode`].
fn inspect_records(input: &mut Input, path: &Path) -> Result<VcfModeChoice> {
    let mut seen = VcfModeChoice::default();
    // The chromosome of the current block, and its lowest and highest POS.
    let mut block: Option<(Vec<u8>, u64, u64)> = None;
    let mut data_bytes = 0usize;
    while seen.inspected < MODE_DETECTION_RECORDS && data_bytes < MODE_DETECTION_BYTES {
        let Some(line) = input.next_line(path)? else {
            break;
        };
        if line.first() == Some(&b'#') || line.trim_ascii().is_empty() {
            continue;
        }
        data_bytes += line.len();
        let mut cols = line.trim_ascii_end().split(|&b| b == b'\t');
        let (Some(chrom), Some(pos), Some(alt), Some(info)) =
            (cols.next(), cols.next(), cols.nth(2), cols.nth(2))
        else {
            continue;
        };
        seen.inspected += 1;
        let has_fq = info_has_fq(info);
        seen.records_with_fq += u64::from(has_fq);
        if alt == b"." {
            seen.alt_dot += 1;
            seen.alt_dot_with_fq += u64::from(has_fq);
        }
        let Some(pos) = std::str::from_utf8(pos)
            .ok()
            .and_then(|p| p.parse::<u64>().ok())
        else {
            continue;
        };
        seen.gvcf_blocks += u64::from(gvcf_block_end(pos, alt, info).is_some());
        match &mut block {
            Some((c, lo, hi)) if c.as_slice() == chrom => {
                *lo = (*lo).min(pos);
                *hi = (*hi).max(pos);
            }
            _ => {
                if let Some((_, lo, hi)) = block.replace((chrom.to_vec(), pos, pos)) {
                    seen.span += hi - lo + 1;
                }
            }
        }
    }
    if let Some((_, lo, hi)) = block {
        seen.span += hi - lo + 1;
    }
    Ok(seen)
}

/// The [`detect_vcf_mode`] rule applied to the counts of [`inspect_records`].
fn choose_mode(seen: VcfModeChoice, path: &Path) -> Result<VcfModeChoice> {
    let VcfModeChoice {
        inspected,
        alt_dot,
        alt_dot_with_fq: with_fq,
        records_with_fq,
        span,
        gvcf_blocks,
        ..
    } = seen;
    let choice = |mode, reason: String| VcfModeChoice {
        mode,
        reason,
        ..seen.clone()
    };
    if inspected == 0 {
        return Ok(choice(
            VcfMode::Mask,
            "mask mode: the VCF has no data records".to_string(),
        ));
    }
    if gvcf_blocks > 0 {
        return Ok(choice(
            VcfMode::Mask,
            format!(
                "mask mode: {} of the first {} records are reference or no-call blocks with \
                 INFO/END, as in a gVCF, so positions that no record covers are missing",
                count(gvcf_blocks),
                count(inspected)
            ),
        ));
    }
    if 2 * alt_dot < inspected {
        let dense = span > 0 && 2 * inspected >= span;
        let density = if span == 0 {
            0.0
        } else {
            100.0 * inspected as f64 / span as f64
        };
        if dense && 10 * records_with_fq >= 9 * inspected {
            return Ok(choice(
                VcfMode::Consensus,
                format!(
                    "consensus mode: only {} of the first {} records have ALT '.', but there is \
                     a record for {density:.0}% of the positions they span and {} carry INFO/FQ, \
                     as in an all-sites VCF from 'bcftools call -c -A', which gives reference \
                     sites ALT alleles",
                    count(alt_dot),
                    count(inspected),
                    count(records_with_fq)
                ),
            ));
        }
        let why = if dense {
            format!(
                "only {} carry the INFO/FQ of 'bcftools call -c', which consensus mode needs",
                count(records_with_fq)
            )
        } else {
            format!(
                "there is a record for only {density:.1}% of the positions they span, as in a \
                 VCF of variant sites"
            )
        };
        return Ok(choice(
            VcfMode::Mask,
            format!(
                "mask mode: {} of the first {} records have ALT '.' (fewer than half), and {why}",
                count(alt_dot),
                count(inspected)
            ),
        ));
    }
    if 10 * with_fq >= 9 * alt_dot {
        return Ok(choice(
            VcfMode::Consensus,
            format!(
                "consensus mode: {} of the first {} records have ALT '.' and {} of those carry \
                 INFO/FQ, as in an all-sites VCF from 'bcftools call -c'",
                count(alt_dot),
                count(inspected),
                count(with_fq)
            ),
        ));
    }
    bail!(
        "{path:?} looks like an all-sites VCF ({} of the first {} records have ALT '.'), but \
         only {} of those carry INFO/FQ, which the consensus mode (vcfutils.pl vcf2fq) needs to \
         tell reliable reference calls from unreliable ones. Rerun the caller as 'bcftools call \
         -c' (the classic PSMC pipeline), or provide a callable mask and force mask mode, which \
         reads the genotypes",
        count(alt_dot),
        count(inspected),
        count(with_fq)
    )
}

/// `fq2psmcfa`'s per-sequence output filter: a sequence is kept when at least
/// `min_good_bases` of its positions are called and they make up at least
/// `min_good_fraction` of the sequence. 0 disables either test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SeqFilter {
    /// `fq2psmcfa -g`, default 10000.
    pub min_good_bases: u64,
    /// Default 0.2 (fixed in `fq2psmcfa`).
    pub min_good_fraction: f64,
}

impl Default for SeqFilter {
    fn default() -> Self {
        Self {
            min_good_bases: 10_000,
            min_good_fraction: 0.2,
        }
    }
}

impl SeqFilter {
    /// A filter that keeps every non-empty sequence.
    pub const DISABLED: SeqFilter = SeqFilter {
        min_good_bases: 0,
        min_good_fraction: 0.0,
    };

    /// Why a sequence of `extent` positions with `good` called ones is
    /// dropped, or `None` when it is kept. The fraction is compared as
    /// `fq2psmcfa` compares it, in double precision.
    pub fn verdict(&self, good: u64, extent: u64) -> Option<DropReason> {
        let few_good_bases = good < self.min_good_bases;
        // An empty sequence gives NaN, which fails the test as in C.
        let ratio = good as f64 / extent as f64;
        let low_good_fraction = self.min_good_fraction > 0.0
            && matches!(
                ratio.partial_cmp(&self.min_good_fraction),
                None | Some(Ordering::Less)
            );
        (few_good_bases || low_good_fraction).then_some(DropReason::Filter {
            few_good_bases,
            low_good_fraction,
        })
    }
}

/// Options for [`load`].
#[derive(Debug, Clone)]
pub struct InputOptions {
    /// The file format; `None` recognises it from the content
    /// ([`InputFormat::sniff`]).
    pub format: Option<InputFormat>,
    /// Bases per bin for multihetsep and VCF input (`fq2psmcfa -s`); psmcfa
    /// input is already binned.
    pub bin_size: usize,
    /// Split each sequence into HMM rows of at most this many bins, chained in
    /// one HMM sequence ([`chunk_sequences`]).
    pub batch_size: Option<usize>,
    /// Multihetsep: the two haplotypes ([`MhsOptions::pair`]).
    pub mhs_pair: (usize, usize),
    /// VCF: how to read it (default [`VcfMode::Auto`]).
    pub vcf_mode: VcfMode,
    /// VCF mask mode: the sample to read ([`VcfOptions::sample`]). Consensus
    /// mode reads only INFO, which bcftools computes over all samples, so it
    /// cannot select a sample: there a VCF with several samples is an error,
    /// and so is a name that the header lacks.
    pub sample: Option<String>,
    /// VCF: callable masks ([`VcfOptions::masks`]); in consensus mode they
    /// are an optional extra filter.
    pub masks: Vec<PathBuf>,
    /// VCF: masks of positions to exclude.
    pub negative_masks: Vec<PathBuf>,
    /// VCF mask mode: [`VcfOptions::keep_filtered`].
    pub keep_filtered: bool,
    /// VCF mask mode: [`VcfOptions::to_contig_end`]. Consensus sequences end
    /// at their last record, as in vcf2fq.
    pub to_contig_end: bool,
    /// VCF consensus mode: the vcf2fq and fq2psmcfa thresholds.
    pub consensus: ConsensusFilters,
    /// The output filter for VCF and multihetsep input.
    pub filter: SeqFilter,
    /// Collect [`Dataset::het_sites`] (VCF and multihetsep input).
    pub collect_het_sites: bool,
}

impl Default for InputOptions {
    fn default() -> Self {
        let vcf = VcfOptions::default();
        Self {
            format: None,
            bin_size: 100,
            batch_size: None,
            mhs_pair: MhsOptions::default().pair,
            vcf_mode: VcfMode::Auto,
            sample: None,
            masks: Vec::new(),
            negative_masks: Vec::new(),
            keep_filtered: vcf.keep_filtered,
            to_contig_end: vcf.to_contig_end,
            consensus: ConsensusFilters::default(),
            filter: SeqFilter::default(),
            collect_het_sites: false,
        }
    }
}

/// One kept sequence, in the order of [`Dataset::obs`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeqInfo {
    /// The psmcfa record name, VCF chromosome or multihetsep first column.
    pub name: String,
    /// Bins of the sequence (over all of its rows).
    pub n_bins: usize,
    /// Positions the sequence covers (`1..=extent`), when the input has them.
    pub extent: Option<u64>,
    /// Of those, positions called (heterozygous or not).
    pub called: Option<u64>,
    /// Heterozygous positions.
    pub het_sites: Option<u64>,
}

/// Why a sequence is not in [`Dataset::obs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DropReason {
    /// The sequence has no bins (an empty psmcfa record, or a VCF chromosome
    /// with no base).
    Empty,
    /// [`SeqFilter`] rejected it, for one or both reasons.
    Filter {
        /// Fewer called positions than [`SeqFilter::min_good_bases`].
        few_good_bases: bool,
        /// Called positions below [`SeqFilter::min_good_fraction`] of the
        /// sequence.
        low_good_fraction: bool,
    },
}

/// A sequence of the input that was dropped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DroppedSeq {
    pub name: String,
    pub n_bins: usize,
    /// Positions the sequence covers, when the input has them.
    pub extent: Option<u64>,
    /// Called positions.
    pub good: Option<u64>,
    pub reason: DropReason,
}

impl DroppedSeq {
    /// One line for the user, e.g. `chrM: 16,569 bases, 0 called (< 10,000)`.
    pub fn describe(&self, filter: &SeqFilter) -> String {
        match self.reason {
            DropReason::Empty => format!("{}: empty", self.name),
            DropReason::Filter {
                few_good_bases,
                low_good_fraction,
            } => {
                let good = self.good.unwrap_or(0);
                let extent = self.extent.unwrap_or(0);
                let mut why = Vec::new();
                if few_good_bases {
                    why.push(format!("< {}", count(filter.min_good_bases)));
                }
                if low_good_fraction {
                    why.push(format!(
                        "< {}% of its length",
                        filter.min_good_fraction * 100.0
                    ));
                }
                format!(
                    "{}: {} bases, {} called ({})",
                    self.name,
                    count(extent),
                    count(good),
                    why.join(", ")
                )
            }
        }
    }
}

/// psmcfa input counts, as C psmc reports them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PsmcfaSummary {
    /// Records in the file.
    pub records: usize,
    /// Records without any bin (not analysed).
    pub empty_records: usize,
    /// Bins over all records.
    pub bins: u64,
    /// Non-missing bins (C psmc's `sum_L`).
    pub called_bins: u64,
    /// Heterozygous bins (C psmc's `sum_n`).
    pub het_bins: u64,
    /// Records with a FASTQ quality block (a `+` line), which `fq2psmcfa`
    /// output never has.
    pub fastq_records: usize,
    /// Bins written as letters `fq2psmcfa` never writes: `A`, `C`, `G`, the
    /// other IUPAC codes, or lowercase letters.
    pub base_letters: u64,
}

impl PsmcfaSummary {
    /// Whether the file looks like a per-base consensus (the FASTQ of
    /// `vcfutils.pl vcf2fq`, or a FASTA of bases) rather than `fq2psmcfa`
    /// output: a record has a FASTQ quality block, or at least 1% of the
    /// bins are written as base letters.
    pub fn looks_like_per_base_consensus(&self) -> bool {
        self.fastq_records > 0 || self.base_letters.saturating_mul(100) >= self.bins.max(1)
    }

    /// Warnings worth showing to the user about how the input was read.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.looks_like_per_base_consensus() {
            out.push(format!(
                "the psmcfa input looks like a per-base consensus (such as the FASTQ of \
                 vcfutils.pl vcf2fq) rather than fq2psmcfa output: {} of its {} records have a \
                 FASTQ quality block and {} of its {} bins are written as A, C, G, other IUPAC \
                 codes or lowercase letters, which fq2psmcfa never writes. Like C psmc, psmc-rs \
                 reads every character as one bin and lowercase (filtered) bases as called \
                 bins, so each base became a bin and the missing data are undercounted; the \
                 time scale, theta and Ne come out as if each base were a whole bin. Bin the \
                 consensus with fq2psmcfa first, or give psmc-rs the VCF it was made from",
                count(self.fastq_records as u64),
                count(self.records as u64),
                count(self.base_letters),
                count(self.bins)
            ));
        }
        out
    }
}

/// The mode-specific part of [`VcfInputSummary`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VcfDetails {
    /// Filter counts, depth statistics and warnings of consensus mode.
    Consensus(ConsensusSummary),
    /// Genotype, mask and depth counts of mask mode.
    Mask(VcfSummary),
}

/// What [`load`] read from a VCF.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VcfInputSummary {
    /// The mode the VCF was read in, and why.
    pub mode: VcfModeChoice,
    pub details: VcfDetails,
    /// Mask mode: whether records with a FILTER other than PASS were used
    /// ([`InputOptions::keep_filtered`]).
    pub keep_filtered: bool,
}

/// Per-format summary of the input.
// One value per load, so the size of the VCF variant does not matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SourceSummary {
    Psmcfa(PsmcfaSummary),
    Mhs(MhsSummary),
    Vcf(VcfInputSummary),
}

impl SourceSummary {
    /// The input format this summary belongs to.
    pub fn format(&self) -> InputFormat {
        match self {
            SourceSummary::Psmcfa(_) => InputFormat::Psmcfa,
            SourceSummary::Mhs(_) => InputFormat::Mhs,
            SourceSummary::Vcf(_) => InputFormat::Vcf,
        }
    }
}

/// A heterozygous site, for multihetsep output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HetSite {
    /// Index into [`Dataset::seqs`].
    pub seq: u32,
    /// 1-based position.
    pub pos: u64,
    /// Called positions since the previous heterozygous site of the same
    /// sequence (exclusive) up to and including this one, as
    /// `generate_multihetsep.py` counts them (but restarting with each
    /// sequence).
    pub nr_called: u64,
    /// The two alleles: in GT order for VCF mask mode (phased order for a
    /// phased GT), REF then ALT in consensus mode, and the pair's alleles in
    /// the first ordering for multihetsep input.
    pub alleles: [u8; 2],
}

/// Everything [`load`] read.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// The kept sequences as HMM rows. Empty when every sequence was dropped
    /// ([`Dataset::ensure_usable`] turns that into an error). Their
    /// callable bases ([`Observations::callable`]) are there for VCF and
    /// multihetsep input binned into bins of at most
    /// [`MAX_COUNTED_BIN_SIZE`](crate::io::MAX_COUNTED_BIN_SIZE) bases, even
    /// when every sequence was dropped, and not for psmcfa input.
    pub obs: Observations,
    /// One entry per kept sequence, in the order of `obs`.
    pub seqs: Vec<SeqInfo>,
    /// Sequences of the input that are not in `obs`, in input order.
    pub dropped: Vec<DroppedSeq>,
    /// What the input held, per format.
    pub source: SourceSummary,
    /// Heterozygous sites of the kept sequences, in order, when
    /// [`InputOptions::collect_het_sites`] was set and the input has
    /// positions (not psmcfa).
    pub het_sites: Option<Vec<HetSite>>,
    /// The output filter that was applied (`None` for psmcfa input).
    pub filter: Option<SeqFilter>,
}

impl Dataset {
    /// Each kept sequence with its bins (rows of one sequence joined).
    pub fn sequences(&self) -> Vec<(&SeqInfo, Cow<'_, [u8]>)> {
        let rows = &self.obs.rows;
        let bins = sequence_ranges(&self.obs.row_starts).into_iter().map(|r| {
            if r.len() == 1 {
                Cow::Borrowed(rows[r.start].as_slice())
            } else {
                Cow::Owned(rows[r].concat())
            }
        });
        self.seqs.iter().zip(bins).collect()
    }

    /// The callable bases of each bin of each kept sequence, joined as
    /// [`Dataset::sequences`] joins the bins; `None` for input without them
    /// (psmcfa). See [`crate::io::callable`].
    pub fn sequence_callable(&self) -> Option<Vec<Cow<'_, BinCounts>>> {
        let counts = self.obs.callable.as_ref()?.rows();
        // Only the kept sequences, as `sequences` pairs them with their bins.
        let ranges = sequence_ranges(&self.obs.row_starts)
            .into_iter()
            .take(self.seqs.len());
        Some(ranges.map(|r| join_counts(counts, r)).collect())
    }

    /// Bins over all kept sequences.
    pub fn n_bins(&self) -> usize {
        self.obs.rows.iter().map(Vec::len).sum()
    }

    /// Heterozygous bins over all kept sequences.
    pub fn het_bins(&self) -> usize {
        self.obs.rows.iter().flatten().filter(|&&b| b == 1).count()
    }

    /// Warnings worth showing to the user about how the input was read.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = match &self.source {
            SourceSummary::Vcf(v) => match &v.details {
                VcfDetails::Consensus(c) => c.warnings(),
                VcfDetails::Mask(m) => m.warnings(),
            },
            SourceSummary::Psmcfa(p) => p.warnings(),
            SourceSummary::Mhs(_) => Vec::new(),
        };
        let filtered: Vec<&DroppedSeq> = self
            .dropped
            .iter()
            .filter(|d| matches!(d.reason, DropReason::Filter { .. }))
            .collect();
        if let (Some(filter), false) = (&self.filter, filtered.is_empty()) {
            let n = filtered.len();
            out.push(format!(
                "{} dropped because fewer than {} of {} positions, or less than {}% of them, are \
                 called (fq2psmcfa's output filter): {}",
                counted(n, "sequence", "sequences"),
                count(filter.min_good_bases),
                plural(n, "its", "their"),
                filter.min_good_fraction * 100.0,
                describe_dropped(filtered.iter().copied(), filter)
            ));
        }
        if !self.obs.rows.is_empty() && self.het_bins() == 0 {
            out.push("no bin is heterozygous, so PSMC cannot be fitted to this input".to_string());
        }
        out
    }

    /// Fail unless the data can be analysed: at least one sequence is left and
    /// at least one bin is heterozygous.
    pub fn ensure_usable(&self) -> Result<()> {
        if self.obs.rows.is_empty() {
            let mut msg = format!(
                "no sequence of the {} input is left to analyse",
                self.source.format().name()
            );
            if let Some(filter) = &self.filter
                && !self.dropped.is_empty()
            {
                let n = self.dropped.len();
                msg.push_str(&format!(
                    ": the output filter (at least {} called positions making up at least {}% of \
                     the sequence, as in fq2psmcfa) dropped {} ({}); lower or disable it for \
                     small test data",
                    count(filter.min_good_bases),
                    filter.min_good_fraction * 100.0,
                    counted(n, "sequence", "sequences"),
                    describe_dropped(self.dropped.iter(), filter),
                ));
            }
            bail!("{msg}");
        }
        if self.het_bins() == 0 {
            if let SourceSummary::Vcf(VcfInputSummary {
                details: VcfDetails::Mask(m),
                keep_filtered,
                ..
            }) = &self.source
                && m.het_snps == 0
            {
                bail!("{}", no_het_snp_message(m, *keep_filtered));
            }
            bail!(
                "no bin of the {} input is heterozygous, so PSMC cannot be fitted",
                self.source.format().name()
            );
        }
        Ok(())
    }
}

/// Up to five of `dropped`, described ([`DroppedSeq::describe`]) and
/// separated by `; `, then `...` if there are more.
fn describe_dropped<'a>(
    dropped: impl ExactSizeIterator<Item = &'a DroppedSeq>,
    filter: &SeqFilter,
) -> String {
    let n = dropped.len();
    let mut shown: Vec<String> = dropped.take(5).map(|d| d.describe(filter)).collect();
    if n > 5 {
        shown.push("...".to_string());
    }
    shown.join("; ")
}

/// One sequence as a reader produced it, before the output filter.
struct RawSeq {
    name: String,
    bins: Vec<u8>,
    /// The callable bases of each bin, from every reader but psmcfa's, for
    /// bins small enough to count.
    callable: Option<BinCounts>,
    extent: Option<u64>,
    called: Option<u64>,
    het: Option<u64>,
    het_sites: Vec<HetCall>,
}

impl RawSeq {
    /// Why the sequence is dropped, with `filter` the output filter of its
    /// input (if any): an empty sequence always is.
    fn drop_reason(&self, filter: Option<&SeqFilter>) -> Option<DropReason> {
        if self.bins.is_empty() {
            return Some(DropReason::Empty);
        }
        filter?.verdict(self.called?, self.extent?)
    }
}

impl From<PsmcfaRecord> for RawSeq {
    fn from(r: PsmcfaRecord) -> Self {
        Self {
            name: r.name,
            bins: r.seq,
            callable: None,
            extent: None,
            called: None,
            het: None,
            het_sites: Vec::new(),
        }
    }
}

impl From<BinnedSeq> for RawSeq {
    fn from(s: BinnedSeq) -> Self {
        Self {
            name: s.name,
            bins: s.bins,
            callable: s.callable,
            extent: Some(s.extent),
            called: Some(s.called),
            het: Some(s.n_het),
            het_sites: s.het_sites,
        }
    }
}

impl From<ConsensusSeq> for RawSeq {
    fn from(s: ConsensusSeq) -> Self {
        Self {
            name: s.name,
            bins: s.bins,
            callable: s.callable,
            extent: Some(s.bases),
            called: Some(s.good_bases),
            het: Some(s.het_bases),
            het_sites: s.het_sites,
        }
    }
}

/// Read `path` into a [`Dataset`]; see the module docs.
///
/// The input is read once, so `path` may be a pipe or FIFO (`/dev/stdin`, a
/// process substitution): the lines read to recognise the format and the VCF
/// mode are kept in memory and read again. The one exception is
/// [`DepthThresholds::Auto`](crate::io::consensus::DepthThresholds::Auto) in
/// consensus mode, which reads the file twice and so fails up front unless
/// `path` is a regular file.
///
/// Fails on unreadable or malformed input. A file whose sequences are all
/// dropped, or that has no heterozygous bin, loads fine: QC wants to report
/// it, and [`Dataset::ensure_usable`] makes it an error for inference.
pub fn load(path: &Path, opts: &InputOptions) -> Result<Dataset> {
    if opts.batch_size == Some(0) {
        bail!("batch_size must be > 0");
    }
    let mut input = Input::open(path)?;
    let format = match opts.format {
        Some(f) => f,
        None => sniff_format(&mut input, path)?,
    };
    let (raw, source, filter) = match format {
        InputFormat::Psmcfa => {
            let (raw, summary) = load_psmcfa(input, path)?;
            (raw, SourceSummary::Psmcfa(summary), None)
        }
        InputFormat::Mhs => {
            let (raw, summary) = load_mhs(input, path, opts)?;
            (raw, SourceSummary::Mhs(summary), Some(opts.filter))
        }
        InputFormat::Vcf => {
            let (raw, summary) = load_vcf(input, path, opts)?;
            (raw, SourceSummary::Vcf(summary), Some(opts.filter))
        }
    };
    // Every reader but psmcfa's bins the input, counting the callable bases
    // of each bin when the bins are small enough.
    let binned = format != InputFormat::Psmcfa;

    let mut kept_bins = Vec::new();
    let mut kept_callable = Vec::new();
    let mut seqs = Vec::new();
    let mut dropped = Vec::new();
    let mut het_sites = (opts.collect_het_sites && binned).then(Vec::new);
    for s in raw {
        if let Some(reason) = s.drop_reason(filter.as_ref()) {
            dropped.push(DroppedSeq {
                name: s.name,
                n_bins: s.bins.len(),
                extent: s.extent,
                good: s.called,
                reason,
            });
            continue;
        }
        if let Some(sites) = het_sites.as_mut() {
            let seq = u32::try_from(seqs.len()).context("too many sequences")?;
            sites.extend(s.het_sites.iter().map(|h| HetSite {
                seq,
                pos: h.pos,
                nr_called: h.nr_called,
                alleles: h.alleles,
            }));
        }
        seqs.push(SeqInfo {
            name: s.name,
            n_bins: s.bins.len(),
            extent: s.extent,
            called: s.called,
            het_sites: s.het,
        });
        kept_bins.push(s.bins);
        kept_callable.push(s.callable);
    }
    // Some even when every sequence was dropped: it tells whether the
    // input records callable bases.
    let callable = binned
        .then(|| CallableBases::of_binned(opts.bin_size, kept_callable))
        .flatten();
    let obs = if kept_bins.is_empty() {
        Observations {
            rows: Vec::new(),
            row_starts: Vec::new(),
            callable,
        }
    } else {
        let kept = Sequences {
            bins: kept_bins,
            callable,
        };
        chunk_sequences(kept, opts.batch_size, "no observations")?
    };
    Ok(Dataset {
        obs,
        seqs,
        dropped,
        source,
        het_sites,
        filter,
    })
}

fn load_psmcfa(input: Input, path: &Path) -> Result<(Vec<RawSeq>, PsmcfaSummary)> {
    let (records, stats) = parse_psmcfa_with_stats(input.into_reader())
        .with_context(|| format!("failed to parse {path:?}"))?;
    if records.is_empty() {
        bail!(
            "no psmcfa records found in {path:?}: each sequence must start with a '>name' header \
             line"
        );
    }
    let mut summary = PsmcfaSummary {
        records: records.len(),
        fastq_records: stats.fastq_records,
        base_letters: stats.base_letters,
        ..PsmcfaSummary::default()
    };
    let raw = records
        .into_iter()
        .map(|r| {
            summary.empty_records += usize::from(r.seq.is_empty());
            summary.bins += r.seq.len() as u64;
            summary.called_bins += r.n_called() as u64;
            summary.het_bins += r.n_het() as u64;
            RawSeq::from(r)
        })
        .collect();
    Ok((raw, summary))
}

fn load_mhs(input: Input, path: &Path, opts: &InputOptions) -> Result<(Vec<RawSeq>, MhsSummary)> {
    let mhs_opts = MhsOptions {
        bin_size: opts.bin_size,
        batch_size: opts.batch_size,
        pair: opts.mhs_pair,
    };
    match read_mhs_seqs_from(input.into_reader(), &mhs_opts, opts.collect_het_sites) {
        Ok((seqs, summary)) => Ok((seqs.into_iter().map(RawSeq::from).collect(), summary)),
        Err(e) => {
            // Multihetsep is also what the format guess falls back to, so
            // an error that suggests another format says how the format
            // was chosen.
            let guessed = if opts.format.is_none() && e.maybe_other_format {
                " (the format was recognised from the first line, which does not start with \
                 '>' or '@' as in psmcfa and has fewer than the 8 tab-separated fields of a VCF \
                 record; give --input-format if the file is something else)"
            } else {
                ""
            };
            Err(e
                .error
                .context(format!("failed to read multihetsep {path:?}{guessed}")))
        }
    }
}

/// Why consensus mode refuses a VCF with several `samples`; `selected` is
/// the sample asked for, if any.
fn pooled_samples_message(samples: &[String], selected: Option<&str>) -> String {
    let selected = selected
        .map(|name| format!(" (it cannot read sample '{name}' alone)"))
        .unwrap_or_default();
    format!(
        "the VCF has {} samples ({}), but consensus mode, like vcfutils.pl vcf2fq, \
         reads only the INFO fields FQ, AF1, DP and MQ, which bcftools computes over all \
         samples together, so it would bin a consensus of the pooled samples, in which any \
         site that segregates among them looks heterozygous{selected}. Call the sample on its \
         own (bcftools mpileup with only its alignments, or with -s NAME, then bcftools call \
         -c), or read that sample's genotypes in mask mode (--vcf-mode mask --vcf-sample NAME) \
         with a callable mask (--vcf-mask)",
        samples.len(),
        list_names(samples)
    )
}

fn load_vcf(
    mut input: Input,
    path: &Path,
    opts: &InputOptions,
) -> Result<(Vec<RawSeq>, VcfInputSummary)> {
    if opts.bin_size == 0 {
        bail!("vcf bin_size must be > 0");
    }
    let choice = match opts.vcf_mode {
        VcfMode::Auto => detect_mode(&mut input, path)?,
        // Mask mode reads a gVCF as one, so the first records are inspected
        // for its blocks.
        VcfMode::Mask => VcfModeChoice {
            mode: VcfMode::Mask,
            forced: true,
            reason: "mask mode was requested".to_string(),
            ..inspect_records(&mut input, path)?
        },
        VcfMode::Consensus => VcfModeChoice {
            mode: VcfMode::Consensus,
            forced: true,
            reason: "consensus mode was requested".to_string(),
            ..VcfModeChoice::default()
        },
    };
    if choice.mode == VcfMode::Consensus {
        let samples = input.vcf_samples(path)?.unwrap_or_default();
        if let Some(name) = &opts.sample
            && !samples.iter().any(|s| s == name)
        {
            bail!("VCF sample '{name}' not found in header");
        }
        if samples.len() > 1 {
            bail!(
                "{}",
                pooled_samples_message(&samples, opts.sample.as_deref())
            );
        }
        let copts = ConsensusOptions {
            bin_size: opts.bin_size,
            filters: opts.consensus,
            masks: opts.masks.clone(),
            negative_masks: opts.negative_masks.clone(),
            collect_het_sites: opts.collect_het_sites,
        };
        // The lines mode detection kept are read again, then the rest.
        let read = read_consensus(path, &copts, || Ok(input.into_reader()))?;
        let raw = read.seqs.into_iter().map(RawSeq::from).collect();
        let summary = VcfInputSummary {
            mode: choice,
            details: VcfDetails::Consensus(read.summary),
            keep_filtered: false,
        };
        return Ok((raw, summary));
    }
    let vopts = VcfOptions {
        bin_size: opts.bin_size,
        batch_size: opts.batch_size,
        sample: opts.sample.clone(),
        masks: opts.masks.clone(),
        negative_masks: opts.negative_masks.clone(),
        keep_filtered: opts.keep_filtered,
        to_contig_end: opts.to_contig_end,
        gvcf: choice.gvcf_blocks > 0,
    };
    let (seqs, summary) =
        read_vcf_seqs_from(input.into_reader(), path, &vopts, opts.collect_het_sites)
            .with_context(|| format!("failed to read VCF {path:?} in mask mode"))?;
    Ok((
        seqs.into_iter().map(RawSeq::from).collect(),
        VcfInputSummary {
            mode: choice,
            details: VcfDetails::Mask(summary),
            keep_filtered: opts.keep_filtered,
        },
    ))
}
