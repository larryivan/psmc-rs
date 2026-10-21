//! The options that say how to read the input, shared by inference,
//! `convert` and `qc`: their values, defaults and the inputs they apply to.

use clap::{Args, ValueEnum};
use std::path::PathBuf;

use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, VcfMode};

use super::values::{positive_usize, whole_number};
use super::{Spelling, defaults, on, one, spelt};

/// `--input-format`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum FormatArg {
    /// Recognise the format from the content.
    Auto,
    /// PSMC's binned FASTA (fq2psmcfa output).
    Psmcfa,
    /// MSMC's multihetsep.
    Mhs,
    /// VCF (plain or gzip/bgzip).
    Vcf,
}

impl FormatArg {
    /// The loader's format; `None` for auto.
    pub fn format(self) -> Option<InputFormat> {
        match self {
            FormatArg::Auto => None,
            FormatArg::Psmcfa => Some(InputFormat::Psmcfa),
            FormatArg::Mhs => Some(InputFormat::Mhs),
            FormatArg::Vcf => Some(InputFormat::Vcf),
        }
    }

    /// The name on the command line.
    pub fn name(self) -> &'static str {
        self.format().map_or("auto", InputFormat::name)
    }

    /// The format as users know it: psmcfa, multihetsep, VCF, or any of
    /// them for auto.
    pub fn label(self) -> &'static str {
        match self {
            FormatArg::Auto => "psmcfa, multihetsep or VCF",
            FormatArg::Psmcfa => "psmcfa",
            FormatArg::Mhs => "multihetsep",
            FormatArg::Vcf => "VCF",
        }
    }
}

/// `--vcf-mode`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum VcfModeArg {
    /// Consensus mode for an all-sites VCF from `bcftools call -c`, mask
    /// mode otherwise (judged from the first records).
    Auto,
    /// Read like `vcfutils.pl vcf2fq | fq2psmcfa`.
    Consensus,
    /// Read the sample's genotypes, with callable masks.
    Mask,
}

impl VcfModeArg {
    fn mode(self) -> VcfMode {
        match self {
            VcfModeArg::Auto => VcfMode::Auto,
            VcfModeArg::Consensus => VcfMode::Consensus,
            VcfModeArg::Mask => VcfMode::Mask,
        }
    }
}

/// fq2psmcfa's bin size.
pub const DEFAULT_BIN_SIZE: usize = 100;

/// `--mhs-pair i,j`: two different 0-based haplotype indices.
fn parse_pair(s: &str) -> Result<(usize, usize), String> {
    let (i, j) = s
        .split_once(',')
        .ok_or_else(|| format!("expected two indices as I,J (such as 0,1), got '{s}'"))?;
    let index = |v: &str| {
        v.trim()
            .parse::<usize>()
            .map_err(|_| format!("'{}' is not a haplotype index (0, 1, 2, ...)", v.trim()))
    };
    let (i, j) = (index(i)?, index(j)?);
    if i == j {
        return Err(format!(
            "the two haplotypes must differ, got {i},{j} (0,1 is the first diploid sample)"
        ));
    }
    Ok((i, j))
}

/// `--depth auto|MIN:MAX`.
fn parse_depth(s: &str) -> Result<DepthThresholds, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(DepthThresholds::Auto);
    }
    let bad = || format!("expected 'auto' or MIN:MAX (such as 3:100000), got '{s}'");
    let (min, max) = s.split_once(':').ok_or_else(bad)?;
    let min: u64 = min.trim().parse().map_err(|_| bad())?;
    let max: u64 = max.trim().parse().map_err(|_| bad())?;
    if min > max {
        return Err(format!(
            "the minimum depth {min} is above the maximum {max}"
        ));
    }
    Ok(DepthThresholds::Fixed { min, max })
}

/// A share between 0 and 1.
fn parse_fraction(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("'{s}' is not a number"))?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!("must be between 0 and 1, got {v}"));
    }
    Ok(v)
}

/// How to read the input; shared by inference, `convert` and `qc`, so that
/// all three read a file identically.
#[derive(Args, Debug, Clone, PartialEq)]
#[command(next_help_heading = "Input")]
pub struct InputArgs {
    /// Input format; auto recognises it from the content
    #[arg(long, value_enum, value_name = "FORMAT", default_value_t = FormatArg::Auto)]
    pub input_format: FormatArg,

    /// Bases per bin (fq2psmcfa -s): bins VCF and multihetsep input; for psmcfa, give the -s it was made with.
    ///
    /// The time and Ne scaling uses it for every format, so a wrong value
    /// rescales the results.
    #[arg(long, value_name = "BP", default_value_t = DEFAULT_BIN_SIZE, alias = "mhs-bin-size", value_parser = positive_usize)]
    pub bin_size: usize,

    /// Split sequences into HMM rows of at most BINS bins; results do not change.
    ///
    /// The rows of a sequence stay one HMM chain.
    #[arg(long, value_name = "BINS", value_parser = positive_usize)]
    pub batch_size: Option<usize>,

    /// Multihetsep: the two haplotypes, 0-based positions in the alleles column.
    ///
    /// 0,1 is the first diploid sample.
    #[arg(long, value_name = "I,J", default_value = "0,1", value_parser = parse_pair)]
    pub mhs_pair: (usize, usize),

    /// VCF mask mode: the sample to read; consensus mode needs a one-sample VCF.
    ///
    /// Mask mode reads this sample's genotypes; the name is required when
    /// the VCF has several samples. Consensus mode reads INFO fields that
    /// bcftools computes over all samples together, so give it a one-sample
    /// VCF; the name is then only checked against that sample.
    #[arg(long, value_name = "NAME")]
    pub vcf_sample: Option<String>,

    /// VCF: consensus (vcf2fq | fq2psmcfa) or mask mode
    #[arg(long, value_enum, value_name = "MODE", default_value_t = VcfModeArg::Auto)]
    pub vcf_mode: VcfModeArg,

    /// VCF: a BED of callable sites (repeatable; a site must be in every one)
    #[arg(long = "vcf-mask", value_name = "BED")]
    pub vcf_mask: Vec<PathBuf>,

    /// VCF: a BED of sites to exclude (repeatable)
    #[arg(long = "vcf-negative-mask", value_name = "BED")]
    pub vcf_negative_mask: Vec<PathBuf>,

    /// VCF mask mode: use records whose FILTER is not PASS or '.'.
    ///
    /// By default they are treated as missing.
    #[arg(long)]
    pub vcf_keep_filtered: bool,

    /// VCF mask mode: end sequences at their last record, not the ##contig length
    #[arg(long)]
    pub vcf_no_contig_end: bool,

    /// VCF consensus mode: INFO/DP bounds (vcf2fq -d, -D); auto = a third and twice the mean.
    ///
    /// The bounds of the depth of a called base. auto, the psmc README's
    /// choice, reads the file twice to find the mean depth; vcf2fq's own
    /// defaults are 3:100000.
    #[arg(long, value_name = "auto|MIN:MAX", default_value = "auto", value_parser = parse_depth)]
    pub depth: DepthThresholds,

    /// VCF consensus mode: minimum INFO/MQ (vcf2fq -Q)
    #[arg(long, value_name = "MQ", default_value_t = 10, value_parser = whole_number::<u64>)]
    pub min_mapq: u64,

    /// VCF consensus mode: bases masked on each side of an indel (vcf2fq -l)
    #[arg(long, value_name = "BP", default_value_t = 5, value_parser = whole_number::<u64>)]
    pub indel_window: u64,

    /// VCF consensus mode: minimum base quality (fq2psmcfa -q)
    #[arg(long, value_name = "Q", default_value_t = 10, value_parser = whole_number::<u64>)]
    pub min_qual: u64,

    /// VCF and multihetsep: drop sequences with fewer called bases (fq2psmcfa -g; 0 disables)
    #[arg(long, value_name = "BASES", default_value_t = 10_000, value_parser = whole_number::<u64>)]
    pub min_good_bases: u64,

    /// VCF and multihetsep: drop sequences whose called share is lower (fq2psmcfa's fixed 0.2; 0 disables)
    #[arg(long, value_name = "SHARE", default_value_t = 0.2, value_parser = parse_fraction)]
    pub min_good_fraction: f64,
}

/// Every input flag, with the inputs it applies to and its spelling.
const FLAGS: [(&str, AppliesTo, Spelling<InputArgs>); 16] = {
    use AppliesTo::*;
    fn paths(paths: &[PathBuf]) -> Option<Vec<String>> {
        let shown = paths.iter().map(|p| p.display().to_string());
        (!paths.is_empty()).then(|| shown.collect())
    }
    fn depth(depth: DepthThresholds) -> Option<Vec<String>> {
        one(match depth {
            DepthThresholds::Fixed { min, max } => format!("{min}:{max}"),
            DepthThresholds::Auto => "auto".to_string(),
        })
    }
    [
        ("--input-format", Every, |a| one(a.input_format.name())),
        ("--bin-size", Every, |a| one(a.bin_size)),
        ("--batch-size", Every, |a| a.batch_size.and_then(one)),
        ("--vcf-sample", Vcf, |a| a.vcf_sample.as_ref().and_then(one)),
        ("--vcf-mode", Vcf, |a| one(a.vcf_mode.mode().name())),
        ("--vcf-mask", Vcf, |a| paths(&a.vcf_mask)),
        ("--vcf-negative-mask", Vcf, |a| paths(&a.vcf_negative_mask)),
        ("--vcf-keep-filtered", VcfMask, |a| on(a.vcf_keep_filtered)),
        ("--vcf-no-contig-end", VcfMask, |a| on(a.vcf_no_contig_end)),
        ("--depth", VcfConsensus, |a| depth(a.depth)),
        ("--min-mapq", VcfConsensus, |a| one(a.min_mapq)),
        ("--indel-window", VcfConsensus, |a| one(a.indel_window)),
        ("--min-qual", VcfConsensus, |a| one(a.min_qual)),
        ("--mhs-pair", Mhs, |a| {
            one(format!("{},{}", a.mhs_pair.0, a.mhs_pair.1))
        }),
        ("--min-good-bases", Binned, |a| one(a.min_good_bases)),
        ("--min-good-fraction", Binned, |a| one(a.min_good_fraction)),
    ]
};

impl InputArgs {
    /// The loader's options.
    pub fn to_options(&self) -> InputOptions {
        InputOptions {
            format: self.input_format.format(),
            bin_size: self.bin_size,
            batch_size: self.batch_size,
            mhs_pair: self.mhs_pair,
            vcf_mode: self.vcf_mode.mode(),
            sample: self.vcf_sample.clone(),
            masks: self.vcf_mask.clone(),
            negative_masks: self.vcf_negative_mask.clone(),
            keep_filtered: self.vcf_keep_filtered,
            to_contig_end: !self.vcf_no_contig_end,
            consensus: ConsensusFilters {
                depth: self.depth,
                min_mq: self.min_mapq,
                indel_window: self.indel_window,
                min_qual: self.min_qual,
            },
            filter: SeqFilter {
                min_good_bases: self.min_good_bases,
                min_good_fraction: self.min_good_fraction,
            },
            collect_het_sites: false,
        }
    }

    /// The flags given a value other than clap's default, with the inputs
    /// they apply to and their values as spelt ([`Spelling`]).
    fn given(&self) -> Vec<(&'static str, AppliesTo, Vec<String>)> {
        let defaults: InputArgs = defaults();
        (FLAGS.iter())
            .filter(|(_, _, spell)| spell(self) != spell(&defaults))
            .filter_map(|&(flag, applies, spell)| Some((flag, applies, spell(self)?)))
            .collect()
    }

    /// The input flags that give these settings on the command line.
    pub(super) fn to_args(&self) -> Vec<String> {
        spelt(
            self.given()
                .into_iter()
                .map(|(flag, _, values)| (flag, values)),
        )
    }

    /// The flags given a value other than their default that do not apply
    /// to input of `format` (read in `vcf_mode`, for VCF), as they are
    /// spelt on the command line. `--vcf-sample` is not among them in
    /// consensus mode, which checks it ([`InputArgs::sample_only_checked`]).
    pub fn ignored_flags(
        &self,
        format: InputFormat,
        vcf_mode: Option<VcfMode>,
    ) -> Vec<&'static str> {
        self.given()
            .into_iter()
            .filter(|&(_, applies, _)| !applies.to(format, vcf_mode))
            .map(|(flag, ..)| flag)
            .collect()
    }

    /// Whether `--vcf-sample` is given for input of `format` (read in
    /// `vcf_mode`, for VCF) that only checks it: consensus mode reads the
    /// INFO fields, which bcftools computes over all samples together, so it
    /// needs a one-sample VCF; the name must be that sample's, but it
    /// selects nothing.
    pub fn sample_only_checked(&self, format: InputFormat, vcf_mode: Option<VcfMode>) -> bool {
        AppliesTo::VcfConsensus.to(format, vcf_mode) && self.vcf_sample.is_some()
    }
}

/// The inputs an input flag applies to.
#[derive(Clone, Copy)]
enum AppliesTo {
    Every,
    Vcf,
    VcfMask,
    VcfConsensus,
    Mhs,
    /// The inputs psmc-rs bins itself, all but psmcfa, which go through
    /// fq2psmcfa's sequence filter.
    Binned,
}

impl AppliesTo {
    /// Whether the flag applies to input of `format`, read in `vcf_mode`
    /// for VCF.
    fn to(self, format: InputFormat, vcf_mode: Option<VcfMode>) -> bool {
        let vcf = format == InputFormat::Vcf;
        match self {
            AppliesTo::Every => true,
            AppliesTo::Vcf => vcf,
            AppliesTo::VcfMask => vcf && vcf_mode == Some(VcfMode::Mask),
            AppliesTo::VcfConsensus => vcf && vcf_mode == Some(VcfMode::Consensus),
            AppliesTo::Mhs => format == InputFormat::Mhs,
            AppliesTo::Binned => format != InputFormat::Psmcfa,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::tests::{parse, parse_err};

    fn input(args: &str) -> InputArgs {
        parse(args).unwrap().input
    }

    /// Every input flag with a value other than its default, except
    /// `--input-format`, `--bin-size` and `--batch-size`, in the order that
    /// [`InputArgs::ignored_flags`] lists them.
    const OTHER_FORMAT_FLAGS: &str = "--vcf-sample s1 --vcf-mode mask --vcf-mask a.bed \
        --vcf-negative-mask c.bed --vcf-keep-filtered --vcf-no-contig-end --depth 3:100000 \
        --min-mapq 20 --indel-window 7 --min-qual 0 --mhs-pair 2,3 --min-good-bases 0 \
        --min-good-fraction 0";

    #[test]
    fn input_defaults_follow_the_classic_pipeline() {
        let o = input("").to_options();
        assert_eq!(o.format, None);
        assert_eq!(o.bin_size, 100);
        assert_eq!(o.mhs_pair, (0, 1));
        assert_eq!(o.vcf_mode, VcfMode::Auto);
        assert!(o.to_contig_end && !o.keep_filtered);
        assert_eq!(
            o.consensus,
            ConsensusFilters {
                depth: DepthThresholds::Auto,
                min_mq: 10,
                indel_window: 5,
                min_qual: 10,
            }
        );
        assert_eq!(o.filter, SeqFilter::default());
        assert!(!o.collect_het_sites);
    }

    #[test]
    fn every_input_flag_reaches_the_loader_and_back() {
        let args = format!(
            "--input-format vcf --bin-size 50 --batch-size 1000 {OTHER_FORMAT_FLAGS} --vcf-mask b.bed"
        );
        let a = input(&args);
        let o = a.to_options();
        assert_eq!(o.format, Some(InputFormat::Vcf));
        assert_eq!((o.bin_size, o.batch_size), (50, Some(1000)));
        assert_eq!(o.mhs_pair, (2, 3));
        assert_eq!(o.sample.as_deref(), Some("s1"));
        assert_eq!(o.vcf_mode, VcfMode::Mask);
        assert_eq!(o.masks, [PathBuf::from("a.bed"), PathBuf::from("b.bed")]);
        assert_eq!(o.negative_masks, [PathBuf::from("c.bed")]);
        assert!(o.keep_filtered && !o.to_contig_end);
        assert_eq!(
            o.consensus,
            ConsensusFilters {
                depth: DepthThresholds::Fixed {
                    min: 3,
                    max: 100_000
                },
                min_mq: 20,
                indel_window: 7,
                min_qual: 0,
            }
        );
        assert_eq!(o.filter, SeqFilter::DISABLED);

        assert_eq!(input(&a.to_args().join(" ")), a);
        assert_eq!(input("").to_args(), Vec::<String>::new());
    }

    #[test]
    fn options_of_another_format_are_found() {
        let a = input(OTHER_FORMAT_FLAGS);
        let flags: Vec<&str> = OTHER_FORMAT_FLAGS
            .split_whitespace()
            .filter(|f| f.starts_with("--"))
            .collect();
        assert_eq!(a.ignored_flags(InputFormat::Psmcfa, None), flags);
        assert_eq!(
            a.ignored_flags(InputFormat::Mhs, None),
            &flags[..10],
            "multihetsep takes the pair and the sequence filter"
        );
        assert_eq!(
            a.ignored_flags(InputFormat::Vcf, Some(VcfMode::Mask)),
            [
                "--depth",
                "--min-mapq",
                "--indel-window",
                "--min-qual",
                "--mhs-pair"
            ]
        );
        assert_eq!(
            a.ignored_flags(InputFormat::Vcf, Some(VcfMode::Consensus)),
            ["--vcf-keep-filtered", "--vcf-no-contig-end", "--mhs-pair"]
        );
        // Consensus mode only checks the sample; mask mode selects it.
        assert!(a.sample_only_checked(InputFormat::Vcf, Some(VcfMode::Consensus)));
        for (format, mode) in [
            (InputFormat::Vcf, Some(VcfMode::Mask)),
            (InputFormat::Psmcfa, None),
            (InputFormat::Mhs, None),
        ] {
            assert!(!a.sample_only_checked(format, mode));
        }
        assert!(!input("").sample_only_checked(InputFormat::Vcf, Some(VcfMode::Consensus)));
        // Defaults are never reported, even when spelt out.
        let defaults = input("--depth auto --mhs-pair 0,1 --min-qual 10");
        assert!(defaults.ignored_flags(InputFormat::Psmcfa, None).is_empty());
    }

    #[test]
    fn invalid_input_values_are_rejected() {
        assert!(parse_err("--mhs-pair 1,1").contains("must differ"));
        assert!(parse_err("--mhs-pair 1").contains("I,J"));
        assert!(parse_err("--depth 10:5").contains("above the maximum"));
        assert!(parse_err("--depth lots").contains("MIN:MAX"));
        assert!(parse_err("--bin-size 0").contains("at least 1"));
        assert!(parse_err("--min-good-fraction 1.5").contains("between 0 and 1"));
    }

    #[test]
    fn the_old_bin_size_flag_still_works() {
        assert_eq!(input("--mhs-bin-size 40").bin_size, 40);
    }
}
