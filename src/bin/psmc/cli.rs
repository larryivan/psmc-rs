//! Command-line arguments.

use clap::{Args, Parser, ValueEnum};
use std::path::PathBuf;

use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, VcfMode};

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
        match self {
            FormatArg::Auto => "auto",
            FormatArg::Psmcfa => "psmcfa",
            FormatArg::Mhs => "mhs",
            FormatArg::Vcf => "vcf",
        }
    }

    /// Every value, in the order the TUI cycles through them.
    pub const ALL: [FormatArg; 4] = [
        FormatArg::Auto,
        FormatArg::Psmcfa,
        FormatArg::Mhs,
        FormatArg::Vcf,
    ];
}

/// `--vcf-mode`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum VcfModeArg {
    /// Consensus mode for an all-sites VCF from `bcftools call -c`, mask
    /// mode otherwise.
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

    fn name(self) -> &'static str {
        self.mode().name()
    }
}

/// A positive integer.
fn positive_usize(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("must be at least 1".to_string()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

/// `--mhs-pair i,j`: two different 0-based haplotype indices.
pub fn parse_pair(s: &str) -> Result<(usize, usize), String> {
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
pub fn parse_depth(s: &str) -> Result<DepthThresholds, String> {
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
    /// Input format; auto recognises psmcfa, multihetsep and VCF from the
    /// content
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub input_format: FormatArg,

    /// Bases per bin: the bin size of VCF and multihetsep input (fq2psmcfa
    /// -s), and the time and Ne scaling for every format (give the -s of a
    /// psmcfa file)
    #[arg(long, value_name = "BP", default_value_t = 100, alias = "mhs-bin-size", value_parser = positive_usize)]
    pub bin_size: usize,

    /// Split each sequence into HMM rows of at most this many bins; the rows
    /// of a sequence stay one HMM chain, so results do not change
    #[arg(long, value_name = "BINS", value_parser = positive_usize)]
    pub batch_size: Option<usize>,

    /// Multihetsep: the two haplotypes to compare, as 0-based positions in
    /// the alleles column (0,1 is the first diploid sample)
    #[arg(long, value_name = "I,J", default_value = "0,1", value_parser = parse_pair)]
    pub mhs_pair: (usize, usize),

    /// VCF: the sample to read (required when the VCF has several)
    #[arg(long, value_name = "NAME")]
    pub vcf_sample: Option<String>,

    /// VCF: consensus mode reproduces `vcfutils.pl vcf2fq | fq2psmcfa` on an
    /// all-sites VCF from `bcftools call -c`; mask mode reads the sample's
    /// genotypes with callable masks; auto picks one from the first records
    #[arg(long, value_enum, default_value_t = VcfModeArg::Auto)]
    pub vcf_mode: VcfModeArg,

    /// VCF: a BED of callable sites (repeatable; a site must be in every
    /// one)
    #[arg(long = "vcf-mask", value_name = "BED")]
    pub vcf_mask: Vec<PathBuf>,

    /// VCF: a BED of sites to exclude (repeatable)
    #[arg(long = "vcf-negative-mask", value_name = "BED")]
    pub vcf_negative_mask: Vec<PathBuf>,

    /// VCF mask mode: use records whose FILTER is not PASS or '.' (by
    /// default they are treated as missing)
    #[arg(long)]
    pub vcf_keep_filtered: bool,

    /// VCF mask mode: end each sequence at its last VCF record instead of
    /// the ##contig length
    #[arg(long)]
    pub vcf_no_contig_end: bool,

    /// VCF consensus mode: INFO/DP bounds of a called base (vcf2fq -d, -D).
    /// auto uses a third and twice the mean depth (Li & Durbin), which reads
    /// the file twice; vcf2fq's own defaults are 3:100000
    #[arg(long, value_name = "auto|MIN:MAX", default_value = "auto", value_parser = parse_depth)]
    pub depth: DepthThresholds,

    /// VCF consensus mode: minimum INFO/MQ (vcf2fq -Q)
    #[arg(long, value_name = "MQ", default_value_t = 10)]
    pub min_mapq: u64,

    /// VCF consensus mode: bases masked on each side of an indel (vcf2fq -l)
    #[arg(long, value_name = "BP", default_value_t = 5)]
    pub indel_window: u64,

    /// VCF consensus mode: minimum base quality (fq2psmcfa -q)
    #[arg(long, value_name = "Q", default_value_t = 10)]
    pub min_qual: u64,

    /// VCF and multihetsep: drop a sequence with fewer called bases than
    /// this (fq2psmcfa -g; 0 disables)
    #[arg(long, value_name = "BASES", default_value_t = 10_000)]
    pub min_good_bases: u64,

    /// VCF and multihetsep: drop a sequence whose called bases are less
    /// than this share of its length (fq2psmcfa's fixed 0.2; 0 disables)
    #[arg(long, value_name = "SHARE", default_value_t = 0.2, value_parser = parse_fraction)]
    pub min_good_fraction: f64,
}

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

    /// The arguments that give these options; defaults are left out.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_string());
            args.push(value);
        };
        if self.input_format != FormatArg::Auto {
            push("--input-format", self.input_format.name().to_string());
        }
        if self.bin_size != 100 {
            push("--bin-size", self.bin_size.to_string());
        }
        if let Some(v) = self.batch_size {
            push("--batch-size", v.to_string());
        }
        if self.mhs_pair != (0, 1) {
            push(
                "--mhs-pair",
                format!("{},{}", self.mhs_pair.0, self.mhs_pair.1),
            );
        }
        if let Some(v) = &self.vcf_sample {
            push("--vcf-sample", v.clone());
        }
        if self.vcf_mode != VcfModeArg::Auto {
            push("--vcf-mode", self.vcf_mode.name().to_string());
        }
        for p in &self.vcf_mask {
            push("--vcf-mask", p.display().to_string());
        }
        for p in &self.vcf_negative_mask {
            push("--vcf-negative-mask", p.display().to_string());
        }
        if let DepthThresholds::Fixed { min, max } = self.depth {
            push("--depth", format!("{min}:{max}"));
        }
        if self.min_mapq != 10 {
            push("--min-mapq", self.min_mapq.to_string());
        }
        if self.indel_window != 5 {
            push("--indel-window", self.indel_window.to_string());
        }
        if self.min_qual != 10 {
            push("--min-qual", self.min_qual.to_string());
        }
        if self.min_good_bases != 10_000 {
            push("--min-good-bases", self.min_good_bases.to_string());
        }
        if self.min_good_fraction != 0.2 {
            push("--min-good-fraction", self.min_good_fraction.to_string());
        }
        if self.vcf_keep_filtered {
            args.push("--vcf-keep-filtered".to_string());
        }
        if self.vcf_no_contig_end {
            args.push("--vcf-no-contig-end".to_string());
        }
        args
    }
}

#[derive(Parser, Debug, Clone)]
#[command(name = "psmc-rs")]
#[command(about = "PSMC in Rust", long_about = None)]
pub struct Cli {
    #[arg(long, help = "Launch interactive TUI")]
    pub tui: bool,

    #[arg(required_unless_present = "tui")]
    pub input_file: Option<PathBuf>,

    #[arg(required_unless_present = "tui")]
    pub output_file: Option<PathBuf>,

    #[arg(required_unless_present = "tui")]
    pub n_iter: Option<usize>,

    #[command(flatten)]
    pub input: InputArgs,

    #[arg(
        long,
        help = "Number of worker threads for E-step (multi-sequence) and bootstrap"
    )]
    pub threads: Option<usize>,

    #[arg(long, default_value_t = 15.0)]
    pub t_max: f64,

    #[arg(long, default_value_t = 64)]
    pub n_steps: usize,

    #[arg(long)]
    pub theta0: Option<f64>,

    #[arg(long)]
    pub rho0: Option<f64>,

    #[arg(long)]
    pub pattern: Option<String>,

    #[arg(long, default_value_t = 2.5e-8)]
    pub mu: f64,

    #[arg(long)]
    pub mstep_iters: Option<usize>,

    #[arg(
        long,
        default_value_t = 1e-3,
        help = "M-step smoothness penalty on log-lambda differences"
    )]
    pub smooth_lambda: f64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of block-bootstrap replicates (0 disables bootstrap)"
    )]
    pub bootstrap: usize,

    #[arg(long, default_value_t = 50_000, help = "Bootstrap block size in bins")]
    pub bootstrap_block_size: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "Random seed for bootstrap resampling"
    )]
    pub bootstrap_seed: u64,

    #[arg(
        long,
        help = "EM iterations for each bootstrap replicate (default: same as N_ITER)"
    )]
    pub bootstrap_iters: Option<usize>,

    #[arg(
        long,
        help = "Directory for bootstrap replicate JSON files and summary outputs"
    )]
    pub bootstrap_dir: Option<PathBuf>,

    #[arg(long)]
    pub no_progress: bool,

    #[arg(
        long,
        help = "Write per-bin TMRCA posterior TSV and add TMRCA charts into HTML report"
    )]
    pub tmrca_out: Option<PathBuf>,

    #[arg(
        long,
        default_value_t = 25.0,
        help = "Generation time (years) used for TMRCA scaling"
    )]
    pub tmrca_gen_years: f64,
}

impl Cli {
    /// The arguments that reproduce this inference run on the command line
    /// (without `--tui`), as the TUI shows and runs it.
    pub fn to_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        if let Some(v) = &self.input_file {
            args.push(v.display().to_string());
        }
        if let Some(v) = &self.output_file {
            args.push(v.display().to_string());
        }
        if let Some(v) = self.n_iter {
            args.push(v.to_string());
        }
        args.extend(self.input.to_args());
        if let Some(v) = &self.pattern {
            args.push("--pattern".to_string());
            args.push(v.clone());
        }
        if let Some(v) = self.threads {
            args.push("--threads".to_string());
            args.push(v.to_string());
        }
        args.push("--mu".to_string());
        args.push(self.mu.to_string());
        args.push("--smooth-lambda".to_string());
        args.push(self.smooth_lambda.to_string());
        if self.bootstrap > 0 {
            args.push("--bootstrap".to_string());
            args.push(self.bootstrap.to_string());
        }
        args.push("--no-progress".to_string());
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(args: &[&str]) -> InputArgs {
        let mut argv = vec!["psmc-rs", "in", "out", "1"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv).unwrap().input
    }

    #[test]
    fn input_defaults_follow_the_classic_pipeline() {
        let o = input(&[]).to_options();
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
        let args = [
            "--input-format",
            "vcf",
            "--bin-size",
            "50",
            "--batch-size",
            "1000",
            "--mhs-pair",
            "2,3",
            "--vcf-sample",
            "s1",
            "--vcf-mode",
            "mask",
            "--vcf-mask",
            "a.bed",
            "--vcf-mask",
            "b.bed",
            "--vcf-negative-mask",
            "c.bed",
            "--vcf-keep-filtered",
            "--vcf-no-contig-end",
            "--depth",
            "3:100000",
            "--min-mapq",
            "20",
            "--indel-window",
            "7",
            "--min-qual",
            "0",
            "--min-good-bases",
            "0",
            "--min-good-fraction",
            "0",
        ];
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

        let back = a.to_args();
        let back: Vec<&str> = back.iter().map(String::as_str).collect();
        assert_eq!(input(&back), a);
        assert_eq!(input(&[]).to_args(), Vec::<String>::new());
    }

    #[test]
    fn the_old_bin_size_flag_still_works() {
        assert_eq!(input(&["--mhs-bin-size", "40"]).bin_size, 40);
    }

    #[test]
    fn invalid_input_values_are_rejected() {
        let bad = |args: &[&str]| {
            let mut argv = vec!["psmc-rs", "in", "out", "1"];
            argv.extend_from_slice(args);
            Cli::try_parse_from(argv).unwrap_err().to_string()
        };
        assert!(bad(&["--mhs-pair", "1,1"]).contains("must differ"));
        assert!(bad(&["--mhs-pair", "1"]).contains("I,J"));
        assert!(bad(&["--depth", "10:5"]).contains("above the maximum"));
        assert!(bad(&["--depth", "lots"]).contains("MIN:MAX"));
        assert!(bad(&["--bin-size", "0"]).contains("at least 1"));
        assert!(bad(&["--min-good-fraction", "1.5"]).contains("between 0 and 1"));
    }
}
