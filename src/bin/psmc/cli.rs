//! Command-line arguments.

use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

use psmc_rs::bootstrap::DEFAULT_BOOTSTRAP_TRUNK;
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, VcfMode};
use psmc_rs::model::{T_INF, n_steps_for_pattern};

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

/// A positive, finite number.
fn positive_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v > 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be a positive number, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// A finite number of at least 0.
fn non_negative_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v >= 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be 0 or more, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// `--t-max`: inside C psmc's time grid, which ends at `T_INF`.
fn parse_t_max(s: &str) -> Result<f64, String> {
    let v = positive_f64(s)?;
    if v >= T_INF {
        return Err(format!(
            "must be below {T_INF}, where C psmc's time grid ends, got {v}"
        ));
    }
    Ok(v)
}

/// C psmc's default pattern (`-p`).
pub const DEFAULT_PATTERN: &str = "4+5*3+4";

/// The model layout `(n_steps, pattern)` that `--pattern` and `--n-steps`
/// give: C psmc's default pattern without either, `n_steps` from the
/// pattern without `--n-steps`, and one free lambda per interval with
/// `--n-steps` alone. Given both, `--n-steps` must match the pattern;
/// its number of intervals (`n_steps + 1`, the form psmc-rs 0.1 used) is
/// accepted too.
pub fn model_layout(
    pattern: Option<&str>,
    n_steps: Option<usize>,
) -> Result<(usize, Option<String>)> {
    let Some(pattern) = pattern else {
        return match n_steps {
            Some(n) => Ok((n, None)),
            None => Ok((
                n_steps_for_pattern(DEFAULT_PATTERN)?,
                Some(DEFAULT_PATTERN.to_string()),
            )),
        };
    };
    let derived =
        n_steps_for_pattern(pattern).with_context(|| format!("invalid --pattern '{pattern}'"))?;
    match n_steps {
        None => Ok((derived, Some(pattern.to_string()))),
        Some(n) if n == derived || n == derived + 1 => Ok((derived, Some(pattern.to_string()))),
        Some(n) => bail!(
            "--n-steps {n} does not match --pattern '{pattern}', which has {} intervals \
             (n_steps {derived}); leave out --n-steps to take it from the pattern",
            derived + 1
        ),
    }
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

const ABOUT: &str = "PSMC: population size history from one diploid genome";

const LONG_ABOUT: &str = "\
PSMC (Li & Durbin 2011): population size history from one diploid genome.

psmc-rs INPUT OUTPUT N_ITER fits the model to INPUT (psmcfa, multihetsep or
VCF) with N_ITER EM iterations and writes the parameters to OUTPUT (JSON)
with an HTML report next to it. `psmc-rs convert` writes psmcfa and
multihetsep files, and --tui opens an interactive interface.";

const AFTER_HELP: &str = "\
Examples:
  psmc-rs sample.psmcfa fit.json 25 --psmc-out fit.psmc
  psmc-rs calls.vcf.gz fit.json 25 --bootstrap 100 --threads 8
  psmc-rs convert calls.vcf.gz --psmcfa sample.psmcfa.gz";

/// The command line: inference unless a subcommand is given.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "psmc-rs",
    version,
    about = ABOUT,
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    #[command(flatten)]
    pub infer: InferArgs,
}

/// The subcommands.
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Write an input (VCF, multihetsep or psmcfa) as psmcfa or
    /// multihetsep, read exactly as inference reads it
    Convert(ConvertArgs),
}

/// `psmc-rs convert`.
#[derive(Args, Debug, Clone)]
#[command(group(
    ArgGroup::new("outputs")
        .required(true)
        .multiple(true)
        .args(["psmcfa", "multihetsep", "multihetsep_per_chrom"])
))]
pub struct ConvertArgs {
    /// Input file: psmcfa, multihetsep or VCF (gzip/bgzip allowed)
    #[arg(value_name = "INPUT")]
    pub input_file: PathBuf,

    #[command(flatten)]
    pub input: InputArgs,

    /// Write psmcfa, as fq2psmcfa lays it out (gzip when FILE ends in .gz)
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub psmcfa: Option<PathBuf>,

    /// Write the heterozygous sites of every sequence as multihetsep, as
    /// generate_multihetsep.py writes them (VCF and multihetsep input)
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub multihetsep: Option<PathBuf>,

    /// Write one multihetsep file per sequence, PREFIX.<name>.multihetsep
    /// (with .gz added, gzip-compressed, when PREFIX ends in .gz), as MSMC2
    /// takes them
    #[arg(long, value_name = "PREFIX", help_heading = "Output")]
    pub multihetsep_per_chrom: Option<PathBuf>,

    /// Write the input QC report as JSON ('-' for stdout)
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub qc_json: Option<PathBuf>,

    /// Print only results and warnings
    #[arg(long, short = 'q')]
    pub quiet: bool,
}

/// Inference: `psmc-rs INPUT OUTPUT N_ITER [options]`.
#[derive(Args, Debug, Clone)]
pub struct InferArgs {
    /// Open the interactive interface (the other arguments fill its form)
    #[arg(long)]
    pub tui: bool,

    /// Input file: psmcfa, multihetsep or VCF (gzip/bgzip allowed); a
    /// directory takes its first input file
    #[arg(value_name = "INPUT", required_unless_present = "tui")]
    pub input_file: Option<PathBuf>,

    /// Output JSON of the fitted parameters; the HTML report is written
    /// next to it
    #[arg(value_name = "OUTPUT", required_unless_present = "tui")]
    pub output_file: Option<PathBuf>,

    /// EM iterations (C psmc -N; 0 writes the starting model)
    #[arg(value_name = "N_ITER", required_unless_present = "tui")]
    pub n_iter: Option<usize>,

    #[command(flatten)]
    pub input: InputArgs,

    /// Worker threads for the E-step (sequences in parallel), the
    /// bootstrap and TMRCA decoding (default: one per core); results do not
    /// depend on it
    #[arg(long, value_name = "N", value_parser = positive_usize)]
    pub threads: Option<usize>,

    /// Free-parameter pattern of the time intervals (C psmc -p); default
    /// C psmc's 4+5*3+4, or one lambda per interval with --n-steps alone
    #[arg(long, help_heading = "Model")]
    pub pattern: Option<String>,

    /// Number of time intervals minus one (C psmc's n); taken from
    /// --pattern, and checked against it when both are given
    #[arg(long, help_heading = "Model")]
    pub n_steps: Option<usize>,

    /// Initial t_max, in units of 2N0 generations (C psmc -t)
    #[arg(long, default_value_t = 15.0, value_parser = parse_t_max, help_heading = "Model")]
    pub t_max: f64,

    /// Initial theta0/rho0 ratio (C psmc -r); rho0 = theta0 / this
    #[arg(long, value_name = "RATIO", default_value_t = 4.0, value_parser = positive_f64, help_heading = "Model")]
    pub tr_ratio: f64,

    /// Initial theta0 per bin (default: -ln(1 - K/L) from the data, as C
    /// psmc)
    #[arg(long, value_parser = positive_f64, help_heading = "Model")]
    pub theta0: Option<f64>,

    /// Initial rho0 per bin (default: theta0 / --tr-ratio)
    #[arg(long, value_parser = positive_f64, help_heading = "Model")]
    pub rho0: Option<f64>,

    /// Start from the parameters saved in this JSON output of an earlier
    /// run (C psmc -i), instead of from theta0, rho0, t_max and flat
    /// lambdas; its pattern and n_steps are used unless given
    #[arg(long, value_name = "FILE", conflicts_with_all = ["theta0", "rho0", "t_max", "tr_ratio"], help_heading = "Model")]
    pub init_params: Option<PathBuf>,

    /// Mutation rate per base pair per generation, for scaling
    #[arg(long, default_value_t = 2.5e-8, value_parser = positive_f64, help_heading = "Model")]
    pub mu: f64,

    /// Maximum L-BFGS iterations per M-step
    #[arg(long, value_name = "N", default_value_t = 100, value_parser = positive_usize, help_heading = "Model")]
    pub mstep_iters: usize,

    /// Weight of the M-step penalty on differences of log lambda between
    /// neighbouring intervals (0 = C psmc's objective)
    #[arg(long, value_name = "WEIGHT", default_value_t = 0.0, value_parser = non_negative_f64, help_heading = "Model")]
    pub smooth_lambda: f64,

    /// Bootstrap replicates to fit (0: none). As C psmc's bootstrap, each
    /// draws trunks of the input with replacement and is fitted from a fresh
    /// start
    #[arg(
        long,
        value_name = "N",
        default_value_t = 0,
        help_heading = "Bootstrap"
    )]
    pub bootstrap: usize,

    /// Bootstrap trunk size in bins (C splitfa; 500000 = 50 Mb at 100 bp
    /// bins); it must leave the input at least two trunks
    #[arg(long, value_name = "BINS", default_value_t = DEFAULT_BOOTSTRAP_TRUNK, value_parser = positive_usize, alias = "bootstrap-trunk-size", help_heading = "Bootstrap")]
    pub bootstrap_block_size: usize,

    /// Seed of the bootstrap resampling
    #[arg(
        long,
        value_name = "SEED",
        default_value_t = 42,
        help_heading = "Bootstrap"
    )]
    pub bootstrap_seed: u64,

    /// EM iterations of each replicate (default: N_ITER)
    #[arg(long, value_name = "N", help_heading = "Bootstrap")]
    pub bootstrap_iters: Option<usize>,

    /// Directory of the replicate files (replicate_NNN.json and .psmc) and
    /// the summary (default: OUTPUT with .bootstrap instead of .json)
    #[arg(long, value_name = "DIR", help_heading = "Bootstrap")]
    pub bootstrap_dir: Option<PathBuf>,

    /// Do not draw progress bars
    #[arg(long)]
    pub no_progress: bool,

    /// Print only results and warnings: no input description, model
    /// summary or QC report (QC warnings are still listed)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    /// Write the input QC report as JSON ('-' for stdout)
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub qc_json: Option<PathBuf>,

    /// Also write C psmc's .psmc output (rounds 0 to N_ITER), for
    /// psmc_plot.pl and other tools made for C psmc
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub psmc_out: Option<PathBuf>,

    #[arg(
        long,
        help = "Write per-bin TMRCA posterior TSV and add TMRCA charts into HTML report"
    )]
    pub tmrca_out: Option<PathBuf>,

    /// Years per generation, for the time axis of the Ne curve (report),
    /// the bootstrap summary and the TMRCA output
    #[arg(long, value_name = "YEARS", default_value_t = 25.0, value_parser = positive_f64, alias = "tmrca-gen-years")]
    pub gen_years: f64,
}

impl InferArgs {
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
        let mut push = |flag: &str, value: String| {
            args.push(flag.to_string());
            args.push(value);
        };
        if let Some(v) = &self.pattern {
            push("--pattern", v.clone());
        }
        if let Some(v) = self.n_steps {
            push("--n-steps", v.to_string());
        }
        if self.t_max != 15.0 {
            push("--t-max", self.t_max.to_string());
        }
        if self.tr_ratio != 4.0 {
            push("--tr-ratio", self.tr_ratio.to_string());
        }
        if let Some(v) = self.theta0 {
            push("--theta0", v.to_string());
        }
        if let Some(v) = self.rho0 {
            push("--rho0", v.to_string());
        }
        if let Some(v) = &self.init_params {
            push("--init-params", v.display().to_string());
        }
        if self.mu != 2.5e-8 {
            push("--mu", self.mu.to_string());
        }
        if self.mstep_iters != 100 {
            push("--mstep-iters", self.mstep_iters.to_string());
        }
        if self.smooth_lambda != 0.0 {
            push("--smooth-lambda", self.smooth_lambda.to_string());
        }
        if let Some(v) = self.threads {
            push("--threads", v.to_string());
        }
        if self.gen_years != 25.0 {
            push("--gen-years", self.gen_years.to_string());
        }
        if self.bootstrap > 0 {
            args.push("--bootstrap".to_string());
            args.push(self.bootstrap.to_string());
            if self.bootstrap_block_size != DEFAULT_BOOTSTRAP_TRUNK {
                args.push("--bootstrap-block-size".to_string());
                args.push(self.bootstrap_block_size.to_string());
            }
            if self.bootstrap_seed != 42 {
                args.push("--bootstrap-seed".to_string());
                args.push(self.bootstrap_seed.to_string());
            }
        }
        if let Some(v) = self.bootstrap_iters {
            args.push("--bootstrap-iters".to_string());
            args.push(v.to_string());
        }
        if let Some(v) = &self.bootstrap_dir {
            args.push("--bootstrap-dir".to_string());
            args.push(v.display().to_string());
        }
        if let Some(v) = &self.psmc_out {
            args.push("--psmc-out".to_string());
            args.push(v.display().to_string());
        }
        if let Some(v) = &self.qc_json {
            args.push("--qc-json".to_string());
            args.push(v.display().to_string());
        }
        args.push("--no-progress".to_string());
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inference arguments of a command line.
    fn parse<I, T>(argv: I) -> Result<InferArgs, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        Cli::try_parse_from(argv).map(|c| c.infer)
    }

    fn input(args: &[&str]) -> InputArgs {
        let mut argv = vec!["psmc-rs", "in", "out", "1"];
        argv.extend_from_slice(args);
        parse(argv).unwrap().input
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
    fn the_layout_follows_c_psmc() {
        assert_eq!(
            model_layout(None, None).unwrap(),
            (22, Some("4+5*3+4".to_string()))
        );
        assert_eq!(model_layout(None, Some(63)).unwrap(), (63, None));
        let human = Some("4+25*2+4+6".to_string());
        assert_eq!(
            model_layout(human.as_deref(), None).unwrap(),
            (63, human.clone())
        );
        assert_eq!(
            model_layout(human.as_deref(), Some(63)).unwrap(),
            (63, human.clone())
        );
        // psmc-rs 0.1 took the number of intervals.
        assert_eq!(
            model_layout(human.as_deref(), Some(64)).unwrap(),
            (63, human.clone())
        );
        let err = model_layout(human.as_deref(), Some(30))
            .unwrap_err()
            .to_string();
        assert!(err.contains("has 64 intervals (n_steps 63)"), "{err}");
        let err = format!("{:#}", model_layout(Some("4+x"), None).unwrap_err());
        assert!(err.contains("invalid --pattern '4+x'"), "{err}");
    }

    #[test]
    fn saved_parameters_replace_the_starting_values() {
        let with_saved = |args: &[&str]| {
            let mut argv = vec!["psmc-rs", "in", "out", "1", "--init-params", "fit.json"];
            argv.extend_from_slice(args);
            parse(argv)
        };
        assert!(with_saved(&[]).is_ok());
        assert!(with_saved(&["--pattern", "4+5*3+4", "--mu", "1e-8"]).is_ok());
        for flag in ["--theta0", "--rho0", "--t-max", "--tr-ratio"] {
            let err = with_saved(&[flag, "5"]).unwrap_err().to_string();
            assert!(err.contains("cannot be used with"), "{flag}: {err}");
        }
    }

    #[test]
    fn model_values_are_checked() {
        let bad = |args: &[&str]| {
            let mut argv = vec!["psmc-rs", "in", "out", "1"];
            argv.extend_from_slice(args);
            parse(argv).unwrap_err().to_string()
        };
        assert!(bad(&["--t-max", "1000"]).contains("below 1000"));
        assert!(bad(&["--t-max", "0"]).contains("positive"));
        assert!(bad(&["--tr-ratio=-4"]).contains("positive"));
        assert!(bad(&["--smooth-lambda=-1"]).contains("0 or more"));
        assert!(bad(&["--mstep-iters", "0"]).contains("at least 1"));
    }

    #[test]
    fn the_old_bin_size_flag_still_works() {
        assert_eq!(input(&["--mhs-bin-size", "40"]).bin_size, 40);
    }

    #[test]
    fn the_old_generation_time_flag_still_works() {
        let cli = parse(["psmc-rs", "in", "out", "1", "--tmrca-gen-years", "29"]);
        assert_eq!(cli.unwrap().gen_years, 29.0);
    }

    #[test]
    fn invalid_input_values_are_rejected() {
        let bad = |args: &[&str]| {
            let mut argv = vec!["psmc-rs", "in", "out", "1"];
            argv.extend_from_slice(args);
            parse(argv).unwrap_err().to_string()
        };
        assert!(bad(&["--mhs-pair", "1,1"]).contains("must differ"));
        assert!(bad(&["--mhs-pair", "1"]).contains("I,J"));
        assert!(bad(&["--depth", "10:5"]).contains("above the maximum"));
        assert!(bad(&["--depth", "lots"]).contains("MIN:MAX"));
        assert!(bad(&["--bin-size", "0"]).contains("at least 1"));
        assert!(bad(&["--min-good-fraction", "1.5"]).contains("between 0 and 1"));
    }

    #[test]
    fn subcommands_take_their_own_arguments() {
        let cli = Cli::try_parse_from([
            "psmc-rs",
            "convert",
            "in.vcf.gz",
            "--psmcfa",
            "out.psmcfa",
            "--bin-size",
            "50",
        ])
        .unwrap();
        let Some(Command::Convert(c)) = cli.command else {
            panic!("not convert")
        };
        assert_eq!(c.input_file, PathBuf::from("in.vcf.gz"));
        assert_eq!(c.input.bin_size, 50);

        // Inference still takes its three positional arguments.
        let cli = Cli::try_parse_from(["psmc-rs", "in", "out", "5"]).unwrap();
        assert!(cli.command.is_none());
        assert_eq!(cli.infer.n_iter, Some(5));
        assert!(Cli::try_parse_from(["psmc-rs", "--tui"]).is_ok());
        assert!(Cli::try_parse_from(["psmc-rs", "in", "out"]).is_err());
    }

    #[test]
    fn invalid_subcommand_uses_are_rejected() {
        let err = |argv: &[&str]| Cli::try_parse_from(argv).unwrap_err().to_string();
        // convert needs an output.
        assert!(err(&["psmc-rs", "convert", "in.vcf"]).contains("--psmcfa"));
        // Inference options do not apply to a subcommand: before it, the
        // subcommand's name is read as the inference input.
        assert!(err(&["psmc-rs", "--threads", "4", "convert", "in.vcf"]).contains("N_ITER"));
        assert!(
            err(&[
                "psmc-rs",
                "convert",
                "in.vcf",
                "--psmcfa",
                "o",
                "--threads",
                "4"
            ])
            .contains("--threads")
        );
    }
}
