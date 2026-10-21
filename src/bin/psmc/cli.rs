//! Command-line arguments.

use anyhow::Result;
use clap::{ArgGroup, Args, Parser, Subcommand, ValueEnum};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

use psmc_rs::bootstrap::DEFAULT_BOOTSTRAP_TRUNK;
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, VcfMode};
use psmc_rs::model::{MAX_STATES, n_steps_for_pattern};
use psmc_rs::opt::{DEFAULT_MSTEP_ITERS, MStepConfig};

/// The exit status of a usage error, as clap gives for the errors it finds.
pub const USAGE_EXIT: u8 = 2;

/// A command line that parses but cannot be run as given: an invalid
/// pattern, options that contradict each other, or outputs that would
/// overwrite each other or the input. It is found before the input is read
/// (only for an output that a pipe's format rules out, once the pipe has
/// been read), and the program then exits with [`USAGE_EXIT`], as for the
/// errors clap finds.
#[derive(Debug)]
pub struct UsageError(pub String);

impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UsageError {}

impl UsageError {
    /// Whether `e`, or an error it wraps, is a usage error.
    pub fn is_in(e: &anyhow::Error) -> bool {
        e.chain().any(|cause| cause.is::<UsageError>())
    }
}

/// Return a [`UsageError`] with a `format!` message.
macro_rules! usage_bail {
    ($($arg:tt)*) => {
        return Err(anyhow::Error::new($crate::cli::UsageError(format!($($arg)*))))
    };
}
pub(crate) use usage_bail;

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
}

/// `--bin-callable`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BinCallableArg {
    /// Weigh each called bin by its callable bases when they are counted exactly (VCF)
    #[default]
    Auto,
    /// Treat every called bin as BIN_SIZE bases, as fq2psmcfa and C psmc do
    Ignore,
    /// As auto, and also weigh multihetsep's bins by their approximate counts
    Approximate,
}

impl BinCallableArg {
    /// The name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            BinCallableArg::Auto => "auto",
            BinCallableArg::Ignore => "ignore",
            BinCallableArg::Approximate => "approximate",
        }
    }

    /// Whether a fit in this mode weighs each called bin by counts of
    /// callable bases obtained as `source` says (`"exact"` from a VCF,
    /// `"approximate"` from multihetsep): `auto` takes exact counts only,
    /// `approximate` both, `ignore` none.
    pub fn uses_counts(self, source: &str) -> bool {
        match self {
            BinCallableArg::Auto => source == "exact",
            BinCallableArg::Approximate => true,
            BinCallableArg::Ignore => false,
        }
    }
}

/// C psmc's default pattern (`-p`).
pub const DEFAULT_PATTERN: &str = "4+5*3+4";

/// C psmc's default initial t_max (`-t`).
pub const DEFAULT_T_MAX: f64 = 15.0;

/// C psmc's default initial θ0/ρ0 (`-r`).
pub const DEFAULT_TR_RATIO: f64 = 4.0;

/// The default mutation rate per base pair per generation (`--mu`).
pub const DEFAULT_MU: f64 = 2.5e-8;

// The defaults of the other options; the consensus filters are vcf2fq's
// and fq2psmcfa's.
pub const DEFAULT_BIN_SIZE: usize = 100;
const DEFAULT_MHS_PAIR: (usize, usize) = (0, 1);
const DEFAULT_MIN_MAPQ: u64 = 10;
const DEFAULT_INDEL_WINDOW: u64 = 5;
const DEFAULT_MIN_QUAL: u64 = 10;
const DEFAULT_MIN_GOOD_BASES: u64 = 10_000;
const DEFAULT_MIN_GOOD_FRACTION: f64 = 0.2;
const DEFAULT_BOOTSTRAP_SEED: u64 = 42;
const DEFAULT_GEN_YEARS: f64 = 25.0;

/// A positive integer.
pub fn positive_usize(s: &str) -> Result<usize, String> {
    match whole_number(s)? {
        0 => Err("must be at least 1".to_string()),
        n => Ok(n),
    }
}

/// A whole number of at least 0.
pub fn whole_number(s: &str) -> Result<usize, String> {
    s.parse()
        .map_err(|_| format!("'{s}' is not a whole number"))
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
pub fn positive_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v > 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be a positive number, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// A finite number of at least 0.
pub fn non_negative_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v >= 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be 0 or more, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// `--t-max`: inside the range the M-step searches. A start outside it
/// would be moved to its edge by the first M-step, and one near `T_INF`,
/// where the time grid ends, breaks the first E-step.
fn parse_t_max(s: &str) -> Result<f64, String> {
    let v = positive_f64(s)?;
    let MStepConfig {
        tmax_lo, tmax_hi, ..
    } = MStepConfig::default();
    if !(tmax_lo..=tmax_hi).contains(&v) {
        return Err(format!(
            "must be between {tmax_lo} and {tmax_hi}, the range the M-step searches, got {v}"
        ));
    }
    Ok(v)
}

/// `--n-steps`: a model of 2 to `MAX_STATES` intervals.
fn parse_n_steps(s: &str) -> Result<usize, String> {
    let n = whole_number(s)?;
    if !(1..MAX_STATES).contains(&n) {
        return Err(format!(
            "must be between 1 and {} (2 to {MAX_STATES} intervals), got {n}",
            MAX_STATES - 1
        ));
    }
    Ok(n)
}

/// The model layout `(n_steps, pattern)` that `--pattern` and `--n-steps`
/// give: C psmc's default pattern without either, `n_steps` from the
/// pattern without `--n-steps`, and one free lambda per interval with
/// `--n-steps` alone. Given both, `--n-steps` must match the pattern;
/// its number of intervals (`n_steps + 1`, the form psmc-rs 0.1 used) is
/// accepted too. An invalid pattern or a mismatch is a [`UsageError`].
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
    let derived = match n_steps_for_pattern(pattern) {
        Ok(n) => n,
        // A term that is not a number gives the parser's bare "invalid
        // digit found in string".
        Err(e) if e.is::<std::num::ParseIntError>() => usage_bail!(
            "invalid --pattern '{pattern}': its terms, joined by '+', are each L or R*L in whole \
             numbers (R free lambdas of L intervals each), as in 4+5*3+4"
        ),
        Err(e) => usage_bail!("invalid --pattern '{pattern}': {e:#}"),
    };
    match n_steps {
        None => Ok((derived, Some(pattern.to_string()))),
        Some(n) if n == derived || n == derived + 1 => Ok((derived, Some(pattern.to_string()))),
        Some(n) => usage_bail!(
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
    /// Input format; auto recognises it from the content
    #[arg(long, value_enum, value_name = "FORMAT", default_value_t = FormatArg::Auto)]
    pub input_format: FormatArg,

    /// Bases per bin: bins VCF and multihetsep input (fq2psmcfa -s), scales every format.
    ///
    /// The time and Ne scaling uses it for every format: for a psmcfa file,
    /// give the -s it was made with.
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

    /// VCF: consensus (vcf2fq | fq2psmcfa) or mask mode.
    ///
    /// Consensus mode reproduces `vcfutils.pl vcf2fq | fq2psmcfa` on an
    /// all-sites VCF from `bcftools call -c`; mask mode reads the sample's
    /// genotypes with callable masks; auto picks one from the first records.
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
    #[arg(long, value_name = "MQ", default_value_t = DEFAULT_MIN_MAPQ)]
    pub min_mapq: u64,

    /// VCF consensus mode: bases masked on each side of an indel (vcf2fq -l)
    #[arg(long, value_name = "BP", default_value_t = DEFAULT_INDEL_WINDOW)]
    pub indel_window: u64,

    /// VCF consensus mode: minimum base quality (fq2psmcfa -q)
    #[arg(long, value_name = "Q", default_value_t = DEFAULT_MIN_QUAL)]
    pub min_qual: u64,

    /// VCF and multihetsep: drop sequences with fewer called bases (fq2psmcfa -g; 0 disables).
    ///
    /// A sequence with fewer called bases than this is left out of the
    /// analysis.
    #[arg(long, value_name = "BASES", default_value_t = DEFAULT_MIN_GOOD_BASES)]
    pub min_good_bases: u64,

    /// VCF and multihetsep: drop sequences whose called share is lower (0 disables).
    ///
    /// A sequence whose called bases are less than this share of its length
    /// is left out of the analysis (fq2psmcfa's fixed 0.2).
    #[arg(long, value_name = "SHARE", default_value_t = DEFAULT_MIN_GOOD_FRACTION, value_parser = parse_fraction)]
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

    /// The flags given a value other than their default, as they are spelt
    /// on the command line, with the inputs they apply to and their values
    /// (none for a switch, one per use for a repeated flag).
    fn given(&self) -> Vec<(&'static str, AppliesTo, Vec<String>)> {
        use AppliesTo::*;
        fn unless<T: PartialEq + fmt::Display>(value: T, default: T) -> Option<Vec<String>> {
            (value != default).then(|| vec![value.to_string()])
        }
        let paths = |paths: &[PathBuf]| {
            let shown = paths.iter().map(|p| p.display().to_string());
            (!paths.is_empty()).then(|| shown.collect())
        };
        let depth = match self.depth {
            DepthThresholds::Fixed { min, max } => Some(vec![format!("{min}:{max}")]),
            DepthThresholds::Auto => None,
        };
        let (i, j) = self.mhs_pair;
        let pair = (self.mhs_pair != DEFAULT_MHS_PAIR).then(|| vec![format!("{i},{j}")]);
        let format = self.input_format.name();
        let vcf_mode = self.vcf_mode.mode().name();
        [
            (
                "--input-format",
                Every,
                unless(format, FormatArg::Auto.name()),
            ),
            ("--bin-size", Every, unless(self.bin_size, DEFAULT_BIN_SIZE)),
            (
                "--batch-size",
                Every,
                self.batch_size.map(|n| vec![n.to_string()]),
            ),
            (
                "--vcf-sample",
                Vcf,
                self.vcf_sample.clone().map(|s| vec![s]),
            ),
            ("--vcf-mode", Vcf, unless(vcf_mode, VcfMode::Auto.name())),
            ("--vcf-mask", Vcf, paths(&self.vcf_mask)),
            ("--vcf-negative-mask", Vcf, paths(&self.vcf_negative_mask)),
            (
                "--vcf-keep-filtered",
                VcfMask,
                self.vcf_keep_filtered.then(Vec::new),
            ),
            (
                "--vcf-no-contig-end",
                VcfMask,
                self.vcf_no_contig_end.then(Vec::new),
            ),
            ("--depth", VcfConsensus, depth),
            (
                "--min-mapq",
                VcfConsensus,
                unless(self.min_mapq, DEFAULT_MIN_MAPQ),
            ),
            (
                "--indel-window",
                VcfConsensus,
                unless(self.indel_window, DEFAULT_INDEL_WINDOW),
            ),
            (
                "--min-qual",
                VcfConsensus,
                unless(self.min_qual, DEFAULT_MIN_QUAL),
            ),
            ("--mhs-pair", Mhs, pair),
            (
                "--min-good-bases",
                Filtered,
                unless(self.min_good_bases, DEFAULT_MIN_GOOD_BASES),
            ),
            (
                "--min-good-fraction",
                Filtered,
                unless(self.min_good_fraction, DEFAULT_MIN_GOOD_FRACTION),
            ),
        ]
        .into_iter()
        .filter_map(|(flag, applies, values)| Some((flag, applies, values?)))
        .collect()
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
    /// The inputs that go through fq2psmcfa's sequence filter: all but
    /// psmcfa.
    Filtered,
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
            AppliesTo::Filtered => format != InputFormat::Psmcfa,
        }
    }
}

/// Command-line arguments, rebuilt from parsed ones.
#[derive(Default)]
struct ArgList(Vec<String>);

impl ArgList {
    /// `flag value` for each of `values` (none, one or several).
    fn each<T: fmt::Display>(
        &mut self,
        flag: &str,
        values: impl IntoIterator<Item = T>,
    ) -> &mut Self {
        for value in values {
            self.0.extend([flag.to_string(), value.to_string()]);
        }
        self
    }

    /// `flag value`, unless `value` is the default.
    fn unless<T: PartialEq + fmt::Display>(
        &mut self,
        flag: &str,
        value: T,
        default: T,
    ) -> &mut Self {
        self.each(flag, (value != default).then_some(value))
    }

    /// `flag` alone, if `on`.
    fn switch(&mut self, flag: &str, on: bool) -> &mut Self {
        if on {
            self.0.push(flag.to_string());
        }
        self
    }

    /// The input flags of `input` given values other than their defaults.
    fn input(&mut self, input: &InputArgs) -> &mut Self {
        for (flag, _, values) in input.given() {
            if values.is_empty() {
                self.switch(flag, true);
            } else {
                self.each(flag, values);
            }
        }
        self
    }
}

const ABOUT: &str = "PSMC: population size history from one diploid genome";

const LONG_ABOUT: &str = "\
PSMC (Li & Durbin 2011): population size history from one diploid genome.

psmc-rs INPUT OUTPUT N_ITER fits the model to INPUT (psmcfa, multihetsep or
VCF) with N_ITER EM iterations and writes the parameters to OUTPUT (JSON)
with an HTML report next to it. `psmc-rs convert` writes psmcfa and
multihetsep files, `psmc-rs qc` reports how suitable an input is, and
--tui opens an interactive interface.";

const AFTER_HELP: &str = "\
Examples:
  psmc-rs sample.psmcfa fit.json 25 --psmc-out fit.psmc
  psmc-rs calls.vcf.gz fit.json 25 --bootstrap 100 --threads 8
  psmc-rs convert calls.vcf.gz --psmcfa sample.psmcfa.gz
  psmc-rs qc sample.psmcfa --json qc.json";

/// The command line: inference unless a subcommand is given.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "psmc-rs",
    version,
    about = ABOUT,
    long_about = LONG_ABOUT,
    after_help = AFTER_HELP,
    arg_required_else_help = true,
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true
)]
pub struct Cli {
    /// `convert` or `qc`; inference without one.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// The arguments of inference.
    #[command(flatten)]
    pub infer: InferArgs,
}

impl Cli {
    /// Parse `argv` as [`Parser::try_parse_from`] does, with clearer errors
    /// for two mistakes ([`explain`]).
    pub fn try_parse_args<I, T>(argv: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        let argv: Vec<OsString> = argv.into_iter().map(Into::into).collect();
        Self::try_parse_from(&argv).map_err(|e| explain(e, &argv))
    }
}

/// Clap's error `e` for `argv`, made clearer for two mistakes. An argument
/// after INPUT OUTPUT N_ITER (such as a second input from a shell pattern)
/// is reported as unexpected: clap takes it for a subcommand, which cannot
/// follow inference's arguments, and says so even when it is none. A
/// subcommand after other arguments is read as inference's INPUT, so the
/// error gets a tip to give it first.
fn explain(e: clap::Error, argv: &[OsString]) -> clap::Error {
    use clap::CommandFactory;
    use clap::error::{ContextKind, ContextValue, ErrorKind};
    let cmd = Cli::command();
    let subcommand = |arg: &OsStr| {
        arg.to_str()
            .filter(|&name| name == "help" || cmd.find_subcommand(name).is_some())
            .map(str::to_string)
    };
    let late_subcommand = match argv {
        [_, first, rest @ ..] if subcommand(first).is_none() => {
            rest.iter().find_map(|arg| subcommand(arg))
        }
        _ => None,
    };
    let (mut e, tip) = match e.get(ContextKind::InvalidSubcommand) {
        Some(ContextValue::String(arg))
            if e.kind() == ErrorKind::ArgumentConflict && subcommand(arg.as_ref()).is_none() =>
        {
            let mut unexpected = clap::Error::new(ErrorKind::UnknownArgument).with_cmd(&cmd);
            unexpected.insert(ContextKind::InvalidArg, ContextValue::String(arg.clone()));
            if let Some(usage) = e.get(ContextKind::Usage) {
                unexpected.insert(ContextKind::Usage, usage.clone());
            }
            let tip = "inference takes three arguments, INPUT OUTPUT N_ITER (one input file per \
                       run)";
            (unexpected, Some(tip.to_string()))
        }
        _ => (e, None),
    };
    let tip = match late_subcommand {
        Some(name) => format!("a subcommand must be the first argument: psmc-rs {name} ..."),
        None => match tip {
            Some(tip) => tip,
            None => return e,
        },
    };
    let mut tips = match e.get(ContextKind::Suggested) {
        Some(ContextValue::StyledStrs(tips)) => tips.clone(),
        _ => Vec::new(),
    };
    tips.push(tip.into());
    e.insert(ContextKind::Suggested, ContextValue::StyledStrs(tips));
    e
}

/// The subcommands.
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Write an input as psmcfa or multihetsep.
    ///
    /// The input (VCF, multihetsep or psmcfa) is read exactly as inference
    /// reads it.
    #[command(arg_required_else_help = true)]
    Convert(ConvertArgs),
    /// Report how suitable an input is for PSMC.
    ///
    /// The input is read exactly as inference reads it.
    #[command(arg_required_else_help = true)]
    Qc(QcArgs),
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

    /// How to read it.
    #[command(flatten)]
    pub input: InputArgs,

    /// Write psmcfa, as fq2psmcfa lays it out (gzip when FILE ends in .gz)
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub psmcfa: Option<PathBuf>,

    /// Write every sequence as multihetsep (VCF and multihetsep input).
    ///
    /// The heterozygous sites, as generate_multihetsep.py writes them.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub multihetsep: Option<PathBuf>,

    /// Write one multihetsep file per sequence, PREFIX.<name>.multihetsep.
    ///
    /// With .gz added, gzip-compressed, when PREFIX ends in .gz; MSMC2 takes
    /// one file per chromosome.
    #[arg(long, value_name = "PREFIX", help_heading = "Output")]
    pub multihetsep_per_chrom: Option<PathBuf>,

    /// Write the input QC report as JSON ('-' for stdout).
    ///
    /// With '-', the result lines go to stderr.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub qc_json: Option<PathBuf>,

    /// Print only results and warnings
    #[arg(long, short = 'q', help_heading = "Output")]
    pub quiet: bool,
}

/// `psmc-rs qc`.
#[derive(Args, Debug, Clone)]
pub struct QcArgs {
    /// Input file: psmcfa, multihetsep or VCF (gzip/bgzip allowed)
    #[arg(value_name = "INPUT")]
    pub input_file: PathBuf,

    /// How to read it.
    #[command(flatten)]
    pub input: InputArgs,

    /// Also write the report as JSON ('-' for stdout).
    ///
    /// With '-', the text report goes to stderr.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub json: Option<PathBuf>,

    /// Check the input for a bootstrap with trunks of BINS bins.
    ///
    /// Inference's default trunk is 500000 bins.
    #[arg(long, value_name = "BINS", value_parser = positive_usize, help_heading = "Report")]
    pub bootstrap_block_size: Option<usize>,

    /// Exit with status 2 when the report has a warning
    #[arg(long, help_heading = "Report")]
    pub strict: bool,
}

/// Inference: `psmc-rs INPUT OUTPUT N_ITER [options]`.
#[derive(Args, Debug, Clone, PartialEq)]
pub struct InferArgs {
    /// Open the terminal interface; the other arguments fill its form
    #[arg(long)]
    pub tui: bool,

    /// Input file: psmcfa, multihetsep or VCF (gzip/bgzip allowed).
    ///
    /// A directory takes its first input file by name.
    #[arg(value_name = "INPUT", required_unless_present = "tui")]
    pub input_file: Option<PathBuf>,

    /// Output JSON of the fitted parameters; the HTML report is written next to it
    #[arg(value_name = "OUTPUT", required_unless_present = "tui")]
    pub output_file: Option<PathBuf>,

    /// EM iterations (C psmc -N; 0 writes the starting model)
    #[arg(value_name = "N_ITER", required_unless_present = "tui")]
    pub n_iter: Option<usize>,

    /// How to read the input.
    #[command(flatten)]
    pub input: InputArgs,

    /// Worker threads (default: one per core); results do not depend on it.
    ///
    /// They run the E-step (sequences in parallel), the bootstrap and the
    /// TMRCA decoding.
    #[arg(long, value_name = "N", value_parser = positive_usize, help_heading = "Run")]
    pub threads: Option<usize>,

    /// Free-parameter pattern of the time intervals (C psmc -p; default 4+5*3+4).
    ///
    /// Without it, the model has C psmc's default pattern 4+5*3+4, or one
    /// lambda per interval when --n-steps is given alone.
    #[arg(long, help_heading = "Model")]
    pub pattern: Option<String>,

    /// Number of time intervals minus one (C psmc's n; 1 to 9999; default: from --pattern).
    ///
    /// Taken from --pattern, and checked against it when both are given;
    /// alone, it fits one lambda per interval.
    #[arg(long, value_name = "N", value_parser = parse_n_steps, help_heading = "Model")]
    pub n_steps: Option<usize>,

    /// Initial t_max, in units of 2N0 generations (C psmc -t; 1 to 200).
    ///
    /// 1 to 200 is the range the M-step searches.
    #[arg(long, default_value_t = DEFAULT_T_MAX, value_parser = parse_t_max, help_heading = "Model")]
    pub t_max: f64,

    /// Initial theta0/rho0 ratio (C psmc -r); rho0 = theta0 / this
    #[arg(long, value_name = "RATIO", default_value_t = DEFAULT_TR_RATIO, value_parser = positive_f64, help_heading = "Model")]
    pub tr_ratio: f64,

    /// Initial theta0 per bin (default: from the data, as C psmc).
    ///
    /// By default -ln(1 - K/L), as C psmc computes it; when the fit uses the
    /// callable bases of each bin (see --bin-callable), the theta0 at which
    /// the called bins' expected heterozygous bins, the sum of
    /// 1 - exp(-theta0 c / BIN_SIZE), equal K.
    #[arg(long, value_parser = positive_f64, help_heading = "Model")]
    pub theta0: Option<f64>,

    /// Initial rho0 per bin (default: theta0 / --tr-ratio)
    #[arg(long, value_parser = positive_f64, help_heading = "Model")]
    pub rho0: Option<f64>,

    /// Start from a saved fit, the JSON output of an earlier run (C psmc -i).
    ///
    /// Its parameters replace theta0, rho0, t_max and flat lambdas as the
    /// start; its pattern and n_steps are used unless given.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["theta0", "rho0", "t_max", "tr_ratio"], help_heading = "Model")]
    pub init_params: Option<PathBuf>,

    /// Mutation rate per base pair per generation, for scaling
    #[arg(long, default_value = "2.5e-8", value_parser = positive_f64, help_heading = "Model")]
    pub mu: f64,

    /// How to fit partly callable bins: by their callable bases (auto) or as full bins (ignore).
    ///
    /// A VCF read with its callable mask counts exactly how many bases of
    /// each bin are callable: auto fits each called bin by them (a bin of c
    /// callable bases emits as c of BIN_SIZE bases), which removes most of
    /// the bias that missing data scattered in short runs cause (theta,
    /// times and Ne low by the missing share or more); ignore treats every
    /// called bin as BIN_SIZE bases, as fq2psmcfa and C psmc do.
    ///
    /// Multihetsep does not say which sites are uncalled, so its counts
    /// follow the reader's placement of them and auto does not use them;
    /// approximate does, which made the fit worse in the validation. Either
    /// way multihetsep with missing data biases theta and Ne low (QC warns
    /// above 5% of the sites uncalled): fit the VCF with its mask instead.
    ///
    /// psmcfa input records no counts and is fitted as C psmc fits it in
    /// every mode.
    #[arg(long, value_enum, value_name = "MODE", default_value_t = BinCallableArg::Auto, help_heading = "Model")]
    pub bin_callable: BinCallableArg,

    /// Cap on the L-BFGS iterations of an M-step, which otherwise runs to convergence.
    ///
    /// The M-step converges, as C psmc's does, in 10 to 20 iterations with
    /// C psmc's patterns and up to about 70 with one lambda per interval.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MSTEP_ITERS, value_parser = positive_usize, help_heading = "Model")]
    pub mstep_iters: usize,

    /// Weight of the M-step's penalty on log-lambda differences of neighbouring intervals.
    ///
    /// 0 is C psmc's objective.
    #[arg(long, value_name = "WEIGHT", default_value_t = 0.0, value_parser = non_negative_f64, help_heading = "Model")]
    pub smooth_lambda: f64,

    /// Bootstrap replicates to fit (0: none).
    ///
    /// As in C psmc's bootstrap, each draws trunks of the input with
    /// replacement and is fitted from a fresh start, or from the
    /// --init-params parameters when given (C psmc -i -b).
    #[arg(
        long,
        value_name = "N",
        default_value_t = 0,
        help_heading = "Bootstrap"
    )]
    pub bootstrap: usize,

    /// Bootstrap trunk size in bins (C splitfa; 500000 = 50 Mb at 100 bp bins).
    ///
    /// It must leave the input at least two trunks.
    #[arg(long, value_name = "BINS", default_value_t = DEFAULT_BOOTSTRAP_TRUNK, value_parser = positive_usize, alias = "bootstrap-trunk-size", help_heading = "Bootstrap")]
    pub bootstrap_block_size: usize,

    /// Seed of the bootstrap resampling
    #[arg(
        long,
        value_name = "SEED",
        default_value_t = DEFAULT_BOOTSTRAP_SEED,
        help_heading = "Bootstrap"
    )]
    pub bootstrap_seed: u64,

    /// EM iterations of each replicate (default: N_ITER)
    #[arg(long, value_name = "N", help_heading = "Bootstrap")]
    pub bootstrap_iters: Option<usize>,

    /// Directory of the replicate and summary files (default: OUTPUT with .bootstrap).
    ///
    /// The replicates are written as replicate_NNN.json and .psmc, the
    /// summary as summary.tsv and .json; the default directory is OUTPUT
    /// with .bootstrap instead of .json. An earlier bootstrap's
    /// replicate_NNN.* and summary.* files there are removed first.
    #[arg(long, value_name = "DIR", help_heading = "Bootstrap")]
    pub bootstrap_dir: Option<PathBuf>,

    /// Do not draw progress bars
    #[arg(long, help_heading = "Run")]
    pub no_progress: bool,

    /// Print only results and warnings.
    ///
    /// No input description, model summary or QC report; QC warnings are
    /// still listed.
    #[arg(long, short = 'q', help_heading = "Run")]
    pub quiet: bool,

    /// Write the input QC report as JSON ('-' for stdout).
    ///
    /// With '-', the result lines go to stderr.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub qc_json: Option<PathBuf>,

    /// Also write C psmc's .psmc output (rounds 0 to N_ITER), for psmc_plot.pl.
    ///
    /// For psmc_plot.pl and other tools made for C psmc. The file does not
    /// record the bin size: plot it with psmc_plot.pl -s BIN_SIZE -u MU -g
    /// GEN_YEARS -n N_ITER (its defaults are -s 100 -u 2.5e-8 -g 25 -n 20).
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub psmc_out: Option<PathBuf>,

    /// Write the posterior TMRCA of every bin as TSV (gzip when FILE ends in .gz).
    ///
    /// Each state is at its average coalescent time in years; the HTML
    /// report then shows the TMRCA panels.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    pub tmrca_out: Option<PathBuf>,

    /// Years per generation, for the time axes.
    ///
    /// The time axes of the Ne curve (report), the bootstrap summary and the
    /// TMRCA output.
    #[arg(long, value_name = "YEARS", default_value_t = DEFAULT_GEN_YEARS, value_parser = positive_f64, alias = "tmrca-gen-years", help_heading = "Output")]
    pub gen_years: f64,
}

impl InferArgs {
    /// The arguments that reproduce this inference run on the command line
    /// (without `--tui`), as the TUI shows and runs it.
    pub fn to_args(&self) -> Vec<String> {
        let path = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string());
        let positional = [
            path(&self.input_file),
            path(&self.output_file),
            self.n_iter.map(|n| n.to_string()),
        ];
        let mut args = ArgList(positional.into_iter().flatten().collect());
        args.input(&self.input)
            .each("--pattern", &self.pattern)
            .each("--n-steps", self.n_steps)
            .unless("--t-max", self.t_max, DEFAULT_T_MAX)
            .unless("--tr-ratio", self.tr_ratio, DEFAULT_TR_RATIO)
            .each("--theta0", self.theta0)
            .each("--rho0", self.rho0)
            .each("--init-params", path(&self.init_params))
            .unless("--mu", self.mu, DEFAULT_MU)
            .unless(
                "--bin-callable",
                self.bin_callable.name(),
                BinCallableArg::Auto.name(),
            )
            .unless("--mstep-iters", self.mstep_iters, DEFAULT_MSTEP_ITERS)
            .unless("--smooth-lambda", self.smooth_lambda, 0.0)
            .each("--threads", self.threads)
            .unless("--gen-years", self.gen_years, DEFAULT_GEN_YEARS);
        if self.bootstrap > 0 {
            args.each("--bootstrap", [self.bootstrap])
                .unless(
                    "--bootstrap-block-size",
                    self.bootstrap_block_size,
                    DEFAULT_BOOTSTRAP_TRUNK,
                )
                .unless(
                    "--bootstrap-seed",
                    self.bootstrap_seed,
                    DEFAULT_BOOTSTRAP_SEED,
                );
        }
        args.each("--bootstrap-iters", self.bootstrap_iters)
            .each("--bootstrap-dir", path(&self.bootstrap_dir))
            .each("--psmc-out", path(&self.psmc_out))
            .each("--qc-json", path(&self.qc_json))
            .each("--tmrca-out", path(&self.tmrca_out))
            .switch("--quiet", self.quiet)
            .switch("--no-progress", self.no_progress);
        args.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inference arguments of `psmc-rs in out 1` followed by `args`,
    /// given as one string split at whitespace.
    fn parse(args: &str) -> Result<InferArgs, clap::Error> {
        let argv = ["psmc-rs", "in", "out", "1"].into_iter();
        Cli::try_parse_from(argv.chain(args.split_whitespace())).map(|c| c.infer)
    }

    fn parse_err(args: &str) -> String {
        parse(args).unwrap_err().to_string()
    }

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

        let to_args = |a: &InputArgs| ArgList::default().input(a).0.clone();
        assert_eq!(input(&to_args(&a).join(" ")), a);
        assert_eq!(to_args(&input("")), Vec::<String>::new());
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
        let err = model_layout(human.as_deref(), Some(30)).unwrap_err();
        assert!(UsageError::is_in(&err));
        let err = err.to_string();
        assert!(err.contains("has 64 intervals (n_steps 63)"), "{err}");
        let err = model_layout(Some("4+x"), None).unwrap_err();
        assert!(UsageError::is_in(&err));
        let err = format!("{err:#}");
        assert!(err.contains("invalid --pattern '4+x'"), "{err}");
    }

    #[test]
    fn the_defaults_are_c_psmcs() {
        let args = parse("").unwrap();
        assert_eq!(
            (args.t_max, args.tr_ratio, args.mu),
            (DEFAULT_T_MAX, DEFAULT_TR_RATIO, DEFAULT_MU)
        );
        // QC's messages about a default start name the same ratio.
        assert_eq!(DEFAULT_TR_RATIO, psmc_rs::qc::DEFAULT_TR_RATIO);
        // The callable bases of each bin are fitted when the input has them.
        assert_eq!(args.bin_callable, BinCallableArg::Auto);
        let mode = |name: &str| parse(&format!("--bin-callable {name}")).map(|a| a.bin_callable);
        assert_eq!(mode("ignore").unwrap(), BinCallableArg::Ignore);
        assert_eq!(mode("approximate").unwrap(), BinCallableArg::Approximate);
        assert!(mode("x").is_err());
        // auto fits exact counts only; approximate also multihetsep's.
        use BinCallableArg::*;
        let uses =
            |mode: BinCallableArg| [mode.uses_counts("exact"), mode.uses_counts("approximate")];
        assert_eq!(uses(Auto), [true, false]);
        assert_eq!(uses(Approximate), [true, true]);
        assert_eq!(uses(Ignore), [false, false]);
    }

    #[test]
    fn saved_parameters_replace_the_starting_values() {
        let with_saved = |args: &str| parse(&format!("--init-params fit.json {args}"));
        assert!(with_saved("").is_ok());
        assert!(with_saved("--pattern 4+5*3+4 --mu 1e-8").is_ok());
        for flag in ["--theta0", "--rho0", "--t-max", "--tr-ratio"] {
            let err = with_saved(&format!("{flag} 5")).unwrap_err().to_string();
            assert!(err.contains("cannot be used with"), "{flag}: {err}");
        }
    }

    #[test]
    fn model_values_are_checked() {
        for t_max in ["1000", "900", "200.5", "0.5"] {
            let err = parse_err(&format!("--t-max {t_max}"));
            assert!(
                err.contains("must be between 1 and 200, the range the M-step searches"),
                "{t_max}: {err}"
            );
        }
        assert!(parse_err("--t-max 0").contains("positive"));
        for n in ["0", "10000", "100000"] {
            let err = parse_err(&format!("--n-steps {n}"));
            assert!(err.contains("must be between 1 and 9999"), "{n}: {err}");
        }
        assert!(parse_err("--n-steps x").contains("not a whole number"));
        assert!(parse_err("--tr-ratio=-4").contains("positive"));
        assert!(parse_err("--smooth-lambda=-1").contains("0 or more"));
        assert!(parse_err("--mstep-iters 0").contains("at least 1"));
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

    /// Inference, `convert` and `qc` with their arguments.
    fn commands_and_arguments() -> Vec<(String, clap::Arg)> {
        use clap::CommandFactory;
        let cmd = Cli::command();
        std::iter::once(&cmd)
            .chain(cmd.get_subcommands())
            .flat_map(|c| {
                c.get_arguments()
                    .map(|a| (c.get_name().to_string(), a.clone()))
            })
            .collect()
    }

    /// `-h` lists each argument on one line, with the first paragraph of its
    /// documentation; `--help` adds the rest.
    #[test]
    fn every_argument_has_a_one_line_summary() {
        for (command, arg) in commands_and_arguments() {
            let id = arg.get_id().to_string();
            let help = arg.get_help().map(ToString::to_string);
            let help = help.unwrap_or_else(|| panic!("{command} {id} has no help"));
            assert!(
                !help.contains('\n') && help.len() <= 90,
                "{command} {id}: {help}"
            );
            if let Some(long) = arg.get_long_help() {
                assert!(long.to_string().starts_with(&help), "{command} {id}");
            }
        }
    }

    /// Only mask mode selects a sample: consensus mode reads INFO fields,
    /// which bcftools computes over every sample of the VCF.
    #[test]
    fn the_sample_help_names_the_mode_that_uses_it() {
        let sample: Vec<clap::Arg> = commands_and_arguments()
            .into_iter()
            .filter_map(|(_, a)| (a.get_id() == "vcf_sample").then_some(a))
            .collect();
        assert_eq!(sample.len(), 3, "inference, convert and qc");
        for arg in sample {
            let summary = arg.get_help().unwrap().to_string();
            assert!(
                summary.starts_with("VCF mask mode: the sample"),
                "{summary}"
            );
            assert!(summary.contains("one-sample VCF"), "{summary}");
            let details = arg.get_long_help().unwrap().to_string();
            assert!(
                details.contains("Consensus mode reads INFO fields"),
                "{details}"
            );
        }
    }

    #[test]
    fn the_old_bin_size_flag_still_works() {
        assert_eq!(input("--mhs-bin-size 40").bin_size, 40);
    }

    #[test]
    fn the_old_generation_time_flag_still_works() {
        assert_eq!(parse("--tmrca-gen-years 29").unwrap().gen_years, 29.0);
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

    /// The TUI shows and runs `to_args`, so it must give back every
    /// setting.
    #[test]
    fn to_args_round_trips_every_inference_setting() {
        let round_trip = |argv: &str| {
            let args = Cli::try_parse_from(argv.split_whitespace()).unwrap().infer;
            let again = std::iter::once("psmc-rs".to_string()).chain(args.to_args());
            assert_eq!(Cli::try_parse_from(again).unwrap().infer, args);
        };
        round_trip(
            "psmc-rs in.vcf out.json 7 --input-format vcf --bin-size 50 --vcf-mask m.bed \
             --depth 5:60 --pattern 4+25*2+4+6 --n-steps 63 --t-max 20 --tr-ratio 5 \
             --theta0 0.01 --rho0 0.002 --mu 1.25e-8 --bin-callable ignore --mstep-iters 300 \
             --smooth-lambda 0.5 --threads 3 --gen-years 29 --bootstrap 20 \
             --bootstrap-block-size 100000 --bootstrap-seed 7 --bootstrap-iters 5 \
             --bootstrap-dir boot --psmc-out out.psmc --qc-json qc.json --tmrca-out t.tsv.gz \
             --quiet --no-progress",
        );
        round_trip("psmc-rs in out 1 --init-params fit.json");
    }

    #[test]
    fn subcommands_take_their_own_arguments() {
        let cli = |argv: &str| Cli::try_parse_from(argv.split_whitespace());
        let Some(Command::Convert(c)) =
            cli("psmc-rs convert in.vcf.gz --psmcfa out.psmcfa --bin-size 50")
                .unwrap()
                .command
        else {
            panic!("not convert")
        };
        assert_eq!(c.input_file, PathBuf::from("in.vcf.gz"));
        assert_eq!(c.input.bin_size, 50);

        let Some(Command::Qc(q)) = cli("psmc-rs qc in.psmcfa --strict").unwrap().command else {
            panic!("not qc")
        };
        assert!(q.strict && q.json.is_none() && q.bootstrap_block_size.is_none());

        // Inference still takes its three positional arguments.
        let infer = cli("psmc-rs in out 5").unwrap();
        assert!(infer.command.is_none());
        assert_eq!(infer.infer.n_iter, Some(5));
        assert!(cli("psmc-rs --tui").is_ok());
        assert!(cli("psmc-rs in out").is_err());
    }

    #[test]
    fn invalid_subcommand_uses_are_rejected() {
        let err = |argv: &str| {
            Cli::try_parse_args(argv.split_whitespace())
                .unwrap_err()
                .to_string()
        };
        // convert needs an output.
        assert!(err("psmc-rs convert in.vcf").contains("--psmcfa"));
        // Inference options do not apply to a subcommand: before it, the
        // subcommand's name is read as the inference input.
        let late = err("psmc-rs --threads 4 qc in.vcf");
        assert!(late.contains("N_ITER"), "{late}");
        assert!(
            late.contains("tip: a subcommand must be the first argument: psmc-rs qc ..."),
            "{late}"
        );
        assert!(err("psmc-rs qc in.vcf --threads 4").contains("--threads"));
        let after = err("psmc-rs in out 3 convert");
        assert!(
            after.contains("the subcommand 'convert' cannot be used with"),
            "{after}"
        );
    }

    /// An argument after N_ITER, such as a second input matched by a shell
    /// pattern, is named as unexpected, not as a subcommand.
    #[test]
    fn an_argument_after_n_iter_is_unexpected() {
        let e = Cli::try_parse_args("psmc-rs a.psmcfa b.psmcfa out.json 3".split_whitespace())
            .unwrap_err();
        assert_eq!(e.kind(), clap::error::ErrorKind::UnknownArgument);
        assert_eq!(e.exit_code(), i32::from(USAGE_EXIT));
        let text = e.to_string();
        assert!(
            text.starts_with("error: unexpected argument '3' found"),
            "{text}"
        );
        assert!(text.contains("one input file per run"), "{text}");
        assert!(text.contains("Usage: psmc-rs [OPTIONS]"), "{text}");
    }

    /// Without arguments, the program and its subcommands print their help,
    /// with the status of a usage error.
    #[test]
    fn no_arguments_show_the_help() {
        for argv in [
            &["psmc-rs"][..],
            &["psmc-rs", "convert"],
            &["psmc-rs", "qc"],
        ] {
            let e = Cli::try_parse_args(argv).unwrap_err();
            assert_eq!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand,
                "{argv:?}"
            );
            assert_eq!(e.exit_code(), i32::from(USAGE_EXIT));
            assert!(e.to_string().contains("Usage: psmc-rs"), "{argv:?}");
        }
    }
}
