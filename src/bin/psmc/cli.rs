//! Command-line arguments.

use clap::{ArgGroup, Args, FromArgMatches, Parser, Subcommand};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::path::PathBuf;

use psmc_rs::bootstrap::DEFAULT_BOOTSTRAP_TRUNK;
use psmc_rs::opt::DEFAULT_MSTEP_ITERS;

mod input;
mod values;

pub use input::{DEFAULT_BIN_SIZE, FormatArg, InputArgs};
pub use values::{
    BinCallableArg, DEFAULT_MU, DEFAULT_T_MAX, DEFAULT_TR_RATIO, model_layout, non_negative_f64,
    positive_f64, positive_usize, whole_number,
};
use values::{parse_n_steps, parse_t_max};

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

/// How a flag is spelt on the command line: `None` when it is not given (an
/// option without a value, a switch that is off), no value for a switch
/// that is on, one per use of a repeated flag. A flag is given when its
/// spelling differs from that of clap's default ([`defaults`]).
type Spelling<A> = fn(&A) -> Option<Vec<String>>;

/// The spelling of a flag given once.
fn one(value: impl fmt::Display) -> Option<Vec<String>> {
    Some(vec![value.to_string()])
}

/// The spelling of a switch.
fn on(switch: bool) -> Option<Vec<String>> {
    switch.then(Vec::new)
}

/// The spelling of a path given once.
fn path(path: &Option<PathBuf>) -> Option<Vec<String>> {
    path.as_ref().and_then(|p| one(p.display()))
}

/// `A` with clap's defaults (and no required argument).
fn defaults<A: Args + FromArgMatches>() -> A {
    let cmd = A::augment_args(clap::Command::new("defaults").ignore_errors(true));
    let matches = cmd.try_get_matches_from(["defaults"]).expect("no argument");
    A::from_arg_matches(&matches).expect("clap's defaults")
}

/// The arguments of `flags` and their values: `flag value` for each value,
/// `flag` alone for a switch.
fn spelt(flags: impl IntoIterator<Item = (&'static str, Vec<String>)>) -> Vec<String> {
    let mut args = Vec::new();
    for (flag, values) in flags {
        if values.is_empty() {
            args.push(flag.to_string());
        }
        args.extend(values.into_iter().flat_map(|v| [flag.to_string(), v]));
    }
    args
}

const ABOUT: &str = "PSMC: population size history from one diploid genome";

const LONG_ABOUT: &str = "\
PSMC (Li & Durbin 2011): population size history from one diploid genome.

psmc-rs INPUT OUTPUT N_ITER fits the model to INPUT (psmcfa, multihetsep or \
VCF) with N_ITER EM iterations and writes the parameters to OUTPUT (JSON) \
with an HTML report next to it. `psmc-rs convert` writes psmcfa and \
multihetsep files, `psmc-rs qc` reports how suitable an input is, and \
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

/// The narrowest COLUMNS that the help is wrapped to, and the width used
/// instead of a narrower one (clap's own when it finds no width).
const MIN_HELP_WIDTH: usize = 20;
const DEFAULT_HELP_WIDTH: usize = 100;

impl Cli {
    /// Parse `argv` as [`Parser::try_parse_from`] does, with clearer errors
    /// for two mistakes ([`explain`]).
    pub fn try_parse_args<I, T>(argv: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString>,
    {
        use clap::CommandFactory;
        let argv: Vec<OsString> = argv.into_iter().map(Into::into).collect();
        let mut cmd = Self::command();
        // clap wraps the help at COLUMNS when it is set; at 0 or a few
        // columns that would put one word on each line.
        let columns = std::env::var("COLUMNS")
            .ok()
            .and_then(|c| c.trim().parse::<usize>().ok());
        if columns.is_some_and(|c| c < MIN_HELP_WIDTH) {
            cmd = cmd.term_width(DEFAULT_HELP_WIDTH);
        }
        let matches = cmd
            .try_get_matches_from_mut(&argv)
            .map_err(|e| explain(e, &argv))?;
        Self::from_arg_matches(&matches).map_err(|e| e.format(&mut cmd))
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
    let (mut e, tip) = match e.get(ContextKind::InvalidSubcommand) {
        Some(ContextValue::String(arg))
            if e.kind() == ErrorKind::ArgumentConflict
                && subcommand(&cmd, arg.as_ref()).is_none() =>
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
    let late = late_subcommand(&cmd, argv)
        .map(|name| format!("a subcommand must be the first argument: psmc-rs {name} ..."));
    let Some(tip) = late.or(tip) else {
        return e;
    };
    let mut tips = match e.get(ContextKind::Suggested) {
        Some(ContextValue::StyledStrs(tips)) => tips.clone(),
        _ => Vec::new(),
    };
    tips.push(tip.into());
    e.insert(ContextKind::Suggested, ContextValue::StyledStrs(tips));
    e
}

/// `arg` if it names a subcommand of `cmd`.
fn subcommand(cmd: &clap::Command, arg: &OsStr) -> Option<String> {
    arg.to_str()
        .filter(|&name| name == "help" || cmd.find_subcommand(name).is_some())
        .map(str::to_string)
}

/// The subcommand named in `argv` after another argument, which is not the
/// value of an option before it (`--qc-json qc` names a file).
fn late_subcommand(cmd: &clap::Command, argv: &[OsString]) -> Option<String> {
    let (first, rest) = argv.get(1..)?.split_first()?;
    if first == "--" || subcommand(cmd, first).is_some() {
        return None;
    }
    let mut is_value = takes_value(cmd, first);
    for arg in rest {
        if !is_value {
            // Everything after `--` is a positional argument.
            if arg == "--" {
                break;
            }
            if let Some(name) = subcommand(cmd, arg) {
                return Some(name);
            }
        }
        is_value = !is_value && takes_value(cmd, arg);
    }
    None
}

/// Whether `arg` is an option of `cmd` whose value is the next argument:
/// `--name` without `=VALUE`, or `-x`.
fn takes_value(cmd: &clap::Command, arg: &OsStr) -> bool {
    let arg = arg.to_str().unwrap_or_default();
    let named = |a: &&clap::Arg| match arg.strip_prefix("--") {
        Some(long) => {
            a.get_long() == Some(long) || a.get_all_aliases().is_some_and(|all| all.contains(&long))
        }
        None => a.get_short().is_some_and(|c| arg == format!("-{c}")),
    };
    cmd.get_arguments()
        .find(named)
        .is_some_and(|a| a.get_action().takes_values())
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

/// `convert`'s usage, one output per line (clap's own is one line of 112
/// characters); outputs can be combined.
const CONVERT_USAGE: &str = "\
psmc-rs convert [OPTIONS] <INPUT> --psmcfa <FILE>
       psmc-rs convert [OPTIONS] <INPUT> --multihetsep <FILE>
       psmc-rs convert [OPTIONS] <INPUT> --multihetsep-per-chrom <PREFIX>";

/// `psmc-rs convert`.
#[derive(Args, Debug, Clone)]
#[command(override_usage = CONVERT_USAGE)]
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
    /// A PREFIX ending in .gz writes gzip-compressed files, the .gz moved to
    /// the end (pre.gz gives pre.chr1.multihetsep.gz). MSMC2 takes one file
    /// per chromosome.
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
    /// Inference's default trunk is 500,000 bins.
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
    #[arg(value_name = "N_ITER", required_unless_present = "tui", value_parser = whole_number::<usize>)]
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
    /// With --n-steps alone, one lambda per interval.
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

    /// Weigh bins by callable bases: VCF's exact counts (auto), also multihetsep's (approximate), or not (ignore).
    ///
    /// A VCF read with its callable mask counts exactly how many bases of
    /// each bin are callable: auto fits each called bin by them (a bin of c
    /// callable bases emits as c of BIN_SIZE bases), which removes most of
    /// the bias that missing data scattered in short runs cause (theta,
    /// times and Ne low by the missing share or more).
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
    #[arg(long, value_name = "N", default_value_t = 0, value_parser = whole_number::<usize>, help_heading = "Bootstrap")]
    pub bootstrap: usize,

    /// Bootstrap trunk size in bins (C splitfa; 500,000 = 50 Mb at 100 bp bins).
    ///
    /// It must leave the input at least two trunks.
    #[arg(long, value_name = "BINS", default_value_t = DEFAULT_BOOTSTRAP_TRUNK, value_parser = positive_usize, alias = "bootstrap-trunk-size", help_heading = "Bootstrap")]
    pub bootstrap_block_size: usize,

    /// Seed of the bootstrap resampling
    #[arg(long, value_name = "SEED", default_value_t = 42, value_parser = whole_number::<u64>, help_heading = "Bootstrap")]
    pub bootstrap_seed: u64,

    /// EM iterations of each replicate (default: N_ITER)
    #[arg(long, value_name = "N", value_parser = whole_number::<usize>, help_heading = "Bootstrap")]
    pub bootstrap_iters: Option<usize>,

    /// Directory of the replicate and summary files (default: OUTPUT with .bootstrap in place of .json).
    ///
    /// The replicates are written as replicate_NNN.json and .psmc, the
    /// summary as summary.tsv and .json. An earlier bootstrap's
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
    /// The file does not record the bin size: plot it with psmc_plot.pl -s
    /// BIN_SIZE -u MU -g GEN_YEARS -n N_ITER (its defaults are -s 100 -u
    /// 2.5e-8 -g 25 -n 20).
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
    #[arg(long, value_name = "YEARS", default_value_t = 25.0, value_parser = positive_f64, alias = "tmrca-gen-years", help_heading = "Output")]
    pub gen_years: f64,
}

/// Every inference flag but the input's and `--tui`, with its spelling.
const INFER_FLAGS: [(&str, Spelling<InferArgs>); 23] = [
    ("--pattern", |a| a.pattern.as_ref().and_then(one)),
    ("--n-steps", |a| a.n_steps.and_then(one)),
    ("--t-max", |a| one(a.t_max)),
    ("--tr-ratio", |a| one(a.tr_ratio)),
    ("--theta0", |a| a.theta0.and_then(one)),
    ("--rho0", |a| a.rho0.and_then(one)),
    ("--init-params", |a| path(&a.init_params)),
    // 2.5e-8 rather than 0.000000025.
    ("--mu", |a| one(format!("{:e}", a.mu))),
    ("--bin-callable", |a| one(a.bin_callable.name())),
    ("--mstep-iters", |a| one(a.mstep_iters)),
    ("--smooth-lambda", |a| one(a.smooth_lambda)),
    ("--threads", |a| a.threads.and_then(one)),
    ("--gen-years", |a| one(a.gen_years)),
    ("--bootstrap", |a| one(a.bootstrap)),
    ("--bootstrap-block-size", |a| one(a.bootstrap_block_size)),
    ("--bootstrap-seed", |a| one(a.bootstrap_seed)),
    ("--bootstrap-iters", |a| a.bootstrap_iters.and_then(one)),
    ("--bootstrap-dir", |a| path(&a.bootstrap_dir)),
    ("--psmc-out", |a| path(&a.psmc_out)),
    ("--qc-json", |a| path(&a.qc_json)),
    ("--tmrca-out", |a| path(&a.tmrca_out)),
    ("--quiet", |a| on(a.quiet)),
    ("--no-progress", |a| on(a.no_progress)),
];

impl InferArgs {
    /// The arguments that reproduce this inference run on the command line
    /// (without `--tui`), as the TUI shows and runs it.
    pub fn to_args(&self) -> Vec<String> {
        let defaults: InferArgs = defaults();
        // The bootstrap's block size and seed only with a bootstrap, which
        // alone uses them.
        let this = &match self.bootstrap {
            0 => InferArgs {
                bootstrap_block_size: defaults.bootstrap_block_size,
                bootstrap_seed: defaults.bootstrap_seed,
                ..self.clone()
            },
            _ => self.clone(),
        };
        let positional = [
            path(&self.input_file),
            path(&self.output_file),
            self.n_iter.and_then(one),
        ];
        let flags = INFER_FLAGS
            .iter()
            .filter(|(_, spell)| spell(this) != spell(&defaults))
            .filter_map(|&(flag, spell)| Some((flag, spell(this)?)));
        positional
            .into_iter()
            .flatten()
            .flatten()
            .chain(self.input.to_args())
            .chain(spelt(flags))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inference arguments of `psmc-rs in out 1` followed by `args`,
    /// given as one string split at whitespace.
    pub(super) fn parse(args: &str) -> Result<InferArgs, clap::Error> {
        let argv = ["psmc-rs", "in", "out", "1"].into_iter();
        Cli::try_parse_args(argv.chain(args.split_whitespace())).map(|c| c.infer)
    }

    /// The error of [`parse`].
    pub(super) fn parse_err(args: &str) -> String {
        parse(args).unwrap_err().to_string()
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

    /// `-h` lists each argument with the first paragraph of its
    /// documentation, one sentence (clap wraps it at the terminal's width);
    /// `--help` adds the rest.
    #[test]
    fn every_argument_has_a_one_sentence_summary() {
        for (command, arg) in commands_and_arguments() {
            let id = arg.get_id().to_string();
            let help = arg.get_help().map(ToString::to_string);
            let help = help.unwrap_or_else(|| panic!("{command} {id} has no help"));
            assert!(
                !help.contains('\n') && help.len() <= 110,
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
    fn the_old_generation_time_flag_still_works() {
        assert_eq!(parse("--tmrca-gen-years 29").unwrap().gen_years, 29.0);
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
        // The bootstrap's settings go with it.
        let args = parse("--bootstrap-seed 7 --bootstrap-block-size 1000").unwrap();
        assert_eq!(args.to_args(), ["in", "out", "1"]);
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
        // An option's value is not a subcommand.
        let value = err("psmc-rs --qc-json qc in.vcf");
        assert!(!value.contains("tip: a subcommand"), "{value}");
        // Nor is an argument after `--`.
        for argv in ["psmc-rs in.vcf -- qc", "psmc-rs -- convert out.json 3x"] {
            assert!(!err(argv).contains("tip: a subcommand"), "{argv}");
        }
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
}
