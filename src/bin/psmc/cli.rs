//! Command-line arguments.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum InputFormat {
    Psmcfa,
    Mhs,
    Vcf,
}

impl InputFormat {
    /// The name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            InputFormat::Psmcfa => "psmcfa",
            InputFormat::Mhs => "mhs",
            InputFormat::Vcf => "vcf",
        }
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

    #[arg(
        long,
        help = "Number of worker threads for E-step (multi-sequence) and bootstrap"
    )]
    pub threads: Option<usize>,

    #[arg(long, value_enum, default_value_t = InputFormat::Psmcfa)]
    pub input_format: InputFormat,

    #[arg(long, default_value_t = 100)]
    pub mhs_bin_size: usize,

    #[arg(
        long,
        help = "Sample name in VCF (required if VCF has multiple samples)"
    )]
    pub vcf_sample: Option<String>,

    #[arg(
        long = "vcf-mask",
        help = "Callable mask BED path for VCF->MHS semantics (can be repeated)"
    )]
    pub vcf_mask: Vec<PathBuf>,

    #[arg(
        long = "vcf-negative-mask",
        help = "Negative mask BED path for VCF->MHS semantics (can be repeated)"
    )]
    pub vcf_negative_mask: Vec<PathBuf>,

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

    #[arg(
        long,
        help = "Split long inputs into chunks for memory control; chunks remain connected in one HMM chain"
    )]
    pub batch_size: Option<usize>,

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
        args.push("--input-format".to_string());
        args.push(self.input_format.name().to_string());
        if matches!(self.input_format, InputFormat::Mhs | InputFormat::Vcf) {
            args.push("--mhs-bin-size".to_string());
            args.push(self.mhs_bin_size.to_string());
        }
        if let Some(sample) = &self.vcf_sample {
            args.push("--vcf-sample".to_string());
            args.push(sample.clone());
        }
        for p in &self.vcf_mask {
            args.push("--vcf-mask".to_string());
            args.push(p.display().to_string());
        }
        for p in &self.vcf_negative_mask {
            args.push("--vcf-negative-mask".to_string());
            args.push(p.display().to_string());
        }
        if let Some(v) = &self.pattern {
            args.push("--pattern".to_string());
            args.push(v.clone());
        }
        if let Some(v) = self.batch_size {
            args.push("--batch-size".to_string());
            args.push(v.to_string());
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
