use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use psmc_rs::PsmcModel;
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::progress;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum InputFormat {
    Psmcfa,
    Mhs,
}

#[derive(Parser, Debug)]
#[command(name = "psmc")]
#[command(about = "PSMC in Rust (work-in-progress)", long_about = None)]
struct Cli {
    input_file: PathBuf,
    output_file: PathBuf,
    n_iter: usize,
    #[arg(long, value_enum, default_value_t = InputFormat::Psmcfa)]
    input_format: InputFormat,
    #[arg(long, default_value_t = 100)]
    mhs_bin_size: usize,
    #[arg(long, default_value_t = 15.0)]
    t_max: f64,
    #[arg(long, default_value_t = 64)]
    n_steps: usize,
    #[arg(long)]
    theta0: Option<f64>,
    #[arg(long)]
    rho0: Option<f64>,
    #[arg(long)]
    pattern: Option<String>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long, default_value_t = 2.5e-8)]
    mu: f64,
    #[arg(long, default_value_t = 30)]
    mstep_iters: usize,
    #[arg(long, default_value_t = 1e-2)]
    smooth_lambda: f64,
    #[arg(long)]
    no_progress: bool,
    #[arg(long)]
    no_ad: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

    let xs = if cli.no_progress {
        match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&cli.input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&cli.input_file, cli.batch_size, cli.mhs_bin_size),
        }
        .with_context(|| format!("failed to read {format_name} input"))?
    } else {
        let pb = progress::spinner("IO", &format!("Reading {format_name}"));
        let xs = match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&cli.input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&cli.input_file, cli.batch_size, cli.mhs_bin_size),
        }
        .with_context(|| format!("failed to read {format_name} input"))?;
        pb.finish_with_message(format!("Reading {format_name} done"));
        xs
    };

    let mut count_k = 0usize;
    let mut total_observed = 0usize;
    for v in xs.iter() {
        if *v == 1 {
            count_k += 1;
        }
        if *v != 2 {
            total_observed += 1;
        }
    }
    if total_observed == 0 {
        bail!("no observed (non-N) sites found in input");
    }

    let theta0 = cli
        .theta0
        .unwrap_or_else(|| (count_k as f64) / (total_observed as f64));
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);

    let model = PsmcModel::new(cli.t_max, cli.n_steps, theta0, rho0, cli.mu, cli.pattern)?;

    let mut model = model;
    if cli.n_iter > 0 {
        let mut config = MStepConfig::default();
        config.max_iters = cli.mstep_iters;
        config.lambda = cli.smooth_lambda;
        if cli.no_ad {
            config.use_ad = false;
        }
        if cli.no_progress {
            config.progress = false;
            config.progress_grads = false;
        }
        let history = em_train(&mut model, &xs, cli.n_iter, &config)?;
        if let Some((before, after)) = history.loglike.last() {
            println!("Last EM loglike: {} -> {}", before, after);
        }
    }

    model.save_params(&cli.output_file)?;
    Ok(())
}
