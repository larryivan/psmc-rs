use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::PathBuf;

use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::PsmcModel;

#[derive(Parser, Debug)]
#[command(name = "psmc")]
#[command(about = "PSMC in Rust (work-in-progress)", long_about = None)]
struct Cli {
    input_file: PathBuf,
    output_file: PathBuf,
    n_iter: usize,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.n_iter > 0 {
        bail!("EM training not implemented yet; set n_iter=0 for now");
    }

    let xs = read_psmcfa(&cli.input_file, cli.batch_size)
        .with_context(|| "failed to read psmcfa input")?;

    let total = (xs.shape()[0] * xs.shape()[1]) as f64;
    let mut count_k = 0usize;
    for v in xs.iter() {
        if *v == 1 {
            count_k += 1;
        }
    }

    let theta0 = cli
        .theta0
        .unwrap_or_else(|| (count_k as f64) / total);
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);

    let model = PsmcModel::new(
        cli.t_max,
        cli.n_steps,
        theta0,
        rho0,
        cli.mu,
        cli.pattern,
    )?;

    model.save_params(&cli.output_file)?;
    Ok(())
}
