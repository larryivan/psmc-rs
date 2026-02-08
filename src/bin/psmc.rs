use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use psmc_rs::PsmcModel;
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, bounds_from_config, em_train};
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
    #[arg(long)]
    threads: Option<usize>,
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
    #[arg(
        long,
        help = "Split long inputs into chunks for memory control; chunks remain connected in one HMM chain"
    )]
    batch_size: Option<usize>,
    #[arg(long, default_value_t = 2.5e-8)]
    mu: f64,
    #[arg(long)]
    mstep_iters: Option<usize>,
    #[arg(
        long,
        default_value_t = 1e-2,
        help = "M-step smoothness penalty on log-lambda differences (set to 0 for C-like behavior)"
    )]
    smooth_lambda: f64,
    #[arg(long, default_value_t = 0)]
    loglike_after_every: usize,
    #[arg(long)]
    no_loglike_after_last: bool,
    #[arg(
        long,
        help = "Use C-like defaults: pattern=4+25*2+4+6 when missing, theta0 init=-log(1-k/L), lambda=0, mstep_iters=100"
    )]
    compat_c: bool,
    #[arg(long)]
    theta_lo: Option<f64>,
    #[arg(long)]
    theta_hi: Option<f64>,
    #[arg(long)]
    rho_lo: Option<f64>,
    #[arg(long)]
    rho_hi: Option<f64>,
    #[arg(long)]
    tmax_lo: Option<f64>,
    #[arg(long)]
    tmax_hi: Option<f64>,
    #[arg(long)]
    lam_lo: Option<f64>,
    #[arg(long)]
    lam_hi: Option<f64>,
    #[arg(long)]
    no_progress: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(n_threads) = cli.threads {
        if n_threads == 0 {
            bail!("--threads must be >= 1");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .map_err(|e| anyhow!("failed to configure Rayon global thread pool: {e}"))?;
    }
    let mut pattern = cli.pattern.clone();
    if cli.compat_c && pattern.is_none() {
        pattern = Some("4+25*2+4+6".to_string());
    }

    let format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

    let obs = if cli.no_progress {
        match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&cli.input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&cli.input_file, cli.batch_size, cli.mhs_bin_size),
        }
        .with_context(|| format!("failed to read {format_name} input"))?
    } else {
        let pb = progress::spinner("IO", &format!("Reading {format_name}"));
        let obs = match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&cli.input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&cli.input_file, cli.batch_size, cli.mhs_bin_size),
        }
        .with_context(|| format!("failed to read {format_name} input"))?;
        pb.finish_with_message(format!("Reading {format_name} done"));
        obs
    };

    let mut count_k = 0usize;
    let mut total_observed = 0usize;
    for row in &obs.rows {
        for v in row {
            if *v == 1 {
                count_k += 1;
            }
            if *v != 2 {
                total_observed += 1;
            }
        }
    }
    if total_observed == 0 {
        bail!("no observed (non-N) sites found in input");
    }

    let theta0 = cli.theta0.unwrap_or_else(|| {
        let frac = (count_k as f64) / (total_observed as f64);
        if cli.compat_c {
            // Match C initialization: theta0 = -log(1 - n_e / L_e).
            if frac >= 1.0 {
                f64::INFINITY
            } else {
                -(1.0 - frac).ln()
            }
        } else {
            frac
        }
    });
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);

    let model = PsmcModel::new(cli.t_max, cli.n_steps, theta0, rho0, cli.mu, pattern)?;

    let mut model = model;
    if cli.n_iter > 0 {
        let mut config = MStepConfig::default();
        config.max_iters = cli
            .mstep_iters
            .unwrap_or(if cli.compat_c { 100 } else { 30 });
        config.lambda = cli.smooth_lambda;
        config.loglike_after_every = cli.loglike_after_every;
        config.loglike_after_last = !cli.no_loglike_after_last;
        if let Some(v) = cli.theta_lo {
            config.theta_lo = v;
        }
        if let Some(v) = cli.theta_hi {
            config.theta_hi = v;
        }
        if let Some(v) = cli.rho_lo {
            config.rho_lo = v;
        }
        if let Some(v) = cli.rho_hi {
            config.rho_hi = v;
        }
        if let Some(v) = cli.tmax_lo {
            config.tmax_lo = v;
        }
        if let Some(v) = cli.tmax_hi {
            config.tmax_hi = v;
        }
        if let Some(v) = cli.lam_lo {
            config.lam_lo = v;
        }
        if let Some(v) = cli.lam_hi {
            config.lam_hi = v;
        }
        if cli.compat_c {
            // C-like mode: disable smooth penalty and keep C-like defaults.
            config.lambda = 0.0;
        }
        if cli.no_progress {
            config.progress = false;
        }
        let history = em_train(&mut model, &obs.rows, &obs.row_starts, cli.n_iter, &config)?;
        if let Some((before, after)) = history.loglike.last() {
            if (before - after).abs() <= 1e-12 {
                println!("Last EM loglike: {} (after-check skipped)", before);
            } else {
                println!("Last EM loglike: {} -> {}", before, after);
            }
        }
        warn_if_near_bounds(&model, &config)?;
    }

    model.save_params(&cli.output_file)?;
    Ok(())
}

fn near(v: f64, lo: f64, hi: f64) -> (bool, bool) {
    let span = (hi - lo).abs().max(1.0);
    let tol = span * 1e-6;
    ((v - lo).abs() <= tol, (v - hi).abs() <= tol)
}

fn warn_if_near_bounds(model: &PsmcModel, cfg: &MStepConfig) -> Result<()> {
    let bounds = bounds_from_config(model, cfg);
    let mut vals: Vec<(String, f64)> = Vec::with_capacity(3 + model.lam.len());
    vals.push(("theta".to_string(), model.theta));
    vals.push(("rho".to_string(), model.rho));
    vals.push(("t_max".to_string(), model.t_max));
    for (i, v) in model.lam.iter().enumerate() {
        vals.push((format!("lam[{i}]"), *v));
    }

    let mut near_lo = 0usize;
    let mut near_hi = 0usize;
    let mut examples: Vec<String> = Vec::new();
    for (i, (name, v)) in vals.iter().enumerate() {
        let b = bounds[i];
        let (is_lo, is_hi) = near(*v, b.lo, b.hi);
        if is_lo {
            near_lo += 1;
            if examples.len() < 6 {
                examples.push(format!("{name}≈lo({})", b.lo));
            }
        }
        if is_hi {
            near_hi += 1;
            if examples.len() < 6 {
                examples.push(format!("{name}≈hi({})", b.hi));
            }
        }
    }
    if near_lo + near_hi > 0 {
        eprintln!(
            "Warning: {} params are on/near optimization bounds (lo={}, hi={}). Examples: {}. Consider wider bounds.",
            near_lo + near_hi,
            near_lo,
            near_hi,
            examples.join(", ")
        );
    }
    Ok(())
}
