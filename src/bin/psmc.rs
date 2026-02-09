use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use psmc_rs::PsmcModel;
use psmc_rs::hmm::write_tmrca_posterior_tsv;
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, bounds_from_config, em_train};
use psmc_rs::progress;
use psmc_rs::report::{default_html_path, write_html_report};

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
        default_value_t = 1e-3,
        help = "M-step smoothness penalty on log-lambda differences"
    )]
    smooth_lambda: f64,
    #[arg(long)]
    no_progress: bool,
    #[arg(
        long,
        help = "Write per-bin TMRCA posterior TSV and add TMRCA charts into HTML report"
    )]
    tmrca_out: Option<PathBuf>,
    #[arg(
        long,
        default_value_t = 25.0,
        help = "Generation time (years) used for TMRCA scaling"
    )]
    tmrca_gen_years: f64,
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
    let pattern = cli.pattern.clone();

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
        // Match C initialization: theta0 = -log(1 - n_e / L_e).
        if frac >= 1.0 {
            f64::INFINITY
        } else {
            -(1.0 - frac).ln()
        }
    });
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);

    let model = PsmcModel::new(cli.t_max, cli.n_steps, theta0, rho0, cli.mu, pattern)?;

    let mut model = model;
    if cli.n_iter > 0 {
        let mut config = MStepConfig::default();
        config.max_iters = cli.mstep_iters.unwrap_or(100);
        config.lambda = cli.smooth_lambda;
        if cli.no_progress {
            config.progress = false;
        }
        let history = em_train(&mut model, &obs.rows, &obs.row_starts, cli.n_iter, &config)?;
        if let Some((before, _)) = history.loglike.last() {
            println!("Last EM loglike: {}", before);
        }
        warn_if_near_bounds(&model, &config)?;
    }

    let report_bin_size = match cli.input_format {
        InputFormat::Psmcfa => 100.0,
        InputFormat::Mhs => cli.mhs_bin_size as f64,
    };
    let input_format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

    // Ensure matrices are synchronized with final optimized parameters.
    model.param_recalculate()?;

    let mut tmrca_report_data = None;
    if let Some(tmrca_out) = &cli.tmrca_out {
        let t = model.compute_t(0.1);
        let n0 = model.theta / (4.0 * model.mu * report_bin_size);
        let mut tmrca_years = Vec::with_capacity(model.n_steps + 1);
        for k in 0..=model.n_steps {
            tmrca_years.push(t[k] * 2.0 * n0 * cli.tmrca_gen_years);
        }
        let tmrca = write_tmrca_posterior_tsv(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            &tmrca_years,
            tmrca_out,
            !cli.no_progress,
            20_000,
        )?;
        println!("TMRCA posterior: {}", tmrca_out.display());
        tmrca_report_data = Some(tmrca);
    }

    model.save_params(&cli.output_file)?;

    let report_html = default_html_path(&cli.output_file);
    write_html_report(
        &model,
        &cli.input_file,
        &cli.output_file,
        &report_html,
        cli.n_iter,
        input_format_name,
        report_bin_size,
        tmrca_report_data.as_ref(),
    )?;
    println!("HTML report: {}", report_html.display());
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
