//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use psmc_rs::bootstrap::{
    default_bootstrap_dir, new_rng, replicate_json_path, rows_to_sequences,
    sample_sequences_block_bootstrap, sequences_to_observations, summarize_bootstrap_models,
    write_bootstrap_summary_json, write_bootstrap_summary_tsv,
};
use psmc_rs::em::{EmOptions, EmProgressEvent, em_train, em_train_with_options};
use psmc_rs::hmm::{Cancelled, EStepPhase, write_tmrca_posterior_tsv};
use psmc_rs::opt::{MStepConfig, bounds_from_config};
use psmc_rs::report::{BootstrapReport, ReportData, default_html_path, write_html_report};
use psmc_rs::{PsmcModel, initial_theta};

use crate::cli::{Cli, model_layout};
use crate::dataset::{describe, ensure_usable, load_input, resolve_input_path};

/// A stage of the pipeline with a progress count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Reading the input (bytes).
    Io,
    /// E-step forward pass (sites).
    EForward,
    /// E-step backward pass (sites).
    EBackward,
    /// M-step (optimiser iterations).
    MStep,
    /// Bootstrap (replicates).
    Bootstrap,
}

/// What the pipeline reports while it runs.
#[derive(Debug, Clone)]
pub enum Event {
    /// A line of information.
    Info(String),
    /// A warning.
    Warning(String),
    /// A result: the final log-likelihood or an output written.
    Output(String),
    /// EM iteration `iter` of `total` starts.
    Iteration { iter: usize, total: usize },
    /// Progress of a stage; `iter` is the EM iteration of E- and M-steps.
    Progress {
        stage: Stage,
        done: u64,
        total: u64,
        iter: Option<(usize, usize)>,
    },
}

/// Where the pipeline sends its events; called from worker threads too.
pub type Sink<'a> = &'a (dyn Fn(Event) + Sync);

/// The outputs of a finished run.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub output_json: PathBuf,
    pub html_report: PathBuf,
    pub tmrca_out: Option<PathBuf>,
    pub bootstrap_dir: Option<PathBuf>,
    pub final_loglike: Option<f64>,
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }
    Ok(())
}

fn bootstrap_replicate_seed(base_seed: u64, rep_idx: usize) -> u64 {
    // SplitMix-like deterministic stream splitting per replicate.
    base_seed.wrapping_add(
        (rep_idx as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
}

fn configure_threads(n_threads: Option<usize>) -> Result<()> {
    if let Some(n_threads) = n_threads {
        if n_threads == 0 {
            bail!("--threads must be >= 1");
        }
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
        {
            let msg = e.to_string();
            if !msg.contains("initialized") {
                return Err(anyhow!(
                    "failed to configure Rayon global thread pool: {msg}"
                ));
            }
        }
    }
    Ok(())
}

/// Run inference as `cli` describes it. `cancel` stops the run at the next
/// check (between stages, and inside EM); the error is then [`Cancelled`].
/// `progress_bars` lets the TMRCA decoding draw its own bar on stderr.
pub fn run_inference(
    cli: &Cli,
    cancel: &AtomicBool,
    sink: Sink,
    progress_bars: bool,
) -> Result<RunSummary> {
    let raw_input = cli
        .input_file
        .clone()
        .ok_or_else(|| anyhow!("missing INPUT_FILE"))?;
    let output_file = cli
        .output_file
        .clone()
        .ok_or_else(|| anyhow!("missing OUTPUT_FILE"))?;
    let n_iter = cli.n_iter.ok_or_else(|| anyhow!("missing N_ITER"))?;

    check_cancel(cancel)?;
    let input_file = resolve_input_path(&raw_input, cli.input.input_format)?;
    if input_file != raw_input {
        sink(Event::Info(format!(
            "Input is a directory, selected file: {}",
            input_file.display()
        )));
    }

    if cli.bootstrap > 0 && cli.bootstrap_block_size == 0 {
        bail!("--bootstrap-block-size must be > 0 when --bootstrap is enabled");
    }
    let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
    let bootstrap_iters = cli.bootstrap_iters.unwrap_or(n_iter);
    if cli.bootstrap > 0 && bootstrap_iters == 0 {
        bail!("bootstrap requires EM iterations; set N_ITER > 0 or use --bootstrap-iters");
    }

    configure_threads(cli.threads)?;

    let io_total = fs::metadata(&input_file).map(|m| m.len()).unwrap_or(1);
    let io_progress = |done| Event::Progress {
        stage: Stage::Io,
        done,
        total: io_total,
        iter: None,
    };
    sink(io_progress(0));
    let data = load_input(&input_file, &cli.input.to_options())?;
    sink(io_progress(io_total));
    for line in describe(&data) {
        sink(Event::Info(line));
    }
    for line in data.warnings() {
        sink(Event::Warning(line));
    }
    ensure_usable(&data)?;
    let obs = &data.obs;
    let format_name = data.source.format().name();

    let theta0 = match cli.theta0 {
        Some(v) => v,
        None => initial_theta(&obs.rows)?,
    };
    let rho0 = cli.rho0.unwrap_or(theta0 / cli.tr_ratio);
    let mut model = PsmcModel::new(cli.t_max, n_steps, theta0, rho0, cli.mu, pattern)?;
    sink(Event::Info(format!(
        "Model: pattern {}, {} intervals, {} free lambdas; start theta0 {theta0:.6}, rho0 \
         {rho0:.6} (theta0/rho0 {}), t_max {}",
        model.pattern.as_deref().unwrap_or("none"),
        model.n_steps + 1,
        model.lam.len(),
        theta0 / rho0,
        model.t_max
    )));

    let mstep_config = MStepConfig {
        max_iters: cli.mstep_iters,
        lambda: cli.smooth_lambda,
        progress: false,
        ..MStepConfig::default()
    };

    let mut final_loglike = None;
    if n_iter > 0 {
        let mut on_progress = |ev: EmProgressEvent| match ev {
            EmProgressEvent::IterStart { iter, total_iters } => sink(Event::Iteration {
                iter,
                total: total_iters,
            }),
            EmProgressEvent::EStep {
                iter,
                total_iters,
                phase,
                done,
                total,
            } => sink(Event::Progress {
                stage: match phase {
                    EStepPhase::Forward => Stage::EForward,
                    EStepPhase::Backward => Stage::EBackward,
                },
                done,
                total,
                iter: Some((iter, total_iters)),
            }),
            EmProgressEvent::MStep {
                iter,
                total_iters,
                done,
                total,
            } => sink(Event::Progress {
                stage: Stage::MStep,
                done: done as u64,
                total: total as u64,
                iter: Some((iter, total_iters)),
            }),
        };
        let history = em_train_with_options(
            &mut model,
            &obs.rows,
            &obs.row_starts,
            n_iter,
            &mstep_config,
            EmOptions {
                on_progress: Some(&mut on_progress),
                cancel: Some(cancel),
                thread_pool: None,
            },
        )?;
        final_loglike = history.final_loglike();
        if let Some(ll) = final_loglike {
            sink(Event::Output(format!("Last EM loglike: {ll}")));
        }
        if let Some(msg) = near_bounds_warning(&model, &mstep_config) {
            sink(Event::Warning(msg));
        }
    }

    check_cancel(cancel)?;
    let report_bin_size = cli.input.bin_size as f64;
    model.param_recalculate()?;

    let mut bootstrap_report_data = None;
    let mut bootstrap_dir_out = None;
    if cli.bootstrap > 0 {
        let bootstrap_dir = cli
            .bootstrap_dir
            .clone()
            .unwrap_or_else(|| default_bootstrap_dir(&output_file));
        fs::create_dir_all(&bootstrap_dir).with_context(|| {
            format!(
                "failed to create bootstrap output directory {}",
                bootstrap_dir.display()
            )
        })?;
        let boot_progress = |done: usize| Event::Progress {
            stage: Stage::Bootstrap,
            done: done as u64,
            total: cli.bootstrap as u64,
            iter: None,
        };
        sink(boot_progress(0));
        sink(Event::Info(format!(
            "Bootstrap: running {} replicates in parallel",
            cli.bootstrap
        )));

        let seqs = rows_to_sequences(&obs.rows, &obs.row_starts)?;
        let boot_models = (0..cli.bootstrap)
            .into_par_iter()
            .map(|rep| -> Result<PsmcModel> {
                check_cancel(cancel)?;
                let mut rng = new_rng(bootstrap_replicate_seed(cli.bootstrap_seed, rep));
                let sampled =
                    sample_sequences_block_bootstrap(&seqs, cli.bootstrap_block_size, &mut rng)?;
                let obs_rep = sequences_to_observations(&sampled, cli.input.batch_size)?;
                let mut model_rep = model.clone();
                em_train(
                    &mut model_rep,
                    &obs_rep.rows,
                    &obs_rep.row_starts,
                    bootstrap_iters,
                    &mstep_config,
                )?;
                model_rep.param_recalculate()?;
                model_rep.save_params(&replicate_json_path(&bootstrap_dir, rep + 1))?;
                Ok(model_rep)
            })
            .collect::<Result<Vec<_>>>()?;
        check_cancel(cancel)?;
        sink(boot_progress(cli.bootstrap));

        let bootstrap_summary = summarize_bootstrap_models(
            &model,
            &boot_models,
            report_bin_size,
            25.0,
            cli.bootstrap_block_size,
            cli.bootstrap_seed,
        )?;
        let summary_tsv = bootstrap_dir.join("summary.tsv");
        let summary_json = bootstrap_dir.join("summary.json");
        write_bootstrap_summary_tsv(&summary_tsv, &bootstrap_summary)?;
        write_bootstrap_summary_json(&summary_json, &bootstrap_summary)?;
        sink(Event::Output(format!(
            "Bootstrap dir: {}",
            bootstrap_dir.display()
        )));
        sink(Event::Output(format!(
            "Bootstrap summary TSV: {}",
            summary_tsv.display()
        )));
        sink(Event::Output(format!(
            "Bootstrap summary JSON: {}",
            summary_json.display()
        )));
        bootstrap_report_data = Some(bootstrap_summary);
        bootstrap_dir_out = Some(bootstrap_dir);
    }

    check_cancel(cancel)?;
    let mut tmrca_report_data = None;
    if let Some(tmrca_out) = &cli.tmrca_out {
        let t = model.compute_t(0.1);
        let n0 = model.n0_for_bin(report_bin_size);
        let tmrca_years: Vec<f64> = t[..=model.n_steps]
            .iter()
            .map(|tk| tk * 2.0 * n0 * cli.tmrca_gen_years)
            .collect();
        let tmrca = write_tmrca_posterior_tsv(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            &tmrca_years,
            tmrca_out,
            progress_bars,
            20_000,
        )?;
        sink(Event::Output(format!(
            "TMRCA posterior: {}",
            tmrca_out.display()
        )));
        tmrca_report_data = Some(tmrca);
    }

    model.save_params(&output_file)?;

    let report_html = default_html_path(&output_file);
    let report = ReportData {
        model: &model,
        bin_size: report_bin_size,
        gen_years: 25.0,
        input_file: &input_file,
        output_json: &output_file,
        input: None,
        n_iter,
        final_loglike,
        settings: vec![("Input format".to_string(), format_name.to_string())],
        qc: None,
        bootstrap: bootstrap_report_data
            .as_ref()
            .map(|summary| BootstrapReport {
                summary,
                em_iters: bootstrap_iters,
                trunks: None,
            }),
        tmrca: tmrca_report_data.as_ref(),
    };
    write_html_report(&report_html, &report)?;
    sink(Event::Output(format!(
        "HTML report: {}",
        report_html.display()
    )));

    Ok(RunSummary {
        output_json: output_file,
        html_report: report_html,
        tmrca_out: cli.tmrca_out.clone(),
        bootstrap_dir: bootstrap_dir_out,
        final_loglike,
    })
}

fn near(v: f64, lo: f64, hi: f64) -> (bool, bool) {
    // Use a very tight tolerance so "near bound" reflects a true optimizer
    // saturation, not ordinary values in wide ranges (e.g. lam in [1e-6, 1e6]).
    let span = (hi - lo).abs();
    let scale = v.abs().max(lo.abs()).max(hi.abs()).max(1.0);
    let tol = (span * 1e-12).max(scale * 1e-12).max(f64::EPSILON * 16.0);
    ((v - lo).abs() <= tol, (v - hi).abs() <= tol)
}

/// A warning when fitted parameters sit on the M-step's bounds.
fn near_bounds_warning(model: &PsmcModel, cfg: &MStepConfig) -> Option<String> {
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
    for ((name, v), b) in vals.iter().zip(&bounds) {
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
    (near_lo + near_hi > 0).then(|| {
        format!(
            "{} params are on/near optimization bounds (lo={}, hi={}). Examples: {}. Consider wider bounds.",
            near_lo + near_hi,
            near_lo,
            near_hi,
            examples.join(", ")
        )
    })
}
