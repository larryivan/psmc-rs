//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

use anyhow::{Context, Result, anyhow, bail};
use rayon::ThreadPool;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, BootstrapSummary, bootstrap_trunk_count,
    default_bootstrap_dir, rows_to_sequences, run_bootstrap_with_options,
    summarize_bootstrap_models,
};
use psmc_rs::em::{EmHistory, EmOptions, EmProgressEvent, em_train_with_options};
use psmc_rs::hmm::{Cancelled, EStepPhase, write_tmrca_posterior_tsv};
use psmc_rs::io::Observations;
use psmc_rs::io::input::Dataset;
use psmc_rs::io::params::{InputSummary, load_params};
use psmc_rs::opt::{MStepConfig, bounds_from_config};
use psmc_rs::psmc_out::c_pattern;
use psmc_rs::qc::{QcReport, Severity, render_text};
use psmc_rs::report::{BootstrapReport, ReportData, default_html_path, write_html_report};
use psmc_rs::{PsmcModel, initial_theta};

use crate::cli::{InferArgs, model_layout};
use crate::dataset::{
    describe, ensure_usable, load_input, loader_warnings, qc_report, resolve_input_path,
};
use crate::output::{
    FitRecord, check_destinations, is_stdout, save_fit, write_bootstrap_summary, write_main_psmc,
    write_qc_json, write_replicate_psmc,
};

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
    /// The input QC report as text, and its warnings one per line (shown
    /// instead of the text when messages are hidden).
    Qc { text: String, warnings: Vec<String> },
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
    /// The fitted parameters.
    pub output_json: PathBuf,
    /// The HTML report.
    pub html_report: PathBuf,
    /// The TMRCA posterior, if asked for.
    pub tmrca_out: Option<PathBuf>,
    /// The bootstrap files, if a bootstrap ran.
    pub bootstrap_dir: Option<PathBuf>,
    /// Log-likelihood of the fitted parameters (none after 0 iterations).
    pub final_loglike: Option<f64>,
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Cancelled.into());
    }
    Ok(())
}

/// Run inference as `cli` describes it, in a thread pool of its own with
/// `--threads` workers. `cancel` stops the run at the next check (between
/// stages, and inside EM and the bootstrap); the error is then
/// [`Cancelled`]. `progress_bars` lets the TMRCA decoding draw its own bar
/// on stderr.
pub fn run_inference(
    cli: &InferArgs,
    cancel: &AtomicBool,
    sink: Sink,
    progress_bars: bool,
) -> Result<RunSummary> {
    // A pool per run, rather than the global one, so that a later run (in
    // the TUI) can use another thread count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.unwrap_or(0))
        .build()
        .context("failed to start the worker threads")?;
    pool.install(|| infer(cli, cancel, sink, progress_bars, &pool))
}

/// The bootstrap a run asks for.
struct BootstrapPlan {
    replicates: usize,
    trunk_size: usize,
    seed: u64,
    em_iters: usize,
    dir: PathBuf,
}

impl BootstrapPlan {
    /// The bootstrap of `cli`, if any, checked before any work is done.
    fn of(cli: &InferArgs, n_iter: usize, output: &Path) -> Result<Option<Self>> {
        if cli.bootstrap == 0 {
            for (set, flag) in [
                (cli.bootstrap_iters.is_some(), "--bootstrap-iters"),
                (cli.bootstrap_dir.is_some(), "--bootstrap-dir"),
            ] {
                if set {
                    bail!("{flag} applies to a bootstrap; add --bootstrap N to run N replicates");
                }
            }
            return Ok(None);
        }
        let em_iters = cli.bootstrap_iters.unwrap_or(n_iter);
        if em_iters == 0 {
            bail!(
                "the bootstrap replicates would run 0 EM iterations; set N_ITER above 0 or give \
                 --bootstrap-iters"
            );
        }
        Ok(Some(Self {
            replicates: cli.bootstrap,
            trunk_size: cli.bootstrap_block_size,
            seed: cli.bootstrap_seed,
            em_iters,
            dir: cli
                .bootstrap_dir
                .clone()
                .unwrap_or_else(|| default_bootstrap_dir(output)),
        }))
    }
}

/// Trunk counts below this make coarse bootstrap bands.
const FEW_TRUNKS: usize = 20;

/// [`run_inference`] inside its pool.
fn infer(
    cli: &InferArgs,
    cancel: &AtomicBool,
    sink: Sink,
    progress_bars: bool,
    pool: &ThreadPool,
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

    // Settings that can be checked without the data fail first.
    let saved = match &cli.init_params {
        Some(path) => Some(saved_start(path, cli)?),
        None => None,
    };
    let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
    let bootstrap = BootstrapPlan::of(cli, n_iter, &output_file)?;

    check_cancel(cancel)?;
    let input_file = resolve_input_path(&raw_input, cli.input.input_format)?;
    if input_file != raw_input {
        sink(Event::Info(format!(
            "Input is a directory, selected file: {}",
            input_file.display()
        )));
    }
    let report_html = default_html_path(&output_file);
    check_outputs(
        cli,
        &input_file,
        &output_file,
        &report_html,
        bootstrap.as_ref(),
    )?;
    let data = read_input(&input_file, cli, sink)?;
    let obs = &data.obs;

    // Input QC, printed before the fit. A QC error is only reported, so
    // that it never stops a fit.
    let free_lambdas = match &saved {
        Some(model) => model.lam.len(),
        None => PsmcModel::new(15.0, n_steps, 1.0, 1.0, 1.0, pattern.clone())?
            .lam
            .len(),
    };
    let trunk = bootstrap.as_ref().map_or(0, |plan| plan.trunk_size);
    let qc = if obs.rows.is_empty() {
        None
    } else {
        match qc_report(&data, cli.input.bin_size, trunk, Some(free_lambdas)) {
            Ok(report) => Some(report),
            Err(e) => {
                sink(Event::Warning(format!("input QC failed: {e:#}")));
                None
            }
        }
    };
    for line in loader_warnings(&data, qc.as_ref()) {
        sink(Event::Warning(line));
    }
    if let Some(report) = &qc {
        sink(qc_event(report));
        if let Some(path) = &cli.qc_json {
            write_qc_json(path, report)?;
            if !is_stdout(path) {
                sink(Event::Output(format!("QC report: {}", path.display())));
            }
        }
    }
    ensure_usable(&data)?;

    // A bootstrap that cannot vary fails before the main fit, not after.
    let boot_seqs = match &bootstrap {
        Some(plan) => {
            let seqs = rows_to_sequences(&obs.rows, &obs.row_starts)?;
            let trunks = bootstrap_trunk_count(&seqs, plan.trunk_size)
                .context("--bootstrap-block-size is too large for this input")?;
            sink(Event::Info(format!(
                "Bootstrap: {} replicates of {} EM iterations, resampling {trunks} trunks of {} \
                 bins (seed {})",
                plan.replicates, plan.em_iters, plan.trunk_size, plan.seed
            )));
            if let Some(path) = &cli.init_params {
                sink(Event::Warning(format!(
                    "every bootstrap replicate starts from the parameters in {}, as the main fit \
                     does (and C psmc -i -b); they were fitted to the whole input, so replicates \
                     that run few EM iterations ({} here) stay close to them and the bands \
                     understate the uncertainty further. For bands of the whole estimator, \
                     bootstrap a fit from a fresh start",
                    path.display(),
                    plan.em_iters
                )));
            }
            if trunks < FEW_TRUNKS {
                sink(Event::Warning(format!(
                    "the input forms only {trunks} bootstrap trunks of {} bins, so replicates \
                     differ only in which of them they draw and the bands are coarse; a \
                     smaller --bootstrap-block-size gives more trunks",
                    plan.trunk_size
                )));
            }
            Some((seqs, trunks))
        }
        None => None,
    };

    let mut model = match saved {
        Some(model) => {
            sink(Event::Info(format!(
                "Starting from the parameters in {}",
                cli.init_params
                    .as_deref()
                    .unwrap_or(Path::new(""))
                    .display()
            )));
            model
        }
        None => {
            let theta0 = match cli.theta0 {
                Some(v) => v,
                None => initial_theta(&obs.rows)?,
            };
            let rho0 = cli.rho0.unwrap_or(theta0 / cli.tr_ratio);
            PsmcModel::new(cli.t_max, n_steps, theta0, rho0, cli.mu, pattern)?
        }
    };
    let (theta0, rho0) = (model.theta, model.rho);
    let tr_ratio = theta0 / rho0;
    sink(Event::Info(format!(
        "Model: pattern {}, {} intervals, {} free lambdas; start theta0 {theta0:.6}, rho0 \
         {rho0:.6} (theta0/rho0 {}), t_max {}",
        model.pattern.as_deref().unwrap_or("none"),
        model.n_steps + 1,
        model.lam.len(),
        tr_ratio,
        model.t_max
    )));
    // The replicates start where the main fit started, as C psmc does: from
    // a fresh start at its t_max and theta0/rho0 (-b), or from the saved
    // parameters (-i with -b).
    let fresh_start = (model.t_max, tr_ratio);
    let saved_start = cli.init_params.is_some().then(|| model.clone());

    let mstep_config = MStepConfig {
        max_iters: cli.mstep_iters,
        lambda: cli.smooth_lambda,
        progress: false,
        ..MStepConfig::default()
    };
    let history = fit(&mut model, obs, n_iter, &mstep_config, cancel, sink, pool)?;
    let final_loglike = history.final_loglike();
    if let Some(ll) = final_loglike {
        sink(Event::Output(format!("Last EM loglike: {ll}")));
    }
    if n_iter > 0
        && let Some(msg) = near_bounds_warning(&model, &mstep_config)
    {
        sink(Event::Warning(msg));
    }
    check_cancel(cancel)?;

    // The main fit's outputs are written before the bootstrap, which can
    // take far longer.
    let input_summary = InputSummary::of(&data);
    let record = FitRecord {
        model: &model,
        history: &history,
        bin_size: cli.input.bin_size,
        gen_years: cli.gen_years,
        tr_ratio,
        smooth_lambda: cli.smooth_lambda,
        input: &input_summary,
        qc: qc.as_ref(),
    };
    save_fit(&output_file, &record)?;
    sink(Event::Output(format!(
        "Parameters: {}",
        output_file.display()
    )));
    if let Some(path) = &cli.psmc_out {
        write_main_psmc(path, &model, &history, &data)?;
        sink(Event::Output(format!("PSMC output: {}", path.display())));
    }

    let bin_size = cli.input.bin_size as f64;
    let mut boot_summary = None;
    if let (Some(plan), Some((seqs, trunks))) = (&bootstrap, &boot_seqs) {
        let cfg = BootstrapConfig {
            replicates: plan.replicates,
            trunk_size: plan.trunk_size,
            seed: plan.seed,
            em_iters: plan.em_iters,
            t_max: fresh_start.0,
            tr_ratio: fresh_start.1,
            batch_size: cli.input.batch_size,
            out_dir: Some(plan.dir.clone()),
        };
        let summary = bootstrap_stage(
            seqs,
            &model,
            &cfg,
            saved_start.as_ref(),
            &mstep_config,
            (bin_size, cli.gen_years),
            cancel,
            sink,
            pool,
        )?;
        boot_summary = Some((summary, *trunks));
    }

    check_cancel(cancel)?;
    let mut tmrca_report_data = None;
    if let Some(tmrca_out) = &cli.tmrca_out {
        // Each state's average coalescent time, the time of its emission
        // (C psmc's TC column).
        let tmrca_years = model.state_tmrca_years(bin_size, cli.gen_years);
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

    let mut settings = vec![
        ("Starting theta0/rho0".to_string(), format!("{tr_ratio}")),
        (
            "Smoothness penalty".to_string(),
            format!("{}", cli.smooth_lambda),
        ),
        (
            "M-step iterations (at most)".to_string(),
            cli.mstep_iters.to_string(),
        ),
    ];
    if let Some(path) = &cli.init_params {
        settings.push(("Started from".to_string(), path.display().to_string()));
    }
    let report = ReportData {
        model: &model,
        bin_size,
        gen_years: cli.gen_years,
        input_file: &input_file,
        output_json: &output_file,
        input: Some(&input_summary),
        n_iter,
        final_loglike,
        settings,
        qc: qc.as_ref(),
        bootstrap: boot_summary.as_ref().zip(bootstrap.as_ref()).map(
            |((summary, trunks), plan)| BootstrapReport {
                summary,
                em_iters: plan.em_iters,
                trunks: Some(*trunks),
                started_from: cli.init_params.as_deref(),
            },
        ),
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
        bootstrap_dir: bootstrap.map(|plan| plan.dir),
        final_loglike,
    })
}

/// Check the files and directories the run will write before reading the
/// input (see [`check_destinations`]).
fn check_outputs(
    cli: &InferArgs,
    input: &Path,
    output: &Path,
    report_html: &Path,
    bootstrap: Option<&BootstrapPlan>,
) -> Result<()> {
    if output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("html"))
    {
        bail!(
            "OUTPUT {} ends in .html, but the HTML report is written to OUTPUT with the \
             extension .html; name the parameters file .json",
            output.display()
        );
    }
    let mut files = vec![("OUTPUT", output), ("the HTML report", report_html)];
    for (what, path) in [
        ("--psmc-out", &cli.psmc_out),
        ("--tmrca-out", &cli.tmrca_out),
        ("--qc-json", &cli.qc_json),
    ] {
        if let Some(path) = path.as_deref().filter(|p| !is_stdout(p)) {
            files.push((what, path));
        }
    }
    let dir_name = match cli.bootstrap_dir {
        Some(_) => "--bootstrap-dir",
        None => "the bootstrap directory",
    };
    let dirs: Vec<(&str, &Path)> = bootstrap
        .map(|plan| (dir_name, plan.dir.as_path()))
        .into_iter()
        .collect();
    check_destinations(input, &files, &dirs)
}

/// The starting model of `--init-params`: the saved parameters, with the
/// layout checked against `--pattern` and `--n-steps` when they are given,
/// and `--mu` for scaling.
fn saved_start(path: &Path, cli: &InferArgs) -> Result<PsmcModel> {
    let what = || format!("--init-params {}", path.display());
    let saved = load_params(path).with_context(what)?;
    let mut model = PsmcModel::from_params_file(&saved).with_context(what)?;
    if cli.pattern.is_some() || cli.n_steps.is_some() {
        let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
        let wanted = PsmcModel::new(model.t_max, n_steps, 1.0, 1.0, model.mu, pattern)?;
        if (wanted.n_steps, c_pattern(&wanted)) != (model.n_steps, c_pattern(&model)) {
            bail!(
                "{} holds a model with pattern {} ({} intervals), but --pattern/--n-steps ask \
                 for {} ({} intervals); leave them out to continue the saved model",
                what(),
                c_pattern(&model),
                model.n_steps + 1,
                c_pattern(&wanted),
                wanted.n_steps + 1
            );
        }
    }
    model.mu = cli.mu;
    Ok(model)
}

/// The QC report as an event: its text, and its warnings for `--quiet`.
pub fn qc_event(report: &QcReport) -> Event {
    Event::Qc {
        text: render_text(report),
        warnings: report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .map(|f| format!("QC {}: {}", f.code.as_str(), f.message))
            .collect(),
    }
}

/// Read the input as `cli` says and report how it was read.
fn read_input(path: &Path, cli: &InferArgs, sink: Sink) -> Result<Dataset> {
    let io_total = fs::metadata(path).map(|m| m.len()).unwrap_or(1);
    let io_progress = |done| Event::Progress {
        stage: Stage::Io,
        done,
        total: io_total,
        iter: None,
    };
    sink(io_progress(0));
    let data = load_input(path, &cli.input.to_options())?;
    sink(io_progress(io_total));
    for line in describe(&data) {
        sink(Event::Info(line));
    }
    Ok(data)
}

/// `n_iter` EM iterations, reporting progress.
fn fit(
    model: &mut PsmcModel,
    obs: &Observations,
    n_iter: usize,
    mstep_config: &MStepConfig,
    cancel: &AtomicBool,
    sink: Sink,
    pool: &ThreadPool,
) -> Result<EmHistory> {
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
        model,
        &obs.rows,
        &obs.row_starts,
        n_iter,
        mstep_config,
        EmOptions {
            on_progress: Some(&mut on_progress),
            cancel: Some(cancel),
            thread_pool: Some(pool),
        },
    )?;
    model.param_recalculate()?;
    Ok(history)
}

/// Run the bootstrap and write its files: each replicate's JSON and
/// `.psmc` output, and the summary of the curves. Replicates start from
/// `start` when given, from the fresh start of `cfg` otherwise.
#[allow(clippy::too_many_arguments)]
fn bootstrap_stage(
    seqs: &[Vec<u8>],
    model: &PsmcModel,
    cfg: &BootstrapConfig,
    start: Option<&PsmcModel>,
    mstep_config: &MStepConfig,
    (bin_size, gen_years): (f64, f64),
    cancel: &AtomicBool,
    sink: Sink,
    pool: &ThreadPool,
) -> Result<BootstrapSummary> {
    let progress = |done: usize| Event::Progress {
        stage: Stage::Bootstrap,
        done: done as u64,
        total: cfg.replicates as u64,
        iter: None,
    };
    sink(progress(0));
    let on_done = |n: usize| sink(progress(n));
    let opts = BootstrapOptions {
        cancel: Some(cancel),
        thread_pool: Some(pool),
        on_done: Some(&on_done),
        start,
    };
    let replicates = run_bootstrap_with_options(seqs, model, cfg, mstep_config, opts)?;
    let dir = cfg
        .out_dir
        .as_deref()
        .expect("the CLI always writes the replicates");
    write_replicate_psmc(dir, &replicates)?;
    let models: Vec<PsmcModel> = replicates.into_iter().map(|r| r.model).collect();
    let summary = summarize_bootstrap_models(
        model,
        &models,
        bin_size,
        gen_years,
        cfg.trunk_size,
        cfg.seed,
    )?;
    let (summary_tsv, summary_json) = write_bootstrap_summary(dir, &summary)?;
    sink(Event::Output(format!(
        "Bootstrap replicates (replicate_NNN.json, replicate_NNN.psmc): {}",
        dir.display()
    )));
    sink(Event::Output(format!(
        "Bootstrap summary: {}, {}",
        summary_tsv.display(),
        summary_json.display()
    )));
    Ok(summary)
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::Mutex;

    /// A directory of its own under the system temp dir.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("psmc_rs_run_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cli(dir: &std::path::Path, extra: &[&str]) -> InferArgs {
        let input = dir.join("in.psmcfa");
        let bins = "TTTTKTTTTTTTNNTTKTTTTTTTTKTTTTTTTTTTKTTTTTTTTTKTT".repeat(4);
        fs::write(&input, format!(">a\n{bins}\n>b\n{bins}\n")).unwrap();
        let mut args = vec![
            "psmc-rs".to_string(),
            input.display().to_string(),
            dir.join("out.json").display().to_string(),
            "1".to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        crate::cli::Cli::try_parse_from(args).unwrap().infer
    }

    /// Each run uses a pool of its own size, also when a process (the TUI)
    /// runs several.
    #[test]
    fn each_run_uses_its_thread_count() {
        let dir = temp_dir("threads");
        for threads in [3, 2] {
            let sizes = Mutex::new(Vec::new());
            let sink = |ev: Event| {
                if let Event::Progress { .. } = ev {
                    sizes.lock().unwrap().push(rayon::current_num_threads());
                }
            };
            let cli = cli(&dir, &["--threads", &threads.to_string()]);
            run_inference(&cli, &AtomicBool::new(false), &sink, false).unwrap();
            let sizes = sizes.into_inner().unwrap();
            assert!(!sizes.is_empty());
            assert!(sizes.iter().all(|&n| n == threads), "{sizes:?}");
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parameters_on_their_bounds_are_reported() {
        let cfg = MStepConfig::default();
        let mut model =
            PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, Some("4+5*3+4".into())).unwrap();
        assert_eq!(near_bounds_warning(&model, &cfg), None);
        model.lam[2] = cfg.lam_hi;
        model.rho = cfg.rho_lo;
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        assert!(
            msg.starts_with("2 params are on/near optimization bounds (lo=1, hi=1)"),
            "{msg}"
        );
        assert!(msg.contains("lam[2]") && msg.contains("rho"), "{msg}");
    }

    #[test]
    fn a_set_cancel_flag_stops_the_run() {
        let dir = temp_dir("cancel");
        let cli = cli(&dir, &[]);
        let err = run_inference(&cli, &AtomicBool::new(true), &|_| {}, false).unwrap_err();
        assert!(err.is::<Cancelled>(), "{err:#}");
        assert!(!dir.join("out.json").exists());
        let _ = fs::remove_dir_all(dir);
    }
}
