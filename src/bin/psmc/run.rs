//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

use anyhow::{Context, Result, anyhow};
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
use psmc_rs::hmm::{
    Cancelled, EStepOptions, EStepPhase, EStepProgress, log_likelihood_with_options,
    write_tmrca_posterior_tsv,
};
use psmc_rs::io::Observations;
use psmc_rs::io::input::Dataset;
use psmc_rs::io::params::{InputSummary, load_params};
use psmc_rs::opt::{MStepConfig, bounds_from_config};
use psmc_rs::psmc_out::c_pattern;
use psmc_rs::qc::{FitStart, QcReport, Severity, render_text};
use psmc_rs::report::{
    BootstrapReport, ReportData, check_echarts_bundle, default_html_path, write_html_report,
};
use psmc_rs::{PsmcModel, initial_theta};

use crate::cli::{InferArgs, model_layout, usage_bail};
use crate::dataset::{
    describe, ensure_usable, ignored_options, layout_model, load_input, loader_warnings, qc_report,
    resolve_input_path,
};
use crate::output::{
    BOOTSTRAP_SUMMARY_FILES, FitRecord, check_destinations, is_stdout, save_fit,
    write_bootstrap_summary, write_main_psmc, write_qc_json, write_replicate_psmc,
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
    /// Log-likelihood of the written parameters: the fitted ones, or the
    /// starting ones after 0 iterations.
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
    /// The bootstrap of `cli`, if any, checked before any work is done;
    /// bootstrap options that cannot apply are usage errors.
    fn of(cli: &InferArgs, n_iter: usize, output: &Path) -> Result<Option<Self>> {
        if cli.bootstrap == 0 {
            for (set, flag) in [
                (cli.bootstrap_iters.is_some(), "--bootstrap-iters"),
                (cli.bootstrap_dir.is_some(), "--bootstrap-dir"),
            ] {
                if set {
                    usage_bail!(
                        "{flag} applies to a bootstrap; add --bootstrap N to run N replicates"
                    );
                }
            }
            return Ok(None);
        }
        let em_iters = cli.bootstrap_iters.unwrap_or(n_iter);
        if em_iters == 0 {
            usage_bail!(
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

/// [`mstep_memory_warning`] warns above this estimate, in bytes.
const MSTEP_MEMORY_WARN_BYTES: f64 = 1e9;

/// Peak memory of an M-step, in bytes, for a model of `states` hidden
/// states and `free_lambdas` free lambdas.
///
/// The M-step differentiates its objective in forward mode
/// (`psmc_rs::opt`): every intermediate value carries the derivatives with
/// respect to all `free_lambdas + 3` parameters, and the transition matrix
/// alone holds `states^2` of them, each a value and a vector of
/// derivatives (8 bytes per parameter, and 48 for the value, the vector
/// and its allocation). Measured peaks are 1.7 to 2.6 times that matrix
/// (2.5 at 401 states and 401 free lambdas, 1.3 GB), so the estimate takes
/// 2.5 times it. With one free lambda per interval this grows with the
/// cube of the number of intervals.
fn mstep_memory_bytes(states: usize, free_lambdas: usize) -> f64 {
    let params = (free_lambdas + 3) as f64;
    let states = states as f64;
    2.5 * states * states * (8.0 * params + 48.0)
}

/// A warning when the M-steps of this layout that may run at once need
/// more than [`MSTEP_MEMORY_WARN_BYTES`] together ([`mstep_memory_bytes`]
/// each), before the run spends time reading the input and then fails or
/// swaps in its first M-step. `at_once` is how many M-steps may run at
/// once: 1 for a fit, and for a bootstrap the replicates that run in
/// parallel, one per thread ([`bootstrap_m_steps_at_once`]).
fn mstep_memory_warning(states: usize, free_lambdas: usize, at_once: usize) -> Option<String> {
    let each = mstep_memory_bytes(states, free_lambdas);
    let bytes = each * at_once.max(1) as f64;
    if bytes <= MSTEP_MEMORY_WARN_BYTES {
        return None;
    }
    let why = format!(
        "carries the derivatives of all {} parameters through each of the {states} x {states} \
         transition probabilities, which grows with the cube of the intervals when each has \
         its own lambda. Group the intervals into fewer free parameters with --pattern (C \
         psmc's 4+5*3+4 has 7)",
        free_lambdas + 3
    );
    Some(if at_once > 1 {
        format!(
            "the bootstrap will need about {:.1} GB of memory for its M-steps: it fits up to \
             {at_once} replicates at once, one per thread, and the M-step of this model \
             ({states} intervals, {free_lambdas} free lambdas) needs about {:.1} GB each: it \
             {why}, use fewer --n-steps, or fit fewer replicates at once with fewer --threads",
            bytes / 1e9,
            each / 1e9
        )
    } else {
        format!(
            "the M-step of this model ({states} intervals, {free_lambdas} free lambdas) will \
             need about {:.1} GB of memory: it {why}, or use fewer --n-steps",
            bytes / 1e9
        )
    })
}

/// How many bootstrap replicates, each with its own single-threaded
/// M-step, may run at once: one per thread of the pool, at most all of
/// them.
fn bootstrap_m_steps_at_once(replicates: usize, threads: usize) -> usize {
    replicates.min(threads).max(1)
}

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
    // The layout of the starting model: its free lambdas, and the time grid
    // on which QC checks the start.
    let layout = match &saved {
        Some(model) => model.clone(),
        None => layout_model(cli.t_max, n_steps, pattern.clone(), cli.mu)?,
    };
    let free_lambdas = layout.lam.len();
    let bootstrap = BootstrapPlan::of(cli, n_iter, &output_file)?;
    // The main fit runs one M-step at a time; the bootstrap runs its
    // replicates in parallel.
    let m_steps_at_once = bootstrap.as_ref().map_or(1, |plan| {
        bootstrap_m_steps_at_once(plan.replicates, pool.current_num_threads())
    });
    if (n_iter > 0 || bootstrap.is_some())
        && let Some(msg) = mstep_memory_warning(layout.n_steps + 1, free_lambdas, m_steps_at_once)
    {
        sink(Event::Warning(msg));
    }

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

    // Input QC, printed before the fit, also when no sequence is left (the
    // report then says why). A QC error is only reported, so that it never
    // stops a fit.
    let trunk = bootstrap.as_ref().map_or(0, |plan| plan.trunk_size);
    let start = fit_start(cli, &layout, obs);
    let qc = match qc_report(&data, cli.input.bin_size, trunk, Some(free_lambdas), start) {
        Ok(report) => Some(report),
        Err(e) => {
            sink(qc_failed(e, cli.qc_json.as_deref()));
            None
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
            let (theta0, rho0) = start_rates(cli, obs)?;
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
    let final_loglike = match history.final_loglike() {
        Some(ll) => ll,
        // EM evaluates no parameters in 0 iterations; the written model is
        // the starting one, and the JSON's loglike starts with its value.
        None => model_loglike(&model, obs, cancel, sink, pool)?,
    };
    sink(Event::Output(format!("Last EM loglike: {final_loglike}")));
    let final_loglike = Some(final_loglike);
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
        final_loglike,
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
    if bootstrap.is_none() {
        let dir = default_bootstrap_dir(&output_file);
        if replicate_files(&dir).is_ok_and(|mut files| files.next().is_some()) {
            sink(Event::Warning(format!(
                "{} holds the bootstrap of an earlier run, not of this fit; this run made \
                 no bootstrap",
                dir.display()
            )));
        }
    }
    if let Some(path) = &cli.psmc_out {
        write_main_psmc(path, &model, &history, &data)?;
        sink(Event::Output(format!("PSMC output: {}", path.display())));
        if n_iter > 0 {
            sink(Event::Info(format!(
                "To plot it: {} (-s is the bin size, which the .psmc file does not record)",
                psmc_plot_command(cli, n_iter, &path.display().to_string())
            )));
        }
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
        if plan.em_iters != n_iter {
            sink(Event::Warning(replicate_plot_warning(cli, n_iter, plan)));
        }
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
/// input (see [`check_destinations`]), and the chart library the report
/// embeds when `PSMC_ECHARTS_JS` names one.
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
        usage_bail!(
            "OUTPUT {} ends in .html, but the HTML report is written to OUTPUT with the \
             extension .html; name the parameters file .json",
            output.display()
        );
    }
    let mut files = vec![("OUTPUT", output), ("the HTML report", report_html)];
    for (what, path) in [
        ("--psmc-out", &cli.psmc_out),
        ("--tmrca-out", &cli.tmrca_out),
    ] {
        if let Some(path) = path.as_deref() {
            files.push((what, path));
        }
    }
    // Only the QC JSON can go to stdout.
    if let Some(path) = cli.qc_json.as_deref().filter(|p| !is_stdout(p)) {
        files.push(("--qc-json", path));
    }
    let dir_name = match cli.bootstrap_dir {
        Some(_) => "--bootstrap-dir",
        None => "the bootstrap directory",
    };
    let dirs: Vec<(&str, &Path)> = bootstrap
        .map(|plan| (dir_name, plan.dir.as_path()))
        .into_iter()
        .collect();
    check_destinations(input, &files, &dirs)?;
    check_echarts_bundle()
}

/// The psmc_plot.pl command that plots round `n_iter` of the `.psmc` files
/// `files` as the report does. The files do not record the bin size, and
/// psmc_plot.pl assumes `-s 100 -u 2.5e-8 -g 25` and round 20.
fn psmc_plot_command(cli: &InferArgs, n_iter: usize, files: &str) -> String {
    format!(
        "psmc_plot.pl -s {} -u {:e} -g {} -n {n_iter} PREFIX {files}",
        cli.input.bin_size, cli.mu, cli.gen_years,
    )
}

/// The warning of a bootstrap whose replicates ran another number of EM
/// iterations than the main fit. psmc_plot.pl -n takes the same round of
/// every run in its input, so the replicates cannot share a plot command
/// with the main fit's `.psmc` output (as concatenated for C psmc's
/// bootstrap): with fewer rounds it fails, with more it plots them at an
/// intermediate round.
fn replicate_plot_warning(cli: &InferArgs, n_iter: usize, plan: &BootstrapPlan) -> String {
    let replicates = plan.dir.join("replicate_*.psmc");
    format!(
        "the bootstrap replicates ran {} EM iterations and the main fit {n_iter}; psmc_plot.pl \
         -n takes the same round of every run it reads, so do not plot the replicates in one \
         file with the main fit's .psmc output, but on their own: {}",
        plan.em_iters,
        psmc_plot_command(cli, plan.em_iters, &replicates.display().to_string())
    )
}

/// θ0 and ρ0 of a fresh start: `--theta0` or C psmc's -ln(1 - K/L) of the
/// data, and `--rho0` or θ0 / `--tr-ratio`.
fn start_rates(cli: &InferArgs, obs: &Observations) -> Result<(f64, f64)> {
    let theta0 = match cli.theta0 {
        Some(v) => v,
        None => initial_theta(&obs.rows)?,
    };
    Ok((theta0, cli.rho0.unwrap_or(theta0 / cli.tr_ratio)))
}

/// The start of the fit, as QC checks it ([`FitStart`]): the saved
/// parameters of `--init-params` (then `layout`), or the rates of
/// [`start_rates`] on `layout`'s time grid; `None` when the data give no
/// θ0, which the run then reports.
fn fit_start(cli: &InferArgs, layout: &PsmcModel, obs: &Observations) -> Option<FitStart> {
    let (theta0, rho0) = match cli.init_params {
        Some(_) => (layout.theta, layout.rho),
        None => start_rates(cli, obs).ok()?,
    };
    Some(FitStart::new(layout, theta0, rho0, false))
}

/// The starting model of `--init-params`: the saved parameters, with the
/// layout checked against `--pattern` and `--n-steps` when they are given
/// (a mismatch is a usage error), and `--mu` for scaling.
fn saved_start(path: &Path, cli: &InferArgs) -> Result<PsmcModel> {
    let what = || format!("--init-params {}", path.display());
    let saved = load_params(path).with_context(what)?;
    let mut model = PsmcModel::from_params_file(&saved).with_context(what)?;
    if cli.pattern.is_some() || cli.n_steps.is_some() {
        let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
        let wanted = layout_model(model.t_max, n_steps, pattern, model.mu)?;
        if (wanted.n_steps, c_pattern(&wanted)) != (model.n_steps, c_pattern(&model)) {
            usage_bail!(
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

/// The warning of a QC report that could not be made, naming the QC JSON
/// that is therefore not written.
pub fn qc_failed(e: anyhow::Error, qc_json: Option<&Path>) -> Event {
    let unwritten = qc_json.map_or(String::new(), |path| {
        format!(", so {} is not written", path.display())
    });
    Event::Warning(format!("input QC failed{unwritten}: {e:#}"))
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
    if let Some(line) = ignored_options(&cli.input, &data) {
        sink(Event::Warning(line));
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

/// Log-likelihood of `model` on `obs`, by the forward pass with which EM
/// evaluates the parameters it ends with.
fn model_loglike(
    model: &PsmcModel,
    obs: &Observations,
    cancel: &AtomicBool,
    sink: Sink,
    pool: &ThreadPool,
) -> Result<f64> {
    let mut on_progress = |p: EStepProgress| {
        sink(Event::Progress {
            stage: Stage::EForward,
            done: p.done,
            total: p.total,
            iter: None,
        })
    };
    let opts = EStepOptions {
        on_progress: Some(&mut on_progress),
        cancel: Some(cancel),
        thread_pool: Some(pool),
        ..EStepOptions::default()
    };
    log_likelihood_with_options(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        opts,
    )
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
    let dir = cfg
        .out_dir
        .as_deref()
        .expect("the CLI always writes the replicates");
    if let Some(msg) = remove_earlier_bootstrap(dir, cfg.replicates)? {
        sink(Event::Warning(msg));
    }
    sink(progress(0));
    let on_done = |n: usize| sink(progress(n));
    let opts = BootstrapOptions {
        cancel: Some(cancel),
        thread_pool: Some(pool),
        on_done: Some(&on_done),
        start,
    };
    let replicates = run_bootstrap_with_options(seqs, model, cfg, mstep_config, opts)?;
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

/// The replicate files in `dir` (`replicate_NNN.json` and
/// `replicate_NNN.psmc`, as a bootstrap writes them) with their replicate
/// numbers; none when `dir` does not exist.
fn replicate_files(dir: &Path) -> Result<impl Iterator<Item = (usize, PathBuf)>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => Some(entries),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("failed to list {}", dir.display())),
    };
    Ok(entries.into_iter().flatten().filter_map(|entry| {
        let entry = entry.ok()?;
        if !entry.file_type().ok()?.is_file() {
            return None;
        }
        let name = entry.file_name();
        let name = name.to_str()?;
        let stem = name
            .strip_suffix(".json")
            .or_else(|| name.strip_suffix(".psmc"))?;
        let number = stem.strip_prefix("replicate_")?;
        if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some((number.parse().ok()?, entry.path()))
    }))
}

/// Remove the files an earlier bootstrap left in `dir` (its replicate
/// files and its summary) before this run writes any, and say so for the
/// replicates beyond this run's `replicates`, which this run would not
/// have replaced.
///
/// This run writes each replicate's JSON as the replicate finishes, the
/// `.psmc` files and the summary only at the end. Replacing the earlier
/// files one by one would leave the two runs mixed when this one stops
/// part way (an error, a kill, running out of memory), and would leave
/// replicates beyond this run's count in place for good, which
/// `cat replicate_*.psmc` (the README's recipe for psmc_plot.pl) and globs
/// of `replicate_*.json` would then mix with this run's while the summary
/// and the report use only the new ones.
fn remove_earlier_bootstrap(dir: &Path, replicates: usize) -> Result<Option<String>> {
    let earlier: Vec<(usize, PathBuf)> = replicate_files(dir)?.collect();
    let summary = BOOTSTRAP_SUMMARY_FILES
        .map(|name| dir.join(name))
        .into_iter()
        .filter(|path| path.is_file());
    for path in earlier.iter().map(|(_, path)| path.clone()).chain(summary) {
        fs::remove_file(&path).with_context(|| format!("failed to remove {}", path.display()))?;
    }
    let mut stale: Vec<(usize, PathBuf)> = earlier
        .into_iter()
        .filter(|&(n, _)| n > replicates)
        .collect();
    if stale.is_empty() {
        return Ok(None);
    }
    stale.sort();
    let (first, last) = (stale[0].0, stale[stale.len() - 1].0);
    let which = if first == last {
        format!("replicate {first}")
    } else {
        format!("replicates {first} to {last}")
    };
    Ok(Some(format!(
        "{} held {which} of an earlier bootstrap, beyond this run's {replicates}; their {} \
         files were removed so that the directory holds only this run's replicates",
        dir.display(),
        stale.len()
    )))
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

    /// The estimate follows the measured peaks of a fit with one M-step
    /// iteration (macOS, peak memory footprint): 23 MB at 101 states, 118 MB
    /// at 201, 401 MB at 301 and 1.30 GB at 401, one free lambda each.
    #[test]
    fn mstep_memory_is_estimated_from_the_layout() {
        for (states, measured) in [(101, 23e6), (201, 118e6), (301, 401e6), (401, 1.30e9)] {
            let estimate = mstep_memory_bytes(states, states);
            assert!(
                (0.9 * measured..2.0 * measured).contains(&estimate),
                "{states}: {estimate} vs {measured}"
            );
        }
        // C psmc's patterns stay far below the limit.
        assert_eq!(mstep_memory_warning(23, 7, 1), None);
        assert_eq!(mstep_memory_warning(64, 28, 1), None);
        assert_eq!(mstep_memory_warning(301, 301, 1), None);
        let msg = mstep_memory_warning(401, 401, 1).unwrap();
        assert!(
            msg.starts_with(
                "the M-step of this model (401 intervals, 401 free lambdas) will need about \
                 1.3 GB of memory: it carries the derivatives of all 404 parameters"
            ),
            "{msg}"
        );
        // Many intervals in few groups need little.
        assert_eq!(mstep_memory_warning(1000, 10, 1), None);
        // M-steps that run at once add up: 0.56 GB each, 2.2 GB for four.
        let msg = mstep_memory_warning(301, 301, 4).unwrap();
        assert!(
            msg.starts_with(
                "the bootstrap will need about 2.2 GB of memory for its M-steps: it fits up to \
                 4 replicates at once, one per thread, and the M-step of this model (301 \
                 intervals, 301 free lambdas) needs about 0.6 GB each"
            ),
            "{msg}"
        );
        assert!(msg.ends_with("with fewer --threads"), "{msg}");
        assert_eq!(mstep_memory_warning(23, 7, 64), None);
        assert_eq!(bootstrap_m_steps_at_once(100, 8), 8);
        assert_eq!(bootstrap_m_steps_at_once(3, 8), 3);
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
