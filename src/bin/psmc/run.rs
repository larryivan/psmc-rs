//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

use anyhow::{Context, Result, anyhow, bail};
use rayon::ThreadPool;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, BootstrapSummary, InvalidReplicateStart,
    bootstrap_trunk_count, default_bootstrap_dir, invalid_replicate_starts_sequences,
    observations_to_sequences, run_bootstrap_sequences, summarize_bootstrap_models,
};
use psmc_rs::em::{EmHistory, EmOptions, EmProgressEvent, em_train_with_options};
use psmc_rs::hmm::{
    Cancelled, EStepOptions, EStepPhase, EStepProgress, TmrcaOptions, log_likelihood_with_options,
    write_tmrca_posterior_tsv_with_options,
};
use psmc_rs::io::input::{Dataset, SourceSummary};
use psmc_rs::io::params::{
    BinCallableRecord, CALLABLE_LENGTH_MODEL, FULL_BINS_MODEL, InputSummary, load_params,
};
use psmc_rs::io::{CallableBases, Observations, Sequences};
use psmc_rs::model::NegativeTransitions;
use psmc_rs::opt::{MStepConfig, bounds_from_config};
use psmc_rs::psmc_out::c_pattern;
use psmc_rs::qc::{FitStart, QcReport, Severity, render_text};
use psmc_rs::report::{
    BootstrapReport, ReportData, check_echarts_bundle, default_html_path, write_html_report,
};
use psmc_rs::{PsmcModel, initial_theta_with_callable};

use crate::cli::{BinCallableArg, InferArgs, model_layout, usage_bail};
use crate::dataset::{
    callable_summary, counts_source, describe, ensure_usable, ignored_options, layout_model,
    load_input, loader_warnings, qc_report, resolve_input_path,
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
    let mut data = read_input(&input_file, cli, sink)?;
    // Whether the fit weighs each bin by its callable bases.
    let bin_callable = bin_callable_record(&data, cli.bin_callable);
    for event in bin_callable_events(&bin_callable, &data, cli) {
        sink(event);
    }
    let counted = bin_callable.model == CALLABLE_LENGTH_MODEL;

    // Input QC, printed before the fit, also when no sequence is left (the
    // report then says why). A QC error is only reported, so that it never
    // stops a fit. It sees the counts whether or not the fit uses them.
    let trunk = bootstrap.as_ref().map_or(0, |plan| plan.trunk_size);
    let fit_counts = data.obs.callable.as_ref().filter(|_| counted);
    let start = fit_start(cli, &layout, &data.obs.rows, fit_counts);
    let qc = match qc_report(
        &data,
        cli.input.bin_size,
        trunk,
        Some(free_lambdas),
        start,
        !counted,
    ) {
        Ok(report) => Some(report),
        Err(e) => {
            sink(qc_failed(e, cli.qc_json.as_deref()));
            None
        }
    };
    // Without the callable-length model the counts go, so that every pass
    // (theta0, EM, the bootstrap, the decoding) fits C psmc's model.
    if !counted {
        data.obs.callable = None;
    }
    let data = data;
    let obs = &data.obs;
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
            let seqs = observations_to_sequences(obs)?;
            let trunks = bootstrap_trunk_count(&seqs.bins, plan.trunk_size)
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
            let (theta0, rho0) = start_rates(cli, &obs.rows, obs.callable.as_ref())?;
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
    let boot = match (&bootstrap, &boot_seqs) {
        (Some(plan), Some((seqs, trunks))) => {
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
            Some((plan, seqs, *trunks, cfg))
        }
        _ => None,
    };
    // A replicate's fresh start takes theta0 from its own resampled data,
    // so it can be past the limit on rho when the main start is not. Such
    // replicates are found now, before the main fit, and the bootstrap
    // skips them.
    let invalid_replicates = match &boot {
        Some((_, seqs, _, cfg)) if saved_start.is_none() => {
            invalid_replicate_starts_sequences(seqs, &model, cfg)?
        }
        _ => Vec::new(),
    };
    // EM would refuse a start it cannot move from; refuse it here, with
    // advice in this run's options that fits the replicates too. With
    // saved parameters, every bootstrap replicate starts from them, so
    // they are checked even when the main fit runs no iteration.
    if n_iter > 0 || (bootstrap.is_some() && saved_start.is_some()) {
        check_start(&model, cli, replicates_tr_ratio(&invalid_replicates))?;
    }
    if let Some((_, _, _, cfg)) = &boot
        && let Some(msg) = invalid_replicates_warning(cli, &model, cfg, &invalid_replicates)?
    {
        sink(Event::Warning(msg));
    }

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
        bin_callable: Some(&bin_callable),
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
        write_main_psmc(path, &model, &history, &data, &bin_callable)?;
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
    if let Some((plan, seqs, trunks, cfg)) = &boot {
        let summary = bootstrap_stage(
            seqs,
            &model,
            cfg,
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
        // Decoded with the emissions the fit used, of each bin's count.
        let tmrca = write_tmrca_posterior_tsv_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            &tmrca_years,
            tmrca_out,
            TmrcaOptions {
                progress_bars,
                max_points: 20_000,
                callable: obs.callable.as_ref(),
            },
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
        bin_callable: Some(&bin_callable),
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

/// θ0 and ρ0 of a fresh start: `--theta0` or θ0 of the data (C psmc's
/// -ln(1 - K/L), per callable base when the fit uses the callable bases of
/// each bin: [`initial_theta_with_callable`]), and `--rho0` or θ0 /
/// `--tr-ratio`.
fn start_rates(
    cli: &InferArgs,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
) -> Result<(f64, f64)> {
    let theta0 = match cli.theta0 {
        Some(v) => v,
        None => initial_theta_with_callable(rows, callable)?,
    };
    Ok((theta0, cli.rho0.unwrap_or(theta0 / cli.tr_ratio)))
}

/// How the fit of `data` uses the callable bases of its bins under
/// `--bin-callable mode`: the callable-length model when the mode is `auto`
/// and the data have counts (VCF and multihetsep), C psmc's model
/// otherwise. The counts are summarised whether or not the fit uses them.
fn bin_callable_record(data: &Dataset, mode: BinCallableArg) -> BinCallableRecord {
    let callable = callable_summary(data);
    let counted = callable.is_some() && mode == BinCallableArg::Auto;
    BinCallableRecord {
        mode: mode.name().to_string(),
        model: if counted {
            CALLABLE_LENGTH_MODEL
        } else {
            FULL_BINS_MODEL
        }
        .to_string(),
        counts: counts_source(data).map(str::to_string),
        callable,
    }
}

/// What the run says about the callable bases of `data`'s bins: what they
/// hold and how the fit uses them, or that `--bin-callable ignore` changes
/// nothing for input without counts.
fn bin_callable_events(b: &BinCallableRecord, data: &Dataset, cli: &InferArgs) -> Vec<Event> {
    let percent = |x: f64| format!("{:.1}%", 100.0 * x);
    let s = cli.input.bin_size;
    match (&b.callable, b.counts.as_deref()) {
        (Some(c), Some(source)) => {
            let how = if b.model == CALLABLE_LENGTH_MODEL {
                "the fit weighs each called bin by its callable bases (--bin-callable auto)"
                    .to_string()
            } else {
                format!(
                    "the fit treats every called bin as {s} bases, as fq2psmcfa and C psmc do \
                     (--bin-callable ignore)"
                )
            };
            vec![Event::Info(format!(
                "Callable bases ({source} counts): called bins hold {} of their {s} bases on \
                 average, and {} of them fewer than {s}; {how}",
                percent(c.mean_fraction),
                percent(c.partial_fraction())
            ))]
        }
        _ if cli.bin_callable == BinCallableArg::Ignore => {
            let what = match &data.source {
                SourceSummary::Psmcfa(_) => {
                    "psmcfa input, which records no callable bases per bin".to_string()
                }
                _ => format!(
                    "bins of {s} bases, whose callable bases are not counted (only bins of at \
                     most {} bases are)",
                    psmc_rs::io::MAX_COUNTED_BIN_SIZE
                ),
            };
            vec![Event::Warning(format!(
                "--bin-callable ignore changes nothing for {what}: every called bin is taken to \
                 hold {s} bases, as C psmc takes them"
            ))]
        }
        _ if !matches!(data.source, SourceSummary::Psmcfa(_)) => vec![Event::Info(format!(
            "Callable bases are counted for bins of at most {} bases; bins of {s} are fitted as \
             C psmc fits them, every called bin as {s} bases",
            psmc_rs::io::MAX_COUNTED_BIN_SIZE
        ))],
        _ => Vec::new(),
    }
}

/// The start of the fit, as QC checks it ([`FitStart`]): the saved
/// parameters of `--init-params` (then `layout`), or the rates of
/// [`start_rates`] on `layout`'s time grid; `None` when the data give no
/// θ0, which the run then reports.
fn fit_start(
    cli: &InferArgs,
    layout: &PsmcModel,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
) -> Option<FitStart> {
    let (theta0, rho0) = match cli.init_params {
        Some(_) => (layout.theta, layout.rho),
        None => start_rates(cli, rows, callable).ok()?,
    };
    let mut start = FitStart::new(layout, theta0, rho0, false);
    start.saved = cli.init_params.is_some();
    Some(start)
}

/// The warning about the bootstrap replicates of `cfg` whose fresh start
/// is `invalid` ([`invalid_replicate_starts`]), which the bootstrap skips,
/// with the start that lets every replicate be fitted; an error when that
/// is all of them, before the main fit. `model` is the main fit's start.
fn invalid_replicates_warning(
    cli: &InferArgs,
    model: &PsmcModel,
    cfg: &BootstrapConfig,
    invalid: &[InvalidReplicateStart],
) -> Result<Option<String>> {
    let Some(first) = invalid.first() else {
        return Ok(None);
    };
    let numbers: Vec<usize> = invalid.iter().map(|r| r.index + 1).collect();
    let highest_rho0 = invalid.iter().map(|r| r.start.rho).fold(0.0, f64::max);
    let past = match first.start.max_rho {
        Some(limit) => {
            format!("rho0 up to {highest_rho0:.6}, above the limit of {limit:.6} per bin")
        }
        None => format!("rho0 up to {highest_rho0:.6} per bin"),
    };
    // Replicates take the main start's theta0/rho0.
    let advice = match (replicates_tr_ratio(invalid), cli.rho0) {
        (Some(ratio), Some(_)) => format!(
            "start from a --rho0 of {} or less, which gives every replicate theta0/rho0 {ratio} \
             or more",
            round_down(model.theta / ratio)
        ),
        (Some(ratio), None) => format!("start from --tr-ratio {ratio} or more (C psmc's -r)"),
        (None, _) => "start from a smaller rho0".to_string(),
    };
    if invalid.len() == cfg.replicates {
        bail!(
            "every one of the {} bootstrap replicates would start from negative transition \
             probabilities ({past}): their theta0, from their own resampled data, sets their \
             rho0, so none of them can be fitted. To fit them, {advice}",
            cfg.replicates
        );
    }
    let fitted = cfg.replicates - invalid.len();
    Ok(Some(format!(
        "bootstrap {} ({} of {}) would start from negative transition probabilities ({past}): \
         their resampled data are more heterozygous than the input, and their theta0, from \
         those data, sets their rho0. They will be skipped and the bands will come from the \
         other {fitted}; leaving out the most heterozygous resamples can make the bands too \
         narrow. To fit every replicate, {advice}",
        list_replicates(&numbers),
        invalid.len(),
        cfg.replicates
    )))
}

/// The smallest theta0/rho0 at which every one of the `invalid` replicate
/// starts is valid; `None` without any, or when one has no limit.
fn replicates_tr_ratio(invalid: &[InvalidReplicateStart]) -> Option<f64> {
    if invalid.is_empty() {
        return None;
    }
    invalid
        .iter()
        .map(|r| r.start.min_tr_ratio())
        .collect::<Option<Vec<f64>>>()
        .map(|ratios| ratios.into_iter().fold(0.0, f64::max))
}

/// `x` rounded down to 6 significant digits, for a bound not to be passed.
fn round_down(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return x;
    }
    let scale = 10f64.powi(5 - x.log10().floor() as i32);
    (x * scale).floor() / scale
}

/// Replicates by number: "replicate 3", "replicates 3 and 7", or the first
/// five and how many more ("replicates 2, 3, 4, 5, 6 and 4 more").
fn list_replicates(numbers: &[usize]) -> String {
    let shown: Vec<String> = numbers.iter().take(5).map(usize::to_string).collect();
    match (numbers.len(), shown.as_slice()) {
        (1, [one]) => format!("replicate {one}"),
        (n, [rest @ .., last]) if n <= 5 => format!("replicates {} and {last}", rest.join(", ")),
        (n, _) => format!(
            "replicates {} and {} more",
            shown.join(", "),
            n - shown.len()
        ),
    }
}

/// Refuse a start from which EM cannot move
/// ([`PsmcModel::check_transitions`]), as EM would, but with advice in the
/// options of this run ([`start_advice`]). `replicates_ratio` is the
/// theta0/rho0 that the bootstrap replicates need
/// ([`replicates_tr_ratio`]), if more than the start's.
fn check_start(model: &PsmcModel, cli: &InferArgs, replicates_ratio: Option<f64>) -> Result<()> {
    let Err(e) = model.check_transitions() else {
        return Ok(());
    };
    let Some(negative) = e.downcast_ref::<NegativeTransitions>() else {
        return Err(e.context("EM cannot start from these parameters"));
    };
    bail!(
        "EM cannot start from these parameters: {}. {}",
        negative.finding(),
        start_advice(cli, negative, replicates_ratio)
    )
}

/// How this run can start from a smaller rho than the start of
/// `negative`: with `--init-params`, which rules out the options of a
/// fresh start, by lowering rho in the saved file; otherwise with a
/// smaller `--rho0` when one was given, and with `--tr-ratio` (C psmc's
/// `-r`) or `--rho0` when not. Smaller bins also lower rho per bin. The
/// ratio advised is at least `replicates_ratio`, which the bootstrap
/// replicates need ([`replicates_tr_ratio`]), so that following the advice
/// fits them too.
fn start_advice(
    cli: &InferArgs,
    negative: &NegativeTransitions,
    replicates_ratio: Option<f64>,
) -> String {
    let own = negative.min_tr_ratio();
    let ratio = own.map(|r| replicates_ratio.map_or(r, |b| r.max(b)));
    // The bootstrap replicates, which take theta0 from their own resampled
    // data, can need a larger ratio than the start.
    let replicates = if ratio > own {
        "; the bootstrap replicates, whose theta0 comes from their own resampled data, need \
         that much"
    } else {
        ""
    };
    match (&cli.init_params, cli.rho0, ratio) {
        (Some(path), ..) => format!(
            "Set rho in {}{}, or fit from a fresh start instead of --init-params",
            path.display(),
            negative
                .max_rho
                .map_or(String::new(), |rho| format!(" below {rho:.6}"))
        ),
        (None, Some(_), Some(ratio)) => format!(
            "Re-bin the data with fewer base pairs per bin, or give a --rho0 of {} or less \
             (theta0/rho0 {ratio} or more{replicates})",
            round_down(negative.theta / ratio)
        ),
        (None, None, Some(ratio)) => format!(
            "Re-bin the data with fewer base pairs per bin, or start from a smaller rho0: \
             --tr-ratio {ratio} or more (C psmc's -r), or a --rho0 of {} or less{replicates}",
            round_down(negative.theta / ratio)
        ),
        (None, _, None) => "Re-bin the data with fewer base pairs per bin, or start from a \
                            smaller rho0 with a larger --tr-ratio (C psmc's -r) or a smaller \
                            --rho0"
            .to_string(),
    }
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
            callable: obs.callable.as_ref(),
        },
    )?;
    model.param_recalculate()?;
    Ok(history)
}

/// Log-likelihood of `model` on `obs` (with its callable bases when the fit
/// uses them), by the forward pass with which EM evaluates the parameters
/// it ends with.
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
        callable: obs.callable.as_ref(),
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
    seqs: &Sequences,
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
    // A fresh start can be invalid for some replicates
    // ([`invalid_replicates_warning`] named them); saved parameters are the
    // main fit's start, checked before it.
    let opts = BootstrapOptions {
        cancel: Some(cancel),
        thread_pool: Some(pool),
        on_done: Some(&on_done),
        start,
        skip_invalid_starts: start.is_none(),
    };
    let replicates = run_bootstrap_sequences(seqs, model, cfg, mstep_config, opts)?;
    write_replicate_psmc(dir, &replicates)?;
    let mut fitted = vec![false; cfg.replicates];
    for r in &replicates {
        fitted[r.index] = true;
    }
    let skipped: Vec<usize> = (1..=cfg.replicates).filter(|&n| !fitted[n - 1]).collect();
    let models: Vec<PsmcModel> = replicates.into_iter().map(|r| r.model).collect();
    let mut summary = summarize_bootstrap_models(
        model,
        &models,
        bin_size,
        gen_years,
        cfg.trunk_size,
        cfg.seed,
    )?;
    let skipped_note = if skipped.is_empty() {
        String::new()
    } else {
        format!(
            "; {} of {}, {} skipped",
            summary.n_replicates,
            cfg.replicates,
            list_replicates(&skipped)
        )
    };
    summary.skipped_replicates = skipped;
    let (summary_tsv, summary_json) = write_bootstrap_summary(dir, &summary)?;
    sink(Event::Output(format!(
        "Bootstrap replicates (replicate_NNN.json, replicate_NNN.psmc{skipped_note}): {}",
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

/// Whether `v` is on the lower or the upper M-step bound of its range: where
/// the M-step's logit coordinate saturates, not merely close in a wide range.
///
/// Above, within 1e-6 of the range of `hi`: the logit's slope there is below
/// 1e-6 of its largest, and a λ the data push to the bound stops there
/// (9,999.99991 for a bound of 1e4 on a simulated bottleneck genome). Below,
/// within twice the M-step's clamp (1e-12 of the range) of `lo`, or 1e-6 of
/// `lo` itself: towards `lo` the logit is a logarithm, so λ = 0.01 against a
/// bound of 1e-6 is an ordinary value.
fn near(v: f64, lo: f64, hi: f64) -> (bool, bool) {
    let span = (hi - lo).abs();
    let below = (2e-12 * span).max(1e-6 * lo.abs()).max(f64::EPSILON * 16.0);
    let above = (1e-6 * span).max(f64::EPSILON * 16.0);
    (v - lo <= below, hi - v <= above)
}

/// A warning when fitted parameters sit on the M-step's bounds, saying
/// what that means for each kind (the CLI has no option to move them).
fn near_bounds_warning(model: &PsmcModel, cfg: &MStepConfig) -> Option<String> {
    let bounds = bounds_from_config(model, cfg);
    let mut vals: Vec<(String, f64)> = Vec::with_capacity(3 + model.lam.len());
    vals.push(("theta".to_string(), model.theta));
    vals.push(("rho".to_string(), model.rho));
    vals.push(("t_max".to_string(), model.t_max));
    for (i, v) in model.lam.iter().enumerate() {
        vals.push((format!("lam[{i}]"), *v));
    }

    let (mut lam_hi, mut lam_lo, mut others) = (Vec::new(), Vec::new(), Vec::new());
    for (i, ((name, v), b)) in vals.iter().zip(&bounds).enumerate() {
        let (is_lo, is_hi) = near(*v, b.lo, b.hi);
        if !(is_lo || is_hi) {
            continue;
        }
        let bound = if is_lo { b.lo } else { b.hi };
        match (i >= 3, is_hi) {
            (true, true) => lam_hi.push(name.clone()),
            (true, false) => lam_lo.push(name.clone()),
            (false, _) => others.push(format!("{name} = {bound:e}")),
        }
    }
    let list = |names: &[String]| {
        if names.len() > 6 {
            format!("{} and {} more", names[..5].join(", "), names.len() - 5)
        } else {
            names.join(", ")
        }
    };
    let mut parts = Vec::new();
    if !lam_hi.is_empty() {
        // With one λ per interval, grouping the intervals is the remedy.
        let grouping = if model.pattern_spec().is_none() {
            "; a pattern that groups them, as C psmc's `4+...` patterns do, fits one λ for several"
        } else {
            ""
        };
        parts.push(format!(
            "{} at the M-step's upper bound of λ ({}): the data hold almost no coalescences in \
             their intervals, so the Ne there is that bound, not an estimate (usual for the \
             most recent intervals{grouping})",
            list(&lam_hi),
            cfg.lam_hi
        ));
    }
    if !lam_lo.is_empty() {
        parts.push(format!(
            "{} at the M-step's lower bound of λ ({}): the fit makes every lineage that reaches \
             those intervals coalesce at once (seen for the oldest interval, beyond t_max)",
            list(&lam_lo),
            cfg.lam_lo
        ));
    }
    if !others.is_empty() {
        parts.push(format!(
            "{} at the M-step's bound: check the input and the starting values",
            others.join(", ")
        ));
    }
    (!parts.is_empty()).then(|| format!("fitted parameters on their bounds: {}.", parts.join("; ")))
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
        // Each kind with what it means, and no advice the CLI cannot follow.
        assert!(
            msg.contains("lam[2] at the M-step's upper bound of λ (10000)")
                && msg.contains("not an estimate"),
            "{msg}"
        );
        assert!(msg.contains("rho = 1e-12 at the M-step's bound"), "{msg}");
        // The pattern already groups intervals: no advice to group them.
        assert!(!msg.contains("groups them"), "{msg}");
        let mut one_each = PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, None).unwrap();
        one_each.lam[0] = cfg.lam_hi;
        let msg = near_bounds_warning(&one_each, &cfg).unwrap();
        assert!(
            msg.contains("lam[0] at the M-step's upper bound") && msg.contains("groups them"),
            "{msg}"
        );
        assert!(!msg.contains("wider bounds"), "{msg}");
        // Where an M-step leaves a λ the data push to the bound, saturated
        // in its logit coordinate but not equal to the bound.
        model.lam[2] = 9_999.999_906;
        model.rho = 0.0025;
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        assert!(msg.contains("lam[2] at the M-step's upper bound"), "{msg}");
        // Ordinary values of a wide range are not on a bound.
        for lam in [0.5 * cfg.lam_hi, 9_990.0, 0.01, 5e-6] {
            model.lam[2] = lam;
            assert_eq!(near_bounds_warning(&model, &cfg), None, "{lam}");
        }
        // The M-step's clamp keeps a λ 1e-12 of its range above lo.
        model.lam[2] = cfg.lam_lo + 1e-12 * (cfg.lam_hi - cfg.lam_lo);
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        assert!(
            msg.contains("lam[2] at the M-step's lower bound of λ"),
            "{msg}"
        );
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
