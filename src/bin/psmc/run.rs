//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

mod bootstrap;
mod fit;
mod start;

use anyhow::{Context as _, Result, bail};
use rayon::ThreadPool;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::BootstrapSummary;
use psmc_rs::hmm::{Cancelled, TmrcaReportData};
use psmc_rs::io::MAX_COUNTED_BIN_SIZE;
use psmc_rs::io::input::{Dataset, SourceSummary};
use psmc_rs::io::params::{
    BinCallableRecord, CALLABLE_LENGTH_MODEL, FULL_BINS_MODEL, InputSummary,
};
use psmc_rs::opt::MStepConfig;
use psmc_rs::qc::fmt::{count, counted};
use psmc_rs::qc::{QcReport, Severity, render_text};
use psmc_rs::report::{
    BootstrapReport, ReportData, check_echarts_bundle, default_html_path, write_html_report,
};

use self::bootstrap::{Bootstrap, BootstrapPlan, earlier_bootstrap_warning};
use self::fit::{decode_tmrca, fit, model_loglike, near_bounds_warning};
use self::start::{check_starts, fit_start, saved_start, starting_model};
use crate::cli::{BinCallableArg, InferArgs, InputArgs, model_layout, usage_bail};
use crate::dataset::{
    callable_summary, counts_source, describe, ensure_usable, fit_uses_counts, ignored_options,
    layout_model, load_input, loader_warnings, qc_report, resolve_input_path, totals,
};
use crate::output::{
    FitRecord, check_destinations, is_stdout, save_fit, write_main_psmc, write_qc_json,
};

/// A stage of the pipeline with a progress count. The TUI indexes its
/// gauges by the order of the variants.
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

/// What the stages of a run share: the flag that cancels it, where its
/// events go, and its thread pool.
#[derive(Clone, Copy)]
struct Context<'a> {
    cancel: &'a AtomicBool,
    sink: Sink<'a>,
    pool: &'a ThreadPool,
}

impl Context<'_> {
    /// [`Cancelled`] once the run is cancelled.
    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Cancelled.into());
        }
        Ok(())
    }

    fn info(&self, line: String) {
        (self.sink)(Event::Info(line));
    }

    fn warn(&self, line: String) {
        (self.sink)(Event::Warning(line));
    }

    fn output(&self, line: String) {
        (self.sink)(Event::Output(line));
    }
}

/// Run inference as `cli` describes it, in a thread pool of its own with
/// `--threads` workers; the files written are named by [`Event::Output`].
/// `cancel` stops the run at the next check (between stages, and inside EM
/// and the bootstrap); the error is then [`Cancelled`]. `progress_bars`
/// lets the TMRCA decoding draw its own bar on stderr.
pub fn run_inference(
    cli: &InferArgs,
    cancel: &AtomicBool,
    sink: Sink,
    progress_bars: bool,
) -> Result<()> {
    // A pool per run, rather than the global one, so that a later run (in
    // the TUI) can use another thread count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.unwrap_or(0))
        .build()
        .context("failed to start the worker threads")?;
    let ctx = Context {
        cancel,
        sink,
        pool: &pool,
    };
    pool.install(|| infer(cli, ctx, progress_bars))
}

/// The input of a run, as read.
struct Input {
    data: Dataset,
    /// How the fit uses the callable bases of each bin.
    bin_callable: BinCallableRecord,
    /// The input QC report, when it could be made.
    qc: Option<QcReport>,
}

/// [`run_inference`] inside its pool.
fn infer(cli: &InferArgs, ctx: Context, progress_bars: bool) -> Result<()> {
    // The command line requires them; the TUI's form fills them.
    let (Some(given_input), Some(output), Some(n_iter)) =
        (&cli.input_file, &cli.output_file, cli.n_iter)
    else {
        bail!("INPUT, OUTPUT and N_ITER are required");
    };
    // Settings that can be checked without the data fail first.
    let saved = cli
        .init_params
        .as_deref()
        .map(|path| saved_start(path, cli))
        .transpose()?;
    let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
    // The layout of the starting model: its free lambdas, and the time grid
    // on which QC checks the start.
    let layout = match &saved {
        Some(model) => model.clone(),
        None => layout_model(cli.t_max, n_steps, pattern.clone(), cli.mu)?,
    };
    let bootstrap = BootstrapPlan::of(cli, n_iter, output)?;
    ctx.check_cancel()?;
    let input_file = resolve_input_path(given_input, cli.input.input_format)?;
    if input_file != *given_input {
        ctx.info(format!(
            "Input is a directory, selected file: {}",
            input_file.display()
        ));
    }
    let report_html = default_html_path(output);
    check_outputs(cli, &input_file, output, &report_html, bootstrap.as_ref())?;
    let trunk = bootstrap.as_ref().map_or(0, |plan| plan.cfg.trunk_size);
    let input = read_input(&input_file, cli, &layout, trunk, ctx)?;
    let obs = &input.data.obs;
    let resampled = bootstrap
        .as_ref()
        .map(|plan| plan.resample(obs, cli, ctx))
        .transpose()?;

    let mut model = starting_model(saved, (n_steps, pattern), obs, cli)?;
    if let Some(path) = &cli.init_params {
        ctx.info(format!(
            "Starting from the parameters in {}",
            path.display()
        ));
    }
    ctx.info(model_line(&model));
    let tr_ratio = model.theta / model.rho;
    let boot = bootstrap
        .zip(resampled)
        .map(|(plan, resampled)| Bootstrap::new(plan, resampled, &model, cli));
    check_starts(&model, boot.as_ref(), n_iter, cli, ctx)?;

    let mstep_config = MStepConfig {
        max_iters: cli.mstep_iters,
        lambda: cli.smooth_lambda,
        progress: false,
        ..MStepConfig::default()
    };
    let history = fit(&mut model, obs, n_iter, &mstep_config, ctx)?;
    let final_loglike = match history.final_loglike() {
        Some(ll) => ll,
        // EM evaluates no parameters in 0 iterations; the written model is
        // the starting one, and the JSON's loglike starts with its value.
        None => model_loglike(&model, obs, ctx)?,
    };
    ctx.output(format!("Last EM loglike: {final_loglike}"));
    if n_iter > 0
        && let Some(msg) = near_bounds_warning(&model, &mstep_config)
    {
        ctx.warn(msg);
    }
    ctx.check_cancel()?;

    // The main fit's outputs are written before the bootstrap, which can
    // take far longer.
    let record = FitRecord {
        cli,
        model: &model,
        history: &history,
        final_loglike,
        tr_ratio,
        input: &InputSummary::of(&input.data),
        qc: input.qc.as_ref(),
        bin_callable: &input.bin_callable,
    };
    write_fit(&record, &input.data, output, boot.is_none(), ctx)?;
    let boot_summary = boot
        .as_ref()
        .map(|b| b.run(&model, &mstep_config, cli, n_iter, ctx))
        .transpose()?;

    ctx.check_cancel()?;
    let tmrca = cli
        .tmrca_out
        .as_deref()
        .map(|path| decode_tmrca(&model, obs, path, cli, progress_bars, ctx))
        .transpose()?;
    let bootstrap = boot.as_ref().zip(boot_summary.as_ref());
    write_report(
        &record,
        &report_html,
        &input_file,
        output,
        bootstrap,
        tmrca.as_ref(),
    )?;
    ctx.output(format!("HTML report: {}", report_html.display()));
    Ok(())
}

/// The line that describes the model a fit starts from.
fn model_line(model: &PsmcModel) -> String {
    format!(
        "Model: pattern {}, {} intervals, {}; start theta0 {:.6}, rho0 {:.6} (theta0/rho0 {}), \
         t_max {}",
        model.pattern.as_deref().unwrap_or("none"),
        model.n_steps + 1,
        counted(model.lam.len(), "free lambda", "free lambdas"),
        model.theta,
        model.rho,
        model.theta / model.rho,
        model.t_max
    )
}

/// Write the main fit of `data` to `output`, and its `.psmc` output when
/// asked for; without a bootstrap (`no_bootstrap`), warn of the replicates
/// of an earlier run in the default bootstrap directory.
fn write_fit(
    fit: &FitRecord,
    data: &Dataset,
    output: &Path,
    no_bootstrap: bool,
    ctx: Context,
) -> Result<()> {
    save_fit(output, fit)?;
    ctx.output(format!("Parameters: {}", output.display()));
    if no_bootstrap && let Some(msg) = earlier_bootstrap_warning(output) {
        ctx.warn(msg);
    }
    if let Some(path) = &fit.cli.psmc_out {
        write_main_psmc(path, fit, data)?;
        ctx.output(format!("PSMC output: {}", path.display()));
        let n_iter = fit.history.loglike.len();
        if n_iter > 0 {
            ctx.info(format!(
                "To plot it: {} (-s is the bin size, which the .psmc file does not record)",
                psmc_plot_command(fit.cli, n_iter, &path.display().to_string())
            ));
        }
    }
    Ok(())
}

/// Write the HTML report of the fit of `input_file` into `output` to
/// `path`, with the bootstrap and the TMRCA posterior when they were made.
fn write_report(
    fit: &FitRecord,
    path: &Path,
    input_file: &Path,
    output: &Path,
    bootstrap: Option<(&Bootstrap, &BootstrapSummary)>,
    tmrca: Option<&TmrcaReportData>,
) -> Result<()> {
    let cli = fit.cli;
    let started_from = cli.init_params.as_ref().map(|p| p.display().to_string());
    let settings = [
        ("Starting theta0/rho0", Some(fit.tr_ratio.to_string())),
        ("Smoothness penalty", Some(cli.smooth_lambda.to_string())),
        (
            "M-step iterations (at most)",
            Some(cli.mstep_iters.to_string()),
        ),
        ("Started from", started_from),
    ]
    .into_iter()
    .filter_map(|(name, value)| Some((name.to_string(), value?)))
    .collect();
    let report = ReportData {
        model: fit.model,
        bin_size: cli.input.bin_size as f64,
        gen_years: cli.gen_years,
        input_file,
        output_json: output,
        input: Some(fit.input),
        n_iter: fit.history.loglike.len(),
        final_loglike: Some(fit.final_loglike),
        settings,
        qc: fit.qc,
        bin_callable: Some(fit.bin_callable),
        bootstrap: bootstrap.map(|(b, summary)| BootstrapReport {
            summary,
            em_iters: b.plan.cfg.em_iters,
            trunks: Some(b.trunks),
            started_from: cli.init_params.as_deref(),
        }),
        tmrca,
    };
    write_html_report(path, &report)
}

/// Read the input as `cli` says, report how it was read and its QC, and
/// keep the callable bases of its bins only when the fit uses them. QC
/// checks the input for a start of `layout`'s free lambdas and time grid,
/// and for a bootstrap of `trunk` bins (0 without one).
fn read_input(
    path: &Path,
    cli: &InferArgs,
    layout: &PsmcModel,
    trunk: usize,
    ctx: Context,
) -> Result<Input> {
    let io_total = fs::metadata(path).map(|m| m.len()).unwrap_or(1);
    let io_progress = |done| Event::Progress {
        stage: Stage::Io,
        done,
        total: io_total,
        iter: None,
    };
    (ctx.sink)(io_progress(0));
    let mut data = load_input(path, &cli.input.to_options())?;
    (ctx.sink)(io_progress(io_total));
    announce_input(&data, &cli.input, ctx.sink);
    // Whether the fit weighs each bin by its callable bases.
    let bin_callable = bin_callable_record(&data, cli.bin_callable);
    if let Some(event) = bin_callable_event(&bin_callable, &data, cli) {
        (ctx.sink)(event);
    }
    let counted = bin_callable.model == CALLABLE_LENGTH_MODEL;

    // Input QC, printed before the fit, also when no sequence is left (the
    // report then says why). A QC error is only reported, so that it never
    // stops a fit. It sees the counts whether or not the fit uses them.
    let fit_counts = data.obs.callable.as_ref().filter(|_| counted);
    let start = fit_start(cli, layout, &data.obs.rows, fit_counts);
    let report = qc_report(
        &data,
        cli.input.bin_size,
        trunk,
        Some(layout.lam.len()),
        start,
        cli.bin_callable,
    );
    let qc = report_qc(&data, report, cli.qc_json.as_deref(), ctx.sink)?;
    // Without the callable-length model the counts go, so that every pass
    // (theta0, EM, the bootstrap, the decoding) fits C psmc's model.
    if !counted {
        data.obs.callable = None;
    }
    ensure_usable(&data)?;
    Ok(Input {
        data,
        bin_callable,
        qc,
    })
}

/// Say how the input was read: what it holds, and the options of `args`
/// it does not use.
pub fn announce_input(data: &Dataset, args: &InputArgs, sink: Sink) {
    for line in describe(data) {
        sink(Event::Info(line));
    }
    if let Some(line) = ignored_options(args, data) {
        sink(Event::Warning(line));
    }
}

/// Show the input QC `report` of `data`, or why it could not be made,
/// after the loader's warnings that it does not repeat, and write it as
/// JSON to `qc_json` when given. A failed report is only a warning, so that
/// it never stops a run; it is returned as `None`.
pub fn report_qc(
    data: &Dataset,
    report: Result<QcReport>,
    qc_json: Option<&Path>,
    sink: Sink,
) -> Result<Option<QcReport>> {
    let qc = match report {
        Ok(report) => Some(report),
        Err(e) => {
            let unwritten = qc_json.map_or(String::new(), |path| {
                format!(", so {} is not written", path.display())
            });
            sink(Event::Warning(format!("input QC failed{unwritten}: {e:#}")));
            sink(Event::Info(totals(data)));
            None
        }
    };
    for line in loader_warnings(data, qc.as_ref()) {
        sink(Event::Warning(line));
    }
    let Some(report) = &qc else {
        return Ok(None);
    };
    sink(Event::Qc {
        text: render_text(report),
        warnings: report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .map(|f| format!("QC {}: {}", f.code.as_str(), f.message))
            .collect(),
    });
    if let Some(path) = qc_json {
        write_qc_json(path, report)?;
        if !is_stdout(path) {
            sink(Event::Output(format!("QC report: {}", path.display())));
        }
    }
    Ok(qc)
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
    let files: Vec<(&str, &Path)> = [
        ("OUTPUT", Some(output)),
        ("the HTML report", Some(report_html)),
        ("--psmc-out", cli.psmc_out.as_deref()),
        ("--tmrca-out", cli.tmrca_out.as_deref()),
        // Only the QC JSON can go to stdout.
        (
            "--qc-json",
            cli.qc_json.as_deref().filter(|p| !is_stdout(p)),
        ),
    ]
    .into_iter()
    .filter_map(|(what, path)| Some((what, path?)))
    .collect();
    let dir_name = match cli.bootstrap_dir {
        Some(_) => "--bootstrap-dir",
        None => "the bootstrap directory",
    };
    let dir = bootstrap.map(|plan| (dir_name, plan.dir.as_path()));
    check_destinations(input, &files, dir.as_slice())?;
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

/// How the fit of `data` uses the callable bases of its bins under
/// `--bin-callable mode`: the callable-length model when the mode uses the
/// data's counts ([`fit_uses_counts`]: exact counts with `auto`, also
/// multihetsep's approximate ones with `approximate`), C psmc's model
/// otherwise. The counts are summarised whether or not the fit uses them.
fn bin_callable_record(data: &Dataset, mode: BinCallableArg) -> BinCallableRecord {
    let callable = callable_summary(data);
    let counted = callable.is_some() && fit_uses_counts(data, mode);
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

/// What the run says about the callable bases of `data`'s bins: how the
/// fit uses them (the QC report says what they hold), or that a
/// `--bin-callable` other than `auto` changes nothing for input without
/// counts.
fn bin_callable_event(b: &BinCallableRecord, data: &Dataset, cli: &InferArgs) -> Option<Event> {
    let s = cli.input.bin_size;
    let mode = cli.bin_callable.name();
    let counted = b.model == CALLABLE_LENGTH_MODEL;
    let max_counted = count(MAX_COUNTED_BIN_SIZE as u64);
    match (&b.callable, b.counts.as_deref()) {
        // Multihetsep's counts follow the reader's placement of the
        // uncalled sites, which QC's APPROX_MISSING_PLACEMENT describes.
        (Some(_), Some(source)) if source != "exact" => {
            let how = if counted {
                format!(
                    "the fit weighs each called bin by them (--bin-callable {mode}), which made \
                     theta and Ne lower than --bin-callable auto in the validation"
                )
            } else {
                format!(
                    "the fit treats every called bin as {s} bases (--bin-callable {mode}; \
                     approximate would use the counts)"
                )
            };
            Some(Event::Info(format!(
                "Callable bases ({source} counts, see APPROX_MISSING_PLACEMENT): {how}"
            )))
        }
        (Some(_), Some(source)) => {
            let how = if counted {
                format!("the fit weighs each called bin by them (--bin-callable {mode})")
            } else {
                format!(
                    "the fit treats every called bin as {s} bases, as fq2psmcfa and C psmc do \
                     (--bin-callable {mode})"
                )
            };
            Some(Event::Info(format!(
                "Callable bases ({source} counts): {how}"
            )))
        }
        _ if cli.bin_callable != BinCallableArg::Auto => {
            let what = match &data.source {
                SourceSummary::Psmcfa(_) => {
                    "psmcfa input, which records no callable bases per bin".to_string()
                }
                _ => format!(
                    "bins of {s} bases, whose callable bases are not counted (only bins of at \
                     most {max_counted} bases are)"
                ),
            };
            Some(Event::Warning(format!(
                "--bin-callable {mode} changes nothing for {what}: every called bin is taken to \
                 hold {s} bases, as C psmc takes them"
            )))
        }
        _ if !matches!(data.source, SourceSummary::Psmcfa(_)) => Some(Event::Info(format!(
            "Callable bases are counted for bins of at most {max_counted} bases; bins of {s} are \
             fitted as C psmc fits them, every called bin as {s} bases"
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
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
        crate::cli::Cli::try_parse_args(args).unwrap().infer
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
    fn a_set_cancel_flag_stops_the_run() {
        let dir = temp_dir("cancel");
        let cli = cli(&dir, &[]);
        let err = run_inference(&cli, &AtomicBool::new(true), &|_| {}, false).unwrap_err();
        assert!(err.is::<Cancelled>(), "{err:#}");
        assert!(!dir.join("out.json").exists());
        let _ = fs::remove_dir_all(dir);
    }
}
