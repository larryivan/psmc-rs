//! The inference pipeline, shared by the command line and the TUI: read the
//! input, fit, bootstrap, decode and write the outputs, reporting progress
//! and messages as [`Event`]s.

mod bootstrap;
mod fit;
mod start;

use anyhow::{Context as _, Result, anyhow};
use rayon::ThreadPool;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::BootstrapSummary;
use psmc_rs::em::EmHistory;
use psmc_rs::hmm::{Cancelled, TmrcaReportData};
use psmc_rs::io::MAX_COUNTED_BIN_SIZE;
use psmc_rs::io::input::{Dataset, SourceSummary};
use psmc_rs::io::params::{
    BinCallableRecord, CALLABLE_LENGTH_MODEL, FULL_BINS_MODEL, InputSummary,
};
use psmc_rs::opt::MStepConfig;
use psmc_rs::qc::{QcReport, Severity, render_text};
use psmc_rs::report::{
    BootstrapReport, ReportData, check_echarts_bundle, default_html_path, write_html_report,
};

use self::bootstrap::{Bootstrap, BootstrapPlan, earlier_bootstrap_warning};
use self::fit::{decode_tmrca, fit, model_loglike, near_bounds_warning};
use self::start::{check_starts, fit_start, saved_start, starting_model};
use crate::cli::{BinCallableArg, InferArgs, InputArgs, model_layout, usage_bail};
use crate::dataset::{
    callable_summary, counted, counts_source, describe, ensure_usable, fit_uses_counts,
    ignored_options, layout_model, load_input, loader_warnings, qc_report, resolve_input_path,
    thousands, totals,
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
    let ctx = Context {
        cancel,
        sink,
        pool: &pool,
    };
    pool.install(|| infer(cli, ctx, progress_bars))
}

/// The settings of a run that are checked before its input is read.
struct Plan {
    input: PathBuf,
    output: PathBuf,
    n_iter: usize,
    /// The parameters of `--init-params`.
    saved: Option<PsmcModel>,
    /// `n_steps` and the pattern of a fresh start.
    fresh: (usize, Option<String>),
    /// The layout of the starting model: its free lambdas, and the time grid
    /// on which QC checks the start.
    layout: PsmcModel,
    bootstrap: Option<BootstrapPlan>,
}

impl Plan {
    /// The settings of `cli`; those that can be checked without the data
    /// fail first.
    fn of(cli: &InferArgs) -> Result<Self> {
        let missing = |what| anyhow!("missing {what}");
        let input = cli
            .input_file
            .clone()
            .ok_or_else(|| missing("INPUT_FILE"))?;
        let output = cli
            .output_file
            .clone()
            .ok_or_else(|| missing("OUTPUT_FILE"))?;
        let n_iter = cli.n_iter.ok_or_else(|| missing("N_ITER"))?;
        let saved = cli
            .init_params
            .as_deref()
            .map(|path| saved_start(path, cli))
            .transpose()?;
        let (n_steps, pattern) = model_layout(cli.pattern.as_deref(), cli.n_steps)?;
        let layout = match &saved {
            Some(model) => model.clone(),
            None => layout_model(cli.t_max, n_steps, pattern.clone(), cli.mu)?,
        };
        let bootstrap = BootstrapPlan::of(cli, n_iter, &output)?;
        Ok(Self {
            input,
            output,
            n_iter,
            saved,
            fresh: (n_steps, pattern),
            layout,
            bootstrap,
        })
    }
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
fn infer(cli: &InferArgs, ctx: Context, progress_bars: bool) -> Result<RunSummary> {
    let Plan {
        input: given_input,
        output,
        n_iter,
        saved,
        fresh,
        layout,
        bootstrap,
    } = Plan::of(cli)?;
    ctx.check_cancel()?;
    let input_file = resolve_input_path(&given_input, cli.input.input_format)?;
    if input_file != given_input {
        ctx.info(format!(
            "Input is a directory, selected file: {}",
            input_file.display()
        ));
    }
    let report_html = default_html_path(&output);
    check_outputs(cli, &input_file, &output, &report_html, bootstrap.as_ref())?;
    let trunk = bootstrap.as_ref().map_or(0, |plan| plan.trunk_size);
    let input = read_input(&input_file, cli, &layout, trunk, ctx)?;
    let obs = &input.data.obs;
    let resampled = bootstrap
        .as_ref()
        .map(|plan| plan.resample(obs, cli, ctx))
        .transpose()?;

    let mut model = starting_model(saved, fresh, obs, cli)?;
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
    let input_summary = InputSummary::of(&input.data);
    let fitted = Fitted {
        model: &model,
        history: &history,
        n_iter,
        final_loglike,
        tr_ratio,
        input: &input,
        input_summary: &input_summary,
    };
    write_fit(&fitted, &output, cli, boot.is_none(), ctx)?;
    let boot_summary = match &boot {
        Some(b) => {
            let scale = (cli.input.bin_size as f64, cli.gen_years);
            let summary = b.run(&model, &mstep_config, scale, ctx)?;
            if let Some(msg) = b.plan.plot_warning(cli, n_iter) {
                ctx.warn(msg);
            }
            Some(summary)
        }
        None => None,
    };

    ctx.check_cancel()?;
    let tmrca = cli
        .tmrca_out
        .as_deref()
        .map(|path| decode_tmrca(&model, obs, path, cli, progress_bars, ctx))
        .transpose()?;
    let bootstrap = boot.as_ref().zip(boot_summary.as_ref());
    write_report(
        &fitted,
        &report_html,
        &input_file,
        &output,
        cli,
        bootstrap,
        tmrca.as_ref(),
    )?;
    ctx.output(format!("HTML report: {}", report_html.display()));

    Ok(RunSummary {
        output_json: output,
        html_report: report_html,
        tmrca_out: cli.tmrca_out.clone(),
        bootstrap_dir: boot.map(|b| b.plan.dir),
        final_loglike: Some(final_loglike),
    })
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

/// A finished main fit, with what its outputs record of it.
struct Fitted<'a> {
    model: &'a PsmcModel,
    history: &'a EmHistory,
    /// EM iterations (N_ITER).
    n_iter: usize,
    /// Log-likelihood of `model`: the history's last, or the start's after
    /// 0 iterations.
    final_loglike: f64,
    /// theta0/rho0 of the start.
    tr_ratio: f64,
    input: &'a Input,
    input_summary: &'a InputSummary,
}

/// Write the main fit's parameters to `output`, and its `.psmc` output when
/// asked for; without a bootstrap (`no_bootstrap`), warn of the replicates
/// of an earlier run in the default bootstrap directory.
fn write_fit(
    fitted: &Fitted,
    output: &Path,
    cli: &InferArgs,
    no_bootstrap: bool,
    ctx: Context,
) -> Result<()> {
    let input = fitted.input;
    let record = FitRecord {
        model: fitted.model,
        history: fitted.history,
        final_loglike: Some(fitted.final_loglike),
        bin_size: cli.input.bin_size,
        gen_years: cli.gen_years,
        tr_ratio: fitted.tr_ratio,
        smooth_lambda: cli.smooth_lambda,
        input: fitted.input_summary,
        qc: input.qc.as_ref(),
        bin_callable: Some(&input.bin_callable),
    };
    save_fit(output, &record)?;
    ctx.output(format!("Parameters: {}", output.display()));
    if no_bootstrap && let Some(msg) = earlier_bootstrap_warning(output) {
        ctx.warn(msg);
    }
    if let Some(path) = &cli.psmc_out {
        write_main_psmc(
            path,
            fitted.model,
            fitted.history,
            &input.data,
            &input.bin_callable,
        )?;
        ctx.output(format!("PSMC output: {}", path.display()));
        if fitted.n_iter > 0 {
            ctx.info(format!(
                "To plot it: {} (-s is the bin size, which the .psmc file does not record)",
                psmc_plot_command(cli, fitted.n_iter, &path.display().to_string())
            ));
        }
    }
    Ok(())
}

/// Write the HTML report of the fit to `path`, with the bootstrap and the
/// TMRCA posterior when they were made.
fn write_report(
    fitted: &Fitted,
    path: &Path,
    input_file: &Path,
    output: &Path,
    cli: &InferArgs,
    bootstrap: Option<(&Bootstrap, &BootstrapSummary)>,
    tmrca: Option<&TmrcaReportData>,
) -> Result<()> {
    let started_from = cli
        .init_params
        .as_ref()
        .map(|path| path.display().to_string());
    let settings = [
        ("Starting theta0/rho0", Some(fitted.tr_ratio.to_string())),
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
        model: fitted.model,
        bin_size: cli.input.bin_size as f64,
        gen_years: cli.gen_years,
        input_file,
        output_json: output,
        input: Some(fitted.input_summary),
        n_iter: fitted.n_iter,
        final_loglike: Some(fitted.final_loglike),
        settings,
        qc: fitted.input.qc.as_ref(),
        bin_callable: Some(&fitted.input.bin_callable),
        bootstrap: bootstrap.map(|(b, summary)| BootstrapReport {
            summary,
            em_iters: b.plan.em_iters,
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
    let max_counted = thousands(MAX_COUNTED_BIN_SIZE as u64);
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
