//! The interactive terminal UI (`--tui`): a form for the main settings, live
//! progress of the shared inference pipeline, and the results.

use anyhow::{Context, Result, anyhow};
use clap::ValueEnum;
use crossterm::cursor::Show;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use std::collections::VecDeque;
use std::io;
use std::panic;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use psmc_rs::hmm::Cancelled;

use crate::cli::{
    DEFAULT_BIN_SIZE, FormatArg, InferArgs, non_negative_f64, positive_f64, positive_usize,
    usage_bail, whole_number,
};
use crate::output::is_stdout;
use crate::run::{self, Event, Stage};

mod picker;
mod view;

use picker::FilePicker;
use view::render_ui;

const MAX_LOG_LINES: usize = 3_000;

/// The labels of the form's fields, in the order of the `FIELD_` indices,
/// and what each holds.
const LABELS: [&str; 10] = [
    "Input",
    "Output",
    "N iter",
    "Pattern",
    "Gen years",
    "Threads",
    "Input fmt",
    "Bin size",
    "Mu",
    "Smooth",
];
const HINTS: [&str; LABELS.len()] = [
    "Input file: psmcfa, multihetsep or VCF, plain or gzip",
    "Output JSON of the parameters; the HTML report is written next to it",
    "EM iterations (C psmc -N; 0 writes the starting model)",
    "Free-parameter pattern of the time intervals (empty: C psmc's 4+5*3+4)",
    "Years per generation, for the time axes",
    "Worker threads (empty: one per core); results do not depend on it",
    "Input format: auto, psmcfa, mhs or vcf",
    "Bases per bin: bins VCF and multihetsep; for psmcfa, the -s it was made with",
    "Mutation rate per base pair per generation, for scaling",
    "Weight of the penalty on log-lambda differences (0: C psmc's objective)",
];
const FIELD_INPUT: usize = 0;
const FIELD_OUTPUT: usize = 1;
const FIELD_N_ITER: usize = 2;
const FIELD_PATTERN: usize = 3;
const FIELD_GEN_YEARS: usize = 4;
const FIELD_THREADS: usize = 5;
const FIELD_INPUT_FORMAT: usize = 6;
const FIELD_BIN_SIZE: usize = 7;
const FIELD_MU: usize = 8;
const FIELD_SMOOTH_LAMBDA: usize = 9;

/// The pages, in the order of their tabs (F1 to F3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Page {
    Setup,
    Run,
    Result,
}

/// Every stage, in the order of [`Stage`]'s variants, so that `stage as
/// usize` indexes [`App::stages`].
const STAGES: [Stage; 5] = [
    Stage::Io,
    Stage::EForward,
    Stage::EBackward,
    Stage::MStep,
    Stage::Bootstrap,
];

/// The progress of a stage: since when it has made it, and when it last
/// did (which keeps the rate of a finished stage).
#[derive(Debug, Clone, Default)]
struct StageSnapshot {
    done: u64,
    total: u64,
    iter: Option<(usize, usize)>,
    started_at: Option<Instant>,
    updated_at: Option<Instant>,
}

/// A run on its worker thread, which sends the pipeline's events with the
/// time it sent them, and returns its outcome.
struct Running {
    events: Receiver<(Instant, Event)>,
    worker: JoinHandle<Result<()>>,
    stop: Arc<AtomicBool>,
    started: Instant,
}

struct App {
    self_bin: PathBuf,
    /// The command line that opened the TUI, which gives the settings the
    /// form does not show.
    base_cli: InferArgs,
    page: Page,
    fields: [String; LABELS.len()],
    selected: usize,
    logs: VecDeque<String>,
    /// How many rows the log is scrolled back from its end.
    scroll_back: usize,
    status: String,
    should_quit: bool,
    running: Option<Running>,
    stages: [StageSnapshot; STAGES.len()],
    /// The result lines of the last run: the files it wrote.
    results: Vec<String>,
    picker: Option<FilePicker>,
}

impl App {
    /// The form filled from `cli`, the command line that opened the TUI.
    fn new(self_bin: PathBuf, cli: &InferArgs) -> Self {
        let path = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string());
        let fields = [
            path(&cli.input_file).unwrap_or_default(),
            path(&cli.output_file).unwrap_or_else(|| "out.json".to_string()),
            cli.n_iter.unwrap_or(25).to_string(),
            cli.pattern.clone().unwrap_or_default(),
            cli.gen_years.to_string(),
            cli.threads.map(|n| n.to_string()).unwrap_or_default(),
            cli.input.input_format.name().to_string(),
            cli.input.bin_size.to_string(),
            format!("{:e}", cli.mu),
            cli.smooth_lambda.to_string(),
        ];
        Self {
            self_bin,
            base_cli: cli.clone(),
            page: Page::Setup,
            fields,
            selected: 0,
            logs: VecDeque::new(),
            scroll_back: 0,
            status: "Ready".to_string(),
            should_quit: false,
            running: None,
            stages: Default::default(),
            results: Vec::new(),
            picker: None,
        }
    }

    fn append_log(&mut self, line: impl Into<String>) {
        if self.logs.len() >= MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(line.into());
    }

    /// The next input format in the format field.
    fn cycle_input_format(&mut self) {
        let all = FormatArg::value_variants();
        let next = parse_input_format(&self.fields[FIELD_INPUT_FORMAT])
            .ok()
            .and_then(|f| all.iter().position(|&x| x == f))
            .map_or(FormatArg::Auto, |i| all[(i + 1) % all.len()]);
        self.fields[FIELD_INPUT_FORMAT] = next.name().to_string();
    }
}

/// The value of the field `name`, read by `parse`, a parser of the command
/// line's values; `None` when it is empty.
fn parsed<T>(name: &str, value: &str, parse: fn(&str) -> Result<T, String>) -> Result<Option<T>> {
    match value.trim() {
        "" => Ok(None),
        v => parse(v).map(Some).map_err(|e| anyhow!("{name}: {e}")),
    }
}

/// [`parsed`] for a field that must be filled.
fn required<T>(name: &str, value: &str, parse: fn(&str) -> Result<T, String>) -> Result<T> {
    parsed(name, value, parse)?.ok_or_else(|| anyhow!("{name} is required"))
}

fn parse_input_format(value: &str) -> Result<FormatArg> {
    FormatArg::from_str(value.trim(), true)
        .map_err(|_| anyhow!("Input fmt must be auto, psmcfa, mhs or vcf"))
}

/// The run the form describes: its fields over the other settings of the
/// command line that opened the TUI.
fn build_cli_from_form(app: &App) -> Result<InferArgs> {
    let field = |idx: usize| app.fields[idx].as_str();
    let text = |s: &str| Ok::<_, String>(s.to_string());
    let mut cli = app.base_cli.clone();
    cli.tui = false;
    cli.input_file = Some(required("Input", field(FIELD_INPUT), text)?.into());
    cli.output_file = Some(required("Output", field(FIELD_OUTPUT), text)?.into());
    cli.n_iter = Some(required("N iter", field(FIELD_N_ITER), whole_number)?);
    cli.pattern = parsed("Pattern", field(FIELD_PATTERN), text)?;
    cli.gen_years = required("Gen years", field(FIELD_GEN_YEARS), positive_f64)?;
    cli.threads = parsed("Threads", field(FIELD_THREADS), positive_usize)?;
    cli.input.input_format = parse_input_format(field(FIELD_INPUT_FORMAT))?;
    cli.input.bin_size =
        parsed("Bin size", field(FIELD_BIN_SIZE), positive_usize)?.unwrap_or(DEFAULT_BIN_SIZE);
    cli.mu = required("Mu", field(FIELD_MU), positive_f64)?;
    cli.smooth_lambda = required("Smooth", field(FIELD_SMOOTH_LAMBDA), non_negative_f64)?;
    Ok(cli)
}

/// The command line of the run the form describes, or why there is none.
fn command_line(app: &App) -> Result<String> {
    let cli = build_cli_from_form(app)?;
    Ok(format!(
        "{} {}",
        app.self_bin.display(),
        cli.to_args().join(" ")
    ))
}

/// Take the events of the run under way and, once it has ended, its
/// outcome.
fn poll_runner(app: &mut App) {
    let Some(run) = &app.running else {
        return;
    };
    // Every event of a finished worker is in the channel.
    let ended = run.worker.is_finished();
    let events: Vec<(Instant, Event)> = run.events.try_iter().collect();
    for (at, ev) in events {
        handle_pipeline_event(app, at, ev);
    }
    let Some(run) = app.running.take_if(|_| ended) else {
        return;
    };
    let outcome = run
        .worker
        .join()
        .unwrap_or_else(|_| Err(anyhow!("the run stopped unexpectedly")));
    match outcome {
        Ok(()) => {
            app.status = format!("Done in {:.1}s", run.started.elapsed().as_secs_f64());
            app.page = Page::Result;
        }
        Err(e) if e.is::<Cancelled>() => {
            app.status = "Cancelled".to_string();
            app.append_log("Run cancelled by user.");
        }
        Err(e) => {
            // Back to the form, to fix what the error names.
            app.status = "Failed: see the log".to_string();
            app.append_log(format!("Run failed: {e:#}"));
            app.page = Page::Setup;
        }
    }
}

/// Show an event of the pipeline, sent `at`: messages in the log, progress
/// in the stage gauges, results on the result page too.
fn handle_pipeline_event(app: &mut App, at: Instant, ev: Event) {
    match ev {
        Event::Info(line) => app.append_log(line),
        Event::Output(line) => {
            app.results.push(line.clone());
            app.append_log(line);
        }
        Event::Warning(line) => app.append_log(format!("Warning: {line}")),
        Event::Qc { text, .. } => text.lines().for_each(|line| app.append_log(line)),
        Event::Iteration { iter, total } => app.append_log(format!("EM iteration {iter}/{total}")),
        Event::Progress {
            stage,
            done,
            total,
            iter,
        } => {
            // A stage starts again when its count goes back or its EM
            // iteration changes. (The M-step's length is its cap until its
            // last event.)
            let snap = &mut app.stages[stage as usize];
            if snap.started_at.is_none() || done < snap.done || iter != snap.iter {
                snap.started_at = Some(at);
            }
            (snap.done, snap.total, snap.iter) = (done.min(total), total, iter);
            snap.updated_at = Some(at);
        }
    }
}

fn start_run(app: &mut App) -> Result<()> {
    if app.running.is_some() {
        return Ok(());
    }
    let cli = build_cli_from_form(app)?;
    app.append_log(format!("$ {}", command_line(app)?));
    app.stages = Default::default();
    app.results.clear();
    app.scroll_back = 0;
    app.page = Page::Run;
    app.status = "Running".to_string();

    let (tx, events) = mpsc::channel();
    let stop = Arc::new(AtomicBool::new(false));
    let cancel = Arc::clone(&stop);
    let worker = thread::spawn(move || {
        run::run_inference(
            &cli,
            &cancel,
            &|ev| _ = tx.send((Instant::now(), ev)),
            false,
        )
    });
    app.running = Some(Running {
        events,
        worker,
        stop,
        started: Instant::now(),
    });
    Ok(())
}

fn stop_run(app: &mut App) {
    if let Some(run) = &app.running {
        run.stop.store(true, Ordering::Relaxed);
        app.status = "Stopping...".to_string();
        app.append_log("Stop requested.");
    }
}

fn handle_key(app: &mut App, key: KeyEvent) -> Result<()> {
    if key.kind != KeyEventKind::Press {
        return Ok(());
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    if ctrl && key.code == KeyCode::Char('c') {
        match app.running {
            Some(_) => stop_run(app),
            None => app.should_quit = true,
        }
        return Ok(());
    }
    if app.picker.is_some() {
        return app.picker_key(key.code);
    }
    // While the form can be edited, it takes every printable character
    // (paths such as "results/run.json" hold the command letters), except
    // Space on the format field, which changes it. Ctrl or Alt makes a
    // command; both together is AltGr on Windows, which types characters
    // such as '@'.
    let editing = app.page == Page::Setup && app.running.is_none();
    let plain = ctrl == key.modifiers.contains(KeyModifiers::ALT);
    let on_format = editing && app.selected == FIELD_INPUT_FORMAT;
    let field = &mut app.fields[app.selected];
    match key.code {
        KeyCode::Char(' ') | KeyCode::Left | KeyCode::Right if on_format && plain => {
            app.cycle_input_format();
        }
        KeyCode::Char(c) if editing && plain => field.push(c),
        KeyCode::Backspace if editing => _ = field.pop(),
        KeyCode::Char('u') if editing && ctrl => field.clear(),
        KeyCode::Char('o') if editing && ctrl => app.open_picker(FIELD_INPUT)?,
        KeyCode::Char('d') if editing && ctrl => app.open_picker(FIELD_OUTPUT)?,
        KeyCode::Tab | KeyCode::Down if editing => app.selected = (app.selected + 1) % LABELS.len(),
        KeyCode::BackTab | KeyCode::Up if editing => {
            app.selected = (app.selected + LABELS.len() - 1) % LABELS.len();
        }
        KeyCode::Char('q') | KeyCode::Esc if app.running.is_some() => {
            app.status = "Stop the run first (x)".to_string();
        }
        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
        KeyCode::Char('x') => stop_run(app),
        KeyCode::F(5) | KeyCode::Enter | KeyCode::Char('r') => start_run(app)?,
        KeyCode::F(n @ 1..=3) => app.page = [Page::Setup, Page::Run, Page::Result][n as usize - 1],
        KeyCode::Up => app.scroll_back += 1,
        KeyCode::PageUp => app.scroll_back += 10,
        KeyCode::Down => app.scroll_back = app.scroll_back.saturating_sub(1),
        KeyCode::PageDown => app.scroll_back = app.scroll_back.saturating_sub(10),
        KeyCode::End => app.scroll_back = 0,
        KeyCode::Char('l') if ctrl => {
            app.logs.clear();
            app.scroll_back = 0;
        }
        _ => {}
    }
    Ok(())
}

/// Settings the TUI cannot honour: stdout is the interface. Like the
/// option combinations the CLI refuses before reading the input, they are
/// usage errors.
fn check_tui_args(cli: &InferArgs) -> Result<()> {
    if cli.qc_json.as_deref().is_some_and(is_stdout) {
        usage_bail!("--qc-json - would print over the interface; give --qc-json a file with --tui");
    }
    Ok(())
}

/// The terminal in raw mode on the alternate screen, given back as it was
/// when dropped, also when the TUI stops with an error or a panic (the
/// release build aborts on a panic without dropping it, but runs the panic
/// hook first).
struct Screen(Terminal<CrosstermBackend<io::Stdout>>);

impl Screen {
    fn enter() -> Result<Self> {
        // Raw mode first: it opens the terminal, so without one it fails
        // with the reason, where the size query of Terminal::new gives a
        // stale one.
        enable_raw_mode().context("failed to enable raw mode")?;
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
            .inspect_err(|_| {
                let _ = disable_raw_mode();
            })
            .context("failed to initialize terminal")?;
        let mut screen = Screen(terminal);
        let hook = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            restore_terminal();
            hook(info);
        }));
        execute!(io::stdout(), EnterAlternateScreen).context("failed to enter alternate screen")?;
        screen.0.clear().context("failed to clear terminal")?;
        Ok(screen)
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Raw mode off, the main screen and the cursor back. ratatui::restore
/// leaves the cursor hidden, as every frame of the TUI hides it.
fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, Show);
}

/// Run the TUI until the user quits; `cli` fills the form.
pub fn run_tui(cli: &InferArgs) -> Result<()> {
    check_tui_args(cli)?;
    let self_bin = std::env::current_exe().context("failed to resolve executable path")?;
    let mut app = App::new(self_bin, cli);
    let mut screen = Screen::enter()?;
    let result = event_loop(&mut app, &mut screen.0);
    stop_run(&mut app);
    if let Some(run) = app.running.take() {
        let _ = run.worker.join();
    }
    result
}

/// Draw the interface and handle the terminal's events until the user
/// quits.
fn event_loop(app: &mut App, term: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    while !app.should_quit {
        poll_runner(app);
        term.draw(|f| render_ui(f, app)).context("draw failed")?;
        if !event::poll(Duration::from_millis(80)).context("event poll failed")? {
            continue;
        }
        match event::read().context("failed to read terminal event")? {
            TermEvent::Key(key) => {
                if let Err(e) = handle_key(app, key) {
                    app.status = format!("{e:#}");
                    app.append_log(format!("Input error: {e:#}"));
                }
            }
            TermEvent::Resize(..) => _ = term.autoresize(),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, UsageError};
    use std::fs;

    /// The TUI opened with `args`.
    pub(super) fn app(args: &[&str]) -> App {
        let argv = ["psmc-rs", "--tui"].iter().chain(args).copied();
        let cli = Cli::try_parse_args(argv).unwrap();
        App::new(PathBuf::from("psmc-rs"), &cli.infer)
    }

    pub(super) fn set(app: &mut App, field: usize, value: &str) {
        app.fields[field] = value.to_string();
    }

    /// The form fills the settings it shows and keeps the others of the
    /// command line that opened the TUI.
    #[test]
    fn the_form_builds_the_run_settings() {
        let mut a = app(&["--bootstrap", "5", "--vcf-mask", "m.bed"]);
        assert_eq!(a.fields[FIELD_INPUT_FORMAT], "auto");
        assert_eq!(a.fields[FIELD_GEN_YEARS], "25");
        assert_eq!(a.fields[FIELD_MU], "2.5e-8");
        for (field, value) in [
            (FIELD_INPUT, "data/in.vcf.gz"),
            (FIELD_OUTPUT, "out/fit.json"),
            (FIELD_N_ITER, "30"),
            (FIELD_PATTERN, "4+25*2+4+6"),
            (FIELD_GEN_YEARS, "29"),
            (FIELD_THREADS, "2"),
            (FIELD_INPUT_FORMAT, "vcf"),
            (FIELD_BIN_SIZE, "50"),
            (FIELD_MU, "1.25e-8"),
            (FIELD_SMOOTH_LAMBDA, "0"),
        ] {
            set(&mut a, field, value);
        }
        let args = build_cli_from_form(&a).unwrap();
        assert!(!args.tui);
        assert_eq!(args.input_file, Some(PathBuf::from("data/in.vcf.gz")));
        assert_eq!(args.output_file, Some(PathBuf::from("out/fit.json")));
        assert_eq!(args.n_iter, Some(30));
        assert_eq!(args.pattern.as_deref(), Some("4+25*2+4+6"));
        assert_eq!(args.gen_years, 29.0);
        assert_eq!(args.threads, Some(2));
        assert_eq!(args.input.input_format, FormatArg::Vcf);
        assert_eq!(args.input.bin_size, 50);
        assert_eq!((args.mu, args.smooth_lambda), (1.25e-8, 0.0));
        assert_eq!(args.bootstrap, 5);
        assert_eq!(args.input.vcf_mask, [PathBuf::from("m.bed")]);
        let command = command_line(&a).unwrap();
        assert!(command.contains("--gen-years 29"), "{command}");
        assert!(command.contains("--bootstrap 5"), "{command}");

        // An empty pattern or thread count means the default.
        set(&mut a, FIELD_PATTERN, " ");
        set(&mut a, FIELD_THREADS, "");
        let args = build_cli_from_form(&a).unwrap();
        assert_eq!((args.pattern, args.threads), (None, None));
    }

    #[test]
    fn the_qc_json_cannot_go_to_the_terminal_of_the_interface() {
        let cli = |args: &[&str]| app(args).base_cli;
        let err = check_tui_args(&cli(&["--qc-json", "-"])).unwrap_err();
        assert!(err.to_string().contains("give --qc-json a file"), "{err}");
        assert!(UsageError::is_in(&err), "{err}");
        assert!(check_tui_args(&cli(&["--qc-json", "qc.json"])).is_ok());
        assert!(check_tui_args(&cli(&[])).is_ok());
    }

    #[test]
    fn invalid_form_values_are_explained() {
        let mut a = app(&[]);
        let err = |a: &App| command_line(a).unwrap_err().to_string();
        assert_eq!(err(&a), "Input is required");
        set(&mut a, FIELD_INPUT, "in.psmcfa");
        set(&mut a, FIELD_GEN_YEARS, "old");
        assert_eq!(err(&a), "Gen years: 'old' is not a number");
        set(&mut a, FIELD_GEN_YEARS, "25");
        set(&mut a, FIELD_INPUT_FORMAT, "bam");
        assert!(err(&a).contains("auto, psmcfa, mhs or vcf"));
    }

    fn key(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
        handle_key(app, KeyEvent::new(code, modifiers)).unwrap();
    }

    fn press(app: &mut App, code: KeyCode) {
        key(app, code, KeyModifiers::NONE);
    }

    /// The form takes every printable character, the command letters
    /// included; Space changes the format field.
    #[test]
    fn typed_text_reaches_the_form() {
        let mut a = app(&[]);
        a.selected = FIELD_OUTPUT;
        key(&mut a, KeyCode::Char('u'), KeyModifiers::CONTROL);
        for c in "results/req x.json".chars() {
            press(&mut a, KeyCode::Char(c));
        }
        assert_eq!(a.fields[FIELD_OUTPUT], "results/req x.json");
        assert!(!a.should_quit && a.running.is_none());
        assert_eq!(a.page, Page::Setup);
        press(&mut a, KeyCode::Backspace);
        assert_eq!(a.fields[FIELD_OUTPUT], "results/req x.jso");

        a.selected = FIELD_INPUT_FORMAT;
        press(&mut a, KeyCode::Char(' '));
        assert_eq!(a.fields[FIELD_INPUT_FORMAT], "psmcfa");
        press(&mut a, KeyCode::Right);
        assert_eq!(a.fields[FIELD_INPUT_FORMAT], "mhs");

        // Ctrl chords are commands, not text.
        a.selected = FIELD_INPUT;
        a.logs.push_back("old".to_string());
        key(&mut a, KeyCode::Char('l'), KeyModifiers::CONTROL);
        assert!(a.logs.is_empty());
        assert_eq!(a.fields[FIELD_INPUT], "");
        key(
            &mut a,
            KeyCode::Char('@'),
            KeyModifiers::CONTROL | KeyModifiers::ALT,
        );
        assert_eq!(a.fields[FIELD_INPUT], "@");
    }

    /// A run whose worker returns `outcome` at once.
    fn finished_run(outcome: Result<()>) -> (Running, mpsc::Sender<(Instant, Event)>) {
        let (tx, events) = mpsc::channel();
        let run = Running {
            events,
            worker: thread::spawn(|| outcome),
            stop: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        };
        (run, tx)
    }

    /// Esc quits from the form; q quits from the other pages; neither while
    /// a run is going, which x stops.
    #[test]
    fn quit_and_stop_keys() {
        let mut a = app(&[]);
        press(&mut a, KeyCode::Esc);
        assert!(a.should_quit);

        let mut a = app(&[]);
        a.page = Page::Result;
        press(&mut a, KeyCode::Char('q'));
        assert!(a.should_quit);

        let mut a = app(&[]);
        a.running = Some(finished_run(Ok(())).0);
        a.page = Page::Run;
        press(&mut a, KeyCode::Char('q'));
        press(&mut a, KeyCode::Esc);
        assert!(!a.should_quit);
        press(&mut a, KeyCode::Char('x'));
        assert!(a.running.as_ref().unwrap().stop.load(Ordering::Relaxed));
        assert_eq!(a.status, "Stopping...");
    }

    #[test]
    fn the_format_field_cycles_through_every_format() {
        let mut a = app(&[]);
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(a.fields[FIELD_INPUT_FORMAT].clone());
            a.cycle_input_format();
        }
        assert_eq!(seen, ["auto", "psmcfa", "mhs", "vcf", "auto"]);
    }

    #[test]
    fn pipeline_events_reach_the_log_the_gauges_and_the_results() {
        let mut a = app(&[]);
        for ev in [
            Event::Info("read".to_string()),
            Event::Warning("odd".to_string()),
            Event::Qc {
                text: "Input QC\nNo warnings or notes.\n".to_string(),
                warnings: Vec::new(),
            },
            Event::Iteration { iter: 2, total: 9 },
            Event::Progress {
                stage: Stage::EForward,
                done: 40,
                total: 100,
                iter: Some((2, 9)),
            },
            Event::Output("Parameters: x".to_string()),
        ] {
            handle_pipeline_event(&mut a, Instant::now(), ev);
        }
        assert_eq!(
            a.logs,
            [
                "read",
                "Warning: odd",
                "Input QC",
                "No warnings or notes.",
                "EM iteration 2/9",
                "Parameters: x"
            ]
        );
        assert_eq!(a.results, ["Parameters: x"]);
        let snap = &a.stages[Stage::EForward as usize];
        assert_eq!((snap.done, snap.total, snap.iter), (40, 100, Some((2, 9))));
    }

    /// The end of a run: its outcome in the status, the results page after
    /// a finished run, the form after a failed one.
    #[test]
    fn the_end_of_a_run_is_shown() {
        let ended = |outcome: Result<()>| {
            let mut a = app(&[]);
            let (run, tx) = finished_run(outcome);
            let ev = Event::Output("Parameters: x".to_string());
            tx.send((Instant::now(), ev)).unwrap();
            a.running = Some(run);
            a.page = Page::Run;
            while a.running.is_some() {
                poll_runner(&mut a);
            }
            a
        };
        let a = ended(Err(Cancelled.into()));
        assert_eq!(a.status, "Cancelled");
        assert_eq!(a.logs, ["Parameters: x", "Run cancelled by user."]);
        let a = ended(Ok(()));
        assert!(a.status.starts_with("Done in"), "{}", a.status);
        assert_eq!(a.page, Page::Result);
        assert_eq!(a.results, ["Parameters: x"]);
        let a = ended(Err(anyhow!("no input")));
        assert_eq!(a.page, Page::Setup);
        assert_eq!(a.logs.back().unwrap(), "Run failed: no input");
    }

    /// The stop key interrupts a run inside EM: 100,000 iterations would
    /// take far longer than the test allows.
    #[test]
    fn stop_cancels_a_running_fit() {
        let dir = std::env::temp_dir().join(format!("psmc_rs_tui_stop_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.psmcfa");
        let bins = "TTTTKTTTTTTTNNTTKTTTTTTTTKTTTTTTTTTTKTTTTTTTTTKTT".repeat(40);
        fs::write(&input, format!(">a\n{bins}\n>b\n{bins}\n")).unwrap();
        let mut a = app(&[]);
        set(&mut a, FIELD_INPUT, input.to_str().unwrap());
        set(&mut a, FIELD_OUTPUT, dir.join("fit.json").to_str().unwrap());
        set(&mut a, FIELD_N_ITER, "100000");
        set(&mut a, FIELD_THREADS, "2");
        start_run(&mut a).unwrap();
        assert_eq!(a.page, Page::Run);

        let started = Instant::now();
        while !a.logs.iter().any(|l| l.starts_with("EM iteration 3/")) {
            poll_runner(&mut a);
            assert!(a.running.is_some(), "the run ended early: {:?}", a.logs);
            assert!(started.elapsed() < Duration::from_secs(60));
            thread::sleep(Duration::from_millis(5));
        }
        stop_run(&mut a);
        let stopped = Instant::now();
        while a.running.is_some() {
            poll_runner(&mut a);
            assert!(stopped.elapsed() < Duration::from_secs(30));
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(a.status, "Cancelled", "{:?}", a.logs);
        assert!(!dir.join("fit.json").exists());
        let _ = fs::remove_dir_all(dir);
    }
}
