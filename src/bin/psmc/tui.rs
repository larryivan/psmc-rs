//! The interactive terminal UI (`--tui`): a form for the main settings, live
//! progress of the shared inference pipeline, and the results.

use anyhow::{Context, Result, anyhow, bail};
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
use std::path::{Path, PathBuf};
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
use crate::run::{self, Event, RunSummary, Stage};

mod picker;
mod view;

use picker::FilePicker;
use view::render_ui;

const MAX_LOG_LINES: usize = 3_000;

// The fields of the form, in the order of [`App::new`].
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

struct FormField {
    label: &'static str,
    value: String,
    hint: &'static str,
}

/// The pages, in the order of their tabs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TuiPage {
    Setup,
    Run,
    Result,
}

/// Every stage, in the order of [`Stage`]'s variants, so that `stage as
/// usize` indexes [`App::stages`].
const STAGE_ORDER: [Stage; 5] = [
    Stage::Io,
    Stage::EForward,
    Stage::EBackward,
    Stage::MStep,
    Stage::Bootstrap,
];

#[derive(Debug, Clone, Default)]
struct StageSnapshot {
    done: u64,
    total: u64,
    iter: Option<(usize, usize)>,
    started_at: Option<Instant>,
    updated_at: Option<Instant>,
}

/// A finished run, as the result page shows it.
#[derive(Debug, Clone)]
struct Finished {
    summary: RunSummary,
    elapsed_sec: f64,
}

/// What the worker thread of a run sends to the UI.
#[derive(Debug)]
enum RunnerEvent {
    Pipeline(Event),
    Completed(Finished),
    Failed(String),
    Cancelled,
}

struct RunningTask {
    rx: Receiver<RunnerEvent>,
    handle: JoinHandle<()>,
    stop: Arc<AtomicBool>,
}

struct App {
    self_bin: PathBuf,
    base_cli: InferArgs,
    page: TuiPage,
    fields: Vec<FormField>,
    selected: usize,
    logs: VecDeque<String>,
    status: String,
    should_quit: bool,
    autoscroll: bool,
    manual_scroll: u16,
    running: Option<RunningTask>,
    stages: [StageSnapshot; STAGE_ORDER.len()],
    result: Option<Finished>,
    picker: Option<FilePicker>,
}

impl App {
    /// The form filled from `cli`, the command line that opened the TUI.
    fn new(self_bin: PathBuf, cli: &InferArgs) -> Self {
        let path = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string());
        let fields = [
            (
                "Input",
                path(&cli.input_file).unwrap_or_default(),
                "Input file (.psmcfa/.mhs/.multihetsep/.vcf/.gz)",
            ),
            (
                "Output",
                path(&cli.output_file).unwrap_or_else(|| "out.json".to_string()),
                "Output JSON path",
            ),
            (
                "N iter",
                cli.n_iter
                    .map_or_else(|| "25".to_string(), |n| n.to_string()),
                "EM iterations (C psmc -N)",
            ),
            (
                "Pattern",
                cli.pattern.clone().unwrap_or_default(),
                "PSMC pattern (empty: C psmc's 4+5*3+4)",
            ),
            (
                "Gen years",
                cli.gen_years.to_string(),
                "Years per generation, for the time axis",
            ),
            (
                "Threads",
                cli.threads.map(|n| n.to_string()).unwrap_or_default(),
                "Worker threads (empty: one per core)",
            ),
            (
                "Input fmt",
                cli.input.input_format.name().to_string(),
                "auto, psmcfa, mhs or vcf (Space cycles)",
            ),
            (
                "Bin size",
                cli.input.bin_size.to_string(),
                "Bases per bin: binning of mhs/vcf, scaling of every format",
            ),
            ("Mu", cli.mu.to_string(), "Mutation rate"),
            (
                "Smooth",
                cli.smooth_lambda.to_string(),
                "Smoothness penalty on log lambda (0: C psmc's objective)",
            ),
        ]
        .map(|(label, value, hint)| FormField { label, value, hint });
        Self {
            self_bin,
            base_cli: cli.clone(),
            page: TuiPage::Setup,
            fields: fields.into(),
            selected: 0,
            logs: VecDeque::new(),
            status: "Idle".to_string(),
            should_quit: false,
            autoscroll: true,
            manual_scroll: 0,
            running: None,
            stages: Default::default(),
            result: None,
            picker: None,
        }
    }

    fn field_value(&self, idx: usize) -> &str {
        &self.fields[idx].value
    }

    fn append_log(&mut self, line: impl Into<String>) {
        if self.logs.len() >= MAX_LOG_LINES {
            self.logs.pop_front();
        }
        self.logs.push_back(line.into());
        if self.autoscroll {
            self.manual_scroll = 0;
        }
    }

    fn clear_logs(&mut self) {
        self.logs.clear();
        self.manual_scroll = 0;
    }

    fn next_field(&mut self) {
        self.selected = (self.selected + 1) % self.fields.len();
    }

    fn prev_field(&mut self) {
        self.selected = (self.selected + self.fields.len() - 1) % self.fields.len();
    }

    fn toggle_input_format(&mut self) {
        let next = match parse_input_format_field(self.field_value(FIELD_INPUT_FORMAT)) {
            Ok(f) => {
                let all = FormatArg::value_variants();
                let i = all.iter().position(|&x| x == f).unwrap_or(0);
                all[(i + 1) % all.len()]
            }
            Err(_) => FormatArg::Auto,
        };
        self.fields[FIELD_INPUT_FORMAT].value = next.name().to_string();
    }

    fn scroll_up(&mut self) {
        self.autoscroll = false;
        self.manual_scroll = self.manual_scroll.saturating_add(2);
    }

    fn scroll_down(&mut self) {
        self.manual_scroll = self.manual_scroll.saturating_sub(2);
        if self.manual_scroll == 0 {
            self.autoscroll = true;
        }
    }

    fn update_stage(&mut self, stage: Stage, done: u64, total: u64, iter: Option<(usize, usize)>) {
        let now = Instant::now();
        let snap = &mut self.stages[stage as usize];
        let reset = snap.started_at.is_none()
            || done < snap.done
            || total != snap.total
            || iter != snap.iter;
        if reset {
            snap.started_at = Some(now);
        }
        snap.done = done.min(total);
        snap.total = total;
        snap.iter = iter;
        snap.updated_at = Some(now);
    }
}

/// The trimmed value of the field `name`, which must not be empty.
fn required<'a>(name: &str, v: &'a str) -> Result<&'a str> {
    let s = v.trim();
    if s.is_empty() {
        bail!("{name} is required");
    }
    Ok(s)
}

/// The value of the field `name`, read by `parse`, a parser of the command
/// line's values.
fn parsed<T>(name: &str, v: &str, parse: fn(&str) -> Result<T, String>) -> Result<T> {
    parse(required(name, v)?).map_err(|e| anyhow!("{name}: {e}"))
}

/// [`parsed`] for a field that may be left empty.
fn parsed_or_none<T>(
    name: &str,
    v: &str,
    parse: fn(&str) -> Result<T, String>,
) -> Result<Option<T>> {
    match v.trim() {
        "" => Ok(None),
        _ => parsed(name, v, parse).map(Some),
    }
}

fn parse_input_format_field(v: &str) -> Result<FormatArg> {
    FormatArg::from_str(v.trim(), true)
        .map_err(|_| anyhow!("Input fmt must be auto, psmcfa, mhs or vcf"))
}

/// The run the form describes: its fields over the other settings of the
/// command line that opened the TUI.
fn build_cli_from_form(app: &App) -> Result<InferArgs> {
    let field = |idx| app.field_value(idx);
    let input = required("Input", field(FIELD_INPUT))?;
    let output = required("Output", field(FIELD_OUTPUT))?;
    let n_iter = parsed("N iter", field(FIELD_N_ITER), whole_number)?;
    let mu = parsed("Mu", field(FIELD_MU), positive_f64)?;
    let gen_years = parsed("Gen years", field(FIELD_GEN_YEARS), positive_f64)?;
    let smooth = parsed("Smooth", field(FIELD_SMOOTH_LAMBDA), non_negative_f64)?;
    let input_format = parse_input_format_field(field(FIELD_INPUT_FORMAT))?;
    let pattern = field(FIELD_PATTERN).trim();
    let threads = parsed_or_none("Threads", field(FIELD_THREADS), positive_usize)?;
    let bin_size = parsed_or_none("Bin size", field(FIELD_BIN_SIZE), positive_usize)?
        .unwrap_or(DEFAULT_BIN_SIZE);

    let mut cli = app.base_cli.clone();
    cli.tui = false;
    cli.input_file = Some(PathBuf::from(input));
    cli.output_file = Some(PathBuf::from(output));
    cli.n_iter = Some(n_iter);
    cli.input.input_format = input_format;
    cli.pattern = (!pattern.is_empty()).then(|| pattern.to_string());
    cli.threads = threads;
    cli.input.bin_size = bin_size;
    cli.mu = mu;
    cli.smooth_lambda = smooth;
    cli.gen_years = gen_years;
    cli.no_progress = true;
    Ok(cli)
}

/// The command line of `cli`, as the TUI shows it.
fn command_line(self_bin: &Path, cli: &InferArgs) -> String {
    format!("{} {}", self_bin.display(), cli.to_args().join(" "))
}

fn preview_cmdline(app: &App) -> String {
    match build_cli_from_form(app) {
        Ok(cli) => command_line(&app.self_bin, &cli),
        Err(e) => format!("Invalid config: {e}"),
    }
}

fn poll_runner(app: &mut App) {
    let Some(task) = &app.running else {
        return;
    };
    let events: Vec<RunnerEvent> = task.rx.try_iter().collect();
    let mut done = false;
    for ev in events {
        match ev {
            RunnerEvent::Pipeline(ev) => handle_pipeline_event(app, ev),
            RunnerEvent::Completed(finished) => {
                app.status = format!("Done in {:.1}s", finished.elapsed_sec);
                app.result = Some(finished);
                app.page = TuiPage::Result;
                done = true;
            }
            RunnerEvent::Failed(err) => {
                app.status = "Failed".to_string();
                app.append_log(format!("Run failed: {err}"));
                done = true;
            }
            RunnerEvent::Cancelled => {
                app.status = "Cancelled".to_string();
                app.append_log("Run cancelled by user.");
                done = true;
            }
        }
    }
    if done && let Some(task) = app.running.take() {
        let _ = task.handle.join();
    }
}

/// Show an event of the pipeline: messages in the log, progress in the
/// stage gauges.
fn handle_pipeline_event(app: &mut App, ev: Event) {
    match ev {
        Event::Info(line) | Event::Output(line) => app.append_log(line),
        Event::Warning(line) => app.append_log(format!("Warning: {line}")),
        Event::Qc { text, .. } => {
            for line in text.lines() {
                app.append_log(line);
            }
        }
        Event::Iteration { iter, total } => app.append_log(format!("EM iteration {iter}/{total}")),
        Event::Progress {
            stage,
            done,
            total,
            iter,
        } => app.update_stage(stage, done, total, iter),
    }
}

fn start_run(app: &mut App) -> Result<()> {
    if app.running.is_some() {
        return Ok(());
    }
    let cli = build_cli_from_form(app)?;
    app.append_log(format!("$ {}", command_line(&app.self_bin, &cli)));
    app.stages = Default::default();
    app.result = None;
    app.page = TuiPage::Run;
    app.status = "Running".to_string();

    let (tx, rx) = mpsc::channel::<RunnerEvent>();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_worker = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let started = Instant::now();
        let sink = |ev: Event| {
            let _ = tx.send(RunnerEvent::Pipeline(ev));
        };
        let outcome = match run::run_inference(&cli, &stop_worker, &sink, false) {
            Ok(summary) => RunnerEvent::Completed(Finished {
                summary,
                elapsed_sec: started.elapsed().as_secs_f64(),
            }),
            Err(e) if e.is::<Cancelled>() => RunnerEvent::Cancelled,
            Err(e) => RunnerEvent::Failed(format!("{e:#}")),
        };
        let _ = tx.send(outcome);
    });
    app.running = Some(RunningTask { rx, handle, stop });
    Ok(())
}

fn stop_run(app: &mut App) {
    if let Some(task) = &app.running {
        task.stop.store(true, Ordering::Relaxed);
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
        if app.running.is_some() {
            stop_run(app);
        } else {
            app.should_quit = true;
        }
        return Ok(());
    }
    if app.picker.is_some() {
        return app.picker_key(key.code);
    }

    // While the form can be edited, it takes every printable character
    // (paths such as "results/run.json" hold the command letters), except
    // Space on the format field, which cycles it.
    let editing = app.page == TuiPage::Setup && app.running.is_none();
    // Ctrl or Alt makes a command; both together is AltGr on Windows, which
    // types characters such as '@'.
    let plain = ctrl == key.modifiers.contains(KeyModifiers::ALT);
    if editing
        && plain
        && let KeyCode::Char(c) = key.code
    {
        if c == ' ' && app.selected == FIELD_INPUT_FORMAT {
            app.toggle_input_format();
        } else {
            app.fields[app.selected].value.push(c);
        }
        return Ok(());
    }

    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            if app.running.is_some() {
                app.status = "Stop running job first (x)".to_string();
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::F(1) => app.page = TuiPage::Setup,
        KeyCode::F(2) => app.page = TuiPage::Run,
        KeyCode::F(3) => app.page = TuiPage::Result,
        KeyCode::Char('x') => stop_run(app),
        KeyCode::F(5) | KeyCode::Enter | KeyCode::Char('r') => start_run(app)?,
        KeyCode::PageUp => app.scroll_up(),
        KeyCode::PageDown => app.scroll_down(),
        KeyCode::End => {
            app.autoscroll = true;
            app.manual_scroll = 0;
        }
        KeyCode::Tab | KeyCode::Down if editing => app.next_field(),
        KeyCode::Tab | KeyCode::Down => app.scroll_down(),
        KeyCode::BackTab | KeyCode::Up if editing => app.prev_field(),
        KeyCode::BackTab | KeyCode::Up => app.scroll_up(),
        KeyCode::Char('l') if ctrl => app.clear_logs(),
        KeyCode::Char('o') if ctrl && editing => app.open_picker(FIELD_INPUT)?,
        KeyCode::Char('d') if ctrl && editing => app.open_picker(FIELD_OUTPUT)?,
        KeyCode::Left | KeyCode::Right if editing && app.selected == FIELD_INPUT_FORMAT => {
            app.toggle_input_format();
        }
        KeyCode::Backspace if editing => {
            app.fields[app.selected].value.pop();
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
    if let Some(task) = app.running.take() {
        let _ = task.handle.join();
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
                    app.status = "Input error".to_string();
                    app.append_log(format!("Input error: {e}"));
                }
            }
            TermEvent::Resize(_, _) => {
                let _ = term.autoresize();
            }
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
        app.fields[field].value = value.to_string();
    }

    /// The form fills the settings it shows and keeps the others of the
    /// command line that opened the TUI.
    #[test]
    fn the_form_builds_the_run_settings() {
        let mut a = app(&["--bootstrap", "5", "--vcf-mask", "m.bed"]);
        assert_eq!(a.field_value(FIELD_INPUT_FORMAT), "auto");
        assert_eq!(a.field_value(FIELD_GEN_YEARS), "25");
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
        assert!(!args.tui && args.no_progress);
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
        let preview = preview_cmdline(&a);
        assert!(preview.contains("--gen-years 29"), "{preview}");
        assert!(preview.contains("--bootstrap 5"), "{preview}");

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
        let err = |a: &App| build_cli_from_form(a).unwrap_err().to_string();
        assert_eq!(err(&a), "Input is required");
        set(&mut a, FIELD_INPUT, "in.psmcfa");
        set(&mut a, FIELD_GEN_YEARS, "old");
        assert!(err(&a).contains("Gen years"));
        set(&mut a, FIELD_GEN_YEARS, "25");
        set(&mut a, FIELD_INPUT_FORMAT, "bam");
        assert!(err(&a).contains("auto, psmcfa, mhs or vcf"));
        assert!(preview_cmdline(&a).starts_with("Invalid config"));
    }

    fn press(app: &mut App, code: KeyCode) {
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE)).unwrap();
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            press(app, KeyCode::Char(c));
        }
    }

    /// The form takes every printable character, the command letters
    /// included; Space cycles the format field.
    #[test]
    fn typed_text_reaches_the_form() {
        let mut a = app(&[]);
        a.selected = FIELD_OUTPUT;
        a.fields[FIELD_OUTPUT].value.clear();
        type_text(&mut a, "results/req x.json");
        assert_eq!(a.field_value(FIELD_OUTPUT), "results/req x.json");
        assert!(!a.should_quit && a.running.is_none());
        assert_eq!(a.page, TuiPage::Setup);
        press(&mut a, KeyCode::Backspace);
        assert_eq!(a.field_value(FIELD_OUTPUT), "results/req x.jso");

        a.selected = FIELD_INPUT_FORMAT;
        press(&mut a, KeyCode::Char(' '));
        assert_eq!(a.field_value(FIELD_INPUT_FORMAT), "psmcfa");
        press(&mut a, KeyCode::Right);
        assert_eq!(a.field_value(FIELD_INPUT_FORMAT), "mhs");

        // Ctrl chords are commands, not text.
        a.selected = FIELD_INPUT;
        a.logs.push_back("old".to_string());
        handle_key(
            &mut a,
            KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL),
        )
        .unwrap();
        assert!(a.logs.is_empty());
        assert_eq!(a.field_value(FIELD_INPUT), "");
        handle_key(
            &mut a,
            KeyEvent::new(
                KeyCode::Char('@'),
                KeyModifiers::CONTROL | KeyModifiers::ALT,
            ),
        )
        .unwrap();
        assert_eq!(a.field_value(FIELD_INPUT), "@");
    }

    /// Esc quits from the form; q quits from the other pages; neither while
    /// a run is going, which x stops.
    #[test]
    fn quit_and_stop_keys() {
        let mut a = app(&[]);
        press(&mut a, KeyCode::Esc);
        assert!(a.should_quit);

        let mut a = app(&[]);
        a.page = TuiPage::Result;
        press(&mut a, KeyCode::Char('q'));
        assert!(a.should_quit);

        let mut a = app(&[]);
        let stop = Arc::new(AtomicBool::new(false));
        a.running = Some(RunningTask {
            rx: mpsc::channel().1,
            handle: thread::spawn(|| {}),
            stop: Arc::clone(&stop),
        });
        a.page = TuiPage::Run;
        press(&mut a, KeyCode::Char('q'));
        press(&mut a, KeyCode::Esc);
        assert!(!a.should_quit);
        press(&mut a, KeyCode::Char('x'));
        assert!(stop.load(Ordering::Relaxed));
        assert_eq!(a.status, "Stopping...");
    }

    #[test]
    fn the_format_field_cycles_through_every_format() {
        let mut a = app(&[]);
        let mut seen = Vec::new();
        for _ in 0..5 {
            seen.push(a.field_value(FIELD_INPUT_FORMAT).to_string());
            a.toggle_input_format();
        }
        assert_eq!(seen, ["auto", "psmcfa", "mhs", "vcf", "auto"]);
    }

    #[test]
    fn pipeline_events_reach_the_log_and_the_gauges() {
        let mut a = app(&[]);
        handle_pipeline_event(&mut a, Event::Info("read".to_string()));
        handle_pipeline_event(&mut a, Event::Warning("odd".to_string()));
        handle_pipeline_event(
            &mut a,
            Event::Qc {
                text: "Input QC\nNo warnings or notes.\n".to_string(),
                warnings: Vec::new(),
            },
        );
        handle_pipeline_event(&mut a, Event::Iteration { iter: 2, total: 9 });
        handle_pipeline_event(
            &mut a,
            Event::Progress {
                stage: Stage::EForward,
                done: 40,
                total: 100,
                iter: Some((2, 9)),
            },
        );
        let logs: Vec<&str> = a.logs.iter().map(String::as_str).collect();
        assert_eq!(
            logs,
            [
                "read",
                "Warning: odd",
                "Input QC",
                "No warnings or notes.",
                "EM iteration 2/9"
            ]
        );
        let snap = &a.stages[Stage::EForward as usize];
        assert_eq!((snap.done, snap.total, snap.iter), (40, 100, Some((2, 9))));
    }

    #[test]
    fn the_end_of_a_run_is_shown() {
        let mut a = app(&[]);
        let (tx, rx) = mpsc::channel();
        a.running = Some(RunningTask {
            rx,
            handle: thread::spawn(|| {}),
            stop: Arc::new(AtomicBool::new(false)),
        });
        tx.send(RunnerEvent::Pipeline(Event::Output(
            "Parameters: x".to_string(),
        )))
        .unwrap();
        tx.send(RunnerEvent::Cancelled).unwrap();
        poll_runner(&mut a);
        assert!(a.running.is_none());
        assert_eq!(a.status, "Cancelled");
        assert_eq!(
            a.logs.iter().map(String::as_str).collect::<Vec<_>>(),
            ["Parameters: x", "Run cancelled by user."]
        );
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
        assert_eq!(a.page, TuiPage::Run);

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
