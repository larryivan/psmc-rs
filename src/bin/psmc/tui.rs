//! The interactive terminal UI (`--tui`): a form for the main settings, live
//! progress of the shared inference pipeline, and the results.

use anyhow::{Context, Result, anyhow, bail};
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use psmc_rs::hmm::Cancelled;

use crate::cli::{FormatArg, InferArgs};
use crate::dataset::is_supported_input_file;
use crate::output::is_stdout;
use crate::run::{self, Event, RunSummary, Stage};

mod view;

use view::render_ui;

const MAX_LOG_LINES: usize = 3_000;

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

#[derive(Clone)]
struct FormField {
    label: &'static str,
    value: String,
    hint: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TuiPage {
    Setup,
    Run,
    Result,
}

const STAGE_ORDER: [Stage; 5] = [
    Stage::Io,
    Stage::EForward,
    Stage::EBackward,
    Stage::MStep,
    Stage::Bootstrap,
];

fn stage_idx(stage: Stage) -> usize {
    match stage {
        Stage::Io => 0,
        Stage::EForward => 1,
        Stage::EBackward => 2,
        Stage::MStep => 3,
        Stage::Bootstrap => 4,
    }
}

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

#[derive(Debug, Clone, Copy)]
enum PickerMode {
    InputPath,
    OutputDir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PickerEntryKind {
    CurrentDir,
    ParentDir,
    Dir,
    File,
}

#[derive(Debug, Clone)]
struct PickerEntry {
    label: String,
    path: PathBuf,
    kind: PickerEntryKind,
}

#[derive(Debug, Clone)]
struct FilePicker {
    mode: PickerMode,
    input_format: Option<FormatArg>,
    cwd: PathBuf,
    entries: Vec<PickerEntry>,
    selected: usize,
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
    stages: Vec<StageSnapshot>,
    result: Option<Finished>,
    picker: Option<FilePicker>,
}

impl App {
    fn new(self_bin: PathBuf, cli: &InferArgs) -> Self {
        let mut fields = vec![
            FormField {
                label: "Input",
                value: String::new(),
                hint: "Input file (.psmcfa/.mhs/.multihetsep/.vcf/.gz)",
            },
            FormField {
                label: "Output",
                value: "out.json".to_string(),
                hint: "Output JSON path",
            },
            FormField {
                label: "N iter",
                value: "25".to_string(),
                hint: "EM iterations (C psmc -N)",
            },
            FormField {
                label: "Pattern",
                value: String::new(),
                hint: "PSMC pattern (empty: C psmc's 4+5*3+4)",
            },
            FormField {
                label: "Gen years",
                value: "25".to_string(),
                hint: "Years per generation, for the time axis",
            },
            FormField {
                label: "Threads",
                value: String::new(),
                hint: "Worker threads (empty: one per core)",
            },
            FormField {
                label: "Input fmt",
                value: "auto".to_string(),
                hint: "auto, psmcfa, mhs or vcf (Space cycles)",
            },
            FormField {
                label: "Bin size",
                value: "100".to_string(),
                hint: "Bases per bin: binning of mhs/vcf, scaling of every format",
            },
            FormField {
                label: "Mu",
                value: "2.5e-8".to_string(),
                hint: "Mutation rate",
            },
            FormField {
                label: "Smooth",
                value: "0".to_string(),
                hint: "Smoothness penalty on log lambda (0: C psmc's objective)",
            },
        ];

        if let Some(v) = &cli.input_file {
            fields[FIELD_INPUT].value = v.display().to_string();
        }
        if let Some(v) = &cli.output_file {
            fields[FIELD_OUTPUT].value = v.display().to_string();
        }
        if let Some(v) = cli.n_iter {
            fields[FIELD_N_ITER].value = v.to_string();
        }
        if let Some(v) = &cli.pattern {
            fields[FIELD_PATTERN].value = v.clone();
        }
        fields[FIELD_GEN_YEARS].value = cli.gen_years.to_string();
        if let Some(v) = cli.threads {
            fields[FIELD_THREADS].value = v.to_string();
        }
        fields[FIELD_INPUT_FORMAT].value = cli.input.input_format.name().to_string();
        fields[FIELD_BIN_SIZE].value = cli.input.bin_size.to_string();
        fields[FIELD_MU].value = cli.mu.to_string();
        fields[FIELD_SMOOTH_LAMBDA].value = cli.smooth_lambda.to_string();

        Self {
            self_bin,
            base_cli: cli.clone(),
            page: TuiPage::Setup,
            fields,
            selected: 0,
            logs: VecDeque::new(),
            status: "Idle".to_string(),
            should_quit: false,
            autoscroll: true,
            manual_scroll: 0,
            running: None,
            stages: vec![StageSnapshot::default(); STAGE_ORDER.len()],
            result: None,
            picker: None,
        }
    }

    fn field_value(&self, idx: usize) -> &str {
        &self.fields[idx].value
    }

    fn field_value_mut(&mut self, idx: usize) -> &mut String {
        &mut self.fields[idx].value
    }

    fn selected_field_mut(&mut self) -> &mut String {
        self.field_value_mut(self.selected)
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
        if self.selected == 0 {
            self.selected = self.fields.len() - 1;
        } else {
            self.selected -= 1;
        }
    }

    fn toggle_input_format(&mut self) {
        let next = match parse_input_format_field(self.field_value(FIELD_INPUT_FORMAT)) {
            Ok(f) => {
                let i = FormatArg::ALL.iter().position(|&x| x == f).unwrap_or(0);
                FormatArg::ALL[(i + 1) % FormatArg::ALL.len()]
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

    fn reset_stages(&mut self) {
        for s in &mut self.stages {
            *s = StageSnapshot::default();
        }
    }

    fn update_stage(&mut self, stage: Stage, done: u64, total: u64, iter: Option<(usize, usize)>) {
        let now = Instant::now();
        let idx = stage_idx(stage);
        let snap = &mut self.stages[idx];
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

    fn close_picker(&mut self) {
        self.picker = None;
    }

    fn open_picker(&mut self, mode: PickerMode) -> Result<()> {
        let input_format = parse_input_format_field(self.field_value(FIELD_INPUT_FORMAT))?;
        let start = match mode {
            PickerMode::InputPath => infer_picker_dir(self.field_value(FIELD_INPUT)),
            PickerMode::OutputDir => infer_picker_dir(self.field_value(FIELD_OUTPUT)),
        };
        let mut picker = FilePicker {
            mode,
            input_format: if matches!(mode, PickerMode::InputPath) {
                Some(input_format)
            } else {
                None
            },
            cwd: start,
            entries: Vec::new(),
            selected: 0,
        };
        refresh_picker_entries(&mut picker)?;
        self.picker = Some(picker);
        Ok(())
    }
}

fn parse_non_empty_str(name: &str, v: &str) -> Result<String> {
    let s = v.trim();
    if s.is_empty() {
        bail!("{name} is required");
    }
    Ok(s.to_string())
}

fn parse_optional_usize(name: &str, v: &str) -> Result<Option<usize>> {
    let s = v.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let n = s
        .parse::<usize>()
        .map_err(|_| anyhow!("{name} must be an integer"))?;
    if n == 0 {
        bail!("{name} must be >= 1");
    }
    Ok(Some(n))
}

fn parse_required_f64(name: &str, v: &str) -> Result<f64> {
    let s = parse_non_empty_str(name, v)?;
    let n = s
        .parse::<f64>()
        .map_err(|_| anyhow!("{name} must be a floating-point number"))?;
    if !n.is_finite() || n <= 0.0 {
        bail!("{name} must be > 0");
    }
    Ok(n)
}

fn parse_non_negative_f64(name: &str, v: &str) -> Result<f64> {
    let s = parse_non_empty_str(name, v)?;
    match s.parse::<f64>() {
        Ok(n) if n.is_finite() && n >= 0.0 => Ok(n),
        Ok(_) => bail!("{name} must be 0 or more"),
        Err(_) => bail!("{name} must be a number"),
    }
}

fn parse_input_format_field(v: &str) -> Result<FormatArg> {
    let s = v.trim().to_ascii_lowercase();
    FormatArg::ALL
        .into_iter()
        .find(|f| f.name() == s)
        .ok_or_else(|| anyhow!("Input fmt must be auto, psmcfa, mhs or vcf"))
}

fn build_cli_from_form(app: &App) -> Result<InferArgs> {
    let input = parse_non_empty_str("Input", app.field_value(FIELD_INPUT))?;
    let output = parse_non_empty_str("Output", app.field_value(FIELD_OUTPUT))?;
    let n_iter = parse_non_empty_str("N iter", app.field_value(FIELD_N_ITER))?
        .parse::<usize>()
        .map_err(|_| anyhow!("N iter must be a whole number"))?;
    let mu = parse_required_f64("Mu", app.field_value(FIELD_MU))?;
    let gen_years = parse_required_f64("Gen years", app.field_value(FIELD_GEN_YEARS))?;
    let smooth = parse_non_negative_f64("Smooth", app.field_value(FIELD_SMOOTH_LAMBDA))?;
    let input_format = parse_input_format_field(app.field_value(FIELD_INPUT_FORMAT))?;

    let pattern = app.field_value(FIELD_PATTERN).trim();
    let threads = parse_optional_usize("Threads", app.field_value(FIELD_THREADS))?;
    let bin_size =
        parse_optional_usize("Bin size", app.field_value(FIELD_BIN_SIZE))?.unwrap_or(100);

    let mut cli = app.base_cli.clone();
    cli.tui = false;
    cli.input_file = Some(PathBuf::from(input));
    cli.output_file = Some(PathBuf::from(output));
    cli.n_iter = Some(n_iter);
    cli.input.input_format = input_format;
    cli.pattern = if pattern.is_empty() {
        None
    } else {
        Some(pattern.to_string())
    };
    cli.threads = threads;
    cli.input.bin_size = bin_size;
    cli.mu = mu;
    cli.smooth_lambda = smooth;
    cli.gen_years = gen_years;
    cli.no_progress = true;
    Ok(cli)
}

fn infer_picker_dir(raw_path: &str) -> PathBuf {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    }
    let p = PathBuf::from(trimmed);
    if p.is_dir() {
        return p;
    }
    if p.is_file() {
        return p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(parent) = p.parent()
        && parent.exists()
    {
        return parent.to_path_buf();
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn refresh_picker_entries(picker: &mut FilePicker) -> Result<()> {
    let mut dirs = Vec::<PickerEntry>::new();
    let mut files = Vec::<PickerEntry>::new();
    for entry in fs::read_dir(&picker.cwd)
        .with_context(|| format!("failed to read directory {}", picker.cwd.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            dirs.push(PickerEntry {
                label: format!("{name}/"),
                path,
                kind: PickerEntryKind::Dir,
            });
        } else if path.is_file() {
            match picker.mode {
                PickerMode::InputPath => {
                    let fmt = picker.input_format.unwrap_or(FormatArg::Auto);
                    if is_supported_input_file(&path, fmt) {
                        files.push(PickerEntry {
                            label: name,
                            path,
                            kind: PickerEntryKind::File,
                        });
                    }
                }
                PickerMode::OutputDir => {}
            }
        }
    }
    dirs.sort_by_key(|e| e.label.to_ascii_lowercase());
    files.sort_by_key(|e| e.label.to_ascii_lowercase());

    let mut entries = Vec::<PickerEntry>::new();
    entries.push(PickerEntry {
        label: "[use this directory]".to_string(),
        path: picker.cwd.clone(),
        kind: PickerEntryKind::CurrentDir,
    });
    if let Some(parent) = picker.cwd.parent() {
        entries.push(PickerEntry {
            label: "../".to_string(),
            path: parent.to_path_buf(),
            kind: PickerEntryKind::ParentDir,
        });
    }
    entries.extend(dirs);
    if matches!(picker.mode, PickerMode::InputPath) {
        entries.extend(files);
    }
    picker.entries = entries;
    picker.selected = picker.selected.min(picker.entries.len().saturating_sub(1));
    Ok(())
}

fn apply_picker_selection(app: &mut App) -> Result<()> {
    let Some(picker) = app.picker.as_mut() else {
        return Ok(());
    };
    if picker.entries.is_empty() {
        return Ok(());
    }
    let selected = picker.entries[picker.selected].clone();
    match selected.kind {
        PickerEntryKind::File => match picker.mode {
            PickerMode::InputPath => {
                let target = selected.path;
                app.fields[FIELD_INPUT].value = target.display().to_string();
                app.close_picker();
                Ok(())
            }
            PickerMode::OutputDir => Ok(()),
        },
        PickerEntryKind::CurrentDir | PickerEntryKind::Dir | PickerEntryKind::ParentDir => {
            match picker.mode {
                PickerMode::InputPath => {
                    // Input requires selecting a file, not a directory.
                    picker.cwd = selected.path;
                    refresh_picker_entries(picker)?;
                    Ok(())
                }
                PickerMode::OutputDir => {
                    let target = selected.path;
                    let current = PathBuf::from(app.field_value(FIELD_OUTPUT));
                    let fname = current
                        .file_name()
                        .map(|v| v.to_os_string())
                        .unwrap_or_else(|| "out.json".into());
                    app.fields[FIELD_OUTPUT].value = target.join(fname).display().to_string();
                    app.close_picker();
                    Ok(())
                }
            }
        }
    }
}

fn browse_picker_dir(app: &mut App) -> Result<()> {
    let Some(picker) = app.picker.as_mut() else {
        return Ok(());
    };
    if picker.entries.is_empty() {
        return Ok(());
    }
    let selected = picker.entries[picker.selected].clone();
    let next = match selected.kind {
        PickerEntryKind::ParentDir | PickerEntryKind::Dir | PickerEntryKind::CurrentDir => {
            Some(selected.path)
        }
        PickerEntryKind::File => None,
    };
    if let Some(path) = next {
        picker.cwd = path;
        refresh_picker_entries(picker)?;
    }
    Ok(())
}

fn preview_cmdline(app: &App) -> String {
    match build_cli_from_form(app) {
        Ok(cli) => format!("{} {}", app.self_bin.display(), cli.to_args().join(" ")),
        Err(e) => format!("Invalid config: {e}"),
    }
}

fn poll_runner(app: &mut App) {
    let mut events = Vec::<RunnerEvent>::new();
    if let Some(task) = app.running.as_mut() {
        while let Ok(ev) = task.rx.try_recv() {
            events.push(ev);
        }
    }
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
                app.append_log(line.to_string());
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
    let cmdline = format!("{} {}", app.self_bin.display(), cli.to_args().join(" "));
    app.append_log(format!("$ {cmdline}"));
    app.reset_stages();
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
    if let Some(task) = app.running.as_mut() {
        task.stop.store(true, Ordering::Relaxed);
        app.status = "Stopping...".to_string();
        app.append_log("Stop requested.");
    }
}

fn handle_key(app: &mut App, key: KeyEvent) -> Result<()> {
    if key.kind != KeyEventKind::Press {
        return Ok(());
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        if app.running.is_some() {
            stop_run(app);
        } else {
            app.should_quit = true;
        }
        return Ok(());
    }

    if app.picker.is_some() {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => app.close_picker(),
            KeyCode::Up => {
                if let Some(picker) = app.picker.as_mut() {
                    if picker.selected == 0 {
                        picker.selected = picker.entries.len().saturating_sub(1);
                    } else {
                        picker.selected -= 1;
                    }
                }
            }
            KeyCode::Down => {
                if let Some(picker) = app.picker.as_mut()
                    && !picker.entries.is_empty()
                {
                    picker.selected = (picker.selected + 1) % picker.entries.len();
                }
            }
            KeyCode::Enter => {
                apply_picker_selection(app)?;
            }
            KeyCode::Right => {
                browse_picker_dir(app)?;
            }
            KeyCode::Backspace | KeyCode::Left => {
                if let Some(picker) = app.picker.as_mut()
                    && let Some(parent) = picker.cwd.parent()
                {
                    picker.cwd = parent.to_path_buf();
                    refresh_picker_entries(picker)?;
                }
            }
            _ => {}
        }
        return Ok(());
    }

    match key.code {
        KeyCode::Char('q') => {
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
        KeyCode::F(5) | KeyCode::Enter | KeyCode::Char('r') => {
            if app.running.is_none() {
                start_run(app)?;
            }
        }
        KeyCode::PageUp => app.scroll_up(),
        KeyCode::PageDown => app.scroll_down(),
        KeyCode::End => {
            app.autoscroll = true;
            app.manual_scroll = 0;
        }
        KeyCode::Tab | KeyCode::Down => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.next_field();
            } else {
                app.scroll_down();
            }
        }
        KeyCode::BackTab | KeyCode::Up => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.prev_field();
            } else {
                app.scroll_up();
            }
        }
        KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => app.clear_logs(),
        KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.open_picker(PickerMode::InputPath)?;
            }
        }
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.open_picker(PickerMode::OutputDir)?;
            }
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => {
            if app.page == TuiPage::Setup
                && app.running.is_none()
                && app.selected == FIELD_INPUT_FORMAT
            {
                app.toggle_input_format();
            }
        }
        KeyCode::Backspace => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.selected_field_mut().pop();
            }
        }
        KeyCode::Char(c) if app.page == TuiPage::Setup && app.running.is_none() => {
            app.selected_field_mut().push(c);
        }
        _ => {}
    }
    Ok(())
}

/// Settings the TUI cannot honour: stdout is the interface.
fn check_tui_args(cli: &InferArgs) -> Result<()> {
    if cli.qc_json.as_deref().is_some_and(is_stdout) {
        bail!("--qc-json - would print over the interface; give --qc-json a file with --tui");
    }
    Ok(())
}

/// Run the TUI until the user quits; `cli` fills the form.
pub fn run_tui(cli: &InferArgs) -> Result<()> {
    check_tui_args(cli)?;
    let self_bin = std::env::current_exe().context("failed to resolve executable path")?;
    let mut app = App::new(self_bin, cli);

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend).context("failed to initialize terminal")?;
    term.clear().context("failed to clear terminal")?;

    let mut final_res = Ok(());
    while !app.should_quit {
        poll_runner(&mut app);
        if let Err(e) = term.draw(|f| render_ui(f, &app)) {
            final_res = Err(anyhow!("draw failed: {e}"));
            break;
        }

        if event::poll(Duration::from_millis(80)).context("event poll failed")? {
            match event::read().context("failed to read terminal event")? {
                TermEvent::Key(key) => {
                    if let Err(e) = handle_key(&mut app, key) {
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
    }

    stop_run(&mut app);
    if let Some(task) = app.running.take() {
        let _ = task.handle.join();
    }

    disable_raw_mode().ok();
    execute!(term.backend_mut(), LeaveAlternateScreen).ok();
    let _ = term.show_cursor();
    final_res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;

    fn app(args: &[&str]) -> App {
        let mut argv = vec!["psmc-rs", "--tui"];
        argv.extend_from_slice(args);
        let cli = Cli::try_parse_from(argv).unwrap();
        App::new(PathBuf::from("psmc-rs"), &cli.infer)
    }

    fn set(app: &mut App, field: usize, value: &str) {
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
        let cli = |args: &[&str]| {
            let mut argv = vec!["psmc-rs", "--tui"];
            argv.extend_from_slice(args);
            Cli::try_parse_from(argv).unwrap().infer
        };
        let err = check_tui_args(&cli(&["--qc-json", "-"])).unwrap_err();
        assert!(err.to_string().contains("give --qc-json a file"), "{err}");
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
        let snap = &a.stages[stage_idx(Stage::EForward)];
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
