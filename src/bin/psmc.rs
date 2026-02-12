use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Gauge, List, ListItem, ListState, Paragraph, Tabs, Wrap,
};
use rayon::prelude::*;
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

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    default_bootstrap_dir, new_rng, replicate_json_path, rows_to_sequences,
    sample_sequences_block_bootstrap, sequences_to_observations, summarize_bootstrap_models,
    write_bootstrap_summary_json, write_bootstrap_summary_tsv,
};
use psmc_rs::hmm::EStepPhase;
use psmc_rs::hmm::write_tmrca_posterior_tsv;
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{
    EmProgressEvent, MStepConfig, bounds_from_config, em_train, em_train_with_progress,
};
use psmc_rs::progress;
use psmc_rs::report::{default_html_path, write_html_report};

const MAX_LOG_LINES: usize = 3_000;

const FIELD_INPUT: usize = 0;
const FIELD_OUTPUT: usize = 1;
const FIELD_N_ITER: usize = 2;
const FIELD_PATTERN: usize = 3;
const FIELD_BATCH_SIZE: usize = 4;
const FIELD_THREADS: usize = 5;
const FIELD_INPUT_FORMAT: usize = 6;
const FIELD_MHS_BIN_SIZE: usize = 7;
const FIELD_MU: usize = 8;
const FIELD_SMOOTH_LAMBDA: usize = 9;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum InputFormat {
    Psmcfa,
    Mhs,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "psmc-rs")]
#[command(about = "PSMC in Rust", long_about = None)]
struct Cli {
    #[arg(long, help = "Launch interactive TUI")]
    tui: bool,

    #[arg(required_unless_present = "tui")]
    input_file: Option<PathBuf>,

    #[arg(required_unless_present = "tui")]
    output_file: Option<PathBuf>,

    #[arg(required_unless_present = "tui")]
    n_iter: Option<usize>,

    #[arg(
        long,
        help = "Number of worker threads for E-step (multi-sequence) and bootstrap"
    )]
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

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of block-bootstrap replicates (0 disables bootstrap)"
    )]
    bootstrap: usize,

    #[arg(long, default_value_t = 50_000, help = "Bootstrap block size in bins")]
    bootstrap_block_size: usize,

    #[arg(
        long,
        default_value_t = 42,
        help = "Random seed for bootstrap resampling"
    )]
    bootstrap_seed: u64,

    #[arg(
        long,
        help = "EM iterations for each bootstrap replicate (default: same as N_ITER)"
    )]
    bootstrap_iters: Option<usize>,

    #[arg(
        long,
        help = "Directory for bootstrap replicate JSON files and summary outputs"
    )]
    bootstrap_dir: Option<PathBuf>,

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StageKind {
    Io,
    EForward,
    EBackward,
    MStep,
    Bootstrap,
}

impl StageKind {
    fn label(self) -> &'static str {
        match self {
            Self::Io => "IO",
            Self::EForward => "E-forward",
            Self::EBackward => "E-backward",
            Self::MStep => "M-step",
            Self::Bootstrap => "Bootstrap",
        }
    }

    fn unit(self) -> &'static str {
        match self {
            Self::Io => "B/s",
            Self::EForward | Self::EBackward => "sites/s",
            Self::MStep => "iter/s",
            Self::Bootstrap => "rep/s",
        }
    }
}

const STAGE_ORDER: [StageKind; 5] = [
    StageKind::Io,
    StageKind::EForward,
    StageKind::EBackward,
    StageKind::MStep,
    StageKind::Bootstrap,
];

fn stage_idx(stage: StageKind) -> usize {
    match stage {
        StageKind::Io => 0,
        StageKind::EForward => 1,
        StageKind::EBackward => 2,
        StageKind::MStep => 3,
        StageKind::Bootstrap => 4,
    }
}

#[derive(Debug, Clone)]
struct StageSnapshot {
    done: u64,
    total: u64,
    iter: Option<(usize, usize)>,
    started_at: Option<Instant>,
    updated_at: Option<Instant>,
}

impl Default for StageSnapshot {
    fn default() -> Self {
        Self {
            done: 0,
            total: 0,
            iter: None,
            started_at: None,
            updated_at: None,
        }
    }
}

#[derive(Debug, Clone)]
struct RunSummary {
    output_json: PathBuf,
    html_report: PathBuf,
    tmrca_out: Option<PathBuf>,
    bootstrap_dir: Option<PathBuf>,
    last_loglike: Option<f64>,
    elapsed_sec: f64,
}

#[derive(Debug, Clone)]
enum RunnerEvent {
    Log(String),
    Stage {
        stage: StageKind,
        done: u64,
        total: u64,
        iter: Option<(usize, usize)>,
    },
    Completed(RunSummary),
    Failed(String),
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
    input_format: Option<InputFormat>,
    cwd: PathBuf,
    entries: Vec<PickerEntry>,
    selected: usize,
}

struct App {
    self_bin: PathBuf,
    base_cli: Cli,
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
    result: Option<RunSummary>,
    picker: Option<FilePicker>,
}

impl App {
    fn new(self_bin: PathBuf, cli: &Cli) -> Self {
        let mut fields = vec![
            FormField {
                label: "Input",
                value: String::new(),
                hint: "Input file (.psmcfa/.mhs/.multihetsep/.gz)",
            },
            FormField {
                label: "Output",
                value: "out.json".to_string(),
                hint: "Output JSON path",
            },
            FormField {
                label: "N iter",
                value: "20".to_string(),
                hint: "EM iterations",
            },
            FormField {
                label: "Pattern",
                value: "4+25*2+4+6".to_string(),
                hint: "PSMC pattern",
            },
            FormField {
                label: "Batch size",
                value: "300000".to_string(),
                hint: "Row chunk size",
            },
            FormField {
                label: "Threads",
                value: "1".to_string(),
                hint: "Rayon threads",
            },
            FormField {
                label: "Input fmt",
                value: "psmcfa".to_string(),
                hint: "psmcfa or mhs",
            },
            FormField {
                label: "MHS bin",
                value: "100".to_string(),
                hint: "Only for mhs",
            },
            FormField {
                label: "Mu",
                value: "2.5e-8".to_string(),
                hint: "Mutation rate",
            },
            FormField {
                label: "Smooth",
                value: "1e-3".to_string(),
                hint: "Smooth penalty",
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
        if let Some(v) = cli.batch_size {
            fields[FIELD_BATCH_SIZE].value = v.to_string();
        }
        if let Some(v) = cli.threads {
            fields[FIELD_THREADS].value = v.to_string();
        }
        fields[FIELD_INPUT_FORMAT].value = match cli.input_format {
            InputFormat::Psmcfa => "psmcfa".to_string(),
            InputFormat::Mhs => "mhs".to_string(),
        };
        fields[FIELD_MHS_BIN_SIZE].value = cli.mhs_bin_size.to_string();
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
        let v = self
            .field_value(FIELD_INPUT_FORMAT)
            .trim()
            .to_ascii_lowercase();
        self.fields[FIELD_INPUT_FORMAT].value = if v == "mhs" {
            "psmcfa".to_string()
        } else {
            "mhs".to_string()
        };
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

    fn update_stage(
        &mut self,
        stage: StageKind,
        done: u64,
        total: u64,
        iter: Option<(usize, usize)>,
    ) {
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

fn parse_required_usize(name: &str, v: &str) -> Result<usize> {
    let s = parse_non_empty_str(name, v)?;
    let n = s
        .parse::<usize>()
        .map_err(|_| anyhow!("{name} must be a positive integer"))?;
    if n == 0 {
        bail!("{name} must be >= 1");
    }
    Ok(n)
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

fn parse_input_format_field(v: &str) -> Result<InputFormat> {
    let s = v.trim().to_ascii_lowercase();
    match s.as_str() {
        "psmcfa" => Ok(InputFormat::Psmcfa),
        "mhs" => Ok(InputFormat::Mhs),
        _ => bail!("Input fmt must be psmcfa or mhs"),
    }
}

fn is_supported_input_file(path: &Path, fmt: InputFormat) -> bool {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match fmt {
        InputFormat::Psmcfa => name.ends_with(".psmcfa") || name.ends_with(".psmcfa.gz"),
        InputFormat::Mhs => {
            name.ends_with(".mhs")
                || name.ends_with(".mhs.gz")
                || name.ends_with(".multihetsep")
                || name.ends_with(".multihetsep.gz")
        }
    }
}

fn build_cli_from_form(app: &App) -> Result<Cli> {
    let input = parse_non_empty_str("Input", app.field_value(FIELD_INPUT))?;
    let output = parse_non_empty_str("Output", app.field_value(FIELD_OUTPUT))?;
    let n_iter = parse_required_usize("N iter", app.field_value(FIELD_N_ITER))?;
    let mu = parse_required_f64("Mu", app.field_value(FIELD_MU))?;
    let smooth = parse_required_f64("Smooth", app.field_value(FIELD_SMOOTH_LAMBDA))?;
    let input_format = parse_input_format_field(app.field_value(FIELD_INPUT_FORMAT))?;

    let pattern = app.field_value(FIELD_PATTERN).trim();
    let batch_size = parse_optional_usize("Batch size", app.field_value(FIELD_BATCH_SIZE))?;
    let threads = parse_optional_usize("Threads", app.field_value(FIELD_THREADS))?;
    let mhs_bin =
        parse_optional_usize("MHS bin", app.field_value(FIELD_MHS_BIN_SIZE))?.unwrap_or(100);

    let mut cli = app.base_cli.clone();
    cli.tui = false;
    cli.input_file = Some(PathBuf::from(input));
    cli.output_file = Some(PathBuf::from(output));
    cli.n_iter = Some(n_iter);
    cli.input_format = input_format;
    cli.pattern = if pattern.is_empty() {
        None
    } else {
        Some(pattern.to_string())
    };
    cli.batch_size = batch_size;
    cli.threads = threads;
    cli.mhs_bin_size = mhs_bin;
    cli.mu = mu;
    cli.smooth_lambda = smooth;
    cli.no_progress = true;
    Ok(cli)
}

fn cli_to_args(cli: &Cli) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(v) = &cli.input_file {
        args.push(v.display().to_string());
    }
    if let Some(v) = &cli.output_file {
        args.push(v.display().to_string());
    }
    if let Some(v) = cli.n_iter {
        args.push(v.to_string());
    }
    args.push("--input-format".to_string());
    args.push(
        match cli.input_format {
            InputFormat::Psmcfa => "psmcfa",
            InputFormat::Mhs => "mhs",
        }
        .to_string(),
    );
    if cli.input_format == InputFormat::Mhs {
        args.push("--mhs-bin-size".to_string());
        args.push(cli.mhs_bin_size.to_string());
    }
    if let Some(v) = &cli.pattern {
        args.push("--pattern".to_string());
        args.push(v.clone());
    }
    if let Some(v) = cli.batch_size {
        args.push("--batch-size".to_string());
        args.push(v.to_string());
    }
    if let Some(v) = cli.threads {
        args.push("--threads".to_string());
        args.push(v.to_string());
    }
    args.push("--mu".to_string());
    args.push(cli.mu.to_string());
    args.push("--smooth-lambda".to_string());
    args.push(cli.smooth_lambda.to_string());
    if cli.bootstrap > 0 {
        args.push("--bootstrap".to_string());
        args.push(cli.bootstrap.to_string());
    }
    args.push("--no-progress".to_string());
    args
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
    if let Some(parent) = p.parent() {
        if parent.exists() {
            return parent.to_path_buf();
        }
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
                    let fmt = picker.input_format.unwrap_or(InputFormat::Psmcfa);
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

fn pick_supported_file_from_dir(dir: &Path, fmt: InputFormat) -> Result<PathBuf> {
    let mut files = Vec::<PathBuf>::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        if is_supported_input_file(&p, fmt) {
            files.push(p);
        }
    }
    files.sort();
    files.into_iter().next().ok_or_else(|| {
        anyhow!(
            "directory {} does not contain a supported {:?} input file",
            dir.display(),
            fmt
        )
    })
}

fn resolve_input_path(path: &Path, fmt: InputFormat) -> Result<PathBuf> {
    if path.is_dir() {
        pick_supported_file_from_dir(path, fmt)
    } else {
        Ok(path.to_path_buf())
    }
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

fn stage_eta_and_rate(stage: &StageSnapshot) -> (Option<f64>, Option<f64>) {
    let Some(start) = stage.started_at else {
        return (None, None);
    };
    if stage.done == 0 {
        return (None, None);
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rate = (stage.done as f64) / elapsed;
    if stage.total > stage.done && rate > 0.0 {
        let rem = (stage.total - stage.done) as f64;
        (Some(rem / rate), Some(rate))
    } else {
        (Some(0.0), Some(rate))
    }
}

fn fmt_seconds(sec: f64) -> String {
    let s = sec.max(0.0).round() as u64;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    let ss = s % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{ss:02}")
    } else {
        format!("{m:02}:{ss:02}")
    }
}

fn fmt_rate(rate: f64, unit: &str) -> String {
    if unit == "B/s" {
        let mut v = rate;
        let mut suffix = "B/s";
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "KB/s";
        }
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "MB/s";
        }
        if v >= 1024.0 {
            v /= 1024.0;
            suffix = "GB/s";
        }
        return format!("{v:.1} {suffix}");
    }
    format!("{rate:.1} {unit}")
}

fn preview_cmdline(app: &App) -> String {
    match build_cli_from_form(app) {
        Ok(cli) => format!("{} {}", app.self_bin.display(), cli_to_args(&cli).join(" ")),
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
            RunnerEvent::Log(line) => app.append_log(line),
            RunnerEvent::Stage {
                stage,
                done: stage_done,
                total,
                iter,
            } => app.update_stage(stage, stage_done, total, iter),
            RunnerEvent::Completed(summary) => {
                app.status = format!("Done in {:.1}s", summary.elapsed_sec);
                app.result = Some(summary);
                app.page = TuiPage::Result;
                done = true;
            }
            RunnerEvent::Failed(err) => {
                app.status = "Failed".to_string();
                app.append_log(format!("Run failed: {err}"));
                done = true;
            }
        }
    }
    if done {
        if let Some(task) = app.running.take() {
            let _ = task.handle.join();
        }
    }
}

fn start_run(app: &mut App) -> Result<()> {
    if app.running.is_some() {
        return Ok(());
    }
    let mut cli = build_cli_from_form(app)?;
    let cmdline = format!("{} {}", app.self_bin.display(), cli_to_args(&cli).join(" "));
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
        cli.no_progress = true;
        let mut send = |ev: RunnerEvent| {
            let _ = tx.send(ev);
        };
        match run_inference_with_events(cli, &stop_worker, &mut send) {
            Ok(mut summary) => {
                summary.elapsed_sec = started.elapsed().as_secs_f64();
                let _ = tx.send(RunnerEvent::Completed(summary));
            }
            Err(e) => {
                let _ = tx.send(RunnerEvent::Failed(e.to_string()));
            }
        }
    });

    app.running = Some(RunningTask { rx, handle, stop });
    Ok(())
}

fn stop_run(app: &mut App) {
    if let Some(task) = app.running.as_mut() {
        task.stop.store(true, Ordering::Relaxed);
        app.status = "Stopping after current phase...".to_string();
        app.append_log("Stop requested.");
    }
}

fn render_logs_panel(f: &mut Frame, area: Rect, app: &App, title: &str) {
    let visible = area.height.saturating_sub(2) as usize;
    let log_count = app.logs.len();
    let max_scroll = log_count.saturating_sub(visible) as u16;
    let scroll = if app.autoscroll {
        max_scroll
    } else {
        app.manual_scroll.min(max_scroll)
    };
    let log_widget = Paragraph::new(
        app.logs
            .iter()
            .map(|s| Line::raw(s.clone()))
            .collect::<Vec<_>>(),
    )
    .wrap(Wrap { trim: false })
    .scroll((scroll, 0))
    .block(
        Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(log_widget, area);
}

fn render_setup_page(f: &mut Frame, area: Rect, app: &App) {
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(area);

    let form_items = app
        .fields
        .iter()
        .map(|field| {
            let value_span = if field.value.trim().is_empty() {
                Span::styled("<empty>", Style::default().fg(Color::DarkGray))
            } else {
                Span::styled(field.value.clone(), Style::default().fg(Color::White))
            };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<11}", field.label),
                    Style::default()
                        .fg(Color::LightBlue)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                value_span,
            ]))
        })
        .collect::<Vec<_>>();

    let mut list_state = ListState::default();
    list_state.select(Some(app.selected));
    let form = List::new(form_items)
        .block(
            Block::default()
                .title("Setup")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(26, 48, 80))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    f.render_stateful_widget(form, body[0], &mut list_state);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Min(6),
        ])
        .split(body[1]);

    let preview_widget = Paragraph::new(format!(
        "{}\n\nCtrl+O: pick input file\nCtrl+D: pick output directory\nF5: run",
        preview_cmdline(app)
    ))
    .wrap(Wrap { trim: false })
    .block(
        Block::default()
            .title("Command Preview")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(preview_widget, right[0]);

    let selected = &app.fields[app.selected];
    let selected_widget = Paragraph::new(format!(
        "{}\n{}",
        selected.label,
        if selected.value.trim().is_empty() {
            "<empty>".to_string()
        } else {
            selected.value.clone()
        }
    ))
    .wrap(Wrap { trim: false })
    .block(
        Block::default()
            .title("Selected Field (Full Value)")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(selected_widget, right[1]);

    render_logs_panel(f, right[2], app, "Recent Log");
}

fn render_run_page(f: &mut Frame, area: Rect, app: &App) {
    let mut constraints = Vec::<Constraint>::new();
    for _ in STAGE_ORDER {
        constraints.push(Constraint::Length(3));
    }
    constraints.push(Constraint::Min(8));
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, stage) in STAGE_ORDER.iter().enumerate() {
        let snap = &app.stages[stage_idx(*stage)];
        let ratio = if snap.total > 0 {
            (snap.done as f64 / snap.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (eta, rate) = stage_eta_and_rate(snap);
        let iter_txt = snap
            .iter
            .map(|(it, all)| format!(" | EM {it}/{all}"))
            .unwrap_or_default();
        let eta_txt = eta.map(fmt_seconds).unwrap_or_else(|| "--:--".to_string());
        let rate_txt = rate
            .map(|v| fmt_rate(v, stage.unit()))
            .unwrap_or_else(|| "--".to_string());
        let label = format!(
            "{}{iter_txt} | {}/{} | ETA {} | {}",
            stage.label(),
            snap.done,
            snap.total,
            eta_txt,
            rate_txt
        );
        let gauge = Gauge::default()
            .block(
                Block::default()
                    .title(stage.label())
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded),
            )
            .gauge_style(
                Style::default()
                    .fg(Color::Cyan)
                    .bg(Color::Rgb(20, 30, 45))
                    .add_modifier(Modifier::BOLD),
            )
            .label(label)
            .ratio(ratio);
        f.render_widget(gauge, chunks[i]);
    }
    render_logs_panel(f, chunks[chunks.len() - 1], app, "Run Log");
}

fn render_result_page(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(8)])
        .split(area);
    let summary = if let Some(r) = &app.result {
        format!(
            "Elapsed: {:.2}s\nOutput JSON: {}\nHTML report: {}\nTMRCA: {}\nBootstrap dir: {}\nLast EM loglike: {}",
            r.elapsed_sec,
            r.output_json.display(),
            r.html_report.display(),
            r.tmrca_out
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.bootstrap_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string()),
            r.last_loglike
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
        )
    } else {
        "No completed run yet.".to_string()
    };
    let summary_widget = Paragraph::new(summary).block(
        Block::default()
            .title("Result")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(summary_widget, chunks[0]);
    render_logs_panel(f, chunks[1], app, "Run Log");
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn render_picker(f: &mut Frame, app: &App) {
    let Some(picker) = app.picker.as_ref() else {
        return;
    };
    let popup = centered_rect(86, 82, f.area());
    f.render_widget(Clear, popup);
    f.render_widget(
        Block::default()
            .style(Style::default().bg(Color::Rgb(6, 10, 18)))
            .borders(Borders::NONE),
        popup,
    );
    let popup_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(2)])
        .split(popup);
    let items = picker
        .entries
        .iter()
        .map(|e| ListItem::new(e.label.clone()))
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    state.select(Some(picker.selected));
    let title = match picker.mode {
        PickerMode::InputPath => {
            let fmt = picker.input_format.unwrap_or(InputFormat::Psmcfa);
            match fmt {
                InputFormat::Psmcfa => "Input picker [psmcfa]: Enter file=select, Enter dir=open",
                InputFormat::Mhs => "Input picker [mhs]: Enter file=select, Enter dir=open",
            }
        }
        PickerMode::OutputDir => "Output directory picker: Enter dir=select",
    };
    let list = List::new(items)
        .style(Style::default().fg(Color::White).bg(Color::Rgb(6, 10, 18)))
        .block(
            Block::default()
                .title(format!("{title} | cwd: {}", picker.cwd.display()))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_symbol("▶ ")
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(26, 48, 80))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
    f.render_stateful_widget(list, popup_chunks[0], &mut state);

    let help = Paragraph::new(
        "Picker keys: Enter select/open | -> browse | <-/Backspace up | Esc/q close",
    )
    .style(Style::default().fg(Color::Gray))
    .block(
        Block::default()
            .borders(Borders::TOP)
            .border_type(BorderType::Plain),
    );
    f.render_widget(help, popup_chunks[1]);
}

fn render_ui(f: &mut Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(4),
        ])
        .split(f.area());

    let header = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(44), Constraint::Min(20)])
        .split(root[0]);

    let titles = vec![" Setup ", " Run ", " Result "];
    let selected_tab = match app.page {
        TuiPage::Setup => 0,
        TuiPage::Run => 1,
        TuiPage::Result => 2,
    };
    let tabs = Tabs::new(titles)
        .select(selected_tab)
        .block(
            Block::default()
                .title("PSMC-RS")
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        )
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .style(Style::default().fg(Color::Gray));
    f.render_widget(tabs, header[0]);

    let status = Paragraph::new(Line::from(vec![
        Span::styled(
            if app.running.is_some() {
                "RUNNING"
            } else {
                "READY"
            },
            Style::default()
                .fg(if app.running.is_some() {
                    Color::Yellow
                } else {
                    Color::Green
                })
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::raw(&app.status),
    ]))
    .block(
        Block::default()
            .title("Status")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(status, header[1]);

    match app.page {
        TuiPage::Setup => render_setup_page(f, root[1], app),
        TuiPage::Run => render_run_page(f, root[1], app),
        TuiPage::Result => render_result_page(f, root[1], app),
    }

    let hint = if app.picker.is_some() {
        "Picker: ↑/↓ move  Enter select/open  → browse dir  Backspace/← up  Esc/q close"
    } else if app.running.is_some() {
        "Run: F1/F2/F3 switch pages  PgUp/PgDn scroll logs  x stop  q quit"
    } else {
        "Setup: Tab/Shift+Tab select field  type edit  Ctrl+O input picker  Ctrl+D output dir picker  Space toggle fmt  F5 run"
    };
    let secondary = if app.picker.is_some() {
        "Pages: F1 Setup | F2 Run | F3 Result"
    } else {
        "Pages: F1 Setup | F2 Run | F3 Result"
    };
    let footer = Paragraph::new(vec![
        Line::raw(hint),
        Line::raw(format!(
            "{secondary} | Field hint: {}",
            app.fields[app.selected].hint
        )),
    ])
    .block(
        Block::default()
            .title("Help")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );
    f.render_widget(footer, root[2]);

    render_picker(f, app);
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
                if let Some(picker) = app.picker.as_mut() {
                    if !picker.entries.is_empty() {
                        picker.selected = (picker.selected + 1) % picker.entries.len();
                    }
                }
            }
            KeyCode::Enter => {
                apply_picker_selection(app)?;
            }
            KeyCode::Right => {
                browse_picker_dir(app)?;
            }
            KeyCode::Backspace | KeyCode::Left => {
                if let Some(picker) = app.picker.as_mut() {
                    if let Some(parent) = picker.cwd.parent() {
                        picker.cwd = parent.to_path_buf();
                        refresh_picker_entries(picker)?;
                    }
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
        KeyCode::Char(c) => {
            if app.page == TuiPage::Setup && app.running.is_none() {
                app.selected_field_mut().push(c);
            }
        }
        _ => {}
    }
    Ok(())
}

fn run_tui(cli: &Cli) -> Result<()> {
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
                Event::Key(key) => {
                    if let Err(e) = handle_key(&mut app, key) {
                        app.status = "Input error".to_string();
                        app.append_log(format!("Input error: {e}"));
                    }
                }
                Event::Resize(_, _) => {
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

fn send_event(on_event: &mut dyn FnMut(RunnerEvent), ev: RunnerEvent) {
    on_event(ev);
}

fn check_cancel(stop: &AtomicBool) -> Result<()> {
    if stop.load(Ordering::Relaxed) {
        bail!("cancelled by user");
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
        match rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
        {
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("initialized") {
                    return Err(anyhow!(
                        "failed to configure Rayon global thread pool: {msg}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn run_inference_with_events(
    cli: Cli,
    stop: &AtomicBool,
    on_event: &mut dyn FnMut(RunnerEvent),
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

    check_cancel(stop)?;
    let input_file = resolve_input_path(&raw_input, cli.input_format)?;
    if input_file != raw_input {
        send_event(
            on_event,
            RunnerEvent::Log(format!(
                "Input is a directory, selected file: {}",
                input_file.display()
            )),
        );
    }

    if cli.bootstrap > 0 && cli.bootstrap_block_size == 0 {
        bail!("--bootstrap-block-size must be > 0 when --bootstrap is enabled");
    }
    let bootstrap_iters = cli.bootstrap_iters.unwrap_or(n_iter);
    if cli.bootstrap > 0 && bootstrap_iters == 0 {
        bail!("bootstrap requires EM iterations; set N_ITER > 0 or use --bootstrap-iters");
    }

    configure_threads(cli.threads)?;

    let io_total = fs::metadata(&input_file).map(|m| m.len()).unwrap_or(1);
    send_event(
        on_event,
        RunnerEvent::Stage {
            stage: StageKind::Io,
            done: 0,
            total: io_total,
            iter: None,
        },
    );
    let format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };
    let obs = match cli.input_format {
        InputFormat::Psmcfa => read_psmcfa(&input_file, cli.batch_size),
        InputFormat::Mhs => read_mhs(&input_file, cli.batch_size, cli.mhs_bin_size),
    }
    .with_context(|| format!("failed to read {format_name} input"))?;
    send_event(
        on_event,
        RunnerEvent::Stage {
            stage: StageKind::Io,
            done: io_total,
            total: io_total,
            iter: None,
        },
    );
    send_event(
        on_event,
        RunnerEvent::Log(format!(
            "Loaded {} rows from {}",
            obs.rows.len(),
            input_file.display()
        )),
    );

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
        if frac >= 1.0 {
            f64::INFINITY
        } else {
            -(1.0 - frac).ln()
        }
    });
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);
    let pattern = cli.pattern.clone();
    let mut model = PsmcModel::new(cli.t_max, cli.n_steps, theta0, rho0, cli.mu, pattern)?;

    let mut base_mstep_config = MStepConfig::default();
    base_mstep_config.max_iters = cli.mstep_iters.unwrap_or(100);
    base_mstep_config.lambda = cli.smooth_lambda;
    base_mstep_config.progress = false;

    let mut last_loglike = None;
    if n_iter > 0 {
        let mut on_progress = |ev: EmProgressEvent| match ev {
            EmProgressEvent::IterStart { iter, total_iters } => {
                send_event(
                    on_event,
                    RunnerEvent::Log(format!("EM iteration {iter}/{total_iters}")),
                );
            }
            EmProgressEvent::EStep {
                iter,
                total_iters,
                phase,
                done,
                total,
            } => {
                let stage = match phase {
                    EStepPhase::Forward => StageKind::EForward,
                    EStepPhase::Backward => StageKind::EBackward,
                };
                send_event(
                    on_event,
                    RunnerEvent::Stage {
                        stage,
                        done,
                        total,
                        iter: Some((iter, total_iters)),
                    },
                );
            }
            EmProgressEvent::MStep {
                iter,
                total_iters,
                done,
                total,
            } => {
                send_event(
                    on_event,
                    RunnerEvent::Stage {
                        stage: StageKind::MStep,
                        done: done as u64,
                        total: total as u64,
                        iter: Some((iter, total_iters)),
                    },
                );
            }
        };
        let history = em_train_with_progress(
            &mut model,
            &obs.rows,
            &obs.row_starts,
            n_iter,
            &base_mstep_config,
            Some(&mut on_progress),
        )?;
        if let Some((before, _)) = history.loglike.last() {
            last_loglike = Some(*before);
            send_event(
                on_event,
                RunnerEvent::Log(format!("Last EM loglike: {before}")),
            );
        }
        if let Some(msg) = near_bounds_warning(&model, &base_mstep_config)? {
            send_event(on_event, RunnerEvent::Log(msg));
        }
    }

    check_cancel(stop)?;
    let report_bin_size = match cli.input_format {
        InputFormat::Psmcfa => 100.0,
        InputFormat::Mhs => cli.mhs_bin_size as f64,
    };
    let input_format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

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
        bootstrap_dir_out = Some(bootstrap_dir.clone());

        send_event(
            on_event,
            RunnerEvent::Stage {
                stage: StageKind::Bootstrap,
                done: 0,
                total: cli.bootstrap as u64,
                iter: None,
            },
        );

        let seqs = rows_to_sequences(&obs.rows, &obs.row_starts)?;
        send_event(
            on_event,
            RunnerEvent::Log(format!(
                "Bootstrap: running {} replicates in parallel",
                cli.bootstrap
            )),
        );

        let mut boot_results = (0..cli.bootstrap)
            .into_par_iter()
            .map(|rep| -> Result<(usize, PsmcModel)> {
                if stop.load(Ordering::Relaxed) {
                    bail!("cancelled by user");
                }
                let mut rng = new_rng(bootstrap_replicate_seed(cli.bootstrap_seed, rep));
                let sampled =
                    sample_sequences_block_bootstrap(&seqs, cli.bootstrap_block_size, &mut rng)?;
                let obs_rep = sequences_to_observations(&sampled, cli.batch_size)?;

                let mut model_rep = model.clone();
                let mut rep_cfg = base_mstep_config.clone();
                rep_cfg.progress = false;
                let _ = em_train(
                    &mut model_rep,
                    &obs_rep.rows,
                    &obs_rep.row_starts,
                    bootstrap_iters,
                    &rep_cfg,
                )?;
                model_rep.param_recalculate()?;

                let rep_json = replicate_json_path(&bootstrap_dir, rep + 1);
                model_rep.save_params(&rep_json)?;
                Ok((rep, model_rep))
            })
            .collect::<Result<Vec<_>>>()?;
        check_cancel(stop)?;
        boot_results.sort_by_key(|(rep, _)| *rep);

        let mut boot_models = Vec::<PsmcModel>::with_capacity(cli.bootstrap);
        for (i, (_, model_rep)) in boot_results.into_iter().enumerate() {
            boot_models.push(model_rep);
            send_event(
                on_event,
                RunnerEvent::Stage {
                    stage: StageKind::Bootstrap,
                    done: (i + 1) as u64,
                    total: cli.bootstrap as u64,
                    iter: None,
                },
            );
        }

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
        send_event(
            on_event,
            RunnerEvent::Log(format!("Bootstrap summary TSV: {}", summary_tsv.display())),
        );
        send_event(
            on_event,
            RunnerEvent::Log(format!(
                "Bootstrap summary JSON: {}",
                summary_json.display()
            )),
        );
        bootstrap_report_data = Some(bootstrap_summary);
    }

    check_cancel(stop)?;
    let mut tmrca_report_data = None;
    let mut tmrca_out_path = None;
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
            false,
            20_000,
        )?;
        send_event(
            on_event,
            RunnerEvent::Log(format!("TMRCA posterior: {}", tmrca_out.display())),
        );
        tmrca_report_data = Some(tmrca);
        tmrca_out_path = Some(tmrca_out.clone());
    }

    model.save_params(&output_file)?;

    let report_html = default_html_path(&output_file);
    write_html_report(
        &model,
        &input_file,
        &output_file,
        &report_html,
        n_iter,
        input_format_name,
        report_bin_size,
        tmrca_report_data.as_ref(),
        bootstrap_report_data.as_ref(),
    )?;
    send_event(
        on_event,
        RunnerEvent::Log(format!("HTML report: {}", report_html.display())),
    );

    Ok(RunSummary {
        output_json: output_file,
        html_report: report_html,
        tmrca_out: tmrca_out_path,
        bootstrap_dir: bootstrap_dir_out,
        last_loglike,
        elapsed_sec: 0.0,
    })
}

fn run_inference(cli: Cli) -> Result<()> {
    let raw_input = cli
        .input_file
        .clone()
        .ok_or_else(|| anyhow!("missing INPUT_FILE"))?;
    let input_file = resolve_input_path(&raw_input, cli.input_format)?;
    if input_file != raw_input {
        println!(
            "Input is a directory, selected file: {}",
            input_file.display()
        );
    }
    let output_file = cli
        .output_file
        .clone()
        .ok_or_else(|| anyhow!("missing OUTPUT_FILE"))?;
    let n_iter = cli.n_iter.ok_or_else(|| anyhow!("missing N_ITER"))?;

    if cli.bootstrap > 0 && cli.bootstrap_block_size == 0 {
        bail!("--bootstrap-block-size must be > 0 when --bootstrap is enabled");
    }
    let bootstrap_iters = cli.bootstrap_iters.unwrap_or(n_iter);
    if cli.bootstrap > 0 && bootstrap_iters == 0 {
        bail!("bootstrap requires EM iterations; set N_ITER > 0 or use --bootstrap-iters");
    }

    configure_threads(cli.threads)?;

    let pattern = cli.pattern.clone();

    let format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

    let obs = if cli.no_progress {
        match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&input_file, cli.batch_size, cli.mhs_bin_size),
        }
        .with_context(|| format!("failed to read {format_name} input"))?
    } else {
        let pb = progress::spinner("IO", &format!("Reading {format_name}"));
        let obs = match cli.input_format {
            InputFormat::Psmcfa => read_psmcfa(&input_file, cli.batch_size),
            InputFormat::Mhs => read_mhs(&input_file, cli.batch_size, cli.mhs_bin_size),
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
        if frac >= 1.0 {
            f64::INFINITY
        } else {
            -(1.0 - frac).ln()
        }
    });
    let rho0 = cli.rho0.unwrap_or_else(|| theta0 / 5.0);

    let mut model = PsmcModel::new(cli.t_max, cli.n_steps, theta0, rho0, cli.mu, pattern)?;

    let mut base_mstep_config = MStepConfig::default();
    base_mstep_config.max_iters = cli.mstep_iters.unwrap_or(100);
    base_mstep_config.lambda = cli.smooth_lambda;
    if cli.no_progress {
        base_mstep_config.progress = false;
    }

    if n_iter > 0 {
        let history = em_train(
            &mut model,
            &obs.rows,
            &obs.row_starts,
            n_iter,
            &base_mstep_config,
        )?;
        if let Some((before, _)) = history.loglike.last() {
            println!("Last EM loglike: {}", before);
        }
        warn_if_near_bounds(&model, &base_mstep_config)?;
    }

    let report_bin_size = match cli.input_format {
        InputFormat::Psmcfa => 100.0,
        InputFormat::Mhs => cli.mhs_bin_size as f64,
    };
    let input_format_name = match cli.input_format {
        InputFormat::Psmcfa => "psmcfa",
        InputFormat::Mhs => "mhs",
    };

    model.param_recalculate()?;

    let mut bootstrap_report_data = None;
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

        let seqs = rows_to_sequences(&obs.rows, &obs.row_starts)?;

        let pb_boot = if !cli.no_progress {
            Some(progress::bar(cli.bootstrap as u64, "BOOT", "replicates"))
        } else {
            None
        };
        if let Some(pb) = &pb_boot {
            pb.set_message("parallel replicates");
        }
        let pb_boot_threads = pb_boot.clone();
        let mut boot_results = (0..cli.bootstrap)
            .into_par_iter()
            .map(|rep| -> Result<(usize, PsmcModel)> {
                let mut rng = new_rng(bootstrap_replicate_seed(cli.bootstrap_seed, rep));
                let sampled =
                    sample_sequences_block_bootstrap(&seqs, cli.bootstrap_block_size, &mut rng)?;
                let obs_rep = sequences_to_observations(&sampled, cli.batch_size)?;

                let mut model_rep = model.clone();
                let mut rep_cfg = base_mstep_config.clone();
                rep_cfg.progress = false;
                let _ = em_train(
                    &mut model_rep,
                    &obs_rep.rows,
                    &obs_rep.row_starts,
                    bootstrap_iters,
                    &rep_cfg,
                )?;
                model_rep.param_recalculate()?;

                let rep_json = replicate_json_path(&bootstrap_dir, rep + 1);
                model_rep.save_params(&rep_json)?;
                if let Some(pb) = &pb_boot_threads {
                    pb.inc(1);
                }
                Ok((rep, model_rep))
            })
            .collect::<Result<Vec<_>>>()?;
        boot_results.sort_by_key(|(rep, _)| *rep);
        let mut boot_models = Vec::<PsmcModel>::with_capacity(cli.bootstrap);
        for (_, model_rep) in boot_results {
            boot_models.push(model_rep);
        }
        if let Some(pb) = pb_boot {
            pb.finish_with_message("bootstrap done");
        }

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
        println!("Bootstrap dir: {}", bootstrap_dir.display());
        println!("Bootstrap summary TSV: {}", summary_tsv.display());
        println!("Bootstrap summary JSON: {}", summary_json.display());
        bootstrap_report_data = Some(bootstrap_summary);
    }

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

    model.save_params(&output_file)?;

    let report_html = default_html_path(&output_file);
    write_html_report(
        &model,
        &input_file,
        &output_file,
        &report_html,
        n_iter,
        input_format_name,
        report_bin_size,
        tmrca_report_data.as_ref(),
        bootstrap_report_data.as_ref(),
    )?;
    println!("HTML report: {}", report_html.display());

    Ok(())
}

fn near(v: f64, lo: f64, hi: f64) -> (bool, bool) {
    let span = (hi - lo).abs().max(1.0);
    let tol = span * 1e-6;
    ((v - lo).abs() <= tol, (v - hi).abs() <= tol)
}

fn near_bounds_warning(model: &PsmcModel, cfg: &MStepConfig) -> Result<Option<String>> {
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
        return Ok(Some(format!(
            "Warning: {} params are on/near optimization bounds (lo={}, hi={}). Examples: {}. Consider wider bounds.",
            near_lo + near_hi,
            near_lo,
            near_hi,
            examples.join(", ")
        )));
    }
    Ok(None)
}

fn warn_if_near_bounds(model: &PsmcModel, cfg: &MStepConfig) -> Result<()> {
    if let Some(msg) = near_bounds_warning(model, cfg)? {
        eprintln!("{msg}");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.tui {
        return run_tui(&cli);
    }
    run_inference(cli)
}
