//! Pipeline events on a terminal: results on stdout, messages and progress
//! bars on stderr.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget};
use std::io::{self, Write};
use std::sync::Mutex;

use psmc_rs::progress;

use crate::run::{Event, Stage};

/// Print `text` on stdout. A closed stdout (a reader that quit early, as
/// `| head` does) only loses the text: the run goes on and writes its
/// files, where `print!` would panic.
pub fn print_out(text: &str) {
    let _ = io::stdout().lock().write_all(text.as_bytes());
}

/// Print `text` on stderr, ignoring a closed stderr as [`print_out`] does.
pub fn print_err(text: &str) {
    let _ = io::stderr().lock().write_all(text.as_bytes());
}

/// Prints [`Event`]s for the command line.
pub struct Console {
    quiet: bool,
    results_on_stderr: bool,
    bars: Option<Bars>,
}

/// The progress bars of a run: one for reading, one for the EM iterations,
/// one for the current E- or M-step phase, and one for the bootstrap.
struct Bars {
    multi: MultiProgress,
    state: Mutex<BarState>,
}

#[derive(Default)]
struct BarState {
    io: Option<ProgressBar>,
    em: Option<ProgressBar>,
    phase: Option<(Stage, ProgressBar)>,
    bootstrap: Option<ProgressBar>,
}

impl Console {
    /// `quiet` hides information lines (warnings and results still show);
    /// `progress` draws progress bars.
    pub fn new(quiet: bool, progress: bool) -> Self {
        let bars = progress.then(|| Bars {
            multi: MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(15)),
            state: Mutex::new(BarState::default()),
        });
        Self {
            quiet,
            results_on_stderr: false,
            bars,
        }
    }

    /// Print results on stderr too, keeping stdout for a report written
    /// there (`--qc-json -`).
    pub fn results_on_stderr(mut self, yes: bool) -> Self {
        self.results_on_stderr = yes;
        self
    }

    /// Print `line` on stderr without breaking the progress bars.
    fn eprint(&self, line: &str) {
        let line = format!("{line}\n");
        match &self.bars {
            Some(b) => b.multi.suspend(|| print_err(&line)),
            None => print_err(&line),
        }
    }

    /// Show one event of the pipeline.
    pub fn emit(&self, ev: Event) {
        match ev {
            Event::Info(line) => {
                if !self.quiet {
                    self.eprint(&line);
                }
            }
            Event::Warning(line) => self.eprint(&format!("Warning: {line}")),
            Event::Qc { text, warnings } => {
                if self.quiet {
                    for w in warnings {
                        self.eprint(&format!("Warning: {w}"));
                    }
                } else {
                    self.eprint(text.trim_end());
                }
            }
            Event::Output(line) if self.results_on_stderr => self.eprint(&line),
            Event::Output(line) => {
                let line = format!("{line}\n");
                match &self.bars {
                    Some(b) => b.multi.suspend(|| print_out(&line)),
                    None => print_out(&line),
                }
            }
            Event::Iteration { iter, total } => {
                if let Some(b) = &self.bars {
                    b.iteration(iter, total);
                }
            }
            Event::Progress {
                stage, done, total, ..
            } => {
                if let Some(b) = &self.bars {
                    b.progress(stage, done, total);
                }
            }
        }
    }

    /// Remove the progress bars.
    pub fn finish(&self) {
        if let Some(b) = &self.bars {
            let mut s = b.state.lock().unwrap_or_else(|e| e.into_inner());
            for pb in [s.io.take(), s.em.take(), s.bootstrap.take()]
                .into_iter()
                .flatten()
                .chain(s.phase.take().map(|(_, pb)| pb))
            {
                pb.finish_and_clear();
            }
        }
    }
}

impl Bars {
    fn add(&self, pb: ProgressBar) -> ProgressBar {
        self.multi.add(pb)
    }

    fn iteration(&self, iter: usize, total: usize) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let em = s
            .em
            .get_or_insert_with(|| self.add(progress::bar_raw(total as u64, "EM", "iterations")));
        em.set_position(iter.saturating_sub(1) as u64);
        em.set_message(format!("iteration {iter}/{total}"));
    }

    fn progress(&self, stage: Stage, done: u64, total: u64) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match stage {
            Stage::Io => {
                if done >= total {
                    if let Some(pb) = s.io.take() {
                        pb.finish_and_clear();
                    }
                } else if s.io.is_none() {
                    let pb = self.add(progress::spinner_raw("IO", "reading input"));
                    pb.enable_steady_tick(std::time::Duration::from_millis(80));
                    s.io = Some(pb);
                }
            }
            Stage::EForward | Stage::EBackward | Stage::MStep => {
                let label = match stage {
                    Stage::EForward => "E-step forward",
                    Stage::EBackward => "E-step backward",
                    _ => "M-step",
                };
                let reuse = matches!(&s.phase, Some((st, pb)) if *st == stage && pb.length() == Some(total));
                if !reuse {
                    if let Some((_, pb)) = s.phase.take() {
                        pb.finish_and_clear();
                    }
                    let pb = self.add(progress::bar_raw(total, "  ", label));
                    s.phase = Some((stage, pb));
                }
                if let Some((_, pb)) = &s.phase {
                    pb.set_position(done.min(total));
                }
            }
            Stage::Bootstrap => {
                if s.bootstrap.is_none() {
                    // The main fit is over.
                    for pb in
                        s.em.take()
                            .into_iter()
                            .chain(s.phase.take().map(|(_, pb)| pb))
                    {
                        pb.finish_and_clear();
                    }
                }
                let pb = s.bootstrap.get_or_insert_with(|| {
                    self.add(progress::bar_raw(total, "BOOT", "replicates"))
                });
                pb.set_position(done.min(total));
            }
        }
    }
}
