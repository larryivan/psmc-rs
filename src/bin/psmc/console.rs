//! Pipeline events on a terminal: results on stdout, messages and progress
//! bars on stderr.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::{self, Write};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

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

/// Print `e` with its causes on stderr as clap prints its own errors, after
/// `error: ` (coloured on a terminal).
pub fn print_error(e: &anyhow::Error) {
    use clap::CommandFactory;
    let error = clap::Error::raw(clap::error::ErrorKind::Io, format!("{e:#}\n"));
    let _ = error.with_cmd(&crate::cli::Cli::command()).print();
}

/// Prints [`Event`]s for the command line.
pub struct Console {
    quiet: bool,
    results_on_stderr: bool,
    bars: Option<Bars>,
}

impl Console {
    /// `quiet` hides information lines (warnings and results still show);
    /// `progress` draws progress bars.
    pub fn new(quiet: bool, progress: bool) -> Self {
        let bars = progress.then(|| Bars {
            multi: MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(15)),
            shown: Mutex::default(),
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

    /// Print `line` with `print` (on stdout or stderr) above the progress
    /// bars.
    fn line(&self, print: fn(&str), line: &str) {
        let line = format!("{line}\n");
        match &self.bars {
            Some(b) => b.multi.suspend(|| print(&line)),
            None => print(&line),
        }
    }

    /// Show one event of the pipeline.
    pub fn emit(&self, ev: Event) {
        match ev {
            Event::Info(line) if !self.quiet => self.line(print_err, &line),
            Event::Warning(line) => self.line(print_err, &format!("warning: {line}")),
            Event::Qc { warnings, .. } if self.quiet => {
                for w in warnings {
                    self.line(print_err, &format!("warning: {w}"));
                }
            }
            Event::Qc { text, .. } => self.line(print_err, text.trim_end()),
            Event::Output(line) => {
                // A result ends the stage before it.
                self.finish();
                let print = if self.results_on_stderr {
                    print_err
                } else {
                    print_out
                };
                self.line(print, &line);
            }
            Event::Iteration { .. } | Event::Progress { .. } => {
                if let Some(b) = &self.bars {
                    b.show(ev);
                }
            }
            Event::Info(_) => {}
        }
    }

    /// Remove the progress bars.
    pub fn finish(&self) {
        if let Some(b) = &self.bars {
            let mut shown = b.shown.lock().unwrap_or_else(PoisonError::into_inner);
            let stage = shown.stage.take().map(|(_, pb)| pb);
            for pb in shown.em.take().into_iter().chain(stage) {
                b.remove(pb);
            }
        }
    }
}

/// The progress bars of a run: the EM iterations, and below them the stage
/// under way (reading, an E- or M-step phase, the bootstrap).
struct Bars {
    multi: MultiProgress,
    shown: Mutex<Shown>,
}

#[derive(Default)]
struct Shown {
    em: Option<ProgressBar>,
    stage: Option<(Stage, ProgressBar)>,
}

/// A bar of `len` steps, or a spinner without; it fills the terminal's
/// width. It starts hidden: a bar drawn on stderr before it joins the
/// [`MultiProgress`] leaves a line there that is never cleared.
fn bar(len: Option<u64>, prefix: &str, msg: &str) -> ProgressBar {
    let style = match len {
        Some(_) => ProgressStyle::with_template(
            "{prefix:.bold} {msg:.bold} [{elapsed_precise}] {wide_bar:.cyan/blue} \
             {human_pos}/{human_len} {percent:>3}% ETA {eta_precise}",
        )
        .map(|s| s.progress_chars("█▇▆▅▄▃▂▁ ")),
        None => ProgressStyle::with_template(
            "{prefix:.bold} {spinner:.magenta} {msg:.bold} [{elapsed_precise}]",
        )
        .map(|s| s.tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")),
    };
    ProgressBar::with_draw_target(len, ProgressDrawTarget::hidden())
        .with_style(style.expect("a valid template"))
        .with_prefix(prefix.to_string())
        .with_message(msg.to_string())
}

impl Bars {
    /// Show an iteration or progress event.
    fn show(&self, ev: Event) {
        let mut shown = self.shown.lock().unwrap_or_else(PoisonError::into_inner);
        match ev {
            Event::Iteration { iter, total } => {
                let em = shown.em.get_or_insert_with(|| {
                    self.multi
                        .insert(0, bar(Some(total as u64), "EM", "iterations"))
                });
                em.set_position(iter.saturating_sub(1) as u64);
                em.set_message(format!("iteration {iter}/{total}"));
            }
            Event::Progress {
                stage, done, total, ..
            } => {
                let read = stage == Stage::Io && done >= total;
                if (read || shown.stage.as_ref().is_some_and(|(s, _)| *s != stage))
                    && let Some((_, pb)) = shown.stage.take()
                {
                    self.remove(pb);
                }
                if read {
                    return;
                }
                let (_, pb) = shown.stage.get_or_insert_with(|| {
                    let pb = self.multi.add(match stage {
                        Stage::Io => bar(None, "IO", "reading input"),
                        Stage::EForward => bar(Some(total), "  ", "E-step forward"),
                        Stage::EBackward => bar(Some(total), "  ", "E-step backward"),
                        Stage::MStep => bar(Some(total), "  ", "M-step"),
                        Stage::Bootstrap => bar(Some(total), "BOOT", "replicates"),
                    });
                    if stage == Stage::Io {
                        pb.enable_steady_tick(Duration::from_millis(80));
                    }
                    (stage, pb)
                });
                // The M-step's length is its cap until its last event.
                pb.set_length(total);
                pb.set_position(done.min(total));
            }
            _ => {}
        }
    }

    fn remove(&self, pb: ProgressBar) {
        pb.finish_and_clear();
        self.multi.remove(&pb);
    }
}
