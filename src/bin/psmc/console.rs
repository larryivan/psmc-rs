//! Pipeline events on a terminal: results on stdout, messages and progress
//! bars on stderr.

use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget};
use std::sync::Mutex;

use psmc_rs::progress;

use crate::run::{Event, Stage};

/// Prints [`Event`]s for the command line.
pub struct Console {
    quiet: bool,
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
        Self { quiet, bars }
    }

    /// Print `line` on stderr without breaking the progress bars.
    fn eprint(&self, line: &str) {
        match &self.bars {
            Some(b) => b.multi.suspend(|| eprintln!("{line}")),
            None => eprintln!("{line}"),
        }
    }

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
            Event::Output(line) => match &self.bars {
                Some(b) => b.multi.suspend(|| println!("{line}")),
                None => println!("{line}"),
            },
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
                let pb = s.bootstrap.get_or_insert_with(|| {
                    self.add(progress::bar_raw(total, "BOOT", "replicates"))
                });
                pb.set_position(done.min(total));
            }
        }
    }
}
