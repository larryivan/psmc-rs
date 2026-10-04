//! Running per-sequence work in parallel with a result that does not depend
//! on the thread count, and reporting progress on the calling thread.

use anyhow::Result;
use indicatif::ProgressBar;
use rayon::ThreadPool;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use super::fb::Progress;
use super::{EStepPhase, EStepProgress};
use crate::progress;

/// Upper bound on the number of work items. Sequences are grouped into at
/// most this many contiguous chunks, each processed serially into one partial
/// result; the partials are then summed in chunk order. The grouping depends
/// only on the number of sequences, so results are the same for any thread
/// count, and memory for partial sums stays bounded however many sequences
/// (e.g. scaffolds) the input has.
const MAX_WORK_ITEMS: usize = 256;

/// Rows of each sequence and the grouping of sequences into work items.
pub(crate) struct Plan {
    /// Row range of each sequence.
    pub(crate) seqs: Vec<Range<usize>>,
    /// Sequence range of each work item.
    pub(crate) items: Vec<Range<usize>>,
    pub(crate) total_sites: u64,
}

impl Plan {
    /// `rows` and `row_starts` must have been validated.
    pub(crate) fn new(rows: &[Vec<u8>], row_starts: &[bool]) -> Self {
        let mut seqs = Vec::new();
        let mut start = 0usize;
        for (i, &starts_seq) in row_starts.iter().enumerate().skip(1) {
            if starts_seq {
                seqs.push(start..i);
                start = i;
            }
        }
        seqs.push(start..rows.len());
        let per_item = seqs.len().div_ceil(MAX_WORK_ITEMS);
        let items = (0..seqs.len())
            .step_by(per_item)
            .map(|s| s..(s + per_item).min(seqs.len()))
            .collect();
        let total_sites = rows.iter().map(|r| r.len() as u64).sum();
        Self {
            seqs,
            items,
            total_sites,
        }
    }
}

/// Turns the shared site counters into progress bars and callback events.
///
/// Events are delivered on the calling thread. Forward events come first; as
/// sequences run in parallel, backward work can start before every forward
/// pass has finished, so backward events are held back until the forward
/// count is complete. Within each phase `done` only increases.
pub(crate) struct Reporter<'a> {
    total: u64,
    with_backward: bool,
    bars: bool,
    label: &'a str,
    on_progress: Option<&'a mut dyn FnMut(EStepProgress)>,
    last_forward: Option<u64>,
    last_backward: Option<u64>,
    bar: Option<ProgressBar>,
}

impl<'a> Reporter<'a> {
    pub(crate) fn new(
        total: u64,
        with_backward: bool,
        bars: bool,
        label: &'a str,
        on_progress: Option<&'a mut dyn FnMut(EStepProgress)>,
    ) -> Self {
        Self {
            total,
            with_backward,
            bars: bars && total > 0,
            label,
            on_progress,
            last_forward: None,
            last_backward: None,
            bar: None,
        }
    }

    fn active(&self) -> bool {
        self.bars || self.on_progress.is_some()
    }

    fn emit(&mut self, phase: EStepPhase, done: u64) {
        if let Some(cb) = self.on_progress.as_mut() {
            (*cb)(EStepProgress {
                phase,
                done,
                total: self.total,
            });
        }
    }

    fn update(&mut self, progress: &Progress) {
        let fwd = progress.forward.load(Ordering::Relaxed).min(self.total);
        if self.last_forward != Some(fwd) {
            if self.last_forward.is_none() && self.bars {
                self.bar = Some(progress::bar_raw(self.total, self.label, "forward"));
            }
            self.last_forward = Some(fwd);
            if let Some(bar) = &self.bar {
                bar.set_position(fwd);
            }
            self.emit(EStepPhase::Forward, fwd);
        }
        if !self.with_backward || fwd < self.total {
            return;
        }
        let bwd = progress.backward.load(Ordering::Relaxed).min(self.total);
        if self.last_backward != Some(bwd) {
            if self.last_backward.is_none()
                && let Some(bar) = self.bar.take()
            {
                bar.finish_with_message(format!("{} forward done", self.label));
                self.bar = Some(progress::bar_raw(self.total, self.label, "backward"));
            }
            self.last_backward = Some(bwd);
            if let Some(bar) = &self.bar {
                bar.set_position(bwd);
            }
            self.emit(EStepPhase::Backward, bwd);
        }
    }

    fn finish(&mut self, progress: &Progress) {
        self.update(progress);
        if let Some(bar) = self.bar.take() {
            let phase = if self.with_backward {
                "backward"
            } else {
                "forward"
            };
            bar.finish_with_message(format!("{} {phase} done", self.label));
        }
    }
}

/// Sends a completion message when dropped, so the progress pump also learns
/// about work items that unwind.
struct DoneGuard(mpsc::Sender<()>);

impl Drop for DoneGuard {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

const POLL: Duration = Duration::from_millis(50);

/// Run `work(i)` for every `i < n_items`, in parallel, and return the results
/// in item order.
///
/// Parallelism comes from the pool the caller runs in (or `pool`), at any
/// nesting level: inside a rayon worker the items are simply more tasks for
/// that pool. With an active reporter the calling thread pumps progress
/// events (helping with pending rayon work in between when it is a worker).
pub(crate) fn run<T, F>(
    n_items: usize,
    pool: Option<&ThreadPool>,
    reporter: &mut Reporter<'_>,
    progress: &Progress,
    work: F,
) -> Vec<Result<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync,
{
    if !reporter.active() {
        let all = || (0..n_items).into_par_iter().map(&work).collect::<Vec<_>>();
        return match pool {
            Some(p) => p.install(all),
            None => all(),
        };
    }

    let slots: Vec<Mutex<Option<Result<T>>>> = (0..n_items).map(|_| Mutex::new(None)).collect();
    match pool {
        Some(p) => {
            p.in_place_scope(|scope| spawn_and_pump(scope, &work, &slots, reporter, progress))
        }
        None => {
            rayon::in_place_scope(|scope| spawn_and_pump(scope, &work, &slots, reporter, progress))
        }
    }
    slots
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .expect("every work item stores its result")
        })
        .collect()
}

fn spawn_and_pump<'scope, T, F>(
    scope: &rayon::Scope<'scope>,
    work: &'scope F,
    slots: &'scope [Mutex<Option<Result<T>>>],
    reporter: &mut Reporter<'_>,
    progress: &Progress,
) where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync,
{
    let (tx, rx) = mpsc::channel::<()>();
    for (i, slot) in slots.iter().enumerate() {
        let done = DoneGuard(tx.clone());
        scope.spawn(move |_| {
            let _done = done;
            let r = work(i);
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        });
    }
    let mut remaining = slots.len();
    reporter.update(progress);
    while remaining > 0 {
        while rx.try_recv().is_ok() {
            remaining -= 1;
        }
        if remaining == 0 {
            break;
        }
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => {}
            _ => {
                if rx.recv_timeout(POLL).is_ok() {
                    remaining -= 1;
                }
            }
        }
        reporter.update(progress);
    }
}

/// Final progress events after successful work.
pub(crate) fn finish(reporter: &mut Reporter<'_>, progress: &Progress) {
    reporter.finish(progress);
}
