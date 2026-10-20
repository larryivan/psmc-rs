//! Running per-sequence work in parallel with a result that does not depend
//! on the thread count, and reporting progress on the calling thread.

use anyhow::Result;
use indicatif::ProgressBar;
use rayon::ThreadPool;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use super::fb::Progress;
use super::{EStepPhase, EStepProgress};
use crate::progress;

/// Upper bound on the number of work items. Sequences are grouped into at
/// most this many contiguous chunks, each processed serially into one partial
/// result; the partials are then summed in chunk order ([`fold_in_order`]).
/// The grouping depends only on the number of sequences, so results are the
/// same for any thread count, and the number of partial results stays
/// bounded however many sequences (e.g. scaffolds) the input has.
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

    /// Final progress events after successful work.
    pub(crate) fn finish(&mut self, progress: &Progress) {
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

impl Drop for Reporter<'_> {
    fn drop(&mut self) {
        // Still set only if the work failed or was cancelled.
        if let Some(bar) = self.bar.take() {
            bar.abandon();
        }
    }
}

/// Sends a completion message when dropped, so the progress pump also learns
/// about workers that unwind.
struct DoneGuard(mpsc::Sender<()>);

impl Drop for DoneGuard {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

const POLL: Duration = Duration::from_millis(50);

/// The fold of [`fold_in_order`]: the results merged so far, and those that
/// finished before an item ahead of them.
struct Ordered<T, A> {
    acc: A,
    /// Results that finished before every item ahead of them did, by item.
    held: BTreeMap<usize, T>,
    /// The next item to merge.
    next: usize,
    /// The error of the earliest failed item so far.
    error: Option<(usize, anyhow::Error)>,
}

impl<T, A> Ordered<T, A> {
    /// Take the result of item `i`, and merge every result that is now next
    /// in item order. Once an item has failed, the results are no longer
    /// merged, and only an earlier item's error replaces its own.
    fn deposit(&mut self, i: usize, result: Result<T>, merge: &impl Fn(&mut A, T)) {
        match result {
            Err(e) => {
                if self.error.as_ref().is_none_or(|&(j, _)| i < j) {
                    self.error = Some((i, e));
                }
                self.held.clear();
            }
            Ok(t) if self.error.is_none() => {
                self.held.insert(i, t);
                while let Some(t) = self.held.remove(&self.next) {
                    merge(&mut self.acc, t);
                    self.next += 1;
                }
            }
            Ok(_) => {}
        }
    }
}

/// Run `work(i)` for every `i < n_items`, in parallel, and fold the results
/// into `acc` with `merge` in item order, as `for i in 0..n_items { merge(&mut
/// acc, work(i)?) }` would, so that the result does not depend on the thread
/// count. Returns the error of the earliest failed item.
///
/// Items start in item order, from a shared counter, and each result is
/// merged as soon as every item before it has been. So at any time the fold
/// holds only the results that finished before an earlier item, about one
/// per thread when the items take similar times, rather than one per item:
/// with a large result per item (the E-step's sums per observation index
/// and state) and many items (a bootstrap of many trunks, an assembly of
/// many sequences), keeping every result until the end would take far more
/// memory than the passes themselves. After an item fails, no new item
/// starts; the items before it have all started and finish.
///
/// Parallelism comes from the pool the caller runs in (or `pool`), at any
/// nesting level: inside a rayon worker the workers that pull items are
/// simply more tasks for that pool. With an active reporter the calling
/// thread pumps progress events (helping with pending rayon work in between
/// when it is a worker).
pub(crate) fn fold_in_order<T, A, F, M>(
    n_items: usize,
    pool: Option<&ThreadPool>,
    reporter: &mut Reporter<'_>,
    progress: &Progress,
    work: F,
    acc: A,
    merge: M,
) -> Result<A>
where
    T: Send,
    A: Send,
    F: Fn(usize) -> Result<T> + Sync,
    M: Fn(&mut A, T) + Sync,
{
    let state = Mutex::new(Ordered {
        acc,
        held: BTreeMap::new(),
        next: 0,
        error: None,
    });
    let next_item = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let worker = || {
        while !failed.load(Ordering::Relaxed) {
            let i = next_item.fetch_add(1, Ordering::Relaxed);
            if i >= n_items {
                break;
            }
            let result = work(i);
            if result.is_err() {
                failed.store(true, Ordering::Relaxed);
            }
            state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .deposit(i, result, &merge);
        }
    };
    let threads = pool.map_or_else(rayon::current_num_threads, ThreadPool::current_num_threads);
    let n_workers = threads.clamp(1, n_items.max(1));
    if reporter.active() {
        let pump = |scope: &_| spawn_and_pump(scope, n_workers, &worker, reporter, progress);
        match pool {
            Some(p) => p.in_place_scope(pump),
            None => rayon::in_place_scope(pump),
        }
    } else {
        let all = || (0..n_workers).into_par_iter().for_each(|_| worker());
        match pool {
            Some(p) => p.install(all),
            None => all(),
        }
    }

    let state = state.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some((_, e)) = state.error {
        return Err(e);
    }
    debug_assert_eq!(state.next, n_items, "every item is merged");
    Ok(state.acc)
}

/// Run `n_workers` copies of `worker` in `scope` while the calling thread
/// pumps `reporter`'s progress events until they are done.
fn spawn_and_pump<'scope, W>(
    scope: &rayon::Scope<'scope>,
    n_workers: usize,
    worker: &'scope W,
    reporter: &mut Reporter<'_>,
    progress: &Progress,
) where
    W: Fn() + Sync,
{
    let (tx, rx) = mpsc::channel::<()>();
    for _ in 0..n_workers {
        let done = DoneGuard(tx.clone());
        scope.spawn(move |_| {
            let _done = done;
            worker();
        });
    }
    let mut remaining = n_workers;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(threads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    /// Items finish out of order (the early ones take longest), yet every
    /// result is merged in item order, with or without a progress pump; on
    /// one thread each result is merged before the next item starts, so
    /// only one is ever held.
    #[test]
    fn folds_in_item_order_whatever_finishes_first() {
        let n = 40;
        for threads in [1, 3, 8] {
            for pump in [false, true] {
                let p = pool(threads);
                let progress = Progress::default();
                let mut on_progress = |_: EStepProgress| {};
                let callback = pump.then_some(&mut on_progress as &mut dyn FnMut(EStepProgress));
                let mut reporter = Reporter::new(n as u64, false, false, "test", callback);
                let held = AtomicUsize::new(0);
                let peak = AtomicUsize::new(0);
                let order = fold_in_order(
                    n,
                    Some(&p),
                    &mut reporter,
                    &progress,
                    |i| {
                        std::thread::sleep(Duration::from_micros(50 * (n - i) as u64));
                        progress.forward.fetch_add(1, Ordering::Relaxed);
                        let now = held.fetch_add(1, Ordering::Relaxed) + 1;
                        peak.fetch_max(now, Ordering::Relaxed);
                        Ok(i)
                    },
                    Vec::new(),
                    |acc: &mut Vec<usize>, i| {
                        held.fetch_sub(1, Ordering::Relaxed);
                        acc.push(i);
                    },
                )
                .unwrap();
                assert_eq!(order, (0..n).collect::<Vec<_>>(), "{threads} {pump}");
                if threads == 1 {
                    assert_eq!(peak.load(Ordering::Relaxed), 1, "{pump}");
                }
            }
        }
    }

    /// The error is the earliest failed item's, as a loop over the items
    /// would return, even when a later item fails first; on one thread no
    /// item starts after the failure.
    #[test]
    fn returns_the_earliest_error() {
        for threads in [1, 4] {
            let p = pool(threads);
            let progress = Progress::default();
            let mut reporter = Reporter::new(0, false, false, "test", None);
            let ran = AtomicUsize::new(0);
            let err = fold_in_order(
                100,
                Some(&p),
                &mut reporter,
                &progress,
                |i| {
                    ran.fetch_add(1, Ordering::Relaxed);
                    if i == 7 {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    match i {
                        7 | 9 => anyhow::bail!("item {i}"),
                        _ => Ok(i),
                    }
                },
                0usize,
                |acc, i| *acc += i,
            )
            .unwrap_err();
            assert_eq!(err.to_string(), "item 7", "{threads}");
            if threads == 1 {
                assert_eq!(ran.load(Ordering::Relaxed), 8);
            }
        }
    }

    /// Nested in other rayon work, as the bootstrap's replicates run, each
    /// fold still merges its own items in order.
    #[test]
    fn nested_folds_are_in_order() {
        let p = pool(4);
        let folds: Vec<Vec<usize>> = p.install(|| {
            (0..6)
                .into_par_iter()
                .map(|k| {
                    let progress = Progress::default();
                    let mut reporter = Reporter::new(0, false, false, "test", None);
                    fold_in_order(
                        20,
                        None,
                        &mut reporter,
                        &progress,
                        |i| Ok(k * 100 + i),
                        Vec::new(),
                        |acc: &mut Vec<usize>, x| acc.push(x),
                    )
                    .unwrap()
                })
                .collect()
        });
        for (k, f) in folds.iter().enumerate() {
            assert_eq!(*f, (0..20).map(|i| k * 100 + i).collect::<Vec<_>>());
        }
    }
}
