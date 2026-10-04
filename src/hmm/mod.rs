//! The PSMC hidden Markov model: E-step (forward-backward with expected
//! sufficient statistics), log-likelihood, and posterior TMRCA decoding.
//!
//! All three run on one per-sequence forward-backward engine in f64 with
//! bounded memory (checkpoints every [`CHECKPOINT_STRIDE`] sites). Sequences
//! are processed in parallel in whichever rayon pool the caller runs in, at
//! any nesting level, and per-sequence results are combined in input order,
//! so results are bit-identical for any thread count, any alpha-cache budget,
//! and whether or not the call is nested inside other rayon work.

use anyhow::{Result, bail};
use ndarray::Array2;
use rayon::ThreadPool;
use std::sync::atomic::AtomicBool;

mod budget;
mod fb;
mod kernel;
mod sched;
mod tmrca;

pub use budget::AlphaCacheBudget;
pub use tmrca::write_tmrca_posterior_tsv;

use fb::{Ctx, Partial, Progress, SeqObs, Workspace};
use kernel::Kernel;
use sched::{Plan, Reporter};

/// Sites per checkpoint segment: the forward pass keeps one forward vector
/// per segment and the backward pass recomputes one segment at a time.
pub const CHECKPOINT_STRIDE: usize = 128;

/// Expected sufficient statistics of one E-step.
#[derive(Debug, Clone)]
pub struct SufficientStats {
    /// `gobs[o][k]`: expected number of sites with observation class `o`
    /// (0 = `T`, 1 = `K`, 2 = `N`) in state `k`.
    pub gobs: [Vec<f64>; 3],
    /// `xi[(k, l)]`: expected number of `k -> l` transitions.
    pub xi: Array2<f64>,
}

/// Result of an E-step: statistics and the log-likelihood of the observations.
#[derive(Debug)]
pub struct EStepResult {
    pub stats: SufficientStats,
    pub loglike: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EStepPhase {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy)]
pub struct EStepProgress {
    pub phase: EStepPhase,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, Clone)]
pub struct TmrcaReportData {
    pub sampled_mean: Vec<[f64; 2]>,
    pub sampled_map: Vec<[f64; 2]>,
    pub sampled_lo: Vec<[f64; 2]>,
    pub sampled_hi: Vec<[f64; 2]>,
    pub state_mass: Vec<f64>,
    pub state_years: Vec<f64>,
    pub total_sites: usize,
    pub seq_count: usize,
}

/// The error returned when a computation is stopped through its cancellation flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Options for [`e_step_with_options`] and [`log_likelihood_with_options`].
pub struct EStepOptions<'a> {
    /// Draw progress bars on stderr.
    pub progress_bars: bool,
    /// Prefix of the progress bars.
    pub label: &'a str,
    /// Called on the calling thread with forward, then backward progress;
    /// `done` increases within each phase.
    pub on_progress: Option<&'a mut dyn FnMut(EStepProgress)>,
    /// Checked between sequences and every [`CHECKPOINT_STRIDE`] sites; once
    /// set, the call stops and returns a [`Cancelled`] error.
    pub cancel: Option<&'a AtomicBool>,
    /// Run the parallel work in this pool instead of the current one.
    pub thread_pool: Option<&'a ThreadPool>,
    /// Budget for caching full forward rows; `None` uses
    /// [`AlphaCacheBudget::global`]. Affects speed and memory only.
    pub alpha_cache: Option<&'a AlphaCacheBudget>,
}

impl Default for EStepOptions<'_> {
    fn default() -> Self {
        Self {
            progress_bars: false,
            label: "E",
            on_progress: None,
            cancel: None,
            thread_pool: None,
            alpha_cache: None,
        }
    }
}

fn validate(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
) -> Result<()> {
    let n_states = pi.len();
    if rows.is_empty() {
        bail!("empty input rows");
    }
    if row_starts.len() != rows.len() {
        bail!("row_starts length mismatch");
    }
    if !row_starts[0] {
        bail!("row_starts[0] must be true");
    }
    if n_states == 0 {
        bail!("model has no states");
    }
    if a.nrows() != n_states || a.ncols() != n_states {
        bail!("transition matrix shape mismatch");
    }
    if em.nrows() < 3 || em.ncols() != n_states {
        bail!("emission matrix shape mismatch");
    }
    if let Some(i) = rows.iter().position(Vec::is_empty) {
        bail!("empty observation row at index {i}");
    }
    Ok(())
}

fn budget_or_global(budget: Option<&AlphaCacheBudget>) -> &AlphaCacheBudget {
    match budget {
        Some(b) => b,
        None => AlphaCacheBudget::global(),
    }
}

/// Return the first error in item order, else the values.
fn collect_in_order<T>(results: Vec<Result<T>>) -> Result<Vec<T>> {
    results.into_iter().collect()
}

/// The E-step with default options; draws progress bars labelled `label`
/// when `progress_enabled`. See [`e_step_with_options`].
pub fn e_step_streaming(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    progress_enabled: bool,
    label: &str,
) -> Result<EStepResult> {
    e_step_streaming_with_progress(pi, a, em, rows, row_starts, progress_enabled, label, None)
}

/// [`e_step_streaming`] with a progress callback. See [`e_step_with_options`].
#[allow(clippy::too_many_arguments)]
pub fn e_step_streaming_with_progress(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    progress_enabled: bool,
    label: &str,
    on_progress: Option<&mut dyn FnMut(EStepProgress)>,
) -> Result<EStepResult> {
    // Shorten the callback's object lifetime to match `label`.
    let on_progress = on_progress.map(|cb| cb as &mut dyn FnMut(EStepProgress));
    e_step_with_options(
        pi,
        a,
        em,
        rows,
        row_starts,
        EStepOptions {
            progress_bars: progress_enabled,
            label,
            on_progress,
            ..EStepOptions::default()
        },
    )
}

/// The E-step: log-likelihood and expected sufficient statistics of the
/// observations under the model (`pi`: initial distribution, `a`: transition
/// matrix, `em`: emission rows for `T`, `K` and `N`).
///
/// `rows[i]` continues the sequence of `rows[i - 1]` unless `row_starts[i]`.
pub fn e_step_with_options(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    opts: EStepOptions<'_>,
) -> Result<EStepResult> {
    validate(pi, a, em, rows, row_starts)?;
    let kernel = Kernel::new(pi, a, em);
    let np = kernel.np();
    let plan = Plan::new(rows, row_starts);
    let progress = Progress::default();
    let ctx = Ctx {
        cancel: opts.cancel,
        progress: &progress,
        budget: budget_or_global(opts.alpha_cache),
    };
    let mut reporter = Reporter::new(
        plan.total_sites,
        true,
        opts.progress_bars,
        opts.label,
        opts.on_progress,
    );
    let partials = sched::run(
        plan.items.len(),
        opts.thread_pool,
        &mut reporter,
        &progress,
        |item| {
            let mut ws = Workspace::new(np);
            let mut acc = Partial::new(np);
            for seq in &plan.seqs[plan.items[item].clone()] {
                ctx.check_cancel()?;
                fb::e_step(
                    &kernel,
                    &SeqObs::new(&rows[seq.clone()]),
                    &mut ws,
                    &mut acc,
                    &ctx,
                )?;
            }
            Ok(acc)
        },
    );
    let partials = collect_in_order(partials)?;
    sched::finish(&mut reporter, &progress);

    let mut total = Partial::new(np);
    for p in &partials {
        total.merge(p);
    }
    let n = kernel.n_states();
    let [g0, g1, g2] = total.gobs;
    Ok(EStepResult {
        stats: SufficientStats {
            gobs: [g0[..n].to_vec(), g1[..n].to_vec(), g2[..n].to_vec()],
            xi: kernel.transition_counts(&total.m),
        },
        loglike: total.loglike,
    })
}

/// Log-likelihood of the observations under the model (forward pass only).
///
/// Bit-identical to the `loglike` of [`e_step_with_options`] on the same input.
pub fn log_likelihood(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
) -> Result<f64> {
    log_likelihood_with_options(pi, a, em, rows, row_starts, EStepOptions::default())
}

/// [`log_likelihood`] with progress (forward phase only), cancellation and
/// thread-pool options.
pub fn log_likelihood_with_options(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    opts: EStepOptions<'_>,
) -> Result<f64> {
    validate(pi, a, em, rows, row_starts)?;
    let kernel = Kernel::new(pi, a, em);
    let np = kernel.np();
    let plan = Plan::new(rows, row_starts);
    let progress = Progress::default();
    let ctx = Ctx {
        cancel: opts.cancel,
        progress: &progress,
        budget: budget_or_global(opts.alpha_cache),
    };
    let mut reporter = Reporter::new(
        plan.total_sites,
        false,
        opts.progress_bars,
        opts.label,
        opts.on_progress,
    );
    let partials = sched::run(
        plan.items.len(),
        opts.thread_pool,
        &mut reporter,
        &progress,
        |item| {
            let mut ws = Workspace::new(np);
            let mut loglike = 0.0;
            for seq in &plan.seqs[plan.items[item].clone()] {
                ctx.check_cancel()?;
                loglike +=
                    fb::log_likelihood(&kernel, &SeqObs::new(&rows[seq.clone()]), &mut ws, &ctx)?;
            }
            Ok(loglike)
        },
    );
    let partials = collect_in_order(partials)?;
    sched::finish(&mut reporter, &progress);
    let mut total = 0.0;
    for p in partials {
        total += p;
    }
    Ok(total)
}
