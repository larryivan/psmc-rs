//! The PSMC hidden Markov model: E-step (forward-backward with expected
//! sufficient statistics), log-likelihood, and posterior TMRCA decoding.
//!
//! All three run on one per-sequence forward-backward engine in f64 with
//! bounded memory (checkpoints every [`CHECKPOINT_STRIDE`] sites). Sequences
//! are processed in parallel in whichever rayon pool the caller runs in, at
//! any nesting level, and per-sequence results are combined in input order,
//! so results are bit-identical for any thread count, any alpha-cache budget,
//! and whether or not the call is nested inside other rayon work.
//!
//! # Callable bases
//!
//! Without callable bases every called bin emits as C psmc's bins do, with
//! the model's emission rows for `T`, `K` and `N`. Given the callable bases
//! of each bin ([`EStepOptions::callable`], [`TmrcaOptions::callable`]), a
//! bin of `c` callable bases out of `s` emits by the callable-length model
//! of `docs/design/bin-callable-model.md`:
//!
//! ```text
//! P(T | k, c) = e_T(k)^(c / s),  P(K | k, c) = 1 - P(T | k, c),  P(N | k, c) = 1
//! ```
//!
//! (`e_T(k) = exp(-θ avg_t[k])` for PSMC, so `P(T | k, c) = exp(-θ (c / s)
//! avg_t[k])`). Every pass builds one emission row per pair of symbol and
//! count that occurs in the input, so the per-bin loops stay lookups. Full
//! bins (`c = s`) use the model's own rows, so input whose bins are all full
//! gives exactly, to the bit, the results of the same input without counts.
//! The E-step then also returns the sums the M-step of that model needs
//! ([`SufficientStats::callable`]).

use anyhow::{Result, bail};
use ndarray::Array2;
use rayon::ThreadPool;
use std::sync::atomic::AtomicBool;

use crate::io::CallableBases;

mod budget;
mod fb;
mod kernel;
mod obs;
mod sched;
mod simd;
mod tmrca;

pub use budget::AlphaCacheBudget;
pub use simd::{SimdPath, set_force_portable, simd_path};
pub use tmrca::{TmrcaOptions, write_tmrca_posterior_tsv, write_tmrca_posterior_tsv_with_options};

use fb::{Ctx, Partial, Progress, SeqObs, Workspace};
use kernel::Kernel;
use obs::ObsTable;
use sched::{Plan, Reporter};

/// Sites per checkpoint segment: the forward pass keeps one forward vector
/// per segment and the backward pass recomputes one segment at a time.
pub const CHECKPOINT_STRIDE: usize = 128;

/// Expected sufficient statistics of one E-step.
#[derive(Debug, Clone)]
pub struct SufficientStats {
    /// `gobs[o][k]`: expected number of sites with observation class `o`
    /// (0 = `T`, 1 = `K`, 2 = `N`) in state `k`, whatever their callable
    /// bases.
    pub gobs: [Vec<f64>; 3],
    /// `xi[(k, l)]`: expected number of `k -> l` transitions.
    pub xi: Array2<f64>,
    /// With callable bases, the expected counts of the callable-length
    /// emission model, which the M-step uses in place of `gobs[0]` and
    /// `gobs[1]`; `None` without counts (C psmc's model).
    pub callable: Option<CallableStats>,
}

/// The expected emission counts of an E-step on bins with callable bases
/// (`docs/design/bin-callable-model.md`). The emission part of the expected
/// complete-data log-likelihood is
///
/// ```text
/// sum_k [ G_T(k) ln e_T(k) + sum_c G_K(c, k) ln(1 - e_T(k)^(c / s)) ]
/// ```
///
/// since the log-probability of a `T` bin, `(c / s) ln e_T(k)`, is linear in
/// its count. With every count `s`, `G_T` is `gobs[0]` and the one `G_K` is
/// `gobs[1]`, and the expression is C psmc's.
#[derive(Debug, Clone, PartialEq)]
pub struct CallableStats {
    /// `s`, the bases of a full bin.
    pub bin_size: usize,
    /// `G_T(k)`: the posterior of state `k` summed over the `T` bins, each
    /// weighted by its share of callable bases `c / s`.
    pub t_weighted: Vec<f64>,
    /// `(c, G_K(c, k))` for each count `c` of the `K` bins, ascending: the
    /// posterior of state `k` summed over the `K` bins with `c` callable
    /// bases.
    pub k_by_count: Vec<(u16, Vec<f64>)>,
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
    /// The callable bases of each bin, row for row
    /// ([`crate::io::Observations::callable`]): bins emit by the
    /// callable-length model (see the [module documentation](self)), and
    /// the E-step returns [`SufficientStats::callable`]. `None` (the
    /// default) is C psmc's model, every called bin full.
    pub callable: Option<&'a CallableBases>,
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
            callable: None,
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

/// One pass's model and observations, laid out for the per-sequence passes.
struct Pass<'a> {
    rows: &'a [Vec<u8>],
    callable: Option<&'a CallableBases>,
    table: ObsTable,
    kernel: Kernel,
    plan: Plan,
}

impl<'a> Pass<'a> {
    /// Checks the shapes and, with callable bases, that they fit the rows
    /// ([`ObsTable::counted`], run in `pool` when given).
    fn new(
        pi: &[f64],
        a: &Array2<f64>,
        em: &Array2<f64>,
        rows: &'a [Vec<u8>],
        row_starts: &[bool],
        callable: Option<&'a CallableBases>,
        pool: Option<&ThreadPool>,
    ) -> Result<Self> {
        validate(pi, a, em, rows, row_starts)?;
        let table = match (callable, pool) {
            (None, _) => ObsTable::plain(),
            (Some(c), None) => ObsTable::counted(rows, c)?,
            (Some(c), Some(p)) => p.install(|| ObsTable::counted(rows, c))?,
        };
        Ok(Self {
            rows,
            callable,
            kernel: Kernel::new(pi, a, em, &table),
            table,
            plan: Plan::new(rows, row_starts),
        })
    }

    /// The observations of sequence `i`.
    fn seq(&self, i: usize) -> SeqObs<'_> {
        let rows = self.plan.seqs[i].clone();
        let counts = self.callable.map(|c| &c.rows()[rows.clone()]);
        SeqObs::new(&self.rows[rows], counts, &self.table)
    }

    /// Runs `per_seq` over every sequence: serially within a work item, into
    /// an accumulator from `zero`, and the items in parallel, merged into
    /// `zero()` with `merge` in item order ([`sched::fold_in_order`]), so
    /// the result does not depend on the thread count. Reports the forward
    /// progress, and the backward progress when `with_backward`.
    fn fold<A: Send>(
        &self,
        opts: EStepOptions<'_>,
        with_backward: bool,
        zero: impl Fn() -> A + Sync,
        per_seq: impl Fn(&SeqObs<'_>, &mut Workspace, &mut A, &Ctx<'_>) -> Result<()> + Sync,
        merge: impl Fn(&mut A, A) + Sync,
    ) -> Result<A> {
        let progress = Progress::default();
        let budget = opts
            .alpha_cache
            .unwrap_or_else(|| AlphaCacheBudget::global());
        let ctx = Ctx {
            cancel: opts.cancel,
            progress: &progress,
            budget,
        };
        let mut reporter = Reporter::new(
            self.plan.total_sites,
            with_backward,
            opts.progress_bars,
            opts.label,
            opts.on_progress,
        );
        let total = sched::fold_in_order(
            self.plan.items.len(),
            opts.thread_pool,
            &mut reporter,
            &progress,
            |item| {
                let mut ws = Workspace::new(self.kernel.np());
                let mut acc = zero();
                for seq in self.plan.items[item].clone() {
                    ctx.check_cancel()?;
                    per_seq(&self.seq(seq), &mut ws, &mut acc, &ctx)?;
                }
                Ok(acc)
            },
            zero(),
            merge,
        )?;
        reporter.finish(&progress);
        Ok(total)
    }
}

/// The E-step with default options; draws progress bars labelled `label`
/// when `progress_enabled`. See [`e_step_with_options`], which also takes
/// callable bases.
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
/// matrix, `em`: emission rows for `T`, `K` and `N`), with the callable
/// bases of [`EStepOptions::callable`] when given.
///
/// `rows[i]` continues the sequence of `rows[i - 1]` unless `row_starts[i]`.
/// Errors when the shapes do not fit, and when the callable bases do not fit
/// the rows ([`crate::io::Observations::validate`]).
pub fn e_step_with_options(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    opts: EStepOptions<'_>,
) -> Result<EStepResult> {
    let pass = Pass::new(pi, a, em, rows, row_starts, opts.callable, opts.thread_pool)?;
    let (kernel, table) = (&pass.kernel, &pass.table);
    let np = kernel.np();
    let total = pass.fold(
        opts,
        true,
        || Partial::new(table.len(), np),
        |seq, ws, acc, ctx| fb::e_step(kernel, seq, ws, acc, ctx),
        |total, p| total.merge(&p),
    )?;
    let (gobs, callable) = table.reduce(&total.g, kernel.n_states(), np);
    Ok(EStepResult {
        stats: SufficientStats {
            gobs,
            xi: kernel.transition_counts(&total.m),
            callable,
        },
        loglike: total.loglike,
    })
}

/// Log-likelihood of the observations under the model (forward pass only),
/// without callable bases (see [`log_likelihood_with_options`]).
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

/// [`log_likelihood`] with progress (forward phase only), cancellation,
/// thread-pool options and callable bases ([`EStepOptions::callable`]).
///
/// Bit-identical to the `loglike` of [`e_step_with_options`] with the same
/// input and counts.
pub fn log_likelihood_with_options(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    opts: EStepOptions<'_>,
) -> Result<f64> {
    let pass = Pass::new(pi, a, em, rows, row_starts, opts.callable, opts.thread_pool)?;
    pass.fold(
        opts,
        false,
        || 0.0,
        |seq, ws, loglike, ctx| {
            *loglike += fb::log_likelihood(&pass.kernel, seq, ws, ctx)?;
            Ok(())
        },
        |total: &mut f64, p| *total += p,
    )
}
