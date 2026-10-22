//! The EM driver: alternate a full forward-backward E-step with an L-BFGS M-step.

use anyhow::{Context, Result};
use rayon::ThreadPool;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::hmm::{
    Cancelled, EStepOptions, EStepPhase, EStepProgress, SufficientStats, e_step_with_options,
    log_likelihood_with_options,
};
use crate::io::CallableBases;
use crate::model::PsmcModel;
use crate::opt::{MStepConfig, MStepProgress, MStepStop, m_step, pack_params, unpack_params};
use crate::progress;

#[derive(Debug)]
pub struct EmHistory {
    /// Per iteration: (log-likelihood of the parameters entering it, of the
    /// parameters leaving it). The second value of iteration `i` equals the
    /// first of iteration `i + 1`; the last one is the log-likelihood of the
    /// final parameters.
    pub loglike: Vec<(f64, f64)>,
    /// Parameters before the first iteration and after each one.
    pub params: Vec<Vec<f64>>,
    /// Per iteration: the posterior state distribution over the non-missing
    /// bins of its E-step ([`posterior_sigma`]), C psmc's `post_sigma`.
    pub post_sigma: Vec<Vec<f64>>,
    /// Per iteration: evaluations of the M-step objective (C's `IT` line).
    pub mstep_evaluations: Vec<usize>,
    /// Per iteration: L-BFGS iterations of the M-step
    /// ([`MStepOutcome::iterations`](crate::opt::MStepOutcome::iterations)).
    pub mstep_iterations: Vec<usize>,
    /// Per iteration: why the M-step stopped.
    pub mstep_stops: Vec<MStepStop>,
}

impl EmHistory {
    /// Log-likelihood of the final parameters (the model EM returns), or
    /// `None` after zero iterations. This is the second value of the last
    /// iteration; the first is that of the parameters entering it.
    pub fn final_loglike(&self) -> Option<f64> {
        self.loglike.last().map(|&(_, after)| after)
    }
}

/// C psmc's `post_sigma`: the posterior distribution of the hidden state over
/// the non-missing bins of an E-step, `(E_0[k] + E_1[k]) / sum_l (E_0[l] + E_1[l])`.
///
/// The expected emission counts here cover every bin, while C's `hmm_expect`
/// leaves out the last bin of each sequence. That is the only difference in
/// the expectations behind C's `post_sigma`: removing those bins' posteriors
/// from `stats.gobs` reproduces C's first-round `post_sigma` values to all 6
/// printed decimals, on the 3 sequences of `tests/data/c_reference` and on
/// 1,010 sequences of 1 to 80 bins (`tests/em_c_reference.rs`).
///
/// The size of the difference follows the share of the data in those bins.
/// With `D` sequences ending in a non-missing bin and `sum_L` non-missing
/// bins, each value differs from C's by at most `D / sum_L`, and the `RI`
/// of the round by a relative amount of the order of `D / sum_L` (0.5 to
/// 0.75 times it, measured). For a few chromosome-length sequences this is
/// negligible: 3 sequences of 150,947 bins in all (`D / sum_L` = 2e-5) move
/// `post_sigma` by at most 2.4e-6. For a fragmented assembly it is not: on
/// the 1,010 short sequences (`D / sum_L` = 2.5%) `RI` differs from C's by
/// 1.2%, so `psmc_plot.pl -n 0`, which plots the round with the smallest
/// `RI` from round `-m` on, can pick a different round than for C's output.
pub fn posterior_sigma(stats: &SufficientStats) -> Vec<f64> {
    let [e0, e1, _] = &stats.gobs;
    let counts: Vec<f64> = e0.iter().zip(e1).map(|(c0, c1)| c0 + c1).collect();
    let total: f64 = counts.iter().sum();
    counts.into_iter().map(|c| c / total).collect()
}

#[derive(Debug, Clone, Copy)]
pub enum EmProgressEvent {
    IterStart {
        iter: usize,
        total_iters: usize,
    },
    EStep {
        iter: usize,
        total_iters: usize,
        phase: EStepPhase,
        done: u64,
        total: u64,
    },
    MStep {
        iter: usize,
        total_iters: usize,
        done: usize,
        total: usize,
    },
}

/// Options for [`em_train_with_options`].
#[derive(Default)]
pub struct EmOptions<'a> {
    /// Called on the calling thread at the start of each iteration and with
    /// E-step and M-step progress.
    pub on_progress: Option<&'a mut dyn FnMut(EmProgressEvent)>,
    /// Checked before each iteration, inside the E-step (between sequences
    /// and every [`CHECKPOINT_STRIDE`](crate::hmm::CHECKPOINT_STRIDE) sites),
    /// between the E- and M-steps, and before the final log-likelihood; once
    /// set, training stops with a [`Cancelled`] error.
    pub cancel: Option<&'a AtomicBool>,
    /// Run the E-steps and the final log-likelihood in this pool instead of
    /// the current one.
    pub thread_pool: Option<&'a ThreadPool>,
    /// The callable bases of each bin, row for row
    /// ([`crate::io::Observations::callable`]): fit the callable-length
    /// model of `docs/design/bin-callable-model.md`, in which a bin of `c`
    /// of `s` callable bases has the `T` emission of a full bin to the power
    /// `c / s` ([`EStepOptions::callable`]). `None` (the default) fits C
    /// psmc's model; so do counts whose called bins are all full, to the
    /// bit.
    pub callable: Option<&'a CallableBases>,
}

/// [`em_train_with_options`] with the default options.
pub fn em_train(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
) -> Result<EmHistory> {
    em_train_with_options(
        model,
        rows,
        row_starts,
        n_iter,
        config,
        EmOptions::default(),
    )
}

fn cancelled_in(iter: usize, err: anyhow::Error) -> anyhow::Error {
    if err.is::<Cancelled>() {
        err.context(format!("EM cancelled in iteration {iter}"))
    } else {
        err
    }
}

/// Fit `model` by `n_iter` EM iterations (E-step, then L-BFGS M-step).
///
/// Returns the log-likelihood and parameter history (see [`EmHistory`]).
/// On cancellation the model holds the parameters of the last completed
/// iteration (with its matrices recalculated) and the error is
/// [`Cancelled`] (with the iteration as context).
///
/// Errors before the first E-step when the starting parameters give
/// negative transition probabilities ([`PsmcModel::check_transitions`]),
/// from which EM cannot move.
pub fn em_train_with_options(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
    opts: EmOptions<'_>,
) -> Result<EmHistory> {
    let EmOptions {
        mut on_progress,
        cancel,
        thread_pool,
        callable,
    } = opts;
    let check_cancel = |iter: usize| -> Result<()> {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(cancelled_in(iter, Cancelled.into()));
        }
        Ok(())
    };
    let mut history = EmHistory {
        loglike: Vec::with_capacity(n_iter),
        params: Vec::with_capacity(n_iter + 1),
        post_sigma: Vec::with_capacity(n_iter),
        mstep_evaluations: Vec::with_capacity(n_iter),
        mstep_iterations: Vec::with_capacity(n_iter),
        mstep_stops: Vec::with_capacity(n_iter),
    };
    history.params.push(pack_params(model));

    let pb_em =
        (config.progress && n_iter > 0).then(|| progress::bar(n_iter as u64, "EM", "overall"));

    if n_iter > 0 {
        model.param_recalculate()?;
        // From negative transition probabilities every M-step would return
        // its start: fail before the first E-step instead of returning that
        // start as a fit.
        model
            .check_transitions()
            .context("EM cannot start from these parameters")?;
    }
    for iter in 1..=n_iter {
        check_cancel(iter)?;
        if let Some(cb) = on_progress.as_mut() {
            cb(EmProgressEvent::IterStart {
                iter,
                total_iters: n_iter,
            });
        }
        if let Some(pb) = &pb_em {
            pb.set_message(format!("overall {iter}/{n_iter}"));
        }
        let mut on_e_step = on_progress.as_mut().map(|cb| {
            move |ev: EStepProgress| {
                cb(EmProgressEvent::EStep {
                    iter,
                    total_iters: n_iter,
                    phase: ev.phase,
                    done: ev.done,
                    total: ev.total,
                })
            }
        });
        let e_opts = EStepOptions {
            progress_bars: config.progress,
            on_progress: on_e_step
                .as_mut()
                .map(|cb| cb as &mut dyn FnMut(EStepProgress)),
            cancel,
            thread_pool,
            callable,
            ..EStepOptions::default()
        };
        let e = e_step_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            rows,
            row_starts,
            e_opts,
        )
        .map_err(|err| cancelled_in(iter, err))?;
        // This E-step evaluates the parameters the previous iteration left.
        if let Some(prev) = history.loglike.last_mut() {
            prev.1 = e.loglike;
        }
        history.loglike.push((e.loglike, f64::NAN));
        let stats = e.stats;
        check_cancel(iter)?;
        let mut on_m_step = on_progress.as_mut().map(|cb| {
            move |ev: MStepProgress| {
                cb(EmProgressEvent::MStep {
                    iter,
                    total_iters: n_iter,
                    done: ev.done,
                    total: ev.total,
                })
            }
        });
        let outcome = m_step(
            model,
            &stats,
            config,
            on_m_step
                .as_mut()
                .map(|cb| cb as &mut dyn FnMut(MStepProgress)),
        )?;
        unpack_params(model, &outcome.params)?;
        // Keep the model's matrices in step with its parameters, also when
        // a later cancellation ends training early.
        model.param_recalculate()?;
        history.params.push(outcome.params);
        history.post_sigma.push(posterior_sigma(&stats));
        history.mstep_evaluations.push(outcome.evaluations);
        history.mstep_iterations.push(outcome.iterations);
        history.mstep_stops.push(outcome.stop);

        if let Some(pb) = &pb_em {
            pb.inc(1);
        }
    }
    if let Some(last) = history.loglike.last_mut() {
        // Only the final parameters have not been evaluated yet: a forward
        // pass suffices.
        check_cancel(n_iter)?;
        let ll_opts = EStepOptions {
            cancel,
            thread_pool,
            callable,
            ..EStepOptions::default()
        };
        last.1 = log_likelihood_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            rows,
            row_starts,
            ll_opts,
        )
        .map_err(|err| cancelled_in(n_iter, err))?;
    }
    if let Some(pb) = pb_em {
        pb.finish_with_message("EM done");
    }

    Ok(history)
}
