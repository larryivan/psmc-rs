//! The EM driver: alternate a full forward-backward E-step with an L-BFGS M-step.

use anyhow::Result;
use rayon::ThreadPool;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::hmm::{
    Cancelled, EStepOptions, EStepPhase, EStepProgress, e_step_with_options,
    log_likelihood_with_options,
};
use crate::model::PsmcModel;
use crate::opt::{
    MStepConfig, MStepProgress, bounds_from_config, m_step_lbfgs, m_step_lbfgs_with_progress,
    pack_params, unpack_params,
};
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
}

impl EmHistory {
    /// Log-likelihood of the final parameters (the model EM returns), or
    /// `None` after zero iterations. This is the second value of the last
    /// iteration; the first is that of the parameters entering it.
    pub fn final_loglike(&self) -> Option<f64> {
        self.loglike.last().map(|&(_, after)| after)
    }
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
}

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

pub fn em_train_with_progress(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
    on_progress: Option<&mut dyn FnMut(EmProgressEvent)>,
) -> Result<EmHistory> {
    let opts = EmOptions {
        on_progress,
        ..EmOptions::default()
    };
    em_train_with_options(model, rows, row_starts, n_iter, config, opts)
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
    };
    let mut params = pack_params(model);
    history.params.push(params.clone());

    let bounds = bounds_from_config(model, config);
    let pb_em = if config.progress && n_iter > 0 {
        Some(progress::bar(n_iter as u64, "EM", "overall"))
    } else {
        None
    };

    if n_iter > 0 {
        model.param_recalculate()?;
    }
    for iter in 0..n_iter {
        check_cancel(iter + 1)?;
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EmProgressEvent::IterStart {
                iter: iter + 1,
                total_iters: n_iter,
            });
        }
        if let Some(pb) = &pb_em {
            pb.set_message(format!("overall {}/{}", iter + 1, n_iter));
        }
        let e = {
            let mut map_es = on_progress.as_mut().map(|cb_outer| {
                move |ev: EStepProgress| {
                    (*cb_outer)(EmProgressEvent::EStep {
                        iter: iter + 1,
                        total_iters: n_iter,
                        phase: ev.phase,
                        done: ev.done,
                        total: ev.total,
                    })
                }
            });
            let e_opts = EStepOptions {
                progress_bars: config.progress,
                label: "E",
                on_progress: map_es
                    .as_mut()
                    .map(|cb| cb as &mut dyn FnMut(EStepProgress)),
                cancel,
                thread_pool,
                alpha_cache: None,
            };
            e_step_with_options(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                rows,
                row_starts,
                e_opts,
            )
            .map_err(|err| cancelled_in(iter + 1, err))?
        };
        // This E-step evaluates the parameters the previous iteration left.
        if let Some(prev) = history.loglike.last_mut() {
            prev.1 = e.loglike;
        }
        history.loglike.push((e.loglike, f64::NAN));
        let stats = e.stats;
        check_cancel(iter + 1)?;
        let params_opt = if let Some(cb_outer) = on_progress.as_mut() {
            let mut map_ms = |ev: MStepProgress| {
                (*cb_outer)(EmProgressEvent::MStep {
                    iter: iter + 1,
                    total_iters: n_iter,
                    done: ev.done,
                    total: ev.total,
                });
            };
            m_step_lbfgs_with_progress(model, &stats, &params, &bounds, config, Some(&mut map_ms))?
        } else {
            m_step_lbfgs(model, &stats, &params, &bounds, config)?
        };
        params = params_opt.clone();
        unpack_params(model, &params)?;
        // Keep the model's matrices in step with its parameters, also when
        // a later cancellation ends training early.
        model.param_recalculate()?;
        history.params.push(params.clone());

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
