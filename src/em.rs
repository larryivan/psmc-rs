//! The EM driver: alternate a full forward-backward E-step with an L-BFGS M-step.

use anyhow::Result;

use crate::hmm::{
    EStepPhase, EStepProgress, e_step_streaming, e_step_streaming_with_progress, log_likelihood,
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

pub fn em_train(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
) -> Result<EmHistory> {
    em_train_with_progress(model, rows, row_starts, n_iter, config, None)
}

pub fn em_train_with_progress(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
    mut on_progress: Option<&mut dyn FnMut(EmProgressEvent)>,
) -> Result<EmHistory> {
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

    for iter in 0..n_iter {
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EmProgressEvent::IterStart {
                iter: iter + 1,
                total_iters: n_iter,
            });
        }
        if let Some(pb) = &pb_em {
            pb.set_message(format!("overall {}/{}", iter + 1, n_iter));
        }
        model.param_recalculate()?;
        let e = if let Some(cb_outer) = on_progress.as_mut() {
            let mut map_es = |ev: EStepProgress| {
                (*cb_outer)(EmProgressEvent::EStep {
                    iter: iter + 1,
                    total_iters: n_iter,
                    phase: ev.phase,
                    done: ev.done,
                    total: ev.total,
                });
            };
            e_step_streaming_with_progress(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                rows,
                row_starts,
                config.progress,
                "E",
                Some(&mut map_es),
            )?
        } else {
            e_step_streaming(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                rows,
                row_starts,
                config.progress,
                "E",
            )?
        };
        // This E-step evaluates the parameters the previous iteration left.
        if let Some(prev) = history.loglike.last_mut() {
            prev.1 = e.loglike;
        }
        history.loglike.push((e.loglike, f64::NAN));
        let stats = e.stats;
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
        history.params.push(params.clone());

        if let Some(pb) = &pb_em {
            pb.inc(1);
        }
    }
    if let Some(last) = history.loglike.last_mut() {
        // Only the final parameters have not been evaluated yet: a forward
        // pass suffices.
        model.param_recalculate()?;
        last.1 = log_likelihood(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            rows,
            row_starts,
        )?;
    }
    if let Some(pb) = pb_em {
        pb.finish_with_message("EM done");
    }

    Ok(history)
}
