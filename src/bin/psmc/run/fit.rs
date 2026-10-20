//! The HMM work of a run: EM, the log-likelihood of the model it writes,
//! the fitted parameters on the M-step's bounds, and the TMRCA decoding.

use anyhow::Result;
use std::path::Path;

use psmc_rs::PsmcModel;
use psmc_rs::em::{EmHistory, EmOptions, EmProgressEvent, em_train_with_options};
use psmc_rs::hmm::{
    EStepOptions, EStepPhase, EStepProgress, TmrcaOptions, TmrcaReportData,
    log_likelihood_with_options, write_tmrca_posterior_tsv_with_options,
};
use psmc_rs::io::Observations;
use psmc_rs::opt::{MStepConfig, bounds_from_config};

use super::{Context, Event, Stage};
use crate::cli::InferArgs;

/// `n_iter` EM iterations of `model` on `obs`, reporting progress.
pub(super) fn fit(
    model: &mut PsmcModel,
    obs: &Observations,
    n_iter: usize,
    mstep_config: &MStepConfig,
    ctx: Context,
) -> Result<EmHistory> {
    let mut on_progress = |ev: EmProgressEvent| {
        (ctx.sink)(match ev {
            EmProgressEvent::IterStart { iter, total_iters } => Event::Iteration {
                iter,
                total: total_iters,
            },
            EmProgressEvent::EStep {
                iter,
                total_iters,
                phase,
                done,
                total,
            } => Event::Progress {
                stage: match phase {
                    EStepPhase::Forward => Stage::EForward,
                    EStepPhase::Backward => Stage::EBackward,
                },
                done,
                total,
                iter: Some((iter, total_iters)),
            },
            EmProgressEvent::MStep {
                iter,
                total_iters,
                done,
                total,
            } => Event::Progress {
                stage: Stage::MStep,
                done: done as u64,
                total: total as u64,
                iter: Some((iter, total_iters)),
            },
        })
    };
    let history = em_train_with_options(
        model,
        &obs.rows,
        &obs.row_starts,
        n_iter,
        mstep_config,
        EmOptions {
            on_progress: Some(&mut on_progress),
            cancel: Some(ctx.cancel),
            thread_pool: Some(ctx.pool),
            callable: obs.callable.as_ref(),
        },
    )?;
    model.param_recalculate()?;
    Ok(history)
}

/// Log-likelihood of `model` on `obs` (with its callable bases when the fit
/// uses them), by the forward pass with which EM evaluates the parameters
/// it ends with.
pub(super) fn model_loglike(model: &PsmcModel, obs: &Observations, ctx: Context) -> Result<f64> {
    let mut on_progress = |p: EStepProgress| {
        (ctx.sink)(Event::Progress {
            stage: Stage::EForward,
            done: p.done,
            total: p.total,
            iter: None,
        })
    };
    let opts = EStepOptions {
        on_progress: Some(&mut on_progress),
        cancel: Some(ctx.cancel),
        thread_pool: Some(ctx.pool),
        callable: obs.callable.as_ref(),
        ..EStepOptions::default()
    };
    log_likelihood_with_options(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        opts,
    )
}

/// Write the posterior TMRCA of every bin of `obs` under the fitted `model`
/// to `path`, decoded with the emissions the fit used (of each bin's
/// count), and return what the report shows of it.
pub(super) fn decode_tmrca(
    model: &PsmcModel,
    obs: &Observations,
    path: &Path,
    cli: &InferArgs,
    progress_bars: bool,
    ctx: Context,
) -> Result<TmrcaReportData> {
    // Each state's average coalescent time, the time of its emission
    // (C psmc's TC column).
    let tmrca_years = model.state_tmrca_years(cli.input.bin_size as f64, cli.gen_years);
    let tmrca = write_tmrca_posterior_tsv_with_options(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        &tmrca_years,
        path,
        TmrcaOptions {
            progress_bars,
            max_points: 20_000,
            callable: obs.callable.as_ref(),
        },
    )?;
    ctx.output(format!("TMRCA posterior: {}", path.display()));
    Ok(tmrca)
}

/// Whether `v` is on the lower or the upper M-step bound of its range: where
/// the M-step's logit coordinate saturates, not merely close in a wide range.
///
/// Above, within 1e-6 of the range of `hi`: the logit's slope there is below
/// 1e-6 of its largest, and a λ the data push to the bound stops there
/// (9,999.99991 for a bound of 1e4 on a simulated bottleneck genome). Below,
/// within twice the M-step's clamp (1e-12 of the range) of `lo`, or 1e-6 of
/// `lo` itself: towards `lo` the logit is a logarithm, so λ = 0.01 against a
/// bound of 1e-6 is an ordinary value.
fn near(v: f64, lo: f64, hi: f64) -> (bool, bool) {
    let span = (hi - lo).abs();
    let below = (2e-12 * span).max(1e-6 * lo.abs()).max(f64::EPSILON * 16.0);
    let above = (1e-6 * span).max(f64::EPSILON * 16.0);
    (v - lo <= below, hi - v <= above)
}

/// A warning when fitted parameters sit on the M-step's bounds, saying
/// what that means for each kind (the CLI has no option to move them).
pub(super) fn near_bounds_warning(model: &PsmcModel, cfg: &MStepConfig) -> Option<String> {
    // The parameters in the order of their bounds: theta, rho, t_max, then
    // the lambdas.
    let names = ["theta", "rho", "t_max"].map(String::from).into_iter();
    let names = names.chain((0..model.lam.len()).map(|i| format!("lam[{i}]")));
    let values = [model.theta, model.rho, model.t_max].into_iter();
    let values = values.chain(model.lam.iter().copied());
    let (mut lam_hi, mut lam_lo, mut others) = (Vec::new(), Vec::new(), Vec::new());
    for (i, ((name, v), b)) in names
        .zip(values)
        .zip(&bounds_from_config(model, cfg))
        .enumerate()
    {
        let (is_lo, is_hi) = near(v, b.lo, b.hi);
        if !(is_lo || is_hi) {
            continue;
        }
        let bound = if is_lo { b.lo } else { b.hi };
        match (i >= 3, is_hi) {
            (true, true) => lam_hi.push(name),
            (true, false) => lam_lo.push(name),
            (false, _) => others.push(format!("{name} = {bound:e}")),
        }
    }
    let list = |names: &[String]| {
        if names.len() > 6 {
            format!("{} and {} more", names[..5].join(", "), names.len() - 5)
        } else {
            names.join(", ")
        }
    };
    let mut parts = Vec::new();
    if !lam_hi.is_empty() {
        // With one λ per interval, grouping the intervals is the remedy.
        let grouping = if model.pattern_spec().is_none() {
            "; a pattern that groups them, as C psmc's `4+...` patterns do, fits one λ for several"
        } else {
            ""
        };
        parts.push(format!(
            "{} at the M-step's upper bound of λ ({}): the data hold almost no coalescences in \
             their intervals, so the Ne there is that bound, not an estimate (usual for the \
             most recent intervals{grouping})",
            list(&lam_hi),
            cfg.lam_hi
        ));
    }
    if !lam_lo.is_empty() {
        parts.push(format!(
            "{} at the M-step's lower bound of λ ({}): the fit makes every lineage that reaches \
             those intervals coalesce at once (seen for the oldest interval, beyond t_max)",
            list(&lam_lo),
            cfg.lam_lo
        ));
    }
    if !others.is_empty() {
        parts.push(format!(
            "{} at the M-step's bound: check the input and the starting values",
            others.join(", ")
        ));
    }
    (!parts.is_empty()).then(|| format!("fitted parameters on their bounds: {}.", parts.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_on_their_bounds_are_reported() {
        let cfg = MStepConfig::default();
        let mut model =
            PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, Some("4+5*3+4".into())).unwrap();
        assert_eq!(near_bounds_warning(&model, &cfg), None);
        model.lam[2] = cfg.lam_hi;
        model.rho = cfg.rho_lo;
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        // Each kind with what it means, and no advice the CLI cannot follow.
        assert!(
            msg.contains("lam[2] at the M-step's upper bound of λ (10000)")
                && msg.contains("not an estimate"),
            "{msg}"
        );
        assert!(msg.contains("rho = 1e-12 at the M-step's bound"), "{msg}");
        // The pattern already groups intervals: no advice to group them.
        assert!(!msg.contains("groups them"), "{msg}");
        let mut one_each = PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, None).unwrap();
        one_each.lam[0] = cfg.lam_hi;
        let msg = near_bounds_warning(&one_each, &cfg).unwrap();
        assert!(
            msg.contains("lam[0] at the M-step's upper bound") && msg.contains("groups them"),
            "{msg}"
        );
        assert!(!msg.contains("wider bounds"), "{msg}");
        // Where an M-step leaves a λ the data push to the bound, saturated
        // in its logit coordinate but not equal to the bound.
        model.lam[2] = 9_999.999_906;
        model.rho = 0.0025;
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        assert!(msg.contains("lam[2] at the M-step's upper bound"), "{msg}");
        // Ordinary values of a wide range are not on a bound.
        for lam in [0.5 * cfg.lam_hi, 9_990.0, 0.01, 5e-6] {
            model.lam[2] = lam;
            assert_eq!(near_bounds_warning(&model, &cfg), None, "{lam}");
        }
        // The M-step's clamp keeps a λ 1e-12 of its range above lo.
        model.lam[2] = cfg.lam_lo + 1e-12 * (cfg.lam_hi - cfg.lam_lo);
        let msg = near_bounds_warning(&model, &cfg).unwrap();
        assert!(
            msg.contains("lam[2] at the M-step's lower bound of λ"),
            "{msg}"
        );
    }
}
