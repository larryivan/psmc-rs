//! The start of a fit: its parameters, fresh or saved, and the checks that
//! refuse a start EM cannot move from, with advice in the run's options.

use anyhow::{Context as _, Result, bail};
use std::path::Path;

use psmc_rs::bootstrap::{
    BootstrapConfig, InvalidReplicateStart, invalid_replicate_starts_sequences,
};
use psmc_rs::io::params::load_params;
use psmc_rs::io::{CallableBases, Observations};
use psmc_rs::model::NegativeTransitions;
use psmc_rs::psmc_out::c_pattern;
use psmc_rs::qc::FitStart;
use psmc_rs::{PsmcModel, initial_theta_with_callable};

use super::Context;
use super::bootstrap::{Bootstrap, list_replicates};
use crate::cli::{InferArgs, layout_model, usage_bail};

/// The starting model of `--init-params`: the saved parameters, with the
/// layout checked against `--pattern` and `--n-steps` when they are given
/// (a mismatch is a usage error), and `--mu` for scaling.
pub(super) fn saved_start(path: &Path, cli: &InferArgs) -> Result<PsmcModel> {
    let what = || format!("--init-params {}", path.display());
    // The reader's errors name the file.
    let saved = load_params(path).context("--init-params")?;
    let mut model = PsmcModel::from_params_file(&saved).with_context(what)?;
    if cli.pattern.is_some() || cli.n_steps.is_some() {
        let wanted = layout_model(cli.pattern.as_deref(), cli.n_steps, model.t_max, model.mu)?;
        if (wanted.n_steps, c_pattern(&wanted)) != (model.n_steps, c_pattern(&model)) {
            usage_bail!(
                "{} holds a model with pattern {} ({} intervals), but --pattern/--n-steps ask \
                 for {} ({} intervals); leave them out to continue the saved model",
                what(),
                c_pattern(&model),
                model.n_steps + 1,
                c_pattern(&wanted),
                wanted.n_steps + 1
            );
        }
    }
    model.mu = cli.mu;
    Ok(model)
}

/// θ0 and ρ0 of a fresh start: `--theta0` or θ0 of the data (C psmc's
/// -ln(1 - K/L), or its analogue for the callable bases of each bin when
/// the fit uses them: [`initial_theta_with_callable`]), and `--rho0` or θ0
/// / `--tr-ratio`.
fn start_rates(
    cli: &InferArgs,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
) -> Result<(f64, f64)> {
    let theta0 = match cli.theta0 {
        Some(v) => v,
        None => initial_theta_with_callable(rows, callable)?,
    };
    Ok((theta0, cli.rho0.unwrap_or(theta0 / cli.tr_ratio)))
}

/// The model the fit of `obs` starts from: `layout` itself when it holds
/// the saved parameters of `--init-params`, otherwise a fresh start of its
/// layout with flat lambdas.
pub(super) fn starting_model(
    layout: PsmcModel,
    obs: &Observations,
    cli: &InferArgs,
) -> Result<PsmcModel> {
    if cli.init_params.is_some() {
        return Ok(layout);
    }
    let (theta0, rho0) = start_rates(cli, &obs.rows, obs.callable.as_ref())?;
    PsmcModel::new(
        layout.t_max,
        layout.n_steps,
        theta0,
        rho0,
        layout.mu,
        layout.pattern,
    )
}

/// The start of the fit, as QC checks it ([`FitStart`]): the saved
/// parameters of `--init-params` (then `layout`), or the rates of
/// [`start_rates`] on `layout`'s time grid; `None` when the data give no
/// θ0, which the run then reports.
pub(super) fn fit_start(
    cli: &InferArgs,
    layout: &PsmcModel,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
) -> Option<FitStart> {
    let (theta0, rho0) = match cli.init_params {
        Some(_) => (layout.theta, layout.rho),
        None => start_rates(cli, rows, callable).ok()?,
    };
    let mut start = FitStart::new(layout, theta0, rho0, false);
    start.saved = cli.init_params.is_some();
    Some(start)
}

/// Check the starts before the main fit, given the bootstrap `boot` that
/// follows it: replicates whose fresh start EM cannot fit are named and
/// skipped (an error when that is all of them), and the main start
/// `model` is refused as EM would refuse it, with advice that fits the
/// replicates too.
pub(super) fn check_starts(
    model: &PsmcModel,
    boot: Option<&Bootstrap>,
    n_iter: usize,
    cli: &InferArgs,
    ctx: Context,
) -> Result<()> {
    // A replicate's fresh start takes theta0 from its own resampled data,
    // so it can be past the limit on rho when the main start is not. Such
    // replicates are found now, before the main fit, and the bootstrap
    // skips them.
    let invalid = match boot {
        Some(b) if b.start.is_none() => {
            invalid_replicate_starts_sequences(&b.seqs, model, &b.plan.cfg)?
        }
        _ => Vec::new(),
    };
    // EM would refuse a start it cannot move from; refuse it here, with
    // advice in this run's options that fits the replicates too. With
    // saved parameters, every bootstrap replicate starts from them, so
    // they are checked even when the main fit runs no iteration.
    if n_iter > 0 || boot.is_some_and(|b| b.start.is_some()) {
        check_start(model, cli, replicates_tr_ratio(&invalid))?;
    }
    if let Some(b) = boot
        && let Some(msg) = invalid_replicates_warning(cli, model, &b.plan.cfg, &invalid)?
    {
        ctx.warn(msg);
    }
    Ok(())
}

/// Refuse a start from which EM cannot move
/// ([`PsmcModel::check_transitions`]), as EM would, but with advice in the
/// options of this run ([`start_advice`]). `replicates_ratio` is the
/// theta0/rho0 that the bootstrap replicates need
/// ([`replicates_tr_ratio`]), if more than the start's.
fn check_start(model: &PsmcModel, cli: &InferArgs, replicates_ratio: Option<f64>) -> Result<()> {
    let Err(e) = model.check_transitions() else {
        return Ok(());
    };
    let Some(negative) = e.downcast_ref::<NegativeTransitions>() else {
        return Err(e.context("EM cannot start from these parameters"));
    };
    bail!(
        "EM cannot start from these parameters: {}. {}",
        negative.finding(),
        start_advice(cli, negative, replicates_ratio)
    )
}

/// How this run can start from a smaller rho than the start of
/// `negative`: with `--init-params`, which rules out the options of a
/// fresh start, by lowering rho in the saved file; otherwise with a
/// smaller `--rho0` when one was given, and with `--tr-ratio` (C psmc's
/// `-r`) or `--rho0` when not. Smaller bins also lower rho per bin. The
/// ratio advised is at least `replicates_ratio`, which the bootstrap
/// replicates need ([`replicates_tr_ratio`]), so that following the advice
/// fits them too.
fn start_advice(
    cli: &InferArgs,
    negative: &NegativeTransitions,
    replicates_ratio: Option<f64>,
) -> String {
    let own = negative.min_tr_ratio();
    let ratio = own.map(|r| replicates_ratio.map_or(r, |b| r.max(b)));
    // The bootstrap replicates, which take theta0 from their own resampled
    // data, can need a larger ratio than the start.
    let replicates = if ratio > own {
        "; the bootstrap replicates, whose theta0 comes from their own resampled data, need \
         that much"
    } else {
        ""
    };
    match (&cli.init_params, cli.rho0, ratio) {
        (Some(path), ..) => format!(
            "Set rho in {}{}, or fit from a fresh start instead of --init-params",
            path.display(),
            negative
                .max_rho
                .map_or(String::new(), |rho| format!(" below {rho:.6}"))
        ),
        (None, Some(_), Some(ratio)) => format!(
            "Re-bin the data with fewer base pairs per bin, or give a --rho0 of {} or less \
             (theta0/rho0 {ratio} or more{replicates})",
            round_down(negative.theta / ratio)
        ),
        (None, None, Some(ratio)) => format!(
            "Re-bin the data with fewer base pairs per bin, or start from a smaller rho0: \
             --tr-ratio {ratio} or more (C psmc's -r), or a --rho0 of {} or less{replicates}",
            round_down(negative.theta / ratio)
        ),
        (None, _, None) => "Re-bin the data with fewer base pairs per bin, or start from a \
                            smaller rho0 with a larger --tr-ratio (C psmc's -r) or a smaller \
                            --rho0"
            .to_string(),
    }
}

/// The warning about the bootstrap replicates of `cfg` whose fresh start
/// is `invalid` ([`invalid_replicate_starts_sequences`]), which the
/// bootstrap skips, with the start that lets every replicate be fitted; an
/// error when that is all of them, before the main fit. `model` is the
/// main fit's start.
fn invalid_replicates_warning(
    cli: &InferArgs,
    model: &PsmcModel,
    cfg: &BootstrapConfig,
    invalid: &[InvalidReplicateStart],
) -> Result<Option<String>> {
    let Some(first) = invalid.first() else {
        return Ok(None);
    };
    let numbers: Vec<usize> = invalid.iter().map(|r| r.index + 1).collect();
    let highest_rho0 = invalid.iter().map(|r| r.start.rho).fold(0.0, f64::max);
    let past = match first.start.max_rho {
        Some(limit) => {
            format!("rho0 up to {highest_rho0:.6}, above the limit of {limit:.6} per bin")
        }
        None => format!("rho0 up to {highest_rho0:.6} per bin"),
    };
    // Replicates take the main start's theta0/rho0.
    let advice = match (replicates_tr_ratio(invalid), cli.rho0) {
        (Some(ratio), Some(_)) => format!(
            "start from a --rho0 of {} or less, which gives every replicate theta0/rho0 {ratio} \
             or more",
            round_down(model.theta / ratio)
        ),
        (Some(ratio), None) => format!("start from --tr-ratio {ratio} or more (C psmc's -r)"),
        (None, _) => "start from a smaller rho0".to_string(),
    };
    if invalid.len() == cfg.replicates {
        bail!(
            "every one of the {} bootstrap replicates would start from negative transition \
             probabilities ({past}): their theta0, from their own resampled data, sets their \
             rho0, so none of them can be fitted. To fit them, {advice}",
            cfg.replicates
        );
    }
    let fitted = cfg.replicates - invalid.len();
    Ok(Some(format!(
        "bootstrap {} ({} of {}) would start from negative transition probabilities ({past}): \
         their resampled data are more heterozygous than the input, and their theta0, from \
         those data, sets their rho0. They will be skipped and the bands will come from the \
         other {fitted}; leaving out the most heterozygous resamples can make the bands too \
         narrow. To fit every replicate, {advice}",
        list_replicates(&numbers),
        invalid.len(),
        cfg.replicates
    )))
}

/// The smallest theta0/rho0 at which every one of the `invalid` replicate
/// starts is valid; `None` without any, or when one has no limit.
fn replicates_tr_ratio(invalid: &[InvalidReplicateStart]) -> Option<f64> {
    if invalid.is_empty() {
        return None;
    }
    invalid
        .iter()
        .try_fold(0.0, |max: f64, r| Some(max.max(r.start.min_tr_ratio()?)))
}

/// `x` rounded down to 6 significant digits, for a bound not to be passed.
fn round_down(x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return x;
    }
    let scale = 10f64.powi(5 - x.log10().floor() as i32);
    (x * scale).floor() / scale
}
