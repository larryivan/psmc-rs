//! An M-step must end at a maximum of Q along every parameter: moving any
//! one parameter alone must not raise Q.
//!
//! On a simulated 2 Mb bottleneck genome fitted with one λ per interval,
//! the first M-step once stopped with the first λ at 9,859 while Q peaks
//! at 49 along it (7.6e-4 higher): the λ, which the data barely constrain,
//! had been carried past its optimum to where Q falls slowly towards the
//! bound, and the L-BFGS memory predicted no further decrease there.

use psmc_rs::em::{hmm_q, hmm_q0};
use psmc_rs::hmm::{SufficientStats, e_step_streaming};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, bounds_from_config, m_step_lbfgs_with_outcome};
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

/// C psmc's Q at `params` (θ, ρ, t_max, λ...) of `template`'s layout;
/// -inf where the transition matrix is not stochastic.
fn q_at(template: &PsmcModel, stats: &SufficientStats, params: &[f64]) -> f64 {
    let mut m = template.clone();
    m.theta = params[0];
    m.rho = params[1];
    m.t_max = params[2];
    m.lam.copy_from_slice(&params[3..]);
    m.param_recalculate().unwrap();
    let rows_ok = m
        .transition_matrix()
        .rows()
        .into_iter()
        .all(|row| (row.iter().fold(0.0, |a, &p| a + p) - 1.0).abs() <= 1e-9);
    if !rows_ok {
        return f64::NEG_INFINITY;
    }
    hmm_q(m.transition_matrix(), m.emission_matrix(), stats) - hmm_q0(stats)
}

#[test]
fn first_m_step_ends_at_a_maximum_along_every_parameter() {
    let obs = read_psmcfa(
        Path::new("tests/data/sim_bottleneck/bottleneck_2mb.psmcfa.gz"),
        None,
    )
    .unwrap();
    let theta0 = initial_theta(&obs.rows).unwrap();
    let model = PsmcModel::new(15.0, 63, theta0, theta0 / 5.0, 2.5e-8, None).unwrap();
    let stats = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        false,
        "E",
    )
    .unwrap()
    .stats;
    let config = MStepConfig {
        progress: false,
        ..MStepConfig::default()
    };
    let bounds = bounds_from_config(&model, &config);
    let mut init = vec![model.theta, model.rho, model.t_max];
    init.extend_from_slice(&model.lam);
    let outcome = m_step_lbfgs_with_outcome(&model, &stats, &init, &bounds, &config, None).unwrap();
    let q = q_at(&model, &stats, &outcome.params);
    assert!(q > -4.45, "Q - Q0 = {q} ({:?})", outcome.stop);

    // Each parameter alone, by factors 10^(k/20) from 1e-3 to 1e3 within
    // its bounds: Q may not rise by more than its rounding.
    for (i, b) in bounds.iter().enumerate() {
        for k in -60..=60 {
            let mut p = outcome.params.clone();
            p[i] *= 10f64.powf(k as f64 / 20.0);
            if p[i] < b.lo || p[i] > b.hi {
                continue;
            }
            let gain = q_at(&model, &stats, &p) - q;
            assert!(
                gain <= 1e-9,
                "parameter {i}: {} -> {} raises Q by {gain:e} (from {q})",
                outcome.params[i],
                p[i]
            );
        }
    }
}
