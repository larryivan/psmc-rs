//! An M-step must end at a maximum of Q along every parameter: moving any
//! one parameter alone must not raise Q.
//!
//! On a simulated 2 Mb bottleneck genome fitted with one λ per interval,
//! the first M-step once stopped with the first λ at 9,859 while Q peaks
//! at 49 along it (7.6e-4 higher): the λ, which the data barely constrain,
//! had been carried past its optimum to where Q falls slowly towards the
//! bound, and the L-BFGS memory predicted no further decrease there.

mod common;

use common::{c_start_free, e_step, fixture_obs, model_at, mstep, params};
use psmc_rs::PsmcModel;
use psmc_rs::em::{hmm_q, hmm_q0};
use psmc_rs::hmm::SufficientStats;
use psmc_rs::opt::{bounds_from_config, m_step_lbfgs_with_outcome};

/// C psmc's Q at `params` (θ, ρ, t_max, λ...) of `template`'s layout;
/// -inf where the transition matrix is not stochastic.
fn q_at(template: &PsmcModel, stats: &SufficientStats, params: &[f64]) -> f64 {
    let m = model_at(template, params);
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
    let obs = fixture_obs("sim_bottleneck/bottleneck_2mb.psmcfa.gz");
    let model = c_start_free(&obs.rows, 63, 5.0);
    let stats = e_step(&model, &obs.rows, &obs.row_starts).stats;
    let config = mstep();
    let bounds = bounds_from_config(&model, &config);
    let outcome =
        m_step_lbfgs_with_outcome(&model, &stats, &params(&model), &bounds, &config, None).unwrap();
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
