//! EM must keep a valid HMM: every M-step's parameters must give a
//! stochastic transition matrix in f64, and the log-likelihood must not
//! fall.
//!
//! On a simulated bottleneck genome fitted with one λ per interval, the
//! L-BFGS M-step once took an early λ to about 3e-5, next to λ at their
//! upper bound. The survival `alpha` then underflows in the squares of
//! C's formulas, so that the f64 transition matrix's rows sum to up to
//! 1.014 and its Q exceeds what any HMM reaches; EM's next E-step lowered
//! the log-likelihood by 367 and the fit never left that point.

mod common;

use common::{c_start_free, e_step, fixture_obs, model_at, mstep};
use psmc_rs::PsmcModel;
use psmc_rs::em::{hmm_q, hmm_q0};
use psmc_rs::opt::{EmHistory, MStepConfig, em_train};

/// The largest deviation of a row of `model`'s transition matrix from
/// summing to 1, and whether every probability is in [0, 1].
fn validity(model: &PsmcModel) -> (f64, bool) {
    let a = model.transition_matrix();
    let in_unit = |p: f64| (0.0..=1.0).contains(&p);
    let valid =
        a.iter().all(|&p| in_unit(p)) && model.emission_matrix().iter().all(|&e| in_unit(e));
    let deviation = a
        .rows()
        .into_iter()
        .map(|row| (row.iter().fold(0.0, |acc, &p| acc + p) - 1.0).abs())
        .fold(0.0, f64::max);
    (deviation, valid)
}

/// EM on the 5 Mb input as `psmc-rs IN OUT N --n-steps 63 --t-max 15
/// --tr-ratio 5` runs it, with the M-step settings of `config`; checks
/// every iteration and returns the history.
fn fit_and_check(n_iter: usize, config: &MStepConfig) -> EmHistory {
    let obs = fixture_obs("sim_bottleneck/bottleneck_5mb.psmcfa.gz");
    let template = c_start_free(&obs.rows, 63, 5.0);
    let mut model = template.clone();
    let history = em_train(&mut model, &obs.rows, &obs.row_starts, n_iter, config).unwrap();

    for (i, params) in history.params.iter().enumerate().skip(1) {
        let (deviation, valid) = validity(&model_at(&template, params));
        assert!(
            valid && deviation <= 1e-9,
            "iteration {i}: rows sum to 1 within {deviation:e} (probabilities in [0, 1]: \
             {valid}) at {params:?}"
        );
    }
    for (i, &(before, after)) in history.q.iter().enumerate() {
        // The M-step only accepts decreases of -Q, and C's Q (Q - Q0) of an
        // HMM is at most 0.
        assert!(
            after >= before && after <= 0.0,
            "iteration {}: Q {before} -> {after}",
            i + 1
        );
    }
    for (i, &(before, after)) in history.loglike.iter().enumerate() {
        // EM does not lower the log-likelihood, up to rounding and the
        // initial state's term, which C's Q leaves out.
        assert!(
            after >= before - 1e-9 * before.abs(),
            "iteration {}: log-likelihood {before} -> {after}",
            i + 1
        );
    }

    // The fitted model is the last one evaluated.
    let stats = e_step(&model, &obs.rows, &obs.row_starts).stats;
    let q = hmm_q(model.transition_matrix(), model.emission_matrix(), &stats) - hmm_q0(&stats);
    assert!(q <= 0.0, "Q - Q0 = {q} at the fitted model");
    history
}

#[test]
fn em_keeps_a_valid_hmm_with_lambda_up_to_1e6() {
    // The bounds under which the M-step left the HMMs (lam_hi 1e6): their
    // 8th iteration did, from round 7's statistics.
    let config = MStepConfig {
        lam_hi: 1e6,
        ..mstep()
    };
    fit_and_check(10, &config);
}

#[test]
fn em_keeps_improving_with_one_lambda_per_interval() {
    // With λ up to 1e6 the first two λ reached that bound and the fit
    // stopped moving at round 16 (log-likelihood -8445.92), every M-step
    // failing among the infeasible points around its start. With the
    // default bound the fit improves every round, as C psmc's does
    // (`psmc -N40 -t15 -r5 -p 64*1`: -8445.3196 in round 40).
    let history = fit_and_check(40, &mstep());
    for (i, &(before, after)) in history.loglike.iter().enumerate().skip(1) {
        assert!(after > before, "iteration {}: {before} -> {after}", i + 1);
    }
    let round_40 = history.loglike[39].0;
    assert!(round_40 > -8445.33, "round 40: {round_40}");
}
