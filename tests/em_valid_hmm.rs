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

use psmc_rs::em::{hmm_q, hmm_q0};
use psmc_rs::hmm::e_step_streaming;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const INPUT: &str = "tests/data/sim_bottleneck/bottleneck_5mb.psmcfa.gz";

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
/// every iteration and returns the final log-likelihood.
fn fit_and_check(n_iter: usize, config: &MStepConfig) -> f64 {
    let obs = read_psmcfa(Path::new(INPUT), None).unwrap();
    let theta0 = initial_theta(&obs.rows).unwrap();
    let mut model = PsmcModel::new(15.0, 63, theta0, theta0 / 5.0, 2.5e-8, None).unwrap();
    let template = model.clone();
    let history = em_train(&mut model, &obs.rows, &obs.row_starts, n_iter, config).unwrap();

    for (i, params) in history.params.iter().enumerate().skip(1) {
        let mut at = template.clone();
        at.theta = params[0];
        at.rho = params[1];
        at.t_max = params[2];
        at.lam.copy_from_slice(&params[3..]);
        at.param_recalculate().unwrap();
        let (deviation, valid) = validity(&at);
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
    let e = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        false,
        "E",
    )
    .unwrap();
    let q = hmm_q(model.transition_matrix(), model.emission_matrix(), &e.stats) - hmm_q0(&e.stats);
    assert!(q <= 0.0, "Q - Q0 = {q} at the fitted model");
    history.final_loglike().unwrap()
}

#[test]
fn em_keeps_a_valid_hmm_with_lambda_up_to_1e6() {
    // The bounds under which the M-step left the HMMs (lam_hi 1e6): their
    // 8th iteration did, from round 7's statistics.
    let config = MStepConfig {
        progress: false,
        lam_hi: 1e6,
        ..MStepConfig::default()
    };
    fit_and_check(10, &config);
}
