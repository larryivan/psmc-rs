//! The per-iteration quantities EmHistory records for C-style `.psmc`
//! output: post_sigma, Q before and after the M-step, M-step evaluations.

mod common;

use common::{c_start, e_step, model_at, mstep};
use psmc_rs::em::{hmm_q, hmm_q0, posterior_sigma};
use psmc_rs::opt::{MStepConfig, MStepStop, em_train};

fn data() -> (Vec<Vec<u8>>, Vec<bool>) {
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|s| {
            (0..5000)
                .map(|i| match (i * 37 + s * 11) % 197 {
                    0..=2 => 1,
                    3 | 4 => 2,
                    _ => 0,
                })
                .collect()
        })
        .collect();
    (rows, vec![true, false, true])
}

fn check(lambda: f64) {
    let (rows, starts) = data();
    let template = c_start(&rows, "4+5*3+4", 4.0);
    let mut model = template.clone();
    let config = MStepConfig {
        max_iters: 30,
        lambda,
        ..mstep()
    };
    let n_iter = 4;
    let history = em_train(&mut model, &rows, &starts, n_iter, &config).unwrap();
    assert_eq!(history.post_sigma.len(), n_iter);
    assert_eq!(history.q.len(), n_iter);
    assert_eq!(history.mstep_evaluations.len(), n_iter);
    assert_eq!(history.mstep_iterations.len(), n_iter);
    assert_eq!(history.mstep_stops.len(), n_iter);

    for i in 0..n_iter {
        // Recompute iteration i: its E-step runs at params[i], its M-step
        // leaves params[i + 1].
        let entering = model_at(&template, &history.params[i]);
        let leaving = model_at(&template, &history.params[i + 1]);
        let stats = e_step(&entering, &rows, &starts).stats;
        let q0 = hmm_q0(&stats);
        let q_before = hmm_q(
            entering.transition_matrix(),
            entering.emission_matrix(),
            &stats,
        ) - q0;
        let q_after = hmm_q(
            leaving.transition_matrix(),
            leaving.emission_matrix(),
            &stats,
        ) - q0;
        assert_eq!(
            history.post_sigma[i],
            posterior_sigma(&stats),
            "iteration {i}"
        );
        assert_eq!(history.q[i], (q_before, q_after), "iteration {i}");

        let (before, after) = history.q[i];
        assert!(
            before < 0.0 && after < 0.0,
            "iteration {i}: {:?}",
            history.q[i]
        );
        if lambda == 0.0 {
            // Without the smoothing penalty the M-step maximises Q itself.
            assert!(after > before, "iteration {i}: {:?}", history.q[i]);
        }
        let total: f64 = history.post_sigma[i].iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "iteration {i}: {total}");
        assert!(history.post_sigma[i].iter().all(|&p| p > 0.0));
        assert!(history.mstep_evaluations[i] > 1, "iteration {i}");
        // Every M-step moved the parameters; it stopped at the cap exactly
        // when it ran that many iterations.
        let (iters, stop) = (history.mstep_iterations[i], history.mstep_stops[i]);
        assert!(
            (1..=config.max_iters).contains(&iters),
            "iteration {i}: {iters}"
        );
        assert_eq!(
            stop == MStepStop::MaxIters,
            iters == config.max_iters,
            "iteration {i}: {stop:?} after {iters}"
        );
        assert!(history.mstep_evaluations[i] > iters, "iteration {i}");
    }
}

#[test]
fn history_records_c_round_quantities() {
    check(0.0);
}

#[test]
fn q_excludes_the_smoothing_penalty() {
    // With the penalty the recorded Q is still C's unpenalised Q, which is
    // what the recomputation in `check` evaluates.
    check(1e-2);
}

#[test]
fn zero_iterations_record_nothing() {
    let (rows, starts) = data();
    let mut model = c_start(&rows, "4+5*3+4", 4.0);
    let history = em_train(&mut model, &rows, &starts, 0, &mstep()).unwrap();
    // The start's parameters alone.
    assert_eq!(history.params.len(), 1);
    assert_eq!(history.final_loglike(), None);
    assert!(history.loglike.is_empty());
    assert!(history.post_sigma.is_empty());
    assert!(history.q.is_empty());
    assert!(history.mstep_evaluations.is_empty());
    assert!(history.mstep_iterations.is_empty());
    assert!(history.mstep_stops.is_empty());
}
