//! The per-iteration quantities EmHistory records for C-style `.psmc`
//! output: post_sigma and the M-step's evaluations.

mod common;

use common::{c_start, e_step, model_at, mstep};
use psmc_rs::em::posterior_sigma;
use psmc_rs::hmm::{SufficientStats, TransitionSums};
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

#[test]
fn history_records_c_round_quantities() {
    let (rows, starts) = data();
    let template = c_start(&rows, "4+5*3+4", 4.0);
    let mut model = template.clone();
    let config = MStepConfig {
        max_iters: 30,
        ..mstep()
    };
    let n_iter = 4;
    let history = em_train(&mut model, &rows, &starts, n_iter, &config).unwrap();
    assert_eq!(history.post_sigma.len(), n_iter);
    assert_eq!(history.mstep_evaluations.len(), n_iter);
    assert_eq!(history.mstep_iterations.len(), n_iter);
    assert_eq!(history.mstep_stops.len(), n_iter);

    for i in 0..n_iter {
        // Recompute iteration i's E-step, which runs at params[i].
        let entering = model_at(&template, &history.params[i]);
        let stats = e_step(&entering, &rows, &starts).stats;
        assert_eq!(
            history.post_sigma[i],
            posterior_sigma(&stats),
            "iteration {i}"
        );
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
fn zero_iterations_record_nothing() {
    let (rows, starts) = data();
    let mut model = c_start(&rows, "4+5*3+4", 4.0);
    let history = em_train(&mut model, &rows, &starts, 0, &mstep()).unwrap();
    // The start's parameters alone.
    assert_eq!(history.params.len(), 1);
    assert_eq!(history.final_loglike(), None);
    assert!(history.loglike.is_empty());
    assert!(history.post_sigma.is_empty());
    assert!(history.mstep_evaluations.is_empty());
    assert!(history.mstep_iterations.is_empty());
    assert!(history.mstep_stops.is_empty());
}

#[test]
fn posterior_sigma_is_the_normalised_non_missing_count() {
    // Two states; the missing symbol's counts are left out.
    let none = || vec![0.0; 2];
    let stats = SufficientStats {
        gobs: [vec![3.0, 0.5], vec![1.0, 0.0], vec![7.0, 2.0]],
        transitions: TransitionSums {
            lower_rows: none(),
            lower_cols: none(),
            upper_rows: none(),
            upper_cols: none(),
            diagonal: none(),
        },
        callable: None,
    };
    assert_eq!(posterior_sigma(&stats), vec![4.0 / 4.5, 0.5 / 4.5]);
}
