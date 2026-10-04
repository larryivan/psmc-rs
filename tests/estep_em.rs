//! The EM driver's log-likelihood history.

use psmc_rs::hmm::log_likelihood;
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};

fn data() -> (Vec<Vec<u8>>, Vec<bool>) {
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|s| {
            (0..4000)
                .map(|i| match (i * 31 + s * 7) % 211 {
                    0..=2 => 1,
                    3 => 2,
                    _ => 0,
                })
                .collect()
        })
        .collect();
    let starts = vec![true; rows.len()];
    (rows, starts)
}

fn config() -> MStepConfig {
    MStepConfig {
        progress: false,
        max_iters: 20,
        ..MStepConfig::default()
    }
}

#[test]
fn history_records_loglike_before_and_after_each_iteration() {
    let (rows, starts) = data();
    let theta0 = initial_theta(&rows).unwrap();
    let mut model = PsmcModel::new(
        15.0,
        22,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some("4+5*3+4".into()),
    )
    .unwrap();
    let n_iter = 4;
    let history = em_train(&mut model, &rows, &starts, n_iter, &config()).unwrap();
    assert_eq!(history.loglike.len(), n_iter);
    assert_eq!(history.params.len(), n_iter + 1);

    for w in history.loglike.windows(2) {
        assert_eq!(
            w[0].1.to_bits(),
            w[1].0.to_bits(),
            "'after' of an iteration must be 'before' of the next: {w:?}"
        );
    }
    for &(before, after) in &history.loglike {
        assert!(after.is_finite());
        assert!(after >= before - 1e-6, "EM decreased the likelihood");
    }
    assert!(
        history.loglike[0].1 > history.loglike[0].0,
        "the first iteration should improve the fit"
    );

    // The last 'after' is the log-likelihood of the parameters EM returns.
    model.param_recalculate().unwrap();
    let final_ll = log_likelihood(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &rows,
        &starts,
    )
    .unwrap();
    assert_eq!(history.loglike[n_iter - 1].1.to_bits(), final_ll.to_bits());
}

#[test]
fn zero_iterations_leave_an_empty_history() {
    let (rows, starts) = data();
    let mut model = PsmcModel::new(15.0, 22, 0.01, 0.002, 2.5e-8, Some("4+5*3+4".into())).unwrap();
    let history = em_train(&mut model, &rows, &starts, 0, &config()).unwrap();
    assert!(history.loglike.is_empty());
    assert_eq!(history.params.len(), 1);
}
