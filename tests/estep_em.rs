//! The EM driver's log-likelihood history.

mod common;

use common::{bits, c_start, model, mstep, pool};
use psmc_rs::PsmcModel;
use psmc_rs::hmm::log_likelihood;
use psmc_rs::opt::{MStepConfig, em_train};

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
        max_iters: 20,
        ..mstep()
    }
}

/// The model the tests of EM's options start from.
fn start() -> PsmcModel {
    model("4+5*3+4", 0.01, 0.002)
}

#[test]
fn history_records_loglike_before_and_after_each_iteration() {
    let (rows, starts) = data();
    let mut model = c_start(&rows, "4+5*3+4", 4.0);
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
    assert_eq!(
        history.final_loglike().map(f64::to_bits),
        Some(final_ll.to_bits())
    );
    assert!(
        history.final_loglike() > Some(history.loglike[n_iter - 1].0),
        "the final log-likelihood is not the one entering the last iteration"
    );
}

mod options {
    use super::*;
    use psmc_rs::em::{EmOptions, EmProgressEvent, em_train_with_options};
    use psmc_rs::hmm::{Cancelled, EStepPhase};
    use std::sync::atomic::{AtomicBool, Ordering};

    fn assert_cancelled(err: &anyhow::Error) {
        assert!(err.is::<Cancelled>(), "{err:#}");
        assert!(err.to_string().contains("cancel"), "{err}");
    }

    #[test]
    fn cancelled_before_training() {
        let (rows, starts) = data();
        let mut m = start();
        let cancel = AtomicBool::new(true);
        let opts = EmOptions {
            cancel: Some(&cancel),
            ..EmOptions::default()
        };
        let err = em_train_with_options(&mut m, &rows, &starts, 3, &config(), opts).unwrap_err();
        assert_cancelled(&err);
    }

    #[test]
    fn cancelled_during_an_e_step_keeps_the_last_completed_iteration() {
        let (rows, starts) = data();
        let mut one_iter = start();
        em_train(&mut one_iter, &rows, &starts, 1, &config()).unwrap();

        let mut m = start();
        let cancel = AtomicBool::new(false);
        let mut events = Vec::new();
        let mut cb = |ev: EmProgressEvent| {
            if let EmProgressEvent::IterStart { iter: 2, .. } = ev {
                cancel.store(true, Ordering::Relaxed);
            }
            events.push(ev);
        };
        let opts = EmOptions {
            on_progress: Some(&mut cb),
            cancel: Some(&cancel),
            ..EmOptions::default()
        };
        let err = em_train_with_options(&mut m, &rows, &starts, 5, &config(), opts).unwrap_err();
        assert_cancelled(&err);
        assert!(err.to_string().contains("iteration 2"), "{err}");
        assert!(!events.iter().any(|e| matches!(
            e,
            EmProgressEvent::MStep { iter: 2, .. } | EmProgressEvent::IterStart { iter: 3, .. }
        )));
        assert_eq!(m.theta.to_bits(), one_iter.theta.to_bits());
        assert_eq!(m.rho.to_bits(), one_iter.rho.to_bits());
        assert_eq!(m.lam, one_iter.lam);
        assert_eq!(m.transition_matrix(), one_iter.transition_matrix());
    }

    #[test]
    fn cancelled_between_e_and_m_step() {
        let (rows, starts) = data();
        let mut m = start();
        let cancel = AtomicBool::new(false);
        let mut m_steps = 0usize;
        let mut cb = |ev: EmProgressEvent| match ev {
            EmProgressEvent::EStep {
                phase: EStepPhase::Backward,
                done,
                total,
                ..
            } if done == total => cancel.store(true, Ordering::Relaxed),
            EmProgressEvent::MStep { .. } => m_steps += 1,
            _ => {}
        };
        let opts = EmOptions {
            on_progress: Some(&mut cb),
            cancel: Some(&cancel),
            ..EmOptions::default()
        };
        let err = em_train_with_options(&mut m, &rows, &starts, 3, &config(), opts).unwrap_err();
        assert_cancelled(&err);
        assert_eq!(m_steps, 0, "the M-step ran after cancellation");
    }

    #[test]
    fn thread_pool_and_callback_do_not_change_the_fit() {
        let (rows, starts) = data();
        let mut a = start();
        let ha = em_train(&mut a, &rows, &starts, 3, &config()).unwrap();

        let pool = pool(2);
        let mut b = start();
        let mut n_events = 0usize;
        let mut cb = |_: EmProgressEvent| n_events += 1;
        let opts = EmOptions {
            on_progress: Some(&mut cb),
            thread_pool: Some(&pool),
            ..EmOptions::default()
        };
        let hb = em_train_with_options(&mut b, &rows, &starts, 3, &config(), opts).unwrap();
        assert!(n_events > 0);
        let values = |h: &psmc_rs::em::EmHistory| -> Vec<f64> {
            let loglike = h.loglike.iter().flat_map(|&(x, y)| [x, y]);
            loglike.chain(h.params.iter().flatten().copied()).collect()
        };
        assert_eq!(bits(&values(&ha)), bits(&values(&hb)));
    }
}
