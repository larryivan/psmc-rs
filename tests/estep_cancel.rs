//! Cancelling the E-step and the log-likelihood through `EStepOptions::cancel`.

use psmc_rs::PsmcModel;
use psmc_rs::hmm::{
    AlphaCacheBudget, Cancelled, EStepOptions, EStepProgress, e_step_with_options,
    log_likelihood_with_options,
};
use std::sync::atomic::{AtomicBool, Ordering};

fn model() -> PsmcModel {
    PsmcModel::new(15.0, 63, 2e-3, 4e-4, 2.5e-8, Some("4+25*2+4+6".into())).unwrap()
}

fn data(lens: &[usize]) -> (Vec<Vec<u8>>, Vec<bool>) {
    let rows: Vec<Vec<u8>> = lens
        .iter()
        .map(|&len| (0..len).map(|i| u8::from(i % 97 == 0)).collect())
        .collect();
    let starts = vec![true; rows.len()];
    (rows, starts)
}

fn assert_cancelled(err: anyhow::Error) {
    assert!(err.downcast_ref::<Cancelled>().is_some(), "{err:#}");
    assert!(err.to_string().contains("cancel"), "{err}");
}

#[test]
fn cancelled_before_start() {
    let m = model();
    let (rows, starts) = data(&[1000, 2000]);
    let cancel = AtomicBool::new(true);
    let budget = AlphaCacheBudget::new(usize::MAX);
    let err = e_step_with_options(
        m.prior_matrix(),
        m.transition_matrix(),
        m.emission_matrix(),
        &rows,
        &starts,
        EStepOptions {
            cancel: Some(&cancel),
            alpha_cache: Some(&budget),
            ..EStepOptions::default()
        },
    )
    .unwrap_err();
    assert_cancelled(err);
    assert_eq!(budget.used_bytes(), 0);

    let err = log_likelihood_with_options(
        m.prior_matrix(),
        m.transition_matrix(),
        m.emission_matrix(),
        &rows,
        &starts,
        EStepOptions {
            cancel: Some(&cancel),
            ..EStepOptions::default()
        },
    )
    .unwrap_err();
    assert_cancelled(err);
}

#[test]
fn cancelled_while_running() {
    // One long sequence, cancelled once the forward pass is under way: this
    // must be noticed inside the sequence, not only between sequences.
    let m = model();
    let len = 1_500_000;
    let (rows, starts) = data(&[len]);
    let cancel = AtomicBool::new(false);
    let budget = AlphaCacheBudget::new(0);
    let mut cb = |e: EStepProgress| {
        if e.done > 0 {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let err = e_step_with_options(
        m.prior_matrix(),
        m.transition_matrix(),
        m.emission_matrix(),
        &rows,
        &starts,
        EStepOptions {
            on_progress: Some(&mut cb),
            cancel: Some(&cancel),
            alpha_cache: Some(&budget),
            ..EStepOptions::default()
        },
    )
    .unwrap_err();
    assert_cancelled(err);
}
