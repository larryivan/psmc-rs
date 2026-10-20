//! Cancelling the E-step and the log-likelihood through `EStepOptions::cancel`.

mod common;

use psmc_rs::hmm::{
    AlphaCacheBudget, Cancelled, EStepOptions, EStepProgress, e_step_with_options,
    log_likelihood_with_options,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// Sequences of `lens` bins, heterozygous every 97th, of one row each.
fn data(lens: &[usize]) -> (Vec<Vec<u8>>, Vec<bool>) {
    let rows: Vec<Vec<u8>> = lens
        .iter()
        .map(|&len| (0..len).map(|i| u8::from(i % 97 == 0)).collect())
        .collect();
    let starts = vec![true; rows.len()];
    (rows, starts)
}

/// The E-step (or with `loglike_only` the log-likelihood) of a 64-state
/// model on sequences of `lens` bins, which must end cancelled.
fn assert_cancelled(lens: &[usize], loglike_only: bool, opts: EStepOptions<'_>) {
    let m = common::model("4+25*2+4+6", 2e-3, 4e-4);
    let (rows, starts) = data(lens);
    let (pi, a, em) = (m.prior_matrix(), m.transition_matrix(), m.emission_matrix());
    let err = if loglike_only {
        log_likelihood_with_options(pi, a, em, &rows, &starts, opts).map(|_| ())
    } else {
        e_step_with_options(pi, a, em, &rows, &starts, opts).map(|_| ())
    }
    .unwrap_err();
    assert!(err.downcast_ref::<Cancelled>().is_some(), "{err:#}");
    assert!(err.to_string().contains("cancel"), "{err}");
}

#[test]
fn cancelled_before_start() {
    let cancel = AtomicBool::new(true);
    let budget = AlphaCacheBudget::new(usize::MAX);
    let opts = EStepOptions {
        cancel: Some(&cancel),
        alpha_cache: Some(&budget),
        ..EStepOptions::default()
    };
    assert_cancelled(&[1000, 2000], false, opts);
    assert_eq!(budget.used_bytes(), 0);

    let opts = EStepOptions {
        cancel: Some(&cancel),
        ..EStepOptions::default()
    };
    assert_cancelled(&[1000, 2000], true, opts);
}

#[test]
fn cancelled_with_progress_bars() {
    let cancel = AtomicBool::new(true);
    let opts = EStepOptions {
        progress_bars: true,
        cancel: Some(&cancel),
        ..EStepOptions::default()
    };
    assert_cancelled(&[1000, 2000, 3000], false, opts);
}

#[test]
fn cancelled_while_running() {
    // One long sequence, cancelled once the forward pass is under way: this
    // must be noticed inside the sequence, not only between sequences.
    let cancel = AtomicBool::new(false);
    let budget = AlphaCacheBudget::new(0);
    let mut cb = |e: EStepProgress| {
        if e.done > 0 {
            cancel.store(true, Ordering::Relaxed);
        }
    };
    let opts = EStepOptions {
        on_progress: Some(&mut cb),
        cancel: Some(&cancel),
        alpha_cache: Some(&budget),
        ..EStepOptions::default()
    };
    assert_cancelled(&[1_500_000], false, opts);
}
