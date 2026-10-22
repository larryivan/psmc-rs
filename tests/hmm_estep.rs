//! The E-step's expected counts on a few bins, by hand.

mod common;

use common::e_step;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::e_step_streaming;

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ~= {b} within eps={eps}, got diff={}",
        (a - b).abs()
    );
}

fn small_model() -> PsmcModel {
    PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, None).unwrap()
}

/// The expected emissions add up to the bins of each symbol, and the
/// expected transitions to the transitions between bins: within each row
/// when every row starts a sequence, and across the cut too when the second
/// row continues the first.
#[test]
fn e_step_counts_add_up_for_independent_and_chained_rows() {
    let model = small_model();
    let n_states = model.n_steps + 1;
    let rows = vec![vec![0u8, 1, 0, 2, 1, 0], vec![1u8, 0, 0, 1, 2, 2]];
    let mut obs_counts = [0usize; 3];
    for v in rows.iter().flatten() {
        obs_counts[*v as usize] += 1;
    }
    for (row_starts, transitions) in [([true, true], 10), ([true, false], 11)] {
        let out = e_step(&model, &rows, &row_starts);
        assert!(out.loglike.is_finite(), "loglike should be finite");
        let sums = out.stats.transitions.parts().concat();
        assert_eq!(sums.len(), 5 * n_states);
        for (obs, count) in obs_counts.iter().enumerate() {
            assert_eq!(out.stats.gobs[obs].len(), n_states);
            let gobs_sum: f64 = out.stats.gobs[obs].iter().sum();
            approx_eq(gobs_sum, *count as f64, 1e-5);
        }
        // Every transition is in a row of one triangle or on the diagonal.
        let t = &out.stats.transitions;
        let xi_sum: f64 = [&t.lower_rows, &t.upper_rows, &t.diagonal]
            .into_iter()
            .flatten()
            .sum();
        approx_eq(xi_sum, f64::from(transitions), 1e-5);
        let cols: f64 = t.lower_cols.iter().chain(&t.upper_cols).sum();
        let rows: f64 = t.lower_rows.iter().chain(&t.upper_rows).sum();
        approx_eq(cols, rows, 1e-9);
    }
}

#[test]
fn e_step_rejects_non_finite_normalisers() {
    // NaN parameters (e.g. from a failed M-step) must not turn into a NaN
    // log-likelihood silently.
    let model = small_model();
    let mut em = model.emission_matrix().clone();
    em[(0, 2)] = f64::NAN;
    let rows = vec![vec![0u8, 1, 0, 2, 1, 0]];
    let err = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        &em,
        &rows,
        &[true],
        false,
        "E",
    )
    .expect_err("a NaN emission probability must be an error");
    assert!(err.to_string().contains("normalization factor"), "{err}");
}
