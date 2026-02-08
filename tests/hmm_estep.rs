use psmc_rs::PsmcModel;
use psmc_rs::hmm::e_step_streaming;

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ~= {b} within eps={eps}, got diff={}",
        (a - b).abs()
    );
}

#[test]
fn e_step_streaming_independent_rows_produces_consistent_statistics() {
    let model = PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, None).expect("model init failed");
    let rows = vec![vec![0u8, 1, 0, 2, 1, 0], vec![1u8, 0, 0, 1, 2, 2]];
    let row_starts = vec![true, true];

    let out = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &rows,
        &row_starts,
        false,
        "E",
    )
    .expect("e-step failed");

    assert!(out.loglike.is_finite(), "loglike should be finite");
    let n_states = model.n_steps + 1;
    assert_eq!(out.stats.xi.shape(), &[n_states, n_states]);
    assert_eq!(out.stats.gobs[0].len(), n_states);
    assert_eq!(out.stats.gobs[1].len(), n_states);
    assert_eq!(out.stats.gobs[2].len(), n_states);

    let mut obs_counts = [0usize; 3];
    let mut total_transitions = 0usize;
    for row in &rows {
        for v in row {
            obs_counts[*v as usize] += 1;
        }
        total_transitions += row.len().saturating_sub(1);
    }
    for (obs, count) in obs_counts.iter().enumerate() {
        let gobs_sum: f64 = out.stats.gobs[obs].iter().sum();
        approx_eq(gobs_sum, *count as f64, 1e-5);
    }

    let xi_sum: f64 = out.stats.xi.iter().sum();
    approx_eq(xi_sum, total_transitions as f64, 1e-5);
}

#[test]
fn e_step_streaming_contiguous_rows_keeps_sequence_chain() {
    let model = PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, None).expect("model init failed");
    let rows = vec![vec![0u8, 1, 0, 2, 1, 0], vec![1u8, 0, 0, 1, 2, 2]];
    let row_starts = vec![true, false];

    let out = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &rows,
        &row_starts,
        false,
        "E",
    )
    .expect("e-step failed");

    let mut obs_counts = [0usize; 3];
    let mut total_len = 0usize;
    for row in &rows {
        for v in row {
            obs_counts[*v as usize] += 1;
        }
        total_len += row.len();
    }
    for (obs, count) in obs_counts.iter().enumerate() {
        let gobs_sum: f64 = out.stats.gobs[obs].iter().sum();
        approx_eq(gobs_sum, *count as f64, 1e-5);
    }

    let xi_sum: f64 = out.stats.xi.iter().sum();
    approx_eq(xi_sum, (total_len - 1) as f64, 1e-5);
}
