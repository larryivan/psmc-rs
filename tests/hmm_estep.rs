use ndarray::arr2;
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
fn e_step_streaming_produces_consistent_sufficient_statistics() {
    let model = PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, None).expect("model init failed");
    let x = arr2(&[[0u8, 1, 0, 2, 1, 0], [1u8, 0, 0, 1, 2, 2]]);
    let (batch, len) = x.dim();

    let out = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &x,
        true,
        false,
        "E",
    )
    .expect("e-step failed");

    assert!(out.loglike.is_finite(), "loglike should be finite");
    let n_states = model.n_steps + 1;
    assert_eq!(out.stats.xi.shape(), &[n_states, n_states]);
    assert_eq!(out.stats.g0.len(), n_states);
    assert_eq!(out.stats.gobs[0].len(), n_states);
    assert_eq!(out.stats.gobs[1].len(), n_states);
    assert_eq!(out.stats.gobs[2].len(), n_states);

    let g0_sum: f64 = out.stats.g0.iter().sum();
    approx_eq(g0_sum, batch as f64, 1e-6);

    let mut obs_counts = [0usize; 3];
    for v in x.iter() {
        obs_counts[*v as usize] += 1;
    }
    for obs in 0..3 {
        let gobs_sum: f64 = out.stats.gobs[obs].iter().sum();
        approx_eq(gobs_sum, obs_counts[obs] as f64, 1e-5);
    }

    let xi_sum: f64 = out.stats.xi.iter().sum();
    approx_eq(xi_sum, (batch * (len - 1)) as f64, 1e-5);
}

#[test]
fn e_step_streaming_contiguous_rows_keeps_sequence_chain() {
    let model = PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, None).expect("model init failed");
    let x = arr2(&[[0u8, 1, 0, 2, 1, 0], [1u8, 0, 0, 1, 2, 2]]);
    let (batch, len) = x.dim();

    let out = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &x,
        false,
        false,
        "E",
    )
    .expect("e-step failed");

    let g0_sum: f64 = out.stats.g0.iter().sum();
    approx_eq(g0_sum, 1.0, 1e-6);

    let mut obs_counts = [0usize; 3];
    for v in x.iter() {
        obs_counts[*v as usize] += 1;
    }
    for obs in 0..3 {
        let gobs_sum: f64 = out.stats.gobs[obs].iter().sum();
        approx_eq(gobs_sum, obs_counts[obs] as f64, 1e-5);
    }

    let xi_sum: f64 = out.stats.xi.iter().sum();
    approx_eq(xi_sum, (batch * len - 1) as f64, 1e-5);
}
