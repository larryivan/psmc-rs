use psmc_rs::PsmcModel;

fn approx_eq(a: f64, b: f64, eps: f64) {
    assert!(
        (a - b).abs() <= eps,
        "expected {a} ~= {b} within eps={eps}, got diff={}",
        (a - b).abs()
    );
}

#[test]
fn model_new_rejects_pattern_mismatch() {
    let err = PsmcModel::new(
        15.0,
        4,
        1e-3,
        2e-4,
        2.5e-8,
        Some("1*5".to_string()), // implies n_steps = 5
    )
    .expect_err("expected pattern/n_steps mismatch error");
    assert!(
        err.to_string()
            .contains("pattern implies n_steps=5, but n_steps=4 was provided")
    );
}

#[test]
fn map_lam_expands_pattern_groups() {
    let model = PsmcModel::new(15.0, 4, 1e-3, 2e-4, 2.5e-8, Some("2*2".to_string()))
        .expect("failed to create model");

    let expanded = model
        .map_lam(&[1.0, 2.0, 3.0])
        .expect("failed to map grouped lam");
    assert_eq!(expanded, vec![1.0, 1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn transition_prior_and_emission_are_valid_probabilities() {
    let model = PsmcModel::new(15.0, 6, 1e-3, 2e-4, 2.5e-8, None).expect("model init failed");
    let n_states = model.n_steps + 1;

    let prior_sum: f64 = model.prior_matrix().iter().sum();
    approx_eq(prior_sum, 1.0, 1e-9);
    for p in model.prior_matrix() {
        assert!(*p >= -1e-12, "prior has negative entry: {p}");
    }

    let a = model.transition_matrix();
    for i in 0..n_states {
        let mut row_sum = 0.0;
        for j in 0..n_states {
            let v = a[(i, j)];
            assert!(v.is_finite(), "transition contains non-finite value");
            assert!(v >= -1e-12, "transition has negative entry {v}");
            row_sum += v;
        }
        approx_eq(row_sum, 1.0, 1e-8);
    }

    let em = model.emission_matrix();
    for k in 0..n_states {
        let p_t = em[(0, k)];
        let p_k = em[(1, k)];
        let p_n = em[(2, k)];
        assert!(p_t.is_finite() && p_k.is_finite() && p_n.is_finite());
        assert!(p_t >= 0.0 && p_k >= 0.0);
        approx_eq(p_t + p_k, 1.0, 1e-9);
        approx_eq(p_n, 1.0, 1e-12);
    }
}
