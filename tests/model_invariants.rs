use psmc_rs::PsmcModel;
use psmc_rs::model::n_steps_for_pattern;

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
        3,
        1e-3,
        2e-4,
        2.5e-8,
        Some("1*5".to_string()), // implies n_steps = 4 (or legacy state-count=5)
    )
    .expect_err("expected pattern/n_steps mismatch error");
    assert!(err.to_string().contains("pattern implies n_steps=4"));
}

#[test]
fn map_lam_expands_pattern_groups() {
    let model = PsmcModel::new(15.0, 3, 1e-3, 2e-4, 2.5e-8, Some("2*2".to_string()))
        .expect("failed to create model");

    let expanded = model
        .map_lam(&[1.0, 2.0])
        .expect("failed to map grouped lam");
    assert_eq!(expanded, vec![1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn model_new_accepts_legacy_state_count_with_pattern() {
    let model = PsmcModel::new(
        15.0,
        64, // legacy style: state-count instead of n_steps
        1e-3,
        2e-4,
        2.5e-8,
        Some("4+25*2+4+6".to_string()),
    )
    .expect("failed to create model");
    assert_eq!(model.n_steps, 63);
    assert_eq!(model.lam.len(), 28);
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

#[test]
fn n_steps_for_pattern_counts_atomic_intervals() {
    assert_eq!(n_steps_for_pattern("4+25*2+4+6").unwrap(), 63);
    assert_eq!(n_steps_for_pattern("4+5*3+4").unwrap(), 22);
    assert_eq!(n_steps_for_pattern("2*2").unwrap(), 3);
    assert_eq!(n_steps_for_pattern("1+1").unwrap(), 1);
    for bad in ["", "1", "4+x", "0*3", "3*0", "4+-1"] {
        assert!(
            n_steps_for_pattern(bad).is_err(),
            "{bad:?} should be rejected"
        );
    }
    // The value it returns is exactly what PsmcModel::new accepts.
    let pattern = "4+25*2+4+6";
    let n = n_steps_for_pattern(pattern).unwrap();
    let model = PsmcModel::new(15.0, n, 1e-3, 2e-4, 2.5e-8, Some(pattern.to_string())).unwrap();
    assert_eq!(model.n_steps, n);
}

#[test]
fn model_new_rejects_single_state_models() {
    // n_steps = 0 leaves one state whose time grid starts at t_max instead of
    // 0 (compute_t writes t[n] = t_max over t[0]); there is nothing to infer.
    assert!(PsmcModel::new(15.0, 0, 1e-3, 2e-4, 2.5e-8, None).is_err());
    assert!(PsmcModel::new(15.0, 0, 1e-3, 2e-4, 2.5e-8, Some("1".to_string())).is_err());
    assert!(PsmcModel::new(15.0, 1, 1e-3, 2e-4, 2.5e-8, Some("1".to_string())).is_err());
}

#[test]
fn legacy_state_count_shim_sizes_matrices_by_effective_n_steps() {
    let model = PsmcModel::new(15.0, 23, 1e-3, 2e-4, 2.5e-8, Some("4+5*3+4".to_string())).unwrap();
    assert_eq!(model.n_steps, 22);
    assert_eq!(model.p_kl.dim(), (23, 23));
    assert_eq!(model.em.dim(), (3, 23));
    assert_eq!(model.sigma.len(), 23);
    assert_eq!(model.t.len(), 24);
}
