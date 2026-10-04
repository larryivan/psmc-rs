use psmc_rs::PsmcModel;
use psmc_rs::model::{MAX_STATES, T_INF, n_steps_for_pattern};

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
fn patterns_whose_state_count_overflows_are_errors_not_panics() {
    let max = usize::MAX;
    for bad in [format!("{max}*2"), format!("{max}+1"), format!("3*{max}+4")] {
        assert!(
            n_steps_for_pattern(&bad).is_err(),
            "{bad:?} should be rejected"
        );
        assert!(PsmcModel::new(15.0, 63, 1e-3, 2e-4, 2.5e-8, Some(bad)).is_err());
    }
}

#[test]
fn patterns_with_too_many_states_are_errors_not_panics() {
    // State counts that fit in usize but not in a dense transition matrix: a
    // typo (extra zeros) and usize::MAX. new used to panic allocating p_kl
    // with the n_steps that n_steps_for_pattern returned for them.
    for (bad, implied_n_steps) in [
        ("4+25*2+4+6000000000", 6_000_000_057),
        ("18446744073709551615", usize::MAX - 1),
    ] {
        let model = PsmcModel::new(
            15.0,
            implied_n_steps,
            1e-3,
            2e-4,
            2.5e-8,
            Some(bad.to_string()),
        );
        assert!(model.is_err(), "{bad:?} should be rejected");
        let n_steps = n_steps_for_pattern(bad);
        assert!(
            n_steps.is_err(),
            "{bad:?} should be rejected, got {n_steps:?}"
        );
    }
}

#[test]
fn state_count_is_limited_to_max_states() {
    assert_eq!(
        n_steps_for_pattern(&MAX_STATES.to_string()).unwrap(),
        MAX_STATES - 1
    );
    let over = (MAX_STATES + 1).to_string();
    assert!(n_steps_for_pattern(&over).is_err());
    assert!(PsmcModel::new(15.0, MAX_STATES, 1e-3, 2e-4, 2.5e-8, Some(over)).is_err());
    // Without a pattern, n_steps = MAX_STATES means MAX_STATES + 1 states.
    for n_steps in [MAX_STATES, usize::MAX] {
        assert!(
            PsmcModel::new(15.0, n_steps, 1e-3, 2e-4, 2.5e-8, None).is_err(),
            "n_steps = {n_steps} should be rejected"
        );
    }
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

#[test]
fn time_grid_ends_at_c_infinity_and_t_max_must_stay_below_it() {
    let model = PsmcModel::new(15.0, 6, 1e-3, 2e-4, 2.5e-8, None).unwrap();
    assert_eq!(model.t[6], 15.0);
    assert_eq!(model.t[7], T_INF);
    assert_eq!(T_INF, 1000.0); // C psmc.h: #define PSMC_T_INF 1000.0

    for bad in [0.0, -1.0, T_INF, 5000.0, f64::NAN, f64::INFINITY] {
        assert!(
            PsmcModel::new(bad, 6, 1e-3, 2e-4, 2.5e-8, None).is_err(),
            "t_max = {bad} should be rejected"
        );
    }
    let mut model = model;
    model.t_max = 2000.0;
    assert!(model.param_recalculate().is_err());
}
