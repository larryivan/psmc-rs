//! What `PsmcModel::new` accepts: patterns and their numbers of intervals,
//! the state count of psmc-rs 0.1, the bounds on the states and on t_max,
//! and the probabilities of the HMM it builds.

use psmc_rs::PsmcModel;
use psmc_rs::model::{MAX_STATES, T_INF, n_steps_for_pattern};

/// A model of `n_steps` intervals with t_max 15, θ 1e-3 and ρ 2e-4.
fn new(n_steps: usize, pattern: Option<&str>) -> anyhow::Result<PsmcModel> {
    PsmcModel::new(
        15.0,
        n_steps,
        1e-3,
        2e-4,
        2.5e-8,
        pattern.map(str::to_string),
    )
}

#[track_caller]
fn assert_near(a: f64, b: f64, eps: f64) {
    assert!((a - b).abs() <= eps, "{a} vs {b}: more than {eps} apart");
}

/// With a pattern, n_steps is the pattern's number of intervals, or psmc-rs
/// 0.1's state count (one more), which sizes the model as the intervals do.
#[test]
fn a_pattern_takes_its_intervals_or_the_legacy_state_count() {
    for (given, pattern, n_steps, n_lam) in [
        (63, "4+25*2+4+6", 63, 28),
        (64, "4+25*2+4+6", 63, 28),
        (22, "4+5*3+4", 22, 7),
        (23, "4+5*3+4", 22, 7),
    ] {
        let m = new(given, Some(pattern)).unwrap();
        assert_eq!(
            (m.n_steps, m.lam.len()),
            (n_steps, n_lam),
            "{given} {pattern}"
        );
        let n = n_steps + 1;
        assert_eq!((m.p_kl.dim(), m.em.dim()), ((n, n), (3, n)));
        assert_eq!((m.sigma.len(), m.t.len()), (n, n + 1));
    }
    // "1*5" has 4 intervals: neither 3 nor its legacy 5 states.
    let err = new(3, Some("1*5")).unwrap_err();
    assert!(
        err.to_string().contains("pattern implies n_steps=4"),
        "{err}"
    );
}

#[test]
fn map_lam_expands_pattern_groups() {
    let model = new(3, Some("2*2")).unwrap();
    assert_eq!(model.map_lam(&[1.0, 2.0]).unwrap(), [1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn transition_prior_and_emission_are_valid_probabilities() {
    let model = new(6, None).unwrap();
    let prior = model.prior_matrix();
    assert_near(prior.iter().sum(), 1.0, 1e-9);
    assert!(prior.iter().all(|&p| p >= -1e-12), "{prior:?}");

    for row in model.transition_matrix().rows() {
        assert!(row.iter().all(|&v| v.is_finite() && v >= -1e-12), "{row}");
        assert_near(row.sum(), 1.0, 1e-8);
    }

    let em = model.emission_matrix();
    for k in 0..=model.n_steps {
        let [p_t, p_k, p_n] = [0, 1, 2].map(|o| em[(o, k)]);
        assert!(p_t.is_finite() && p_k.is_finite() && p_n.is_finite());
        assert!(p_t >= 0.0 && p_k >= 0.0);
        assert_near(p_t + p_k, 1.0, 1e-9);
        assert_near(p_n, 1.0, 1e-12);
    }
}

#[test]
fn n_steps_for_pattern_counts_atomic_intervals() {
    for (pattern, n_steps) in [("4+25*2+4+6", 63), ("4+5*3+4", 22), ("2*2", 3), ("1+1", 1)] {
        assert_eq!(n_steps_for_pattern(pattern).unwrap(), n_steps, "{pattern}");
        // The value it returns is exactly what PsmcModel::new accepts.
        assert_eq!(new(n_steps, Some(pattern)).unwrap().n_steps, n_steps);
    }
    for bad in ["", "1", "4+x", "0*3", "3*0", "4+-1"] {
        assert!(n_steps_for_pattern(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn patterns_whose_state_count_overflows_are_errors_not_panics() {
    let max = usize::MAX;
    for bad in [format!("{max}*2"), format!("{max}+1"), format!("3*{max}+4")] {
        assert!(n_steps_for_pattern(&bad).is_err(), "{bad:?}");
        assert!(new(63, Some(&bad)).is_err(), "{bad:?}");
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
        assert!(new(implied_n_steps, Some(bad)).is_err(), "{bad:?}");
        let n_steps = n_steps_for_pattern(bad);
        assert!(n_steps.is_err(), "{bad:?}: {n_steps:?}");
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
    assert!(new(MAX_STATES, Some(&over)).is_err());
    // Without a pattern, n_steps = MAX_STATES means MAX_STATES + 1 states.
    for n_steps in [MAX_STATES, usize::MAX] {
        assert!(new(n_steps, None).is_err(), "n_steps = {n_steps}");
    }
}

#[test]
fn model_new_rejects_single_state_models() {
    // n_steps = 0 leaves one state whose time grid starts at t_max instead of
    // 0 (compute_t writes t[n] = t_max over t[0]); there is nothing to infer.
    for (n_steps, pattern) in [(0, None), (0, Some("1")), (1, Some("1"))] {
        assert!(new(n_steps, pattern).is_err(), "{n_steps} {pattern:?}");
    }
}

#[test]
fn time_grid_ends_at_c_infinity_and_t_max_must_stay_below_it() {
    let mut model = new(6, None).unwrap();
    assert_eq!(model.t[6], 15.0);
    assert_eq!(model.t[7], T_INF);
    // C's psmc.h: #define PSMC_T_INF 1000.0
    assert_eq!(T_INF, 1000.0);

    for bad in [0.0, -1.0, T_INF, 5000.0, f64::NAN, f64::INFINITY] {
        assert!(
            PsmcModel::new(bad, 6, 1e-3, 2e-4, 2.5e-8, None).is_err(),
            "t_max = {bad}"
        );
    }
    model.t_max = 2000.0;
    assert!(model.param_recalculate().is_err());
}
