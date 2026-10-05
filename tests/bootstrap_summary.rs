//! The bootstrap summary reads each replicate's Ne at the main curve's time
//! points as a step function: from the replicate's own time 0 on, state 0's
//! size, then each state's size from its interval's start.

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::summarize_bootstrap_models;

const BIN: f64 = 100.0;
const GEN_YEARS: f64 = 25.0;

fn model(theta: f64, t_max: f64, pattern: Option<&str>, n_steps: usize, lam: &[f64]) -> PsmcModel {
    let mut m = PsmcModel::new(
        t_max,
        n_steps,
        theta,
        theta / 4.0,
        2.5e-8,
        pattern.map(str::to_string),
    )
    .unwrap();
    m.lam = lam.to_vec();
    m.param_recalculate().unwrap();
    m
}

/// `m`'s size at `x` years, straight from its curve: the size of the last
/// state that starts at or before `x`.
fn size_at(m: &PsmcModel, x: f64) -> f64 {
    let curve = m.ne_curve(BIN, GEN_YEARS).unwrap();
    assert_eq!(curve[0][0], 0.0);
    curve.iter().rfind(|[start, _]| *start <= x).unwrap()[1]
}

/// Replicates whose curves start later and earlier than the main curve's
/// (other N0 and t_max), every state with its own λ.
fn check_layout(pattern: Option<&str>, n_steps: usize, n_lam: usize) {
    let lam = |shift: f64| -> Vec<f64> {
        (0..n_lam)
            .map(|k| 0.5 + ((k as f64 + shift) * 1.7).sin().abs() * 3.0)
            .collect()
    };
    let main = model(0.001, 15.0, pattern, n_steps, &lam(0.0));
    let replicates = [
        model(0.0013, 15.0, pattern, n_steps, &lam(1.0)),
        model(0.0011, 18.0, pattern, n_steps, &lam(2.0)),
        model(0.0009, 12.0, pattern, n_steps, &lam(3.0)),
        model(0.001, 15.0, pattern, n_steps, &lam(4.0)),
    ];
    let main_curve = main.ne_curve(BIN, GEN_YEARS).unwrap();
    let first_x = main_curve[1][0];
    // Some replicate's state 1 starts after the main curve's first point,
    // so that point lies in that replicate's state 0.
    assert!(
        replicates
            .iter()
            .any(|r| r.ne_curve(BIN, GEN_YEARS).unwrap()[1][0] > first_x)
    );
    for rep in &replicates {
        let summary =
            summarize_bootstrap_models(&main, std::slice::from_ref(rep), BIN, GEN_YEARS, 1000, 7)
                .unwrap();
        assert_eq!(summary.points.len(), n_steps);
        for (p, [x, y]) in summary.points.iter().zip(&main_curve[1..]) {
            assert_eq!((p.x_years, p.ne_main), (*x, *y));
            let want = size_at(rep, *x);
            // One replicate: every quantile is its size.
            assert_eq!(
                [p.ne_q025, p.ne_q500, p.ne_q975],
                [want; 3],
                "{pattern:?} at {x} years"
            );
        }
    }
}

#[test]
fn one_lambda_per_state() {
    check_layout(None, 12, 13);
}

#[test]
fn first_pattern_group_of_one_interval() {
    // 1 + 2*2 + 3 = 8 intervals, 4 free λ; states 0 and 1 have their own.
    check_layout(Some("1+2*2+3"), 7, 4);
}

#[test]
fn quantiles_at_the_first_point_use_state_0() {
    // Five replicates; four have their state 1 start after the main curve's
    // first point and give their state-0 size there.
    let lam = [4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let main = model(0.001, 15.0, None, 10, &lam);
    let reps: Vec<PsmcModel> = [0.0014, 0.0013, 0.0012, 0.0011, 0.0005]
        .iter()
        .map(|&theta| model(theta, 15.0, None, 10, &lam))
        .collect();
    let summary = summarize_bootstrap_models(&main, &reps, BIN, GEN_YEARS, 1000, 7).unwrap();
    let first = &summary.points[0];
    let mut want: Vec<f64> = reps.iter().map(|r| size_at(r, first.x_years)).collect();
    want.sort_by(f64::total_cmp);
    // Four replicates are still in state 0 (size 4 N0), one in state 1.
    let n0 = |r: &PsmcModel| r.n0_for_bin(BIN);
    assert_eq!(want[0], n0(&reps[4]));
    assert_eq!(first.ne_q500, want[2]);
    assert!(
        (first.ne_q500 - 4.0 * n0(&reps[2])).abs() < 1e-9 * first.ne_q500,
        "median {} vs 4 N0 = {}",
        first.ne_q500,
        4.0 * n0(&reps[2])
    );
}
