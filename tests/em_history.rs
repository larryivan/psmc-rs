//! The per-iteration quantities EmHistory records for C-style `.psmc`
//! output: post_sigma, Q before and after the M-step, M-step evaluations.

use psmc_rs::em::{hmm_q, hmm_q0, posterior_sigma};
use psmc_rs::hmm::{SufficientStats, e_step_streaming};
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};

fn data() -> (Vec<Vec<u8>>, Vec<bool>) {
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|s| {
            (0..5000)
                .map(|i| match (i * 37 + s * 11) % 197 {
                    0..=2 => 1,
                    3 | 4 => 2,
                    _ => 0,
                })
                .collect()
        })
        .collect();
    (rows, vec![true, false, true])
}

fn initial_model(rows: &[Vec<u8>]) -> PsmcModel {
    let theta0 = initial_theta(rows).unwrap();
    PsmcModel::new(
        15.0,
        22,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some("4+5*3+4".into()),
    )
    .unwrap()
}

/// The template model with parameters `params` (θ, ρ, t_max, λ...).
fn model_at(template: &PsmcModel, params: &[f64]) -> PsmcModel {
    let mut m = template.clone();
    m.theta = params[0];
    m.rho = params[1];
    m.t_max = params[2];
    m.lam.copy_from_slice(&params[3..]);
    m.param_recalculate().unwrap();
    m
}

fn e_step(model: &PsmcModel, rows: &[Vec<u8>], starts: &[bool]) -> SufficientStats {
    e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        rows,
        starts,
        false,
        "E",
    )
    .unwrap()
    .stats
}

fn check(lambda: f64) {
    let (rows, starts) = data();
    let template = initial_model(&rows);
    let mut model = template.clone();
    let config = MStepConfig {
        progress: false,
        max_iters: 30,
        lambda,
        ..MStepConfig::default()
    };
    let n_iter = 4;
    let history = em_train(&mut model, &rows, &starts, n_iter, &config).unwrap();
    assert_eq!(history.post_sigma.len(), n_iter);
    assert_eq!(history.q.len(), n_iter);
    assert_eq!(history.mstep_evaluations.len(), n_iter);

    for i in 0..n_iter {
        // Recompute iteration i: its E-step runs at params[i], its M-step
        // leaves params[i + 1].
        let entering = model_at(&template, &history.params[i]);
        let leaving = model_at(&template, &history.params[i + 1]);
        let stats = e_step(&entering, &rows, &starts);
        let q0 = hmm_q0(&stats);
        let q_before = hmm_q(
            entering.transition_matrix(),
            entering.emission_matrix(),
            &stats,
        ) - q0;
        let q_after = hmm_q(
            leaving.transition_matrix(),
            leaving.emission_matrix(),
            &stats,
        ) - q0;
        assert_eq!(
            history.post_sigma[i],
            posterior_sigma(&stats),
            "iteration {i}"
        );
        assert_eq!(history.q[i], (q_before, q_after), "iteration {i}");

        let (before, after) = history.q[i];
        assert!(
            before < 0.0 && after < 0.0,
            "iteration {i}: {:?}",
            history.q[i]
        );
        if lambda == 0.0 {
            // Without the smoothing penalty the M-step maximises Q itself.
            assert!(after > before, "iteration {i}: {:?}", history.q[i]);
        }
        let total: f64 = history.post_sigma[i].iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "iteration {i}: {total}");
        assert!(history.post_sigma[i].iter().all(|&p| p > 0.0));
        assert!(history.mstep_evaluations[i] > 1, "iteration {i}");
    }
}

#[test]
fn history_records_c_round_quantities() {
    check(0.0);
}

#[test]
fn q_excludes_the_smoothing_penalty() {
    // With the penalty the recorded Q is still C's unpenalised Q, which is
    // what the recomputation in `check` evaluates.
    check(1e-2);
}

#[test]
fn zero_iterations_record_nothing() {
    let (rows, starts) = data();
    let mut model = initial_model(&rows);
    let config = MStepConfig {
        progress: false,
        ..MStepConfig::default()
    };
    let history = em_train(&mut model, &rows, &starts, 0, &config).unwrap();
    assert!(history.post_sigma.is_empty());
    assert!(history.q.is_empty());
    assert!(history.mstep_evaluations.is_empty());
}
