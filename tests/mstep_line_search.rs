//! M-step behaviour that must not depend on the size of the input.

use psmc_rs::hmm::{SufficientStats, e_step_streaming};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::model::n_steps_for_pattern;
use psmc_rs::opt::{MStepConfig, MStepStop, bounds_from_config, m_step_lbfgs_with_outcome};
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const PATTERN: &str = "4+25*2+4+6";

/// C psmc's starting point on the 180,000-bin reference input and the
/// statistics of the first E-step there.
fn first_e_step() -> (PsmcModel, SufficientStats) {
    let obs = read_psmcfa(
        Path::new("tests/data/c_reference/papuan_chr1_3seg.psmcfa"),
        None,
    )
    .unwrap();
    let theta0 = initial_theta(&obs.rows).unwrap();
    let model = PsmcModel::new(
        15.0,
        n_steps_for_pattern(PATTERN).unwrap(),
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some(PATTERN.to_string()),
    )
    .unwrap();
    let e = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        false,
        "E",
    )
    .unwrap();
    (model, e.stats)
}

fn scaled(stats: &SufficientStats, factor: f64) -> SufficientStats {
    SufficientStats {
        gobs: [
            stats.gobs[0].iter().map(|v| v * factor).collect(),
            stats.gobs[1].iter().map(|v| v * factor).collect(),
            stats.gobs[2].iter().map(|v| v * factor).collect(),
        ],
        xi: stats.xi.mapv(|v| v * factor),
    }
}

#[test]
fn m_step_iterates_do_not_depend_on_the_amount_of_data() {
    // Without smoothing, multiplying every expected count by 2^k multiplies
    // the objective and its gradient by exactly 2^k: neither the optimum,
    // nor the L-BFGS iterates, nor where the convergence test stops them may
    // move. 2^8 times this input is 46M bins, the size of a whole human
    // genome in 100-bp bins.
    let (model, stats) = first_e_step();
    let config = MStepConfig {
        lambda: 0.0,
        progress: false,
        ..MStepConfig::default()
    };
    let bounds = bounds_from_config(&model, &config);
    let mut init = vec![model.theta, model.rho, model.t_max];
    init.extend_from_slice(&model.lam);

    let base = m_step_lbfgs_with_outcome(&model, &stats, &init, &bounds, &config, None).unwrap();
    assert_eq!(base.stop, MStepStop::Converged);
    for k in [2, 4, 6, 8] {
        let out = m_step_lbfgs_with_outcome(
            &model,
            &scaled(&stats, f64::powi(2.0, k)),
            &init,
            &bounds,
            &config,
            None,
        )
        .unwrap();
        assert_eq!(
            (out.iterations, out.evaluations, out.stop),
            (base.iterations, base.evaluations, base.stop),
            "counts x 2^{k}"
        );
        for (i, (a, b)) in base.params.iter().zip(&out.params).enumerate() {
            assert!(
                ((a - b) / a).abs() < 1e-12,
                "counts x 2^{k}: parameter {i} is {b}, unscaled {a}"
            );
        }
    }
}
