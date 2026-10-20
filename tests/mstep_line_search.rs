//! M-step behaviour that must not depend on the size of the input.

mod common;

use common::{REFERENCE_PSMCFA, c_start, e_step, fixture_obs, mstep, params};
use psmc_rs::PsmcModel;
use psmc_rs::hmm::SufficientStats;
use psmc_rs::opt::{MStepConfig, MStepStop, bounds_from_config, m_step_lbfgs_with_outcome};

/// C psmc's starting point on the 180,000-bin reference input with pattern
/// 4+25*2+4+6, and the statistics of the first E-step there.
fn first_e_step() -> (PsmcModel, SufficientStats) {
    let obs = fixture_obs(REFERENCE_PSMCFA);
    let model = c_start(&obs.rows, "4+25*2+4+6", 4.0);
    let stats = e_step(&model, &obs.rows, &obs.row_starts).stats;
    (model, stats)
}

fn scaled(stats: &SufficientStats, factor: f64) -> SufficientStats {
    SufficientStats {
        gobs: stats
            .gobs
            .each_ref()
            .map(|g| g.iter().map(|v| v * factor).collect()),
        xi: stats.xi.mapv(|v| v * factor),
        callable: None,
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
        ..mstep()
    };
    let bounds = bounds_from_config(&model, &config);
    let init = params(&model);

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
