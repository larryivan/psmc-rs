//! An M-step must end at a maximum of Q along every parameter: moving any
//! one parameter alone must not raise Q.
//!
//! On a simulated 2 Mb bottleneck genome fitted with one λ per interval,
//! the first M-step once stopped with the first λ at 9,859 while Q peaks
//! at 49 along it (7.6e-4 higher): the λ, which the data barely constrain,
//! had been carried past its optimum to where Q falls slowly towards the
//! bound, and the L-BFGS memory predicted no further decrease there.

mod common;

use common::{c_start_free, e_step, fixture_obs, model_at, mstep, params};
use psmc_rs::PsmcModel;
use psmc_rs::hmm::SufficientStats;
use psmc_rs::opt::{bounds_from_config, m_step};

/// C psmc's Q at `params` (θ, ρ, t_max, λ...) of `template`'s layout, from
/// the sums of the expected transition counts: in row `k` of the transition
/// matrix `a` (`n + 1` states), the probabilities below the diagonal are
/// `a[k][0] a[n][l] / a[n][0]` and those above it `a[k][n] a[0][l] / a[0][n]`
/// (a factor of the row times one of the column); -inf where a probability
/// is not positive or the transition matrix is not stochastic.
fn q_at(template: &PsmcModel, stats: &SufficientStats, params: &[f64]) -> f64 {
    let m = model_at(template, params);
    let (a, e) = (m.transition_matrix(), m.emission_matrix());
    let rows_ok = a
        .rows()
        .into_iter()
        .all(|row| (row.iter().fold(0.0, |a, &p| a + p) - 1.0).abs() <= 1e-9);
    let positive = a
        .iter()
        .chain(e.rows().into_iter().take(2).flatten())
        .all(|&p| p > 0.0);
    if !(rows_ok && positive) {
        return f64::NEG_INFINITY;
    }
    let (t, n, ln) = (&stats.transitions, a.nrows() - 1, f64::ln);
    let mut q = 0.0;
    for k in 0..=n {
        if k > 0 {
            q += t.lower_rows[k] * (ln(a[(k, 0)]) - ln(a[(n, 0)]));
            q += t.upper_cols[k] * ln(a[(0, k)]);
        }
        if k < n {
            q += t.upper_rows[k] * (ln(a[(k, n)]) - ln(a[(0, n)]));
            q += t.lower_cols[k] * ln(a[(n, k)]);
        }
        q += t.diagonal[k] * ln(a[(k, k)]);
        q += stats.gobs[0][k] * ln(e[(0, k)]) + stats.gobs[1][k] * ln(e[(1, k)]);
    }
    q
}

#[test]
fn first_m_step_ends_at_a_maximum_along_every_parameter() {
    let obs = fixture_obs("sim_bottleneck/bottleneck_2mb.psmcfa.gz");
    let model = c_start_free(&obs.rows, 63, 5.0);
    let stats = e_step(&model, &obs.rows, &obs.row_starts).stats;
    let config = mstep();
    let bounds = bounds_from_config(&model, &config);
    let outcome = m_step(&model, &stats, &config, None).unwrap();
    let q = q_at(&model, &stats, &outcome.params);
    let q_start = q_at(&model, &stats, &params(&model));
    assert!(q > q_start, "Q {q_start} -> {q} ({:?})", outcome.stop);

    // Each parameter alone, by factors 10^(k/20) from 1e-3 to 1e3 within
    // its bounds: Q may not rise by more than its rounding.
    for (i, b) in bounds.iter().enumerate() {
        for k in -60..=60 {
            let mut p = outcome.params.clone();
            p[i] *= 10f64.powf(k as f64 / 20.0);
            if p[i] < b.lo || p[i] > b.hi {
                continue;
            }
            let gain = q_at(&model, &stats, &p) - q;
            assert!(
                gain <= 1e-9,
                "parameter {i}: {} -> {} raises Q by {gain:e} (from {q})",
                outcome.params[i],
                p[i]
            );
        }
    }
}
