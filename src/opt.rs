//! The M-step: minimise minus C psmc's EM Q (plus an optional smoothing
//! penalty) over θ, ρ, t_max and the free λ.
//!
//! Each parameter is searched in an unconstrained coordinate, the logit of
//! its position between its bounds ([`bounds_from_config`]). The objective
//! and its gradient are computed together, the gradient by reverse-mode
//! (adjoint) differentiation of C's `psmc_update_hmm` and Q, written out by
//! hand: one evaluation costs O(states²) time and O(states) memory besides
//! the expected counts, whatever the number of free λ. L-BFGS minimises the
//! objective, with an initial matrix from its curvature in each coordinate
//! (estimated from the gradient at the start of each M-step), until a
//! scale-free convergence test holds ([`MStepStop`]). The search runs to
//! convergence, as C psmc's Hooke-Jeeves search does;
//! [`MStepConfig::max_iters`] only caps it.

use anyhow::{Result, bail, ensure};

use crate::hmm::SufficientStats;
use crate::model::PsmcModel;

pub use crate::em::{EmHistory, EmProgressEvent, em_train, em_train_with_progress};
use crate::progress;
use crate::utils::{logit, sigmoid};

#[cfg(test)]
mod dual;
mod lbfgs;
mod objective;

use objective::{Objective, ROW_SUM_TOLERANCE};

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone)]
pub struct MStepConfig {
    /// The most L-BFGS iterations an M-step runs (default
    /// [`DEFAULT_MSTEP_ITERS`]): a safety cap, far above the 10 to 20 that
    /// converge an M-step with C psmc's patterns (up to about 70 with one λ
    /// per interval).
    pub max_iters: usize,
    /// Curvature pairs the L-BFGS memory keeps (default 20).
    pub lbfgs_m: usize,
    /// Weight of the smoothing penalty
    /// `lambda * sum_k (ln lambda_{k+1} - ln lambda_k)^2` (over all states)
    /// that the M-step adds to minus C psmc's Q. The default, 0, leaves C
    /// psmc's objective, as the CLI's `--smooth-lambda` does by default
    /// (0.1 used 1e-3). The penalty does not grow with the number of bins,
    /// so a given weight matters less on larger inputs.
    pub lambda: f64,
    /// The Armijo constant of the line search.
    pub line_search_c1: f64,
    /// Trial points of a line search before it fails.
    pub max_ls_steps: usize,
    /// The convergence threshold, relative to the objective: the search has
    /// converged ([`MStepStop::Converged`]) when, in each of two consecutive
    /// iterations, the objective decreased by at most `rel_tol` times its
    /// size and the next quasi-Newton step predicts no larger a decrease.
    /// The default, 1e-13, is about ten times the objective's rounding.
    /// Both quantities grow with the number of bins as the objective does,
    /// so the test does not depend on the input's size.
    pub rel_tol: f64,
    pub progress: bool,
    pub theta_lo: f64,
    pub theta_hi: f64,
    pub rho_lo: f64,
    pub rho_hi: f64,
    pub tmax_lo: f64,
    pub tmax_hi: f64,
    /// The range of each free λ, by default 1e-6 to 1e4.
    ///
    /// The upper bound is where C psmc's formulas still resolve an early
    /// interval in f64. Over an interval of length τ, the survival falls by
    /// `1 - exp(-τ/λ)`, which f64 holds to about 1.1e-16 absolute, and the
    /// interval's terms `λ (alpha_k - alpha_{k+1}) - alpha_{k+1} τ` of C's
    /// `cpik` and `q_aux` cancel to about `τ² / (2 λ)`: their relative error
    /// is about `2.2e-16 (λ / τ)²`. With 64 intervals and `t_max` 15 the
    /// first lasts τ = 0.0083, where that error is 3e-4 at λ = 1e4 but 3 at
    /// 1e6: the probabilities of moving into that state are then rounding
    /// noise, negative at most points however close to a valid one, so
    /// that the M-step's objective is infeasible all around such a start
    /// and EM stopped moving (with 101 intervals, from round 27 of a
    /// simulated 4 × 25 Mb genome). 1e4 keeps the error below 1 up to about
    /// 1,600 intervals. A λ of 1e4 already means no coalescence in its
    /// intervals (Ne 1e4 times N0); C psmc's λ have no bound.
    pub lam_lo: f64,
    pub lam_hi: f64,
}

/// The default [`MStepConfig::max_iters`] (and `--mstep-iters`).
pub const DEFAULT_MSTEP_ITERS: usize = 1000;

impl Default for MStepConfig {
    fn default() -> Self {
        Self {
            max_iters: DEFAULT_MSTEP_ITERS,
            lbfgs_m: 20,
            lambda: 0.0,
            line_search_c1: 1e-4,
            max_ls_steps: 20,
            rel_tol: 1e-13,
            progress: true,
            theta_lo: 1e-12,
            theta_hi: 10.0,
            rho_lo: 1e-12,
            rho_hi: 10.0,
            tmax_lo: 1.0,
            tmax_hi: 200.0,
            lam_lo: 1e-6,
            lam_hi: 1e4,
        }
    }
}

/// Progress of an M-step: the L-BFGS iterations `done`, out of `total`.
///
/// While the search runs, `total` is the cap [`MStepConfig::max_iters`],
/// which converged M-steps end far below; the last event of each M-step
/// has `total` equal to `done`, the iterations it took, so that a bar
/// drawn from the events ends full.
#[derive(Debug, Clone, Copy)]
pub struct MStepProgress {
    pub done: usize,
    pub total: usize,
}

/// The bounds of the M-step's parameters, in their order: θ, ρ, t_max,
/// then one per free λ of `model`.
pub fn bounds_from_config(model: &PsmcModel, config: &MStepConfig) -> Vec<Bounds> {
    let bounds = |lo, hi| Bounds { lo, hi };
    [
        bounds(config.theta_lo, config.theta_hi),
        bounds(config.rho_lo, config.rho_hi),
        bounds(config.tmax_lo, config.tmax_hi),
    ]
    .into_iter()
    .chain(std::iter::repeat_n(
        bounds(config.lam_lo, config.lam_hi),
        model.lam.len(),
    ))
    .collect()
}

/// The M-step's parameters of `model`: θ, ρ, t_max, then the free λ.
pub(crate) fn pack_params(model: &PsmcModel) -> Vec<f64> {
    [model.theta, model.rho, model.t_max]
        .into_iter()
        .chain(model.lam.iter().copied())
        .collect()
}

/// Sets `model`'s parameters to `params` ([`pack_params`]), without
/// recomputing its HMM.
pub(crate) fn unpack_params(model: &mut PsmcModel, params: &[f64]) -> Result<()> {
    ensure!(params.len() == 3 + model.lam.len(), "param length mismatch");
    model.theta = params[0];
    model.rho = params[1];
    model.t_max = params[2];
    model.lam.copy_from_slice(&params[3..]);
    Ok(())
}

/// The margin, as a share of each parameter's range, within which the
/// M-step keeps parameters from their bounds: the logit coordinates stay
/// within ±`logit(1 - BOUND_MARGIN)` (27.6).
const BOUND_MARGIN: f64 = 1e-12;

fn to_unconstrained(params: &[f64], bounds: &[Bounds]) -> Result<Vec<f64>> {
    ensure!(params.len() == bounds.len(), "bounds length mismatch");
    Ok(params
        .iter()
        .zip(bounds)
        .map(|(x, b)| logit(((x - b.lo) / (b.hi - b.lo)).clamp(BOUND_MARGIN, 1.0 - BOUND_MARGIN)))
        .collect())
}

fn from_unconstrained(p: &[f64], bounds: &[Bounds]) -> Result<Vec<f64>> {
    ensure!(p.len() == bounds.len(), "bounds length mismatch");
    Ok(p.iter()
        .zip(bounds)
        .map(|(&v, b)| b.lo + (b.hi - b.lo) * sigmoid(v))
        .collect())
}

pub fn m_step_lbfgs(
    model: &PsmcModel,
    stats: &SufficientStats,
    init_params: &[f64],
    bounds: &[Bounds],
    config: &MStepConfig,
) -> Result<Vec<f64>> {
    m_step_lbfgs_with_progress(model, stats, init_params, bounds, config, None)
}

pub fn m_step_lbfgs_with_progress(
    model: &PsmcModel,
    stats: &SufficientStats,
    init_params: &[f64],
    bounds: &[Bounds],
    config: &MStepConfig,
    on_progress: Option<&mut dyn FnMut(MStepProgress)>,
) -> Result<Vec<f64>> {
    m_step_lbfgs_with_outcome(model, stats, init_params, bounds, config, on_progress)
        .map(|outcome| outcome.params)
}

/// Why [`m_step_lbfgs_with_outcome`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MStepStop {
    /// The search converged: in each of two consecutive iterations the
    /// objective decreased by at most [`MStepConfig::rel_tol`] times its
    /// size, and the next quasi-Newton step predicted no more; nor did,
    /// from fresh curvatures, moving alone a parameter along which the
    /// objective does not curve upwards, or a diagonal Newton step.
    Converged,
    /// No trial point of the line search lowered the objective, along the
    /// L-BFGS direction nor then along the preconditioned gradient, and the
    /// check with fresh curvatures promised no more than `rel_tol` (or
    /// three such restarts ran). Near the optimum this means the objective
    /// cannot be lowered at its precision: it also ends converged searches
    /// whose last decreases were not yet within `rel_tol`.
    LineSearchFailed,
    /// [`MStepConfig::max_iters`] iterations ran before either.
    MaxIters,
}

/// The result of [`m_step_lbfgs_with_outcome`].
#[derive(Debug, Clone)]
pub struct MStepOutcome {
    /// The new parameters, in the order of `init_params`: θ, ρ, t_max, then
    /// the free λ.
    pub params: Vec<f64>,
    /// Evaluations of the objective (with its gradient): the one at
    /// `init_params`, one per parameter for the curvatures that scale the
    /// search, those of the line searches, and those of each check before
    /// the search stops (one per parameter, and the moves of flat ones).
    /// C psmc prints its M-step's count of objective evaluations on the `IT`
    /// line of its `.psmc` output.
    pub evaluations: usize,
    /// L-BFGS iterations, each of which moved the parameters to a lower
    /// objective (at most [`MStepConfig::max_iters`]; 0 when the result is
    /// `init_params`).
    pub iterations: usize,
    /// Why the search stopped.
    pub stop: MStepStop,
}

/// [`m_step_lbfgs_with_progress`], also returning how many times the
/// objective was evaluated, how many iterations ran and why the search
/// stopped. See the [module documentation](self) for the search.
pub fn m_step_lbfgs_with_outcome(
    model: &PsmcModel,
    stats: &SufficientStats,
    init_params: &[f64],
    bounds: &[Bounds],
    config: &MStepConfig,
    mut on_progress: Option<&mut dyn FnMut(MStepProgress)>,
) -> Result<MStepOutcome> {
    let mut objective = Objective::new(model, stats, bounds, config.lambda)?;
    let x0 = to_unconstrained(init_params, bounds)?;
    // The search moves only to points whose rows sum to 1 within
    // ROW_SUM_TOLERANCE, or within the start's own deviation when that is
    // larger (as from parameters that EM did not fit), so that it never
    // makes the HMM less valid than it was.
    objective.set_row_tolerance(f64::INFINITY);
    let (f0, g0) = objective.value_and_gradient(&x0);
    let start_deviation = objective.row_deviation();
    objective.set_row_tolerance(ROW_SUM_TOLERANCE.max(start_deviation));
    let debug = std::env::var("PSMC_DEBUG_MSTEP").is_ok_and(|v| v == "1");
    if debug {
        eprintln!("M-step start: f={f0:.17e}, rows sum to 1 within {start_deviation:.2e}");
    }
    if !f0.is_finite() {
        // No trial point can compare as lower, so the search would spend
        // its whole budget and return the start as if it were a result.
        bail!(
            "the M-step objective is {f0} at its starting parameters, so it cannot be \
             minimised (its transition or emission probabilities are negative or not \
             numbers; see PsmcModel::check_transitions)"
        );
    }

    let pb = config
        .progress
        .then(|| progress::bar(config.max_iters as u64, "M", "L-BFGS"));
    let total = config.max_iters;
    if let Some(cb) = on_progress.as_mut() {
        cb(MStepProgress { done: 0, total });
    }
    let mut on_iteration = |done: usize| {
        if let Some(pb) = &pb {
            pb.set_position(done as u64);
        }
        if let Some(cb) = on_progress.as_mut() {
            cb(MStepProgress { done, total });
        }
    };
    let settings = lbfgs::Settings {
        max_iters: config.max_iters,
        memory: config.lbfgs_m,
        c1: config.line_search_c1,
        max_ls_steps: config.max_ls_steps,
        rel_tol: config.rel_tol,
        x_limit: logit(1.0 - BOUND_MARGIN),
        debug,
    };
    let minimum = lbfgs::minimize(&mut objective, x0, f0, g0, &settings, &mut on_iteration);
    // The cap is no estimate of the search's length: the bar and the last
    // event end at the iterations it took.
    let done = minimum.iterations;
    if let Some(pb) = pb {
        pb.set_length(done as u64);
        pb.set_position(done as u64);
        pb.finish_with_message("M-step done");
    }
    if let Some(cb) = on_progress.as_mut() {
        cb(MStepProgress { done, total: done });
    }

    // Without an iteration the result is the start itself, not its round
    // trip through the unconstrained coordinates (which can move a
    // parameter clamped near its bound).
    let params = if minimum.iterations == 0 {
        init_params.to_vec()
    } else {
        from_unconstrained(&minimum.x, bounds)?
    };
    Ok(MStepOutcome {
        params,
        evaluations: objective.evaluations,
        iterations: minimum.iterations,
        stop: minimum.stop,
    })
}

#[cfg(test)]
mod tests {
    use super::dual::{Dd, Dual, Real, compute_params_dual, eval_dual_in, expand_lam_dual};
    use super::objective::{EVAL_CALLS, LARGEST_COORDINATE};
    use super::*;
    use crate::hmm::e_step_streaming;
    use crate::model::T_INF;

    /// The objective and its adjoint gradient at `x`.
    fn eval_adjoint(
        model: &PsmcModel,
        stats: &SufficientStats,
        x: &[f64],
        bounds: &[Bounds],
        lambda: f64,
    ) -> (f64, Vec<f64>) {
        Objective::new(model, stats, bounds, lambda)
            .unwrap()
            .value_and_gradient(x)
    }

    /// Deterministic xorshift64 in [0, 1).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
            (lo.ln() + (hi.ln() - lo.ln()) * self.next()).exp()
        }
    }

    /// The model of `n_steps` and `pattern` with θ, ρ and t_max as given and
    /// its free λ drawn by `lam`, in order; `param_recalculate` has run.
    fn model_with(
        n_steps: usize,
        pattern: Option<&str>,
        [theta, rho, t_max]: [f64; 3],
        lam: impl FnMut() -> f64,
    ) -> PsmcModel {
        let pattern = pattern.map(str::to_string);
        let mut model = PsmcModel::new(t_max, n_steps, theta, rho, 2.5e-8, pattern).unwrap();
        model.lam.fill_with(lam);
        model.param_recalculate().unwrap();
        model
    }

    /// Sets the last λ of `model` to a draw from 900 to 1e6, beyond
    /// `T_INF - t_max`, and recomputes its HMM.
    fn make_last_huge(rng: &mut Rng, model: &mut PsmcModel) {
        *model.lam.last_mut().unwrap() = rng.log_uniform(900.0, 1e6);
        model.param_recalculate().unwrap();
    }

    /// A model with random θ, ρ, t_max and λ; `param_recalculate` has run.
    fn random_model(rng: &mut Rng, n_steps: usize, pattern: Option<&str>) -> PsmcModel {
        let theta = rng.log_uniform(1e-4, 0.05);
        let rho = theta * rng.log_uniform(0.05, 2.0);
        let t_max = 5.0 + 30.0 * rng.next();
        model_with(n_steps, pattern, [theta, rho, t_max], || {
            rng.log_uniform(0.05, 20.0)
        })
    }

    const LAYOUTS: [(usize, Option<&str>); 3] =
        [(10, None), (22, Some("4+5*3+4")), (63, Some("4+25*2+4+6"))];

    /// A model drawn from the whole region the M-step can reach: ρ and t_max
    /// up to (near) their bounds and λ over six decades, so that the average
    /// time falls back to the interval's conditional mean both for k < n and
    /// for the last state. With `huge_last`, the last λ exceeds
    /// `T_INF - t_max`, where that fallback depends on `T_INF`.
    fn wide_random_model(
        rng: &mut Rng,
        n_steps: usize,
        pattern: Option<&str>,
        huge_last: bool,
    ) -> PsmcModel {
        let theta = rng.log_uniform(1e-5, 1.0);
        let rho = rng.log_uniform(1e-6, 5.0);
        let t_max = rng.log_uniform(1.0, 200.0);
        let mut model = model_with(n_steps, pattern, [theta, rho, t_max], || {
            rng.log_uniform(1e-3, 1e3)
        });
        if huge_last {
            make_last_huge(rng, &mut model);
        }
        model
    }

    /// States whose average time is the fallback (the interval's conditional
    /// mean coalescent time) rather than the log formula, as
    /// `compute_params_dual` (and so `param_recalculate`) decides.
    fn fallback_states(model: &PsmcModel) -> Vec<usize> {
        let value = |v: f64| Dual::<f64>::constant(v, 0);
        let lam: Vec<Dual> = model
            .map_lam(&model.lam)
            .unwrap()
            .into_iter()
            .map(value)
            .collect();
        let (theta, rho, t_max) = (value(model.theta), value(model.rho), value(model.t_max));
        compute_params_dual(model.n_steps, &theta, &rho, &t_max, &lam)
            .unwrap()
            .fallbacks
    }

    /// Asserts that `compute_params_dual`'s values are bit-identical to
    /// `param_recalculate`'s (NaN where it gives NaN).
    fn assert_dual_values_equal(model: &PsmcModel) {
        let n = model.n_steps;
        let pattern = model.pattern.as_deref();
        let params = pack_params(model);
        let n_vars = params.len();
        let vars: Vec<Dual> = (0..n_vars)
            .map(|i| Dual::variable(params[i], n_vars, i))
            .collect();
        let lam_full = expand_lam_dual(model, &vars[3..]).unwrap();
        let dual = compute_params_dual(n, &vars[0], &vars[1], &vars[2], &lam_full).unwrap();
        let (a, em) = (dual.p_kl, dual.em);
        let same = |x: f64, y: f64| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan());
        // Same operations in the same order, so equal to the bit.
        for ((k, l), v) in model.p_kl.indexed_iter() {
            assert!(
                same(a[k][l].re, *v),
                "{pattern:?} theta={} rho={} t_max={} lam={:?} p_kl[{k}][{l}]: {} vs {v}",
                model.theta,
                model.rho,
                model.t_max,
                model.lam,
                a[k][l].re
            );
        }
        for ((b, k), v) in model.em.indexed_iter() {
            assert!(
                same(em[b][k].re, *v),
                "{pattern:?} theta={} rho={} t_max={} lam={:?} em[{b}][{k}]: {} vs {v}",
                model.theta,
                model.rho,
                model.t_max,
                model.lam,
                em[b][k].re
            );
        }
    }

    #[test]
    fn dual_values_equal_param_recalculate() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut inner_fallbacks, mut last_fallbacks) = (0, 0);
        for (n, pattern) in LAYOUTS {
            for _ in 0..10 {
                assert_dual_values_equal(&random_model(&mut rng, n, pattern));
            }
            for case in 0..60 {
                let model = wide_random_model(&mut rng, n, pattern, case % 2 == 0);
                for k in fallback_states(&model) {
                    if k < n {
                        inner_fallbacks += 1;
                    } else if model.avg_t()[n] > T_INF {
                        last_fallbacks += 1;
                    }
                }
                assert_dual_values_equal(&model);
            }
        }
        // Both fallback branches of the average time must be exercised, the
        // last one beyond T_INF (where the value of T_INF matters).
        assert!(
            inner_fallbacks >= 20 && last_fallbacks >= 20,
            "fallbacks exercised: {inner_fallbacks} for k < n, {last_fallbacks} for k = n"
        );
    }

    /// Sufficient statistics of an E-step on a random sequence drawn with
    /// roughly the model's heterozygosity and a few missing bins.
    fn random_stats(rng: &mut Rng, model: &PsmcModel, n_bins: usize) -> SufficientStats {
        let het = (1.0 - (-model.theta * 2.0).exp()).max(1e-3);
        let row: Vec<u8> = (0..n_bins)
            .map(|_| {
                let u = rng.next();
                if u < 0.02 {
                    2
                } else if u < 0.02 + het {
                    1
                } else {
                    0
                }
            })
            .collect();
        e_step_streaming(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &[row],
            &[true],
            false,
            "E",
        )
        .unwrap()
        .stats
    }

    /// Random expected counts for `states` states: positive, of the sizes an
    /// E-step on `n_bins` bins gives, with a share `zeros` of them 0.
    fn synthetic_stats(rng: &mut Rng, states: usize, n_bins: f64, zeros: f64) -> SufficientStats {
        let mut count = |scale: f64| {
            if rng.next() < zeros {
                0.0
            } else {
                scale * rng.log_uniform(1e-4, 1.0)
            }
        };
        let per_state = n_bins / states as f64;
        let mut xi = ndarray::Array2::zeros((states, states));
        for ((k, l), v) in xi.indexed_iter_mut() {
            *v = count(if k == l { per_state } else { 1e-3 * per_state });
        }
        let gobs = [
            (0..states).map(|_| count(per_state)).collect(),
            (0..states).map(|_| count(0.05 * per_state)).collect(),
            (0..states).map(|_| count(0.02 * per_state)).collect(),
        ];
        SufficientStats {
            gobs,
            xi,
            callable: None,
        }
    }

    /// Sufficient statistics of an E-step on a random sequence with callable
    /// bases (bins of 100 bases): stretches of dispersed missing data,
    /// whose bins hold 10 to 99 callable bases and show heterozygous sites
    /// in proportion, between full ones, and a few missing bins.
    fn random_counted_stats(rng: &mut Rng, model: &PsmcModel, n_bins: usize) -> SufficientStats {
        // Enough K bins for many counts among them.
        let het = (1.0 - (-model.theta * 2.0).exp()).max(0.03);
        let (mut row, mut counts) = (Vec::with_capacity(n_bins), Vec::with_capacity(n_bins));
        let mut partial = false;
        while row.len() < n_bins {
            if rng.next() < 0.01 {
                partial = !partial;
            }
            let u = rng.next();
            let c = if partial {
                10 + (rng.next() * 90.0) as u16
            } else {
                100
            };
            if u < 0.02 {
                row.push(2);
                counts.push((rng.next() * 10.0) as u16);
            } else {
                row.push(u8::from(u < 0.02 + het * f64::from(c) / 100.0));
                counts.push(c);
            }
        }
        let callable = crate::io::CallableBases::from_counts(100, vec![counts]).unwrap();
        let stats = crate::hmm::e_step_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &[row],
            &[true],
            crate::hmm::EStepOptions {
                callable: Some(&callable),
                ..crate::hmm::EStepOptions::default()
            },
        )
        .unwrap()
        .stats;
        assert!(stats.callable.as_ref().unwrap().k_by_count.len() > 10);
        stats
    }

    /// [`synthetic_stats`] with the counts of the callable-length model:
    /// `G_T` and the `K` counts of bins of 1 to 100 callable bases out of
    /// 100, at a few counts (the full one among them), with a share `zeros`
    /// of them 0.
    fn synthetic_counted_stats(
        rng: &mut Rng,
        states: usize,
        n_bins: f64,
        zeros: f64,
    ) -> SufficientStats {
        let mut stats = synthetic_stats(rng, states, n_bins, zeros);
        let per_state = n_bins / states as f64;
        let mut count = |scale: f64| {
            if rng.next() < zeros {
                0.0
            } else {
                scale * rng.log_uniform(1e-4, 1.0)
            }
        };
        let t_weighted = (0..states).map(|_| count(per_state)).collect();
        let k_by_count = [1u16, 10, 37, 64, 99, 100]
            .into_iter()
            .map(|c| (c, (0..states).map(|_| count(0.01 * per_state)).collect()))
            .collect();
        stats.callable = Some(crate::hmm::CallableStats {
            bin_size: 100,
            t_weighted,
            k_by_count,
        });
        stats
    }

    /// `stats` as an E-step with callable bases gives them when every bin
    /// is full: `G_T` is `gobs[0]` and the one count of the `K` bins is the
    /// bin size, with `gobs[1]`.
    fn as_full_counts(stats: &SufficientStats) -> SufficientStats {
        let mut full = stats.clone();
        full.callable = Some(crate::hmm::CallableStats {
            bin_size: 100,
            t_weighted: stats.gobs[0].clone(),
            k_by_count: vec![(100, stats.gobs[1].clone())],
        });
        full
    }

    /// A model of moderate λ (0.1 to 10) whose transition probabilities are
    /// valid and in which some inner state's average time falls back to its
    /// interval's mean (which takes a large ρ, 0.02 to 1 per bin); with
    /// `huge_last`, the last λ (900 to 1e6) sends the last state's to its
    /// fallback beyond `T_INF`.
    fn fallback_model(
        rng: &mut Rng,
        n_steps: usize,
        pattern: Option<&str>,
        huge_last: bool,
    ) -> PsmcModel {
        loop {
            let theta = rng.log_uniform(1e-4, 1.0);
            let rho = rng.log_uniform(0.02, 1.0);
            let t_max = rng.log_uniform(2.0, 100.0);
            let mut model = model_with(n_steps, pattern, [theta, rho, t_max], || {
                rng.log_uniform(0.1, 10.0)
            });
            if huge_last {
                make_last_huge(rng, &mut model);
            }
            if model.check_transitions().is_ok()
                && fallback_states(&model).iter().any(|&k| k < n_steps)
            {
                return model;
            }
        }
    }

    /// The unconstrained point of `model`'s parameters with each of the
    /// coordinates `which` moved, with probability 1/2, to within 1e-6 to
    /// 1e-13 of a bound (of its range's width).
    fn near_bounds_point(
        rng: &mut Rng,
        model: &PsmcModel,
        bounds: &[Bounds],
        which: &[usize],
    ) -> Vec<f64> {
        let mut x = to_unconstrained(&pack_params(model), bounds).unwrap();
        for &i in which {
            if rng.next() < 0.5 {
                let depth = 14.0 + 16.0 * rng.next();
                x[i] = if rng.next() < 0.5 { -depth } else { depth };
            }
        }
        x
    }

    /// What [`compare_with_forward_mode`] saw.
    #[derive(Default, Debug)]
    struct Compared {
        /// Points compared, and those where the objective is infeasible
        /// (`+inf`: negative transition probabilities, or rows that do not
        /// sum to 1), and those of these where forward mode's objective
        /// is finite (rows that do not sum to 1).
        points: usize,
        infeasible: usize,
        infeasible_dual_finite: usize,
        /// Points where every gradient component agreed with forward mode
        /// to 1e-10, and the largest relative difference among them.
        agreed: usize,
        worst_agreed: f64,
        /// Points where they differed by more, which the double-double
        /// evaluation decided, and those of these where the forward-mode
        /// gradient is not finite (overflow in its derivatives) although
        /// the objective is.
        decided: usize,
        dual_not_finite: usize,
        /// Points the double-double evaluation could not decide: exact
        /// arithmetic sends other states' average times to the fallback.
        other_branches: usize,
        /// The adjoint's error over forward mode's at each decided point.
        error_ratios: Vec<f64>,
        /// Points whose survival to t_max is below 1e-140, where forward
        /// mode's squared denominators underflow (in double-double too, whose
        /// exponent range is that of f64): its gradient is no derivative
        /// there, and the gradients were not compared.
        underflow: usize,
        /// States of the compared points whose average time is the
        /// fallback, for k < n and for the last state.
        inner_fallbacks: usize,
        last_fallbacks: usize,
    }

    /// `-ln alpha[n]`, the cumulative coalescence rate up to t_max, of the
    /// parameters at the unconstrained point `x`.
    fn log_survival_to_t_max(model: &PsmcModel, x: &[f64], bounds: &[Bounds]) -> f64 {
        let mut at_x = model.clone();
        unpack_params(&mut at_x, &from_unconstrained(x, bounds).unwrap()).unwrap();
        let t = at_x.compute_t(crate::model::GRID_ALPHA);
        let lam = at_x.map_lam(&at_x.lam).unwrap();
        (0..at_x.n_steps).map(|k| (t[k + 1] - t[k]) / lam[k]).sum()
    }

    /// The largest relative error of `g` against `truth` over its
    /// components, with a floor of 1e-3 of the largest component of `truth`
    /// (infinite when `g` has a component that is not finite).
    fn worst_relative_error(g: &[f64], truth: &[f64]) -> f64 {
        let floor = 1e-3 * truth.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        g.iter()
            .zip(truth)
            .map(|(a, b)| {
                if a.is_finite() {
                    (a - b).abs() / b.abs().max(floor)
                } else {
                    f64::INFINITY
                }
            })
            .fold(0.0, f64::max)
    }

    /// Whether the HMM of the parameters at the unconstrained point `x`, as
    /// `param_recalculate` computes it, has its probabilities in [0, 1],
    /// and the largest deviation of a row of transition probabilities from
    /// summing to 1 (summed in order).
    fn hmm_validity(model: &PsmcModel, x: &[f64], bounds: &[Bounds]) -> (bool, f64) {
        let mut at_x = model.clone();
        unpack_params(&mut at_x, &from_unconstrained(x, bounds).unwrap()).unwrap();
        at_x.param_recalculate().unwrap();
        let in_unit = |p: f64| (0.0..=1.0).contains(&p);
        let mut valid = at_x.emission_matrix().iter().all(|&e| in_unit(e));
        let mut deviation = 0.0f64;
        for row in at_x.transition_matrix().rows() {
            valid &= row.iter().all(|&p| p >= 0.0);
            let sum = row.iter().fold(0.0, |acc, &p| acc + p);
            deviation = deviation.max((sum - 1.0).abs());
        }
        (valid, deviation)
    }

    /// Checks the adjoint objective against the forward-mode one at `x`.
    ///
    /// Where the HMM is valid (probabilities in [0, 1], rows summing to 1
    /// within `ROW_SUM_TOLERANCE`), the values must be equal to the bit,
    /// and the adjoint gradient finite; elsewhere the adjoint objective must
    /// be `+inf`. Every gradient component must agree with forward mode
    /// to 1e-10 relative, with a floor of 1e-3 of the largest component
    /// ([`worst_relative_error`]), unless the two gradients' own rounding
    /// errors are larger: then the forward-mode code evaluated in
    /// double-double arithmetic gives the objective's derivatives (when
    /// exact arithmetic takes the same fallback branches), and the adjoint
    /// gradient must be within 1e-10 of them or at most four times as far
    /// from them as the forward-mode gradient.
    fn compare_with_forward_mode(
        model: &PsmcModel,
        stats: &SufficientStats,
        x: &[f64],
        bounds: &[Bounds],
        lambda: f64,
        seen: &mut Compared,
    ) {
        let (f, g) = eval_adjoint(model, stats, x, bounds, lambda);
        let (f_dual, g_dual, fallbacks) =
            eval_dual_in::<f64>(model, stats, x, bounds, lambda).unwrap();
        let what = || {
            format!(
                "{:?} lambda={lambda} at {:?}",
                model.pattern,
                from_unconstrained(x, bounds).unwrap()
            )
        };
        seen.points += 1;
        let (valid, deviation) = hmm_validity(model, x, bounds);
        if !valid || deviation > ROW_SUM_TOLERANCE {
            assert_eq!(
                f,
                f64::INFINITY,
                "{}: invalid HMM (rows within {deviation:e})",
                what()
            );
            assert!(g.iter().all(|v| v.is_nan()), "{}: {g:?}", what());
            seen.infeasible += 1;
            if f_dual.is_finite() {
                seen.infeasible_dual_finite += 1;
            }
            return;
        }
        assert!(
            f.to_bits() == f_dual.to_bits(),
            "{}: value {f} vs forward mode {f_dual}",
            what()
        );
        assert!(g.iter().all(|v| v.is_finite()), "{}: {g:?}", what());
        for &k in &fallbacks {
            if k < model.n_steps {
                seen.inner_fallbacks += 1;
            } else {
                seen.last_fallbacks += 1;
            }
        }
        if log_survival_to_t_max(model, x, bounds) > 140.0 * std::f64::consts::LN_10 {
            seen.underflow += 1;
            return;
        }
        let vs_dual = worst_relative_error(&g, &g_dual);
        if vs_dual <= 1e-10 {
            seen.agreed += 1;
            seen.worst_agreed = seen.worst_agreed.max(vs_dual);
            return;
        }
        let (_, g_dd, fallbacks_dd) = eval_dual_in::<Dd>(model, stats, x, bounds, lambda).unwrap();
        if fallbacks_dd != fallbacks {
            seen.other_branches += 1;
            return;
        }
        let truth: Vec<f64> = g_dd.iter().map(|v| v.to_f64()).collect();
        let (adjoint_error, dual_error) = (
            worst_relative_error(&g, &truth),
            worst_relative_error(&g_dual, &truth),
        );
        assert!(
            adjoint_error <= 1e-10 || adjoint_error <= 4.0 * dual_error,
            "{}: the adjoint gradient {g:?} differs from forward mode's {g_dual:?} by \
             {vs_dual:.2e}, and from the derivatives {truth:?} by {adjoint_error:.2e} \
             (forward mode by {dual_error:.2e})",
            what()
        );
        seen.decided += 1;
        seen.error_ratios.push(adjoint_error / dual_error);
        if !g_dual.iter().all(|v| v.is_finite()) {
            seen.dual_not_finite += 1;
        }
    }

    /// Layouts of 11 to 301 states, with and without a pattern.
    const ORACLE_LAYOUTS: [(usize, Option<&str>); 5] = [
        (10, None),
        (22, Some("4+5*3+4")),
        (63, Some("4+25*2+4+6")),
        (100, None),
        (300, Some("1+30*10")),
    ];

    /// At parameters a fit can have (λ within two decades, ρ per bin from
    /// 1e-4), the adjoint and forward-mode objectives agree: the value to
    /// the bit and every gradient component to 1e-10 relative, except where
    /// rounding in the objective itself makes forward mode less accurate
    /// than that (at small ρ, the average times' -ln(1 - u) / ρ cancels in
    /// its derivative with respect to ρ). The points include the average
    /// time's fallback for inner and last states, θ, ρ, t_max and the last
    /// λ near their bounds, and zero counts.
    #[test]
    fn adjoint_matches_forward_mode() {
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        let config = MStepConfig::default();
        let mut seen = Compared::default();
        for (n, pattern) in ORACLE_LAYOUTS {
            for case in 0..(if n >= 100 { 3 } else { 10 }) {
                let lambda = if case % 2 == 0 { 0.0 } else { 1e-2 };
                let model = random_model(&mut rng, n, pattern);
                let bounds = bounds_from_config(&model, &config);
                // E-step counts, and synthetic ones with some zero.
                let stats = if n < 100 {
                    random_stats(&mut rng, &model, 5_000)
                } else {
                    synthetic_stats(&mut rng, n + 1, 1e6, 0.0)
                };
                let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                let stats = synthetic_stats(&mut rng, n + 1, 1e6, 0.05);
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);

                let model = fallback_model(&mut rng, n, pattern, case % 2 == 0);
                let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                // θ, ρ, t_max and the last λ near their bounds.
                let last = bounds.len() - 1;
                let x = near_bounds_point(&mut rng, &model, &bounds, &[0, 1, 2, last]);
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
            }
        }
        assert_eq!(seen.points, 4 * (3 * 10 + 2 * 3));
        // ρ near its upper bound makes transition probabilities negative.
        assert!(seen.infeasible * 4 < seen.points, "{seen:?}");
        // Most points agree to 1e-10 with forward mode; at the others the
        // adjoint gradient is the more accurate, or as accurate.
        let compared = seen.points - seen.infeasible - seen.underflow;
        assert!(seen.agreed * 4 >= compared * 3, "{seen:?}");
        assert!(seen.error_ratios.iter().all(|&r| r <= 1.1), "{seen:?}");
        assert!(
            seen.inner_fallbacks >= 100 && seen.last_fallbacks >= 20,
            "{seen:?}"
        );
    }

    /// Over the whole region the M-step can reach, λ over six decades and
    /// any parameter near its bounds, the values still agree to the bit
    /// wherever the HMM is valid. The float objective is ill-conditioned
    /// there (λ near 1e6 leaves the survival barely decaying over an
    /// interval, and C's β sums cancel): it can be wrong in its 7th digit
    /// and forward mode's gradient in its 3rd, so the double-double
    /// evaluation decides most points, and the adjoint gradient is mostly
    /// the more accurate one.
    #[test]
    fn adjoint_matches_forward_mode_everywhere() {
        let mut rng = Rng(0x1319_8A2E_0370_7344);
        let config = MStepConfig::default();
        let mut seen = Compared::default();
        for (n, pattern) in ORACLE_LAYOUTS.into_iter().take(3) {
            for case in 0..16 {
                let lambda = if case % 2 == 0 { 0.0 } else { 1e-2 };
                let model = loop {
                    let model = wide_random_model(&mut rng, n, pattern, case % 2 == 0);
                    if model.check_transitions().is_ok() {
                        break model;
                    }
                };
                let bounds = bounds_from_config(&model, &config);
                let stats = synthetic_stats(&mut rng, n + 1, 1e6, 0.05);
                let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                let all: Vec<usize> = (0..bounds.len()).collect();
                let model = random_model(&mut rng, n, pattern);
                let x = near_bounds_point(&mut rng, &model, &bounds, &all);
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
            }
        }
        // About half the points of this region are no HMM in f64: ρ near
        // its upper bound makes transition probabilities negative, and small
        // early λ make the survival underflow (where forward mode's value is
        // finite but its rows do not sum to 1).
        assert!(seen.points - seen.infeasible >= 30, "{seen:?}");
        assert!(seen.infeasible_dual_finite >= 1, "{seen:?}");
        assert!(seen.other_branches * 10 < seen.points, "{seen:?}");
        assert!(seen.decided >= 20, "{seen:?}");
        // Where they differ, the adjoint gradient is mostly the more
        // accurate one (or both are as far off, through the value itself),
        // and rarely, by a little, the less.
        let worse = seen.error_ratios.iter().filter(|&&r| r > 1.01).count();
        assert!(worse * 10 <= seen.decided, "{seen:?}");
        assert!(
            seen.inner_fallbacks >= 5 && seen.last_fallbacks >= 10,
            "{seen:?}"
        );
    }

    #[test]
    fn adjoint_gradient_matches_central_differences() {
        check_central_differences(Rng(0xD1B5_4A32_D192_ED03), random_stats);
    }

    /// With callable bases, the gradient matches central differences too.
    #[test]
    fn counted_gradient_matches_central_differences() {
        check_central_differences(Rng(0x8C3F_1A5E_77D2_04B9), random_counted_stats);
    }

    /// The adjoint gradient at random models against Richardson-extrapolated
    /// central differences, on the counts of an E-step that `stats` draws.
    fn check_central_differences(
        mut rng: Rng,
        stats: fn(&mut Rng, &PsmcModel, usize) -> SufficientStats,
    ) {
        let config = MStepConfig::default();
        for (n, pattern) in LAYOUTS {
            for lambda in [0.0, 1e-2] {
                for _ in 0..3 {
                    let model = random_model(&mut rng, n, pattern);
                    let stats = stats(&mut rng, &model, 20_000);
                    let bounds = bounds_from_config(&model, &config);
                    let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                    let (_, grad) = eval_adjoint(&model, &stats, &x, &bounds, lambda);
                    let f = |x: &[f64]| eval_adjoint(&model, &stats, x, &bounds, lambda).0;
                    // Richardson-extrapolated central differences: O(h^4).
                    let central = |i: usize, h: f64| {
                        let (mut xp, mut xm) = (x.clone(), x.clone());
                        xp[i] += h;
                        xm[i] -= h;
                        (f(&xp) - f(&xm)) / (2.0 * h)
                    };
                    let fd: Vec<f64> = (0..x.len())
                        .map(|i| (4.0 * central(i, 5e-4) - central(i, 1e-3)) / 3.0)
                        .collect();
                    let scale = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                    for i in 0..x.len() {
                        let rel = (grad[i] - fd[i]).abs() / grad[i].abs().max(1e-3 * scale);
                        assert!(
                            rel < 1e-6,
                            "{pattern:?} lambda={lambda} d/dx[{i}]: AD {} vs FD {} (rel {rel:.2e})",
                            grad[i],
                            fd[i]
                        );
                    }
                }
            }
        }
    }

    /// With callable bases (counted E-step statistics, and synthetic ones
    /// with zero counts and K bins of 1 to 100 callable bases), the adjoint
    /// and forward-mode objectives agree as without them: the value to the
    /// bit and the gradient to 1e-10 relative, or the double-double
    /// evaluation decides.
    #[test]
    fn adjoint_matches_forward_mode_with_callable_bases() {
        let mut rng = Rng(0x452A_8C1D_19E5_7B33);
        let config = MStepConfig::default();
        let mut seen = Compared::default();
        for (n, pattern) in ORACLE_LAYOUTS {
            for case in 0..(if n >= 100 { 2 } else { 6 }) {
                let lambda = if case % 2 == 0 { 0.0 } else { 1e-2 };
                let model = random_model(&mut rng, n, pattern);
                let bounds = bounds_from_config(&model, &config);
                let stats = if n < 100 {
                    random_counted_stats(&mut rng, &model, 5_000)
                } else {
                    synthetic_counted_stats(&mut rng, n + 1, 1e6, 0.0)
                };
                let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                let stats = synthetic_counted_stats(&mut rng, n + 1, 1e6, 0.05);
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                let model = fallback_model(&mut rng, n, pattern, case % 2 == 0);
                let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
                let last = bounds.len() - 1;
                let x = near_bounds_point(&mut rng, &model, &bounds, &[0, 1, 2, last]);
                compare_with_forward_mode(&model, &stats, &x, &bounds, lambda, &mut seen);
            }
        }
        assert_eq!(seen.points, 4 * (3 * 6 + 2 * 2));
        assert!(seen.infeasible * 4 < seen.points, "{seen:?}");
        let compared = seen.points - seen.infeasible - seen.underflow;
        assert!(seen.agreed * 4 >= compared * 3, "{seen:?}");
        assert!(seen.error_ratios.iter().all(|&r| r <= 1.1), "{seen:?}");
        assert!(seen.inner_fallbacks >= 30, "{seen:?}");
    }

    /// Statistics with callable bases whose bins are all full give C psmc's
    /// objective, gradient and M-step to the bit.
    #[test]
    fn full_callable_bases_give_c_psmc_m_step_to_the_bit() {
        let mut rng = Rng(0x7F4A_7C15_9E37_79B9);
        let config = MStepConfig {
            progress: false,
            ..MStepConfig::default()
        };
        for (n, pattern) in LAYOUTS {
            let model = random_model(&mut rng, n, pattern);
            let bounds = bounds_from_config(&model, &config);
            for stats in [
                random_stats(&mut rng, &model, 20_000),
                synthetic_stats(&mut rng, n + 1, 1e6, 0.05),
            ] {
                let full = as_full_counts(&stats);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                for lambda in [0.0, 1e-2] {
                    let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                    let (f, g) = eval_adjoint(&model, &stats, &x, &bounds, lambda);
                    let (f_full, g_full) = eval_adjoint(&model, &full, &x, &bounds, lambda);
                    assert_eq!(f.to_bits(), f_full.to_bits(), "{pattern:?}");
                    assert_eq!(bits(&g), bits(&g_full), "{pattern:?}");
                    let (f_dual, _, _) =
                        eval_dual_in::<f64>(&model, &full, &x, &bounds, lambda).unwrap();
                    assert_eq!(f.to_bits(), f_dual.to_bits(), "{pattern:?}");
                }
                let run = |s: &SufficientStats| {
                    m_step_lbfgs_with_outcome(
                        &model,
                        s,
                        &pack_params(&model),
                        &bounds,
                        &config,
                        None,
                    )
                    .unwrap()
                };
                let (plain, counted) = (run(&stats), run(&full));
                assert_eq!(bits(&plain.params), bits(&counted.params), "{pattern:?}");
                assert_eq!(plain.evaluations, counted.evaluations);
            }
        }
    }

    /// With callable bases the objective is minus the expected complete-data
    /// log-likelihood of the callable-length model, evaluated directly from
    /// the model's matrices and average times: a `T` bin of `c` of `s`
    /// callable bases emits with probability `exp(-θ (c / s) avg_t[k])`, a
    /// `K` bin with one minus that.
    #[test]
    fn counted_objective_is_minus_the_expected_log_likelihood() {
        let mut rng = Rng(0x1B87_3593_CC9E_2D51);
        let config = MStepConfig::default();
        for (n, pattern) in LAYOUTS {
            let model = random_model(&mut rng, n, pattern);
            let stats = random_counted_stats(&mut rng, &model, 20_000);
            let bounds = bounds_from_config(&model, &config);
            let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
            let (f, _) = eval_adjoint(&model, &stats, &x, &bounds, 0.0);
            let mut at_x = model.clone();
            unpack_params(&mut at_x, &from_unconstrained(&x, &bounds).unwrap()).unwrap();
            at_x.param_recalculate().unwrap();
            let avg = at_x.avg_t();
            let c = stats.callable.as_ref().unwrap();
            let mut q = 0.0;
            for (&count, &p) in stats.xi.iter().zip(at_x.transition_matrix().iter()) {
                q += count * p.ln();
            }
            for k in 0..=n {
                q += c.t_weighted[k] * (-at_x.theta * avg[k]);
                for (count, g) in &c.k_by_count {
                    let f = f64::from(*count) / 100.0;
                    q += g[k] * (1.0 - (-at_x.theta * avg[k] * f).exp()).ln();
                }
            }
            assert!(
                (f + q).abs() <= 1e-10 * q.abs(),
                "{pattern:?}: objective {f} vs -Q {}",
                -q
            );
            // And Q differs from C psmc's on the same E-step: the partly
            // called bins weigh less than full ones.
            let mut plain = stats.clone();
            plain.callable = None;
            let (f_plain, _) = eval_adjoint(&model, &plain, &x, &bounds, 0.0);
            assert!((f - f_plain).abs() > 1e-6 * f.abs(), "{pattern:?}");
        }
    }

    /// Where an early λ is so small that the survival falls below about
    /// 1e-154, the squares of `param_recalculate`'s formulas underflow: its
    /// transition probabilities are finite and not negative (so
    /// `check_transitions` accepts them) but its rows do not sum to 1, and
    /// Q can exceed Q0. The objective calls such a point infeasible
    /// (`tests/em_valid_hmm.rs` has the EM fit that reached it).
    #[test]
    fn underflowed_survival_is_infeasible() {
        let mut rng = Rng(0x6A09_E667_F3BC_C908);
        let config = MStepConfig::default();
        // The point an M-step reached on a simulated bottleneck genome:
        // the first two λ at 1e6 and the third at 2.9e-5, so that the
        // cumulative coalescence rate jumps to 370 at t_3.
        let mut model = PsmcModel::new(21.6, 63, 0.037, 0.0085, 2.5e-8, None).unwrap();
        for (k, lam) in model.lam.iter_mut().enumerate() {
            *lam = match k {
                0 | 1 => 1e6,
                2 => 2.86e-5,
                63 => 1.2e-6,
                _ => 0.5 + 3.0 * rng.next(),
            };
        }
        model.param_recalculate().unwrap();
        model.check_transitions().unwrap();
        let bounds = bounds_from_config(&model, &config);
        let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
        let (valid, deviation) = hmm_validity(&model, &x, &bounds);
        assert!(valid && deviation > 1e-3, "rows within {deviation:e}");
        // Counts of an E-step under the model with that λ at 1.
        let mut healthy = model.clone();
        healthy.lam[2] = 1.0;
        healthy.param_recalculate().unwrap();
        let stats = random_stats(&mut rng, &healthy, 20_000);
        let q = crate::em::hmm_q(model.transition_matrix(), model.emission_matrix(), &stats);
        assert!(q.is_finite());
        let (f, g) = eval_adjoint(&model, &stats, &x, &bounds, 0.0);
        assert_eq!(f, f64::INFINITY);
        assert!(g.iter().all(|v| v.is_nan()));
        // A third λ of 3e-3 keeps the survival normal: feasible.
        let mut x = x;
        x[3 + 2] = to_unconstrained(&[3e-3], &bounds[5..6]).unwrap()[0];
        let (f, g) = eval_adjoint(&model, &stats, &x, &bounds, 0.0);
        assert!(f.is_finite() && g.iter().all(|v| v.is_finite()), "{f}");
    }

    /// At the default upper bound of λ, the objective of a model whose
    /// early λ sit at that bound is feasible however `t_max` moves, for
    /// layouts of 23 to 301 intervals; at 1e6 rounding makes it infeasible
    /// at most points near such a start (see `MStepConfig::lam_hi`).
    #[test]
    fn lambda_at_its_upper_bound_keeps_the_objective_feasible() {
        /// The share of t_max perturbations (relative 1e-9 to 1e-7) at which
        /// the objective of `model`, with its first `saturated` free λ at
        /// `lam`, is not finite.
        fn infeasible_share(rng: &mut Rng, model: &PsmcModel, saturated: usize, lam: f64) -> f64 {
            let config = MStepConfig::default();
            let mut at = model.clone();
            at.lam[..saturated].fill(lam);
            at.param_recalculate().unwrap();
            let stats = random_stats(rng, model, 5_000);
            let bounds = bounds_from_config(
                &at,
                &MStepConfig {
                    lam_hi: lam.max(config.lam_hi),
                    ..config.clone()
                },
            );
            let x = to_unconstrained(&pack_params(&at), &bounds).unwrap();
            let mut bad = 0;
            for i in 1..=100 {
                let mut moved = at.clone();
                moved.t_max *= 1.0 + 1e-9 * i as f64;
                let mut x = x.clone();
                x[2] = to_unconstrained(&[moved.t_max], &bounds[2..3]).unwrap()[0];
                if !eval_adjoint(&at, &stats, &x, &bounds, 0.0).0.is_finite() {
                    bad += 1;
                }
            }
            bad as f64 / 100.0
        }
        let mut rng = Rng(0xBB67_AE85_84CA_A73B);
        let lam_hi = MStepConfig::default().lam_hi;
        for (n, pattern) in [
            (22, Some("4+5*3+4")),
            (63, None),
            (63, Some("4+25*2+4+6")),
            (100, None),
            (300, Some("1+30*10")),
        ] {
            let model = random_model(&mut rng, n, pattern);
            let share = infeasible_share(&mut rng, &model, 3, lam_hi);
            assert_eq!(share, 0.0, "{n} {pattern:?}: {share} infeasible at λ = 1e4");
        }
        for n in [63, 100] {
            let model = random_model(&mut rng, n, None);
            let share = infeasible_share(&mut rng, &model, 3, 1e6);
            assert!(share > 0.25, "{n}: only {share} infeasible at λ = 1e6");
        }
    }

    /// By default the M-step minimises minus C psmc's Q, as the CLI does: no
    /// smoothing penalty is added.
    #[test]
    fn default_objective_is_c_psmc_q() {
        let mut rng = Rng(0x5851_F42D_4C95_7F2D);
        let config = MStepConfig::default();
        for (n, pattern) in LAYOUTS {
            // Random λ, so a smoothing penalty would not vanish.
            let model = random_model(&mut rng, n, pattern);
            let stats = random_stats(&mut rng, &model, 5_000);
            let bounds = bounds_from_config(&model, &config);
            let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
            let (f, _) = eval_adjoint(&model, &stats, &x, &bounds, config.lambda);
            let mut at_x = model.clone();
            unpack_params(&mut at_x, &from_unconstrained(&x, &bounds).unwrap()).unwrap();
            at_x.param_recalculate().unwrap();
            let q = crate::em::hmm_q(at_x.transition_matrix(), at_x.emission_matrix(), &stats);
            assert!(
                (f + q).abs() <= 1e-12 * q.abs(),
                "{pattern:?}: objective {f} vs -Q {}",
                -q
            );
        }
    }

    /// Runs the M-step; returns its outcome, the number of iterations it
    /// reported as done and the number of objective evaluations it made.
    /// Its last progress event must report the iterations as done out of as
    /// many.
    fn counted_m_step(
        model: &PsmcModel,
        stats: &SufficientStats,
        config: &MStepConfig,
    ) -> (MStepOutcome, usize, usize) {
        let bounds = bounds_from_config(model, config);
        let mut last = None;
        let mut on_progress = |p: MStepProgress| last = Some((p.done, p.total));
        let calls_before = EVAL_CALLS.with(|c| c.get());
        let outcome = m_step_lbfgs_with_outcome(
            model,
            stats,
            &pack_params(model),
            &bounds,
            config,
            Some(&mut on_progress),
        )
        .unwrap();
        let calls = EVAL_CALLS.with(|c| c.get()) - calls_before;
        let (done, total) = last.unwrap();
        assert_eq!(done, total, "the last progress event");
        (outcome, done, calls)
    }

    #[test]
    fn outcome_counts_objective_evaluations() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let model = random_model(&mut rng, 22, Some("4+5*3+4"));
        let stats = random_stats(&mut rng, &model, 20_000);
        let n_params = 3 + model.lam.len();
        let base = MStepConfig {
            progress: false,
            ..MStepConfig::default()
        };

        // Without iterations only the starting point is evaluated.
        let config = MStepConfig {
            max_iters: 0,
            ..base.clone()
        };
        let (outcome, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!((outcome.evaluations, done, calls), (1, 0, 1));
        assert_eq!((outcome.iterations, outcome.stop), (0, MStepStop::MaxIters));

        // A one-trial line search whose unit step does not decrease enough
        // fails at once: the start, the curvatures and the trial count, and
        // so do the searches after each check before stopping, whose fresh
        // curvatures still predict a decrease (no coordinate is flat here,
        // so they move nothing): one more set of curvatures and one trial
        // for each of the three restarts.
        let config = MStepConfig {
            max_ls_steps: 1,
            ..base.clone()
        };
        let (outcome, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::LineSearchFailed);
        let expected = 1 + n_params + 1 + 3 * (n_params + 1);
        assert_eq!((outcome.evaluations, done, calls), (expected, 0, expected));

        // The default search: the start, one evaluation per parameter for
        // the curvatures, then at least one per iteration. The parameters
        // are those of m_step_lbfgs.
        let (outcome, done, calls) = counted_m_step(&model, &stats, &base);
        assert_eq!(outcome.evaluations, calls);
        assert_eq!(outcome.iterations, done);
        assert!(
            calls >= 1 + n_params + done,
            "{calls} evaluations, {done} iterations"
        );
        let bounds = bounds_from_config(&model, &base);
        let params = m_step_lbfgs(&model, &stats, &pack_params(&model), &bounds, &base).unwrap();
        assert_eq!(outcome.params, params);
    }

    /// When Q keeps rising as a parameter approaches its bound, the search
    /// takes it there but keeps its coordinate within the margin of the
    /// start's clamp: further out the logit saturates, steps move the
    /// coordinate without changing the objective, and nothing brings it
    /// back (the oldest λ of a simulated genome ran to -57).
    #[test]
    fn search_keeps_coordinates_within_the_bound_margin() {
        let mut rng = Rng(0x3C6E_F372_FE94_F82B);
        let limit = logit(1.0 - BOUND_MARGIN);
        for (n, pattern) in LAYOUTS {
            let model = random_model(&mut rng, n, pattern);
            let stats = random_stats(&mut rng, &model, 20_000);
            // An upper bound of θ below what the data say.
            let config = MStepConfig {
                progress: false,
                theta_hi: 0.5 * model.theta,
                ..MStepConfig::default()
            };
            LARGEST_COORDINATE.with(|c| c.set(0.0));
            let (outcome, _, _) = counted_m_step(&model, &stats, &config);
            let largest = LARGEST_COORDINATE.with(|c| c.get());
            assert!(
                largest <= limit,
                "{pattern:?}: coordinate {largest} beyond {limit}"
            );
            let theta = outcome.params[0];
            assert!(theta > 0.999 * config.theta_hi, "{pattern:?}: θ {theta}");
        }
    }

    #[test]
    fn outcome_says_why_the_m_step_stopped() {
        let mut rng = Rng(0x0BAD_5EED_1234_5678);
        let model = random_model(&mut rng, 22, Some("4+5*3+4"));
        let stats = random_stats(&mut rng, &model, 20_000);
        let base = MStepConfig {
            progress: false,
            ..MStepConfig::default()
        };

        // Every decrease is within an infinite tolerance: the search stops
        // after the two iterations of the test.
        let config = MStepConfig {
            rel_tol: f64::INFINITY,
            ..base.clone()
        };
        let (outcome, done, _) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::Converged);
        assert_eq!((outcome.iterations, done), (2, 2));

        // A few iterations from a random start: the cap ends the search.
        let config = MStepConfig {
            max_iters: 3,
            rel_tol: 0.0,
            ..base.clone()
        };
        let (outcome, done, _) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::MaxIters);
        assert_eq!((outcome.iterations, done), (3, 3));

        // Without a tolerance the search runs until the objective cannot be
        // lowered at its precision, and its last line searches fail.
        let config = MStepConfig {
            max_iters: 100_000,
            rel_tol: 0.0,
            ..base.clone()
        };
        let (precise, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!(precise.stop, MStepStop::LineSearchFailed);
        assert_eq!(precise.iterations, done);
        assert!(done > 3 && done < 1000, "{done} iterations");
        assert!(
            calls > done + model.lam.len() + 20,
            "{calls} evaluations, {done} iterations"
        );

        // By default the convergence test ends it, a few iterations earlier,
        // where the objective is that of the precision limit to 1e-13 and
        // the parameters within 1e-6. An objective known to 1e-13 fixes a
        // parameter along a flat direction only to about its square root,
        // 3e-7, and where in that range the search stops depends on the
        // rounding of libm's exp and ln (1.3e-7 with glibc on x86-64).
        let (outcome, done, calls) = counted_m_step(&model, &stats, &base);
        assert_eq!(outcome.stop, MStepStop::Converged);
        assert!(done <= precise.iterations && calls < precise.evaluations);
        let bounds = bounds_from_config(&model, &base);
        let objective = |params: &[f64]| {
            let x = to_unconstrained(params, &bounds).unwrap();
            eval_adjoint(&model, &stats, &x, &bounds, 0.0).0
        };
        let (f, f_precise) = (objective(&outcome.params), objective(&precise.params));
        assert!(
            (f - f_precise).abs() <= 1e-13 * f_precise.abs(),
            "{f} vs {f_precise}"
        );
        for (a, b) in outcome.params.iter().zip(&precise.params) {
            assert!((a / b).ln().abs() < 1e-6, "{a} vs {b}");
        }
    }
}
