//! The M-step: L-BFGS on the EM objective, over logit-transformed
//! parameters, with the gradient of [`objective`].

use anyhow::{Result, bail};

use crate::hmm::SufficientStats;
use crate::model::PsmcModel;

pub use crate::em::{EmHistory, EmProgressEvent, em_train, em_train_with_progress};
use crate::progress;
use crate::utils::{logit, sigmoid};

#[cfg(test)]
mod dual;
mod objective;

use objective::Objective;

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone)]
pub struct MStepConfig {
    pub max_iters: usize,
    pub lbfgs_m: usize,
    /// Weight of the smoothing penalty
    /// `lambda * sum_k (ln lambda_{k+1} - ln lambda_k)^2` (over all states)
    /// that the M-step adds to minus C psmc's Q. The default, 0, leaves C
    /// psmc's objective, as the CLI's `--smooth-lambda` does by default
    /// (0.1 used 1e-3). The penalty does not grow with the number of bins,
    /// so a given weight matters less on larger inputs.
    pub lambda: f64,
    pub line_search_c1: f64,
    pub max_ls_steps: usize,
    /// Stop when the norm of the gradient of the objective (in the
    /// unconstrained coordinates) falls below this. It is absolute and the
    /// gradient grows with the number of bins. In practice the M-step ends at
    /// `max_iters` (100 by default) well before either this or the precision
    /// of the objective is reached: in the last iterations it still lowers
    /// the objective by about 1e-11 to 1e-7 relative per iteration.
    pub tol_grad: f64,
    pub progress: bool,
    pub theta_lo: f64,
    pub theta_hi: f64,
    pub rho_lo: f64,
    pub rho_hi: f64,
    pub tmax_lo: f64,
    pub tmax_hi: f64,
    pub lam_lo: f64,
    pub lam_hi: f64,
}

impl Default for MStepConfig {
    fn default() -> Self {
        Self {
            max_iters: 100,
            lbfgs_m: 7,
            lambda: 0.0,
            line_search_c1: 1e-4,
            max_ls_steps: 20,
            tol_grad: 1e-4,
            progress: true,
            theta_lo: 1e-12,
            theta_hi: 10.0,
            rho_lo: 1e-12,
            rho_hi: 10.0,
            tmax_lo: 1.0,
            tmax_hi: 200.0,
            lam_lo: 1e-6,
            lam_hi: 1e6,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MStepProgress {
    pub done: usize,
    pub total: usize,
}

pub fn bounds_from_config(model: &PsmcModel, config: &MStepConfig) -> Vec<Bounds> {
    let mut bounds = Vec::new();
    bounds.push(Bounds {
        lo: config.theta_lo,
        hi: config.theta_hi,
    }); // theta
    bounds.push(Bounds {
        lo: config.rho_lo,
        hi: config.rho_hi,
    }); // rho
    bounds.push(Bounds {
        lo: config.tmax_lo,
        hi: config.tmax_hi,
    }); // t_max
    for _ in 0..model.lam.len() {
        bounds.push(Bounds {
            lo: config.lam_lo,
            hi: config.lam_hi,
        });
    }
    bounds
}

pub(crate) fn pack_params(model: &PsmcModel) -> Vec<f64> {
    let mut params = Vec::with_capacity(3 + model.lam.len());
    params.push(model.theta);
    params.push(model.rho);
    params.push(model.t_max);
    params.extend_from_slice(&model.lam);
    params
}

pub(crate) fn unpack_params(model: &mut PsmcModel, params: &[f64]) -> Result<()> {
    if params.len() != 3 + model.lam.len() {
        bail!("param length mismatch");
    }
    model.theta = params[0];
    model.rho = params[1];
    model.t_max = params[2];
    model.lam.clone_from_slice(&params[3..]);
    Ok(())
}

fn to_unconstrained(params: &[f64], bounds: &[Bounds]) -> Result<Vec<f64>> {
    if params.len() != bounds.len() {
        bail!("bounds length mismatch");
    }
    let mut out = Vec::with_capacity(params.len());
    for (x, b) in params.iter().zip(bounds.iter()) {
        let z = (x - b.lo) / (b.hi - b.lo);
        let z = z.clamp(1e-12, 1.0 - 1e-12);
        out.push(logit(z));
    }
    Ok(out)
}

fn from_unconstrained(p: &[f64], bounds: &[Bounds]) -> Result<Vec<f64>> {
    if p.len() != bounds.len() {
        bail!("bounds length mismatch");
    }
    let mut out = Vec::with_capacity(p.len());
    for (v, b) in p.iter().zip(bounds.iter()) {
        let z = sigmoid(*v);
        out.push(b.lo + (b.hi - b.lo) * z);
    }
    Ok(out)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
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
    /// The norm of the gradient fell below [`MStepConfig::tol_grad`].
    GradientTolerance,
    /// No trial point lowered the objective: not the line search along the
    /// L-BFGS direction, nor the one along its reverse, nor the coordinate
    /// search. Near the optimum this means the objective cannot be lowered
    /// at its precision.
    LineSearchFailed,
    /// [`MStepConfig::max_iters`] iterations ran.
    MaxIters,
}

/// The result of [`m_step_lbfgs_with_outcome`].
#[derive(Debug, Clone)]
pub struct MStepOutcome {
    /// The new parameters, in the order of `init_params`: θ, ρ, t_max, then
    /// the free λ.
    pub params: Vec<f64>,
    /// Evaluations of the objective (with its gradient), the first one at
    /// `init_params` included. C psmc prints its M-step's count of
    /// objective evaluations on the `IT` line of its `.psmc` output. A
    /// failed search costs many evaluations, so this is no measure of how
    /// close the M-step came to [`MStepConfig::max_iters`]; `iterations`
    /// and `stop` are.
    pub evaluations: usize,
    /// L-BFGS iterations, each of which moved the parameters to a lower
    /// objective (at most [`MStepConfig::max_iters`]; 0 when the result is
    /// `init_params`).
    pub iterations: usize,
    /// Why the search stopped.
    pub stop: MStepStop,
}

/// [`m_step_lbfgs_with_progress`], also returning how many times the
/// objective was evaluated.
pub fn m_step_lbfgs_with_outcome(
    model: &PsmcModel,
    stats: &SufficientStats,
    init_params: &[f64],
    bounds: &[Bounds],
    config: &MStepConfig,
    mut on_progress: Option<&mut dyn FnMut(MStepProgress)>,
) -> Result<MStepOutcome> {
    let mut objective = Objective::new(model, stats, bounds, config.lambda)?;
    let mut xk = to_unconstrained(init_params, bounds)?;
    let (mut f0, mut gk) = objective.value_and_gradient(&xk);
    let debug_mstep = std::env::var("PSMC_DEBUG_MSTEP")
        .map(|v| v == "1")
        .unwrap_or(false);
    if debug_mstep {
        eprintln!("M-step init: f0={:.15e}, gnorm={:.6e}", f0, norm(&gk));
    }
    if !f0.is_finite() {
        // No trial point can compare as lower, so the search would spend
        // its whole budget and return the start as if it were a result.
        bail!(
            "the M-step objective is {f0} at its starting parameters, so it cannot be \
             minimised (negative transition probabilities make it NaN; see \
             PsmcModel::check_transitions)"
        );
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();
    let mut iterations = 0;
    let mut stop = MStepStop::MaxIters;

    let pb = if config.progress {
        Some(progress::bar(config.max_iters as u64, "M", "L-BFGS"))
    } else {
        None
    };
    if let Some(cb) = on_progress.as_mut() {
        (*cb)(MStepProgress {
            done: 0,
            total: config.max_iters,
        });
    }

    for iter in 0..config.max_iters {
        if let Some(pb) = &pb {
            pb.set_message(format!("L-BFGS {}/{}", iter + 1, config.max_iters));
        }
        if norm(&gk) < config.tol_grad {
            if debug_mstep {
                eprintln!(
                    "M-step converged by grad tol at iter {} (gnorm={:.6e})",
                    iter,
                    norm(&gk)
                );
            }
            stop = MStepStop::GradientTolerance;
            break;
        }
        let mut q = gk.clone();
        let mut alpha = vec![0.0; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let rho = rho_hist[i];
            let a = rho * dot(&s_hist[i], &q);
            alpha[i] = a;
            for j in 0..q.len() {
                q[j] -= a * y_hist[i][j];
            }
        }
        let mut r = if let Some(last) = y_hist.last() {
            let s_last = s_hist.last().unwrap();
            let ys = dot(last, s_last);
            let yy = dot(last, last);
            let h0 = if yy > 0.0 { ys / yy } else { 1.0 };
            q.iter().map(|v| v * h0).collect::<Vec<f64>>()
        } else {
            q.clone()
        };
        for i in 0..s_hist.len() {
            let rho = rho_hist[i];
            let beta = rho * dot(&y_hist[i], &r);
            for j in 0..r.len() {
                r[j] += s_hist[i][j] * (alpha[i] - beta);
            }
        }
        for v in r.iter_mut() {
            *v = -*v;
        }

        let gdotp = dot(&gk, &r);
        // Without curvature pairs the direction is -g, whose length grows
        // with the number of bins; start at unit length instead, so that the
        // backtracking budget does not run out on large inputs.
        let step0 = if s_hist.is_empty() {
            (1.0 / norm(&gk)).min(1.0)
        } else {
            1.0
        };
        let mut step = step0;
        let mut x_new = xk.clone();
        let mut accepted: Option<(f64, Vec<f64>)> = None;
        for ls_iter in 0..config.max_ls_steps {
            for i in 0..xk.len() {
                x_new[i] = xk[i] + step * r[i];
            }
            let (f_new, g_new) = objective.value_and_gradient(&x_new);
            if debug_mstep {
                eprintln!(
                    "  line-search iter {}: step={:.3e}, f_new={:.15e}, rhs={:.15e}",
                    ls_iter + 1,
                    step,
                    f_new,
                    f0 + config.line_search_c1 * step * gdotp
                );
            }
            if f_new <= f0 + config.line_search_c1 * step * gdotp {
                accepted = Some((f_new, g_new));
                break;
            }
            step *= 0.5;
        }
        if accepted.is_none() {
            // Fallback: AD gradients can be locally inconsistent in some regions due
            // piecewise safeguards in the coalescent parameterization. Try reverse
            // direction and accept any strict decrease.
            let mut step_rev = step0;
            for ls_iter in 0..config.max_ls_steps {
                for i in 0..xk.len() {
                    x_new[i] = xk[i] - step_rev * r[i];
                }
                let (f_new, g_new) = objective.value_and_gradient(&x_new);
                if debug_mstep {
                    eprintln!(
                        "  reverse-search iter {}: step={:.3e}, f_new={:.15e}, f0={:.15e}",
                        ls_iter + 1,
                        step_rev,
                        f_new,
                        f0
                    );
                }
                if f_new.is_finite() && f_new < f0 {
                    accepted = Some((f_new, g_new));
                    if debug_mstep {
                        eprintln!("  reverse-search accepted");
                    }
                    break;
                }
                step_rev *= 0.5;
            }
        }
        if accepted.is_none() {
            // Second fallback: coordinate pattern search around current point.
            // This is cheap (small parameter count) and provides a robust escape
            // when line-search along quasi-Newton directions fails.
            let radii = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5];
            for radius in radii {
                let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
                for i in 0..xk.len() {
                    for sign in [-1.0f64, 1.0f64] {
                        let mut x_try = xk.clone();
                        x_try[i] += sign * radius;
                        let (f_try, g_try) = objective.value_and_gradient(&x_try);
                        if !f_try.is_finite() {
                            continue;
                        }
                        if f_try < f0 {
                            match &best {
                                Some((best_f, _, _)) if f_try >= *best_f => {}
                                _ => best = Some((f_try, x_try, g_try)),
                            }
                        }
                    }
                }
                if let Some((f_best, x_best, g_best)) = best {
                    if debug_mstep {
                        eprintln!(
                            "  coordinate-search accepted: radius={:.3e}, f_new={:.15e}, f0={:.15e}",
                            radius, f_best, f0
                        );
                    }
                    accepted = Some((f_best, g_best));
                    x_new = x_best;
                    break;
                } else if debug_mstep {
                    eprintln!(
                        "  coordinate-search radius={:.3e}: no improving move",
                        radius
                    );
                }
            }
        }
        let Some((f_new, g_new)) = accepted else {
            if debug_mstep {
                eprintln!(
                    "M-step line search failed at iter {} (f0={:.15e}, gnorm={:.6e})",
                    iter,
                    f0,
                    norm(&gk)
                );
            }
            stop = MStepStop::LineSearchFailed;
            break;
        };
        let mut s = vec![0.0; xk.len()];
        let mut y = vec![0.0; xk.len()];
        for i in 0..xk.len() {
            s[i] = x_new[i] - xk[i];
            y[i] = g_new[i] - gk[i];
        }
        let ys = dot(&y, &s);
        let dx = norm(&s);
        if ys > 1e-12 {
            if s_hist.len() == config.lbfgs_m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / ys);
        }
        xk = x_new;
        gk = g_new;
        f0 = f_new;
        iterations = iter + 1;
        if debug_mstep {
            eprintln!(
                "M-step iter {} accepted: f={:.15e}, gnorm={:.6e}, |dx|={:.3e}",
                iter + 1,
                f0,
                norm(&gk),
                dx
            );
        }
        if let Some(pb) = &pb {
            pb.inc(1);
        }
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(MStepProgress {
                done: iter + 1,
                total: config.max_iters,
            });
        }
    }
    if let Some(pb) = pb {
        pb.finish_with_message("M-step done");
    }

    Ok(MStepOutcome {
        params: from_unconstrained(&xk, bounds)?,
        evaluations: objective.evaluations,
        iterations,
        stop,
    })
}

#[cfg(test)]
mod tests {
    use super::dual::{Dd, Dual, Real, compute_params_dual, eval_dual_in, expand_lam_dual};
    use super::objective::EVAL_CALLS;
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

    /// A model with random θ, ρ, t_max and λ; `param_recalculate` has run.
    fn random_model(rng: &mut Rng, n_steps: usize, pattern: Option<&str>) -> PsmcModel {
        let theta = rng.log_uniform(1e-4, 0.05);
        let rho = theta * rng.log_uniform(0.05, 2.0);
        let t_max = 5.0 + 30.0 * rng.next();
        let mut model = PsmcModel::new(
            t_max,
            n_steps,
            theta,
            rho,
            2.5e-8,
            pattern.map(str::to_string),
        )
        .unwrap();
        for lam in model.lam.iter_mut() {
            *lam = rng.log_uniform(0.05, 20.0);
        }
        model.param_recalculate().unwrap();
        model
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
        let mut model = PsmcModel::new(
            t_max,
            n_steps,
            theta,
            rho,
            2.5e-8,
            pattern.map(str::to_string),
        )
        .unwrap();
        for lam in model.lam.iter_mut() {
            *lam = rng.log_uniform(1e-3, 1e3);
        }
        if huge_last {
            *model.lam.last_mut().unwrap() = rng.log_uniform(900.0, 1e6);
        }
        model.param_recalculate().unwrap();
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
        let lam_full = expand_lam_dual(&vars[3..], model.pattern_spec(), n).unwrap();
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
        SufficientStats { gobs, xi }
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
            let mut model = PsmcModel::new(
                t_max,
                n_steps,
                theta,
                rho,
                2.5e-8,
                pattern.map(str::to_string),
            )
            .unwrap();
            for lam in model.lam.iter_mut() {
                *lam = rng.log_uniform(0.1, 10.0);
            }
            if huge_last {
                *model.lam.last_mut().unwrap() = rng.log_uniform(900.0, 1e6);
            }
            model.param_recalculate().unwrap();
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
        /// Points compared, and those where the objective is not finite
        /// (negative transition probabilities).
        points: usize,
        not_finite: usize,
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
        let t = at_x.compute_t(0.1);
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

    /// Checks the adjoint objective against the forward-mode one at `x`.
    ///
    /// The values must be equal to the bit (both NaN where transition
    /// probabilities are negative), and the adjoint gradient finite where
    /// the value is. Every gradient component must agree with forward mode
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
        assert!(
            f.to_bits() == f_dual.to_bits() || (f.is_nan() && f_dual.is_nan()),
            "{}: value {f} vs forward mode {f_dual}",
            what()
        );
        if !f.is_finite() {
            seen.not_finite += 1;
            return;
        }
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
        assert!(seen.not_finite * 4 < seen.points, "{seen:?}");
        // Most points agree to 1e-10 with forward mode; at the others the
        // adjoint gradient is the more accurate, or as accurate.
        let compared = seen.points - seen.not_finite - seen.underflow;
        assert!(seen.agreed * 4 >= compared * 3, "{seen:?}");
        assert!(seen.error_ratios.iter().all(|&r| r <= 1.1), "{seen:?}");
        assert!(
            seen.inner_fallbacks >= 100 && seen.last_fallbacks >= 20,
            "{seen:?}"
        );
    }

    /// Over the whole region the M-step can reach, λ over six decades and
    /// any parameter near its bounds, the values still agree to the bit.
    /// The float objective is ill-conditioned there (λ near 1e6 leaves the
    /// survival barely decaying over an interval, and C's β sums cancel):
    /// it can be wrong in its 7th digit and forward mode's gradient in its
    /// 3rd, so the double-double evaluation decides most points, and the
    /// adjoint gradient is mostly the more accurate one.
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
        assert!(seen.not_finite * 2 < seen.points, "{seen:?}");
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
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let config = MStepConfig::default();
        for (n, pattern) in LAYOUTS {
            for lambda in [0.0, 1e-2] {
                for _ in 0..3 {
                    let model = random_model(&mut rng, n, pattern);
                    let stats = random_stats(&mut rng, &model, 20_000);
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
                    let scale = norm(&grad);
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
    fn counted_m_step(
        model: &PsmcModel,
        stats: &SufficientStats,
        config: &MStepConfig,
    ) -> (MStepOutcome, usize, usize) {
        let bounds = bounds_from_config(model, config);
        let mut done = 0;
        let mut on_progress = |p: MStepProgress| done = p.done;
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
        (outcome, done, calls)
    }

    #[test]
    fn outcome_counts_objective_evaluations() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let model = random_model(&mut rng, 22, Some("4+5*3+4"));
        let stats = random_stats(&mut rng, &model, 20_000);
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

        // A one-trial line search fails often enough to reach the reverse
        // and coordinate searches, whose evaluations count too.
        let config = MStepConfig {
            max_iters: 12,
            max_ls_steps: 1,
            ..base.clone()
        };
        let (outcome, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.evaluations, calls);
        assert_eq!(outcome.iterations, done);
        assert!(
            calls > 1 + 2 * done,
            "{calls} evaluations, {done} iterations"
        );

        // The default search; the parameters are those of m_step_lbfgs.
        let (outcome, done, calls) = counted_m_step(&model, &stats, &base);
        assert_eq!(outcome.evaluations, calls);
        assert_eq!(outcome.iterations, done);
        assert!(calls > done, "{calls} evaluations, {done} iterations");
        let bounds = bounds_from_config(&model, &base);
        let params = m_step_lbfgs(&model, &stats, &pack_params(&model), &bounds, &base).unwrap();
        assert_eq!(outcome.params, params);
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

        // Any gradient is below an infinite tolerance: no step is taken.
        let config = MStepConfig {
            tol_grad: f64::INFINITY,
            ..base.clone()
        };
        let (outcome, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::GradientTolerance);
        assert_eq!((outcome.iterations, done, calls), (0, 0, 1));
        // The start, up to the round trip through the unconstrained space.
        for (a, b) in outcome.params.iter().zip(pack_params(&model)) {
            assert!((a - b).abs() <= 1e-12 * b, "{a} vs {b}");
        }

        // A few iterations from a random start: the cap ends the search.
        let config = MStepConfig {
            max_iters: 3,
            tol_grad: 0.0,
            ..base.clone()
        };
        let (outcome, done, _) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::MaxIters);
        assert_eq!((outcome.iterations, done), (3, 3));

        // Without a tolerance and a cap out of reach, the search runs until
        // the objective cannot be lowered at its precision, after more
        // evaluations than iterations.
        let config = MStepConfig {
            max_iters: 100_000,
            tol_grad: 0.0,
            ..base.clone()
        };
        let (outcome, done, calls) = counted_m_step(&model, &stats, &config);
        assert_eq!(outcome.stop, MStepStop::LineSearchFailed);
        assert_eq!(outcome.iterations, done);
        assert!(done > 3 && done < 100_000, "{done} iterations");
        assert!(calls > done + 20, "{calls} evaluations, {done} iterations");
    }
}
