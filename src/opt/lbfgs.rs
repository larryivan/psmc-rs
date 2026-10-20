//! The M-step's minimiser: L-BFGS with a diagonal initial matrix from the
//! objective's curvature, an Armijo backtracking line search and a
//! scale-free convergence test.
//!
//! The curvature of the M-step objective in its unconstrained coordinates
//! (about the logarithms of the parameters) spans up to eight orders of
//! magnitude: on a simulated genome of 1 M bins its second derivative was
//! 5e4 along θ and 10 to 1,000 along the λ, but 1e-3 along the most recent
//! λ, which the data barely constrain, once EM had taken it to 1e4. With
//! the classic initial matrix `(s'y / y'y) I`, which follows the steep
//! directions, L-BFGS steps along such a flat one are too short to lower
//! the objective above its rounding, and it stalls: there it ended an
//! M-step with that λ 37% short of its optimum (which C psmc's Hooke-Jeeves
//! search reaches) and the others up to 1e-5 off. The inverse of the
//! Hessian's diagonal, from forward differences of the gradient at the
//! start (one evaluation per parameter), gives each direction a step of
//! its own scale: the M-step then converges in 10 to 20 iterations with C
//! psmc's patterns (up to about 70 with one λ per interval), to about 1e-7
//! of the optimum.
//!
//! The search can also stop short of the maximum of Q where the L-BFGS
//! memory, built from curvatures elsewhere, mispredicts what is left. A λ
//! the data barely constrain (a recent or the oldest interval) has a flat
//! profile beyond its optimum: Q hardly depends on it once its state has
//! little mass. A first step scaled by its small curvature, or the joint
//! steps of the others, can carry it far past its optimum, where its
//! gradient is small and the memory predicts no decrease worth a step. On
//! a 2 Mb simulation the first λ went from 1 to 9,859 with its optimum at
//! 49 (Q 7.6e-4 short); on a 4 × 25 Mb one to 7,387 with its optimum at 23
//! (3.1e-2 short), where its gradient (1.7e-3) and curvature (3.8e-3)
//! promised a decrease of 3.7e-4 that the memory put below 2.6e-8; the
//! oldest λ went from 1 to 3.8e-4 with its optimum at 0.68, along a
//! profile that curves downwards. And the logit coordinates saturate: once
//! a parameter reaches its bound, steps move its coordinate without
//! changing the objective, as far as -57 on a 4 × 25 Mb genome, from where
//! nothing brings it back.
//!
//! So the coordinates are kept within ±[`Settings::x_limit`], the logit of
//! the margin within which the M-step's start is clamped to the bounds,
//! and the search is checked before it stops ([`check_before_stopping`]):
//! with fresh curvatures, each coordinate along which the objective does
//! not curve upwards is moved alone, downhill, by the best of
//! [`FLAT_MOVES`], and the other coordinates' diagonal quadratic model
//! predicts what is left. When either promises more than the convergence
//! threshold, the search restarts with an empty memory and those
//! curvatures (at most [`MAX_RESTARTS`] times). On the statistics of 55
//! E-steps of simulated genomes (23 to 101 intervals), no M-step then
//! left a parameter that raises Q by more than 1e-8 when moved alone.

use std::collections::VecDeque;

use super::MStepStop;
use super::objective::Objective;

/// Step of the forward differences of the gradient that estimate the
/// Hessian's diagonal, in the unconstrained coordinates (about a relative
/// change of 1e-4 in each parameter).
const CURVATURE_STEP: f64 = 1e-4;

/// The moves of one coordinate that [`move_flat_coordinates`] tries, in
/// the unconstrained coordinates (in which 1 is about a factor e in a
/// parameter away from its bounds); the largest crosses the whole range.
const FLAT_MOVES: [f64; 9] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

/// Restarts by [`check_before_stopping`] in one minimisation.
const MAX_RESTARTS: usize = 3;

/// Consecutive iterations that must meet the convergence test.
const CONVERGED_ITERATIONS: usize = 2;

/// The settings of [`minimize`].
pub(crate) struct Settings {
    pub max_iters: usize,
    pub memory: usize,
    pub c1: f64,
    pub max_ls_steps: usize,
    pub rel_tol: f64,
    /// The largest magnitude of a coordinate.
    pub x_limit: f64,
    pub debug: bool,
}

/// The result of [`minimize`].
pub(crate) struct Minimum {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub stop: MStepStop,
}

/// A point with its objective `f` and gradient `g`.
struct Point {
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a - b`, element by element.
fn difference(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(a, b)| a - b).collect()
}

/// The L-BFGS curvature pairs `(s, y)` and the diagonal initial matrix.
struct Memory {
    pairs: VecDeque<(Vec<f64>, Vec<f64>)>,
    capacity: usize,
    /// The initial inverse Hessian, diagonal.
    h0: Vec<f64>,
}

impl Memory {
    /// The quasi-Newton direction `-H g` (two-loop recursion).
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let mut q = g.to_vec();
        let mut a = vec![0.0; self.pairs.len()];
        for ((s, y), ai) in self.pairs.iter().zip(&mut a).rev() {
            *ai = dot(s, &q) / dot(y, s);
            for (qj, yj) in q.iter_mut().zip(y) {
                *qj -= *ai * yj;
            }
        }
        for (qj, hj) in q.iter_mut().zip(&self.h0) {
            *qj *= hj;
        }
        for ((s, y), ai) in self.pairs.iter().zip(&a) {
            let b = dot(y, &q) / dot(y, s);
            for (qj, sj) in q.iter_mut().zip(s) {
                *qj += (ai - b) * sj;
            }
        }
        for qj in q.iter_mut() {
            *qj = -*qj;
        }
        q
    }

    /// Keeps the pair of a step `s` and its change of gradient `y` when it
    /// satisfies the curvature condition `s'y > 0`, with a margin relative
    /// to `|s| |y|` for rounding: otherwise the update would make the
    /// inverse Hessian indefinite.
    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) {
        let sy = dot(&s, &y);
        if self.capacity > 0
            && sy.is_finite()
            && sy > 1e-10 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt()
        {
            if self.pairs.len() == self.capacity {
                self.pairs.pop_front();
            }
            self.pairs.push_back((s, y));
        }
    }

    /// Forgets the curvature pairs: the direction becomes `-h0 g`.
    fn clear(&mut self) {
        self.pairs.clear();
    }
}

/// The Hessian's diagonal at `x` (gradient `g`), from forward differences
/// of the gradient, or backward ones where the forward point is infeasible
/// or beyond ±`limit` (NaN where neither can be taken).
fn curvatures(obj: &mut Objective, x: &[f64], g: &[f64], limit: f64) -> Vec<f64> {
    let mut curvature = vec![f64::NAN; x.len()];
    let mut moved = x.to_vec();
    let mut g_moved = vec![0.0; x.len()];
    for i in 0..x.len() {
        for h in [CURVATURE_STEP, -CURVATURE_STEP] {
            moved[i] = x[i] + h;
            if moved[i].abs() <= limit && obj.eval(&moved, &mut g_moved).is_finite() {
                curvature[i] = (g_moved[i] - g[i]) / h;
                break;
            }
        }
        moved[i] = x[i];
    }
    curvature
}

/// The smallest curvature taken as positive: 1e-12 of the steepest.
fn curvature_floor(curvature: &[f64]) -> f64 {
    let steepest = curvature
        .iter()
        .filter(|c| c.is_finite())
        .fold(0.0f64, |m, c| m.max(c.abs()));
    (1e-12 * steepest).max(f64::MIN_POSITIVE)
}

/// `1 / c`, or `1 / floor` where the curvature `c` is not above `floor`.
fn inverse(c: f64, floor: f64) -> f64 {
    if c > floor { 1.0 / c } else { 1.0 / floor }
}

/// The inverse of the Hessian's diagonal at `x` (gradient `g`), from
/// [`curvatures`]: positive, and at most 1e12 times that of the steepest
/// direction where a curvature is zero, negative or not finite (the line
/// search then shortens the step).
fn inverse_curvatures(obj: &mut Objective, x: &[f64], g: &[f64], limit: f64) -> Vec<f64> {
    let curvature = curvatures(obj, x, g, limit);
    let floor = curvature_floor(&curvature);
    curvature
        .into_iter()
        .map(|c| inverse(if c.is_finite() { c } else { floor }, floor))
        .collect()
}

/// The check before the search stops, at `at`: from fresh curvatures,
/// moves the flat coordinates ([`move_flat_coordinates`]), and returns the
/// inverse curvatures to go on with when the objective could still fall by
/// more than `threshold`: when those moves lowered it by more, or when the
/// diagonal quadratic model of the other coordinates predicts more
/// (`g' D g / 2`, with `D` their inverse curvatures). The latter catches a
/// stop that the L-BFGS memory, built from curvatures elsewhere, predicted
/// wrongly: a λ carried far past its optimum to where Q falls slowly
/// towards the bound kept a gradient of 1.7e-3 at curvature 3.8e-3 (a
/// Newton decrease of 3.7e-4) while the memory predicted less than 2.6e-8.
fn check_before_stopping(
    obj: &mut Objective,
    at: &mut Point,
    settings: &Settings,
    threshold: f64,
) -> Option<Vec<f64>> {
    let limit = settings.x_limit;
    let curvature = curvatures(obj, &at.x, &at.g, limit);
    let floor = curvature_floor(&curvature);
    let moved = move_flat_coordinates(obj, at, &curvature, floor, settings);
    if moved > threshold {
        return Some(inverse_curvatures(obj, &at.x, &at.g, limit));
    }
    let predicted: f64 = curvature
        .iter()
        .zip(&at.g)
        .filter(|(c, _)| **c > floor)
        .map(|(c, gi)| 0.5 * gi * gi / c)
        .sum();
    if settings.debug {
        eprintln!(
            "M-step: before stopping, flat moves decreased {moved:.3e} and the diagonal \
             model predicts {predicted:.3e}, against {threshold:.3e}"
        );
    }
    (predicted > threshold).then(|| curvature.iter().map(|&c| inverse(c, floor)).collect())
}

/// Moves each coordinate along which the objective at `at` does not curve
/// upwards (`curvature` not above `floor`) alone, downhill by each of
/// [`FLAT_MOVES`] within ±`x_limit`, to the lowest objective found below
/// the current one. Returns the decrease of the objective.
fn move_flat_coordinates(
    obj: &mut Objective,
    at: &mut Point,
    curvature: &[f64],
    floor: f64,
    settings: &Settings,
) -> f64 {
    let limit = settings.x_limit;
    let start = at.f;
    let mut trial = at.x.clone();
    let mut g_trial = vec![0.0; trial.len()];
    let mut g_best = vec![0.0; trial.len()];
    for i in 0..trial.len() {
        let gi = at.g[i];
        if curvature[i] > floor || !(gi != 0.0 && gi.is_finite()) {
            continue;
        }
        let downhill = -gi.signum();
        let mut best: Option<(f64, f64)> = None;
        for &moved in &FLAT_MOVES {
            trial[i] = (at.x[i] + downhill * moved).clamp(-limit, limit);
            if trial[i] == at.x[i] {
                continue;
            }
            let f_trial = obj.eval(&trial, &mut g_trial);
            if f_trial < best.map_or(at.f, |(f_best, _)| f_best) {
                best = Some((f_trial, trial[i]));
                std::mem::swap(&mut g_best, &mut g_trial);
            }
        }
        let Some((f_best, x_best)) = best else {
            trial[i] = at.x[i];
            continue;
        };
        if settings.debug {
            eprintln!(
                "M-step: coordinate {i} (curvature {:.2e}) moved from {:.3} to {:.3}, \
                 decrease={:.3e}",
                curvature[i],
                at.x[i],
                x_best,
                (at.f - f_best) / at.f.abs()
            );
        }
        at.x[i] = x_best;
        trial[i] = x_best;
        at.f = f_best;
        at.g.copy_from_slice(&g_best);
    }
    start - at.f
}

/// Searches along `d` (slope `slope` at `at`) from a unit step,
/// backtracking by safeguarded quadratic interpolation (to between 0.1 and
/// 0.5 of the step) until the Armijo condition holds with a strict
/// decrease. Returns the accepted step, whose point is then in `trial`.
fn line_search(
    obj: &mut Objective,
    at: &Point,
    d: &[f64],
    slope: f64,
    settings: &Settings,
    trial: &mut Point,
) -> Option<f64> {
    let mut step = 1.0;
    for _ in 0..settings.max_ls_steps {
        for ((xt, xi), di) in trial.x.iter_mut().zip(&at.x).zip(d) {
            *xt = (xi + step * di).clamp(-settings.x_limit, settings.x_limit);
        }
        trial.f = obj.eval(&trial.x, &mut trial.g);
        if trial.f < at.f && trial.f <= at.f + settings.c1 * step * slope {
            return Some(step);
        }
        // The minimum of the quadratic through f, the slope and the trial.
        let quadratic = -slope * step * step / (2.0 * (trial.f - at.f - slope * step));
        step = if quadratic.is_finite() {
            quadratic.clamp(0.1 * step, 0.5 * step)
        } else {
            0.5 * step
        };
    }
    None
}

/// Minimises the objective from `x` (where it is `f`, finite, with gradient
/// `g`); calls `on_iteration` with the count after each iteration.
///
/// Each iteration searches along the L-BFGS direction ([`line_search`]).
/// When no step of the line search lowers the objective, the memory is
/// cleared and the search repeated along the preconditioned gradient; when
/// that fails too, the objective cannot be lowered at its precision
/// ([`MStepStop::LineSearchFailed`]).
///
/// The search has converged ([`MStepStop::Converged`]) when, in each of two
/// consecutive iterations, the objective decreased by at most
/// `rel_tol * |f|` and the next quasi-Newton step predicts no larger a
/// decrease (`g' H g / 2`, the gradient's size in the metric of the inverse
/// Hessian approximation `H`). Both are relative to the objective, which
/// grows with the data, so the test does not depend on the input's size.
///
/// Trial points are kept within ±`x_limit`. Before either stop, the
/// search is checked with fresh curvatures ([`check_before_stopping`]),
/// and restarts from its point (with an empty memory, counting one
/// iteration when the check moved flat coordinates) when the check
/// promises a decrease of more than `rel_tol * |f|`, at most
/// [`MAX_RESTARTS`] times.
pub(crate) fn minimize(
    obj: &mut Objective,
    x: Vec<f64>,
    f: f64,
    g: Vec<f64>,
    settings: &Settings,
    on_iteration: &mut dyn FnMut(usize),
) -> Minimum {
    let n = x.len();
    if settings.max_iters == 0 {
        return Minimum {
            x,
            iterations: 0,
            stop: MStepStop::MaxIters,
        };
    }
    let mut memory = Memory {
        pairs: VecDeque::new(),
        capacity: settings.memory,
        h0: inverse_curvatures(obj, &x, &g, settings.x_limit),
    };
    if settings.debug {
        let curvatures: Vec<String> = memory
            .h0
            .iter()
            .map(|h| format!("{:.2e}", 1.0 / h))
            .collect();
        eprintln!("M-step curvatures (f={f:.6e}): {}", curvatures.join(" "));
    }
    let mut at = Point { x, f, g };
    let mut trial = Point {
        x: vec![0.0; n],
        f: f64::NAN,
        g: vec![0.0; n],
    };
    let mut d = memory.direction(&at.g);
    let mut iterations = 0;
    let mut quiet = 0;
    let mut restarts = 0;
    loop {
        if iterations >= settings.max_iters {
            return Minimum {
                x: at.x,
                iterations,
                stop: MStepStop::MaxIters,
            };
        }
        let mut slope = dot(&at.g, &d);
        if slope.is_nan() || slope >= 0.0 {
            // Not a descent direction (rounding in the memory): restart.
            memory.clear();
            d = memory.direction(&at.g);
            slope = dot(&at.g, &d);
        }
        let stop = match line_search(obj, &at, &d, slope, settings, &mut trial) {
            None if memory.pairs.is_empty() => {
                if settings.debug {
                    eprintln!("M-step: no decrease along the preconditioned gradient");
                }
                MStepStop::LineSearchFailed
            }
            None => {
                if settings.debug {
                    eprintln!("M-step: line search failed; clearing the memory");
                }
                memory.clear();
                d = memory.direction(&at.g);
                continue;
            }
            Some(step) => {
                // The accepted trial point was the last one evaluated.
                let row_deviation = obj.row_deviation();
                memory.push(difference(&trial.x, &at.x), difference(&trial.g, &at.g));
                let decrease = at.f - trial.f;
                std::mem::swap(&mut at, &mut trial);
                iterations += 1;
                on_iteration(iterations);

                d = memory.direction(&at.g);
                let predicted = -0.5 * dot(&at.g, &d);
                let scale = settings.rel_tol * at.f.abs();
                if settings.debug {
                    eprintln!(
                        "M-step iteration {iterations}: f={:.17e}, decrease={:.3e}, \
                         predicted={:.3e}, step={step:.3e}, evaluations={}, rows sum to 1 \
                         within {row_deviation:.2e}",
                        at.f,
                        decrease / at.f.abs(),
                        predicted / at.f.abs(),
                        obj.evaluations
                    );
                }
                if decrease > scale || predicted > scale {
                    quiet = 0;
                    continue;
                }
                quiet += 1;
                if quiet < CONVERGED_ITERATIONS {
                    continue;
                }
                MStepStop::Converged
            }
        };
        // About to stop: check with fresh curvatures first.
        if restarts < MAX_RESTARTS && iterations < settings.max_iters {
            restarts += 1;
            let before = at.x.clone();
            let threshold = settings.rel_tol * at.f.abs();
            if let Some(h0) = check_before_stopping(obj, &mut at, settings, threshold) {
                if at.x != before {
                    iterations += 1;
                    on_iteration(iterations);
                }
                memory.clear();
                memory.h0 = h0;
                d = memory.direction(&at.g);
                quiet = 0;
                continue;
            }
        }
        return Minimum {
            x: at.x,
            iterations,
            stop,
        };
    }
}
