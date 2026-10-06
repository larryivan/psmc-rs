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
//! its own scale: the M-step then converges in 10 to 50 iterations, to
//! about 1e-7 of the optimum.

use std::collections::VecDeque;

use super::objective::Objective;

/// Step of the forward differences of the gradient that estimate the
/// Hessian's diagonal, in the unconstrained coordinates (about a relative
/// change of 1e-4 in each parameter).
const CURVATURE_STEP: f64 = 1e-4;

/// The settings of [`minimize`].
pub(crate) struct Settings {
    pub max_iters: usize,
    pub memory: usize,
    pub c1: f64,
    pub max_ls_steps: usize,
    pub rel_tol: f64,
    pub debug: bool,
}

/// Consecutive iterations that must meet the convergence test.
const CONVERGED_ITERATIONS: usize = 2;

/// Why [`minimize`] stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    Converged,
    LineSearchFailed,
    MaxIters,
}

/// The result of [`minimize`].
pub(crate) struct Minimum {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub stop: Stop,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The L-BFGS curvature pairs and the diagonal initial matrix.
struct Memory {
    s: VecDeque<Vec<f64>>,
    y: VecDeque<Vec<f64>>,
    capacity: usize,
    /// The initial inverse Hessian, diagonal.
    h0: Vec<f64>,
}

impl Memory {
    /// The quasi-Newton direction `-H g` (two-loop recursion).
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let k = self.s.len();
        let mut q = g.to_vec();
        let mut a = vec![0.0; k];
        for i in (0..k).rev() {
            a[i] = dot(&self.s[i], &q) / dot(&self.y[i], &self.s[i]);
            for (qj, yj) in q.iter_mut().zip(&self.y[i]) {
                *qj -= a[i] * yj;
            }
        }
        for (qj, hj) in q.iter_mut().zip(&self.h0) {
            *qj *= hj;
        }
        for ((s, y), ai) in self.s.iter().zip(&self.y).zip(&a) {
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
            if self.s.len() == self.capacity {
                self.s.pop_front();
                self.y.pop_front();
            }
            self.s.push_back(s);
            self.y.push_back(y);
        }
    }
}

/// The inverse of the Hessian's diagonal at `x` (gradient `g`), from
/// forward differences of the gradient (backward ones where the forward
/// point is infeasible): positive, and at most 1e12 times that of the
/// steepest direction where a curvature is zero, negative or not finite
/// (the line search then shortens the step).
fn inverse_curvatures(obj: &mut Objective, x: &[f64], g: &[f64]) -> Vec<f64> {
    let mut curvature = vec![0.0; x.len()];
    let mut moved = x.to_vec();
    let mut g_moved = vec![0.0; x.len()];
    for i in 0..x.len() {
        moved[i] = x[i] + CURVATURE_STEP;
        curvature[i] = if obj.eval(&moved, &mut g_moved).is_finite() {
            (g_moved[i] - g[i]) / CURVATURE_STEP
        } else {
            moved[i] = x[i] - CURVATURE_STEP;
            obj.eval(&moved, &mut g_moved);
            (g[i] - g_moved[i]) / CURVATURE_STEP
        };
        moved[i] = x[i];
    }
    let steepest = curvature
        .iter()
        .filter(|c| c.is_finite())
        .fold(0.0f64, |m, c| m.max(c.abs()));
    let floor = (1e-12 * steepest).max(f64::MIN_POSITIVE);
    curvature
        .into_iter()
        .map(|c| {
            if c.is_finite() && c > floor {
                1.0 / c
            } else {
                1.0 / floor
            }
        })
        .collect()
}

/// Minimises the objective from `x` (where it is `f`, finite, with gradient
/// `g`); calls `on_iteration` with the count after each iteration.
///
/// Each iteration searches along the L-BFGS direction from a unit step,
/// backtracking by safeguarded quadratic interpolation (to between 0.1 and
/// 0.5 of the step) until the Armijo condition holds with a strict
/// decrease. When no step of the line search does, the memory is cleared
/// and the search repeated along the preconditioned gradient; when that
/// fails too, the objective cannot be lowered at its precision
/// ([`Stop::LineSearchFailed`]).
///
/// The search has converged ([`Stop::Converged`]) when, in each of two
/// consecutive iterations, the objective decreased by at most
/// `rel_tol * |f|` and the next quasi-Newton step predicts no larger a
/// decrease (`g' H g / 2`, the gradient's size in the metric of the inverse
/// Hessian approximation `H`). Both are relative to the objective, which
/// grows with the data, so the test does not depend on the input's size.
pub(crate) fn minimize(
    obj: &mut Objective,
    mut x: Vec<f64>,
    mut f: f64,
    mut g: Vec<f64>,
    settings: &Settings,
    on_iteration: &mut dyn FnMut(usize),
) -> Minimum {
    let n = x.len();
    if settings.max_iters == 0 {
        return Minimum {
            x,
            iterations: 0,
            stop: Stop::MaxIters,
        };
    }
    let mut memory = Memory {
        s: VecDeque::new(),
        y: VecDeque::new(),
        capacity: settings.memory,
        h0: inverse_curvatures(obj, &x, &g),
    };
    if settings.debug {
        let curvatures: Vec<String> = memory
            .h0
            .iter()
            .map(|h| format!("{:.2e}", 1.0 / h))
            .collect();
        eprintln!("M-step curvatures (f={f:.6e}): {}", curvatures.join(" "));
    }
    let mut d = memory.direction(&g);
    let mut x_new = vec![0.0; n];
    let mut g_new = vec![0.0; n];
    let mut iterations = 0;
    let mut quiet = 0;
    loop {
        let mut slope = dot(&g, &d);
        if slope.is_nan() || slope >= 0.0 {
            // Not a descent direction (rounding in the memory): restart.
            memory.s.clear();
            memory.y.clear();
            d = memory.direction(&g);
            slope = dot(&g, &d);
        }
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..settings.max_ls_steps {
            for ((xn, xi), di) in x_new.iter_mut().zip(&x).zip(&d) {
                *xn = xi + step * di;
            }
            let f_try = obj.eval(&x_new, &mut g_new);
            if f_try < f && f_try <= f + settings.c1 * step * slope {
                accepted = Some(f_try);
                break;
            }
            // The minimum of the quadratic through f, the slope and f_try.
            let quadratic = -slope * step * step / (2.0 * (f_try - f - slope * step));
            step = if quadratic.is_finite() {
                quadratic.clamp(0.1 * step, 0.5 * step)
            } else {
                0.5 * step
            };
        }
        let Some(f_new) = accepted else {
            if memory.s.is_empty() {
                if settings.debug {
                    eprintln!("M-step: no decrease along the preconditioned gradient");
                }
                return Minimum {
                    x,
                    iterations,
                    stop: Stop::LineSearchFailed,
                };
            }
            if settings.debug {
                eprintln!("M-step: line search failed; clearing the memory");
            }
            memory.s.clear();
            memory.y.clear();
            d = memory.direction(&g);
            continue;
        };
        // The accepted trial point was the last one evaluated.
        let row_deviation = obj.row_deviation();
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        memory.push(s, y);
        let decrease = f - f_new;
        std::mem::swap(&mut x, &mut x_new);
        std::mem::swap(&mut g, &mut g_new);
        f = f_new;
        iterations += 1;
        on_iteration(iterations);

        d = memory.direction(&g);
        let predicted = -0.5 * dot(&g, &d);
        let scale = settings.rel_tol * f.abs();
        if settings.debug {
            eprintln!(
                "M-step iteration {iterations}: f={f:.17e}, decrease={:.3e}, predicted={:.3e}, \
                 step={step:.3e}, evaluations={}, rows sum to 1 within {row_deviation:.2e}",
                decrease / f.abs(),
                predicted / f.abs(),
                obj.evaluations
            );
        }
        if decrease <= scale && predicted <= scale {
            quiet += 1;
            if quiet >= CONVERGED_ITERATIONS {
                return Minimum {
                    x,
                    iterations,
                    stop: Stop::Converged,
                };
            }
        } else {
            quiet = 0;
        }
        if iterations >= settings.max_iters {
            return Minimum {
                x,
                iterations,
                stop: Stop::MaxIters,
            };
        }
    }
}
