use anyhow::{Result, bail};

use crate::hmm::SufficientStats;
use crate::model::{PsmcModel, T_INF};

pub use crate::em::{EmHistory, EmProgressEvent, em_train, em_train_with_progress};
use crate::progress;
use crate::utils::{logit, sigmoid};

#[derive(Clone, Debug)]
struct Dual {
    re: f64,
    grad: Vec<f64>,
}

impl Dual {
    fn constant(val: f64, n: usize) -> Self {
        Self {
            re: val,
            grad: vec![0.0; n],
        }
    }

    fn variable(val: f64, n: usize, idx: usize) -> Self {
        let mut grad = vec![0.0; n];
        grad[idx] = 1.0;
        Self { re: val, grad }
    }

    fn exp(&self) -> Self {
        let v = self.re.exp();
        let grad = self.grad.iter().map(|g| g * v).collect();
        Self { re: v, grad }
    }

    fn ln(&self) -> Self {
        let v = self.re.ln();
        let grad = self.grad.iter().map(|g| g / self.re).collect();
        Self { re: v, grad }
    }
}

use std::ops::{Add, Div, Mul, Neg, Sub};

impl Add<&Dual> for &Dual {
    type Output = Dual;
    fn add(self, rhs: &Dual) -> Dual {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(a, b)| a + b)
            .collect();
        Dual {
            re: self.re + rhs.re,
            grad,
        }
    }
}

impl Sub<&Dual> for &Dual {
    type Output = Dual;
    fn sub(self, rhs: &Dual) -> Dual {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(a, b)| a - b)
            .collect();
        Dual {
            re: self.re - rhs.re,
            grad,
        }
    }
}

impl Mul<&Dual> for &Dual {
    type Output = Dual;
    // The product rule adds the two partial products.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &Dual) -> Dual {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(a, b)| a * rhs.re + b * self.re)
            .collect();
        Dual {
            re: self.re * rhs.re,
            grad,
        }
    }
}

impl Div<&Dual> for &Dual {
    type Output = Dual;
    fn div(self, rhs: &Dual) -> Dual {
        let denom = rhs.re * rhs.re;
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(a, b)| (a * rhs.re - b * self.re) / denom)
            .collect();
        Dual {
            re: self.re / rhs.re,
            grad,
        }
    }
}

impl Add<f64> for &Dual {
    type Output = Dual;
    fn add(self, rhs: f64) -> Dual {
        Dual {
            re: self.re + rhs,
            grad: self.grad.clone(),
        }
    }
}

impl Sub<f64> for &Dual {
    type Output = Dual;
    fn sub(self, rhs: f64) -> Dual {
        Dual {
            re: self.re - rhs,
            grad: self.grad.clone(),
        }
    }
}

impl Mul<f64> for &Dual {
    type Output = Dual;
    fn mul(self, rhs: f64) -> Dual {
        let grad = self.grad.iter().map(|g| g * rhs).collect();
        Dual {
            re: self.re * rhs,
            grad,
        }
    }
}

impl Div<f64> for &Dual {
    type Output = Dual;
    fn div(self, rhs: f64) -> Dual {
        let grad = self.grad.iter().map(|g| g / rhs).collect();
        Dual {
            re: self.re / rhs,
            grad,
        }
    }
}

impl Add<&Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: &Dual) -> Dual {
        Dual {
            re: self + rhs.re,
            grad: rhs.grad.clone(),
        }
    }
}

impl Sub<&Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: &Dual) -> Dual {
        let grad = rhs.grad.iter().map(|g| -g).collect();
        Dual {
            re: self - rhs.re,
            grad,
        }
    }
}

impl Mul<&Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: &Dual) -> Dual {
        let grad = rhs.grad.iter().map(|g| g * self).collect();
        Dual {
            re: self * rhs.re,
            grad,
        }
    }
}

impl Div<&Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: &Dual) -> Dual {
        let denom = rhs.re * rhs.re;
        let grad = rhs.grad.iter().map(|g| -self * g / denom).collect();
        Dual {
            re: self / rhs.re,
            grad,
        }
    }
}

impl Neg for &Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        let grad = self.grad.iter().map(|g| -g).collect();
        Dual { re: -self.re, grad }
    }
}

fn sigmoid_dual(x: &Dual) -> Dual {
    let n = x.grad.len();
    let one = Dual::constant(1.0, n);
    let denom = &one + &(-x).exp();
    &one / &denom
}

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

fn from_unconstrained_dual(p: &[Dual], bounds: &[Bounds]) -> Result<Vec<Dual>> {
    if p.len() != bounds.len() {
        bail!("bounds length mismatch");
    }
    let mut out = Vec::with_capacity(p.len());
    for (v, b) in p.iter().zip(bounds.iter()) {
        let z = sigmoid_dual(v);
        let scaled = &z * (b.hi - b.lo);
        let scaled = &scaled + b.lo;
        out.push(scaled);
    }
    Ok(out)
}

fn expand_lam_dual(
    lam_grouped: &[Dual],
    spec: Option<&[(usize, usize)]>,
    n_steps: usize,
) -> Result<Vec<Dual>> {
    match spec {
        None => {
            if lam_grouped.len() != n_steps + 1 {
                bail!("lam length mismatch for ungrouped params");
            }
            Ok(lam_grouped.to_vec())
        }
        Some(spec) => {
            let expected = spec.iter().map(|(nr, _)| nr).sum::<usize>();
            if lam_grouped.len() != expected {
                bail!(
                    "lam length {} does not match grouped params {}",
                    lam_grouped.len(),
                    expected
                );
            }
            let mut lam = Vec::with_capacity(n_steps + 1);
            let mut counter = 0usize;
            for (nr, gl) in spec.iter().cloned() {
                for _ in 0..nr {
                    for _ in 0..gl {
                        lam.push(lam_grouped[counter].clone());
                    }
                    counter += 1;
                }
            }
            if lam.len() != n_steps + 1 {
                bail!(
                    "expanded dual lam length {} != n_steps+1 {}",
                    lam.len(),
                    n_steps + 1
                );
            }
            Ok(lam)
        }
    }
}

/// Transition and emission matrices of [`compute_params_dual`], row by row.
type DualMatrices = (Vec<Vec<Dual>>, Vec<Vec<Dual>>);

fn compute_params_dual(
    n_steps: usize,
    theta: &Dual,
    rho: &Dual,
    t_max: &Dual,
    lam_full: &[Dual],
) -> Result<DualMatrices> {
    let n = n_steps;
    let n_params = theta.grad.len();
    if lam_full.len() != n + 1 {
        bail!("lam_full length {} != n_steps+1 {}", lam_full.len(), n + 1);
    }

    let alpha_c = Dual::constant(0.1, n_params);
    let one = Dual::constant(1.0, n_params);
    let beta = (&(&one + &(t_max / &alpha_c)).ln()) / (n as f64);

    let mut t = Vec::with_capacity(n + 2);
    for k in 0..n {
        let exp_term = (&beta * (k as f64)).exp();
        let expm1 = &exp_term - 1.0;
        let tk = &alpha_c * &expm1;
        t.push(tk);
    }
    t.push(t_max.clone());
    t.push(Dual::constant(T_INF, n_params));

    let mut tau = vec![Dual::constant(0.0, n_params); n + 1];
    for k in 0..=n {
        tau[k] = &t[k + 1] - &t[k];
    }

    let mut alpha = vec![Dual::constant(0.0, n_params); n + 2];
    alpha[0] = one.clone();
    for k in 1..=n {
        let ratio = &tau[k - 1] / &lam_full[k - 1];
        let term = (-&ratio).exp();
        alpha[k] = &alpha[k - 1] * &term;
    }
    alpha[n + 1] = Dual::constant(0.0, n_params);

    let mut beta_arr = vec![Dual::constant(0.0, n_params); n + 1];
    for k in 1..=n {
        let inv_a = 1.0 / &alpha[k];
        let inv_prev = 1.0 / &alpha[k - 1];
        let diff = &inv_a - &inv_prev;
        let term = &lam_full[k - 1] * &diff;
        beta_arr[k] = &beta_arr[k - 1] + &term;
    }

    let mut c_pi = Dual::constant(0.0, n_params);
    for i in 0..=n {
        let diff = &alpha[i] - &alpha[i + 1];
        c_pi = &c_pi + &(&lam_full[i] * &diff);
    }
    let c_sigma = &(1.0 / &(&c_pi * rho)) + 0.5;

    let mut q_aux = vec![Dual::constant(0.0, n_params); n];
    for m in 0..n {
        let diff_alpha = &alpha[m] - &alpha[m + 1];
        let term = &beta_arr[m] - &(&lam_full[m] / &alpha[m]);
        q_aux[m] = &(&diff_alpha * &term) + &tau[m];
    }

    let mut sigma = vec![Dual::constant(0.0, n_params); n + 1];
    let mut p_kl = vec![vec![Dual::constant(0.0, n_params); n + 1]; n + 1];
    let mut em = vec![vec![Dual::constant(0.0, n_params); n + 1]; 3];
    let mut q = vec![Dual::constant(0.0, n_params); n + 1];

    let mut sum_t = Dual::constant(0.0, n_params);

    for k in 0..=n {
        let ak1 = &alpha[k] - &alpha[k + 1];
        let lak = lam_full[k].clone();

        let cpik = &(&ak1 * &(&sum_t + &lak)) - &(&alpha[k + 1] * &tau[k]);
        let pik = &cpik / &c_pi;
        sigma[k] = &(&(&ak1 / &(&c_pi * rho)) + &(&pik / 2.0)) / &c_sigma;

        let tmp_avg = -(&(&one - &(&pik / &(&c_sigma * &sigma[k]))).ln());
        let mut avg_t = &tmp_avg / rho;
        let avg_re = avg_t.re;
        let sum_re = sum_t.re;
        let tau_re = tau[k].re;
        if !avg_re.is_finite() || avg_re < sum_re || avg_re > (sum_re + tau_re) {
            // (tau * alpha[k+1]) / denom, in C's (and param_recalculate's)
            // order: the cancellation in lak - term magnifies any rounding
            // difference.
            let denom = &alpha[k] - &alpha[k + 1];
            let term = &(&tau[k] * &alpha[k + 1]) / &denom;
            avg_t = &sum_t + &(&lak - &term);
        }

        let tmp = &ak1 / &cpik;
        for m in 0..k {
            q[m] = &tmp * &q_aux[m];
        }
        let term1 = &(&ak1 * &ak1) * &(&beta_arr[k] - &(&lak / &alpha[k]));
        let term2 = &(&lak * 2.0) * &ak1;
        let term3 = &(&alpha[k + 1] * 2.0) * &tau[k];
        q[k] = &(&term1 + &term2) - &term3;
        q[k] = &q[k] / &cpik;

        if k < n {
            let tmp2 = &q_aux[k] / &cpik;
            for m in (k + 1)..=n {
                let diff_alpha = &alpha[m] - &alpha[m + 1];
                q[m] = &diff_alpha * &tmp2;
            }
        }

        let tmp3 = &pik / &(&c_sigma * &sigma[k]);
        for m in 0..=n {
            p_kl[k][m] = &tmp3 * &q[m];
        }
        p_kl[k][k] = &(&tmp3 * &q[k]) + &(1.0 - &tmp3);

        let prod = theta * &avg_t;
        let exp_term = (-&prod).exp();
        em[0][k] = exp_term.clone();
        em[1][k] = &one - &exp_term;
        em[2][k] = one.clone();

        sum_t = &sum_t + &tau[k];
    }

    Ok((p_kl, em))
}

fn smooth_penalty_dual(lam_full: &[Dual]) -> Dual {
    let n_params = lam_full[0].grad.len();
    let mut sum = Dual::constant(0.0, n_params);
    if lam_full.len() < 2 {
        return sum;
    }
    for k in 0..(lam_full.len() - 1) {
        let d = &lam_full[k + 1].ln() - &lam_full[k].ln();
        sum = &sum + &(&d * &d);
    }
    sum
}

// The loops index parallel arrays (matrices and expected counts) by state.
#[allow(clippy::needless_range_loop)]
fn q_function_stats_dual(a: &[Vec<Dual>], em: &[Vec<Dual>], stats: &SufficientStats) -> Dual {
    let n_params = a[0][0].grad.len();
    let n_states = a.len();
    let mut q = Dual::constant(0.0, n_params);
    // Numerical guard:
    // - skip exact zero-count terms so we never evaluate 0 * log(0)
    // - add a tiny epsilon before log to keep AD stable near boundaries
    let ln_safe = |v: &Dual| (v + 1e-300).ln();
    for i in 0..n_states {
        for j in 0..n_states {
            let c = stats.xi[(i, j)];
            if c != 0.0 {
                q = &q + &(&ln_safe(&a[i][j]) * c);
            }
        }
    }
    for k in 0..n_states {
        let c0 = stats.gobs[0][k];
        if c0 != 0.0 {
            q = &q + &(&ln_safe(&em[0][k]) * c0);
        }
        let c1 = stats.gobs[1][k];
        if c1 != 0.0 {
            q = &q + &(&ln_safe(&em[1][k]) * c1);
        }
        let c2 = stats.gobs[2][k];
        if c2 != 0.0 {
            q = &q + &(&ln_safe(&em[2][k]) * c2);
        }
    }
    q
}

// Calls of `eval_ad` on this thread, for checking `MStepOutcome::evaluations`
// (the M-step runs on the calling thread).
#[cfg(test)]
thread_local! {
    static EVAL_AD_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn eval_ad(
    base_model: &PsmcModel,
    stats: &SufficientStats,
    params: &[f64],
    bounds: &[Bounds],
    lambda: f64,
) -> Result<(f64, Vec<f64>)> {
    #[cfg(test)]
    EVAL_AD_CALLS.with(|c| c.set(c.get() + 1));
    let n = params.len();
    let vars: Vec<Dual> = (0..n).map(|i| Dual::variable(params[i], n, i)).collect();
    let constrained = from_unconstrained_dual(&vars, bounds)?;

    let theta = &constrained[0];
    let rho = &constrained[1];
    let t_max = &constrained[2];
    let lam_grouped = &constrained[3..];
    let lam_full = expand_lam_dual(lam_grouped, base_model.pattern_spec(), base_model.n_steps)?;

    let (a, em) = compute_params_dual(base_model.n_steps, theta, rho, t_max, &lam_full)?;
    let q = q_function_stats_dual(&a, &em, stats);
    let penalty = smooth_penalty_dual(&lam_full);
    let neg_q = -&q;
    let pen = &penalty * lambda;
    let f = &neg_q + &pen;
    Ok((f.re, f.grad))
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

/// The objective of the M-step, counting its evaluations.
struct Objective<'a> {
    model: &'a PsmcModel,
    stats: &'a SufficientStats,
    bounds: &'a [Bounds],
    lambda: f64,
    evaluations: usize,
}

impl Objective<'_> {
    fn eval(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.evaluations += 1;
        eval_ad(self.model, self.stats, x, self.bounds, self.lambda)
    }
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
    let mut objective = Objective {
        model,
        stats,
        bounds,
        lambda: config.lambda,
        evaluations: 0,
    };
    let mut xk = to_unconstrained(init_params, bounds)?;
    let (mut f0, mut gk) = objective.eval(&xk)?;
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
            let (f_new, g_new) = objective.eval(&x_new)?;
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
                let (f_new, g_new) = objective.eval(&x_new)?;
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
                        let (f_try, g_try) = objective.eval(&x_try)?;
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
    use super::*;
    use crate::hmm::e_step_streaming;

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
    /// mean coalescent time) rather than the log formula.
    fn fallback_states(model: &PsmcModel) -> Vec<usize> {
        let n = model.n_steps;
        let (t, lam) = (&model.t, model.map_lam(&model.lam).unwrap());
        let mut alpha = vec![1.0; n + 2];
        for k in 1..=n {
            alpha[k] = alpha[k - 1] * (-(t[k] - t[k - 1]) / lam[k - 1]).exp();
        }
        alpha[n + 1] = 0.0;
        let avg = model.avg_t();
        (0..=n)
            .filter(|&k| {
                let tau = t[k + 1] - t[k];
                let mean = t[k] + lam[k] - tau * alpha[k + 1] / (alpha[k] - alpha[k + 1]);
                (avg[k] - mean).abs() <= 1e-9 * mean.abs()
            })
            .collect()
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
        let (a, em) = compute_params_dual(n, &vars[0], &vars[1], &vars[2], &lam_full).unwrap();
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

    #[test]
    fn ad_gradient_matches_central_differences() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let config = MStepConfig::default();
        for (n, pattern) in LAYOUTS {
            for lambda in [0.0, 1e-2] {
                for _ in 0..3 {
                    let model = random_model(&mut rng, n, pattern);
                    let stats = random_stats(&mut rng, &model, 20_000);
                    let bounds = bounds_from_config(&model, &config);
                    let x = to_unconstrained(&pack_params(&model), &bounds).unwrap();
                    let (_, grad) = eval_ad(&model, &stats, &x, &bounds, lambda).unwrap();
                    let f = |x: &[f64]| eval_ad(&model, &stats, x, &bounds, lambda).unwrap().0;
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
            let (f, _) = eval_ad(&model, &stats, &x, &bounds, config.lambda).unwrap();
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
    /// reported as done and the number of `eval_ad` calls it made.
    fn counted_m_step(
        model: &PsmcModel,
        stats: &SufficientStats,
        config: &MStepConfig,
    ) -> (MStepOutcome, usize, usize) {
        let bounds = bounds_from_config(model, config);
        let mut done = 0;
        let mut on_progress = |p: MStepProgress| done = p.done;
        let calls_before = EVAL_AD_CALLS.with(|c| c.get());
        let outcome = m_step_lbfgs_with_outcome(
            model,
            stats,
            &pack_params(model),
            &bounds,
            config,
            Some(&mut on_progress),
        )
        .unwrap();
        let calls = EVAL_AD_CALLS.with(|c| c.get()) - calls_before;
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
