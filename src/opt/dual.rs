//! The M-step objective differentiated in forward mode: every value carries
//! its derivatives with respect to all parameters (a heap vector per
//! operation). The M-step used it before the adjoint gradient of
//! [`super::objective`]; it is kept as the test oracle of that gradient.
//! `compute_params_dual` is a line-by-line copy of
//! [`PsmcModel::param_recalculate`](crate::model::PsmcModel::param_recalculate)
//! on dual numbers.
//!
//! The dual numbers are generic over their scalar: in `f64` they perform
//! the operations of the M-step code before the adjoint gradient, to the
//! bit; in double-double arithmetic ([`Dd`], about 32 digits) the same code
//! gives the objective's derivatives to far more digits than either `f64`
//! gradient has, as a reference where the objective is ill-conditioned.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{Result, bail};

use super::Bounds;
use super::objective::EmissionCounts;
use crate::hmm::SufficientStats;
use crate::model::{PsmcModel, T_INF};

/// The scalar of a [`Dual`].
pub(super) trait Real:
    Copy
    + PartialOrd
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn of(v: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn is_finite(self) -> bool;
    fn to_f64(self) -> f64;
}

impl Real for f64 {
    fn of(v: f64) -> Self {
        v
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// A double-double number `hi + lo` (`|lo| <= ulp(hi) / 2`), with about 32
/// significant digits; the algorithms of Hida, Li and Bailey's QD library.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Dd {
    hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

impl Dd {
    const LN2: Dd = Dd {
        hi: std::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };

    fn new(hi: f64, lo: f64) -> Self {
        if hi.is_finite() {
            let (hi, lo) = quick_two_sum(hi, lo);
            Self { hi, lo }
        } else {
            Self { hi, lo: 0.0 }
        }
    }

    /// `self * 2^k`, exactly.
    fn scale(self, k: i32) -> Self {
        let half = f64::powi(2.0, k / 2);
        let rest = f64::powi(2.0, k - k / 2);
        Self {
            hi: self.hi * half * rest,
            lo: self.lo * half * rest,
        }
    }
}

impl PartialOrd for Dd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi) {
            Some(Ordering::Equal) => self.lo.partial_cmp(&other.lo),
            order => order,
        }
    }
}

impl Add for Dd {
    type Output = Dd;
    fn add(self, b: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, b.hi);
        if !s.is_finite() {
            return Dd { hi: s, lo: 0.0 };
        }
        let (t, f) = two_sum(self.lo, b.lo);
        let (s, e) = quick_two_sum(s, e + t);
        Dd::new(s, e + f)
    }
}

impl Neg for Dd {
    type Output = Dd;
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Dd {
    type Output = Dd;
    fn sub(self, b: Dd) -> Dd {
        self + (-b)
    }
}

impl Mul for Dd {
    type Output = Dd;
    fn mul(self, b: Dd) -> Dd {
        let p = self.hi * b.hi;
        if !p.is_finite() {
            return Dd { hi: p, lo: 0.0 };
        }
        let e = self.hi.mul_add(b.hi, -p);
        Dd::new(p, e + (self.hi * b.lo + self.lo * b.hi))
    }
}

impl Div for Dd {
    type Output = Dd;
    fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        if !q1.is_finite() || q1 == 0.0 {
            return Dd { hi: q1, lo: 0.0 };
        }
        let r = self - b * Dd::of(q1);
        let q2 = r.hi / b.hi;
        let r = r - b * Dd::of(q2);
        let q3 = r.hi / b.hi;
        Dd::new(q1, q2) + Dd::of(q3)
    }
}

impl Real for Dd {
    fn of(v: f64) -> Self {
        Dd { hi: v, lo: 0.0 }
    }

    fn exp(self) -> Self {
        if self.hi.is_nan() {
            return self;
        }
        if self.hi > 709.8 {
            return Dd::of(f64::INFINITY);
        }
        if self.hi < -745.2 {
            return Dd::of(0.0);
        }
        // exp(x) = 2^k exp(r)^1024 with x = k ln 2 + 1024 r, |r| < 3.4e-4.
        let k = (self.hi / std::f64::consts::LN_2).round();
        let r = (self - Dd::LN2 * Dd::of(k)).scale(-10);
        // s = exp(r) - 1 by its Taylor series, then (1 + s)^2 - 1 ten times.
        let mut s = r;
        let mut term = r;
        for i in 2..30 {
            term = term * r / Dd::of(i as f64);
            s = s + term;
            if term.hi.abs() < 1e-36 * s.hi.abs() {
                break;
            }
        }
        for _ in 0..10 {
            s = s * (s + Dd::of(2.0));
        }
        (s + Dd::of(1.0)).scale(k as i32)
    }

    fn ln(self) -> Self {
        if self.hi.is_nan() || self.hi < 0.0 {
            return Dd::of(f64::NAN);
        }
        if self.hi == 0.0 {
            return Dd::of(f64::NEG_INFINITY);
        }
        if self.hi.is_infinite() {
            return self;
        }
        // Newton's method on exp(y) = x from the f64 logarithm.
        let mut y = Dd::of(self.hi.ln());
        for _ in 0..2 {
            y = y + self * (-y).exp() - Dd::of(1.0);
        }
        y
    }

    fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

#[derive(Clone, Debug)]
pub(super) struct Dual<T: Real = f64> {
    pub(super) re: T,
    pub(super) grad: Vec<T>,
}

impl<T: Real> Dual<T> {
    pub(super) fn constant(val: f64, n: usize) -> Self {
        Self {
            re: T::of(val),
            grad: vec![T::of(0.0); n],
        }
    }

    pub(super) fn variable(val: f64, n: usize, idx: usize) -> Self {
        let mut grad = vec![T::of(0.0); n];
        grad[idx] = T::of(1.0);
        Self {
            re: T::of(val),
            grad,
        }
    }

    fn exp(&self) -> Self {
        let v = self.re.exp();
        let grad = self.grad.iter().map(|&g| g * v).collect();
        Self { re: v, grad }
    }

    fn ln(&self) -> Self {
        let v = self.re.ln();
        let grad = self.grad.iter().map(|&g| g / self.re).collect();
        Self { re: v, grad }
    }
}

impl<T: Real> Add<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    fn add(self, rhs: &Dual<T>) -> Dual<T> {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(&a, &b)| a + b)
            .collect();
        Dual {
            re: self.re + rhs.re,
            grad,
        }
    }
}

impl<T: Real> Sub<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    fn sub(self, rhs: &Dual<T>) -> Dual<T> {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(&a, &b)| a - b)
            .collect();
        Dual {
            re: self.re - rhs.re,
            grad,
        }
    }
}

impl<T: Real> Mul<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    // The product rule adds the two partial products.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: &Dual<T>) -> Dual<T> {
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(&a, &b)| a * rhs.re + b * self.re)
            .collect();
        Dual {
            re: self.re * rhs.re,
            grad,
        }
    }
}

impl<T: Real> Div<&Dual<T>> for &Dual<T> {
    type Output = Dual<T>;
    fn div(self, rhs: &Dual<T>) -> Dual<T> {
        let denom = rhs.re * rhs.re;
        let grad = self
            .grad
            .iter()
            .zip(&rhs.grad)
            .map(|(&a, &b)| (a * rhs.re - b * self.re) / denom)
            .collect();
        Dual {
            re: self.re / rhs.re,
            grad,
        }
    }
}

impl<T: Real> Add<f64> for &Dual<T> {
    type Output = Dual<T>;
    fn add(self, rhs: f64) -> Dual<T> {
        Dual {
            re: self.re + T::of(rhs),
            grad: self.grad.clone(),
        }
    }
}

impl<T: Real> Sub<f64> for &Dual<T> {
    type Output = Dual<T>;
    fn sub(self, rhs: f64) -> Dual<T> {
        Dual {
            re: self.re - T::of(rhs),
            grad: self.grad.clone(),
        }
    }
}

impl<T: Real> Mul<f64> for &Dual<T> {
    type Output = Dual<T>;
    fn mul(self, rhs: f64) -> Dual<T> {
        let grad = self.grad.iter().map(|&g| g * T::of(rhs)).collect();
        Dual {
            re: self.re * T::of(rhs),
            grad,
        }
    }
}

impl<T: Real> Div<f64> for &Dual<T> {
    type Output = Dual<T>;
    fn div(self, rhs: f64) -> Dual<T> {
        let grad = self.grad.iter().map(|&g| g / T::of(rhs)).collect();
        Dual {
            re: self.re / T::of(rhs),
            grad,
        }
    }
}

impl<T: Real> Add<&Dual<T>> for f64 {
    type Output = Dual<T>;
    fn add(self, rhs: &Dual<T>) -> Dual<T> {
        Dual {
            re: T::of(self) + rhs.re,
            grad: rhs.grad.clone(),
        }
    }
}

impl<T: Real> Sub<&Dual<T>> for f64 {
    type Output = Dual<T>;
    fn sub(self, rhs: &Dual<T>) -> Dual<T> {
        let grad = rhs.grad.iter().map(|&g| -g).collect();
        Dual {
            re: T::of(self) - rhs.re,
            grad,
        }
    }
}

impl<T: Real> Mul<&Dual<T>> for f64 {
    type Output = Dual<T>;
    fn mul(self, rhs: &Dual<T>) -> Dual<T> {
        let grad = rhs.grad.iter().map(|&g| g * T::of(self)).collect();
        Dual {
            re: T::of(self) * rhs.re,
            grad,
        }
    }
}

impl<T: Real> Div<&Dual<T>> for f64 {
    type Output = Dual<T>;
    fn div(self, rhs: &Dual<T>) -> Dual<T> {
        let denom = rhs.re * rhs.re;
        let grad = rhs.grad.iter().map(|&g| T::of(-self) * g / denom).collect();
        Dual {
            re: T::of(self) / rhs.re,
            grad,
        }
    }
}

impl<T: Real> Neg for &Dual<T> {
    type Output = Dual<T>;
    fn neg(self) -> Dual<T> {
        let grad = self.grad.iter().map(|&g| -g).collect();
        Dual { re: -self.re, grad }
    }
}

fn sigmoid_dual<T: Real>(x: &Dual<T>) -> Dual<T> {
    let n = x.grad.len();
    let one = Dual::constant(1.0, n);
    let denom = &one + &(-x).exp();
    &one / &denom
}

fn from_unconstrained_dual<T: Real>(p: &[Dual<T>], bounds: &[Bounds]) -> Result<Vec<Dual<T>>> {
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

pub(super) fn expand_lam_dual<T: Real>(
    lam_grouped: &[Dual<T>],
    spec: Option<&[(usize, usize)]>,
    n_steps: usize,
) -> Result<Vec<Dual<T>>> {
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

/// The result of [`compute_params_dual`]: the transition and emission
/// matrices, row by row, `θ avg_t[k]` of each state (whose exponential is
/// the emission of `T`), and the states whose average time is the
/// fallback.
pub(super) struct DualParams<T: Real> {
    pub(super) p_kl: Vec<Vec<Dual<T>>>,
    pub(super) em: Vec<Vec<Dual<T>>>,
    pub(super) prod: Vec<Dual<T>>,
    pub(super) fallbacks: Vec<usize>,
}

pub(super) fn compute_params_dual<T: Real>(
    n_steps: usize,
    theta: &Dual<T>,
    rho: &Dual<T>,
    t_max: &Dual<T>,
    lam_full: &[Dual<T>],
) -> Result<DualParams<T>> {
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
    let mut prods = Vec::with_capacity(n + 1);
    let mut fallbacks = Vec::new();

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
            fallbacks.push(k);
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
        prods.push(prod);

        sum_t = &sum_t + &tau[k];
    }

    Ok(DualParams {
        p_kl,
        em,
        prod: prods,
        fallbacks,
    })
}

fn smooth_penalty_dual<T: Real>(lam_full: &[Dual<T>]) -> Dual<T> {
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
fn q_function_stats_dual<T: Real>(
    params: &DualParams<T>,
    stats: &SufficientStats,
) -> Result<Dual<T>> {
    let (a, em) = (&params.p_kl, &params.em);
    let n_params = a[0][0].grad.len();
    let n_states = a.len();
    let counts = EmissionCounts::of(stats, n_states)?;
    let mut q = Dual::constant(0.0, n_params);
    // Numerical guard:
    // - skip exact zero-count terms so we never evaluate 0 * log(0)
    // - add a tiny epsilon before log to keep AD stable near boundaries
    let ln_safe = |v: &Dual<T>| (v + 1e-300).ln();
    for i in 0..n_states {
        for j in 0..n_states {
            let c = stats.xi[(i, j)];
            if c != 0.0 {
                q = &q + &(&ln_safe(&a[i][j]) * c);
            }
        }
    }
    for k in 0..n_states {
        let c0 = counts.t[k];
        if c0 != 0.0 {
            q = &q + &(&ln_safe(&em[0][k]) * c0);
        }
        // The K bins of each share f of callable bases emit with
        // probability 1 - exp(-(θ avg) f), summed per state first, as the
        // adjoint objective sums them.
        let mut k_term: Option<Dual<T>> = None;
        for &(f, g) in &counts.k {
            let c1 = g[k];
            if c1 != 0.0 {
                let e = (-&(&params.prod[k] * f)).exp();
                let term = &ln_safe(&(1.0 - &e)) * c1;
                k_term = Some(match k_term {
                    None => term,
                    Some(sum) => &sum + &term,
                });
            }
        }
        if let Some(term) = k_term {
            q = &q + &term;
        }
        let c2 = counts.n[k];
        if c2 != 0.0 {
            q = &q + &(&ln_safe(&em[2][k]) * c2);
        }
    }
    Ok(q)
}

/// The objective at the unconstrained point `params`, its gradient and the
/// states whose average time is the fallback, in the scalar `T`.
pub(super) fn eval_dual_in<T: Real>(
    base_model: &PsmcModel,
    stats: &SufficientStats,
    params: &[f64],
    bounds: &[Bounds],
    lambda: f64,
) -> Result<(T, Vec<T>, Vec<usize>)> {
    let n = params.len();
    let vars: Vec<Dual<T>> = (0..n).map(|i| Dual::variable(params[i], n, i)).collect();
    let constrained = from_unconstrained_dual(&vars, bounds)?;

    let theta = &constrained[0];
    let rho = &constrained[1];
    let t_max = &constrained[2];
    let lam_grouped = &constrained[3..];
    let lam_full = expand_lam_dual(lam_grouped, base_model.pattern_spec(), base_model.n_steps)?;

    let params = compute_params_dual(base_model.n_steps, theta, rho, t_max, &lam_full)?;
    let q = q_function_stats_dual(&params, stats)?;
    let penalty = smooth_penalty_dual(&lam_full);
    let neg_q = -&q;
    let pen = &penalty * lambda;
    let f = &neg_q + &pen;
    Ok((f.re, f.grad, params.fallbacks))
}

mod tests {
    use super::*;

    #[test]
    fn double_double_exp_and_ln_are_accurate() {
        // e, ln 2 and ln 10 to 32 digits.
        let e = Dd::of(1.0).exp();
        assert_eq!(e.hi, std::f64::consts::E);
        assert!((e.lo - 1.445_646_891_729_250_2e-16).abs() < 1e-31, "{e:?}");
        let ln2 = Dd::of(2.0).ln();
        assert_eq!(ln2.hi, Dd::LN2.hi);
        assert!((ln2.lo - Dd::LN2.lo).abs() < 1e-31, "{ln2:?}");
        let ln10 = Dd::of(10.0).ln();
        assert_eq!(ln10.hi, std::f64::consts::LN_10);
        assert!(
            (ln10.lo - (-2.170_756_223_382_249_4e-16)).abs() < 1e-31,
            "{ln10:?}"
        );
        // exp and ln invert each other to about 31 digits while exp keeps
        // both its parts normal.
        let mut x = -600.0;
        while x < 600.0 {
            let v = Dd::of(x) + Dd::of(x * 1e-17);
            let back = v.exp().ln();
            let err = (back - v).to_f64().abs();
            assert!(err <= 1e-30 * x.abs().max(1.0), "{x}: {err:e}");
            x += 13.37;
        }
        // Division and products.
        let third = Dd::of(1.0) / Dd::of(3.0);
        let one = third * Dd::of(3.0);
        assert!((one - Dd::of(1.0)).to_f64().abs() < 1e-31);
    }
}
