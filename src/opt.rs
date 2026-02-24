use anyhow::{Result, bail};

use crate::hmm::{
    EStepPhase, EStepProgress, SufficientStats, e_step_streaming, e_step_streaming_with_progress,
};
use crate::model::PsmcModel;
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

impl<'a, 'b> Add<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn add(self, rhs: &'b Dual) -> Dual {
        let mut grad = vec![0.0; self.grad.len()];
        for i in 0..grad.len() {
            grad[i] = self.grad[i] + rhs.grad[i];
        }
        Dual {
            re: self.re + rhs.re,
            grad,
        }
    }
}

impl<'a, 'b> Sub<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn sub(self, rhs: &'b Dual) -> Dual {
        let mut grad = vec![0.0; self.grad.len()];
        for i in 0..grad.len() {
            grad[i] = self.grad[i] - rhs.grad[i];
        }
        Dual {
            re: self.re - rhs.re,
            grad,
        }
    }
}

impl<'a, 'b> Mul<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn mul(self, rhs: &'b Dual) -> Dual {
        let mut grad = vec![0.0; self.grad.len()];
        for i in 0..grad.len() {
            grad[i] = self.grad[i] * rhs.re + rhs.grad[i] * self.re;
        }
        Dual {
            re: self.re * rhs.re,
            grad,
        }
    }
}

impl<'a, 'b> Div<&'b Dual> for &'a Dual {
    type Output = Dual;
    fn div(self, rhs: &'b Dual) -> Dual {
        let denom = rhs.re * rhs.re;
        let mut grad = vec![0.0; self.grad.len()];
        for i in 0..grad.len() {
            grad[i] = (self.grad[i] * rhs.re - rhs.grad[i] * self.re) / denom;
        }
        Dual {
            re: self.re / rhs.re,
            grad,
        }
    }
}

impl<'a> Add<f64> for &'a Dual {
    type Output = Dual;
    fn add(self, rhs: f64) -> Dual {
        Dual {
            re: self.re + rhs,
            grad: self.grad.clone(),
        }
    }
}

impl<'a> Sub<f64> for &'a Dual {
    type Output = Dual;
    fn sub(self, rhs: f64) -> Dual {
        Dual {
            re: self.re - rhs,
            grad: self.grad.clone(),
        }
    }
}

impl<'a> Mul<f64> for &'a Dual {
    type Output = Dual;
    fn mul(self, rhs: f64) -> Dual {
        let grad = self.grad.iter().map(|g| g * rhs).collect();
        Dual {
            re: self.re * rhs,
            grad,
        }
    }
}

impl<'a> Div<f64> for &'a Dual {
    type Output = Dual;
    fn div(self, rhs: f64) -> Dual {
        let grad = self.grad.iter().map(|g| g / rhs).collect();
        Dual {
            re: self.re / rhs,
            grad,
        }
    }
}

impl<'a> Add<&'a Dual> for f64 {
    type Output = Dual;
    fn add(self, rhs: &'a Dual) -> Dual {
        Dual {
            re: self + rhs.re,
            grad: rhs.grad.clone(),
        }
    }
}

impl<'a> Sub<&'a Dual> for f64 {
    type Output = Dual;
    fn sub(self, rhs: &'a Dual) -> Dual {
        let grad = rhs.grad.iter().map(|g| -g).collect();
        Dual {
            re: self - rhs.re,
            grad,
        }
    }
}

impl<'a> Mul<&'a Dual> for f64 {
    type Output = Dual;
    fn mul(self, rhs: &'a Dual) -> Dual {
        let grad = rhs.grad.iter().map(|g| g * self).collect();
        Dual {
            re: self * rhs.re,
            grad,
        }
    }
}

impl<'a> Div<&'a Dual> for f64 {
    type Output = Dual;
    fn div(self, rhs: &'a Dual) -> Dual {
        let denom = rhs.re * rhs.re;
        let grad = rhs.grad.iter().map(|g| -self * g / denom).collect();
        Dual {
            re: self / rhs.re,
            grad,
        }
    }
}

impl<'a> Neg for &'a Dual {
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
    pub lambda: f64,
    pub line_search_c1: f64,
    pub max_ls_steps: usize,
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
            lambda: 1e-3,
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

#[derive(Debug)]
pub struct EmHistory {
    pub loglike: Vec<(f64, f64)>,
    pub params: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
pub enum EmProgressEvent {
    IterStart {
        iter: usize,
        total_iters: usize,
    },
    EStep {
        iter: usize,
        total_iters: usize,
        phase: EStepPhase,
        done: u64,
        total: u64,
    },
    MStep {
        iter: usize,
        total_iters: usize,
        done: usize,
        total: usize,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct MStepProgress {
    pub done: usize,
    pub total: usize,
}

pub fn em_train(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
) -> Result<EmHistory> {
    em_train_with_progress(model, rows, row_starts, n_iter, config, None)
}

pub fn em_train_with_progress(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
    mut on_progress: Option<&mut dyn FnMut(EmProgressEvent)>,
) -> Result<EmHistory> {
    let mut history = EmHistory {
        loglike: Vec::with_capacity(n_iter),
        params: Vec::with_capacity(n_iter + 1),
    };
    let mut params = pack_params(model);
    history.params.push(params.clone());

    let bounds = bounds_from_config(model, config);
    let pb_em = if config.progress && n_iter > 0 {
        Some(progress::bar(n_iter as u64, "EM", "overall"))
    } else {
        None
    };

    for iter in 0..n_iter {
        if let Some(cb) = on_progress.as_mut() {
            (*cb)(EmProgressEvent::IterStart {
                iter: iter + 1,
                total_iters: n_iter,
            });
        }
        if let Some(pb) = &pb_em {
            pb.set_message(format!("overall {}/{}", iter + 1, n_iter));
        }
        model.param_recalculate()?;
        let e = if let Some(cb_outer) = on_progress.as_mut() {
            let mut map_es = |ev: EStepProgress| {
                (*cb_outer)(EmProgressEvent::EStep {
                    iter: iter + 1,
                    total_iters: n_iter,
                    phase: ev.phase,
                    done: ev.done,
                    total: ev.total,
                });
            };
            e_step_streaming_with_progress(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                rows,
                row_starts,
                config.progress,
                "E",
                Some(&mut map_es),
            )?
        } else {
            e_step_streaming(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                rows,
                row_starts,
                config.progress,
                "E",
            )?
        };
        let loglike_before = e.loglike;
        let stats = e.stats;
        let params_opt = if let Some(cb_outer) = on_progress.as_mut() {
            let mut map_ms = |ev: MStepProgress| {
                (*cb_outer)(EmProgressEvent::MStep {
                    iter: iter + 1,
                    total_iters: n_iter,
                    done: ev.done,
                    total: ev.total,
                });
            };
            m_step_lbfgs_with_progress(model, &stats, &params, &bounds, config, Some(&mut map_ms))?
        } else {
            m_step_lbfgs(model, &stats, &params, &bounds, config)?
        };
        params = params_opt.clone();
        unpack_params(model, &params)?;
        // Keep EM history shape stable without paying for a second full E-step.
        let loglike_after = loglike_before;

        history.loglike.push((loglike_before, loglike_after));
        history.params.push(params.clone());

        if let Some(pb) = &pb_em {
            pb.inc(1);
        }
    }
    if let Some(pb) = pb_em {
        pb.finish_with_message("EM done");
    }

    Ok(history)
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

fn pack_params(model: &PsmcModel) -> Vec<f64> {
    let mut params = Vec::with_capacity(3 + model.lam.len());
    params.push(model.theta);
    params.push(model.rho);
    params.push(model.t_max);
    params.extend_from_slice(&model.lam);
    params
}

fn unpack_params(model: &mut PsmcModel, params: &[f64]) -> Result<()> {
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

fn compute_params_dual(
    n_steps: usize,
    theta: &Dual,
    rho: &Dual,
    t_max: &Dual,
    lam_full: &[Dual],
) -> Result<(Vec<Vec<Dual>>, Vec<Vec<Dual>>)> {
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
    t.push(Dual::constant(1e300, n_params));

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
            let denom = &alpha[k] - &alpha[k + 1];
            let term = &alpha[k + 1] / &denom;
            avg_t = &sum_t + &(&lak - &(&tau[k] * &term));
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

fn eval_ad(
    base_model: &PsmcModel,
    stats: &SufficientStats,
    params: &[f64],
    bounds: &[Bounds],
    lambda: f64,
) -> Result<(f64, Vec<f64>)> {
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
    mut on_progress: Option<&mut dyn FnMut(MStepProgress)>,
) -> Result<Vec<f64>> {
    let mut xk = to_unconstrained(init_params, bounds)?;
    let (mut f0, mut gk) = eval_ad(model, stats, &xk, bounds, config.lambda)?;
    let debug_mstep = std::env::var("PSMC_DEBUG_MSTEP")
        .map(|v| v == "1")
        .unwrap_or(false);
    if debug_mstep {
        eprintln!("M-step init: f0={:.6e}, gnorm={:.6e}", f0, norm(&gk));
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

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
        let mut step = 1.0;
        let mut x_new = xk.clone();
        let mut accepted: Option<(f64, Vec<f64>)> = None;
        for ls_iter in 0..config.max_ls_steps {
            for i in 0..xk.len() {
                x_new[i] = xk[i] + step * r[i];
            }
            let (f_new, g_new) = eval_ad(model, stats, &x_new, bounds, config.lambda)?;
            if debug_mstep {
                eprintln!(
                    "  line-search iter {}: step={:.3e}, f_new={:.6e}, rhs={:.6e}",
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
            let mut step_rev = 1.0;
            for ls_iter in 0..config.max_ls_steps {
                for i in 0..xk.len() {
                    x_new[i] = xk[i] - step_rev * r[i];
                }
                let (f_new, g_new) = eval_ad(model, stats, &x_new, bounds, config.lambda)?;
                if debug_mstep {
                    eprintln!(
                        "  reverse-search iter {}: step={:.3e}, f_new={:.6e}, f0={:.6e}",
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
                        let (f_try, g_try) = eval_ad(model, stats, &x_try, bounds, config.lambda)?;
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
                            "  coordinate-search accepted: radius={:.3e}, f_new={:.6e}, f0={:.6e}",
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
                    "M-step line search failed at iter {} (f0={:.6e}, gnorm={:.6e})",
                    iter,
                    f0,
                    norm(&gk)
                );
            }
            break;
        };
        let mut s = vec![0.0; xk.len()];
        let mut y = vec![0.0; xk.len()];
        for i in 0..xk.len() {
            s[i] = x_new[i] - xk[i];
            y[i] = g_new[i] - gk[i];
        }
        let ys = dot(&y, &s);
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
        if debug_mstep {
            eprintln!(
                "M-step iter {} accepted: f={:.6e}, gnorm={:.6e}, step={:.3e}",
                iter + 1,
                f0,
                norm(&gk),
                step
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

    from_unconstrained(&xk, bounds)
}
