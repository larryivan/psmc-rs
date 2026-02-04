use anyhow::{bail, Result};
use ndarray::{Array2, Array3};

use crate::hmm::forward_backward;
use crate::model::PsmcModel;
use crate::utils::{logit, sigmoid};

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone)]
pub struct MStepConfig {
    pub max_iters: usize,
    pub lbfgs_m: usize,
    pub grad_eps: f64,
    pub lambda: f64,
    pub line_search_c1: f64,
    pub max_ls_steps: usize,
    pub tol_grad: f64,
}

impl Default for MStepConfig {
    fn default() -> Self {
        Self {
            max_iters: 30,
            lbfgs_m: 7,
            grad_eps: 1e-4,
            lambda: 1e-2,
            line_search_c1: 1e-4,
            max_ls_steps: 20,
            tol_grad: 1e-4,
        }
    }
}

#[derive(Debug)]
pub struct EmHistory {
    pub loglike: Vec<(f64, f64)>,
    pub params: Vec<Vec<f64>>,
}

pub fn em_train(
    model: &mut PsmcModel,
    x: &Array2<u8>,
    n_iter: usize,
    config: &MStepConfig,
) -> Result<EmHistory> {
    let mut history = EmHistory {
        loglike: Vec::with_capacity(n_iter),
        params: Vec::with_capacity(n_iter + 1),
    };
    let mut params = pack_params(model);
    history.params.push(params.clone());

    let bounds = default_bounds(model);

    for _ in 0..n_iter {
        model.param_recalculate()?;
        let fb = forward_backward(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            x,
        )?;
        let loglike_before = fb.loglike;

        let params_opt = m_step_lbfgs(model, x, &fb.gamma, &fb.xi, &params, &bounds, config)?;
        params = params_opt.clone();
        unpack_params(model, &params)?;

        model.param_recalculate()?;
        let fb_after = forward_backward(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            x,
        )?;
        let loglike_after = fb_after.loglike;

        history.loglike.push((loglike_before, loglike_after));
        history.params.push(params.clone());
    }

    Ok(history)
}

pub fn default_bounds(model: &PsmcModel) -> Vec<Bounds> {
    let mut bounds = Vec::new();
    bounds.push(Bounds { lo: 1e-4, hi: 1e-1 }); // theta
    bounds.push(Bounds { lo: 1e-5, hi: 1e-1 }); // rho
    bounds.push(Bounds { lo: 12.0, hi: 20.0 }); // t_max
    for _ in 0..model.lam.len() {
        bounds.push(Bounds { lo: 0.1, hi: 10.0 });
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

fn smooth_penalty(lam_full: &[f64]) -> f64 {
    if lam_full.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 0..(lam_full.len() - 1) {
        let a = lam_full[k].ln();
        let b = lam_full[k + 1].ln();
        let d = b - a;
        sum += d * d;
    }
    sum
}

fn q_function(model: &PsmcModel, x: &Array2<u8>, gamma: &Array3<f64>, xi: &Array2<f64>) -> f64 {
    let (batch, s_max) = x.dim();
    let n_states = model.sigma.len();
    let pi = model.prior_matrix();
    let a = model.transition_matrix();
    let em = model.emission_matrix();

    let mut q = 0.0;
    let ln = |v: f64| -> f64 { (v.max(1e-300)).ln() };
    for b in 0..batch {
        for k in 0..n_states {
            q += gamma[(b, 0, k)] * ln(pi[k]);
        }
    }
    for i in 0..n_states {
        for j in 0..n_states {
            q += xi[(i, j)] * ln(a[(i, j)]);
        }
    }
    for b in 0..batch {
        for t in 0..s_max {
            let obs = x[(b, t)] as usize;
            for k in 0..n_states {
                q += gamma[(b, t, k)] * ln(em[(obs, k)]);
            }
        }
    }
    q
}

fn cost_function(
    base_model: &PsmcModel,
    x: &Array2<u8>,
    gamma: &Array3<f64>,
    xi: &Array2<f64>,
    params: &[f64],
    bounds: &[Bounds],
    lambda: f64,
) -> Result<f64> {
    let constrained = from_unconstrained(params, bounds)?;
    let mut model = base_model.clone();
    unpack_params(&mut model, &constrained)?;
    model.param_recalculate()?;
    let q = q_function(&model, x, gamma, xi);
    let lam_full = model.map_lam(&model.lam)?;
    let penalty = smooth_penalty(&lam_full);
    Ok(-q + lambda * penalty)
}

fn numerical_grad(
    base_model: &PsmcModel,
    x: &Array2<u8>,
    gamma: &Array3<f64>,
    xi: &Array2<f64>,
    params: &[f64],
    bounds: &[Bounds],
    lambda: f64,
    eps: f64,
) -> Result<Vec<f64>> {
    let mut grad = vec![0.0; params.len()];
    for i in 0..params.len() {
        let step = eps * params[i].abs().max(1.0);
        let mut p1 = params.to_vec();
        let mut p2 = params.to_vec();
        p1[i] += step;
        p2[i] -= step;
        let f1 = cost_function(base_model, x, gamma, xi, &p1, bounds, lambda)?;
        let f2 = cost_function(base_model, x, gamma, xi, &p2, bounds, lambda)?;
        grad[i] = (f1 - f2) / (2.0 * step);
    }
    Ok(grad)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub fn m_step_lbfgs(
    model: &PsmcModel,
    x: &Array2<u8>,
    gamma: &Array3<f64>,
    xi: &Array2<f64>,
    init_params: &[f64],
    bounds: &[Bounds],
    config: &MStepConfig,
) -> Result<Vec<f64>> {
    let mut xk = to_unconstrained(init_params, bounds)?;
    let mut gk = numerical_grad(model, x, gamma, xi, &xk, bounds, config.lambda, config.grad_eps)?;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for _ in 0..config.max_iters {
        if norm(&gk) < config.tol_grad {
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

        let f0 = cost_function(model, x, gamma, xi, &xk, bounds, config.lambda)?;
        let mut step = 1.0;
        let gdotp = dot(&gk, &r);
        let mut x_new = xk.clone();
        let mut f_new;
        let mut ls_ok = false;
        for _ in 0..config.max_ls_steps {
            for i in 0..xk.len() {
                x_new[i] = xk[i] + step * r[i];
            }
            f_new = cost_function(model, x, gamma, xi, &x_new, bounds, config.lambda)?;
            if f_new <= f0 + config.line_search_c1 * step * gdotp {
                ls_ok = true;
                break;
            }
            step *= 0.5;
        }
        if !ls_ok {
            break;
        }

        let g_new =
            numerical_grad(model, x, gamma, xi, &x_new, bounds, config.lambda, config.grad_eps)?;
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
    }

    from_unconstrained(&xk, bounds)
}
