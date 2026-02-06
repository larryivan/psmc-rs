use anyhow::{Result, bail};
use ndarray::Array2;

use crate::io::params::{PsmcParamsFile, save_params};

#[derive(Debug, Clone)]
pub struct PsmcModel {
    pub n_steps: usize,
    pub mu: f64,
    pub pattern: Option<String>,
    pattern_spec: Option<Vec<(usize, usize)>>,

    pub theta: f64,
    pub rho: f64,
    pub lam: Vec<f64>,
    pub t_max: f64,

    pub t: Vec<f64>,
    pub c_pi: f64,
    pub c_sigma: f64,
    pub p_kl: Array2<f64>,
    pub em: Array2<f64>,
    pub sigma: Vec<f64>,
    pub pi_k: Vec<f64>,
}

impl PsmcModel {
    pub fn new(
        t_max: f64,
        n_steps: usize,
        theta0: f64,
        rho0: f64,
        mu: f64,
        pattern: Option<String>,
    ) -> Result<Self> {
        let pattern_spec = match pattern.as_deref() {
            Some(p) => Some(parse_pattern(p)?),
            None => None,
        };
        if let Some(spec) = &pattern_spec {
            let n_from_pattern = spec.iter().map(|(ts, gs)| ts * gs).sum::<usize>();
            if n_from_pattern != n_steps {
                bail!(
                    "pattern implies n_steps={}, but n_steps={} was provided",
                    n_from_pattern,
                    n_steps
                );
            }
        }
        let n_free = n_free_params(n_steps, pattern_spec.as_deref());
        let lam = vec![1.0; n_free - 3];
        let mut model = Self {
            n_steps,
            mu,
            pattern,
            pattern_spec,
            theta: theta0,
            rho: rho0,
            lam,
            t_max,
            t: Vec::new(),
            c_pi: 0.0,
            c_sigma: 0.0,
            p_kl: Array2::zeros((n_steps + 1, n_steps + 1)),
            em: Array2::zeros((3, n_steps + 1)),
            sigma: vec![0.0; n_steps + 1],
            pi_k: vec![0.0; n_steps + 1],
        };
        model.param_recalculate()?;
        Ok(model)
    }

    pub fn n_free_params(&self) -> usize {
        n_free_params(self.n_steps, self.pattern_spec.as_deref())
    }

    pub fn n0(&self) -> f64 {
        self.theta / (4.0 * self.mu)
    }

    pub fn compute_t(&self, alpha: f64) -> Vec<f64> {
        let beta = (1.0 + self.t_max / alpha).ln() / self.n_steps as f64;
        let mut t = Vec::with_capacity(self.n_steps + 2);
        for k in 0..self.n_steps {
            t.push(alpha * ((beta * k as f64).exp() - 1.0));
        }
        t.push(self.t_max);
        t.push(1e300);
        t
    }

    pub fn map_lam(&self, lam_grouped: &[f64]) -> Result<Vec<f64>> {
        match &self.pattern_spec {
            None => Ok(lam_grouped.to_vec()),
            Some(spec) => {
                let expected = spec.iter().map(|(ts, _)| ts).sum::<usize>() + 1;
                if lam_grouped.len() != expected {
                    bail!(
                        "lam length {} does not match grouped parameters {}",
                        lam_grouped.len(),
                        expected
                    );
                }
                let mut lam = Vec::with_capacity(self.n_steps + 1);
                let mut counter = 0usize;
                for (ts, gs) in spec.iter().cloned() {
                    for _ in 0..ts {
                        for _ in 0..gs {
                            lam.push(lam_grouped[counter]);
                        }
                        counter += 1;
                    }
                }
                lam.push(*lam_grouped.last().unwrap());
                Ok(lam)
            }
        }
    }

    pub fn param_recalculate(&mut self) -> Result<()> {
        let t = self.compute_t(0.1);
        let n = self.n_steps;
        let lam = self.map_lam(&self.lam)?;
        if lam.len() != n + 1 {
            bail!(
                "expanded lam length {} does not match n_steps+1 {}",
                lam.len(),
                n + 1
            );
        }

        let mut alpha = vec![0.0f64; n + 2];
        let mut tau = vec![0.0f64; n + 1];
        let mut beta = vec![0.0f64; n + 1];
        let mut sigma = vec![0.0f64; n + 1];
        let mut q_aux = vec![0.0f64; n];
        let mut q = vec![f64::NAN; n + 1];
        let mut e = Array2::zeros((3, n + 1));
        let mut p_kl = Array2::zeros((n + 1, n + 1));
        let mut pi_k = vec![0.0f64; n + 1];

        for k in 0..=n {
            tau[k] = t[k + 1] - t[k];
        }

        alpha[0] = 1.0;
        for k in 1..=n {
            alpha[k] = alpha[k - 1] * (-tau[k - 1] / lam[k - 1]).exp();
        }
        alpha[n + 1] = 0.0;

        beta[0] = 0.0;
        for k in 1..=n {
            beta[k] = beta[k - 1] + lam[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1]);
        }

        let mut c_pi = 0.0;
        for i in 0..=n {
            c_pi += lam[i] * (alpha[i] - alpha[i + 1]);
        }
        let c_sigma = 1.0 / (c_pi * self.rho) + 0.5;

        for m in 0..n {
            q_aux[m] = (alpha[m] - alpha[m + 1]) * (beta[m] - lam[m] / alpha[m]) + tau[m];
        }

        let mut sum_t = 0.0;
        for k in 0..=n {
            let ak1 = alpha[k] - alpha[k + 1];
            let lak = lam[k];

            let cpik = ak1 * (sum_t + lak) - alpha[k + 1] * tau[k];
            let pik = cpik / c_pi;
            pi_k[k] = pik;
            sigma[k] = (ak1 / (c_pi * self.rho) + pik / 2.0) / c_sigma;

            let mut avg_t = -(1.0 - pik / (c_sigma * sigma[k])).ln() / self.rho;
            if !avg_t.is_finite() || avg_t < sum_t || avg_t > sum_t + tau[k] {
                avg_t = sum_t + (lak - tau[k] * alpha[k + 1] / (alpha[k] - alpha[k + 1]));
            }

            let tmp = ak1 / cpik;
            for m in 0..k {
                q[m] = tmp * q_aux[m];
            }
            q[k] = (ak1 * ak1 * (beta[k] - lak / alpha[k]) + 2.0 * lak * ak1
                - 2.0 * alpha[k + 1] * tau[k])
                / cpik;

            if k < n {
                let tmp2 = q_aux[k] / cpik;
                for m in (k + 1)..=n {
                    q[m] = (alpha[m] - alpha[m + 1]) * tmp2;
                }
            }

            let tmp3 = pik / (c_sigma * sigma[k]);
            for m in 0..=n {
                p_kl[(k, m)] = tmp3 * q[m];
            }
            p_kl[(k, k)] = tmp3 * q[k] + (1.0 - tmp3);

            let exp_term = (-self.theta * avg_t).exp();
            e[(0, k)] = exp_term;
            e[(1, k)] = 1.0 - exp_term;
            e[(2, k)] = 1.0;

            sum_t += tau[k];
        }

        self.t = t;
        self.c_pi = c_pi;
        self.c_sigma = c_sigma;
        self.p_kl = p_kl;
        self.em = e;
        self.sigma = sigma;
        self.pi_k = pi_k;

        Ok(())
    }

    pub fn save_params(&self, path: &std::path::Path) -> Result<()> {
        let params = PsmcParamsFile {
            theta: self.theta,
            rho: self.rho,
            lam: self.lam.clone(),
            t_max: self.t_max,
            n_steps: self.n_steps,
            mu: self.mu,
            pattern: self.pattern.clone(),
        };
        save_params(path, &params)
    }

    pub fn prior_matrix(&self) -> &[f64] {
        &self.sigma
    }

    pub fn transition_matrix(&self) -> &Array2<f64> {
        &self.p_kl
    }

    pub fn emission_matrix(&self) -> &Array2<f64> {
        &self.em
    }

    pub fn pattern_spec(&self) -> Option<&[(usize, usize)]> {
        self.pattern_spec.as_deref()
    }
}

fn n_free_params(n_steps: usize, spec: Option<&[(usize, usize)]>) -> usize {
    match spec {
        None => 4 + n_steps,
        Some(v) => 4 + v.iter().map(|(ts, _)| ts).sum::<usize>(),
    }
}

fn parse_pattern(pattern: &str) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for part in pattern.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('*') {
            let ts: usize = a.trim().parse()?;
            let gs: usize = b.trim().parse()?;
            if ts == 0 || gs == 0 {
                bail!("pattern values must be > 0");
            }
            out.push((ts, gs));
        } else {
            let ts: usize = part.parse()?;
            if ts == 0 {
                bail!("pattern values must be > 0");
            }
            out.push((ts, 1));
        }
    }
    if out.is_empty() {
        bail!("pattern is empty");
    }
    Ok(out)
}
