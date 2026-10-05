use anyhow::{Result, bail};
use ndarray::Array2;

use crate::io::params::{PsmcParamsFile, RunMetadata, save_params};

/// C psmc's `PSMC_T_INF`: the end of the last interval, `t[n_steps + 1]`,
/// in units of 2N0 generations. It only matters as the upper bound that
/// sends the last state's average time to its fallback (see
/// [`PsmcModel::avg_t`]), and `t_max` must stay below it.
pub const T_INF: f64 = 1000.0;

/// The largest number of hidden states (`n_steps + 1`) a model may have.
///
/// C psmc's default pattern `4+5*3+4`, which the CLI also uses by default,
/// has 23 states, and the psmc README's human pattern `4+25*2+4+6` has 64.
/// The transition matrix is dense, with `(n_steps + 1)^2` entries (800 MB at
/// this limit), and the E-step does that much work per bin, so larger models
/// cannot be fitted anyway. The limit turns a typo such as an extra digit in
/// a pattern into an error instead of an allocation that panics or exhausts
/// memory.
pub const MAX_STATES: usize = 10_000;

#[derive(Debug, Clone)]
pub struct PsmcModel {
    pub n_steps: usize,
    pub mu: f64,
    pub pattern: Option<String>,
    // C-compatible pattern groups: (n_repeats, group_len)
    pattern_spec: Option<Vec<(usize, usize)>>,

    pub theta: f64,
    pub rho: f64,
    pub lam: Vec<f64>,
    pub t_max: f64,

    pub t: Vec<f64>,
    pub p_kl: Array2<f64>,
    pub em: Array2<f64>,
    pub sigma: Vec<f64>,
    // C's normalisers, kept in step with `sigma` by `param_recalculate`.
    c_pi: f64,
    c_sigma: f64,
}

impl PsmcModel {
    /// Build a model with flat λ and compute its HMM parameters.
    ///
    /// `n_steps` is C psmc's `n`: the model has `n_steps + 1` hidden states.
    /// With a `pattern`, `n_steps` must equal [`n_steps_for_pattern`] of it;
    /// for backward compatibility the state count (`n_steps + 1`) is also
    /// accepted and silently converted. Errors when there would be fewer
    /// than 2 or more than [`MAX_STATES`] states, or `t_max` is outside
    /// `(0, T_INF)`.
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
        let mut n_steps_eff = n_steps;
        if let Some(spec) = &pattern_spec {
            let n_states_from_pattern = n_states_of_spec(spec);
            if n_states_from_pattern < 2 {
                bail!("pattern implies {n_states_from_pattern} state(s); at least 2 are needed");
            }
            let n_steps_from_pattern = n_states_from_pattern - 1;
            if n_steps == n_states_from_pattern {
                // Compatibility shim: previous Rust/Python style often passed state-count here.
                n_steps_eff = n_steps_from_pattern;
            } else if n_steps != n_steps_from_pattern {
                bail!(
                    "pattern implies n_steps={} (legacy state-count={}), but n_steps={} was provided",
                    n_steps_from_pattern,
                    n_states_from_pattern,
                    n_steps
                );
            }
        }
        if n_steps_eff == 0 {
            bail!("n_steps must be at least 1 (the model needs at least 2 states)");
        }
        if n_steps_eff >= MAX_STATES {
            bail!(
                "n_steps must be at most {} (at most {MAX_STATES} states), got {n_steps_eff}",
                MAX_STATES - 1
            );
        }
        check_t_max(t_max)?;
        let n_free = n_free_params(n_steps_eff, pattern_spec.as_deref());
        let lam = vec![1.0; n_free - 3];
        let n_states = n_steps_eff + 1;
        let mut model = Self {
            n_steps: n_steps_eff,
            mu,
            pattern,
            pattern_spec,
            theta: theta0,
            rho: rho0,
            lam,
            t_max,
            t: Vec::new(),
            p_kl: Array2::zeros((n_states, n_states)),
            em: Array2::zeros((3, n_states)),
            sigma: vec![0.0; n_states],
            c_pi: f64::NAN,
            c_sigma: f64::NAN,
        };
        model.param_recalculate()?;
        Ok(model)
    }

    pub fn n_free_params(&self) -> usize {
        n_free_params(self.n_steps, self.pattern_spec.as_deref())
    }

    /// `theta / (4 mu)`: N0 only if `theta` were per base pair.
    ///
    /// The fitted `theta` is per *bin* (C psmc's θ0 for a psmcfa of 100-bp
    /// bins), so this overstates N0 by the bin size. Equal to
    /// `n0_for_bin(1.0)`.
    #[deprecated(note = "theta is per bin; use n0_for_bin(bin_size)")]
    pub fn n0(&self) -> f64 {
        self.theta / (4.0 * self.mu)
    }

    /// Reference effective population size `N0 = theta / (4 mu bin_size)`,
    /// with `theta` per bin, `mu` per base pair per generation and
    /// `bin_size` in base pairs (100 for psmcfa input), as in psmc_plot.pl.
    pub fn n0_for_bin(&self, bin_size: f64) -> f64 {
        self.theta / (4.0 * self.mu * bin_size)
    }

    /// The population-size history on psmc_plot.pl's scales: one point
    /// `[t_k * 2 N0 * gen_years, lambda_k * N0]` (years ago, effective size)
    /// for each state `k = 0..=n_steps`, where the size is `lambda_k * N0`
    /// from that time back to the next point. The first point has x = 0, so
    /// callers drawing on a log axis must drop it.
    ///
    /// These are all the states, as `psmc_plot.pl -L` draws them. Without
    /// `-L`, psmc_plot.pl leaves out the states that share the last state's
    /// λ (the last group of the pattern, or only the last state of a model
    /// without one), so its default curve does not show the oldest size.
    ///
    /// Errors when `bin_size` or `gen_years` is not positive and finite, when
    /// N0 is not (`theta` or `mu` not positive), or when `lam` has a length
    /// the pattern does not allow.
    pub fn ne_curve(&self, bin_size: f64, gen_years: f64) -> Result<Vec<[f64; 2]>> {
        let n0 = self.checked_n0(bin_size, gen_years)?;
        let t = self.compute_t(0.1);
        let lam = self.map_lam(&self.lam)?;
        Ok((0..=self.n_steps)
            .map(|k| [t[k] * 2.0 * n0 * gen_years, lam[k] * n0])
            .collect())
    }

    /// Average coalescent time of each state ([`avg_t`](Self::avg_t)) in
    /// years: `avg_t[k] * 2 N0 * gen_years`, the time to report for a
    /// posterior over states (TMRCA output). Like `avg_t`, it lies inside
    /// the state's interval of [`ne_curve`](Self::ne_curve)'s x values.
    ///
    /// # Panics
    ///
    /// Under the same conditions as [`avg_t`](Self::avg_t).
    pub fn state_tmrca_years(&self, bin_size: f64, gen_years: f64) -> Vec<f64> {
        let n0 = self.n0_for_bin(bin_size);
        self.avg_t()
            .into_iter()
            .map(|a| a * 2.0 * n0 * gen_years)
            .collect()
    }

    fn checked_n0(&self, bin_size: f64, gen_years: f64) -> Result<f64> {
        if !(bin_size > 0.0 && bin_size.is_finite()) {
            bail!("bin size must be positive and finite, got {bin_size}");
        }
        if !(gen_years > 0.0 && gen_years.is_finite()) {
            bail!("generation time must be positive and finite, got {gen_years}");
        }
        let n0 = self.n0_for_bin(bin_size);
        if !(n0 > 0.0 && n0.is_finite()) {
            bail!(
                "N0 = theta / (4 mu bin_size) = {n0} is not positive and finite \
                 (theta = {}, mu = {})",
                self.theta,
                self.mu
            );
        }
        Ok(n0)
    }

    /// The interval boundaries `t_0 = 0 < ... < t_n = t_max < t_{n+1} = T_INF`
    /// (`n_steps + 2` values, in units of 2N0 generations), spaced
    /// log-linearly with C psmc's `alpha` (`-l`, 0.1): C `psmc_update_intv`.
    pub fn compute_t(&self, alpha: f64) -> Vec<f64> {
        let beta = (1.0 + self.t_max / alpha).ln() / self.n_steps as f64;
        let mut t = Vec::with_capacity(self.n_steps + 2);
        for k in 0..self.n_steps {
            t.push(alpha * ((beta * k as f64).exp() - 1.0));
        }
        t.push(self.t_max);
        t.push(T_INF);
        t
    }

    /// Expand per-group λ (as stored in `lam`) to one value per state.
    /// Errors unless the result has `n_steps + 1` values.
    pub fn map_lam(&self, lam_grouped: &[f64]) -> Result<Vec<f64>> {
        match &self.pattern_spec {
            None => {
                if lam_grouped.len() != self.n_steps + 1 {
                    bail!(
                        "lam length {} does not match n_steps+1 {}",
                        lam_grouped.len(),
                        self.n_steps + 1
                    );
                }
                Ok(lam_grouped.to_vec())
            }
            Some(spec) => {
                let expected = spec.iter().map(|(nr, _)| nr).sum::<usize>();
                if lam_grouped.len() != expected {
                    bail!(
                        "lam length {} does not match grouped parameters {}",
                        lam_grouped.len(),
                        expected
                    );
                }
                let mut lam = Vec::with_capacity(self.n_steps + 1);
                let mut counter = 0usize;
                for (nr, gl) in spec.iter().cloned() {
                    for _ in 0..nr {
                        for _ in 0..gl {
                            lam.push(lam_grouped[counter]);
                        }
                        counter += 1;
                    }
                }
                if lam.len() != self.n_steps + 1 {
                    bail!(
                        "expanded lam length {} does not match n_steps+1 {}",
                        lam.len(),
                        self.n_steps + 1
                    );
                }
                Ok(lam)
            }
        }
    }

    /// Recompute the time grid `t` and the HMM parameters (`sigma`, `p_kl`,
    /// `em`) from `theta`, `rho`, `t_max` and `lam`; a port of C
    /// `psmc_update_hmm`.
    pub fn param_recalculate(&mut self) -> Result<()> {
        let Coalescent {
            t,
            lam,
            tau,
            alpha,
            c_pi,
            c_sigma,
            cpi,
            sigma,
            avg_t,
        } = self.coalescent()?;
        let n = self.n_steps;

        let mut beta = vec![0.0f64; n + 1];
        let mut q_aux = vec![0.0f64; n];
        let mut q = vec![f64::NAN; n + 1];
        let mut e = Array2::zeros((3, n + 1));
        let mut p_kl = Array2::zeros((n + 1, n + 1));

        beta[0] = 0.0;
        for k in 1..=n {
            beta[k] = beta[k - 1] + lam[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1]);
        }

        for m in 0..n {
            q_aux[m] = (alpha[m] - alpha[m + 1]) * (beta[m] - lam[m] / alpha[m]) + tau[m];
        }

        for k in 0..=n {
            let ak1 = alpha[k] - alpha[k + 1];
            let lak = lam[k];
            let cpik = cpi[k];
            let pik = cpik / c_pi;

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

            let exp_term = (-self.theta * avg_t[k]).exp();
            e[(0, k)] = exp_term;
            e[(1, k)] = 1.0 - exp_term;
            e[(2, k)] = 1.0;
        }

        self.t = t;
        self.p_kl = p_kl;
        self.em = e;
        self.sigma = sigma;
        self.c_pi = c_pi;
        self.c_sigma = c_sigma;

        Ok(())
    }

    /// C psmc's `C_pi`: `sum_k lambda_k (alpha_k - alpha_{k+1})`, the mean
    /// coalescence time in units of 2N0 generations under the λ history.
    /// C prints it on the `MM C_pi:` line of each `.psmc` round.
    ///
    /// Like `sigma`, it is the value of the last
    /// [`param_recalculate`](Self::param_recalculate) (which
    /// [`new`](Self::new) runs), not recomputed from the current parameters.
    pub fn c_pi(&self) -> f64 {
        self.c_pi
    }

    /// C psmc's `C_sigma = 1 / (C_pi rho) + 1/2`, the normaliser of the
    /// initial distribution `sigma`. Set by
    /// [`param_recalculate`](Self::param_recalculate), like
    /// [`c_pi`](Self::c_pi).
    pub fn c_sigma(&self) -> f64 {
        self.c_sigma
    }

    /// C psmc's `n_recomb = sum_L / C_sigma` for `sum_l` non-missing bins
    /// (C's `sum_L`), printed on the `MM ... n_recomb:` line; C scales the
    /// `RS` lines' fourth column, `n_recomb * sigma_k`, by it.
    pub fn n_recomb(&self, sum_l: u64) -> f64 {
        sum_l as f64 / self.c_sigma
    }

    /// Per-state average coalescent time, in units of 2N0 generations: for
    /// each state k, the time in `[t_k, t_{k+1}]` (`>= t_n` for the last
    /// state) that the emission probability
    /// `em[(0, k)] = exp(-theta * avg_t[k])` is evaluated at.
    ///
    /// This is C `psmc_avg_t` (the `TC` column of `psmc -d`) without the
    /// divergence-time term, including its fallback to the conditional mean
    /// coalescent time of the interval when the log formula leaves it. It is
    /// computed from the current parameters, so it matches `em` exactly
    /// whenever `param_recalculate` has run since they last changed.
    ///
    /// # Panics
    ///
    /// If `param_recalculate` would fail: `lam` has a length the pattern
    /// does not allow or `t_max` is outside `(0, T_INF)`, both of which
    /// `PsmcModel::new` and the M-step bounds rule out.
    pub fn avg_t(&self) -> Vec<f64> {
        self.coalescent()
            .expect("invalid model parameters (lam length or t_max)")
            .avg_t
    }

    /// Quantities of C `psmc_update_hmm` that depend only on the interval
    /// structure; shared by `param_recalculate` and `avg_t` so that the two
    /// cannot disagree. Names follow the C code.
    fn coalescent(&self) -> Result<Coalescent> {
        check_t_max(self.t_max)?;
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

        let mut tau = vec![0.0f64; n + 1];
        for k in 0..=n {
            tau[k] = t[k + 1] - t[k];
        }

        let mut alpha = vec![0.0f64; n + 2];
        alpha[0] = 1.0;
        for k in 1..=n {
            alpha[k] = alpha[k - 1] * (-tau[k - 1] / lam[k - 1]).exp();
        }
        alpha[n + 1] = 0.0;

        let mut c_pi = 0.0;
        for i in 0..=n {
            c_pi += lam[i] * (alpha[i] - alpha[i + 1]);
        }
        let c_sigma = 1.0 / (c_pi * self.rho) + 0.5;

        let mut cpi = vec![0.0f64; n + 1];
        let mut sigma = vec![0.0f64; n + 1];
        let mut avg_t = vec![0.0f64; n + 1];
        let mut sum_t = 0.0;
        for k in 0..=n {
            let ak1 = alpha[k] - alpha[k + 1];
            let lak = lam[k];

            let cpik = ak1 * (sum_t + lak) - alpha[k + 1] * tau[k];
            let pik = cpik / c_pi;
            sigma[k] = (ak1 / (c_pi * self.rho) + pik / 2.0) / c_sigma;

            let mut avg = -(1.0 - pik / (c_sigma * sigma[k])).ln() / self.rho;
            if !avg.is_finite() || avg < sum_t || avg > sum_t + tau[k] {
                avg = sum_t + (lak - tau[k] * alpha[k + 1] / (alpha[k] - alpha[k + 1]));
            }
            cpi[k] = cpik;
            avg_t[k] = avg;

            sum_t += tau[k];
        }

        Ok(Coalescent {
            t,
            lam,
            tau,
            alpha,
            c_pi,
            c_sigma,
            cpi,
            sigma,
            avg_t,
        })
    }

    /// Save the parameters as JSON ([`params_file`](Self::params_file),
    /// without run metadata).
    pub fn save_params(&self, path: &std::path::Path) -> Result<()> {
        save_params(path, &self.params_file())
    }

    /// The parameters as [`PsmcParamsFile`] stores them, with empty run
    /// metadata.
    pub fn params_file(&self) -> PsmcParamsFile {
        PsmcParamsFile {
            theta: self.theta,
            rho: self.rho,
            lam: self.lam.clone(),
            t_max: self.t_max,
            n_steps: self.n_steps,
            mu: self.mu,
            pattern: self.pattern.clone(),
            run: RunMetadata::default(),
        }
    }

    /// The model of saved parameters (for example a previous fit, to start
    /// from as C psmc's `-i` does), with its HMM parameters computed.
    ///
    /// Errors when the layout is invalid (as for [`new`](Self::new)), when
    /// `lam` does not have one value per free λ of that layout, or when a
    /// parameter is not positive and finite.
    pub fn from_params_file(p: &PsmcParamsFile) -> Result<Self> {
        let mut model = Self::new(p.t_max, p.n_steps, p.theta, p.rho, p.mu, p.pattern.clone())?;
        if p.lam.len() != model.lam.len() {
            bail!(
                "{} lambda values for a model with {} free lambdas (n_steps {}, pattern {})",
                p.lam.len(),
                model.lam.len(),
                model.n_steps,
                p.pattern.as_deref().unwrap_or("none")
            );
        }
        let positive = |x: f64| x > 0.0 && x.is_finite();
        for (name, v) in [("theta", p.theta), ("rho", p.rho), ("mu", p.mu)] {
            if !positive(v) {
                bail!("{name} must be positive and finite, got {v}");
            }
        }
        if let Some(v) = p.lam.iter().find(|&&v| !positive(v)) {
            bail!("lambda values must be positive and finite, got {v}");
        }
        model.lam.copy_from_slice(&p.lam);
        model.param_recalculate()?;
        Ok(model)
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

    /// Checks that the transition probabilities of the last
    /// [`param_recalculate`](Self::param_recalculate) are finite and not
    /// negative, as EM needs.
    ///
    /// PSMC's transition probabilities assume at most about one
    /// recombination per bin. When `rho` is too high for `t_max` and the λ,
    /// the probability of staying in the oldest states turns negative. The
    /// M-step's objective is then not a number at the start, so EM (C
    /// psmc's too) never moves from it and returns a flat history. The
    /// error is then a [`NegativeTransitions`] (reach it with
    /// `downcast_ref`), which says how far `rho` must come down.
    pub fn check_transitions(&self) -> Result<()> {
        let mut worst: Option<((usize, usize), f64)> = None;
        for ((from, to), &p) in self.p_kl.indexed_iter() {
            if !p.is_finite() {
                bail!("the transition probability from state {from} to state {to} is {p}");
            }
            if p < 0.0 && worst.is_none_or(|(_, w)| p < w) {
                worst = Some(((from, to), p));
            }
        }
        let Some(((from, to), p)) = worst else {
            return Ok(());
        };
        let limit = self.max_valid_rho();
        Err(NegativeTransitions {
            p,
            from,
            to,
            theta: self.theta,
            rho: self.rho,
            t_max: self.t_max,
            max_rho: (limit > 0.0).then_some(limit),
        }
        .into())
    }

    /// The largest `rho` (to about 9 significant digits) at which, with the
    /// other parameters as they are, no transition probability is negative;
    /// 0 when there is none below the current `rho`. The probabilities of a
    /// recombination grow with `rho`, so they are valid below this value.
    fn max_valid_rho(&self) -> f64 {
        let mut probe = self.clone();
        let mut valid = |rho: f64| {
            probe.rho = rho;
            probe.param_recalculate().is_ok()
                && probe.p_kl.iter().all(|&p| p.is_finite() && p >= 0.0)
        };
        let (mut lo, mut hi) = (0.0, self.rho);
        for _ in 0..32 {
            let mid = 0.5 * (lo + hi);
            if valid(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    pub fn pattern_spec(&self) -> Option<&[(usize, usize)]> {
        self.pattern_spec.as_deref()
    }
}

/// The error of [`PsmcModel::check_transitions`] when transition
/// probabilities are negative: where, and how far `rho` must come down.
///
/// Its message gives the limit on `rho` and the θ/ρ ratio that reaches it,
/// in the model's own terms; a program that sets the start through options
/// of its own can build its advice from the fields instead.
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeTransitions {
    /// The most negative transition probability, from state `from` to
    /// state `to`.
    pub p: f64,
    pub from: usize,
    pub to: usize,
    /// θ, ρ and `t_max` of the model.
    pub theta: f64,
    pub rho: f64,
    pub t_max: f64,
    /// The largest ρ (to about 9 significant digits) at which, with the
    /// other parameters as they are, no transition probability is
    /// negative; `None` when there is none below the model's ρ.
    pub max_rho: Option<f64>,
}

impl NegativeTransitions {
    /// The smallest θ/ρ ratio, rounded up to 2 decimals, that brings ρ below
    /// [`max_rho`](Self::max_rho) at this θ.
    pub fn min_tr_ratio(&self) -> Option<f64> {
        self.max_rho
            .map(|limit| (self.theta / limit * 100.0).ceil() / 100.0)
    }

    /// What is wrong, without advice: the negative probabilities and, when
    /// there is one, the limit on ρ.
    pub fn finding(&self) -> String {
        let found = format!(
            "the transition probabilities are negative (down to {:.3e}, from state {} to \
             state {})",
            self.p, self.from, self.to
        );
        match self.max_rho {
            Some(limit) => format!(
                "{found}: PSMC assumes at most about one recombination per bin, and rho = {:.6} \
                 per bin is too high for it; with t_max {} and these lambdas rho must be below \
                 {limit:.6}",
                self.rho, self.t_max
            ),
            None => found,
        }
    }
}

impl std::fmt::Display for NegativeTransitions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.finding())?;
        if let Some(ratio) = self.min_tr_ratio() {
            write!(
                f,
                ". Re-bin the data with fewer base pairs per bin, or start from a smaller rho \
                 (a theta/rho ratio of {ratio} or more)"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for NegativeTransitions {}

/// See [`PsmcModel::coalescent`].
struct Coalescent {
    t: Vec<f64>,
    /// λ expanded to one value per state.
    lam: Vec<f64>,
    tau: Vec<f64>,
    alpha: Vec<f64>,
    c_pi: f64,
    c_sigma: f64,
    /// `C_pi * pi_k`: the integral of t times the coalescent density over
    /// interval k (C `cpik`).
    cpi: Vec<f64>,
    sigma: Vec<f64>,
    avg_t: Vec<f64>,
}

/// Initial per-bin θ0 from the data, as in C `psmc_new_data`:
/// θ0 = −ln(1 − K/L), where K counts heterozygous (`K`) bins and L counts
/// non-missing bins.
pub fn initial_theta(rows: &[Vec<u8>]) -> Result<f64> {
    let (n_het, n_observed) = rows
        .iter()
        .map(|row| theta_counts(row))
        .fold((0, 0), |(k, l), (rk, rl)| (k + rk, l + rl));
    theta_from_counts(n_het, n_observed)
}

/// The counts [`initial_theta`] takes from `bins`: heterozygous (`K`) bins
/// and non-missing bins.
pub(crate) fn theta_counts(bins: &[u8]) -> (usize, usize) {
    let mut n_het = 0usize;
    let mut n_observed = 0usize;
    for &v in bins {
        if v == 1 {
            n_het += 1;
        }
        if v != 2 {
            n_observed += 1;
        }
    }
    (n_het, n_observed)
}

/// [`initial_theta`] from its counts ([`theta_counts`]).
pub(crate) fn theta_from_counts(n_het: usize, n_observed: usize) -> Result<f64> {
    if n_observed == 0 {
        bail!("no observed (non-N) sites found in input");
    }
    if n_het == n_observed {
        bail!("every observed bin is heterozygous; cannot initialise theta");
    }
    Ok(-(1.0 - n_het as f64 / n_observed as f64).ln())
}

/// `n_steps` (C psmc's `n`) implied by a `--pattern` such as `"4+25*2+4+6"`:
/// the pattern lists `n_steps + 1` atomic intervals, so this is their count
/// minus one. Errors on a malformed pattern or one with fewer than 2 or more
/// than [`MAX_STATES`] intervals; [`PsmcModel::new`] rejects the same patterns.
pub fn n_steps_for_pattern(pattern: &str) -> Result<usize> {
    let n_states = n_states_of_spec(&parse_pattern(pattern)?);
    if n_states < 2 {
        bail!("pattern {pattern:?} implies {n_states} state(s); at least 2 are needed");
    }
    Ok(n_states - 1)
}

fn check_t_max(t_max: f64) -> Result<()> {
    if !(t_max > 0.0 && t_max < T_INF) {
        bail!("t_max must be in (0, {T_INF}), got {t_max}");
    }
    Ok(())
}

fn n_states_of_spec(spec: &[(usize, usize)]) -> usize {
    spec.iter().map(|(nr, gl)| nr * gl).sum()
}

fn n_free_params(n_steps: usize, spec: Option<&[(usize, usize)]>) -> usize {
    match spec {
        None => 4 + n_steps,
        Some(v) => 3 + v.iter().map(|(nr, _)| nr).sum::<usize>(),
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
            let nr: usize = a.trim().parse()?;
            let gl: usize = b.trim().parse()?;
            if nr == 0 || gl == 0 {
                bail!("pattern values must be > 0");
            }
            out.push((nr, gl));
        } else {
            // C behavior: token "L" means one free parameter repeated L times.
            let gl: usize = part.parse()?;
            if gl == 0 {
                bail!("pattern values must be > 0");
            }
            out.push((1, gl));
        }
    }
    if out.is_empty() {
        bail!("pattern is empty");
    }
    let n_states = out.iter().try_fold(0usize, |acc, &(nr, gl)| {
        nr.checked_mul(gl)?.checked_add(acc)
    });
    match n_states {
        Some(n) if n <= MAX_STATES => Ok(out),
        Some(n) => {
            bail!("pattern {pattern:?} implies {n} states; at most {MAX_STATES} are supported")
        }
        None => {
            bail!("pattern {pattern:?} implies too many states; at most {MAX_STATES} are supported")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `C_pi` from its definition, written independently of `coalescent`:
    /// the integral of the survival function `exp(-int_0^t ds / lambda(s))`.
    fn c_pi_direct(model: &PsmcModel) -> f64 {
        let t = model.compute_t(0.1);
        let lam = model.map_lam(&model.lam).unwrap();
        let mut survival = 1.0;
        let mut sum = 0.0;
        for k in 0..=model.n_steps {
            let leaving = if k == model.n_steps {
                0.0
            } else {
                survival * (-(t[k + 1] - t[k]) / lam[k]).exp()
            };
            sum += lam[k] * (survival - leaving);
            survival = leaving;
        }
        sum
    }

    #[test]
    fn normalisers_match_their_definitions() {
        let mut model =
            PsmcModel::new(15.0, 22, 0.05, 0.0125, 2.5e-8, Some("4+5*3+4".into())).unwrap();
        model
            .lam
            .copy_from_slice(&[0.4, 2.5, 0.9, 7.0, 0.05, 1.3, 30.0]);
        model.param_recalculate().unwrap();

        let c_pi = c_pi_direct(&model);
        assert!(
            (model.c_pi() - c_pi).abs() <= 1e-12 * c_pi,
            "{} vs {c_pi}",
            model.c_pi()
        );
        assert_eq!(model.c_sigma(), 1.0 / (model.c_pi() * model.rho) + 0.5);
        assert_eq!(model.n_recomb(150_947), 150_947.0 / model.c_sigma());
        // C_sigma normalises sigma.
        let total: f64 = model.sigma.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "sum of sigma = {total}");
    }

    #[test]
    fn normalisers_follow_param_recalculate() {
        let mut model = PsmcModel::new(15.0, 10, 0.05, 0.0125, 2.5e-8, None).unwrap();
        // With every lambda = 1 the survival function is exp(-t): C_pi = 1.
        assert!((model.c_pi() - 1.0).abs() < 1e-12, "{}", model.c_pi());
        let (c_pi, c_sigma) = (model.c_pi(), model.c_sigma());

        model.rho *= 2.0;
        model.lam[3] = 5.0;
        assert_eq!((model.c_pi(), model.c_sigma()), (c_pi, c_sigma));
        model.param_recalculate().unwrap();
        assert!((model.c_pi() - c_pi_direct(&model)).abs() < 1e-12 * model.c_pi());
        assert_ne!(model.c_pi(), c_pi);
        assert_eq!(model.c_sigma(), 1.0 / (model.c_pi() * model.rho) + 0.5);
    }
}
