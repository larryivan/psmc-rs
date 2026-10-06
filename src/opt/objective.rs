//! The M-step objective and its gradient, by reverse-mode (adjoint)
//! differentiation written out by hand.
//!
//! The objective is `f(x) = -Q(x) + lambda * penalty(x)`: `Q` is C psmc's
//! expected complete-data log-likelihood of the transitions and emissions
//! (with the `ln(p + 1e-300)` guard and the skipped zero counts of the
//! forward-mode code it replaces), or with callable bases that of the
//! callable-length model ([`EmissionCounts`]), at the parameters
//! `p_i = lo_i + (hi_i - lo_i) sigmoid(x_i)` (θ, ρ, t_max, then the free λ),
//! whose HMM is C `psmc_update_hmm` as [`PsmcModel::param_recalculate`]
//! ports it. The forward pass performs the operations of
//! `param_recalculate` in the same order, so the value is the one the
//! model's own matrices give, to the bit; the backward pass follows the same
//! branches (the average time's fallback to the interval's conditional mean,
//! the last interval up to `T_INF`, the guards), so the gradient is that of
//! the function actually evaluated.
//!
//! The transition probabilities of row `k` are `tmp3_k q_k[m]` (plus
//! `1 - tmp3_k` on the diagonal), where `q_k[m]` is `tmp_k q_aux[m]` below
//! the diagonal and `(alpha_m - alpha_{m+1}) tmp2_k` above it. A row
//! therefore depends on the shared quantities (α, the β sums, `q_aux`, C_pi,
//! C_sigma, ρ, the interval lengths) only through a few scalars of its own,
//! and its expected counts are known before it is computed. Each row is
//! evaluated and immediately differentiated, accumulating adjoints of the
//! shared quantities, which are then propagated back to the parameters in
//! one O(states) sweep. An evaluation thus costs O(states²), a logarithm (as
//! the value alone needs) and a division per expected transition count, and
//! O(states) memory besides the counts: the transition matrix is never
//! stored, and nothing is allocated once the workspace exists. With
//! callable bases, the `K` bins of each count add an exponential, a
//! logarithm and a division per state (a `K` bin holds `ceil(s / 10)` to
//! `s` callable bases, so there are at most `s - ceil(s / 10) + 1` counts,
//! 91 for bins of 100 bases).
//! Forward-mode differentiation, which this replaces, carried the
//! derivatives with respect to every parameter through each of the
//! states² transition probabilities, at O(states² × parameters) time and
//! memory and a heap allocation per arithmetic operation.
//!
//! The objective is `+inf` where the HMM computed in f64 is not one: where a
//! transition or emission probability is negative or not a number, or where
//! a row of the transition matrix does not sum to 1 within
//! [`ROW_SUM_TOLERANCE`]. In exact arithmetic every row sums to 1, but the
//! formulas square the survival `alpha` (C's `ak1 * ak1`): once it falls
//! below about 1e-154, which a small λ of an early interval causes (a
//! cumulative coalescence rate above about 354), those squares lose their
//! digits or vanish, rows sum to anything (1.014 was seen), and `Q` can
//! exceed its maximum `Q0`. Such a point evaluates to a finite `Q` that is
//! no HMM's, so the M-step must not move there: on a simulated genome, an
//! E-step with one lowered the log-likelihood by 367 and EM never left it.

use std::borrow::Cow;

use anyhow::{Result, bail, ensure};

use super::Bounds;
use crate::hmm::SufficientStats;
use crate::model::{PsmcModel, T_INF};

/// C psmc's `alpha` (`-l`) of the time grid, as in `compute_params_dual` and
/// [`PsmcModel::compute_t`].
const GRID_ALPHA: f64 = 0.1;

/// The guard added to each probability before its logarithm is taken.
const LN_GUARD: f64 = 1e-300;

/// The most a row of transition probabilities may deviate from summing to
/// 1 before the objective calls the point infeasible. Rows of valid HMMs
/// deviate by about 1e-12 (rounding over the states), those of points
/// whose survival has underflowed by 1e-5 to 1e-2.
pub(crate) const ROW_SUM_TOLERANCE: f64 = 1e-9;

// Calls of `Objective::eval` on this thread, for checking
// `MStepOutcome::evaluations` (the M-step runs on the calling thread), and
// the largest magnitude of a coordinate they were given.
#[cfg(test)]
thread_local! {
    pub(super) static EVAL_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static LARGEST_COORDINATE: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// The expected emission counts the objective weighs, per state: those of
/// `T` bins, of `K` bins by their share `f = c / s` of callable bases, and
/// of `N` bins. The emission part of `Q` for state `k` is
///
/// ```text
/// t[k] ln(e_T(k) + g) + sum_(f, K) K[k] ln(1 - e_T(k)^f + g) + n[k] ln(1 + g)
/// ```
///
/// with `e_T(k) = exp(-θ avg_t[k])` and the guard `g` ([`LN_GUARD`]),
/// leaving out zero counts. Without callable bases (C psmc's model) `t` is
/// `gobs[0]` and the one `K` entry is `(1, gobs[1])`; with them
/// ([`SufficientStats::callable`], `docs/design/bin-callable-model.md`)
/// `t` is the count-weighted `G_T`, since the log-probability of a `T` bin
/// of `c` callable bases, `(c / s) ln e_T(k)`, is linear in its count, and
/// there is one `K` entry per count. A count of `s` (`f = 1`) evaluates
/// exactly as C psmc's term, so input whose bins are all full gives C's
/// objective to the bit.
pub(super) struct EmissionCounts<'a> {
    pub(super) t: &'a [f64],
    pub(super) k: Vec<(f64, &'a [f64])>,
    pub(super) n: &'a [f64],
}

impl<'a> EmissionCounts<'a> {
    /// The emission counts of `stats` for a model of `states` states;
    /// errors when they do not fit it.
    pub(super) fn of(stats: &'a SufficientStats, states: usize) -> Result<Self> {
        let [g0, g1, g2] = &stats.gobs;
        ensure!(
            [g0, g1, g2].iter().all(|g| g.len() == states),
            "expected emission counts of {}, {} and {} states for a model of {states} states",
            g0.len(),
            g1.len(),
            g2.len()
        );
        let Some(c) = &stats.callable else {
            return Ok(Self {
                t: g0,
                k: vec![(1.0, g1.as_slice())],
                n: g2,
            });
        };
        ensure!(c.bin_size > 0, "callable bases of bins of 0 bases");
        ensure!(
            c.t_weighted.len() == states && c.k_by_count.iter().all(|(_, g)| g.len() == states),
            "expected counts of the callable-length model do not have {states} states"
        );
        let s = c.bin_size as f64;
        let mut k = Vec::with_capacity(c.k_by_count.len());
        for (count, g) in &c.k_by_count {
            ensure!(
                usize::from(*count) <= c.bin_size,
                "K bins with {count} callable bases, in bins of {}",
                c.bin_size
            );
            k.push((f64::from(*count) / s, g.as_slice()));
        }
        Ok(Self {
            t: &c.t_weighted,
            k,
            n: g2,
        })
    }
}

/// The M-step objective of one E-step's expected counts, with its
/// workspace.
pub(crate) struct Objective<'a> {
    /// The expected emission counts.
    emissions: EmissionCounts<'a>,
    /// `stats.xi`, row-major.
    xi: Cow<'a, [f64]>,
    bounds: &'a [Bounds],
    lambda: f64,
    /// C psmc's `n`: the model has `n + 1` states.
    n: usize,
    /// The free λ of each state: an index into the λ of the parameters.
    group: Vec<usize>,
    ws: Workspace,
    /// The most a row sum may deviate from 1 at a feasible point.
    row_tolerance: f64,
    /// The largest deviation of a row sum from 1 at the last evaluation
    /// (NaN when a probability was negative or not a number there).
    row_deviation: f64,
    /// Evaluations so far.
    pub(crate) evaluations: usize,
}

/// Values and adjoints of one evaluation, reused by the next.
struct Workspace {
    /// The parameters and their derivatives with respect to `x`.
    p: Vec<f64>,
    dp_dx: Vec<f64>,
    /// λ of each state, its logarithm and its adjoint.
    lam: Vec<f64>,
    ln_lam: Vec<f64>,
    bar_lam: Vec<f64>,
    /// `exp(beta * k)` of the time grid, for `k < n`.
    grid_exp: Vec<f64>,
    /// Interval lengths `t_{k+1} - t_k` (the last one up to `T_INF`).
    tau: Vec<f64>,
    bar_tau: Vec<f64>,
    /// Survival to `t_k` (`alpha[n + 1] = 0`) and `alpha[k] - alpha[k + 1]`,
    /// with the adjoints of their uses other than through each other.
    alpha: Vec<f64>,
    dalpha: Vec<f64>,
    bar_alpha: Vec<f64>,
    bar_dalpha: Vec<f64>,
    /// Adjoints of `-ln alpha[k]` and of the rate `tau[k] / lam[k]` at which
    /// it grows over interval k.
    bar_log_surv: Vec<f64>,
    bar_rate: Vec<f64>,
    /// C's `beta` sums.
    beta: Vec<f64>,
    bar_beta: Vec<f64>,
    /// C's `q_aux`.
    q_aux: Vec<f64>,
    bar_q_aux: Vec<f64>,
    /// Adjoint of `sum_t` (the start of interval `k`) as row `k` uses it.
    bar_sum_t: Vec<f64>,
    /// Emission probabilities of `T`, and the terms of `Q` of the `K` bins
    /// (`None` where every count is zero).
    em0: Vec<f64>,
    k_terms: Vec<Option<f64>>,
}

impl Workspace {
    fn new(n_params: usize, n: usize) -> Self {
        let states = n + 1;
        Self {
            p: vec![0.0; n_params],
            dp_dx: vec![0.0; n_params],
            lam: vec![0.0; states],
            ln_lam: vec![0.0; states],
            bar_lam: vec![0.0; states],
            grid_exp: vec![0.0; n],
            tau: vec![0.0; states],
            bar_tau: vec![0.0; states],
            alpha: vec![0.0; states + 1],
            dalpha: vec![0.0; states],
            bar_alpha: vec![0.0; states + 1],
            bar_dalpha: vec![0.0; states],
            bar_log_surv: vec![0.0; states],
            bar_rate: vec![0.0; states],
            beta: vec![0.0; states],
            bar_beta: vec![0.0; states],
            q_aux: vec![0.0; n],
            bar_q_aux: vec![0.0; n],
            bar_sum_t: vec![0.0; states],
            em0: vec![0.0; states],
            k_terms: vec![None; states],
        }
    }
}

impl<'a> Objective<'a> {
    /// The objective of `model`'s layout (its `n_steps` and pattern) on the
    /// expected counts `stats`, over the unconstrained coordinates of
    /// `bounds` (θ, ρ, t_max, then one per free λ), with penalty weight
    /// `lambda`. Errors when the counts, the bounds and the layout do not
    /// fit together.
    pub(crate) fn new(
        model: &PsmcModel,
        stats: &'a SufficientStats,
        bounds: &'a [Bounds],
        lambda: f64,
    ) -> Result<Self> {
        let n = model.n_steps;
        let states = n + 1;
        let group: Vec<usize> = match model.pattern_spec() {
            None => (0..states).collect(),
            Some(spec) => {
                let mut group = Vec::with_capacity(states);
                let mut free = 0;
                for &(repeats, length) in spec {
                    for _ in 0..repeats {
                        group.extend(std::iter::repeat_n(free, length));
                        free += 1;
                    }
                }
                group
            }
        };
        let n_free = group.last().map_or(0, |g| g + 1);
        ensure!(
            group.len() == states,
            "the pattern gives {} states for n_steps {n}",
            group.len()
        );
        ensure!(
            bounds.len() == 3 + n_free,
            "{} bounds for 3 + {n_free} free parameters",
            bounds.len()
        );
        if stats.xi.dim() != (states, states) {
            bail!(
                "expected counts of {} x {} transitions for a model of {states} states",
                stats.xi.nrows(),
                stats.xi.ncols()
            );
        }
        let emissions = EmissionCounts::of(stats, states)?;
        let xi = match stats.xi.as_slice() {
            Some(rows) => Cow::Borrowed(rows),
            None => Cow::Owned(stats.xi.iter().copied().collect()),
        };
        Ok(Self {
            emissions,
            xi,
            bounds,
            lambda,
            n,
            group,
            ws: Workspace::new(bounds.len(), n),
            row_tolerance: ROW_SUM_TOLERANCE,
            row_deviation: f64::NAN,
            evaluations: 0,
        })
    }

    /// Lets the rows of a feasible point deviate from summing to 1 by up to
    /// `tolerance` (by default [`ROW_SUM_TOLERANCE`]).
    pub(crate) fn set_row_tolerance(&mut self, tolerance: f64) {
        self.row_tolerance = tolerance;
    }

    /// The largest deviation of a row of transition probabilities from
    /// summing to 1 at the last evaluated point, feasible or not; NaN when a
    /// probability there was negative or not a number.
    pub(crate) fn row_deviation(&self) -> f64 {
        self.row_deviation
    }

    /// The objective at `x` and its gradient.
    pub(crate) fn value_and_gradient(&mut self, x: &[f64]) -> (f64, Vec<f64>) {
        let mut grad = vec![0.0; x.len()];
        let f = self.eval(x, &mut grad);
        (f, grad)
    }

    /// The objective at `x`; its gradient goes to `grad`.
    ///
    /// At an infeasible point (see the [module documentation](self)) the
    /// objective is `+inf` and the gradient NaN.
    ///
    /// # Panics
    ///
    /// If `x` or `grad` does not have one value per bound.
    pub(crate) fn eval(&mut self, x: &[f64], grad: &mut [f64]) -> f64 {
        #[cfg(test)]
        {
            EVAL_CALLS.with(|c| c.set(c.get() + 1));
            let largest = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            LARGEST_COORDINATE.with(|c| c.set(c.get().max(largest)));
        }
        self.evaluations += 1;
        assert!(
            x.len() == self.bounds.len() && grad.len() == x.len(),
            "{} coordinates and {} gradient values for {} bounds",
            x.len(),
            grad.len(),
            self.bounds.len()
        );
        let n = self.n;
        let nf = n as f64;
        let Workspace {
            p,
            dp_dx,
            lam,
            ln_lam,
            bar_lam,
            grid_exp,
            tau,
            bar_tau,
            alpha,
            dalpha,
            bar_alpha,
            bar_dalpha,
            bar_log_surv,
            bar_rate,
            beta,
            bar_beta,
            q_aux,
            bar_q_aux,
            bar_sum_t,
            em0,
            k_terms,
        } = &mut self.ws;

        // Parameters: lo + (hi - lo) * sigmoid(x).
        for (i, (&xi, b)) in x.iter().zip(self.bounds).enumerate() {
            let e = (-xi).exp();
            let denom = 1.0 + e;
            let z = 1.0 / denom;
            p[i] = z * (b.hi - b.lo) + b.lo;
            dp_dx[i] = e / (denom * denom) * (b.hi - b.lo);
        }
        let (theta, rho, t_max) = (p[0], p[1], p[2]);
        for (l, &g) in lam.iter_mut().zip(&self.group) {
            *l = p[3 + g];
        }
        let lam: &[f64] = lam;

        // ---- Forward: the quantities all rows share ----
        // The time grid (C psmc_update_intv): t[k] = 0.1 (exp(beta k) - 1)
        // for k < n, t[n] = t_max and t[n + 1] = T_INF.
        let grid_arg = 1.0 + t_max / GRID_ALPHA;
        let grid_beta = grid_arg.ln() / nf;
        for (k, e) in grid_exp.iter_mut().enumerate() {
            *e = (grid_beta * k as f64).exp();
        }
        let t = |k: usize| -> f64 {
            match k.cmp(&n) {
                std::cmp::Ordering::Less => GRID_ALPHA * (grid_exp[k] - 1.0),
                std::cmp::Ordering::Equal => t_max,
                std::cmp::Ordering::Greater => T_INF,
            }
        };
        for (k, v) in tau.iter_mut().enumerate() {
            *v = t(k + 1) - t(k);
        }
        alpha[0] = 1.0;
        for k in 1..=n {
            alpha[k] = alpha[k - 1] * (-(tau[k - 1] / lam[k - 1])).exp();
        }
        alpha[n + 1] = 0.0;
        beta[0] = 0.0;
        for k in 1..=n {
            beta[k] = beta[k - 1] + lam[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1]);
        }
        let mut c_pi = 0.0;
        for k in 0..=n {
            dalpha[k] = alpha[k] - alpha[k + 1];
            c_pi += lam[k] * dalpha[k];
        }
        let c_pi_rho = c_pi * rho;
        let c_sigma = 1.0 / c_pi_rho + 0.5;
        for m in 0..n {
            q_aux[m] = dalpha[m] * (beta[m] - lam[m] / alpha[m]) + tau[m];
        }

        // ---- Each row: its value, then the adjoints (of Q) it adds ----
        for v in [
            &mut *bar_lam,
            &mut *bar_tau,
            &mut *bar_alpha,
            &mut *bar_dalpha,
            &mut *bar_log_surv,
            &mut *bar_rate,
            &mut *bar_beta,
            &mut *bar_q_aux,
        ] {
            v.fill(0.0);
        }
        let (mut bar_theta, mut bar_rho) = (0.0, 0.0);
        let (mut bar_c_pi, mut bar_c_pi_rho, mut bar_c_sigma) = (0.0, 0.0, 0.0);
        let EmissionCounts {
            t: count_t,
            k: count_k,
            n: count_n,
        } = &self.emissions;
        let mut q_sum = 0.0;
        let mut sum_t = 0.0;
        // Whether every probability is a number in [0, 1] (for transitions,
        // not negative), and the largest deviation of a row sum from 1.
        let mut valid = true;
        let mut row_deviation = 0.0f64;
        for k in 0..=n {
            // Forward, as param_recalculate.
            let (a_k, a_next) = (alpha[k], alpha[k + 1]);
            let ak1 = a_k - a_next;
            let lak = lam[k];
            let tau_k = tau[k];
            let start_plus_lam = sum_t + lak;
            let cpik = ak1 * start_plus_lam - a_next * tau_k;
            let pik = cpik / c_pi;
            let sigma_num = ak1 / c_pi_rho + pik / 2.0;
            let sigma_k = sigma_num / c_sigma;
            let d3 = c_sigma * sigma_k;
            // C's tmp3, which is also the average time's 1 - exp(-rho avg).
            let tmp3 = pik / d3;
            let one_minus = 1.0 - tmp3;
            let mut avg = -(one_minus.ln()) / rho;
            // The fallback to the interval's conditional mean coalescent time.
            let fallback = !avg.is_finite() || avg < sum_t || avg > sum_t + tau_k;
            let (mut fb_den, mut fb_term) = (0.0, 0.0);
            if fallback {
                fb_den = a_k - a_next;
                fb_term = (tau_k * a_next) / fb_den;
                avg = sum_t + (lak - fb_term);
            }
            let tmp = ak1 / cpik;
            let a2 = ak1 * ak1;
            let b_k = beta[k] - lak / a_k;
            let qk_num = (a2 * b_k + (lak * 2.0) * ak1) - (a_next * 2.0) * tau_k;
            let qk = qk_num / cpik;
            let tmp2 = if k < n { q_aux[k] / cpik } else { 0.0 };
            let prod = theta * avg;
            let e0 = (-prod).exp();
            em0[k] = e0;
            valid &= (0.0..=1.0).contains(&e0);

            // The transitions out of state k, summed in C's order; the row's
            // sum covers the probabilities of zero counts too.
            let row = &self.xi[k * (n + 1)..(k + 1) * (n + 1)];
            let (mut bar_tmp3, mut bar_tmp, mut bar_tmp2, mut bar_qk) = (0.0, 0.0, 0.0, 0.0);
            let mut row_sum = 0.0;
            for m in 0..k {
                let q = tmp * q_aux[m];
                let p = tmp3 * q;
                row_sum += p;
                valid &= p >= 0.0;
                let c = row[m];
                if c != 0.0 {
                    let guarded = p + LN_GUARD;
                    q_sum += guarded.ln() * c;
                    let bar_p = c / guarded;
                    bar_tmp3 += bar_p * q;
                    let bar_q = bar_p * tmp3;
                    bar_tmp += bar_q * q_aux[m];
                    bar_q_aux[m] += bar_q * tmp;
                }
            }
            let p = tmp3 * qk + (1.0 - tmp3);
            row_sum += p;
            valid &= p >= 0.0;
            let c = row[k];
            if c != 0.0 {
                let guarded = p + LN_GUARD;
                q_sum += guarded.ln() * c;
                let bar_p = c / guarded;
                bar_tmp3 += bar_p * (qk - 1.0);
                bar_qk = bar_p * tmp3;
            }
            for m in (k + 1)..=n {
                let q = dalpha[m] * tmp2;
                let p = tmp3 * q;
                row_sum += p;
                valid &= p >= 0.0;
                let c = row[m];
                if c != 0.0 {
                    let guarded = p + LN_GUARD;
                    q_sum += guarded.ln() * c;
                    let bar_p = c / guarded;
                    bar_tmp3 += bar_p * q;
                    let bar_q = bar_p * tmp3;
                    bar_tmp2 += bar_q * dalpha[m];
                    bar_dalpha[m] += bar_q * tmp2;
                }
            }
            row_deviation = row_deviation.max((row_sum - 1.0).abs());

            // The emissions of T and K (that of a missing bin is 1); their
            // terms are added to Q after all transitions, as C adds them.
            // e0 = exp(-(theta avg)), so d e0 / d(theta avg) = -e0; a K
            // bin with a share f of callable bases emits with probability
            // 1 - e_f, e_f = exp(-(theta avg) f), whose derivative is f e_f.
            let mut bar_prod = 0.0;
            if count_t[k] != 0.0 {
                bar_prod -= count_t[k] * (e0 / (e0 + LN_GUARD));
            }
            let mut k_term: Option<f64> = None;
            for &(f, g) in count_k {
                let g = g[k];
                if g != 0.0 {
                    let e = (-(prod * f)).exp();
                    let em1 = 1.0 - e;
                    let term = (em1 + LN_GUARD).ln() * g;
                    k_term = Some(k_term.map_or(term, |sum| sum + term));
                    bar_prod += g * (f * e / (em1 + LN_GUARD));
                }
            }
            k_terms[k] = k_term;
            bar_theta += bar_prod * avg;
            let bar_avg = bar_prod * theta;

            // Backward through the row's scalars.
            let (mut bar_ak1, mut bar_lak, mut bar_tau_k) = (0.0, 0.0, 0.0);
            let (mut bar_cpik, mut bar_pik, mut bar_start) = (0.0, 0.0, 0.0);
            if fallback {
                // avg = sum_t + (lak - (tau_k a_next) / (a_k - a_next))
                bar_start += bar_avg;
                bar_lak += bar_avg;
                let bar_num = -bar_avg / fb_den;
                bar_tau_k += bar_num * a_next;
                bar_alpha[k + 1] += bar_num * tau_k;
                // The denominator is a_k - a_next, as ak1 is.
                bar_ak1 += bar_avg * fb_term / fb_den;
            } else {
                // avg = -ln(1 - tmp3) / rho
                bar_rho -= bar_avg * avg / rho;
                bar_tmp3 += bar_avg / rho / one_minus;
            }
            // qk = qk_num / cpik
            let bar_qk_num = bar_qk / cpik;
            bar_cpik -= bar_qk * qk / cpik;
            bar_ak1 += bar_qk_num * (2.0 * ak1 * b_k + 2.0 * lak);
            bar_beta[k] += bar_qk_num * a2;
            bar_lak += bar_qk_num * (2.0 * ak1 - a2 / a_k);
            bar_alpha[k] += bar_qk_num * a2 * (lak / a_k) / a_k;
            bar_alpha[k + 1] -= bar_qk_num * 2.0 * tau_k;
            bar_tau_k -= bar_qk_num * 2.0 * a_next;
            // tmp2 = q_aux[k] / cpik
            if k < n {
                bar_q_aux[k] += bar_tmp2 / cpik;
                bar_cpik -= bar_tmp2 * tmp2 / cpik;
            }
            // tmp = ak1 / cpik
            bar_ak1 += bar_tmp / cpik;
            bar_cpik -= bar_tmp * tmp / cpik;
            // tmp3 = pik / (c_sigma sigma_k)
            bar_pik += bar_tmp3 / d3;
            let bar_d3 = -bar_tmp3 * tmp3 / d3;
            bar_c_sigma += bar_d3 * sigma_k;
            let bar_sigma_k = bar_d3 * c_sigma;
            // sigma_k = (ak1 / (c_pi rho) + pik / 2) / c_sigma
            let bar_sigma_num = bar_sigma_k / c_sigma;
            bar_c_sigma -= bar_sigma_k * sigma_k / c_sigma;
            bar_ak1 += bar_sigma_num / c_pi_rho;
            bar_c_pi_rho -= bar_sigma_num * (ak1 / c_pi_rho) / c_pi_rho;
            bar_pik += bar_sigma_num * 0.5;
            // pik = cpik / c_pi
            bar_cpik += bar_pik / c_pi;
            bar_c_pi -= bar_pik * pik / c_pi;
            // cpik = ak1 (sum_t + lak) - a_next tau_k
            bar_ak1 += bar_cpik * start_plus_lam;
            bar_start += bar_cpik * ak1;
            bar_lak += bar_cpik * ak1;
            bar_alpha[k + 1] -= bar_cpik * tau_k;
            bar_tau_k -= bar_cpik * a_next;
            // ak1 = a_k - a_next = dalpha[k]
            bar_dalpha[k] += bar_ak1;
            bar_lam[k] += bar_lak;
            bar_tau[k] += bar_tau_k;
            bar_sum_t[k] = bar_start;

            sum_t += tau_k;
        }
        let ln_one = (1.0f64 + LN_GUARD).ln();
        for k in 0..=n {
            if count_t[k] != 0.0 {
                q_sum += (em0[k] + LN_GUARD).ln() * count_t[k];
            }
            if let Some(term) = k_terms[k] {
                q_sum += term;
            }
            if count_n[k] != 0.0 {
                q_sum += ln_one * count_n[k];
            }
        }

        self.row_deviation = if valid { row_deviation } else { f64::NAN };
        // Not a valid HMM (a NaN probability made `valid` false, so
        // `row_deviation`, whose `max` skips NaN, is a number), or a value
        // that is not finite.
        if !valid || row_deviation > self.row_tolerance || !q_sum.is_finite() {
            grad.fill(f64::NAN);
            return f64::INFINITY;
        }

        // ---- Backward through the shared quantities ----
        // c_sigma = 1 / (c_pi rho) + 1/2
        bar_c_pi_rho -= bar_c_sigma / (c_pi_rho * c_pi_rho);
        bar_c_pi += bar_c_pi_rho * rho;
        bar_rho += bar_c_pi_rho * c_pi;
        // q_aux[m] = dalpha[m] (beta[m] - lam[m] / alpha[m]) + tau[m]
        for m in 0..n {
            let bar = bar_q_aux[m];
            let (a, d) = (alpha[m], dalpha[m]);
            bar_dalpha[m] += bar * (beta[m] - lam[m] / a);
            bar_beta[m] += bar * d;
            bar_lam[m] -= bar * d / a;
            bar_alpha[m] += bar * d * (lam[m] / a) / a;
            bar_tau[m] += bar;
        }
        // c_pi = sum_k lam[k] dalpha[k]
        for k in 0..=n {
            bar_lam[k] += bar_c_pi * dalpha[k];
            bar_dalpha[k] += bar_c_pi * lam[k];
        }
        // The survival alpha[k] = exp(-L_k) with L_k = r_0 + ... + r_{k-1}
        // and r_k = tau[k] / lam[k]: the adjoints are taken with respect to
        // L and r, in which alpha[k] - alpha[k + 1] = alpha[k] (1 - exp(-r_k))
        // has the derivatives -dalpha[k] and alpha[k + 1], and the β
        // increments lam (1 / alpha[k] - 1 / alpha[k - 1]) theirs likewise.
        // Through alpha itself, the large opposite adjoints of neighbouring
        // alpha would cancel and lose precision where the survival barely
        // decays over an interval.
        // beta[k] = beta[k - 1] + lam[k - 1] (exp(L_k) - exp(L_{k-1}))
        for k in (1..=n).rev() {
            let bar = bar_beta[k];
            bar_beta[k - 1] += bar;
            let increase = 1.0 / alpha[k] - 1.0 / alpha[k - 1];
            bar_lam[k - 1] += bar * increase;
            bar_log_surv[k - 1] += bar * lam[k - 1] * increase;
            bar_rate[k - 1] += bar * lam[k - 1] / alpha[k];
        }
        // alpha[k] = exp(-L_k); dalpha[k] = exp(-L_k) (1 - exp(-r_k)), and
        // dalpha[n] = alpha[n] (alpha[n + 1] = 0 does not depend on r_n).
        for k in 0..=n {
            bar_log_surv[k] -= bar_alpha[k] * alpha[k] + bar_dalpha[k] * dalpha[k];
            if k < n {
                bar_rate[k] += bar_dalpha[k] * alpha[k + 1];
            }
        }
        // L_k = r_0 + ... + r_{k-1} (L_0 = 0)
        let mut later = 0.0;
        for k in (1..=n).rev() {
            later += bar_log_surv[k];
            bar_rate[k - 1] += later;
        }
        // r_k = tau[k] / lam[k]
        for k in 0..n {
            let l = lam[k];
            bar_tau[k] += bar_rate[k] / l;
            bar_lam[k] -= bar_rate[k] * (tau[k] / l) / l;
        }
        // The start of interval k, sum_t = tau[0] + ... + tau[k - 1]
        let mut later = 0.0;
        for k in (1..=n).rev() {
            later += bar_sum_t[k];
            bar_tau[k - 1] += later;
        }
        // tau[k] = t[k + 1] - t[k], where t[k] = 0.1 (exp(grid_beta k) - 1)
        // for k < n with grid_beta = ln(1 + t_max / 0.1) / n, t[n] = t_max and
        // t[n + 1] = T_INF. The adjoints of neighbouring intervals share
        // most of their size (the later intervals' start times), so the
        // derivative is the sum of tau's adjoints times tau's derivatives,
        // as forward mode forms it, not the difference of the adjoints.
        let d_grid_beta = 1.0 / grid_arg / GRID_ALPHA / nf;
        let dt = |k: usize| -> f64 {
            match k.cmp(&n) {
                std::cmp::Ordering::Less => GRID_ALPHA * (d_grid_beta * k as f64 * grid_exp[k]),
                std::cmp::Ordering::Equal => 1.0,
                std::cmp::Ordering::Greater => 0.0,
            }
        };
        let mut bar_t_max = 0.0;
        for (k, &bar) in bar_tau.iter().enumerate() {
            bar_t_max += bar * (dt(k + 1) - dt(k));
        }

        // ---- The smoothing penalty, and the objective ----
        for (ln, &l) in ln_lam.iter_mut().zip(lam) {
            *ln = l.ln();
        }
        let mut penalty = 0.0;
        for k in 0..n {
            let d = ln_lam[k + 1] - ln_lam[k];
            penalty += d * d;
        }
        let f = -q_sum + penalty * self.lambda;

        // df/dp = -dQ/dp + lambda dpenalty/dp, then through the sigmoids.
        grad.fill(0.0);
        grad[0] = -bar_theta;
        grad[1] = -bar_rho;
        grad[2] = -bar_t_max;
        for (k, &g) in self.group.iter().enumerate() {
            let mut d_penalty = 0.0;
            if k > 0 {
                d_penalty += 2.0 * (ln_lam[k] - ln_lam[k - 1]) / lam[k];
            }
            if k < n {
                d_penalty -= 2.0 * (ln_lam[k + 1] - ln_lam[k]) / lam[k];
            }
            grad[3 + g] += -bar_lam[k] + self.lambda * d_penalty;
        }
        for (g, d) in grad.iter_mut().zip(dp_dx.iter()) {
            *g *= d;
        }
        f
    }
}
