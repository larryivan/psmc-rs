//! The EM driver: alternate a full forward-backward E-step with an L-BFGS M-step.

use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::ThreadPool;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::hmm::{
    Cancelled, EStepOptions, EStepPhase, EStepProgress, SufficientStats, e_step_with_options,
    log_likelihood_with_options,
};
use crate::io::CallableBases;
use crate::model::PsmcModel;
use crate::opt::{
    MStepConfig, MStepProgress, MStepStop, bounds_from_config, m_step_lbfgs_with_outcome,
    pack_params, unpack_params,
};
use crate::progress;

#[derive(Debug)]
pub struct EmHistory {
    /// Per iteration: (log-likelihood of the parameters entering it, of the
    /// parameters leaving it). The second value of iteration `i` equals the
    /// first of iteration `i + 1`; the last one is the log-likelihood of the
    /// final parameters.
    pub loglike: Vec<(f64, f64)>,
    /// Parameters before the first iteration and after each one.
    pub params: Vec<Vec<f64>>,
    /// Per iteration: the posterior state distribution over the non-missing
    /// bins of its E-step ([`posterior_sigma`]), C psmc's `post_sigma`.
    pub post_sigma: Vec<Vec<f64>>,
    /// Per iteration: C psmc's EM Q, [`hmm_q`] minus [`hmm_q0`] of its
    /// E-step's expectations, at the parameters entering and leaving its
    /// M-step (C's `QD` line). Both are at most 0. The smoothing penalty
    /// the M-step also weighs (`MStepConfig::lambda`) is not included.
    pub q: Vec<(f64, f64)>,
    /// Per iteration: evaluations of the M-step objective (C's `IT` line).
    pub mstep_evaluations: Vec<usize>,
    /// Per iteration: L-BFGS iterations of the M-step
    /// ([`MStepOutcome::iterations`](crate::opt::MStepOutcome::iterations)).
    pub mstep_iterations: Vec<usize>,
    /// Per iteration: why the M-step stopped.
    pub mstep_stops: Vec<MStepStop>,
}

impl EmHistory {
    /// Log-likelihood of the final parameters (the model EM returns), or
    /// `None` after zero iterations. This is the second value of the last
    /// iteration; the first is that of the parameters entering it.
    pub fn final_loglike(&self) -> Option<f64> {
        self.loglike.last().map(|&(_, after)| after)
    }
}

/// C psmc's `hmm_Q` before it subtracts `Q0`: the expected complete-data
/// log-likelihood of the emissions of symbols 0 (`T`) and 1 (`K`) and of the
/// transitions,
///
/// `sum_k (E_0[k] ln e_0[k] + E_1[k] ln e_1[k]) + sum_{k,l} A[k][l] ln a[k][l]`,
///
/// with the expected counts `E = stats.gobs` and `A = stats.xi` and the
/// probabilities `e = emission` and `a = transition`. Like C, it leaves out
/// the missing symbol (whose emission probability is 1) and the initial
/// state. It is `-inf` when any of these probabilities is not positive, where
/// C returns `-1e300`.
///
/// With callable bases ([`SufficientStats::callable`]) the emission terms
/// are those of the callable-length model, with the emissions the E-step
/// used:
///
/// `sum_k (G_T[k] ln e_0[k] + sum_c G_K[c][k] ln(1 - e_0[k]^(c / s)))`,
///
/// which with every count `s` is C's expression, to the bit.
pub fn hmm_q(transition: &Array2<f64>, emission: &Array2<f64>, stats: &SufficientStats) -> f64 {
    q_sum(transition, emission, stats).unwrap_or(f64::NEG_INFINITY)
}

/// [`hmm_q`], or `None` where a probability is not positive.
fn q_sum(transition: &Array2<f64>, emission: &Array2<f64>, stats: &SufficientStats) -> Option<f64> {
    // A NaN probability is let through: it makes the sum NaN.
    let term = |count: f64, p: f64| if p <= 0.0 { None } else { Some(count * p.ln()) };
    let mut sum = 0.0;
    match &stats.callable {
        None => {
            for (b, counts) in stats.gobs[..2].iter().enumerate() {
                for (k, &count) in counts.iter().enumerate() {
                    sum += term(count, emission[(b, k)])?;
                }
            }
        }
        Some(c) => {
            for (k, &count) in c.t_weighted.iter().enumerate() {
                sum += term(count, emission[(0, k)])?;
            }
            let s = c.bin_size as f64;
            for (n, counts) in &c.k_by_count {
                let f = f64::from(*n) / s;
                for (k, &count) in counts.iter().enumerate() {
                    // The E-step's emission of a K bin of this count.
                    let p = if usize::from(*n) == c.bin_size {
                        emission[(1, k)]
                    } else {
                        1.0 - (emission[(0, k)].ln() * f).exp()
                    };
                    sum += term(count, p)?;
                }
            }
        }
    }
    for (&count, &p) in stats.xi.iter().zip(transition) {
        sum += term(count, p)?;
    }
    Some(sum)
}

/// C psmc's `hmm_Q0`: the largest value [`hmm_q`] takes for the expected
/// counts `stats` over all emission and transition probabilities, reached at
/// `e_b[k] = E_b[k] / (E_0[k] + E_1[k])` and `a[k][l] = A[k][l] / sum_l A[k][l]`.
/// C's Q is `hmm_q - hmm_q0`, so it is at most 0.
///
/// A zero count contributes 0, the limit of `x ln x`; C instead adds 1e-25
/// to every count, which changes the value by less than 1e-20. A negative
/// count, which only an E-step with negative transition probabilities
/// gives (EM refuses to start from those), makes it NaN, as in C.
///
/// With callable bases the emission probabilities of a state are those of
/// the callable-length model, one free parameter `e_0[k]` per state, so
/// that `Q - Q0` stays at most 0: its emission part is the largest value of
/// `G_T[k] ln e + sum_c G_K[c][k] ln(1 - e^(c / s))` over `e` in (0, 1),
/// which is concave in `-ln e` and found by Newton's method. When every `K`
/// bin is full the maximum has C's closed form, with `G_T` and `G_K` in
/// place of `E_0` and `E_1`, so input whose bins are all full gives C's
/// value to the bit.
pub fn hmm_q0(stats: &SufficientStats) -> f64 {
    fn x_ln_x_over(x: f64, total: f64) -> f64 {
        if x == 0.0 { 0.0 } else { x * (x / total).ln() }
    }
    let mut sum = 0.0;
    match &stats.callable {
        None => {
            let [e0, e1, _] = &stats.gobs;
            for (&c0, &c1) in e0.iter().zip(e1) {
                let total = c0 + c1;
                sum += x_ln_x_over(c0, total) + x_ln_x_over(c1, total);
            }
        }
        Some(c) => {
            let s = c.bin_size as f64;
            for (k, &t) in c.t_weighted.iter().enumerate() {
                // The K counts of this state by share of callable bases.
                let partial: Vec<(f64, f64)> = c
                    .k_by_count
                    .iter()
                    .filter(|(n, g)| usize::from(*n) != c.bin_size && g[k] != 0.0)
                    .map(|(n, g)| (f64::from(*n) / s, g[k]))
                    .collect();
                let full: f64 = c
                    .k_by_count
                    .iter()
                    .filter(|(n, _)| usize::from(*n) == c.bin_size)
                    .map(|(_, g)| g[k])
                    .sum();
                sum += if partial.is_empty() {
                    let total = t + full;
                    x_ln_x_over(t, total) + x_ln_x_over(full, total)
                } else {
                    max_callable_emission_q(t, full, &partial)
                };
            }
        }
    }
    for row in stats.xi.rows() {
        let total: f64 = row.sum();
        for &count in row {
            sum += x_ln_x_over(count, total);
        }
    }
    sum
}

/// The largest value over `mu > 0` of `h(mu) = -t mu + full ln(1 - e^-mu)
/// + sum_(f, g) g ln(1 - e^(-f mu))`: the emission part of one state's `Q`
/// in the callable-length model, with `e_0 = e^-mu`, for `t` the
/// count-weighted `T` sum, `full` the `K` sum of full bins and `partial` the
/// `K` sums of partly callable bins by their share `f` of callable bases
/// (`0 < f < 1`, at least one positive sum).
///
/// `h` is concave, with `h'` falling from `+inf` near 0 to `-t`, so for `t
/// > 0` the maximum is at the root of `h'`, found by Newton's method from
/// the left, where it climbs monotonically; without `T` counts `h` rises to
/// its supremum 0 as `mu` grows.
fn max_callable_emission_q(t: f64, full: f64, partial: &[(f64, f64)]) -> f64 {
    if t == 0.0 {
        return 0.0;
    }
    let terms = || std::iter::once((1.0, full)).chain(partial.iter().copied());
    let h = |mu: f64| -> f64 {
        let mut v = -t * mu;
        for (f, g) in terms() {
            if g != 0.0 {
                v += g * (-(-f * mu).exp_m1()).ln();
            }
        }
        v
    };
    // h'(mu) = -t + sum g f / (e^(f mu) - 1), h''(mu) = -sum g f² e^(f mu) / (e^(f mu) - 1)².
    let slope = |mu: f64| -> (f64, f64) {
        let (mut d1, mut d2) = (-t, 0.0);
        for (f, g) in terms() {
            if g != 0.0 {
                let em1 = (f * mu).exp_m1();
                d1 += g * f / em1;
                d2 -= g * f * f * (em1 + 1.0) / (em1 * em1);
            }
        }
        (d1, d2)
    };
    // A start left of the root: with every share raised to 1, C's closed
    // form, whose h' is no larger than ours.
    let k_total: f64 = terms().map(|(_, g)| g).sum();
    let mut mu = (1.0 + k_total / t).ln();
    for _ in 0..200 {
        let (d1, d2) = slope(mu);
        if d1.is_nan() || d1 <= 0.0 {
            break;
        }
        let step = -d1 / d2;
        mu += step;
        if step.is_nan() || step <= 1e-15 * mu {
            break;
        }
    }
    h(mu)
}

/// C psmc's `post_sigma`: the posterior distribution of the hidden state over
/// the non-missing bins of an E-step, `(E_0[k] + E_1[k]) / sum_l (E_0[l] + E_1[l])`.
///
/// The expected emission counts here cover every bin, while C's `hmm_expect`
/// leaves out the last bin of each sequence. That is the only difference in
/// the expectations behind C's `post_sigma`, `hmm_Q` and `hmm_Q0`: removing
/// those bins' posteriors from `stats.gobs` reproduces C's first-round
/// `post_sigma` and `QD` values to all 6 printed decimals, on the 3
/// sequences of `tests/data/c_reference` and on 1,010 sequences of 1 to 80
/// bins (`tests/em_c_reference.rs`).
///
/// The size of the difference follows the share of the data in those bins.
/// With `D` sequences ending in a non-missing bin and `sum_L` non-missing
/// bins, each value differs from C's by at most `D / sum_L`, and the `QD`
/// and `RI` of the round by a relative amount of the order of `D / sum_L`
/// (0.5 to 2 times it, measured). For a few chromosome-length sequences this
/// is negligible: 3 sequences of 150,947 bins in all (`D / sum_L` = 2e-5)
/// move `post_sigma` by at most 2.4e-6 and `QD` by 4e-5. For a fragmented
/// assembly it is not: on the 1,010 short sequences (`D / sum_L` = 2.5%)
/// `QD` differs from C's by 3.5% and `RI` by 1.2%, so `psmc_plot.pl -n 0`,
/// which plots the round with the smallest `RI` from round `-m` on, can pick
/// a different round than for C's output.
pub fn posterior_sigma(stats: &SufficientStats) -> Vec<f64> {
    let [e0, e1, _] = &stats.gobs;
    let counts: Vec<f64> = e0.iter().zip(e1).map(|(c0, c1)| c0 + c1).collect();
    let total: f64 = counts.iter().sum();
    counts.into_iter().map(|c| c / total).collect()
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

/// Options for [`em_train_with_options`].
#[derive(Default)]
pub struct EmOptions<'a> {
    /// Called on the calling thread at the start of each iteration and with
    /// E-step and M-step progress.
    pub on_progress: Option<&'a mut dyn FnMut(EmProgressEvent)>,
    /// Checked before each iteration, inside the E-step (between sequences
    /// and every [`CHECKPOINT_STRIDE`](crate::hmm::CHECKPOINT_STRIDE) sites),
    /// between the E- and M-steps, and before the final log-likelihood; once
    /// set, training stops with a [`Cancelled`] error.
    pub cancel: Option<&'a AtomicBool>,
    /// Run the E-steps and the final log-likelihood in this pool instead of
    /// the current one.
    pub thread_pool: Option<&'a ThreadPool>,
    /// The callable bases of each bin, row for row
    /// ([`crate::io::Observations::callable`]): fit the callable-length
    /// model of `docs/design/bin-callable-model.md`, in which a bin of `c`
    /// of `s` callable bases has the `T` emission of a full bin to the power
    /// `c / s` ([`EStepOptions::callable`]). `None` (the default) fits C
    /// psmc's model; so do counts whose called bins are all full, to the
    /// bit.
    pub callable: Option<&'a CallableBases>,
}

/// [`em_train_with_options`] with the default options.
pub fn em_train(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
) -> Result<EmHistory> {
    em_train_with_options(
        model,
        rows,
        row_starts,
        n_iter,
        config,
        EmOptions::default(),
    )
}

fn cancelled_in(iter: usize, err: anyhow::Error) -> anyhow::Error {
    if err.is::<Cancelled>() {
        err.context(format!("EM cancelled in iteration {iter}"))
    } else {
        err
    }
}

/// Fit `model` by `n_iter` EM iterations (E-step, then L-BFGS M-step).
///
/// Returns the log-likelihood and parameter history (see [`EmHistory`]).
/// On cancellation the model holds the parameters of the last completed
/// iteration (with its matrices recalculated) and the error is
/// [`Cancelled`] (with the iteration as context).
///
/// Errors before the first E-step when the starting parameters give
/// negative transition probabilities ([`PsmcModel::check_transitions`]),
/// from which EM cannot move.
pub fn em_train_with_options(
    model: &mut PsmcModel,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    n_iter: usize,
    config: &MStepConfig,
    opts: EmOptions<'_>,
) -> Result<EmHistory> {
    let EmOptions {
        mut on_progress,
        cancel,
        thread_pool,
        callable,
    } = opts;
    let check_cancel = |iter: usize| -> Result<()> {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(cancelled_in(iter, Cancelled.into()));
        }
        Ok(())
    };
    let mut history = EmHistory {
        loglike: Vec::with_capacity(n_iter),
        params: Vec::with_capacity(n_iter + 1),
        post_sigma: Vec::with_capacity(n_iter),
        q: Vec::with_capacity(n_iter),
        mstep_evaluations: Vec::with_capacity(n_iter),
        mstep_iterations: Vec::with_capacity(n_iter),
        mstep_stops: Vec::with_capacity(n_iter),
    };
    let mut params = pack_params(model);
    history.params.push(params.clone());

    let bounds = bounds_from_config(model, config);
    let pb_em =
        (config.progress && n_iter > 0).then(|| progress::bar(n_iter as u64, "EM", "overall"));

    if n_iter > 0 {
        model.param_recalculate()?;
        // From negative transition probabilities every M-step would return
        // its start: fail before the first E-step instead of returning that
        // start as a fit.
        model
            .check_transitions()
            .context("EM cannot start from these parameters")?;
    }
    for iter in 1..=n_iter {
        check_cancel(iter)?;
        if let Some(cb) = on_progress.as_mut() {
            cb(EmProgressEvent::IterStart {
                iter,
                total_iters: n_iter,
            });
        }
        if let Some(pb) = &pb_em {
            pb.set_message(format!("overall {iter}/{n_iter}"));
        }
        let mut on_e_step = on_progress.as_mut().map(|cb| {
            move |ev: EStepProgress| {
                cb(EmProgressEvent::EStep {
                    iter,
                    total_iters: n_iter,
                    phase: ev.phase,
                    done: ev.done,
                    total: ev.total,
                })
            }
        });
        let e_opts = EStepOptions {
            progress_bars: config.progress,
            label: "E",
            on_progress: on_e_step
                .as_mut()
                .map(|cb| cb as &mut dyn FnMut(EStepProgress)),
            cancel,
            thread_pool,
            alpha_cache: None,
            callable,
        };
        let e = e_step_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            rows,
            row_starts,
            e_opts,
        )
        .map_err(|err| cancelled_in(iter, err))?;
        // This E-step evaluates the parameters the previous iteration left.
        if let Some(prev) = history.loglike.last_mut() {
            prev.1 = e.loglike;
        }
        history.loglike.push((e.loglike, f64::NAN));
        let stats = e.stats;
        let q0 = hmm_q0(&stats);
        let q_before = hmm_q(model.transition_matrix(), model.emission_matrix(), &stats) - q0;
        check_cancel(iter)?;
        let mut on_m_step = on_progress.as_mut().map(|cb| {
            move |ev: MStepProgress| {
                cb(EmProgressEvent::MStep {
                    iter,
                    total_iters: n_iter,
                    done: ev.done,
                    total: ev.total,
                })
            }
        });
        let outcome = m_step_lbfgs_with_outcome(
            model,
            &stats,
            &params,
            &bounds,
            config,
            on_m_step
                .as_mut()
                .map(|cb| cb as &mut dyn FnMut(MStepProgress)),
        )?;
        params = outcome.params;
        unpack_params(model, &params)?;
        // Keep the model's matrices in step with its parameters, also when
        // a later cancellation ends training early.
        model.param_recalculate()?;
        history.params.push(params.clone());
        let q_after = hmm_q(model.transition_matrix(), model.emission_matrix(), &stats) - q0;
        history.post_sigma.push(posterior_sigma(&stats));
        history.q.push((q_before, q_after));
        history.mstep_evaluations.push(outcome.evaluations);
        history.mstep_iterations.push(outcome.iterations);
        history.mstep_stops.push(outcome.stop);

        if let Some(pb) = &pb_em {
            pb.inc(1);
        }
    }
    if let Some(last) = history.loglike.last_mut() {
        // Only the final parameters have not been evaluated yet: a forward
        // pass suffices.
        check_cancel(n_iter)?;
        let ll_opts = EStepOptions {
            cancel,
            thread_pool,
            callable,
            ..EStepOptions::default()
        };
        last.1 = log_likelihood_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            rows,
            row_starts,
            ll_opts,
        )
        .map_err(|err| cancelled_in(n_iter, err))?;
    }
    if let Some(pb) = pb_em {
        pb.finish_with_message("EM done");
    }

    Ok(history)
}
