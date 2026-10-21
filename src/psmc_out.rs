//! C psmc's `.psmc` text output, read by `psmc_plot.pl` and other tools
//! written for Heng Li's psmc.
//!
//! [`write_psmc`] writes what C psmc (lh3/psmc 0.6.5) writes for a run
//! without decoding: a `CC` block describing the format, `MM` lines with the
//! settings and the size of the data, then one block per round, for round 0
//! (the starting parameters) and each EM iteration:
//!
//! ```text
//! IT  objective evaluations of the round's M-step (not before round 0)
//! RD  round number
//! LK  log-likelihood of the round's E-step
//! QD  EM Q before -> after the M-step
//! RI  relative information sum_k sigma_k ln(sigma_k / post_sigma_k)
//! TR  theta_0 rho_0
//! MT  t_max
//! MM  C_pi: .., n_recomb: ..
//! RS  k t_k lambda_k n_recomb*sigma_k sigma_k post_sigma_k     (each state)
//! PA  pattern theta_0 rho_0 t_max lambda..                     (free parameters)
//! //
//! ```
//!
//! Round `i >= 1` reports the E-step of EM iteration `i` (`LK`, `QD`,
//! `post_sigma`, which C computes from the expectations of that E-step) and
//! the parameters its M-step produced (everything else). Round 0 has no
//! E-step: as in C, `LK` and `QD` are 0 and `post_sigma` is 0, so `RI` is
//! `inf`. Numbers are formatted as C's `printf` formats them.
//!
//! The values differ from C's only where the methods differ: the M-step
//! (L-BFGS here, Hooke-Jeeves in C) and its evaluation count (`IT`), the
//! expected emission counts (C leaves out the last bin of each sequence; this
//! moves `QD` and `RI` by a relative amount of the order of the number of
//! sequences per non-missing bin, negligible for chromosomes but percents
//! for fragmented assemblies, see
//! [`posterior_sigma`](crate::em::posterior_sigma)), and the `Version` line.
//!
//! Both M-steps run until their searches converge (ours caps them at
//! [`MStepConfig::max_iters`](crate::opt::MStepConfig::max_iters) L-BFGS
//! iterations, far above the 10 to 20 they take with C psmc's patterns). On
//! the reference data of `tests/psmc_out_c_reference.rs` the rounds then
//! agree with C's to a relative 7.3e-8 in LK, with a pattern or with one λ
//! per state; what is left comes from the expected counts, not from the
//! M-steps: with the last bin of each sequence left out of ours too, rounds
//! 2 to 5 agree to 2.8e-9, and C's own Hooke-Jeeves search in place of ours
//! changes that by less than 1e-9. Ours evaluates its objective 30 to 120
//! times per round with C psmc's patterns, C's 1,000 to 7,000.

use anyhow::{Result, bail};
use std::io::{self, Write};

use crate::em::EmHistory;
use crate::io::CallableSummary;
use crate::model::{GRID_ALPHA, PsmcModel};
use crate::opt::unpack_params;

/// The `CC` block that starts every C `.psmc` file (C psmc's cli.c).
const HEADER: &str = "CC
CC\tBrief Description of the file format:
CC\t  CC  comments
CC\t  MM  useful-messages
CC\t  RD  round-of-iterations
CC\t  LL  \\log[P(sequence)]
CC\t  QD  Q-before-opt Q-after-opt
CC\t  TR  \\theta_0 \\rho_0
CC\t  RS  k t_k \\lambda_k \\pi_k \\sum_{l\\not=k}A_{kl} A_{kk}
CC\t  DC  begin end best-k t_k+\\Delta_k max-prob
CC
";

/// The run-level values of the `MM` header lines.
#[derive(Debug, Clone, PartialEq)]
pub struct PsmcRunInfo {
    /// The `Version:` line; C prints its own version there.
    pub version: String,
    /// The parameter pattern in C's syntax (see [`c_pattern`]).
    pub pattern: String,
    /// C's `n`: the model has `n_steps + 1` states.
    pub n_steps: usize,
    /// Number of free λ (C's `n_free_lambdas`).
    pub n_free_lambdas: usize,
    /// Number of EM iterations (C's `-N`).
    pub n_iterations: usize,
    /// The starting `t_max` (C's `-t`).
    pub max_t: f64,
    /// The starting θ0/ρ0 ratio (C's `-r`).
    pub tr_ratio: f64,
    /// Number of sequences.
    pub n_seqs: usize,
    /// Number of non-missing bins (C's `sum_L`).
    pub sum_l: u64,
    /// Number of heterozygous bins (C's `sum_n`).
    pub sum_n: u64,
    /// An `MM` line of psmc-rs's own, after C's, saying how the fit used
    /// the callable bases of each bin ([`bin_callable_line`]); `None` for
    /// input without them, whose output is C's.
    pub bin_callable: Option<String>,
}

/// The text of the `MM` line ([`PsmcRunInfo::bin_callable`]) of a fit of
/// input with callable bases: the emission model fitted (`callable-length`
/// or `full-bins`), how the counts were obtained (`exact` or `approximate`,
/// when known) and what they hold, such as `bin_model:callable-length,
/// counts:exact, mean_callable:0.701234, partial_bins:0.850000`.
/// psmc_plot.pl reads only the `MM` lines it knows, so it plots the file
/// as C's.
pub fn bin_callable_line(model: &str, counts: Option<&str>, summary: &CallableSummary) -> String {
    let counts = counts.map_or(String::new(), |c| format!(", counts:{c}"));
    format!(
        "bin_model:{model}{counts}, mean_callable:{:.6}, partial_bins:{:.6}",
        summary.mean_fraction,
        summary.partial_fraction()
    )
}

impl PsmcRunInfo {
    /// The header of an EM run that produced `history` on the observations
    /// `rows` (as passed to EM). `template` supplies the layout (pattern and
    /// number of states) and may be the fitted or the starting model;
    /// `max_t` and `tr_ratio` are taken from the starting parameters
    /// `history.params[0]`, and the version is `psmc-rs <crate version>`.
    ///
    /// Sequences are counted from `row_starts`; an empty input record that
    /// never became a row is not counted (C counts it).
    pub fn from_history(
        template: &PsmcModel,
        history: &EmHistory,
        rows: &[Vec<u8>],
        row_starts: &[bool],
    ) -> Result<Self> {
        if rows.len() != row_starts.len() {
            bail!("row_starts length mismatch");
        }
        let (sum_l, sum_n) = called_and_het_bins(rows);
        let n_seqs = row_starts.iter().filter(|&&s| s).count();
        Self::from_counts(template, history, n_seqs, sum_l, sum_n)
    }

    /// [`from_history`](Self::from_history) for data already counted:
    /// `n_seqs` sequences with `sum_l` non-missing and `sum_n` heterozygous
    /// bins (for a run whose observations are no longer at hand, such as a
    /// bootstrap replicate).
    pub fn from_counts(
        template: &PsmcModel,
        history: &EmHistory,
        n_seqs: usize,
        sum_l: u64,
        sum_n: u64,
    ) -> Result<Self> {
        let Some(start) = history.params.first() else {
            bail!("EM history has no starting parameters");
        };
        if start.len() != 3 + template.lam.len() {
            bail!(
                "EM history has {} parameters, the model {}",
                start.len(),
                3 + template.lam.len()
            );
        }
        Ok(Self {
            version: format!("psmc-rs {}", env!("CARGO_PKG_VERSION")),
            pattern: c_pattern(template),
            n_steps: template.n_steps,
            n_free_lambdas: template.lam.len(),
            n_iterations: history.loglike.len(),
            max_t: start[2],
            tr_ratio: start[0] / start[1],
            n_seqs,
            sum_l,
            sum_n,
            bin_callable: None,
        })
    }
}

/// C's `sum_L` and `sum_n` of `rows`: their non-missing and their
/// heterozygous bins.
pub(crate) fn called_and_het_bins(rows: &[Vec<u8>]) -> (u64, u64) {
    let (mut sum_l, mut sum_n) = (0u64, 0u64);
    for &bin in rows.iter().flatten() {
        if bin < 2 {
            sum_l += 1;
            sum_n += u64::from(bin);
        }
    }
    (sum_l, sum_n)
}

/// The model's parameter pattern in the syntax C psmc and psmc_plot.pl
/// parse: `"4+5*3+4"` for the groups (1 x 4, 5 x 3, 1 x 4), and `"64*1"`
/// (one free λ per state) for a model with 64 states and no pattern.
pub fn c_pattern(model: &PsmcModel) -> String {
    match model.pattern_spec() {
        None => format!("{}*1", model.n_steps + 1),
        Some(spec) => spec
            .iter()
            .map(|&(n_repeats, len)| match n_repeats {
                1 => len.to_string(),
                _ => format!("{n_repeats}*{len}"),
            })
            .collect::<Vec<_>>()
            .join("+"),
    }
}

/// One `RD` block of a `.psmc` file.
#[derive(Debug, Clone, PartialEq)]
pub struct PsmcRound {
    /// The round number (0 for the starting parameters).
    pub round: usize,
    /// Objective evaluations of the round's M-step, printed as the `IT` line
    /// before `RD`; `None` for round 0, which has no M-step.
    pub mstep_evaluations: Option<usize>,
    /// Log-likelihood of the round's E-step (`LK`).
    pub loglike: f64,
    /// EM Q before and after the M-step (`QD`), see [`EmHistory::q`].
    pub q: (f64, f64),
    /// Posterior state distribution of the round's E-step (`RS` column 6).
    pub post_sigma: Vec<f64>,
    /// θ0 (`TR`).
    pub theta: f64,
    /// ρ0 (`TR`).
    pub rho: f64,
    /// `t_max` (`MT`).
    pub t_max: f64,
    /// C's `C_pi`.
    pub c_pi: f64,
    /// C's `n_recomb`.
    pub n_recomb: f64,
    /// Left boundary `t_k` of each state's interval.
    pub t: Vec<f64>,
    /// λ of each state.
    pub lambda: Vec<f64>,
    /// Initial state distribution of the round's parameters (`RS` column 5).
    pub sigma: Vec<f64>,
    /// The free λ, as stored in [`PsmcModel::lam`] (`PA`).
    pub free_lambdas: Vec<f64>,
}

impl PsmcRound {
    /// Round 0 of a run starting from `model` on `sum_l` non-missing bins:
    /// the model's parameters with no E-step (`LK` and `QD` 0, `post_sigma`
    /// all 0). The model's matrices must be up to date
    /// ([`PsmcModel::param_recalculate`]).
    pub fn initial(model: &PsmcModel, sum_l: u64) -> Result<Self> {
        let n_states = model.n_steps + 1;
        let t = model.compute_t(GRID_ALPHA);
        let lambda = model.map_lam(&model.lam)?;
        if model.sigma.len() != n_states {
            bail!(
                "model sigma has {} states, not {n_states}",
                model.sigma.len()
            );
        }
        Ok(Self {
            round: 0,
            mstep_evaluations: None,
            loglike: 0.0,
            q: (0.0, 0.0),
            post_sigma: vec![0.0; n_states],
            theta: model.theta,
            rho: model.rho,
            t_max: model.t_max,
            c_pi: model.c_pi(),
            n_recomb: model.n_recomb(sum_l),
            t: t[..n_states].to_vec(),
            lambda,
            sigma: model.sigma.clone(),
            free_lambdas: model.lam.clone(),
        })
    }

    /// C's `RI`: `sum_k sigma_k ln(sigma_k / post_sigma_k)`, the
    /// Kullback-Leibler divergence of the E-step's posterior state
    /// distribution from the new parameters' initial distribution; `inf` in
    /// round 0.
    pub fn relative_information(&self) -> f64 {
        self.sigma
            .iter()
            .zip(&self.post_sigma)
            .map(|(&s, &p)| s * (s / p).ln())
            .sum()
    }
}

/// The rounds of an EM run: round 0 at `history.params[0]` and round `i`
/// for iteration `i`, on data with `sum_l` non-missing bins. `template`
/// supplies the layout (pattern and number of states); it may be the
/// starting or the fitted model, as the parameters come from `history`.
pub fn psmc_rounds(
    template: &PsmcModel,
    history: &EmHistory,
    sum_l: u64,
) -> Result<Vec<PsmcRound>> {
    let n_iter = history.loglike.len();
    if history.params.len() != n_iter + 1
        || history.post_sigma.len() != n_iter
        || history.q.len() != n_iter
        || history.mstep_evaluations.len() != n_iter
    {
        bail!("inconsistent EM history lengths");
    }
    let mut model = template.clone();
    let mut rounds = Vec::with_capacity(n_iter + 1);
    for (i, params) in history.params.iter().enumerate() {
        unpack_params(&mut model, params)?;
        model.param_recalculate()?;
        let mut round = PsmcRound::initial(&model, sum_l)?;
        round.round = i;
        if i > 0 {
            if history.post_sigma[i - 1].len() != round.sigma.len() {
                bail!("EM history post_sigma has the wrong number of states");
            }
            round.mstep_evaluations = Some(history.mstep_evaluations[i - 1]);
            round.loglike = history.loglike[i - 1].0;
            round.q = history.q[i - 1];
            round.post_sigma = history.post_sigma[i - 1].clone();
        }
        rounds.push(round);
    }
    Ok(rounds)
}

/// Write a C psmc `.psmc` file: the header for `info`, then `rounds`.
pub fn write_psmc<W: Write>(mut w: W, info: &PsmcRunInfo, rounds: &[PsmcRound]) -> io::Result<()> {
    w.write_all(HEADER.as_bytes())?;
    writeln!(w, "MM\tVersion: {}", info.version)?;
    writeln!(
        w,
        "MM\tpattern:{}, n:{}, n_free_lambdas:{}",
        info.pattern, info.n_steps, info.n_free_lambdas
    )?;
    writeln!(
        w,
        "MM\tn_iterations:{}, skip:1, max_t:{}, theta/rho:{}",
        info.n_iterations,
        c_g(info.max_t),
        c_g(info.tr_ratio)
    )?;
    writeln!(w, "MM\tis_decoding:0")?;
    writeln!(
        w,
        "MM\tn_seqs:{}, sum_L:{}, sum_n:{}",
        info.n_seqs, info.sum_l, info.sum_n
    )?;
    if let Some(line) = &info.bin_callable {
        writeln!(w, "MM\t{line}")?;
    }
    for round in rounds {
        write_round(&mut w, &info.pattern, round)?;
    }
    w.flush()
}

/// [`write_psmc`] for an EM run: the header from
/// [`PsmcRunInfo::from_history`] and the rounds from [`psmc_rounds`].
pub fn write_psmc_from_history<W: Write>(
    w: W,
    template: &PsmcModel,
    history: &EmHistory,
    rows: &[Vec<u8>],
    row_starts: &[bool],
) -> Result<()> {
    let info = PsmcRunInfo::from_history(template, history, rows, row_starts)?;
    let rounds = psmc_rounds(template, history, info.sum_l)?;
    write_psmc(w, &info, &rounds)?;
    Ok(())
}

fn write_round<W: Write>(w: &mut W, pattern: &str, r: &PsmcRound) -> io::Result<()> {
    let n_states = r.sigma.len();
    if [r.t.len(), r.lambda.len(), r.post_sigma.len()] != [n_states; 3] {
        let msg = format!("round {}: per-state vectors differ in length", r.round);
        return Err(io::Error::new(io::ErrorKind::InvalidInput, msg));
    }
    if let Some(n) = r.mstep_evaluations {
        writeln!(w, "IT\t{n}")?;
    }
    writeln!(w, "RD\t{}", r.round)?;
    writeln!(w, "LK\t{}", fixed(r.loglike, 6))?;
    writeln!(w, "QD\t{} -> {}", fixed(r.q.0, 6), fixed(r.q.1, 6))?;
    writeln!(w, "RI\t{}", fixed(r.relative_information(), 10))?;
    writeln!(w, "TR\t{}\t{}", fixed(r.theta, 6), fixed(r.rho, 6))?;
    writeln!(w, "MT\t{}", fixed(r.t_max, 6))?;
    writeln!(
        w,
        "MM\tC_pi: {}, n_recomb: {}",
        fixed(r.c_pi, 6),
        fixed(r.n_recomb, 6)
    )?;
    for k in 0..n_states {
        writeln!(
            w,
            "RS\t{k}\t{}\t{}\t{}\t{}\t{}",
            fixed(r.t[k], 6),
            fixed(r.lambda[k], 6),
            fixed(r.n_recomb * r.sigma[k], 6),
            fixed(r.sigma[k], 6),
            fixed(r.post_sigma[k], 6)
        )?;
    }
    write!(w, "PA\t{pattern}")?;
    for &p in [r.theta, r.rho, r.t_max].iter().chain(&r.free_lambdas) {
        write!(w, " {}", fixed(p, 9))?;
    }
    writeln!(w, "\n//")
}

/// C's `printf("%.<prec>f", x)`. Rust rounds the exact binary value the same
/// way (half to even on exact ties) and spells infinities the same; only NaN
/// differs.
fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else {
        format!("{x:.prec$}")
    }
}

/// C's `printf("%g", x)`: 6 significant digits, `%e` style when the
/// exponent is below -4 or at least 6, trailing zeros removed.
fn c_g(x: f64) -> String {
    const P: i32 = 6;
    if !x.is_finite() {
        return fixed(x, 0);
    }
    // The exponent of the value rounded to P significant digits.
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    if (-4..P).contains(&exp) {
        trim_fraction(&format!("{:.*}", (P - 1 - exp) as usize, x)).to_string()
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim_fraction(mantissa), exp.abs())
    }
}

/// Drop trailing zeros after a decimal point, and the point if nothing
/// follows it.
fn trim_fraction(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C printf's output for 2,054 doubles (special values, exact ties, %g
    /// boundaries, random values between 2^-70 and 2^70), written by
    /// `tests/data/psmc_out/printf_probe.c` (see the README there).
    const PRINTF_GOLDEN: &str = include_str!("../tests/data/psmc_out/printf_golden.tsv");

    #[test]
    fn number_formats_match_c_printf() {
        let mut n = 0;
        for line in PRINTF_GOLDEN.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            let x = f64::from_bits(u64::from_str_radix(cols[0], 16).unwrap());
            assert_eq!(fixed(x, 6), cols[1], "%lf of {x:e}");
            assert_eq!(fixed(x, 9), cols[2], "%.9lf of {x:e}");
            assert_eq!(fixed(x, 10), cols[3], "%.10lf of {x:e}");
            assert_eq!(c_g(x), cols[4], "%g of {x:e}");
            n += 1;
        }
        assert_eq!(n, 2054);
    }
}
