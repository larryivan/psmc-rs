//! EM with the callable bases of each bin (`docs/design/bin-callable-model.md`):
//!
//! - the starting θ0 solves `sum_i (1 - exp(-θ c_i / s)) = K` and is C psmc's
//!   `-ln(1 - K/L)` when every called bin is full;
//! - with every called bin full, EM is C psmc's to the bit: its history, Q
//!   and Q0;
//! - with partly called bins, EM does not lower the likelihood, `Q - Q0`
//!   stays at most 0, and on data simulated from the callable-length model
//!   the fit recovers θ, which C psmc's model underestimates by about the
//!   missing share.

use psmc_rs::em::{EmHistory, EmOptions, em_train_with_options, hmm_q, hmm_q0};
use psmc_rs::hmm::{EStepOptions, SufficientStats, e_step_with_options};
use psmc_rs::io::CallableBases;
use psmc_rs::model::ThetaCounts;
use psmc_rs::opt::MStepConfig;
use psmc_rs::{PsmcModel, initial_theta, initial_theta_with_callable};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const S: u16 = 100;
const PATTERN: &str = "4+5*3+4";

/// A PSMC model with a bottleneck-like history.
fn true_model(theta: f64) -> PsmcModel {
    let mut m = PsmcModel::new(15.0, 22, theta, theta / 4.0, 2.5e-8, Some(PATTERN.into())).unwrap();
    m.lam.copy_from_slice(&[2.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.0]);
    m.param_recalculate().unwrap();
    m
}

fn cumulative(p: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut acc = 0.0;
    p.into_iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect()
}

fn draw(cum: &[f64], u: f64) -> usize {
    let total = *cum.last().unwrap();
    cum.partition_point(|&c| c < u * total).min(cum.len() - 1)
}

/// The hidden states of `n_seqs` sequences of `len` bins drawn from
/// `model`'s HMM.
fn hidden_states(model: &PsmcModel, n_seqs: usize, len: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = model.n_steps + 1;
    let start = cumulative(model.sigma.iter().copied());
    let trans: Vec<Vec<f64>> = (0..n)
        .map(|k| cumulative(model.p_kl.row(k).iter().copied()))
        .collect();
    (0..n_seqs)
        .map(|_| {
            let mut k = draw(&start, rng.r#gen());
            (0..len)
                .map(|i| {
                    if i > 0 {
                        k = draw(&trans[k], rng.r#gen());
                    }
                    k
                })
                .collect()
        })
        .collect()
}

/// Bins of the hidden `states`, each with a count of callable bases drawn
/// by `count` (N when fewer than 10%) and a heterozygous site with the
/// probability of the callable-length model, `1 - exp(-θ (c / s)
/// avg_t[k])`.
fn observe(
    model: &PsmcModel,
    states: &[Vec<usize>],
    seed: u64,
    mut count: impl FnMut(&mut SmallRng) -> u16,
) -> (Vec<Vec<u8>>, CallableBases) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let avg = model.avg_t();
    let (mut rows, mut counts) = (Vec::new(), Vec::new());
    for seq in states {
        let (mut row, mut cs) = (Vec::with_capacity(seq.len()), Vec::with_capacity(seq.len()));
        for &k in seq {
            let c = count(&mut rng);
            let p_het = -(-model.theta * avg[k] * f64::from(c) / f64::from(S)).exp_m1();
            let het = rng.gen_bool(p_het);
            row.push(if 10 * c < S { 2 } else { u8::from(het) });
            cs.push(c);
        }
        rows.push(row);
        counts.push(cs);
    }
    let callable = CallableBases::from_counts(usize::from(S), counts).unwrap();
    (rows, callable)
}

/// [`observe`] of fresh hidden states.
fn simulate(
    model: &PsmcModel,
    n_seqs: usize,
    len: usize,
    seed: u64,
    count: impl FnMut(&mut SmallRng) -> u16,
) -> (Vec<Vec<u8>>, CallableBases) {
    let states = hidden_states(model, n_seqs, len, seed);
    observe(model, &states, seed ^ 0x5eed, count)
}

/// Dispersed missing data: each base of a bin missing with probability
/// `missing`, independently.
fn dispersed(missing: f64) -> impl FnMut(&mut SmallRng) -> u16 {
    move |rng| (0..S).filter(|_| !rng.gen_bool(missing)).count() as u16
}

fn config() -> MStepConfig {
    MStepConfig {
        progress: false,
        ..MStepConfig::default()
    }
}

fn fit(
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
    n_iter: usize,
) -> (PsmcModel, EmHistory) {
    let theta0 = initial_theta_with_callable(rows, callable).unwrap();
    let mut model =
        PsmcModel::new(15.0, 22, theta0, theta0 / 4.0, 2.5e-8, Some(PATTERN.into())).unwrap();
    let starts = vec![true; rows.len()];
    let opts = EmOptions {
        callable,
        ..EmOptions::default()
    };
    let history =
        em_train_with_options(&mut model, rows, &starts, n_iter, &config(), opts).unwrap();
    (model, history)
}

/// Every value of an EM history as bits.
fn history_bits(h: &EmHistory) -> Vec<u64> {
    let mut v = Vec::new();
    for &(a, b) in &h.loglike {
        v.extend([a.to_bits(), b.to_bits()]);
    }
    for p in h.params.iter().chain(&h.post_sigma) {
        v.extend(p.iter().map(|x| x.to_bits()));
    }
    for &(a, b) in &h.q {
        v.extend([a.to_bits(), b.to_bits()]);
    }
    v.extend(h.mstep_evaluations.iter().map(|&n| n as u64));
    v
}

#[test]
fn theta0_is_c_psmcs_for_full_bins_and_solves_the_moment_equation() {
    let mut rng = SmallRng::seed_from_u64(1);
    let bins: Vec<u8> = (0..20_000)
        .map(|_| match rng.gen_range(0..100) {
            0..=2 => 2,
            3..=7 => 1,
            _ => 0,
        })
        .collect();
    let plain = initial_theta(std::slice::from_ref(&bins)).unwrap();
    let with = |counts: Vec<u16>| {
        let callable = CallableBases::from_counts(100, vec![counts]).unwrap();
        initial_theta_with_callable(std::slice::from_ref(&bins), Some(&callable)).unwrap()
    };
    let n_count = |v: u8| if v == 2 { 3 } else { S };
    // Without counts, and with every called bin full, C psmc's value.
    assert_eq!(
        initial_theta_with_callable(std::slice::from_ref(&bins), None)
            .unwrap()
            .to_bits(),
        plain.to_bits()
    );
    assert_eq!(
        with(bins.iter().map(|&v| n_count(v)).collect()).to_bits(),
        plain.to_bits()
    );
    // One count c for every called bin: C's value per callable base.
    let half = with(bins.iter().map(|&v| if v == 2 { 0 } else { 50 }).collect());
    assert_eq!(half, plain / 0.5);
    // Mixed counts: the root of the moment equation, between the values of
    // the largest and the smallest share.
    let counts: Vec<u16> = bins
        .iter()
        .map(|&v| if v == 2 { 5 } else { rng.gen_range(10..=100) })
        .collect();
    let theta = with(counts.clone());
    let (mut expected_k, mut k) = (0.0, 0.0);
    for (&v, &c) in bins.iter().zip(&counts) {
        if v < 2 {
            expected_k += -(-theta * f64::from(c) / 100.0).exp_m1();
            k += f64::from(v);
        }
    }
    assert!((expected_k - k).abs() <= 1e-12 * k, "{expected_k} vs {k}");
    assert!(plain < theta && theta < plain / 0.1);
    // Dividing C's value by the mean share starts lower: a mix of shares
    // shows fewer K bins than bins of the mean share would.
    let called: Vec<f64> = bins
        .iter()
        .zip(&counts)
        .filter(|(v, _)| **v < 2)
        .map(|(_, &c)| f64::from(c) / 100.0)
        .collect();
    let mean_share = called.iter().sum::<f64>() / called.len() as f64;
    assert!(theta > plain / mean_share);

    // Counts of parts add up to those of the whole.
    let callable = CallableBases::from_counts(100, vec![counts]).unwrap();
    let c = &callable.rows()[0];
    let whole = ThetaCounts::of_counted(&bins, c).unwrap();
    let mut parts = ThetaCounts::new(Some(100));
    for (a, b) in [(0, 7000), (7000, 7001), (7001, 20_000)] {
        let part: Vec<u16> = c.to_vec()[a..b].to_vec();
        let part = CallableBases::from_counts(100, vec![part]).unwrap();
        parts.add(&ThetaCounts::of_counted(&bins[a..b], &part.rows()[0]).unwrap());
    }
    assert_eq!(parts, whole);
    assert_eq!(parts.theta0().unwrap().to_bits(), theta.to_bits());
}

#[test]
fn theta0_recovers_theta_under_one_coalescence_time() {
    // Every bin at t = 1: P(K | c) = 1 - exp(-θ c / s).
    let mut rng = SmallRng::seed_from_u64(2);
    let theta = 0.05;
    let mut count = dispersed(0.4);
    let (mut bins, mut counts) = (Vec::new(), Vec::new());
    for _ in 0..400_000 {
        let c = count(&mut rng);
        let k = rng.gen_bool(-(-theta * f64::from(c) / 100.0).exp_m1());
        bins.push(if 10 * c < 100 { 2 } else { u8::from(k) });
        counts.push(c);
    }
    let callable = CallableBases::from_counts(100, vec![counts]).unwrap();
    let rows = vec![bins];
    let counted = initial_theta_with_callable(&rows, Some(&callable)).unwrap();
    let plain = initial_theta(&rows).unwrap();
    assert!((counted / theta - 1.0).abs() < 0.02, "{counted}");
    assert!((plain / theta - 0.6).abs() < 0.02, "{plain}");
}

#[test]
fn full_bins_fit_as_c_psmc_to_the_bit() {
    let truth = true_model(0.02);
    let (rows, partial) = simulate(&truth, 4, 3000, 3, dispersed(0.3));
    // The same bins, every called one full.
    let counts = rows
        .iter()
        .zip(partial.to_counts())
        .map(|(r, c)| {
            r.iter()
                .zip(c)
                .map(|(&v, c)| if v == 2 { c } else { S })
                .collect()
        })
        .collect();
    let full = CallableBases::from_counts(100, counts).unwrap();
    let (m_plain, h_plain) = fit(&rows, None, 3);
    let (m_full, h_full) = fit(&rows, Some(&full), 3);
    assert_eq!(history_bits(&h_full), history_bits(&h_plain));
    assert_eq!(m_full.theta.to_bits(), m_plain.theta.to_bits());

    // Q and Q0 of an E-step with full counts are C's, to the bit.
    let starts = vec![true; rows.len()];
    let e = |callable| -> SufficientStats {
        e_step_with_options(
            m_plain.prior_matrix(),
            m_plain.transition_matrix(),
            m_plain.emission_matrix(),
            &rows,
            &starts,
            EStepOptions {
                callable,
                ..EStepOptions::default()
            },
        )
        .unwrap()
        .stats
    };
    let (plain, with) = (e(None), e(Some(&full)));
    let q = |s: &SufficientStats| hmm_q(m_plain.transition_matrix(), m_plain.emission_matrix(), s);
    assert_eq!(q(&plain).to_bits(), q(&with).to_bits());
    assert_eq!(hmm_q0(&plain).to_bits(), hmm_q0(&with).to_bits());
}

#[test]
fn em_with_partly_called_bins_raises_the_likelihood_and_keeps_q_below_q0() {
    let truth = true_model(0.02);
    let (rows, callable) = simulate(&truth, 4, 5000, 4, dispersed(0.3));
    let (_, h) = fit(&rows, Some(&callable), 6);
    for &(before, after) in &h.loglike {
        assert!(after >= before - 1e-9 * before.abs(), "{before} -> {after}");
    }
    for &(before, after) in &h.q {
        assert!(before <= 1e-9 && after <= 1e-9, "QD {before} -> {after}");
        assert!(
            after >= before - 1e-9 * before.abs(),
            "QD {before} -> {after}"
        );
    }
}

/// Q0 with counts is the largest Q over each state's emission parameter:
/// above Q at any parameter, and equal to Q at the best one found by a
/// golden-section search.
#[test]
fn q0_with_counts_is_the_maximum_over_the_emissions() {
    let truth = true_model(0.03);
    let (rows, callable) = simulate(&truth, 2, 4000, 5, dispersed(0.4));
    let starts = vec![true; rows.len()];
    let stats = e_step_with_options(
        truth.prior_matrix(),
        truth.transition_matrix(),
        truth.emission_matrix(),
        &rows,
        &starts,
        EStepOptions {
            callable: Some(&callable),
            ..EStepOptions::default()
        },
    )
    .unwrap()
    .stats;
    let c = stats.callable.as_ref().unwrap();
    assert!(c.k_by_count.len() > 20);
    let q0 = hmm_q0(&stats);
    // The emission part of state k's Q at e_T = exp(-mu).
    let h = |k: usize, mu: f64| -> f64 {
        let mut v = -c.t_weighted[k] * mu;
        for (n, g) in &c.k_by_count {
            if g[k] != 0.0 {
                v += g[k] * (-(-mu * f64::from(*n) / 100.0).exp_m1()).ln();
            }
        }
        v
    };
    let mut best = 0.0;
    for k in 0..c.t_weighted.len() {
        // Golden-section search for the maximum of the concave h on a log scale.
        let (mut a, mut b) = ((1e-9f64).ln(), (10.0f64).ln());
        let g = (5f64.sqrt() - 1.0) / 2.0;
        for _ in 0..200 {
            let (x1, x2) = (b - g * (b - a), a + g * (b - a));
            if h(k, x1.exp()) < h(k, x2.exp()) {
                a = x1;
            } else {
                b = x2;
            }
        }
        best += h(k, (0.5 * (a + b)).exp());
    }
    // The transitions' part, at the normalised counts.
    for row in stats.xi.rows() {
        let total: f64 = row.sum();
        for &x in row {
            if x > 0.0 {
                best += x * (x / total).ln();
            }
        }
    }
    assert!((q0 - best).abs() <= 1e-9 * q0.abs(), "{q0} vs {best}");
    // Q at the model is below.
    let q = hmm_q(truth.transition_matrix(), truth.emission_matrix(), &stats);
    assert!(q < q0);
}

/// The expected heterozygosity of a full bin under `m`'s coalescent, θ
/// times the mean coalescence time `C_pi`: what the data identify of θ (θ
/// alone trades off with the scale of the λ).
fn diversity(m: &PsmcModel) -> f64 {
    m.theta * m.c_pi()
}

/// On data simulated from the callable-length model with 30% of the bases
/// missing at random, the fit with the counts recovers the diversity per
/// bin of the fit of the same genealogy without missing data, and C psmc's
/// model takes the missing bases for absent heterozygosity: its diversity
/// is low by about the missing share.
#[test]
fn counts_recover_diversity_that_c_psmcs_model_underestimates() {
    let truth = true_model(0.02);
    let states = hidden_states(&truth, 20, 25_000, 6);
    let (complete, _) = observe(&truth, &states, 7, |_| S);
    let (rows, callable) = observe(&truth, &states, 8, dispersed(0.3));
    let (reference, _) = fit(&complete, None, 20);
    let (counted, _) = fit(&rows, Some(&callable), 20);
    let (plain, _) = fit(&rows, None, 20);
    let (rc, rp) = (
        diversity(&counted) / diversity(&reference),
        diversity(&plain) / diversity(&reference),
    );
    eprintln!(
        "diversity / complete data: with counts {rc:.4}, without {rp:.4} (complete data vs \
         truth {:.4})",
        diversity(&reference) / diversity(&truth)
    );
    assert!((rc - 1.0).abs() < 0.03, "with counts: {rc}");
    assert!((rp - 0.7).abs() < 0.03, "without counts: {rp}");
}
