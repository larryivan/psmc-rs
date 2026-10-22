//! Accuracy of the E-step and the log-likelihood against a reference: the
//! straightforward f64 forward-backward of psmc-rs before the checkpointed
//! engine (`e_step_sequence_no_cache` at commit ac53c53), copied below. The
//! reference sums the whole matrix of expected transition counts, site by
//! site, and its sums over each triangle's rows and columns and the
//! diagonal are compared with those the E-step sums in O(states) per site.
//!
//! Inputs cover chained rows, single-site sequences, sequences and rows whose
//! lengths straddle the checkpoint stride, all-N stretches, PSMC models of
//! several sizes (including ones that need padding) and unstructured random
//! transition matrices. With callable bases, the reference builds each bin's
//! emissions from its own count and sums the statistics of the
//! callable-length model bin by bin.

mod common;

use common::reference_hmm::{forward_fill_row, obs_to_idx, row_major};
use ndarray::Array2;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::{
    AlphaCacheBudget, CHECKPOINT_STRIDE, CallableStats, EStepOptions, EStepResult, TransitionSums,
    e_step_with_options, log_likelihood, log_likelihood_with_options,
};
use psmc_rs::io::CallableBases;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

const K: usize = CHECKPOINT_STRIDE;
const REL_TOL: f64 = 1e-12;

// ---- reference implementation (ac53c53, src/hmm.rs) ------------------------
// (its forward pass in tests/common/reference_hmm.rs)

/// The emission row of a bin of observation class `obs` with `count`
/// callable bases out of `s` (`None`: no counts), built from the bin's own
/// count as the callable-length model says: `e_T^(c / s)` for `T`, one minus
/// that for `K`.
fn bin_emission(em_rows: &[Vec<f64>; 3], obs: usize, count: Option<(u16, u16)>) -> Vec<f64> {
    match (obs, count) {
        (0 | 1, Some((c, s))) if c != s => {
            let f = f64::from(c) / f64::from(s);
            em_rows[0]
                .iter()
                .map(|&e| {
                    let t = (e.ln() * f).exp();
                    if obs == 0 { t } else { 1.0 - t }
                })
                .collect()
        }
        _ => em_rows[obs].clone(),
    }
}

/// The callable bases of every bin, row for row, and the bin size.
type Counts<'a> = Option<(&'a [Vec<u16>], u16)>;

/// The emission rows of the bins of row `r`.
fn row_emissions(
    em_rows: &[Vec<f64>; 3],
    rows: &[Vec<u8>],
    counts: Counts,
    r: usize,
) -> Vec<Vec<f64>> {
    rows[r]
        .iter()
        .enumerate()
        .map(|(t, &v)| bin_emission(em_rows, obs_to_idx(v), counts.map(|(c, s)| (c[r][t], s))))
        .collect()
}

/// The reference E-step, on bins with callable bases when `counts` are
/// given: each bin's emission is built from its own count, and the
/// statistics of the callable-length model are summed bin by bin.
fn reference_e_step(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
    counts: Counts,
) -> EStepResult {
    let n_rows = rows.len();
    let n_states = pi.len();
    let a_row = row_major(a);
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
    let mut t_weighted = vec![0.0; n_states];
    let mut k_by_count: BTreeMap<u16, Vec<f64>> = BTreeMap::new();
    let mut xi_flat = vec![0.0f64; n_states * n_states];
    let mut loglike = 0.0;

    let max_row_len = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut alpha_row_buf = vec![0.0f64; max_row_len * n_states];
    let mut c_norm_buf = vec![0.0f64; max_row_len];
    let mut tmp = vec![0.0f64; n_states];

    let mut row_prev_alpha: Vec<Option<Vec<f64>>> = vec![None; n_rows];
    let mut prev_end = vec![0.0f64; n_states];
    for (r, obs_row) in rows.iter().enumerate() {
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        if !row_starts[r] {
            row_prev_alpha[r] = Some(prev_end.clone());
        }
        let start_prev = row_prev_alpha[r].as_deref();
        let bin_em = row_emissions(&em_rows, rows, counts, r);
        let em_at = |t: usize| bin_em[t].as_slice();
        forward_fill_row(
            pi,
            &a_row,
            obs_row.len(),
            em_at,
            start_prev,
            alpha_row,
            c_norm,
            &mut tmp,
        );
        for v in c_norm.iter() {
            loglike += v.ln();
        }
        prev_end.copy_from_slice(&alpha_row[(s_max - 1) * n_states..s_max * n_states]);
    }

    let mut beta = vec![1.0f64; n_states];
    let mut beta_new = vec![0.0f64; n_states];
    let mut emit_beta = vec![0.0f64; n_states];
    for r_rev in 0..n_rows {
        let r = n_rows - 1 - r_rev;
        let obs_row = &rows[r];
        let s_max = obs_row.len();
        let alpha_row = &mut alpha_row_buf[..s_max * n_states];
        let c_norm = &mut c_norm_buf[..s_max];
        let start_prev = row_prev_alpha[r].as_deref();
        let bin_em = row_emissions(&em_rows, rows, counts, r);
        let em_at = |t: usize| bin_em[t].as_slice();
        forward_fill_row(
            pi,
            &a_row,
            obs_row.len(),
            em_at,
            start_prev,
            alpha_row,
            c_norm,
            &mut tmp,
        );
        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            let obs = obs_to_idx(obs_row[t]);
            let alpha_t = &alpha_row[t * n_states..(t + 1) * n_states];
            let count = counts.map(|(c, s)| (c[r][t], s));
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                match obs {
                    0 => gobs0[k] += g,
                    1 => gobs1[k] += g,
                    _ => gobs2[k] += g,
                }
                match (obs, count) {
                    (0, Some((c, s))) => t_weighted[k] += g * f64::from(c) / f64::from(s),
                    (1, Some((c, _))) => {
                        k_by_count.entry(c).or_insert_with(|| vec![0.0; n_states])[k] += g
                    }
                    _ => {}
                }
            }
            if t == 0 && row_starts[r] {
                continue;
            }
            let inv_norm_t = 1.0 / c_norm[t];
            let em_obs = &bin_em[t];
            for j in 0..n_states {
                emit_beta[j] = em_obs[j] * beta[j];
            }
            let alpha_prev = if t > 0 {
                &alpha_row[(t - 1) * n_states..t * n_states]
            } else {
                row_prev_alpha[r].as_deref().unwrap()
            };
            for i in 0..n_states {
                let a_i = &a_row[i * n_states..(i + 1) * n_states];
                let xi_i = &mut xi_flat[i * n_states..(i + 1) * n_states];
                let xi_scale = alpha_prev[i] * inv_norm_t;
                let mut acc = 0.0;
                for j in 0..n_states {
                    let trans = a_i[j];
                    xi_i[j] += xi_scale * trans * emit_beta[j];
                    acc += trans * emit_beta[j];
                }
                beta_new[i] = acc * inv_norm_t;
            }
            std::mem::swap(&mut beta, &mut beta_new);
        }
        if row_starts[r] {
            beta.fill(1.0);
        }
    }

    EStepResult {
        stats: psmc_rs::hmm::SufficientStats {
            gobs: [gobs0, gobs1, gobs2],
            transitions: TransitionSums::of(
                &Array2::from_shape_vec((n_states, n_states), xi_flat).unwrap(),
            ),
            callable: counts.map(|(_, s)| CallableStats {
                bin_size: usize::from(s),
                t_weighted,
                k_by_count: k_by_count.into_iter().collect(),
            }),
        },
        loglike,
    }
}

// ---- inputs -----------------------------------------------------------------

struct Case {
    name: String,
    pi: Vec<f64>,
    a: Array2<f64>,
    em: Array2<f64>,
    rows: Vec<Vec<u8>>,
    row_starts: Vec<bool>,
}

fn psmc_model(rng: &mut SmallRng, n_steps: usize, pattern: Option<&str>) -> PsmcModel {
    let theta = rng.gen_range(1e-4..5e-2);
    let rho = theta * rng.gen_range(0.05..1.0);
    let t_max = rng.gen_range(5.0..20.0);
    let mut m = PsmcModel::new(
        t_max,
        n_steps,
        theta,
        rho,
        2.5e-8,
        pattern.map(String::from),
    )
    .expect("model");
    for l in m.lam.iter_mut() {
        *l = rng.gen_range(0.1..10.0);
    }
    m.param_recalculate().expect("recalculate");
    m
}

/// A dense random model: no PSMC structure in the transition matrix.
fn random_model(rng: &mut SmallRng, n: usize) -> (Vec<f64>, Array2<f64>, Array2<f64>) {
    let mut pi: Vec<f64> = (0..n).map(|_| rng.gen_range(0.01..1.0)).collect();
    let s: f64 = pi.iter().sum();
    pi.iter_mut().for_each(|v| *v /= s);
    let mut a = Array2::from_shape_fn((n, n), |(i, j)| {
        rng.gen_range(0.001..1.0) * if i == j { 20.0 } else { 1.0 }
    });
    for mut row in a.rows_mut() {
        let s: f64 = row.sum();
        row.mapv_inplace(|v| v / s);
    }
    let mut em = Array2::zeros((3, n));
    for k in 0..n {
        let p: f64 = rng.gen_range(1e-4..0.3);
        em[(0, k)] = 1.0 - p;
        em[(1, k)] = p;
        em[(2, k)] = 1.0;
    }
    (pi, a, em)
}

fn random_obs(rng: &mut SmallRng, len: usize) -> Vec<u8> {
    let p_het = rng.gen_range(0.0..0.2);
    let mut v = Vec::with_capacity(len);
    while v.len() < len {
        if rng.gen_bool(0.02) {
            // An all-N stretch, sometimes with symbols other than 0/1/2.
            let run = rng.gen_range(1..3 * K).min(len - v.len());
            let sym = if rng.gen_bool(0.2) { 7 } else { 2 };
            v.extend(std::iter::repeat_n(sym, run));
        } else {
            v.push(u8::from(rng.gen_bool(p_het)));
        }
    }
    v
}

/// Sequence lengths that straddle the checkpoint stride, plus a few others.
fn interesting_len(rng: &mut SmallRng) -> usize {
    let choices = [
        1,
        2,
        K - 1,
        K,
        K + 1,
        2 * K - 1,
        2 * K,
        2 * K + 1,
        3 * K + 17,
    ];
    if rng.gen_bool(0.8) {
        choices[rng.gen_range(0..choices.len())]
    } else {
        rng.gen_range(1..6 * K)
    }
}

/// Split a sequence into chained rows, with cuts at, around and away from
/// segment boundaries.
fn chain_rows(rng: &mut SmallRng, seq: Vec<u8>, rows: &mut Vec<Vec<u8>>, starts: &mut Vec<bool>) {
    let mut cuts = vec![0usize];
    for c in [K - 1, K, K + 1, 2 * K, rng.gen_range(1..seq.len().max(2))] {
        if rng.gen_bool(0.5) && c > 0 && c < seq.len() {
            cuts.push(c);
        }
    }
    cuts.push(seq.len());
    cuts.sort_unstable();
    cuts.dedup();
    for (i, w) in cuts.windows(2).enumerate() {
        rows.push(seq[w[0]..w[1]].to_vec());
        starts.push(i == 0);
    }
}

fn random_data(rng: &mut SmallRng) -> (Vec<Vec<u8>>, Vec<bool>) {
    let n_seq = rng.gen_range(1..6);
    let mut rows = Vec::new();
    let mut starts = Vec::new();
    for _ in 0..n_seq {
        let len = interesting_len(rng);
        let mut seq = random_obs(rng, len);
        if rng.gen_bool(0.1) {
            seq.fill(2);
        }
        if rng.gen_bool(0.5) {
            chain_rows(rng, seq, &mut rows, &mut starts);
        } else {
            rows.push(seq);
            starts.push(true);
        }
    }
    (rows, starts)
}

fn cases() -> Vec<Case> {
    let mut rng = SmallRng::seed_from_u64(0x5eed_e57e);
    let mut out = Vec::new();
    let models: Vec<(usize, Option<&str>)> = vec![
        (1, None),
        (4, None),
        (8, None),
        (22, Some("4+5*3+4")),
        (63, Some("4+25*2+4+6")),
        (30, None),
        (300, Some("1+30*10")),
    ];
    for (mi, (n_steps, pattern)) in models.iter().enumerate() {
        for rep in 0..6 {
            let m = psmc_model(&mut rng, *n_steps, *pattern);
            let (rows, row_starts) = random_data(&mut rng);
            out.push(Case {
                name: format!("psmc model {mi} (n_steps {n_steps}) rep {rep}"),
                pi: m.prior_matrix().to_vec(),
                a: m.transition_matrix().clone(),
                em: m.emission_matrix().clone(),
                rows,
                row_starts,
            });
        }
    }
    for n in [1, 3, 9, 17] {
        for rep in 0..4 {
            let (pi, a, em) = random_model(&mut rng, n);
            let (rows, row_starts) = random_data(&mut rng);
            out.push(Case {
                name: format!("random model n={n} rep {rep}"),
                pi,
                a,
                em,
                rows,
                row_starts,
            });
        }
    }
    out
}

fn rel_err(x: f64, reference: f64) -> f64 {
    if x == reference {
        0.0
    } else {
        (x - reference).abs() / reference.abs()
    }
}

/// Relative error of a log-likelihood. An all-`N` sequence has a
/// log-likelihood of zero up to rounding (every normaliser is 1 ± 1 ulp), so
/// the error is taken relative to at least 1% of a nat per site.
fn loglike_err(x: f64, reference: f64, n_sites: usize) -> f64 {
    (x - reference).abs() / reference.abs().max(0.01 * n_sites as f64)
}

fn max_rel(xs: &[f64], refs: &[f64]) -> f64 {
    xs.iter()
        .zip(refs)
        .map(|(&x, &r)| rel_err(x, r))
        .fold(0.0, f64::max)
}

/// The relative errors of `got` against the reference `want` on `rows`:
/// its log-likelihood, expected counts and, with callable bases, the
/// statistics of the callable-length model.
fn errors(got: &EStepResult, want: &EStepResult, rows: &[Vec<u8>]) -> Vec<(&'static str, f64)> {
    let n_sites = rows.iter().map(Vec::len).sum();
    let xi = |e: &EStepResult| e.stats.transitions.parts().concat();
    let mut errs = vec![
        ("loglike", loglike_err(got.loglike, want.loglike, n_sites)),
        (
            "gobs",
            (0..3)
                .map(|o| max_rel(&got.stats.gobs[o], &want.stats.gobs[o]))
                .fold(0.0, f64::max),
        ),
        ("xi", max_rel(&xi(got), &xi(want))),
    ];
    if let (Some(g), Some(w)) = (&got.stats.callable, &want.stats.callable) {
        errs.push(("G_T", max_rel(&g.t_weighted, &w.t_weighted)));
        let g_k = g.k_by_count.iter().zip(&w.k_by_count);
        errs.push((
            "G_K",
            g_k.map(|((_, g), (_, w))| max_rel(g, w))
                .fold(0.0, f64::max),
        ));
    }
    errs
}

/// The worst relative error seen, and where.
#[derive(Default)]
struct Worst(f64, String);

impl Worst {
    /// Assert every error of `errs` is within [`REL_TOL`], and keep the worst.
    #[track_caller]
    fn check(&mut self, errs: Vec<(&str, f64)>, what: &str) {
        for (quantity, e) in errs {
            assert!(e <= REL_TOL, "{what}: {quantity} relative error {e:.3e}");
            if e > self.0 {
                *self = Worst(e, format!("{what} {quantity}"));
            }
        }
    }
}

impl std::fmt::Display for Worst {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "worst relative error {:.3e} ({})", self.0, self.1)
    }
}

/// The E-step of `case`'s model on `rows` (the case's own, or with
/// `callable` its rows as counted readers write them), with checkpoints only
/// (no cached rows) or with every row cached.
fn e_step(
    case: &Case,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
    cache_rows: bool,
) -> EStepResult {
    let budget = AlphaCacheBudget::new(if cache_rows { usize::MAX } else { 0 });
    let opts = EStepOptions {
        alpha_cache: Some(&budget),
        callable,
        ..EStepOptions::default()
    };
    e_step_with_options(&case.pi, &case.a, &case.em, rows, &case.row_starts, opts)
        .expect("E-step failed")
}

/// The E-step matches the reference, and the log-likelihood alone is the
/// reference's, and the E-step's to the bit with and without cached rows.
#[test]
fn e_step_and_log_likelihood_match_reference() {
    let mut worst = Worst::default();
    for case in cases() {
        let rows = &case.rows;
        let want = reference_e_step(&case.pi, &case.a, &case.em, rows, &case.row_starts, None);
        let got = e_step(&case, rows, None, false);
        let lens: Vec<usize> = rows.iter().map(Vec::len).collect();
        worst.check(
            errors(&got, &want, rows),
            &format!("{} (rows {lens:?})", case.name),
        );

        let ll = log_likelihood(&case.pi, &case.a, &case.em, rows, &case.row_starts)
            .expect("log-likelihood failed");
        let n_sites = lens.iter().sum();
        assert!(
            loglike_err(ll, want.loglike, n_sites) <= REL_TOL,
            "{}: {ll} vs reference {}",
            case.name,
            want.loglike
        );
        for e in [got, e_step(&case, rows, None, true)] {
            assert_eq!(ll.to_bits(), e.loglike.to_bits(), "{}", case.name);
        }
    }
    eprintln!("{worst}");
}

// ---- callable bases ---------------------------------------------------------

/// The case's rows with every symbol other than 0 and 1 written as 2 (as the
/// readers that count callable bases write them).
fn counted_rows(case: &Case) -> Vec<Vec<u8>> {
    case.rows
        .iter()
        .map(|r| r.iter().map(|&v| v.min(2)).collect())
        .collect()
}

/// Random callable bases for `rows` in bins of `s` bases that agree with
/// the symbols (N exactly when fewer than 10% are callable): full bins, the
/// fewest a called bin can have, and counts in between.
fn random_counts(rng: &mut SmallRng, rows: &[Vec<u8>], s: u16) -> Vec<Vec<u16>> {
    let min_called = s.div_ceil(10);
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|&v| {
                    if v == 2 {
                        rng.gen_range(0..min_called)
                    } else if rng.gen_bool(0.4) {
                        s
                    } else if rng.gen_bool(0.1) {
                        min_called
                    } else {
                        rng.gen_range(min_called..=s)
                    }
                })
                .collect()
        })
        .collect()
}

/// With callable bases, the E-step matches the reference that builds every
/// bin's emissions from its own count, in its log-likelihood, its expected
/// counts and the statistics of the callable-length model; the
/// log-likelihood alone is the E-step's to the bit.
#[test]
fn counted_e_step_matches_reference() {
    let mut rng = SmallRng::seed_from_u64(0xca11_ab1e);
    let mut worst = Worst::default();
    let mut partial_k = 0;
    for (i, case) in cases().into_iter().enumerate() {
        let s = [1, 7, 10, 100, 255, 1000][i % 6];
        let rows = counted_rows(&case);
        let counts = random_counts(&mut rng, &rows, s);
        let callable = CallableBases::from_counts(usize::from(s), counts.clone()).unwrap();
        let want = reference_e_step(
            &case.pi,
            &case.a,
            &case.em,
            &rows,
            &case.row_starts,
            Some((&counts, s)),
        );
        let want_c = want.stats.callable.as_ref().unwrap();
        for cache_rows in [false, true] {
            let got = e_step(&case, &rows, Some(&callable), cache_rows);
            let got_c = got.stats.callable.as_ref().expect("counted statistics");
            assert_eq!(got_c.bin_size, usize::from(s), "{}", case.name);
            let got_counts: Vec<u16> = got_c.k_by_count.iter().map(|(c, _)| *c).collect();
            let want_counts: Vec<u16> = want_c.k_by_count.iter().map(|(c, _)| *c).collect();
            assert_eq!(got_counts, want_counts, "{}: counts of K bins", case.name);
            partial_k += got_counts.iter().filter(|&&c| c != s).count();
            worst.check(
                errors(&got, &want, &rows),
                &format!("{} (s = {s})", case.name),
            );
            let opts = EStepOptions {
                callable: Some(&callable),
                ..EStepOptions::default()
            };
            let ll = log_likelihood_with_options(
                &case.pi,
                &case.a,
                &case.em,
                &rows,
                &case.row_starts,
                opts,
            )
            .unwrap();
            assert_eq!(ll.to_bits(), got.loglike.to_bits(), "{}", case.name);
        }
    }
    assert!(partial_k > 50, "only {partial_k} partly callable K counts");
    eprintln!("{worst}");
}

/// Counts that do not fit the bins are an error, not a silent misfit.
#[test]
fn counts_that_do_not_fit_the_bins_are_rejected() {
    let case = &cases()[30];
    let rows = counted_rows(case);
    let full: Vec<Vec<u16>> = rows
        .iter()
        .map(|r| r.iter().map(|&v| if v == 2 { 0 } else { 100 }).collect())
        .collect();
    let run = |counts: Vec<Vec<u16>>| {
        let callable = CallableBases::from_counts(100, counts).unwrap();
        let opts = EStepOptions {
            callable: Some(&callable),
            ..EStepOptions::default()
        };
        e_step_with_options(&case.pi, &case.a, &case.em, &rows, &case.row_starts, opts)
            .map(|_| ())
            .map_err(|e| format!("{e:#}"))
    };
    run(full.clone()).unwrap();
    let mut short = full.clone();
    short[0].pop();
    assert!(run(short).unwrap_err().contains("callable bases for"));
    let mut over = full.clone();
    over[0][0] = 101;
    assert!(run(over).unwrap_err().contains("callable bases of 100"));
    // A called bin with fewer than 10% of its bases callable.
    let mut sparse = full.clone();
    let (r, t) = (0..rows.len())
        .find_map(|r| rows[r].iter().position(|&v| v < 2).map(|t| (r, t)))
        .unwrap();
    sparse[r][t] = 9;
    assert!(run(sparse).unwrap_err().contains("9 callable bases"));
    let mut fewer_rows = full;
    fewer_rows.pop();
    if rows.len() > 1 {
        assert!(run(fewer_rows).unwrap_err().contains("rows of bins"));
    }
}
