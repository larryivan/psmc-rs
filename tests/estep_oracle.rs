//! Accuracy of the E-step against a reference: the
//! straightforward f64 forward-backward of psmc-rs before the checkpointed
//! engine (`e_step_sequence_no_cache` at commit ac53c53), copied below.
//!
//! Inputs cover chained rows, single-site sequences, sequences and rows whose
//! lengths straddle the checkpoint stride, all-N stretches, PSMC models of
//! several sizes (including ones that need padding) and unstructured random
//! transition matrices.

use ndarray::Array2;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::{
    AlphaCacheBudget, CHECKPOINT_STRIDE, EStepOptions, EStepResult, e_step_with_options,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const K: usize = CHECKPOINT_STRIDE;
const REL_TOL: f64 = 1e-12;

// ---- reference implementation (ac53c53, src/hmm.rs) ------------------------

fn obs_to_idx(v: u8) -> usize {
    if v == 0 {
        0
    } else if v == 1 {
        1
    } else {
        2
    }
}

fn matvec_row_major(src: &[f64], a_row: &[f64], dst: &mut [f64]) {
    let n_states = src.len();
    dst.fill(0.0);
    for i in 0..n_states {
        let s = src[i];
        let row = &a_row[i * n_states..(i + 1) * n_states];
        for k in 0..n_states {
            dst[k] += s * row[k];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_fill_row(
    pi: &[f64],
    a_row: &[f64],
    em_rows: &[Vec<f64>; 3],
    obs_row: &[u8],
    start_prev: Option<&[f64]>,
    alpha_row: &mut [f64],
    c_norm: &mut [f64],
    tmp: &mut [f64],
) {
    let n_states = pi.len();
    for t in 0..obs_row.len() {
        let em_obs = &em_rows[obs_to_idx(obs_row[t])];
        let mut norm = 0.0f64;
        if t == 0 && start_prev.is_none() {
            for k in 0..n_states {
                let v = pi[k] * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        } else {
            let src = if t == 0 {
                start_prev.unwrap()
            } else {
                &alpha_row[(t - 1) * n_states..t * n_states]
            };
            matvec_row_major(src, a_row, tmp);
            for k in 0..n_states {
                let v = tmp[k] * em_obs[k];
                tmp[k] = v;
                norm += v;
            }
        }
        assert!(norm > 0.0);
        c_norm[t] = norm;
        for k in 0..n_states {
            alpha_row[t * n_states + k] = tmp[k] / norm;
        }
    }
}

fn reference_e_step(
    pi: &[f64],
    a: &Array2<f64>,
    em: &Array2<f64>,
    rows: &[Vec<u8>],
    row_starts: &[bool],
) -> EStepResult {
    let n_rows = rows.len();
    let n_states = pi.len();
    let a_row: Vec<f64> = a.iter().copied().collect();
    let em_rows = [em.row(0).to_vec(), em.row(1).to_vec(), em.row(2).to_vec()];

    let mut gobs0 = vec![0.0; n_states];
    let mut gobs1 = vec![0.0; n_states];
    let mut gobs2 = vec![0.0; n_states];
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
        forward_fill_row(
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
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
        forward_fill_row(
            pi, &a_row, &em_rows, obs_row, start_prev, alpha_row, c_norm, &mut tmp,
        );
        for t_rev in 0..s_max {
            let t = s_max - 1 - t_rev;
            let obs = obs_to_idx(obs_row[t]);
            let alpha_t = &alpha_row[t * n_states..(t + 1) * n_states];
            for k in 0..n_states {
                let g = alpha_t[k] * beta[k];
                match obs {
                    0 => gobs0[k] += g,
                    1 => gobs1[k] += g,
                    _ => gobs2[k] += g,
                }
            }
            if t == 0 && row_starts[r] {
                continue;
            }
            let inv_norm_t = 1.0 / c_norm[t];
            let em_obs = &em_rows[obs];
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
            xi: Array2::from_shape_vec((n_states, n_states), xi_flat).unwrap(),
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

fn n_sites(case: &Case) -> usize {
    case.rows.iter().map(Vec::len).sum()
}

fn max_rel(xs: impl IntoIterator<Item = f64>, refs: impl IntoIterator<Item = f64>) -> f64 {
    xs.into_iter()
        .zip(refs)
        .map(|(x, r)| rel_err(x, r))
        .fold(0.0, f64::max)
}

/// E-step with checkpoints only (no cached rows) or with every row cached.
fn e_step(case: &Case, cache_rows: bool) -> EStepResult {
    let budget = AlphaCacheBudget::new(if cache_rows { usize::MAX } else { 0 });
    let opts = EStepOptions {
        alpha_cache: Some(&budget),
        ..EStepOptions::default()
    };
    e_step_with_options(
        &case.pi,
        &case.a,
        &case.em,
        &case.rows,
        &case.row_starts,
        opts,
    )
    .expect("E-step failed")
}

#[test]
fn e_step_matches_reference() {
    let mut worst = (0.0f64, String::new());
    for case in cases() {
        let got = e_step(&case, false);
        let want = reference_e_step(&case.pi, &case.a, &case.em, &case.rows, &case.row_starts);
        let errs = [
            (
                "loglike",
                loglike_err(got.loglike, want.loglike, n_sites(&case)),
            ),
            (
                "gobs",
                (0..3)
                    .map(|o| max_rel(got.stats.gobs[o].clone(), want.stats.gobs[o].clone()))
                    .fold(0.0, f64::max),
            ),
            (
                "xi",
                max_rel(got.stats.xi.iter().copied(), want.stats.xi.iter().copied()),
            ),
        ];
        for (what, e) in errs {
            assert!(
                e <= REL_TOL,
                "{}: {what} relative error {e:.3e} (rows {:?})",
                case.name,
                case.rows.iter().map(Vec::len).collect::<Vec<_>>()
            );
            if e > worst.0 {
                worst = (e, format!("{} {what}", case.name));
            }
        }
    }
    eprintln!("worst relative error {:.3e} ({})", worst.0, worst.1);
}
