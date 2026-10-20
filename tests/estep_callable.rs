//! The per-bin passes with callable bases (`docs/design/bin-callable-model.md`):
//!
//! - bins that are all full give exactly, to the bit, the results of the
//!   same bins without counts, in the E-step, the log-likelihood and the
//!   TMRCA decoding, with any batching;
//! - bins that all have the same count `c` are the model whose `T` emission
//!   is the full bin's to the power `c / s`, to the bit;
//! - with mixed counts, results do not depend on the thread count, nesting
//!   in other rayon work, the alpha-cache budget or the batching.
//!
//! `tests/estep_oracle.rs` checks the counted E-step against a brute-force
//! reference.

mod common;

use common::{TempDir, full_bins, model, pool};
use ndarray::Array2;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::{
    AlphaCacheBudget, CHECKPOINT_STRIDE, EStepOptions, EStepResult, TmrcaOptions,
    e_step_with_options, log_likelihood_with_options, write_tmrca_posterior_tsv_with_options,
};
use psmc_rs::io::common::chunk_sequences;
use psmc_rs::io::{CallableBases, Observations, Sequences};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const S: u16 = 100;

struct Model {
    pi: Vec<f64>,
    a: Array2<f64>,
    em: Array2<f64>,
}

fn psmc(seed: u64) -> (Model, PsmcModel) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut m = model("4+25*2+4+6", 3e-3, 6e-4);
    for l in m.lam.iter_mut() {
        *l = rng.gen_range(0.2..5.0);
    }
    m.param_recalculate().unwrap();
    let model = Model {
        pi: m.prior_matrix().to_vec(),
        a: m.transition_matrix().clone(),
        em: m.emission_matrix().clone(),
    };
    (model, m)
}

/// The PSMC model with its transition matrix perturbed (rows renormalised),
/// so that the E-step uses the dense kernels and caches forward rows when
/// its budget allows.
fn dense(seed: u64) -> Model {
    let (mut m, _) = psmc(seed);
    let mut rng = SmallRng::seed_from_u64(seed ^ 3);
    for mut row in m.a.rows_mut() {
        row.mapv_inplace(|v| v * (1.0 + 1e-3 * rng.gen_range(0.0..1.0)));
        let s: f64 = row.sum();
        row.mapv_inplace(|v| v / s);
    }
    m
}

/// Sequences of `lens` bins with callable bases: stretches of dispersed
/// missing data (bins of 40 to 100 callable bases), missing bins (N, 0 to
/// 9), and full bins elsewhere.
fn sequences(seed: u64, lens: &[usize]) -> Sequences {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut bins = Vec::new();
    let mut counts = Vec::new();
    for &len in lens {
        let mut b = Vec::with_capacity(len);
        let mut c = Vec::with_capacity(len);
        while b.len() < len {
            let run = rng.gen_range(1..300).min(len - b.len());
            let kind = rng.gen_range(0..10);
            for _ in 0..run {
                let (count, het) = match kind {
                    0 => (rng.gen_range(0..10), None),
                    1..=4 => {
                        let c = rng.gen_range(40..=S);
                        (c, Some(0.004 * f64::from(c) / 100.0))
                    }
                    _ => (S, Some(0.004)),
                };
                let symbol = match het {
                    None => 2,
                    Some(p) => u8::from(rng.gen_bool(p * 2.0)),
                };
                b.push(symbol);
                c.push(count);
            }
        }
        bins.push(b);
        counts.push(c);
    }
    Sequences {
        bins,
        callable: Some(CallableBases::from_counts(usize::from(S), counts).unwrap()),
    }
}

fn lens() -> Vec<usize> {
    vec![
        9000,
        4 * CHECKPOINT_STRIDE + 3,
        1,
        15000,
        700,
        2 * CHECKPOINT_STRIDE,
    ]
}

/// The sequences cut into rows of at most `batch` bins.
fn observations(seqs: &Sequences, batch: Option<usize>) -> Observations {
    chunk_sequences(seqs.clone(), batch, "empty").unwrap()
}

/// The same bins with every called bin full (N bins keep their counts).
fn all_full(seqs: &Sequences) -> Sequences {
    Sequences {
        bins: seqs.bins.clone(),
        callable: Some(full_bins(&seqs.bins, seqs.callable.as_ref().unwrap())),
    }
}

fn e_step(m: &Model, obs: &Observations, opts: EStepOptions<'_>) -> EStepResult {
    e_step_with_options(&m.pi, &m.a, &m.em, &obs.rows, &obs.row_starts, opts).unwrap()
}

fn counted(m: &Model, obs: &Observations) -> EStepResult {
    e_step(
        m,
        obs,
        EStepOptions {
            callable: obs.callable.as_ref(),
            ..EStepOptions::default()
        },
    )
}

/// Every value of an E-step result as bits, its counted statistics included.
fn bits(e: &EStepResult) -> Vec<u64> {
    let mut v = vec![e.loglike.to_bits()];
    for g in &e.stats.gobs {
        v.extend(g.iter().map(|x| x.to_bits()));
    }
    v.extend(e.stats.xi.iter().map(|x| x.to_bits()));
    if let Some(c) = &e.stats.callable {
        v.push(c.bin_size as u64);
        v.extend(c.t_weighted.iter().map(|x| x.to_bits()));
        for (count, g) in &c.k_by_count {
            v.push(u64::from(*count));
            v.extend(g.iter().map(|x| x.to_bits()));
        }
    }
    v
}

/// The TMRCA TSV of `m` on `obs` (with its counts when `counted`).
fn tmrca(m: &Model, years: &[f64], obs: &Observations, counted: bool) -> Vec<u8> {
    let dir = TempDir::new("estep_callable_tmrca");
    let path = dir.path("tmrca.tsv");
    let opts = TmrcaOptions {
        progress_bars: false,
        max_points: 100,
        callable: if counted { obs.callable.as_ref() } else { None },
    };
    write_tmrca_posterior_tsv_with_options(
        &m.pi,
        &m.a,
        &m.em,
        &obs.rows,
        &obs.row_starts,
        years,
        &path,
        opts,
    )
    .unwrap();
    std::fs::read(&path).unwrap()
}

#[test]
fn full_bins_give_the_results_of_input_without_counts() {
    let (m, model) = psmc(1);
    let years = model.state_tmrca_years(100.0, 25.0);
    let seqs = all_full(&sequences(2, &lens()));
    for batch in [None, Some(1000), Some(CHECKPOINT_STRIDE + 1)] {
        let obs = observations(&seqs, batch);
        let plain = e_step(&m, &obs, EStepOptions::default());
        let with = counted(&m, &obs);
        let mut want = bits(&plain);
        // The counted statistics are the plain ones: G_T is gobs[0], and the
        // one count of the K bins is s, with gobs[1].
        let c = with.stats.callable.as_ref().unwrap();
        assert_eq!(c.k_by_count.len(), 1, "{batch:?}");
        want.push(u64::from(S));
        want.extend(plain.stats.gobs[0].iter().map(|x| x.to_bits()));
        want.push(u64::from(S));
        want.extend(plain.stats.gobs[1].iter().map(|x| x.to_bits()));
        assert_eq!(bits(&with), want, "{batch:?}");

        let ll = |callable| {
            log_likelihood_with_options(
                &m.pi,
                &m.a,
                &m.em,
                &obs.rows,
                &obs.row_starts,
                EStepOptions {
                    callable,
                    ..EStepOptions::default()
                },
            )
            .unwrap()
        };
        assert_eq!(ll(None).to_bits(), ll(obs.callable.as_ref()).to_bits());
        assert_eq!(ll(None).to_bits(), plain.loglike.to_bits());
        assert!(
            tmrca(&m, &years, &obs, true) == tmrca(&m, &years, &obs, false),
            "TMRCA output differs, batch {batch:?}"
        );
    }
}

#[test]
fn a_uniform_count_is_the_model_of_thinned_emissions() {
    let (m, model) = psmc(3);
    let years = model.state_tmrca_years(100.0, 25.0);
    let seqs = sequences(4, &lens());
    for c in [10u16, 37, 99] {
        let f = f64::from(c) / f64::from(S);
        let counts = seqs
            .bins
            .iter()
            .map(|b| b.iter().map(|&v| if v == 2 { 0 } else { c }).collect())
            .collect();
        let uniform = Sequences {
            bins: seqs.bins.clone(),
            callable: Some(CallableBases::from_counts(usize::from(S), counts).unwrap()),
        };
        let obs = observations(&uniform, Some(2000));
        let mut thinned = m.em.clone();
        for k in 0..thinned.ncols() {
            let t = (m.em[(0, k)].ln() * f).exp();
            thinned[(0, k)] = t;
            thinned[(1, k)] = 1.0 - t;
        }
        let as_plain = Model {
            pi: m.pi.clone(),
            a: m.a.clone(),
            em: thinned,
        };
        let plain = e_step(&as_plain, &obs, EStepOptions::default());
        let with = counted(&m, &obs);
        let mut want = bits(&plain);
        want.push(u64::from(S));
        // G_T = (c / s) gobs[0], one product per state.
        want.extend(plain.stats.gobs[0].iter().map(|x| (f * x).to_bits()));
        want.push(u64::from(c));
        want.extend(plain.stats.gobs[1].iter().map(|x| x.to_bits()));
        assert_eq!(bits(&with), want, "count {c}");
        assert!(
            tmrca(&m, &years, &obs, true) == tmrca(&as_plain, &years, &obs, false),
            "TMRCA output differs, count {c}"
        );
        // Fewer callable bases make heterozygous bins likelier to come from
        // old states, so the counted fit sees the data differently.
        assert_ne!(
            with.loglike,
            counted(&m, &observations(&seqs, None)).loglike
        );
    }
}

#[test]
fn counted_results_do_not_depend_on_threads_nesting_or_batching() {
    let (m, _) = psmc(5);
    let seqs = sequences(6, &lens());
    let obs = observations(&seqs, None);
    let reference = bits(&counted(&m, &obs));
    assert!(counted(&m, &obs).stats.callable.unwrap().k_by_count.len() > 10);
    for batch in [Some(1000), Some(CHECKPOINT_STRIDE - 1)] {
        assert_eq!(
            bits(&counted(&m, &observations(&seqs, batch))),
            reference,
            "batch {batch:?}"
        );
    }
    for threads in [1, 2, 8] {
        let p = pool(threads);
        assert_eq!(
            bits(&p.install(|| counted(&m, &obs))),
            reference,
            "{threads} threads"
        );
        let nested: Vec<Vec<u64>> = p.install(|| {
            (0..3)
                .into_par_iter()
                .map(|_| bits(&counted(&m, &obs)))
                .collect()
        });
        assert!(nested.iter().all(|b| *b == reference), "{threads} nested");
        // An explicit pool and a progress callback.
        let mut events = 0;
        let mut cb = |_| events += 1;
        let e = e_step(
            &m,
            &obs,
            EStepOptions {
                on_progress: Some(&mut cb),
                thread_pool: Some(&p),
                callable: obs.callable.as_ref(),
                ..EStepOptions::default()
            },
        );
        assert!(events >= 2);
        assert_eq!(bits(&e), reference, "{threads} threads, explicit pool");
    }
}

#[test]
fn counted_results_do_not_depend_on_the_alpha_cache_budget() {
    let m = dense(7);
    let seqs = sequences(8, &lens());
    let obs = observations(&seqs, Some(5000));
    let mut seen = Vec::new();
    for limit in [0, 8 * 64 * 800, usize::MAX] {
        let budget = AlphaCacheBudget::new(limit);
        for threads in [1, 4] {
            let e = pool(threads).install(|| {
                e_step(
                    &m,
                    &obs,
                    EStepOptions {
                        alpha_cache: Some(&budget),
                        callable: obs.callable.as_ref(),
                        ..EStepOptions::default()
                    },
                )
            });
            if limit == usize::MAX {
                assert!(budget.peak_bytes() > 0, "rows were never cached");
            }
            seen.push(bits(&e));
        }
    }
    assert!(seen.iter().all(|b| *b == seen[0]));
}

#[test]
fn counted_tmrca_decoding_does_not_depend_on_threads_or_batching() {
    let (m, model) = psmc(9);
    let years = model.state_tmrca_years(100.0, 25.0);
    let seqs = sequences(10, &lens());
    let obs = observations(&seqs, None);
    let reference = tmrca(&m, &years, &obs, true);
    assert!(reference != tmrca(&m, &years, &obs, false));
    for threads in [1, 3, 8] {
        let out =
            pool(threads).install(|| tmrca(&m, &years, &observations(&seqs, Some(777)), true));
        assert!(out == reference, "{threads} threads");
    }
}
