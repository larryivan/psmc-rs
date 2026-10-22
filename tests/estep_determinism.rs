//! The E-step must give bit-identical results however it is run on one
//! build of the HMM kernels: any thread count, top level or nested inside
//! rayon work (`ThreadPool::install`, a parallel iterator), any alpha-cache
//! budget, with or without a progress callback. On x86-64 the portable and
//! the AVX2 and FMA kernels must agree to a relative 1e-12, and to the bit
//! with the dense kernels.

mod common;

use common::{densify, model, pool};
use ndarray::Array2;
use psmc_rs::hmm::{
    AlphaCacheBudget, CHECKPOINT_STRIDE, EStepOptions, EStepResult, SimdPath, TmrcaOptions,
    TmrcaReportData, e_step_streaming, e_step_with_options, log_likelihood, set_force_portable,
    simd_path, write_tmrca_posterior_tsv_with_options,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::{RwLock, RwLockReadGuard};

/// Held for reading by every test of this file but [`simd_paths_agree`],
/// which switches the kernels of the whole process and holds it for
/// writing: each test runs all its passes on one build of the kernels.
static ONE_PATH: RwLock<()> = RwLock::new(());

fn one_path() -> RwLockReadGuard<'static, ()> {
    ONE_PATH.read().unwrap_or_else(|e| e.into_inner())
}

struct Input {
    pi: Vec<f64>,
    a: Array2<f64>,
    em: Array2<f64>,
    rows: Vec<Vec<u8>>,
    row_starts: Vec<bool>,
}

/// `lens[i]` sites for sequence `i`, every other sequence split into chained rows.
fn input(seed: u64, pattern: &str, lens: &[usize]) -> Input {
    let mut rng = SmallRng::seed_from_u64(seed);
    let model = model(pattern, 2e-3, 4e-4);
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();
    for (i, &len) in lens.iter().enumerate() {
        let seq: Vec<u8> = (0..len)
            .map(|_| match rng.gen_range(0..100) {
                0 => 1,
                1..=4 => 2,
                _ => 0,
            })
            .collect();
        if i % 2 == 1 && len > 3 * CHECKPOINT_STRIDE {
            let cut1 = CHECKPOINT_STRIDE + 1;
            let cut2 = len / 2;
            for (j, part) in [&seq[..cut1], &seq[cut1..cut2], &seq[cut2..]]
                .iter()
                .enumerate()
            {
                rows.push(part.to_vec());
                row_starts.push(j == 0);
            }
        } else {
            rows.push(seq);
            row_starts.push(true);
        }
    }
    Input {
        pi: model.prior_matrix().to_vec(),
        a: model.transition_matrix().clone(),
        em: model.emission_matrix().clone(),
        rows,
        row_starts,
    }
}

/// The log-likelihood, the expected emissions and the expected transitions.
fn values(e: &EStepResult) -> Vec<f64> {
    let mut v = vec![e.loglike];
    for g in &e.stats.gobs {
        v.extend(g.iter());
    }
    v.extend(e.stats.transitions.parts().concat());
    v
}

fn bits(e: &EStepResult) -> Vec<u64> {
    values(e).iter().map(|x| x.to_bits()).collect()
}

/// Assert two results are bit-identical, reporting the first difference.
#[track_caller]
fn assert_same(got: &EStepResult, want: &[u64], what: &str) {
    let got = bits(got);
    assert_eq!(got.len(), want.len(), "{what}: shape differs");
    let diffs = got.iter().zip(want).filter(|(a, b)| a != b).count();
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: {diffs} of {} values differ; first at index {i} (0 = loglike): {} vs {}",
            want.len(),
            f64::from_bits(got[i]),
            f64::from_bits(want[i])
        );
    }
}

fn run(inp: &Input) -> EStepResult {
    e_step_streaming(
        &inp.pi,
        &inp.a,
        &inp.em,
        &inp.rows,
        &inp.row_starts,
        false,
        "E",
    )
    .unwrap()
}

fn run_with(inp: &Input, opts: EStepOptions<'_>) -> EStepResult {
    e_step_with_options(&inp.pi, &inp.a, &inp.em, &inp.rows, &inp.row_starts, opts).unwrap()
}

fn log_likelihood_of(inp: &Input) -> f64 {
    log_likelihood(&inp.pi, &inp.a, &inp.em, &inp.rows, &inp.row_starts).unwrap()
}

fn standard_input() -> Input {
    input(
        11,
        "4+25*2+4+6",
        &[
            9000,
            4 * CHECKPOINT_STRIDE + 3,
            1,
            15000,
            700,
            2 * CHECKPOINT_STRIDE,
        ],
    )
}

/// The standard input with a transition matrix of the dense kernels.
fn dense_input() -> Input {
    let mut inp = standard_input();
    densify(&mut inp.a, 3);
    inp
}

#[test]
fn thread_count_and_nesting_do_not_change_results() {
    let _path = one_path();
    for inp in [standard_input(), dense_input()] {
        check_threads_and_nesting(&inp);
    }
}

fn check_threads_and_nesting(inp: &Input) {
    let reference = bits(&run(inp));
    for threads in [1, 2, 8] {
        let p = pool(threads);
        let installed = p.install(|| run(inp));
        assert_same(
            &installed,
            &reference,
            &format!("{threads} threads via install"),
        );
        // Nested inside a parallel iterator, as bootstrap replicates call it.
        let nested: Vec<EStepResult> =
            p.install(|| (0..3).into_par_iter().map(|_| run(inp)).collect());
        for e in &nested {
            assert_same(
                e,
                &reference,
                &format!("{threads} threads nested in par_iter"),
            );
        }
    }
}

/// A sequence of about a hundred segments beside a short one, which leaves
/// the other threads of the pool nothing to do while it runs.
#[test]
fn a_long_sequence_gives_the_same_bits_on_any_thread_count() {
    let _path = one_path();
    let inp = input(17, "4+25*2+4+6", &[300, 96 * CHECKPOINT_STRIDE + 1000]);
    let reference = bits(&pool(1).install(|| run(&inp)));
    for threads in [2, 3, 4, 8, 16] {
        let e = pool(threads).install(|| run(&inp));
        assert_same(&e, &reference, &format!("{threads} threads"));
    }
}

#[test]
fn many_sequences_are_reduced_deterministically() {
    let _path = one_path();
    // More sequences than work items, so sequences are grouped.
    let lens: Vec<usize> = (0..700).map(|i| 1 + (i * 37) % 300).collect();
    let inp = input(5, "4+5*3+4", &lens);
    let reference = bits(&run(&inp));
    for threads in [1, 3, 8] {
        let e = pool(threads).install(|| run(&inp));
        assert_same(&e, &reference, &format!("{threads} threads"));
    }
}

#[test]
fn alpha_cache_budget_does_not_change_results() {
    let _path = one_path();
    // Rows are cached only with the dense kernels, so use a dense model.
    let inp = dense_input();
    let one_row = 8 * 64 * 800; // enough for the short sequences only
    let mut seen = Vec::new();
    for limit in [0, 1, one_row, usize::MAX] {
        let budget = AlphaCacheBudget::new(limit);
        for threads in [1, 4] {
            let e = pool(threads).install(|| {
                let opts = EStepOptions {
                    alpha_cache: Some(&budget),
                    ..EStepOptions::default()
                };
                run_with(&inp, opts)
            });
            assert_eq!(budget.used_bytes(), 0, "reservations must be released");
            assert!(budget.peak_bytes() <= limit);
            if limit == usize::MAX {
                assert!(budget.peak_bytes() > 0, "full rows were never cached");
            }
            seen.push((limit, threads, e));
        }
    }
    let reference = bits(&seen[0].2);
    for (limit, threads, e) in &seen {
        assert_same(
            e,
            &reference,
            &format!("budget {limit} with {threads} threads"),
        );
    }
}

#[test]
fn progress_callback_and_explicit_pool_do_not_change_results() {
    let _path = one_path();
    let inp = standard_input();
    let reference = bits(&run(&inp));
    for threads in [1, 2, 8] {
        let p = pool(threads);
        let mut events = 0usize;
        let mut cb = |_: psmc_rs::hmm::EStepProgress| events += 1;
        let opts = EStepOptions {
            on_progress: Some(&mut cb),
            thread_pool: Some(&p),
            ..EStepOptions::default()
        };
        let e = run_with(&inp, opts);
        assert!(events >= 2);
        assert_same(&e, &reference, &format!("{threads} threads with callback"));
        // The same with the callback pumped from inside a pool worker.
        let e = p.install(|| {
            let mut cb = |_: psmc_rs::hmm::EStepProgress| {};
            let opts = EStepOptions {
                on_progress: Some(&mut cb),
                ..EStepOptions::default()
            };
            run_with(&inp, opts)
        });
        assert_same(
            &e,
            &reference,
            &format!("{threads} threads, callback in worker"),
        );
    }
}

#[test]
fn log_likelihood_is_deterministic_and_matches_e_step() {
    let _path = one_path();
    let inp = standard_input();
    let e = run(&inp);
    for threads in [1, 2, 8] {
        let ll = pool(threads).install(|| log_likelihood_of(&inp));
        assert_eq!(ll.to_bits(), e.loglike.to_bits(), "{threads} threads");
    }
}

#[test]
fn psmc_models_recompute_instead_of_caching_rows() {
    let _path = one_path();
    let inp = standard_input();
    let budget = AlphaCacheBudget::new(usize::MAX);
    let opts = EStepOptions {
        alpha_cache: Some(&budget),
        ..EStepOptions::default()
    };
    run_with(&inp, opts);
    assert_eq!(budget.peak_bytes(), 0);
}

/// The posterior mean TMRCA of the sampled bins and the posterior mass of
/// every state, of a TMRCA decoding's report data. (Its MAP and quantiles
/// are discrete: they move to the next state where two states nearly tie.)
fn tmrca_values(d: &TmrcaReportData) -> Vec<f64> {
    let mut v: Vec<f64> = d.sampled_mean.iter().flatten().copied().collect();
    v.extend(&d.state_mass);
    v
}

/// Every value of a TMRCA decoding's report data, as bits.
fn tmrca_bits(d: &TmrcaReportData) -> Vec<u64> {
    let sampled = [
        &d.sampled_mean,
        &d.sampled_map,
        &d.sampled_lo,
        &d.sampled_hi,
    ];
    let mut v: Vec<u64> = sampled
        .iter()
        .flat_map(|p| p.iter().flatten().map(|x| x.to_bits()))
        .collect();
    v.extend(
        d.state_mass
            .iter()
            .chain(&d.state_years)
            .map(|x| x.to_bits()),
    );
    v
}

/// How far apart the portable and the AVX2 and FMA kernels may be: a
/// relative 1e-12, as the E-step oracle allows (`tests/estep_oracle.rs`).
/// They differ only in whether the O(states) forward and backward products
/// and transition sums fuse their multiply-adds, which moved no value by
/// more than a relative 1.4e-14 over a sequence of 15,000 sites
/// (`src/hmm/fb.rs` checks the same bound on every target), 3.0e-15 here.
const PATHS_REL: f64 = 1e-12;

/// Assert two vectors agree to [`PATHS_REL`], reporting the worst difference.
#[track_caller]
fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: shape differs");
    let (mut worst, mut at) = (0.0f64, 0);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let rel = if a == b { 0.0 } else { ((a - b) / b).abs() };
        if rel > worst || rel.is_nan() {
            (worst, at) = (rel, i);
        }
    }
    eprintln!("{what}: largest relative difference {worst:.2e}");
    assert!(
        worst <= PATHS_REL,
        "{what}: value {at} differs by a relative {worst:.2e}: {} vs {}",
        got[at],
        want[at]
    );
}

/// What [`simd_paths_agree`] compares of one run on one path.
struct PathRun {
    e_step: EStepResult,
    loglike: f64,
    tmrca: TmrcaReportData,
    tsv: Vec<u8>,
}

/// On x86-64 CPUs with AVX2 and FMA, the portable kernels and the AVX2 and
/// FMA ones give E-steps, log-likelihoods and TMRCA posteriors within
/// [`PATHS_REL`] with the O(n) semiseparable kernels, whose multiply-adds
/// only the AVX2 and FMA build fuses, and to the bit with the dense ones,
/// which never fuse: there the whole TMRCA decoding, its MAP, quantiles
/// and posterior TSV included, must be the same. Elsewhere there is one
/// path only, and the test is skipped. It switches the kernels of the
/// whole process, so it holds [`ONE_PATH`] for writing, and puts back the
/// choice it found (`PSMC_RS_SIMD`'s) when it ends, so that tests run
/// after it (`--test-threads=1`) keep that.
#[test]
fn simd_paths_agree() {
    /// Puts back whether the portable kernels were forced, even on a panic.
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_force_portable(self.0);
        }
    }
    let _path = ONE_PATH.write().unwrap_or_else(|e| e.into_inner());
    let _restore = Restore(set_force_portable(false));
    if simd_path() != SimdPath::Avx2Fma {
        eprintln!(
            "skipped: this CPU runs the {} kernels only; the test needs x86-64 with AVX2 and FMA",
            simd_path()
        );
        return;
    }
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let run_all = |inp: &Input, path: SimdPath| -> PathRun {
        set_force_portable(path == SimdPath::Portable);
        assert_eq!(simd_path(), path);
        let e_step = run(inp);
        let loglike = log_likelihood_of(inp);
        let years: Vec<f64> = (0..inp.pi.len()).map(|k| 1000.0 * k as f64).collect();
        let tsv = dir.join(format!("simd_{}_{}.tsv", std::process::id(), path));
        let opts = TmrcaOptions {
            max_points: 500,
            ..TmrcaOptions::default()
        };
        let tmrca = write_tmrca_posterior_tsv_with_options(
            &inp.pi,
            &inp.a,
            &inp.em,
            &inp.rows,
            &inp.row_starts,
            &years,
            &tsv,
            opts,
        )
        .unwrap();
        let text = std::fs::read(&tsv).unwrap();
        let _ = std::fs::remove_file(&tsv);
        PathRun {
            e_step,
            loglike,
            tmrca,
            tsv: text,
        }
    };
    for (name, inp, dense) in [
        ("semiseparable", standard_input(), false),
        ("dense", dense_input(), true),
    ] {
        let portable = run_all(&inp, SimdPath::Portable);
        let avx2 = run_all(&inp, SimdPath::Avx2Fma);
        if dense {
            assert_same(
                &avx2.e_step,
                &bits(&portable.e_step),
                &format!("{name}: E-step"),
            );
            assert_eq!(
                avx2.loglike.to_bits(),
                portable.loglike.to_bits(),
                "{name}: log-likelihood"
            );
            assert_eq!(
                tmrca_bits(&avx2.tmrca),
                tmrca_bits(&portable.tmrca),
                "{name}: TMRCA report data"
            );
            assert!(avx2.tsv == portable.tsv, "{name}: TMRCA posterior TSV");
        } else {
            let parts = [
                ("E-step", values(&avx2.e_step), values(&portable.e_step)),
                ("log-likelihood", vec![avx2.loglike], vec![portable.loglike]),
                (
                    "TMRCA posterior",
                    tmrca_values(&avx2.tmrca),
                    tmrca_values(&portable.tmrca),
                ),
            ];
            for (what, a, p) in parts {
                assert_close(&a, &p, &format!("{name}: {what}"));
            }
        }
    }
}
