//! The E-step must give bit-identical results however it is run: any thread
//! count, top level or nested inside rayon work (`ThreadPool::install`, a
//! parallel iterator), any alpha-cache budget, with or without a progress
//! callback.

use ndarray::Array2;
use psmc_rs::PsmcModel;
use psmc_rs::hmm::{
    AlphaCacheBudget, CHECKPOINT_STRIDE, EStepOptions, EStepResult, e_step_streaming,
    e_step_with_options,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

struct Input {
    pi: Vec<f64>,
    a: Array2<f64>,
    em: Array2<f64>,
    rows: Vec<Vec<u8>>,
    row_starts: Vec<bool>,
}

/// `lens[i]` sites for sequence `i`, every other sequence split into chained rows.
fn input(seed: u64, pattern: &str, n_steps: usize, lens: &[usize]) -> Input {
    let mut rng = SmallRng::seed_from_u64(seed);
    let model =
        PsmcModel::new(15.0, n_steps, 2e-3, 4e-4, 2.5e-8, Some(pattern.to_string())).unwrap();
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

fn bits(e: &EStepResult) -> Vec<u64> {
    let mut v = vec![e.loglike.to_bits()];
    for g in &e.stats.gobs {
        v.extend(g.iter().map(|x| x.to_bits()));
    }
    v.extend(e.stats.xi.iter().map(|x| x.to_bits()));
    v
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

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn standard_input() -> Input {
    input(
        11,
        "4+25*2+4+6",
        63,
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

#[test]
fn thread_count_and_nesting_do_not_change_results() {
    let inp = standard_input();
    let reference = bits(&run(&inp));
    for threads in [1, 2, 8] {
        let p = pool(threads);
        let installed = p.install(|| run(&inp));
        assert_same(
            &installed,
            &reference,
            &format!("{threads} threads via install"),
        );
        // Nested inside a parallel iterator, as bootstrap replicates call it.
        let nested: Vec<EStepResult> =
            p.install(|| (0..3).into_par_iter().map(|_| run(&inp)).collect());
        for e in &nested {
            assert_same(
                e,
                &reference,
                &format!("{threads} threads nested in par_iter"),
            );
        }
    }
}

#[test]
fn many_sequences_are_reduced_deterministically() {
    // More sequences than work items, so sequences are grouped.
    let lens: Vec<usize> = (0..700).map(|i| 1 + (i * 37) % 300).collect();
    let inp = input(5, "4+5*3+4", 22, &lens);
    let reference = bits(&run(&inp));
    for threads in [1, 3, 8] {
        let e = pool(threads).install(|| run(&inp));
        assert_same(&e, &reference, &format!("{threads} threads"));
    }
}

#[test]
fn alpha_cache_budget_does_not_change_results() {
    let inp = standard_input();
    let one_row = 8 * 64 * 800; // enough for the short sequences only
    let mut seen = Vec::new();
    for limit in [0, 1, one_row, usize::MAX] {
        let budget = AlphaCacheBudget::new(limit);
        for threads in [1, 4] {
            let e = pool(threads).install(|| {
                e_step_with_options(
                    &inp.pi,
                    &inp.a,
                    &inp.em,
                    &inp.rows,
                    &inp.row_starts,
                    EStepOptions {
                        alpha_cache: Some(&budget),
                        ..EStepOptions::default()
                    },
                )
                .unwrap()
            });
            assert_eq!(budget.used_bytes(), 0, "reservations must be released");
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
    let inp = standard_input();
    let reference = bits(&run(&inp));
    for threads in [1, 2, 8] {
        let p = pool(threads);
        let mut events = 0usize;
        let mut cb = |_: psmc_rs::hmm::EStepProgress| events += 1;
        let e = e_step_with_options(
            &inp.pi,
            &inp.a,
            &inp.em,
            &inp.rows,
            &inp.row_starts,
            EStepOptions {
                on_progress: Some(&mut cb),
                thread_pool: Some(&p),
                ..EStepOptions::default()
            },
        )
        .unwrap();
        assert!(events >= 2);
        assert_same(&e, &reference, &format!("{threads} threads with callback"));
        // The same with the callback pumped from inside a pool worker.
        let e = p.install(|| {
            let mut cb = |_: psmc_rs::hmm::EStepProgress| {};
            e_step_with_options(
                &inp.pi,
                &inp.a,
                &inp.em,
                &inp.rows,
                &inp.row_starts,
                EStepOptions {
                    on_progress: Some(&mut cb),
                    ..EStepOptions::default()
                },
            )
            .unwrap()
        });
        assert_same(
            &e,
            &reference,
            &format!("{threads} threads, callback in worker"),
        );
    }
}
