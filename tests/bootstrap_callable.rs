//! The callable bases of each bin through the bootstrap: joined back into
//! sequences, cut into trunks at the same bins as the symbols, drawn with
//! their trunks without changing which trunks are drawn, and fitted with the
//! callable-length model in each replicate, from a fresh start whose θ0
//! comes from the replicate's own counts.

use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, invalid_replicate_starts_sequences,
    observations_to_sequences, replicate_initial_model_with_callable, replicate_rng,
    resample_chunks, resample_sequences, rows_to_sequences, run_bootstrap_sequences,
    run_bootstrap_with_options, split_for_bootstrap, split_sequences_for_bootstrap,
};
use psmc_rs::em::{EmOptions, em_train_with_options};
use psmc_rs::io::common::chunk_sequences;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::{CallableBases, Sequences};
use psmc_rs::model::NegativeTransitions;
use psmc_rs::opt::MStepConfig;
use psmc_rs::{PsmcModel, initial_theta, initial_theta_with_callable};
use std::path::Path;

/// The three reference sequences cut down to `len` bins each, with made-up
/// callable bases that agree with the symbols (bins of 100 bases).
fn counted_seqs(len: usize) -> Sequences {
    let obs = read_psmcfa(
        Path::new("tests/data/c_reference/papuan_chr1_3seg.psmcfa"),
        None,
    )
    .unwrap();
    let bins: Vec<Vec<u8>> = rows_to_sequences(&obs.rows, &obs.row_starts)
        .unwrap()
        .into_iter()
        .map(|s| s[..len].to_vec())
        .collect();
    let counts = bins
        .iter()
        .enumerate()
        .map(|(i, seq)| {
            seq.iter()
                .enumerate()
                .map(|(j, &b)| {
                    // Full bins in stretches, and partly called ones.
                    let k = (7 * i + 13 * j) as u16;
                    match b {
                        2 => k % 10,
                        _ if (j / 50) % 3 == 0 => 100,
                        _ => 10 + k % 91,
                    }
                })
                .collect()
        })
        .collect();
    let seqs = Sequences {
        bins,
        callable: Some(CallableBases::from_counts(100, counts).unwrap()),
    };
    seqs.validate().unwrap();
    seqs
}

#[test]
fn observations_join_back_into_their_sequences_with_their_counts() {
    let seqs = counted_seqs(2500);
    for batch in [None, Some(1), Some(7), Some(1000)] {
        let obs = chunk_sequences(seqs.clone(), batch, "empty").unwrap();
        let back = observations_to_sequences(&obs).unwrap();
        assert_eq!(back, seqs, "batch {batch:?}");
        assert_eq!(
            back.bins,
            rows_to_sequences(&obs.rows, &obs.row_starts).unwrap()
        );
        let plain = chunk_sequences(seqs.bins.clone(), batch, "empty").unwrap();
        let back = observations_to_sequences(&plain).unwrap();
        assert_eq!(back, Sequences::from(seqs.bins.clone()));
    }
    // Rows whose counts do not fit are refused.
    let mut obs = chunk_sequences(seqs.clone(), Some(100), "empty").unwrap();
    obs.callable = Some(CallableBases::from_counts(100, vec![vec![100]]).unwrap());
    assert!(observations_to_sequences(&obs).is_err());
}

#[test]
fn trunks_keep_the_counts_of_their_bins() {
    let seqs = counted_seqs(2500);
    for trunk in [1, 2, 99, 400, 1000, 1666, 1667, 2500, 10_000] {
        let trunks = split_sequences_for_bootstrap(&seqs, trunk);
        assert_eq!(
            trunks.bins,
            split_for_bootstrap(&seqs.bins, trunk),
            "trunk {trunk}"
        );
        trunks.validate().unwrap();
        let counts = trunks.callable.as_ref().unwrap();
        assert_eq!(counts.bin_size(), 100);
        let lengths: Vec<usize> = counts.rows().iter().map(|r| r.len()).collect();
        let bin_lengths: Vec<usize> = trunks.bins.iter().map(Vec::len).collect();
        assert_eq!(lengths, bin_lengths, "trunk {trunk}");
        // In order, the trunks are the sequences cut up.
        let all = seqs.callable.as_ref().unwrap().to_counts().concat();
        assert_eq!(counts.to_counts().concat(), all, "trunk {trunk}");
    }
    let plain = split_sequences_for_bootstrap(&Sequences::from(seqs.bins.clone()), 400);
    assert_eq!(plain.bins, split_for_bootstrap(&seqs.bins, 400));
    assert!(plain.callable.is_none());
}

#[test]
fn resampling_draws_the_same_trunks_with_their_counts() {
    // Trunk i holds only T bins with 10 + i callable bases, so each drawn
    // trunk says which one it is.
    let bins: Vec<Vec<u8>> = (0..40).map(|i| vec![0; 1 + (i * 37) % 23]).collect();
    let counts: Vec<Vec<u16>> = bins
        .iter()
        .enumerate()
        .map(|(i, b)| vec![10 + i as u16; b.len()])
        .collect();
    let trunks = Sequences {
        bins: bins.clone(),
        callable: Some(CallableBases::from_counts(100, counts).unwrap()),
    };
    for rep in 0..50 {
        let drawn = resample_sequences(&trunks, &mut replicate_rng(9, rep));
        let plain = resample_chunks(&bins, &mut replicate_rng(9, rep));
        assert_eq!(drawn.bins, plain, "replicate {rep}");
        let c = drawn.callable.as_ref().unwrap().to_counts();
        assert_eq!(c.len(), drawn.bins.len());
        for (b, row) in drawn.bins.iter().zip(&c) {
            let i = (row[0] - 10) as usize;
            assert_eq!(*b, bins[i], "replicate {rep}");
            assert!(row.iter().all(|&x| x == row[0]) && row.len() == b.len());
        }
        let without =
            resample_sequences(&Sequences::from(bins.clone()), &mut replicate_rng(9, rep));
        assert_eq!(without.bins, plain);
        assert!(without.callable.is_none());
    }
}

fn template() -> PsmcModel {
    PsmcModel::new(15.0, 22, 0.02, 0.004, 2.5e-8, Some("4+5*3+4".to_string())).unwrap()
}

fn mstep() -> MStepConfig {
    MStepConfig {
        progress: false,
        ..MStepConfig::default()
    }
}

fn config() -> BootstrapConfig {
    BootstrapConfig {
        replicates: 3,
        trunk_size: 1000,
        seed: 77,
        em_iters: 2,
        tr_ratio: 5.0,
        t_max: 15.0,
        batch_size: Some(700),
        out_dir: None,
    }
}

/// The parameters of a model as bits.
fn bits(m: &PsmcModel) -> Vec<u64> {
    [m.theta, m.rho, m.t_max]
        .iter()
        .chain(&m.lam)
        .map(|v| v.to_bits())
        .collect()
}

/// Each replicate is the fit of its resampled bins with their counts: from
/// θ0 of its own counts, with the callable-length model; with every called
/// bin full, the fit of the bins alone, to the bit.
#[test]
fn replicates_fit_their_resampled_counts() {
    let seqs = counted_seqs(3000);
    let cfg = config();
    let with = run_bootstrap_sequences(
        &seqs,
        &template(),
        &cfg,
        &mstep(),
        BootstrapOptions::default(),
    )
    .unwrap();
    let without = run_bootstrap_with_options(
        &seqs.bins,
        &template(),
        &cfg,
        &mstep(),
        BootstrapOptions::default(),
    )
    .unwrap();
    assert_eq!(with.len(), 3);
    let trunks = split_sequences_for_bootstrap(&seqs, cfg.trunk_size);
    for (a, b) in with.iter().zip(&without) {
        assert_eq!((a.index, a.data), (b.index, b.data));
        assert_ne!(bits(&a.model), bits(&b.model), "the counts change the fit");
        // The same fit, by hand.
        let sampled = resample_sequences(&trunks, &mut replicate_rng(cfg.seed, a.index));
        let obs = chunk_sequences(sampled, cfg.batch_size, "empty").unwrap();
        let mut model = replicate_initial_model_with_callable(
            &template(),
            &obs.rows,
            obs.callable.as_ref(),
            cfg.t_max,
            cfg.tr_ratio,
        )
        .unwrap();
        let theta0 = initial_theta_with_callable(&obs.rows, obs.callable.as_ref()).unwrap();
        assert_eq!(model.theta.to_bits(), theta0.to_bits());
        assert!(theta0 > initial_theta(&obs.rows).unwrap());
        let opts = EmOptions {
            callable: obs.callable.as_ref(),
            ..EmOptions::default()
        };
        em_train_with_options(
            &mut model,
            &obs.rows,
            &obs.row_starts,
            cfg.em_iters,
            &mstep(),
            opts,
        )
        .unwrap();
        assert_eq!(bits(&a.model), bits(&model), "replicate {}", a.index);
    }

    // With every called bin full, the counts change nothing.
    let mut full = seqs.clone();
    let counts = seqs
        .bins
        .iter()
        .zip(seqs.callable.as_ref().unwrap().to_counts())
        .map(|(b, c)| {
            b.iter()
                .zip(c)
                .map(|(&v, c)| if v == 2 { c } else { 100 })
                .collect()
        })
        .collect();
    full.callable = Some(CallableBases::from_counts(100, counts).unwrap());
    let full = run_bootstrap_sequences(
        &full,
        &template(),
        &cfg,
        &mstep(),
        BootstrapOptions::default(),
    )
    .unwrap();
    for (a, b) in full.iter().zip(&without) {
        assert_eq!(bits(&a.model), bits(&b.model), "replicate {}", a.index);
    }

    // Invalid counts stop the bootstrap before anything is fitted.
    let mut bad = seqs.clone();
    bad.callable = Some(CallableBases::from_counts(100, vec![vec![100]]).unwrap());
    let err = run_bootstrap_sequences(
        &bad,
        &template(),
        &cfg,
        &mstep(),
        BootstrapOptions::default(),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("callable bases for 1 sequences"));
}

/// Replicates with counts do not depend on the thread count or the
/// batching.
#[test]
fn counted_replicates_do_not_depend_on_threads_or_batching() {
    let seqs = counted_seqs(3000);
    let run = |threads: usize, batch: Option<usize>| -> Vec<Vec<u64>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let cfg = BootstrapConfig {
            batch_size: batch,
            ..config()
        };
        let opts = BootstrapOptions {
            thread_pool: Some(&pool),
            ..BootstrapOptions::default()
        };
        run_bootstrap_sequences(&seqs, &template(), &cfg, &mstep(), opts)
            .unwrap()
            .iter()
            .map(|r| bits(&r.model))
            .collect()
    };
    let reference = run(1, None);
    for (threads, batch) in [(2, Some(700)), (8, Some(129)), (3, None)] {
        assert_eq!(
            run(threads, batch),
            reference,
            "{threads} threads, {batch:?}"
        );
    }
}

/// With counts, a replicate's fresh start takes θ0 from its own counts, and
/// `invalid_replicate_starts_sequences` finds exactly the replicates whose
/// start that makes invalid, which the bootstrap then skips.
#[test]
fn invalid_counted_starts_are_found_and_skipped() {
    let seqs = counted_seqs(3000);
    let template = template();
    let theta0 = initial_theta_with_callable(&seqs.bins, seqs.callable.as_ref()).unwrap();
    let limit = PsmcModel::new(15.0, 22, theta0, 0.5, 2.5e-8, Some("4+5*3+4".into()))
        .unwrap()
        .check_transitions()
        .unwrap_err()
        .downcast::<NegativeTransitions>()
        .unwrap()
        .max_rho
        .unwrap();
    let cfg = BootstrapConfig {
        replicates: 12,
        tr_ratio: theta0 / (0.995 * limit),
        em_iters: 1,
        ..config()
    };
    let invalid = invalid_replicate_starts_sequences(&seqs, &template, &cfg).unwrap();
    let trunks = split_sequences_for_bootstrap(&seqs, cfg.trunk_size);
    let mut expected = Vec::new();
    for rep in 0..cfg.replicates {
        let sampled = resample_sequences(&trunks, &mut replicate_rng(cfg.seed, rep));
        let start = replicate_initial_model_with_callable(
            &template,
            &sampled.bins,
            sampled.callable.as_ref(),
            cfg.t_max,
            cfg.tr_ratio,
        )
        .unwrap();
        if start.check_transitions().is_err() {
            expected.push((rep, start.theta.to_bits(), start.rho.to_bits()));
        }
    }
    let found: Vec<(usize, u64, u64)> = invalid
        .iter()
        .map(|r| (r.index, r.start.theta.to_bits(), r.start.rho.to_bits()))
        .collect();
    assert_eq!(found, expected);
    assert!(
        !found.is_empty() && found.len() < cfg.replicates,
        "{} of {} replicates invalid",
        found.len(),
        cfg.replicates
    );
    let opts = BootstrapOptions {
        skip_invalid_starts: true,
        ..BootstrapOptions::default()
    };
    let fitted: Vec<usize> = run_bootstrap_sequences(&seqs, &template, &cfg, &mstep(), opts)
        .unwrap()
        .iter()
        .map(|r| r.index)
        .collect();
    let skipped: Vec<usize> = found.iter().map(|r| r.0).collect();
    assert_eq!(
        fitted,
        (0..cfg.replicates)
            .filter(|i| !skipped.contains(i))
            .collect::<Vec<_>>()
    );
}
