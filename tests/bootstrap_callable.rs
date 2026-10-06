//! The callable bases of each bin through the bootstrap: joined back into
//! sequences, cut into trunks at the same bins as the symbols, drawn with
//! their trunks, and carried into the replicates' rows, without changing
//! which trunks are drawn or, as long as the model does not read them, any
//! fit.

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, observations_to_sequences, replicate_rng, resample_chunks,
    resample_sequences, rows_to_sequences, run_bootstrap_sequences, run_bootstrap_with_options,
    split_for_bootstrap, split_sequences_for_bootstrap,
};
use psmc_rs::io::common::chunk_sequences;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::{CallableBases, Sequences};
use psmc_rs::opt::MStepConfig;
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
                    let k = (7 * i + 13 * j) as u32;
                    if b == 2 { k % 10 } else { 10 + k % 91 }
                })
                .collect()
        })
        .collect();
    let seqs = Sequences {
        bins,
        callable: Some(CallableBases::new(100, counts).unwrap()),
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
    obs.callable = Some(CallableBases::new(100, vec![vec![100]]).unwrap());
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
        let lengths = |rows: &[Vec<u32>]| rows.iter().map(Vec::len).collect::<Vec<_>>();
        let bin_lengths: Vec<usize> = trunks.bins.iter().map(Vec::len).collect();
        assert_eq!(lengths(counts.counts()), bin_lengths, "trunk {trunk}");
        // In order, the trunks are the sequences cut up.
        let all = seqs.callable.as_ref().unwrap().counts().concat();
        assert_eq!(counts.counts().concat(), all, "trunk {trunk}");
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
    let counts: Vec<Vec<u32>> = bins
        .iter()
        .enumerate()
        .map(|(i, b)| vec![10 + i as u32; b.len()])
        .collect();
    let trunks = Sequences {
        bins: bins.clone(),
        callable: Some(CallableBases::new(100, counts).unwrap()),
    };
    for rep in 0..50 {
        let drawn = resample_sequences(&trunks, &mut replicate_rng(9, rep));
        let plain = resample_chunks(&bins, &mut replicate_rng(9, rep));
        assert_eq!(drawn.bins, plain, "replicate {rep}");
        let c = drawn.callable.as_ref().unwrap();
        assert_eq!(c.counts().len(), drawn.bins.len());
        for (b, row) in drawn.bins.iter().zip(c.counts()) {
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

#[test]
fn replicates_with_counts_fit_as_without_them() {
    let seqs = counted_seqs(3000);
    let cfg = BootstrapConfig {
        replicates: 3,
        trunk_size: 1000,
        seed: 77,
        em_iters: 1,
        tr_ratio: 5.0,
        t_max: 15.0,
        batch_size: Some(700),
        out_dir: None,
    };
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
    for (a, b) in with.iter().zip(&without) {
        assert_eq!((a.index, a.data), (b.index, b.data));
        assert_eq!(a.model.theta.to_bits(), b.model.theta.to_bits());
        assert_eq!(a.model.rho.to_bits(), b.model.rho.to_bits());
        assert_eq!(a.model.lam, b.model.lam);
    }

    // Invalid counts stop the bootstrap before anything is fitted.
    let mut bad = seqs.clone();
    bad.callable = Some(CallableBases::new(100, vec![vec![100]]).unwrap());
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
