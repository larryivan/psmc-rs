//! QC must stay cheap enough to run before every fit: 30 million bins (a
//! 3 Gb genome at 100 bp per bin) in well under a second.

use psmc_rs::qc::{QcContext, QcInput, QcSeq, analyze};
use std::time::Instant;

#[test]
fn thirty_million_bins_in_well_under_a_second() {
    // 3,000 scaffolds of 5,000 bins plus 15 chromosomes of 1M bins, so that
    // both the per-bin scan and the per-sequence work are exercised.
    let mut lens = vec![5_000usize; 3_000];
    lens.extend([1_000_000; 15]);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let rows: Vec<Vec<u8>> = lens
        .iter()
        .map(|&len| {
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    match state >> 57 {
                        0..=7 => 1,
                        8..=11 => 2,
                        _ => 0,
                    }
                })
                .collect()
        })
        .collect();
    let starts = vec![true; rows.len()];
    let seqs: Vec<QcSeq> = lens
        .iter()
        .enumerate()
        .map(|(i, &n)| QcSeq::new(format!("scaffold_{i}"), n))
        .collect();
    let input = QcInput {
        rows: &rows,
        row_starts: &starts,
        seqs: &seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 500_000,
        context: QcContext::default(),
    };
    assert_eq!(rows.iter().map(Vec::len).sum::<usize>(), 30_000_000);

    let start = Instant::now();
    let report = analyze(&input);
    let elapsed = start.elapsed();
    eprintln!("analyze: 30M bins in {elapsed:?}");
    assert_eq!(report.totals.bins, 30_000_000);
    // Typically ~0.05 s; the bound leaves room for a loaded machine.
    assert!(elapsed.as_secs_f64() < 1.0, "analyze took {elapsed:?}");
}
