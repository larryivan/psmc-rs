//! QC metrics against hand-computed values.

use psmc_rs::bootstrap::split_for_bootstrap;
use psmc_rs::initial_theta;
use psmc_rs::io::common::chunk_sequences;
use psmc_rs::qc::{
    MAX_SEQUENCE_ROWS, QcContext, QcInput, QcSeq, analyze, bootstrap_trunks_of,
    initial_recombinations, n50, theta0_per_bin, try_analyze,
};

fn input<'a>(
    rows: &'a [Vec<u8>],
    starts: &'a [bool],
    seqs: &'a [QcSeq],
    trunk: usize,
) -> QcInput<'a> {
    QcInput {
        rows,
        row_starts: starts,
        seqs,
        bin_size: 100,
        bootstrap_trunk_bins: trunk,
        context: QcContext::default(),
        callable: None,
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} != {b}");
}

/// chr1 spans two rows: T K T N N | N T K K; chr2 is one row:
/// T T T T K N T N N.
fn two_sequences() -> (Vec<Vec<u8>>, Vec<bool>, Vec<QcSeq>) {
    let rows = vec![
        vec![0, 1, 0, 2, 2],
        vec![2, 0, 1, 1],
        vec![0, 0, 0, 0, 1, 2, 0, 2, 2],
    ];
    let starts = vec![true, false, true];
    let mut chr1 = QcSeq::new("chr1", 9);
    chr1.extent_bases = Some(850);
    chr1.called_bases = Some(520);
    chr1.het_sites = Some(4);
    let chr2 = QcSeq::new("chr2", 9);
    (rows, starts, vec![chr1, chr2])
}

#[test]
fn per_sequence_metrics_match_hand_counts() {
    let (rows, starts, seqs) = two_sequences();
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    assert_eq!(r.n_sequences, 2);
    assert_eq!(r.sequences.len(), 2);
    assert!(r.omitted.is_none());

    let chr1 = &r.sequences[0];
    assert_eq!((chr1.index, chr1.name.as_str()), (0, "chr1"));
    let s = &chr1.stats;
    assert_eq!((s.bins, s.t_bins, s.k_bins, s.missing_bins), (9, 3, 3, 3));
    close(s.t_fraction, 3.0 / 9.0);
    close(s.k_fraction, 3.0 / 9.0);
    close(s.missing_fraction, 3.0 / 9.0);
    close(s.het_bin_fraction.unwrap(), 0.5);
    close(s.theta0_per_bin.unwrap(), 2f64.ln());
    close(s.theta_per_bp.unwrap(), 2f64.ln() / 100.0);
    close(s.het_per_kb.unwrap(), 10.0 * 2f64.ln());
    assert_eq!(s.called_bases_from_bins, 600);
    assert_eq!(s.called_bases, Some(520));
    assert_eq!(s.het_sites, Some(4));
    close(s.site_het_per_kb.unwrap(), 4000.0 / 520.0);
    close(s.uncalled_site_fraction.unwrap(), 1.0 - 520.0 / 850.0);
    assert_eq!(s.length_bases, 850);
    // N N | N crosses the row boundary: one run of 3.
    assert_eq!(s.longest_missing_run_bins, 3);

    let s = &r.sequences[1].stats;
    assert_eq!((s.bins, s.t_bins, s.k_bins, s.missing_bins), (9, 5, 1, 3));
    close(s.theta0_per_bin.unwrap(), -(5f64 / 6.0).ln());
    assert_eq!(s.called_bases_from_bins, 600);
    assert_eq!(
        (
            s.called_bases,
            s.het_sites,
            s.site_het_per_kb,
            s.uncalled_site_fraction
        ),
        (None, None, None, None)
    );
    assert_eq!(s.length_bases, 900);
    assert_eq!(s.longest_missing_run_bins, 2);
}

#[test]
fn whole_input_metrics_match_hand_counts() {
    let (rows, starts, seqs) = two_sequences();
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    let s = &r.totals;
    assert_eq!((s.bins, s.t_bins, s.k_bins, s.missing_bins), (18, 8, 4, 6));
    close(s.missing_fraction, 6.0 / 18.0);
    close(s.theta0_per_bin.unwrap(), 1.5f64.ln());
    close(s.het_per_kb.unwrap(), 10.0 * 1.5f64.ln());
    assert_eq!(s.called_bases_from_bins, 1200);
    // chr2 has no site counts, so neither do the totals.
    assert_eq!((s.called_bases, s.het_sites), (None, None));
    assert_eq!(s.length_bases, 1750);
    assert_eq!(s.longest_missing_run_bins, 3);

    let run = r.longest_missing_run.as_ref().unwrap();
    assert_eq!(
        (run.sequence.as_str(), run.start_bin, run.bins, run.bases),
        ("chr1", 3, 3, 300)
    );

    let l = &r.lengths;
    assert_eq!((l.total_bases, l.longest_bases), (1750, 900));
    // 900 alone reaches half of 1750.
    assert_eq!((l.n50_bases, l.l50), (900, 1));
    assert_eq!(
        (
            l.below_1mb.sequences,
            l.below_1mb.bins,
            l.below_1mb.fraction
        ),
        (2, 18, 1.0)
    );
    assert_eq!((l.below_trunk.clone(), l.bootstrap_trunks), (None, None));

    let want = 12.0 / (4.0 / 1.5f64.ln() + 0.5);
    close(r.expected_recombinations.unwrap(), want);
}

#[test]
fn site_counts_add_up_when_every_sequence_has_them() {
    let (rows, starts, mut seqs) = two_sequences();
    seqs[1].called_bases = Some(580);
    seqs[1].het_sites = Some(1);
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    assert_eq!(r.totals.called_bases, Some(1100));
    assert_eq!(r.totals.het_sites, Some(5));
    close(r.totals.site_het_per_kb.unwrap(), 5000.0 / 1100.0);
    // chr2 has no extent: 9 bins of 100 bp.
    close(
        r.totals.uncalled_site_fraction.unwrap(),
        1.0 - 1100.0 / 1750.0,
    );
    close(
        r.sequences[1].stats.uncalled_site_fraction.unwrap(),
        1.0 - 580.0 / 900.0,
    );

    // More called sites than the extent: clamped to none uncalled.
    seqs[0].called_bases = Some(900);
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    assert_eq!(r.sequences[0].stats.uncalled_site_fraction, Some(0.0));
}

#[test]
fn trunk_metrics() {
    let (rows, starts, seqs) = two_sequences();
    // Trunk 5: both 9-bin sequences reach 1.5 trunks (7 bins), so each is
    // cut into 5 + 4.
    let r = analyze(&input(&rows, &starts, &seqs, 5));
    let below = r.lengths.below_trunk.as_ref().unwrap();
    assert_eq!((below.sequences, below.bins, below.fraction), (0, 0, 0.0));
    assert_eq!(r.lengths.bootstrap_trunks, Some(4));
    // Trunk 10: both are shorter than one trunk and stay whole.
    let r = analyze(&input(&rows, &starts, &seqs, 10));
    let below = r.lengths.below_trunk.as_ref().unwrap();
    assert_eq!((below.sequences, below.bins, below.fraction), (2, 18, 1.0));
    assert_eq!(r.lengths.bootstrap_trunks, Some(2));
}

#[test]
fn trunk_count_matches_split_for_bootstrap() {
    for trunk in 1..=12usize {
        for len in 0..=80usize {
            let want = split_for_bootstrap(&[vec![0u8; len]], trunk).len() as u64;
            assert_eq!(
                bootstrap_trunks_of(len as u64, trunk as u64),
                want,
                "len {len}, trunk {trunk}"
            );
        }
    }
    let big = 3_000_000_000u64;
    assert_eq!(bootstrap_trunks_of(big, 500_000), 6000);
    assert_eq!(bootstrap_trunks_of(big + 249_999, 500_000), 6000);
    assert_eq!(bootstrap_trunks_of(big + 250_000, 500_000), 6001);
    assert_eq!(bootstrap_trunks_of(10, 0), 0);
}

#[test]
fn n50_examples() {
    // Total 54, half 27: 10 + 9 + 8 = 27.
    assert_eq!(n50(&[2, 3, 4, 5, 6, 7, 8, 9, 10]), (8, 3));
    // Order does not matter; exactly half (5 of 10) counts as reaching it.
    assert_eq!(n50(&[1, 5, 4]), (5, 1));
    assert_eq!(n50(&[4, 1, 5, 1]), (4, 2));
    assert_eq!(n50(&[3, 3]), (3, 1));
    assert_eq!(n50(&[7]), (7, 1));
    assert_eq!(n50(&[]), (0, 0));
    assert_eq!(n50(&[0, 0]), (0, 0));
    // No overflow on huge totals.
    assert_eq!(n50(&[u64::MAX, u64::MAX, 1]), (u64::MAX, 2));
}

#[test]
fn theta_formulas() {
    close(theta0_per_bin(3, 1).unwrap(), -(0.75f64).ln());
    // No heterozygous bin: exactly +0 (C prints -0).
    let zero = theta0_per_bin(10, 0).unwrap();
    assert_eq!(zero.to_bits(), 0f64.to_bits());
    assert_eq!(theta0_per_bin(0, 0), None);
    assert_eq!(theta0_per_bin(0, 5), None);
    close(initial_recombinations(1000, 0.1, 4.0), 1000.0 / 40.5);
    assert_eq!(initial_recombinations(1000, 0.0, 4.0), 0.0);
}

#[test]
fn theta_equals_the_models_starting_value() {
    let (rows, starts, seqs) = two_sequences();
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    let want = initial_theta(&rows).unwrap();
    assert_eq!(r.totals.theta0_per_bin.unwrap().to_bits(), want.to_bits());
}

#[test]
fn row_splitting_does_not_change_the_metrics() {
    let seqs_bins: Vec<Vec<u8>> = (0..5)
        .map(|s| {
            (0..(1000 + 337 * s))
                .map(|i| match (i * 7 + s * 3) % 23 {
                    _ if (300..420).contains(&i) => 2,
                    0 | 5 => 1,
                    3..=4 => 2,
                    _ => 0,
                })
                .collect()
        })
        .collect();
    let whole = chunk_sequences(seqs_bins.clone(), None, "empty").unwrap();
    let split = chunk_sequences(seqs_bins, Some(97), "empty").unwrap();
    assert!(split.rows.len() > whole.rows.len());
    let a = analyze(&input(&whole.rows, &whole.row_starts, &[], 1000));
    let b = analyze(&input(&split.rows, &split.row_starts, &[], 1000));
    assert_eq!(a, b);
    assert_eq!(a.n_sequences, 5);
    assert_eq!(a.sequences[3].name, "seq4");
    assert_eq!(a.sequences[0].stats.longest_missing_run_bins, 120);
}

#[test]
fn empty_sequences_are_counted_and_skipped() {
    let rows = vec![vec![0, 1], vec![], vec![2, 0]];
    let starts = vec![true, true, true];
    let seqs = vec![
        QcSeq::new("a", 2),
        QcSeq::new("empty1", 0),
        QcSeq::new("b", 2),
        QcSeq::new("empty2", 0),
    ];
    let r = analyze(&input(&rows, &starts, &seqs, 0));
    assert_eq!((r.n_sequences, r.empty_sequences), (2, 2));
    let names: Vec<&str> = r.sequences.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(r.sequences[1].index, 1);
}

#[test]
fn empty_input_gives_zero_metrics() {
    let r = analyze(&input(&[], &[], &[], 500));
    assert_eq!(r.n_sequences, 0);
    assert_eq!(r.totals.bins, 0);
    assert_eq!(r.totals.missing_fraction, 0.0);
    assert_eq!(r.totals.theta0_per_bin, None);
    assert_eq!(r.totals.called_bases, None);
    assert_eq!(r.longest_missing_run, None);
    assert_eq!(r.expected_recombinations, None);
    assert_eq!(r.lengths.bootstrap_trunks, Some(0));
    assert!(r.sequences.is_empty());
}

#[test]
fn symbols_other_than_t_and_k_are_missing() {
    let rows = vec![vec![0, 1, 2, 3, 255, 1]];
    let r = analyze(&input(&rows, &[true], &[], 0));
    let s = &r.totals;
    assert_eq!((s.t_bins, s.k_bins, s.missing_bins), (1, 2, 3));
    assert_eq!(s.longest_missing_run_bins, 3);
}

#[test]
fn longest_missing_run_reports_the_first_of_equal_runs() {
    let rows = vec![vec![0, 2, 2, 0], vec![2, 2, 0, 0], vec![0, 0, 0, 0]];
    let r = analyze(&input(&rows, &[true, true, true], &[], 0));
    let run = r.longest_missing_run.unwrap();
    assert_eq!((run.sequence.as_str(), run.start_bin), ("seq1", 1));
    assert_eq!(r.sequences[2].stats.longest_missing_run_bins, 0);
}

#[test]
fn many_sequences_are_capped_but_totals_cover_all() {
    let n = MAX_SEQUENCE_ROWS + 150;
    // Lengths cycle so that several sequences share each length.
    let rows: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let len = 10 + (i * 37) % 200;
            (0..len)
                .map(|j| (j % 5 == 0) as u8 + 2 * (j % 11 == 1) as u8)
                .collect()
        })
        .collect();
    let starts = vec![true; n];
    let r = analyze(&input(&rows, &starts, &[], 0));
    assert_eq!(r.n_sequences, n);
    assert_eq!(r.sequences.len(), MAX_SEQUENCE_ROWS);
    let omitted = r.omitted.as_ref().unwrap();
    assert_eq!(omitted.sequences, 150);

    // Rows are in input order and are the longest sequences.
    let idx: Vec<usize> = r.sequences.iter().map(|s| s.index).collect();
    assert!(idx.windows(2).all(|w| w[0] < w[1]));
    let shortest_shown = r.sequences.iter().map(|s| s.stats.bins).min().unwrap();
    let mut lens: Vec<(u64, usize)> = rows.iter().map(|r| r.len() as u64).zip(0..).collect();
    lens.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let want: Vec<usize> = {
        let mut v: Vec<usize> = lens[..MAX_SEQUENCE_ROWS].iter().map(|x| x.1).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(idx, want);
    assert!(
        lens[MAX_SEQUENCE_ROWS..]
            .iter()
            .all(|x| x.0 <= shortest_shown)
    );

    // Shown + omitted = totals.
    let sum = |f: fn(&psmc_rs::qc::BinStats) -> u64| {
        r.sequences.iter().map(|s| f(&s.stats)).sum::<u64>() + f(&omitted.stats)
    };
    assert_eq!(sum(|s| s.bins), r.totals.bins);
    assert_eq!(sum(|s| s.t_bins), r.totals.t_bins);
    assert_eq!(sum(|s| s.k_bins), r.totals.k_bins);
    assert_eq!(sum(|s| s.missing_bins), r.totals.missing_bins);
}

#[test]
fn inconsistent_inputs_are_errors() {
    let (rows, starts, seqs) = two_sequences();
    let ok = input(&rows, &starts, &seqs, 0);
    assert!(try_analyze(&ok).is_ok());

    let mut bad = ok.clone();
    bad.bin_size = 0;
    assert!(format!("{:#}", try_analyze(&bad).unwrap_err()).contains("bin_size"));

    let short_starts = [true, false];
    let bad = input(&rows, &short_starts, &seqs, 0);
    assert!(format!("{:#}", try_analyze(&bad).unwrap_err()).contains("row_starts"));

    let cont_first = [false, false, true];
    let bad = input(&rows, &cont_first, &seqs, 0);
    assert!(format!("{:#}", try_analyze(&bad).unwrap_err()).contains("first row"));

    let one_seq = [seqs[0].clone()];
    let bad = input(&rows, &starts, &one_seq, 0);
    assert!(format!("{:#}", try_analyze(&bad).unwrap_err()).contains("1 non-empty"));

    let mut wrong_len = seqs.clone();
    wrong_len[1].n_bins = 8;
    let bad = input(&rows, &starts, &wrong_len, 0);
    let msg = format!("{:#}", try_analyze(&bad).unwrap_err());
    assert!(msg.contains("'chr2'") && msg.contains("8 bins"), "{msg}");
}

#[test]
#[should_panic(expected = "bin_size")]
fn analyze_panics_on_inconsistent_input() {
    let rows = vec![vec![0u8]];
    let mut i = input(&rows, &[true], &[], 0);
    i.bin_size = 0;
    analyze(&i);
}

#[test]
fn json_is_deterministic_and_complete() {
    let (rows, starts, seqs) = two_sequences();
    let a = analyze(&input(&rows, &starts, &seqs, 5)).to_json();
    let b = analyze(&input(&rows, &starts, &seqs, 5)).to_json();
    assert_eq!(a, b);
    let v: serde_json::Value = serde_json::from_str(&a).unwrap();
    assert_eq!(v["totals"]["k_bins"], 4);
    assert_eq!(v["lengths"]["bootstrap_trunks"], 4);
    assert_eq!(v["longest_missing_run"]["sequence"], "chr1");
    // Per-sequence rows are flat: name next to the metrics.
    assert_eq!(v["sequences"][0]["name"], "chr1");
    assert_eq!(v["sequences"][0]["called_bases"], 520);
    assert!(v["sequences"][1]["called_bases"].is_null());
    assert!(v["omitted"].is_null());
}
