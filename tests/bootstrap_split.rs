//! `split_for_bootstrap` must cut sequences exactly like psmc's `utils/splitfa`.
//!
//! Every expected trunk-length list below was produced by running the original
//! `splitfa` binary (lh3/psmc 0.6.5-r74) on psmcfa files holding sequences of
//! the listed lengths, with the listed trunk size.

use psmc_rs::bootstrap::{DEFAULT_BOOTSTRAP_TRUNK, bootstrap_trunk_count, split_for_bootstrap};

/// A sequence whose bins encode their position, so trunk contents can be checked.
fn seq_of_len(len: usize, salt: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 7 + salt) % 3) as u8).collect()
}

/// Split all `cases` (sequence length -> splitfa trunk lengths) in one call, as
/// splitfa does for a multi-record file, and check lengths, order and content.
fn check(trunk: usize, cases: &[(usize, Vec<usize>)]) {
    let seqs: Vec<Vec<u8>> = cases
        .iter()
        .enumerate()
        .map(|(i, (len, _))| seq_of_len(*len, i))
        .collect();
    let trunks = split_for_bootstrap(&seqs, trunk);

    let want: Vec<usize> = cases.iter().flat_map(|(_, l)| l.iter().copied()).collect();
    let got: Vec<usize> = trunks.iter().map(Vec::len).collect();
    assert_eq!(got, want, "trunk lengths for trunk_size {trunk}");

    let mut next = 0;
    for (seq, (_, lens)) in seqs.iter().zip(cases) {
        let rebuilt: Vec<u8> = trunks[next..next + lens.len()].concat();
        assert_eq!(&rebuilt, seq, "trunks must tile the sequence in order");
        next += lens.len();
    }
}

#[test]
fn matches_splitfa_trunk_10() {
    let t = 10;
    // `n` full trunks, then one of `last` bins.
    let then = |n: usize, last: usize| [vec![t; n], vec![last]].concat();
    let cases = vec![
        (1, vec![1]),
        (9, vec![9]),
        (10, vec![10]),
        (14, vec![14]),
        (15, vec![10, 5]),
        (16, vec![10, 6]),
        (19, vec![10, 9]),
        (20, vec![10, 10]),
        (24, vec![10, 14]),
        (25, vec![10, 10, 5]),
        (26, vec![10, 10, 6]),
        (34, vec![10, 10, 14]),
        (35, vec![10, 10, 10, 5]),
        (36, vec![10, 10, 10, 6]),
        (100, vec![t; 10]),
        (101, then(9, 11)),
        (104, then(9, 14)),
        (105, then(10, 5)),
    ];
    check(t, &cases);
}

#[test]
fn matches_splitfa_odd_trunk_uses_integer_tail_limit() {
    // 7 * 3 / 2 == 10 in C integer arithmetic: a remainder of exactly 10 is
    // still split, leaving a last trunk of 3 (< trunk / 2).
    check(
        7,
        &[
            (1, vec![1]),
            (6, vec![6]),
            (7, vec![7]),
            (9, vec![9]),
            (10, vec![7, 3]),
            (11, vec![7, 4]),
            (13, vec![7, 6]),
            (14, vec![7, 7]),
            (16, vec![7, 9]),
            (17, vec![7, 7, 3]),
            (18, vec![7, 7, 4]),
            (20, vec![7, 7, 6]),
            (21, vec![7, 7, 7]),
            (24, vec![7, 7, 7, 3]),
            (25, vec![7, 7, 7, 4]),
        ],
    );
}

#[test]
fn matches_splitfa_tiny_trunks() {
    check(1, &(1..=6).map(|n| (n, vec![1; n])).collect::<Vec<_>>());
    check(
        2,
        &[
            (1, vec![1]),
            (2, vec![2]),
            (3, vec![2, 1]),
            (4, vec![2, 2]),
            (5, vec![2, 2, 1]),
            (6, vec![2, 2, 2]),
        ],
    );
    check(
        3,
        &[
            (1, vec![1]),
            (2, vec![2]),
            (3, vec![3]),
            (4, vec![3, 1]),
            (5, vec![3, 2]),
            (6, vec![3, 3]),
            (7, vec![3, 3, 1]),
        ],
    );
}

#[test]
fn matches_splitfa_default_trunk() {
    assert_eq!(DEFAULT_BOOTSTRAP_TRUNK, 500_000);
    let t = DEFAULT_BOOTSTRAP_TRUNK;
    check(
        t,
        &[
            (1000, vec![1000]),
            (749_999, vec![749_999]),
            (750_000, vec![t, 250_000]),
            (1_000_000, vec![t, t]),
            (1_249_999, vec![t, 749_999]),
            (1_250_000, vec![t, t, 250_000]),
            (1_750_000, vec![t, t, t, 250_000]),
        ],
    );
}

#[test]
fn sequences_shorter_than_a_trunk_stay_whole() {
    let seqs: Vec<Vec<u8>> = (0..40).map(|i| seq_of_len(10_000, i)).collect();
    let trunks = split_for_bootstrap(&seqs, DEFAULT_BOOTSTRAP_TRUNK);
    assert_eq!(trunks, seqs);
}

#[test]
fn empty_sequences_yield_no_trunks() {
    assert!(split_for_bootstrap(&[], 10).is_empty());
    let seqs = vec![Vec::new(), seq_of_len(12, 0), Vec::new()];
    let trunks = split_for_bootstrap(&seqs, 10);
    assert_eq!(trunks, vec![seq_of_len(12, 0)]);
}

#[test]
#[should_panic(expected = "trunk_size must be > 0")]
fn zero_trunk_size_panics() {
    split_for_bootstrap(&[vec![0u8; 5]], 0);
}

#[test]
fn trunk_count_matches_split() {
    for trunk in [1, 2, 3, 7, 10] {
        for a in 0..40 {
            for b in [0, 1, 5, 17] {
                let seqs = vec![seq_of_len(a, 0), seq_of_len(b, 1)];
                let n = split_for_bootstrap(&seqs, trunk).len();
                let want = if n >= 2 { Some(n) } else { None };
                assert_eq!(
                    bootstrap_trunk_count(&seqs, trunk).ok(),
                    want,
                    "lengths {a}, {b}, trunk {trunk}"
                );
            }
        }
    }
}

#[test]
fn trunk_count_rejects_unusable_inputs() {
    assert!(bootstrap_trunk_count(&[seq_of_len(5, 0)], 0).is_err());
    assert!(bootstrap_trunk_count(&[], 10).is_err());
    assert!(bootstrap_trunk_count(&[Vec::new(), Vec::new()], 10).is_err());
    let err = bootstrap_trunk_count(&[seq_of_len(14, 0)], 10).unwrap_err();
    assert!(err.to_string().contains("single trunk"), "{err}");
    assert_eq!(bootstrap_trunk_count(&[seq_of_len(15, 0)], 10).unwrap(), 2);
    assert_eq!(
        bootstrap_trunk_count(&[seq_of_len(14, 0), seq_of_len(1, 1)], 10).unwrap(),
        2
    );
}

/// The error for a single trunk suggests a trunk size of at most 2/3 of the
/// input; that size must always give at least two trunks.
#[test]
fn suggested_trunk_size_gives_two_trunks() {
    for bins in 2..2000 {
        let trunks = split_for_bootstrap(&[seq_of_len(bins, 0)], bins * 2 / 3);
        assert!(trunks.len() >= 2, "{bins} bins");
    }
}
