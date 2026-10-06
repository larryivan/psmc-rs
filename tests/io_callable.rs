//! Callable bases per bin: the invariants of `Observations::validate` and
//! `Sequences::validate`, and the counts travelling with their bins when
//! sequences are cut into rows. How the counts are stored is tested in
//! `src/io/callable.rs`.

use psmc_rs::bootstrap::sequences_to_observations;
use psmc_rs::io::common::chunk_sequences;
use psmc_rs::io::{BinCounts, CallableBases, MAX_COUNTED_BIN_SIZE, Observations, Sequences};

/// SplitMix64: deterministic and dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
}

/// Random sequences of bins of `s` bases whose symbols agree with their
/// counts: N exactly when fewer than 10% of the bases are callable. Some
/// sequences are empty.
fn random_sequences(rng: &mut Rng, s: usize, n_seqs: usize) -> Sequences {
    let mut bins = Vec::new();
    let mut counts = Vec::new();
    for _ in 0..n_seqs {
        let len = match rng.range(0, 5) {
            0 => 0,
            1 => 1,
            _ => rng.range(2, 120) as usize,
        };
        let mut b = Vec::with_capacity(len);
        let mut c = Vec::with_capacity(len);
        // Mostly full bins, as in real data, or mostly partly called ones.
        let partial_odds = rng.range(1, 3);
        for _ in 0..len {
            let count = match rng.range(0, 3) {
                x if x < partial_odds => rng.range(0, s as u64),
                _ => s as u64,
            } as usize;
            let symbol = if 10 * count < s {
                2
            } else {
                rng.range(0, 1) as u8
            };
            b.push(symbol);
            c.push(count as u16);
        }
        bins.push(b);
        counts.push(c);
    }
    Sequences {
        bins,
        callable: Some(CallableBases::from_counts(s, counts).unwrap()),
    }
}

fn obs(rows: Vec<Vec<u8>>, row_starts: Vec<bool>, s: usize, counts: Vec<Vec<u16>>) -> Observations {
    Observations {
        rows,
        row_starts,
        callable: Some(CallableBases::from_counts(s, counts).unwrap()),
    }
}

fn error(o: &Observations) -> String {
    format!("{:#}", o.validate().unwrap_err())
}

#[test]
fn the_bin_size_must_fit_in_16_bits() {
    assert_eq!(MAX_COUNTED_BIN_SIZE, 65_535);
    for s in [0, MAX_COUNTED_BIN_SIZE + 1, usize::MAX] {
        let want = "between 1 and 65535";
        let err = CallableBases::new(s, Vec::new()).unwrap_err().to_string();
        assert!(err.contains(want), "{s}: {err}");
        let err = CallableBases::from_counts(s, Vec::new()).unwrap_err();
        assert!(err.to_string().contains(want), "{s}: {err}");
        let err = BinCounts::new(s, Vec::new()).unwrap_err().to_string();
        assert!(err.contains(want), "{s}: {err}");
    }
    for s in [1, 100, MAX_COUNTED_BIN_SIZE] {
        let c = CallableBases::from_counts(s, vec![vec![0, 1]]).unwrap();
        assert_eq!(c.bin_size(), s);
        assert_eq!(c.to_counts(), [vec![0, 1]]);
        let row = BinCounts::new(s, vec![0, 1]).unwrap();
        assert_eq!((row.bin_size(), row.len()), (s, 2));
        assert_eq!(c.rows(), std::slice::from_ref(&row));
        assert_eq!(CallableBases::new(s, vec![row]).unwrap(), c);
        let other = c.with_rows(vec![BinCounts::new(s, vec![5]).unwrap()]);
        assert_eq!(other.bin_size(), s);
        assert_eq!(other.clone().into_rows()[0].to_vec(), [5]);
    }
}

#[test]
fn symbols_and_counts_agree_on_the_ninety_percent_rule() {
    // s = 100: N exactly when fewer than 10 bases are callable.
    let ok = obs(
        vec![vec![2, 2, 0, 1, 0, 1, 2]],
        vec![true],
        100,
        vec![vec![0, 9, 10, 10, 100, 100, 9]],
    );
    ok.validate().unwrap();
    // A sequence's last bin follows the same rule.
    let ok = obs(
        vec![vec![0, 0], vec![1], vec![2]],
        vec![true, false, true],
        100,
        vec![vec![100, 100], vec![10], vec![0]],
    );
    ok.validate().unwrap();

    let t_with_9 = obs(vec![vec![0, 0]], vec![true], 100, vec![vec![100, 9]]);
    let err = error(&t_with_9);
    assert!(
        err.contains("row 0, bin 1") && err.contains("(T) with 9"),
        "{err}"
    );
    let k_with_0 = obs(vec![vec![1]], vec![true], 100, vec![vec![0]]);
    assert!(error(&k_with_0).contains("(K) with 0"));
    let n_with_10 = obs(
        vec![vec![0], vec![2]],
        vec![true, false],
        100,
        vec![vec![50], vec![10]],
    );
    let err = error(&n_with_10);
    assert!(
        err.contains("row 1, bin 0") && err.contains("(N) with 10"),
        "{err}"
    );

    // Small bins: with s = 1 a bin is N exactly when its base is missing,
    // with s = 7 or 10 when none of its bases is callable.
    for (s, n_max, called_min) in [
        (1, 0, 1),
        (7, 0, 1),
        (10, 0, 1),
        (11, 1, 2),
        (65_535, 6553, 6554),
    ] {
        obs(
            vec![vec![2, 0, 1]],
            vec![true],
            s,
            vec![vec![n_max, called_min, s as u16]],
        )
        .validate()
        .unwrap_or_else(|e| panic!("s {s}: {e:#}"));
        let n = obs(vec![vec![2]], vec![true], s, vec![vec![called_min]]);
        assert!(error(&n).contains("missing bin (N)"), "s {s}");
        let t = obs(vec![vec![0]], vec![true], s, vec![vec![n_max]]);
        assert!(error(&t).contains("called bin (T)"), "s {s}");
    }
}

#[test]
fn every_shape_and_range_violation_is_reported() {
    let rows = || vec![vec![0u8, 1], vec![0]];
    let starts = || vec![true, false];
    let counts = || vec![vec![100u16, 100], vec![100]];
    obs(rows(), starts(), 100, counts()).validate().unwrap();

    let cases: Vec<(Observations, &str)> = vec![
        (
            Observations {
                rows: rows(),
                row_starts: vec![true],
                callable: None,
            },
            "1 row starts for 2 rows",
        ),
        (
            Observations {
                rows: rows(),
                row_starts: vec![false, false],
                callable: None,
            },
            "first row must start a sequence",
        ),
        (
            obs(rows(), starts(), 100, vec![vec![100, 100]]),
            "callable bases for 1 rows, but there are 2 rows",
        ),
        (
            obs(rows(), starts(), 100, vec![vec![100, 100], vec![100, 100]]),
            "row 1 has 1 bins but callable bases for 2",
        ),
        (
            obs(rows(), starts(), 100, vec![vec![100, 101], vec![100]]),
            "row 0, bin 1: 101 callable bases in a bin of 100",
        ),
        (
            obs(vec![vec![0, 3]], vec![true], 100, vec![vec![100, 100]]),
            "invalid bin symbol 3",
        ),
        (
            Observations {
                rows: rows(),
                row_starts: starts(),
                callable: Some(
                    CallableBases::new(
                        100,
                        vec![
                            BinCounts::new(100, vec![100, 100]).unwrap(),
                            BinCounts::new(99, vec![99]).unwrap(),
                        ],
                    )
                    .unwrap(),
                ),
            },
            "row 1 has callable bases of bins of 99 bases, in counts of bins of 100",
        ),
    ];
    for (o, want) in cases {
        let err = error(&o);
        assert!(err.contains(want), "want {want:?}, got {err}");
    }

    // Without counts only the row starts are checked; empty input is valid.
    let plain = Observations {
        rows: rows(),
        row_starts: starts(),
        callable: None,
    };
    plain.validate().unwrap();
    let empty = Observations {
        rows: Vec::new(),
        row_starts: Vec::new(),
        callable: Some(CallableBases::new(100, Vec::new()).unwrap()),
    };
    empty.validate().unwrap();

    // Sequences are checked the same way, each entry its own sequence.
    let seqs = Sequences {
        bins: vec![vec![0], vec![2, 2]],
        callable: Some(CallableBases::from_counts(100, vec![vec![100], vec![0, 50]]).unwrap()),
    };
    let err = format!("{:#}", seqs.validate().unwrap_err());
    assert!(
        err.contains("sequence 1, bin 1") && err.contains("(N) with 50"),
        "{err}"
    );
    let seqs = Sequences {
        bins: vec![vec![0], vec![]],
        callable: Some(CallableBases::from_counts(100, vec![vec![100]]).unwrap()),
    };
    let err = format!("{:#}", seqs.validate().unwrap_err());
    assert!(
        err.contains("callable bases for 1 sequences, but there are 2"),
        "{err}"
    );
}

#[test]
fn chunking_cuts_the_counts_at_the_same_bins() {
    let mut rng = Rng(7);
    for case in 0..200 {
        let s = [1, 7, 100, MAX_COUNTED_BIN_SIZE][case % 4];
        let seqs = random_sequences(&mut rng, s, 1 + case % 6);
        seqs.validate().unwrap();
        if seqs.bins.iter().all(Vec::is_empty) {
            let err = chunk_sequences(seqs, None, "nothing left").unwrap_err();
            assert_eq!(err.to_string(), "nothing left");
            continue;
        }
        let whole_counts = seqs.callable.as_ref().unwrap().to_counts();
        for batch in [None, Some(1), Some(3), Some(40), Some(1000)] {
            let ctx = format!("case {case}, batch {batch:?}");
            let o = chunk_sequences(seqs.clone(), batch, "empty").unwrap();
            o.validate().unwrap();
            // The bins and row starts are those of the bins alone.
            let plain = chunk_sequences(seqs.bins.clone(), batch, "empty").unwrap();
            assert_eq!(o.rows, plain.rows, "{ctx}");
            assert_eq!(o.row_starts, plain.row_starts, "{ctx}");
            assert!(plain.callable.is_none(), "{ctx}");
            let c = o.callable.as_ref().unwrap();
            assert_eq!(c.bin_size(), s, "{ctx}");
            assert_eq!(c.rows().len(), o.rows.len(), "{ctx}");
            // Rejoined, the rows of each sequence give back its counts.
            let mut joined: Vec<Vec<u16>> = Vec::new();
            for (row, &start) in c.rows().iter().zip(&o.row_starts) {
                if start {
                    joined.push(Vec::new());
                }
                joined.last_mut().unwrap().extend(row);
            }
            let non_empty: Vec<Vec<u16>> = whole_counts
                .iter()
                .filter(|c| !c.is_empty())
                .cloned()
                .collect();
            assert_eq!(joined, non_empty, "{ctx}");
            if let Some(b) = batch {
                assert!(c.rows().iter().all(|r| r.len() <= b), "{ctx}");
            }
            // Each row's counts take no more than 2 bytes per bin.
            for row in c.rows() {
                assert!(row.heap_bytes() <= 2 * row.len(), "{ctx}");
            }
        }
    }
}

#[test]
fn chunking_checks_the_counts_first() {
    let bad = Sequences {
        bins: vec![vec![0, 0]],
        callable: Some(CallableBases::from_counts(100, vec![vec![100]]).unwrap()),
    };
    let err = format!("{:#}", chunk_sequences(bad, None, "empty").unwrap_err());
    assert!(
        err.contains("sequence 0 has 2 bins but callable bases for 1"),
        "{err}"
    );
    let inconsistent = Sequences {
        bins: vec![vec![2]],
        callable: Some(CallableBases::from_counts(100, vec![vec![100]]).unwrap()),
    };
    let err = format!(
        "{:#}",
        chunk_sequences(inconsistent, Some(5), "empty").unwrap_err()
    );
    assert!(err.contains("missing bin (N) with 100"), "{err}");
    let err = chunk_sequences(vec![vec![0u8]], Some(0), "empty").unwrap_err();
    assert!(err.to_string().contains("batch_size must be > 0"));
}

#[test]
fn sequences_to_observations_matches_chunking_without_counts() {
    let seqs = vec![vec![0u8, 1, 2, 0, 0], vec![], vec![1, 1, 0]];
    for batch in [None, Some(1), Some(2), Some(9)] {
        let a = sequences_to_observations(&seqs, batch).unwrap();
        let b = chunk_sequences(seqs.clone(), batch, "empty").unwrap();
        assert_eq!((&a.rows, &a.row_starts), (&b.rows, &b.row_starts));
        assert!(a.callable.is_none());
    }
    assert!(sequences_to_observations(&[], None).is_err());
    assert!(sequences_to_observations(&[vec![]], None).is_err());
}
