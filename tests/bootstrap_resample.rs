//! `resample_chunks` must follow psmc's `psmc_resamp` (aux.c) rule exactly,
//! keep the resampled length close to the original, and be reproducible from
//! the seed alone.

use psmc_rs::bootstrap::{
    DEFAULT_BOOTSTRAP_TRUNK, new_rng, replicate_rng, resample_chunk_indices, resample_chunks,
    split_for_bootstrap,
};
use rand::RngCore;
use rayon::prelude::*;

/// POSIX `drand48` / `srand48`, the generator C psmc draws from.
struct Drand48(u64);

impl Drand48 {
    fn new(seed: u32) -> Self {
        Self((u64::from(seed) << 16) | 0x330E)
    }

    fn next(&mut self) -> f64 {
        self.0 = 0x5_DEEC_E66D_u64.wrapping_mul(self.0).wrapping_add(0xB) & ((1u64 << 48) - 1);
        self.0 as f64 / (1u64 << 48) as f64
    }
}

/// (srand48 seed, sequence lengths, kept indices, kept total, next drand48 state)
type CResampCase = (u32, &'static [usize], &'static [usize], usize, u64);

/// Expected draws were produced by linking the original `aux.c` into a small
/// harness that calls `srand48(seed)`, then `psmc_resamp()` on sequences of the
/// given lengths, and prints the indices of the sequences it kept, their total
/// length, and the next `drand48` state (which shows how many draws it used).
#[test]
fn matches_c_psmc_resamp_draw_for_draw() {
    #[rustfmt::skip]
    let cases: &[CResampCase] = &[
        (11, &[10, 10, 10, 10, 15], &[3, 2, 1, 4, 2], 55, 45_442_290_741_244),
        (12, &[10, 10, 10, 10, 15], &[3, 1, 4, 2, 2], 55, 106_231_780_995_068),
        (7, &[7, 7, 7, 3], &[1, 2, 1], 21, 139_218_870_111_509),
        (123, &[500_000, 500_000, 749_999, 1000, 250_000], &[1, 2, 4, 0], 1_999_999, 98_465_758_113_788),
        (5, &[3, 1, 4, 1, 5, 9, 2, 6], &[4, 2, 6, 1, 4, 6, 1, 3, 2, 6, 7], 33, 127_173_971_998_074),
        (99, &[100], &[0], 100, 141_964_139_586_424),
        (2024, &[13, 8, 21, 5], &[2, 3, 2], 47, 23_804_546_540_274),
        (1, &[1, 1, 1, 1000], &[0, 1, 3, 1], 1003, 159_171_116_698_901),
        (3, &[1, 1, 1, 1000], &[3], 1000, 87_738_098_657_827),
        (8, &[20, 1, 1], &[0, 1], 21, 167_373_930_390_258),
        (42, &[5; 12], &[8, 4, 1, 5, 0, 10, 5, 5, 8, 10, 5, 6], 60, 150_299_970_445_565),
        (77, &[1000, 999, 998, 1, 2, 3], &[1, 0, 5, 4, 4, 2], 3004, 235_773_896_201_815),
        // Overshoot equal to the deficit: C rejects the draw and stops short.
        (1, &[4, 2], &[0], 4, 234_980_157_041_187),
        (3, &[4, 2], &[1, 1], 4, 75_161_597_731_570),
        (2, &[4, 2], &[1, 0], 6, 161_359_127_849_507),
        // Landing exactly on the original total stops without another draw.
        (1, &[3, 3, 3], &[0, 1, 2], 9, 94_571_660_010_226),
        (3, &[3, 3, 3], &[2, 2, 0], 9, 75_161_597_731_570),
    ];
    for &(seed, lengths, want, want_total, next_state) in cases {
        let mut d = Drand48::new(seed);
        let got = resample_chunk_indices(lengths, |n| (n as f64 * d.next()) as usize);
        assert_eq!(got, want, "seed {seed}, lengths {lengths:?}");
        let total: usize = got.iter().map(|&i| lengths[i]).sum();
        assert_eq!(total, want_total, "seed {seed}");
        d.next();
        assert_eq!(
            d.0, next_state,
            "seed {seed}: number of draws differs from C"
        );
    }
}

fn random_lengths(rng: &mut impl RngCore, n: usize, max_len: u64) -> Vec<usize> {
    (0..n)
        .map(|_| 1 + (rng.next_u64() % max_len) as usize)
        .collect()
}

#[test]
fn resampled_total_is_within_half_a_trunk_of_the_original() {
    let mut gen_rng = new_rng(2024);
    for case in 0..300 {
        let n = 1 + (gen_rng.next_u64() % 60) as usize;
        let max_len = 1 + gen_rng.next_u64() % 5000;
        let lengths = random_lengths(&mut gen_rng, n, max_len);
        let chunks: Vec<Vec<u8>> = lengths.iter().map(|&l| vec![0u8; l]).collect();
        let target: usize = lengths.iter().sum();
        let longest = *lengths.iter().max().unwrap();

        let sampled = resample_chunks(&chunks, &mut replicate_rng(case, 0));
        assert!(!sampled.is_empty());
        let total: usize = sampled.iter().map(Vec::len).sum();
        assert!(
            2 * total.abs_diff(target) <= longest,
            "case {case}: total {total} vs target {target}, longest trunk {longest}"
        );
    }
}

#[test]
fn resampled_chunks_are_copies_of_input_chunks() {
    let mut gen_rng = new_rng(7);
    let chunks: Vec<Vec<u8>> = (0..25)
        .map(|i| {
            let len = 50 + (gen_rng.next_u64() % 200) as usize;
            (0..len).map(|j| ((i + j) % 3) as u8).collect()
        })
        .collect();
    for rep in 0..20 {
        for c in resample_chunks(&chunks, &mut replicate_rng(1, rep)) {
            assert!(chunks.contains(&c));
        }
    }
}

#[test]
fn resampling_is_deterministic_per_seed() {
    let chunks: Vec<Vec<u8>> = (0..30).map(|i| vec![(i % 3) as u8; 10 + i]).collect();
    let a = resample_chunks(&chunks, &mut replicate_rng(42, 5));
    let b = resample_chunks(&chunks, &mut replicate_rng(42, 5));
    assert_eq!(a, b);
    let c = resample_chunks(&chunks, &mut replicate_rng(42, 6));
    let d = resample_chunks(&chunks, &mut replicate_rng(43, 5));
    assert_ne!(a, c);
    assert_ne!(a, d);
}

/// Pins the whole pipeline (SplitMix64 seed, ChaCha8 stream, index sampling,
/// resampling rule); the expected draw was cross-checked with an independent
/// Python implementation.
#[test]
fn resampling_is_pinned_for_a_seed() {
    let lengths = [10usize, 20, 30, 40, 50, 60, 70, 80];
    let chunks: Vec<Vec<u8>> = lengths.iter().map(|&l| vec![0u8; l]).collect();
    let sampled: Vec<usize> = resample_chunks(&chunks, &mut replicate_rng(42, 0))
        .iter()
        .map(Vec::len)
        .collect();
    assert_eq!(sampled, [50, 60, 80, 70, 20, 20, 30]);
}

#[test]
fn resampling_does_not_depend_on_thread_count() {
    let chunks: Vec<Vec<u8>> = (0..50)
        .map(|i| (0..(100 + 13 * i)).map(|j| ((i * j) % 3) as u8).collect())
        .collect();
    let run = |threads: usize| -> Vec<Vec<Vec<u8>>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                (0..64)
                    .into_par_iter()
                    .map(|rep| resample_chunks(&chunks, &mut replicate_rng(9, rep)))
                    .collect()
            })
    };
    let one = run(1);
    assert_eq!(one, run(4));
    assert_eq!(one, run(7));
}

/// Inputs made of sequences shorter than a trunk (scaffold-level assemblies)
/// must still get resampling variance: whole sequences are drawn with
/// replacement, as C psmc does after `splitfa`.
#[test]
fn short_sequences_get_resampling_variance() {
    let mut gen_rng = new_rng(11);
    let seqs: Vec<Vec<u8>> = (0..40)
        .map(|_| {
            // Heterozygosity varies between sequences, as it does along a genome.
            let p_het = 1 + gen_rng.next_u64() % 10;
            (0..10_000)
                .map(|_| u8::from(gen_rng.next_u64() % 100 < p_het))
                .collect()
        })
        .collect();
    let chunks = split_for_bootstrap(&seqs, DEFAULT_BOOTSTRAP_TRUNK);
    let het = |s: &[Vec<u8>]| -> usize { s.iter().flatten().filter(|&&v| v == 1).count() };
    let original = het(&seqs);

    let mut counts = Vec::new();
    for rep in 0..20 {
        let sampled = resample_chunks(&chunks, &mut replicate_rng(3, rep));
        assert_ne!(
            sampled, seqs,
            "replicate {rep} returned the input unchanged"
        );
        counts.push(het(&sampled));
    }
    let distinct = {
        let mut c = counts.clone();
        c.sort_unstable();
        c.dedup();
        c.len()
    };
    assert!(distinct > 10, "het counts barely vary: {counts:?}");
    assert!(counts.iter().any(|&c| c != original));
}
