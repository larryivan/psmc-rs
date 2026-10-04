//! The bootstrap RNG must give the same stream on every platform and with
//! every `rand` release, so a seed reproduces a bootstrap run anywhere.
//!
//! The pinned values were cross-checked against an independent Python
//! implementation of SplitMix64 and the ChaCha8 block function.

use psmc_rs::bootstrap::{new_rng, replicate_rng, replicate_seed};
use rand::RngCore;

#[test]
fn replicate_seeds_are_the_splitmix64_stream() {
    // Reference outputs of SplitMix64 seeded with 1234567 (Vigna's splitmix64.c).
    let expected = [
        6457827717110365317u64,
        3203168211198807973,
        9817491932198370423,
        4593380528125082431,
        16408922859458223821,
    ];
    for (rep, want) in expected.iter().enumerate() {
        assert_eq!(replicate_seed(1234567, rep), *want, "replicate {rep}");
    }
    assert_eq!(replicate_seed(0, 0), 0xE220_A839_7B1D_CDAF);
}

#[test]
fn new_rng_stream_is_pinned() {
    let mut rng = new_rng(42);
    let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
    assert_eq!(
        got,
        [
            0x3115_9EF9_87C9_1AFC,
            0x1755_9844_B416_9001,
            0xF7D0_AFBF_9AD9_A69F,
            0xB920_7AD5_FD37_495A,
        ]
    );
}

#[test]
fn replicate_rng_stream_is_pinned() {
    let mut rng = replicate_rng(42, 3);
    let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
    assert_eq!(
        got,
        [
            0xEC90_A523_13CE_395B,
            0x35B6_2645_A14B_EA63,
            0x0A6A_CF43_D3A0_1814,
            0x30CE_4CAB_A39A_4F82,
        ]
    );

    // Past the first ChaCha block (16 u32 words = 8 u64 per block).
    let mut rng = replicate_rng(7, 0);
    let got: Vec<u64> = (0..22).map(|_| rng.next_u64()).collect();
    assert_eq!(got[20..], [0xA8F0_0EF6_6FB8_629E, 0x0DFE_2E4E_9B06_2059]);
}

#[test]
fn replicate_streams_differ_and_repeat() {
    let first: Vec<u64> = (0..8)
        .map(|rep| replicate_rng(42, rep).next_u64())
        .collect();
    for i in 0..first.len() {
        for j in (i + 1)..first.len() {
            assert_ne!(first[i], first[j], "replicates {i} and {j} share a stream");
        }
    }
    let again: Vec<u64> = (0..8)
        .map(|rep| replicate_rng(42, rep).next_u64())
        .collect();
    assert_eq!(first, again);
    assert_ne!(
        replicate_rng(42, 0).next_u64(),
        replicate_rng(43, 0).next_u64()
    );
}
