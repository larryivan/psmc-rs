//! Equivalence with Heng Li's C implementation (lh3/psmc 0.6.5-r74).
//!
//! `tests/data/c_reference/` holds C outputs on a 180,000-bin psmcfa (see the
//! README there). With C's initialisation (θ0 = −ln(1 − K/L), ρ0 = θ0/4,
//! λ = 1, t_max = 15) the first E-step sees identical parameters, so its
//! log-likelihood must agree to C's printed precision. Later rounds depend on
//! the M-step optimiser (C: Hooke-Jeeves; here: L-BFGS), which both run to
//! convergence, and must stay within a relative 2e-7 (see `check`).

mod common;

use common::psmc_text::parse_rounds;
use common::{REFERENCE_PSMCFA, c_fit, fixture, read};

fn check(pattern: &str, c_file: &str) {
    // LK of rounds 1.. of C's run (round 0 is the unfitted start).
    let c_lk: Vec<f64> = parse_rounds(&read(fixture(&format!("c_reference/{c_file}"))))[1..]
        .iter()
        .map(|r| r.lk)
        .collect();
    assert_eq!(c_lk.len(), 5, "expected 5 C rounds");

    let history = c_fit(REFERENCE_PSMCFA, Some(pattern), c_lk.len()).history;
    let rust_lk: Vec<f64> = history.loglike.iter().map(|(before, _)| *before).collect();

    // Round 1: same parameters, so equal up to C's 6 printed decimals.
    assert!(
        (rust_lk[0] - c_lk[0]).abs() < 1e-5,
        "{pattern}: first E-step log-likelihood {} != C {}",
        rust_lk[0],
        c_lk[0]
    );
    // Later rounds: measured within a relative 7.3e-8 (rounds 2 to 5; 6.8e-8
    // and 7.3e-8 at most for the two patterns), so 2e-7 leaves a margin of
    // 2.7. The rest is C's expected counts, which leave out the last bin of
    // each sequence (3 of 150,947 here, see psmc_rs::em::posterior_sigma),
    // and EM carries that into the next rounds: with those bins left out of
    // psmc-rs's counts too, rounds 2 to 5 agree to 2.8e-9 and 8.1e-10, and a
    // port of C's Hooke-Jeeves M-step in place of psmc-rs's L-BFGS gives
    // 2.4e-9 and 1.3e-9 (6.9e-8 and 7.5e-8 with every bin), so the M-steps
    // agree. With the M-step capped at 100 iterations, as psmc-rs 0.1's
    // was, 4+25*2+4+6 differed by 3.8e-7.
    for (i, (r, c)) in rust_lk.iter().zip(&c_lk).enumerate().skip(1) {
        let rel = ((r - c) / c).abs();
        assert!(
            rel < 2e-7,
            "{pattern}: round {} log-likelihood {r} vs C {c} (rel {rel:.2e})",
            i + 1
        );
    }
    // EM never decreases the likelihood.
    for w in rust_lk.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "{pattern}: likelihood decreased: {w:?}"
        );
    }
}

#[test]
fn matches_c_psmc_pattern_4_5x3_4() {
    check("4+5*3+4", "c_4+5x3+4_N5.psmc");
}

#[test]
fn matches_c_psmc_pattern_4_25x2_4_6() {
    check("4+25*2+4+6", "c_4+25x2+4+6_N5.psmc");
}
