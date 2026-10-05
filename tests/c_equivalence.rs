//! Equivalence with Heng Li's C implementation (lh3/psmc 0.6.5-r74).
//!
//! `tests/data/c_reference/` holds C outputs on a 180,000-bin psmcfa (see the
//! README there). With C's initialisation (θ0 = −ln(1 − K/L), ρ0 = θ0/4,
//! λ = 1, t_max = 15) the first E-step sees identical parameters, so its
//! log-likelihood must agree to C's printed precision. Later rounds depend on
//! the M-step optimiser (C: Hooke-Jeeves; here: L-BFGS) and must stay close.

use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const DATA: &str = "tests/data/c_reference";

/// LK values of rounds 1.. from a C `.psmc` file (round 0 is the unfitted start).
fn c_loglikes(path: &Path) -> Vec<f64> {
    let text = std::fs::read_to_string(path).expect("missing C reference output");
    let mut out = Vec::new();
    let mut round: Option<usize> = None;
    for line in text.lines() {
        let mut cols = line.split('\t');
        match cols.next() {
            Some("RD") => round = cols.next().and_then(|v| v.parse().ok()),
            Some("LK") if round.unwrap_or(0) > 0 => {
                out.push(cols.next().unwrap().parse().expect("bad LK"));
            }
            _ => {}
        }
    }
    out
}

fn n_states(pattern: &str) -> usize {
    pattern
        .split('+')
        .map(|t| match t.split_once('*') {
            Some((a, b)) => a.parse::<usize>().unwrap() * b.parse::<usize>().unwrap(),
            None => t.parse::<usize>().unwrap(),
        })
        .sum()
}

fn check(pattern: &str, c_file: &str) {
    let obs = read_psmcfa(&Path::new(DATA).join("papuan_chr1_3seg.psmcfa"), None)
        .expect("failed to read reference psmcfa");
    let c_lk = c_loglikes(&Path::new(DATA).join(c_file));
    assert_eq!(c_lk.len(), 5, "expected 5 C rounds");

    let theta0 = initial_theta(&obs.rows).unwrap();
    let mut model = PsmcModel::new(
        15.0,
        n_states(pattern) - 1,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some(pattern.to_string()),
    )
    .unwrap();
    let cfg = MStepConfig {
        progress: false,
        lambda: 0.0,
        ..MStepConfig::default()
    };
    let history = em_train(&mut model, &obs.rows, &obs.row_starts, c_lk.len(), &cfg).unwrap();
    let rust_lk: Vec<f64> = history.loglike.iter().map(|(before, _)| *before).collect();

    // Round 1: same parameters, so equal up to C's 6 printed decimals.
    assert!(
        (rust_lk[0] - c_lk[0]).abs() < 1e-5,
        "{pattern}: first E-step log-likelihood {} != C {}",
        rust_lk[0],
        c_lk[0]
    );
    for (i, (r, c)) in rust_lk.iter().zip(&c_lk).enumerate().skip(1) {
        let rel = ((r - c) / c).abs();
        assert!(
            rel < 1e-5,
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
