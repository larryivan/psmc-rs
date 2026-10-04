//! The E-step quantities of C psmc's `.psmc` rounds (`QD` before the M-step,
//! `post_sigma`, `RI`) against C's output on `tests/data/c_reference` and on
//! the fragmented input of `tests/data/psmc_out` (see the READMEs there).
//!
//! Round 1's E-step runs at C's initial parameters, which ours reproduce, so
//! C's printed values are matched to every printed decimal once the one
//! known difference in the expectations is undone: C's `hmm_expect` skips the
//! emission of the last bin of each sequence. How much that difference
//! matters depends on the share of such bins in the data.

use psmc_rs::em::{hmm_q, hmm_q0, posterior_sigma};
use psmc_rs::hmm::{SufficientStats, e_step_streaming};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::model::n_steps_for_pattern;
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const DATA: &str = "tests/data/c_reference";
const FIXTURES: &str = "tests/data/psmc_out";

/// The lines of round `round` of a C `.psmc` file, from `RD` to `//`.
fn c_round(path: &Path, round: usize) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap();
    let start = format!("RD\t{round}");
    text.lines()
        .skip_while(|l| *l != start)
        .take_while(|l| *l != "//")
        .map(str::to_string)
        .collect()
}

/// The filtered state distribution at the last bin of `row`, which is the
/// posterior there (the backward vector of the last bin is uniform).
fn last_bin_posterior(model: &PsmcModel, row: &[u8]) -> Vec<f64> {
    let n = model.n_steps + 1;
    let (a, e) = (model.transition_matrix(), model.emission_matrix());
    let normalise = |v: &mut Vec<f64>| {
        let s: f64 = v.iter().sum();
        v.iter_mut().for_each(|x| *x /= s);
    };
    let mut f: Vec<f64> = (0..n)
        .map(|k| model.prior_matrix()[k] * e[(row[0] as usize, k)])
        .collect();
    normalise(&mut f);
    for &o in &row[1..] {
        f = (0..n)
            .map(|l| (0..n).map(|k| f[k] * a[(k, l)]).sum::<f64>() * e[(o as usize, l)])
            .collect();
        normalise(&mut f);
    }
    f
}

/// `sum_k sigma_k ln(sigma_k / post_sigma_k)`, C's `RI`.
fn relative_information(sigma: &[f64], post_sigma: &[f64]) -> f64 {
    sigma
        .iter()
        .zip(post_sigma)
        .map(|(&s, &p)| s * (s / p).ln())
        .sum()
}

/// How far our round-1 values, which keep every bin, are from C's.
#[derive(Debug)]
struct Gap {
    /// Sequences whose last bin is not missing, over the non-missing bins:
    /// the share of the data C's expectations leave out.
    share: f64,
    /// Largest absolute difference of a `post_sigma` value.
    post_sigma: f64,
    /// Relative difference of `QD` before the M-step.
    qd: f64,
    /// Relative difference of `RI` at C's round-1 `sigma`.
    ri: f64,
}

/// Runs our E-step at C's starting parameters on `psmcfa` and compares it
/// with round 1 of `c_file`, C's `psmc -p pattern` output on the same file.
fn check(psmcfa: &Path, pattern: &str, c_file: &Path) -> Gap {
    let obs = read_psmcfa(psmcfa, None).unwrap();
    assert!(obs.row_starts.iter().all(|&s| s), "one row per sequence");
    let theta0 = initial_theta(&obs.rows).unwrap();
    let n_steps = n_steps_for_pattern(pattern).unwrap();
    let model = PsmcModel::new(
        15.0,
        n_steps,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some(pattern.into()),
    )
    .unwrap();
    let mut stats: SufficientStats = e_step_streaming(
        model.prior_matrix(),
        model.transition_matrix(),
        model.emission_matrix(),
        &obs.rows,
        &obs.row_starts,
        false,
        "E",
    )
    .unwrap()
    .stats;
    let q = |s: &SufficientStats| {
        hmm_q(model.transition_matrix(), model.emission_matrix(), s) - hmm_q0(s)
    };
    let q_all_bins = q(&stats);
    let post_all_bins = posterior_sigma(&stats);

    let mut n_skipped = 0;
    for row in &obs.rows {
        let symbol = *row.last().unwrap() as usize;
        if symbol < 2 {
            n_skipped += 1;
            for (g, p) in stats.gobs[symbol]
                .iter_mut()
                .zip(last_bin_posterior(&model, row))
            {
                *g -= p;
            }
        }
    }
    let q_c = q(&stats);
    let post_c = posterior_sigma(&stats);

    let round = c_round(c_file, 1);
    let field = |tag: &str| {
        round
            .iter()
            .find_map(|l| l.strip_prefix(tag))
            .unwrap_or_else(|| panic!("no {tag} line"))
    };
    let (c_qd, _) = field("QD\t").split_once(" -> ").unwrap();
    assert_eq!(format!("{q_c:.6}"), c_qd, "{pattern}: QD before");
    let rs: Vec<Vec<&str>> = round
        .iter()
        .filter_map(|l| l.strip_prefix("RS\t"))
        .map(|l| l.split('\t').collect())
        .collect();
    assert_eq!(rs.len(), n_steps + 1);
    for (k, cols) in rs.iter().enumerate() {
        assert_eq!(
            format!("{:.6}", post_c[k]),
            cols[5],
            "{pattern}: post_sigma[{k}]"
        );
    }

    // Each value is (d_k - post_c_k D) / sum_L, where d_k (between 0 and
    // D) is the skipped bins' posterior mass in state k.
    let sum_l = obs.rows.iter().flatten().filter(|&&b| b < 2).count();
    let share = n_skipped as f64 / sum_l as f64;
    let post_sigma = post_all_bins
        .iter()
        .zip(&post_c)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0, f64::max);
    assert!(post_sigma <= share, "{pattern}: {post_sigma} > {share}");

    // C's sigma after the round-1 M-step, from the n_recomb * sigma_k
    // column, which has more significant digits than the sigma_k one.
    let n_recomb: f64 = field("MM\tC_pi: ")
        .split_once("n_recomb: ")
        .unwrap()
        .1
        .parse()
        .unwrap();
    let sigma: Vec<f64> = rs
        .iter()
        .map(|cols| cols[3].parse::<f64>().unwrap() / n_recomb)
        .collect();
    let c_ri: f64 = field("RI\t").parse().unwrap();
    let ri_c = relative_information(&sigma, &post_c);
    assert!(
        ((ri_c - c_ri) / c_ri).abs() < 1e-3,
        "{pattern}: RI {ri_c} vs C's {c_ri}"
    );
    let ri_all_bins = relative_information(&sigma, &post_all_bins);

    Gap {
        share,
        post_sigma,
        qd: ((q_all_bins - q_c) / q_c).abs(),
        ri: ((ri_all_bins - ri_c) / ri_c).abs(),
    }
}

/// `QD` and `RI` move by a relative amount of the order of the share of
/// skipped bins (measured: 0.49 to 2.03 times it).
fn check_scale(gap: &Gap) {
    for (what, rel) in [("QD", gap.qd), ("RI", gap.ri)] {
        assert!(
            rel > gap.share / 10.0 && rel < 3.0 * gap.share,
            "{what} moves by {rel}, the share is {}",
            gap.share
        );
    }
}

/// 3 sequences, 150,947 non-missing bins: the share is 2.0e-5.
fn check_reference(pattern: &str, c_file: &str) {
    let gap = check(
        &Path::new(DATA).join("papuan_chr1_3seg.psmcfa"),
        pattern,
        &Path::new(DATA).join(c_file),
    );
    assert_eq!(gap.share, 3.0 / 150_947.0);
    // Measured: post_sigma 2.4e-6 (4+5*3+4) and 8.6e-7 (4+25*2+4+6), QD
    // 4.0e-5, RI 1.5e-5.
    assert!(gap.post_sigma < 1e-5, "{gap:?}");
    check_scale(&gap);
}

#[test]
fn first_round_matches_c_pattern_4_5x3_4() {
    check_reference("4+5*3+4", "c_4+5x3+4_N5.psmc");
}

#[test]
fn first_round_matches_c_pattern_4_25x2_4_6() {
    check_reference("4+25*2+4+6", "c_4+25x2+4+6_N5.psmc");
}

/// 1,010 sequences of 1 to 80 bins, among them 10 of one bin, 40 with no
/// non-missing bin and 117 ending in a missing bin: the share is 893 of
/// 35,465 bins, 2.5%.
#[test]
fn first_round_matches_c_on_fragments() {
    let gap = check(
        &Path::new(FIXTURES).join("fragmented.psmcfa"),
        "4+5*3+4",
        &Path::new(FIXTURES).join("c_4+5x3+4_fragmented_N1.psmc"),
    );
    assert_eq!(gap.share, 893.0 / 35_465.0);
    // Measured: post_sigma 9.3e-5, QD 3.5%, RI 1.2%.
    assert!(gap.post_sigma > 1e-5, "{gap:?}");
    check_scale(&gap);
}
