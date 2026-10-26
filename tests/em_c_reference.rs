//! The E-step quantities of C psmc's `.psmc` rounds (`post_sigma`, `RI`)
//! against C's output on `tests/data/c_reference` and on the fragmented
//! input of `tests/data/psmc_out` (see the READMEs there).
//!
//! Round 1's E-step runs at C's initial parameters, which ours reproduce, so
//! C's printed values are matched to every printed decimal once the one
//! known difference in the expectations is undone: C's `hmm_expect` skips the
//! emission of the last bin of each sequence. How much that difference
//! matters depends on the share of such bins in the data.

mod common;

use common::psmc_text::round_lines;
use common::{c_start, e_step, fixture, fixture_obs, read};
use psmc_rs::PsmcModel;
use psmc_rs::em::posterior_sigma;

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
    /// Relative difference of `RI` at C's round-1 `sigma`.
    ri: f64,
}

/// Runs our E-step at C's starting parameters on the psmcfa fixture
/// `psmcfa` and compares it with round 1 of `c_file`, C's `psmc -p pattern`
/// output on the same file.
fn check(psmcfa: &str, pattern: &str, c_file: &str) -> Gap {
    let obs = fixture_obs(psmcfa);
    assert!(obs.row_starts.iter().all(|&s| s), "one row per sequence");
    let model = c_start(&obs.rows, pattern, 4.0);
    let mut stats = e_step(&model, &obs.rows, &obs.row_starts).stats;
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
    let post_c = posterior_sigma(&stats);

    let c_text = read(fixture(c_file));
    let round = round_lines(&c_text, 1);
    let field = |tag: &str| {
        round
            .iter()
            .find_map(|l| l.strip_prefix(tag))
            .unwrap_or_else(|| panic!("no {tag} line"))
    };
    let rs: Vec<Vec<&str>> = round
        .iter()
        .filter_map(|l| l.strip_prefix("RS\t"))
        .map(|l| l.split('\t').collect())
        .collect();
    assert_eq!(rs.len(), model.n_steps + 1);
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
        ri: ((ri_all_bins - ri_c) / ri_c).abs(),
    }
}

/// `RI` moves by a relative amount of the order of the share of skipped
/// bins (measured: 0.49 to 0.74 times it).
fn check_scale(gap: &Gap) {
    assert!(
        gap.ri > gap.share / 10.0 && gap.ri < 3.0 * gap.share,
        "RI moves by {}, the share is {}",
        gap.ri,
        gap.share
    );
}

/// 3 sequences, 150,947 non-missing bins: the share is 2.0e-5.
fn check_reference(pattern: &str, c_file: &str) {
    let gap = check(
        common::REFERENCE_PSMCFA,
        pattern,
        &format!("c_reference/{c_file}"),
    );
    assert_eq!(gap.share, 3.0 / 150_947.0);
    // Measured: post_sigma 2.4e-6 (4+5*3+4) and 8.6e-7 (4+25*2+4+6), RI
    // 1.5e-5.
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
        "psmc_out/fragmented.psmcfa",
        "4+5*3+4",
        "psmc_out/c_4+5x3+4_fragmented_N1.psmc",
    );
    assert_eq!(gap.share, 893.0 / 35_465.0);
    // Measured: post_sigma 9.3e-5, RI 1.2%.
    assert!(gap.post_sigma > 1e-5, "{gap:?}");
    check_scale(&gap);
}
