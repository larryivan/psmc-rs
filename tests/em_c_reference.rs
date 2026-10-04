//! The E-step quantities of C psmc's `.psmc` rounds (`QD` before the M-step,
//! `post_sigma`) against C's output on `tests/data/c_reference` (see the
//! README there).
//!
//! Round 1's E-step runs at C's initial parameters, which ours reproduce, so
//! C's printed values are matched to every printed decimal once the one
//! known difference in the expectations is undone: C's `hmm_expect` skips the
//! emission of the last bin of each sequence.

use psmc_rs::em::{hmm_q, hmm_q0, posterior_sigma};
use psmc_rs::hmm::{SufficientStats, e_step_streaming};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::model::n_steps_for_pattern;
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const DATA: &str = "tests/data/c_reference";

/// The lines of round `round` of a C `.psmc` file, from `RD` to `//`.
fn c_round(file: &str, round: usize) -> Vec<String> {
    let text = std::fs::read_to_string(Path::new(DATA).join(file)).unwrap();
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

fn check(pattern: &str, c_file: &str) {
    let obs = read_psmcfa(&Path::new(DATA).join("papuan_chr1_3seg.psmcfa"), None).unwrap();
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

    for row in &obs.rows {
        let symbol = *row.last().unwrap() as usize;
        for (g, p) in stats.gobs[symbol]
            .iter_mut()
            .zip(last_bin_posterior(&model, row))
        {
            *g -= p;
        }
    }
    let q_c = q(&stats);
    let post_c = posterior_sigma(&stats);

    let round = c_round(c_file, 1);
    let qd = round.iter().find_map(|l| l.strip_prefix("QD\t")).unwrap();
    let (c_before, _) = qd.split_once(" -> ").unwrap();
    assert_eq!(format!("{q_c:.6}"), c_before, "{pattern}: QD before");
    // With every bin, Q differs by the three skipped bins' share only.
    let c_before: f64 = c_before.parse().unwrap();
    assert!(
        (q_all_bins - c_before).abs() < 1e-2,
        "{pattern}: {q_all_bins}"
    );

    let rs: Vec<&str> = round
        .iter()
        .filter_map(|l| l.strip_prefix("RS\t"))
        .map(|l| l.rsplit('\t').next().unwrap())
        .collect();
    assert_eq!(rs.len(), n_steps + 1);
    for (k, c_post) in rs.iter().enumerate() {
        assert_eq!(
            &format!("{:.6}", post_c[k]),
            c_post,
            "{pattern}: post_sigma[{k}]"
        );
        let c_post: f64 = c_post.parse().unwrap();
        assert!((post_all_bins[k] - c_post).abs() < 1e-5, "{pattern}: {k}");
    }
}

#[test]
fn first_round_matches_c_pattern_4_5x3_4() {
    check("4+5*3+4", "c_4+5x3+4_N5.psmc");
}

#[test]
fn first_round_matches_c_pattern_4_25x2_4_6() {
    check("4+25*2+4+6", "c_4+25x2+4+6_N5.psmc");
}
