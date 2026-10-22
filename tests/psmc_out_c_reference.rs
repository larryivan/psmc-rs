//! `.psmc` output against C psmc's (lh3/psmc 0.6.5-r74) on the reference
//! data of `tests/c_equivalence.rs`: 5 EM rounds from C's initialisation.
//!
//! The header and round 0 depend only on the data and the starting
//! parameters, so they must equal C's as text (apart from the `Version`
//! line, and C's `QD` lines, which psmc-rs does not write). Later rounds
//! depend on the M-step (L-BFGS here, Hooke-Jeeves in C), which both run to
//! convergence, and agree within tolerances in line with
//! `tests/c_equivalence.rs`, with and without a pattern.
//!
//! A run of one round on many short sequences checks the header counts and
//! round 0 on fragmented input.
//!
//! The C files are in `tests/data/c_reference` and, for the pattern-free
//! model and the short sequences, `tests/data/psmc_out` (see the READMEs
//! there).

mod common;

use common::psmc_text::{parse_round, round_lines, tag};
use common::{CFit, REFERENCE_PSMCFA, c_fit, fixture, model, model_at, mstep, read};
use psmc_rs::PsmcModel;
use psmc_rs::em::EmHistory;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::{MStepStop, em_train};
use psmc_rs::psmc_out::{PsmcRunInfo, c_pattern, psmc_rounds, write_psmc, write_psmc_from_history};
use std::sync::OnceLock;

const N_ITER: usize = 5;
const FRAGMENTS: &str = "psmc_out/fragmented.psmcfa";

/// [`c_fit`] of `n_iter` rounds on `name`, as in the CLI. Every M-step must
/// end by its convergence criteria, as C's Hooke-Jeeves search does, not at
/// the cap.
fn checked_fit(name: &str, pattern: Option<&str>, n_iter: usize) -> CFit {
    let fit = c_fit(name, pattern, n_iter);
    let (stops, iterations) = (&fit.history.mstep_stops, &fit.history.mstep_iterations);
    assert!(
        stops.iter().all(|&s| s != MStepStop::MaxIters),
        "{pattern:?}: {stops:?} after {iterations:?} iterations"
    );
    fit
}

/// C's run on the reference data: `N_ITER` rounds with `pattern`, or
/// without one 23 states with a λ each (C's `-p "23*1"`). Each is fitted
/// once, for every test that reads it.
fn fit(pattern: Option<&str>) -> &'static CFit {
    static FITS: [OnceLock<CFit>; 3] = [const { OnceLock::new() }; 3];
    let i = match pattern {
        Some("4+5*3+4") => 0,
        Some("4+25*2+4+6") => 1,
        None => 2,
        Some(p) => panic!("no fit with pattern {p}"),
    };
    FITS[i].get_or_init(|| checked_fit(REFERENCE_PSMCFA, pattern, N_ITER))
}

fn psmc_text(fit: &CFit) -> String {
    let mut out = Vec::new();
    let CFit {
        obs,
        template,
        history,
    } = fit;
    write_psmc_from_history(&mut out, template, history, &obs.rows, &obs.row_starts).unwrap();
    String::from_utf8(out).unwrap()
}

fn rel(a: f64, b: f64) -> f64 {
    ((a - b) / b).abs()
}

/// `line` with its values blanked out but their printed precision kept: a
/// run of digits after a decimal point becomes as many `d`s, any other run
/// of digits one `#`. Two lines of the same shape print the same fields
/// with the same `printf` conversions.
fn shape(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let (mut in_digits, mut in_fraction) = (false, false);
    for c in line.chars() {
        if c.is_ascii_digit() {
            if in_fraction {
                out.push('d');
            } else if !in_digits {
                out.push('#');
            }
            in_digits = true;
        } else {
            out.push(c);
            in_digits = false;
            in_fraction = c == '.';
        }
    }
    out
}

/// C's `.psmc` text `c` without its `QD` lines.
fn without_qd(c: &str) -> String {
    c.lines()
        .filter(|l| tag(l) != "QD")
        .map(|l| format!("{l}\n"))
        .collect()
}

/// Checks the format: line tags in C's order but `QD`, the header (but
/// `Version`) and round 0 identical to C's, and every line of the later
/// rounds printed with C's conversions (see [`shape`]).
fn check_format(ours: &str, c: &str) {
    let c = &without_qd(c);
    fn tags(text: &str) -> Vec<&str> {
        text.lines().map(tag).collect()
    }
    fn header(text: &str) -> Vec<&str> {
        text.lines().take_while(|l| tag(l) != "RD").collect()
    }
    assert_eq!(tags(ours), tags(c), "line tags differ");

    let (h_ours, h_c) = (header(ours), header(c));
    assert_eq!(h_ours.len(), h_c.len());
    for (a, b) in h_ours.iter().zip(&h_c) {
        if b.starts_with("MM\tVersion: ") {
            assert_eq!(
                *a,
                format!("MM\tVersion: psmc-rs {}", env!("CARGO_PKG_VERSION"))
            );
        } else {
            assert_eq!(a, b);
        }
    }
    assert_eq!(round_lines(ours, 0), round_lines(c, 0), "round 0 differs");

    for (a, b) in ours.lines().zip(c.lines()).skip(h_c.len()) {
        assert_eq!(shape(a), shape(b), "{a:?} is not printed like {b:?}");
    }
}

#[test]
fn shapes_keep_the_printed_precision() {
    assert_eq!(shape("RI\t0.1017096683"), "RI\t#.dddddddddd");
    assert_eq!(shape("RI\tinf"), "RI\tinf");
    assert_eq!(shape("IT\t7002"), "IT\t#");
    assert_eq!(shape("RS\t12\t1.25\t140.538959"), "RS\t#\t#.dd\t#.dddddd");
    assert_eq!(shape("PA\t4+5*3+4 0.050078200"), "PA\t#+#*#+# #.ddddddddd");
    assert_ne!(shape("RI\t0.101710"), shape("RI\t0.1017096683"));
}

/// Checks rounds 1.. against C's within the tolerances of a fit whose
/// M-step differs from C's.
fn check_rounds(ours: &str, c: &str) {
    for i in 1..=N_ITER {
        let (a, b) = (
            parse_round(&round_lines(ours, i)),
            parse_round(&round_lines(c, i)),
        );
        if i == 1 {
            // The same parameters enter the first E-step.
            assert!(
                (a.lk - b.lk).abs() < 1e-5,
                "round 1 LK {} vs {}",
                a.lk,
                b.lk
            );
        } else {
            assert!(rel(a.lk, b.lk) < 1e-5, "round {i} LK {} vs {}", a.lk, b.lk);
        }
        assert!(a.ri.is_finite() && a.ri > 0.0, "round {i} RI {}", a.ri);
        for (what, x, y, tol) in [
            ("RI", a.ri, b.ri, 0.1),
            ("theta", a.tr.0, b.tr.0, 2e-3),
            ("rho", a.tr.1, b.tr.1, 2e-3),
            ("t_max", a.mt, b.mt, 2e-3),
        ] {
            assert!(rel(x, y) < tol, "round {i} {what} {x} vs {y}");
        }
        assert_eq!(a.rs.len(), b.rs.len());
        for (k, (x, y)) in a.rs.iter().zip(&b.rs).enumerate() {
            if y[0] > 0.0 {
                assert!(
                    rel(x[0], y[0]) < 2e-3,
                    "round {i} t_{k} {} vs {}",
                    x[0],
                    y[0]
                );
            }
            assert!(
                rel(x[1], y[1]) < 2e-3,
                "round {i} lambda_{k} {} vs {}",
                x[1],
                y[1]
            );
        }
    }
}

/// Our output for `pattern` against C's `c_file`, whose rounds then agree
/// within the tolerances of [`check_rounds`].
fn check_against_c(pattern: Option<&str>, c_file: &str) {
    let ours = psmc_text(fit(pattern));
    let c = read(fixture(c_file));
    check_format(&ours, &c);
    check_rounds(&ours, &c);
}

#[test]
fn matches_c_psmc_output_pattern_4_5x3_4() {
    // Measured (relative): LK 6.8e-8, lambda 2.5e-4, theta, rho and t_max
    // 2.4e-4, RI 1.1e-4.
    check_against_c(Some("4+5*3+4"), "c_reference/c_4+5x3+4_N5.psmc");
}

#[test]
fn matches_c_psmc_output_pattern_4_25x2_4_6() {
    // Measured (relative): LK 7.3e-8, lambda 3.3e-4, theta, rho and t_max
    // 2.3e-4, RI 2.6e-5.
    check_against_c(Some("4+25*2+4+6"), "c_reference/c_4+25x2+4+6_N5.psmc");
}

#[test]
fn matches_c_psmc_output_without_pattern() {
    // C's "23*1": every state has its own lambda. (With M-steps capped at
    // 100 L-BFGS iterations, as psmc-rs's default was, the fits drifted
    // apart in every state: by round 5 lambda_0 was 4.33 vs C's 6.52.)
    // Measured (relative): LK 7.1e-8, lambda 4.1e-4, theta, rho and t_max
    // 2.3e-4, RI 2.5e-4.
    check_against_c(None, "psmc_out/c_23x1_N5.psmc");
}

/// Many short sequences (`tests/data/psmc_out/fragmented.psmcfa`, 1,010 of
/// 1 to 80 bins, some of one bin, some with every bin missing): the header
/// counts them as C does and round 0 is C's. Round 1's LK is that of the
/// same parameters; the effect of C's skipped last bins on the other E-step
/// values is checked in `tests/em_c_reference.rs`.
#[test]
fn matches_c_psmc_output_on_fragments() {
    let ours = psmc_text(&checked_fit(FRAGMENTS, Some("4+5*3+4"), 1));
    let c = read(fixture("psmc_out/c_4+5x3+4_fragmented_N1.psmc"));
    check_format(&ours, &c);
    assert!(ours.contains("\nMM\tn_seqs:1010, sum_L:35465, sum_n:1869\n"));
    let (a, b) = (
        parse_round(&round_lines(&ours, 1)),
        parse_round(&round_lines(&c, 1)),
    );
    assert!((a.lk - b.lk).abs() < 1e-5, "LK {} vs {}", a.lk, b.lk);
}

#[test]
fn patterns_are_written_in_c_syntax() {
    let with = |pattern: &str| model(pattern, 0.05, 0.0125);
    assert_eq!(c_pattern(&with("4+5*3+4")), "4+5*3+4");
    assert_eq!(c_pattern(&with("4+25*2+4+6")), "4+25*2+4+6");
    assert_eq!(c_pattern(&with("1*4+ 5*3 +4")), "4+5*3+4");
    for n_steps in [1, 22, 63] {
        let model = PsmcModel::new(15.0, n_steps, 0.05, 0.0125, 2.5e-8, None).unwrap();
        let pattern = c_pattern(&model);
        assert_eq!(pattern, format!("{}*1", n_steps + 1));
        // The explicit pattern describes the same model.
        assert_eq!(c_pattern(&with(&pattern)), pattern);
        assert_eq!(with(&pattern).lam.len(), model.lam.len());
    }
}

#[test]
fn rounds_report_the_em_history() {
    let fit = fit(Some("4+5*3+4"));
    let (template, h, obs) = (&fit.template, &fit.history, &fit.obs);
    let info = PsmcRunInfo::from_history(template, h, &obs.rows, &obs.row_starts).unwrap();
    assert_eq!(
        (info.n_seqs, info.sum_l, info.sum_n, info.n_iterations),
        (3, 150_947, 7373, N_ITER)
    );
    assert_eq!((info.max_t, info.tr_ratio), (15.0, 4.0));
    let rounds = psmc_rounds(template, h, info.sum_l).unwrap();
    assert_eq!(rounds.len(), N_ITER + 1);
    for (i, r) in rounds.iter().enumerate() {
        assert_eq!(r.round, i);
        let p = &h.params[i];
        assert_eq!((r.theta, r.rho, r.t_max), (p[0], p[1], p[2]));
        assert_eq!(r.free_lambdas, p[3..]);
        let model = model_at(template, p);
        assert_eq!(r.sigma, model.sigma);
        assert_eq!(r.lambda, model.map_lam(&model.lam).unwrap());
        assert_eq!(r.t, model.compute_t()[..r.sigma.len()]);
        assert_eq!(
            (r.c_pi, r.n_recomb),
            (model.c_pi(), model.n_recomb(150_947))
        );
        if i == 0 {
            assert_eq!((r.loglike, r.mstep_evaluations), (0.0, None));
            assert!(r.post_sigma.iter().all(|&x| x == 0.0));
            assert_eq!(r.relative_information(), f64::INFINITY);
        } else {
            assert_eq!(r.loglike, h.loglike[i - 1].0);
            assert_eq!(r.post_sigma, h.post_sigma[i - 1]);
            assert_eq!(r.mstep_evaluations, Some(h.mstep_evaluations[i - 1]));
            let ri: f64 = (0..r.sigma.len())
                .map(|k| r.sigma[k] * (r.sigma[k] / r.post_sigma[k]).ln())
                .sum();
            assert_eq!(r.relative_information(), ri);
        }
    }

    // write_psmc_from_history is write_psmc of these.
    let mut out = Vec::new();
    write_psmc(&mut out, &info, &rounds).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text, psmc_text(fit));

    // The printed IT and RI lines are the rounds' values.
    let lines: Vec<&str> = text.lines().collect();
    for r in &rounds[1..] {
        let rd = lines
            .iter()
            .position(|&l| l == format!("RD\t{}", r.round))
            .unwrap();
        assert_eq!(
            lines[rd - 1],
            format!("IT\t{}", h.mstep_evaluations[r.round - 1])
        );
        let ri = round_lines(&text, r.round)
            .into_iter()
            .find_map(|l| l.strip_prefix("RI\t"))
            .unwrap();
        assert_eq!(ri, format!("{:.10}", r.relative_information()));
    }

    // The template may also be the fitted model: only its layout is used.
    let mut fitted = template.clone();
    fitted.theta *= 3.0;
    fitted.lam[2] = 7.0;
    assert_eq!(psmc_rounds(&fitted, h, info.sum_l).unwrap(), rounds);
}

#[test]
fn chunked_rows_count_as_one_sequence() {
    let obs = read_psmcfa(&fixture(REFERENCE_PSMCFA), Some(7000)).unwrap();
    assert!(obs.rows.len() > 3);
    let model = model("4+5*3+4", 0.05, 0.0125);
    let history = em_train(&mut model.clone(), &obs.rows, &obs.row_starts, 0, &mstep()).unwrap();
    let info = PsmcRunInfo::from_history(&model, &history, &obs.rows, &obs.row_starts).unwrap();
    // C's MM line for this file: n_seqs:3, sum_L:150947, sum_n:7373.
    assert_eq!((info.n_seqs, info.sum_l, info.sum_n), (3, 150_947, 7373));
    assert_eq!(info.n_iterations, 0);
    assert_eq!(psmc_rounds(&model, &history, info.sum_l).unwrap().len(), 1);

    // The same header from the counts alone.
    let counted = PsmcRunInfo::from_counts(&model, &history, 3, 150_947, 7373).unwrap();
    assert_eq!(counted, info);
}

#[test]
fn inconsistent_history_is_rejected() {
    let CFit {
        obs,
        template,
        history,
    } = fit(Some("4+5*3+4"));
    let mut h = EmHistory {
        loglike: history.loglike.clone(),
        params: history.params.clone(),
        post_sigma: history.post_sigma.clone(),
        mstep_evaluations: history.mstep_evaluations.clone(),
        mstep_iterations: history.mstep_iterations.clone(),
        mstep_stops: history.mstep_stops.clone(),
    };
    h.mstep_evaluations.pop();
    assert!(psmc_rounds(template, &h, 150_947).is_err());
    h.mstep_evaluations.push(0);
    h.post_sigma[1].pop();
    assert!(psmc_rounds(template, &h, 150_947).is_err());

    let other = PsmcModel::new(15.0, 22, 0.05, 0.0125, 2.5e-8, None).unwrap();
    assert!(psmc_rounds(&other, history, 150_947).is_err());
    assert!(PsmcRunInfo::from_history(&other, history, &obs.rows, &obs.row_starts).is_err());
    assert!(PsmcRunInfo::from_history(template, history, &obs.rows, &[true]).is_err());
}

/// Rewrites the psmc-rs `.psmc` files that `tests/psmc_out_plot.rs` runs
/// through psmc_plot.pl's rules (see `tests/data/psmc_out/README.md`).
#[test]
#[ignore = "regenerates fixtures; run by tests/data/psmc_out/generate.sh"]
fn regenerate_fixtures() {
    for (pattern, name) in [
        (Some("4+5*3+4"), "rs_4+5x3+4_N5.psmc"),
        (None, "rs_23x1_N5.psmc"),
    ] {
        let text = psmc_text(fit(pattern));
        std::fs::write(fixture(&format!("psmc_out/{name}")), text).unwrap();
    }
}
