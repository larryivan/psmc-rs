//! `.psmc` output against C psmc's (lh3/psmc 0.6.5-r74) on the reference
//! data of `tests/c_equivalence.rs`: 5 EM rounds from C's initialisation.
//!
//! The header and round 0 depend only on the data and the starting
//! parameters, so they must equal C's as text (apart from the `Version`
//! line). Later rounds depend on the M-step (L-BFGS here, Hooke-Jeeves in C)
//! and agree within tolerances in line with `tests/c_equivalence.rs`; without
//! a pattern only once the M-steps run to convergence, which the default
//! cap of 100 iterations does not allow.
//!
//! The C files are in `tests/data/c_reference` and, for the pattern-free
//! model, `tests/data/psmc_out` (see the READMEs there).

use psmc_rs::em::EmHistory;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::model::n_steps_for_pattern;
use psmc_rs::opt::{MStepConfig, em_train};
use psmc_rs::psmc_out::{PsmcRunInfo, c_pattern, psmc_rounds, write_psmc, write_psmc_from_history};
use psmc_rs::{PsmcModel, initial_theta};
use std::path::Path;

const C_REF: &str = "tests/data/c_reference";
const FIXTURES: &str = "tests/data/psmc_out";
const N_ITER: usize = 5;

struct Fit {
    template: PsmcModel,
    history: EmHistory,
    rows: Vec<Vec<u8>>,
    row_starts: Vec<bool>,
}

/// C's run: θ0 = -ln(1 - K/L), ρ0 = θ0/4, t_max = 15, λ = 1, `N_ITER` rounds,
/// with the default M-step (at most 100 L-BFGS iterations, as in the CLI).
/// Without a pattern the model has 23 states, each with its own λ (C's
/// `-p "23*1"`).
fn fit(pattern: Option<&str>) -> Fit {
    fit_with(pattern, MStepConfig::default().max_iters)
}

/// An M-step iteration cap these runs stay below: their M-steps run until
/// L-BFGS converges, as C's Hooke-Jeeves search runs until it converges.
const CONVERGED_ITERS: usize = 1000;

/// [`fit`] with at most `max_iters` L-BFGS iterations per M-step.
fn fit_with(pattern: Option<&str>, max_iters: usize) -> Fit {
    let obs = read_psmcfa(&Path::new(C_REF).join("papuan_chr1_3seg.psmcfa"), None).unwrap();
    let theta0 = initial_theta(&obs.rows).unwrap();
    let n_steps = pattern.map_or(22, |p| n_steps_for_pattern(p).unwrap());
    let template = PsmcModel::new(
        15.0,
        n_steps,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        pattern.map(str::to_string),
    )
    .unwrap();
    let mut model = template.clone();
    let config = MStepConfig {
        progress: false,
        lambda: 0.0,
        max_iters,
        ..MStepConfig::default()
    };
    let history = em_train(&mut model, &obs.rows, &obs.row_starts, N_ITER, &config).unwrap();
    Fit {
        template,
        history,
        rows: obs.rows,
        row_starts: obs.row_starts,
    }
}

fn psmc_text(fit: &Fit) -> String {
    let mut out = Vec::new();
    write_psmc_from_history(
        &mut out,
        &fit.template,
        &fit.history,
        &fit.rows,
        &fit.row_starts,
    )
    .unwrap();
    String::from_utf8(out).unwrap()
}

fn tag(line: &str) -> &str {
    line.split('\t').next().unwrap()
}

/// The lines of round `round`, from `RD` to `//`.
fn round_lines(text: &str, round: usize) -> Vec<&str> {
    let start = format!("RD\t{round}");
    let mut lines: Vec<&str> = text
        .lines()
        .skip_while(|l| *l != start)
        .take_while(|l| *l != "//")
        .collect();
    assert!(!lines.is_empty(), "no round {round}");
    lines.push("//");
    lines
}

/// The numbers of one round.
#[derive(Debug)]
struct Round {
    lk: f64,
    qd: (f64, f64),
    ri: f64,
    tr: (f64, f64),
    mt: f64,
    /// RS rows: t_k, lambda_k, n_recomb sigma_k, sigma_k, post_sigma_k.
    rs: Vec<[f64; 5]>,
}

fn parse_round(lines: &[&str]) -> Round {
    let field = |t: &str| {
        lines
            .iter()
            .find_map(|l| l.strip_prefix(&format!("{t}\t")))
            .unwrap_or_else(|| panic!("no {t} line"))
    };
    let num = |s: &str| s.parse::<f64>().unwrap();
    let (q0, q1) = field("QD").split_once(" -> ").unwrap();
    let (theta, rho) = field("TR").split_once('\t').unwrap();
    let rs = lines
        .iter()
        .filter_map(|l| l.strip_prefix("RS\t"))
        .enumerate()
        .map(|(k, l)| {
            let cols: Vec<&str> = l.split('\t').collect();
            assert_eq!(cols[0], k.to_string());
            [1, 2, 3, 4, 5].map(|i| num(cols[i]))
        })
        .collect();
    Round {
        lk: num(field("LK")),
        qd: (num(q0), num(q1)),
        ri: num(field("RI")),
        tr: (num(theta), num(rho)),
        mt: num(field("MT")),
        rs,
    }
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

/// Checks the format: line tags in C's order, the header (but `Version`)
/// and round 0 identical to C's, and every line of the later rounds printed
/// with C's conversions (see [`shape`]).
fn check_format(ours: &str, c: &str) {
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
    assert_eq!(
        shape("QD\t-93.804750 -> -33.557910"),
        "QD\t-#.dddddd -> -#.dddddd"
    );
    assert_eq!(shape("RS\t12\t1.25\t140.538959"), "RS\t#\t#.dd\t#.dddddd");
    assert_eq!(shape("PA\t4+5*3+4 0.050078200"), "PA\t#+#*#+# #.ddddddddd");
    assert_ne!(shape("RI\t0.101710"), shape("RI\t0.1017096683"));
}

/// Checks rounds 1.. against C's within the tolerances of a fit whose
/// M-step differs from C's.
fn check_rounds(ours: &str, c: &str, max_lambda_rel: f64) {
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
        assert!(
            a.qd.0 <= 0.0 && a.qd.1 <= 0.0 && a.qd.1 > a.qd.0,
            "{:?}",
            a.qd
        );
        assert!(
            rel(a.qd.0, b.qd.0) < 5e-3,
            "round {i} QD {:?} vs {:?}",
            a.qd,
            b.qd
        );
        assert!(
            rel(a.qd.1, b.qd.1) < 5e-3,
            "round {i} QD {:?} vs {:?}",
            a.qd,
            b.qd
        );
        assert!(a.ri.is_finite() && a.ri > 0.0, "round {i} RI {}", a.ri);
        assert!(rel(a.ri, b.ri) < 0.1, "round {i} RI {} vs {}", a.ri, b.ri);
        assert!(
            rel(a.tr.0, b.tr.0) < 2e-3,
            "round {i} theta {:?} vs {:?}",
            a.tr,
            b.tr
        );
        assert!(
            rel(a.tr.1, b.tr.1) < 2e-3,
            "round {i} rho {:?} vs {:?}",
            a.tr,
            b.tr
        );
        assert!(
            rel(a.mt, b.mt) < 2e-3,
            "round {i} t_max {} vs {}",
            a.mt,
            b.mt
        );
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
                rel(x[1], y[1]) < max_lambda_rel,
                "round {i} lambda_{k} {} vs {}",
                x[1],
                y[1]
            );
        }
    }
}

fn read(dir: &str, file: &str) -> String {
    std::fs::read_to_string(Path::new(dir).join(file)).unwrap()
}

#[test]
fn matches_c_psmc_output_pattern_4_5x3_4() {
    let ours = psmc_text(&fit(Some("4+5*3+4")));
    let c = read(C_REF, "c_4+5x3+4_N5.psmc");
    check_format(&ours, &c);
    // Measured (relative): LK 6.8e-8, lambda 2.5e-4, theta, rho and t_max
    // 2.4e-4, RI 1.1e-4, QD 6.2e-5.
    check_rounds(&ours, &c, 2e-3);
}

#[test]
fn matches_c_psmc_output_pattern_4_25x2_4_6() {
    let ours = psmc_text(&fit(Some("4+25*2+4+6")));
    let c = read(C_REF, "c_4+25x2+4+6_N5.psmc");
    check_format(&ours, &c);
    // Measured (relative): LK 6.1e-7, lambda 7.7e-3, theta, rho and t_max
    // 8.8e-4, RI 7.3e-3, QD 7.3e-4. These M-steps stop at the 100-iteration
    // cap; run to convergence they are closer to C's (see
    // matches_c_psmc_output_with_converged_m_steps).
    check_rounds(&ours, &c, 2e-2);
}

#[test]
fn matches_c_psmc_output_without_pattern() {
    // C's "23*1": every state has its own lambda.
    let fit = fit(None);
    let ours = psmc_text(&fit);
    let c = read(FIXTURES, "c_23x1_N5.psmc");
    check_format(&ours, &c);
    // With 23 free lambdas, 100 L-BFGS iterations do not reach the maximum
    // of Q that C's Hooke-Jeeves search finds: from the same round-1
    // E-step, Q after the M-step is -33.580 here, -33.559 with a converged
    // M-step and -33.558 in C. The fits then drift apart in every state,
    // not only in the last ones: by round 5 lambda_0 is 4.33 vs C's 6.52
    // and lambda_22 1.26 vs 6.30, RI differs from C's by up to 46% (round
    // 4), and from round 3 on C's LK is higher (-25385.386 vs -25385.653 in
    // round 5). So only the log-likelihoods are compared here; with
    // converged M-steps the output agrees with C's as closely as with a
    // pattern (matches_c_psmc_output_with_converged_m_steps).
    for i in 1..=N_ITER {
        let (a, b) = (
            parse_round(&round_lines(&ours, i)),
            parse_round(&round_lines(&c, i)),
        );
        assert!(rel(a.lk, b.lk) < 1e-4, "round {i} LK {} vs {}", a.lk, b.lk);
    }
}

/// The differences from C's later rounds beyond those of the pattern
/// 4+5*3+4 come from M-steps stopped at the default cap of 100 L-BFGS
/// iterations. With M-steps that run until L-BFGS converges, the runs
/// without a pattern and with 4+25*2+4+6 agree with C's within the
/// tolerances of 4+5*3+4.
#[test]
fn matches_c_psmc_output_with_converged_m_steps() {
    // Measured (relative), "23*1" / 4+25*2+4+6: LK 7.0e-8 / 7.3e-8, lambda
    // 5.7e-4 / 3.3e-4, theta, rho and t_max 2.3e-4 / 2.3e-4, RI 1.8e-4 /
    // 2.6e-5, QD 1.5e-4 / 1.2e-4.
    for (pattern, dir, c_file) in [
        (None, FIXTURES, "c_23x1_N5.psmc"),
        (Some("4+25*2+4+6"), C_REF, "c_4+25x2+4+6_N5.psmc"),
    ] {
        let fit = fit_with(pattern, CONVERGED_ITERS);
        // Each L-BFGS iteration evaluates the objective at least once, so
        // no M-step reached the cap.
        let evaluations = &fit.history.mstep_evaluations;
        assert!(
            evaluations.iter().all(|&n| n < CONVERGED_ITERS),
            "{pattern:?}: {evaluations:?}"
        );
        let ours = psmc_text(&fit);
        let c = read(dir, c_file);
        check_format(&ours, &c);
        check_rounds(&ours, &c, 2e-3);
    }
}

#[test]
fn patterns_are_written_in_c_syntax() {
    let with = |pattern: &str| {
        let n_steps = n_steps_for_pattern(pattern).unwrap();
        PsmcModel::new(15.0, n_steps, 0.05, 0.0125, 2.5e-8, Some(pattern.into())).unwrap()
    };
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
    let info =
        PsmcRunInfo::from_history(&fit.template, &fit.history, &fit.rows, &fit.row_starts).unwrap();
    assert_eq!(
        (info.n_seqs, info.sum_l, info.sum_n, info.n_iterations),
        (3, 150_947, 7373, N_ITER)
    );
    assert_eq!((info.max_t, info.tr_ratio), (15.0, 4.0));
    let rounds = psmc_rounds(&fit.template, &fit.history, info.sum_l).unwrap();
    assert_eq!(rounds.len(), N_ITER + 1);
    let h = &fit.history;
    for (i, r) in rounds.iter().enumerate() {
        assert_eq!(r.round, i);
        let p = &h.params[i];
        assert_eq!((r.theta, r.rho, r.t_max), (p[0], p[1], p[2]));
        assert_eq!(r.free_lambdas, p[3..]);
        let mut model = fit.template.clone();
        model.theta = p[0];
        model.rho = p[1];
        model.t_max = p[2];
        model.lam.copy_from_slice(&p[3..]);
        model.param_recalculate().unwrap();
        assert_eq!(r.sigma, model.sigma);
        assert_eq!(r.lambda, model.map_lam(&model.lam).unwrap());
        assert_eq!(r.t, model.compute_t(0.1)[..r.sigma.len()]);
        assert_eq!(
            (r.c_pi, r.n_recomb),
            (model.c_pi(), model.n_recomb(150_947))
        );
        if i == 0 {
            assert_eq!(
                (r.loglike, r.q, r.mstep_evaluations),
                (0.0, (0.0, 0.0), None)
            );
            assert!(r.post_sigma.iter().all(|&x| x == 0.0));
            assert_eq!(r.relative_information(), f64::INFINITY);
        } else {
            assert_eq!(r.loglike, h.loglike[i - 1].0);
            assert_eq!(r.q, h.q[i - 1]);
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
    assert_eq!(text, psmc_text(&fit));

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
    let mut fitted = fit.template.clone();
    fitted.theta *= 3.0;
    fitted.lam[2] = 7.0;
    assert_eq!(psmc_rounds(&fitted, h, info.sum_l).unwrap(), rounds);
}

#[test]
fn chunked_rows_count_as_one_sequence() {
    let obs = read_psmcfa(
        &Path::new(C_REF).join("papuan_chr1_3seg.psmcfa"),
        Some(7000),
    )
    .unwrap();
    assert!(obs.rows.len() > 3);
    let model = PsmcModel::new(15.0, 22, 0.05, 0.0125, 2.5e-8, Some("4+5*3+4".into())).unwrap();
    let mut m = model.clone();
    let config = MStepConfig {
        progress: false,
        ..MStepConfig::default()
    };
    let history = em_train(&mut m, &obs.rows, &obs.row_starts, 0, &config).unwrap();
    let info = PsmcRunInfo::from_history(&model, &history, &obs.rows, &obs.row_starts).unwrap();
    // C's MM line for this file: n_seqs:3, sum_L:150947, sum_n:7373.
    assert_eq!((info.n_seqs, info.sum_l, info.sum_n), (3, 150_947, 7373));
    assert_eq!(info.n_iterations, 0);
    assert_eq!(psmc_rounds(&model, &history, info.sum_l).unwrap().len(), 1);
}

#[test]
fn inconsistent_history_is_rejected() {
    let fit = fit(Some("4+5*3+4"));
    let mut h = EmHistory {
        loglike: fit.history.loglike.clone(),
        params: fit.history.params.clone(),
        post_sigma: fit.history.post_sigma.clone(),
        q: fit.history.q.clone(),
        mstep_evaluations: fit.history.mstep_evaluations.clone(),
    };
    h.q.pop();
    assert!(psmc_rounds(&fit.template, &h, 150_947).is_err());
    h.q.push((0.0, 0.0));
    h.post_sigma[1].pop();
    assert!(psmc_rounds(&fit.template, &h, 150_947).is_err());

    let other = PsmcModel::new(15.0, 22, 0.05, 0.0125, 2.5e-8, None).unwrap();
    assert!(psmc_rounds(&other, &fit.history, 150_947).is_err());
    assert!(PsmcRunInfo::from_history(&other, &fit.history, &fit.rows, &fit.row_starts).is_err());
    assert!(PsmcRunInfo::from_history(&fit.template, &fit.history, &fit.rows, &[true]).is_err());
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
        let text = psmc_text(&fit(pattern));
        std::fs::write(Path::new(FIXTURES).join(name), text).unwrap();
    }
}
