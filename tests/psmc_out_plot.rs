//! psmc_plot.pl compatibility of the `.psmc` output.
//!
//! [`plot_data`] re-implements how psmc_plot.pl (lh3/psmc, version 0.2.0)
//! reads `.psmc` files and computes the curves it plots, including the round
//! it picks (the smallest finite `RI` from round `-m` on, or round `-n`), its
//! `max_intv` rule (the states of the last pattern group are dropped unless
//! `-L`), and the scaling to years and N_e. It is checked against the data
//! files the real psmc_plot.pl wrote for C's and our output
//! (`tests/data/psmc_out/plot`, see the README there), and then used to
//! check that our output plots as `PsmcModel::ne_curve` of the picked round.

mod common;

use common::{CFit, REFERENCE_PSMCFA, c_fit, fixture, model_at, read};
use psmc_rs::PsmcModel;
use psmc_rs::psmc_out::{PsmcRunInfo, psmc_rounds, write_psmc_from_history};
use std::sync::OnceLock;

/// The psmc_plot.pl options that change its data files, except `-M`
/// (several titled runs) and `-S` (no scaling), which are not ported.
#[derive(Debug, Clone)]
struct PlotOptions {
    /// `-u`: mutation rate per base pair per generation.
    u: f64,
    /// `-s`: bin size (C's "skip").
    s: f64,
    /// `-g`: years per generation.
    g: f64,
    /// `-m`: first round eligible for the smallest RI.
    m: f64,
    /// `-n`: take this round (0: smallest RI).
    n: f64,
    /// `-N`: false negative rate (N0 is divided by 1 - N).
    fnr: f64,
    /// `-L`: keep the states of the last pattern group.
    last: bool,
}

impl PlotOptions {
    /// psmc_plot.pl's defaults overridden by `args` such as `"-n 0 -m 1 -L"`.
    fn parse(args: &str) -> Self {
        let mut o = Self {
            u: 2.5e-8,
            s: 100.0,
            g: 25.0,
            m: 5.0,
            n: 20.0,
            fnr: 0.0,
            last: false,
        };
        let mut words = args.split_whitespace();
        while let Some(flag) = words.next() {
            if flag == "-L" {
                o.last = true;
                continue;
            }
            let value: f64 = words.next().unwrap().parse().unwrap();
            match flag {
                "-u" => o.u = value,
                "-s" => o.s = value,
                "-g" => o.g = value,
                "-m" => o.m = value,
                "-n" => o.n = value,
                "-N" => o.fnr = value,
                _ => panic!("unsupported option {flag}"),
            }
        }
        o
    }
}

/// One line of a psmc_plot.pl data file: time in years, N_e in units of
/// 10^4, and the 4th to 6th `RS` columns as written in the `.psmc` file.
#[derive(Debug, Clone, PartialEq)]
struct PlotRow {
    x: f64,
    y: f64,
    rest: [String; 3],
}

/// One data set: psmc_plot.pl starts one at each `MM ... is_decoding:` line
/// (one per run in a concatenation of `.psmc` files).
#[derive(Debug, Default)]
struct DataSet {
    /// The round whose `TR` line was stored last (what is plotted).
    round: Option<String>,
    /// The rows by state.
    rows: Vec<Option<PlotRow>>,
    /// The `PA` line after its tag (psmc_plot.pl's `.par` file).
    par: Option<String>,
}

/// The fields of `line` matched by Perl's `/^TAG(\s(\S+)){n}/`: `tag`, then
/// `n` runs of non-whitespace, each after exactly one whitespace character.
fn fields<'a>(line: &'a str, tag: &str, n: usize) -> Option<Vec<&'a str>> {
    let mut rest = line.strip_prefix(tag)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rest.chars().next().filter(|c| c.is_whitespace())?;
        rest = &rest[c.len_utf8()..];
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        if end == 0 {
            return None;
        }
        out.push(&rest[..end]);
        rest = &rest[end..];
    }
    Some(out)
}

/// Perl's `/^MM.*key(\d+)/` (the last such match, as `.*` is greedy).
fn mm_digits_after(line: &str, key: &str) -> Option<u64> {
    let body = line.strip_prefix("MM")?;
    body.rmatch_indices(key).find_map(|(i, _)| {
        let rest = &body[i + key.len()..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    })
}

/// Perl's `/^MM.*pattern:(\S+),/`.
fn mm_pattern(line: &str) -> Option<&str> {
    let body = line.strip_prefix("MM")?;
    body.rmatch_indices("pattern:").find_map(|(i, _)| {
        let rest = &body[i + "pattern:".len()..];
        let run = &rest[..rest.find(char::is_whitespace).unwrap_or(rest.len())];
        match run.rfind(',') {
            Some(comma) if comma > 0 => Some(&run[..comma]),
            _ => None,
        }
    })
}

/// psmc_plot.pl's `parse_pattern`: the pattern group of each state.
fn group_of_state(pattern: &str) -> Vec<usize> {
    let mut sizes = Vec::new();
    for part in pattern.split('+') {
        let (times, size) = match part.split_once('*') {
            Some((a, b)) => (a.parse().unwrap(), b.parse::<usize>().unwrap()),
            None => (1, part.parse().unwrap()),
        };
        sizes.extend(std::iter::repeat_n(size, times));
    }
    sizes
        .iter()
        .enumerate()
        .flat_map(|(group, &size)| std::iter::repeat_n(group, size))
        .collect()
}

/// The data sets psmc_plot.pl builds from `text`, following its main loop
/// line by line (without `-M`).
fn plot_data(text: &str, o: &PlotOptions) -> Vec<DataSet> {
    let mut data: Vec<DataSet> = Vec::new();
    let mut skip: Option<f64> = None;
    let mut max_intv: i64 = 10000;
    let mut min_ri = 0.0;
    let mut do_store = false;
    let mut round = String::new();
    let mut n0 = 0.0;
    let dt = 0.0; // only changed by DT lines, which need the divergence model
    for line in text.lines() {
        if let Some(s) = mm_digits_after(line, "skip:") {
            skip = Some(s as f64 * o.s);
        } else if let Some(pattern) = mm_pattern(line) {
            let groups = group_of_state(pattern);
            let last = *groups.last().unwrap();
            // The last state outside the last group (-1 if there is none).
            max_intv = groups
                .iter()
                .rposition(|&g| g != last)
                .map_or(-1, |k| k as i64);
        } else if line.starts_with("MM") && line.contains("is_decoding:") {
            data.push(DataSet::default());
            min_ri = 1e30;
        } else if let Some(f) = fields(line, "RD", 1) {
            round = f[0].to_string();
        } else if let Some(ri) = fields(line, "RI", 1)
            .map(|f| f[0])
            .filter(|v| !v.contains("nan") && !v.contains("inf"))
        {
            let ri: f64 = ri.parse().unwrap();
            let round_num: f64 = round.parse().unwrap_or(0.0); // Perl numifies "" to 0
            do_store = false;
            if round_num >= o.m && ri < min_ri {
                min_ri = ri;
                do_store = true;
            }
            if o.n > 0.0 {
                do_store = round_num == o.n;
            }
        } else if let Some(f) = fields(line, "TR", 2).filter(|_| do_store) {
            let theta: f64 = f[0].parse().unwrap();
            let n_scale = 1.0 / (1.0 - o.fnr);
            n0 = theta / skip.expect("no MM skip line") / (4.0 * o.u) * n_scale;
            data.last_mut().unwrap().round = Some(round.clone());
        } else if let Some(f) = fields(line, "RS", 6).filter(|_| do_store) {
            let Ok(k) = f[0].parse::<usize>() else {
                continue;
            };
            if !o.last && k as i64 > max_intv {
                continue;
            }
            let (t, lambda): (f64, f64) = (f[1].parse().unwrap(), f[2].parse().unwrap());
            let (nscale, tscale, xshift) = (1.0, 0.0, 0.0);
            let row = PlotRow {
                x: 2.0 * n0 * (t + dt) * (1.0 - tscale) * o.g + xshift,
                y: lambda * n0 * nscale / 10000.0,
                rest: [f[3].to_string(), f[4].to_string(), f[5].to_string()],
            };
            let rows = &mut data.last_mut().unwrap().rows;
            if rows.len() <= k {
                rows.resize(k + 1, None);
            }
            rows[k] = Some(row);
        } else if let Some(par) = line.strip_prefix("PA").filter(|_| do_store) {
            let mut chars = par.chars();
            if chars.next().is_some_and(char::is_whitespace) {
                data.last_mut().unwrap().par = Some(chars.as_str().to_string());
            }
        }
    }
    data
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs())
}

/// A file of `tests/data/psmc_out/plot`.
fn plot_file(name: &str) -> String {
    read(fixture(&format!("psmc_out/plot/{name}")))
}

/// The cases of `plot/cases.tsv`: name, options, input files.
fn cases() -> Vec<(String, String, Vec<String>)> {
    plot_file("cases.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            let inputs = cols[2].split(' ').map(str::to_string).collect();
            (cols[0].to_string(), cols[1].to_string(), inputs)
        })
        .collect()
}

#[test]
fn port_reproduces_psmc_plot_pl() {
    let cases = cases();
    assert_eq!(cases.len(), 11);
    for (name, options, inputs) in cases {
        let text: String = inputs.iter().map(|f| read(fixture(f))).collect();
        let data = plot_data(&text, &PlotOptions::parse(&options));
        assert_eq!(data.len(), inputs.len(), "{name}: data sets");
        for (i, set) in data.iter().enumerate() {
            let golden = plot_file(&format!("{name}.{i}.txt"));
            let lines: Vec<&str> = golden.lines().collect();
            assert_eq!(set.rows.len(), lines.len(), "{name}.{i}: rows");
            for (k, (row, line)) in set.rows.iter().zip(&lines).enumerate() {
                let row = row.as_ref().unwrap();
                let cols: Vec<&str> = line.split('\t').collect();
                assert_eq!(cols.len(), 5, "{name}.{i}: {line}");
                // psmc_plot.pl prints numbers with Perl's %.15g.
                let (x, y): (f64, f64) = (cols[0].parse().unwrap(), cols[1].parse().unwrap());
                assert!(
                    close(row.x, x, 1e-14),
                    "{name}.{i} row {k}: x {} vs {x}",
                    row.x
                );
                assert!(
                    close(row.y, y, 1e-14),
                    "{name}.{i} row {k}: y {} vs {y}",
                    row.y
                );
                assert_eq!(row.rest, [cols[2], cols[3], cols[4]], "{name}.{i} row {k}");
            }
        }
        let par = plot_file(&format!("{name}.par"));
        assert_eq!(
            data[0].par.as_deref(),
            Some(par.trim_end_matches('\n')),
            "{name}"
        );
    }
}

#[test]
fn psmc_plot_pl_draws_our_curves_like_cs() {
    // The data files psmc_plot.pl wrote from our and C's output for the same
    // run (5 rounds of pattern 4+5*3+4 from C's start; round 1 has the
    // smallest RI in both) give the same curve up to the M-step differences
    // of tests/psmc_out_c_reference.rs: 3.4e-5 relative at most.
    let read = |name: &str| -> Vec<(f64, f64)> {
        plot_file(name)
            .lines()
            .map(|l| {
                let mut c = l.split('\t').map(|v| v.parse::<f64>().unwrap());
                (c.next().unwrap(), c.next().unwrap())
            })
            .collect()
    };
    let (ours, c) = (read("rs_4+5x3+4.0.txt"), read("c_4+5x3+4.0.txt"));
    assert_eq!(ours.len(), 19);
    assert_eq!(ours.len(), c.len());
    for ((x1, y1), (x2, y2)) in ours.iter().zip(&c) {
        assert!(close(*x1, *x2, 1e-4), "x {x1} vs {x2}");
        assert!(close(*y1, *y2, 1e-4), "y {y1} vs {y2}");
    }
}

/// A run from C's starting point on the reference data, as psmc_plot.pl
/// reads it: the `.psmc` text, the template, the parameters of each round
/// and its RI.
struct Run {
    text: String,
    template: PsmcModel,
    params: Vec<Vec<f64>>,
    ri: Vec<f64>,
}

fn run(pattern: Option<&str>, n_iter: usize) -> Run {
    let CFit {
        obs,
        template,
        history,
    } = c_fit(REFERENCE_PSMCFA, pattern, n_iter);
    let mut out = Vec::new();
    write_psmc_from_history(&mut out, &template, &history, &obs.rows, &obs.row_starts).unwrap();
    let info = PsmcRunInfo::from_history(&template, &history, &obs.rows, &obs.row_starts).unwrap();
    let ri = psmc_rounds(&template, &history, info.sum_l)
        .unwrap()
        .iter()
        .map(|r| r.relative_information())
        .collect();
    Run {
        text: String::from_utf8(out).unwrap(),
        template,
        params: history.params,
        ri,
    }
}

/// Six rounds of pattern 4+5*3+4, run once for the tests that read them.
fn pattern_run() -> &'static Run {
    static RUN: OnceLock<Run> = OnceLock::new();
    RUN.get_or_init(|| run(Some("4+5*3+4"), 6))
}

/// Checks that psmc_plot.pl with `options` plots round `expected_round` of
/// `run` as `ne_curve` of that round's parameters, over its first
/// `n_points` states.
fn check_curve(run: &Run, options: &str, expected_round: usize, n_points: usize) {
    let o = PlotOptions::parse(options);
    let data = plot_data(&run.text, &o);
    assert_eq!(data.len(), 1);
    let round: usize = data[0].round.as_deref().unwrap().parse().unwrap();
    assert_eq!(round, expected_round, "{options}");

    let p = &run.params[round];
    let mut model = model_at(&run.template, p);
    model.mu = o.u;
    model.param_recalculate().unwrap();
    let curve = model.ne_curve(o.s, o.g).unwrap();
    let t = model.compute_t(0.1);
    let lam = model.map_lam(&model.lam).unwrap();

    assert_eq!(data[0].rows.len(), n_points, "{options}");
    for (k, row) in data[0].rows.iter().enumerate() {
        let row = row.as_ref().unwrap();
        // The file holds theta, t_k and lambda_k to 6 decimals.
        let printed = |v: f64| if v == 0.0 { 0.0 } else { 5e-7 / v };
        let x_tol = printed(p[0]) + printed(t[k]) + 1e-12;
        let y_tol = printed(p[0]) + printed(lam[k]) + 1e-12;
        let [x, y] = curve[k];
        if k == 0 {
            assert_eq!((row.x, x), (0.0, 0.0));
        } else {
            assert!(
                close(row.x, x, x_tol),
                "{options} k={k}: x {} vs {x}",
                row.x
            );
        }
        assert!(
            close(row.y * 1e4, y, y_tol),
            "{options} k={k}: y {} vs {y}",
            row.y * 1e4
        );
    }
}

/// The round psmc_plot.pl's RI rule picks: the smallest RI from round `m` on.
fn smallest_ri(ri: &[f64], m: usize) -> usize {
    (m..ri.len())
        .min_by(|&a, &b| ri[a].total_cmp(&ri[b]))
        .unwrap()
}

#[test]
fn psmc_plot_pl_draws_ne_curve_of_the_picked_round() {
    let run = pattern_run();
    assert!(run.ri[0].is_infinite());
    // 4+5*3+4 has 23 states, the last pattern group 4: 19 are plotted.
    for m in [1, 3, 5] {
        let options = format!("-n 0 -m {m}");
        check_curve(run, &options, smallest_ri(&run.ri, m), 19);
    }
    check_curve(run, "-n 2", 2, 19);
    // psmc_plot.pl's defaults take round 20, which this run lacks; -n 6.
    check_curve(run, "-n 6", 6, 19);
    // Other scalings, and the last group shown.
    let scaled = "-n 0 -m 2 -L -u 1.25e-8 -g 29 -s 50";
    check_curve(run, scaled, smallest_ri(&run.ri, 2), 23);
}

#[test]
fn psmc_plot_pl_draws_ne_curve_without_pattern() {
    let run = run(None, 3);
    // "23*1": every state is its own group, so only the last is dropped.
    check_curve(&run, "-n 0 -m 1", smallest_ri(&run.ri, 1), 22);
    check_curve(&run, "-n 3 -L", 3, 23);
}

#[test]
fn round_0_and_nan_ri_are_never_picked() {
    // RI is inf in round 0 and nan when a posterior state probability is 0;
    // psmc_plot.pl skips both, so C's spelling ("inf", "nan") matters.
    let Run { text, ri, .. } = pattern_run();
    assert_eq!(text.matches("RI\tinf").count(), 1);
    let data = plot_data(text, &PlotOptions::parse("-n 0 -m 0"));
    let picked = smallest_ri(ri, 1).to_string();
    assert_eq!(data[0].round.as_deref(), Some(picked.as_str()));
    let nan_ri = text.replace("RI\t0.", "RI\tnan");
    assert_eq!(nan_ri.matches("RI\tnan").count(), ri.len() - 1);
    let data = plot_data(&nan_ri, &PlotOptions::parse("-n 0 -m 0"));
    assert_eq!(data[0].round, None);
}
