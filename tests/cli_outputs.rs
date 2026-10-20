//! The outputs of an inference run through the binary: determinism across
//! thread counts, C psmc's .psmc output, the bootstrap, TMRCA decoding and
//! time scaling.

mod common;

use common::psmc_text::{Round, header_value, parse_rounds};
use common::{
    TempDir, args, command, gunzipped, json, outcome, outcome_of, read, run_err, run_ok,
    small_psmcfa, words,
};
use psmc_rs::PsmcModel;
use psmc_rs::io::params::load_params;
use serde_json::Value;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

/// Run inference on `input` into `dir/<name>.json` with `extra` arguments.
fn fit(dir: &TempDir, input: &Path, name: &str, n_iter: usize, extra: Vec<OsString>) -> PathBuf {
    let out = dir.path(&format!("{name}.json"));
    run_ok(args![
        input,
        out,
        n_iter.to_string(),
        "--no-progress",
        ..extra
    ]);
    out
}

/// `dir/in.psmcfa`: the first `bins` bins of each reference sequence.
fn small_input(dir: &TempDir, bins: usize) -> PathBuf {
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, bins);
    input
}

/// The data rows of a TMRCA TSV: the MAP state and the columns of numbers
/// after it.
fn tmrca_rows(text: &str) -> Vec<(usize, Vec<f64>)> {
    text.lines()
        .skip(1)
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            let state = cols[4].parse().unwrap();
            (
                state,
                cols[5..].iter().map(|c| c.parse().unwrap()).collect(),
            )
        })
        .collect()
}

/// The same run with 1 and 4 threads writes the same bytes: the JSON, the
/// .psmc output, the TMRCA posterior and every bootstrap file.
#[test]
fn outputs_do_not_depend_on_the_thread_count() {
    let dir = TempDir::new("threads");
    let input = small_input(&dir, 20_000);
    let run = |threads: &str| -> Vec<(String, Vec<u8>)> {
        let tmrca = dir.path(&format!("t{threads}.tsv"));
        let psmc = dir.path(&format!("t{threads}.psmc"));
        let boot = dir.path(&format!("boot{threads}"));
        let out = fit(
            &dir,
            &input,
            &format!("t{threads}"),
            3,
            args![
                "--threads",
                threads,
                "--tmrca-out",
                tmrca,
                "--psmc-out",
                psmc,
                ..words("--bootstrap 3 --bootstrap-block-size 5000 --bootstrap-iters 2"),
                "--bootstrap-dir",
                boot,
            ],
        );
        let mut boot_files: Vec<PathBuf> = fs::read_dir(&boot)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        boot_files.sort();
        assert_eq!(boot_files.len(), 3 * 2 + 2, "{boot_files:?}");
        [out, tmrca, psmc]
            .into_iter()
            .chain(boot_files)
            .map(|p| {
                let name = p.file_name().unwrap().to_string_lossy();
                (name.replace(threads, "N"), fs::read(&p).unwrap())
            })
            .collect()
    };
    let one = run("1");
    let four = run("4");
    for ((name, a), (_, b)) in one.iter().zip(&four) {
        assert!(a == b, "{name} differs between 1 and 4 threads");
    }
}

/// The kernels the binary picks by itself on this CPU (`PSMC_RS_SIMD` unset).
fn auto_simd() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return "avx2+fma";
    }
    if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "portable"
    }
}

/// How far apart fits with the portable and with the AVX2 and FMA kernels
/// may be, relative to the values: log-likelihoods, and everything else.
/// The kernels differ in the last digits of each E-step (`simd_paths_agree`
/// in `tests/estep_determinism.rs`), and the M-step stops where its
/// objective is flat, which fixes the parameters to far fewer digits (see
/// `tests/c_equivalence.rs`). Measured with the two roundings of the
/// kernels on this input: log-likelihoods 1.4e-11, parameters 1.9e-7, TMRCA
/// posteriors 2.4e-7; on the whole input of `tests/c_equivalence.rs`, 5
/// iterations with its pattern 4+25*2+4+6: log-likelihoods 2.1e-10,
/// parameters 1.2e-6, and 2.0e-6 in the derived values of the .psmc output.
const PATHS_LOGLIKE: f64 = 1e-9;
const PATHS_REL: f64 = 1e-5;

/// Whether `a` and `b` agree to `rel`, give or take one unit in the last
/// decimal of `printed` (the text `a` was read from, if any); equal
/// infinities agree.
fn close(a: f64, b: f64, rel: f64, printed: Option<&str>) -> bool {
    let unit = printed
        .and_then(|t| t.split_once('.'))
        .map_or(0.0, |(_, frac)| 10f64.powi(-(frac.len() as i32)));
    a == b || (a - b).abs() <= rel * a.abs().max(b.abs()) + unit
}

/// Assert that two output JSONs agree: numbers to [`PATHS_REL`] (the
/// log-likelihoods to [`PATHS_LOGLIKE`]), everything else exactly.
#[track_caller]
fn assert_close_json(a: &Value, b: &Value, at: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            assert!(x.keys().eq(y.keys()), "{at}: keys differ");
            for (k, v) in x {
                assert_close_json(v, &y[k], &format!("{at}.{k}"));
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{at}: lengths differ");
            for (i, (v, w)) in x.iter().zip(y).enumerate() {
                assert_close_json(v, w, &format!("{at}[{i}]"));
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            let rel = if at.contains("loglike") {
                PATHS_LOGLIKE
            } else {
                PATHS_REL
            };
            assert!(close(x, y, rel, None), "{at}: {x} vs {y}");
        }
        _ => assert_eq!(a, b, "{at}"),
    }
}

/// Assert that two .psmc outputs agree: the same lines with the same tags
/// and words, and numbers as in [`assert_close_json`], to the decimals
/// printed. `IT` lines are left out: the M-step's L-BFGS can take a
/// different number of iterations to stop at nearly the same parameters.
#[track_caller]
fn assert_close_psmc(a: &str, b: &str) {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    assert_eq!(a.len(), b.len(), ".psmc: line counts differ");
    fn words(l: &str) -> Vec<&str> {
        l.split(|c: char| c.is_whitespace() || c == ':' || c == ',')
            .filter(|w| !w.is_empty())
            .collect()
    }
    for (x, y) in a.iter().zip(&b) {
        let (wx, wy) = (words(x), words(y));
        assert_eq!(wx.len(), wy.len(), ".psmc: {x:?} vs {y:?}");
        let rel = match wx.first() {
            Some(&"IT") => continue,
            Some(&"LK") => PATHS_LOGLIKE,
            _ => PATHS_REL,
        };
        for (u, v) in wx.iter().zip(&wy) {
            match (u.parse::<f64>(), v.parse::<f64>()) {
                (Ok(p), Ok(q)) => assert!(close(p, q, rel, Some(u)), ".psmc: {x:?} vs {y:?}"),
                _ => assert_eq!(u, v, ".psmc: {x:?} vs {y:?}"),
            }
        }
    }
}

/// Assert that two TMRCA posteriors agree: the same bins, posterior means,
/// maximum posteriors and entropies to [`PATHS_REL`] in every bin, and the
/// MAP state and the 95% interval in all but one bin in 1,000 (they move to
/// the next state where two states nearly tie).
#[track_caller]
fn assert_close_tmrca(a: &str, b: &str) {
    let (a, b): (Vec<&str>, Vec<&str>) = (a.lines().collect(), b.lines().collect());
    assert_eq!(a.len(), b.len(), "TMRCA: line counts differ");
    assert_eq!(a[0], b[0], "TMRCA: headers differ");
    let mut moved = 0;
    for (x, y) in a[1..].iter().zip(&b[1..]) {
        let (cx, cy): (Vec<&str>, Vec<&str>) = (x.split('\t').collect(), y.split('\t').collect());
        assert_eq!(cx.len(), 11, "TMRCA: {x:?}");
        assert_eq!(cx[..4], cy[..4], "TMRCA: {x:?} vs {y:?}");
        let agree = |i: usize| {
            let (p, q): (f64, f64) = (cx[i].parse().unwrap(), cy[i].parse().unwrap());
            close(p, q, PATHS_REL, None)
        };
        for i in [6, 9, 10] {
            assert!(agree(i), "TMRCA: {x:?} vs {y:?}");
        }
        if ![4, 5, 7, 8].into_iter().all(agree) {
            moved += 1;
        }
    }
    let bins = a.len() - 1;
    eprintln!("TMRCA: MAP or interval moved in {moved} of {bins} bins");
    assert!(
        moved * 1000 <= bins,
        "TMRCA: MAP or interval moved in {moved} of {bins} bins"
    );
}

/// PSMC_RS_SIMD=portable forces the portable kernels on x86-64, whose fits
/// agree closely with those of the AVX2 and FMA kernels: the JSON, the
/// .psmc output and the TMRCA posterior. The JSON's "simd" names the
/// kernels that ran. On other CPUs both runs use the same kernels and give
/// the same bytes.
#[test]
fn simd_paths_give_close_fits() {
    let dir = TempDir::new("simd");
    let input = small_input(&dir, 20_000);
    let run = |name: &str, portable: bool| -> (Value, String, String) {
        let [out, tmrca, psmc] = ["json", "tsv", "psmc"].map(|e| dir.path(&format!("{name}.{e}")));
        let mut cmd = command();
        cmd.args(args![
            input,
            out,
            "3",
            "--no-progress",
            "--threads",
            "2",
            "--tmrca-out",
            tmrca,
            "--psmc-out",
            psmc,
        ]);
        if portable {
            cmd.env("PSMC_RS_SIMD", "portable");
        } else {
            cmd.env_remove("PSMC_RS_SIMD");
        }
        outcome_of(&mut cmd).ok();
        (json(&out), read(tmrca), read(psmc))
    };
    let (mut auto, auto_tmrca, auto_psmc) = run("auto", false);
    let (mut portable, portable_tmrca, portable_psmc) = run("portable", true);
    let forced = if cfg!(target_arch = "aarch64") {
        "neon"
    } else {
        "portable"
    };
    eprintln!("simd: {} against {forced}", auto["simd"]);
    assert_eq!(auto["simd"], auto_simd());
    assert_eq!(portable["simd"], forced);
    for v in [&mut auto, &mut portable] {
        v.as_object_mut().unwrap().remove("simd");
    }
    if auto_simd() == forced {
        // Compared as text: serde_json's numbers compare 0.0 and -0.0 equal.
        assert_eq!(auto.to_string(), portable.to_string(), "the JSON differs");
        assert!(auto_tmrca == portable_tmrca, "the TMRCA posterior differs");
        assert!(auto_psmc == portable_psmc, "the .psmc output differs");
    } else {
        assert_close_json(&auto, &portable, "JSON");
        assert_close_psmc(&auto_psmc, &portable_psmc);
        assert_close_tmrca(&auto_tmrca, &portable_tmrca);
    }
}

/// --psmc-out writes C psmc's format with rounds 0 to N_ITER; the header
/// counts the input as C does, including empty records.
#[test]
fn psmc_output_has_a_round_per_iteration() {
    let dir = TempDir::new("psmc_out");
    let input = small_input(&dir, 4000);
    // An empty record, which C psmc counts in n_seqs.
    dir.write("in.psmcfa", read(&input) + ">empty\n");
    let psmc = dir.path("fit.psmc");
    let out = fit(&dir, &input, "fit", 3, args!["--psmc-out", psmc]);
    let text = read(&psmc);
    let rounds = parse_rounds(&text);
    assert_eq!(rounds.len(), 4);
    for (key, value) in [
        ("n_iterations", "3"),
        ("n_seqs", "4"),
        ("pattern", "4+5*3+4"),
        ("theta/rho", "4"),
    ] {
        assert_eq!(header_value(&text, key), Some(value), "{key}");
    }
    assert!(text.contains(&format!(
        "MM\tVersion: psmc-rs {}",
        env!("CARGO_PKG_VERSION")
    )));
    // Round 0 holds the start, the last round the parameters written.
    let theta = json(&out)["theta"].as_f64().unwrap();
    let last = rounds.last().unwrap();
    assert!((last.tr.0 - theta).abs() < 1e-6, "{} vs {theta}", last.tr.0);
    assert_eq!(rounds[0].lk, 0.0);
    assert!(rounds[1..].iter().all(|r| r.lk < 0.0));
    assert_eq!(rounds[0].rs.len(), 23);
}

/// The .psmc file does not record the bin size, mutation rate or generation
/// time that psmc_plot.pl scales by, so the run prints the options to give
/// it.
#[test]
fn the_psmc_plot_options_are_printed() {
    let dir = TempDir::new("psmc_plot_hint");
    let input = small_input(&dir, 2000);
    let (out, psmc) = (dir.path("fit.json"), dir.path("fit.psmc"));
    let stderr = |extra: Vec<OsString>| {
        outcome(args![
            input,
            out,
            "2",
            "--no-progress",
            "--psmc-out",
            psmc,
            ..extra
        ])
        .ok()
        .stderr
    };
    let err = stderr(args![..words("--bin-size 50 --mu 1.25e-8 --gen-years 29")]);
    assert!(
        err.contains(&format!(
            "To plot it: psmc_plot.pl -s 50 -u 1.25e-8 -g 29 -n 2 PREFIX {} (-s is the bin size",
            psmc.display()
        )),
        "{err}"
    );
    let help = run_ok(["--help"]);
    assert!(
        help.contains("psmc_plot.pl -s BIN_SIZE -u MU -g GEN_YEARS -n N_ITER"),
        "{help}"
    );

    // Replicates of another iteration count need their own -n: plotted in
    // one file with the main fit at -n 2, the 3-round replicates would make
    // psmc_plot.pl fail.
    let bootstrap = |iters: &str| {
        stderr(args![
            ..words("--bootstrap 2 --bootstrap-block-size 1000"),
            "--bootstrap-iters",
            iters,
        ])
    };
    let err = bootstrap("3");
    assert!(
        err.contains(&format!(
            "warning: the bootstrap replicates ran 3 EM iterations and the main fit 2; \
             psmc_plot.pl -n takes the same round of every run it reads, so do not plot the \
             replicates in one file with the main fit's .psmc output, but on their own: \
             psmc_plot.pl -s 100 -u 2.5e-8 -g 25 -n 3 PREFIX {}",
            dir.path("fit.bootstrap/replicate_*.psmc").display()
        )),
        "{err}"
    );
    let err = bootstrap("2");
    assert!(!err.contains("bootstrap replicates ran"), "{err}");
}

/// A bootstrap writes every replicate's JSON and .psmc file and the summary,
/// and the report shows it.
#[test]
fn bootstrap_writes_replicates_and_summary() {
    let dir = TempDir::new("bootstrap");
    let input = small_input(&dir, 3000);
    let stderr = outcome(args![
        input,
        dir.path("fit.json"),
        "2",
        ..words("--no-progress --bootstrap 3 --bootstrap-block-size 1000"),
    ])
    .ok()
    .stderr;
    // 3 sequences of 3 trunks: few enough for a warning.
    assert!(
        stderr.contains("warning: the input forms only 9 bootstrap trunks of 1000 bins"),
        "{stderr}"
    );
    let boot = dir.path("fit.bootstrap");
    for rep in 1..=3 {
        let params = load_params(&boot.join(format!("replicate_{rep:03}.json"))).unwrap();
        assert_eq!(params.pattern.as_deref(), Some("4+5*3+4"));
        let text = read(boot.join(format!("replicate_{rep:03}.psmc")));
        let rounds = parse_rounds(&text);
        assert_eq!(rounds.len(), 3, "replicate {rep}");
        // Each replicate fits its own resampled trunks from a fresh start.
        assert_eq!(rounds[0].mt, 15.0);
        assert!((rounds[2].tr.0 - params.theta).abs() < 1e-6);
        let n_seqs: usize = header_value(&text, "n_seqs").unwrap().parse().unwrap();
        assert!(
            (7..=11).contains(&n_seqs),
            "{n_seqs} trunks of about 1000 bins"
        );
    }
    assert!(!boot.join("replicate_004.json").exists());
    let summary = json(&boot.join("summary.json"));
    assert_eq!(summary["n_replicates"], 3);
    assert_eq!(summary["block_size"], 1000);
    assert_eq!(summary["seed"], 42);
    let tsv = read(boot.join("summary.tsv"));
    assert!(tsv.starts_with("x_years\tne_main\tne_q025\tne_q500\tne_q975\n"));

    let html = read(dir.path("fit.html"));
    assert!(html.contains("<h2>Bootstrap</h2>"));
    assert!(html.contains("<tr><th>Trunks resampled</th><td>9</td></tr>"));
}

/// A bootstrap into the directory of an earlier one leaves only its own
/// replicates: `cat fit.psmc fit.bootstrap/replicate_*.psmc` (the README's
/// psmc_plot.pl recipe) must not mix in an earlier run's. A fit without a
/// bootstrap says that the directory next to it is not its own.
#[test]
fn a_rerun_bootstrap_leaves_no_earlier_replicates() {
    let dir = TempDir::new("bootstrap_rerun");
    let input = small_input(&dir, 3000);
    let out = dir.path("fit.json");
    let boot = dir.path("fit.bootstrap");
    let rep = |n: usize, ext: &str| boot.join(format!("replicate_{n:03}.{ext}"));
    let run = |n_iter: &str, extra: &[&str]| -> String {
        let block = ["--bootstrap-block-size", "1000"];
        outcome(args![input, out, n_iter, "--no-progress", ..block, ..extra])
            .ok()
            .stderr
    };
    let first = run("1", &["--bootstrap", "4"]);
    assert!(!first.contains("earlier"), "{first}");
    assert!(rep(4, "json").exists() && rep(4, "psmc").exists());
    // Not a replicate file of psmc-rs: left alone.
    fs::write(boot.join("replicate_notes.txt"), "mine").unwrap();

    let second = run("2", &["--bootstrap", "2", "--bootstrap-seed", "7"]);
    assert!(
        second.contains(&format!(
            "warning: {} held replicates 3 to 4 of an earlier bootstrap, beyond this run's 2; \
             their 4 files were removed",
            boot.display()
        )),
        "{second}"
    );
    for n in [1, 2] {
        let rounds = parse_rounds(&read(rep(n, "psmc")));
        assert_eq!(rounds.len(), 3, "replicate {n} is of this run");
        assert!(rep(n, "json").exists());
    }
    for n in [3, 4] {
        assert!(!rep(n, "json").exists() && !rep(n, "psmc").exists());
    }
    assert!(boot.join("replicate_notes.txt").exists());
    assert_eq!(json(&boot.join("summary.json"))["n_replicates"], 2);

    // A fit without a bootstrap leaves the directory, and says it is not
    // of this fit.
    let third = run("1", &[]);
    assert!(
        third.contains(&format!(
            "warning: {} holds the bootstrap of an earlier run, not of this fit",
            boot.display()
        )),
        "{third}"
    );
    assert!(rep(1, "json").exists());
}

/// A rerun that stops during its bootstrap leaves none of the earlier
/// bootstrap's files: every replicate file and the summary of the earlier
/// run are removed before the first replicate is written, so the
/// directory never mixes two runs.
#[test]
fn a_rerun_that_fails_in_its_bootstrap_leaves_no_earlier_files() {
    let dir = TempDir::new("bootstrap_rerun_fails");
    let input = small_input(&dir, 3000);
    let boot = dir.path("fit.bootstrap");
    let rep = |n: usize, ext: &str| boot.join(format!("replicate_{n:03}.{ext}"));
    let run = |seed: &str| {
        outcome(args![
            input,
            dir.path("fit.json"),
            "1",
            ..words("--no-progress --threads 1 --bootstrap 4 --bootstrap-block-size 1000"),
            "--bootstrap-seed",
            seed,
        ])
    };
    run("1").ok();
    let first: Vec<_> = (1..=4).map(|n| fs::read(rep(n, "json")).unwrap()).collect();
    let summaries = ["summary.tsv", "summary.json"].map(|name| boot.join(name));
    assert!(summaries.iter().all(|s| s.exists()));

    // Replicate 3's JSON cannot be written: a directory has its name.
    fs::remove_file(rep(3, "json")).unwrap();
    fs::create_dir(rep(3, "json")).unwrap();
    let stderr = run("2").exits(1);
    assert!(stderr.contains("bootstrap replicate 3"), "{stderr}");
    for n in [1, 2, 4] {
        assert!(!rep(n, "psmc").exists(), "replicate {n}");
        // Absent, or written by the second run.
        if let Ok(json) = fs::read(rep(n, "json")) {
            assert_ne!(json, first[n - 1], "replicate {n} is of the first run");
        }
    }
    for summary in summaries {
        assert!(!summary.exists(), "{}", summary.display());
    }
}

/// Replicates whose fresh start EM cannot fit are named before the main
/// fit, with the --tr-ratio that would fit them, and skipped: the run
/// finishes, and the summary, its JSON and the report say which were left
/// out. When none can be fitted, the run fails before the main fit.
#[test]
fn replicates_with_an_invalid_start_are_skipped_and_reported() {
    let dir = TempDir::new("bootstrap_invalid_starts");
    // Two sequences of ten 1000-bin trunks, alternately 38% and 58%
    // heterozygous: the main start (rho0 = theta0/4 = 0.1635) is just below
    // the limit of 1/6 on rho per bin, and the start of a replicate that
    // draws more of the 58% trunks is past it.
    let trunk = |het_per_50: usize| -> String {
        (0..1000)
            .map(|i| if i % 50 < het_per_50 { 'K' } else { 'T' })
            .collect()
    };
    let seq: String = (0..10)
        .map(|t| trunk(if t % 2 == 0 { 19 } else { 29 }))
        .collect();
    let input = dir.write("in.psmcfa", format!(">a\n{seq}\n>b\n{seq}\n"));
    let out = dir.path("fit.json");
    let boot = dir.path("fit.bootstrap");
    let argv = |n_iter: &str, extra: &[&str]| {
        args![
            input,
            out,
            n_iter,
            ..words("--no-progress --bootstrap 8 --bootstrap-iters 1 --bootstrap-block-size 1000"),
            ..extra
        ]
    };
    // With the QC JSON on stdout, every message and result goes to stderr,
    // in order.
    let stderr = outcome(argv("1", &["--qc-json", "-"])).ok().stderr;
    let warning = "warning: bootstrap replicates 2, 3 and 7 (3 of 8) would start from negative \
                   transition probabilities (rho0 up to ";
    let at = stderr.find(warning).unwrap_or_else(|| panic!("{stderr}"));
    // Before the main fit.
    let fitted = stderr.find("Last EM loglike").unwrap();
    assert!(at < fitted, "{stderr}");
    let line = stderr[at..].lines().next().unwrap();
    assert!(
        line.contains("above the limit of 0.166667 per bin)")
            && line.contains("They will be skipped and the bands will come from the other 5")
            && line.ends_with(
                "To fit every replicate, start from --tr-ratio 4.41 or more (C psmc's -r)"
            ),
        "{line}"
    );
    assert!(
        stderr[fitted..].contains("replicate_NNN.psmc; 5 of 8, replicates 2, 3 and 7 skipped)"),
        "{stderr}"
    );
    let summary = json(&boot.join("summary.json"));
    assert_eq!(summary["n_replicates"], 5);
    assert_eq!(summary["skipped_replicates"], serde_json::json!([2, 3, 7]));
    for n in 1..=8 {
        let skipped = [2, 3, 7].contains(&n);
        for ext in ["json", "psmc"] {
            let file = boot.join(format!("replicate_{n:03}.{ext}"));
            assert_eq!(file.exists(), !skipped, "{}", file.display());
        }
    }
    assert!(
        read(dir.path("fit.html"))
            .contains("<tr><th>Skipped replicates</th><td>3 of 8 (2, 3, 7)</td></tr>"),
        "the report names the skipped replicates"
    );

    // The advice fits every replicate.
    let stderr = outcome(argv("1", &["--tr-ratio", "4.41"])).ok().stderr;
    assert!(!stderr.contains("negative transition"), "{stderr}");
    let summary = json(&boot.join("summary.json"));
    assert_eq!(summary["n_replicates"], 8);
    assert_eq!(summary["skipped_replicates"], serde_json::json!([]));
    fs::remove_file(&out).unwrap();

    // No replicate can be fitted: an error before the main fit (here of no
    // iterations, which refuses no start of its own).
    let err = run_err(argv("0", &["--tr-ratio", "2"]));
    assert!(
        err.contains(
            "error: every one of the 8 bootstrap replicates would start from negative \
             transition probabilities"
        ) && err.contains("start from --tr-ratio 4.41 or more"),
        "{err}"
    );
    assert!(!err.contains("Last EM loglike"), "{err}");
    assert!(!out.exists());
}

/// A bootstrap that cannot vary fails before the main fit, as do bootstrap
/// options without a bootstrap and a bootstrap without iterations.
#[test]
fn a_single_trunk_bootstrap_fails_before_fitting() {
    let dir = TempDir::new("one_trunk");
    // One sequence of 3,000 bins.
    let input = dir.write(
        "in.psmcfa",
        format!(">chr1\n{}\n", "TTTTTTTTKTTTTTTTTTNT".repeat(150)),
    );
    let out = dir.path("fit.json");
    let fails = |n_iter: &str, extra: Vec<OsString>| run_err(args![input, out, n_iter, ..extra]);
    let err = fails(
        "30",
        args!["--bootstrap", "10", "--bootstrap-block-size", "100000"],
    );
    assert!(err.contains("--bootstrap-block-size is too large"), "{err}");
    assert!(err.contains("single trunk"), "{err}");
    assert!(!out.exists());

    let err = fails("5", args!["--bootstrap-dir", dir.path("boot")]);
    assert!(
        err.contains("--bootstrap-dir applies to a bootstrap"),
        "{err}"
    );
    let err = fails(
        "0",
        args!["--bootstrap", "2", "--bootstrap-block-size", "1000"],
    );
    assert!(err.contains("0 EM iterations"), "{err}");
}

/// TMRCA times are each state's average coalescent time (C's TC), scaled
/// by --bin-size and --gen-years; a .gz path gives gzip.
#[test]
fn tmrca_times_are_the_states_average_times() {
    let dir = TempDir::new("tmrca");
    let input = small_input(&dir, 3000);
    let plain = dir.path("t.tsv");
    let out = fit(&dir, &input, "fit", 1, args!["--tmrca-out", plain]);
    let model = PsmcModel::from_params_file(&load_params(&out).unwrap()).unwrap();
    let years = model.state_tmrca_years(100.0, 25.0);
    let text = read(&plain);
    let rows = tmrca_rows(&text);
    assert_eq!(rows.len(), 3 * 3000);
    for (state, cols) in &rows {
        // tmrca_map_years is printed with 9 significant digits.
        let map = cols[0];
        assert!(
            (map - years[*state]).abs() <= 1e-8 * years[*state],
            "{map} vs {}",
            years[*state]
        );
    }

    let gz = dir.path("t.tsv.gz");
    fit(&dir, &input, "fit_gz", 1, args!["--tmrca-out", gz]);
    assert_eq!(gunzipped(&gz), text.as_bytes());
}

/// --gen-years scales every time: doubling it doubles the TMRCA years and
/// the bootstrap summary's time axis, and leaves the fits alone.
#[test]
fn generation_time_scales_the_time_axis() {
    let dir = TempDir::new("gen_years");
    let input = small_input(&dir, 3000);
    let run = |years: &str| {
        let tmrca = dir.path(&format!("g{years}.tsv"));
        let out = fit(
            &dir,
            &input,
            &format!("g{years}"),
            1,
            args![
                "--gen-years",
                years,
                "--tmrca-out",
                tmrca,
                ..words("--bootstrap 2 --bootstrap-block-size 1000"),
            ],
        );
        let summary = json(&dir.path(&format!("g{years}.bootstrap/summary.json")));
        (json(&out), tmrca_rows(&read(tmrca)), summary)
    };
    let (fit25, rows25, boot25) = run("25");
    let (fit50, rows50, boot50) = run("50");
    let points = |v: &Value| v["points"].as_array().unwrap().clone();
    for (a, b) in points(&boot25).iter().zip(&points(&boot50)) {
        assert_eq!(
            b["x_years"].as_f64().unwrap(),
            2.0 * a["x_years"].as_f64().unwrap()
        );
        assert_eq!(b["ne_q500"], a["ne_q500"]);
    }
    assert_eq!(fit25["lam"], fit50["lam"]);
    assert_eq!(fit25["theta"], fit50["theta"]);
    for ((s25, a), (s50, b)) in rows25.iter().zip(&rows50) {
        assert_eq!(s25, s50);
        // map, mean, q2.5 and q97.5 are times; pmax and entropy are not.
        for k in 0..4 {
            assert!((b[k] / a[k] - 2.0).abs() < 1e-8, "{} vs {}", b[k], a[k]);
        }
        assert_eq!(a[4..], b[4..]);
    }
    assert_eq!(fit25["gen_years"], 25.0);
    assert_eq!(fit50["gen_years"], 50.0);
}

/// The output JSON records how the fit was made, next to the model.
#[test]
fn the_output_json_records_the_run() {
    let dir = TempDir::new("metadata");
    let input = small_input(&dir, 4000);
    let psmc = dir.path("fit.psmc");
    let qc_json = dir.path("qc.json");
    let out = fit(
        &dir,
        &input,
        "fit",
        3,
        args!["--psmc-out", psmc, "--qc-json", qc_json, "--bin-size", "50"],
    );
    let v = json(&out);
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    // The binary inherits this process's PSMC_RS_SIMD and CPU.
    assert_eq!(v["simd"], psmc_rs::hmm::simd_path().name());
    assert_eq!(v["n_iter"], 3);
    assert_eq!(v["bin_size"], 50);
    assert_eq!(v["gen_years"], 25.0);
    assert_eq!(v["tr_ratio"], 4.0);
    assert_eq!(v["smooth_lambda"], 0.0);
    let loglike: Vec<f64> = serde_json::from_value(v["loglike"].clone()).unwrap();
    assert_eq!(loglike.len(), 4);
    assert_eq!(v["final_loglike"].as_f64().unwrap(), loglike[3]);
    // The .psmc rounds give the log-likelihood of each E-step's parameters.
    let rounds = parse_rounds(&read(&psmc));
    for (round, ll) in rounds[1..].iter().zip(&loglike) {
        assert!((round.lk - ll).abs() < 1e-6, "{} vs {ll}", round.lk);
    }

    let text = read(&input);
    let bins: Vec<char> = text
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(str::chars)
        .collect();
    let count = |c: char| bins.iter().filter(|&&b| b == c).count();
    let input_summary = &v["input"];
    assert_eq!(input_summary["format"], "psmcfa");
    assert!(input_summary.get("vcf_mode").is_none());
    assert_eq!(input_summary["sequences"], 3);
    assert_eq!(input_summary["bins"], bins.len());
    assert_eq!(input_summary["het_bins"], count('K'));
    assert_eq!(input_summary["missing_bins"], count('N'));
    assert_eq!(input_summary["dropped_sequences"], 0);

    // The QC summary lists the codes of the full report.
    let qc = json(&qc_json);
    let codes = |severity: &str| -> Vec<Value> {
        qc["findings"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["severity"] == severity)
            .map(|f| f["code"].clone())
            .collect()
    };
    assert_eq!(v["qc"]["warnings"].as_array().unwrap(), &codes("warning"));
    assert_eq!(v["qc"]["notes"].as_array().unwrap(), &codes("info"));
    assert!(codes("warning").contains(&"TOO_SHORT".into()));
    assert_eq!(v["qc"]["bins"], qc["totals"]["bins"]);
    assert_eq!(qc["bin_size"], 50);

    let html = read(dir.path("fit.html"));
    assert!(html.contains("<h2>Input QC</h2>"));
    assert!(html.contains("<code>TOO_SHORT</code>"));
    assert!(html.contains("<tr><th>Starting theta0/rho0</th><td>4</td></tr>"));
}

/// "Last EM loglike" after N iterations is the log-likelihood of the
/// parameters written, which C psmc prints as the LK of round N + 1 (the
/// E-step of the next iteration); psmc-rs 0.1 printed round N's. The JSON's
/// `loglike[i]` is the value after i iterations, C's LK of round i + 1.
#[test]
fn last_em_loglike_is_c_psmcs_next_round() {
    let dir = TempDir::new("loglike_convention");
    let input = small_input(&dir, 3000);
    let fit_n = |n: usize| -> (f64, Value, Vec<Round>) {
        let psmc = dir.path(&format!("n{n}.psmc"));
        let out = dir.path(&format!("n{n}.json"));
        let stdout = run_ok(args![
            input,
            out,
            n.to_string(),
            "--no-progress",
            "--psmc-out",
            psmc,
        ]);
        let last = stdout
            .lines()
            .find_map(|l| l.strip_prefix("Last EM loglike: "))
            .unwrap();
        (last.parse().unwrap(), json(&out), parse_rounds(&read(psmc)))
    };
    let (two, two_json, _) = fit_n(2);
    let (_, _, three_rounds) = fit_n(3);
    // The .psmc prints LK with 6 decimals.
    assert!((two - three_rounds[3].lk).abs() < 1e-6, "{two} vs RD 3");
    assert!((two - three_rounds[2].lk).abs() > 1e-3, "not RD 2");
    let loglike: Vec<f64> = serde_json::from_value(two_json["loglike"].clone()).unwrap();
    assert_eq!(loglike[2], two);
    for i in 0..2 {
        assert!(
            (loglike[i] - three_rounds[i + 1].lk).abs() < 1e-6,
            "loglike[{i}]"
        );
    }
}

/// The QC report goes to stderr before the fit; --quiet leaves only the
/// results and the warnings.
#[test]
fn qc_text_is_printed_unless_quiet() {
    let dir = TempDir::new("quiet");
    let input = small_input(&dir, 2000);
    let run = |extra: &[&str]| {
        outcome(args![
            input,
            dir.path("fit.json"),
            "1",
            "--no-progress",
            ..extra
        ])
        .ok()
    };
    let loud = run(&[]);
    assert!(loud.stdout.contains("Last EM loglike: "), "{}", loud.stdout);
    for line in [
        "Input QC (psmcfa): 3 sequences",
        "WARNING  TOO_SHORT",
        "Model: pattern 4+5*3+4",
    ] {
        assert!(loud.stderr.contains(line), "{}", loud.stderr);
    }

    let quiet = run(&["--quiet"]);
    assert_eq!(quiet.stdout, loud.stdout);
    assert!(!quiet.stderr.contains("Input QC"), "{}", quiet.stderr);
    assert!(!quiet.stderr.contains("Model:"), "{}", quiet.stderr);
    assert!(
        quiet.stderr.contains("warning: QC TOO_SHORT: Only "),
        "{}",
        quiet.stderr
    );
}

/// With --qc-json -, stdout holds the QC JSON alone, so that it parses; the
/// result lines go to stderr.
#[test]
fn qc_json_on_stdout_is_the_only_stdout() {
    let dir = TempDir::new("qc_stdout");
    let input = small_input(&dir, 2000);
    for quiet in [&[][..], &["-q"]] {
        let o = outcome(args![
            input,
            dir.path("fit.json"),
            "1",
            "--qc-json",
            "-",
            ..quiet
        ])
        .ok();
        let qc: Value = serde_json::from_str(&o.stdout).unwrap();
        assert_eq!(qc["n_sequences"], 3);
        for line in ["Last EM loglike: ", "Parameters: ", "HTML report: "] {
            assert!(o.stderr.contains(line), "{quiet:?}: {}", o.stderr);
        }
        assert!(!o.stderr.contains("QC report: -"), "{}", o.stderr);
    }
}

/// Only the QC JSON goes to stdout with `-`. Any other output named `-` is
/// a usage error, not a file of that name (where two of them would
/// overwrite each other), and nothing is written.
#[test]
fn other_outputs_cannot_go_to_stdout() {
    let dir = TempDir::new("dash_outputs");
    let input = small_input(&dir, 2000);
    let input = input.to_str().unwrap();
    // Run in the test's directory, where a file `-` would land.
    let run_here = |args: &[&str]| {
        let stderr = outcome_of(command().args(args).current_dir(dir.path(""))).exits(2);
        assert!(!stderr.contains("Input:"), "{stderr}");
        assert!(!dir.path("-").exists(), "{args:?}");
        stderr
    };
    let fit = |extra: &[&str]| {
        run_here(&[&[input, "fit.json", "1", "--no-progress"][..], extra].concat())
    };
    for flag in ["--psmc-out", "--tmrca-out"] {
        let err = fit(&[flag, "-"]);
        assert!(err.contains(&format!("{flag} cannot be '-'")), "{err}");
        assert!(err.contains("only the QC JSON"), "{err}");
    }
    let err = fit(&["--psmc-out", "-", "--tmrca-out", "-", "--qc-json", "-"]);
    assert!(err.contains("--psmc-out cannot be '-'"), "{err}");
    let err = run_here(&[input, "-", "1"]);
    assert!(err.contains("OUTPUT cannot be '-'"), "{err}");
    let err = fit(&["--bootstrap", "2", "--bootstrap-dir", "-"]);
    assert!(err.contains("--bootstrap-dir cannot be '-'"), "{err}");
    assert!(!dir.path("fit.json").exists());

    for flag in ["--psmcfa", "--multihetsep", "--multihetsep-per-chrom"] {
        let err = run_here(&["convert", input, flag, "-", "--qc-json", "-"]);
        assert!(err.contains(&format!("{flag} cannot be '-'")), "{err}");
    }
}

/// A reader that quits early (`psmc-rs ... | head`) only loses stdout
/// lines: the run still writes its files and succeeds.
#[test]
fn a_closed_stdout_does_not_stop_the_run() {
    use std::process::Stdio;
    let dir = TempDir::new("closed_stdout");
    let input = small_input(&dir, 2000);
    let out = dir.path("fit.json");
    let run_closed = |args: Vec<OsString>| {
        let mut child = command()
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        // Close the reading end before the run prints anything.
        drop(child.stdout.take());
        common::Outcome::from(child.wait_with_output().unwrap()).ok();
    };
    run_closed(args![input, out, "2", "-q", "--no-progress"]);
    assert!(out.exists() && dir.path("fit.html").exists());
    run_closed(args!["qc", input]);
}

/// A VCF read without a callable mask gets one warning about it, from QC.
#[test]
fn the_missing_mask_warning_is_given_once() {
    let dir = TempDir::new("no_mask");
    let vcf = dir.write(
        "in.vcf",
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\nchr1\t24\t.\tA\tC\t.\tPASS\t.\tGT\t1/0\n",
    );
    let out = dir.path("fit.json");
    let stderr = outcome(args![
        vcf,
        out,
        "0",
        ..words("--bin-size 4 --min-good-bases 0 --no-progress"),
    ])
    .ok()
    .stderr;
    assert!(stderr.contains("VARIANTS_ONLY_NO_MASK"), "{stderr}");
    assert!(
        !stderr.contains("no callable mask (--vcf-mask) was given"),
        "{stderr}"
    );
    let v = json(&out);
    assert_eq!(v["input"]["vcf_mode"], "mask");
    assert!(
        v["qc"]["warnings"]
            .as_array()
            .unwrap()
            .contains(&"VARIANTS_ONLY_NO_MASK".into())
    );
}

/// PSMC_ECHARTS_JS embeds a local chart library in the report, for use
/// without a network; otherwise the report loads it from the CDN.
#[test]
fn the_report_can_embed_a_local_chart_library() {
    let dir = TempDir::new("echarts");
    let input = small_input(&dir, 1000);
    let js = dir.write("echarts.min.js", "/* a local copy of echarts */");
    let run = |name: &str, env: Option<&Path>| {
        let mut cmd = command();
        cmd.args(args![
            input,
            dir.path(&format!("{name}.json")),
            "0",
            "--quiet"
        ]);
        match env {
            Some(path) => cmd.env("PSMC_ECHARTS_JS", path),
            None => cmd.env_remove("PSMC_ECHARTS_JS"),
        };
        outcome_of(&mut cmd).ok();
        read(dir.path(&format!("{name}.html")))
    };
    let embedded = run("local", Some(&js));
    assert!(embedded.contains("<script>\n/* a local copy of echarts */\n</script>"));
    let cdn = r#"<script src="https://cdn.jsdelivr.net/npm/echarts@5"#;
    assert!(!embedded.contains(cdn));
    let online = run("cdn", None);
    assert!(online.contains(cdn));
    // Either way, a page without the library says what to do.
    for html in [&embedded, &online] {
        assert!(html.contains(r#"id="echarts-missing" hidden"#));
    }

    // A path that cannot be read fails before the input is read, not after
    // the fit (and the bootstrap) with no report.
    let stderr = outcome_of(
        command()
            .args(args![
                input,
                dir.path("bad.json"),
                "1",
                ..words("--bootstrap 2 --bootstrap-block-size 500"),
            ])
            .env("PSMC_ECHARTS_JS", dir.path("missing.js")),
    )
    .exits(1);
    assert!(stderr.contains("PSMC_ECHARTS_JS"), "{stderr}");
    assert!(!stderr.contains("Input:"), "{stderr}");
    assert!(!dir.path("bad.json").exists());
    assert!(!dir.path("bad.bootstrap").exists());
}
