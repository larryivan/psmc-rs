//! Inference through the binary: defaults, settings and their errors.

mod common;

use common::psmc_text::{Round, parse_rounds};
use common::{
    TempDir, args, finding_codes, json, outcome, read, run_err, run_ok, small_psmcfa, words,
};
use serde_json::Value;
use std::path::PathBuf;

/// Fit `dir/in.psmcfa` (5,000 bins of each reference sequence, written on
/// the first call) for `n_iter` iterations into `dir/<name>.json`.
fn fit_json(dir: &TempDir, name: &str, n_iter: &str, extra: &[&str]) -> Value {
    let input = dir.path("in.psmcfa");
    if !input.exists() {
        small_psmcfa(&input, 5000);
    }
    let out = dir.path(&format!("{name}.json"));
    run_ok(args![input, out, n_iter, "--no-progress", ..extra]);
    json(&out)
}

/// With neither --pattern nor --n-steps the model is C psmc's default, and
/// rho0 = theta0 / 4 (C's -r 4).
#[test]
fn defaults_follow_c_psmc() {
    let dir = TempDir::new("defaults");
    let v = fit_json(&dir, "out", "0", &[]);
    assert_eq!(v["pattern"], "4+5*3+4");
    assert_eq!(v["n_steps"], 22);
    assert_eq!(v["lam"].as_array().unwrap().len(), 7);
    assert_eq!(v["t_max"], 15.0);
    let theta = v["theta"].as_f64().unwrap();
    assert_eq!(v["rho"].as_f64().unwrap(), theta / 4.0);
}

#[test]
fn n_steps_alone_gives_one_lambda_per_interval() {
    let dir = TempDir::new("n_steps");
    let v = fit_json(&dir, "n10", "0", &["--n-steps", "10"]);
    assert!(v["pattern"].is_null());
    assert_eq!(v["n_steps"], 10);
    assert_eq!(v["lam"].as_array().unwrap().len(), 11);
}

#[test]
fn n_steps_is_taken_from_the_pattern_and_checked() {
    let dir = TempDir::new("pattern");
    let v = fit_json(&dir, "human", "0", &["--pattern", "4+25*2+4+6"]);
    assert_eq!(v["n_steps"], 63);
    assert_eq!(v["lam"].as_array().unwrap().len(), 28);
    // psmc-rs 0.1 took the number of intervals with a pattern.
    let v = fit_json(
        &dir,
        "legacy",
        "0",
        &["--pattern", "4+25*2+4+6", "--n-steps", "64"],
    );
    assert_eq!(v["n_steps"], 63);

    let (input, out) = (dir.path("in.psmcfa"), dir.path("bad.json"));
    let err = run_err(args![
        input,
        out,
        "0",
        ..words("--pattern 4+5*3+4 --n-steps 30")
    ]);
    assert!(
        err.contains("--n-steps 30 does not match --pattern '4+5*3+4', which has 23 intervals"),
        "{err}"
    );
    assert!(!out.exists());
    let err = run_err(args![input, out, "0", "--pattern", "4+5*"]);
    assert!(err.contains("invalid --pattern '4+5*'"), "{err}");
}

#[test]
fn rho0_follows_tr_ratio_unless_given() {
    let dir = TempDir::new("rho0");
    let v = fit_json(&dir, "r5", "0", &["--tr-ratio", "5"]);
    assert_eq!(
        v["rho"].as_f64().unwrap(),
        v["theta"].as_f64().unwrap() / 5.0
    );
    let v = fit_json(&dir, "rho", "0", &["--rho0", "0.001", "--theta0", "0.02"]);
    assert_eq!(
        (v["theta"].as_f64(), v["rho"].as_f64()),
        (Some(0.02), Some(0.001))
    );
}

/// The M-step objective is C psmc's by default: no smoothness penalty.
#[test]
fn smoothing_is_off_by_default() {
    let dir = TempDir::new("smooth");
    let default = fit_json(&dir, "default", "2", &[]);
    let zero = fit_json(&dir, "zero", "2", &["--smooth-lambda", "0"]);
    let smooth = fit_json(&dir, "smooth", "2", &["--smooth-lambda", "10"]);
    assert_eq!(default["lam"], zero["lam"]);
    assert_ne!(default["lam"], smooth["lam"]);
}

/// The model fields of a fit, as written.
fn model_fields(v: &Value) -> Vec<Value> {
    ["theta", "rho", "t_max", "lam", "n_steps", "pattern", "mu"]
        .iter()
        .map(|k| v[k].clone())
        .collect()
}

/// EM keeps no state beyond the parameters, so 2 iterations from the saved
/// result of 2 iterations are 4 iterations, to the bit.
#[test]
fn init_params_continues_a_saved_fit() {
    let dir = TempDir::new("init_params");
    let two = fit_json(&dir, "two", "2", &[]);
    let saved = dir.path("two.json");
    let saved = saved.to_str().unwrap();
    let again = fit_json(&dir, "again", "0", &["--init-params", saved]);
    assert_eq!(model_fields(&again), model_fields(&two));
    let continued = fit_json(&dir, "continued", "2", &["--init-params", saved]);
    let four = fit_json(&dir, "four", "4", &[]);
    assert_eq!(model_fields(&continued), model_fields(&four));
    // The pattern can be restated, but not changed.
    let same = fit_json(
        &dir,
        "same",
        "0",
        &["--init-params", saved, "--pattern", "4+5*3+4"],
    );
    assert_eq!(model_fields(&same), model_fields(&two));
    let fails = |extra: &[&str]| {
        run_err(args![
            dir.path("in.psmcfa"),
            dir.path("bad.json"),
            "1",
            "--init-params",
            saved,
            ..extra
        ])
    };
    let err = fails(&["--pattern", "4+25*2+4+6"]);
    assert!(
        err.contains("holds a model with pattern 4+5*3+4 (23 intervals)"),
        "{err}"
    );
    let err = fails(&["--theta0", "0.01"]);
    assert!(err.contains("cannot be used with"), "{err}");
}

/// N_ITER 0 writes the starting model with its log-likelihood: `loglike`
/// holds N_ITER + 1 = 1 value, `final_loglike` equals it, and the run
/// prints it. For saved parameters it is their saved log-likelihood, to
/// the bit, and for a fresh start the first value of a longer fit.
#[test]
fn zero_iterations_record_the_starting_loglike() {
    let dir = TempDir::new("zero_iter_loglike");
    let two = fit_json(&dir, "two", "2", &[]);
    let saved_ll = two["final_loglike"].as_f64().unwrap();
    let fresh = fit_json(&dir, "fresh", "0", &[]);
    assert_eq!(fresh["n_iter"], 0);
    assert_eq!(fresh["loglike"], serde_json::json!([two["loglike"][0]]));
    assert_eq!(fresh["final_loglike"], two["loglike"][0]);

    let out = run_ok(args![
        dir.path("in.psmcfa"),
        dir.path("again.json"),
        "0",
        "--no-progress",
        "--init-params",
        dir.path("two.json"),
    ]);
    assert!(
        out.contains(&format!("Last EM loglike: {saved_ll}\n")),
        "{out}"
    );
    let again = json(&dir.path("again.json"));
    assert_eq!(again["loglike"], serde_json::json!([saved_ll]));
    assert_eq!(again["final_loglike"].as_f64(), Some(saved_ll));
    // The report shows it too.
    let html = read(dir.path("again.html"));
    assert!(
        html.contains(&format!(
            "<tr><th>Final log-likelihood</th><td>{saved_ll:.6}</td></tr>"
        )),
        "{html}"
    );
}

/// With --init-params, the bootstrap replicates start from the saved
/// parameters, as the main fit does (C psmc -i -b), not from flat lambdas.
#[test]
fn init_params_bootstrap_replicates_start_from_the_saved_fit() {
    let dir = TempDir::new("init_bootstrap");
    fit_json(&dir, "two", "2", &[]);
    let saved = dir.path("two.json");
    let main_psmc = dir.path("main.psmc");
    let stderr = outcome(args![
        dir.path("in.psmcfa"),
        dir.path("cont.json"),
        "1",
        "--no-progress",
        "--init-params",
        saved,
        "--psmc-out",
        main_psmc,
        ..words("--bootstrap 2 --bootstrap-block-size 1000"),
    ])
    .ok()
    .stderr;
    assert!(
        stderr.contains(&format!(
            "warning: every bootstrap replicate starts from the parameters in {}",
            saved.display()
        )),
        "{stderr}"
    );
    let main = parse_rounds(&read(&main_psmc));
    let start = &main[0];
    assert!(start.rs.iter().any(|rs| rs[1] != 1.0), "a fitted start");
    let lambdas = |r: &Round| -> Vec<f64> { r.rs.iter().map(|rs| rs[1]).collect() };
    for rep in 1..=2 {
        let rounds = parse_rounds(&read(
            dir.path(&format!("cont.bootstrap/replicate_{rep:03}.psmc")),
        ));
        assert_eq!(rounds.len(), 2, "replicate {rep}");
        let first = &rounds[0];
        assert_eq!(
            (first.tr, first.mt),
            (start.tr, start.mt),
            "replicate {rep}"
        );
        assert_eq!(lambdas(first), lambdas(start), "replicate {rep}");
    }
    assert!(read(dir.path("cont.html")).contains(&format!(
        "<tr><th>Replicates start from</th><td>the parameters in {}</td></tr>",
        saved.display()
    )));
}

/// Outputs that cannot be written, or that would overwrite each other or
/// the input, fail before the input is read, not after the fit.
#[test]
fn output_paths_are_checked_before_the_fit() {
    let dir = TempDir::new("output_paths");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let original = std::fs::read(&input).unwrap();
    let out = dir.path("fit.json");
    let fails = |output: &PathBuf, extra: &[&str]| -> String {
        let o = outcome(args![input, output, "2", "--no-progress", ..extra]);
        assert!(!o.success());
        assert!(o.stdout.is_empty(), "{}", o.stdout);
        // Nothing was read.
        assert!(!o.stderr.contains("Input QC"), "{}", o.stderr);
        o.stderr
    };
    let missing = dir.path("nowhere/t.tsv");
    let missing_str = missing.to_str().unwrap();
    let err = fails(
        &out,
        &[
            "--tmrca-out",
            missing_str,
            "--bootstrap",
            "2",
            "--bootstrap-block-size",
            "1000",
        ],
    );
    assert!(
        err.contains(&format!("cannot write --tmrca-out {missing_str}")),
        "{err}"
    );
    assert!(err.contains("does not exist"), "{err}");
    let cases: [(PathBuf, &[&str], &[&str]); 5] = [
        (dir.path("nowhere/fit.json"), &[], &["cannot write OUTPUT"]),
        (dir.path("fit.html"), &[], &["ends in .html"]),
        (
            out.clone(),
            &["--psmc-out", out.to_str().unwrap()],
            &["are the same file"],
        ),
        (input.clone(), &[], &["is the input file"]),
        (
            out.clone(),
            &["--qc-json", input.to_str().unwrap()],
            &["--qc-json", "is the input file"],
        ),
    ];
    for (output, extra, messages) in cases {
        let err = fails(&output, extra);
        for message in messages {
            assert!(err.contains(message), "{extra:?}: {err}");
        }
    }

    for name in ["fit.json", "fit.html", "fit.bootstrap", "nowhere"] {
        assert!(!dir.path(name).exists(), "{name}");
    }
    assert_eq!(std::fs::read(&input).unwrap(), original);
}

/// Options of another input format are named in a warning rather than
/// dropped without a word.
#[test]
fn options_of_another_format_are_reported() {
    let dir = TempDir::new("ignored_options");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 2000);
    let stderr = outcome(args![
        input,
        dir.path("fit.json"),
        "0",
        "-q",
        "--no-progress",
        "--vcf-mask",
        dir.path("missing.bed"),
        ..words("--vcf-sample s1 --depth 5:10 --mhs-pair 2,3"),
    ])
    .ok()
    .stderr;
    assert!(
        stderr.contains(
            "warning: --vcf-sample, --vcf-mask, --depth and --mhs-pair do not apply to psmcfa \
             input and were ignored"
        ),
        "{stderr}"
    );
    let plain = outcome(args![input, dir.path("plain.json"), "0", "--no-progress"]);
    assert!(!plain.stderr.contains("do not apply"));
}

/// When the sequence filter leaves nothing, the fit fails, but the QC
/// report asked for is written first and says why.
#[test]
fn qc_json_is_written_when_nothing_is_left_to_fit() {
    let dir = TempDir::new("nothing_left");
    let qc_json = dir.path("qc.json");
    let err = run_err(args![
        common::fixture("route_a/allsites.vcf.gz"),
        dir.path("fit.json"),
        "1",
        ..words("--no-progress --depth 3:100000 --min-good-bases 1000000"),
        "--qc-json",
        qc_json,
    ]);
    assert!(
        err.contains("pass --min-good-bases 0 --min-good-fraction 0"),
        "{err}"
    );
    let qc = json(&qc_json);
    assert_eq!(qc["n_sequences"], 0);
    assert!(finding_codes(&qc).contains(&"DROPPED_SEQUENCES".to_string()));
    assert!(!dir.path("fit.json").exists());
}

/// Errors start with clap's `error: ` and name each file once, with what
/// went wrong.
#[test]
fn errors_name_the_file_once() {
    let dir = TempDir::new("error_style");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 2000);
    let missing = dir.path("missing.psmcfa");
    let saved = dir.write("saved.json", "{}");
    for (args, file, what) in [
        (
            args![missing, dir.path("a.json"), "1"],
            &missing,
            "failed to open",
        ),
        (
            args![input, dir.path("b.json"), "1", "--init-params", saved],
            &saved,
            "--init-params: failed to parse",
        ),
    ] {
        let stderr = outcome(args).exits(1);
        assert!(stderr.starts_with("error: "), "{stderr}");
        assert!(stderr.contains(what), "{stderr}");
        let name = file.display().to_string();
        assert_eq!(stderr.matches(&name).count(), 1, "{stderr}");
    }
}

/// The JSON of psmc-rs 0.1, without run metadata, is a valid start.
#[test]
fn init_params_reads_files_of_version_0_1() {
    let dir = TempDir::new("init_v01");
    let old = dir.write(
        "old.json",
        r#"{"theta": 0.0068, "rho": 0.0014, "lam": [1.5, 0.7, 2.0, 3.0, 1.0, 0.5, 4.0],
            "t_max": 14.2, "n_steps": 22, "mu": 2.5e-8, "pattern": "4+5*3+4"}"#,
    );
    let v = fit_json(&dir, "out", "0", &["--init-params", old.to_str().unwrap()]);
    assert_eq!(v["theta"], 0.0068);
    assert_eq!(v["lam"][6], 4.0);
    let err = run_err(args![
        dir.path("in.psmcfa"),
        dir.path("x.json"),
        "0",
        "--init-params",
        dir.path("missing.json"),
    ]);
    assert!(err.contains("--init-params"), "{err}");
}

/// A command line that parses but cannot run as given exits with status 2,
/// as clap's own usage errors do (README: "2 on a usage error"); errors met
/// while reading or writing files exit with status 1. None reads the input.
#[test]
fn usage_errors_exit_with_status_2() {
    let dir = TempDir::new("usage_status");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 2000);
    let saved = dir.path("saved.json");
    run_ok(args![input, saved, "0", "--no-progress"]);
    let out = dir.path("fit.json");
    let exits = |code: i32, extra: &[&str]| {
        let stderr = outcome(args![input, out, "1", "--no-progress", ..extra]).exits(code);
        assert!(!stderr.contains("Input QC"), "{extra:?}: {stderr}");
        stderr
    };
    let saved = saved.to_str().unwrap();
    let usage: &[(&[&str], &str)] = &[
        (&["--pattern", "4+x"], "invalid --pattern '4+x'"),
        (
            &["--pattern", "4+5*3+4", "--n-steps", "30"],
            "does not match",
        ),
        (&["--bootstrap-iters", "3"], "--bootstrap-iters applies"),
        (&["--bootstrap-dir", "boot"], "--bootstrap-dir applies"),
        (
            &["--bootstrap", "2", "--bootstrap-iters", "0"],
            "0 EM iterations",
        ),
        (
            &["--init-params", saved, "--pattern", "4+25*2+4+6"],
            "holds a model with pattern",
        ),
        (&["--psmc-out", out.to_str().unwrap()], "are the same file"),
        (&["--qc-json", input.to_str().unwrap()], "is the input file"),
        // clap's own.
        (&["--threads", "0"], "at least 1"),
    ];
    for (extra, message) in usage {
        let stderr = exits(2, extra);
        assert!(stderr.contains(message), "{extra:?}: {stderr}");
    }
    outcome(args![input, dir.path("fit.html"), "1"]).exits(2);
    // The TUI's refusal of the QC JSON on stdout, before the interface
    // starts.
    let stderr = outcome(args![input, out, "1", "--tui", "--qc-json", "-"]).exits(2);
    assert!(
        stderr.contains("give --qc-json a file with --tui"),
        "{stderr}"
    );

    // Not usage errors: a file that cannot be read or written.
    exits(
        1,
        &["--init-params", dir.path("missing.json").to_str().unwrap()],
    );
    exits(
        1,
        &["--tmrca-out", dir.path("nowhere/t.tsv").to_str().unwrap()],
    );
    assert!(!out.exists());

    // convert: multihetsep output asked of psmcfa input.
    outcome(args![
        "convert",
        input,
        "--input-format",
        "psmcfa",
        "--multihetsep",
        dir.path("x.mhs"),
    ])
    .exits(2);
}

/// A start EM cannot move from is refused before the fit, with advice this
/// run can follow: --tr-ratio or --rho0 for a fresh start, a smaller --rho0
/// when one was given, and the saved file's rho for --init-params, which
/// rules out the options of a fresh start.
#[test]
fn an_invalid_start_is_refused_with_advice_the_run_can_follow() {
    let dir = TempDir::new("invalid_start");
    // Two sequences whose bins are 60% heterozygous: C's start (theta0 =
    // -ln(1 - K/L) = 0.92, rho0 = theta0 / 4 = 0.23) is past the limit of
    // about 1/6 on rho per bin at t_max 15.
    let bins = "KKKTT".repeat(2000);
    let input = dir.write("diverse.psmcfa", format!(">a\n{bins}\n>b\n{bins}\n"));
    let out = dir.path("fit.json");
    // The error line (QC's own advice comes before it).
    let fit_err = |extra: &[&str]| {
        let stderr = outcome(args![input, out, "1", "--no-progress", ..extra]).exits(1);
        assert!(!out.exists(), "{stderr}");
        stderr
            .lines()
            .find(|l| l.starts_with("error: "))
            .unwrap_or_else(|| panic!("{stderr}"))
            .to_string()
    };
    let begins = "error: EM cannot start from these parameters: the transition probabilities are \
                  negative";
    let err = fit_err(&[]);
    assert!(err.contains(begins), "{err}");
    let key = "start from a smaller rho0: --tr-ratio ";
    let rest = &err[err.find(key).unwrap_or_else(|| panic!("{err}")) + key.len()..];
    let ratio = rest.split_whitespace().next().unwrap();
    assert!(
        rest.starts_with(&format!(
            "{ratio} or more (C psmc's -r), or a --rho0 of 0.16"
        )),
        "{err}"
    );
    // The advice works.
    run_ok(args![input, out, "1", "--no-progress", "--tr-ratio", ratio]);
    std::fs::remove_file(&out).unwrap();

    let err = fit_err(&["--rho0", "0.3"]);
    assert!(err.contains(begins), "{err}");
    assert!(err.contains("or give a --rho0 of 0.16"), "{err}");
    assert!(!err.contains("--tr-ratio"), "{err}");

    // Saved parameters with that rho (written without iterations, which
    // fit nothing and so refuse nothing).
    let saved = dir.path("saved.json");
    run_ok(args![input, saved, "0", "--no-progress", "--rho0", "0.3"]);
    let from_saved = ["--init-params", saved.to_str().unwrap()];
    let err = fit_err(&from_saved);
    assert!(err.contains(begins), "{err}");
    assert!(
        err.contains(&format!("Set rho in {} below 0.16", saved.display())),
        "{err}"
    );
    assert!(
        err.contains("fit from a fresh start instead of --init-params"),
        "{err}"
    );
    assert!(
        !err.contains("--tr-ratio") && !err.contains("--rho0"),
        "{err}"
    );
    // QC's advice, printed before the error, also speaks of the saved file.
    let stderr = outcome(args![input, out, "1", "--no-progress", ..from_saved]).stderr;
    assert!(stderr.contains("INVALID_START"), "{stderr}");
    assert!(
        stderr.contains("lower rho in the --init-params file"),
        "{stderr}"
    );
    assert!(
        !stderr.contains("--tr-ratio") && !stderr.contains("--rho0"),
        "{stderr}"
    );
}
