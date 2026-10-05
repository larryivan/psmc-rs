//! Inference through the binary: defaults, settings and their errors.

mod common;

use common::psmc_text::parse_rounds;
use common::{TempDir, json, run_err, run_ok, small_psmcfa};

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

fn fit_json(dir: &TempDir, name: &str, n_iter: &str, extra: &[&str]) -> serde_json::Value {
    let input = dir.path("in.psmcfa");
    if !input.exists() {
        small_psmcfa(&input, 5000);
    }
    let out = dir.path(&format!("{name}.json"));
    let mut args = vec![
        input.display().to_string(),
        out.display().to_string(),
        n_iter.to_string(),
        "--no-progress".to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    run_ok(&args);
    json(&out)
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

    let input = dir.path("in.psmcfa");
    let out = dir.path("bad.json");
    let err = run_err([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "0",
        "--pattern",
        "4+5*3+4",
        "--n-steps",
        "30",
    ]);
    assert!(
        err.contains("--n-steps 30 does not match --pattern '4+5*3+4', which has 23 intervals"),
        "{err}"
    );
    assert!(!out.exists());
    let err = run_err([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "0",
        "--pattern",
        "4+5*",
    ]);
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
fn model_fields(v: &serde_json::Value) -> Vec<serde_json::Value> {
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
    let input = dir.path("in.psmcfa");
    let err = run_err([
        input.to_str().unwrap(),
        dir.path("bad.json").to_str().unwrap(),
        "1",
        "--init-params",
        saved,
        "--pattern",
        "4+25*2+4+6",
    ]);
    assert!(
        err.contains("holds a model with pattern 4+5*3+4 (23 intervals)"),
        "{err}"
    );
    let err = run_err([
        input.to_str().unwrap(),
        dir.path("bad.json").to_str().unwrap(),
        "1",
        "--init-params",
        saved,
        "--theta0",
        "0.01",
    ]);
    assert!(err.contains("cannot be used with"), "{err}");
}

/// With --init-params, the bootstrap replicates start from the saved
/// parameters, as the main fit does (C psmc -i -b), not from flat lambdas.
#[test]
fn init_params_bootstrap_replicates_start_from_the_saved_fit() {
    let dir = TempDir::new("init_bootstrap");
    fit_json(&dir, "two", "2", &[]);
    let saved = dir.path("two.json");
    let main_psmc = dir.path("main.psmc");
    let out = dir.path("cont.json");
    let o = common::run([
        dir.path("in.psmcfa").to_str().unwrap(),
        out.to_str().unwrap(),
        "1",
        "--no-progress",
        "--init-params",
        saved.to_str().unwrap(),
        "--psmc-out",
        main_psmc.to_str().unwrap(),
        "--bootstrap",
        "2",
        "--bootstrap-block-size",
        "1000",
    ]);
    assert!(o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains(&format!(
            "Warning: every bootstrap replicate starts from the parameters in {}",
            saved.display()
        )),
        "{stderr}"
    );
    let main = parse_rounds(&std::fs::read_to_string(&main_psmc).unwrap());
    let start = &main[0];
    assert!(start.rs.iter().any(|rs| rs[1] != 1.0), "a fitted start");
    for rep in 1..=2 {
        let path = dir.path(&format!("cont.bootstrap/replicate_{rep:03}.psmc"));
        let rounds = parse_rounds(&std::fs::read_to_string(path).unwrap());
        assert_eq!(rounds.len(), 2, "replicate {rep}");
        let first = &rounds[0];
        assert_eq!(
            (first.tr, first.mt),
            (start.tr, start.mt),
            "replicate {rep}"
        );
        let lambdas =
            |r: &common::psmc_text::Round| -> Vec<f64> { r.rs.iter().map(|rs| rs[1]).collect() };
        assert_eq!(lambdas(first), lambdas(start), "replicate {rep}");
    }
    let html = std::fs::read_to_string(out.with_extension("html")).unwrap();
    assert!(html.contains(&format!(
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
    let fails = |output: &std::path::Path, extra: &[&str]| -> String {
        let mut args = vec![
            input.display().to_string(),
            output.display().to_string(),
            "2".to_string(),
            "--no-progress".to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let o = common::run(&args);
        assert!(!o.status.success());
        assert!(
            o.stdout.is_empty(),
            "{}",
            String::from_utf8_lossy(&o.stdout)
        );
        let stderr = String::from_utf8(o.stderr).unwrap();
        // Nothing was read.
        assert!(!stderr.contains("Input:"), "{stderr}");
        stderr
    };
    let missing = dir.path("nowhere/t.tsv");
    let err = fails(
        &out,
        &[
            "--tmrca-out",
            missing.to_str().unwrap(),
            "--bootstrap",
            "2",
            "--bootstrap-block-size",
            "1000",
        ],
    );
    assert!(
        err.contains(&format!("cannot write --tmrca-out {}", missing.display())),
        "{err}"
    );
    assert!(err.contains("does not exist"), "{err}");
    let err = fails(&dir.path("nowhere/fit.json"), &[]);
    assert!(err.contains("cannot write OUTPUT"), "{err}");

    let err = fails(&dir.path("fit.html"), &[]);
    assert!(err.contains("ends in .html"), "{err}");
    let err = fails(&out, &["--psmc-out", out.to_str().unwrap()]);
    assert!(err.contains("are the same file"), "{err}");
    let err = fails(&input, &[]);
    assert!(err.contains("is the input file"), "{err}");
    let err = fails(&out, &["--qc-json", input.to_str().unwrap()]);
    assert!(err.contains("--qc-json"), "{err}");
    assert!(err.contains("is the input file"), "{err}");

    for name in ["fit.json", "fit.html", "fit.bootstrap", "nowhere"] {
        assert!(!dir.path(name).exists(), "{name}");
    }
    assert_eq!(std::fs::read(&input).unwrap(), original);
}

/// The JSON of psmc-rs 0.1, without run metadata, is a valid start.
#[test]
fn init_params_reads_files_of_version_0_1() {
    let dir = TempDir::new("init_v01");
    let old = dir.path("old.json");
    std::fs::write(
        &old,
        r#"{"theta": 0.0068, "rho": 0.0014, "lam": [1.5, 0.7, 2.0, 3.0, 1.0, 0.5, 4.0],
            "t_max": 14.2, "n_steps": 22, "mu": 2.5e-8, "pattern": "4+5*3+4"}"#,
    )
    .unwrap();
    let v = fit_json(&dir, "out", "0", &["--init-params", old.to_str().unwrap()]);
    assert_eq!(v["theta"], 0.0068);
    assert_eq!(v["lam"][6], 4.0);
    let err = run_err([
        dir.path("in.psmcfa").to_str().unwrap(),
        dir.path("x.json").to_str().unwrap(),
        "0",
        "--init-params",
        dir.path("missing.json").to_str().unwrap(),
    ]);
    assert!(err.contains("--init-params"), "{err}");
}
