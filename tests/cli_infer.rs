//! Inference through the binary: defaults, settings and their errors.

mod common;

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
