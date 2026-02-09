use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before unix epoch")
        .as_nanos();
    path.push(format!("{prefix}_{}_{}.{}", std::process::id(), nanos, ext));
    path
}

fn find_psmc_binary() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_psmc") {
        return PathBuf::from(path);
    }

    let current = std::env::current_exe().expect("failed to get current exe path");
    let deps_dir = current
        .parent()
        .expect("failed to get deps dir from current exe")
        .to_path_buf();
    let debug_dir = deps_dir
        .parent()
        .expect("failed to get debug dir from deps dir")
        .to_path_buf();

    let direct = debug_dir.join("psmc");
    if direct.exists() {
        return direct;
    }

    for entry in fs::read_dir(&deps_dir).expect("failed to read target deps dir") {
        let entry = entry.expect("failed to read deps entry");
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if !name.starts_with("psmc-") {
            continue;
        }
        if name.ends_with(".d") || name.ends_with(".rlib") || name.ends_with(".rmeta") {
            continue;
        }
        if path.is_file() {
            return path;
        }
    }

    panic!("failed to find psmc binary in CARGO_BIN_EXE_psmc or target/debug");
}

#[test]
fn cli_runs_and_writes_json() {
    let input = unique_temp_path("psmc_cli_input", "psmcfa");
    let output = unique_temp_path("psmc_cli_output", "json");
    let report = output.with_extension("html");
    let content = "> chr1\nTTTTTKTTTTKTTTTNTTTTKTTTTKTTTT\n";
    fs::write(&input, content).expect("failed to write cli test input");

    let exe = find_psmc_binary();
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .arg("0")
        .arg("--no-progress")
        .status()
        .expect("failed to run psmc binary");
    assert!(status.success(), "psmc exited with non-zero status");

    let out = fs::read_to_string(&output).expect("failed to read psmc output json");
    let v: Value = serde_json::from_str(&out).expect("output json is invalid");
    assert!(v.get("theta").is_some());
    assert!(v.get("rho").is_some());
    assert!(v.get("lam").is_some());
    assert!(v.get("t_max").is_some());
    assert!(v.get("n_steps").is_some());
    assert!(v.get("mu").is_some());
    assert!(report.exists(), "expected html report to be generated");

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(report);
}

#[test]
fn cli_tmrca_writes_tsv_and_html_tmrca_panel() {
    let input = unique_temp_path("psmc_cli_tmrca_input", "psmcfa");
    let output = unique_temp_path("psmc_cli_tmrca_output", "json");
    let tmrca = unique_temp_path("psmc_cli_tmrca_posterior", "tsv");
    let report = output.with_extension("html");
    let content = "> chr1\nTTTTTKTTTTKTTTTNTTTTKTTTTKTTTT\n";
    fs::write(&input, content).expect("failed to write cli test input");

    let exe = find_psmc_binary();
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .arg("0")
        .arg("--tmrca-out")
        .arg(&tmrca)
        .arg("--no-progress")
        .status()
        .expect("failed to run psmc binary");
    assert!(status.success(), "psmc exited with non-zero status");

    let tmrca_text = fs::read_to_string(&tmrca).expect("failed to read tmrca tsv");
    assert!(
        tmrca_text.starts_with(
            "seq_id\tseq_bin\tglobal_bin\tobs\tmap_state\ttmrca_map_years\ttmrca_mean_years\ttmrca_q025_years\ttmrca_q975_years\tpmax\tentropy"
        ),
        "tmrca header is missing"
    );
    assert!(
        tmrca_text.lines().count() > 1,
        "tmrca output should contain data rows"
    );

    let html = fs::read_to_string(&report).expect("failed to read html report");
    assert!(
        html.contains("TMRCA Posterior Track"),
        "html report should include tmrca panel"
    );

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(tmrca);
    let _ = fs::remove_file(report);
}
