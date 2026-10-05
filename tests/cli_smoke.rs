use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

/// A path no other test of this process uses: tests run in parallel, and
/// the clock alone is too coarse to tell them apart.
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

/// A directory path no other test of this process uses.
fn unique_temp_dir(prefix: &str) -> PathBuf {
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}", std::process::id()))
}

fn find_psmc_binary() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_psmc-rs") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_psmc_rs") {
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

    let direct = debug_dir.join("psmc-rs");
    if direct.exists() {
        return direct;
    }
    let direct_compat = debug_dir.join("psmc");
    if direct_compat.exists() {
        return direct_compat;
    }

    for entry in fs::read_dir(&deps_dir).expect("failed to read target deps dir") {
        let entry = entry.expect("failed to read deps entry");
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        if !(name.starts_with("psmc_rs-") || name.starts_with("psmc-")) {
            continue;
        }
        if name.ends_with(".d") || name.ends_with(".rlib") || name.ends_with(".rmeta") {
            continue;
        }
        if path.is_file() {
            return path;
        }
    }

    panic!("failed to find psmc-rs binary in Cargo env or target/debug");
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

#[test]
fn cli_bootstrap_writes_replicates_summary_and_html_overlay() {
    let input = unique_temp_path("psmc_cli_boot_input", "psmcfa");
    let output = unique_temp_path("psmc_cli_boot_output", "json");
    let report = output.with_extension("html");
    let boot_dir = unique_temp_dir("psmc_cli_boot_dir");
    let content = "> chr1\nTTTTTKTTTTKTTTTNTTTTKTTTTKTTTTTTTTTKTTTTKTTTTNTTTTKTTTTKTTTT\n";
    fs::write(&input, content).expect("failed to write cli bootstrap input");

    let exe = find_psmc_binary();
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .arg("1")
        .arg("--bootstrap")
        .arg("2")
        .arg("--bootstrap-block-size")
        .arg("8")
        .arg("--bootstrap-seed")
        .arg("7")
        .arg("--bootstrap-dir")
        .arg(&boot_dir)
        .arg("--no-progress")
        .status()
        .expect("failed to run psmc binary");
    assert!(status.success(), "psmc exited with non-zero status");

    let rep1 = boot_dir.join("replicate_001.json");
    let rep2 = boot_dir.join("replicate_002.json");
    let summary_tsv = boot_dir.join("summary.tsv");
    let summary_json = boot_dir.join("summary.json");
    assert!(rep1.exists(), "expected bootstrap replicate_001.json");
    assert!(rep2.exists(), "expected bootstrap replicate_002.json");
    assert!(summary_tsv.exists(), "expected bootstrap summary.tsv");
    assert!(summary_json.exists(), "expected bootstrap summary.json");

    let summary_text = fs::read_to_string(&summary_tsv).expect("failed to read bootstrap summary");
    assert!(
        summary_text.starts_with("x_years\tne_main\tne_q025\tne_q500\tne_q975"),
        "bootstrap summary header is missing"
    );

    let html = fs::read_to_string(&report).expect("failed to read html report");
    assert!(
        html.contains("Bootstrap q2.5"),
        "html report should include bootstrap series"
    );

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(report);
    let _ = fs::remove_dir_all(boot_dir);
}

const SMALL_VCF: &str = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1
chr1\t24\t.\tA\tC\t.\tPASS\t.\tGT\t1/0
";

#[test]
fn cli_vcf_input_runs_and_writes_json() {
    let input = unique_temp_path("psmc_cli_vcf_input", "vcf");
    let output = unique_temp_path("psmc_cli_vcf_output", "json");
    let report = output.with_extension("html");
    fs::write(&input, SMALL_VCF).expect("failed to write cli vcf input");

    // VCF input goes through fq2psmcfa's output filter, which drops a
    // sequence with fewer than 10,000 called bases; this 24 bp chromosome
    // needs it off. The old --mhs-bin-size spelling of --bin-size still works.
    let exe = find_psmc_binary();
    let status = Command::new(exe)
        .arg(&input)
        .arg(&output)
        .arg("0")
        .arg("--input-format")
        .arg("vcf")
        .arg("--mhs-bin-size")
        .arg("4")
        .arg("--min-good-bases")
        .arg("0")
        .arg("--min-good-fraction")
        .arg("0")
        .arg("--no-progress")
        .status()
        .expect("failed to run psmc binary with vcf input");
    assert!(status.success(), "psmc exited with non-zero status");

    let out = fs::read_to_string(&output).expect("failed to read psmc output json");
    let v: Value = serde_json::from_str(&out).expect("output json is invalid");
    assert!(v.get("theta").is_some());
    assert!(v.get("rho").is_some());
    assert!(v.get("lam").is_some());
    assert!(report.exists(), "expected html report to be generated");

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
    let _ = fs::remove_file(report);
}

#[test]
fn cli_vcf_input_dropped_by_the_output_filter_is_explained() {
    let input = unique_temp_path("psmc_cli_vcf_filtered", "vcf");
    let output = unique_temp_path("psmc_cli_vcf_filtered_output", "json");
    fs::write(&input, SMALL_VCF).expect("failed to write cli vcf input");

    let out = Command::new(find_psmc_binary())
        .arg(&input)
        .arg(&output)
        .arg("0")
        .arg("--bin-size")
        .arg("4")
        .arg("--no-progress")
        .output()
        .expect("failed to run psmc binary with vcf input");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("chr1: 24 bases, 24 called (< 10000)"),
        "{stderr}"
    );
    assert!(
        stderr.contains("--min-good-bases 0 --min-good-fraction 0"),
        "{stderr}"
    );
    assert!(!output.exists());
    let _ = fs::remove_file(input);
}

/// Multihetsep input is recognised from its content, and --mhs-pair picks
/// the two haplotypes: the starting theta follows the pair's het bins.
#[test]
fn cli_reads_multihetsep_with_the_chosen_pair() {
    let input = unique_temp_path("psmc_cli_mhs_input", "txt");
    // Haplotypes 0,1 differ at every row, 2,3 at the first row only.
    let content = "\
chr1\t150\t150\tACAG
chr1\t450\t300\tGTGG
chr1\t850\t400\tCACC
chr1\t1000\t150\tTGTT
";
    fs::write(&input, content).expect("failed to write mhs input");
    let theta = |extra: &[&str]| -> f64 {
        let output = unique_temp_path("psmc_cli_mhs_output", "json");
        let status = Command::new(find_psmc_binary())
            .arg(&input)
            .arg(&output)
            .arg("0")
            .args(["--min-good-bases", "0", "--no-progress"])
            .args(extra)
            .status()
            .expect("failed to run psmc binary with mhs input");
        assert!(status.success(), "{extra:?}");
        let v: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
        let _ = fs::remove_file(&output);
        let _ = fs::remove_file(output.with_extension("html"));
        v["theta"].as_f64().unwrap()
    };
    let auto = theta(&[]);
    assert_eq!(auto, theta(&["--input-format", "mhs"]));
    assert_eq!(auto, theta(&["--mhs-pair", "0,1"]));
    let other = theta(&["--mhs-pair", "2,3"]);
    // 10 bins of 100 bp: 4 heterozygous for 0,1, 1 for 2,3.
    assert_eq!(auto, -(1.0f64 - 4.0 / 10.0).ln());
    assert_eq!(other, -(1.0f64 - 1.0 / 10.0).ln());
    let _ = fs::remove_file(input);
}
