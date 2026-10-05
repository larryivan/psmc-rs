//! `psmc-rs qc` through the binary: the report of an input read as
//! inference and `convert` read it, and the exit status of --strict.

mod common;

use common::{TempDir, fixture, json, run, run_ok};
use std::fs;

/// `qc` reads the input as convert and inference do, and --strict turns its
/// warnings into exit status 2.
#[test]
fn qc_reports_and_fails_strictly_on_warnings() {
    let dir = TempDir::new("qc");
    let vcf = fixture("route_a/allsites.vcf.gz");
    let json_path = dir.path("qc.json");
    let opts = ["--depth", "3:100000", "--min-good-bases", "1000"];
    let mut args = vec![
        "qc",
        vcf.to_str().unwrap(),
        "--json",
        json_path.to_str().unwrap(),
    ];
    args.extend_from_slice(&opts);
    let stdout = run_ok(&args);
    assert!(
        stdout.starts_with("Input QC (vcf, consensus mode): 3 sequences"),
        "{stdout}"
    );
    assert!(stdout.contains("WARNING  TOO_SHORT"), "{stdout}");
    let report = json(&json_path);
    assert_eq!(report["input_format"], "vcf, consensus mode");
    assert_eq!(report["n_sequences"], 3);
    // Consensus mode passes its depth histogram and bounds to QC.
    assert_eq!(report["depth"]["min_depth"], 3);
    assert_eq!(report["depth"]["max_depth"], 100000);
    assert!(report["depth"]["mean"].as_f64().unwrap() > 20.0);
    let codes: Vec<&str> = report["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["code"].as_str().unwrap())
        .collect();
    assert!(codes.contains(&"TOO_SHORT"), "{codes:?}");

    // The same report from convert.
    let convert_json = dir.path("convert_qc.json");
    let psmcfa = dir.path("x.psmcfa");
    let mut args = vec![
        "convert",
        vcf.to_str().unwrap(),
        "--psmcfa",
        psmcfa.to_str().unwrap(),
        "--qc-json",
        convert_json.to_str().unwrap(),
    ];
    args.extend_from_slice(&opts);
    run_ok(&args);
    assert_eq!(json(&convert_json), report);

    // --strict: status 2 with warnings, 0 without.
    let mut args = vec!["qc", vcf.to_str().unwrap(), "--strict"];
    args.extend_from_slice(&opts);
    let o = run(&args);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8(o.stdout).unwrap().contains("WARNING"));

    // --json - writes the JSON on stdout and the text on stderr.
    let o = run([
        "qc",
        vcf.to_str().unwrap(),
        "--json",
        "-",
        "--depth",
        "3:100000",
    ]);
    assert!(o.status.success());
    let parsed: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(parsed["n_sequences"], 1);
    assert!(String::from_utf8(o.stderr).unwrap().contains("Input QC"));
}

#[test]
fn qc_without_warnings_passes_strict() {
    let dir = TempDir::new("qc_clean");
    // 120 Mb of human-like data in one sequence: no warning applies.
    let input = dir.path("clean.psmcfa");
    let unit = format!("{}K", "T".repeat(199));
    let seq = unit.repeat(6000);
    fs::write(&input, format!(">chr1\n{seq}\n")).unwrap();
    let o = run(["qc", input.to_str().unwrap(), "--strict"]);
    let stdout = String::from_utf8(o.stdout).unwrap();
    assert_eq!(o.status.code(), Some(0), "{stdout}");
    assert!(!stdout.contains("WARNING"), "{stdout}");
}

/// --json is checked as inference and convert check their outputs, before
/// the input is read: it may not be the input, and its directory must
/// exist.
#[test]
fn the_json_path_is_checked_before_reading() {
    let dir = TempDir::new("qc_json_path");
    let input = dir.path("in.psmcfa");
    let text = format!(">chr1\n{}\n", "TTTTKTTTTN".repeat(200));
    fs::write(&input, &text).unwrap();
    let o = run([
        "qc",
        input.to_str().unwrap(),
        "--json",
        input.to_str().unwrap(),
    ]);
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert_eq!(o.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--json"), "{stderr}");
    assert!(stderr.contains("is the input file"), "{stderr}");
    assert!(o.stdout.is_empty());
    assert_eq!(fs::read_to_string(&input).unwrap(), text);

    let missing = dir.path("nowhere/qc.json");
    let o = run([
        "qc",
        input.to_str().unwrap(),
        "--json",
        missing.to_str().unwrap(),
    ]);
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert_eq!(o.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("cannot write --json"), "{stderr}");
    // Nothing was read: no report and no description of the input.
    assert!(o.stdout.is_empty());
    assert!(!stderr.contains("psmcfa:"), "{stderr}");
}

/// Options of the other VCF mode are named in a warning.
#[test]
fn options_of_the_other_vcf_mode_are_reported() {
    let vcf = fixture("route_a/allsites.vcf.gz");
    let o = run([
        "qc",
        vcf.to_str().unwrap(),
        "--depth",
        "3:100000",
        "--vcf-keep-filtered",
    ]);
    assert!(o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains(
            "Warning: --vcf-keep-filtered does not apply to a VCF read in consensus mode and \
             was ignored"
        ),
        "{stderr}"
    );
    assert!(!stderr.contains("--vcf-sample"), "{stderr}");

    // Consensus mode checks --vcf-sample against its one sample, but
    // selects nothing with it; the warning says so, after the flags it
    // ignores.
    let o = run([
        "qc",
        vcf.to_str().unwrap(),
        "--depth",
        "3:100000",
        "--vcf-keep-filtered",
        "--vcf-sample",
        "aln.sam",
    ]);
    assert!(o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains(
            "Warning: --vcf-keep-filtered does not apply to a VCF read in consensus mode and \
             was ignored; --vcf-sample selects no sample in consensus mode, which reads the \
             INFO fields that bcftools computes over all samples together; it was only checked \
             against the VCF's one sample"
        ),
        "{stderr}"
    );
}
