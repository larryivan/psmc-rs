//! `psmc-rs qc` through the binary: the report of an input read as
//! inference and `convert` read it, and the exit status of --strict.

mod common;

use common::{TempDir, args, finding_codes, fixture, json, outcome, read, run_ok};
use serde_json::Value;

/// `qc` reads the input as convert and inference do, and --strict turns its
/// warnings into exit status 2.
#[test]
fn qc_reports_and_fails_strictly_on_warnings() {
    let dir = TempDir::new("qc");
    let vcf = fixture("route_a/allsites.vcf.gz");
    let json_path = dir.path("qc.json");
    let opts = ["--depth", "3:100000", "--min-good-bases", "1000"];
    let stdout = run_ok(args!["qc", vcf, "--json", json_path, ..opts]);
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
    let codes = finding_codes(&report);
    assert!(codes.contains(&"TOO_SHORT".to_string()), "{codes:?}");

    // The same report from convert.
    let convert_json = dir.path("convert_qc.json");
    run_ok(args![
        "convert",
        vcf,
        "--psmcfa",
        dir.path("x.psmcfa"),
        "--qc-json",
        convert_json,
        ..opts
    ]);
    assert_eq!(json(&convert_json), report);

    // --strict: status 2 with warnings, 0 without.
    let o = outcome(args!["qc", vcf, "--strict", ..opts]);
    assert_eq!(o.code, Some(2));
    assert!(o.stdout.contains("WARNING"));

    // --json - writes the JSON on stdout and the text on stderr.
    let o = outcome(args!["qc", vcf, "--json", "-", "--depth", "3:100000"]).ok();
    let parsed: Value = serde_json::from_str(&o.stdout).unwrap();
    assert_eq!(parsed["n_sequences"], 1);
    assert!(o.stderr.contains("Input QC"));
}

#[test]
fn qc_without_warnings_passes_strict() {
    let dir = TempDir::new("qc_clean");
    // 120 Mb of human-like data in one sequence: no warning applies.
    let unit = format!("{}K", "T".repeat(199));
    let input = dir.write("clean.psmcfa", format!(">chr1\n{}\n", unit.repeat(6000)));
    let o = outcome(args!["qc", input, "--strict"]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);
    assert!(!o.stdout.contains("WARNING"), "{}", o.stdout);
}

/// A start whose transition probabilities are negative is named by QC:
/// `qc` checks the start of a fit with the default settings, inference the
/// start of its own settings (here `--tr-ratio 1`, at which nothing else
/// warns about 17% heterozygous bins).
#[test]
fn an_invalid_start_is_reported() {
    let dir = TempDir::new("qc_invalid_start");
    // 120 Mb of one repeated unit, with K bins at a given share.
    let write = |name: &str, unit: &str| {
        dir.write(
            name,
            format!(">chr1\n{}\n", unit.repeat(1_200_000 / unit.len())),
        )
    };
    let k60 = write("k60.psmcfa", "KKKKKKTTTT");
    let o = outcome(args!["qc", k60, "--json", "-"]);
    let report: Value = serde_json::from_str(&o.stdout).unwrap();
    assert_eq!(report["findings"][0]["code"], "INVALID_START");
    let message = report["findings"][0]["message"].as_str().unwrap();
    assert!(
        message.starts_with("A fit with psmc-rs's default settings would start from theta0 0.916"),
        "{message}"
    );
    assert!(o.stderr.contains("WARNING  INVALID_START"), "{}", o.stderr);

    let k17 = write(
        "k17.psmcfa",
        &format!("{}{}", "K".repeat(17), "T".repeat(83)),
    );
    let o = outcome(args!["qc", k17, "--strict"]);
    assert_eq!(o.code, Some(0), "{}", o.stdout);

    // Inference checks its own start. The QC JSON is written before the
    // fit, whatever the fit then does.
    let qc_codes = |name: &str, extra: &[&str]| {
        let path = dir.path(&format!("{name}.qc.json"));
        let fit = dir.path(&format!("{name}.json"));
        outcome(args![
            k17,
            fit,
            "0",
            "--no-progress",
            "--qc-json",
            path,
            ..extra
        ]);
        let report = json(&path);
        (finding_codes(&report), report)
    };
    let (codes, r1) = qc_codes("r1", &["--tr-ratio", "1"]);
    assert_eq!(codes, ["INVALID_START"]);
    assert!(
        r1["findings"][0]["message"]
            .as_str()
            .unwrap()
            .starts_with("The fit starts from theta0 0.186 and rho0 0.186 per bin"),
        "{r1}"
    );
    assert!(qc_codes("r4", &[]).0.is_empty());
    assert!(
        qc_codes("rho", &["--rho0", "0.2"])
            .0
            .contains(&"INVALID_START".to_string())
    );
}

/// --json is checked as inference and convert check their outputs, before
/// the input is read: it may not be the input, and its directory must
/// exist.
#[test]
fn the_json_path_is_checked_before_reading() {
    let dir = TempDir::new("qc_json_path");
    let text = format!(">chr1\n{}\n", "TTTTKTTTTN".repeat(200));
    let input = dir.write("in.psmcfa", &text);
    let o = outcome(args!["qc", input, "--json", input]);
    assert!(o.stdout.is_empty());
    let stderr = o.exits(2);
    assert!(stderr.contains("--json"), "{stderr}");
    assert!(stderr.contains("is the input file"), "{stderr}");
    assert_eq!(read(&input), text);

    let o = outcome(args!["qc", input, "--json", dir.path("nowhere/qc.json")]);
    // Nothing was read: no report and no description of the input.
    assert!(o.stdout.is_empty());
    let stderr = o.exits(1);
    assert!(stderr.contains("cannot write --json"), "{stderr}");
    assert!(!stderr.contains("psmcfa:"), "{stderr}");
}

/// Options of the other VCF mode are named in a warning.
#[test]
fn options_of_the_other_vcf_mode_are_reported() {
    let vcf = fixture("route_a/allsites.vcf.gz");
    let stderr = |extra: &[&str]| {
        outcome(args![
            "qc",
            vcf,
            "--depth",
            "3:100000",
            "--vcf-keep-filtered",
            ..extra
        ])
        .ok()
        .stderr
    };
    let err = stderr(&[]);
    assert!(
        err.contains(
            "Warning: --vcf-keep-filtered does not apply to a VCF read in consensus mode and \
             was ignored"
        ),
        "{err}"
    );
    assert!(!err.contains("--vcf-sample"), "{err}");

    // Consensus mode checks --vcf-sample against its one sample, but
    // selects nothing with it; the warning says so, after the flags it
    // ignores.
    let err = stderr(&["--vcf-sample", "aln.sam"]);
    assert!(
        err.contains(
            "Warning: --vcf-keep-filtered does not apply to a VCF read in consensus mode and \
             was ignored; --vcf-sample selects no sample in consensus mode, which reads the \
             INFO fields that bcftools computes over all samples together; it was only checked \
             against the VCF's one sample"
        ),
        "{err}"
    );
}

/// A gVCF read without a mask gets the gVCF wording of
/// VARIANTS_ONLY_NO_MASK: positions no block covers are missing there.
#[test]
fn a_gvcf_without_a_mask_is_reported_as_a_gvcf() {
    let dir = TempDir::new("qc_gvcf");
    let vcf = dir.write(
        "sample.g.vcf",
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         chr1\t1\t.\tG\t.\t.\t.\tEND=298;MIN_DP=1\tGT:DP\t0/0:1\n\
         chr1\t299\t.\tA\tG\t50\t.\tDP=5\tGT:DP\t0/1:5\n\
         chr1\t600\t.\tA\t.\t.\t.\tEND=899;MIN_DP=1\tGT:DP\t0/0:1\n",
    );
    let json_path = dir.path("qc.json");
    run_ok(args![
        "qc",
        vcf,
        "--bin-size",
        "10",
        "--min-good-bases",
        "0",
        "--min-good-fraction",
        "0",
        "--json",
        json_path,
    ]);
    let report = json(&json_path);
    let finding = report["findings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["code"] == "VARIANTS_ONLY_NO_MASK")
        .expect("VARIANTS_ONLY_NO_MASK");
    let message = finding["message"].as_str().unwrap();
    assert!(
        message.starts_with("The gVCF was read without a callable-sites mask"),
        "{message}"
    );
}
