//! `psmc-rs convert` through the binary. The conversions must reproduce
//! the golden files the real tools wrote
//! (tests/data/route_a, see its README): `vcfutils.pl vcf2fq | fq2psmcfa`
//! for psmcfa and `generate_multihetsep.py` for multihetsep.

mod common;

use common::{
    Outcome, TempDir, args, command, finding_codes, fixture, gunzipped, json, outcome, read,
    route_a_option_sets, run_err, run_ok, words,
};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

fn route_a(name: &str) -> PathBuf {
    fixture(&format!("route_a/{name}"))
}

#[test]
fn convert_reproduces_vcf2fq_and_fq2psmcfa() {
    let dir = TempDir::new("convert_route_a");
    let sets = route_a_option_sets();
    assert_eq!(sets.len(), 6);
    for set in sets {
        // Plain and gzip output hold the same bytes.
        for ext in ["psmcfa", "psmcfa.gz"] {
            let out = dir.path(&format!("{}.{ext}", set.name));
            run_ok(args!["convert", set.vcf, "--psmcfa", out, ..set.args]);
            assert!(
                gunzipped(&out) == gunzipped(&set.expected),
                "{} ({ext}) differs from fq2psmcfa's output",
                set.name
            );
        }
    }
}

/// `bcftools call -c -A` at 250x: consensus mode is detected although most
/// reference sites have an ALT allele, and the automatic depth bounds match.
#[test]
fn convert_reads_call_c_with_alt_alleles() {
    let dir = TempDir::new("convert_keep_alts");
    let out = dir.path("keep_alts.psmcfa");
    let stderr = outcome(args![
        "convert",
        fixture("route_a_keep_alts/keep_alts.vcf.gz"),
        "--psmcfa",
        out,
        ..words("--bin-size 50 --min-good-bases 1000"),
    ])
    .ok()
    .stderr;
    assert!(stderr.contains("VCF read in consensus mode"), "{stderr}");
    assert!(
        stderr.contains("bases kept at depth 77-463 (auto)"),
        "{stderr}"
    );
    assert_eq!(
        fs::read(&out).unwrap(),
        fs::read(fixture("route_a_keep_alts/expected/keep_alts.psmcfa")).unwrap()
    );
}

#[test]
fn convert_reproduces_generate_multihetsep() {
    let dir = TempDir::new("convert_mhs");
    // The three per-chromosome callable masks as one (multi-member gzip)
    // file: several --vcf-mask files must all contain a site.
    let chroms = ["chr1", "chr2", "chr3"];
    let masks: Vec<u8> = chroms
        .iter()
        .flat_map(|c| fs::read(route_a(&format!("masks/mask_{c}.bed.gz"))).unwrap())
        .collect();
    let mask = dir.write("mask.bed.gz", masks);
    for vcf in ["snps", "snps_phased"] {
        let all = dir.path(&format!("{vcf}.all.multihetsep.gz"));
        run_ok(args![
            "convert",
            route_a(&format!("{vcf}.vcf.gz")),
            "--vcf-mask",
            mask,
            // generate_multihetsep.py has no output filter.
            ..words("--min-good-bases 0 --min-good-fraction 0"),
            "--multihetsep-per-chrom",
            dir.path(vcf),
            "--multihetsep",
            all,
        ]);
        let mut concatenated = Vec::new();
        for chrom in chroms {
            let ours = fs::read(dir.path(&format!("{vcf}.{chrom}.multihetsep"))).unwrap();
            let expected =
                fs::read(route_a(&format!("expected/{vcf}.{chrom}.multihetsep"))).unwrap();
            assert!(
                ours == expected,
                "{vcf} {chrom}:\nours:\n{}\nexpected:\n{}",
                String::from_utf8_lossy(&ours),
                String::from_utf8_lossy(&expected)
            );
            concatenated.extend(expected);
        }
        let sites = concatenated.iter().filter(|&&b| b == b'\n').count();
        assert!(sites >= 30, "{vcf}: only {sites} sites compared");
        // nr_called restarts with each chromosome, so the single file is
        // the per-chromosome files one after the other.
        assert!(gunzipped(&all) == concatenated, "{vcf}: --multihetsep");
    }
}

#[test]
fn convert_normalises_psmcfa() {
    let dir = TempDir::new("convert_psmcfa");
    let bins = "tTkKNn".repeat(15);
    let input = dir.write(
        "in.psmcfa",
        format!(">a desc\n{bins}\n>empty\n\n>b\nTTKT\r\n"),
    );
    let out = dir.path("out.psmcfa");
    run_ok(args!["convert", input, "--psmcfa", out]);
    let upper = "TTKKNN".repeat(15);
    assert_eq!(
        read(&out),
        format!(">a\n{}\n{}\n>b\nTTKT\n", &upper[..60], &upper[60..])
    );
}

/// Run convert on `bytes` written to its stdin through a pipe.
fn convert_piped(bytes: Vec<u8>, args: &[&str]) -> Outcome {
    let mut child = command()
        .args(["convert", "/dev/stdin"])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        // The reader may stop early, closing the pipe.
        let _ = stdin.write_all(&bytes);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    out.into()
}

/// Multihetsep output of psmcfa input is a usage error (exit status 2)
/// whether the format is given or recognised, found before the input is
/// read: the rest of this file, which cannot be read, is never reached.
/// From a pipe, whose format is known only once read, it exits 2 too.
#[test]
fn psmcfa_has_no_sites_for_multihetsep() {
    let dir = TempDir::new("convert_no_sites");
    // The second record ends inside a FASTQ quality block, which the
    // loader refuses.
    let input = dir.write("in.psmcfa", ">a\nTTKTTT\n>b\nTT+\n");
    let out = dir.path("out.mhs");
    let to_mhs = ["--multihetsep", out.to_str().unwrap()];
    for extra in [&[][..], &["--input-format", "psmcfa"]] {
        let err = outcome(args!["convert", input, ..to_mhs, ..extra]).exits(2);
        assert!(
            err.contains("psmcfa input has no sites"),
            "{extra:?}: {err}"
        );
        assert!(!out.exists());
    }
    let err = convert_piped(b">a\nTTKTTT\n".to_vec(), &to_mhs).exits(2);
    assert!(err.contains("psmcfa input has no sites"), "{err}");
    assert!(!out.exists());
    let err = run_err(args!["convert", input]);
    assert!(err.contains("--psmcfa"), "{err}");
}

/// With --qc-json -, stdout holds the QC JSON alone; the result lines go to
/// stderr.
#[test]
fn qc_json_on_stdout_is_the_only_stdout() {
    let dir = TempDir::new("convert_qc_stdout");
    let input = dir.write("in.psmcfa", format!(">a\n{}\n", "TTTTKTTTTN".repeat(30)));
    let out = dir.path("out.psmcfa");
    let o = outcome(args!["convert", input, "--psmcfa", out, "--qc-json", "-"]).ok();
    let qc: serde_json::Value = serde_json::from_str(&o.stdout).unwrap();
    assert_eq!(qc["n_sequences"], 1);
    assert!(
        o.stderr.contains(&format!(
            "psmcfa: {} (1 sequences, 300 bins)",
            out.display()
        )),
        "{}",
        o.stderr
    );
    assert!(!o.stderr.contains("QC report: -"), "{}", o.stderr);
}

/// Outputs that cannot be written fail before the input is read.
#[test]
fn output_paths_are_checked_before_reading() {
    let dir = TempDir::new("convert_paths");
    let input = dir.write("in.psmcfa", ">a\nTTKT\n");
    let err = run_err(args![
        "convert",
        input,
        "--psmcfa",
        dir.path("nowhere/out.psmcfa")
    ]);
    assert!(err.contains("cannot write --psmcfa"), "{err}");
    assert!(!err.contains("Input:"), "{err}");
    let err = run_err(args!["convert", input, "--psmcfa", input]);
    assert!(err.contains("is the input file"), "{err}");
    assert_eq!(read(&input), ">a\nTTKT\n");
    let err = run_err(args![
        "convert",
        route_a("allsites.vcf.gz"),
        "--multihetsep-per-chrom",
        dir.path("nowhere/snps"),
    ]);
    assert!(
        err.contains("cannot write --multihetsep-per-chrom"),
        "{err}"
    );
}

/// The files of --multihetsep-per-chrom are named after the sequences, so
/// they are checked once the input is read, before anything is written:
/// none may be the input (as when a pair is taken from MSMC-style
/// per-chromosome files under their own prefix) or another output.
#[test]
fn per_chromosome_files_are_checked_before_writing() {
    let dir = TempDir::new("per_chrom_names");
    // Four haplotypes; --mhs-pair 2,3 takes the second sample.
    let text = "chr1\t1000\t1000\tACAC\nchr1\t2500\t1500\tAAGT\n\
                chr1\t3100\t600\tCCCG\nchr1\t5000\t1900\tTATA\n";
    let input = dir.write("sample.chr1.multihetsep", text);
    let convert = |extra: &[&str]| {
        outcome(args![
            "convert",
            input,
            ..words("--mhs-pair 2,3 --min-good-bases 0 --min-good-fraction 0"),
            ..extra
        ])
    };
    let prefix = dir.path("sample");
    let err = convert(&["--multihetsep-per-chrom", prefix.to_str().unwrap()]).exits(2);
    assert!(
        err.contains(&format!(
            "--multihetsep-per-chrom {} is the input file",
            input.display()
        )),
        "{err}"
    );
    assert_eq!(read(&input), text);

    // The per-chromosome file would replace another output, or the QC JSON,
    // which is written first: nothing is written at all.
    let pair = dir.path("pair");
    let pair_chr1 = dir.path("pair.chr1.multihetsep");
    for flag in ["--multihetsep", "--qc-json", "--psmcfa"] {
        let err = convert(&[
            flag,
            pair_chr1.to_str().unwrap(),
            "--multihetsep-per-chrom",
            pair.to_str().unwrap(),
        ])
        .exits(2);
        assert!(err.contains("are the same file"), "{flag}: {err}");
        assert!(!pair_chr1.exists(), "{flag}");
    }
    let ok = convert(&["--multihetsep-per-chrom", pair.to_str().unwrap()])
        .ok()
        .stdout;
    assert!(ok.contains(&format!("multihetsep: {}", pair_chr1.display())));
    assert!(read(&pair_chr1).contains("\tGT\n"));
}

/// Multihetsep can be binned, with a note that missing sites were placed by
/// assumption.
#[test]
fn multihetsep_converts_to_psmcfa_with_a_note() {
    let dir = TempDir::new("convert_mhs_psmcfa");
    let input = dir.write(
        "in.mhs",
        "chr1\t150\t150\tAC\nchr1\t450\t100\tGT\nchr1\t850\t400\tCA\n",
    );
    let out = dir.path("out.psmcfa");
    let stderr = outcome(args![
        "convert",
        input,
        "--psmcfa",
        out,
        "--min-good-bases",
        "0"
    ])
    .ok()
    .stderr;
    assert!(
        stderr.contains("placed at the start of each interval"),
        "{stderr}"
    );
    assert!(stderr.contains("APPROX_MISSING_PLACEMENT"), "{stderr}");
    // The 200 uncalled sites before the second row go to 151-350, which
    // leaves bin 3 (201-300) missing.
    assert_eq!(read(&out), ">chr1\nTKNTKTTTK\n");
}

/// Like fq2psmcfa, convert writes an empty file when every sequence is
/// filtered out, and says so; the QC report says why.
#[test]
fn convert_writes_an_empty_file_when_everything_is_dropped() {
    let dir = TempDir::new("convert_empty");
    let out = dir.path("out.psmcfa");
    let qc_json = dir.path("qc.json");
    let stderr = outcome(args![
        "convert",
        route_a("allsites.vcf.gz"),
        ..words("--depth 3:100000 --min-good-bases 1000000"),
        "--psmcfa",
        out,
        "--qc-json",
        qc_json,
    ])
    .ok()
    .stderr;
    assert!(stderr.contains("no sequence is left to write"), "{stderr}");
    assert!(
        stderr.contains("Input QC (vcf, consensus mode): 0 sequences"),
        "{stderr}"
    );
    assert_eq!(fs::read(&out).unwrap(), b"");
    let qc = json(&qc_json);
    assert_eq!(qc["n_sequences"], 0);
    let codes = finding_codes(&qc);
    assert!(
        codes.contains(&"DROPPED_SEQUENCES".to_string()),
        "{codes:?}"
    );
}

/// A pipe can be read once, so the automatic depth bounds (two passes)
/// cannot read it; the error names the option that can.
#[test]
fn a_piped_vcf_needs_explicit_depth_bounds() {
    let dir = TempDir::new("convert_pipe");
    let vcf = fs::read(route_a("allsites.vcf.gz")).unwrap();
    let out = dir.path("out.psmcfa");
    let to_psmcfa = ["--psmcfa", out.to_str().unwrap()];
    let o = convert_piped(vcf.clone(), &to_psmcfa);
    assert!(!o.success());
    assert!(
        o.stderr.contains("Give them as --depth MIN:MAX"),
        "{}",
        o.stderr
    );

    convert_piped(vcf, &[&to_psmcfa[..], &["--depth", "3:100000"]].concat()).ok();
    assert!(gunzipped(&out) == gunzipped(route_a("expected/default.psmcfa")));
}
