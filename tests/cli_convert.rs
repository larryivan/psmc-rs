//! `psmc-rs convert` through the binary. The conversions must reproduce
//! the golden files the real tools wrote
//! (tests/data/route_a, see its README): `vcfutils.pl vcf2fq | fq2psmcfa`
//! for psmcfa and `generate_multihetsep.py` for multihetsep.

mod common;

use common::{TempDir, fixture, run, run_err, run_ok};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// File contents, gunzipped when the name ends in `.gz`.
fn read_bytes(path: &Path) -> Vec<u8> {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if !path.to_string_lossy().ends_with(".gz") {
        return raw;
    }
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(raw.as_slice())
        .read_to_end(&mut out)
        .unwrap();
    out
}

fn route_a(name: &str) -> PathBuf {
    fixture(&format!("route_a/{name}"))
}

/// The tool options of one row of expected/option_sets.tsv.
fn flags(text: &str, tool: &str) -> HashMap<char, String> {
    let rest = text.strip_prefix(tool).expect("tool name");
    let words: Vec<&str> = rest.split_whitespace().collect();
    words
        .chunks(2)
        .map(|p| {
            (
                p[0].trim_start_matches('-').chars().next().unwrap(),
                p[1].to_string(),
            )
        })
        .collect()
}

/// The convert arguments equivalent to a row's vcf2fq and fq2psmcfa options
/// (their defaults where it gives none); the `auto` row lets psmc-rs choose
/// the depth bounds.
fn convert_args(
    name: &str,
    vcf2fq: &HashMap<char, String>,
    fq2psmcfa: &HashMap<char, String>,
) -> Vec<String> {
    let get = |m: &HashMap<char, String>, k: char, default: &str| {
        m.get(&k).cloned().unwrap_or_else(|| default.to_string())
    };
    let depth = if name == "auto" {
        "auto".to_string()
    } else {
        format!("{}:{}", get(vcf2fq, 'd', "3"), get(vcf2fq, 'D', "100000"))
    };
    vec![
        "--depth".to_string(),
        depth,
        "--min-mapq".to_string(),
        get(vcf2fq, 'Q', "10"),
        "--indel-window".to_string(),
        get(vcf2fq, 'l', "5"),
        "--min-qual".to_string(),
        get(fq2psmcfa, 'q', "10"),
        "--min-good-bases".to_string(),
        get(fq2psmcfa, 'g', "10000"),
        "--bin-size".to_string(),
        get(fq2psmcfa, 's', "100"),
    ]
}

#[test]
fn convert_reproduces_vcf2fq_and_fq2psmcfa() {
    let dir = TempDir::new("convert_route_a");
    let table = fs::read_to_string(route_a("expected/option_sets.tsv")).unwrap();
    let mut n = 0;
    for line in table.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        let (name, input) = (cols[0], cols[3]);
        let args = convert_args(
            name,
            &flags(cols[1], "vcf2fq"),
            &flags(cols[2], "fq2psmcfa"),
        );
        let expected = ["psmcfa", "psmcfa.gz"]
            .iter()
            .map(|ext| route_a(&format!("expected/{name}.{ext}")))
            .find(|p| p.exists())
            .unwrap();
        // Plain and gzip output hold the same bytes.
        for ext in ["psmcfa", "psmcfa.gz"] {
            let out = dir.path(&format!("{name}.{ext}"));
            let mut all = vec![
                "convert".to_string(),
                route_a(input).display().to_string(),
                "--psmcfa".to_string(),
                out.display().to_string(),
            ];
            all.extend(args.iter().cloned());
            run_ok(&all);
            assert!(
                read_bytes(&out) == read_bytes(&expected),
                "{name} ({ext}) differs from fq2psmcfa's output"
            );
        }
        n += 1;
    }
    assert_eq!(n, 6);
}

/// `bcftools call -c -A` at 250x: consensus mode is detected although most
/// reference sites have an ALT allele, and the automatic depth bounds match.
#[test]
fn convert_reads_call_c_with_alt_alleles() {
    let dir = TempDir::new("convert_keep_alts");
    let out = dir.path("keep_alts.psmcfa");
    let stderr_of = |args: &[&str]| {
        let o = run(args);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8(o.stderr).unwrap()
    };
    let stderr = stderr_of(&[
        "convert",
        fixture("route_a_keep_alts/keep_alts.vcf.gz")
            .to_str()
            .unwrap(),
        "--psmcfa",
        out.to_str().unwrap(),
        "--bin-size",
        "50",
        "--min-good-bases",
        "1000",
    ]);
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

/// The three per-chromosome callable masks as one (multi-member gzip) file:
/// several --vcf-mask files must all contain a site.
fn combined_mask(dir: &TempDir) -> PathBuf {
    let mut bytes = Vec::new();
    for chrom in ["chr1", "chr2", "chr3"] {
        bytes.extend(fs::read(route_a(&format!("masks/mask_{chrom}.bed.gz"))).unwrap());
    }
    let path = dir.path("mask.bed.gz");
    fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn convert_reproduces_generate_multihetsep() {
    let dir = TempDir::new("convert_mhs");
    let mask = combined_mask(&dir);
    for vcf in ["snps", "snps_phased"] {
        let prefix = dir.path(vcf);
        let all = dir.path(&format!("{vcf}.all.multihetsep.gz"));
        run_ok([
            "convert",
            route_a(&format!("{vcf}.vcf.gz")).to_str().unwrap(),
            "--vcf-mask",
            mask.to_str().unwrap(),
            // generate_multihetsep.py has no output filter.
            "--min-good-bases",
            "0",
            "--min-good-fraction",
            "0",
            "--multihetsep-per-chrom",
            prefix.to_str().unwrap(),
            "--multihetsep",
            all.to_str().unwrap(),
        ]);
        let mut concatenated = Vec::new();
        for chrom in ["chr1", "chr2", "chr3"] {
            let ours = read_bytes(&dir.path(&format!("{vcf}.{chrom}.multihetsep")));
            let expected = read_bytes(&route_a(&format!("expected/{vcf}.{chrom}.multihetsep")));
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
        assert!(read_bytes(&all) == concatenated, "{vcf}: --multihetsep");
    }
}

#[test]
fn convert_normalises_psmcfa() {
    let dir = TempDir::new("convert_psmcfa");
    let input = dir.path("in.psmcfa");
    let bins = "tTkKNn".repeat(15);
    fs::write(&input, format!(">a desc\n{bins}\n>empty\n\n>b\nTTKT\r\n")).unwrap();
    let out = dir.path("out.psmcfa");
    run_ok([
        "convert",
        input.to_str().unwrap(),
        "--psmcfa",
        out.to_str().unwrap(),
    ]);
    let expected = format!(
        ">a\n{}\n{}\n>b\nTTKT\n",
        &"TTKKNN".repeat(15)[..60],
        &"TTKKNN".repeat(15)[60..]
    );
    assert_eq!(fs::read_to_string(&out).unwrap(), expected);
}

/// Multihetsep output of psmcfa input is a usage error (exit status 2)
/// whether the format is given or recognised, found before the input is
/// read: the rest of this file, which cannot be read, is never reached.
/// From a pipe, whose format is known only once read, it exits 2 too.
#[test]
fn psmcfa_has_no_sites_for_multihetsep() {
    let dir = TempDir::new("convert_no_sites");
    let input = dir.path("in.psmcfa");
    // The second record ends inside a FASTQ quality block, which the
    // loader refuses.
    let text = ">a\nTTKTTT\n>b\nTT+\n";
    fs::write(&input, text).unwrap();
    let out = dir.path("out.mhs");
    for extra in [&[][..], &["--input-format", "psmcfa"][..]] {
        let mut args = vec![
            "convert",
            input.to_str().unwrap(),
            "--multihetsep",
            out.to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let o = run(&args);
        let err = String::from_utf8(o.stderr).unwrap();
        assert_eq!(o.status.code(), Some(2), "{extra:?}: {err}");
        assert!(err.contains("psmcfa input has no sites"), "{err}");
        assert!(!out.exists());
    }
    let o = convert_piped(
        b">a\nTTKTTT\n".to_vec(),
        &["--multihetsep", out.to_str().unwrap()],
    );
    let err = String::from_utf8(o.stderr).unwrap();
    assert_eq!(o.status.code(), Some(2), "{err}");
    assert!(err.contains("psmcfa input has no sites"), "{err}");
    assert!(!out.exists());
    let err = run_err(["convert", input.to_str().unwrap()]);
    assert!(err.contains("--psmcfa"), "{err}");
}

/// With --qc-json -, stdout holds the QC JSON alone; the result lines go to
/// stderr.
#[test]
fn qc_json_on_stdout_is_the_only_stdout() {
    let dir = TempDir::new("convert_qc_stdout");
    let input = dir.path("in.psmcfa");
    fs::write(&input, format!(">a\n{}\n", "TTTTKTTTTN".repeat(30))).unwrap();
    let out = dir.path("out.psmcfa");
    let o = run([
        "convert",
        input.to_str().unwrap(),
        "--psmcfa",
        out.to_str().unwrap(),
        "--qc-json",
        "-",
    ]);
    assert!(o.status.success());
    let qc: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(qc["n_sequences"], 1);
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains(&format!(
            "psmcfa: {} (1 sequences, 300 bins)",
            out.display()
        )),
        "{stderr}"
    );
    assert!(!stderr.contains("QC report: -"), "{stderr}");
}

/// Outputs that cannot be written fail before the input is read.
#[test]
fn output_paths_are_checked_before_reading() {
    let dir = TempDir::new("convert_paths");
    let input = dir.path("in.psmcfa");
    fs::write(&input, ">a\nTTKT\n").unwrap();
    let missing = dir.path("nowhere/out.psmcfa");
    let err = run_err([
        "convert",
        input.to_str().unwrap(),
        "--psmcfa",
        missing.to_str().unwrap(),
    ]);
    assert!(err.contains("cannot write --psmcfa"), "{err}");
    assert!(!err.contains("Input:"), "{err}");
    let err = run_err([
        "convert",
        input.to_str().unwrap(),
        "--psmcfa",
        input.to_str().unwrap(),
    ]);
    assert!(err.contains("is the input file"), "{err}");
    assert_eq!(fs::read_to_string(&input).unwrap(), ">a\nTTKT\n");
    let err = run_err([
        "convert",
        route_a("allsites.vcf.gz").to_str().unwrap(),
        "--multihetsep-per-chrom",
        dir.path("nowhere/snps").to_str().unwrap(),
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
    let input = dir.path("sample.chr1.multihetsep");
    let text = "chr1\t1000\t1000\tACAC\nchr1\t2500\t1500\tAAGT\n\
                chr1\t3100\t600\tCCCG\nchr1\t5000\t1900\tTATA\n";
    fs::write(&input, text).unwrap();
    let convert = |extra: &[&str]| {
        let mut args = vec![
            "convert",
            input.to_str().unwrap(),
            "--mhs-pair",
            "2,3",
            "--min-good-bases",
            "0",
            "--min-good-fraction",
            "0",
        ];
        args.extend_from_slice(extra);
        let o = run(&args);
        assert_eq!(o.status.code(), Some(2), "{extra:?}");
        String::from_utf8(o.stderr).unwrap()
    };
    let prefix = dir.path("sample");
    let err = convert(&["--multihetsep-per-chrom", prefix.to_str().unwrap()]);
    assert!(
        err.contains(&format!(
            "--multihetsep-per-chrom {} is the input file",
            input.display()
        )),
        "{err}"
    );
    assert_eq!(fs::read_to_string(&input).unwrap(), text);

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
        ]);
        assert!(err.contains("are the same file"), "{flag}: {err}");
        assert!(!pair_chr1.exists(), "{flag}");
    }
    let ok = run_ok([
        "convert",
        input.to_str().unwrap(),
        "--mhs-pair",
        "2,3",
        "--min-good-bases",
        "0",
        "--min-good-fraction",
        "0",
        "--multihetsep-per-chrom",
        pair.to_str().unwrap(),
    ]);
    assert!(ok.contains(&format!("multihetsep: {}", pair_chr1.display())));
    assert!(fs::read_to_string(&pair_chr1).unwrap().contains("\tGT\n"));
}

/// Multihetsep can be binned, with a note that missing sites were placed by
/// assumption.
#[test]
fn multihetsep_converts_to_psmcfa_with_a_note() {
    let dir = TempDir::new("convert_mhs_psmcfa");
    let input = dir.path("in.mhs");
    fs::write(
        &input,
        "chr1\t150\t150\tAC\nchr1\t450\t100\tGT\nchr1\t850\t400\tCA\n",
    )
    .unwrap();
    let out = dir.path("out.psmcfa");
    let o = run([
        "convert",
        input.to_str().unwrap(),
        "--psmcfa",
        out.to_str().unwrap(),
        "--min-good-bases",
        "0",
    ]);
    assert!(o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains("placed at the start of each interval"),
        "{stderr}"
    );
    assert!(stderr.contains("APPROX_MISSING_PLACEMENT"), "{stderr}");
    // The 200 uncalled sites before the second row go to 151-350, which
    // leaves bin 3 (201-300) missing.
    assert_eq!(fs::read_to_string(&out).unwrap(), ">chr1\nTKNTKTTTK\n");
}

/// Like fq2psmcfa, convert writes an empty file when every sequence is
/// filtered out, and says so; the QC report says why.
#[test]
fn convert_writes_an_empty_file_when_everything_is_dropped() {
    let dir = TempDir::new("convert_empty");
    let out = dir.path("out.psmcfa");
    let qc_json = dir.path("qc.json");
    let o = run([
        "convert",
        route_a("allsites.vcf.gz").to_str().unwrap(),
        "--depth",
        "3:100000",
        "--min-good-bases",
        "1000000",
        "--psmcfa",
        out.to_str().unwrap(),
        "--qc-json",
        qc_json.to_str().unwrap(),
    ]);
    assert!(o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(stderr.contains("no sequence is left to write"), "{stderr}");
    assert!(
        stderr.contains("Input QC (vcf, consensus mode): 0 sequences"),
        "{stderr}"
    );
    assert_eq!(fs::read(&out).unwrap(), b"");
    let qc = common::json(&qc_json);
    assert_eq!(qc["n_sequences"], 0);
    let codes: Vec<&str> = qc["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["code"].as_str().unwrap())
        .collect();
    assert!(codes.contains(&"DROPPED_SEQUENCES"), "{codes:?}");
}

/// Run convert on `bytes` written to its stdin through a pipe.
fn convert_piped(bytes: Vec<u8>, args: &[&str]) -> std::process::Output {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(common::bin())
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
    out
}

/// A pipe can be read once, so the automatic depth bounds (two passes)
/// cannot read it; the error names the option that can.
#[test]
fn a_piped_vcf_needs_explicit_depth_bounds() {
    let dir = TempDir::new("convert_pipe");
    let vcf = fs::read(route_a("allsites.vcf.gz")).unwrap();
    let out = dir.path("out.psmcfa");
    let o = convert_piped(vcf.clone(), &["--psmcfa", out.to_str().unwrap()]);
    assert!(!o.status.success());
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(stderr.contains("Give them as --depth MIN:MAX"), "{stderr}");

    let o = convert_piped(
        vcf,
        &["--psmcfa", out.to_str().unwrap(), "--depth", "3:100000"],
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(read_bytes(&out) == read_bytes(&route_a("expected/default.psmcfa")));
}
