//! The binary's main outputs on tiny inputs of each format.

mod common;

use common::{TempDir, args, json, read, run_err, run_ok, words};

const PSMCFA: &str = "> chr1\nTTTTTKTTTTKTTTTNTTTTKTTTTKTTTT\n";

const SMALL_VCF: &str = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1
chr1\t24\t.\tA\tC\t.\tPASS\t.\tGT\t1/0
";

#[test]
fn cli_runs_and_writes_json() {
    let dir = TempDir::new("smoke");
    let input = dir.write("in.psmcfa", PSMCFA);
    let out = dir.path("out.json");
    run_ok(args![input, out, "0", "--no-progress"]);
    let v = json(&out);
    for key in ["theta", "rho", "lam", "t_max", "n_steps", "mu"] {
        assert!(v.get(key).is_some(), "{key}");
    }
    assert!(dir.path("out.html").exists(), "no HTML report");
}

#[test]
fn cli_tmrca_writes_tsv_and_html_tmrca_panel() {
    let dir = TempDir::new("smoke_tmrca");
    let input = dir.write("in.psmcfa", PSMCFA);
    let tmrca = dir.path("posterior.tsv");
    run_ok(args![
        input,
        dir.path("out.json"),
        "0",
        "--tmrca-out",
        tmrca,
        "--no-progress"
    ]);
    let text = read(&tmrca);
    assert!(
        text.starts_with(
            "seq_id\tseq_bin\tglobal_bin\tobs\tmap_state\ttmrca_map_years\ttmrca_mean_years\t\
             tmrca_q025_years\ttmrca_q975_years\tpmax\tentropy"
        ),
        "tmrca header is missing"
    );
    assert!(text.lines().count() > 1, "no TMRCA rows");
    assert!(read(dir.path("out.html")).contains("TMRCA Posterior Track"));
}

#[test]
fn cli_bootstrap_writes_replicates_summary_and_html_overlay() {
    let dir = TempDir::new("smoke_boot");
    let input = dir.write(
        "in.psmcfa",
        "> chr1\nTTTTTKTTTTKTTTTNTTTTKTTTTKTTTTTTTTTKTTTTKTTTTNTTTTKTTTTKTTTT\n",
    );
    let boot = dir.path("boot");
    run_ok(args![
        input,
        dir.path("out.json"),
        "1",
        ..words("--bootstrap 2 --bootstrap-block-size 8 --bootstrap-seed 7"),
        "--bootstrap-dir",
        boot,
        "--no-progress",
    ]);
    for name in ["replicate_001.json", "replicate_002.json", "summary.json"] {
        assert!(boot.join(name).exists(), "no {name}");
    }
    assert!(
        read(boot.join("summary.tsv")).starts_with("x_years\tne_main\tne_q025\tne_q500\tne_q975"),
        "bootstrap summary header is missing"
    );
    assert!(read(dir.path("out.html")).contains("Bootstrap q2.5"));
}

#[test]
fn cli_vcf_input_runs_and_writes_json() {
    let dir = TempDir::new("smoke_vcf");
    let input = dir.write("in.vcf", SMALL_VCF);
    let out = dir.path("out.json");
    // VCF input goes through fq2psmcfa's output filter, which drops a
    // sequence with fewer than 10,000 called bases; this 24 bp chromosome
    // needs it off. The old --mhs-bin-size spelling of --bin-size still works.
    run_ok(args![
        input,
        out,
        "0",
        ..words(
            "--input-format vcf --mhs-bin-size 4 --min-good-bases 0 --min-good-fraction 0 --no-progress"
        ),
    ]);
    let v = json(&out);
    for key in ["theta", "rho", "lam"] {
        assert!(v.get(key).is_some(), "{key}");
    }
    assert!(dir.path("out.html").exists(), "no HTML report");
}

#[test]
fn cli_vcf_input_dropped_by_the_output_filter_is_explained() {
    let dir = TempDir::new("smoke_vcf_filtered");
    let input = dir.write("in.vcf", SMALL_VCF);
    let out = dir.path("out.json");
    let err = run_err(args![input, out, "0", "--bin-size", "4", "--no-progress"]);
    assert!(
        err.contains("chr1: 24 bases, 24 called (< 10,000)"),
        "{err}"
    );
    assert!(
        err.contains("--min-good-bases 0 --min-good-fraction 0"),
        "{err}"
    );
    assert!(!out.exists());
}

/// Multihetsep input is recognised from its content, and --mhs-pair picks
/// the two haplotypes: the starting theta follows the pair's het bins.
#[test]
fn cli_reads_multihetsep_with_the_chosen_pair() {
    let dir = TempDir::new("smoke_mhs");
    // Haplotypes 0,1 differ at every row, 2,3 at the first row only.
    let input = dir.write(
        "in.txt",
        "chr1\t150\t150\tACAG\nchr1\t450\t300\tGTGG\nchr1\t850\t400\tCACC\nchr1\t1000\t150\tTGTT\n",
    );
    let out = dir.path("out.json");
    let theta = |extra: &[&str]| -> f64 {
        let _ = std::fs::remove_file(&out);
        run_ok(args![
            input,
            out,
            "0",
            "--min-good-bases",
            "0",
            "--no-progress",
            ..extra
        ]);
        json(&out)["theta"].as_f64().unwrap()
    };
    let auto = theta(&[]);
    assert_eq!(auto, theta(&["--input-format", "mhs"]));
    assert_eq!(auto, theta(&["--mhs-pair", "0,1"]));
    // 10 bins of 100 bp: 4 heterozygous for 0,1, 1 for 2,3.
    assert_eq!(auto, -(1.0f64 - 4.0 / 10.0).ln());
    assert_eq!(theta(&["--mhs-pair", "2,3"]), -(1.0f64 - 1.0 / 10.0).ln());
}

/// Without arguments the program and its subcommands print their help,
/// with the status of a usage error.
#[test]
fn no_arguments_print_the_help() {
    for argv in [&[][..], &["convert"], &["qc"]] {
        let o = common::outcome(argv);
        assert!(o.stdout.is_empty(), "{}", o.stdout);
        let stderr = o.exits(2);
        assert!(stderr.contains("Usage: psmc-rs"), "{stderr}");
        if argv.is_empty() {
            assert!(
                stderr.contains("Usage: psmc-rs [OPTIONS] [INPUT] [OUTPUT] [N_ITER]"),
                "{stderr}"
            );
            assert!(stderr.contains("--bootstrap <N>"), "{stderr}");
        }
    }
}
