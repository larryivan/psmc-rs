//! `--bin-callable` end to end:
//!
//! - `ignore` on a VCF fits exactly, to the bit, what fq2psmcfa's psmcfa of
//!   the same calls fits (the route A fixtures of `tests/data/route_a`);
//! - `auto` on complete data fits exactly what `ignore` and the psmcfa fit;
//! - `auto` on data with partly callable bins fits the callable-length
//!   model, records it in the JSON and the `.psmc` output, shows it in the
//!   report, and `convert` says what a psmcfa of such data loses;
//! - `auto` on multihetsep, whose counts are approximate, fits as `ignore`
//!   does, to the bit, and `approximate` fits the counts;
//! - the counts reach every pass of a counted run, not only EM: the TMRCA
//!   decoding, the log-likelihood of a run of 0 iterations and every
//!   bootstrap replicate are the library's with the counts.

mod common;

use common::{TempDir, fixture, json, run_ok};
use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, observations_to_sequences, run_bootstrap_sequences,
};
use psmc_rs::hmm::{
    EStepOptions, TmrcaOptions, log_likelihood_with_options, write_tmrca_posterior_tsv_with_options,
};
use psmc_rs::io::input::{InputOptions, load};
use psmc_rs::io::params::load_params;
use psmc_rs::io::{CallableBases, Sequences};
use psmc_rs::opt::MStepConfig;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::Path;

/// The fitted model and the log-likelihoods of a run's JSON, as bits.
fn fit_bits(v: &Value) -> Vec<u64> {
    let num = |x: &Value| x.as_f64().unwrap().to_bits();
    let mut out: Vec<u64> = ["theta", "rho", "t_max"]
        .iter()
        .map(|k| num(&v[k]))
        .collect();
    out.extend(v["lam"].as_array().unwrap().iter().map(num));
    out.extend(v["loglike"].as_array().unwrap().iter().map(num));
    out.push(v["n_iter"].as_u64().unwrap());
    out
}

/// The rounds of a `.psmc` file: everything from the first `RD` on (the
/// header records the input, which differs between formats).
fn rounds(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap();
    let first = text.find("\nRD\t").or_else(|| text.find("\nIT\t")).unwrap();
    text[first..].to_string()
}

/// The `MM` line psmc-rs adds for input with callable bases, if any.
fn bin_line(path: &Path) -> Option<String> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .find(|l| l.starts_with("MM\tbin_model:"))
        .map(str::to_string)
}

/// Fit `input` with `args`, 3 EM iterations; returns its JSON and the path
/// of its `.psmc` output.
fn fit(dir: &TempDir, name: &str, input: &Path, args: &[&str]) -> (Value, std::path::PathBuf) {
    let out = dir.path(&format!("{name}.json"));
    let psmc = dir.path(&format!("{name}.psmc"));
    let mut argv = vec![
        input.to_str().unwrap().to_string(),
        out.to_str().unwrap().to_string(),
        "3".to_string(),
        "--psmc-out".to_string(),
        psmc.to_str().unwrap().to_string(),
        "--no-progress".to_string(),
        "--quiet".to_string(),
    ];
    argv.extend(args.iter().map(|a| a.to_string()));
    run_ok(&argv);
    (json(&out), psmc)
}

/// The command-line options of a row of `route_a/expected/option_sets.tsv`
/// that read the VCF as `vcfutils.pl vcf2fq | fq2psmcfa` did.
fn route_a_args(vcf2fq: &str, fq2psmcfa: &str, auto_depth: bool) -> Vec<String> {
    let flags = |text: &str, tool: &str| -> Vec<(char, String)> {
        let words: Vec<&str> = text
            .strip_prefix(tool)
            .unwrap()
            .split_whitespace()
            .collect();
        words
            .chunks(2)
            .map(|p| (p[0].chars().nth(1).unwrap(), p[1].to_string()))
            .collect()
    };
    let (v, f) = (flags(vcf2fq, "vcf2fq"), flags(fq2psmcfa, "fq2psmcfa"));
    let get = |flags: &[(char, String)], key: char, default: &str| {
        flags
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(default.to_string(), |(_, v)| v.clone())
    };
    let depth = if auto_depth {
        "auto".to_string()
    } else {
        format!("{}:{}", get(&v, 'd', "3"), get(&v, 'D', "100000"))
    };
    let mut args = vec!["--depth".to_string(), depth];
    for (flag, value) in [
        ("--min-mapq", get(&v, 'Q', "10")),
        ("--indel-window", get(&v, 'l', "5")),
        ("--min-qual", get(&f, 'q', "10")),
        ("--min-good-bases", get(&f, 'g', "10000")),
        ("--bin-size", get(&f, 's', "100")),
    ] {
        args.push(flag.to_string());
        args.push(value);
    }
    args
}

/// `--bin-callable ignore` on each route A VCF fits, to the bit, what the
/// real fq2psmcfa's psmcfa of the same calls fits; `auto` differs exactly
/// where the bins are partly callable.
#[test]
fn ignore_fits_what_fq2psmcfas_output_fits() {
    let dir = TempDir::new("bin_callable_route_a");
    let sets = fs::read_to_string(fixture("route_a/expected/option_sets.tsv")).unwrap();
    let mut differ = 0;
    for line in sets.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        let (name, vcf2fq, fq2psmcfa, input) = (cols[0], cols[1], cols[2], cols[3]);
        let args = route_a_args(vcf2fq, fq2psmcfa, name == "auto");
        let bin_size = args[args.len() - 1].clone();
        let mut psmcfa = fixture(&format!("route_a/expected/{name}.psmcfa"));
        if !psmcfa.exists() {
            psmcfa = fixture(&format!("route_a/expected/{name}.psmcfa.gz"));
        }
        let vcf = fixture(&format!("route_a/{input}"));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (from_psmcfa, psmcfa_out) = fit(
            &dir,
            &format!("{name}_psmcfa"),
            &psmcfa,
            &["--bin-size", &bin_size],
        );
        let mut ignore_args = args.clone();
        ignore_args.extend(["--bin-callable", "ignore"]);
        let (ignore, ignore_out) = fit(&dir, &format!("{name}_ignore"), &vcf, &ignore_args);
        let (auto, auto_out) = fit(&dir, &format!("{name}_auto"), &vcf, &args);

        assert_eq!(fit_bits(&ignore), fit_bits(&from_psmcfa), "{name}");
        assert_eq!(rounds(&ignore_out), rounds(&psmcfa_out), "{name}");
        assert_eq!(bin_line(&psmcfa_out), None, "{name}: psmcfa output is C's");
        assert!(from_psmcfa.get("bin_callable").unwrap()["counts"].is_null());
        assert_eq!(from_psmcfa["bin_callable"]["model"], "full-bins");
        assert_eq!(ignore["bin_callable"]["model"], "full-bins", "{name}");
        assert_eq!(auto["bin_callable"]["model"], "callable-length", "{name}");
        assert_eq!(auto["bin_callable"]["counts"], "exact", "{name}");
        let partial = auto["bin_callable"]["callable"]["partial_bins"]
            .as_u64()
            .unwrap();
        let line = bin_line(&auto_out).unwrap();
        assert!(
            line.starts_with("MM\tbin_model:callable-length, counts:exact, mean_callable:"),
            "{line}"
        );
        assert!(
            bin_line(&ignore_out)
                .unwrap()
                .starts_with("MM\tbin_model:full-bins")
        );
        if partial == 0 {
            assert_eq!(fit_bits(&auto), fit_bits(&ignore), "{name}");
        } else {
            differ += 1;
            assert_ne!(fit_bits(&auto), fit_bits(&ignore), "{name}");
        }
    }
    assert!(
        differ >= 3,
        "only {differ} option sets with partly callable bins"
    );
}

/// A one-sample VCF of a 3 Mb chromosome with heterozygous SNPs, and a BED
/// of callable sites that leaves out `missing` of the bases in runs of
/// about 20 bp (none when 0).
fn synthetic(dir: &TempDir, missing: f64) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
    const LEN: u64 = 3_000_000;
    let mut rng = SmallRng::seed_from_u64(11);
    let vcf = dir.path(&format!("calls_{missing}.vcf"));
    let mut f = fs::File::create(&vcf).unwrap();
    writeln!(f, "##fileformat=VCFv4.2\n##contig=<ID=chr1,length={LEN}>").unwrap();
    writeln!(
        f,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    writeln!(
        f,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
    )
    .unwrap();
    // Heterozygosity varying along the chromosome, as TMRCAs do.
    let mut pos = 0u64;
    loop {
        let block = pos / 200_000;
        let rate = [8e-4, 2e-4, 1.5e-3, 5e-4][(block % 4) as usize];
        pos += 1 + (-(1.0 - rng.r#gen::<f64>()).ln() / rate) as u64;
        if pos > LEN {
            break;
        }
        writeln!(f, "chr1\t{pos}\t.\tA\tC\t50\tPASS\t.\tGT\t0/1").unwrap();
    }
    if missing == 0.0 {
        return (vcf, None);
    }
    let bed = dir.path(&format!("callable_{missing}.bed"));
    let mut b = fs::File::create(&bed).unwrap();
    let mean_gap = 20.0 * (1.0 - missing) / missing;
    let mut at = 0u64;
    while at < LEN {
        let called = 1 + (-(1.0 - rng.r#gen::<f64>()).ln() * mean_gap) as u64;
        let end = (at + called).min(LEN);
        writeln!(b, "chr1\t{at}\t{end}").unwrap();
        at = end + 1 + (-(1.0 - rng.r#gen::<f64>()).ln() * 20.0) as u64;
    }
    (vcf, Some(bed))
}

#[test]
fn auto_on_complete_data_fits_as_ignore_and_the_psmcfa_do() {
    let dir = TempDir::new("bin_callable_complete");
    let (vcf, _) = synthetic(&dir, 0.0);
    let (auto, auto_out) = fit(&dir, "auto", &vcf, &["--vcf-mode", "mask"]);
    let (ignore, ignore_out) = fit(
        &dir,
        "ignore",
        &vcf,
        &["--vcf-mode", "mask", "--bin-callable", "ignore"],
    );
    let psmcfa = dir.path("calls.psmcfa");
    run_ok([
        "convert",
        vcf.to_str().unwrap(),
        "--vcf-mode",
        "mask",
        "--psmcfa",
        psmcfa.to_str().unwrap(),
        "--quiet",
    ]);
    let (plain, plain_out) = fit(&dir, "plain", &psmcfa, &[]);
    assert_eq!(auto["bin_callable"]["model"], "callable-length");
    assert_eq!(auto["bin_callable"]["callable"]["mean_fraction"], 1.0);
    assert_eq!(fit_bits(&auto), fit_bits(&ignore));
    assert_eq!(fit_bits(&auto), fit_bits(&plain));
    assert_eq!(rounds(&auto_out), rounds(&plain_out));
    assert_eq!(rounds(&ignore_out), rounds(&plain_out));
}

#[test]
fn auto_fits_partly_callable_bins_and_says_so() {
    let dir = TempDir::new("bin_callable_dispersed");
    let (vcf, bed) = synthetic(&dir, 0.3);
    let bed = bed.unwrap();
    let mask = ["--vcf-mask", bed.to_str().unwrap()];
    let tmrca = dir.path("tmrca.tsv.gz");
    let mut auto_args = mask.to_vec();
    auto_args.extend(["--tmrca-out", tmrca.to_str().unwrap()]);
    let (auto, auto_out) = fit(&dir, "auto", &vcf, &auto_args);
    let mut ignore_args = mask.to_vec();
    ignore_args.extend(["--bin-callable", "ignore"]);
    let (ignore, _) = fit(&dir, "ignore", &vcf, &ignore_args);

    let b = &auto["bin_callable"];
    assert_eq!(
        (&b["mode"], &b["model"]),
        (&"auto".into(), &"callable-length".into())
    );
    let mean = b["callable"]["mean_fraction"].as_f64().unwrap();
    assert!((0.6..0.8).contains(&mean), "mean callable share {mean}");
    assert!(b["callable"]["partial_bins"].as_u64().unwrap() > 20_000);
    assert_eq!(ignore["bin_callable"]["mode"], "ignore");
    assert_eq!(ignore["bin_callable"]["callable"], b["callable"]);
    // QC notes the correction, and warns when the fit ignores the counts.
    let has = |v: &Value, list: &str| {
        v["qc"][list]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "PARTLY_CALLABLE_BINS")
    };
    assert!(has(&auto, "notes") && !has(&auto, "warnings"));
    assert!(has(&ignore, "warnings"));
    // Ignoring the missing bases takes them for absent heterozygosity.
    let (theta_auto, theta_ignore) = (
        auto["theta"].as_f64().unwrap(),
        ignore["theta"].as_f64().unwrap(),
    );
    assert!(
        theta_ignore < 0.85 * theta_auto,
        "{theta_ignore} vs {theta_auto}"
    );
    let line = bin_line(&auto_out).unwrap();
    assert!(line.contains(&format!("mean_callable:{mean:.6}")), "{line}");
    assert!(tmrca.exists());
    let html = fs::read_to_string(dir.path("auto.html")).unwrap();
    assert!(html.contains("callable-length model"));
    // Its per-sequence het/kb is per callable base, and says so; the JSON
    // keeps both.
    assert!(html.contains("Het/kb*") && html.contains("het/kb* is per kb of callable bases"));
    let per_callable = auto["qc"]["het_per_callable_kb"].as_f64().unwrap();
    assert!(per_callable > auto["qc"]["het_per_kb"].as_f64().unwrap());

    // The run says so, and convert says what a psmcfa loses.
    let out = common::run([
        vcf.to_str().unwrap(),
        dir.path("talk.json").to_str().unwrap(),
        "0",
        "--vcf-mask",
        bed.to_str().unwrap(),
        "--no-progress",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Callable bases (exact counts): called bins hold")
            && stderr.contains("the fit weighs each called bin by its callable bases"),
        "{stderr}"
    );
    let out = common::run([
        "convert",
        vcf.to_str().unwrap(),
        "--vcf-mask",
        bed.to_str().unwrap(),
        "--psmcfa",
        dir.path("lossy.psmcfa").to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("of called bins are partly callable")
            && stderr.contains("psmcfa cannot record that")
            && stderr.contains("fit the VCF itself"),
        "{stderr}"
    );

    // psmcfa input has no counts: ignore changes nothing, and says so.
    let out = common::run([
        dir.path("lossy.psmcfa").to_str().unwrap(),
        dir.path("p.json").to_str().unwrap(),
        "0",
        "--bin-callable",
        "ignore",
        "--no-progress",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--bin-callable ignore changes nothing for psmcfa input"),
        "{stderr}"
    );
}

/// Multihetsep's counts follow the reader's placement of the uncalled
/// sites: `auto` does not fit them (it fits exactly what `ignore` and the
/// multihetsep's psmcfa fit), `approximate` does, and the run says which.
#[test]
fn auto_does_not_fit_multihetseps_approximate_counts() {
    let dir = TempDir::new("bin_callable_mhs");
    let (vcf, bed) = synthetic(&dir, 0.3);
    let mhs = dir.path("calls.multihetsep");
    let psmcfa = dir.path("calls_mhs.psmcfa");
    // Writing multihetsep from a VCF with uncalled sites says what reading
    // it back costs.
    let out = common::run([
        "convert",
        vcf.to_str().unwrap(),
        "--vcf-mask",
        bed.unwrap().to_str().unwrap(),
        "--multihetsep",
        mhs.to_str().unwrap(),
    ]);
    assert!(out.status.success());
    let says = String::from_utf8_lossy(&out.stderr);
    assert!(
        says.contains("of the sites here are uncalled")
            && says.contains("so to fit with psmc-rs, fit the VCF itself"),
        "{says}"
    );
    run_ok([
        "convert",
        mhs.to_str().unwrap(),
        "--psmcfa",
        psmcfa.to_str().unwrap(),
        "--quiet",
    ]);
    let (auto, auto_out) = fit(&dir, "mhs_auto", &mhs, &[]);
    let (ignore, ignore_out) = fit(&dir, "mhs_ignore", &mhs, &["--bin-callable", "ignore"]);
    let (approx, approx_out) = fit(&dir, "mhs_approx", &mhs, &["--bin-callable", "approximate"]);
    let (plain, plain_out) = fit(&dir, "mhs_psmcfa", &psmcfa, &[]);

    let b = &auto["bin_callable"];
    assert_eq!(
        (&b["mode"], &b["model"], &b["counts"]),
        (&"auto".into(), &"full-bins".into(), &"approximate".into())
    );
    // With 30% of the sites uncalled, QC warns about the placement in
    // every mode, and sizes no bias from the approximate counts.
    for v in [&auto, &approx] {
        let codes = |list: &str| -> Vec<String> {
            v["qc"][list]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap().to_string())
                .collect()
        };
        assert!(codes("warnings").contains(&"APPROX_MISSING_PLACEMENT".to_string()));
        assert!(!codes("warnings").contains(&"PARTLY_CALLABLE_BINS".to_string()));
        assert!(!codes("notes").contains(&"PARTLY_CALLABLE_BINS".to_string()));
    }
    assert!(b["callable"]["partial_bins"].as_u64().unwrap() > 0);
    assert_eq!(fit_bits(&auto), fit_bits(&ignore));
    assert_eq!(fit_bits(&auto), fit_bits(&plain));
    assert_eq!(rounds(&auto_out), rounds(&ignore_out));
    assert_eq!(rounds(&auto_out), rounds(&plain_out));
    assert!(
        bin_line(&auto_out)
            .unwrap()
            .starts_with("MM\tbin_model:full-bins, counts:approximate")
    );
    assert_eq!(approx["bin_callable"]["model"], "callable-length");
    assert_ne!(fit_bits(&approx), fit_bits(&auto));
    assert!(
        bin_line(&approx_out)
            .unwrap()
            .starts_with("MM\tbin_model:callable-length, counts:approximate")
    );

    let stderr = |mode: &str| {
        let out = common::run([
            mhs.to_str().unwrap(),
            dir.path(&format!("talk_{mode}.json")).to_str().unwrap(),
            "0",
            "--bin-callable",
            mode,
            "--no-progress",
        ]);
        String::from_utf8_lossy(&out.stderr).to_string()
    };
    let auto_says = stderr("auto");
    assert!(
        auto_says.contains("Callable bases (approximate counts)")
            && auto_says.contains("the fit treats every called bin as 100 bases"),
        "{auto_says}"
    );
    let approx_says = stderr("approximate");
    assert!(
        approx_says.contains("the fit weighs each called bin by them (--bin-callable approximate)"),
        "{approx_says}"
    );
    // A psmcfa of the multihetsep fits as the multihetsep does, so convert
    // has no partial bins to warn about.
    let out = common::run([
        "convert",
        mhs.to_str().unwrap(),
        "--psmcfa",
        dir.path("again.psmcfa").to_str().unwrap(),
    ]);
    let says = String::from_utf8_lossy(&out.stderr);
    assert!(!says.contains("psmcfa cannot record that"), "{says}");
}

/// The model a run saved, with its matrices.
fn saved_model(path: &Path) -> PsmcModel {
    PsmcModel::from_params_file(&load_params(path).unwrap()).unwrap()
}

/// The counts of a counted run reach the TMRCA decoding, the log-likelihood
/// of a run of 0 iterations and the bootstrap, as the library computes them
/// with the counts, and each of these differs without them, so a run that
/// dropped the counts on the way to one of them would fail here.
#[test]
fn the_counts_reach_decoding_the_start_and_the_bootstrap() {
    let dir = TempDir::new("bin_callable_passes");
    let (vcf, bed) = synthetic(&dir, 0.3);
    let bed = bed.unwrap();
    let data = load(
        &vcf,
        &InputOptions {
            masks: vec![bed.clone()],
            ..InputOptions::default()
        },
    )
    .unwrap();
    let obs = &data.obs;
    let callable: &CallableBases = obs.callable.as_ref().unwrap();
    let cli = |out: &Path, n_iter: &str, extra: &[&str]| {
        let mut argv = vec![
            vcf.to_str().unwrap().to_string(),
            out.to_str().unwrap().to_string(),
            n_iter.to_string(),
            "--vcf-mask".to_string(),
            bed.to_str().unwrap().to_string(),
            "--no-progress".to_string(),
            "--quiet".to_string(),
        ];
        argv.extend(extra.iter().map(|a| a.to_string()));
        run_ok(&argv);
    };

    // The TMRCA decoding of the fitted model, with the counts.
    let fit_json = dir.path("decoded.json");
    let tmrca = dir.path("decoded.tmrca.tsv");
    cli(&fit_json, "2", &["--tmrca-out", tmrca.to_str().unwrap()]);
    let model = saved_model(&fit_json);
    let years = model.state_tmrca_years(100.0, 25.0);
    let decode = |path: &Path, counts: Option<&CallableBases>| {
        write_tmrca_posterior_tsv_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            &years,
            path,
            TmrcaOptions {
                progress_bars: false,
                max_points: 20_000,
                callable: counts,
            },
        )
        .unwrap();
        fs::read(path).unwrap()
    };
    let cli_tmrca = fs::read(&tmrca).unwrap();
    assert!(
        cli_tmrca == decode(&dir.path("with.tsv"), Some(callable)),
        "the CLI decodes with the counts"
    );
    assert!(cli_tmrca != decode(&dir.path("without.tsv"), None));

    // The log-likelihood of a run of 0 iterations: its start, with the counts.
    let start_json = dir.path("start.json");
    cli(&start_json, "0", &[]);
    let start = saved_model(&start_json);
    let loglike = |counts: Option<&CallableBases>| {
        log_likelihood_with_options(
            start.prior_matrix(),
            start.transition_matrix(),
            start.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            EStepOptions {
                callable: counts,
                ..EStepOptions::default()
            },
        )
        .unwrap()
    };
    let reported = json(&start_json)["final_loglike"].as_f64().unwrap();
    assert_eq!(reported.to_bits(), loglike(Some(callable)).to_bits());
    assert_ne!(reported, loglike(None));

    // Every bootstrap replicate is fitted on its resampled bins and counts,
    // and its .psmc file says so.
    let boot_json = dir.path("boot.json");
    cli(
        &boot_json,
        "1",
        &[
            "--bootstrap",
            "2",
            "--bootstrap-iters",
            "1",
            "--bootstrap-block-size",
            "3000",
            "--bootstrap-seed",
            "7",
        ],
    );
    let boot = json(&boot_json);
    let cfg = BootstrapConfig {
        replicates: 2,
        trunk_size: 3000,
        seed: 7,
        em_iters: 1,
        tr_ratio: boot["tr_ratio"].as_f64().unwrap(),
        t_max: 15.0,
        batch_size: None,
        out_dir: None,
    };
    let mstep = MStepConfig {
        progress: false,
        ..MStepConfig::default()
    };
    let template = saved_model(&boot_json);
    let replicates = |seqs: &Sequences| {
        let opts = BootstrapOptions {
            skip_invalid_starts: true,
            ..BootstrapOptions::default()
        };
        run_bootstrap_sequences(seqs, &template, &cfg, &mstep, opts).unwrap()
    };
    let seqs = observations_to_sequences(obs).unwrap();
    assert!(seqs.callable.is_some());
    let bins_only = Sequences::from(seqs.bins.clone());
    let with = replicates(&seqs);
    let without = replicates(&bins_only);
    assert_eq!(with.len(), 2);
    let boot_dir = dir.path("boot.bootstrap");
    for (r, w) in with.iter().zip(&without) {
        let n = r.index + 1;
        let saved = load_params(&boot_dir.join(format!("replicate_{n:03}.json"))).unwrap();
        let fitted = |m: &PsmcModel| (m.theta, m.rho, m.t_max, m.lam.clone());
        assert_eq!(
            (saved.theta, saved.rho, saved.t_max, saved.lam.clone()),
            fitted(&r.model),
            "replicate {n}"
        );
        assert_ne!(saved.theta, w.model.theta, "replicate {n}");
        let psmc = fs::read_to_string(boot_dir.join(format!("replicate_{n:03}.psmc"))).unwrap();
        assert!(
            psmc.lines()
                .any(|l| l.starts_with("MM\tbin_model:callable-length, mean_callable:")),
            "replicate {n}"
        );
    }
}
