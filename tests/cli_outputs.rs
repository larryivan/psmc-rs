//! The outputs of an inference run through the binary: determinism across
//! thread counts, C psmc's .psmc output, the bootstrap, TMRCA decoding and
//! time scaling.

mod common;

use common::psmc_text::{header_value, parse_rounds};
use common::{TempDir, json, run_err, run_ok, small_psmcfa};
use psmc_rs::PsmcModel;
use psmc_rs::io::params::load_params;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Run inference on `input` into `dir/<name>.json` with `args`.
fn fit(dir: &TempDir, input: &Path, name: &str, n_iter: usize, args: &[&str]) -> PathBuf {
    let out = dir.path(&format!("{name}.json"));
    let mut all = vec![
        input.display().to_string(),
        out.display().to_string(),
        n_iter.to_string(),
        "--no-progress".to_string(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    run_ok(&all);
    out
}

fn gunzip(path: &Path) -> String {
    let mut text = String::new();
    flate2::read::MultiGzDecoder::new(fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    text
}

/// The data rows of a TMRCA TSV as columns of numbers (after the four
/// identifying columns).
fn tmrca_rows(text: &str) -> Vec<(usize, Vec<f64>)> {
    text.lines()
        .skip(1)
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            let state = cols[4].parse().unwrap();
            (
                state,
                cols[5..].iter().map(|c| c.parse().unwrap()).collect(),
            )
        })
        .collect()
}

/// The same run with 1 and 4 threads writes the same bytes: the JSON, the
/// .psmc output, the TMRCA posterior and every bootstrap file.
#[test]
fn outputs_do_not_depend_on_the_thread_count() {
    let dir = TempDir::new("threads");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 20_000);
    let run = |threads: &str| -> Vec<(String, Vec<u8>)> {
        let tmrca = dir.path(&format!("t{threads}.tsv"));
        let psmc = dir.path(&format!("t{threads}.psmc"));
        let boot = dir.path(&format!("boot{threads}"));
        let out = fit(
            &dir,
            &input,
            &format!("t{threads}"),
            3,
            &[
                "--threads",
                threads,
                "--tmrca-out",
                tmrca.to_str().unwrap(),
                "--psmc-out",
                psmc.to_str().unwrap(),
                "--bootstrap",
                "3",
                "--bootstrap-block-size",
                "5000",
                "--bootstrap-iters",
                "2",
                "--bootstrap-dir",
                boot.to_str().unwrap(),
            ],
        );
        let mut files = vec![out, tmrca, psmc];
        let mut boot_files: Vec<PathBuf> = fs::read_dir(&boot)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        boot_files.sort();
        assert_eq!(boot_files.len(), 3 * 2 + 2, "{boot_files:?}");
        files.extend(boot_files);
        files
            .into_iter()
            .map(|p| {
                let name = p
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .replace(threads, "N");
                (name, fs::read(&p).unwrap())
            })
            .collect()
    };
    let one = run("1");
    let four = run("4");
    for ((name, a), (_, b)) in one.iter().zip(&four) {
        assert!(a == b, "{name} differs between 1 and 4 threads");
    }
}

/// --psmc-out writes C psmc's format with rounds 0 to N_ITER; the header
/// counts the input as C does, including empty records.
#[test]
fn psmc_output_has_a_round_per_iteration() {
    let dir = TempDir::new("psmc_out");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 4000);
    // An empty record, which C psmc counts in n_seqs.
    let mut text = fs::read_to_string(&input).unwrap();
    text.push_str(">empty\n");
    fs::write(&input, text).unwrap();
    let psmc = dir.path("fit.psmc");
    let out = fit(
        &dir,
        &input,
        "fit",
        3,
        &["--psmc-out", psmc.to_str().unwrap()],
    );
    let text = fs::read_to_string(&psmc).unwrap();
    let rounds = parse_rounds(&text);
    assert_eq!(rounds.len(), 4);
    assert_eq!(header_value(&text, "n_iterations"), Some("3"));
    assert_eq!(header_value(&text, "n_seqs"), Some("4"));
    assert_eq!(header_value(&text, "pattern"), Some("4+5*3+4"));
    assert_eq!(header_value(&text, "theta/rho"), Some("4"));
    assert!(text.contains(&format!(
        "MM\tVersion: psmc-rs {}",
        env!("CARGO_PKG_VERSION")
    )));
    // Round 0 holds the start, the last round the parameters written.
    let v = json(&out);
    let last = rounds.last().unwrap();
    let theta = v["theta"].as_f64().unwrap();
    assert!((last.tr.0 - theta).abs() < 1e-6, "{} vs {theta}", last.tr.0);
    assert_eq!(rounds[0].lk, 0.0);
    assert!(rounds[1..].iter().all(|r| r.lk < 0.0));
    assert_eq!(rounds[0].rs.len(), 23);
}

/// A bootstrap writes every replicate's JSON and .psmc file and the summary,
/// and the report shows it.
#[test]
fn bootstrap_writes_replicates_and_summary() {
    let dir = TempDir::new("bootstrap");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let out = dir.path("fit.json");
    let o = common::run([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "2",
        "--no-progress",
        "--bootstrap",
        "3",
        "--bootstrap-block-size",
        "1000",
    ]);
    assert!(o.status.success());
    // 3 sequences of 3 trunks: few enough for a warning.
    let stderr = String::from_utf8(o.stderr).unwrap();
    assert!(
        stderr.contains("Warning: the input forms only 9 bootstrap trunks of 1000 bins"),
        "{stderr}"
    );
    let boot = dir.path("fit.bootstrap");
    for rep in 1..=3 {
        let params = load_params(&boot.join(format!("replicate_{rep:03}.json"))).unwrap();
        assert_eq!(params.pattern.as_deref(), Some("4+5*3+4"));
        let text = fs::read_to_string(boot.join(format!("replicate_{rep:03}.psmc"))).unwrap();
        let rounds = parse_rounds(&text);
        assert_eq!(rounds.len(), 3, "replicate {rep}");
        // Each replicate fits its own resampled trunks from a fresh start.
        assert_eq!(rounds[0].mt, 15.0);
        assert!((rounds[2].tr.0 - params.theta).abs() < 1e-6);
        let n_seqs: usize = header_value(&text, "n_seqs").unwrap().parse().unwrap();
        assert!(
            (7..=11).contains(&n_seqs),
            "{n_seqs} trunks of about 1000 bins"
        );
    }
    assert!(!boot.join("replicate_004.json").exists());
    let summary = json(&boot.join("summary.json"));
    assert_eq!(summary["n_replicates"], 3);
    assert_eq!(summary["block_size"], 1000);
    assert_eq!(summary["seed"], 42);
    let tsv = fs::read_to_string(boot.join("summary.tsv")).unwrap();
    assert!(tsv.starts_with("x_years\tne_main\tne_q025\tne_q500\tne_q975\n"));

    let html = fs::read_to_string(out.with_extension("html")).unwrap();
    assert!(html.contains("<h2>Bootstrap</h2>"));
    assert!(html.contains("<tr><th>Trunks resampled</th><td>9</td></tr>"));
}

/// A bootstrap that cannot vary fails before the main fit.
#[test]
fn a_single_trunk_bootstrap_fails_before_fitting() {
    let dir = TempDir::new("one_trunk");
    let input = dir.path("in.psmcfa");
    // One sequence of 3,000 bins.
    fs::write(
        &input,
        format!(">chr1\n{}\n", "TTTTTTTTKTTTTTTTTTNT".repeat(150)),
    )
    .unwrap();
    let out = dir.path("fit.json");
    let err = run_err([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "30",
        "--bootstrap",
        "10",
        "--bootstrap-block-size",
        "100000",
    ]);
    assert!(err.contains("--bootstrap-block-size is too large"), "{err}");
    assert!(err.contains("single trunk"), "{err}");
    assert!(!out.exists());

    let err = run_err([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "5",
        "--bootstrap-dir",
        dir.path("boot").to_str().unwrap(),
    ]);
    assert!(
        err.contains("--bootstrap-dir applies to a bootstrap"),
        "{err}"
    );
    let err = run_err([
        input.to_str().unwrap(),
        out.to_str().unwrap(),
        "0",
        "--bootstrap",
        "2",
        "--bootstrap-block-size",
        "1000",
    ]);
    assert!(err.contains("0 EM iterations"), "{err}");
}

/// TMRCA times are each state's average coalescent time (C's TC), scaled
/// by --bin-size and --gen-years; a .gz path gives gzip.
#[test]
fn tmrca_times_are_the_states_average_times() {
    let dir = TempDir::new("tmrca");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let plain = dir.path("t.tsv");
    let out = fit(
        &dir,
        &input,
        "fit",
        1,
        &["--tmrca-out", plain.to_str().unwrap()],
    );
    let model = PsmcModel::from_params_file(&load_params(&out).unwrap()).unwrap();
    let years = model.state_tmrca_years(100.0, 25.0);
    let text = fs::read_to_string(&plain).unwrap();
    let rows = tmrca_rows(&text);
    assert_eq!(rows.len(), 3 * 3000);
    for (state, cols) in &rows {
        // tmrca_map_years is printed with 9 significant digits.
        let map = cols[0];
        assert!(
            (map - years[*state]).abs() <= 1e-8 * years[*state],
            "{map} vs {}",
            years[*state]
        );
    }

    let gz = dir.path("t.tsv.gz");
    fit(
        &dir,
        &input,
        "fit_gz",
        1,
        &["--tmrca-out", gz.to_str().unwrap()],
    );
    assert_eq!(&fs::read(&gz).unwrap()[..2], &[0x1f, 0x8b]);
    assert_eq!(gunzip(&gz), text);
}

/// --gen-years scales every time: doubling it doubles the TMRCA years and
/// the bootstrap summary's time axis, and leaves the fits alone.
#[test]
fn generation_time_scales_the_time_axis() {
    let dir = TempDir::new("gen_years");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let run = |years: &str| {
        let tmrca = dir.path(&format!("g{years}.tsv"));
        let out = fit(
            &dir,
            &input,
            &format!("g{years}"),
            1,
            &[
                "--gen-years",
                years,
                "--tmrca-out",
                tmrca.to_str().unwrap(),
                "--bootstrap",
                "2",
                "--bootstrap-block-size",
                "1000",
            ],
        );
        let summary = json(&dir.path(&format!("g{years}.bootstrap/summary.json")));
        (
            json(&out),
            tmrca_rows(&fs::read_to_string(tmrca).unwrap()),
            summary,
        )
    };
    let (fit25, rows25, boot25) = run("25");
    let (fit50, rows50, boot50) = run("50");
    let points = |v: &serde_json::Value| v["points"].as_array().unwrap().clone();
    for (a, b) in points(&boot25).iter().zip(&points(&boot50)) {
        assert_eq!(
            b["x_years"].as_f64().unwrap(),
            2.0 * a["x_years"].as_f64().unwrap()
        );
        assert_eq!(b["ne_q500"], a["ne_q500"]);
    }
    assert_eq!(fit25["lam"], fit50["lam"]);
    assert_eq!(fit25["theta"], fit50["theta"]);
    for ((s25, a), (s50, b)) in rows25.iter().zip(&rows50) {
        assert_eq!(s25, s50);
        // map, mean, q2.5 and q97.5 are times; pmax and entropy are not.
        for k in 0..4 {
            assert!((b[k] / a[k] - 2.0).abs() < 1e-8, "{} vs {}", b[k], a[k]);
        }
        assert_eq!(a[4..], b[4..]);
    }
    assert_eq!(fit25["gen_years"], 25.0);
    assert_eq!(fit50["gen_years"], 50.0);
}

/// The output JSON records how the fit was made, next to the model.
#[test]
fn the_output_json_records_the_run() {
    let dir = TempDir::new("metadata");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 4000);
    let psmc = dir.path("fit.psmc");
    let qc_json = dir.path("qc.json");
    let out = fit(
        &dir,
        &input,
        "fit",
        3,
        &[
            "--psmc-out",
            psmc.to_str().unwrap(),
            "--qc-json",
            qc_json.to_str().unwrap(),
            "--bin-size",
            "50",
        ],
    );
    let v = json(&out);
    assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(v["n_iter"], 3);
    assert_eq!(v["bin_size"], 50);
    assert_eq!(v["gen_years"], 25.0);
    assert_eq!(v["tr_ratio"], 4.0);
    assert_eq!(v["smooth_lambda"], 0.0);
    let loglike: Vec<f64> = serde_json::from_value(v["loglike"].clone()).unwrap();
    assert_eq!(loglike.len(), 4);
    assert_eq!(v["final_loglike"].as_f64().unwrap(), loglike[3]);
    // The .psmc rounds give the log-likelihood of each E-step's parameters.
    let rounds = parse_rounds(&fs::read_to_string(&psmc).unwrap());
    for (round, ll) in rounds[1..].iter().zip(&loglike) {
        assert!((round.lk - ll).abs() < 1e-6, "{} vs {ll}", round.lk);
    }

    let text = fs::read_to_string(&input).unwrap();
    let bins: Vec<char> = text
        .lines()
        .filter(|l| !l.starts_with('>'))
        .flat_map(str::chars)
        .collect();
    let count = |c: char| bins.iter().filter(|&&b| b == c).count();
    let input_summary = &v["input"];
    assert_eq!(input_summary["format"], "psmcfa");
    assert!(input_summary.get("vcf_mode").is_none());
    assert_eq!(input_summary["sequences"], 3);
    assert_eq!(input_summary["bins"], bins.len());
    assert_eq!(input_summary["het_bins"], count('K'));
    assert_eq!(input_summary["missing_bins"], count('N'));
    assert_eq!(input_summary["dropped_sequences"], 0);

    // The QC summary lists the codes of the full report.
    let qc = json(&qc_json);
    let codes = |severity: &str| -> Vec<serde_json::Value> {
        qc["findings"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["severity"] == severity)
            .map(|f| f["code"].clone())
            .collect()
    };
    assert_eq!(v["qc"]["warnings"].as_array().unwrap(), &codes("warning"));
    assert_eq!(v["qc"]["notes"].as_array().unwrap(), &codes("info"));
    assert!(codes("warning").contains(&"TOO_SHORT".into()));
    assert_eq!(v["qc"]["bins"], qc["totals"]["bins"]);
    assert_eq!(qc["bin_size"], 50);

    let html = fs::read_to_string(out.with_extension("html")).unwrap();
    assert!(html.contains("<h2>Input QC</h2>"));
    assert!(html.contains("<code>TOO_SHORT</code>"));
    assert!(html.contains("<tr><th>Starting theta0/rho0</th><td>4</td></tr>"));
}

/// "Last EM loglike" after N iterations is the log-likelihood of the
/// parameters written, which C psmc prints as the LK of round N + 1 (the
/// E-step of the next iteration); psmc-rs 0.1 printed round N's. The JSON's
/// `loglike[i]` is the value after i iterations, C's LK of round i + 1.
#[test]
fn last_em_loglike_is_c_psmcs_next_round() {
    let dir = TempDir::new("loglike_convention");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let fit_n = |n: usize| -> (f64, serde_json::Value, Vec<common::psmc_text::Round>) {
        let psmc = dir.path(&format!("n{n}.psmc"));
        let out = dir.path(&format!("n{n}.json"));
        let stdout = run_ok([
            input.to_str().unwrap(),
            out.to_str().unwrap(),
            &n.to_string(),
            "--no-progress",
            "--psmc-out",
            psmc.to_str().unwrap(),
        ]);
        let last = stdout
            .lines()
            .find_map(|l| l.strip_prefix("Last EM loglike: "))
            .unwrap();
        (
            last.parse().unwrap(),
            json(&out),
            parse_rounds(&fs::read_to_string(&psmc).unwrap()),
        )
    };
    let (two, two_json, _) = fit_n(2);
    let (_, _, three_rounds) = fit_n(3);
    // The .psmc prints LK with 6 decimals.
    assert!((two - three_rounds[3].lk).abs() < 1e-6, "{two} vs RD 3");
    assert!((two - three_rounds[2].lk).abs() > 1e-3, "not RD 2");
    let loglike: Vec<f64> = serde_json::from_value(two_json["loglike"].clone()).unwrap();
    assert_eq!(loglike[2], two);
    for i in 0..2 {
        assert!(
            (loglike[i] - three_rounds[i + 1].lk).abs() < 1e-6,
            "loglike[{i}]"
        );
    }
}

/// The QC report goes to stderr before the fit; --quiet leaves only the
/// results and the warnings.
#[test]
fn qc_text_is_printed_unless_quiet() {
    let dir = TempDir::new("quiet");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 2000);
    let run = |extra: &[&str]| {
        let mut args = vec![
            input.display().to_string(),
            dir.path("fit.json").display().to_string(),
            "1".to_string(),
            "--no-progress".to_string(),
        ];
        args.extend(extra.iter().map(|s| s.to_string()));
        let out = common::run(&args);
        assert!(out.status.success());
        (
            String::from_utf8(out.stdout).unwrap(),
            String::from_utf8(out.stderr).unwrap(),
        )
    };
    let (stdout, stderr) = run(&[]);
    assert!(stdout.contains("Last EM loglike: "), "{stdout}");
    assert!(
        stderr.contains("Input QC (psmcfa): 3 sequences"),
        "{stderr}"
    );
    assert!(stderr.contains("WARNING  TOO_SHORT"), "{stderr}");
    assert!(stderr.contains("Model: pattern 4+5*3+4"), "{stderr}");

    let (quiet_out, quiet_err) = run(&["--quiet"]);
    assert_eq!(quiet_out, stdout);
    assert!(!quiet_err.contains("Input QC"), "{quiet_err}");
    assert!(!quiet_err.contains("Model:"), "{quiet_err}");
    assert!(
        quiet_err.contains("Warning: QC TOO_SHORT: Only "),
        "{quiet_err}"
    );
}

/// A VCF read without a callable mask gets one warning about it, from QC.
#[test]
fn the_missing_mask_warning_is_given_once() {
    let dir = TempDir::new("no_mask");
    let vcf = dir.path("in.vcf");
    fs::write(
        &vcf,
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
         chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\nchr1\t24\t.\tA\tC\t.\tPASS\t.\tGT\t1/0\n",
    )
    .unwrap();
    let out = common::run([
        vcf.to_str().unwrap(),
        dir.path("fit.json").to_str().unwrap(),
        "0",
        "--bin-size",
        "4",
        "--min-good-bases",
        "0",
        "--no-progress",
    ]);
    assert!(out.status.success());
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains("VARIANTS_ONLY_NO_MASK"), "{stderr}");
    assert!(
        !stderr.contains("no callable mask (--vcf-mask) was given"),
        "{stderr}"
    );
    let v = json(&dir.path("fit.json"));
    assert_eq!(v["input"]["vcf_mode"], "mask");
    assert!(
        v["qc"]["warnings"]
            .as_array()
            .unwrap()
            .contains(&"VARIANTS_ONLY_NO_MASK".into())
    );
}
