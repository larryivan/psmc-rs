//! The HTML report: the curve comes from the model's scaling helpers, and
//! the settings, QC, bootstrap and TMRCA sections show what they are given.

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{BootstrapCurvePoint, BootstrapSummary};
use psmc_rs::hmm::TmrcaReportData;
use psmc_rs::io::CallableSummary;
use psmc_rs::io::params::{
    BinCallableRecord, CALLABLE_LENGTH_MODEL, FULL_BINS_MODEL, InputSummary,
};
use psmc_rs::qc::{QcContext, QcInput, QcSeq, analyze};
use psmc_rs::report::{BootstrapReport, ReportData, write_html_report};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

fn temp_html(prefix: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.html", std::process::id()))
}

fn model() -> PsmcModel {
    let mut m = PsmcModel::new(15.0, 22, 0.0072, 0.0015, 2.5e-8, Some("4+5*3+4".into())).unwrap();
    m.lam = vec![2.0, 0.5, 0.8, 1.6, 3.0, 6.0, 9.0];
    m.param_recalculate().unwrap();
    m
}

fn base<'a>(m: &'a PsmcModel) -> ReportData<'a> {
    ReportData {
        model: m,
        bin_size: 100.0,
        gen_years: 25.0,
        input_file: Path::new("data/in.psmcfa"),
        output_json: Path::new("out/fit.json"),
        input: None,
        n_iter: 7,
        final_loglike: Some(-12345.678901),
        settings: vec![("theta/rho at start".to_string(), "4".to_string())],
        qc: None,
        bin_callable: None,
        bootstrap: None,
        tmrca: None,
    }
}

/// The report says which emission model the fit used and what the counts
/// of callable bases held.
#[test]
fn the_report_shows_how_the_fit_used_the_callable_bases() {
    let m = model();
    let mut record = BinCallableRecord {
        mode: "auto".to_string(),
        model: CALLABLE_LENGTH_MODEL.to_string(),
        counts: Some("exact".to_string()),
        callable: Some(CallableSummary {
            bin_size: 100,
            called_bins: 1000,
            partial_bins: 850,
            mean_fraction: 0.7012,
            histogram: [0, 5, 10, 40, 100, 200, 250, 200, 45, 150],
        }),
    };
    let html = render(&ReportData {
        bin_callable: Some(&record),
        ..base(&m)
    });
    assert!(html.contains(
        "<tr><th>Bin model</th><td>each called bin by its callable bases (callable-length \
         model); --bin-callable auto</td></tr>"
    ));
    assert!(html.contains(
        "<tr><th>Callable bases of called bins</th><td>70.1% on average; 85.0% of called bins \
         partly callable (counts exact)</td></tr>"
    ));
    record.mode = "ignore".to_string();
    record.model = FULL_BINS_MODEL.to_string();
    let html = render(&ReportData {
        bin_callable: Some(&record),
        ..base(&m)
    });
    assert!(html.contains(
        "every called bin as the bin size in bases (C psmc&#39;s model); --bin-callable ignore"
    ));
    assert!(html.contains("70.1% on average"));
}

fn render(data: &ReportData) -> String {
    let path = temp_html("psmc_report");
    write_html_report(&path, data).unwrap();
    let html = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(path);
    html
}

/// The JSON assigned to `const <name> = ` on one line of the page.
fn js_const(html: &str, name: &str) -> Value {
    let prefix = format!("const {name} = ");
    let line = html
        .lines()
        .find_map(|l| l.trim().strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no {name}"));
    serde_json::from_str(line.trim_end_matches(';')).unwrap()
}

/// The main curve's (x, y) points.
fn curve_points(html: &str) -> Vec<[f64; 2]> {
    let series = js_const(html, "series");
    let main = series
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "psmc-rs")
        .expect("the curve is labelled psmc-rs");
    serde_json::from_value(main["data"].clone()).unwrap()
}

#[test]
fn the_curve_is_the_models_ne_curve() {
    let m = model();
    let html = render(&base(&m));
    let expected: Vec<[f64; 2]> = m
        .ne_curve(100.0, 25.0)
        .unwrap()
        .into_iter()
        .filter(|p| p[0] > 0.0)
        .collect();
    assert_eq!(curve_points(&html), expected);
    assert!(!html.contains("Rust estimate"));

    // Generation time and bin size scale the axes as psmc_plot.pl does.
    let html = render(&ReportData {
        gen_years: 50.0,
        bin_size: 50.0,
        ..base(&m)
    });
    let scaled = curve_points(&html);
    for (a, b) in scaled.iter().zip(&expected) {
        assert!((a[0] / b[0] - 4.0).abs() < 1e-12, "{a:?} {b:?}");
        assert!((a[1] / b[1] - 2.0).abs() < 1e-12, "{a:?} {b:?}");
    }
}

#[test]
fn the_summary_shows_version_loglike_and_settings() {
    let m = model();
    let input = InputSummary {
        format: "vcf".to_string(),
        vcf_mode: Some("consensus".to_string()),
        sequences: 3,
        bins: 1000,
        het_bins: 25,
        missing_bins: 100,
        dropped_sequences: 2,
    };
    let html = render(&ReportData {
        input: Some(&input),
        ..base(&m)
    });
    assert!(html.contains(&format!(
        "<tr><th>psmc-rs</th><td>{}</td></tr>",
        env!("CARGO_PKG_VERSION")
    )));
    assert!(html.contains(&format!(
        "<tr><th>HMM kernels</th><td>{}</td></tr>",
        psmc_rs::hmm::simd_path()
    )));
    assert!(html.contains("<tr><th>Final log-likelihood</th><td>-12345.678901</td></tr>"));
    assert!(html.contains("<tr><th>theta/rho at start</th><td>4</td></tr>"));
    assert!(html.contains("<tr><th>Input format</th><td>vcf (consensus mode)</td></tr>"));
    assert!(html.contains("1,000 (2.5% heterozygous, 10.0% missing)"));
    assert!(html.contains("<tr><th>Pattern</th><td>4+5*3+4</td></tr>"));
    // No QC, bootstrap or TMRCA given: no such sections.
    assert!(!html.contains("<h2>Input QC</h2>"));
    assert!(!html.contains("<h2>Bootstrap</h2>"));
    assert!(!html.contains("TMRCA Posterior Track"));
}

#[test]
fn a_missing_chart_library_is_explained() {
    let m = model();
    let html = render(&base(&m));
    // The script stops and shows the message when ECharts did not load.
    let guard = html.find(r#"if (typeof echarts === "undefined")"#).unwrap();
    let first_use = html.find("echarts.init").unwrap();
    assert!(guard < first_use);
    assert!(html.contains(r#"id="echarts-missing" hidden"#));
    assert!(html.contains("<code>PSMC_ECHARTS_JS</code>"));
    assert!(html.contains("cdn.jsdelivr.net/npm/echarts"));
}

#[test]
fn qc_findings_and_metrics_are_listed() {
    let rows = vec![{
        let mut r = vec![0u8; 3000];
        r[..40].fill(1);
        r[100..400].fill(2);
        r
    }];
    let seqs = [QcSeq::new("chr<1>", 3000)];
    let qc = analyze(&QcInput {
        rows: &rows,
        row_starts: &[true],
        seqs: &seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context: QcContext::default(),
        callable: None,
    });
    assert!(qc.has_warnings());
    let m = model();
    let html = render(&ReportData {
        qc: Some(&qc),
        ..base(&m)
    });
    assert!(html.contains("<h2>Input QC</h2>"));
    for f in &qc.findings {
        assert!(html.contains(&format!("<code>{}</code>", f.code.as_str())));
    }
    assert!(html.contains(r#"<span class="badge warning">warning</span>"#));
    assert!(html.contains("<tr><th>Bins T / K / N</th><td>88.7% / 1.3% / 10.0%</td></tr>"));
    // Names are escaped.
    assert!(html.contains("chr&lt;1&gt;"));
    assert!(!html.contains("chr<1>"));
}

#[test]
fn the_bootstrap_section_states_its_caveat() {
    let m = model();
    let summary = BootstrapSummary {
        n_replicates: 30,
        skipped_replicates: Vec::new(),
        block_size: 500_000,
        seed: 42,
        points: vec![BootstrapCurvePoint {
            x_years: 1000.0,
            ne_main: 5000.0,
            ne_q025: 4000.0,
            ne_q500: 5100.0,
            ne_q975: 6500.0,
        }],
    };
    let html = render(&ReportData {
        bootstrap: Some(BootstrapReport {
            summary: &summary,
            em_iters: 25,
            trunks: Some(61),
            started_from: None,
        }),
        ..base(&m)
    });
    assert!(html.contains("<h2>Bootstrap</h2>"));
    assert!(html.contains("<tr><th>Trunk size (bins)</th><td>500,000</td></tr>"));
    assert!(html.contains("<tr><th>Trunks resampled</th><td>61</td></tr>"));
    assert!(html.contains("<tr><th>Replicates start from</th><td>a fresh start</td></tr>"));
    assert!(html.contains("Read it as a lower bound on the uncertainty"));
    // The calibration of the README, at the CLI's defaults.
    assert!(html.contains("with psmc-rs's default start and objective"));
    assert!(html.contains(
        "a median 0.85 (whole sequences as trunks) to 0.91 (1,000-bin trunks) of the true \
         sampling standard deviation, and 0.74 to 0.78 at the oldest times"
    ));
    assert!(html.contains("about a third of that standard deviation"));
    assert!(!html.contains("0.90 to 0.92"));
    assert!(html.contains("each fitted from a fresh start"));
    assert!(!html.contains("C psmc -i -b"));
    assert!(!html.contains("Skipped replicates") && !html.contains("were left out"));
    let series = js_const(&html, "series");
    assert!(
        series
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["name"] == "Bootstrap q2.5")
    );
    // The band stacks the spread on q2.5. ECharts stacks on the first
    // numeric dimension of a series' data, so its points must hold Ne first
    // and map it to the y axis; with [years, ne] points it added the years
    // and drew the spread itself as Ne.
    assert!(html.contains(r#"const ribbonDims = ["ne", "years"];"#));
    assert!(html.contains(r#"const ribbonEncode = { x: "years", y: "ne" };"#));
    assert!(html.contains("q025Data.map((p) => [p[1], p[0]])"));
    assert!(html.contains("return [Math.max(0, p[1] - lo), p[0], lo, p[1]];"));
    for name in ["__Bootstrap CI base", "Bootstrap 95% CI"] {
        let at = html.find(&format!(r#"name: "{name}","#)).unwrap();
        let def = &html[at..at + 400];
        assert!(def.contains(r#"stack: "bootstrap_ci""#), "{def}");
        assert!(def.contains("dimensions: ribbonDims"), "{def}");
        assert!(def.contains("ribbonEncode"), "{def}");
    }
}

/// Replicates that continue saved parameters (C psmc -i -b) say so, with
/// the further caveat that brings.
#[test]
fn a_bootstrap_from_saved_parameters_says_so() {
    let m = model();
    let summary = BootstrapSummary {
        n_replicates: 10,
        skipped_replicates: Vec::new(),
        block_size: 20_000,
        seed: 42,
        points: Vec::new(),
    };
    let html = render(&ReportData {
        bootstrap: Some(BootstrapReport {
            summary: &summary,
            em_iters: 2,
            trunks: Some(30),
            started_from: Some(Path::new("fits/<old>.json")),
        }),
        ..base(&m)
    });
    assert!(html.contains(
        "<tr><th>Replicates start from</th><td>the parameters in fits/&lt;old&gt;.json</td></tr>"
    ));
    assert!(html.contains("each fitted from the saved parameters the main fit continued"));
    assert!(html.contains("the band understates the uncertainty further"));
    assert!(!html.contains("each fitted from a fresh start"));
}

/// Replicates left out because their fresh start was invalid are counted
/// and named, with the caveat that brings.
#[test]
fn skipped_replicates_are_reported() {
    let m = model();
    let summary = BootstrapSummary {
        n_replicates: 18,
        skipped_replicates: vec![3, 12],
        block_size: 1_000,
        seed: 42,
        points: Vec::new(),
    };
    let html = render(&ReportData {
        bootstrap: Some(BootstrapReport {
            summary: &summary,
            em_iters: 5,
            trunks: Some(40),
            started_from: None,
        }),
        ..base(&m)
    });
    assert!(html.contains("<tr><th>Replicates</th><td>18</td></tr>"));
    assert!(html.contains("<tr><th>Skipped replicates</th><td>2 of 20 (3, 12)</td></tr>"));
    assert!(html.contains(
        "2 of the 20 replicates were left out: the fresh start of their resampled data had \
         negative transition probabilities"
    ));
    assert!(html.contains("the band may be too narrow"));
}

#[test]
fn tmrca_states_are_labelled_with_their_intervals() {
    let m = model();
    let n = m.n_steps + 1;
    let tmrca = TmrcaReportData {
        sampled_mean: vec![[0.0, 1000.0], [1.0, 2000.0]],
        sampled_map: vec![[0.0, 900.0], [1.0, 2100.0]],
        sampled_lo: vec![[0.0, 100.0], [1.0, 200.0]],
        sampled_hi: vec![[0.0, 10000.0], [1.0, 20000.0]],
        state_mass: vec![1.0; n],
        state_years: m.state_tmrca_years(100.0, 25.0),
        total_sites: 2,
        seq_count: 1,
    };
    let html = render(&ReportData {
        tmrca: Some(&tmrca),
        ..base(&m)
    });
    assert!(html.contains("TMRCA Posterior Track"));
    let bins = js_const(&html, "tmrcaStateBins");
    let bins = bins.as_array().unwrap();
    let curve = m.ne_curve(100.0, 25.0).unwrap();
    assert_eq!(bins.len(), n);
    for (k, b) in bins.iter().enumerate() {
        assert_eq!(b["start_years"].as_f64().unwrap(), curve[k][0], "state {k}");
        match curve.get(k + 1) {
            Some(next) => assert_eq!(b["end_years"].as_f64().unwrap(), next[0]),
            None => assert!(b["end_years"].is_null()),
        }
    }
}
