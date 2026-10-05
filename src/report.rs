//! The interactive HTML report of a fit: the Ne curve (with the bootstrap
//! band), the run's settings, input QC, and the TMRCA panels.
//!
//! The charts use ECharts, loaded from a CDN unless `PSMC_ECHARTS_JS` names
//! a local copy to embed; without it the report says how to fix that.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::bootstrap::BootstrapSummary;
use crate::hmm::TmrcaReportData;
use crate::io::params::InputSummary;
use crate::model::PsmcModel;
use crate::qc::fmt::{count, pct, sig};
use crate::qc::{QcReport, Severity};

/// Everything [`write_html_report`] shows.
pub struct ReportData<'a> {
    /// The fitted model.
    pub model: &'a PsmcModel,
    /// Bases per bin, for the N0 and time scaling.
    pub bin_size: f64,
    /// Years per generation, for the time scaling.
    pub gen_years: f64,
    /// The input file (shown by name).
    pub input_file: &'a Path,
    /// The JSON output the report belongs to.
    pub output_json: &'a Path,
    /// The size of the input, when known.
    pub input: Option<&'a InputSummary>,
    /// EM iterations run.
    pub n_iter: usize,
    /// Log-likelihood of the fitted parameters (none after 0 iterations).
    pub final_loglike: Option<f64>,
    /// Further settings of the run to list, as (name, value).
    pub settings: Vec<(String, String)>,
    /// Input QC of the analysed data.
    pub qc: Option<&'a QcReport>,
    /// The bootstrap, if one was run.
    pub bootstrap: Option<BootstrapReport<'a>>,
    /// Posterior TMRCA decoding, if it was run.
    pub tmrca: Option<&'a TmrcaReportData>,
}

/// The bootstrap section of a report.
pub struct BootstrapReport<'a> {
    /// The curve quantiles over the replicates.
    pub summary: &'a BootstrapSummary,
    /// EM iterations per replicate.
    pub em_iters: usize,
    /// Trunks the input was cut into for resampling.
    pub trunks: Option<usize>,
    /// The saved parameters every replicate started from (C psmc `-i` with
    /// `-b`), or `None` for the usual fresh start.
    pub started_from: Option<&'a Path>,
}

#[derive(Serialize)]
struct JsSeries {
    name: String,
    data: Vec<[f64; 2]>,
}

#[derive(Serialize)]
struct TmrcaStateBin {
    start_years: f64,
    end_years: Option<f64>,
    mass: f64,
    mass_pct: f64,
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn fmt_num(v: f64) -> String {
    if !v.is_finite() {
        return "NaN".to_string();
    }
    if v != 0.0 && (v.abs() >= 1e4 || v.abs() < 1e-3) {
        format!("{v:.4e}")
    } else {
        format!("{v:.6}")
    }
}

fn echarts_script_tag() -> Result<String> {
    if let Ok(js_path) = std::env::var("PSMC_ECHARTS_JS") {
        let path = Path::new(&js_path);
        let js = fs::read_to_string(path).with_context(|| {
            format!(
                "failed to read ECharts bundle from PSMC_ECHARTS_JS={}",
                path.display()
            )
        })?;
        return Ok(format!("<script>\n{}\n</script>", js));
    }
    Ok(
        r#"<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>"#
            .to_string(),
    )
}

/// The report path for a JSON output: the same path with `.html`.
pub fn default_html_path(output_json: &Path) -> std::path::PathBuf {
    output_json.with_extension("html")
}

/// One `<tr>` of a two-column table; `value` is HTML.
fn row(out: &mut String, name: &str, value: &str) {
    let _ = writeln!(
        out,
        "<tr><th>{}</th><td>{value}</td></tr>",
        escape_html(name)
    );
}

/// The run summary table.
fn meta_rows(d: &ReportData, n0: f64, ranges: [(f64, f64); 2], points: usize) -> String {
    let m = d.model;
    let mut out = String::new();
    row(&mut out, "psmc-rs", env!("CARGO_PKG_VERSION"));
    row(
        &mut out,
        "Input",
        &format!(
            r#"<span class="mono">{}</span>"#,
            escape_html(&d.input_file.display().to_string())
        ),
    );
    row(
        &mut out,
        "Output JSON",
        &format!(
            r#"<span class="mono">{}</span>"#,
            escape_html(&d.output_json.display().to_string())
        ),
    );
    if let Some(i) = d.input {
        let format = match &i.vcf_mode {
            Some(mode) => format!("{} ({mode} mode)", i.format),
            None => i.format.clone(),
        };
        row(&mut out, "Input format", &escape_html(&format));
        row(
            &mut out,
            "Sequences",
            &format!(
                "{} analysed, {} left out",
                count(i.sequences as u64),
                count(i.dropped_sequences as u64)
            ),
        );
        let share = |n: u64| {
            if i.bins == 0 {
                "0.0%".to_string()
            } else {
                pct(n as f64 / i.bins as f64)
            }
        };
        row(
            &mut out,
            "Bins",
            &format!(
                "{} ({} heterozygous, {} missing)",
                count(i.bins),
                share(i.het_bins),
                share(i.missing_bins)
            ),
        );
    }
    row(&mut out, "EM iterations", &d.n_iter.to_string());
    row(
        &mut out,
        "Final log-likelihood",
        &d.final_loglike
            .map_or("-".to_string(), |ll| format!("{ll:.6}")),
    );
    row(
        &mut out,
        "Pattern",
        &escape_html(
            m.pattern
                .as_deref()
                .unwrap_or("none (one lambda per state)"),
        ),
    );
    row(
        &mut out,
        "States / free lambdas",
        &format!("{} / {}", m.n_steps + 1, m.lam.len()),
    );
    for (name, v) in [
        ("theta (per bin)", m.theta),
        ("rho (per bin)", m.rho),
        ("t_max (2N0 generations)", m.t_max),
        ("mu (per bp per generation)", m.mu),
        ("N0", n0),
        ("Bin size (bp)", d.bin_size),
        ("Generation time (years)", d.gen_years),
    ] {
        row(&mut out, name, &fmt_num(v));
    }
    for (name, value) in &d.settings {
        row(&mut out, name, &escape_html(value));
    }
    row(&mut out, "Curve points", &points.to_string());
    let [(x_min, x_max), (y_min, y_max)] = ranges;
    row(
        &mut out,
        "Time range (years)",
        &format!("{} - {}", fmt_num(x_min), fmt_num(x_max)),
    );
    row(
        &mut out,
        "Ne range",
        &format!("{} - {}", fmt_num(y_min), fmt_num(y_max)),
    );
    out
}

/// The input QC section: findings, then the key metrics and the
/// per-sequence table.
fn qc_section(r: &QcReport) -> String {
    let mut out = String::new();
    let warnings = r
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Warning)
        .count();
    let notes = r.findings.len() - warnings;
    let _ = writeln!(
        out,
        r#"<section class="card"><div class="meta"><h2>Input QC</h2>
<div class="qc-headline">{warnings} warning(s), {notes} note(s) on {} sequence(s), {} bins of {} bp</div>"#,
        count(r.n_sequences as u64),
        count(r.totals.bins),
        r.bin_size
    );
    if r.findings.is_empty() {
        out.push_str(r#"<div class="note">No warnings or notes.</div>"#);
    } else {
        out.push_str(r#"<ul class="findings">"#);
        for f in &r.findings {
            let sev = f.severity.as_str();
            let _ = writeln!(
                out,
                r#"<li class="finding {sev}"><span class="badge {sev}">{sev}</span> <code>{}</code> {}<div class="rec">{}</div></li>"#,
                f.code.as_str(),
                escape_html(&f.message),
                escape_html(&f.recommendation)
            );
        }
        out.push_str("</ul>");
    }

    out.push_str("<table>");
    let t = &r.totals;
    let opt = |v: Option<f64>, digits| v.map_or("n/a".to_string(), |v| sig(v, digits));
    row(
        &mut out,
        "Bins T / K / N",
        &format!(
            "{} / {} / {}",
            pct(t.t_fraction),
            pct(t.k_fraction),
            pct(t.missing_fraction)
        ),
    );
    row(&mut out, "theta0 per bin", &opt(t.theta0_per_bin, 4));
    row(&mut out, "Heterozygous sites per kb", &opt(t.het_per_kb, 3));
    let called = match t.called_bases {
        Some(n) => format!("{} bp (called sites)", count(n)),
        None => format!("{} bp (in called bins)", count(t.called_bases_from_bins)),
    };
    row(&mut out, "Called length", &called);
    row(
        &mut out,
        "N50 / L50",
        &format!(
            "{} bp / {}",
            count(r.lengths.n50_bases),
            count(r.lengths.l50 as u64)
        ),
    );
    if let Some(run) = &r.longest_missing_run {
        row(
            &mut out,
            "Longest missing run",
            &format!("{} bins in {}", count(run.bins), escape_html(&run.sequence)),
        );
    }
    if let Some(s) = &r.saturation {
        row(
            &mut out,
            "K bins with 2+ het sites",
            &format!("about {}", pct(s.multi_het_share)),
        );
    }
    if let Some(n) = r.expected_recombinations {
        row(
            &mut out,
            "Recombinations (C psmc's start)",
            &format!("about {}", count(n.round() as u64)),
        );
    }
    if let Some(trunks) = r.lengths.bootstrap_trunks {
        row(&mut out, "Bootstrap trunks", &count(trunks));
    }
    if let Some(dp) = &r.depth {
        let bounds = match (dp.min_depth, dp.max_depth) {
            (Some(lo), Some(hi)) => format!(", calls kept at {lo}-{hi}"),
            _ => String::new(),
        };
        row(
            &mut out,
            "Read depth",
            &format!(
                "mean {}, median {}{bounds}",
                sig(dp.mean, 3),
                sig(dp.median, 3)
            ),
        );
    }
    for d in &r.dropped {
        row(
            &mut out,
            "Left out",
            &format!("{} {}", count(d.count as u64), escape_html(&d.reason)),
        );
    }
    out.push_str("</table>");

    if !r.sequences.is_empty() {
        let _ = writeln!(
            out,
            r#"<details><summary>Per-sequence metrics ({} of {})</summary><table class="seq-table">
<tr><th>Sequence</th><th>Bins</th><th>K</th><th>N</th><th>Het/kb</th><th>vs others</th></tr>"#,
            r.sequences.len(),
            r.n_sequences
        );
        for s in &r.sequences {
            let kind = s
                .kind
                .map_or(String::new(), |k| format!(" ({})", k.label()));
            let _ = writeln!(
                out,
                "<tr><td>{}{kind}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                escape_html(&s.name),
                count(s.stats.bins),
                pct(s.stats.k_fraction),
                pct(s.stats.missing_fraction),
                opt(s.stats.het_per_kb, 3),
                s.het_vs_reference
                    .map_or("-".to_string(), |v| format!("{}x", sig(v, 2)))
            );
        }
        out.push_str("</table></details>");
    }
    out.push_str("</div></section>");
    out
}

/// The bootstrap section, with the coverage caveat measured in
/// [`crate::bootstrap`].
fn bootstrap_section(b: &BootstrapReport) -> String {
    let s = b.summary;
    let mut out =
        String::from(r#"<section class="card"><div class="meta"><h2>Bootstrap</h2><table>"#);
    row(&mut out, "Replicates", &count(s.n_replicates as u64));
    row(&mut out, "Trunk size (bins)", &count(s.block_size as u64));
    if let Some(n) = b.trunks {
        row(&mut out, "Trunks resampled", &count(n as u64));
    }
    row(
        &mut out,
        "EM iterations per replicate",
        &b.em_iters.to_string(),
    );
    row(&mut out, "Seed", &s.seed.to_string());
    let start = match b.started_from {
        Some(path) => format!("the parameters in {}", path.display()),
        None => "a fresh start".to_string(),
    };
    row(&mut out, "Replicates start from", &escape_html(&start));
    out.push_str("</table>");
    let (fitted, saved_note) = match b.started_from {
        Some(_) => (
            "each fitted from the saved parameters the main fit continued (as C psmc -i -b does)",
            " Those parameters were fitted to the whole input, so replicates that run few iterations from them stay close to them, and the band understates the uncertainty further.",
        ),
        None => ("each fitted from a fresh start", ""),
    };
    let _ = write!(
        out,
        r#"<div class="note caveat">The band spans the 2.5% to 97.5% quantiles of the replicate curves, {fitted} on trunks of the input resampled with replacement, as C psmc's bootstrap does. Read it as a lower bound on the uncertainty. In simulations (documented in psmc-rs's bootstrap module), the bootstrap standard deviation of log10 Ne was a median 0.90 to 0.92 of the true sampling standard deviation, and 0.74 at the oldest times; the estimator's bias, about half that standard deviation, does not show in the band at all. Few trunks give coarse bands.{saved_note}</div>"#,
    );
    out.push_str("</div></section>");
    out
}

/// Write the report of `data` to `output_html`.
pub fn write_html_report(output_html: &Path, data: &ReportData) -> Result<()> {
    let model = data.model;
    let echarts_script = echarts_script_tag()?;
    let curve = model.ne_curve(data.bin_size, data.gen_years)?;
    let n0 = model.n0_for_bin(data.bin_size);
    let main_data: Vec<[f64; 2]> = curve
        .iter()
        .copied()
        .filter(|&[x, y]| x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0)
        .collect();

    let range = |i: usize| {
        let lo = main_data
            .iter()
            .map(|p| p[i])
            .fold(f64::INFINITY, f64::min)
            .max(1e-12);
        let hi = main_data
            .iter()
            .map(|p| p[i])
            .fold(0.0f64, f64::max)
            .max(lo);
        (lo, hi)
    };
    let ranges = [range(0), range(1)];
    let points = main_data.len();

    let mut series = vec![JsSeries {
        name: "psmc-rs".to_string(),
        data: main_data,
    }];
    if let Some(b) = &data.bootstrap {
        let bs = b.summary;
        series.push(JsSeries {
            name: "Bootstrap q2.5".to_string(),
            data: bs.points.iter().map(|p| [p.x_years, p.ne_q025]).collect(),
        });
        series.push(JsSeries {
            name: "Bootstrap median".to_string(),
            data: bs.points.iter().map(|p| [p.x_years, p.ne_q500]).collect(),
        });
        series.push(JsSeries {
            name: "Bootstrap q97.5".to_string(),
            data: bs.points.iter().map(|p| [p.x_years, p.ne_q975]).collect(),
        });
    }
    let series_json = serde_json::to_string(&series)?;

    let title = "PSMC Inference Report";
    let input_name = data
        .input_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let meta_rows = meta_rows(data, n0, ranges, points);
    let qc_html = data.qc.map(qc_section).unwrap_or_default();
    let bootstrap_html = data
        .bootstrap
        .as_ref()
        .map(bootstrap_section)
        .unwrap_or_default();

    let (tmrca_panel, tmrca_script) = if let Some(tm) = data.tmrca {
        let tmrca_mean_json = serde_json::to_string(&tm.sampled_mean)?;
        let tmrca_map_json = serde_json::to_string(&tm.sampled_map)?;
        let tmrca_lo_json = serde_json::to_string(&tm.sampled_lo)?;
        let tmrca_hi_json = serde_json::to_string(&tm.sampled_hi)?;
        let total_mass: f64 = tm.state_mass.iter().sum();
        // Each state's interval of the time grid, in years.
        let state_bins: Vec<TmrcaStateBin> = tm
            .state_mass
            .iter()
            .enumerate()
            .map(|(k, &mass)| TmrcaStateBin {
                start_years: curve.get(k).map_or(f64::NAN, |p| p[0]),
                end_years: curve.get(k + 1).map(|p| p[0]),
                mass,
                mass_pct: if total_mass > 0.0 {
                    100.0 * mass / total_mass
                } else {
                    0.0
                },
            })
            .collect();
        let tmrca_state_bins_json = serde_json::to_string(&state_bins)?;
        let panel = format!(
            r#"
    <section class="card tmrca-card">
      <div class="chart-wrap">
        <div class="chart-head">
          <div class="chart-title">TMRCA Posterior Track</div>
          <div class="chart-actions">
            <label class="ctrl"><input id="ctrl-tmrca-map" type="checkbox" checked /> MAP</label>
            <label class="ctrl"><input id="ctrl-tmrca-ci" type="checkbox" checked /> 95% CI</label>
            <label class="ctrl">
              Y
              <select id="ctrl-tmrca-yscale">
                <option value="log" selected>log</option>
                <option value="value">linear</option>
              </select>
            </label>
            <button class="btn ghost" id="btn-tmrca-reset" type="button">Reset</button>
            <button class="btn" id="btn-tmrca-save" type="button">Save PNG</button>
          </div>
        </div>
        <div id="tmrca-track"></div>
      </div>
      <div class="chart-wrap compact">
        <div class="chart-head">
          <div class="chart-title">Posterior Mass by State Interval</div>
          <div class="chart-actions">
            <label class="ctrl">
              Metric
              <select id="ctrl-state-metric">
                <option value="pct" selected>percent</option>
                <option value="raw">raw mass</option>
              </select>
            </label>
            <button class="btn ghost" id="btn-state-reset" type="button">Reset</button>
            <button class="btn" id="btn-state-save" type="button">Save PNG</button>
          </div>
        </div>
        <div id="tmrca-state"></div>
      </div>
      <div class="meta tmrca-meta">
        <table>
          <tr><th>TMRCA rows</th><td>{}</td></tr>
          <tr><th>Sampled points</th><td>{}</td></tr>
          <tr><th>Posterior CI</th><td>q2.5 to q97.5 (per sampled bin)</td></tr>
          <tr><th>Sequences</th><td>{}</td></tr>
        </table>
      </div>
    </section>
"#,
            tm.total_sites,
            tm.sampled_mean.len(),
            tm.seq_count
        );
        let script = format!(
            r##"
    const tmrcaMeanRaw = {};
    const tmrcaMapRaw = {};
    const tmrcaLoRaw = {};
    const tmrcaHiRaw = {};
    const tmrcaStateBins = {};
    const positiveY = (pt) => Number.isFinite(pt[0]) && Number.isFinite(pt[1]) && pt[1] > 0;
    const tmrcaMean = tmrcaMeanRaw.filter(positiveY);
    const tmrcaMap = tmrcaMapRaw.filter(positiveY);
    const tmrcaLo = tmrcaLoRaw.filter(positiveY);
    const tmrcaHi = tmrcaHiRaw.filter(positiveY);
    const tmrcaTrack = echarts.init(document.getElementById("tmrca-track"), null, {{ renderer: "canvas" }});
    const tmrcaStateChart = echarts.init(document.getElementById("tmrca-state"), null, {{ renderer: "canvas" }});
    const tmrcaCtrl = {{
      map: document.getElementById("ctrl-tmrca-map"),
      ci: document.getElementById("ctrl-tmrca-ci"),
      yScale: document.getElementById("ctrl-tmrca-yscale"),
      stateMetric: document.getElementById("ctrl-state-metric")
    }};

    const fmtYears = (v) => {{
      if (!Number.isFinite(v)) return "NA";
      if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
      if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
      return v.toFixed(1);
    }};
    const tmrcaStateLabels = tmrcaStateBins.map((b) => {{
      const s = fmtYears(b.start_years);
      if (b.end_years === null || !Number.isFinite(b.end_years)) {{
        return `>= ${{s}}`;
      }}
      return `${{s}} - ${{fmtYears(b.end_years)}}`;
    }});
    const tmrcaStatePct = tmrcaStateBins.map((b) => b.mass_pct);
    const tmrcaStateRaw = tmrcaStateBins.map((b) => b.mass);
    const tickStep = Math.max(1, Math.ceil(tmrcaStateLabels.length / 18));

    const buildTmrcaTrackOption = (theme) => {{
      const yType = tmrcaCtrl.yScale ? tmrcaCtrl.yScale.value : "log";
      const showMap = tmrcaCtrl.map ? tmrcaCtrl.map.checked : true;
      const showCi = tmrcaCtrl.ci ? tmrcaCtrl.ci.checked : true;
      const widthSlider = document.getElementById("ctrl-main-width");
      const lineW = widthSlider ? Number(widthSlider.value) : 2.4;
      const trackSeries = [
        {{
          name: "mean",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMean,
          lineStyle: {{ width: Math.max(1.4, lineW - 0.4), color: theme.trackMean }},
          itemStyle: {{ color: theme.trackMean }}
        }}
      ];
      if (showMap) {{
        trackSeries.push({{
          name: "map",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMap,
          lineStyle: {{ width: Math.max(1.2, lineW - 0.6), color: theme.trackMap, type: "dashed" }},
          itemStyle: {{ color: theme.trackMap }}
        }});
      }}
      if (showCi) {{
        trackSeries.push(
          {{
            name: "q2.5",
            type: "line",
            showSymbol: false,
            symbol: "none",
            data: tmrcaLo,
            lineStyle: {{ width: 1.2, color: theme.ci, type: "dotted" }},
            itemStyle: {{ color: theme.ci }}
          }},
          {{
            name: "q97.5",
            type: "line",
            showSymbol: false,
            symbol: "none",
            data: tmrcaHi,
            lineStyle: {{ width: 1.2, color: theme.ci, type: "dotted" }},
            itemStyle: {{ color: theme.ci }}
          }}
        );
      }}

      return {{
        animationDuration: 650,
        animationEasing: "cubicOut",
        legend: {{
          top: 8,
          left: 14,
          textStyle: {{ color: "#223a52", fontSize: 12, fontFamily: "Sora, sans-serif" }}
        }},
        grid: {{ left: 84, right: 42, top: 44, bottom: 68, containLabel: true }},
        tooltip: {{
          trigger: "axis",
          axisPointer: {{ type: "cross" }},
          backgroundColor: "rgba(13, 22, 34, 0.90)",
          borderWidth: 0,
          textStyle: {{ color: "#f2f6fb" }}
        }},
        toolbox: {{
          right: 8,
          top: 6,
          itemSize: 14,
          feature: {{
            dataZoom: {{ yAxisIndex: "none" }},
            restore: {{}},
            saveAsImage: {{ pixelRatio: 2 }}
          }}
        }},
        xAxis: {{
          type: "value",
          name: "Genome bin index",
          nameLocation: "middle",
          nameGap: 34,
          axisLine: {{ lineStyle: {{ color: "#698198" }} }},
          axisLabel: {{ color: "#51697f" }},
          splitLine: {{ lineStyle: {{ color: "rgba(81,105,127,0.15)" }} }}
        }},
        yAxis: {{
          type: yType,
          name: "TMRCA (years)",
          nameLocation: "middle",
          nameRotate: 90,
          nameGap: 62,
          axisLine: {{ lineStyle: {{ color: "#698198" }} }},
          axisLabel: {{ color: "#51697f" }},
          splitLine: {{ lineStyle: {{ color: "rgba(81,105,127,0.15)" }} }}
        }},
        dataZoom: [
          {{ type: "inside", xAxisIndex: 0 }},
          {{
            type: "slider",
            xAxisIndex: 0,
            bottom: 16,
            height: 18,
            borderColor: "rgba(17,34,51,0.15)",
            backgroundColor: "rgba(255,255,255,0.62)",
            fillerColor: "rgba(66,165,245,0.20)",
            handleStyle: {{ color: theme.accent }}
          }}
        ],
        series: trackSeries
      }};
    }};

    const buildTmrcaStateOption = (theme) => {{
      const metric = tmrcaCtrl.stateMetric ? tmrcaCtrl.stateMetric.value : "pct";
      const isPct = metric !== "raw";
      const yData = isPct ? tmrcaStatePct : tmrcaStateRaw;
      const yName = isPct ? "Posterior mass (%)" : "Posterior mass";
      return {{
        animationDuration: 600,
        animationEasing: "cubicOut",
        tooltip: {{
          trigger: "axis",
          axisPointer: {{ type: "shadow" }},
          backgroundColor: "rgba(13, 22, 34, 0.90)",
          borderWidth: 0,
          textStyle: {{ color: "#f2f6fb" }},
          formatter: (params) => {{
            const p = Array.isArray(params) ? params[0] : params;
            const idx = p.dataIndex;
            const b = tmrcaStateBins[idx];
            if (!b) return "";
            const endTxt = (b.end_years === null || !Number.isFinite(b.end_years))
              ? "inf"
              : b.end_years.toExponential(3);
            return [
              `State ${{idx}}`,
              `Interval: ${{b.start_years.toExponential(3)}} - ${{endTxt}} years`,
              `Mass: ${{b.mass.toExponential(3)}}`,
              `Mass %: ${{b.mass_pct.toFixed(2)}}%`
            ].join("<br/>");
          }}
        }},
        toolbox: {{
          right: 8,
          top: 6,
          itemSize: 14,
          feature: {{
            dataZoom: {{ yAxisIndex: "none" }},
            restore: {{}},
            saveAsImage: {{ pixelRatio: 2 }}
          }}
        }},
        grid: {{ left: 84, right: 42, top: 32, bottom: 64, containLabel: true }},
        xAxis: {{
          type: "category",
          data: tmrcaStateLabels,
          name: "State interval (years)",
          nameLocation: "middle",
          nameGap: 44,
          axisLine: {{ lineStyle: {{ color: "#698198" }} }},
          axisLabel: {{
            color: "#51697f",
            interval: (idx) => idx % tickStep === 0,
            rotate: 24,
            fontSize: 10
          }},
          splitLine: {{ show: false }}
        }},
        yAxis: {{
          type: "value",
          name: yName,
          nameLocation: "middle",
          nameRotate: 90,
          nameGap: 56,
          axisLine: {{ lineStyle: {{ color: "#698198" }} }},
          axisLabel: {{ color: "#51697f" }},
          splitLine: {{ lineStyle: {{ color: "rgba(81,105,127,0.15)" }} }}
        }},
        dataZoom: [
          {{ type: "inside", xAxisIndex: 0 }},
          {{
            type: "slider",
            xAxisIndex: 0,
            bottom: 12,
            height: 16,
            borderColor: "rgba(17,34,51,0.15)",
            backgroundColor: "rgba(255,255,255,0.62)",
            fillerColor: "rgba(38,166,154,0.22)",
            handleStyle: {{ color: theme.accent2 }}
          }}
        ],
        series: [{{
          name: isPct ? "mass %" : "mass",
          type: "bar",
          data: yData,
          barMaxWidth: 24,
          itemStyle: {{
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              {{ offset: 0, color: theme.stateTop }},
              {{ offset: 1, color: theme.stateBottom }}
            ]),
            borderRadius: [4, 4, 0, 0]
          }}
        }}]
      }};
    }};

    const applyTmrcaCharts = (theme) => {{
      tmrcaTrack.setOption(buildTmrcaTrackOption(theme), true);
      tmrcaStateChart.setOption(buildTmrcaStateOption(theme), true);
    }};
    window.__psmcTmrca = {{
      track: tmrcaTrack,
      state: tmrcaStateChart,
      apply: applyTmrcaCharts
    }};
"##,
            tmrca_mean_json, tmrca_map_json, tmrca_lo_json, tmrca_hi_json, tmrca_state_bins_json
        );
        (panel, script)
    } else {
        (String::new(), String::new())
    };

    let template = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f5f8ff;
      --ink: #15253a;
      --muted: #58708c;
      --card: rgba(255,255,255,0.78);
      --stroke: rgba(17, 34, 51, 0.12);
      --shadow: 0 20px 70px rgba(27, 45, 72, 0.15);
      --accent: #f4511e;
      --accent2: #26a69a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(900px 480px at -8% -10%, rgba(244,81,30,0.18), transparent 70%),
        radial-gradient(980px 480px at 108% -10%, rgba(38,166,154,0.18), transparent 68%),
        linear-gradient(135deg, #f7f9ff, #edf6ff 52%, #f7fbff);
      font-family: "Sora", "Segoe UI", sans-serif;
      min-height: 100vh;
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 26px 18px 36px; }
    .hero h1 {
      margin: 0;
      font-size: clamp(1.45rem, 2.5vw, 2.5rem);
      letter-spacing: -0.02em;
      line-height: 1.08;
      font-weight: 800;
    }
    .hero .sub {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .card {
      margin-top: 18px;
      border-radius: 20px;
      border: 1px solid var(--stroke);
      box-shadow: var(--shadow);
      background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.74));
      backdrop-filter: blur(6px);
      overflow: hidden;
    }
    .control-card {
      padding: 12px 14px;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,251,255,0.78));
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px 12px;
      align-items: center;
    }
    .ctrl-block {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.83rem;
      color: #35506a;
      font-weight: 600;
    }
    .ctrl-block select,
    .ctrl-block input[type="range"] {
      width: 100%;
    }
    .ctrl-block select {
      border: 1px solid rgba(17, 34, 51, 0.18);
      border-radius: 8px;
      padding: 6px 8px;
      color: #1f354a;
      background: rgba(255,255,255,0.86);
      font-size: 0.82rem;
    }
    .ctrl-block input[type="checkbox"] {
      accent-color: #f4511e;
    }
    .ctrl-block .val {
      min-width: 32px;
      text-align: right;
      color: #1f354a;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.78rem;
    }
    .chart-wrap { padding: 12px 12px 0; }
    .chart-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin: 6px 8px 8px;
      flex-wrap: wrap;
    }
    .chart-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .ctrl {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #345168;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(21,37,58,0.12);
    }
    .ctrl select {
      border: 0;
      background: transparent;
      color: #26445d;
      font-size: 0.78rem;
      font-weight: 600;
      outline: none;
    }
    .btn {
      border: 1px solid rgba(17,34,51,0.16);
      background: linear-gradient(135deg, #f4511e, #ff7043);
      color: #fff;
      border-radius: 999px;
      padding: 6px 11px;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      cursor: pointer;
      transition: transform 0.14s ease, box-shadow 0.16s ease;
      box-shadow: 0 8px 18px rgba(244,81,30,0.25);
    }
    .btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 20px rgba(244,81,30,0.28);
    }
    .btn.ghost {
      background: rgba(255,255,255,0.78);
      color: #2f4a62;
      box-shadow: none;
    }
    .btn:disabled {
      opacity: 0.5;
      cursor: default;
      transform: none;
      box-shadow: none;
    }
    #chart {
      width: 100%;
      height: min(72vh, 760px);
      min-height: 410px;
      border-radius: 14px;
    }
    .meta {
      margin-top: 8px;
      border-top: 1px solid var(--stroke);
      padding: 14px 16px 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.32));
    }
    .tmrca-card { margin-top: 18px; }
    .chart-title {
      color: #2a425b;
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    #tmrca-track {
      width: 100%;
      height: min(46vh, 420px);
      min-height: 300px;
      border-radius: 12px;
    }
    #tmrca-state {
      width: 100%;
      height: min(34vh, 260px);
      min-height: 200px;
      border-radius: 12px;
    }
    .chart-wrap.compact {
      border-top: 1px solid rgba(17, 34, 51, 0.08);
      padding-top: 10px;
    }
    .tmrca-meta {
      margin-top: 2px;
      border-top: 1px solid rgba(17, 34, 51, 0.08);
    }
    .meta h2 {
      margin: 0 0 12px;
      font-size: 1rem;
      font-weight: 700;
      color: #23384e;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td {
      padding: 8px 8px;
      border-bottom: 1px dashed rgba(21,37,58,0.12);
      text-align: left;
      font-size: 0.9rem;
      vertical-align: top;
    }
    th {
      width: 180px;
      color: #35506a;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.78rem;
    }
    .mono {
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.78rem;
      color: #4a637d;
      word-break: break-all;
    }
    .note {
      margin-top: 10px;
      color: #5d7289;
      font-size: 0.83rem;
    }
    .offline {
      padding: 14px 18px;
      border-color: rgba(198, 40, 40, 0.45);
      background: #fff4f2;
      color: #7f1d1d;
      font-size: 0.92rem;
      line-height: 1.5;
    }
    .offline[hidden] { display: none; }
    .qc-headline { color: #35506a; font-weight: 600; margin-bottom: 10px; }
    .findings { list-style: none; margin: 0 0 14px; padding: 0; }
    .finding {
      padding: 10px 12px;
      margin-bottom: 8px;
      border-radius: 12px;
      border: 1px solid var(--stroke);
      background: rgba(255,255,255,0.7);
      font-size: 0.88rem;
      line-height: 1.45;
    }
    .finding.warning { border-left: 4px solid #e65100; }
    .finding.info { border-left: 4px solid #1e88e5; }
    .finding .rec { margin-top: 6px; color: #4a637d; }
    .badge {
      display: inline-block;
      padding: 1px 8px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #fff;
    }
    .badge.warning { background: #e65100; }
    .badge.info { background: #1e88e5; }
    details { margin-top: 12px; }
    summary { cursor: pointer; font-weight: 600; color: #35506a; }
    .seq-table th { width: auto; }
    .caveat {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255, 243, 224, 0.8);
      line-height: 1.5;
    }
    @media (max-width: 900px) {
      .wrap { padding: 16px 10px 22px; }
      #chart { min-height: 360px; }
      .chart-head { margin-bottom: 6px; }
      .chart-actions { gap: 6px; }
      .ctrl { font-size: 0.72rem; }
      .btn { font-size: 0.72rem; padding: 5px 10px; }
      th, td { font-size: 0.84rem; }
      th { width: 130px; font-size: 0.72rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>__TITLE__</h1>
      <div class="sub">Input: __INPUT_NAME__ · psmc-rs __VERSION__ · Interactive report with per-chart export.</div>
    </header>

    <section class="card offline" id="echarts-missing" hidden>
      <strong>The charts could not be drawn.</strong> This report loads its chart library,
      ECharts, from cdn.jsdelivr.net, and it did not load (no network connection, or the site
      is blocked). Open the report on a computer with internet access, or rerun psmc-rs with
      the environment variable <code>PSMC_ECHARTS_JS</code> set to a local copy of
      <code>echarts.min.js</code>, which is then embedded in the report. The tables below do
      not need it.
    </section>

    <section class="card control-card">
      <div class="control-grid">
        <label class="ctrl-block">Theme
          <select id="ctrl-theme">
            <option value="sunrise" selected>Sunrise</option>
            <option value="tealglass">Teal Glass</option>
            <option value="graphite">Graphite</option>
          </select>
        </label>
        <label class="ctrl-block">Line mode
          <select id="ctrl-main-mode">
            <option value="step" selected>PSMC step</option>
            <option value="linear">linear</option>
            <option value="smooth">smooth</option>
          </select>
        </label>
        <label class="ctrl-block">Main X scale
          <select id="ctrl-main-xscale">
            <option value="log" selected>log</option>
            <option value="value">linear</option>
          </select>
        </label>
        <label class="ctrl-block">Main Y scale
          <select id="ctrl-main-yscale">
            <option value="value" selected>linear</option>
            <option value="log">log</option>
          </select>
        </label>
        <label class="ctrl-block">Line width
          <input id="ctrl-main-width" type="range" min="1.2" max="5.0" step="0.2" value="3.0" />
          <span class="val" id="ctrl-main-width-val">3.0</span>
        </label>
        <label class="ctrl-block"><input id="ctrl-main-area" type="checkbox" checked /> fill area</label>
      </div>
    </section>

    <section class="card">
      <div class="chart-wrap">
        <div class="chart-head">
          <div class="chart-title">Estimated Effective Population Size</div>
          <div class="chart-actions">
            <button class="btn ghost" id="btn-main-reset" type="button">Reset</button>
            <button class="btn" id="btn-main-save" type="button">Save PNG</button>
            <button class="btn" id="btn-save-all" type="button">Save All</button>
          </div>
        </div>
        <div id="chart"></div>
      </div>
      <div class="meta">
        <h2>Run Summary</h2>
        <table>
          __META_ROWS__
        </table>
        <div class="note">Use controls above to switch theme, axis scale, line style, and export each chart independently.</div>
      </div>
    </section>

    __QC_SECTION__

    __BOOTSTRAP_SECTION__

    __TMRCA_PANEL__
  </div>

  __ECHARTS_SCRIPT__
  <script>
    if (typeof echarts === "undefined") {
      document.getElementById("echarts-missing").hidden = false;
      throw new Error("ECharts did not load; see the message at the top of the report.");
    }
    const series = __SERIES_JSON__;
    const STYLE_KEY = "psmc_report_style_v2";
    const themes = {
      sunrise: {
        accent: "#f4511e",
        accent2: "#26a69a",
        mainLine: "#f4511e",
        mainArea: "rgba(244,81,30,0.10)",
        trackMean: "#1e88e5",
        trackMap: "#8e24aa",
        ci: "#64b5f6",
        stateTop: "#26a69a",
        stateBottom: "#80cbc4"
      },
      tealglass: {
        accent: "#00897b",
        accent2: "#00796b",
        mainLine: "#00897b",
        mainArea: "rgba(0,137,123,0.12)",
        trackMean: "#0277bd",
        trackMap: "#1565c0",
        ci: "#4fc3f7",
        stateTop: "#00acc1",
        stateBottom: "#80deea"
      },
      graphite: {
        accent: "#455a64",
        accent2: "#546e7a",
        mainLine: "#37474f",
        mainArea: "rgba(55,71,79,0.10)",
        trackMean: "#455a64",
        trackMap: "#263238",
        ci: "#78909c",
        stateTop: "#546e7a",
        stateBottom: "#90a4ae"
      }
    };
    const getEl = (id) => document.getElementById(id);
    const controls = {
      theme: getEl("ctrl-theme"),
      mode: getEl("ctrl-main-mode"),
      xScale: getEl("ctrl-main-xscale"),
      yScale: getEl("ctrl-main-yscale"),
      width: getEl("ctrl-main-width"),
      widthVal: getEl("ctrl-main-width-val"),
      area: getEl("ctrl-main-area")
    };

    const exportStamp = new Date().toISOString().replace(/[:.]/g, "-");
    const exportChartPng = (targetChart, name) => {
      const url = targetChart.getDataURL({
        type: "png",
        pixelRatio: 3,
        backgroundColor: "#f7fbff"
      });
      const a = document.createElement("a");
      a.href = url;
      a.download = `${name}_${exportStamp}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    };
    const resetZoom = (targetChart) => {
      targetChart.dispatchAction({ type: "dataZoom", start: 0, end: 100 });
    };

    const chart = echarts.init(document.getElementById("chart"), null, { renderer: "canvas" });
    const rgba = (hex, alpha) => {
      const h = String(hex || "").trim();
      const m = /^#?([0-9a-f]{6})$/i.exec(h);
      if (!m) return `rgba(100,181,246,${alpha})`;
      const s = m[1];
      const r = parseInt(s.slice(0, 2), 16);
      const g = parseInt(s.slice(2, 4), 16);
      const b = parseInt(s.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    };

    const sanitizeScale = (v, fallback) => (v === "log" || v === "value" ? v : fallback);
    const sanitizeMode = (v) => (v === "step" || v === "linear" || v === "smooth" ? v : "step");
    const numeric = (v, fallback) => {
      const n = Number(v);
      return Number.isFinite(n) ? n : fallback;
    };
    const readConfig = () => ({
      theme: controls.theme ? controls.theme.value : "sunrise",
      mode: sanitizeMode(controls.mode ? controls.mode.value : "step"),
      xScale: sanitizeScale(controls.xScale ? controls.xScale.value : "log", "log"),
      yScale: sanitizeScale(controls.yScale ? controls.yScale.value : "value", "value"),
      width: Math.min(5.0, Math.max(1.2, numeric(controls.width ? controls.width.value : 3.0, 3.0))),
      area: controls.area ? controls.area.checked : true
    });
    const saveConfig = (cfg) => {
      try {
        localStorage.setItem(STYLE_KEY, JSON.stringify(cfg));
      } catch (_) {}
    };
    const restoreConfig = () => {
      try {
        const raw = localStorage.getItem(STYLE_KEY);
        if (!raw) return;
        const cfg = JSON.parse(raw);
        if (controls.theme && cfg.theme) controls.theme.value = cfg.theme;
        if (controls.mode && cfg.mode) controls.mode.value = sanitizeMode(cfg.mode);
        if (controls.xScale && cfg.xScale) controls.xScale.value = sanitizeScale(cfg.xScale, "log");
        if (controls.yScale && cfg.yScale) controls.yScale.value = sanitizeScale(cfg.yScale, "value");
        if (controls.width && Number.isFinite(Number(cfg.width))) controls.width.value = String(cfg.width);
        if (controls.area && typeof cfg.area === "boolean") controls.area.checked = cfg.area;
      } catch (_) {}
    };

    const buildMainOption = (cfg, palette) => {
      const mode = cfg.mode;
      const stepValue = mode === "step" ? "end" : false;
      const smoothValue = mode === "smooth";
      const areaStyle = cfg.area ? { color: palette.mainArea } : undefined;
      const findSeries = (name) => {
        const key = String(name || "").toLowerCase();
        return series.find((s) => String(s.name || "").toLowerCase() === key);
      };
      const sMain = findSeries("psmc-rs");
      const sQ025 = findSeries("bootstrap q2.5");
      const sQ500 = findSeries("bootstrap median");
      const sQ975 = findSeries("bootstrap q97.5");
      const mainData = sMain ? sMain.data : [];
      const q025Data = sQ025 ? sQ025.data : [];
      const q500Data = sQ500 ? sQ500.data : [];
      const q975Data = sQ975 ? sQ975.data : [];
      const hasRibbon = q025Data.length > 0 && q975Data.length > 0;

      const ribbonBase = hasRibbon ? q025Data : [];
      const ribbonDelta = hasRibbon
        ? q975Data.map((p, i) => {
            const lo = (q025Data[i] && Number.isFinite(q025Data[i][1])) ? q025Data[i][1] : p[1];
            return [p[0], Math.max(0, p[1] - lo)];
          })
        : [];
      const ciBandColor = rgba(palette.ci, 0.22);
      const legendData = ["psmc-rs"];
      if (q500Data.length > 0) legendData.push("Bootstrap median");
      if (hasRibbon) legendData.push("Bootstrap 95% CI");

      const mainSeries = [];
      if (hasRibbon) {
        mainSeries.push({
          name: "__Bootstrap CI base",
          type: "line",
          stack: "bootstrap_ci",
          step: stepValue,
          data: ribbonBase,
          showSymbol: false,
          symbol: "none",
          lineStyle: { width: 0, opacity: 0 },
          areaStyle: { opacity: 0 },
          silent: true,
          tooltip: { show: false }
        });
        mainSeries.push({
          name: "Bootstrap 95% CI",
          type: "line",
          stack: "bootstrap_ci",
          step: stepValue,
          data: ribbonDelta,
          showSymbol: false,
          symbol: "none",
          lineStyle: { width: 0, opacity: 0 },
          areaStyle: { color: ciBandColor },
          silent: true
        });
      }
      if (q500Data.length > 0) {
        mainSeries.push({
          name: "Bootstrap median",
          type: "line",
          data: q500Data,
          step: stepValue,
          smooth: false,
          showSymbol: false,
          symbol: "none",
          lineStyle: {
            width: Math.max(1.2, cfg.width - 1.0),
            color: palette.trackMap,
            type: "dashed"
          },
          itemStyle: { color: palette.trackMap }
        });
      }
      mainSeries.push({
        name: "psmc-rs",
        type: "line",
        data: mainData,
        step: stepValue,
        smooth: smoothValue,
        showSymbol: false,
        symbol: "none",
        areaStyle,
        lineStyle: { width: cfg.width, color: palette.mainLine },
        itemStyle: { color: palette.mainLine }
      });
      return {
        animationDuration: 750,
        animationEasing: "cubicOut",
        backgroundColor: "transparent",
        legend: {
          data: legendData,
          top: 14,
          left: 18,
          textStyle: { color: "#223a52", fontSize: 13, fontFamily: "Sora, sans-serif" }
        },
        grid: { left: 96, right: 58, top: 82, bottom: 98, containLabel: true },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross" },
          backgroundColor: "rgba(13, 22, 34, 0.90)",
          borderWidth: 0,
          textStyle: { color: "#f2f6fb" }
        },
        toolbox: {
          right: 16,
          top: 12,
          itemSize: 15,
          feature: {
            dataZoom: { yAxisIndex: "none" },
            restore: {},
            saveAsImage: { pixelRatio: 2 }
          }
        },
        xAxis: {
          type: cfg.xScale,
          name: "Years",
          nameLocation: "middle",
          nameGap: 38,
          nameTextStyle: { color: "#334f67", fontWeight: 600 },
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: { color: "#51697f" },
          splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } },
          minorSplitLine: { show: cfg.xScale === "log", lineStyle: { color: "rgba(81,105,127,0.08)" } }
        },
        yAxis: {
          type: cfg.yScale,
          name: "Effective population size (Ne)",
          nameLocation: "middle",
          nameRotate: 90,
          nameGap: 74,
          nameTextStyle: { color: "#334f67", fontWeight: 600 },
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: { color: "#51697f" },
          splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } }
        },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          {
            type: "slider",
            xAxisIndex: 0,
            bottom: 18,
            height: 20,
            borderColor: "rgba(17,34,51,0.15)",
            backgroundColor: "rgba(255,255,255,0.62)",
            fillerColor: palette.mainArea,
            handleStyle: { color: palette.accent }
          }
        ],
        series: mainSeries
      };
    };

    __TMRCA_SCRIPT__
    const applyAllCharts = () => {
      const cfg = readConfig();
      const palette = themes[cfg.theme] || themes.sunrise;
      window.__psmcTheme = palette;
      if (controls.widthVal) controls.widthVal.textContent = cfg.width.toFixed(1);
      chart.setOption(buildMainOption(cfg, palette), true);
      if (window.__psmcTmrca && typeof window.__psmcTmrca.apply === "function") {
        window.__psmcTmrca.apply(palette);
      }
      saveConfig(cfg);
    };

    const bindChange = (id, eventName = "change") => {
      const el = getEl(id);
      if (el) el.addEventListener(eventName, applyAllCharts);
    };
    bindChange("ctrl-theme");
    bindChange("ctrl-main-mode");
    bindChange("ctrl-main-xscale");
    bindChange("ctrl-main-yscale");
    bindChange("ctrl-main-area");
    bindChange("ctrl-main-width", "input");
    bindChange("ctrl-tmrca-map");
    bindChange("ctrl-tmrca-ci");
    bindChange("ctrl-tmrca-yscale");
    bindChange("ctrl-state-metric");

    const btnMainSave = getEl("btn-main-save");
    if (btnMainSave) btnMainSave.addEventListener("click", () => exportChartPng(chart, "psmc_ne_curve"));
    const btnMainReset = getEl("btn-main-reset");
    if (btnMainReset) btnMainReset.addEventListener("click", () => resetZoom(chart));
    const btnSaveAll = getEl("btn-save-all");
    if (btnSaveAll) {
      btnSaveAll.addEventListener("click", () => {
        exportChartPng(chart, "psmc_ne_curve");
        if (window.__psmcTmrca && window.__psmcTmrca.track) {
          setTimeout(() => exportChartPng(window.__psmcTmrca.track, "psmc_tmrca_track"), 160);
        }
        if (window.__psmcTmrca && window.__psmcTmrca.state) {
          setTimeout(() => exportChartPng(window.__psmcTmrca.state, "psmc_tmrca_state"), 320);
        }
      });
    }
    const btnTmrcaSave = getEl("btn-tmrca-save");
    if (btnTmrcaSave) {
      btnTmrcaSave.addEventListener("click", () => {
        if (window.__psmcTmrca && window.__psmcTmrca.track) exportChartPng(window.__psmcTmrca.track, "psmc_tmrca_track");
      });
    }
    const btnTmrcaReset = getEl("btn-tmrca-reset");
    if (btnTmrcaReset) {
      btnTmrcaReset.addEventListener("click", () => {
        if (window.__psmcTmrca && window.__psmcTmrca.track) resetZoom(window.__psmcTmrca.track);
      });
    }
    const btnStateSave = getEl("btn-state-save");
    if (btnStateSave) {
      btnStateSave.addEventListener("click", () => {
        if (window.__psmcTmrca && window.__psmcTmrca.state) exportChartPng(window.__psmcTmrca.state, "psmc_tmrca_state");
      });
    }
    const btnStateReset = getEl("btn-state-reset");
    if (btnStateReset) {
      btnStateReset.addEventListener("click", () => {
        if (window.__psmcTmrca && window.__psmcTmrca.state) resetZoom(window.__psmcTmrca.state);
      });
    }

    restoreConfig();
    applyAllCharts();
    window.addEventListener("resize", () => {
      chart.resize();
      if (window.__psmcTmrca && window.__psmcTmrca.track) window.__psmcTmrca.track.resize();
      if (window.__psmcTmrca && window.__psmcTmrca.state) window.__psmcTmrca.state.resize();
    });
  </script>
</body>
</html>
"##;

    let html = template
        .replace("__TITLE__", &escape_html(title))
        .replace("__INPUT_NAME__", &escape_html(input_name))
        .replace("__VERSION__", env!("CARGO_PKG_VERSION"))
        .replace("__META_ROWS__", &meta_rows)
        .replace("__QC_SECTION__", &qc_html)
        .replace("__BOOTSTRAP_SECTION__", &bootstrap_html)
        .replace("__TMRCA_PANEL__", &tmrca_panel)
        .replace("__TMRCA_SCRIPT__", &tmrca_script)
        .replace("__SERIES_JSON__", &series_json)
        .replace("__ECHARTS_SCRIPT__", &echarts_script);
    fs::write(output_html, html)
        .with_context(|| format!("failed to write {}", output_html.display()))?;
    Ok(())
}
