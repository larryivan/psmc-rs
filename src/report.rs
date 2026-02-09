use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;

use crate::hmm::TmrcaReportData;
use crate::model::PsmcModel;

const REPORT_GEN_YEARS: f64 = 25.0;

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

fn default_title() -> String {
    "PSMC Inference Report".to_string()
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

pub fn default_html_path(output_json: &Path) -> std::path::PathBuf {
    output_json.with_extension("html")
}

pub fn write_html_report(
    model: &PsmcModel,
    input_file: &Path,
    output_json: &Path,
    output_html: &Path,
    n_iter: usize,
    input_format: &str,
    bin_size: f64,
    tmrca: Option<&TmrcaReportData>,
) -> Result<()> {
    let lam_full = model.map_lam(&model.lam)?;
    let t = model.compute_t(0.1);
    let n0 = model.theta / (4.0 * model.mu * bin_size);

    let mut data = Vec::with_capacity(model.n_steps + 1);
    for k in 0..=model.n_steps {
        let x = t[k] * 2.0 * n0 * REPORT_GEN_YEARS;
        let y = lam_full[k] * n0;
        if x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0 {
            data.push([x, y]);
        }
    }

    let y_min = data
        .iter()
        .map(|p| p[1])
        .fold(f64::INFINITY, f64::min)
        .max(1e-12);
    let y_max = data.iter().map(|p| p[1]).fold(0.0f64, f64::max).max(y_min);
    let x_min = data
        .iter()
        .map(|p| p[0])
        .fold(f64::INFINITY, f64::min)
        .max(1e-12);
    let x_max = data.iter().map(|p| p[0]).fold(0.0f64, f64::max).max(x_min);

    let series_json = serde_json::to_string(&vec![JsSeries {
        name: "Rust estimate".to_string(),
        data,
    }])?;

    let title = default_title();
    let input_name = input_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let meta_rows = format!(
        r#"
<tr><th>Input</th><td class="mono">{}</td></tr>
<tr><th>Output JSON</th><td class="mono">{}</td></tr>
<tr><th>Input format</th><td>{}</td></tr>
<tr><th>EM iterations</th><td>{}</td></tr>
<tr><th>n_steps</th><td>{}</td></tr>
<tr><th>theta</th><td>{}</td></tr>
<tr><th>rho</th><td>{}</td></tr>
<tr><th>t_max</th><td>{}</td></tr>
<tr><th>mu</th><td>{}</td></tr>
<tr><th>N0</th><td>{}</td></tr>
<tr><th>bin size</th><td>{}</td></tr>
<tr><th>gen years</th><td>{}</td></tr>
<tr><th>curve points</th><td>{}</td></tr>
<tr><th>x range (years)</th><td>{} - {}</td></tr>
<tr><th>y range (Ne)</th><td>{} - {}</td></tr>
"#,
        escape_html(&input_file.display().to_string()),
        escape_html(&output_json.display().to_string()),
        escape_html(input_format),
        n_iter,
        model.n_steps,
        fmt_num(model.theta),
        fmt_num(model.rho),
        fmt_num(model.t_max),
        fmt_num(model.mu),
        fmt_num(n0),
        fmt_num(bin_size),
        fmt_num(REPORT_GEN_YEARS),
        model.n_steps + 1,
        fmt_num(x_min),
        fmt_num(x_max),
        fmt_num(y_min),
        fmt_num(y_max)
    );

    let (tmrca_panel, tmrca_script) = if let Some(tm) = tmrca {
        let tmrca_mean_json = serde_json::to_string(&tm.sampled_mean)?;
        let tmrca_map_json = serde_json::to_string(&tm.sampled_map)?;
        let tmrca_lo_json = serde_json::to_string(&tm.sampled_lo)?;
        let tmrca_hi_json = serde_json::to_string(&tm.sampled_hi)?;
        let total_mass: f64 = tm.state_mass.iter().sum();
        let mut state_bins = Vec::<TmrcaStateBin>::with_capacity(tm.state_mass.len());
        for k in 0..tm.state_mass.len() {
            let start_years = tm.state_years.get(k).copied().unwrap_or(k as f64);
            let end_years = tm.state_years.get(k + 1).copied();
            let mass = tm.state_mass[k];
            let mass_pct = if total_mass > 0.0 {
                100.0 * mass / total_mass
            } else {
                0.0
            };
            state_bins.push(TmrcaStateBin {
                start_years,
                end_years,
                mass,
                mass_pct,
            });
        }
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
      <div class="sub">Input: __INPUT_NAME__ · Auto-generated from `psmc-rs` CLI · Interactive report with per-chart export.</div>
    </header>

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

    __TMRCA_PANEL__
  </div>

  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script>
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
      return {
        animationDuration: 750,
        animationEasing: "cubicOut",
        backgroundColor: "transparent",
        legend: {
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
        series: series.map((s) => ({
          ...s,
          type: "line",
          step: stepValue,
          smooth: smoothValue,
          showSymbol: false,
          symbol: "none",
          areaStyle,
          lineStyle: { width: cfg.width, color: palette.mainLine },
          itemStyle: { color: palette.mainLine }
        }))
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
        .replace("__TITLE__", &escape_html(&title))
        .replace("__INPUT_NAME__", &escape_html(input_name))
        .replace("__META_ROWS__", &meta_rows)
        .replace("__TMRCA_PANEL__", &tmrca_panel)
        .replace("__TMRCA_SCRIPT__", &tmrca_script)
        .replace("__SERIES_JSON__", &series_json);
    fs::write(output_html, html)?;
    Ok(())
}
