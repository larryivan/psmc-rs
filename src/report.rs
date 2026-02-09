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
        let mut state_mass_data = Vec::<[f64; 2]>::with_capacity(tm.state_mass.len());
        for (k, mass) in tm.state_mass.iter().enumerate() {
            let x = if k < tm.state_years.len() {
                tm.state_years[k]
            } else {
                k as f64
            };
            state_mass_data.push([x, *mass]);
        }
        let tmrca_state_json = serde_json::to_string(&state_mass_data)?;

        let panel = format!(
            r#"
    <section class="card tmrca-card">
      <div class="chart-wrap">
        <div class="chart-title">TMRCA Posterior Track</div>
        <div id="tmrca-track"></div>
      </div>
      <div class="chart-wrap compact">
        <div class="chart-title">Posterior Mass by State</div>
        <div id="tmrca-state"></div>
      </div>
      <div class="meta tmrca-meta">
        <table>
          <tr><th>TMRCA rows</th><td>{}</td></tr>
          <tr><th>Sampled points</th><td>{}</td></tr>
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
    const tmrcaMean = {};
    const tmrcaMap = {};
    const tmrcaState = {};
    const tmrcaTrack = echarts.init(document.getElementById("tmrca-track"), null, {{ renderer: "canvas" }});
    tmrcaTrack.setOption({{
      animationDuration: 650,
      animationEasing: "cubicOut",
      legend: {{
        top: 8,
        left: 14,
        textStyle: {{ color: "#223a52", fontSize: 12, fontFamily: "Sora, sans-serif" }}
      }},
      grid: {{ left: 84, right: 36, top: 44, bottom: 68, containLabel: true }},
      tooltip: {{
        trigger: "axis",
        axisPointer: {{ type: "cross" }},
        backgroundColor: "rgba(13, 22, 34, 0.90)",
        borderWidth: 0,
        textStyle: {{ color: "#f2f6fb" }}
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
        type: "log",
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
          handleStyle: {{ color: "#42a5f5" }}
        }}
      ],
      series: [
        {{
          name: "mean",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMean,
          lineStyle: {{ width: 2.2, color: "#1E88E5" }},
          itemStyle: {{ color: "#1E88E5" }}
        }},
        {{
          name: "map",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMap,
          lineStyle: {{ width: 2.0, color: "#8E24AA", type: "dashed" }},
          itemStyle: {{ color: "#8E24AA" }}
        }}
      ]
    }});

    const tmrcaStateChart = echarts.init(document.getElementById("tmrca-state"), null, {{ renderer: "canvas" }});
    tmrcaStateChart.setOption({{
      animationDuration: 600,
      animationEasing: "cubicOut",
      tooltip: {{
        trigger: "axis",
        axisPointer: {{ type: "line" }},
        backgroundColor: "rgba(13, 22, 34, 0.90)",
        borderWidth: 0,
        textStyle: {{ color: "#f2f6fb" }}
      }},
      grid: {{ left: 84, right: 36, top: 32, bottom: 44, containLabel: true }},
      xAxis: {{
        type: "log",
        name: "State time (years)",
        nameLocation: "middle",
        nameGap: 30,
        axisLine: {{ lineStyle: {{ color: "#698198" }} }},
        axisLabel: {{ color: "#51697f" }},
        splitLine: {{ lineStyle: {{ color: "rgba(81,105,127,0.15)" }} }}
      }},
      yAxis: {{
        type: "value",
        name: "Posterior mass",
        nameLocation: "middle",
        nameRotate: 90,
        nameGap: 56,
        axisLine: {{ lineStyle: {{ color: "#698198" }} }},
        axisLabel: {{ color: "#51697f" }},
        splitLine: {{ lineStyle: {{ color: "rgba(81,105,127,0.15)" }} }}
      }},
      series: [{{
        type: "line",
        showSymbol: false,
        symbol: "none",
        data: tmrcaState,
        lineStyle: {{ width: 2.2, color: "#26A69A" }},
        areaStyle: {{ color: "rgba(38,166,154,0.18)" }},
        itemStyle: {{ color: "#26A69A" }}
      }}]
    }});
"##,
            tmrca_mean_json, tmrca_map_json, tmrca_state_json
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
      background: var(--card);
      backdrop-filter: blur(6px);
      overflow: hidden;
    }
    .chart-wrap { padding: 12px 12px 0; }
    #chart {
      width: 100%;
      height: min(74vh, 760px);
      min-height: 420px;
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
      margin: 8px 8px 6px;
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
      th, td { font-size: 0.84rem; }
      th { width: 130px; font-size: 0.72rem; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>__TITLE__</h1>
      <div class="sub">Input: __INPUT_NAME__ · Auto-generated from `psmc-rs` CLI · Interactive zoom and PNG export enabled.</div>
    </header>

    <section class="card">
      <div class="chart-wrap">
        <div id="chart"></div>
      </div>
      <div class="meta">
        <h2>Run Summary</h2>
        <table>
          __META_ROWS__
        </table>
        <div class="note">X axis uses log scale in years. Curve is rendered as a post-step line to match PSMC style.</div>
      </div>
    </section>

    __TMRCA_PANEL__
  </div>

  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <script>
    const series = __SERIES_JSON__;
    const chart = echarts.init(document.getElementById("chart"), null, { renderer: "canvas" });
    chart.setOption({
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
        feature: {
          dataZoom: { yAxisIndex: "none" },
          restore: {},
          saveAsImage: { pixelRatio: 2 }
        }
      },
      xAxis: {
        type: "log",
        name: "Years",
        nameLocation: "middle",
        nameGap: 38,
        nameTextStyle: { color: "#334f67", fontWeight: 600 },
        axisLine: { lineStyle: { color: "#698198" } },
        axisLabel: { color: "#51697f" },
        splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } },
        minorSplitLine: { show: true, lineStyle: { color: "rgba(81,105,127,0.08)" } }
      },
      yAxis: {
        type: "value",
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
          fillerColor: "rgba(244,81,30,0.18)",
          handleStyle: { color: "#f4511e" }
        }
      ],
      series: series.map(s => ({
        ...s,
        type: "line",
        step: "end",
        showSymbol: false,
        symbol: "none",
        lineStyle: { width: 3.0, color: "#f4511e" },
        itemStyle: { color: "#f4511e" }
      }))
    });
    __TMRCA_SCRIPT__
    window.addEventListener("resize", () => {
      chart.resize();
      if (typeof tmrcaTrack !== "undefined") tmrcaTrack.resize();
      if (typeof tmrcaStateChart !== "undefined") tmrcaStateChart.resize();
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
