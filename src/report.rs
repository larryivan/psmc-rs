//! The interactive HTML report of a fit: the Ne curve (with the bootstrap
//! band), the run's settings, input QC, and the TMRCA panels.
//!
//! The charts use ECharts, loaded from a CDN unless `PSMC_ECHARTS_JS` names
//! a local copy to embed; without it the report says how to fix that.

use anyhow::{Context, Result};
use serde::Serialize;
use std::fmt::{Display, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

use crate::bootstrap::{BootstrapCurvePoint, BootstrapSummary};
use crate::hmm::TmrcaReportData;
use crate::io::params::{BinCallableRecord, CALLABLE_LENGTH_MODEL, InputSummary};
use crate::model::PsmcModel;
use crate::qc::fmt::{count, counted, pct, sig, sig_or_na};
use crate::qc::{QcReport, fraction};

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
    /// Log-likelihood of the fitted parameters (of the starting ones after 0
    /// iterations), if known.
    pub final_loglike: Option<f64>,
    /// Further settings of the run to list, as (name, value).
    pub settings: Vec<(String, String)>,
    /// Input QC of the analysed data.
    pub qc: Option<&'a QcReport>,
    /// How the fit used the callable bases of each bin, when known.
    pub bin_callable: Option<&'a BinCallableRecord>,
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
    name: &'static str,
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

/// Check that the ECharts bundle `PSMC_ECHARTS_JS` names, if any, can be
/// read, as [`write_html_report`] will read it. A run calls this before its
/// work, so that a wrong path fails at once rather than after the fit and
/// the bootstrap, with no report written.
pub fn check_echarts_bundle() -> Result<()> {
    echarts_script_tag().map(drop)
}

/// The `<script>` that loads ECharts: the bundle `PSMC_ECHARTS_JS` names,
/// embedded, or the CDN.
fn echarts_script_tag() -> Result<String> {
    const CDN: &str =
        r#"<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>"#;
    let Ok(path) = std::env::var("PSMC_ECHARTS_JS") else {
        return Ok(CDN.to_string());
    };
    let js = fs::read_to_string(&path)
        .with_context(|| format!("failed to read ECharts bundle from PSMC_ECHARTS_JS={path}"))?;
    Ok(format!("<script>\n{js}\n</script>"))
}

/// The report path for a JSON output: the same path with `.html`.
pub fn default_html_path(output_json: &Path) -> PathBuf {
    output_json.with_extension("html")
}

/// One `<tr>` of a two-column table, of text.
fn row(out: &mut String, name: &str, value: impl Display) {
    row_html(out, name, &escape_html(&value.to_string()));
}

/// [`row`] with a value that is HTML.
fn row_html(out: &mut String, name: &str, value: &str) {
    let name = escape_html(name);
    let _ = writeln!(out, "<tr><th>{name}</th><td>{value}</td></tr>");
}

/// The rows of the run summary that say how the fit used the callable bases
/// of each bin: the emission model, and what the counts hold.
pub fn bin_callable_rows(b: &BinCallableRecord) -> Vec<(&'static str, String)> {
    let model = if b.model == CALLABLE_LENGTH_MODEL {
        "each called bin by its callable bases (callable-length model)"
    } else {
        "every called bin as the bin size in bases (C psmc's model)"
    };
    let mut rows = vec![("Bin model", format!("{model}; --bin-callable {}", b.mode))];
    if let (Some(c), Some(source)) = (&b.callable, &b.counts) {
        rows.push((
            "Callable bases of called bins",
            format!(
                "{} on average; {} of called bins partly callable (counts {source})",
                pct(c.mean_fraction),
                pct(c.partial_fraction())
            ),
        ));
    }
    rows
}

/// The run summary table.
fn meta_rows(d: &ReportData, n0: f64, ranges: [(f64, f64); 2], points: usize) -> String {
    let m = d.model;
    let mono = |path: &Path| {
        let path = escape_html(&path.display().to_string());
        format!(r#"<span class="mono">{path}</span>"#)
    };
    let mut out = String::new();
    row(&mut out, "psmc-rs", env!("CARGO_PKG_VERSION"));
    row(&mut out, "HMM kernels", crate::hmm::simd_path().name());
    row_html(&mut out, "Input", &mono(d.input_file));
    row_html(&mut out, "Output JSON", &mono(d.output_json));
    if let Some(i) = d.input {
        input_rows(&mut out, i);
    }
    for (name, value) in d.bin_callable.map(bin_callable_rows).unwrap_or_default() {
        row(&mut out, name, value);
    }
    row(&mut out, "EM iterations", d.n_iter);
    let loglike = d
        .final_loglike
        .map_or("-".to_string(), |ll| format!("{ll:.6}"));
    row(&mut out, "Final log-likelihood", loglike);
    let pattern = m
        .pattern
        .as_deref()
        .unwrap_or("none (one lambda per state)");
    row(&mut out, "Pattern", pattern);
    let states = format!("{} / {}", m.n_steps + 1, m.lam.len());
    row(&mut out, "States / free lambdas", states);
    for (name, v) in [
        ("theta (per bin)", m.theta),
        ("rho (per bin)", m.rho),
        ("t_max (2N0 generations)", m.t_max),
        ("mu (per bp per generation)", m.mu),
        ("N0", n0),
        ("Bin size (bp)", d.bin_size),
        ("Generation time (years)", d.gen_years),
    ] {
        row(&mut out, name, fmt_num(v));
    }
    for (name, value) in &d.settings {
        row(&mut out, name, value);
    }
    row(&mut out, "Curve points", points);
    for (name, (lo, hi)) in ["Time range (years)", "Ne range"].into_iter().zip(ranges) {
        row(&mut out, name, format!("{} - {}", fmt_num(lo), fmt_num(hi)));
    }
    out
}

/// The run summary's rows on the input: its format and size.
fn input_rows(out: &mut String, i: &InputSummary) {
    let format = match &i.vcf_mode {
        Some(mode) => format!("{} ({mode} mode)", i.format),
        None => i.format.clone(),
    };
    row(out, "Input format", format);
    let (kept, dropped) = (count(i.sequences as u64), count(i.dropped_sequences as u64));
    let sequences = format!("{kept} analysed, {dropped} left out");
    row(out, "Sequences", sequences);
    let share = |n: u64| pct(fraction(n, i.bins));
    let (het, missing) = (share(i.het_bins), share(i.missing_bins));
    let bins = format!("{} ({het} heterozygous, {missing} missing)", count(i.bins));
    row(out, "Bins", bins);
}

/// The input QC section: findings, then the key metrics and the
/// per-sequence table.
fn qc_section(r: &QcReport) -> String {
    let mut out = String::new();
    let (warnings, notes) = r.warnings_and_notes();
    let _ = writeln!(
        out,
        r#"<section class="card"><div class="meta"><h2>Input QC</h2>
<div class="qc-headline">{}, {} on {}, {} bins of {} bp</div>"#,
        counted(warnings, "warning", "warnings"),
        counted(notes, "note", "notes"),
        counted(r.n_sequences, "sequence", "sequences"),
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
    qc_metric_rows(&mut out, r);
    out.push_str("</table>");
    if !r.sequences.is_empty() {
        sequence_table(&mut out, r);
    }
    out.push_str("</div></section>");
    out
}

/// The key metrics of the input QC, as table rows.
fn qc_metric_rows(out: &mut String, r: &QcReport) {
    let t = &r.totals;
    let bins = [t.t_fraction, t.k_fraction, t.missing_fraction].map(pct);
    row(out, "Bins T / K / N", bins.join(" / "));
    row(out, "theta0 per bin", sig_or_na(t.theta0_per_bin, 4));
    row(out, "Heterozygous sites per kb", sig_or_na(t.het_per_kb, 3));
    let called = match t.called_bases {
        Some(n) => format!("{} bp (called sites)", count(n)),
        None => format!("{} bp (in called bins)", count(t.called_bases_from_bins)),
    };
    row(out, "Called length", called);
    if let Some(c) = &r.callable {
        let callable = format!(
            "{} on average; {}{}",
            pct(c.mean_fraction),
            crate::qc::callable_distribution(c),
            crate::qc::approximate_counts_note(r)
        );
        row(out, "Callable bases of called bins", callable);
    }
    let (n50, l50) = (count(r.lengths.n50_bases), count(r.lengths.l50 as u64));
    row(out, "N50 / L50", format!("{n50} bp / {l50}"));
    if let Some(run) = &r.longest_missing_run {
        let bins = counted(run.bins as usize, "bin", "bins");
        let longest = format!("{bins} in {}", run.sequence);
        row(out, "Longest missing run", longest);
    }
    if let Some(s) = &r.saturation {
        let share = pct(s.multi_het_share);
        row(out, "K bins with 2+ het sites", format!("about {share}"));
    }
    if let Some(n) = r.expected_recombinations {
        let n = count(n.round() as u64);
        row(out, "Recombinations (C psmc's start)", format!("about {n}"));
    }
    if let Some(trunks) = r.lengths.bootstrap_trunks {
        row(out, "Bootstrap trunks", count(trunks));
    }
    if let Some(dp) = &r.depth {
        let bounds = match (dp.min_depth, dp.max_depth) {
            (Some(lo), Some(hi)) => format!(", calls kept at {lo}-{hi}"),
            _ => String::new(),
        };
        let depth = format!(
            "mean {}, median {}{bounds}",
            sig(dp.mean, 3),
            sig(dp.median, 3)
        );
        row(out, "Read depth", depth);
    }
    for d in &r.dropped {
        let left_out = format!("{} {}", count(d.count as u64), d.reason);
        row(out, "Left out", left_out);
    }
}

/// The per-sequence metrics of the input QC, in a collapsed table.
fn sequence_table(out: &mut String, r: &QcReport) {
    let per_callable = r.het_per_callable_kb();
    let _ = writeln!(
        out,
        r#"<details><summary>Per-sequence metrics ({} of {})</summary><table class="seq-table">
<tr><th>Sequence</th><th>Bins</th><th>K</th><th>N</th><th>{}</th><th>vs others</th></tr>"#,
        r.sequences.len(),
        r.n_sequences,
        if per_callable { "Het/kb*" } else { "Het/kb" }
    );
    for s in &r.sequences {
        let kind = s
            .kind
            .map_or(String::new(), |k| format!(" ({})", k.label()));
        let vs_others = s
            .het_vs_reference
            .map_or("-".to_string(), |v| format!("{}x", sig(v, 2)));
        let _ = writeln!(
            out,
            "<tr><td>{}{kind}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{vs_others}</td></tr>",
            escape_html(&s.name),
            count(s.stats.bins),
            pct(s.stats.k_fraction),
            pct(s.stats.missing_fraction),
            sig_or_na(s.stats.het_per_kb_comparable(), 3),
        );
    }
    out.push_str("</table>");
    if per_callable {
        let note = escape_html(crate::qc::HET_PER_CALLABLE_KB_NOTE);
        let _ = write!(out, "<div class=\"note\">* {note}</div>");
    }
    out.push_str("</details>");
}

/// The bootstrap section, with the coverage caveat that
/// `examples/bootstrap_calibration.rs` measures at the CLI's default start
/// and objective (the numbers the `bootstrap` module docs give).
fn bootstrap_section(b: &BootstrapReport) -> String {
    let s = b.summary;
    let skipped = s.skipped_replicates.len();
    let mut out =
        String::from(r#"<section class="card"><div class="meta"><h2>Bootstrap</h2><table>"#);
    row(&mut out, "Replicates", count(s.n_replicates as u64));
    if skipped > 0 {
        let numbers: Vec<String> = s.skipped_replicates.iter().map(usize::to_string).collect();
        let (n, total) = (
            count(skipped as u64),
            count((s.n_replicates + skipped) as u64),
        );
        let skipped = format!("{n} of {total} ({})", numbers.join(", "));
        row(&mut out, "Skipped replicates", skipped);
    }
    row(&mut out, "Trunk size (bins)", count(s.block_size as u64));
    if let Some(n) = b.trunks {
        row(&mut out, "Trunks resampled", count(n as u64));
    }
    row(&mut out, "EM iterations per replicate", b.em_iters);
    row(&mut out, "Seed", s.seed);
    let start = match b.started_from {
        Some(path) => format!("the parameters in {}", path.display()),
        None => "a fresh start".to_string(),
    };
    row(&mut out, "Replicates start from", start);
    let (fitted, saved_note) = match b.started_from {
        Some(_) => (
            "each fitted from the saved parameters the main fit continued (as C psmc -i -b does)",
            " Those parameters were fitted to the whole input, so replicates that run few \
             iterations from them stay close to them, and the band understates the uncertainty \
             further.",
        ),
        None => ("each fitted from a fresh start", ""),
    };
    let skipped_note = match skipped {
        0 => String::new(),
        n => format!(
            " {n} of the {} replicates were left out: the fresh start of their resampled data \
             had negative transition probabilities, which EM cannot fit. Their resamples were the \
             most heterozygous, so without them the band may be too narrow.",
            s.n_replicates + n
        ),
    };
    let _ = write!(
        out,
        "</table><div class=\"note caveat\">The band spans the 2.5% to 97.5% quantiles of the \
         replicate curves, {fitted} on trunks of the input resampled with replacement, as C \
         psmc's bootstrap does. Read it as a lower bound on the uncertainty. In simulations with \
         psmc-rs's default start and objective (examples/bootstrap_calibration.rs, described in \
         the README), the bootstrap standard deviation of log10 Ne was a median 0.85 (whole \
         sequences as trunks) to 0.91 (1,000-bin trunks) of the true sampling standard \
         deviation, and 0.74 to 0.78 at the oldest times, though at the most recent times \
         whole-sequence trunks gave a band wider than the true spread; the estimator's bias, \
         about a third of that standard deviation, does not show in the band at all. Few trunks \
         give coarse bands.{saved_note}{skipped_note}</div></div></section>"
    );
    out
}

/// The smallest (at least 1e-12) and the largest of coordinate `i` of
/// `points`.
fn axis_range(points: &[[f64; 2]], i: usize) -> (f64, f64) {
    let lo = points
        .iter()
        .map(|p| p[i])
        .fold(f64::INFINITY, f64::min)
        .max(1e-12);
    let hi = points.iter().map(|p| p[i]).fold(0.0f64, f64::max).max(lo);
    (lo, hi)
}

/// The chart's curves: the fit's, and the bootstrap's quantiles.
fn curve_series(main: Vec<[f64; 2]>, bootstrap: Option<&BootstrapReport>) -> Vec<JsSeries> {
    let mut series = vec![JsSeries {
        name: "psmc-rs",
        data: main,
    }];
    if let Some(b) = bootstrap {
        let points = &b.summary.points;
        let quantile = |name, ne: fn(&BootstrapCurvePoint) -> f64| JsSeries {
            name,
            data: points.iter().map(|p| [p.x_years, ne(p)]).collect(),
        };
        series.extend([
            quantile("Bootstrap q2.5", |p| p.ne_q025),
            quantile("Bootstrap median", |p| p.ne_q500),
            quantile("Bootstrap q97.5", |p| p.ne_q975),
        ]);
    }
    series
}

/// The TMRCA panel and its script, with each state's interval of the time
/// grid `curve` in years.
fn tmrca_html(tm: &TmrcaReportData, curve: &[[f64; 2]]) -> Result<(String, String)> {
    let total_mass: f64 = tm.state_mass.iter().sum();
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
    let panel = TMRCA_PANEL
        .replace("__ROWS__", &tm.total_sites.to_string())
        .replace("__POINTS__", &tm.sampled_mean.len().to_string())
        .replace("__SEQUENCES__", &tm.seq_count.to_string());
    let script = TMRCA_SCRIPT
        .replace("__MEAN__", &serde_json::to_string(&tm.sampled_mean)?)
        .replace("__MAP__", &serde_json::to_string(&tm.sampled_map)?)
        .replace("__LO__", &serde_json::to_string(&tm.sampled_lo)?)
        .replace("__HI__", &serde_json::to_string(&tm.sampled_hi)?)
        .replace("__STATE_BINS__", &serde_json::to_string(&state_bins)?);
    Ok((panel, script))
}

/// Write the report of `data` to `output_html`.
pub fn write_html_report(output_html: &Path, data: &ReportData) -> Result<()> {
    let echarts_script = echarts_script_tag()?;
    let curve = data.model.ne_curve(data.bin_size, data.gen_years)?;
    let main: Vec<[f64; 2]> = curve
        .iter()
        .copied()
        .filter(|&[x, y]| x.is_finite() && y.is_finite() && x > 0.0 && y > 0.0)
        .collect();
    let n0 = data.model.n0_for_bin(data.bin_size);
    let ranges = [axis_range(&main, 0), axis_range(&main, 1)];
    let meta_rows = meta_rows(data, n0, ranges, main.len());
    let series = curve_series(main, data.bootstrap.as_ref());
    let (tmrca_panel, tmrca_script) = match data.tmrca {
        Some(tm) => tmrca_html(tm, &curve)?,
        None => Default::default(),
    };
    let input_name = data
        .input_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let qc = data.qc.map(qc_section).unwrap_or_default();
    let bootstrap = data.bootstrap.as_ref().map(bootstrap_section);
    // In this order: each value is searched for the placeholders after it.
    let mut html = PAGE.to_string();
    for (placeholder, value) in [
        ("__INPUT_NAME__", escape_html(input_name)),
        ("__VERSION__", env!("CARGO_PKG_VERSION").to_string()),
        ("__META_ROWS__", meta_rows),
        ("__QC_SECTION__", qc),
        ("__BOOTSTRAP_SECTION__", bootstrap.unwrap_or_default()),
        ("__TMRCA_PANEL__", tmrca_panel),
        ("__TMRCA_SCRIPT__", tmrca_script),
        ("__SERIES_JSON__", serde_json::to_string(&series)?),
        ("__ECHARTS_SCRIPT__", echarts_script),
    ] {
        html = html.replace(placeholder, &value);
    }
    fs::write(output_html, html)
        .with_context(|| format!("failed to write {}", output_html.display()))
}

/// The page; `__NAME__` marks where [`write_html_report`] puts each part.
const PAGE: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PSMC Inference Report</title>
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
      <h1>PSMC Inference Report</h1>
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

      // The band stacks the spread q97.5 - q2.5 on an invisible q2.5 line.
      // ECharts stacks on the first numeric dimension of the data, which for
      // the curves' [years, ne] points is years, so the band's points put Ne
      // first and map it to the y axis: [ne, years], and for the spread also
      // the bounds, which its tooltip shows.
      const ribbonBase = hasRibbon ? q025Data.map((p) => [p[1], p[0]]) : [];
      const ribbonDelta = hasRibbon
        ? q975Data.map((p, i) => {
            const lo = (q025Data[i] && Number.isFinite(q025Data[i][1])) ? q025Data[i][1] : p[1];
            return [Math.max(0, p[1] - lo), p[0], lo, p[1]];
          })
        : [];
      const ribbonDims = ["ne", "years"];
      const ribbonEncode = { x: "years", y: "ne" };
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
          dimensions: ribbonDims,
          encode: ribbonEncode,
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
          dimensions: ribbonDims.concat([
            { name: "lo", displayName: "2.5%" },
            { name: "hi", displayName: "97.5%" }
          ]),
          encode: Object.assign({ tooltip: ["lo", "hi"] }, ribbonEncode),
          data: ribbonDelta,
          showSymbol: false,
          symbol: "none",
          lineStyle: { width: 0, opacity: 0 },
          itemStyle: { color: palette.ci },
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

/// The TMRCA panel of the page; `__NAME__` marks each number.
const TMRCA_PANEL: &str = r#"
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
          <tr><th>TMRCA rows</th><td>__ROWS__</td></tr>
          <tr><th>Sampled points</th><td>__POINTS__</td></tr>
          <tr><th>Posterior CI</th><td>q2.5 to q97.5 (per sampled bin)</td></tr>
          <tr><th>Sequences</th><td>__SEQUENCES__</td></tr>
        </table>
      </div>
    </section>
"#;

/// The script of the TMRCA charts; `__NAME__` marks each data set.
const TMRCA_SCRIPT: &str = r##"
    const tmrcaMeanRaw = __MEAN__;
    const tmrcaMapRaw = __MAP__;
    const tmrcaLoRaw = __LO__;
    const tmrcaHiRaw = __HI__;
    const tmrcaStateBins = __STATE_BINS__;
    const positiveY = (pt) => Number.isFinite(pt[0]) && Number.isFinite(pt[1]) && pt[1] > 0;
    const tmrcaMean = tmrcaMeanRaw.filter(positiveY);
    const tmrcaMap = tmrcaMapRaw.filter(positiveY);
    const tmrcaLo = tmrcaLoRaw.filter(positiveY);
    const tmrcaHi = tmrcaHiRaw.filter(positiveY);
    const tmrcaTrack = echarts.init(document.getElementById("tmrca-track"), null, { renderer: "canvas" });
    const tmrcaStateChart = echarts.init(document.getElementById("tmrca-state"), null, { renderer: "canvas" });
    const tmrcaCtrl = {
      map: document.getElementById("ctrl-tmrca-map"),
      ci: document.getElementById("ctrl-tmrca-ci"),
      yScale: document.getElementById("ctrl-tmrca-yscale"),
      stateMetric: document.getElementById("ctrl-state-metric")
    };

    const fmtYears = (v) => {
      if (!Number.isFinite(v)) return "NA";
      if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
      if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
      return v.toFixed(1);
    };
    const tmrcaStateLabels = tmrcaStateBins.map((b) => {
      const s = fmtYears(b.start_years);
      if (b.end_years === null || !Number.isFinite(b.end_years)) {
        return `>= ${s}`;
      }
      return `${s} - ${fmtYears(b.end_years)}`;
    });
    const tmrcaStatePct = tmrcaStateBins.map((b) => b.mass_pct);
    const tmrcaStateRaw = tmrcaStateBins.map((b) => b.mass);
    const tickStep = Math.max(1, Math.ceil(tmrcaStateLabels.length / 18));

    const buildTmrcaTrackOption = (theme) => {
      const yType = tmrcaCtrl.yScale ? tmrcaCtrl.yScale.value : "log";
      const showMap = tmrcaCtrl.map ? tmrcaCtrl.map.checked : true;
      const showCi = tmrcaCtrl.ci ? tmrcaCtrl.ci.checked : true;
      const widthSlider = document.getElementById("ctrl-main-width");
      const lineW = widthSlider ? Number(widthSlider.value) : 2.4;
      const trackSeries = [
        {
          name: "mean",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMean,
          lineStyle: { width: Math.max(1.4, lineW - 0.4), color: theme.trackMean },
          itemStyle: { color: theme.trackMean }
        }
      ];
      if (showMap) {
        trackSeries.push({
          name: "map",
          type: "line",
          showSymbol: false,
          symbol: "none",
          data: tmrcaMap,
          lineStyle: { width: Math.max(1.2, lineW - 0.6), color: theme.trackMap, type: "dashed" },
          itemStyle: { color: theme.trackMap }
        });
      }
      if (showCi) {
        trackSeries.push(
          {
            name: "q2.5",
            type: "line",
            showSymbol: false,
            symbol: "none",
            data: tmrcaLo,
            lineStyle: { width: 1.2, color: theme.ci, type: "dotted" },
            itemStyle: { color: theme.ci }
          },
          {
            name: "q97.5",
            type: "line",
            showSymbol: false,
            symbol: "none",
            data: tmrcaHi,
            lineStyle: { width: 1.2, color: theme.ci, type: "dotted" },
            itemStyle: { color: theme.ci }
          }
        );
      }

      return {
        animationDuration: 650,
        animationEasing: "cubicOut",
        legend: {
          top: 8,
          left: 14,
          textStyle: { color: "#223a52", fontSize: 12, fontFamily: "Sora, sans-serif" }
        },
        grid: { left: 84, right: 42, top: 44, bottom: 68, containLabel: true },
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "cross" },
          backgroundColor: "rgba(13, 22, 34, 0.90)",
          borderWidth: 0,
          textStyle: { color: "#f2f6fb" }
        },
        toolbox: {
          right: 8,
          top: 6,
          itemSize: 14,
          feature: {
            dataZoom: { yAxisIndex: "none" },
            restore: {},
            saveAsImage: { pixelRatio: 2 }
          }
        },
        xAxis: {
          type: "value",
          name: "Genome bin index",
          nameLocation: "middle",
          nameGap: 34,
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: { color: "#51697f" },
          splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } }
        },
        yAxis: {
          type: yType,
          name: "TMRCA (years)",
          nameLocation: "middle",
          nameRotate: 90,
          nameGap: 62,
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: { color: "#51697f" },
          splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } }
        },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          {
            type: "slider",
            xAxisIndex: 0,
            bottom: 16,
            height: 18,
            borderColor: "rgba(17,34,51,0.15)",
            backgroundColor: "rgba(255,255,255,0.62)",
            fillerColor: "rgba(66,165,245,0.20)",
            handleStyle: { color: theme.accent }
          }
        ],
        series: trackSeries
      };
    };

    const buildTmrcaStateOption = (theme) => {
      const metric = tmrcaCtrl.stateMetric ? tmrcaCtrl.stateMetric.value : "pct";
      const isPct = metric !== "raw";
      const yData = isPct ? tmrcaStatePct : tmrcaStateRaw;
      const yName = isPct ? "Posterior mass (%)" : "Posterior mass";
      return {
        animationDuration: 600,
        animationEasing: "cubicOut",
        tooltip: {
          trigger: "axis",
          axisPointer: { type: "shadow" },
          backgroundColor: "rgba(13, 22, 34, 0.90)",
          borderWidth: 0,
          textStyle: { color: "#f2f6fb" },
          formatter: (params) => {
            const p = Array.isArray(params) ? params[0] : params;
            const idx = p.dataIndex;
            const b = tmrcaStateBins[idx];
            if (!b) return "";
            const endTxt = (b.end_years === null || !Number.isFinite(b.end_years))
              ? "inf"
              : b.end_years.toExponential(3);
            return [
              `State ${idx}`,
              `Interval: ${b.start_years.toExponential(3)} - ${endTxt} years`,
              `Mass: ${b.mass.toExponential(3)}`,
              `Mass %: ${b.mass_pct.toFixed(2)}%`
            ].join("<br/>");
          }
        },
        toolbox: {
          right: 8,
          top: 6,
          itemSize: 14,
          feature: {
            dataZoom: { yAxisIndex: "none" },
            restore: {},
            saveAsImage: { pixelRatio: 2 }
          }
        },
        grid: { left: 84, right: 42, top: 32, bottom: 64, containLabel: true },
        xAxis: {
          type: "category",
          data: tmrcaStateLabels,
          name: "State interval (years)",
          nameLocation: "middle",
          nameGap: 44,
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: {
            color: "#51697f",
            interval: (idx) => idx % tickStep === 0,
            rotate: 24,
            fontSize: 10
          },
          splitLine: { show: false }
        },
        yAxis: {
          type: "value",
          name: yName,
          nameLocation: "middle",
          nameRotate: 90,
          nameGap: 56,
          axisLine: { lineStyle: { color: "#698198" } },
          axisLabel: { color: "#51697f" },
          splitLine: { lineStyle: { color: "rgba(81,105,127,0.15)" } }
        },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          {
            type: "slider",
            xAxisIndex: 0,
            bottom: 12,
            height: 16,
            borderColor: "rgba(17,34,51,0.15)",
            backgroundColor: "rgba(255,255,255,0.62)",
            fillerColor: "rgba(38,166,154,0.22)",
            handleStyle: { color: theme.accent2 }
          }
        ],
        series: [{
          name: isPct ? "mass %" : "mass",
          type: "bar",
          data: yData,
          barMaxWidth: 24,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: theme.stateTop },
              { offset: 1, color: theme.stateBottom }
            ]),
            borderRadius: [4, 4, 0, 0]
          }
        }]
      };
    };

    const applyTmrcaCharts = (theme) => {
      tmrcaTrack.setOption(buildTmrcaTrackOption(theme), true);
      tmrcaStateChart.setOption(buildTmrcaStateOption(theme), true);
    };
    window.__psmcTmrca = {
      track: tmrcaTrack,
      state: tmrcaStateChart,
      apply: applyTmrcaCharts
    };
"##;
