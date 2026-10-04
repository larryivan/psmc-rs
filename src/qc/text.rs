//! Plain-text rendering of a [`QcReport`].

use std::collections::HashSet;
use std::fmt::Write as _;

use super::fmt::{bases, count, pct, sig};
use super::{
    BinStats, Finding, QcReport, ReferenceMethod, SATURATION_TARGET_SHARE, SequenceRow, Severity,
};

/// How [`render_text_with`] lays out a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextOptions {
    /// At most this many per-sequence rows (0: no table). Sequences named in
    /// findings come first, then the longest; rows stay in input order.
    pub sequence_rows: usize,
    /// Wrap finding text at this many columns.
    pub width: usize,
}

impl Default for TextOptions {
    /// Findings, metrics and up to 30 sequence rows, wrapped at 100 columns.
    fn default() -> Self {
        Self {
            sequence_rows: 30,
            width: 100,
        }
    }
}

impl TextOptions {
    /// Findings and metrics without the sequence table, for the QC block
    /// printed before a fit.
    pub fn brief() -> Self {
        Self {
            sequence_rows: 0,
            ..Self::default()
        }
    }
}

/// [`render_text_with`] with [`TextOptions::default`].
pub fn render_text(report: &QcReport) -> String {
    render_text_with(report, &TextOptions::default())
}

/// Render `report` as aligned plain text: a headline, the findings
/// (warnings first, each with its recommendation), the metrics, and a
/// per-sequence table. The output depends only on the report and options.
pub fn render_text_with(report: &QcReport, opts: &TextOptions) -> String {
    let mut out = String::new();
    headline(&mut out, report);
    findings(&mut out, &report.findings, opts.width);
    metrics(&mut out, report);
    if opts.sequence_rows > 0 && !report.sequences.is_empty() {
        sequence_table(&mut out, report, opts.sequence_rows);
    }
    out
}

fn headline(out: &mut String, r: &QcReport) {
    let format = r
        .input_format
        .as_deref()
        .map_or(String::new(), |f| format!(" ({f})"));
    let _ = writeln!(
        out,
        "Input QC{format}: {} {}, {} bins of {} bp",
        count(r.n_sequences as u64),
        if r.n_sequences == 1 {
            "sequence"
        } else {
            "sequences"
        },
        count(r.totals.bins),
        r.bin_size
    );
}

/// Greedy word wrap of `text` into lines of at most `width` columns; the
/// first line starts with `first`, the others with `rest` (a single word too
/// long for a line stays whole).
fn wrap(out: &mut String, text: &str, first: &str, rest: &str, width: usize) {
    let mut line = String::from(first);
    let mut empty = true;
    for word in text.split_whitespace() {
        if !empty && line.len() + 1 + word.len() > width {
            out.push_str(&line);
            out.push('\n');
            line = String::from(rest);
            empty = true;
        }
        if !empty {
            line.push(' ');
        }
        line.push_str(word);
        empty = false;
    }
    if !empty {
        out.push_str(&line);
        out.push('\n');
    }
}

fn findings(out: &mut String, findings: &[Finding], width: usize) {
    let warnings = findings
        .iter()
        .filter(|f| f.severity == Severity::Warning)
        .count();
    let infos = findings.len() - warnings;
    if findings.is_empty() {
        out.push_str("No warnings or notes.\n");
        return;
    }
    let plural = |n: usize, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
    let _ = writeln!(
        out,
        "{}, {}",
        plural(warnings, "warning"),
        plural(infos, "note")
    );
    for f in findings {
        let label = match f.severity {
            Severity::Warning => "WARNING",
            Severity::Info => "INFO",
        };
        let _ = writeln!(out, "\n{label:<8} {}", f.code.as_str());
        wrap(out, &f.message, "  ", "  ", width);
        wrap(out, &f.recommendation, "  -> ", "     ", width);
    }
}

/// A count with its share, e.g. `1,234 (5.6%)`.
fn with_share(n: u64, fraction: f64) -> String {
    format!("{} ({})", count(n), pct(fraction))
}

fn opt_sig(v: Option<f64>, digits: usize) -> String {
    v.map_or("n/a".to_string(), |x| sig(x, digits))
}

fn metrics(out: &mut String, r: &QcReport) {
    let t = &r.totals;
    let mut rows: Vec<(&str, String)> = Vec::new();
    rows.push((
        "Bins",
        format!(
            "T {}, K {}, N {}",
            with_share(t.t_bins, t.t_fraction),
            with_share(t.k_bins, t.k_fraction),
            with_share(t.missing_bins, t.missing_fraction)
        ),
    ));
    rows.push((
        "theta0",
        format!(
            "{} per bin, {} per bp ({} per kb)",
            opt_sig(t.theta0_per_bin, 3),
            opt_sig(t.theta_per_bp, 3),
            opt_sig(t.het_per_kb, 3)
        ),
    ));
    let mut called = format!("{} in called bins", bases(t.called_bases_from_bins));
    if let Some(c) = t.called_bases {
        let _ = write!(called, "; {} called sites", bases(c));
    }
    if let Some(h) = t.site_het_per_kb {
        let _ = write!(called, ", {} het sites per kb", sig(h, 3));
    }
    rows.push(("Called length", called));

    let l = &r.lengths;
    rows.push((
        "Sequence lengths",
        format!(
            "N50 {} (L50 {}), longest {}; {} of bins in sequences < 1 Mb",
            bases(l.n50_bases),
            count(l.l50 as u64),
            bases(l.longest_bases),
            pct(l.below_1mb.fraction)
        ),
    ));
    if let (Some(trunks), Some(below)) = (l.bootstrap_trunks, &l.below_trunk) {
        rows.push((
            "Bootstrap",
            format!(
                "{} trunks of {} bins; {} of bins in sequences shorter than a trunk",
                count(trunks),
                count(r.bootstrap_trunk_bins as u64),
                pct(below.fraction)
            ),
        ));
    }
    rows.push((
        "Longest N run",
        r.longest_missing_run
            .as_ref()
            .map_or("none".to_string(), |m| {
                format!(
                    "{} bins ({}) in {} from bin {}",
                    count(m.bins),
                    bases(m.bases),
                    m.sequence,
                    count(m.start_bin)
                )
            }),
    ));
    if let Some(s) = &r.saturation {
        let target = s.max_bin_size_below_target.map_or(String::new(), |b| {
            format!(
                "; below {} up to {b} bp per bin",
                pct(SATURATION_TARGET_SHARE)
            )
        });
        rows.push((
            "Bin saturation",
            format!(
                "about {} of K bins hold >= 2 het sites ({} per kb){target}",
                pct(s.multi_het_share),
                sig(1000.0 * s.het_per_site, 3)
            ),
        ));
    }
    if let Some(h) = &r.het_reference {
        let how = match h.method {
            ReferenceMethod::Median => "median of",
            ReferenceMethod::Pooled => "pooled over",
        };
        rows.push((
            "Autosomal het",
            format!(
                "{} per kb ({how} {} {})",
                sig(h.het_per_kb, 3),
                count(h.sequences as u64),
                if h.sequences == 1 {
                    "sequence"
                } else {
                    "sequences"
                }
            ),
        ));
    }
    if let Some(n) = r.expected_recombinations {
        rows.push((
            "Recombinations",
            format!(
                "about {} expected by C psmc's starting model (theta/rho = 4)",
                count(n.round() as u64)
            ),
        ));
    }
    if let Some(d) = &r.depth {
        let mut depth = format!("mean {}x, median {}x", sig(d.mean, 3), sig(d.median, 3));
        if let Some([p05, _, _, p95]) = d.quantiles {
            // A quantile in the open last class is only a lower bound.
            let open = d.histogram.len().saturating_sub(1);
            let q = |p: u32| {
                if p as usize == open {
                    format!("{p}+")
                } else {
                    p.to_string()
                }
            };
            let _ = write!(depth, ", 5-95% {}-{}x", q(p05), q(p95));
        }
        match (d.min_depth, d.max_depth) {
            (Some(lo), Some(hi)) => {
                let _ = write!(depth, "; kept {lo}-{hi}x");
            }
            (Some(lo), None) => {
                let _ = write!(depth, "; kept >= {lo}x");
            }
            (None, Some(hi)) => {
                let _ = write!(depth, "; kept <= {hi}x");
            }
            (None, None) => {}
        }
        let shares: Vec<String> = [
            d.below_min_fraction.map(|f| format!("{} below", pct(f))),
            d.above_max_fraction.map(|f| format!("{} above", pct(f))),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !shares.is_empty() {
            let _ = write!(depth, " ({})", shares.join(", "));
        }
        rows.push(("Read depth", depth));
        if !d.filters.is_empty() {
            let list: Vec<String> = d
                .filters
                .iter()
                .map(|f| format!("{} {}", f.name, pct(f.fraction)))
                .collect();
            rows.push((
                "Filters",
                format!(
                    "{} of {} bases removed",
                    list.join(", "),
                    count(d.total_bases)
                ),
            ));
        }
    }
    if !r.dropped.is_empty() {
        let list: Vec<String> = r
            .dropped
            .iter()
            .map(|d| format!("{} {}", count(d.count as u64), d.reason))
            .collect();
        rows.push(("Dropped", list.join("; ")));
    }
    if r.empty_sequences > 0 {
        rows.push((
            "Empty",
            format!("{} sequences without bins", count(r.empty_sequences as u64)),
        ));
    }

    out.push_str("\nMetrics\n");
    let label_width = rows.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
    for (label, value) in rows {
        let _ = writeln!(out, "  {label:<label_width$}  {value}");
    }
}

/// Longest name shown in the sequence table; longer ones are cut.
const MAX_NAME: usize = 32;

fn cut_name(name: &str) -> String {
    if name.chars().count() <= MAX_NAME {
        name.to_string()
    } else {
        let head: String = name.chars().take(MAX_NAME - 3).collect();
        format!("{head}...")
    }
}

fn table_cells(row: &SequenceRow) -> Vec<String> {
    let s: &BinStats = &row.stats;
    vec![
        cut_name(&row.name),
        count(s.bins),
        format!("{:.1}", 100.0 * s.t_fraction),
        format!("{:.1}", 100.0 * s.k_fraction),
        format!("{:.1}", 100.0 * s.missing_fraction),
        opt_sig(s.het_per_kb, 3),
        row.het_vs_reference
            .map_or("n/a".to_string(), |x| format!("{x:.2}")),
        count(s.longest_missing_run_bins),
        row.kind.map_or(String::new(), |k| k.label().to_string()),
    ]
}

fn sequence_table(out: &mut String, r: &QcReport, cap: usize) {
    let named: HashSet<&str> = r
        .findings
        .iter()
        .flat_map(|f| f.sequences.iter().map(String::as_str))
        .collect();
    let mut chosen: Vec<usize> = (0..r.sequences.len()).collect();
    if chosen.len() > cap {
        chosen.sort_by(|&a, &b| {
            let (ra, rb) = (&r.sequences[a], &r.sequences[b]);
            let (na, nb) = (
                named.contains(ra.name.as_str()),
                named.contains(rb.name.as_str()),
            );
            nb.cmp(&na)
                .then(rb.stats.bins.cmp(&ra.stats.bins))
                .then(a.cmp(&b))
        });
        chosen.truncate(cap);
        chosen.sort_unstable();
    }

    let header = [
        "sequence",
        "bins",
        "T%",
        "K%",
        "N%",
        "het/kb",
        "vs ref",
        "longest N",
        "kind",
    ];
    let cells: Vec<Vec<String>> = chosen
        .iter()
        .map(|&i| table_cells(&r.sequences[i]))
        .collect();
    let widths: Vec<usize> = (0..header.len())
        .map(|c| {
            cells
                .iter()
                .map(|row| row[c].len())
                .chain([header[c].len()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cols: &[&str]| -> String {
        let mut s = String::from(" ");
        for (c, text) in cols.iter().enumerate() {
            let w = widths[c];
            // Names and kinds left-aligned, numbers right-aligned.
            if c == 0 || c == header.len() - 1 {
                let _ = write!(s, " {text:<w$}");
            } else {
                let _ = write!(s, " {text:>w$}");
            }
        }
        s.trim_end().to_string()
    };

    let _ = writeln!(
        out,
        "\nSequences ({} of {}, input order)",
        count(chosen.len() as u64),
        count(r.n_sequences as u64)
    );
    out.push_str(&line(&header));
    out.push('\n');
    for row in &cells {
        let refs: Vec<&str> = row.iter().map(String::as_str).collect();
        out.push_str(&line(&refs));
        out.push('\n');
    }
    let hidden = r.n_sequences - chosen.len();
    if hidden > 0 {
        let shown_bins: u64 = chosen.iter().map(|&i| r.sequences[i].stats.bins).sum();
        let _ = writeln!(
            out,
            "  ... {} more {} with {} bins (included in the totals)",
            count(hidden as u64),
            if hidden == 1 { "sequence" } else { "sequences" },
            count(r.totals.bins - shown_bins)
        );
    }
}
