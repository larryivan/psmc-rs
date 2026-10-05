//! Loading the input as inference, `convert` and `qc` all do, and describing
//! what was read.

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};

use psmc_rs::io::consensus::LossReason;
use psmc_rs::io::input::{Dataset, DropReason, InputOptions, SourceSummary, VcfDetails, load};
use psmc_rs::qc::{
    ConsensusStats, DepthStats, DroppedSequences, FilterCount, QcContext, QcInput, QcReport, QcSeq,
    try_analyze,
};

use crate::cli::FormatArg;

/// The input format as QC and the report name it, with the VCF mode.
pub fn format_label(data: &Dataset) -> String {
    match &data.source {
        SourceSummary::Vcf(v) => format!("vcf, {} mode", v.mode.mode.name()),
        s => s.format().name().to_string(),
    }
}

/// Input QC of a loaded dataset. `bootstrap_trunk` is the bootstrap trunk
/// size in bins (0 without a bootstrap) and `free_lambdas` the number of
/// free lambdas of the model to fit, when known.
pub fn qc_report(
    data: &Dataset,
    bin_size: usize,
    bootstrap_trunk: usize,
    free_lambdas: Option<usize>,
) -> Result<QcReport> {
    let seqs: Vec<QcSeq> = data
        .seqs
        .iter()
        .map(|s| QcSeq {
            extent_bases: s.extent,
            called_bases: s.called,
            het_sites: s.het_sites,
            ..QcSeq::new(s.name.clone(), s.n_bins)
        })
        .collect();
    let mut context = QcContext {
        input_format: Some(format_label(data)),
        free_lambdas,
        dropped: dropped_groups(data),
        ..QcContext::default()
    };
    let u32_of = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    match &data.source {
        SourceSummary::Psmcfa(_) => {}
        SourceSummary::Mhs(_) => context.approximate_missing_placement = true,
        SourceSummary::Vcf(v) => match &v.details {
            VcfDetails::Consensus(c) => {
                context.consensus = Some(ConsensusStats {
                    depth: DepthStats {
                        mean: c.depth.mean,
                        median: c.depth.median,
                        min_depth: Some(u32_of(c.min_depth)),
                        max_depth: Some(u32_of(c.max_depth)),
                        histogram: c.depth.histogram.clone(),
                    },
                    total_bases: c.bases,
                    // Positions without a record are missing data, which
                    // HIGH_MISSING covers, rather than a filter.
                    filters: LossReason::ALL
                        .iter()
                        .filter(|&&r| r != LossReason::NoRecord && c.lost.get(r) > 0)
                        .map(|&r| FilterCount {
                            name: r.label().to_string(),
                            bases: c.lost.get(r),
                        })
                        .collect(),
                });
            }
            VcfDetails::Mask(m) => {
                context.variants_only_no_mask = !m.positive_mask;
                if m.depth.records > 0 {
                    context.consensus = Some(ConsensusStats {
                        depth: DepthStats {
                            mean: m.depth.mean,
                            median: m.depth.median,
                            min_depth: None,
                            max_depth: None,
                            histogram: m.depth.histogram.clone(),
                        },
                        total_bases: m.sites,
                        filters: Vec::new(),
                    });
                }
            }
        },
    }
    try_analyze(&QcInput {
        rows: &data.obs.rows,
        row_starts: &data.obs.row_starts,
        seqs: &seqs,
        bin_size,
        bootstrap_trunk_bins: bootstrap_trunk,
        context,
    })
}

/// The dropped sequences of `data`, counted by reason.
fn dropped_groups(data: &Dataset) -> Vec<DroppedSequences> {
    let mut empty = (0, 0);
    let mut filtered = (0, 0);
    for d in &data.dropped {
        let group = match d.reason {
            DropReason::Empty => &mut empty,
            DropReason::Filter { .. } => &mut filtered,
        };
        group.0 += 1;
        group.1 += d.extent.unwrap_or(0);
    }
    let mut out = Vec::new();
    if empty.0 > 0 {
        out.push(DroppedSequences {
            reason: "without any bin".to_string(),
            count: empty.0,
            bases: None,
        });
    }
    if let (true, Some(f)) = (filtered.0 > 0, &data.filter) {
        out.push(DroppedSequences {
            reason: format!(
                "dropped by the output filter (--min-good-bases {}, --min-good-fraction {})",
                f.min_good_bases, f.min_good_fraction
            ),
            count: filtered.0,
            bases: Some(filtered.1),
        });
    }
    out
}

/// The loader's warnings that QC does not repeat: without a callable mask,
/// QC's VARIANTS_ONLY_NO_MASK replaces the mask-mode reader's warning.
pub fn loader_warnings(data: &Dataset, qc: Option<&QcReport>) -> Vec<String> {
    let mut out = data.warnings();
    if let (Some(qc), SourceSummary::Vcf(v)) = (qc, &data.source)
        && let VcfDetails::Mask(m) = &v.details
        && !m.positive_mask
        && qc
            .finding(psmc_rs::qc::QcCode::VariantsOnlyNoMask)
            .is_some()
        && let Some(no_mask) = m.warnings().first()
    {
        out.retain(|w| w != no_mask);
    }
    out
}

/// Whether the file name of `path` has an extension of `fmt` input (any
/// supported extension for auto).
pub fn is_supported_input_file(path: &Path, fmt: FormatArg) -> bool {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let name = name.strip_suffix(".gz").unwrap_or(&name);
    let psmcfa = name.ends_with(".psmcfa");
    let mhs = name.ends_with(".mhs") || name.ends_with(".multihetsep");
    let vcf = name.ends_with(".vcf");
    match fmt {
        FormatArg::Auto => psmcfa || mhs || vcf,
        FormatArg::Psmcfa => psmcfa,
        FormatArg::Mhs => mhs,
        FormatArg::Vcf => vcf,
    }
}

/// Files in `dir` that look like `fmt` input, sorted by name.
pub fn supported_files_in(dir: &Path, fmt: FormatArg) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let p = entry?.path();
        if p.is_file() && is_supported_input_file(&p, fmt) {
            files.push(p);
        }
    }
    files.sort();
    Ok(files)
}

/// `path`, or for a directory the first input file in it.
pub fn resolve_input_path(path: &Path, fmt: FormatArg) -> Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_path_buf());
    }
    supported_files_in(path, fmt)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            let what = match fmt {
                FormatArg::Auto => "psmcfa, multihetsep or VCF".to_string(),
                f => f.name().to_string(),
            };
            anyhow!(
                "directory {} does not contain a {what} input file",
                path.display()
            )
        })
}

/// [`load`] with the path in the error.
pub fn load_input(path: &Path, opts: &InputOptions) -> Result<Dataset> {
    load(path, opts).with_context(|| format!("failed to read the input {}", path.display()))
}

/// [`Dataset::ensure_usable`], naming the options that keep sequences the
/// output filter dropped.
pub fn ensure_usable(data: &Dataset) -> Result<()> {
    data.ensure_usable().map_err(|e| {
        let filtered = data
            .dropped
            .iter()
            .any(|d| matches!(d.reason, DropReason::Filter { .. }));
        if data.obs.rows.is_empty() && filtered {
            anyhow!("{e:#}. To keep every sequence, pass --min-good-bases 0 --min-good-fraction 0")
        } else {
            e
        }
    })
}

fn share(n: u64, total: u64) -> String {
    if total == 0 {
        "0.0%".to_string()
    } else {
        format!("{:.1}%", 100.0 * n as f64 / total as f64)
    }
}

/// Lines telling the user how the input was read and what it holds.
pub fn describe(data: &Dataset) -> Vec<String> {
    let mut out = Vec::new();
    match &data.source {
        SourceSummary::Psmcfa(p) => out.push(format!(
            "psmcfa: {} records ({} empty), {} bins; as C psmc counts them, n_seqs:{}, \
             sum_L:{}, sum_n:{}",
            p.records, p.empty_records, p.bins, p.records, p.called_bins, p.het_bins
        )),
        SourceSummary::Mhs(m) => out.push(format!(
            "multihetsep: {} rows of {} haplotypes; for the pair, {} heterozygous, {} \
             homozygous and {} missing rows, and {} uncalled sites (placed at the start of \
             each interval between rows)",
            m.rows, m.haplotypes, m.het_rows, m.hom_rows, m.missing_rows, m.uncalled_sites
        )),
        SourceSummary::Vcf(v) => {
            out.push(format!("VCF read in {}", v.mode.reason));
            match &v.details {
                VcfDetails::Consensus(c) => {
                    out.push(format!(
                        "consensus: depth mean {:.1}, median {}; bases kept at depth {}-{}{}, \
                         MQ >= {}, quality >= {}, {} bp from indels",
                        c.depth.mean,
                        c.depth.median,
                        c.min_depth,
                        c.max_depth,
                        if c.auto_depth { " (auto)" } else { "" },
                        c.min_mq,
                        c.min_qual,
                        c.indel_window
                    ));
                    let losses: Vec<String> = LossReason::ALL
                        .iter()
                        .filter(|&&r| c.lost.get(r) > 0)
                        .map(|&r| format!("{} {}", r.label(), c.lost.get(r)))
                        .collect();
                    out.push(format!(
                        "consensus: {} of {} bases called ({}); missing: {}",
                        c.good_bases,
                        c.bases,
                        share(c.good_bases, c.bases),
                        if losses.is_empty() {
                            "none".to_string()
                        } else {
                            losses.join(", ")
                        }
                    ));
                }
                VcfDetails::Mask(m) => {
                    let extended = if m.contig_lengths_used.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "; extended to the ##contig length: {}",
                            m.contig_lengths_used.join(", ")
                        )
                    };
                    out.push(format!(
                        "VCF: {} records; {} het SNPs used ({} het SNP records); {}/{} sites \
                         missing ({}); {} filtered; {} het non-SNP; {} missing GT; {} \
                         positions with several records{extended}",
                        m.records,
                        m.het_snps,
                        m.het_snp_records,
                        m.missing_sites,
                        m.sites,
                        share(m.missing_sites, m.sites),
                        m.filtered_records,
                        m.het_non_snp_records,
                        m.missing_genotypes,
                        m.multi_record_positions
                    ));
                }
            }
        }
    }
    let bins = data.n_bins() as u64;
    let het = data.het_bins() as u64;
    let missing = data.obs.rows.iter().flatten().filter(|&&b| b >= 2).count() as u64;
    out.push(format!(
        "Input: {} sequence(s) kept, {} left out; {bins} bins ({} heterozygous, {} missing)",
        data.seqs.len(),
        data.dropped.len(),
        share(het, bins),
        share(missing, bins)
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_files_are_recognised_by_extension() {
        let yes = |name: &str, fmt| is_supported_input_file(Path::new(name), fmt);
        assert!(yes("a.psmcfa", FormatArg::Psmcfa));
        assert!(yes("dir/A.PSMCFA.GZ", FormatArg::Psmcfa));
        assert!(yes("a.multihetsep.gz", FormatArg::Mhs));
        assert!(yes("a.vcf.gz", FormatArg::Vcf));
        assert!(!yes("a.vcf.gz", FormatArg::Psmcfa));
        for name in ["a.psmcfa", "a.mhs", "a.vcf.gz"] {
            assert!(yes(name, FormatArg::Auto), "{name}");
        }
        assert!(!yes("a.txt", FormatArg::Auto));
        assert!(!yes("a.gz", FormatArg::Auto));
    }
}
