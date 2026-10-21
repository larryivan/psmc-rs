//! Loading the input as inference, `convert` and `qc` all do, and saying
//! what was read: how it was read, and its QC report.

use anyhow::{Context, Result, anyhow, bail};
use std::fs;
use std::path::{Path, PathBuf};

use psmc_rs::io::CallableSummary;
use psmc_rs::io::consensus::{
    ConsensusSummary, DepthStats as ReadDepth, DepthThresholds, LossReason,
};
use psmc_rs::io::input::{Dataset, DropReason, InputOptions, SourceSummary, VcfDetails, load};
use psmc_rs::io::vcf::VcfSummary;
use psmc_rs::qc::fmt::{count, counted, pct, plural};
use psmc_rs::qc::{
    ConsensusStats, DepthStats, DroppedSequences, FilterCount, FitStart, QcContext, QcInput,
    QcReport, QcSeq, Severity, render_text, try_analyze,
};
use psmc_rs::{PsmcModel, initial_theta_with_callable};

use crate::cli::{
    BinCallableArg, DEFAULT_MU, DEFAULT_T_MAX, DEFAULT_TR_RATIO, FormatArg, InputArgs, model_layout,
};
use crate::output::{is_stdout, write_qc_json};
use crate::run::{Event, Sink};

/// A model of this layout, for its free lambdas and time grid only: its
/// rates are small enough for a valid transition matrix at every t_max the
/// command line accepts (ρ0 per bin must stay below about 2 / t_max).
pub fn layout_model(
    t_max: f64,
    n_steps: usize,
    pattern: Option<String>,
    mu: f64,
) -> Result<PsmcModel> {
    const RATE: f64 = 1e-4;
    PsmcModel::new(t_max, n_steps, RATE, RATE, mu, pattern)
}

/// How the callable bases of `data`'s bins were counted: exactly from a VCF,
/// or approximately from multihetsep (whose uncalled sites are placed at the
/// start of each interval); `None` when the data have no counts (psmcfa,
/// or bins too large to count).
pub fn counts_source(data: &Dataset) -> Option<&'static str> {
    data.obs.callable.as_ref()?;
    match &data.source {
        SourceSummary::Vcf(_) => Some("exact"),
        SourceSummary::Mhs(_) => Some("approximate"),
        SourceSummary::Psmcfa(_) => None,
    }
}

/// What the callable bases of `data`'s called bins hold, when it has them.
pub fn callable_summary(data: &Dataset) -> Option<CallableSummary> {
    let callable = data.obs.callable.as_ref()?;
    Some(CallableSummary::of(&data.obs.rows, callable))
}

/// Whether a fit of `data` with `--bin-callable mode` weighs each called
/// bin by its callable bases ([`BinCallableArg::uses_counts`]): the data
/// must have counts, and `auto` takes only exact ones (VCF), not
/// multihetsep's approximate ones.
pub fn fit_uses_counts(data: &Dataset, mode: BinCallableArg) -> bool {
    counts_source(data).is_some_and(|source| mode.uses_counts(source))
}

/// The input QC of `data` for `qc` and `convert`, which have no fit of
/// their own: it checks the start of a fit with the default settings (C
/// psmc's: θ0 from the data, ρ0 = θ0 / 4, t_max 15, pattern `4+5*3+4`, and
/// the callable bases of the bins when `--bin-callable auto` fits them,
/// that is when they are exact), when the data give a θ0.
pub fn default_qc_report(
    data: &Dataset,
    bin_size: usize,
    bootstrap_trunk: usize,
) -> Result<QcReport> {
    let auto = BinCallableArg::Auto;
    let callable = data.obs.callable.as_ref();
    let counts = callable.filter(|_| fit_uses_counts(data, auto));
    let start = || {
        let theta0 = initial_theta_with_callable(&data.obs.rows, counts).ok()?;
        let (n_steps, pattern) = model_layout(None, None).ok()?;
        let layout = layout_model(DEFAULT_T_MAX, n_steps, pattern, DEFAULT_MU).ok()?;
        Some(FitStart::new(
            &layout,
            theta0,
            theta0 / DEFAULT_TR_RATIO,
            true,
        ))
    };
    qc_report(data, bin_size, bootstrap_trunk, None, start(), auto)
}

/// Input QC of a loaded dataset, with the callable bases of its bins when
/// it has them. `bootstrap_trunk` is the bootstrap trunk size in bins (0
/// without a bootstrap), `free_lambdas` the number of free lambdas of the
/// model to fit and `start` its starting model, when known; `mode` is the
/// fit's `--bin-callable` (`auto` for `qc` and `convert`, which report on
/// the data as a fit with the defaults would use them), from which QC
/// learns whether the fit treats every called bin as full although the
/// data count their callable bases ([`QcContext::counts_ignored`]).
pub fn qc_report(
    data: &Dataset,
    bin_size: usize,
    bootstrap_trunk: usize,
    free_lambdas: Option<usize>,
    start: Option<FitStart>,
    mode: BinCallableArg,
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
    // The format as QC and the report name it, with the VCF mode.
    let input_format = match &data.source {
        SourceSummary::Vcf(v) => format!("vcf, {} mode", v.mode.mode.name()),
        s => s.format().name().to_string(),
    };
    let context = QcContext {
        input_format: Some(input_format),
        free_lambdas,
        start,
        dropped: dropped_groups(data),
        counts_ignored: data.obs.callable.is_some() && !fit_uses_counts(data, mode),
        ..source_context(&data.source)
    };
    try_analyze(&QcInput {
        rows: &data.obs.rows,
        row_starts: &data.obs.row_starts,
        seqs: &seqs,
        bin_size,
        bootstrap_trunk_bins: bootstrap_trunk,
        context,
        callable: data.obs.callable.as_ref(),
    })
}

/// What QC learns from how the input of `source` was read: where
/// multihetsep placed the uncalled sites, and for VCF the mask, the read
/// depth and consensus mode's filters.
fn source_context(source: &SourceSummary) -> QcContext {
    let u32_of = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    // The read depth with the bounds of a called base, if any.
    let depth = |d: &ReadDepth, bounds: Option<(u64, u64)>| DepthStats {
        mean: d.mean,
        median: d.median,
        min_depth: bounds.map(|(min, _)| u32_of(min)),
        max_depth: bounds.map(|(_, max)| u32_of(max)),
        histogram: d.histogram.clone(),
    };
    let mut context = QcContext::default();
    match source {
        SourceSummary::Psmcfa(_) => {}
        SourceSummary::Mhs(_) => context.approximate_missing_placement = true,
        SourceSummary::Vcf(v) => match &v.details {
            VcfDetails::Consensus(c) => {
                context.consensus = Some(ConsensusStats {
                    depth: depth(&c.depth, Some((c.min_depth, c.max_depth))),
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
                context.gvcf = m.read_as_gvcf;
                context.consensus = (m.depth.records > 0).then(|| ConsensusStats {
                    depth: depth(&m.depth, None),
                    total_bases: m.sites,
                    filters: Vec::new(),
                });
            }
        },
    }
    context
}

/// The dropped sequences of `data`, counted by reason.
fn dropped_groups(data: &Dataset) -> Vec<DroppedSequences> {
    let (empty, filtered): (Vec<_>, Vec<_>) = data
        .dropped
        .iter()
        .partition(|d| matches!(d.reason, DropReason::Empty));
    let mut out = Vec::new();
    if !empty.is_empty() {
        out.push(DroppedSequences {
            reason: "without any bin".to_string(),
            count: empty.len(),
            bases: None,
        });
    }
    if let (false, Some(f)) = (filtered.is_empty(), &data.filter) {
        out.push(DroppedSequences {
            reason: format!(
                "dropped by the output filter (--min-good-bases {}, --min-good-fraction {})",
                f.min_good_bases, f.min_good_fraction
            ),
            count: filtered.len(),
            bases: Some(filtered.iter().map(|d| d.extent.unwrap_or(0)).sum()),
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

/// A warning naming the options of `args` that the input, as read, does
/// not use: those of another format or of the other VCF mode, and those it
/// only checks (`--vcf-sample` in consensus mode).
fn ignored_options(args: &InputArgs, data: &Dataset) -> Option<String> {
    let (vcf_mode, what) = match &data.source {
        SourceSummary::Psmcfa(_) => (None, "psmcfa input".to_string()),
        SourceSummary::Mhs(_) => (None, "multihetsep input".to_string()),
        SourceSummary::Vcf(v) => (
            Some(v.mode.mode),
            format!("a VCF read in {} mode", v.mode.mode.name()),
        ),
    };
    let format = data.source.format();
    let flags = args.ignored_flags(format, vcf_mode);
    let mut parts = Vec::new();
    if !flags.is_empty() {
        let (list, verb) = match flags.as_slice() {
            [one] => (one.to_string(), "does"),
            [rest @ .., last] => (format!("{} and {last}", rest.join(", ")), "do"),
            [] => unreachable!(),
        };
        parts.push(format!(
            "{list} {verb} not apply to {what} and {} ignored",
            if flags.len() == 1 { "was" } else { "were" }
        ));
    }
    if args.sample_only_checked(format, vcf_mode) {
        parts.push(
            "--vcf-sample selects no sample in consensus mode, which reads the INFO fields that \
             bcftools computes over all samples together; it was only checked against the \
             VCF's one sample"
                .to_string(),
        );
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

/// The file name endings of `fmt` input (of every format for auto), each
/// also with `.gz` after it.
fn extensions(fmt: FormatArg) -> &'static [&'static str] {
    match fmt {
        FormatArg::Auto => &[".psmcfa", ".mhs", ".multihetsep", ".vcf"],
        FormatArg::Psmcfa => &[".psmcfa"],
        FormatArg::Mhs => &[".mhs", ".multihetsep"],
        FormatArg::Vcf => &[".vcf"],
    }
}

/// Whether the file name of `path` has an extension of `fmt` input.
pub fn is_supported_input_file(path: &Path, fmt: FormatArg) -> bool {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let name = name.strip_suffix(".gz").unwrap_or(&name);
    extensions(fmt).iter().any(|ext| name.ends_with(ext))
}

/// `path`, or for a directory the first input file in it by name.
pub fn resolve_input_path(path: &Path, fmt: FormatArg) -> Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))? {
        let p = entry?.path();
        if p.is_file() && is_supported_input_file(&p, fmt) {
            files.push(p);
        }
    }
    files.into_iter().min().ok_or_else(|| {
        let names: Vec<String> = extensions(fmt)
            .iter()
            .map(|ext| format!("*{ext}"))
            .collect();
        anyhow!(
            "directory {} holds no {} file ({}, with or without .gz)",
            path.display(),
            fmt.label(),
            names.join(", ")
        )
    })
}

/// [`load`], naming the input in the error unless the reader's error does
/// (the readers quote it, as `{path:?}`), and with the option to use when
/// the automatic depth bounds cannot read a pipe twice.
pub fn load_input(path: &Path, opts: &InputOptions) -> Result<Dataset> {
    load(path, opts).map_err(|e| {
        let quoted = format!("{path:?}");
        let e = if e.chain().any(|cause| cause.to_string().contains(&quoted)) {
            e
        } else {
            e.context(format!("failed to read the input {}", path.display()))
        };
        let two_passes = opts.consensus.depth == DepthThresholds::Auto
            && !path.is_file()
            && format!("{e:#}").contains("automatic depth thresholds need two passes");
        if two_passes {
            anyhow!(
                "{e:#}. Give them as --depth MIN:MAX to read the VCF in one pass (vcf2fq's \
                 defaults are 3:100000)"
            )
        } else {
            e
        }
    })
}

/// [`Dataset::ensure_usable`], with the options that keep the sequences
/// when the output filter left none; the loader's warning before the error
/// lists them.
pub fn ensure_usable(data: &Dataset) -> Result<()> {
    let filtered = data
        .dropped
        .iter()
        .filter(|d| matches!(d.reason, DropReason::Filter { .. }))
        .count();
    if data.obs.rows.is_empty() && filtered > 0 {
        bail!(
            "no sequence is left to analyse: fq2psmcfa's output filter dropped {} (the \
             'dropped because' warning lists {}); lower --min-good-bases or \
             --min-good-fraction, or pass --min-good-bases 0 --min-good-fraction 0 to keep \
             every sequence",
            counted(filtered, "sequence", "sequences"),
            plural(filtered, "it", "them")
        );
    }
    data.ensure_usable()
}

/// Lines telling the user how the input was read: what the input QC report
/// does not say of it.
fn describe(data: &Dataset) -> Vec<String> {
    match &data.source {
        SourceSummary::Psmcfa(p) => {
            // C psmc counts the empty records in n_seqs (its MM line).
            let empty = match p.empty_records {
                0 => String::new(),
                n => format!(
                    " (n_seqs includes {})",
                    counted(n, "empty record", "empty records")
                ),
            };
            vec![format!(
                "psmcfa: n_seqs:{}, sum_L:{}, sum_n:{}, as C psmc counts them{empty}",
                p.records, p.called_bins, p.het_bins
            )]
        }
        SourceSummary::Mhs(m) => vec![format!(
            "multihetsep: {} of {} haplotypes; for the pair, {} heterozygous, {} homozygous \
             and {} missing rows, and {} uncalled sites (placed at the start of each interval \
             between rows)",
            counted(m.rows as usize, "row", "rows"),
            m.haplotypes,
            count(m.het_rows),
            count(m.hom_rows),
            count(m.missing_rows),
            count(m.uncalled_sites)
        )],
        SourceSummary::Vcf(v) => {
            let mode = v.mode.mode.name();
            let how = if v.mode.forced {
                format!("VCF read in {mode} mode (--vcf-mode {mode})")
            } else {
                format!("VCF read in {}", v.mode.reason)
            };
            let details = match &v.details {
                VcfDetails::Consensus(c) => describe_consensus(c),
                VcfDetails::Mask(m) => vec![describe_mask(m)],
            };
            std::iter::once(how).chain(details).collect()
        }
    }
}

/// How consensus mode called the bases (the QC report gives the depth and
/// the share of each loss).
fn describe_consensus(c: &ConsensusSummary) -> Vec<String> {
    let losses: Vec<String> = LossReason::ALL
        .iter()
        .filter(|&&r| c.lost.get(r) > 0)
        .map(|&r| format!("{} {}", r.label(), count(c.lost.get(r))))
        .collect();
    vec![
        format!(
            "consensus: bases kept at depth {}-{}{}, MQ >= {}, quality >= {}, {} bp from indels",
            c.min_depth,
            c.max_depth,
            if c.auto_depth { " (auto)" } else { "" },
            c.min_mq,
            c.min_qual,
            c.indel_window
        ),
        format!(
            "consensus: {} of {} bases called; missing: {}",
            count(c.good_bases),
            count(c.bases),
            if losses.is_empty() {
                "none".to_string()
            } else {
                losses.join(", ")
            }
        ),
    ]
}

/// What mask mode read of the records (the QC report gives the share of
/// missing sites).
fn describe_mask(m: &VcfSummary) -> String {
    let extended = if m.contig_lengths_used.is_empty() {
        String::new()
    } else {
        format!(
            "; extended to the ##contig length: {}",
            m.contig_lengths_used.join(", ")
        )
    };
    let n = |k: u64, one, many| counted(k as usize, one, many);
    format!(
        "VCF: {}, among them {} ({} SNPs used), {}, {} filtered and {} without a genotype; {} \
         with several records; {} of {} sites missing{extended}",
        n(m.records, "record", "records"),
        n(
            m.het_snp_records,
            "heterozygous SNP record",
            "heterozygous SNP records"
        ),
        count(m.het_snps),
        n(
            m.het_non_snp_records,
            "heterozygous non-SNP",
            "heterozygous non-SNPs"
        ),
        count(m.filtered_records),
        count(m.missing_genotypes),
        n(m.multi_record_positions, "position", "positions"),
        count(m.missing_sites),
        count(m.sites)
    )
}

/// Say how the input was read: what it holds, and the options of `args`
/// it does not use.
pub fn announce_input(data: &Dataset, args: &InputArgs, sink: Sink) {
    for line in describe(data) {
        sink(Event::Info(line));
    }
    if let Some(line) = ignored_options(args, data) {
        sink(Event::Warning(line));
    }
}

/// Show the input QC `report` of `data`, or why it could not be made,
/// after the loader's warnings that it does not repeat, and write it as
/// JSON to `qc_json` when given. A failed report is only a warning, so that
/// it never stops a run; it is returned as `None`.
pub fn report_qc(
    data: &Dataset,
    report: Result<QcReport>,
    qc_json: Option<&Path>,
    sink: Sink,
) -> Result<Option<QcReport>> {
    let qc = match report {
        Ok(report) => Some(report),
        Err(e) => {
            let unwritten = qc_json.map_or(String::new(), |path| {
                format!(", so {} is not written", path.display())
            });
            sink(Event::Warning(format!("input QC failed{unwritten}: {e:#}")));
            sink(Event::Info(totals(data)));
            None
        }
    };
    for line in loader_warnings(data, qc.as_ref()) {
        sink(Event::Warning(line));
    }
    let Some(report) = &qc else {
        return Ok(None);
    };
    sink(Event::Qc {
        text: render_text(report),
        warnings: report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .map(|f| format!("QC {}: {}", f.code.as_str(), f.message))
            .collect(),
    });
    if let Some(path) = qc_json {
        write_qc_json(path, report)?;
        if !is_stdout(path) {
            sink(Event::Output(format!("QC report: {}", path.display())));
        }
    }
    Ok(qc)
}

/// The sequences and bins of `data`, for when the input QC report, which
/// gives them, could not be made.
fn totals(data: &Dataset) -> String {
    let bins = data.n_bins();
    let missing = data.obs.rows.iter().flatten().filter(|&&b| b >= 2).count();
    // As the QC report gives shares (0.0% of nothing).
    let share = |n: usize| pct(n as f64 / bins.max(1) as f64);
    format!(
        "Input: {} kept, {} left out; {} ({} heterozygous, {} missing)",
        counted(data.seqs.len(), "sequence", "sequences"),
        count(data.dropped.len() as u64),
        counted(bins, "bin", "bins"),
        share(data.het_bins()),
        share(missing)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TempDir;

    #[test]
    fn a_directory_gives_its_first_input_file() {
        let dir = TempDir::new("dir_input");
        for name in ["notes.txt", "c.psmcfa", "b.vcf.gz"] {
            fs::write(dir.join(name), "").unwrap();
        }
        assert_eq!(
            resolve_input_path(&dir, FormatArg::Auto).unwrap(),
            dir.join("b.vcf.gz")
        );
        assert_eq!(
            resolve_input_path(&dir, FormatArg::Psmcfa).unwrap(),
            dir.join("c.psmcfa")
        );
        let err = resolve_input_path(&dir, FormatArg::Mhs)
            .unwrap_err()
            .to_string();
        assert!(
            err.ends_with("holds no multihetsep file (*.mhs, *.multihetsep, with or without .gz)"),
            "{err}"
        );
        let file = dir.join("notes.txt");
        assert_eq!(resolve_input_path(&file, FormatArg::Auto).unwrap(), file);
    }

    /// The input is named in the error unless the reader quotes its path:
    /// a sample name that holds the path does not name it.
    #[test]
    fn errors_name_the_input() {
        let dir = TempDir::new("load_error");
        let path = dir.join("in.vcf");
        fs::write(
            &path,
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
        )
        .unwrap();
        let shown = path.display().to_string();
        let argv = ["psmc-rs", "in", "out", "1", "--vcf-mode", "consensus"];
        let cli = crate::cli::Cli::try_parse_args(argv.into_iter().chain(["--vcf-sample", &shown]))
            .unwrap();
        let err = load_input(&path, &cli.infer.input.to_options()).unwrap_err();
        assert_eq!(
            format!("{err:#}"),
            format!("failed to read the input {shown}: VCF sample '{shown}' not found in header")
        );
    }

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

    /// The psmcfa line gives C psmc's counts as its MM line writes them,
    /// with the empty records that n_seqs counts.
    #[test]
    fn a_psmcfa_input_is_described_by_c_psmcs_counts() {
        let dir = TempDir::new("describe");
        let path = dir.join("in.psmcfa");
        let cli = crate::cli::Cli::try_parse_args(["psmc-rs", "in", "out", "1"]).unwrap();
        let opts = cli.infer.input.to_options();
        fs::write(&path, ">a\nTTKTNTTTTTKT\n>b\n\n").unwrap();
        let data = load(&path, &opts).unwrap();
        assert_eq!(
            describe(&data),
            [
                "psmcfa: n_seqs:2, sum_L:11, sum_n:2, as C psmc counts them (n_seqs includes 1 \
              empty record)"
            ]
        );
        assert_eq!(
            totals(&data),
            "Input: 1 sequence kept, 1 left out; 12 bins (16.7% heterozygous, 8.3% missing)"
        );
    }
}
