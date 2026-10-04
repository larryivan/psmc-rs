//! Warnings and recommendations derived from the QC metrics.
//!
//! Each check has a stable [`QcCode`] whose rustdoc gives the rationale, and
//! named threshold constants.

use serde::Serialize;

use super::fmt::{bases, count, pct, sig};
use super::names::SeqKind;
use super::stats::median;
use super::{
    QcInput, QcReport, SATURATION_TARGET_SHARE, SATURATION_WARN_SHARE, SeqInfo,
    largest_trunk_within,
};

/// [`QcCode::Depth`] warns below this mean read depth (Nadachowska-Brzyska
/// et al. 2016 recommend at least 18x for PSMC).
pub const MIN_MEAN_DEPTH: f64 = 18.0;

/// [`QcCode::Depth`] warns when one consensus filter removes more than this
/// share of the bases.
pub const MAX_FILTER_FRACTION: f64 = 0.30;

/// `N` fraction above which [`QcCode::HighMissing`] is a warning.
pub const HIGH_MISSING_WARN_FRACTION: f64 = 0.5;

/// `N` fraction above which [`QcCode::HighMissing`] is reported as info:
/// Nadachowska-Brzyska et al. (2016, Mol Ecol 25:1058) recommend at most 25%
/// missing data for PSMC.
pub const HIGH_MISSING_INFO_FRACTION: f64 = 0.25;

/// [`QcCode::FewHets`] fires below this many `K` bins.
pub const MIN_HET_BINS: u64 = 1_000;

/// [`QcCode::TooShort`] fires below this many called bases.
pub const MIN_CALLED_BASES: u64 = 100_000_000;

/// [`QcCode::ShortSequences`] fires when more than this share of all bins
/// lies in short sequences.
pub const SHORT_SEQUENCES_MAX_FRACTION: f64 = 0.20;

/// Sequences need at least this many bases in called bins to enter the
/// autosomal reference heterozygosity ([`super::HetReference`]) and the
/// [`QcCode::HetOutlier`] test: below it the `K` count of a sequence is too
/// noisy (a 1 Mb sequence at 5% `K` bins has 500 of them, +-4.5%).
pub const REFERENCE_MIN_CALLED_BASES: u64 = 1_000_000;

/// A sex chromosome or organelle looks hemizygous below this ratio of its
/// heterozygosity to the autosomal reference ([`QcCode::SexOrOrganelle`]).
pub const HEMIZYGOUS_MAX_RATIO: f64 = 0.25;

/// ... and only when the reference rate predicts at least this many `K` bins
/// in it, so that the shortfall is not chance: seeing at most 2 where 10 are
/// expected has a Poisson probability of 0.3%.
pub const HEMIZYGOUS_MIN_EXPECTED_HET_BINS: f64 = 10.0;

/// [`QcCode::HetOutlier`] flags a sequence whose log heterozygosity is more
/// than this many median absolute deviations from the median. Iglewicz and
/// Hoaglin's (1993) modified z-score cut-off of 3.5 is 3.5 / 0.6745 = 5.2 MAD.
pub const HET_OUTLIER_MADS: f64 = 5.0;

/// ... and at least this many times above or below the median, so that small
/// differences between very uniform sequences (tiny MAD) are not flagged.
pub const HET_OUTLIER_MIN_FOLD: f64 = 1.5;

/// [`QcCode::HetOutlier`] needs at least this many sequences with
/// [`REFERENCE_MIN_CALLED_BASES`], or the median and MAD mean little.
pub const HET_OUTLIER_MIN_SEQUENCES: usize = 5;

/// At most this many sequence names are listed in a message (all are in
/// [`Finding::sequences`]).
const MAX_LISTED: usize = 10;

/// The kind of a [`Finding`]. The serialised names (`HIGH_MISSING`, ...) are
/// stable and meant for scripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum QcCode {
    /// A VCF was read in mask mode without a positive callable-sites mask
    /// (warning).
    ///
    /// The VCF reader then counts every position without a record as a
    /// called homozygous site. A variants-only VCF (the GATK and bcftools
    /// default) has no records for positions that could not be called (no or
    /// low coverage, low mapping quality, repeats), so all of them count as
    /// homozygous: missing data disappear, the `K` fraction is diluted, and
    /// θ and Ne come out too low, while uncalled stretches look like long
    /// runs of recent common ancestry.
    VariantsOnlyNoMask,
    /// Many bins are missing: more than [`HIGH_MISSING_WARN_FRACTION`]
    /// (warning) or [`HIGH_MISSING_INFO_FRACTION`] (info).
    ///
    /// `N` bins carry no information, so the effective input shrinks. Worse,
    /// the `fq2psmcfa` rule calls a bin as soon as 10% of its sites are
    /// called, so heavy missingness also leaves many partly called `T`/`K`
    /// bins; a partly called bin is less likely to show a heterozygous site,
    /// which biases θ and Ne low. Missing data also concentrate in repeats
    /// and divergent regions, so the remaining sites are not a random sample
    /// of the genome.
    HighMissing,
    /// More than [`SATURATION_WARN_SHARE`] of `K` bins are expected to hold
    /// two or more heterozygous sites (warning; see [`super::Saturation`]).
    ///
    /// PSMC's emission model only sees whether a bin holds a heterozygous
    /// site. The chance of one in a bin, 1 - exp(-θ t), flattens as θ t
    /// grows, so saturated bins mostly cost resolution between old TMRCAs:
    /// the ancient part of the curve. The recommendation is the largest bin
    /// size that brings the share below [`SATURATION_TARGET_SHARE`].
    BinSaturation,
    /// Fewer than [`MIN_HET_BINS`] heterozygous bins (warning).
    ///
    /// PSMC learns each λ from the genealogical segments whose TMRCA falls
    /// in its intervals; psmc's README asks that after 20 iterations about
    /// 10 recombinations be inferred in the intervals each free parameter
    /// spans. Recombinations and mutations both accumulate in proportion to
    /// the TMRCA, so the data hold about ρ/θ recombinations per heterozygous
    /// site: a quarter at C psmc's starting ratio θ/ρ = 4, and every `K` bin
    /// holds at least one site ([`super::QcReport::expected_recombinations`]
    /// computes C's number). 1,000 `K` bins thus mean roughly 250
    /// recombinations (more when bins hold several sites). Under a constant population size the default pattern
    /// `4+25*2+4+6` (28 free λ) spreads them unevenly: the most recent group
    /// gets 3.6% and the oldest 0.005% of the segments, and only 13 of the 28
    /// groups reach 10. Fewer `K` bins leave most parameters undetermined.
    FewHets,
    /// Less than [`MIN_CALLED_BASES`] of called sequence (warning).
    ///
    /// PSMC was developed and tested on whole genomes (Li & Durbin 2011:
    /// gigabases per human genome). By the count in [`QcCode::FewHets`], 100
    /// Mb at human heterozygosity (1e-3 per bp, 1e5 heterozygous sites, about
    /// 25,000 recombinations) give every group of the default pattern but the
    /// oldest at least 10 recombinations under a constant size; at 30 Mb the
    /// second-oldest group gets about 5. Species with lower heterozygosity
    /// need more sequence.
    TooShort,
    /// More than [`SHORT_SEQUENCES_MAX_FRACTION`] of the bins lie in
    /// sequences shorter than 1 Mb ([`super::SHORT_SEQUENCE_BASES`]) or than
    /// the bootstrap trunk (warning).
    ///
    /// Every sequence is its own HMM chain, started from the stationary
    /// distribution. Segments of recent common ancestry are long (about
    /// 1 / (2 r T) bp for a TMRCA of T generations: 500 kb for T = 100 at
    /// r = 1e-8), so short sequences cut them and the recent part of the
    /// curve loses information. The bootstrap cannot split a sequence shorter
    /// than its trunk, so it resamples such sequences whole, as units of
    /// unequal size, rather than trunks of the chosen size: the resampling
    /// is only as fine as the sequences allow.
    ShortSequences,
    /// Sequences named like sex chromosomes or organelle genomes (info; see
    /// [`super::classify_name`]); a warning when one looks hemizygous: its
    /// heterozygosity is below [`HEMIZYGOUS_MAX_RATIO`] of the autosomal
    /// reference ([`super::HetReference`]) while that reference predicts at
    /// least [`HEMIZYGOUS_MIN_EXPECTED_HET_BINS`] `K` bins in it.
    ///
    /// PSMC models the two copies of one diploid genome sharing one
    /// history. A hemizygous sequence (X or Y in males of XY species, Z or W
    /// in females of ZW species, organelle genomes) has a single copy, so its
    /// only heterozygous calls are errors: PSMC reads it as a long stretch of
    /// recent common ancestry, which pulls θ down and adds a false signal of
    /// recent coalescence (small recent Ne). Two-copy X and Z chromosomes are
    /// expected near 3/4 of the autosomal heterozygosity with equal sex
    /// ratios (often lower in practice) and still follow a different history
    /// (effective size, mutation rate), which is why most PSMC analyses use
    /// autosomes only; a single copy shows only genotyping errors, usually a
    /// few percent of the autosomal level. A quarter separates the two.
    SexOrOrganelle,
    /// Sequences whose heterozygosity is far from the median of the others
    /// (info): log θ more than [`HET_OUTLIER_MADS`] median absolute
    /// deviations and at least [`HET_OUTLIER_MIN_FOLD`]-fold from the median,
    /// among sequences with at least [`REFERENCE_MIN_CALLED_BASES`] in called
    /// bins (sex chromosomes and organelles excluded), when there are at
    /// least [`HET_OUTLIER_MIN_SEQUENCES`] of them.
    ///
    /// Median and MAD are not moved by the outliers themselves. Elevated
    /// heterozygosity points to contamination, collapsed duplications or
    /// copy-number variants (paralogous differences called as heterozygous)
    /// or reference problems. Reduced heterozygosity points to sex-linked
    /// scaffolds whose names do not say so (near zero when hemizygous, as on
    /// the Z-linked scaffolds of a female bird), long runs of homozygosity
    /// (inbreeding, which is real signal), deletions, or a mask that removed
    /// most variable sites.
    HetOutlier,
    /// Multihetsep input, where uncalled sites have no positions (info).
    ///
    /// Multihetsep gives, for each segregating site, how many sites since
    /// the previous one were called, but not where the uncalled ones are.
    /// The reader places them together at the start of each interval
    /// ([`crate::io::mhs::read_mhs_with`]). Called and heterozygous sites
    /// keep their numbers and the heterozygous sites their positions, but a
    /// bin is `N` only when more than 90% of its sites are uncalled, so
    /// moving the uncalled sites changes which bins are `N`, `T` or `K`
    /// compared with `fq2psmcfa` on the same calls. VCF input with a callable
    /// mask bins exactly.
    ApproxMissingPlacement,
    /// Consensus-mode read depth (warning): mean depth below
    /// [`MIN_MEAN_DEPTH`], or one filter removing more than
    /// [`MAX_FILTER_FRACTION`] of the bases.
    ///
    /// Calling a heterozygous site takes reads from both alleles: with `d`
    /// reads both appear with probability 1 - 2^(1-d) (94% at 5 reads,
    /// before the caller's own thresholds), so at low depth heterozygous
    /// sites are lost and θ and Ne come out low. Nadachowska-Brzyska et al.
    /// (2016, Mol Ecol 25:1058) found PSMC curves to shift at low coverage
    /// and recommend at least 18x. psmc's README suggests minimum and maximum
    /// depths of a third and twice the mean (`vcf2fq -d`, `-D`). A filter that
    /// alone removes more than 30% of the bases usually has a threshold that
    /// does not fit the data (a depth window too narrow for the coverage, a
    /// mapping-quality cut-off too strict for the reference), and it shrinks
    /// and biases the called set as in [`QcCode::HighMissing`].
    Depth,
    /// The loader left out whole sequences (info), counted by reason.
    ///
    /// The metrics cover only the sequences that were kept, so a dropped
    /// chromosome (for example one whose name the mask does not match) would
    /// otherwise go unnoticed.
    DroppedSequences,
}

impl QcCode {
    /// Every code, in report order within a severity.
    pub const ALL: &'static [QcCode] = &[
        Self::VariantsOnlyNoMask,
        Self::HighMissing,
        Self::BinSaturation,
        Self::FewHets,
        Self::TooShort,
        Self::ShortSequences,
        Self::SexOrOrganelle,
        Self::HetOutlier,
        Self::ApproxMissingPlacement,
        Self::Depth,
        Self::DroppedSequences,
    ];

    /// The stable name, as serialised (`HIGH_MISSING`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VariantsOnlyNoMask => "VARIANTS_ONLY_NO_MASK",
            Self::HighMissing => "HIGH_MISSING",
            Self::BinSaturation => "BIN_SATURATION",
            Self::FewHets => "FEW_HETS",
            Self::TooShort => "TOO_SHORT",
            Self::ShortSequences => "SHORT_SEQUENCES",
            Self::SexOrOrganelle => "SEX_OR_ORGANELLE",
            Self::HetOutlier => "HET_OUTLIER",
            Self::ApproxMissingPlacement => "APPROX_MISSING_PLACEMENT",
            Self::Depth => "DEPTH",
            Self::DroppedSequences => "DROPPED_SEQUENCES",
        }
    }
}

/// How serious a [`Finding`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Worth knowing; the results are probably still usable.
    Info,
    /// Likely to bias the results or make them unreliable.
    Warning,
}

impl Severity {
    /// `"info"` or `"warning"`, as serialised.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
        }
    }
}

/// One warning or note about the input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// Stable code.
    pub code: QcCode,
    /// Info or warning.
    pub severity: Severity,
    /// What was measured, with the numbers.
    pub message: String,
    /// What to do about it.
    pub recommendation: String,
    /// Sequences the finding is about, if it names any.
    pub sequences: Vec<String>,
}

impl Finding {
    fn new(code: QcCode, severity: Severity, message: String, recommendation: String) -> Self {
        Self {
            code,
            severity,
            message,
            recommendation,
            sequences: Vec::new(),
        }
    }
}

/// Run every check on `report` (computed from `input`, with `seqs` holding
/// every sequence); warnings first, then in [`QcCode`] order. Also returns
/// the indices of the sequences the findings name, ascending.
pub(super) fn run(
    input: &QcInput,
    report: &QcReport,
    seqs: &[SeqInfo],
) -> (Vec<Finding>, Vec<usize>) {
    let seq_bins: Vec<u64> = seqs.iter().map(|s| s.stats.bins).collect();
    let mut named = Vec::new();
    let mut out: Vec<Finding> = [
        variants_only_no_mask(input, report),
        high_missing(report),
        bin_saturation(report),
        few_hets(report),
        too_short(report),
        short_sequences(report, &seq_bins),
        sex_or_organelle(report, seqs, &mut named),
        het_outliers(report, seqs, &mut named),
        approx_missing_placement(input),
        depth(report),
        dropped_sequences(report),
    ]
    .into_iter()
    .flatten()
    .collect();
    out.sort_by_key(|f| (f.severity != Severity::Warning, f.code));
    named.sort_unstable();
    named.dedup();
    (out, named)
}

/// `names` joined with commas, at most [`MAX_LISTED`] of them.
fn listing(names: &[String]) -> String {
    let mut s = names
        .iter()
        .take(MAX_LISTED)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if names.len() > MAX_LISTED {
        s.push_str(&format!(" and {} more", names.len() - MAX_LISTED));
    }
    s
}

fn sex_or_organelle(
    report: &QcReport,
    seqs: &[SeqInfo],
    named: &mut Vec<usize>,
) -> Option<Finding> {
    let flagged: Vec<&SeqInfo> = seqs.iter().filter(|s| s.kind.is_some()).collect();
    if flagged.is_empty() {
        return None;
    }
    let reference = report.het_reference.as_ref();
    let mut hemizygous = Vec::new();
    let mut described = Vec::new();
    for s in &flagged {
        named.push(s.index);
        let kind = s.kind.map_or("", SeqKind::label);
        let called_bins = s.stats.t_bins + s.stats.k_bins;
        let judged = match (reference, s.stats.theta0_per_bin) {
            (Some(r), Some(theta)) if r.theta0_per_bin > 0.0 => {
                let ratio = theta / r.theta0_per_bin;
                let expected_k = called_bins as f64 * -(-r.theta0_per_bin).exp_m1();
                Some((ratio, expected_k))
            }
            _ => None,
        };
        let note = match judged {
            Some((ratio, expected)) if expected >= HEMIZYGOUS_MIN_EXPECTED_HET_BINS => {
                if ratio < HEMIZYGOUS_MAX_RATIO {
                    hemizygous.push(s.name.clone());
                    format!(
                        "{}x the autosomal heterozygosity, hemizygous?",
                        sig(ratio, 2)
                    )
                } else {
                    format!("{}x the autosomal heterozygosity", sig(ratio, 2))
                }
            }
            Some(_) => "too few called bins to judge".to_string(),
            None if called_bins == 0 => "no called bins".to_string(),
            None => "nothing to compare with".to_string(),
        };
        described.push(format!("{} ({kind}, {note})", s.name));
    }
    let names: Vec<String> = flagged.iter().map(|s| s.name.clone()).collect();
    let (severity, consequence, advice) = if hemizygous.is_empty() {
        (
            Severity::Info,
            "PSMC models one diploid history; sex chromosomes have their own effective \
             size and mutation rate, and organelles have one copy and do not recombine."
                .to_string(),
            "Most PSMC analyses use autosomes only; consider excluding these sequences."
                .to_string(),
        )
    } else {
        (
            Severity::Warning,
            format!(
                "{} {} far less heterozygous than the autosomes, as a single (hemizygous) \
                 copy is; PSMC reads such a sequence as a long stretch of recent common \
                 ancestry, which lowers theta and adds a false signal of small recent Ne.",
                listing(&hemizygous),
                if hemizygous.len() == 1 { "is" } else { "are" },
            ),
            format!(
                "Exclude {} (keep autosomes only, e.g. by restricting the mask, VCF or \
                 psmcfa to them).",
                listing(&hemizygous)
            ),
        )
    };
    let mut f = Finding::new(
        QcCode::SexOrOrganelle,
        severity,
        format!(
            "{} {} named like a sex chromosome or organelle: {}. {consequence}",
            count(flagged.len() as u64),
            if flagged.len() == 1 {
                "sequence is"
            } else {
                "sequences are"
            },
            listing(&described),
        ),
        advice,
    );
    f.sequences = names;
    Some(f)
}

fn het_outliers(report: &QcReport, seqs: &[SeqInfo], named: &mut Vec<usize>) -> Option<Finding> {
    let eligible: Vec<(&SeqInfo, f64)> = seqs
        .iter()
        .filter(|s| s.kind.is_none())
        .filter(|s| s.stats.called_bases_from_bins >= REFERENCE_MIN_CALLED_BASES)
        .filter_map(|s| s.stats.theta0_per_bin.map(|t| (s, t.ln())))
        .collect();
    if eligible.len() < HET_OUTLIER_MIN_SEQUENCES {
        return None;
    }
    let mut logs: Vec<f64> = eligible.iter().map(|&(_, x)| x).collect();
    let center = median(&mut logs);
    if !center.is_finite() {
        return None;
    }
    let mut deviations: Vec<f64> = eligible.iter().map(|&(_, x)| (x - center).abs()).collect();
    let mad = median(&mut deviations);
    let limit = (HET_OUTLIER_MADS * mad).max(HET_OUTLIER_MIN_FOLD.ln());
    let mut outliers: Vec<(&SeqInfo, f64)> = eligible
        .iter()
        .filter(|&&(_, x)| (x - center).abs() > limit)
        .map(|&(s, x)| (s, x - center))
        .collect();
    if outliers.is_empty() {
        return None;
    }
    outliers.sort_by(|a, b| {
        b.1.abs()
            .total_cmp(&a.1.abs())
            .then(a.0.index.cmp(&b.0.index))
    });
    let described: Vec<String> = outliers
        .iter()
        .map(|(s, d)| format!("{} ({}x)", s.name, sig(d.exp(), 2)))
        .collect();
    named.extend(outliers.iter().map(|(s, _)| s.index));
    let bin_size = report.bin_size as f64;
    let mut f = Finding::new(
        QcCode::HetOutlier,
        Severity::Info,
        format!(
            "{} of {} sequences with at least {} in called bins have a heterozygosity far \
             from their median of {} per kb (more than {} MADs of log theta and {}-fold): \
             {}.",
            count(outliers.len() as u64),
            count(eligible.len() as u64),
            bases(REFERENCE_MIN_CALLED_BASES),
            sig(1000.0 * center.exp() / bin_size, 3),
            HET_OUTLIER_MADS,
            HET_OUTLIER_MIN_FOLD,
            listing(&described),
        ),
        "High heterozygosity can come from contamination, collapsed duplications or \
         copy-number variants, or reference errors; low heterozygosity from sex-linked \
         scaffolds (near zero when hemizygous), runs of homozygosity (inbreeding, a real \
         signal), deletions or an over-strict mask. Check depth, mapping quality, the mask \
         and the chromosome assignment of these sequences, and exclude any that are \
         artefacts or hemizygous."
            .to_string(),
    );
    f.sequences = outliers.iter().map(|(s, _)| s.name.clone()).collect();
    Some(f)
}

fn variants_only_no_mask(input: &QcInput, report: &QcReport) -> Option<Finding> {
    if !input.context.variants_only_no_mask {
        return None;
    }
    Some(Finding::new(
        QcCode::VariantsOnlyNoMask,
        Severity::Warning,
        format!(
            "The VCF was read without a callable-sites mask, so every position without a \
             record counted as a called homozygous site ({} of bins are missing). In a \
             variants-only VCF that hides every uncalled position: theta ({}) and Ne are \
             biased low, and uncalled stretches look like recent common ancestry.",
            pct(report.totals.missing_fraction),
            report
                .totals
                .theta0_per_bin
                .map_or("n/a".to_string(), |t| format!("{} per bin", sig(t, 3))),
        ),
        "Pass a BED of callable sites (--vcf-mask), e.g. from your variant-calling \
         pipeline or a depth profile, or call all sites (`bcftools mpileup -f ref.fa \
         aln.bam | bcftools call -c`) so that every called position has a record."
            .to_string(),
    ))
}

fn high_missing(report: &QcReport) -> Option<Finding> {
    let t = &report.totals;
    let f = t.missing_fraction;
    let severity = if f > HIGH_MISSING_WARN_FRACTION {
        Severity::Warning
    } else if f > HIGH_MISSING_INFO_FRACTION {
        Severity::Info
    } else {
        return None;
    };
    let sites = match t.called_bases {
        Some(called) if t.length_bases > 0 => format!(
            "; {} of sites are uncalled",
            pct(1.0 - called as f64 / t.length_bases as f64)
        ),
        _ => String::new(),
    };
    Some(Finding::new(
        QcCode::HighMissing,
        severity,
        format!(
            "{} of bins are missing (N: more than 90% of their sites uncalled){sites}, so \
             only {} of sequence is used. A bin is called when just 10% of its sites are, \
             so heavy missingness also leaves partly called bins, which show fewer \
             heterozygous sites and bias theta and Ne low.",
            pct(f),
            bases(t.called_bases_from_bins),
        ),
        "Check what removes the data: depth and mapping-quality filters, a callable mask \
         that does not match the chromosome names, or low coverage. Nadachowska-Brzyska \
         et al. (2016) recommend at most 25% missing data for PSMC."
            .to_string(),
    ))
}

fn bin_saturation(report: &QcReport) -> Option<Finding> {
    let sat = report.saturation.as_ref()?;
    if sat.multi_het_share <= SATURATION_WARN_SHARE {
        return None;
    }
    let s = report.bin_size;
    let advice = match sat.max_bin_size_below_target {
        Some(r) => format!(
            "Re-bin the per-site data at {r} bp per bin or less, the largest size that \
             keeps the share below {} at this heterozygosity (`fq2psmcfa -s {r}`, or the \
             bin size option for VCF and multihetsep input), and scale the results with \
             the same bin size.",
            pct(SATURATION_TARGET_SHARE)
        ),
        None => "Re-bin the per-site data with smaller bins.".to_string(),
    };
    Some(Finding::new(
        QcCode::BinSaturation,
        Severity::Warning,
        format!(
            "At {s} bp per bin, {} of called bins are heterozygous (K), which implies {} \
             heterozygous sites per kb and about {} of K bins holding two or more of them \
             (more if heterozygosity varies along the genome). PSMC only sees whether a \
             bin holds one, so the oldest part of the curve loses resolution.",
            pct(report.totals.het_bin_fraction.unwrap_or(0.0)),
            sig(1000.0 * sat.het_per_site, 3),
            pct(sat.multi_het_share),
        ),
        advice,
    ))
}

fn few_hets(report: &QcReport) -> Option<Finding> {
    let k = report.totals.k_bins;
    if k >= MIN_HET_BINS {
        return None;
    }
    let recomb = report.expected_recombinations.map_or(String::new(), |n| {
        format!(
            "; C psmc's starting model expects about {} recombinations in these data",
            count(n.round() as u64)
        )
    });
    Some(Finding::new(
        QcCode::FewHets,
        Severity::Warning,
        format!(
            "Only {} heterozygous (K) bins, fewer than {}{recomb}. psmc's documentation \
             asks for about 10 recombinations in the intervals of each free parameter, and \
             the default pattern 4+25*2+4+6 has 28 of them.",
            count(k),
            count(MIN_HET_BINS),
        ),
        "Add data (more of the genome, a less strict mask) or fit a --pattern with fewer, \
         wider parameter groups; read the curve's oldest and most recent parts with \
         caution."
            .to_string(),
    ))
}

/// Called length: the loader's site count when every sequence has one,
/// otherwise the bases of the called bins.
fn called_length(report: &QcReport) -> (u64, &'static str) {
    match report.totals.called_bases {
        Some(n) => (n, "called sites"),
        None => (report.totals.called_bases_from_bins, "bases in called bins"),
    }
}

fn too_short(report: &QcReport) -> Option<Finding> {
    let (called, what) = called_length(report);
    if called >= MIN_CALLED_BASES {
        return None;
    }
    Some(Finding::new(
        QcCode::TooShort,
        Severity::Warning,
        format!(
            "Only {} of called sequence ({what}), less than {}. PSMC was developed for \
             whole genomes; below ~100 Mb, even at human-like heterozygosity, the \
             parameters for the oldest times get fewer than the ~10 recombinations each \
             that psmc's documentation asks for, so the ends of the curve are poorly \
             determined.",
            bases(called),
            bases(MIN_CALLED_BASES),
        ),
        "Use more of the genome if possible (all autosomes, a less strict mask); \
         otherwise check the curve with bootstrap replicates and read its ends with \
         caution."
            .to_string(),
    ))
}

fn short_sequences(report: &QcReport, seq_bins: &[u64]) -> Option<Finding> {
    let l = &report.lengths;
    let short_1mb = l.below_1mb.fraction > SHORT_SEQUENCES_MAX_FRACTION;
    let short_trunk = l
        .below_trunk
        .as_ref()
        .is_some_and(|s| s.fraction > SHORT_SEQUENCES_MAX_FRACTION);
    if !short_1mb && !short_trunk {
        return None;
    }
    let mut parts = Vec::new();
    let mut advice = Vec::new();
    if short_1mb {
        parts.push(format!(
            "{} of bins lie in {} sequences shorter than 1 Mb, where sequence ends (each \
             sequence is a separate HMM chain) cut the long segments of recent common \
             ancestry",
            pct(l.below_1mb.fraction),
            count(l.below_1mb.sequences as u64)
        ));
        advice.push(
            "Prefer a more contiguous assembly, or drop sequences shorter than 1 Mb if \
             enough data remain."
                .to_string(),
        );
    }
    if let (true, Some(below)) = (short_trunk, &l.below_trunk) {
        parts.push(format!(
            "{} of bins lie in {} sequences shorter than the bootstrap trunk ({} bins), \
             which the bootstrap can only resample whole",
            pct(below.fraction),
            count(below.sequences as u64),
            count(report.bootstrap_trunk_bins as u64)
        ));
        let trunk = largest_trunk_within(seq_bins, SHORT_SEQUENCES_MAX_FRACTION);
        advice.push(format!(
            "For the bootstrap, a trunk of at most {} bins keeps at least {} of bins in \
             sequences of one trunk or more.",
            count(trunk),
            pct(1.0 - SHORT_SEQUENCES_MAX_FRACTION)
        ));
    }
    Some(Finding::new(
        QcCode::ShortSequences,
        Severity::Warning,
        format!("{}.", parts.join("; ")),
        advice.join(" "),
    ))
}

fn approx_missing_placement(input: &QcInput) -> Option<Finding> {
    if !input.context.approximate_missing_placement {
        return None;
    }
    Some(Finding::new(
        QcCode::ApproxMissingPlacement,
        Severity::Info,
        "Multihetsep gives the number of called sites between variant rows but not their \
         positions, so the uncalled sites of each interval were placed together at its \
         start. Called and heterozygous sites keep their counts, but which bins are missing \
         (more than 90% of sites uncalled) depends on that placement, so the T/K/N bins can \
         differ from fq2psmcfa binning of the same calls."
            .to_string(),
        "For exact binning, read the VCF with its callable-sites mask instead.".to_string(),
    ))
}

fn depth(report: &QcReport) -> Option<Finding> {
    let d = report.depth.as_ref()?;
    let mut problems = Vec::new();
    let mut advice = Vec::new();
    if d.mean < MIN_MEAN_DEPTH {
        problems.push(format!(
            "Mean read depth is {}x, below the {}x that Nadachowska-Brzyska et al. (2016) \
             recommend for PSMC: at low depth heterozygous sites are often called \
             homozygous, which biases theta and Ne low.",
            sig(d.mean, 3),
            MIN_MEAN_DEPTH
        ));
        advice.push(
            "Sequence deeper if possible; otherwise read the curve as a lower bound on \
             heterozygosity and compare it with a deeper sample."
                .to_string(),
        );
    }
    let heavy: Vec<&super::FilterShare> = d
        .filters
        .iter()
        .filter(|f| f.fraction > MAX_FILTER_FRACTION)
        .collect();
    if !heavy.is_empty() {
        let list: Vec<String> = heavy
            .iter()
            .map(|f| format!("'{}' {}", f.name, pct(f.fraction)))
            .collect();
        problems.push(format!(
            "Single filters removed more than {} of the {} bases: {}.",
            pct(MAX_FILTER_FRACTION),
            count(d.total_bases),
            list.join(", ")
        ));
        let thresholds = match (d.min_depth, d.max_depth) {
            (Some(lo), Some(hi)) => format!(" (now {lo}-{hi} at a mean of {}x)", sig(d.mean, 3)),
            _ => String::new(),
        };
        advice.push(format!(
            "Check those filters' thresholds against the data: psmc's README suggests \
             depth limits of a third and twice the mean depth{thresholds}; a strict \
             mapping-quality cut-off removes much of a divergent reference."
        ));
    }
    if problems.is_empty() {
        return None;
    }
    Some(Finding::new(
        QcCode::Depth,
        Severity::Warning,
        problems.join(" "),
        advice.join(" "),
    ))
}

fn dropped_sequences(report: &QcReport) -> Option<Finding> {
    if report.dropped.is_empty() {
        return None;
    }
    let total: usize = report.dropped.iter().map(|d| d.count).sum();
    let reasons: Vec<String> = report
        .dropped
        .iter()
        .map(|d| match d.bases {
            Some(b) => format!("{} {} ({})", count(d.count as u64), d.reason, bases(b)),
            None => format!("{} {}", count(d.count as u64), d.reason),
        })
        .collect();
    Some(Finding::new(
        QcCode::DroppedSequences,
        Severity::Info,
        format!(
            "{} {} left out of the analysis: {}.",
            count(total as u64),
            if total == 1 {
                "sequence was"
            } else {
                "sequences were"
            },
            reasons.join("; ")
        ),
        "Check that these sequences were meant to be left out (for example, chromosome \
         names in the mask or VCF that do not match)."
            .to_string(),
    ))
}
