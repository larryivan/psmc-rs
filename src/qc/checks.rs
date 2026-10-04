//! Warnings and recommendations derived from the QC metrics.
//!
//! Each check has a stable [`QcCode`] whose rustdoc gives the rationale, and
//! named threshold constants.

use serde::Serialize;

use super::fmt::{bases, count, pct, sig};
use super::{
    QcInput, QcReport, SATURATION_TARGET_SHARE, SATURATION_WARN_SHARE, largest_trunk_within,
};

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

/// Run every check on `report` (computed from `input`); warnings first, then
/// in [`QcCode`] order.
pub(super) fn run(input: &QcInput, report: &QcReport, seq_bins: &[u64]) -> Vec<Finding> {
    let mut out: Vec<Finding> = [
        variants_only_no_mask(input, report),
        high_missing(report),
        bin_saturation(report),
        few_hets(report),
        too_short(report),
        short_sequences(report, seq_bins),
    ]
    .into_iter()
    .flatten()
    .collect();
    out.sort_by_key(|f| (f.severity != Severity::Warning, f.code));
    out
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
