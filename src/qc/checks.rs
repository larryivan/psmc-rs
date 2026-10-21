//! Warnings and recommendations derived from the QC metrics.
//!
//! Each check has a stable [`QcCode`] whose rustdoc gives the rationale, and
//! named threshold constants.

use serde::Serialize;

use super::fmt::{bases, count, counted, pct, plural, sig};
use super::names::SeqKind;
use super::stats::{fraction, median};
use super::{
    DEFAULT_TR_RATIO, QcInput, QcReport, SATURATION_TARGET_SHARE, SATURATION_WARN_SHARE,
    SequenceRow, largest_trunk_within,
};
use crate::io::CallableSummary;

/// [`QcCode::Depth`] warns below this mean read depth (Nadachowska-Brzyska
/// et al. 2016 recommend at least 18x for PSMC).
pub const MIN_MEAN_DEPTH: f64 = 18.0;

/// [`QcCode::Depth`] warns when one consensus filter removes more than this
/// share of the bases.
pub const MAX_FILTER_FRACTION: f64 = 0.30;

/// Share of `N` bins, or of uncalled sites when the loader counted them,
/// above which [`QcCode::HighMissing`] is a warning.
pub const HIGH_MISSING_WARN_FRACTION: f64 = 0.5;

/// Share of `N` bins, or of uncalled sites when the loader counted them,
/// above which [`QcCode::HighMissing`] is reported as info:
/// Nadachowska-Brzyska et al. (2016, Mol Ecol 25:1058) recommend at most 25%
/// missing data for PSMC.
pub const HIGH_MISSING_INFO_FRACTION: f64 = 0.25;

/// [`QcCode::FewHets`] fires below this many `K` bins (see there for why).
pub const MIN_HET_BINS: u64 = 1_000;

/// [`QcCode::TooShort`] fires below this many called bases (see there for
/// why).
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

/// [`QcCode::SexOrOrganelle`] judges a sequence only when the autosomal rate
/// predicts at least this many `K` bins in it, so that a shortfall is not
/// chance: seeing at most 2 where 10 are expected has a Poisson probability
/// of 0.3%.
pub const HEMIZYGOUS_MIN_EXPECTED_HET_BINS: f64 = 10.0;

/// [`QcCode::HetOutlier`] flags a sequence whose log heterozygosity is more
/// than this many median absolute deviations from the median. Iglewicz and
/// Hoaglin's (1993) modified z-score cut-off of 3.5 is 3.5 / 0.6745 = 5.2 MAD.
pub const HET_OUTLIER_MADS: f64 = 5.0;

/// [`QcCode::HetOutlier`] also requires a sequence to be at least this many
/// times above or below the median, so that small differences between very
/// uniform sequences (tiny MAD) are not flagged.
pub const HET_OUTLIER_MIN_FOLD: f64 = 1.5;

/// [`QcCode::HetOutlier`] needs at least this many sequences with
/// [`REFERENCE_MIN_CALLED_BASES`], or the median and MAD mean little.
pub const HET_OUTLIER_MIN_SEQUENCES: usize = 5;

/// [`QcCode::PartlyCallableBins`] says nothing about counts of callable
/// bases when fewer than this share of the called bins are partly callable.
pub const PARTLY_CALLABLE_NOTE_FRACTION: f64 = 0.01;

/// [`QcCode::PartlyCallableBins`] warns when the fit treats every called
/// bin as full (`--bin-callable ignore`) and the called bins hold less than
/// this share of their bases callable on average: θ, the times and Ne then
/// come out more than about 5% low. The bias exceeds the missing share of
/// the called bins: on the simulated genomes of `experiment/validation/`,
/// θ of such fits was 0.87 to 0.89 of the fit of the same genome without
/// missing data at a mean share of 0.90, and 0.63 to 0.66 at 0.705, 1.1 to
/// 1.3 times the missing share below it, so a threshold at 0.9 would leave
/// a bias of more than 10% with only a note.
pub const PARTLY_CALLABLE_WARN_MEAN: f64 = 0.95;

/// [`QcCode::ApproxMissingPlacement`] warns when more than this share of
/// the sites of multihetsep input are uncalled: the reader's placement of
/// them then biases θ, the times and Ne low by a few percent or more
/// (validation: θ 7-11% low with 30% of the sites uncalled in runs of about
/// 20 kb, 14-18% with 10% in runs of about 20 bp).
pub const APPROX_PLACEMENT_WARN_FRACTION: f64 = 0.05;

/// At most this many sequence names are listed in a message (all are in
/// [`Finding::sequences`]).
const MAX_LISTED: usize = 10;

/// The kind of a [`Finding`]. The serialised names (`HIGH_MISSING`, ...) are
/// stable and meant for scripts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum QcCode {
    /// The fit's starting model has negative transition probabilities
    /// (warning; [`super::QcContext::start`]): ρ0 is too large per bin for
    /// the time grid, and EM cannot move from such a start.
    ///
    /// C psmc's transition matrix (`psmc_update_hmm`, ported in
    /// [`crate::PsmcModel::param_recalculate`]) stays in state `k` with
    /// probability `1 - r_k (1 - q_kk)`, where `r_k`, the chance of a
    /// recombination counted in state `k`, grows with ρ. Above a limit it
    /// turns negative for some states: with flat λ, ρ0 of about 1/6 per bin
    /// at C's default t_max of 15 (0.167 for `4+5*3+4`, 0.161 for
    /// `4+25*2+4+6`), and about 2 / t_max for larger t_max. The expected
    /// log-likelihood the M-step maximises then takes the logarithm of a
    /// negative probability, so it is undefined everywhere the search
    /// starts: C psmc prints `QD -1e300`, keeps the starting parameters in
    /// every round and reports a flat curve. With C's starting θ0/ρ0 of 4
    /// the limit lies at θ0 of about 0.67, 49% of called bins heterozygous,
    /// far beyond [`QcCode::BinSaturation`]; with `--tr-ratio 1` at about
    /// 15%, where nothing else warns. θ0 and ρ0 per bin shrink in proportion
    /// with the bin size (θ0 = -s ln(1 - h) for bins of `s` sites at
    /// heterozygosity `h`), so smaller bins, a larger θ0/ρ0 or a smaller
    /// `--rho0` bring the start back into range.
    InvalidStart,
    /// A VCF was read in mask mode without a positive callable-sites mask
    /// (warning).
    ///
    /// The VCF reader then counts every position without a record as a
    /// called homozygous site. A variants-only VCF (the GATK and bcftools
    /// default) has no records for positions that could not be called (no or
    /// low coverage, low mapping quality, repeats), so all of them count as
    /// homozygous: missing data disappear, the `K` fraction is diluted, which
    /// pulls θ and Ne down, and uncalled stretches look like long runs of
    /// recent common ancestry. A callable mask would also exclude repeats and
    /// collapsed duplications, whose false heterozygous calls push θ up, so
    /// the net bias has no fixed direction. A gVCF read without a mask gets
    /// its own wording: positions no block covers are missing there, but
    /// reference blocks count as called whatever their depth and quality.
    VariantsOnlyNoMask,
    /// Called bins hold fewer callable bases than the bin size (info, or a
    /// warning when the fit ignores it and the called bins hold less than
    /// [`PARTLY_CALLABLE_WARN_MEAN`] of their bases callable on average).
    ///
    /// `fq2psmcfa` calls a bin as soon as a tenth of its bases are callable,
    /// and C psmc treats every called bin as the bin size `s` in bases. A
    /// bin of `c` callable bases is less likely to show a heterozygous site
    /// (`1 - exp(-θ (c / s) t)` instead of `1 - exp(-θ t)`), so with missing
    /// data scattered in runs shorter than a bin, θ per bin, N0 = θ / (4 μ
    /// s) and with them every time and every size come out low by about the
    /// missing share of the called bins: on simulated genomes with 30% of
    /// the bases missing in runs of about 20 bp, the RMSE of log10 Ne
    /// against the truth was 0.447 against 0.332 without missing data.
    /// Missing data in long runs mostly make whole bins `N`, which C's model
    /// handles, and leave partly callable bins only at their ends.
    ///
    /// A VCF read with its callable mask records the callable bases of each
    /// bin exactly ([`super::QcInput::callable`]), and psmc-rs fits each
    /// called bin by them by default (`--bin-callable auto`, the
    /// callable-length model of `docs/design/bin-callable-model.md`), which
    /// corrects for that bias: the finding then reports how much was
    /// corrected (info). With `--bin-callable ignore`
    /// ([`super::QcContext::counts_ignored`]) the bias stays, and the
    /// finding is a warning below [`PARTLY_CALLABLE_WARN_MEAN`]. Nothing is
    /// said when fewer than [`PARTLY_CALLABLE_NOTE_FRACTION`] of the called
    /// bins are partly callable. A psmcfa file records only which bins are
    /// called, not how many of their bases are; when it has missing bins,
    /// the finding (info) counts the called bins next to them, which are
    /// partly callable unless a missing run starts or ends exactly at a bin
    /// boundary, and says how to measure the rest from the data the psmcfa
    /// was made from. Multihetsep's counts follow the reader's placement of
    /// the uncalled sites, which leaves nearly every called bin full whatever
    /// the data, so they do not measure the bias and the finding is not
    /// given for multihetsep input: [`QcCode::ApproxMissingPlacement`] says
    /// what the placement does instead.
    PartlyCallableBins,
    /// Much of the input is missing: more than [`HIGH_MISSING_WARN_FRACTION`]
    /// (warning) or [`HIGH_MISSING_INFO_FRACTION`] (info) of the bins are
    /// `N`, or of the sites are uncalled
    /// ([`super::BinStats::uncalled_site_fraction`], when the loader counted
    /// called sites).
    ///
    /// `N` bins carry no information, so the effective input shrinks. Worse,
    /// the `fq2psmcfa` rule calls a bin as soon as 10% of its sites are
    /// called, so missing sites scattered along the genome (per-site depth
    /// or quality filters) leave partly called `T`/`K` bins rather than `N`
    /// bins: the `N` share then understates the missing data, and a partly
    /// called bin is less likely to show a heterozygous site, which biases θ
    /// and Ne low. Missing data also concentrate in repeats and divergent
    /// regions, so the remaining sites are not a random sample of the
    /// genome. Without site counts only the `N` share is known; as every `N`
    /// bin has more than 90% of its sites uncalled, 0.9 times it is a lower
    /// bound on the uncalled share, so the per-site thresholds apply to it
    /// almost unchanged.
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
    /// recombinations (more when bins hold several sites), spread unevenly
    /// over the parameters under a constant population size. With the
    /// pattern psmc's README uses for human data, `4+25*2+4+6` (28 free λ),
    /// the most recent group gets 3.6% and the oldest 0.005% of the
    /// segments, and only 13 of the 28 groups reach 10. With C psmc's
    /// default pattern `4+5*3+4` (7 free λ), which psmc-rs also fits when
    /// no pattern is given, 5 of the 7 groups reach 10; with one λ per
    /// interval (`--n-steps 64` without a pattern: 65 free λ), none does.
    /// Fewer `K` bins leave most parameters undetermined. Given
    /// [`super::QcContext::free_lambdas`], the message
    /// divides C's expected recombinations by the fit's own parameter count;
    /// the oldest group usually gets far fewer than that average (less than
    /// a tenth of it in each of the three patterns above).
    FewHets,
    /// Less than [`MIN_CALLED_BASES`] of called sequence (warning).
    ///
    /// PSMC was developed and tested on whole genomes (Li & Durbin 2011:
    /// gigabases per human genome). By the count in [`QcCode::FewHets`], 100
    /// Mb at human heterozygosity (1e-3 per bp, 1e5 heterozygous sites, about
    /// 25,000 recombinations) give every group of `4+25*2+4+6` but the
    /// oldest, every group of `4+5*3+4` and 55 of 65 single intervals at
    /// least 10 recombinations under a constant size; at 30 Mb the
    /// second-oldest group of `4+25*2+4+6` gets about 5. Species with lower
    /// heterozygosity need more sequence.
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
    /// Multihetsep input, where uncalled sites have no positions (info; a
    /// warning when more than [`APPROX_PLACEMENT_WARN_FRACTION`] of the
    /// sites are uncalled).
    ///
    /// Multihetsep gives, for each segregating site, how many sites since
    /// the previous one were called, but not where the uncalled ones are.
    /// The reader places them together at the start of each interval
    /// ([`crate::io::mhs::read_mhs_with`]). Called and heterozygous sites
    /// keep their numbers and the heterozygous sites their positions, but a
    /// bin is `N` only when more than 90% of its sites are uncalled, so
    /// moving the uncalled sites changes which bins are `N`, `T` or `K`
    /// compared with `fq2psmcfa` on the same calls: missing sites scattered
    /// among called ones become runs of `N` bins after heterozygous sites,
    /// the bin of a heterozygous site followed by uncalled sites is partly
    /// callable or lost, and θ, the times and Ne come out low whatever the
    /// emission model. On the simulated genomes of
    /// `experiment/validation/` (three seeds of the bottleneck model and
    /// three of the expansion model), θ of the multihetsep fit was 0.82 to
    /// 0.86 of the fit of the same genome without missing data with 10% of
    /// the sites uncalled in runs of about 20 bp, 0.67 to 0.70 with 30%, and
    /// 0.89 to 0.93 with 30% in runs of about 20 kb, against 0.94 to 1.00
    /// for the VCF read with its callable mask; the share of uncalled sites
    /// is the measure the finding has, as multihetsep cannot tell scattered
    /// missing data from long runs. The callable bases of its bins follow
    /// the same placement, so `--bin-callable auto` does not fit them;
    /// `--bin-callable approximate` does, which lowered θ further (0.52 to
    /// 0.60 with 30% in short runs). VCF input with a callable mask bins
    /// exactly.
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
        Self::InvalidStart,
        Self::VariantsOnlyNoMask,
        Self::PartlyCallableBins,
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
            Self::InvalidStart => "INVALID_START",
            Self::VariantsOnlyNoMask => "VARIANTS_ONLY_NO_MASK",
            Self::PartlyCallableBins => "PARTLY_CALLABLE_BINS",
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
    fn new(
        code: QcCode,
        severity: Severity,
        message: impl Into<String>,
        recommendation: impl Into<String>,
    ) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            recommendation: recommendation.into(),
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
    seqs: &[SequenceRow],
) -> (Vec<Finding>, Vec<usize>) {
    let mut named = Vec::new();
    let mut out: Vec<Finding> = [
        invalid_start(input, report),
        variants_only_no_mask(input, report),
        partly_callable_bins(input, report),
        high_missing(input, report),
        bin_saturation(report),
        few_hets(input, report),
        too_short(report),
        short_sequences(report, seqs),
        sex_or_organelle(report, seqs, &mut named),
        het_outliers(report, seqs, &mut named),
        approx_missing_placement(input, report),
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
    let mut s = names[..names.len().min(MAX_LISTED)].join(", ");
    if names.len() > MAX_LISTED {
        s.push_str(&format!(" and {} more", names.len() - MAX_LISTED));
    }
    s
}

fn sex_or_organelle(
    report: &QcReport,
    seqs: &[SequenceRow],
    named: &mut Vec<usize>,
) -> Option<Finding> {
    let flagged: Vec<&SequenceRow> = seqs.iter().filter(|s| s.kind.is_some()).collect();
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
        // Per callable base when the input records callable bases, so that a
        // sequence with more missing data does not look less heterozygous.
        let judged = match (reference, s.stats.theta0_comparable()) {
            (Some(r), Some(theta)) if r.theta0_per_bin > 0.0 => {
                let ratio = theta / r.theta0_per_bin;
                let share = s.stats.mean_callable_fraction.unwrap_or(1.0);
                let expected_k = called_bins as f64 * -(-r.theta0_per_bin * share).exp_m1();
                Some((ratio, expected_k))
            }
            _ => None,
        };
        let note = match judged {
            Some((ratio, expected)) if expected >= HEMIZYGOUS_MIN_EXPECTED_HET_BINS => {
                let hemi = ratio < HEMIZYGOUS_MAX_RATIO;
                if hemi {
                    hemizygous.push(s.name.clone());
                }
                let hemi = if hemi { ", hemizygous?" } else { "" };
                format!("{}x the autosomal heterozygosity{hemi}", sig(ratio, 2))
            }
            Some(_) => "too few called bins to judge".to_string(),
            None if called_bins == 0 => "no called bins".to_string(),
            None => "nothing to compare with".to_string(),
        };
        described.push(format!("{} ({kind}, {note})", s.name));
    }
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
        let listed = listing(&hemizygous);
        (
            Severity::Warning,
            format!(
                "{listed} {} far less heterozygous than the autosomes, as a single (hemizygous) \
                 copy is; PSMC reads such a sequence as a long stretch of recent common \
                 ancestry, which lowers theta and adds a false signal of small recent Ne.",
                plural(hemizygous.len(), "is", "are"),
            ),
            format!(
                "Exclude {listed} (keep autosomes only, e.g. by restricting the mask, VCF or \
                 psmcfa to them)."
            ),
        )
    };
    let mut f = Finding::new(
        QcCode::SexOrOrganelle,
        severity,
        format!(
            "{} {} named like a sex chromosome or organelle: {}. {consequence}",
            count(flagged.len() as u64),
            plural(flagged.len(), "sequence is", "sequences are"),
            listing(&described),
        ),
        advice,
    );
    f.sequences = flagged.iter().map(|s| s.name.clone()).collect();
    Some(f)
}

fn het_outliers(
    report: &QcReport,
    seqs: &[SequenceRow],
    named: &mut Vec<usize>,
) -> Option<Finding> {
    let eligible: Vec<(&SequenceRow, f64)> = seqs
        .iter()
        .filter(|s| s.kind.is_none())
        .filter(|s| s.stats.called_bases_from_bins >= REFERENCE_MIN_CALLED_BASES)
        .filter_map(|s| s.stats.theta0_comparable().map(|t| (s, t.ln())))
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
    let mut outliers: Vec<(&SequenceRow, f64)> = eligible
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
         artefacts or hemizygous.",
    );
    f.sequences = outliers.iter().map(|(s, _)| s.name.clone()).collect();
    Some(f)
}

fn invalid_start(input: &QcInput, report: &QcReport) -> Option<Finding> {
    let start = input.context.start.as_ref()?;
    // Without a heterozygous bin theta0 is -0 (or NaN), which FEW_HETS and
    // the loader report: rho0 is then not too large, but missing.
    if start.is_valid() || start.theta0.is_nan() || start.theta0 <= 0.0 {
        return None;
    }
    let (theta, rho, t_max) = (sig(start.theta0, 3), sig(start.rho0, 3), start.t_max);
    let pattern = start
        .pattern
        .as_deref()
        .map_or(String::new(), |p| format!(" and pattern {p}"));
    let whose = if start.saved {
        format!(
            "The fit starts from the saved parameters, theta0 {theta} and rho0 {rho} per bin, \
             with t_max {t_max}{pattern}"
        )
    } else if start.assumed {
        format!(
            "A fit with psmc-rs's default settings would start from theta0 {theta} per bin \
             (from these data) and rho0 = theta0/{} = {rho} per bin, with t_max \
             {t_max}{pattern}",
            DEFAULT_TR_RATIO
        )
    } else {
        format!(
            "The fit starts from theta0 {theta} and rho0 {rho} per bin, with t_max \
             {t_max}{pattern}"
        )
    };
    let lowest = if start.min_transition.is_finite() {
        format!(" (down to {})", sig(start.min_transition, 3))
    } else {
        String::new()
    };
    let limit = start.max_rho0.map_or(String::new(), |m| {
        format!(", which allows up to about {}", sig(m, 3))
    });
    let message = format!(
        "{whose}. Some of PSMC's transition probabilities are then negative{lowest}: rho0 is \
         too large per bin for the model's time grid{limit}. That is not a valid model, and EM \
         cannot move from it: the expected log-likelihood that the M-step maximises is \
         undefined there. C psmc keeps the starting parameters and reports a flat curve; \
         psmc-rs refuses to fit from such a start."
    );
    // theta0 and rho0 per bin shrink in proportion with the bin size; a
    // tenth below the limit leaves a margin.
    let recommendation = match start.max_rho0 {
        // A saved fit's rho comes from its file: the options of a fresh
        // start (--tr-ratio, --rho0) do not apply to it.
        _ if start.saved => format!(
            "Start from a smaller rho0: lower rho in the --init-params file{}, or fit from a \
             fresh start instead of --init-params.",
            start.max_rho0.map_or(String::new(), |max| format!(
                " to {} or less",
                sig(0.9 * max, 2)
            ))
        ),
        Some(max) => {
            let target = 0.9 * max;
            let s = report.bin_size;
            let bins = ((s as f64 * target / start.rho0).floor() as usize).max(1);
            let ratio = (10.0 * start.theta0 / target).ceil() / 10.0;
            format!(
                "Start from a smaller rho0. Smaller bins shrink theta0 and rho0 per bin in \
                 proportion: re-bin the per-site data at {bins} bp per bin or less \
                 (`--bin-size {bins}` for VCF and multihetsep input, or `fq2psmcfa -s {bins}` \
                 for a psmcfa file, which psmc-rs then reads with `--bin-size {bins}`). \
                 Otherwise fit with `--tr-ratio {ratio}` or more, or `--rho0 {}` or less.",
                sig(target, 2)
            )
        }
        None => "Start from a smaller rho0: re-bin the per-site data with smaller bins, which \
                 shrink theta0 and rho0 per bin in proportion, or fit with a larger \
                 --tr-ratio or a smaller --rho0."
            .to_string(),
    };
    Some(Finding::new(
        QcCode::InvalidStart,
        Severity::Warning,
        message,
        recommendation,
    ))
}

fn variants_only_no_mask(input: &QcInput, report: &QcReport) -> Option<Finding> {
    if !input.context.variants_only_no_mask {
        return None;
    }
    let missing = pct(report.totals.missing_fraction);
    let theta = report
        .totals
        .theta0_per_bin
        .map_or("n/a".to_string(), |t| format!("{} per bin", sig(t, 3)));
    let (message, recommendation) = if input.context.gvcf {
        (
            format!(
                "The gVCF was read without a callable-sites mask. Positions that no record or \
                 block covers are missing ({missing} of bins are missing), but every reference \
                 block counts as called homozygous over its span, whatever its depth and \
                 genotype quality: GATK writes stretches without reads as 0/0 blocks, which \
                 pulls theta ({theta}) and Ne down, and blocks over repeats and collapsed \
                 duplications keep their false heterozygous calls nearby, which pushes them up."
            ),
            "Make a callable-sites BED from the gVCF's blocks with depth and genotype-quality \
             bounds, pass it with --vcf-mask, and give the variant records alone (`bcftools \
             view -v snps,indels`); see docs/preprocessing.md, route B.",
        )
    } else {
        (
            format!(
                "The VCF was read without a callable-sites mask, so every position without a \
                 record counted as a called homozygous site ({missing} of bins are missing). In \
                 a variants-only VCF that hides every uncalled position, which pulls theta \
                 ({theta}) and Ne down and makes uncalled stretches look like recent common \
                 ancestry; meanwhile false heterozygous calls in repeats and collapsed \
                 duplications, which a callable mask would exclude, push them up. The net bias \
                 has no fixed direction."
            ),
            "Pass a BED of callable sites (--vcf-mask), e.g. from your variant-calling \
             pipeline or a depth profile, or call all sites (`bcftools mpileup -f ref.fa \
             aln.bam | bcftools call -c`) so that every called position has a record.",
        )
    };
    Some(Finding::new(
        QcCode::VariantsOnlyNoMask,
        Severity::Warning,
        message,
        recommendation,
    ))
}

/// Whether the fit weighs each called bin by its callable bases: the input
/// records them and the fit does not ignore them.
fn counts_used(input: &QcInput) -> bool {
    input.callable.is_some() && !input.context.counts_ignored
}

/// The distribution of the called bins' shares of callable bases, as the
/// share of called bins in each of the classes `[0.1, 0.5)`, `[0.5, 0.9)`,
/// `[0.9, 1)` and full.
pub(crate) fn callable_distribution(c: &CallableSummary) -> String {
    let h = &c.histogram;
    let share = |n: u64| fraction(n, c.called_bins);
    format!(
        "{} hold 10-50% of their bases callable, {} 50-90%, {} 90-99% and {} all",
        pct(share(h[..4].iter().sum())),
        pct(share(h[4..8].iter().sum())),
        pct(share(h[8])),
        pct(share(h[9]))
    )
}

fn partly_callable_bins(input: &QcInput, report: &QcReport) -> Option<Finding> {
    // Multihetsep's counts follow the reader's placement of the uncalled
    // sites, which leaves its called bins nearly full whatever the data, so
    // their share does not measure the bias; APPROX_MISSING_PLACEMENT says
    // what the placement does.
    if input.context.approximate_missing_placement {
        return None;
    }
    let Some(c) = &report.callable else {
        return partly_callable_psmcfa(report);
    };
    let s = report.bin_size;
    if c.partial_fraction() < PARTLY_CALLABLE_NOTE_FRACTION {
        return None;
    }
    let measured = format!(
        "{} of called bins hold fewer than {s} callable bases, {} of their bases on average \
         ({}).",
        pct(c.partial_fraction()),
        pct(c.mean_fraction),
        callable_distribution(c)
    );
    let bias = format!(
        "Treated as {s} bases, as fq2psmcfa's psmcfa and C psmc treat every called bin, they \
         would make theta, the times and Ne about {} low or more (in simulations 1.1 to 1.3 \
         times the missing share)",
        pct(1.0 - c.mean_fraction)
    );
    if counts_used(input) {
        return Some(Finding::new(
            QcCode::PartlyCallableBins,
            Severity::Info,
            format!(
                "{measured} {bias}; the fit weighs each called bin by its callable bases \
                 (--bin-callable auto), which removes most of that bias."
            ),
            "Nothing to do: keep --bin-callable auto (the default). A psmcfa written from this \
             input (psmc-rs convert --psmcfa, fq2psmcfa) cannot record the counts, so fit the \
             input itself rather than such a psmcfa.",
        ));
    }
    let severity = if c.mean_fraction < PARTLY_CALLABLE_WARN_MEAN {
        Severity::Warning
    } else {
        Severity::Info
    };
    Some(Finding::new(
        QcCode::PartlyCallableBins,
        severity,
        format!(
            "{measured} The fit treats every called bin as {s} bases (--bin-callable ignore): \
             theta, the times and Ne come out about {} low or more (in simulations 1.1 to 1.3 \
             times the missing share of the called bins).",
            pct(1.0 - c.mean_fraction)
        ),
        "Fit with --bin-callable auto (the default), which weighs each called bin by its \
         callable bases.",
    ))
}

/// [`QcCode::PartlyCallableBins`] for input that records which bins are
/// called but not how many of their bases (psmcfa).
fn partly_callable_psmcfa(report: &QcReport) -> Option<Finding> {
    let (s, t) = (report.bin_size, &report.totals);
    let next = t.called_next_to_missing;
    let called = t.t_bins + t.k_bins;
    if t.missing_bins == 0
        || (next as f64) < PARTLY_CALLABLE_NOTE_FRACTION * called as f64
        || next == 0
    {
        return None;
    }
    Some(Finding::new(
        QcCode::PartlyCallableBins,
        Severity::Info,
        format!(
            "This input records which bins are called (at least 10% of their {s} bases \
             callable, fq2psmcfa's rule) but not how many of their bases are, as psmcfa \
             does, so the fit treats every called bin as {s} bases, as C psmc does. {} of \
             bins are missing (N), and {} called bins ({} of them) border those runs: they \
             hold fewer than {s} callable bases unless a run starts or ends at a bin \
             boundary. If missing data are also scattered in runs shorter than a bin, most \
             called bins do, which this input cannot show. Partly callable bins show fewer \
             heterozygous sites, which biases theta, the times and Ne low by about their \
             missing share.",
            pct(t.missing_fraction),
            count(next),
            pct(next as f64 / called as f64)
        ),
        "To measure it, run `psmc-rs qc` on the data the psmcfa was made from: a VCF with \
         its callable mask (--vcf-mask) or an all-sites VCF from `bcftools call -c`. It \
         reports the share of callable bases of the called bins, and a fit of that VCF \
         weighs each bin by its callable bases (--bin-callable auto), which removes the \
         bias.",
    ))
}

fn high_missing(input: &QcInput, report: &QcReport) -> Option<Finding> {
    let t = &report.totals;
    let bins = t.missing_fraction;
    let sites = t.uncalled_site_fraction.zip(t.called_bases);
    let missing = sites.map_or(bins, |(u, _)| u.max(bins));
    let severity = if missing > HIGH_MISSING_WARN_FRACTION {
        Severity::Warning
    } else if missing > HIGH_MISSING_INFO_FRACTION {
        Severity::Info
    } else {
        return None;
    };
    // The fit corrects the bias of partly called bins when it weighs each
    // bin by its exact callable bases; multihetsep's placement of the
    // uncalled sites biases the fit whatever it does.
    let corrected = if input.context.approximate_missing_placement {
        " Multihetsep does not say where the uncalled sites are, and where its reader places \
         them biases theta and Ne low whatever --bin-callable (see APPROX_MISSING_PLACEMENT)."
    } else if counts_used(input) {
        " The fit weighs each called bin by its callable bases (--bin-callable auto), which \
         corrects that bias (see PARTLY_CALLABLE_BINS); the other effects of missing data \
         remain."
    } else {
        ""
    };
    let message = match sites {
        Some((uncalled, called)) => {
            let het = match (t.het_per_kb, t.site_het_per_kb) {
                (Some(from_bins), Some(from_sites)) => format!(
                    "; here the called bins imply {} heterozygous sites per kb, against {} \
                     per kb of called sites",
                    sig(from_bins, 3),
                    sig(from_sites, 3)
                ),
                _ => String::new(),
            };
            format!(
                "{} of sites are uncalled ({} called) and {} of bins are missing (N: more \
                 than 90% of their sites uncalled). A bin is called when just 10% of its \
                 sites are, so missing sites scattered along the genome leave partly called \
                 bins, which show fewer heterozygous sites and bias theta and Ne low if they \
                 are treated as full bins{het}.{corrected}",
                pct(uncalled),
                bases(called),
                pct(bins),
            )
        }
        None => format!(
            "{} of bins are missing (N: more than 90% of their sites uncalled), so only {} \
             of sequence is used. A bin is called when just 10% of its sites are, so heavy \
             missingness also leaves partly called bins, which show fewer heterozygous sites \
             and bias theta and Ne low.",
            pct(bins),
            bases(t.called_bases_from_bins),
        ),
    };
    Some(Finding::new(
        QcCode::HighMissing,
        severity,
        message,
        "Check what removes the data: depth and mapping-quality filters, a callable mask \
         that does not match the chromosome names, or low coverage; per-site filters \
         scatter the uncalled sites over called bins. Nadachowska-Brzyska et al. (2016) \
         recommend at most 25% missing data for PSMC.",
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
             keeps the share below {} at this heterozygosity: `--bin-size {r}` for VCF and \
             multihetsep input, or `fq2psmcfa -s {r}` for a psmcfa file, which psmc-rs \
             then reads with `--bin-size {r}` so that the results are scaled with the \
             same bin size.",
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

fn few_hets(input: &QcInput, report: &QcReport) -> Option<Finding> {
    let k = report.totals.k_bins;
    if k >= MIN_HET_BINS {
        return None;
    }
    let expected = report.expected_recombinations;
    let recomb = expected.map_or(String::new(), |n| {
        let n = n.round() as usize;
        format!(
            "; C psmc's starting model expects about {} in these data",
            counted(n, "recombination", "recombinations")
        )
    });
    let parameters = match (input.context.free_lambdas.filter(|&n| n > 0), expected) {
        (Some(n), Some(r)) => format!(
            "this fit has {} of them, about {} recombinations each on average (the oldest \
             intervals usually get far fewer)",
            count(n as u64),
            sig(r / n as f64, 2)
        ),
        (Some(n), None) => format!("this fit has {} of them", count(n as u64)),
        (None, _) => {
            "the pattern its README uses for human data, 4+25*2+4+6, has 28 of them".to_string()
        }
    };
    Some(Finding::new(
        QcCode::FewHets,
        Severity::Warning,
        format!(
            "Only {} heterozygous (K) {}, fewer than {}{recomb}. psmc's documentation \
             asks for about 10 recombinations in the intervals of each free parameter, and \
             {parameters}.",
            count(k),
            plural(k as usize, "bin", "bins"),
            count(MIN_HET_BINS),
        ),
        "Add data (more of the genome, a less strict mask) or fit a --pattern with fewer, \
         wider parameter groups; read the curve's oldest and most recent parts with \
         caution.",
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
         caution.",
    ))
}

fn short_sequences(report: &QcReport, seqs: &[SequenceRow]) -> Option<Finding> {
    let l = &report.lengths;
    let short_1mb = l.below_1mb.fraction > SHORT_SEQUENCES_MAX_FRACTION;
    let short_trunk = l
        .below_trunk
        .as_ref()
        .filter(|s| s.fraction > SHORT_SEQUENCES_MAX_FRACTION);
    if !short_1mb && short_trunk.is_none() {
        return None;
    }
    let mut parts = Vec::new();
    let mut advice = Vec::new();
    if short_1mb {
        parts.push(format!(
            "{} of bins lie in {} shorter than 1 Mb, where sequence ends (each sequence is \
             a separate HMM chain) cut the long segments of recent common ancestry",
            pct(l.below_1mb.fraction),
            counted(l.below_1mb.sequences, "sequence", "sequences")
        ));
        advice.push(
            "Prefer a more contiguous assembly, or drop sequences shorter than 1 Mb if \
             enough data remain."
                .to_string(),
        );
    }
    if let Some(below) = short_trunk {
        parts.push(format!(
            "{} of bins lie in {} shorter than the bootstrap trunk ({} bins), which the \
             bootstrap can only resample whole",
            pct(below.fraction),
            counted(below.sequences, "sequence", "sequences"),
            count(report.bootstrap_trunk_bins as u64)
        ));
        let bins: Vec<u64> = seqs.iter().map(|s| s.stats.bins).collect();
        let trunk = largest_trunk_within(&bins, SHORT_SEQUENCES_MAX_FRACTION);
        advice.push(format!(
            "For the bootstrap, a trunk of at most {} bins keeps at least {} of bins in \
             sequences of one trunk or more (--bootstrap-block-size).",
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

fn approx_missing_placement(input: &QcInput, report: &QcReport) -> Option<Finding> {
    if !input.context.approximate_missing_placement {
        return None;
    }
    let counts = if counts_used(input) {
        " The fit weighs each called bin by the callable bases that placement gives it \
         (--bin-callable approximate): their sum over each interval between rows is exact, \
         but how it is split among the bins, and the count of the bin holding the next \
         heterozygous site, are not, and in those simulations they lowered theta further \
         (0.52-0.60 of the complete-data fit with 30% of the sites uncalled in short runs)."
    } else if input.callable.is_some() {
        " The callable bases of each bin follow the same placement, so the fit treats every \
         called bin as full (--bin-callable auto uses exact counts only)."
    } else {
        ""
    };
    let t = &report.totals;
    // The loader counts the called sites of multihetsep exactly; the N
    // share is a lower bound without them.
    let uncalled = t.uncalled_site_fraction.unwrap_or(t.missing_fraction);
    let placement = "Multihetsep gives the number of called sites between variant rows but not \
                     their positions, so the uncalled sites of each interval were placed \
                     together at its start.";
    if uncalled <= APPROX_PLACEMENT_WARN_FRACTION {
        return Some(Finding::new(
            QcCode::ApproxMissingPlacement,
            Severity::Info,
            format!(
                "{placement} {} of the sites are uncalled. Called and heterozygous sites keep \
                 their counts, but which bins are missing (more than 90% of sites uncalled) \
                 depends on that placement, so the T/K/N bins can differ from fq2psmcfa \
                 binning of the same calls.{counts}",
                pct(uncalled)
            ),
            "For exact bins and callable bases, read the VCF with its callable-sites mask \
             instead.",
        ));
    }
    Some(Finding::new(
        QcCode::ApproxMissingPlacement,
        Severity::Warning,
        format!(
            "{placement} {} of the sites are uncalled. That changes which bins are missing \
             (more than 90% of sites uncalled): uncalled sites scattered among called ones \
             become runs of N bins after heterozygous sites, and the bin of a heterozygous \
             site followed by uncalled sites is partly callable or lost. It biases theta, the \
             times and Ne low whatever --bin-callable: on simulated genomes theta came out \
             0.82-0.86 of the fit of the same genome without missing data with 10% of the \
             sites uncalled in runs of about 20 bp, 0.67-0.70 with 30%, and 0.89-0.93 with \
             30% in runs of about 20 kb, against 0.94-1.00 for the VCF read with its callable \
             mask.{counts}",
            pct(uncalled)
        ),
        "Fit the VCF the multihetsep was made from, with its callable-sites mask \
         (--vcf-mask): it bins exactly and weighs each called bin by its callable bases.",
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
    let heavy: Vec<String> = d
        .filters
        .iter()
        .filter(|f| f.fraction > MAX_FILTER_FRACTION)
        .map(|f| format!("'{}' {}", f.name, pct(f.fraction)))
        .collect();
    if !heavy.is_empty() {
        problems.push(format!(
            "Single filters removed more than {} of the {} bases: {}.",
            pct(MAX_FILTER_FRACTION),
            count(d.total_bases),
            heavy.join(", ")
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
            "{} left out of the analysis: {}.",
            counted(total, "sequence was", "sequences were"),
            reasons.join("; ")
        ),
        format!(
            "Check that {} meant to be left out (for example, chromosome names in the mask \
             or VCF that do not match).",
            plural(total, "this sequence was", "these sequences were")
        ),
    ))
}
