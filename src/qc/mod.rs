//! Input quality control: metrics, warnings and recommendations for a binned
//! PSMC input.
//!
//! [`analyze`] turns the bins of a loaded dataset ([`QcInput`]) into a
//! [`QcReport`]: T/K/N counts, θ0 (C psmc's starting value), called length,
//! the sequence length distribution and missing-data runs, for the whole input
//! and for each sequence, plus [`Finding`]s: warnings and notes with a stable
//! [`QcCode`], the measured numbers and a recommendation. The rationale of
//! each check is documented on its code. The report serialises to JSON with
//! serde ([`QcReport::to_json`]) and renders as aligned text
//! ([`render_text`], or [`render_text_with`] and [`TextOptions::brief`] for
//! the short block printed before a fit).
//!
//! Bins follow the psmcfa convention: 0 = `T` (called, no heterozygous site),
//! 1 = `K` (at least one heterozygous site), anything else = `N` (missing).
//!
//! ```
//! use psmc_rs::qc::{QcCode, QcContext, QcInput, QcSeq, TextOptions, analyze, render_text_with};
//!
//! // chr1 is split into two rows of one HMM chain; chrM is one row.
//! let rows = vec![vec![0, 1, 0, 2], vec![2, 0, 1], vec![0; 166]];
//! let seqs = [QcSeq::new("chr1", 7), QcSeq::new("chrM", 166)];
//! let report = analyze(&QcInput {
//!     rows: &rows,
//!     row_starts: &[true, false, true],
//!     seqs: &seqs,
//!     bin_size: 100,
//!     bootstrap_trunk_bins: 0,
//!     context: QcContext {
//!         input_format: Some("psmcfa".to_string()),
//!         ..QcContext::default()
//!     },
//! });
//! assert_eq!(report.totals.k_bins, 2);
//! assert_eq!(report.sequences[0].stats.longest_missing_run_bins, 2);
//! assert!(report.finding(QcCode::SexOrOrganelle).is_some());
//! println!("{}", render_text_with(&report, &TextOptions::brief()));
//! let json = report.to_json();
//! # assert!(json.contains("\"SEX_OR_ORGANELLE\""));
//! ```

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::ops::Range;

use crate::model::PsmcModel;

mod checks;
pub(crate) mod fmt;
mod names;
mod stats;
mod text;

pub use checks::{
    Finding, HEMIZYGOUS_MAX_RATIO, HEMIZYGOUS_MIN_EXPECTED_HET_BINS, HET_OUTLIER_MADS,
    HET_OUTLIER_MIN_FOLD, HET_OUTLIER_MIN_SEQUENCES, HIGH_MISSING_INFO_FRACTION,
    HIGH_MISSING_WARN_FRACTION, MAX_FILTER_FRACTION, MIN_CALLED_BASES, MIN_HET_BINS,
    MIN_MEAN_DEPTH, QcCode, REFERENCE_MIN_CALLED_BASES, SHORT_SEQUENCES_MAX_FRACTION, Severity,
};
pub use names::{SeqKind, classify_name};
pub use text::{TextOptions, render_text, render_text_with};

pub use stats::{
    bootstrap_trunks_of, het_per_site, initial_recombinations, largest_trunk_within,
    max_bin_size_for_share, multi_het_share, n50, theta0_per_bin,
};

/// Share of `K` bins holding two or more heterozygous sites
/// ([`Saturation::multi_het_share`]) above which bins are saturated.
///
/// It is twice [`SATURATION_TARGET_SHARE`], the level of the human data PSMC
/// was designed for.
pub const SATURATION_WARN_SHARE: f64 = 0.10;

/// Share of `K` bins holding two or more heterozygous sites that a
/// recommended bin size must stay below
/// ([`Saturation::max_bin_size_below_target`]).
///
/// Li & Durbin (2011) chose 100 bp bins for human genomes, whose
/// heterozygosity of about 1e-3 per bp puts 4.9% of `K` bins at two or more
/// heterozygous sites ([`multi_het_share`]`(0.001, 100)`); 5% keeps a
/// recommendation at that level.
pub const SATURATION_TARGET_SHARE: f64 = 0.05;

/// Sequences shorter than this many bases are short: their ends cut the long
/// segments of recent common ancestry ([`QcCode::ShortSequences`],
/// [`LengthStats::below_1mb`]).
pub const SHORT_SEQUENCE_BASES: u64 = 1_000_000;

/// C psmc's default initial θ0/ρ0 ratio (`-r 4`), used for
/// [`QcReport::expected_recombinations`].
pub const DEFAULT_TR_RATIO: f64 = 4.0;

/// At most this many per-sequence rows are kept in [`QcReport::sequences`];
/// the rest are summed in [`QcReport::omitted`].
pub const MAX_SEQUENCE_ROWS: usize = 100;

/// One sequence of the input, as the loader saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QcSeq {
    /// Sequence name (psmcfa record name, VCF chromosome, ...).
    pub name: String,
    /// Number of bins of this sequence in [`QcInput::rows`].
    pub n_bins: usize,
    /// Bases the sequence spans, when the loader knows them; otherwise
    /// `n_bins * bin_size` is used.
    pub extent_bases: Option<u64>,
    /// Called (non-missing) bases, when the loader saw individual sites.
    pub called_bases: Option<u64>,
    /// Heterozygous sites that reached the bins, when the loader saw
    /// individual sites.
    pub het_sites: Option<u64>,
}

impl QcSeq {
    /// A sequence known only by its name and bin count.
    pub fn new(name: impl Into<String>, n_bins: usize) -> Self {
        Self {
            name: name.into(),
            n_bins,
            extent_bases: None,
            called_bases: None,
            het_sites: None,
        }
    }
}

/// Read depth seen by a consensus-mode (all-sites) conversion.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DepthStats {
    /// Mean depth over the positions considered.
    pub mean: f64,
    /// Median depth.
    pub median: f64,
    /// Smallest depth a site needed to be called (`vcf2fq -d`), if any.
    pub min_depth: Option<u32>,
    /// Largest depth a site could have to be called (`vcf2fq -D`), if any.
    pub max_depth: Option<u32>,
    /// `histogram[d]` positions with depth `d`; the last entry counts every
    /// depth from its index up.
    pub histogram: Vec<u64>,
}

/// Bases removed by one consensus filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FilterCount {
    /// Filter name, e.g. `"min depth"` or `"mapping quality"`.
    pub name: String,
    /// Bases it removed (a base failing several filters counts for each).
    pub bases: u64,
}

/// What a consensus-mode conversion saw.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ConsensusStats {
    /// Read depth.
    pub depth: DepthStats,
    /// Positions the filters were applied to: the denominator of their
    /// shares.
    pub total_bases: u64,
    /// Bases each filter removed.
    pub filters: Vec<FilterCount>,
}

/// Sequences the loader left out, for one reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DroppedSequences {
    /// Why, e.g. `"without any called site"`.
    pub reason: String,
    /// How many sequences.
    pub count: usize,
    /// Their total length, when known.
    pub bases: Option<u64>,
}

/// Facts about how the input was read, filled in by the loader where they
/// apply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QcContext {
    /// Input format, e.g. `"psmcfa"`, `"vcf"` or `"mhs"`; shown as is.
    pub input_format: Option<String>,
    /// A VCF was read in mask mode without a positive (callable-sites) mask,
    /// so every position without a record counted as called homozygous
    /// ([`QcCode::VariantsOnlyNoMask`]).
    pub variants_only_no_mask: bool,
    /// The VCF was read as a gVCF: positions that no record or block covers
    /// are missing, but reference blocks count as called homozygous over
    /// their span ([`QcCode::VariantsOnlyNoMask`] words its finding for it).
    pub gvcf: bool,
    /// Uncalled sites were placed by assumption, as for multihetsep input
    /// ([`QcCode::ApproxMissingPlacement`]).
    pub approximate_missing_placement: bool,
    /// Depth and filter statistics of a consensus-mode conversion.
    pub consensus: Option<ConsensusStats>,
    /// Sequences the loader left out, by reason.
    pub dropped: Vec<DroppedSequences>,
    /// Free λ parameters the fit will estimate, when known: C psmc's
    /// `n_free_lambdas`, the number of groups in the pattern, or one per
    /// interval (`n_steps + 1`) without a pattern. [`QcCode::FewHets`] then
    /// gives the recombinations per parameter of this fit rather than of
    /// psmc's human pattern.
    pub free_lambdas: Option<usize>,
    /// The starting model of the fit, when known ([`QcCode::InvalidStart`]).
    pub start: Option<FitStart>,
}

/// The starting model of a fit, as [`QcCode::InvalidStart`] checks it: its
/// θ0, ρ0 and t_max, and whether its transition matrix is valid.
#[derive(Debug, Clone, PartialEq)]
pub struct FitStart {
    /// θ0 per bin.
    pub theta0: f64,
    /// ρ0 per bin.
    pub rho0: f64,
    /// Initial t_max, in units of 2N0 generations.
    pub t_max: f64,
    /// The pattern of the model, if any.
    pub pattern: Option<String>,
    /// The smallest entry of the starting transition matrix; NaN when an
    /// entry is not finite or the model cannot be computed at all.
    pub min_transition: f64,
    /// The largest ρ0 per bin whose transition matrix is valid, with this
    /// start's time grid and λ, when the start is invalid and the model has
    /// fewer than [`FitStart::MAX_SEARCHED_STATES`] states.
    pub max_rho0: Option<f64>,
    /// The start is not a fit's own but the one psmc-rs would take with its
    /// default settings (θ0 from the data, ρ0 = θ0 / [`DEFAULT_TR_RATIO`],
    /// t_max 15 and pattern `4+5*3+4`), for a report made without a fit
    /// (`psmc-rs qc`, `convert`).
    pub assumed: bool,
    /// The start is a saved fit's (`--init-params`), whose θ and ρ come from
    /// the file rather than from options; [`FitStart::new`] sets it false.
    pub saved: bool,
}

impl FitStart {
    /// [`FitStart::max_rho0`] is searched for in models of fewer states
    /// than this; the search recomputes the model about 60 times.
    pub const MAX_SEARCHED_STATES: usize = 1000;

    /// The start at θ0 = `theta0` and ρ0 = `rho0` of a model laid out as
    /// `layout`: its pattern, time intervals, t_max and λ (flat for a fresh
    /// start), whatever its own θ and ρ are. Every entry of the transition
    /// matrix must be a probability, at least 0.
    pub fn new(layout: &PsmcModel, theta0: f64, rho0: f64, assumed: bool) -> Self {
        let min_at = |rho: f64| {
            let mut model = layout.clone();
            model.theta = theta0;
            model.rho = rho;
            model.param_recalculate().ok()?;
            model
                .transition_matrix()
                .iter()
                .try_fold(f64::INFINITY, |min, &p| p.is_finite().then(|| min.min(p)))
        };
        let valid = |rho: f64| min_at(rho).is_some_and(|min| min >= 0.0);
        let min_transition = min_at(rho0).unwrap_or(f64::NAN);
        let searchable = layout.n_steps + 1 < Self::MAX_SEARCHED_STATES && rho0 > 0.0;
        let max_rho0 = if min_transition >= 0.0 || !searchable {
            None
        } else {
            // ρ only grows the chance of leaving a state, so the valid ρ0
            // form an interval from 0; bisect its end on a log scale.
            let (mut lo, mut hi) = (rho0 * 1e-9, rho0);
            if valid(lo) {
                for _ in 0..60 {
                    let mid = (lo * hi).sqrt();
                    if valid(mid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                Some(lo)
            } else {
                None
            }
        };
        Self {
            theta0,
            rho0,
            t_max: layout.t_max,
            pattern: layout.pattern.clone(),
            min_transition,
            max_rho0,
            assumed,
            saved: false,
        }
    }

    /// Whether every entry of the starting transition matrix is a
    /// probability (finite and at least 0).
    pub fn is_valid(&self) -> bool {
        self.min_transition >= 0.0
    }
}

/// What [`analyze`] looks at.
#[derive(Debug, Clone)]
pub struct QcInput<'a> {
    /// HMM rows of bins (0 = T, 1 = K, other = N).
    pub rows: &'a [Vec<u8>],
    /// `true` where a row starts a new sequence; `false` continues the
    /// previous one.
    pub row_starts: &'a [bool],
    /// The sequences of `rows`, in order. Sequences with 0 bins have no rows
    /// and are only counted. When empty, the sequences are taken from
    /// `row_starts` and named `seq1`, `seq2`, ...
    pub seqs: &'a [QcSeq],
    /// Bases per bin.
    pub bin_size: usize,
    /// Bootstrap trunk size in bins; 0 when no bootstrap is planned, which
    /// skips the trunk metrics.
    pub bootstrap_trunk_bins: usize,
    /// Facts about how the input was read.
    pub context: QcContext,
}

/// Bin counts and derived metrics of one sequence or of several.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct BinStats {
    /// All bins.
    pub bins: u64,
    /// Called bins without a heterozygous site (`T`).
    pub t_bins: u64,
    /// Bins with at least one heterozygous site (`K`).
    pub k_bins: u64,
    /// Missing bins (`N`).
    pub missing_bins: u64,
    /// `t_bins / bins` (0 when there are no bins).
    pub t_fraction: f64,
    /// `k_bins / bins` (0 when there are no bins).
    pub k_fraction: f64,
    /// `missing_bins / bins` (0 when there are no bins).
    pub missing_fraction: f64,
    /// Share of called bins that are heterozygous, `K / (T + K)`.
    pub het_bin_fraction: Option<f64>,
    /// θ0 per bin, `-ln(1 - K/(T+K))`, as C psmc starts from
    /// ([`theta0_per_bin`]).
    pub theta0_per_bin: Option<f64>,
    /// θ0 per base pair: `theta0_per_bin / bin_size`.
    pub theta_per_bp: Option<f64>,
    /// `theta_per_bp` per kilobase, i.e. heterozygous sites per kb.
    pub het_per_kb: Option<f64>,
    /// `(T + K) * bin_size`: bases in called bins. A called bin may still
    /// hold up to 90% missing sites (the `fq2psmcfa` rule), so this can
    /// exceed the called bases.
    pub called_bases_from_bins: u64,
    /// Called bases as counted by the loader, when it saw individual sites
    /// (for several sequences: only when every one has the count).
    pub called_bases: Option<u64>,
    /// Share of the sites that are uncalled, `1 - called_bases /
    /// length_bases` (0 if the loader counted more called sites than the
    /// length). Unlike `missing_fraction` it also counts the uncalled sites
    /// inside called bins. Without [`QcSeq::extent_bases`] the length is
    /// `bins * bin_size`, which counts a partial last bin as full.
    pub uncalled_site_fraction: Option<f64>,
    /// Heterozygous sites as counted by the loader, under the same rule.
    pub het_sites: Option<u64>,
    /// `het_sites / called_bases` per kilobase.
    pub site_het_per_kb: Option<f64>,
    /// Bases spanned ([`QcSeq::extent_bases`], else `bins * bin_size`).
    pub length_bases: u64,
    /// Longest run of consecutive `N` bins within one sequence.
    pub longest_missing_run_bins: u64,
}

/// How many sequences, and how many bins, lie below a length limit.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ShortShare {
    /// Sequences below the limit.
    pub sequences: usize,
    /// Bins in those sequences.
    pub bins: u64,
    /// `bins` as a fraction of all bins (0 when there are none).
    pub fraction: f64,
}

/// Distribution of sequence lengths.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LengthStats {
    /// Sum of the sequence lengths ([`BinStats::length_bases`]).
    pub total_bases: u64,
    /// Longest sequence.
    pub longest_bases: u64,
    /// N50 of the sequence lengths ([`n50`]).
    pub n50_bases: u64,
    /// Number of sequences that reach half the total at the N50.
    pub l50: usize,
    /// Sequences shorter than [`SHORT_SEQUENCE_BASES`].
    pub below_1mb: ShortShare,
    /// Sequences with fewer bins than the bootstrap trunk; `None` when no
    /// trunk size was given.
    pub below_trunk: Option<ShortShare>,
    /// Trunks the bootstrap cuts the input into ([`bootstrap_trunks_of`]);
    /// `None` when no trunk size was given.
    pub bootstrap_trunks: Option<u64>,
}

/// The longest run of `N` bins in the input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MissingRun {
    /// Sequence holding the run.
    pub sequence: String,
    /// 0-based bin offset of the run's first bin in that sequence.
    pub start_bin: u64,
    /// Length of the run in bins.
    pub bins: u64,
    /// Length of the run in bases (`bins * bin_size`).
    pub bases: u64,
}

/// How many heterozygous sites the `K` bins hide.
///
/// PSMC's emission model only asks whether a bin holds at least one
/// heterozygous site, so the bins discard how many each `K` bin holds. With
/// per-site heterozygosity `h` constant along the genome, the share of `K`
/// bins holding two or more follows from the observed `K` fraction alone
/// ([`het_per_site`], [`multi_het_share`]).
///
/// Heterozygosity varies along a real genome with the TMRCA, and that only
/// raises the share for the same `K` fraction, so these values are a lower
/// bound. Writing `u = P(X >= 1)` and `v = 1 - u`, the binomial gives
/// `P(X >= 2) = 1 - v - s v^((s-1)/s) + s v`, whose second derivative in `v`,
/// `((s-1)/s) v^(-1/s - 1)`, is positive: `P(X >= 2)` is convex in `u`, so by
/// Jensen's inequality a mixture of heterozygosities with mean `u` = the
/// observed `K` fraction has at least the share computed at that mean.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Saturation {
    /// Per-site heterozygosity implied by `K / (T + K)` at this bin size.
    pub het_per_site: f64,
    /// Expected share of `K` bins with two or more heterozygous sites.
    pub multi_het_share: f64,
    /// Largest bin size that would bring the share below
    /// [`SATURATION_TARGET_SHARE`] at the same heterozygosity; `None` when
    /// no bin size reaches it (no heterozygous bin).
    pub max_bin_size_below_target: Option<usize>,
}

impl Saturation {
    /// Saturation of bins of `bin_size` sites of which a share `p` of the
    /// called ones is heterozygous.
    pub fn from_het_fraction(p: f64, bin_size: usize) -> Self {
        let h = het_per_site(p, bin_size);
        Self {
            het_per_site: h,
            multi_het_share: multi_het_share(h, bin_size),
            max_bin_size_below_target: max_bin_size_for_share(h, SATURATION_TARGET_SHARE),
        }
    }
}

/// One consensus filter's share of the bases.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FilterShare {
    /// Filter name.
    pub name: String,
    /// Bases it removed.
    pub bases: u64,
    /// `bases / total_bases` (0 when there are none).
    pub fraction: f64,
}

/// Read depth of a consensus-mode conversion, with what the histogram
/// implies.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DepthReport {
    /// Mean depth, as given.
    pub mean: f64,
    /// Median depth, as given.
    pub median: f64,
    /// Minimum depth for a call, as given.
    pub min_depth: Option<u32>,
    /// Maximum depth for a call, as given.
    pub max_depth: Option<u32>,
    /// Depth quantiles 5%, 25%, 75% and 95% from the histogram (the smallest
    /// depth reaching each share); a value in the open last class is a lower
    /// bound. `None` without a histogram.
    pub quantiles: Option<[u32; 4]>,
    /// Share of positions below `min_depth`, from the histogram; `None` when
    /// the histogram's open last class does not tell (`min_depth` at least
    /// the histogram's length).
    pub below_min_fraction: Option<f64>,
    /// Share of positions above `max_depth`; `None` when the histogram's
    /// open last class does not tell.
    pub above_max_fraction: Option<f64>,
    /// The histogram, as given.
    pub histogram: Vec<u64>,
    /// Positions the filters were applied to.
    pub total_bases: u64,
    /// Each filter's share.
    pub filters: Vec<FilterShare>,
}

impl DepthReport {
    fn new(c: &ConsensusStats) -> Self {
        let h = &c.depth.histogram;
        let total: u64 = h.iter().sum();
        let share = |n: u64| n as f64 / total as f64;
        let quantiles = (total > 0).then(|| {
            [0.05, 0.25, 0.75, 0.95].map(|q| stats::histogram_quantile(h, q).unwrap_or(0))
        });
        // Classes 0..len-1 are exact and the last one holds every depth from
        // len-1 up, so it lies wholly at or above `lo` only when lo <= len-1.
        let below_min = c
            .depth
            .min_depth
            .filter(|&lo| total > 0 && (lo as usize) < h.len())
            .map(|lo| share(h.iter().take(lo as usize).sum()));
        let above_max = c
            .depth
            .max_depth
            .filter(|&hi| total > 0 && (hi as usize) + 1 < h.len())
            .map(|hi| share(h.iter().skip(hi as usize + 1).sum()));
        Self {
            mean: c.depth.mean,
            median: c.depth.median,
            min_depth: c.depth.min_depth,
            max_depth: c.depth.max_depth,
            quantiles,
            below_min_fraction: below_min,
            above_max_fraction: above_max,
            histogram: h.clone(),
            total_bases: c.total_bases,
            filters: c
                .filters
                .iter()
                .map(|f| FilterShare {
                    name: f.name.clone(),
                    bases: f.bases,
                    fraction: if c.total_bases == 0 {
                        0.0
                    } else {
                        f.bases as f64 / c.total_bases as f64
                    },
                })
                .collect(),
        }
    }
}

/// How [`HetReference`] was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceMethod {
    /// Median θ0 of the sequences with at least
    /// [`REFERENCE_MIN_CALLED_BASES`] in called bins.
    Median,
    /// θ0 of all their bins together, when no sequence is that long.
    Pooled,
}

/// The heterozygosity sequences are compared with: that of the sequences not
/// named like sex chromosomes or organelles ([`classify_name`]).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HetReference {
    /// θ0 per bin.
    pub theta0_per_bin: f64,
    /// The same per kb.
    pub het_per_kb: f64,
    /// Sequences it was computed from.
    pub sequences: usize,
    /// How it was computed.
    pub method: ReferenceMethod,
}

/// Metrics of one sequence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SequenceRow {
    /// 0-based position of the sequence among the non-empty input sequences.
    pub index: usize,
    /// Sequence name.
    pub name: String,
    /// Sex chromosome or organelle, judged by name ([`classify_name`]).
    pub kind: Option<SeqKind>,
    /// θ0 of this sequence over [`QcReport::het_reference`].
    pub het_vs_reference: Option<f64>,
    /// Its metrics.
    #[serde(flatten)]
    pub stats: BinStats,
}

/// Sequences left out of [`QcReport::sequences`], summed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OmittedSequences {
    /// How many sequences were left out.
    pub sequences: usize,
    /// Their metrics, summed as for the whole input.
    #[serde(flatten)]
    pub stats: BinStats,
}

/// Everything [`analyze`] found.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QcReport {
    /// [`QcContext::input_format`].
    pub input_format: Option<String>,
    /// Bases per bin.
    pub bin_size: usize,
    /// Bootstrap trunk size in bins (0: none).
    pub bootstrap_trunk_bins: usize,
    /// Sequences with at least one bin.
    pub n_sequences: usize,
    /// Sequences listed with 0 bins (not otherwise analysed).
    pub empty_sequences: usize,
    /// Metrics of the whole input.
    pub totals: BinStats,
    /// Sequence length distribution.
    pub lengths: LengthStats,
    /// Longest run of `N` bins; `None` when no bin is missing.
    pub longest_missing_run: Option<MissingRun>,
    /// Bin saturation of the whole input; `None` when no bin is called.
    pub saturation: Option<Saturation>,
    /// Heterozygosity of the sequences not named like sex chromosomes or
    /// organelles; `None` when they have no called bin.
    pub het_reference: Option<HetReference>,
    /// Read depth and filters of a consensus-mode conversion.
    pub depth: Option<DepthReport>,
    /// Sequences the loader left out, by reason.
    pub dropped: Vec<DroppedSequences>,
    /// C psmc's round-0 `n_recomb` for the whole input
    /// ([`initial_recombinations`] at [`DEFAULT_TR_RATIO`]).
    pub expected_recombinations: Option<f64>,
    /// Warnings and notes: warnings first, then in [`QcCode`] order.
    pub findings: Vec<Finding>,
    /// Per-sequence metrics: every sequence when there are at most
    /// [`MAX_SEQUENCE_ROWS`], otherwise those named in findings and then the
    /// longest ones, up to that many. Always in input order.
    pub sequences: Vec<SequenceRow>,
    /// The sequences not in `sequences`, if any.
    pub omitted: Option<OmittedSequences>,
}

impl QcReport {
    /// Whether any finding is a warning.
    pub fn has_warnings(&self) -> bool {
        self.findings
            .iter()
            .any(|f| f.severity == Severity::Warning)
    }

    /// The finding with this code, if any.
    pub fn finding(&self, code: QcCode) -> Option<&Finding> {
        self.findings.iter().find(|f| f.code == code)
    }

    /// The report as pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a QC report always serialises")
    }

    /// The finding codes and key metrics, for a fit's output file.
    pub fn summary(&self) -> QcSummary {
        let codes = |severity: Severity| {
            self.findings
                .iter()
                .filter(|f| f.severity == severity)
                .map(|f| f.code.as_str().to_string())
                .collect()
        };
        let t = &self.totals;
        QcSummary {
            warnings: codes(Severity::Warning),
            notes: codes(Severity::Info),
            sequences: self.n_sequences,
            bins: t.bins,
            missing_fraction: t.missing_fraction,
            theta0_per_bin: t.theta0_per_bin,
            het_per_kb: t.het_per_kb,
            called_bases: t.called_bases.unwrap_or(t.called_bases_from_bins),
            n50_bases: self.lengths.n50_bases,
        }
    }
}

/// The codes of a [`QcReport`]'s findings and its key metrics
/// ([`QcReport::summary`]), as stored with a fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QcSummary {
    /// Codes of the warnings ([`QcCode::as_str`]).
    pub warnings: Vec<String>,
    /// Codes of the notes.
    pub notes: Vec<String>,
    /// [`QcReport::n_sequences`].
    pub sequences: usize,
    /// [`BinStats::bins`] of the whole input.
    pub bins: u64,
    /// [`BinStats::missing_fraction`].
    pub missing_fraction: f64,
    /// [`BinStats::theta0_per_bin`].
    pub theta0_per_bin: Option<f64>,
    /// [`BinStats::het_per_kb`].
    pub het_per_kb: Option<f64>,
    /// Called length: [`BinStats::called_bases`] when the loader counted
    /// sites, otherwise [`BinStats::called_bases_from_bins`].
    pub called_bases: u64,
    /// [`LengthStats::n50_bases`].
    pub n50_bases: u64,
}

/// Raw counts of one sequence, before derived metrics.
#[derive(Debug, Clone, Default)]
struct Counts {
    t: u64,
    k: u64,
    n: u64,
    longest_run: u64,
    longest_run_start: u64,
    length_bases: u64,
    called_bases: Option<u64>,
    het_sites: Option<u64>,
}

impl Counts {
    fn bins(&self) -> u64 {
        self.t + self.k + self.n
    }

    /// Count the bins of one sequence, held in `rows`.
    fn scan(rows: &[Vec<u8>]) -> Self {
        let mut c = [0u64; 3];
        let (mut run, mut best, mut best_end) = (0u64, 0u64, 0u64);
        let mut pos = 0u64;
        for row in rows {
            for &b in row {
                let class = usize::from(b.min(2));
                c[class] += 1;
                if class == 2 {
                    run += 1;
                    if run > best {
                        best = run;
                        best_end = pos;
                    }
                } else {
                    run = 0;
                }
                pos += 1;
            }
        }
        Self {
            t: c[0],
            k: c[1],
            n: c[2],
            longest_run: best,
            longest_run_start: if best > 0 { best_end + 1 - best } else { 0 },
            ..Self::default()
        }
    }

    /// Add `other` (a further sequence) to these totals.
    fn add(&mut self, other: &Counts) {
        self.t += other.t;
        self.k += other.k;
        self.n += other.n;
        self.longest_run = self.longest_run.max(other.longest_run);
        self.length_bases += other.length_bases;
        self.called_bases = self
            .called_bases
            .zip(other.called_bases)
            .map(|(a, b)| a + b);
        self.het_sites = self.het_sites.zip(other.het_sites).map(|(a, b)| a + b);
    }

    /// Neutral start for [`Counts::add`].
    fn empty_total() -> Self {
        Self {
            called_bases: Some(0),
            het_sites: Some(0),
            ..Self::default()
        }
    }

    fn stats(&self, bin_size: usize) -> BinStats {
        let bins = self.bins();
        let frac = |x: u64| {
            if bins == 0 {
                0.0
            } else {
                x as f64 / bins as f64
            }
        };
        let called = self.t + self.k;
        let theta0 = theta0_per_bin(self.t, self.k);
        let theta_bp = theta0.map(|t| t / bin_size as f64);
        let site_het = match (self.het_sites, self.called_bases) {
            (Some(h), Some(c)) if c > 0 => Some(1000.0 * h as f64 / c as f64),
            _ => None,
        };
        let uncalled = self
            .called_bases
            .filter(|_| self.length_bases > 0)
            .map(|c| (1.0 - c as f64 / self.length_bases as f64).max(0.0));
        BinStats {
            bins,
            t_bins: self.t,
            k_bins: self.k,
            missing_bins: self.n,
            t_fraction: frac(self.t),
            k_fraction: frac(self.k),
            missing_fraction: frac(self.n),
            het_bin_fraction: (called > 0).then(|| self.k as f64 / called as f64),
            theta0_per_bin: theta0,
            theta_per_bp: theta_bp,
            het_per_kb: theta_bp.map(|t| 1000.0 * t),
            called_bases_from_bins: called * bin_size as u64,
            called_bases: self.called_bases,
            uncalled_site_fraction: uncalled,
            het_sites: self.het_sites,
            site_het_per_kb: site_het,
            length_bases: self.length_bases,
            longest_missing_run_bins: self.longest_run,
        }
    }
}

/// One analysed sequence, as the checks see it.
pub(crate) struct SeqInfo {
    index: usize,
    name: String,
    kind: Option<SeqKind>,
    stats: BinStats,
}

/// One non-empty sequence of the input: its name, rows and loader facts.
struct Span<'a> {
    name: Cow<'a, str>,
    rows: Range<usize>,
    seq: Option<&'a QcSeq>,
}

/// Group `rows` into sequences and pair them with `seqs`.
fn spans<'a>(input: &QcInput<'a>) -> Result<Vec<Span<'a>>> {
    if input.bin_size == 0 {
        bail!("QC: bin_size must be > 0");
    }
    if input.rows.len() != input.row_starts.len() {
        bail!(
            "QC: {} rows but {} row_starts",
            input.rows.len(),
            input.row_starts.len()
        );
    }
    if input.row_starts.first() == Some(&false) {
        bail!("QC: the first row must start a sequence (row_starts[0] is false)");
    }
    let mut groups: Vec<(Range<usize>, usize)> = Vec::new();
    for (i, (row, &start)) in input.rows.iter().zip(input.row_starts).enumerate() {
        match groups.last_mut() {
            Some((range, bins)) if !start => {
                range.end = i + 1;
                *bins += row.len();
            }
            _ => groups.push((i..i + 1, row.len())),
        }
    }
    groups.retain(|(_, bins)| *bins > 0);

    if input.seqs.is_empty() {
        return Ok(groups
            .into_iter()
            .enumerate()
            .map(|(i, (rows, _))| Span {
                name: Cow::Owned(format!("seq{}", i + 1)),
                rows,
                seq: None,
            })
            .collect());
    }
    let listed: Vec<&QcSeq> = input.seqs.iter().filter(|s| s.n_bins > 0).collect();
    if listed.len() != groups.len() {
        bail!(
            "QC: {} non-empty sequences listed but the rows hold {}",
            listed.len(),
            groups.len()
        );
    }
    groups
        .into_iter()
        .zip(listed)
        .map(|((rows, bins), seq)| {
            if seq.n_bins != bins {
                bail!(
                    "QC: sequence '{}' is listed with {} bins but its rows hold {bins}",
                    seq.name,
                    seq.n_bins
                );
            }
            Ok(Span {
                name: Cow::Borrowed(seq.name.as_str()),
                rows,
                seq: Some(seq),
            })
        })
        .collect()
}

/// Analyse `input`; see [`try_analyze`].
///
/// # Panics
///
/// When `input` is inconsistent (see [`try_analyze`]).
pub fn analyze(input: &QcInput) -> QcReport {
    try_analyze(input).unwrap_or_else(|e| panic!("{e:#}"))
}

/// Analyse `input` into a [`QcReport`]. The output depends only on the input.
///
/// # Errors
///
/// When `bin_size` is 0, `rows` and `row_starts` differ in length, the first
/// row continues a sequence, or `seqs` (when not empty) does not list the
/// sequences of `rows` in order with their bin counts.
pub fn try_analyze(input: &QcInput) -> Result<QcReport> {
    let spans = spans(input)?;
    let bin_size = input.bin_size;
    let counts: Vec<Counts> = spans
        .iter()
        .map(|span| {
            let mut c = Counts::scan(&input.rows[span.rows.clone()]);
            let seq = span.seq;
            c.length_bases = seq
                .and_then(|s| s.extent_bases)
                .unwrap_or(c.bins() * bin_size as u64);
            c.called_bases = seq.and_then(|s| s.called_bases);
            c.het_sites = seq.and_then(|s| s.het_sites);
            c
        })
        .collect();

    let mut total = Counts::empty_total();
    counts.iter().for_each(|c| total.add(c));
    if counts.is_empty() {
        total.called_bases = None;
        total.het_sites = None;
    }
    let totals = total.stats(bin_size);

    // The first sequence wins ties.
    let longest_missing_run = counts
        .iter()
        .zip(&spans)
        .fold(
            None,
            |best: Option<(&Counts, &Span)>, (c, span)| match best {
                Some((b, _)) if b.longest_run >= c.longest_run => best,
                _ if c.longest_run > 0 => Some((c, span)),
                _ => best,
            },
        )
        .map(|(c, span)| MissingRun {
            sequence: span.name.to_string(),
            start_bin: c.longest_run_start,
            bins: c.longest_run,
            bases: c.longest_run * bin_size as u64,
        });

    let lengths = length_stats(&counts, input.bootstrap_trunk_bins as u64, totals.bins);
    let saturation = totals
        .het_bin_fraction
        .map(|p| Saturation::from_het_fraction(p, bin_size));
    let expected_recombinations = totals.theta0_per_bin.map(|theta| {
        initial_recombinations(totals.t_bins + totals.k_bins, theta, DEFAULT_TR_RATIO)
    });

    let infos: Vec<SeqInfo> = spans
        .iter()
        .zip(&counts)
        .enumerate()
        .map(|(index, (span, c))| SeqInfo {
            index,
            name: span.name.to_string(),
            kind: classify_name(&span.name),
            stats: c.stats(bin_size),
        })
        .collect();
    let het_reference = het_reference(&infos, bin_size);

    let mut report = QcReport {
        input_format: input.context.input_format.clone(),
        bin_size,
        bootstrap_trunk_bins: input.bootstrap_trunk_bins,
        n_sequences: counts.len(),
        empty_sequences: input.seqs.iter().filter(|s| s.n_bins == 0).count(),
        totals,
        lengths,
        longest_missing_run,
        saturation,
        het_reference,
        depth: input.context.consensus.as_ref().map(DepthReport::new),
        dropped: input.context.dropped.clone(),
        expected_recombinations,
        findings: Vec::new(),
        sequences: Vec::new(),
        omitted: None,
    };
    let (findings, named) = checks::run(input, &report, &infos);
    report.findings = findings;

    let shown = rows_to_show(&counts, &named, MAX_SEQUENCE_ROWS);
    let reference = report.het_reference.as_ref().map(|r| r.theta0_per_bin);
    report.sequences = shown
        .iter()
        .map(|&i| {
            let info = &infos[i];
            SequenceRow {
                index: i,
                name: info.name.clone(),
                kind: info.kind,
                het_vs_reference: info
                    .stats
                    .theta0_per_bin
                    .zip(reference)
                    .filter(|&(_, r)| r > 0.0)
                    .map(|(t, r)| t / r),
                stats: info.stats.clone(),
            }
        })
        .collect();
    report.omitted = (shown.len() < counts.len()).then(|| {
        let mut rest = Counts::empty_total();
        let mut keep = shown.iter().peekable();
        for (i, c) in counts.iter().enumerate() {
            if keep.peek() == Some(&&i) {
                keep.next();
            } else {
                rest.add(c);
            }
        }
        OmittedSequences {
            sequences: counts.len() - shown.len(),
            stats: rest.stats(bin_size),
        }
    });
    Ok(report)
}

/// See [`HetReference`].
fn het_reference(infos: &[SeqInfo], bin_size: usize) -> Option<HetReference> {
    let autosomes = || infos.iter().filter(|s| s.kind.is_none());
    let mut long: Vec<f64> = autosomes()
        .filter(|s| s.stats.called_bases_from_bins >= REFERENCE_MIN_CALLED_BASES)
        .filter_map(|s| s.stats.theta0_per_bin)
        .collect();
    let (theta, sequences, method) = if long.is_empty() {
        let (t, k, n) = autosomes()
            .filter(|s| s.stats.t_bins + s.stats.k_bins > 0)
            .fold((0, 0, 0), |(t, k, n), s| {
                (t + s.stats.t_bins, k + s.stats.k_bins, n + 1)
            });
        (theta0_per_bin(t, k)?, n, ReferenceMethod::Pooled)
    } else {
        let n = long.len();
        (stats::median(&mut long), n, ReferenceMethod::Median)
    };
    Some(HetReference {
        theta0_per_bin: theta,
        het_per_kb: 1000.0 * theta / bin_size as f64,
        sequences,
        method,
    })
}

fn length_stats(counts: &[Counts], trunk: u64, total_bins: u64) -> LengthStats {
    let lengths: Vec<u64> = counts.iter().map(|c| c.length_bases).collect();
    let (n50_bases, l50) = n50(&lengths);
    let share = |below: &dyn Fn(&Counts) -> bool| {
        let (sequences, bins) = counts
            .iter()
            .filter(|c| below(c))
            .fold((0usize, 0u64), |(n, b), c| (n + 1, b + c.bins()));
        ShortShare {
            sequences,
            bins,
            fraction: if total_bins == 0 {
                0.0
            } else {
                bins as f64 / total_bins as f64
            },
        }
    };
    LengthStats {
        total_bases: lengths.iter().sum(),
        longest_bases: lengths.iter().copied().max().unwrap_or(0),
        n50_bases,
        l50,
        below_1mb: share(&|c| c.length_bases < SHORT_SEQUENCE_BASES),
        below_trunk: (trunk > 0).then(|| share(&|c| c.bins() < trunk)),
        bootstrap_trunks: (trunk > 0).then(|| {
            counts
                .iter()
                .map(|c| bootstrap_trunks_of(c.bins(), trunk))
                .sum()
        }),
    }
}

/// Indices (ascending) of the sequences that get their own row: all of them
/// up to `cap`, otherwise those in `named` (ascending; the first `cap` of
/// them) and then the most bins (earlier first on ties).
fn rows_to_show(counts: &[Counts], named: &[usize], cap: usize) -> Vec<usize> {
    if counts.len() <= cap {
        return (0..counts.len()).collect();
    }
    let mut keep = vec![false; counts.len()];
    let mut order: Vec<usize> = named.iter().copied().take(cap).collect();
    order.iter().for_each(|&i| keep[i] = true);
    let mut rest: Vec<usize> = (0..counts.len()).filter(|&i| !keep[i]).collect();
    rest.sort_by(|&a, &b| counts[b].bins().cmp(&counts[a].bins()).then(a.cmp(&b)));
    order.extend(rest.into_iter().take(cap - order.len()));
    order.sort_unstable();
    order
}
