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
//!     callable: None,
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
use std::cmp::Reverse;
use std::ops::Range;

use crate::io::{BinCounts, CallableBases, CallableSummary, sequence_ranges};
use crate::model::{PsmcModel, ThetaCounts};

mod checks;
#[doc(hidden)]
pub mod fmt;
mod names;
mod stats;
mod text;

pub(crate) use checks::callable_distribution;
use stats::fraction;

pub use checks::{
    APPROX_PLACEMENT_WARN_FRACTION, Finding, HEMIZYGOUS_MAX_RATIO,
    HEMIZYGOUS_MIN_EXPECTED_HET_BINS, HET_OUTLIER_MADS, HET_OUTLIER_MIN_FOLD,
    HET_OUTLIER_MIN_SEQUENCES, HIGH_MISSING_INFO_FRACTION, HIGH_MISSING_WARN_FRACTION,
    MAX_FILTER_FRACTION, MIN_CALLED_BASES, MIN_HET_BINS, MIN_MEAN_DEPTH,
    PARTLY_CALLABLE_NOTE_FRACTION, PARTLY_CALLABLE_WARN_MEAN, QcCode, REFERENCE_MIN_CALLED_BASES,
    SHORT_SEQUENCES_MAX_FRACTION, Severity,
};
pub use names::{SeqKind, classify_name};
pub use text::{TextOptions, render_text, render_text_with};

pub use stats::{
    bootstrap_trunks_of, het_per_site, initial_recombinations, largest_trunk_within,
    max_bin_size_for_share, multi_het_share, n50, theta0_per_bin,
};

/// What the reports add to the callable bases of the called bins when they
/// follow multihetsep's placement of the uncalled sites, which
/// [`QcCode::ApproxMissingPlacement`] describes (that finding is given for
/// every multihetsep input); empty otherwise.
pub(crate) fn approximate_counts_note(r: &QcReport) -> &'static str {
    if r.callable.is_some() && r.finding(QcCode::ApproxMissingPlacement).is_some() {
        " (approximate: they follow multihetsep's placement of the uncalled sites, see \
         APPROX_MISSING_PLACEMENT)"
    } else {
        ""
    }
}

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
    /// The fit treats every called bin as `bin_size` bases although the
    /// input records their callable bases: `--bin-callable ignore`, or
    /// `auto` on multihetsep input, whose approximate counts `auto` does not
    /// fit. QC then warns about the bias that causes when the counts are
    /// exact ([`QcCode::PartlyCallableBins`]). False by default, as for a
    /// fit that uses the counts.
    pub counts_ignored: bool,
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
    /// The callable bases of each bin, row for row, when the input records
    /// them (VCF exactly, multihetsep approximately): QC then reports how
    /// much of the called bins is callable ([`QcReport::callable`],
    /// [`QcCode::PartlyCallableBins`]) and compares the heterozygosity of
    /// sequences per callable base ([`BinStats::theta0_callable`]).
    pub callable: Option<&'a CallableBases>,
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
    /// With callable bases, θ0 per bin of `bin_size` callable bases, as a
    /// fit with them starts from ([`crate::model::ThetaCounts::theta0`]):
    /// unlike `theta0_per_bin` it does not fall with the share of missing
    /// bases in called bins, so QC compares sequences by it.
    pub theta0_callable: Option<f64>,
    /// With callable bases, the mean share of callable bases of the called
    /// bins.
    pub mean_callable_fraction: Option<f64>,
    /// With callable bases, the share of called bins that hold fewer than
    /// `bin_size` callable bases.
    pub partly_callable_fraction: Option<f64>,
    /// Called bins next to a missing (`N`) bin, at the edges of runs of
    /// missing data, where called bins usually hold fewer than `bin_size`
    /// callable bases: a lower bound on the partly callable bins of input
    /// that does not record callable bases (psmcfa).
    pub called_next_to_missing: u64,
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

impl BinStats {
    /// The θ0 by which QC compares the heterozygosity of sequences: per
    /// callable base when the input records callable bases
    /// ([`BinStats::theta0_callable`]), so that sequences with more missing
    /// data do not look less heterozygous, and C psmc's otherwise.
    pub fn theta0_comparable(&self) -> Option<f64> {
        self.theta0_callable.or(self.theta0_per_bin)
    }

    /// Heterozygous sites per kb of [`BinStats::theta0_comparable`]: per
    /// callable base when the input records callable bases, else
    /// [`BinStats::het_per_kb`].
    pub fn het_per_kb_comparable(&self) -> Option<f64> {
        match (self.theta0_callable, self.theta0_per_bin, self.het_per_kb) {
            (Some(c), Some(t), Some(h)) if t > 0.0 => Some(h * (c / t)),
            _ => self.het_per_kb,
        }
    }
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
                    fraction: fraction(f.bases, c.total_bases),
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
/// named like sex chromosomes or organelles ([`classify_name`]). With
/// callable bases ([`QcInput::callable`]) it is per callable base
/// ([`BinStats::theta0_comparable`]), so that sequences with more missing
/// data compare fairly.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HetReference {
    /// θ0 per bin (of `bin_size` callable bases, with callable bases).
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
    /// How much of the called bins is callable, when the input records the
    /// callable bases of each bin ([`QcInput::callable`]): their mean share
    /// and distribution.
    pub callable: Option<CallableSummary>,
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
            het_per_callable_kb: t.theta0_callable.and(t.het_per_kb_comparable()),
            called_bases: t.called_bases.unwrap_or(t.called_bases_from_bins),
            n50_bases: self.lengths.n50_bases,
        }
    }

    /// Whether the per-sequence tables give heterozygosity per kb of
    /// callable bases ([`BinStats::het_per_kb_comparable`]) rather than per
    /// kb of called bins, as they do when the input records callable bases;
    /// the reports then mark the column and say so under it
    /// ([`HET_PER_CALLABLE_KB_NOTE`]).
    pub fn het_per_callable_kb(&self) -> bool {
        self.totals.theta0_callable.is_some()
    }
}

/// What the reports say under a per-sequence table whose heterozygosity is
/// per kb of callable bases ([`QcReport::het_per_callable_kb`]).
pub const HET_PER_CALLABLE_KB_NOTE: &str = "het/kb* is per kb of callable bases (from theta0 of \
     the callable-length model), so that sequences with more missing data do not look less \
     heterozygous; the totals' theta0 and het per kb, and qc.het_per_kb of the output JSON, are \
     per kb of called bins (C psmc's -ln(1 - K/L)), and qc.het_per_callable_kb is the total per \
     kb of callable bases.";

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
    /// [`BinStats::het_per_kb`]: per kb of called bins.
    pub het_per_kb: Option<f64>,
    /// With callable bases, heterozygous sites per kb of callable bases of
    /// the whole input ([`BinStats::het_per_kb_comparable`]), the
    /// heterozygosity the per-sequence tables give; absent without them
    /// (and in files of psmc-rs before it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub het_per_callable_kb: Option<f64>,
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
    /// Called bins next to an `N` bin.
    next_to_missing: u64,
    longest_run: u64,
    longest_run_start: u64,
    length_bases: u64,
    called_bases: Option<u64>,
    het_sites: Option<u64>,
    /// With callable bases: what θ0 per callable base is estimated from,
    /// and how many called bins hold each count of callable bases.
    theta: Option<ThetaCounts>,
}

impl Counts {
    fn bins(&self) -> u64 {
        self.t + self.k + self.n
    }

    /// Count the bins of one sequence, held in `rows`, with their callable
    /// bases `counts` (row for row) when the input records them.
    ///
    /// The bins are taken run by run; a run of missing or of called bins
    /// can continue from one row into the next.
    fn scan(rows: &[Vec<u8>], counts: Option<&[BinCounts]>) -> Result<Self> {
        let mut out = Self::default();
        let (mut missing_run, mut called_run) = (0u64, 0u64);
        // Whether a missing run comes before the called run.
        let mut after_missing = false;
        let mut pos = 0u64;
        for row in rows {
            let mut rest = row.as_slice();
            while let Some(&first) = rest.first() {
                let missing = first >= 2;
                let len = rest
                    .iter()
                    .position(|&b| (b >= 2) != missing)
                    .unwrap_or(rest.len());
                let (run, tail) = rest.split_at(len);
                rest = tail;
                let len = len as u64;
                pos += len;
                if missing {
                    out.n += len;
                    // The called run before borders this missing run with
                    // its last bin, and the missing run before it with its
                    // first.
                    out.next_to_missing += called_run.min(1 + u64::from(after_missing));
                    called_run = 0;
                    missing_run += len;
                    // Only a longer run replaces the first longest one.
                    if missing_run > out.longest_run {
                        out.longest_run = missing_run;
                        out.longest_run_start = pos - missing_run;
                    }
                } else {
                    let k = run.iter().filter(|&&b| b == 1).count() as u64;
                    out.k += k;
                    out.t += len - k;
                    if called_run == 0 {
                        after_missing = missing_run > 0;
                    }
                    missing_run = 0;
                    called_run += len;
                }
            }
        }
        // A called run at the end borders only the missing run before it.
        out.next_to_missing += called_run.min(u64::from(after_missing));
        if let Some(counts) = counts {
            if counts.len() != rows.len() {
                bail!(
                    "QC: callable bases for {} rows, but the sequence has {}",
                    counts.len(),
                    rows.len()
                );
            }
            let mut theta = ThetaCounts::new(counts.first().map(BinCounts::bin_size));
            for (row, cs) in rows.iter().zip(counts) {
                theta.add(&ThetaCounts::of_counted(row, cs)?);
            }
            out.theta = Some(theta);
        }
        Ok(out)
    }

    /// The totals of several sequences, except for `longest_run_start`: the
    /// loader's site counts only when every one has them, and none at all
    /// without sequences.
    fn sum<'a>(counts: impl IntoIterator<Item = &'a Counts>) -> Self {
        let mut counts = counts.into_iter();
        let Some(mut total) = counts.next().cloned() else {
            return Self::default();
        };
        for c in counts {
            total.t += c.t;
            total.k += c.k;
            total.n += c.n;
            total.next_to_missing += c.next_to_missing;
            total.longest_run = total.longest_run.max(c.longest_run);
            total.length_bases += c.length_bases;
            total.called_bases = total.called_bases.zip(c.called_bases).map(|(a, b)| a + b);
            total.het_sites = total.het_sites.zip(c.het_sites).map(|(a, b)| a + b);
            match (&mut total.theta, &c.theta) {
                (Some(a), Some(b)) => a.add(b),
                (a @ None, Some(b)) => *a = Some(b.clone()),
                _ => {}
            }
        }
        total
    }

    fn stats(&self, bin_size: usize) -> BinStats {
        let bins = self.bins();
        let frac = |x: u64| fraction(x, bins);
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
        // With callable bases: their mean share in the called bins, and the
        // share of called bins with fewer than the bin size.
        let callable = self.theta.as_ref().filter(|_| called > 0).map(|t| {
            let s = t.called_by_count.len() - 1;
            let sum: u64 = (0..).zip(&t.called_by_count).map(|(c, &n)| c * n).sum();
            let partial = called - t.called_by_count[s];
            (
                sum as f64 / (s as f64 * called as f64),
                partial as f64 / called as f64,
            )
        });
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
            theta0_callable: self.theta.as_ref().and_then(|t| t.theta0().ok()),
            mean_callable_fraction: callable.map(|(mean, _)| mean),
            partly_callable_fraction: callable.map(|(_, partly)| partly),
            called_next_to_missing: self.next_to_missing,
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

/// [`CallableSummary::of`] the input, from the counts of all its sequences,
/// `total`, and their `stats`, without another pass over its bins.
fn callable_summary(bin_size: usize, total: &Counts, stats: &BinStats) -> CallableSummary {
    let by_count = total.theta.as_ref().map_or(&[][..], |t| &t.called_by_count);
    let mut histogram = [0u64; 10];
    for (c, &n) in by_count.iter().enumerate() {
        // Tenths of the bin size, the full bins apart.
        let class = if c == bin_size {
            9
        } else {
            (10 * c / bin_size).clamp(1, 9) - 1
        };
        histogram[class] += n;
    }
    let called = stats.t_bins + stats.k_bins;
    CallableSummary {
        bin_size,
        called_bins: called,
        partial_bins: called - histogram[9],
        mean_fraction: stats.mean_callable_fraction.unwrap_or(0.0),
        histogram,
    }
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
    let groups: Vec<(Range<usize>, usize)> = sequence_ranges(input.row_starts)
        .into_iter()
        .map(|rows| {
            let bins = input.rows[rows.clone()].iter().map(Vec::len).sum();
            (rows, bins)
        })
        .filter(|&(_, bins)| bins > 0)
        .collect();

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
    if let Some(c) = input.callable {
        if c.rows().len() != input.rows.len() {
            bail!(
                "QC: callable bases for {} rows, but there are {} rows of bins",
                c.rows().len(),
                input.rows.len()
            );
        }
        if c.bin_size() != bin_size {
            bail!(
                "QC: callable bases of bins of {} bases, for bins of {bin_size}",
                c.bin_size()
            );
        }
    }
    let counts: Vec<Counts> = spans
        .iter()
        .map(|span| -> Result<Counts> {
            let callable = input.callable.map(|c| &c.rows()[span.rows.clone()]);
            let mut c = Counts::scan(&input.rows[span.rows.clone()], callable)?;
            let seq = span.seq;
            c.length_bases = seq
                .and_then(|s| s.extent_bases)
                .unwrap_or(c.bins() * bin_size as u64);
            c.called_bases = seq.and_then(|s| s.called_bases);
            c.het_sites = seq.and_then(|s| s.het_sites);
            Ok(c)
        })
        .collect::<Result<_>>()?;
    let total = Counts::sum(&counts);
    let totals = total.stats(bin_size);
    let callable = input
        .callable
        .map(|c| callable_summary(c.bin_size(), &total, &totals));

    // min_by_key keeps the first sequence of the longest run.
    let longest_missing_run = counts
        .iter()
        .zip(&spans)
        .min_by_key(|(c, _)| Reverse(c.longest_run))
        .filter(|(c, _)| c.longest_run > 0)
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

    let mut sequences: Vec<SequenceRow> = spans
        .iter()
        .zip(&counts)
        .enumerate()
        .map(|(index, (span, c))| SequenceRow {
            index,
            name: span.name.to_string(),
            kind: classify_name(&span.name),
            het_vs_reference: None,
            stats: c.stats(bin_size),
        })
        .collect();
    let het_reference = het_reference(&sequences, &counts, bin_size);
    let reference = het_reference
        .as_ref()
        .map(|r| r.theta0_per_bin)
        .filter(|&r| r > 0.0);
    for s in &mut sequences {
        s.het_vs_reference = s
            .stats
            .theta0_comparable()
            .zip(reference)
            .map(|(t, r)| t / r);
    }

    let mut report = QcReport {
        input_format: input.context.input_format.clone(),
        bin_size,
        bootstrap_trunk_bins: input.bootstrap_trunk_bins,
        n_sequences: counts.len(),
        empty_sequences: input.seqs.iter().filter(|s| s.n_bins == 0).count(),
        totals,
        lengths,
        longest_missing_run,
        callable,
        saturation,
        het_reference,
        depth: input.context.consensus.as_ref().map(DepthReport::new),
        dropped: input.context.dropped.clone(),
        expected_recombinations,
        findings: Vec::new(),
        sequences: Vec::new(),
        omitted: None,
    };
    let (findings, named) = checks::run(input, &report, &sequences);
    report.findings = findings;

    let shown = rows_to_show(&counts, &named, MAX_SEQUENCE_ROWS);
    let hidden: Vec<&Counts> = counts
        .iter()
        .zip(&shown)
        .filter(|(_, shown)| !**shown)
        .map(|(c, _)| c)
        .collect();
    report.omitted = (!hidden.is_empty()).then(|| OmittedSequences {
        sequences: hidden.len(),
        stats: Counts::sum(hidden).stats(bin_size),
    });
    report.sequences = sequences
        .into_iter()
        .zip(shown)
        .filter_map(|(s, shown)| shown.then_some(s))
        .collect();
    Ok(report)
}

/// See [`HetReference`]. θ0 is per callable base when the input records
/// callable bases ([`BinStats::theta0_comparable`]).
fn het_reference(
    sequences: &[SequenceRow],
    counts: &[Counts],
    bin_size: usize,
) -> Option<HetReference> {
    let autosomes = || {
        sequences
            .iter()
            .zip(counts)
            .filter(|(s, _)| s.kind.is_none())
    };
    let mut long: Vec<f64> = autosomes()
        .filter(|(s, _)| s.stats.called_bases_from_bins >= REFERENCE_MIN_CALLED_BASES)
        .filter_map(|(s, _)| s.stats.theta0_comparable())
        .collect();
    let (theta, sequences, method) = if long.is_empty() {
        let called: Vec<&Counts> = autosomes()
            .map(|(_, c)| c)
            .filter(|c| c.t + c.k > 0)
            .collect();
        let pooled = Counts::sum(called.iter().copied());
        let theta = match &pooled.theta {
            Some(p) => p.theta0().ok()?,
            None => theta0_per_bin(pooled.t, pooled.k)?,
        };
        (theta, called.len(), ReferenceMethod::Pooled)
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
            .fold((0, 0), |(n, b), c| (n + 1, b + c.bins()));
        ShortShare {
            sequences,
            bins,
            fraction: fraction(bins, total_bins),
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

/// Which sequences get their own row: all of them up to `cap`, otherwise
/// those in `named` (ascending; the first `cap` of them) and then those with
/// the most bins (earlier first on ties).
fn rows_to_show(counts: &[Counts], named: &[usize], cap: usize) -> Vec<bool> {
    if counts.len() <= cap {
        return vec![true; counts.len()];
    }
    let mut keep = vec![false; counts.len()];
    let named = &named[..named.len().min(cap)];
    named.iter().for_each(|&i| keep[i] = true);
    let mut rest: Vec<usize> = (0..counts.len()).filter(|&i| !keep[i]).collect();
    rest.sort_by_key(|&i| (Reverse(counts[i].bins()), i));
    rest.into_iter()
        .take(cap - named.len())
        .for_each(|i| keep[i] = true);
    keep
}
