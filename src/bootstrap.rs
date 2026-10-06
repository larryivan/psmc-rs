//! Bootstrap for PSMC curves, following C psmc.
//!
//! C psmc bootstraps in two steps. `utils/splitfa` cuts every input sequence
//! into trunks (500,000 bins by default; a remainder shorter than 1.5 trunks
//! stays in the last trunk). Each `psmc -b` run then draws trunks with
//! replacement from the whole input until the original length is matched
//! (`psmc_resamp`), treats every trunk as its own sequence, and fits from the
//! usual fresh start. [`run_bootstrap`] does the same with
//! [`split_for_bootstrap`], [`resample_chunks`] and
//! [`replicate_initial_model`], running replicates in parallel. Replicate `i`
//! draws from [`replicate_rng`]`(seed, i)`, so a seed reproduces the resampled
//! data on any platform, and the fits with any thread count, batch size and
//! alpha-cache budget. A run that continues saved parameters (`psmc -i`)
//! starts its replicates from those parameters too; see
//! [`BootstrapOptions::start`].
//!
//! Input with the callable bases of each bin ([`Sequences`]) keeps them
//! through the bootstrap: [`observations_to_sequences`],
//! [`split_sequences_for_bootstrap`] and [`resample_sequences`] cut and draw
//! the counts with their bins, and [`run_bootstrap_sequences`] fits each
//! replicate with the callable-length model of
//! `docs/design/bin-callable-model.md` from a fresh start whose θ0 comes
//! from its own counts ([`replicate_initial_model_with_callable`]), as the
//! main fit of such data does. The trunks drawn are the same with or
//! without counts.
//!
//! A replicate's fresh start takes θ0 from its own resampled data, so a
//! resample more heterozygous than the input starts from a larger ρ0. Near
//! the limit on ρ of PSMC's transition probabilities
//! ([`PsmcModel::check_transitions`]), some replicates can start past it
//! while the main fit does not. [`invalid_replicate_starts`] finds them
//! before anything is fitted, and [`BootstrapOptions::skip_invalid_starts`]
//! leaves them out instead of failing.
//!
//! # Choosing the trunk size
//!
//! Replicates are built from whole trunks, so the trunk count bounds how much
//! they can vary. A single trunk gives every replicate exactly the input and
//! a spread of zero; C psmc would run that, [`run_bootstrap`] refuses it
//! ([`bootstrap_trunk_count`]). A handful of trunks gives only a few distinct
//! resamples and coarse bands. `splitfa`'s default (500,000 bins, 50 Mb at
//! 100 bp per bin) gives roughly 60 trunks on a human genome; for a smaller
//! input, pick a trunk size that still gives a few dozen trunks.
//!
//! # How far to trust the bands
//!
//! The bootstrap measures how much the estimate moves when trunks are
//! resampled. It misses the estimator's bias and tends to understate its
//! sampling variability. `examples/bootstrap_calibration.rs` simulated 40
//! sequences of 10,000 bins from a known `4+5*3+4` model (θ = 0.05 and
//! ρ = 0.01 per bin) and fitted them as the CLI does by default (a start at
//! θ0/4, no smoothing) with 20 EM iterations. Bootstrap SDs came from 30
//! replicates on each of 6 datasets, and the true sampling SD from 40
//! independent datasets. Results, for log10 Ne at 30 time points:
//!
//! - The bootstrap SD was a median 0.85 of the true SD with 500,000-bin
//!   trunks (here, whole sequences) and 0.91 with 1,000-bin trunks. It was
//!   below the true SD at 62% and 63% of time points, fell to 0.78 and 0.74
//!   of it at the oldest times, and ranged from 0.68 to 1.03 between
//!   datasets.
//! - At the most recent times the two trunk sizes differ: whole sequences
//!   gave 1.24 times the true SD, 1,000-bin trunks 0.81 times.
//! - Measured by the interquartile range instead, which a few runaway fits
//!   do not drive, the bootstrap spread was a median 1.01 (whole sequences)
//!   and 1.08 (1,000-bin trunks) of the true one.
//! - The fits' median bias, 0.094, was a third of the true SD (0.280). The
//!   bootstrap cannot see it.
//! - The previous within-sequence scheme (50,000-bin blocks, replicates
//!   started from the full fit) gave an SD of exactly 0, because no sequence
//!   was longer than one block.
//!
//! Read the bands as a lower bound on the uncertainty, especially at the
//! oldest times.

use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::ThreadPool;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::em::{EmHistory, EmOptions, em_train_with_options};
use crate::hmm::Cancelled;
use crate::io::callable::join_counts;
use crate::io::common::chunk_sequences;
use crate::io::{CallableBases, Observations, Sequences};
use crate::model::{NegativeTransitions, PsmcModel, ThetaCounts, initial_theta_with_callable};
use crate::opt::MStepConfig;
use crate::psmc_out::{PsmcRunInfo, psmc_rounds, write_psmc};

/// The bootstrap distribution of Ne at one time point of the main curve.
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapCurvePoint {
    /// Years ago (the start of a time interval of the main fit).
    pub x_years: f64,
    /// Ne of the main fit from this time back to the next point.
    pub ne_main: f64,
    /// 2.5% quantile of the replicates' Ne at this time.
    pub ne_q025: f64,
    /// Median of the replicates' Ne at this time.
    pub ne_q500: f64,
    /// 97.5% quantile of the replicates' Ne at this time.
    pub ne_q975: f64,
}

/// The replicate curves of a bootstrap, summarised at the time points of
/// the main curve ([`summarize_bootstrap_models`]).
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapSummary {
    /// Number of replicates summarised.
    pub n_replicates: usize,
    /// The replicates left out (numbered from 1, as their files), because
    /// the fresh start of their resampled data had negative transition
    /// probabilities ([`BootstrapOptions::skip_invalid_starts`]); empty
    /// from [`summarize_bootstrap_models`], which sees only the fitted
    /// replicates.
    pub skipped_replicates: Vec<usize>,
    /// Trunk size in bins ([`BootstrapConfig::trunk_size`]).
    pub block_size: usize,
    /// Base seed of the resampling.
    pub seed: u64,
    /// One point per positive time of the main curve.
    pub points: Vec<BootstrapCurvePoint>,
}

/// The default directory of a run's bootstrap files: the output JSON's path
/// with `.bootstrap` in place of its extension.
pub fn default_bootstrap_dir(output_json: &Path) -> PathBuf {
    let stem = output_json
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("psmc");
    output_json.with_file_name(format!("{stem}.bootstrap"))
}

/// The parameter file of replicate `replicate_index_1based` in `dir`:
/// `replicate_NNN.json`.
pub fn replicate_json_path(dir: &Path, replicate_index_1based: usize) -> PathBuf {
    dir.join(format!("replicate_{replicate_index_1based:03}.json"))
}

/// Random number generator used for bootstrap resampling.
///
/// ChaCha8 is value-stable: the same seed yields the same stream on every
/// platform and with every `rand_chacha` release, unlike `SmallRng`, whose
/// algorithm depends on the pointer width and may change between `rand`
/// versions.
pub type BootstrapRng = ChaCha8Rng;

const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 output function (Steele, Lea & Flood 2014).
fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Expand a 64-bit seed into a 256-bit ChaCha key: four consecutive outputs of
/// a SplitMix64 generator started at `seed`, little-endian.
fn expand_seed(seed: u64) -> [u8; 32] {
    let mut state = seed;
    let mut key = [0u8; 32];
    for word in key.as_chunks_mut::<8>().0 {
        state = state.wrapping_add(SPLITMIX64_GAMMA);
        *word = splitmix64_mix(state).to_le_bytes();
    }
    key
}

/// Seed of bootstrap replicate `replicate` (0-based): output number
/// `replicate + 1` of a SplitMix64 generator started at `base_seed`.
///
/// Each replicate gets its own stream, so a replicate's data does not depend
/// on how many threads run the replicates or in which order they finish.
pub fn replicate_seed(base_seed: u64, replicate: usize) -> u64 {
    let step = (replicate as u64)
        .wrapping_add(1)
        .wrapping_mul(SPLITMIX64_GAMMA);
    splitmix64_mix(base_seed.wrapping_add(step))
}

/// Portable deterministic generator for `seed` (see [`BootstrapRng`]).
pub fn new_rng(seed: u64) -> BootstrapRng {
    ChaCha8Rng::from_seed(expand_seed(seed))
}

/// Generator of bootstrap replicate `replicate` (0-based) under `base_seed`.
pub fn replicate_rng(base_seed: u64, replicate: usize) -> BootstrapRng {
    new_rng(replicate_seed(base_seed, replicate))
}

/// Uniform index in `0..n` built from `next_u64` alone (Lemire's
/// multiply-and-reject method), so the stream depends only on the generator
/// and not on how a given `rand` release implements `gen_range`.
fn uniform_index<R: Rng + ?Sized>(rng: &mut R, n: usize) -> usize {
    assert!(n > 0, "cannot draw an index from an empty range");
    let range = n as u64;
    let mut m = u128::from(rng.next_u64()) * u128::from(range);
    if (m as u64) < range {
        let threshold = range.wrapping_neg() % range;
        while (m as u64) < threshold {
            m = u128::from(rng.next_u64()) * u128::from(range);
        }
    }
    (m >> 64) as usize
}

/// Default trunk size of [`split_for_bootstrap`], in bins: `splitfa`'s default
/// of 500,000 psmcfa characters (50 Mb at 100 bp per bin).
pub const DEFAULT_BOOTSTRAP_TRUNK: usize = 500_000;

/// Cut sequences into bootstrap trunks exactly as psmc's `utils/splitfa` does.
///
/// Each sequence is cut into consecutive trunks of `trunk_size` bins, except
/// that once fewer than `trunk_size * 3 / 2` bins (integer division, as in
/// `splitfa`) remain from the current offset, the whole remainder becomes the
/// last trunk. A sequence shorter than that limit is therefore kept whole.
/// Empty sequences yield no trunks.
///
/// # Panics
///
/// If `trunk_size` is 0 (`splitfa` would loop forever).
pub fn split_for_bootstrap(seqs: &[Vec<u8>], trunk_size: usize) -> Vec<Vec<u8>> {
    assert!(trunk_size > 0, "bootstrap trunk_size must be > 0");
    split_trunks(seqs, trunk_size)
}

/// [`split_for_bootstrap`] for sequences with the callable bases of their
/// bins: the counts are cut at the same bins as the symbols, so each trunk
/// keeps the counts of its bins. The sequences must be valid
/// ([`Sequences::validate`]).
///
/// # Panics
///
/// If `trunk_size` is 0.
pub fn split_sequences_for_bootstrap(seqs: &Sequences, trunk_size: usize) -> Sequences {
    assert!(trunk_size > 0, "bootstrap trunk_size must be > 0");
    debug_assert!(seqs.validate().is_ok(), "invalid bootstrap input");
    Sequences {
        bins: split_trunks(&seqs.bins, trunk_size),
        callable: seqs.callable.as_ref().map(|c| split_counts(c, trunk_size)),
    }
}

/// The trunks of [`split_for_bootstrap`].
fn split_trunks(seqs: &[Vec<u8>], trunk_size: usize) -> Vec<Vec<u8>> {
    seqs.iter()
        .flat_map(|seq| trunk_ranges(seq.len(), trunk_size).map(|r| seq[r].to_vec()))
        .collect()
}

/// The callable bases of the trunks of [`split_for_bootstrap`], cut from
/// those of the sequences.
fn split_counts(callable: &CallableBases, trunk_size: usize) -> CallableBases {
    let trunks = callable
        .rows()
        .iter()
        .flat_map(|seq| trunk_ranges(seq.len(), trunk_size).map(|r| seq.slice(r)))
        .collect();
    callable.with_rows(trunks)
}

/// Bin ranges of the trunks [`split_for_bootstrap`] cuts a sequence of `len`
/// bins into.
fn trunk_ranges(len: usize, trunk_size: usize) -> impl Iterator<Item = Range<usize>> {
    let tail_limit = trunk_size.saturating_mul(3) / 2;
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= len {
            return None;
        }
        // tail_limit >= trunk_size, so a full trunk always fits otherwise.
        let end = if len - start < tail_limit {
            len
        } else {
            start + trunk_size
        };
        let range = start..end;
        start = end;
        Some(range)
    })
}

/// Number of trunks [`split_for_bootstrap`] cuts `seqs` into, checked to give
/// a usable bootstrap. [`run_bootstrap`] runs this check first; a caller can
/// run it before a long main fit to fail early or to report the count.
///
/// # Errors
///
/// If `trunk_size` is 0, if `seqs` hold no bins, or if they form a single
/// trunk (one sequence shorter than 1.5 trunks). With one trunk every
/// replicate redraws exactly the input, so the bootstrap spread would be zero
/// whatever the data; C psmc runs such a bootstrap anyway, this refuses it.
pub fn bootstrap_trunk_count(seqs: &[Vec<u8>], trunk_size: usize) -> Result<usize> {
    if trunk_size == 0 {
        bail!("bootstrap trunk_size must be > 0");
    }
    let count: usize = seqs
        .iter()
        .map(|s| trunk_ranges(s.len(), trunk_size).count())
        .sum();
    match count {
        0 => bail!("bootstrap input has no data"),
        1 => {
            let bins: usize = seqs.iter().map(Vec::len).sum();
            bail!(
                "bootstrap input of {bins} bins forms a single trunk at trunk size {trunk_size}, \
                 so every replicate would refit the same data; use a trunk size of at most {} \
                 bins (a few dozen trunks or more give smoother bands)",
                (bins * 2 / 3).max(1)
            )
        }
        n => Ok(n),
    }
}

/// Indices of the trunks drawn by one psmc `-b` resampling (`psmc_resamp` in
/// psmc's `aux.c`), given the trunk lengths.
///
/// `pick(n)` must return a uniform random index in `0..n`; it is called once
/// per draw. A drawn trunk is kept if the running total stays at or below the
/// original total, or if it overshoots by less than the deficit before the
/// draw. Drawing stops at the first draw that would reach or pass the original
/// total (whether or not that trunk was kept), so the resampled total differs
/// from the original by at most half the longest trunk. The same trunk can be
/// drawn more than once.
pub fn resample_chunk_indices(
    lengths: &[usize],
    mut pick: impl FnMut(usize) -> usize,
) -> Vec<usize> {
    if lengths.is_empty() {
        return Vec::new();
    }
    let target: i128 = lengths.iter().map(|&l| l as i128).sum();
    let mut total: i128 = 0;
    let mut drawn = Vec::new();
    loop {
        let idx = pick(lengths.len());
        let len = lengths[idx] as i128;
        // `tmp1` and `tmp2` in psmc_resamp.
        let deficit = target - total;
        let overshoot = total + len - target;
        if overshoot <= 0 || (deficit > 0 && overshoot < deficit) {
            drawn.push(idx);
            total += len;
        }
        if deficit >= 0 && overshoot >= 0 {
            break;
        }
    }
    drawn
}

/// One bootstrap resample of `chunks` (normally the output of
/// [`split_for_bootstrap`]) by psmc's `psmc_resamp` rule; see
/// [`resample_chunk_indices`].
///
/// Each returned chunk is meant to be an independent sequence (its own HMM
/// chain, `row_starts` true), as each `splitfa` record is in C psmc.
pub fn resample_chunks(chunks: &[Vec<u8>], rng: &mut impl Rng) -> Vec<Vec<u8>> {
    draw_trunks(chunks, rng)
        .into_iter()
        .map(|i| chunks[i].clone())
        .collect()
}

/// [`resample_chunks`] for trunks with the callable bases of their bins
/// (normally the output of [`split_sequences_for_bootstrap`]): the same
/// trunks are drawn from the same random numbers, each with its counts.
pub fn resample_sequences(trunks: &Sequences, rng: &mut impl Rng) -> Sequences {
    let drawn = draw_trunks(&trunks.bins, rng);
    Sequences {
        bins: drawn.iter().map(|&i| trunks.bins[i].clone()).collect(),
        callable: trunks.callable.as_ref().map(|c| {
            let rows = c.rows();
            c.with_rows(drawn.iter().map(|&i| rows[i].clone()).collect())
        }),
    }
}

/// The indices of the trunks one `psmc_resamp` draws from `chunks`
/// ([`resample_chunk_indices`]).
fn draw_trunks(chunks: &[Vec<u8>], rng: &mut impl Rng) -> Vec<usize> {
    let lengths: Vec<usize> = chunks.iter().map(Vec::len).collect();
    resample_chunk_indices(&lengths, |n| uniform_index(rng, n))
}

/// Join HMM rows back into whole sequences (`row_starts` marks the rows
/// that start one). Errors on empty or inconsistent input.
pub fn rows_to_sequences(rows: &[Vec<u8>], row_starts: &[bool]) -> Result<Vec<Vec<u8>>> {
    Ok(checked_sequence_rows(rows, row_starts)?
        .into_iter()
        .map(|r| rows[r].concat())
        .collect())
}

/// The whole sequences of `obs`, joined as [`rows_to_sequences`] joins
/// them, with the callable bases of their bins joined the same way.
pub fn observations_to_sequences(obs: &Observations) -> Result<Sequences> {
    obs.validate()?;
    let seqs = checked_sequence_rows(&obs.rows, &obs.row_starts)?;
    let bins = seqs.iter().map(|r| obs.rows[r.clone()].concat()).collect();
    let callable = obs.callable.as_ref().map(|c| {
        let rows = seqs
            .iter()
            .map(|r| join_counts(c.rows(), r.clone()).into_owned())
            .collect();
        c.with_rows(rows)
    });
    Ok(Sequences { bins, callable })
}

/// The rows of each sequence that [`rows_to_sequences`] joins, as ranges
/// of row indices. Errors on empty or inconsistent input.
fn checked_sequence_rows(rows: &[Vec<u8>], row_starts: &[bool]) -> Result<Vec<Range<usize>>> {
    if rows.is_empty() {
        bail!("cannot build sequences from empty rows");
    }
    if row_starts.len() != rows.len() {
        bail!(
            "row_starts length {} does not match rows {}",
            row_starts.len(),
            rows.len()
        );
    }
    if !row_starts[0] {
        bail!("row_starts[0] must be true");
    }

    let mut seqs = Vec::<Range<usize>>::new();
    for (i, row) in rows.iter().enumerate() {
        if row.is_empty() {
            bail!("row {i} is empty");
        }
        if row_starts[i] {
            seqs.push(i..i + 1);
        } else {
            match seqs.last_mut() {
                Some(seq) => seq.end = i + 1,
                None => bail!("row {i} continues a sequence but no previous sequence exists"),
            }
        }
    }
    if seqs.is_empty() {
        bail!("no sequences reconstructed from rows");
    }
    Ok(seqs)
}

/// Cut sequences into HMM rows of at most `batch_size` bins (whole
/// sequences without it), the rows of each sequence chained in one HMM
/// sequence; empty sequences are skipped. The rows have no callable bases;
/// [`chunk_sequences`] cuts a [`Sequences`] with them.
pub fn sequences_to_observations(
    seqs: &[Vec<u8>],
    batch_size: Option<usize>,
) -> Result<Observations> {
    if seqs.is_empty() {
        bail!("sequences are empty");
    }
    chunk_sequences(seqs.to_vec(), batch_size, "no rows produced from sequences")
}

/// Legacy within-sequence block bootstrap: each sequence is cut into blocks of
/// `block_size` bins and rebuilt from blocks of that same sequence.
///
/// Superseded by [`split_for_bootstrap`] + [`resample_chunks`], which follow
/// C psmc. Here a sequence no longer than one block has a single block and
/// comes back unchanged, so inputs made of short sequences or scaffolds get
/// little or no resampling variance.
pub fn sample_sequences_block_bootstrap(
    seqs: &[Vec<u8>],
    block_size: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Vec<u8>>> {
    if block_size == 0 {
        bail!("bootstrap block_size must be > 0");
    }
    if seqs.is_empty() {
        bail!("input sequences are empty");
    }

    let mut out = Vec::<Vec<u8>>::with_capacity(seqs.len());
    for (seq_id, seq) in seqs.iter().enumerate() {
        if seq.is_empty() {
            bail!("sequence {seq_id} is empty");
        }
        let blocks: Vec<&[u8]> = seq.chunks(block_size).collect();
        if blocks.is_empty() {
            bail!("sequence {seq_id} produced no bootstrap blocks");
        }
        let draw_n = seq.len().div_ceil(block_size);
        let mut sampled = Vec::<u8>::with_capacity(seq.len() + block_size);
        for _ in 0..draw_n {
            let idx = uniform_index(rng, blocks.len());
            sampled.extend_from_slice(blocks[idx]);
        }
        sampled.truncate(seq.len());
        out.push(sampled);
    }
    Ok(out)
}

/// Fresh starting model for one bootstrap replicate, built the way C
/// `psmc -b` initialises every run: the template's pattern, `n_steps` and
/// `mu`; the configured initial `t_max` (not a fitted one); θ0 =
/// [`initial_theta`](crate::initial_theta) of the replicate's own rows; ρ0 =
/// θ0 / `tr_ratio`; every λ = 1.
///
/// Starting from the full-data fit instead would not re-run the estimator
/// whose sampling variability the bootstrap is meant to measure.
pub fn replicate_initial_model(
    template: &PsmcModel,
    rows: &[Vec<u8>],
    t_max: f64,
    tr_ratio: f64,
) -> Result<PsmcModel> {
    replicate_initial_model_with_callable(template, rows, None, t_max, tr_ratio)
}

/// [`replicate_initial_model`] for rows with the callable bases `callable`:
/// θ0 is [`initial_theta_with_callable`] of the replicate's own rows and
/// counts, the starting value the main fit of such data takes.
pub fn replicate_initial_model_with_callable(
    template: &PsmcModel,
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
    t_max: f64,
    tr_ratio: f64,
) -> Result<PsmcModel> {
    check_fresh_start(t_max, tr_ratio)?;
    let theta0 = initial_theta_with_callable(rows, callable)?;
    fresh_start(template, theta0, t_max, tr_ratio)
}

/// [`replicate_initial_model`] at θ0 = `theta0`.
fn fresh_start(template: &PsmcModel, theta0: f64, t_max: f64, tr_ratio: f64) -> Result<PsmcModel> {
    PsmcModel::new(
        t_max,
        template.n_steps,
        theta0,
        theta0 / tr_ratio,
        template.mu,
        template.pattern.clone(),
    )
}

/// A bootstrap replicate whose fresh start has negative transition
/// probabilities ([`invalid_replicate_starts`]).
#[derive(Debug, Clone, PartialEq)]
pub struct InvalidReplicateStart {
    /// The replicate, 0-based as for [`replicate_rng`] (its files are
    /// numbered `index + 1`).
    pub index: usize,
    /// Its start: θ0 of its resampled data, ρ0 = θ0 / `tr_ratio`, the limit
    /// on ρ and the smallest θ/ρ ratio under it.
    pub start: NegativeTransitions,
}

/// The replicates of `cfg` whose fresh start ([`replicate_initial_model`]
/// on the resampled data) has negative transition probabilities, so that
/// EM cannot fit them, in replicate order.
///
/// The resampling is redone as [`run_bootstrap`] does it, from trunk
/// counts alone and without copying the data, so this is cheap enough to
/// run before a long main fit. It finds exactly the replicates that
/// [`BootstrapOptions::skip_invalid_starts`] leaves out. It applies to
/// fresh starts only: replicates that start from saved parameters
/// ([`BootstrapOptions::start`]) all start where the main fit does.
///
/// # Errors
///
/// As [`run_bootstrap`] for unusable settings (see
/// [`bootstrap_trunk_count`]), and for a replicate whose resampled data
/// give no θ0 (no non-missing bin, or every one heterozygous), which
/// [`run_bootstrap`] would fail on too.
pub fn invalid_replicate_starts(
    seqs: &[Vec<u8>],
    template: &PsmcModel,
    cfg: &BootstrapConfig,
) -> Result<Vec<InvalidReplicateStart>> {
    invalid_starts(seqs, None, template, cfg)
}

/// [`invalid_replicate_starts`] for sequences with the callable bases of
/// their bins, whose replicates start from θ0 of their counts too
/// ([`replicate_initial_model_with_callable`]), as
/// [`run_bootstrap_sequences`] fits them: it finds exactly the replicates
/// that bootstrap skips.
pub fn invalid_replicate_starts_sequences(
    seqs: &Sequences,
    template: &PsmcModel,
    cfg: &BootstrapConfig,
) -> Result<Vec<InvalidReplicateStart>> {
    seqs.validate()?;
    invalid_starts(&seqs.bins, seqs.callable.as_ref(), template, cfg)
}

/// [`invalid_replicate_starts_sequences`] on the bins `seqs` and their
/// callable bases.
fn invalid_starts(
    seqs: &[Vec<u8>],
    callable: Option<&CallableBases>,
    template: &PsmcModel,
    cfg: &BootstrapConfig,
) -> Result<Vec<InvalidReplicateStart>> {
    bootstrap_trunk_count(seqs, cfg.trunk_size)?;
    check_fresh_start(cfg.t_max, cfg.tr_ratio)?;
    let mut lengths = Vec::new();
    let mut counts = Vec::new();
    for (i, seq) in seqs.iter().enumerate() {
        for range in trunk_ranges(seq.len(), cfg.trunk_size) {
            lengths.push(range.len());
            counts.push(match callable {
                Some(c) => ThetaCounts::of_counted(&seq[range.clone()], &c.rows()[i].slice(range))?,
                None => ThetaCounts::of(&seq[range]),
            });
        }
    }
    let empty = ThetaCounts::new(callable.map(CallableBases::bin_size));
    let mut invalid = Vec::new();
    for index in 0..cfg.replicates {
        let rng = &mut replicate_rng(cfg.seed, index);
        let mut drawn = empty.clone();
        for i in resample_chunk_indices(&lengths, |n| uniform_index(rng, n)) {
            drawn.add(&counts[i]);
        }
        let start = drawn
            .theta0()
            .and_then(|theta0| fresh_start(template, theta0, cfg.t_max, cfg.tr_ratio))
            .with_context(|| format!("bootstrap replicate {}", index + 1))?;
        if let Some(start) = negative_transitions(&start)? {
            invalid.push(InvalidReplicateStart { index, start });
        }
    }
    Ok(invalid)
}

/// The [`NegativeTransitions`] of `model`, if its transition probabilities
/// are negative; other failures of [`PsmcModel::check_transitions`] are
/// errors.
fn negative_transitions(model: &PsmcModel) -> Result<Option<NegativeTransitions>> {
    match model.check_transitions() {
        Ok(()) => Ok(None),
        Err(e) => match e.downcast::<NegativeTransitions>() {
            Ok(negative) => Ok(Some(negative)),
            Err(e) => Err(e),
        },
    }
}

fn check_fresh_start(t_max: f64, tr_ratio: f64) -> Result<()> {
    if !(t_max.is_finite() && t_max > 0.0) {
        bail!("bootstrap t_max must be a positive number, got {t_max}");
    }
    if !(tr_ratio.is_finite() && tr_ratio > 0.0) {
        bail!("bootstrap theta/rho ratio must be a positive number, got {tr_ratio}");
    }
    Ok(())
}

/// Settings for [`run_bootstrap`].
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap replicates.
    pub replicates: usize,
    /// Trunk size in bins for [`split_for_bootstrap`] (`splitfa`'s argument).
    pub trunk_size: usize,
    /// Base seed: replicate `i` (0-based) draws from [`replicate_rng`]`(seed, i)`.
    pub seed: u64,
    /// EM iterations per replicate; C reuses the main run's `-N`.
    pub em_iters: usize,
    /// θ0/ρ0 ratio of each replicate's fresh start (C `-r`). Use the ratio
    /// the main fit started from.
    pub tr_ratio: f64,
    /// Initial `t_max` of each replicate's fresh start (C `-t`): the
    /// configured value, not the main fit's estimate.
    pub t_max: f64,
    /// Optional row length within a trunk (see [`sequences_to_observations`]).
    /// The rows of a trunk stay one HMM chain, and the E-step treats them as
    /// one sequence, so the fits do not depend on it.
    pub batch_size: Option<usize>,
    /// When set, each fitted replicate is saved there as
    /// `replicate_NNN.json` (1-based, see [`replicate_json_path`]); the
    /// directory is created if needed.
    pub out_dir: Option<PathBuf>,
}

impl Default for BootstrapConfig {
    /// C psmc's defaults (`-N30 -r4 -t15`, `splitfa` trunk 500,000) with 100
    /// replicates and seed 42.
    fn default() -> Self {
        Self {
            replicates: 100,
            trunk_size: DEFAULT_BOOTSTRAP_TRUNK,
            seed: 42,
            em_iters: 30,
            tr_ratio: 4.0,
            t_max: 15.0,
            batch_size: None,
            out_dir: None,
        }
    }
}

/// Run a PSMC bootstrap the way C psmc's documented procedure does
/// (`splitfa`, then one `psmc -b` per replicate) and return the fitted
/// replicate models in replicate order.
///
/// `seqs` are the whole input sequences (e.g. from [`rows_to_sequences`]).
/// They must form at least two trunks (see [`bootstrap_trunk_count`]).
/// They are cut once with [`split_for_bootstrap`]; each replicate then draws
/// trunks with [`resample_chunks`] using [`replicate_rng`]`(cfg.seed, i)`,
/// treats every drawn trunk as an independent sequence, starts from
/// [`replicate_initial_model`] and runs `cfg.em_iters` EM iterations with
/// `mstep_cfg` (progress output forced off).
///
/// Replicates run in parallel on the current rayon pool. Each one depends
/// only on its inputs and `(cfg.seed, i)`, and the E-step's results do not
/// depend on threads, row lengths or its cache budget, so results do not
/// depend on the thread count, on scheduling or on `cfg.batch_size`. Each
/// replicate in flight holds its own copy of the resampled data and its own
/// E-step buffers, so peak memory grows with the number of threads.
///
/// `cancel` is checked before each replicate starts and inside its EM; once
/// it is set, running replicates stop, the remaining ones are skipped and a
/// [`Cancelled`] error is returned. `on_done(n)` is called from the worker
/// threads once per finished replicate, with `n` the number finished so
/// far. The calls are serialised, so `n` goes 1, 2, 3, … in call order and
/// the last call of a successful run has `n == cfg.replicates`; other
/// workers wait while it runs, so keep it cheap.
///
/// A replicate whose fresh start has negative transition probabilities is
/// an error here, as for EM (see [`invalid_replicate_starts`]);
/// [`BootstrapOptions::skip_invalid_starts`] leaves such replicates out
/// instead.
///
/// [`run_bootstrap_with_options`] also returns each replicate's EM history
/// and data counts.
pub fn run_bootstrap(
    seqs: &[Vec<u8>],
    template: &PsmcModel,
    cfg: &BootstrapConfig,
    mstep_cfg: &MStepConfig,
    cancel: Option<&AtomicBool>,
    on_done: &(dyn Fn(usize) + Sync),
) -> Result<Vec<PsmcModel>> {
    let opts = BootstrapOptions {
        cancel,
        on_done: Some(on_done),
        ..BootstrapOptions::default()
    };
    let replicates = run_bootstrap_with_options(seqs, template, cfg, mstep_cfg, opts)?;
    Ok(replicates.into_iter().map(|r| r.model).collect())
}

/// Options for [`run_bootstrap_with_options`].
#[derive(Default)]
pub struct BootstrapOptions<'a> {
    /// Checked before each replicate starts and inside its EM (see
    /// [`EmOptions::cancel`]); once set, the bootstrap stops with a
    /// [`Cancelled`] error.
    pub cancel: Option<&'a AtomicBool>,
    /// Run the replicates in this pool instead of the current one.
    pub thread_pool: Option<&'a ThreadPool>,
    /// Called once per finished replicate with the number finished so far,
    /// as `on_done` of [`run_bootstrap`].
    pub on_done: Option<&'a (dyn Fn(usize) + Sync)>,
    /// Start every replicate from these parameters instead of the fresh
    /// start of [`replicate_initial_model`] (`cfg.t_max` and `cfg.tr_ratio`
    /// are then unused). This is what C psmc does when `-i` loads saved
    /// parameters: every run, `-b` ones included, starts from them. Its
    /// layout (`n_steps` and free lambdas) must be the template's.
    ///
    /// Parameters fitted to the whole input already hold what it says, so
    /// replicates that run few iterations from them stay close to them, and
    /// the bands understate the uncertainty more than those of a fresh start.
    pub start: Option<&'a PsmcModel>,
    /// Leave out the replicates whose fresh start has negative transition
    /// probabilities (exactly those of [`invalid_replicate_starts`]) instead
    /// of failing: they are not fitted, no file is written for them, they
    /// count as done for `on_done`, and the result lacks them. Without it,
    /// such a replicate stops the bootstrap with EM's error.
    ///
    /// These are the resamples more heterozygous than the others, so
    /// leaving them out can narrow the bands; report how many were left
    /// out. Saved starting parameters ([`BootstrapOptions::start`]) are the
    /// same for every replicate and never skipped.
    pub skip_invalid_starts: bool,
}

/// The size of a replicate's resampled data, as C psmc's `MM` header line
/// of its `.psmc` file reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplicateData {
    /// Trunks drawn (each one a sequence of the replicate).
    pub n_seqs: usize,
    /// Bins over all drawn trunks.
    pub bins: u64,
    /// Non-missing bins (C's `sum_L`).
    pub sum_l: u64,
    /// Heterozygous bins (C's `sum_n`).
    pub sum_n: u64,
}

impl ReplicateData {
    fn of(seqs: &[Vec<u8>]) -> Self {
        let mut data = Self {
            n_seqs: seqs.len(),
            bins: 0,
            sum_l: 0,
            sum_n: 0,
        };
        for &bin in seqs.iter().flatten() {
            data.bins += 1;
            if bin < 2 {
                data.sum_l += 1;
                data.sum_n += u64::from(bin);
            }
        }
        data
    }
}

/// One fitted bootstrap replicate.
#[derive(Debug)]
pub struct BootstrapReplicate {
    /// Which replicate it is, 0-based as for [`replicate_rng`]; its files
    /// are numbered `index + 1` ([`replicate_json_path`]).
    pub index: usize,
    /// The fitted model, with its matrices recalculated.
    pub model: PsmcModel,
    /// Its EM history, from its starting parameters on.
    pub history: EmHistory,
    /// Its resampled data.
    pub data: ReplicateData,
}

impl BootstrapReplicate {
    /// Write the replicate's EM run as C psmc's `.psmc` output
    /// ([`crate::psmc_out`]), as `psmc -b` would have written it.
    pub fn write_psmc<W: Write>(&self, w: W) -> Result<()> {
        let d = &self.data;
        let info =
            PsmcRunInfo::from_counts(&self.model, &self.history, d.n_seqs, d.sum_l, d.sum_n)?;
        let rounds = psmc_rounds(&self.model, &self.history, d.sum_l)?;
        write_psmc(w, &info, &rounds)?;
        Ok(())
    }
}

/// [`run_bootstrap`] with a thread pool and, optionally, saved starting
/// parameters ([`BootstrapOptions::start`]) or skipped invalid starts
/// ([`BootstrapOptions::skip_invalid_starts`]), returning each fitted
/// replicate's number, model, EM history and data counts (for its `.psmc`
/// output), in replicate order.
pub fn run_bootstrap_with_options(
    seqs: &[Vec<u8>],
    template: &PsmcModel,
    cfg: &BootstrapConfig,
    mstep_cfg: &MStepConfig,
    opts: BootstrapOptions<'_>,
) -> Result<Vec<BootstrapReplicate>> {
    run_replicates(seqs, None, template, cfg, mstep_cfg, opts)
}

/// [`run_bootstrap_with_options`] on sequences with the callable bases of
/// their bins: the trunks are cut with [`split_sequences_for_bootstrap`] and
/// drawn with [`resample_sequences`], so that each replicate's rows carry
/// the counts of their bins ([`Observations::callable`]), and each
/// replicate is fitted with them ([`EmOptions::callable`]), from a fresh
/// start at θ0 of its own counts
/// ([`replicate_initial_model_with_callable`]). A replicate draws the same
/// trunks as from the bins alone; with every called bin full, it is the
/// replicate of the bins alone, to the bit.
pub fn run_bootstrap_sequences(
    seqs: &Sequences,
    template: &PsmcModel,
    cfg: &BootstrapConfig,
    mstep_cfg: &MStepConfig,
    opts: BootstrapOptions<'_>,
) -> Result<Vec<BootstrapReplicate>> {
    seqs.validate()?;
    run_replicates(
        &seqs.bins,
        seqs.callable.as_ref(),
        template,
        cfg,
        mstep_cfg,
        opts,
    )
}

/// [`run_bootstrap_sequences`] on the bins `seqs` and their callable bases.
fn run_replicates(
    seqs: &[Vec<u8>],
    callable: Option<&CallableBases>,
    template: &PsmcModel,
    cfg: &BootstrapConfig,
    mstep_cfg: &MStepConfig,
    opts: BootstrapOptions<'_>,
) -> Result<Vec<BootstrapReplicate>> {
    if cfg.replicates == 0 {
        return Ok(Vec::new());
    }
    bootstrap_trunk_count(seqs, cfg.trunk_size)?;
    if cfg.batch_size == Some(0) {
        bail!("batch_size must be > 0");
    }
    let BootstrapOptions {
        cancel,
        thread_pool,
        on_done,
        start,
        skip_invalid_starts,
    } = opts;
    match start {
        Some(start) => {
            if (start.n_steps, start.lam.len()) != (template.n_steps, template.lam.len()) {
                bail!(
                    "the bootstrap start has n_steps {} and {} free lambdas, the template {} and \
                     {}",
                    start.n_steps,
                    start.lam.len(),
                    template.n_steps,
                    template.lam.len()
                );
            }
        }
        None => check_fresh_start(cfg.t_max, cfg.tr_ratio)?,
    }
    let chunks = Sequences {
        bins: split_for_bootstrap(seqs, cfg.trunk_size),
        callable: callable.map(|c| split_counts(c, cfg.trunk_size)),
    };
    if let Some(dir) = &cfg.out_dir {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create bootstrap directory {dir:?}"))?;
    }

    let mut rep_cfg = mstep_cfg.clone();
    rep_cfg.progress = false;
    let is_cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
    // Count and report under one lock so calls arrive in order: with a
    // separate atomic counter, the worker holding count n + 1 could report
    // before the one holding n, and a progress display would step backwards.
    let finished = Mutex::new(0usize);

    // `None` for a replicate left out (`skip_invalid_starts`).
    let fit_replicate = |rep: usize| -> Result<Option<BootstrapReplicate>> {
        if is_cancelled() {
            return Err(Cancelled.into());
        }
        let fit = || -> Result<Option<BootstrapReplicate>> {
            let sampled = resample_sequences(&chunks, &mut replicate_rng(cfg.seed, rep));
            let data = ReplicateData::of(&sampled.bins);
            let obs = chunk_sequences(sampled, cfg.batch_size, "resampled data is empty")?;
            let mut model = match start {
                Some(start) => start.clone(),
                None => {
                    let model = replicate_initial_model_with_callable(
                        template,
                        &obs.rows,
                        obs.callable.as_ref(),
                        cfg.t_max,
                        cfg.tr_ratio,
                    )?;
                    if skip_invalid_starts && negative_transitions(&model)?.is_some() {
                        return Ok(None);
                    }
                    model
                }
            };
            let em_opts = EmOptions {
                on_progress: None,
                cancel,
                thread_pool,
                callable: obs.callable.as_ref(),
            };
            let history = em_train_with_options(
                &mut model,
                &obs.rows,
                &obs.row_starts,
                cfg.em_iters,
                &rep_cfg,
                em_opts,
            )?;
            model.param_recalculate()?;
            if let Some(dir) = &cfg.out_dir {
                model.save_params(&replicate_json_path(dir, rep + 1))?;
            }
            Ok(Some(BootstrapReplicate {
                index: rep,
                model,
                history,
                data,
            }))
        };
        let replicate = fit().with_context(|| format!("bootstrap replicate {}", rep + 1))?;
        let mut n = finished.lock().unwrap_or_else(PoisonError::into_inner);
        *n += 1;
        if let Some(on_done) = on_done {
            on_done(*n);
        }
        Ok(replicate)
    };
    let run = || {
        (0..cfg.replicates)
            .into_par_iter()
            .map(fit_replicate)
            .collect::<Result<Vec<_>>>()
    };
    let replicates = match thread_pool {
        Some(pool) => pool.install(run),
        None => run(),
    }?;
    Ok(replicates.into_iter().flatten().collect())
}

/// The points of [`PsmcModel::ne_curve`] with finite coordinates, a positive
/// size and a time that `keep` accepts, as x and y columns.
fn curve_points(
    model: &PsmcModel,
    bin_size: f64,
    gen_years: f64,
    keep: impl Fn(f64) -> bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (xs, ys): (Vec<f64>, Vec<f64>) = model
        .ne_curve(bin_size, gen_years)?
        .into_iter()
        .filter(|&[x, y]| x.is_finite() && y.is_finite() && keep(x) && y > 0.0)
        .map(|[x, y]| (x, y))
        .unzip();
    if xs.is_empty() {
        bail!("model curve has no valid points");
    }
    Ok((xs, ys))
}

/// The points of [`PsmcModel::ne_curve`] that a log-scale plot can show
/// (positive and finite), as x and y columns.
fn model_curve(model: &PsmcModel, bin_size: f64, gen_years: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    curve_points(model, bin_size, gen_years, |x| x > 0.0)
}

/// [`PsmcModel::ne_curve`] as a step function for [`step_lookup`]: like
/// [`model_curve`], but keeping the first point (x = 0), which carries the
/// size of state 0. Without it, a time before the model's own first
/// positive point would read the size of state 1, which differs from state
/// 0's whenever the two have separate λ (no pattern, or a first pattern
/// group of one interval).
fn step_curve(model: &PsmcModel, bin_size: f64, gen_years: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    curve_points(model, bin_size, gen_years, |x| x >= 0.0)
}

/// The value at `x` of the step function that is `ys[k]` from `xs[k]` up to
/// `xs[k + 1]` (and `ys[0]` before `xs[0]`).
#[inline]
fn step_lookup(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert!(!xs.is_empty() && xs.len() == ys.len());
    let pos = xs.partition_point(|v| *v <= x);
    ys[pos.max(1) - 1]
}

fn quantile(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let qq = q.clamp(0.0, 1.0);
    let n = vals.len();
    if n == 1 {
        return vals[0];
    }
    let pos = qq * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        vals[lo]
    } else {
        let w = pos - lo as f64;
        vals[lo] * (1.0 - w) + vals[hi] * w
    }
}

/// The 2.5%, 50% and 97.5% quantiles of the replicates' Ne at each
/// positive time point of the main curve ([`PsmcModel::ne_curve`] at
/// `bin_size` and `gen_years`); each replicate's curve is read as a step
/// function from its own time 0, where its state 0 begins. `block_size` and
/// `seed` are recorded as given.
pub fn summarize_bootstrap_models(
    main_model: &PsmcModel,
    bootstrap_models: &[PsmcModel],
    bin_size: f64,
    gen_years: f64,
    block_size: usize,
    seed: u64,
) -> Result<BootstrapSummary> {
    if bootstrap_models.is_empty() {
        bail!("bootstrap model list is empty");
    }
    let (x_main, y_main) = model_curve(main_model, bin_size, gen_years)?;
    let n_points = x_main.len();
    let n_boot = bootstrap_models.len();

    let mut sample_mat = vec![vec![0.0f64; n_boot]; n_points];
    for (r, model) in bootstrap_models.iter().enumerate() {
        let (x_rep, y_rep) = step_curve(model, bin_size, gen_years)?;
        for i in 0..n_points {
            sample_mat[i][r] = step_lookup(&x_rep, &y_rep, x_main[i]);
        }
    }

    let mut points = Vec::<BootstrapCurvePoint>::with_capacity(n_points);
    for i in 0..n_points {
        let mut qbuf = sample_mat[i].clone();
        let q025 = quantile(&mut qbuf, 0.025);
        let q500 = quantile(&mut qbuf, 0.5);
        let q975 = quantile(&mut qbuf, 0.975);
        points.push(BootstrapCurvePoint {
            x_years: x_main[i],
            ne_main: y_main[i],
            ne_q025: q025,
            ne_q500: q500,
            ne_q975: q975,
        });
    }

    Ok(BootstrapSummary {
        n_replicates: n_boot,
        skipped_replicates: Vec::new(),
        block_size,
        seed,
        points,
    })
}

/// Write the summary's points as TSV: `x_years`, `ne_main` and the three
/// quantiles, in `%.8e`.
pub fn write_bootstrap_summary_tsv(path: &Path, summary: &BootstrapSummary) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create bootstrap summary dir {parent:?}"))?;
    }
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let mut w = BufWriter::new(file);
    w.write_all(b"x_years\tne_main\tne_q025\tne_q500\tne_q975\n")?;
    for p in &summary.points {
        writeln!(
            w,
            "{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}",
            p.x_years, p.ne_main, p.ne_q025, p.ne_q500, p.ne_q975
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Write the summary as pretty-printed JSON.
pub fn write_bootstrap_summary_json(path: &Path, summary: &BootstrapSummary) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create bootstrap summary dir {parent:?}"))?;
    }
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, summary)
        .with_context(|| format!("failed to write {path:?}"))?;
    Ok(())
}
