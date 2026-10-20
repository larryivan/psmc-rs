//! The JSON file of fitted parameters that `psmc-rs` writes as its main
//! output, and reads back with `--init-params`.
//!
//! The model fields (`theta` to `pattern`) are those of psmc-rs 0.1. From
//! 0.2 on, the main output also records how the parameters were obtained
//! ([`RunMetadata`]); every one of those fields is optional, so files of
//! 0.1 and bootstrap replicate files, which hold the model only, still load.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::io::CallableSummary;
use crate::io::input::{Dataset, SourceSummary};
use crate::qc::QcSummary;

/// A saved model, with what produced it when known.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsmcParamsFile {
    /// θ per bin.
    pub theta: f64,
    /// ρ per bin.
    pub rho: f64,
    /// The free λ (one per pattern group, or one per state without a
    /// pattern).
    pub lam: Vec<f64>,
    /// `t_max`, in units of 2N0 generations.
    pub t_max: f64,
    /// C psmc's `n`: the model has `n_steps + 1` states.
    pub n_steps: usize,
    /// Mutation rate per base pair per generation used for scaling.
    pub mu: f64,
    /// The parameter pattern, e.g. `"4+5*3+4"`.
    pub pattern: Option<String>,
    /// How the parameters were obtained (absent in files of psmc-rs 0.1
    /// and in bootstrap replicate files).
    #[serde(flatten)]
    pub run: RunMetadata,
}

/// How a saved model was obtained. Every field is optional and omitted from
/// the JSON when unknown.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Version of psmc-rs that wrote the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// The kernels the HMM passes ran ([`crate::hmm::SimdPath::name`]):
    /// `avx2+fma`, `portable` or `neon`. On x86-64 the portable kernels run
    /// about half as fast as the AVX2 and FMA ones, and their results
    /// differ in the last digits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub simd: Option<String>,
    /// EM iterations run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_iter: Option<usize>,
    /// Log-likelihood of these parameters (after 0 iterations, the starting
    /// parameters these are).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loglike: Option<f64>,
    /// `loglike[i]`: log-likelihood of the parameters after `i` EM
    /// iterations, from the starting parameters (`loglike[0]`) to these
    /// (`loglike[n_iter]`, equal to `final_loglike`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglike: Option<Vec<f64>>,
    /// Bases per bin, used for binning (VCF, multihetsep) and scaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_size: Option<usize>,
    /// Years per generation used for time scaling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_years: Option<f64>,
    /// θ0/ρ0 of the starting parameters (C psmc's `-r`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tr_ratio: Option<f64>,
    /// Weight of the M-step's smoothness penalty on log λ.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_lambda: Option<f64>,
    /// What the input held.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<InputSummary>,
    /// Input QC: finding codes and key metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qc: Option<QcSummary>,
    /// How the fit used the callable bases of each bin (absent in files of
    /// psmc-rs before 0.2's callable-length model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_callable: Option<BinCallableRecord>,
}

/// The emission model of a fit that weighs each called bin by its callable
/// bases (`docs/design/bin-callable-model.md`).
pub const CALLABLE_LENGTH_MODEL: &str = "callable-length";

/// The emission model of C psmc, in which every called bin holds the bin
/// size in callable bases.
pub const FULL_BINS_MODEL: &str = "full-bins";

/// How a fit used the callable bases of each bin (`--bin-callable`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinCallableRecord {
    /// `--bin-callable`: `auto` or `ignore`.
    pub mode: String,
    /// The emission model fitted: [`CALLABLE_LENGTH_MODEL`] (each called
    /// bin by its callable bases) or [`FULL_BINS_MODEL`] (C psmc's).
    pub model: String,
    /// How the input's counts were obtained: `exact` (VCF) or `approximate`
    /// (multihetsep, whose uncalled sites are placed at the start of each
    /// interval); absent when the input has none (psmcfa, or bins of more
    /// than [`crate::io::MAX_COUNTED_BIN_SIZE`] bases).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counts: Option<String>,
    /// What the counts hold, when the input has them, whether or not the fit
    /// used them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub callable: Option<CallableSummary>,
}

impl BinCallableRecord {
    /// The text of the `MM` line C psmc's `.psmc` output gets for a fit of
    /// input with counts ([`crate::psmc_out::PsmcRunInfo::bin_callable`]),
    /// such as `bin_model:callable-length, counts:exact, mean_callable:0.701234,
    /// partial_bins:0.850000`; `None` without counts, whose output is C's.
    pub fn psmc_line(&self) -> Option<String> {
        let (counts, c) = (self.counts.as_ref()?, self.callable.as_ref()?);
        Some(crate::psmc_out::bin_callable_line(
            &self.model,
            Some(counts),
            c,
        ))
    }
}

/// The size of the analysed input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSummary {
    /// `psmcfa`, `mhs` or `vcf`.
    pub format: String,
    /// For VCF input, the mode it was read in (`consensus` or `mask`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vcf_mode: Option<String>,
    /// Sequences analysed.
    pub sequences: usize,
    /// Bins analysed.
    pub bins: u64,
    /// Heterozygous (`K`) bins.
    pub het_bins: u64,
    /// Missing (`N`) bins.
    pub missing_bins: u64,
    /// Input sequences left out (empty, or dropped by the output filter).
    pub dropped_sequences: usize,
}

impl InputSummary {
    /// The summary of a loaded dataset.
    pub fn of(data: &Dataset) -> Self {
        let mut counts = [0u64; 3];
        for &b in data.obs.rows.iter().flatten() {
            counts[usize::from(b.min(2))] += 1;
        }
        let vcf_mode = match &data.source {
            SourceSummary::Vcf(v) => Some(v.mode.mode.name().to_string()),
            _ => None,
        };
        Self {
            format: data.source.format().name().to_string(),
            vcf_mode,
            sequences: data.seqs.len(),
            bins: counts.iter().sum(),
            het_bins: counts[1],
            missing_bins: counts[2],
            dropped_sequences: data.dropped.len(),
        }
    }
}

/// Write `params` as pretty-printed JSON.
pub fn save_params(path: &Path, params: &PsmcParamsFile) -> Result<()> {
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, params)
        .with_context(|| format!("failed to write {path:?}"))?;
    writer
        .flush()
        .with_context(|| format!("failed to write {path:?}"))?;
    Ok(())
}

/// Read a file written by [`save_params`] (by any psmc-rs version).
pub fn load_params(path: &Path) -> Result<PsmcParamsFile> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse {path:?}"))
}
