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
    let file = File::create(path).with_context(|| format!("failed to create {:?}", path))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, params)
        .with_context(|| format!("failed to write {:?}", path))?;
    writer
        .flush()
        .with_context(|| format!("failed to write {:?}", path))?;
    Ok(())
}

/// Read a file written by [`save_params`] (by any psmc-rs version).
pub fn load_params(path: &Path) -> Result<PsmcParamsFile> {
    let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let params =
        serde_json::from_reader(reader).with_context(|| format!("failed to parse {:?}", path))?;
    Ok(params)
}
