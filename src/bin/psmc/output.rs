//! The files a run writes besides the HTML report: the parameters with
//! their run metadata, C psmc's `.psmc` output, the bootstrap files and the
//! QC JSON.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapReplicate, BootstrapSummary, replicate_json_path, write_bootstrap_summary_json,
    write_bootstrap_summary_tsv,
};
use psmc_rs::em::EmHistory;
use psmc_rs::io::input::{Dataset, SourceSummary};
use psmc_rs::io::params::{InputSummary, RunMetadata, save_params};
use psmc_rs::psmc_out::{PsmcRunInfo, psmc_rounds, write_psmc};
use psmc_rs::qc::QcReport;

/// A finished main fit and how it was made, as the output JSON records it.
pub struct FitRecord<'a> {
    /// The fitted model.
    pub model: &'a PsmcModel,
    /// Its EM history, from the starting parameters on.
    pub history: &'a EmHistory,
    /// Bases per bin.
    pub bin_size: usize,
    /// Years per generation.
    pub gen_years: f64,
    /// theta0/rho0 of the starting parameters.
    pub tr_ratio: f64,
    /// Weight of the M-step's smoothness penalty.
    pub smooth_lambda: f64,
    /// What the input held.
    pub input: &'a InputSummary,
    /// The input QC report, when it could be made.
    pub qc: Option<&'a QcReport>,
}

/// The log-likelihood after each number of EM iterations, from the start
/// (0) to the end ([`RunMetadata::loglike`]).
fn loglike_path(history: &EmHistory) -> Vec<f64> {
    let mut path: Vec<f64> = history.loglike.iter().map(|&(before, _)| before).collect();
    path.extend(history.final_loglike());
    path
}

/// Write the fitted parameters with their run metadata.
pub fn save_fit(path: &Path, fit: &FitRecord) -> Result<()> {
    let mut params = fit.model.params_file();
    params.run = RunMetadata {
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        n_iter: Some(fit.history.loglike.len()),
        final_loglike: fit.history.final_loglike(),
        loglike: Some(loglike_path(fit.history)),
        bin_size: Some(fit.bin_size),
        gen_years: Some(fit.gen_years),
        tr_ratio: Some(fit.tr_ratio),
        smooth_lambda: Some(fit.smooth_lambda),
        input: Some(fit.input.clone()),
        qc: fit.qc.map(QcReport::summary),
    };
    save_params(path, &params)
}

/// Write the main fit as C psmc's `.psmc` output.
pub fn write_main_psmc(
    path: &Path,
    model: &PsmcModel,
    history: &EmHistory,
    data: &Dataset,
) -> Result<()> {
    let mut info = PsmcRunInfo::from_history(model, history, &data.obs.rows, &data.obs.row_starts)?;
    // C psmc counts the empty records of a psmcfa file in n_seqs.
    if let SourceSummary::Psmcfa(p) = &data.source {
        info.n_seqs = p.records;
    }
    let rounds = psmc_rounds(model, history, info.sum_l)?;
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    write_psmc(BufWriter::new(file), &info, &rounds)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Write each replicate's `.psmc` output next to its JSON file,
/// `replicate_NNN.psmc` in `dir`.
pub fn write_replicate_psmc(dir: &Path, replicates: &[BootstrapReplicate]) -> Result<()> {
    for (i, rep) in replicates.iter().enumerate() {
        let path = replicate_json_path(dir, i + 1).with_extension("psmc");
        let file =
            File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
        rep.write_psmc(BufWriter::new(file))
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}

/// Write the bootstrap summary as `summary.tsv` and `summary.json` in `dir`;
/// returns their paths.
pub fn write_bootstrap_summary(
    dir: &Path,
    summary: &BootstrapSummary,
) -> Result<(PathBuf, PathBuf)> {
    let tsv = dir.join("summary.tsv");
    let json = dir.join("summary.json");
    write_bootstrap_summary_tsv(&tsv, summary)?;
    write_bootstrap_summary_json(&json, summary)?;
    Ok((tsv, json))
}

/// Write `report` as JSON to `path`, or to stdout for `-`.
pub fn write_qc_json(path: &Path, report: &QcReport) -> Result<()> {
    let json = report.to_json() + "\n";
    if path == Path::new("-") {
        print!("{json}");
        return Ok(());
    }
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}
