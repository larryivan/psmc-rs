//! The files a run writes besides the HTML report: the parameters with
//! their run metadata, C psmc's `.psmc` output, the bootstrap files and the
//! QC JSON.

use anyhow::{Context, Result, bail};
use std::fs::{self, File, OpenOptions};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapReplicate, BootstrapSummary, replicate_json_path, write_bootstrap_summary_json,
    write_bootstrap_summary_tsv,
};
use psmc_rs::em::EmHistory;
use psmc_rs::io::input::{Dataset, SourceSummary};
use psmc_rs::io::params::{BinCallableRecord, InputSummary, RunMetadata, save_params};
use psmc_rs::psmc_out::{PsmcRunInfo, psmc_rounds, write_psmc};
use psmc_rs::qc::QcReport;

use crate::cli::{InferArgs, usage_bail};

/// A finished main fit and how it was made, as the output JSON and the
/// report record it.
pub struct FitRecord<'a> {
    /// The run's settings.
    pub cli: &'a InferArgs,
    /// The fitted model.
    pub model: &'a PsmcModel,
    /// Its EM history, from the starting parameters on: one entry per EM
    /// iteration.
    pub history: &'a EmHistory,
    /// Log-likelihood of `model`: the history's final value, or the
    /// starting parameters' after 0 iterations.
    pub final_loglike: f64,
    /// theta0/rho0 of the starting parameters.
    pub tr_ratio: f64,
    /// What the input held.
    pub input: &'a InputSummary,
    /// The input QC report, when it could be made.
    pub qc: Option<&'a QcReport>,
    /// How the fit used the callable bases of each bin.
    pub bin_callable: &'a BinCallableRecord,
}

/// Write the fitted parameters with their run metadata.
pub fn save_fit(path: &Path, fit: &FitRecord) -> Result<()> {
    // The log-likelihood after each number of EM iterations, from 0 on.
    let loglike = fit.history.loglike.iter().map(|&(before, _)| before);
    let mut params = fit.model.params_file();
    params.run = RunMetadata {
        version: Some(env!("CARGO_PKG_VERSION").to_string()),
        simd: Some(psmc_rs::hmm::simd_path().name().to_string()),
        n_iter: Some(fit.history.loglike.len()),
        final_loglike: Some(fit.final_loglike),
        loglike: Some(loglike.chain([fit.final_loglike]).collect()),
        bin_size: Some(fit.cli.input.bin_size),
        gen_years: Some(fit.cli.gen_years),
        tr_ratio: Some(fit.tr_ratio),
        smooth_lambda: Some(fit.cli.smooth_lambda),
        input: Some(fit.input.clone()),
        qc: fit.qc.map(QcReport::summary),
        bin_callable: Some(fit.bin_callable.clone()),
    };
    save_params(path, &params)
}

/// Write the main fit of `data` as C psmc's `.psmc` output, with an `MM`
/// line saying how it used the callable bases of each bin when the input
/// has them.
pub fn write_main_psmc(path: &Path, fit: &FitRecord, data: &Dataset) -> Result<()> {
    let (model, history) = (fit.model, fit.history);
    let mut info = PsmcRunInfo::from_history(model, history, &data.obs.rows, &data.obs.row_starts)?;
    // C psmc counts the empty records of a psmcfa file in n_seqs.
    if let SourceSummary::Psmcfa(p) = &data.source {
        info.n_seqs = p.records;
    }
    info.bin_callable = fit.bin_callable.psmc_line();
    let rounds = psmc_rounds(model, history, info.sum_l)?;
    write_buffered(path, |w| Ok(write_psmc(w, &info, &rounds)?))
}

/// Write each replicate's `.psmc` output next to its JSON file,
/// `replicate_NNN.psmc` in `dir` (numbered as the replicate, so skipped
/// replicates leave gaps).
pub fn write_replicate_psmc(dir: &Path, replicates: &[BootstrapReplicate]) -> Result<()> {
    for rep in replicates {
        let path = replicate_json_path(dir, rep.index + 1).with_extension("psmc");
        write_buffered(&path, |w| rep.write_psmc(w))?;
    }
    Ok(())
}

/// Create the file `path` and fill it through a buffer with `write`,
/// naming the file in errors.
fn write_buffered(path: &Path, write: impl FnOnce(BufWriter<File>) -> Result<()>) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    write(BufWriter::new(file)).with_context(|| format!("failed to write {}", path.display()))
}

/// The file names of the bootstrap summary in its directory: TSV, JSON.
pub const BOOTSTRAP_SUMMARY_FILES: [&str; 2] = ["summary.tsv", "summary.json"];

/// Write the bootstrap summary as `summary.tsv` and `summary.json` in `dir`
/// ([`BOOTSTRAP_SUMMARY_FILES`]); returns their paths.
pub fn write_bootstrap_summary(
    dir: &Path,
    summary: &BootstrapSummary,
) -> Result<(PathBuf, PathBuf)> {
    let [tsv, json] = BOOTSTRAP_SUMMARY_FILES.map(|name| dir.join(name));
    write_bootstrap_summary_tsv(&tsv, summary)?;
    write_bootstrap_summary_json(&json, summary)?;
    Ok((tsv, json))
}

/// Whether `path` names stdout (`-`).
pub fn is_stdout(path: &Path) -> bool {
    path == Path::new("-")
}

/// Why an output other than the QC JSON cannot be `-`.
pub const STDOUT_ONLY_QC: &str =
    "only the QC JSON (--qc-json, qc --json) can go to stdout; give a file name";

/// Write `report` as JSON to `path`, or to stdout for `-`.
pub fn write_qc_json(path: &Path, report: &QcReport) -> Result<()> {
    let json = report.to_json() + "\n";
    if is_stdout(path) {
        crate::console::print_out(&json);
        return Ok(());
    }
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

/// Check, before any work, that a run can write its outputs: `files` (each
/// named as the user gave it, such as `--psmc-out`; `None` when not asked
/// for) are distinct, none is the input or `dir`, and each can be written in
/// an existing directory; `dir` is a directory or can be created. Otherwise
/// a typo in a directory would only fail after the fit and the bootstrap,
/// and two outputs of one name would overwrite each other. Names that
/// collide are a usage error, and so is `-`: only the QC JSON can go to
/// stdout, so callers leave it out when it does, and any other output
/// named `-` would be a file of that name.
pub fn check_destinations(
    input: &Path,
    files: &[(&str, Option<&Path>)],
    dir: Option<(&str, &Path)>,
) -> Result<()> {
    let input = resolved(input);
    let files: Vec<(&str, &Path)> = (files.iter())
        .filter_map(|&(what, path)| Some((what, path?)))
        .collect();
    let mut seen: Vec<(&str, &Path, PathBuf)> = Vec::new();
    for &(what, path) in files.iter().chain(&dir) {
        if is_stdout(path) {
            usage_bail!("{what} cannot be '-': {STDOUT_ONLY_QC}");
        }
        let at = resolved(path);
        if at == input {
            usage_bail!(
                "{what} {} is the input file; give it another name",
                path.display()
            );
        }
        if let Some((other, other_path, _)) = seen.iter().find(|(_, _, p)| *p == at) {
            usage_bail!(
                "{other} {} and {what} {} are the same file; give them different names",
                other_path.display(),
                path.display()
            );
        }
        seen.push((what, path, at));
    }
    for (what, path) in files {
        check_writable_file(path)
            .with_context(|| format!("cannot write {what} {}", path.display()))?;
    }
    if let Some((what, path)) = dir {
        check_creatable_dir(path)
            .with_context(|| format!("cannot create {what} {}", path.display()))?;
    }
    Ok(())
}

/// `path` with its directory resolved (symbolic links, `.` and `..`), so
/// that two names of one file compare equal, also before it exists.
fn resolved(path: &Path) -> PathBuf {
    if let Ok(p) = fs::canonicalize(path) {
        return p;
    }
    match (fs::canonicalize(parent_dir(path)), path.file_name()) {
        (Ok(dir), Some(name)) => dir.join(name),
        _ => std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()),
    }
}

/// The directory `path` is in (`.` for a bare file name).
pub fn parent_dir(path: &Path) -> &Path {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    }
}

/// Whether the file `path` can be written, without changing it: an
/// existing file is opened for appending, a new one created and removed.
fn check_writable_file(path: &Path) -> Result<()> {
    if path.is_dir() {
        bail!("it is a directory");
    }
    let dir = parent_dir(path);
    if !dir.is_dir() {
        bail!("the directory {} does not exist", dir.display());
    }
    if path.exists() {
        OpenOptions::new().append(true).open(path)?;
    } else {
        File::create_new(path)?;
        fs::remove_file(path)?;
    }
    Ok(())
}

/// Whether the directory `path` exists or can be created: its nearest
/// existing ancestor must be a directory.
fn check_creatable_dir(path: &Path) -> Result<()> {
    let mut at = path;
    loop {
        if at.exists() {
            if !at.is_dir() {
                bail!("{} is not a directory", at.display());
            }
            return Ok(());
        }
        match at.parent() {
            Some(up) if !up.as_os_str().is_empty() => at = up,
            _ => return Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TempDir;

    #[test]
    fn destinations_must_be_distinct_writable_files() {
        let dir = TempDir::new("output_destinations");
        let input = dir.join("in.psmcfa");
        fs::write(&input, ">a\nT\n").unwrap();
        let out = dir.join("fit.json");
        let check = |files: &[(&str, &Path)], dir: Option<(&str, &Path)>| {
            let files: Vec<_> = files
                .iter()
                .map(|&(what, path)| (what, Some(path)))
                .collect();
            check_destinations(&input, &files, dir).map_err(|e| format!("{e:#}"))
        };
        let boot = dir.join("new/fit.bootstrap");
        check(&[("OUTPUT", &out)], Some(("--bootstrap-dir", &boot))).unwrap();
        // The check leaves no file behind.
        assert!(!out.exists() && !dir.join("new").exists());
        // An existing file is not changed.
        fs::write(&out, "{}").unwrap();
        check(&[("OUTPUT", &out)], None).unwrap();
        assert_eq!(fs::read_to_string(&out).unwrap(), "{}");

        // Two names of one file.
        let other = dir.join("sub/../fit.json");
        fs::create_dir_all(dir.join("sub")).unwrap();
        let err = check(&[("OUTPUT", &out), ("--psmc-out", &other)], None).unwrap_err();
        assert!(err.contains("are the same file"), "{err}");
        let err = check(&[("--tmrca-out", &input)], None).unwrap_err();
        assert!(err.contains("--tmrca-out"), "{err}");
        assert!(err.contains("is the input file"), "{err}");
        let err = check(&[("OUTPUT", &out)], Some(("--bootstrap-dir", &out))).unwrap_err();
        assert!(err.contains("are the same file"), "{err}");

        let missing = dir.join("nowhere/t.tsv");
        let err = check(&[("--tmrca-out", &missing)], None).unwrap_err();
        assert!(
            err.starts_with(&format!("cannot write --tmrca-out {}", missing.display())),
            "{err}"
        );
        assert!(err.contains("does not exist"), "{err}");
        let err = check(&[("OUTPUT", &dir)], None).unwrap_err();
        assert!(err.contains("is a directory"), "{err}");
        let err = check(&[], Some(("--bootstrap-dir", &input.join("boot")))).unwrap_err();
        assert!(err.contains("is not a directory"), "{err}");
        // `-` is stdout, which only the QC JSON can use.
        let dash = Path::new("-");
        let err = check(&[("--psmc-out", dash)], None).unwrap_err();
        assert!(err.starts_with("--psmc-out cannot be '-'"), "{err}");
        let err = check(&[], Some(("--bootstrap-dir", dash))).unwrap_err();
        assert!(err.starts_with("--bootstrap-dir cannot be '-'"), "{err}");
    }
}
