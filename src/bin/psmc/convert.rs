//! `psmc-rs convert`: write an input, read as inference reads it, as psmcfa
//! or multihetsep.

use anyhow::{Result, bail};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use psmc_rs::io::input::{Dataset, InputFormat};
use psmc_rs::io::write::{MultihetsepOptions, write_multihetsep, write_psmcfa};

use crate::cli::{ConvertArgs, FormatArg, usage_bail};
use crate::console::Console;
use crate::dataset::{describe, ignored_options, load_input, loader_warnings, qc_report};
use crate::output::{STDOUT_ONLY_QC, check_destinations, is_stdout, parent_dir, write_qc_json};
use crate::run::{Event, qc_event, qc_failed};

const NO_SITES: &str = "psmcfa input has no sites to write as multihetsep: its bins only say \
                        whether each holds a heterozygous site, not where. Convert the VCF or \
                        multihetsep file the psmcfa was made from instead";

/// Run `convert`.
pub fn run(args: &ConvertArgs) -> Result<ExitCode> {
    // With the QC JSON on stdout, the result lines go to stderr.
    let json_to_stdout = args.qc_json.as_deref().is_some_and(is_stdout);
    let console = Console::new(args.quiet, false).results_on_stderr(json_to_stdout);
    let sink = |ev| console.emit(ev);
    let want_sites = args.multihetsep.is_some() || args.multihetsep_per_chrom.is_some();
    if want_sites && args.input.input_format == FormatArg::Psmcfa {
        usage_bail!("{NO_SITES}");
    }
    check_outputs(args)?;
    let mut opts = args.input.to_options();
    opts.collect_het_sites = want_sites;
    let data = load_input(&args.input_file, &opts)?;
    let format = data.source.format();
    if want_sites && format == InputFormat::Psmcfa {
        bail!("{NO_SITES}");
    }
    for line in describe(&data) {
        sink(Event::Info(line));
    }
    if let Some(line) = ignored_options(&args.input, &data) {
        sink(Event::Warning(line));
    }

    // Also when no sequence is left: the report then says why.
    let qc = match qc_report(&data, args.input.bin_size, 0, None) {
        Ok(report) => Some(report),
        Err(e) => {
            sink(qc_failed(e, args.qc_json.as_deref()));
            None
        }
    };
    for line in loader_warnings(&data, qc.as_ref()) {
        sink(Event::Warning(line));
    }
    if let Some(report) = &qc {
        sink(qc_event(report));
        if let Some(path) = &args.qc_json {
            write_qc_json(path, report)?;
            if !json_to_stdout {
                sink(Event::Output(format!("QC report: {}", path.display())));
            }
        }
    }
    if data.obs.rows.is_empty() {
        // fq2psmcfa writes an empty file too; the outputs say so.
        sink(Event::Warning(
            "no sequence is left to write, so the outputs are empty".to_string(),
        ));
    }

    if let Some(path) = &args.psmcfa {
        if format == InputFormat::Mhs {
            sink(Event::Info(
                "multihetsep gives no position for the uncalled sites between two rows, so they \
                 were placed at the start of each interval; which bins are missing can differ \
                 from fq2psmcfa's binning of the same calls (read the VCF with its callable \
                 mask for exact bins)"
                    .to_string(),
            ));
        }
        write_psmcfa(path, &data)?;
        sink(Event::Output(format!(
            "psmcfa: {} ({} sequences, {} bins)",
            path.display(),
            data.seqs.len(),
            data.n_bins()
        )));
    }
    if let Some(path) = &args.multihetsep {
        write_multihetsep(path, &data, &MultihetsepOptions::default())?;
        sink(Event::Output(format!(
            "multihetsep: {} ({} heterozygous sites)",
            path.display(),
            het_sites(&data, None)
        )));
    }
    if let Some(prefix) = &args.multihetsep_per_chrom {
        let mut written = HashSet::new();
        for seq in &data.seqs {
            if !written.insert(seq.name.as_str()) {
                // A repeated chromosome went into the file of its first
                // block already.
                continue;
            }
            let path = per_chrom_path(prefix, &seq.name)?;
            let opts = MultihetsepOptions {
                sequence: Some(seq.name.clone()),
            };
            write_multihetsep(&path, &data, &opts)?;
            sink(Event::Output(format!(
                "multihetsep: {} ({} heterozygous sites)",
                path.display(),
                het_sites(&data, Some(&seq.name))
            )));
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Check the files `convert` will write before reading the input (see
/// [`check_destinations`]); the per-sequence files need their directory.
fn check_outputs(args: &ConvertArgs) -> Result<()> {
    let mut files = Vec::new();
    for (what, path) in [
        ("--psmcfa", &args.psmcfa),
        ("--multihetsep", &args.multihetsep),
    ] {
        if let Some(path) = path.as_deref() {
            files.push((what, path));
        }
    }
    // Only the QC JSON can go to stdout.
    if let Some(path) = args.qc_json.as_deref().filter(|p| !is_stdout(p)) {
        files.push(("--qc-json", path));
    }
    check_destinations(&args.input_file, &files, &[])?;
    if let Some(prefix) = &args.multihetsep_per_chrom {
        if is_stdout(prefix) {
            usage_bail!("--multihetsep-per-chrom cannot be '-': {STDOUT_ONLY_QC}");
        }
        let dir = parent_dir(prefix);
        if !dir.is_dir() {
            bail!(
                "cannot write --multihetsep-per-chrom {}: the directory {} does not exist",
                prefix.display(),
                dir.display()
            );
        }
    }
    Ok(())
}

/// Heterozygous sites of `data`, of the sequences named `name` if given.
fn het_sites(data: &Dataset, name: Option<&str>) -> usize {
    data.het_sites.as_ref().map_or(0, |sites| {
        sites
            .iter()
            .filter(|h| name.is_none_or(|n| data.seqs[h.seq as usize].name == n))
            .count()
    })
}

/// The file of sequence `name` under `--multihetsep-per-chrom PREFIX`:
/// `PREFIX.<name>.multihetsep`, with `.gz` at the end when `PREFIX` ends in
/// `.gz`.
fn per_chrom_path(prefix: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        bail!("the sequence name '{name}' cannot be part of a file name");
    }
    let prefix = prefix.to_string_lossy();
    Ok(PathBuf::from(match prefix.strip_suffix(".gz") {
        Some(stem) => format!("{stem}.{name}.multihetsep.gz"),
        None => format!("{prefix}.{name}.multihetsep"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_chromosome_files_are_named_after_the_prefix() {
        let path = |prefix: &str, name: &str| per_chrom_path(Path::new(prefix), name);
        assert_eq!(
            path("out/snps", "chr1").unwrap(),
            PathBuf::from("out/snps.chr1.multihetsep")
        );
        assert_eq!(
            path("out/snps.gz", "chr2").unwrap(),
            PathBuf::from("out/snps.chr2.multihetsep.gz")
        );
        assert!(path("snps", "a/b").is_err());
        assert!(path("snps", "..").is_err());
    }
}
