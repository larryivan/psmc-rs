//! `psmc-rs convert`: write an input, read as inference reads it, as psmcfa
//! or multihetsep.

use anyhow::{Result, bail};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use psmc_rs::io::input::{Dataset, InputFormat};
use psmc_rs::io::write::{MultihetsepOptions, write_multihetsep, write_psmcfa};
use psmc_rs::qc::QcReport;
use psmc_rs::qc::fmt::{counted, pct};

use crate::cli::{ConvertArgs, FormatArg, usage_bail};
use crate::console::Console;
use crate::dataset::{
    announce_input, callable_summary, counts_source, default_qc_report, load_input, report_qc,
};
use crate::output::{STDOUT_ONLY_QC, check_destinations, is_stdout, parent_dir};
use crate::run::{Event, Sink};

/// Run `convert`.
pub fn run(args: &ConvertArgs) -> Result<ExitCode> {
    // With the QC JSON on stdout, the result lines go to stderr.
    let json_to_stdout = args.qc_json.as_deref().is_some_and(is_stdout);
    let console = Console::new(args.quiet, false).results_on_stderr(json_to_stdout);
    let sink = |ev| console.emit(ev);
    let want_sites = args.multihetsep.is_some() || args.multihetsep_per_chrom.is_some();
    // Multihetsep of psmcfa input is refused as soon as the format is
    // known: from --input-format, from a file's first lines, or, for a
    // pipe, once it is read.
    check_sites(want_sites, args.input.input_format.format())?;
    check_outputs(args)?;
    check_sites(want_sites, sniffed_format(args))?;
    let mut opts = args.input.to_options();
    opts.collect_het_sites = want_sites;
    let data = load_input(&args.input_file, &opts)?;
    check_sites(want_sites, Some(data.source.format()))?;
    let per_chrom = per_chrom_files(args, &data)?;
    announce_input(&data, &args.input, &sink);

    // Also when no sequence is left: the report then says why.
    let report = default_qc_report(&data, args.input.bin_size, 0);
    let qc = report_qc(&data, report, args.qc_json.as_deref(), &sink)?;
    if data.obs.rows.is_empty() {
        // fq2psmcfa writes an empty file too; the outputs say so.
        sink(Event::Warning(
            "no sequence is left to write, so the outputs are empty".to_string(),
        ));
    }
    if let Some(path) = &args.psmcfa {
        write_psmcfa_file(path, &data, args.input.bin_size, &sink)?;
    }
    if want_sites {
        let whole = args.multihetsep.as_deref();
        write_multihetsep_files(whole, &per_chrom, &data, qc.as_ref(), &sink)?;
    }
    Ok(ExitCode::SUCCESS)
}

/// Refuse multihetsep output (`want_sites`) of input whose format is
/// psmcfa, which has no sites to write.
fn check_sites(want_sites: bool, format: Option<InputFormat>) -> Result<()> {
    if want_sites && format == Some(InputFormat::Psmcfa) {
        usage_bail!(
            "psmcfa input has no sites to write as multihetsep: its bins only say whether each \
             holds a heterozygous site, not where. Convert the VCF or multihetsep file the \
             psmcfa was made from instead"
        );
    }
    Ok(())
}

/// The format of the input recognised before it is read: with
/// `--input-format auto`, from the first lines of a regular file. A pipe or
/// FIFO can be read only once, so only the loader can recognise its format.
/// A file whose format cannot be recognised is left to the loader, which
/// says why.
fn sniffed_format(args: &ConvertArgs) -> Option<InputFormat> {
    let auto = args.input.input_format == FormatArg::Auto;
    (auto && args.input_file.is_file())
        .then(|| InputFormat::sniff(&args.input_file).ok())
        .flatten()
}

/// Write `data` as psmcfa to `path`, after what it cannot record.
fn write_psmcfa_file(path: &Path, data: &Dataset, bin_size: usize, sink: Sink) -> Result<()> {
    if data.source.format() == InputFormat::Mhs {
        sink(Event::Info(
            "multihetsep gives no position for the uncalled sites between two rows, so they \
             were placed at the start of each interval; which bins are missing can differ \
             from fq2psmcfa's binning of the same calls (read the VCF with its callable \
             mask for exact bins)"
                .to_string(),
        ));
    }
    if let Some(line) = partial_bins_warning(data, bin_size, path) {
        sink(Event::Warning(line));
    }
    write_psmcfa(path, data)?;
    sink(Event::Output(format!(
        "psmcfa: {} ({}, {})",
        path.display(),
        counted(data.seqs.len(), "sequence", "sequences"),
        counted(data.n_bins(), "bin", "bins")
    )));
    Ok(())
}

/// The warning for writing `data` as psmcfa to `path` when some of its
/// called bins are partly callable by exact counts (VCF): psmcfa records
/// each bin's symbol only, so a fit of the file takes every called bin as
/// `bin_size` bases, as fq2psmcfa's output and C psmc do, while a fit of
/// the VCF itself weighs each bin by its callable bases. Multihetsep's
/// approximate counts are not fitted by default either, so its psmcfa
/// fits as it does.
fn partial_bins_warning(data: &Dataset, bin_size: usize, path: &Path) -> Option<String> {
    if counts_source(data) != Some("exact") {
        return None;
    }
    let c = callable_summary(data).filter(|c| c.partial_bins > 0)?;
    Some(format!(
        "{} of called bins are partly callable (they hold {} of their {bin_size} bases on \
         average), and psmcfa cannot record that: a fit of {} takes every called bin as \
         {bin_size} bases, as C psmc does, which biases theta, the times and Ne low by about \
         the missing share or more ({} here). To fit with the callable bases of each bin, fit \
         the VCF itself (psmc-rs does with --bin-callable auto, its default)",
        pct(c.partial_fraction()),
        pct(c.mean_fraction),
        path.display(),
        pct(1.0 - c.mean_fraction)
    ))
}

/// Write `data` as multihetsep, the whole input to `whole` and then each
/// sequence to its file of `per_chrom`, after what multihetsep cannot
/// record of a VCF whose QC report `qc` finds uncalled sites.
fn write_multihetsep_files(
    whole: Option<&Path>,
    per_chrom: &[(String, PathBuf)],
    data: &Dataset,
    qc: Option<&QcReport>,
    sink: Sink,
) -> Result<()> {
    let uncalled = qc.and_then(|r| r.totals.uncalled_site_fraction);
    if data.source.format() == InputFormat::Vcf
        && let Some(uncalled) = uncalled.filter(|&u| u > 0.0)
    {
        sink(Event::Info(format!(
            "multihetsep records how many sites between two rows are called, not which ({} \
             of the sites here are uncalled); psmc-rs reads the uncalled sites of each interval \
             back at its start, which biases theta and Ne low whatever --bin-callable (QC's \
             APPROX_MISSING_PLACEMENT on the multihetsep says how much), so to fit with psmc-rs, \
             fit the VCF itself",
            pct(uncalled)
        )));
    }
    let whole = whole.map(|path| (path, None));
    let each = per_chrom
        .iter()
        .map(|(name, path)| (path.as_path(), Some(name.as_str())));
    for (path, sequence) in whole.into_iter().chain(each) {
        let opts = MultihetsepOptions {
            sequence: sequence.map(str::to_string),
        };
        write_multihetsep(path, data, &opts)?;
        sink(Event::Output(format!(
            "multihetsep: {} ({})",
            path.display(),
            counted(
                het_sites(data, sequence),
                "heterozygous site",
                "heterozygous sites"
            )
        )));
    }
    Ok(())
}

/// The files `convert` writes whose names are known before the input is
/// read, as `(flag, path)`.
fn output_files(args: &ConvertArgs) -> Vec<(&'static str, Option<&Path>)> {
    vec![
        ("--psmcfa", args.psmcfa.as_deref()),
        ("--multihetsep", args.multihetsep.as_deref()),
        // Only the QC JSON can go to stdout.
        (
            "--qc-json",
            args.qc_json.as_deref().filter(|p| !is_stdout(p)),
        ),
    ]
}

/// Check the files `convert` will write before reading the input (see
/// [`check_destinations`]); the per-sequence files need their directory,
/// and are checked by name once the input is read.
fn check_outputs(args: &ConvertArgs) -> Result<()> {
    check_destinations(&args.input_file, &output_files(args), None)?;
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

/// The per-sequence files of `--multihetsep-per-chrom PREFIX`: each name
/// of `data`'s sequences once (the blocks of a repeated name share a file),
/// with its path. They are named after the sequences, so only now can they
/// be checked against the input and the other outputs, before anything is
/// written.
fn per_chrom_files(args: &ConvertArgs, data: &Dataset) -> Result<Vec<(String, PathBuf)>> {
    let Some(prefix) = &args.multihetsep_per_chrom else {
        return Ok(Vec::new());
    };
    let mut seen = HashSet::new();
    let files: Vec<(String, PathBuf)> = data
        .seqs
        .iter()
        .filter(|seq| seen.insert(seq.name.as_str()))
        .map(|seq| Ok((seq.name.clone(), per_chrom_path(prefix, &seq.name)?)))
        .collect::<Result<_>>()?;
    if !files.is_empty() {
        let mut outputs = output_files(args);
        let each = files.iter().map(|(_, path)| Some(path.as_path()));
        outputs.extend(each.map(|path| ("--multihetsep-per-chrom", path)));
        check_destinations(&args.input_file, &outputs, None)?;
    }
    Ok(files)
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
