//! Writers for the files other tools read: psmcfa as `fq2psmcfa` writes it,
//! and single-sample multihetsep as `generate_multihetsep.py` writes it.
//! Paths ending in `.gz` are written gzip-compressed.

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::io::input::{Dataset, HetSite};

/// Create `path` and write it with `write`, gzip-compressed when its name
/// ends in `.gz`.
fn write_file(path: &Path, write: impl FnOnce(&mut dyn Write) -> Result<()>) -> Result<()> {
    let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
    let mut out = BufWriter::with_capacity(1 << 16, file);
    let written = if path.to_string_lossy().ends_with(".gz") {
        // Lines reach the compressor 64 KiB at a time, not one by one.
        let gz = GzEncoder::new(&mut out, Compression::default());
        let mut gz = BufWriter::with_capacity(1 << 16, gz);
        write(&mut gz).and_then(|()| {
            gz.into_inner().map_err(|e| e.into_error())?.finish()?;
            Ok(())
        })
    } else {
        write(&mut out)
    };
    written
        .and_then(|()| Ok(out.flush()?))
        .with_context(|| format!("failed to write {path:?}"))
}

const PSMCFA_SYMBOLS: [u8; 3] = *b"TKN";

fn check_psmcfa_name(name: &str) -> Result<()> {
    if name.bytes().any(|b| b.is_ascii_whitespace()) || name.is_empty() {
        bail!("psmcfa sequence names must be non-empty and contain no whitespace: '{name}'");
    }
    Ok(())
}

fn invalid_symbol(b: u8, name: &str) -> anyhow::Error {
    anyhow::anyhow!("invalid bin symbol {b} in sequence '{name}' (expected 0, 1 or 2)")
}

/// Write sequences in `fq2psmcfa`'s exact layout: `>name`, then the bins as
/// `T` (0), `K` (1) and `N` (2) with a newline before every 60th symbol
/// starting with the first, and a final newline.
pub fn write_psmcfa_to<'a, W, I>(out: &mut W, seqs: I) -> Result<()>
where
    W: Write + ?Sized,
    I: IntoIterator<Item = (&'a str, &'a [u8])>,
{
    let mut line = Vec::with_capacity(61);
    for (name, bins) in seqs {
        check_psmcfa_name(name)?;
        out.write_all(b">")?;
        out.write_all(name.as_bytes())?;
        for chunk in bins.chunks(60) {
            line.clear();
            line.push(b'\n');
            for &b in chunk {
                let &sym = PSMCFA_SYMBOLS
                    .get(b as usize)
                    .ok_or_else(|| invalid_symbol(b, name))?;
                line.push(sym);
            }
            out.write_all(&line)?;
        }
        out.write_all(b"\n")?;
    }
    Ok(())
}

/// Write the kept sequences of `data` as psmcfa (gzip when `path` ends in
/// `.gz`); see [`write_psmcfa_to`]. Names and bins are checked before the
/// file is created, so a sequence that cannot be written leaves an existing
/// file as it was.
pub fn write_psmcfa(path: &Path, data: &Dataset) -> Result<()> {
    let seqs = data.sequences();
    let named = || {
        seqs.iter()
            .map(|(info, bins)| (info.name.as_str(), bins.as_ref()))
    };
    let check = || -> Result<()> {
        for (name, bins) in named() {
            check_psmcfa_name(name)?;
            if let Some(&b) = bins
                .iter()
                .find(|&&b| usize::from(b) >= PSMCFA_SYMBOLS.len())
            {
                return Err(invalid_symbol(b, name));
            }
        }
        Ok(())
    };
    check().with_context(|| format!("failed to write {path:?}"))?;
    write_file(path, |out| write_psmcfa_to(out, named()))
}

/// Options for [`write_multihetsep`].
#[derive(Debug, Clone, Default)]
pub struct MultihetsepOptions {
    /// Write only the sequences with this name (MSMC2 is usually given one
    /// file per chromosome); `None` writes every kept sequence.
    pub sequence: Option<String>,
}

/// Write the heterozygous sites of `data` in `generate_multihetsep.py`'s
/// single-sample layout: `chrom`, `pos`, `nr_called` and the two alleles,
/// tab-separated, one heterozygous site per line.
///
/// For one chromosome of a single-sample VCF of PASS SNP calls and a callable
/// mask, the output is byte for byte that of
/// `generate_multihetsep.py --mask mask.bed.gz file.vcf.gz`. Elsewhere the
/// script behaves differently: it writes rows for indels (alleles of several
/// characters, which also widen the row, see [`crate::io::mhs::read_mhs_with`])
/// where the loader treats a heterozygous indel as missing; it ignores FILTER
/// and fails on missing genotypes; it gives a second record at one position
/// `nr_called` 0; it reads mask rows without looking at their chromosome and
/// counts `nr_called` across chromosomes, so it takes one chromosome per run;
/// and past the end of a negative mask it treats every position as uncallable.
pub fn write_multihetsep_to<W: Write + ?Sized>(
    out: &mut W,
    data: &Dataset,
    opts: &MultihetsepOptions,
) -> Result<()> {
    let (sites, selected) = multihetsep_selection(data, opts)?;
    let mut line = Vec::with_capacity(64);
    for site in sites {
        let seq = site.seq as usize;
        if !selected[seq] {
            continue;
        }
        line.clear();
        line.extend_from_slice(data.seqs[seq].name.as_bytes());
        write!(line, "\t{}\t{}\t", site.pos, site.nr_called)?;
        line.extend_from_slice(&site.alleles);
        line.push(b'\n');
        out.write_all(&line)?;
    }
    Ok(())
}

/// The heterozygous sites of `data`, and which of its sequences `opts`
/// selects; an error when there are no sites or no sequence is selected.
fn multihetsep_selection<'a>(
    data: &'a Dataset,
    opts: &MultihetsepOptions,
) -> Result<(&'a [HetSite], Vec<bool>)> {
    let Some(sites) = &data.het_sites else {
        bail!(
            "the input was loaded without heterozygous sites (psmcfa input has no positions, \
             otherwise load it with collect_het_sites)"
        );
    };
    let selected: Vec<bool> = data
        .seqs
        .iter()
        .map(|s| opts.sequence.as_ref().is_none_or(|name| *name == s.name))
        .collect();
    if let Some(name) = &opts.sequence
        && !selected.contains(&true)
    {
        bail!("no kept sequence is named '{name}'");
    }
    Ok((sites, selected))
}

/// [`write_multihetsep_to`] into `path` (gzip when it ends in `.gz`). The
/// request is checked before the file is created, so a missing sequence or
/// missing heterozygous sites leave an existing file as it was.
pub fn write_multihetsep(path: &Path, data: &Dataset, opts: &MultihetsepOptions) -> Result<()> {
    multihetsep_selection(data, opts).with_context(|| format!("failed to write {path:?}"))?;
    write_file(path, |out| write_multihetsep_to(out, data, opts))
}
