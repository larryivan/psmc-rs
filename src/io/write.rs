//! Writers for the files other tools read: psmcfa as `fq2psmcfa` writes it,
//! and single-sample multihetsep as `generate_multihetsep.py` writes it.
//! Paths ending in `.gz` are written gzip-compressed.

use anyhow::{Context, Result, bail};
use flate2::Compression;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::io::input::Dataset;

/// An output file, gzip-compressed when its name ends in `.gz`.
enum Output {
    Plain(BufWriter<File>),
    Gzip(GzEncoder<BufWriter<File>>),
}

impl Output {
    fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).with_context(|| format!("failed to create {path:?}"))?;
        let buf = BufWriter::with_capacity(1 << 16, file);
        Ok(if path.to_string_lossy().ends_with(".gz") {
            Output::Gzip(GzEncoder::new(buf, Compression::default()))
        } else {
            Output::Plain(buf)
        })
    }

    fn writer(&mut self) -> &mut dyn Write {
        match self {
            Output::Plain(w) => w,
            Output::Gzip(w) => w,
        }
    }

    fn finish(self, path: &Path) -> Result<()> {
        let mut buf = match self {
            Output::Plain(w) => w,
            Output::Gzip(w) => w
                .finish()
                .with_context(|| format!("failed to write {path:?}"))?,
        };
        buf.flush()
            .with_context(|| format!("failed to write {path:?}"))?;
        Ok(())
    }
}

/// Write sequences in `fq2psmcfa`'s exact layout: `>name`, then the bins as
/// `T` (0), `K` (1) and `N` (2) with a newline before every 60th symbol
/// starting with the first, and a final newline.
pub fn write_psmcfa_to<'a, W, I>(out: &mut W, seqs: I) -> Result<()>
where
    W: Write + ?Sized,
    I: IntoIterator<Item = (&'a str, &'a [u8])>,
{
    const SYMBOLS: [u8; 3] = *b"TKN";
    let mut line = Vec::with_capacity(61);
    for (name, bins) in seqs {
        if name.bytes().any(|b| b.is_ascii_whitespace()) || name.is_empty() {
            bail!("psmcfa sequence names must be non-empty and contain no whitespace: '{name}'");
        }
        out.write_all(b">")?;
        out.write_all(name.as_bytes())?;
        for chunk in bins.chunks(60) {
            line.clear();
            line.push(b'\n');
            for &b in chunk {
                let Some(&sym) = SYMBOLS.get(b as usize) else {
                    bail!("invalid bin symbol {b} in sequence '{name}' (expected 0, 1 or 2)");
                };
                line.push(sym);
            }
            out.write_all(&line)?;
        }
        out.write_all(b"\n")?;
    }
    Ok(())
}

/// Write the kept sequences of `data` as psmcfa (gzip when `path` ends in
/// `.gz`); see [`write_psmcfa_to`].
pub fn write_psmcfa(path: &Path, data: &Dataset) -> Result<()> {
    let mut out = Output::create(path)?;
    let seqs = data.sequences();
    write_psmcfa_to(
        out.writer(),
        seqs.iter()
            .map(|(info, bins)| (info.name.as_str(), bins.as_ref())),
    )
    .with_context(|| format!("failed to write {path:?}"))?;
    out.finish(path)
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
pub fn write_multihetsep_to<W: Write + ?Sized>(
    out: &mut W,
    data: &Dataset,
    opts: &MultihetsepOptions,
) -> Result<()> {
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

/// [`write_multihetsep_to`] into `path` (gzip when it ends in `.gz`).
pub fn write_multihetsep(path: &Path, data: &Dataset, opts: &MultihetsepOptions) -> Result<()> {
    let mut out = Output::create(path)?;
    write_multihetsep_to(out.writer(), data, opts)
        .with_context(|| format!("failed to write {path:?}"))?;
    out.finish(path)
}
