use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;

use crate::io::Observations;
use crate::io::common::{chunk_sequences, open_reader};

fn read_to_string(path: &Path) -> Result<String> {
    let mut reader = open_reader(path)?;
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .with_context(|| format!("failed to read {path:?}"))?;
    Ok(content)
}

fn map_base(b: u8) -> Option<u8> {
    match b {
        b'T' => Some(0),
        b'K' => Some(1),
        b'N' => Some(2),
        _ => None,
    }
}

fn parse_sequences(content: &str) -> Result<Vec<Vec<u8>>> {
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    for chunk in content.split('>').skip(1) {
        let mut seq = Vec::new();
        for b in chunk.as_bytes() {
            if let Some(v) = map_base(*b) {
                seq.push(v);
            }
        }
        if !seq.is_empty() {
            seqs.push(seq);
        }
    }

    // Fallback for header-less inputs.
    if seqs.is_empty() {
        let mut seq = Vec::new();
        for b in content.as_bytes() {
            if let Some(v) = map_base(*b) {
                seq.push(v);
            }
        }
        if !seq.is_empty() {
            seqs.push(seq);
        }
    }

    if seqs.is_empty() {
        bail!("no sequences found in psmcfa data");
    }
    Ok(seqs)
}

pub fn read_psmcfa(path: &Path, batch_size: Option<usize>) -> Result<Observations> {
    let content = read_to_string(path)?;
    let seqs = parse_sequences(&content)?;
    chunk_sequences(
        seqs,
        batch_size,
        "no valid observations found in psmcfa input",
    )
}
