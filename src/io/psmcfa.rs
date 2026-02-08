use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::io::Observations;

fn read_to_string(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    let mut reader: Box<dyn Read> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(BufReader::new(file))
    };
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

fn chunk_sequences(seqs: Vec<Vec<u8>>, batch_size: Option<usize>) -> Result<Observations> {
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();

    match batch_size {
        None => {
            for seq in seqs {
                if seq.is_empty() {
                    continue;
                }
                rows.push(seq);
                row_starts.push(true);
            }
        }
        Some(batch) => {
            if batch == 0 {
                bail!("batch_size must be > 0");
            }
            for seq in seqs {
                if seq.is_empty() {
                    continue;
                }
                for (i, chunk) in seq.chunks(batch).enumerate() {
                    rows.push(chunk.to_vec());
                    row_starts.push(i == 0);
                }
            }
        }
    }

    if rows.is_empty() {
        bail!("no valid observations found in psmcfa input");
    }

    Ok(Observations { rows, row_starts })
}

pub fn read_psmcfa(path: &Path, batch_size: Option<usize>) -> Result<Observations> {
    let content = read_to_string(path)?;
    let seqs = parse_sequences(&content)?;
    chunk_sequences(seqs, batch_size)
}
