use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use ndarray::Array2;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn read_to_string(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
    let mut reader: Box<dyn Read> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(BufReader::new(file))
    };
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .with_context(|| format!("failed to read {:?}", path))?;
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

pub fn read_psmcfa(path: &Path, batch_size: Option<usize>) -> Result<Array2<u8>> {
    let content = read_to_string(path)?;
    match batch_size {
        Some(batch) => read_psmcfa_batched(&content, batch),
        None => read_psmcfa_unbatched(&content),
    }
}

fn read_psmcfa_batched(content: &str, batch: usize) -> Result<Array2<u8>> {
    if batch == 0 {
        bail!("batch_size must be > 0");
    }
    let mut data = Vec::new();
    for b in content.as_bytes() {
        if let Some(v) = map_base(*b) {
            data.push(v);
        }
    }
    let residual = data.len() % batch;
    if residual != 0 {
        let pad = batch - residual;
        data.extend(std::iter::repeat(2).take(pad));
    }
    let n_batches = data.len() / batch;
    Array2::from_shape_vec((n_batches, batch), data)
        .context("failed to reshape batched psmcfa data")
}

fn read_psmcfa_unbatched(content: &str) -> Result<Array2<u8>> {
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
    if seqs.is_empty() {
        bail!("no sequences found in psmcfa data");
    }
    let len0 = seqs[0].len();
    if len0 == 0 {
        bail!("empty sequence in psmcfa data");
    }
    for (i, s) in seqs.iter().enumerate() {
        if s.len() != len0 {
            bail!("sequence {} length {} does not match {}", i, s.len(), len0);
        }
    }
    let rows = seqs.len();
    let mut data = Vec::with_capacity(rows * len0);
    for s in seqs {
        data.extend(s);
    }
    Array2::from_shape_vec((rows, len0), data).context("failed to reshape unbatched psmcfa data")
}
