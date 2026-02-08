use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::io::Observations;

fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    let reader: Box<dyn Read> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

fn flush_bin(seq: &mut Vec<u8>, bin_has_k: &mut bool, bin_len: &mut usize, bin_size: usize) {
    if *bin_len == bin_size {
        seq.push(if *bin_has_k { 1 } else { 0 });
        *bin_has_k = false;
        *bin_len = 0;
    }
}

fn add_t_sites(
    seq: &mut Vec<u8>,
    mut n: usize,
    bin_has_k: &mut bool,
    bin_len: &mut usize,
    bin_size: usize,
) {
    while n > 0 {
        let room = bin_size - *bin_len;
        let take = n.min(room);
        *bin_len += take;
        n -= take;
        flush_bin(seq, bin_has_k, bin_len, bin_size);
    }
}

fn add_k_site(seq: &mut Vec<u8>, bin_has_k: &mut bool, bin_len: &mut usize, bin_size: usize) {
    *bin_has_k = true;
    *bin_len += 1;
    flush_bin(seq, bin_has_k, bin_len, bin_size);
}

fn chunk_single_sequence(seq: Vec<u8>, batch_size: Option<usize>) -> Result<Observations> {
    if seq.is_empty() {
        bail!("no valid mhs rows found in input");
    }
    match batch_size {
        None => Ok(Observations {
            rows: vec![seq],
            row_starts: vec![true],
        }),
        Some(batch) => {
            if batch == 0 {
                bail!("batch_size must be > 0");
            }
            let mut rows = Vec::new();
            let mut row_starts = Vec::new();
            for (i, chunk) in seq.chunks(batch).enumerate() {
                rows.push(chunk.to_vec());
                row_starts.push(i == 0);
            }
            Ok(Observations { rows, row_starts })
        }
    }
}

pub fn read_mhs(path: &Path, batch_size: Option<usize>, bin_size: usize) -> Result<Observations> {
    if bin_size == 0 {
        bail!("mhs bin_size must be > 0");
    }

    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut seq: Vec<u8> = Vec::new();
    let mut bin_has_k = false;
    let mut bin_len = 0usize;
    let mut row_no = 0usize;

    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .with_context(|| format!("failed to read {path:?}"))?;
        if bytes == 0 {
            break;
        }
        row_no += 1;

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let mut cols = trimmed.split_whitespace();
        let _chrom = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing chrom"))?;
        let pos_str = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing pos"))?;
        let called_str = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing nr_called"))?;

        let _pos: u64 = pos_str
            .parse()
            .with_context(|| format!("invalid mhs row {row_no}: bad pos '{pos_str}'"))?;
        let called: usize = called_str
            .parse()
            .with_context(|| format!("invalid mhs row {row_no}: bad nr_called '{called_str}'"))?;
        if called == 0 {
            bail!("invalid mhs row {row_no}: nr_called must be > 0");
        }

        add_t_sites(&mut seq, called - 1, &mut bin_has_k, &mut bin_len, bin_size);
        add_k_site(&mut seq, &mut bin_has_k, &mut bin_len, bin_size);
    }

    if bin_len > 0 {
        seq.push(if bin_has_k { 1 } else { 0 });
    }

    chunk_single_sequence(seq, batch_size)
}
