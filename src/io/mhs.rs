use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::io::Observations;

// Keep this aligned with psmc/utils/fq2psmcfa.c:
// output N when missing ratio in a bin is > 0.9.
const N_RATIO: f64 = 0.9;

fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    let reader: Box<dyn Read> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    Ok(Box::new(BufReader::new(reader)))
}

#[derive(Debug, Clone, Default)]
struct BinAcc {
    len: usize,
    n_n: usize,
    has_k: bool,
}

fn flush_full_bin(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    if bin.len != bin_size {
        return;
    }
    let sym = if (bin.n_n as f64) / (bin_size as f64) > N_RATIO {
        2 // N
    } else if bin.has_k {
        1 // K
    } else {
        0 // T
    };
    seq.push(sym);
    *bin = BinAcc::default();
}

fn flush_tail_bin(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    if bin.len == 0 {
        return;
    }
    // fq2psmcfa uses BLOCK_LEN in denominator even for final short block.
    let sym = if (bin.n_n as f64) / (bin_size as f64) > N_RATIO {
        2
    } else if bin.has_k {
        1
    } else {
        0
    };
    seq.push(sym);
    *bin = BinAcc::default();
}

fn add_run_t(seq: &mut Vec<u8>, mut n: u64, bin: &mut BinAcc, bin_size: usize) {
    while n > 0 {
        let room = (bin_size - bin.len) as u64;
        let take = n.min(room);
        bin.len += take as usize;
        n -= take;
        flush_full_bin(seq, bin, bin_size);
    }
}

fn add_run_n(seq: &mut Vec<u8>, mut n: u64, bin: &mut BinAcc, bin_size: usize) {
    while n > 0 {
        let room = (bin_size - bin.len) as u64;
        let take = n.min(room);
        bin.len += take as usize;
        bin.n_n += take as usize;
        n -= take;
        flush_full_bin(seq, bin, bin_size);
    }
}

fn add_k(seq: &mut Vec<u8>, bin: &mut BinAcc, bin_size: usize) {
    bin.len += 1;
    bin.has_k = true;
    flush_full_bin(seq, bin, bin_size);
}

fn split_orders(alleles: &str) -> impl Iterator<Item = &str> {
    alleles.split(',').map(str::trim).filter(|s| !s.is_empty())
}

// Same logic as generate_multihetsep.py:is_segregating
fn is_segregating(alleles: &str) -> bool {
    for order in split_orders(alleles) {
        let mut it = order.chars();
        let Some(first) = it.next() else {
            continue;
        };
        if it.any(|c| c != first) {
            return true;
        }
    }
    false
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
        bail!("no valid mhs rows found in input");
    }
    Ok(Observations { rows, row_starts })
}

pub fn read_mhs(path: &Path, batch_size: Option<usize>, bin_size: usize) -> Result<Observations> {
    if bin_size == 0 {
        bail!("mhs bin_size must be > 0");
    }

    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut row_no = 0usize;
    let mut n_rows = 0usize;

    let mut seqs: Vec<Vec<u8>> = Vec::new();
    let mut cur_seq: Vec<u8> = Vec::new();
    let mut cur_bin = BinAcc::default();
    let mut cur_chrom: Option<String> = None;
    let mut prev_pos: u64 = 0;

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
        let chrom = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing chrom"))?
            .to_string();
        let pos_str = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing pos"))?;
        let called_str = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing nr_called"))?;
        let alleles = cols
            .next()
            .with_context(|| format!("invalid mhs row {row_no}: missing alleles"))?;

        let pos: u64 = pos_str
            .parse()
            .with_context(|| format!("invalid mhs row {row_no}: bad pos '{pos_str}'"))?;
        if pos == 0 {
            bail!("invalid mhs row {row_no}: pos must be >= 1");
        }
        let called: u64 = called_str
            .parse()
            .with_context(|| format!("invalid mhs row {row_no}: bad nr_called '{called_str}'"))?;
        if called == 0 {
            bail!("invalid mhs row {row_no}: nr_called must be > 0");
        }

        let is_new_chrom = match &cur_chrom {
            None => true,
            Some(c) => c != &chrom,
        };
        if is_new_chrom {
            if cur_chrom.is_some() {
                flush_tail_bin(&mut cur_seq, &mut cur_bin, bin_size);
                if !cur_seq.is_empty() {
                    seqs.push(std::mem::take(&mut cur_seq));
                }
            }
            cur_chrom = Some(chrom);
            prev_pos = 0;
            cur_bin = BinAcc::default();
        }

        if pos <= prev_pos {
            bail!("invalid mhs row {row_no}: pos must increase within chromosome");
        }
        let delta = pos - prev_pos;
        if called > delta {
            bail!(
                "invalid mhs row {row_no}: nr_called ({called}) cannot exceed distance ({delta})"
            );
        }

        // multihetsep semantics:
        // - distance between variants is "delta"
        // - among them, "called" sites are callable
        // - one callable site is current SNP position
        // - non-callable count = delta - called
        let n_uncalled = delta - called;
        let n_called_non_variant = called - 1;
        add_run_n(&mut cur_seq, n_uncalled, &mut cur_bin, bin_size);
        add_run_t(&mut cur_seq, n_called_non_variant, &mut cur_bin, bin_size);

        // Official multihetsep rows are segregating variant sites.
        if !is_segregating(alleles) {
            bail!("invalid mhs row {row_no}: alleles must describe a segregating site");
        }
        add_k(&mut cur_seq, &mut cur_bin, bin_size);

        prev_pos = pos;
        n_rows += 1;
    }

    if n_rows == 0 {
        bail!("no valid mhs rows found in input");
    }

    flush_tail_bin(&mut cur_seq, &mut cur_bin, bin_size);
    if !cur_seq.is_empty() {
        seqs.push(cur_seq);
    }

    chunk_sequences(seqs, batch_size)
}
