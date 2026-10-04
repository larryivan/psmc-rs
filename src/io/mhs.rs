use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::io::Observations;
use crate::io::common::{Binner, chunk_sequences, open_reader};

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

pub fn read_mhs(path: &Path, batch_size: Option<usize>, bin_size: usize) -> Result<Observations> {
    if bin_size == 0 {
        bail!("mhs bin_size must be > 0");
    }

    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut row_no = 0usize;
    let mut n_rows = 0usize;

    let mut seqs: Vec<Vec<u8>> = Vec::new();
    let mut binner = Binner::new(bin_size);
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
                let seq = binner.finish();
                if !seq.is_empty() {
                    seqs.push(seq);
                }
            }
            cur_chrom = Some(chrom);
            prev_pos = 0;
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
        binner.add_missing(n_uncalled);
        binner.add_called(n_called_non_variant);

        // Official multihetsep rows are segregating variant sites.
        if !is_segregating(alleles) {
            bail!("invalid mhs row {row_no}: alleles must describe a segregating site");
        }
        binner.add_het();

        prev_pos = pos;
        n_rows += 1;
    }

    if n_rows == 0 {
        bail!("no valid mhs rows found in input");
    }

    let seq = binner.finish();
    if !seq.is_empty() {
        seqs.push(seq);
    }

    chunk_sequences(seqs, batch_size, "no valid mhs rows found in input")
}
