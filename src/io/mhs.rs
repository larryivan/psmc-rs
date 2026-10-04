//! Reader for MSMC's multihetsep format (as written by `generate_multihetsep.py`),
//! binned into PSMC observations for one pair of haplotypes.

use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::io::Observations;
use crate::io::common::{Binner, chunk_sequences, open_reader};

/// Options for [`read_mhs_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MhsOptions {
    /// Sites per PSMC bin (`fq2psmcfa -s`); must be > 0.
    pub bin_size: usize,
    /// Split each chromosome into HMM rows of at most this many bins that stay
    /// chained in one HMM sequence; `None` keeps one row per chromosome.
    pub batch_size: Option<usize>,
    /// The two haplotypes whose pairwise coalescence PSMC models, as 0-based
    /// positions in each allele ordering. The default `(0, 1)` is the first
    /// diploid sample.
    pub pair: (usize, usize),
}

impl Default for MhsOptions {
    fn default() -> Self {
        Self {
            bin_size: 100,
            batch_size: None,
            pair: (0, 1),
        }
    }
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

/// What one multihetsep row says about the selected pair of haplotypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairCall {
    Het,
    Hom,
    Missing,
}

fn is_unknown_allele(a: u8) -> bool {
    matches!(a, b'?' | b'N' | b'n')
}

/// The pair `(i, j)` in one allele ordering; both indices are in bounds.
fn classify(haps: &[u8], i: usize, j: usize) -> PairCall {
    if is_unknown_allele(haps[i]) || is_unknown_allele(haps[j]) {
        PairCall::Missing
    } else if haps[i] != haps[j] {
        PairCall::Het
    } else {
        PairCall::Hom
    }
}

/// Classify the pair `(i, j)` at a row from its alleles column: heterozygous
/// or homozygous when every ordering agrees, missing when the orderings
/// disagree (the phase of unphased samples decides) or an allele is unknown.
///
/// `file_haps` is the haplotype count of the file's first row and that row's
/// number; the first call sets it, and every later row must match it.
fn pair_call(
    alleles: &str,
    (i, j): (usize, usize),
    row_no: usize,
    file_haps: &mut Option<(usize, usize)>,
) -> Result<PairCall> {
    let mut orders = split_orders(alleles);
    let Some(first) = orders.next() else {
        bail!("invalid mhs row {row_no}: no allele ordering in '{alleles}'");
    };
    let n_haps = first.len();
    let (want, first_row) = *file_haps.get_or_insert((n_haps, row_no));
    // Checked before the pair so that a widened row is reported as such,
    // whatever the pair.
    if n_haps != want {
        bail!(
            "invalid mhs row {row_no}: alleles '{first}' list {n_haps} haplotypes, but row \
             {first_row} lists {want}; multihetsep has one character per haplotype, so one of \
             the two rows probably holds a multi-character (indel) allele: keep only SNPs in \
             the input VCFs"
        );
    }
    // Compared without adding 1, which would overflow for an index of
    // usize::MAX.
    let hi = i.max(j);
    if n_haps <= hi {
        bail!(
            "invalid mhs row {row_no}: alleles '{first}' list {n_haps} haplotypes, but haplotype \
             pair ({i},{j}) needs at least {}",
            hi as u128 + 1
        );
    }
    let mut call = classify(first.as_bytes(), i, j);
    for order in orders {
        if order.len() != n_haps {
            bail!("invalid mhs row {row_no}: allele orderings in '{alleles}' differ in length");
        }
        if classify(order.as_bytes(), i, j) != call {
            call = PairCall::Missing;
        }
    }
    // Official multihetsep rows are segregating variant sites.
    if !is_segregating(alleles) {
        bail!("invalid mhs row {row_no}: alleles must describe a segregating site");
    }
    Ok(call)
}

/// Read a multihetsep file with default [`MhsOptions`] except for `batch_size`
/// and `bin_size`, i.e. for haplotypes 0 and 1 (the first diploid sample).
/// See [`read_mhs_with`] for the semantics, including how uncalled sites are
/// placed.
pub fn read_mhs(path: &Path, batch_size: Option<usize>, bin_size: usize) -> Result<Observations> {
    read_mhs_with(
        path,
        &MhsOptions {
            bin_size,
            batch_size,
            ..MhsOptions::default()
        },
    )
}

/// Read a multihetsep file (MSMC input, optionally gzip-compressed) as PSMC
/// observations for the haplotype pair `opts.pair`.
///
/// Each row `chrom pos nr_called alleles` is a segregating site at 1-based
/// `pos`; `nr_called` counts the callable sites since the previous row of the
/// chromosome, this one included. The other callable sites are called and not
/// heterozygous, since only segregating sites get a row. The alleles column
/// holds one or more comma-separated orderings of the haplotypes
/// (`generate_multihetsep.py` lists the possible phasings of unphased
/// samples); the pair is read from each ordering and the site is
/// - heterozygous when the two haplotypes differ in every ordering,
/// - called and not heterozygous when they are equal in every ordering,
/// - missing when the orderings disagree, which happens when the pair spans
///   two unphased samples so that phase decides, or when either allele is
///   unknown (`?` or `N`).
///
/// Multihetsep carries no positions for uncalled sites, only how many there
/// are per interval, so they are placed at the start of each interval, right
/// after the previous row. This assumes missingness is clustered, and bins
/// match those of `fq2psmcfa` exactly only where that holds. Exact binning
/// needs per-site calls, i.e. a VCF plus callability mask
/// ([`crate::io::vcf::read_vcf`]).
///
/// A new value in the first column starts a new sequence. Rows must have
/// increasing positions within a chromosome, `1 <= nr_called <= distance to
/// the previous row`, and alleles segregating across all haplotypes (as in
/// files from `generate_multihetsep.py`); the pair must name two different
/// haplotypes. Every ordering of every row must list the same number of
/// haplotypes as the first row, and more than `max(pair)`. An ordering has one
/// character per haplotype, and `generate_multihetsep.py` widens the row of an
/// indel (it joins multi-character VCF alleles), which would shift the pair.
pub fn read_mhs_with(path: &Path, opts: &MhsOptions) -> Result<Observations> {
    let MhsOptions {
        bin_size,
        batch_size,
        pair,
    } = *opts;
    if bin_size == 0 {
        bail!("mhs bin_size must be > 0");
    }
    if pair.0 == pair.1 {
        bail!(
            "mhs haplotype pair must name two different haplotypes, got ({},{})",
            pair.0,
            pair.1
        );
    }

    let mut reader = open_reader(path)?;
    let mut line = String::new();
    let mut row_no = 0usize;
    let mut n_rows = 0usize;

    let mut seqs: Vec<Vec<u8>> = Vec::new();
    let mut binner = Binner::new(bin_size);
    let mut cur_chrom: Option<String> = None;
    let mut prev_pos: u64 = 0;
    let mut file_haps = None;

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
            .with_context(|| format!("invalid mhs row {row_no}: missing chrom"))?;
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

        if cur_chrom.as_deref() != Some(chrom) {
            if cur_chrom.is_some() {
                let seq = binner.finish();
                if !seq.is_empty() {
                    seqs.push(seq);
                }
            }
            cur_chrom = Some(chrom.to_string());
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
        let call = pair_call(alleles, pair, row_no, &mut file_haps)?;

        // Of the `delta` sites up to and including this row, `called` are
        // callable; the uncalled ones go first (see the doc comment).
        binner.add_missing(delta - called);
        binner.add_called(called - 1);
        match call {
            PairCall::Het => binner.add_het(),
            PairCall::Hom => binner.add_called(1),
            PairCall::Missing => binner.add_missing(1),
        }

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
