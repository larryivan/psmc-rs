//! Reader for MSMC's multihetsep format (as written by `generate_multihetsep.py`),
//! binned into PSMC observations for one pair of haplotypes.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::io::BufRead;
use std::path::Path;

use crate::io::callable::check_bin_size;
use crate::io::common::{Binner, HetCall, HetTracker, chunk_sequences, open_reader};
use crate::io::{CallableBases, Observations, Sequences};

/// Options for [`read_mhs_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MhsOptions {
    /// Sites per PSMC bin (`fq2psmcfa -s`); between 1 and
    /// [`MAX_BIN_SIZE`](crate::io::MAX_BIN_SIZE).
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

// A row is segregating when any ordering holds two different alleles, as in
// generate_multihetsep.py:is_segregating.
fn has_two_alleles(order: &str) -> bool {
    let mut it = order.chars();
    let first = it.next();
    it.any(|c| Some(c) != first)
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
/// Also returns the pair's alleles in the first ordering.
///
/// `file_haps` is the haplotype count of the file's first row and that row's
/// number; the first call sets it, and every later row must match it.
fn pair_call(
    alleles: &str,
    (i, j): (usize, usize),
    row_no: usize,
    file_haps: &mut Option<(usize, usize)>,
) -> Result<(PairCall, [u8; 2])> {
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
    let haps = first.as_bytes();
    let pair_alleles = [haps[i], haps[j]];
    let mut call = classify(haps, i, j);
    let mut segregating = has_two_alleles(first);
    for order in orders {
        if order.len() != n_haps {
            bail!("invalid mhs row {row_no}: allele orderings in '{alleles}' differ in length");
        }
        if classify(order.as_bytes(), i, j) != call {
            call = PairCall::Missing;
        }
        segregating = segregating || has_two_alleles(order);
    }
    // Official multihetsep rows are segregating variant sites.
    if !segregating {
        bail!("invalid mhs row {row_no}: alleles must describe a segregating site");
    }
    Ok((call, pair_alleles))
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
/// The observations carry the callable bases of each bin
/// ([`Observations::callable`]), the bin size minus its missing sites under
/// the same placement. They are approximate in the same way: the callable
/// sites of each interval are counted exactly, but their split between the
/// interval's bins is the placement's.
///
/// A new value in the first column starts a new sequence, and the rows of a
/// chromosome must form one block: a chromosome that comes back after another
/// one is an error, since its `nr_called` would be counted from the start of
/// the chromosome rather than from its previous row. Rows must have
/// increasing positions within a chromosome, `1 <= nr_called <= distance to
/// the previous row`, and alleles segregating across all haplotypes (as in
/// files from `generate_multihetsep.py`); the pair must name two different
/// haplotypes. Every ordering of every row must list the same number of
/// haplotypes as the first row, and more than `max(pair)`. An ordering has one
/// character per haplotype, and `generate_multihetsep.py` widens the row of an
/// indel (it joins multi-character VCF alleles), which would shift the pair.
pub fn read_mhs_with(path: &Path, opts: &MhsOptions) -> Result<Observations> {
    let (seqs, _) = read_mhs_seqs(path, opts, false)?;
    let (bins, counts) = seqs.into_iter().map(|s| (s.bins, s.callable)).unzip();
    let seqs = Sequences {
        bins,
        callable: Some(CallableBases::new(opts.bin_size, counts)?),
    };
    chunk_sequences(seqs, opts.batch_size, "no valid mhs rows found in input")
}

/// Counts over a whole multihetsep file, for the selected pair.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct MhsSummary {
    /// Data rows.
    pub rows: u64,
    /// Haplotypes per allele ordering (from the first row).
    pub haplotypes: usize,
    /// Rows where the pair is heterozygous.
    pub het_rows: u64,
    /// Rows where the pair is homozygous (the site segregates elsewhere).
    pub hom_rows: u64,
    /// Rows where the pair is missing (phase-dependent or unknown allele).
    pub missing_rows: u64,
    /// Sites before each row that `nr_called` leaves uncalled.
    pub uncalled_sites: u64,
}

/// One chromosome of a multihetsep file, binned for the selected pair.
#[derive(Debug, Clone)]
pub(crate) struct MhsSeq {
    pub name: String,
    pub bins: Vec<u8>,
    /// The callable bases of each bin: the bin size minus the bin's sites
    /// missing for the pair, with the uncalled sites placed at the start of
    /// each interval ([`read_mhs_with`]).
    pub callable: Vec<u32>,
    /// Position of the last row: the sequence covers `1..=extent`.
    pub extent: u64,
    /// Sites binned as called for the pair.
    pub called: u64,
    /// Rows where the pair is heterozygous.
    pub n_het: u64,
    /// The heterozygous sites, when they were asked for.
    pub het_sites: Vec<HetCall>,
}

struct MhsChrom {
    name: String,
    binner: Binner,
    prev_pos: u64,
    called: u64,
    n_het: u64,
    het: HetTracker,
}

impl MhsChrom {
    fn finish(mut self) -> MhsSeq {
        let (bins, callable) = self.binner.finish_with_callable();
        MhsSeq {
            name: self.name,
            bins,
            callable,
            extent: self.prev_pos,
            called: self.called,
            n_het: self.n_het,
            het_sites: self.het.take(),
        }
    }
}

/// Read every chromosome of a multihetsep file as [`read_mhs_with`] does,
/// without chunking. With `collect_het`, each sequence carries the sites
/// where the pair is heterozygous, with `nr_called` counted over the sites
/// binned as called for the pair.
pub(crate) fn read_mhs_seqs(
    path: &Path,
    opts: &MhsOptions,
    collect_het: bool,
) -> Result<(Vec<MhsSeq>, MhsSummary)> {
    check_options(opts)?;
    read_mhs_seqs_from(open_reader(path)?, path, opts, collect_het).map_err(|e| e.error)
}

fn check_options(opts: &MhsOptions) -> Result<()> {
    check_bin_size(opts.bin_size, "mhs")?;
    if opts.pair.0 == opts.pair.1 {
        bail!(
            "mhs haplotype pair must name two different haplotypes, got ({},{})",
            opts.pair.0,
            opts.pair.1
        );
    }
    Ok(())
}

/// A failure of [`read_mhs_seqs_from`].
#[derive(Debug)]
pub(crate) struct MhsReadError {
    pub error: anyhow::Error,
    /// Whether the input may not be multihetsep at all, which matters when
    /// its format was guessed: the error is in the first data row, the
    /// input is not text, a row has fewer than the four columns, or no row
    /// was found. Errors in later rows of a file that started as
    /// multihetsep (positions out of order, a chromosome that comes back)
    /// are errors of a multihetsep file.
    pub maybe_other_format: bool,
}

/// [`read_mhs_seqs`] over an opened input; `path` names it in messages.
pub(crate) fn read_mhs_seqs_from(
    reader: impl BufRead,
    path: &Path,
    opts: &MhsOptions,
    collect_het: bool,
) -> Result<(Vec<MhsSeq>, MhsSummary), MhsReadError> {
    let mut maybe_other_format = false;
    read_rows(reader, path, opts, collect_het, &mut maybe_other_format).map_err(|error| {
        MhsReadError {
            error,
            maybe_other_format,
        }
    })
}

/// [`read_mhs_seqs_from`]; sets `maybe_other_format` as
/// [`MhsReadError::maybe_other_format`] says before it fails.
fn read_rows(
    mut reader: impl BufRead,
    path: &Path,
    opts: &MhsOptions,
    collect_het: bool,
    maybe_other_format: &mut bool,
) -> Result<(Vec<MhsSeq>, MhsSummary)> {
    check_options(opts)?;
    let MhsOptions { bin_size, pair, .. } = *opts;
    let mut line = String::new();
    let mut row_no = 0usize;
    let mut summary = MhsSummary::default();

    let mut seqs: Vec<MhsSeq> = Vec::new();
    let mut cur: Option<MhsChrom> = None;
    let mut finished = HashSet::<String>::new();
    let mut file_haps = None;

    loop {
        line.clear();
        // Text that is not UTF-8 from the start is not multihetsep; a read
        // error after a data row (e.g. a truncated bgzip) is one of a
        // multihetsep file.
        *maybe_other_format = summary.rows == 0;
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

        // A row without four columns may be a file of another format,
        // wherever it is.
        *maybe_other_format = true;
        let cols: Vec<&str> = trimmed.split_whitespace().take(4).collect();
        let &[chrom, pos_str, called_str, alleles] = cols.as_slice() else {
            let missing = ["chrom", "pos", "nr_called", "alleles"][cols.len()];
            bail!("invalid mhs row {row_no}: missing {missing}");
        };
        // From here on, an error in any row but the first is one of a
        // multihetsep file.
        *maybe_other_format = summary.rows == 0;

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

        if cur.as_ref().is_none_or(|c| c.name != chrom) {
            if let Some(done) = cur.take() {
                finished.insert(done.name.clone());
                seqs.push(done.finish());
            }
            if finished.contains(chrom) {
                bail!(
                    "invalid mhs row {row_no}: chromosome '{chrom}' appears again after another \
                     chromosome; the rows of each chromosome must form one block in position \
                     order (concatenate whole per-chromosome files, or sort the rows with \
                     sort -k1,1 -k2,2n)"
                );
            }
            cur = Some(MhsChrom {
                name: chrom.to_string(),
                binner: Binner::new(bin_size),
                prev_pos: 0,
                called: 0,
                n_het: 0,
                het: HetTracker::new(collect_het),
            });
        }
        let c = cur.as_mut().expect("chromosome state was just set");

        if pos <= c.prev_pos {
            bail!("invalid mhs row {row_no}: pos must increase within chromosome");
        }
        let delta = pos - c.prev_pos;
        if called > delta {
            bail!(
                "invalid mhs row {row_no}: nr_called ({called}) cannot exceed distance ({delta})"
            );
        }
        let (call, pair_alleles) = pair_call(alleles, pair, row_no, &mut file_haps)?;

        // Of the `delta` sites up to and including this row, `called` are
        // callable; the uncalled ones go first (see the doc comment).
        c.binner.add_missing(delta - called);
        c.binner.add_called(called - 1);
        c.het.called(called - 1);
        c.called += called - 1;
        summary.uncalled_sites += delta - called;
        match call {
            PairCall::Het => {
                c.binner.add_het();
                c.het.het(pos, pair_alleles);
                c.called += 1;
                c.n_het += 1;
                summary.het_rows += 1;
            }
            PairCall::Hom => {
                c.binner.add_called(1);
                c.het.called(1);
                c.called += 1;
                summary.hom_rows += 1;
            }
            PairCall::Missing => {
                c.binner.add_missing(1);
                summary.missing_rows += 1;
            }
        }

        c.prev_pos = pos;
        summary.rows += 1;
    }

    if summary.rows == 0 {
        *maybe_other_format = true;
        bail!("no valid mhs rows found in input");
    }
    if let Some(done) = cur.take() {
        seqs.push(done.finish());
    }
    summary.haplotypes = file_haps.map_or(0, |(n, _)| n);
    Ok((seqs, summary))
}
