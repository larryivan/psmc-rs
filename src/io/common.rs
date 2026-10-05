//! Helpers shared by the input readers: transparent (b)gzip decoding,
//! `fq2psmcfa`-style binning, and splitting sequences into HMM rows.

use anyhow::{Context, Result, bail};
use flate2::read::MultiGzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::Path;

use crate::io::Observations;

/// Open a text input for line-oriented reading.
///
/// Gzip input is detected from the magic bytes rather than the file extension,
/// and every gzip member is decoded. This matters for bgzip files (the usual
/// `.vcf.gz` written by bcftools, GATK and tabix), which are a chain of
/// independent members: a single-member decoder silently stops after the first
/// block, which for current bgzip is often just the VCF header.
pub fn open_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    decoding_reader(file, path)
}

/// Open `path` again, from its start, for another pass over it. Only regular
/// files can be read twice ([`is_regular_file`]). The rewind matters for a
/// path such as `/dev/stdin` redirected from a file, which macOS reopens at
/// the offset where the previous pass stopped.
pub(crate) fn reopen_reader(path: &Path) -> Result<Box<dyn BufRead>> {
    let mut file = File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    file.rewind()
        .with_context(|| format!("failed to rewind {path:?}"))?;
    decoding_reader(file, path)
}

/// Whether `path` is a regular file, which, unlike a pipe, FIFO or terminal,
/// can be read more than once.
pub(crate) fn is_regular_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file())
}

fn decoding_reader(file: File, path: &Path) -> Result<Box<dyn BufRead>> {
    let mut buf = BufReader::new(file);
    let is_gzip = {
        let head = buf
            .fill_buf()
            .with_context(|| format!("failed to read {path:?}"))?;
        head.len() >= 2 && head[0] == 0x1f && head[1] == 0x8b
    };
    if is_gzip {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(buf))))
    } else {
        Ok(Box::new(buf))
    }
}

/// Accumulates per-site calls into fixed-size bins with the `fq2psmcfa` rule:
/// a bin is `N` (2) when more than 90% of its sites are missing, otherwise `K`
/// (1) if it contains a heterozygous site, otherwise `T` (0).
///
/// As in `fq2psmcfa`, a trailing partial bin is judged against the full bin size.
#[derive(Debug, Clone)]
pub struct Binner {
    bin_size: usize,
    seq: Vec<u8>,
    len: usize,
    n_missing: usize,
    has_het: bool,
}

impl Binner {
    pub fn new(bin_size: usize) -> Self {
        assert!(bin_size > 0, "bin_size must be > 0");
        Self {
            bin_size,
            seq: Vec::new(),
            len: 0,
            n_missing: 0,
            has_het: false,
        }
    }

    pub fn bin_size(&self) -> usize {
        self.bin_size
    }

    /// Number of finished bins emitted so far.
    pub fn n_bins(&self) -> usize {
        self.seq.len()
    }

    /// Whether no site has been added since the last [`Binner::finish`].
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty() && self.len == 0
    }

    fn emit(&mut self) {
        // Exact form of `n_missing / bin_size > 0.9`.
        let sym = if 10 * self.n_missing > 9 * self.bin_size {
            2
        } else if self.has_het {
            1
        } else {
            0
        };
        self.seq.push(sym);
        self.len = 0;
        self.n_missing = 0;
        self.has_het = false;
    }

    fn add_run(&mut self, mut n: u64, missing: bool) {
        while n > 0 {
            let room = (self.bin_size - self.len) as u64;
            let take = n.min(room) as usize;
            self.len += take;
            if missing {
                self.n_missing += take;
            }
            n -= take as u64;
            if self.len == self.bin_size {
                self.emit();
            }
        }
    }

    /// Add `n` called, non-heterozygous sites.
    pub fn add_called(&mut self, n: u64) {
        self.add_run(n, false);
    }

    /// Add `n` missing (uncalled) sites.
    pub fn add_missing(&mut self, n: u64) {
        self.add_run(n, true);
    }

    /// Add one called heterozygous site.
    pub fn add_het(&mut self) {
        self.len += 1;
        self.has_het = true;
        if self.len == self.bin_size {
            self.emit();
        }
    }

    /// Flush the trailing partial bin (if any) and return the binned sequence,
    /// leaving the binner empty and ready for the next sequence.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.len > 0 {
            self.emit();
        }
        std::mem::take(&mut self.seq)
    }
}

/// A heterozygous site found by a reader, with multihetsep's `nr_called`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HetCall {
    /// 1-based position.
    pub pos: u64,
    /// Called sites since the previous heterozygous site of the sequence
    /// (exclusive) up to and including this one.
    pub nr_called: u64,
    pub alleles: [u8; 2],
}

/// Follows the called sites of one sequence, in position order, and records
/// its heterozygous sites with their `nr_called` when `collect` is set.
#[derive(Debug, Clone, Default)]
pub(crate) struct HetTracker {
    collect: bool,
    called_since: u64,
    sites: Vec<HetCall>,
}

impl HetTracker {
    pub fn new(collect: bool) -> Self {
        Self {
            collect,
            ..Self::default()
        }
    }

    /// `n` called sites that are not heterozygous.
    pub fn called(&mut self, n: u64) {
        self.called_since += n;
    }

    /// One called heterozygous site.
    pub fn het(&mut self, pos: u64, alleles: [u8; 2]) {
        if self.collect {
            self.sites.push(HetCall {
                pos,
                nr_called: self.called_since + 1,
                alleles,
            });
        }
        self.called_since = 0;
    }

    /// The recorded sites; the tracker starts over for the next sequence.
    pub fn take(&mut self) -> Vec<HetCall> {
        self.called_since = 0;
        std::mem::take(&mut self.sites)
    }
}

/// Turn whole sequences into HMM rows. With `batch_size`, long sequences are
/// split into chunks that stay connected in one HMM chain (`row_starts` is
/// `true` only for the first chunk of each sequence). Empty sequences are dropped.
pub fn chunk_sequences(
    seqs: Vec<Vec<u8>>,
    batch_size: Option<usize>,
    empty_msg: &str,
) -> Result<Observations> {
    if batch_size == Some(0) {
        bail!("batch_size must be > 0");
    }
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();
    for seq in seqs {
        if seq.is_empty() {
            continue;
        }
        match batch_size {
            None => {
                rows.push(seq);
                row_starts.push(true);
            }
            Some(batch) => {
                for (i, chunk) in seq.chunks(batch).enumerate() {
                    rows.push(chunk.to_vec());
                    row_starts.push(i == 0);
                }
            }
        }
    }
    if rows.is_empty() {
        bail!("{empty_msg}");
    }
    Ok(Observations { rows, row_starts })
}
