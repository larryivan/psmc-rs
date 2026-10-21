//! Helpers shared by the input readers: transparent (b)gzip decoding,
//! `fq2psmcfa`-style binning, depth statistics, and splitting sequences into
//! HMM rows.

use anyhow::{Context, Result, bail};
use flate2::read::MultiGzDecoder;
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read, Seek};
use std::path::Path;
use std::sync::mpsc::{Receiver, sync_channel};

use crate::io::callable::{CountsBuilder, counted_bin_size};
use crate::io::{BinCounts, CallableBases, Observations, Sequences};

/// Open a text input for line-oriented reading.
///
/// Gzip input is detected from the magic bytes rather than the file extension,
/// and every gzip member is decoded. This matters for bgzip files (the usual
/// `.vcf.gz` written by bcftools, GATK and tabix), which are a chain of
/// independent members: a single-member decoder silently stops after the first
/// block, which for current bgzip is often just the VCF header.
///
/// A BGZF file must end with BGZF's empty end-of-file block, as htslib
/// checks: reading one without it to the end is an error, because a bgzip
/// file cut at a block boundary (by an interrupted job) decodes cleanly and
/// would otherwise pass for the whole input. Gzip input is decoded in a
/// thread of its own while it is read.
///
/// The decoded text must not be BCF, which is refused with a hint to
/// convert it; a leading UTF-8 byte order mark is dropped.
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

/// The empty block that ends every complete BGZF (bgzip) file, as htslib
/// writes it and checks for it (SAM/BAM format specification, 4.1.2).
const BGZF_EOF: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, b'B', b'C', 0x02, 0, 0x1b, 0, 0x03, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// Bytes of text read at a time: large enough that reading a big file, or
/// a pipe, takes few system calls and decoder calls.
const READ_BUFFER: usize = 128 << 10;

fn decoding_reader(mut file: File, path: &Path) -> Result<Box<dyn BufRead>> {
    // The gzip header up to the first extra subfield's identifier, which is
    // `BC` in a BGZF block. A pipe may deliver it in several reads.
    let mut head = Vec::with_capacity(14);
    (&mut file)
        .take(14)
        .read_to_end(&mut head)
        .with_context(|| format!("failed to read {path:?}"))?;
    let is_gzip = head.starts_with(&[0x1f, 0x8b]);
    let is_bgzf = is_gzip && head.len() == 14 && head[3] & 4 != 0 && head[12..14] == *b"BC";
    let raw = Cursor::new(head).chain(file);
    let decoded: Box<dyn BufRead> = if is_bgzf {
        Box::new(DecoderThread::spawn(BgzfReader {
            decoder: MultiGzDecoder::new(TailKeeper::new(raw)),
        })?)
    } else if is_gzip {
        Box::new(DecoderThread::spawn(MultiGzDecoder::new(raw))?)
    } else {
        Box::new(BufReader::with_capacity(READ_BUFFER, raw))
    };
    check_text_start(decoded, path)
}

/// Refuse BCF, which every reader would otherwise misread as text, and
/// drop a leading UTF-8 byte order mark, which some editors write and which
/// would otherwise become part of the first line (C psmc skips it in a
/// psmcfa file, as it skips everything before the first `>`).
fn check_text_start(mut reader: Box<dyn BufRead>, path: &Path) -> Result<Box<dyn BufRead>> {
    let mut head = Vec::with_capacity(4);
    (&mut reader)
        .take(4)
        .read_to_end(&mut head)
        .with_context(|| format!("failed to read {path:?}"))?;
    // BCF 2 starts with "BCF\2", the old BCF 1 with "BCF\4".
    if head.starts_with(b"BCF") && matches!(head.get(3), Some(1 | 2 | 4)) {
        bail!(
            "{path:?} is a BCF file, which psmc-rs does not read; convert it to a VCF \
             first: bcftools view -Oz -o OUT.vcf.gz {}",
            path.display()
        );
    }
    let start = if head.starts_with(b"\xef\xbb\xbf") {
        3
    } else {
        0
    };
    let mut head = Cursor::new(head);
    head.set_position(start);
    Ok(Box::new(head.chain(reader)))
}

/// Decodes gzip in a thread of its own, a few chunks ahead of the reader,
/// so that decoding and parsing a compressed input run side by side (a
/// third faster for a large bgzip VCF). The reader gets the same bytes and
/// errors, in the same order, as from the decoder itself.
struct DecoderThread {
    /// Decoded chunks, an empty one at the end.
    chunks: Receiver<io::Result<Vec<u8>>>,
    chunk: Vec<u8>,
    /// Bytes of `chunk` already read.
    pos: usize,
    /// Whether the end of the data was reached.
    done: bool,
}

impl DecoderThread {
    fn spawn(mut decoder: impl Read + Send + 'static) -> io::Result<Self> {
        let (tx, rx) = sync_channel(4);
        std::thread::Builder::new()
            .name("gzip decoder".to_string())
            .spawn(move || {
                loop {
                    let mut chunk = Vec::with_capacity(READ_BUFFER);
                    let read = (&mut decoder)
                        .take(READ_BUFFER as u64)
                        .read_to_end(&mut chunk);
                    // A failed send means the reader stopped early.
                    let more = match read {
                        Ok(n) => tx.send(Ok(chunk)).is_ok() && n > 0,
                        // The bytes decoded before the error go first.
                        Err(e) => {
                            if !chunk.is_empty() {
                                let _ = tx.send(Ok(chunk));
                            }
                            let _ = tx.send(Err(e));
                            false
                        }
                    };
                    if !more {
                        break;
                    }
                }
            })?;
        Ok(Self {
            chunks: rx,
            chunk: Vec::new(),
            pos: 0,
            done: false,
        })
    }
}

impl Read for DecoderThread {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.fill_buf()?.read(buf)?;
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for DecoderThread {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos == self.chunk.len() && !self.done {
            // The decoder's thread ends after sending an error, or panics.
            let chunk = self
                .chunks
                .recv()
                .map_err(|_| io::Error::other("gzip decoding stopped"))??;
            self.done = chunk.is_empty();
            (self.chunk, self.pos) = (chunk, 0);
        }
        Ok(&self.chunk[self.pos..])
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
    }
}

/// The lines of a text, as `read_until(b'\n')` gives them (the last may
/// lack its `\n`), read 1 MiB at a time and handed out in place: only a
/// line that straddles two reads is copied.
pub(crate) struct LineReader<R> {
    reader: BufReader<R>,
    /// The current line when it straddled two reads.
    copied: Vec<u8>,
    /// The length of the current line when it is at the start of the
    /// reader's buffer (consumed at the next [`Self::advance`]), else 0.
    in_place: usize,
}

impl<R: Read> LineReader<R> {
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(1 << 20, reader),
            copied: Vec::new(),
            in_place: 0,
        }
    }

    /// Move to the next line; false at the end of the text.
    pub(crate) fn advance(&mut self) -> io::Result<bool> {
        self.reader.consume(std::mem::take(&mut self.in_place));
        let newline = loop {
            match self.reader.fill_buf() {
                Ok(buf) => break memchr::memchr(b'\n', buf),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        };
        if let Some(i) = newline {
            self.in_place = i + 1;
            return Ok(true);
        }
        self.copied.clear();
        Ok(self.reader.read_until(b'\n', &mut self.copied)? > 0)
    }

    /// The current line.
    pub(crate) fn line(&self) -> &[u8] {
        match self.in_place {
            0 => &self.copied,
            n => &self.reader.buffer()[..n],
        }
    }
}

/// A reader that remembers the last [`BGZF_EOF`]`.len()` bytes it passed on.
struct TailKeeper<R> {
    inner: R,
    tail: [u8; BGZF_EOF.len()],
    /// Bytes of `tail` that have been filled.
    filled: usize,
}

impl<R> TailKeeper<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            tail: [0; BGZF_EOF.len()],
            filled: 0,
        }
    }

    fn ends_with_bgzf_eof(&self) -> bool {
        self.filled == BGZF_EOF.len() && self.tail == BGZF_EOF
    }
}

impl<R: Read> Read for TailKeeper<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        let k = n.min(self.tail.len());
        self.tail.copy_within(k.., 0);
        let keep = self.tail.len() - k;
        self.tail[keep..].copy_from_slice(&buf[n - k..n]);
        self.filled = (self.filled + k).min(self.tail.len());
        Ok(n)
    }
}

/// Decodes a BGZF file and, at its end, checks for the empty block that
/// ends a complete one. bgzip and bcftools write whole blocks, so a file
/// left by an interrupted job usually ends at a block boundary and decodes
/// without error; only the missing end-of-file block shows that the rest of
/// the genome is gone. htslib warns about it, and bcftools refuses such a
/// file; reading on would silently analyse part of the input.
struct BgzfReader<R> {
    decoder: MultiGzDecoder<TailKeeper<R>>,
}

impl<R: Read> Read for BgzfReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.decoder.read(buf)?;
        if n == 0 && !buf.is_empty() && !self.decoder.get_ref().ends_with_bgzf_eof() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the bgzip (BGZF) data end without the end-of-file marker block that ends \
                 every complete BGZF file, so the file is probably truncated, for example by \
                 an interrupted bcftools or bgzip run (bcftools refuses it too); create it \
                 again",
            ));
        }
        Ok(n)
    }
}

/// Accumulates per-site calls into fixed-size bins with the `fq2psmcfa` rule:
/// a bin is `N` (2) when more than 90% of its sites are missing, otherwise `K`
/// (1) if it contains a heterozygous site, otherwise `T` (0).
///
/// As in `fq2psmcfa`, a trailing partial bin is judged against the full bin size.
///
/// It also counts the callable bases of each bin
/// ([`Binner::finish_with_callable`]) when the bin size is at most
/// [`MAX_COUNTED_BIN_SIZE`](crate::io::MAX_COUNTED_BIN_SIZE): the bin size
/// minus the bin's missing sites, so that the sites past the end of a
/// trailing partial bin count as callable, as the rule above treats them
/// (see [`crate::io::callable`]).
#[derive(Debug, Clone)]
pub struct Binner {
    bin_size: usize,
    seq: Vec<u8>,
    /// The callable bases of the bins emitted, when bins of this size are
    /// counted.
    callable: Option<CountsBuilder>,
    len: usize,
    n_missing: usize,
    has_het: bool,
}

impl Binner {
    /// A binner for bins of `bin_size` sites.
    ///
    /// # Panics
    ///
    /// If `bin_size` is 0.
    pub fn new(bin_size: usize) -> Self {
        assert!(bin_size > 0, "bin_size must be > 0");
        Self {
            bin_size,
            seq: Vec::new(),
            callable: counted_bin_size(bin_size).map(CountsBuilder::new),
            len: 0,
            n_missing: 0,
            has_het: false,
        }
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
        if let Some(callable) = &mut self.callable {
            let c = u16::try_from(self.bin_size - self.n_missing)
                .expect("counted bins have at most 65535 bases");
            callable.push(c);
        }
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
        self.finish_with_callable().0
    }

    /// [`Binner::finish`], also returning the callable bases of each bin
    /// (the bin size minus the bin's missing sites) when the bin size is at
    /// most [`MAX_COUNTED_BIN_SIZE`](crate::io::MAX_COUNTED_BIN_SIZE), and
    /// `None` for larger bins.
    pub fn finish_with_callable(&mut self) -> (Vec<u8>, Option<BinCounts>) {
        if self.len > 0 {
            self.emit();
        }
        (
            std::mem::take(&mut self.seq),
            self.callable.as_mut().map(CountsBuilder::finish),
        )
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

/// One sequence as a reader of per-site calls binned it ([`SeqBuilder`]).
#[derive(Debug, Clone)]
pub(crate) struct BinnedSeq {
    pub name: String,
    pub bins: Vec<u8>,
    /// The callable bases of each bin ([`Binner::finish_with_callable`]);
    /// `None` for bins too large to count.
    pub callable: Option<BinCounts>,
    /// Positions `1..=extent` make up the sequence.
    pub extent: u64,
    /// Positions binned as called (heterozygous or not).
    pub called: u64,
    /// Heterozygous positions.
    pub n_het: u64,
    /// The heterozygous sites, when they were collected.
    pub het_sites: Vec<HetCall>,
}

/// Bins the sites of one sequence, given in position order ([`Binner`]),
/// counts the called and heterozygous ones and, when collecting, records
/// the heterozygous sites with their `nr_called`.
#[derive(Debug, Clone)]
pub(crate) struct SeqBuilder {
    binner: Binner,
    collect_het: bool,
    /// Called sites since the last heterozygous one.
    called_since_het: u64,
    het_sites: Vec<HetCall>,
    called: u64,
    n_het: u64,
}

impl SeqBuilder {
    pub fn new(bin_size: usize, collect_het: bool) -> Self {
        Self {
            binner: Binner::new(bin_size),
            collect_het,
            called_since_het: 0,
            het_sites: Vec::new(),
            called: 0,
            n_het: 0,
        }
    }

    /// `n` called sites that are not heterozygous.
    pub fn called(&mut self, n: u64) {
        self.binner.add_called(n);
        self.called_since_het += n;
        self.called += n;
    }

    /// `n` missing sites.
    pub fn missing(&mut self, n: u64) {
        self.binner.add_missing(n);
    }

    /// One called heterozygous site, at `pos`.
    pub fn het(&mut self, pos: u64, alleles: [u8; 2]) {
        self.binner.add_het();
        if self.collect_het {
            self.het_sites.push(HetCall {
                pos,
                nr_called: self.called_since_het + 1,
                alleles,
            });
        }
        self.called_since_het = 0;
        self.called += 1;
        self.n_het += 1;
    }

    /// The sequence `name`, whose sites are positions `1..=extent`.
    pub fn finish(mut self, name: String, extent: u64) -> BinnedSeq {
        let (bins, callable) = self.binner.finish_with_callable();
        BinnedSeq {
            name,
            bins,
            callable,
            extent,
            called: self.called,
            n_het: self.n_het,
            het_sites: self.het_sites,
        }
    }
}

/// Distribution of read depth (DP) over VCF records: INFO/DP of vcf2fq's
/// SNP/reference records in consensus mode, the sample's DP of every record
/// in mask mode ([`crate::io::vcf::VcfSummary::depth`]).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DepthStats {
    /// Records counted.
    pub records: u64,
    /// Records without a DP value (in consensus mode, always lowercase in
    /// vcf2fq).
    pub records_without_dp: u64,
    /// The mean depth (0 when no record was counted).
    pub mean: f64,
    /// The middle value (the mean of the two middle values for an even count).
    pub median: f64,
    /// `histogram[d]` records have `DP == d`, for `d <` [`DP_HISTOGRAM_CAP`];
    /// the last entry counts `DP >= DP_HISTOGRAM_CAP`.
    pub histogram: Vec<u64>,
}

/// Depths from this value on share the last bin of [`DepthStats::histogram`].
pub const DP_HISTOGRAM_CAP: usize = 1000;

/// Exact counts of every DP value, compact for the usual small values.
#[derive(Debug, Clone, Default)]
pub(crate) struct DepthCounter {
    small: Vec<u64>,
    large: std::collections::BTreeMap<u64, u64>,
    without_dp: u64,
}

impl DepthCounter {
    const SMALL: u64 = 1 << 16;

    pub fn add(&mut self, dp: Option<u64>) {
        let Some(dp) = dp else {
            self.without_dp += 1;
            return;
        };
        if dp < Self::SMALL {
            let i = dp as usize;
            if i >= self.small.len() {
                self.small.resize(i + 1, 0);
            }
            self.small[i] += 1;
        } else {
            *self.large.entry(dp).or_insert(0) += 1;
        }
    }

    fn counts(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let small = self
            .small
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(d, &n)| (d as u64, n));
        small.chain(self.large.iter().map(|(&d, &n)| (d, n)))
    }

    pub fn stats(&self) -> DepthStats {
        let n: u64 = self.counts().map(|(_, c)| c).sum();
        let sum: u128 = self.counts().map(|(d, c)| d as u128 * c as u128).sum();
        let mut histogram = vec![0u64; DP_HISTOGRAM_CAP + 1];
        for (d, c) in self.counts() {
            histogram[(d as usize).min(DP_HISTOGRAM_CAP)] += c;
        }
        // The values of rank n/2 and (n-1)/2 (0-based) are the middle ones.
        let value_at = |rank: u64| {
            let mut seen = 0;
            for (d, c) in self.counts() {
                seen += c;
                if seen > rank {
                    return d;
                }
            }
            0
        };
        let median = if n == 0 {
            0.0
        } else {
            (value_at((n - 1) / 2) as f64 + value_at(n / 2) as f64) / 2.0
        };
        DepthStats {
            records: n,
            records_without_dp: self.without_dp,
            mean: if n == 0 { 0.0 } else { sum as f64 / n as f64 },
            median,
            histogram,
        }
    }
}

/// Turn whole sequences into HMM rows. With `batch_size`, long sequences are
/// split into chunks that stay connected in one HMM chain (`row_starts` is
/// `true` only for the first chunk of each sequence). Empty sequences are dropped.
///
/// `seqs` is a [`Sequences`] or, for sequences without callable bases, the
/// bins alone (`Vec<Vec<u8>>`). Callable bases are cut at the same bins as
/// the symbols; they are checked first ([`Sequences::validate`]).
pub fn chunk_sequences(
    seqs: impl Into<Sequences>,
    batch_size: Option<usize>,
    empty_msg: &str,
) -> Result<Observations> {
    if batch_size == Some(0) {
        bail!("batch_size must be > 0");
    }
    let seqs = seqs.into();
    seqs.validate()?;
    let Sequences { bins, callable } = seqs;
    let bin_size = callable.as_ref().map(CallableBases::bin_size);
    let mut counts = callable.map(|c| c.into_rows().into_iter());
    let mut rows = Vec::new();
    let mut row_starts = Vec::new();
    let mut count_rows = Vec::new();
    for seq in bins {
        let seq_counts = counts.as_mut().and_then(Iterator::next);
        if seq.is_empty() {
            continue;
        }
        match batch_size {
            None => {
                rows.push(seq);
                row_starts.push(true);
                count_rows.extend(seq_counts);
            }
            Some(batch) => {
                for (i, chunk) in seq.chunks(batch).enumerate() {
                    rows.push(chunk.to_vec());
                    row_starts.push(i == 0);
                }
                if let Some(c) = seq_counts {
                    let n = c.len();
                    count_rows.extend(
                        (0..n)
                            .step_by(batch)
                            .map(|start| c.slice(start..n.min(start + batch))),
                    );
                }
            }
        }
    }
    if rows.is_empty() {
        bail!("{empty_msg}");
    }
    let callable = bin_size
        .map(|s| CallableBases::new(s, count_rows))
        .transpose()?;
    let obs = Observations {
        rows,
        row_starts,
        callable,
    };
    debug_assert!(obs.validate().is_ok(), "chunking keeps the counts valid");
    Ok(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gives `data` a few bytes at a time, then fails.
    struct FailsAfter(Cursor<Vec<u8>>);

    impl Read for FailsAfter {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(7);
            match self.0.read(&mut buf[..n])? {
                0 => Err(io::Error::other("broken")),
                n => Ok(n),
            }
        }
    }

    #[test]
    fn the_decoder_thread_passes_on_every_byte_then_the_end_or_the_error() {
        for len in [0, 5, READ_BUFFER - 1, READ_BUFFER, 3 * READ_BUFFER + 11] {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let mut got = Vec::new();
            let mut ok = DecoderThread::spawn(Cursor::new(data.clone())).unwrap();
            ok.read_to_end(&mut got).unwrap();
            assert_eq!(got, data, "{len} bytes");
            assert_eq!(ok.read(&mut [0; 4]).unwrap(), 0, "{len} bytes");

            got.clear();
            let mut failing = DecoderThread::spawn(FailsAfter(Cursor::new(data.clone()))).unwrap();
            let err = failing.read_to_end(&mut got).unwrap_err();
            assert!(got == data && err.to_string() == "broken", "{len} bytes");
        }
    }

    /// Gives `data` a few bytes at a time, each read after an interrupted one.
    struct Trickle(Cursor<Vec<u8>>, bool);

    impl Read for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.1 = !self.1;
            if self.1 {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let n = buf.len().min(7);
            self.0.read(&mut buf[..n])
        }
    }

    #[test]
    fn the_line_reader_gives_the_lines_of_read_until() {
        let long = [&[b'x'; (1 << 20) + 3][..], b"\n\nz"].concat();
        for text in [&b""[..], b"\n", b"a", b"a\n\nbc\r\n", b"ab\ncd", &long] {
            let want: Vec<&[u8]> = text.split_inclusive(|&b| b == b'\n').collect();
            for trickle in [false, true] {
                let reader: Box<dyn Read> = match trickle {
                    false => Box::new(Cursor::new(text.to_vec())),
                    true => Box::new(Trickle(Cursor::new(text.to_vec()), false)),
                };
                let mut lines = LineReader::new(reader);
                let mut got = Vec::new();
                while lines.advance().unwrap() {
                    got.push(lines.line().to_vec());
                }
                assert!(got == want, "{} bytes, trickle {trickle}", text.len());
            }
        }
    }
}
