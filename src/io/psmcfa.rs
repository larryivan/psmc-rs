//! Reader for PSMC's own input format (`.psmcfa`, as written by `fq2psmcfa`).
//!
//! Parsing mirrors C psmc 0.6.5 byte for byte: records are split the way the
//! `kseq.h` bundled with it splits FASTA/FASTQ, and every sequence character is
//! mapped through the `conv_table` of its `cli.c`.

use anyhow::{Context, Result, bail};
use std::io::{BufRead, ErrorKind};
use std::path::Path;

use crate::io::Observations;
use crate::io::common::{chunk_sequences, open_reader};

/// `conv_table` from C psmc's `cli.c`, copied verbatim. `0`, `A`/`C`/`G`/`T`
/// and their lowercase forms map to 0 (called, not heterozygous); `1` and the
/// two-base IUPAC codes `K`/`M`/`R`/`S`/`W`/`Y` (either case) map to 1
/// (heterozygous); every other byte maps to 2 (missing).
#[rustfmt::skip]
const CONV_TABLE: [u8; 256] = [
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    0, 1, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 0, 2, 0,  2, 2, 2, 0,  2, 2, 2, 1,  2, 1, 2, 2,
    2, 2, 1, 1,  0, 2, 2, 1,  2, 1, 2, 2,  2, 2, 2, 2,
    2, 0, 2, 0,  2, 2, 2, 0,  2, 2, 2, 1,  2, 1, 2, 2,
    2, 2, 1, 1,  0, 2, 2, 1,  2, 1, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
    2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,  2, 2, 2, 2,
];

// Classes of a byte met inside a sequence; values 0..=2 are PSMC symbols.
const SKIP: u8 = 3;
const NEXT_HEADER: u8 = 4;
const QUAL_SEP: u8 = 5;

/// kseq's sequence loop: `>` and `@` end the record and start the next header,
/// `+` starts a FASTQ quality block, and bytes failing C's `isgraph()` (anything
/// outside 0x21..=0x7e: whitespace, `\r`, control and non-ASCII bytes) are
/// dropped. Only what is left reaches `conv_table`, so the non-ASCII half of
/// the table (indexed through a signed `char` in C) is never consulted.
const SEQ_CLASS: [u8; 256] = {
    let mut t = [SKIP; 256];
    let mut b = 0x21;
    while b <= 0x7e {
        t[b] = CONV_TABLE[b];
        b += 1;
    }
    t[b'>' as usize] = NEXT_HEADER;
    t[b'@' as usize] = NEXT_HEADER;
    t[b'+' as usize] = QUAL_SEP;
    t
};

/// C `isspace()` in the "C" locale, which ends a kseq record name. Unlike
/// `u8::is_ascii_whitespace` it includes the vertical tab.
fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t'..=b'\r')
}

/// One record of a psmcfa file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsmcfaRecord {
    /// Header text after `>` up to the first whitespace (kseq's `name`).
    pub name: String,
    /// One symbol per bin: 0 = `T` (called, no heterozygote), 1 = `K`
    /// (heterozygous), 2 = `N` (missing).
    pub seq: Vec<u8>,
}

impl PsmcfaRecord {
    /// Number of called (non-missing) bins, C psmc's `L_e`.
    pub fn n_called(&self) -> usize {
        self.seq.iter().filter(|&&s| s < 2).count()
    }

    /// Number of heterozygous bins, C psmc's `n_e`.
    pub fn n_het(&self) -> usize {
        self.seq.iter().filter(|&&s| s == 1).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Skipping to the next `>` or `@` (kseq's `last_char == 0`): before the
    /// first record and after a FASTQ record.
    SeekHeader,
    /// Right after `>`/`@`, before any byte of the name.
    HeaderStart,
    Name,
    /// Rest of the header line after the name.
    Comment,
    Seq,
    /// Rest of a FASTQ `+` line.
    QualLine,
    /// FASTQ quality block with `left` quality characters still expected.
    /// kseq's loop reads one byte before testing the count, so at 0 the record
    /// ends only after one more byte has been consumed (or at end of input).
    Qual {
        left: usize,
    },
}

/// Incremental psmcfa parser following `kseq_read` byte for byte.
struct Parser {
    state: State,
    name: Vec<u8>,
    seq: Vec<u8>,
    records: Vec<PsmcfaRecord>,
}

impl Parser {
    fn new() -> Self {
        Self {
            state: State::SeekHeader,
            name: Vec::new(),
            seq: Vec::new(),
            records: Vec::new(),
        }
    }

    fn end_record(&mut self) {
        self.records.push(PsmcfaRecord {
            name: String::from_utf8_lossy(&self.name).into_owned(),
            seq: std::mem::take(&mut self.seq),
        });
        self.name.clear();
    }

    fn feed(&mut self, buf: &[u8]) {
        let mut i = 0;
        while i < buf.len() {
            let rest = &buf[i..];
            match self.state {
                State::SeekHeader => match rest.iter().position(|&b| b == b'>' || b == b'@') {
                    Some(p) => {
                        i += p + 1;
                        self.state = State::HeaderStart;
                    }
                    None => i = buf.len(),
                },
                State::HeaderStart | State::Name => {
                    match rest.iter().position(|&b| is_c_space(b)) {
                        Some(p) => {
                            self.name.extend_from_slice(&rest[..p]);
                            self.state = if rest[p] == b'\n' {
                                State::Seq
                            } else {
                                State::Comment
                            };
                            i += p + 1;
                        }
                        None => {
                            self.name.extend_from_slice(rest);
                            self.state = State::Name;
                            i = buf.len();
                        }
                    }
                }
                State::Comment | State::QualLine => match rest.iter().position(|&b| b == b'\n') {
                    Some(p) => {
                        i += p + 1;
                        self.state = if self.state == State::Comment {
                            State::Seq
                        } else {
                            State::Qual {
                                left: self.seq.len(),
                            }
                        };
                    }
                    None => i = buf.len(),
                },
                State::Seq => {
                    for &b in rest {
                        i += 1;
                        match SEQ_CLASS[b as usize] {
                            SKIP => {}
                            NEXT_HEADER => {
                                self.end_record();
                                self.state = State::HeaderStart;
                                break;
                            }
                            QUAL_SEP => {
                                self.state = State::QualLine;
                                break;
                            }
                            sym => self.seq.push(sym),
                        }
                    }
                }
                State::Qual { left } => {
                    i += 1;
                    if left == 0 {
                        self.end_record();
                        self.state = State::SeekHeader;
                    } else if (33..=127).contains(&rest[0]) {
                        self.state = State::Qual { left: left - 1 };
                    }
                }
            }
        }
    }

    fn finish(mut self) -> Result<Vec<PsmcfaRecord>> {
        match self.state {
            // A `>` as the very last byte: kseq reports end of file, no record.
            State::SeekHeader | State::HeaderStart => {}
            State::Name | State::Comment | State::Seq | State::Qual { left: 0 } => {
                self.end_record()
            }
            State::QualLine | State::Qual { .. } => {
                // kseq returns -2 here and C psmc silently stops reading, keeping
                // only the records before this one. Refuse instead of truncating.
                bail!(
                    "psmcfa record '{}' (#{}) has a '+' in its sequence, which starts a FASTQ \
                     quality block, but the input ends before the quality string is complete",
                    String::from_utf8_lossy(&self.name),
                    self.records.len() + 1
                );
            }
        }
        Ok(self.records)
    }
}

/// Parse every record of a psmcfa stream exactly as C psmc reads it.
///
/// - A record starts at `>` (or `@`) anywhere in the input, even mid-line, and
///   its header line is never sequence data. The name is the header text up to
///   the first whitespace. Bytes before the first header are ignored, so input
///   without any `>` line yields no records.
/// - In sequence lines, whitespace (including `\r` of CRLF files), control and
///   non-ASCII bytes are skipped; every other byte is mapped like C's
///   `conv_table`: `T`, `A`, `C`, `G`, `0` and lowercase forms are 0, `K`, `M`,
///   `R`, `S`, `W`, `Y`, `1` and lowercase forms are 1, all others are 2 (`N`).
/// - Records with an empty sequence are returned (C psmc counts them too).
/// - A `+` in a sequence starts a FASTQ quality block, as in kseq. When that
///   block is truncated, C psmc silently stops reading; this returns an error.
///
/// The input is streamed through the reader's buffer, so memory use is that of
/// the parsed bins (one byte each) rather than of the whole text.
pub fn parse_psmcfa<R: BufRead>(mut reader: R) -> Result<Vec<PsmcfaRecord>> {
    let mut parser = Parser::new();
    loop {
        let buf = match reader.fill_buf() {
            Ok(buf) => buf,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e).context("failed to read psmcfa input"),
        };
        if buf.is_empty() {
            break;
        }
        parser.feed(buf);
        let n = buf.len();
        reader.consume(n);
    }
    parser.finish()
}

/// Read all records of a (optionally gzip-compressed) psmcfa file; see
/// [`parse_psmcfa`] for the exact rules.
pub fn read_psmcfa_records(path: &Path) -> Result<Vec<PsmcfaRecord>> {
    let reader = open_reader(path)?;
    parse_psmcfa(reader).with_context(|| format!("failed to parse {path:?}"))
}

/// Read a psmcfa file into HMM rows, one per record (or per `batch_size` bins
/// of a record, see [`chunk_sequences`]). Records with an empty sequence are
/// dropped. Fails when the file holds no `>` record at all.
pub fn read_psmcfa(path: &Path, batch_size: Option<usize>) -> Result<Observations> {
    let records = read_psmcfa_records(path)?;
    if records.is_empty() {
        bail!(
            "no psmcfa records found in {path:?}: each sequence must start with a '>name' header line"
        );
    }
    chunk_sequences(
        records.into_iter().map(|r| r.seq).collect(),
        batch_size,
        "no valid observations found in psmcfa input",
    )
}
