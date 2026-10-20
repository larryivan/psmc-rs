//! The psmcfa reader: records and their T/K/N bins as C psmc reads them.
//! `tests/io_text_psmcfa_c.rs` checks the parser against C psmc's counts.

mod common;

use common::TempDir;
use flate2::Compression;
use flate2::write::GzEncoder;
use psmc_rs::io::Observations;
use psmc_rs::io::psmcfa::{PsmcfaRecord, parse_psmcfa, read_psmcfa};
use std::fs;
use std::io::Write;

/// `bytes` written to the file `name` and read in rows of `batch_size`.
fn read(name: &str, bytes: &[u8], batch_size: Option<usize>) -> anyhow::Result<Observations> {
    let dir = TempDir::new("io_psmcfa");
    let path = dir.path(name);
    fs::write(&path, bytes).unwrap();
    read_psmcfa(&path, batch_size)
}

fn rows(bytes: &[u8]) -> Vec<Vec<u8>> {
    read("in.psmcfa", bytes, None).unwrap().rows
}

fn parse(bytes: &[u8]) -> Vec<PsmcfaRecord> {
    parse_psmcfa(bytes).unwrap()
}

#[test]
fn records_are_rows_cut_into_batches() {
    let content = b"> chr1\nTTKN\n> chr2\nTNTK\n";
    let obs = read("in.psmcfa", content, None).unwrap();
    assert_eq!(obs.row_starts, [true, true]);
    assert_eq!(obs.rows, [[0, 0, 1, 2], [0, 2, 0, 1]]);
    let obs = read("in.psmcfa", content, Some(3)).unwrap();
    assert_eq!(obs.row_starts, [true, false, true, false]);
    assert_eq!(obs.rows, [vec![0, 0, 1], vec![2], vec![0, 2, 0], vec![1]]);
}

#[test]
fn gzip_input_is_read() {
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(b"> chr1\nTKNT\n").unwrap();
    let obs = read("in.psmcfa.gz", &gz.finish().unwrap(), None).unwrap();
    assert_eq!(obs.row_starts, [true]);
    assert_eq!(obs.rows, [[0, 1, 2, 0]]);
}

#[test]
fn header_lines_are_not_sequence_data() {
    // T, K and N in names and descriptions (common in GenBank, RefSeq and UCSC
    // names) used to be read as extra bins.
    let content = b">KTN_scaffold_1\nTTTT\n>NC_000001.11 Homo sapiens chr1, GRCh38\nTKTK\n\
>chrUn_KI270302v1\tunlocalized\nNNKT\n";
    assert_eq!(rows(content), [[0, 0, 0, 0], [0, 1, 0, 1], [2, 2, 1, 0]]);
    let names: Vec<_> = parse(content).into_iter().map(|r| r.name).collect();
    assert_eq!(
        names,
        ["KTN_scaffold_1", "NC_000001.11", "chrUn_KI270302v1"]
    );
}

#[test]
fn sequence_bytes_map_like_c_conv_table() {
    // Bases and '0' are called homozygous, two-base IUPAC codes and '1' are
    // heterozygous, any other printable byte is missing; none is dropped.
    let content = b">s\nACGTacgt0\n1KMRSWYkmrswy\nBDHVNbdhvnXxU2-.*?\n";
    let expected = [vec![0; 9], vec![1; 13], vec![2; 18]].concat();
    assert_eq!(rows(content), [expected]);
}

#[test]
fn crlf_and_whitespace_inside_sequence_are_skipped() {
    let content = b">s1\r\nTT KT\r\n\tKK\x0bTT\x0c\r\n>s2 desc\r\nTN\r\n";
    assert_eq!(rows(content), [vec![0, 0, 1, 0, 1, 1, 0, 0], vec![0, 2]]);
}

#[test]
fn header_marker_starts_a_record_anywhere() {
    // kseq starts a record at '>' or '@' even in the middle of a line.
    let recs = parse(b">s\nTT>x desc\nKK@y\nKT\n");
    let got: Vec<_> = recs.iter().map(|r| (r.name.as_str(), &r.seq[..])).collect();
    assert_eq!(got, [("s", &[0, 0][..]), ("x", &[1, 1]), ("y", &[1, 0])]);
}

#[test]
fn empty_records_are_kept_by_the_parser_and_dropped_as_rows() {
    let content = b">s\nTK\n>e1\n>e2 comment\n\n>f\nK\n";
    let recs = parse(content);
    let lens: Vec<_> = recs
        .iter()
        .map(|r| (r.name.as_str(), r.seq.len()))
        .collect();
    assert_eq!(lens, [("s", 2), ("e1", 0), ("e2", 0), ("f", 1)]);
    let obs = read("in.psmcfa", content, None).unwrap();
    assert_eq!(obs.rows, [vec![0, 1], vec![1]]);
    assert_eq!(obs.row_starts, [true, true]);
}

#[test]
fn input_without_header_is_rejected() {
    // C psmc skips everything before the first '>' and so reads no sequence
    // at all from such a file (its estimates are all NaN).
    assert!(parse(b"TTKT\nKKTT\n").is_empty());
    let err = read("in.psmcfa", b"TTKT\nKKTT\n", None).unwrap_err();
    assert!(format!("{err:#}").contains("header"), "{err:#}");
    // Bytes before the first header are ignored.
    let recs = parse(b"TTTT\nKK\n>s\nKT\n");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].seq, [1, 0]);
}
