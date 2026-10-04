use flate2::Compression;
use flate2::write::GzEncoder;
use psmc_rs::io::Observations;
use psmc_rs::io::psmcfa::{PsmcfaRecord, parse_psmcfa, read_psmcfa};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

/// A path no other test of this process uses: tests run in parallel, and
/// the clock alone is too coarse to tell them apart.
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

#[test]
fn read_unbatched_psmcfa_parses_sequences() {
    let path = unique_temp_path("psmc_unbatched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let obs = read_psmcfa(&path, None).expect("failed to parse psmcfa");
    assert_eq!(obs.rows.len(), 2);
    assert_eq!(obs.row_starts, vec![true, true]);
    assert_eq!(obs.rows[0], vec![0, 0, 1, 2]); // T T K N
    assert_eq!(obs.rows[1], vec![0, 2, 0, 1]); // T N T K

    let _ = fs::remove_file(path);
}

#[test]
fn read_batched_psmcfa_keeps_sequence_chain() {
    let path = unique_temp_path("psmc_batched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let obs = read_psmcfa(&path, Some(3)).expect("failed to parse batched psmcfa");
    assert_eq!(obs.rows.len(), 4);
    assert_eq!(obs.row_starts, vec![true, false, true, false]);
    assert_eq!(obs.rows[0], vec![0, 0, 1]);
    assert_eq!(obs.rows[1], vec![2]);
    assert_eq!(obs.rows[2], vec![0, 2, 0]);
    assert_eq!(obs.rows[3], vec![1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_gz_psmcfa_works() {
    let path = unique_temp_path("psmc_gz", "psmcfa.gz");
    let file = fs::File::create(&path).expect("failed to create gz path");
    let mut writer = GzEncoder::new(file, Compression::default());
    writer
        .write_all(b"> chr1\nTKNT\n")
        .expect("failed to write gz data");
    writer.finish().expect("failed to finish gzip stream");

    let obs = read_psmcfa(&path, None).expect("failed to parse gz psmcfa");
    assert_eq!(obs.rows.len(), 1);
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows[0], vec![0, 1, 2, 0]);

    let _ = fs::remove_file(path);
}

fn parse(bytes: &[u8]) -> Vec<PsmcfaRecord> {
    parse_psmcfa(bytes).expect("failed to parse psmcfa bytes")
}

fn read_bytes(prefix: &str, bytes: &[u8]) -> anyhow::Result<Observations> {
    let path = unique_temp_path(prefix, "psmcfa");
    fs::write(&path, bytes).expect("failed to write test psmcfa");
    let obs = read_psmcfa(&path, None);
    let _ = fs::remove_file(path);
    obs
}

#[test]
fn header_lines_are_not_sequence_data() {
    // T, K and N in names and descriptions (common in GenBank, RefSeq and UCSC
    // names) used to be read as extra bins.
    let content = b">KTN_scaffold_1\nTTTT\n>NC_000001.11 Homo sapiens chr1, GRCh38\nTKTK\n\
>chrUn_KI270302v1\tunlocalized\nNNKT\n";
    let obs = read_bytes("psmc_hdr_tkn", content).expect("failed to parse psmcfa");
    assert_eq!(
        obs.rows,
        vec![vec![0, 0, 0, 0], vec![0, 1, 0, 1], vec![2, 2, 1, 0]]
    );
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
    let obs = read_bytes("psmc_conv", content).expect("failed to parse psmcfa");
    let mut expected = vec![0u8; 9];
    expected.extend([1u8; 13]);
    expected.extend([2u8; 18]);
    assert_eq!(obs.rows, vec![expected]);
}

#[test]
fn crlf_and_whitespace_inside_sequence_are_skipped() {
    let content = b">s1\r\nTT KT\r\n\tKK\x0bTT\x0c\r\n>s2 desc\r\nTN\r\n";
    let obs = read_bytes("psmc_crlf", content).expect("failed to parse psmcfa");
    assert_eq!(obs.rows, vec![vec![0, 0, 1, 0, 1, 1, 0, 0], vec![0, 2]]);
}

#[test]
fn header_marker_starts_a_record_anywhere() {
    // kseq starts a record at '>' or '@' even in the middle of a line.
    let recs = parse(b">s\nTT>x desc\nKK@y\nKT\n");
    let got: Vec<_> = recs
        .iter()
        .map(|r| (r.name.as_str(), r.seq.clone()))
        .collect();
    assert_eq!(
        got,
        vec![("s", vec![0, 0]), ("x", vec![1, 1]), ("y", vec![1, 0])]
    );
}

#[test]
fn empty_records_are_kept_by_the_parser_and_dropped_as_rows() {
    let content = b">s\nTK\n>e1\n>e2 comment\n\n>f\nK\n";
    let recs = parse(content);
    let lens: Vec<_> = recs
        .iter()
        .map(|r| (r.name.as_str(), r.seq.len()))
        .collect();
    assert_eq!(lens, vec![("s", 2), ("e1", 0), ("e2", 0), ("f", 1)]);
    let obs = read_bytes("psmc_empty_rec", content).expect("failed to parse psmcfa");
    assert_eq!(obs.rows, vec![vec![0, 1], vec![1]]);
    assert_eq!(obs.row_starts, vec![true, true]);
}

#[test]
fn input_without_header_is_rejected() {
    // C psmc skips everything before the first '>' and so reads no sequence
    // at all from such a file (its estimates are all NaN).
    assert!(parse(b"TTKT\nKKTT\n").is_empty());
    let err = read_bytes("psmc_nohdr", b"TTKT\nKKTT\n").expect_err("expected an error");
    assert!(format!("{err:#}").contains("header"), "{err:#}");
    // Bytes before the first header are ignored.
    let recs = parse(b"TTTT\nKK\n>s\nKT\n");
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].seq, vec![1, 0]);
}

#[test]
fn truncated_fastq_quality_is_an_error() {
    // C psmc stops reading here without a word and analyses only record 's'.
    let err = parse_psmcfa(&b">s\nTK\n>t\nTTKT\n+\nII\n"[..]).expect_err("expected an error");
    let msg = format!("{err:#}");
    assert!(msg.contains("'t'") && msg.contains("FASTQ"), "{msg}");
}
