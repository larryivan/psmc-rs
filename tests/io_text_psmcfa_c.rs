//! psmcfa parsing cross-checked against the original C psmc (0.6.5-r74).
//!
//! The expected numbers were produced by running `psmc -N0 <file>` on each
//! input. Its `MM n_seqs:.., sum_L:.., sum_n:..` line gives the number of
//! records, of called (non-N) bins and of heterozygous (K) bins.

use psmc_rs::io::psmcfa::{PsmcfaRecord, parse_psmcfa, read_psmcfa_records};
use std::io::BufReader;
use std::path::Path;

/// `(n_seqs, sum_L, sum_n)` as printed by C psmc.
type Counts = (usize, usize, usize);

fn c_counts(recs: &[PsmcfaRecord]) -> Counts {
    (
        recs.len(),
        recs.iter().map(PsmcfaRecord::n_called).sum(),
        recs.iter().map(PsmcfaRecord::n_het).sum(),
    )
}

/// Inputs on which C psmc runs normally, with its counts.
const MATCHES_C: &[(&str, &[u8], Counts)] = &[
    (
        "headers with T/K/N",
        b">KTN_scaffold_1\nTTTT\n>NC_000001.11 Homo sapiens chromosome 1, GRCh38.p14\nTKTK\n\
          >chrUn_KI270302v1\tunlocalized\nNNKT\n",
        (3, 10, 3),
    ),
    (
        "IUPAC, lowercase, digits, punctuation",
        b">s\nACGTacgt01KMRSWYkmrswyBDHVNbdhvnXx23456789-.*?\n",
        (1, 22, 13),
    ),
    (
        "CRLF",
        b">s1\r\nTTKT\r\nKKTT\r\n>s2 desc\r\nTN\r\n",
        (2, 9, 3),
    ),
    (
        "empty records after the first",
        b">s\nTK\n>e1\n>e2 comment\n\n>f\nK\n",
        (4, 3, 2),
    ),
    (
        "whitespace inside sequence lines",
        b">s\nT T\tK\x0bT\x0cK\r\n  K\n",
        (1, 6, 3),
    ),
    ("'>' in the middle of a line", b">s\nTT>x\nKK\n", (2, 4, 2)),
    ("'@' headers", b"@s\nTK\n@t\nKK\n", (2, 4, 3)),
    ("FASTQ record", b">s\nTTKT\n+\nIIII\nKK\n>t\nK\n", (2, 5, 2)),
    (
        "FASTQ quality made of '>'",
        b">s\nTK\n+\n>>\n>t\nK\n",
        (2, 3, 2),
    ),
    (
        "byte after FASTQ quality is swallowed, even a '>'",
        b">s\nTK\n+\nII>t\nK\n>u\nT\n",
        (2, 3, 1),
    ),
    (
        "FASTQ record with empty sequence",
        b">a\nT\n>s\n+\nX>t\nK\n",
        (3, 2, 1),
    ),
    (
        "junk before the first header",
        b"TTTT\nKK\n>s\nKT\n",
        (1, 2, 1),
    ),
    ("no trailing newline", b">s\nTK", (1, 2, 1)),
    ("header at end of input", b">s\nTK\n>t", (2, 2, 1)),
    ("bare '>' at end of input", b">s\nTK\n>", (1, 2, 1)),
    (
        "markers inside header comments",
        b">s a>b+c@d\nTK\n>t\tx+y\nK\n",
        (2, 3, 2),
    ),
    (
        "control and non-ASCII bytes",
        b">s\nT\x80\xffK\x01\x7fT\x00K\n",
        (1, 4, 2),
    ),
    ("no header at all", b"TTKT\n", (0, 0, 0)),
    ("empty input", b"", (0, 0, 0)),
];

#[test]
fn record_and_bin_counts_match_c_psmc() {
    for (case, input, expected) in MATCHES_C {
        let recs = parse_psmcfa(*input).unwrap_or_else(|e| panic!("{case}: {e:#}"));
        assert_eq!(c_counts(&recs), *expected, "{case}");
    }
}

#[test]
fn header_only_first_record_is_an_empty_record() {
    // C psmc segfaults on these (kseq writes the terminator of the still
    // unallocated sequence buffer); after a non-empty record the same empty
    // record is counted (see "empty records after the first").
    for input in [&b">e\n>s\nTK\n"[..], &b">e\n\n>s\nTK\n"[..]] {
        let recs = parse_psmcfa(input).expect("parse failed");
        assert_eq!(c_counts(&recs), (2, 2, 1));
        assert!(recs[0].seq.is_empty());
    }
}

#[test]
fn truncated_fastq_quality_errors_where_c_stops_silently() {
    // C psmc reports n_seqs:1, sum_L:2, sum_n:1 and n_seqs:0 here: kseq returns
    // -2 and psmc quietly analyses only the records read so far. The error
    // names the record.
    for (input, record) in [
        (&b">s\nTK\n>t\nTTKT\n+\nII\n"[..], "'t'"),
        (b">s\nTK+", "'s'"),
    ] {
        let msg = format!("{:#}", parse_psmcfa(input).unwrap_err());
        assert!(msg.contains(record) && msg.contains("FASTQ"), "{msg}");
    }
}

#[test]
fn parsing_does_not_depend_on_read_boundaries() {
    // The parser is fed one reader buffer at a time; tiny buffers put a
    // boundary at every position of every header, CRLF and FASTQ block.
    let extra: [&[u8]; 4] = [
        b">e\n>s\nTK\n",
        b">s\nTK\n>t\nTTKT\n+\nII\n",
        b">s\nTK+",
        b">s\r\nTK\r\n>t desc\r\nKT\r\n",
    ];
    let inputs = MATCHES_C.iter().map(|c| c.1).chain(extra);
    for input in inputs {
        let whole = parse_psmcfa(input).map_err(|e| format!("{e:#}"));
        for cap in [1, 2, 3, 7] {
            let reader = BufReader::with_capacity(cap, input);
            let chunked = parse_psmcfa(reader).map_err(|e| format!("{e:#}"));
            assert_eq!(chunked, whole, "capacity {cap}, input {input:?}");
        }
    }
}

/// What C psmc made of byte `b` in `>s\nTK<b>KT\n` (row `b >> 4`, column
/// `b & 15`), measured for all 256 values with `psmc -N0 -d`: `0`/`1`/`2` the
/// byte became a T/K/N bin, `.` it was skipped, `h` it started a new record
/// (named `KT`, empty), `q` C read no sequence at all (a `+` starts a FASTQ
/// quality block that is never completed; we return an error).
const C_BYTE_CLASSES: [&str; 16] = [
    "................",
    "................",
    ".2222222222q2222",
    "01222222222222h2",
    "h020222022212122",
    "2211022121222222",
    "2020222022212122",
    "221102212122222.",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
    "................",
];

fn byte_class(b: u8) -> char {
    let input = [b'>', b's', b'\n', b'T', b'K', b, b'K', b'T', b'\n'];
    let Ok(recs) = parse_psmcfa(&input[..]) else {
        return 'q';
    };
    match recs.as_slice() {
        [r] if r.name == "s" && r.seq.len() == 5 => (b'0' + r.seq[2]) as char,
        [r] if r.name == "s" && r.seq == [0, 1, 1, 0] => '.',
        [r, e] if r.seq == [0, 1] && e.name == "KT" && e.seq.is_empty() => 'h',
        _ => panic!("byte {b:#04x}: unexpected records {recs:?}"),
    }
}

#[test]
fn every_byte_value_is_read_like_c_psmc() {
    for b in 0..=255u8 {
        let expected = C_BYTE_CLASSES[(b >> 4) as usize].as_bytes()[(b & 15) as usize] as char;
        assert_eq!(byte_class(b), expected, "byte {b:#04x}");
    }
}

#[test]
fn real_psmcfa_files_match_c_psmc() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // HLpelCas1 has 523 headers such as '>VOCT01035352.1' that used to add a T bin.
    let files: &[(&str, Counts)] = &[
        (
            "tests/data/c_reference/papuan_chr1_3seg.psmcfa",
            (3, 150_947, 7_373),
        ),
        (
            "experiment/real_data/Papuan_highlands_diploid.psmcfa",
            (1, 2_240_137, 123_684),
        ),
        (
            "experiment/real_data/HLhydTec1_diploid.psmcfa",
            (34, 18_410_259, 2_422_447),
        ),
        (
            "experiment/real_data/HLemySub1_diploid.psmcfa",
            (889, 18_835_403, 4_234_287),
        ),
        (
            "experiment/real_data/HLpelCas1_diploid.psmcfa",
            (1_220, 18_694_606, 3_972_041),
        ),
    ];
    for (rel, expected) in files {
        let path = root.join(rel);
        if !path.exists() {
            eprintln!("skipping {rel}: not present");
            continue;
        }
        let recs = read_psmcfa_records(&path).expect("parse failed");
        assert_eq!(c_counts(&recs), *expected, "{rel}");
    }
}
