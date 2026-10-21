//! Compressed inputs: every reader must decode all gzip members (bgzip files are
//! a chain of independent members) and must not rely on the `.gz` extension.

mod common;

use common::TempDir;
use flate2::write::GzEncoder;
use flate2::{Compression, GzBuilder};
use psmc_rs::io::input::{InputOptions, load};
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::vcf::read_vcf;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// `bytes` written to `name` in `dir`.
fn write(dir: &TempDir, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
    let path = dir.path(name);
    fs::write(&path, bytes).unwrap();
    path
}

fn gzip_member(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// Compress `parts` as consecutive gzip members, like bgzip does. Current bgzip
/// puts a VCF header in its own first block, so splitting at arbitrary points
/// (including mid-line) is what real files look like.
fn multi_member(parts: &[&[u8]]) -> Vec<u8> {
    parts.iter().flat_map(|p| gzip_member(p)).collect()
}

/// A VCF of 300 heterozygous SNPs, 1 kb apart, and where its records start.
fn vcf_text() -> (String, usize) {
    let mut s = String::from(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
    );
    let header = s.len();
    for i in 0..300 {
        s.push_str(&format!(
            "chr1\t{}\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n",
            i * 1000 + 50
        ));
    }
    (s, header)
}

/// The mask-mode reader's bins of 100 for `vcf` with the callable `masks`.
fn vcf_rows(vcf: &Path, masks: &[PathBuf]) -> anyhow::Result<Vec<Vec<u8>>> {
    Ok(read_vcf(vcf, None, 100, None, masks, &[])?.rows)
}

fn het_bins(rows: &[Vec<u8>]) -> usize {
    rows.iter().flatten().filter(|&&b| b == 1).count()
}

#[test]
fn bgzip_style_vcf_is_read_in_full() {
    let dir = TempDir::new("io_gzip");
    let (text, header_end) = vcf_text();
    let (header, body) = text.as_bytes().split_at(header_end);
    // Header in its own member, records split across members mid-line.
    let mid = body.len() / 3;
    let parts = [header, &body[..mid], &body[mid..2 * mid], &body[2 * mid..]];
    let gz = write(&dir, "calls.vcf.gz", multi_member(&parts));
    let plain = write(&dir, "calls.vcf", &text);
    let from_gz = vcf_rows(&gz, &[]).unwrap();
    assert_eq!(from_gz, vcf_rows(&plain, &[]).unwrap());
    assert_eq!(het_bins(&from_gz), 300);
}

#[test]
fn bgzip_style_psmcfa_is_read_in_full() {
    let mut text = String::from(">s1\n");
    for i in 0..5000 {
        text.push(if i % 10 == 0 { 'K' } else { 'T' });
        if i % 60 == 59 {
            text.push('\n');
        }
    }
    text.push_str("\n>s2\n");
    text.push_str(&"T".repeat(3000));
    text.push('\n');
    let bytes = text.as_bytes();
    let dir = TempDir::new("io_gzip");
    let gz = multi_member(&[&bytes[..4], &bytes[4..2000], &bytes[2000..]]);
    let obs = read_psmcfa(&write(&dir, "in.psmcfa.gz", gz), None).unwrap();
    let lens: Vec<usize> = obs.rows.iter().map(Vec::len).collect();
    assert_eq!(lens, [5000, 3000]);
}

#[test]
fn bgzip_style_mhs_is_read_in_full() {
    let text: String = (1..=200u64)
        .map(|i| format!("chr1\t{}\t100\tAC\n", i * 100))
        .collect();
    let bytes = text.as_bytes();
    let dir = TempDir::new("io_gzip");
    let gz = write(
        &dir,
        "in.mhs.gz",
        multi_member(&[&bytes[..37], &bytes[37..]]),
    );
    assert_eq!(read_mhs(&gz, None, 100).unwrap().rows, [vec![1u8; 200]]);
}

#[test]
fn bgzip_style_mask_is_read_in_full() {
    // Callable mask split over two members: [1, 1000] and [1001, 300000].
    let mask_text = "chr1\t0\t1000\nchr1\t1000\t300000\n";
    let bytes = mask_text.as_bytes();
    let dir = TempDir::new("io_gzip");
    let gz = write(
        &dir,
        "mask.bed.gz",
        multi_member(&[&bytes[..13], &bytes[13..]]),
    );
    let plain = write(&dir, "mask.bed", mask_text);
    let vcf = write(&dir, "calls.vcf", vcf_text().0);
    let with_gz = vcf_rows(&vcf, &[gz]).unwrap();
    assert_eq!(with_gz, vcf_rows(&vcf, &[plain]).unwrap());
    // A truncated mask would leave everything after the first block uncallable.
    assert!(with_gz[0].iter().all(|&v| v != 2));
}

/// The empty block that ends a complete BGZF file.
const BGZF_EOF: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0, 0, 0, 0, 0, 0xff, 0x06, 0, b'B', b'C', 0x02, 0, 0x1b, 0, 0x03, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
];

/// One BGZF block: a gzip member with BGZF's `BC` extra subfield, which
/// holds the block size minus 1.
fn bgzf_block(data: &[u8]) -> Vec<u8> {
    let mut enc = GzBuilder::new()
        .extra(vec![b'B', b'C', 2, 0, 0, 0])
        .write(Vec::new(), Compression::default());
    enc.write_all(data).unwrap();
    let mut block = enc.finish().unwrap();
    let bsize = u16::try_from(block.len() - 1).expect("block too large");
    block[16..18].copy_from_slice(&bsize.to_le_bytes());
    block
}

/// `parts` as BGZF blocks, as bgzip writes them, with or without the
/// end-of-file block.
fn bgzf(parts: &[&[u8]], eof: bool) -> Vec<u8> {
    let mut out: Vec<u8> = parts.iter().flat_map(|p| bgzf_block(p)).collect();
    if eof {
        out.extend_from_slice(&BGZF_EOF);
    }
    out
}

/// The error of a BGZF file cut after a block.
fn assert_truncated(e: anyhow::Error) {
    let msg = format!("{e:#}");
    assert!(
        msg.contains("end-of-file marker") && msg.contains("truncated"),
        "{msg}"
    );
}

#[test]
fn a_bgzip_file_without_its_end_of_file_block_is_an_error() {
    let dir = TempDir::new("io_gzip");
    let (text, header_end) = vcf_text();
    let (header, body) = text.as_bytes().split_at(header_end);
    // Blocks end at line ends, as bcftools writes them: a file cut after a
    // block decodes cleanly.
    let cut_at = body[..body.len() / 2]
        .iter()
        .rposition(|&b| b == b'\n')
        .unwrap()
        + 1;
    let blocks = [header, &body[..cut_at], &body[cut_at..]];
    let vcf = write(&dir, "calls.vcf.gz", bgzf(&blocks, true));
    assert_eq!(het_bins(&vcf_rows(&vcf, &[]).unwrap()), 300);
    for cut in [bgzf(&blocks[..2], false), bgzf(&blocks, false)] {
        fs::write(&vcf, cut).unwrap();
        assert_truncated(vcf_rows(&vcf, &[]).unwrap_err());
    }

    // The same for psmcfa input and for masks.
    let records = [&b">s1\nTTKT\n"[..], b">s2\nKKTT\n"];
    let psmcfa = write(&dir, "in.psmcfa.gz", bgzf(&records, true));
    assert_eq!(read_psmcfa(&psmcfa, None).unwrap().rows.len(), 2);
    fs::write(&psmcfa, bgzf(&records[..1], false)).unwrap();
    assert_truncated(read_psmcfa(&psmcfa, None).unwrap_err());

    let rows = [&b"chr1\t0\t1000\n"[..], b"chr1\t1000\t300000\n"];
    let mask = write(&dir, "mask.bed.gz", bgzf(&rows[..1], false));
    let plain = write(&dir, "calls.vcf", &text);
    let masks = [mask];
    assert_truncated(vcf_rows(&plain, &masks).unwrap_err());
    fs::write(&masks[0], bgzf(&rows, true)).unwrap();
    assert!(vcf_rows(&plain, &masks).is_ok());
}

#[test]
fn gzip_is_detected_by_content_not_extension() {
    let text = ">s1\nTTKT\n";
    let dir = TempDir::new("io_gzip");
    let gz_no_ext = write(&dir, "gz.psmcfa", gzip_member(text.as_bytes()));
    let plain_with_gz_ext = write(&dir, "plain.psmcfa.gz", text);
    for path in [&gz_no_ext, &plain_with_gz_ext] {
        let obs = read_psmcfa(path, None).unwrap();
        assert_eq!(obs.rows, [[0, 0, 1, 0]], "{path:?}");
    }
}

/// A cut-short bgzip of a real multihetsep file is that file's error: the
/// note about how the format was recognised, which suits a file of another
/// format, is not added to it.
#[test]
fn a_truncated_bgzip_multihetsep_is_not_blamed_on_the_format_guess() {
    let rows: Vec<String> = (1..=400u64)
        .map(|i| format!("chr1\t{}\t100\tAC\n", i * 100))
        .collect();
    let (first, rest) = (rows[..200].concat(), rows[200..].concat());
    let dir = TempDir::new("io_gzip");
    let mhs = write(
        &dir,
        "in.mhs.gz",
        bgzf(&[first.as_bytes(), rest.as_bytes()], false),
    );
    let err = load(&mhs, &InputOptions::default()).unwrap_err();
    let msg = format!("{err:#}");
    assert!(!msg.contains("recognised from the first line"), "{msg}");
    assert_eq!(msg.matches(&format!("{mhs:?}")).count(), 1, "{msg}");
    assert_truncated(err);
}
