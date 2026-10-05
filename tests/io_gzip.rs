//! Compressed inputs: every reader must decode all gzip members (bgzip files are
//! a chain of independent members) and must not rely on the `.gz` extension.

use flate2::write::GzEncoder;
use flate2::{Compression, GzBuilder};
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::vcf::read_vcf;
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

fn gzip_member(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("gzip write failed");
    enc.finish().expect("gzip finish failed")
}

/// Compress `parts` as consecutive gzip members, like bgzip does. Current bgzip
/// puts a VCF header in its own first block, so splitting at arbitrary points
/// (including mid-line) is what real files look like.
fn multi_member(parts: &[&[u8]]) -> Vec<u8> {
    parts.iter().flat_map(|p| gzip_member(p)).collect()
}

fn vcf_text() -> String {
    let mut s = String::from(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
    );
    for i in 0..300 {
        s.push_str(&format!(
            "chr1\t{}\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n",
            i * 1000 + 50
        ));
    }
    s
}

#[test]
fn bgzip_style_vcf_is_read_in_full() {
    let text = vcf_text();
    let header_end = text.find("chr1\t").expect("records present");
    let (header, body) = text.as_bytes().split_at(header_end);
    // Header in its own member, records split across members mid-line.
    let mid = body.len() / 3;
    let path = unique_temp_path("bgzip_vcf", "vcf.gz");
    fs::write(
        &path,
        multi_member(&[header, &body[..mid], &body[mid..2 * mid], &body[2 * mid..]]),
    )
    .expect("write failed");

    let plain = unique_temp_path("plain_vcf", "vcf");
    fs::write(&plain, &text).expect("write failed");

    let from_gz = read_vcf(&path, None, 100, None, &[], &[]).expect("bgzip VCF must parse");
    let from_plain = read_vcf(&plain, None, 100, None, &[], &[]).expect("plain VCF must parse");
    assert_eq!(from_gz.rows, from_plain.rows);
    let n_het: usize = from_gz
        .rows
        .iter()
        .map(|r| r.iter().filter(|&&v| v == 1).count())
        .sum();
    assert_eq!(n_het, 300);

    let _ = fs::remove_file(path);
    let _ = fs::remove_file(plain);
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
    let path = unique_temp_path("bgzip_psmcfa", "psmcfa.gz");
    fs::write(
        &path,
        multi_member(&[&bytes[..4], &bytes[4..2000], &bytes[2000..]]),
    )
    .expect("write failed");

    let obs = read_psmcfa(&path, None).expect("bgzip psmcfa must parse");
    assert_eq!(obs.rows.len(), 2);
    assert_eq!(obs.rows[0].len(), 5000);
    assert_eq!(obs.rows[1].len(), 3000);
    let _ = fs::remove_file(path);
}

#[test]
fn bgzip_style_mhs_is_read_in_full() {
    let mut text = String::new();
    for i in 1..=200u64 {
        text.push_str(&format!("chr1\t{}\t100\tAC\n", i * 100));
    }
    let bytes = text.as_bytes();
    let path = unique_temp_path("bgzip_mhs", "mhs.gz");
    fs::write(&path, multi_member(&[&bytes[..37], &bytes[37..]])).expect("write failed");

    let obs = read_mhs(&path, None, 100).expect("bgzip mhs must parse");
    assert_eq!(obs.rows, vec![vec![1u8; 200]]);
    let _ = fs::remove_file(path);
}

#[test]
fn bgzip_style_mask_is_read_in_full() {
    // Callable mask split over two members: [1, 1000] and [1001, 300000].
    let mask_text = "chr1\t0\t1000\nchr1\t1000\t300000\n";
    let bytes = mask_text.as_bytes();
    let mask = unique_temp_path("bgzip_mask", "bed.gz");
    fs::write(&mask, multi_member(&[&bytes[..13], &bytes[13..]])).expect("write failed");
    let plain_mask = unique_temp_path("plain_mask", "bed");
    fs::write(&plain_mask, mask_text).expect("write failed");

    let vcf = unique_temp_path("mask_vcf", "vcf");
    fs::write(&vcf, vcf_text()).expect("write failed");

    let with_gz =
        read_vcf(&vcf, None, 100, None, std::slice::from_ref(&mask), &[]).expect("parse failed");
    let with_plain = read_vcf(
        &vcf,
        None,
        100,
        None,
        std::slice::from_ref(&plain_mask),
        &[],
    )
    .expect("parse failed");
    assert_eq!(with_gz.rows, with_plain.rows);
    // A truncated mask would leave everything after the first block uncallable.
    assert!(with_gz.rows[0].iter().all(|&v| v != 2));

    let _ = fs::remove_file(mask);
    let _ = fs::remove_file(plain_mask);
    let _ = fs::remove_file(vcf);
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
    enc.write_all(data).expect("gzip write failed");
    let mut block = enc.finish().expect("gzip finish failed");
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

#[test]
fn a_bgzip_file_without_its_end_of_file_block_is_an_error() {
    let text = vcf_text();
    let header_end = text.find("chr1\t").expect("records present");
    let (header, body) = text.as_bytes().split_at(header_end);
    // Blocks end at line ends, as bcftools writes them: a file cut after a
    // block decodes cleanly.
    let cut_at = body[..body.len() / 2]
        .iter()
        .rposition(|&b| b == b'\n')
        .unwrap()
        + 1;
    let blocks = [header, &body[..cut_at], &body[cut_at..]];
    let truncated = |e: anyhow::Error| {
        let msg = format!("{e:#}");
        assert!(
            msg.contains("end-of-file marker") && msg.contains("truncated"),
            "{msg}"
        );
    };

    let path = unique_temp_path("bgzf_vcf", "vcf.gz");
    fs::write(&path, bgzf(&blocks, true)).unwrap();
    let full = read_vcf(&path, None, 100, None, &[], &[]).expect("complete BGZF must parse");
    let n_het: usize = full.rows.iter().flatten().filter(|&&v| v == 1).count();
    assert_eq!(n_het, 300);
    for cut in [bgzf(&blocks[..2], false), bgzf(&blocks, false)] {
        fs::write(&path, cut).unwrap();
        truncated(read_vcf(&path, None, 100, None, &[], &[]).unwrap_err());
    }

    // The same for psmcfa input and for masks.
    let psmcfa = unique_temp_path("bgzf_psmcfa", "psmcfa.gz");
    let records = [&b">s1\nTTKT\n"[..], b">s2\nKKTT\n"];
    fs::write(&psmcfa, bgzf(&records, true)).unwrap();
    assert_eq!(read_psmcfa(&psmcfa, None).unwrap().rows.len(), 2);
    fs::write(&psmcfa, bgzf(&records[..1], false)).unwrap();
    truncated(read_psmcfa(&psmcfa, None).unwrap_err());

    let mask = unique_temp_path("bgzf_mask", "bed.gz");
    let rows = [&b"chr1\t0\t1000\n"[..], b"chr1\t1000\t300000\n"];
    fs::write(&mask, bgzf(&rows[..1], false)).unwrap();
    let plain = unique_temp_path("bgzf_mask_vcf", "vcf");
    fs::write(&plain, &text).unwrap();
    truncated(read_vcf(&plain, None, 100, None, std::slice::from_ref(&mask), &[]).unwrap_err());
    fs::write(&mask, bgzf(&rows, true)).unwrap();
    assert!(read_vcf(&plain, None, 100, None, std::slice::from_ref(&mask), &[]).is_ok());

    for p in [path, psmcfa, mask, plain] {
        let _ = fs::remove_file(p);
    }
}

#[test]
fn gzip_is_detected_by_content_not_extension() {
    let text = ">s1\nTTKT\n";
    let gz_no_ext = unique_temp_path("gz_no_ext", "psmcfa");
    fs::write(&gz_no_ext, gzip_member(text.as_bytes())).expect("write failed");
    let plain_with_gz_ext = unique_temp_path("plain_gz_ext", "psmcfa.gz");
    fs::write(&plain_with_gz_ext, text).expect("write failed");

    for path in [&gz_no_ext, &plain_with_gz_ext] {
        let obs = read_psmcfa(path, None).expect("parse failed");
        assert_eq!(obs.rows, vec![vec![0, 0, 1, 0]], "{path:?}");
    }
    let _ = fs::remove_file(gz_no_ext);
    let _ = fs::remove_file(plain_with_gz_ext);
}
