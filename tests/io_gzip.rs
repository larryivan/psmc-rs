//! Compressed inputs: every reader must decode all gzip members (bgzip files are
//! a chain of independent members) and must not rely on the `.gz` extension.

use flate2::Compression;
use flate2::write::GzEncoder;
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

    let with_gz = read_vcf(&vcf, None, 100, None, &[mask.clone()], &[]).expect("parse failed");
    let with_plain =
        read_vcf(&vcf, None, 100, None, &[plain_mask.clone()], &[]).expect("parse failed");
    assert_eq!(with_gz.rows, with_plain.rows);
    // A truncated mask would leave everything after the first block uncallable.
    assert!(with_gz.rows[0].iter().all(|&v| v != 2));

    let _ = fs::remove_file(mask);
    let _ = fs::remove_file(plain_mask);
    let _ = fs::remove_file(vcf);
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
