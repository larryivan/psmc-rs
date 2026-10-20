//! `load` reads its input once, so a pipe or FIFO (`/dev/stdin`,
//! `<(bcftools call -c ...)`) gives the same dataset as the file it carries.
//! The inputs are the route A fixtures (tests/data/route_a).
#![cfg(unix)]

mod common;

use common::TempDir;
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{Dataset, InputFormat, InputOptions, SeqFilter, VcfMode, load};
use psmc_rs::io::write::write_psmcfa_to;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// A file of tests/data/route_a.
fn fixture(name: &str) -> PathBuf {
    common::fixture(&format!("route_a/{name}"))
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(bytes)
        .read_to_end(&mut out)
        .unwrap();
    out
}

/// `load` of a FIFO whose writer sends `content` once, to the first reader
/// that opens it, as a shell pipe or process substitution does. It runs on
/// another thread: a loader that opens the FIFO a second time waits for a
/// writer that never comes, which fails the test instead of hanging it.
fn load_fifo(content: Vec<u8>, opts: &InputOptions) -> anyhow::Result<Dataset> {
    let dir = TempDir::new("io_pipes");
    let path = dir.path("in.fifo");
    let status = Command::new("mkfifo")
        .arg(&path)
        .status()
        .expect("failed to run mkfifo");
    assert!(status.success(), "mkfifo {} failed", path.display());
    let target = path.clone();
    thread::spawn(move || {
        // Opening blocks until a reader opens the other end. A reader that
        // stops early makes the write fail (EPIPE), which is fine here.
        if let Ok(mut f) = fs::OpenOptions::new().write(true).open(&target) {
            let _ = f.write_all(&content);
        }
    });
    let (tx, rx) = mpsc::channel();
    let opts = opts.clone();
    thread::spawn(move || {
        let _ = tx.send(load(&path, &opts));
    });
    rx.recv_timeout(Duration::from_secs(120))
        .expect("load() did not return: it opened the FIFO a second time")
}

fn psmcfa_bytes(data: &Dataset) -> Vec<u8> {
    let seqs = data.sequences();
    let mut out = Vec::new();
    write_psmcfa_to(
        &mut out,
        seqs.iter()
            .map(|(s, bins)| (s.name.as_str(), bins.as_ref())),
    )
    .unwrap();
    out
}

fn assert_same(piped: &Dataset, file: &Dataset) {
    assert_eq!(piped.obs.rows, file.obs.rows);
    assert_eq!(piped.obs.row_starts, file.obs.row_starts);
    assert_eq!(piped.seqs, file.seqs);
    assert_eq!(piped.dropped, file.dropped);
    assert_eq!(piped.source, file.source);
    assert_eq!(piped.het_sites, file.het_sites);
    assert_eq!(piped.filter, file.filter);
}

#[test]
fn an_all_sites_vcf_from_a_fifo_is_read_like_the_file() {
    let path = fixture("allsites.vcf.gz");
    let gz = fs::read(&path).unwrap();
    // 15,819 records: mode detection keeps the first 10,000 in memory, and
    // the reader must see them again before the rest of the stream.
    let option_sets = [
        InputOptions::default(),
        InputOptions {
            vcf_mode: VcfMode::Consensus,
            sample: Some("aln.sam".to_string()),
            collect_het_sites: true,
            ..InputOptions::default()
        },
        InputOptions {
            format: Some(InputFormat::Vcf),
            vcf_mode: VcfMode::Mask,
            bin_size: 50,
            ..InputOptions::default()
        },
    ];
    let default_psmcfa = fs::read(fixture("expected/default.psmcfa")).unwrap();
    for (i, opts) in option_sets.iter().enumerate() {
        let file = load(&path, opts).unwrap();
        for content in [gz.clone(), gunzip(&gz)] {
            let piped = load_fifo(content, opts).unwrap();
            assert_same(&piped, &file);
            // The defaults are those of vcf2fq and fq2psmcfa.
            if i == 0 {
                assert_eq!(psmcfa_bytes(&piped), default_psmcfa);
            }
        }
    }
}

#[test]
fn a_missing_sample_is_reported_for_a_fifo() {
    let gz = fs::read(fixture("allsites.vcf.gz")).unwrap();
    let opts = InputOptions {
        sample: Some("nobody".to_string()),
        ..InputOptions::default()
    };
    let err = load_fifo(gz, &opts).unwrap_err();
    assert!(
        format!("{err:#}").contains("VCF sample 'nobody' not found in header"),
        "{err:#}"
    );
}

#[test]
fn a_variants_vcf_with_a_mask_from_a_fifo_is_read_like_the_file() {
    let path = fixture("snps.vcf.gz");
    let opts = InputOptions {
        masks: vec![fixture("masks/mask_chr1.bed.gz")],
        collect_het_sites: true,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let file = load(&path, &opts).unwrap();
    assert!(!file.seqs.is_empty());
    let piped = load_fifo(fs::read(&path).unwrap(), &opts).unwrap();
    assert_same(&piped, &file);
}

#[test]
fn multihetsep_and_psmcfa_from_a_fifo_are_read_like_the_files() {
    let mhs = fixture("expected/snps.chr1.multihetsep");
    let opts = InputOptions {
        collect_het_sites: true,
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let file = load(&mhs, &opts).unwrap();
    let piped = load_fifo(fs::read(&mhs).unwrap(), &opts).unwrap();
    assert_same(&piped, &file);

    let psmcfa = fixture("expected/default.psmcfa");
    let file = load(&psmcfa, &InputOptions::default()).unwrap();
    let piped = load_fifo(fs::read(&psmcfa).unwrap(), &InputOptions::default()).unwrap();
    assert_same(&piped, &file);
    assert_eq!(psmcfa_bytes(&piped), fs::read(&psmcfa).unwrap());
}

#[test]
fn automatic_depth_thresholds_need_a_regular_file() {
    let opts = InputOptions {
        consensus: ConsensusFilters {
            depth: DepthThresholds::Auto,
            ..ConsensusFilters::default()
        },
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let path = fixture("allsites.vcf.gz");
    let file = load(&path, &opts).unwrap();
    assert!(!file.seqs.is_empty());

    let err = load_fifo(fs::read(&path).unwrap(), &opts).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not a regular file") && msg.contains("give the depth thresholds"),
        "{msg}"
    );
}
