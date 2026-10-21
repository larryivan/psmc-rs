//! Throughput of consensus mode on a large all-sites VCF. Writes a
//! deterministic `bcftools call -c` style file (5 chromosomes of 1M records
//! by default, about 0.5 GB) and times the loader on it:
//!
//! ```text
//! cargo test --release --test io_consensus_perf -- --ignored --nocapture
//! ```
//!
//! `CONSENSUS_PERF_RECORDS` sets the number of records per chromosome and
//! `CONSENSUS_PERF_VCF` keeps the file at that path. An existing file is
//! reused as it is, so a compressed copy (e.g. from bgzip) can be timed too;
//! the throughput in MB/s refers to the file's size on disk.

mod common;

use common::Rng;
use psmc_rs::io::common::open_reader;
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, VcfMode, load};
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

fn write_vcf(path: &PathBuf, chroms: usize, per_chrom: u64) -> u64 {
    let mut w = BufWriter::with_capacity(1 << 20, File::create(path).unwrap());
    writeln!(w, "##fileformat=VCFv4.2").unwrap();
    writeln!(
        w,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample"
    )
    .unwrap();
    let mut rng = Rng(42);
    let mut n = 0;
    for c in 1..=chroms {
        let mut pos = 0;
        for _ in 0..per_chrom {
            pos += if rng.range(0, 1999) == 0 {
                rng.range(1, 500)
            } else {
                1
            };
            let reference = rng.pick(b"ACGT") as char;
            let dp = rng.range(10, 49);
            let mq = rng.range(40, 60);
            if rng.range(0, 999) == 0 {
                writeln!(
                    w,
                    "chr{c}\t{pos}\t.\t{reference}\tG\t222\t.\tDP={dp};VDB=0.5;SGB=-0.69;\
                     RPBZ=0.4;MQBZ=-1.2;BQBZ=0;SCBZ=0;MQ0F=0;AF1=0.5;AC1=1;DP4=5,5,6,4;MQ={mq};\
                     FQ=180.5;PV4=1,1,0.2,1\tGT:PL:AD\t0/1:255,0,255:10,10"
                )
                .unwrap();
            } else {
                writeln!(
                    w,
                    "chr{c}\t{pos}\t.\t{reference}\t.\t{}.9946\t.\tDP={dp};MQ0F=0;AF1=0;AC1=0;\
                     DP4={dp},0,0,0;MQ={mq};FQ=-{}.9865\tGT:PL:AD\t0/0:0:{dp}",
                    30 + dp,
                    27 + dp
                )
                .unwrap();
            }
            n += 1;
            if rng.range(0, 4999) == 0 {
                writeln!(
                    w,
                    "chr{c}\t{pos}\t.\t{reference}AT\t{reference}\t99\t.\tINDEL;IDV=10;IMF=0.3;\
                     DP={dp};MQ0F=0;AF1=0.5;AC1=1;DP4=10,10,3,5;MQ={mq};FQ=102.4\tGT:PL:AD\t\
                     0/1:137,0,173:20,8"
                )
                .unwrap();
                n += 1;
            }
        }
    }
    w.flush().unwrap();
    n
}

#[test]
#[ignore = "benchmark: writes a ~0.5 GB VCF"]
fn consensus_mode_throughput() {
    let per_chrom: u64 = std::env::var("CONSENSUS_PERF_RECORDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);
    let (path, keep) = match std::env::var("CONSENSUS_PERF_VCF") {
        Ok(p) => (PathBuf::from(p), true),
        Err(_) => (
            std::env::temp_dir().join(format!("psmc_consensus_perf_{}.vcf", std::process::id())),
            false,
        ),
    };
    let t = Instant::now();
    let records = if path.exists() {
        let mut reader = open_reader(&path).unwrap();
        let (mut line, mut n) = (Vec::new(), 0);
        while reader.read_until(b'\n', &mut line).unwrap() > 0 {
            n += u64::from(line.first() != Some(&b'#'));
            line.clear();
        }
        n
    } else {
        write_vcf(&path, 5, per_chrom)
    };
    let mb = std::fs::metadata(&path).unwrap().len() as f64 / 1e6;
    eprintln!(
        "{records} records, {mb:.0} MB ({:.1} s to write or count)",
        t.elapsed().as_secs_f64()
    );
    let base = InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: VcfMode::Consensus,
        ..InputOptions::default()
    };
    let auto = InputOptions {
        consensus: ConsensusFilters {
            depth: DepthThresholds::Auto,
            ..ConsensusFilters::default()
        },
        ..base.clone()
    };
    let detected = InputOptions {
        vcf_mode: VcfMode::Auto,
        ..base.clone()
    };
    for (label, opts) in [
        ("consensus, fixed depth", &base),
        ("consensus, auto mode", &detected),
        ("consensus, auto depth (2 passes)", &auto),
    ] {
        let mut times = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            let data = load(&path, opts).unwrap();
            times.push(t.elapsed().as_secs_f64());
            assert_eq!(data.seqs.len(), 5);
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: median {:.2} s, {:.2} M records/s, {:.0} MB/s (runs: {times:.2?})",
            times[1],
            records as f64 / times[1] / 1e6,
            mb / times[1]
        );
    }
    if !keep {
        let _ = std::fs::remove_file(&path);
    }
}
