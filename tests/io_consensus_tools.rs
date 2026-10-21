//! Consensus mode against the real tools on random `bcftools call -c` style
//! VCFs full of edge cases: `vcfutils.pl vcf2fq` (Perl) piped into the C
//! `fq2psmcfa`, compared byte for byte with `write_psmcfa(load(vcf))`, with
//! random options, including `-s 1 -g 0`, which compares every base.
//!
//! Needs perl, bcftools' `vcfutils.pl` and lh3/psmc's `fq2psmcfa`:
//!
//! ```text
//! VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa \
//!   cargo test --release --test io_consensus_tools -- --ignored --nocapture
//! ```
//!
//! `CONSENSUS_FUZZ_CASES` sets the number of random files (default 300).

mod common;

use common::{Rng, VCF_HEADER};
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, VcfMode, load};
use psmc_rs::io::write::write_psmcfa_to;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Copy)]
struct Opts {
    d: u64,
    big_d: u64,
    q_mq: u64,
    l: u64,
    q: u64,
    s: usize,
    g: u64,
}

fn random_opts(rng: &mut Rng) -> Opts {
    Opts {
        d: rng.range(0, 6),
        big_d: rng.range(6, 40),
        q_mq: rng.pick(&[0, 10, 20, 25, 30, 45]),
        l: rng.pick(&[0, 1, 3, 5, 5, 10]),
        q: rng.pick(&[0, 10, 10, 20, 30, 60]),
        s: rng.pick(&[1, 1, 3, 10, 100]),
        g: rng.pick(&[0, 0, 5, 50, 500]),
    }
}

fn number(rng: &mut Rng, max: u64) -> String {
    let int = rng.range(0, max);
    match rng.range(0, 4) {
        0 => int.to_string(),
        1 => format!("{int}.{}", rng.range(0, 9)),
        2 => format!("{int}.{:04}", rng.range(0, 9999)),
        _ => format!("{int}.{:03}", rng.range(0, 999)),
    }
}

/// FQ values, including forms that probe Perl's number parsing and the
/// rounding of the quality character. Negative values with two dots are left
/// out: vcf2fq negates them as strings (`-"-1.2.3"` is `"+1.2.3"`), and the
/// number Perl then reads depends on interpreter state (34 the first time,
/// 32 after that, in one loop).
fn fq_value(rng: &mut Rng, negative: bool) -> String {
    let sign = if negative { "-" } else { "" };
    if rng.chance(0.1) {
        return rng
            .pick(&[
                "-0", "0", ".5", "-.5", "5.", "-5.", "1.2.3", "-.", ".", "-9.501", "-10.501",
                "9.501", "-60.5", "-92.501", "-93.5", "-200", "-x", "--5", "-0.499", "-0.501",
                "-6.501", "6.499", "-1.2x", "-1e5",
            ])
            .to_string();
    }
    format!("{sign}{}", number(rng, 120))
}

fn base(rng: &mut Rng) -> char {
    rng.pick(&['A', 'C', 'G', 'T'])
}

struct Ctx {
    d: u64,
    big_d: u64,
    q_mq: u64,
}

/// INFO of a base (SNP or reference) record.
fn base_info(rng: &mut Rng, c: &Ctx, het: bool, alt_is_dot: bool) -> String {
    // Mostly within the thresholds, so that enough bases are called for
    // sequences to pass fq2psmcfa's output filter, with the boundaries
    // and arbitrary values mixed in.
    let dp = match rng.range(0, 9) {
        0..=6 => rng.range(c.d, c.big_d),
        7 => rng.pick(&[
            c.d.saturating_sub(1),
            c.d,
            c.d + 1,
            c.big_d - 1,
            c.big_d,
            c.big_d + 1,
        ]),
        _ => rng.range(0, 50),
    };
    let mq = match rng.range(0, 9) {
        0..=6 => rng.range(c.q_mq, 60),
        7 => rng.pick(&[c.q_mq.saturating_sub(1), c.q_mq, c.q_mq + 1]),
        _ => rng.range(0, 60),
    };
    let af1 = if het {
        rng.pick(&["0.5", "0.5", "1", "0.499", ".5", "0"])
    } else if alt_is_dot {
        "0"
    } else {
        rng.pick(&["1", "0.5", "0.49", "0", "1.0.1"])
    };
    let negative = !het || rng.chance(0.05);
    let fq = fq_value(rng, negative);
    let mut parts: Vec<String> = Vec::new();
    if !rng.chance(0.03) {
        parts.push(format!("DP={dp}"));
    }
    parts.push("MQ0F=0".to_string());
    if !rng.chance(0.03) {
        parts.push(format!("AF1={af1}"));
    }
    parts.push("AC1=0".to_string());
    parts.push(format!("DP4={dp},0,0,0"));
    if !rng.chance(0.03) {
        parts.push(format!("MQ={mq}"));
    }
    if !rng.chance(0.04) {
        parts.push(format!("FQ={fq}"));
    }
    if rng.chance(0.03) {
        // A key ending in DP before DP itself: the regex takes the first.
        parts.insert(0, format!("XDP={}", rng.range(0, 40)));
    }
    if rng.chance(0.02) {
        parts.push(format!("MQ={}", rng.range(0, 60)));
    }
    if rng.chance(0.02) {
        parts.insert(0, "DP=.".to_string());
    }
    if rng.chance(0.03) {
        let k = rng.range(0, parts.len() as u64 - 1) as usize;
        parts.swap(0, k);
    }
    parts.join(";")
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    /// A reference call (ALT `.`).
    Reference,
    /// A SNP call: heterozygous or homozygous ALT, sometimes multi-allelic.
    Snp,
    /// Symbolic alleles fail vcf2fq's SNP test and become INDEL windows.
    Symbolic,
    /// An INDEL call as bcftools writes it.
    Indel,
    /// A single-base record carrying the INDEL flag.
    IndelFlag,
    /// ALT `.` with a longer REF: neither branch.
    Ignored,
}

fn record(rng: &mut Rng, c: &Ctx, chrom: &str, pos: u64, kind: Kind) -> String {
    let mut reference = base(rng).to_string();
    if rng.chance(0.02) {
        reference = rng.pick(&["N", "a", "c", "M", "R", "B", "n"]).to_string();
    }
    let (reference, alt, info) = match kind {
        Kind::Reference => {
            let info = base_info(rng, c, false, true);
            (reference, ".".to_string(), info)
        }
        Kind::Snp => {
            let mut alt = base(rng).to_string();
            if rng.chance(0.15) {
                alt = format!("{alt},{}", base(rng));
            }
            if rng.chance(0.03) {
                alt = rng.pick(&["X", "a", ".,A", "A,c"]).to_string();
            }
            let het = rng.chance(0.6);
            let info = base_info(rng, c, het, false);
            (reference, alt, info)
        }
        Kind::Symbolic => {
            let alt = rng.pick(&["<*>", "A,<*>", "<DEL>"]).to_string();
            let info = base_info(rng, c, false, false);
            (reference, alt, info)
        }
        Kind::Indel => {
            let len = rng.range(1, 6) as usize;
            let reference: String = (0..len).map(|_| base(rng)).collect();
            let alt: String = (0..rng.range(1, 6)).map(|_| base(rng)).collect();
            let het = rng.chance(0.5);
            let info = format!("INDEL;IDV=3;IMF=0.5;{}", base_info(rng, c, het, false));
            (reference, alt, info)
        }
        Kind::IndelFlag => {
            let info = format!("INDEL;{}", base_info(rng, c, false, false));
            (reference, base(rng).to_string(), info)
        }
        Kind::Ignored => {
            let reference: String = (0..rng.range(2, 4)).map(|_| base(rng)).collect();
            let info = base_info(rng, c, false, true);
            (reference, ".".to_string(), info)
        }
    };
    format!("{chrom}\t{pos}\t.\t{reference}\t{alt}\t30\t.\t{info}\tGT:PL\t0/0:0\n")
}

fn other_kind(rng: &mut Rng) -> Kind {
    rng.pick(&[
        Kind::Indel,
        Kind::Indel,
        Kind::Indel,
        Kind::Symbolic,
        Kind::IndelFlag,
        Kind::Ignored,
    ])
}

/// Positions as `bcftools call -c` writes them, one base record each with
/// an occasional INDEL or other record at the same POS, plus rarer shapes:
/// gaps, a second base record at one POS, and positions without a base
/// record. Each of the last shifts vcf2fq's string by one base.
fn random_vcf(rng: &mut Rng, o: &Opts) -> String {
    let c = Ctx {
        d: o.d,
        big_d: o.big_d,
        q_mq: o.q_mq,
    };
    // One sample: the loader refuses a VCF with several in consensus mode,
    // since vcf2fq would pool them (tests/io_input.rs checks that).
    let mut out = String::from("##fileformat=VCFv4.2\n##source=fuzz\n");
    writeln!(
        out,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
    )
    .unwrap();
    let n_chroms = if rng.chance(0.02) { 0 } else { rng.range(1, 4) };
    let names = ["chr1", "chr2", "0", "X", "scaffold_7"];
    let mut last = "";
    for _ in 0..n_chroms {
        // A name may come back, but not right away (that would restart
        // the same chromosome at a lower POS).
        let mut chrom = rng.pick(&names);
        while chrom == last {
            chrom = rng.pick(&names);
        }
        last = chrom;
        let n = rng.range(1, 2500);
        let mut pos = if rng.chance(0.05) {
            0
        } else {
            rng.range(1, 40)
        };
        for _ in 0..n {
            if rng.chance(0.995) {
                let kind = if rng.chance(0.8) {
                    Kind::Reference
                } else {
                    Kind::Snp
                };
                out.push_str(&record(rng, &c, chrom, pos, kind));
                if rng.chance(0.01) {
                    out.push_str(&record(rng, &c, chrom, pos, Kind::Reference));
                }
            }
            if rng.chance(0.05) {
                let kind = other_kind(rng);
                out.push_str(&record(rng, &c, chrom, pos, kind));
            }
            // About 75% of the positions get records.
            pos += match rng.range(0, 9999) {
                0..=9888 => 1,
                9889..=9988 => rng.range(2, 20),
                9989..=9998 => rng.range(20, 200),
                _ => rng.range(200, 3000),
            };
        }
    }
    if rng.chance(0.05) {
        out = out.replace('\n', "\r\n");
    }
    out
}

/// What the Perl and C tools made of one file.
enum Tools {
    /// fq2psmcfa's output, and the number of empty FASTQ records removed
    /// before running it.
    Output(Vec<u8>, usize),
    /// vcf2fq died (e.g. "substr outside of string").
    Vcf2fqDied,
}

/// Remove the records without bases from vcf2fq's FASTQ. fq2psmcfa cannot
/// read them: kseq reads one byte past an empty quality string, which is the
/// `@` of the next header, so the next record is misread from inside its
/// quality string (or, for an empty first record, it writes through a null
/// sequence buffer and crashes). The reader drops empty sequences instead.
fn drop_empty_records(fq: &[u8]) -> (Vec<u8>, usize) {
    let lines: Vec<&[u8]> = fq.split(|&b| b == b'\n').collect();
    let (mut out, mut dropped, mut i) = (Vec::new(), 0, 0);
    while i < lines.len() && !lines[i].is_empty() {
        let start = i;
        assert_eq!(lines[i].first(), Some(&b'@'), "unexpected FASTQ layout");
        i += 1;
        let mut seq_len = 0;
        while lines[i] != b"+" {
            seq_len += lines[i].len();
            i += 1;
        }
        i += 1;
        let mut qual_len = 0;
        while qual_len < seq_len {
            qual_len += lines[i].len();
            i += 1;
        }
        if seq_len == 0 {
            dropped += 1;
        } else {
            for line in &lines[start..i] {
                out.extend_from_slice(line);
                out.push(b'\n');
            }
        }
    }
    (out, dropped)
}

fn run_tools(vcfutils: &Path, fq2psmcfa: &Path, vcf: &Path, o: &Opts) -> Tools {
    let vcf2fq = Command::new("perl")
        .arg(vcfutils)
        .arg("vcf2fq")
        .args(["-d", &o.d.to_string(), "-D", &o.big_d.to_string()])
        .args(["-Q", &o.q_mq.to_string(), "-l", &o.l.to_string()])
        .arg(vcf)
        .stderr(Stdio::null())
        .output()
        .expect("failed to run perl vcfutils.pl");
    if !vcf2fq.status.success() {
        return Tools::Vcf2fqDied;
    }
    let (fq_text, empty) = drop_empty_records(&vcf2fq.stdout);
    let fq = vcf.with_extension("fq");
    fs::write(&fq, &fq_text).unwrap();
    let psmcfa = Command::new(fq2psmcfa)
        .args(["-q", &o.q.to_string(), "-g", &o.g.to_string()])
        .args(["-s", &o.s.to_string()])
        .arg(&fq)
        .output()
        .expect("failed to run fq2psmcfa");
    let _ = fs::remove_file(&fq);
    assert!(
        psmcfa.status.success(),
        "fq2psmcfa failed on {}: {:?}",
        vcf.display(),
        psmcfa.status
    );
    Tools::Output(psmcfa.stdout, empty)
}

fn ours(vcf: &Path, o: &Opts) -> anyhow::Result<Vec<u8>> {
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: VcfMode::Consensus,
        bin_size: o.s,
        consensus: ConsensusFilters {
            depth: DepthThresholds::Fixed {
                min: o.d,
                max: o.big_d,
            },
            min_mq: o.q_mq,
            indel_window: o.l,
            min_qual: o.q,
        },
        filter: SeqFilter {
            min_good_bases: o.g,
            min_good_fraction: 0.2,
        },
        ..InputOptions::default()
    };
    let data = load(vcf, &opts)?;
    let mut out = Vec::new();
    let seqs = data.sequences();
    write_psmcfa_to(
        &mut out,
        seqs.iter().map(|(i, b)| (i.name.as_str(), b.as_ref())),
    )?;
    Ok(out)
}

fn first_difference(a: &[u8], b: &[u8]) -> String {
    let i = a
        .iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()));
    let lo = i.saturating_sub(80);
    format!(
        "at byte {i} (lengths {} vs {}):\n tools: {:?}\n ours:  {:?}",
        a.len(),
        b.len(),
        String::from_utf8_lossy(&a[lo..(i + 40).min(a.len())]),
        String::from_utf8_lossy(&b[lo..(i + 40).min(b.len())])
    )
}

/// Hand-written files for the cases random files rarely reach.
fn fixed_cases() -> Vec<(&'static str, String, Opts)> {
    let good = |c: &str, from: u64, to: u64| -> String {
        (from..=to)
            .map(|p| format!("{c}\t{p}\t.\tA\t.\t30\t.\tDP=20;MQ=50;FQ=-60\tGT\t0/0\n"))
            .collect()
    };
    let indel =
        |c: &str, p: u64| format!("{c}\t{p}\t.\tAC\tA\t30\t.\tINDEL;DP=20;MQ=50;FQ=40\tGT\t0/1\n");
    let o = Opts {
        d: 3,
        big_d: 100,
        q_mq: 10,
        l: 5,
        q: 10,
        s: 1,
        g: 0,
    };
    vec![
        // The window of the INDEL at 5 starts at base 5 of a 4-base string:
        // vcf2fq dies with "substr outside of string".
        (
            "window past the end",
            format!("{VCF_HEADER}{}{}", good("c", 1, 4), indel("c", 5)),
            Opts { l: 0, ..o },
        ),
        // With l = 1 it starts at the end and is empty.
        (
            "empty window at the end",
            format!("{VCF_HEADER}{}{}", good("c", 1, 4), indel("c", 5)),
            Opts { l: 1, ..o },
        ),
        // A chromosome named 0 is dropped unless it is the last one.
        (
            "chromosome 0",
            format!(
                "{VCF_HEADER}{}{}{}",
                good("0", 1, 30),
                good("c", 1, 30),
                good("0", 1, 30)
            ),
            o,
        ),
        // A chromosome without any base between two others.
        (
            "empty chromosome",
            format!(
                "{VCF_HEADER}{}{}{}",
                good("a", 1, 30),
                indel("b", 1),
                good("c", 1, 30)
            ),
            o,
        ),
        ("no records", VCF_HEADER.to_string(), o),
    ]
}

/// How the tools and the reader compared on one file.
enum Agreement {
    /// Identical psmcfa: its length and the empty FASTQ records removed.
    Same(usize, usize),
    /// Both failed.
    BothFailed,
}

fn compare(tools: (&Path, &Path), vcf: &Path, text: &str, o: &Opts, ctx: &str) -> Agreement {
    fs::write(vcf, text).unwrap();
    let outcome = match (run_tools(tools.0, tools.1, vcf, o), ours(vcf, o)) {
        (Tools::Output(expected, empty), Ok(got)) => {
            assert!(
                expected == got,
                "{ctx}: {}",
                first_difference(&expected, &got)
            );
            Agreement::Same(expected.len(), empty)
        }
        (Tools::Vcf2fqDied, Err(_)) => Agreement::BothFailed,
        (Tools::Vcf2fqDied, Ok(_)) => panic!("{ctx}: vcf2fq failed but the reader did not"),
        (_, Err(e)) => panic!("{ctx}: the reader failed: {e:#}"),
    };
    // A failing case panics above and keeps its file for inspection.
    let _ = fs::remove_file(vcf);
    outcome
}

#[test]
#[ignore = "needs perl, vcfutils.pl in $VCFUTILS and fq2psmcfa in $FQ2PSMCFA"]
fn consensus_mode_matches_vcf2fq_and_fq2psmcfa() {
    let vcfutils = PathBuf::from(std::env::var("VCFUTILS").expect("set VCFUTILS"));
    let fq2psmcfa = PathBuf::from(std::env::var("FQ2PSMCFA").expect("set FQ2PSMCFA"));
    let tools = (vcfutils.as_path(), fq2psmcfa.as_path());
    let cases: u64 = std::env::var("CONSENSUS_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let dir = std::env::temp_dir().join(format!("psmc_consensus_fuzz_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();

    let mut fixed_failed = 0;
    for (name, text, o) in fixed_cases() {
        let vcf = dir.join("fixed.vcf");
        let ctx = format!("fixed case '{name}' {o:?}");
        if let Agreement::BothFailed = compare(tools, &vcf, &text, &o, &ctx) {
            fixed_failed += 1;
        }
    }
    assert_eq!(fixed_failed, 1, "only the window past the end fails");

    let (mut compared, mut both_failed, mut empty_records, mut bytes) = (0, 0, 0, 0usize);
    for case in 0..cases {
        let mut rng = Rng(0xC0FFEE ^ case.wrapping_mul(0x9E37_79B9));
        let o = random_opts(&mut rng);
        let text = random_vcf(&mut rng, &o);
        let vcf = dir.join(format!("case{case}.vcf"));
        let ctx = format!("case {case} {o:?} ({})", vcf.display());
        match compare(tools, &vcf, &text, &o, &ctx) {
            Agreement::Same(len, empty) => {
                bytes += len;
                empty_records += empty;
                compared += 1;
            }
            Agreement::BothFailed => both_failed += 1,
        }
    }
    let _ = fs::remove_dir(&dir);
    eprintln!(
        "{} fixed cases agreed; {compared} random VCFs gave byte-identical psmcfa ({bytes} \
         bytes; {empty_records} empty FASTQ records removed before fq2psmcfa); in \
         {both_failed} more, vcf2fq and the reader both failed",
        fixed_cases().len()
    );
    assert!(compared >= cases * 9 / 10);
}
