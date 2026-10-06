//! Golden files from the real tools (tests/data/route_a, see its README):
//! simulated reads called with `bcftools mpileup | bcftools call -c`, then
//! `vcfutils.pl vcf2fq | fq2psmcfa` with several option sets, and SNP calls
//! with callable masks through `generate_multihetsep.py`; and a 250x
//! chromosome called with `bcftools call -c -A` (tests/data/route_a_keep_alts).
//! The loader and the writers must reproduce them byte for byte, and the
//! loader's callable bases per bin must be the good bases `fq2psmcfa` reads
//! in vcf2fq's FASTQ.

use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{
    Dataset, InputFormat, InputOptions, SeqFilter, SourceSummary, VcfDetails, VcfMode,
    detect_vcf_mode, load,
};
use psmc_rs::io::psmcfa::{parse_psmcfa, read_psmcfa};
use psmc_rs::io::write::{
    MultihetsepOptions, write_multihetsep, write_multihetsep_to, write_psmcfa,
};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/route_a");

fn fixture(name: &str) -> PathBuf {
    Path::new(DIR).join(name)
}

/// A path no other test of this process uses (tests run in parallel, and the
/// clock alone is too coarse to tell them apart).
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

/// File contents, gunzipped when the name ends in `.gz`.
fn read_bytes(path: &Path) -> Vec<u8> {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if !path.to_string_lossy().ends_with(".gz") {
        return raw;
    }
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(raw.as_slice())
        .read_to_end(&mut out)
        .unwrap();
    out
}

/// One row of expected/option_sets.tsv.
struct OptionSet {
    name: String,
    vcf2fq: HashMap<char, String>,
    fq2psmcfa: HashMap<char, String>,
    input: String,
}

fn flags(text: &str, tool: &str) -> HashMap<char, String> {
    let rest = text.strip_prefix(tool).expect("tool name");
    let words: Vec<&str> = rest.split_whitespace().collect();
    words
        .chunks(2)
        .map(|p| {
            (
                p[0].trim_start_matches('-').chars().next().unwrap(),
                p[1].to_string(),
            )
        })
        .collect()
}

fn option_sets() -> Vec<OptionSet> {
    let text = fs::read_to_string(fixture("expected/option_sets.tsv")).unwrap();
    text.lines()
        .skip(1)
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            OptionSet {
                name: cols[0].to_string(),
                vcf2fq: flags(cols[1], "vcf2fq"),
                fq2psmcfa: flags(cols[2], "fq2psmcfa"),
                input: cols[3].to_string(),
            }
        })
        .collect()
}

impl OptionSet {
    fn get(map: &HashMap<char, String>, key: char, default: u64) -> u64 {
        map.get(&key).map_or(default, |v| v.parse().unwrap())
    }

    /// The loader options equivalent to this row (vcf2fq and fq2psmcfa
    /// defaults where it gives none); the `auto` row uses automatic depth
    /// thresholds instead of the ones the script computed.
    fn input_options(&self) -> InputOptions {
        let v = &self.vcf2fq;
        let f = &self.fq2psmcfa;
        let depth = if self.name == "auto" {
            DepthThresholds::Auto
        } else {
            DepthThresholds::Fixed {
                min: Self::get(v, 'd', 3),
                max: Self::get(v, 'D', 100_000),
            }
        };
        InputOptions {
            format: Some(InputFormat::Vcf),
            vcf_mode: VcfMode::Auto,
            bin_size: Self::get(f, 's', 100) as usize,
            consensus: ConsensusFilters {
                depth,
                min_mq: Self::get(v, 'Q', 10),
                indel_window: Self::get(v, 'l', 5),
                min_qual: Self::get(f, 'q', 10),
            },
            filter: SeqFilter {
                min_good_bases: Self::get(f, 'g', 10_000),
                min_good_fraction: 0.2,
            },
            ..InputOptions::default()
        }
    }

    fn expected_path(&self) -> PathBuf {
        let plain = fixture(&format!("expected/{}.psmcfa", self.name));
        if plain.exists() {
            plain
        } else {
            fixture(&format!("expected/{}.psmcfa.gz", self.name))
        }
    }
}

#[test]
fn consensus_mode_reproduces_vcf2fq_and_fq2psmcfa() {
    let sets = option_sets();
    assert_eq!(sets.len(), 6);
    for set in &sets {
        let data = load(&fixture(&set.input), &set.input_options())
            .unwrap_or_else(|e| panic!("{}: {e:#}", set.name));
        let SourceSummary::Vcf(v) = &data.source else {
            panic!("{}: not a VCF summary", set.name)
        };
        assert_eq!(v.mode.mode, VcfMode::Consensus, "{}", set.name);

        let out = unique_temp_path(&format!("route_a_{}", set.name), "psmcfa");
        write_psmcfa(&out, &data).unwrap();
        let ours = fs::read(&out).unwrap();
        let _ = fs::remove_file(&out);
        let expected = read_bytes(&set.expected_path());
        assert!(
            ours == expected,
            "{}: psmcfa differs from fq2psmcfa's\nours:\n{}\nexpected:\n{}",
            set.name,
            String::from_utf8_lossy(&ours),
            String::from_utf8_lossy(&expected)
        );

        // The same bins as reading fq2psmcfa's file.
        let records = parse_psmcfa(expected.as_slice()).unwrap();
        let rows: Vec<Vec<u8>> = records.iter().map(|r| r.seq.clone()).collect();
        assert_eq!(data.obs.rows, rows, "{}", set.name);
        let names: Vec<&str> = data.seqs.iter().map(|s| s.name.as_str()).collect();
        let expected_names: Vec<&str> = records.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, expected_names, "{}", set.name);
    }
}

/// The records of vcf2fq's FASTQ: name, bases and quality characters.
fn read_fastq(bytes: &[u8]) -> Vec<(String, Vec<u8>, Vec<u8>)> {
    let text = std::str::from_utf8(bytes).expect("vcf2fq writes ASCII");
    let mut lines = text.lines();
    let mut out = Vec::new();
    while let Some(header) = lines.next() {
        let name = header.strip_prefix('@').expect("a FASTQ header");
        let mut seq = Vec::new();
        for line in lines.by_ref() {
            if line == "+" {
                break;
            }
            seq.extend_from_slice(line.as_bytes());
        }
        // Quality lines may start with '@' or '+', so they are read by length.
        let mut qual = Vec::new();
        while qual.len() < seq.len() {
            qual.extend_from_slice(lines.next().expect("a quality line").as_bytes());
        }
        assert_eq!(qual.len(), seq.len(), "{name}");
        out.push((name.to_string(), seq, qual));
    }
    out
}

/// What `fq2psmcfa -q min_qual` reads at one base of vcf2fq's FASTQ:
/// `Some(true)` for a heterozygous base (the IUPAC code of two bases),
/// `Some(false)` for a called homozygous one, `None` for a missing one (`N`,
/// or lowercase, which a quality below `min_qual` makes it).
fn fq2psmcfa_base(base: u8, qual: u8, min_qual: u64) -> Option<bool> {
    let base = if u64::from(qual - 33) < min_qual {
        base.to_ascii_lowercase()
    } else {
        base
    };
    match base {
        b'A' | b'C' | b'G' | b'T' => Some(false),
        b'M' | b'R' | b'W' | b'S' | b'Y' | b'K' => Some(true),
        b'N' => None,
        b if b.is_ascii_lowercase() => None,
        b => panic!("unexpected base '{}' in vcf2fq output", b as char),
    }
}

/// `fq2psmcfa -s s`'s bins of one sequence of bases read by
/// [`fq2psmcfa_base`], and each bin's callable bases: `s` minus its missing
/// bases (a last bin shorter than `s` is judged against the full `s`).
fn bin_bases(bases: &[Option<bool>], s: usize) -> (Vec<u8>, Vec<u16>) {
    bases
        .chunks(s)
        .map(|block| {
            let missing = block.iter().filter(|b| b.is_none()).count();
            let symbol = if 10 * missing > 9 * s {
                2
            } else {
                u8::from(block.contains(&Some(true)))
            };
            (symbol, u16::try_from(s - missing).unwrap())
        })
        .unzip()
}

#[test]
fn consensus_mode_counts_the_good_bases_of_vcf2fq_output() {
    let mut checked_against_fq2psmcfa = 0;
    for set in option_sets() {
        let s = OptionSet::get(&set.fq2psmcfa, 's', 100) as usize;
        let min_qual = OptionSet::get(&set.fq2psmcfa, 'q', 10);
        let fastq = read_fastq(&read_bytes(&fixture(&format!(
            "expected/{}.fq.gz",
            set.name
        ))));
        let expected = parse_psmcfa(read_bytes(&set.expected_path()).as_slice()).unwrap();
        // Every sequence, not only those fq2psmcfa's output filter keeps.
        let opts = InputOptions {
            filter: SeqFilter::DISABLED,
            ..set.input_options()
        };
        let data =
            load(&fixture(&set.input), &opts).unwrap_or_else(|e| panic!("{}: {e:#}", set.name));
        let callable = data.sequence_callable().expect("consensus mode counts");
        assert_eq!(data.obs.callable.as_ref().unwrap().bin_size(), s);
        assert_eq!(data.seqs.len(), fastq.len(), "{}", set.name);
        for (((info, bins), counts), (name, seq, qual)) in
            data.sequences().iter().zip(&callable).zip(&fastq)
        {
            let ctx = format!("{} {name}", set.name);
            assert_eq!(&info.name, name, "{ctx}");
            let bases: Vec<Option<bool>> = seq
                .iter()
                .zip(qual)
                .map(|(&b, &q)| fq2psmcfa_base(b, q, min_qual))
                .collect();
            let (want_bins, want_counts) = bin_bases(&bases, s);
            assert_eq!(**bins, want_bins, "{ctx}: bins");
            assert_eq!(counts.to_vec(), want_counts, "{ctx}: callable bases");
            let good = bases.iter().filter(|b| b.is_some()).count() as u64;
            assert_eq!(info.called, Some(good), "{ctx}");
            // Where fq2psmcfa kept the sequence, the bins counted from the
            // FASTQ are its output.
            if let Some(r) = expected.iter().find(|r| r.name == *name) {
                assert_eq!(r.seq, want_bins, "{ctx}: fq2psmcfa's bins");
                checked_against_fq2psmcfa += 1;
            }
        }
        // Every sequence fq2psmcfa wrote was compared.
        assert!(
            expected
                .iter()
                .all(|r| fastq.iter().any(|(name, ..)| *name == r.name)),
            "{}",
            set.name
        );
    }
    assert!(
        checked_against_fq2psmcfa >= 8,
        "{checked_against_fq2psmcfa}"
    );
}

#[test]
fn the_default_pipeline_keeps_only_the_chromosome_with_10000_called_bases() {
    let set = option_sets()
        .into_iter()
        .find(|s| s.name == "default")
        .unwrap();
    let data = load(&fixture(&set.input), &set.input_options()).unwrap();
    let kept: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(kept, ["chr1"]);
    let chr1 = &data.seqs[0];
    assert!(chr1.called.unwrap() >= 10_000);
    let dropped: Vec<_> = data.dropped.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(dropped, ["chr2", "chr3"]);
    // Every kind of loss the simulation was built to cause shows up.
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Consensus(c) = &v.details else {
        panic!("not consensus mode")
    };
    let lost = &c.lost;
    assert!(lost.no_record >= 400, "{lost:?}");
    assert!(lost.low_depth > 0 && lost.indel_window > 0, "{lost:?}");
    assert_eq!(c.good_bases + lost.total(), c.bases);
    assert_eq!((c.extra_bases_at_pos, c.positions_without_base), (0, 0));
    assert!(c.indel_records > 0 && c.records_without_fq == 0);
    let mq = option_sets()
        .into_iter()
        .find(|s| s.name == "mq_window")
        .unwrap();
    let data = load(&fixture(&mq.input), &mq.input_options()).unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Consensus(c) = &v.details else {
        panic!("not consensus mode")
    };
    assert!(c.lost.low_mq >= 700, "{:?}", c.lost);
}

#[test]
fn auto_depth_matches_the_independent_computation() {
    let text = fs::read_to_string(fixture("expected/auto_depth.tsv")).unwrap();
    let cols: Vec<&str> = text.split_whitespace().collect();
    let mean: f64 = cols[0].parse().unwrap();
    let (min, max): (u64, u64) = (cols[1].parse().unwrap(), cols[2].parse().unwrap());
    let set = option_sets()
        .into_iter()
        .find(|s| s.name == "auto")
        .unwrap();
    let data = load(&fixture(&set.input), &set.input_options()).unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Consensus(c) = &v.details else {
        panic!("not consensus mode")
    };
    assert!(c.auto_depth);
    assert_eq!(c.depth.mean, mean);
    assert_eq!((c.min_depth, c.max_depth), (min, max));
    assert!(c.depth.median > 0.0);
    assert_eq!(c.depth.histogram.iter().sum::<u64>(), c.depth.records);
}

/// tests/data/route_a_keep_alts (see its README): `bcftools call -c -A` at
/// 250x, where most reference sites get an ALT allele.
fn keep_alts(name: &str) -> PathBuf {
    Path::new(DIR).join("../route_a_keep_alts").join(name)
}

#[test]
fn call_c_with_alt_alleles_at_reference_sites_is_read_like_vcf2fq() {
    let vcf = keep_alts("keep_alts.vcf.gz");
    let choice = detect_vcf_mode(&vcf).unwrap();
    assert_eq!(choice.mode, VcfMode::Consensus, "{}", choice.reason);
    // The ALT '.' rule alone would not have chosen consensus mode.
    assert_eq!(
        (
            choice.inspected,
            choice.alt_dot,
            choice.records_with_fq,
            choice.span
        ),
        (3000, 1297, 3000, 3000)
    );

    let text = fs::read_to_string(keep_alts("expected/auto_depth.tsv")).unwrap();
    let cols: Vec<&str> = text.split_whitespace().collect();
    let (min, max): (u64, u64) = (cols[1].parse().unwrap(), cols[2].parse().unwrap());
    let opts = InputOptions {
        bin_size: 50,
        consensus: ConsensusFilters {
            depth: DepthThresholds::Auto,
            ..ConsensusFilters::default()
        },
        filter: SeqFilter {
            min_good_bases: 1000,
            ..SeqFilter::default()
        },
        ..InputOptions::default()
    };
    let data = load(&vcf, &opts).unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Consensus(c) = &v.details else {
        panic!("not consensus mode")
    };
    assert_eq!((c.min_depth, c.max_depth), (min, max));
    let out = unique_temp_path("route_a_keep_alts", "psmcfa");
    write_psmcfa(&out, &data).unwrap();
    let ours = fs::read(&out).unwrap();
    let _ = fs::remove_file(&out);
    let expected = fs::read(keep_alts("expected/keep_alts.psmcfa")).unwrap();
    assert!(
        ours == expected,
        "ours:\n{}\nexpected:\n{}",
        String::from_utf8_lossy(&ours),
        String::from_utf8_lossy(&expected)
    );

    // `call -cv` on the same reads: FQ on every record, but variant sites
    // only, so mask mode.
    let choice = detect_vcf_mode(&keep_alts("variants_c.vcf.gz")).unwrap();
    assert_eq!(choice.mode, VcfMode::Mask, "{}", choice.reason);
    assert_eq!((choice.inspected, choice.records_with_fq), (17, 17));
}

#[test]
fn psmcfa_round_trip_keeps_every_row() {
    for set in option_sets() {
        let data = load(&fixture(&set.input), &set.input_options()).unwrap();
        for ext in ["psmcfa", "psmcfa.gz"] {
            let out = unique_temp_path("route_a_rt", ext);
            write_psmcfa(&out, &data).unwrap();
            let back = read_psmcfa(&out, None);
            let _ = fs::remove_file(&out);
            if data.obs.rows.is_empty() {
                assert!(back.is_err(), "{}: an empty file has no record", set.name);
            } else {
                assert_eq!(back.unwrap().rows, data.obs.rows, "{} {ext}", set.name);
            }
        }
    }
}

/// The three per-chromosome callable masks as one (multi-member gzip) file.
fn combined_mask() -> PathBuf {
    let mut bytes = Vec::new();
    for chrom in ["chr1", "chr2", "chr3"] {
        bytes.extend(fs::read(fixture(&format!("masks/mask_{chrom}.bed.gz"))).unwrap());
    }
    let path = unique_temp_path("route_a_mask", "bed.gz");
    fs::write(&path, bytes).unwrap();
    path
}

fn load_snps(vcf: &str, mask: &Path) -> Dataset {
    let opts = InputOptions {
        format: Some(InputFormat::Vcf),
        masks: vec![mask.to_path_buf()],
        collect_het_sites: true,
        // generate_multihetsep.py has no output filter.
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&fixture(vcf), &opts).unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    assert_eq!(v.mode.mode, VcfMode::Mask, "{vcf}: variants only");
    data
}

#[test]
fn multihetsep_matches_generate_multihetsep() {
    let mask = combined_mask();
    let mut sites = 0;
    for vcf in ["snps", "snps_phased"] {
        let data = load_snps(&format!("{vcf}.vcf.gz"), &mask);
        for chrom in ["chr1", "chr2", "chr3"] {
            let mut ours = Vec::new();
            let opts = MultihetsepOptions {
                sequence: Some(chrom.to_string()),
            };
            write_multihetsep_to(&mut ours, &data, &opts).unwrap();
            let expected = read_bytes(&fixture(&format!("expected/{vcf}.{chrom}.multihetsep")));
            assert!(
                ours == expected,
                "{vcf} {chrom}:\nours:\n{}\nexpected:\n{}",
                String::from_utf8_lossy(&ours),
                String::from_utf8_lossy(&expected)
            );
            sites += expected.iter().filter(|&&b| b == b'\n').count();
        }
    }
    let _ = fs::remove_file(mask);
    assert!(sites >= 60, "only {sites} sites compared");
}

#[test]
fn multihetsep_round_trip_keeps_positions_and_called_counts() {
    let mask = combined_mask();
    let snps = load_snps("snps_phased.vcf.gz", &mask);
    let _ = fs::remove_file(mask);
    let consensus = {
        let set = option_sets()
            .into_iter()
            .find(|s| s.name == "per_base")
            .unwrap();
        let opts = InputOptions {
            collect_het_sites: true,
            ..set.input_options()
        };
        load(&fixture(&set.input), &opts).unwrap()
    };
    for (label, data) in [("mask mode", &snps), ("consensus mode", &consensus)] {
        for ext in ["mhs", "mhs.gz"] {
            let out = unique_temp_path("route_a_mhs", ext);
            write_multihetsep(&out, data, &MultihetsepOptions::default()).unwrap();
            let opts = InputOptions {
                format: Some(InputFormat::Mhs),
                collect_het_sites: true,
                filter: SeqFilter::DISABLED,
                ..InputOptions::default()
            };
            let back = load(&out, &opts).unwrap();
            let _ = fs::remove_file(&out);
            let key = |d: &Dataset| -> Vec<(String, u64, u64, [u8; 2])> {
                d.het_sites
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|h| {
                        (
                            d.seqs[h.seq as usize].name.clone(),
                            h.pos,
                            h.nr_called,
                            h.alleles,
                        )
                    })
                    .collect()
            };
            let (a, b) = (key(data), key(&back));
            assert!(a.len() >= 30, "{label}: {} sites", a.len());
            assert_eq!(a, b, "{label} {ext}");
        }
    }
}

#[test]
fn mask_mode_reports_the_depth_of_the_sample() {
    // `bcftools call -mv` writes no FORMAT/DP; with one sample, INFO/DP is
    // the sample's depth.
    let data = load_snps("snps.vcf.gz", &fixture("masks/mask_chr1.bed.gz"));
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Mask(m) = &v.details else {
        panic!("not mask mode")
    };
    let text = String::from_utf8(read_bytes(&fixture("snps.vcf.gz"))).unwrap();
    let depths: Vec<u64> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let info = l.split('\t').nth(7).unwrap();
            let dp = info.split(';').find_map(|kv| kv.strip_prefix("DP="));
            dp.unwrap().parse().unwrap()
        })
        .collect();
    assert_eq!(m.records, depths.len() as u64);
    assert_eq!(
        (
            m.depth.records,
            m.depth.records_without_dp,
            m.depth_from_info
        ),
        (m.records, 0, m.records)
    );
    let mean = depths.iter().sum::<u64>() as f64 / depths.len() as f64;
    assert!(
        (m.depth.mean - mean).abs() < 1e-9,
        "{} {mean}",
        m.depth.mean
    );
    assert_eq!(m.depth.histogram.iter().sum::<u64>(), m.records);
}
