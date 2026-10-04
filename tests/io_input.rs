//! The unified loader: format and VCF mode detection, the fq2psmcfa output
//! filter, per-sequence information and heterozygous sites.

use psmc_rs::io::input::{
    DropReason, HetSite, InputFormat, InputOptions, MODE_DETECTION_RECORDS, SeqFilter,
    SourceSummary, VcfDetails, VcfMode, detect_vcf_mode, load,
};
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::io::vcf::read_vcf;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A path no other test of this process uses (tests run in parallel, and the
/// clock alone is too coarse to tell them apart).
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

/// A temporary file removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(prefix: &str, ext: &str, content: &str) -> Self {
        let path = unique_temp_path(prefix, ext);
        fs::write(&path, content).expect("failed to write temp file");
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

const HEADER: &str =
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";

/// An all-sites VCF as `bcftools call -c` writes it, with a heterozygous
/// SNP every `het_every` positions.
fn call_c_vcf(chrom: &str, len: u64, het_every: u64, depth: u64) -> String {
    let mut out = String::from(HEADER);
    for pos in 1..=len {
        if pos % het_every == 0 {
            writeln!(
                out,
                "{chrom}\t{pos}\t.\tA\tG\t50\t.\tDP={depth};MQ=50;AF1=0.5;FQ=40\tGT\t0/1"
            )
            .unwrap();
        } else {
            writeln!(
                out,
                "{chrom}\t{pos}\t.\tA\t.\t50\t.\tDP={depth};MQ=50;AF1=0;FQ=-60\tGT\t0/0"
            )
            .unwrap();
        }
    }
    out
}

fn vcf_opts(mode: VcfMode) -> InputOptions {
    InputOptions {
        format: Some(InputFormat::Vcf),
        vcf_mode: mode,
        ..InputOptions::default()
    }
}

#[test]
fn formats_are_recognised_from_the_content() {
    let vcf = TempFile::new("sniff", "txt", &call_c_vcf("c", 3, 2, 10));
    let psmcfa = TempFile::new("sniff", "txt", "\n>chr1\nTTKN\n");
    let mhs = TempFile::new("sniff", "txt", "chr1\t10\t10\tAC\n");
    let empty = TempFile::new("sniff", "txt", "\n\n");
    assert_eq!(InputFormat::sniff(&vcf.0).unwrap(), InputFormat::Vcf);
    assert_eq!(InputFormat::sniff(&psmcfa.0).unwrap(), InputFormat::Psmcfa);
    assert_eq!(InputFormat::sniff(&mhs.0).unwrap(), InputFormat::Mhs);
    assert!(InputFormat::sniff(&empty.0).is_err());

    let data = load(&psmcfa.0, &InputOptions::default()).unwrap();
    assert_eq!(data.source.format(), InputFormat::Psmcfa);
    assert_eq!(data.obs.rows, vec![vec![0, 0, 1, 2]]);
}

#[test]
fn auto_mode_reads_call_c_output_as_consensus() {
    let vcf = TempFile::new("mode_c", "vcf", &call_c_vcf("c", 200, 10, 20));
    let choice = detect_vcf_mode(&vcf.0).unwrap();
    assert_eq!(choice.mode, VcfMode::Consensus);
    assert!(!choice.forced);
    assert_eq!(
        (choice.inspected, choice.alt_dot, choice.alt_dot_with_fq),
        (200, 180, 180)
    );
    assert!(
        choice.reason.contains("bcftools call -c"),
        "{}",
        choice.reason
    );

    let data = load(
        &vcf.0,
        &InputOptions {
            bin_size: 10,
            filter: SeqFilter::DISABLED,
            ..InputOptions::default()
        },
    )
    .unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    assert_eq!(v.mode.mode, VcfMode::Consensus);
    assert!(matches!(v.details, VcfDetails::Consensus(_)));
    assert_eq!(data.obs.rows, vec![vec![1; 20]]);
}

#[test]
fn auto_mode_reads_variants_only_vcfs_with_masks() {
    let text = format!(
        "{HEADER}c\t5\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\nc\t9\t.\tA\t.\t50\tPASS\tFQ=-3\tGT\t0/0\n\
         c\t12\t.\tA\tG\t50\tPASS\t.\tGT\t1/1\n"
    );
    let vcf = TempFile::new("mode_m", "vcf", &text);
    let choice = detect_vcf_mode(&vcf.0).unwrap();
    assert_eq!(choice.mode, VcfMode::Mask);
    assert_eq!((choice.inspected, choice.alt_dot), (3, 1));
    assert!(choice.reason.contains("variants-only"), "{}", choice.reason);

    let data = load(
        &vcf.0,
        &InputOptions {
            bin_size: 4,
            filter: SeqFilter::DISABLED,
            ..InputOptions::default()
        },
    )
    .unwrap();
    // Identical to the existing mask-mode reader.
    let expected = read_vcf(&vcf.0, None, 4, None, &[], &[]).unwrap();
    assert_eq!(data.obs.rows, expected.rows);
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    let VcfDetails::Mask(m) = &v.details else {
        panic!("not mask mode")
    };
    assert_eq!(m.het_snps, 1);
    // Without a mask, the mask-mode warning is passed on.
    assert!(data.warnings()[0].contains("no callable mask"));
}

#[test]
fn an_all_sites_vcf_without_fq_is_an_error_with_guidance() {
    // `bcftools call -m` without -v: ALT '.' everywhere, no FQ.
    let mut text = String::from(HEADER);
    for pos in 1..=100 {
        writeln!(text, "c\t{pos}\t.\tA\t.\t50\t.\tDP=10;MQ=50\tGT\t0/0").unwrap();
    }
    let vcf = TempFile::new("mode_nofq", "vcf", &text);
    let err = detect_vcf_mode(&vcf.0).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("all-sites VCF"), "{msg}");
    assert!(msg.contains("bcftools call -c"), "{msg}");
    assert!(msg.contains("callable mask"), "{msg}");
    assert!(load(&vcf.0, &vcf_opts(VcfMode::Auto)).is_err());

    // Forcing mask mode reads the genotypes.
    let opts = InputOptions {
        filter: SeqFilter::DISABLED,
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf.0, &opts).unwrap();
    assert_eq!(data.seqs[0].called, Some(100));
}

#[test]
fn mode_detection_thresholds() {
    let line = |pos: u64, alt: &str, info: &str| {
        format!("c\t{pos}\t.\tA\t{alt}\t50\t.\t{info}\tGT\t0/0\n")
    };
    let build = |n_dot: u64, n_dot_fq: u64, n_var: u64| {
        let mut s = String::from(HEADER);
        let mut pos = 0;
        for i in 0..n_dot {
            pos += 1;
            let info = if i < n_dot_fq { "DP=5;FQ=-30" } else { "DP=5" };
            s.push_str(&line(pos, ".", info));
        }
        for _ in 0..n_var {
            pos += 1;
            s.push_str(&line(pos, "C", "DP=5;FQ=30"));
        }
        s
    };
    // Exactly 50% ALT '.' with exactly 90% FQ: consensus.
    let f = TempFile::new("mode_t", "vcf", &build(10, 9, 10));
    assert_eq!(detect_vcf_mode(&f.0).unwrap().mode, VcfMode::Consensus);
    // 89% FQ: error.
    let f = TempFile::new("mode_t", "vcf", &build(100, 89, 100));
    assert!(detect_vcf_mode(&f.0).is_err());
    // Just under 50% ALT '.': mask.
    let f = TempFile::new("mode_t", "vcf", &build(9, 9, 10));
    assert_eq!(detect_vcf_mode(&f.0).unwrap().mode, VcfMode::Mask);
    // No records at all: mask.
    let f = TempFile::new("mode_t", "vcf", HEADER);
    let choice = detect_vcf_mode(&f.0).unwrap();
    assert_eq!((choice.mode, choice.inspected), (VcfMode::Mask, 0));
    // Only the first records are inspected.
    let n = MODE_DETECTION_RECORDS;
    let mut s = build(0, 0, n);
    for pos in n + 1..=3 * n {
        s.push_str(&line(pos, ".", "DP=5;FQ=-30"));
    }
    let f = TempFile::new("mode_t", "vcf", &s);
    let choice = detect_vcf_mode(&f.0).unwrap();
    assert_eq!((choice.mode, choice.inspected), (VcfMode::Mask, n));
}

#[test]
fn forced_modes_are_reported() {
    let vcf = TempFile::new("forced", "vcf", &call_c_vcf("c", 50, 5, 20));
    let data = load(
        &vcf.0,
        &InputOptions {
            filter: SeqFilter::DISABLED,
            ..vcf_opts(VcfMode::Mask)
        },
    )
    .unwrap();
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not a VCF summary")
    };
    assert_eq!(v.mode.mode, VcfMode::Mask);
    assert!(v.mode.forced);
    assert_eq!(v.mode.inspected, 0);
    assert!(v.mode.reason.contains("requested"));

    // Consensus mode reads INFO only, but a sample the header lacks is
    // still an error.
    let opts = InputOptions {
        sample: Some("nobody".to_string()),
        ..vcf_opts(VcfMode::Consensus)
    };
    let err = load(&vcf.0, &opts).unwrap_err();
    assert!(format!("{err:#}").contains("'nobody' not found"), "{err:#}");
}

#[test]
fn the_output_filter_follows_fq2psmcfa() {
    let f = SeqFilter::default();
    assert_eq!((f.min_good_bases, f.min_good_fraction), (10_000, 0.2));
    assert_eq!(f.verdict(10_000, 50_000), None);
    assert_eq!(
        f.verdict(9_999, 20_000),
        Some(DropReason::Filter {
            few_good_bases: true,
            low_good_fraction: false
        })
    );
    assert_eq!(
        f.verdict(10_000, 50_001),
        Some(DropReason::Filter {
            few_good_bases: false,
            low_good_fraction: true
        })
    );
    assert_eq!(
        f.verdict(10, 1_000_000),
        Some(DropReason::Filter {
            few_good_bases: true,
            low_good_fraction: true
        })
    );
    assert_eq!(SeqFilter::DISABLED.verdict(0, 1), None);
    let only_fraction = SeqFilter {
        min_good_bases: 0,
        ..SeqFilter::default()
    };
    assert_eq!(only_fraction.verdict(1, 5), None);
    assert!(only_fraction.verdict(1, 6).is_some());
}

#[test]
fn the_output_filter_applies_to_every_vcf_mode() {
    // c1: 20,000 bases, all called. c2: 5,000 called bases only.
    let mut text = call_c_vcf("c1", 20_000, 100, 20);
    text.push_str(&call_c_vcf("c2", 5_000, 100, 20).replace(HEADER, ""));
    let vcf = TempFile::new("filter_vcf", "vcf", &text);
    for mode in [VcfMode::Consensus, VcfMode::Mask] {
        let data = load(&vcf.0, &vcf_opts(mode)).unwrap();
        let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["c1"], "{mode:?}");
        assert_eq!(data.dropped.len(), 1);
        let d = &data.dropped[0];
        assert_eq!(
            (d.name.as_str(), d.extent, d.good),
            ("c2", Some(5_000), Some(5_000))
        );
        assert_eq!(
            d.reason,
            DropReason::Filter {
                few_good_bases: true,
                low_good_fraction: false
            }
        );
        assert_eq!(data.obs.rows.len(), 1);
        let w = data.warnings();
        assert!(
            w.iter()
                .any(|w| w.contains("1 sequence(s) dropped") && w.contains("c2")),
            "{w:?}"
        );
        // Disabled, both are kept.
        let all = load(
            &vcf.0,
            &InputOptions {
                filter: SeqFilter::DISABLED,
                ..vcf_opts(mode)
            },
        )
        .unwrap();
        assert_eq!(all.seqs.len(), 2);
        assert!(all.dropped.is_empty());
    }
}

#[test]
fn mask_mode_good_bases_are_the_callable_sites() {
    // A 30,000 bp contig: the good bases are the callable positions, and
    // the sequence extends to the ##contig length.
    let text = "##fileformat=VCFv4.2\n##contig=<ID=c,length=30000>\n\
                #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
                c\t100\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n";
    let vcf = TempFile::new("filter_mask", "vcf", text);
    // 10,000 callable positions: enough, and a third of the sequence.
    let mask = TempFile::new("filter_mask", "bed", "c\t0\t10000\n");
    let opts = InputOptions {
        masks: vec![mask.0.clone()],
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf.0, &opts).unwrap();
    assert_eq!(data.seqs.len(), 1);
    let s = &data.seqs[0];
    assert_eq!(
        (s.extent, s.called, s.het_sites),
        (Some(30_000), Some(10_000), Some(1))
    );
    assert_eq!(s.n_bins, 300);

    // 5,999 callable positions: just under a fifth.
    let mask = TempFile::new("filter_mask", "bed", "c\t0\t5999\n");
    let opts = InputOptions {
        masks: vec![mask.0.clone()],
        filter: SeqFilter {
            min_good_bases: 0,
            ..SeqFilter::default()
        },
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf.0, &opts).unwrap();
    assert!(data.seqs.is_empty());
    assert_eq!(
        data.dropped[0].reason,
        DropReason::Filter {
            few_good_bases: false,
            low_good_fraction: true
        }
    );
    // Nothing left: the loader succeeds, ensure_usable explains.
    assert!(data.obs.rows.is_empty());
    let err = data.ensure_usable().unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no sequence of the vcf input is left"),
        "{msg}"
    );
    assert!(msg.contains("c: 30000 bases, 5999 called"), "{msg}");
}

#[test]
fn multihetsep_input_is_filtered_and_described() {
    let text = "A\t20000\t19000\tAC\nA\t20010\t10\tAA,CC\nA\t20020\t10\tAC,CA\n\
                B\t100\t50\tAC\n";
    let mhs = TempFile::new("filter_mhs", "mhs", text);
    let opts = InputOptions {
        format: Some(InputFormat::Mhs),
        bin_size: 10,
        mhs_pair: (0, 2),
        collect_het_sites: true,
        ..InputOptions::default()
    };
    let err = load(&mhs.0, &opts).unwrap_err();
    assert!(format!("{err:#}").contains("haplotypes"), "{err:#}");

    // Four haplotypes; the pair (0, 1) is heterozygous, homozygous, then
    // heterozygous on A.
    let text = "A\t20000\t19000\tACAC\nA\t20010\t10\tAACC\nA\t20030\t10\tCAAA\n\
                B\t100\t50\tACAA\n";
    let mhs = TempFile::new("filter_mhs", "mhs", text);
    let opts = InputOptions {
        mhs_pair: (0, 1),
        ..opts
    };
    let data = load(&mhs.0, &opts).unwrap();
    let SourceSummary::Mhs(m) = &data.source else {
        panic!("not an mhs summary")
    };
    assert_eq!((m.rows, m.het_rows, m.hom_rows, m.haplotypes), (4, 3, 1, 4));
    assert_eq!(m.uncalled_sites, 1000 + 10 + 50);
    assert_eq!(data.seqs.len(), 1);
    let a = &data.seqs[0];
    assert_eq!(
        (a.extent, a.called, a.het_sites),
        (Some(20_030), Some(19_020), Some(2))
    );
    assert_eq!(data.dropped[0].name, "B");
    // Same bins as the multihetsep reader for the kept sequence.
    let all = read_mhs(&mhs.0, None, 10).unwrap();
    assert_eq!(data.obs.rows[0], all.rows[0]);
    // nr_called counts the called sites since the previous heterozygous row.
    let sites = data.het_sites.as_ref().unwrap();
    assert_eq!(
        sites,
        &vec![
            HetSite {
                seq: 0,
                pos: 20_000,
                nr_called: 19_000,
                alleles: *b"AC"
            },
            HetSite {
                seq: 0,
                pos: 20_030,
                nr_called: 20,
                alleles: *b"CA"
            },
        ]
    );
}

#[test]
fn mask_mode_het_sites_carry_alleles_in_gt_order() {
    let text = format!(
        "{HEADER}\
         c\t3\t.\tA\tC\t50\tPASS\t.\tGT\t0/1\n\
         c\t5\t.\tA\tC\t50\tPASS\t.\tGT\t1|0\n\
         c\t7\t.\tA\tC,G\t50\tPASS\t.\tGT\t1/2\n\
         c\t9\t.\tA\tC\t50\tPASS\t.\tGT\t1/1\n\
         c\t10\t.\tA\tAT\t50\tPASS\t.\tGT\t0/1\n\
         c\t12\t.\tA\tC\t50\tPASS\t.\tGT\t1/0\n\
         d\t2\t.\tG\tT\t50\tPASS\t.\tGT\t0|1\n\
         e\t4\t.\tG\tT\t50\tPASS\t.\tGT\t0|1\n"
    );
    let vcf = TempFile::new("het_mask", "vcf", &text);
    // Position 6 of c is not callable.
    let mask = TempFile::new("het_mask", "bed", "c\t0\t5\nc\t6\t20\nd\t0\t1\ne\t0\t10\n");
    let opts = InputOptions {
        masks: vec![mask.0.clone()],
        collect_het_sites: true,
        filter: SeqFilter {
            min_good_bases: 2,
            min_good_fraction: 0.0,
        },
        ..vcf_opts(VcfMode::Mask)
    };
    let data = load(&vcf.0, &opts).unwrap();
    // d has one callable position and is dropped; e becomes sequence 1.
    let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["c", "e"]);
    let got: Vec<_> = data
        .het_sites
        .unwrap()
        .iter()
        .map(|h| {
            (
                h.seq,
                h.pos,
                h.nr_called,
                String::from_utf8(h.alleles.to_vec()).unwrap(),
            )
        })
        .collect();
    let want = [
        (0, 3, 3, "AC"),
        (0, 5, 2, "CA"),
        (0, 7, 1, "CG"),
        // 8, 9 (hom-alt), 11, 12; the het indel at 10 is missing.
        (0, 12, 4, "CA"),
        (1, 4, 4, "GT"),
    ];
    let want: Vec<_> = want
        .iter()
        .map(|&(s, p, n, a)| (s, p, n, a.to_string()))
        .collect();
    assert_eq!(got, want);
    let c = &data.seqs[0];
    assert_eq!(
        (c.extent, c.called, c.het_sites),
        (Some(12), Some(10), Some(4))
    );
}

#[test]
fn het_sites_are_only_collected_on_request_and_not_for_psmcfa() {
    let vcf = TempFile::new("het_req", "vcf", &call_c_vcf("c", 30, 7, 20));
    let opts = InputOptions {
        filter: SeqFilter::DISABLED,
        ..vcf_opts(VcfMode::Auto)
    };
    assert!(load(&vcf.0, &opts).unwrap().het_sites.is_none());
    let opts = InputOptions {
        collect_het_sites: true,
        ..opts
    };
    let sites = load(&vcf.0, &opts).unwrap().het_sites.unwrap();
    let pos: Vec<_> = sites.iter().map(|h| (h.pos, h.nr_called)).collect();
    assert_eq!(pos, vec![(7, 7), (14, 7), (21, 7), (28, 7)]);
    assert!(sites.iter().all(|h| h.alleles == *b"AG"));

    let psmcfa = TempFile::new("het_req", "psmcfa", ">a\nTKT\n");
    let data = load(
        &psmcfa.0,
        &InputOptions {
            collect_het_sites: true,
            ..InputOptions::default()
        },
    )
    .unwrap();
    assert!(data.het_sites.is_none());
}

#[test]
fn psmcfa_input_is_not_filtered() {
    let psmcfa = TempFile::new("psmcfa_in", "psmcfa", ">a\nTKN\n>empty\n>b\nNNNNK\n");
    let data = load(&psmcfa.0, &InputOptions::default()).unwrap();
    let names: Vec<_> = data.seqs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(data.seqs[1].n_bins, 5);
    assert_eq!(data.seqs[1].extent, None);
    assert_eq!(data.dropped.len(), 1);
    assert_eq!(data.dropped[0].reason, DropReason::Empty);
    assert!(data.filter.is_none());
    let SourceSummary::Psmcfa(p) = &data.source else {
        panic!("not a psmcfa summary")
    };
    assert_eq!(
        (
            p.records,
            p.empty_records,
            p.bins,
            p.called_bins,
            p.het_bins
        ),
        (3, 1, 8, 3, 2)
    );
    assert_eq!(data.obs.rows, read_psmcfa(&psmcfa.0, None).unwrap().rows);
    data.ensure_usable().unwrap();
}

#[test]
fn batching_keeps_whole_sequences_available() {
    let psmcfa = TempFile::new("batch", "psmcfa", ">a\nTKNTKNT\n>b\nKK\n");
    let data = load(
        &psmcfa.0,
        &InputOptions {
            batch_size: Some(3),
            ..InputOptions::default()
        },
    )
    .unwrap();
    assert_eq!(data.obs.rows.len(), 4);
    assert_eq!(data.obs.row_starts, vec![true, false, false, true]);
    let seqs = data.sequences();
    assert_eq!(seqs[0].1.as_ref(), &[0, 1, 2, 0, 1, 2, 0]);
    assert_eq!(seqs[1].1.as_ref(), &[1, 1]);
    assert_eq!((data.n_bins(), data.het_bins()), (9, 4));

    let err = load(
        &psmcfa.0,
        &InputOptions {
            batch_size: Some(0),
            ..InputOptions::default()
        },
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("batch_size"));
}

#[test]
fn input_without_heterozygous_bins_loads_but_is_not_usable() {
    let psmcfa = TempFile::new("nohet", "psmcfa", ">a\nTTNT\n");
    let data = load(&psmcfa.0, &InputOptions::default()).unwrap();
    assert!(
        data.warnings()
            .iter()
            .any(|w| w.contains("no bin is heterozygous"))
    );
    let err = data.ensure_usable().unwrap_err();
    assert!(format!("{err:#}").contains("heterozygous"), "{err:#}");

    // Mask mode keeps the reader's detailed message.
    let text = format!("{HEADER}c\t5\t.\tA\tC\t50\tq10\t.\tGT\t0/1\n");
    let vcf = TempFile::new("nohet", "vcf", &text);
    let data = load(
        &vcf.0,
        &InputOptions {
            filter: SeqFilter::DISABLED,
            ..vcf_opts(VcfMode::Mask)
        },
    )
    .unwrap();
    let err = data.ensure_usable().unwrap_err();
    assert!(
        format!("{err:#}").contains("FILTER other than PASS"),
        "{err:#}"
    );
}
