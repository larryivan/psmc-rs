//! Consensus mode, rule by rule: hand-made `bcftools call -c` style records
//! read with one base per bin, so every base of vcf2fq's string is visible.
//! `tests/io_consensus_tools.rs` checks the same rules against the real
//! Perl and C tools on random files.

use psmc_rs::io::consensus::{
    ConsensusFilters, ConsensusOptions, ConsensusRead, DepthThresholds, LossReason,
    auto_depth_thresholds, read_vcf_consensus, scan_depth,
};
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

const HEADER: &str =
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";

/// INFO of a confident reference call that passes the default filters.
const GOOD: &str = "DP=20;MQ=50;AF1=0;FQ=-60";

fn rec(chrom: &str, pos: u64, reference: &str, alt: &str, info: &str) -> String {
    format!("{chrom}\t{pos}\t.\t{reference}\t{alt}\t30\t.\t{info}\tGT\t0/0\n")
}

/// Reference calls with [`GOOD`] INFO at `from..=to`.
fn good_run(chrom: &str, from: u64, to: u64) -> String {
    (from..=to).map(|p| rec(chrom, p, "A", ".", GOOD)).collect()
}

fn opts() -> ConsensusOptions {
    ConsensusOptions {
        bin_size: 1,
        ..ConsensusOptions::default()
    }
}

fn read(records: &str, opts: &ConsensusOptions) -> anyhow::Result<ConsensusRead> {
    let path = unique_temp_path("consensus", "vcf");
    fs::write(&path, format!("{HEADER}{records}")).expect("failed to write VCF");
    let out = read_vcf_consensus(&path, opts);
    let _ = fs::remove_file(path);
    out
}

/// One character per base: T (called), K (heterozygous) or N (missing).
fn bases(read: &ConsensusRead, seq: usize) -> String {
    read.seqs[seq]
        .bins
        .iter()
        .map(|&b| ['T', 'K', 'N'][b as usize])
        .collect()
}

#[test]
fn snp_or_reference_branch_follows_vcf2fq() {
    let records = [
        rec("c", 1, "A", ".", GOOD),
        rec("c", 2, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        // Several ALTs: the first one counts.
        rec("c", 3, "A", "G,T", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        // A symbolic ALT fails the test, so the record is an INDEL window.
        rec("c", 4, "A", "<*>", GOOD),
        // The INDEL flag anywhere in INFO, even with SNP alleles.
        rec("c", 5, "A", "C", &format!("INDEL;{GOOD}")),
        // ALT '.' with a longer REF: neither branch, so no base for POS 6.
        rec("c", 6, "AC", ".", GOOD),
    ]
    .concat();
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            indel_window: 2,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    let s = &r.summary;
    assert_eq!(
        (
            s.records,
            s.base_records,
            s.indel_records,
            s.ignored_records
        ),
        (6, 3, 2, 1)
    );
    // Six positions with records but only three bases: POS 4 to 6 have none.
    assert_eq!(s.positions_without_base, 3);
    assert_eq!(r.seqs[0].bases, 3);
    // The window of the INDEL at 4 (l = 2) starts at string index 2, the
    // third base.
    assert_eq!(bases(&r, 0), "TKN");
    assert_eq!(s.lost.indel_window, 1);
    let het: Vec<_> = r.seqs[0].het_sites().collect();
    assert!(het.is_empty(), "het sites are only collected on request");
}

#[test]
fn homozygous_and_heterozygous_calls() {
    let records = [
        // FQ < 0: homozygous. REF when AF1 < 0.5 or ALT is '.'.
        rec("c", 1, "A", "C", "DP=20;MQ=50;AF1=0.49;FQ=-40"),
        // AF1 >= 0.5: the ALT base, still called (not heterozygous).
        rec("c", 2, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=-40"),
        // No AF1 reads as 0.
        rec("c", 3, "A", "C", "DP=20;MQ=50;FQ=-40"),
        // FQ >= 0: the IUPAC code of REF and ALT, heterozygous.
        rec("c", 4, "G", "T", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        // ... or N when they have none (same base, ALT '.', lowercase).
        rec("c", 5, "G", "G", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        rec("c", 6, "G", ".", "DP=20;MQ=50;AF1=0;FQ=40"),
        rec("c", 7, "g", "T", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        // A homozygous IUPAC REF is read as heterozygous by fq2psmcfa, and
        // REF N as missing.
        rec("c", 8, "M", ".", "DP=20;MQ=50;AF1=0;FQ=-40"),
        rec("c", 9, "N", ".", "DP=20;MQ=50;AF1=0;FQ=-40"),
        // fq2psmcfa's table reads X as a called base.
        rec("c", 10, "X", ".", "DP=20;MQ=50;AF1=0;FQ=-40"),
    ]
    .concat();
    let r = read(
        &records,
        &ConsensusOptions {
            collect_het_sites: true,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(bases(&r, 0), "TTTKNNNKNT");
    let lost = &r.seqs[0].lost;
    assert_eq!(lost.no_het_code, 3);
    assert_eq!(lost.not_a_base, 1);
    // REF then ALT for vcf2fq's heterozygous branch, the IUPAC pair (in
    // ACGT order) for an ambiguous REF.
    let het: Vec<_> = r.seqs[0].het_sites().collect();
    // Bases 5 to 7 are missing, so the second site has nr_called 1.
    assert_eq!(het, vec![(4, 4, *b"GT"), (8, 1, *b"AC")]);
}

#[test]
fn missing_fq_is_a_heterozygous_call_of_quality_zero() {
    let records = [
        rec("c", 1, "A", "C", "DP=20;MQ=50;AF1=0.5"),
        rec("c", 2, "A", ".", "DP=20;MQ=50;AF1=0"),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    // Quality 0 is below -q 10.
    assert_eq!(bases(&r, 0), "NN");
    assert_eq!(r.summary.records_without_fq, 2);
    assert_eq!(r.summary.lost.low_quality, 2);
    assert!(
        r.summary
            .warnings()
            .iter()
            .any(|w| w.contains("no INFO/FQ"))
    );

    // With -q 0 the first is heterozygous; the second has no code.
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            min_qual: 0,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(bases(&r, 0), "KN");
}

#[test]
fn depth_and_mapping_quality_bounds_are_inclusive() {
    let info = |dp: &str, mq: &str| format!("{dp}{mq}AF1=0;FQ=-60");
    let records = [
        rec("c", 1, "A", ".", &info("DP=3;", "MQ=10;")),
        rec("c", 2, "A", ".", &info("DP=100000;", "MQ=10;")),
        rec("c", 3, "A", ".", &info("DP=2;", "MQ=10;")),
        rec("c", 4, "A", ".", &info("DP=100001;", "MQ=10;")),
        rec("c", 5, "A", ".", &info("DP=20;", "MQ=9;")),
        rec("c", 6, "A", ".", &info("", "MQ=50;")),
        rec("c", 7, "A", ".", &info("DP=20;", "")),
        // MQ is checked first.
        rec("c", 8, "A", ".", &info("DP=1;", "MQ=1;")),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    assert_eq!(bases(&r, 0), "TTNNNNNN");
    let lost = &r.summary.lost;
    assert_eq!(
        (lost.low_depth, lost.high_depth, lost.low_mq),
        (2, 1, 3),
        "{lost:?}"
    );

    let o = ConsensusOptions {
        filters: ConsensusFilters {
            depth: DepthThresholds::Fixed { min: 2, max: 3 },
            min_mq: 9,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(bases(&r, 0), "TNTNNNNN");
}

#[test]
fn info_values_are_found_like_vcf2fq_regular_expressions() {
    let records = [
        // /DP=(\d+)/ matches inside XDP first.
        rec("c", 1, "A", ".", "XDP=1;DP=20;MQ=50;FQ=-60"),
        // A DP= without digits is skipped and the next one is used.
        rec("c", 2, "A", ".", "DP=.;DP=20;MQ=50;FQ=-60"),
        // The first MQ= wins.
        rec("c", 3, "A", ".", "DP=20;MQ=5;MQ=50;FQ=-60"),
        // MQ0F= and DP4= are not MQ= and DP=.
        rec("c", 4, "A", ".", "MQ0F=0;DP4=1,2,3,4;DP=20;MQ=50;FQ=-60"),
        // FQ=-x is no match, so FQ is missing (quality 0).
        rec("c", 5, "A", ".", "DP=20;MQ=50;FQ=-x"),
        // Perl reads the longest number prefix: -1.5.2 is -1.5.
        rec("c", 6, "A", "C", "DP=20;MQ=50;AF1=1.0.1;FQ=-45.5.2"),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    assert_eq!(bases(&r, 0), "NTNTNT");
    let lost = &r.summary.lost;
    assert_eq!((lost.low_depth, lost.low_mq, lost.low_quality), (1, 1, 1));
}

#[test]
fn quality_is_rounded_and_capped_like_vcf2fq() {
    // int(|FQ| + 33.499), at most 126; fq2psmcfa needs quality - 33 >= q.
    let records = [
        rec("c", 1, "A", ".", "DP=20;MQ=50;FQ=-9.5"),
        rec("c", 2, "A", ".", "DP=20;MQ=50;FQ=-9.6"),
        rec("c", 3, "A", ".", "DP=20;MQ=50;FQ=-500"),
        rec("c", 4, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=9.6"),
    ]
    .concat();
    assert_eq!(bases(&read(&records, &opts()).unwrap(), 0), "NTTK");
    let at = |q| ConsensusOptions {
        filters: ConsensusFilters {
            min_qual: q,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    // The cap: 126 - 33 = 93.
    assert_eq!(bases(&read(&records, &at(93)).unwrap(), 0), "NNTN");
    assert_eq!(bases(&read(&records, &at(94)).unwrap(), 0), "NNNN");
}

#[test]
fn indel_windows_cover_l_bases_on_each_side() {
    // An INDEL at POS 10 with a 3-base REF masks 10-5+1 ..= 10+3+5, i.e.
    // positions 6..=18, after the chromosome is read.
    let records = [
        good_run("c", 1, 10),
        rec("c", 10, "ACG", "A", &format!("INDEL;{GOOD}")),
        good_run("c", 11, 25),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    assert_eq!(
        bases(&r, 0),
        format!("{}{}{}", "T".repeat(5), "N".repeat(13), "T".repeat(7))
    );
    assert_eq!(r.summary.lost.indel_window, 13);

    // l = 0 leaves POS itself and masks the REF bases after it.
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            indel_window: 0,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(
        bases(&r, 0),
        format!("{}NNN{}", "T".repeat(10), "T".repeat(12))
    );

    // Windows are clipped at both ends of the chromosome and overlap freely.
    let indel = |pos| rec("c", pos, "AC", "A", &format!("INDEL;{GOOD}"));
    let records = [
        good_run("c", 1, 2),
        indel(2),
        good_run("c", 3, 3),
        indel(3),
        good_run("c", 4, 12),
    ]
    .concat();
    // 1..=9 and 1..=10 merge; 11 and 12 stay.
    assert_eq!(bases(&read(&records, &opts()).unwrap(), 0), "NNNNNNNNNNTT");
}

#[test]
fn a_window_starting_beyond_the_string_is_an_error_as_in_perl() {
    // POS 5 has only an INDEL: the string ends at 4 bases, and with l = 0
    // the window starts at index 5. Perl dies with "substr outside of
    // string".
    let records = [
        good_run("c", 1, 4),
        rec("c", 5, "AC", "A", &format!("INDEL;{GOOD}")),
    ]
    .concat();
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            indel_window: 0,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let err = read(&records, &o).unwrap_err();
    assert!(
        format!("{err:#}").contains("substr outside of string"),
        "{err:#}"
    );
    // With l = 1 the window starts at index 4, the end of the string, and
    // is merely empty.
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            indel_window: 1,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(bases(&r, 0), "TTTT");
}

#[test]
fn gaps_are_missing_and_the_sequence_ends_at_the_last_record() {
    let records = [good_run("c", 3, 4), good_run("c", 8, 8)].concat();
    let r = read(&records, &opts()).unwrap();
    assert_eq!(bases(&r, 0), "NNTTNNNT");
    assert_eq!(r.seqs[0].bases, 8);
    assert_eq!(r.summary.lost.no_record, 5);
}

#[test]
fn bases_are_appended_so_duplicate_positions_shift_the_rest() {
    let records = [
        good_run("c", 1, 2),
        // A second base record at POS 2 adds a base; POS 3 lands at 4.
        rec("c", 2, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        rec("c", 3, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=40"),
    ]
    .concat();
    let r = read(
        &records,
        &ConsensusOptions {
            collect_het_sites: true,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(bases(&r, 0), "TTKK");
    assert_eq!(r.summary.extra_bases_at_pos, 1);
    let het: Vec<_> = r.seqs[0].het_sites().map(|h| (h.0, h.1)).collect();
    assert_eq!(het, vec![(3, 3), (4, 1)]);
    assert!(r.summary.warnings().iter().any(|w| w.contains("shifts")));
}

#[test]
fn unsorted_positions_are_an_error() {
    let records = [good_run("c", 5, 6), good_run("c", 4, 4)].concat();
    let err = read(&records, &opts()).unwrap_err();
    assert!(format!("{err:#}").contains("unsorted"), "{err:#}");
}

#[test]
fn chromosome_blocks_follow_vcf2fq() {
    let records = [
        good_run("a", 1, 3),
        good_run("0", 1, 2),
        good_run("b", 1, 1),
        good_run("a", 1, 2),
        good_run("0", 1, 4),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    let names: Vec<_> = r.seqs.iter().map(|s| s.name.as_str()).collect();
    // '0' is false in Perl: dropped unless it is the last chromosome. A
    // chromosome that comes back starts a new sequence.
    assert_eq!(names, ["a", "b", "a", "0"]);
    assert_eq!(r.seqs[3].bases, 4);
    assert_eq!(r.summary.discarded_chroms, ["0"]);
    assert_eq!(r.summary.repeated_chroms, ["a", "0"]);
    assert_eq!(r.summary.warnings().len(), 2);
}

#[test]
fn whitespace_and_line_endings_are_split_like_perl() {
    let crlf = good_run("c", 1, 3).replace('\n', "\r\n");
    let r = read(&format!("{crlf}\n   \n"), &opts()).unwrap();
    assert_eq!(bases(&r, 0), "TTT");
    let err = read("c\t1\t.\tA\t.\n", &opts()).unwrap_err();
    assert!(format!("{err:#}").contains("at least 8 fields"), "{err:#}");
}

#[test]
fn every_base_is_good_or_lost_for_one_reason() {
    let records = [
        good_run("c", 1, 20),
        rec("c", 21, "A", ".", "DP=1;MQ=50;FQ=-60"),
        rec("c", 22, "A", ".", "DP=20;MQ=1;FQ=-60"),
        rec("c", 23, "A", ".", "DP=20;MQ=50;FQ=-1"),
        rec("c", 23, "AC", "A", &format!("INDEL;{GOOD}")),
        good_run("c", 30, 40),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    let s = &r.summary;
    assert_eq!(s.good_bases + s.lost.total(), s.bases);
    assert_eq!(s.bases, 40);
    let by_reason: Vec<_> = LossReason::ALL
        .iter()
        .map(|&r| (r.label(), s.lost.get(r)))
        .filter(|&(_, n)| n > 0)
        .collect();
    // 21..=23 lose to their own filters first; 18..=20 and 24..=29 are in
    // the window (24..=29 also have no record, which comes first).
    assert_eq!(
        by_reason,
        [
            ("no record", 6),
            ("low MQ", 1),
            ("low depth", 1),
            ("low quality", 1),
            ("indel window", 3),
        ]
    );
}

#[test]
fn auto_depth_uses_the_mean_of_base_records() {
    assert_eq!(auto_depth_thresholds(30.0), (10, 60));
    assert_eq!(auto_depth_thresholds(1.2), (1, 2));
    assert_eq!(auto_depth_thresholds(4.5), (2, 9));

    // DP 10, 20, 30, 60 on base records; the INDEL record and the record
    // without DP do not count.
    let records = [
        rec("c", 1, "A", ".", "DP=10;MQ=50;FQ=-60"),
        rec("c", 2, "A", ".", "DP=20;MQ=50;FQ=-60"),
        rec("c", 3, "A", ".", "DP=30;MQ=50;FQ=-60"),
        rec("c", 4, "A", ".", "DP=60;MQ=50;FQ=-60"),
        rec("c", 4, "AC", "A", "INDEL;DP=1000;MQ=50;FQ=-60"),
        rec("c", 5, "A", ".", "MQ=50;FQ=-60"),
    ]
    .concat();
    let path = unique_temp_path("consensus_depth", "vcf");
    fs::write(&path, format!("{HEADER}{records}")).unwrap();
    let stats = scan_depth(&path).unwrap();
    assert_eq!((stats.records, stats.records_without_dp), (4, 1));
    assert_eq!((stats.mean, stats.median), (30.0, 25.0));
    assert_eq!(stats.histogram.iter().sum::<u64>(), 4);
    assert_eq!((stats.histogram[10], stats.histogram[60]), (1, 1));

    // Mean 30: d = 10, D = 60.
    let o = ConsensusOptions {
        filters: ConsensusFilters {
            depth: DepthThresholds::Auto,
            ..ConsensusFilters::default()
        },
        ..opts()
    };
    let r = read_vcf_consensus(&path, &o).unwrap();
    let s = &r.summary;
    assert!(s.auto_depth);
    assert_eq!((s.min_depth, s.max_depth), (10, 60));
    assert_eq!(s.depth.mean, 30.0);
    // POS 4 is masked by the INDEL window; POS 5 has no DP.
    assert_eq!(bases(&r, 0), "NNNNN");
    let _ = fs::remove_file(&path);

    let err = read(&rec("c", 1, "A", ".", "MQ=50;FQ=-60"), &o).unwrap_err();
    assert!(
        format!("{err:#}").contains("no SNP/reference record"),
        "{err:#}"
    );
}

#[test]
fn depth_histogram_caps_at_1000() {
    let records = [
        rec("c", 1, "A", ".", "DP=999;MQ=50;FQ=-60"),
        rec("c", 2, "A", ".", "DP=1000;MQ=50;FQ=-60"),
        rec("c", 3, "A", ".", "DP=123456;MQ=50;FQ=-60"),
    ]
    .concat();
    let r = read(&records, &opts()).unwrap();
    let h = &r.summary.depth.histogram;
    assert_eq!(h.len(), 1001);
    assert_eq!((h[999], h[1000]), (1, 2));
    assert_eq!(r.summary.depth.median, 1000.0);
}

#[test]
fn het_sites_count_called_bases_since_the_previous_one() {
    let het = |chrom, pos| rec(chrom, pos, "A", "G", "DP=20;MQ=50;AF1=0.5;FQ=40");
    let records = [
        good_run("c", 1, 3),
        het("c", 4),
        // Missing bases are not counted.
        rec("c", 5, "A", ".", "DP=1;MQ=50;FQ=-60"),
        good_run("c", 8, 9),
        het("c", 10),
        good_run("d", 1, 1),
        het("d", 2),
    ]
    .concat();
    let r = read(
        &records,
        &ConsensusOptions {
            collect_het_sites: true,
            ..opts()
        },
    )
    .unwrap();
    let a: Vec<_> = r.seqs[0].het_sites().collect();
    assert_eq!(a, vec![(4, 4, *b"AG"), (10, 3, *b"AG")]);
    let d: Vec<_> = r.seqs[1].het_sites().collect();
    assert_eq!(d, vec![(2, 2, *b"AG")]);
    assert_eq!(r.seqs[0].het_bases, 2);
    assert_eq!(r.seqs[0].good_bases, 7);
}

#[test]
fn masks_are_an_optional_extra_filter() {
    let mask = unique_temp_path("consensus_mask", "bed");
    // BED rows: positions 3..=5 of c only; d has no row.
    fs::write(&mask, "c\t2\t5\n").unwrap();
    let records = [good_run("c", 1, 6), good_run("d", 1, 2)].concat();
    let o = ConsensusOptions {
        masks: vec![mask.clone()],
        ..opts()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(bases(&r, 0), "NNTTTN");
    assert_eq!(bases(&r, 1), "NN");
    assert!(r.summary.masked);
    assert_eq!(r.summary.lost.outside_mask, 5);
    assert_eq!(r.summary.chroms_outside_mask, ["d"]);
    let _ = fs::remove_file(mask);
}

#[test]
fn bins_follow_the_fq2psmcfa_rule() {
    // 10-base bins: 9 missing of 10 is not more than 90%, so the bin is
    // called; the partial last bin is judged against the full size.
    let records = [
        good_run("c", 10, 20),
        rec("c", 21, "A", "C", "DP=20;MQ=50;AF1=0.5;FQ=40"),
        good_run("c", 22, 22),
    ]
    .concat();
    let o = ConsensusOptions {
        bin_size: 10,
        ..ConsensusOptions::default()
    };
    let r = read(&records, &o).unwrap();
    assert_eq!(r.seqs[0].bins, vec![0, 0, 1]);
    let r = read(&good_run("c", 10, 10), &o).unwrap();
    assert_eq!(r.seqs[0].bins, vec![0]);
    // Ten missing bases make a missing bin.
    let r = read(&good_run("c", 11, 11), &o).unwrap();
    assert_eq!(r.seqs[0].bins, vec![2, 0]);
    // A last bin of 9 bases, all missing, is called: 9 of a full 10 is not
    // more than 90%.
    let records = [
        good_run("c", 1, 10),
        rec("c", 19, "A", ".", "DP=1;MQ=50;FQ=-60"),
    ]
    .concat();
    let r = read(&records, &o).unwrap();
    assert_eq!(r.seqs[0].bins, vec![0, 0]);
    assert_eq!(r.seqs[0].good_bases, 10);
}
