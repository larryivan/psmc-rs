//! The mask-mode VCF reader: one sample's genotypes, callable masks,
//! `##contig` lengths, gVCF blocks, and the summary of what it read.

mod common;

use common::TempDir;
use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::vcf::{VcfOptions, VcfRead, read_vcf, read_vcf_with};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const COLUMNS: &str = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1";

/// `text` written to `name` in `dir`.
fn write(dir: &TempDir, name: &str, text: &str) -> PathBuf {
    let path = dir.path(name);
    fs::write(&path, text).unwrap();
    path
}

/// A one-sample VCF `name` in `dir`: the `##` lines `meta`, then `records`
/// given as `chrom pos ref alt filter gt`, or as `chrom pos ref alt filter
/// info format sample`, separated by spaces.
fn write_vcf(dir: &TempDir, name: &str, meta: &[&str], records: &[&str]) -> PathBuf {
    let mut text = String::from("##fileformat=VCFv4.2\n");
    for m in meta {
        writeln!(text, "{m}").unwrap();
    }
    writeln!(text, "{COLUMNS}").unwrap();
    for r in records {
        let c: Vec<&str> = r.split_whitespace().collect();
        let (info, format, sample) = match c[..] {
            [.., gt] if c.len() == 6 => (".", "GT", gt),
            [.., info, format, sample] if c.len() == 8 => (info, format, sample),
            _ => panic!("bad test record {r:?}"),
        };
        let [chrom, pos, reference, alt, filter] = [c[0], c[1], c[2], c[3], c[4]];
        writeln!(
            text,
            "{chrom}\t{pos}\t.\t{reference}\t{alt}\t50\t{filter}\t{info}\t{format}\t{sample}"
        )
        .unwrap();
    }
    write(dir, name, &text)
}

fn opts(bin_size: usize) -> VcfOptions {
    VcfOptions {
        bin_size,
        ..VcfOptions::default()
    }
}

fn read(vcf: &Path, opts: &VcfOptions) -> VcfRead {
    read_vcf_with(vcf, opts).unwrap_or_else(|e| panic!("{e:#}"))
}

/// The message of the error reading `vcf` with `opts`.
fn error(vcf: &Path, opts: &VcfOptions) -> String {
    format!("{:#}", read_vcf_with(vcf, opts).unwrap_err())
}

#[test]
fn heterozygous_genotypes_are_k_bins() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &["chr1 20 A C PASS 0/1", "chr1 24 A C PASS 1/0"],
    );
    let obs = read_vcf(&vcf, None, 4, None, &[], &[]).unwrap();
    assert_eq!(obs.row_starts, [true]);
    assert_eq!(obs.rows, [[0, 0, 0, 0, 1, 1]]);
}

#[test]
fn a_vcf_of_several_samples_needs_the_sample_named() {
    let dir = TempDir::new("io_vcf");
    let vcf = write(
        &dir,
        "two.vcf",
        &format!(
            "##fileformat=VCFv4.2\n{COLUMNS}\ts2\nchr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        ),
    );
    let err = read_vcf(&vcf, None, 4, None, &[], &[]).unwrap_err();
    assert!(
        format!("{err:#}").contains("please specify --vcf-sample"),
        "{err:#}"
    );
    let obs = read_vcf(&vcf, None, 4, Some("s2"), &[], &[]).unwrap();
    assert_eq!(obs.row_starts, [true]);
    assert_eq!(obs.rows, [[0, 0, 0, 0, 1]]);
}

#[test]
fn masks_follow_multihetsep_called_semantics() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &["chr1 1 A C PASS 0/1", "chr1 100 G T PASS 1/0"],
    );
    // Callable: [1, 101] (1-based) less [2, 99], which leaves positions 1
    // and 100 of the sequence.
    let mask = write(&dir, "mask.bed", "chr1\t0\t101\n");
    let neg = write(&dir, "neg.bed", "chr1\t1\t99\n");
    let mhs = write(&dir, "calls.mhs", "chr1 1 1 AC\nchr1 100 1 GT\n");
    let from_vcf = read_vcf(&vcf, None, 100, None, &[mask], &[neg]).unwrap();
    let from_mhs = read_mhs(&mhs, None, 100).unwrap();
    assert_eq!(from_vcf.row_starts, from_mhs.row_starts);
    assert_eq!(from_vcf.rows, from_mhs.rows);
    assert_eq!(from_vcf.rows, [[2]]);
}

#[test]
fn duplicate_alt_alleles_are_not_heterozygous() {
    // ALT C,C with GT 1/2 is homozygous C; only POS 20 makes a K bin.
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &["chr1 10 A C,C PASS 1/2", "chr1 20 A G PASS 0/1"],
    );
    let obs = read_vcf(&vcf, None, 10, None, &[], &[]).unwrap();
    assert_eq!(obs.row_starts, [true]);
    assert_eq!(obs.rows, [[0, 1]]);
}

#[test]
fn missing_sites_stay_at_their_own_positions() {
    // 30 bases in bins of 10, het SNP at 30, and the even positions 2..=20
    // excluded: five missing bases in each of the first two bins, so both are
    // T. Packing the ten missing bases at the start of the interval (the old
    // multihetsep-style conversion) turned the first bin into N.
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(&dir, "calls.vcf", &[], &["chr1 30 A C PASS 0/1"]);
    let neg_rows: String = (1..=10)
        .map(|i| format!("chr1\t{}\t{}\n", 2 * i - 1, 2 * i))
        .collect();
    let neg = write(&dir, "neg.bed", &neg_rows);
    let obs = read_vcf(&vcf, None, 10, None, &[], &[neg]).unwrap();
    assert_eq!(obs.rows, [[0, 0, 1]]);
}

/// The rows of one-base bins of a VCF of `records`.
fn one_base_bins(records: &[&str]) -> Vec<Vec<u8>> {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(&dir, "calls.vcf", &[], records);
    read_vcf(&vcf, None, 1, None, &[], &[]).unwrap().rows
}

#[test]
fn records_at_one_position_combine() {
    // Split multiallelic sites (bcftools norm -m-) put several records at one
    // POS. They combine with precedence missing > het SNP > homozygous.
    let rows = one_base_bins(&[
        "chr1 2 A C PASS 0/1",
        "chr1 2 A AT PASS 0/0",
        "chr1 4 G T PASS 0/1",
        "chr1 4 G T PASS ./.",
        "chr1 6 C CA PASS 0/0",
        "chr1 6 C G PASS 1|0",
    ]);
    assert_eq!(rows, [[0, 1, 0, 2, 0, 1]]);
}

#[test]
fn a_missing_genotype_is_a_missing_site() {
    // A missing or partial genotype says nothing about the sample at that
    // position. It makes only POS missing, even for a longer REF: 7 and 8
    // are positions without a record, so called homozygous.
    let rows = one_base_bins(&[
        "chr1 2 A C PASS ./1",
        "chr1 3 A C PASS ./.",
        "chr1 4 A C PASS .",
        "chr1 5 A C PASS 0/1",
        "chr1 6 AGT A PASS .|.",
        "chr1 9 A C PASS 1/0",
    ]);
    assert_eq!(rows, [[0, 2, 2, 2, 1, 2, 0, 0, 1]]);
}

#[test]
fn a_heterozygous_non_snp_is_missing_not_heterozygous() {
    // PSMC models point mutations: a heterozygous indel, spanning deletion or
    // symbolic allele is not a het SNP, and its REF bases are unreliable.
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &[
            "chr1 2 A AT PASS 0/1",
            "chr1 4 ACG A PASS 0|1",
            "chr1 8 A C,* PASS 1/2",
            "chr1 9 A <DEL> PASS 0/1",
            "chr1 10 a c PASS 0/1",
            "chr1 11 A AT PASS 1/1",
        ],
    );
    let read = read(&vcf, &opts(1));
    assert_eq!(read.obs.rows, [[0, 2, 0, 2, 2, 2, 0, 2, 2, 1, 0]]);
    let s = &read.summary;
    assert_eq!(
        (s.het_non_snp_records, s.het_snp_records, s.hom_calls),
        (4, 1, 1)
    );
    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).unwrap();
    assert_eq!(obs.rows, read.obs.rows);
}

#[test]
fn filtered_records_are_missing_unless_kept() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &[
            "chr1 3 A C LowQual 0/1",
            "chr1 5 A C PASS 0/1",
            "chr1 6 ACG A q10;s50 0/0",
            "chr1 9 A C . 0/1",
        ],
    );
    // A filtered record makes only its POS missing, whatever its REF length.
    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).unwrap();
    assert_eq!(obs.rows, [[0, 0, 2, 0, 1, 2, 0, 0, 1]]);
    let kept = read(
        &vcf,
        &VcfOptions {
            keep_filtered: true,
            ..opts(1)
        },
    );
    assert_eq!(kept.obs.rows, [[0, 0, 1, 0, 1, 0, 0, 0, 1]]);
    let s = &kept.summary;
    assert_eq!(
        (s.filtered_records, s.het_snp_records, s.hom_calls),
        (2, 3, 1)
    );
    let s = read(&vcf, &opts(1)).summary;
    assert_eq!(
        (s.filtered_records, s.het_snp_records, s.hom_calls),
        (2, 2, 0)
    );
}

#[test]
fn an_unusable_long_record_does_not_hide_calls_inside_its_ref() {
    // All-sites VCF: every position has a PASS record. A filtered deletion at
    // 5 and a no-call deletion at 12 cover PASS calls (a het SNP at 9 and at
    // 14). Those records carry no call, so only their own POS is missing and
    // the PASS calls stand. A heterozygous deletion at 16 is a real call of
    // an indel: its whole REF span is missing, including the het SNP at 18.
    let mut records: Vec<String> = (1..=20).map(|p| format!("chr1 {p} A . PASS 0/0")).collect();
    for (pos, record) in [
        (5, "ACGTACGTAC A LowQual 0/0"),
        (9, "A C PASS 0/1"),
        (12, "ACGTA A PASS ./."),
        (14, "A G PASS 1|0"),
        (16, "ACGT A PASS 0/1"),
        (18, "A T PASS 0/1"),
        (20, "A C PASS 0/1"),
    ] {
        records[pos - 1] = format!("chr1 {pos} {record}");
    }
    let records: Vec<&str> = records.iter().map(String::as_str).collect();
    let dir = TempDir::new("io_vcf");
    let read = read(&write_vcf(&dir, "calls.vcf", &[], &records), &opts(1));
    let bins: String = read.obs.rows[0]
        .iter()
        .map(|&b| b"TKN"[b as usize] as char)
        .collect();
    assert_eq!(bins, "TTTTNTTTKTTNTKTNNNNK");
    assert_eq!((read.summary.het_snps, read.summary.missing_sites), (3, 6));
}

#[test]
fn sequences_extend_to_the_contig_length() {
    let dir = TempDir::new("io_vcf");
    let meta = [
        "##contig=<ID=chr1,length=40>",
        "##contig=<ID=chr2,assembly=\"test, v1\",length=7>",
        "##contig=<ID=chr3>",
        "##contig=<ID=chrM,length=16569>",
    ];
    let records = [
        "chr1 3 A C PASS 0/1",
        "chr2 2 A C PASS 0/1",
        "chr3 4 A C PASS 0/1",
    ];
    let vcf = write_vcf(&dir, "calls.vcf", &meta, &records);
    // Callable: chr1 1-12, all of chr2 and chr3.
    let mask = write(
        &dir,
        "mask.bed",
        "chr1\t0\t12\nchr2\t0\t100\nchr3\t0\t100\n",
    );

    // chr1 runs to 40: bins 1-10 K, 11-20 T (8 missing), 21-30 and 31-40 N.
    // chr3 has no length and ends at its last record; chrM has no records.
    let obs = read_vcf(&vcf, None, 10, None, std::slice::from_ref(&mask), &[]).unwrap();
    assert_eq!(obs.rows, vec![vec![1, 0, 2, 2], vec![1], vec![1]]);
    let opts = VcfOptions {
        masks: vec![mask],
        ..opts(10)
    };
    let s = read(&vcf, &opts).summary;
    assert_eq!(s.contig_lengths_used, ["chr1", "chr2"]);
    assert_eq!((s.sites, s.missing_sites), (40 + 7 + 4, 28));

    // Ending at the last record instead.
    let read = read(
        &vcf,
        &VcfOptions {
            to_contig_end: false,
            ..opts
        },
    );
    assert_eq!(read.obs.rows, [[1], [1], [1]]);
    assert!(read.summary.contig_lengths_used.is_empty());
    assert_eq!(read.summary.sites, 3 + 2 + 4);
}

#[test]
fn a_record_beyond_the_contig_length_is_an_error() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &["##contig=<ID=chr1,length=10>"],
        &["chr1 5 A C PASS 0/1", "chr1 11 A C PASS 0/1"],
    );
    let msg = error(&vcf, &opts(5));
    assert!(msg.contains("beyond the ##contig length 10"), "{msg}");
    // Lengths are ignored when not extending to them.
    let short = VcfOptions {
        to_contig_end: false,
        ..opts(5)
    };
    assert_eq!(read(&vcf, &short).obs.rows, [[1, 0, 1]]);
}

#[test]
fn implausible_contig_lengths_and_positions_are_errors() {
    // 2^36 (68.7 Gb) is the limit. The huge bin size keeps the accepted cases
    // (and the code before the limit, which let these lengths through) quick.
    let dir = TempDir::new("io_vcf");
    let opts = opts(1 << 30);
    let record = ["chr1 100 A C PASS 0/1"];
    let with_length = |len: &str| {
        let meta = format!("##contig=<ID=chr1,length={len}>");
        write_vcf(&dir, "calls.vcf", &[meta.as_str()], &record)
    };

    let mut row = vec![0; 64];
    row[0] = 1;
    assert_eq!(read(&with_length("68719476736"), &opts).obs.rows, [row]);
    for (len, shown) in [
        ("68719476737", "(68719476737)"),
        (
            "1000000000000000000000000",
            "(a number of 20 or more digits)",
        ),
    ] {
        let vcf = with_length(len);
        let msg = error(&vcf, &opts);
        assert!(
            msg.contains("##contig length of 'chr1'")
                && msg.contains(shown)
                && msg.contains("supported maximum"),
            "{len}: {msg}"
        );
        // Not extending to the contig end ignores the length.
        let short = VcfOptions {
            to_contig_end: false,
            ..opts.clone()
        };
        assert_eq!(read(&vcf, &short).obs.rows, [[1]]);
    }

    // The case that used to run until memory ran out: 2^60 at the default
    // bin size. A bad length on a chromosome without records is never used.
    let meta = [
        "##contig=<ID=chr1,length=1152921504606846976>",
        "##contig=<ID=chr2,length=1152921504606846976>",
        "##contig=<ID=chr3,length=50>",
    ];
    let vcf3 = write_vcf(&dir, "chr3.vcf", &meta, &["chr3 10 A C PASS 0/1"]);
    let obs = read_vcf(&vcf3, None, 10, None, &[], &[]).unwrap();
    assert_eq!(obs.rows, [[1, 0, 0, 0, 0]]);
    let huge = write_vcf(&dir, "huge.vcf", &meta, &record);
    let err = read_vcf(&huge, None, 100, None, &[], &[]).unwrap_err();
    assert!(format!("{err:#}").contains("supported maximum"), "{err:#}");

    let far = write_vcf(&dir, "far.vcf", &[], &["chr1 68719476737 A C PASS 0/1"]);
    let msg = error(&far, &opts);
    assert!(
        msg.contains("POS 68719476737 is beyond the supported maximum"),
        "{msg}"
    );
}

#[test]
fn records_out_of_order_are_errors() {
    let dir = TempDir::new("io_vcf");
    for (records, parts) in [
        (
            &["chr1 20 A C PASS 0/1", "chr1 10 A C PASS 0/1"][..],
            &["must be sorted"][..],
        ),
        (
            &[
                "chr1 20 A C PASS 0/1",
                "chr2 10 A C PASS 0/1",
                "chr1 30 A C PASS 0/1",
            ],
            &["appears again", "chr1"],
        ),
    ] {
        let msg = error(&write_vcf(&dir, "calls.vcf", &[], records), &opts(10));
        assert!(parts.iter().all(|p| msg.contains(p)), "{msg}");
    }
}

#[test]
fn without_a_positive_mask_gaps_count_as_called_with_a_warning() {
    // Like generate_multihetsep without --mask: everything between records is
    // a called homozygous site. That is kept, but reported, because a
    // variants-only VCF then has no missing data at all.
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &["chr1 100 A C PASS 0/1", "chr1 2000100 A C PASS 0/1"],
    );
    let read_with = |masks| {
        read(
            &vcf,
            &VcfOptions {
                masks,
                ..VcfOptions::default()
            },
        )
    };
    let read = read_with(vec![]);
    let row = &read.obs.rows[0];
    assert_eq!(row.len(), 20_001);
    assert_eq!(row.iter().filter(|&&b| b == 0).count(), 19_999);
    assert_eq!((row[0], row[20_000]), (1, 1));
    assert!(!read.summary.positive_mask);
    assert_eq!(read.summary.missing_sites, 0);
    let warnings = read.summary.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("--vcf-mask"), "{warnings:?}");

    let read = read_with(vec![write(&dir, "mask.bed", "chr1\t0\t2000100\n")]);
    assert!(read.summary.positive_mask);
    assert!(read.summary.warnings().is_empty());
}

#[test]
fn chromosomes_the_mask_does_not_mention_are_reported() {
    // A mask written with Ensembl names for a UCSC-named VCF.
    let dir = TempDir::new("io_vcf");
    let opts = VcfOptions {
        masks: vec![write(&dir, "mask.bed", "1\t0\t1000\n")],
        ..opts(10)
    };
    let ucsc = write_vcf(
        &dir,
        "ucsc.vcf",
        &[],
        &["chr1 10 A C PASS 0/1", "chr1 20 A C PASS 0/1"],
    );
    let msg = error(&ucsc, &opts);
    assert!(
        msg.contains("no heterozygous SNP") && msg.contains("(chr1)"),
        "{msg}"
    );

    let mixed = write_vcf(
        &dir,
        "mixed.vcf",
        &[],
        &["1 20 A C PASS 0/1", "chrX 10 A C PASS 0/1"],
    );
    let read = read(&mixed, &opts);
    assert_eq!(read.summary.records_outside_mask, 1);
    assert_eq!(read.summary.chroms_outside_mask, ["chrX"]);
    let warnings = read.summary.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("chrX"), "{warnings:?}");
    assert_eq!(read.obs.rows, vec![vec![0, 1], vec![2]]);
}

#[test]
fn gvcf_blocks_without_a_call_are_missing_over_their_span() {
    let dir = TempDir::new("io_vcf");
    let opts = VcfOptions {
        masks: vec![write(&dir, "mask.bed", "chr1\t0\t100\n")],
        ..opts(10)
    };
    let missing_40 = [0, 0, 2, 2, 2, 2, 1, 0, 0, 0];
    // A DeepVariant no-call block over 21..=60, and the same block
    // filtered (missing unless filtered records are kept) or written with
    // GATK's <NON_REF>.
    for (no_call, keep_filtered, bins, missing) in [
        (
            "chr1 21 A <*> . END=60 GT:GQ:MIN_DP ./.:0:0",
            false,
            missing_40,
            40,
        ),
        (
            "chr1 21 A <NON_REF> . END=60 GT:GQ:MIN_DP ./.:0:0",
            false,
            missing_40,
            40,
        ),
        (
            "chr1 21 A <*> LowGQ END=60 GT:GQ:MIN_DP 0/0:2:1",
            false,
            missing_40,
            40,
        ),
        (
            "chr1 21 A <*> LowGQ END=60 GT:GQ:MIN_DP 0/0:2:1",
            true,
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            0,
        ),
    ] {
        let records = [
            "chr1 1 A <*> . END=20 GT:GQ:MIN_DP 0/0:50:30",
            no_call,
            "chr1 61 A G,<*> PASS . GT:GQ:MIN_DP 0/1:40:30",
            "chr1 62 A <*> . END=100 GT:GQ:MIN_DP 0/0:50:30",
        ];
        let meta = ["##contig=<ID=chr1,length=100>"];
        let opts = VcfOptions {
            keep_filtered,
            ..opts.clone()
        };
        let read = read(&write_vcf(&dir, "calls.vcf", &meta, &records), &opts);
        assert_eq!(read.obs.rows, [bins], "{no_call}");
        let s = &read.summary;
        assert_eq!(s.missing_sites, missing, "{no_call}");
        assert_eq!(s.gvcf_blocks, 3);
        assert_eq!(s.gvcf_unusable_blocks, u64::from(missing > 0));
        let warnings = s.warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("gVCF") && warnings[0].contains("bcftools view -v snps,indels"),
            "{warnings:?}"
        );
    }
}

#[test]
fn gvcf_block_ends_are_clamped_and_end_sequences() {
    let dir = TempDir::new("io_vcf");
    let records = ["chr1 1 A G PASS . GT 0/1", "chr1 2 A <*> . END=35 GT 0/0"];
    // Without a ##contig length the sequence ends at the last block's END;
    // an END beyond the ##contig length stops there.
    for (meta, bins, sites) in [
        (&[][..], &[1, 0, 0, 0][..], 35),
        (&["##contig=<ID=chr1,length=30>"], &[1, 0, 0], 30),
    ] {
        let read = read(&write_vcf(&dir, "calls.vcf", meta, &records), &opts(10));
        assert_eq!(read.obs.rows, [bins.to_vec()]);
        assert_eq!(read.summary.sites, sites);
    }

    // Not blocks: an END before POS, and END on a structural variant or
    // another non-reference allele, which describes the variant rather
    // than a run of positions with one call.
    let other = write_vcf(
        &dir,
        "other.vcf",
        &["##contig=<ID=chr1,length=40>"],
        &[
            "chr1 1 A G PASS . GT 0/1",
            "chr1 10 A <*> . END=5 GT ./.",
            "chr1 20 A <DEL> PASS SVTYPE=DEL;END=35 GT ./.",
            "chr1 25 A C,<*> PASS END=35 GT ./.",
        ],
    );
    let s = read(&other, &opts(10)).summary;
    assert_eq!((s.missing_sites, s.gvcf_blocks), (3, 0));
    assert!(s.warnings().iter().all(|w| !w.contains("gVCF")));
}

/// Read as a gVCF, a position that no record covers, by its POS, its REF
/// span or its block, is missing: before the first record, between
/// records, and after the last one up to the ##contig length. Otherwise
/// such positions are called, as in any VCF, and the gVCF warning says so.
#[test]
fn reading_as_a_gvcf_makes_uncovered_positions_missing() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &["##contig=<ID=chr1,length=60>"],
        &[
            // 1..=4 uncovered.
            "chr1 5 A <*> . END=20 GT 0/0",
            // A homozygous deletion covers 21..=24 with its REF.
            "chr1 21 ACGT A PASS . GT 1/1",
            // 25..=29 uncovered.
            "chr1 30 A G PASS . GT 0/1",
            // A bcftools reference block (ALT '.'); 41..=60 uncovered.
            "chr1 31 A . . END=40 GT 0/0",
        ],
    );
    let missing = |s: &psmc_rs::io::vcf::VcfSummary| (s.missing_sites, s.uncovered_sites);
    let read_as = |gvcf, masks| {
        let opts = VcfOptions {
            gvcf,
            masks,
            ..opts(10)
        };
        read(&vcf, &opts).summary
    };
    let s = read_as(false, vec![]);
    assert_eq!(missing(&s), (0, 0));
    let warnings = s.warnings();
    assert!(
        warnings.iter().any(|w| w.contains("is a gVCF")
            && w.contains("positions that no record or block covers count as called")),
        "{warnings:?}"
    );

    let s = read_as(true, vec![]);
    assert!(s.read_as_gvcf);
    assert_eq!(missing(&s), (29, 29));
    assert_eq!(s.het_snps, 1);
    let warnings = s.warnings();
    assert!(
        warnings.iter().any(|w| w.contains("is a gVCF")
            && w.contains("so are the positions that no record or block covers (29 here")),
        "{warnings:?}"
    );
    // Uncovered positions are missing also where a mask would call them,
    // and a position the mask leaves out is not counted twice.
    let s = read_as(true, vec![write(&dir, "mask.bed", "chr1\t0\t50\n")]);
    assert_eq!(missing(&s), (29, 29));
}

#[test]
fn separate_per_chromosome_masks_are_said_to_be_intersected() {
    // One mask per chromosome, each given as its own --vcf-mask: every
    // chromosome is in one mask and missing from the others.
    let dir = TempDir::new("io_vcf");
    let mask = |c: &str| write(&dir, &format!("{c}.bed"), &format!("{c}\t0\t1000\n"));
    let (chr1, chr2) = (mask("chr1"), mask("chr2"));
    let vcf2 = write_vcf(
        &dir,
        "two.vcf",
        &[],
        &["chr1 10 A C PASS 0/1", "chr2 20 A C PASS 0/1"],
    );
    let with_masks = |masks: &[&PathBuf]| VcfOptions {
        masks: masks.iter().map(|&m| m.clone()).collect(),
        ..opts(10)
    };
    let err = error(&vcf2, &with_masks(&[&chr1, &chr2]));
    assert!(
        err.contains("(chr1, chr2) have intervals in some of the callable masks")
            && err.contains("intersected")
            && err.contains("concatenate"),
        "{err}"
    );
    assert!(!err.contains("'chr1' vs '1'"), "{err}");

    // With a third mask that covers both, and a chromosome no mask has.
    let both = write(&dir, "both.bed", "chr1\t0\t1000\nchr2\t0\t1000\n");
    let vcf3 = write_vcf(
        &dir,
        "three.vcf",
        &[],
        &[
            "chr1 10 A C PASS 0/1",
            "chr2 20 A C PASS 0/1",
            "chrX 5 A C PASS 0/1",
        ],
    );
    let s = read(&vcf3, &with_masks(&[&chr1, &both])).summary;
    assert_eq!(s.chroms_outside_some_masks, ["chr2"]);
    assert_eq!(s.chroms_outside_mask, ["chrX"]);
    let warnings = s.warnings();
    assert_eq!(warnings.len(), 2, "{warnings:?}");
    assert!(warnings[0].contains("(chrX)") && warnings[0].contains("'chr1' vs '1'"));
    assert!(warnings[1].contains("(chr2)") && warnings[1].contains("intersected"));
}

#[test]
fn the_no_het_snp_error_points_at_filtered_records() {
    // A pipeline that puts a non-PASS label on every record: the het SNPs are
    // there, but all of them are filtered. The error must say so rather than
    // claim the VCF has no heterozygous SNP records.
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "filtered.vcf",
        &[],
        &[
            "chr1 5 A C LowQual 0/1",
            "chr1 7 A G PASS 0/0",
            "chr1 9 A C LowQual 0/1",
        ],
    );
    let msg = error(&vcf, &opts(1));
    assert!(
        msg.contains("no heterozygous SNP")
            && msg.contains("2 records have a FILTER other than PASS")
            && msg.contains("--vcf-keep-filtered"),
        "{msg}"
    );
    let kept = VcfOptions {
        keep_filtered: true,
        ..opts(1)
    };
    assert_eq!(read(&vcf, &kept).summary.het_snps, 2);

    // Without filtered records the hint stays out of the message.
    let unfiltered = write_vcf(&dir, "unfiltered.vcf", &[], &["chr1 5 A C PASS 0/0"]);
    let err = read_vcf(&unfiltered, None, 1, None, &[], &[]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no heterozygous SNP") && !msg.contains("FILTER"),
        "{msg}"
    );
}

#[test]
fn the_summary_counts_records_by_kind() {
    let dir = TempDir::new("io_vcf");
    let vcf = write_vcf(
        &dir,
        "calls.vcf",
        &[],
        &[
            "chr1 1 A C PASS 0/1",
            "chr1 1 A AT PASS 0/0",
            "chr1 3 A C PASS 1/1",
            "chr1 4 A C PASS ./.",
            "chr1 5 A C PASS 1",
            "chr1 7 A C PASS 0|1",
            "chr2 2 A G PASS 0/1",
        ],
    );
    let opts = VcfOptions {
        negative_masks: vec![write(&dir, "neg.bed", "chr1\t6\t7\n")],
        ..opts(2)
    };
    let read = read(&vcf, &opts);
    let s = &read.summary;
    assert_eq!((s.records, s.het_snp_records), (7, 3));
    // POS 7 is excluded by the negative mask.
    assert_eq!((s.het_snps, s.records_outside_mask), (2, 1));
    assert_eq!((s.hom_calls, s.missing_genotypes), (3, 1));
    assert_eq!(s.multi_record_positions, 1);
    assert_eq!((s.sequences, s.bins), (2, 4 + 1));
    assert_eq!((s.sites, s.missing_sites), (7 + 2, 2));
    assert!(!s.positive_mask);
    // chr1 is K T T N T T N, chr2 is T K.
    assert_eq!(read.obs.rows, vec![vec![1, 0, 0, 0], vec![1]]);
}

#[test]
fn the_summary_reports_the_sample_depth() {
    // One sample: FORMAT/DP, else INFO/DP; a DP of '.' is no depth.
    let dir = TempDir::new("io_vcf");
    let one = write(
        &dir,
        "one.vcf",
        &format!(
            "##fileformat=VCFv4.2\n{COLUMNS}\n\
             c\t1\t.\tA\tC\t.\tPASS\tDP=99\tGT:DP\t0/1:10\n\
             c\t2\t.\tA\t.\t.\tPASS\t.\tGT:DP\t0/0:20\n\
             c\t3\t.\tA\t.\t.\tPASS\tDP=5\tGT:DP\t0/0:.\n\
             c\t4\t.\tA\tG\t.\tq10\tMQ=3;DP=7\tGT\t0/1\n\
             c\t5\t.\tA\t.\t.\tPASS\tDP4=1,2,3,4\tGT\t0/0\n"
        ),
    );
    let s = read(&one, &VcfOptions::default()).summary;
    let d = &s.depth;
    assert_eq!((d.records, d.records_without_dp), (3, 2));
    assert_eq!(s.depth_from_info, 1);
    assert_eq!((d.mean, d.median), (37.0 / 3.0, 10.0));
    assert_eq!(d.histogram.len(), 1001);
    let counted: Vec<usize> = (0..d.histogram.len())
        .filter(|&i| d.histogram[i] > 0)
        .collect();
    assert_eq!(counted, [7, 10, 20]);

    // Several samples: the selected sample's FORMAT/DP only, since INFO/DP
    // sums over all of them.
    let two = write(
        &dir,
        "two.vcf",
        &format!(
            "##fileformat=VCFv4.2\n{COLUMNS}\ts2\n\
             c\t1\t.\tA\tC\t.\tPASS\tDP=130\tGT:DP\t0/0:100\t0/1:30\n\
             c\t2\t.\tA\t.\t.\tPASS\tDP=50\tGT\t0/0\t0/0\n"
        ),
    );
    let opts = VcfOptions {
        sample: Some("s2".to_string()),
        ..VcfOptions::default()
    };
    let s = read(&two, &opts).summary;
    let d = &s.depth;
    assert_eq!((d.records, d.records_without_dp, d.mean), (1, 1, 30.0));
    assert_eq!(s.depth_from_info, 0);
}
