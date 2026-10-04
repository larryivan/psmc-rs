use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::vcf::{VcfOptions, read_vcf, read_vcf_with};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

/// A path no other test of this process uses: tests run in parallel, and
/// the clock alone is too coarse to tell them apart.
fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let n = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}_{}_{n}.{ext}", std::process::id()))
}

#[test]
fn read_vcf_single_sample_matches_expected_bins() {
    let vcf = unique_temp_path("vcf_single", "vcf");
    let content = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/1
chr1\t24\t.\tA\tC\t.\tPASS\t.\tGT\t1/0
";
    fs::write(&vcf, content).expect("failed to write VCF");

    let obs = read_vcf(&vcf, None, 4, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows, vec![vec![0, 0, 0, 0, 1, 1]]);

    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_multi_sample_requires_explicit_sample() {
    let vcf = unique_temp_path("vcf_multi", "vcf");
    let content = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
chr1\t20\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1
";
    fs::write(&vcf, content).expect("failed to write VCF");

    let err = read_vcf(&vcf, None, 4, None, &[], &[]).expect_err("expected sample selection error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("please specify --vcf-sample"),
        "unexpected error message: {msg}"
    );

    let obs = read_vcf(&vcf, None, 4, Some("s2"), &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows, vec![vec![0, 0, 0, 0, 1]]);

    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_masks_follow_multihetsep_called_semantics() {
    let vcf = unique_temp_path("vcf_masked", "vcf");
    let mask = unique_temp_path("vcf_mask", "bed");
    let neg = unique_temp_path("vcf_neg", "bed");
    let mhs = unique_temp_path("vcf_equiv", "mhs");

    let vcf_content = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
chr1\t1\t.\tA\tC\t.\tPASS\t.\tGT\t0/1
chr1\t100\t.\tG\tT\t.\tPASS\t.\tGT\t1/0
";
    fs::write(&vcf, vcf_content).expect("failed to write VCF");
    // Callable baseline interval: [1,101] in 1-based coordinates.
    fs::write(&mask, "chr1\t0\t101\n").expect("failed to write mask");
    // Remove [2,99], leaving position 1 and 100 callable.
    fs::write(&neg, "chr1\t1\t99\n").expect("failed to write negative mask");

    let mhs_content = "\
chr1 1 1 AC
chr1 100 1 GT
";
    fs::write(&mhs, mhs_content).expect("failed to write mhs");

    let from_vcf = read_vcf(
        &vcf,
        None,
        100,
        None,
        std::slice::from_ref(&mask),
        std::slice::from_ref(&neg),
    )
    .expect("failed to parse masked VCF");
    let from_mhs = read_mhs(&mhs, None, 100).expect("failed to parse mhs");

    assert_eq!(from_vcf.row_starts, from_mhs.row_starts);
    assert_eq!(from_vcf.rows, from_mhs.rows);
    assert_eq!(from_vcf.rows, vec![vec![2]]);

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(mask);
    let _ = fs::remove_file(neg);
    let _ = fs::remove_file(mhs);
}

#[test]
fn read_vcf_does_not_treat_duplicate_alt_indices_as_heterozygous() {
    let vcf = unique_temp_path("vcf_dup_alt", "vcf");
    let content = "\
##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
chr1\t10\t.\tA\tC,C\t.\tPASS\t.\tGT\t1/2
chr1\t20\t.\tA\tG\t.\tPASS\t.\tGT\t0/1
";
    fs::write(&vcf, content).expect("failed to write VCF");

    let obs = read_vcf(&vcf, None, 10, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.row_starts, vec![true]);
    // POS 10 has duplicated ALT and should be homozygous in final allele space.
    // Only POS 20 contributes a K in bin#2.
    assert_eq!(obs.rows, vec![vec![0, 1]]);

    let _ = fs::remove_file(vcf);
}

const HEADER: &str =
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";

/// Write a single-sample VCF whose records are given as
/// `chrom pos ref alt filter gt`, whitespace separated.
fn write_vcf(prefix: &str, records: &[&str]) -> PathBuf {
    write_vcf_with_meta(prefix, &[], records)
}

/// [`write_vcf`] with extra `##` meta lines, such as `##contig`.
fn write_vcf_with_meta(prefix: &str, meta: &[&str], records: &[&str]) -> PathBuf {
    let mut text = String::from("##fileformat=VCFv4.2\n");
    for m in meta {
        text.push_str(m);
        text.push('\n');
    }
    text.push_str(HEADER.trim_start_matches("##fileformat=VCFv4.2\n"));
    for r in records {
        let c: Vec<&str> = r.split_whitespace().collect();
        assert_eq!(c.len(), 6, "bad test record {r:?}");
        text.push_str(&format!(
            "{}\t{}\t.\t{}\t{}\t50\t{}\t.\tGT\t{}\n",
            c[0], c[1], c[2], c[3], c[4], c[5]
        ));
    }
    write_text(prefix, "vcf", &text)
}

fn write_text(prefix: &str, ext: &str, text: &str) -> PathBuf {
    let path = unique_temp_path(prefix, ext);
    fs::write(&path, text).expect("failed to write file");
    path
}

#[test]
fn read_vcf_places_missing_sites_at_their_own_positions() {
    // 30 bases in bins of 10, het SNP at 30, and the even positions 2..=20
    // excluded: five missing bases in each of the first two bins, so both are
    // T. Packing the ten missing bases at the start of the interval (the old
    // multihetsep-style conversion) turned the first bin into N.
    let vcf = write_vcf("vcf_placement", &["chr1 30 A C PASS 0/1"]);
    let neg_rows: String = (1..=10)
        .map(|i| format!("chr1\t{}\t{}\n", 2 * i - 1, 2 * i))
        .collect();
    let neg = write_text("vcf_placement_neg", "bed", &neg_rows);

    let obs = read_vcf(&vcf, None, 10, None, &[], std::slice::from_ref(&neg))
        .expect("failed to parse VCF");
    assert_eq!(obs.rows, vec![vec![0, 0, 1]]);

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(neg);
}

#[test]
fn read_vcf_combines_records_at_the_same_position() {
    // Split multiallelic sites (bcftools norm -m-) put several records at one
    // POS. They combine with precedence missing > het SNP > homozygous.
    let vcf = write_vcf(
        "vcf_same_pos",
        &[
            "chr1 2 A C PASS 0/1",
            "chr1 2 A AT PASS 0/0",
            "chr1 4 G T PASS 0/1",
            "chr1 4 G T PASS ./.",
            "chr1 6 C CA PASS 0/0",
            "chr1 6 C G PASS 1|0",
        ],
    );
    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.rows, vec![vec![0, 1, 0, 2, 0, 1]]);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_missing_genotype_is_a_missing_site() {
    // A missing or partial genotype says nothing about the sample at that
    // position. It makes only POS missing, even for a longer REF: 7 and 8
    // are positions without a record, so called homozygous.
    let vcf = write_vcf(
        "vcf_missing_gt",
        &[
            "chr1 2 A C PASS ./1",
            "chr1 3 A C PASS ./.",
            "chr1 4 A C PASS .",
            "chr1 5 A C PASS 0/1",
            "chr1 6 AGT A PASS .|.",
            "chr1 9 A C PASS 1/0",
        ],
    );
    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.rows, vec![vec![0, 2, 2, 2, 1, 2, 0, 0, 1]]);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_het_non_snp_is_missing_not_heterozygous() {
    // PSMC models point mutations: a heterozygous indel, spanning deletion or
    // symbolic allele is not a het SNP, and its REF bases are unreliable.
    let vcf = write_vcf(
        "vcf_het_indel",
        &[
            "chr1 2 A AT PASS 0/1",
            "chr1 4 ACG A PASS 0|1",
            "chr1 8 A C,* PASS 1/2",
            "chr1 9 A <DEL> PASS 0/1",
            "chr1 10 a c PASS 0/1",
            "chr1 11 A AT PASS 1/1",
        ],
    );
    let read = read_vcf_with(
        &vcf,
        &VcfOptions {
            bin_size: 1,
            ..VcfOptions::default()
        },
    )
    .expect("failed to parse VCF");
    assert_eq!(read.obs.rows, vec![vec![0, 2, 0, 2, 2, 2, 0, 2, 2, 1, 0]]);
    assert_eq!(read.summary.het_non_snp_records, 4);
    assert_eq!(read.summary.het_snp_records, 1);
    assert_eq!(read.summary.hom_calls, 1);

    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.rows, read.obs.rows);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_filtered_records_are_missing_unless_kept() {
    let vcf = write_vcf(
        "vcf_filtered",
        &[
            "chr1 3 A C LowQual 0/1",
            "chr1 5 A C PASS 0/1",
            "chr1 6 ACG A q10;s50 0/0",
            "chr1 9 A C . 0/1",
        ],
    );
    // A filtered record makes only its POS missing, whatever its REF length.
    let obs = read_vcf(&vcf, None, 1, None, &[], &[]).expect("failed to parse VCF");
    assert_eq!(obs.rows, vec![vec![0, 0, 2, 0, 1, 2, 0, 0, 1]]);

    let kept = read_vcf_with(
        &vcf,
        &VcfOptions {
            bin_size: 1,
            keep_filtered: true,
            ..VcfOptions::default()
        },
    )
    .expect("failed to parse VCF");
    assert_eq!(kept.obs.rows, vec![vec![0, 0, 1, 0, 1, 0, 0, 0, 1]]);
    assert_eq!(kept.summary.filtered_records, 2);
    assert_eq!(kept.summary.het_snp_records, 3);
    assert_eq!(kept.summary.hom_calls, 1);

    let dropped = read_vcf_with(
        &vcf,
        &VcfOptions {
            bin_size: 1,
            ..VcfOptions::default()
        },
    )
    .expect("failed to parse VCF");
    assert_eq!(dropped.summary.filtered_records, 2);
    assert_eq!(dropped.summary.het_snp_records, 2);
    assert_eq!(dropped.summary.hom_calls, 0);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_unusable_long_record_does_not_hide_calls_inside_its_ref() {
    // All-sites VCF: every position has a PASS record. A filtered deletion at
    // 5 and a no-call deletion at 12 cover PASS calls (a het SNP at 9 and at
    // 14). Those records carry no call, so only their own POS is missing and
    // the PASS calls stand. A heterozygous deletion at 16 is a real call of
    // an indel: its whole REF span is missing, including the het SNP at 18.
    let mut records: Vec<String> = (1..=20).map(|p| format!("chr1 {p} A . PASS 0/0")).collect();
    records[4] = "chr1 5 ACGTACGTAC A LowQual 0/0".to_string();
    records[8] = "chr1 9 A C PASS 0/1".to_string();
    records[11] = "chr1 12 ACGTA A PASS ./.".to_string();
    records[13] = "chr1 14 A G PASS 1|0".to_string();
    records[15] = "chr1 16 ACGT A PASS 0/1".to_string();
    records[17] = "chr1 18 A T PASS 0/1".to_string();
    records[19] = "chr1 20 A C PASS 0/1".to_string();
    let records: Vec<&str> = records.iter().map(String::as_str).collect();
    let vcf = write_vcf("vcf_all_sites_span", &records);

    let read = read_vcf_with(
        &vcf,
        &VcfOptions {
            bin_size: 1,
            ..VcfOptions::default()
        },
    )
    .expect("failed to parse VCF");
    let bins: String = read.obs.rows[0]
        .iter()
        .map(|&b| b"TKN"[b as usize] as char)
        .collect();
    assert_eq!(bins, "TTTTNTTTKTTNTKTNNNNK");
    assert_eq!(read.summary.het_snps, 3);
    assert_eq!(read.summary.missing_sites, 6);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_extends_sequences_to_the_contig_length() {
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
    let vcf = write_vcf_with_meta("vcf_contig", &meta, &records);
    // Callable: chr1 1-12, all of chr2 and chr3.
    let mask = write_text(
        "vcf_contig_bed",
        "bed",
        "chr1\t0\t12\nchr2\t0\t100\nchr3\t0\t100\n",
    );

    // chr1 runs to 40: bins 1-10 K, 11-20 T (8 missing), 21-30 and 31-40 N.
    // chr3 has no length and ends at its last record; chrM has no records.
    let obs = read_vcf(&vcf, None, 10, None, std::slice::from_ref(&mask), &[])
        .expect("failed to parse VCF");
    assert_eq!(obs.rows, vec![vec![1, 0, 2, 2], vec![1], vec![1]]);

    let opts = VcfOptions {
        bin_size: 10,
        masks: vec![mask.clone()],
        ..VcfOptions::default()
    };
    let read = read_vcf_with(&vcf, &opts).expect("failed to parse VCF");
    assert_eq!(read.summary.contig_lengths_used, vec!["chr1", "chr2"]);
    assert_eq!(read.summary.sites, 40 + 7 + 4);
    assert_eq!(read.summary.missing_sites, 28);

    // Ending at the last record instead.
    let opts = VcfOptions {
        to_contig_end: false,
        ..opts
    };
    let read = read_vcf_with(&vcf, &opts).expect("failed to parse VCF");
    assert_eq!(read.obs.rows, vec![vec![1], vec![1], vec![1]]);
    assert!(read.summary.contig_lengths_used.is_empty());
    assert_eq!(read.summary.sites, 3 + 2 + 4);

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(mask);
}

#[test]
fn read_vcf_rejects_a_record_beyond_the_contig_length() {
    let vcf = write_vcf_with_meta(
        "vcf_beyond_contig",
        &["##contig=<ID=chr1,length=10>"],
        &["chr1 5 A C PASS 0/1", "chr1 11 A C PASS 0/1"],
    );
    let err = read_vcf(&vcf, None, 5, None, &[], &[]).expect_err("POS 11 > length 10");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("beyond the ##contig length 10"),
        "unexpected error: {msg}"
    );

    let opts = VcfOptions {
        bin_size: 5,
        to_contig_end: false,
        ..VcfOptions::default()
    };
    let read = read_vcf_with(&vcf, &opts).expect("lengths are ignored when not extending");
    assert_eq!(read.obs.rows, vec![vec![1, 0, 1]]);
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_rejects_unsorted_positions() {
    let vcf = write_vcf(
        "vcf_unsorted",
        &["chr1 20 A C PASS 0/1", "chr1 10 A C PASS 0/1"],
    );
    let err = read_vcf(&vcf, None, 10, None, &[], &[]).expect_err("unsorted POS must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("must be sorted"), "unexpected error: {msg}");
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_rejects_a_chromosome_that_reappears() {
    let vcf = write_vcf(
        "vcf_reappear",
        &[
            "chr1 20 A C PASS 0/1",
            "chr2 10 A C PASS 0/1",
            "chr1 30 A C PASS 0/1",
        ],
    );
    let err = read_vcf(&vcf, None, 10, None, &[], &[]).expect_err("chr1 twice must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("appears again") && msg.contains("chr1"),
        "unexpected error: {msg}"
    );
    let _ = fs::remove_file(vcf);
}

#[test]
fn read_vcf_without_positive_mask_counts_gaps_as_called_and_warns() {
    // Like generate_multihetsep without --mask: everything between records is
    // a called homozygous site. That is kept, but reported, because a
    // variants-only VCF then has no missing data at all.
    let vcf = write_vcf(
        "vcf_no_mask",
        &["chr1 100 A C PASS 0/1", "chr1 2000100 A C PASS 0/1"],
    );
    let read = read_vcf_with(&vcf, &VcfOptions::default()).expect("failed to parse VCF");
    let row = &read.obs.rows[0];
    assert_eq!(row.len(), 20_001);
    assert_eq!(row.iter().filter(|&&b| b == 0).count(), 19_999);
    assert_eq!((row[0], row[20_000]), (1, 1));
    assert!(!read.summary.positive_mask);
    assert_eq!(read.summary.missing_sites, 0);
    let warnings = read.summary.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("--vcf-mask"), "{warnings:?}");

    let mask = write_text("vcf_no_mask_bed", "bed", "chr1\t0\t2000100\n");
    let opts = VcfOptions {
        masks: vec![mask.clone()],
        ..VcfOptions::default()
    };
    let read = read_vcf_with(&vcf, &opts).expect("failed to parse VCF");
    assert!(read.summary.positive_mask);
    assert!(read.summary.warnings().is_empty());

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(mask);
}

#[test]
fn read_vcf_reports_chromosomes_the_mask_does_not_mention() {
    // A mask written with Ensembl names for a UCSC-named VCF.
    let mask = write_text("vcf_name_mismatch_bed", "bed", "1\t0\t1000\n");
    let opts = VcfOptions {
        bin_size: 10,
        masks: vec![mask.clone()],
        ..VcfOptions::default()
    };
    let vcf = write_vcf(
        "vcf_name_mismatch",
        &["chr1 10 A C PASS 0/1", "chr1 20 A C PASS 0/1"],
    );
    let err = read_vcf_with(&vcf, &opts).expect_err("no callable het SNP must fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("no heterozygous SNP") && msg.contains("(chr1)"),
        "unexpected error: {msg}"
    );

    let mixed = write_vcf(
        "vcf_name_mismatch2",
        &["1 20 A C PASS 0/1", "chrX 10 A C PASS 0/1"],
    );
    let read = read_vcf_with(&mixed, &opts).expect("failed to parse VCF");
    assert_eq!(read.summary.records_outside_mask, 1);
    assert_eq!(read.summary.chroms_outside_mask, vec!["chrX".to_string()]);
    let warnings = read.summary.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("chrX"), "{warnings:?}");
    assert_eq!(read.obs.rows, vec![vec![0, 1], vec![2]]);

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(mixed);
    let _ = fs::remove_file(mask);
}

#[test]
fn read_vcf_summary_counts_records_by_kind() {
    let vcf = write_vcf(
        "vcf_summary",
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
    let neg = write_text("vcf_summary_neg", "bed", "chr1\t6\t7\n");
    let opts = VcfOptions {
        bin_size: 2,
        negative_masks: vec![neg.clone()],
        ..VcfOptions::default()
    };
    let read = read_vcf_with(&vcf, &opts).expect("failed to parse VCF");
    let s = &read.summary;
    assert_eq!(s.records, 7);
    assert_eq!(s.het_snp_records, 3);
    // POS 7 is excluded by the negative mask.
    assert_eq!(s.het_snps, 2);
    assert_eq!(s.records_outside_mask, 1);
    assert_eq!(s.hom_calls, 3);
    assert_eq!(s.missing_genotypes, 1);
    assert_eq!(s.multi_record_positions, 1);
    assert_eq!(s.sequences, 2);
    assert_eq!(s.bins, 4 + 1);
    assert_eq!(s.sites, 7 + 2);
    assert_eq!(s.missing_sites, 2);
    assert!(!s.positive_mask);
    // chr1 is K T T N T T N, chr2 is T K.
    assert_eq!(read.obs.rows, vec![vec![1, 0, 0, 0], vec![1]]);

    let _ = fs::remove_file(vcf);
    let _ = fs::remove_file(neg);
}
