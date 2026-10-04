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
    let mut text = String::from(HEADER);
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
    // position (or, for a longer REF, at any base of the REF allele).
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
    assert_eq!(obs.rows, vec![vec![0, 2, 2, 2, 1, 2, 2, 2, 1]]);
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
