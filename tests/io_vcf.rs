use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::vcf::read_vcf;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before unix epoch")
        .as_nanos();
    path.push(format!("{prefix}_{}_{}.{}", std::process::id(), nanos, ext));
    path
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

    let from_vcf = read_vcf(&vcf, None, 100, None, &[mask.clone()], &[neg.clone()])
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
