//! The multihetsep reader: MSMC's rows of segregating sites as the bins of
//! one pair of haplotypes.

mod common;

use common::TempDir;
use psmc_rs::io::Observations;
use psmc_rs::io::mhs::{MhsOptions, read_mhs, read_mhs_with};
use std::fs;
use std::path::Path;

/// `content` read as a multihetsep file by `read`.
fn read_text<T>(content: &str, read: impl FnOnce(&Path) -> T) -> T {
    let dir = TempDir::new("io_mhs");
    let path = dir.path("in.multihetsep");
    fs::write(&path, content).unwrap();
    read(&path)
}

/// `content` read with `read_mhs`, which reads the first sample.
fn read(content: &str, batch_size: Option<usize>, bin_size: usize) -> Observations {
    read_text(content, |p| read_mhs(p, batch_size, bin_size)).unwrap()
}

/// The rows of `content` for the haplotypes `pair`.
fn rows(content: &str, bin_size: usize, pair: (usize, usize)) -> anyhow::Result<Vec<Vec<u8>>> {
    let opts = MhsOptions {
        bin_size,
        pair,
        ..MhsOptions::default()
    };
    read_text(content, |p| read_mhs_with(p, &opts)).map(|o| o.rows)
}

/// Reading `content` for `pair` fails with an error holding all of `parts`.
fn assert_error(content: &str, bin_size: usize, pair: (usize, usize), parts: &[&str]) {
    let msg = format!("{:#}", rows(content, bin_size, pair).unwrap_err());
    assert!(
        parts.iter().all(|p| msg.contains(p)),
        "{content:?} {pair:?}: {msg}"
    );
}

#[test]
fn uncalled_sites_are_missing_and_rows_heterozygous() {
    // Row 1: 19 uncalled sites and a K, [N, N, N, N, K]; row 2: 3 called
    // sites and a K, [K].
    let obs = read("chr1 20 1 AC\nchr1 24 4 AC\n", None, 4);
    assert_eq!(obs.row_starts, [true]);
    assert_eq!(obs.rows, [vec![2, 2, 2, 2, 1, 1]]);
}

#[test]
fn chromosomes_are_sequences_cut_into_batches() {
    let obs = read("chr1 8 8 AC\nchr2 8 8 AC\n", Some(1), 4);
    assert_eq!(obs.row_starts, [true, false, true, false]);
    assert_eq!(obs.rows, [[0], [1], [0], [1]]);
}

#[test]
fn a_chromosome_that_comes_back_is_an_error() {
    // Interleaved rows: the second chr1 block would be counted from
    // position 0, misplacing its uncalled sites.
    assert_error(
        "chr1\t100\t50\tAC\nchr2\t90\t10\tAC\nchr1\t300\t100\tAG\n",
        10,
        (0, 1),
        &["row 3", "'chr1' appears again", "sort"],
    );
    // The same rows grouped by chromosome are fine.
    let grouped = rows(
        "chr1\t100\t50\tAC\nchr1\t300\t100\tAG\nchr2\t90\t10\tAC\n",
        10,
        (0, 1),
    );
    assert_eq!(grouped.unwrap().len(), 2);
}

#[test]
fn malformed_rows_are_errors() {
    for (content, pair, parts) in [
        ("chr1 4 4 AA\n", (0, 1), &["segregating site"][..]),
        ("chr1 4 4\n", (0, 1), &["missing alleles"]),
        (
            "chr1 10 5 AC\nchr1 12 5 AC\n",
            (0, 1),
            &["cannot exceed distance"],
        ),
        ("chr1 4 4 AC\n", (0, 2), &["row 1", "needs at least 3"]),
        ("chr1 4 4 ACAC,AC\n", (0, 1), &["differ in length"]),
        ("chr1 4 4 AC\n", (1, 1), &["two different haplotypes"]),
        // Rows must still be segregating across all haplotypes.
        ("chr1 4 4 AAAA,AAAA\n", (0, 1), &["segregating site"]),
        ("chr1 4 4 ,\n", (0, 1), &["no allele ordering"]),
    ] {
        assert_error(content, 4, pair, parts);
    }
}

// Two diploid samples, all sites called; only the second row is heterozygous
// in sample 0 (haplotypes 0 and 1).
const TWO_SAMPLES: &str = "\
chr1 15 15 AACC
chr1 35 20 ACAA
chr1 55 20 AACA
";

#[test]
fn multi_haplotype_rows_use_only_the_first_sample_by_default() {
    // Any difference among the four haplotypes used to count as heterozygous:
    // [T, K, T, K, T, K].
    assert_eq!(read(TWO_SAMPLES, None, 10).rows, [[0, 0, 0, 1, 0, 0]]);
    assert_eq!(rows(TWO_SAMPLES, 10, (0, 1)).unwrap(), [[0, 0, 0, 1, 0, 0]]);
}

#[test]
fn pair_option_selects_the_haplotypes() {
    let rows = |pair| rows(TWO_SAMPLES, 10, pair).unwrap();
    // Sample 1 is heterozygous only at 55.
    assert_eq!(rows((2, 3)), [[0, 0, 0, 0, 0, 1]]);
    // Cross-sample pairs; the order of the two indices does not matter.
    assert_eq!(rows((0, 2)), [[0, 1, 0, 0, 0, 1]]);
    assert_eq!(rows((2, 0)), rows((0, 2)));
    assert_eq!(rows((1, 3)), [[0, 1, 0, 1, 0, 0]]);
}

#[test]
fn phase_ambiguous_orderings_are_missing_for_the_pair() {
    // Unphased samples: generate_multihetsep lists every phasing. Within one
    // sample all orderings agree; across samples they can disagree, and the
    // site is then missing (N with one site per bin).
    // Rows: both samples heterozygous; sample 0 AA and sample 1 heterozygous;
    // sample 0 heterozygous and sample 1 AA.
    let content = "\
chr1 1 1 ACAC,ACCA,CAAC,CACA
chr1 2 1 AAAC,AACA
chr1 3 1 ACAA,CAAA
";
    let rows = |c, pair| rows(c, 1, pair).unwrap();
    assert_eq!(rows(content, (0, 1)), [[1, 0, 1]]);
    assert_eq!(rows(content, (2, 3)), [[1, 1, 0]]);
    assert_eq!(rows(content, (0, 2)), [[2, 2, 2]]);
    assert_eq!(rows(content, (1, 3)), [[2, 2, 2]]);
    // A single unphased sample: both orderings are heterozygous.
    assert_eq!(rows("chr1 1 1 AC,CA\n", (0, 1)), [[1]]);
    // Through the default pair: sample 1's heterozygote used to be counted.
    assert_eq!(read(content, None, 1).rows, [[1, 0, 1]]);
}

#[test]
fn unknown_alleles_are_missing_for_the_pair() {
    let content = "\
chr1 1 1 A?CC
chr1 2 1 AC??
chr1 3 1 ACNA
";
    let rows = |pair| rows(content, 1, pair).unwrap();
    assert_eq!(rows((0, 1)), [[2, 1, 1]]);
    assert_eq!(rows((2, 3)), [[0, 2, 2]]);
    assert_eq!(rows((0, 3)), [[1, 2, 0]]);
    // '?' used to count as a different allele, i.e. as heterozygous.
    assert_eq!(read("chr1 1 1 A?\n", None, 1).rows, [[2]]);
}

#[test]
fn mhs_options_default_is_first_sample_at_bin_100() {
    assert_eq!(
        MhsOptions::default(),
        MhsOptions {
            bin_size: 100,
            batch_size: None,
            pair: (0, 1),
        }
    );
}

#[test]
fn out_of_range_haplotype_index_is_an_error_not_a_panic() {
    // `max(pair) + 1` used to overflow: a panic in debug builds and an index
    // out of bounds in release builds.
    for pair in [
        (0, usize::MAX),
        (usize::MAX, 0),
        (usize::MAX - 1, usize::MAX),
    ] {
        let parts = [
            "row 1",
            "list 4 haplotypes",
            "needs at least 18446744073709551616",
        ];
        assert_error(TWO_SAMPLES, 10, pair, &parts);
    }
}

#[test]
fn rows_must_list_as_many_haplotypes_as_the_first_row() {
    let check = |content: &str, parts: &[&str]| assert_error(content, 10, (0, 1), parts);
    // generate_multihetsep.py joins VCF alleles as strings, so an unfiltered
    // indel (REF `AT`) widens its row. Sample 0 is homozygous in every row,
    // but character 1 of `ATATATA` is a `T` and used to be read as a
    // heterozygote: [T, K, T].
    check(
        "chr1 10 10 AAAC\nchr1 20 10 ATATATA\nchr1 30 10 AAAC\n",
        &["row 2", "7 haplotypes", "row 1 lists 4", "indel"],
    );
    // Narrower rows, and rows of a later chromosome, are held to the same count.
    check(
        "chr1 10 10 AACC\nchr1 20 10 AC\n",
        &["row 2", "row 1 lists 4"],
    );
    check(
        "chr1 10 10 AC\nchr2 10 10 AACC\n",
        &["row 2", "row 1 lists 2"],
    );
    // Comment and blank lines are not the first row.
    check(
        "# x\n\nchr1 10 10 AC\nchr1 20 10 AACC\n",
        &["row 4", "row 3 lists 2"],
    );
    // Orderings of one row that disagree are still reported as such.
    check(
        "chr1 10 10 AACC\nchr1 20 10 AACA,AC\n",
        &["row 2", "differ in length"],
    );
}
