use psmc_rs::io::mhs::{MhsOptions, read_mhs, read_mhs_with};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    // Tests run in parallel and the clock may be coarse, so add a counter.
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before unix epoch")
        .as_nanos();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!(
        "{prefix}_{}_{}_{n}.{}",
        std::process::id(),
        nanos,
        ext
    ));
    path
}

#[test]
fn read_mhs_multihetsep_semantics_with_missing_and_k() {
    let path = unique_temp_path("mhs_semantics", "multihetsep");
    let content = "\
chr1 20 1 AC
chr1 24 4 AC
";
    fs::write(&path, content).expect("failed to write test multihetsep");

    let obs = read_mhs(&path, None, 4).expect("failed to parse multihetsep");
    assert_eq!(obs.row_starts, vec![true]);
    // row1: 19 N + K -> [N,N,N,N,K], row2: 3 T + K -> [K]
    assert_eq!(obs.rows, vec![vec![2, 2, 2, 2, 1, 1]]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_splits_chromosomes_and_batches() {
    let path = unique_temp_path("mhs_chrom_batch", "mhs");
    let content = "\
chr1 8 8 AC
chr2 8 8 AC
";
    fs::write(&path, content).expect("failed to write test mhs");

    let obs = read_mhs(&path, Some(1), 4).expect("failed to parse batched mhs");
    assert_eq!(obs.row_starts, vec![true, false, true, false]);
    assert_eq!(obs.rows, vec![vec![0], vec![1], vec![0], vec![1]]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_rejects_a_chromosome_that_comes_back() {
    // Interleaved rows: the second chr1 block would be counted from
    // position 0, misplacing its uncalled sites.
    let opts = pair_opts(10, (0, 1));
    let err = read_with(
        "chr1\t100\t50\tAC\nchr2\t90\t10\tAC\nchr1\t300\t100\tAG\n",
        &opts,
    )
    .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("row 3") && msg.contains("'chr1' appears again") && msg.contains("sort"),
        "{msg}"
    );
    // The same rows grouped by chromosome are fine.
    let rows = read_with(
        "chr1\t100\t50\tAC\nchr1\t300\t100\tAG\nchr2\t90\t10\tAC\n",
        &opts,
    )
    .unwrap();
    assert_eq!(rows.len(), 2);
}

#[test]
fn read_mhs_non_segregating_event_errors() {
    let path = unique_temp_path("mhs_nonseg", "mhs");
    let content = "chr1 4 4 AA\n";
    fs::write(&path, content).expect("failed to write test mhs");

    let err = read_mhs(&path, None, 4).expect_err("expected non-segregating mhs error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("segregating site"),
        "unexpected error message: {msg}"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_missing_alleles_errors() {
    let path = unique_temp_path("mhs_missing_alleles", "mhs");
    let content = "chr1 4 4\n";
    fs::write(&path, content).expect("failed to write test mhs");

    let err = read_mhs(&path, None, 4).expect_err("expected missing-alleles mhs error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("missing alleles"),
        "unexpected error message: {msg}"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_invalid_called_greater_than_distance_errors() {
    let path = unique_temp_path("mhs_invalid_called", "mhs");
    let content = "\
chr1 10 5 AC
chr1 12 5 AC
";
    fs::write(&path, content).expect("failed to write test mhs");

    let err = read_mhs(&path, None, 4).expect_err("expected invalid mhs error");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("cannot exceed distance"),
        "unexpected error message: {msg}"
    );

    let _ = fs::remove_file(path);
}

/// Write `content` to a temporary multihetsep file and read it with `opts`.
fn read_with(content: &str, opts: &MhsOptions) -> anyhow::Result<Vec<Vec<u8>>> {
    let path = unique_temp_path("mhs_pair", "mhs");
    fs::write(&path, content).expect("failed to write test mhs");
    let obs = read_mhs_with(&path, opts);
    let _ = fs::remove_file(path);
    obs.map(|o| o.rows)
}

fn pair_opts(bin_size: usize, pair: (usize, usize)) -> MhsOptions {
    MhsOptions {
        bin_size,
        pair,
        ..MhsOptions::default()
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
    let path = unique_temp_path("mhs_two_samples", "mhs");
    fs::write(&path, TWO_SAMPLES).expect("failed to write test mhs");
    let obs = read_mhs(&path, None, 10).expect("failed to parse mhs");
    assert_eq!(obs.rows, vec![vec![0, 0, 0, 1, 0, 0]]);
    let _ = fs::remove_file(path);

    let default = read_with(TWO_SAMPLES, &pair_opts(10, (0, 1))).expect("parse failed");
    assert_eq!(default, vec![vec![0, 0, 0, 1, 0, 0]]);
}

#[test]
fn pair_option_selects_the_haplotypes() {
    let rows = |pair| read_with(TWO_SAMPLES, &pair_opts(10, pair)).expect("parse failed");
    // Sample 1 is heterozygous only at 55.
    assert_eq!(rows((2, 3)), vec![vec![0, 0, 0, 0, 0, 1]]);
    // Cross-sample pairs; the order of the two indices does not matter.
    assert_eq!(rows((0, 2)), vec![vec![0, 1, 0, 0, 0, 1]]);
    assert_eq!(rows((2, 0)), rows((0, 2)));
    assert_eq!(rows((1, 3)), vec![vec![0, 1, 0, 1, 0, 0]]);
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
    let rows = |c, pair| read_with(c, &pair_opts(1, pair)).expect("parse failed");
    assert_eq!(rows(content, (0, 1)), vec![vec![1, 0, 1]]);
    assert_eq!(rows(content, (2, 3)), vec![vec![1, 1, 0]]);
    assert_eq!(rows(content, (0, 2)), vec![vec![2, 2, 2]]);
    assert_eq!(rows(content, (1, 3)), vec![vec![2, 2, 2]]);
    // A single unphased sample: both orderings are heterozygous.
    assert_eq!(rows("chr1 1 1 AC,CA\n", (0, 1)), vec![vec![1]]);

    // Through the default pair: sample 1's heterozygote used to be counted.
    let path = unique_temp_path("mhs_unphased", "mhs");
    fs::write(&path, content).expect("failed to write test mhs");
    let obs = read_mhs(&path, None, 1).expect("failed to parse mhs");
    assert_eq!(obs.rows, vec![vec![1, 0, 1]]);
    let _ = fs::remove_file(path);
}

#[test]
fn unknown_alleles_are_missing_for_the_pair() {
    let content = "\
chr1 1 1 A?CC
chr1 2 1 AC??
chr1 3 1 ACNA
";
    let rows = |pair| read_with(content, &pair_opts(1, pair)).expect("parse failed");
    assert_eq!(rows((0, 1)), vec![vec![2, 1, 1]]);
    assert_eq!(rows((2, 3)), vec![vec![0, 2, 2]]);
    assert_eq!(rows((0, 3)), vec![vec![1, 2, 0]]);

    // '?' used to count as a different allele, i.e. as heterozygous.
    let path = unique_temp_path("mhs_unknown", "mhs");
    fs::write(&path, "chr1 1 1 A?\n").expect("failed to write test mhs");
    let obs = read_mhs(&path, None, 1).expect("failed to parse mhs");
    assert_eq!(obs.rows, vec![vec![2]]);
    let _ = fs::remove_file(path);
}

#[test]
fn malformed_allele_orderings_error() {
    let err = |content: &str, pair| {
        let e = read_with(content, &pair_opts(4, pair)).expect_err("expected an error");
        format!("{e:#}")
    };
    let msg = err("chr1 4 4 AC\n", (0, 2));
    assert!(
        msg.contains("row 1") && msg.contains("needs at least 3"),
        "{msg}"
    );
    let msg = err("chr1 4 4 ACAC,AC\n", (0, 1));
    assert!(msg.contains("differ in length"), "{msg}");
    let msg = err("chr1 4 4 AC\n", (1, 1));
    assert!(msg.contains("two different haplotypes"), "{msg}");
    // Rows must still be segregating across all haplotypes.
    let msg = err("chr1 4 4 AAAA,AAAA\n", (0, 1));
    assert!(msg.contains("segregating site"), "{msg}");
    let msg = err("chr1 4 4 ,\n", (0, 1));
    assert!(msg.contains("no allele ordering"), "{msg}");
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
        let e = read_with(TWO_SAMPLES, &pair_opts(10, pair)).expect_err("expected an error");
        let msg = format!("{e:#}");
        assert!(
            msg.contains("row 1")
                && msg.contains("list 4 haplotypes")
                && msg.contains("needs at least 18446744073709551616"),
            "{msg}"
        );
    }
}

#[test]
fn rows_must_list_as_many_haplotypes_as_the_first_row() {
    let err = |content: &str| {
        let e = read_with(content, &pair_opts(10, (0, 1))).expect_err("expected an error");
        format!("{e:#}")
    };
    // generate_multihetsep.py joins VCF alleles as strings, so an unfiltered
    // indel (REF `AT`) widens its row. Sample 0 is homozygous in every row,
    // but character 1 of `ATATATA` is a `T` and used to be read as a
    // heterozygote: [T, K, T].
    let msg = err("chr1 10 10 AAAC\nchr1 20 10 ATATATA\nchr1 30 10 AAAC\n");
    assert!(
        msg.contains("row 2")
            && msg.contains("7 haplotypes")
            && msg.contains("row 1 lists 4")
            && msg.contains("indel"),
        "{msg}"
    );
    // Narrower rows, and rows of a later chromosome, are held to the same count.
    let msg = err("chr1 10 10 AACC\nchr1 20 10 AC\n");
    assert!(
        msg.contains("row 2") && msg.contains("row 1 lists 4"),
        "{msg}"
    );
    let msg = err("chr1 10 10 AC\nchr2 10 10 AACC\n");
    assert!(
        msg.contains("row 2") && msg.contains("row 1 lists 2"),
        "{msg}"
    );
    // Comment and blank lines are not the first row.
    let msg = err("# x\n\nchr1 10 10 AC\nchr1 20 10 AACC\n");
    assert!(
        msg.contains("row 4") && msg.contains("row 3 lists 2"),
        "{msg}"
    );
    // Orderings of one row that disagree are still reported as such.
    let msg = err("chr1 10 10 AACC\nchr1 20 10 AACA,AC\n");
    assert!(
        msg.contains("row 2") && msg.contains("differ in length"),
        "{msg}"
    );
}
