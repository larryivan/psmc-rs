use psmc_rs::io::mhs::read_mhs;
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
