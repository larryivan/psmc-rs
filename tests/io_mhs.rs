use flate2::Compression;
use flate2::write::GzEncoder;
use psmc_rs::io::mhs::read_mhs;
use std::fs;
use std::io::Write;
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
fn read_mhs_unbatched_bin_one() {
    let path = unique_temp_path("psmc_mhs", "mhs");
    let content = "chr1\t10\t3\tAT\nchr1\t20\t2\tAC\n";
    fs::write(&path, content).expect("failed to write mhs");

    let obs = read_mhs(&path, None, 1).expect("failed to parse mhs");
    assert_eq!(obs.rows.len(), 1);
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows[0], vec![0, 0, 1, 0, 1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_unbatched_with_binning() {
    let path = unique_temp_path("psmc_mhs_bin", "mhs");
    let content = "chr1\t10\t3\tAT\nchr1\t20\t2\tAC\n";
    fs::write(&path, content).expect("failed to write mhs");

    let obs = read_mhs(&path, None, 2).expect("failed to parse binned mhs");
    assert_eq!(obs.rows.len(), 1);
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows[0], vec![0, 1, 1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_batched_keeps_sequence_chain() {
    let path = unique_temp_path("psmc_mhs_batch", "mhs");
    let content = "chr1\t10\t3\tAT\nchr1\t20\t2\tAC\n";
    fs::write(&path, content).expect("failed to write mhs");

    let obs = read_mhs(&path, Some(4), 1).expect("failed to parse batched mhs");
    assert_eq!(obs.rows.len(), 2);
    assert_eq!(obs.row_starts, vec![true, false]);
    assert_eq!(obs.rows[0], vec![0, 0, 1, 0]);
    assert_eq!(obs.rows[1], vec![1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_gz_works() {
    let path = unique_temp_path("psmc_mhs_gz", "mhs.gz");
    let file = fs::File::create(&path).expect("failed to create gz mhs");
    let mut writer = GzEncoder::new(file, Compression::default());
    writer
        .write_all(b"chr1\t10\t1\tAT\nchr1\t20\t1\tAC\n")
        .expect("failed to write gz mhs");
    writer.finish().expect("failed to finish gzip");

    let obs = read_mhs(&path, None, 1).expect("failed to parse gz mhs");
    assert_eq!(obs.rows.len(), 1);
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows[0], vec![1, 1]);

    let _ = fs::remove_file(path);
}
