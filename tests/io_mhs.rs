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

    let xs = read_mhs(&path, None, 1).expect("failed to parse mhs");
    assert_eq!(xs.shape(), &[1, 5]);
    let got = xs.row(0).to_vec();
    assert_eq!(got, vec![0, 0, 1, 0, 1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_unbatched_with_binning() {
    let path = unique_temp_path("psmc_mhs_bin", "mhs");
    let content = "chr1\t10\t3\tAT\nchr1\t20\t2\tAC\n";
    fs::write(&path, content).expect("failed to write mhs");

    let xs = read_mhs(&path, None, 2).expect("failed to parse binned mhs");
    assert_eq!(xs.shape(), &[1, 3]);
    let got = xs.row(0).to_vec();
    assert_eq!(got, vec![0, 1, 1]);

    let _ = fs::remove_file(path);
}

#[test]
fn read_mhs_batched_pads_with_n() {
    let path = unique_temp_path("psmc_mhs_batch", "mhs");
    let content = "chr1\t10\t3\tAT\nchr1\t20\t2\tAC\n";
    fs::write(&path, content).expect("failed to write mhs");

    let xs = read_mhs(&path, Some(4), 1).expect("failed to parse batched mhs");
    assert_eq!(xs.shape(), &[2, 4]);
    assert_eq!(xs[[0, 0]], 0);
    assert_eq!(xs[[0, 1]], 0);
    assert_eq!(xs[[0, 2]], 1);
    assert_eq!(xs[[0, 3]], 0);
    assert_eq!(xs[[1, 0]], 1);
    assert_eq!(xs[[1, 1]], 2);
    assert_eq!(xs[[1, 2]], 2);
    assert_eq!(xs[[1, 3]], 2);

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

    let xs = read_mhs(&path, None, 1).expect("failed to parse gz mhs");
    assert_eq!(xs.shape(), &[1, 2]);
    let got = xs.row(0).to_vec();
    assert_eq!(got, vec![1, 1]);

    let _ = fs::remove_file(path);
}
