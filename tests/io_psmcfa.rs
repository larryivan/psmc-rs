use flate2::Compression;
use flate2::write::GzEncoder;
use psmc_rs::io::psmcfa::read_psmcfa;
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
fn read_unbatched_psmcfa_parses_sequences() {
    let path = unique_temp_path("psmc_unbatched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let xs = read_psmcfa(&path, None).expect("failed to parse psmcfa");
    assert_eq!(xs.shape(), &[2, 4]);
    assert_eq!(xs[[0, 0]], 0); // T
    assert_eq!(xs[[0, 2]], 1); // K
    assert_eq!(xs[[1, 1]], 2); // N

    let _ = fs::remove_file(path);
}

#[test]
fn read_batched_psmcfa_pads_with_n() {
    let path = unique_temp_path("psmc_batched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let xs = read_psmcfa(&path, Some(3)).expect("failed to parse batched psmcfa");
    assert_eq!(xs.shape(), &[3, 3]);
    assert_eq!(xs[[2, 2]], 2); // padding N

    let _ = fs::remove_file(path);
}

#[test]
fn read_gz_psmcfa_works() {
    let path = unique_temp_path("psmc_gz", "psmcfa.gz");
    let file = fs::File::create(&path).expect("failed to create gz path");
    let mut writer = GzEncoder::new(file, Compression::default());
    writer
        .write_all(b"> chr1\nTKNT\n")
        .expect("failed to write gz data");
    writer.finish().expect("failed to finish gzip stream");

    let xs = read_psmcfa(&path, None).expect("failed to parse gz psmcfa");
    assert_eq!(xs.shape(), &[1, 4]);
    assert_eq!(xs[[0, 0]], 0);
    assert_eq!(xs[[0, 1]], 1);
    assert_eq!(xs[[0, 2]], 2);

    let _ = fs::remove_file(path);
}
