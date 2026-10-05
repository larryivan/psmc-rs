use flate2::Compression;
use flate2::write::GzEncoder;
use psmc_rs::io::psmcfa::read_psmcfa;
use std::fs;
use std::io::Write;
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
fn read_unbatched_psmcfa_parses_sequences() {
    let path = unique_temp_path("psmc_unbatched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let obs = read_psmcfa(&path, None).expect("failed to parse psmcfa");
    assert_eq!(obs.rows.len(), 2);
    assert_eq!(obs.row_starts, vec![true, true]);
    assert_eq!(obs.rows[0], vec![0, 0, 1, 2]); // T T K N
    assert_eq!(obs.rows[1], vec![0, 2, 0, 1]); // T N T K

    let _ = fs::remove_file(path);
}

#[test]
fn read_batched_psmcfa_keeps_sequence_chain() {
    let path = unique_temp_path("psmc_batched", "psmcfa");
    let content = "> chr1\nTTKN\n> chr2\nTNTK\n";
    fs::write(&path, content).expect("failed to write test psmcfa");

    let obs = read_psmcfa(&path, Some(3)).expect("failed to parse batched psmcfa");
    assert_eq!(obs.rows.len(), 4);
    assert_eq!(obs.row_starts, vec![true, false, true, false]);
    assert_eq!(obs.rows[0], vec![0, 0, 1]);
    assert_eq!(obs.rows[1], vec![2]);
    assert_eq!(obs.rows[2], vec![0, 2, 0]);
    assert_eq!(obs.rows[3], vec![1]);

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

    let obs = read_psmcfa(&path, None).expect("failed to parse gz psmcfa");
    assert_eq!(obs.rows.len(), 1);
    assert_eq!(obs.row_starts, vec![true]);
    assert_eq!(obs.rows[0], vec![0, 1, 2, 0]);

    let _ = fs::remove_file(path);
}
