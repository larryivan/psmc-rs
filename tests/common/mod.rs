//! Helpers for the tests that run the `psmc-rs` binary.

#![allow(dead_code)]

use psmc_rs::io::psmcfa::read_psmcfa_records;
use psmc_rs::io::write::write_psmcfa_to;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

/// The `psmc-rs` binary of this build.
pub fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_psmc-rs"))
}

/// A directory of its own for one test, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("cli_{name}_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Run the binary with `args`.
pub fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    Command::new(bin()).args(args).output().unwrap()
}

/// Run the binary and fail the test unless it succeeds; returns stdout.
pub fn run_ok<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let out = run(args);
    assert!(
        out.status.success(),
        "psmc-rs failed ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Run the binary expecting it to fail; returns stderr.
pub fn run_err<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let out = run(args);
    assert!(
        !out.status.success(),
        "psmc-rs succeeded:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8(out.stderr).unwrap()
}

pub fn json(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// A file of `tests/data`.
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// The three reference sequences of `tests/data/c_reference` cut to their
/// first `bins` bins, written as psmcfa to `path`: real data small enough
/// for a few EM iterations in a test.
pub fn small_psmcfa(path: &Path, bins: usize) {
    let records = read_psmcfa_records(&fixture("c_reference/papuan_chr1_3seg.psmcfa")).unwrap();
    let mut out = Vec::new();
    write_psmcfa_to(
        &mut out,
        records
            .iter()
            .map(|r| (r.name.as_str(), &r.seq[..bins.min(r.seq.len())])),
    )
    .unwrap();
    fs::write(path, out).unwrap();
}
