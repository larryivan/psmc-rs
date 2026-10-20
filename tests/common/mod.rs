//! Helpers shared by the integration tests: runs of the `psmc-rs` binary,
//! the files of `tests/data`, and the models, fits and E-steps that many
//! tests build alike.

#![allow(dead_code)]

pub mod psmc_text;
pub mod reference_hmm;

use psmc_rs::hmm::{EStepResult, e_step_streaming};
use psmc_rs::io::psmcfa::{read_psmcfa, read_psmcfa_records};
use psmc_rs::io::write::write_psmcfa_to;
use psmc_rs::io::{CallableBases, Observations};
use psmc_rs::model::{NegativeTransitions, n_steps_for_pattern};
use psmc_rs::opt::{EmHistory, MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

// ---- the binary ---------------------------------------------------------

/// The `psmc-rs` binary of this build.
pub fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_psmc-rs"))
}

/// A string or a path as an `OsString`, for [`args!`].
pub fn os(x: impl AsRef<OsStr>) -> OsString {
    x.as_ref().to_owned()
}

/// Command-line arguments from strings and paths alike, as a
/// `Vec<OsString>`; `..xs` splices in the items of `xs`.
#[allow(unused_macros)]
macro_rules! args {
    (@chain ($it:expr)) => {
        $it.collect::<Vec<::std::ffi::OsString>>()
    };
    (@chain ($it:expr) ..$xs:expr $(, $($rest:tt)*)?) => {
        args!(@chain ($it.chain($xs.iter().map($crate::common::os))) $($($rest)*)?)
    };
    (@chain ($it:expr) $x:expr $(, $($rest:tt)*)?) => {
        args!(@chain ($it.chain([$crate::common::os(&$x)])) $($($rest)*)?)
    };
    ($($t:tt)*) => {
        args!(@chain (::std::iter::empty()) $($t)*)
    };
}
#[allow(unused_imports)]
pub(crate) use args;

/// Run the binary with `args`.
pub fn run<I, S>(args: I) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(bin()).args(args).output().unwrap()
}

/// Run the binary and fail the test unless it succeeds; returns stdout.
pub fn run_ok<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    outcome(args).ok().stdout
}

/// Run the binary expecting it to fail; returns stderr.
pub fn run_err<I, S>(args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let o = outcome(args);
    assert!(!o.success(), "psmc-rs succeeded:\n{}", o.stdout);
    o.stderr
}

/// How a run of the binary ended, and what it printed.
#[derive(Debug)]
pub struct Outcome {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl Outcome {
    pub fn success(&self) -> bool {
        self.code == Some(0)
    }

    /// Fail the test unless the run succeeded.
    #[track_caller]
    pub fn ok(self) -> Self {
        assert!(
            self.success(),
            "psmc-rs failed ({:?}):\n{}",
            self.code,
            self.stderr
        );
        self
    }

    /// Fail the test unless the run exited with `code`; returns stderr.
    #[track_caller]
    pub fn exits(self, code: i32) -> String {
        assert_eq!(self.code, Some(code), "{}", self.stderr);
        self.stderr
    }
}

impl From<Output> for Outcome {
    fn from(o: Output) -> Self {
        Self {
            code: o.status.code(),
            stdout: String::from_utf8(o.stdout).unwrap(),
            stderr: String::from_utf8(o.stderr).unwrap(),
        }
    }
}

/// Run the binary with `args` to its end, however it exits.
pub fn outcome<I, S>(args: I) -> Outcome
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run(args).into()
}

/// Run `cmd`, a [`command`] given its own environment or directory, to its
/// end.
pub fn outcome_of(cmd: &mut Command) -> Outcome {
    cmd.output().unwrap().into()
}

/// A command of the binary.
pub fn command() -> Command {
    Command::new(bin())
}

// ---- files ----------------------------------------------------------------

/// A directory of its own for one test, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{name}_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Write `contents` to the file `name`; returns its path.
    pub fn write(&self, name: &str, contents: impl AsRef<[u8]>) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The text of a file.
#[track_caller]
pub fn read(path: impl AsRef<Path>) -> String {
    let path = path.as_ref();
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The contents of a file, gunzipped when its name ends in `.gz` (which it
/// must then be).
#[track_caller]
pub fn gunzipped(path: impl AsRef<Path>) -> Vec<u8> {
    let path = path.as_ref();
    let raw = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if !path.to_string_lossy().ends_with(".gz") {
        return raw;
    }
    assert!(
        raw.starts_with(&[0x1f, 0x8b]),
        "{}: not gzip",
        path.display()
    );
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(raw.as_slice())
        .read_to_end(&mut out)
        .unwrap();
    out
}

pub fn json(path: &Path) -> Value {
    serde_json::from_str(&read(path)).unwrap()
}

/// The codes of the findings of a QC report's JSON, in order.
pub fn finding_codes(report: &Value) -> Vec<String> {
    report["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["code"].as_str().unwrap().to_string())
        .collect()
}

// ---- fixtures ---------------------------------------------------------------

/// A file of `tests/data`.
pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// The 180,000-bin psmcfa of three Papuan sequences that C psmc's
/// reference outputs of `tests/data/c_reference` were fitted to.
pub const REFERENCE_PSMCFA: &str = "c_reference/papuan_chr1_3seg.psmcfa";

/// The psmcfa fixture `name`, one row per sequence, as C psmc reads it.
pub fn fixture_obs(name: &str) -> Observations {
    read_psmcfa(&fixture(name), None).unwrap()
}

/// The three reference sequences of [`REFERENCE_PSMCFA`] cut to their
/// first `bins` bins, written as psmcfa to `path`: real data small enough
/// for a few EM iterations in a test.
pub fn small_psmcfa(path: &Path, bins: usize) {
    let records = read_psmcfa_records(&fixture(REFERENCE_PSMCFA)).unwrap();
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

/// One row of `route_a/expected/option_sets.tsv`: a VCF of `tests/data/route_a`
/// and the psmc-rs options that read it as `vcfutils.pl vcf2fq | fq2psmcfa`
/// did with the row's options (their defaults where it gives none) to write
/// the golden psmcfa `expected`.
pub struct OptionSet {
    pub name: String,
    pub vcf: PathBuf,
    pub expected: PathBuf,
    /// `--depth`, `--min-mapq`, `--indel-window`, `--min-qual`,
    /// `--min-good-bases` and `--bin-size`, in that order; the `auto` row
    /// lets psmc-rs choose the depth bounds.
    pub args: Vec<String>,
    pub bin_size: String,
}

pub fn route_a_option_sets() -> Vec<OptionSet> {
    let flags = |text: &str, tool: &str| -> Vec<(char, String)> {
        let words: Vec<&str> = text
            .strip_prefix(tool)
            .expect("tool name")
            .split_whitespace()
            .collect();
        words
            .chunks(2)
            .map(|p| (p[0].chars().nth(1).unwrap(), p[1].to_string()))
            .collect()
    };
    let route_a = |name: &str| fixture(&format!("route_a/{name}"));
    read(route_a("expected/option_sets.tsv"))
        .lines()
        .skip(1)
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            let name = cols[0].to_string();
            let (v, f) = (flags(cols[1], "vcf2fq"), flags(cols[2], "fq2psmcfa"));
            let get = |flags: &[(char, String)], key: char, default: &str| {
                flags
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map_or(default.to_string(), |(_, v)| v.clone())
            };
            let depth = if name == "auto" {
                "auto".to_string()
            } else {
                format!("{}:{}", get(&v, 'd', "3"), get(&v, 'D', "100000"))
            };
            let bin_size = get(&f, 's', "100");
            let args = [
                ("--depth", depth),
                ("--min-mapq", get(&v, 'Q', "10")),
                ("--indel-window", get(&v, 'l', "5")),
                ("--min-qual", get(&f, 'q', "10")),
                ("--min-good-bases", get(&f, 'g', "10000")),
                ("--bin-size", bin_size.clone()),
            ]
            .into_iter()
            .flat_map(|(flag, value)| [flag.to_string(), value])
            .collect();
            let expected = ["psmcfa", "psmcfa.gz"]
                .iter()
                .map(|ext| route_a(&format!("expected/{name}.{ext}")))
                .find(|p| p.exists())
                .unwrap();
            OptionSet {
                vcf: route_a(cols[3]),
                expected,
                args,
                bin_size,
                name,
            }
        })
        .collect()
}

// ---- models, fits and E-steps -------------------------------------------------

/// The default M-step, without progress bars.
pub fn mstep() -> MStepConfig {
    MStepConfig {
        progress: false,
        ..MStepConfig::default()
    }
}

/// A model of `pattern` with t_max 15 and mu 2.5e-8, at λ = 1.
pub fn model(pattern: &str, theta: f64, rho: f64) -> PsmcModel {
    let n_steps = n_steps_for_pattern(pattern).unwrap();
    PsmcModel::new(15.0, n_steps, theta, rho, 2.5e-8, Some(pattern.into())).unwrap()
}

/// `callable` with every called bin of `bins` full (N bins keep their
/// counts): data that C psmc's model fits as the callable-length one does.
pub fn full_bins(bins: &[Vec<u8>], callable: &CallableBases) -> CallableBases {
    let s = callable.bin_size();
    let counts = bins
        .iter()
        .zip(callable.to_counts())
        .map(|(b, c)| {
            b.iter()
                .zip(c)
                .map(|(&v, c)| if v == 2 { c } else { s as u16 })
                .collect()
        })
        .collect();
    CallableBases::from_counts(s, counts).unwrap()
}

/// The limit on ρ of a start of θ `theta` with pattern 4+5*3+4 and t_max
/// 15: above it some transition probabilities are negative.
pub fn max_rho(theta: f64) -> f64 {
    let err = model("4+5*3+4", theta, 0.5)
        .check_transitions()
        .unwrap_err();
    err.downcast::<NegativeTransitions>()
        .unwrap()
        .max_rho
        .unwrap()
}

/// C psmc's start on `rows` with the intervals of `pattern`: θ0 =
/// -ln(1 - K/L), ρ0 = θ0 / `tr_ratio`, t_max 15, λ = 1.
pub fn c_start(rows: &[Vec<u8>], pattern: &str, tr_ratio: f64) -> PsmcModel {
    let theta0 = initial_theta(rows).unwrap();
    model(pattern, theta0, theta0 / tr_ratio)
}

/// [`c_start`] without a pattern: `n_steps` intervals of a λ each (C's
/// `-p "N*1"` with N = `n_steps` + 1).
pub fn c_start_free(rows: &[Vec<u8>], n_steps: usize, tr_ratio: f64) -> PsmcModel {
    let theta0 = initial_theta(rows).unwrap();
    PsmcModel::new(15.0, n_steps, theta0, theta0 / tr_ratio, 2.5e-8, None).unwrap()
}

/// A run of C psmc's EM: its input, its start and its history.
pub struct CFit {
    pub obs: Observations,
    pub template: PsmcModel,
    pub history: EmHistory,
}

/// C psmc's run on the psmcfa fixture `name` (one row per sequence):
/// `n_iter` EM iterations from [`c_start`] of `pattern` and ρ0 = θ0/4, or
/// without a pattern from 23 states of a λ each (C's `-p "23*1"`), with C's
/// objective (no smoothing).
pub fn c_fit(name: &str, pattern: Option<&str>, n_iter: usize) -> CFit {
    let obs = fixture_obs(name);
    let template = match pattern {
        Some(p) => c_start(&obs.rows, p, 4.0),
        None => c_start_free(&obs.rows, 22, 4.0),
    };
    let mut model = template.clone();
    let config = MStepConfig {
        lambda: 0.0,
        ..mstep()
    };
    let history = em_train(&mut model, &obs.rows, &obs.row_starts, n_iter, &config).unwrap();
    CFit {
        obs,
        template,
        history,
    }
}

/// The parameters EM fits, as its history records them: θ, ρ, t_max, λ...
pub fn params(m: &PsmcModel) -> Vec<f64> {
    let mut p = vec![m.theta, m.rho, m.t_max];
    p.extend(&m.lam);
    p
}

/// `template` (its layout) at the parameters `p` (θ, ρ, t_max, λ...).
pub fn model_at(template: &PsmcModel, p: &[f64]) -> PsmcModel {
    let mut m = template.clone();
    m.theta = p[0];
    m.rho = p[1];
    m.t_max = p[2];
    m.lam.copy_from_slice(&p[3..]);
    m.param_recalculate().unwrap();
    m
}

/// The bits of `xs`, to compare values to the bit (`-0.0` and `0.0`, and
/// NaNs, apart).
pub fn bits<'a>(xs: impl IntoIterator<Item = &'a f64>) -> Vec<u64> {
    xs.into_iter().map(|x| x.to_bits()).collect()
}

/// The E-step of `m` on `rows`.
pub fn e_step(m: &PsmcModel, rows: &[Vec<u8>], starts: &[bool]) -> EStepResult {
    e_step_streaming(
        m.prior_matrix(),
        m.transition_matrix(),
        m.emission_matrix(),
        rows,
        starts,
        false,
        "E",
    )
    .unwrap()
}

/// A rayon pool of `threads` threads.
pub fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}
