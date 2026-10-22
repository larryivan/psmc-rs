//! Helpers shared by the integration tests: runs of the `psmc-rs` binary,
//! temporary and gzip files, the generator of the random scenarios, the files
//! of `tests/data`, and the models, fits and E-steps that many tests build
//! alike.

#![allow(dead_code)]

pub mod psmc_text;
pub mod reference_hmm;

use ndarray::Array2;
use psmc_rs::hmm::{EStepResult, TransitionSums, e_step_streaming};
use psmc_rs::io::consensus::{ConsensusFilters, DepthThresholds};
use psmc_rs::io::input::{
    Dataset, InputFormat, InputOptions, SeqFilter, SourceSummary, VcfDetails, VcfInputSummary,
    VcfMode,
};
use psmc_rs::io::psmcfa::{read_psmcfa, read_psmcfa_records};
use psmc_rs::io::vcf::VcfSummary;
use psmc_rs::io::write::write_psmcfa_to;
use psmc_rs::io::{CallableBases, Observations};
use psmc_rs::model::{NegativeTransitions, n_steps_for_pattern};
use psmc_rs::opt::{EmHistory, MStepConfig, em_train};
use psmc_rs::{PsmcModel, initial_theta};
use rand::rngs::SmallRng;
use rand::{Rng as _, SeedableRng};
use serde_json::Value;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{Read, Write};
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

/// The words of `s`, to splice into [`args!`] a run of options that read
/// better on one line: `..words("--bootstrap 2 --bootstrap-seed 7")`.
pub fn words(s: &str) -> Vec<&str> {
    s.split_whitespace().collect()
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

/// `data` as one gzip member.
pub fn gzip(data: &[u8]) -> Vec<u8> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
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

/// The JSON of a file.
pub fn json(path: &Path) -> Value {
    serde_json::from_str(&read(path)).unwrap()
}

/// The header of a VCF of one sample, `s1`.
pub const VCF_HEADER: &str =
    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";

/// The codes of the findings of a QC report's JSON, in order.
pub fn finding_codes(report: &Value) -> Vec<String> {
    report["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["code"].as_str().unwrap().to_string())
        .collect()
}

// ---- random scenarios ---------------------------------------------------------

/// SplitMix64, the generator of the random scenarios: deterministic and
/// dependency-free, so that a failing scenario's seed reproduces it on any
/// platform and with any `rand` release.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `lo..=hi`.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }

    /// One of `xs`, uniformly.
    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.range(0, xs.len() as u64 - 1) as usize]
    }
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
/// and the options `vcfutils.pl vcf2fq | fq2psmcfa` read it with (their
/// defaults where the row gives none) to write the golden psmcfa `expected`.
pub struct OptionSet {
    pub name: String,
    pub vcf: PathBuf,
    pub expected: PathBuf,
    /// vcf2fq's `-d`, `-D`, `-Q` and `-l`.
    pub min_depth: u64,
    pub max_depth: u64,
    pub min_mq: u64,
    pub indel_window: u64,
    /// fq2psmcfa's `-q`, `-g` and `-s`.
    pub min_qual: u64,
    pub min_good_bases: u64,
    pub bin_size: usize,
}

impl OptionSet {
    /// The row's depth bounds, but psmc-rs's own for the `auto` row, whose
    /// bounds the script that made the goldens computed.
    fn depth(&self) -> DepthThresholds {
        if self.name == "auto" {
            DepthThresholds::Auto
        } else {
            DepthThresholds::Fixed {
                min: self.min_depth,
                max: self.max_depth,
            }
        }
    }

    /// The command-line options that read `vcf` as the tools did.
    pub fn args(&self) -> Vec<String> {
        let depth = match self.depth() {
            DepthThresholds::Fixed { min, max } => format!("{min}:{max}"),
            DepthThresholds::Auto => "auto".to_string(),
        };
        [
            ("--depth", depth),
            ("--min-mapq", self.min_mq.to_string()),
            ("--indel-window", self.indel_window.to_string()),
            ("--min-qual", self.min_qual.to_string()),
            ("--min-good-bases", self.min_good_bases.to_string()),
            ("--bin-size", self.bin_size.to_string()),
        ]
        .into_iter()
        .flat_map(|(flag, value)| [flag.to_string(), value])
        .collect()
    }

    /// The loader's options that read `vcf` as the tools did.
    pub fn input_options(&self) -> InputOptions {
        InputOptions {
            format: Some(InputFormat::Vcf),
            vcf_mode: VcfMode::Auto,
            bin_size: self.bin_size,
            consensus: ConsensusFilters {
                depth: self.depth(),
                min_mq: self.min_mq,
                indel_window: self.indel_window,
                min_qual: self.min_qual,
            },
            filter: SeqFilter {
                min_good_bases: self.min_good_bases,
                min_good_fraction: 0.2,
            },
            ..InputOptions::default()
        }
    }
}

/// The rows of `route_a/expected/option_sets.tsv`.
pub fn route_a_option_sets() -> Vec<OptionSet> {
    // A tool's flags by their letters: `vcf2fq -d 10 -D 60`.
    let flags = |text: &str, tool: &str| -> Vec<(char, u64)> {
        let words: Vec<&str> = text
            .strip_prefix(tool)
            .expect("tool name")
            .split_whitespace()
            .collect();
        words
            .chunks(2)
            .map(|p| (p[0].chars().nth(1).unwrap(), p[1].parse().unwrap()))
            .collect()
    };
    let get = |flags: &[(char, u64)], key: char, default: u64| {
        flags
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(default, |&(_, v)| v)
    };
    let route_a = |name: &str| fixture(&format!("route_a/{name}"));
    read(route_a("expected/option_sets.tsv"))
        .lines()
        .skip(1)
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            let name = cols[0].to_string();
            let (v, f) = (flags(cols[1], "vcf2fq"), flags(cols[2], "fq2psmcfa"));
            let expected = ["psmcfa", "psmcfa.gz"]
                .iter()
                .map(|ext| route_a(&format!("expected/{name}.{ext}")))
                .find(|p| p.exists())
                .unwrap();
            OptionSet {
                vcf: route_a(cols[3]),
                expected,
                min_depth: get(&v, 'd', 3),
                max_depth: get(&v, 'D', 100_000),
                min_mq: get(&v, 'Q', 10),
                indel_window: get(&v, 'l', 5),
                min_qual: get(&f, 'q', 10),
                min_good_bases: get(&f, 'g', 10_000),
                bin_size: get(&f, 's', 100) as usize,
                name,
            }
        })
        .collect()
}

/// The summary of `data`, which must have been read as a VCF.
pub fn vcf_summary(data: &Dataset) -> &VcfInputSummary {
    let SourceSummary::Vcf(v) = &data.source else {
        panic!("not read as a VCF: {:?}", data.source)
    };
    v
}

/// The reader's summary of `data`, which must have been read as a VCF in
/// mask mode.
pub fn mask_summary(data: &Dataset) -> &VcfSummary {
    let VcfDetails::Mask(m) = &vcf_summary(data).details else {
        panic!("not read in mask mode")
    };
    m
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

/// The sums of the expected transition counts `xi[(k, l)]` that an E-step
/// returns: over the rows and the columns of each triangle, and the
/// diagonal.
pub fn transition_sums(xi: &Array2<f64>) -> TransitionSums {
    let n = xi.nrows();
    let sum = |cells: &mut dyn Iterator<Item = (usize, usize)>| cells.map(|c| xi[c]).sum();
    TransitionSums {
        lower_rows: (0..n).map(|k| sum(&mut (0..k).map(|l| (k, l)))).collect(),
        lower_cols: (0..n)
            .map(|l| sum(&mut (l + 1..n).map(|k| (k, l))))
            .collect(),
        upper_rows: (0..n)
            .map(|k| sum(&mut (k + 1..n).map(|l| (k, l))))
            .collect(),
        upper_cols: (0..n).map(|l| sum(&mut (0..l).map(|k| (k, l)))).collect(),
        diagonal: (0..n).map(|k| xi[(k, k)]).collect(),
    }
}

/// The five sums of `t`, one after another.
pub fn transition_values(t: &TransitionSums) -> Vec<f64> {
    [
        &t.lower_rows,
        &t.lower_cols,
        &t.upper_rows,
        &t.upper_cols,
        &t.diagonal,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect()
}

/// Make `a` a transition matrix without PSMC's semiseparable structure, on
/// which the E-step runs its dense kernels and caches forward rows when its
/// budget allows: each probability raised by a random share below 1e-3 (of
/// `seed`), and the rows renormalised.
pub fn densify(a: &mut Array2<f64>, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    for mut row in a.rows_mut() {
        row.mapv_inplace(|v| v * (1.0 + 1e-3 * rng.gen_range(0.0..1.0)));
        let s: f64 = row.sum();
        row.mapv_inplace(|v| v / s);
    }
}

/// A rayon pool of `threads` threads.
pub fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}
