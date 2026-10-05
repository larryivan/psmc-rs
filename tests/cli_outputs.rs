//! The outputs of an inference run through the binary: determinism across
//! thread counts, TMRCA decoding and time scaling.

mod common;

use common::{TempDir, json, run_ok, small_psmcfa};
use psmc_rs::PsmcModel;
use psmc_rs::io::params::load_params;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Run inference on `input` into `dir/<name>.json` with `args`.
fn fit(dir: &TempDir, input: &Path, name: &str, n_iter: usize, args: &[&str]) -> PathBuf {
    let out = dir.path(&format!("{name}.json"));
    let mut all = vec![
        input.display().to_string(),
        out.display().to_string(),
        n_iter.to_string(),
        "--no-progress".to_string(),
    ];
    all.extend(args.iter().map(|s| s.to_string()));
    run_ok(&all);
    out
}

fn gunzip(path: &Path) -> String {
    let mut text = String::new();
    flate2::read::MultiGzDecoder::new(fs::File::open(path).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    text
}

/// The data rows of a TMRCA TSV as columns of numbers (after the four
/// identifying columns).
fn tmrca_rows(text: &str) -> Vec<(usize, Vec<f64>)> {
    text.lines()
        .skip(1)
        .map(|l| {
            let cols: Vec<&str> = l.split('\t').collect();
            let state = cols[4].parse().unwrap();
            (
                state,
                cols[5..].iter().map(|c| c.parse().unwrap()).collect(),
            )
        })
        .collect()
}

/// The same run with 1 and 4 threads writes the same bytes.
#[test]
fn outputs_do_not_depend_on_the_thread_count() {
    let dir = TempDir::new("threads");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 20_000);
    let run = |threads: &str| {
        let tmrca = dir.path(&format!("t{threads}.tsv"));
        let out = fit(
            &dir,
            &input,
            &format!("t{threads}"),
            3,
            &["--threads", threads, "--tmrca-out", tmrca.to_str().unwrap()],
        );
        (fs::read(out).unwrap(), fs::read(tmrca).unwrap())
    };
    let (json1, tmrca1) = run("1");
    let (json4, tmrca4) = run("4");
    assert!(
        json1 == json4,
        "JSON output differs between 1 and 4 threads"
    );
    assert!(
        tmrca1 == tmrca4,
        "TMRCA output differs between 1 and 4 threads"
    );
}

/// TMRCA times are each state's average coalescent time (C's TC), scaled
/// by --bin-size and --gen-years; a .gz path gives gzip.
#[test]
fn tmrca_times_are_the_states_average_times() {
    let dir = TempDir::new("tmrca");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let plain = dir.path("t.tsv");
    let out = fit(
        &dir,
        &input,
        "fit",
        1,
        &["--tmrca-out", plain.to_str().unwrap()],
    );
    let model = PsmcModel::from_params_file(&load_params(&out).unwrap()).unwrap();
    let years = model.state_tmrca_years(100.0, 25.0);
    let text = fs::read_to_string(&plain).unwrap();
    let rows = tmrca_rows(&text);
    assert_eq!(rows.len(), 3 * 3000);
    for (state, cols) in &rows {
        // tmrca_map_years is printed with 9 significant digits.
        let map = cols[0];
        assert!(
            (map - years[*state]).abs() <= 1e-8 * years[*state],
            "{map} vs {}",
            years[*state]
        );
    }

    let gz = dir.path("t.tsv.gz");
    fit(
        &dir,
        &input,
        "fit_gz",
        1,
        &["--tmrca-out", gz.to_str().unwrap()],
    );
    assert_eq!(&fs::read(&gz).unwrap()[..2], &[0x1f, 0x8b]);
    assert_eq!(gunzip(&gz), text);
}

/// --gen-years scales every time: doubling it doubles the TMRCA years
/// and leaves the fit alone.
#[test]
fn generation_time_scales_the_time_axis() {
    let dir = TempDir::new("gen_years");
    let input = dir.path("in.psmcfa");
    small_psmcfa(&input, 3000);
    let run = |years: &str| {
        let tmrca = dir.path(&format!("g{years}.tsv"));
        let out = fit(
            &dir,
            &input,
            &format!("g{years}"),
            1,
            &["--gen-years", years, "--tmrca-out", tmrca.to_str().unwrap()],
        );
        (json(&out), tmrca_rows(&fs::read_to_string(tmrca).unwrap()))
    };
    let (fit25, rows25) = run("25");
    let (fit50, rows50) = run("50");
    assert_eq!(fit25["lam"], fit50["lam"]);
    assert_eq!(fit25["theta"], fit50["theta"]);
    for ((s25, a), (s50, b)) in rows25.iter().zip(&rows50) {
        assert_eq!(s25, s50);
        // map, mean, q2.5 and q97.5 are times; pmax and entropy are not.
        for k in 0..4 {
            assert!((b[k] / a[k] - 2.0).abs() < 1e-8, "{} vs {}", b[k], a[k]);
        }
        assert_eq!(a[4..], b[4..]);
    }
}
