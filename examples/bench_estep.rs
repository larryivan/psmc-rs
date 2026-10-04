//! E-step benchmark on a psmcfa input.
//!
//! ```text
//! cargo run --release --example bench_estep -- INPUT.psmcfa [--threads N] [--reps R]
//!     [--pattern P] [--install] [--loglike | --tmrca OUT.tsv[.gz]]
//! ```
//!
//! `--loglike` times the forward-only `log_likelihood` instead of the E-step;
//! `--tmrca` times posterior TMRCA decoding written to `OUT`.
//! `--threads` sizes the global rayon pool (default: rayon's default). With
//! `--install` the E-step runs inside `ThreadPool::install`, the way bootstrap
//! replicates and a per-run `--threads` pool call it. Each repetition prints the
//! wall time, the time per bin, the log-likelihood and a checksum of the
//! sufficient statistics (bit patterns, so runs can be compared for exact
//! equality). Measure peak memory with `/usr/bin/time -l` (macOS) or
//! `/usr/bin/time -v` (Linux). The alpha-cache budget follows
//! `PSMC_ALPHA_CACHE_MB`.

use psmc_rs::hmm::{EStepResult, e_step_streaming, log_likelihood, write_tmrca_posterior_tsv};
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::{PsmcModel, initial_theta};
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    input: PathBuf,
    threads: Option<usize>,
    reps: usize,
    pattern: String,
    install: bool,
    loglike: bool,
    tmrca: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut args = Args {
        input: PathBuf::new(),
        threads: None,
        reps: 3,
        pattern: "4+25*2+4+6".to_string(),
        install: false,
        loglike: false,
        tmrca: None,
    };
    let mut input = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => args.threads = it.next().and_then(|v| v.parse().ok()),
            "--reps" => args.reps = it.next().and_then(|v| v.parse().ok()).unwrap_or(3),
            "--pattern" => args.pattern = it.next().expect("--pattern needs a value"),
            "--install" => args.install = true,
            "--loglike" => args.loglike = true,
            "--tmrca" => args.tmrca = it.next().map(PathBuf::from),
            _ => input = Some(PathBuf::from(a)),
        }
    }
    args.input = input.expect(
        "usage: bench_estep INPUT.psmcfa [--threads N] [--reps R] [--pattern P] [--install] [--loglike | --tmrca OUT]",
    );
    args
}

fn n_states(pattern: &str) -> usize {
    pattern
        .split('+')
        .map(|t| match t.split_once('*') {
            Some((a, b)) => a.parse::<usize>().unwrap() * b.parse::<usize>().unwrap(),
            None => t.parse::<usize>().unwrap(),
        })
        .sum()
}

fn checksum(e: &EStepResult) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: f64| {
        h ^= v.to_bits();
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(e.loglike);
    for g in &e.stats.gobs {
        g.iter().for_each(|v| mix(*v));
    }
    e.stats.xi.iter().for_each(|v| mix(*v));
    h
}

fn main() {
    let args = parse_args();
    if let Some(t) = args.threads
        && !args.install
    {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .expect("failed to size the global rayon pool");
    }
    let pool = args.install.then(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads.unwrap_or(0))
            .build()
            .expect("failed to build rayon pool")
    });

    let obs = read_psmcfa(&args.input, None).expect("failed to read input");
    let bins: usize = obs.rows.iter().map(Vec::len).sum();
    let theta0 = initial_theta(&obs.rows).expect("no usable bins");
    let model = PsmcModel::new(
        15.0,
        n_states(&args.pattern) - 1,
        theta0,
        theta0 / 4.0,
        2.5e-8,
        Some(args.pattern.clone()),
    )
    .expect("model init failed");
    let threads = pool
        .as_ref()
        .map_or_else(rayon::current_num_threads, |p| p.current_num_threads());
    println!(
        "input={} seqs={} bins={bins} states={} threads={threads} install={}",
        args.input.display(),
        obs.row_starts.iter().filter(|s| **s).count(),
        model.n_steps + 1,
        args.install,
    );

    let run = || {
        if let Some(out) = &args.tmrca {
            let t = model.compute_t(0.1);
            let years: Vec<f64> = t[..=model.n_steps].iter().map(|v| v * 1e4).collect();
            let data = write_tmrca_posterior_tsv(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                &obs.rows,
                &obs.row_starts,
                &years,
                out,
                false,
                20_000,
            )
            .expect("TMRCA decoding failed");
            let mass: f64 = data.state_mass.iter().sum();
            return (mass, 0);
        }
        if args.loglike {
            let loglike = log_likelihood(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                &obs.rows,
                &obs.row_starts,
            )
            .expect("log-likelihood failed");
            return (loglike, 0);
        }
        let e = e_step_streaming(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            false,
            "E",
        )
        .expect("E-step failed");
        (e.loglike, checksum(&e))
    };
    let mut times = Vec::with_capacity(args.reps);
    for rep in 0..args.reps {
        let t0 = Instant::now();
        let (loglike, sum) = match &pool {
            Some(p) => p.install(run),
            None => run(),
        };
        let secs = t0.elapsed().as_secs_f64();
        times.push(secs);
        println!(
            "rep={rep} estep_s={secs:.3} ns_per_bin={:.0} loglike={:.10} checksum={:016x}",
            secs * 1e9 / bins as f64,
            loglike,
            sum
        );
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    println!(
        "median estep_s={median:.3} ns_per_bin={:.0}",
        median * 1e9 / bins as f64
    );
}
