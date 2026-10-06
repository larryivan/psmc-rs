//! E-step benchmark on any input (psmcfa, VCF, multihetsep).
//!
//! ```text
//! cargo run --release --example bench_estep -- INPUT [--threads N] [--reps R]
//!     [--pattern P] [--install] [--loglike | --tmrca OUT.tsv[.gz]]
//!     [--mask BED] [--callable]
//! ```
//!
//! `--loglike` times the forward-only `log_likelihood` instead of the E-step;
//! `--tmrca` times posterior TMRCA decoding written to `OUT`. The input is
//! read as the CLI reads it, with the callable mask `--mask` for a VCF;
//! `--callable` runs every pass with the callable bases of each bin (VCF and
//! multihetsep input), as `--bin-callable auto` fits them, and without it
//! with C psmc's emissions, so that the two time the cost of the counts.
//! `--threads` sizes the global rayon pool (default: rayon's default). With
//! `--install` the E-step runs inside `ThreadPool::install`, the way bootstrap
//! replicates and a per-run `--threads` pool call it. Each repetition prints the
//! wall time, the time per bin, the log-likelihood and a checksum of the
//! sufficient statistics, the callable-length sums included with
//! `--callable` (bit patterns, so runs can be compared for exact
//! equality). Measure peak memory with `/usr/bin/time -l` (macOS) or
//! `/usr/bin/time -v` (Linux). The alpha-cache budget follows
//! `PSMC_ALPHA_CACHE_MB`.

use psmc_rs::hmm::{
    EStepOptions, EStepResult, TmrcaOptions, e_step_with_options, log_likelihood_with_options,
    write_tmrca_posterior_tsv_with_options,
};
use psmc_rs::io::input::{InputOptions, load};
use psmc_rs::{PsmcModel, initial_theta_with_callable};
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
    mask: Option<PathBuf>,
    callable: bool,
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
        mask: None,
        callable: false,
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
            "--mask" => args.mask = it.next().map(PathBuf::from),
            "--callable" => args.callable = true,
            _ => input = Some(PathBuf::from(a)),
        }
    }
    args.input = input.expect(
        "usage: bench_estep INPUT [--threads N] [--reps R] [--pattern P] [--install] [--loglike | --tmrca OUT] [--mask BED] [--callable]",
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

/// FNV-1a over the bit patterns of the log-likelihood and every sufficient
/// statistic the M-step reads: the expected symbol counts, the transition
/// counts and, with callable bases, the count-weighted `T` sums and the `K`
/// sums of each count (with the counts themselves).
fn checksum(e: &EStepResult) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |bits: u64| {
        h ^= bits;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(e.loglike.to_bits());
    for g in &e.stats.gobs {
        g.iter().for_each(|v| mix(v.to_bits()));
    }
    e.stats.xi.iter().for_each(|v| mix(v.to_bits()));
    if let Some(c) = &e.stats.callable {
        mix(c.bin_size as u64);
        c.t_weighted.iter().for_each(|v| mix(v.to_bits()));
        for (count, row) in &c.k_by_count {
            mix(u64::from(*count));
            row.iter().for_each(|v| mix(v.to_bits()));
        }
    }
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

    let opts = InputOptions {
        masks: args.mask.iter().cloned().collect(),
        ..InputOptions::default()
    };
    let obs = load(&args.input, &opts).expect("failed to read input").obs;
    let callable = obs.callable.as_ref().filter(|_| args.callable);
    assert!(
        callable.is_some() || !args.callable,
        "--callable: the input has no callable bases (psmcfa?)"
    );
    let bins: usize = obs.rows.iter().map(Vec::len).sum();
    let theta0 = initial_theta_with_callable(&obs.rows, callable).expect("no usable bins");
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
        "input={} seqs={} bins={bins} states={} threads={threads} install={} callable={}",
        args.input.display(),
        obs.row_starts.iter().filter(|s| **s).count(),
        model.n_steps + 1,
        args.install,
        callable.is_some(),
    );

    let run = || {
        if let Some(out) = &args.tmrca {
            let t = model.compute_t(0.1);
            let years: Vec<f64> = t[..=model.n_steps].iter().map(|v| v * 1e4).collect();
            let opts = TmrcaOptions {
                progress_bars: false,
                max_points: 20_000,
                callable,
            };
            let data = write_tmrca_posterior_tsv_with_options(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                &obs.rows,
                &obs.row_starts,
                &years,
                out,
                opts,
            )
            .expect("TMRCA decoding failed");
            let mass: f64 = data.state_mass.iter().sum();
            return (mass, 0);
        }
        if args.loglike {
            let opts = EStepOptions {
                callable,
                ..EStepOptions::default()
            };
            let loglike = log_likelihood_with_options(
                model.prior_matrix(),
                model.transition_matrix(),
                model.emission_matrix(),
                &obs.rows,
                &obs.row_starts,
                opts,
            )
            .expect("log-likelihood failed");
            return (loglike, 0);
        }
        let opts = EStepOptions {
            callable,
            ..EStepOptions::default()
        };
        let e = e_step_with_options(
            model.prior_matrix(),
            model.transition_matrix(),
            model.emission_matrix(),
            &obs.rows,
            &obs.row_starts,
            opts,
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
