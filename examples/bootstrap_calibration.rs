//! Calibration study for the PSMC bootstrap (not a CI test).
//!
//! Data are simulated from a known PSMC model (pattern `4+5*3+4`): hidden
//! states from the model's `sigma` and `p_kl`, emissions from `em`. Every fit
//! is the same estimator: C-style fresh start at θ0/`--tr-ratio`, then
//! `--iters` EM iterations with smoothing weight `--smooth-lambda`. Both
//! default to the CLI's defaults (4 and 0), so by default every fit, main or
//! replicate, is the one `psmc-rs` makes with its default settings.
//!
//! * The *true* sampling SD of log10 Ne is the SD across fits to independently
//!   simulated datasets.
//! * The *bootstrap* SD is the SD across replicates of one dataset, for each
//!   of the first `--base` datasets, for:
//!   - the old CLI default: 50,000-bin blocks resampled within each sequence,
//!     each replicate starting from the full-data fit;
//!   - the C scheme (`run_bootstrap`) with several `splitfa` trunk sizes.
//!
//! SDs are taken at 30 log-spaced times between the true model's first and
//! last interval boundaries (years, with mu = 2.5e-8, 100 bp bins, 25-year
//! generations). The table reports the median over time points of the SD
//! and of the per-point ratio bootstrap SD / true SD, pooled over base
//! datasets, plus the same for a robust spread (IQR / 1.349).
//!
//! The default run fits 40 + 6 x 90 models and takes about 4 minutes with
//! `--threads 4` on an Apple M1 Pro, at 34 MB of resident memory; its
//! tables do not depend on `--threads`. The numbers in the
//! `psmc_rs::bootstrap` module docs come from this default run.
//!
//! ```text
//! cargo run --release --example bootstrap_calibration -- [--threads 3]
//!     [--datasets 40] [--base 6] [--reps 30] [--seqs 40] [--len 10000]
//!     [--iters 20] [--theta 0.05] [--trunks 500000,10000,1000]
//!     [--tr-ratio 4] [--smooth-lambda 0]
//! ```

use anyhow::{Result, bail};
use psmc_rs::PsmcModel;
use psmc_rs::bootstrap::{
    BootstrapConfig, new_rng, replicate_initial_model, replicate_rng, run_bootstrap,
    sample_sequences_block_bootstrap, split_for_bootstrap,
};
use psmc_rs::em::em_train;
use psmc_rs::opt::MStepConfig;
use rand::RngCore;
use rayon::prelude::*;
use std::time::Instant;

const PATTERN: &str = "4+5*3+4";
const N_STEPS: usize = 22;
const T_MAX: f64 = 15.0;
/// θ/ρ of the true model.
const TRUE_TR_RATIO: f64 = 5.0;
const MU: f64 = 2.5e-8;
const BIN: f64 = 100.0;
const GEN_YEARS: f64 = 25.0;
const N_GRID: usize = 30;
/// Grouped lambdas of the true model: recent expansion, older bottleneck.
const TRUE_LAMBDA: [f64; 7] = [2.0, 1.0, 0.4, 0.3, 0.8, 1.6, 1.2];

struct Args {
    threads: usize,
    datasets: usize,
    base: usize,
    reps: usize,
    seqs: usize,
    len: usize,
    iters: usize,
    theta: f64,
    trunks: Vec<usize>,
    /// θ0/ρ0 of every fit's start (CLI `--tr-ratio`).
    tr_ratio: f64,
    /// Smoothing weight of every M-step (CLI `--smooth-lambda`).
    smooth_lambda: f64,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        threads: 3,
        datasets: 40,
        base: 6,
        reps: 30,
        seqs: 40,
        len: 10_000,
        iters: 20,
        theta: 0.05,
        trunks: vec![500_000, 10_000, 1000],
        tr_ratio: 4.0,
        smooth_lambda: 0.0,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    for pair in argv.chunks(2) {
        let [key, val] = pair else {
            bail!("expected --name value pairs");
        };
        match key.as_str() {
            "--threads" => a.threads = val.parse()?,
            "--datasets" => a.datasets = val.parse()?,
            "--base" => a.base = val.parse()?,
            "--reps" => a.reps = val.parse()?,
            "--seqs" => a.seqs = val.parse()?,
            "--len" => a.len = val.parse()?,
            "--iters" => a.iters = val.parse()?,
            "--theta" => a.theta = val.parse()?,
            "--trunks" => a.trunks = val.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--tr-ratio" => a.tr_ratio = val.parse()?,
            "--smooth-lambda" => a.smooth_lambda = val.parse()?,
            _ => bail!("unknown option {key}"),
        }
    }
    if a.datasets < 2 || a.reps < 2 || a.base < 1 {
        bail!("need --datasets >= 2, --reps >= 2 and --base >= 1");
    }
    Ok(a)
}

fn true_model(theta: f64) -> Result<PsmcModel> {
    let mut m = PsmcModel::new(
        T_MAX,
        N_STEPS,
        theta,
        theta / TRUE_TR_RATIO,
        MU,
        Some(PATTERN.to_string()),
    )?;
    m.lam = TRUE_LAMBDA.to_vec();
    m.param_recalculate()?;
    Ok(m)
}

fn unit(rng: &mut impl RngCore) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn cumulative(p: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let mut acc = 0.0;
    p.into_iter()
        .map(|v| {
            acc += v;
            acc
        })
        .collect()
}

fn draw(cum: &[f64], u: f64) -> usize {
    let total = *cum.last().unwrap();
    cum.partition_point(|&c| c < u * total).min(cum.len() - 1)
}

/// `n_seqs` independent sequences of `len` bins from the PSMC HMM (no missing data).
fn simulate(model: &PsmcModel, n_seqs: usize, len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = new_rng(seed);
    let n = model.n_steps + 1;
    let start = cumulative(model.sigma.iter().copied());
    let trans: Vec<Vec<f64>> = (0..n)
        .map(|k| cumulative(model.p_kl.row(k).iter().copied()))
        .collect();
    let p_het: Vec<f64> = (0..n).map(|k| model.em[(1, k)]).collect();
    (0..n_seqs)
        .map(|_| {
            let mut k = draw(&start, unit(&mut rng));
            (0..len)
                .map(|i| {
                    if i > 0 {
                        k = draw(&trans[k], unit(&mut rng));
                    }
                    u8::from(unit(&mut rng) < p_het[k])
                })
                .collect()
        })
        .collect()
}

fn mstep(args: &Args) -> MStepConfig {
    MStepConfig {
        progress: false,
        lambda: args.smooth_lambda,
        ..MStepConfig::default()
    }
}

/// The estimator: C-style fresh start, then EM.
fn fit(seqs: &[Vec<u8>], template: &PsmcModel, args: &Args) -> Result<PsmcModel> {
    let starts = vec![true; seqs.len()];
    let mut m = replicate_initial_model(template, seqs, T_MAX, args.tr_ratio)?;
    em_train(&mut m, seqs, &starts, args.iters, &mstep(args))?;
    m.param_recalculate()?;
    Ok(m)
}

/// (time in years, Ne) at each interval start.
fn curve(m: &PsmcModel) -> Result<(Vec<f64>, Vec<f64>)> {
    Ok(m.ne_curve(BIN, GEN_YEARS)?
        .into_iter()
        .map(|[x, y]| (x, y))
        .unzip())
}

fn log10_ne_on_grid(m: &PsmcModel, grid: &[f64]) -> Result<Vec<f64>> {
    let (xs, ys) = curve(m)?;
    Ok(grid
        .iter()
        .map(|&x| {
            let pos = xs.partition_point(|&v| v <= x).max(1);
            ys[pos - 1].log10()
        })
        .collect())
}

/// Spread of log10 Ne across fits at each grid point.
struct Spread {
    /// Sample SD.
    sd: Vec<f64>,
    /// Interquartile range / 1.349: equals the SD for normal data, but is not
    /// driven by a few runaway fits.
    robust: Vec<f64>,
}

fn spread(rows: &[Vec<f64>]) -> Spread {
    let n = rows.len() as f64;
    let mut sd = Vec::with_capacity(N_GRID);
    let mut robust = Vec::with_capacity(N_GRID);
    for j in 0..rows[0].len() {
        let col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
        let mean = col.iter().sum::<f64>() / n;
        sd.push((col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt());
        robust.push((quantile(&col, 0.75) - quantile(&col, 0.25)) / 1.349);
    }
    Spread { sd, robust }
}

fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = q * (s.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

fn median(v: &[f64]) -> f64 {
    quantile(v, 0.5)
}

/// Replicate curves of one trunk size on the current base dataset.
struct Fitted {
    trunk: usize,
    chunks: Vec<Vec<u8>>,
    curves: Vec<Vec<f64>>,
}

/// Bootstrap spreads of one scheme, one entry per base dataset.
struct Scheme {
    name: String,
    per_base: Vec<Spread>,
}

impl Scheme {
    /// Per-point mean over base datasets of `pick(spread)`.
    fn mean(&self, pick: impl Fn(&Spread) -> &Vec<f64>) -> Vec<f64> {
        let nb = self.per_base.len() as f64;
        (0..N_GRID)
            .map(|j| self.per_base.iter().map(|s| pick(s)[j]).sum::<f64>() / nb)
            .collect()
    }
}

fn push(schemes: &mut Vec<Scheme>, name: &str, s: Spread) {
    match schemes.iter_mut().find(|x| x.name == name) {
        Some(x) => x.per_base.push(s),
        None => schemes.push(Scheme {
            name: name.to_string(),
            per_base: vec![s],
        }),
    }
}

fn ratios(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x / y).collect()
}

fn main() -> Result<()> {
    let args = parse_args()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let truth = true_model(args.theta)?;
    let (tx, _) = curve(&truth)?;
    let (lo, hi) = (tx[1].ln(), tx[N_STEPS].ln());
    let grid: Vec<f64> = (0..N_GRID)
        .map(|i| (lo + (hi - lo) * (i as f64 + 0.5) / N_GRID as f64).exp())
        .collect();
    println!(
        "true model: pattern {PATTERN}, theta {} / bin, rho {} / bin, lambda {TRUE_LAMBDA:?}",
        args.theta,
        args.theta / TRUE_TR_RATIO
    );
    println!(
        "data: {} sequences x {} bins; {} EM iterations per fit; {} threads",
        args.seqs, args.len, args.iters, args.threads
    );
    println!(
        "fits: start at theta0/{}, smoothing weight {}",
        args.tr_ratio, args.smooth_lambda
    );

    let started = Instant::now();
    let datasets: Vec<Vec<Vec<u8>>> = (0..args.datasets)
        .map(|d| simulate(&truth, args.seqs, args.len, 1000 + d as u64))
        .collect();
    let fits: Vec<PsmcModel> = datasets
        .par_iter()
        .map(|seqs| fit(seqs, &truth, &args))
        .collect::<Result<_>>()?;
    let curves: Vec<Vec<f64>> = fits
        .iter()
        .map(|m| log10_ne_on_grid(m, &grid))
        .collect::<Result<_>>()?;
    let true_spread = spread(&curves);
    let truth_curve = log10_ne_on_grid(&truth, &grid)?;
    let bias: Vec<f64> = (0..N_GRID)
        .map(|j| {
            let mean = curves.iter().map(|c| c[j]).sum::<f64>() / curves.len() as f64;
            (mean - truth_curve[j]).abs()
        })
        .collect();
    println!(
        "true sampling SD over {} datasets: median {:.3} (robust {:.3}); median |bias| {:.3} [{:.0}s]",
        args.datasets,
        median(&true_spread.sd),
        median(&true_spread.robust),
        median(&bias),
        started.elapsed().as_secs_f64()
    );

    let mut schemes: Vec<Scheme> = Vec::new();
    for (b, seqs) in datasets.iter().take(args.base).enumerate() {
        let main = &fits[b];
        let t0 = Instant::now();

        // Old CLI default: within-sequence 50,000-bin blocks, warm start.
        let old: Vec<Vec<f64>> = (0..args.reps)
            .into_par_iter()
            .map(|rep| -> Result<Vec<f64>> {
                let sampled =
                    sample_sequences_block_bootstrap(seqs, 50_000, &mut replicate_rng(7, rep))?;
                let starts = vec![true; sampled.len()];
                let mut m = main.clone();
                em_train(&mut m, &sampled, &starts, args.iters, &mstep(&args))?;
                m.param_recalculate()?;
                log10_ne_on_grid(&m, &grid)
            })
            .collect::<Result<_>>()?;
        push(
            &mut schemes,
            "old default (50k blocks in-seq, warm start)",
            spread(&old),
        );

        // Trunk sizes that cut this dataset identically give identical
        // replicates (same seeds), so they are fitted once.
        let mut done: Vec<Fitted> = Vec::new();
        for &trunk in &args.trunks {
            let name = format!("splitfa trunk {trunk} (fresh start)");
            let chunks = split_for_bootstrap(seqs, trunk);
            if let Some(same) = done.iter().find(|f| f.chunks == chunks) {
                if b == 0 {
                    println!(
                        "  trunk {trunk} cuts the data like trunk {}: same replicates",
                        same.trunk
                    );
                }
                let s = spread(&same.curves);
                push(&mut schemes, &name, s);
                continue;
            }
            let cfg = BootstrapConfig {
                replicates: args.reps,
                trunk_size: trunk,
                seed: 7,
                em_iters: args.iters,
                tr_ratio: args.tr_ratio,
                t_max: T_MAX,
                batch_size: None,
                out_dir: None,
            };
            let models = run_bootstrap(seqs, main, &cfg, &mstep(&args), None, &|_| {})?;
            let reps: Vec<Vec<f64>> = models
                .iter()
                .map(|m| log10_ne_on_grid(m, &grid))
                .collect::<Result<_>>()?;
            push(&mut schemes, &name, spread(&reps));
            done.push(Fitted {
                trunk,
                chunks,
                curves: reps,
            });
        }
        println!(
            "  base dataset {}: {} replicates per scheme [{:.0}s]",
            b + 1,
            args.reps,
            t0.elapsed().as_secs_f64()
        );
    }

    // Ratios are per time point (bootstrap spread / true spread), pooled over
    // time points and base datasets.
    println!();
    println!(
        "{:<44} {:>7} {:>8} {:>6} {:>7} {:>8}  per base dataset (median SD/true)",
        "scheme", "SD", "SD/true", "<true", "robust", "rob/true"
    );
    println!(
        "{:<44} {:>7.3} {:>8} {:>6} {:>7.3} {:>8}",
        "true sampling spread",
        median(&true_spread.sd),
        "1.00",
        "",
        median(&true_spread.robust),
        "1.00"
    );
    for s in &schemes {
        let sd_ratio: Vec<f64> = s
            .per_base
            .iter()
            .flat_map(|p| ratios(&p.sd, &true_spread.sd))
            .collect();
        let rob_ratio: Vec<f64> = s
            .per_base
            .iter()
            .flat_map(|p| ratios(&p.robust, &true_spread.robust))
            .collect();
        let below = sd_ratio.iter().filter(|&&r| r < 1.0).count() as f64 / sd_ratio.len() as f64;
        let per_base: Vec<String> = s
            .per_base
            .iter()
            .map(|p| format!("{:.2}", median(&ratios(&p.sd, &true_spread.sd))))
            .collect();
        println!(
            "{:<44} {:>7.3} {:>8.2} {:>5.0}% {:>7.3} {:>8.2}  {}",
            s.name,
            median(&s.mean(|p| &p.sd)),
            median(&sd_ratio),
            100.0 * below,
            median(&s.mean(|p| &p.robust)),
            median(&rob_ratio),
            per_base.join(" ")
        );
    }

    println!("\nSD of log10 Ne by time (bootstrap columns: mean over base datasets)");
    print!("{:>12} {:>8} {:>8}", "years", "truth", "true SD");
    for i in 0..schemes.len() {
        print!(" {:>8}", format!("[{}]", i + 1));
    }
    println!();
    let means: Vec<Vec<f64>> = schemes.iter().map(|s| s.mean(|p| &p.sd)).collect();
    for j in (0..N_GRID).step_by(3) {
        print!(
            "{:>12.0} {:>8.2} {:>8.3}",
            grid[j], truth_curve[j], true_spread.sd[j]
        );
        for m in &means {
            print!(" {:>8.3}", m[j]);
        }
        println!();
    }
    for (i, s) in schemes.iter().enumerate() {
        println!("[{}] {}", i + 1, s.name);
    }
    println!("\ntotal {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}
