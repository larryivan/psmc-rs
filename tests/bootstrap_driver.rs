//! `run_bootstrap` re-runs the full estimator on every replicate, as C
//! `psmc -b` does: splitfa trunks resampled across the whole input, a fresh
//! C-style start, then EM. Replicates must be reproducible from the seed and
//! independent of the thread count.

use psmc_rs::bootstrap::{
    BootstrapConfig, DEFAULT_BOOTSTRAP_TRUNK, replicate_initial_model, replicate_rng,
    resample_chunks, rows_to_sequences, run_bootstrap, split_for_bootstrap,
};
use psmc_rs::io::params::load_params;
use psmc_rs::io::psmcfa::read_psmcfa;
use psmc_rs::opt::MStepConfig;
use psmc_rs::{PsmcModel, initial_theta};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

const PATTERN: &str = "4+5*3+4";
const N_STEPS: usize = 22;

/// The three reference sequences cut down to `len` bins each.
fn reference_seqs(len: usize) -> Vec<Vec<u8>> {
    let obs = read_psmcfa(
        Path::new("tests/data/c_reference/papuan_chr1_3seg.psmcfa"),
        None,
    )
    .unwrap();
    rows_to_sequences(&obs.rows, &obs.row_starts)
        .unwrap()
        .into_iter()
        .map(|s| s[..len].to_vec())
        .collect()
}

/// A model that looks like a finished fit: nothing at its initial value.
fn fitted_template() -> PsmcModel {
    let mut m = PsmcModel::new(
        15.0,
        N_STEPS,
        0.02,
        0.004,
        2.5e-8,
        Some(PATTERN.to_string()),
    )
    .unwrap();
    m.theta = 0.0137;
    m.rho = 0.0021;
    m.t_max = 27.5;
    for (i, l) in m.lam.iter_mut().enumerate() {
        *l = 0.4 + 0.3 * i as f64;
    }
    m.param_recalculate().unwrap();
    m
}

fn mstep() -> MStepConfig {
    MStepConfig {
        progress: false,
        ..MStepConfig::default()
    }
}

fn config(replicates: usize, trunk_size: usize, em_iters: usize) -> BootstrapConfig {
    BootstrapConfig {
        replicates,
        trunk_size,
        seed: 2024,
        em_iters,
        tr_ratio: 5.0,
        t_max: 15.0,
        batch_size: None,
        out_dir: None,
    }
}

fn params(m: &PsmcModel) -> Vec<u64> {
    let mut v = vec![m.theta.to_bits(), m.rho.to_bits(), m.t_max.to_bits()];
    v.extend(m.lam.iter().map(|x| x.to_bits()));
    v
}

fn no_progress(_: usize) {}

#[test]
fn replicate_initial_model_is_a_fresh_c_start() {
    let rows = reference_seqs(4000);
    let template = fitted_template();
    let m = replicate_initial_model(&template, &rows, 15.0, 5.0).unwrap();

    let theta0 = initial_theta(&rows).unwrap();
    assert_eq!(m.theta, theta0);
    assert_eq!(m.rho, theta0 / 5.0);
    assert_eq!(
        m.t_max, 15.0,
        "must use the configured t_max, not the fitted one"
    );
    assert_eq!(m.lam, vec![1.0; 7]);
    assert_eq!(m.n_steps, N_STEPS);
    assert_eq!(m.mu, template.mu);
    assert_eq!(m.pattern.as_deref(), Some(PATTERN));

    let fresh = PsmcModel::new(
        15.0,
        N_STEPS,
        theta0,
        theta0 / 5.0,
        2.5e-8,
        Some(PATTERN.to_string()),
    )
    .unwrap();
    assert_eq!(m.sigma, fresh.sigma);
    assert_eq!(m.p_kl, fresh.p_kl);
    assert_eq!(m.em, fresh.em);

    assert!(replicate_initial_model(&template, &rows, 15.0, 0.0).is_err());
    assert!(replicate_initial_model(&template, &rows, f64::NAN, 4.0).is_err());
}

#[test]
fn replicates_start_fresh_rather_than_from_the_full_fit() {
    let seqs = reference_seqs(6000);
    let template = fitted_template();
    let cfg = config(4, 1000, 0);
    let models = run_bootstrap(&seqs, &template, &cfg, &mstep(), None, &no_progress).unwrap();
    assert_eq!(models.len(), 4);

    let chunks = split_for_bootstrap(&seqs, cfg.trunk_size);
    for (rep, m) in models.iter().enumerate() {
        let sampled = resample_chunks(&chunks, &mut replicate_rng(cfg.seed, rep));
        let theta0 = initial_theta(&sampled).unwrap();
        assert_eq!(m.theta, theta0, "replicate {rep}: theta0 of its own data");
        assert_eq!(m.rho, theta0 / cfg.tr_ratio, "replicate {rep}");
        assert_eq!(m.t_max, cfg.t_max, "replicate {rep}");
        assert_eq!(m.lam, vec![1.0; 7], "replicate {rep}");
    }
}

#[test]
fn bootstrap_does_not_depend_on_thread_count() {
    let seqs = reference_seqs(6000);
    let template = fitted_template();
    let cfg = config(6, 1000, 2);
    let run = |threads: usize| -> Vec<Vec<u64>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_bootstrap(&seqs, &template, &cfg, &mstep(), None, &no_progress))
            .unwrap()
            .iter()
            .map(params)
            .collect()
    };
    let one = run(1);
    assert_eq!(one, run(3));
    assert_eq!(one, run(1), "same seed must reproduce the same replicates");
    let distinct: std::collections::HashSet<_> = one.iter().collect();
    assert_eq!(distinct.len(), one.len(), "replicates should differ");
}

/// Data made only of sequences shorter than a trunk: every replicate used to
/// see the input unchanged and returned the same curve.
#[test]
fn short_sequences_give_replicate_variance() {
    let seqs: Vec<Vec<u8>> = reference_seqs(30_000)
        .iter()
        .flat_map(|s| s.chunks(2500).map(<[u8]>::to_vec).collect::<Vec<_>>())
        .collect();
    assert!(seqs.iter().all(|s| s.len() < DEFAULT_BOOTSTRAP_TRUNK));
    let cfg = config(5, DEFAULT_BOOTSTRAP_TRUNK, 1);
    let models = run_bootstrap(
        &seqs,
        &fitted_template(),
        &cfg,
        &mstep(),
        None,
        &no_progress,
    )
    .unwrap();
    let thetas: Vec<f64> = models.iter().map(|m| m.theta).collect();
    let mean = thetas.iter().sum::<f64>() / thetas.len() as f64;
    let var = thetas.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / thetas.len() as f64;
    assert!(var > 0.0, "all replicates identical: {thetas:?}");
    for (i, a) in models.iter().enumerate() {
        for b in &models[i + 1..] {
            assert_ne!(params(a), params(b));
        }
    }
}

#[test]
fn replicate_files_are_written() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bootstrap_driver_out");
    let _ = std::fs::remove_dir_all(&dir);
    let seqs = reference_seqs(3000);
    let cfg = BootstrapConfig {
        out_dir: Some(dir.clone()),
        ..config(3, 1000, 1)
    };
    let models = run_bootstrap(
        &seqs,
        &fitted_template(),
        &cfg,
        &mstep(),
        None,
        &no_progress,
    )
    .unwrap();
    for (rep, m) in models.iter().enumerate() {
        let saved = load_params(&dir.join(format!("replicate_{:03}.json", rep + 1))).unwrap();
        assert_eq!(saved.theta, m.theta);
        assert_eq!(saved.rho, m.rho);
        assert_eq!(saved.t_max, m.t_max);
        assert_eq!(saved.lam, m.lam);
        assert_eq!(saved.pattern.as_deref(), Some(PATTERN));
    }
    assert!(!dir.join("replicate_004.json").exists());
}

/// Progress counts must arrive in order, ending at the replicate count, even
/// when several workers finish together and the callback is slow.
#[test]
fn on_done_reports_counts_in_order() {
    let seqs = reference_seqs(3000);
    let template = fitted_template();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for _ in 0..3 {
        let seen = Mutex::new(Vec::new());
        let on_done = |n: usize| {
            // A slow sink for odd counts gives the worker holding the next
            // count a chance to overtake.
            if n % 2 == 1 {
                std::thread::sleep(std::time::Duration::from_micros(500));
            }
            seen.lock().unwrap().push(n);
        };
        pool.install(|| {
            run_bootstrap(
                &seqs,
                &template,
                &config(40, 1000, 0),
                &mstep(),
                None,
                &on_done,
            )
        })
        .unwrap();
        assert_eq!(
            seen.into_inner().unwrap(),
            (1..=40).collect::<Vec<_>>(),
            "on_done counts out of order"
        );
    }
}

#[test]
fn cancelled_bootstrap_stops_with_an_error() {
    let seqs = reference_seqs(3000);
    let cancel = AtomicBool::new(true);
    let err = run_bootstrap(
        &seqs,
        &fitted_template(),
        &config(3, 1000, 1),
        &mstep(),
        Some(&cancel),
        &no_progress,
    )
    .unwrap_err();
    assert!(err.to_string().contains("cancelled"), "{err}");
}

#[test]
fn invalid_settings_are_rejected() {
    let seqs = reference_seqs(3000);
    let t = fitted_template();
    let run = |cfg: BootstrapConfig| run_bootstrap(&seqs, &t, &cfg, &mstep(), None, &no_progress);
    assert!(run(config(2, 0, 1)).is_err());
    assert!(
        run(BootstrapConfig {
            tr_ratio: 0.0,
            ..config(2, 1000, 1)
        })
        .is_err()
    );
    assert!(
        run(BootstrapConfig {
            t_max: f64::NAN,
            ..config(2, 1000, 1)
        })
        .is_err()
    );
    assert!(
        run(BootstrapConfig {
            batch_size: Some(0),
            ..config(2, 1000, 1)
        })
        .is_err()
    );
    assert!(run(config(0, 1000, 1)).unwrap().is_empty());
    assert!(run_bootstrap(&[], &t, &config(2, 1000, 1), &mstep(), None, &no_progress).is_err());
}

#[test]
fn batch_size_only_changes_row_layout() {
    let seqs = reference_seqs(4000);
    let t = fitted_template();
    let plain =
        run_bootstrap(&seqs, &t, &config(2, 1000, 0), &mstep(), None, &no_progress).unwrap();
    let batched = run_bootstrap(
        &seqs,
        &t,
        &BootstrapConfig {
            batch_size: Some(300),
            ..config(2, 1000, 0)
        },
        &mstep(),
        None,
        &no_progress,
    )
    .unwrap();
    let a: Vec<_> = plain.iter().map(params).collect();
    let b: Vec<_> = batched.iter().map(params).collect();
    assert_eq!(a, b);
}
