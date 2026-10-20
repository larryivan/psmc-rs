//! `run_bootstrap` re-runs the full estimator on every replicate, as C
//! `psmc -b` does: splitfa trunks resampled across the whole input, a fresh
//! C-style start, then EM. Replicates must be reproducible from the seed and
//! independent of the thread count.

mod common;

use common::{REFERENCE_PSMCFA, TempDir, bits, fixture_obs, max_rho, model, mstep, params, pool};
use psmc_rs::bootstrap::{
    BootstrapConfig, BootstrapOptions, DEFAULT_BOOTSTRAP_TRUNK, invalid_replicate_starts,
    replicate_initial_model, replicate_rng, resample_chunks, rows_to_sequences, run_bootstrap,
    run_bootstrap_with_options, split_for_bootstrap,
};
use psmc_rs::hmm::Cancelled;
use psmc_rs::io::params::load_params;
use psmc_rs::model::NegativeTransitions;
use psmc_rs::{PsmcModel, initial_theta};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const PATTERN: &str = "4+5*3+4";

/// The three reference sequences cut down to `len` bins each.
fn reference_seqs(len: usize) -> Vec<Vec<u8>> {
    let obs = fixture_obs(REFERENCE_PSMCFA);
    rows_to_sequences(&obs.rows, &obs.row_starts)
        .unwrap()
        .into_iter()
        .map(|s| s[..len].to_vec())
        .collect()
}

/// A model that looks like a finished fit: nothing at its initial value.
fn fitted_template() -> PsmcModel {
    let mut m = model(PATTERN, 0.0137, 0.0021);
    m.t_max = 27.5;
    for (i, l) in m.lam.iter_mut().enumerate() {
        *l = 0.4 + 0.3 * i as f64;
    }
    m.param_recalculate().unwrap();
    m
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

/// The fitted parameters as bits.
fn param_bits(m: &PsmcModel) -> Vec<u64> {
    bits(&params(m))
}

/// `run_bootstrap` without cancellation or progress.
fn bootstrap(
    seqs: &[Vec<u8>],
    template: &PsmcModel,
    cfg: &BootstrapConfig,
) -> anyhow::Result<Vec<PsmcModel>> {
    run_bootstrap(seqs, template, cfg, &mstep(), None, &|_| {})
}

#[test]
fn replicate_initial_model_is_a_fresh_c_start() {
    let rows = reference_seqs(4000);
    let template = fitted_template();
    let m = replicate_initial_model(&template, &rows, 15.0, 5.0).unwrap();

    let theta0 = initial_theta(&rows).unwrap();
    let fresh = model(PATTERN, theta0, theta0 / 5.0);
    assert_eq!((m.theta, m.rho), (theta0, theta0 / 5.0));
    assert_eq!(
        m.t_max, 15.0,
        "must use the configured t_max, not the fitted one"
    );
    assert_eq!(m.lam, vec![1.0; 7]);
    assert_eq!(m.n_steps, fresh.n_steps);
    assert_eq!(m.mu, template.mu);
    assert_eq!(m.pattern.as_deref(), Some(PATTERN));
    assert_eq!(m.sigma, fresh.sigma);
    assert_eq!(m.p_kl, fresh.p_kl);
    assert_eq!(m.em, fresh.em);

    assert!(replicate_initial_model(&template, &rows, 15.0, 0.0).is_err());
    assert!(replicate_initial_model(&template, &rows, f64::NAN, 4.0).is_err());
}

#[test]
fn replicates_start_fresh_rather_than_from_the_full_fit() {
    let seqs = reference_seqs(6000);
    let cfg = config(4, 1000, 0);
    let models = bootstrap(&seqs, &fitted_template(), &cfg).unwrap();
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
        let models = pool(threads).install(|| bootstrap(&seqs, &template, &cfg));
        models.unwrap().iter().map(param_bits).collect()
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
    let models = bootstrap(&seqs, &fitted_template(), &cfg).unwrap();
    let thetas: Vec<f64> = models.iter().map(|m| m.theta).collect();
    let mean = thetas.iter().sum::<f64>() / thetas.len() as f64;
    let var = thetas.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / thetas.len() as f64;
    assert!(var > 0.0, "all replicates identical: {thetas:?}");
    for (i, a) in models.iter().enumerate() {
        for b in &models[i + 1..] {
            assert_ne!(param_bits(a), param_bits(b));
        }
    }
}

/// One sequence shorter than 1.5 trunks forms a single trunk: every replicate
/// would redraw the input unchanged, and the bands would have zero width.
#[test]
fn single_trunk_input_is_rejected() {
    let one = vec![reference_seqs(50_000).remove(0)];
    let template = fitted_template();
    let err = bootstrap(&one, &template, &config(3, DEFAULT_BOOTSTRAP_TRUNK, 1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("single trunk"), "{err}");
    assert!(err.contains("at most 33333 bins"), "{err}");

    let models = bootstrap(&one, &template, &config(3, 33_333, 0)).unwrap();
    assert_eq!(models.len(), 3);
}

#[test]
fn replicate_files_are_written() {
    let dir = TempDir::new("bootstrap_driver_out");
    let cfg = BootstrapConfig {
        out_dir: Some(dir.path("")),
        ..config(3, 1000, 1)
    };
    let models = bootstrap(&reference_seqs(3000), &fitted_template(), &cfg).unwrap();
    for (rep, m) in models.iter().enumerate() {
        let saved = load_params(&dir.path(&format!("replicate_{:03}.json", rep + 1))).unwrap();
        assert_eq!(
            (saved.theta, saved.rho, saved.t_max, &saved.lam),
            (m.theta, m.rho, m.t_max, &m.lam)
        );
        assert_eq!(saved.pattern.as_deref(), Some(PATTERN));
    }
    assert!(!dir.path("replicate_004.json").exists());
}

/// Progress counts must arrive in order, ending at the replicate count, even
/// when several workers finish together and the callback is slow.
#[test]
fn on_done_reports_counts_in_order() {
    let seqs = reference_seqs(3000);
    let template = fitted_template();
    let pool = pool(4);
    for _ in 0..3 {
        let seen = Mutex::new(Vec::new());
        let on_done = |n: usize| {
            // A slow sink for odd counts gives the worker holding the next
            // count a chance to overtake.
            if n % 2 == 1 {
                std::thread::sleep(Duration::from_micros(500));
            }
            seen.lock().unwrap().push(n);
        };
        let cfg = config(40, 1000, 0);
        pool.install(|| run_bootstrap(&seqs, &template, &cfg, &mstep(), None, &on_done))
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
    let cancel = AtomicBool::new(true);
    let err = run_bootstrap(
        &reference_seqs(3000),
        &fitted_template(),
        &config(3, 1000, 1),
        &mstep(),
        Some(&cancel),
        &|_| {},
    )
    .unwrap_err();
    assert!(err.is::<Cancelled>(), "{err:#}");
}

/// A cancel during a replicate's EM stops it there, not after its last
/// iteration: 100,000 iterations would take minutes.
#[test]
fn cancel_interrupts_a_running_replicate() {
    let seqs = reference_seqs(2000);
    let cancel = AtomicBool::new(false);
    let started = Instant::now();
    let err = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(200));
            cancel.store(true, Ordering::Relaxed);
        });
        let cfg = config(1, 1000, 100_000);
        run_bootstrap(
            &seqs,
            &fitted_template(),
            &cfg,
            &mstep(),
            Some(&cancel),
            &|_| {},
        )
        .unwrap_err()
    });
    assert!(err.is::<Cancelled>(), "{err:#}");
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "took {:?}",
        started.elapsed()
    );
}

/// The richer entry point returns what `.psmc` output needs: the EM history
/// and the counts of the resampled data, alongside the same models.
#[test]
fn replicates_carry_their_history_and_data_counts() {
    let seqs = reference_seqs(3000);
    let template = fitted_template();
    let cfg = config(3, 1000, 2);
    let models = bootstrap(&seqs, &template, &cfg).unwrap();
    let reps = run_bootstrap_with_options(
        &seqs,
        &template,
        &cfg,
        &mstep(),
        BootstrapOptions::default(),
    )
    .unwrap();
    assert_eq!(reps.len(), 3);
    let chunks = split_for_bootstrap(&seqs, cfg.trunk_size);
    for (rep, r) in reps.iter().enumerate() {
        assert_eq!(
            param_bits(&r.model),
            param_bits(&models[rep]),
            "replicate {rep}"
        );
        assert_eq!(r.history.loglike.len(), 2);
        assert_eq!(r.history.params.len(), 3);
        assert_eq!(r.history.params[0][2], cfg.t_max, "fresh start");

        let bins = resample_chunks(&chunks, &mut replicate_rng(cfg.seed, rep));
        let count = |f: fn(u8) -> bool| bins.iter().flatten().filter(|&&b| f(b)).count() as u64;
        assert_eq!(r.data.n_seqs, bins.len());
        assert_eq!(r.data.bins, count(|_| true));
        assert_eq!(r.data.sum_l, count(|b| b < 2));
        assert_eq!(r.data.sum_n, count(|b| b == 1));

        let mut out = Vec::new();
        r.write_psmc(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let header = format!(
            "MM\tn_seqs:{}, sum_L:{}, sum_n:{}",
            r.data.n_seqs, r.data.sum_l, r.data.sum_n
        );
        assert!(text.contains(&header), "{text}");
        let rounds: Vec<&str> = text.lines().filter(|l| l.starts_with("RD\t")).collect();
        assert_eq!(rounds, ["RD\t0", "RD\t1", "RD\t2"]);
    }
}

/// With `start`, every replicate starts from those parameters, as C psmc
/// -i -b does, rather than from a fresh start.
#[test]
fn replicates_can_start_from_saved_parameters() {
    let seqs = reference_seqs(3000);
    let template = fitted_template();
    let cfg = config(3, 1000, 1);
    let from = |start: Option<&PsmcModel>| {
        let opts = BootstrapOptions {
            start,
            ..BootstrapOptions::default()
        };
        run_bootstrap_with_options(&seqs, &template, &cfg, &mstep(), opts)
    };
    let start = params(&template);
    for (rep, r) in from(Some(&template)).unwrap().iter().enumerate() {
        assert_eq!(r.history.params[0], start, "replicate {rep}");
        assert_ne!(
            param_bits(&r.model),
            param_bits(&template),
            "replicate {rep} ran EM"
        );
    }
    // The fresh start of the same replicates differs.
    assert_ne!(from(None).unwrap()[0].history.params[0], start);

    // The start must have the template's layout.
    let other = PsmcModel::new(15.0, 10, 0.01, 0.002, 2.5e-8, None).unwrap();
    let err = from(Some(&other)).unwrap_err();
    assert!(err.to_string().contains("n_steps 10"), "{err}");
}

/// `thread_pool` runs the replicates (and their callbacks) in that pool.
#[test]
fn replicates_run_in_the_given_pool() {
    let pool = pool(3);
    let sizes = Mutex::new(Vec::new());
    let on_done = |_: usize| sizes.lock().unwrap().push(rayon::current_num_threads());
    let opts = BootstrapOptions {
        thread_pool: Some(&pool),
        on_done: Some(&on_done),
        ..BootstrapOptions::default()
    };
    let (seqs, template) = (reference_seqs(3000), fitted_template());
    run_bootstrap_with_options(&seqs, &template, &config(4, 1000, 0), &mstep(), opts).unwrap();
    assert_eq!(sizes.into_inner().unwrap(), vec![3; 4]);
}

#[test]
fn invalid_settings_are_rejected() {
    let seqs = reference_seqs(3000);
    let t = fitted_template();
    let base = config(2, 1000, 1);
    for cfg in [
        config(2, 0, 1),
        BootstrapConfig {
            tr_ratio: 0.0,
            ..base.clone()
        },
        BootstrapConfig {
            t_max: f64::NAN,
            ..base.clone()
        },
        BootstrapConfig {
            batch_size: Some(0),
            ..base.clone()
        },
    ] {
        assert!(bootstrap(&seqs, &t, &cfg).is_err(), "{cfg:?}");
    }
    assert!(
        bootstrap(&seqs, &t, &config(0, 1000, 1))
            .unwrap()
            .is_empty()
    );
    assert!(bootstrap(&[], &t, &base).is_err());
}

/// `batch_size` cuts each trunk into rows of one HMM chain, which the
/// E-step runs as one sequence in double precision, so the fits are the
/// same to the bit even after 10 EM iterations. Starting a new chain at
/// every 300-bin row instead moves them by about 0.35.
#[test]
fn batch_size_keeps_each_trunk_one_chain() {
    let seqs = reference_seqs(4000);
    let t = fitted_template();
    let plain = bootstrap(&seqs, &t, &config(2, 1000, 10)).unwrap();
    let cfg = BootstrapConfig {
        batch_size: Some(300),
        ..config(2, 1000, 10)
    };
    let batched = bootstrap(&seqs, &t, &cfg).unwrap();
    for (rep, (a, b)) in plain.iter().zip(&batched).enumerate() {
        assert_eq!(param_bits(a), param_bits(b), "replicate {rep}");
    }
}

/// Two sequences of ten 1000-bin trunks each, alternately 38% and 58%
/// heterozygous (48% in all), so that resamples differ in their θ0.
fn uneven_seqs() -> Vec<Vec<u8>> {
    let trunk = |het_per_50: usize| (0..1000).map(move |i| u8::from(i % 50 < het_per_50));
    let seq: Vec<u8> = (0..10)
        .flat_map(|t| trunk(if t % 2 == 0 { 19 } else { 29 }))
        .collect();
    vec![seq.clone(), seq]
}

/// A fresh start just below the limit on ρ for the whole input leaves
/// some replicates, whose resampled data are more heterozygous, past it.
/// `invalid_replicate_starts` finds exactly those before anything is
/// fitted; `skip_invalid_starts` leaves them out, and without it they stop
/// the bootstrap with EM's error.
#[test]
fn replicates_whose_fresh_start_is_invalid_are_found_and_can_be_skipped() {
    let seqs = uneven_seqs();
    let template = fitted_template();
    let theta0 = initial_theta(&seqs).unwrap();
    let cfg = BootstrapConfig {
        tr_ratio: theta0 / (0.995 * max_rho(theta0)),
        ..config(8, 1000, 1)
    };
    // The main start is valid.
    replicate_initial_model(&template, &seqs, cfg.t_max, cfg.tr_ratio)
        .unwrap()
        .check_transitions()
        .unwrap();

    let invalid = invalid_replicate_starts(&seqs, &template, &cfg).unwrap();
    let chunks = split_for_bootstrap(&seqs, cfg.trunk_size);
    let expected: Vec<(usize, f64, f64)> = (0..cfg.replicates)
        .filter_map(|rep| {
            let rows = resample_chunks(&chunks, &mut replicate_rng(cfg.seed, rep));
            let start = replicate_initial_model(&template, &rows, cfg.t_max, cfg.tr_ratio).unwrap();
            start
                .check_transitions()
                .is_err()
                .then_some((rep, start.theta, start.rho))
        })
        .collect();
    let found: Vec<(usize, f64, f64)> = invalid
        .iter()
        .map(|r| (r.index, r.start.theta, r.start.rho))
        .collect();
    assert_eq!(found, expected);
    assert!(
        !invalid.is_empty() && invalid.len() < cfg.replicates,
        "{found:?}"
    );
    for r in &invalid {
        assert!(r.start.rho > r.start.max_rho.unwrap());
        assert!(r.start.min_tr_ratio().unwrap() > cfg.tr_ratio);
    }

    let dir = TempDir::new("skip_invalid");
    let done = Mutex::new(Vec::new());
    let on_done = |n: usize| done.lock().unwrap().push(n);
    let opts = BootstrapOptions {
        on_done: Some(&on_done),
        skip_invalid_starts: true,
        ..BootstrapOptions::default()
    };
    let cfg_files = BootstrapConfig {
        out_dir: Some(dir.path("")),
        ..cfg.clone()
    };
    let reps = run_bootstrap_with_options(&seqs, &template, &cfg_files, &mstep(), opts).unwrap();
    let fitted: Vec<usize> = reps.iter().map(|r| r.index).collect();
    let skipped: Vec<usize> = invalid.iter().map(|r| r.index).collect();
    let kept: Vec<usize> = (0..cfg.replicates)
        .filter(|i| !skipped.contains(i))
        .collect();
    assert_eq!(fitted, kept);
    // Skipped replicates count as done, and have no file.
    assert_eq!(
        *done.lock().unwrap(),
        (1..=cfg.replicates).collect::<Vec<_>>()
    );
    for i in 0..cfg.replicates {
        let file = dir.path(&format!("replicate_{:03}.json", i + 1));
        assert_eq!(file.exists(), fitted.contains(&i), "replicate {}", i + 1);
    }

    let err = bootstrap(&seqs, &template, &cfg).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.starts_with("bootstrap replicate ")
            && message.contains("EM cannot start from these parameters"),
        "{message}"
    );
    assert!(err.downcast_ref::<NegativeTransitions>().is_some());
}

/// Saved starting parameters are the same for every replicate: none is
/// skipped for its data.
#[test]
fn saved_starts_are_never_skipped() {
    let template = fitted_template();
    let opts = BootstrapOptions {
        start: Some(&template),
        skip_invalid_starts: true,
        ..BootstrapOptions::default()
    };
    let cfg = config(3, 1000, 1);
    let reps = run_bootstrap_with_options(&uneven_seqs(), &template, &cfg, &mstep(), opts).unwrap();
    assert_eq!(reps.iter().map(|r| r.index).collect::<Vec<_>>(), [0, 1, 2]);
}
