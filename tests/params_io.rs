use psmc_rs::io::params::{PsmcParamsFile, load_params, save_params};
use std::fs;
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
fn params_json_roundtrip() {
    let path = unique_temp_path("psmc_params", "json");
    let params = PsmcParamsFile {
        theta: 0.001,
        rho: 0.0002,
        lam: vec![1.0, 1.2, 0.9],
        t_max: 15.0,
        n_steps: 4,
        mu: 2.5e-8,
        pattern: Some("2*2".to_string()),
    };

    save_params(&path, &params).expect("failed to save params");
    let loaded = load_params(&path).expect("failed to load params");

    assert!((loaded.theta - params.theta).abs() < 1e-12);
    assert!((loaded.rho - params.rho).abs() < 1e-12);
    assert_eq!(loaded.lam, params.lam);
    assert!((loaded.t_max - params.t_max).abs() < 1e-12);
    assert_eq!(loaded.n_steps, params.n_steps);
    assert!((loaded.mu - params.mu).abs() < 1e-15);
    assert_eq!(loaded.pattern, params.pattern);

    let _ = fs::remove_file(path);
}
