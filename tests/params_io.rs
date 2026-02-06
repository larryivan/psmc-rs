use psmc_rs::io::params::{PsmcParamsFile, load_params, save_params};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before unix epoch")
        .as_nanos();
    path.push(format!("{prefix}_{}_{}.{}", std::process::id(), nanos, ext));
    path
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
