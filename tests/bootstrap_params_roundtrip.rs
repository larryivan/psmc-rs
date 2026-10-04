//! Bootstrap replicate files (`replicate_NNN.json`) must reload to exactly the
//! fitted parameters, so summaries rebuilt from the files match the run.

use psmc_rs::PsmcModel;
use psmc_rs::io::params::load_params;
use std::path::PathBuf;

#[test]
fn saved_params_reload_bit_for_bit() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("bootstrap_params_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = PsmcModel::new(15.0, 22, 0.01, 0.002, 2.5e-8, Some("4+5*3+4".to_string())).unwrap();
    // Values that serde_json's default (fast, inexact) float parser reads back
    // one ulp off.
    m.t_max = 19.320098981185446;
    m.theta = 0.007_283_917_201_550_173;
    m.rho = 0.001_234_567_890_123_456_7;
    for (i, l) in m.lam.iter_mut().enumerate() {
        *l = (1.0 + i as f64).sqrt() * std::f64::consts::PI / 7.0;
    }
    let path = dir.join("replicate_001.json");
    m.save_params(&path).unwrap();
    let back = load_params(&path).unwrap();
    assert_eq!(back.t_max.to_bits(), m.t_max.to_bits(), "t_max");
    assert_eq!(back.theta.to_bits(), m.theta.to_bits(), "theta");
    assert_eq!(back.rho.to_bits(), m.rho.to_bits(), "rho");
    for (a, b) in back.lam.iter().zip(&m.lam) {
        assert_eq!(a.to_bits(), b.to_bits(), "lambda");
    }
}
