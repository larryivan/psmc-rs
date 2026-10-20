//! Bootstrap replicate files (`replicate_NNN.json`) must reload to exactly the
//! fitted parameters, so summaries rebuilt from the files match the run.

mod common;

use common::{TempDir, bits, model, params};
use psmc_rs::io::params::load_params;

#[test]
fn saved_params_reload_bit_for_bit() {
    let dir = TempDir::new("bootstrap_params_roundtrip");
    let mut m = model("4+5*3+4", 0.01, 0.002);
    // Values that serde_json's default (fast, inexact) float parser reads back
    // one ulp off.
    m.t_max = 19.320098981185446;
    m.theta = 0.007_283_917_201_550_173;
    m.rho = 0.001_234_567_890_123_456_7;
    for (i, l) in m.lam.iter_mut().enumerate() {
        *l = (1.0 + i as f64).sqrt() * std::f64::consts::PI / 7.0;
    }
    let path = dir.path("replicate_001.json");
    m.save_params(&path).unwrap();
    let back = load_params(&path).unwrap();
    let loaded: Vec<f64> = [back.theta, back.rho, back.t_max]
        .into_iter()
        .chain(back.lam)
        .collect();
    assert_eq!(bits(&loaded), bits(&params(&m)));
}
