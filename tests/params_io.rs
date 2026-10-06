use psmc_rs::PsmcModel;
use psmc_rs::io::CallableSummary;
use psmc_rs::io::input::{InputFormat, InputOptions, SeqFilter, load};
use psmc_rs::io::params::{
    BinCallableRecord, CALLABLE_LENGTH_MODEL, InputSummary, PsmcParamsFile, RunMetadata,
    load_params, save_params,
};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_temp_path(prefix: &str, ext: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
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
        run: RunMetadata::default(),
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
    assert_eq!(loaded.run, RunMetadata::default());

    // Without metadata the file holds the fields of psmc-rs 0.1 only.
    let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["lam", "mu", "n_steps", "pattern", "rho", "t_max", "theta"]
    );

    let _ = fs::remove_file(path);
}

#[test]
fn run_metadata_round_trips_at_the_top_level() {
    let path = unique_temp_path("psmc_params_meta", "json");
    let params = PsmcParamsFile {
        run: RunMetadata {
            version: Some("0.2.0".to_string()),
            n_iter: Some(2),
            final_loglike: Some(-1234.5678901234567),
            loglike: Some(vec![-1300.25, -1240.0, -1234.5678901234567]),
            bin_size: Some(50),
            gen_years: Some(29.0),
            tr_ratio: Some(4.0),
            smooth_lambda: Some(0.0),
            input: Some(InputSummary {
                format: "vcf".to_string(),
                vcf_mode: Some("consensus".to_string()),
                sequences: 1,
                bins: 230,
                het_bins: 7,
                missing_bins: 12,
                dropped_sequences: 2,
            }),
            qc: None,
            bin_callable: Some(BinCallableRecord {
                mode: "auto".to_string(),
                model: CALLABLE_LENGTH_MODEL.to_string(),
                counts: Some("exact".to_string()),
                callable: Some(CallableSummary {
                    bin_size: 50,
                    called_bins: 218,
                    partial_bins: 31,
                    mean_fraction: 0.9612,
                    histogram: [1, 0, 2, 3, 0, 5, 7, 6, 7, 187],
                }),
            }),
        },
        ..PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, Some("4+5*3+4".into()))
            .unwrap()
            .params_file()
    };
    save_params(&path, &params).unwrap();
    let loaded = load_params(&path).unwrap();
    assert_eq!(loaded, params);

    let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(v["final_loglike"], -1234.5678901234567);
    assert_eq!(v["n_iter"], 2);
    assert_eq!(v["input"]["vcf_mode"], "consensus");
    assert_eq!(v["bin_callable"]["model"], "callable-length");
    assert_eq!(v["bin_callable"]["callable"]["partial_bins"], 31);
    assert_eq!(
        params
            .run
            .bin_callable
            .as_ref()
            .unwrap()
            .psmc_line()
            .unwrap(),
        "bin_model:callable-length, counts:exact, mean_callable:0.961200, partial_bins:0.142202"
    );
    assert!(v.get("qc").is_none(), "unknown fields are left out");
    let _ = fs::remove_file(path);
}

/// A file written by psmc-rs 0.1 (no metadata) still loads.
#[test]
fn files_of_version_0_1_load() {
    let text = r#"{
  "theta": 0.0068,
  "rho": 0.0014,
  "lam": [1.5, 0.7, 2.0, 3.0, 1.0, 0.5, 4.0],
  "t_max": 14.2,
  "n_steps": 22,
  "mu": 2.5e-8,
  "pattern": "4+5*3+4"
}"#;
    let path = unique_temp_path("psmc_params_v01", "json");
    fs::write(&path, text).unwrap();
    let p = load_params(&path).unwrap();
    let _ = fs::remove_file(path);
    assert_eq!(p.run, RunMetadata::default());
    let m = PsmcModel::from_params_file(&p).unwrap();
    assert_eq!(m.lam, p.lam);
    assert_eq!((m.theta, m.rho, m.t_max), (0.0068, 0.0014, 14.2));
}

#[test]
fn a_saved_model_rebuilds_exactly() {
    let mut m = PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, Some("4+5*3+4".into())).unwrap();
    m.theta = 0.0072;
    m.rho = 0.0013;
    m.t_max = 17.25;
    m.lam = vec![2.0, 0.5, 0.8, 1.6, 3.0, 6.0, 9.0];
    m.param_recalculate().unwrap();

    let back = PsmcModel::from_params_file(&m.params_file()).unwrap();
    assert_eq!(back.lam, m.lam);
    assert_eq!(back.sigma, m.sigma);
    assert_eq!(back.p_kl, m.p_kl);
    assert_eq!(back.em, m.em);

    // Without a pattern every state has its own lambda.
    let free = PsmcModel::new(15.0, 5, 0.01, 0.0025, 2.5e-8, None).unwrap();
    let back = PsmcModel::from_params_file(&free.params_file()).unwrap();
    assert_eq!(back.lam.len(), 6);
}

#[test]
fn invalid_saved_models_are_rejected() {
    let good = PsmcModel::new(15.0, 22, 0.01, 0.0025, 2.5e-8, Some("4+5*3+4".into()))
        .unwrap()
        .params_file();
    let err = |p: PsmcParamsFile| {
        PsmcModel::from_params_file(&p)
            .map(|_| ())
            .unwrap_err()
            .to_string()
    };
    let msg = err(PsmcParamsFile {
        lam: vec![1.0; 6],
        ..good.clone()
    });
    assert!(
        msg.contains("6 lambda values") && msg.contains("7 free"),
        "{msg}"
    );
    assert!(
        err(PsmcParamsFile {
            n_steps: 30,
            ..good.clone()
        })
        .contains("pattern implies n_steps=22")
    );
    assert!(
        err(PsmcParamsFile {
            theta: -1.0,
            ..good.clone()
        })
        .contains("theta")
    );
    let mut bad_lam = good.clone();
    bad_lam.lam[3] = f64::NAN;
    assert!(err(bad_lam).contains("lambda"));
    assert!(
        err(PsmcParamsFile {
            t_max: 2000.0,
            ..good
        })
        .contains("t_max")
    );
}

#[test]
fn input_summary_counts_the_dataset() {
    let path = unique_temp_path("psmc_params_input", "psmcfa");
    fs::write(&path, ">a\nTTKTNNTT\n>empty\n>b\nKTTT\n").unwrap();
    let opts = InputOptions {
        format: Some(InputFormat::Psmcfa),
        filter: SeqFilter::DISABLED,
        ..InputOptions::default()
    };
    let data = load(&path, &opts).unwrap();
    let _ = fs::remove_file(path);
    let s = InputSummary::of(&data);
    assert_eq!(
        s,
        InputSummary {
            format: "psmcfa".to_string(),
            vcf_mode: None,
            sequences: 2,
            bins: 12,
            het_bins: 2,
            missing_bins: 2,
            dropped_sequences: 1,
        }
    );
}
