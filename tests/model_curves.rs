//! Scaling helpers: N0, the Ne(t) curve and per-state TMRCA in years.

mod common;

use psmc_rs::PsmcModel;

fn fitted_like_model() -> PsmcModel {
    // θ = 0.0072 per 100-bp bin and μ = 2.5e-8 give N0 = 720.
    let mut model = common::model("4+5*3+4", 0.0072, 0.0015);
    model.lam = vec![2.0, 0.5, 0.8, 1.6, 3.0, 6.0, 9.0];
    model.param_recalculate().unwrap();
    model
}

#[test]
fn n0_for_bin_divides_theta_by_four_mu_and_bin_size() {
    let model = fitted_like_model();
    assert!((model.n0_for_bin(100.0) - 720.0).abs() < 1e-9);
    assert_eq!(
        model.n0_for_bin(100.0),
        model.theta / (4.0 * model.mu * 100.0)
    );
    #[allow(deprecated)]
    let per_bp = model.n0();
    assert_eq!(per_bp, model.n0_for_bin(1.0));
}

#[test]
fn ne_curve_follows_psmc_plot_scaling() {
    let model = fitted_like_model();
    let (bin, gen_years) = (100.0, 25.0);
    let n0 = model.n0_for_bin(bin);
    let curve = model.ne_curve(bin, gen_years).unwrap();
    let t = model.compute_t();
    let lam = model.map_lam(&model.lam).unwrap();

    assert_eq!(curve.len(), model.n_steps + 1);
    assert_eq!(curve[0], [0.0, 2.0 * n0]);
    for (k, [x, y]) in curve.iter().enumerate() {
        // The report, bootstrap summary and TMRCA output use exactly this.
        assert_eq!(*x, t[k] * 2.0 * n0 * gen_years);
        assert_eq!(*y, lam[k] * n0);
    }
    for w in curve.windows(2) {
        assert!(w[1][0] > w[0][0], "x must increase: {w:?}");
    }
    // Last point: t_max * 2 N0 * g = 15 * 2 * 720 * 25 years, size 9 N0.
    let last = curve[model.n_steps];
    assert!((last[0] - 540_000.0).abs() < 1e-6);
    assert!((last[1] - 6480.0).abs() < 1e-9);
    // Grouped λ: the three states of the second group share one size.
    assert_eq!(curve[4][1], curve[5][1]);
    assert_eq!(curve[5][1], curve[6][1]);
}

#[test]
fn ne_curve_rejects_meaningless_scales() {
    let model = fitted_like_model();
    for (bin, gen_years) in [
        (0.0, 25.0),
        (-100.0, 25.0),
        (f64::NAN, 25.0),
        (100.0, 0.0),
        (100.0, f64::INFINITY),
    ] {
        assert!(
            model.ne_curve(bin, gen_years).is_err(),
            "bin {bin}, gen {gen_years} should be rejected"
        );
    }
    let mut zero_mu = model.clone();
    zero_mu.mu = 0.0;
    assert!(zero_mu.ne_curve(100.0, 25.0).is_err());
}

#[test]
fn ne_curve_reports_a_wrong_lam_length_instead_of_panicking() {
    // Without a pattern, map_lam used to pass any length through.
    let mut model = PsmcModel::new(15.0, 8, 0.0072, 0.0015, 2.5e-8, None).unwrap();
    model.lam.pop();
    assert!(model.map_lam(&model.lam).is_err());
    assert!(model.ne_curve(100.0, 25.0).is_err());
    model.lam.extend([1.0, 1.0]);
    assert!(model.ne_curve(100.0, 25.0).is_err());
}

#[test]
fn state_tmrca_years_scales_avg_t_and_stays_inside_each_interval() {
    let model = fitted_like_model();
    let (bin, gen_years) = (100.0, 25.0);
    let n0 = model.n0_for_bin(bin);
    let years = model.state_tmrca_years(bin, gen_years);
    let avg = model.avg_t();
    let curve = model.ne_curve(bin, gen_years).unwrap();

    assert_eq!(years.len(), model.n_steps + 1);
    for (k, y) in years.iter().enumerate() {
        assert_eq!(*y, avg[k] * 2.0 * n0 * gen_years);
        assert!(*y >= curve[k][0], "state {k}: {y} before its interval");
        if k < model.n_steps {
            assert!(*y <= curve[k + 1][0], "state {k}: {y} after its interval");
        }
    }
}
