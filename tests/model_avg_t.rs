//! `PsmcModel::avg_t`: the per-state average coalescent time of C `psmc_avg_t`.

use psmc_rs::PsmcModel;

/// Deterministic xorshift64 in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (lo.ln() + (hi.ln() - lo.ln()) * self.next()).exp()
    }
}

/// Models with random θ, ρ, t_max and λ over a few state layouts.
fn random_models(seed: u64, per_layout: usize) -> Vec<PsmcModel> {
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for (n, pattern) in [
        (8, None),
        (22, Some("4+5*3+4")),
        (63, Some("4+25*2+4+6")),
        (64, None),
    ] {
        for _ in 0..per_layout {
            let theta = rng.log_uniform(1e-4, 0.1);
            let rho = theta * rng.log_uniform(0.05, 2.0);
            let t_max = 5.0 + 30.0 * rng.next();
            let mut model =
                PsmcModel::new(t_max, n, theta, rho, 2.5e-8, pattern.map(str::to_string)).unwrap();
            for lam in model.lam.iter_mut() {
                *lam = rng.log_uniform(0.03, 30.0);
            }
            model.param_recalculate().unwrap();
            out.push(model);
        }
    }
    out
}

#[test]
fn avg_t_lies_in_its_interval() {
    for model in random_models(0x9E37_79B9_7F4A_7C15, 25) {
        let avg = model.avg_t();
        let n = model.n_steps;
        assert_eq!(avg.len(), n + 1);
        for (k, (a, edges)) in avg.iter().zip(model.t.windows(2)).take(n).enumerate() {
            assert!(
                edges[0] <= *a && *a <= edges[1],
                "avg_t[{k}] = {a} outside {edges:?}"
            );
        }
        assert!(avg[n].is_finite() && avg[n] >= model.t[n]);
    }
}

#[test]
fn emission_is_exp_of_minus_theta_avg_t_bit_for_bit() {
    for model in random_models(0xD1B5_4A32_D192_ED03, 10) {
        let avg = model.avg_t();
        for (k, a) in avg.iter().enumerate() {
            let expected = (-model.theta * a).exp();
            assert_eq!(model.em[(0, k)].to_bits(), expected.to_bits(), "state {k}");
        }
    }
}

/// Mean of the coalescent density restricted to `[lo, hi]` with constant λ,
/// by composite Simpson integration. Earlier intervals only scale the
/// density, so they cancel.
fn conditional_mean_by_quadrature(lo: f64, hi: f64, lam: f64) -> f64 {
    let m = 20_000;
    let h = (hi - lo) / m as f64;
    let (mut mass, mut moment) = (0.0, 0.0);
    for i in 0..=m {
        let s = lo + i as f64 * h;
        let w = match i {
            0 => 1.0,
            _ if i == m => 1.0,
            _ if i % 2 == 1 => 4.0,
            _ => 2.0,
        };
        let f = (-(s - lo) / lam).exp();
        mass += w * f;
        moment += w * s * f;
    }
    moment / mass
}

#[test]
fn avg_t_matches_an_independent_computation() {
    // C's average is -ln(1 - pi_k / (C_sigma sigma_k)) / rho. Substituting
    // the definitions of pi_k and sigma_k, C_pi cancels and the argument is
    // x / (1 + x/2) with x = rho * E_k, E_k the mean coalescent time given
    // coalescence in interval k; hence avg_t = 2 atanh(x / 2) / rho, with E_k
    // itself as the fallback when that leaves the interval.
    for model in random_models(0x2545_F491_4F6C_DD1D, 10) {
        let avg = model.avg_t();
        let t = model.compute_t(0.1);
        let lam = model.map_lam(&model.lam).unwrap();
        let n = model.n_steps;
        for k in 0..=n {
            let mean = if k < n {
                conditional_mean_by_quadrature(t[k], t[k + 1], lam[k])
            } else {
                t[n] + lam[n] // memoryless last interval
            };
            let x = model.rho * mean;
            let mut expected = 2.0 * (x / 2.0).atanh() / model.rho;
            let upper = if k < n { t[k + 1] } else { f64::INFINITY };
            if !(expected >= t[k] && expected <= upper) {
                expected = mean;
            }
            let rel = ((avg[k] - expected) / expected).abs();
            assert!(
                rel < 1e-8,
                "state {k}: avg_t {} vs independent {expected} (rel {rel:.2e}, rho {})",
                avg[k],
                model.rho
            );
        }
    }
}

/// `TC` lines (third column: avg_t * theta) of the original C psmc,
/// 0.6.5-r74, built with -O1:
///
/// ```text
/// printf '4+5*3+4\n1000 0.3 15 0.5 2.0 0.08 1.3 7.5 30.0 0.9\n' > par.txt
/// psmc -N0 -d -i par.txt tiny.psmcfa
/// ```
///
/// θ = 1000 only scales the printed times to keep more digits; ρ = 0.3 is
/// large enough that the log formula fails in the late intervals, so the
/// fallback is exercised too.
const C_TC_AVG: [f64; 23] = [
    12.698578,
    41.532600,
    77.734699,
    123.184945,
    180.752457,
    252.680328,
    343.086963,
    441.387180,
    576.088198,
    743.859790,
    1007.097662,
    1295.942396,
    1665.068617,
    2156.829957,
    2796.583867,
    3677.158737,
    4232.916567,
    5341.963260,
    6734.600979,
    8166.023413,
    10197.286497,
    12716.787785,
    15900.000000,
];

#[test]
fn avg_t_matches_c_psmc_decode_output() {
    let mut model =
        PsmcModel::new(15.0, 22, 1000.0, 0.3, 2.5e-8, Some("4+5*3+4".to_string())).unwrap();
    model.lam = vec![0.5, 2.0, 0.08, 1.3, 7.5, 30.0, 0.9];
    model.param_recalculate().unwrap();
    let avg = model.avg_t();
    for (k, c) in C_TC_AVG.iter().enumerate() {
        let ours = avg[k] * model.theta;
        assert!((ours - c).abs() < 1e-6, "state {k}: {ours:.6} vs C {c:.6}");
    }
}
