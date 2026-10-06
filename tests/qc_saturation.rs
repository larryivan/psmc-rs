//! Bin saturation formulas, checked against exact binomial sums and against
//! brute-force sampling of heterozygous sites.

use psmc_rs::qc::{
    QcContext, QcInput, SATURATION_TARGET_SHARE, Saturation, analyze, het_per_site,
    max_bin_size_for_share, multi_het_share,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// P(X >= 2 | X >= 1) for X ~ Binomial(s, h), from the probability mass
/// function term by term.
fn share_by_pmf(h: f64, s: usize) -> f64 {
    let p0 = (1.0 - h).powi(s as i32);
    // P(X = 1), then P(X = k + 1) from P(X = k).
    let mut pmf = p0 * s as f64 * h / (1.0 - h);
    let mut ge2 = 0.0;
    for k in 1..s {
        pmf *= (s - k) as f64 / (k + 1) as f64 * h / (1.0 - h);
        ge2 += pmf;
    }
    ge2 / (1.0 - p0)
}

#[test]
fn closed_form_matches_the_binomial_sum() {
    for &s in &[2usize, 3, 10, 50, 100, 200, 1000] {
        for &h in &[1e-6, 1e-4, 1e-3, 3e-3, 0.01, 0.05, 0.2, 0.5] {
            let want = share_by_pmf(h, s);
            let got = multi_het_share(h, s);
            assert!(
                (got - want).abs() <= 1e-9 * want.max(1e-6),
                "h {h}, s {s}: {got} vs {want}"
            );
        }
    }
    assert_eq!(multi_het_share(0.1, 1), 0.0);
    assert_eq!(multi_het_share(0.0, 100), 0.0);
    assert_eq!(multi_het_share(1.0, 2), 1.0);
    // Small h: about (s - 1) h / 2.
    let f = multi_het_share(1e-5, 100);
    assert!((f / 4.95e-4 - 1.0).abs() < 1e-3, "{f}");
}

#[test]
fn het_per_site_inverts_the_bin_probability() {
    for &s in &[1usize, 10, 100, 1000] {
        for &h in &[1e-7f64, 1e-4, 1e-3, 0.02, 0.3] {
            let all_hom = (1.0 - h).powi(s as i32);
            if all_hom < 1e-6 {
                // p rounds too close to 1 to carry h.
                continue;
            }
            let got = het_per_site(1.0 - all_hom, s);
            assert!((got / h - 1.0).abs() < 1e-9, "h {h}, s {s}: {got}");
        }
    }
    assert_eq!(het_per_site(0.0, 100), 0.0);
    assert_eq!(het_per_site(1.0, 100), 1.0);
}

/// Sample `n` bins of `s` sites, each site heterozygous with probability
/// drawn per bin from `rate`; return (K bins, K bins with >= 2 het sites).
fn sample_bins(
    n: usize,
    s: usize,
    rng: &mut ChaCha8Rng,
    rate: impl Fn(usize) -> f64,
) -> (u64, u64) {
    let (mut k, mut multi) = (0u64, 0u64);
    for bin in 0..n {
        let h = rate(bin);
        let hets = (0..s).filter(|_| rng.r#gen::<f64>() < h).count();
        k += u64::from(hets >= 1);
        multi += u64::from(hets >= 2);
    }
    (k, multi)
}

#[test]
fn formulas_match_brute_force_sampling() {
    let mut rng = ChaCha8Rng::seed_from_u64(20261005);
    let n = 100_000usize;
    for &(h, s) in &[
        (0.001, 100usize),
        (0.004, 100),
        (0.01, 50),
        (0.05, 20),
        (0.2, 5),
    ] {
        let (k, multi) = sample_bins(n, s, &mut rng, |_| h);
        let p = k as f64 / n as f64;

        // h recovered from the K fraction, within 4 standard errors (delta
        // method: dh/dp = (1-p)^(1/s - 1) / s).
        let h_hat = het_per_site(p, s);
        let se_p = (p * (1.0 - p) / n as f64).sqrt();
        let se_h = se_p * (1.0 - p).powf(1.0 / s as f64 - 1.0) / s as f64;
        assert!(
            (h_hat - h).abs() < 4.0 * se_h,
            "h {h}, s {s}: estimated {h_hat}"
        );

        // The observed share of multi-het K bins, against the formula at the
        // true h and at the h estimated from the bins.
        let observed = multi as f64 / k as f64;
        let se = (observed * (1.0 - observed) / k as f64).sqrt();
        let at_true = multi_het_share(h, s);
        let at_estimate = Saturation::from_het_fraction(p, s).multi_het_share;
        for (what, f) in [("true h", at_true), ("estimated h", at_estimate)] {
            assert!(
                (observed - f).abs() < 4.0 * se + 1e-4,
                "h {h}, s {s}: observed {observed}, formula at {what} {f}"
            );
        }
    }
}

#[test]
fn varying_heterozygosity_only_raises_the_share() {
    // Exact mixtures: the formula applied to the mixture's K fraction never
    // exceeds the mixture's own share (convexity, see qc::Saturation).
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for _ in 0..2000 {
        let s = [2usize, 10, 100, 500][rng.gen_range(0..4)];
        let parts = rng.gen_range(2..6);
        let hs: Vec<f64> = (0..parts)
            .map(|_| 10f64.powf(rng.gen_range(-5.0..-0.5)))
            .collect();
        let ws: Vec<f64> = (0..parts).map(|_| rng.r#gen::<f64>() + 0.01).collect();
        let total: f64 = ws.iter().sum();
        let (mut u, mut ge2) = (0.0, 0.0);
        for (&h, &w) in hs.iter().zip(&ws) {
            let any = 1.0 - (1.0 - h).powi(s as i32);
            u += w / total * any;
            ge2 += w / total * any * multi_het_share(h, s);
        }
        let mixture = ge2 / u;
        let formula = multi_het_share(het_per_site(u, s), s);
        assert!(
            mixture >= formula - 1e-12,
            "{hs:?} {ws:?} s {s}: {mixture} < {formula}"
        );
    }

    // Sampled: half the bins at 0.0005 and half at 0.004 per site.
    let n = 200_000;
    let (k, multi) = sample_bins(
        n,
        100,
        &mut rng,
        |bin| if bin % 2 == 0 { 0.0005 } else { 0.004 },
    );
    let observed = multi as f64 / k as f64;
    let se = (observed * (1.0 - observed) / k as f64).sqrt();
    let formula = Saturation::from_het_fraction(k as f64 / n as f64, 100).multi_het_share;
    assert!(
        observed > formula + 4.0 * se,
        "observed {observed}, formula {formula}"
    );
}

#[test]
fn share_grows_with_the_bin_size() {
    for &h in &[1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9] {
        let mut prev = 0.0;
        for s in 1..=2000 {
            let f = multi_het_share(h, s);
            assert!(f >= prev, "h {h}: share falls from {prev} to {f} at s {s}");
            prev = f;
        }
    }
}

#[test]
fn recommended_bin_size_is_the_largest_below_target() {
    for &h in &[1e-7, 1e-5, 1e-4, 1e-3, 2.5e-3, 0.01, 0.03, 0.1, 0.5, 1.0] {
        let r = max_bin_size_for_share(h, SATURATION_TARGET_SHARE).unwrap();
        assert!(
            multi_het_share(h, r) < SATURATION_TARGET_SHARE,
            "h {h}: {r}"
        );
        assert!(
            multi_het_share(h, r + 1) >= SATURATION_TARGET_SHARE,
            "h {h}: {r}"
        );
    }
    assert_eq!(max_bin_size_for_share(1.0, 0.05), Some(1));
    assert_eq!(max_bin_size_for_share(0.0, 0.05), None);
    assert_eq!(max_bin_size_for_share(1e-18, 0.05), None);
}

#[test]
fn human_data_at_100_bp_sits_at_the_target() {
    // The level that SATURATION_TARGET_SHARE is taken from.
    let f = multi_het_share(0.001, 100);
    assert!((0.0485..0.0490).contains(&f), "{f}");
    assert!(max_bin_size_for_share(0.001, SATURATION_TARGET_SHARE).unwrap() >= 100);
}

#[test]
fn report_carries_the_saturation_of_the_whole_input() {
    // 900 T, 100 K, 50 N over two sequences.
    let mut a = vec![0u8; 500];
    a[..60].fill(1);
    a[400..450].fill(2);
    let mut b = vec![0u8; 550];
    b[..40].fill(1);
    let rows = vec![a, b];
    let input = QcInput {
        rows: &rows,
        row_starts: &[true, true],
        seqs: &[],
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context: QcContext::default(),
        callable: None,
    };
    let s = analyze(&input).saturation.unwrap();
    let h = 1.0 - 0.9f64.powf(0.01);
    assert!((s.het_per_site / h - 1.0).abs() < 1e-12);
    assert!((s.multi_het_share - multi_het_share(h, 100)).abs() < 1e-15);
    assert_eq!(
        s.max_bin_size_below_target,
        max_bin_size_for_share(h, SATURATION_TARGET_SHARE)
    );

    let rows = vec![vec![2u8; 10]];
    let input = QcInput {
        rows: &rows,
        row_starts: &[true],
        ..input
    };
    assert_eq!(analyze(&input).saturation, None);
}
