//! The numbers quoted in the rationale of the QC checks (rustdoc of
//! `psmc_rs::qc::QcCode`), recomputed from the model and the readers.

use psmc_rs::PsmcModel;
use psmc_rs::io::common::Binner;
use psmc_rs::io::vcf::{VcfOptions, read_vcf_with};
use psmc_rs::qc::{
    HEMIZYGOUS_MIN_EXPECTED_HET_BINS, QcContext, QcInput, REFERENCE_MIN_CALLED_BASES, analyze,
    initial_recombinations, theta0_per_bin,
};
use std::fs;

/// Share of the segments (C's sigma, the stationary state distribution) in
/// each parameter group of the default pattern 4+25*2+4+6, at C psmc's
/// starting model (λ = 1, t_max = 15, θ/ρ = 4).
fn default_pattern_group_shares(theta: f64) -> Vec<f64> {
    let model = PsmcModel::new(
        15.0,
        63,
        theta,
        theta / 4.0,
        2.5e-8,
        Some("4+25*2+4+6".to_string()),
    )
    .unwrap();
    let sigma = model.prior_matrix();
    let sizes: Vec<usize> = [4].into_iter().chain([2; 25]).chain([4, 6]).collect();
    assert_eq!(sizes.iter().sum::<usize>(), sigma.len());
    let mut start = 0;
    sizes
        .iter()
        .map(|&n| {
            let share = sigma[start..start + n].iter().sum();
            start += n;
            share
        })
        .collect()
}

/// Groups that get at least 10 of `n_recomb` recombinations.
fn groups_reaching_10(shares: &[f64], n_recomb: f64) -> usize {
    shares.iter().filter(|&&s| n_recomb * s >= 10.0).count()
}

#[test]
fn few_hets_and_too_short_figures() {
    // Human-like 1e-3 per bp at 100 bp bins.
    let theta = 0.1;
    let shares = default_pattern_group_shares(theta);
    assert_eq!(shares.len(), 28);
    // "the most recent group gets 3.6% and the oldest 0.005%".
    assert!((0.0355..0.0375).contains(&shares[0]), "{}", shares[0]);
    assert!((4e-5..5.5e-5).contains(&shares[27]), "{}", shares[27]);

    // "1,000 K bins mean roughly 250 recombinations ... only 13 of the 28
    // groups reach 10": with one het site per K bin, 1,000 / theta called bins.
    let called = (1000.0 / -(-theta).exp_m1()) as u64;
    let n = initial_recombinations(called, theta, 4.0);
    assert!((240.0..260.0).contains(&n), "{n}");
    assert_eq!(groups_reaching_10(&shares, n), 13);

    // "100 Mb at human heterozygosity (1e5 het sites, about 25,000
    // recombinations) give every group but the oldest at least 10".
    let n = initial_recombinations(1_000_000, theta, 4.0);
    assert!((24_000.0..25_500.0).contains(&n), "{n}");
    assert_eq!(groups_reaching_10(&shares, n), 27);
    assert!(n * shares[27] < 10.0);
    // "at 30 Mb the second-oldest group gets about 5".
    let n = initial_recombinations(300_000, theta, 4.0);
    assert!((4.5..5.5).contains(&(n * shares[26])), "{}", n * shares[26]);
}

#[test]
fn missing_sites_inside_called_bins_lower_theta() {
    // HIGH_MISSING: a bin is called with up to 90% of its sites missing, and
    // a partly called bin is less often K. Same sites, 60% of them made
    // missing at random: theta0 of the called bins drops.
    let mut state = 42u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut full, mut holey) = (Binner::new(100), Binner::new(100));
    for _ in 0..2_000_000 {
        let het = uniform() < 0.002;
        if het {
            full.add_het();
        } else {
            full.add_called(1);
        }
        if uniform() < 0.6 {
            holey.add_missing(1);
        } else if het {
            holey.add_het();
        } else {
            holey.add_called(1);
        }
    }
    let theta = |bins: Vec<u8>| {
        let rows = vec![bins];
        analyze(&QcInput {
            rows: &rows,
            row_starts: &[true],
            seqs: &[],
            bin_size: 100,
            bootstrap_trunk_bins: 0,
            context: QcContext::default(),
        })
        .totals
        .theta0_per_bin
        .unwrap()
    };
    let (a, b) = (theta(full.finish()), theta(holey.finish()));
    // About 0.2 vs 0.08 per bin.
    assert!(b < 0.5 * a, "{a} vs {b}");
}

#[test]
fn variants_only_vcf_without_mask_lowers_theta() {
    // VARIANTS_ONLY_NO_MASK: the same calls read with and without the
    // callable mask that says only half of the chromosome was callable.
    let dir = std::env::temp_dir().join(format!("psmc_qc_rationale_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let vcf = dir.join("calls.vcf");
    let mut text = String::from(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=200000>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
    );
    // A het SNP every 400 bp in the callable first half.
    for pos in (200..100_000).step_by(400) {
        text.push_str(&format!("chr1\t{pos}\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"));
    }
    fs::write(&vcf, text).unwrap();
    let mask = dir.join("callable.bed");
    fs::write(&mask, "chr1\t0\t100000\n").unwrap();

    let theta = |masks: Vec<std::path::PathBuf>| {
        let obs = read_vcf_with(
            &vcf,
            &VcfOptions {
                masks,
                ..VcfOptions::default()
            },
        )
        .unwrap()
        .obs;
        let r = analyze(&QcInput {
            rows: &obs.rows,
            row_starts: &obs.row_starts,
            seqs: &[],
            bin_size: 100,
            bootstrap_trunk_bins: 0,
            context: QcContext::default(),
        });
        (r.totals.theta0_per_bin.unwrap(), r.totals.missing_fraction)
    };
    let (masked, masked_missing) = theta(vec![mask]);
    let (unmasked, unmasked_missing) = theta(vec![]);
    assert_eq!((masked_missing, unmasked_missing), (0.5, 0.0));
    // 250 K of 1,000 called bins against 250 K of 2,000.
    assert_eq!(masked, theta0_per_bin(750, 250).unwrap());
    assert_eq!(unmasked, theta0_per_bin(1750, 250).unwrap());
    assert!(unmasked < 0.5 * masked);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn small_figures_in_the_constant_docs() {
    // HEMIZYGOUS_MIN_EXPECTED_HET_BINS: P(X <= 2) for X ~ Poisson(10) is 0.3%.
    let l = HEMIZYGOUS_MIN_EXPECTED_HET_BINS;
    let p = (-l).exp() * (1.0 + l + l * l / 2.0);
    assert!((0.0025..0.0030).contains(&p), "{p}");
    // REFERENCE_MIN_CALLED_BASES: 1 Mb at 5% K bins is 500 K bins, whose
    // Poisson SD is 4.5% of the count.
    let k = REFERENCE_MIN_CALLED_BASES as f64 / 100.0 * 0.05;
    assert_eq!(k, 500.0);
    assert!((k.sqrt() / k - 0.045).abs() < 0.001);
    // DEPTH: both alleles among 5 reads with probability 1 - 2^-4 = 94%.
    assert_eq!(1.0 - 2f64.powi(1 - 5), 0.9375);
}
