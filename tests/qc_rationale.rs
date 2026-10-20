//! The numbers quoted in the rationale of the QC checks (rustdoc of
//! `psmc_rs::qc::QcCode`), recomputed from the model and the readers.

mod common;

use common::TempDir;
use psmc_rs::PsmcModel;
use psmc_rs::io::common::Binner;
use psmc_rs::io::vcf::{VcfOptions, read_vcf_with};
use psmc_rs::qc::{
    HEMIZYGOUS_MIN_EXPECTED_HET_BINS, QcCode, QcContext, QcInput, QcReport, QcSeq,
    REFERENCE_MIN_CALLED_BASES, Severity, analyze, initial_recombinations, theta0_per_bin,
};
use std::fs;

/// The QC report of `rows` and `seqs` in bins of 100 bp, with the default
/// context and no bootstrap.
fn qc(rows: &[Vec<u8>], row_starts: &[bool], seqs: &[QcSeq]) -> QcReport {
    analyze(&QcInput {
        rows,
        row_starts,
        seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context: QcContext::default(),
        callable: None,
    })
}

/// A 64-bit LCG (Knuth's MMIX constants): deterministic and dependency-free.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 33) % n
    }
}

/// 2 Mb with a heterozygous site at 0.2% of the sites, binned twice: whole
/// (`full`), and with 60% of the sites missing at random (`holey`), whose
/// called and heterozygous sites are counted.
struct Holey {
    full: Vec<u8>,
    holey: Vec<u8>,
    called: u64,
    hets: u64,
}

fn holey_sites(seed: u64) -> Holey {
    let mut rng = Lcg(seed);
    let (mut full, mut holey) = (Binner::new(100), Binner::new(100));
    let (mut called, mut hets) = (0, 0);
    for _ in 0..2_000_000 {
        let het = rng.uniform() < 0.002;
        if het {
            full.add_het();
        } else {
            full.add_called(1);
        }
        if rng.uniform() < 0.6 {
            holey.add_missing(1);
        } else {
            if het {
                holey.add_het();
                hets += 1;
            } else {
                holey.add_called(1);
            }
            called += 1;
        }
    }
    Holey {
        full: full.finish(),
        holey: holey.finish(),
        called,
        hets,
    }
}

/// Share of the segments (C's sigma, the stationary state distribution) in
/// each parameter group of `pattern` (groups of `sizes` intervals; one group
/// per interval without a pattern), at C psmc's starting model (λ = 1,
/// t_max = 15, θ/ρ = 4). Also checks that the model has one free λ per group.
fn group_shares(theta: f64, pattern: Option<&str>, n_steps: usize, sizes: &[usize]) -> Vec<f64> {
    let model = PsmcModel::new(
        15.0,
        n_steps,
        theta,
        theta / 4.0,
        2.5e-8,
        pattern.map(str::to_string),
    )
    .unwrap();
    // n_free_params counts θ, ρ and t_max besides the λ.
    assert_eq!(model.n_free_params() - 3, sizes.len(), "{pattern:?}");
    let sigma = model.prior_matrix();
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

/// The pattern psmc's README uses for human data, 4+25*2+4+6 (28 free λ).
fn human_pattern_shares(theta: f64) -> Vec<f64> {
    let sizes: Vec<usize> = [4].into_iter().chain([2; 25]).chain([4, 6]).collect();
    group_shares(theta, Some("4+25*2+4+6"), 63, &sizes)
}

/// Groups that get at least 10 of `n_recomb` recombinations.
fn groups_reaching_10(shares: &[f64], n_recomb: f64) -> usize {
    shares.iter().filter(|&&s| n_recomb * s >= 10.0).count()
}

#[test]
fn few_hets_and_too_short_figures() {
    // Human-like 1e-3 per bp at 100 bp bins.
    let theta = 0.1;
    let shares = human_pattern_shares(theta);
    assert_eq!(shares.len(), 28);
    // "the most recent group gets 3.6% and the oldest 0.005%".
    assert!((0.0355..0.0375).contains(&shares[0]), "{}", shares[0]);
    assert!((4e-5..5.5e-5).contains(&shares[27]), "{}", shares[27]);

    // "1,000 K bins mean roughly 250 recombinations ... only 13 of the 28
    // groups reach 10": with one het site per K bin, 1,000 / theta called bins.
    let called = (1000.0 / -(-theta).exp_m1()) as u64;
    let few = initial_recombinations(called, theta, 4.0);
    assert!((240.0..260.0).contains(&few), "{few}");
    assert_eq!(groups_reaching_10(&shares, few), 13);

    // "100 Mb at human heterozygosity (1e5 het sites, about 25,000
    // recombinations) give every group of 4+25*2+4+6 but the oldest ... at
    // least 10".
    let n = initial_recombinations(1_000_000, theta, 4.0);
    assert!((24_000.0..25_500.0).contains(&n), "{n}");
    assert_eq!(groups_reaching_10(&shares, n), 27);
    assert!(n * shares[27] < 10.0);
    // "at 30 Mb the second-oldest group of 4+25*2+4+6 gets about 5".
    let thirty = initial_recombinations(300_000, theta, 4.0);
    let second_oldest = thirty * shares[26];
    assert!((4.5..5.5).contains(&second_oldest), "{second_oldest}");

    // "With C psmc's default pattern 4+5*3+4 (7 free λ), 5 of the 7 groups
    // reach 10" at 1,000 K bins; "every group of 4+5*3+4" does at 100 Mb.
    let sizes: Vec<usize> = [4].into_iter().chain([3; 5]).chain([4]).collect();
    let c_default = group_shares(theta, Some("4+5*3+4"), 22, &sizes);
    assert_eq!(c_default.len(), 7);
    assert_eq!(groups_reaching_10(&c_default, few), 5);
    assert_eq!(groups_reaching_10(&c_default, n), 7);

    // "with one λ per interval (--n-steps 64 without a pattern: 65 free
    // λ), none does"; "55 of 65 single intervals" reach 10 at 100 Mb.
    let per_interval = group_shares(theta, None, 64, &[1; 65]);
    assert_eq!(groups_reaching_10(&per_interval, few), 0);
    assert_eq!(groups_reaching_10(&per_interval, n), 55);

    // FEW_HETS with QcContext::free_lambdas: "the oldest group usually gets
    // far fewer than that average (less than a tenth of it in each of the
    // three patterns above)".
    for shares in [&shares, &c_default, &per_interval] {
        let average = 1.0 / shares.len() as f64;
        let oldest = *shares.last().unwrap();
        assert!(oldest < 0.1 * average, "{oldest} vs {average}");
    }
}

#[test]
fn missing_sites_inside_called_bins_lower_theta() {
    // HIGH_MISSING: a bin is called with up to 90% of its sites missing, and
    // a partly called bin is less often K. Same sites, 60% of them made
    // missing at random: theta0 of the called bins drops.
    let sites = holey_sites(42);
    let theta = |bins: Vec<u8>| qc(&[bins], &[true], &[]).totals.theta0_per_bin.unwrap();
    let (a, b) = (theta(sites.full), theta(sites.holey));
    // About 0.2 vs 0.08 per bin.
    assert!(b < 0.5 * a, "{a} vs {b}");
}

/// The uncalled-site share and the N-bin share of one sequence of 100 bp
/// bins, given as runs of (sites, missing).
fn missing_shares(runs: impl IntoIterator<Item = (u64, bool)>) -> (f64, f64) {
    let mut binner = Binner::new(100);
    let (mut len, mut uncalled) = (0u64, 0u64);
    for (run, missing) in runs {
        if missing {
            binner.add_missing(run);
            uncalled += run;
        } else {
            binner.add_called(run);
        }
        len += run;
    }
    let rows = vec![binner.finish()];
    let mut seq = QcSeq::new("chr1", rows[0].len());
    seq.extent_bases = Some(len);
    seq.called_bases = Some(len - uncalled);
    let t = qc(&rows, &[true], &[seq]).totals;
    (t.uncalled_site_fraction.unwrap(), t.missing_fraction)
}

#[test]
fn n_bin_share_bounds_the_uncalled_site_share() {
    // HIGH_MISSING: every N bin has more than 90% of its sites uncalled, so
    // 0.9 times the N share is a lower bound on the uncalled share, for
    // clustered and scattered missing data alike.
    let mut rng = Lcg(99);
    for _ in 0..300 {
        let (max_run, missing_pct) = (1 + rng.below(400), rng.below(101));
        let target = 1_000 + rng.below(50_000);
        let mut runs = Vec::new();
        let mut len = 0;
        while len < target {
            let run = 1 + rng.below(max_run);
            runs.push((run, rng.below(100) < missing_pct));
            len += run;
        }
        let (uncalled, n_share) = missing_shares(runs);
        assert!(uncalled >= 0.9 * n_share, "{uncalled} vs {n_share}");
    }
    // Nearly reached when every bin has 91 of 100 sites uncalled.
    let (uncalled, n_share) = missing_shares([(91, true), (9, false)].repeat(1_000));
    assert_eq!(n_share, 1.0);
    assert!((uncalled - 0.91).abs() < 1e-12, "{uncalled}");
}

#[test]
fn scattered_missing_sites_are_caught_by_the_site_count() {
    // HIGH_MISSING: 60% of 2 Mb missing at random leaves almost no N bin,
    // so only the loader's site count shows the missing data, and the
    // heterozygosity of the called sites shows the bias of the bins.
    let sites = holey_sites(7);
    let rows = vec![sites.holey];
    let mut seq = QcSeq::new("chr1", rows[0].len());
    seq.extent_bases = Some(2_000_000);
    seq.called_bases = Some(sites.called);
    seq.het_sites = Some(sites.hets);
    let r = qc(&rows, &[true], &[seq]);
    let t = &r.totals;
    assert_eq!(t.missing_bins, 0);
    let uncalled = t.uncalled_site_fraction.unwrap();
    assert!((0.59..0.61).contains(&uncalled), "{uncalled}");
    let f = r.finding(QcCode::HighMissing).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    // About 0.8 het sites per kb from the bins against 2 in the called sites.
    let (from_bins, from_sites) = (t.het_per_kb.unwrap(), t.site_het_per_kb.unwrap());
    assert!((1.9..2.1).contains(&from_sites), "{from_sites}");
    assert!(from_bins < 0.5 * from_sites, "{from_bins} vs {from_sites}");
}

#[test]
fn variants_only_vcf_without_mask_lowers_theta() {
    // VARIANTS_ONLY_NO_MASK: the same calls read with and without the
    // callable mask that says only half of the chromosome was callable.
    let dir = TempDir::new("qc_rationale");
    let vcf = dir.path("calls.vcf");
    let mut text = String::from(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=200000>\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
    );
    // A het SNP every 400 bp in the callable first half.
    for pos in (200..100_000).step_by(400) {
        text.push_str(&format!("chr1\t{pos}\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"));
    }
    fs::write(&vcf, text).unwrap();
    let mask = dir.path("callable.bed");
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
        let r = qc(&obs.rows, &obs.row_starts, &[]);
        (r.totals.theta0_per_bin.unwrap(), r.totals.missing_fraction)
    };
    let (masked, masked_missing) = theta(vec![mask]);
    let (unmasked, unmasked_missing) = theta(vec![]);
    assert_eq!((masked_missing, unmasked_missing), (0.5, 0.0));
    // 250 K of 1,000 called bins against 250 K of 2,000.
    assert_eq!(masked, theta0_per_bin(750, 250).unwrap());
    assert_eq!(unmasked, theta0_per_bin(1750, 250).unwrap());
    assert!(unmasked < 0.5 * masked);
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
