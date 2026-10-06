//! Every QC warning, triggered and not triggered at its threshold.

use psmc_rs::PsmcModel;
use psmc_rs::qc::{
    Finding, FitStart, HIGH_MISSING_INFO_FRACTION, HIGH_MISSING_WARN_FRACTION, MIN_CALLED_BASES,
    MIN_HET_BINS, QcCode, QcContext, QcInput, QcReport, QcSeq, SATURATION_TARGET_SHARE,
    SATURATION_WARN_SHARE, SHORT_SEQUENCES_MAX_FRACTION, Severity, analyze, het_per_site,
    largest_trunk_within, max_bin_size_for_share, multi_het_share,
};

/// Sequences given as (name, T, K, N) bin counts, laid out K..., T..., N....
struct Data {
    rows: Vec<Vec<u8>>,
    starts: Vec<bool>,
    seqs: Vec<QcSeq>,
}

fn data(spec: &[(&str, usize, usize, usize)]) -> Data {
    let mut d = Data {
        rows: Vec::new(),
        starts: Vec::new(),
        seqs: Vec::new(),
    };
    for &(name, t, k, n) in spec {
        let mut row = vec![1u8; k];
        row.resize(k + t, 0);
        row.resize(k + t + n, 2);
        d.seqs.push(QcSeq::new(name, row.len()));
        d.rows.push(row);
        d.starts.push(true);
    }
    d
}

fn qc(d: &Data, bin_size: usize, trunk: usize, context: QcContext) -> QcReport {
    analyze(&QcInput {
        rows: &d.rows,
        row_starts: &d.starts,
        seqs: &d.seqs,
        bin_size,
        bootstrap_trunk_bins: trunk,
        context,
        callable: None,
    })
}

fn qc100(d: &Data) -> QcReport {
    qc(d, 100, 0, QcContext::default())
}

fn find(r: &QcReport, code: QcCode) -> Option<&Finding> {
    r.findings.iter().find(|f| f.code == code)
}

fn severity(r: &QcReport, code: QcCode) -> Option<Severity> {
    find(r, code).map(|f| f.severity)
}

#[test]
fn codes_serialise_to_their_stable_names() {
    for &code in QcCode::ALL {
        let json = serde_json::to_value(code).unwrap();
        assert_eq!(json, code.as_str());
    }
    let names: Vec<&str> = QcCode::ALL.iter().map(|c| c.as_str()).collect();
    assert_eq!(
        names,
        [
            "INVALID_START",
            "VARIANTS_ONLY_NO_MASK",
            "PARTLY_CALLABLE_BINS",
            "HIGH_MISSING",
            "BIN_SATURATION",
            "FEW_HETS",
            "TOO_SHORT",
            "SHORT_SEQUENCES",
            "SEX_OR_ORGANELLE",
            "HET_OUTLIER",
            "APPROX_MISSING_PLACEMENT",
            "DEPTH",
            "DROPPED_SEQUENCES",
        ]
    );
    assert_eq!(serde_json::to_value(Severity::Info).unwrap(), "info");
    assert_eq!(serde_json::to_value(Severity::Warning).unwrap(), "warning");
    assert_eq!(Severity::Warning.as_str(), "warning");
}

#[test]
fn variants_only_no_mask() {
    let d = data(&[("chr1", 2_000_000, 5_000, 10)]);
    assert_eq!(severity(&qc100(&d), QcCode::VariantsOnlyNoMask), None);
    let ctx = QcContext {
        variants_only_no_mask: true,
        ..QcContext::default()
    };
    let r = qc(&d, 100, 0, ctx);
    let f = find(&r, QcCode::VariantsOnlyNoMask).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(f.message.contains("0.00050%"), "{}", f.message);
    assert!(f.recommendation.contains("--vcf-mask"));
    assert!(f.recommendation.contains("bcftools call -c"));

    // A gVCF read without a mask: positions no block covers are missing,
    // so the finding speaks of its reference blocks instead.
    let r = qc(
        &d,
        100,
        0,
        QcContext {
            variants_only_no_mask: true,
            gvcf: true,
            ..QcContext::default()
        },
    );
    let f = find(&r, QcCode::VariantsOnlyNoMask).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message
            .starts_with("The gVCF was read without a callable-sites mask")
            && f.message
                .contains("reference block counts as called homozygous"),
        "{}",
        f.message
    );
    assert!(
        !f.message.contains("every position without a record"),
        "{}",
        f.message
    );
    assert!(
        f.recommendation.contains("from the gVCF's blocks")
            && f.recommendation.contains("--vcf-mask"),
        "{}",
        f.recommendation
    );
}

#[test]
fn high_missing_thresholds() {
    assert_eq!(
        (HIGH_MISSING_INFO_FRACTION, HIGH_MISSING_WARN_FRACTION),
        (0.25, 0.5)
    );
    // 1,000 bins, the given number missing.
    let missing = |n: usize| qc100(&data(&[("chr1", 1000 - n - 100, 100, n)]));
    assert_eq!(severity(&missing(250), QcCode::HighMissing), None);
    assert_eq!(
        severity(&missing(251), QcCode::HighMissing),
        Some(Severity::Info)
    );
    assert_eq!(
        severity(&missing(500), QcCode::HighMissing),
        Some(Severity::Info)
    );
    let r = missing(501);
    let f = find(&r, QcCode::HighMissing).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message.starts_with("50.1% of bins are missing"),
        "{}",
        f.message
    );
    assert!(
        f.message.contains("only 49.9 kb of sequence"),
        "{}",
        f.message
    );
    assert!(f.recommendation.contains("25%"));
}

#[test]
fn high_missing_reports_uncalled_sites_when_known() {
    let mut d = data(&[("chr1", 300, 100, 600)]);
    d.seqs[0].extent_bases = Some(100_000);
    d.seqs[0].called_bases = Some(30_000);
    let r = qc100(&d);
    let f = find(&r, QcCode::HighMissing).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message.starts_with(
            "70.0% of sites are uncalled (30.0 kb called) and 60.0% of bins are missing (N: \
             more than 90% of their sites uncalled)."
        ),
        "{}",
        f.message
    );
    // No heterozygous-site count, so no comparison with the bins.
    assert!(!f.message.contains("per kb"), "{}", f.message);
}

#[test]
fn high_missing_thresholds_on_uncalled_sites() {
    // 1,000 called bins (no N) over 100 kb, of which `called` sites are
    // called.
    let with_called = |called: u64| {
        let mut d = data(&[("chr1", 900, 100, 0)]);
        d.seqs[0].extent_bases = Some(100_000);
        d.seqs[0].called_bases = Some(called);
        qc100(&d)
    };
    assert_eq!(severity(&with_called(75_000), QcCode::HighMissing), None);
    assert_eq!(
        severity(&with_called(74_999), QcCode::HighMissing),
        Some(Severity::Info)
    );
    assert_eq!(
        severity(&with_called(50_000), QcCode::HighMissing),
        Some(Severity::Info)
    );
    assert_eq!(
        severity(&with_called(49_999), QcCode::HighMissing),
        Some(Severity::Warning)
    );
    // Without the loader's count only the N bins are seen.
    let d = data(&[("chr1", 900, 100, 0)]);
    assert_eq!(severity(&qc100(&d), QcCode::HighMissing), None);
}

#[test]
fn high_missing_sees_uncalled_sites_scattered_over_called_bins() {
    // 2,000,000 called bins, one in 12 K, no N bin; the loader saw 80 Mb of
    // 200 Mb called and one heterozygous site per K bin.
    let n = 2_000_000;
    let mut d = data(&[("chr1", n - n / 12, n / 12, 0)]);
    d.seqs[0].extent_bases = Some(200_000_000);
    d.seqs[0].called_bases = Some(80_000_000);
    d.seqs[0].het_sites = Some(n as u64 / 12);
    let r = qc100(&d);
    assert_eq!(r.totals.missing_fraction, 0.0);
    assert!((r.totals.uncalled_site_fraction.unwrap() - 0.6).abs() < 1e-12);
    let f = find(&r, QcCode::HighMissing).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    // theta0 = -ln(11/12) = 0.0870 per bin; 166,666 / 80 Mb = 2.08 per kb.
    assert_eq!(
        f.message,
        "60.0% of sites are uncalled (80.0 Mb called) and 0.0% of bins are missing (N: more \
         than 90% of their sites uncalled). A bin is called when just 10% of its sites are, \
         so missing sites scattered along the genome leave partly called bins, which show \
         fewer heterozygous sites and bias theta and Ne low if they are treated as full bins; \
         here the called bins imply 0.870 heterozygous sites per kb, against 2.08 per kb of \
         called sites."
    );
    assert!(f.recommendation.contains("per-site filters"));
}

#[test]
fn high_missing_takes_the_larger_of_the_two_shares() {
    // 52% N bins (a warning on its own), but only 48% of sites uncalled:
    // the N bins hold some called sites.
    let mut d = data(&[("chr1", 380, 100, 520)]);
    d.seqs[0].extent_bases = Some(100_000);
    d.seqs[0].called_bases = Some(52_000);
    let r = qc100(&d);
    assert_eq!(severity(&r, QcCode::HighMissing), Some(Severity::Warning));
}

/// K fraction at which the multi-het share reaches `share`, by bisection.
fn het_fraction_at_share(share: f64, bin_size: usize) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if multi_het_share(het_per_site(mid, bin_size), bin_size) > share {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

#[test]
fn bin_saturation_threshold() {
    assert_eq!(
        (SATURATION_WARN_SHARE, SATURATION_TARGET_SHARE),
        (0.10, 0.05)
    );
    let called = 1_000_000usize;
    let p = het_fraction_at_share(SATURATION_WARN_SHARE, 100);
    let k_below = (p * called as f64).floor() as usize;
    for (k, fires) in [(k_below, false), (k_below + 1, true)] {
        let r = qc100(&data(&[("chr1", called - k, k, 0)]));
        let share = r.saturation.as_ref().unwrap().multi_het_share;
        assert_eq!(share > SATURATION_WARN_SHARE, fires, "k {k}: share {share}");
        assert_eq!(find(&r, QcCode::BinSaturation).is_some(), fires, "k {k}");
    }

    let r = qc100(&data(&[("chr1", called - k_below - 1, k_below + 1, 0)]));
    let f = find(&r, QcCode::BinSaturation).unwrap();
    let h = het_per_site((k_below + 1) as f64 / called as f64, 100);
    let best = max_bin_size_for_share(h, SATURATION_TARGET_SHARE).unwrap();
    assert!((40..60).contains(&best), "{best}");
    assert_eq!(f.severity, Severity::Warning);
    assert!(f.message.contains("At 100 bp per bin"), "{}", f.message);
    assert!(f.message.contains("about 10.0% of K bins"), "{}", f.message);
    assert!(
        f.recommendation
            .contains(&format!("at {best} bp per bin or less")),
        "{}",
        f.recommendation
    );
    assert!(f.recommendation.contains(&format!("fq2psmcfa -s {best}")));
    assert!(f.recommendation.contains(&format!("`--bin-size {best}`")));
}

/// A model of this layout with rates far inside the valid range.
fn layout(t_max: f64, n_steps: usize, pattern: Option<&str>) -> PsmcModel {
    PsmcModel::new(
        t_max,
        n_steps,
        1e-4,
        1e-4,
        2.5e-8,
        pattern.map(str::to_string),
    )
    .unwrap()
}

/// A start whose rho0 per bin is too large for the time grid has negative
/// transition probabilities, from which EM cannot move: INVALID_START, the
/// first warning, with the limit and how to get below it.
#[test]
fn invalid_start() {
    let c_default = layout(15.0, 22, Some("4+5*3+4"));
    // 60% K bins: theta0 = -ln(0.4), rho0 = theta0 / 4 = 0.229.
    let theta0 = -(0.4f64.ln());
    let start = FitStart::new(&c_default, theta0, theta0 / 4.0, false);
    assert!(!start.is_valid());
    assert!(start.min_transition < -0.1, "{}", start.min_transition);
    assert_eq!((start.t_max, start.assumed), (15.0, false));
    // The limit of C psmc's default start: rho0 of about 1/6 per bin.
    let max = start.max_rho0.unwrap();
    assert!((0.1665..0.1668).contains(&max), "{max}");
    // Every rho0 below the limit is valid, and every one above is not.
    for i in 1..=40 {
        let rho = max * i as f64 / 20.0;
        let s = FitStart::new(&c_default, theta0, rho, false);
        assert_eq!(s.is_valid(), rho <= max, "rho0 {rho}");
        assert_eq!(s.max_rho0.is_none(), s.is_valid());
    }
    // psmc's human pattern and a larger t_max allow less.
    let human = FitStart::new(&layout(15.0, 63, Some("4+25*2+4+6")), 0.9, 0.9, false);
    assert!((0.160..0.162).contains(&human.max_rho0.unwrap()));
    let wide = FitStart::new(&layout(50.0, 22, Some("4+5*3+4")), 0.9, 0.9, false);
    assert!((0.042..0.043).contains(&wide.max_rho0.unwrap()));

    let d = data(&[("chr1", 400_000, 600_000, 0)]);
    let with = |start: Option<FitStart>| {
        qc(
            &d,
            100,
            0,
            QcContext {
                start,
                ..QcContext::default()
            },
        )
    };
    assert_eq!(severity(&with(None), QcCode::InvalidStart), None);
    let valid = FitStart::new(&c_default, theta0, 0.15, false);
    assert_eq!(severity(&with(Some(valid)), QcCode::InvalidStart), None);

    let r = with(Some(start.clone()));
    let f = &r.findings[0];
    assert_eq!(
        (f.code, f.severity),
        (QcCode::InvalidStart, Severity::Warning)
    );
    assert!(r.has_warnings());
    assert!(
        f.message.starts_with(
            "The fit starts from theta0 0.916 and rho0 0.229 per bin, with t_max 15 and pattern \
             4+5*3+4. Some of PSMC's transition probabilities are then negative (down to -0.1"
        ),
        "{}",
        f.message
    );
    assert!(
        f.message.contains("which allows up to about 0.167"),
        "{}",
        f.message
    );
    assert!(
        f.message.contains("EM cannot move from it"),
        "{}",
        f.message
    );
    // A tenth below the limit: 0.9 * 0.1667 / 0.229 of 100 bp, and the
    // ratio theta0 / (0.9 * 0.1667) rounded up.
    for advice in [
        "re-bin the per-site data at 65 bp per bin or less",
        "`--bin-size 65`",
        "`fq2psmcfa -s 65`",
        "`--tr-ratio 6.2` or more",
        "`--rho0 0.15` or less",
    ] {
        assert!(
            f.recommendation.contains(advice),
            "{advice}: {}",
            f.recommendation
        );
    }

    // A saved fit's rho comes from the --init-params file: the options of
    // a fresh start (--tr-ratio, --rho0) do not apply to it.
    let saved = FitStart {
        saved: true,
        ..start.clone()
    };
    let f = with(Some(saved)).findings[0].clone();
    assert!(
        f.message
            .starts_with("The fit starts from the saved parameters, theta0 0.916 and rho0 0.229"),
        "{}",
        f.message
    );
    assert!(
        f.recommendation
            .contains("lower rho in the --init-params file to")
            && f.recommendation
                .contains("fresh start instead of --init-params"),
        "{}",
        f.recommendation
    );
    for flag in ["--tr-ratio", "--rho0", "--bin-size"] {
        assert!(
            !f.recommendation.contains(flag),
            "{flag}: {}",
            f.recommendation
        );
    }

    // Without a fit (psmc-rs qc), the default start is assumed.
    let assumed = FitStart {
        assumed: true,
        ..start
    };
    let f = with(Some(assumed)).findings[0].clone();
    assert!(
        f.message.starts_with(
            "A fit with psmc-rs's default settings would start from theta0 0.916 per bin (from \
             these data) and rho0 = theta0/4 = 0.229 per bin"
        ),
        "{}",
        f.message
    );
}

#[test]
fn share_grows_with_the_het_fraction() {
    // The threshold search above relies on this.
    for s in [2usize, 10, 100, 1000] {
        let mut prev = 0.0;
        for i in 1..1000 {
            let f = multi_het_share(het_per_site(i as f64 / 1000.0, s), s);
            assert!(f >= prev, "s {s}, p {}", i as f64 / 1000.0);
            prev = f;
        }
    }
}

#[test]
fn few_hets_threshold() {
    assert_eq!(MIN_HET_BINS, 1000);
    let r = qc100(&data(&[("chr1", 2_000_000, 999, 0)]));
    let f = find(&r, QcCode::FewHets).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message
            .starts_with("Only 999 heterozygous (K) bins, fewer than 1,000")
    );
    // C psmc's round-0 n_recomb for these data.
    let n = r.expected_recombinations.unwrap();
    assert!((n - 249.75).abs() < 0.1, "{n}");
    assert!(
        f.message.contains("about 250 recombinations"),
        "{}",
        f.message
    );
    // 4+25*2+4+6 is psmc's suggestion for human data, not a default (C
    // psmc's is 4+5*3+4).
    assert!(
        f.message
            .contains("the pattern its README uses for human data, 4+25*2+4+6, has 28 of them"),
        "{}",
        f.message
    );
    assert!(!f.message.contains("default"), "{}", f.message);

    // With the fit's own count: 249.75 / 65 = 3.84 and 249.75 / 28 = 8.92
    // recombinations per free parameter.
    let d = data(&[("chr1", 2_000_000, 999, 0)]);
    for (lambdas, quoted) in [(65, "3.8"), (28, "8.9")] {
        let ctx = QcContext {
            free_lambdas: Some(lambdas),
            ..QcContext::default()
        };
        let r = qc(&d, 100, 0, ctx);
        let f = find(&r, QcCode::FewHets).unwrap();
        let expected = format!(
            "each free parameter, and this fit has {lambdas} of them, about {quoted} \
             recombinations each on average (the oldest intervals usually get far fewer)."
        );
        assert!(f.message.ends_with(&expected), "{}", f.message);
        assert!(!f.message.contains("4+25*2+4+6"), "{}", f.message);
    }
    // No called bin is T, so theta0 and the expected recombinations are
    // undefined: only the count is quoted. A count of 0 means unknown.
    let all_k = data(&[("chr1", 0, 10, 0)]);
    let with = |lambdas| {
        let ctx = QcContext {
            free_lambdas: Some(lambdas),
            ..QcContext::default()
        };
        find(&qc(&all_k, 100, 0, ctx), QcCode::FewHets)
            .unwrap()
            .message
            .clone()
    };
    assert!(
        with(7).ends_with("and this fit has 7 of them."),
        "{}",
        with(7)
    );
    assert!(
        with(0).ends_with("4+25*2+4+6, has 28 of them."),
        "{}",
        with(0)
    );

    let r = qc100(&data(&[("chr1", 2_000_000, 1000, 0)]));
    assert_eq!(find(&r, QcCode::FewHets), None);
}

#[test]
fn too_short_threshold_on_bins() {
    assert_eq!(MIN_CALLED_BASES, 100_000_000);
    // 100 bp bins: 1,000,000 called bins are exactly 100 Mb; N bins do not count.
    let r = qc100(&data(&[("chr1", 990_000, 10_000, 5_000)]));
    assert_eq!(find(&r, QcCode::TooShort), None);
    let r = qc100(&data(&[("chr1", 989_999, 10_000, 5_000)]));
    let f = find(&r, QcCode::TooShort).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message
            .starts_with("Only 99.9 Mb of called sequence (bases in called bins)"),
        "{}",
        f.message
    );
}

#[test]
fn too_short_prefers_the_loaders_site_count() {
    let mut d = data(&[("chr1", 990_000, 10_000, 0)]);
    d.seqs[0].called_bases = Some(99_999_999);
    let r = qc100(&d);
    let f = find(&r, QcCode::TooShort).unwrap();
    assert!(f.message.contains("(called sites)"), "{}", f.message);
    d.seqs[0].called_bases = Some(100_000_000);
    assert_eq!(find(&qc100(&d), QcCode::TooShort), None);
}

#[test]
fn short_sequences_below_1mb() {
    assert_eq!(SHORT_SEQUENCES_MAX_FRACTION, 0.20);
    // Two 9,999-bin sequences (999,900 bp) and one long one; at 100 bp per bin
    // 1 Mb is 10,000 bins. 19,998 / (4 * 19,998 + 19,998) is exactly 20%.
    let with_long = |long: usize| {
        data(&[
            ("a", 9_999, 0, 0),
            ("long", long, 0, 0),
            ("b", 9_000, 999, 0),
        ])
    };
    let r = qc100(&with_long(79_992));
    assert_eq!(r.lengths.below_1mb.fraction, 0.2);
    assert_eq!(find(&r, QcCode::ShortSequences), None);
    let r = qc100(&with_long(79_991));
    let f = find(&r, QcCode::ShortSequences).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message
            .starts_with("20.0% of bins lie in 2 sequences shorter than 1 Mb"),
        "{}",
        f.message
    );
    assert!(
        f.recommendation
            .contains("drop sequences shorter than 1 Mb")
    );
    assert!(!f.message.contains("trunk"));
}

#[test]
fn short_sequences_use_the_loaders_extent() {
    // 10,000 bins look like 1 Mb, but the sequence spans 999,999 bases.
    let mut d = data(&[("a", 10_000, 0, 0), ("long", 30_000, 0, 0)]);
    assert_eq!(find(&qc100(&d), QcCode::ShortSequences), None);
    d.seqs[0].extent_bases = Some(999_999);
    let r = qc100(&d);
    assert_eq!(r.lengths.below_1mb.sequences, 1);
    assert!(find(&r, QcCode::ShortSequences).is_some());
}

#[test]
fn short_sequences_below_the_trunk() {
    let trunk = 50_000;
    // All sequences are longer than 1 Mb (10,000 bins).
    let with_long = |long: usize| data(&[("long", long, 0, 0), ("short", 49_999, 0, 0)]);
    let r = qc(&with_long(199_996), 100, trunk, QcContext::default());
    assert_eq!(r.lengths.below_trunk.as_ref().unwrap().fraction, 0.2);
    assert_eq!(find(&r, QcCode::ShortSequences), None);

    let r = qc(&with_long(199_995), 100, trunk, QcContext::default());
    let f = find(&r, QcCode::ShortSequences).unwrap();
    assert!(
        f.message.starts_with(
            "20.0% of bins lie in 1 sequences shorter than the bootstrap trunk (50,000 bins)"
        ),
        "{}",
        f.message
    );
    assert!(!f.message.contains("1 Mb"));
    assert!(
        f.recommendation
            .contains("a trunk of at most 49,999 bins keeps at least 80.0% of bins"),
        "{}",
        f.recommendation
    );
    assert!(f.recommendation.contains("--bootstrap-block-size"));

    // No bootstrap planned: no trunk check.
    let r = qc(&with_long(10_000), 100, 0, QcContext::default());
    assert_eq!(find(&r, QcCode::ShortSequences), None);
}

#[test]
fn largest_trunk_within_is_the_largest_that_keeps_the_share() {
    let share_below = |lens: &[u64], trunk: u64| {
        let total: u64 = lens.iter().sum();
        lens.iter().filter(|&&l| l < trunk).sum::<u64>() as f64 / total as f64
    };
    let mut state = 12345u64;
    for _ in 0..500 {
        let n = 1 + (state % 12) as usize;
        let lens: Vec<u64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                1 + (state >> 33) % 50
            })
            .collect();
        for &frac in &[0.0, 0.1, 0.2, 0.5, 0.9] {
            let t = largest_trunk_within(&lens, frac);
            assert!(share_below(&lens, t) <= frac, "{lens:?} {frac}: {t}");
            assert!(share_below(&lens, t + 1) > frac, "{lens:?} {frac}: {t}");
        }
    }
    assert_eq!(largest_trunk_within(&[5, 5, 5, 5, 5], 0.2), 5);
    assert_eq!(largest_trunk_within(&[], 0.2), 0);
}

#[test]
fn clean_input_has_no_findings() {
    // 3 Gb at 100 bp bins would be slow to build; 150 Mb in 1.5M-bin
    // chromosomes is enough: no missing data, 5% K bins (share ~2.5%).
    let spec: Vec<(String, usize, usize, usize)> = (1..=10)
        .map(|i| (format!("chr{i}"), 142_500, 7_500, 0))
        .collect();
    let spec: Vec<(&str, usize, usize, usize)> = spec
        .iter()
        .map(|(n, t, k, m)| (n.as_str(), *t, *k, *m))
        .collect();
    let r = qc(&data(&spec), 100, 50_000, QcContext::default());
    assert!(r.findings.is_empty(), "{:#?}", r.findings);
    assert!(!r.has_warnings());
}

#[test]
fn warnings_come_first() {
    // Info-level missingness (30%), few hets, no mask; the one sequence is
    // exactly 1 Mb, which is not short.
    let d = data(&[("chr1", 6_500, 500, 3_000)]);
    let ctx = QcContext {
        variants_only_no_mask: true,
        ..QcContext::default()
    };
    let r = qc(&d, 100, 0, ctx);
    let got: Vec<(QcCode, Severity)> = r.findings.iter().map(|f| (f.code, f.severity)).collect();
    assert_eq!(
        got,
        [
            (QcCode::VariantsOnlyNoMask, Severity::Warning),
            (QcCode::FewHets, Severity::Warning),
            (QcCode::TooShort, Severity::Warning),
            (QcCode::HighMissing, Severity::Info),
        ]
    );
    assert!(r.has_warnings());
    assert_eq!(
        r.finding(QcCode::HighMissing).unwrap().severity,
        Severity::Info
    );

    let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(v["findings"][0]["code"], "VARIANTS_ONLY_NO_MASK");
    assert_eq!(v["findings"][3]["severity"], "info");
    assert!(v["findings"][3]["recommendation"].is_string());
}

/// The summary stored with a fit lists the codes by severity and copies the
/// key metrics.
#[test]
fn summary_lists_codes_and_key_metrics() {
    // Few K bins (a warning), placement by assumption with 32% of the
    // bins missing (a warning) and that much missing data (a note).
    let d = data(&[("chr1", 3000, 40, 1500), ("chrM", 160, 0, 0)]);
    let r = qc(
        &d,
        100,
        0,
        QcContext {
            approximate_missing_placement: true,
            ..QcContext::default()
        },
    );
    let s = r.summary();
    let codes = |sev: Severity| -> Vec<&str> {
        r.findings
            .iter()
            .filter(|f| f.severity == sev)
            .map(|f| f.code.as_str())
            .collect()
    };
    assert_eq!(s.warnings, codes(Severity::Warning));
    assert_eq!(s.notes, codes(Severity::Info));
    assert!(s.warnings.iter().any(|c| c == "FEW_HETS"), "{s:?}");
    assert!(
        s.warnings.iter().any(|c| c == "APPROX_MISSING_PLACEMENT"),
        "{s:?}"
    );
    assert!(s.notes.iter().any(|c| c == "HIGH_MISSING"), "{s:?}");
    assert_eq!((s.sequences, s.bins), (2, 4700));
    assert_eq!(s.missing_fraction, r.totals.missing_fraction);
    assert_eq!(s.theta0_per_bin, r.totals.theta0_per_bin);
    assert_eq!(s.het_per_kb, r.totals.het_per_kb);
    assert_eq!(s.called_bases, 3200 * 100);
    assert_eq!(s.n50_bases, r.lengths.n50_bases);

    let back: psmc_rs::qc::QcSummary =
        serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(back, s);
}
