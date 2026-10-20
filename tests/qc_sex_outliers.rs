//! Sex chromosome / organelle detection and per-sequence heterozygosity
//! outliers.

use psmc_rs::qc::{
    HEMIZYGOUS_MAX_RATIO, HEMIZYGOUS_MIN_EXPECTED_HET_BINS, HET_OUTLIER_MADS, HET_OUTLIER_MIN_FOLD,
    HET_OUTLIER_MIN_SEQUENCES, MAX_SEQUENCE_ROWS, QcCode, QcContext, QcInput, QcReport, QcSeq,
    REFERENCE_MIN_CALLED_BASES, ReferenceMethod, SeqKind, Severity, analyze, classify_name,
};

#[test]
fn sequence_names_are_classified() {
    let sex = "X x Y Z W chrX ChrX CHRY chrZ chrW X1 chrY2 chrY_KI270740v1_random \
               chromosome_X chromosomeX chrom-Y chr.Z Y_unplaced_1 NC_000023.11 \
               NC_000024.10 NC_000086.8";
    let organelle = "MT Mt mt M chrM chrMT chrM_alt mito mtDNA MtDNA Pt ChrC chrc \
                     mitochondrion dmel_mitochondrion_genome Mitochondrial_genome chloroplast \
                     NC_037304_chloroplast Plastid apicoplast NC_012920.1 NC_001807 \
                     NC_005089.1";
    let other = "1 chr1 chr10 Chr2A chrX1A XY Xenopus_7 Zm00001 scaffold_12 scaffold_X \
                 chrUn_KI270302v1 C MTR CP012345.1 PT12 NC_000001.11 LG7";
    for (names, kind) in [
        (sex, Some(SeqKind::Sex)),
        (organelle, Some(SeqKind::Organelle)),
        (other, None),
    ] {
        for name in names.split_whitespace() {
            assert_eq!(classify_name(name), kind, "{name}");
        }
    }
    assert_eq!(classify_name(""), None);
}

/// Sequences as (name, called bins, K bins), laid out K..., T....
fn qc(spec: &[(&str, usize, usize)]) -> QcReport {
    let rows: Vec<Vec<u8>> = spec
        .iter()
        .map(|&(_, called, k)| {
            let mut row = vec![1u8; k];
            row.resize(called, 0);
            row
        })
        .collect();
    let starts = vec![true; rows.len()];
    let seqs: Vec<QcSeq> = spec
        .iter()
        .map(|&(name, called, _)| QcSeq::new(name, called))
        .collect();
    analyze(&QcInput {
        rows: &rows,
        row_starts: &starts,
        seqs: &seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context: QcContext::default(),
        callable: None,
    })
}

/// θ0 per bin of `k` K bins among `called`.
fn theta(called: usize, k: usize) -> f64 {
    -(1.0 - k as f64 / called as f64).ln()
}

/// Five 2 Mb autosomes with 5% K bins, plus `extra`.
fn with_autosomes(extra: &[(&str, usize, usize)]) -> QcReport {
    let mut spec: Vec<(&str, usize, usize)> = ["1", "2", "3", "4", "5"]
        .iter()
        .map(|&n| (n, 20_000, 1_000))
        .collect();
    spec.extend_from_slice(extra);
    qc(&spec)
}

#[test]
fn autosomal_reference_is_the_median_of_long_sequences() {
    // 2 Mb autosomes at 4%, 5%, 6%; a short one at 50% and chrX at 1% are
    // left out.
    let r = qc(&[
        ("a", 20_000, 800),
        ("b", 20_000, 1_200),
        ("c", 20_000, 1_000),
        ("short", 9_999, 5_000),
        ("chrX", 20_000, 200),
    ]);
    let reference = r.het_reference.as_ref().unwrap();
    assert_eq!(reference.method, ReferenceMethod::Median);
    assert_eq!(reference.sequences, 3);
    assert_eq!(reference.theta0_per_bin, theta(20_000, 1_000));
    assert_eq!(reference.het_per_kb, 10.0 * theta(20_000, 1_000));

    let row = |name: &str| r.sequences.iter().find(|s| s.name == name).unwrap();
    assert_eq!(row("chrX").kind, Some(SeqKind::Sex));
    assert_eq!(row("a").kind, None);
    let ratio = row("b").het_vs_reference.unwrap();
    assert!((ratio - theta(20_000, 1_200) / theta(20_000, 1_000)).abs() < 1e-15);
}

#[test]
fn autosomal_reference_pools_short_sequences() {
    let r = qc(&[("a", 5_000, 100), ("b", 5_000, 300), ("chrM", 100, 0)]);
    let reference = r.het_reference.as_ref().unwrap();
    assert_eq!(reference.method, ReferenceMethod::Pooled);
    assert_eq!(reference.sequences, 2);
    assert_eq!(reference.theta0_per_bin, theta(10_000, 400));
    // Only sex chromosomes: nothing to compare with.
    let r = qc(&[("chrX", 20_000, 0)]);
    assert_eq!(r.het_reference, None);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(
        f.message
            .contains("chrX (sex chromosome, nothing to compare with)")
    );
}

#[test]
fn hemizygous_ratio_threshold() {
    assert_eq!(HEMIZYGOUS_MAX_RATIO, 0.25);
    // The K count at which chrX's θ crosses a quarter of the autosomes'.
    let reference = theta(20_000, 1_000);
    let p = -(-HEMIZYGOUS_MAX_RATIO * reference).exp_m1();
    let k_below = (p * 20_000.0).floor() as usize;
    assert!(theta(20_000, k_below) / reference < HEMIZYGOUS_MAX_RATIO);
    assert!(theta(20_000, k_below + 1) / reference > HEMIZYGOUS_MAX_RATIO);

    let r = with_autosomes(&[("chrX", 20_000, k_below)]);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert_eq!(f.sequences, ["chrX"]);
    assert!(
        f.message
            .starts_with("1 sequence is named like a sex chromosome or organelle: chrX (sex chromosome, 0.25x the autosomal heterozygosity, hemizygous?)"),
        "{}",
        f.message
    );
    assert!(f.message.contains("chrX is far less heterozygous"));
    assert!(f.recommendation.starts_with("Exclude chrX"));

    let r = with_autosomes(&[("chrX", 20_000, k_below + 1)]);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(!f.message.contains("hemizygous?"), "{}", f.message);
    assert!(f.recommendation.contains("autosomes only"));
}

#[test]
fn hemizygous_needs_enough_expected_het_bins() {
    assert_eq!(HEMIZYGOUS_MIN_EXPECTED_HET_BINS, 10.0);
    // The autosomes have 5% K bins: 201 called bins expect 10.05, 199 expect
    // 9.95.
    let r = with_autosomes(&[("chrY", 201, 0)]);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(f.message.contains("chrY (sex chromosome, 0x the autosomal"));

    let r = with_autosomes(&[("chrY", 199, 0)]);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(
        f.message
            .contains("chrY (sex chromosome, too few called bins to judge)")
    );
}

#[test]
fn sex_or_organelle_lists_every_flagged_sequence() {
    let r = with_autosomes(&[("chrX", 20_000, 700), ("chrY", 5_000, 1), ("chrM", 166, 0)]);
    let f = r.finding(QcCode::SexOrOrganelle).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert_eq!(f.sequences, ["chrX", "chrY", "chrM"]);
    assert!(f.message.starts_with("3 sequences are named like"));
    assert!(
        f.message
            .contains("chrX (sex chromosome, 0.69x the autosomal heterozygosity)")
    );
    assert!(
        f.message
            .contains("chrM (organelle, too few called bins to judge)")
    );
    assert!(
        f.recommendation.starts_with("Exclude chrY "),
        "{}",
        f.recommendation
    );
    // Autosomes only: no finding.
    assert_eq!(with_autosomes(&[]).finding(QcCode::SexOrOrganelle), None);
}

/// `base` autosomes of 100 Mb called (1M bins) with K counts `ks`, plus one
/// named "test" with `k_test`.
fn outlier_report(ks: &[usize], k_test: usize) -> QcReport {
    let names: Vec<String> = (1..=ks.len()).map(|i| format!("chr{i}")).collect();
    let mut spec: Vec<(&str, usize, usize)> = names
        .iter()
        .zip(ks)
        .map(|(n, &k)| (n.as_str(), 1_000_000, k))
        .collect();
    spec.push(("test", 1_000_000, k_test));
    qc(&spec)
}

fn outliers(r: &QcReport) -> Vec<String> {
    r.finding(QcCode::HetOutlier)
        .map(|f| f.sequences.clone())
        .unwrap_or_default()
}

/// K count of a 1M-bin sequence whose θ is `theta`.
fn k_for(theta: f64) -> f64 {
    -(-theta).exp_m1() * 1e6
}

#[test]
fn outlier_fold_floor() {
    assert_eq!(HET_OUTLIER_MIN_FOLD, 1.5);
    // Identical base sequences: MAD 0, so the 1.5-fold floor decides.
    let base = vec![20_000usize; 5];
    let t = theta(1_000_000, 20_000) * HET_OUTLIER_MIN_FOLD;
    let k = k_for(t);
    let (lo, hi) = (k.floor() as usize, k.ceil() as usize);
    assert!(theta(1_000_000, lo) < t && theta(1_000_000, hi) > t);
    assert!(outliers(&outlier_report(&base, lo)).is_empty());
    assert_eq!(outliers(&outlier_report(&base, hi)), ["test"]);

    // Below the median as well.
    let t = theta(1_000_000, 20_000) / HET_OUTLIER_MIN_FOLD;
    let k = k_for(t);
    assert!(outliers(&outlier_report(&base, k.ceil() as usize)).is_empty());
    assert_eq!(
        outliers(&outlier_report(&base, k.floor() as usize)),
        ["test"]
    );
}

#[test]
fn outlier_mad_rule() {
    assert_eq!(HET_OUTLIER_MADS, 5.0);
    // Base θ at 0.02 * e^x for spread-out x: 5 MAD exceeds ln 1.5.
    let xs = [-0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24];
    let ks: Vec<usize> = xs
        .iter()
        .map(|x| k_for(0.02 * f64::exp(*x)).round() as usize)
        .collect();
    let mut logs: Vec<f64> = ks.iter().map(|&k| theta(1_000_000, k).ln()).collect();
    // The test sequence is the largest value, so the median of all eight is
    // the mean of the 4th and 5th smallest, and the median deviation does not
    // depend on its exact value either.
    logs.push(f64::INFINITY);
    logs.sort_by(f64::total_cmp);
    let center = 0.5 * (logs[3] + logs[4]);
    let mut dev: Vec<f64> = logs.iter().map(|x| (x - center).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let mad = 0.5 * (dev[3] + dev[4]);
    let limit = HET_OUTLIER_MADS * mad;
    assert!(limit > HET_OUTLIER_MIN_FOLD.ln(), "{limit}");

    let k = k_for((center + limit).exp());
    let (lo, hi) = (k.floor() as usize, k.ceil() as usize);
    assert!(outliers(&outlier_report(&ks, lo)).is_empty());
    let r = outlier_report(&ks, hi);
    assert_eq!(outliers(&r), ["test"]);
    let f = r.finding(QcCode::HetOutlier).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(
        f.message.starts_with(
            "1 of 8 sequences with at least 1.00 Mb in called bins have a heterozygosity far"
        ),
        "{}",
        f.message
    );
    // exp(5 * 0.16) = 2.2.
    assert!(f.message.contains("test (2.2x)"), "{}", f.message);
    assert!(f.recommendation.contains("sex-linked scaffolds"));
}

#[test]
fn outliers_need_enough_long_autosomes() {
    assert_eq!(HET_OUTLIER_MIN_SEQUENCES, 5);
    assert_eq!(REFERENCE_MIN_CALLED_BASES, 1_000_000);
    // 4 base sequences + a 10x one: 5 eligible, flagged.
    let r = outlier_report(&[20_000; 4], 200_000);
    assert_eq!(outliers(&r), ["test"]);
    // 3 + 1: too few.
    let r = outlier_report(&[20_000; 3], 200_000);
    assert!(outliers(&r).is_empty());
    // Sequences under 1 Mb called and sex chromosomes do not count.
    let r = qc(&[
        ("1", 10_000, 500),
        ("2", 10_000, 500),
        ("3", 10_000, 500),
        ("4", 9_999, 500),
        ("chrX", 10_000, 500),
        ("hot", 10_000, 5_000),
    ]);
    assert!(outliers(&r).is_empty());
}

#[test]
fn sequences_without_hets_and_ordering() {
    // A 2 Mb autosome without any K bin is an outlier (0x): its log deviation
    // is infinite, so it is listed before the 5.6x one. chrX is left to
    // SEX_OR_ORGANELLE.
    let r = qc(&[
        ("a", 20_000, 1_000),
        ("b", 20_000, 1_000),
        ("c", 20_000, 1_000),
        ("d", 20_000, 1_000),
        ("hot", 20_000, 5_000),
        ("cold", 20_000, 0),
        ("chrX", 20_000, 0),
    ]);
    assert_eq!(outliers(&r), ["cold", "hot"]);
    let f = r.finding(QcCode::HetOutlier).unwrap();
    assert!(
        f.message.ends_with("cold (0x), hot (5.6x)."),
        "{}",
        f.message
    );
}

#[test]
fn named_sequences_keep_their_rows_when_capped() {
    // More sequences than rows: chrM and chrY are tiny but named in a
    // finding, so they keep their rows.
    let names: Vec<String> = (0..MAX_SEQUENCE_ROWS + 20)
        .map(|i| format!("scaffold_{i}"))
        .collect();
    let mut spec: Vec<(&str, usize, usize)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), 10_000 + i, 500))
        .collect();
    spec.insert(3, ("chrM", 166, 0));
    spec.push(("chrY", 50, 0));
    let r = qc(&spec);
    assert_eq!(r.sequences.len(), MAX_SEQUENCE_ROWS);
    let shown: Vec<&str> = r.sequences.iter().map(|s| s.name.as_str()).collect();
    assert!(shown.contains(&"chrM") && shown.contains(&"chrY"));
    // The rest are the longest scaffolds, in input order.
    assert_eq!(shown[0], "chrM");
    assert_eq!(*shown.last().unwrap(), "chrY");
    assert_eq!(shown[1], "scaffold_22");
    assert_eq!(r.omitted.as_ref().unwrap().sequences, 22);
}
