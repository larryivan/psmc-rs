//! QC of the callable bases of each bin: their summary, the
//! PARTLY_CALLABLE_BINS finding (corrected by the fit, ignored by it, or
//! unknown for psmcfa input), and heterozygosity compared per callable base.

use psmc_rs::io::CallableBases;
use psmc_rs::qc::{
    PARTLY_CALLABLE_WARN_MEAN, QcCode, QcContext, QcInput, QcReport, QcSeq, Severity, analyze,
    render_text,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Sequences of `n` bins of 100 bases each: every bin has `share` of its
/// bases callable on average (each base independently), shows a
/// heterozygous site with probability `1 - exp(-theta * c / 100)`, and is
/// N below 10 callable bases.
fn sequences(specs: &[(&str, usize, f64, f64)], seed: u64) -> (Vec<Vec<u8>>, Vec<Vec<u16>>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut rows, mut counts) = (Vec::new(), Vec::new());
    for &(_, n, share, theta) in specs {
        let (mut row, mut cs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let c = (0..100).filter(|_| rng.gen_bool(share)).count() as u16;
            let het = rng.gen_bool(-(-theta * f64::from(c) / 100.0).exp_m1());
            row.push(if c < 10 { 2 } else { u8::from(het) });
            cs.push(c);
        }
        rows.push(row);
        counts.push(cs);
    }
    (rows, counts)
}

fn qc(
    specs: &[(&str, usize, f64, f64)],
    rows: &[Vec<u8>],
    callable: Option<&CallableBases>,
    context: QcContext,
) -> QcReport {
    let seqs: Vec<QcSeq> = specs
        .iter()
        .zip(rows)
        .map(|(s, r)| QcSeq::new(s.0, r.len()))
        .collect();
    let starts = vec![true; rows.len()];
    analyze(&QcInput {
        rows,
        row_starts: &starts,
        seqs: &seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context,
        callable,
    })
}

#[test]
fn counts_used_by_the_fit_are_reported_as_corrected() {
    let specs = [("chr1", 200_000, 0.7, 0.01)];
    let (rows, counts) = sequences(&specs, 1);
    let callable = CallableBases::from_counts(100, counts).unwrap();
    let r = qc(&specs, &rows, Some(&callable), QcContext::default());
    let c = r.callable.as_ref().unwrap();
    assert!((c.mean_fraction - 0.7).abs() < 0.005, "{}", c.mean_fraction);
    assert!(c.partial_fraction() > 0.99);
    assert_eq!(c.histogram.iter().sum::<u64>(), c.called_bins);
    let t = &r.totals;
    assert!((t.mean_callable_fraction.unwrap() - c.mean_fraction).abs() < 1e-12);
    // θ0 per callable base is the simulated one; C's per bin is ~0.7 of it.
    let theta = t.theta0_callable.unwrap();
    assert!((theta / 0.01 - 1.0).abs() < 0.05, "{theta}");
    assert!((t.theta0_per_bin.unwrap() / theta - 0.7).abs() < 0.03);

    let f = r.finding(QcCode::PartlyCallableBins).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(
        f.message
            .contains("of called bins hold fewer than 100 callable bases, 70.")
            && f.message.contains("which removes that bias"),
        "{}",
        f.message
    );
    assert!(f.recommendation.starts_with("Nothing to do"));
    let text = render_text(&r);
    assert!(text.contains("Callable bases"), "{text}");
    assert!(text.contains("called bins hold 70."), "{text}");
}

#[test]
fn counts_ignored_by_the_fit_are_a_warning_below_the_threshold() {
    let ignored = QcContext {
        counts_ignored: true,
        ..QcContext::default()
    };
    for (share, severity) in [
        (0.7, Some(Severity::Warning)),
        (0.95, Some(Severity::Info)),
        (0.99995, None),
    ] {
        let specs = [("chr1", 100_000, share, 0.01)];
        let (rows, counts) = sequences(&specs, 2);
        let callable = CallableBases::from_counts(100, counts).unwrap();
        let r = qc(&specs, &rows, Some(&callable), ignored.clone());
        let f = r.finding(QcCode::PartlyCallableBins);
        assert_eq!(f.map(|f| f.severity), severity, "{share}");
        if let Some(f) = f {
            assert!(f.message.contains("--bin-callable ignore"), "{}", f.message);
            assert!(f.recommendation.contains("--bin-callable auto"));
            let mean = r.callable.as_ref().unwrap().mean_fraction;
            assert_eq!(mean < PARTLY_CALLABLE_WARN_MEAN, share < 0.9);
        }
    }
}

#[test]
fn psmcfa_input_says_how_to_check() {
    // N bins in runs: the called bins next to them are partly callable.
    let mut row = vec![0u8; 10_000];
    for start in (50..9_900).step_by(100) {
        row[start..start + 5].fill(2);
    }
    row[9_999] = 2;
    let specs = [("chr1", row.len(), 1.0, 0.0)];
    let rows = vec![row];
    let r = qc(&specs, &rows, None, QcContext::default());
    // Two neighbours per run, one for the run at the end.
    assert_eq!(r.totals.called_next_to_missing, 2 * 99 + 1);
    assert!(r.callable.is_none());
    let f = r.finding(QcCode::PartlyCallableBins).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(f.message.contains("199 called bins"), "{}", f.message);
    assert!(f.recommendation.contains("psmc-rs qc"));
    // A few missing runs (fewer than 1% of the called bins next to them),
    // or none, are no reason to say anything.
    for n_runs in [0, 3] {
        let mut row = vec![0u8; 10_000];
        for i in 0..n_runs {
            row[1000 * (i + 1)..1000 * (i + 1) + 50].fill(2);
        }
        let rows = vec![row];
        let r = qc(
            &[("chr1", 10_000, 1.0, 0.0)],
            &rows,
            None,
            QcContext::default(),
        );
        assert!(r.finding(QcCode::PartlyCallableBins).is_none(), "{n_runs}");
    }
}

#[test]
fn multihetsep_counts_are_used_and_called_approximate() {
    let specs = [("chr1", 50_000, 0.8, 0.01)];
    let (rows, counts) = sequences(&specs, 3);
    let callable = CallableBases::from_counts(100, counts).unwrap();
    let ctx = QcContext {
        approximate_missing_placement: true,
        ..QcContext::default()
    };
    let r = qc(&specs, &rows, Some(&callable), ctx);
    let f = r.finding(QcCode::PartlyCallableBins).unwrap();
    assert!(f.message.contains("approximate"), "{}", f.message);
    let f = r.finding(QcCode::ApproxMissingPlacement).unwrap();
    assert!(f.message.contains("which the fit uses"), "{}", f.message);
}

/// A sequence with more missing data has fewer heterozygous called bins,
/// but not fewer per callable base: QC compares sequences per callable
/// base when it has the counts.
#[test]
fn heterozygosity_is_compared_per_callable_base() {
    let mut specs: Vec<(String, usize, f64, f64)> = (1..=6)
        .map(|i| (format!("chr{i}"), 30_000, 0.95, 0.01))
        .collect();
    specs[5].2 = 0.45;
    let specs: Vec<(&str, usize, f64, f64)> = specs
        .iter()
        .map(|(n, a, b, c)| (n.as_str(), *a, *b, *c))
        .collect();
    let (rows, counts) = sequences(&specs, 4);
    let callable = CallableBases::from_counts(100, counts).unwrap();
    let with = qc(&specs, &rows, Some(&callable), QcContext::default());
    let without = qc(&specs, &rows, None, QcContext::default());
    let ratio = |r: &QcReport| r.sequences[5].het_vs_reference.unwrap();
    assert!((ratio(&with) - 1.0).abs() < 0.1, "{}", ratio(&with));
    assert!(ratio(&without) < 0.6, "{}", ratio(&without));
    assert!(with.finding(QcCode::HetOutlier).is_none());
    let outliers = without.finding(QcCode::HetOutlier).unwrap();
    assert_eq!(outliers.sequences, ["chr6"]);
}
