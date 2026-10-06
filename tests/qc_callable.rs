//! QC of the callable bases of each bin: their summary, the
//! PARTLY_CALLABLE_BINS finding (corrected by the fit, ignored by it, or
//! unknown for psmcfa input), and heterozygosity compared per callable base.

use psmc_rs::io::CallableBases;
use psmc_rs::qc::{
    APPROX_PLACEMENT_WARN_FRACTION, PARTLY_CALLABLE_WARN_MEAN, QcCode, QcContext, QcInput,
    QcReport, QcSeq, Severity, analyze, render_text,
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
            && f.message.contains("which removes most of that bias"),
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
    // 10% of the bases missing already biases theta by more than 10%, so
    // it is a warning, as 30% is; 3% is a note.
    for (share, severity) in [
        (0.7, Some(Severity::Warning)),
        (0.9, Some(Severity::Warning)),
        (0.97, Some(Severity::Info)),
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
            assert_eq!(mean < PARTLY_CALLABLE_WARN_MEAN, share < 0.95);
            assert!(f.message.contains("low or more"), "{}", f.message);
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

/// Multihetsep's counts follow the reader's placement of the uncalled
/// sites, so PARTLY_CALLABLE_BINS does not size a bias from them, the
/// reports call them approximate, and APPROX_MISSING_PLACEMENT says whether
/// the fit uses them (`--bin-callable approximate`) or not (`auto`,
/// `ignore`). Above APPROX_PLACEMENT_WARN_FRACTION of the sites uncalled it
/// warns, with the bias measured in the validation, whatever the mode, and
/// HIGH_MISSING does not say that the fit corrects anything.
#[test]
fn multihetsep_placement_is_a_warning_whatever_the_mode() {
    let specs = [("chr1", 50_000, 0.8, 0.01)];
    let (rows, counts) = sequences(&specs, 3);
    let callable = CallableBases::from_counts(100, counts).unwrap();
    let extent = 50_000 * 100;
    for (uncalled, severity) in [
        (0.3, Severity::Warning),
        (0.06, Severity::Warning),
        (0.04, Severity::Info),
    ] {
        for (ignored, says) in [
            (false, "(--bin-callable approximate)"),
            (true, "--bin-callable auto uses exact counts only"),
        ] {
            let ctx = QcContext {
                approximate_missing_placement: true,
                counts_ignored: ignored,
                ..QcContext::default()
            };
            let seqs = [QcSeq {
                extent_bases: Some(extent),
                called_bases: Some(((1.0 - uncalled) * extent as f64) as u64),
                ..QcSeq::new("chr1", rows[0].len())
            }];
            let starts = [true];
            let r = analyze(&QcInput {
                rows: &rows,
                row_starts: &starts,
                seqs: &seqs,
                bin_size: 100,
                bootstrap_trunk_bins: 0,
                context: ctx,
                callable: Some(&callable),
            });
            assert!(r.finding(QcCode::PartlyCallableBins).is_none());
            let f = r.finding(QcCode::ApproxMissingPlacement).unwrap();
            assert_eq!(f.severity, severity, "{uncalled} {ignored}");
            assert!(f.message.contains(says), "{}", f.message);
            let share = format!("{:.1}% of the sites are uncalled", 100.0 * uncalled);
            assert!(f.message.contains(&share), "{}", f.message);
            if severity == Severity::Warning {
                assert!(
                    f.message.contains("whatever --bin-callable")
                        && f.message.contains("0.67-0.70 with 30%"),
                    "{}",
                    f.message
                );
                assert!(f.recommendation.contains("--vcf-mask"));
            }
            if let Some(h) = r.finding(QcCode::HighMissing) {
                assert!(
                    h.message.contains("APPROX_MISSING_PLACEMENT"),
                    "{}",
                    h.message
                );
                assert!(!h.message.contains("corrects"), "{}", h.message);
            }
            assert_eq!(
                r.finding(QcCode::HighMissing).is_some(),
                uncalled > 0.25,
                "{uncalled}"
            );
            let text = render_text(&r);
            assert!(
                text.contains("(approximate: they follow multihetsep"),
                "{text}"
            );
        }
    }
    const { assert!(APPROX_PLACEMENT_WARN_FRACTION > 0.04 && APPROX_PLACEMENT_WARN_FRACTION < 0.06) };
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
