//! QC checks driven by how the input was read: multihetsep placement,
//! consensus-mode depth and filters, dropped sequences.

use psmc_rs::io::mhs::read_mhs;
use psmc_rs::io::vcf::{VcfOptions, read_vcf_with};
use psmc_rs::qc::{
    ConsensusStats, DepthStats, DroppedSequences, FilterCount, MAX_FILTER_FRACTION, MIN_MEAN_DEPTH,
    QcCode, QcContext, QcInput, QcReport, Severity, analyze, render_text,
};
use std::fs;
use std::path::PathBuf;

/// 2,000 called bins with 100 K bins, read with `context`.
fn qc(context: QcContext) -> QcReport {
    let mut row = vec![0u8; 2_000];
    row[..100].fill(1);
    let rows = vec![row];
    analyze(&QcInput {
        rows: &rows,
        row_starts: &[true],
        seqs: &[],
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context,
        callable: None,
    })
}

#[test]
fn approximate_missing_placement() {
    assert_eq!(
        qc(QcContext::default()).finding(QcCode::ApproxMissingPlacement),
        None
    );
    let r = qc(QcContext {
        approximate_missing_placement: true,
        ..QcContext::default()
    });
    let f = r.finding(QcCode::ApproxMissingPlacement).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert!(f.message.contains("placed together at its start"));
    assert!(f.recommendation.contains("callable-sites mask"));
    // Info comes after the warnings this tiny input triggers.
    assert_eq!(
        r.findings.last().unwrap().code,
        QcCode::ApproxMissingPlacement
    );
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("psmc_qc_{tag}_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The same calls as VCF + callable mask (exact) and as multihetsep (uncalled
/// sites placed at the start of each interval) give different bins, which is
/// what the APPROX_MISSING_PLACEMENT note says.
#[test]
fn multihetsep_placement_changes_which_bins_are_missing() {
    let dir = temp_dir("placement");
    // Callable: 1-150 and 400-1000; heterozygous SNPs at 120, 450, 455, 990.
    let vcf = dir.join("calls.vcf");
    let mut text = String::from(
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n",
    );
    for pos in [120, 450, 455, 990] {
        text.push_str(&format!("chr1\t{pos}\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n"));
    }
    fs::write(&vcf, text).unwrap();
    let mask = dir.join("callable.bed");
    fs::write(&mask, "chr1\t0\t150\nchr1\t399\t1000\n").unwrap();
    // The multihetsep rows count called sites since the previous row:
    // 1-120: 120; 121-450: 30 + 51 = 81; 451-455: 5; 456-990: 535.
    let mhs = dir.join("calls.multihetsep");
    fs::write(
        &mhs,
        "chr1\t120\t120\tAC\nchr1\t450\t81\tAC\nchr1\t455\t5\tAC\nchr1\t990\t535\tAC\n",
    )
    .unwrap();

    let exact = read_vcf_with(
        &vcf,
        &VcfOptions {
            masks: vec![mask],
            ..VcfOptions::default()
        },
    )
    .unwrap()
    .obs;
    let placed = read_mhs(&mhs, None, 100).unwrap();
    // Bin 301-400 holds 99 uncalled sites (301-399): N. Multihetsep puts the
    // 249 uncalled sites of 121-450 at 121-369, leaving 69 in that bin: T.
    assert_eq!(exact.rows, [vec![0, 1, 2, 2, 1, 0, 0, 0, 0, 1]]);
    assert_eq!(placed.rows, [vec![0, 1, 2, 0, 1, 0, 0, 0, 0, 1]]);
    let _ = fs::remove_dir_all(dir);
}

fn consensus(mean: f64, filters: &[(&str, u64)]) -> QcContext {
    // Depths 0..=40 with the last class open: 10 positions each at 0-9, 100
    // at 10-29, 10 at 30-39 and 20 at 40 or more (2,220 in all).
    let mut histogram = vec![10u64; 10];
    histogram.extend([100; 20]);
    histogram.extend([10; 10]);
    histogram.push(20);
    QcContext {
        consensus: Some(ConsensusStats {
            depth: DepthStats {
                mean,
                median: 20.0,
                min_depth: Some(7),
                max_depth: Some(36),
                histogram,
            },
            total_bases: 1_000_000,
            filters: filters
                .iter()
                .map(|&(name, bases)| FilterCount {
                    name: name.to_string(),
                    bases,
                })
                .collect(),
        }),
        ..QcContext::default()
    }
}

#[test]
fn depth_distribution_is_summarised() {
    let r = qc(consensus(
        20.0,
        &[("min depth", 50_000), ("max depth", 2_000)],
    ));
    let d = r.depth.as_ref().unwrap();
    assert_eq!((d.mean, d.median), (20.0, 20.0));
    assert_eq!((d.min_depth, d.max_depth), (Some(7), Some(36)));
    // 5% of 2,220 = 111: classes 0-9 hold 100, depth 10 reaches 200.
    // 25% = 555: 100 + 100 * 5 at depth 14. 75% = 1,665: 100 + 100 * 16 at
    // depth 25. 95% = 2,109: 2,100 + 10 at depth 30.
    assert_eq!(d.quantiles, Some([10, 14, 25, 30]));
    assert_eq!(d.below_min_fraction, Some(70.0 / 2220.0));
    // Above 36: depths 37, 38, 39 and the open class 40+.
    assert_eq!(d.above_max_fraction, Some(50.0 / 2220.0));
    assert_eq!(d.filters[0].fraction, 0.05);
    assert_eq!(d.filters[1].name, "max depth");
    assert_eq!(r.finding(QcCode::Depth), None);

    let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(v["depth"]["quantiles"][1], 14);
    assert_eq!(v["depth"]["histogram"].as_array().unwrap().len(), 41);
}

#[test]
fn depth_above_max_is_unknown_inside_the_open_class() {
    let mut ctx = consensus(20.0, &[]);
    let c = ctx.consensus.as_mut().unwrap();
    c.depth.max_depth = Some(40);
    let r = qc(ctx.clone());
    assert_eq!(r.depth.as_ref().unwrap().above_max_fraction, None);
    let c = ctx.consensus.as_mut().unwrap();
    c.depth.max_depth = Some(39);
    let r = qc(ctx.clone());
    assert_eq!(
        r.depth.as_ref().unwrap().above_max_fraction,
        Some(20.0 / 2220.0)
    );
    let c = ctx.consensus.as_mut().unwrap();
    c.depth.histogram.clear();
    c.depth.min_depth = None;
    let r = qc(ctx);
    let d = r.depth.as_ref().unwrap();
    assert_eq!(
        (d.quantiles, d.below_min_fraction, d.above_max_fraction),
        (None, None, None)
    );
}

#[test]
fn depth_below_min_is_unknown_inside_the_open_class() {
    // 41 classes: depths 0-39 exact, 20 positions in the open class 40+.
    let mut ctx = consensus(20.0, &[]);
    for (min_depth, below) in [
        (39, Some(2190.0 / 2220.0)),
        (40, Some(2200.0 / 2220.0)),
        // From 41 on, the open class may hold depths on either side.
        (41, None),
        (60, None),
    ] {
        ctx.consensus.as_mut().unwrap().depth.min_depth = Some(min_depth);
        let r = qc(ctx.clone());
        assert_eq!(
            r.depth.as_ref().unwrap().below_min_fraction,
            below,
            "min_depth {min_depth}"
        );
    }

    // Every position in the open class 9+, minimum depth 20: nothing is
    // known to lie below it.
    let (r, line) = depth_line([vec![0; 9], vec![1_000]].concat(), 20);
    let d = r.depth.as_ref().unwrap();
    assert_eq!((d.below_min_fraction, d.above_max_fraction), (None, None));
    assert!(!line.contains("below"), "{line}");
}

/// The report and its "Read depth" text line for `histogram` (last class
/// open) and `min_depth`, with a maximum depth of 60.
fn depth_line(histogram: Vec<u64>, min_depth: u32) -> (QcReport, String) {
    let r = qc(QcContext {
        consensus: Some(ConsensusStats {
            depth: DepthStats {
                mean: 30.0,
                median: 30.0,
                min_depth: Some(min_depth),
                max_depth: Some(60),
                histogram,
            },
            total_bases: 1_000,
            filters: Vec::new(),
        }),
        ..QcContext::default()
    });
    let line = render_text(&r)
        .lines()
        .find(|l| l.contains("Read depth"))
        .unwrap()
        .to_string();
    (r, line)
}

#[test]
fn depth_quantiles_in_the_open_class_are_lower_bounds() {
    // All 1,000 positions in the open class 9+.
    let (r, line) = depth_line([vec![0; 9], vec![1_000]].concat(), 5);
    assert_eq!(r.depth.as_ref().unwrap().quantiles, Some([9; 4]));
    assert!(line.contains("5-95% 9+-9+x;"), "{line}");
    // 100 positions at each depth 5-9 and 100 at 10 or more: only the 95%
    // quantile (570 of 600) lies in the open class.
    let (r, line) = depth_line([vec![0; 5], vec![100; 5], vec![100]].concat(), 5);
    assert_eq!(r.depth.as_ref().unwrap().quantiles, Some([5, 6, 9, 10]));
    assert!(line.contains("5-95% 5-10+x;"), "{line}");
    assert!(line.contains("(0.0% below)"), "{line}");
}

#[test]
fn low_mean_depth() {
    assert_eq!(MIN_MEAN_DEPTH, 18.0);
    assert_eq!(qc(consensus(18.0, &[])).finding(QcCode::Depth), None);
    let r = qc(consensus(17.9, &[]));
    let f = r.finding(QcCode::Depth).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert!(
        f.message
            .starts_with("Mean read depth is 17.9x, below the 18x")
    );
    assert!(f.recommendation.starts_with("Sequence deeper"));
    assert!(!f.message.contains("filters"));
}

#[test]
fn filter_removing_too_much() {
    assert_eq!(MAX_FILTER_FRACTION, 0.30);
    // 300,000 of 1,000,000 is exactly 30%.
    let ok = consensus(30.0, &[("min depth", 300_000), ("mapping quality", 10)]);
    assert_eq!(qc(ok).finding(QcCode::Depth), None);
    let r = qc(consensus(
        30.0,
        &[
            ("min depth", 300_001),
            ("mapping quality", 10),
            ("max depth", 450_000),
        ],
    ));
    let f = r.finding(QcCode::Depth).unwrap();
    assert_eq!(f.severity, Severity::Warning);
    assert_eq!(
        f.message,
        "Single filters removed more than 30.0% of the 1,000,000 bases: 'min depth' 30.0%, \
         'max depth' 45.0%."
    );
    assert!(
        f.recommendation.contains("(now 7-36 at a mean of 30.0x)"),
        "{}",
        f.recommendation
    );
    assert!(!f.message.contains("Mean read depth"));

    // Both problems: one finding.
    let r = qc(consensus(10.0, &[("min depth", 600_000)]));
    let found: Vec<_> = r
        .findings
        .iter()
        .filter(|f| f.code == QcCode::Depth)
        .collect();
    assert_eq!(found.len(), 1);
    assert!(found[0].message.starts_with("Mean read depth is 10.0x"));
    assert!(found[0].message.contains("'min depth' 60.0%"));
}

#[test]
fn dropped_sequences() {
    assert_eq!(
        qc(QcContext::default()).finding(QcCode::DroppedSequences),
        None
    );
    let r = qc(QcContext {
        dropped: vec![
            DroppedSequences {
                reason: "without any called site".to_string(),
                count: 3,
                bases: Some(2_500_000),
            },
            DroppedSequences {
                reason: "not in the mask".to_string(),
                count: 1,
                bases: None,
            },
        ],
        ..QcContext::default()
    });
    assert_eq!(r.dropped.len(), 2);
    let f = r.finding(QcCode::DroppedSequences).unwrap();
    assert_eq!(f.severity, Severity::Info);
    assert_eq!(
        f.message,
        "4 sequences were left out of the analysis: 3 without any called site (2.50 Mb); 1 \
         not in the mask."
    );
    let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
    assert_eq!(v["dropped"][0]["count"], 3);
    assert!(v["dropped"][1]["bases"].is_null());
}
