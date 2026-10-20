//! Plain-text rendering of the QC report.

use psmc_rs::qc::{
    ConsensusStats, DepthStats, DroppedSequences, FilterCount, QcContext, QcInput, QcReport, QcSeq,
    TextOptions, analyze, render_text, render_text_with,
};

/// The report of `rows`, one sequence each, described by `seqs`, in bins of
/// 100 bp with the default context and no bootstrap.
fn qc(rows: &[Vec<u8>], seqs: &[QcSeq]) -> QcReport {
    analyze(&QcInput {
        rows,
        row_starts: &vec![true; rows.len()],
        seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 0,
        context: QcContext::default(),
        callable: None,
    })
}

/// Three sequences (one hemizygous-looking chrX), consensus depth stats and a
/// dropped sequence.
fn report() -> QcReport {
    let mut chr1 = vec![1u8; 1_000];
    chr1.resize(19_000, 0);
    chr1.resize(20_000, 2);
    let mut chr2 = vec![1u8; 600];
    chr2.resize(15_000, 0);
    let mut chr_x = vec![1u8; 10];
    chr_x.resize(10_000, 0);
    let rows = vec![chr1, chr2, chr_x];
    let mut seqs = vec![
        QcSeq::new("chr1", 20_000),
        QcSeq::new("chr2", 15_000),
        QcSeq::new("chrX", 10_000),
    ];
    seqs[0].extent_bases = Some(1_999_950);
    let mut histogram = vec![0u64; 8];
    histogram.extend([1_000; 30]);
    histogram.push(100);
    analyze(&QcInput {
        rows: &rows,
        row_starts: &[true, true, true],
        seqs: &seqs,
        bin_size: 100,
        bootstrap_trunk_bins: 10_000,
        context: QcContext {
            input_format: Some("vcf".to_string()),
            consensus: Some(ConsensusStats {
                depth: DepthStats {
                    mean: 22.5,
                    median: 22.0,
                    min_depth: Some(8),
                    max_depth: Some(45),
                    histogram,
                },
                total_bases: 4_600_000,
                filters: vec![
                    FilterCount {
                        name: "min depth".to_string(),
                        bases: 46_000,
                    },
                    FilterCount {
                        name: "max depth".to_string(),
                        bases: 2_000_000,
                    },
                ],
            }),
            dropped: vec![DroppedSequences {
                reason: "without any called site".to_string(),
                count: 2,
                bases: Some(35_000),
            }],
            ..QcContext::default()
        },
        callable: None,
    })
}

const EXPECTED: &str = r#"Input QC (vcf): 3 sequences, 45,000 bins of 100 bp
3 warnings, 1 note

WARNING  TOO_SHORT
  Only 4.40 Mb of called sequence (bases in called bins), less than 100 Mb. PSMC was developed for
  whole genomes; below ~100 Mb, even at human-like heterozygosity, the parameters for the oldest
  times get fewer than the ~10 recombinations each that psmc's documentation asks for, so the ends
  of the curve are poorly determined.
  -> Use more of the genome if possible (all autosomes, a less strict mask); otherwise check the
     curve with bootstrap replicates and read its ends with caution.

WARNING  SEX_OR_ORGANELLE
  1 sequence is named like a sex chromosome or organelle: chrX (sex chromosome, 0.021x the autosomal
  heterozygosity, hemizygous?). chrX is far less heterozygous than the autosomes, as a single
  (hemizygous) copy is; PSMC reads such a sequence as a long stretch of recent common ancestry,
  which lowers theta and adds a false signal of small recent Ne.
  -> Exclude chrX (keep autosomes only, e.g. by restricting the mask, VCF or psmcfa to them).

WARNING  DEPTH
  Single filters removed more than 30.0% of the 4,600,000 bases: 'max depth' 43.5%.
  -> Check those filters' thresholds against the data: psmc's README suggests depth limits of a
     third and twice the mean depth (now 8-45 at a mean of 22.5x); a strict mapping-quality cut-off
     removes much of a divergent reference.

INFO     DROPPED_SEQUENCES
  2 sequences were left out of the analysis: 2 without any called site (35.0 kb).
  -> Check that these sequences were meant to be left out (for example, chromosome names in the mask
     or VCF that do not match).

Metrics
  Bins              T 42,390 (94.2%), K 1,610 (3.6%), N 1,000 (2.2%)
  theta0            0.0373 per bin, 0.000373 per bp (0.373 per kb)
  Called length     4.40 Mb in called bins
  Sequence lengths  N50 1.50 Mb (L50 2), longest 1.99 Mb; 0.0% of bins in sequences < 1 Mb
  Bootstrap         5 trunks of 10,000 bins; 0.0% of bins in sequences shorter than a trunk
  Longest N run     1,000 bins (100 kb) in chr1 from bin 19,000
  Bin saturation    about 1.8% of K bins hold >= 2 het sites (0.373 per kb); below 5.0% up to 273 bp per bin
  Autosomal het     0.474 per kb (median of 2 sequences)
  Recombinations    about 408 expected by C psmc's starting model (theta/rho = 4)
  Read depth        mean 22.5x, median 22.0x, 5-95% 9-36x; kept 8-45x (0.0% below)
  Filters           min depth 1.0%, max depth 43.5% of 4,600,000 bases removed
  Dropped           2 without any called site

Sequences (3 of 3, input order)
  sequence   bins   T%  K%  N% het/kb vs ref longest N kind
  chr1     20,000 90.0 5.0 5.0  0.541   1.14     1,000
  chr2     15,000 96.0 4.0 0.0  0.408   0.86         0
  chrX     10,000 99.9 0.1 0.0 0.0100   0.02         0 sex chromosome
"#;

#[test]
fn full_text_matches_the_reference_layout() {
    let r = report();
    assert_eq!(render_text(&r), EXPECTED);
    assert_eq!(render_text(&r), render_text(&report()));
}

#[test]
fn called_length_shows_the_uncalled_site_share() {
    let mut seq = QcSeq::new("chr1", 1_000);
    seq.extent_bases = Some(100_000);
    seq.called_bases = Some(40_000);
    seq.het_sites = Some(80);
    let text = render_text(&qc(&[vec![0u8; 1_000]], &[seq]));
    let line = text.lines().find(|l| l.contains("Called length")).unwrap();
    assert!(
        line.ends_with(
            "100 kb in called bins; 40.0 kb called sites (60.0% uncalled), 2.00 het sites per kb"
        ),
        "{line}"
    );
}

#[test]
fn warnings_come_before_notes() {
    let text = render_text(&report());
    let last_warning = text.rfind("\nWARNING ").unwrap();
    let first_info = text.find("\nINFO ").unwrap();
    assert!(last_warning < first_info);
    assert!(text.find("\nWARNING ").unwrap() < text.find("\nMetrics").unwrap());
}

#[test]
fn brief_text_has_no_sequence_table() {
    let r = report();
    let brief = render_text_with(&r, &TextOptions::brief());
    assert!(!brief.contains("Sequences ("));
    let full = render_text(&r);
    assert!(full.starts_with(&brief));
    assert!(full[brief.len()..].starts_with("\nSequences (3 of 3, input order)"));
}

#[test]
fn findings_wrap_at_the_requested_width() {
    let r = report();
    for width in [60, 80, 100] {
        let text = render_text_with(
            &r,
            &TextOptions {
                sequence_rows: 0,
                width,
            },
        );
        let findings = &text[..text.find("\nMetrics").unwrap()];
        for line in findings.lines() {
            assert!(line.len() <= width, "{width}: {line:?}");
        }
        // Wrapping only moves line breaks.
        let words = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        let msg = &r.findings[0].message;
        let joined = words(findings).join(" ");
        assert!(joined.contains(&words(msg).join(" ")));
    }
}

#[test]
fn table_rows_prefer_named_sequences() {
    // 40 autosomes (the later ones longer), a cold one and a short chrX.
    let mut rows: Vec<Vec<u8>> = (0..40)
        .map(|i| {
            let mut r = vec![1u8; 600 + 15 * i];
            r.resize(12_000 + 300 * i, 0);
            r
        })
        .collect();
    rows.insert(5, vec![0u8; 15_000]);
    rows.push(vec![0u8; 500]);
    let mut seqs: Vec<QcSeq> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| QcSeq::new(format!("s{i}"), r.len()))
        .collect();
    seqs[5].name = "cold".to_string();
    seqs[41].name = "chrX".to_string();
    let r = qc(&rows, &seqs);
    let text = render_text_with(
        &r,
        &TextOptions {
            sequence_rows: 4,
            width: 100,
        },
    );
    let table = &text[text.find("\nSequences (").unwrap()..];
    let names: Vec<&str> = table
        .lines()
        .skip(3)
        .take(4)
        .map(|l| l.split_whitespace().next().unwrap())
        .collect();
    // Named in findings first (cold: HET_OUTLIER, chrX: SEX_OR_ORGANELLE),
    // then the two longest, all in input order.
    assert_eq!(names, ["cold", "s39", "s40", "chrX"]);
    assert!(table.starts_with("\nSequences (4 of 42, input order)"));
    let hidden_bins: u64 = r.totals.bins - [15_000u64, 23_400, 23_700, 500].iter().sum::<u64>();
    assert!(
        table.contains(&format!(
            "  ... 38 more sequences with {} bins (included in the totals)",
            thousands(hidden_bins)
        )),
        "{table}"
    );
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[test]
fn reports_without_findings_or_data_render() {
    let mut r = report();
    r.findings.clear();
    let text = render_text(&r);
    assert!(text.contains("\nNo warnings or notes.\n"));

    let text = render_text(&qc(&[], &[]));
    assert!(text.starts_with("Input QC: 0 sequences, 0 bins of 100 bp\n"));
    assert!(text.contains("  theta0            n/a per bin"), "{text}");
    assert!(text.contains("  Longest N run     none"), "{text}");
    assert!(!text.contains("Sequences ("));
}
