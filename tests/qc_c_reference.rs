//! QC totals against C psmc (lh3/psmc 0.6.5-r74).
//!
//! `tests/data/qc_c_reference/c_psmc_round0.tsv` holds what `psmc -N 0`
//! printed for a synthetic multi-record psmcfa, the 180,000-bin Papuan
//! excerpt and the four real-data inputs of `experiment/real_data`, for the
//! whole files and for single records (see `generate.sh` there): the number
//! of sequences, `sum_L` (non-missing bins), `sum_n` (heterozygous bins), the
//! round-0 θ0 (`%.9lf`) and `n_recomb` (`%lf`).

use psmc_rs::io::psmcfa::{PsmcfaRecord, read_psmcfa_records};
use psmc_rs::qc::{QcContext, QcInput, QcReport, QcSeq, analyze, initial_recombinations};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

struct CRow {
    record: String,
    n_seqs: usize,
    sum_l: u64,
    sum_n: u64,
    theta0: String,
    n_recomb: String,
}

fn c_rows() -> BTreeMap<String, Vec<CRow>> {
    let text = std::fs::read_to_string("tests/data/qc_c_reference/c_psmc_round0.tsv").unwrap();
    let mut out: BTreeMap<String, Vec<CRow>> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 7, "bad row {line:?}");
        out.entry(f[0].to_string()).or_default().push(CRow {
            record: f[1].to_string(),
            n_seqs: f[2].parse().unwrap(),
            sum_l: f[3].parse().unwrap(),
            sum_n: f[4].parse().unwrap(),
            theta0: f[5].to_string(),
            n_recomb: f[6].to_string(),
        });
    }
    out
}

/// `printf("%.*f")` as C prints it, with C's `-0` for negative zero.
fn c_fixed(v: f64, digits: usize) -> String {
    let s = format!("{v:.digits$}");
    if v == 0.0 && v.is_sign_negative() {
        format!("-{s}")
    } else {
        s
    }
}

/// Equal as printed, except that C prints -0 for an all-T record
/// (θ0 = -ln(1)) where the QC reports +0.
fn same_number(ours: String, c: &str) -> bool {
    ours == c
        || (ours.trim_start_matches(['0', '.']).is_empty() && c.strip_prefix('-') == Some(&ours))
}

fn qc_of(records: &[PsmcfaRecord]) -> QcReport {
    let rows: Vec<Vec<u8>> = records
        .iter()
        .filter(|r| !r.seq.is_empty())
        .map(|r| r.seq.clone())
        .collect();
    let starts = vec![true; rows.len()];
    let seqs: Vec<QcSeq> = records
        .iter()
        .map(|r| QcSeq::new(r.name.clone(), r.seq.len()))
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

fn check(label: &str, c: &CRow, n_seqs: usize, t: u64, k: u64, theta: Option<f64>) {
    assert_eq!(n_seqs, c.n_seqs, "{label}: n_seqs");
    assert_eq!(t + k, c.sum_l, "{label}: sum_L");
    assert_eq!(k, c.sum_n, "{label}: sum_n");
    let theta = theta.expect("theta0");
    assert!(
        same_number(c_fixed(theta, 9), &c.theta0),
        "{label}: theta0 {theta:.9} vs C {}",
        c.theta0
    );
    let n_recomb = initial_recombinations(t + k, theta, 4.0);
    assert!(
        same_number(c_fixed(n_recomb, 6), &c.n_recomb),
        "{label}: n_recomb {n_recomb:.6} vs C {}",
        c.n_recomb
    );
}

#[test]
fn totals_and_records_match_c_psmc() {
    let table = c_rows();
    assert_eq!(table.len(), 6, "inputs in the reference table");
    // The inputs are independent: read and check them on every core.
    let records_checked: usize = table
        .par_iter()
        .map(|(path, rows)| {
            let records = read_psmcfa_records(Path::new(path)).unwrap();
            let report = qc_of(&records);
            let mut records_checked = 0;
            for c in rows {
                if c.record == "*" {
                    let s = &report.totals;
                    let n = report.n_sequences + report.empty_sequences;
                    check(path, c, n, s.t_bins, s.k_bins, s.theta0_per_bin);
                    let ours = report.expected_recombinations.unwrap();
                    assert!(same_number(c_fixed(ours, 6), &c.n_recomb), "{path}");
                } else {
                    let i = records.iter().position(|r| r.name == c.record).unwrap();
                    let row = report
                        .sequences
                        .iter()
                        .find(|s| s.name == c.record)
                        .unwrap();
                    assert_eq!(row.stats.bins as usize, records[i].seq.len());
                    let s = &row.stats;
                    let label = format!("{path}:{}", c.record);
                    check(&label, c, 1, s.t_bins, s.k_bins, s.theta0_per_bin);
                    records_checked += 1;
                }
            }
            records_checked
        })
        .sum();
    assert_eq!(records_checked, 6 + 3 + 34);
}
