//! C psmc's EM Q function (`hmm_Q`, `hmm_Q0` in khmm.c) and `post_sigma`
//! on small hand-made expectations.

use ndarray::{Array2, array};
use psmc_rs::em::{hmm_q, hmm_q0, posterior_sigma};
use psmc_rs::hmm::SufficientStats;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14 * a.abs().max(b.abs()).max(1.0)
}

/// Two states; one zero emission count and missing-symbol counts that must
/// be ignored.
fn stats() -> SufficientStats {
    SufficientStats {
        gobs: [vec![3.0, 0.5], vec![1.0, 0.0], vec![7.0, 2.0]],
        xi: array![[2.0, 1.0], [0.25, 4.0]],
    }
}

fn emission() -> Array2<f64> {
    array![[0.9, 0.6], [0.1, 0.4], [1.0, 1.0]]
}

fn transition() -> Array2<f64> {
    array![[0.7, 0.3], [0.2, 0.8]]
}

#[test]
fn q_is_the_sum_of_count_times_log_probability() {
    let ln = f64::ln;
    let direct = 3.0 * ln(0.9)
        + 0.5 * ln(0.6)
        + 1.0 * ln(0.1)
        + 0.0 * ln(0.4)
        + 2.0 * ln(0.7)
        + 1.0 * ln(0.3)
        + 0.25 * ln(0.2)
        + 4.0 * ln(0.8);
    let q = hmm_q(&transition(), &emission(), &stats());
    assert!(close(q, direct), "{q} vs {direct}");
}

#[test]
fn q0_is_q_at_the_normalised_counts() {
    let ln = f64::ln;
    // State 1 emits only symbol 0, so its symbol-1 term is 0 ln 0 = 0.
    let direct = 3.0 * ln(3.0 / 4.0)
        + 1.0 * ln(1.0 / 4.0)
        + 0.5 * ln(1.0)
        + 2.0 * ln(2.0 / 3.0)
        + 1.0 * ln(1.0 / 3.0)
        + 0.25 * ln(0.25 / 4.25)
        + 4.0 * ln(4.0 / 4.25);
    let q0 = hmm_q0(&stats());
    assert!(close(q0, direct), "{q0} vs {direct}");

    // hmm_q reaches hmm_q0 at the normalised counts (all positive here, as
    // hmm_q rejects a zero probability even where the count is zero).
    let mut s = stats();
    s.gobs[1][1] = 0.5;
    let best_e = array![[0.75, 0.5], [0.25, 0.5], [1.0, 1.0]];
    let best_a = array![[2.0 / 3.0, 1.0 / 3.0], [0.25 / 4.25, 4.0 / 4.25]];
    assert!(close(hmm_q(&best_a, &best_e, &s), hmm_q0(&s)));
}

#[test]
fn q_minus_q0_is_minus_the_expected_kl_divergence() {
    // Q - Q0 = -sum_k n_k KL(ê_k || e_k) - sum_k m_k KL(â_k || a_k), with n_k
    // and m_k the emission and transition counts of state k and ê, â the
    // normalised counts: a second, independent evaluation, and why C's
    // printed QD values are never positive.
    let s = SufficientStats {
        gobs: [
            vec![10.0, 0.3, 2.5],
            vec![0.7, 4.0, 2.5],
            vec![1.0, 1.0, 1.0],
        ],
        xi: array![[5.0, 0.5, 0.1], [0.2, 3.0, 0.9], [0.0, 1.5, 6.0]],
    };
    let e = array![[0.95, 0.2, 0.5], [0.05, 0.8, 0.5], [1.0, 1.0, 1.0]];
    let a = array![[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.05, 0.25, 0.7]];
    let kl = |counts: &[f64], probs: &[f64]| -> f64 {
        let n: f64 = counts.iter().sum();
        counts
            .iter()
            .zip(probs)
            .filter(|(c, _)| **c > 0.0)
            .map(|(c, p)| c * ((c / n) / p).ln())
            .sum()
    };
    let mut expected = 0.0;
    for k in 0..3 {
        expected -= kl(&[s.gobs[0][k], s.gobs[1][k]], &[e[(0, k)], e[(1, k)]]);
        expected -= kl(
            s.xi.row(k).as_slice().unwrap(),
            a.row(k).as_slice().unwrap(),
        );
    }
    let q = hmm_q(&a, &e, &s) - hmm_q0(&s);
    assert!(q < 0.0);
    assert!((q - expected).abs() < 1e-12, "{q} vs {expected}");
}

#[test]
fn missing_symbol_and_initial_state_are_left_out() {
    let mut s = stats();
    let (q, q0) = (hmm_q(&transition(), &emission(), &s), hmm_q0(&s));
    s.gobs[2] = vec![1e6, 0.0];
    let mut e = emission();
    e[(2, 0)] = 0.5;
    assert_eq!(hmm_q(&transition(), &e, &s), q);
    assert_eq!(hmm_q0(&s), q0);
}

#[test]
fn non_positive_probabilities_give_minus_infinity() {
    let mut a = transition();
    a[(1, 0)] = 0.0;
    assert_eq!(hmm_q(&a, &emission(), &stats()), f64::NEG_INFINITY);
    let mut e = emission();
    e[(0, 1)] = -1e-17;
    assert_eq!(hmm_q(&transition(), &e, &stats()), f64::NEG_INFINITY);
}

#[test]
fn posterior_sigma_is_the_normalised_non_missing_count() {
    let p = posterior_sigma(&stats());
    assert_eq!(p, vec![4.0 / 4.5, 0.5 / 4.5]);
}
