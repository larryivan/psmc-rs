//! Starting parameters whose transition probabilities are negative: PSMC's
//! model allows at most about one recombination per bin, and above a limit
//! on rho the probability of staying in the oldest states turns negative.
//! EM cannot move from such a start (C psmc stays at it and reports a flat
//! history), so it must refuse it and say how far rho has to come down.

mod common;

use common::{e_step, model, mstep};
use psmc_rs::initial_theta;
use psmc_rs::model::NegativeTransitions;
use psmc_rs::opt::{MStepConfig, em_train, m_step};

/// `n_seqs` sequences of `len` bins, each heterozygous with probability
/// `p_het`, from a fixed xorshift stream.
fn rows(p_het: f64, n_seqs: usize, len: usize) -> (Vec<Vec<u8>>, Vec<bool>) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let rows: Vec<Vec<u8>> = (0..n_seqs)
        .map(|_| (0..len).map(|_| u8::from(next() < p_het)).collect())
        .collect();
    (rows, vec![true; n_seqs])
}

fn config() -> MStepConfig {
    MStepConfig {
        max_iters: 30,
        ..mstep()
    }
}

/// The limit on rho and the smallest theta/rho ratio that the error of
/// `check_transitions` gives, from its fields; its message gives them too,
/// and names no command-line option, which a library caller does not have.
fn advice(err: &anyhow::Error) -> (f64, f64) {
    let negative = err
        .downcast_ref::<NegativeTransitions>()
        .unwrap_or_else(|| panic!("{err:#}"));
    let (limit, ratio) = (negative.max_rho.unwrap(), negative.min_tr_ratio().unwrap());
    let message = format!("{err:#}");
    assert!(
        message.contains(&format!("rho must be below {limit:.6}"))
            && message.contains(&format!("a theta/rho ratio of {ratio} or more")),
        "{message}"
    );
    assert!(!message.contains("--"), "{message}");
    (limit, ratio)
}

#[test]
fn em_refuses_a_start_with_negative_transitions() {
    // 60% of bins heterozygous: C's start (theta0 = -ln(1 - K/L) = 0.92,
    // rho0 = theta0 / 4 = 0.23) is past the limit of 1/6 for t_max 15.
    let (rows, starts) = rows(0.6, 3, 20_000);
    let theta0 = initial_theta(&rows).unwrap();
    let start = model("4+5*3+4", theta0, theta0 / 4.0);
    assert!(start.transition_matrix().iter().any(|&p| p < 0.0));

    let mut m = start.clone();
    let err = em_train(&mut m, &rows, &starts, 3, &config()).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.starts_with(
            "EM cannot start from these parameters: the transition probabilities are negative"
        ),
        "{message}"
    );
    let (limit, ratio) = advice(&err);
    assert!((limit - 1.0 / 6.0).abs() < 1e-6, "{message}");

    // The advice works: from the suggested ratio EM fits, and the history
    // is no longer flat.
    assert!(theta0 / ratio < limit);
    let mut m = model("4+5*3+4", theta0, theta0 / ratio);
    let history = em_train(&mut m, &rows, &starts, 3, &config()).unwrap();
    assert!(history.mstep_iterations.iter().all(|&n| n > 0));
    assert!(
        m.lam.iter().any(|&l| (l - m.lam[0]).abs() > 1e-3),
        "{:?}",
        m.lam
    );

    // Without iterations nothing is fitted, so nothing is refused.
    let mut m = start.clone();
    assert!(em_train(&mut m, &rows, &starts, 0, &config()).is_ok());
}

#[test]
fn the_limit_is_where_transitions_turn_negative() {
    for pattern in ["4+5*3+4", "4+25*2+4+6"] {
        let theta = 0.9;
        let err = model(pattern, theta, 0.3).check_transitions().unwrap_err();
        let (limit, ratio) = advice(&err);
        let below = model(pattern, theta, limit * (1.0 - 1e-5));
        assert!(below.check_transitions().is_ok(), "{pattern}: {limit}");
        assert!(below.transition_matrix().iter().all(|&p| p >= 0.0));
        let above = model(pattern, theta, limit * (1.0 + 1e-5));
        assert!(above.check_transitions().is_err(), "{pattern}: {limit}");
        // The ratio is the smallest with 2 decimals below the limit.
        assert!(theta / ratio < limit && theta / (ratio - 0.01) >= limit * (1.0 - 1e-5));
    }
    // C's start for ordinary data passes.
    assert!(model("4+5*3+4", 0.01, 0.0025).check_transitions().is_ok());
}

#[test]
fn m_step_refuses_a_start_where_its_objective_is_infeasible() {
    let (rows, starts) = rows(0.3, 1, 20_000);
    let valid = model("4+5*3+4", 0.36, 0.09);
    let stats = e_step(&valid, &rows, &starts).stats;
    let invalid = model("4+5*3+4", 0.9, 0.3);
    let err = m_step(&invalid, &stats, &config(), None).unwrap_err();
    // Negative transition probabilities: no HMM, so the objective is +inf.
    assert!(err.to_string().contains("objective is inf"), "{err}");
}
