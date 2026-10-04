//! Pure formulas behind the QC metrics, kept apart so that each can be checked
//! on its own.

/// C psmc's initial θ0 per bin, `-ln(1 - K/(T+K))` (`psmc_new_data` in
/// `core.c`, and [`crate::model::initial_theta`]): the Poisson rate whose
/// probability of at least one heterozygous site matches the observed share
/// of heterozygous bins among called bins.
///
/// `None` when no bin is called, or when every called bin is heterozygous (the
/// estimate is infinite).
pub fn theta0_per_bin(t_bins: u64, k_bins: u64) -> Option<f64> {
    let called = t_bins + k_bins;
    if called == 0 || k_bins == called {
        return None;
    }
    // Same expression as C, so the value is bit-identical to its θ0; `+ 0.0`
    // turns the -0.0 of K = 0 into 0.0.
    Some(-(1.0 - k_bins as f64 / called as f64).ln() + 0.0)
}

/// C psmc's `n_recomb` at round 0: the expected number of recombinations in
/// `called_bins` non-missing bins under its starting model (λ = 1,
/// ρ0 = θ0 / `tr_ratio`).
///
/// With λ = 1, `C_pi` = 1 and `C_sigma` = 1 / ρ0 + 1/2 (`psmc_update_hmm`),
/// and C prints `n_recomb = sum_L / C_sigma` (`psmc_print_data`). Each
/// recombination starts a new genealogical segment, so this is roughly the
/// number of independent TMRCA segments the data hold before fitting.
pub fn initial_recombinations(called_bins: u64, theta0: f64, tr_ratio: f64) -> f64 {
    called_bins as f64 / (tr_ratio / theta0 + 0.5)
}

/// N50 and L50 of `lengths`: the largest length `L` such that sequences at
/// least `L` long hold at least half of the total, and how many sequences
/// that takes. `(0, 0)` when the total is 0.
pub fn n50(lengths: &[u64]) -> (u64, usize) {
    let total: u128 = lengths.iter().map(|&l| u128::from(l)).sum();
    if total == 0 {
        return (0, 0);
    }
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    let mut cum: u128 = 0;
    for (i, &len) in sorted.iter().enumerate() {
        cum += u128::from(len);
        if 2 * cum >= total {
            return (len, i + 1);
        }
    }
    unreachable!("the cumulative sum reaches the total")
}

/// Number of trunks psmc's `splitfa` (and
/// [`crate::bootstrap::split_for_bootstrap`]) cuts a sequence of `len` bins
/// into at trunk size `trunk`: full trunks while at least `trunk * 3 / 2`
/// bins remain, then the remainder as one trunk. 0 when `trunk` is 0.
pub fn bootstrap_trunks_of(len: u64, trunk: u64) -> u64 {
    if len == 0 || trunk == 0 {
        return 0;
    }
    let tail = trunk.saturating_mul(3) / 2;
    if len < tail {
        return 1;
    }
    let full = (len - tail) / trunk + 1;
    full + u64::from(len > full * trunk)
}
