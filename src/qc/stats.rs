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

/// Largest trunk size (in bins) at which sequences shorter than one trunk
/// hold at most `max_fraction` (in `[0, 1)`) of all bins, given the
/// sequence lengths in bins: the length of the sequence at which the running
/// total, shortest first, first exceeds that share. 0 when there are no bins.
pub fn largest_trunk_within(lengths: &[u64], max_fraction: f64) -> u64 {
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    let total: u128 = sorted.iter().map(|&l| u128::from(l)).sum();
    let limit = max_fraction * total as f64;
    let mut cum: u128 = 0;
    for &len in &sorted {
        cum += u128::from(len);
        if cum as f64 > limit {
            return len;
        }
    }
    sorted.last().copied().unwrap_or(0)
}

/// Median of `xs` (the mean of the two middle values for an even count),
/// sorting `xs` in place; NaN for an empty slice.
pub(crate) fn median(xs: &mut [f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_unstable_by(f64::total_cmp);
    let mid = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[mid]
    } else {
        0.5 * (xs[mid - 1] + xs[mid])
    }
}

/// Per-site heterozygosity `h` implied by a share `p` of heterozygous bins
/// among called bins of `bin_size` sites, when every site is heterozygous
/// independently with probability `h`: a bin is `K` unless all its sites are
/// homozygous, so `p = 1 - (1-h)^s` and `h = 1 - (1-p)^(1/s)`.
///
/// `p` is clamped to `[0, 1]`.
pub fn het_per_site(p: f64, bin_size: usize) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // 1 - exp(ln(1-p)/s), without cancellation for small p.
    -((-p).ln_1p() / bin_size as f64).exp_m1()
}

/// Expected share of heterozygous bins that hold two or more heterozygous
/// sites, for bins of `bin_size` sites that are each heterozygous
/// independently with probability `h`.
///
/// The number of heterozygous sites in a bin is X ~ Binomial(s, h), so
/// P(X >= 1) = 1 - (1-h)^s, P(X = 1) = s h (1-h)^(s-1), and the share is
/// P(X >= 2 | X >= 1) = 1 - s h (1-h)^(s-1) / (1 - (1-h)^s).
/// For small `h` it is close to `(s - 1) h / 2`. It is 0 for `h = 0` (the limit)
/// or `s <= 1`, and 1 for `h = 1` with `s >= 2`.
pub fn multi_het_share(h: f64, bin_size: usize) -> f64 {
    if bin_size <= 1 || h <= 0.0 {
        return 0.0;
    }
    if h >= 1.0 {
        return 1.0;
    }
    let s = bin_size as f64;
    let ln_hom = (-h).ln_1p();
    let p_any = -(s * ln_hom).exp_m1();
    let p_one = s * h * ((s - 1.0) * ln_hom).exp();
    (1.0 - p_one / p_any).max(0.0)
}

/// Largest bin size whose [`multi_het_share`] at per-site heterozygosity `h`
/// stays below `target` (in `(0, 1)`). The share grows with the bin size, and
/// a 1-site bin never holds two sites, so the answer is at least 1. `None`
/// when no practical bin size reaches `target` (`h` = 0, or `h` so small that
/// it would take more than 2^52 sites).
///
/// # Panics
///
/// If `target` is not in `(0, 1)`.
pub fn max_bin_size_for_share(h: f64, target: f64) -> Option<usize> {
    assert!(target > 0.0 && target < 1.0, "target must be in (0, 1)");
    const LIMIT: usize = 1 << 52;
    let below = |s: usize| multi_het_share(h, s) < target;
    let mut hi = 2usize;
    while below(hi) {
        if hi >= LIMIT {
            return None;
        }
        hi *= 2;
    }
    // below(lo) holds (it does for 1) and below(hi) does not.
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if below(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}
