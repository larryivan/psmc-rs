//! Process-wide memory budget for keeping full forward (alpha) rows.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

const DEFAULT_LIMIT_MB: usize = 2048;

/// A memory budget for caching every forward (alpha) row of a sequence during
/// the E-step, shared by everything that reserves from it.
///
/// Without a reservation a sequence keeps only one alpha row every
/// [`CHECKPOINT_STRIDE`](super::CHECKPOINT_STRIDE) sites and recomputes the
/// rest during the backward pass. Recomputed rows are bit-identical to stored
/// ones, so the budget only trades memory for speed; results never depend on it.
///
/// Rows are cached only where recomputing them is expensive: where the
/// forward step is dense, O(n²), as for transition matrices without PSMC's
/// semiseparable structure and for models of more than 256 states. Other
/// PSMC models reserve nothing.
///
/// Reservations are atomic, so concurrent E-steps (sequences processed in
/// parallel, bootstrap replicates) together never exceed the limit.
#[derive(Debug)]
pub struct AlphaCacheBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
}

impl AlphaCacheBudget {
    /// A budget of `limit_bytes` (0 disables caching).
    pub const fn new(limit_bytes: usize) -> Self {
        Self {
            limit: limit_bytes,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    /// The process-wide budget used when no other is given: `PSMC_ALPHA_CACHE_MB`
    /// MiB (read once; 0 disables caching), 2048 MiB if unset or invalid.
    pub fn global() -> &'static AlphaCacheBudget {
        static GLOBAL: OnceLock<AlphaCacheBudget> = OnceLock::new();
        GLOBAL.get_or_init(|| {
            let mb = std::env::var("PSMC_ALPHA_CACHE_MB")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(DEFAULT_LIMIT_MB);
            AlphaCacheBudget::new(mb.saturating_mul(1024 * 1024))
        })
    }

    /// Total bytes this budget allows.
    pub fn limit_bytes(&self) -> usize {
        self.limit
    }

    /// Bytes currently reserved.
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// The most bytes reserved at any one time so far.
    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Reserve `bytes` if they fit; the reservation is returned on drop.
    pub(crate) fn try_reserve(&self, bytes: usize) -> Option<Reservation<'_>> {
        // fetch_update's loop: Rust 1.99 deprecates fetch_update for
        // try_update, stable only since 1.95 (the crate's minimum is 1.88).
        let mut used = self.used.load(Ordering::Relaxed);
        while let Err(now) = self.used.compare_exchange_weak(
            used,
            used.checked_add(bytes).filter(|&v| v <= self.limit)?,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            used = now;
        }
        self.peak.fetch_max(used + bytes, Ordering::Relaxed);
        Some(Reservation {
            budget: self,
            bytes,
        })
    }
}

/// Bytes held from an [`AlphaCacheBudget`] until dropped.
#[derive(Debug)]
pub(crate) struct Reservation<'a> {
    budget: &'a AlphaCacheBudget,
    bytes: usize,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
