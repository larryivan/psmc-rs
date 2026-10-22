//! Which build of the per-site kernels the HMM passes run.
//!
//! On x86-64 the per-sequence passes are compiled twice: for the target as
//! given (SSE2 only in the portable release binary) and with AVX2 and FMA,
//! which runs when the CPU has both unless `PSMC_RS_SIMD=portable` (read
//! once, for tests and benchmarks) or [`set_force_portable`] says otherwise.
//! AVX-512 is not used: on AMD Zen 5 it was slower than AVX2. aarch64 always
//! runs its NEON kernels; other targets the portable ones.
//!
//! Each build fixes at compile time, by the `FUSED` parameter of the passes,
//! whether the multiply-adds of the O(states) forward and backward products
//! and transition sums are fused: in the AVX2 and FMA build and on aarch64, where it is one
//! instruction, but not in the portable build ([`DEFAULT_FUSED`]), where
//! `f64::mul_add` would call libm's `fma` (a fit took 3.75 times as long on
//! AMD Zen 5, and CPUs without FMA emulate it in software). Nothing else
//! fuses (Rust never contracts `a * b + c`) and no sum changes its order
//! with the vector width, so the AVX2 and FMA build gives the bits of the
//! portable code instantiated fused (and of an `x86-64-v3` build, which
//! takes the same path); the portable build differs in the last digits.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

/// The kernels the HMM passes run ([`simd_path`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdPath {
    /// Compiled for the target's own features, with unfused multiply-adds:
    /// on x86-64, the kernels of CPUs without AVX2 and FMA, and those that
    /// `PSMC_RS_SIMD=portable` forces.
    Portable,
    /// x86-64 code compiled with AVX2 and FMA, picked at run time.
    Avx2Fma,
    /// The NEON kernels of aarch64.
    Neon,
}

impl SimdPath {
    /// `portable`, `avx2+fma` or `neon`, as the output JSON records it.
    pub fn name(self) -> &'static str {
        match self {
            SimdPath::Portable => "portable",
            SimdPath::Avx2Fma => "avx2+fma",
            SimdPath::Neon => "neon",
        }
    }
}

impl fmt::Display for SimdPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

const UNREAD: u8 = 0;
const AUTO: u8 = 1;
const PORTABLE: u8 = 2;

/// [`AUTO`] or [`PORTABLE`]; [`UNREAD`] until `PSMC_RS_SIMD` is read.
static MODE: AtomicU8 = AtomicU8::new(UNREAD);

/// Whether the portable path is forced, from `PSMC_RS_SIMD` on first use.
fn portable_forced() -> bool {
    let mut mode = MODE.load(Ordering::Relaxed);
    if mode == UNREAD {
        let env = std::env::var_os("PSMC_RS_SIMD");
        let read = if env.is_some_and(|v| v == "portable") {
            PORTABLE
        } else {
            AUTO
        };
        // A concurrent set_force_portable wins over the environment.
        mode = match MODE.compare_exchange(UNREAD, read, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => read,
            Err(set) => set,
        };
    }
    mode == PORTABLE
}

/// Force the portable kernels (`true`) or pick them by CPU again (`false`),
/// overriding `PSMC_RS_SIMD`, for the passes started afterwards; for tests
/// and benchmarks. On x86-64 CPUs with AVX2 and FMA this changes the speed,
/// and the results in their last digits (the portable kernels do not fuse
/// multiply-adds); elsewhere it changes nothing. Returns whether the
/// portable kernels were forced before, by `PSMC_RS_SIMD` or by an earlier
/// call, so that a test can put back what it found.
pub fn set_force_portable(force: bool) -> bool {
    // Read PSMC_RS_SIMD first: it is the choice found until a call sets one.
    portable_forced();
    MODE.swap(if force { PORTABLE } else { AUTO }, Ordering::Relaxed) == PORTABLE
}

/// The kernels that HMM passes started now run.
pub fn simd_path() -> SimdPath {
    if cfg!(target_arch = "aarch64") {
        SimdPath::Neon
    } else if avx2_fma() {
        SimdPath::Avx2Fma
    } else {
        SimdPath::Portable
    }
}

/// Whether to run the AVX2 and FMA build of a pass: the CPU has both and
/// the portable path is not forced. Only then may the `avx2_fma` functions
/// of [`dispatch`] be called.
#[inline]
pub(crate) fn avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && !portable_forced()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Whether the passes compiled for the target as given fuse the
/// multiply-adds of the O(states) products: only on aarch64, whose NEON
/// kernels fuse anyway. Elsewhere `f64::mul_add` calls libm's `fma` unless
/// the build enables FMA, and an x86-64 build that does runs the AVX2 and
/// FMA path anyway.
pub(crate) const DEFAULT_FUSED: bool = cfg!(target_arch = "aarch64");
#[cfg(target_arch = "x86_64")]
const _: () = assert!(!DEFAULT_FUSED);

/// `dispatch! { vis fn name<'a>(arg: Type, ...) -> Ret = imp; }` defines
/// `name`, which calls `imp::<FUSED>` (an `#[inline(always)]` function of
/// the same signature and a `const FUSED: bool` parameter) compiled with
/// AVX2 and FMA and fused when [`avx2_fma`] says so, and as compiled for the
/// target, with [`DEFAULT_FUSED`], otherwise. Everything `imp` runs per site
/// must be `#[inline(always)]` too: a function that is not inlined is
/// compiled once, without the features.
///
/// So is a closure, which does not get the features of the pass it is
/// written in: work that a pass hands to other threads calls a function
/// defined by `dispatch! { vis fn name<const FUSED: bool>(arg: Type, ...) =
/// imp; }`, whose `name::<FUSED>` calls `imp::<FUSED>` with the caller's
/// `FUSED`, compiled with AVX2 and FMA when it is fused and the CPU has both.
macro_rules! dispatch {
    (
        $(#[$doc:meta])*
        $vis:vis fn $name:ident<const FUSED: bool>($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?
            = $imp:ident;
    ) => {
        $(#[$doc])*
        $vis fn $name<const FUSED: bool>($($arg: $ty),*) $(-> $ret)? {
            #[cfg(target_arch = "x86_64")]
            if FUSED && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn avx2_fma($($arg: $ty),*) $(-> $ret)? {
                    $imp::<true>($($arg),*)
                }
                // SAFETY: the CPU has AVX2 and FMA.
                return unsafe { avx2_fma($($arg),*) };
            }
            $imp::<FUSED>($($arg),*)
        }
    };
    (
        $(#[$doc:meta])*
        $vis:vis fn $name:ident $(<$($lt:lifetime),*>)? ($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty
            = $imp:ident;
    ) => {
        $(#[$doc])*
        $vis fn $name $(<$($lt),*>)? ($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            if $crate::hmm::simd::avx2_fma() {
                #[target_feature(enable = "avx2,fma")]
                unsafe fn avx2_fma $(<$($lt),*>)? ($($arg: $ty),*) -> $ret {
                    $imp::<true>($($arg),*)
                }
                // SAFETY: simd::avx2_fma() found AVX2 and FMA on this CPU.
                return unsafe { avx2_fma($($arg),*) };
            }
            $imp::<{ $crate::hmm::simd::DEFAULT_FUSED }>($($arg),*)
        }
    };
}
pub(crate) use dispatch;
