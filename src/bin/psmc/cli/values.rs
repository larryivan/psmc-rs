//! The values of the model, bootstrap and output options: their defaults,
//! their parsers, and the model layout of `--pattern` and `--n-steps`.

use anyhow::Result;
use clap::ValueEnum;
use std::str::FromStr;

use psmc_rs::model::{MAX_STATES, n_steps_for_pattern};
use psmc_rs::opt::MStepConfig;

use super::usage_bail;

/// `--bin-callable`.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BinCallableArg {
    /// Weigh each called bin by its callable bases when they are counted exactly (VCF)
    #[default]
    Auto,
    /// Treat every called bin as BIN_SIZE bases, as fq2psmcfa and C psmc do
    Ignore,
    /// As auto, and also weigh multihetsep's bins by their approximate counts
    Approximate,
}

impl BinCallableArg {
    /// The name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            BinCallableArg::Auto => "auto",
            BinCallableArg::Ignore => "ignore",
            BinCallableArg::Approximate => "approximate",
        }
    }

    /// Whether a fit in this mode weighs each called bin by counts of
    /// callable bases obtained as `source` says (`"exact"` from a VCF,
    /// `"approximate"` from multihetsep): `auto` takes exact counts only,
    /// `approximate` both, `ignore` none.
    pub fn uses_counts(self, source: &str) -> bool {
        match self {
            BinCallableArg::Auto => source == "exact",
            BinCallableArg::Approximate => true,
            BinCallableArg::Ignore => false,
        }
    }
}

/// C psmc's default pattern (`-p`).
const DEFAULT_PATTERN: &str = "4+5*3+4";

/// C psmc's default initial t_max (`-t`).
pub const DEFAULT_T_MAX: f64 = 15.0;

/// C psmc's default initial θ0/ρ0 (`-r`).
pub const DEFAULT_TR_RATIO: f64 = 4.0;

/// The default mutation rate per base pair per generation (`--mu`).
pub const DEFAULT_MU: f64 = 2.5e-8;

// The defaults of the bootstrap and output options.
pub(super) const DEFAULT_BOOTSTRAP_SEED: u64 = 42;
pub(super) const DEFAULT_GEN_YEARS: f64 = 25.0;

/// A positive integer.
pub fn positive_usize(s: &str) -> Result<usize, String> {
    match whole_number(s)? {
        0 => Err("must be at least 1".to_string()),
        n => Ok(n),
    }
}

/// A whole number of at least 0 (clap's own error for one would be Rust's
/// "invalid digit found in string").
pub fn whole_number<T: FromStr>(s: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("'{s}' is not a whole number"))
}

/// A positive, finite number.
pub fn positive_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v > 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be a positive number, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// A finite number of at least 0.
pub fn non_negative_f64(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v >= 0.0 && v.is_finite() => Ok(v),
        Ok(v) => Err(format!("must be 0 or more, got {v}")),
        Err(_) => Err(format!("'{s}' is not a number")),
    }
}

/// `--t-max`: inside the range the M-step searches. A start outside it
/// would be moved to its edge by the first M-step, and one near `T_INF`,
/// where the time grid ends, breaks the first E-step.
pub(super) fn parse_t_max(s: &str) -> Result<f64, String> {
    let v = positive_f64(s)?;
    let MStepConfig {
        tmax_lo, tmax_hi, ..
    } = MStepConfig::default();
    if !(tmax_lo..=tmax_hi).contains(&v) {
        return Err(format!(
            "must be between {tmax_lo} and {tmax_hi}, the range the M-step searches, got {v}"
        ));
    }
    Ok(v)
}

/// `--n-steps`: a model of 2 to `MAX_STATES` intervals.
pub(super) fn parse_n_steps(s: &str) -> Result<usize, String> {
    let n = whole_number(s)?;
    if !(1..MAX_STATES).contains(&n) {
        return Err(format!(
            "must be between 1 and {} (2 to {MAX_STATES} intervals), got {n}",
            MAX_STATES - 1
        ));
    }
    Ok(n)
}

/// The model layout `(n_steps, pattern)` that `--pattern` and `--n-steps`
/// give: C psmc's default pattern without either, `n_steps` from the
/// pattern without `--n-steps`, and one free lambda per interval with
/// `--n-steps` alone. Given both, `--n-steps` must match the pattern;
/// its number of intervals (`n_steps + 1`, the form psmc-rs 0.1 used) is
/// accepted too. An invalid pattern or a mismatch is a [`UsageError`].
pub fn model_layout(
    pattern: Option<&str>,
    n_steps: Option<usize>,
) -> Result<(usize, Option<String>)> {
    let Some(pattern) = pattern else {
        return match n_steps {
            Some(n) => Ok((n, None)),
            None => Ok((
                n_steps_for_pattern(DEFAULT_PATTERN)?,
                Some(DEFAULT_PATTERN.to_string()),
            )),
        };
    };
    let derived = match n_steps_for_pattern(pattern) {
        Ok(n) => n,
        // A term that is not a number gives the parser's bare "invalid
        // digit found in string".
        Err(e) if e.is::<std::num::ParseIntError>() => usage_bail!(
            "invalid --pattern '{pattern}': its terms, joined by '+', are each L or R*L in whole \
             numbers (R free lambdas of L intervals each), as in 4+5*3+4"
        ),
        Err(e) => usage_bail!("invalid --pattern '{pattern}': {e:#}"),
    };
    match n_steps {
        None => Ok((derived, Some(pattern.to_string()))),
        Some(n) if n == derived || n == derived + 1 => Ok((derived, Some(pattern.to_string()))),
        Some(n) => usage_bail!(
            "--n-steps {n} does not match --pattern '{pattern}', which has {} intervals \
             (n_steps {derived}); leave out --n-steps to take it from the pattern",
            derived + 1
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::UsageError;
    use crate::cli::tests::{parse, parse_err};

    #[test]
    fn the_layout_follows_c_psmc() {
        assert_eq!(
            model_layout(None, None).unwrap(),
            (22, Some("4+5*3+4".to_string()))
        );
        assert_eq!(model_layout(None, Some(63)).unwrap(), (63, None));
        let human = Some("4+25*2+4+6".to_string());
        assert_eq!(
            model_layout(human.as_deref(), None).unwrap(),
            (63, human.clone())
        );
        assert_eq!(
            model_layout(human.as_deref(), Some(63)).unwrap(),
            (63, human.clone())
        );
        // psmc-rs 0.1 took the number of intervals.
        assert_eq!(
            model_layout(human.as_deref(), Some(64)).unwrap(),
            (63, human.clone())
        );
        let err = model_layout(human.as_deref(), Some(30)).unwrap_err();
        assert!(UsageError::is_in(&err));
        let err = err.to_string();
        assert!(err.contains("has 64 intervals (n_steps 63)"), "{err}");
        let err = model_layout(Some("4+x"), None).unwrap_err();
        assert!(UsageError::is_in(&err));
        let err = format!("{err:#}");
        assert!(err.contains("invalid --pattern '4+x'"), "{err}");
    }

    #[test]
    fn the_defaults_are_c_psmcs() {
        let args = parse("").unwrap();
        assert_eq!(
            (args.t_max, args.tr_ratio, args.mu),
            (DEFAULT_T_MAX, DEFAULT_TR_RATIO, DEFAULT_MU)
        );
        // QC's messages about a default start name the same ratio.
        assert_eq!(DEFAULT_TR_RATIO, psmc_rs::qc::DEFAULT_TR_RATIO);
        // The callable bases of each bin are fitted when the input has them.
        assert_eq!(args.bin_callable, BinCallableArg::Auto);
        let mode = |name: &str| parse(&format!("--bin-callable {name}")).map(|a| a.bin_callable);
        assert_eq!(mode("ignore").unwrap(), BinCallableArg::Ignore);
        assert_eq!(mode("approximate").unwrap(), BinCallableArg::Approximate);
        assert!(mode("x").is_err());
        // auto fits exact counts only; approximate also multihetsep's.
        use BinCallableArg::*;
        let uses =
            |mode: BinCallableArg| [mode.uses_counts("exact"), mode.uses_counts("approximate")];
        assert_eq!(uses(Auto), [true, false]);
        assert_eq!(uses(Approximate), [true, true]);
        assert_eq!(uses(Ignore), [false, false]);
    }

    #[test]
    fn model_values_are_checked() {
        for t_max in ["1000", "900", "200.5", "0.5"] {
            let err = parse_err(&format!("--t-max {t_max}"));
            assert!(
                err.contains("must be between 1 and 200, the range the M-step searches"),
                "{t_max}: {err}"
            );
        }
        assert!(parse_err("--t-max 0").contains("positive"));
        for n in ["0", "10000", "100000"] {
            let err = parse_err(&format!("--n-steps {n}"));
            assert!(err.contains("must be between 1 and 9999"), "{n}: {err}");
        }
        assert!(parse_err("--n-steps x").contains("not a whole number"));
        // Every whole number, not Rust's "invalid digit found in string".
        for flag in [
            "--bootstrap",
            "--bootstrap-seed",
            "--bootstrap-iters",
            "--min-mapq",
        ] {
            let err = parse_err(&format!("{flag} 2.5"));
            assert!(err.contains("'2.5' is not a whole number"), "{flag}: {err}");
        }
        assert!(parse_err("--tr-ratio=-4").contains("positive"));
        assert!(parse_err("--smooth-lambda=-1").contains("0 or more"));
        assert!(parse_err("--mstep-iters 0").contains("at least 1"));
    }
}
