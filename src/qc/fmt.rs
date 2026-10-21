//! Number formatting shared by the QC messages, the reports and the
//! program's messages.

/// `1234567` -> `"1,234,567"`.
pub fn count(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// `v` with `digits` significant digits, in fixed notation from 1e-4 up to
/// 1e6 and in scientific notation outside.
pub(crate) fn sig(v: f64, digits: usize) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let exp = v.abs().log10().floor() as i32;
    if (-4..6).contains(&exp) {
        let decimals = (digits as i32 - 1 - exp).max(0) as usize;
        format!("{v:.decimals$}")
    } else {
        format!("{v:.prec$e}", prec = digits.saturating_sub(1))
    }
}

/// [`sig`] of a value that may be missing (`n/a`).
pub(crate) fn sig_or_na(v: Option<f64>, digits: usize) -> String {
    v.map_or("n/a".to_string(), |v| sig(v, digits))
}

/// A fraction as a percentage: one decimal, or two significant digits below
/// 0.1% so that small non-zero shares do not print as `0.0%`.
pub fn pct(fraction: f64) -> String {
    let p = 100.0 * fraction;
    if p > 0.0 && p < 0.1 {
        format!("{}%", sig(p, 2))
    } else {
        format!("{p:.1}%")
    }
}

/// `one` for a count of 1, else `many`.
pub fn plural<'a>(n: usize, one: &'a str, many: &'a str) -> &'a str {
    if n == 1 { one } else { many }
}

/// `n` and `one` or `many` after it: `1 sequence`, `1,234 sequences`.
pub fn counted(n: usize, one: &str, many: &str) -> String {
    format!("{} {}", count(n as u64), plural(n, one, many))
}

/// A length in bases with a unit and three significant digits: `850 bp`,
/// `12.3 kb`, `45.6 Mb`, `2.94 Gb`. Digits are cut, not rounded, so that a
/// length just below a limit never prints as the limit (99,999,900 bp is
/// `99.9 Mb`).
pub(crate) fn bases(n: u64) -> String {
    let (unit_size, unit) = match n {
        0..1_000 => return format!("{n} bp"),
        1_000..1_000_000 => (1_000u128, "kb"),
        1_000_000..1_000_000_000 => (1_000_000, "Mb"),
        _ => (1_000_000_000, "Gb"),
    };
    let whole = u128::from(n) / unit_size;
    let decimals: u32 = match whole {
        0..10 => 2,
        10..100 => 1,
        _ => 0,
    };
    let scaled = u128::from(n) * 10u128.pow(decimals) / unit_size;
    if decimals == 0 {
        format!("{scaled} {unit}")
    } else {
        let div = 10u128.pow(decimals);
        let frac = scaled % div;
        format!(
            "{}.{frac:0width$} {unit}",
            scaled / div,
            width = decimals as usize
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(count(0), "0");
        assert_eq!(count(999), "999");
        assert_eq!(count(1000), "1,000");
        assert_eq!(count(30_000_000), "30,000,000");
        assert_eq!(sig(0.058795, 3), "0.0588");
        assert_eq!(sig(2.5e-4, 2), "0.00025");
        assert_eq!(sig(2.5e-5, 2), "2.5e-5");
        assert_eq!(sig(123456.0, 3), "123456");
        assert_eq!(sig(2.0e7, 3), "2.00e7");
        assert_eq!(pct(0.62), "62.0%");
        assert_eq!(pct(0.0), "0.0%");
        assert_eq!(pct(0.00042), "0.042%");
        assert_eq!(pct(1.0), "100.0%");
        assert_eq!(counted(1, "record", "records"), "1 record");
        assert_eq!(counted(10_000, "record", "records"), "10,000 records");
        assert_eq!(bases(850), "850 bp");
        assert_eq!(bases(12_345), "12.3 kb");
        assert_eq!(bases(1_000_000), "1.00 Mb");
        assert_eq!(bases(456_000_000), "456 Mb");
        assert_eq!(bases(2_940_000_000), "2.94 Gb");
        assert_eq!(bases(99_999_900), "99.9 Mb");
        assert_eq!(bases(1_999_999), "1.99 Mb");
        assert_eq!(bases(1_050), "1.05 kb");
        assert_eq!(bases(u64::MAX), "18446744073 Gb");
    }
}
