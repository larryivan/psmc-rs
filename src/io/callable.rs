//! The callable bases of each bin, for the callable-length emission model
//! of `docs/design/bin-callable-model.md`.
//!
//! A bin of `s` bases is `N` when more than 90% of them are missing
//! (`fq2psmcfa`'s rule), and C psmc treats every other bin as `s` callable
//! bases. Input binned from per-site data can also record each bin's count
//! `c`: `s` minus the positions of the bin classified missing. The last bin
//! of a sequence may be shorter than `s`; as `fq2psmcfa` judges it against
//! the full `s`, its positions past the end of the sequence are not missing
//! and count as callable. So the count and the symbol always agree: a bin
//! is `N` exactly when `10 * c < s`, and a `T` or `K` bin has
//! `0 < c <= s`. Complete data give `c = s` for every bin.

use anyhow::{Result, bail};

use crate::io::{Observations, Sequences};

/// The largest bin size for input that psmc-rs bins itself (VCF and
/// multihetsep): a bin's callable bases are kept in 32 bits
/// ([`CallableBases`]). PSMC bins are normally 100 bases; much larger bins
/// are useful only to bin long test inputs quickly.
pub const MAX_BIN_SIZE: usize = u32::MAX as usize;

/// The callable bases of each bin of a list of rows or sequences of bins:
/// `counts()[i][j]` belongs to bin `j` of entry `i`. See the module docs for
/// what a count is; [`Observations::validate`] and [`Sequences::validate`]
/// check the counts against their bins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallableBases {
    bin_size: u32,
    counts: Vec<Vec<u32>>,
}

impl CallableBases {
    /// Counts for bins of `bin_size` bases, which must be between 1 and
    /// [`MAX_BIN_SIZE`]. The counts themselves are checked with their bins
    /// ([`Observations::validate`]).
    pub fn new(bin_size: usize, counts: Vec<Vec<u32>>) -> Result<Self> {
        let Ok(bin_size @ 1..) = u32::try_from(bin_size) else {
            bail!(
                "the bin size of callable bases must be between 1 and {MAX_BIN_SIZE}, got \
                 {bin_size}"
            );
        };
        Ok(Self { bin_size, counts })
    }

    /// Bases per bin, `s`.
    pub fn bin_size(&self) -> usize {
        // Lossless: MAX_BIN_SIZE is a usize.
        self.bin_size as usize
    }

    /// The counts, one list per row or sequence.
    pub fn counts(&self) -> &[Vec<u32>] {
        &self.counts
    }

    /// The counts, one list per row or sequence.
    pub fn into_counts(self) -> Vec<Vec<u32>> {
        self.counts
    }

    /// Other counts for bins of the same size, such as the rows or trunks
    /// these counts are cut into.
    pub fn with_counts(&self, counts: Vec<Vec<u32>>) -> Self {
        Self {
            bin_size: self.bin_size,
            counts,
        }
    }
}

/// The checks of [`Observations::validate`] and [`Sequences::validate`]:
/// `bins` are rows chained by `row_starts`, or sequences of their own when it
/// is `None`.
pub(crate) fn validate(
    bins: &[Vec<u8>],
    row_starts: Option<&[bool]>,
    callable: Option<&CallableBases>,
) -> Result<()> {
    let entry = if row_starts.is_some() {
        "row"
    } else {
        "sequence"
    };
    if let Some(starts) = row_starts {
        if starts.len() != bins.len() {
            bail!(
                "{} row starts for {} rows of bins",
                starts.len(),
                bins.len()
            );
        }
        if starts.first() == Some(&false) {
            bail!("the first row must start a sequence");
        }
    }
    let Some(callable) = callable else {
        return Ok(());
    };
    let counts = callable.counts();
    if counts.len() != bins.len() {
        bail!(
            "callable bases for {} {entry}s, but there are {} {entry}s of bins",
            counts.len(),
            bins.len()
        );
    }
    // In 64 bits, where 10 times any count fits.
    let s = u64::from(callable.bin_size);
    for (i, (row, row_counts)) in bins.iter().zip(counts).enumerate() {
        if row_counts.len() != row.len() {
            bail!(
                "{entry} {i} has {} bins but callable bases for {}",
                row.len(),
                row_counts.len()
            );
        }
        for (j, (&symbol, &c)) in row.iter().zip(row_counts).enumerate() {
            let c = u64::from(c);
            let at = || format!("{entry} {i}, bin {j}");
            if c > s {
                bail!("{}: {c} callable bases in a bin of {s}", at());
            }
            // fq2psmcfa's rule: N when more than 90% of the bases are missing.
            let missing = 10 * c < s;
            match symbol {
                0 | 1 if missing => bail!(
                    "{}: a called bin ({}) with {c} callable bases of {s}, fewer than 10%",
                    at(),
                    if symbol == 0 { "T" } else { "K" }
                ),
                2 if !missing => bail!(
                    "{}: a missing bin (N) with {c} callable bases of {s}, at least 10%",
                    at()
                ),
                0..=2 => {}
                _ => bail!("{}: invalid bin symbol {symbol} (expected 0, 1 or 2)", at()),
            }
        }
    }
    Ok(())
}

impl Observations {
    /// Check that the rows, their starts and their callable bases fit
    /// together: one start per row, the first row starting a sequence; and,
    /// with callable bases, one count per bin, none above the bin size, and
    /// every symbol T (0), K (1) or N (2) as its count says (N exactly when
    /// fewer than 10% of its bases are callable).
    pub fn validate(&self) -> Result<()> {
        validate(&self.rows, Some(&self.row_starts), self.callable.as_ref())
    }
}

impl Sequences {
    /// [`Observations::validate`] for whole sequences: with callable bases,
    /// one count per bin, none above the bin size, and every symbol as its
    /// count says.
    pub fn validate(&self) -> Result<()> {
        validate(&self.bins, None, self.callable.as_ref())
    }
}
