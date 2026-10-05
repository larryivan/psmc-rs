pub mod callable;
pub mod common;
pub mod consensus;
pub mod input;
pub mod mhs;
pub mod params;
pub mod psmcfa;
pub mod vcf;
pub mod write;

pub use callable::{CallableBases, MAX_BIN_SIZE};

/// Binned input as the HMM reads it: rows of bins, `0` (`T`), `1` (`K`) or
/// `2` (`N`), chained into sequences by `row_starts` (`true` for the first
/// row of each sequence).
#[derive(Debug, Clone)]
pub struct Observations {
    pub rows: Vec<Vec<u8>>,
    pub row_starts: Vec<bool>,
    /// The callable bases of each bin, row for row, when the input records
    /// them (VCF and multihetsep); `None` when every called bin is taken to
    /// hold the bin size in callable bases, as in C psmc (psmcfa input). See
    /// [`callable`] and [`Observations::validate`].
    pub callable: Option<CallableBases>,
}

/// Whole binned sequences, each one HMM chain (before they are cut into
/// rows, and the trunks of a bootstrap), with the callable bases of each
/// bin when the input records them, sequence for sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sequences {
    pub bins: Vec<Vec<u8>>,
    pub callable: Option<CallableBases>,
}

impl From<Vec<Vec<u8>>> for Sequences {
    /// Sequences without callable bases.
    fn from(bins: Vec<Vec<u8>>) -> Self {
        Self {
            bins,
            callable: None,
        }
    }
}
