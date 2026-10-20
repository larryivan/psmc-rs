//! What the per-bin passes read at each bin: an observation index into a
//! table of emission rows.
//!
//! Without callable bases the indices are the symbol classes themselves (0
//! `T`, 1 `K`, 2 `N`) and the rows are the model's three emission rows, as
//! in C psmc. With callable bases ([`CallableBases`]) a bin of `c` callable
//! bases out of `s` emits by the callable-length model of
//! `docs/design/bin-callable-model.md`: mutations along its callable part
//! form a Poisson process, so the probability of seeing no heterozygous
//! site is the full bin's to the power `c / s`,
//!
//! ```text
//! e_T(c, k) = e_T(k)^(c / s) = exp(ln e_T(k) * c / s)
//! e_K(c, k) = 1 - e_T(c, k)
//! e_N(c, k) = e_N(k)
//! ```
//!
//! which assumes `e_K = 1 - e_T`, as PSMC's emissions are. Each `(symbol,
//! count)` pair that occurs in the input gets its own index and row, built
//! once per pass, so that the inner loops stay a lookup: a `T` or `K` bin
//! holds `ceil(s / 10)` to `s` callable bases and every `N` bin shares one
//! index, so there are at most `2 (s - ceil(s / 10) + 1) + 1` of them, 183
//! for bins of 100 bases. The rows of full bins (`c = s`) are the model's
//! own rows, so that input whose bins are all full gives exactly the
//! results of input without counts.

use anyhow::{Result, bail};
use ndarray::Array2;
use rayon::prelude::*;

use super::CallableStats;
use crate::io::{BinCounts, CallableBases};

/// Observation class of a raw bin symbol: 0 (`T`), 1 (`K`), 2 (`N` or other).
#[inline]
pub(crate) fn obs_class(v: u8) -> u8 {
    if v <= 1 { v } else { 2 }
}

/// The observation indices of an input and their emission rows.
#[derive(Debug, Clone)]
pub(crate) struct ObsTable {
    /// The symbol class (0 `T`, 1 `K`, 2 `N`) of each index.
    class: Vec<u8>,
    /// The callable bases of each `T` and `K` index (`s` without counts,
    /// unused for `N`).
    count: Vec<u16>,
    /// `s`, the bases of a full bin; 0 without counts.
    bin_size: u16,
    /// With counts, the index of `(class, c)` at `class * (s + 1) + c`
    /// (every count of class `N` gives the one `N` index); `u16::MAX` for
    /// pairs that do not occur.
    lookup: Vec<u16>,
}

/// Which `(class, count)` pairs occur among the bins of some rows: `seen[class
/// * (s + 1) + c]`.
type Seen = Vec<bool>;

/// The pairs that occur in row `i`, `symbols` with the callable bases
/// `counts` of bins of `s` bases; errors unless they fit the row.
fn seen_in_row(i: usize, symbols: &[u8], counts: &BinCounts, s: usize) -> Result<Seen> {
    if counts.bin_size() != s {
        bail!(
            "row {i} has callable bases of bins of {} bases, in counts of bins of {s}",
            counts.bin_size()
        );
    }
    if counts.len() != symbols.len() {
        bail!(
            "row {i} has {} bins but callable bases for {}",
            symbols.len(),
            counts.len()
        );
    }
    let mut seen = vec![false; 3 * (s + 1)];
    for (j, (&symbol, count)) in symbols.iter().zip(counts).enumerate() {
        let count = usize::from(count);
        // fq2psmcfa's rule: N when more than 90% of the bases are missing.
        let missing = 10 * count < s;
        let class = match symbol {
            0 | 1 if count <= s && !missing => symbol,
            2 if missing => 2,
            _ => bail!(
                "row {i}, bin {j}: symbol {symbol} with {count} callable bases of {s} \
                 (see Observations::validate)"
            ),
        };
        seen[usize::from(class) * (s + 1) + count] = true;
    }
    Ok(seen)
}

impl ObsTable {
    /// The table of input without callable bases: the three symbol classes.
    pub(crate) fn plain() -> Self {
        Self {
            class: vec![0, 1, 2],
            count: vec![0; 3],
            bin_size: 0,
            lookup: Vec::new(),
        }
    }

    /// The table of `rows` with callable bases `callable`, one index per
    /// `(symbol, count)` pair that occurs: `T` bins by ascending count, then
    /// `K` bins by ascending count, then `N` bins (whatever their count).
    ///
    /// Errors unless the counts fit the rows: one [`BinCounts`] per row, of
    /// the row's length and the bin size of `callable`, every count at most
    /// the bin size and every symbol `T`, `K` or `N` as its count says
    /// ([`crate::io::Observations::validate`]).
    pub(crate) fn counted(rows: &[Vec<u8>], callable: &CallableBases) -> Result<Self> {
        let s = callable.bin_size();
        let counts = callable.rows();
        if counts.len() != rows.len() {
            bail!(
                "callable bases for {} rows, but there are {} rows of bins",
                counts.len(),
                rows.len()
            );
        }
        let width = s + 1;
        let per_row: Vec<Result<Seen>> = rows
            .par_iter()
            .zip(counts)
            .enumerate()
            .map(|(i, (row, c))| seen_in_row(i, row, c, s))
            .collect();
        let mut seen = vec![false; 3 * width];
        for row in per_row {
            for (a, b) in seen.iter_mut().zip(row?) {
                *a |= b;
            }
        }

        let bin_size = u16::try_from(s).expect("callable bases are counted for bins of u16");
        let mut table = Self {
            class: Vec::new(),
            count: Vec::new(),
            bin_size,
            lookup: vec![u16::MAX; 3 * width],
        };
        let add = |table: &mut Self, class: u8, count: usize| -> Result<u16> {
            let Ok(index) = u16::try_from(table.class.len()) else {
                bail!("more than {} distinct observations", u16::MAX);
            };
            table.class.push(class);
            table.count.push(count as u16);
            Ok(index)
        };
        for class in [0u8, 1] {
            for count in 0..width {
                if seen[usize::from(class) * width + count] {
                    let index = add(&mut table, class, count)?;
                    table.lookup[usize::from(class) * width + count] = index;
                }
            }
        }
        if seen[2 * width..].iter().any(|&b| b) {
            let index = add(&mut table, 2, 0)?;
            table.lookup[2 * width..].fill(index);
        }
        Ok(table)
    }

    /// Number of observation indices.
    pub(crate) fn len(&self) -> usize {
        self.class.len()
    }

    /// The symbol class (0 `T`, 1 `K`, 2 `N`) of every index.
    pub(crate) fn classes(&self) -> &[u8] {
        &self.class
    }

    /// Whether the table has callable bases.
    pub(crate) fn is_counted(&self) -> bool {
        self.bin_size > 0
    }

    /// The index of a bin of symbol `v` with `count` callable bases (in a
    /// counted table).
    #[inline]
    pub(crate) fn index(&self, v: u8, count: u16) -> u16 {
        let width = usize::from(self.bin_size) + 1;
        self.lookup[usize::from(obs_class(v)) * width + usize::from(count)]
    }

    /// The share of a full bin's bases that index `i` has callable, when it
    /// is a `T` or `K` index that is not full.
    fn thinning(&self, i: usize) -> Option<f64> {
        let full = self.bin_size;
        (self.class[i] < 2 && full > 0 && self.count[i] != full)
            .then(|| f64::from(self.count[i]) / f64::from(full))
    }

    /// The emission row of every index, padded to `np` states with zeros,
    /// from the model's emission rows `em` (`T`, `K`, `N`) of `n` states:
    /// row `i` is `rows[i * np..(i + 1) * np]`. See the module documentation
    /// for the rows of partly callable bins.
    pub(crate) fn emissions(&self, em: &Array2<f64>, n: usize, np: usize) -> Vec<f64> {
        let mut rows = vec![0.0; self.len() * np];
        for (i, row) in rows.chunks_exact_mut(np).enumerate() {
            let class = usize::from(self.class[i]);
            match self.thinning(i) {
                None => {
                    for (k, v) in row[..n].iter_mut().enumerate() {
                        *v = em[(class, k)];
                    }
                }
                Some(f) => {
                    for (k, v) in row[..n].iter_mut().enumerate() {
                        let t = (em[(0, k)].ln() * f).exp();
                        *v = if class == 0 { t } else { 1.0 - t };
                    }
                }
            }
        }
        rows
    }

    /// The expected counts of an E-step from its sums per index, `g[i * np +
    /// k]` (the posterior of state `k` summed over the bins of index `i`):
    /// the sums per symbol class (C psmc's expected emission counts), and
    /// with callable bases the sums the M-step of the callable-length model
    /// takes ([`CallableStats`]). Sums are taken in index order, from 0;
    /// with one index per class (no counts, or every bin full) they are the
    /// sums of those indices to the bit.
    pub(crate) fn reduce(
        &self,
        g: &[f64],
        n: usize,
        np: usize,
    ) -> ([Vec<f64>; 3], Option<CallableStats>) {
        let mut gobs = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (i, row) in g.chunks_exact(np).enumerate() {
            let dst = &mut gobs[usize::from(self.class[i])];
            for (d, &v) in dst.iter_mut().zip(&row[..n]) {
                *d += v;
            }
        }
        if !self.is_counted() {
            return (gobs, None);
        }
        let s = f64::from(self.bin_size);
        let mut t_weighted = vec![0.0; n];
        let mut k_by_count = Vec::new();
        for (i, row) in g.chunks_exact(np).enumerate() {
            match self.class[i] {
                0 => {
                    let f = f64::from(self.count[i]) / s;
                    for (d, &v) in t_weighted.iter_mut().zip(&row[..n]) {
                        *d += f * v;
                    }
                }
                1 => k_by_count.push((self.count[i], row[..n].to_vec())),
                _ => {}
            }
        }
        let stats = CallableStats {
            bin_size: usize::from(self.bin_size),
            t_weighted,
            k_by_count,
        };
        (gobs, Some(stats))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PsmcModel;

    /// The rows of a counted table are the callable-length model's
    /// emissions, `exp(-θ (c / s) avg_t[k])` for `T` and one minus that for
    /// `K`, as the model's own parameters give them: to 1e-15 relative for
    /// `T`, and for `K` to the rounding of the full bin's `T` emission
    /// itself (which C psmc's `1 - exp(-θ avg_t)` shares), 1e-15 absolute.
    #[test]
    fn rows_are_the_callable_length_emissions() {
        let model = PsmcModel::new(15.0, 22, 0.02, 0.005, 2.5e-8, Some("4+5*3+4".into())).unwrap();
        let (n, np) = (model.n_steps + 1, 24);
        let s = 100usize;
        let rows = vec![vec![0u8, 1, 2, 0, 1, 0, 1]];
        let counts = vec![vec![100u16, 100, 3, 10, 10, 57, 99]];
        let callable = CallableBases::from_counts(s, counts).unwrap();
        let table = ObsTable::counted(&rows, &callable).unwrap();
        // T 10, 57, 100; K 10, 99, 100; N.
        assert_eq!(table.classes(), [0, 0, 0, 1, 1, 1, 2]);
        assert_eq!(table.count[..6], [10, 57, 100, 10, 99, 100]);
        assert_eq!(table.index(2, 7), 6);
        assert_eq!(table.index(0, 57), 1);
        assert_eq!(table.index(7, 0), 6, "symbols above 2 read as N");
        let em = table.emissions(model.emission_matrix(), n, np);
        let avg = model.avg_t();
        for i in 0..table.len() {
            let row = &em[i * np..(i + 1) * np];
            assert!(row[n..].iter().all(|&v| v == 0.0), "padding");
            let c = f64::from(table.count[i]) / s as f64;
            for k in 0..n {
                let t = (-model.theta * c * avg[k]).exp();
                match table.class[i] {
                    0 => assert!((row[k] - t).abs() <= 1e-15 * t, "T {i} {k}"),
                    1 => assert!((row[k] - (1.0 - t)).abs() <= 1e-15, "K {i} {k}"),
                    _ => assert_eq!(row[k], 1.0),
                }
                if table.count[i] == 100 && table.class[i] < 2 {
                    let full = model.emission_matrix()[(usize::from(table.class[i]), k)];
                    assert_eq!(row[k].to_bits(), full.to_bits(), "full bins use C's rows");
                }
            }
        }
        // The sums of an E-step: per class, G_T weighted by c / s, G_K by count.
        let g: Vec<f64> = (0..table.len() * np)
            .map(|i| (i % np) as f64 + 1.0)
            .collect();
        let (gobs, stats) = table.reduce(&g, n, np);
        let stats = stats.unwrap();
        assert_eq!(gobs[0][3], 3.0 * 4.0);
        assert_eq!(gobs[2][3], 4.0);
        assert!((stats.t_weighted[3] - (0.1 + 0.57 + 1.0) * 4.0).abs() < 1e-14);
        let k_counts: Vec<u16> = stats.k_by_count.iter().map(|(c, _)| *c).collect();
        assert_eq!(k_counts, [10, 99, 100]);
    }

    /// Every valid pair of symbol and count gives at most `2 (s - ceil(s /
    /// 10) + 1) + 1` indices (183 for `s = 100`), the bound the module
    /// documentation and the M-step's cost state.
    #[test]
    fn indices_are_bounded_by_the_valid_counts() {
        for s in [1usize, 9, 10, 37, 100, 255, 1000] {
            let called = s.div_ceil(10)..=s;
            let mut row = Vec::new();
            let mut counts = Vec::new();
            for c in called.clone() {
                row.extend([0u8, 1]);
                counts.extend([c as u16, c as u16]);
            }
            for c in 0..s.div_ceil(10) {
                row.push(2);
                counts.push(c as u16);
            }
            let callable = CallableBases::from_counts(s, vec![counts]).unwrap();
            let table = ObsTable::counted(&[row], &callable).unwrap();
            let k_counts = s - s.div_ceil(10) + 1;
            assert_eq!(table.len(), 2 * k_counts + 1, "s = {s}");
            if s == 100 {
                assert_eq!((table.len(), k_counts), (183, 91));
            }
        }
    }

    #[test]
    fn the_plain_table_is_the_three_symbols() {
        let table = ObsTable::plain();
        assert_eq!((table.len(), table.is_counted()), (3, false));
        let em = Array2::from_shape_fn((3, 2), |(r, k)| (r * 2 + k) as f64);
        let rows = table.emissions(&em, 2, 8);
        assert_eq!(rows[..2], [0.0, 1.0]);
        assert_eq!(rows[16..18], [4.0, 5.0]);
        let (_, stats) = table.reduce(&[0.0; 24], 2, 8);
        assert!(stats.is_none());
    }
}
