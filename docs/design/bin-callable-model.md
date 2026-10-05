# Per-bin callable length

Design note for counting the callable bases of every bin and using them in
the PSMC emissions. The change comes in two stages:

1. **Input layer**: every reader that bins per-site data records how many
   bases of each bin are callable, and the counts travel with their bins
   through batching, the loaded dataset and the bootstrap. Nothing reads
   them yet, so no result changes.
2. **Model**, after the M-step rewrite in `src/opt.rs` lands: the E-step,
   M-step, starting values and posterior decoding use the counts, the
   command line gets `--bin-callable`, and the README, CHANGELOG and
   `docs/preprocessing.md` describe the result.

## The problem

PSMC sees a genome as bins of `s` bases (`fq2psmcfa -s`, 100 by default).
`fq2psmcfa` makes a bin `N` (missing) when more than 90% of its bases are
missing, `K` when one of its bases is heterozygous, and `T` otherwise. C
psmc then treats every `T` or `K` bin as `s` bases: the per-bin mutation
parameter θ = 4 N0 μ s, and N0, every time and every size are read off θ
with that `s`.

With missing data spread along the genome in short runs, almost every bin
is still called but holds fewer than `s` callable bases. A bin of 70
callable bases has about 70% of the chance of a full bin to show a
heterozygous site, so θ per bin, and with it N0, the times and Ne, come out
low by about the missing fraction. Experiment E4 (`experiment/pipeline/README.md`)
measured it: with 30% of the bases missing in runs of about 20 bp, the RMSE
of log10 Ne against the truth was 0.447, against 0.332 for the same genomes
without missing data; rescaling the fitted times and sizes by the callable
bases per called bin brought it back to 0.334. Missing data in long runs
(E4's 20 kb runs) mostly empty whole bins, which the `N` rule already
handles; only the bins at the edges of each run are partly called.

The rescaling in E4 corrects the average. The model below corrects each
bin, so that bins with many missing bases weigh less than full ones in the
fit, and the posterior over TMRCA at each bin knows how much of it was seen.

## The model

Bin `i` holds `c_i` callable bases, `0 <= c_i <= s`. Mutations along the
callable part of a bin whose two lineages coalesced at time `t` (in units of
2 N0 generations) form a Poisson process of rate θ/s per base, so for hidden
state `k`, with `avg_t[k]` its average coalescence time (C's `psmc_avg_t`):

```text
P(T | k, c) = exp(-θ (c / s) avg_t[k])
P(K | k, c) = 1 - exp(-θ (c / s) avg_t[k])
P(N | k, c) = 1
```

θ stays the mutation parameter of `s` bases of sequence, so N0 = θ / (4 μ s)
and the scaling of times and sizes keep their meaning; ρ, the
recombination parameter per bin, is unchanged too, since recombination
happens along the whole bin whether its bases were called or not.

With `c = s` these are exactly C psmc's emissions (`em[(0, k)] = exp(-θ
avg_t[k])` in `PsmcModel::param_recalculate`). Every bin has `c = s` when
the data are complete, and every bin of a psmcfa file is taken to have
`c = s`, since a psmcfa file does not record callable bases. In both cases
the model, the fit and every output are exactly C psmc's.

A bin with `c = 0` emits with probability 1 whatever its symbol could be, as
an `N` bin does: it carries no information. Under the counts rule below a
`T` or `K` bin always has `c >= s / 10`, so this happens only for `N` bins.

## The counts

The count of a bin is `s` minus its missing positions:

```text
c = s - (positions of the bin classified missing)
```

This is the statistic `fq2psmcfa` already computes for every bin (`nN`)
and thresholds: a bin is `N` exactly when more than 90% of its `s` bases
are missing, that is when `10 * (s - c) > 9 * s`, or `10 * c < s`. So the
symbols and the counts always agree:

- `N` bins have `0 <= c` and `10 * c < s` (their counts are kept, but
  ignored by the model);
- `T` and `K` bins have `10 * c >= s`, so `0 < c <= s`.

The last bin of a sequence whose length is not a multiple of `s` is
shorter than `s`. `fq2psmcfa` judges it against the full `s` (it counts its
missing bases only), and C psmc treats it as `s` bases when it is called.
Its count follows the same convention: the positions past the end of the
sequence are not missing, so they count as callable. This keeps the rule
above exact for every bin and makes complete data give `c = s` everywhere,
hence exactly C's results; the price is that the last bin of each sequence
is modelled as up to `s - 1` bases longer than it is, as in C, which for
chromosomes of millions of bins does not matter. (Counting only the
positions inside the sequence was considered and rejected: it would make a
complete VCF fit differ from C psmc on its psmcfa in one bin per sequence,
and leave `T` bins with 0 callable bases at sequence ends.)

## Data representation

`psmc_rs::io::Observations`, the HMM input, gains an optional structure
parallel to its rows:

```rust
pub struct Observations {
    pub rows: Vec<Vec<u8>>,           // 0 = T, 1 = K, 2 = N
    pub row_starts: Vec<bool>,
    pub callable: Option<CallableBases>,
}

pub struct CallableBases {            // in src/io/callable.rs
    bin_size: u32,                    // s
    counts: Vec<Vec<u32>>,            // counts[i][j]: c of bin j of row i
}
```

`callable: None` means that every called bin holds `s` callable bases: the
input does not know better (psmcfa), and the model is C's. `Some` holds one
count per bin, row for row. The counts live outside the rows so that the
E-step's hot loop over symbols is untouched when there are none, and so
that a run without counts costs no memory.

The counts are `u32`, so `s` is at most 4,294,967,295 (`io::MAX_BIN_SIZE`)
for the input psmc-rs bins itself (VCF and multihetsep); the readers refuse
a larger bin size. `u16` would have halved the memory of the counts but
capped `s` at 65,535, below bin sizes the readers accept and the tests use
(a bin of 2^30 bases bins a 2^36-base contig quickly). PSMC bins are
normally 100 bases, where `u32` costs 4 bytes per bin next to the 1 byte of
the symbol: about 120 MB for a 3 Gb genome, small next to the E-step's
buffers. psmcfa input has no counts, and its `--bin-size` (used only for
scaling) stays unbounded.

`Observations::validate` checks every invariant in one place:

- `row_starts` has one entry per row, and the first is `true`;
- `counts` has one row per row of bins, of the same length;
- `1 <= bin_size <= MAX_BIN_SIZE` (checked when the counts are made) and
  every `c <= bin_size`;
- every symbol is 0, 1 or 2, and agrees with its count: `N` exactly when
  `10 * c < s`.

Whole sequences before batching, and bootstrap trunks, are a
`psmc_rs::io::Sequences` (bins and optional counts, one entry per
sequence), checked the same way by `Sequences::validate`. The constructor
`common::chunk_sequences`, which cuts sequences into rows, checks its input
with it.

## Counts from each input

All readers bin through `io::common::Binner`, which counts each bin's
missing positions as it emits the bin; `Binner::finish_with_callable`
returns the counts with the symbols. The bins themselves, and so every
psmcfa and multihetsep file psmc-rs writes, are unchanged byte for byte.

| input | counts |
| --- | --- |
| VCF, mask mode | Exact: `s` minus the positions the per-site classification makes missing (outside the callable masks, filtered or missing genotypes, REF spans of heterozygous non-SNPs, gVCF blocks without a call, positions no gVCF record covers). |
| VCF, consensus mode | Exact: `s` minus the bases `fq2psmcfa` reads as missing (lowercase after vcf2fq's filters and `fq2psmcfa -q`, `N` and other non-IUPAC characters, positions without a record, INDEL windows, positions outside the optional masks), so `c` is the number of good bases of a full bin. |
| multihetsep | Approximate: multihetsep records only how many sites between two rows are callable (`nr_called`), not which. The reader places the uncalled ones at the start of each interval, right after the previous row, as it already does for the symbols, and the counts follow that placement. |
| psmcfa | None. |

The multihetsep approximation is milder than it looks. Within the
interval between two heterozygous rows every bin is `T` except the one that
holds the second row, and the likelihood of a run of `T` bins in one state
depends only on the sum of their counts, which the placement gets right.
What the placement can get wrong is (a) which bins are `N`, as it already
can for the symbols, (b) the count of the `K` bin at the end of the
interval, which the placement fills with called sites, and (c) the split
of the sum between bins on either side of a change of state. A VCF with a
callable mask gives exact counts.

## Interactions

**The `N` rule.** The symbols stay `fq2psmcfa`'s in both modes, so a run
with `--bin-callable ignore`, the psmcfa written by `convert` and QC see the
same bins as now. Only the emissions of `T` and `K` bins change. An `N` bin
holds at most 10% of a bin's bases, so lowering the threshold to use them
would gain little; it would also change the bins themselves, and is left
out.

**Batching.** `chunk_sequences` splits the counts with the same row
boundaries as the bins, and batching never cuts a bin, so the E-step sees
the same per-bin emissions with any `--batch-size`, as it sees the same
symbols now.

**Loaded data.** `io::input::load` fills `Dataset::obs.callable` for VCF
(both modes) and multihetsep input, and leaves it `None` for psmcfa. It is
`Some` also when every sequence was dropped (with no rows), so that it says
whether the input format reports counts. `Dataset::sequence_callable`
joins the counts of each kept sequence as `Dataset::sequences` joins its
bins. Per sequence, the counts sum to the called positions (`SeqInfo::called`)
plus the padding of the last bin (`s * n_bins - extent`).

**Bootstrap.** C psmc's bootstrap cuts the input into trunks
(`utils/splitfa`) and draws trunks with replacement (`psmc_resamp`).
`bootstrap::observations_to_sequences` joins the rows of each sequence
with their counts, `split_sequences_for_bootstrap` cuts the counts at the
same bin ranges as `split_for_bootstrap` cuts the bins, and
`resample_sequences` draws the same trunks from the same random stream as
`resample_chunks`, so a replicate's bins are the same with or without
counts. `run_bootstrap_sequences` runs a bootstrap on counted sequences
(the CLI uses it); the older `&[Vec<u8>]` entry points run it without
counts.

**Starting values.** C psmc starts from θ0 = -ln(1 - K/L), with K the
heterozygous and L the called bins. With counts, stage 2 should start from
θ0 = -ln(1 - K/L) * s / c̄, where c̄ is the mean count of the called bins:
the same moment estimate per callable base, scaled to `s` bases. It equals
C's value on complete data. The bootstrap's fresh starts
(`replicate_initial_model`), `invalid_replicate_starts` (which must find
exactly the replicates the bootstrap skips) and QC's `theta0_per_bin` must
use the same rule.

**E-step (stage 2).** The forward and backward passes multiply by
`e(o_i, c_i, k)` instead of `e(o_i, k)`. For a `T` bin this is
`exp(-θ avg_t[k])^(c / s)`, so a table of `T` and `K` emissions for each
count value present in the input (at most `s + 1` of them, few for `s =
100`) keeps the inner loop a lookup. Rows with no counts, and bins with
`c = s`, must use the current emissions, so that results without counts or
with complete data stay bit-identical.

**M-step (stage 2).** The emission part of the expected complete-data
log-likelihood becomes

```text
sum_k [ -θ avg_t[k] G_T(k) + sum_c G_K(c, k) ln(1 - exp(-θ (c / s) avg_t[k])) ]
G_T(k)    = sum over T bins i of γ_i(k) c_i / s
G_K(c, k) = sum over K bins i with c_i = c of γ_i(k)
```

so the E-step must hand over, instead of the expected symbol counts
`gobs[0]` and `gobs[1]`, the count-weighted `T` sum and the per-count `K`
sums (a sparse histogram over the counts that occur; `K` bins are a few
percent of the bins). With every `c = s` these are `gobs[0]` and `gobs[1]`
and the objective is C's. The transition part is unchanged.

**Posterior decoding.** The TMRCA output must decode with the same per-bin
emissions as the fit; its columns stay as they are (`obs` is the symbol).

**`convert` and the writers.** psmcfa has no place for counts: a psmcfa
written from counted input holds exactly the bins `fq2psmcfa` would write,
and a fit of it is the `ignore` fit. Stage 2 should say so when `convert`
writes psmcfa from input whose counts differ from `s`. The multihetsep
writer keeps `nr_called`, from which the reader rebuilds approximate
counts.

**QC.** Unchanged in stage 1. In stage 2, QC can report the mean count of
the called bins (the effective bin length E4 rescaled by) and the share of
called bins with `c < s`; the `high_missing` finding, which warns that
partly called bins bias theta and Ne low, should say that the fit corrects
for it when it uses the counts. The per-sequence θ0 behind the sex
chromosome check should then be per callable base too, so that sequences
with different missingness compare fairly.

**`.psmc` output and the JSON.** θ is still per `s` bases, so
`psmc_plot.pl -s s -u μ` scales a `.psmc` file correctly. The `MM` line's
`sum_L` and `sum_n` stay C's bin counts, and C's expected recombination count
(`n_recomb`, from `sum_L`) is unchanged, as recombination does not depend
on callability. The run's JSON records which mode was used.

## Command line (stage 2)

```text
--bin-callable auto|ignore    default auto
```

- `auto`: use the counts when the input has them (VCF and multihetsep),
  and C's model otherwise (psmcfa);
- `ignore`: drop the counts (`callable = None`) and reproduce
  `fq2psmcfa` and C psmc exactly, bias included.

The flag applies to inference and the bootstrap; `convert` and `qc` read
the input the same way in both modes. It is recorded in the output JSON
and in the command line that reproduces a run, and
`InputArgs::ignored_flags` lists `--bin-callable ignore` for psmcfa input,
where it changes nothing.

## Validation plan

Stage 1 (in the test suite):

- the counts of the VCF mask-mode reader against an independent per-base
  reference on random scenarios (`tests/io_vcf_exact.rs`);
- consensus-mode counts against the real `vcfutils.pl vcf2fq` output
  (`tests/data/route_a/expected/*.fq.gz`), counted with an independent
  port of `fq2psmcfa`'s per-base rule that must also reproduce the real
  `fq2psmcfa` bins;
- multihetsep counts against the placement rule;
- batching, bootstrap trunks and resampling carry the counts with the same
  slices, and every invariant violation is caught;
- every existing test, golden file and writer output unchanged.

Stage 2:

- unit level: the emissions with `c = s` equal the current ones bit for
  bit; the E-step with counts matches a brute-force forward-backward oracle
  that builds every bin's emission from its count; the new M-step
  objective and its gradient match finite differences, and reduce to the
  current objective with every `c = s`;
- identical to C when complete: a complete-data VCF fitted with `auto`, the
  same VCF with `ignore`, and C psmc on its `fq2psmcfa` psmcfa give
  identical `.psmc` output and identical JSON (apart from the recorded mode);
- unbiased under missing data: E4's masks (10% and 30% missing, clustered
  in 20 kb runs and dispersed in 20 bp runs) fitted with `auto` by the
  exact (VCF and mask) route must show no systematic shrink of θ, N0, times
  or Ne: RMSE of log10 Ne against the truth within replicate noise of the
  complete-data fit (0.332 at 30% dispersed, where E4's rescaling reached
  0.334), mean log ratio of fitted to true Ne near 0 at every time, while
  `ignore` reproduces E4's biased numbers exactly; the multihetsep route
  should come close to the exact one;
- the bootstrap and posterior decoding with counts: replicates
  reproducible from the seed with any thread count and batch size, and the
  decoding of complete data unchanged.
