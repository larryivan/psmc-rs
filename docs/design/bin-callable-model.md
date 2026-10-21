# Per-bin callable length

Design note for counting the callable bases of every bin and using them in
the PSMC emissions. The change comes in two stages:

1. **Input layer**: every reader that bins per-site data records how many
   bases of each bin are callable, and the counts travel with their bins
   through batching, the loaded dataset and the bootstrap. Nothing reads
   them yet, so no result changes.
2. **Model**: the E-step, M-step, starting values, bootstrap and posterior
   decoding use the counts, the command line gets `--bin-callable`, QC
   reports the counts, and the README, CHANGELOG and
   `docs/preprocessing.md` describe the result, including the library
   changes of stage 1 ([Changes made in stage 1](#changes-made-in-stage-1)).

Both stages are built. This note describes what was built; where stage 2
departs from the plan of stage 1 it says so and why
([What stage 2 built](#what-stage-2-built)), and
[Validation](#validation) gives the measured results.

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
low by about the missing fraction, or more (1.1 to 1.3 times it in the
fits of [Validation](#validation), 1.2 times it in experiment E4).
Experiment E4 (`experiment/pipeline/README.md`; five 500 Mb genomes of the
bottleneck model) measured it: with 30% of the bases missing in runs of
about 20 bp, the RMSE of log10 Ne against the truth was 0.449 on average,
against 0.273 for the same genomes without missing data; rescaling the
fitted times and sizes by the callable bases per called bin brought it
back to 0.286, and the model below to 0.280 (`E4_rescaling.csv`; an
earlier run of E4, before the model, gave 0.447, 0.332 and 0.334 for the
first three). Missing data in long runs (E4's 20 kb runs)
mostly empty whole bins, which the `N` rule already handles; only the bins
at the edges of each run are partly called.

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
    bin_size: u16,                    // s
    rows: Vec<BinCounts>,             // the counts of the bins of each row
}
```

`callable: None` means that every called bin holds `s` callable bases: the
input does not know better (psmcfa), and the model is C's. `Some` holds one
`BinCounts` per row, which gives one count per bin (`BinCounts::iter`). The
counts live outside the rows so that the E-step's hot loop over symbols is
untouched when there are none, and so that a run without counts costs no
memory.

Counts are `u16`, so they are recorded for bins of at most 65,535 bases
(`io::MAX_COUNTED_BIN_SIZE`). The readers accept any bin size, as before
stage 1, and bin larger bins without counts (`callable: None`, as for
psmcfa input). PSMC bins are normally 100 bases; a bin of 65,536 bases
holds dozens of heterozygous sites at human diversity, far outside the
model, so no use of the counts is lost.

**Storage.** Every copy of the data the CLI holds carries the counts of
its bins: `Dataset::obs`, and with a bootstrap also the joined sequences
(`BootstrapPlan::resample` in `src/bin/psmc/run/bootstrap.rs`, built
before the main fit), the trunks and the resampled data of each replicate
being fitted. One count
per bin would add 2 bytes (`u16`) or 4 (`u32`) per bin to each copy, next
to its 1-byte symbol, whether the counts carry information or not. A first
draft did so with `u32`, and the E-step's own buffers are no larger (one
alpha row per 128 bins of the sequence being processed,
`hmm::CHECKPOINT_STRIDE`: 1.4 to 4 bytes per bin for 23 to 64 states), so
the counts dominated. A `BinCounts` keeps its counts in the smaller of two
forms instead:

- runs: the bins whose count is not `s`, as runs of consecutive bins with
  the same count (start, length and count: 10 bytes per run). Full bins
  cost nothing, so complete data take no memory, and a stretch of missing
  data takes at most three runs whatever its length: a partly called bin
  at each end and the empty bins (count 0) between them;
- dense: one count per bin, 1 byte when every count fits in a byte (always
  for `s <= 255`, so for the usual 100) and 2 bytes otherwise. It is the
  smaller when there are more runs than a tenth of the bins (a fifth with
  2-byte counts), as when missing data come in stretches a few bins long
  or shorter.

So the counts never take more than 2 bytes per bin, nor more than the
symbols themselves for `s <= 255`. The form is chosen whenever counts are
built, cut or joined, and the builder behind the binner switches from runs
to one count per bin as soon as the runs would be larger, so that it never
holds much more than its result.

Peak resident memory (median of three runs) of `psmc-rs IN OUT 1
--vcf-mode mask --bin-size 100 --threads 3 --pattern 4+5*3+4`, alone (fit)
and with `--bootstrap 3 --bootstrap-iters 1` (fit + bootstrap): before
stage 1 (bf12203), with the first draft's `u32` per bin (2101d69) and with
the storage above. All three write the same JSON and bootstrap files byte
for byte.

Peak MB, fit / fit + bootstrap:

| input (mask-mode VCF, bins of 100) | bins | counts kept as | before stage 1 | `u32` per bin | stage 1 |
| --- | ---: | --- | ---: | ---: | ---: |
| complete: 1 Gb, no mask | 10 M | nothing | 32 / 86 | 75 / 298 | 32 / 87 |
| masked: 1 Gb, a mask with 407,594 gaps of 205 bp on average (8.4% missing, 12% of the bins not full) | 10 M | 1 byte per bin (10 MB) | 47 / 100 | 88 / 312 | 59 / 160 |
| clustered: 200 Mb, gaps of 20 kb (30% missing) | 2 M | 9,113 runs (0.09 MB) | 10 / 24 | 21 / 74 | 11 / 26 |
| dispersed: 200 Mb, gaps of 20 bp (30% missing) | 2 M | 1 byte per bin (2 MB) | 109 / 121 | 119 / 169 | 113 / 134 |

(The dispersed input's 3 million mask intervals dominate its memory.)

On complete data and on missing data in long stretches the counts cost
nothing. The masked input is the realistic case: a mask with many short
gaps leaves an eighth of the bins partly called or empty, in about as many
runs as such bins, so runs (10.3 MB) and one byte per bin (10.0 MB) cost
about the same, and the counts double each copy of the data. That is the
cost of the information itself kept beside the symbols. Stage 2, which
rewrites the E-step to read the counts, could remove it by coding each
bin's symbol and count in its one byte: for `s = 100` there are 192 valid
pairs (`T` or `K` with 10 to 100 callable bases, `N` with 0 to 9), up to
`s = 134` they fit in a byte (`2s - ceil(s / 10) + 2` pairs), and an
emission table indexed by that byte keeps the inner loop a lookup. As that
changes `Observations::rows` itself, it was left to stage 2, which kept the
counts beside the symbols ([What stage 2 built](#what-stage-2-built) says
why and what it costs).

The bootstrap's joined sequences are built before the main fit, to count
the trunks and find the replicates whose fresh start is invalid, and kept
until the bootstrap, as before stage 1. Building them only when the
bootstrap starts would not lower the peak, which comes during the
bootstrap, when the input, the joined sequences, the trunks and the
replicates are all held: on the masked input the fit alone peaks at
59 MB, the joined sequences add one copy of the bins and counts (20 MB) to
the main fit, and the run with the bootstrap peaks at 160 MB. Freeing the
joined sequences once the trunks are cut would save one copy there, for
every input; it is independent of the counts and left out.

`Observations::validate` checks every invariant in one place:

- `row_starts` has one entry per row, and the first is `true`;
- the counts have one `BinCounts` per row, for bins of `s` bases, with one
  count per bin;
- `1 <= s <= MAX_COUNTED_BIN_SIZE` (checked when the counts are made) and
  every `c <= s`;
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
returns the counts with the symbols, for bins of at most 65,535 bases. The
bins themselves, and so every psmcfa and multihetsep file psmc-rs writes,
are unchanged byte for byte.

| input | counts |
| --- | --- |
| VCF, mask mode | Exact: `s` minus the positions the per-site classification makes missing (outside the callable masks, filtered or missing genotypes, REF spans of heterozygous non-SNPs, gVCF blocks without a call, positions no gVCF record covers). |
| VCF, consensus mode | Exact: `s` minus the bases `fq2psmcfa` reads as missing (lowercase after vcf2fq's filters and `fq2psmcfa -q`, `N` and other non-IUPAC characters, positions without a record, INDEL windows, positions outside the optional masks), so `c` is the number of good bases of a full bin. |
| multihetsep | Approximate: multihetsep records only how many sites between two rows are callable (`nr_called`), not which. The reader places the uncalled ones at the start of each interval, right after the previous row, as it already does for the symbols, and the counts follow that placement. |
| psmcfa | None. |

Stage 1 expected the multihetsep approximation to be mild: within the
interval between two heterozygous rows every bin is `T` except the one that
holds the second row, and the likelihood of a run of `T` bins in one state
depends only on the sum of their counts, which the placement gets right.
The validation shows that the placement itself is not mild, with or
without the counts, when missing data are dispersed (θ 30% low with 30% of
the bases missing in 20 bp runs) and less so in long runs (θ 7 to 11% low
with 30% in 20 kb runs; [Validation](#validation)).
Placing all of an interval's uncalled sites right after its first row
turns scattered missing bases into a run of `N` bins after every
heterozygous site (27.5% of the bins `N` on one simulated genome of the
validation, with 30% of the bases missing in 20 bp runs, against 0.8% in
the exact binning), loses the heterozygous sites whose bin those uncalled
sites fill (7.1% of them), and leaves the bin of each heterozygous site
callable only up to it: `K` bins hold 58% of their bases callable on
average under the placement, against 77% in fact, while the `T` bins are
nearly all full. The
callable-length model takes such `K` bins as evidence of old coalescences,
and with the counts the fit came out worse than C's model on the same bins
([Validation](#validation)). So `--bin-callable auto` fits exact counts
only, and multihetsep input with C's model; `--bin-callable approximate`
fits its counts ([Command line](#command-line)). A VCF with a callable mask
gives exact bins and counts.

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
(both modes) and multihetsep input binned into bins of at most 65,535
bases, and leaves it `None` for psmcfa and larger bins. It is `Some` also
when every sequence was dropped (with no rows), so that it says whether the
input reports counts. `Dataset::sequence_callable` joins the counts of
each kept sequence as `Dataset::sequences` joins its bins. Per sequence,
the counts sum to the called positions (`SeqInfo::called`) plus the
padding of the last bin (`s * n_bins - extent`).

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
heterozygous and L the called bins: the θ at which the expected share of
heterozygous bins under a single coalescence time, `1 - exp(-θ)`, equals the
observed one. Stage 1 proposed `-ln(1 - K/L) * s / c̄`, with c̄ the mean
count of the called bins. Stage 2 takes the same moment estimate under the
callable-length model instead (`model::ThetaCounts::theta0`,
`initial_theta_with_callable`): θ0 solves

```text
sum over called bins i of (1 - exp(-θ0 c_i / s)) = K
```

The left side rises from 0 to L and is concave in θ0, so the root is
unique for `0 < K < L`, and Newton's method from `-ln(1 - K/L) s / max c`,
where the left side is at most K, climbs to it monotonically; it sums over
the distinct counts only. With every called bin of one count `c` the root
is `-ln(1 - K/L) s / c`, and with full bins C's value, to the bit (the code
takes those closed forms). Dividing C's value by `c̄ / s` instead starts
low whenever the counts vary: the probability of a heterozygous site is
concave in the count, so a mix of counts shows fewer heterozygous bins than
bins of the mean count would (`tests/em_callable.rs` checks both, and that
the estimate recovers θ on bins simulated under one coalescence time, to
2%, where C's is 40% low with 40% of the bases missing). Every θ0 uses this
rule:

- the main fit's start (`start_rates` in `src/bin/psmc/run/start.rs`), which
  `fit_start` also hands to QC;
- the start that `qc` and `convert` check when they have no fit of their
  own (`default_qc_report` in `src/bin/psmc/dataset.rs`), as a fit with
  `--bin-callable auto` would start: with the counts of a VCF, not with
  multihetsep's;
- the bootstrap's fresh starts (`replicate_initial_model_with_callable`),
  and `invalid_replicate_starts_sequences`, which adds the `ThetaCounts`
  of the drawn trunks and finds exactly the replicates the bootstrap
  skips;
- QC's per-sequence θ0 per callable base (`BinStats::theta0_callable`),
  next to C's `theta0_per_bin`.

When the fit does not use the counts (`--bin-callable ignore`, or `auto`
on multihetsep input) they are dropped before any of these, and every θ0
is C's.

**E-step.** The forward and backward passes multiply by `e(o_i, c_i, k)`
instead of `e(o_i, k)` (`src/hmm/obs.rs`). Each pass first scans the bins
and their counts (in parallel over rows) for the pairs of symbol and count
that occur, checks them as `Observations::validate` would (one count per
bin, none above `s`, `N` exactly when `10 c < s`), and gives each pair an
observation index: `T` bins by ascending count, then `K` bins, then one
index for every `N` bin. A table holds one emission row per index (at most
`2 (s - ceil(s / 10) + 1) + 1`, 183 for `s = 100`), built from the model's
rows as `e_T(c) = exp(ln e_T * c / s)` and `e_K(c) = 1 - e_T(c)`, which is
the callable-length model wherever `e_K = 1 - e_T`, as PSMC's emissions
are; full bins (`c = s`) get the model's own rows. The per-segment reader
(`fb::SeqObs`) turns each bin's symbol and count into its index (the
counts of a segment are copied out of their `BinCounts`, by a binary
search and fills for runs), and the kernel indexes the table as it indexed
the three symbol rows before. Without counts the indices are the symbols
and the table is the model's three rows: the same loops, at the same speed
(psmcfa fits measured before and after, below).

The E-step's sums are kept per observation index (`Partial::g`), merged in
work-item order as before (each item's as soon as the items before it are,
which bounds their memory: *Memory of the E-step's sums* below), and
reduced in index order at the end into the
expected symbol counts `gobs` (which stay what they were: C's `post_sigma`
uses them) and, with counts, `SufficientStats::callable`: `G_T` and the
`G_K` of each count (below). With every count `s` there is one index per
symbol and every one of these is C's value to the bit; results do not
depend on the thread count, the alpha-cache budget, nesting or batching,
with or without counts (`tests/estep_callable.rs`; the brute-force
comparison is in `tests/estep_oracle.rs`).

Every pass over the bins takes the counts the same way, as an
`Option<&CallableBases>` in its options, so that none computes the
likelihood of another model than the one fitted:

- `hmm::e_step_with_options` (`EStepOptions::callable`), EM's E-step for
  the main fit and every bootstrap replicate (`EmOptions::callable`);
- `hmm::log_likelihood_with_options`: EM's log-likelihood of the final
  model, and the CLI's `model_loglike`, the log-likelihood of a run of 0
  iterations;
- `hmm::write_tmrca_posterior_tsv_with_options` (`TmrcaOptions::callable`),
  the posterior decoding, whose `obs` column stays the symbol.

`hmm::e_step_streaming`, `e_step_streaming_with_progress`,
`log_likelihood`, `write_tmrca_posterior_tsv` and `em::em_train` keep
their signatures and C's model, so `tests/c_equivalence.rs` and every
caller without counts is untouched. Passing `&Observations` in place of
`rows` and `row_starts`, the other option the plan named, would have
changed every one of those signatures.

**M-step.** The emission part of the expected complete-data
log-likelihood becomes

```text
sum_k [ G_T(k) ln e_T(k) + sum_c G_K(c, k) ln(1 - e_T(k)^(c / s)) ]
G_T(k)    = sum over T bins i of γ_i(k) c_i / s
G_K(c, k) = sum over K bins i with c_i = c of γ_i(k)
```

since the log-probability of a `T` bin, `(c / s) ln e_T(k)`, is linear in
its count. The objective (`opt::objective::EmissionCounts`) evaluates it
as C's terms are evaluated, `ln(p + 1e-300)` with zero counts left out,
with one `K` entry per count: C's case is the one entry `(f = 1,
gobs[1])`, with `e = exp(-(θ avg) f)` and `1 - e`, which for `f = 1` are
C's values to the bit, so the objective, its adjoint gradient and the
whole M-step are C's when every bin is full. The `K` terms of each state
are summed before they join Q, in the forward-mode oracle too, whose value
equals the adjoint's to the bit. Each count adds an exponential, a
logarithm and a division per state to an evaluation. The transition part
is unchanged.

`em::hmm_q` evaluates the same Q with the E-step's emissions (for the `QD`
lines of the `.psmc` output), and `em::hmm_q0` its maximum: over the
transition probabilities as before, and over one emission parameter per
state, `e_T(k)`, which is the model's own freedom. The emission part of
each state, `-t μ + sum_f g ln(1 - exp(-f μ))` in `μ = -ln e_T`, is concave
with a derivative falling from `+inf` to `-t`, so Newton's method from the
left (from C's closed form with every count raised to `s`) finds its
maximum; with every `K` bin full it is C's closed form, with `G_T` in place
of `gobs[0]`. So `Q - Q0` stays at most 0 (checked on EM runs) and is C's
with full bins.

**Posterior decoding.** The TMRCA output decodes with the same per-bin
emissions as the fit, through the same table; its columns stay as they are
(`obs` is the symbol).

**`convert` and the writers.** psmcfa has no place for counts: a psmcfa
written from counted input holds exactly the bins `fq2psmcfa` would write,
and a fit of it is the `ignore` fit. `convert --psmcfa` warns when some
called bins of a VCF are partly callable, with their share and the bias,
and advises fitting the VCF itself; a psmcfa of multihetsep input holds the
bins that `auto` fits, as full bins. The multihetsep writer keeps
`nr_called`, from which the reader rebuilds approximate counts.

**QC.** `QcInput::callable` gives QC the counts, whether or not the fit
uses them (`QcContext::counts_ignored` says it does not). The report gives
their mean share over the called bins and their distribution in tenths
(`QcReport::callable`, a `CallableSummary`; per sequence in `BinStats`),
and a new finding, `PARTLY_CALLABLE_BINS`, the bias of treating partly
callable bins as full: a note when the fit corrects it, a warning when
`--bin-callable ignore` leaves the called bins less than 95% callable on
average (a note above), nothing when fewer than 1% of them are partly
callable. The bias exceeds the missing share (θ of the `ignore` fits of
the validation was 1.1 to 1.3 times the missing share of the called bins
below the complete-data fit), so a threshold at 90% would leave a bias of
more than 10% with a note. psmcfa input cannot tell; when at least 1% of its called bins
border a missing run, the note counts them and says how to measure the rest
from the VCF the psmcfa was made from. Multihetsep's counts do not measure
the bias, which the placement makes nearly every called bin hide, so the
finding is not given for multihetsep input. `APPROX_MISSING_PLACEMENT`
says instead that multihetsep's bins and counts follow its placement,
whether the fit uses the counts, and, as a warning when more than 5% of
the sites are uncalled (`APPROX_PLACEMENT_WARN_FRACTION`), the bias the
placement caused in the validation whatever `--bin-callable`: multihetsep
does not tell scattered missing data from long runs, and both bias θ, if
by different amounts ([Validation](#validation)). `HIGH_MISSING` says that
the fit corrects the bias of partly called bins when it uses exact counts,
and points to `APPROX_MISSING_PLACEMENT` for multihetsep.
With counts, sequences are compared per callable base
(`BinStats::theta0_callable`, `theta0_comparable`): the autosomal
reference, the table's het/kb and `vs ref`, `SEX_OR_ORGANELLE` (whose
expected `K` bins use the sequence's mean share) and `HET_OUTLIER`, so that
a sequence with more missing data does not look less heterozygous. The
table's column is then `het/kb*`, with a note under it, since the totals'
het per kb (and `QcSummary::het_per_kb` in the fit's JSON) stay per kb of
called bins; `QcSummary::het_per_callable_kb` gives the total per callable
base.

**`.psmc` output and the JSON.** θ is still per `s` bases, so
`psmc_plot.pl -s s -u μ` scales a `.psmc` file correctly. The `MM` line's
`sum_L` and `sum_n` stay C's bin counts, and C's expected recombination count
(`n_recomb`, from `sum_L`) is unchanged, as recombination does not depend
on callability. For input with counts, one more `MM` line follows
`n_seqs`: `bin_model:callable-length` (or `full-bins` with `ignore`),
`counts:exact` (VCF) or `approximate` (multihetsep), `mean_callable` and
`partial_bins`; psmc_plot.pl skips `MM` lines it does not know, and
psmcfa input gets none, so its output is C's. Bootstrap replicates fitted
with counts get the line without `counts`. The JSON records the same in
`bin_callable`: the mode, the model, the source of the counts and their
`CallableSummary`.

## Changes made in stage 1

Stage 1 changes no result and no option: every output, and every psmcfa
and multihetsep file psmc-rs writes, is byte for byte as before, and the
readers accept the same bin sizes as before. Its library changes, for the
*Library API* part of the CHANGELOG entry stage 2 writes:

- New `io::callable` module: `CallableBases`, `BinCounts` and
  `MAX_COUNTED_BIN_SIZE` (65,535), all three re-exported from `io`, and
  the counts' iterator `callable::Iter`; `Observations::validate`.
- New `io::Sequences`, whole sequences with optional counts
  (`From<Vec<Vec<u8>>>` for bins without counts), and
  `Sequences::validate`.
- `Observations` gains the public field `callable`, and
  `io::consensus::ConsensusSeq` the public field `callable`, which breaks
  code that builds them with a struct literal (add `callable: None`).
- `io::common::chunk_sequences` takes `impl Into<Sequences>` instead of
  `Vec<Vec<u8>>`, so a call that passes an un-annotated `.collect()` no
  longer compiles: the target type must be named
  (`.collect::<Vec<_>>()`, as `io::psmcfa::read_psmcfa` now does).
- New `io::common::Binner::finish_with_callable` and
  `io::input::Dataset::sequence_callable`. `Binner::new` still panics only
  for a bin size of 0.
- `io::input::load`, `io::vcf::read_vcf_with` (and `read_vcf`) and
  `io::mhs::read_mhs_with` (and `read_mhs`) return the counts in
  `Observations::callable` for bins of at most 65,535 bases.
- New `bootstrap::observations_to_sequences`,
  `split_sequences_for_bootstrap`, `resample_sequences` and
  `run_bootstrap_sequences`, through which the CLI's bootstrap now goes;
  `rows_to_sequences`, `split_for_bootstrap`, `resample_chunks` and
  `run_bootstrap_with_options` are unchanged.

## Command line

```text
--bin-callable auto|ignore|approximate    default auto
```

- `auto`: use exact counts (VCF binned into bins of at most 65,535
  bases), and C's model otherwise (multihetsep, psmcfa, and larger bins,
  for which the run says so);
- `ignore`: drop the counts (`callable = None`) and reproduce
  `fq2psmcfa` and C psmc exactly, bias included;
- `approximate`: as `auto`, and also multihetsep's approximate counts.

The plan had `auto` use every count the input has. Multihetsep's follow
the reader's placement of the uncalled sites, and with them the fit came
out worse than with C's model on the same bins ([Validation](#validation)),
so `auto` leaves them out and `approximate` keeps the comparison
reproducible.

The flag applies to inference and the bootstrap, and is an inference
option (`InferArgs`), not an input option: `convert` and `qc` read the
input the same way in both modes and report on the counts as `auto` would
use them. The run says which model it fits (`Callable bases (exact
counts): the fit weighs each called bin by them (--bin-callable auto)`),
and the QC report's `Callable bases` row says what the counts hold. For
input without counts, `--bin-callable ignore` and `approximate` get a
warning that they change nothing, in the run itself rather than in
`InputArgs::ignored_flags`, which lists options of `InputArgs`. The flag
is recorded in the output JSON (`bin_callable`) and in the command
line that reproduces a run (`InferArgs::to_args`, which the TUI shows).

## What stage 2 built

The model, the counts and their storage are as planned above. Stage 2
departs from the plan of stage 1 in six places, each said in its section:

- θ0 is the moment estimate of the callable-length model, not C's
  estimate divided by the mean share of callable bases, which starts low
  when the counts vary (*Starting values*);
- Q0, which the plan did not cover, is maximised over one emission
  parameter per state, so that `Q - Q0` stays at most 0 (*M-step*);
- the passes take the counts as an option rather than `&Observations`,
  so that the C-compatible entry points keep their signatures (*E-step*);
- `--bin-callable` is an inference option, and the warning for psmcfa
  input is the run's (*Command line*);
- `auto` fits exact counts only: multihetsep's approximate counts made the
  fit worse, so they need `--bin-callable approximate` (*Command line*);
- the multihetsep approximation is not mild with missing data, above all
  dispersed ones (*Counts from each input*, *Validation*).

The storage note proposed coding each bin's symbol and count in one byte.
Stage 2 keeps the counts in `BinCounts` beside the symbols: the E-step
copies one segment's counts at a time (128 bins) and maps them with the
symbols to observation indices, which costs 3.7% of a single-threaded
E-step on dispersed counts (0.961 s against 0.926 s per E-step on a 2 Mbin
genome with 30% of the bases missing in 20 bp runs, 64 states; medians of
three runs of three E-steps, `examples/bench_estep.rs --callable`) and 0.9%
when every bin is full (0.935 s against 0.927 s), and 3% with 8 threads
(0.231 s against 0.224 s). The coding would save one copy of the counts
(1 byte per bin) in each copy of the data, at the price of changing
`Observations::rows`, which every reader, writer and caller uses; it is
left out. psmcfa input runs the same loops as before: a fit of
HLemySub1 (18.9 Mbin, `4+25*2+4+6`, 3 iterations, 8 threads) took a median
of 7.24 s against 7.36 s before stage 2 (three runs each, interleaved), and
wrote the same JSON but for the new `bin_callable` record. Whole fits of
the dispersed genome above, 25 iterations with 8 threads, took 6.22 s with
the counts and 5.97 s without (`--bin-callable ignore`; 6.00 s before
stage 2), and of its complete version 5.84 s and 5.76 s.

**Memory of the E-step's sums.** The sums of a work item (`fb::Partial`)
hold one row of `np` doubles per observation index: up to 183 rows for
`s = 100` and about 1,800 for `s = 1000`, against three without counts.
The E-step first kept every work item's sums until all were done, up to
256 of them (`sched::MAX_WORK_ITEMS`), and a bootstrap runs about one
E-step per thread at once, so a bootstrap of an input with many sequences
or trunks took twice the memory with counts, and nine times with bins of
1,000 bases; time hardly changed. Work items now start in item order, from
a shared counter, and each item's sums are merged into the total as soon as
every item before it has been (`hmm::sched::fold_in_order`, which the
log-likelihood uses too): the same sums in the same order, so every result
is unchanged to the bit, and only the sums of items that finished before
an earlier one are held, about one per thread. Peak resident memory of
`psmc-rs IN OUT 1 --vcf-mask ... --pattern 4+25*2+4+6 --threads 8
--bootstrap 8 --bootstrap-iters 1` on replicate 1's dispersed 30% genome
of the validation cut into 500 contigs of 400 kb (three runs each):

| bins | before stage 2 (5ddbd43) | stage 2, sums kept to the end: auto / ignore | sums merged in order: auto / ignore |
| --- | ---: | ---: | ---: |
| 100 bases, 500 trunks | 162-163 MB | 307-308 / 142-144 MB | 103-106 / 80-83 MB |
| 1,000 bases, `--bootstrap-block-size 50000` (500 trunks) | 104-119 MB | 900-1051 / 107-128 MB | 98-115 / 65 MB |

The fit alone at 1,000 bases peaked at 165 MB with the counts and 66 MB
without before, and at 88 MB and 62 MB after. Every input gains from the
change, with or without counts: the transition sums of each item
(`np * np` doubles) were held to the end as well.

The change costs no time. psmcfa fits before (7942d83) and after it,
interleaved, medians of three runs, 8 threads, at load averages of 5 to 20
from other work: HLemySub1 (889 sequences, 3 iterations) 6.00 s against
7.24 s, and its bootstrap of 8 replicates of 2 iterations 29.9 s against
31.0 s (its large scaffolds come first in the file, and work items now
start in order); of seven runs each, HLhydTec1 (34 sequences, 3
iterations) 5.91 s against 6.15 s, the 500-contig genome above (25
iterations) 4.06 s against 4.09 s, and the dispersed genome above as
psmcfa (10 sequences, 25 iterations) 5.42 s against 5.35 s, with the same
CPU time (28.1 s and 27.0 s both ways). With the counts, the 25-iteration
fit of the dispersed genome took 5.95 s and 6.04 s in two series of runs
against 5.71 s and 5.62 s without them (`--bin-callable ignore`), 4 to 7%
more, and its counted bootstrap on 500 contigs 2.21 s against 2.26 s
before the change.

## Validation

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
- the storage: counts read back exactly whether built at once or bin by
  bin, cut and joined, always in the smaller form, never above 2 bytes per
  bin (1 for `s <= 255`) and nothing for full bins (`src/io/callable.rs`),
  and the loader's counts of complete, gapped and dispersed masks take the
  memory the forms promise (`tests/io_callable_readers.rs`);
- bins too large to count load as before, without counts;
- every existing test, golden file and writer output unchanged.

Stage 2 (in the test suite):

- the emission table: rows with `c = s` are the model's own, to the bit,
  and partly callable rows are `exp(-θ (c/s) avg_t)` and one minus it
  (`src/hmm/obs.rs`); the E-step and the log-likelihood with counts match
  a brute-force forward-backward that builds every bin's emission from its
  count, to 2e-13 relative (`tests/estep_oracle.rs`);
- with every called bin full, the E-step, the log-likelihood and the TMRCA
  output equal those without counts to the bit, with any batching; with
  one count `c` everywhere they equal those of the model whose `T`
  emission is the full bin's to the power `c / s`; with mixed counts they
  do not depend on threads, nesting, the cache budget or batching
  (`tests/estep_callable.rs`);
- the M-step: with counts, the adjoint objective equals the forward-mode
  oracle to the bit and its gradient to 1e-10 relative (or the
  double-double evaluation decides), the gradient matches Richardson
  central differences to 1e-6, and the value is minus Q evaluated
  directly from the model; with full counts the objective, the gradient
  and the whole M-step are C's to the bit (`src/opt.rs`);
- EM: θ0 (above); EM with full counts is C psmc's to the bit, in its
  history, Q and Q0; with partly callable bins it does not lower the
  likelihood and keeps `Q - Q0 <= 0`; Q0 equals a golden-section
  maximisation; and on bins simulated from the callable-length model with
  30% of the bases missing at random, the fit with counts recovers the
  diversity per bin (θ C_pi) of the fit of the same genealogy without
  missing data (ratio 0.999), where the fit without counts gives 0.70
  (`tests/em_callable.rs`);
- the bootstrap: each replicate is the fit of its resampled bins and
  counts, by hand; full counts give the replicates without counts to the
  bit; replicates do not depend on threads or batching; and
  `invalid_replicate_starts_sequences` finds exactly the replicates the
  bootstrap skips (`tests/bootstrap_callable.rs`);
- the command line: `--bin-callable ignore` on each route A VCF fits
  exactly, to the bit, what the real `fq2psmcfa`'s psmcfa of the same
  calls fits (model, log-likelihoods and every `.psmc` round), `auto` on
  complete data fits exactly what `ignore` and the psmcfa fit, and `auto`
  on partly callable bins records the model in the JSON, the `.psmc`
  output and the report, and is warned about by `convert`; `auto` on
  multihetsep fits exactly what `ignore` and its psmcfa fit, and
  `approximate` fits the counts (`tests/cli_bin_callable.rs`); QC's summary
  and findings (`tests/qc_callable.rs`);
- `tests/c_equivalence.rs` is unchanged and passes.

Simulations (not in CI; `experiment/validation/bin_callable.py`, numbers
and commands in `experiment/validation/README.md`): three runs of three
genomes of 10 × 20 Mb, two of the bottleneck model (seeds `BC` and `BC2`)
and one of the expansion model (`BCX`), with callable masks of none, 10%
and 30% dispersed (20 bp runs) and 30% clustered (20 kb runs) missing data,
fitted from the masked VCF with its mask and from its multihetsep (`-N25
-t15 -r5 -p 4+25*2+4+6`). θ relative to the complete-data fit of the same
genome, as the range over the nine genomes, and the RMSE of log10 Ne
against the truth and against the complete-data fit, as the range of the
three runs' means:

| mask | input | `--bin-callable` | model fitted | θ / complete-data fit | RMSE vs truth | RMSE vs complete-data fit |
| --- | --- | --- | --- | ---: | ---: | ---: |
| none | VCF + mask | auto, ignore | callable-length = full-bins | 1 | 0.230-0.341 | 0 |
| dispersed 10% | VCF + mask | auto | callable-length | 0.985-0.999 | 0.222-0.314 | 0.027-0.093 |
| dispersed 10% | VCF + mask | ignore | full-bins | 0.873-0.890 | 0.231-0.351 | 0.126-0.281 |
| dispersed 10% | multihetsep | auto | full-bins | 0.824-0.856 | 0.233-0.363 | 0.146-0.335 |
| dispersed 10% | multihetsep | approximate | callable-length | 0.773-0.820 | 0.239-0.371 | 0.151-0.358 |
| dispersed 30% | VCF + mask | auto | callable-length | 0.944-0.991 | 0.224-0.315 | 0.074-0.150 |
| dispersed 30% | VCF + mask | ignore | full-bins | 0.634-0.658 | 0.353-0.455 | 0.278-0.478 |
| dispersed 30% | multihetsep | auto | full-bins | 0.666-0.701 | 0.275-0.412 | 0.192-0.440 |
| dispersed 30% | multihetsep | approximate | callable-length | 0.524-0.601 | 0.300-0.425 | 0.225-0.467 |
| clustered 30% | VCF + mask | auto | callable-length | 0.959-1.003 | 0.224-0.330 | 0.061-0.125 |
| clustered 30% | VCF + mask | ignore | full-bins | 0.958-1.001 | 0.224-0.330 | 0.063-0.125 |
| clustered 30% | multihetsep | auto | full-bins | 0.886-0.928 | 0.220-0.343 | 0.118-0.249 |
| clustered 30% | multihetsep | approximate | callable-length | 0.885-0.928 | 0.220-0.342 | 0.118-0.243 |

- **Complete data**: `auto` and `ignore` give the same fit, to the bit.
- **Dispersed missing data.** `auto` removes most of the bias, not all:
  θ is 2 to 4% low on average with 30% of the bases missing (run means
  0.963 to 0.982, against 0.990 to 0.994 at 10%), and fitting 80 EM
  iterations instead of 25 does not change that (0.961 against 0.965 on
  one genome). Its curves are about as close to the truth as the
  complete-data fit's: the runs' mean RMSE against the truth is below it
  in five of the six runs and masks, 0.002 above it in the sixth. On E4's
  five 500 Mb genomes with 30% missing, replicate by replicate, it was
  above it in four of the five, by 0.007 on average and at most 0.020
  (`E4_routes.csv`), where `ignore` was 0.16 to 0.20 above it. `ignore`
  is biased by more than the missing share of the called bins, 1.1 to 1.3
  times it
  (θ 0.87 to 0.89 at a mean share of 0.90, 0.63 to 0.66 at 0.705), with an
  RMSE against the complete-data fit 3 to 7 times `auto`'s. Its RMSE
  against the truth is clearly higher at 30%, as E4 measured (0.449 on
  500 Mb genomes, against 0.273 with complete data), but at 10% it can
  match complete data (0.231 against 0.230 for the expansion model),
  where PSMC's own error dominates: θ and the RMSE against the
  complete-data fit measure the bias, the RMSE against the truth adds
  PSMC's error.
- **Clustered missing data** change little either way: the called bins
  hold 99.8% of their bases, and the runs' mean RMSEs against the
  complete-data fit of `auto` and `ignore` differ by at most 0.002, and
  in E4 single genomes did not differ more: by 0.001 to 0.002 with 10%
  clustered and 0.003 to 0.004 with 30%, `auto` the closer in every
  replicate.
- **Multihetsep.** No mode recovers the exact route, and missing data in
  long runs bias it too (θ 0.89 to 0.93 where the exact route is
  unbiased). The bias comes from the placement of the uncalled sites,
  which the fit with C psmc's model on the placed bins (`auto`) shows
  alone (*Counts from each input*); with the counts (`approximate`) the
  `K` bins that the placement leaves partly callable lower θ further,
  which is why `auto` does not use them. Placing the uncalled sites evenly
  over each interval, which would match dispersed missing data and not
  long runs, would change the reader's bins as well and is left out; QC
  warns when more than 5% of a multihetsep's sites are uncalled, and the
  VCF with its callable mask is the input to use.
