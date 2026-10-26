# Validation of `--bin-callable`

`bin_callable.py` checks, on simulated genomes, how much of the bias that
missing data scattered in short runs cause when every called bin is treated
as full (`--bin-callable ignore`, fq2psmcfa and C psmc's model) fitting the
callable bases of each bin removes (`--bin-callable auto`, the
callable-length model of `docs/design/bin-callable-model.md`), and what
multihetsep input, which does not say which sites are uncalled, gives with
and without its approximate counts. It is not part of CI; a run of three
replicates takes a few minutes on 8 cores.

## Design

- **Genomes.** Three replicates of a demographic model of the experiment
  pipeline (`config.MODELS`, the bottleneck model by default), 10
  chromosomes of 20 Mb each, simulated with msprime as
  `experiment/pipeline/simulate.py` does (μ = 2.5e-8, r = 1e-8 per bp per
  generation), with the seeds of `config.seed` under an experiment name
  (`BC` by default; `--model` and `--experiment` choose others).
- **Masks** (`experiment/pipeline/inputprep.py`): none; dispersed 10% and
  30% (uncallable runs of mean 20 bp); clustered 30% (mean 20 kb). The VCF
  records at uncallable sites are removed, as a caller would not call them.
- **Fits.** psmc-rs on each masked VCF with its mask (mask mode), with
  `--bin-callable auto` and `ignore`, and on the multihetsep of each mask
  with missing data (`psmc-rs convert --multihetsep`) with `--bin-callable
  auto`, which fits multihetsep with C psmc's model (it fits exact counts
  only), and `approximate`, which fits its counts: the psmc README's human
  recipe, `N_ITER 25 --t-max 15 --tr-ratio 5 --pattern 4+25*2+4+6`, bins of
  100.
- **Metrics** (`experiment/pipeline/metrics.py`): θ relative to the
  complete-data fit of the same replicate (the `none` mask, `auto`); RMSE of
  log10 Ne against the truth on the model's informative window (200 times
  between the 1% and 99% quantiles of the true coalescence time) and
  against the complete-data fit; the mean log10 ratio of fitted to true Ne
  over the window.

## Run

From the repository root, with the experiment's Python environment and a
release build:

```bash
cargo build --release
experiment/.venv/bin/python -m experiment.validation.bin_callable \
    --psmc-rs target/release/psmc-rs --out experiment/validation/results/BC
# the other two runs reported below
experiment/.venv/bin/python -m experiment.validation.bin_callable \
    --psmc-rs target/release/psmc-rs --out experiment/validation/results/BC2 --experiment BC2
experiment/.venv/bin/python -m experiment.validation.bin_callable \
    --psmc-rs target/release/psmc-rs --out experiment/validation/results/BCX --model expansion --experiment BCX
# B2's first two replicates with 80 iterations (the convergence check below)
experiment/.venv/bin/python -m experiment.validation.bin_callable \
    --psmc-rs target/release/psmc-rs --out experiment/validation/results/BC2_80 --experiment BC2 \
    --n-iter 80 --replicates 2
```

It writes to `--out` the fits (`rep01/` to `rep03/`, each fit's JSON
beside its inputs), `fits.json` (where each fit's JSON is, and its wall
time), `results.tsv` (one row per replicate, mask, route and mode) and
`summary.tsv` (means over the replicates, with the range of θ), and prints
the summary; `--evaluate-only` recomputes the tables from the fits there.
The simulations, masks and fits are deterministic: two runs gave the same
fits to the bit. `results/` keeps all of it but the inputs (the simulated
VCFs, masks and multihetsep files) and psmc-rs's HTML reports.

## Results

psmc-rs 0.2.0 (src tree `dcf29ca`), macOS arm64, 8 threads.
Three runs of three replicates each: the bottleneck model with the seeds of
`BC` (B1) and of `BC2` (B2), and the expansion model with those of `BCX`
(E), in `results/BC`, `results/BC2` and `results/BCX`, whose `summary.tsv`
the table rounds (`results.tsv` holds every fit; its seconds, like
`summary.tsv`'s, are rough wall times of a shared machine, see Timing).
Means over each run's three replicates, and the range of θ over all
nine; the mean share of callable bases of the called bins is B1's (the
multihetsep's is 0.98 for the expansion model's dispersed masks, whose
genomes are less heterozygous):

| mask | input | `--bin-callable` | model fitted | mean share of callable bases | θ / complete-data fit: B1, B2, E (range) | RMSE vs truth: B1, B2, E | RMSE vs complete-data fit: B1, B2, E |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| none | VCF + mask | auto, ignore | callable-length = full-bins | 1.000 | 1 | 0.307, 0.341, 0.230 | 0 |
| dispersed 10% | VCF + mask | auto | callable-length | 0.902 | 0.993, 0.990, 0.994 (0.985-0.999) | 0.306, 0.314, 0.222 | 0.027, 0.093, 0.031 |
| dispersed 10% | VCF + mask | ignore | full-bins | 0.902 | 0.881, 0.877, 0.884 (0.873-0.890) | 0.348, 0.351, 0.231 | 0.194, 0.281, 0.126 |
| dispersed 10% | multihetsep | auto | full-bins | 0.973 | 0.828, 0.825, 0.853 (0.824-0.856) | 0.363, 0.359, 0.233 | 0.240, 0.335, 0.146 |
| dispersed 10% | multihetsep | approximate | callable-length | 0.973 | 0.778, 0.776, 0.816 (0.773-0.820) | 0.371, 0.368, 0.239 | 0.256, 0.358, 0.151 |
| dispersed 30% | VCF + mask | auto | callable-length | 0.705 | 0.982, 0.969, 0.963 (0.944-0.991) | 0.309, 0.315, 0.224 | 0.074, 0.150, 0.100 |
| dispersed 30% | VCF + mask | ignore | full-bins | 0.705 | 0.652, 0.643, 0.645 (0.634-0.658) | 0.453, 0.455, 0.353 | 0.355, 0.478, 0.278 |
| dispersed 30% | multihetsep | auto | full-bins | 0.964 | 0.683, 0.672, 0.689 (0.666-0.701) | 0.406, 0.412, 0.275 | 0.308, 0.440, 0.192 |
| dispersed 30% | multihetsep | approximate | callable-length | 0.964 | 0.534, 0.529, 0.586 (0.524-0.601) | 0.421, 0.425, 0.300 | 0.335, 0.467, 0.225 |
| clustered 30% | VCF + mask | auto | callable-length | 0.998 | 0.976, 0.995, 0.962 (0.959-1.003) | 0.330, 0.295, 0.224 | 0.088, 0.125, 0.061 |
| clustered 30% | VCF + mask | ignore | full-bins | 0.998 | 0.975, 0.993, 0.961 (0.958-1.001) | 0.330, 0.295, 0.224 | 0.090, 0.125, 0.063 |
| clustered 30% | multihetsep | auto | full-bins | 0.998 | 0.903, 0.923, 0.888 (0.886-0.928) | 0.343, 0.316, 0.220 | 0.172, 0.249, 0.118 |
| clustered 30% | multihetsep | approximate | callable-length | 0.998 | 0.903, 0.923, 0.887 (0.885-0.928) | 0.342, 0.314, 0.220 | 0.172, 0.243, 0.118 |

- **Complete data**: `auto` and `ignore` give the same fit, to the bit.
- **Dispersed missing data, VCF with its mask.** `auto` removes most of
  the bias, not all of it. θ is 0.985 to 0.999 of the complete-data fit of
  the same genome with 10% of the bases missing, and 0.944 to 0.991 with
  30% (run means 0.963 to 0.982), so with 30% missing it stays 2 to 4% low
  on average; refitting one replicate (B2 replicate 2) with 80 iterations
  instead of 25 left it where it was (0.961 against 0.965;
  `results/BC2_80`), so the residual is not one of convergence. The curves are about as good as
  without missing data: the RMSE against the truth is below that of the
  complete-data fit in five of the six cases and 0.001 above it in the
  sixth (B1 at 30%: 0.3087 against 0.3074). `ignore` is biased by more than
  the missing share of the called bins: θ 0.873 to 0.890 at 10% and 0.634
  to 0.658 at 30%, 1.1 to 1.3 times the missing share below the
  complete-data fit, and its RMSE against the complete-data fit is 3 to 7
  times `auto`'s. Its RMSE against the truth is higher than `auto`'s at
  30% in every run (0.353 to 0.455 against 0.224 to 0.315), but at 10% its
  contrast with complete data depends on the run: 0.348 against 0.307
  (B1), 0.351 against 0.341 (B2) and 0.231 against 0.230 (E), where PSMC's
  own error on complete data dominates. θ and the RMSE against the
  complete-data fit measure the bias itself; the RMSE against the truth
  adds the error PSMC makes on complete data.
- **Clustered missing data** change little either way: the called bins
  hold 99.8% of their bases, and `auto` and `ignore` differ by at most
  0.002 in RMSE against the complete-data fit.
- **Multihetsep** does not recover the exact route under either mode, and
  its approximate counts make it worse. The reader places the uncalled
  sites of each interval, which multihetsep does not locate, at its start:
  on replicate 1 of B1 with 30% dispersed missing data that makes 27.5% of
  the bins `N` (0.8% in the exact binning; the fits' QC), loses the 7.1% of
  heterozygous sites whose bin the uncalled sites fill, and leaves the `K`
  bins 58% callable on average (77% in fact) while the `T` bins are nearly
  all full (a one-off comparison of the two binnings, not kept). With C psmc's model on those bins (`auto`), θ is 0.82 to 0.86 of
  the complete-data fit at 10% dispersed, 0.67 to 0.70 at 30%, and 0.89
  to 0.93 at 30% clustered, where the exact route is unbiased. The
  callable-length model reads the partly callable `K` bins as evidence of
  old coalescences, which lowers θ further (0.52 to 0.60 at 30%
  dispersed): this is why `--bin-callable auto` does not fit multihetsep's
  counts and `approximate` must be asked for. QC warns
  (`APPROX_MISSING_PLACEMENT`) when more than 5% of a multihetsep's sites
  are uncalled. For missing data of either kind, fit the VCF with its
  callable mask.

## Timing

The timing column of `summary.tsv` is indicative only, as other work
shared the machine. Measured on their own on an Apple M1 Pro (on AC
power, nothing else heavy running; medians of three interleaved runs),
whole fits (25 iterations, 8 threads, 2 Mbins, 64 states, VCF reading
included) of B1's dispersed 30% genome of replicate 1 took 3.99 s with
the counts and 3.69 s without, and of its complete version 3.62 s and
3.53 s; the E-step alone is 4.4% slower with dispersed counts (one
thread; 1.3% when every bin is full),
`examples/bench_estep.rs --callable`.
