# Validation of `--bin-callable`

`bin_callable.py` checks, on simulated genomes, that fitting the callable
bases of each bin (`--bin-callable auto`, the callable-length model of
`docs/design/bin-callable-model.md`) removes the bias that missing data
scattered in short runs cause when every called bin is treated as full
(`--bin-callable ignore`, fq2psmcfa and C psmc's model). It is not part of
CI; a run takes about 15 minutes on 8 cores.

## Design

- **Genomes.** Three replicates of the bottleneck model of the experiment
  pipeline (`config.BOTTLENECK`), 10 chromosomes of 20 Mb each, simulated
  with msprime as `experiment/pipeline/simulate.py` does (μ = 2.5e-8, r =
  1e-8 per bp per generation), with the seeds of `config.seed` under the
  experiment name `BC`.
- **Masks** (`experiment/pipeline/inputprep.py`): none; dispersed 10% and
  30% (uncallable runs of mean 20 bp); clustered 30% (mean 20 kb). The VCF
  records at uncallable sites are removed, as a caller would not call them.
- **Fits.** psmc-rs on each masked VCF with its mask (mask mode), with
  `--bin-callable auto` and `ignore`, and both ways on the multihetsep of
  the dispersed 30% mask (`psmc-rs convert --multihetsep`), whose counts
  are approximate: the psmc README's human recipe, `N_ITER 25 --t-max 15
  --tr-ratio 5 --pattern 4+25*2+4+6`, bins of 100.
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
    --psmc-rs target/release/psmc-rs --out /tmp/bin_callable
```

It writes `results.tsv` (one row per replicate, mask, route and mode),
`summary.tsv` (means over the replicates) and `fits.json` to `--out`, and
prints the summary; `--evaluate-only` recomputes the tables from the fits
there. The simulations, masks and fits are deterministic: two runs gave
the same fits to the bit.

## Results

psmc-rs of branch `callable/model` (the model as of commit `13cb912`),
macOS arm64, 8 threads. Means over the three replicates (the RMSE against
the truth with its standard deviation over replicates):

| mask | input | `--bin-callable` | mean share of callable bases (called bins) | θ / complete-data fit | RMSE vs truth (sd) | RMSE vs complete-data fit | mean log10(fitted / true Ne) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| none | VCF + mask | auto | 1.000 | 1.000 | 0.307 (0.030) | 0 | -0.077 |
| none | VCF + mask | ignore | 1.000 | 1.000 | 0.307 (0.030) | 0 | -0.077 |
| dispersed 10% | VCF + mask | auto | 0.902 | 0.993 | 0.306 (0.030) | 0.027 | -0.075 |
| dispersed 10% | VCF + mask | ignore | 0.902 | 0.881 | 0.348 (0.020) | 0.194 | -0.134 |
| dispersed 30% | VCF + mask | auto | 0.705 | 0.982 | 0.309 (0.019) | 0.074 | -0.087 |
| dispersed 30% | VCF + mask | ignore | 0.705 | 0.652 | 0.453 (0.010) | 0.355 | -0.270 |
| dispersed 30% | multihetsep | auto | 0.964 | 0.534 | 0.421 (0.009) | 0.335 | -0.150 |
| dispersed 30% | multihetsep | ignore | 0.964 | 0.683 | 0.406 (0.009) | 0.308 | -0.182 |
| clustered 30% | VCF + mask | auto | 0.998 | 0.976 | 0.330 (0.062) | 0.088 | -0.080 |
| clustered 30% | VCF + mask | ignore | 0.998 | 0.975 | 0.330 (0.061) | 0.090 | -0.081 |

- **Complete data**: `auto` and `ignore` give the same fit, to the bit.
- **Dispersed missing data**: `auto` recovers the complete-data fit. θ is
  within 0.7% (10% missing) and 1.8% (30%) of the complete-data fit of the
  same genome, and the RMSE against the truth (0.306, 0.309) is that of
  complete data (0.307). `ignore` is biased by about the missing share: θ
  at 0.88 and 0.65 of the complete-data fit, the curve 0.06 and 0.19 log10
  units below it on average, and the RMSE against the truth 0.348 and
  0.453 (E4 measured 0.447 at 30% on 500 Mb genomes).
- **Clustered missing data**: the called bins hold 99.8% of their bases,
  and `auto` and `ignore` differ by 0.002 in RMSE against the complete-data
  fit.
- **Multihetsep** does not recover the exact route, with or without the
  counts (RMSE 0.421 with `auto`, 0.406 with `ignore`, against 0.309 for the
  VCF with its mask). The bias comes from how the reader places the
  uncalled sites that multihetsep does not locate (all at the start of the
  interval after each row): on replicate 1 it makes 27.5% of the bins `N`
  (0.8% in the exact binning), loses the 7.1% of heterozygous sites whose
  bin the uncalled sites fill, and leaves the `K` bins 58% callable on
  average (77% in fact) while the `T` bins are nearly all full. The
  callable-length model reads those `K` bins as evidence of old
  coalescences, which makes the fit with the counts slightly worse than
  without. For dispersed missing data, fit the VCF with its callable mask.

## Timing

The fits took 5.7 to 6.7 s each (25 iterations, 8 threads, 2 Mbins, 64
states, VCF reading included); the timing column of `summary.tsv` is
indicative only, as other work shared the machine. Measured on their own
(medians of three interleaved runs), whole fits of the dispersed 30%
genome of replicate 1 took 6.22 s with the counts and 5.97 s without, and
of its complete version 5.84 s and 5.76 s; the E-step alone is 3.7% slower
with dispersed counts (one thread; 0.9% when every bin is full),
`examples/bench_estep.rs --callable`.
