# Outputs

| file | written | contents |
| --- | --- | --- |
| `OUTPUT` (JSON) | always, right after the main fit | the parameters and how the run made them ([JSON](#json)) |
| `OUTPUT` with `.html` in place of `.json` | always | the interactive report ([HTML report](#html-report)) |
| `--psmc-out FILE` | on request | C psmc's `.psmc` output, rounds 0 to N_ITER ([.psmc](#psmc-for-psmc_plotpl)) |
| `--tmrca-out FILE` | on request | the posterior TMRCA of every bin (TSV, gzip when FILE ends in `.gz`; [TMRCA](#tmrca-tsv)) |
| `--qc-json FILE` | on request | the input QC report ([qc.md](qc.md); `-` for stdout, and the result lines then go to stderr) |
| `--bootstrap-dir DIR` | with `--bootstrap` | the replicates and their summary ([bootstrap files](#bootstrap-files)); default `OUTPUT` with `.bootstrap` in place of `.json` |

OUTPUT may not end in `.html`, the name of the report. Output paths are
checked before the input is read: each must be writable,
no two of them, nor any of them and the input, may be the same file, and
only the QC JSON can be `-` (stdout). The `PSMC_ECHARTS_JS` bundle (see
[HTML report](#html-report)) is read then too.

The outputs do not depend on `--threads` or `--batch-size`
([performance.md](performance.md#determinism)).

## JSON

- **The parameters**: `theta` and `rho` (per bin), `lam` (one value per
  free parameter of the pattern, or per interval without one), `t_max`,
  `n_steps`, `mu` and `pattern`; `--init-params` reads them back, but for
  `mu`.
- **The run**: `version`; `simd`, the HMM kernels that ran (`avx2+fma`,
  `portable` or `neon`); `n_iter`; `final_loglike`; `loglike`, N_ITER + 1
  values ([fitting.md](fitting.md#log-likelihoods)); `bin_size`,
  `gen_years`, `tr_ratio` and `smooth_lambda`.
- **`input`**: the format, the VCF mode, and the numbers of sequences,
  bins, heterozygous and missing bins, and dropped sequences.
- **`qc`**: the codes of the QC warnings and notes, and key metrics (such
  as `qc.het_per_kb` and `qc.het_per_callable_kb`).
- **`bin_callable`**: the `--bin-callable` mode; the emission model fitted
  (`callable-length`, each called bin by its callable bases, or
  `full-bins`, C psmc's); how the input's counts were obtained (`exact`
  from a VCF, `approximate` from multihetsep; absent for psmcfa); and what
  they hold (`callable`: the called and partly callable bins, the mean
  share of callable bases and its distribution in tenths).

Bootstrap replicate JSON files hold the parameters only.

## HTML report

The report shows:

- the Ne curve, with the bootstrap's median and 95% band when there are
  replicates;
- a run summary: the input, the bin model and callable bases, the
  log-likelihood, the pattern, the parameters, N0, the bin size, the
  generation time, the starting values, the smoothing and the M-step cap;
- the QC findings, metrics and per-sequence table;
- the bootstrap settings (and skipped replicates), with a caveat on what
  the band covers;
- with `--tmrca-out`, TMRCA panels.

Charts can be restyled and saved as PNG. The page loads ECharts from
cdn.jsdelivr.net. For offline use, set `PSMC_ECHARTS_JS` to a local
`echarts.min.js` when running psmc-rs, which embeds it; without either,
the page says that it cannot draw the charts.

## .psmc for psmc_plot.pl

The `.psmc` file follows C psmc's format; the differences are listed below.
It does not record the bin size, so give psmc_plot.pl the run's settings.
The run prints the command on stderr (not with `-q`), for example

```text
To plot it: psmc_plot.pl -s 100 -u 2.5e-8 -g 25 -n 25 PREFIX fit.psmc (-s is the bin size, which the .psmc file does not record)
```

psmc_plot.pl plots round 20 unless given `-n`, so pass `-n N_ITER`, as the
printed command does: a run with fewer iterations has no round 20, and in
a longer one round 20 is not the model of the JSON and the report.

**With the bootstrap.** To plot the replicates as C psmc users do,
concatenate the files:

```text
To plot the fit with its bootstrap replicates: cat fit.psmc fit.bootstrap/replicate_*.psmc > combined.psmc; psmc_plot.pl -s 100 -u 2.5e-8 -g 25 -n 25 PREFIX combined.psmc
```

The run prints this recipe when the replicates have N_ITER iterations, the
default of `--bootstrap-iters`, and `--psmc-out` is given. The glob takes
every replicate file in the directory, which holds only this run's
([bootstrap files](#bootstrap-files)). `-n` picks the same round in every
file of the combined one, so the recipe needs replicates of N_ITER
iterations: with fewer, the replicates have no round N_ITER and
psmc_plot.pl stops with a division by zero; with more, it plots each
replicate at round N_ITER instead of its final fit. Plot such replicates on
their own, with `-n` set to their iteration count; the run warns and prints
that command.

**Differences from C psmc's output:**

- the `MM Version` line reads `psmc-rs` and its version;
- for VCF and multihetsep input, one more `MM` line after `n_seqs` says how
  the fit used the callable bases of each bin, such as
  `bin_model:callable-length, counts:exact, mean_callable:0.923136,
  partial_bins:0.271512` (bootstrap replicates fitted with the counts get
  it without `counts`; psmcfa input gets none, and psmc_plot.pl skips `MM`
  lines it does not know);
- `IT` counts the M-step's objective evaluations, those for its
  curvatures and its stopping check included, which depend on the data and
  the pattern (per round with the default pattern, 34 to 74 on the
  180,000-bin test input, `tests/data/c_reference/papuan_chr1_3seg.psmcfa`,
  where C psmc's Hooke–Jeeves search needs 1,132 to 1,540);
- `post_sigma` and `RI` include the emission of each sequence's last bin,
  which C psmc skips (differences of the order of the number of sequences
  divided by the called length: 2e-5 on the reference data);
- there is no `QD` line. C psmc's `QD` is its EM Q before and after the
  M-step less Q's largest value over all HMMs, which needs every expected
  transition count, and psmc-rs does not keep them
  ([performance.md](performance.md#the-e-step)). psmc_plot.pl does not read
  `QD`.

## TMRCA TSV

`--tmrca-out FILE` writes one line per bin: `seq_id`, `seq_bin`,
`global_bin`, `obs` (`T`, `K` or `N`), `map_state`, `tmrca_map_years`,
`tmrca_mean_years`, `tmrca_q025_years`, `tmrca_q975_years`, and the `pmax`
and `entropy` of the posterior.

- `seq_id` numbers the fitted sequences from 0, in input order; sequences
  that the input's filters left out are not counted. `seq_bin` is the bin's
  index within its sequence, `global_bin` its index over all fitted
  sequences.
- Each state is placed at its average coalescent time, scaled to years
  with N0 (from θ, `--mu` and `--bin-size`) and `--gen-years`.
- Each bin is decoded with the emissions the fit used (of its callable
  bases, with the callable-length model).
- Decoding runs in parallel within and across sequences.

## Bootstrap files

- `replicate_NNN.json` and `replicate_NNN.psmc`, numbered by replicate from
  001 (skipped replicates leave gaps, [fitting.md](fitting.md#the-bootstrap)).
- `summary.tsv`: `x_years`, `ne_main`, `ne_q025`, `ne_q500` and `ne_q975`
  at the main curve's time points.
- `summary.json`: the same points, with the number of replicates summarised
  (`n_replicates`), the skipped ones (`skipped_replicates`, numbered from 1,
  `[]` when none), the trunk size (`block_size`) and the seed.

When the bootstrap starts, after the main fit's JSON and `--psmc-out` are
written, it removes every `replicate_NNN.json`, `replicate_NNN.psmc`,
`summary.tsv` and `summary.json` in its directory, with a warning that
names the earlier replicates beyond this run's count; other files are left
alone. A bootstrap that fails or is stopped part way therefore leaves only
the JSON files of the replicates it finished, never files of an earlier
run. A fit without `--bootstrap` warns when that directory holds
replicates.

## Console

- **stdout** carries only results: `Last EM loglike`, `Parameters`, `PSMC
  output`, `QC report`, `Bootstrap ...`, `TMRCA posterior`, `HTML report`.
- **stderr** carries how the input was read (for psmcfa, C psmc's `n_seqs`,
  `sum_L` and `sum_n`; for a VCF, the mode and why, the filters, and the
  bases each removed), the QC report, the model, warnings and progress
  bars.
- `-q` keeps the results and warnings; `--no-progress` drops the bars.

## Exit status

- **0** on success.
- **1** on an error: stderr gets `error: ...` with its causes (warnings are
  `warning: ...`). A file that cannot be read or written is an error.
- **2** on a usage error, and for `qc --strict` when the report has a
  warning. Usage errors are options that are invalid alone or together,
  also when this is found after parsing (a `--pattern` that does not parse,
  outputs that name the same file, `--psmc-out -`, an OUTPUT ending in
  `.html`, `--bootstrap-iters` or `--bootstrap-dir` without `--bootstrap`). Without arguments,
  `psmc-rs`, `psmc-rs convert` and `psmc-rs qc` print their help, with
  status 2.
