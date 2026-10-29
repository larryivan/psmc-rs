# Fitting

```bash
psmc-rs INPUT OUTPUT N_ITER [OPTIONS]
```

fits PSMC to INPUT with N_ITER EM iterations (C psmc's `-N`) and writes the
parameters to OUTPUT (JSON), with an HTML report next to it
([outputs.md](outputs.md)). N_ITER 0 writes the starting model.

- [The model](#the-model)
- [The starting point](#the-starting-point)
- [The M-step](#the-m-step)
- [Log-likelihoods](#log-likelihoods)
- [The bootstrap](#the-bootstrap)

## The model

psmc-rs fits C psmc's model, with its time grid (psmc's `PSMC_T_INF` =
1000) and its options:

| option | C psmc | default | what it sets |
| --- | --- | --- | --- |
| `--pattern P` | `-p` | `4+5*3+4` | how the time intervals share free parameters (λ) |
| `--n-steps N` | `n` | from the pattern | the number of intervals minus one |
| `--t-max T` | `-t` | 15 | the starting maximum time, in units of 2N0 generations (1 to 200) |
| `--tr-ratio R` | `-r` | 4 | the starting ratio θ0/ρ0 |
| `--init-params FILE` | `-i` | | start from the parameters of an earlier run's JSON |

- **`--pattern`** groups the intervals into free parameters: `a*b` is `a`
  parameters each spanning `b` intervals, and a plain `b` is one parameter
  spanning `b` intervals. `4+5*3+4` has 23 intervals and 7 free λ;
  `4+25*2+4+6`, the psmc README's setting for modern humans, has 64 and 28.
  Each free λ needs enough recombinations in its intervals to be
  determined: psmc's documentation asks for about 10.
- **`--n-steps`** follows the pattern; given with `--pattern` it must
  agree. `--n-steps` alone fits one λ per interval, which costs time and
  memory with many intervals ([performance.md](performance.md#the-m-step)).
  A model has at most 10,000 states.
- **`--init-params FILE`** reads the parameters (`theta`, `rho`, `lam`,
  `t_max`, `n_steps`, `mu`, `pattern`) from the JSON of an earlier fit; the
  file's pattern is used.
- **`--mu`** (2.5e-8 per base and generation) and **`--gen-years`** (25)
  only scale the curve to individuals and years: N0 = θ / (4 μ × bin size).
- **`--bin-callable auto|ignore|approximate`** (default `auto`) says how a
  called bin of a VCF counts: by its callable bases (`auto`), or as
  `--bin-size` bases, as C psmc counts every bin (`ignore`); `approximate`
  also uses multihetsep's approximate counts
  ([inputs.md](inputs.md#partly-callable-bins)). For psmcfa input, which
  holds no counts, `ignore` and `approximate` change nothing, and the run
  says so.
- **`--smooth-lambda W`** (default 0, C psmc's objective) adds a penalty on
  the differences of log λ between neighbouring intervals.

## The starting point

The fit starts where C psmc starts: θ0 = −ln(1 − K/L) (K heterozygous of L
called bins; with callable counts, its callable-length analogue), ρ0 =
θ0 / `--tr-ratio`, every λ = 1 and t_max = `--t-max`. `--theta0` and
`--rho0` set θ0 and ρ0 directly.

**ρ0 has an upper limit.** PSMC's transition probabilities assume at most
about one recombination per bin. When ρ0 per bin is too large for the time
grid, some of them are negative and EM cannot move from the start (C psmc
then reports a flat curve). With t_max 15, flat λ and C psmc's default
pattern the limit is about 1/6 per bin, which C psmc's ρ0 = θ0/4 passes at
about 49% heterozygous bins (0.161 and about 47% with `4+25*2+4+6`).

psmc-rs checks the start before the main fit (also with N_ITER 0 when a
bootstrap starts from `--init-params`) and stops with an error that gives
the limit and what to change: smaller bins, a larger `--tr-ratio` or a
smaller `--rho0` (with the ratio the bootstrap replicates need, when that
is more), or, with `--init-params`, a smaller rho in the saved file. QC
reports the same check as `INVALID_START`.

## The M-step

Each M-step maximises C psmc's objective over all free parameters with
L-BFGS and runs to convergence, as C psmc's Hooke–Jeeves search does.

- **`--mstep-iters N`** (1000) caps its iterations, as a safeguard that
  normal fits do not reach. In the manuscript's 40 simulated fits with
  `4+25*2+4+6`, an M-step took a median 76 evaluations of its objective
  and at most 135 (the `.psmc` output's `IT`, which counts the evaluations
  for its curvatures and its stopping check too, two per free parameter),
  where C psmc's search took a median 3,934.
- **Each λ stays between 1e-6 and 1e4** (C psmc's search has no bound).
  Beyond 1e4 the transition probabilities of the shortest, earliest
  intervals are rounding noise in double precision; a λ of 1e4 already
  means that the data hold almost no coalescences in its intervals, so the
  Ne there is the bound, not an estimate. This is common for the most
  recent intervals when each has its own λ, and the run warns when a
  fitted parameter sits on a bound.

## Log-likelihoods

`Last EM loglike`, printed at the end, is the log-likelihood of the written
model, which C psmc prints as the LK of round N_ITER + 1. The JSON's
`loglike[i]`, the log-likelihood after i iterations, is C psmc's LK of
round i + 1. With N_ITER 0 the run evaluates the starting model, so
`loglike` holds its value alone.

## The bootstrap

`--bootstrap N` fits N replicates after the main fit, as C psmc's
`splitfa` and `psmc -b` do:

1. every sequence is cut into trunks of `--bootstrap-block-size` bins
   (default 500,000, splitfa's 50 Mb at 100 bp); a remainder shorter than
   1.5 trunks stays in the last trunk;
2. each replicate draws trunks with replacement from the whole input until
   it reaches the original length;
3. each replicate is fitted from a fresh start (the main fit's starting
   t_max and θ0/ρ0 ratio, θ0 from the resampled data) for
   `--bootstrap-iters` iterations (default N_ITER).

With `--init-params`, the replicates start from the saved parameters
instead, as `psmc -i -b` does; the run warns that a start fitted to the
whole input narrows the bands.

- **Parallel and reproducible.** Replicates run in parallel, and replicate
  i draws from its own seeded ChaCha8 stream (`--bootstrap-seed`, 42), so a
  seed gives the same replicates on any platform and with any `--threads`.
- **Replicates past the start limit.** A replicate's fresh start takes θ0
  from its own resampled data, so when the main start is close to the
  [limit on ρ0](#the-starting-point), the replicates more heterozygous than
  the input can start past it, where EM cannot fit. The run finds them
  before the main fit and warns that they will be skipped, that the bands
  then come from the other replicates and can be too narrow, and which
  `--tr-ratio` (or `--rho0`) would fit them all; it stops before the main
  fit when that is every replicate. Skipped replicates get no files, so the
  numbering has gaps. Replicates started from `--init-params` share the
  main fit's start and are never skipped.
- **Enough trunks.** An input that forms a single trunk is refused before
  fitting, since every replicate would equal the input; with fewer than 20
  trunks the run warns that the bands are coarse. Pick a trunk size that
  leaves a few dozen trunks: `psmc-rs qc INPUT --bootstrap-block-size BINS`
  counts them.

**Read the bands as a lower bound on the uncertainty**, not as an interval
for the true history: the bootstrap measures the sampling spread of the
fit, not PSMC's bias. In the manuscript's calibration (experiment E5), the
95% bands held the true Ne at fewer than half of the times
([validation.md](validation.md#bootstrap-calibration)). A small version of
that check runs in a few minutes:
`cargo run --release --example bootstrap_calibration -- --threads 4`.

The files the bootstrap writes, and how to plot them with `psmc_plot.pl`,
are in [outputs.md](outputs.md#bootstrap-files).
