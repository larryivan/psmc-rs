# Experiment pipeline

`experiment/pipeline/` produces the manuscript's experiments: it simulates
genomes, runs C psmc and psmc-rs with identical settings, records how every
output was made, and writes the tables. It replaces `experiment/scripts/`,
which stays until every experiment has moved here. Experiments E1, E4
and E5 are implemented; the others have their parameters in `config.py`
already.

Run it from the repository root with the experiment's environment
(`experiment/.venv`, see `experiment/requirements.txt`):

```bash
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools          # what will run: tool paths, versions, commits
$PY -m experiment.pipeline plan E1        # CPU hours and wall time on the budget
$PY -m experiment.pipeline start E1       # detached runner; returns at once
$PY -m experiment.pipeline status         # done/running/failed/pending, ETA
$PY -m experiment.pipeline stop           # or: stop --drain (finish running jobs first)
$PY -m experiment.pipeline start E1       # resume: only what is not done runs
```

`--profile smoke` runs the same jobs at a small size (E1: 2 models x 2
replicates x 2 chromosomes of 2 Mb, 3 EM iterations, under a minute; E4: 1
replicate of 2 chromosomes of 2 Mb with every mask and route, 3 EM
iterations, 78 jobs in about 15 s) in `experiment/.runs-smoke/`;
`--profile tiny` is for the unit tests.

## Modules

| module | contents |
| --- | --- |
| `config.py` | every parameter: inference settings, scaling, simulation, demographic models, seed scheme, experiments E1-E6, resource budget and cost model, profiles |
| `paths.py` | repository, tools directory (`PSMC_EXP_TOOLS`), runs directory (`PSMC_EXP_RUNS`) |
| `demography.py` | models as `msprime.Demography`; true Ne(t) and pairwise coalescence-time distribution in closed form; checks against msprime's `DemographyDebugger` and stdpopsim |
| `simulate.py` | one diploid genome: psmcfa, bgzipped VCF, truth record; true TMRCA track on demand |
| `tools.py` | psmc-rs 0.2 (this checkout), psmc-rs 0.1 (built from e50d69d), C psmc and utilities: locate, build, identify; exact command lines |
| `results.py` | C `.psmc` (every round: LK, QD, RI, TR, MT, RS, PA) and psmc-rs JSON as PSMC models and Ne(t) curves |
| `metrics.py` | time grids, RMSE of log10 Ne against the truth and between tools |
| `provenance.py` | checksums and provenance records: when a job is done |
| `runner.py` | the job graph, scheduler, detached daemon, status |
| `experiments/e1.py` | E1 as jobs, and its tables |
| `inputprep.py` | E4's data steps: callable masks, masked VCF, consensus FASTA through fq2psmcfa, bins of the routes compared, generate_multihetsep.py cross-check (`python -m experiment.pipeline.inputprep STEP`) |
| `experiments/e4.py` | E4 as jobs, and its tables |
| `figures/e4.py` | Fig 5, from E4's tables |
| `experiments/e5.py` | E5 as jobs, and its tables |
| `figures/e5.py` | Fig 6, from E5's tables |
| `experiments/fig.py` | the figures as runner jobs (experiment `FIG`) |
| `figures/` | the manuscript's figures: style, table schemas, one module per figure ([README](figures/README.md)) |

## Design

**Inference.** Every fit, by either tool, uses the psmc README's human
recipe: `psmc -N25 -t15 -r5 -p "4+25*2+4+6"`, which psmc-rs gets as
`N_ITER 25 --pattern 4+25*2+4+6 --t-max 15 --tr-ratio 5 --smooth-lambda 0
--mstep-iters 1000 --bin-size 100`, plus `--threads` per job. Simulations
are scaled with mu = 2.5e-8 and g = 25.

**M-step budget.** C psmc's Hooke-Jeeves search runs every M-step to
convergence; psmc-rs's L-BFGS stops after `--mstep-iters` iterations, 100 by
default, and after 25 EM iterations the two can then sit at different points
of a flat likelihood surface. On simulated bottleneck genomes the RMSE of
log10 Ne between psmc-rs and C psmc on the informative window was 0.156
(40 Mb) and 0.046 (300 Mb) with the default, 3.6e-3 and 3.2e-3 with 200,
and 2.9e-4 and 5.3e-5 with 1000 (largest differences 1.6, 0.17 and 1e-3
log10 units). Every psmc-rs 0.2 fit therefore uses 1000. E1 also fits every
genome with the default (`fit-rs-default`, `E1_mstep.csv`) and E3 times
psmc-rs 0.2 with the default too (`rs02-default`), so that the paper can say
what the default costs in agreement and gains in speed. The price is an
M-step on one thread whose length varies between genomes: on 300 Mb and one
thread, 37.5 s with 100, 80 s with 300 and 248 s with 1000 L-BFGS
iterations, against 471 s for C psmc; on 40 Mb, 7 s and 166 s. It dominates
the cost of E2's and E5's bootstraps and limits psmc-rs's thread scaling.

**Simulation.** Direct msprime (no stdpopsim engine): `sim_ancestry` with
the Hudson coalescent and a discrete genome for one diploid individual,
then `sim_mutations` with JC69; mutation rate 2.5e-8, recombination rate
1e-8. Chromosomes are unlinked and simulated independently. The models
`constant`, `bottleneck` and `expansion` are the ms models of the earlier
pipeline (time in 4*N0 generations, size relative to N0), converted exactly
(`config.ms_model`); `zigzag` is stdpopsim 0.3.0's `HomSap/Zigzag_1S14`,
recorded in `config.py` and checked against stdpopsim epoch by epoch
(`$PY -m experiment.pipeline check-models`). Each replicate writes a psmcfa
(one record per chromosome, laid out as fq2psmcfa writes it, `K` where a
100 bp bin holds a heterozygous site), a bgzipped and indexed VCF (with
`##contig` lengths and the phased GT of the diploid; `psmc-rs convert` turns
it into the identical psmcfa) and a truth record (settings, seeds, software
versions, per-chromosome counts and checksums, the true Ne(t) epochs and
coalescence-time quantiles). `simulate.write_tmrca` re-simulates a genome
from its truth record, checks it against the recorded checksums and writes
the true pairwise TMRCA as run-length intervals.

**Seeds.** `config.seed(experiment, model, replicate, stage)`: the first 8
bytes of SHA-256 of `psmc-rs-manuscript/<experiment>/<model>/<replicate>/<stage>`
modulo 2^31 - 1, plus 1. Stages are `ancestry/<chrom>`, `mutations/<chrom>`
and, for later experiments, `mask/<chrom>` and `bootstrap`. A seed depends on
nothing but its key. Every seed is in the truth record, the job's provenance
record and `tables/E1_seeds.csv`.

**Metrics.** Curves are compared at 200 log-spaced times. The informative
window of a model runs from the 1% to the 99% quantile of the true pairwise
coalescence time (in years): PSMC reads Ne(t) from the coalescence times of
the two haplotypes along the genome, and outside this range less than 1% of
the genome carries information, so both tools only extend their first and
last lambda there. The window depends on the model alone, so it is the same
for both tools and all replicates (constant: 5.0 ky to 2.30 My; bottleneck:
10.1 ky to 1.32 My; expansion: 5.0 ky to 1.05 My; zigzag: 7.1 ky to 2.17 My).
A fixed window, 10 ky to 1 My, is reported next to it. The truth is
evaluated exactly at each time (exponential growth included); fitted curves
are C psmc's step functions, scaled as `psmc_plot.pl` does. The first free lambda of `4+25*2+4+6` spans
intervals 0-3 (about 0-28 ky for zigzag, whose window starts at 7 ky), so
part of the RMSE against the truth is the error of approximating the truth
by the pattern's steps. Each fit is therefore also compared with the truth
projected onto its own free-parameter spans (the true Ne averaged over each
span in coalescence-rate terms, `metrics.projected_truth`): that RMSE is the
estimation error, and the projection's RMSE against the truth the pattern's.

## Runner

Experiments declare jobs: a command, input and output files, cores, memory,
an estimated duration, an exclusive flag, and the settings, tool
identities and seeds that determine the outputs. Dependencies follow from
the files. A job is done when its outputs exist with a matching provenance
record in `runs/<exp>/provenance/<job>.json`: the record's key holds the
command, settings, tool identities (psmc-rs 0.2: version and the last commit
of the Rust sources plus a digest of uncommitted changes; C psmc: version and
SHA-256 of the binary; Python steps: SHA-256 of their modules and package
versions), seeds and the SHA-256 of every input, and the outputs must still
have the checksums recorded at the end of the run. Paths are recorded as
`<repo>`, `<runs>`, `<tools>`, `<python>`, so records stay valid in another
clone. Since inputs count by content, an upstream job that reproduces its
outputs (simulations are deterministic) leaves everything downstream done.

`start` validates the jobs, then detaches the runner (`start_new_session`):
it writes `runs/_runner/runner.pid`, holds `runs/_runner/runner.lock` while
it lives (one runner per runs directory), and logs to
`runs/_runner/runner.log` and `runs/<exp>/logs/` (`runner.log`,
`attempts.jsonl`, and one log per job under `jobs/`). It runs jobs within
the budget (`--cores 7 --mem-gb 12` by default), longest chain of dependent
work first, and backfills small jobs only when they do not delay a waiting
large one; exclusive jobs run only when nothing else runs. `stop` sends
SIGTERM: running jobs are terminated and stay pending; `stop --drain`
starts nothing new and exits when the running jobs finish. A failed job is
recorded and blocks its dependents; the next `start` retries it. If the
runner itself is killed, the next one stops the jobs it left running.
`--keep-awake` (macOS) holds `caffeinate -i` while the runner lives.

Every attempt records its wall time, the peak RSS of the process tree
(psutil, every 50 ms) and the main process's `ru_maxrss`, user and system
CPU time, exit status and the load average before and after; exclusive
jobs also record the busiest other processes before they start.

## E1: simulated accuracy and equivalence

4 models x 10 replicates of 500 Mb (10 chromosomes of 50 Mb). Per replicate:
`sim` (simulation), `fit-c` (C psmc) and `fit-rs` (psmc-rs 0.2, one thread,
also writing `--psmc-out`) on the identical psmcfa; then `report` writes:

| table | contents |
| --- | --- |
| `E1_table1.csv` | Table 1, per model: RMSE of log10 Ne against the truth per tool (mean, SD; informative and fixed windows) and against the pattern-projected truth, the pattern's own error, RMSE between the tools (mean, max), largest relative difference in log-likelihood and theta |
| `E1_replicates.csv` | per model, replicate and tool: theta, rho, t_max, N0, RMSE against the truth, log-likelihoods, wall time, peak RSS, load |
| `E1_rs_vs_c.csv` | per model and replicate: RMSE and largest difference between the tools, relative differences of theta, rho, t_max and the log-likelihood |
| `E1_traces.csv` | per iteration (Fig S1): log-likelihood, theta, rho, t_max for both tools |
| `E1_curves.csv` | every fitted Ne(t) step curve (Fig 2) |
| `E1_truth.csv` | the true Ne(t) of each model on a dense grid (Fig 2) |
| `E1_seeds.csv` | the seeds of every chromosome |

Iterations count completed EM iterations: C psmc's round k prints the
parameters after k iterations and the log-likelihood after k - 1 (`LK`);
psmc-rs's `loglike[i]` is after i. The report checks that both tools read
the simulated genome (bins and heterozygous bins agree with the truth record).

Layout of `experiment/runs/E1/`: `sims/` (genomes; not committed), `truth/`,
`fits/` (`repNN.c.psmc`, `repNN.rs.json`, `repNN.rs.psmc`), `provenance/`,
`tables/` (committed), `logs/` (not committed).

**Production plan** (`plan E1`; costs measured on the development machine
with its load average near 6): 121 jobs, about 10 CPU hours (C psmc 8.3,
psmc-rs 1.0, simulations 0.6), about 1.5 hours of wall time on the default
budget of 7 cores and 12 GB. C psmc holds one chromosome's forward-backward
matrices at a time, about 0.6 GB per job, so memory does not limit E1.

```bash
experiment/.venv/bin/python -m experiment.pipeline start E1 --keep-awake
```

## E4: input preparation

Bottleneck model, 5 replicates of 500 Mb (10 chromosomes of 50 Mb). Each
replicate gets five callable masks: the no-missing control (`nomissing`)
and missing fractions 0.1 and 0.3, either clustered (uncallable runs with
a mean of 20 kb) or dispersed (mean 20 bp). A mask is an alternating renewal
process along each chromosome, callable stretches and uncallable runs of
geometric (discrete exponential) length, with mean callable stretch
`mean_run * (1 - f) / f`; seed stage `mask/<mask>/<chrom>`. The records at
uncallable sites are removed from the simulated VCF, as a caller would not
call them. The masked VCF then goes to PSMC by four routes, all fitted with
psmc-rs 0.2 and checked with `psmc-rs qc`:

| route | input |
| --- | --- |
| `exact` | the VCF with `--vcf-mask callable.bed.gz` (mask mode) |
| `fq2psmcfa` | a consensus FASTA (homozygous: the base; heterozygous: the IUPAC code; uncallable: `N`; other sites: a seeded uniform ACGT base, stage `reference/<chrom>`) streamed into `fq2psmcfa -s 100 -`; the `bins` job fails unless its output equals `psmc-rs convert`'s bins of the exact route, bin for bin |
| `multihetsep` | `psmc-rs convert --multihetsep` of the exact route's input, read back by psmc-rs (uncalled sites at the start of each interval); for replicate 1 every mask's file is compared row for row with `generate_multihetsep.py --mask` run on each chromosome, and the job fails if they differ |
| `nomask` | the same VCF without the mask: uncallable sites count as called homozygous |

The routes are bin-exact, not unbiased: with fq2psmcfa's rule a bin is
missing only when nearly all its bases are uncallable, so a dispersed mask
leaves almost every bin called with only about (1 - f) of its bases, and
theta per bin, N0 = theta / (4 mu s) and every time and size shrink by about
(1 - f), as without the mask. The routes whose input records the callable
sites (`exact` and `multihetsep`, `config.E4.rescaled_routes`) are also
reported rescaled (`exact_rescaled`, `multihetsep_rescaled`): s is the
callable bases per called bin (`called_bases / (bins - missing_bins)` of
their `psmc-rs qc` JSON), and times and sizes are multiplied by 100 / s. The
`fq2psmcfa` and `nomask` routes cannot be rescaled, since their inputs do not
know which sites were callable.

Jobs per replicate and mask: `mask` (BED, masked VCF, record), `consensus`
(the fq2psmcfa route's psmcfa), `convert` (the exact route's bins and the
multihetsep file), `convert/<mask>/{multihetsep,nomask}` (those routes' bins
as psmc-rs reads them), `bins` (every route against the exact route),
`crosscheck` (replicate 1), and a `fit` and a `qc` per route; then `report`
(tables) and `figure` (Fig 5). The report checks that every fit read the
bins of its route's psmcfa.

| table | contents |
| --- | --- |
| `E4_summary.csv` | per mask and route, rescaled routes included (Fig 5): realised missing fraction, share of bins unlike the exact route's, callable bases per called bin and the rescaling factor, theta against the exact route and against the complete data of the same genome, RMSE of log10 Ne against the truth (informative and fixed windows), against the exact route and against the complete-data fit of the same genome, fits identical to the exact route's, QC codes in every / any replicate |
| `E4_routes.csv` | per mask, route (rescaled routes included) and replicate: the fit, its input as read (bins, heterozygous and missing bins, callable bases), metrics, QC codes, wall time and peak RSS |
| `E4_bias.csv` | log10(fitted / true Ne) on the informative-window grid: mean, SD, min, max over replicates |
| `E4_bins.csv` | per mask, replicate and route: the exact route's T/K/N bins against the route's (`N_T`: missing in the exact route, called in the route), bins only one has (the multihetsep tail after its last row) |
| `E4_masks.csv` | per mask and replicate: target and realised missing fraction, uncallable runs and their mean length, heterozygous sites removed |
| `E4_qc.csv` | every QC finding (mask, route, replicate, code, severity) |
| `E4_crosscheck.csv` | per chromosome of replicate 1: rows of generate_multihetsep.py and psmc-rs, identical |
| `E4_curves.csv`, `E4_truth.csv`, `E4_seeds.csv` | every fitted Ne(t) step curve, the true Ne(t), every seed |

Layout of `experiment/runs/E4/`: `sims/repNN/` (the genome, and per mask
`callable.bed.gz`, `variants.vcf.gz`, `fq2psmcfa.psmcfa`,
`input.multihetsep`, `bins/*.psmcfa`; not committed), `truth/`, `masks/`,
`consensus/`, `bins/`, `crosscheck/`, `fits/<mask>/<route>/`,
`qc/<mask>/<route>/`, `provenance/`, `tables/`, `figures/` (committed).

**Production plan** (`plan E4`; step costs measured on one 50 Mb chromosome
with the dispersed 30% mask, the largest, at a load average of 5 to 9): 362
jobs, about 3.7 CPU hours (100 fits 2.4, generate_multihetsep.py 0.5, the rest
under 1), about 35 minutes of wall time on 7 cores, at most 1.5 GB per job
(the consensus FASTA streamed through fq2psmcfa: 1.1 GB measured for 500 Mb).
A full-size run of replicate 1 with the dispersed 30% mask (7.5 million
uncallable runs) took 2.5 minutes on 4 cores: each fit 88 to 97 s and at
most 154 MB, the mask 5 s, the consensus 28 s.

```bash
experiment/.venv/bin/python -m experiment.pipeline start E4 --keep-awake
```

## E5: bootstrap calibration

How well does psmc-rs's bootstrap, which follows C psmc's, measure the
uncertainty of a PSMC curve? The truth: 50 independent genomes of the
bottleneck model, each 300 Mb (6 chromosomes of 50 Mb), simulated as in E1
and fitted once by psmc-rs 0.2: their spread at each time is the true
sampling distribution of log10 Ne(t). The bootstrap: each of the first 10
genomes is bootstrapped by psmc-rs 0.2 with 50 replicates at two trunk
sizes, the default 500,000 bins (50 Mb, splitfa's: one trunk per
chromosome, 6 per genome) and 50,000 bins (5 Mb, 60 per genome), exactly as a
user runs it:

```bash
psmc-rs genome.psmcfa boot.json 25 <the recipe> --threads 1 \
  --bootstrap 50 --bootstrap-block-size B --bootstrap-seed S --bootstrap-dir DIR
```

As in C psmc, every replicate draws trunks with replacement up to the
genome's length and is fitted from a fresh start with the settings of the
fits. Jobs: `sim` (psmcfa only, no VCF), `fit`, `boot-trunk500000`,
`boot-trunk50000`, `report` and `figure`. Seeds: the genomes
`seed("E5", "bottleneck", r, "ancestry/<chrom>")` and `"mutations/<chrom>"`;
each bootstrap run `seed("E5", "bottleneck", r, "bootstrap/trunk<B>")`, so the
two trunk sizes draw independently. All are in `tables/E5_seeds.csv`.

**Outcomes.** Every curve is read at the 200 times of the model's
informative window (10.1 ky to 1.32 My, as E1) and, for robustness, of the
fixed window (10 ky to 1 My). At each time:

* the true sampling SD is the SD of log10 Ne over the 50 fits, with a
  chi-square 95% interval (normal theory: 0.84 to 1.25 times the SD for 50
  fits);
* a bootstrap's SD is the SD of log10 Ne over its replicates, and its 95% band
  the 2.5% and 97.5% quantiles of the replicates' Ne, interpolated linearly
  as psmc-rs's own summary does;
* the SD ratio of a genome is its bootstrap SD over the true sampling SD; the
  tables give its median over the 10 genomes (the headline), quartiles and
  range;
* the coverage is the share of the 10 genomes whose band holds the true
  Ne(t), with a Clopper-Pearson 95% interval; next to it, the share whose
  band holds the geometric mean of the 50 fits, the estimator's expectation.
  A bootstrap sees the spread of the estimate, not its bias: the gap between
  the two coverages is PSMC's bias, the shortfall of the second from 95% is
  the bootstrap's.

The report checks that every fit read its simulated genome, that every
bootstrap ran with its seed, trunk size and replicate count (a replicate whose
fresh start is invalid is skipped by psmc-rs and listed in `summary.json`; the
report reads exactly the replicates the summary names), and that the band
recomputed from the replicate files equals psmc-rs's `summary.json`. It
records whether the main fit of every bootstrap run equals the genome's own
fit (`E5_datasets.csv`, `main_fit_equals_fit`; psmc-rs's results do not depend
on the run).

| table | contents |
| --- | --- |
| `E5_summary.csv` | per trunk size and window: median, range and share below 1 of the median SD ratio, with a 95% interval from resampling the datasets; median IQR ratio; mean and least pointwise coverage of the truth and of the fits' mean, the mean's 95% interval from resampling the datasets (the independent units: the points of one dataset are strongly correlated) and the spread of the datasets' shares of covered points; median band width and true 95% range (log10); median true SD, bias and RMSE |
| `E5_sd_ratio.csv` | per trunk size and time: the true SD and its interval, the bootstrap SD, the SD ratio (median, quartiles, range over genomes), the IQR ratio |
| `E5_coverage.csv` | per trunk size and time: genomes covered, coverage and its interval, coverage of the fits' mean, band width |
| `E5_sampling.csv` | per time: true Ne, and the 50 fits' mean log10 Ne, median and 95% range of Ne, SD and its interval, IQR, bias and RMSE |
| `E5_bands.csv` | per bootstrapped genome, trunk size and time: its fit, its band and median, its bootstrap SD, whether the band covers the truth |
| `E5_datasets.csv` | per bootstrap run: trunks, seed, replicates fitted and skipped, median SD ratio, coverage, the checks, wall time, CPU time and peak RSS |
| `E5_fits.csv` | per genome: heterozygous sites and bins, theta, rho, t_max, N0, final log-likelihood, RMSE against the truth, wall time and peak RSS |
| `E5_seeds.csv` | every seed |

**Figure 6** (`figures/E5_fig6.pdf` and `.png`, by `figures/e5.py` from the
tables and the truth record; 180 x 128 mm, 7 pt Arial, fonts embedded,
byte-for-byte reproducible): A and B, genome 1's fit and bootstrap band at each
trunk size over the true Ne and the 95% range of the 50 fits; C, the SD ratio
by time (median and IQR over the 10 genomes; grey: where a bootstrap with the
true SD would fall given the true SD's own sampling error); D, the pointwise
coverage of the true Ne (thick) and of the fits' mean (thin) against the
nominal 95%. Redraw it without re-running anything with
`$PY -m experiment.pipeline.figures.e5 [--profile P]`.

Layout of `experiment/runs/E5/`: `sims/` (genomes; not committed),
`truth/`, `fits/bottleneck/repNN.rs.json`, `boot/trunk<B>/repNN.rs.json` (the
bootstrap run's main fit), `boot/trunk<B>/repNN/` (`replicate_NNN.json`,
`summary.json`, `summary.tsv`; the replicates' `.psmc` files, 0.1 MB each, are
not committed), `provenance/`, `tables/`, `figures/`, `logs/` (not committed).

**Production plan** (`plan E5`): 122 jobs, about 16 CPU hours, about 2.3
hours of wall time alone on the default budget of 7 cores and 12 GB. Measured
on the development machine at a load average of 4 to 8: a 300 Mb genome
simulates in 7 s (0.19 GB) and one fit of it takes 49 s and 26 MB, so a
bootstrap run (its main fit and 50 replicates, one thread) takes about 42
minutes; the cost model, at 0.007 s per Mb and iteration, expects 45. The 20
bootstrap runs dominate (15 CPU hours) and run one thread each, 7 at a time,
in three waves; memory does not limit E5. The jobs are independent of the
other experiments, so E5 can run alongside them on the shared budget.

```bash
experiment/.venv/bin/python -m experiment.pipeline start E5 --keep-awake
```

`--profile smoke` runs E5 with 8 genomes of 2 x 2 Mb, the first 3
bootstrapped with 8 replicates at trunks of 5,000 and 2,000 bins (8 and 20
trunks), 3 EM iterations, in about 10 seconds.

## Figures

`FIG` draws each figure as a runner job after the tables it reads, into
`runs/<experiment>/figures/<name>.{pdf,svg,png}`; `python -m
experiment.pipeline.figures` draws them directly. The style, the tables
each figure reads and the production run are in
[figures/README.md](figures/README.md).

## Tools

The tools directory is `$PSMC_EXP_TOOLS`, else `experiment/.tools` (of the
main checkout when run from a worktree without one): `psmc_c/psmc` (lh3/psmc
0.6.5-r74 built with `-O1`, since `-O2` segfaults on macOS arm64),
`psmc_c/utils/{fq2psmcfa,splitfa}`, `psmc_c/psmc_plot.pl`, `vcfutils.pl`,
`msmc-tools/generate_multihetsep.py`. `tools --build-rs01` builds psmc-rs 0.1
from commit e50d69d in a detached git worktree at
`<tools>/psmc-rs-0.1/src` with `CARGO_TARGET_DIR=<tools>/psmc-rs-0.1/target`
(offline: the dependencies must be in cargo's cache). psmc-rs 0.2 is this
checkout's `target/release/psmc-rs`, built by `start`.

## Tests

```bash
experiment/.venv/bin/python -m unittest discover -s experiment/tests
```

`test_pipeline_*.py` cover every module; parts that need msprime, stdpopsim,
psutil, htslib, C psmc or a psmc-rs build skip without them. With all of
them present, `test_pipeline_e1.EndToEnd` runs the tiny profile through the
runner and checks that the first iteration's log-likelihood of psmc-rs
equals C psmc's to its printed precision; `test_pipeline_e5.EndToEnd` runs
E5's tiny profile (simulations, fits, bootstraps, tables and figure) and
checks that nothing is left to do afterwards.
