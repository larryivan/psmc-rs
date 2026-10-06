# Experiment pipeline

`experiment/pipeline/` produces the manuscript's experiments and figures: it
simulates genomes, runs C psmc and psmc-rs with identical settings, records
how every output was made, writes the tables and draws the figures from
them. Every experiment of the design is implemented: E1 (simulated accuracy
and equivalence, Fig 2, Table 1, Fig S1), E2 (real genomes, Fig 3), E3
(performance, Fig 4, Table 2, Fig S4), E4 (input preparation, Fig 5), E5
(bootstrap calibration, Fig 6), E6 (supplementary analyses, Figs S2 and S3,
Table S5) and FIG (every figure, Fig 1 included). `experiment/scripts/`
holds the earlier pipeline of psmc-rs 0.1's runs.

Run it from the repository root with the experiment's environment
(`experiment/.venv` with the exact versions of
`experiment/pipeline/requirements.lock`; `tools` and `start` warn when they
differ):

```bash
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools          # what will run: tool paths, versions, identities
$PY -m experiment.pipeline plan           # CPU hours and wall time on the budget
$PY -m experiment.pipeline start E1       # detached runner; returns at once
$PY -m experiment.pipeline status         # done/running/failed/pending, ETA
$PY -m experiment.pipeline stop           # or: stop --drain (finish running jobs first)
$PY -m experiment.pipeline start E1       # resume: only what is not done runs
```

`--profile smoke` runs the same jobs at a small size in
`experiment/.runs-smoke/` (3 EM iterations; the 164 jobs of every
experiment and figure took under a minute on the development machine,
while `plan` expects about 5); `--profile tiny` is for the unit tests. The production run
is described at the end ([Production](#production)).

## Modules

| module | contents |
| --- | --- |
| `config.py` | every parameter: inference settings, scaling, simulation, demographic models, seed scheme, experiments E1-E6, figures, resource budget and cost model, profiles |
| `paths.py` | repository, tools directory (`PSMC_EXP_TOOLS`), runs directory (`PSMC_EXP_RUNS`) |
| `demography.py` | models as `msprime.Demography`; true Ne(t) and pairwise coalescence-time distribution in closed form; checks against msprime's `DemographyDebugger` and stdpopsim |
| `simulate.py` | one diploid genome: psmcfa, bgzipped VCF, truth record; true TMRCA track on demand |
| `tools.py` | psmc-rs 0.2 (this checkout), psmc-rs 0.1 (built from e50d69d), C psmc and utilities: locate, build, identify; exact command lines; the requirements lock |
| `results.py` | C `.psmc` (every round: LK, QD, RI, TR, MT, RS, PA) and psmc-rs JSON as PSMC models and Ne(t) curves |
| `metrics.py` | time grids, RMSE of log10 Ne against the truth, against the pattern-projected truth and between tools |
| `provenance.py` | checksums and provenance records: when a job is done |
| `runner.py` | the job graph, scheduler, detached daemon, machine lock, status |
| `inputprep.py` | E4's data steps: callable masks, masked VCF, consensus FASTA through fq2psmcfa, bins of the routes compared, generate_multihetsep.py cross-check (`python -m experiment.pipeline.inputprep STEP`) |
| `experiments/e1.py` ... `e6.py` | each experiment as jobs, and its tables (`e3_run.py`: E3's timing wrapper and genome simulation; `e6_analysis.py`: E6's analyses) |
| `experiments/fig.py` | the figures as runner jobs (experiment `FIG`) |
| `figures/` | the manuscript's figures: style, table schemas, one module per figure ([README](figures/README.md)) |

## Design

**Inference.** Every fit, by either tool, uses the psmc README's human
recipe: `psmc -N25 -t15 -r5 -p "4+25*2+4+6"`, which psmc-rs gets as
`N_ITER 25 --pattern 4+25*2+4+6 --t-max 15 --tr-ratio 5 --smooth-lambda 0
--mstep-iters 1000 --bin-size 100`, plus `--threads` per job. Simulations
are scaled with mu = 2.5e-8 and g = 25.

**M-step.** C psmc's Hooke-Jeeves search runs every M-step to
convergence, and so does psmc-rs 0.2's L-BFGS: it stops by a convergence
test, in 10 to 70 iterations and a few ms per EM iteration on one thread,
and `--mstep-iters` (default 1000, passed explicitly) is only a cap. On a
simulated 4 x 25 Mb bottleneck genome the RMSE of log10 Ne between psmc-rs
and C psmc on the informative window is 7.7e-5, as small as with C psmc's
own Hooke-Jeeves search run inside psmc-rs's EM (7.5e-5). psmc-rs 0.1's
L-BFGS stopped after 100 iterations, which left the two 0.03 to 0.16 apart;
that is why the experiments pass the cap explicitly. Since psmc-rs's
default M-step is the converged one, no experiment fits or times a separate
default arm: `E1.default_mstep_iters` and `E3.default_mstep_iters` are
unset, and setting one of them to a cap adds a fit (`fit-rs-default`,
`E1_mstep.csv`) or timed runs (`rs02-default`) with M-steps capped at that
many iterations (the tiny profile does, at 5, to test them).
`test_pipeline_fit_settings.py` checks that every psmc-rs fit of the
production and smoke profiles passes psmc-rs's own cap
(`DEFAULT_MSTEP_ITERS` in `src/opt.rs`) and that neither fits a capped arm.

**Callable bases.** psmc-rs weighs each called bin by its callable bases
when its input counts them (`--bin-callable auto`, its default; see E4).
E1, E2, E3 and E5 fit psmcfa files, which record no callable bases, so
psmc-rs fits C psmc's model there, every called bin taken as 100 bases,
whatever `--bin-callable` says: the fits are what they were before the
model, and their JSON only gains the record `bin_callable: {mode: auto,
model: full-bins}`. E6's S2 reads complete
data as VCF and multihetsep, whose bins are all full, so the callable-length
model is C psmc's there too. Only E4 fits partly callable bins, and passes
`--bin-callable` explicitly for every route. `test_pipeline_fit_settings.py`
checks both: the psmcfa experiments fit psmcfa input only, and the tiny
runs' fits record what is said here.

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
modulo 2^31 - 1, plus 1. Stages are `ancestry/<chrom>` and
`mutations/<chrom>` of every simulation, E4's `mask/<mask>/<chrom>` and
`reference/<chrom>`, the bootstraps' `bootstrap` (E2's default trunks) and
`bootstrap/trunk<B>`, and E5's `summary/<window>/trunk<B>` (resampling the
datasets). A seed depends on nothing but its key, and no seed was chosen
after looking at a result. Every seed is in the truth record, the job's
provenance record and the experiment's `*_seeds.csv` table.

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
are C psmc's step functions, scaled as `psmc_plot.pl` does. The first free
lambda of `4+25*2+4+6` spans intervals 0-3 (about 0-28 ky for zigzag,
whose window starts at 7 ky), so part of the RMSE against the truth is the
error of approximating the truth by the pattern's steps. Each fit is therefore also compared with the truth
projected onto its own free-parameter spans (the true Ne averaged over each
span in coalescence-rate terms, `metrics.projected_truth`): that RMSE is the
estimation error, and the projection's RMSE against the truth the pattern's.

## Runner

Experiments declare jobs: a command, input and output files, cores, memory,
an estimated duration, an exclusive flag, and the settings, tool
identities and seeds that determine the outputs. Dependencies follow from
the files. A job is done when its outputs exist with a matching provenance
record in `runs/<exp>/provenance/<job>.json`: the record's key holds the
command, settings, tool identities (psmc-rs 0.2: version, the git tree and
blob ids of `src/`, `Cargo.toml` and `Cargo.lock`, a digest of uncommitted
changes, `rustc -vV` and `RUSTFLAGS`; C psmc: version and SHA-256 of the
binary; Python steps: SHA-256 of their modules and package versions), seeds
and the SHA-256 of every input, and the outputs must still have the
checksums recorded at the end of the run. Paths are recorded as `<repo>`,
`<runs>`, `<tools>`, `<python>`, so records stay valid in another clone,
and a rebase or squash that keeps the sources keeps them valid too. Since
inputs count by content, an upstream job that reproduces its outputs
(simulations are deterministic) leaves everything downstream done.

`start` validates the jobs, then detaches the runner (`start_new_session`):
it writes `runs/_runner/runner.pid`, holds `runs/_runner/runner.lock` while
it lives (one runner per runs directory), and logs to
`runs/_runner/runner.log` and `runs/<exp>/logs/` (`runner.log`,
`attempts.jsonl`, and one log per job under `jobs/`). It runs jobs within
the budget (`--cores 7 --mem-gb 12` by default), longest chain of dependent
work first, and backfills small jobs only when they do not delay a waiting
large one; exclusive jobs run only when nothing else runs, in order of
their rank (E3 ranks its timing runs by repeat). Runners of different runs
directories or worktrees coordinate through a machine-wide lock
(`$PSMC_EXP_MACHINE_LOCK`, default `~/.cache/psmc-exp/machine.lock`): held
shared while a runner runs ordinary jobs and exclusively while it runs an
exclusive one. `stop` sends SIGTERM: running jobs are terminated and stay
pending; `stop --drain` starts nothing new and exits when the running jobs
finish. A failed job is recorded and blocks its dependents; the next
`start` retries it. A job that exits with status 75 (E3's timing wrapper on
a busy machine) is deferred and starts again a minute later. If the runner
itself is killed, the next one stops the jobs it left running. `start`
keeps the machine awake (`caffeinate -i` on macOS; `--no-keep-awake` to opt
out).

The runner takes the SHA-256 of the pipeline's Python files and of every
executable its jobs name when it starts, and starts no further job once one
of them changes (a rebuild of psmc-rs, an edit, a branch checkout): the
running jobs finish and the runner exits, saying which file changed. Run
the production in a checkout nobody develops in (a dedicated worktree).

Every attempt records its wall time, the peak RSS of the process tree
(psutil, every 50 ms, except for exclusive jobs, which their own wrapper
measures) and the main process's `ru_maxrss`, user and system CPU time,
exit status, the load average before and after, and the SHA-256 of the
executables it ran; exclusive jobs also record the busiest other processes
before they start. `status` without experiments reports those of the
running runner, or of the last one; a job whose input no job makes is
reported blocked.

## E1: simulated accuracy and equivalence

4 models x 10 replicates of 500 Mb (10 chromosomes of 50 Mb). Per replicate:
`sim` (simulation), `fit-c` (C psmc) and `fit-rs` (psmc-rs 0.2, one thread,
also writing `--psmc-out`) on the identical psmcfa (and `fit-rs-default`,
psmc-rs with capped M-steps, when `E1.default_mstep_iters` sets a cap); then
`report` writes:

| table | contents |
| --- | --- |
| `E1_table1.csv` | Table 1, per model: RMSE of log10 Ne against the truth per tool (mean, SD; informative and fixed windows) and against the pattern-projected truth, the pattern's own error, RMSE between the tools (mean, max), largest relative difference in log-likelihood and theta |
| `E1_replicates.csv` | per model, replicate and tool: theta, rho, t_max, N0, RMSE against the truth, log-likelihoods, wall time, peak RSS, load |
| `E1_rs_vs_c.csv` | per model and replicate: RMSE and largest difference between the tools, relative differences of theta, rho, t_max and the log-likelihood |
| `E1_mstep.csv` | per model and replicate (no rows without a capped fit): psmc-rs with capped M-steps against C psmc, against the main fit and against the truth, wall times |
| `E1_traces.csv` | per iteration (Fig S1): log-likelihood, theta, rho, t_max for both tools |
| `E1_curves.csv` | every fitted Ne(t) step curve (Fig 2) |
| `E1_truth.csv` | the true Ne(t) of each model on a dense grid (Fig 2) |
| `E1_seeds.csv` | the seeds of every chromosome, its heterozygous sites, trees and longest stretch without a heterozygous site |

Iterations count completed EM iterations: C psmc's round k prints the
parameters after k iterations and the log-likelihood after k - 1 (`LK`);
psmc-rs's `loglike[i]` is after i. The report checks that both tools read
the simulated genome (bins and heterozygous bins agree with the truth record).

Layout of `experiment/runs/E1/`: `sims/` (genomes; not committed), `truth/`,
`fits/<model>/` (`repNN.c.psmc`, `repNN.rs.json`, `repNN.rs.psmc`,
`repNN.rs-default.json` with a capped fit), `provenance/`, `tables/`, `figures/` (committed),
`logs/` (not committed).

**Cost** (`plan E1`): 121 jobs, about 10 CPU hours (C psmc 8.3, psmc-rs 1.0,
simulations 0.6), about 1.4 hours alone on the default budget. C psmc holds
one chromosome's forward-backward matrices at a time, about 0.6 GB per job.

## E2: real genomes

The four genomes of `experiment/real_data` (`config.REAL_GENOMES`): the
turtles HLemySub1 (*Emydura subglobosa*), HLhydTec1 (*Hydromedusa
tectifera*) and HLpelCas1 (*Pelusios castaneus*), scaled with mu 7.9e-9 and
g 15 (the values of the psmc-rs 0.1 runs; the repository records no
citation for them, and the config marks them unverified), and chromosome 1
of a Papuan Highlands human (*Homo sapiens*, chr1 only: every table and
figure says so), scaled with mu 1.25e-8 and g 25. Per genome: `fit-c` (C
psmc), `fit-rs` (psmc-rs 0.2, the identical file and settings),
`boot-rs/trunk500000` (psmc-rs 0.2 with `--bootstrap 100` on the default
trunks of 500,000 bins, splitfa's 50 Mb; seed `seed("E2", sample, 0,
"bootstrap")`; the bootstrap run's main fit must equal `fit-rs`'s JSON byte
for byte) and `qc` (`psmc-rs qc --json`). A genome whose default trunks are
fewer than 20 (Papuan chr1: 5) is bootstrapped a second time on 5 Mb trunks
(`boot-rs/trunk50000`, seed stage `bootstrap/trunk50000`), the size E5
also calibrates. E5's calibration applies to chromosome-scale sequences cut
into trunks; the turtles' default bootstraps resample whole scaffolds of
very different lengths (889, 34 and 1,220 sequences; 50%, 23% and 97% of
their bins lie in scaffolds shorter than a trunk), a regime E5 does not
calibrate.

The tools are compared on the window between the 1% and 99% quantiles of
the pairwise coalescence time under C psmc's fitted model (a real genome
has no truth); the band is the 2.5% to 97.5% quantiles of the replicates'
Ne at each time, as psmc-rs's summary computes it, and the report checks it
against `summary.json`.

| table | contents |
| --- | --- |
| `E2_fit_summary.csv` | per genome and tool: scaling and its source, sequences, called and heterozygous bins, theta, rho, t_max, N0, log-likelihoods, wall time and peak RSS |
| `E2_rs_vs_c.csv` | per genome: RMSE of log10 Ne between the tools (window and fixed window), largest difference, relative differences of theta, rho, t_max and the log-likelihoods, whether the bootstrap runs' main fits equal `fit-rs`'s |
| `E2_bootstrap.csv` | per genome and trunk size: replicates, trunks (`coarse_band` under 20), share of bins in sequences shorter than a trunk, seed, band width, how much of each tool's curve the band holds, the check against psmc-rs's summary |
| `E2_bootstrap_replicates.csv` | every replicate's parameters |
| `E2_bands.csv` | per genome and trunk size: both curves and the band on a dense grid (Fig 3) |
| `E2_curves.csv`, `E2_traces.csv` | the fitted step curves (Fig 3); log-likelihood and parameters per EM iteration |
| `E2_qc.csv`, `E2_qc_findings.csv` | `psmc-rs qc`'s metrics and findings for each input fitted (in production the packaged files; Table S5 is E6's `E6_S5_*`) |

Layout of `experiment/runs/E2/`: `fits/` (`<sample>.c.psmc`, `.rs.json`,
`.rs.psmc`), `bootstrap/<sample>/trunk<B>/` (`main.json`,
`replicate_NNN.json`, `summary.json`, `summary.tsv`; the replicates' `.psmc`
files are not committed), `qc/`, `provenance/`, `tables/`, `figures/`;
`inputs/` (smoke and tiny profiles: the first bins of each genome) and
`logs/` are not committed.

**Cost** (`plan E2`): 18 jobs, about 34 CPU hours (the five bootstrap runs
31, each on 7 threads, up to about 1.4 hours for a turtle; C psmc 2.5, up to
49 minutes and 3.4 GB for HLhydTec1), about 5.3 hours alone.

## E3: performance

Run time and peak memory of C psmc, psmc-rs 0.1 and psmc-rs 0.2 on
constant-model genomes of 100, 300, 1000 and 3000 Mb (1, 3, 10 and 30
chromosomes of 100 Mb; one simulation per size, the chromosomes shared
between sizes), 25 EM iterations of the recipe, 3 repeats each: C psmc on
its one thread; psmc-rs 0.1 (commit e50d69d, built out of the tree; it has
no `--tr-ratio`, and its default start rho0 = theta0/5 is the recipe's
`-r5`) and psmc-rs 0.2 on 1 and 8 threads (and psmc-rs 0.2 with capped
M-steps, `rs02-default`, when `E3.default_mstep_iters` sets a cap); and psmc-rs 0.2 on the
3000 Mb genome with 1, 2, 4 and 8 threads (thread scaling). A setup whose
estimated peak memory exceeds 12 GB is not run and Table 2 says why (none
with 100 Mb chromosomes: C psmc holds about 1 GB).

Every run is an exclusive job (nothing else runs, on this runner or any
other), ranked by repeat, and timed by `e3_run measure` (wall time, CPU
time, peak RSS of the tool's process tree, sampled every 0.5 s with the
tool's own `ru_maxrss`, the machine's load and the busiest processes before
the run, the CPU other processes used during it, the power source). The
tool runs without `PSMC_*` and Rust build variables in its environment. A
run is kept only when it is clean: the machine became quiet (at most 10%
of all cores busy) within 3 minutes, other processes used at most 5% of
the cores during the run, the machine did not sleep, and it ran on AC power
without Low Power Mode. Otherwise `measure` writes nothing and exits with
status 75, and the runner starts the run again a minute later; the report
takes medians from clean runs only.

| table | contents |
| --- | --- |
| `E3_table2.csv` | Table 2, per tool, thread count and size: status (measured, skipped with the reason, incomplete), estimated peak memory, clean and measured repeats, median, minimum and maximum wall time, median CPU time, seconds per iteration, peak RSS, speed-up over C psmc and over psmc-rs 0.1 at the same threads, C psmc's RSS over the tool's, the M-step budget, whether the repeats' outputs are identical |
| `E3_runs.csv` | every run: measurements, machine state, power, clean or not, theta, rho, t_max, log-likelihood, output SHA-256 |
| `E3_thread_scaling.csv` | Fig S4: psmc-rs 0.2 on 3000 Mb per thread count: median wall time, speed-up, efficiency, peak RSS, whether the JSON is byte-identical to the one-thread run |
| `E3_determinism.csv` | Fig S4: every output against its tool's reference at the same size (fewest threads, first repeat) |
| `E3_environment.csv` | machine, tools (identities, binary SHA-256, rustc), C compiler, protocol, power sources, runs not clean |
| `E3_seeds.csv` | every seed, with each chromosome's trees and longest stretch without a heterozygous site |

Layout of `experiment/runs/E3/`: `sims/` (not committed), `truth/`,
`fits/<tool>/<size>Mb/t<threads>/rep<r>.{psmc,json}`, `timings/...` (the
timing records), `provenance/`, `tables/`, `figures/`.

**Cost** (`plan E3`): 71 jobs; the 66 timing runs take about 11 hours of
wall time one after another (C psmc about 1 hour each on 3000 Mb; psmc-rs
0.1's 1000 forward-mode L-BFGS iterations per M-step add about 200 s per
run, 25 M-steps on one thread, whatever the size, where psmc-rs 0.2's
converged M-steps take under a second). They need the machine to themselves: close
other work, stay on AC power, and expect the runner to defer runs while the
machine is busy (`status --jobs` says why).

## E4: input preparation

**The question.** PSMC sees a genome as bins of s = 100 bases. fq2psmcfa
calls a bin missing (`N`) only when more than 90% of its bases are
uncallable, and C psmc takes every called bin as s bases. With missing data
dispersed in short runs almost every bin is still called but holds only
about (1 - f) of its bases, so theta per bin, N0 = theta / (4 mu s) and
every time and size come out low by about the missing fraction f. psmc-rs
counts the callable bases c of each bin from the VCF and its callable mask
and, by default (`--bin-callable auto`), fits each bin by them: P(T | k) =
exp(-theta (c / s) avg_t_k), the callable-length model of
`docs/design/bin-callable-model.md`, which is C psmc's model to the bit
when every bin is full. E4 measures the bias that partly callable bins
cause, whether the model removes it, and how the two other ways of losing
the callable sites compare: multihetsep, which records how many sites of
each interval are callable but not which, and a VCF without its mask.

**Data.** Bottleneck model, 5 replicates of 500 Mb (10 chromosomes of
50 Mb). Each replicate gets five callable masks: the no-missing control
(`nomissing`) and missing fractions 0.1 and 0.3, either clustered
(uncallable runs with a mean of 20 kb) or dispersed (mean 20 bp). A mask is
an alternating renewal process along each chromosome, callable stretches
and uncallable runs of geometric (discrete exponential) length, with mean
callable stretch `mean_run * (1 - f) / f`; seed stage `mask/<mask>/<chrom>`.
The records at uncallable sites are removed from the simulated VCF, as a
caller would not call them.

**Routes.** The masked VCF goes to PSMC by five routes, all fitted with
psmc-rs 0.2 with the inference recipe and their `--bin-callable` passed
explicitly (`config.E4_BIN_CALLABLE`). The exact bins are psmc-rs's bins of
the VCF with its mask (`psmc-rs convert`), which both mask routes fit:

| route | input | `--bin-callable` | model fitted |
| --- | --- | --- | --- |
| `mask_auto` | the VCF with `--vcf-mask callable.bed.gz` (mask mode) | `auto` | callable-length, with the exact callable bases of each bin |
| `mask_ignore` | the same | `ignore` | every called bin full: fq2psmcfa's and C psmc's |
| `fq2psmcfa` | a consensus FASTA (homozygous: the base; heterozygous: the IUPAC code; uncallable: `N`; other sites: a seeded uniform ACGT base, stage `reference/<chrom>`) streamed into `fq2psmcfa -s 100 -`; the `bins` job fails unless its bins equal the exact bins, and the report fails unless its fit equals `mask_ignore`'s, every value of the JSON but those that describe the input (`input`, `qc`, `bin_callable`) | `auto` (a psmcfa records no callable bases) | every called bin full |
| `multihetsep` | `psmc-rs convert --multihetsep` of the mask routes' input, read back by psmc-rs; for replicate 1 every mask's file is compared row for row with `generate_multihetsep.py --mask` run on each chromosome, and the job fails if they differ | `auto` | every called bin full: multihetsep says how many sites between two rows are callable, not which, and psmc-rs places the uncalled ones at the start of each interval, as if missing data came in one run per interval; the callable bases of its bins follow that placement (counts `approximate`), and `auto` fits exact counts only, so this route measures the placement alone (`experiment/validation/` also fits its counts, with `--bin-callable approximate`, which lowers theta further) |
| `nomask` | the same VCF without the mask (variants only): uncallable sites count as called homozygous | `auto` | every bin full |

On the control every bin is full, so the callable-length model is C
psmc's: the report fails unless `mask_auto` and `mask_ignore` fit
identically there, and that fit is the complete-data fit with which every
route of the replicate is compared. The tables report, per route: theta
against the complete-data fit, the RMSE of log10 Ne against the truth
(informative and fixed windows) and against the complete-data fit, the
mean log10 bias and the bias at every time of the informative window, the
callable bases of the called bins (mean share, and the share of bins
partly callable), the bins unlike the exact bins, and the QC codes. Those
are the codes of each fit's own QC (its JSON's `qc` record), which psmc-rs
computes with the fit's settings, `--bin-callable` included:
`PARTLY_CALLABLE_BINS` is a warning when a fit takes as full called bins
that are on average less than 95% callable (a note above), a note when the
fit weighs them by their callable bases, and for psmcfa input, which cannot
tell, a note counting the called bins next to missing runs; `APPROX_MISSING_PLACEMENT`
marks multihetsep, as a warning when more than 5% of its sites are
uncalled, and `VARIANTS_ONLY_NO_MASK` the VCF without a mask.

What to expect, from the simulations of `experiment/validation/` (3
replicates of 200 Mb, masks of the same kinds, the same recipe): dispersed
missing data bias the full-bins routes by about the missing share (theta
at 0.88 and 0.65 of the complete-data fit with 10% and 30% missing, RMSE
of log10 Ne against the truth 0.348 and 0.453 against 0.307 with complete
data), and the callable-length model removes the bias (theta within 0.7%
and 1.8%, RMSE 0.306 and 0.309). Clustered missing data empty whole bins, which both
models leave out, and change little either way. multihetsep does not
recover the exact route when missing data are dispersed (RMSE 0.421 at
30%): placing every uncalled site of an interval at its start turns
scattered missing bases into runs of `N` bins and leaves the bin of each
heterozygous site less callable than it is, so the VCF with its mask is the
input to use. E4 measures all of it on 500 Mb genomes, with the control,
the VCF without a mask and the fq2psmcfa route.

**Rescaling cross-check** (`E4_rescaling.csv`; not a route, not in Fig 5).
Before the callable-length model, E4 corrected the bias after the fit: the
full-bins fit read with s = the mean callable bases of its called bins
instead of 100, that is times and sizes multiplied by 100 / s (an earlier
run, at 30% dispersed missing data: RMSE against the truth 0.447 as
fitted, 0.334 rescaled, 0.332 with complete data). The table keeps that
global correction of `mask_ignore` (s from its JSON's
`bin_callable.callable.mean_fraction`) beside `mask_auto`, per mask and
replicate, as a check of the model: the two should agree on the scale, and
the per-bin model, which also weighs each bin by how much of it was seen,
should be at least as close to the complete-data fit.

Jobs per replicate and mask: `mask` (BED, masked VCF, record), `consensus`
(the fq2psmcfa route's psmcfa), `convert` (the exact bins and the
multihetsep file), `convert/<mask>/{multihetsep,nomask}` (those routes'
bins as psmc-rs reads them), `bins` (those routes against the exact bins),
`crosscheck` (replicate 1) and a `fit` per route; then `report` (tables;
Fig 5 is drawn from them by `FIG/fig5_input_preparation`). The report
checks that every fit read the bins of its route and ran with its route's
`--bin-callable`, and that the fits that must be identical are.

| table | contents |
| --- | --- |
| `E4_summary.csv` | per mask and route (Fig 5): `--bin-callable`, the model fitted and where its callable bases came from (`exact`, `approximate`, none), realised missing fraction, mean share of callable bases of the called bins and share of them partly callable, share of missing bins and of bins unlike the exact bins, theta against the complete-data fit, RMSE of log10 Ne against the truth (informative and fixed windows) and against the complete-data fit of the same genome, mean log10 bias, fits identical to `mask_auto`'s and to `mask_ignore`'s, QC codes in every / any replicate |
| `E4_routes.csv` | per mask, route and replicate: the fit, the model, its input as read (bins, heterozygous and missing bins, called bases, callable bases of the called bins), metrics, QC codes, wall time and peak RSS |
| `E4_bias.csv` | log10(fitted / true Ne) on the informative-window grid: mean, SD, min, max over replicates |
| `E4_rescaling.csv` | per mask and replicate: the rescaling cross-check, `mask_auto`, `mask_ignore` and `mask_ignore` rescaled (theta against the complete-data fit, RMSE against the truth and against the complete-data fit, mean log10 bias), and the RMSE between the rescaled curve and `mask_auto`'s |
| `E4_bins.csv` | per mask, replicate and route: the exact T/K/N bins against the route's (`N_T`: missing in the exact bins, called in the route), bins only one has (the multihetsep tail after its last row) |
| `E4_masks.csv` | per mask and replicate: target and realised missing fraction, uncallable runs and their mean length, heterozygous sites removed |
| `E4_qc.csv` | every QC finding of every fit (mask, route, replicate, code, severity: `warning` or `note`) |
| `E4_crosscheck.csv` | per chromosome of replicate 1: rows of generate_multihetsep.py and psmc-rs, identical |
| `E4_curves.csv`, `E4_truth.csv`, `E4_seeds.csv` | every fitted Ne(t) step curve, the true Ne(t), every seed |

Layout of `experiment/runs/E4/`: `sims/repNN/` (the genome, and per mask
`callable.bed.gz`, `variants.vcf.gz`, `fq2psmcfa.psmcfa`,
`input.multihetsep`, `bins/*.psmcfa`; not committed), `truth/`, `masks/`,
`consensus/`, `bins/`, `crosscheck/`, `fits/<mask>/<route>/`,
`provenance/`, `tables/`, `figures/` (committed).

**Cost** (`plan E4`): 286 jobs, about 4 CPU hours (the 125 fits 3.1; the
generate_multihetsep.py cross-checks 0.5), about 0.6 hours alone, at most
1.5 GB per job (the consensus FASTA streamed through fq2psmcfa).

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
`boot-trunk50000` and `report`. Seeds: the genomes
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

**Figure 6** (`figures/fig6_bootstrap_calibration.{pdf,svg,png}`, drawn by
`FIG` from the tables, see `figures/README.md`): a, genome 1's fit and its
bootstrap bands at both trunk sizes over the true Ne and the 95% range of
the 50 fits; b, the true sampling SD and the bootstrap SDs; c, the SD ratio
by time (median and IQR over the 10 genomes; grey: where a bootstrap with
the true SD would fall given the true SD's own sampling error); d, the
pointwise coverage of the true Ne (thick, with its Clopper-Pearson
interval) and of the fits' mean (thin) against the nominal 95%.

Layout of `experiment/runs/E5/`: `sims/` (genomes; not committed),
`truth/`, `fits/bottleneck/repNN.rs.json`, `boot/trunk<B>/repNN.rs.json` (the
bootstrap run's main fit), `boot/trunk<B>/repNN/` (`replicate_NNN.json`,
`summary.json`, `summary.tsv`; the replicates' `.psmc` files, 0.1 MB each, are
not committed), `provenance/`, `tables/`, `figures/`, `logs/` (not committed).

**Cost** (`plan E5`): 121 jobs, about 16 CPU hours (the 20 bootstrap runs,
51 fits each on one thread, about 45 minutes each), about 2.3 hours alone;
at most 0.2 GB per job.

## E6: supplementary analyses

E6 reads what E1 and E2 make and adds the runs only it needs
(`config.E6`; models, inference, scaling and grid of S1-S3 are E1's):

* **S1, EM convergence** (Fig S1, drawn from `E1_traces`): per model and
  tool, how many iterations bring the log-likelihood within 1 unit of the
  last iteration both tools evaluate, whether it ever decreases, and how
  closely the tools agree at each iteration (`E6_S1_traces`, `_summary`,
  `_agreement`).
* **S2, format consistency** (Fig S2): replicate 1 of every E1 model read
  as psmcfa (E1's fit), as its VCF (no missing data) and as multihetsep
  (`psmc-rs convert --multihetsep` of the VCF): the VCF must give the
  psmcfa's bins and fit exactly, the multihetsep may lack only the bins
  after its last heterozygous site, which are counted (`E6_S2_bins`,
  `_fits`, `_curves`). The report writes its tables and then fails if not.
* **S3, TMRCA decoding** (Fig S3): E1's psmc-rs fit of bottleneck replicate
  1 decoded per bin (`--init-params` of the fit, 0 iterations,
  `--tmrca-out`) against the true pairwise TMRCA re-simulated from the
  replicate's seeds: correlation of the posterior mean and MAP with the
  truth, coverage of the 95% interval by decile, the density of (true,
  posterior mean) and the first 5 Mb of chr1 (`E6_S3_summary`,
  `_calibration`, `_density`, `_track`).
* **S4, thread scaling and determinism** is E3's own tables
  (`E3_thread_scaling`, `E3_determinism`; Fig S4).
* **S5, QC table** (Table S5): `psmc-rs qc` of E2's four genomes with E2's
  bin size and trunk (`E6_S5_qc`, `_findings`).

**Cost** (`plan E6`): 26 jobs, under 1 CPU hour, minutes once E1 is done.

## Figures

`FIG` draws each figure as a runner job after the tables it reads, into
`runs/<experiment>/figures/<name>.{pdf,svg,png}` (Fig 1 into `runs/FIG/`);
`python -m experiment.pipeline.figures` draws them directly from the
tables. Every figure is drawn there and only there. The style, the tables
each figure reads and what captions must add are in
[figures/README.md](figures/README.md).

## Production

Run the production in a checkout nobody develops in, since the runner stops
starting jobs once the pipeline's code or a tool changes: a dedicated
worktree of the branch that holds the pipeline (`git worktree add
../psmc-rs-production <branch>`). A worktree without `experiment/.tools`
uses the main checkout's tools; the runs go to its own `experiment/runs`,
and `start` builds its own psmc-rs 0.2. Then, from its root, with the main
checkout's environment:

```bash
PY=<main checkout>/experiment/.venv/bin/python
$PY -m experiment.pipeline tools                 # identities; no warning about the lock
$PY -m experiment.pipeline tools --build-rs01    # psmc-rs 0.1 from e50d69d, once
$PY -m experiment.pipeline plan                  # the estimates below
$PY -m experiment.pipeline start E3              # first, alone, on a quiet machine on AC power
$PY -m experiment.pipeline status --jobs         # deferred runs say why
$PY -m experiment.pipeline start E1 E2 E4 E5 E6 FIG   # once E3's runner has exited: everything else
```

`start` refuses while a runner of the same runs directory lives. Running
E3 first gives it the quiet machine its timings need while nothing of the
pipeline runs; `start E1 E2 E3 E4 E5 E6 FIG` at once also works, and then
E3's exclusive runs wait until nothing else is left.

| experiment | jobs | CPU hours | wall hours alone (7 cores) | limits |
| --- | --- | --- | --- | --- |
| E1 | 121 | 10 | 1.4 | C psmc 12.5 min and 0.6 GB per fit |
| E2 | 18 | 34 | 5.3 | bootstraps on 7 threads, up to 1.4 h each; C psmc up to 49 min and 3.4 GB |
| E3 | 71 | 21 | 11 (exclusive) | needs the machine to itself; busy periods defer runs |
| E4 | 286 | 4 | 0.6 | at most 1.5 GB per job |
| E5 | 121 | 16 | 2.3 | 20 bootstrap runs of about 45 min on one thread |
| E6 | 26 | under 1 | 0.1 after E1 | S1-S3 wait for E1 |
| FIG | 10 | under 0.1 | minutes | each figure waits for its tables |

The estimates come from `config.Costs` and `config.ResourceModel`, measured
on the development machine (Apple M1 Pro, 8 cores, 16 GB) under load, so
the runs may well finish sooner; psmc-rs 0.2's converged M-steps cost a few
ms per EM iteration (allowed 0.02 s), its E-steps the rest. E3 alone takes
about 11 hours (psmc-rs 0.1's M-steps of 1000 forward-mode iterations and C
psmc dominate it); E1, E2, E4, E5, E6 and FIG together about 9 hours on 7
cores and 12 GB (E2's bootstraps dominate): about 20 hours in all, either
way. Every job is resumable: `stop` (or a crash, or a reboot) and
`start` again rerun only what is not done, at most the jobs that were
running.

Afterwards commit, for every experiment, `experiment/runs/<EXP>/` without
what `.gitignore` drops (simulated genomes, masks, logs, HTML reports,
bootstrap replicates' `.psmc` files): fits, truth and provenance records,
QC reports, bootstrap replicates' JSON, tables and figures.

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

`test_pipeline_*.py` and `test_figures_*.py` cover every module; parts that
need msprime, stdpopsim, psutil, htslib, C psmc or the psmc-rs builds skip
without them. With all of them present, `tiny_runs.py` runs every experiment
at the tiny profile through the runner once per test process (about a
minute); the `EndToEnd` tests check each experiment's tables (for example
that the first iteration's log-likelihood of psmc-rs equals C psmc's to its
printed precision, and that nothing is left to do afterwards), and
`test_figures_draw` draws every figure from those tables and checks every
table schema the figures read against them.
