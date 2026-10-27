# Experiment pipeline

`experiment/pipeline/` produces the manuscript's experiments and figures: it
simulates genomes, runs C psmc and psmc-rs with identical settings, records
how every output was made, writes the tables and draws the figures from
them. Every experiment of the design is implemented: E1 (simulated accuracy
and equivalence, Fig 2, Table 1, Fig S1), E2 (real genomes, Fig 3), E3
(performance, Fig 4, Table 2, Fig S4), E4 (input preparation, Fig 5), E5
(bootstrap calibration, Fig 6), E6 (supplementary analyses, Figs S2 and S3),
E7 (1000 Genomes, Fig 7, Figs S5 and S6) and FIG (every figure,
Fig 1 included). Figure and table numbers in this README follow an earlier
plan of the article; Supplementary Note 1, section 3, maps them to the
article's.

Run it from the repository root with the experiment's environment
(`experiment/.venv` with the exact versions of
`experiment/pipeline/requirements.lock`; `tools` and `start` warn when they
differ):

```bash
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools          # what will run: tool paths, versions, identities
$PY -m experiment.pipeline plan           # CPU hours and wall time on the budget
$PY -m experiment.pipeline start E1       # detached runner; returns at once
$PY -m experiment.pipeline status         # done, running, failed, blocked, pending; ETA
$PY -m experiment.pipeline stop           # or: stop --drain (finish running jobs first)
$PY -m experiment.pipeline start E1       # resume: only what is not done runs
```

The default runs directory, `experiment/runs` (unless `$PSMC_EXP_RUNS`
names another), holds the committed production: give `start`, `status` and
`stop` `--runs-dir NEW` to run into a new directory. In the committed runs,
a job whose tool is another build than the production's (psmc-rs built by
another rustc, another C psmc binary) counts as not done, and `start`
would run it again over the committed results.

`--profile smoke` runs the same jobs at a small size in
`experiment/.runs-smoke/` (3 EM iterations; `plan --profile smoke` counts
162 jobs of E1-E6 and the figures and expects about 3 minutes; E7 has none
until its data stage has run there); `--profile tiny` is for the unit
tests. The production run is described at the end
([Production](#production)).

## Modules

| module | contents |
| --- | --- |
| `config.py` | every parameter: inference settings, scaling, simulation, demographic models, seed scheme, experiments E1-E7, figures, resource budget and cost model, profiles |
| `paths.py` | repository, tools directory (`PSMC_EXP_TOOLS`), runs directory (`PSMC_EXP_RUNS`) |
| `demography.py` | models as `msprime.Demography`; true Ne(t) and pairwise coalescence-time distribution in closed form; checks against msprime's `DemographyDebugger` and stdpopsim |
| `simulate.py` | one diploid genome: psmcfa, bgzipped VCF, truth record; true TMRCA track on demand |
| `tools.py` | psmc-rs 0.2 (this checkout), C psmc and utilities: locate, build, identify; exact command lines; the requirements lock |
| `results.py` | C `.psmc` (every round: LK, QD, RI, TR, MT, RS, PA) and psmc-rs JSON as PSMC models and Ne(t) curves |
| `metrics.py` | time grids, RMSE of log10 Ne against the truth, against the pattern-projected truth and between tools |
| `provenance.py` | checksums and provenance records: when a job is done |
| `runner.py` | the job graph, scheduler, detached daemon, machine lock, status |
| `inputprep.py` | E4's data steps: callable masks, masked VCF, consensus FASTA through fq2psmcfa, bins of the routes compared, generate_multihetsep.py cross-check (`python -m experiment.pipeline.inputprep STEP`) |
| `kg1000.py` | E7's data stage, outside the runner: the 1000 Genomes VCFs to one psmcfa per sample (`python -m experiment.pipeline.kg1000 STEP`) |
| `experiments/e1.py` ... `e7.py` | each experiment as jobs, and its tables (`e3_run.py`: E3's timing wrapper and genome simulation; `e6_analysis.py`: E6's analyses) |
| `experiments/fig.py` | the figures as runner jobs (experiment `FIG`) |
| `figures/` | the manuscript's figures: style, table schemas, one module per figure ([README](figures/README.md)) |

## Design

**Inference.** Every fit, by either tool, uses the psmc README's human
recipe: `psmc -N25 -t15 -r5 -p "4+25*2+4+6"`, which psmc-rs gets as
`N_ITER 25 --pattern 4+25*2+4+6 --t-max 15 --tr-ratio 5 --smooth-lambda 0
--mstep-iters 1000 --bin-size 100`, plus `--threads` per job. Simulations
are scaled with mu = 2.5e-8 and g = 25.

**M-step.** C psmc's Hooke-Jeeves search runs every M-step to convergence,
and so does psmc-rs 0.2's L-BFGS: it stops by a convergence test, and
`--mstep-iters` (default 1000, passed explicitly) is only a cap. In E1 its
M-steps take a median 76 objective evaluations and at most 135 (the `IT`
lines of its `.psmc` output), against a median 3,934 for C psmc's; on E3's
100 Mb genome they take under a millisecond per EM iteration on one thread
(0.55 to 0.84 ms on an Apple M1 Pro). Over E1's 40 genomes the RMSE of
log10 Ne between psmc-rs and C psmc on the informative window, compared
interval by interval, is 3.7e-6 to 2.0e-4 (median 2.5e-5), as small as with
C psmc's own Hooke-Jeeves search run inside psmc-rs's EM in a diagnostic
build (median 2.6e-5; [`experiment/c_equivalence/`](../c_equivalence/README.md)),
and over E2's four real genomes 9.7e-6 to 2.8e-4 (`E1_rs_vs_c.csv`,
`E2_rs_vs_c.csv`). Since psmc-rs's default M-step is the converged one, no
experiment fits or times a separate capped arm: `E1.capped_mstep_iters`
and `E3.capped_mstep_iters` are unset, and setting one of them to a cap
adds a fit (`fit-rs-capped`, `E1_mstep.csv`) or timed runs
(`rs02-capped`) with M-steps capped at that many iterations (the tiny
profile does, at 5, to test them). `test_pipeline_fit_settings.py` checks
that every psmc-rs fit of the production and smoke profiles passes
psmc-rs's own cap (`DEFAULT_MSTEP_ITERS` in `src/opt.rs`) and that neither
fits a capped arm.

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
`constant`, `bottleneck` and `expansion` are ms models (time in 4*N0
generations, size relative to N0), those of psmc-rs 0.1's manuscript, whose
simulation script the truth records name (it is in the git history),
converted exactly (`config.ms_model`); `zigzag` is stdpopsim 0.3.0's `HomSap/Zigzag_1S14`,
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
`reference/<chrom>`, the bootstraps' `bootstrap` (E2's default trunks and
E7) and `bootstrap/trunk<B>`, E5's `summary/<window>/trunk<B>` (resampling
the datasets), and E7's `select/<sample>` (drawing the default set). A seed
depends on nothing but its key, and no seed was chosen after looking at a
result. Every seed is in the truth record, the job's
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
lambda of `4+25*2+4+6` spans intervals 0-3 (about 0-37 ky for zigzag,
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
changes, and the build flags (see Tools); C psmc: version and SHA-256 of the
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
(`$PSMC_EXP_MACHINE_LOCK`, default `~/.cache/psmc-exp/machine.lock` on
macOS and `~/.cache/psmc-exp/machine-<host>.lock` elsewhere, since a
cluster's nodes share the home directory): held shared while a runner runs
ordinary jobs and exclusively while it runs an exclusive one. `stop` sends
SIGTERM: running jobs are terminated and stay pending (also when they die
of the signal first, as under SLURM, which signals every process of a job
at its time limit); `stop --drain` (SIGUSR1) starts nothing new and exits
when the running jobs finish. The runner writes its host to
`runs/_runner/runner.host` and the state, so that `stop` refuses a runner
on another host and a new runner leaves alone the jobs an earlier one on
another host left recorded as running (not on macOS, whose host name
follows the network). A failed job is recorded and blocks
its dependents; the next `start` retries it. A job that exits with status
75 (E3's timing wrapper on a busy machine) is deferred and starts again a
minute later. If the runner itself is killed, the next one stops the jobs
it left running. `start` keeps the machine awake (`caffeinate -i` on macOS;
`--no-keep-awake` to opt out).

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
before they start. The interval of the samples is
`$PSMC_EXP_SAMPLE_SECONDS` (default 0.05, at least 0.01; `start` checks
it): each one lists the machine's
processes, which with ~190 jobs at once on a 192-core node keeps the runner
busy. Every attempt also records the host it ran on and `$SLURM_JOB_ID`;
the runner's state records `provenance.machine()` (CPU model, memory, host,
libc, usable CPUs and the SLURM job on Linux). None of this enters a job's
key.

Without experiments or a jobs file, `status` takes them and `--only` from
the running runner, or else the last one (it says when it keeps the last
`--only`; `--only ''` shows all jobs); `--cores` and `--mem-gb` default to
its budget. It reports a job blocked when no runner would start it: an input it reads is missing and no
job makes it, it does not fit the budget, or it needs a blocked job; an
experiment with a blocked job has the ETA `never`, and its CPU left
counts the blocked jobs. It lists at most 10 running, 10 failed and 10
blocked jobs of each experiment; `--jobs` lists every job that is not
done, and why. `plan` names the jobs that never start on its budget (on
the default 7 cores, E2's and E7's bootstraps), and its times of what
needs them are `never`.

`start`, `status`, `plan` and `list` take `--only REGEX`: only the jobs
whose id matches (`re.search`) are run or shown, and the status, the ETA
and the plan are those of the kept jobs. A kept job whose input a dropped
job would make treats it as an input that no job makes: it runs if the file
exists and is blocked otherwise. This shards a run over runs directories,
e.g. E3's timing runs by repeat over nodes of one CPU model
(`--only 'E3/(sim/|run/.*/rep2$)'` on one node), whose runs directories
are merged before `E3/report` runs once on the merged one.

## E1: simulated accuracy and equivalence

4 models x 10 replicates of 500 Mb (10 chromosomes of 50 Mb). Per replicate:
`sim` (simulation), `fit-c` (C psmc) and `fit-rs` (psmc-rs 0.2, one thread,
also writing `--psmc-out`) on the identical psmcfa (and `fit-rs-capped`,
psmc-rs with capped M-steps, when `E1.capped_mstep_iters` sets a cap); then
`report` writes:

| table | contents |
| --- | --- |
| `E1_table1.csv` | Table 1, per model: RMSE of log10 Ne against the truth per tool (mean, SD, median; informative and fixed windows) and against the pattern-projected truth, the pattern's own error, the tools compared interval by interval (`rmse_rs_vs_c_per_interval_*`: mean, median and largest RMSE over replicates; `max_abs_rs_vs_c_per_interval`: the largest difference of one interval in a window; `max_abs_rs_vs_c_any_interval`: of any interval), largest relative difference in log-likelihood and theta |
| `E1_replicates.csv` | per model, replicate and tool: theta, rho, t_max, N0, RMSE against the truth, log-likelihoods, wall time, peak RSS, load |
| `E1_rs_vs_c.csv` | per model and replicate: the tools compared interval by interval (below): RMSE of log10 Ne (informative and fixed windows) and largest difference of an interval in the window and of any interval, intervals in the window; relative differences of theta, rho, t_max and the log-likelihood |
| `E1_mstep.csv` | per model and replicate (no rows without a capped fit): psmc-rs with capped M-steps against C psmc and against the main fit (interval by interval) and against the truth, wall times |
| `E1_traces.csv` | per iteration (Fig S1): log-likelihood, theta, rho, t_max for both tools |
| `E1_curves.csv` | every fitted Ne(t) step curve (Fig 2) |
| `E1_truth.csv` | the true Ne(t) of each model on a dense grid (Fig 2) |
| `E1_seeds.csv` | the seeds of every chromosome, its heterozygous sites, trees and longest stretch without a heterozygous site |

Iterations count completed EM iterations: C psmc's round k prints the
parameters after k iterations and the log-likelihood after k - 1 (`LK`);
psmc-rs's `loglike[i]` is after i. The report checks that both tools read
the simulated genome (bins and heterozygous bins agree with the truth record).

A fit is compared with the truth at 200 log-spaced times of the window. The
two tools' fits are compared interval by interval instead
(`metrics.interval_diff`): both have the pattern's intervals, whose
boundaries in years differ only through each fit's theta and t_max (by at
most 5.5e-4 in E1 and 1.0e-3 in E2), so a fixed time between two such
boundaries would read a whole step of one curve against the next of the
other. The `*_per_interval` columns hold log10(Ne_rs,k / Ne_c,k) over C
psmc's intervals that overlap the window, the RMSE weighting each by its
log-time width inside the window (as the log-spaced grid weighs the RMSE
against the truth). A first lambda the data barely constrain can run away
outside the window, and `max_abs_log10_ne_rs_vs_c_any_interval` gives the
difference there. In expansion replicate 5 it ran away in both tools:
psmc-rs's stopped at its upper bound (lambda 1e4; psmc-rs bounds every free
lambda) and C psmc's, unbounded, reached about 1.1e6, so the 2.0 there is
set by psmc-rs's bound.

Layout of `experiment/runs/E1/`: `sims/` (genomes; not committed), `truth/`,
`fits/<model>/` (`repNN.c.psmc`, `repNN.rs.json`, `repNN.rs.psmc`,
`repNN.rs-capped.json` with a capped fit), `provenance/`, `tables/`, `figures/` (committed),
`logs/` (not committed).

**Cost** (`plan E1`): 121 jobs, about 10 CPU hours (C psmc 8.3, psmc-rs 1.0,
simulations 0.6), about 1.4 hours alone on the default budget. C psmc holds
one chromosome's forward-backward matrices at a time, about 0.6 GB per job.
Measured in the production run on Ibex ([Production](#production)): 12.4
CPU hours; C psmc 918 to 1,049 s per fit (11.1 hours in all, kept from an
earlier `prod` job) and 0.52 GB, psmc-rs 73 to 78 s (0.8 hours) and
114 MB, the simulations 0.5 hours.

## E2: real genomes

Four genomes of `experiment/real_data` (`config.REAL_GENOMES`): the
turtles HLemySub1 (*Emydura subglobosa*), HLhydTec1 (*Hydromedusa
tectifera*) and HLpelCas1 (*Pelusios castaneus*), scaled with mu 4.6e-9 per
bp per generation, the mean germline rate of nine *Chrysemys picta* trios
(Bergeron et al. 2023, Nature 615:285; a cryptodire's rate: no pleurodire
rate has been measured) and g 15 years for all three (close to Sozzoni et
al. 2025's 15.9 for *P. castaneus* and 12.4 for *E. subglobosa*; *H.
tectifera* has none published), and the human NA12878 (*Homo sapiens*,
CEU; GIAB's HG001), scaled as E7 scales it, with mu 1.25e-8 (Scally &
Durbin 2012) and g 29 years (Fenner 2005, as in Bergström et al. 2020 and
Cousins et al. 2025). NA12878's file, `NA12878_1kGP30x_diploid.psmcfa.gz`,
is its whole genome, chr1-22 (22 sequences, 28,750,027 bins, one gzip
member per chromosome): the psmcfa E7's data stage makes from the 1000
Genomes 30x NYGC joint-called VCF, with the GRCh38 strict mask and every
non-PASS position missing, and E7's `inputs/psmcfa/NA12878.psmcfa.gz` byte
for byte (`config.RealGenome.sha256` is that file's SHA-256 in E7's
`MANIFEST.json`, and a test checks the committed file against it). Both
tools read the gzip file as it is. E7 fits the same file with the same
settings (`fit-c`, `fit-rs-psmcfa`), and E2 bootstraps it with E7's seed,
replicates, trunks and iterations
(`config.RealGenome.bootstrap_shared_with`): E2's run on 50 threads and
E7's on 32 wrote the same 103 files byte for byte (Supplementary Note 1,
section 8). `fig3_real_genomes` and the article's Fig 1c draw E2's band;
no figure draws E7's.
`real_data/Papuan_highlands_diploid.psmcfa`, the chromosome 1 of a Papuan
Highlands human that E2 fitted before, stays for the top-level README's
benchmarks and the Rust tests; E2 does not read it.
`config.RealGenome.scaling_source` gives the citations in full, and
`E2_fit_summary.csv` repeats them. Per genome: `fit-c` (C
psmc), `fit-rs` (psmc-rs 0.2, the identical file and settings),
`boot-rs/trunk500000` (psmc-rs 0.2 with `--bootstrap 100` on the default
trunks of 500,000 bins, splitfa's 50 Mb; seed `seed("E2", sample, 0,
"bootstrap")`, NA12878's `seed("E7", "NA12878", 0, "bootstrap")`; the
bootstrap run's main fit must be `fit-rs`'s fit, every parameter,
log-likelihood, iteration count and the pattern equal, and the
report fails otherwise; its `qc` record may differ, since a bootstrap adds
SHORT_SEQUENCES where sequences are shorter than its trunks) and `qc` (`psmc-rs qc --json`). A genome whose default trunks are
fewer than 20 is bootstrapped a second time on 5 Mb trunks
(`boot-rs/trunk50000`, seed stage `bootstrap/trunk50000`), the size E5
also calibrates. No genome of the production run is: NA12878's chromosomes
form 59 trunks of 50 Mb (only chr21, 1.6% of its bins, is shorter than a
trunk) and the turtles 51 to 1,220, so production runs no second
bootstrap; the smoke and tiny profiles' prefixes (one sequence of 10 and 4
trunks) do. E5's calibration applies to chromosome-scale sequences cut
into trunks, as NA12878's are (trunks of 50 Mb, the size of E5's default
arm, and 59 of them, about as many as its 5 Mb arm's 60); the turtles'
default bootstraps resample whole scaffolds of very different lengths
(889, 34 and 1,220 sequences; 50%, 23% and 97% of their bins lie in
scaffolds shorter than a trunk), a regime E5 does not calibrate.

**Split-first-window check** (the turtles). Hilgers et al. 2025 (Curr Biol
35:927) found false recent peaks of Ne in PSMC curves across many studies,
in all their turtles among them, under the pattern `4+25*2+4+6` that E2
uses: its first free lambda spans the first four intervals, and splitting
that window removed the peaks. psmc-rs fits each turtle again with
`2+2+25*2+4+6` (`fit-rs-split2`) and `1+1+1+1+25*2+4+6` (`fit-rs-split1`),
every other setting `fit-rs`'s (`config.E2.split_patterns`,
`split_genomes`). `E2_split_patterns.csv` compares each fit with `fit-rs`'s:
the largest Ne of the first 8 intervals (the default's first window and the
two after it) and the first interval's Ne, each relative to the plateau (the
median Ne of the next 8 intervals), whether that is a recent peak (the
largest lies after the first interval and is at least 1.25 times both;
`split_recent_intervals`, `split_peak_ratio`), and how far each fit lies from
`fit-rs`'s interval by interval. Figure `figS_turtle_patterns` draws them.

The tools are compared on the window between the 1% and 99% quantiles of
the pairwise coalescence time under C psmc's fitted model (a real genome
has no truth), interval by interval as in E1; the band is the 2.5% to 97.5% quantiles of the replicates'
Ne at each time, as psmc-rs's summary computes it, and the report checks it
against `summary.json`.

| table | contents |
| --- | --- |
| `E2_fit_summary.csv` | per genome and tool: scaling and its source, sequences, called and heterozygous bins, theta, rho, t_max, N0, log-likelihoods, wall time and peak RSS |
| `E2_rs_vs_c.csv` | per genome: the tools compared interval by interval (RMSE of log10 Ne on the window and the fixed window, largest difference of an interval in the window and of any interval), relative differences of theta, rho, t_max and the log-likelihoods, whether the bootstrap runs' main fits equal `fit-rs`'s fit |
| `E2_bootstrap.csv` | per genome and trunk size: replicates, trunks (`coarse_band` under 20), share of bins in sequences shorter than a trunk, seed, band width, how much of each tool's curve the band holds, the check against psmc-rs's summary, whether the main fit is `fit-rs`'s (the fields that differ and the largest difference) |
| `E2_bootstrap_replicates.csv` | every replicate's parameters |
| `E2_bands.csv` | per genome and trunk size: both curves and the band on a dense grid (Fig 3) |
| `E2_curves.csv`, `E2_traces.csv` | the fitted step curves (Fig 3); log-likelihood and parameters per EM iteration |
| `E2_qc.csv`, `E2_qc_findings.csv` | `psmc-rs qc`'s metrics and findings for each input fitted (in production the packaged files) |
| `E2_split_patterns.csv` | per turtle and pattern (`default` is `fit-rs`'s fit): theta, rho, t_max, N0, the final log-likelihood and its gain over the default, the recent intervals' Ne against the plateau (`recent_max_vs_plateau`, `first_vs_plateau`, `recent_peak`), the largest difference from `fit-rs`'s fit in the recent intervals and the RMSE after them in the window |
| `E2_split_curves.csv` | the step curves of those fits (`figS_turtle_patterns`) |

Layout of `experiment/runs/E2/`: `fits/` (`<sample>.c.psmc`, `.rs.json`,
`.rs.psmc`; the turtles' `<sample>.split2.rs.{json,psmc}` and
`.split1.rs.{json,psmc}`), `bootstrap/<sample>/trunk<B>/` (`main.json`,
`replicate_NNN.json`, `summary.json`, `summary.tsv`; the replicates' `.psmc`
files are not committed), `qc/`, `provenance/`, `tables/`, `figures/`;
`inputs/` (smoke and tiny profiles: the first bins of each genome) and
`logs/` are not committed.

**Cost** (`plan E2`): 23 jobs, about 47 CPU hours (the four bootstrap
runs 42.5, each on 50 threads, one replicate per thread, so the runner
needs `--cores 50`, NA12878's the longest at about 17 minutes; C psmc 3.6,
up to 72 minutes for NA12878, and 3.4 GB for HLhydTec1; the six
split-first-window fits 0.6), about 2.1 hours alone on 50 cores. The
bootstraps' output does not depend on their threads (replicate i draws
from its own seed). Measured in the production run on Ibex
([Production](#production)): 41.4 CPU hours; the bootstrap runs 36.0
(574 to 631 s each for the turtles and 957 s for NA12878, at most
1.9 GB), C psmc 4.5 (3,419 to 3,610 s for the turtles and 6,645 s for
NA12878, at most 3.3 GB, HLhydTec1's; kept from earlier `prod` jobs),
psmc-rs 0.8 (272 to 425 s per fit, the split-first-window fits 272 to
288 s, at most 115 MB).

## E3: performance

Run time and peak memory of C psmc and psmc-rs 0.2 on constant-model
genomes of 100, 300, 1000 and 3000 Mb (1, 3, 10 and 30 chromosomes of
100 Mb; one simulation per size, the chromosomes shared between sizes), 25
EM iterations of the recipe, 5 repeats each: C psmc on its one thread;
psmc-rs 0.2 on 1, 8 and 30 threads (and with capped M-steps,
`rs02-capped`, when `E3.capped_mstep_iters` sets a cap); and psmc-rs 0.2
on the 3000 Mb genome with 1, 2, 4, 8, 16 and 30 threads (thread scaling):
19 setups, 95 timed runs. psmc-rs works on one sequence per thread, so the
30 chromosomes of the largest genome make 30 threads the most that can
help. A setup whose estimated peak memory exceeds 64 GB is not run and
Table 2 says why (none with 100 Mb chromosomes: C psmc holds about 1 GB,
psmc-rs 0.2 well under 1 GB).

Every run is an exclusive job (nothing else runs, on this runner or any
other), ranked by repeat, with as many cores as threads, and timed by
`e3_run measure` (wall time, CPU time, peak RSS of the tool's process tree,
sampled every 0.5 s with the tool's own `ru_maxrss`, the machine's load and
the busiest processes before the run, the CPU other processes used during
it, the power source, and on Linux the CPU model, cpufreq governor, boost,
host and SLURM job). The runner starts an exclusive job whatever its
`--cores`, but its memory must fit `--mem-gb` (the default 12 GB fits
every E3 run). `measure` refuses (exit status 69, a failure the next `start`
retries) to time a run of more threads than the CPUs it may use (its
affinity: a SLURM allocation's), since it would be timed oversubscribed.
The tool runs without `PSMC_*` and Rust build variables in its environment.
A run is kept only when it is clean: the machine became quiet (at most 10%
of all cores and at most 1 core busy) within 3 minutes, other processes
kept at most 5% of all cores and at most 0.5 core busy on average during
the run, the machine did not sleep, and it ran on AC power without Low
Power Mode. On a laptop's 8 cores the shares decide; on a node of 128 or
192 cores the core counts do. Quiet before the run is judged on the whole
machine's busy CPU time (everything but idle and iowait: processes, kernel
threads, interrupts, steal), since nothing of ours runs then (`measure`
reads the processes outside that window). During the run, "other
processes" are the user-space processes other than the tool's and the
`measure` wrapper: `measure` reads every process's CPU time, and that of
the children it has reaped (Linux's cutime and cstime), just before the
tool starts and right after it ends, matched by pid and creation time (a
process started during the run counts whole). A process that ended during
the run, whether it started before it (`other_processes_exited` counts
those) or in it (a `make -j`'s compilers, a script's commands), counts
through the process that waited for it, its parent or, orphaned, init;
one that started before the run counts with its CPU before it too, which
errs towards rejecting the run. Only a process that nothing waits for
(its parent ignores SIGCHLD) is missed: its CPU shows in
`unaccounted_cores`, the machine's busy CPU that nothing measured
explains, which is recorded but not limited, since it also holds the
differences between how the kernel counts the machine's CPU time and the
processes'. Steal time (a hypervisor's other machines; 0 on Ibex's bare
metal) is limited with the other processes. Kernel threads (kthreadd's
children) and interrupt time (irq, softirq) are left out of the limit and
recorded apart: part of them is the kernel's work for the tool itself (a
run keeping 30 cores busy adds about 0.2 core on Ibex's 128-core nodes,
whose idle busy CPU is already 0.2 to 0.9 core, a median 0.3: counted, it
rejected those runs against the 0.5 core again and again), the rest the
machine's housekeeping that the quiet check bounds, and neither is another
workload. Where not every process can be read (other users' processes on
macOS, or a `/proc` mounted `hidepid`) or reaped children's CPU time is not
readable (every system but Linux), the limit applies to the machine's busy
CPU time but the tool's, kernel and interrupt time included, so nothing is
under-counted; each timing record says which (`during.other_method`:
`processes` or `machine`) and keeps the machine-wide measure either way.
Otherwise `measure` writes nothing and exits with status 75, and the runner
starts the run again a minute later; the report takes medians from clean
runs only, and flags clean runs timed on more than one kind of machine (CPU
model, governor, boost), whose medians and speed-ups would mix them: keep
E3's runs on nodes of one CPU model.

| table | contents |
| --- | --- |
| `E3_table2.csv` | Table 2, per tool, thread count and size: status (measured, skipped with the reason, incomplete), estimated peak memory, clean and measured repeats, median, minimum and maximum wall time, median CPU time, seconds per iteration, peak RSS, speed-up over C psmc, C psmc's RSS over the tool's, the M-step budget, whether the repeats' outputs are identical |
| `E3_runs.csv` | every run: measurements, machine state (before: busy cores, and of them processes, kernel threads, interrupts; during: the other work the clean check limited by `other_method`, the machine-wide measure, kernel threads, interrupts, steal, what none explains, processes that ended), power, clean or not, CPU model, governor, boost, usable CPUs, host, SLURM job, theta, rho, t_max, log-likelihood, output SHA-256 |
| `E3_thread_scaling.csv` | Fig S4: psmc-rs 0.2 on 3000 Mb per thread count: median wall time, speed-up, efficiency, peak RSS, whether the JSON is byte-identical to the one-thread run |
| `E3_determinism.csv` | Fig S4: every output against its tool's reference at the same size (fewest threads, first repeat) |
| `E3_environment.csv` | machine (each distinct value: the hosts of merged shards), tools (identities from the runs' records, with C psmc's compiler and CFLAGS and psmc-rs's rustc; binary SHA-256), protocol, power sources, runs not clean, the measures of other work used and the median and maximum of each CPU-use column of `E3_runs.csv` (the limited other work for each measure apart), the kinds of machine of the clean runs and whether they mix (`clean_runs_on_mixed_machines`) |
| `E3_seeds.csv` | every seed, with each chromosome's trees and longest stretch without a heterozygous site |

Layout of `experiment/runs/E3/`: `sims/` (not committed), `truth/`,
`fits/<tool>/<size>Mb/t<threads>/rep<r>.{psmc,json}`, `timings/...` (the
timing records), `provenance/`, `tables/`, `figures/`.

**Cost** (`plan E3`): 100 jobs; the 95 timing runs take about 10 hours of
wall time one after another by the laptop's estimates (C psmc about 1 hour
each on 3000 Mb; psmc-rs 0.2's converged M-steps take under a second per
run). They need the machine to themselves: close other work, stay on AC
power, and expect the runner to defer runs while the machine is busy
(`status --jobs` says why). Measured on Ibex, one exclusive AMD EPYC 7702
node per repeat: 12.3 CPU hours in the 95 timed runs (the sum of
`cpu_seconds` in `E3_runs.csv`: C psmc 9.0, psmc-rs 0.2 3.3) and 10.3
hours of wall time, 2.04 to 2.07 per repeat (the sum of `wall_seconds`);
C psmc took a median 74 minutes on 3000 Mb.
[Supplementary Note 1, section 10](../Supplementary_Note_1_Reproducibility.md#10-performance-e3)
gives the results.

## E4: input preparation

**The question.** PSMC sees a genome as bins of s = 100 bases. fq2psmcfa
calls a bin missing (`N`) only when more than 90% of its bases are
uncallable, and C psmc takes every called bin as s bases. With missing data
dispersed in short runs almost every bin is still called but holds only
about (1 - f) of its bases, so theta per bin, N0 = theta / (4 mu s) and
every time and size come out low by about the missing fraction f, or
more (about 1.2 f in the results below). psmc-rs counts the callable bases
c of each bin from the VCF and its callable mask and, by default
(`--bin-callable auto`), fits each bin by them: P(T | k) = exp(-theta (c / s) avg_t_k), the callable-length model of
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
(informative and fixed windows) and against the complete-data fit (exact
over log time on the informative window, `metrics.log_time_diff`: the
fits' time scales differ by percents, and on the window's 200 times a time
between two nearby boundaries reads a whole step, which made the RMSE of
close fits noisy by tens of percent), the
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

**Results** (production run, 5 replicates; `E4_summary.csv`,
`E4_routes.csv`, `E4_bins.csv`, `E4_crosscheck.csv`). Dispersed missing
data bias the full-bins routes by about 1.2 times the missing share of the
called bins (theta at 0.877 to 0.883 of the complete-data fit with 10%
missing and 0.644 to 0.658 with 30%, whose called bins are 90.1% and 70.5%
callable on average), and the callable-length model removes most of the
bias (0.988 to 0.995 and 0.971 to 0.988: at 30% it stays 1.9% low on
average). Its RMSE of log10 Ne against the complete-data fit is 4.6 and
4.4 times smaller than `mask_ignore`'s (mean 0.041 against 0.191 at 10%,
0.080 against 0.349 at 30%). Against the truth the contrast is clear at 30%
(mean 0.280 against 0.449; complete data 0.273) and smaller at 10% (0.275
against 0.323), where PSMC's own error weighs more, so theta and the RMSE
against the complete-data fit are the measures of the bias. Clustered
missing data empty whole bins, which both models leave out, and change
little either way: at 30% both fits are about 2.4% low in theta, as low as
the callable-length fit of dispersed data, so what that fit leaves is not
the bins' doing. multihetsep does not recover the exact route, dispersed or
clustered (theta 0.669 to 0.695 at 30% dispersed, 0.888 to 0.914 at 30%
clustered): placing every uncalled site of an interval at its start turns
scattered missing bases into runs of `N` bins and leaves the bin of each
heterozygous site less callable than it is, so the VCF with its mask is
the input to use. The VCF without its mask is the worst route (theta 0.638
to 0.652 at 30% dispersed and 0.417 to 0.440 at 30% clustered, whose RMSE
against the truth is 0.606). Every identity held: the fq2psmcfa route's
bins equal the exact bins (25 of 25) and its fit `mask_ignore`'s (5 of 5
per mask), `mask_auto` equals `mask_ignore` on the control (5 of 5), and
psmc-rs's multihetsep equals `generate_multihetsep.py`'s (50 of 50
chromosomes). The smaller simulations of `experiment/validation/` (three
runs of three 200 Mb genomes, two of this model and one of the expansion
model) gave similar theta (0.63 to 0.66 full bins, 0.944 to 0.991 with the
counts at 30% dispersed); its multihetsep numbers also include
`--bin-callable approximate`, which E4 does not fit.

**Rescaling cross-check** (`E4_rescaling.csv`; not a route, not in Fig 5).
Before the callable-length model, E4 corrected the bias after the fit: the
full-bins fit read with s = the mean callable bases of its called bins
instead of 100, that is times and sizes multiplied by 100 / s (at 30%
dispersed missing data, mean RMSE against the truth 0.449 as fitted, 0.286
rescaled, 0.280 with the callable-length model and 0.273 with complete
data). The table keeps that global correction of
`mask_ignore` (s from its JSON's `bin_callable.callable.mean_fraction`)
beside `mask_auto`, per mask and replicate, as a check of the model: the two should agree on the scale, and
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
| `E4_rescaling.csv` | per mask and replicate: the rescaling cross-check, `mask_auto`, `mask_ignore` and `mask_ignore` rescaled (theta against the complete-data fit, RMSE against the truth and against the complete-data fit, mean log10 bias), and the RMSE and largest difference of log10 Ne between the rescaled curve and `mask_auto`'s, interval by interval (`*_per_interval`; the two have the same pattern and bins, and their scales are the theta columns) |
| `E4_bins.csv` | per mask, replicate and route: the exact T/K/N bins against the route's (`N_T`: missing in the exact bins, called in the route), bins only one has (the multihetsep tail after its last row) |
| `E4_masks.csv` | per mask and replicate: target and realised missing fraction, uncallable runs and their mean length, heterozygous sites removed |
| `E4_qc.csv` | every QC finding of every fit (mask, route, replicate, code, severity: `warning` or `note`, from the fit JSON's `qc.warnings` and `qc.notes`; the QC JSON that E2's tables read calls a note's severity `info`) |
| `E4_crosscheck.csv` | per chromosome of replicate 1: rows of generate_multihetsep.py and psmc-rs, identical |
| `E4_curves.csv`, `E4_truth.csv`, `E4_seeds.csv` | every fitted Ne(t) step curve, the true Ne(t), every seed |

Layout of `experiment/runs/E4/`: `sims/repNN/` (the genome, and per mask
`callable.bed.gz`, `variants.vcf.gz`, `fq2psmcfa.psmcfa`,
`input.multihetsep`, `bins/*.psmcfa`; not committed), `truth/`, `masks/`,
`consensus/`, `bins/`, `crosscheck/`, `fits/<mask>/<route>/`,
`provenance/`, `tables/`, `figures/` (committed).

**Cost** (`plan E4`): 286 jobs, about 4 CPU hours (the 125 fits 3.1; the
generate_multihetsep.py cross-checks 0.5), about 0.6 hours alone, at most
1.5 GB per job (the consensus FASTA streamed through fq2psmcfa). Measured on
Ibex: 3.4 CPU hours; the fits 2.7 (70 to 101 s each), the cross-checks
0.5; at most 0.8 GB (the consensus FASTA).

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
  the bootstrap's. Part of the bias is the pattern's: its steps cannot
  follow the truth within a free lambda's span. So the tables also give the
  share of genomes whose band holds that genome's pattern-projected truth
  (`metrics.projected_truth` of its fit, the true Ne averaged over each
  span in coalescence-rate terms, as E1's Table 1 uses it).

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
| `E5_summary.csv` | per trunk size and window: median, range and share below 1 of the median SD ratio, with a 95% interval from resampling the datasets; median IQR ratio; mean and least pointwise coverage of the truth, of the fits' mean and of the pattern-projected truth, the mean's 95% interval from resampling the datasets (the independent units: the points of one dataset are strongly correlated) and the spread of the datasets' shares of covered points; median band width and true 95% range (log10); median true SD, bias and RMSE |
| `E5_sd_ratio.csv` | per trunk size and time: the true SD and its interval, the bootstrap SD, the SD ratio (median, quartiles, range over genomes), the IQR ratio |
| `E5_coverage.csv` | per trunk size and time: genomes covered, coverage and its interval, coverage of the fits' mean, coverage of each genome's pattern-projected truth and its interval, band width |
| `E5_sampling.csv` | per time: true Ne, and the 50 fits' mean log10 Ne, median and 95% range of Ne, SD and its interval, IQR, bias and RMSE |
| `E5_bands.csv` | per bootstrapped genome, trunk size and time: its fit, its band and median, its bootstrap SD, whether the band covers the truth; its pattern-projected truth and whether the band covers it |
| `E5_datasets.csv` | per bootstrap run: trunks, seed, replicates fitted and skipped, median SD ratio, coverage of the truth and of the projected truth, the checks, wall time, CPU time and peak RSS |
| `E5_fits.csv` | per genome: heterozygous sites and bins, theta, rho, t_max, N0, final log-likelihood, RMSE against the truth and against its pattern-projected truth, wall time and peak RSS |
| `E5_seeds.csv` | every seed |

**Figure 6** (`figures/fig6_bootstrap_calibration.{pdf,svg,png}`, drawn by
`FIG` from the tables, see `figures/README.md`; the supplement's Fig S3): a,
one bootstrapped genome's fit and bootstrap 95% band (5 Mb blocks) over the
true Ne and the 95% range of the 50 independent fits; b, the bootstrap SD
(median over the 10 genomes) against the true SD, one dot per time of the
window, for both block sizes, with the band where a calibrated bootstrap
would fall; c, per genome, the share of the window's times at which its band
holds the fits' mean and the truth, for both block sizes, against the
nominal 95%.

Layout of `experiment/runs/E5/`: `sims/` (genomes; not committed),
`truth/`, `fits/bottleneck/repNN.rs.json`, `boot/trunk<B>/repNN.rs.json` (the
bootstrap run's main fit), `boot/trunk<B>/repNN/` (`replicate_NNN.json`,
`summary.json`, `summary.tsv`; the replicates' `.psmc` files, 0.1 MB each, are
not committed), `provenance/`, `tables/`, `figures/`, `logs/` (not committed).

**Cost** (`plan E5`): 121 jobs, about 16 CPU hours (the 20 bootstrap runs,
51 fits each on one thread, about 45 minutes each), about 2.3 hours alone;
at most 0.2 GB per job. Measured on Ibex: 12.2 CPU hours; the bootstrap runs
1,707 to 2,117 s each (11.4 hours), the fits 45 to 55 s; at most 0.14 GB.

## E6: supplementary analyses

E6 reads what E1 makes and adds the runs only it needs
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
  after its last heterozygous site, which are counted; the fits are
  compared with the psmcfa fit interval by interval, as E1 compares the
  tools (`E6_S2_bins`, `_fits`, `_curves`). The report writes its tables and then fails if not.
* **S3, TMRCA decoding** (Fig S3): E1's psmc-rs fit of bottleneck replicate
  1 decoded per bin (`--init-params` of the fit, 0 iterations,
  `--tmrca-out`) against the true pairwise TMRCA re-simulated from the
  replicate's seeds: correlation of the posterior mean and MAP with the
  truth, coverage of the 95% interval by decile, the density of (true,
  posterior mean) and the first 5 Mb of chr1 (`E6_S3_summary`,
  `_calibration`, `_density`, `_track`).
* **S4, thread scaling and determinism** is E3's own tables
  (`E3_thread_scaling`, `E3_determinism`; Fig S4).

**Cost** (`plan E6`): 21 jobs, under 1 CPU hour, minutes once E1 is done.
Measured on Ibex: 0.2 CPU hours; S3's tables at most 1.3 GB, its decoding
at most the 112 MB of `ru_maxrss` that its job inherits from the runner
(33 MB in the sampled process tree).

## E7: 1000 Genomes

psmc-rs on the 2,504 unrelated genomes of the 1000 Genomes Project's 30x
NYGC data (26 populations; Byrska-Bishop et al. 2022, Cell 185:3426), with
the GRCh38 strict accessibility mask (`config.E7`; mu 1.25e-8, g 29).

**Data stage** (`kg1000.py`, outside the runner: its inputs are 1.68 TB).
`setup` records the sources (HEAD of every VCF on EBI and its S3 mirror,
checked against the 1000 Genomes manifest; the indexes, sequence index,
pedigree, strict mask and GIAB HG001 truth downloaded, SHA-256 recorded),
picks the samples (the default set: `per_population` per population,
NA18507 and NA19239, two of Li & Durbin 2011's genomes, and NA12878
(GIAB's HG001) first, the others drawn by seed) and cuts chr1-22 into chunks of 15 Mb. Each chunk is read once,
reduced to GT and FILTER and split into one het-or-missing VCF per sample
(`bcftools +split`); every non-PASS position goes into one negative mask
shared by all samples; then `psmc-rs convert --vcf-mode mask` writes each
sample's psmcfa. The chunks' records must sum to the index's count of each
chromosome, and the pilot (chr21) must give, for the three samples, the same
psmcfa byte for byte as `psmc-rs convert --vcf-sample` on the whole joint
VCF. `assemble-samples` joins the chromosomes and writes `MANIFEST.json`
(sources, tool versions, commands, checks and the SHA-256 of every output;
it is committed, and the files E7's jobs read are to be deposited on
Zenodo: the psmcfa files, the default set's VCFs, `mask/callable.bed`,
`nonpass/nonpass.bed.gz`, `qc/samples.tsv` and GIAB's benchmark in
`meta/`). NA12878's
psmcfa is also E2's human genome, committed as
`experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz` (the same
SHA-256). A chromosome
is either streamed by region over HTTPS (`chunk`) or, where one connection
is slow, downloaded to node-local disk in parallel byte ranges, MD5
checked, and processed there (`chrom`; on Ibex 0.4 to 1.7 MB/s per
connection). `E7.input = "panel"` reads the phased panel instead (MAC >= 2,
imputed genotypes: it lacks part of each genome's PASS heterozygous
SNVs), a fallback whose figures must say so. `run` does every step in one
process; on a cluster `experiment/slurm/submit.sh e7-prep` chains setup, a
task per chromosome and the samples. `setup`, `chunk`, `chrom` and
`assemble-chrom` skip what their records in `inputs/records/` say is done,
and the repository commits those records with `MANIFEST.json`,
`samples.tsv` and `chunks.tsv`: in the committed runs the stage remakes
nothing (the chr21 pilot, which runs again, and `assemble-samples` stop for
want of the files those records describe).
It remakes the inputs into a new `--runs-dir` (given to `start E7` too), or
once `runs/E7/inputs/records/` and `MANIFEST.json` are removed.

**Fits** (the runner, once the data stage's `MANIFEST.json` exists, as in
the committed runs; before, E7 has no jobs, and FIG declares the figures
that read E7's tables only where those tables exist; in a clone, without
the inputs, E7's 2,639 jobs wait, blocked): every sample's psmcfa with
psmc-rs (`fit-rs-psmcfa`); the default set's VCFs with the mask and
psmc-rs's callable bases per bin (`fit-rs-vcf`: Fig S5's dark lines and
Fig S6b; Fig 7's curves are the psmcfa fits), and its
psmcfa with C psmc (`fit-c`); psmc-rs's bootstrap of the three samples
(`boot-rs`, 100 replicates; its main fit must equal `fit-rs-psmcfa`'s);
the first default sample of every population with the first time window
split (`fit-rs-split`, `1+1+1+1+25*2+4+6`: a check for the false recent
peaks of Hilgers et al. 2025); NA12878's heterozygous SNVs against GIAB
(`giab`); `report`. Tables `E7_samples`, `_inputs`, `_fits`, `_curves`,
`_pop_bands` (median and 10%-90% spread of Ne across a group's genomes),
`_rs_vs_c`, `_bootstrap`, `_bands`, `_route_shift` (VCF route against
psmcfa route), `_split`, `_giab` and `_throughput`; what each holds is in
`experiments/e7.py` and [figures/README.md](figures/README.md).

**Cost** (`plan E7`, from `config.Costs`): 2,639 jobs and 480 CPU
hours, 2,504 single-thread fits of about 8 minutes and 52 C psmc fits of
about 70 minutes dominating; the bootstraps run on 32 threads. The data
stage is network-bound: hours per chromosome where the link is slow.

## Figures

`FIG` draws each figure as a runner job after the tables it reads, into
`runs/<experiment>/figures/<name>.{pdf,svg,png}` (the article's Fig 1,
`art1_agreement`, and the schematic, `fig1_workflow`, into
`runs/FIG/figures/`); `python -m experiment.pipeline.figures` draws them
directly from the tables. Every figure is drawn there and only there. The
style, the tables each figure reads and what captions must add are in
[figures/README.md](figures/README.md).

## Production

Run the production in a checkout nobody develops in, since the runner stops
starting jobs once the pipeline's code or a tool changes: a dedicated
worktree of the production commit (`git worktree add --detach
../psmc-rs-production <commit>`). A worktree without `experiment/.tools`
uses the main checkout's tools; the runs go to its own `experiment/runs`,
and `start` builds its own psmc-rs 0.2. Then, from its root, with the main
checkout's environment:

```bash
PY=<main checkout>/experiment/.venv/bin/python
$PY -m experiment.pipeline tools                 # identities; exits 1 if a tool is missing
$PY -m experiment.pipeline tools --build-c       # C psmc with a recorded build (not needed on the laptop)
$PY -m experiment.pipeline plan                  # the estimates below
$PY -m experiment.pipeline start E3              # first, alone, on a quiet machine on AC power
$PY -m experiment.pipeline status --jobs         # deferred runs say why
$PY -m experiment.pipeline start E1 E2 E4 E5 E6 FIG --cores 50 --mem-gb 64   # once E3's runner has exited
$PY -m experiment.pipeline.kg1000 run --jobs 16 --pilot chr21   # E7's data stage (or submit.sh e7-prep; below)
$PY -m experiment.pipeline start E7 FIG --cores 50 --mem-gb 64   # then its fits and figures
```

E7's data stage remakes nothing while the committed
`runs/E7/inputs/records/` and `MANIFEST.json` are there
([E7](#e7-1000-genomes)): remove them first, or copy E7's inputs into
`runs/E7/inputs/` instead.

`start` refuses while a runner of the same runs directory lives. Running
E3 first gives it the quiet machine its timings need while nothing of the
pipeline runs; `start E1 E2 E3 E4 E5 E6 FIG` at once also works, and then
E3's exclusive runs wait until nothing else is left. On a SLURM cluster,
the scripts of [`experiment/slurm/`](../slurm/README.md) run the same
commands in allocations: the manuscript's production ran there, on KAUST's
Ibex, and drew the figures on a laptop from the runs directory copied
back.

| experiment | jobs | CPU hours | wall hours alone (7 cores) | CPU hours measured on Ibex | limits |
| --- | --- | --- | --- | --- | --- |
| E1 | 121 | 10 | 1.4 | 12.4 | C psmc 12.5 min (Ibex: 15 to 17.5) and 0.6 GB per fit |
| E2 | 23 | 47 | 2.1 (50 cores) | 41.4 (the C psmc fits, 4.5, from earlier `prod` jobs, the rest from the production's; Supplementary Note 1, section 5) | bootstraps on 50 threads: `--cores 50`; C psmc up to 72 min (NA12878; the turtles' 46 to 49 took 57 to 60 on Ibex) and 3.4 GB |
| E3 | 100 | 15 | 9.9 (exclusive) | 12.3 (the 95 timed runs: the sum of `cpu_seconds` in `E3_runs.csv`; five exclusive `e3` jobs on four nodes, below) | needs the machine to itself; busy periods defer runs |
| E4 | 286 | 4 | 0.6 | 3.4 | at most 1.5 GB per job |
| E5 | 121 | 16 | 2.3 | 12.2 | 20 bootstrap runs of about 45 min (Ibex: 28 to 35) on one thread |
| E6 | 21 | under 1 | 0.1 after E1 | 0.2 | S1-S3 wait for E1 |
| E7 | 2,639 | 480 | 11 (50 cores) | 358.1 (the C psmc fits 71.7; every fit and bootstrap in one job, below) | after its data stage; bootstraps on 32 threads |
| FIG | 16 | under 0.1 | minutes | (laptop: 16 figures in 8 s, 32 CPU seconds) | each figure waits for its tables |

The estimates come from `config.Costs` and `config.ResourceModel`, measured
on the development machine (Apple M1 Pro, 8 cores, 16 GB) under load, so
the runs may well finish sooner; psmc-rs 0.2's converged M-steps cost under
a millisecond per EM iteration (0.55 to 0.84 ms on E3's 100 Mb genome, on
one thread of the M1 Pro; allowed 0.02 s), its E-steps the rest. E3 alone
takes about 10 hours (C psmc's runs dominate it); E1, E2, E4, E5, E6 and
FIG together about 2.1 hours on 50 cores and 64 GB (E2's bootstraps
dominate; E2 cannot start on fewer than 50 cores): about 12 hours in all,
either way. Every job is resumable: `stop` (or a crash, or a reboot) and
`start` again rerun only what is not done, at most the jobs that were
running.

The production run of E1, E2 and E4 to E6 (`experiment/slurm`'s `prod` job
54629001 on one Ibex node, at ae34d727) ran 498 of their 572 jobs in 1 hour
6 minutes, 53.4 CPU hours in all; the other 74 come from earlier `prod`
jobs with the same keys (Supplementary Note 1, section 5): the C psmc fits
of E1 and E2 (53386770 and 53452276), E4's cross-checks against
generate_multihetsep.py (54256959) and its `bins` jobs (54232546). E7's
2,639 jobs, C psmc's and psmc-rs's fits, the bootstraps, the GIAB
comparison and the report, ran in one job on one Ibex node (54634918), in 6
hours 21 minutes, 358.1 CPU hours in all. E3 ran in five `e3` jobs at the
same commit on 27 October 2026, one per repeat, each alone on an exclusive
AMD EPYC 7702 node (54629002 to 54629006 on cn513-09-l, cn513-02-r,
cn514-15-l, cn513-02-l and, again, cn513-09-l): each simulated E3's genomes
(the committed records are repeat 5's, the copy `merge_e3.sh` keeps; 0.2
CPU hours) and then timed its repeat's 19 runs, in 2 hours 8 to 9 minutes
from the first's start to the last's end, 12.8 CPU hours over the 95 runs
with the `measure` wrapper; its report ran in `e3-report` job 54677793
(cn511-08). The measured costs above come from the provenance records. The
`prod` jobs' fits and bootstraps shared the node with one another (1-minute
load averages of 68 to 134 when they started; 78 to 168 in E7's job), so
their times are costs, not controlled timings, though E7's two programs,
sharing one node in one job, compare with each other under the same
conditions; E3 times the tools, each run alone on its node (1-minute load
averages of 1 to 10, left by E3's own jobs before it). E3's measured cost
is the tools' CPU time in its 95 timed runs (`E3_runs.csv`), without the
simulations each repeat ran or the `measure` wrapper.

Afterwards commit, for every experiment, `experiment/runs/<EXP>/` without
what `.gitignore` drops (simulated genomes, masks, logs, HTML reports,
bootstrap replicates' `.psmc` files): fits, truth and provenance records,
QC reports, bootstrap replicates' JSON, tables and figures.

## Tools

The tools directory is `$PSMC_EXP_TOOLS`, else `experiment/.tools` (of the
main checkout when run from a worktree without one): `psmc_c/psmc` (lh3/psmc
0.6.5-r74), `psmc_c/utils/{fq2psmcfa,splitfa}` and
`msmc-tools/generate_multihetsep.py`. psmc-rs 0.2 is this
checkout's `target/release/psmc-rs`, built by `start`. A psmc-rs build is
identified by rustc, the rustflags and `CARGO_PROFILE_RELEASE_*` variables
and the target features they enable; one with `-C target-cpu=native`, or
whose flags or release profile a cargo configuration file sets
(`$CARGO_HOME/config.toml`, a `.cargo/config.toml` above the build), is
refused, since its identity could not tell what changed it.

`tools --build-c` builds C psmc, fq2psmcfa and splitfa from the sources in
`<tools>/psmc_c` (lh3/psmc's) with `$CC` (default `gcc`, else `cc`),
`$PSMC_C_CFLAGS` (default the Makefile's `-g -Wall -O2 -Wno-unused-function`)
and `$LDFLAGS`, and records the compiler, flags, commands and checksums in
`psmc_c/BUILD.json`, which adds the compiler and CFLAGS to C psmc's
identity and E3's environment table. The cluster's build, which made every
C psmc fit of the production run, is gcc 11.4.1 with `-g -Wall -O3
-march=x86-64-v3 -ffp-contract=off -Wno-unused-function`
(`experiment/slurm/site.env.example`; every C psmc provenance record holds
the compiler and the flags): its output was byte-identical to the
Makefile's `-O2` on AMD Zen 2, Zen 5 and Intel Cascade Lake nodes, and it
ran 1.2 to 2.0 times as fast
([`experiment/crossplatform/x86_cpus/`](../crossplatform/x86_cpus/README.md),
C psmc's production build; `-ffp-contract=off` keeps gcc from fusing
multiply-adds, which would change the results). On macOS arm64 the default
is `-O1`, since an `-O2` build segfaults there; the laptop's psmc, built by
hand with `-O1`, has no `BUILD.json` and is identified by its version and
checksum, so its compiler and flags are in no provenance record. `--build-c`
does not replace programs that `BUILD.json` does not describe unless given
`--force`, and keeps whatever it replaces as `<program>.prev` (and
`BUILD.json.prev`).

## Tests

```bash
experiment/.venv/bin/python -m unittest discover -s experiment/tests
```

`test_pipeline_*.py` and `test_figures_*.py` cover every module; parts that
need msprime, stdpopsim, psutil, htslib, C psmc or the psmc-rs builds skip
without them. With all of them present, `tiny_runs.py` runs E7's data stage
on a synthetic cohort (`kg1000 fixture`: msprime genomes written as the
1000 Genomes files, read locally; it needs `bcftools +split`) and then every
experiment at the tiny profile through the runner once per test process
(about three minutes); the `EndToEnd` tests check each experiment's tables (for example
that the first iteration's log-likelihood of psmc-rs equals C psmc's to its
printed precision, and that nothing is left to do afterwards), and
`test_figures_draw` draws every figure from those tables and checks every
table schema the figures read against them.
