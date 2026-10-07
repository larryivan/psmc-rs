# Supplementary Note 1 — Reproducibility

This note describes how to reproduce the manuscript's experiments, tables
and figures from the `psmc-rs` repository: from the committed results
without running any tool, from scratch on one machine, or on a SLURM
cluster, as the production runs were made. It also says how every output
records the way it was made, how C psmc was built, and which results do
not depend on the thread count or the CPU. Paths are relative to the
repository root. The pipeline itself is described in
`experiment/pipeline/README.md`, the figures in
`experiment/pipeline/figures/README.md` and the cluster scripts in
`experiment/slurm/README.md`.

The experiments are E1 (simulated accuracy and equivalence with C psmc:
Fig 2, Table 1, Fig S1), E2 (four real genomes: Fig 3), E3 (performance:
Fig 4, Table 2, Fig S4), E4 (input preparation and partly callable bins:
Fig 5), E5 (bootstrap calibration: Fig 6) and E6 (supplementary analyses:
Figs S2 and S3, Table S5); FIG draws every figure. `experiment/scripts/`,
`experiment/run.py`, the notebook and `experiment/runs/{main_text,
real_data,supplementary}/` are the earlier pipeline of psmc-rs 0.1 and
make none of the manuscript's results.

## 1) What the repository holds

Each experiment writes `experiment/runs/<EXP>/`. The release commits
(until it is tagged the production runs of E1, E2, E4, E5, E6 and FIG are
not yet committed, and E3 is still being run):

| item | where |
| --- | --- |
| real genomes (inputs of E2 and E6's S5) | `experiment/real_data/*.psmcfa` |
| truth records of the simulated genomes (settings, seeds, software versions, per-chromosome counts and checksums, the true Ne(t)) | `runs/<EXP>/truth/` |
| every fit: psmc-rs JSON and `.psmc`, C psmc `.psmc` | `runs/<EXP>/fits/` |
| bootstrap runs: main fit, every replicate's JSON, `summary.json` and `summary.tsv` | `runs/E2/bootstrap/`, `runs/E5/boot/` |
| E4's masks, bins and cross-checks; QC reports | `runs/E4/{masks,bins,consensus,crosscheck}/`, `runs/{E2,E6}/qc/` |
| one provenance record per job | `runs/<EXP>/provenance/` |
| tables and figures | `runs/<EXP>/tables/`, `runs/<EXP>/figures/` (Fig 1: `runs/FIG/`) |

Not committed, and remade on demand from the seeds: the simulated genomes
(psmcfa, VCF, true TMRCA tracks; `runs/*/sims/`), the bootstrap
replicates' `.psmc` files, HTML reports, logs and the runner's state
(`runs/_runner/`).

## 2) Software

- **Python** 3.13.13 with the exact packages of
  `experiment/pipeline/requirements.lock` (msprime 1.4.4, tskit 1.0.3,
  numpy 2.5.3, stdpopsim 0.3.0, scipy 1.18.1, pandas 3.0.6, matplotlib
  3.11.2, ...), and htslib 1.24 (`bgzip`, `tabix`, `bcftools`). The
  simulations are byte-identical only with these versions, which every
  truth record lists; `python -m experiment.pipeline tools` and `start`
  warn when the running interpreter differs.
- **psmc-rs 0.2.0**, built from the repository with `cargo build --release
  --locked` (Rust 1.88 or later; the production build used rustc 1.99.0,
  the release profile and no `RUSTFLAGS`), and psmc-rs 0.1 for E3, built
  from commit e50d69d by `tools --build-rs01`.
- **C psmc** (lh3/psmc 0.6.5-r74) with its `fq2psmcfa`, `splitfa` and
  `psmc_plot.pl`, `vcfutils.pl`, and `generate_multihetsep.py` of
  msmc-tools, in `experiment/.tools` (or `$PSMC_EXP_TOOLS`); section 7
  describes the builds.
- **Figures:** Arial (else Helvetica, else DejaVu Sans).

```bash
git clone https://github.com/larryivan/psmc-rs.git && cd psmc-rs
git checkout <MANUSCRIPT_TAG>
python3.13 -m venv experiment/.venv
experiment/.venv/bin/python -m pip install -r experiment/pipeline/requirements.lock
cargo build --release --locked
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools      # the tools and their identities; exits 1 if one is missing
```

## 3) Tables and figures from the committed results

No tool runs for these. Each experiment's report reads the committed
fits, bootstrap files, truth and QC records and writes its tables; the
figures are drawn from the tables:

```bash
$PY -m experiment.pipeline report E1     # likewise E2, E4, E5
$PY -m experiment.pipeline report E6     # S1 and S5; S2 and S3 need the simulated genomes (section 4)
$PY -m experiment.pipeline.figures       # every figure whose tables exist, into runs/<EXP>/figures/
```

`python -m experiment.pipeline.figures --list` names the tables each
figure reads. Fig 4 and Fig S4 read E3's tables.

E6's S2 compares the fits of E1's first replicates read as psmcfa, VCF and
multihetsep, and reads the simulated psmcfa to count the bins; S3 compares
the decoded TMRCA with the true one re-simulated from the replicate's
seeds. Both need the simulated genomes, which section 4 regenerates.

The analysis does not depend on the platform beyond its last digits.
The analysis code of the production commit (34f2b00), run on an Apple M1
Pro laptop (macOS, arm64) on the fits made on the Ibex cluster (Linux,
x86-64), wrote tables of E1, E2, E4 and E5 that agreed with those it had
written on Ibex to a relative 1e-13 in every value above 1e-3 in
magnitude (at most 9.3e-14), and to a relative 2.2e-11 in the
differences between psmc-rs and C psmc, of order 1e-5 (about 1e-15
absolute); the differences come from the two platforms' math libraries.
The tables shipped now come from later analysis code (the tools compared
interval by interval, E4's RMSE against the complete-data fit integrated
over log time), which reads the same fits.

## 4) The runs, on one machine

The pipeline is a job graph run by a detached, resumable runner (the
Runner section of `experiment/pipeline/README.md`). Into a fresh runs
directory, to reproduce everything from scratch:

```bash
RUNS=/path/to/new/runs
$PY -m experiment.pipeline plan                       # estimated CPU hours and wall time
$PY -m experiment.pipeline start --runs-dir $RUNS E1 E4 E5 E6 FIG
$PY -m experiment.pipeline status --runs-dir $RUNS    # done/running/failed/pending, ETA
$PY -m experiment.pipeline stop --runs-dir $RUNS      # any time; `start` again resumes
```

FIG draws the figures into `$RUNS` once their tables exist, and comparing
the tables of `$RUNS` with the committed ones shows what was reproduced.

`--cores` and `--mem-gb` set the runner's budget (default 7 cores and
12 GB). E2's bootstraps run on 50 threads each, so E2 needs `--cores 50`
and a machine with at least 50 cores; E3 needs the machine to itself
(`start E3 --mem-gb 15` on a quiet machine on AC power; see the pipeline
README's E3 section). On the development laptop (Apple M1 Pro, 8 cores,
16 GB), `plan` estimates E1 at 1.4 hours, E4 at 0.6, E5 at 2.3 and E6 at
0.1 (wall time alone on 7 cores). Section 5 gives the costs measured in
the production run.

In the repository's own `experiment/runs`, where the results are
committed, a job counts as done when its outputs exist and its provenance
record matches: the same command, settings, seeds, input checksums and tool
identities (section 6). Inputs count by content, so a simulation that
reproduces its recorded checksums leaves the jobs that read it done. With
the production's tools, `start E1 E6` would therefore remake only the
simulated genomes and S3's true TMRCA track, which E6's S2 and S3 read, and
`report E6` then writes their tables. With other builds of the tools (for
example psmc-rs built by another rustc, or another C psmc binary) their
fits count as not done and run again; `status --jobs` lists what `start`
would run, and `start --only REGEX` restricts it to the matching jobs.

## 5) The runs, on a SLURM cluster

The production runs of E1, E2, E4, E5 and E6 were made with the scripts of
`experiment/slurm/` on KAUST's Ibex cluster, and the figures drawn on the
laptop from the runs directory copied back. The scripts run the same
pipeline in SLURM allocations; site settings (account, partition, node
constraints, paths, C psmc's flags) go in `experiment/slurm/site.env`,
copied from `site.env.example`. Every job sources `env.sh`, which gives it
one environment whatever shell submitted it: the pinned Rust toolchain
without build flags, the Python environment with its htslib first on
`PATH`, the tools directory, and a machine lock in the job's own `/tmp`.

```bash
cd $PROD/experiment/slurm          # $PROD: a dedicated worktree of the production commit
./submit.sh build                  # build psmc-rs 0.2 and 0.1 and C psmc, run every Rust and Python test
./submit.sh prod                   # E1 E2 E4 E5 E6 in one allocation; submit again if the time ran out
./submit.sh e3 1 2 3 4 5           # E3: one exclusive node of one CPU model per repeat
./merge_e3.sh                      # once every e3 job has ended
./submit.sh e3-report              # E3's tables
```

The `prod` job ran on one node with a runner budget of 62 cores and
180 GB; its 572 jobs took 1 hour 48 minutes and 81 CPU hours: E1 13.0,
E2 43.8 (the five bootstrap runs on 50 threads 40.3), E4 5.2, E5 19.1 and
E6 0.3 (from the jobs' provenance records). Its jobs shared the node with
one another (1-minute load averages of 5 to 72 when they started), so
these are costs, not timings; E3 measures performance (section 10).

## 6) Provenance

Every job writes `runs/<EXP>/provenance/<job>.json` when it ends. Its key
holds what determines the outputs:

- the exact command line, with paths written as `<repo>`, `<runs>`,
  `<tools>` and `<python>`, so that the record stays valid in another
  clone;
- the settings and seeds; every seed is
  `config.seed(experiment, model, replicate, stage)`, the first 8 bytes of
  SHA-256 of `psmc-rs-manuscript/<experiment>/<model>/<replicate>/<stage>`
  modulo 2^31 − 1, plus 1, fixed before any result was seen and listed in
  each experiment's `*_seeds.csv` table;
- the SHA-256 of every input;
- the identities of the tools: for psmc-rs 0.2, its version, the git tree
  and blob ids of `src/`, `Cargo.toml` and `Cargo.lock`, a digest of
  uncommitted changes, `rustc -vV`, the Rust build flags and the target
  features they enable; for C psmc, its version and the SHA-256 of the
  binary, plus the compiler and CFLAGS of a recorded build; for Python
  steps, the SHA-256 of their modules and the package versions.

The record then adds the SHA-256 of every output and how the job ran: host,
SLURM job, cores, wall, user and system time, peak resident memory, load
average and the SHA-256 of the executables it ran. In the production run
every psmc-rs 0.2 fit was made from the same sources (the `src/` tree
that commits 0c84f05 to 34f2b00 share), by rustc 1.99.0 without build
flags, and the JSON of every fit and every bootstrap run's main fit
records the HMM kernels that ran (`simd`: `avx2+fma` in all 253; section
8).

## 7) The C psmc builds

C psmc is lh3/psmc 0.6.5-r74. `python -m experiment.pipeline tools
--build-c` builds it, `fq2psmcfa` and `splitfa` from the sources in
`experiment/.tools/psmc_c` with `$CC` and `$PSMC_C_CFLAGS` (default: the
Makefile's `-g -Wall -O2 -Wno-unused-function`) and records the compiler,
flags, commands and checksums in `psmc_c/BUILD.json`, which adds the
compiler and the flags to C psmc's identity.

- **Production (Ibex).** gcc 11.4.1 with `-g -Wall -O3 -march=x86-64-v3
  -ffp-contract=off -Wno-unused-function`, as every C psmc provenance
  record of E1 and E2 holds. Its output was byte-identical to the
  Makefile's `-O2` build on AMD Zen 2, Zen 5 and Intel Cascade Lake nodes,
  and it ran 1.2 to 2.0 times as fast; `-ffp-contract=off` keeps gcc from
  fusing multiply-adds, which would change the results. These are the C
  psmc results of the manuscript.
- **Laptop (macOS, arm64).** An `-O2` build of C psmc segfaults there, so
  the laptop's C psmc was built by hand with `-O1`. It has no `BUILD.json`:
  it is identified by its version and checksum, and its compiler and flags
  are in no provenance record.

## 8) Determinism

**Threads.** psmc-rs writes the same bytes (JSON, `.psmc`, TMRCA and
bootstrap files) with any `--threads` and `--batch-size`. The E-step and
the log-likelihood merge their work items in a fixed order, so their sums
do not depend on how the work is split; bootstrap replicate i draws its
trunks from its own ChaCha8 stream, seeded from `--bootstrap-seed` and i,
so a seed gives the same replicates with any thread count and on any
platform. The tests check this (`tests/estep_determinism.rs`,
`tests/cli_outputs.rs`, `tests/bootstrap_driver.rs`,
`tests/bootstrap_resample.rs`), and so did the production run: the main
fit of every E2 bootstrap run, on 50 threads, equals the genome's own fit
on one thread in every parameter, log-likelihood and iteration count
(`E2_bootstrap.csv`, `main_identical_to_fit`, largest difference 0), and
the main fit of every E5 bootstrap run equals the genome's fit (20 of 20,
`E5_datasets.csv`).

**Run-time HMM kernels.** For models of up to 256 states the forward and
backward steps take O(states) time per bin, from the rank-one triangles of
PSMC's transition matrix, as Harris, Sheehan, Kamm and Song (2014, RECOMB,
LNCS 8394:100–114) used them to decode coalescent HMMs in linear time. On
x86-64 the release binary picks its kernels when it starts: code compiled
for AVX2 and FMA on CPUs that have both, the portable code otherwise (or
with `PSMC_RS_SIMD=portable`); aarch64 runs NEON code. The output JSON's
`simd` and the HTML report name the kernels that ran.

- The AVX2 and FMA kernels fuse the multiply-adds of the O(states) forward
  and backward products, as the NEON kernels do; no other x86-64 kernel
  fuses, and no sum changes its order with the vector width. Their results
  are identical, to the bit, to those of a build for `x86-64-v3`, and from
  one CPU with AVX2 and FMA to another as long as the system math library
  gives the same `log` and `exp` on both (glibc picks a variant of them by
  CPU).
- The portable kernels do not fuse those multiply-adds, so their results
  differ in the last digits: by a relative 1e-14 or so in an E-step, which
  the M-step, stopping on a flat optimum, turns into about 1e-7 to 1e-6 in
  the fitted parameters after 5 EM iterations.
  `tests/estep_determinism.rs` checks that the two agree to a relative
  1e-12 in an E-step, and `tests/cli_outputs.rs` that their fits agree
  closely.
- Every psmc-rs fit of the production run used the AVX2 and FMA kernels.

**Simulations.** Each genome is simulated from its seeds with the exact
package versions of the lock file, and its truth record holds the
checksums of every chromosome. The provenance records hold the SHA-256 of
every simulated file, so a regenerated genome that differs re-keys every
job that reads it instead of passing silently, and the true TMRCA track of
S3, simulated again from the seeds, must reproduce the recorded checksum
of the genome's heterozygous positions.

**C psmc** is single-threaded and deterministic for one binary; how much
two builds of it differ is part of section 9.

## 9) Cross-platform comparison — PLACEHOLDER

> **To be written.** This section will compare the production results
> computed on Ibex (x86-64, Linux) with the same jobs run on the laptop
> (Apple M1 Pro, arm64, macOS): the simulated inputs and truth records,
> psmc-rs's fits and bootstrap bands, and C psmc's fits from its two
> builds, against the pre-registered agreement tiers, with the reference
> used for each tier and how sensitive the result is to it. It will name
> both platforms' CPUs, operating systems and toolchains, and the psmc-rs
> source revision each platform built. The analysis is not final.

## 10) Performance (E3) — PLACEHOLDER

> **To be written once E3 has run.** E3 times C psmc, psmc-rs 0.1 and
> psmc-rs 0.2 on constant-model genomes of 100 to 3000 Mb (25 EM
> iterations, 5 repeats, every run alone on an exclusive node of one CPU
> model), and psmc-rs 0.2's thread scaling and determinism on 3000 Mb
> (Fig 4, Table 2, Fig S4). This section will give the machines (CPU model,
> governor, boost), the builds (`E3_environment.csv`, C psmc's
> `BUILD.json`), the timing protocol and which runs it rejected, and how
> to reproduce the timings with `experiment/slurm` (`submit.sh e3`,
> `merge_e3.sh`, `submit.sh e3-report`) or on one machine (`start E3`).
>
> The protocol (`config.TimingProtocol`, `experiments/e3_run.py`): before
> each run the whole machine must be quiet, at most 1 core and 10% of all
> cores busy (all but idle and iowait time, so processes, kernel threads,
> interrupts and steal), since nothing of the experiment runs then.
> During the run, the other user-space processes (all but the tool's and
> the timing wrapper, each one's CPU time read just before and right after
> the run on Linux, with that of the children it reaped meanwhile) and
> steal time together may keep at most 0.5 core and 5% of all cores busy;
> otherwise the run is discarded and repeated later. A process that ended
> during the run, short-lived or not, is counted through the process that
> reaped it, its parent or init (one that had started before the run with
> its earlier CPU time too, which errs towards rejecting runs); only a
> process that nothing waits for is not measured, and its CPU appears
> only in the machine-wide time that no measured process explains, which
> each record keeps but the protocol does not limit, since it also holds
> accounting differences between the machine's and the processes' CPU
> time. Kernel threads and interrupt time are recorded but not limited
> during the run: part of them is the kernel's work for the timed tool
> itself (on the 128-core nodes, a run keeping 30 cores busy adds about
> 0.1 core to an idle node's 0.35 to 0.49 busy core), the rest
> housekeeping the quiet check bounds, and neither is interference from
> another workload. Where not every process, or the CPU time of reaped
> children, can be read (macOS), the machine's busy CPU time but the
> tool's is limited instead; each timing record names the measure used.

## 11) Release archive integrity

Each release archive is published with its SHA-256 checksum in
`<archive>.sha256`, and `SHA256SUMS.txt` lists the checksums of all
archives. Check a downloaded `.tar.gz` archive, in the directory it was
saved to, against either file:

```bash
shasum -a 256 -c psmc-rs-<tag>-<target>.tar.gz.sha256   # or: sha256sum -c ...
shasum -a 256 -c --ignore-missing SHA256SUMS.txt        # the archives present
```

On Windows, compare the output of `Get-FileHash <archive>.zip` with the
value in `<archive>.zip.sha256`.
