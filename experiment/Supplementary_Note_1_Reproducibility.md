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

The experiments are E1 (simulated accuracy and agreement with C psmc:
Fig 1, Table 1, Table S1, Fig S1), E2 (four real genomes: Fig 1, Table 1),
E3 (performance: Fig 2, Table 2), E4 (partly callable bins: Fig S2), E5
(bootstrap calibration: Fig S3), E6 (supplementary analyses, quoted in the
text) and E7 (the 1000 Genomes 30x genomes: Fig 3, Fig 1, Fig 2c, Table 1); FIG
draws every figure. The pipeline names most figure files after an earlier
plan of the article, and draws some that the article leaves out; section
3 maps them.

## 1) What the repository holds

Each experiment writes `experiment/runs/<EXP>/`. The repository holds the
production runs of E1 to E7 and FIG:

| item | where |
| --- | --- |
| real genomes (inputs of E2 and E6's S5): the three turtles, and NA12878's chr1-22, E7's psmcfa of that sample byte for byte (its SHA-256 is in E7's `MANIFEST.json` and `config.RealGenome.sha256`) | `experiment/real_data/{HLemySub1,HLhydTec1,HLpelCas1}_diploid.psmcfa`, `experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz` (`Papuan_highlands_diploid.psmcfa`, a chromosome 1, serves the README's benchmarks and the Rust tests; E2 does not read it) |
| truth records of the simulated genomes (settings, seeds, software versions, per-chromosome counts and checksums, the true Ne(t)) | `runs/<EXP>/truth/` |
| every fit: psmc-rs JSON and `.psmc`, C psmc `.psmc` | `runs/<EXP>/fits/` |
| bootstrap runs: main fit, every replicate's JSON, `summary.json` and `summary.tsv` | `runs/E2/bootstrap/`, `runs/E5/boot/`, `runs/E7/boot/` |
| E4's masks, bins and cross-checks; QC reports | `runs/E4/{masks,bins,consensus,crosscheck}/`, `runs/{E2,E6}/qc/` |
| one provenance record per job | `runs/<EXP>/provenance/` |
| tables and figures | `runs/<EXP>/tables/`, `runs/<EXP>/figures/` (the software's schematic, not in the article: `runs/FIG/`) |
| E7's inputs: one psmcfa per 1000 Genomes sample (the strict mask applied, non-PASS positions missing), the default set's het-or-missing VCFs and the masks | Zenodo, not the repository (`runs/E7/inputs/`; remade by `python -m experiment.pipeline.kg1000`) |
| E7's data stage: `MANIFEST.json` (sources, tool versions, commands, checks, SHA-256 of every file it wrote) and its records (how each chromosome was read, its chunks, the chr21 pilot) | `runs/E7/inputs/MANIFEST.json`, `runs/E7/inputs/records/` |

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
  the release profile and no `RUSTFLAGS`).
- **C psmc** (lh3/psmc 0.6.5-r74) with its `fq2psmcfa` and `splitfa`,
  and `generate_multihetsep.py` of msmc-tools, in `experiment/.tools` (or
  `$PSMC_EXP_TOOLS`); section 7 describes the builds.
- **Figures:** Arial (else Helvetica, else DejaVu Sans).

```bash
git clone https://github.com/larryivan/psmc-rs.git && cd psmc-rs
git checkout <MANUSCRIPT_TAG>
python3.13 -m venv experiment/.venv
experiment/.venv/bin/python -m pip install -r experiment/pipeline/requirements.lock
cargo build --release --locked
T=experiment/.tools
git clone https://github.com/lh3/psmc $T/psmc_c && git -C $T/psmc_c checkout 82fa2ad
git clone https://github.com/stschiff/msmc-tools $T/msmc-tools
curl -fsSLo $T/vcfutils.pl https://raw.githubusercontent.com/samtools/bcftools/1.24/misc/vcfutils.pl
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools --build-c   # builds C psmc; exits 1 if a tool is missing
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
figure reads. The article's figures and tables are these files of the
runs (`manuscript/make_numbers.py` writes the tables' LaTeX):

| article | file |
| --- | --- |
| Fig 1 | `runs/FIG/figures/art1_agreement` (from E1, E2 and E7's tables) |
| Fig 2 | `runs/E3/figures/art2_performance` (from E3 and E7's tables) |
| Fig 3 | `runs/E7/figures/fig7_1000_genomes` |
| Table 1 | from `E1_rs_vs_c`, `E1_replicates`, `E2_rs_vs_c`, `E2_fit_summary`, `E7_rs_vs_c` and `E7_fits` |
| Table 2 | `runs/E3/tables/E3_table2.csv` |
| Fig S1 | `runs/E1/figures/figS1_em_convergence` |
| Fig S2 | `runs/E4/figures/fig5_input_preparation` |
| Fig S3 | `runs/E5/figures/fig6_bootstrap_calibration` |
| Table S1 | `runs/E1/tables/E1_table1.csv` |

The pipeline also draws figures that the article leaves out, for the
analyses it quotes in its text: the software's schematic
(`runs/FIG/figures/fig1_workflow`), E1's and E2's fits in more detail
(`runs/E1/figures/fig2_simulated_accuracy`,
`runs/E2/figures/fig3_real_genomes`), E3's timings at every size and
thread count of section 10 and its thread scaling
(`runs/E3/figures/fig4_performance`, `figS4_thread_scaling`), E6's format consistency and TMRCA decoding
(`runs/E6/figures/figS2_format_consistency`, `figS3_tmrca_decoding`),
E2's split-first-window check (`runs/E2/figures/figS_turtle_patterns`)
and E7's populations and checks
(`runs/E7/figures/figS5_1000g_populations`, `figS6_1000g_checks`). The
numbers in the article's text and tables, and the counts in Fig 1, come
from the same tables through `manuscript/make_numbers.py`.

E6's S2 compares the fits of E1's first replicates read as psmcfa, VCF and
multihetsep, and reads the simulated psmcfa to count the bins; S3 compares
the decoded TMRCA with the true one re-simulated from the replicate's
seeds. Both need the simulated genomes, which section 4 regenerates.

The analysis does not depend on the platform beyond its last digits.
The analysis code of the production commit (34f2b00), run on an Apple M1
Pro laptop (macOS, arm64) on the fits made on the Ibex cluster (Linux,
x86-64), wrote tables of E1, E2, E4 and E5 (E2 then with the Papuan
chromosome 1 in place of NA12878) that agreed with those it had
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
directory, to reproduce E1, E4, E5 and E6 from scratch (E2 and E3 below;
E7 needs its inputs, section 1):

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
(`start E3` on a quiet machine on AC power; see the pipeline README's E3
section). On the development laptop (Apple M1 Pro, 8 cores,
16 GB), `plan` estimates E1 at 1.4 hours, E4 at 0.6, E5 at 2.3 and E6 at
0.1 (wall time alone on 7 cores). Section 5 gives the costs measured in
the production run.

In the repository's own `experiment/runs`, where the results are
committed, a job counts as done when its outputs exist and its provenance
record matches: the same command, settings, seeds, input checksums and tool
identities (section 6). Inputs count by content, so a simulation that
reproduces its recorded checksums leaves the jobs that read it done. With
the production's tools, `start E1 E6` would therefore remake only what is
not committed: the simulated genomes, S2's conversions and S3's true and
decoded TMRCA tracks. `report E6` then writes their tables. With other builds of the tools (for
example psmc-rs built by another rustc, or another C psmc binary) their
fits count as not done and run again; `status --jobs` lists what `start`
would run, and `start --only REGEX` restricts it to the matching jobs.

## 5) The runs, on a SLURM cluster

The production runs of E1 to E7 were made with the scripts of
`experiment/slurm/` on KAUST's Ibex cluster (E3's repeats as `e3` jobs,
section 10; E7's data stage as `e7-prep`), and the figures drawn on the
laptop from the runs directory copied back. The scripts run the same pipeline in SLURM allocations;
site settings (account, partition, node constraints, paths, C psmc's
flags) go in `experiment/slurm/site.env`, copied from
`site.env.example`. Every job sources `env.sh`, which gives it
one environment whatever shell submitted it: the pinned Rust toolchain
without build flags, the Python environment with its htslib first on
`PATH`, the tools directory, and a machine lock in the job's own `/tmp`.

```bash
cd $PROD/experiment/slurm          # $PROD: a dedicated worktree of the production commit
./submit.sh build                  # build psmc-rs 0.2 and C psmc, run every Rust and Python test
./submit.sh prod                   # E1 E2 E4 E5 E6 in one allocation; submit again if the time ran out
./submit.sh e3 1 2 3 4 5           # E3: one exclusive node of one CPU model per repeat
./merge_e3.sh                      # once every e3 job has ended
./submit.sh e3-report              # E3's tables
./submit.sh e7-prep                # E7's inputs from the 1000 Genomes VCFs; then prod again for E7's fits
```

The production ran in two `prod` jobs, each on one AMD EPYC 7702 node
(128 cores). The first (SLURM job 53386770, commit 34f2b00, a runner
budget of 62 cores and 180 GB; 1 hour 48 minutes) made E1, E4, E5, E6's
S1 to S3 and the turtles' C psmc fits and QC. The second (53452276, commit
d338500, 94 cores and 324 GB; 8 hours 11 minutes) made, once NA12878 had
replaced the Papuan chromosome 1 in E2 and the turtles had their own
scaling, the rest of E2 (NA12878's fits, bootstrap and QC, the turtles'
psmc-rs fits and bootstraps, the six split-first-window fits), E6's S5 QC
of NA12878 and E7's 2,637 fits and bootstraps; a short job (53467834)
wrote the reports. From the jobs' provenance records, in CPU hours (user
and system time):

| experiment | jobs | CPU hours | made in |
| --- | --- | --- | --- |
| E1 | 121 | 13.0 | 53386770 (report: 53467834) |
| E2 | 23 | 59.8 | 6 in 53386770, 16 in 53452276 (report: 53467834) |
| E3 | 100 | 15.1 | five exclusive nodes, 53420812 to 53420816 (section 10; report: 53467842) |
| E4 | 286 | 5.2 | 53386770 (report: 53467834) |
| E5 | 121 | 19.1 | 53386770 (report: 53467834) |
| E6 | 26 | 0.3 | 53386770 and 53452276 (reports: 53467834) |
| E7 | 2,639 | 642.9 | 53452276 (GIAB comparison and report: 53467834) |

The fits of each `prod` job shared their node with one another (1-minute
load averages of 5 to 72 and of 33 to 133 when they started), so these
are costs, not timings; E3 measures performance (section 10). E7's data
stage ran before the second job, as SLURM jobs of `kg1000`'s steps, and
read 1.68 TB of VCFs (each step's host and job are in
`runs/E7/inputs/records/`, the sources and checks in `MANIFEST.json`).

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
average and the SHA-256 of the executables it ran. Every psmc-rs 0.2 job
of the production, E1 to E7 (3,009 records), ran a build of the same
sources (the `src/` tree 4c0d777, `Cargo.toml` 7ff6ee3 and `Cargo.lock`
b552a2a, which commits 0c84f05 to d338500 share, without uncommitted
changes), made by rustc 1.99.0 without build flags, and the JSON of every
fit and of every bootstrap run's main fit records the HMM kernels that ran
(`simd`: `avx2+fma` in all 2,898; section 8).

## 7) The C psmc builds

C psmc is lh3/psmc 0.6.5-r74: the sources in `experiment/.tools/psmc_c`,
from which every C psmc binary of this note was built, are identical, file
for file, to lh3/psmc at commit 82fa2ad (2025-11-16), whose `cli.c` gives
the version string `0.6.5-r74-dirty` that the provenance records hold.
`python -m experiment.pipeline tools
--build-c` builds it, `fq2psmcfa` and `splitfa` from the sources in
`experiment/.tools/psmc_c` with `$CC` and `$PSMC_C_CFLAGS` (default: the
Makefile's `-g -Wall -O2 -Wno-unused-function`; `-O1` on macOS arm64,
where `-O2` segfaults) and records the compiler,
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
`E5_datasets.csv`), as does that of each of E7's three (`E7_bootstrap.csv`).
NA12878's bootstrap ran twice with one seed, in E2 on 50 threads and in E7
on 32: the two runs wrote the same 103 files byte for byte
(`runs/E2/bootstrap/NA12878/trunk500000/` and `runs/E7/boot/NA12878/`).

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

## 9) Cross-platform comparison

The production ran on Ibex (x86-64). To measure how much its results
depend on the platform, the 40 E1 genomes and NA12878 were fitted again
with psmc-rs on the development laptop, with the production's commands
(only the paths changed), and the laptop's C psmc fits of the same 40
genomes, from an earlier laptop run, were compared with Ibex's. The
scripts, tables and laptop fits are in `experiment/crossplatform/`, whose
`summary.md` gives every number below.

| | laptop | Ibex (production) |
| --- | --- | --- |
| CPU and system | Apple M1 Pro (arm64), macOS 27.0, Apple's math library | AMD EPYC 7702 (x86-64), Linux, glibc 2.34 |
| psmc-rs | the same sources (`src/` 4c0d777, no uncommitted changes), rustc 1.97.1, release profile, no build flags; NEON kernels | rustc 1.99.0, release profile, no build flags; AVX2 and FMA kernels |
| C psmc | lh3/psmc 0.6.5-r74, the laptop's `-O1` build by Apple clang, which fuses multiply-adds (identified by its machine code; it has no build record) | gcc 11.4.1, `-O3 -march=x86-64-v3 -ffp-contract=off` (section 7) |

**Inputs.** The laptop's 40 E1 genomes were byte-identical to Ibex's
(psmcfa, VCF and index, 40 of 40; the heterozygous positions and counts of
all 400 chromosomes), and both platforms fitted the same NA12878 file.

**Outputs.** No output was byte-identical across the platforms, for either
program (psmc-rs 0 of 41, C psmc 0 of 40), but psmc-rs agreed with itself
far more closely than the two builds of C psmc agreed with each other
(minimum / median / maximum over the 40 E1 fits; the last column is the
difference between the programs on Ibex, as in E1):

| | psmc-rs, laptop vs Ibex | C psmc, laptop vs Ibex | psmc-rs vs C psmc, Ibex |
| --- | --- | --- | --- |
| theta, relative | 1.6e-9 / 1.8e-7 / 7.6e-7 | 1.5e-7 / 5.8e-6 / 5.3e-5 | 1.2e-6 / 6.3e-5 / 3.8e-4 |
| largest relative difference of any parameter | 5.5e-8 / 7.9e-7 / 9.6e-6 | 3.0e-5 / 1.1e-4 / 0.21 | 6.0e-5 / 2.2e-4 / 0.99 |
| RMSE of log10 Ne over the window | 2.2e-9 / 1.2e-7 / 1.0e-6 | 3.6e-6 / 8.5e-6 / 2.4e-5 | 3.7e-6 / 2.5e-5 / 2.0e-4 |
| largest difference of log10 Ne, any interval | 1.2e-8 / 3.1e-7 / 4.1e-6 | 1.4e-5 / 4.9e-5 / 0.10 | 1.0e-5 / 7.9e-5 / 2.0 |
| final log-likelihood, relative | 4.6e-14 / 2.1e-12 / 2.8e-11 | 4.9e-12 / 1.7e-10 / 1.2e-9 | 1.5e-10 / 3.3e-9 / 1.3e-8 |

C psmc's 0.21 and 0.10 are expansion replicate 5's first lambda, which
runs away outside the window in both builds (without that fit, 5.4e-4 and
2.4e-4). Replicate by replicate, psmc-rs's RMSE between the platforms was
smaller than C psmc's between its builds in 40 of 40 (by a median factor
of 80, at least 7.2), and smaller than the difference between the
programs in 40 of 40 (median 196, at least 45). NA12878's two psmc-rs
fits agreed to an RMSE of 9.1e-8 and within a relative 6.6e-7 in every
parameter.

**Where the runs part.** psmc-rs's log-likelihood at the starting point
was bit-identical in 40 of the 41 fits (the other differed in the last
place), as section 8 leads one to expect: the NEON and the AVX2 and FMA
kernels fuse the same multiply-adds, and no sum changes its order with the
vector width. The first differences appear in iterations 1 to 3. What
still differs is the system math library (Apple's `exp` and `log` against
glibc's), whose last-bit differences the M-step, stopping on a flat
optimum, turns into the parameter differences above; this comparison does
not separate the math library from other architecture-specific code. C
psmc's builds part at the first M-step's evaluation count in every fit;
they differ in compiler, optimization level, floating-point contraction
and math library, which this comparison cannot separate.

**Controls.** The same sources built on the laptop by Ibex's rustc 1.99.0
wrote the same bytes as the rustc 1.97.1 build in 41 of 41 fits, and one
laptop binary run twice wrote the same bytes (2 of 2): the differences are
the platform's, not the compiler version's, nor noise from run to run.

**Agreement tiers.** Tiers fixed on 7 October 2026, before these runs
(`experiment/crossplatform/scripts/tiers_2026-10-07.py`), were applied
unchanged: A, identical bytes; B, every parameter within a relative 1e-6,
log10 Ne within 1e-6 in every interval and the final log-likelihood within
a relative 1e-8; C, log10 Ne within 1e-3 in every interval and within a
fiftieth of the difference between the programs (the per-model mean RMSE
between psmc-rs and C psmc in E1); otherwise fail. psmc-rs's E1 fits were
B in 24, C in 11 and failed in 5 of 40, and NA12878's fit was B; C psmc's
builds failed in 40 of 40. All 16 psmc-rs fits that missed B did so on the
parameter limit, at the most recent or the oldest free lambdas, which the
data constrain least; the five that failed differed by 5e-7 to 4e-6 in
log10 Ne, 17 to 36 times less than the difference between the programs
rather than the required 50. The tier-C reference was not fixed in advance;
the other references considered, and two deviations from the tiers, are
reported in `experiment/crossplatform/summary.md` (section 5) and change
none of these conclusions.

**Caveats.** The laptop's C psmc build is identified by its checksum and
machine code, not by a build record, and a laptop build without
floating-point contraction was not run. Only E1 and NA12878 were refitted
on the laptop; E2's turtles, the bootstraps and E3 to E7 were not.

**Conclusion.** psmc-rs's results depend on the platform only in their
last digits: across two processor architectures, operating systems and
math libraries its curves agreed to an RMSE of at most 1.0e-6 in log10
Ne, two orders of magnitude below the difference between psmc-rs and C
psmc and below the difference between two builds of C psmc on these
platforms. Byte-identical results across platforms need the same kernels
and the same math library on both (section 8).

## 10) Performance (E3)

E3 times C psmc and psmc-rs on constant-model genomes of 100, 300, 1000
and 3000 Mb (1, 3, 10 and 30 chromosomes of 100 Mb, shared between the
sizes; `E3_seeds.csv`), each fit 25 EM iterations of the recipe
(`-N25 -t15 -r5 -p "4+25*2+4+6"`, 100 bp bins): C psmc on its one
thread, psmc-rs on 1, 8 and 30 threads, and psmc-rs on 3000 Mb also on
2, 4 and 16 threads, 5 repeats of each (Fig 2, Table 2, and the
pipeline's thread-scaling figure): 19 setups and 95 timed runs, all
measured (`E3_table2.csv`, `status`). The numbers below are computed from
E3's tables in `runs/E3/tables/`, named with each.

**Machines and builds.** Each repeat ran on its own exclusive Ibex node,
all five of one CPU model: AMD EPYC 7702, 128 cores, 503.7 GB of
memory, the cpufreq governor `performance` (acpi-cpufreq) with boost on,
Linux 5.14.0-427.20.1.el9_4.0.1.x86_64 and glibc 2.34
(`E3_environment.csv`, whose `clean_runs_on_mixed_machines` is False). Repeats 1 to 5 ran on
cn506-02-l, cn506-18-l, cn514-07-r, cn513-19-l and cn514-04-l, as SLURM
jobs 53420812 to 53420816 in that order (`E3_runs.csv`: `hostname`,
`slurm_job_id`); each repeat's 19 runs took 2.19 to 2.22 hours of wall
time (the sum of its `wall_seconds`).

| tool | build |
| --- | --- |
| C psmc | lh3/psmc 0.6.5-r74, built on 2026-10-07 by `tools --build-c` with gcc 11.4.1 (Red Hat 11.4.1-3) and CFLAGS `-g -Wall -O3 -march=x86-64-v3 -ffp-contract=off -Wno-unused-function` (its `BUILD.json`): the production build of section 7 |
| psmc-rs | 0.2.0, built with rustc 1.99.0, the release profile and no build flags from the `src/` tree 4c0d777 with `Cargo.toml` 7ff6ee3 and `Cargo.lock` b552a2a, and no uncommitted changes (`tool.rs02.source_tree`, `source_dirty`): the sources that commits 0c84f05 to 3ed8a64 share, from which the earlier production run's fits were made too (section 6); it picks its AVX2 and FMA kernels at run time (section 8) |

`E3_environment.csv` also holds the SHA-256 of every binary, the fit
settings and the protocol's limits.

The protocol (`config.TimingProtocol`, `experiments/e3_run.py`): before
each run the machine must be quiet for 2 s, with at most 1 core and 10% of
all cores busy (all but idle and iowait time). During the run, other
user-space processes (each counted with the children it reaped) and steal
time may keep at most 0.5 core and 5% of all cores busy; otherwise the run
is discarded and repeated later. Kernel threads and interrupts are
recorded but not limited, because part of their time is the kernel's work
for the tool: on the 128-core nodes, a run keeping 30 cores busy adds
about 0.1 core to an idle node's 0.35 to 0.49 busy core. On macOS, where
not every process can be read, the limit applies instead to all of the
machine's busy time but the tool's, and each record names the measure
used.

**Runs.** Before its runs, each `e3` job measured its idle node for 30 s,
to stop if the node was busier than the protocol allows; none stopped.
The protocol deferred runs while a node was busy and ran them again
later: the runner logs have 4 deferral lines in repeat 2 and 2 in
repeat 3 and none in repeats 1, 4 and 5, and every deferred run was clean
when it ran. `merge_e3.sh` merged the five repeats and found every run
done for the current code. All 95 runs (20 of C psmc and 75 of psmc-rs)
are clean (`E3_environment.csv`: `runs_not_quiet` and `runs_not_clean`
are 0). In the quiet check's 2 s window before them the node had 0.215 to
0.715 core busy, median 0.31 (the limit is 1; the five nodes' medians
0.285 to 0.32): in 60 runs less than the 0.35 to 0.49 busy core of an idle
node given above, in 30 within it and in 5 more. During them the other
processes kept a median 0.26 and at most 0.42 core busy (the limit is
0.5), at most 0.33% of all cores (the limit is 5%), and steal time was
0 (`E3_runs.csv`: `busy_before_cores`, `other_cores_during`,
`other_cpu_share_during`, `steal_cores_during`). Each median below is
taken over the five repeats, one per node, so the spread of the repeats
includes the spread between the nodes.

**Wall time** in seconds (`E3_table2.csv`: `wall_median_s`, and in
parentheses `wall_min_s` to `wall_max_s`); – marks a setup E3 does not
run:

| tool | threads | 100 Mb | 300 Mb | 1000 Mb | 3000 Mb |
| --- | --- | --- | --- | --- | --- |
| C psmc | 1 | 152.5 (151.4–153.7) | 450.8 (442.4–453.7) | 1,480.1 (1,466.0–1,499.2) | 4,445.9 (4,404.8–4,470.1) |
| psmc-rs | 1 | 17.25 (17.24–18.27) | 51.72 (51.14–52.25) | 174.6 (172.4–176.2) | 519.1 (517.7–522.5) |
| psmc-rs | 2 | – | – | – | 262.1 (261.7–264.9) |
| psmc-rs | 4 | – | – | – | 139.9 (139.6–141.0) |
| psmc-rs | 8 | 17.24 (17.23–17.77) | 18.27 (17.75–18.29) | 35.01 (34.97–35.03) | 70.57 (70.54–71.06) |
| psmc-rs | 16 | – | – | – | 36.04 (35.61–36.09) |
| psmc-rs | 30 | 17.25 (17.23–17.78) | 18.27 (17.78–18.32) | 18.30 (18.26–18.32) | 18.32 (18.31–18.84) |

The fastest and the slowest repeat were less than 3.5% of the median
apart in every setup but one, psmc-rs on 100 Mb on one thread (6.0%).

**Speed-ups** (`E3_table2.csv`, `speedup_vs_c`: C psmc's median wall time
on the same genome over the tool's). psmc-rs on one thread was 8.5 to
8.8 times as fast as C psmc at every size: 8.6 on 3000 Mb, 519.1 s
against 4,445.9 s, or 20.8 s against 177.8 s per EM iteration
(`seconds_per_iteration_median`). On 3000 Mb it was 63.0 times as fast on
8 threads, 123 on 16 and 243 on 30. Paired within each repeat, on one
node, C psmc's wall time over psmc-rs's on 3000 Mb was 8.51 to 8.60
on one thread, and 62.3 to 63.3, 122.2 to 125.5 and 237.2 to 243.7 on 8,
16 and 30 threads; on one thread it was 8.30 to 8.91 over the four sizes
(`E3_runs.csv`, `wall_seconds`). psmc-rs's
threads added less than 1% to its CPU time on 1000 and 3000 Mb and at
most 2.3% on 100 and 300 Mb: its median on 3000 Mb was 516.6 s on one
thread and 516.1 to 521.5 s on any number, and on 300 Mb 51.1 s on one
thread and 52.1 and 52.3 s on 8 and 30 (`cpu_median_s`).

**Peak memory** in MB of 2^20 bytes (`E3_table2.csv`,
`peak_rss_median_mb`: the median over the repeats of the larger of the
tool's `ru_maxrss` and the resident memory of its process tree sampled
every 0.5 s):

| tool | threads | 100 Mb | 300 Mb | 1000 Mb | 3000 Mb |
| --- | --- | --- | --- | --- | --- |
| C psmc | 1 | 1,035 | 1,037 | 1,044 | 1,062 |
| psmc-rs | 1 | 25.0\* | 25.0\* | 25.0\* | 35.8 |
| psmc-rs | 2 | – | – | – | 39.8 |
| psmc-rs | 4 | – | – | – | 46.2 |
| psmc-rs | 8 | 25.0\* | 25.0\* | 43.3 | 61.5 |
| psmc-rs | 16 | – | – | – | 93.9 |
| psmc-rs | 30 | 25.0\* | 25.0\* | 73.0 | 148.0 |

\* In these 35 runs the tool's `ru_maxrss` was 25.0 MB whatever the size
and the threads, and the sampled peak 7.0 to 20.2 MB (`E3_runs.csv`:
`maxrss_mb`, `peak_rss_tree_sampled_mb`). `ru_maxrss` is never below the
tool's peak and the sampled peak never above it, so these cells are an
upper bound: psmc-rs's own peak lies between the two. Where its peak
is larger (1000 Mb on 8 and 30 threads, 3000 Mb), `ru_maxrss` equals the
sampled peak or exceeds it by at most 1.0 MB.

C psmc held about 1 GB whatever the size. psmc-rs peaked at 35.8 MB
on 3000 Mb on one thread and at 148.0 MB on 30, where C psmc held 29.7
and 7.2 times as much (`c_rss_over_rss`).

**Thread scaling** of psmc-rs on 3000 Mb (`E3_thread_scaling.csv`;
speed-up: the one-thread median wall time over the median on t threads;
efficiency: the speed-up over t):

| threads | 1 | 2 | 4 | 8 | 16 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| median wall time (s) | 519.1 | 262.1 | 139.9 | 70.57 | 36.04 | 18.32 |
| speed-up | 1 | 1.98 | 3.71 | 7.36 | 14.40 | 28.33 |
| efficiency | 1 | 0.99 | 0.93 | 0.92 | 0.90 | 0.94 |
| peak memory (MB) | 35.8 | 39.8 | 46.2 | 61.5 | 93.9 | 148.0 |

The psmc-rs of these runs processed sequences in parallel, each on one
thread, so its wall time stops improving once the threads reach the
number of chromosomes: on 100 Mb (one chromosome) it took 17.25, 17.24 and
17.25 s on 1, 8 and 30 threads, on 300 Mb (three) 18.27 s on both 8 and
30, and every genome took 17.2 to 18.3 s on at least as many threads as
chromosomes, about the time of one chromosome on one thread. With
chromosomes of equal length, t threads need at least ⌈chromosomes/t⌉
times one chromosome's time, and the median wall time divided by
⌈chromosomes/t⌉ is 17.2 to 18.3 s in every psmc-rs setup
(`E3_table2.csv`: `wall_median_s`, `chromosomes`, `threads`). On the 30
chromosomes of 3000 Mb this caps the efficiency at 30/(t⌈30/t⌉): 1 on 2
and 30 threads and 0.94 on 4, 8 and 16, which the measured 0.93, 0.92 and
0.90 approach.

**Determinism** (`E3_determinism.csv`: every output, C psmc's `.psmc` and
psmc-rs's JSON, against its tool's output at the same size on the fewest
threads in repeat 1). C psmc's five repeats wrote the same bytes at every
size. psmc-rs wrote the same bytes at every size whatever the thread
count, the repeat and the node, one SHA-256 per size: 15 outputs on each
of 100, 300 and 1000 Mb (1, 8 and 30 threads) and 30 on 3000 Mb (1 to 30
threads).

**Reproducing the timings.** On a SLURM cluster, with the scripts of
section 5 and `E3_CONSTRAINT` in `site.env` naming one CPU model:

```bash
cd $PROD/experiment/slurm
./submit.sh e3 1 2 3 4 5           # one exclusive node of $E3_CONSTRAINT per repeat
./merge_e3.sh                      # once every e3 job has ended
./submit.sh e3-report              # E3's tables
```

A resubmitted repeat returns to the node of its first runs; `merge_e3.sh`
refuses a repeat timed on more than one host and shards whose simulations
differ, and checks that every run is done for the current code and tools.
On one machine, with nothing else running, on AC power:

```bash
$PY -m experiment.pipeline start --runs-dir $RUNS E3
$PY -m experiment.pipeline status --runs-dir $RUNS --jobs    # deferred runs say why
```

The 30-thread runs need 30 usable CPUs (`measure` refuses to time a run
on more threads than the CPUs it may use), and the spread of the repeats
then leaves out the spread between machines. FIG draws Fig 2 and the thread-scaling figure
from the tables (section 3).
