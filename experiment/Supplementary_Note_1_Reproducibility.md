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
| E3's timing records (one JSON for each of the 95 timed runs: wall and CPU time, peak memory, the machine's load before and during the run) | `runs/E3/timings/` |
| E4's masks, bins and cross-checks; QC reports (E7's: NA12878's heterozygous SNVs against GIAB) | `runs/E4/{masks,bins,consensus,crosscheck}/`, `runs/{E2,E6,E7}/qc/` |
| one provenance record per job | `runs/<EXP>/provenance/` |
| tables and figures | `runs/<EXP>/tables/`, `runs/<EXP>/figures/` (`runs/FIG/figures/`: the article's Fig 1, `art1_agreement`, and the software's schematic, `fig1_workflow`, which the article leaves out) |
| E7's inputs, every file its jobs read (about 21 GB): one psmcfa per 1000 Genomes sample (the strict mask applied, non-PASS positions missing), the default set's het-or-missing VCFs, the callable and non-PASS masks, the per-sample QC table and GIAB's NA12878 benchmark | to be deposited on Zenodo, not in the repository (`runs/E7/inputs/`; remade by `python -m experiment.pipeline.kg1000` in a runs directory without the stage's committed records, section 5) |
| E7's data stage: its sample table and chunk plan (`samples.tsv`, `chunks.tsv`, which E7's job graph reads), `MANIFEST.json` (sources, tool versions, commands, checks, SHA-256 of every file it wrote) and its records (how each chromosome was read, its chunks, the chr21 pilot as the stage ran it); the chr21 pilot and the default genomes' conversions redone with the production's psmc-rs | `runs/E7/inputs/{samples.tsv,chunks.tsv,MANIFEST.json}`, `runs/E7/inputs/records/`, `experiment/e7_pilot/` |
| the analyses outside the pipeline, each with its scripts, results and README: the cross-platform comparison (section 9) and the diagnostic of the residual difference from C psmc (variants of psmc-rs, as a patch, and their fits of E1's genomes) | `experiment/crossplatform/`, `experiment/c_equivalence/` |

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
  and `generate_multihetsep.py` of msmc-tools at commit 0d1d169
  (2023-06-27; its SHA-256, below, is the one the provenance records of
  E4's cross-checks hold), in `experiment/.tools` (or `$PSMC_EXP_TOOLS`);
  section 7 describes the builds.
- **Figures:** Arial (else Helvetica, else DejaVu Sans).

```bash
git clone https://github.com/larryivan/psmc-rs.git && cd psmc-rs
git checkout v0.2.0
python3.13 -m venv experiment/.venv
experiment/.venv/bin/python -m pip install -r experiment/pipeline/requirements.lock
cargo build --release --locked
T=experiment/.tools
git clone https://github.com/lh3/psmc $T/psmc_c && git -C $T/psmc_c checkout 82fa2ad
git clone https://github.com/stschiff/msmc-tools $T/msmc-tools && git -C $T/msmc-tools checkout 0d1d169
# generate_multihetsep.py's SHA-256: caaa87a07e0fe2dc7228f30c9aff759cf86e9f61b8332aabd41398399ea6331b
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools --build-c   # builds C psmc; exits 1 if a tool is missing
```

## 3) Tables and figures from the committed results

No tool runs for these. Each experiment's report reads the committed
fits, bootstrap files, truth and QC records and writes its tables; the
figures are drawn from the tables. Here they run on a copy of the
committed runs:

```bash
CHECK=/path/to/new/copy
cp -R experiment/runs $CHECK                            # the reports and figures rewrite their files
$PY -m experiment.pipeline report E1 --runs-dir $CHECK  # likewise E2, E4, E5
$PY -m experiment.pipeline report E6 --runs-dir $CHECK  # S1 and S5; S2 and S3 need the simulated genomes
$PY -m experiment.pipeline.figures --runs-dir $CHECK    # every figure whose tables exist
```

The reports write their tables into `$CHECK/<EXP>/tables/` and the
figures go into `$CHECK/<EXP>/figures/`. Neither updates the jobs'
provenance records (section 6), so run in `experiment/runs` itself on
another platform they would change committed tables in their last digits
(as on the laptop, the last paragraph of this section), and those reports
and the figures that read their tables would then count as not done.

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
The reports of E1, E2, E4 and E5, run on an Apple M1 Pro laptop (macOS,
arm64) on the committed fits, with the code that wrote the committed
tables on the Ibex cluster (Linux, x86-64) in the production of section 5
(the same code digests and package versions), wrote 16 of the 38 tables
byte for byte; the others agreed with those written on Ibex to a relative
2.5e-13 in every value of magnitude 1e-3 or more, and to a relative
1.9e-11 in the differences between psmc-rs and C psmc, of order 1e-5 (at
most 8.9e-16 absolute); the differences come from the two platforms' math
libraries (`experiment/crossplatform/summary.md`, section 4.6).

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

`prod` also declares E7's jobs, which wait, blocked, until E7's inputs
(section 1) are in `runs/E7/inputs/`, copied there or written by
`e7-prep`; the next `prod` then runs them. The repository commits the data
stage's `records/`, and the stage skips every step whose record exists, so
`e7-prep` converts nothing until `runs/E7/inputs/records/` is removed in
`$PROD` (with `MANIFEST.json`, by which the runner takes the stage as
done). Run by hand, `python -m experiment.pipeline.kg1000` needs the same,
or a new runs directory (`--runs-dir`).

The production ran in two `prod` jobs of commit 4b2c636. The first
(SLURM job 54256959, on node cn513-11-l on 22 and 23 October 2026; 5 hours
53 minutes) ran every job whose key the code had changed: the psmc-rs
fits, bootstraps, conversions, decoding and QC of E1, E2 and E4 to E6,
E4's cross-checks, and the reports of E1, E2, E4 and E5. The second
(54555780, on node cn514-10-l on 25 and 26 October) ran every fit and
bootstrap of E7, C psmc's and psmc-rs's together, so that their times
compare like for like, and E7's report. The jobs whose keys had not
changed were kept from earlier jobs: the C psmc fits of E1 and the
turtles (53386770, commit 34f2b00), that of NA12878 (53452276, commit
d338500), and the simulations, E4's masks, consensus calls and bins, E6's
reports and S3's true TMRCA and E7's GIAB comparison (54211974 and
54232546, two `prod` jobs of 22 October). From the jobs' provenance
records, in CPU hours (user and system time):

| experiment | jobs | CPU hours | made in |
| --- | --- | --- | --- |
| E1 | 121 | 12.3 | 41 in 54256959; C psmc fits 53386770; simulations 54211974 |
| E2 | 23 | 36.1 | 19 in 54256959; C psmc fits 53386770 (turtles) and 53452276 (NA12878) |
| E3 | 100 | 13.0 (12.3 the tools' own in the 95 timed runs, `cpu_seconds` in `E3_runs.csv`; 0.47 the `measure` wrapper around them; 0.19 the simulations) | five exclusive nodes, 54256961 to 54256965 (section 10; report: 54528410) |
| E4 | 286 | 2.7 | 206 in 54256959; simulations, masks, consensus calls and bins (80) 54232546 |
| E5 | 121 | 11.2 | 71 in 54256959; simulations 54211974 (13) and 54232546 (37) |
| E6 | 26 | 0.2 | 21 in 54256959; S3's truth 54211974; reports 54232546 |
| E7 | 2,639 | 418.1 | 2,638 in 54555780; GIAB comparison 54232546 |

The fits of each `prod` job shared their node with one another (1-minute
load averages of 14 to 81 in 54256959 and 63 to 136 in 54555780, and of 5
to 71 and 37 for the C psmc fits of 53386770 and 53452276, when they
started), so these are costs, not timings, though E7's two programs,
sharing one node in one job, compare with each other like for like; E3
measures performance (section 10). E7's data stage ran on 7 and 8 October
2026, as SLURM jobs of `kg1000`'s steps, and read 1.68 TB of VCFs (each
step's host and job are in `runs/E7/inputs/records/`, the sources and
checks in `MANIFEST.json`). Its conversions ran psmc-rs from `src/`
4c0d777 (`MANIFEST.json`); the production's psmc-rs writes the same
content: converting with the stage's command, chromosome by chromosome,
the per-sample VCFs the stage kept (those of the 52 default genomes) gave
1,144 of 1,144 chromosome files and 52 of 52 genomes identical to E7's
after decompression, with equal QC; only the gzip bytes differ (Ibex job
54553271, an EPYC 7702 node; `experiment/e7_pilot/stage1.json`). The chr21
pilot, run again with the production's binary, gave the same files from
the joint VCF directly as through the per-sample VCFs, both identical to
E7's after decompression (`experiment/e7_pilot/pilot_chr21.json`).

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
sources (the `src/` tree dcf29ca, `Cargo.toml` c0be5cf and `Cargo.lock`
ae10354, which commit 3969ebb and every later commit up to the tag
`v0.2.0` share, without uncommitted changes), made by rustc 1.99.0
without build flags, and the JSON of every fit and of every bootstrap
run's main fit records the HMM kernels that ran (`simd`: `avx2+fma` in all
2,918; section 8).

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
  record of E1, E2, E3 and E7 holds. Its output was byte-identical to the
  Makefile's `-O2` build on AMD Zen 2, Zen 5 and Intel Cascade Lake nodes,
  and it ran 1.2 to 2.0 times as fast (one fit of HLpelCas1 on each;
  `experiment/crossplatform/x86_cpus/`, section "C psmc's production
  build"); `-ffp-contract=off` keeps gcc from fusing multiply-adds, which
  would change the results. These are the C psmc results of the
  manuscript.
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
LNCS 8394:100–114) used them to decode coalescent HMMs in linear time. So
do the expected transition counts, for models of any size: within each
triangle a transition probability is a factor of its row times a factor
of its column, so C psmc's EM Q needs the counts only summed over each
triangle's rows and columns and over the diagonal, which the E-step adds
up from each bin's prefix and suffix sums (C's `QD` line, which needs
every count, is not in psmc-rs's `.psmc` output). On
x86-64 the release binary picks its kernels when it starts: code compiled
for AVX2 and FMA on CPUs that have both, the portable code otherwise (or
with `PSMC_RS_SIMD=portable`); aarch64 runs NEON code. The output JSON's
`simd` and the HTML report name the kernels that ran.

- The AVX2 and FMA kernels fuse the multiply-adds of the O(states) forward
  and backward products and transition sums, as the NEON kernels do; no
  other x86-64 kernel fuses, and no sum changes its order with the vector
  width. Their results
  are identical, to the bit, to those of a build for `x86-64-v3`, and from
  one CPU with AVX2 and FMA to another as long as the system math library
  gives the same `log` and `exp` on both (glibc picks a variant of them by
  CPU). The production's binary refitted E1's bottleneck replicate 1 byte
  for byte on AMD EPYC 9655 (Zen 5), Intel Xeon Gold 6248 (Cascade Lake)
  and Gold 6148 (Skylake) nodes, with glibc 2.34 as on the production's
  EPYC 7702 nodes (`experiment/crossplatform/x86_cpus/`).
- The portable kernels do not fuse those multiply-adds, so their results
  differ in the last digits: by a relative 1e-14 or so in an E-step (2.2e-14
  at most on the test inputs), which the M-step, stopping on a flat
  optimum, turns into about 1e-8 to 1e-6 in the fitted parameters after 3
  to 5 EM iterations (on an EPYC 7702: 7.8e-9 after 3 and 1.4e-8 after 5 on
  the 60,000-bin test input, 1.1e-8 after 5 on the 180,000-bin one, and
  8.6e-7 on the latter with the pattern `4+25*2+4+6`).
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

The production of section 5 ran on Ibex (x86-64). To measure how much its
results depend on the platform, its 40 E1 genomes and NA12878 were fitted
again with psmc-rs on the development laptop, built from the sources of
the production's psmc-rs fits (the `src/` tree dcf29ca) and run with the
production's commands (only the paths changed), and the laptop's C psmc
fits of the same 40 genomes, from an earlier laptop run, were compared
with Ibex's, made by the same C psmc binary as every C psmc fit of the
production (section 5). The scripts, tables and laptop fits are in
`experiment/crossplatform/`, whose `summary.md` gives every number below.

| | laptop | Ibex (production) |
| --- | --- | --- |
| CPU and system | Apple M1 Pro (arm64), macOS 27.0, Apple's math library | AMD EPYC 7702 (x86-64), Linux, glibc 2.34 |
| psmc-rs | the same sources (`src/` dcf29ca, no uncommitted changes), rustc 1.97.1, release profile, no build flags; NEON kernels | rustc 1.99.0, release profile, no build flags; AVX2 and FMA kernels |
| C psmc | lh3/psmc 0.6.5-r74, the laptop's `-O1` build by Apple clang, which fuses multiply-adds (identified by its machine code; it has no build record) | gcc 11.4.1, `-O3 -march=x86-64-v3 -ffp-contract=off` (section 7) |

**Inputs.** The laptop's 40 E1 genomes were byte-identical to Ibex's
(psmcfa, VCF and index, 40 of 40; the heterozygous positions and counts of
all 400 chromosomes), and both platforms fitted the same NA12878 file.

**Outputs.** No output was byte-identical across the platforms, for either
program (psmc-rs 0 of 41, C psmc 0 of 40), but psmc-rs agreed with itself
far more closely than the two builds of C psmc agreed with each other
(minimum / median / maximum over the 40 E1 fits; the last column is the
difference between the programs in the production on Ibex):

| | psmc-rs, laptop vs Ibex | C psmc, laptop vs Ibex | psmc-rs vs C psmc, Ibex |
| --- | --- | --- | --- |
| theta, relative | 9.1e-9 / 1.2e-7 / 1.9e-6 | 1.5e-7 / 5.8e-6 / 5.3e-5 | 8.5e-7 / 6.3e-5 / 3.8e-4 |
| largest relative difference of any parameter | 3.1e-8 / 5.3e-7 / 2.1e-5 | 3.0e-5 / 1.1e-4 / 0.21 | 6.0e-5 / 2.2e-4 / 0.99 |
| RMSE of log10 Ne over the window | 4.8e-9 / 8.8e-8 / 2.4e-6 | 3.6e-6 / 8.5e-6 / 2.4e-5 | 3.7e-6 / 2.5e-5 / 2.0e-4 |
| largest difference of log10 Ne, any interval | 1.3e-8 / 2.1e-7 / 8.2e-6 | 1.4e-5 / 4.9e-5 / 0.10 | 1.0e-5 / 7.8e-5 / 2.0 |
| final log-likelihood, relative | 3.6e-15 / 1.9e-12 / 1.5e-11 | 4.9e-12 / 1.7e-10 / 1.2e-9 | 1.5e-10 / 3.3e-9 / 1.3e-8 |

C psmc's 0.21 and 0.10 are expansion replicate 5's first lambda, which
runs away outside the window in both builds (without that fit, 5.4e-4 and
2.4e-4). In psmc-rs the same lambda stops at its bound of 1e4 on both
platforms, but on the way there it differed between them by up to a
relative 3.0e-3 (at iteration 14), and this replicate gave psmc-rs's
largest differences: 2.1e-5 in the next lambda, 8.2e-6 in log10 Ne inside
the window, an RMSE of 2.4e-6. Replicate by replicate, psmc-rs's RMSE
between the platforms was smaller than C psmc's between its builds in 40
of 40 (by a median factor of 124, at least 10), and smaller than the
difference between the programs in 40 of 40 (median 383, at least 39).
NA12878's two psmc-rs fits agreed to an RMSE of 1.5e-7 and within a
relative 1.1e-6 in every parameter.

**Where the runs part.** psmc-rs's log-likelihood at the starting point
was bit-identical in 40 of the 41 fits (the other, expansion replicate 5,
differed in the last place), as section 8 leads one to expect: the NEON
and the AVX2 and FMA kernels fuse the same multiply-adds, and no sum
changes its order with the vector width. In the others the first
differences appear in iterations 1 to 3. What
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
B in 27, C in 10 and failed in 3 of 40, and NA12878's fit was C; C psmc's
builds failed in 40 of 40. All 13 psmc-rs fits that missed B, and
NA12878's, did so on the parameter limit, at the most recent free lambdas
(the first to the third), which the data constrain least; the three that
failed differed by 1.2e-6 to 8.2e-6 in log10 Ne, 7 to 46 times less than
the difference between the programs rather than the required 50. The
tier-C reference was not fixed in advance;
the other references considered, and two deviations from the tiers, are
reported in `experiment/crossplatform/summary.md` (section 5) and change
none of these conclusions.

**Caveats.** The laptop's C psmc build is identified by its checksum and
machine code, not by a build record, and a laptop build without
floating-point contraction was not run. Only E1 and NA12878 were refitted
on the laptop; E2's turtles, the bootstraps and E3 to E7 were not.

**Conclusion.** The production's psmc-rs results depend on the platform
only in their last digits: across two processor architectures, operating
systems and math libraries its curves agreed to an RMSE of at most 2.4e-6
in log10 Ne, a median of 383 times (at least 39 times) below the
difference between psmc-rs and C psmc on the same genome, and a median of
124 times (at least 10 times) below the difference between two builds of
C psmc on these platforms. Byte-identical results across platforms need
the same kernels and the same math library on both (section 8).

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
cn513-10-r, cn506-18-r, cn506-15-l, cn514-02-l and cn513-17-l, as SLURM
jobs 54256961 to 54256965 in that order (`E3_runs.csv`: `hostname`,
`slurm_job_id`); each repeat's 19 runs took 2.04 to 2.06 hours of wall
time (the sum of its `wall_seconds`).

| tool | build |
| --- | --- |
| C psmc | lh3/psmc 0.6.5-r74, built on 2026-10-07 by `tools --build-c` with gcc 11.4.1 (Red Hat 11.4.1-3) and CFLAGS `-g -Wall -O3 -march=x86-64-v3 -ffp-contract=off -Wno-unused-function` (its `BUILD.json`): the production build of section 7 |
| psmc-rs | 0.2.0, built with rustc 1.99.0, the release profile and no build flags from the `src/` tree dcf29ca with `Cargo.toml` c0be5cf and `Cargo.lock` ae10354, and no uncommitted changes (`tool.rs02.source_tree`, `source_dirty`): the binary that made every psmc-rs fit of the production (section 6); it picks its AVX2 and FMA kernels at run time (section 8) |

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
about 0.2 core (0.17 to 0.20 core more than the same run on one thread) to
an idle node's 0.20 to 0.925 busy core (median 0.32). On macOS, where
not every process can be read, the limit applies instead to all of the
machine's busy time but the tool's, and each record names the measure
used.

**Runs.** Before its runs, each `e3` job measured its idle node for 30 s,
to stop if the node was busier than the protocol allows; none stopped.
The protocol deferred runs while a node was busy and ran them again
later: the runner logs have one deferral line in each of repeats 1, 2, 4
and 5 and none in repeat 3, and every deferred run was clean when it ran.
`merge_e3.sh` merged the five repeats and found every run
done for the current code. All 95 runs (20 of C psmc and 75 of psmc-rs)
are clean (`E3_environment.csv`: `runs_not_quiet` and `runs_not_clean`
are 0). In the quiet check's 2 s window before them the node had 0.20 to
0.925 core busy, median 0.32 (the limit is 1; the five nodes' medians
0.26 to 0.335). During them the other
processes kept a median 0.27 and at most 0.48 core busy (the limit is
0.5), at most 0.37% of all cores (the limit is 5%), and steal time was
0 (`E3_runs.csv`: `busy_before_cores`, `other_cores_during`,
`other_cpu_share_during`, `steal_cores_during`). Each median below is
taken over the five repeats, one per node, so the spread of the repeats
includes the spread between the nodes.

**Wall time** in seconds (`E3_table2.csv`: `wall_median_s`, and in
parentheses `wall_min_s` to `wall_max_s`); – marks a setup E3 does not
run:

| tool | threads | 100 Mb | 300 Mb | 1000 Mb | 3000 Mb |
| --- | --- | --- | --- | --- | --- |
| C psmc | 1 | 152.2 (152.0–155.1) | 446.6 (445.9–454.7) | 1,479.1 (1,477.0–1,494.1) | 4,424.6 (4,414.2–4,455.8) |
| psmc-rs | 1 | 10.98 (10.97–10.99) | 31.85 (31.32–31.89) | 103.9 (103.7–105.1) | 317.7 (315.7–318.1) |
| psmc-rs | 2 | – | – | – | 163.4 (163.0–163.8) |
| psmc-rs | 4 | – | – | – | 87.44 (86.89–87.90) |
| psmc-rs | 8 | 10.46 (10.46–10.96) | 11.01 (11.00–11.53) | 21.46 (21.44–21.46) | 43.93 (43.90–44.49) |
| psmc-rs | 16 | – | – | – | 22.50 (22.49–22.56) |
| psmc-rs | 30 | 10.45 (10.43–10.47) | 11.48 (10.97–11.52) | 11.52 (11.49–11.56) | 11.53 (11.52–12.04) |

The fastest and the slowest repeat were at most 2.0% of the median apart
in every setup but four, all psmc-rs: 100 and 300 Mb on 8 threads and
300 and 3000 Mb on 30 (4.6 to 4.8%).

**Speed-ups** (`E3_table2.csv`, `speedup_vs_c`: C psmc's median wall time
on the same genome over the tool's). psmc-rs on one thread was 13.9 to
14.2 times as fast as C psmc at every size: 13.9 on 3000 Mb, 317.7 s
against 4,424.6 s, or 12.7 s against 177.0 s per EM iteration
(`seconds_per_iteration_median`). On 3000 Mb it was 101 times as fast on
8 threads, 197 on 16 and 384 on 30. Paired within each repeat, on one
node, C psmc's wall time over psmc-rs's on 3000 Mb was 13.91 to 14.05
on one thread, and 100.2 to 101.0, 196.1 to 198.0 and 367.4 to 386.4 on
8, 16 and 30 threads; on one thread it was 13.84 to 14.52 over the four
sizes (`E3_runs.csv`, `wall_seconds`). psmc-rs's threads added at most
4.8% to its CPU time, and none on 100 Mb: its median on 3000 Mb was
315.8 s on one thread and 320.3 to 325.1 s on 2 to 30, on 1000 Mb 103.1 s
on one thread and 106.9 and 108.0 s on 8 and 30, and on 300 Mb 31.2 s on
one thread and 32.1 s on both 8 and 30 (`cpu_median_s`).

**Peak memory** in MB of 2^20 bytes (`E3_table2.csv`,
`peak_rss_median_mb`: the median over the repeats of the larger of the
tool's `ru_maxrss` and the resident memory of its process tree sampled
every 0.5 s):

| tool | threads | 100 Mb | 300 Mb | 1000 Mb | 3000 Mb |
| --- | --- | --- | --- | --- | --- |
| C psmc | 1 | 1,035 | 1,037 | 1,043 | 1,063 |
| psmc-rs | 1 | 26.0\* | 26.0\* | 26.0\* | 35.7 |
| psmc-rs | 2 | – | – | – | 38.9 |
| psmc-rs | 4 | – | – | – | 45.0 |
| psmc-rs | 8 | 26.0\* | 26.0\* | 40.8 | 60.7 |
| psmc-rs | 16 | – | – | – | 90.5 |
| psmc-rs | 30 | 26.0\* | 26.0\* | 74.5 | 148.9 |

\* In these 35 runs the tool's `ru_maxrss` was 26.0 MB whatever the size
and the threads, and the sampled peak 7.0 to 19.6 MB (`E3_runs.csv`:
`maxrss_mb`, `peak_rss_tree_sampled_mb`). `ru_maxrss` is never below the
tool's peak and the sampled peak never above it, so these cells are an
upper bound: psmc-rs's own peak lies between the two. Where its peak
is larger (1000 Mb on 8 and 30 threads, 3000 Mb), `ru_maxrss` equals the
sampled peak or exceeds it by at most 1.0 MB.

C psmc held about 1 GB whatever the size. psmc-rs peaked at 35.7 MB
on 3000 Mb on one thread and at 148.9 MB on 30, where C psmc held 29.7
and 7.1 times as much (`c_rss_over_rss`).

**Thread scaling** of psmc-rs on 3000 Mb (`E3_thread_scaling.csv`;
speed-up: the one-thread median wall time over the median on t threads;
efficiency: the speed-up over t):

| threads | 1 | 2 | 4 | 8 | 16 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| median wall time (s) | 317.7 | 163.4 | 87.44 | 43.93 | 22.50 | 11.53 |
| speed-up | 1 | 1.94 | 3.63 | 7.23 | 14.12 | 27.55 |
| efficiency | 1 | 0.97 | 0.91 | 0.90 | 0.88 | 0.92 |
| peak memory (MB) | 35.7 | 38.9 | 45.0 | 60.7 | 90.5 | 148.9 |

psmc-rs processes sequences in parallel, and a single sequence runs on
one thread, so its wall time stops improving once the threads reach the
number of chromosomes: on 100 Mb (one chromosome) it took 10.98, 10.46 and
10.45 s on 1, 8 and 30 threads, on 300 Mb (three) 11.01 and 11.48 s on 8
and 30, and every genome took 10.5 to 11.5 s on at least as many threads
as chromosomes, about the time of one chromosome on one thread. With
chromosomes of equal length, t threads need at least ⌈chromosomes/t⌉
times one chromosome's time, and the median wall time divided by
⌈chromosomes/t⌉ is 10.4 to 11.5 s in every psmc-rs setup
(`E3_table2.csv`: `wall_median_s`, `chromosomes`, `threads`). On the 30
chromosomes of 3000 Mb this caps the efficiency at 30/(t⌈30/t⌉): 1 on 2
and 30 threads and 0.94 on 4, 8 and 16, which the measured 0.91, 0.90 and
0.88 approach.

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
