# Cross-platform comparison

How much the production's results, computed on KAUST's Ibex cluster (AMD
EPYC 7702, x86-64, Linux), depend on the platform: E1's 40 simulated
genomes and NA12878 fitted again by psmc-rs 0.2 on the development laptop
(Apple M1 Pro, arm64, macOS), built from the sources of the production's
psmc-rs fits (`src` dcf29ca) and run with their commands; the laptop's C
psmc fits of the 40 genomes compared with Ibex's; and the reports of E1,
E2, E4 and E5 run again on the laptop on the committed fits. The Ibex
results are those of `experiment/runs` (the production committed at
cbab860c, whose psmc-rs fits ran at 4b2c636f; its C psmc fits come from
earlier jobs of the same C psmc binary). Supplementary Note 1, section 9,
gives the results; `summary.md` gives every number, the toolchains, the
controls, the agreement tiers and the deviations from them. An earlier
version of this directory is in git history (c6bbb340).

The analysis is a one-off, outside the pipeline, run on 2026-10-25 from
this directory, in this order:

| script | what it does |
| --- | --- |
| `scripts/make_jobs.py` | the laptop command of each Ibex fit, from its provenance record (`jobs.json`; with `--bin`, `--runs`, `--out` and `--only`, the job lists of the two controls) |
| `scripts/run_job.py` | runs one job, or `all` of a job list (each in its own process, at most `-P 4` at once, under `nice 19`), and writes its record (`records/`) |
| `scripts/toolchain.py` | the laptop, the two builds and the laptop C psmc binary (`toolchain.txt`) |
| `scripts/inputs.py` | step 1: the laptop's genomes against the Ibex records |
| `scripts/compare.py` | the comparisons (the CSV tables, `summary.json`) |
| `scripts/reports.py` | E1, E2, E4 and E5's tables written again on the laptop, against the committed ones (`reports_laptop_vs_ibex.csv`, `reports.json`) |
| `scripts/make_summary.py` | `summary.md`, every number in it from the files the scripts above write and the Ibex records |
| `scripts/tiers_2026-10-07.py` | the earlier laptop-against-Ibex review that fixed the agreement tiers on 2026-10-07, before these runs, kept as it was |

```bash
cargo build --release --locked                                   # from a clean checkout of src dcf29ca; then
cargo +1.99 build --release --locked                             # the control, in another CARGO_TARGET_DIR
# copy the two binaries to bin/psmc-rs and bin/psmc-rs-rustc1.99
PY=../.venv/bin/python
$PY -I -B scripts/make_jobs.py
$PY -I -B scripts/make_jobs.py --bin bin/psmc-rs-rustc1.99 --runs runs-rustc1.99 --out jobs_rustc1.99.json
$PY -I -B scripts/make_jobs.py --runs runs-repeat --out jobs_repeat.json --only E1/fit-rs/constant/rep01 E1/fit-rs/bottleneck/rep06
$PY -I -B scripts/run_job.py all jobs.json records
$PY -I -B scripts/run_job.py all jobs_rustc1.99.json records-rustc1.99
$PY -I -B scripts/run_job.py all jobs_repeat.json records-repeat
$PY -I -B scripts/toolchain.py                                   # run again later: --built-at ac4fed5d, the builds' HEAD
$PY -I -B scripts/inputs.py
$PY -I -B scripts/compare.py
$PY -B scripts/reports.py
$PY -I -B scripts/make_summary.py
```

The scripts read the Ibex results from `experiment/runs`, and the laptop's
simulated genomes and C psmc fits from a frozen laptop run,
`experiment/.production` (not in the repository; the C fits it compared
are copied to `laptop_c/`, and `compare.py` checks the copies);
`toolchain.py` also reads the laptop's C psmc builds in
`experiment/.tools/psmc_c`. `runs/` holds the laptop's psmc-rs fits and
`records/` one record per laptop run; the two controls' fits are in
`runs-rustc1.99/` and `runs-repeat/`, their job lists in
`jobs_rustc1.99.json` and `jobs_repeat.json` and their records in
`records-rustc1.99/` and `records-repeat/`. Paths in the job lists and
records are written as `<repo>` (this repository) and `<xplat>` (this
directory), which `run_job.py` expands. Not kept: the binaries (`bin/`),
logs and the HTML reports psmc-rs writes beside each JSON.

A binary's whole-file sha256 (`toolchain.txt`, the records) depends on
the target directory it was built in, through its Mach-O UUID and the
ad-hoc code signature over it; `toolchain.txt` also gives each build's
sha256 with those bytes zeroed, which a rebuild of the same sources by
the same toolchain in another target directory reproduces
(`scripts/toolchain.py --digest <binary>`). Both depend on the source
paths the binary holds in its panic messages, under the builder's home
directory (`~/.cargo`, `~/.rustup`), so another user's or machine's build
is not expected to reproduce either.

`scripts/tiers_2026-10-07.py` is the record of the tiers (`tier()`, lines
225-232, which `compare.py` applies verbatim) and does not run on the
repository as it is now: it hard-codes the author's checkout path, reads
columns of E1's tables since renamed (`rmse_rs_vs_c_mean` of
`E1_table1.csv` and `rmse_log10_ne_rs_vs_c` of `E1_rs_vs_c.csv`, then
from step curves sampled at fixed times), and reads E2's C psmc fit of
the Papuan chromosome 1, which E2 no longer fits. Its outputs
(`xplatform.json`, `xplatform_cfits.csv` and `xplatform_bootstrap.csv`,
written into `scripts/`) are not kept; `summary.md` (section 5) gives the
tiers and how its references compare with the ones used here.

`x86_cpus/` holds a separate check: the production binary refitting an E1 genome byte for byte on
three other x86-64 CPU models (its README).
