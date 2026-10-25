# Cross-platform comparison

How much the results of an earlier production, computed on KAUST's Ibex
cluster (AMD EPYC 7702, x86-64, Linux) with psmc-rs from `src` 4c0d777,
depend on the platform: E1's 40 simulated genomes and NA12878 fitted again
by psmc-rs 0.2 on the development laptop (Apple M1 Pro, arm64, macOS) from
the same sources and with that production's commands, and the laptop's C
psmc fits of the 40 genomes compared with Ibex's. The production committed
since (4b2c636f) fitted every psmc-rs job again, from `src` dcf29ca, and
was not compared; its C psmc fits are the ones compared here. Supplementary
Note 1, section 9, gives the results; `summary.md` gives every number, the
toolchains, the controls, the agreement tiers and the deviations from them.

The analysis is a one-off, outside the pipeline, run on 2026-10-11:

| script | what it does |
| --- | --- |
| `scripts/make_jobs.py` | the laptop command of each Ibex fit, from its provenance record (`jobs.json`) |
| `scripts/run_job.py` | runs one job and writes its record (`records/`) |
| `scripts/inputs.py` | step 1: the laptop's genomes against the Ibex records |
| `scripts/compare.py` | the comparisons (the CSV tables) |
| `scripts/tables.py`, `scripts/make_summary.py` | `summary.json` and the tables of `summary.md` |
| `scripts/tiers_2026-10-07.py` | the earlier laptop-against-Ibex review that fixed the agreement tiers on 2026-10-07, before these runs, kept as it was |

The scripts read the Ibex results from `experiment/runs` as it was before
cbab860c (against the current one, whose psmc-rs fits a later build made,
they would not reproduce `summary.md`) and the laptop's simulated
genomes and C psmc fits from a frozen laptop run, `experiment/.production`
(not in the repository; the C fits it compared are copied to `laptop_c/`). `runs/` holds the laptop's psmc-rs fits and
`records/` one record per laptop run; paths in the records are written as
`<repo>` (this repository) and `<xplat>` (the directory the analysis ran
in, which this one reproduces).
