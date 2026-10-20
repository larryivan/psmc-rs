# Experiment: reproduction package

Everything that makes the manuscript's results. The pipeline in
[`pipeline/`](pipeline/README.md) (`python -m experiment.pipeline`)
simulates genomes, runs psmc-rs 0.2 and C psmc (lh3/psmc 0.6.5-r74) with
identical settings, records how every output was made, writes each
experiment's tables and draws the figures from them. Its production runs
are committed in `runs/`, so the tables and figures can be made again
without running either tool. Commands are run from the repository root;
the pipeline finds the repository from its own path, so any working
directory works.

| experiment | what | in the article |
| --- | --- | --- |
| E1 | simulated accuracy and agreement with C psmc: four demographic models, ten genomes of 500 Mb each | Fig 1, Table 1, Table S1, Fig S1 |
| E2 | four real genomes: three turtles and NA12878 (chr1-22), with bootstraps | Fig 1, Table 1 |
| E3 | performance: run time and peak memory of C psmc and psmc-rs on 100 to 3,000 Mb, thread scaling, determinism | Fig 2, Table 2 |
| E4 | partly callable bins: the routes from a VCF and its callable mask to PSMC | Fig S2 |
| E5 | bootstrap calibration on simulated genomes | Fig S3 |
| E6 | supplementary analyses: EM convergence, input formats, TMRCA decoding, the QC report | quoted in the text |
| E7 | the 2,504 genomes of the 1000 Genomes 30x data | Fig 3, Fig 1, Table 1 |
| FIG | every figure, drawn from the experiments' tables | |

Where to read on:

- [`pipeline/README.md`](pipeline/README.md): the design, each
  experiment's jobs and tables, the runner, the production run and the
  tools.
- [`pipeline/figures/README.md`](pipeline/figures/README.md): the figures,
  their house style and the tables they read.
- [`Supplementary_Note_1_Reproducibility.md`](Supplementary_Note_1_Reproducibility.md):
  how to reproduce the results (from the committed runs, on one machine or
  on a SLURM cluster), the provenance records, the C psmc builds,
  determinism, the cross-platform comparison and the performance results
  (E3).
- [`slurm/README.md`](slurm/README.md): the scripts that ran the
  production on KAUST's Ibex cluster.
- [`validation/README.md`](validation/README.md): the simulation study of
  `--bin-callable`, what E4 expects; [`crossplatform/README.md`](crossplatform/README.md):
  the same jobs on a laptop and on Ibex.

## Layout

```text
experiment/
├── README.md                          # this file
├── Supplementary_Note_1_Reproducibility.md
├── pipeline/                          # python -m experiment.pipeline: E1-E7, FIG, the runner, the tools
├── slurm/                             # the pipeline in SLURM allocations
├── real_data/                         # input psmcfa: the three turtles and NA12878's chr1-22 (E2);
│                                      #   Papuan chr1 (the Rust tests, the top-level README's benchmarks)
├── runs/                              # E1 ... E7, FIG: fits, truth and provenance records, tables, figures
├── validation/                        # the --bin-callable simulation study
├── crossplatform/                     # laptop against Ibex
└── tests/                             # python -m unittest discover -s experiment/tests
```

Committed under `runs/`: every fit (psmc-rs JSON and `.psmc`, C psmc
`.psmc`), bootstrap replicates' JSON, truth, QC and provenance records,
tables and figures. Not committed, and made again on demand from the
seeds: the simulated genomes, the bootstrap replicates' `.psmc` files,
HTML reports, logs and the runner's state (`.gitignore`). E7's inputs, one
psmcfa per 1000 Genomes sample, are on Zenodo.

## Quick start

Python 3.13 with the exact versions of `pipeline/requirements.lock` (the
simulations are byte-identical only with them), htslib (`bgzip`, `tabix`,
`bcftools`), Rust 1.88 or later, and, to run C psmc, lh3/psmc and its
utilities in `experiment/.tools` (`pipeline/README.md`, Tools;
`tools --build-c` builds C psmc and records how).

```bash
python3.13 -m venv experiment/.venv
experiment/.venv/bin/python -m pip install -r experiment/pipeline/requirements.lock
cargo build --release --locked
PY=experiment/.venv/bin/python
$PY -m experiment.pipeline tools          # the tools and their identities
$PY -m experiment.pipeline report E1      # an experiment's tables from the committed runs
$PY -m experiment.pipeline.figures        # every figure whose tables exist
$PY -m experiment.pipeline plan           # what running everything from scratch costs
$PY -m experiment.pipeline start E1       # a detached, resumable runner; status, stop
```

`--profile smoke` runs every experiment at a small size in
`experiment/.runs-smoke/`. Supplementary Note 1 gives the full
procedures.

## Tests

```bash
experiment/.venv/bin/python -m unittest discover -s experiment/tests
```

The tests that need msprime, numpy, pandas, matplotlib, psutil, htslib, C
psmc or a psmc-rs build skip without them.
