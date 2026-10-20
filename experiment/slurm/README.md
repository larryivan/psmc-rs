# Running the experiments on a SLURM cluster

The production runs of the paper were made on KAUST's Ibex cluster with
these scripts. They run the same pipeline as a single machine
(`python -m experiment.pipeline`, see `experiment/pipeline/README.md`), in
SLURM allocations:

| job | what | allocation |
| --- | --- | --- |
| `build` | builds psmc-rs 0.2 (release profile, no RUSTFLAGS) and C psmc (`tools --build-c`, recorded in `BUILD.json`), checks every tool (`tools`), runs every Rust and Python test; ends with "all passed" | 32 CPUs, 2 h |
| `prod` | E1 E2 E4 E5 E6 E7: one runner in the foreground within the allocation (E7 once `e7-prep` is done) | `$PROD_CPUS` CPUs (at least 52: E2's bootstraps run on 50 threads), `$PROD_MEM`, `$PROD_TIME` |
| `e3 R` | E3's simulations and the timing runs of repeat R, each run alone | an exclusive node of `$E3_CONSTRAINT`, `$E3_TIME` |
| `e3-report` | E3's tables from the merged repeats | 2 CPUs, 1 h |
| `e7-prep [N...]` | E7's data stage (`kg1000`): setup; one task per chromosome (default 1-22): its VCF downloaded to node-local disk and checked, split per sample, every sample's psmcfa (chr21: then the pilot); the samples joined and `MANIFEST.json` | setup 2 CPUs, 3 h; chromosomes `$E7_PREP_PARALLEL` at once, each `$E7_PREP_CPUS` CPUs, `$E7_PREP_MEM`, `$E7_PREP_TIME`; samples 16 CPUs, 4 h |

Every repeat of E3 runs on its own exclusive node of one CPU model, so that
the speed-ups of a repeat are paired on one node and the spread of the
repeats includes the spread between nodes. A resubmitted repeat goes back
to the node of its first runs (`--nodelist`), and `merge_e3.sh` refuses a
repeat timed on more than one host. The report says which CPU models and
hosts the clean runs came from, and warns when they mix. Before its runs,
an `e3` job measures the idle node for 30 s and stops if it is busier than
the timing protocol allows (its log names the busiest processes), rather
than deferring every run until the time limit.

## Setup, once, on a login node

```bash
git clone <repository> $MAIN && cd $MAIN
git worktree add --detach $PROD <commit>              # the production commit
cp experiment/slurm/site.env.example $PROD/experiment/slurm/site.env   # and fill it in
rustup toolchain install 1.99.0 --profile minimal
(cd $PROD && cargo fetch --locked)                     # compute nodes build offline
# experiment/.tools: lh3/psmc 0.6.5-r74 sources in psmc_c/, vcfutils.pl,
# msmc-tools/generate_multihetsep.py (see experiment/pipeline/README.md)
# Python 3.13.13 with experiment/pipeline/requirements.lock and htslib 1.24:
micromamba create -p $MAIN/experiment/.venv -c conda-forge -c bioconda python=3.13.13 bcftools=1.24 htslib=1.24 pip
$MAIN/experiment/.venv/bin/python -m pip install -r experiment/pipeline/requirements.lock
```

## Runs

```bash
cd $PROD/experiment/slurm
./submit.sh build                 # its log ends with "all passed"
./submit.sh prod                  # E1 E2 E4 E5 E6; submit again if the time ran out
./submit.sh e3 1 2 3 4 5          # in parallel with prod
./merge_e3.sh                     # once every e3 job has ended
./submit.sh e3-report             # waits for prod: both use $PROD's runs directory
./submit.sh e7-prep               # E7's inputs (network-bound: hours per chromosome); then prod again for E7
```

Every job waits for a build job still queued or running. `prod` and
`e3-report` share one job name (`psmcrs-runs`) and run one after the other,
since a runs directory holds one runner; each `e3 R` is its own singleton.
The `e7-prep` jobs write only `$PROD/experiment/runs/E7/inputs/`, which no
runner holds, and E7 has no runner jobs until its last step has written the
manifest; a chromosome task that failed is submitted again with `e7-prep N`
(setup and finished chromosomes are skipped). A job that reaches its time
limit leaves its unfinished runs pending: 90
minutes (E3: 2 hours) before the limit the runner starts nothing new, and
submitting again runs only what is not done. Status, from a login node:

```bash
cd $PROD && PSMC_SLURM_DIR=$PROD/experiment/slurm bash -c 'source experiment/slurm/env.sh && \
  $PY -m experiment.pipeline status --profile production --runs-dir $PROD/experiment/runs --only "" E1 E2 E3 E4 E5 E6 E7'
```

Draw the figures on a machine with Arial (`start FIG`) from the runs
directory copied back.

`env.sh` gives every job the same environment whatever shell submitted it:
it purges modules, drops the inherited build, Python and pipeline variables,
and sets the pinned toolchain without build flags (a psmc-rs built otherwise
would be another tool to the pipeline, and every psmc-rs job would run
again), the Python environment and its htslib first on `PATH`, the tools
directory, and a machine lock in the job's own `/tmp`.
