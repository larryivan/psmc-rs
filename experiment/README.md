# Experiment: reproduction package

Everything needed to reproduce the manuscript's figures and tables: the real
genomes, the scripts that simulate data and run `psmc-rs` and the original C
`psmc`, the result tables, and the notebook that draws the figures. All
commands below are run from the repository root, but every script locates the
repository from its own path, so any working directory works.

## Read first: state of the packaged results

- **All psmc-rs results must be regenerated after the 0.2 fixes.** The tables
  and figures in `runs/` were produced with psmc-rs 0.1. Version 0.2 changes
  the numerics, the input readers, the bootstrap and several defaults
  (initial `rho0 = theta0/4` instead of `theta0/5`, as in lh3/psmc; the
  `4+5*3+4` pattern; no smoothing). Every psmc-rs number in Figures 2-4,
  Tables 1-2 and Supplementary Tables S2-S5 therefore describes 0.1. The C
  psmc runs were made with `-r5` to match 0.1; for 0.2 both tools are run
  with `RHO_T_RATIO` [4] (C `-r4`, psmc-rs `--tr-ratio 4`), so regenerate
  both sides (the scripts do: they reuse no output made with other
  settings, see step 2). The scripts pass every setting that shapes the
  fit to psmc-rs explicitly (see the table in step 2), so they need a
  psmc-rs 0.2 build:
  before the first run compared with C psmc they check, with a
  zero-iteration run, that the binary starts from `theta0/RHO_T_RATIO`, and
  a 0.1 build (which has no `--tr-ratio`) is refused with an explanation.
- **Inference outputs are not committed yet.** `runs/main_text/outputs/` and
  `runs/real_data/outputs/` (psmc-rs JSON, C `.psmc`), from which Figures 2
  and 3 and Table 1 are computed, were excluded by `.gitignore` and never
  committed. The ignore rules now keep them, so commit them after
  regenerating. Until then `run.py` lists them as missing together with the
  command that regenerates them, and `run.py --keep-going` draws the figures
  whose inputs are present (Figure 4 and Supplementary Figures S1-S3).
- **The zigzag row of `S5_format_consistency.csv` is exactly 0.0 by
  construction.** S5 writes one simulation per model as psmcfa, multihetsep
  and VCF. The psmcfa covers the whole 500 Mb (5,000,000 bins of 100 bp),
  but psmc-rs 0.1 ended a multihetsep or VCF sequence at its last
  heterozygous site, so the homozygous bins after it existed only in the
  psmcfa. They were the only difference between the three inputs, so the
  packaged table measures their effect, not a disagreement between readers:
  multihetsep and VCF agree exactly for every model
  (`rmse_curve_log10_vcf_vs_mhs` is 0). In the zigzag simulation (seed 45)
  the last heterozygous site, at position 499,999,906, falls in the final
  bin, so the three inputs are identical and every column is 0. Checked
  with msprime 1.4.4: zigzag gives byte-identical psmc-rs output from the
  three files, and the constant model drops 14 bins, after whose removal
  its psmcfa output is byte-identical to the multihetsep one. **With 0.2 the
  VCF columns change meaning:** the VCF reader extends each sequence to the
  `##contig` length that msprime's VCF writer records, so the VCF input
  gives exactly the psmcfa's bins, and the `vcf_vs_psmcfa` columns become 0
  for every model, while the multihetsep input still ends at the last
  heterozygous site (multihetsep has no positions after it). Checked with
  psmc-rs 0.2 on a tskit-style VCF of 200 kb with its last heterozygous
  site 59 bins before the end: `psmc-rs convert` gives the psmcfa's 2,000
  bins from the VCF and the first 1,941 from the multihetsep. Regenerated
  tables record the count in `psmcfa_bins_after_last_het`, and the Figure
  S2 caption should explain what S5 measures. To compare the readers on
  identical data instead, trim the psmcfa to its last heterozygous bin.
- **The packaged S2 table includes C psmc's round 0**, whose `LK 0` is a
  placeholder (no E-step has run), and Supplementary Figure S1 plots it, which
  squeezes both curves into a flat line. The S2 runner now leaves it out
  (see step 3).
- **The human sample is chromosome 1 only.**
  `real_data/Papuan_highlands_diploid.psmcfa` holds one sequence, `>chr1`,
  of 2,487,383 bins (about 249 Mb). The tables and the notebook now label it
  "Homo sapiens (Papuan Highlands, chr1)"; the committed Figure 3 files still
  carry the old title until they are redrawn.
- The stdpopsim and msprime versions used for the 0.1 simulations were not
  recorded. The simulation scripts now print them into the command logs.

## Layout

```text
experiment/
├── README.md                       # this file
├── Supplementary_Note_1_Reproducibility.md
├── requirements.txt                # Python packages
├── run.py                          # check results + draw every figure
├── notebooks/
│   └── all_plots_precomputed.ipynb # figure code, executed by run.py
├── scripts/                        # simulation and benchmark runners (see scripts/README.md)
│   ├── simulate_msprime_to_psmcfa.py
│   ├── simulate_msprime_to_mhs.py
│   ├── run_main_text.py
│   ├── run_real_data.py
│   ├── run_supplementary.py
│   └── exp_common.py
├── tests/                          # python3 -m unittest discover -s experiment/tests
├── real_data/                      # input psmcfa of the four empirical samples
├── assets/ui/                      # screenshots for Supplementary Figure S3
└── runs/
    ├── main_text/    {inputs,outputs,perf,logs,tables,figures}
    ├── real_data/    {outputs,logs,tables,figures}
    └── supplementary/{shared_inputs,s1_smooth,s2_em,s3_memory,s4_threads,s5_format,
                       logs,tables,figures,diagnostic_figures}
```

Committed: `real_data/`, every `runs/*/tables/*.csv`, `runs/*/figures/`, and
the small `runs/**/outputs/*.json` (psmc-rs outputs and their
`.settings.json` records) and `*.psmc` files.
Ignored as regenerable: simulated inputs (5 MB per 500 Mb model, more for
VCF), timing repeats (`main_text/perf/`, `supplementary/s3_memory/outputs/`,
`supplementary/s4_threads/outputs/`), logs, HTML reports, bootstrap
directories, `diagnostic_figures/` and the `.tsv`/`.md` copies of the tables.

## Requirements

- Rust stable, 1.88 or later, for `psmc-rs` (edition 2024 needs 1.85; the
  locked dependencies of `ratatui` 0.30 need 1.88).
- A C compiler and zlib for lh3/psmc.
- Python 3.9 or later with the packages in `requirements.txt`. Drawing the
  figures needs only numpy, pandas and matplotlib; regenerating results also
  needs stdpopsim, msprime, tskit and psutil.

  ```bash
  python3 -m pip install -r experiment/requirements.txt
  ```

  The scripts and tests were last exercised with Python 3.12 (numpy 2.5.3,
  pandas 2.3.3, matplotlib 3.11.2, msprime 1.4.4, tskit 1.0.3, psutil 7.2.2)
  and, without those packages, Python 3.14. stdpopsim was not installed
  there, so the stdpopsim path (`simulate_msprime_to_psmcfa.py`) was not
  re-run.

## Pipeline

### 0. Build the two tools

```bash
cargo build --release --locked            # target/release/psmc-rs (the default)

git clone https://github.com/lh3/psmc.git /some/dir/psmc
cd /some/dir/psmc
cc -O1 -o psmc cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz
export C_PSMC_BIN=/some/dir/psmc/psmc     # or pass --c-psmc, or put psmc on PATH
```

Build lh3/psmc with `-O1` (or `-O0`). On macOS arm64 with Apple clang 21 an
`-O2` build (the upstream Makefile's default) segfaults after the first EM
round; `-O0` and `-O1` builds produce byte-identical output. The reference
outputs in `tests/data/c_reference/`, which `tests/c_equivalence.rs` checks
psmc-rs against, come from lh3/psmc 0.6.5-r74 built this way; `psmc` prints
its version when run without arguments.

psmc-rs is taken from `--psmc-rs`, else `$PSMC_RS_BIN`, else
`target/release/psmc-rs` of this checkout (built if missing; a `psmc-rs` on
`PATH` is never used implicitly).

### 1. Simulate

```bash
python3 experiment/scripts/run_main_text.py --steps simulate
```

writes `runs/main_text/inputs/{constant,bottleneck,expansion,zigzag}.psmcfa`:
one diploid individual, 500 Mb, simulated with stdpopsim's generic HomSap
contig and msprime engine (mutation rate 2.5e-8 per bp per generation,
recombination rate from the stdpopsim HomSap catalog unless `SIM_RECOMB` is
set), coded in 100 bp windows (`K` if the window holds a heterozygous site,
else `T`). The models use ms conventions (time in units of 4 N0 generations,
size as a ratio to N0):

| model | N0 | size changes (time: size/N0) | seed |
| --- | --- | --- | --- |
| constant | 10,000 | none | 42 |
| bottleneck | 20,000 | 0.01: 0.05, 0.015: 0.5, 0.05: 0.25, 0.5: 0.5 | 43 |
| expansion | 10,000 | 0.01: 0.1, 0.06: 1, 0.2: 0.5, 1: 1, 2: 2 | 44 |
| zigzag | 1,000 | 0.1: 5, 0.6: 20, 2: 5, 10: 10, 20: 5 | 45 |

One model by hand:

```bash
python3 experiment/scripts/simulate_msprime_to_psmcfa.py --model bottleneck --ne 20000 \
  --length 500000000 --window 100 --mutation 2.5e-8 --seed 43 \
  --out experiment/runs/main_text/inputs/bottleneck.psmcfa
```

`run_supplementary.py` simulates its own inputs with the same script and
seeds (identical files) in `runs/supplementary/shared_inputs/`, plus the
constant model at 100, 250, 500 and 1000 Mb for S3, 8 zigzag contigs of 75 Mb
(seeds 7400-7407) for S4, and, for S5, one msprime simulation per model
(recombination rate 1.25e-8) written as psmcfa, multihetsep and VCF. The
multihetsep file lacks the homozygous bins after the last heterozygous site
(see "Read first").

### 2. Run psmc-rs and C psmc

```bash
python3 experiment/scripts/run_main_text.py --steps main,perf  # Figure 2, Table 1; Table 2
python3 experiment/scripts/run_real_data.py                    # Figure 3
python3 experiment/scripts/run_supplementary.py                # S2-S5 (S1 with --steps S1)
```

Both tools get the same EM settings, passed to psmc-rs explicitly so that
the comparison does not depend on its defaults:

| setting | psmc-rs | C psmc | variable |
| --- | --- | --- | --- |
| EM iterations | `N_ITER` positional (20) | `-N20` | `N_ITER` |
| max coalescent time | `--t-max 15` | `-t15` | `T_MAX` |
| free-parameter pattern (and so the 64 intervals) | `--pattern 4+25*2+4+6` | `-p 4+25*2+4+6` | `PATTERN` |
| initial theta0/rho0 | `--tr-ratio 4` | `-r4` | `RHO_T_RATIO` |
| smoothing penalty | `--smooth-lambda 0` | none | `SMOOTH_LAMBDA` |
| M-step iteration cap | `--mstep-iters 100` | (Hooke-Jeeves to convergence) | `MSTEP_ITERS` |
| threads | `--threads 1` | 1 | `RUST_THREADS` |

plus `--mu <mu> --no-progress` for psmc-rs (`--mu` only scales the curve),
`--batch-size` only when `BATCH_SIZE` is set (from 0.2 on it changes neither
results nor memory), and, in S5, `--input-format`, `--bin-size` and, for the
VCF, `--vcf-mode mask`. The supplementary runner builds its commands from
the same settings. Before the first run that is compared with C psmc, the
scripts run the psmc-rs binary for zero iterations on a tiny synthetic input
with these model options and stop with an explanation unless it starts from
`theta0/RHO_T_RATIO`. `MSTEP_ITERS` stays at psmc-rs's default; C psmc's
optimiser runs each M-step to convergence, which psmc-rs matches more
closely with about 1000 (at ten times the M-step's small share of the run
time).

The scripts reuse an existing output only if it is a finished run of the
current settings, and otherwise say why they run it again (for example
`HomoSapiens.c.psmc was made with -r5 (RHO_T_RATIO is 4); re-running`):

- a C psmc output must have exactly `N_ITER + 1` rounds, and its header
  (`MM pattern:...` and `MM n_iterations:..., max_t:..., theta/rho:...`)
  must give the current `PATTERN`, `N_ITER`, `T_MAX` and `RHO_T_RATIO`;
- a psmc-rs JSON must parse, record the current `n_iter` (0.1 outputs
  record none), and come with a matching `<name>.settings.json`. The
  scripts write that record after each successful psmc-rs run, since the
  JSON does not record the starting `t_max` or the M-step cap. It lists
  the run's arguments except the input and output paths and `--threads`,
  `--batch-size` and `--no-progress`, which do not change the results, so
  changing `RUST_THREADS` or `BATCH_SIZE` reuses an output and changing any
  other setting (`MU` included) does not.

So outputs left from the 0.1 runs (C psmc at `-r5`, psmc-rs JSON without
`n_iter`) are run again on both sides without `--force`. C psmc writes each
round as it goes, so it writes to `<output>.partial`, which is renamed only
when the run succeeds; an interrupted run therefore leaves no output.
`--force-run` (main text), `--force` (real data, supplementary) re-run
everything. Timing steps (`perf`, S3, S4) always re-run because cached
outputs carry no timing.

Expect hours: on the 0.1 runs C psmc needed about 7.5 min and 5 GB RSS per
500 Mb model (7.8 GB at 1 Gb in S3), psmc-rs about 5 min and 1.4 GB;
`perf` repeats each run 10 times. Run the timing steps on an otherwise idle
machine and report its hardware.

Every command, with wall time and peak RSS (psutil, sampled every 20 ms over
the process tree), is appended to `runs/<part>/logs/commands.jsonl`.

### 3. Tables

| table | file (`runs/...`) | written by |
| --- | --- | --- |
| Table 1 (RMSE vs truth) | `main_text/tables/table_1_rmse_notitle.csv` | notebook (`run.py`) |
| Table 2 (runtime, memory) | `main_text/tables/table_2_runtime_memory.csv`, `perf_raw_repeats.csv` | `run_main_text.py --steps perf` |
| Table 2 as plotted | `main_text/tables/table_2_runtime_memory_notitle.csv` | notebook |
| real-data index | `real_data/tables/real_data_rust_vs_c_fit_summary.csv` | `run_real_data.py` |
| Figure 3 RMSE | `real_data/tables/figure_3_empirical_rmse_rs_vs_c.csv` | notebook |
| S2 EM convergence | `supplementary/tables/S2_em_convergence_trace.csv` | `run_supplementary.py` S2 |
| S3 memory scaling | `supplementary/tables/S3_memory_scaling_summary.csv` | S3 |
| S4 thread scaling | `supplementary/tables/S4_thread_scaling_summary.csv` | S4 |
| S5 format consistency | `supplementary/tables/S5_format_consistency.csv` | S5 |

Paths stored in tables are repository-relative. RMSE is that of log10 Ne on
500 log-spaced points between 1e3 and 1e8 years.

In the S2 table, row `iter` k holds each tool's log-likelihood of its
parameters after k - 1 EM iterations, for k = 1 to `N_ITER`: C psmc's `LK`
of round k, and `loglike[k-1]` from the JSON of one psmc-rs run of `N_ITER`
iterations. psmc-rs's `Last EM loglike`, the log-likelihood of the model it
writes, is what C psmc would print as round `N_ITER + 1`, so it is not in
the table; C's round 0 runs no E-step and prints a placeholder `LK 0`,
which is left out too.

### 4. Figures

```bash
python3 experiment/run.py              # check the results, then draw every figure
python3 experiment/run.py --skip-plot  # checks only (standard library only)
python3 experiment/run.py --skip-check # draw only
python3 experiment/run.py --keep-going # draw what can be drawn, exit 1 if anything is missing
```

`run.py` uses the interpreter it is started with and names any missing
package. It executes `notebooks/all_plots_precomputed.ipynb` headlessly and
writes PNG, SVG and PDF to `runs/*/figures/`:

| figure | file | inputs |
| --- | --- | --- |
| Figure 2 | `main_text/figures/figure_2_simulated_equivalence` | `main_text/outputs/` |
| Figure 3 | `real_data/figures/figure_3_empirical_genomes_rust_vs_c` | fit summary + `real_data/outputs/` |
| Figure 4 | `main_text/figures/figure_4_performance_scalability` | Table 2, S3, S4 |
| Supp. Figure S1 | `supplementary/figures/supplementary_figure_S1_em_convergence` | S2 |
| Supp. Figure S2 | `supplementary/figures/supplementary_figure_S2_format_consistency` | S5 |
| Supp. Figure S3 | `supplementary/figures/supplementary_figure_S3_interfaces` | `assets/ui/*.png` |

The notebook also runs in Jupyter: it finds the repository by searching
upward from the working directory, or from `PSMC_RS_ROOT` if set.

## Configuration reference

Environment variables read by the scripts (defaults in brackets):

- EM settings: `N_ITER` [20], `T_MAX` [15], `PATTERN` [`4+25*2+4+6`],
  `RHO_T_RATIO` [4], `MSTEP_ITERS` [100], `BATCH_SIZE` [unset], `MU`
  [2.5e-8], `SMOOTH_LAMBDA` [0] and `RUST_THREADS` [1] (main text, real
  data), `RUST_THREADS_BASE` [1] (supplementary), `PSMC_ALPHA_CACHE_MB`
  [2048; psmc-rs uses it only when a model has more than 256 states]. The
  number of intervals follows `PATTERN`, as in C psmc (`N_STEPS` is no
  longer read).
- Simulation: `SIM_LENGTH_BP` [500000000], `SIM_WINDOW_BP` [100],
  `SIM_MUTATION` [`MU`], `SIM_RECOMB` [catalog rate].
- Repeats and grids: `PERF_REPEATS` [10], `S3_REPEATS` [3], `S3_LENGTHS_MB`
  [100,250,500,1000], `S3_INCLUDE_2000` [0], `THREAD_LIST` [1,2,4,8],
  `THREAD_REPEATS` [3], `S4_CONTIGS` [8], `S4_CONTIG_MB` [75], `S4_SEED_BASE`
  [7400], `S1_SMOOTH_LAMBDAS` [0,1e-5,1e-4,1e-3,1e-2].
- Tools: `PSMC_RS_BIN`, `C_PSMC_BIN`, `SIM_SCRIPT`; notebook: `PSMC_RS_ROOT`.

A quick end-to-end smoke run (minutes, needs msprime and psutil):

```bash
SIM_LENGTH_BP=2000000 N_ITER=2 python3 experiment/scripts/run_supplementary.py \
  --steps S5 --out-dir /tmp/psmc-rs-s5-smoke
```

## Tests

```bash
python3 -m unittest discover -s experiment/tests -v
```

They need no extra packages; the rendering and msprime tests skip when
numpy/pandas/matplotlib or msprime are missing.
