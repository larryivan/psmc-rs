# Experiment: reproduction package

Everything needed to reproduce the manuscript's figures and tables: the real
genomes, the scripts that simulate data and run `psmc-rs` and the original C
`psmc`, the result tables, and the notebook that draws the figures. All
commands below are run from the repository root, but every script locates the
repository from its own path, so any working directory works.

## Read first: state of the packaged results

- **All psmc-rs results must be regenerated after the 0.2 fixes.** The tables
  and figures in `runs/` were produced with psmc-rs 0.1. Version 0.2 changes
  defaults (initial `rho0 = theta0/4` instead of `theta0/5`, as in lh3/psmc,
  and the smoothing default), the numerics, and the bootstrap. Every psmc-rs
  number in Figures 2-4, Tables 1-2 and Supplementary Tables S2-S5 therefore
  describes 0.1. The C psmc runs were made with
  `-r5` to match 0.1; for a matched comparison with 0.2 they are re-run with
  `-r4` (the scripts' default, `RHO_T_RATIO`), so regenerate both sides.
- **Inference outputs are not committed yet.** `runs/main_text/outputs/` and
  `runs/real_data/outputs/` (psmc-rs JSON, C `.psmc`), from which Figures 2
  and 3 and Table 1 are computed, were excluded by `.gitignore` and never
  committed. The ignore rules now keep them, so commit them after
  regenerating. Until then `run.py` lists them as missing together with the
  command that regenerates them, and `run.py --keep-going` draws the figures
  whose inputs are present (Figure 4 and Supplementary Figures S1-S3).
- **`S5_format_consistency.csv` needs regeneration.** Its zigzag row is
  exactly 0.0 in every column while the other models are not, which points to
  a stale or mis-cached run rather than a real result (a 2 Mb smoke run of the
  same code gives non-zero zigzag differences). Supplementary Figure S2 is
  drawn from this table and must not be used until S5 is re-run.
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
the small `runs/**/outputs/*.json`, `*.psmc` and `*.loglike.txt` files.
Ignored as regenerable: simulated inputs (5 MB per 500 Mb model, more for
VCF), timing repeats (`main_text/perf/`, `supplementary/s3_memory/outputs/`,
`supplementary/s4_threads/outputs/`), logs, HTML reports, bootstrap
directories, `diagnostic_figures/` and the `.tsv`/`.md` copies of the tables.

## Requirements

- Rust stable (edition 2024, Rust 1.85 or later) for `psmc-rs`.
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
(recombination rate 1.25e-8) written as psmcfa, multihetsep and VCF.

### 2. Run psmc-rs and C psmc

```bash
python3 experiment/scripts/run_main_text.py --steps main,perf  # Figure 2, Table 1; Table 2
python3 experiment/scripts/run_real_data.py                    # Figure 3
python3 experiment/scripts/run_supplementary.py                # S2-S5 (S1 with --steps S1)
```

Both tools get the same EM settings:

| setting | psmc-rs | C psmc | variable |
| --- | --- | --- | --- |
| EM iterations | `N_ITER` positional (20) | `-N20` | `N_ITER` |
| max coalescent time | `--t-max 15` | `-t15` | `T_MAX` |
| free-parameter pattern | `--pattern 4+25*2+4+6` | `-p 4+25*2+4+6` | `PATTERN` |
| initial theta/rho ratio | default `rho0 = theta0/4` (0.2) | `-r4` | `RHO_T_RATIO` |
| smoothing penalty | `--smooth-lambda 0` | none | `SMOOTH_LAMBDA` |
| threads | `--threads 1` | 1 | `RUST_THREADS` |

plus `--n-steps 64 --batch-size 300000 --mu <mu> --no-progress` for psmc-rs
(`--mu` only scales the HTML report). `RHO_T_RATIO` must equal the psmc-rs
default ratio, otherwise the two tools start from different parameters. The
scripts reuse finished runs: a psmc-rs JSON that parses, and a C psmc output
with exactly `N_ITER + 1` rounds. C psmc writes each round as it goes, so it
writes to `<output>.partial`, which is renamed only when the run succeeds; an
interrupted run therefore leaves no output, and an incomplete one (or one made
with another `N_ITER`) is run again. `--force-run` (main text), `--force`
(real data, supplementary) re-run everything. Timing steps (`perf`, S3, S4)
always re-run because cached outputs carry no timing.

Expect hours: on the 0.1 runs C psmc needed about 7.5 min and 5 GB RSS per
500 Mb model (7.8 GB at 1 Gb in S3), psmc-rs about 5 min and 1.4 GB; `perf`
repeats each run 10 times and S2 re-runs psmc-rs for 1..20 iterations. Run
the timing steps on an otherwise idle machine and report its hardware.

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

- EM settings: `N_ITER` [20], `T_MAX` [15], `N_STEPS` [64], `PATTERN`
  [`4+25*2+4+6`], `RHO_T_RATIO` [4], `BATCH_SIZE` [300000], `MU` [2.5e-8],
  `SMOOTH_LAMBDA` [0] and `RUST_THREADS` [1] (main text, real data),
  `RUST_THREADS_BASE` [1] (supplementary), `PSMC_ALPHA_CACHE_MB` [2048].
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
