# Experiment Workspace

This directory is the single workspace for manuscript experiments (scripts, notebooks, generated data, plots, and tables).

## Layout

```text
experiment/
├── README.md
├── scripts/
│   ├── simulate_msprime_to_psmcfa.py
│   ├── simulate_msprime_to_mhs.py
│   └── run_supplementary.py
├── notebooks/
│   ├── core_experiments.ipynb
│   ├── mhs_vs_psmcfa_vcf_500mb.ipynb
│   ├── main_text_6items.ipynb
│   └── supplementary_S1_S5.ipynb
└── runs/
    ├── core/
    │   ├── inputs/      # simulated inputs used by core benchmark
    │   ├── outputs/     # rust/c outputs, json/psmc/html/bootstrap artifacts
    │   ├── perf/        # repeated runtime/memory raw logs
    │   ├── figures/     # manuscript figures (png/pdf/svg)
    │   ├── tables/      # manuscript tables (csv/tsv)
    │   └── logs/        # command logs and metadata
    ├── format_consistency/
    │   ├── inputs/      # shared psmcfa/mhs/vcf inputs from same simulation
    │   ├── outputs/     # inference outputs per format
    │   ├── figures/     # comparison figures
    │   ├── tables/      # RMSE/runtime summary tables
    │   └── logs/        # run logs
    ├── main_text/
    │   ├── inputs/      # 500Mb simulated inputs for main-text figures
    │   ├── outputs/     # Rust/C main outputs
    │   ├── perf/        # repeated runtime/memory outputs
    │   ├── bootstrap/   # bootstrap replicates and summaries
    │   ├── figures/     # Figure 1-4 (png/svg/pdf)
    │   ├── tables/      # Table 1-2 (+ bootstrap CI width table)
    │   └── logs/        # command logs
    └── supplementary/
        ├── figures/     # S1-S5 figures (png/svg/pdf)
        ├── tables/      # S1-S5 tables (csv/tsv/md)
        ├── logs/        # command logs
        ├── shared_inputs/
        ├── s1_smooth/outputs/
        ├── s2_em/outputs/
        ├── s3_memory/outputs/
        ├── s4_threads/outputs/
        └── s5_format/{inputs,outputs}/
```

## Scope of each notebook/script

- `notebooks/main_text_6items.ipynb`
  - one-click generation of manuscript main-text 6 items (Figure 1-4, Table 1-2)
  - writes to `runs/main_text/*`
- `notebooks/supplementary_S1_S5.ipynb`
  - one-click generation of supplementary S1-S5 figures/tables
  - writes to `runs/supplementary/*`
- `scripts/run_supplementary.py`
  - CLI runner for supplementary S1-S5
  - example: `python experiment/scripts/run_supplementary.py --steps S1,S2,S3,S4,S5`
- `notebooks/mhs_vs_psmcfa_vcf_500mb.ipynb`
  - format-consistency experiment (`psmcfa` vs `mhs` vs `vcf`)
  - writes to `runs/format_consistency/*`
- `notebooks/core_experiments.ipynb`
  - extended Rust vs C validation/benchmarking
  - writes to `runs/core/*`

## Conventions

- Keep random seeds and CLI commands in notebook outputs and/or `runs/*/logs`.
- Keep manuscript-ready plots in `runs/*/figures` and tables in `runs/*/tables`.
- Treat `runs/*/inputs`, `runs/*/outputs`, `runs/*/perf`, and `runs/*/bootstrap` as regenerable artifacts.

## Repro run order

1. Build binary: `cargo build --release`
2. Run `notebooks/main_text_6items.ipynb`
3. Run `notebooks/supplementary_S1_S5.ipynb`
4. Run `notebooks/mhs_vs_psmcfa_vcf_500mb.ipynb`
5. Run `notebooks/core_experiments.ipynb` (optional extended analysis)
