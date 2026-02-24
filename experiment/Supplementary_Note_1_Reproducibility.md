# Supplementary Note 1 — Reproducibility Guide

This note provides a reviewer-facing, machine-independent procedure to reproduce all manuscript figures and tables from the archived `psmc-rs` repository.

## 1) Scope

The reproducibility package includes:

- precomputed inference outputs (`PSMC-RS` and C baseline)
- precomputed benchmark tables
- one-command plotting entrypoint

The command below regenerates:

- Main text: Figure 2, Figure 3, Figure 4, Table 1, Table 2
- Supplementary: Figure S1, Figure S2, Figure S3

## 2) System requirements

- OS: Linux/macOS (Windows via WSL is also acceptable)
- Python: `>=3.9`
- Python packages: `numpy`, `pandas`, `matplotlib`

No Rust/C compilation is required for figure regeneration from the packaged results.

## 3) Reproduce all figures (recommended path)

```bash
git clone https://github.com/larryivan/psmc-rs.git
cd psmc-rs
git checkout <MANUSCRIPT_TAG>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install numpy pandas matplotlib

./experiment/run.py
```

Notes:

- `experiment/run.py` first validates required result files and key tables, then executes `experiment/notebooks/all_plots_precomputed.ipynb` headlessly.
- Harmless `matplotlib` cache warnings may appear depending on local permissions.

## 4) Expected outputs

After completion, the following files are regenerated:

- `/experiment/runs/main_text/figures/figure_2_simulated_equivalence.(png|svg|pdf)`
- `/experiment/runs/real_data/figures/figure_3_empirical_genomes_rust_vs_c.(png|svg|pdf)`
- `/experiment/runs/main_text/figures/figure_4_performance_scalability.(png|svg|pdf)`
- `/experiment/runs/supplementary/figures/supplementary_figure_S1_em_convergence.(png|svg|pdf)`
- `/experiment/runs/supplementary/figures/supplementary_figure_S2_format_consistency.(png|svg|pdf)`
- `/experiment/runs/supplementary/figures/supplementary_figure_S3_interfaces.(png|svg|pdf)`

And tables:

- `/experiment/runs/main_text/tables/table_1_rmse_notitle.csv`
- `/experiment/runs/main_text/tables/table_2_runtime_memory.csv`
- `/experiment/runs/real_data/tables/real_data_rust_vs_c_fit_summary.csv`
- `/experiment/runs/supplementary/tables/S2_em_convergence_trace.csv`
- `/experiment/runs/supplementary/tables/S3_memory_scaling_summary.csv`
- `/experiment/runs/supplementary/tables/S4_thread_scaling_summary.csv`
- `/experiment/runs/supplementary/tables/S5_format_consistency.csv`

## 5) Optional: run checks only or plot only

```bash
# checks only
./experiment/run.py --skip-plot

# plot only
./experiment/run.py --skip-check
```

## 6) Optional: recompute one empirical sample from raw input

This step is optional and computationally heavier than figure regeneration. It is provided for method-level spot checks.

### 6.1 Build `psmc-rs`

```bash
cargo build --release --locked --bin psmc-rs
```

### 6.2 Run `psmc-rs` on one sample

```bash
./target/release/psmc-rs \
  ./experiment/real_data/Papuan_highlands_diploid.psmcfa \
  /tmp/HomoSapiens.rust.json \
  20 \
  --pattern '4+25*2+4+6' \
  --no-progress
```

### 6.3 (Optional) C baseline command

If C `psmc` is available in `PATH`:

```bash
psmc \
  -N20 -t15 -r5 -p '4+25*2+4+6' \
  -o /tmp/HomoSapiens.c.psmc \
  ./experiment/real_data/Papuan_highlands_diploid.psmcfa
```

## 7) Archive integrity

For release artifacts, verify checksums:

```bash
sha256sum -c SHA256SUMS.txt
```

(`shasum -a 256` can be used on macOS if `sha256sum` is unavailable.)
