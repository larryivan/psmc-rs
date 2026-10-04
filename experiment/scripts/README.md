# Experiment Scripts

Simulation and benchmark runners that produce the inputs, outputs and tables
used by `experiment/notebooks/all_plots_precomputed.ipynb`. See
`experiment/README.md` for the full pipeline.

- `simulate_msprime_to_psmcfa.py`: simulate a demographic model with stdpopsim
  (msprime engine) and write `psmcfa`
- `simulate_msprime_to_mhs.py`: simulate a constant-size model with msprime and
  write official `multihetsep/mhs`
- `run_supplementary.py`: produce the supplementary tables S2-S5 (and the
  optional S1 smoothing ablation)
- `exp_common.py`: shared helpers (repository paths, tool lookup, command log)

All scripts locate the repository from their own path, so they can be run from
any working directory. Output locations are command-line arguments; tool
locations follow `flag > environment variable > default` (`--psmc-rs` /
`PSMC_RS_BIN`, default `target/release/psmc-rs`; `--c-psmc` / `C_PSMC_BIN`,
default `psmc` on `PATH`).

These scripts are experiment utilities (not end-user CLI surface).
