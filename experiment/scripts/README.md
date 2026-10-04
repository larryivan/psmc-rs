# Experiment Scripts

Simulation and benchmark runners that produce the inputs, outputs and tables
used by `experiment/notebooks/all_plots_precomputed.ipynb`. See
`experiment/README.md` for the full pipeline.

| Script | Produces |
| --- | --- |
| `simulate_msprime_to_psmcfa.py` | one simulated diploid `psmcfa` (stdpopsim, msprime engine) |
| `simulate_msprime_to_mhs.py` | one simulated constant-size `multihetsep/mhs` (msprime) |
| `run_main_text.py` | `runs/main_text/`: simulated inputs, psmc-rs and C outputs (Figure 2, Table 1), timing tables (Figure 4 A/B, Table 2) |
| `run_real_data.py` | `runs/real_data/`: psmc-rs and C outputs on `experiment/real_data/` and the fit summary (Figure 3) |
| `run_supplementary.py` | `runs/supplementary/tables/`: S2-S5 (Supplementary Figures S1, S2 and Figure 4 C/D); S1 smoothing ablation optional |
| `exp_common.py` | shared helpers: repository paths, tool lookup, matched EM settings, command log, Ne curves |

All scripts locate the repository from their own path, so they can be run from
any working directory. Output locations are command-line arguments; tool
locations follow `flag > environment variable > default` (`--psmc-rs` /
`PSMC_RS_BIN`, default `target/release/psmc-rs`, built if missing;
`--c-psmc` / `C_PSMC_BIN`, default `psmc` on `PATH`). Run parameters are
environment variables (`N_ITER`, `SIM_LENGTH_BP`, `PERF_REPEATS`, ...), listed
in `experiment/README.md`. Every command, with its wall time and peak RSS, is
appended to `<out-dir>/logs/commands.jsonl`.

`run_main_text.py` and `run_real_data.py` turn the run logic of the notebooks
`main_text_6items.ipynb` and `real_data_rust_vs_c.ipynb` (deleted in commit
79ac364) into scripts; the other three scripts are the originals from that
commit's parent, made path independent.

These scripts are experiment utilities (not end-user CLI surface).
