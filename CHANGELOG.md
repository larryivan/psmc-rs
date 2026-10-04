# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Library and CLI changes: to be completed at integration.

### Added

- Reproduction package: the msprime/stdpopsim simulation scripts and the
  supplementary-table runner deleted in 79ac364 are restored under
  `experiment/scripts/`, and two new runners, `run_main_text.py` and
  `run_real_data.py`, regenerate the main-text and real-data outputs that the
  figures are drawn from (previously produced by deleted notebooks).
- `experiment/requirements.txt` listing the Python packages of the
  reproduction package.
- `experiment/run.py --keep-going` draws every figure whose inputs are present
  and reports the rest (a figure whose code fails for any reason, such as a
  sample missing from the fit summary, is skipped with its error).
- `tests/c_equivalence.rs`: regression test against outputs of the original
  C psmc (lh3/psmc 0.6.5-r74).
- Python tests for the reproduction package (`experiment/tests/`, run with
  `python3 -m unittest discover -s experiment/tests`).
- `CONTRIBUTING.md`, issue templates, a pull-request template and
  `CITATION.cff`.
- This changelog.

### Changed

- The experiment scripts compare psmc-rs and C psmc from the same initial
  theta/rho ratio, 4 (C `-r4`, the psmc-rs 0.2 default `rho0 = theta0/4`);
  the 0.1 results used 5 on both sides. They need a psmc-rs build with the
  0.2 default: before the first run compared with C they read the starting
  ratio of the psmc-rs binary (zero-iteration run) and stop if it differs
  from `RHO_T_RATIO`.
- `run_supplementary.py` writes its quick-look figures to
  `diagnostic_figures/` instead of next to the manuscript figures, runs
  S2-S5 by default (S1, the smoothing ablation, on request) and simulates only
  the constant model for the S3 length series.
- `experiment/README.md` and Supplementary Note 1 document the whole pipeline
  (building lh3/psmc with `-O1`, simulation, matched runs, tables, figures)
  and state which results are packaged and which must be regenerated.

### Fixed

- Gzip input with several members (bgzip: `.vcf.gz` from bcftools/GATK/tabix,
  bgzipped psmcfa and BED masks) is read in full instead of only its first
  block.
- `experiment/run.py` no longer re-executes a hardcoded personal Python
  interpreter; it names missing packages for the interpreter it runs under,
  needs no third-party packages for `--skip-plot`, and explains how to
  regenerate missing results instead of failing with a bare
  `FileNotFoundError`.
- The plotting notebook and every experiment script locate the repository
  independently of the working directory (`PSMC_RS_ROOT` is passed by
  `run.py`).
- `.gitignore` no longer excludes the small psmc-rs JSON and C `.psmc`
  outputs the figures are drawn from; `outputs/` directories exist in a clone.
- Absolute personal paths removed from the real-data fit summary (now
  repository-relative), the experiment README and the UI-assets README.
- The Papuan sample, which contains chromosome 1 only, is labelled
  "Homo sapiens (Papuan Highlands, chr1)" in the notebook and the tables.
- `run_supplementary.py` S3 re-runs its timed runs every time, as S4 does. A
  second run into the same directory used to reuse the first run's outputs,
  write empty timings to `S3_memory_scaling_summary.csv` (Figure 4C) and then
  crash while plotting. The S3/S4 timing repeats are no longer tracked by git.
- An interrupted experiment run no longer leaves a partial C psmc output that
  the next run reuses (C psmc writes each EM round as it goes; the RMSE was
  then computed from an early round without warning). C psmc now writes to
  `<output>.partial`, renamed on success, and cached outputs are reused only
  when they hold `N_ITER + 1` rounds (psmc-rs JSON: only when it parses).

### Known issues

- All psmc-rs results in `experiment/runs/` were produced with 0.1 and must be
  regenerated with the 0.2 fixes; the inference outputs behind Figures 2 and 3
  are not committed yet.
- The zigzag row of `S5_format_consistency.csv` is exactly zero in every
  column. This is not an artefact: S5's inputs differ only in the homozygous
  bins after the last heterozygous site (multihetsep and VCF end there), and
  the zigzag simulation has no such bins. The caption of Supplementary Figure
  S2 should say so; `run_supplementary.py` now records the count in
  `psmcfa_bins_after_last_het`.
- The committed Figure 3 files still show the old Papuan panel title.

## [0.1.2] - 2026-02-24

### Fixed

- Release workflow fixes; no change to the library or the CLI.

## [0.1.1] - 2026-02-24

First tagged release (the crate version in `Cargo.toml` still read 0.1.0).

### Added

- PSMC inference from psmcfa, multihetsep and VCF input (plain or gzip),
  JSON parameter output and an interactive HTML report.
- Per-bin TMRCA posterior decoding, block-bootstrap confidence intervals and
  a terminal user interface.
- Release binaries for Linux, macOS and Windows with SHA-256 checksums.

[Unreleased]: https://github.com/larryivan/psmc-rs/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/larryivan/psmc-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/larryivan/psmc-rs/releases/tag/v0.1.1
