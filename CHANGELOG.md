# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

0.2.0 rebuilds the E-step, the input readers and the bootstrap, follows
C psmc in its defaults, and adds the `qc` and `convert` commands. **Results
are not comparable with 0.1 runs that relied on the defaults** (first part
of *Changed*); the psmc-rs results packaged under `experiment/runs/` were
made with 0.1 and are being regenerated.

### Changed

Defaults, each of which changes the results of a run that relied on it:

- Without `--pattern` the model is C psmc's default pattern `4+5*3+4` (23
  intervals, 7 free λ); 0.1 fitted 64 intervals with one λ each (65 free
  parameters). `--n-steps` follows the pattern, and accepts the interval
  count that 0.1 took; `--n-steps` alone keeps one λ per interval.
- EM starts from ρ0 = θ0/4 (C psmc's `-r 4`, new `--tr-ratio`) instead of
  θ0/5.
- `--smooth-lambda` defaults to 0, C psmc's objective, instead of 1e-3.
- The bootstrap follows C psmc's `splitfa` and `psmc -b`: trunks of
  `--bootstrap-block-size` bins (default 500,000 instead of 50,000) are
  drawn with replacement across the whole input, and each replicate is
  fitted from a fresh start. 0.1 resampled blocks within each sequence and
  started every replicate from the main fit, so a sequence shorter than one
  block came back unchanged and the bands could have no width at all.
  Replicates draw from per-replicate ChaCha8 streams, so a
  `--bootstrap-seed` gives other replicates than in 0.1, and the same ones
  on every platform and with any `--threads`.
- `--input-format` defaults to `auto` (from the content) instead of
  `psmcfa`.
- VCF and multihetsep sequences go through fq2psmcfa's output filter
  (`--min-good-bases 10000`, `--min-good-fraction 0.2`; 0 disables).
- A VCF read in mask mode runs to each chromosome's `##contig` length
  instead of stopping at the last record (`--vcf-no-contig-end`).
- TMRCA output gives each state its average coalescent time, the time its
  emission uses, instead of the left edge of its interval; every
  `tmrca_*_years` column changes.
- `--bin-size` (formerly `--mhs-bin-size`, still accepted) also scales psmcfa
  input, which 0.1 always scaled at 100 bp; `--gen-years` (formerly
  `--tmrca-gen-years`, still accepted) also scales the report's curve and
  the bootstrap summary, which were fixed at 25 years.
- `Last EM loglike` is the log-likelihood of the written model (C psmc's LK
  of round N_ITER + 1); 0.1 printed the value before the last iteration.

Command line and outputs:

- Out-of-range model settings are usage errors before the input is read:
  `--t-max` must lie in [1, 200], the range the M-step searches, and
  `--n-steps` in 1 to 9999; a pattern must parse and agree with
  `--n-steps`.
- Output paths are checked before the input is read: each must be
  writable, none may be the input or another output, and `OUTPUT` may not
  end in `.html`.
- The output JSON is written right after the main fit, before the
  bootstrap, and carries run metadata (see *Added*).
- The HTML report labels the curve psmc-rs, scales it with the run's bin
  size and generation time, and has run summary, QC and bootstrap
  sections.
- `PSMC_ALPHA_CACHE_MB` is one budget for the whole process and is used
  only by models of more than 256 states; 0 now disables the cache instead
  of selecting the default (2048).
- `--threads` builds a thread pool per run; the TUI no longer keeps the
  first run's thread count.

Library API:

- `report::write_html_report(path, &ReportData)` replaces the positional
  form.
- `bootstrap::new_rng` returns `BootstrapRng` (ChaCha8) instead of
  `SmallRng`; `sample_sequences_block_bootstrap` takes any `Rng` and is kept
  as the 0.1 scheme.
- `EmHistory` gains `post_sigma`, `q`, `mstep_evaluations`,
  `mstep_iterations` and `mstep_stops` (`opt::MStepStop`, why each M-step
  stopped), and `loglike[i].1` is the log-likelihood after iteration i (it
  was a copy of `.0`); `em_train` returns the model with its matrices
  recalculated.
- `MStepConfig::default()` sets `lambda` to 0, C psmc's objective and the
  CLI's default, instead of 1e-3, so `em_train` and `run_bootstrap` with
  default settings no longer fit a penalised objective.
- `PsmcModel::new` rejects `n_steps` 0, single-state and overflowing
  patterns, more than `model::MAX_STATES` (10,000) states, and `t_max`
  outside (0, 1000).
- `read_psmcfa` fails on input without a `>` record; `read_vcf` fails on
  unsorted positions, a chromosome that comes back, a record beyond its
  `##contig` length and a genotype allele that ALT lacks.
- `hmm::write_tmrca_posterior_tsv` writes gzip when the path ends in `.gz`.
- `bootstrap::BootstrapSummary` gains `skipped_replicates` (the replicates
  left out for an invalid start), which breaks code that builds it with a
  struct literal.

Reproduction package (`experiment/`):

- The scripts pass every setting that shapes a fit to psmc-rs explicitly
  (`--pattern`, `--t-max`, `--tr-ratio`, `--smooth-lambda`,
  `--mstep-iters`) and compare it with C psmc at `-r4`; the 0.1 results
  used 5 on both sides. Before the first run compared with C they check,
  with a zero-iteration run of those options, that the binary starts from
  `theta0/RHO_T_RATIO`, and refuse a 0.1 build, which has no `--tr-ratio`.
  `--n-steps` and `N_STEPS` are gone (the pattern sets the intervals, as in
  C psmc), `--batch-size` is passed only when `BATCH_SIZE` is set, and
  `MSTEP_ITERS` sets the M-step cap.
- S2 reads the psmc-rs log-likelihood trace from one run's JSON
  (`loglike`) instead of running psmc-rs once per iteration count and
  parsing `Last EM loglike`, which now belongs to C psmc's next round; row
  `iter` k holds both tools' log-likelihood after k - 1 iterations (C's
  round k).
- S5 reads the VCF in mask mode, and `--bin-size` is passed for every
  format.
- `run_supplementary.py` writes its quick-look figures to
  `diagnostic_figures/` instead of next to the manuscript figures, runs
  S2-S5 by default (S1, the smoothing ablation, on request) and simulates
  only the constant model for the S3 length series.
- `experiment/README.md` and Supplementary Note 1 document the whole
  pipeline (building lh3/psmc with `-O1`, simulation, matched runs, tables,
  figures) and state which results are packaged and which must be
  regenerated.

### Added

- `psmc-rs qc INPUT`: a report of an input's suitability, without fitting:
  bin counts, θ0, heterozygosity, called length, N50, missing runs, bin
  saturation, read depth, per-sequence statistics, and findings with stable
  codes (`VARIANTS_ONLY_NO_MASK`, `HIGH_MISSING`, `BIN_SATURATION`,
  `FEW_HETS`, `TOO_SHORT`, `SHORT_SEQUENCES`, `SEX_OR_ORGANELLE`,
  `HET_OUTLIER`, `APPROX_MISSING_PLACEMENT`, `DEPTH`, `DROPPED_SEQUENCES`)
  and recommendations. `--json`, `--strict` (exit 2 on a warning) and
  `--bootstrap-block-size` (trunk checks). Every fit and conversion prints
  the same report and `--qc-json` writes it.
- `psmc-rs convert INPUT`: psmcfa as fq2psmcfa lays it out, and multihetsep
  as `generate_multihetsep.py` writes it, in one file or one per sequence
  (`--psmcfa`, `--multihetsep`, `--multihetsep-per-chrom`), gzip by name.
- VCF consensus mode: an all-sites VCF from `bcftools call -c` is read
  exactly as `vcfutils.pl vcf2fq | fq2psmcfa` would read it (`--depth
  auto|MIN:MAX`, `--min-mapq`, `--indel-window`, `--min-qual`), with
  automatic detection between consensus and mask mode (`--vcf-mode`).
- `scripts/bam_to_psmc.sh`: from the aligned reads of one diploid sample
  (BAM, CRAM or SAM) to psmc-rs input by the classic route: `bcftools
  mpileup -C50 -q20 -Q20 | bcftools call -c` per chromosome in parallel,
  one bgzipped and indexed all-sites VCF, `psmc-rs qc`, the psmcfa (the
  same bytes as `vcfutils.pl vcf2fq | fq2psmcfa`), and on request one
  multihetsep file per chromosome (`--multihetsep`) and a fit (`--run
  N_ITER`). It leaves out sex chromosomes and organelles by name, stops on
  reads of several samples, and a rerun writes the same bytes with any
  `--threads`.
- `docs/preprocessing.md`: how to get from aligned reads to psmc-rs input,
  by the classic consensus (route A, the script) or by variant calls and a
  callable mask (route B, which also feeds MSMC2 and SMC++), with
  `psmc-rs qc` and the pitfalls (missing masks, sex chromosomes and
  organelles, low coverage, repeats, bin size, scaling).
- VCF mask mode options `--vcf-keep-filtered` and `--vcf-no-contig-end`;
  the read depth of the sample in mask mode.
- `--mhs-pair I,J` selects the two haplotypes of a multihetsep file.
- `--psmc-out FILE`: C psmc's `.psmc` output (rounds 0 to N_ITER) for
  `psmc_plot.pl`; with `--bootstrap`, `replicate_NNN.psmc` for every
  replicate. The run prints the matching `psmc_plot.pl` command.
- `--init-params FILE` starts from the parameters of an earlier run (C
  `-i`), and its replicates start there too, as `psmc -i -b`.
- `--tr-ratio`, `--gen-years`, `--qc-json`, `-q`/`--quiet` and
  `--version`.
- Output JSON metadata: `version`, `n_iter`, `final_loglike`, `loglike`
  (the log-likelihood after each iteration), `bin_size`, `gen_years`,
  `tr_ratio`, `smooth_lambda`, `input` and `qc`. Files without them (0.1)
  still load.
- HTML report: run summary, QC findings and per-sequence table, bootstrap
  settings with a coverage caveat, and an offline notice when ECharts
  cannot load (`PSMC_ECHARTS_JS` embeds a local copy).
- Warnings for input options that do not apply to the input as read and
  for bootstraps of fewer than 20 trunks; a single-trunk bootstrap is an
  error before fitting.
- Input from pipes and FIFOs (the file is read once; only `--depth auto`
  needs a regular file), and a directory as INPUT.
- TUI: the stop key interrupts EM and the bootstrap mid-step, the form has
  a generation-time field and shows the equivalent command line.
- Library: `io::input::load` and `Dataset` (one loader for every format),
  `io::consensus`, `io::write` (psmcfa and multihetsep writers),
  `io::mhs::read_mhs_with`, `io::vcf::read_vcf_with`, the `qc` and
  `psmc_out` modules, `em::em_train_with_options` with `EmOptions`
  (cancellation, explicit thread pool), `em::{hmm_q, hmm_q0,
  posterior_sigma}`, `hmm::{log_likelihood, e_step_with_options,
  EStepOptions, AlphaCacheBudget, Cancelled}`, `PsmcModel::{avg_t,
  n0_for_bin, ne_curve, state_tmrca_years, c_pi, c_sigma, n_recomb}`,
  `model::{n_steps_for_pattern, T_INF, MAX_STATES}`, `bootstrap::{
  run_bootstrap, run_bootstrap_with_options, BootstrapConfig,
  BootstrapOptions, split_for_bootstrap, resample_chunks,
  bootstrap_trunk_count, replicate_rng}` and
  `opt::m_step_lbfgs_with_outcome`.
- `examples/bootstrap_calibration.rs`, the simulation behind the bootstrap
  coverage figures, and `examples/bench_estep.rs`.
- Tests against the original tools' outputs: `tests/c_equivalence.rs` (C
  psmc 0.6.5-r74), the `.psmc` output and `psmc_plot.pl`, psmcfa parsing,
  route A goldens from bcftools, `vcfutils.pl`, `fq2psmcfa` and
  `generate_multihetsep.py`, `splitfa` and `psmc_resamp`; and an
  end-to-end test of `scripts/bam_to_psmc.sh` (`tests/pipeline/`), which
  runs it on simulated reads and compares its psmcfa with `vcf2fq |
  fq2psmcfa` and with the simulation, and runs route B with the guide's
  commands.
- CI: formatting, clippy with warnings denied, tests on Linux and macOS, a
  Rust 1.88 build, the Python tests of the reproduction package, and a
  `pipeline` job that checks the shell scripts with shellcheck, runs the
  end-to-end test with Ubuntu's bcftools and `vcfutils.pl` and with
  `fq2psmcfa` built from lh3/psmc 0.6.5-r74, and runs the ignored tests
  that compare consensus and mask mode with those tools.
- Reproduction package: the simulation scripts and the supplementary-table
  runner deleted in 79ac364 are restored under `experiment/scripts/`, with
  new runners `run_main_text.py` and `run_real_data.py`;
  `experiment/requirements.txt`; `experiment/run.py --keep-going`; Python
  tests (`python3 -m unittest discover -s experiment/tests`).
- `CONTRIBUTING.md`, issue and pull-request templates, `CITATION.cff` and
  this changelog.

### Fixed

- Gzip input with several members (bgzip: `.vcf.gz` from bcftools, GATK or
  tabix, bgzipped psmcfa and BED masks) is read in full instead of only its
  first block.
- psmcfa: header lines were read as bins, so names containing T, K or N
  (GenBank, RefSeq and UCSC accessions) added bins (HLpelCas1: 523 extra
  T bins), and every byte other than T, K and N was dropped, shifting later
  bins. Records and bytes are now read as C psmc reads them.
- Multihetsep: a site was heterozygous when any two haplotypes differed, so
  multi-sample files gave a K bin at almost every row; now only the
  selected pair counts, and an unknown allele (`?`, `N`) is missing instead
  of heterozygous. An out-of-range haplotype index panicked, and a row
  widened by an indel silently shifted the pair; both are errors now.
- VCF mask mode classifies every base: 0.1 moved the missing sites of each
  interval between heterozygous SNPs together, which got 14.7% of bins
  wrong at 20% random missingness, counted missing genotypes as callable,
  aborted on two records at one position, dropped everything after the
  last heterozygous SNP and started a new sequence when a chromosome came
  back. FILTER was ignored and heterozygous indels counted as heterozygous
  SNPs; filtered records and missing genotypes are now missing at their
  position, heterozygous non-SNPs over their REF span.
- The time grid ends at C psmc's PSMC_T_INF = 1000 instead of 1e300, which
  changed the last state's emission when its λ was large.
- The first L-BFGS step of each M-step started at unit step length, which
  on whole-genome inputs needed all 20 halvings and failed (the M-step then
  took another path); it now starts at unit distance.
- The E-step had three code paths whose results depended on the thread
  count and the cache budget (up to 1% in the parameters after 20
  iterations, more than the difference from C psmc); results are now
  bit-identical for any thread count and budget.
- `EmHistory` never recorded the log-likelihood of the final parameters.
- Saved parameters reload bit for bit (exact JSON float parsing).
- A closed stdout (`psmc-rs ... | head -1`) aborted the run before any file
  was written.
- Output files with the same name overwrote each other (`OUTPUT fit.html`
  was replaced by the report), and a bad output path failed only after the
  fit and the bootstrap.
- The TUI form could not take q, x, r or a space, and its stop key could
  not interrupt EM.
- The HTML report was blank, without a word, when ECharts could not be
  fetched; it now says how to fix that.
- The HTML report stacked the bootstrap band on years instead of Ne, so
  the band it drew had nothing to do with the replicates' quantiles.
- The bootstrap summary (`summary.tsv`, `summary.json`, the report's band)
  took each replicate's Ne at the main curve's first time point from its
  state 1 when the replicate's own state 1 began later; without a pattern
  (0.1's default), or with a first pattern group of one interval, states
  0 and 1 differ, and that first row could be wrong several-fold.
- Release checksums named `dist/` paths, so `sha256sum -c` failed next to
  the downloaded assets.
- Reproduction package: `run.py` no longer re-executes a hardcoded personal
  interpreter and explains how to regenerate missing results; the
  notebook and every script work from any directory and clone; small
  inference outputs are no longer ignored by git; personal paths are gone
  from tables and docs; the Papuan sample is labelled chromosome 1 only.
- Reproduction package: S3 re-ran nothing on a second invocation and wrote
  empty timings; an interrupted C psmc run left a partial output that was
  reused; `--models` and `--samples` subsets dropped the other rows of
  Table 2 and of the fit summary; `run_main_text.py` ignored `$SIM_SCRIPT`.
- Reproduction package: S2 plotted C psmc's round 0, whose LK is a
  placeholder 0.
- Reproduction package: cached outputs were reused whatever settings made
  them (a C psmc output only needed `N_ITER + 1` rounds, a psmc-rs output
  only had to parse), so a rerun over the 0.1 outputs compared new psmc-rs
  fits with C psmc runs made at `-r5`. An output is now reused only when C
  psmc's header, or the record the scripts keep next to a psmc-rs JSON
  (`<name>.settings.json`), shows the current settings; otherwise the
  script says why and runs it again.
- The documented minimum Rust version is 1.88, which the locked
  dependencies of `ratatui` 0.30 need.

### Performance

- The E-step works in O(states) per bin instead of O(states²) for PSMC
  transition matrices (their triangles are rank one), with NEON kernels on
  aarch64, and keeps forward and backward values every 128 bins instead of
  whole matrices. On the human chromosome 1 in `experiment/real_data` (one
  sequence of 2.49 M bins, `4+25*2+4+6`, 20 iterations, 1 thread, Apple M1
  Pro) a fit takes 25 s and 21 MB instead of 139 s and 1.95 GB; on
  HLhydTec1 (34 sequences, 18.5 M bins, 5 iterations, 8 threads) 9.9 s and
  96 MB instead of 79 s and 6.9 GB.
- TMRCA decoding runs in parallel within and across sequences (30,000
  sequences of 50-300 bins at 8 threads: 10.8 s to 0.86 s) and formats
  numbers without the standard library's float formatter.
- psmcfa input is streamed instead of read whole, and the VCF reader is two
  to three times as fast.

### Deprecated

- `PsmcModel::n0`, which leaves out the bin size (θ is per bin); use
  `PsmcModel::n0_for_bin`.

### Removed

- `psmc_rs::utils::logsumexp` (unused).
- The CLI's within-sequence bootstrap (the library function remains).
- Reproduction package: the `N_STEPS` variable and the per-iteration S2 runs
  (`*.rust.iterNN.json`, `*.loglike.txt`).

### Known issues

- All psmc-rs results in `experiment/runs/` were produced with 0.1 and must
  be regenerated with 0.2; the inference outputs behind Figures 2 and 3 are
  not committed yet.
- The packaged S2 table and Supplementary Figure S1 include C psmc's round
  0, whose LK is a placeholder 0, which squeezes the curves into a line;
  the regenerated table leaves it out.
- The zigzag row of `S5_format_consistency.csv` is exactly zero in every
  column. This is not an artefact: with 0.1, S5's inputs differed only in
  the homozygous bins after the last heterozygous site, and the zigzag
  simulation has none. With 0.2 the VCF input gives exactly the psmcfa's
  bins, so a regenerated table has zero VCF columns for every model. The
  caption of Supplementary Figure S2 should say what S5 measures;
  `run_supplementary.py` records the count in `psmcfa_bins_after_last_het`.
- The committed Figure 3 files still show the old Papuan panel title, and
  the screenshots behind Supplementary Figure S3 (`experiment/assets/ui/`)
  show the 0.1 terminal interface and report.

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

[0.2.0]: https://github.com/larryivan/psmc-rs/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/larryivan/psmc-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/larryivan/psmc-rs/releases/tag/v0.1.1
