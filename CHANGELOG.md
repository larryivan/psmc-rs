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
  intervals, 7 free λ); 0.1 fitted 65 intervals (`--n-steps 64`) with one λ
  each. `--n-steps` follows the pattern, and accepts the interval count
  that 0.1 took; `--n-steps` alone keeps one λ per interval, so
  `--n-steps 64` gives 0.1's layout.
- EM starts from ρ0 = θ0/4 (C psmc's `-r 4`, new `--tr-ratio`) instead of
  θ0/5.
- `--smooth-lambda` defaults to 0, C psmc's objective, instead of 1e-3.
- Each M-step runs until it converges, as C psmc's Hooke-Jeeves search
  does, instead of stopping after 100 L-BFGS iterations; `--mstep-iters`
  (default 1000 instead of 100) is only a cap, which normal fits do not
  reach (10 to 20 iterations with C psmc's patterns, up to about 70 with
  one λ per interval). The 100-iteration M-steps stopped short of the
  maximum of Q, and EM carried the shortfall through the rounds: on a
  simulated 4 × 25 Mb bottleneck genome (`-N25 -t15 -r5 -p 4+25*2+4+6`)
  the curve was 0.034 (RMSE of log10 Ne over the informative window) from
  C psmc's, and is now 7.7e-5 from it, and 5e-7 from a fit whose M-steps
  run to the precision limit. The M-steps no longer differ: C's
  Hooke-Jeeves search, ported in place of the L-BFGS, gives a curve 7.5e-5
  from C's. Nearly all the rest comes from C's expected counts, which leave
  out each sequence's last bin: with that bin left out of psmc-rs's counts
  too, the curve is 6.4e-6 from C's, and on the reference data rounds 2 to
  5 agree with C's log-likelihoods to a relative 2.8e-9 instead of 7.3e-8.
  Without a pattern (one λ per interval) the capped M-steps had let the fit
  drift away from C's in every interval.
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
- VCF input is fitted with the callable bases of each bin
  (`--bin-callable auto`, new): a called bin of `c` callable bases out of
  `s` emits as `c` bases of sequence, `P(T | k) = exp(-θ (c/s) avg_t[k])`,
  instead of as a full bin, as fq2psmcfa's psmcfa and C psmc treat every
  called bin. With missing data scattered in runs shorter than a bin,
  treating such bins as full biased θ, N0, the times and Ne low by about
  the missing share of the called bins; on simulated genomes with 30% of
  the bases missing in runs of about 20 bp the RMSE of log10 Ne against
  the truth goes from 0.453 (ignored) to 0.309 (complete data: 0.307; see
  `experiment/validation/README.md`). The E-step, the M-step, the starting
  θ0 (which solves `sum (1 - exp(-θ0 c/s)) = K` over called bins, C's
  −ln(1 − K/L) for full bins), the bootstrap and the TMRCA decoding all
  use the counts. Complete data, and psmcfa input, which
  records no counts, are fitted exactly as before, to the bit;
  `--bin-callable ignore` fits C psmc's model on any input. Multihetsep
  does not say which sites are uncalled, so the counts of its bins follow
  the reader's placement of them; they made the fit worse in the
  validation, so `auto` does not use them and fits multihetsep input as C
  psmc would fit its bins, and `--bin-callable approximate` fits them. See
  `docs/design/bin-callable-model.md`.
- A VCF read in mask mode runs to each chromosome's `##contig` length
  (`--vcf-no-contig-end`: to its last record) instead of stopping at the
  last heterozygous SNP.
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
  end in `.html`. Only the QC JSON (`--qc-json`, `qc --json`) can be `-`,
  for stdout. `convert` checks its per-sequence multihetsep files, named
  after the sequences, once the input is read and before anything is
  written, and the `PSMC_ECHARTS_JS` bundle is read before the input too.
- Usage errors found after parsing (an invalid `--pattern`, outputs that
  name one file, `--bootstrap-iters` without `--bootstrap`, multihetsep
  output of psmcfa input, `--tui --qc-json -`, ...) exit with status 2, as
  clap's own do; errors reading or writing files exit with 1.
- The output JSON is written right after the main fit, before the
  bootstrap, and carries run metadata (see *Added*).
- For VCF and multihetsep input, the `.psmc` output has one more `MM` line
  after `n_seqs`, `MM bin_model:..., counts:..., mean_callable:...,
  partial_bins:...`, which says how the fit used the callable bases of each
  bin (psmc_plot.pl skips it; psmcfa input gets none), and so do bootstrap
  replicates fitted with the counts. `convert --psmcfa` warns when it
  writes partly callable bins of a VCF, which psmcfa cannot record, and
  advises fitting the VCF itself.
- QC compares the heterozygosity of sequences per callable base when the
  input records callable bases (the per-sequence table's het/kb and
  `vs ref`, the autosomal reference, `SEX_OR_ORGANELLE` and `HET_OUTLIER`),
  so that a sequence with more missing data does not look less
  heterozygous; `HIGH_MISSING` says when the fit corrects the bias of
  partly called bins, and `APPROX_MISSING_PLACEMENT` that multihetsep's
  counts follow its placement of the uncalled sites. The start that `qc`
  and `convert` check (`INVALID_START`) takes θ0 from the callable bases
  of each bin when a fit with `--bin-callable auto` does (VCF).
- A bootstrap starts by removing every `replicate_NNN.json`,
  `replicate_NNN.psmc`, `summary.tsv` and `summary.json` in its directory,
  with a warning that names the earlier replicates beyond its own count,
  so that a rerun with fewer replicates, or one that stops part way,
  leaves no earlier replicates for `cat replicate_*.psmc` or a glob of the
  JSON files to pick up; other files there are left alone. A fit without
  `--bootstrap` warns when `OUTPUT.bootstrap` holds replicates.
- The HTML report labels the curve psmc-rs, scales it with the run's bin
  size and generation time, and has run summary, QC and bootstrap
  sections.
- `PSMC_ALPHA_CACHE_MB` is one budget for the whole process and is used
  only by models of more than 256 states; 0 now disables the cache instead
  of selecting the default (2048).
- `--threads` builds a thread pool per run; the TUI no longer keeps the
  first run's thread count.
- BCF input is an error that gives the `bcftools view -Oz` command to
  convert it, and a UTF-8 byte order mark at the start of an input is
  skipped. When a file recognised as multihetsep does not read as one from
  its start, the error says how its format was recognised and names
  `--input-format`.

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
- The M-step stops by a scale-free convergence test: `MStepConfig::rel_tol`
  (1e-13; the objective's decrease and the next quasi-Newton step's
  predicted decrease, relative to the objective, over two iterations)
  replaces `tol_grad`, an absolute gradient norm that grew with the input,
  and `MStepStop::Converged` replaces `GradientTolerance`.
  `MStepConfig::max_iters` defaults to `opt::DEFAULT_MSTEP_ITERS` (1000)
  and `lbfgs_m` to 20 (7). The L-BFGS initial matrix is the inverse of the
  objective's curvature in each coordinate (one objective evaluation per
  parameter at the start of each M-step) instead of a multiple of the
  identity; curvature pairs are kept when `s'y` is positive relative to
  `|s| |y|` (it was compared with 1e-12, a scale that changes with the
  input); and the line search backtracks by interpolation and, when it
  fails, restarts along the preconditioned gradient, instead of trying the
  reverse direction and a coordinate search.
- Before an M-step stops, it takes fresh curvatures: it moves each
  parameter along which the objective does not curve upwards alone to the
  best of a ladder of moves, and restarts with an empty memory when that
  or the diagonal Newton step of the others promises more than the
  convergence threshold (at most three times). Trial points keep each
  parameter within 1e-12 of its range from its bounds, the margin of the
  start. Without that, weakly determined λ carried past their optimum
  stopped M-steps short (the first λ of a simulated genome at 7,387 with Q
  peaking at 23 along it, 3.1e-2 higher), and the logit coordinates of
  parameters pushed against a bound ran off to -57.
- The M-step's objective is infinite wherever the HMM computed in f64 is
  not one, so that the search never moves there: a probability negative
  or not a number, or a row of transition probabilities whose sum
  differs from 1 by more than 1e-9 (or by more than the start's own
  deviation, when that is larger). A small λ of an early interval makes the
  survival underflow in the squares of C's formulas (below about 1e-154),
  where rows summed to up to 1.014 and Q exceeded its maximum Q0; on a
  simulated genome fitted with one λ per interval an M-step moved there,
  and the next E-step lowered the log-likelihood by 367.
- The last `opt::MStepProgress` event of each M-step reports the
  iterations it took as its `total` (earlier events report the cap,
  `max_iters`), and the M-step's progress bar ends there: converged
  M-steps take 10 to 70 iterations, so bars drawn against the cap of 1000
  ended at about 1%.
- `MStepConfig::lam_hi` defaults to 1e4 instead of 1e6. Over an early
  interval of length τ, C's formulas cancel to about τ² / (2λ), so f64
  holds them to a relative 2.2e-16 (λ/τ)²: at λ = 1e6 the first of 64
  intervals (τ = 0.0083) gets probabilities that are rounding noise,
  negative at most points near any valid one. EM drove recent λ to 1e6
  and then stopped moving (a simulated 4 × 25 Mb genome with 101
  intervals, from round 27; a 5 Mb one with 64, from round 16); at 1e4 the
  error is 3e-4, and below 1 up to about 1,600 intervals. λ = 1e4 already
  means no coalescence in the interval: holding such λ at 1e4 instead of
  1e6 lowered the maximum of Q by 1e-5 to 7e-4 on the simulations' counts.
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
- The callable bases of each bin (`docs/design/bin-callable-model.md`).
  Input: the new `io::callable` module holds `CallableBases` (the counts
  of a list of rows or sequences), `BinCounts` (of one, kept as runs of
  partly callable bins or one byte per bin, whichever is smaller),
  `CallableSummary` (their mean share and distribution over the called
  bins) and `MAX_COUNTED_BIN_SIZE` (65,535), all re-exported from `io`,
  with the counts' iterator `callable::Iter`; `Observations::validate`
  checks rows, starts and counts together. `Observations` gains the public
  field `callable` and `io::consensus::ConsensusSeq` the field `callable`,
  which breaks code that builds them with a struct literal (add
  `callable: None`). The new `io::Sequences` holds whole sequences with
  optional counts (`From<Vec<Vec<u8>>>` for bins without them,
  `Sequences::validate`), and `io::common::chunk_sequences` takes `impl
  Into<Sequences>` instead of `Vec<Vec<u8>>`, so an un-annotated
  `.collect()` passed to it no longer compiles. `io::input::load`,
  `io::vcf::read_vcf_with` (and `read_vcf`) and `io::mhs::read_mhs_with`
  (and `read_mhs`) return the counts for bins of at most 65,535 bases;
  `io::common::Binner::finish_with_callable` and
  `io::input::Dataset::sequence_callable` are new.
- Model: `hmm::EStepOptions::callable` runs the E-step and
  `log_likelihood_with_options` on the counts, and the E-step returns
  `SufficientStats::callable` (`hmm::CallableStats`: the count-weighted `T`
  sums and the `K` sums by count), which breaks code that builds
  `SufficientStats` with a struct literal. `hmm::TmrcaOptions` and
  `write_tmrca_posterior_tsv_with_options` decode with them.
  `em::EmOptions::callable` fits them, the M-step weighs each count's `K`
  bins, and `em::hmm_q` and `hmm_q0` evaluate Q and its maximum of the
  callable-length model. `initial_theta_with_callable` and
  `model::ThetaCounts` give the starting θ0. Without counts, or with every
  called bin full, every one of these gives C psmc's results to the bit;
  `e_step_streaming`, `log_likelihood`, `write_tmrca_posterior_tsv`,
  `em_train` and `initial_theta` keep their signatures and C's model.
- Bootstrap: `observations_to_sequences`, `split_sequences_for_bootstrap`,
  `resample_sequences` and `run_bootstrap_sequences` carry the counts with
  their bins and draw the same trunks as without them; the replicates of
  `run_bootstrap_sequences` are fitted with them, from
  `replicate_initial_model_with_callable`, and
  `invalid_replicate_starts_sequences` finds the replicates it skips.
  `BootstrapReplicate` gains `callable`.
- Outputs: `io::params::RunMetadata::bin_callable`
  (`io::params::BinCallableRecord`, with `CALLABLE_LENGTH_MODEL` and
  `FULL_BINS_MODEL`), `psmc_out::PsmcRunInfo::bin_callable` and
  `psmc_out::bin_callable_line`, `report::ReportData::bin_callable` and
  `report::bin_callable_rows`; struct literals of these types need the new
  fields.
- QC: `qc::QcInput::callable` (struct literals need `callable: None`),
  `QcContext::counts_ignored`, `QcReport::callable`,
  `BinStats::{theta0_callable, mean_callable_fraction,
  partly_callable_fraction, called_next_to_missing}` with
  `theta0_comparable` and `het_per_kb_comparable`,
  `QcCode::PartlyCallableBins` (between `VariantsOnlyNoMask` and
  `HighMissing` in report order) and `PARTLY_CALLABLE_NOTE_FRACTION`,
  `PARTLY_CALLABLE_WARN_MEAN`.

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
  saturation, read depth, the callable bases of the called bins (VCF and
  multihetsep), per-sequence statistics, and findings with stable
  codes (`INVALID_START`, `VARIANTS_ONLY_NO_MASK`, `PARTLY_CALLABLE_BINS`,
  `HIGH_MISSING`, `BIN_SATURATION`, `FEW_HETS`, `TOO_SHORT`,
  `SHORT_SEQUENCES`, `SEX_OR_ORGANELLE`, `HET_OUTLIER`,
  `APPROX_MISSING_PLACEMENT`, `DEPTH`, `DROPPED_SEQUENCES`) and
  recommendations. `PARTLY_CALLABLE_BINS` reports how much of the called
  bins of a VCF is callable and the bias of treating them as full: a note
  when the fit corrects it, a warning when `--bin-callable ignore` leaves
  called bins less than 90% callable on average, and for psmcfa input,
  which cannot tell, a note with the called bins next to missing runs and
  how to measure the rest. `--json`, `--strict` (exit 2
  on a warning) and `--bootstrap-block-size` (trunk checks). Every fit and
  conversion prints the same report and `--qc-json` writes it.
- `psmc-rs convert INPUT`: psmcfa as fq2psmcfa lays it out, and multihetsep
  as `generate_multihetsep.py` writes it, in one file or one per sequence
  (`--psmcfa`, `--multihetsep`, `--multihetsep-per-chrom`), gzip by name.
- VCF consensus mode: an all-sites VCF from `bcftools call -c` is read
  exactly as `vcfutils.pl vcf2fq | fq2psmcfa` would read it (`--depth
  auto|MIN:MAX`, `--min-mapq`, `--indel-window`, `--min-qual`), with
  automatic detection between consensus and mask mode (`--vcf-mode`). It
  refuses a VCF of several samples, whose INFO fields bcftools computes
  over all of them; `--vcf-sample` only checks the one sample's name
  there, with a warning that it selected nothing.
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
  replicate. The run prints the matching `psmc_plot.pl` command, and its
  own command for replicates whose `--bootstrap-iters` differ from N_ITER,
  which cannot share a plot with the main fit.
- Bootstrap replicates whose fresh start, with θ0 from their own resampled
  data, has negative transition probabilities are found before the main
  fit and skipped, with a warning that gives the `--tr-ratio` or `--rho0`
  that fits them all; when every replicate is such, the run stops before
  the main fit. `summary.json` lists them in `skipped_replicates`, the
  output line and the report name them, and their numbers are missing
  from the replicate files. In the library,
  `bootstrap::invalid_replicate_starts` finds them and
  `BootstrapOptions::skip_invalid_starts` skips them; without it, such a
  replicate fails with EM's error.
- `--init-params FILE` starts from the parameters of an earlier run (C
  `-i`), and its replicates start there too, as `psmc -i -b`.
- `--tr-ratio`, `--gen-years`, `--qc-json`, `-q`/`--quiet`, `--version`
  and `--bin-callable auto|ignore|approximate` (see *Changed*).
- Output JSON metadata: `version`, `n_iter`, `final_loglike`, `loglike`
  (the log-likelihood after each iteration, from the start), `bin_size`,
  `gen_years`, `tr_ratio`, `smooth_lambda`, `input`, `qc` and
  `bin_callable` (the `--bin-callable` mode, the emission model fitted,
  how the counts were obtained and what they hold); with N_ITER
  0 the run evaluates the starting model for them and for `Last EM
  loglike`. Files without them (0.1) still load.
- HTML report: run summary (with the bin model and the callable bases of
  the called bins), QC findings and per-sequence table, bootstrap
  settings with a coverage caveat, and an offline notice when ECharts
  cannot load (`PSMC_ECHARTS_JS` embeds a local copy).
- `experiment/validation/bin_callable.py`, the simulation study of
  `--bin-callable` (see *Changed*), with its results in
  `experiment/validation/README.md`.
- Warnings for input options that do not apply to the input as read, for
  a psmcfa input that looks like a per-base consensus (a FASTQ quality
  block, or at least 1% of bins written as letters fq2psmcfa never
  writes), which is read one base per bin, for chromosomes that only some
  `--vcf-mask` files cover, and for bootstraps of fewer than 20 trunks; a
  single-trunk bootstrap is an error before fitting.
- Input from pipes and FIFOs (the file is read once; only `--depth auto`
  needs a regular file).
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
  coverage figures (by default with the CLI's start and smoothing;
  `--tr-ratio`, `--smooth-lambda`), and `examples/bench_estep.rs`.
- Tests against the original tools' outputs: `tests/c_equivalence.rs` (C
  psmc 0.6.5-r74), the `.psmc` output and `psmc_plot.pl`, psmcfa parsing,
  route A goldens from bcftools, `vcfutils.pl`, `fq2psmcfa` and
  `generate_multihetsep.py`, `splitfa` and `psmc_resamp`; and an
  end-to-end test of `scripts/bam_to_psmc.sh` (`tests/pipeline/`), which
  runs it on simulated reads and compares its psmcfa with `vcf2fq |
  fq2psmcfa` and with the simulation, and runs route B with the guide's
  commands.
- CI: formatting and clippy with warnings denied (pinned to Rust 1.99.0),
  tests on Linux and macOS, a Rust 1.88 build, the Python tests of the
  reproduction package, and a `pipeline` job that checks the shell
  scripts with shellcheck, runs the end-to-end test with Ubuntu's bcftools
  and `vcfutils.pl` and with `fq2psmcfa` built from lh3/psmc 0.6.5-r74, and
  runs the ignored tests that compare consensus and mask mode with those
  tools.
- Reproduction package: the simulation scripts and the supplementary-table
  runner deleted in 79ac364 are restored under `experiment/scripts/`, with
  new runners `run_main_text.py` and `run_real_data.py`;
  `experiment/requirements.txt`; `experiment/run.py --keep-going`; Python
  tests (`python3 -m unittest discover -s experiment/tests`).
- `CONTRIBUTING.md`, issue and pull-request templates, `CITATION.cff` and
  this changelog.

### Fixed

- The warning about fitted parameters on the M-step's bounds advised
  "wider bounds", which no option sets. It now says what each case means:
  λ at the upper bound (1e4) are intervals with almost no coalescences in
  the data, whose Ne is the bound rather than an estimate (as for the
  first λ of `4+25*2+4+6` on one of two simulated bottleneck genomes,
  where C psmc's unbounded search reaches 1.5e5); λ at the lower bound
  make every lineage reaching their intervals coalesce at once; θ, ρ or
  `t_max` at a bound point to the input or the starting values. A
  parameter counts as on its upper bound within 1e-6 of its range, where
  the M-step's logit coordinate saturates and a λ the data push there
  stops (9,999.99991 on that genome; the test within 1e-12 missed it),
  and on its lower bound within twice the M-step's clamp.
- Gzip input with several members (bgzip: `.vcf.gz` from bcftools, GATK or
  tabix, bgzipped psmcfa and BED masks) is read in full instead of only its
  first block, and a bgzip file that lacks BGZF's end-of-file block, as
  one cut short by an interrupted job does, is refused instead of being
  read as the whole input.
- psmcfa: header lines were read as bins, so names containing T, K or N
  (GenBank, RefSeq and UCSC accessions) added bins (HLpelCas1: 523 extra
  T bins), and every byte other than T, K and N was dropped, shifting later
  bins. Records and bytes are now read as C psmc reads them.
- Multihetsep: a site was heterozygous when any two haplotypes differed, so
  multi-sample files gave a K bin at almost every row; now only the
  selected pair counts, and an unknown allele (`?`, `N`) is missing instead
  of heterozygous. An out-of-range haplotype index panicked, a row
  widened by an indel silently shifted the pair, and the rows of a
  chromosome that comes back after another became a new sequence whose
  uncalled sites were counted from position 0; all are errors now.
- VCF mask mode classifies every base: 0.1 moved the missing sites of each
  interval between heterozygous SNPs together, which got 14.7% of bins
  wrong at 20% random missingness, counted missing genotypes as callable,
  aborted on two records at one position, dropped everything after the
  last heterozygous SNP and started a new sequence when a chromosome came
  back. FILTER was ignored and heterozygous indels counted as heterozygous
  SNPs; filtered records and missing genotypes are now missing at their
  position, heterozygous non-SNPs over their REF span.
- gVCFs (GATK, DeepVariant, `bcftools call -g`, Strelka2) are recognised by
  the reference and no-call blocks among their first records (ALT only
  `.`, `<*>`, `<NON_REF>` or `<X>`, with INFO/END) and read in mask mode,
  each block from POS to END: a no-call or filtered block is missing over
  its whole span, and positions that no record or block covers are missing
  whatever the masks say. 0.1 read a block as its first position, and the
  rest of it, like every position without a record, as called. A
  reference block still counts as called whatever its depth and quality,
  which the run now warns about: a gVCF needs a callable mask made from
  its blocks. In the library, `io::input::load` sets the new
  `VcfOptions::gvcf` for such files; `VcfSummary::uncovered_sites` counts
  the uncovered positions.
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
- EM could start from negative transition probabilities, which PSMC's
  model has when ρ0 per bin is too large for the time grid (at t_max 15
  and flat λ, above about 1/6, which θ0/4 passes at about 49%
  heterozygous bins). Its M-step objective was then NaN, EM never moved,
  and the run wrote a flat curve without a word, as C psmc does. The run
  now checks the start before the main fit and stops with the limit and
  the bin size, `--tr-ratio`, `--rho0` or saved rho that fixes it; QC
  reports such a start as `INVALID_START`. In the library, the new
  `PsmcModel::check_transitions` finds such a start and returns a
  `model::NegativeTransitions` error, with which `em_train` refuses it
  (reachable with `downcast_ref` through EM's and the bootstrap's
  context), and the M-step refuses a start where its objective is not
  finite.
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
  the downloaded assets, and the Windows `.sha256` line ended in CRLF,
  whose CR macOS's `shasum -c` reads as part of the file name. The release
  now checks every list against the assets before publishing.
- The Linux release binary is linked against glibc 2.17 (with
  cargo-zigbuild) and runs on CentOS and RHEL 7 and later. 0.1's was
  linked against the build machine's glibc 2.39: it did not start before
  glibc 2.34 (RHEL and Rocky 8, Ubuntu 20.04, Debian 11) and printed a
  `GLIBC_2.39` warning on every run before 2.39. The release checks the
  binary in CentOS 7, Rocky Linux 8 and Ubuntu 22.04 before publishing it.
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

- The E-step's forward and backward steps work in O(states) per bin instead
  of O(states²) for PSMC transition matrices of up to 256 states (their
  triangles are rank one; larger models use the dense O(states²) kernels);
  the expected transition counts are still accumulated in O(states²) per
  bin, by a blocked kernel with NEON code on aarch64, so the E-step as a
  whole still grows with the square of the number of states. The E-step
  keeps one forward vector every 128 bins instead of whole forward and
  backward matrices. On the human chromosome 1 in `experiment/real_data` (one
  sequence of 2.49 M bins, `4+25*2+4+6`, 20 iterations, 1 thread, Apple M1
  Pro) a fit takes 25 s and 21 MB instead of 139 s and 1.95 GB; on
  HLhydTec1 (34 sequences, 18.5 M bins, 5 iterations, 8 threads) 9.9 s and
  96 MB instead of 79 s and 6.9 GB.
- TMRCA decoding runs in parallel within and across sequences (30,000
  sequences of 50-300 bins at 8 threads: 10.8 s to 0.86 s) and formats
  numbers without the standard library's float formatter.
- psmcfa input is streamed instead of read whole, and the VCF reader is two
  to three times as fast.
- The M-step differentiates its objective in reverse mode (adjoints written
  out by hand) instead of with forward-mode dual numbers, which carried the
  derivatives with respect to every parameter, in a heap vector, through
  every operation: an evaluation takes O(states²) time and O(states) memory
  besides the expected counts, however many λ are free (about 27 µs
  instead of 1.0 ms for `4+25*2+4+6`). With one λ per interval, one EM
  iteration on the 180,000-bin test input with 401 intervals peaked at 235
  MB of resident memory instead of 1.30 GB (20 MB without the E-step's
  cache of forward rows) and took 13.8 s instead of 36.7 s; with 201
  intervals 9.0 MB and 0.71 s instead of 156 MB and 4.6 s (load averages
  3 to 4). The warning that such models' M-steps would need gigabytes,
  given before the input was read, is gone.
- With that gradient and a curvature-scaled L-BFGS, the M-step converges
  (see *Changed*) in a small fraction of the E-step's time: with
  `4+25*2+4+6` on one thread, 2.1 ms per EM iteration on a simulated
  500 Mb sequence (E-step 2.3 s, so 0.09%), where 100 forward-mode L-BFGS
  iterations took 123 ms and 1000 took 1.35 s, and 2.2 ms on a simulated
  4 × 25 Mb genome (E-step 0.46 s), where 100 iterations took 122 ms. 25 EM
  iterations on the 4 × 25 Mb genome take 11.9 s, against 14.7 s with 100
  forward-mode iterations and 161 s with 1000; on the 500 Mb sequence,
  where the E-step dominates, 58.6 s, against 61.5 s and 91.5 s. (Medians
  of three runs, interleaved, at load averages of 3.1 to 7.4.) The M-step's
  time grows with the free parameters times states²: with one λ per
  interval on the 180,000-bin test input it took 13 to 18 ms with 101
  intervals and 0.29 to 0.60 s with 301, against E-steps of 0.17 s and
  4.9 s.
- Callable masks are merged as they are read, so their memory follows the
  merged intervals instead of the rows (40 million adjacent per-site rows:
  646 MB to 3.9 MB).
- The E-step and the log-likelihood start their work items in order and
  merge each item's sums as soon as the items before it are merged,
  instead of holding every item's sums until all are done; results are
  unchanged, to the bit. A bootstrap of 8 replicates with 8 threads on a
  200 Mb genome in 500 sequences peaks at 81 MB, against 163 MB when every
  item's sums were held, and at 103 MB with the callable bases of each bin,
  whose sums take one row per count (307 MB when held).

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
