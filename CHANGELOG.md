# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

0.2.0 rebuilds the E-step, the input readers and the bootstrap, follows
C psmc in its defaults, and adds the `qc` and `convert` commands. **Results
are not comparable with 0.1 runs that relied on the defaults** (first part
of *Changed*). The manuscript's results are made again with 0.2 by the new
pipeline in `experiment/pipeline/`, which replaces 0.1's reproduction
package (*Removed*).

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
- Each M-step runs until it converges, as C psmc's does; `--mstep-iters`
  (1000, was 100) is only a cap. 0.1's 100 L-BFGS iterations stopped short
  of the maximum of Q (all 25 M-steps of its fit of E1's first bottleneck
  genome reached the cap) and left the curve 0.0007 to 0.0065 (RMSE of
  log10 Ne) from C psmc's on four of E1's simulated genomes. For 0.2's
  agreement with C psmc see `experiment/pipeline/README.md` (E1, E2).
- The M-step bounds each λ by 1e4 instead of 1e6 (`MStepConfig::lam_hi`):
  above 1e4, C's formulas give rounding noise in f64 for short early
  intervals (with λ up to 1e6, 36 of the 40 M-steps of a fit with one λ
  per interval of a simulated bottleneck genome end in a failed line
  search, against 1 with 1e4), and a λ of 1e4 already means no
  coalescences in the interval.
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
  called bin. With missing data scattered in runs shorter than a bin, full
  bins biased θ, N0, the times and Ne low by about 1.2 times the missing
  share of the called bins (E4 in `experiment/pipeline/README.md`). The
  E-step, the M-step, the starting θ0, the bootstrap and the TMRCA decoding
  all use the counts. Complete data, and psmcfa input (which records no
  counts), give to the bit the fit of `--bin-callable ignore`, which fits
  C psmc's model on any input. Multihetsep does not say where its uncalled
  sites are, so the counts of its bins are approximate; they made the fit
  worse in the validation, so `auto` fits multihetsep input as C psmc
  would fit its bins, and `--bin-callable approximate` uses them. See
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

- Settings and outputs are checked before the input is read. Usage errors
  exit with 2, and errors reading or writing files exit with 1.
  - `--t-max` must be in [1, 200] and `--n-steps` in 1 to 9999.
  - `--pattern` must parse and agree with `--n-steps`.
  - `--bootstrap-iters` needs `--bootstrap`.
  - Outputs must be writable, and distinct from each other and from the
    input. `OUTPUT` may not end in `.html` (0.1 let outputs overwrite each
    other, as the report did `OUTPUT fit.html`, and failed on a bad path
    only after the fit and the bootstrap). Only `--qc-json` and
    `qc --json` may be `-`.
- `-h` summarises each option and `--help` gives the details. `psmc-rs`,
  `convert` and `qc` alone print their help.
- Errors and warnings start with `error:` and `warning:`.
- A run prints how it read the input and then the QC report. For psmcfa
  that is C psmc's `n_seqs`, `sum_L` and `sum_n`; for VCF it is the mode,
  the filters and the bases lost.
- The output JSON is written after the main fit, before the bootstrap.
- For VCF and multihetsep input, the `.psmc` output has an
  `MM bin_model:...` line after `n_seqs`, which psmc_plot.pl skips.
- The `.psmc` output has no `QD` line: C psmc's EM Q before and after each
  M-step, less Q's largest value over all HMMs, needs every expected
  transition count, which the E-step no longer keeps (*Performance*).
  psmc_plot.pl does not read it.
- A bootstrap first removes the `replicate_NNN.*` and `summary.*` files in
  its directory, and warns when there were more replicates than it makes.
  A fit without `--bootstrap` warns when `OUTPUT.bootstrap` holds
  replicates.
- The HTML report labels the curve psmc-rs and scales it by the run's bin
  size and generation time.
- `PSMC_ALPHA_CACHE_MB` is one budget for the whole process, used only by
  models of more than 256 states. 0 disables the cache (in 0.1 it selected
  the default, 2048).
- `--threads` sets the thread pool of each run (the TUI kept the first
  run's).
- BCF input is refused with the `bcftools view -Oz` command that converts
  it, and a UTF-8 byte order mark is skipped.

Library API:

- EM lives in the new `em` module; `opt` re-exports `em_train`, `EmHistory`
  and `EmProgressEvent`. `EmHistory` gains `post_sigma`,
  `mstep_evaluations`, `mstep_iterations` and `mstep_stops`
  (`opt::MStepStop`), and `loglike[i].1` is the log-likelihood after
  iteration i (0.1: a copy of `.0`). `em_train` returns the model with its
  matrices recalculated.
- `MStepConfig`: `rel_tol` (1e-13, relative to the objective) replaces
  `tol_grad`, an absolute gradient norm that grew with the input. The
  defaults are now `max_iters` 1000 (`opt::DEFAULT_MSTEP_ITERS`; was 100),
  `lbfgs_m` 20 (7), `lambda` 0 (1e-3) and `lam_hi` 1e4 (1e6; see
  Defaults). The last `MStepProgress` event of an M-step has `total` equal
  to the iterations taken.
- `opt::m_step(model, stats, config, on_progress)` replaces `m_step_lbfgs`
  and `m_step_lbfgs_with_progress`: it starts from the model's parameters,
  searches within `config`'s bounds and returns an `MStepOutcome` (the
  parameters, the objective's evaluations, the iterations and why it
  stopped).
- The M-step's objective is infinite where the f64 HMM is invalid
  (negative or NaN probabilities, or transition rows that do not sum to 1
  within 1e-9), so the search no longer moves where λ underflows.
- `report::write_html_report(path, &ReportData)` replaces the positional
  form.
- `bootstrap::new_rng` returns `BootstrapRng` (ChaCha8) instead of
  `SmallRng`. `sample_sequences_block_bootstrap` (0.1's scheme) takes any
  `Rng`. `BootstrapSummary` gains `skipped_replicates`.
- `hmm::SufficientStats::xi`, the matrix of expected transition counts,
  becomes `transitions` (`hmm::TransitionSums`): their sums over the rows
  and the columns of each triangle of the transition matrix and over its
  diagonal, all that C psmc's Q needs of them (*Performance*).
- `Observations` and `hmm::SufficientStats` gain a `callable` field (the
  callable bases of each bin, `docs/design/bin-callable-model.md`), which
  breaks struct literals (add `callable: None`); `read_vcf` and `read_mhs`
  fill it for bins of at most 65,535 bases. Without counts, results are
  C psmc's to the bit.
- `PsmcModel::new` rejects `n_steps` 0, single-state and overflowing
  patterns, more than `model::MAX_STATES` (10,000) states, and `t_max`
  outside (0, 1000).
- `PsmcModel::compute_t` takes no argument: the grid's `alpha` is
  C psmc's 0.1, the one the model's HMM is computed with.
- `read_psmcfa` fails on input without a `>` record; `read_vcf` fails on
  unsorted positions, a chromosome that comes back, a record beyond its
  `##contig` length and a genotype allele that ALT lacks.
- `hmm::write_tmrca_posterior_tsv` and `hmm::e_step_streaming_with_progress`
  are gone: `hmm::write_tmrca_posterior_tsv_with_options` (`TmrcaOptions`)
  and `hmm::e_step_with_options` (`EStepOptions::on_progress`) take their
  arguments. The TMRCA output is gzip when the path ends in `.gz`.

### Added

- `psmc-rs qc INPUT`: a report of an input's suitability, without fitting:
  bin counts, θ0, heterozygosity, called length, N50, missing runs, bin
  saturation, read depth, the callable bases of the called bins (VCF and
  multihetsep), per-sequence statistics, and findings with stable
  codes (`INVALID_START`, `VARIANTS_ONLY_NO_MASK`, `PARTLY_CALLABLE_BINS`,
  `HIGH_MISSING`, `BIN_SATURATION`, `FEW_HETS`, `TOO_SHORT`,
  `SHORT_SEQUENCES`, `SEX_OR_ORGANELLE`, `HET_OUTLIER`,
  `APPROX_MISSING_PLACEMENT`, `DEPTH`, `DROPPED_SEQUENCES`) and
  recommendations. `PARTLY_CALLABLE_BINS` says how much of the called bins
  is callable and the bias of `--bin-callable ignore`. When the input
  records callable bases, heterozygosity is compared per callable base
  (table column `het/kb*`, JSON `qc.het_per_callable_kb` beside
  `qc.het_per_kb`), so that missing data does not make a sequence look
  less heterozygous. `APPROX_MISSING_PLACEMENT` gives the share of uncalled
  sites in multihetsep input and warns above 5%: their placement biases θ,
  the times and Ne low, whatever `--bin-callable` is set to
  (`convert --multihetsep` of a VCF with uncalled sites says so).
  `INVALID_START` takes θ0 from the callable bases of each bin when the
  fit does (VCF, `--bin-callable auto`). `--json`, `--strict` (exit 2 on a
  warning) and `--bootstrap-block-size` (trunk checks). Every fit and
  conversion prints the same report and `--qc-json` writes it.
- `psmc-rs convert INPUT`: psmcfa as fq2psmcfa lays it out, and multihetsep
  as `generate_multihetsep.py` writes it, in one file or one per sequence
  (`--psmcfa`, `--multihetsep`, `--multihetsep-per-chrom`), gzip by name;
  gzip outputs (also `--tmrca-out FILE.gz`) are the same bytes from run to
  run and with any `--threads`; compare those of two versions decompressed.
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
- VCF mask mode options `--vcf-keep-filtered` and `--vcf-no-contig-end`.
- `--mhs-pair I,J` selects the two haplotypes of a multihetsep file.
- `--psmc-out FILE`: C psmc's `.psmc` output (rounds 0 to N_ITER) for
  `psmc_plot.pl`; with `--bootstrap`, `replicate_NNN.psmc` for every
  replicate. The run prints the matching `psmc_plot.pl` command, after a
  bootstrap the commands that plot the replicates with the main fit (as C
  psmc's), and their own command for replicates whose `--bootstrap-iters`
  differ from N_ITER, which cannot share a plot with the main fit.
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
- `--tr-ratio`, `--qc-json`, `-q`/`--quiet`, `--version` and
  `--bin-callable auto|ignore|approximate` (see *Changed*).
- Output JSON metadata: `version`, `simd` (the HMM kernels that ran:
  `avx2+fma`, `portable` or `neon`), `n_iter`, `final_loglike`, `loglike`
  (the log-likelihood after each iteration, from the start), `bin_size`,
  `gen_years`, `tr_ratio`, `smooth_lambda`, `input`, `qc` and
  `bin_callable` (the `--bin-callable` mode, the emission model fitted,
  how the counts were obtained and what they hold); with N_ITER
  0 the run evaluates the starting model for them and for `Last EM
  loglike`. Files without them (0.1) still load.
- HTML report: run summary (with the bin model and the callable bases of
  the called bins), QC findings and per-sequence table, and bootstrap
  settings with a coverage caveat.
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
  a generation-time field, Ctrl+U clears a field, and a failed run goes
  back to the form.
- Library: `io::input::load` and `Dataset` (one loader for every format),
  `io::consensus`, `io::write` (psmcfa and multihetsep writers),
  `io::mhs::read_mhs_with`, `io::vcf::read_vcf_with`, the `qc` and
  `psmc_out` modules, `em::em_train_with_options` with `EmOptions`
  (cancellation, explicit thread pool), `em::posterior_sigma`,
  `hmm::{log_likelihood, e_step_with_options,
  EStepOptions, AlphaCacheBudget, Cancelled}`, `PsmcModel::{avg_t,
  n0_for_bin, ne_curve, state_tmrca_years, c_pi, c_sigma, n_recomb}`,
  `model::{n_steps_for_pattern, T_INF, MAX_STATES}`, `bootstrap::{
  run_bootstrap, run_bootstrap_with_options, BootstrapConfig,
  BootstrapOptions, split_for_bootstrap, resample_chunks,
  bootstrap_trunk_count, replicate_rng}`; for the callable bases of each
  bin, `io::callable` (`CallableBases`, `BinCounts`, `CallableSummary`),
  `io::Sequences`, the `callable` options of `EStepOptions`, `EmOptions`
  and `hmm::TmrcaOptions`, `initial_theta_with_callable`,
  `model::ThetaCounts` and `bootstrap::run_bootstrap_sequences`.
- `examples/bootstrap_calibration.rs`, a small simulation that compares
  bootstrap SDs with true sampling SDs (by default with the CLI's start
  and smoothing; `--tr-ratio`, `--smooth-lambda`), and
  `examples/bench_estep.rs`.
- Tests against the original tools' outputs: `tests/c_equivalence.rs` (C
  psmc 0.6.5-r74), the `.psmc` output and `psmc_plot.pl`, psmcfa parsing,
  route A goldens from bcftools, `vcfutils.pl`, `fq2psmcfa` and
  `generate_multihetsep.py`, `splitfa` and `psmc_resamp`; and an
  end-to-end test of `scripts/bam_to_psmc.sh` (`tests/pipeline/`), which
  runs it on simulated reads and compares its psmcfa with `vcf2fq |
  fq2psmcfa` and with the simulation, and runs route B with the guide's
  commands.
- CI: formatting and clippy with warnings denied (pinned to Rust 1.99.0),
  tests on Linux and macOS, a build with the minimum Rust version, 1.88
  (`rust-version` in `Cargo.toml`), the Python tests of the reproduction
  package, and a `pipeline` job that checks the shell scripts with
  shellcheck, runs the end-to-end test with Ubuntu's bcftools and
  `vcfutils.pl` and with `fq2psmcfa` built from lh3/psmc 0.6.5-r74, and
  runs the ignored tests that compare consensus and mask mode with those
  tools.
- Reproduction package: the experiment pipeline (`experiment/pipeline/`,
  `python -m experiment.pipeline`), which makes the manuscript's
  experiments, tables and figures with C psmc and psmc-rs 0.2, the scripts
  that ran it on a SLURM cluster (`experiment/slurm/`), and Python tests
  (`experiment/tests`).
- `CONTRIBUTING.md`, issue and pull-request templates, `CITATION.cff` and
  this changelog.

### Fixed

- The warning about fitted parameters on the M-step's bounds advised
  "wider bounds", which no option sets. It now says what each case means:
  λ at the upper bound (1e4) are intervals with almost no coalescences in
  the data, whose Ne is the bound rather than an estimate (as for the
  first λ of `4+25*2+4+6` on one of E1's 40 simulated genomes, an
  expansion replicate, where C psmc's unbounded search reaches 1.08e6); λ
  at the lower bound make every lineage reaching their intervals coalesce
  at once; θ, ρ or `t_max` at a bound point to the input or the starting
  values. A parameter counts as on its upper bound within 1e-6 of its
  range, where the M-step's logit coordinate saturates and a λ the data
  push there stops (9,999.99996 on E5's bottleneck replicate 40, 3.9e-9 of
  the range below it, which a test within 1e-12 misses), and on its lower
  bound within twice the M-step's clamp.
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
  aborted on two records at one position and started a new sequence when
  a chromosome came back. FILTER was ignored and heterozygous indels
  counted as heterozygous SNPs; filtered records and missing genotypes are
  now missing at their position, heterozygous non-SNPs over their REF
  span.
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
- The E-step had three code paths whose results depended on the thread
  count and the cache budget (up to 1% in the parameters after 20
  iterations, more than the difference from C psmc); results are now
  bit-identical for any thread count and budget.
- EM could start from negative transition probabilities, which PSMC's
  model has when ρ0 per bin is too large for the time grid (at t_max 15,
  flat λ and C's default pattern, above about 1/6, which θ0/4 passes at
  about 49% heterozygous bins). Its M-step objective was then NaN, EM
  never moved, and the run wrote a flat curve without a word, as C psmc
  does. The run now checks the start before the main fit and stops with
  the limit and the bin size, `--tr-ratio`, `--rho0` or saved rho that
  fixes it; QC reports such a start as `INVALID_START` (an input without a
  heterozygous bin is left to the loader and `FEW_HETS`). In the library,
  the new `PsmcModel::check_transitions` finds such a start and returns a
  `model::NegativeTransitions` error, with which `em_train` refuses it
  (reachable with `downcast_ref` through EM's and the bootstrap's
  context), and the M-step refuses a start where its objective is not
  finite.
- Saved parameters reload bit for bit (exact JSON float parsing).
- A closed stdout (`psmc-rs ... | head -1`) aborted the run before any file
  was written.
- The progress bars left a stale line on the terminal at each E- or M-step
  phase and at the start of reading (a bar drew itself before it joined
  the others), about a hundred lines for 25 iterations; they now leave
  none, fit the terminal's width and count sites with separators. Their
  elapsed times stood still between updates (for a whole iteration on the
  EM bar), the EM bar gave the iteration under way beside the count of
  those done, and the M-step's bar a share and ETA of its cap, which
  converged M-steps stay far below; the bars now tick on their own, the EM
  bar counts the iterations done, and an M-step counts its iterations.
- The TUI's file picker (Ctrl+O) failed with "failed to read directory"
  when the input field held a file name without a directory, and going up
  (`../`, Backspace) failed the same way from there, or for the output
  directory took the current one. The output-directory picker (Ctrl+D)
  refused to open while the input-format field was invalid, though it
  does not use it.
- The TUI form could not take q, x, r or a space.
- The TUI's log stopped short of its last lines when long lines wrapped,
  hiding the end of a run; PageUp jumped to the top of the log and PageDown
  scrolled towards it. Its key hints were cut off at 120 columns and named
  the Setup keys on every page, its Result page listed four of the files a
  run writes (now each result line, as the command line prints it), and
  its rates kept falling after a stage finished (the gauges now show the
  time taken and left, an M-step its iterations rather than a share and
  ETA of its cap, and the clock of a stage left unfinished stops with the
  run). The Command panel no longer adds `--no-progress`, so that a
  command copied from it draws progress bars.
- The HTML report was blank, without a word, when ECharts could not be
  fetched; it now says so, and `PSMC_ECHARTS_JS` embeds a local copy.
- The HTML report stacked the bootstrap band on years instead of Ne, so
  the band it drew had nothing to do with the replicates' quantiles.
- The bootstrap summary (`summary.tsv`, `summary.json`, the report's band)
  took each replicate's Ne at the main curve's first time point from its
  state 1 when the replicate's own state 1 began later; without a pattern
  (0.1's default), or with a first pattern group of one interval, states
  0 and 1 differ, and that first row could be wrong several-fold.
- The bootstrap's `summary.json` was written through a buffer that was
  never flushed, so an error on its last write (a full disk, a quota) left
  a truncated file and still exited 0; the error is now reported, naming
  the file.
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

### Performance

- The E-step's forward and backward steps work in O(states) per bin instead
  of O(states²) for PSMC transition matrices of up to 256 states (their
  triangles are rank one, which Harris et al. 2014 used to decode
  coalescent HMMs in linear time; larger models use the dense O(states²)
  kernels), and so do its expected transition counts, for models of any
  size: C psmc's Q needs them only summed over each triangle's rows and
  columns and over the diagonal (`hmm::TransitionSums`), which the E-step
  adds up from prefix and suffix sums at every bin, with NEON code on
  aarch64 and AVX2 code on x86-64, instead of the whole states × states
  matrix of counts (38% of an E-step's time with 64 states on an M1 Pro;
  their sums take 12%). The E-step keeps one forward vector every 128
  bins instead of whole forward and backward matrices. On the human
  chromosome 1 in `experiment/real_data` (one sequence of 2.49 M bins,
  `4+25*2+4+6`, 20 iterations, 1 thread, Apple M1 Pro) a fit takes 16 s
  and 19 MB instead of 139 s and 1.95 GB; on HLhydTec1 (34 sequences,
  18.5 M bins, 5 iterations, 8 threads) 7.2 s and 92 MB instead of 79 s
  and 6.9 GB.
- The E-step and the log-likelihood merge each work item's sums as soon as
  those before it are merged, instead of holding them all, with results
  unchanged to the bit: a bootstrap of 8 replicates on 8 threads of a
  200 Mb genome in 500 sequences peaks at 81 MB instead of 163 MB.
- TMRCA decoding runs in parallel within and across sequences (30,000
  sequences of 50-300 bins at 8 threads: 10.8 s to 0.86 s).
- Input reading is faster and leaner: psmcfa is streamed instead of read
  whole, gzip and bgzip input is decoded in a thread of its own beside the
  parsing, and callable masks are merged as they are read (40 million
  per-site rows: 646 MB to 3.9 MB).
- The M-step differentiates its objective in reverse mode instead of with
  forward-mode dual numbers, which carried the derivatives with respect to
  every parameter through every operation, and evaluates it from the sums
  of the expected transition counts: an evaluation takes O(states) time
  and memory, however many λ are free (about 7.5 µs instead of 1.0 ms for
  `4+25*2+4+6`). With one λ per interval, one EM iteration on the
  180,000-bin test input with 401 intervals peaked at 417 MB of resident
  memory instead of 1.30 GB and took 11.0 s instead of 36.7 s. With that
  gradient and a curvature-scaled L-BFGS, the converged M-step (see
  *Changed*) takes a small fraction of the E-step's time: with
  `4+25*2+4+6` on one thread, 0.55 to 0.87 ms per EM iteration on a
  simulated 100 Mb chromosome (E-step 0.32 s, on an Apple M1 Pro), where
  100 forward-mode L-BFGS iterations took 123 ms and 1000 took 1.35 s.
- On x86-64 the E-step, log-likelihood and TMRCA decoding use AVX2 and FMA
  when the CPU has both, chosen at run time: a 1.9 Gb genome (5 EM
  iterations, 1 thread) took 71.7 s instead of 161 s on an AMD EPYC 7702
  (both timed before the expected transitions were summed in O(states)).
  Results are bit-identical on any CPU with AVX2 and FMA that gets the
  same libm `log` and `exp`. The portable code (CPUs without AVX2 or FMA,
  or `PSMC_RS_SIMD=portable`) does not fuse multiply-adds, so its fitted
  parameters differ by a relative 1e-7 to 1e-6 after 5 iterations. The
  JSON's `simd` and the HTML report record which code ran.

### Deprecated

- `PsmcModel::n0`, which leaves out the bin size (θ is per bin); use
  `PsmcModel::n0_for_bin`.

### Removed

- The public `progress` and `utils` modules, which only the library itself
  uses (0.1's `progress::spinner` and `utils::logsumexp` are gone);
  `opt::em_train_with_progress` (pass the callback as
  `EmOptions::on_progress` to `em::em_train_with_options`);
  `opt::m_step_lbfgs` and `opt::m_step_lbfgs_with_progress` (use
  `opt::m_step`).
- The CLI's within-sequence bootstrap (`sample_sequences_block_bootstrap`
  remains).
- 0.1's reproduction package (`experiment/run.py`, its notebook,
  `experiment/runs/main_text`, `real_data` and `supplementary`,
  `experiment/assets/ui/`), replaced by the pipeline; it stays in the
  v0.1.x tags.

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
