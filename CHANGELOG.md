# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-29

The release behind the manuscript. psmc-rs fits C psmc's model with C
psmc's defaults and reproduces its curves: on the 95 genomes both programs
fitted in the manuscript, the population sizes differ by at most 0.15% in
any time interval of the informative window. It replaces 0.1.x, whose
defaults and results differ.

### Fitting

- C psmc's model, time grid (`PSMC_T_INF` = 1000), starting point (θ0 =
  −ln(1 − K/L), ρ0 = θ0 / `--tr-ratio`, flat λ, `--t-max` 15) and
  defaults (`--pattern 4+5*3+4`, `--tr-ratio 4`, `--smooth-lambda 0`);
  `--init-params` (C psmc's `-i`), `--theta0` and `--rho0`.
- Each M-step maximises C psmc's objective with L-BFGS until it converges
  (`--mstep-iters`, 1000, is only a cap), with each λ between 1e-6 and 1e4.
- With VCF input, each called bin is fitted by its callable bases
  (`--bin-callable auto`), which removes most of the bias of missing data
  scattered in runs shorter than a bin; `--bin-callable ignore` gives C
  psmc's model.
- The start is checked before the fit: a ρ0 too large for the time grid,
  from which EM cannot move, stops the run with what to change.
- `--bootstrap N`: C psmc's `splitfa` and `psmc -b` in one command, the
  replicates in parallel (trunks of `--bootstrap-block-size` bins, 500,000
  by default, drawn across the input; fresh starts; per-replicate ChaCha8
  streams, so a seed gives the same replicates on any platform and with any
  `--threads`).
- `--tmrca-out`: the posterior TMRCA of every bin.

### Inputs

- psmcfa, parsed as C psmc parses it; VCF, in consensus mode (the
  all-sites VCF of `bcftools call -c`, binned as `vcfutils.pl vcf2fq |
  fq2psmcfa` would bin it, byte for byte) or in mask mode (any VCF or gVCF,
  with callable masks: `--vcf-mask`, `--vcf-negative-mask`); MSMC's
  multihetsep (`--mhs-pair`). Plain, gzip or bgzip, from files or pipes;
  the format and the VCF mode are recognised from the content.
- fq2psmcfa's sequence filter for VCF and multihetsep input
  (`--min-good-bases`, `--min-good-fraction`).
- `psmc-rs qc`: a report on an input's suitability, whose findings name a
  code and what to do; every fit prints it.
- `psmc-rs convert`: psmcfa as fq2psmcfa writes it, multihetsep as
  `generate_multihetsep.py` writes it.
- `scripts/bam_to_psmc.sh` and `docs/preprocessing.md`: from the aligned
  reads of one sample to an input.

### Outputs

- The parameters as JSON, with the run's metadata; an interactive HTML
  report; C psmc's `.psmc` output (`--psmc-out`) for `psmc_plot.pl`, with
  the plotting commands printed; the bootstrap's replicates and summary.
- The same bytes with any `--threads`.
- `psmc-rs --tui`: a terminal interface for the fit.

### Performance

- O(states) forward and backward steps for models of up to 256 states
  (Harris et al. 2014), the sums of the expected transition counts instead
  of their matrix, and one forward vector kept every 128 bins (a fit of a
  human chromosome 1 peaks at 19 MB); AVX2 and FMA, or NEON, kernels picked
  at run time; the sequences, the replicates and the TMRCA decoding in
  parallel. On a simulated 3 Gb genome, 14 times as fast as C psmc on one
  thread and 386 times on 30.

### Release and reproducibility

- Binaries for Linux (x86-64, glibc 2.17 or later) and macOS (x86-64 and
  Apple silicon), with SHA-256 checksums; none for Windows (use the Linux
  binary under WSL). The release checks that its Linux binary reproduces
  the production binary's fit in Rocky Linux 9.
- Tests against the outputs of C psmc, `vcfutils.pl`, `fq2psmcfa`,
  `splitfa`, `psmc_resamp` and `generate_multihetsep.py`; CI on Linux and
  macOS, with the minimum Rust version (1.88).
- `experiment/pipeline/`: the manuscript's experiments, tables and figures,
  with a provenance record for every output; `experiment/runs/`: their
  results.
- `CITATION.cff` and `CONTRIBUTING.md`.

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

[0.2.0]: https://github.com/larryivan/psmc-rs/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/larryivan/psmc-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/larryivan/psmc-rs/releases/tag/v0.1.1
