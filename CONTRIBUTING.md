# Contributing to psmc-rs

Thank you for your interest in psmc-rs. Bug reports, questions, documentation
fixes and code contributions are all welcome.

## Getting help and reporting problems

- **Questions** about usage or results: open an issue on
  <https://github.com/larryivan/psmc-rs/issues> (a blank issue is fine) and
  include the command you ran and what you expected.
- **Bugs**: open an issue with the *Bug report* template. Please include the
  psmc-rs version (`psmc-rs --version`, plus `git rev-parse --short HEAD`
  for a source build), your OS and CPU, the exact command line, the full
  error output and, if possible, a small input that reproduces the problem
  (for example a few thousand lines of the psmcfa file). The output of
  `psmc-rs qc INPUT --json -` describes an input without sharing it. If the
  result differs from the original C psmc, also give the `psmc` command and
  its version (printed by `psmc` without arguments).
- **Feature requests**: open an issue with the *Feature request* template and
  describe the use case first, then the proposed interface.
- **Security issues**: please do not open a public issue; report them
  privately with GitHub's *Report a vulnerability* button on the repository's
  *Security* tab.

## Development setup

You need a stable Rust toolchain, 1.88 or later. The crate uses edition 2024
(Rust 1.85), and the locked dependencies of `ratatui` 0.30 (`instability`,
`darling`, `time`) need 1.88.

```bash
git clone https://github.com/larryivan/psmc-rs.git
cd psmc-rs
cargo build                  # debug build
cargo build --release        # optimised binary: target/release/psmc-rs
```

## Before you open a pull request

Run the same checks as CI (`.github/workflows/ci.yml`):

```bash
rustup toolchain install 1.99.0 --profile minimal --component rustfmt,clippy   # once
cargo +1.99.0 fmt --all --check
cargo +1.99.0 clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
python3 -m unittest discover -s experiment/tests
```

CI formats and lints with Rust 1.99.0, the release pinned in its `lint`
job, because every Rust release adds clippy lints: clippy of another
release can pass code that CI rejects, or the other way round. The pin
moves on purpose, together with the fixes its new lints ask for. Tests run
on the current stable release, and CI also builds every target with Rust
1.88 (`cargo +1.88 build --workspace --all-targets --locked`).

- The whole test suite must pass, and clippy 1.99.0 must report no warning.
- Every behavioural fix needs a test that fails without the fix and passes
  with it. Please check this (for example by temporarily reverting the source
  change) and say so in the pull request.
- New public items need rustdoc comments. Comments should explain
  non-obvious reasoning, not restate the code.
- Keep existing public functions compiling with their current signatures
  where possible; add new functions or option structs instead of changing
  signatures, and record user-visible changes in the top section of
  `CHANGELOG.md`, the one not released yet (`## [X.Y.Z] - Unreleased`, or
  `## [Unreleased]` before a version number is chosen).

### Equivalence with the original C psmc

`tests/c_equivalence.rs` compares psmc-rs with outputs of Heng Li's C
implementation (lh3/psmc 0.6.5-r74) stored in `tests/data/c_reference/`.
With C's initialisation the first E-step sees identical parameters, so its
log-likelihood must agree to C's printed precision (1e-5); later rounds depend
on the M-step optimiser and must stay within a relative 1e-5. This test
protects the numerical core, so:

- A change must not loosen these tolerances or edit the reference outputs to
  make the test pass.
- If a change intentionally alters results relative to C, explain why in the
  pull request, quantify the difference (log-likelihoods, parameters) on the
  reference data and on a realistic input, and discuss it in an issue first.
- Reference outputs are regenerated only with the C build described in
  `tests/data/c_reference/README.md` (`cc -O1 cli.c core.c em.c khmm.c kmin.c
  main.c aux.c -lm -lz`; an `-O2` build segfaults on macOS arm64), in a
  separate commit that states the exact `psmc` command.

The other comparisons with the original tools work the same way: their
outputs live under `tests/data/` next to the script that made them
(`psmc_out/generate.sh` and `qc_c_reference/generate.sh` for C psmc, and
`route_a/make_fixtures.sh` and `route_a_keep_alts/make_fixtures.sh` for
bcftools, `vcfutils.pl`, `fq2psmcfa` and `generate_multihetsep.py`), and
each directory's README names the tool versions. Two ignored tests run the
tools directly:

```bash
VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa \
  cargo test --release --test io_consensus_tools -- --ignored
FQ2PSMCFA=/path/to/fq2psmcfa cargo test --release --test io_vcf_exact -- --ignored
```

### Performance changes

Measure old and new builds back to back on the same idle machine, interleaved,
and report the median of at least three runs together with the input size,
thread count and hardware (`cargo build --release`; see `.cargo/config.toml`
for `target-cpu=native` benchmarks).

### Experiment and reproduction package

The scripts and notebook under `experiment/` regenerate the manuscript's
figures and tables; see `experiment/README.md`. If you change them:

```bash
python3 -m unittest discover -s experiment/tests -v
```

Scripts must keep working from any working directory and must not contain
machine-specific paths; paths written into result tables are
repository-relative.

## Commit messages

One logical change per commit, with a message of the form

```text
type(scope): short summary

What was wrong or missing, why this change is right, and numbers where you
measured something.
```

where `type` is one of `fix`, `feat`, `perf`, `refactor`, `test`, `docs`,
`style` or `ci`.

## Releasing (maintainers)

1. Set the version in `Cargo.toml` and run `cargo build` to update
   `Cargo.lock`.
2. In `CHANGELOG.md`, replace `Unreleased` in the version's heading with the
   date (`## [X.Y.Z] - YYYY-MM-DD`) and update the comparison links at the
   bottom.
3. Set `version` and `date-released` in `CITATION.cff`, and give every
   author `given-names` and `family-names` (and `orcid` where there is
   one): the README asks users to cite from this file, and GitHub's *Cite
   this repository* and reference managers otherwise credit only the
   GitHub handle.
4. Commit, tag and push the default branch with the tag:

   ```bash
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z
   git push origin master --tags
   ```

The tag starts `.github/workflows/release.yml`, which builds `psmc-rs` for
Linux (x86-64), macOS (x86-64 and Apple silicon) and Windows (x86-64),
packages each binary with `README.md` and `LICENSE`, and publishes the
archives, their `.sha256` files and `SHA256SUMS.txt` as a GitHub release.
If a run fails, re-run *Release* from the Actions tab (`workflow_dispatch`)
with the same tag instead of moving it. The Linux binary is linked with
`cargo zigbuild` for the glibc in the job's `glibc` matrix entry (2.17),
and `.github/scripts/check-linux-binary.sh` stops the release when it needs
a newer glibc symbol, or when `--version` and a short fit do not run
cleanly in CentOS 7, Rocky Linux 8 and Ubuntu 22.04 containers. Moving
that floor means changing *Install* in `README.md` too.

## License

By contributing you agree that your contributions are licensed under the MIT
license of this repository (see `LICENSE`).
