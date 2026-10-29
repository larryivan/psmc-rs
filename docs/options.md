# Options

Every option of psmc-rs 0.2.0, in brief. `psmc-rs --help`, `psmc-rs convert
--help` and `psmc-rs qc --help` give the full text, and `-h` a
one-sentence summary of each.

## Input

For a fit, `convert` and `qc` ([inputs.md](inputs.md)).

| option | default | meaning |
| --- | --- | --- |
| `--input-format auto\|psmcfa\|mhs\|vcf` | `auto` | the input format; `auto` recognises it from the content |
| `--bin-size BP` | 100 | bases per bin (`fq2psmcfa -s`): bins VCF and multihetsep input; for psmcfa, give the `-s` it was made with |
| `--batch-size BINS` | none | split sequences into HMM rows of at most BINS bins; the results do not change |
| `--mhs-pair I,J` | `0,1` | multihetsep: the two haplotypes, 0-based positions in the alleles column |
| `--vcf-sample NAME` | | VCF mask mode: the sample to read (required when there are several); consensus mode needs a one-sample VCF |
| `--vcf-mode auto\|consensus\|mask` | `auto` | VCF: consensus (`vcf2fq \| fq2psmcfa`) or mask mode |
| `--vcf-mask BED` | | VCF: callable sites (repeatable; a site must be in every one) |
| `--vcf-negative-mask BED` | | VCF: sites to exclude (repeatable) |
| `--vcf-keep-filtered` | off | VCF mask mode: use records whose FILTER is not PASS or `.` |
| `--vcf-no-contig-end` | off | VCF mask mode: end sequences at their last record, not the `##contig` length |
| `--depth auto\|MIN:MAX` | `auto` | VCF consensus mode: INFO/DP bounds (vcf2fq `-d`, `-D`); `auto` = a third and twice the mean |
| `--min-mapq MQ` | 10 | VCF consensus mode: minimum INFO/MQ (vcf2fq `-Q`) |
| `--indel-window BP` | 5 | VCF consensus mode: bases masked on each side of an indel (vcf2fq `-l`) |
| `--min-qual Q` | 10 | VCF consensus mode: minimum base quality (fq2psmcfa `-q`) |
| `--min-good-bases BASES` | 10000 | VCF and multihetsep: drop sequences with fewer called bases (fq2psmcfa `-g`; 0 disables) |
| `--min-good-fraction SHARE` | 0.2 | VCF and multihetsep: drop sequences whose called share is lower (0 disables) |

## Fitting

`psmc-rs [OPTIONS] INPUT OUTPUT N_ITER` ([fitting.md](fitting.md)).

| option | default | meaning |
| --- | --- | --- |
| `--tui` | | open the terminal interface; the other arguments fill its form ([tui.md](tui.md)) |
| `--threads N` | one per core | worker threads (E-step, bootstrap, TMRCA); the results do not depend on it |
| `--no-progress` | | no progress bars |
| `-q`, `--quiet` | | only results and warnings |
| `--pattern P` | `4+5*3+4` | free-parameter pattern (C `-p`) |
| `--n-steps N` | from the pattern | intervals minus one (C's `n`, 1 to 9999); alone, one λ per interval |
| `--t-max T` | 15 | starting t_max in units of 2N0 generations (C `-t`; 1 to 200) |
| `--tr-ratio RATIO` | 4 | starting θ0/ρ0 (C `-r`) |
| `--theta0 X` | −ln(1 − K/L); with callable counts, its callable-length analogue | starting θ0 per bin |
| `--rho0 X` | θ0 / `--tr-ratio` | starting ρ0 per bin |
| `--init-params FILE` | | start from a saved fit (C `-i`) |
| `--mu MU` | 2.5e-8 | mutation rate per bp per generation, for scaling |
| `--gen-years YEARS` | 25 | years per generation, for the time axes |
| `--bin-callable auto\|ignore\|approximate` | `auto` | VCF: fit each called bin by its callable bases (`auto`), or treat every called bin as BIN_SIZE bases (`ignore`, C psmc); `approximate` also fits multihetsep's approximate counts |
| `--mstep-iters N` | 1000 | cap on the L-BFGS iterations of an M-step, which runs to convergence |
| `--smooth-lambda W` | 0 | penalty on log-λ differences of neighbouring intervals |
| `--bootstrap N` | 0 | bootstrap replicates |
| `--bootstrap-block-size BINS` | 500000 | trunk size (splitfa); alias `--bootstrap-trunk-size` |
| `--bootstrap-seed SEED` | 42 | seed of the resampling |
| `--bootstrap-iters N` | N_ITER | EM iterations per replicate |
| `--bootstrap-dir DIR` | `OUTPUT` with `.bootstrap` in place of `.json` | replicate and summary files; an earlier bootstrap's `replicate_NNN.*` and `summary.*` there are removed first |
| `--qc-json FILE` | | the input QC report as JSON (`-` for stdout) |
| `--psmc-out FILE` | | C psmc's `.psmc` output |
| `--tmrca-out FILE` | | posterior TMRCA per bin (TSV; gzip for `.gz`) |

## convert

`psmc-rs convert [OPTIONS] INPUT`, with at least one output
([inputs.md](inputs.md#writing-inputs-psmc-rs-convert)).

| option | meaning |
| --- | --- |
| `--psmcfa FILE` | psmcfa as fq2psmcfa lays it out (gzip for `.gz`) |
| `--multihetsep FILE` | all sequences as multihetsep (VCF and multihetsep input) |
| `--multihetsep-per-chrom PREFIX` | one file per sequence, `PREFIX.<name>.multihetsep` |
| `--qc-json FILE` | the input QC report as JSON (`-` for stdout) |
| `-q`, `--quiet` | only results and warnings |

## qc

`psmc-rs qc [OPTIONS] INPUT` ([qc.md](qc.md)).

| option | meaning |
| --- | --- |
| `--json FILE` | also write the report as JSON (`-` for stdout; the text then goes to stderr) |
| `--bootstrap-block-size BINS` | check the input for a bootstrap with trunks of BINS bins |
| `--strict` | exit with status 2 when the report has a warning |

## Environment

| variable | meaning |
| --- | --- |
| `PSMC_ECHARTS_JS` | a local `echarts.min.js` to embed in the HTML report, for offline use |
| `PSMC_ALPHA_CACHE_MB` | the memory budget of the dense kernels' forward-row cache, for models of more than 256 states (2048; 0 disables it) |
| `PSMC_RS_SIMD=portable` | run the portable kernels instead of AVX2+FMA or NEON |
| `PSMC_DEBUG_MSTEP=1` | print diagnostics of each M-step on stderr (for debugging; not a stable interface) |
