# psmc-rs

psmc-rs is a Rust implementation of PSMC, the Pairwise Sequentially Markovian
Coalescent of Li and Durbin (2011), which infers the history of effective
population size from the two genome copies of one diploid individual.

It fits the same model as Heng Li's C `psmc`, with the same time grid,
starting point (θ0 = −ln(1 − K/L), ρ0 = θ0/4, flat λ, t_max = 15) and `-p`
patterns. With C's initialisation the first round's log-likelihood equals
C psmc's to its printed precision, and later rounds,
which depend on the M-step optimiser (L-BFGS here, Hooke-Jeeves in C), stay
within a relative 1e-5 on the reference data; `tests/c_equivalence.rs`
enforces both. On top of the model, psmc-rs

- reads psmcfa, VCF (including the classic `vcfutils.pl vcf2fq | fq2psmcfa`
  route in one step) and MSMC's multihetsep, plain or gzip/bgzip;
- checks an input before fitting (`psmc-rs qc`) and converts between formats
  (`psmc-rs convert`);
- runs C psmc's bootstrap in one command, in parallel;
- writes JSON parameters, an interactive HTML report, C psmc's `.psmc` output
  for `psmc_plot.pl`, and per-bin TMRCA posteriors;
- is deterministic and multithreaded, and its E-step keeps 1/128 of the
  forward-backward matrices (a fit of human chromosome 1 peaks at 21 MB).

Version 0.2.0 changed several defaults to C psmc's, so its results are not
comparable with 0.1 runs made with default settings; see
[Changes from 0.1](#changes-from-01) and [CHANGELOG.md](CHANGELOG.md).

## Install

Release binaries for Linux (x86-64), macOS (x86-64 and Apple silicon) and
Windows (x86-64) are on the [releases page](https://github.com/larryivan/psmc-rs/releases),
each archive with a `.sha256` checksum (`SHA256SUMS.txt` lists them all).
The Linux binary is linked against glibc 2.17 and runs with glibc 2.17 or
later (CentOS and RHEL 7 and newer distributions).

From source, with Rust 1.88 or later:

```bash
git clone https://github.com/larryivan/psmc-rs.git
cd psmc-rs
cargo install --path . --locked   # or: cargo build --release (target/release/psmc-rs)
psmc-rs --version
```

## Quick start

```bash
# psmcfa from fq2psmcfa, with C psmc's defaults (-t15 -r4 -p "4+5*3+4"),
# also writing C psmc's output for psmc_plot.pl
psmc-rs sample.psmcfa fit.json 25 --psmc-out fit.psmc

# The classic route in one step: an all-sites VCF from bcftools call -c is
# read exactly as vcfutils.pl vcf2fq | fq2psmcfa would read it
bcftools mpileup -C50 -q20 -Q20 -f ref.fa sample.bam | bcftools call -c -Oz -o calls.vcf.gz
psmc-rs qc calls.vcf.gz                             # is this input suitable?
psmc-rs calls.vcf.gz fit.json 25 --bootstrap 100    # fit with 100 bootstrap replicates

# Keep the binned input for other tools
psmc-rs convert calls.vcf.gz --psmcfa sample.psmcfa.gz
```

`fit.json` holds the parameters, `fit.html` the report. The run prints the
input summary, a QC report and the model on stderr, and its results on
stdout:

```text
Last EM loglike: -402414.07486276847
Parameters: fit.json
PSMC output: fit.psmc
HTML report: fit.html
```

The first command corresponds to `psmc -N25 -t15 -r4 -p "4+5*3+4" -o fit.psmc
sample.psmcfa`, C psmc's defaults. For modern humans, psmc's README makes
the input with `fq2psmcfa -q20` and runs `psmc -N25 -t15 -r5 -p
"4+25*2+4+6"`; in psmc-rs that is `--pattern '4+25*2+4+6' --tr-ratio 5`,
plus `--min-qual 20` when psmc-rs bins the VCF itself (consensus mode). How
to get from reads to an input file (calling, masks, multihetsep for MSMC)
is described in [docs/preprocessing.md](docs/preprocessing.md).

`psmc-rs --tui` opens a terminal interface with the same pipeline (see
[Terminal interface](#terminal-interface)).

## Input

psmc-rs reads three formats, each plain, gzip or bgzip (gzip is recognised
from its magic bytes, and every member of a bgzip file is read). A bgzip
file must end with BGZF's end-of-file block, as htslib checks: without it,
the file was probably cut short by an interrupted job, and it is refused.
BCF is refused too (`bcftools view -Oz` converts it), and a UTF-8 byte
order mark is skipped. `--input-format auto` (the default) recognises the
format from the content: a `##` or `#CHROM` line makes it VCF; other `#`
lines and blank lines are skipped; then a line starting with `>` or `@`
means psmcfa, a line of at least 8 tab-separated fields a VCF without
header, and anything else multihetsep (if such a file fails to read as
multihetsep from the start, the error says how its format was recognised).
For inference (not `convert` or `qc`), INPUT may also be a directory: it
stands for its first file, by name, ending in `.psmcfa`, `.vcf`, `.mhs` or
`.multihetsep` (each optionally `.gz`), or only in the extensions of
`--input-format` when that is set.

`--bin-size` (default 100, `fq2psmcfa -s`) is the number of bases per bin. It
bins VCF and multihetsep input, and scales time and Ne for every format, so
give the `-s` a psmcfa file was made with.

Inference, `convert` and `qc` share the input options and read a file
identically. Options that do not apply to the input as read (for example
`--vcf-mask` with psmcfa input) are named in a warning and ignored.

### psmcfa

Parsed as C psmc parses it. A record starts at `>` (or `@`) and its header
line is never sequence data; the name runs to the first whitespace, and text
before the first header is ignored. In sequence lines `T`, `A`, `C`, `G`,
`0` and their lowercase forms are called bins, `K`, `M`, `R`, `S`, `W`, `Y`,
`1` and their lowercase forms are heterozygous bins, any other printable
byte (such as `N`) is missing, and whitespace, CR, control and non-ASCII
bytes are skipped. Differences from C psmc: a file without any `>` record is
an error, a truncated FASTQ quality block is an error (C psmc silently stops
reading), and empty records are dropped from the fit (they still count in the
`.psmc` header's `n_seqs`, as in C). psmcfa input is not filtered further.
When a record has a FASTQ quality block, or at least 1% of the bins are
letters that fq2psmcfa never writes (`A`, `C`, `G`, other IUPAC codes,
lowercase), the run warns that the input looks like a per-base consensus,
such as `vcfutils.pl vcf2fq` output, which is then read one base per bin.

### VCF

One diploid sample is read, in one of two modes. Mask mode reads the
genotypes of one sample column, which `--vcf-sample NAME` picks when the
VCF has several. Consensus mode, like `vcfutils.pl vcf2fq`, reads INFO
fields that bcftools computes over all samples together, so it refuses a
VCF of several samples, and `--vcf-sample` only checks the name of the one
sample there (the run warns that it selected nothing).

**Consensus mode** reproduces the classic PSMC input pipeline,
`vcfutils.pl vcf2fq | fq2psmcfa`, in one streaming pass over the all-sites
VCF of `bcftools call -c`, with bins identical to fq2psmcfa's (checked byte
for byte against bcftools 1.24, vcfutils.pl and fq2psmcfa; see
`tests/data/route_a/`). The tools' options map as follows:

| vcf2fq / fq2psmcfa | psmc-rs | default |
| --- | --- | --- |
| `vcf2fq -d MIN -D MAX` | `--depth MIN:MAX` | `auto`: a third and twice the mean INFO/DP, rounded |
| `vcf2fq -Q` | `--min-mapq` | 10 |
| `vcf2fq -l` | `--indel-window` | 5 |
| `fq2psmcfa -q` | `--min-qual` | 10 |
| `fq2psmcfa -s` | `--bin-size` | 100 |
| `fq2psmcfa -g` | `--min-good-bases` | 10000 |
| fq2psmcfa's fixed 20% of a sequence | `--min-good-fraction` | 0.2 |

`--depth auto` follows the psmc README's advice and differs from vcf2fq's own
default, `3:100000`. It needs a first pass over the file to find the mean, so
it cannot read a pipe; give `--depth MIN:MAX` there. Callable masks
(`--vcf-mask`) can be added in this mode too. Where the Perl and C tools
would misread malformed input, psmc-rs reports an error instead; the module
documentation of `psmc_rs::io::consensus` lists these cases, none of which
arise from `bcftools call -c` output.

**Mask mode** reads the sample's genotypes, for any other VCF (GATK,
`bcftools call -m`, gVCFs, simulations). Each position is a called
homozygous site, a heterozygous SNP or missing, and the positions are binned
with fq2psmcfa's rule (a bin is missing when more than 90% of its sites
are, else heterozygous when one of its sites is). Per record:

- FILTER other than `PASS` or `.`: the position is missing (kept with
  `--vcf-keep-filtered`);
- a missing or partial genotype (`./.`, `./1`): the position is missing;
- heterozygous with every allele a single base: a heterozygous SNP;
- heterozygous with another allele (indel, `*`, symbolic): its whole REF span
  is missing;
- homozygous (including hom-alt and homozygous indels): called.

A gVCF block, a record whose ALT alleles are only `.`, `<*>`, `<NON_REF>`
or `<X>` and that has an INFO/END, stands for every position from POS to
END: a block that these rules make missing (a no-call or filtered block) is
missing over its whole span, and a reference block's positions are called
homozygous unless a mask says otherwise. Several records at one position are
combined. A position without any record is called homozygous unless a mask
says otherwise, except in a gVCF, where a position that no record or block
covers is missing (see *Masks*), and each sequence runs to its `##contig`
length when the header gives one (`--vcf-no-contig-end` ends it at the last
record, or at the END of a last block). A chromosome without records gives
no sequence. The VCF must be sorted (`bcftools sort`): a position that goes
backwards, or a chromosome that comes back after another, is an error, as
is a record beyond its `##contig` length.

**Masks.** `--vcf-mask BED` gives callable sites and `--vcf-negative-mask
BED` sites to exclude; both repeat, and a site is callable when every
`--vcf-mask` contains it and no negative mask does, so masks that come one
per chromosome must be joined into one file first (the run warns about a
chromosome that only some masks cover). Rows of three or more columns are
BED (`chrom start end`, 0-based, end-exclusive); rows of exactly two
columns are `start end`, 1-based inclusive, and apply to every chromosome.
Rows are merged as they are read, so memory follows the merged intervals,
not the rows. **A variants-only VCF needs a mask**: without one, every
position the caller could not call counts as homozygous, which hides the
missing data and pulls θ and Ne down, while false heterozygous calls in
repeats that a mask would remove push them up. QC then reports
`VARIANTS_ONLY_NO_MASK`. **A gVCF needs a mask too.** In a gVCF, positions
that no record or block covers are missing whatever the masks say (bcftools
writes nothing where there are no reads), but a reference block counts as
called over its span whatever its depth and genotype quality (GATK writes
stretches without reads as `0/0` blocks), and the run warns about it. Make
the callable mask from the gVCF's blocks and give it with the gVCF or with
its variant records, as
[docs/preprocessing.md](docs/preprocessing.md#the-callable-mask) shows.

**Mode detection** (`--vcf-mode auto`, the default) looks at the first
10,000 records (at most 64 MB of them). If any of them is a gVCF block, the
file is a gVCF and is read in mask mode. Otherwise, if at least half have
ALT `.`, the file is an all-sites VCF: it is read in consensus mode when at
least 90% of those carry INFO/FQ (as `bcftools call -c` writes), and is an
error otherwise. A file whose records cover at least half of the positions
they span and at least 90% of which carry FQ (`bcftools call -c -A` at high
depth) is also read in consensus mode. Any other VCF is read in mask mode.
`--vcf-mode consensus|mask` forces a mode (forced mask mode still reads a
gVCF as one); the run says which mode it used and why.

### multihetsep

MSMC's format, as `generate_multihetsep.py` writes it: one row per
segregating site, `chrom pos nr_called alleles`, where `nr_called` counts
the callable sites since the previous row, this one included.
`--mhs-pair I,J` (default `0,1`, the first diploid sample) selects two
haplotypes, as 0-based positions in the alleles column. A site is
heterozygous when the two differ in every comma-separated ordering, called
when they are equal in every ordering, and missing when the orderings
disagree (a pair across two unphased samples) or an allele is `?` or `N`.
Every row must list as many haplotypes as the first, so remove indels before
`generate_multihetsep.py`, and the rows of each chromosome must form one
block in position order: a chromosome that comes back after another is an
error.

Multihetsep does not say where the uncalled sites are, only how many lie
between two rows. psmc-rs places them at the start of each interval, which
changes which bins are missing compared with fq2psmcfa on the same calls
(QC notes `APPROX_MISSING_PLACEMENT`); for exact binning, use the VCF with a
callable mask. A sequence ends at its last row.

### The sequence filter

VCF and multihetsep sequences go through fq2psmcfa's output filter: a
sequence is dropped when it has fewer than `--min-good-bases` (10,000)
called bases or when they are less than `--min-good-fraction` (0.2) of its
length; 0 disables either. The run lists what it dropped. If nothing is
left, the error says how to keep every sequence (`--min-good-bases 0
--min-good-fraction 0`); `convert` writes empty files instead, as fq2psmcfa
does.

## Commands

| command | what it does |
| --- | --- |
| `psmc-rs INPUT OUTPUT N_ITER` | fit PSMC with N_ITER EM iterations (C `-N`; 0 writes the starting model) |
| `psmc-rs --tui` | the same in a terminal interface; other options fill its form |
| `psmc-rs convert INPUT --psmcfa FILE` | write the input, as read, as psmcfa and/or multihetsep |
| `psmc-rs qc INPUT` | report how suitable an input is, without fitting |

Exit status: 0 on success, 1 on an error (stderr gets `Error: ...` with its
causes), 2 on a usage error and for `qc --strict` when the report has a
warning. Usage errors are options that are invalid alone or together,
also when they are found after parsing (a `--pattern` that does not parse,
outputs that name the same file, `--psmc-out -`); a file that cannot be
read or written is an error. The full option reference is under
[Option reference](#option-reference) and in `psmc-rs --help`,
`psmc-rs convert --help` and `psmc-rs qc --help`.

### Fitting

The model options follow C psmc:

- `--pattern` (C `-p`, default `4+5*3+4`) groups the time intervals into
  free parameters: `a*b` is `a` parameters each spanning `b` intervals, a
  plain `b` is one parameter spanning `b` intervals. `4+5*3+4` has 23
  intervals and 7 free λ; `4+25*2+4+6` has 64 and 28.
- `--n-steps` is the number of intervals minus one (C's `n`). It follows the
  pattern; given with `--pattern` it must agree (the interval count, as 0.1
  took it, is accepted too). `--n-steps` alone fits one λ per interval; the
  M-step's memory and time then grow with the cube of the number of
  intervals (see [Determinism and performance](#determinism-and-performance)).
- `--t-max` (C `-t`, 15) is the initial maximum time in units of 2N0
  generations, in [1, 200]; `--tr-ratio` (C `-r`, 4) sets ρ0 = θ0/ratio.
  `--theta0` and `--rho0` set the starting values directly.
- `--init-params FILE` (C `-i`) starts from the parameters in a JSON output
  of an earlier run (0.1 or 0.2); the file's pattern is used.
- `--smooth-lambda` (default 0, C's objective) adds a penalty on
  differences of log λ between neighbouring intervals.
- `--mstep-iters` (100) caps the L-BFGS iterations of each M-step. C psmc
  runs its optimiser to convergence; about 1000 matches it more closely, at
  ten times the M-step's share of the run time, which is small with the
  patterns above.
- `--mu` (2.5e-8) and `--gen-years` (25) only scale the curve to years and
  individuals: N0 = θ / (4 μ × bin size).

PSMC's transition probabilities assume at most about one recombination per
bin. When ρ0 per bin is too large for the time grid, some of them are
negative and EM cannot move from the start (C psmc then reports a flat
curve): with t_max 15 and flat λ the limit is about 1/6 per bin, which C's
ρ0 = θ0/4 passes at about 49% heterozygous bins. psmc-rs checks the start
before the main fit (with N_ITER 0 too when a bootstrap starts from
`--init-params`) and stops with an error that gives the limit and what to
change: smaller bins, a larger `--tr-ratio` or a smaller `--rho0` (with
the ratio the bootstrap replicates need, when that is more), or, with
`--init-params`, a smaller rho in the saved file.

`Last EM loglike` is the log-likelihood of the written model, which C psmc
prints as the LK of round N_ITER + 1. The JSON's `loglike[i]`, the
log-likelihood after i iterations, is C's LK of round i + 1. With N_ITER 0
the run evaluates the starting model, so `loglike` holds its value alone.

### Bootstrap

`--bootstrap N` fits N replicates after the main fit, as C psmc's
`splitfa` and `psmc -b` do: every sequence is cut into trunks of
`--bootstrap-block-size` bins (default 500,000, splitfa's 50 Mb at 100 bp;
a remainder shorter than 1.5 trunks stays in the last trunk), and each
replicate draws trunks with replacement from the whole input until the
original length is reached, then is fitted from a fresh start (the main
fit's starting t_max and θ0/ρ0 ratio, θ0 from the resampled data) for
`--bootstrap-iters` iterations (default N_ITER). With `--init-params`, the
replicates start from the saved parameters instead, as `psmc -i -b` does;
the run warns that a start fitted to the whole input narrows the bands.
Replicates run in parallel, and replicate i draws from its own seeded
ChaCha8 stream (`--bootstrap-seed`, 42), so a seed gives the same
replicates on any platform and with any `--threads`.

A replicate's fresh start takes θ0 from its own resampled data, so when the
main start is close to the limit on ρ0 (see [Fitting](#fitting)), the
replicates more heterozygous than the input can start past it, from
negative transition probabilities, which EM cannot fit. The run finds them
before the main fit and warns that they will be skipped, that the bands
then come from the other replicates and can be too narrow, and which
`--tr-ratio` (or `--rho0`) would fit them all; it stops before the main
fit when that is every replicate. Skipped replicates get no files, so the
numbering has gaps. Replicates started from `--init-params` share the main
fit's start and are never skipped.

An input that forms a single trunk is refused before fitting, since every
replicate would equal the input; with fewer than 20 trunks the run warns
that the bands are coarse. Pick a trunk size that leaves a few dozen trunks
(`psmc-rs qc INPUT --bootstrap-block-size BINS` counts them).

**Read the bands as a lower bound on the uncertainty.**
`examples/bootstrap_calibration.rs` simulates 40 sequences of 10,000 bins
from a known `4+5*3+4` model and fits them with psmc-rs's default start and
objective (θ0/4, no smoothing) for 20 EM iterations; `cargo run --release
--example bootstrap_calibration` takes about 5 minutes on 4 threads. There
the bootstrap standard deviation of log10 Ne was a median 0.85 of the true
sampling standard deviation with 500,000-bin trunks (here, whole sequences)
and 0.91 with 1,000-bin trunks, below it at 62% to 63% of time points, 0.74
to 0.78 at the oldest times, and 0.68 to 1.03 between datasets. The fits'
median bias (0.094 in log10 Ne, a third of the true standard deviation) does
not show in the band at all. Measured by the interquartile range instead,
which a few runaway fits do not drive, the bootstrap spread was a median
1.01 (whole sequences) and 1.08 (1,000-bin trunks) of the true one, and at
the most recent times whole-sequence trunks gave a band 1.24 times as wide
as the true spread.

### convert

```bash
psmc-rs convert calls.vcf.gz --psmcfa sample.psmcfa.gz
psmc-rs convert calls.vcf.gz --multihetsep-per-chrom sample   # sample.<chrom>.multihetsep
psmc-rs convert sample.psmcfa --psmcfa normalised.psmcfa
```

At least one of `--psmcfa FILE`, `--multihetsep FILE` and
`--multihetsep-per-chrom PREFIX` is required; a name ending in `.gz` is
written compressed (`PREFIX.gz` gives `PREFIX.<name>.multihetsep.gz`).
psmcfa is laid out as fq2psmcfa writes it. Multihetsep is written for VCF
and multihetsep input as `generate_multihetsep.py` would write it for the
selected sample (heterozygous sites only; differences are listed in the
rustdoc of `psmc_rs::io::write::write_multihetsep_to`); psmcfa input has no
sites to write, and asking for multihetsep from it is a usage error.
Written from multihetsep, psmcfa places the uncalled sites as described
above. `--qc-json FILE` writes the QC report. The per-sequence files are
named after the sequences, so they are checked against the input and the
other outputs once the input is read, before anything is written.

### qc

`psmc-rs qc INPUT` loads the input as inference would and prints a report:
bin counts, θ0 (C psmc's starting value), heterozygosity per kb, called
length, sequence lengths (N50), the longest missing run, the share of bins
likely to hold several heterozygous sites, read depth (VCF), a per-sequence
table, and findings with recommendations. `--json FILE` (or
`-` for stdout) writes it as JSON; `--bootstrap-block-size BINS` adds the
trunk checks; `--strict` exits with status 2 when a finding is a warning.
Every fit and conversion prints the same report (`-q` reduces it to its
warnings), and the output JSON keeps its finding codes.

| code | severity | when |
| --- | --- | --- |
| `INVALID_START` | warning | the fit's start (for `qc`, the default start) has negative transition probabilities, from which EM cannot move: ρ0 per bin is too large for the time grid; names the bin size, `--tr-ratio` or `--rho0` that fixes it (a fit then stops, see [Fitting](#fitting)) |
| `VARIANTS_ONLY_NO_MASK` | warning | a VCF read in mask mode without a callable mask |
| `HIGH_MISSING` | warning above 50%, note above 25% | missing bins, or uncalled sites when they are counted |
| `BIN_SATURATION` | warning | more than 10% of heterozygous bins likely hold two or more sites; names the largest bin size that brings it below 5% |
| `FEW_HETS` | warning | fewer than 1,000 heterozygous bins |
| `TOO_SHORT` | warning | less than 100 Mb of called sequence |
| `SHORT_SEQUENCES` | warning | more than 20% of bins in sequences shorter than 1 Mb or than the bootstrap trunk |
| `SEX_OR_ORGANELLE` | note; warning when one looks hemizygous | sequences named like sex chromosomes or organelles |
| `HET_OUTLIER` | note | sequences whose heterozygosity is far from the others' median |
| `APPROX_MISSING_PLACEMENT` | note | multihetsep input (see above) |
| `DEPTH` | warning | consensus mode: mean depth below 18x, or one filter removing more than 30% of the bases |
| `DROPPED_SEQUENCES` | note | the loader left out whole sequences |

The rationale and the sources of each threshold are documented on
`psmc_rs::qc::QcCode` (`cargo doc --open`).

### Terminal interface

`psmc-rs --tui` has three pages: Setup (F1), Run (F2) and Result (F3). The
form takes the input, output, N_ITER (25), pattern (empty for `4+5*3+4`),
generation time, threads, input format, bin size, μ and smoothing; any other
option given with `--tui` is kept, and the page shows the equivalent
command line. On the Setup page every character types into the selected
field: Tab and Shift+Tab move between fields, Space or Left/Right cycle the
input format, Ctrl+O and Ctrl+D pick the input file and the output
directory, F5 or Enter runs, and Esc quits. x or Ctrl+C stops a run (EM and
the bootstrap stop at their next check, and the run shows "Cancelled");
Ctrl+C with no run quits. On the other pages q quits and r runs. PageUp,
PageDown and End scroll the log, and Ctrl+L clears it. `--qc-json -` is
refused with `--tui`, whose screen is stdout.

## Outputs

| file | written | contents |
| --- | --- | --- |
| `OUTPUT` (JSON) | always, right after the main fit | parameters and run metadata (below) |
| `OUTPUT` with `.html` | always | interactive report |
| `--psmc-out FILE` | on request | C psmc's `.psmc` output, rounds 0 to N_ITER |
| `--tmrca-out FILE` | on request | posterior TMRCA of every bin (TSV, gzip when FILE ends in `.gz`) |
| `--qc-json FILE` | on request | the input QC report (`-` for stdout; result lines then go to stderr) |
| `--bootstrap-dir DIR` | with `--bootstrap` | `replicate_NNN.json`, `replicate_NNN.psmc`, `summary.tsv`, `summary.json` (default `OUTPUT` with `.bootstrap` instead of `.json`); an earlier bootstrap's files there are removed first |

Output paths are checked before the input is read: each must be writable,
no two of them, nor any of them and the input, may be the same file, and
only the QC JSON can be `-` (stdout). The `PSMC_ECHARTS_JS` bundle (see
*HTML report*) is read then too.

**JSON.** `theta` and `rho` (per bin), `lam` (one value per free parameter
of the pattern, or per interval without one), `t_max`, `n_steps`, `mu` and
`pattern` are the 0.1 fields, and `--init-params` reads them back. 0.2 adds
`version`, `n_iter`, `final_loglike`, `loglike` (N_ITER + 1 values, see
above), `bin_size`, `gen_years`, `tr_ratio`, `smooth_lambda`, `input`
(format, VCF mode, sequences, bins, heterozygous and missing bins, dropped
sequences) and `qc` (warning and note codes, and key metrics). Replicate
JSON files hold the 0.1 fields.

**HTML report.** The Ne curve (with the bootstrap band and median when there
are replicates), a run summary (input, log-likelihood, pattern, parameters,
N0, bin size, generation time, starting values, smoothing, M-step cap), the
QC findings, metrics and per-sequence table, the bootstrap settings (and
skipped replicates) with a coverage caveat, and, with `--tmrca-out`, TMRCA
panels. Charts can be restyled
and saved as PNG. The page loads ECharts from cdn.jsdelivr.net; for
offline use set `PSMC_ECHARTS_JS` to a local `echarts.min.js` when running
psmc-rs, which embeds it (without either, the page says so).

**.psmc for psmc_plot.pl.** The file follows C psmc's format line for line.
It does not record the bin size, so give psmc_plot.pl the run's settings;
the run prints the command, for example

```text
To plot it: psmc_plot.pl -s 100 -u 2.5e-8 -g 25 -n 25 PREFIX fit.psmc (-s is the bin size, which the .psmc file does not record)
```

psmc_plot.pl plots round 20 unless given `-n`, so pass `-n N_ITER`, as the
printed command does: a run with fewer iterations has no round 20, and in a
longer one round 20 is not the model of `fit.json` and the report. To plot
the bootstrap as C users do, concatenate the files:
`cat fit.psmc fit.bootstrap/replicate_*.psmc > combined.psmc`. The glob
takes every replicate file in the directory, which holds only this run's
(see *Bootstrap files* below). `-n` picks the same round in every file of
the combined one, so the recipe also needs replicates of N_ITER
iterations, the default of `--bootstrap-iters`: with fewer, the replicates
have no round N_ITER and psmc_plot.pl stops with a division by zero; with
more, it plots each replicate at round N_ITER instead of its final fit.
Plot such replicates on their own, with `-n` set to their iteration count;
the run warns and prints that command.

Differences from C's output: the `MM Version` line reads `psmc-rs 0.2.0`;
`IT` counts L-BFGS objective evaluations, which depend on the data and the
pattern (per round, 65 to 124 on the 180,000-bin test input and 155 to 372
on Papuan chr1 with the default pattern, where C's Hooke-Jeeves needs 1,132
to 7,002 and 939 to 1,334); `QD` leaves out the smoothing penalty; and
`post_sigma`, `QD` and `RI` include the emission of each sequence's last bin,
which C skips (differences of the order of the number of sequences divided
by the called length: 2e-5 on the reference data).

**TMRCA TSV** (`--tmrca-out`). One line per bin: `seq_id`, `seq_bin`,
`global_bin`, `obs` (T, K or N), `map_state`, `tmrca_map_years`,
`tmrca_mean_years`, `tmrca_q025_years`, `tmrca_q975_years`, `pmax` and
`entropy` of the posterior. Each state is placed at its average coalescent
time, scaled by `--gen-years`; decoding runs in parallel within and across
sequences.

**Bootstrap files.** `replicate_NNN.json` and `replicate_NNN.psmc` are
numbered by replicate, from 001. `summary.tsv` has `x_years`, `ne_main`,
`ne_q025`, `ne_q500` and `ne_q975` at the main curve's time points;
`summary.json` the same points with the number of replicates summarised
(`n_replicates`), the skipped ones (`skipped_replicates`, numbered from 1,
`[]` when none), the trunk size (`block_size`) and the seed. When the
bootstrap starts, after the main fit's JSON and `--psmc-out` are written,
it removes every `replicate_NNN.json`, `replicate_NNN.psmc`, `summary.tsv`
and `summary.json` in its directory, with a warning that names the
earlier replicates beyond this run's count; other files are left alone. A
bootstrap that fails or is stopped part way therefore leaves only the JSON
files of the replicates it finished, never files of an earlier run. A fit
without `--bootstrap` warns when `OUTPUT.bootstrap` holds replicates.

**Console.** stdout carries only results (`Last EM loglike`, `Parameters`,
`PSMC output`, `QC report`, `Bootstrap ...`, `TMRCA posterior`, `HTML
report`); stderr carries how the input was read, the QC report, the model,
warnings and progress bars. `-q` keeps results and warnings, and
`--no-progress` drops the bars.

## Changes from 0.1

Defaults changed in 0.2.0, so a 0.1 command line without these options
gives a different fit:

| setting | 0.1 | 0.2 | why |
| --- | --- | --- | --- |
| pattern | none: 65 intervals (`--n-steps 64`), 65 free λ | `4+5*3+4` (23 intervals, 7 free λ) | C psmc's default; each free λ needs enough recombinations in its intervals to be determined |
| initial ρ0 | θ0/5 | θ0/4 (`--tr-ratio`) | C psmc's `-r` default |
| `--smooth-lambda` | 1e-3 | 0 | C psmc's objective |
| bootstrap | 50,000-bin blocks within each sequence, replicates started from the fit | splitfa trunks of 500,000 bins across the input, fresh start | C psmc's scheme; the old one gave zero spread for sequences shorter than a block |
| `--input-format` | psmcfa | auto | |
| VCF and multihetsep sequences | all kept | fq2psmcfa's filter | as fq2psmcfa does |
| VCF sequence end (mask mode) | last heterozygous SNP | `##contig` length (`--vcf-no-contig-end`: last record) | the sites after the last heterozygous SNP are data (called where callable) |
| TMRCA years | left edge of each interval | average coalescent time of each state | the time the emission uses |
| `Last EM loglike` | log-likelihood before the last iteration | of the written model | |

`--n-steps 64` without `--pattern` fits 0.1's model layout.
`--mhs-bin-size` is now `--bin-size` and `--tmrca-gen-years` is now
`--gen-years`; the old names still work. The same `--bootstrap-seed` draws
different replicates than in 0.1. 0.1 also read psmcfa headers as bins (one
per T, K or N in names such as GenBank accessions) and skipped IUPAC codes,
digits and lowercase letters, and it misplaced missing sites in VCF input.
Inputs, flags and outputs are covered in detail in [CHANGELOG.md](CHANGELOG.md).

## Determinism and performance

Results do not depend on `--threads` or `--batch-size`: a run writes the
same bytes (JSON, `.psmc`, TMRCA and bootstrap files) with any thread
count, which the tests check by running the binary. Instead of whole
forward and backward matrices, the E-step keeps one forward vector every
128 bins and recomputes each 128-bin segment for the backward pass, so its
memory still grows with sequence length, but at 1/128 of the rate: about
bins / 128 × states × 8 bytes per sequence in flight (plus about 0.3 MB).
With pattern `4+25*2+4+6` (64 states), one iteration on one thread peaked
at 21 MB on Papuan chr1 (2.49 M bins) and at 108 MB on a single sequence
of eight copies of it (19.9 M bins). `--batch-size` is no longer needed
for memory. Sequences are processed in parallel, so `--threads` helps
inputs of many sequences; a single long sequence runs on one thread.

Measured on the inputs in `experiment/real_data` with pattern
`4+25*2+4+6`, on an Apple M1 Pro (8 cores, 16 GB, macOS 27, Rust 1.97,
release builds), one run each on a machine shared with other work (load
average 1 to 6); wall time and peak resident memory from `/usr/bin/time -l`:

| input | EM iterations | threads | psmc-rs 0.2 | psmc-rs 0.1 | C psmc 0.6.5 (`-O1`) |
| --- | --- | --- | --- | --- | --- |
| Papuan chr1: 1 sequence, 2.49 M bins | 20 | 1 | 24.7 s, 21 MB | 139 s, 1.95 GB | 312 s, 2.63 GB |
| | 20 | 8 | 24.9 s, 22 MB | | |
| HLhydTec1: 34 sequences, 18.5 M bins | 5 | 1 | 44.2 s, 87 MB | | 588 s, 3.40 GB |
| | 5 | 8 | 9.9 s, 96 MB | 78.7 s, 6.88 GB | |

The psmc-rs 0.2 figure for the first row is the median of three runs
(24.7 to 24.8 s); C psmc is single-threaded, and 0.1 started from θ0/5.
The JSON outputs at 1 and 8 threads were byte-identical.

Use a release build, and `-q --no-progress` for batch runs. The model has
psmc's PSMC_T_INF = 1000 time grid and at most 10,000 states; models of more
than 256 states use dense kernels, which cache forward rows within a
process-wide budget of `PSMC_ALPHA_CACHE_MB` (2048; 0 disables the cache).

The M-step's memory and time grow with the number of free λ. It
differentiates its objective in forward mode, so each evaluation holds the
transition matrix with the gradient of every free parameter: about
states² × (free λ + 7) × 8 bytes, with as many operations. A pattern keeps
this small (64 states and 28 free λ: about 1 MB), but with one λ per
interval (`--n-steps` alone) it grows with the cube of the number of states
and takes over the run: about 0.2 GB at 301 states, 0.5 GB at 401 and
8 GB at 1001, and the measured peaks were 1.3 to 2.6 times these figures.
On 42,000 bins, one iteration with 201 states took 3.9 s with one λ per
interval and 0.3 s with a single λ. Before it reads the input, a run warns
when the M-steps that may run at once need more than 1 GB together, taking
2.5 times the figure above for each. A fit runs one M-step at a time, but
a bootstrap fits up to one replicate per thread at once (`--threads`, by
default one per core or `RAYON_NUM_THREADS`), each with its own M-step:
`--n-steps 300` needs about 0.6 GB per M-step, and 2.2 GB for a bootstrap
on 4 threads.

## Testing and C equivalence

```bash
cargo test                        # unit, integration and doc tests
python3 -m unittest discover -s experiment/tests
```

The suite compares psmc-rs with the original tools' outputs, stored under
`tests/data/` with the scripts that made them:

- `tests/c_equivalence.rs`: EM on a 180,000-bin psmcfa against C psmc
  0.6.5-r74 (patterns `4+5*3+4` and `4+25*2+4+6`): round 1 equal to C's
  printed precision, rounds 2 to 5 within a relative 1e-5.
- `tests/psmc_out_c_reference.rs`, `tests/em_c_reference.rs` and
  `tests/psmc_out_plot.rs`: the `.psmc` output line by line against C's,
  C's expectations (`post_sigma`, Q) and psmc_plot.pl's reading of the file.
- `tests/io_text_psmcfa_c.rs`: psmcfa parsing against C psmc's counts on the
  real data in `experiment/real_data`.
- `tests/io_route_a.rs`: consensus mode against goldens made with
  `bcftools call -c | vcfutils.pl vcf2fq | fq2psmcfa`, and multihetsep output
  against `generate_multihetsep.py`, byte for byte.
- `tests/bootstrap_split.rs` and `tests/bootstrap_resample.rs`: trunks
  against `splitfa`, draws against `psmc_resamp`.

Two ignored tests run the tools themselves:
`VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa cargo test
--release --test io_consensus_tools -- --ignored` and `FQ2PSMCFA=...
cargo test --release --test io_vcf_exact -- --ignored`. CONTRIBUTING.md
describes how reference outputs may be regenerated.

## Reproducing the manuscript

`experiment/` holds the scripts, data and notebook behind the manuscript's
figures and tables; [experiment/README.md](experiment/README.md) describes
the pipeline (simulate, run psmc-rs and C psmc with matched settings, draw
the figures). The packaged results were made with 0.1 and are being
regenerated with 0.2.

## Citing

If you use psmc-rs, please cite it as given in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository"), together with the PSMC paper:

> Li H, Durbin R (2011). Inference of human population history from
> individual whole-genome sequences. *Nature* 475:493-496.
> doi:10.1038/nature10231

## Contributing and license

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
psmc-rs is released under the MIT license ([LICENSE](LICENSE)). The C psmc
outputs under `tests/data/c_reference/` come with C psmc's license
(`PSMC_LICENSE.txt` there).

## Option reference

From `psmc-rs --help` (0.2.0), which gives the full text. Inference,
`convert` and `qc` share the input options.

**Input** (inference, `convert`, `qc`)

| option | default | meaning |
| --- | --- | --- |
| `--input-format auto\|psmcfa\|mhs\|vcf` | `auto` | input format; auto recognises it from the content |
| `--bin-size BP` | 100 | bases per bin: bins VCF and multihetsep input (fq2psmcfa `-s`), scales every format |
| `--batch-size BINS` | none | split sequences into HMM rows of at most BINS bins; results do not change |
| `--mhs-pair I,J` | `0,1` | multihetsep: the two haplotypes, 0-based positions in the alleles column |
| `--vcf-sample NAME` | | VCF mask mode: the sample (required when there are several); consensus mode needs a one-sample VCF |
| `--vcf-mode auto\|consensus\|mask` | `auto` | VCF: consensus (`vcf2fq \| fq2psmcfa`) or mask mode |
| `--vcf-mask BED` | | VCF: callable sites (repeatable; a site must be in every one) |
| `--vcf-negative-mask BED` | | VCF: sites to exclude (repeatable) |
| `--vcf-keep-filtered` | off | VCF mask mode: use records whose FILTER is not PASS or `.` |
| `--vcf-no-contig-end` | off | VCF mask mode: end sequences at their last record, not the `##contig` length |
| `--depth auto\|MIN:MAX` | `auto` | VCF consensus mode: INFO/DP bounds (vcf2fq `-d`, `-D`); auto = a third and twice the mean |
| `--min-mapq MQ` | 10 | VCF consensus mode: minimum INFO/MQ (vcf2fq `-Q`) |
| `--indel-window BP` | 5 | VCF consensus mode: bases masked around indels (vcf2fq `-l`) |
| `--min-qual Q` | 10 | VCF consensus mode: minimum base quality (fq2psmcfa `-q`) |
| `--min-good-bases BASES` | 10000 | VCF and multihetsep: drop sequences with fewer called bases (fq2psmcfa `-g`; 0 disables) |
| `--min-good-fraction SHARE` | 0.2 | VCF and multihetsep: drop sequences whose called share is lower (0 disables) |

**Inference** (`psmc-rs [OPTIONS] INPUT OUTPUT N_ITER`)

| option | default | meaning |
| --- | --- | --- |
| `--tui` | | open the terminal interface; the other arguments fill its form |
| `--threads N` | one per core | worker threads (E-step, bootstrap, TMRCA); results do not depend on it |
| `--no-progress` | | no progress bars |
| `-q`, `--quiet` | | only results and warnings |
| `--pattern P` | `4+5*3+4` | free-parameter pattern (C `-p`) |
| `--n-steps N` | from the pattern | intervals minus one (C's `n`, 1 to 9999); alone, one λ per interval |
| `--t-max T` | 15 | initial t_max in units of 2N0 generations (C `-t`; 1 to 200) |
| `--tr-ratio RATIO` | 4 | initial θ0/ρ0 (C `-r`) |
| `--theta0 X` | −ln(1 − K/L) | initial θ0 per bin |
| `--rho0 X` | θ0 / `--tr-ratio` | initial ρ0 per bin |
| `--init-params FILE` | | start from a saved fit (C `-i`) |
| `--mu MU` | 2.5e-8 | mutation rate per bp per generation, for scaling |
| `--mstep-iters N` | 100 | maximum L-BFGS iterations per M-step |
| `--smooth-lambda W` | 0 | penalty on log-λ differences of neighbouring intervals |
| `--bootstrap N` | 0 | bootstrap replicates |
| `--bootstrap-block-size BINS` | 500000 | trunk size (splitfa); alias `--bootstrap-trunk-size` |
| `--bootstrap-seed SEED` | 42 | seed of the resampling |
| `--bootstrap-iters N` | N_ITER | EM iterations per replicate |
| `--bootstrap-dir DIR` | `OUTPUT` with `.bootstrap` | replicate and summary files |
| `--qc-json FILE` | | input QC report as JSON (`-` for stdout) |
| `--psmc-out FILE` | | C psmc's `.psmc` output |
| `--tmrca-out FILE` | | posterior TMRCA per bin (TSV; gzip for `.gz`) |
| `--gen-years YEARS` | 25 | years per generation, for the time axes |

**convert** (`psmc-rs convert [OPTIONS] INPUT`, at least one output)

| option | meaning |
| --- | --- |
| `--psmcfa FILE` | psmcfa as fq2psmcfa lays it out (gzip for `.gz`) |
| `--multihetsep FILE` | all sequences as multihetsep (VCF and multihetsep input) |
| `--multihetsep-per-chrom PREFIX` | one file per sequence, `PREFIX.<name>.multihetsep` |
| `--qc-json FILE` | input QC report as JSON (`-` for stdout) |
| `-q`, `--quiet` | only results and warnings |

**qc** (`psmc-rs qc [OPTIONS] INPUT`)

| option | meaning |
| --- | --- |
| `--json FILE` | also write the report as JSON (`-` for stdout; the text then goes to stderr) |
| `--bootstrap-block-size BINS` | check the input for a bootstrap with trunks of BINS bins |
| `--strict` | exit with status 2 when the report has a warning |
