# psmc-rs

psmc-rs is a Rust implementation of PSMC, the Pairwise Sequentially Markovian
Coalescent of Li and Durbin (2011), which infers the history of effective
population size from the two genome copies of one diploid individual.

It fits the same model as Heng Li's C `psmc`, with the same time grid,
starting point (θ0 = −ln(1 − K/L), ρ0 = θ0/4, flat λ, t_max = 15) and `-p`
patterns. With C's initialisation the first round's log-likelihood equals
C psmc's to its printed precision, and later rounds,
which depend on the M-step optimiser (L-BFGS here, Hooke-Jeeves in C; both
run to convergence), stay within a relative 2e-7 on the reference data;
`tests/c_equivalence.rs` enforces both. One extension, on by default for
VCF input, weighs each bin by how many of its bases are callable, which
removes most of the bias that missing data scattered in short runs cause;
complete data and psmcfa input are fitted exactly as C psmc fits them, and
so is multihetsep input, which does not say which of its sites are uncalled
(see [Partly callable bins](#partly-callable-bins)). On top of the model,
psmc-rs

- reads psmcfa, VCF (including the classic `vcfutils.pl vcf2fq | fq2psmcfa`
  route in one step) and MSMC's multihetsep, plain or gzip/bgzip;
- checks an input before fitting (`psmc-rs qc`) and converts between formats
  (`psmc-rs convert`);
- runs C psmc's bootstrap in one command, in parallel;
- writes JSON parameters, an interactive HTML report, C psmc's `.psmc` output
  for `psmc_plot.pl`, and per-bin TMRCA posteriors;
- is deterministic and multithreaded, and its E-step keeps 1/128 of the
  forward-backward matrices (a fit of human chromosome 1 peaks at 19 MB).

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

`fit.json` holds the parameters, `fit.html` the report. The run prints how
it read the input (for psmcfa, C psmc's `n_seqs`, `sum_L` and `sum_n`), the
QC report and the model on stderr, and its results on stdout:

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
from its magic bytes, every member of a bgzip file is read, and the
decoding runs in a thread of its own, beside the parsing). A bgzip
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
`vcfutils.pl vcf2fq | fq2psmcfa`, in one step from the all-sites VCF of
`bcftools call -c`, with bins identical to fq2psmcfa's (checked byte
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
changes which bins are missing compared with fq2psmcfa on the same calls:
uncalled sites scattered among called ones become runs of `N` bins after
heterozygous sites, and the bin of a heterozygous site is partly callable
or lost. With missing data this biases θ, the times and Ne low (on E4's
simulated genomes θ came out 0.67 to 0.69 of the fit without missing data
with 30% of the sites uncalled in runs of about 20 bp, and 0.89 to 0.91 in
runs of about 20 kb), and QC warns (`APPROX_MISSING_PLACEMENT`) when more
than 5% of the sites are uncalled. The callable bases of each bin follow
the same placement: their sum over each interval is exact, how it is split
among the bins is not. `--bin-callable auto` therefore fits every
called bin of multihetsep input as `--bin-size` bases, as C psmc would fit
its bins; `--bin-callable approximate` fits the approximate counts, which
made the fit worse in the validation ([Partly callable
bins](#partly-callable-bins)). For exact bins and counts, use the VCF with
a callable mask. A sequence ends at its last row.

### Partly callable bins

PSMC sees a genome as bins of `--bin-size` bases. fq2psmcfa calls a bin as
soon as a tenth of its bases are callable, and C psmc treats every called
bin as a full bin. A bin of `c` callable bases out of `s` is less likely to
show a heterozygous site, so when missing data are scattered in runs
shorter than a bin (per-site depth or quality filters, a mask of short
gaps), almost every bin is still called but holds fewer bases, and θ per
bin, N0 and with them every time and every Ne come out low by about 1.2
times the missing share of the called bins. Missing data in long runs
mostly make whole bins `N`, which C's model handles.

VCF input (both modes) records which positions of each bin are callable.
By default (`--bin-callable auto`) psmc-rs fits each called bin of a VCF by
its count: a bin of `c` callable bases emits as `c` bases of sequence,

```text
P(T | k, c) = exp(-θ (c / s) avg_t[k]),   P(K | k, c) = 1 - P(T | k, c),   P(N | k, c) = 1
```

with θ still per `s` bases, so N0 = θ / (4 μ s) and `psmc_plot.pl -s` keep
their meaning. The E-step, the M-step, the starting θ0, the bootstrap and
the TMRCA decoding all use the counts. The starting θ0 solves `sum over
called bins of (1 - exp(-θ0 c / s)) = K`, the moment estimate that C's
−ln(1 − K/L) is for full bins. With every called bin full (complete data)
the model, the fit and every output are C psmc's, to the bit, and psmcfa
input, which records no counts, is fitted as C psmc fits it.
`--bin-callable ignore` drops the counts and treats every called bin as
`s` bases, as fq2psmcfa and C psmc do: a VCF fitted with it gives exactly
the fit of fq2psmcfa's psmcfa of the same calls. Counts are kept for bins
of at most 65,535 bases.

On five simulated 500 Mb genomes of a bottleneck model (the manuscript's
experiment E4, `-N25 -t15 -r5 -p 4+25*2+4+6`) with 30% of the bases
missing in runs of about 20 bp, the fit with the counts removed most of
the bias: θ was 0.97 to 0.99 of the fit of the same genome without missing
data (1.9% low on average), where ignoring the counts gave 0.64 to 0.66,
and the RMSE of log10 Ne against the truth was 0.007 above that of the
complete-data fit on average (at most 0.020), where ignoring the counts
raised it by 0.16 to 0.20. Missing data in 20 kb runs changed the fits
little either way. Multihetsep input does not recover the exact fit
([multihetsep](#multihetsep)); its approximate counts made θ 0.52 to 0.60
of the complete-data fit in the smaller runs of
[experiment/validation/](experiment/validation/README.md) (three runs of
three 10 × 20 Mb genomes), so `--bin-callable auto` does not use them.
E4's numbers are in
`experiment/runs/E4/tables/` and its design in
[experiment/pipeline/README.md](experiment/pipeline/README.md#e4-input-preparation).

A psmcfa file cannot hold the counts: `psmc-rs convert --psmcfa` warns when
it writes partly callable bins, and such a file is fitted as with
`--bin-callable ignore`. To check how much of an existing psmcfa's called
bins is callable, run `psmc-rs qc` on the VCF it was made from.

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

Exit status: 0 on success, 1 on an error (stderr gets `error: ...` with its
causes, as for clap's own errors; warnings are `warning: ...`), 2 on a
usage error and for `qc --strict` when the report has a warning. Usage
errors are options that are invalid alone or together, also when they are
found after parsing (a `--pattern` that does not parse, outputs that name
the same file, `--psmc-out -`); a file that cannot be read or written is
an error. Without arguments, `psmc-rs`, `psmc-rs convert` and `psmc-rs qc`
print their help, with status 2. `-h` gives a one-sentence summary of each
option; a table of the options is under
[Option reference](#option-reference), and the full text is in
`psmc-rs --help`, `psmc-rs convert --help` and `psmc-rs qc --help`.

### Fitting

The model options follow C psmc:

- `--pattern` (C `-p`, default `4+5*3+4`) groups the time intervals into
  free parameters: `a*b` is `a` parameters each spanning `b` intervals, a
  plain `b` is one parameter spanning `b` intervals. `4+5*3+4` has 23
  intervals and 7 free λ; `4+25*2+4+6` has 64 and 28.
- `--n-steps` is the number of intervals minus one (C's `n`). It follows the
  pattern; given with `--pattern` it must agree (the interval count, as 0.1
  took it, is accepted too). `--n-steps` alone fits one λ per interval
  (see [Determinism and performance](#determinism-and-performance) for
  what that costs).
- `--t-max` (C `-t`, 15) is the initial maximum time in units of 2N0
  generations, in [1, 200]; `--tr-ratio` (C `-r`, 4) sets ρ0 = θ0/ratio.
  `--theta0` and `--rho0` set the starting values directly.
- `--init-params FILE` (C `-i`) starts from the parameters in a JSON output
  of an earlier run (0.1 or 0.2); the file's pattern is used.
- `--smooth-lambda` (default 0, C's objective) adds a penalty on
  differences of log λ between neighbouring intervals.
- `--mstep-iters` (1000) caps the L-BFGS iterations of each M-step, which
  runs to convergence, as C psmc's does: in the manuscript's 40 E1 fits
  with `4+25*2+4+6`, an M-step took a median 76 evaluations of its
  objective and at most 135 (the `.psmc` output's `IT`; these include the
  evaluations for its curvatures and its stopping check, two per free
  parameter), where C psmc's Hooke-Jeeves search took a median 3,934.
  The cap is a safeguard that normal fits do not reach.
- The M-step keeps each λ between 1e-6 and 1e4 (C psmc's search has no
  bound). Beyond 1e4 the f64 transition probabilities of the shortest,
  earliest intervals are rounding noise; a λ of 1e4 already means that
  the data hold almost no coalescences in its intervals, so the Ne there
  is the bound, not an estimate. This is common for the most recent
  intervals when each has its own λ, and the run warns when a fitted
  parameter sits on a bound.
- `--mu` (2.5e-8) and `--gen-years` (25) only scale the curve to years and
  individuals: N0 = θ / (4 μ × bin size).
- `--bin-callable auto|ignore|approximate` (default `auto`): weigh each
  called bin of VCF input by its callable bases (`auto`), or treat every
  called bin as `--bin-size` bases, as C psmc does (`ignore`);
  `approximate` also weighs the bins of multihetsep input by its
  approximate counts, which `auto` leaves out (see
  [Partly callable bins](#partly-callable-bins)). The run says what the
  counts hold and which model it fits; for psmcfa input `ignore` and
  `approximate` change nothing, and the run says so.

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

**Read the bands as a lower bound on the uncertainty, and not as an
interval for the true history.** The manuscript's experiment E5
([experiment/pipeline/README.md](experiment/pipeline/README.md#e5-bootstrap-calibration))
bootstrapped 10 simulated 300 Mb genomes of a bottleneck model (6
chromosomes of 50 Mb, 50 replicates each, `-N25 -t15 -r5 -p 4+25*2+4+6`)
and compared each band with the spread of 50 independent fits of the
model, at 200 times of its informative window (`E5_summary.csv`). The
bootstrap standard deviation of log10 Ne was a median 0.84 of the true
sampling standard deviation with the default 50 Mb trunks (one per
chromosome) and 0.93 with 5 Mb trunks, below it at 84% and 67% of the
times. The 95% bands held the true Ne at only 37.2% (50 Mb trunks) and
43.0% (5 Mb) of the times, averaged over the genomes, while they held the
mean of the 50 fits at 82.7% and 91.8%: most of the shortfall is PSMC's
bias, which a bootstrap cannot see (its median, 0.061 in log10 Ne, is 2.2
times the true standard deviation, 0.028), at the bottleneck and the size
changes above all. Part of that bias is the pattern's, whose steps cannot
follow the truth within a free λ's span: the bands held the truth averaged
over each span at 44.4% and 50.5% of the times. A smaller version of the
check, `cargo run --release --example bootstrap_calibration`
(`examples/bootstrap_calibration.rs`: 40 sequences of 10,000 bins of a
`4+5*3+4` model), runs in about 5 minutes on 4 threads.

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
above. psmcfa records only each bin's symbol, not its callable bases, so a
fit of the psmcfa treats every called bin as full, as C psmc does; when
some called bins of a VCF are partly callable, `convert` warns with their
share and the bias that causes, and advises fitting the VCF itself. The
multihetsep writer keeps `nr_called`, from which the reader rebuilds
approximate counts, which a fit uses only with `--bin-callable
approximate`. `--qc-json FILE` writes the QC report. The
per-sequence files are named after the sequences, so they are checked
against the input and the other outputs once the input is read, before
anything is written.

### qc

`psmc-rs qc INPUT` loads the input as inference would and prints a report:
bin counts, θ0 (C psmc's starting value), heterozygosity per kb, called
length, the callable bases of the called bins (VCF and multihetsep: their
mean share and distribution), sequence lengths (N50), the longest missing
run, the share of bins likely to hold several heterozygous sites, read
depth (VCF), a per-sequence table, and findings with recommendations. With
callable bases, sequences are compared per callable base (the table's
het/kb, marked `het/kb*` with a note under the table, and the sex
chromosome and outlier checks), so that one with more missing data does
not look less heterozygous; the totals' θ0 and het per kb stay per kb of
called bins, as C psmc counts them, and the fit's JSON keeps both
(`qc.het_per_kb`, `qc.het_per_callable_kb`). `--json FILE` (or
`-` for stdout) writes it as JSON; `--bootstrap-block-size BINS` adds the
trunk checks; `--strict` exits with status 2 when a finding is a warning.
Every fit and conversion prints the same report (`-q` reduces it to its
warnings), and the output JSON keeps its finding codes. A finding is a
warning or a note: the text report labels them `WARNING` and `INFO`, the
QC JSON gives their `severity` as `warning` and `info`, and a fit's JSON
lists their codes under `qc.warnings` and `qc.notes`.

| code | severity | when |
| --- | --- | --- |
| `INVALID_START` | warning | the fit's start (for `qc`, the default start) has negative transition probabilities, from which EM cannot move: ρ0 per bin is too large for the time grid; names the bin size, `--tr-ratio` or `--rho0` that fixes it, or for `--init-params` the rho to set in the saved file (a fit then stops, see [Fitting](#fitting)) |
| `VARIANTS_ONLY_NO_MASK` | warning | a VCF read in mask mode without a callable mask |
| `PARTLY_CALLABLE_BINS` | note; warning with `--bin-callable ignore` below a 95% mean | at least 1% of called bins of a VCF hold fewer than `--bin-size` callable bases: gives their share, mean and distribution and the bias of treating them as full, which the fit corrects by default; for psmcfa input, which cannot tell, the called bins next to missing runs and how to measure the rest; not given for multihetsep, whose counts follow its placement (`APPROX_MISSING_PLACEMENT`) |
| `HIGH_MISSING` | warning above 50%, note above 25% | missing bins, or uncalled sites when they are counted |
| `BIN_SATURATION` | warning | more than 10% of heterozygous bins likely hold two or more sites; names the largest bin size that brings it below 5% |
| `FEW_HETS` | warning | fewer than 1,000 heterozygous bins |
| `TOO_SHORT` | warning | less than 100 Mb of called sequence |
| `SHORT_SEQUENCES` | warning | more than 20% of bins in sequences shorter than 1 Mb or than the bootstrap trunk |
| `SEX_OR_ORGANELLE` | note; warning when one looks hemizygous | sequences named like sex chromosomes or organelles |
| `HET_OUTLIER` | note | sequences whose heterozygosity is far from the others' median |
| `APPROX_MISSING_PLACEMENT` | note; warning when more than 5% of the sites are uncalled | multihetsep input, whose reader places the uncalled sites (see above): gives their share and the bias that placement caused in simulations, whatever `--bin-callable` |
| `DEPTH` | warning | consensus mode: mean depth below 18x, or one filter removing more than 30% of the bases |
| `DROPPED_SEQUENCES` | note | the loader left out whole sequences |

The rationale and the sources of each threshold are documented on
`psmc_rs::qc::QcCode` (`cargo doc --open`).

### Terminal interface

`psmc-rs --tui` has three pages, Setup (F1), Run (F2) and Result (F3), and
a line of the keys that apply. The form takes the input, output, N_ITER
(25), pattern (empty for `4+5*3+4`), generation time, threads, input
format, bin size, μ and smoothing; any other option given with `--tui` is
kept, and the page shows the equivalent command line, or what the form
still lacks. On the Setup page every character types into the selected
field: Tab, Shift+Tab and Up/Down move between fields, Ctrl+U clears one,
Space or Left/Right change the input format, Ctrl+O and Ctrl+D pick the
input file and the output directory, F5 or Enter runs, and Esc quits. x or
Ctrl+C stops a run (EM and the bootstrap stop at their next check, and the
run shows "Cancelled"); Ctrl+C with no run quits. A run that fails goes
back to the Setup page with the error in its log; one that finishes opens
the Result page, which lists what it wrote. On the other pages q quits and
r runs. Up/Down and PageUp/PageDown scroll the log, End goes back to its
end, and Ctrl+L clears it. `--qc-json -` is refused with `--tui`, whose
screen is stdout.

## Outputs

| file | written | contents |
| --- | --- | --- |
| `OUTPUT` (JSON) | always, right after the main fit | parameters and run metadata (below) |
| `OUTPUT` with `.html` | always | interactive report |
| `--psmc-out FILE` | on request | C psmc's `.psmc` output, rounds 0 to N_ITER |
| `--tmrca-out FILE` | on request | posterior TMRCA of every bin (TSV, gzip when FILE ends in `.gz`) |
| `--qc-json FILE` | on request | the input QC report (`-` for stdout; result lines then go to stderr) |
| `--bootstrap-dir DIR` | with `--bootstrap` | `replicate_NNN.json`, `replicate_NNN.psmc`, `summary.tsv`, `summary.json` (default `OUTPUT` with `.bootstrap` in place of `.json`); an earlier bootstrap's files there are removed first |

Output paths are checked before the input is read: each must be writable,
no two of them, nor any of them and the input, may be the same file, and
only the QC JSON can be `-` (stdout). The `PSMC_ECHARTS_JS` bundle (see
*HTML report*) is read then too.

**JSON.** `theta` and `rho` (per bin), `lam` (one value per free parameter
of the pattern, or per interval without one), `t_max`, `n_steps`, `mu` and
`pattern` are the 0.1 fields, and `--init-params` reads them back. 0.2 adds
`version`, `simd` (the HMM kernels that ran: `avx2+fma`, `portable` or
`neon`), `n_iter`, `final_loglike`, `loglike` (N_ITER + 1 values, see
above), `bin_size`, `gen_years`, `tr_ratio`, `smooth_lambda`, `input`
(format, VCF mode, sequences, bins, heterozygous and missing bins, dropped
sequences), `qc` (warning and note codes, and key metrics) and
`bin_callable`: the `--bin-callable` mode, the emission model fitted
(`callable-length`, each called bin by its callable bases, or `full-bins`,
C psmc's), how the input's counts were obtained (`exact` from a VCF,
`approximate` from multihetsep; absent for psmcfa) and what they hold
(`callable`: called and partly callable bins, the mean share of callable
bases and its distribution in tenths). Replicate JSON files hold the 0.1
fields.

**HTML report.** The Ne curve (with the bootstrap band and median when there
are replicates), a run summary (input, bin model and callable bases,
log-likelihood, pattern, parameters, N0, bin size, generation time,
starting values, smoothing, M-step cap), the QC findings, metrics and
per-sequence table, the bootstrap settings (and
skipped replicates) with a coverage caveat, and, with `--tmrca-out`, TMRCA
panels. Charts can be restyled and saved as PNG. The page loads ECharts
from cdn.jsdelivr.net; for offline use set `PSMC_ECHARTS_JS` to a local
`echarts.min.js` when running psmc-rs, which embeds it (without either,
the page says so).

**.psmc for psmc_plot.pl.** The file follows C psmc's format line for line,
but for C's `QD` line (below). It does not record the bin size, so give
psmc_plot.pl the run's settings; the run prints the command, for example

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
the run warns and prints that command. With replicates of N_ITER
iterations and `--psmc-out`, it prints the recipe:

```text
To plot the fit with its bootstrap replicates: cat fit.psmc fit.bootstrap/replicate_*.psmc > combined.psmc; psmc_plot.pl -s 100 -u 2.5e-8 -g 25 -n 25 PREFIX combined.psmc
```

Differences from C's output: the `MM Version` line reads `psmc-rs 0.2.0`;
for VCF and multihetsep input, one more `MM` line after `n_seqs` says how
the fit used the callable bases of each bin, such as
`bin_model:callable-length, counts:exact, mean_callable:0.701234,
partial_bins:0.850000` (bootstrap replicates fitted with the counts get it
without `counts`; psmc_plot.pl skips `MM` lines it does not know, and
psmcfa input gets none);
`IT` counts L-BFGS objective evaluations, which depend on the data and the
pattern (per round with the default pattern, 34 to 74 on the 180,000-bin
test input and 32 to 72 on Papuan chr1, where C's Hooke-Jeeves needs 1,132
to 1,540 and 939 to 1,334); `post_sigma` and `RI` include the emission of
each sequence's last bin, which C skips (differences of the order of the
number of sequences divided by the called length: 2e-5 on the reference
data); and there is no `QD` line, C's EM Q before and after the M-step less
Q's largest value over all HMMs, because that value needs every expected
transition count, which psmc-rs does not keep (see *Determinism and
performance*; psmc_plot.pl does not read `QD`).

**TMRCA TSV** (`--tmrca-out`). One line per bin: `seq_id`, `seq_bin`,
`global_bin`, `obs` (T, K or N), `map_state`, `tmrca_map_years`,
`tmrca_mean_years`, `tmrca_q025_years`, `tmrca_q975_years`, `pmax` and
`entropy` of the posterior. Each state is placed at its average coalescent
time, scaled by `--gen-years`; each bin is decoded with the emissions the
fit used (of its callable bases, with the callable-length model); decoding
runs in parallel within and across sequences.

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
without `--bootstrap` warns when that directory holds replicates.

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
| M-step | at most 100 L-BFGS iterations (`--mstep-iters`) | runs to convergence; `--mstep-iters` (1000) only caps it | C psmc's M-step runs to convergence; 100 iterations left the curve 0.03 to 0.16 (RMSE of log10 Ne) from C's |
| λ range of the M-step | 1e-6 to 1e6 | 1e-6 to 1e4 | above 1e4 the earliest intervals' transition probabilities are rounding noise in f64 |
| bootstrap | 50,000-bin blocks within each sequence, replicates started from the fit | splitfa trunks of 500,000 bins across the input, fresh start | C psmc's scheme; the old one gave zero spread for sequences shorter than a block |
| `--input-format` | psmcfa | auto | |
| VCF and multihetsep sequences | all kept | fq2psmcfa's filter | as fq2psmcfa does |
| partly callable bins (VCF) | treated as full bins | weighted by their callable bases (`--bin-callable auto`; `ignore` gives C psmc's model; multihetsep's approximate counts only with `approximate`) | missing data in short runs otherwise bias θ, times and Ne low by the missing share or more |
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
count, which the tests check by running the binary. Gzip outputs are
compressed by zlib-rs, whose version `Cargo.lock` pins: another version
of psmc-rs may compress the same text to other bytes, so compare the
`.gz` outputs of two versions decompressed (`gzip -dc`).
With models of up to
256 states, the E-step's forward and backward steps take O(states) time per
bin instead of O(states²), from the rank-one triangles of PSMC's transition
matrix, with which Harris et al. (2014) decode coalescent HMMs in linear
time (see [Citing](#citing)). With models of any size, so do the expected
transition counts: below and above the diagonal, a transition probability
is a factor of its row times a factor of its column, so C psmc's EM Q, and
the M-step with it, needs the counts only summed over each triangle's rows
and columns and over the diagonal, 5 × states sums, which the E-step adds
up from each bin's prefix and suffix sums instead of the whole
states × states matrix of counts. Instead of whole forward and backward
matrices, the E-step keeps one forward vector every 128 bins and
recomputes each 128-bin segment for the backward pass, so its
memory still grows with sequence length, but at 1/128 of the rate: about
bins / 128 × states × 8 bytes per sequence in flight (plus about 0.3 MB).
With pattern `4+25*2+4+6` (64 states), one iteration on one thread peaked
at 19 MB on Papuan chr1 (2.49 M bins) and at 108 MB on a single sequence
of eight copies of it (19.9 M bins). Sequences are processed in parallel,
each on one thread, so `--threads` helps inputs of many sequences: a
single long sequence runs on one thread, and t threads are up to t times
as fast only on at least t sequences of similar length. The threads left
without a sequence add no CPU time: in E3 (below), 25 EM iterations on
one simulated 100 Mb chromosome (1 M bins) took a median 10.3 to 10.5 s
of CPU time and 10.45 to 10.98 s of wall time on 1, 8 or 30 threads of
an AMD EPYC 7702 alone on its node.

Measured on the inputs in `experiment/real_data` with pattern
`4+25*2+4+6`, on an Apple M1 Pro (8 cores, 16 GB, macOS 27, Rust 1.97,
release builds) shared with other work; wall time and peak resident memory
from `/usr/bin/time -l`:

| input | EM iterations | threads | psmc-rs 0.2 | C psmc 0.6.5 (`-O1`) |
| --- | --- | --- | --- | --- |
| Papuan chr1: 1 sequence, 2.49 M bins | 20 | 1 | 15.8 s, 19 MB | 312 s, 2.63 GB |
| | 20 | 8 | 15.9 s, 19 MB | |
| HLhydTec1: 34 sequences, 18.5 M bins | 5 | 1 | 30.8 s, 46 MB | 588 s, 3.40 GB |
| | 5 | 8 | 7.2 s, 92 MB | |

psmc-rs: medians of three runs, at load averages of 3 to 11; C psmc: one
run each, at load averages of 1 to 6. C psmc is single-threaded. The
JSON outputs at 1 and 8 threads were byte-identical.

The manuscript's experiment E3 timed the tools under a controlled protocol
on AMD EPYC 7702 nodes of KAUST's Ibex cluster: each run alone on an
exclusive node, 5 repeats on five nodes, 25 EM iterations of
`-N25 -t15 -r5 -p "4+25*2+4+6"` on simulated genomes of 100 Mb
chromosomes, C psmc built by gcc with `-O3 -march=x86-64-v3`. On 3000 Mb
(30 chromosomes) psmc-rs took a median 318 s on 1 thread, 22.5 s on 16
and 11.5 s on 30, against 4,425 s for C psmc: 13.9, 197 and 384 times as
fast (13.9 to 14.2 times on 1 thread at every size). It peaked at 36 MB
of resident memory on 1 thread and 149 MB on 30, C psmc at 1,063 MB. Every
size and thread count, the ranges over the repeats, the thread scaling
and the outputs' determinism are in
[Supplementary Note 1, section 10](experiment/Supplementary_Note_1_Reproducibility.md#10-performance-e3).

Use a release build, and `-q --no-progress` for batch runs. The model has
psmc's PSMC_T_INF = 1000 time grid and at most 10,000 states; models of more
than 256 states use dense kernels, which cache forward rows within a
process-wide budget of `PSMC_ALPHA_CACHE_MB` (2048; 0 disables the cache).

On x86-64 the E-step, the log-likelihood and TMRCA decoding run code
compiled with AVX2 and FMA on CPUs that have them (Intel Core since
Haswell, AMD since Excavator and Zen), picked at run time, so the
release binaries need no `-C target-cpu` flags for full speed: they
fitted a 1.9 Gb genome (5 EM iterations, 1 thread) in 71.7 s on an AMD
EPYC 7702 and in 47.1 s on an EPYC 9655, as fast as a
`-C target-cpu=x86-64-v3` build, against 161 s and 97.0 s for the
portable code (all four timed while the E-step still summed the expected
transitions in O(states²), the portable ones in the release build that
preceded the AVX2 and FMA code). The AVX2 and FMA code fuses the
multiply-adds of the O(states) forward and backward products (Harris et
al. 2014's linear-time recursion) and of the transition sums, as the NEON
code of aarch64 does;
the other x86-64 kernels never fuse, and no sum changes its order with
the vector width, so its results are identical to the bit to an
`x86-64-v3` build's, and on every CPU with AVX2 and FMA that gets the
same `log` and `exp` from the system's math library (glibc picks a
variant of them by CPU). x86-64 CPUs without AVX2 or FMA, and virtual
machines and emulators that hide them, run the portable code, which does
not fuse those multiply-adds (each would be a call to libm's `fma`,
several times slower), so its results differ in the last digits: by a
relative 1e-14 or so in an E-step (1.4e-14 at most on the test inputs),
which the M-step, stopping on a flat optimum, turns into about 1e-8 to
1e-6 in the fitted parameters (on an EPYC 7702, 7.8e-9 on a 20,000-bin
test input after 3 EM iterations, 8.6e-7 on the 180,000-bin one after 5
with the pattern `4+25*2+4+6`). `PSMC_RS_SIMD=portable` forces the
portable code, to compare speeds or results; the output JSON's `simd`
and the report say which ran (`avx2+fma`, `portable`, or `neon` on Apple
silicon and other aarch64).

The M-step needs little memory and time, however many λ are free. It
differentiates its objective in reverse mode, one row of the transition
matrix at a time, from the sums of the expected transition counts: an
evaluation holds O(states) values and costs O(states) time, 7.5 µs with
64 states on one thread, about what the E-step spends on 25 bins. An
M-step evaluates its objective once per free parameter for the
curvatures that scale its search, once or twice per L-BFGS iteration, and
once per free parameter again for the check before it stops, so its time
grows with the number of free parameters times states. With pattern
`4+25*2+4+6` (31 parameters) on one thread of the Apple M1 Pro, an M-step
took 10 to 12 iterations, 74 to 113 evaluations and 0.55 to 0.87 ms on a
simulated 100 Mb chromosome (E3's), whose E-steps took 0.32 s there
(0.2%; three runs of 25 EM iterations). With one λ per interval
(`--n-steps` alone) on the 180,000-bin test input, M-steps over 104
parameters took 2.8 to 3.4 ms and over 304 took 22 to 44 ms, against
E-steps of 0.09 s and 4.1 s. One
EM iteration there on one thread peaked at 5.6, 8.6, 311 and 417 MB of
resident memory with 101, 201, 301 and 401 intervals and took 0.12,
0.25, 6.2 and 11.0 s (medians of three runs). Above 256 intervals the
memory is the E-step's cache of forward rows (`PSMC_ALPHA_CACHE_MB`):
without it, the iteration with 401 intervals peaked at 14 MB.

## Testing and C equivalence

```bash
cargo test                        # unit, integration and doc tests
python3 -m unittest discover -s experiment/tests
```

The suite compares psmc-rs with the original tools' outputs, stored under
`tests/data/` with the scripts that made them:

- `tests/c_equivalence.rs`: EM on a 180,000-bin psmcfa against C psmc
  0.6.5-r74 (patterns `4+5*3+4` and `4+25*2+4+6`): round 1 equal to C's
  printed precision, rounds 2 to 5 within a relative 2e-7.
- `tests/psmc_out_c_reference.rs`, `tests/em_c_reference.rs` and
  `tests/psmc_out_plot.rs`: the `.psmc` output line by line against C's
  (but `QD`), C's expectations (`post_sigma`, `RI`) and psmc_plot.pl's
  reading of the file.
- `tests/io_text_psmcfa_c.rs`: psmcfa parsing against C psmc's counts on the
  turtle and Papuan psmcfa files in `experiment/real_data`.
- `tests/io_route_a.rs`: consensus mode against goldens made with
  `bcftools call -c | vcfutils.pl vcf2fq | fq2psmcfa`, and multihetsep output
  against `generate_multihetsep.py`, byte for byte.
- `tests/bootstrap_split.rs` and `tests/bootstrap_resample.rs`: trunks
  against `splitfa`, draws against `psmc_resamp`.

Beyond the tests: the manuscript's experiments fit 40 simulated 500 Mb
genomes of four models (E1) and four real genomes (E2) with both tools as
`psmc -N25 -t15 -r5 -p "4+25*2+4+6"` and compare the curves interval by
interval over the informative window, the 1% to 99% quantiles of the
coalescence time (`experiment/pipeline/README.md`; `E1_rs_vs_c.csv`,
`E2_rs_vs_c.csv`). The RMSE of log10 Ne between psmc-rs and C psmc was
3.7e-6 to 2.0e-4 over E1's genomes (median 2.5e-5) and 9.7e-6 to 2.8e-4
over the real ones, and no interval in the window differed by more than
6.6e-4 (E1) and 4.5e-4 (E2). θ, ρ and t_max agreed within a relative
5.5e-4 in E1 and 1.0e-3 in E2, and the final log-likelihood within
3.5e-8 and 2.3e-7. On one simulated 4 × 25 Mb bottleneck genome (msprime,
Hudson, μ = 2.5e-8, r = 1e-8), the curves are 7.7e-5 apart (at most
2.6e-4), with θ, ρ and t_max within 1.4e-4 and the log-likelihood within
4e-9 (relative); on this genome the sources of the difference were
traced. The M-steps are not what keeps them apart. psmc-rs's default fit
is within 5e-7 of one whose M-steps run to the precision limit, and a port
of C's Hooke-Jeeves search in place of its L-BFGS gives a curve 7.5e-5
from C's. Nearly all the rest comes from C's expected counts, which leave
out the last bin of each sequence (`em::posterior_sigma`): with that bin
left out of psmc-rs's counts too, its curve is within 6.4e-6 of C's (the
Hooke-Jeeves port within 8.0e-6), and on the reference data rounds 2 to 5
agree with C's log-likelihoods to a relative 2.8e-9 instead of 7.3e-8.
Before the window starts the curves can differ more: the first λ, which
the data barely determine, ran to the M-step's upper bound of 1e4 in
psmc-rs (the run says so) and to 1.5e5 in C psmc, whose search has no
bound; Ne in the first four intervals (up to about 8,600 years) is then
1.2 log10 units apart, and neither is an estimate. In E1 this happened in
one of the 40 genomes (an expansion replicate, C psmc's first λ at
1.08e6), 2.03 log10 units apart.

Two ignored tests run the tools themselves:
`VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa cargo test
--release --test io_consensus_tools -- --ignored` and `FQ2PSMCFA=...
cargo test --release --test io_vcf_exact -- --ignored`. CONTRIBUTING.md
describes how reference outputs may be regenerated.

## Reproducing the manuscript

`experiment/pipeline/` makes the manuscript's experiments, tables and
figures with psmc-rs 0.2 (simulate, run psmc-rs and C psmc with matched
settings, record how every output was made, draw the figures from the
tables); [experiment/pipeline/README.md](experiment/pipeline/README.md)
describes it, and
[Supplementary Note 1](experiment/Supplementary_Note_1_Reproducibility.md)
how to reproduce the results on a laptop or a SLURM cluster. Its results
go to `experiment/runs/E1` to `E7` and `FIG`;
[experiment/README.md](experiment/README.md) gives an overview.

## Citing

If you use psmc-rs, please cite it as given in [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository"), together with the PSMC paper:

> Li H, Durbin R (2011). Inference of human population history from
> individual whole-genome sequences. *Nature* 475:493-496.
> doi:10.1038/nature10231

Its O(states) forward and backward steps, and its sums of the expected
transition counts, use the rank-one triangles of the transition matrix, as
in:

> Harris K, Sheehan S, Kamm JA, Song YS (2014). Decoding coalescent hidden
> Markov models in linear time. In *Research in Computational Molecular
> Biology (RECOMB 2014)*, LNCS 8394:100-114.

## Contributing and license

Bug reports and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
psmc-rs is released under the MIT license ([LICENSE](LICENSE)). The C psmc
outputs under `tests/data/c_reference/` come with C psmc's license
(`PSMC_LICENSE.txt` there).

## Option reference

A condensed table of the options (0.2.0).

**Input** (inference, `convert`, `qc`)

| option | default | meaning |
| --- | --- | --- |
| `--input-format auto\|psmcfa\|mhs\|vcf` | `auto` | input format; auto recognises it from the content |
| `--bin-size BP` | 100 | bases per bin (fq2psmcfa `-s`): bins VCF and multihetsep input; for psmcfa, give the `-s` it was made with |
| `--batch-size BINS` | none | split sequences into HMM rows of at most BINS bins; results do not change |
| `--mhs-pair I,J` | `0,1` | multihetsep: the two haplotypes, 0-based positions in the alleles column |
| `--vcf-sample NAME` | | VCF mask mode: the sample to read (required when there are several); consensus mode needs a one-sample VCF |
| `--vcf-mode auto\|consensus\|mask` | `auto` | VCF: consensus (`vcf2fq \| fq2psmcfa`) or mask mode |
| `--vcf-mask BED` | | VCF: callable sites (repeatable; a site must be in every one) |
| `--vcf-negative-mask BED` | | VCF: sites to exclude (repeatable) |
| `--vcf-keep-filtered` | off | VCF mask mode: use records whose FILTER is not PASS or `.` |
| `--vcf-no-contig-end` | off | VCF mask mode: end sequences at their last record, not the `##contig` length |
| `--depth auto\|MIN:MAX` | `auto` | VCF consensus mode: INFO/DP bounds (vcf2fq `-d`, `-D`); auto = a third and twice the mean |
| `--min-mapq MQ` | 10 | VCF consensus mode: minimum INFO/MQ (vcf2fq `-Q`) |
| `--indel-window BP` | 5 | VCF consensus mode: bases masked on each side of an indel (vcf2fq `-l`) |
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
| `--theta0 X` | −ln(1 − K/L); with the counts, its callable-length analogue | initial θ0 per bin |
| `--rho0 X` | θ0 / `--tr-ratio` | initial ρ0 per bin |
| `--init-params FILE` | | start from a saved fit (C `-i`) |
| `--mu MU` | 2.5e-8 | mutation rate per bp per generation, for scaling |
| `--bin-callable auto\|ignore\|approximate` | `auto` | VCF: fit each called bin by its callable bases (`auto`), or treat every called bin as BIN_SIZE bases (`ignore`, C psmc); `approximate` also fits multihetsep's approximate counts |
| `--mstep-iters N` | 1000 | cap on the L-BFGS iterations of an M-step, which runs to convergence |
| `--smooth-lambda W` | 0 | penalty on log-λ differences of neighbouring intervals |
| `--bootstrap N` | 0 | bootstrap replicates |
| `--bootstrap-block-size BINS` | 500000 | trunk size (splitfa); alias `--bootstrap-trunk-size` |
| `--bootstrap-seed SEED` | 42 | seed of the resampling |
| `--bootstrap-iters N` | N_ITER | EM iterations per replicate |
| `--bootstrap-dir DIR` | `OUTPUT` with `.bootstrap` in place of `.json` | replicate and summary files; an earlier bootstrap's `replicate_NNN.*` and `summary.*` there are removed first |
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
