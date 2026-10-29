# Inputs

psmc-rs reads three formats: psmcfa, VCF (including gVCF) and MSMC's
multihetsep, each plain, gzip or bgzip. A fit, `psmc-rs qc` and `psmc-rs
convert` read a file the same way and share the input options. This page
says how each format is read; [preprocessing.md](preprocessing.md) says how
to make one from aligned reads.

- [Format and compression](#format-and-compression)
- [Bins](#bins)
- [psmcfa](#psmcfa)
- [VCF](#vcf): [which mode](#which-mode), [consensus mode](#consensus-mode),
  [mask mode](#mask-mode), [masks](#masks)
- [multihetsep](#multihetsep)
- [Partly callable bins](#partly-callable-bins)
- [The sequence filter](#the-sequence-filter)
- [Writing inputs: `psmc-rs convert`](#writing-inputs-psmc-rs-convert)

## Format and compression

`--input-format auto` (the default) recognises the format from the
content, line by line:

1. a `##` or `#CHROM` line makes it a VCF;
2. other `#` lines and blank lines are skipped;
3. a line that starts with `>` or `@` makes it psmcfa;
4. a line of at least 8 tab-separated fields makes it a VCF without a
   header;
5. anything else makes it multihetsep. If such a file fails to read as
   multihetsep from the start, the error says how its format was
   recognised.

`--input-format psmcfa|vcf|mhs` sets the format instead.

- **Compression** is recognised from the file's first bytes, whatever its
  name. Every member of a bgzip file is read, and decompression runs in a
  thread of its own, beside the parsing. A bgzip file must end with BGZF's
  end-of-file block, as htslib checks: without it, the file was probably
  cut short by an interrupted job, and it is refused.
- **Pipes.** INPUT may be a pipe or a FIFO, read once; only
  `--depth auto` needs a regular file ([consensus mode](#consensus-mode)).
- **BCF** is refused; `bcftools view -Oz` converts it to a VCF. A UTF-8
  byte order mark is skipped.
- **A directory** can be the INPUT of a fit (not of `qc` or `convert`). It
  stands for its first file, by name, that ends in `.psmcfa`, `.vcf`,
  `.mhs` or `.multihetsep`, each optionally followed by `.gz`; with
  `--input-format`, only that format's extensions count.
- **Options that do not apply** to the input as read (for example
  `--vcf-mask` with psmcfa input) are named in a warning and ignored.

## Bins

PSMC sees a genome as bins of `--bin-size` bases (default 100, as
`fq2psmcfa -s`), each homozygous (`T`), heterozygous (`K`) or missing
(`N`). psmc-rs bins VCF and multihetsep input itself, with fq2psmcfa's
rule: a bin is missing when more than 90% of its sites are, and otherwise
heterozygous when one of its sites is. A psmcfa file is binned already.

`--bin-size` also scales time and Ne for every format (N0 = θ / (4 μ s)
for bins of s bases), so give a psmcfa file the `-s` it was made with.

## psmcfa

psmc-rs parses psmcfa as C psmc does:

- A record starts at `>` (or `@`), even in the middle of a line. Its header
  line is never sequence data, and the name runs to the first whitespace.
  Text before the first header is ignored (with `--input-format psmcfa`;
  `auto` recognises psmcfa only when its first line is a header).
- In sequence lines, `T`, `A`, `C`, `G`, `0` and their lowercase forms are
  called bins; `K`, `M`, `R`, `S`, `W`, `Y`, `1` and their lowercase forms
  are heterozygous bins; `+` starts a FASTQ quality block; any other
  printable byte (such as `N`) is a missing bin. Whitespace, CR, control
  and non-ASCII bytes are skipped.

Differences from C psmc:

- a file without any `>` record is an error;
- a truncated FASTQ quality block is an error (C psmc silently stops
  reading);
- empty records are left out of the fit, though they still count in the
  `n_seqs` of the `.psmc` header, as in C.

psmcfa input is not filtered further ([the sequence
filter](#the-sequence-filter) applies to VCF and multihetsep). When a
record has a FASTQ quality block, or at least 1% of the bins are letters
that fq2psmcfa never writes (`A`, `C`, `G`, other IUPAC codes, lowercase),
the run warns that the input looks like a per-base consensus, such as
`vcfutils.pl vcf2fq` output, which would be read one base per bin.

## VCF

psmc-rs reads one diploid sample of a VCF, in one of two modes:

| | consensus mode | mask mode |
| --- | --- | --- |
| for | the all-sites VCF of `bcftools call -c` | any other VCF: GATK, DeepVariant, `bcftools call -m`, gVCFs, simulations |
| reads | the INFO fields that `vcfutils.pl vcf2fq` reads, as `vcf2fq \| fq2psmcfa` would | the genotypes of one sample |
| missing | positions without a record, and bases that the depth, quality and indel filters remove | positions outside the callable mask, filtered records, missing genotypes |
| sample | a one-sample VCF (the INFO fields are computed over all samples) | `--vcf-sample NAME` when the VCF has several |
| route in [preprocessing.md](preprocessing.md) | route A | route B |

In consensus mode, `--vcf-sample` only checks the name of the one sample
(the run warns that it selected nothing).

### Which mode

`--vcf-mode auto` (the default) looks at the first 10,000 records (at most
64 MB of them):

- if any of them is a gVCF block, the file is a gVCF and is read in mask
  mode;
- otherwise, if at least half have ALT `.`, the file is an all-sites VCF:
  it is read in consensus mode when at least 90% of those carry INFO/FQ (as
  `bcftools call -c` writes), and is an error otherwise;
- otherwise (fewer than half with ALT `.`), a file whose records cover at
  least half of the positions they span and at least 90% of which carry FQ
  (`bcftools call -c -A` at high depth) is read in consensus mode too;
- any other VCF is read in mask mode.

`--vcf-mode consensus|mask` sets the mode (a gVCF is still read as one in
forced mask mode). The run says which mode it used and why.

### Consensus mode

Consensus mode reproduces the classic PSMC input,
`vcfutils.pl vcf2fq | fq2psmcfa`, in one step, with bins identical to
fq2psmcfa's ([validation.md](validation.md#the-classic-tools)). The two
tools' options map to psmc-rs as follows:

| psmc-rs | classic tool | default | what it does |
| --- | --- | --- | --- |
| `--depth MIN:MAX` | `vcf2fq -d MIN -D MAX` | `auto`: a third and twice the mean depth (vcf2fq: 3:100000) | bases with INFO/DP outside the bounds are missing |
| `--min-mapq` | `vcf2fq -Q` | 10 | bases with INFO/MQ (root-mean-square mapping quality) below it are missing |
| `--indel-window` | `vcf2fq -l` | 5 | bases this close to an indel call are missing |
| `--min-qual` | `fq2psmcfa -q` | 10 | bases with a consensus quality (the size of INFO/FQ) below it are missing |
| `--bin-size` | `fq2psmcfa -s` | 100 | bases per bin |
| `--min-good-bases` | `fq2psmcfa -g` | 10000 | sequences with fewer called bases are dropped |
| `--min-good-fraction` | fixed in fq2psmcfa | 0.2 | sequences with a smaller called share are dropped |

- **`--depth auto`** follows the psmc README's advice and differs from
  vcf2fq's own default. A first pass over the file computes the mean
  INFO/DP of the records that vcf2fq turns into bases (one-base REF, no
  `INDEL` flag, a base or `.` as ALT); the bounds are
  `max(1, round(mean / 3))` and `round(2 × mean)`. Reading the file twice
  needs a regular file: from a pipe, give `--depth MIN:MAX`. The run prints
  the bounds it used, and the QC report gives the mean depth.
- **Masks** (`--vcf-mask`, `--vcf-negative-mask`) can be added in this mode
  too, for example a mappability mask: the bases outside the callable set
  become missing. The bins are then no longer exactly what vcf2fq and
  fq2psmcfa give.
- **Malformed input.** Where vcf2fq and fq2psmcfa would misread malformed
  input, psmc-rs reports an error instead. The module documentation of
  `psmc_rs::io::consensus` lists these cases; `bcftools call -c` output
  has none of them.
- **A chromosome that comes back** after another starts a second sequence,
  as vcf2fq does, with a warning.

fq2psmcfa's `-x` (which masks the human pseudoautosomal regions, in NCBI36
coordinates) and its mutation-class filters (`-v`, `-n`, `-c`, `-C`) have
no psmc-rs equivalent.

### Mask mode

Mask mode reads the genotypes of the sample. Each position is a called
homozygous site, a heterozygous SNP, or missing. Per record:

- FILTER other than `PASS` or `.`: the position is missing
  (`--vcf-keep-filtered` uses the record anyway);
- a missing or partial genotype (`./.`, `./1`): the position is missing;
- heterozygous, with every allele one of `A`, `C`, `G`, `T`: a heterozygous
  SNP;
- heterozygous, with another allele (an indel, `*`, a symbolic allele): its
  whole REF span is missing;
- homozygous (hom-alt and homozygous indels included): called.

Several records at one position are combined, the most missing class
winning: missing over heterozygous over homozygous.

- **A position without any record** is called homozygous unless a mask says
  otherwise. In a gVCF, a position that no record or block covers is
  missing instead.
- **gVCF blocks.** A record whose ALT alleles are only `.`, `<*>`,
  `<NON_REF>` or `<X>` and that has an INFO/END stands for every position
  from POS to END. A no-call or filtered block is missing over its whole
  span; a reference block's positions are called homozygous unless a mask
  says otherwise.
- **Sequence ends.** Each sequence runs to its `##contig` length when the
  header gives one; `--vcf-no-contig-end` ends it at its last record (or at
  the END of a last block). A chromosome without any record gives no
  sequence.
- **Order.** The VCF must be sorted (`bcftools sort`): a position that goes
  backwards and a chromosome that comes back after another are errors, and
  so is a record beyond its `##contig` length, unless
  `--vcf-no-contig-end` is given.

### Masks

`--vcf-mask BED` gives the callable sites and `--vcf-negative-mask BED`
sites to exclude. Both can be repeated: a site is callable when every
`--vcf-mask` contains it and no `--vcf-negative-mask` does.

- **One file per kind.** Because every `--vcf-mask` must contain a site,
  masks that come one per chromosome must be joined into one file first
  ([preprocessing.md](preprocessing.md#the-callable-mask) shows how). The
  run warns about a chromosome that only some masks cover.
- **Rows.** A row of three or more columns is BED (`chrom start end`,
  0-based, end-exclusive); a row of exactly two columns is `start end`,
  1-based inclusive, for every chromosome. Rows may overlap and need not be
  sorted; they are merged as they are read, so memory follows the merged
  intervals, not the rows.

**A variants-only VCF needs a mask.** Without one, every position the
caller could not call counts as homozygous: the missing data disappear and
θ and Ne come out too low, while the false heterozygous calls of repeats,
which a mask would remove, push them up. QC reports `VARIANTS_ONLY_NO_MASK`.

**A gVCF needs a mask too.** Positions that no record or block covers are
missing whatever the masks say (bcftools writes nothing where there are no
reads), but a reference block counts as called over its span whatever its
depth and genotype quality (GATK writes stretches without reads as `0/0`
blocks), and the run warns about it. Make the callable mask from the
gVCF's blocks and give it with the gVCF or with its variant records
([preprocessing.md](preprocessing.md#the-callable-mask)).

## multihetsep

MSMC's format, as `generate_multihetsep.py` writes it: one row per
segregating site, `chrom pos nr_called alleles`, where `nr_called` counts
the callable sites since the previous row, this one included.

- `--mhs-pair I,J` (default `0,1`, the first diploid sample) selects two
  haplotypes, by their 0-based positions in the alleles column.
- A site is heterozygous when the two differ in every comma-separated
  ordering, called when they are equal in every ordering, and missing when
  the orderings disagree (a pair across two unphased samples) or an allele
  is `?` or `N`.
- Every row must list as many haplotypes as the first, so remove indels
  before `generate_multihetsep.py`. The rows of each chromosome must form
  one block in position order: a chromosome that comes back after another
  is an error. A sequence ends at its last row.

**Missing data are placed approximately.** multihetsep says how many sites
between two rows are uncalled, not which. psmc-rs places them at the start
of each interval. Compared with fq2psmcfa on the same calls, uncalled sites
scattered among called ones then become runs of `N` bins after
heterozygous sites, and the bin of a heterozygous site is partly callable
or lost. With missing data this biases θ, the times and Ne low
([validation.md](validation.md#missing-data)), and QC warns
(`APPROX_MISSING_PLACEMENT`) when more than 5% of the sites are uncalled.
For exact bins, fit the VCF with its callable mask.

The callable bases of each bin follow the same placement: their sum over
each interval is exact, how it is split among the bins is not. A fit
therefore treats every called bin of multihetsep input as `--bin-size`
bases, as C psmc treats its bins, unless `--bin-callable approximate` asks
for the approximate counts, which made the fit worse in the validation.

## Partly callable bins

fq2psmcfa calls a bin as soon as a tenth of its bases are callable, and C
psmc treats every called bin as a full bin. A bin of `c` callable bases out
of `s` is less likely to show a heterozygous site. So when missing data are
scattered in runs shorter than a bin (per-site depth or quality filters, a
mask of short gaps), almost every bin is still called but holds fewer
bases, and θ per bin, N0, every time and every Ne come out low by about 1.1
to 1.3 times the missing share of the called bins. Missing data in long runs
mostly make whole bins `N`, which C psmc's model handles.

A VCF (either mode) says which positions of each bin are callable. By
default (`--bin-callable auto`) psmc-rs fits each called bin of a VCF by
its count: a bin of `c` callable bases emits as `c` bases of sequence,

```text
P(T | k, c) = exp(-θ (c / s) avg_t[k]),   P(K | k, c) = 1 - P(T | k, c),   P(N | k, c) = 1
```

with θ still per `s` bases, so N0 = θ / (4 μ s) and `psmc_plot.pl -s` keep
their meaning.

- The E-step, the M-step, the starting θ0, the bootstrap and the TMRCA
  decoding all use the counts. The starting θ0 solves
  `sum over called bins of (1 - exp(-θ0 c / s)) = K`, the moment estimate
  that C psmc's −ln(1 − K/L) is for full bins.
- With every called bin full (complete data), the model, the fit and every
  output are C psmc's, to the bit.
- `--bin-callable ignore` treats every called bin as `s` bases, as
  fq2psmcfa and C psmc do: a VCF fitted with it gives exactly the fit of
  fq2psmcfa's psmcfa of the same calls.
- Counts are kept for bins of at most 65,535 bases; larger bins are fitted
  as C psmc fits them, and the run says so.
- The run says what the counts hold and which model it fits; the JSON's
  `bin_callable` records it ([outputs.md](outputs.md#json)).

How much bias the counts remove is in
[validation.md](validation.md#missing-data), and the design in
[design/bin-callable-model.md](design/bin-callable-model.md).

**A psmcfa file cannot hold the counts.** It records only each bin's
symbol, so a psmcfa is fitted as C psmc fits it, bias included, whatever
`--bin-callable`. `psmc-rs convert --psmcfa` warns when it writes partly
callable bins. To see how much of an existing psmcfa's called bins is
callable, run `psmc-rs qc` on the VCF it was made from.

## The sequence filter

VCF and multihetsep sequences go through fq2psmcfa's output filter: a
sequence is dropped when it has fewer than `--min-good-bases` (10,000)
called bases, or when they are less than `--min-good-fraction` (0.2) of its
length; 0 disables either. The run lists what it dropped. If nothing is
left, a fit stops with an error that says how to keep every sequence
(`--min-good-bases 0 --min-good-fraction 0`); `qc` reports the dropped
sequences; `convert` warns and writes empty files, as fq2psmcfa does (none
for `--multihetsep-per-chrom`).

## Writing inputs: `psmc-rs convert`

```bash
psmc-rs convert calls.vcf.gz --psmcfa sample.psmcfa.gz
psmc-rs convert calls.vcf.gz --multihetsep-per-chrom sample   # sample.<chrom>.multihetsep
psmc-rs convert sample.psmcfa --psmcfa normalised.psmcfa
```

`convert` writes the input, as read, in other formats. At least one of
`--psmcfa FILE`, `--multihetsep FILE` and `--multihetsep-per-chrom PREFIX`
is required; a name ending in `.gz` is written compressed (`PREFIX.gz`
gives `PREFIX.<name>.multihetsep.gz`).

- **psmcfa** is laid out as fq2psmcfa writes it. It holds only each bin's
  symbol, so when some called bins of a VCF are partly callable, `convert`
  warns with their share and the bias that causes, and advises fitting the
  VCF itself. Written from multihetsep, it places the uncalled sites as
  described [above](#multihetsep).
- **multihetsep** is written for VCF and multihetsep input as
  `generate_multihetsep.py` would write it for the selected sample
  (heterozygous sites only; the differences are listed in the rustdoc of
  `psmc_rs::io::write::write_multihetsep_to`). It keeps `nr_called`, from
  which the reader rebuilds approximate counts, which a fit uses only with
  `--bin-callable approximate`. psmcfa input has no sites to write, and
  asking for multihetsep from it is a usage error.
- `--qc-json FILE` writes the QC report as JSON.
- The per-sequence files are named after the sequences, so they are
  checked against the input and the other outputs once the input is read,
  before anything is written.
