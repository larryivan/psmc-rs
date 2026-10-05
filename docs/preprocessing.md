# From aligned reads to psmc-rs input

PSMC (Li & Durbin 2011) fits a population size history to one diploid
genome: the two copies of each autosome, seen through which bins of the
genome hold a heterozygous site. psmc-rs reads that genome as a psmcfa file
(bins of `T`, homozygous; `K`, heterozygous; `N`, missing), as a VCF, or as
an MSMC multihetsep file. This guide goes from the aligned reads of one
sample to those inputs, by two routes:

- **Route A**, the classic PSMC consensus: call every covered site with
  `bcftools mpileup | bcftools call -c` and let psmc-rs bin it exactly as
  `vcfutils.pl vcf2fq | fq2psmcfa` would. Use it for one diploid genome when
  you start from reads. `scripts/bam_to_psmc.sh` runs it from end to end.
- **Route B**, variant calls plus a callable mask: the layout MSMC2 and
  SMC++ use. Use it when variant calls from another pipeline (GATK,
  DeepVariant, bcftools `call -m`) already exist, or when the same files
  should also feed MSMC2.

Both routes end with `psmc-rs qc`, which reports on the input before
anything is fitted, and the pitfalls at the end of this guide apply to both.

## Before you start: the alignments

Alignment itself is out of scope. Follow the standard short-read practice
for your organism: map the reads to the reference (for example with BWA-MEM
or minimap2), sort by coordinate, mark duplicates (samtools markdup or
Picard MarkDuplicates) and index the result (`samtools index`). What the
steps below assume:

- **One sample.** All reads come from one individual. bcftools treats each
  read-group `SM` tag as a sample; `scripts/bam_to_psmc.sh` stops when it
  finds more than one.
- **Duplicates marked.** `bcftools mpileup` skips reads flagged as
  duplicates, unmapped, secondary or failing QC by default (its
  `--skip-any-set`), so marking is enough; nothing needs to be removed.
- **The reference the reads were aligned to**, with its `.fai` index (made
  by `samtools faidx`, or by the script when missing). CRAM files need it
  to be decoded at all.
- **An index of the alignments** (`.bai`, `.csi` or `.crai`), so that each
  chromosome can be called on its own, in parallel. Without one, every
  per-chromosome job reads the whole file.
- **Enough coverage**: see [low coverage](#low-coverage) below; about 18x
  or more.

## Route A: the classic consensus

### One command

```sh
scripts/bam_to_psmc.sh --bam sample.bam --ref ref.fa --out results/sample \
  --threads 8 --run 25 -- --pattern '4+25*2+4+6'
```

This calls every chromosome that is not named like a sex chromosome or an
organelle (the log lists what it leaves out), 8 at a time, and writes:

| File | What it is |
| --- | --- |
| `results/sample.vcf.gz`, `.tbi` | The all-sites VCF of `bcftools call -c`, bgzipped and indexed |
| `results/sample.qc.txt`, `.qc.json` | `psmc-rs qc` on that VCF |
| `results/sample.psmcfa` | The input of psmc-rs and C psmc |
| `results/sample.<chrom>.multihetsep` | With `--multihetsep`: one file per chromosome, for MSMC2 |
| `results/sample.json`, `.html`, `.psmc` | With `--run N_ITER`: the fit, its report and C psmc's output format |
| `results/sample.log` | Every command run, with the tools' messages |

`scripts/bam_to_psmc.sh --help` lists the options. The defaults are
discussed below: `--min-mapq 20 --min-baseq 20` for mpileup, `--depth auto`,
`--min-qual 20` and `--bin-size 100`. `--chroms`
takes a comma-separated list or a file of names (the first column, so a
`.fai` or a BED file works, but only the names count: each chromosome is
called whole); `--dry-run` prints the chromosomes and commands
without running anything. Options after `--` go to the psmc-rs fit. The
script needs bcftools 1.10 or later and psmc-rs 0.2.0 or later, and runs
with the bash 3.2 that macOS ships as well as with current bash. The files
it writes itself (VCF, QC reports, psmcfa, multihetsep) are written in a
working directory next to them and renamed when complete, and a rerun with
the same inputs writes the same bytes to every output but the log, whatever
`--threads` is.

### What it runs

For each chromosome, in parallel:

```sh
bcftools mpileup -C50 -q20 -Q20 -d 10000 -f ref.fa -r chr1 sample.bam \
  | bcftools call -c -Ob -o chr1.bcf
```

then:

```sh
bcftools concat -Oz -o sample.vcf.gz chr1.bcf chr2.bcf ...   # in reference order
bcftools index -t sample.vcf.gz
psmc-rs qc sample.vcf.gz --depth auto --min-qual 20 --json sample.qc.json > sample.qc.txt
psmc-rs convert sample.vcf.gz --depth auto --min-qual 20 --psmcfa sample.psmcfa
psmc-rs sample.psmcfa sample.json 25 --psmc-out sample.psmc   # with --run 25
```

(The script also passes `--input-format vcf --vcf-mode consensus`, which
psmc-rs would otherwise detect, and `--no-version` to bcftools, so that the
VCF header does not change from run to run; the log has the exact commands.)
psmc-rs can also fit the VCF directly, `psmc-rs sample.vcf.gz fit.json 25
--min-qual 20`, which bins it the same way. Note `--min-qual 20`: psmc-rs's
default is fq2psmcfa's own default, 10.

- **`-C50`** caps the mapping quality of reads with many mismatches; the
  psmc README's command line uses it (`samtools mpileup -C50`, from before
  mpileup moved to bcftools).
- **`-q20 -Q20`** drop reads of mapping quality below 20 (a 1% chance or
  more, by the aligner's estimate, that the read belongs elsewhere) and
  bases of base quality below 20. The psmc README keeps mpileup's defaults
  here; 20 and 20 are the values of msmc-tools' `bamCaller.py` example.
- **`-d 10000`** lets mpileup read up to 10,000 reads per position rather
  than 250, so that the upper depth bound still works on deep data (see
  [depth bounds](#depth-bounds)).
- **`bcftools call -c`** is the consensus caller. It writes a record for
  every covered position, variant or not, with the `FQ`, `AF1`, `MQ` and
  `DP` fields that `vcfutils.pl vcf2fq` reads. Positions without a record
  (no reads) become missing data.

`vcf2fq` then turns each record into one base of a consensus sequence and
`fq2psmcfa` bins it. psmc-rs does both in one pass, with the same options:

| psmc-rs | classic tool | default | what it does |
| --- | --- | --- | --- |
| `--depth MIN:MAX` | `vcf2fq -d MIN -D MAX` | auto (vcf2fq: 3:100000) | bases with INFO/DP outside the bounds are missing |
| `--min-mapq` | `vcf2fq -Q` | 10 | bases with INFO/MQ (root-mean-square mapping quality) below it are missing |
| `--indel-window` | `vcf2fq -l` | 5 | bases this close to an indel call are missing |
| `--min-qual` | `fq2psmcfa -q` | 10 (the script: 20) | bases with a consensus quality (the size of INFO/FQ) below it are missing |
| `--bin-size` | `fq2psmcfa -s` | 100 | bases per bin; a bin is `N` when more than 90% of its bases are missing, `K` when one is heterozygous |
| `--min-good-bases` | `fq2psmcfa -g` | 10000 | sequences with fewer called bases are dropped |
| `--min-good-fraction` | fixed in fq2psmcfa | 0.2 | sequences with a smaller called share are dropped |

### Depth bounds

Collapsed repeats and duplications pile up reads, and where few reads cover
a heterozygous site, one allele is often not seen. Bounds on the depth of
each site remove both:

- The psmc README recommends a minimum of **a third** and a maximum of
  **twice** the average depth (`vcf2fq -d` and `-D`).
- Li & Durbin (2011, Methods) masked sites with depth below **half** or above
  twice the average (estimated at HapMap3 loci), together with a
  root-mean-square mapping quality below 25, 10 bp around short indels,
  consensus quality below 20 for Illumina data, and a 35-mer mappability
  filter.
- `--depth auto` (the default of psmc-rs and of the script) follows the
  README: a first pass over the VCF computes the mean INFO/DP of the records
  vcf2fq turns into bases (one-base REF, no `INDEL` flag, a base or `.` as
  ALT), and the bounds are `max(1, round(mean / 3))` and `round(2 * mean)`.
  Reading the file twice needs a regular file; from a pipe, give
  `--depth MIN:MAX`. psmc-rs prints the mean and the bounds it used
  (`consensus: depth mean 25.3, median 25; bases kept at depth 8-51
  (auto)`).

The mean covers every chromosome in the VCF, so call only the chromosomes
you analyse: a male X at half depth, or an organelle at hundreds of times
the nuclear depth, moves it. (The script leaves both out by default.)
`bcftools mpileup` reads at most 250 reads per position and file by default
(`-d`, `--max-depth`), and the INFO/DP of deeper sites then stays near that
cap. Once the upper bound comes close to it, collapsed repeats pass the
bound and their paralog differences become heterozygous calls: in the
end-to-end test's 140x simulation (bound 278), mpileup's default turned 20
of the 25 bins of a collapsed repeat into `K` bins. Keep the cap at least
twice the upper bound. The script runs mpileup with `-d 10000` (or twice
MAX of `--depth MIN:MAX`, if that is more), takes `--max-depth N` to change
it, and stops when the automatic bound needs more; by hand, give
`bcftools mpileup -d` yourself whenever twice the bound exceeds 250, that is
from a mean depth of about 60x.

### The same bytes as vcf2fq and fq2psmcfa

For the same VCF and the same thresholds, the psmcfa that psmc-rs writes
(`psmc-rs convert --psmcfa`, and so the script's) is byte for byte the
output of

```sh
gzip -dc sample.vcf.gz | vcfutils.pl vcf2fq -d MIN -D MAX | fq2psmcfa -q20 - > sample.psmcfa
```

and inference on the VCF fits exactly these bins. This is tested three ways:

1. `tests/io_route_a.rs` compares psmc-rs with golden files that the real
   tools made from simulated reads (`tests/data/route_a`): six option sets,
   including one bin per base (`-s 1 -g 0`), which compares every base.
2. `tests/io_consensus_tools.rs` runs the real tools on hundreds of random
   VCFs full of edge cases (`--ignored`; it needs `VCFUTILS` and
   `FQ2PSMCFA`, and the `pipeline` job of CI runs it).
3. `tests/pipeline/run_e2e.sh` runs `scripts/bam_to_psmc.sh` on simulated
   reads and compares its psmcfa with `vcf2fq -d MIN -D MAX | fq2psmcfa -q20`
   on the script's own VCF, with MIN and MAX computed independently of
   psmc-rs; CI runs it with Ubuntu's bcftools and `vcfutils.pl` and with
   `fq2psmcfa` built from lh3/psmc 0.6.5-r74.

The goldens were made with the `vcfutils.pl` of bcftools 1.24. The few
inputs where psmc-rs differs on purpose (malformed lines are errors, a
chromosome without any base is dropped where fq2psmcfa misreads the next
record, chromosome names must be UTF-8) are listed in the documentation of
`psmc_rs::io::consensus`; `bcftools call -c` output has none of them.
fq2psmcfa's `-x` (mask the human pseudoautosomal regions, in NCBI36
coordinates) and its mutation-class filters (`-v`, `-n`, `-c`, `-C`) have no
psmc-rs equivalent.

Consensus mode also accepts `--vcf-mask` and `--vcf-negative-mask` (for
example a mappability mask): bases outside the callable set become missing.
That is an extension, so the output is then no longer what vcf2fq and
fq2psmcfa give.

## Route B: variant calls and a callable mask

A VCF of variant sites says nothing about the positions it does not list:
they may be homozygous, or not callable at all. A callable mask, a BED file
of the positions where the sample could have been genotyped, supplies that.
psmc-rs reads the pair in **mask mode**: a position is missing outside the
mask, heterozygous where the sample's genotype is a heterozygous SNP, and a
called homozygous site everywhere else in the mask; the bins then follow
fq2psmcfa's rule exactly.

### The variant calls

Any VCF of the sample's calls, sorted (`bcftools sort`), plain or
compressed: GATK HaplotypeCaller and GenotypeGVCFs, DeepVariant, or

```sh
bcftools mpileup -C50 -q20 -Q20 -d 10000 -a FORMAT/DP -f ref.fa sample.bam \
  | bcftools call -m -v -Oz -o variants.vcf.gz
```

In mask mode, for the selected sample (`--vcf-sample` when the VCF has
several): a record whose FILTER is not `PASS` or `.` makes its position
missing (`--vcf-keep-filtered` uses it anyway), so does a missing genotype; a
heterozygous indel makes its whole REF span missing; homozygous calls,
indels included, are homozygous. Several records at one position are
combined. Each sequence runs to its `##contig` length
(`--vcf-no-contig-end` stops it at its last record), and a chromosome
without any record gives no sequence.

### The callable mask

Common ways to make one:

- **msmc-tools `bamCaller.py`** (Schiffels & Durbin 2014): run per
  chromosome on `samtools mpileup -q 20 -Q 20 -C 50 -u -r <chr> -f ref.fa
  sample.bam | bcftools call -c -V indels`, given the mean coverage. It
  writes the chromosome's variant VCF and a mask of the sites whose depth
  is between half and twice the mean coverage, with mapping quality and
  consensus quality of at least 20 (`--minMapQ`, `--minConsQ`); join the
  per-chromosome files as shown below.
- **From a gVCF or an all-sites VCF**: the records and reference blocks
  that pass your depth (and quality) filters are the callable set. With
  bcftools, whose gVCF blocks group sites by bands of depth (`-g MIN,MAX+1`
  puts the sites with depth MIN to MAX in blocks of their own):

  ```sh
  bcftools mpileup -C50 -q20 -Q20 -d 10000 -a FORMAT/DP -f ref.fa sample.bam \
    | bcftools call -m -g 8,51 -Oz -o sample.g.vcf.gz
  bcftools query -i 'FMT/DP>=8 && FMT/DP<=50' -f '%CHROM\t%POS0\t%END\n' \
    sample.g.vcf.gz > callable.bed
  bcftools view -v snps,indels -Oz -o variants.vcf.gz sample.g.vcf.gz
  ```

  (8 and 50 are a third and twice a mean depth of 25.) With GATK, the same
  idea applies to the output of `GenotypeGVCFs --include-non-variant-sites`
  or to the gVCF itself, filtered on depth and genotype quality.
- **mosdepth** (Pedersen & Quinlan 2018): `mosdepth --quantize` writes the
  runs of positions in each depth class to `PREFIX.quantized.bed.gz`,
  labelled with the names in the `MOSDEPTH_Q0`, `MOSDEPTH_Q1`, ...
  environment variables; with `--quantize 0:1:8:51:`, the third class (8
  to 50 reads) is the callable set. mosdepth counts reads, not base
  qualities, so give it a mapping quality cut-off (`--mapq 20`). (This one
  is not part of the tests here.)

Add a **mappability mask** where one exists: the regions where reads of
your length map uniquely, made with Heng Li's SNPable tools (the msmc-tools
README links masks for human GRCh37 and a script, `makeMappabilityMask.py`,
for SNPable output). Li & Durbin (2011) used such a filter; route A
benefits from it too.

psmc-rs intersects all `--vcf-mask` files (a site must be in every one) and
removes every `--vcf-negative-mask` region. A mask row with three or more
columns is BED (`chrom start end`, 0-based, end-exclusive); a row with
exactly two is `start end`, 1-based inclusive, for every chromosome.
Overlapping and adjacent intervals are fine, so per-site rows work, and
rows need not be sorted.

Each `--vcf-mask` file must therefore cover every chromosome you analyse:
a chromosome that one of them lacks has no callable site at all (psmc-rs
warns about it). Masks that come one per
chromosome, as `bamCaller.py` writes them and as msmc-tools' GRCh37
mappability masks are distributed, are joined into one file of each kind
first, and `bamCaller.py`'s per-chromosome VCFs into one VCF:

```sh
gzip -dc mask_chr*.bed.gz > callable.bed
gzip -dc mappability_mask_chr*.bed.txt.gz > mappability.bed
bcftools concat -Oz -o variants.vcf.gz sample_chr1.vcf.gz sample_chr2.vcf.gz ...   # in reference order
```

Given as repeated `--vcf-mask` options instead, per-chromosome masks
intersect to an empty callable set. Repeat `--vcf-mask` for masks of
different kinds, as the commands below do with the callable and the
mappability mask.

### psmc-rs in mask mode

```sh
psmc-rs qc variants.vcf.gz --vcf-mask callable.bed --vcf-mask mappability.bed
psmc-rs variants.vcf.gz fit.json 25 --vcf-mask callable.bed --vcf-mask mappability.bed
psmc-rs convert variants.vcf.gz --vcf-mask callable.bed --vcf-mask mappability.bed \
  --psmcfa sample.psmcfa --multihetsep-per-chrom sample
```

psmc-rs picks mask mode for a VCF that is not an all-sites `bcftools call
-c` file (it says why on stderr); `--vcf-mode mask` forces it.

- **MSMC2** takes one multihetsep file per chromosome: `convert
  --multihetsep-per-chrom sample` writes `sample.<chrom>.multihetsep` for
  the sample's two haplotypes, in `generate_multihetsep.py`'s layout (for one
  chromosome of PASS SNP calls and a mask, the same bytes; the differences
  elsewhere are listed in the documentation of
  `psmc_rs::io::write::write_multihetsep_to`). For several samples, use
  msmc-tools' `generate_multihetsep.py`, which also handles phasing.
- **SMC++**'s `vcf2smc` takes a VCF and, with `--mask`, a BED of regions to
  treat as *missing*: the complement of a callable mask
  (`bedtools complement -i callable.bed -g genome.txt`). In psmc-rs, such a
  file is a `--vcf-negative-mask`.

## Check the input: `psmc-rs qc`

`psmc-rs qc INPUT` reads any input as inference would and reports on it:
missing data, heterozygosity per sequence, depth, sequence lengths, bin
saturation and what the bootstrap would have to work with, with findings
that name a code (`VARIANTS_ONLY_NO_MASK`, `HIGH_MISSING`, `BIN_SATURATION`,
`FEW_HETS`, `TOO_SHORT`, `SHORT_SEQUENCES`, `SEX_OR_ORGANELLE`,
`HET_OUTLIER`, `APPROX_MISSING_PLACEMENT`, `DEPTH`, `DROPPED_SEQUENCES`),
a severity and a recommendation. `--json FILE` writes it for scripts,
`--strict` exits with status 2 when a finding is a warning, and every
inference run prints the same findings and stores their codes in its JSON
output. The script keeps the report as `PREFIX.qc.txt` and `.qc.json`.

## Pitfalls

### A variants-only VCF without a mask

Without a mask, psmc-rs (like SMC++, whose README makes the same point)
counts every position missing from the VCF as a called homozygous site. A
variants-only VCF (what GATK's HaplotypeCaller writes by default, and
`bcftools call -v`) has no record for the positions that could not be
called (no or few reads, low mapping quality, repeats), so all of them
become homozygous calls. The
missing data disappear, the share of `K` bins is diluted, θ and Ne come out
too low, and an uncalled stretch looks like a long run of recent common
ancestry. At the same time nothing removes the false heterozygous calls of
repeats and collapsed duplications, which push the other way. QC reports
`VARIANTS_ONLY_NO_MASK` (a warning) whenever a VCF is read in mask mode
without a positive mask. Give a callable mask (route B) or call all sites
(route A). (`tests/pipeline/run_e2e.sh` reads the same simulated variants
with and without the gVCF-derived mask and checks both outcomes.)

### Sex chromosomes and organelles

PSMC models the two copies of one diploid genome sharing one history.

- **A single copy** (X or Y in males of XY species, Z or W in females of ZW
  species) has no real heterozygous sites; the few that are called are
  errors. PSMC reads such a chromosome as a long stretch of very recent
  common ancestry, which lowers θ and adds a false recent decline. Li &
  Durbin (2011) built their autosomal consensus without them and treated the
  heterozygous calls on male X chromosomes as errors.
- **Two copies of X or Z** (XX females, ZZ males) are diploid but have their own
  effective size (three quarters of the autosomal one under equal sex
  ratios) and mutation rate, so they do not belong in an autosomal fit
  either. The pseudoautosomal regions behave like autosomes.
  `fq2psmcfa -x`, which masks the human pseudoautosomal regions, uses NCBI36
  coordinates, which do not fit GRCh37 or GRCh38.
- **Organelle genomes** are haploid, do not recombine, and are present in
  many copies per cell, so their depth is far above any bound; their reads
  also map to the nuclear copies of organelle DNA (NUMTs).

`scripts/bam_to_psmc.sh` leaves out by default every sequence named like a
sex chromosome or organelle (`X`, `Y`, `Z`, `W` with or without `chr` and a
number, `M`, `MT`, `chrM`, `Pt`, names containing `mitochondri` or
`chloroplast`, and the RefSeq accessions of the human and mouse X, Y and
mitochondrion), with the rule that `psmc-rs qc` uses; the end-to-end test
checks that the two agree on 72 names. The one exception is a reference
whose chromosomes carry Roman numerals past VIII (`chrIX` or `chrXI`, as in
the UCSC stickleback and yeast assemblies): there `chrX` is chromosome 10,
so the script keeps it and says so, while `qc` still notes the name.
(C. elegans's `I` to `V` and `X` stop before IX, and its X, the sex
chromosome, is left out.) Sequences that are sex-linked but not named so
(scaffolds of a draft assembly, or stickleback's `chrXIX`) need an explicit
`--chroms` list. QC's `SEX_OR_ORGANELLE` finding is a note for every such name and a
warning when the sequence is less than a quarter as heterozygous as the
autosomes (judged once the autosomal rate predicts at least 10 `K` bins in
it), as a single copy is; `HET_OUTLIER` flags other sequences far from the
median heterozygosity.

### Low coverage

A heterozygous site is only called when reads of both alleles are sampled,
so at low depth heterozygous sites are lost at random. The psmc README calls
not correcting for this the most common pitfall when using PSMC; the
correction it gives divides θ0 by (1 - FNR), where FNR is the false-negative
rate of heterozygous calls, which is hard to estimate.

psmc-rs has no FNR option, and needs none: the correction changes the
scaling, not the fit. Li & Durbin (2011, Methods) note that missing
heterozygotes uniformly at a rate p is equivalent to a mutation rate of
μ(1 - p). Fitting with `--mu` set to μ(1 - FNR) therefore gives the
corrected curve, HTML report and TMRCA output: N0 = θ0 / (4 μ s) grows by
1 / (1 - FNR), and the times with it. Alternatively, plot the `--psmc-out`
file with `psmc_plot.pl -N FNR`, which applies the same factor (the README
shows the per-sample form, `-M 'sample=FNR'`). Report μ and the FNR, not
only their product. The equivalence holds for a rate that is the same along
the genome; where the depth varies, so does the share of heterozygotes
missed, and one factor only approximates it.

Nadachowska-Brzyska et al. (2016) compared coverages and per-site
depth filters on flycatcher genomes: the curves of different individuals
did not converge below a mean coverage of 18x, Ne rose with the per-site
filter (by about 30% on average at 10 reads against none), and at low
coverage the filter changed the shape of the curve as well. They recommend
a mean coverage of at least 18x, a per-site minimum of 10 reads and at most
25% missing data. At 25x, a third of the mean is about 8, below their 10;
to follow them, give `--depth 10:MAX`. QC warns (`DEPTH`)
when the mean depth is below 18x or when one consensus filter alone removes
more than 30% of the bases, and reports missing data above 25% (note) and
50% (warning) as `HIGH_MISSING`.

### Repeats and mappability

Reads from repeats map to the wrong copy. Where copies differ, their
differences look like heterozygous sites; where the reference lacks a copy
(a collapsed repeat or duplication), its reads pile up on the other.
`-C50`, `-q20`, the upper depth bound and a mappability mask each remove
part of this; none removes all of it. The simulation of the end-to-end test
shows the remainder: at the edges of a collapsed repeat, where its depth
ramps down into the bounds, a few paralog differences survive as
heterozygous calls. Use a mappability mask where one exists, and look at QC's
`HET_OUTLIER` for sequences with too many heterozygous sites.

### Bin size and saturation

PSMC only sees whether a bin holds a heterozygous site, not how many. When
heterozygosity is high, many `K` bins hold two or more, and the bins lose
resolution between old coalescence times, the ancient part of the curve.
QC's `BIN_SATURATION` warns when more than 10% of `K` bins are expected to
hold two or more sites, and recommends the largest bin size that brings
that share below 5%. Give it as `--bin-size` (fq2psmcfa `-s`; the script's
`--bin-size` sets it for the psmcfa and the fit together), and use the same
value whenever the result is scaled: psmc-rs takes it from `--bin-size`,
and `psmc_plot.pl` from `-s`, since a `.psmc` file does not record it
(N0 = θ0 / (4 μ s), psmc README). The bin size of a psmcfa file is fixed
when it is made; changing it means converting again.

### bgzip, sorting and indexes

`tabix` and `bcftools index` need BGZF compression, which `bgzip` and
`bcftools ... -Oz` write; a file compressed with plain `gzip` cannot be
indexed (`bcftools index` stops with "the file is not BGZF compressed").
psmc-rs itself reads plain, gzip and bgzip files, but a VCF must be sorted
(`bcftools sort`): unsorted positions are an error, and so, in mask mode, is
a chromosome that comes back after another (in consensus mode psmc-rs does
what vcf2fq does and starts a second sequence, with a warning). Chromosome
names must agree between the alignments, the reference, the VCF and every
mask (`chr1` against `1`): psmc-rs warns about
chromosomes that a mask does not cover, and the script stops when bcftools
calls no site at all.

### Mutation rate and generation time

μ (`--mu`, per base and generation) and the generation time (`--gen-years`)
do not change the fit, only its scaling (Nadachowska-Brzyska et al. 2016):
Ne scales with 1/μ and time in years with g/μ, so halving μ doubles both
axes. Li & Durbin (2011) used μ = 2.5 × 10⁻⁸ and 25 years for humans;
direct estimates from human pedigrees are about 1.2 × 10⁻⁸ (Kong et al.
2012), which roughly doubles Ne and times (Scally & Durbin 2012 review the
consequences). μ also varies along the genome and between mutation types
(psmc README). The generation time is the mean age of parents at the birth
of their offspring, which is often well above the age at maturity. For a
rate known per year, multiply by the generation time. Report both values
with any curve.

### Reference bias and contamination

These act on the heterozygous calls themselves; how much they move a curve
depends on the data, and nothing here measures it.

- **Reference bias**: a read carrying the non-reference allele has one more
  mismatch, so it maps a little worse and is more often filtered, and
  heterozygous sites can be under-called. The effect grows with the
  divergence between sample and reference; a reference of another species
  or population also leaves more of the genome unmapped, concentrating
  missing data in divergent regions. Use a reference of the same species
  where one exists.
- **Contamination** with DNA of another individual of the same species adds
  alleles: false heterozygous calls with unbalanced allele counts (far from
  half the reads), which raise θ. Check the allele balance at heterozygous
  sites (`FORMAT/AD`, which bcftools mpileup writes; older versions need
  `-a AD`), and QC's `HET_OUTLIER`.

### Haploid and inbred samples

- **Haploid samples** (males of haplodiploid species, haploid cell lines,
  gametophytes) have no heterozygous sites at all, so one cannot be
  analysed alone. Two haploid genomes of one population can be combined
  into a pseudo-diploid, marking their differences as heterozygous, as Li &
  Durbin (2011) did with the X chromosomes of two males; psmc-rs has no
  dedicated tool for this.
- **Inbred samples** have long runs of homozygosity: real stretches of very
  recent common ancestry, which PSMC turns into a small recent Ne. That is
  a signal of inbreeding, not necessarily of a population decline. Locate
  the runs (for example with `bcftools roh`) before reading the recent end
  of the curve; QC's per-sequence table and `HET_OUTLIER` show chromosomes
  with unusually low heterozygosity.

## References

- Li H, Durbin R (2011) Inference of human population history from
  individual whole-genome sequences. *Nature* 475:493–496.
- psmc README, https://github.com/lh3/psmc (commands, the depth bounds, the
  low-coverage correction, scaling).
- Nadachowska-Brzyska K, Burri R, Smeds L, Ellegren H (2016) PSMC analysis of
  effective population sizes in molecular ecology and its application to
  black-and-white *Ficedula* flycatchers. *Molecular Ecology* 25:1058–1072.
- Schiffels S, Durbin R (2014) Inferring human population size and
  separation history from multiple genome sequences. *Nature Genetics*
  46:919–925; msmc-tools, https://github.com/stschiff/msmc-tools
  (`bamCaller.py`, `generate_multihetsep.py`, mappability masks).
- Terhorst J, Kamm JA, Song YS (2017) Robust and scalable inference of
  population history from hundreds of unphased whole genomes. *Nature
  Genetics* 49:303–309 (SMC++).
- Pedersen BS, Quinlan AR (2018) Mosdepth: quick coverage calculation for
  genomes and exomes. *Bioinformatics* 34:867–868.
- Kong A et al. (2012) Rate of de novo mutations and the importance of
  father's age to disease risk. *Nature* 488:471–475.
- Scally A, Durbin R (2012) Revising the human mutation rate: implications
  for understanding human evolution. *Nature Reviews Genetics* 13:745–753.
