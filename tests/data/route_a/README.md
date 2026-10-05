# Route A fixtures: the classic PSMC input pipeline

Golden files for `tests/io_route_a.rs`. They come from the real tools, so the
tests can check psmc-rs against them without the tools installed.

## What is here

| File | What it is |
| --- | --- |
| `simulate.py` | Deterministic simulator (Python standard library, seed 20261005): a reference of three chromosomes, two haplotypes with SNPs and indels, and SAM alignments of 100 bp reads. |
| `helpers.py` | Independent re-implementations used by the script: the automatic depth thresholds, a `bamCaller.py`-style callable mask, and phasing of heterozygous genotypes. |
| `make_fixtures.sh` | Regenerates everything below from scratch. |
| `allsites.vcf.gz` | `bcftools mpileup -C50 -q20 -Q20 -f ref.fa aln.sam \| bcftools call -c`, all three chromosomes (15,819 records), with the fields vcf2fq does not read removed (see below). |
| `allsites_chr3_unmodified.vcf.gz` | chr3 of the same call, with every field `bcftools call -c` wrote. |
| `snps.vcf.gz` | `bcftools mpileup ... \| bcftools call -mv -V indels`: SNP calls only. |
| `snps_phased.vcf.gz` | The same, with the heterozygous genotypes written as phased, `0\|1` and `1\|0` in turn. |
| `masks/mask_<chrom>.bed.gz` | Callable sites per chromosome, as `bamCaller.py` defines them: one-base ACGT REF, ALT one of `ACGTacgt.`, mean/2 <= DP <= 2*mean, MQ >= 20, \|FQ\| >= 20 (from `allsites.vcf.gz`). |
| `expected/<name>.psmcfa[.gz]` | `vcfutils.pl vcf2fq ... \| fq2psmcfa ...` for each row of `expected/option_sets.tsv`. |
| `expected/<name>.fq.gz` | The FASTQ `vcfutils.pl vcf2fq ...` writes for each row, which `fq2psmcfa` reads: the tests count the callable bases of each bin from it. |
| `expected/option_sets.tsv` | The option sets: name, vcf2fq options, fq2psmcfa options, input. |
| `expected/auto_depth.tsv` | Mean INFO/DP over vcf2fq's SNP/reference records and the derived bounds `max(1, round(mean/3))`, `round(2*mean)`, used by the `auto` row. |
| `expected/<vcf>.<chrom>.multihetsep` | `generate_multihetsep.py --mask masks/mask_<chrom>.bed.gz <vcf>.<chrom>.vcf.gz`, one chromosome per run (the script assumes it). |
| `expected/versions.txt` | Tool versions and checksums of the run that produced these files. |

## The simulated genome

`simulate.py` lays out coverage so that every filter of the pipeline acts:
chr1 (11,500 bp) has 30x coverage with a 4x dip, a 400 bp gap without reads,
a 90x pile-up and an 800 bp stretch of mapping quality 25; chr2 (3,000 bp)
has a 12x dip; chr3 (1,800 bp) is at 8x. Every chromosome also gets 2x of
reads with mapping quality 5 (dropped by `-q20`), 5% of bases have quality
10 (dropped by `-Q20`), and 0.4% of bases are sequencing errors. Variants:
heterozygous SNPs at 1/250, homozygous SNPs at 1/2000, and 0.4 indels per kb
(80% heterozygous). chr1 is just long enough to pass fq2psmcfa's default
output filter (10,000 called bases); chr2 and chr3 are not.

## Option sets

| name | vcf2fq | fq2psmcfa | input |
| --- | --- | --- | --- |
| `default` | (defaults: `-d 3 -D 100000 -Q 10 -l 5`) | (defaults: `-q 10 -g 10000 -s 100`) | `allsites.vcf.gz` |
| `depth` | `-d 10 -D 60` | `-g 1000 -s 50` | `allsites.vcf.gz` |
| `mq_window` | `-Q 30 -l 10` | `-q 20 -g 500 -s 10` | `allsites.vcf.gz` |
| `per_base` | `-d 1 -D 1000 -Q 0 -l 1` | `-q 0 -g 0 -s 1` | `allsites.vcf.gz` |
| `auto` | `-d 9 -D 57` (from `auto_depth.tsv`) | `-g 2000` | `allsites.vcf.gz` |
| `unmodified_chr3` | `-d 3` | `-g 100 -s 20` | `allsites_chr3_unmodified.vcf.gz` |

The tests read the VCF with psmc-rs's loader in automatic mode (which must
pick consensus mode), the same thresholds (`DepthThresholds::Auto` for the
`auto` row), write it with `write_psmcfa`, and compare the bytes.

## Removed fields

With every field `bcftools call -c` writes, `allsites.vcf.gz` would be 183 KB.
vcf2fq reads only CHROM, POS, REF, ALT and the INFO keys DP, MQ, FQ, AF1 and
INDEL, so the script removes the rest with
`bcftools annotate -x 'QUAL,^INFO/DP,INFO/MQ,INFO/FQ,INFO/AF1,INFO/INDEL,^FORMAT/GT'`
(83 KB). Keys that vcf2fq's regular expressions could confuse with these,
such as `DP4` or `MQ0F`, are still present in
`allsites_chr3_unmodified.vcf.gz` and covered by `tests/io_consensus.rs` and
`tests/io_consensus_tools.rs`.

## Regenerating

```sh
VCFUTILS=/path/to/bcftools/misc/vcfutils.pl \
FQ2PSMCFA=/path/to/psmc/utils/fq2psmcfa \
GENERATE_MULTIHETSEP=/path/to/msmc-tools/generate_multihetsep.py \
tests/data/route_a/make_fixtures.sh
```

`bcftools` must be on `PATH`; perl and python3 are needed for the scripts.
The output is byte for byte reproducible (bcftools runs with `--no-version`
on relative paths, and gzip with `-n`): with the same tools, a rerun leaves
every file unchanged except the date in `expected/versions.txt`.
The files in this directory were made on macOS arm64 with bcftools 1.24
(htslib 1.24) and the `vcfutils.pl` shipped with it, `fq2psmcfa` built from
the lh3/psmc 0.6.5-r74 sources also used for `tests/data/c_reference`,
`generate_multihetsep.py` from msmc-tools, perl 5.34.1 and Python 3.14.6;
`expected/versions.txt` has the checksums of the three scripts and binary. The
simulation uses only `random.Random` with a fixed seed, whose sequence is
stable across Python 3 releases. The `.fq.gz` files were added later, by the
same commands with the same `vcfutils.pl` (its checksum matches
`versions.txt`) and perl; `tests/io_route_a.rs` checks that binning them
gives the `fq2psmcfa` output above, bin for bin.
