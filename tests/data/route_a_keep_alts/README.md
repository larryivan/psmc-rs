# Keep-alts fixtures: `bcftools call -c -A` at high depth

Golden files for `tests/io_route_a.rs`, made by the real tools so that the
tests can check psmc-rs against them without the tools installed.

With `-A` (keep alternative alleles), `bcftools call -c` writes an ALT allele
at every reference site where some read shows another base, while the call
stays a reference call (`FQ < 0`, `AF1=0`). At high depth most sites have
such a read, so fewer than half of the records of the all-sites VCF have ALT
`.`. Automatic mode detection must still read the file in consensus mode, as
`vcfutils.pl vcf2fq` does, and a variants-only `call -cv` VCF of the same
reads, which also carries INFO/FQ on every record, must stay in mask mode.

## What is here

| File | What it is |
| --- | --- |
| `simulate_high_depth.py` | `../route_a/simulate.py` (same seed, variants, errors and qualities) with one chromosome of 3,000 bp at 250x. |
| `make_fixtures.sh` | Regenerates everything below from scratch. |
| `keep_alts.vcf.gz` | `bcftools mpileup -C50 -q20 -Q20 -f ref.fa aln.sam \| bcftools call -c -A`, with the fields vcf2fq does not read removed as in `../route_a` (3,000 records, 1,297 with ALT `.`, all with INFO/FQ). |
| `variants_c.vcf.gz` | `bcftools mpileup ... \| bcftools call -cv` from the same reads, unmodified (17 records, all with INFO/FQ). |
| `expected/auto_depth.tsv` | Mean INFO/DP over vcf2fq's SNP/reference records and the bounds `max(1, round(mean/3))`, `round(2*mean)`, from `../route_a/helpers.py auto-depth`. |
| `expected/keep_alts.psmcfa` | `vcfutils.pl vcf2fq -d 77 -D 463 \| fq2psmcfa -g 1000 -s 50` (the bounds of `auto_depth.tsv`). |
| `expected/records.tsv` | Records, records with ALT `.` and records with INFO/FQ of both VCFs. |
| `expected/versions.txt` | Tool versions and checksums of the run that produced these files. |

mpileup's default `--max-depth 250` caps the depth, so the mean INFO/DP is
231.4 rather than 250.

## Regenerating

```sh
VCFUTILS=/path/to/bcftools/misc/vcfutils.pl \
FQ2PSMCFA=/path/to/psmc/utils/fq2psmcfa \
tests/data/route_a_keep_alts/make_fixtures.sh
```

`bcftools` must be on `PATH`; perl and python3 are needed for the scripts.
With the same tools a rerun leaves every file unchanged except the date in
`expected/versions.txt`. The files here were made on macOS arm64 with
bcftools 1.24 (htslib 1.24) and its `vcfutils.pl`, `fq2psmcfa` from lh3/psmc
0.6.5-r74 (the build used for `../route_a`), perl 5.34.1 and Python 3.14.6.
