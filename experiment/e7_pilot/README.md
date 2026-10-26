# E7's data stage, with the production's psmc-rs

Two checks that the binary of the production's psmc-rs fits (`src/` tree
dcf29ca, sha256 1d141b81…) writes the content E7's data stage wrote with its
own build.

## The chr21 pilot

`pilot_chr21.json` is E7's chr21 pilot run with that binary: KAUST Ibex,
SLURM job 54553272 on 25 October 2026, with `kg1000`'s own functions. The
joint chr21 VCF was downloaded whole (its size and MD5 those of E7's
`MANIFEST.json`), streamed once through `bcftools view` and converted
directly for NA18507, NA19239 and NA12878 (`commands.convert`), and their
per-sample VCFs that E7's data stage kept were converted with the stage's
command (`commands.split_convert`). The two routes gave byte-identical
`psmcfa` files and equal QC (`identical`), both identical after
decompression to E7's chr21 files and QC rows (`identical_to_e7`).
`manuscript/make_numbers.py` reads it.

E7's data stage keeps its own pilot, run with its own build, in
`runs/E7/inputs/records/pilot_chr21.json` and `MANIFEST.json`; this record
is kept apart because the data stage collects every `records/pilot_*.json`
into `MANIFEST.json`, which E7's report reads.

## The default genomes' conversions

`stage1.json` (the counts), `stage1_chrom.tsv` (one row per genome and
chromosome) and `stage1_samples.tsv` (one row per genome): KAUST Ibex, SLURM
job 54553271 on 25 October 2026, an AMD EPYC 7702 node. Every chromosome of
the 52 default genomes, whose per-sample VCFs the data stage kept, was
converted again from the VCF the stage converted (its size the one in
`qc/<chrom>.tsv`), with the stage's command (`MANIFEST.json`'s
`commands.assemble_chrom.convert`):

```bash
psmc-rs convert $I/vcf/$C/$S.vcf.gz --vcf-mode mask --vcf-mask $I/mask/$C.bed \
  --vcf-negative-mask $I/nonpass/$C.bed.gz --bin-size 100 \
  --psmcfa $NEW/$C/$S.psmcfa.gz --qc-json $NEW/$C/$S.qc.json -q
```

with `I=<runs>/E7/inputs`, and `e7_check.py` compared the outputs with E7's
(`SAMPLES.txt`: the 52 genomes, one per line, those of `$I/vcf/chr1`):

```bash
experiment/.venv/bin/python experiment/e7_pilot/e7_check.py . $I $NEW SAMPLES.txt OUT \
  experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz
```

All 1,144 chromosome files and all 52 genomes (their 22 chromosome files
concatenated, as the fits read them) are identical to E7's after
decompression, with QC equal to E7's `qc/<chrom>.tsv` rows and
`qc/samples.tsv` sums; only the gzip bytes differ. The repository's
`experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz` is E7's NA12878
file byte for byte.
