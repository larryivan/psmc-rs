# E7's data stage, with the production's binary

Two checks that the binary of the production's psmc-rs fits (`src/` tree
dcf29ca, sha256 1d141b81…) writes the content E7's data stage wrote with its
own build.

## The chr21 pilot

`pilot_chr21.json` is E7's chr21 pilot run with that binary by `pilot.py`,
which chains `kg1000`'s own functions: KAUST Ibex, SLURM job 54553272 on 25
October 2026. The joint chr21 VCF was downloaded whole by
`kg1000.fetch_ranges` (`fetch`: its size and MD5 those of E7's
`MANIFEST.json`), streamed once through `bcftools view` (`commands.stream`)
and fed by `kg1000.tee` to a direct conversion for each of NA18507,
NA19239 and NA12878 (`commands.convert`, shown for NA18507), and their
per-sample VCFs that E7's data stage kept were converted with the stage's
command (`commands.split_convert`). The two routes gave byte-identical
`psmcfa` files and equal QC (`identical`), both identical after
decompression to E7's chr21 files and QC rows (`identical_to_e7`).
`manuscript/make_numbers.py` reads it.

The job ran `pilot.py` on an archive of commit 4b2c636f (the production's
`src/` and `Cargo.lock`; its `Cargo.toml`, c0be5cf, lacks the package
metadata of 4813f6d5, the `Cargo.toml` that the psmc-rs records of E1,
E2 and E4 to E7 name) holding the production's binary as
`target/release/psmc-rs`, with `PY` as `experiment/slurm/env.sh` sets it,
`I=<runs>/E7/inputs`, `LOCAL_TMP` node-local scratch for the 26.1 GB
download and `SPLIT` the directory of the split route's files, which
`pilot.py` converts when they are absent, as it did here (`split_route`):

```bash
A=$(mktemp -d)
git archive 4b2c636f experiment/pipeline Cargo.toml Cargo.lock | tar -x -C $A
mkdir -p $A/target/release && cp $BIN $A/target/release/psmc-rs   # the production's binary
$PY experiment/e7_pilot/pilot.py $A $I $LOCAL_TMP $OUT $SPLIT
```

Its docstring's I4 is this job and I1 the stage-1 job below (54553271).
`pilot_chr21.json` is the job's `$OUT/pilot_chr21.json` with the job's
paths in `commands` and `split_route` written as placeholders (`<repo>`
for the archive, `<runs>`, `<local-tmp>`, `<split>`), `split_route` naming
the data stage's command, a `note` added, and `psmc-rs.source_commit` and
`source_tree` those of the checkout the binary was copied from, at
4b2c636f, since `kg1000.rs_identity()` found no git checkout in the archive
(`psmc-rs.source_note`).

E7's data stage keeps its own pilot, run with its own build, in
`runs/E7/inputs/records/pilot_chr21.json` and `MANIFEST.json`; this record
is kept apart because the data stage collects every `records/pilot_*.json`
into `MANIFEST.json`, which E7's report reads.

## The default genomes' conversions

`stage1.json` (the counts), `stage1_chrom.tsv` (one row per genome and
chromosome) and `stage1_samples.tsv` (one row per genome): KAUST Ibex, SLURM
job 54553271 on 25 October 2026, an AMD EPYC 7702 node; `stage1.json`'s
`psmc-rs` and `where` (the binary, its source and the node) are from the
job's identity record, since `e7_check.py` records counts and checksums
only. Every chromosome of the 52 default genomes, whose per-sample VCFs
the data stage kept, was converted again from the VCF the stage converted
(its size the one in `qc/<chrom>.tsv`), with the stage's command
(`MANIFEST.json`'s `commands.assemble_chrom.convert`):

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
