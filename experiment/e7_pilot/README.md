# E7's chr21 pilot, with the production's psmc-rs

`pilot_chr21.json` is E7's chr21 pilot run with the binary of the production's
psmc-rs fits (`src/` tree dcf29ca, sha256 1d141b81…): KAUST Ibex, SLURM job
54553272 on 25 October 2026, with `kg1000`'s own functions. The joint chr21
VCF was downloaded whole (its size and MD5 those of E7's `MANIFEST.json`),
streamed once through `bcftools view` and converted directly for NA18507,
NA19239 and NA12878 (`commands.convert`), and their per-sample VCFs that E7's
data stage kept were converted with the stage's command
(`commands.split_convert`). The two routes gave byte-identical `psmcfa` files
and equal QC (`identical`), both identical after decompression to E7's chr21
files and QC rows (`identical_to_e7`). `manuscript/make_numbers.py` reads it.

E7's data stage keeps its own pilot, run with its own build, in
`runs/E7/inputs/records/pilot_chr21.json` and `MANIFEST.json`; this record
is kept apart because the data stage collects every `records/pilot_*.json`
into `MANIFEST.json`, which E7's report reads.
