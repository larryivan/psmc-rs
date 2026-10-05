# End-to-end test of `scripts/bam_to_psmc.sh`

`run_e2e.sh` simulates the reads of a small diploid genome, runs
`scripts/bam_to_psmc.sh` on them and checks the outputs against the classic
tools and the simulation truth. CI runs it in the `pipeline` job of
`.github/workflows/ci.yml`.

```sh
cargo build --release
VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa \
  tests/pipeline/run_e2e.sh
```

It needs bcftools (1.10 or later), bgzip, tabix, perl and python3 on `PATH`;
`vcfutils.pl` (shipped with bcftools, as `misc/vcfutils.pl` in its sources and
in `/usr/bin` of Debian's and Ubuntu's package) and `fq2psmcfa` (built from
`utils/fq2psmcfa.c` of lh3/psmc) default to the copies on `PATH`, and
psmc-rs to `target/release/psmc-rs` (`PSMC_RS`). `E2E_WORK=DIR` keeps the
files in DIR; `E2E_BASH=/bin/bash` runs the script with macOS's bash 3.2. It
takes about half a minute.

| File | What it is |
| --- | --- |
| `simulate.py` | Deterministic simulator (Python standard library, seed 20261005). Writes `ref.fa`, `aln.sam` (coordinate-sorted SAM text with one read group), `truth.tsv` (the variants), `design.tsv` (the coverage layout), `callable.bed` and `masked.bed` (where the pipeline must call every base, and none). |
| `check_outputs.py` | The checks against the truth: heterozygous-site recovery, the QC findings, the multihetsep rows, the fit, and the sex/organelle name rule. |
| `run_e2e.sh` | Runs everything below and stops at the first failure. |

## The simulated genome

A male sample at 25x, 100 bp single-end reads: three autosomes of 150, 100
and 70 kb, an 8 kb unplaced scaffold, an 80 kb chrX that is a single copy
(half the depth, no heterozygous sites) outside an 8 kb pseudoautosomal
region, and a 3 kb chrM at 150x. Heterozygosity varies along the autosomes in
segments of about 20 kb (0.0015 per bp times an exponential draw), with
homozygous SNPs and heterozygous and homozygous indels. The coverage puts
every filter to work: a 2x dip and a stretch without reads (below the
automatic lower depth bound), a collapsed repeat (100x of reads from a 1.5%
divergent paralog, which look heterozygous, above the upper bound), a
soft-masked repeat whose reads have mapping quality 0, a 12x dip that stays
callable, an assembly gap, 2x of reads with mapping quality 5 everywhere, 5%
of bases at quality 10, and 0.3% sequencing errors.

## What is checked

- The command line: `--help`, usage errors (exit 2), a missing input, a
  bcftools older than 1.10 and a psmc-rs older than 0.2.0 (fake binaries),
  unknown or repeated `--chroms` names, and `--chroms` from a BED file with
  Windows line ends, track and browser lines and several rows for one
  chromosome. The fit gets `--threads` only when
  it is given; where `taskset` exists (Linux), `taskset -c 0` makes the
  script run one job at a time. The dry run leaves out exactly the
  names that `psmc-rs qc` classifies as sex chromosomes or organelles, for
  72 names in the styles of UCSC, Ensembl, RefSeq and other assemblies.
- Run 1, the plain SAM (no index, so each job streams the file), 3 jobs,
  `--multihetsep --run 3`:
  - chrX and chrM are left out by name, and the VCF holds only the other
    four sequences;
  - (a) the psmcfa equals `vcfutils.pl vcf2fq -d MIN -D MAX | fq2psmcfa -q20
    -s100` on the same VCF, byte for byte, where MIN and MAX are computed
    from the VCF independently of psmc-rs (`tests/data/route_a/helpers.py
    auto-depth`) and must equal the bounds psmc-rs reports; the scaffold
    with fewer than 10,000 called bases is dropped by both;
  - (b) at least 97% of the simulated heterozygous SNPs in `callable.bed` lie
    in `K` bins (the run reports the rate; it is above 99%), at most 1% of
    `K` bins hold no simulated heterozygous variant, and at least 97% of the
    bins inside `masked.bed` are `N`; every multihetsep row lies in a `K`
    bin and at least 98% are simulated heterozygous SNPs (the others are
    paralog differences at the edge of the collapsed repeat);
  - (c) the QC report holds the three autosomes, one sequence dropped by the
    output filter, and no sequence named like a sex chromosome or organelle;
  - (d) the fit has 3 iterations, 4 finite log-likelihoods and rounds 0 to 3
    in the `.psmc` file.
- Run 2, the same reads as bgzipped SAM with a CSI index (`tabix -C -p
  sam`), so each job reads its chromosome by region, 1 job, the same prefix:
  every output, the HTML report included, is byte-identical to run 1.
- Run 3, `--chroms` from a file that adds chrX: the script warns, the psmcfa
  again equals vcf2fq | fq2psmcfa, and QC gives a SEX_OR_ORGANELLE warning
  calling chrX hemizygous.
- Run 4, chr3 renamed `HLA-A*01:01:01:01`: called from the plain and the
  indexed SAM, the name survives (`bcftools mpileup -r` would read the
  colons as a range) and both give the same psmcfa.
- High depth: chr1 simulated at 140x with 200x of paralog reads in the
  collapsed repeat (`simulate.py --depth 140 --paralog-depth 200 --chroms
  chr1`). The automatic upper bound is about 280, above mpileup's default
  cap of 250 reads; with the script's `-d 10000`, at most half of the
  repeat's bins may be `K` (mpileup's default made 20 of 25 `K`), and the
  psmcfa again equals vcf2fq | fq2psmcfa.
- Route B of `docs/preprocessing.md` on the same reads: a gVCF from
  `bcftools call -m -g`, its depth-banded blocks as the callable mask and its
  variant records as the VCF. Without the mask, QC warns
  VARIANTS_ONLY_NO_MASK; with it, the warning is gone and recovery, `K` and
  `N` bins and the multihetsep rows meet the same thresholds as run 1.
- Failures: no read passing `--min-mapq 255`, two samples in one file, and
  a `--max-depth` below twice the automatic upper bound each end with an
  error and leave only the log (no output file, no working directory);
  `--max-depth` below twice MAX of `--depth MIN:MAX` is a usage error.
- Checks before the calls: two samples in an indexed file, and a typo in
  the fit's options after `--` (`--patern`), each stop the run before any
  chromosome is called, the latter with psmc-rs's error line and tip. A fit
  that fails after reading its input (a single bootstrap trunk) shows
  psmc-rs's error line, which follows pages of QC text.
- Interruptions, with a stand-in that never finishes: SIGTERM while the
  calls run, with a `PATH` that has no `pkill`, `pgrep` or `ps`, leaves only
  the log; SIGTERM to the script, and SIGINT to its process group (what
  Ctrl+C sends), while the fit runs stop it within seconds (exit 143 and
  130) and leave no fit output. In every case no working directory and no
  process is left.
