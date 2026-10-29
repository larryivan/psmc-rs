# Validation

How psmc-rs is checked against C psmc and the classic tools, and what the
manuscript's experiments measured. The experiments are described in
[experiment/pipeline/README.md](../experiment/pipeline/README.md) and their
reproduction in
[Supplementary Note 1](../experiment/Supplementary_Note_1_Reproducibility.md);
their tables are under `experiment/runs/`.

- [Agreement with C psmc](#agreement-with-c-psmc)
- [The tests](#the-tests)
- [The classic tools](#the-classic-tools)
- [Missing data](#missing-data)
- [Bootstrap calibration](#bootstrap-calibration)
- [Speed and memory](#speed-and-memory)
- [Across platforms](#across-platforms)

## Agreement with C psmc

psmc-rs fits C psmc's model with its time grid, its starting point (θ0 =
−ln(1 − K/L), ρ0 = θ0/4, flat λ, t_max = 15) and its `-p` patterns.

**On the reference data** (a 180,000-bin psmcfa, C psmc 0.6.5-r74, patterns
`4+5*3+4` and `4+25*2+4+6`), the first round's log-likelihood equals C
psmc's to its printed precision, and rounds 2 to 5 stay within a relative
2e-7; `tests/c_equivalence.rs` enforces both. The later rounds differ
mostly because C psmc's expected counts leave out the last bin of each
sequence; the M-steps (L-BFGS here, Hooke–Jeeves in C psmc) both run to
convergence.

**On the manuscript's genomes**, both programs fitted with
`psmc -N25 -t15 -r5 -p "4+25*2+4+6"` and compared interval by interval over
the informative window (the 1% to 99% quantiles of the coalescence time):

| genomes | RMSE of log10 Ne, psmc-rs against C psmc |
| --- | --- |
| E1: 40 simulated 500 Mb genomes of four models | 3.7e-6 to 2.0e-4 (median 2.5e-5) |
| E2: four real genomes (three turtles and NA12878) | 9.7e-6 to 2.8e-4 |
| E7: the 52 default genomes of the 1000 Genomes cohort | 5.7e-6 to 2.1e-5 |

Over the 4,620 intervals in the window of the 95 genomes both fitted
(NA12878 counted once), the median absolute difference in log10 Ne is
5.5e-6 and the largest 6.6e-4: the population sizes differ by at most
0.15%. In E1 and E2, θ, ρ and t_max agreed within a relative 5.5e-4 and
1.0e-3, and the log-likelihood of the last round C psmc prints within
1.3e-8 and 2.3e-7 (at every round within 3.5e-8 and 3.7e-7). The tables are
`E1_rs_vs_c.csv`, `E2_rs_vs_c.csv` and `E7_rs_vs_c.csv`.

**Where the difference comes from.** On E1's genomes the sources were
traced with variants of psmc-rs built for the purpose
(`experiment/c_equivalence`):

- the M-steps are not what keeps the programs apart: psmc-rs's default fits
  are within a median 1.0e-7 (at most 2.5e-6) of fits whose M-steps run to
  the precision limit, and a port of C psmc's Hooke–Jeeves search in place
  of L-BFGS leaves the curves as far from C psmc's (a median 2.6e-5,
  against 2.5e-5);
- most of the rest comes from C psmc's expected counts, which leave out the
  last bin of each sequence (`em::posterior_sigma`). With that bin left out
  of psmc-rs's counts too, 37 of the 40 curves come closer to C psmc's and
  the median falls from 2.5e-5 to 6.2e-6, below the median of 8.5e-6
  between two builds of C psmc (`experiment/crossplatform`); 27 of the 40
  are then at least as close to C psmc's as C psmc's two builds are to each
  other. On E1's first bottleneck replicate, for one, leaving out the last
  bin takes the curve from 2.8e-5 to 1.6e-5 of C psmc's, where C psmc's two
  builds are 2.2e-5 apart. With the last bin left out, rounds 2 to 5 on the
  reference data agree with C psmc's log-likelihoods to a relative 2.8e-9
  instead of 7.3e-8.

**Before the window** the curves can differ more. In one of E1's 40
genomes (an expansion replicate) the first λ, which the data barely
determine, ran to the M-step's upper bound of 1e4 in psmc-rs (the run says
so) and to 1.08e6 in C psmc, whose search has no bound (the Hooke–Jeeves
port, unbounded too, took it to 1.03e6). Ne in the first four intervals (up
to about 4,800 years; the window starts at about 5,000) is then 2.03 log10
units apart, and neither is an estimate.

## The tests

```bash
cargo test                        # unit, integration and doc tests
python3 -m unittest discover -s experiment/tests
```

The suite compares psmc-rs with the original tools' outputs, stored under
`tests/data/` with the scripts that made them:

- `tests/c_equivalence.rs`: EM on a 180,000-bin psmcfa against C psmc
  0.6.5-r74 (patterns `4+5*3+4` and `4+25*2+4+6`): round 1 equal to C
  psmc's printed precision, rounds 2 to 5 within a relative 2e-7.
- `tests/psmc_out_c_reference.rs`, `tests/em_c_reference.rs` and
  `tests/psmc_out_plot.rs`: the `.psmc` output line by line against C
  psmc's (but `QD`), C psmc's expectations (`post_sigma`, `RI`) and
  psmc_plot.pl's reading of the file.
- `tests/io_text_psmcfa_c.rs`: psmcfa parsing against C psmc's counts on
  the turtle and Papuan psmcfa files in `experiment/real_data`.
- `tests/io_route_a.rs`: consensus mode against goldens made with
  `bcftools call -c | vcfutils.pl vcf2fq | fq2psmcfa`, and multihetsep
  output against `generate_multihetsep.py`, byte for byte.
- `tests/bootstrap_split.rs` and `tests/bootstrap_resample.rs`: trunks
  against `splitfa`, draws against `psmc_resamp`.

Two ignored tests run the tools themselves:
`VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa cargo test
--release --test io_consensus_tools -- --ignored` and `FQ2PSMCFA=...
cargo test --release --test io_vcf_exact -- --ignored`. CONTRIBUTING.md
describes how reference outputs may be regenerated.

## The classic tools

For the same VCF and the same thresholds, the psmcfa that psmc-rs writes
from an all-sites VCF of `bcftools call -c` (`psmc-rs convert --psmcfa`,
and so `scripts/bam_to_psmc.sh`'s) is byte for byte the output of

```sh
gzip -dc sample.vcf.gz | vcfutils.pl vcf2fq -d MIN -D MAX | fq2psmcfa -q20 - > sample.psmcfa
```

and a fit of the VCF fits exactly these bins (with complete data, or with
`--bin-callable ignore`). This is tested three ways:

1. `tests/io_route_a.rs` compares psmc-rs with golden files that the real
   tools made from simulated reads (`tests/data/route_a`): six option sets,
   one of them one bin per base (`-s 1 -g 0`), which compares every base.
2. `tests/io_consensus_tools.rs` runs the real tools on hundreds of random
   VCFs full of edge cases (`--ignored`; it needs `VCFUTILS` and
   `FQ2PSMCFA`, and the `pipeline` job of CI runs it).
3. `tests/pipeline/run_e2e.sh` runs `scripts/bam_to_psmc.sh` on simulated
   reads and compares its psmcfa with `vcf2fq -d MIN -D MAX | fq2psmcfa
   -q20` on the script's own VCF, with MIN and MAX computed independently
   of psmc-rs; CI runs it with Ubuntu's bcftools and `vcfutils.pl` and with
   `fq2psmcfa` built from lh3/psmc 0.6.5-r74.

The goldens were made with the `vcfutils.pl` of bcftools 1.24. The few
inputs where psmc-rs differs on purpose (malformed lines are errors, a
chromosome without any base is dropped where fq2psmcfa misreads the next
record, chromosome names must be UTF-8) are listed in the documentation of
`psmc_rs::io::consensus`; `bcftools call -c` output has none of them.

## Missing data

**Partly callable bins** ([inputs.md](inputs.md#partly-callable-bins)). On
five simulated 500 Mb genomes of a bottleneck model (experiment E4,
`-N25 -t15 -r5 -p 4+25*2+4+6`) with 30% of the bases missing in runs of
about 20 bp, the fit with the callable counts removed most of the bias: θ
was 0.97 to 0.99 of the fit of the same genome without missing data (1.9%
low on average), where ignoring the counts gave 0.64 to 0.66. The RMSE of
log10 Ne against the truth was 0.007 above that of the complete-data fit on
average (at most 0.020), where ignoring the counts raised it by 0.16 to
0.20. Missing data in 20 kb runs changed the fits little either way. E4's
tables are in `experiment/runs/E4/tables/`, its design in
[experiment/pipeline/README.md](../experiment/pipeline/README.md#e4-input-preparation).

**multihetsep** ([inputs.md](inputs.md#multihetsep)), whose reader places
the uncalled sites itself:

- on E4's genomes, θ came out 0.67 to 0.69 of the fit without missing data
  with 30% of the sites uncalled in runs of about 20 bp, and 0.89 to 0.91 in
  runs of about 20 kb;
- in the smaller runs of [experiment/validation/](../experiment/validation/README.md)
  (three runs of three 10 × 20 Mb genomes), θ came out 0.82 to 0.86 of the
  complete-data fit with 10% of the sites uncalled in runs of about 20 bp,
  0.67 to 0.70 with 30%, and 0.89 to 0.93 with 30% in runs of about 20 kb;
  the approximate counts (`--bin-callable approximate`) made θ 0.52 to 0.60
  of the complete-data fit with 30% uncalled in runs of about 20 bp, which
  is why the default does not use them;
- in the same runs, a VCF fitted with its partly callable bins treated as
  full (`--bin-callable ignore`) gave θ 0.87 to 0.89 of the complete-data
  fit with 10% of the bases missing in runs of about 20 bp and 0.63 to 0.66
  with 30%, against 0.985 to 0.999 and 0.944 to 0.991 with the counts
  (`auto`).

## Bootstrap calibration

Experiment E5
([experiment/pipeline/README.md](../experiment/pipeline/README.md#e5-bootstrap-calibration))
bootstrapped 10 simulated 300 Mb genomes of a bottleneck model (6
chromosomes of 50 Mb, 50 replicates each, `-N25 -t15 -r5 -p 4+25*2+4+6`)
and compared each band with the spread of 50 independent fits of the
model, at 200 times of its informative window (`E5_summary.csv`):

- the bootstrap standard deviation of log10 Ne was a median 0.84 of the
  true sampling standard deviation with the default 50 Mb trunks (one per
  chromosome) and 0.93 with 5 Mb trunks, below it at 84% and 67% of the
  times;
- the 95% bands held the true Ne at only 37.2% (50 Mb trunks) and 43.0%
  (5 Mb) of the times, averaged over the genomes, while they held the mean
  of the 50 fits at 82.7% and 91.8%;
- most of the shortfall is PSMC's bias, which a bootstrap cannot see: its
  median, 0.061 in log10 Ne, is 2.2 times the true standard deviation,
  0.028, at the bottleneck and the size changes above all. Part of that
  bias is the pattern's, whose steps cannot follow the truth within a free
  λ's span: the bands held the truth averaged over each span at 44.4% and
  50.5% of the times.

A smaller version of the check,
`cargo run --release --example bootstrap_calibration -- --threads 4`
(`examples/bootstrap_calibration.rs`: 40 sequences of 10,000 bins of a
`4+5*3+4` model), runs in about 3.5 minutes on 4 threads of an Apple M1
Pro and peaks at 36 MB.

## Speed and memory

Experiment E3 timed the programs under a controlled protocol on AMD EPYC
7702 nodes of KAUST's Ibex cluster: each run alone on an exclusive node, 5
repeats on four nodes, 25 EM iterations of
`-N25 -t15 -r5 -p "4+25*2+4+6"` on simulated genomes of 100 Mb chromosomes,
C psmc built by gcc with `-O3 -march=x86-64-v3`.

- On 3000 Mb (30 chromosomes) psmc-rs took a median 317 s on 1 thread,
  22.5 s on 16 and 11.5 s on 30, against 4,456 s for C psmc: 14.1, 198 and
  386 times as fast (13.9 to 14.2 times on 1 thread at every size).
- It peaked at 36 MB of resident memory on 1 thread and 148 MB on 30, C
  psmc at 1,062 MB.
- 25 EM iterations on one simulated 100 Mb chromosome (1 M bins) took a
  median 10.3 to 10.5 s of CPU time and 10.46 to 10.97 s of wall time on 1,
  8 or 30 threads: a single sequence runs on one thread, and the others add
  no CPU time.

Every size and thread count, the ranges over the repeats, the thread
scaling and the outputs' determinism are in
[Supplementary Note 1, section 10](../experiment/Supplementary_Note_1_Reproducibility.md#10-performance-e3).
Laptop measurements are in [performance.md](performance.md#measured-on-a-laptop).

## Across platforms

The production's psmc-rs fits of E1 and NA12878 were made again on a
laptop (Apple M1 Pro, NEON kernels, macOS) from the same sources and
compared with Ibex's (x86-64, AVX2 and FMA kernels, Linux)
([experiment/crossplatform/summary.md](../experiment/crossplatform/summary.md)):

- none of the 41 outputs is byte-identical: the kernels and the system math
  libraries differ;
- the differences are in the last digits: over the 40 E1 fits, a median
  (max) 1.2e-7 (1.9e-6) relative in θ and 8.8e-8 (2.4e-6) in the window
  RMSE of log10 Ne;
- psmc-rs's cross-platform difference is a median 383 times smaller than
  its difference from C psmc, and 124 times smaller than C psmc's own
  difference between a laptop build and Ibex's;
- built by Ibex's rustc (1.99.0) instead of 1.97.1, the laptop's outputs are
  byte-identical (41/41), and one binary run twice gives identical bytes.

On x86-64 CPUs with AVX2 and FMA and the same math library, the outputs
are identical to the bit ([performance.md](performance.md#the-cpu-kernels)).
