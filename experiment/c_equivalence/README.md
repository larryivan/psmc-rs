# C-equivalence diagnostic

Where the remaining difference between psmc-rs's and C psmc's fits of the
same input comes from (the manuscript's Results and Methods). The two
programs' EM differs in two places: the M-step's optimizer (L-BFGS within
bounds, against C psmc's Hooke-Jeeves search) and the last bin of each
sequence, which C psmc's expected counts leave out. `diag.patch` adds a
variant of each to the sources of psmc-rs 0.2.0 (`src/` tree dcf29ca, the
production's), switched on by environment variables; with none set, the
patched build writes the release's output byte for byte. The patch is never
applied to `src/`, and the variants are in no release.

| variant | set | what changes |
| --- | --- | --- |
| `lastbin` | `PSMC_DIAG_LASTBIN=1` | each sequence's last bin is left out of the E-step's expected emission counts, as C psmc's `hmm_expect` leaves it out (the expected transitions are unchanged, as in C psmc) |
| `hj` | `PSMC_DIAG_HJ=1` | the M-step is a statement-for-statement port of C psmc's search (`kmin_hj` of `kmin.c`, with its radius, tolerance and call limit), started and read as `psmc_em` does, on psmc-rs's objective, without bounds or row-sum test; not ported: the constant C psmc subtracts from the objective and the 1e-25 it adds to every count |
| `both` | both | both changes |
| `prec` | `PSMC_DIAG_REL_TOL=0`, `--mstep-iters 100000` | psmc-rs's own M-step, run to the limit of precision |

## Results

**E1's 40 genomes** (`e1_variants.csv`, one row per genome and variant;
`summary.json`): the RMSE of log10 Ne against C psmc's committed fit,
interval by interval over the genome's informative window, as
`E1_rs_vs_c.csv` computes it (the default rows reproduce it), next to the
difference between two builds of C psmc (the laptop's `-O1` clang build
and Ibex's `-O3` gcc build, `experiment/crossplatform/side_by_side_E1.csv`:
median 8.5e-6, 3.6e-6 to 2.4e-5; 2.2e-5 on bottleneck replicate 1):

| variant | median | min to max | closer to C psmc than the default | at or below the two C psmc builds' difference | bottleneck replicate 1 |
| --- | --- | --- | --- | --- | --- |
| default (the committed fits) | 2.5e-5 | 3.7e-6 to 2.0e-4 | – | 7 | 2.8e-5 |
| `lastbin` | 6.2e-6 | 2.6e-6 to 5.8e-5 | 37 | 27 | 1.6e-5 |
| `hj` | 2.6e-5 | 3.1e-6 to 2.0e-4 | 23 | 8 | 2.8e-5 |
| `both` | 6.2e-6 | 2.7e-6 to 4.3e-5 | 36 | 27 | 1.5e-5 |

- The optimizer is not what keeps the programs apart. The port's fits are
  within a median 2.0e-6 of psmc-rs's own (1.4e-6 on bottleneck replicate
  1) and closer to C psmc's in 23 of the 40 genomes; psmc-rs's default
  fits are within a median 1.0e-7 (at most 2.5e-6; 5.3e-8 on bottleneck
  replicate 1) of `prec`'s.
- The last bin accounts for most of the rest. Left out, it brings 37 of
  the 40 curves closer to C psmc's and the median from 2.5e-5 to 6.2e-6,
  below the two C psmc builds' median. By default, the other three
  (constant 3 and 9, zigzag 2) are already within the two builds'
  difference on the same genome.
- The port searches as C psmc does: 2,688 to 6,471 evaluations per M-step
  on E1 (mean 4,120), against C psmc's 2,815 to 6,738 (mean 4,016). Traced
  in the first M-step on the reference input (`trace_p23.txt`,
  `trace_p64.txt`, `trace_f23.txt`), run with the last bin left out too, as
  C psmc leaves it out, so that the two objectives agree
  (`scripts/run_trace.sh`), it evaluates C psmc's parameters bit for bit
  for C's first 1,197 of 1,540 evaluations (p23), 2,459 of 4,097 (p64) and
  4,854 of 7,002 (f23), the objective's change from the first evaluation
  agreeing within 5.2e-10 (of 57 to 60). The two searches part where they
  rank differently two points whose objectives differ by less than that
  (at most 1.3e-10), a rounding tie; each search then lowers the objective
  by a further 2.2e-9 (p23; the port 3.1e-9), 2.9e-5 (p64) and 2.2e-6
  (f23), and they end 4.8e-6 to 3.3e-5 apart in the parameters.

**The inputs of C psmc's reference outputs** (`ref_variants.json`;
`papuan_chr1_3seg.psmcfa`, 5 iterations, with `-p 4+5*3+4`, p23,
`-p 4+25*2+4+6`, p64, and 23 free lambdas, f23): the largest relative
difference from C psmc's log-likelihood over rounds 2 to 5 (`prec` gives
the default's values):

| variant | p23 | p64 | f23 |
| --- | --- | --- | --- |
| default | 6.8e-8 | 7.3e-8 | 7.1e-8 |
| `lastbin` | 2.8e-9 | 7.9e-10 | 8.3e-10 |
| `hj` | 6.8e-8 | 7.3e-8 | 7.0e-8 |
| `both` | 1.9e-9 | 8.7e-10 | 7.9e-10 |

On the 1,010-record `fragmented.psmcfa`, round 1's parameters differ from
C psmc's by up to a relative 7.4e-3 by default and 6.8e-6 with the last bin
left out.

## What ran

- `fits/` (E1): KAUST Ibex, SLURM job 54553237 on 25 October 2026, a
  shared AMD EPYC 7702 node, with the commands of `scripts/run_e1.py` on
  the production's genomes. The diagnostic build was made as
  `scripts/build.sh` makes it, from commit 4b2c636f (`src/` dcf29ca) with
  rustc 1.99.0 in the production's environment (`experiment/slurm/env.sh`);
  the binary's sha256 is
  cfcaf23f0600c87fe7657b874fe7efd39f1079d26385ba83ae2c4b869d9d952e.
  With no variant set it wrote bottleneck replicate 1's committed JSON and
  `.psmc` byte for byte, and the production's C psmc binary (sha256
  d5e0e160…) fitted that genome again to its committed `.psmc` byte for
  byte.
- `ref/` and the traces: the development laptop (Apple M1 Pro, macOS 27),
  the diagnostic build by rustc 1.97.1 (NEON kernels); C psmc with
  `c_trace.patch`, built with `cc -O1` as `tests/data/c_reference` was,
  writes that directory's `c_4+5x3+4_N5.psmc` byte for byte.

## Running it again

```bash
cd experiment/c_equivalence
BIN=$(scripts/build.sh HEAD $WORK)        # a commit whose src/ is dcf29ca; $WORK outside this checkout
PY=../.venv/bin/python
$PY -I -B scripts/run_e1.py $BIN $RUNS -P 4  # $RUNS holds E1/sims, each genome checked against its record
scripts/run_ref.sh $BIN
scripts/run_trace.sh $BIN $WORK            # after building C psmc with c_trace.patch, as the script says
$PY -I -B scripts/compare_e1.py              # e1_variants.csv
python3 -I -B scripts/compare_ref.py         # ref_variants.json
python3 -I -B scripts/summary.py             # summary.json, which manuscript/make_numbers.py reads
```

`$RUNS` is the production's runs directory, the laptop's identical copy
(`experiment/.production/experiment/runs`, not in the repository; see
`experiment/crossplatform`) or one where the pipeline simulated E1 again
(Supplementary Note 1, section 4). On an x86-64 CPU with AVX2 and FMA and
glibc 2.34 the build reproduces the production's bytes (the no-variant
check above); elsewhere the fits differ in their last digits
(Supplementary Note 1, sections 8 and 9). Not kept: the binaries, logs,
traces and the HTML reports psmc-rs writes beside each JSON.
