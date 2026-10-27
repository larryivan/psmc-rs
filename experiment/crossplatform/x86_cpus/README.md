# The production binary on other x86-64 CPUs

`xcpu.sbatch` refits E1's bottleneck replicate 1 with the production's psmc-rs binary (sha256 1d141b81…, `src`
dcf29ca, built on an AMD EPYC 7702 node) and its exact command, on a node of another CPU model, and compares the
bytes of the `.json` and `.psmc` it writes with the committed fit (`experiment/runs/E1/fits/bottleneck/rep01.rs.*`).
Run on Ibex on 2026-10-26 (four shared jobs on three CPU models, glibc 2.34 on every node):

| job | CPU | kernels | result |
| --- | --- | --- | --- |
| 54604495 | AMD EPYC 9655 (Zen 5) | AVX2 and FMA | identical |
| 54604735 | Intel Xeon Gold 6248 (Cascade Lake) | AVX2 and FMA | identical |
| 54604496, 54604497 | Intel Xeon Gold 6148 (Skylake) | AVX2 and FMA | identical |

The `xcpu-*.out` files are the jobs' logs. Their wall times are not timings: the nodes were shared. The jobs were
submitted as below, with `D` the directory holding `xcpu.sbatch`, so each log is named after the node feature its
job asked for. The `cascadelake` request (54604496) got a Gold 6148 node, like the `skylake` one (54604497), so
54604735 asked for a Gold 6248 by its own feature.

```bash
for c in turin cascadelake skylake; do sbatch -A <account> -p batch --constraint=$c --cpus-per-task=2 --mem=8G --time=00:30:00 -J xcpu-$c -o $D/xcpu-$c-%j.out --parsable $D/xcpu.sbatch; done
sbatch -A <account> -p batch --constraint=cpu_intel_gold_6248 --cpus-per-task=2 --mem=8G --time=00:30:00 -J xcpu-6248 -o $D/xcpu-6248-%j.out --parsable $D/xcpu.sbatch
```

`P` (line 4 of `xcpu.sbatch`) is the production worktree on Ibex, at 4b2c636f when these jobs ran (`PROD` of
`experiment/slurm/site.env.example`). Elsewhere, set it to a checkout whose `target/release/psmc-rs` is the binary
to test and whose `experiment/runs` holds E1's simulated genomes: the script reads
`experiment/runs/E1/sims/bottleneck/rep01/genome.psmcfa`, which is not committed, so E1's simulation of that
replicate must have run there.

## C psmc's production build

The production's C psmc (gcc 11.4.1, `-g -Wall -O3 -march=x86-64-v3 -ffp-contract=off -Wno-unused-function`, as its
provenance records hold) was compared with the Makefile's `-O2` on Ibex on 2026-10-06, in three shared jobs, one per
CPU model. Each job built C psmc from the same sources (lh3/psmc's, `experiment/.tools/psmc_c`) with gcc 11.4.1 three
ways, `make CFLAGS="-g -Wall <flags> -Wno-unused-function"` with `-O2`, `-O3 -march=x86-64-v3 -ffp-contract=off` and
`-O3 -march=native -ffp-contract=off`, and ran each build once on HLpelCas1
(`experiment/real_data/HLpelCas1_diploid.psmcfa`, 1.9 Gb) with E2's C psmc command for it but 3 EM iterations,
`psmc -N3 -t15 -r5 -p "4+25*2+4+6" -o <out>.psmc HLpelCas1_diploid.psmcfa`, in an allocation of 16 CPUs. On all three
nodes both `-O3` builds wrote a `.psmc` identical (`cmp`) to the `-O2` build's:

| job | node | CPU | `-O2` wall | `-O3 -march=x86-64-v3` wall | speed-up |
| --- | --- | --- | --- | --- | --- |
| 53382580 | cn514-15-r | AMD EPYC 7702 (Zen 2) | 608.74 s | 397.83 s | 1.53x |
| 53382581 | cn511-09 | AMD EPYC 9655 (Zen 5) | 319.68 s | 163.36 s | 1.96x |
| 53382582 | cn512-23-l | Intel Xeon Gold 6248 (Cascade Lake) | 606.77 s | 510.17 s | 1.19x |

The wall times are of one run each on a shared node.
