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

`P` (line 4 of `xcpu.sbatch`) is the production worktree on Ibex, at 4b2c636f (`PROD` of
`experiment/slurm/site.env.example`). Elsewhere, set it to a checkout whose `target/release/psmc-rs` is the binary
to test and whose `experiment/runs` holds E1's simulated genomes: the script reads
`experiment/runs/E1/sims/bottleneck/rep01/genome.psmcfa`, which is not committed, so E1's simulation of that
replicate must have run there.
