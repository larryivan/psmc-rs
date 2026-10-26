# The production binary on other x86-64 CPUs

`xcpu.sbatch` refits E1's bottleneck replicate 1 with the production's psmc-rs binary (sha256 1d141b81…, `src`
dcf29ca, built on an AMD EPYC 7702 node) and its exact command, on a node of another CPU model, and compares the
bytes of the `.json` and `.psmc` it writes with the committed fit (`experiment/runs/E1/fits/bottleneck/rep01.rs.*`).
Run on Ibex on 2026-10-26 (one shared job per CPU model, glibc 2.34 on every node):

| job | CPU | kernels | result |
| --- | --- | --- | --- |
| 54604495 | AMD EPYC 9655 (Zen 5) | AVX2 and FMA | identical |
| 54604735 | Intel Xeon Gold 6248 (Cascade Lake) | AVX2 and FMA | identical |
| 54604496, 54604497 | Intel Xeon Gold 6148 (Skylake) | AVX2 and FMA | identical |

The `xcpu-*.out` files are the jobs' logs. Their wall times are not timings: the nodes were shared.
