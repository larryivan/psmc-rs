# Simulated bottleneck genomes

One diploid chromosome each, in 100-bp bins, simulated with the experiment
pipeline's own simulator:

```python
from experiment.pipeline import config, simulate
chrom = simulate.simulate_chromosome(
    config.MODELS["bottleneck"], config.SIMULATION, "chr1", LENGTH,
    {"ancestry": SEED + 1, "mutations": SEED + 2},
)
record = simulate.psmcfa_record("chr1", simulate.psmcfa_bins(chrom, 100))
```

That is msprime's Hudson coalescent with the `bottleneck` model of
`experiment/pipeline/config.py`, μ = 2.5e-8 and r = 1e-8 per bp per
generation, then `gzip -9 -n`.

| file | LENGTH | SEED | bins (`K`) | MD5 of the uncompressed file |
| --- | --- | --- | --- | --- |
| `bottleneck_5mb.psmcfa.gz` | 5,000,000 | 17000 | 50,000 (2,271) | `820a743f8b9fbeda28ebc44208e1897b` |
| `bottleneck_2mb.psmcfa.gz` | 2,000,000 | 23000 | 20,000 (1,065) | `c757f93eebf6993ebc637c811858fcb1` |

Both are fitted with one λ per interval (`--n-steps 63`, no pattern), where
earlier builds of the converged M-step went wrong:
`tests/em_valid_hmm.rs` fits the 5 Mb one, on which an M-step stepped into
parameters whose transition matrix in f64 is not stochastic (with λ up to
1e6, most M-steps end in a failed line search, while EM still improves
every round); `tests/mstep_coordinates.rs` runs the first M-step on the
2 Mb one, which stopped with its first λ far past the peak of Q along it
(48.8, where the M-step ends).
