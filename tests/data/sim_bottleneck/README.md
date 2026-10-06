# Simulated bottleneck genome

`bottleneck_5mb.psmcfa.gz`: one diploid chromosome of 5,000,000 bp in 100-bp
bins (50,000 bins, 2,271 of them `K`), simulated with the experiment
pipeline's own simulator:

```python
from experiment.pipeline import config, simulate
chrom = simulate.simulate_chromosome(
    config.MODELS["bottleneck"], config.SIMULATION, "chr1", 5_000_000,
    {"ancestry": 17001, "mutations": 17002},
)
record = simulate.psmcfa_record("chr1", simulate.psmcfa_bins(chrom, 100))
```

That is msprime's Hudson coalescent with the `bottleneck` model of
`experiment/pipeline/config.py`, μ = 2.5e-8 and r = 1e-8 per bp per
generation, then `gzip -9 -n`. The uncompressed file's MD5 is
`820a743f8b9fbeda28ebc44208e1897b`.

`tests/em_valid_hmm.rs` fits it with one λ per interval (`--n-steps 63`, no
pattern), where an earlier build of the converged M-step stepped into
parameters whose transition matrix in f64 is not stochastic.
