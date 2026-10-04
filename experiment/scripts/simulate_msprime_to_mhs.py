#!/usr/bin/env python3
"""Simulate a diploid constant-size genome with msprime and write multihetsep/mhs output.

Output format follows generate_multihetsep.py:
    chrom  pos  nr_called  alleles
Only segregating sites for the single diploid sample are emitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate constant model and output multihetsep/mhs")
    p.add_argument("--out", required=True, help="Output .mhs/.multihetsep path")
    p.add_argument("--length", type=int, default=5_000_000, help="Sequence length (bp)")
    p.add_argument("--ne", type=float, default=10_000.0, help="Constant effective population size")
    p.add_argument(
        "--recomb",
        type=float,
        default=1.25e-8,
        help="Recombination rate per bp per generation",
    )
    p.add_argument(
        "--mutation",
        type=float,
        default=2.5e-8,
        help="Mutation rate per bp per generation",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--chrom", default="chr1", help="Chromosome label in output")
    return p.parse_args()


def main() -> int:
    try:
        import msprime  # type: ignore
    except ImportError:
        raise SystemExit("msprime is required. Please install it first.")

    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = msprime.sim_ancestry(
        samples=1,
        ploidy=2,
        sequence_length=float(args.length),
        recombination_rate=float(args.recomb),
        population_size=float(args.ne),
        random_seed=int(args.seed),
    )
    mts = msprime.sim_mutations(
        ts,
        rate=float(args.mutation),
        random_seed=int(args.seed) + 1,
    )

    prev_emitted_pos = 0
    n_rows = 0
    with out.open("w", encoding="utf-8") as f:
        for var in mts.variants():
            # 1-based genomic coordinate.
            pos = int(var.site.position) + 1
            if pos <= prev_emitted_pos:
                # Should not happen for well-formed discrete-genome simulations.
                continue

            g0 = int(var.genotypes[0])
            g1 = int(var.genotypes[1])
            a0 = var.alleles[g0]
            a1 = var.alleles[g1]
            if a0 == a1:
                # Not segregating in the diploid sample, skip.
                continue

            nr_called = pos - prev_emitted_pos
            if nr_called <= 0:
                continue

            alleles = f"{a0}{a1}"
            f.write(f"{args.chrom}\t{pos}\t{nr_called}\t{alleles}\n")
            prev_emitted_pos = pos
            n_rows += 1

    print(f"[ok] wrote {out} with {n_rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
