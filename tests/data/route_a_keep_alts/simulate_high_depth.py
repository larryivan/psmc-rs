#!/usr/bin/env python3
"""Simulate one deeply covered chromosome for the keep-alts fixtures (see
README.md).

This is ../route_a/simulate.py (same seed, variants, sequencing errors, base
and mapping qualities) with a single chromosome of 3,000 bp at 250x, deep
enough that `bcftools call -c -A` lists an ALT allele at most reference
sites. Writes ref.fa, aln.sam and truth.tsv into the output directory.
Standard library only.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "route_a")
)
import simulate  # noqa: E402

simulate.CHROMS = [("chr1", 3000, [(1, 3000, 250, 60)])]

if __name__ == "__main__":
    simulate.main()
