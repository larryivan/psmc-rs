#!/usr/bin/env python3
"""Write synthetic.psmcfa, the multi-record input of the QC reference test.

Deterministic (a 64-bit LCG, not Python's `random`), so the file is the same
on every Python 3 version. Records differ in length, heterozygosity, missing
runs, case and name style; see generate.sh for how the C reference values are
produced from it.
"""

import sys

STATE = 0x2545F4914F6CDD1D


def uniform():
    """Next value in [0, 1) of Knuth's MMIX LCG (top 53 bits)."""
    global STATE
    STATE = (STATE * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (STATE >> 11) / float(1 << 53)


def record(n_bins, p_het, missing_runs=(), lowercase_from=None):
    out = []
    for i in range(n_bins):
        if any(start <= i < start + length for start, length in missing_runs):
            sym = "N"
        else:
            sym = "K" if uniform() < p_het else "T"
        if lowercase_from is not None and i >= lowercase_from:
            sym = sym.lower()
        out.append(sym)
    return "".join(out)


RECORDS = [
    # name line, bins, het fraction, N runs (start, length), lowercase from
    ("chr1 first autosome", 25000, 0.06, [(10000, 3000), (24990, 10)], None),
    ("chr2", 12345, 0.03, [(0, 7)], 6000),
    ("chrX hemizygous-like", 8000, 0.003, [(4000, 1500)], None),
    ("chrM", 166, 0.0, [], None),
    ("scaffold_7", 999, 0.1, [(100, 50), (300, 1), (600, 120)], None),
    ("KTN_unplaced_12 header letters must not count", 500, 0.05, [], 250),
]


def main(path):
    with open(path, "w") as fh:
        for name, n_bins, p_het, runs, lower in RECORDS:
            seq = record(n_bins, p_het, runs, lower)
            fh.write(">" + name + "\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic.psmcfa")
