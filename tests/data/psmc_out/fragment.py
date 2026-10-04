#!/usr/bin/env python3
"""Writes fragmented.psmcfa (see README.md): the first 20,000 bins of seg1
and of seg2 of ../c_reference/papuan_chr1_3seg.psmcfa, cut into records of
1 to 80 bins, as in a highly fragmented assembly.

Record lengths come from a fixed linear congruential generator, so the
output does not depend on the Python version. Standard library only.
"""

import os

BINS_PER_SEGMENT = 20_000
SEGMENTS = ("seg1", "seg2")
LINE_WIDTH = 60


def read_psmcfa(path):
    records, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                records[name] = []
            elif line:
                records[name].append(line)
    return {name: "".join(parts) for name, parts in records.items()}


def lengths(seed=20261005):
    x = seed
    while True:
        x = (1103515245 * x + 12345) % 2**31
        yield 1 + (x >> 16) % 80


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    seqs = read_psmcfa(os.path.join(here, "../c_reference/papuan_chr1_3seg.psmcfa"))
    out, n_records, length = [], 0, lengths()
    for seg in SEGMENTS:
        seq = seqs[seg][:BINS_PER_SEGMENT]
        start = 0
        while start < len(seq):
            piece = seq[start : start + next(length)]
            start += len(piece)
            n_records += 1
            out.append(f">{seg}_{n_records}\n")
            for i in range(0, len(piece), LINE_WIDTH):
                out.append(piece[i : i + LINE_WIDTH] + "\n")
    with open(os.path.join(here, "fragmented.psmcfa"), "w") as f:
        f.writelines(out)


if __name__ == "__main__":
    main()
