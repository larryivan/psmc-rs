#!/usr/bin/env python3
"""Simulate a diploid genome with msprime and write a PSMCFA file.

This script is designed for simulated data (no missing sites). It encodes
windows as:
- 'K' if at least one heterozygous site in the window
- 'T' otherwise
- 'N' is not used unless --allow-missing is set (not implemented)
"""

import argparse
import gzip
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Simulate msprime and output PSMCFA")
    p.add_argument("--out", required=True, help="Output .psmcfa or .psmcfa.gz")
    p.add_argument("--length", type=float, default=500_000_000, help="Sequence length (bp)")
    p.add_argument("--ne", type=float, default=10_000, help="Effective population size")
    p.add_argument("--recomb", type=float, default=1e-8, help="Recombination rate per bp")
    p.add_argument("--mutation", type=float, default=2.5e-8, help="Mutation rate per bp")
    p.add_argument("--samples", type=int, default=2, help="Number of haploid samples")
    p.add_argument("--window", type=int, default=100, help="Window size (bp)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--model",
        choices=["constant", "zigzag"],
        default="constant",
        help="Demography model (constant or zigzag; zigzag defaults to Li & Durbin sim-2 style)",
    )
    p.add_argument(
        "--zigzag-sizes",
        default=None,
        help="Comma-separated Ne sizes for zigzag (len = times+1)",
    )
    p.add_argument(
        "--zigzag-times",
        default=None,
        help="Comma-separated change times (generations) for zigzag",
    )
    return p.parse_args()


def write_psmcfa(path: Path, seq: str, header: str = "chr1") -> None:
    lines = [f"> {header}\n"]
    wrap = 60
    for i in range(0, len(seq), wrap):
        lines.append(seq[i : i + wrap] + "\n")
    data = "".join(lines)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as f:
            f.write(data)
    else:
        path.write_text(data)

def parse_csv_floats(s: str):
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    return items


def main() -> int:
    args = parse_args()
    try:
        import msprime  # type: ignore
    except Exception as e:
        print("msprime is required. Install with: pip install msprime tskit", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    # Simulate ancestry and mutations
    if args.model == "constant":
        ts = msprime.sim_ancestry(
            samples=args.samples,
            ploidy=1,
            sequence_length=args.length,
            recombination_rate=args.recomb,
            population_size=args.ne,
            random_seed=args.seed,
        )
    else:
        # times as coalescent units (4N0) and scale by Ne to generations.
        if args.zigzag_sizes is None and args.zigzag_times is None:
            base_ne = float(args.ne)
            times_coal = [0.1, 0.6, 2.0, 10.0, 20.0]
            size_mult = [1.0, 5.0, 20.0, 5.0, 10.0, 5.0]
            times = [t * 4.0 * base_ne for t in times_coal]
            sizes = [m * base_ne for m in size_mult]
        else:
            sizes = parse_csv_floats(args.zigzag_sizes) if args.zigzag_sizes else None
            times = parse_csv_floats(args.zigzag_times) if args.zigzag_times else None
            if sizes is None or times is None:
                print(
                    "zigzag requires both --zigzag-sizes and --zigzag-times when custom.",
                    file=sys.stderr,
                )
                return 1
            if len(sizes) != len(times) + 1:
                print(
                    "zigzag sizes must be one longer than times: "
                    f"len(sizes)={len(sizes)}, len(times)={len(times)}",
                    file=sys.stderr,
                )
                return 1

        demography = msprime.Demography()
        demography.add_population(name="pop0", initial_size=sizes[0])
        for t, size in zip(times, sizes[1:]):
            demography.add_population_parameters_change(
                time=t, initial_size=size, population="pop0"
            )
        ts = msprime.sim_ancestry(
            samples=args.samples,
            ploidy=1,
            sequence_length=args.length,
            recombination_rate=args.recomb,
            demography=demography,
            random_seed=args.seed,
        )
    ts = msprime.sim_mutations(ts, rate=args.mutation, random_seed=args.seed + 1)

    if ts.num_samples < 2:
        print("Need at least 2 haploid samples to form a diploid.", file=sys.stderr)
        return 1

    # Use first two samples as one diploid individual
    sample_nodes = ts.samples()[:2]
    ts = ts.simplify(samples=sample_nodes)

    window = args.window
    n_windows = int(ts.sequence_length) // window
    seq = ["T"] * n_windows

    for var in ts.variants():
        pos = int(var.site.position)
        if pos >= n_windows * window:
            continue
        w = pos // window
        g = var.genotypes
        if g[0] != g[1]:
            seq[w] = "K"

    out_path = Path(args.out)
    write_psmcfa(out_path, "".join(seq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
