#!/usr/bin/env python3
"""Simulate diploid data with stdpopsim and write a PSMCFA file.

This script encodes windows as:
- 'K' if at least one heterozygous site in the window
- 'T' otherwise
"""

import argparse
import gzip
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Simulate with stdpopsim and output PSMCFA")
    p.add_argument("--out", required=True, help="Output .psmcfa or .psmcfa.gz")
    p.add_argument("--length", type=float, default=500_000_000, help="Sequence length (bp)")
    p.add_argument("--species", default="HomSap", help="stdpopsim species ID (default: HomSap)")
    p.add_argument(
        "--population",
        default=None,
        help="Population ID in the selected model; auto-selected if model has one population",
    )
    p.add_argument(
        "--demographic-model-id",
        default=None,
        help="Override demographic model ID (used by --model zigzag/zigzag_1s14)",
    )
    p.add_argument(
        "--ne",
        type=float,
        default=None,
        help="Base effective size for preset models (model-specific default if omitted)",
    )
    p.add_argument(
        "--recomb",
        type=float,
        default=None,
        help="Override recombination rate per bp (default uses stdpopsim catalog rate)",
    )
    p.add_argument(
        "--mutation",
        type=float,
        default=None,
        help="Override mutation rate per bp (default uses stdpopsim catalog rate)",
    )
    p.add_argument("--diploids", type=int, default=1, help="Number of diploid individuals to simulate")
    p.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Deprecated: number of haploid samples; converted to diploids via ceil(samples/2)",
    )
    p.add_argument("--window", type=int, default=100, help="Window size (bp)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--model",
        choices=[
            "constant",
            "zigzag",
            "zigzag_1s14",
            "sim2_zigzag",
            "bottleneck",
            "expansion",
            "piecewise",
        ],
        default="constant",
        help="Demography model preset: constant / bottleneck / expansion / sim2_zigzag "
        "/ zigzag_1s14 (catalog) / piecewise(custom)",
    )
    p.add_argument(
        "--piecewise-events",
        default=None,
        help="For --model piecewise: comma-separated `time_gen:size` pairs, e.g. "
        "`500:20000,2000:5000,10000:15000`",
    )
    p.add_argument(
        "--print-model",
        action="store_true",
        help="Print chosen model schedule before simulation",
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


def resolve_population_name(model, population_arg: Optional[str]) -> str:
    pops = []
    for pop in getattr(model, "populations", []):
        name = getattr(pop, "name", None)
        if name is None:
            name = getattr(pop, "id", None)
        if name is not None:
            pops.append(str(name))

    if population_arg is not None:
        if pops and population_arg not in pops:
            raise ValueError(
                f"Population '{population_arg}' not found in model populations: {', '.join(pops)}"
            )
        return population_arg

    if len(pops) == 1:
        return pops[0]

    if "generic" in pops:
        return "generic"
    if "pop_0" in pops:
        return "pop_0"

    if pops:
        raise ValueError(
            "Model has multiple populations. Please set --population. "
            f"Available: {', '.join(pops)}"
        )
    return "pop_0"


def select_first_diploid_nodes(ts):
    if ts.num_individuals > 0:
        for ind in ts.individuals():
            nodes = [n for n in ind.nodes if ts.node(n).is_sample()]
            if len(nodes) >= 2:
                return nodes[:2]
    sample_nodes = list(ts.samples())
    if len(sample_nodes) >= 2:
        return sample_nodes[:2]
    return []


def parse_piecewise_events(spec: str) -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid event '{part}', expected time:size")
        t_str, n_str = part.split(":", 1)
        t = float(t_str)
        n = float(n_str)
        if t <= 0:
            raise ValueError(f"invalid event time '{t}', must be > 0")
        if n <= 0:
            raise ValueError(f"invalid event size '{n}', must be > 0")
        events.append((t, n))
    if not events:
        raise ValueError("no valid piecewise events parsed")
    events.sort(key=lambda x: x[0])
    return events


def base_ne_for_model(model: str, ne_arg: Optional[float]) -> float:
    if ne_arg is not None:
        if ne_arg <= 0:
            raise ValueError("--ne must be > 0")
        return float(ne_arg)
    # Defaults chosen to mirror common PSMC simulation scales.
    defaults = {
        "constant": 10_000.0,
        "expansion": 10_000.0,
        "bottleneck": 20_000.0,
        "sim2_zigzag": 1_000.0,
        "piecewise": 10_000.0,
    }
    return defaults.get(model, 10_000.0)


def coalescent_events_to_generations(n0: float, events: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # Convert ms-style (time in 4N0 units, size as ratio to N0) to
    # generation-time piecewise schedule used by stdpopsim generic model.
    out: List[Tuple[float, float]] = []
    for t_4n0, ratio in events:
        out.append((t_4n0 * 4.0 * n0, ratio * n0))
    return out


def preset_piecewise_events(model: str, n0: float) -> List[Tuple[float, float]]:
    if model == "sim2_zigzag":
        events_4n0 = [
            (0.1, 5.0),
            (0.6, 20.0),
            (2.0, 5.0),
            (10.0, 10.0),
            (20.0, 5.0),
        ]
        return coalescent_events_to_generations(n0, events_4n0)
    if model == "bottleneck":
        events_4n0 = [
            (0.01, 0.05),
            (0.015, 0.5),
            (0.05, 0.25),
            (0.5, 0.5),
        ]
        return coalescent_events_to_generations(n0, events_4n0)
    if model == "expansion":
        events_4n0 = [
            (0.01, 0.1),
            (0.06, 1.0),
            (0.2, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
        ]
        return coalescent_events_to_generations(n0, events_4n0)
    raise ValueError(f"no preset piecewise events for model={model}")


def build_piecewise_model(stdpopsim, n0: float, events: List[Tuple[float, float]]):
    # stdpopsim.PiecewiseConstantSize signature is N0, *events where each event is (time, size).
    # Keep compatibility with API differences by trying a fallback call form.
    try:
        return stdpopsim.PiecewiseConstantSize(n0, *events)
    except TypeError:
        return stdpopsim.PiecewiseConstantSize(n0, events)


def describe_schedule(initial_ne: float, events: List[Tuple[float, float]]) -> str:
    chunks = [f"t=0 -> Ne={initial_ne:g}"]
    for t, n in events:
        chunks.append(f"t={t:g} gen -> Ne={n:g}")
    return "; ".join(chunks)


def main() -> int:
    args = parse_args()
    try:
        import stdpopsim  # type: ignore
    except Exception as e:
        print("stdpopsim is required. Install with: pip install stdpopsim msprime tskit", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 1

    if args.window <= 0:
        print("--window must be positive.", file=sys.stderr)
        return 1
    if args.length <= 0:
        print("--length must be positive.", file=sys.stderr)
        return 1

    try:
        species = stdpopsim.get_species(args.species)

        if args.model in ("zigzag", "zigzag_1s14"):
            model_id = args.demographic_model_id or "Zigzag_1S14"
            model = species.get_demographic_model(model_id)
            if args.print_model:
                print(f"[model] catalog={model_id}")
        elif args.model == "constant":
            n0 = base_ne_for_model("constant", args.ne)
            events: List[Tuple[float, float]] = []
            model = build_piecewise_model(stdpopsim, n0, events)
            if args.print_model:
                print("[model] constant")
                print("[schedule]", describe_schedule(n0, events))
        elif args.model in ("sim2_zigzag", "bottleneck", "expansion"):
            n0 = base_ne_for_model(args.model, args.ne)
            events = preset_piecewise_events(args.model, n0)
            model = build_piecewise_model(stdpopsim, n0, events)
            if args.print_model:
                print(f"[model] preset={args.model}")
                print("[schedule]", describe_schedule(n0, events))
        else:
            # custom piecewise schedule
            n0 = base_ne_for_model("piecewise", args.ne)
            if args.piecewise_events is None:
                raise ValueError("model=piecewise requires --piecewise-events")
            events = parse_piecewise_events(args.piecewise_events)
            model = build_piecewise_model(stdpopsim, n0, events)
            if args.print_model:
                print("[model] custom piecewise")
                print("[schedule]", describe_schedule(n0, events))

        contig_kwargs = {"length": args.length}
        if args.mutation is not None:
            contig_kwargs["mutation_rate"] = args.mutation
        if args.recomb is not None:
            contig_kwargs["recombination_rate"] = args.recomb
        contig = species.get_contig(**contig_kwargs)

        diploids = args.diploids
        if args.samples is not None:
            diploids = max(1, math.ceil(args.samples / 2))

        pop_name = resolve_population_name(model, args.population)
        samples = {pop_name: diploids}
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples, seed=args.seed)
    except Exception as e:
        print(f"Simulation setup failed: {e}", file=sys.stderr)
        return 1

    if ts.num_samples < 2:
        print("Need at least 2 haploid samples to form a diploid.", file=sys.stderr)
        return 1

    sample_nodes = select_first_diploid_nodes(ts)
    if len(sample_nodes) < 2:
        print("Could not identify two haploid nodes for a diploid individual.", file=sys.stderr)
        return 1
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
