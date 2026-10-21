"""Draw the manuscript's figures: ``python -m experiment.pipeline.figures [NAME ...]``.

Without names, every figure whose tables exist is drawn and the others are
listed with the tables they lack. NAME is a figure's file stem or its prefix
(``fig2``, ``figS1``). ``--profile`` and ``--runs-dir`` are those of the
pipeline (default: production, ``experiment/runs``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .. import config, paths
from . import FIGURES, missing_tables, render, spec, style


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.figures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figures", nargs="*", metavar="NAME", help="figures to draw (default: all whose tables exist)")
    p.add_argument("--profile", default="production", choices=sorted(config.PROFILES),
                   help="the profile the tables were made with (default: production)")
    p.add_argument("--runs-dir", type=Path, help="runs directory (default: the profile's)")
    p.add_argument("--out-dir", type=Path, help="write here instead of <runs>/<experiment>/figures")
    p.add_argument("--dpi", type=int, default=style.PNG_DPI, help=f"PNG resolution (default {style.PNG_DPI})")
    p.add_argument("--list", action="store_true", help="list the figures and the tables they read")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runs_dir = (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()
    if args.list:
        for fs in FIGURES:
            tables = ", ".join(p.relative_to(runs_dir).as_posix() for p in fs.table_paths(runs_dir)) or "none"
            print(f"{fs.name}  ->  {fs.experiment}/figures/\n    {fs.title}\n    reads: {tables}")
        return 0
    try:
        wanted = [spec(n) for n in args.figures] if args.figures else list(FIGURES)
    except KeyError as exc:
        parser.error(exc.args[0])  # exits 2
    profile = config.get_profile(args.profile)
    status = 0
    for fs in wanted:
        missing = missing_tables(fs, runs_dir)
        if missing:
            print(f"skip {fs.name}: missing {', '.join(str(p) for p in missing)}", file=sys.stderr)
            status = 1 if args.figures else status
            continue
        for path in render(fs, runs_dir, profile, dpi=args.dpi, out_dir=args.out_dir):
            print(path)
    return status


if __name__ == "__main__":
    sys.exit(main())
