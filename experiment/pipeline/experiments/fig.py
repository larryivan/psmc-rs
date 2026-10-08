"""FIG: the manuscript's figures as runner jobs.

One job per figure of :data:`..figures.FIGURES`: its inputs are the tables
the figure reads (so it runs after the experiment's ``report`` job, and again
whenever a table changes), its outputs ``<runs>/<EXP>/figures/<name>.{pdf,
svg,png}``. A figure is declared once every experiment it reads is
implemented (registered in :data:`.REGISTRY`) and, for E7, its data stage
has finished; Fig 1, a schematic, needs none. The provenance key holds the figure's code (style, data, schemas and
its module), the figure settings (:data:`..config.FIGURES`), the profile's
settings of the experiments it reads, the package versions and the font
file the text is set in.

``report FIG`` draws every figure whose tables exist, outside the runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .. import config, figures, paths, provenance
from ..runner import Job
from . import REGISTRY, Context, module

NAME = "FIG"
PACKAGES = ("matplotlib", "numpy", "pandas")


def font_identity() -> Dict[str, str]:
    """The font file the figures' text is set in (its name and SHA-256)."""
    from matplotlib import font_manager

    path = Path(font_manager.findfont(font_manager.FontProperties(family=list(figures.style.FONT_FAMILY))))
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
    return {"file": path.name, "sha256": digest}


def declared(fs: figures.FigureSpec, ctx: Context) -> bool:
    """Whether the figure can be drawn: every experiment it reads is implemented, and ready if it says when.

    An experiment whose inputs come from outside the runner (E7's data
    stage) has a ``ready(ctx)``; until then it has no jobs, and its figures
    none either (they would wait for tables no job writes), unless the
    tables the figure reads are there already (committed runs, copied
    without the data stage's inputs).
    """
    tables_there = not figures.missing_tables(fs, ctx.runs_dir)
    for exp in fs.sources:
        if exp not in REGISTRY:
            return False
        ready = getattr(module(exp), "ready", None)
        if ready is not None and not ready(ctx) and not tables_there:
            return False
    return True



def job(ctx: Context, fs: figures.FigureSpec, fonts: Dict[str, str]) -> Job:
    figs = config.FIGURES
    return Job(
        id=f"{NAME}/{fs.name}",
        command=[paths.python(), "-m", "experiment.pipeline.figures", fs.name, *ctx.profile_args()],
        inputs=fs.table_paths(ctx.runs_dir),
        outputs=fs.outputs(ctx.runs_dir),
        cores=1,
        mem_gb=figs.job_gb,
        est_seconds=figs.job_seconds,
        settings={
            "figure": fs.name,
            "figures": asdict(figs),
            "experiments": {exp: asdict(getattr(ctx.profile, exp.lower())) for exp in fs.sources},
        },
        tools={"pipeline": provenance.python_identity(fs.code_files(), PACKAGES), "font": fonts},
        description=fs.title,
    )


def jobs(ctx: Context) -> List[Job]:
    fonts = font_identity()
    return [job(ctx, fs, fonts) for fs in figures.FIGURES if declared(fs, ctx)]


def report(ctx: Context) -> Dict[str, Path]:
    """Draw every figure whose tables exist; returns the PDF of each one drawn."""
    written: Dict[str, Path] = {}
    for fs in figures.FIGURES:
        missing = figures.missing_tables(fs, ctx.runs_dir)
        if missing:
            ctx.log(f"skip {fs.name}: missing {', '.join(str(p) for p in missing)}")
            continue
        written[fs.name] = figures.render(fs, ctx.runs_dir, ctx.profile)[0]
    return written


__all__ = ["NAME", "declared", "font_identity", "jobs", "report"]
