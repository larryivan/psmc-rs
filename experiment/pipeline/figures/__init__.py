"""The manuscript's figures, drawn from the experiments' committed tables.

Every figure is a module of this package with ``draw(source) -> Figure``
(``source`` is a :class:`.data.Source`); :data:`FIGURES` lists them with the
tables they read and the experiment directory they are written to:
``<runs>/<experiment>/figures/<name>.{pdf,svg,png}``. :func:`render` draws
one and saves it with :func:`.style.save`; ``python -m
experiment.pipeline.figures`` draws them.

:mod:`.style` holds the shared style (type, sizes, colours, export),
:mod:`.schemas` the tables the figures read, :mod:`.data` their readers and
the curve arithmetic, and :mod:`.fake` small stand-in tables in those
schemas for developing and testing the figures.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config
from . import style
from .schemas import SCHEMAS


@dataclass(frozen=True)
class FigureSpec:
    name: str  # file stem
    module: str  # module of this package with draw(source)
    experiment: str  # written to <runs>/<experiment>/figures/
    tables: Tuple[str, ...]  # keys of schemas.SCHEMAS
    title: str  # what the figure shows (the caption carries it)

    @property
    def sources(self) -> Tuple[str, ...]:
        """The experiments whose tables the figure reads."""
        out: List[str] = []
        for key in self.tables:
            exp = SCHEMAS[key].experiment
            if exp not in out:
                out.append(exp)
        return tuple(out)

    def table_paths(self, runs_dir: Path) -> List[Path]:
        return [SCHEMAS[k].path(runs_dir) for k in self.tables]

    def outputs(self, runs_dir: Path) -> List[Path]:
        d = Path(runs_dir) / self.experiment / "figures"
        return [d / f"{self.name}.{ext}" for ext in style.FORMATS_OUT]

    def code_files(self) -> List[str]:
        """The pipeline files (relative to experiment/pipeline) that draw it."""
        return sorted({"config.py", "results.py", "metrics.py", "demography.py", "figures/__init__.py",
                       "figures/style.py", "figures/data.py", "figures/schemas.py", f"figures/{self.module}.py"})


FIGURES: Tuple[FigureSpec, ...] = (
    FigureSpec("fig1_workflow", "fig1", "FIG", (), "From reads to a PSMC history with psmc-rs"),
    FigureSpec("fig2_simulated_accuracy", "fig2", "E1",
               ("E1_curves", "E1_truth", "E1_table1", "E1_replicates", "E1_rs_vs_c"),
               "Accuracy on simulated genomes, and C psmc and psmc-rs 0.2 compared"),
    FigureSpec("figS1_em_convergence", "figs1", "E1", ("E1_traces",),
               "EM convergence of C psmc and psmc-rs 0.2"),
)
BY_NAME: Dict[str, FigureSpec] = {f.name: f for f in FIGURES}


def spec(name: str) -> FigureSpec:
    """The figure called ``name`` (its file stem, or a prefix such as ``fig2`` or ``figS1``)."""
    if name in BY_NAME:
        return BY_NAME[name]
    hits = [f for f in FIGURES if f.name.split("_", 1)[0] == name]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"unknown figure {name!r}; known: {', '.join(f.name for f in FIGURES)}")


def missing_tables(fs: FigureSpec, runs_dir: Path) -> List[Path]:
    return [p for p in fs.table_paths(runs_dir) if not p.exists()]


def render(fs: FigureSpec, runs_dir: Path, profile: config.Profile, dpi: int = style.PNG_DPI,
           out_dir: Optional[Path] = None) -> List[Path]:
    """Draw one figure from the tables of ``runs_dir`` and write its three files."""
    from .data import Source

    mod = importlib.import_module(f".{fs.module}", __name__)
    fig = mod.draw(Source(Path(runs_dir), profile))
    target = Path(out_dir) if out_dir is not None else fs.outputs(runs_dir)[0].parent
    return style.save(fig, target, fs.name, dpi=dpi)


__all__ = ["BY_NAME", "FIGURES", "SCHEMAS", "FigureSpec", "missing_tables", "render", "spec"]
