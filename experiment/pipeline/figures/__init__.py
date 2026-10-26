"""The manuscript's figures, drawn from the experiments' committed tables.

Every figure is a module of this package with ``draw(source) -> Figure``
(``source`` is a :class:`.data.Source`); :data:`FIGURES` lists them with the
tables they read and the experiment directory they are written to:
``<runs>/<experiment>/figures/<name>.{pdf,svg,png}``. :func:`render` draws
one and saves it with :func:`.style.save`. The experiment ``FIG``
(``experiments/fig.py``) makes each figure a runner job whose inputs are its
tables, and ``python -m experiment.pipeline.figures`` draws them directly.

:mod:`.style` holds the shared style (type, sizes, colours, export),
:mod:`.schemas` the experiments' tables the figures read (the columns they
need), and :mod:`.data` their readers and the curve arithmetic. Every
figure of the manuscript is drawn here, and only here.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import config
from . import style
from .schemas import SCHEMAS

HERE = Path(__file__).resolve().parent
_FROM_SIBLING = re.compile(r"^from \.(\w+) import", re.M)  # a module of this package another one imports from


@dataclass(frozen=True)
class FigureSpec:
    name: str  # file stem
    module: str  # module of this package with draw(source)
    experiment: str  # written to <runs>/<experiment>/figures/
    tables: Tuple[str, ...]  # keys of schemas.SCHEMAS
    title: str  # what the figure shows (the caption carries it)

    @property
    def sources(self) -> Tuple[str, ...]:
        """The experiments whose tables the figure reads, in order."""
        return tuple(dict.fromkeys(SCHEMAS[key].experiment for key in self.tables))

    def table_paths(self, runs_dir: Path) -> List[Path]:
        return [SCHEMAS[k].path(runs_dir) for k in self.tables]

    def outputs(self, runs_dir: Path) -> List[Path]:
        d = Path(runs_dir) / self.experiment / "figures"
        return [d / f"{self.name}.{ext}" for ext in style.FORMATS_OUT]

    def code_files(self) -> List[str]:
        """The pipeline files (relative to experiment/pipeline) that draw it.

        The shared ones, its module and the modules of this package it imports from, at any depth
        (figS1_em_convergence takes fig2_simulated_accuracy's model names), so that a change to any of them
        redraws it.
        """
        modules, todo = set(), [self.module]
        while todo:
            m = todo.pop()
            if m not in modules:
                modules.add(m)
                todo += _FROM_SIBLING.findall((HERE / f"{m}.py").read_text())
        return sorted({"config.py", "results.py", "metrics.py", "demography.py", "figures/__init__.py",
                       "figures/style.py", "figures/data.py", "figures/schemas.py",
                       *(f"figures/{m}.py" for m in modules)})


FIGURES: Tuple[FigureSpec, ...] = (
    # The article's Figs 1 and 2 (Supplementary Note 1, section 3, maps every article figure to its file); its
    # Fig 3 is fig7_1000_genomes, and the supplement's Figs S1-S3 are figS1_em_convergence,
    # fig5_input_preparation and fig6_bootstrap_calibration.
    FigureSpec("art1_agreement", "art1_agreement", "FIG",
               ("E1_curves", "E1_table1", "E1_replicates", "E1_rs_vs_c", "E2_curves", "E2_bands", "E2_rs_vs_c",
                "E7_rs_vs_c"),
               "psmc-rs reproduces C psmc, on simulated and on real genomes"),
    FigureSpec("art2_performance", "art2_performance", "E3", ("E3_table2", "E7_fits"),
               "psmc-rs is fast and lean: run time, speed-up over C psmc and memory, and the 1000 Genomes fits"),
    FigureSpec("fig1_workflow", "fig1", "FIG", (), "From reads to a PSMC history with psmc-rs"),
    FigureSpec("fig2_simulated_accuracy", "fig2", "E1",
               ("E1_curves", "E1_truth", "E1_table1", "E1_replicates", "E1_rs_vs_c", "E1_mstep"),
               "Accuracy on simulated genomes, and C psmc and psmc-rs 0.2 compared"),
    FigureSpec("fig3_real_genomes", "fig3", "E2", ("E2_curves", "E2_bands", "E2_bootstrap", "E2_rs_vs_c",
                                                   "E2_fit_summary"),
               "Four real genomes: C psmc, psmc-rs 0.2 and its bootstrap"),
    FigureSpec("fig4_performance", "fig4", "E3", ("E3_table2", "E3_runs"),
               "Run time and memory against genome size, and the speed-up over C psmc"),
    FigureSpec("fig5_input_preparation", "fig5", "E4", ("E4_routes", "E4_curves"),
               "Partly callable bins bias PSMC; weighting each bin by its callable bases removes most of the bias"),
    FigureSpec("fig6_bootstrap_calibration", "fig6", "E5",
               ("E5_summary", "E5_sampling", "E5_bands", "E5_sd_ratio", "E5_coverage"),
               "Calibration of the bootstrap"),
    FigureSpec("figS1_em_convergence", "figs1", "E1", ("E1_traces",),
               "The EM iterations of C psmc and psmc-rs"),
    FigureSpec("figS2_format_consistency", "figs2", "E6", ("E6_S2_curves", "E6_S2_bins", "E6_S2_fits"),
               "One genome read as psmcfa, VCF and multihetsep"),
    FigureSpec("figS3_tmrca_decoding", "figs3", "E6",
               ("E6_S3_track", "E6_S3_density", "E6_S3_calibration", "E6_S3_summary"),
               "Posterior TMRCA against the true TMRCA"),
    FigureSpec("figS4_thread_scaling", "figs4", "E3", ("E3_thread_scaling", "E3_table2", "E3_runs", "E3_determinism"),
               "Thread scaling and determinism of psmc-rs 0.2"),
    # fig7_1000_genomes is the article's Fig 3; the other E7 figures are not in the article.
    FigureSpec("fig7_1000_genomes", "fig7", "E7", ("E7_pop_bands",),
               "1000 Genomes: the superpopulations compared, and their populations"),
    FigureSpec("figS5_1000g_populations", "figs5", "E7", ("E7_pop_bands", "E7_samples"),
               "1000 Genomes: every population, every genome's spread"),
    FigureSpec("figS6_1000g_checks", "figs6", "E7",
               ("E7_inputs", "E7_route_shift", "E7_split", "E7_fits", "E7_curves", "E7_giab", "E7_throughput"),
               "1000 Genomes: the inputs, the two routes, the first window, GIAB and throughput"),
    FigureSpec("figS_turtle_patterns", "figs_turtle", "E2", ("E2_split_curves", "E2_split_patterns"),
               "The turtles fitted with the first time window split: a check for false recent peaks"),
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
