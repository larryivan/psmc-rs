"""Figure 3: C psmc and psmc-rs on real genomes, with psmc-rs's bootstrap band.

One panel per genome (A-D in the order given): C psmc's and psmc-rs's
final Ne(t) step curves, the bootstrap band (2.5% to 97.5% of the
replicates at each time, ``E2_bands.csv``), and grey shading outside the
comparison window (1% to 99% of pairwise coalescence times under C psmc's
model). Above each panel: the genome (the human input is chromosome 1
only), its scaling, the RMSE of log10 Ne between the tools on the window,
and the bootstrap's replicates and trunks, marked when there are fewer
than 20, which psmc-rs warns makes the band coarse. Everything is read from the E2
tables, so the figure can be redrawn from a clone without the fits::

    python -m experiment.pipeline.figures.fig3_real_genomes \\
        --tables experiment/runs/E2/tables --out experiment/runs/E2/figures/E2_fig3_real_genomes

Both axes are logarithmic: the curves are compared in log10 Ne, and a band
of multiplicative spread reads the same at every size. Colours: categorical
slots 1 and 2 of the reference palette (blue psmc-rs, orange C psmc; their
separation passes the colour-vision-deficiency checks), with psmc-rs dashed
on top so that the two curves stay distinct where they coincide and in
greyscale. Text is Arial (Helvetica, DejaVu Sans as fallbacks), embedded in
the PDF as TrueType; neither file records a creation date, so a rerun on
the same tables and software writes the same bytes.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TABLE_FILES = {
    "fit_summary": "E2_fit_summary.csv",
    "pairs": "E2_rs_vs_c.csv",
    "bootstrap": "E2_bootstrap.csv",
    "bands": "E2_bands.csv",
    "curves": "E2_curves.csv",
}

# Colours (reference palette, light mode) and ink.
C_COLOR = "#eb6834"  # categorical slot 2, orange: C psmc
RS_COLOR = "#2a78d6"  # categorical slot 1, blue: psmc-rs
BAND_COLOR = "#b7d3f6"  # blue ramp step 150: psmc-rs bootstrap band
OUTSIDE_COLOR = "#f0efec"  # neutral grey: outside the comparison window
INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#898781"
GRID = "#e1e0d9"

WIDTH_IN = 7.09  # 180 mm, a double-column figure
PANEL_HEIGHT_IN = 2.75
LEGEND_IN = 0.42
MARGIN_DECADES = 0.45  # x range: the window widened by this many decades on each side
DPI = 300
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.sf": "sans",
    "mathtext.cal": "sans:italic",  # unused; keeps matplotlib from looking up a cursive font
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "axes.labelsize": 7,
    "axes.labelcolor": INK,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
}


@dataclass
class Panel:
    """What one panel shows, read from the tables."""

    sample_id: str
    species: str
    label: str
    mu: float
    gen_years: float
    window: Tuple[float, float]
    rmse: float
    replicates: int
    trunks: Optional[int]
    trunk_mb: Optional[float]
    coarse_band: bool  # fewer trunks than psmc-rs's threshold for a coarse band
    band_quantiles: Tuple[float, float]
    curves: Dict[str, Tuple[List[float], List[float]]]  # tool -> (years, ne) of the steps
    band: Tuple[List[float], List[float], List[float]]  # years, lower, upper


def read_table(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x: Optional[str]) -> Optional[float]:
    return None if x in ("", None) else float(x)


def load_panels(tables: Path) -> Dict[str, Panel]:
    """One :class:`Panel` per genome of the E2 tables in directory ``tables``."""
    t = {k: read_table(Path(tables) / name) for k, name in TABLE_FILES.items()}
    fits = {r["sample_id"]: r for r in t["fit_summary"]}
    boots = {r["sample_id"]: r for r in t["bootstrap"] if r.get("default_trunks", "True") == "True"}
    panels: Dict[str, Panel] = {}
    for pair in t["pairs"]:
        sid = pair["sample_id"]
        fit, boot = fits[sid], boots[sid]
        curves: Dict[str, Tuple[List[float], List[float]]] = {}
        for tool in ("c", "rs"):
            rows = sorted(
                (r for r in t["curves"] if r["sample_id"] == sid and r["tool"] == tool), key=lambda r: int(r["interval"])
            )
            curves[tool] = ([float(r["years"]) for r in rows], [float(r["ne"]) for r in rows])
        band_rows = [r for r in t["bands"] if r["sample_id"] == sid and r.get("trunk_bins") == boot.get("trunk_bins")]
        trunks = _num(boot.get("trunks"))
        panels[sid] = Panel(
            sample_id=sid,
            species=fit["species"],
            label=pair["label"],
            mu=float(fit["mu"]),
            gen_years=float(fit["gen_years"]),
            window=(float(pair["window_lo_years"]), float(pair["window_hi_years"])),
            rmse=float(pair["rmse_log10_ne_rs_vs_c"]),
            replicates=int(boot["replicates"]),
            trunks=None if trunks is None else int(trunks),
            trunk_mb=_num(boot.get("trunk_mb")),
            coarse_band=boot.get("coarse_band") == "True",
            band_quantiles=(float(boot["band_lo_quantile"]), float(boot["band_hi_quantile"])),
            curves=curves,
            band=(
                [float(r["years"]) for r in band_rows],
                [float(r["ne_band_lo"]) for r in band_rows],
                [float(r["ne_band_hi"]) for r in band_rows],
            ),
        )
    return panels


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def step_xy(years: Sequence[float], ne: Sequence[float], lo: float, hi: float) -> Tuple[List[float], List[float]]:
    """A step curve (``ne[k]`` from ``years[k]`` on) cut to ``[lo, hi]``, for ``drawstyle='steps-post'``."""
    k0 = max(i for i, y in enumerate(years) if y <= lo) if years[0] <= lo else 0
    xs, ys = [lo], [ne[k0]]
    for y, n in zip(years[k0 + 1 :], ne[k0 + 1 :]):
        if y >= hi:
            break
        xs.append(y)
        ys.append(n)
    xs.append(hi)
    ys.append(ys[-1])
    return xs, ys


def x_range(window: Tuple[float, float]) -> Tuple[float, float]:
    return window[0] / 10**MARGIN_DECADES, window[1] * 10**MARGIN_DECADES


def sci(x: float, digits: int = 3) -> str:
    """``x`` as mathtext ``a×10^b`` (``7.9\\times10^{-9}``), without the dollar signs."""
    exp = int(math.floor(math.log10(abs(x))))
    mant = f"{x / 10**exp:.{digits}g}"
    return rf"{mant}\times10^{{{exp}}}"


def title(panel: Panel) -> str:
    """The species in italics, then the rest of the label (sample, 'chr1 only')."""
    rest = panel.label[len(panel.species):].strip() if panel.label.startswith(panel.species) else panel.label
    species = panel.species.replace(" ", r"\ ")
    return rf"$\mathit{{{species}}}$ {rest}".strip()


def subtitle(panel: Panel) -> str:
    """Two lines: the scaling and the tools' RMSE; the bootstrap's replicates and trunks (and a coarse band)."""
    trunks = ""
    if panel.trunks is not None:
        trunks = f", {panel.trunks} trunks" + (f" of {panel.trunk_mb:g} Mb" if panel.trunk_mb else "")
        trunks += " (few trunks: a coarse band)" if panel.coarse_band else ""
    return (
        rf"$\mu$ = ${sci(panel.mu)}$, g = {panel.gen_years:g} y;  "
        rf"RMSE of log$_{{10}}$ $N_e$, psmc-rs vs C: {panel.rmse:.2g}"
        "\n"
        f"bootstrap: {panel.replicates} replicates{trunks}"
    )


def ne_label(v: float, _pos=None) -> str:
    """A size as a plain number with thousands separators (2,000; 50,000; 1,000,000)."""
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def style_axes(ax, ylim: Tuple[float, float]) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    decades = math.log10(ylim[1] / ylim[0])
    subs = (1.0, 2.0, 5.0) if decades <= 2.6 else (1.0,)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(ne_label))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, width=0.6, length=2.5, which="major")
    ax.tick_params(colors=INK_2, width=0.4, length=1.5, which="minor")


def draw_panel(ax, panel: Panel, letter: str) -> None:
    import numpy as np

    lo, hi = x_range(panel.window)
    by, blo, bhi = (np.asarray(v, dtype=float) for v in panel.band)
    keep = (by >= lo) & (by <= hi)
    ax.axvspan(lo, panel.window[0], color=OUTSIDE_COLOR, lw=0, zorder=0)
    ax.axvspan(panel.window[1], hi, color=OUTSIDE_COLOR, lw=0, zorder=0)
    ax.fill_between(by[keep], blo[keep], bhi[keep], step="post", color=BAND_COLOR, lw=0, zorder=1)
    cx, cy = step_xy(*panel.curves["c"], lo, hi)
    rx, ry = step_xy(*panel.curves["rs"], lo, hi)
    ax.plot(cx, cy, drawstyle="steps-post", color=C_COLOR, lw=1.8, solid_capstyle="butt", zorder=3)
    ax.plot(rx, ry, drawstyle="steps-post", color=RS_COLOR, lw=1.0, dashes=(3.0, 1.6), zorder=4)
    inside = (by >= panel.window[0]) & (by <= panel.window[1])
    ys = [*cy, *ry, *bhi[inside], *blo[inside]]
    style_axes(ax, (min(ys) / 1.4, max(ys) * 1.4))
    ax.set_xlim(lo, hi)
    ax.set_title(title(panel), fontsize=7.5, color=INK, loc="left", pad=20)
    ax.text(0.0, 1.025, subtitle(panel), transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
            color=INK_2, linespacing=1.3)
    ax.text(-0.14, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


def legend_handles(panels: Dict[str, Panel]):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    q = next(iter(panels.values())).band_quantiles if panels else (0.025, 0.975)
    reps = "/".join(str(n) for n in sorted({p.replicates for p in panels.values()}))
    return [
        Line2D([], [], color=C_COLOR, lw=1.8, label="C psmc 0.6.5"),
        Patch(facecolor=BAND_COLOR, edgecolor="none",
              label=f"psmc-rs bootstrap: {100 * q[0]:g}% to {100 * q[1]:g}% of {reps} replicates"),
        Line2D([], [], color=RS_COLOR, lw=1.0, dashes=(3.0, 1.6), label="psmc-rs 0.2"),
        Patch(facecolor=OUTSIDE_COLOR, edgecolor="none",
              label="outside the window (1% to 99% of coalescence times, C psmc model)"),
    ]


def draw(panels: Dict[str, Panel], order: Sequence[str]):
    """The figure: one panel per genome in ``order`` (then any others), two per row, one legend.

    Call it inside ``matplotlib.rc_context(STYLE)`` and save it there too
    (:func:`render` does), since tick labels and math text take their fonts
    when drawn.
    """
    import matplotlib.pyplot as plt

    ids = [s for s in order if s in panels] + [s for s in panels if s not in order]
    ncols = 2 if len(ids) > 1 else 1
    nrows = max(math.ceil(len(ids) / ncols), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_IN, PANEL_HEIGHT_IN * nrows + LEGEND_IN), squeeze=False)
    for i, sid in enumerate(ids):
        ax = axes[i // ncols][i % ncols]
        draw_panel(ax, panels[sid], "ABCDEFGH"[i])
        if i % ncols == 0:
            ax.set_ylabel("Effective population size, $N_e$")
        if i + ncols >= len(ids):
            ax.set_xlabel("Years ago")
    for j in range(len(ids), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.legend(handles=legend_handles(panels), loc="upper center", ncol=2, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=2.0, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - LEGEND_IN / fig.get_figheight()), h_pad=1.0, w_pad=2.0)
    return fig


def save(fig, stem: Path, formats: Sequence[str]) -> List[Path]:
    """Write ``stem.<format>`` for each format, without a creation date (reruns give the same bytes)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        out = stem.with_name(f"{stem.name}.{ext}")
        tmp = out.with_name(f"{stem.name}.partial.{ext}")
        meta = {"pdf": {"CreationDate": None}, "png": {"Software": None}}.get(ext, {})
        fig.savefig(tmp, dpi=DPI, metadata=meta, facecolor="white")
        tmp.replace(out)
        written.append(out)
    return written


def render(panels: Dict[str, Panel], order: Sequence[str], stem: Path, formats: Sequence[str]) -> List[Path]:
    """Draw and save the figure in its style; returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig = draw(panels, order)
        try:
            return save(fig, stem, formats)
        finally:
            plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.figures.fig3_real_genomes", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", type=Path, required=True, help="directory of the E2 tables")
    p.add_argument("--out", type=Path, required=True, help="output path without extension")
    p.add_argument("--order", default="", help="sample ids of panels A, B, ... (comma-separated)")
    p.add_argument("--formats", default="pdf,png", help="comma-separated formats (default pdf,png)")
    args = p.parse_args(argv)
    panels = load_panels(args.tables)
    order = [s for s in args.order.split(",") if s]
    for path in render(panels, order, args.out, [f for f in args.formats.split(",") if f]):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
