"""The shared style of every manuscript figure: type, sizes, colours, marks and export.

Every figure module draws through this module, so the figures read as one
set:

* **Type.** Arial, else Helvetica, else DejaVu Sans; 7 pt text (axis
  labels, legends, notes in the panels, colorbar labels) and 8 pt bold
  lowercase panel letters. Mathtext (superscripts on log axes, N_e) uses
  the same family. **One exception to 7 pt: tick labels are 6 pt**
  (:data:`TICK_PT`, also the colorbar's ticks). The log axes label every
  decade as 10 with a superscript, and at 7 pt the end labels of
  neighbouring panels of the 4-column figures (Figs 2 and S2) run into one
  another. Category labels set as tick labels (Fig 5's masks, Fig S3's
  deciles) follow the tick labels. PDF and SVG keep text as text (TrueType,
  fonttype 42; SVG ``svg.fonttype = none``).
* **Sizes.** Widths of the journal's columns: single 89 mm, 1.5 column
  120 mm, double 183 mm; heights are chosen per figure.
* **Marks.** Data lines 0.75-1.25 pt, thin replicate lines 0.5 pt, axes
  0.6 pt; where C psmc and psmc-rs coincide, C psmc's line is drawn 2 pt
  wide underneath psmc-rs's 0.8 pt line, so both stay visible. No top or
  right spines, no titles (captions carry them), no dual axes; a recessive
  hairline grid where reading values off a log axis needs it.
* **Colour follows the entity in every figure** (dataviz reference palette,
  light mode, validated with ``dataviz/scripts/validate_palette.js``):

  ===============  =========  =================================================
  psmc-rs 0.2      #2a78d6    slot 1, blue
  C psmc           #eb6834    slot 2, orange
  psmc-rs 0.1      #1baf7a    slot 3, aqua (performance only)
  truth            #0b0b0b    primary ink, dashed step line
  ===============  =========  =================================================

  ``validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
  --surface "#ffffff"`` passes every check (worst CVD pair 9.2, normal vision
  24.0); aqua is below 3:1 on white, so it always has its own marker and
  the legend, and Table 2 is its table view. Further categories take the
  next slots in order: the multihetsep route (Fig 5) and input (Fig S2)
  slot 4 #eda100, the VCF read without its mask slot 5 #e87ba4;
  ``"#2a78d6,#eda100,#e87ba4" --pairs all`` passes (CVD 13.0, normal vision
  19.6) with both new slots below 3:1, so their lines also carry their
  marker shapes. A psmc-rs variant of the default (in Fig 5 the VCF + mask
  route with every called bin taken as full, ``--bin-callable ignore``, and
  the fq2psmcfa route it equals; the VCF input of Fig S2; the 5 Mb
  bootstrap trunks of Figs 3 and 6; the capped M-step of Fig 4) is
  the next step of the blue ramp, 650 #104281, dotted or dashed
  (``"#2a78d6,#104281" --ordinal`` passes). Uncertainty bands use
  the entity's colour at low alpha; counts use the sequential blue ramp.

Export (:func:`save`) writes ``<name>.pdf``, ``.svg`` and a 600-dpi ``.png``
without creation dates, so drawing the same tables twice gives the same
bytes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

MM = 1.0 / 25.4  # inches per millimetre
WIDTH_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}

FONT_FAMILY = ("Arial", "Helvetica", "DejaVu Sans")
TEXT_PT = 7.0
TICK_PT = 6.0  # tick labels only: the documented exception to 7 pt (module docstring)
PANEL_PT = 8.0

LW_THIN = 0.5  # one replicate among many
LW = 1.0  # a data line
LW_BOLD = 1.25  # a median or the line the panel is about
LW_AXIS = 0.6
LW_GRID = 0.4
LW_REF = 0.6  # reference lines (zero, one, nominal coverage, ideal scaling)

# Ink and chrome (dataviz reference palette, light mode; the surface is white).
SURFACE = "#ffffff"
INK = "#0b0b0b"  # primary text
INK2 = "#52514e"  # secondary text, tick labels, axes
MUTED = "#898781"  # reference lines, annotations of secondary importance
GRID = "#e1e0d9"  # hairline grid
NEUTRAL = "#f0efec"  # neutral shading (outside the informative window)
REPLICATE_ALPHA = 0.35  # thin replicate lines in an entity's colour (see tint)
BAND_ALPHA = 0.18  # uncertainty bands in an entity's colour

PNG_DPI = 600
DASH = (0, (3.2, 1.6))  # the truth's dashes
DOT = (0, (1.0, 1.4))


@dataclass(frozen=True)
class Entity:
    """How one thing that recurs across figures (a tool, the truth, a route) is drawn."""

    label: str
    color: str
    linestyle: Union[str, Tuple] = "-"
    marker: str = "o"
    filled: bool = True

    def line(self, **kw) -> Dict:
        """Keyword arguments of ``ax.plot`` for this entity's line."""
        out = {"color": self.color, "linestyle": self.linestyle, "label": self.label}
        out.update(kw)
        return out

    def points(self, **kw) -> Dict:
        """Keyword arguments of ``ax.plot`` for this entity's markers alone."""
        out = {
            "color": self.color,
            "linestyle": "none",
            "marker": self.marker,
            "markersize": 3.2,
            "markeredgewidth": 0.6,
            "markerfacecolor": self.color if self.filled else SURFACE,
            "markeredgecolor": self.color,
            "label": self.label,
        }
        out.update(kw)
        return out


BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_650 = "#104281"

RS02 = Entity("psmc-rs 0.2", BLUE, "-", "o")
C_PSMC = Entity("C psmc", ORANGE, "-", "s")
RS01 = Entity("psmc-rs 0.1", AQUA, "-", "D")
TRUTH = Entity("Truth", INK, DASH, "none")

# Tools by the keys the tables use.
TOOLS: Dict[str, Entity] = {"rs": RS02, "rs02": RS02, "c": C_PSMC, "rs01": RS01}

# Fig 5 (E4): routes from a VCF and a callable mask to psmc-rs. psmc-rs's
# default, its callable-length model on the VCF with its mask, is psmc-rs's
# blue; the same bins with every called bin taken as full (fq2psmcfa's and C
# psmc's model) are its dashed variant.
ROUTES: Dict[str, Entity] = {
    "mask_auto": Entity("VCF + mask, callable bases per bin (psmc-rs default)", BLUE, "-", "o"),
    "mask_ignore": Entity("VCF + mask, every called bin full (= consensus → fq2psmcfa)", BLUE_650,
                          (0, (3.0, 1.5)), "D", filled=False),
    "fq2psmcfa": Entity("Consensus FASTA → fq2psmcfa", BLUE_650, DOT, "s", filled=False),
    "multihetsep": Entity("multihetsep (uncalled sites placed)", YELLOW, "-", "^"),
    "nomask": Entity("VCF without its mask", MAGENTA, "-", "v"),
}
ROUTE_ORDER = ("mask_auto", "mask_ignore", "fq2psmcfa", "multihetsep", "nomask")

# Fig S2 (E6): the same genome in three formats.
FORMATS: Dict[str, Entity] = {
    "psmcfa": Entity("psmcfa", BLUE, "-", "o"),
    "vcf": Entity("VCF", BLUE_650, DOT, "o", filled=False),
    "multihetsep": Entity("multihetsep", YELLOW, (0, (4.0, 1.5)), "^"),
}
FORMAT_ORDER = ("psmcfa", "vcf", "multihetsep")

# Sequential blue (magnitude, e.g. counts of a 2-D histogram), steps 100-700 of the ramp.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b")


def tint(color: str, share: float = REPLICATE_ALPHA) -> str:
    """``color`` at opacity ``share`` over the white surface, as an opaque colour.

    Replicate lines drawn in such a tint do not blend where two entities'
    lines coincide (half-transparent orange and blue make a grey violet
    that is neither), and their legend key shows the colour drawn.
    """
    from matplotlib.colors import to_hex, to_rgb

    c, s = to_rgb(color), to_rgb(SURFACE)
    return to_hex(tuple(share * a + (1.0 - share) * b for a, b in zip(c, s)))


def blue_cmap():
    """The sequential blue ramp as a colormap, light (few) to dark (many)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("psmc_blue", BLUE_RAMP)


# Fig 6 (E5): bootstrap trunk sizes, an ordered pair of steps of the blue ramp.
TRUNK_COLORS = (BLUE, BLUE_650)  # default (largest) trunk first
TRUNK_STYLES = ("-", (0, (3.0, 1.2)))


def rc_params() -> Dict:
    """The matplotlib settings of every figure."""
    family = list(FONT_FAMILY)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "font.size": TEXT_PT,
        "axes.labelsize": TEXT_PT,
        "axes.titlesize": TEXT_PT,
        "legend.fontsize": TEXT_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": LW_AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.labelpad": 2.5,
        "axes.facecolor": SURFACE,
        "axes.unicode_minus": True,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color": GRID,
        "grid.linewidth": LW_GRID,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "lines.linewidth": LW,
        "lines.solid_joinstyle": "miter",
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.2,
        "mathtext.fontset": "custom",
        "mathtext.rm": family[0],
        "mathtext.it": f"{family[0]}:italic",
        "mathtext.bf": f"{family[0]}:bold",
        "mathtext.sf": family[0],
        "mathtext.cal": family[0],
        "mathtext.tt": family[0],
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "psmc-rs-manuscript",
        "savefig.dpi": PNG_DPI,
        "path.simplify": False,
    }


def apply() -> None:
    """Use the Agg backend and this module's settings (idempotent)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update(rc_params())


def figure(width: Union[str, float], height_mm: float):
    """A new figure ``width`` (``single``, ``onehalf``, ``double`` or mm) by ``height_mm``."""
    import matplotlib.pyplot as plt

    apply()
    w = WIDTH_MM[width] if isinstance(width, str) else float(width)
    return plt.figure(figsize=(w * MM, height_mm * MM))


def grid(
    fig,
    nrows: int,
    ncols: int,
    *,
    left_mm: float = 11.0,
    right_mm: float = 2.0,
    top_mm: float = 4.0,
    bottom_mm: float = 8.0,
    wspace_mm: float = 6.0,
    hspace_mm: float = 7.0,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
):
    """A GridSpec whose margins and gaps are given in millimetres."""
    w_mm = fig.get_figwidth() / MM
    h_mm = fig.get_figheight() / MM
    plot_w = w_mm - left_mm - right_mm
    plot_h = h_mm - top_mm - bottom_mm
    wr = list(width_ratios or [1.0] * ncols)
    hr = list(height_ratios or [1.0] * nrows)
    mean_w = (plot_w - wspace_mm * (ncols - 1)) / ncols
    mean_h = (plot_h - hspace_mm * (nrows - 1)) / nrows
    return fig.add_gridspec(
        nrows,
        ncols,
        left=left_mm / w_mm,
        right=1.0 - right_mm / w_mm,
        bottom=bottom_mm / h_mm,
        top=1.0 - top_mm / h_mm,
        wspace=wspace_mm / mean_w if ncols > 1 else 0.0,
        hspace=hspace_mm / mean_h if nrows > 1 else 0.0,
        width_ratios=wr,
        height_ratios=hr,
    )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def log_axis(ax, which: str = "x", grid: bool = False) -> None:
    """A log axis with 10^k labels, unlabelled minor ticks, and optionally a hairline major grid."""
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis.set_minor_formatter(NullFormatter())
    if grid:
        ax.grid(True, which="major", axis=which, color=GRID, linewidth=LW_GRID)


def years_axis(ax, lim: Tuple[float, float], label: bool = True, grid: bool = True) -> None:
    """The time axis of an Ne(t) panel: years ago, log scale."""
    log_axis(ax, "x", grid=grid)
    ax.set_xlim(*lim)
    if label:
        ax.set_xlabel("Years ago")


NE_LABEL = r"Effective population size, $N_\mathrm{e}$"
NE_LABEL_SHORT = r"$N_\mathrm{e}$"


def ne_axis(ax, lim: Tuple[float, float], label: Optional[str] = NE_LABEL, grid: bool = True) -> None:
    """The size axis of an Ne(t) panel: log scale."""
    log_axis(ax, "y", grid=grid)
    ax.set_ylim(*lim)
    if label:
        ax.set_ylabel(label)


def shade_outside(ax, window: Tuple[float, float], lim: Tuple[float, float]) -> None:
    """Shade the times outside the informative window (where no metric is computed)."""
    lo, hi = window
    if lim[0] < lo:
        ax.axvspan(lim[0], lo, color=NEUTRAL, linewidth=0, zorder=0)
    if hi < lim[1]:
        ax.axvspan(hi, lim[1], color=NEUTRAL, linewidth=0, zorder=0)


def ref_line(ax, y: Optional[float] = None, x: Optional[float] = None, **kw) -> None:
    """A thin reference line (zero difference, ratio one, nominal coverage)."""
    style = {"color": MUTED, "linewidth": LW_REF, "linestyle": "-", "zorder": 1}
    style.update(kw)
    if y is not None:
        ax.axhline(y, **style)
    if x is not None:
        ax.axvline(x, **style)


def note(ax, x: float, y: float, text: str, *, ha: str = "left", va: str = "top", color: str = INK2,
         size: float = TEXT_PT, transform=None, **kw):
    """A 7 pt annotation in secondary ink (never in a series colour)."""
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=size,
                   transform=transform if transform is not None else ax.transAxes, **kw)


def legend(ax_or_fig, handles: Sequence, labels: Optional[Sequence[str]] = None, **kw):
    """A frameless legend in 7 pt primary ink."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    opts = {"frameon": False, "fontsize": TEXT_PT, "handlelength": 2.4}
    opts.update(kw)
    return ax_or_fig.legend(list(handles), list(labels), **opts)


def handle(entity: Entity, kind: str = "line", **kw):
    """A legend handle for an entity: its line, its marker, or both."""
    from matplotlib.lines import Line2D

    if kind == "line":
        style = entity.line(linewidth=kw.pop("linewidth", LW_BOLD))
        style.update(kw)
        return Line2D([], [], **style)
    if kind == "points":
        style = entity.points()
        style.update(kw)
        return Line2D([], [], **style)
    style = entity.line(linewidth=kw.pop("linewidth", LW))
    style.update(marker=entity.marker, markersize=3.2, markeredgewidth=0.6,
                 markerfacecolor=entity.color if entity.filled else SURFACE, markeredgecolor=entity.color)
    style.update(kw)
    return Line2D([], [], **style)


# ---------------------------------------------------------------------------
# Panel letters
# ---------------------------------------------------------------------------


def panel_letters(fig, axes: Sequence, letters: Optional[Sequence[str]] = None, *, pad_pt: float = 2.0,
                  align_rows: bool = True, align_cols: bool = False) -> List:
    """Bold lowercase letters at the top left of each panel, outside its tick labels.

    Each letter sits at the left edge of everything the panel draws (tick
    labels, axis label) and just above the axes. Letters of panels whose
    axes share a top edge are aligned on one baseline; with ``align_cols``,
    letters of panels whose axes share a left edge (of their grid cells,
    before a fixed aspect shrinks them) are also aligned on one vertical,
    the leftmost of theirs.
    """
    import string

    from matplotlib.transforms import Bbox

    letters = list(letters or string.ascii_lowercase[: len(axes)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pad = pad_pt / 72.0 / fig.get_figheight()
    tops = {}
    out = []
    for ax, letter in zip(axes, letters):
        box = ax.get_tightbbox(renderer)
        (x0, _), _ = inv.transform(Bbox(box.get_points()).get_points())
        top = ax.get_position().y1
        key = round(top, 3) if align_rows else id(ax)
        tops.setdefault(key, []).append((ax, letter, x0, top))
    lefts: Dict = {}
    if align_cols:
        for row in tops.values():
            for ax, _, x0, _ in row:
                key = round(ax.get_position(original=True).x0, 3)
                lefts[key] = min(lefts.get(key, x0), x0)
    for row in tops.values():
        y = max(t for *_, t in row) + pad
        for ax, letter, x0, _ in row:
            x = lefts.get(round(ax.get_position(original=True).x0, 3), x0)
            out.append(fig.text(x, y, letter, fontsize=PANEL_PT, fontweight="bold", ha="left", va="bottom",
                                color=INK))
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FORMATS_OUT = ("pdf", "svg", "png")


def save(fig, out_dir: Path, name: str, dpi: int = PNG_DPI) -> List[Path]:
    """Write ``name.pdf``, ``name.svg`` and ``name.png`` (``dpi``) to ``out_dir``, reproducibly."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    metadata = {
        "pdf": {"Creator": "psmc-rs manuscript figures", "Producer": None, "CreationDate": None, "ModDate": None},
        "svg": {"Creator": "psmc-rs manuscript figures", "Date": None},
        "png": {"Software": None},
    }
    written = []
    for ext in FORMATS_OUT:
        path = out_dir / f"{name}.{ext}"
        tmp = path.with_name(path.name + ".partial")
        fig.savefig(tmp, format=ext, dpi=dpi if ext == "png" else None, metadata=metadata[ext])
        tmp.replace(path)
        written.append(path)
    plt.close(fig)
    return written


def step_xy(years: Sequence[float], ne: Sequence[float], xlim: Tuple[float, float]):
    """Coordinates of a PSMC step curve for ``drawstyle='steps-post'`` within ``xlim``.

    ``ne[k]`` holds from ``years[k]`` to ``years[k + 1]``; the first interval
    starts at the left edge of the axis and the last one runs to its right
    edge.
    """
    import numpy as np

    years = np.asarray(years, dtype=float)
    ne = np.asarray(ne, dtype=float)
    lo, hi = xlim
    starts = years.copy()
    starts[0] = lo
    keep = starts < hi
    keep[0] = True
    x = np.append(np.maximum(starts[keep], lo), hi)
    y = np.append(ne[keep], ne[keep][-1])
    # Intervals that end before the left edge collapse onto it; keep the last of them.
    first = np.searchsorted(x, lo, side="right") - 1
    return x[first:], y[first:]

