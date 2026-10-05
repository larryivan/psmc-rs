"""Fig 1: from reads to a PSMC history with psmc-rs (a schematic; no data).

Reads of one diploid sample reach psmc-rs by one of two routes: consensus
calling of every site (Route A: ``bcftools mpileup | bcftools call -c``,
whose all-sites VCF psmc-rs reads exactly as ``vcfutils.pl vcf2fq |
fq2psmcfa`` would) or variant calls with a callable mask (Route B, binned
exactly with fq2psmcfa's rule); psmcfa and multihetsep files made by other
tools are read as they are. ``psmc-rs qc`` reports how suitable an input
is, ``convert`` writes it as psmcfa or multihetsep, and the fit writes the
parameters (JSON), an HTML report, C psmc's ``.psmc``, per-bin TMRCA
posteriors and bootstrap replicates. psmc-rs's own steps are blue; inputs
and other tools are grey.

The drawing is laid out in millimetres on one axes spanning the figure.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import style
from .data import Source

W, H = 183.0, 82.0  # mm
EDGE = style.MUTED
RS_FILL = "#e6f0fc"  # lighter than step 100 of the blue ramp, behind psmc-rs's boxes
RS_EDGE = style.BLUE
Box = Tuple[float, float, float, float]


def box(ax, x: float, y: float, w: float, h: float, title: str, detail: str = "", *, rs: bool = False,
        bold: bool = False) -> Box:
    """A rounded box (mm) with a 7 pt title and optional 6 pt detail lines; returns its extent."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=RS_FILL if rs else style.SURFACE, edgecolor=RS_EDGE if rs else EDGE,
                                linewidth=0.7 if rs else 0.6, zorder=2))
    weight = "bold" if bold else "normal"
    if detail:
        ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK,
                fontweight=weight, zorder=3)
        ax.text(x + w / 2, y + h - 5.3, detail, ha="center", va="top", fontsize=style.TICK_PT, color=style.INK2,
                linespacing=1.25, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=style.TEXT_PT, color=style.INK,
                fontweight=weight, linespacing=1.2, zorder=3)
    return x, y, w, h


def arrow(ax, start: Tuple[float, float], end: Tuple[float, float], *, color: str = style.INK2,
          elbow: bool = False) -> None:
    """An arrow from ``start`` to ``end``; with ``elbow``, horizontal-vertical-horizontal at the midpoint."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path

    if elbow and abs(start[1] - end[1]) > 1e-9:
        xm = (start[0] + end[0]) / 2
        path = Path([start, (xm, start[1]), (xm, end[1]), end],
                    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=5.5, linewidth=0.6, color=color,
                                shrinkA=0.0, shrinkB=0.4, joinstyle="miter", zorder=1)
    else:
        patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=5.5, linewidth=0.6, color=color,
                                shrinkA=0.0, shrinkB=0.4, zorder=1)
    ax.add_patch(patch)


def right(b: Box, dy: float = 0.0) -> Tuple[float, float]:
    return b[0] + b[2], b[1] + b[3] / 2 + dy


def left(b: Box, dy: float = 0.0) -> Tuple[float, float]:
    return b[0], b[1] + b[3] / 2 + dy


def heading(ax, x: float, y: float, text: str, note: str = "") -> None:
    ax.text(x, y, text, ha="left", va="bottom", fontsize=style.TEXT_PT, color=style.INK, fontweight="bold")
    if note:
        ax.text(x, y - 3.2, note, ha="left", va="bottom", fontsize=style.TICK_PT, color=style.INK2)




def draw(src: Optional[Source] = None):
    from matplotlib.patches import FancyBboxPatch

    fig = style.figure("double", H)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    # Column 1: reads.
    reads = box(ax, 1.5, 36.0, 21.0, 14.0, "BAM / CRAM", "one diploid\nsample")

    # Column 2: two routes to an input; inputs made by other tools.
    x2, w2, x3, w3 = 30.0, 33.0, 68.0, 25.0
    heading(ax, x2, 75.0, "Route A: consensus calls", "read exactly as vcfutils.pl vcf2fq | fq2psmcfa")
    a1 = box(ax, x2, 57.5, w2, 11.5, "bcftools mpileup |\nbcftools call -c")
    a2 = box(ax, x3, 57.5, w3, 11.5, "All-sites VCF")
    heading(ax, x2, 46.5, "Route B: variants and a callable mask", "binned exactly with fq2psmcfa's rule")
    b1 = box(ax, x2, 28.0, w2, 13.0, "Variant calling", "bcftools call -m, GATK, …")
    b2 = box(ax, x3, 34.5, w3, 7.0, "Variants VCF / gVCF")
    b3 = box(ax, x3, 26.0, w3, 7.0, "Callable mask (BED)")
    heading(ax, x2, 16.5, "Inputs made by other tools")
    o1 = box(ax, x2, 3.0, w2, 11.0, "fq2psmcfa, msmc-tools,\nsimulators")
    o2 = box(ax, x3, 8.5, w3, 6.0, "psmcfa")
    o3 = box(ax, x3, 1.5, w3, 6.0, "multihetsep")
    arrow(ax, right(reads, 2.0), left(a1), elbow=True)
    arrow(ax, right(reads, -2.0), left(b1), elbow=True)
    arrow(ax, right(a1), left(a2))
    arrow(ax, right(b1, 2.5), left(b2), elbow=True)
    arrow(ax, right(b1, -2.5), left(b3), elbow=True)
    arrow(ax, right(o1, 2.0), left(o2), elbow=True)
    arrow(ax, right(o1, -2.0), left(o3), elbow=True)

    # Column 3: psmc-rs.
    px, py, pw, ph = 99.0, 1.5, 39.0, 78.5
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=1.8", facecolor="none",
                                edgecolor=RS_EDGE, linewidth=0.9, zorder=1))
    ax.text(px + pw / 2, py + ph - 2.0, "psmc-rs", ha="center", va="top", fontsize=style.PANEL_PT,
            fontweight="bold", color=style.INK)
    bx, bw = px + 3.0, pw - 6.0
    box(ax, bx, 54.0, bw, 17.0, "qc", "is the input suitable?\ncounts, heterozygosity,\nmissing data, QC codes",
        rs=True, bold=True)
    box(ax, bx, 35.5, bw, 14.5, "convert", "psmcfa or multihetsep,\nread as the fit reads it", rs=True, bold=True)
    box(ax, bx, 4.5, bw, 27.0, "fit",
        "C psmc's model and EM\nexact linear-time\nforward-backward\nsame result for\nany --threads\nC-compatible bootstrap",
        rs=True, bold=True)
    for b in (a2, b2, b3, o2, o3):
        arrow(ax, right(b), (px, right(b)[1]), color=style.BLUE)

    # Column 4: outputs, written by psmc-rs.
    heading(ax, 144.0, 75.0, "Outputs")
    outs = [
        ("JSON", "parameters, log-likelihoods,\ninput summary, QC codes"),
        ("HTML report", "Ne(t), bootstrap band, QC"),
        (".psmc", "C psmc's format\n(psmc_plot.pl)"),
        ("TMRCA posterior", "per bin: mean, MAP,\n95% interval"),
        ("Bootstrap", "replicates and summary"),
    ]
    y = 62.0
    for title, detail in outs:
        b = box(ax, 144.0, y, 37.5, 11.0, title, detail)
        arrow(ax, (px + pw, left(b)[1]), left(b), color=style.BLUE)
        y -= 14.6
    return fig
