"""Fig 1: from reads to a PSMC history with psmc-rs (a schematic; no data).

Reads of one diploid sample reach psmc-rs by one of two routes: consensus
calling of every site (Route A: ``bcftools mpileup | bcftools call -c``,
whose all-sites VCF psmc-rs reads exactly as ``vcfutils.pl vcf2fq |
fq2psmcfa`` would) or variant calls with a callable mask (Route B, binned
with fq2psmcfa's rule, each called bin keeping the number of its bases the
mask calls callable); psmcfa and multihetsep files made by other tools are
read as they are. ``qc`` and ``convert`` are subcommands: ``psmc-rs qc``
reports how suitable an input is, and ``psmc-rs convert`` writes it as
psmcfa or multihetsep. The fit is psmc-rs's command without one
(``psmc-rs INPUT OUTPUT N_ITER``, the line under its title); listed as
bullets, it fits C psmc's model with each called bin weighted by its callable
bases (``--bin-callable``, the default ``auto``: a homozygous bin with c of
its s bases callable has probability e_T(k)^(c/s) in state k) and writes
the outputs: the parameters (JSON), an HTML report, C psmc's ``.psmc``,
per-bin TMRCA posteriors and bootstrap replicates. psmc-rs's own steps are
blue; inputs and other tools are grey.

The drawing is laid out in millimetres on one axes spanning the figure.
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import style
from .data import Source

W, H = 183.0, 86.0  # mm
EDGE = style.MUTED
RS_FILL = "#e6f0fc"  # lighter than step 100 of the blue ramp, behind psmc-rs's boxes
RS_EDGE = style.BLUE
Box = Tuple[float, float, float, float]


def box(ax, x: float, y: float, w: float, h: float, title: str, detail: str = "", *, rs: bool = False,
        bold: bool = False, title_top: bool = False) -> Box:
    """A rounded box (mm) with a 7 pt title and optional detail lines; returns its extent.

    The title is centred in a box without detail lines, unless ``title_top``
    (the box's content is then drawn by the caller).
    """
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=RS_FILL if rs else style.SURFACE, edgecolor=RS_EDGE if rs else EDGE,
                                linewidth=0.7 if rs else 0.6, zorder=2))
    weight = "bold" if bold else "normal"
    if detail or title_top:
        ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK,
                fontweight=weight, zorder=3)
    if detail:
        ax.text(x + w / 2, y + h - 5.3, detail, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK2,
                linespacing=1.2, zorder=3)
    elif not title_top:
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
        ax.text(x, y - 0.8, note, ha="left", va="top", fontsize=style.TEXT_PT, color=style.INK2, linespacing=1.2)




def bullets(ax, x: float, top: float, items, *, step: float = 3.0, gap: float = 1.0) -> float:
    """Bullet points (7 pt, secondary ink) from ``top`` down at ``x`` (mm); returns the bottom of the last."""
    y = top
    for item in items:
        ax.text(x, y, "\u2022", ha="left", va="top", fontsize=style.TEXT_PT, color=style.INK2, zorder=3)
        ax.text(x + 2.2, y, item, ha="left", va="top", fontsize=style.TEXT_PT, color=style.INK2, linespacing=1.2,
                zorder=3)
        y -= step * (item.count("\n") + 1) + gap
    return y + gap


FIT_USAGE = "psmc-rs INPUT OUTPUT N_ITER"  # the fit has no subcommand (src/bin/psmc/cli.rs)
FIT_ITEMS = (
    "C psmc's model and EM",
    "each called bin weighted by its\ncallable bases, c of its s\n(--bin-callable, default auto):\n"
    r"$P(T\,|\,k, c) = e_T(k)^{\,c/s}$",
    "exact linear-time\nforward-backward",
    "same result for any --threads",
    "C-compatible bootstrap",
)


def draw(src: Optional[Source] = None):
    from matplotlib.patches import FancyBboxPatch

    fig = style.figure("double", H)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    # Column 1: reads.
    reads = box(ax, 1.5, 36.0, 20.0, 15.0, "BAM / CRAM", "one diploid\nsample")

    # Column 2: two routes to an input; inputs made by other tools.
    x2, w2, x3, w3 = 27.0, 31.0, 63.0, 25.0
    heading(ax, x2, 80.0, "Route A: consensus calls", "read exactly as vcfutils.pl vcf2fq | fq2psmcfa")
    a1 = box(ax, x2, 60.0, w2, 11.5, "bcftools mpileup |\nbcftools call -c")
    a2 = box(ax, x3, 60.0, w3, 11.5, "All-sites VCF")
    heading(ax, x2, 51.5, "Route B: variants and a callable mask",
            "binned with fq2psmcfa's rule; each bin keeps\nhow many of its bases are callable")
    b1 = box(ax, x2, 26.5, w2, 13.0, "Variant calling", "bcftools call -m,\nGATK, …")
    b2 = box(ax, x3, 33.0, w3, 7.0, "Variants VCF / gVCF")
    b3 = box(ax, x3, 24.5, w3, 7.0, "Callable mask (BED)")
    heading(ax, x2, 17.5, "Inputs made by other tools")
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
    px, py, pw, ph = 93.0, 1.5, 46.0, H - 3.0
    ax.add_patch(FancyBboxPatch((px, py), pw, ph, boxstyle="round,pad=0,rounding_size=1.8", facecolor="none",
                                edgecolor=RS_EDGE, linewidth=0.9, zorder=1))
    ax.text(px + pw / 2, py + ph - 2.0, "psmc-rs", ha="center", va="top", fontsize=style.PANEL_PT,
            fontweight="bold", color=style.INK)
    bx, bw = px + 2.0, pw - 4.0  # inset 2 mm: the fit's invocation line is 38 mm wide
    box(ax, bx, 61.5, bw, 15.5, "qc", "is the input suitable?\ncounts, heterozygosity,\nmissing data, QC codes",
        rs=True, bold=True)
    box(ax, bx, 47.0, bw, 12.0, "convert", "psmcfa or multihetsep,\nread as the fit reads it", rs=True, bold=True)
    # The fit is psmc-rs without a subcommand: its invocation under the title, so
    # that "fit" does not read as a third subcommand beside qc and convert.
    fit = box(ax, bx, py + 2.0, bw, 44.5 - (py + 2.0), "fit", FIT_USAGE, rs=True, bold=True)
    bullets(ax, bx + 2.0, fit[1] + fit[3] - 9.4, FIT_ITEMS, gap=0.6)
    for b in (a2, b2, b3, o2, o3):
        arrow(ax, right(b), (px, right(b)[1]), color=style.BLUE)

    # Column 4: outputs, written by the fit: one line from the fit box to a bus that feeds every output.
    ox, ow = 146.0, 35.5
    heading(ax, ox, 80.0, "Outputs of the fit")
    outs = [
        ("JSON", "parameters, log-likelihoods,\ninput summary, QC codes"),
        ("HTML report", "Ne(t), bootstrap band, QC"),
        (".psmc", "C psmc's format\n(psmc_plot.pl)"),
        ("TMRCA posterior", "per bin: mean, MAP,\n95% interval"),
        ("Bootstrap", "replicates and summary"),
    ]
    y, ys = 64.0, []
    for title, detail in outs:
        b = box(ax, ox, y, ow, 12.5, title, detail)
        ys.append(left(b)[1])
        y -= 15.5
    bus = (px + pw + ox) / 2
    yfit = right(fit, 4.0)[1]
    ax.plot([bx + bw, bus], [yfit, yfit], color=style.BLUE, linewidth=0.6, solid_capstyle="butt", zorder=1)
    ax.plot([bus, bus], [min(ys + [yfit]), max(ys + [yfit])], color=style.BLUE, linewidth=0.6,
            solid_capstyle="butt", zorder=1)
    for yo in ys:
        arrow(ax, (bus, yo), (ox, yo), color=style.BLUE)
    return fig
