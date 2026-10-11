"""Fig 7 (E7; the article's Fig 4): the 1000 Genomes 30x genomes, superpopulations and their populations.

a: each superpopulation's curve, the median of all its genomes' fits
(fits of their psmcfa, ``fit-rs-psmcfa``), one colour per superpopulation
(:data:`SUP_COLORS`), on one axis, so that they can be compared directly.
b-f: one panel per superpopulation (``config.E7.superpopulations``): the
median curve of each of its populations (61 to 113 genomes each), in
shades of the superpopulation's colour, admixed populations
(``config.E7.admixed``) dashed, over every other population's median in
light grey; b-f share one Ne axis. Every panel spans the cohort's
comparison window (``E7_pop_bands``, group ``all``: the median over the
genomes of 1% to 99% of pairwise coalescence times), so nothing drawn
lies where the data say little. The curves are labelled in a gutter left
of the time axis, each label tied by a thin leader to where its curve
starts; curves closer there than a line's width share one label. A
psmcfa records no callable bases per bin, so these fits take every called
bin as full, which makes theta, and every size and time, a little low:
the VCF route on the default set gives theta 8.4% higher, a shift that
leaves the populations' order and shapes as they are. Curves are scaled
with ``config.E7.scaling`` (mu = 1.25e-8 per bp per generation, g = 29
years).

Colours: the superpopulations take slots 4, 3, 7, 5 and 6 of the dataviz
categorical palette (yellow, aqua, violet, magenta, green) for AFR, AMR,
EAS, EUR and SAS, as slots 1 and 2 are psmc-rs and C psmc in every other
figure. ``validate_palette.js "#eda100,#1baf7a,#4a3aa7,#e87ba4,#008300"
--mode light --surface "#ffffff"`` passes (worst adjacent CVD 9.1, normal
vision 22.9; with ``--pairs all``, CVD 6.1, legal with the direct label
every curve has); three slots are below 3:1 on white, which the labels
relieve. A population's shade mixes its superpopulation's colour with
ink, darker for a higher curve where the curves start.

Everything comes from E7's table ``E7_pop_bands``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import style

ROUTE = "rs-psmcfa"  # every genome's psmcfa fit
SUP_COLORS = {"AFR": "#eda100", "AMR": "#1baf7a", "EAS": "#4a3aa7", "EUR": "#e87ba4", "SAS": "#008300"}
SUP_NAMES = {"AFR": "Africa", "AMR": "Americas", "EAS": "East Asia", "EUR": "Europe", "SAS": "South Asia"}
SHADE_MAX = 0.5  # the share of ink in the darkest population's shade
CONTEXT_LW = 0.4
LABEL_PT = style.TICK_PT  # curve labels in the gutter, set like tick labels (6 pt)
LABEL_LINE = 1.2  # a label's line height, in font sizes
LEADER_MM = 2.0  # horizontal run of a label's leader
GUTTER_PAD_MM = 0.6  # between the y axis and the widest label
LEADER_LW = 0.4
LEADER_GAP_PT = 1.0  # between a label and its leader
MERGE_MM = 0.45  # curves closer than this where they start are one stroke (1 pt lines): one label


def _ylim(values: Sequence[float], room_top: float = 0.12, room_bottom: float = 0.04) -> Tuple[float, float]:
    """Log limits around ``values`` with ``room_top``/``room_bottom`` of the axis kept free (axes fractions)."""
    lo, hi = min(values) / 1.3, max(values) * 1.3
    span = math.log10(hi / lo) / (1.0 - room_top - room_bottom)
    return lo / 10 ** (room_bottom * span), hi * 10 ** (room_top * span)


def label_blocks(ys: Dict[str, float], merge_dex: float) -> List[Tuple[List[str], float]]:
    """Labels for curves at ``ys`` (Ne): ``[(codes, log10 Ne)]``, low to high.

    Curves whose log10 Ne lies within ``merge_dex`` of the next one's form
    one stroke and share one label, its codes top down; the label points at
    their mean.
    """
    groups: List[List[Tuple[str, float]]] = []
    for k in sorted(ys, key=lambda k: ys[k]):
        y = math.log10(ys[k])
        if groups and y - groups[-1][-1][1] < merge_dex:
            groups[-1].append((k, y))
        else:
            groups.append([(k, y)])
    return [([k for k, _ in reversed(g)], sum(y for _, y in g) / len(g)) for g in groups]


def _isotonic(z: Sequence[float]) -> List[float]:
    """The non-decreasing sequence closest to ``z`` in least squares (pool adjacent violators)."""
    blocks: List[List[float]] = []
    for v in z:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out: List[float] = []
    for s, n in blocks:
        out += [s / n] * int(n)
    return out


def place_labels(targets: Sequence[float], heights: Sequence[float], lo: float, hi: float) -> List[float]:
    """Centres of labels stacked in the order of ``targets`` (ascending), as close to them as possible.

    Neighbouring labels do not overlap (centres at least half their heights
    apart), the stack lies within ``[lo, hi]``, and the centres are the
    least-squares closest to the targets under those constraints.
    """
    if not targets:
        return []
    offsets = [0.0]
    for a, b in zip(heights[:-1], heights[1:]):
        offsets.append(offsets[-1] + (a + b) / 2.0)
    z = _isotonic([t - c for t, c in zip(targets, offsets)])
    z_lo, z_hi = lo + heights[0] / 2.0, hi - heights[-1] / 2.0 - offsets[-1]
    return [min(max(v, z_lo), max(z_lo, z_hi)) + c for v, c in zip(z, offsets)]


def _group(bands, kind: str, route: str) -> Dict[str, object]:
    """``{group: rows sorted by years}`` of one kind and route of ``E7_pop_bands``."""
    sub = bands[(bands.group_kind == kind) & (bands.route == route)]
    return {g: d.sort_values("years") for g, d in sub.groupby("group", sort=True)}


def _time_axis(ax, data: Tuple[float, float], left: float, label: bool) -> None:
    """Years ago on ``data``, the axis widened to ``left`` for the labels' gutter, which shows no time."""
    from matplotlib.ticker import FixedLocator

    style.years_axis(ax, (left, data[1]), label=label)
    minor = [m * 10.0**e for e in range(math.floor(math.log10(data[0])), math.ceil(math.log10(data[1])))
             for m in range(2, 10) if data[0] < m * 10.0**e < data[1]]
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    # The gutter holds labels, not times: no grid lines through them.
    ax.axvspan(left, data[0] / 10**0.004, color=style.SURFACE, linewidth=0, zorder=0.8)


def shade(color: str, share: float) -> str:
    """``color`` mixed with ink at ``share`` (0: the colour, 1: ink), as an opaque colour."""
    from matplotlib.colors import to_hex, to_rgb

    c, k = to_rgb(color), to_rgb(style.INK)
    return to_hex(tuple((1.0 - share) * a + share * b for a, b in zip(c, k)))


def _gutter(fig, ax, data_x: Tuple[float, float], ylim: Tuple[float, float], widest_mm: float) -> Dict[str, float]:
    """Where the labels of ``ax`` go: the axis's left end, the labels' right end, the leaders' start, line heights."""
    fig_w_mm, fig_h_mm = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    pos = ax.get_position()
    ax_w_mm, ax_h_mm = pos.width * fig_w_mm, pos.height * fig_h_mm
    dex_per_mm = math.log10(ylim[1] / ylim[0]) / ax_h_mm
    gutter_mm = widest_mm + LEADER_MM + GUTTER_PAD_MM
    x_dex_per_mm = math.log10(data_x[1] / data_x[0]) / (ax_w_mm - gutter_mm)
    x_text = data_x[0] / 10 ** (LEADER_MM * x_dex_per_mm)
    return {
        "left": data_x[0] / 10 ** (gutter_mm * x_dex_per_mm),
        "x_text": x_text,
        "x_bar": x_text * 10 ** (LEADER_GAP_PT / 72.0 * 25.4 * x_dex_per_mm),
        "line_dex": LABEL_PT * LABEL_LINE / 72.0 * 25.4 * dex_per_mm,
        "merge_dex": MERGE_MM * dex_per_mm,
    }


def _put_labels(ax, blocks, g: Dict[str, float], x_curve: float, ylim: Tuple[float, float]) -> None:
    """Each block's label in the gutter, tied by a leader to where its curves start."""
    heights = [len(codes) * g["line_dex"] for codes, _ in blocks]
    centres = place_labels([y for _, y in blocks], heights, math.log10(ylim[0]), math.log10(ylim[1]))
    leader = {"color": style.MUTED, "linewidth": LEADER_LW, "solid_capstyle": "butt", "zorder": 6}
    for (codes, y), c, h in zip(blocks, centres, heights):
        ax.text(g["x_text"], 10**c, "\n".join(codes), fontsize=LABEL_PT, color=style.INK2, ha="right", va="center",
                linespacing=LABEL_LINE, zorder=6)
        ax.plot([g["x_bar"], x_curve], [10**c, 10**y], **leader)
        if len(codes) > 1:  # one stroke, one label: a bar joins its lines where the leader starts
            half = h / 2.0 - 0.3 * g["line_dex"]
            ax.plot([g["x_bar"], g["x_bar"]], [10 ** (c - half), 10 ** (c + half)], **leader)


def _mantissa_label(v: float, _pos=None) -> str:
    """A log-axis tick at m x 10^e as "10^e" or "m x 10^e", in the axis's mathtext."""
    e = math.floor(math.log10(v) + 1e-9)
    m = round(v / 10**e)
    return rf"$\mathdefault{{10^{{{e}}}}}$" if m == 1 else rf"$\mathdefault{{{m}\times10^{{{e}}}}}$"


def draw(src):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    pops = _group(bands, "population", ROUTE)
    sups = _group(bands, "superpopulation", ROUTE)
    every = _group(bands, "all", ROUTE).get("all")
    pop_sup = {g: str(d.superpopulation.iloc[0]) for g, d in pops.items()}
    admixed = set(cfg.admixed)
    panels = [s for s in cfg.superpopulations if s in sups][:5]
    if every is not None:
        data_x = (float(every.window_lo_years.iloc[0]), float(every.window_hi_years.iloc[0]))
    else:
        data_x = tuple(cfg.figure_years)

    def in_x(d):
        return d[(d.years >= data_x[0]) & (d.years <= data_x[1])]

    def start(d) -> float:
        """Ne where the curve starts on the time axis (its label points there)."""
        d = in_x(d)
        return float(d.ne_median.iloc[0]) if not d.empty else float("nan")

    fig = style.figure("double", 138.0)
    gs = style.grid(fig, 2, 5, left_mm=13.0, right_mm=3.0, top_mm=4.0, bottom_mm=8.5, wspace_mm=4.0,
                    hspace_mm=10.0, height_ratios=[1.45, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
    ylim_a = _ylim([v for d in sups.values() for v in in_x(d).ne_median if v > 0] or [1e3, 1e5], room_top=0.05)
    ylim_b = _ylim([v for d in pops.values() for v in in_x(d).ne_median if v > 0] or [1e3, 1e5], room_top=0.16)

    def width_mm(text: str) -> float:
        """The width of a label as it prints: measured at the PNG's dpi (at the screen's, hinting narrows it)."""
        dpi = fig.dpi
        fig.set_dpi(style.PNG_DPI)
        t = fig.text(0, 0, text, fontsize=LABEL_PT)
        w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
        t.remove()
        fig.set_dpi(dpi)
        return w

    # a: the superpopulations on one axis.
    g_a = _gutter(fig, ax_a, data_x, ylim_a, max([width_mm(SUP_NAMES.get(s, s)) for s in panels] or [0.0]))
    for s in panels:
        d = in_x(sups[s])
        ax_a.plot(d.years, d.ne_median, color=SUP_COLORS.get(s, style.INK2), linewidth=style.LW_BOLD, zorder=4)
    ends = {SUP_NAMES.get(s, s): start(sups[s]) for s in panels}
    _put_labels(ax_a, label_blocks({k: v for k, v in ends.items() if v > 0}, g_a["merge_dex"]), g_a, data_x[0],
                ylim_a)
    _time_axis(ax_a, data_x, g_a["left"], label=True)
    style.ne_axis(ax_a, ylim_a, label=style.NE_LABEL)
    # a spans little more than a decade of Ne: label 2 and 5 times the decades too
    ticks = [m * 10.0**e for e in range(2, 7) for m in (1, 2, 5) if ylim_a[0] < m * 10.0**e < ylim_a[1]]
    ax_a.yaxis.set_major_locator(FixedLocator(ticks))
    ax_a.yaxis.set_major_formatter(FuncFormatter(_mantissa_label))

    # b-f: each superpopulation's populations over every other population.
    widest = max([width_mm(g) for g in pops] or [0.0])
    g_b = _gutter(fig, axes[0], data_x, ylim_b, widest)
    context = style.tint(style.MUTED, 0.45)
    for i, sup in enumerate(panels):
        ax = axes[i]
        for g, d in pops.items():
            if pop_sup[g] != sup:
                dd = in_x(d)
                ax.plot(dd.years, dd.ne_median, color=context, linewidth=CONTEXT_LW, zorder=2)
        mine = sorted((g for g in pops if pop_sup[g] == sup), key=lambda g: -start(pops[g]))
        base = SUP_COLORS.get(sup, style.INK2)
        for k, g in enumerate(mine):
            d = in_x(pops[g])
            share = SHADE_MAX * (1.0 - k / max(len(mine) - 1, 1)) if len(mine) > 1 else 0.0
            ax.plot(d.years, d.ne_median, color=shade(base, share), linestyle=style.DASH if g in admixed else "-",
                    linewidth=style.LW, zorder=4)
        blocks = label_blocks({g: start(pops[g]) for g in mine if start(pops[g]) > 0}, g_b["merge_dex"])
        _put_labels(ax, blocks, g_b, data_x[0], ylim_b)
        _time_axis(ax, data_x, g_b["left"], label=i == 2)
        style.ne_axis(ax, ylim_b, label=style.NE_LABEL_SHORT if i == 0 else None)
        if i > 0:
            ax.tick_params(axis="y", labelleft=False)
        style.note(ax, 0.97, 0.97, SUP_NAMES.get(sup, sup), ha="right", color=style.INK, size=style.TEXT_PT, zorder=7)
    handles = [Line2D([], [], color=style.INK2, linewidth=style.LW,
                      label="Median of a superpopulation's (a) or a population's (b–f) genomes"),
               Line2D([], [], color=style.INK2, linestyle=style.DASH, linewidth=style.LW, label="Admixed population"),
               Line2D([], [], color=context, linewidth=1.0, label="The other populations")]
    style.legend(ax_a, handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=2.4)
    style.panel_letters(fig, [ax_a] + axes, align_cols=False)
    return fig


__all__ = ["ROUTE", "draw", "label_blocks", "place_labels"]
