"""Fig 7 (E7): 1000 Genomes, psmc-rs on the 30x genomes of 26 populations.

a-e: one panel per superpopulation (``config.E7.superpopulations``): each
population's curve is the median of all its genomes' fits (fits of their
psmcfa, ``fit-rs-psmcfa``; 61 to 113 genomes per population, the panel
gives the superpopulation's populations and genomes), blue, admixed
populations (``config.E7.admixed``) dashed; the other superpopulations'
medians (the same route) are grey hairlines for context. Fig S5 gives each
population's spread between its genomes. f: every genome of the cohort
(``config.E7.cohort``; the same psmcfa fits): the median of all (blue)
with the spread between individuals (``config.E7.spread``, a band of the
individuals, not a confidence interval), and each superpopulation's median
(dark). a-f share one Ne axis (the union of what they draw), so a 10-fold
change is the same height in every panel. The curves are labelled in a
gutter left of the time axis, each label tied by a thin leader to where
its curve starts (10^4 years); curves closer there than a line's width
share one label. A psmcfa records no callable bases per bin, so these fits
take every called bin as full, which makes theta, and every size and
time, a little low: the VCF route on the same genomes gives theta 8.4%
higher (Fig S6b; Fig 5 for the mechanism), a constant shift that leaves
the populations' order and shapes as they are.
g-i: the bootstrap samples (``config.E7.bootstrap_samples``, Li & Durbin
2011's), psmc-rs (blue) over C psmc (orange, wider, underneath) on the same
psmcfa, with psmc-rs's bootstrap band; the note gives the per-interval RMSE
between the two tools; each has its own Ne axis. Times outside the
comparison window (1% to 99% of pairwise coalescence times) are shaded
grey. Curves are scaled with ``config.E7.scaling`` (mu = 1.25e-8 per bp per
generation, g = 29 years).

Everything comes from E7's tables: ``E7_pop_bands`` (a-f), ``E7_bands`` and
``E7_rs_vs_c`` (g-i).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import style
from .fig3 import NOTE_LINE_SHARE

ROUTE = "rs-psmcfa"  # a-f: every genome's psmcfa fit
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
    ax.spines["bottom"].set_bounds(*data)
    minor = [m * 10.0**e for e in range(math.floor(math.log10(data[0])), math.ceil(math.log10(data[1])))
             for m in range(2, 10) if data[0] < m * 10.0**e < data[1]]
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    # The gutter holds labels, not times: no grid lines through them.
    ax.axvspan(left, data[0] / 10**0.004, color=style.SURFACE, linewidth=0, zorder=0.8)


def draw(src):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    boot = src.table("E7_bands")
    rsc = src.table("E7_rs_vs_c")
    data_x = tuple(cfg.figure_years)
    pops = _group(bands, "population", ROUTE)
    sups = _group(bands, "superpopulation", ROUTE)
    every = _group(bands, "all", ROUTE).get("all")
    pop_sup = {g: str(d.superpopulation.iloc[0]) for g, d in pops.items()}
    admixed = set(cfg.admixed)
    panels = list(cfg.superpopulations)[:5]

    def in_x(d):
        return d[(d.years >= data_x[0]) & (d.years <= data_x[1])]

    def window_of(d) -> Tuple[float, float]:
        return float(d.window_lo_years.iloc[0]), float(d.window_hi_years.iloc[0])

    def start(d) -> float:
        """Ne where the curve starts on the time axis (its label points there)."""
        d = in_x(d)
        return float(d.ne_median.iloc[0]) if not d.empty else float("nan")

    fig = style.figure("double", 155.0)
    gs = style.grid(fig, 3, 3, left_mm=13.0, right_mm=3.0, top_mm=13.0, bottom_mm=8.5, wspace_mm=9.0, hspace_mm=9.0)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

    # a-f share one Ne axis: the union of everything they draw.
    vals: List[float] = []
    for d in list(pops.values()) + list(sups.values()):
        vals += list(in_x(d).ne_median)
    if every is not None:
        d = in_x(every)
        vals += list(d.ne_lo) + list(d.ne_hi)
    ylim = _ylim([v for v in vals if v > 0] or [1e3, 1e5])

    # Labels: one block per stroke, measured to size the gutter left of the time axis.
    fig_w_mm, fig_h_mm = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    pos = axes[0].get_position()
    ax_w_mm, ax_h_mm = pos.width * fig_w_mm, pos.height * fig_h_mm
    span = math.log10(ylim[1] / ylim[0])
    dex_per_mm = span / ax_h_mm
    labels: Dict[int, List[Tuple[List[str], float]]] = {}
    for i, sup in enumerate(panels):
        ends = {g: start(pops[g]) for g in pops if pop_sup[g] == sup}
        labels[i] = label_blocks({g: y for g, y in ends.items() if y > 0}, MERGE_MM * dex_per_mm)
    labels[5] = label_blocks({s: start(d) for s, d in sups.items() if start(d) > 0}, MERGE_MM * dex_per_mm)
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for blocks in labels.values():
        for codes, _ in blocks:
            for code in codes:
                t = fig.text(0, 0, code, fontsize=LABEL_PT)
                widest = max(widest, t.get_window_extent(renderer).width / fig.dpi * 25.4)
                t.remove()
    gutter_mm = widest + LEADER_MM + GUTTER_PAD_MM
    data_dex = math.log10(data_x[1] / data_x[0])
    gutter_dex = data_dex * gutter_mm / (ax_w_mm - gutter_mm)
    left = data_x[0] / 10**gutter_dex
    x_dex_per_mm = data_dex / (ax_w_mm - gutter_mm)
    x_text = data_x[0] / 10 ** (LEADER_MM * x_dex_per_mm)
    x_bar = x_text * 10 ** (LEADER_GAP_PT / 72.0 * 25.4 * x_dex_per_mm)  # where a leader starts
    line_dex = LABEL_PT * LABEL_LINE / 72.0 * 25.4 * dex_per_mm

    def put_labels(ax, blocks, x_curve: float) -> None:
        heights = [len(codes) * line_dex for codes, _ in blocks]
        centres = place_labels([y for _, y in blocks], heights, math.log10(ylim[0]), math.log10(ylim[1]))
        leader = {"color": style.MUTED, "linewidth": LEADER_LW, "solid_capstyle": "butt", "zorder": 6}
        for (codes, y), c, h in zip(blocks, centres, heights):
            ax.text(x_text, 10**c, "\n".join(codes), fontsize=LABEL_PT, color=style.INK2, ha="right", va="center",
                    linespacing=LABEL_LINE, zorder=6)
            ax.plot([x_bar, x_curve], [10**c, 10**y], **leader)
            if len(codes) > 1:  # one stroke, one label: a bar joins its lines where the leader starts
                half = h / 2.0 - 0.3 * line_dex
                ax.plot([x_bar, x_bar], [10 ** (c - half), 10 ** (c + half)], **leader)

    # a-e: the superpopulations' populations.
    for i, sup in enumerate(panels):
        ax = axes[i]
        mine = [g for g in pops if pop_sup[g] == sup]
        for other, d in sups.items():
            if other != sup:
                dd = in_x(d)
                ax.plot(dd.years, dd.ne_median, color=style.MUTED, linewidth=CONTEXT_LW, zorder=2)
        x_curve = data_x[0]
        for g in mine:
            d = in_x(pops[g])
            ls = style.DASH if g in admixed else "-"
            ax.plot(d.years, d.ne_median, color=style.BLUE, linestyle=ls, linewidth=style.LW, zorder=4)
            x_curve = float(d.years.iloc[0]) if not d.empty else x_curve
        style.shade_outside(ax, window_of(sups[sup]) if sup in sups else data_x, data_x)
        put_labels(ax, labels[i], x_curve)
        _time_axis(ax, data_x, left, label=i >= 3)
        style.ne_axis(ax, ylim, label=style.NE_LABEL_SHORT if i % 3 == 0 else None)
        n = int(sups[sup].n.iloc[0]) if sup in sups else 0
        plural = "s" if len(mine) != 1 else ""
        style.note(ax, 0.97, 0.97, f"{sup}: {len(mine)} population{plural}, {n:,} genomes", ha="right",
                   color=style.INK, size=style.TEXT_PT, zorder=7)
    # f: the cohort.
    ax = axes[5]
    if every is not None:
        d = in_x(every)
        ax.fill_between(d.years, d.ne_lo, d.ne_hi, color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(d.years, d.ne_median, color=style.BLUE, linewidth=style.LW_BOLD, zorder=4)
        style.shade_outside(ax, window_of(every), data_x)
        n_all = int(every.n.iloc[0])
    else:
        n_all = 0
    x_curve = data_x[0]
    for sup, dsup in sups.items():
        d = in_x(dsup)
        ax.plot(d.years, d.ne_median, color=style.INK2, linewidth=style.LW_THIN, zorder=3)
        x_curve = float(d.years.iloc[0]) if not d.empty else x_curve
    put_labels(ax, labels[5], x_curve)
    _time_axis(ax, data_x, left, label=True)
    style.ne_axis(ax, ylim, label=None)
    lo_q, hi_q = cfg.spread
    style.note(ax, 0.97, 0.97, f"All {n_all:,} genomes", ha="right", color=style.INK, size=style.TEXT_PT, zorder=7)
    # g-i: psmc-rs against C psmc, with psmc-rs's bootstrap band (each its own Ne axis).
    for j, sid in enumerate(list(cfg.bootstrap_samples)[:3]):
        ax = axes[6 + j]
        d = boot[boot.sample_id == sid].sort_values("years")
        row = rsc[rsc.sample_id == sid]
        if d.empty or row.empty:
            style.note(ax, 0.5, 0.5, f"{sid}: no fits", ha="center", va="center")
            continue
        d = in_x(d)
        r = row.iloc[0]
        window = (float(r.window_lo_years), float(r.window_hi_years))
        style.shade_outside(ax, window, data_x)
        ax.fill_between(d.years, d.ne_band_lo, d.ne_band_hi, color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
                        zorder=2)
        ax.plot(d.years, d.ne_c, color=style.ORANGE, linewidth=2.0, zorder=3)
        ax.plot(d.years, d.ne_rs, color=style.BLUE, linewidth=style.LW, zorder=4)
        _time_axis(ax, data_x, left, label=True)
        style.ne_axis(ax, _ylim(list(d.ne_band_lo) + list(d.ne_band_hi) + list(d.ne_c), room_top=0.2),
                      label=style.NE_LABEL_SHORT if j == 0 else None)
        pop = str(r.population)
        style.note(ax, 0.97, 0.97, f"{sid} ({pop})", ha="right", color=style.INK, size=style.TEXT_PT, zorder=7)
        style.note(ax, 0.97, 0.97 - NOTE_LINE_SHARE,
                   f"psmc-rs vs C: RMSE {float(r.rmse_log10_ne_rs_vs_c_per_interval):.1e} log10",
                   ha="right", zorder=7)
    handles = [
        Line2D([], [], color=style.BLUE, linewidth=style.LW,
               label="psmc-rs 0.2 (a-e: median of a population's genomes; f: of all)"),
        Line2D([], [], color=style.BLUE, linestyle=style.DASH, linewidth=style.LW, label="Admixed population"),
        Line2D([], [], color=style.MUTED, linewidth=CONTEXT_LW, label="Other superpopulations (a-e)"),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"f: {lo_q:.0%}-{hi_q:.0%} of genomes; g-i: bootstrap 95%"),
        Line2D([], [], color=style.ORANGE, linewidth=2.0, label="C psmc"),
        Patch(facecolor=style.NEUTRAL, linewidth=0, label="Outside the comparison window"),
    ]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.2,
                 handlelength=2.4)
    style.panel_letters(fig, axes, align_cols=True)
    return fig


__all__ = ["ROUTE", "draw", "label_blocks", "place_labels"]
