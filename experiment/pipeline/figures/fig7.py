"""Fig 7 (E7): 1000 Genomes, psmc-rs on the 30x genomes of 26 populations.

a-e: one panel per superpopulation (``config.E7.superpopulations``): each
population's curve (the median of its default samples' fits, from their
VCFs with the strict mask and psmc-rs's callable bases per bin), blue,
admixed populations (``config.E7.admixed``) dashed, labelled with their
three-letter code; the other superpopulations' medians are grey hairlines
for context. f: every genome of the cohort (``config.E7.cohort``; fits of
their psmcfa): the median of all (blue) with the spread between
individuals (``config.E7.spread``, a band of the individuals, not a
confidence interval), and each superpopulation's median (dark, labelled).
g-i: the bootstrap samples (``config.E7.bootstrap_samples``, Li & Durbin
2011's), psmc-rs (blue) over C psmc (orange, wider, underneath) on the same
psmcfa, with psmc-rs's bootstrap band; the note gives the per-interval RMSE
between the two tools. Times outside the comparison window (1% to 99% of
pairwise coalescence times) are shaded grey. Curves are scaled with
``config.E7.scaling`` (mu = 1.25e-8 per bp per generation, g = 29 years).

Everything comes from E7's tables: ``E7_pop_bands`` (a-f), ``E7_bands`` and
``E7_rs_vs_c`` (g-i), ``E7_samples`` (populations).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import style
from .fig3 import NOTE_LINE_SHARE

CONTEXT_LW = 0.4
LABEL_GAP = 0.065  # the least vertical gap between two labels, as a share of the axis height


def _ylim(values: Sequence[float], room_top: float = 0.12, room_bottom: float = 0.04) -> Tuple[float, float]:
    """Log limits around ``values`` with ``room_top``/``room_bottom`` of the axis kept free (axes fractions)."""
    lo, hi = min(values) / 1.3, max(values) * 1.3
    span = math.log10(hi / lo) / (1.0 - room_top - room_bottom)
    return lo / 10 ** (room_bottom * span), hi * 10 ** (room_top * span)


def spread_labels(ys: Dict[str, float], gap_dex: float) -> Dict[str, float]:
    """Label heights for curves ending at ``ys`` (Ne), pushed apart to at least ``gap_dex`` decades."""
    order = sorted(ys, key=lambda k: ys[k])
    out: Dict[str, float] = {}
    last = None
    for k in order:
        y = math.log10(ys[k])
        if last is not None and y < last + gap_dex:
            y = last + gap_dex
        out[k] = y
        last = y
    # Centre the block on the curves again so that pushing up does not drift it.
    shift = (sum(math.log10(ys[k]) for k in order) - sum(out.values())) / max(len(order), 1)
    return {k: 10 ** (v + shift) for k, v in out.items()}


def _group(bands, kind: str, route: str) -> Dict[str, object]:
    """``{group: rows sorted by years}`` of one kind and route of ``E7_pop_bands``."""
    sub = bands[(bands.group_kind == kind) & (bands.route == route)]
    return {g: d.sort_values("years") for g, d in sub.groupby("group", sort=True)}


def draw(src):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    boot = src.table("E7_bands")
    rsc = src.table("E7_rs_vs_c")
    xlim = tuple(cfg.figure_years)
    pops = _group(bands, "population", "rs-vcf")
    sups_vcf = _group(bands, "superpopulation", "rs-vcf")
    sups = _group(bands, "superpopulation", "rs-psmcfa")
    every = _group(bands, "all", "rs-psmcfa").get("all")
    pop_sup = {g: str(d.superpopulation.iloc[0]) for g, d in pops.items()}
    admixed = set(cfg.admixed)

    def in_x(d):
        return d[(d.years >= xlim[0]) & (d.years <= xlim[1])]

    def window_of(d) -> Tuple[float, float]:
        return float(d.window_lo_years.iloc[0]), float(d.window_hi_years.iloc[0])

    label_at = 10 ** (math.log10(xlim[0]) + 0.015 * math.log10(xlim[1] / xlim[0]))  # labels start here, inside

    fig = style.figure("double", 155.0)
    gs = style.grid(fig, 3, 3, left_mm=13.0, right_mm=3.0, top_mm=13.0, bottom_mm=8.5, wspace_mm=9.0, hspace_mm=9.0)
    axes = []
    # a-e: the superpopulations' populations; f: the cohort.
    for i, sup in enumerate(list(cfg.superpopulations)[:5]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)
        mine = [g for g in pops if pop_sup[g] == sup]
        vals: List[float] = []
        for other, d in sups_vcf.items():
            if other != sup:
                dd = in_x(d)
                ax.plot(dd.years, dd.ne_median, color=style.MUTED, linewidth=CONTEXT_LW, zorder=2)
        ends: Dict[str, float] = {}
        for g in mine:
            d = in_x(pops[g])
            ls = style.DASH if g in admixed else "-"
            ax.plot(d.years, d.ne_median, color=style.BLUE, linestyle=ls, linewidth=style.LW, zorder=4)
            vals += list(d.ne_median)
            ends[g] = float(np.interp(math.log10(label_at), np.log10(d.years), d.ne_median))
        window = window_of(sups_vcf[sup]) if sup in sups_vcf else xlim
        style.shade_outside(ax, window, xlim)
        ylim = _ylim(vals or [1e3, 1e5])
        for g, y in spread_labels(ends, LABEL_GAP * math.log10(ylim[1] / ylim[0])).items():
            ax.text(label_at, y, g, fontsize=style.TICK_PT, color=style.INK2, ha="left", va="bottom", zorder=6)
        style.years_axis(ax, xlim, label=i >= 3)
        style.ne_axis(ax, ylim, label=style.NE_LABEL_SHORT if i % 3 == 0 else None)
        plural = "s" if len(mine) != 1 else ""
        style.note(ax, 0.97, 0.97, f"{sup} ({len(mine)} population{plural})", ha="right", color=style.INK,
                   size=style.TEXT_PT, zorder=7)
    ax = fig.add_subplot(gs[1, 2])
    axes.append(ax)
    vals = []
    if every is not None:
        d = in_x(every)
        ax.fill_between(d.years, d.ne_lo, d.ne_hi, color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(d.years, d.ne_median, color=style.BLUE, linewidth=style.LW_BOLD, zorder=4)
        vals += list(d.ne_lo) + list(d.ne_hi)
        style.shade_outside(ax, window_of(every), xlim)
        n_all = int(every.n.iloc[0])
    else:
        n_all = 0
    ends = {}
    for sup, dsup in sups.items():
        d = in_x(dsup)
        ax.plot(d.years, d.ne_median, color=style.INK2, linewidth=style.LW_THIN, zorder=3)
        ends[sup] = float(np.interp(math.log10(label_at), np.log10(d.years), d.ne_median))
        vals += list(d.ne_median)
    ylim = _ylim(vals or [1e3, 1e5])
    for g, y in spread_labels(ends, LABEL_GAP * math.log10(ylim[1] / ylim[0])).items():
        ax.text(label_at, y, g, fontsize=style.TICK_PT, color=style.INK2, ha="left", va="bottom", zorder=6)
    style.years_axis(ax, xlim)
    style.ne_axis(ax, ylim, label=None)
    lo_q, hi_q = cfg.spread
    style.note(ax, 0.97, 0.97, f"All {n_all:,} genomes", ha="right", color=style.INK, size=style.TEXT_PT, zorder=7)
    # g-i: psmc-rs against C psmc, with psmc-rs's bootstrap band.
    for j, sid in enumerate(list(cfg.bootstrap_samples)[:3]):
        ax = fig.add_subplot(gs[2, j])
        axes.append(ax)
        d = boot[boot.sample_id == sid].sort_values("years")
        row = rsc[rsc.sample_id == sid]
        if d.empty or row.empty:
            style.note(ax, 0.5, 0.5, f"{sid}: no fits", ha="center", va="center")
            continue
        d = in_x(d)
        r = row.iloc[0]
        window = (float(r.window_lo_years), float(r.window_hi_years))
        style.shade_outside(ax, window, xlim)
        ax.fill_between(d.years, d.ne_band_lo, d.ne_band_hi, color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
                        zorder=2)
        ax.plot(d.years, d.ne_c, color=style.ORANGE, linewidth=2.0, zorder=3)
        ax.plot(d.years, d.ne_rs, color=style.BLUE, linewidth=style.LW, zorder=4)
        style.years_axis(ax, xlim)
        style.ne_axis(ax, _ylim(list(d.ne_band_lo) + list(d.ne_band_hi) + list(d.ne_c), room_top=0.2),
                      label=style.NE_LABEL_SHORT if j == 0 else None)
        pop = str(r.population)
        style.note(ax, 0.97, 0.97, f"{sid} ({pop})", ha="right", color=style.INK, size=style.TEXT_PT, zorder=7)
        style.note(ax, 0.97, 0.97 - NOTE_LINE_SHARE,
                   f"psmc-rs vs C: RMSE {float(r.rmse_log10_ne_rs_vs_c_per_interval):.1e} log10",
                   ha="right", zorder=7)
    handles = [
        Line2D([], [], color=style.BLUE, linewidth=style.LW, label="psmc-rs 0.2 (a-e: a population's VCF fits)"),
        Line2D([], [], color=style.BLUE, linestyle=style.DASH, linewidth=style.LW, label="Admixed population"),
        Line2D([], [], color=style.MUTED, linewidth=CONTEXT_LW, label="Other superpopulations"),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"f: {lo_q:.0%}-{hi_q:.0%} of genomes; g-i: bootstrap 95%"),
        Line2D([], [], color=style.ORANGE, linewidth=2.0, label="C psmc"),
        Patch(facecolor=style.NEUTRAL, linewidth=0, label="Outside the comparison window"),
    ]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.2,
                 handlelength=2.4)
    style.panel_letters(fig, axes, align_cols=True)
    return fig


__all__ = ["draw", "spread_labels"]
