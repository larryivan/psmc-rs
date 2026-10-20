"""Fig S2 (E6 S2): one simulated genome per model read as psmcfa, VCF and multihetsep.

The genome is E1's replicate ``config.E6.format_replicate`` of each model,
without missing data, in the three formats and fitted by psmc-rs 0.2 with
E1's settings.

* a-d: the three fits and the truth: the psmcfa fit wide, underneath; the
  multihetsep fit thin and solid over it; the VCF fit dotted on top (it
  equals the psmcfa fit, so its dots run along the wide line).
* e-h: |log10(Ne / Ne of the psmcfa fit)| interval by interval for the VCF
  and multihetsep inputs, drawn as a step curve on the psmcfa fit's
  intervals (log scale; the fits have the same intervals, whose boundaries
  differ in years only through theta and t_max, see
  :func:`..metrics.interval_diff`); an input whose fit equals the psmcfa's
  exactly is named instead of drawn. Each panel gives the bins that differ
  from the psmcfa's and the bins after the last heterozygous site that a
  multihetsep, which ends at its last row, lacks.

Everything comes from ``E6_S2_curves``, ``E6_S2_bins`` and ``E6_S2_fits``.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import replace
from typing import List

from .. import config, metrics
from . import style
from .data import Source, curves, decades, time_grid, truth_xy
from .fig2 import MODEL_LABEL, model_order

FLOOR = 1e-6  # the floor of the differences axis (a smaller difference is drawn on it)
# How the formats are drawn here: the multihetsep solid (style's dashes are
# too close to the truth's), the VCF on top of it.
FORMATS = {**style.FORMATS, "multihetsep": replace(style.FORMATS["multihetsep"], linestyle="-")}
FORMAT_LW = {"psmcfa": 2.0}
FORMAT_Z = {"psmcfa": 2, "multihetsep": 3, "vcf": 4}
NOTE_WIDTH = 27  # characters per line of a panel's note
NOTE_LINE_SHARE = 0.09  # share of a bottom panel's height one line of a note takes


def draw(src: Source):
    import numpy as np

    g = src.profile.e1.scaling.gen_years
    table = src.table("E6_S2_curves")
    fits = curves(table[table.format != "truth"], ["model", "format"])
    bins = src.table("E6_S2_bins")
    bins = bins[bins.sequence == "all"]
    fit_rows = src.table("E6_S2_fits")
    models = model_order({m for m, _ in fits})
    inputs = [i for i in style.FORMAT_ORDER if any(k[1] == i for k in fits)]
    xlim = config.FIGURES.sim_years
    grid = time_grid(xlim, config.FIGURES.curve_grid_points)
    truths = {m: truth_xy(m, g, xlim) for m in models}
    windows = {r.model: (float(r.window_lo_years), float(r.window_hi_years)) for r in fit_rows.itertuples()}
    n = len(models)
    fig = style.figure("double", 96)
    gs = style.grid(fig, 2, n, left_mm=13.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=4.5, hspace_mm=4.5,
                    height_ratios=[1.0, 0.75])
    top = [fig.add_subplot(gs[0, j]) for j in range(n)]
    bottom = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(n)]

    vals: List[float] = []
    diffs = {}  # |log10 ratio| of each interval to the psmcfa fit's
    for m in models:
        window = windows[m]
        inside = (grid >= window[0]) & (grid <= window[1])
        for i in inputs:
            if (m, i) in fits:
                vals += list(fits[(m, i)].at(grid[inside]))
                if i != "psmcfa" and (m, "psmcfa") in fits:
                    diffs[(m, i)] = np.abs(metrics.interval_log10_ratio(fits[(m, i)], fits[(m, "psmcfa")]))
        _, ty = truths[m]
        vals += [float(ty.min()), float(ty.max())]
    ylim = decades(min(vals) / 1.5, max(vals) * 1.5)
    notes = {m: _note(m, inputs, diffs, bins) for m in models}
    positive = [d[d > 0] for d in diffs.values()]
    hi = max([float(p.max()) for p in positive if p.size] + [FLOOR * 10]) * 1.5
    # Room above the data for the longest note.
    room = NOTE_LINE_SHARE * max([len(t.splitlines()) for t in notes.values()] + [0]) + 0.04
    dlim = (FLOOR, FLOOR * 10 ** (math.log10(hi / FLOOR) / (1.0 - room)))

    for j, m in enumerate(models):
        ax, axd = top[j], bottom[j]
        for a in (ax, axd):
            style.shade_outside(a, windows[m], xlim)
        for i in inputs:
            c = fits.get((m, i))
            if c is None:
                continue
            ent = FORMATS[i]
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linestyle=ent.linestyle,
                    linewidth=FORMAT_LW.get(i, style.LW_DATA), zorder=FORMAT_Z.get(i, 3))
        ax.plot(*truths[m], **style.truth_kw())
        style.years_axis(ax, xlim, label=False)
        style.ne_axis(ax, ylim, label=style.NE_LABEL if j == 0 else None)
        ax.tick_params(labelbottom=False)
        style.note(ax, 0.97, 0.03, MODEL_LABEL.get(m, m), ha="right", va="bottom", color=style.INK, size=style.TEXT_PT)

        for i in inputs:
            d = diffs.get((m, i))
            if d is None or np.all(d == 0):
                continue
            x, y = style.step_xy(fits[(m, "psmcfa")].years, np.maximum(d, dlim[0]), xlim)
            axd.plot(x, y, drawstyle="steps-post", color=FORMATS[i].color, linewidth=style.LW_DATA, zorder=3)
        style.note(axd, 0.03, 0.97, notes[m], linespacing=1.2)
        style.years_axis(axd, xlim)
        style.log_axis(axd, "y", grid=True, lim=dlim)
        if j == 0:
            axd.set_ylabel("|log$_{10}$ ratio| per\ninterval to psmcfa fit")
        else:
            axd.tick_params(labelleft=False)
            ax.tick_params(labelleft=False)

    handles = [style.handle(FORMATS[i], linewidth=FORMAT_LW.get(i, style.LW_DATA)) for i in inputs]
    handles.append(style.handle(style.TRUTH))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, [*top, *bottom])
    return fig


def _note(model: str, inputs, diffs, bins) -> str:
    """What a bottom panel says of each format against the psmcfa: its bins, and whether its fit is the same."""
    import numpy as np

    lines = []
    for i in inputs:
        if i == "psmcfa" or (model, i) not in diffs:
            continue
        row = bins[(bins.model == model) & (bins.format == i)]
        parts = []
        if len(row):
            r = row.iloc[0]
            differ, missing = int(r.bins_differing), int(r.bins_missing_at_end)
            parts.append(f"{differ:,} bins differ" if differ else "same bins")
            if missing:
                parts.append(f"{missing:,}\u00a0tail bins missing")  # the count stays with its words
        if np.all(diffs[(model, i)] == 0):
            parts.append("same fit")
        lines.append(textwrap.fill(f"{FORMATS[i].label}: " + ", ".join(parts), NOTE_WIDTH, subsequent_indent="  "))
    return "\n".join(lines)


__all__ = ["draw"]
