"""Fig S2 (E6 S2): one simulated genome per model read as psmcfa, VCF and multihetsep.

The genome is E1's replicate ``config.E6.format_replicate`` of each model,
without missing data, in the three formats and fitted by psmc-rs 0.2 with
E1's settings.

* a-d: the three fits and the truth (the psmcfa fit wider, underneath);
* e-h: |log10(Ne / Ne of the psmcfa fit)| for the VCF and multihetsep
  inputs (log scale); an input whose fit equals the psmcfa's exactly is
  named instead of drawn. Each panel gives the bins that differ from the
  psmcfa's and the bins after the last heterozygous site that a
  multihetsep, which ends at its last row, lacks.

Everything comes from ``E6_S2_curves``, ``E6_S2_bins`` and ``E6_S2_fits``.
"""

from __future__ import annotations

from typing import List

from .. import config
from . import style
from .data import Source, curves, decades, log10_ratio, time_grid, truth_xy
from .fig2 import MODEL_LABEL, model_order

FLOOR = 1e-6


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e1
    figs = config.FIGURES
    g = cfg.scaling.gen_years
    table = src.table("E6_S2_curves")
    fits = curves(table[table.format != "truth"], ["model", "format"])
    bins = src.table("E6_S2_bins")
    bins = bins[bins.sequence == "all"]
    fit_rows = src.table("E6_S2_fits")
    models = model_order({m for m, _ in fits})
    inputs = [i for i in style.FORMAT_ORDER if any(k[1] == i for k in fits)]
    xlim = figs.sim_years
    grid = time_grid(xlim, figs.curve_grid_points)
    windows = {r.model: (float(r.window_lo_years), float(r.window_hi_years)) for r in fit_rows.itertuples()}
    n = len(models)
    fig = style.figure("double", 96)
    gs = style.grid(fig, 2, n, left_mm=13.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=4.5, hspace_mm=4.5,
                    height_ratios=[1.0, 0.75])
    top = [fig.add_subplot(gs[0, j]) for j in range(n)]
    bottom = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(n)]

    vals: List[float] = []
    diffs = {}
    for m in models:
        window = windows[m]
        inside = (grid >= window[0]) & (grid <= window[1])
        for i in inputs:
            if (m, i) in fits:
                vals += list(fits[(m, i)].at(grid[inside]))
                if i != "psmcfa" and (m, "psmcfa") in fits:
                    diffs[(m, i)] = np.abs(log10_ratio(fits[(m, i)], fits[(m, "psmcfa")], grid))
        _, ty = truth_xy(m, g, xlim)
        vals += [float(ty.min()), float(ty.max())]
    ylim = decades(min(vals) / 1.5, max(vals) * 1.5)
    positive = [d[d > 0] for d in diffs.values()]
    hi = max([float(p.max()) for p in positive if p.size] + [FLOOR * 10])
    dlim = decades(FLOOR, hi * 1.5)

    for j, m in enumerate(models):
        ax, axd = top[j], bottom[j]
        for a in (ax, axd):
            style.shade_outside(a, windows[m], xlim)
        for i in inputs:
            c = fits.get((m, i))
            if c is None:
                continue
            ent = style.FORMATS[i]
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linestyle=ent.linestyle,
                    linewidth=2.0 if i == "psmcfa" else style.LW, zorder=2 if i == "psmcfa" else 3)
        tx, ty = truth_xy(m, g, xlim)
        ax.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=4)
        style.years_axis(ax, xlim, label=False)
        style.ne_axis(ax, ylim, label=style.NE_LABEL if j == 0 else None)
        ax.tick_params(labelbottom=False)
        style.note(ax, 0.97, 0.03, MODEL_LABEL.get(m, m), ha="right", va="bottom", color=style.INK, size=style.TEXT_PT)

        lines = []
        for i in inputs:
            if i == "psmcfa" or (m, i) not in diffs:
                continue
            d = diffs[(m, i)]
            ent = style.FORMATS[i]
            row = bins[(bins.model == m) & (bins.format == i)]
            parts = []
            if len(row):
                r = row.iloc[0]
                parts.append(f"{int(r.bins_differing):,} bins differ")
                if int(r.bins_missing_at_end):
                    parts.append(f"{int(r.bins_missing_at_end):,} tail bins missing")
            if np.all(d == 0):
                parts.append("same fit")
            else:
                axd.plot(grid, np.maximum(d, dlim[0]), drawstyle="steps-post", color=ent.color, linewidth=style.LW,
                         zorder=3)
            lines.append(f"{ent.label}: " + ", ".join(parts))
        style.note(axd, 0.03, 0.97, "\n".join(lines), ha="left", va="top", linespacing=1.25)
        style.years_axis(axd, xlim)
        style.log_axis(axd, "y", grid=True)
        axd.set_ylim(*dlim)
        if j == 0:
            axd.set_ylabel("|log$_{10}$ ratio| to\npsmcfa fit")
        else:
            axd.tick_params(labelleft=False)
            ax.tick_params(labelleft=False)

    handles = [style.handle(style.FORMATS[i], linewidth=2.0 if i == "psmcfa" else style.LW) for i in inputs]
    handles.append(style.handle(style.TRUTH, linewidth=style.LW))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, [*top, *bottom])
    return fig


__all__ = ["draw"]
