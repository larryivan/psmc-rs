"""Fig 3 (E2): four real genomes fitted by C psmc and psmc-rs 0.2, with psmc-rs's bootstrap.

One panel per genome: the pointwise 95% percentile band of psmc-rs's
bootstrap replicates (blue, light), C psmc's fit (orange, wider, underneath)
and psmc-rs's fit (blue), each scaled with the species' mu and g (given in
the panel). Times outside the window on which the tools' RMSE is computed
are shaded; the RMSE is given in the panel. The human sample is chromosome
1 only, and the panel says so.
"""

from __future__ import annotations

from typing import Dict, List

from .. import config
from . import style
from .data import Source, curve_span, curves, decades, fmt_sig, quantiles, time_grid

C_LW, RS_LW = 2.0, 0.8


def _genomes(present) -> List[config.RealGenome]:
    known = [g for g in config.REAL_GENOMES if g.sample_id in set(present)]
    return known


def _sample_note(gnm: config.RealGenome, mu: float, g: float) -> str:
    where = gnm.label[len(gnm.species):].strip(" ()") if gnm.label.startswith(gnm.species) else gnm.sample_id
    mant, exp = f"{mu:.2e}".split("e")
    return rf"{where}; $\mu$ = {float(mant):g}$\times$10$^{{{int(exp)}}}$, g = {g:g} y"


def draw(src: Source):
    import numpy as np
    from matplotlib.patches import Patch

    figs = config.FIGURES
    cfg = src.profile.e2
    df = src.table("E2_curves")
    summary = {r.sample: r for r in src.table("E2_summary").itertuples()}
    genomes = _genomes(df["sample"].unique())
    main = curves(df[df.replicate == 0], ["sample", "tool"])
    boots = curves(df[(df.replicate > 0) & (df.tool == "rs")], ["sample", "replicate"])
    xlim = decades(*curve_span(main.values()))
    grid = time_grid(xlim, figs.curve_grid_points)

    bands: Dict[str, np.ndarray] = {}
    vals: List[float] = []
    for gnm in genomes:
        span = curve_span([main[(gnm.sample_id, t)] for t in ("c", "rs") if (gnm.sample_id, t) in main])
        inside = (grid >= span[0]) & (grid <= span[1])
        mine = [c for (s, _), c in boots.items() if s == gnm.sample_id]
        if mine:
            bands[gnm.sample_id] = quantiles(mine, grid, cfg.band)
            vals += [float(np.min(bands[gnm.sample_id][0][inside])), float(np.max(bands[gnm.sample_id][1][inside]))]
        for t in ("c", "rs"):
            if (gnm.sample_id, t) in main:
                ne = main[(gnm.sample_id, t)].at(grid[inside])
                vals += [float(ne.min()), float(ne.max())]
    ylim = decades(min(vals) / 1.3, max(vals) * 1.3)

    fig = style.figure("double", 112)
    nrow = (len(genomes) + 1) // 2
    gs = style.grid(fig, nrow, 2, left_mm=13.0, top_mm=7.5, bottom_mm=8.5, hspace_mm=6.0, wspace_mm=5.0)
    axes = []
    for i, gnm in enumerate(genomes):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        axes.append(ax)
        s = summary.get(gnm.sample_id)
        mu = float(s.mu) if s is not None else gnm.scaling.mu
        g = float(s.gen_years) if s is not None else gnm.scaling.gen_years
        if s is not None:
            style.shade_outside(ax, (float(s.window_lo_years), float(s.window_hi_years)), xlim)
        if gnm.sample_id in bands:
            lo, hi = bands[gnm.sample_id]
            ax.fill_between(grid, lo, hi, step="post", color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, zorder=2)
        for tool, ent, lw, z in (("c", style.C_PSMC, C_LW, 4), ("rs", style.RS02, RS_LW, 5)):
            c = main.get((gnm.sample_id, tool))
            if c is None:
                continue
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linewidth=lw, zorder=z)
        style.years_axis(ax, xlim, label=i // 2 == nrow - 1)
        style.ne_axis(ax, ylim, label=style.NE_LABEL if i % 2 == 0 else None)
        if i % 2:
            ax.tick_params(labelleft=False)
        if i // 2 != nrow - 1:
            ax.tick_params(labelbottom=False)
        style.note(ax, 0.03, 0.97, gnm.species, color=style.INK, size=style.TEXT_PT, fontstyle="italic")
        style.note(ax, 0.03, 0.885, _sample_note(gnm, mu, g))
        if s is not None:
            n_boot = int(s.bootstrap_replicates)
            style.note(ax, 0.97, 0.03,
                       f"RMSE psmc-rs vs C: {fmt_sig(float(s.rmse_log10_ne_rs_vs_c))}\n{n_boot} bootstrap replicates",
                       ha="right", va="bottom")
    handles = [
        style.handle(style.C_PSMC, linewidth=C_LW),
        style.handle(style.RS02, linewidth=RS_LW),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"psmc-rs bootstrap, {100 * (cfg.band[1] - cfg.band[0]):.0f}% band"),
    ]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style.panel_letters(fig, axes)
    return fig
