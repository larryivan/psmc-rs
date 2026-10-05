"""Fig 6 (E5): calibration of psmc-rs's bootstrap.

The bottleneck model is simulated as many independent genomes, each fitted
once: their spread at each time is the true sampling distribution of the
estimate. The first genomes are bootstrapped with two trunk sizes (blue
ramp: the default trunk light, the smaller one dark and dashed).

* a: one dataset: the truth, the pointwise 95% range of the independent
  fits (grey), the dataset's fit, and its bootstrap 95% bands;
* b: SD of log10 Ne: the true sampling SD and the bootstrap SD (median over
  datasets, with the interquartile range across datasets);
* c: bootstrap SD / true SD (median over datasets), 1 = calibrated;
* d: share of datasets whose 95% band contains the true Ne(t), with the
  nominal 95%.
"""

from __future__ import annotations

from typing import List

from .. import config
from . import style
from .data import Source, curves, decades, fmt_bp, informative_window, quantiles, time_grid, truth_xy

GREY_BAND = "#c3c2b7"


def trunk_label(trunk_bins: int, bin_size: int) -> str:
    return f"{fmt_bp(trunk_bins * bin_size)} trunks"


def draw(src: Source):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = src.profile.e5
    figs = config.FIGURES
    g = cfg.scaling.gen_years
    bin_size = cfg.inference.bin_size
    cal = src.table("E5_calibration").sort_values(["trunk_bins", "years"])
    cdf = src.table("E5_curves")
    trunks = sorted((int(t) for t in cal["trunk_bins"].unique()), reverse=True)  # default (largest) first
    style_of = {t: (style.TRUNK_COLORS[min(i, 1)], style.TRUNK_STYLES[min(i, 1)], style.TRUNK_MARKERS[min(i, 1)])
                for i, t in enumerate(trunks)}
    xlim = figs.sim_years
    window = informative_window(cfg.model, cfg.grid, g)
    grid = time_grid(xlim, figs.curve_grid_points)

    fig = style.figure("double", 108)
    gs = style.grid(fig, 2, 2, left_mm=13.0, right_mm=2.0, top_mm=12.0, bottom_mm=9.0, wspace_mm=14.0, hspace_mm=10.0)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))

    # a: one dataset.
    fits = curves(cdf[cdf.kind == "fit"], ["dataset"])
    boots = curves(cdf[cdf.kind == "bootstrap"], ["dataset", "trunk_bins", "replicate"])
    ds = figs.e5_example_dataset if (figs.e5_example_dataset,) in fits else min(fits)[0]
    style.shade_outside(ax_a, window, xlim)
    lo, hi = quantiles(list(fits.values()), grid, cfg.band)
    ax_a.fill_between(grid, lo, hi, step="post", color=GREY_BAND, alpha=0.55, linewidth=0, zorder=1)
    for t in trunks:
        mine = [c for (d, tb, _), c in boots.items() if d == ds and tb == t]
        if not mine:
            continue
        blo, bhi = quantiles(mine, grid, cfg.band)
        color, ls, _ = style_of[t]
        if t == trunks[0]:
            ax_a.fill_between(grid, blo, bhi, step="post", color=color, alpha=style.BAND_ALPHA + 0.07, linewidth=0,
                              zorder=2)
        else:
            for edge in (blo, bhi):
                ax_a.plot(grid, edge, drawstyle="steps-post", color=color, linestyle=ls, linewidth=style.LW_THIN + 0.1,
                          zorder=3)
    c = fits[(ds,)]
    x, y = style.step_xy(c.years, c.ne, xlim)
    ax_a.plot(x, y, drawstyle="steps-post", color=style.BLUE, linewidth=style.LW, zorder=4)
    tx, ty = truth_xy(cfg.model, g, xlim)
    ax_a.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=5)
    inside = (grid >= window[0]) & (grid <= window[1])
    vals = [float(lo[inside].min()), float(hi[inside].max()), float(ty.min()), float(ty.max())]
    style.years_axis(ax_a, xlim)
    style.ne_axis(ax_a, decades(min(vals) / 1.3, max(vals) * 1.3))
    style.note(ax_a, 0.97, 0.03, f"dataset {ds}", ha="right", va="bottom")

    # b-d: calibration on the window's grid.
    for t in trunks:
        sub = cal[cal.trunk_bins == t]
        color, ls, marker = style_of[t]
        yrs = sub["years"].to_numpy()
        ax_b.fill_between(yrs, sub["sd_boot_q25"], sub["sd_boot_q75"], color=color, alpha=style.BAND_ALPHA,
                          linewidth=0, zorder=1)
        for ax, col in ((ax_b, "sd_boot_median"), (ax_c, "sd_ratio_median"), (ax_d, "coverage")):
            ax.plot(yrs, sub[col], color=color, linestyle=ls, linewidth=style.LW, zorder=3)
            every = max(len(yrs) // 6, 1)
            ax.plot(yrs[every // 2::every], sub[col].to_numpy()[every // 2::every],
                    **style.Entity("", color, ls, marker).points(markersize=2.8, zorder=4, label=None))
    first = cal[cal.trunk_bins == trunks[0]]
    ax_b.plot(first["years"], first["sd_true"], color=style.INK, linewidth=style.LW, zorder=2)
    ax_b.set_ylim(bottom=0.0)
    ax_b.set_ylabel("SD of log$_{10}$ $N_\\mathrm{e}$")
    style.ref_line(ax_c, y=1.0)
    ax_c.set_ylabel("Bootstrap SD / true SD")
    ax_c.set_ylim(0.0, max(1.5, float(cal["sd_ratio_median"].max()) * 1.1))
    nominal = cfg.band[1] - cfg.band[0]
    style.ref_line(ax_d, y=nominal, linestyle=style.DOT)
    ax_d.set_ylim(-0.03, 1.03)
    ax_d.yaxis.set_major_formatter(_percent())
    n = int(first["n_datasets"].iloc[0])
    ax_d.set_ylabel(f"Coverage of true $N_\\mathrm{{e}}$ ({n} datasets)")
    # The calibration is computed on the informative window only.
    wlim = (window[0] / 1.25, window[1] * 1.25)
    for ax in (ax_b, ax_c, ax_d):
        style.years_axis(ax, wlim)
        ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    style.note(ax_d, 0.98, 0.03, f"dotted: nominal {100 * nominal:.0f}%", ha="right", va="bottom")

    handles: List = [
        style.handle(style.TRUTH, linewidth=style.LW),
        Patch(facecolor=GREY_BAND, alpha=0.55, linewidth=0,
              label=f"{len(fits)} independent fits, 95% range"),
        Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Fit of one dataset"),
        Line2D([], [], color=style.INK, linewidth=style.LW, label="True sampling SD"),
    ]
    for t in trunks:
        color, ls, marker = style_of[t]
        handles.append(Line2D([], [], color=color, linestyle=ls, linewidth=style.LW, marker=marker, markersize=2.8,
                              markerfacecolor=color, markeredgecolor=color,
                              label=f"Bootstrap, {trunk_label(t, bin_size)}"))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style.panel_letters(fig, [ax_a, ax_b, ax_c, ax_d])
    return fig


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")
