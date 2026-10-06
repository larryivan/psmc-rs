"""Fig 2 (E1): accuracy on simulated genomes, and the two tools compared.

* a-d: one panel per model: the true Ne(t) (black, dashed), every replicate's
  fit by C psmc and psmc-rs 0.2 (thin), and the pointwise median of each
  tool's fits (C psmc drawn wider underneath, so that where the medians
  coincide the blue line runs inside an orange one). Times outside the
  model's informative window, where no metric is computed, are shaded. The
  Ne axis covers the truth and the fits inside the windows; fits outside
  them can leave the panel (they are clipped; e-h show them).
* e-h: |log10(Ne_psmc-rs / Ne_C)| over time for every replicate (log scale;
  smaller values are drawn on the axis floor).
* i: RMSE of log10 Ne on the informative window, per replicate: each tool
  against the truth and psmc-rs against C psmc (psmc-rs's M-steps converge
  as C psmc's do; see ``config.Inference``), with the means; and, when E1
  fitted psmc-rs with capped M-steps too (``E1.default_mstep_iters``, not by
  default), that fit against C psmc.
"""

from __future__ import annotations

from typing import Dict, List

from .. import config
from . import style
from .data import Source, check_truth_table, curves, decades, fmt_sig, log10_ratio, median, time_grid, truth_xy

DIFF_FLOOR = 1e-6  # lowest floor of the differences axis
MODEL_LABEL = {"constant": "Constant", "bottleneck": "Bottleneck", "expansion": "Expansion", "zigzag": "Zigzag"}
C_MEDIAN_LW = 2.0  # the wider line underneath
RS_MEDIAN_LW = 0.8


def model_order(models) -> List[str]:
    present = set(models)
    return [m for m in config.MODEL_ORDER if m in present] + sorted(present - set(config.MODEL_ORDER))


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e1
    figs = config.FIGURES
    g = cfg.scaling.gen_years
    df = src.table("E1_curves")
    check_truth_table(src.table("E1_truth"), g)
    windows = {r.model: (r.window_lo_years, r.window_hi_years) for r in src.table("E1_table1").itertuples()}
    reps = src.table("E1_replicates")
    pairs = src.table("E1_rs_vs_c")
    models = model_order(df["model"].unique())
    fits = curves(df, ["model", "tool", "replicate"])
    xlim = figs.sim_years
    grid = time_grid(xlim, figs.curve_grid_points)

    def of(m: str, tool: str):
        return [c for (mm, t, _), c in fits.items() if mm == m and t == tool]

    # Shared Ne range: the truth over the axis, and the 1%-99% range of the
    # fits' values inside each window (a runaway first or last lambda that
    # reaches into a window must not flatten every panel).
    vals: List[float] = []
    for m in models:
        _, ty = truth_xy(m, g, xlim)
        vals += [float(ty.min()), float(ty.max())]
        wgrid = time_grid(windows[m], 200)
        inside = np.concatenate([c.at(wgrid) for c in of(m, "c") + of(m, "rs")])
        vals += [float(np.quantile(inside, 0.01)), float(np.quantile(inside, 0.99))]
    ylim = decades(min(vals) / 1.5, max(vals) * 1.5)

    n = len(models)
    fig = style.figure("double", 150)
    gs = style.grid(fig, 3, n, left_mm=13.0, top_mm=8.5, bottom_mm=6.0, hspace_mm=10.0, wspace_mm=4.5,
                    height_ratios=[1.0, 0.55, 0.8])
    top = [fig.add_subplot(gs[0, j]) for j in range(n)]
    mid = [fig.add_subplot(gs[1, j], sharex=top[j]) for j in range(n)]
    bottom = fig.add_subplot(gs[2, : max(n - 1, 1)])

    diffs: Dict[str, List] = {}
    for j, m in enumerate(models):
        ax = top[j]
        style.shade_outside(ax, windows[m], xlim)
        for tool, ent in (("c", style.C_PSMC), ("rs", style.RS02)):
            for c in of(m, tool):
                x, y = style.step_xy(c.years, c.ne, xlim)
                ax.plot(x, y, drawstyle="steps-post", color=ent.color, linewidth=style.LW_THIN,
                        alpha=style.REPLICATE_ALPHA, zorder=2)
        for tool, ent, lw, z in (("c", style.C_PSMC, C_MEDIAN_LW, 4), ("rs", style.RS02, RS_MEDIAN_LW, 5)):
            ax.plot(grid, median(of(m, tool), grid), drawstyle="steps-post", color=ent.color, linewidth=lw, zorder=z)
        tx, ty = truth_xy(m, g, xlim)
        ax.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=6)
        style.years_axis(ax, xlim, label=False)
        style.ne_axis(ax, ylim, label=style.NE_LABEL if j == 0 else None)
        if j:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelbottom=False)
        style.note(ax, 0.97, 0.97, MODEL_LABEL.get(m, m), ha="right", va="top", color=style.INK, size=style.TEXT_PT)
        diffs[m] = [np.abs(log10_ratio(c, fits[(m, "c", r)], grid))
                    for (mm, t, r), c in fits.items() if mm == m and t == "rs" and (m, "c", r) in fits]

    hi = max(float(np.max(d)) for ds in diffs.values() for d in ds)
    inside = np.concatenate([d[(grid >= windows[m][0]) & (grid <= windows[m][1])] for m, ds in diffs.items() for d in ds])
    positive = inside[inside > 0]
    lo = float(np.quantile(positive, 0.02)) if positive.size else hi / 1e3
    dlim = decades(max(min(lo, hi / 10.0), DIFF_FLOOR), max(hi, DIFF_FLOOR * 10))
    for j, m in enumerate(models):
        ax = mid[j]
        style.shade_outside(ax, windows[m], xlim)
        for d in diffs[m]:
            ax.plot(grid, np.maximum(d, dlim[0]), drawstyle="steps-post", color=style.BLUE, linewidth=style.LW_THIN,
                    alpha=0.8, zorder=3)
        style.years_axis(ax, xlim, label=True)
        style.log_axis(ax, "y", grid=True)
        ax.set_ylim(*dlim)
        if j == 0:
            ax.set_ylabel("|log$_{10}$ ratio|,\npsmc-rs vs C")
        else:
            ax.tick_params(labelleft=False)

    _rmse_panel(bottom, models, reps, pairs, src.table("E1_mstep") if src.has("E1_mstep") else None)
    handles = [
        style.handle(style.TRUTH, linewidth=style.LW),
        style.handle(style.C_PSMC, linewidth=C_MEDIAN_LW, label="C psmc, median"),
        style.handle(style.RS02, linewidth=RS_MEDIAN_LW, label="psmc-rs 0.2, median"),
        style.handle(style.Entity("One replicate (thin)", style.MUTED), linewidth=style.LW_THIN),
    ]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    style.panel_letters(fig, [*top, *mid, bottom])
    return fig


def _rmse_panel(ax, models: List[str], reps, pairs, mstep=None) -> None:
    """RMSE of log10 Ne per replicate: C and psmc-rs against the truth, psmc-rs against C (and a capped M-step fit)."""
    import numpy as np

    groups = [
        ("C psmc vs truth", style.C_PSMC,
         lambda m: reps[(reps.model == m) & (reps.tool == "c")]["rmse_log10_ne_vs_truth"]),
        ("psmc-rs 0.2 vs truth", style.RS02,
         lambda m: reps[(reps.model == m) & (reps.tool == "rs")]["rmse_log10_ne_vs_truth"]),
        ("psmc-rs 0.2 vs C psmc", style.Entity("psmc-rs 0.2 vs C psmc", style.INK, "-", "D"),
         lambda m: pairs[pairs.model == m]["rmse_log10_ne_rs_vs_c"]),
    ]
    if mstep is not None and len(mstep):
        cap = int(mstep["mstep_iters_default"].iloc[0])
        groups.append((f"psmc-rs 0.2, M-steps capped at {cap}, vs C psmc",
                       style.Entity("", style.BLUE_650, "-", "^", filled=False),
                       lambda m: mstep[mstep.model == m]["rmse_log10_ne_default_vs_c"]))
    width = 0.8 / len(groups)
    handles = []
    lows, highs = [], []
    for gi, (label, ent, get) in enumerate(groups):
        for i, m in enumerate(models):
            v = np.asarray(get(m), dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            lows.append(float(v.min()))
            highs.append(float(v.max()))
            x0 = i + (gi - (len(groups) - 1) / 2) * width
            jitter = (np.arange(v.size) - (v.size - 1) / 2) * (0.6 * width / max(v.size, 2))
            ax.plot(x0 + jitter, v, **ent.points(markersize=2.6, alpha=0.85, label=label))
            mean = float(np.mean(v))
            ax.plot([x0 - 0.4 * width, x0 + 0.4 * width], [mean, mean], color=style.INK, linewidth=style.LW,
                    zorder=5)
            if gi >= 2:
                style.note(ax, x0 + 0.45 * width, mean, fmt_sig(mean), ha="left", va="center",
                           transform=ax.transData, size=5.5)
        handles.append(style.handle(ent, kind="points", label=label))
    style.log_axis(ax, "y", grid=True)
    lo = min(lows) if lows else 1e-6
    hi = max(highs) if highs else 1.0
    ax.set_ylim(*decades(lo / 1.5, hi * 1.5))
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=style.TEXT_PT, color=style.INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("RMSE of log$_{10}$ $N_\\mathrm{e}$")
    handles.append(style.handle(style.Entity("Mean", style.INK), linewidth=style.LW))
    style.legend(ax, handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, handlelength=1.2)
