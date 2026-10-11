"""Fig S1 (E1): EM convergence of C psmc and psmc-rs (0.2) on the same genomes.

One column per model; every replicate is a line. Iteration i counts
completed EM iterations: C psmc's LK of round i + 1 and psmc-rs's
``loglike[i]`` are both the log-likelihood after i iterations. C psmc
prints none after its last one, so the log-likelihoods of both tools are
drawn up to the last iteration both give (psmc-rs's after the last
iteration is in its JSON).

* a-d: log-likelihood gained since the start, LL(i) - LL(0), for both tools
  (C psmc wider, underneath);
* e-h: psmc-rs's log-likelihood minus C psmc's after the same iteration
  (log-likelihood units; iteration 0 evaluates the same starting model);
* i-l: theta per bin of psmc-rs relative to C psmc's after each iteration,
  in percent. C psmc prints theta to 6 decimals, so its theta is known to
  +-0.5e-6 only: the grey band is that rounding, relative to C psmc's
  smallest theta of the model at each iteration, and the steps of the
  lines are its quanta (about 0.004% for the expansion model, whose theta
  is smallest).
"""

from __future__ import annotations

from typing import List

from . import style
from .data import Source
from .fig2 import MODEL_LABEL, model_order

C_LW, RS_LW = 1.6, 0.6
C_THETA_DECIMALS = 6  # C psmc prints theta with "%.6lf"
ROUNDING = "#deddd5"  # the band of C psmc's rounding of theta (i-l)


def _symmetric(ax, values, floor: float) -> None:
    import numpy as np

    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    lim = max(float(np.max(np.abs(v))) if v.size else 0.0, floor) * 1.15
    ax.set_ylim(-lim, lim)


def draw(src: Source):
    import numpy as np

    df = src.table("E1_traces").sort_values(["model", "replicate", "tool", "iteration"])
    models = model_order(df["model"].unique())
    n = len(models)
    n_iter = int(df["iteration"].max())
    fig = style.figure("double", 118)
    gs = style.grid(fig, 3, n, left_mm=15.0, top_mm=9.0, bottom_mm=9.0, wspace_mm=8.0, hspace_mm=5.5)
    rows: List[List] = [[fig.add_subplot(gs[r, j]) for j in range(n)] for r in range(3)]
    for r in (1, 2):
        for j in range(n):
            rows[r][j].sharex(rows[0][j])

    lims = {"ll": [], "theta": []}
    for j, m in enumerate(models):
        sub = df[df.model == m]
        ax_g, ax_l, ax_t = rows[0][j], rows[1][j], rows[2][j]
        lims["gain"] = []
        for rep, rsub in sub.groupby("replicate"):
            by = {t: s.set_index("iteration") for t, s in rsub.groupby("tool")}
            if "c" not in by or "rs" not in by:
                continue
            # Both tools over the iterations both give a log-likelihood for (C psmc prints none after its last).
            common = by["c"]["loglike"].dropna().index.intersection(by["rs"]["loglike"].dropna().index)
            for tool, ent, lw, z in (("c", style.C_PSMC, C_LW, 2), ("rs", style.RS02, RS_LW, 3)):
                ll = by[tool]["loglike"].loc[common]
                gain = ll - ll.loc[0]
                ax_g.plot(gain.index, gain.values, color=ent.color, linewidth=lw, zorder=z)
                lims["gain"] += list(gain.values)
            d = by["rs"]["loglike"].loc[common] - by["c"]["loglike"].loc[common]
            ax_l.plot(common, d.values, color=style.BLUE, linewidth=style.LW_THIN + 0.2, marker="o", markersize=1.2,
                      zorder=3)
            lims["ll"] += list(d.values)
            th = by["rs"]["theta"] / by["c"]["theta"] - 1.0
            th = th.dropna()
            ax_t.plot(th.index, 100.0 * th.values, color=style.BLUE, linewidth=style.LW_THIN + 0.2, marker="o",
                      markersize=1.2, zorder=3)
            lims["theta"] += list(100.0 * th.values)
        # The rounding of C psmc's printed theta, relative to its smallest theta at each iteration.
        theta_c = sub[sub.tool == "c"].groupby("iteration")["theta"].min()
        half = 100.0 * 0.5 * 10.0 ** -C_THETA_DECIMALS / theta_c
        ax_t.fill_between(half.index, -half.values, half.values, color=ROUNDING, linewidth=0, zorder=1)
        lims["theta"] += list(half.values)
        style.note(ax_g, 0.97, 0.05, MODEL_LABEL.get(m, m), ha="right", va="bottom", color=style.INK)
        # The gain depends on the model: each panel of the first row has its own scale.
        v = np.asarray(lims["gain"], dtype=float)
        if v.size:
            ax_g.set_ylim(min(0.0, float(np.nanmin(v))) * 1.05, max(float(np.nanmax(v)), 1e-9) * 1.1)
        for ax in (ax_l, ax_t):
            style.ref_line(ax, y=0.0)
    for j in range(n):
        for r, key in enumerate(("gain", "ll", "theta")):
            ax = rows[r][j]
            ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
            ax.set_xlim(-0.5, n_iter + 0.5)
            if r:
                _symmetric(ax, lims[key], 1e-6)
            if j and r:
                ax.tick_params(labelleft=False)
            if r < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("EM iteration")
    rows[0][0].set_ylabel("Log-likelihood gain,\nLL(i) − LL(0)")
    rows[1][0].set_ylabel("Log-likelihood,\npsmc-rs − C psmc")
    rows[2][0].set_ylabel("θ per bin, psmc-rs\nrelative to C psmc (%)")
    fig.align_ylabels([row[0] for row in rows])
    from matplotlib.patches import Patch

    handles = [style.handle(style.C_PSMC, linewidth=C_LW),
               style.handle(style.Entity("psmc-rs", style.BLUE, "-", "o"), linewidth=RS_LW),
               style.handle(style.Entity("psmc-rs − C psmc (one replicate)", style.BLUE, "-", "o"),
                        kind="both", linewidth=style.LW_THIN + 0.2, markersize=1.6),
               Patch(facecolor=ROUNDING, linewidth=0,
                     label=f"Rounding of C psmc's printed θ ({C_THETA_DECIMALS} decimals; i\u2013l)")]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    style.panel_letters(fig, [ax for row in rows for ax in row], align_cols=True)
    return fig
