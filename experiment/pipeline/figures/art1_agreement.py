"""The article's Fig 1: psmc-rs reproduces C psmc, on simulated and on real genomes.

* a-d (E1): one panel per demographic model, on its informative window
  (1% to 99% of the true pairwise coalescence times): the true Ne(t)
  (dashed ink), the pointwise median of the ten replicates' fits by C psmc
  (orange, wide, underneath) and by psmc-rs (blue, on top), and the range
  of psmc-rs's ten fits (light blue): the curves of the two programs
  coincide, and the band shows how much an estimate varies between
  genomes with the same history. One Ne axis for the row.
* e-h (E2): one panel per real genome (``config.E2.panel_order``), on the
  window on which the programs are compared (1% to 99% of pairwise
  coalescence times under C psmc's fit): C psmc's fit (orange, wide),
  psmc-rs's (blue) and the pointwise 95% band of psmc-rs's bootstrap at
  the default block size (light blue), with the RMSE between the programs
  (``E2_rs_vs_c``). Each genome is scaled with its own mu and g
  (``config.RealGenome``).
* i: every interval inside its window of every genome both programs
  fitted (E1's 40, E2's four, E7's default set; NA12878, in E2 and in E7,
  whose fits are the same, once): the Ne of psmc-rs against that of C
  psmc, interval k against interval k (the identity line in grey), and,
  inset, the distribution of |log10(Ne_psmc-rs / Ne_C)| over those
  intervals (:func:`agreement_intervals`).
* j: RMSE of log10 Ne over the window, one point per genome: each program
  against the truth (E1; orange and blue), and psmc-rs against C psmc
  interval by interval (``metrics.interval_diff``) on E1, E2 and E7's
  default set (ink); bars are medians. The programs differ by three to
  four orders of magnitude less than either errs.

Everything comes from the tables ``E1_curves``, ``E1_table1``,
``E1_replicates``, ``E1_rs_vs_c``, ``E2_curves``, ``E2_bands``,
``E2_rs_vs_c``, ``E7_curves`` and ``E7_rs_vs_c``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .. import config
from . import style
from .data import Source, curves, decades, median, time_grid, truth_xy

H_MM = 176.0
ROW_A_MM, ROW_B_MM, ROW_C_MM = 34.0, 34.0, 52.0
LEFT_MM, RIGHT_MM, TOP_MM = 13.0, 3.0, 12.0
GAP_X_MM, GAP_ROW_MM = 6.0, 17.0
J_GAP_MM, J_LABEL_MM = 11.0, 24.0  # from i's right edge to j's labels; j's label column
C_LW, RS_LW = 2.2, 0.9  # C psmc wide underneath, psmc-rs thin on top
MODEL_NAMES = {"constant": "Constant", "bottleneck": "Bottleneck", "expansion": "Expansion", "zigzag": "Zigzag"}
GROUPS = (("E1", "Simulated"), ("E2", "Turtles and\nNA12878"), ("E7", "1000\nGenomes"))


def axes_mm(fig, x: float, y: float, w: float, h: float):
    """Axes whose top-left corner is ``x`` mm right and ``y`` mm down from the figure's, ``w`` x ``h`` mm."""
    fw, fh = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    return fig.add_axes((x / fw, 1.0 - (y + h) / fh, w / fw, h / fh))


def paired_intervals(c, rs, window: Tuple[float, float]) -> List[Tuple[float, float]]:
    """``(Ne_C, Ne_rs)`` of every interval of C psmc's fit ``c`` that overlaps ``window``, with psmc-rs's same interval."""
    out = []
    lo, hi = window
    for k, (start, ne_c) in enumerate(zip(c.years, c.ne)):
        end = c.years[k + 1] if k + 1 < len(c.years) else math.inf
        if end > lo and start < hi and k < len(rs.ne):
            out.append((ne_c, rs.ne[k]))
    return out


def agreement_intervals(e1c, win1, e2c, win2, e7c, win7) -> Tuple[Dict[str, List[Tuple[float, float]]], int]:
    """Panel i's ``(Ne_C, Ne_rs)`` pairs by experiment, and the number of genomes they come from.

    A genome of E7's default set that E2 also fits (NA12878: the same psmcfa
    and settings, so the same fits) is counted once, in E2.
    """
    pairs: Dict[str, List[Tuple[float, float]]] = {"E1": [], "E2": [], "E7": []}
    genomes = 0
    for (m, tool, r), c in sorted(e1c.items()):
        if tool == "c" and (m, "rs", r) in e1c and m in win1:
            pairs["E1"] += paired_intervals(c, e1c[(m, "rs", r)], win1[m])
            genomes += 1
    for sid in sorted(win2):
        if (sid, "c") in e2c and (sid, "rs") in e2c:
            pairs["E2"] += paired_intervals(e2c[(sid, "c")], e2c[(sid, "rs")], win2[sid])
            genomes += 1
    for sid in sorted(win7):
        if sid not in win2 and (sid, "c") in e7c and (sid, "rs-psmcfa") in e7c:
            pairs["E7"] += paired_intervals(e7c[(sid, "c")], e7c[(sid, "rs-psmcfa")], win7[sid])
            genomes += 1
    return pairs, genomes


def _sci(x: float) -> str:
    """``x`` as mathtext, two significant digits: 1.2 x 10^-5."""
    if not x > 0:
        return "0"
    e = math.floor(math.log10(x))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return r"$%.1f\times10^{%d}$" % (m, e)


def _steps(ax, grid, values, **kw):
    ax.plot(grid, values, drawstyle="steps-post", solid_capstyle="butt", **kw)


def _title(ax, text: str) -> None:
    """The panel's name above its axes, left."""
    style.note(ax, 0.0, 1.03, text, va="bottom", color=style.INK, size=style.TEXT_PT)


def _row_label(fig, axes, text: str, below_mm: float = 7.5) -> None:
    """One x label centred under a row of panels."""
    fh = fig.get_figheight() / style.MM
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    y = axes[0].get_position().y0 - below_mm / fh
    fig.text((x0 + x1) / 2.0, y, text, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK)


def _x_axis(ax, lim, label: bool) -> None:
    style.years_axis(ax, lim, label=label)
    ax.grid(False)


def draw(src: Source):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    from matplotlib.transforms import blended_transform_factory

    prof = src.profile
    g1 = prof.e1.scaling.gen_years
    e1c = curves(src.table("E1_curves"), ["model", "tool", "replicate"])
    win1 = {r.model: (float(r.window_lo_years), float(r.window_hi_years)) for r in src.table("E1_table1").itertuples()}
    models = [m for m in prof.e1.models if m in win1]
    e2c = curves(src.table("E2_curves"), ["sample_id", "tool"])
    bands = src.table("E2_bands")
    rsc2 = src.table("E2_rs_vs_c")
    win2 = {r.sample_id: (float(r.window_lo_years), float(r.window_hi_years)) for r in rsc2.itertuples()}
    genomes = {g.sample_id: g for g in prof.e2.genomes}
    order2 = [s for s in prof.e2.panel_order if s in win2] + sorted(s for s in win2 if s not in prof.e2.panel_order)
    e7c = curves(src.table("E7_curves"), ["sample_id", "route"])
    rsc7 = src.table("E7_rs_vs_c")
    win7 = {r.sample_id: (float(r.window_lo_years), float(r.window_hi_years)) for r in rsc7.itertuples()}

    fig = style.figure("double", H_MM)
    w_mm = fig.get_figwidth() / style.MM
    plot_w = w_mm - LEFT_MM - RIGHT_MM
    pw = (plot_w - 3 * GAP_X_MM) / 4.0
    y_a = TOP_MM
    y_b = y_a + ROW_A_MM + GAP_ROW_MM
    y_c = y_b + ROW_B_MM + GAP_ROW_MM
    row_a = [axes_mm(fig, LEFT_MM + i * (pw + GAP_X_MM), y_a, pw, ROW_A_MM) for i in range(4)]
    row_b = [axes_mm(fig, LEFT_MM + i * (pw + GAP_X_MM), y_b, pw, ROW_B_MM) for i in range(4)]
    ax_i = axes_mm(fig, LEFT_MM, y_c, ROW_C_MM, ROW_C_MM)
    x_labels = LEFT_MM + ROW_C_MM + J_GAP_MM
    ax_j = axes_mm(fig, x_labels + J_LABEL_MM, y_c, plot_w - ROW_C_MM - J_GAP_MM - J_LABEL_MM, ROW_C_MM)

    # a-d: the four models.
    vals: List[float] = []
    drawn = {}
    for m in models:
        w = win1[m]
        grid = time_grid(w, 500)
        reps = sorted(r for (mm, tool, r) in e1c if mm == m and tool == "rs")
        rs_curves = [e1c[(m, "rs", r)] for r in reps]
        c_curves = [e1c[(m, "c", r)] for r in sorted(r for (mm, tool, r) in e1c if mm == m and tool == "c")]
        if not rs_curves or not c_curves:
            continue
        on = np.vstack([c.at(grid) for c in rs_curves])
        lo_band, hi_band = on.min(axis=0), on.max(axis=0)
        med_c, med_rs = median(c_curves, grid), median(rs_curves, grid)
        tx, ty = truth_xy(m, g1, w)
        drawn[m] = (grid, lo_band, hi_band, med_c, med_rs, tx, ty)
        vals += list(lo_band) + list(hi_band) + list(ty)
    ylim_a = decades(min(vals) / 1.15, max(vals) * 1.15) if vals else (1e3, 1e5)
    for i, m in enumerate(models):
        ax = row_a[i]
        if m not in drawn:
            continue
        grid, lo_band, hi_band, med_c, med_rs, tx, ty = drawn[m]
        ax.fill_between(grid, lo_band, hi_band, step="post", color=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
                        zorder=1)
        _steps(ax, grid, med_c, color=style.ORANGE, linewidth=C_LW, zorder=3)
        _steps(ax, grid, med_rs, color=style.BLUE, linewidth=RS_LW, zorder=4)
        ax.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=5)
        _x_axis(ax, win1[m], label=False)
        style.ne_axis(ax, ylim_a, label=style.NE_LABEL_SHORT if i == 0 else None, grid=False)
        _title(ax, MODEL_NAMES.get(m, m))

    # e-h: the real genomes.
    trunk = prof.e2.bootstrap_trunk_bins
    for i, sid in enumerate(order2[:4]):
        ax = row_b[i]
        w = win2[sid]
        grid = time_grid(w, 600)
        c, rs = e2c.get((sid, "c")), e2c.get((sid, "rs"))
        b = bands[(bands.sample_id == sid) & (bands.trunk_bins == trunk)].sort_values("years")
        b = b[(b.years >= w[0]) & (b.years <= w[1])]
        v: List[float] = []
        if not b.empty:
            ax.fill_between(b.years, b.ne_band_lo, b.ne_band_hi, color=style.BLUE, alpha=style.BAND_ALPHA,
                            linewidth=0, zorder=1)
            v += list(b.ne_band_lo) + list(b.ne_band_hi)
        if c is not None:
            yc = c.at(grid)
            _steps(ax, grid, yc, color=style.ORANGE, linewidth=C_LW, zorder=3)
            v += list(yc)
        if rs is not None:
            _steps(ax, grid, rs.at(grid), color=style.BLUE, linewidth=RS_LW, zorder=4)
        _x_axis(ax, w, label=False)
        ylim = decades(min(v) / 1.25, max(v) * 1.6) if v else (1e3, 1e6)
        style.ne_axis(ax, ylim, label=style.NE_LABEL_SHORT if i == 0 else None, grid=False)
        gnm = genomes.get(sid)
        name = r"$\mathit{%s}$" % gnm.species.replace(" ", r"\ ") if gnm else sid
        _title(ax, f"{name} ({sid})" if sid == "NA12878" else name)
        rmse = rsc2[rsc2.sample_id == sid].rmse_log10_ne_rs_vs_c_per_interval
        if not rmse.empty:
            style.note(ax, 0.97, 0.04, f"RMSE {_sci(float(rmse.iloc[0]))}", ha="right", va="bottom",
                       size=style.TICK_PT)
    for ax in row_a + row_b:
        ax.set_xlabel("")
    _row_label(fig, row_a, "Years ago")
    _row_label(fig, row_b, "Years ago")

    # i: every interval of every genome both programs fitted.
    pairs, n_fits = agreement_intervals(e1c, win1, e2c, win2, e7c, win7)
    allp = np.array([p for k in pairs for p in pairs[k]] or [(1e4, 1e4)])
    lo = decades(allp.min() / 1.3, allp.max() * 1.3)
    ax_i.plot(lo, lo, color=style.MUTED, linewidth=style.LW_REF, zorder=1)
    ax_i.scatter(allp[:, 0], allp[:, 1], s=2.0, color=style.INK, alpha=0.35, linewidths=0, zorder=3, rasterized=True)
    for which in ("x", "y"):
        style.log_axis(ax_i, which)
    ax_i.set_xlim(lo)
    ax_i.set_ylim(lo)
    ax_i.set_xlabel(r"$N_\mathrm{e}$, C psmc")
    ax_i.set_ylabel(r"$N_\mathrm{e}$, psmc-rs")
    style.note(ax_i, 0.04, 0.97, f"{len(allp):,} intervals\nof {n_fits} genomes", size=style.TICK_PT)
    inset = ax_i.inset_axes((0.55, 0.17, 0.40, 0.27))
    d = np.abs(np.log10(allp[:, 1] / allp[:, 0]))
    d = d[d > 0]
    bins = np.logspace(-9, -3, 31)
    inset.hist(d, bins=bins, color=style.INK2, linewidth=0)
    inset.set_xscale("log")
    inset.set_xlim(1e-9, 1e-3)
    inset.set_xticks([1e-9, 1e-6, 1e-3])
    inset.xaxis.set_minor_locator(NullLocator())
    inset.tick_params(labelsize=style.TICK_PT, length=2, pad=1)
    inset.set_yticks([])
    for s in ("left", "top", "right"):
        inset.spines[s].set_visible(False)
    inset.set_xlabel(r"|$\Delta\log_{10}N_\mathrm{e}$|", fontsize=style.TICK_PT, labelpad=1)
    inset.patch.set_alpha(0.0)

    # j: the programs' difference against their error, one row per comparison.
    rep = src.table("E1_replicates")
    e1d = src.table("E1_rs_vs_c")
    groups = (
        ("Against the truth (simulated genomes)", (
            ("C psmc", rep[rep.tool == "c"].rmse_log10_ne_vs_truth, style.ORANGE, "s"),
            ("psmc-rs", rep[rep.tool == "rs"].rmse_log10_ne_vs_truth, style.BLUE, "o"),
        )),
        ("psmc-rs against C psmc", (
            ("Simulated", e1d.rmse_log10_ne_rs_vs_c_per_interval, style.INK, "D"),
            ("Turtles and NA12878", rsc2.rmse_log10_ne_rs_vs_c_per_interval, style.INK, "D"),
            ("1000 Genomes", rsc7.rmse_log10_ne_rs_vs_c_per_interval, style.INK, "D"),
        )),
    )
    fw = fig.get_figwidth() / style.MM
    header_at = blended_transform_factory(fig.transFigure, ax_j.transData)
    rng = np.random.default_rng(7)
    y, ticks, names, medians = 0.0, [], [], {}
    for header, rows in groups:
        ax_j.text(x_labels / fw, y, header, transform=header_at, ha="left", va="center", fontsize=style.TEXT_PT,
                  color=style.INK, bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 1.0})
        for name, v, color, marker in rows:
            y += 1.0
            v = np.asarray(v, dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if not v.size:
                continue
            ax_j.scatter(v, y + (rng.random(v.size) - 0.5) * 0.4, s=8, color=color, alpha=0.75, linewidths=0,
                         marker=marker, zorder=3)
            medians[name] = (float(np.median(v)), y)
            ax_j.plot([medians[name][0]] * 2, [y - 0.32, y + 0.32], color=style.INK, linewidth=style.LW_BOLD,
                      zorder=4)
            ticks.append(y)
            names.append(name)
        y += 1.5
    ylim = (ticks[-1] + 0.6, -0.55)
    # the gap between psmc-rs's error and its difference from C psmc, on the simulated genomes
    if "psmc-rs" in medians and "Simulated" in medians:
        (err, y_err), (dif, y_dif) = medians["psmc-rs"], medians["Simulated"]
        ya = y_err + 0.85
        for x, y0 in ((err, y_err), (dif, y_dif)):
            ax_j.plot([x, x], [y0, ya], color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=2)
        ax_j.annotate("", xy=(dif, ya), xytext=(err, ya), arrowprops={
            "arrowstyle": "<->", "color": style.INK2, "linewidth": style.LW_REF, "shrinkA": 0, "shrinkB": 0,
            "mutation_scale": 6})
        fold = err / dif
        mag = 10 ** math.floor(math.log10(fold))
        ax_j.text(math.sqrt(err * dif), ya - 0.12, f"{round(fold / mag) * mag:,.0f}-fold (medians)", ha="center",
                  va="bottom", fontsize=style.TICK_PT, color=style.INK2)
    style.log_axis(ax_j, "x")
    ax_j.set_xlim(1e-6, 1.0)
    ax_j.set_ylim(ylim)
    ax_j.set_yticks(ticks)
    ax_j.set_yticklabels(names)
    ax_j.tick_params(axis="y", length=0, pad=3)
    ax_j.spines["left"].set_visible(False)
    ax_j.set_xlabel(r"RMSE of $\log_{10}N_\mathrm{e}$")
    ax_j.grid(True, which="major", axis="x", color=style.GRID, linewidth=style.LW_GRID)

    handles = [
        Line2D([], [], color=style.INK, linestyle=style.DASH, linewidth=style.LW, label="Truth"),
        Line2D([], [], color=style.ORANGE, linewidth=C_LW, label="C psmc"),
        Line2D([], [], color=style.BLUE, linewidth=RS_LW, label="psmc-rs"),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label="a–d: range of the replicates; e–h: bootstrap 95% band"),
    ]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=1.6,
                 handlelength=2.4)
    style.panel_letters(fig, row_a + row_b + [ax_i, ax_j])
    return fig


__all__ = ["agreement_intervals", "draw", "paired_intervals"]
