"""Supplement Fig S1 (E1): C psmc and psmc-rs climb the same likelihood at every EM iteration.

One row of panels, one per simulated model of E1, in config.MODEL_ORDER
(Constant, Bottleneck, Expansion, Zigzag), every genome (replicate) of the
model in each. All panels share one log axis of log-likelihood differences:

* the upper bundle is the log-likelihood gained in iteration i,
  LL(i) - LL(i - 1), one line per genome and program, drawn by the underlay
  rule: C psmc a wide pale-orange underlay, psmc-rs a thin blue line on top
  (the psmc-rs lines read as one blue ribbon with an orange rim). A gain
  can only be drawn on a log axis if it is positive, so the bundle drawn at
  all is the claim that the log-likelihood never decreased; a gain that is
  not positive raises :class:`ValueError` instead of vanishing from the axis;
* the grey line is the largest difference between the two programs'
  log-likelihoods after iteration i, max over the genomes of
  |LL_rs(i) - LL_c(i)| (a difference of exactly 0, possible at C psmc's six
  printed decimals, is left out of the line);
* a double arrow, just right of the iteration where the two come closest
  (the smallest gain of any genome in that iteration over the grey line's
  value), is labelled with that ratio floored to two significant figures
  (``84×``): at every iteration the programs differed at least that many
  times less than the iteration gained.

y is shared (tick labels and its label on panel a only), and so is x, with
one label under the row; the key (C psmc, psmc-rs) is one line at the top.
The y hairlines lie at the labelled decades strictly inside the axes (one at
the top would read as a rule under the titles, the bottom one would lie under
the spine) and span the data only, so the arrows' column is free of grid.

Panel a names the bundle (just above it) and the grey line (just below it)
in place, both names starting at iteration 1. Iteration i counts completed
EM iterations: C psmc's LK of round i + 1 and psmc-rs's ``loglike[i]`` are
both the log-likelihood after i iterations. C psmc prints none after its
last iteration, so both programs are drawn up to the last iteration both
give. Iteration 0 (the same starting model in both, agreeing to C psmc's
printed precision), the sign of the difference and theta and rho are left
to the caption.

:func:`series`, :func:`ratio_floor` and :func:`ratio_label` compute what
the panels draw; the manuscript's numbers (make_numbers.py) use them too, so
the caption's bound (the smallest :func:`ratio_floor` of the arrows' ratios)
cannot drift from the arrows'.
"""

from __future__ import annotations

import math
from typing import Dict

from . import style
from .data import Source
from .fig2 import MODEL_LABEL, model_order

WIDTH_MM, HEIGHT_MM = style.WIDTH_MM["double"], 52.0
# Between columns: the y axis is shared (tick labels on the first panel
# only), but each panel's letter sits in the gutter left of its title, so the
# gutter is wider than style.GUTTER_SHARED_MM (3 mm would leave a letter
# touching the panel to its left) and narrower than style.GUTTER_MM.
GUTTER_MM = 5.0
# Right of the last iteration, in mm (whatever the number of iterations): room
# for the arrow, the arrow's offset from its iteration and its label's gap.
X_PAD_MM = 2.4
ARROW_DX_MM = 1.4
LABEL_DX_MM = 0.9
# The arrow's ends stop 1.2 pt short of the two lines' values: the upper tip level with the lower edge of
# the 2.4 pt underlay (half of 2.4 pt below the lowest gain), the lower one just clear of the grey line.
ARROW_SHRINK_PT = 1.2
LABEL_X = 1.0  # panel a: both names start at the first iteration, one left edge
GAIN_LABEL_RISE = 1.6  # panel a: the bundle's name, this factor above its highest point under the name
# Panel a: the grey line's name, this factor below its lowest point under the name: its top about 1 mm under
# the line, near enough to read as its name and still clear of the 10^-2 hairline the line runs along.
GAP_LABEL_DROP = 2.0
LABEL_DECADES = 0.6  # the height of a 6 pt label on the y axis, in decades, with room to spare
LINE = {"solid_joinstyle": "round"}  # round joins: no miter spikes where the 2.4 pt underlay bends sharply
# Round caps on the underlays only: ten 2.4 pt butt ends cut across lines of different slopes leave a notched
# edge and pale tabs at i = 1; round ones close the rim around the blue ribbon's ends (0.42 mm past the first and
# last iteration, clear of the arrow).
UNDER = dict(LINE, solid_capstyle="round")


def ratio_floor(r: float) -> float:
    """``r`` floored to two significant figures: 84.5 is 84, 143.6 is 140, 299.1 is 290.

    Floored, never rounded, so the number is a bound ("at least 84 times").
    """
    if not (r > 0 and math.isfinite(r)):
        raise ValueError(f"no ratio label for {r!r}")
    mag = 10.0 ** (math.floor(math.log10(r)) - 1)
    return round(math.floor(r / mag * (1.0 + 1e-12)) * mag, 12)


def ratio_label(r: float) -> str:
    """:func:`ratio_floor` of ``r`` as a fold, the arrow's label: 84.5 is '84×', 143.6 '140×', 299.1 '290×'."""
    return style.fmt_num(ratio_floor(r)) + style.TIMES


def series(df) -> Dict[str, Dict]:
    """What each model's panel draws, from E1_traces (models in no particular order).

    Per model: ``it`` (iterations 1..N, an index), ``N``, ``gains`` (one
    ``(gain_c, gain_rs)`` pair of Series per genome), ``gmin`` and ``gmax``
    (the smallest and largest gain of any genome and either program, per
    iteration), ``env`` (the largest |LL_rs - LL_c| over the genomes per
    iteration, NaN where it is 0) and ``arrow`` (``k``, the iteration where
    gmin / env is lowest, ``ratio`` and ``label``; None when no difference is
    positive). A genome lacking either program is left out; a model without
    any genome with both is absent.
    """
    import numpy as np
    import pandas as pd

    df = df.sort_values(["model", "replicate", "tool", "iteration"])
    out: Dict[str, Dict] = {}
    for m, sub in df.groupby("model", sort=False):
        gc_cols, gr_cols, gap_cols = {}, {}, {}
        for rep, rsub in sub.groupby("replicate"):
            by = {t: s.set_index("iteration")["loglike"].astype(float) for t, s in rsub.groupby("tool")}
            if "c" not in by or "rs" not in by:
                continue
            # Only the iterations both programs give a log-likelihood after (C psmc prints none after its last).
            common = by["c"].dropna().index.intersection(by["rs"].dropna().index).sort_values()
            if len(common) < 2:
                continue
            for tool, cols in (("c", gc_cols), ("rs", gr_cols)):
                g = by[tool].loc[common].diff().iloc[1:]  # the gain of each iteration after the first given
                bad = g[~(g > 0)]
                if len(bad):  # a log axis would silently drop it
                    raise ValueError(f"log-likelihood fell: {m} replicate {rep} {tool} iteration {int(bad.index[0])}")
                cols[rep] = g
            gap_cols[rep] = (by["rs"].loc[common] - by["c"].loc[common]).abs().iloc[1:]
        if not gc_cols:
            continue
        gc, gr, gaps = (pd.DataFrame(c).sort_index() for c in (gc_cols, gr_cols, gap_cols))
        it = gc.index
        gmin = np.minimum(gc.min(axis=1), gr.min(axis=1))
        gmax = np.maximum(gc.max(axis=1), gr.max(axis=1))
        env = gaps.max(axis=1).reindex(it)
        env = env.where(env > 0)
        arrow = None
        ratio = gmin / env
        if ratio.notna().any():
            k = ratio.idxmin()
            arrow = {"k": int(k), "ratio": float(ratio.loc[k]), "label": ratio_label(float(ratio.loc[k]))}
        out[m] = {"it": it, "N": int(it.max()), "gains": [(gc_cols[r], gr_cols[r]) for r in gc_cols], "gmin": gmin,
                  "gmax": gmax, "env": env, "arrow": arrow}
    return out


def _pow10(v: float, _pos=None) -> str:
    return r"$10^{%s}$" % f"{round(math.log10(v))}".replace("-", style.MINUS)


def _even_decades(lo: float, hi: float):
    """The even decades enclosing ``lo``-``hi`` (1e-4 to 1e4 for 0.003-4,767)."""
    dlo = 2 * math.floor(math.log10(lo) / 2)
    dhi = 2 * math.ceil(math.log10(hi) / 2)
    return dlo, max(dhi, dlo + 2)


def _name_in_place(ax, text: str, values, *, above: bool, x_bound: float, ylim) -> None:
    """Name a line of panel a beside it, from iteration :data:`LABEL_X` on: ``above`` or below ``values``.

    The name's height follows the line under the whole name (the highest
    of ``values`` under it, or the lowest), so it clears the line from its
    first letter to its last. A name that would run past ``x_bound`` (the
    arrow's label column, or the axes' right edge) when only a few
    iterations are drawn ends exactly there instead (``ha='right'``). Its
    width is measured at the figure's current dpi, which :func:`draw` sets
    to the PNG's: at matplotlib's default 100 dpi plain text measures about
    10% short.
    """
    import numpy as np

    vals = values.dropna()
    if vals.empty:
        return
    t = style.direct_label(ax, LABEL_X, float(vals.iloc[0]), text, va="bottom" if above else "top")
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    inv = ax.transData.inverted()
    width = inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]
    x0 = LABEL_X
    if x0 + width > x_bound:
        x0 = x_bound - width
        t.set_horizontalalignment("right")
        t.set_x(x_bound)
    under = vals[(vals.index >= math.floor(x0)) & (vals.index <= math.ceil(x0 + width))]
    if under.empty:
        under = vals
    room = 10.0 ** LABEL_DECADES
    lo, hi = ylim
    if above:
        t.set_y(min(GAIN_LABEL_RISE * float(np.max(under.values)), hi / room))
    else:
        y = float(np.min(under.values)) / GAP_LABEL_DROP
        if y / room >= lo:
            t.set_y(y)
        else:  # no room under the line: above it
            t.set_y(float(np.max(under.values)) * GAP_LABEL_DROP)
            t.set_verticalalignment("bottom")


def _panel(ax, p: Dict, title: str, x_max: float, per_mm: float) -> float:
    """One model's gains, grey line and arrow; return where names in the panel must end (left of the arrow)."""
    # The underlay rule: every C psmc line pale and wide beneath, every psmc-rs line thin on top.
    for gc, _ in p["gains"]:
        ax.plot(gc.index, gc.values, **style.underlay_kw(style.C_PSMC, **UNDER))
    for _, gr in p["gains"]:
        ax.plot(gr.index, gr.values, **style.overlay_kw(style.RS02, linewidth=style.LW_THIN, **LINE))
    # The difference between the programs in INK2, its colour in every figure of the set.
    ax.plot(p["it"], p["env"].values, color=style.INK2, linewidth=style.LW_DATA, solid_capstyle="butt",
            zorder=style.Z_DATA + 1, **LINE)
    bound = x_max
    a = p["arrow"]
    if a is not None:
        k = a["k"]
        lo, hi = float(p["env"].loc[k]), float(p["gmin"].loc[k])
        x = k + ARROW_DX_MM * per_mm
        ax.annotate("", xy=(x, hi), xytext=(x, lo),
                    arrowprops=dict(style.ARROW_PROPS, shrinkA=ARROW_SHRINK_PT, shrinkB=ARROW_SHRINK_PT))
        left = k / per_mm > 8.0  # the label left of the arrow where it fits inside the axes
        dx = LABEL_DX_MM * per_mm
        style.direct_label(ax, x - dx if left else x + dx, math.sqrt(lo * hi), a["label"],
                           ha="right" if left else "left", zorder=style.Z_TEXT)
        bound = x - dx  # names never cross the arrow
    style.panel_title(ax, title)
    return bound


def draw(src: Source):
    import numpy as np
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

    per = series(src.table("E1_traces"))
    models = model_order(per.keys())
    n = max(len(models), 1)
    fig = style.figure(WIDTH_MM, HEIGHT_MM)
    # Lay out at the PNG's dpi: the placements below (the names in panel a, the x label, the letters) are
    # measured from text extents, and at matplotlib's default 100 dpi plain text measures ~10% short.
    # Everything is placed in figure or data coordinates, so the figure's own dpi is restored at the end.
    dpi0 = fig.get_dpi()
    fig.set_dpi(style.PNG_DPI)
    top, bottom = style.TOP_LEGEND_MM, style.BOTTOM_MM
    pw = (WIDTH_MM - style.LEFT_MM - style.RIGHT_MM - (n - 1) * GUTTER_MM) / n
    axes = [style.axes_mm(fig, style.LEFT_MM + j * (pw + GUTTER_MM), top, pw, HEIGHT_MM - top - bottom)
            for j in range(n)]

    # One y axis for all panels: even decades around every gain and every positive difference.
    lows = [float(p["gmin"].min()) for p in per.values()] + \
           [float(np.nanmin(p["env"].values)) for p in per.values() if p["env"].notna().any()]
    highs = [float(p["gmax"].max()) for p in per.values()]
    dlo, dhi = _even_decades(min(lows), max(highs)) if lows else (-2, 2)
    # One x axis: iterations 0 to the last, then X_PAD_MM for the arrows (24 iterations: 0-25.6).
    n_it = max((p["N"] for p in per.values()), default=1)
    x_max = n_it / (1.0 - X_PAD_MM / pw)
    per_mm = x_max / pw  # iterations per mm
    step = 6 if n_it >= 12 else max(1, math.ceil(n_it / 4))
    bounds = [_panel(ax, per[m], MODEL_LABEL.get(m, m), x_max, per_mm) for ax, m in zip(axes, models)]

    # The y hairlines: at the labelled decades strictly inside the axes, from the y axis to where the lines'
    # ink ends (the underlay's round cap, half its width right of the last iteration), so they never reach the
    # arrow column. Drawn as one collection, where value_grid's gridlines would span the whole axes.
    cap = style.UNDERLAY_LW / 2.0 / style.PT_PER_MM * per_mm
    hairlines = [10.0 ** e for e in range(dlo + 2, dhi, 2)]
    for j, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_ylim(10.0 ** dlo, 10.0 ** dhi)
        ax.yaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(dlo, dhi + 1, 2)]))
        ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
        ax.yaxis.set_minor_locator(FixedLocator([10.0 ** e for e in range(dlo + 1, dhi, 2)]))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.0, x_max)
        ax.xaxis.set_major_locator(FixedLocator(list(range(0, n_it + 1, step))))
        ax.xaxis.set_minor_locator(NullLocator())
        if hairlines:
            end = (per[models[j]]["N"] if j < len(models) else n_it) + cap
            ax.hlines(hairlines, 0.0, end, colors=style.GRID, linewidths=style.LW_HAIR, capstyle="butt",
                      zorder=0.5)
        if j:
            ax.tick_params(axis="y", which="both", labelleft=False)
    axes[0].set_ylabel("Log-likelihood difference")

    # Panel a names its two bundles in place, both from iteration 1: the gains just above the bundle, the
    # grey line just below it (above it where that would leave the axes).
    if models:
        p, ax = per[models[0]], axes[0]
        ylim = (10.0 ** dlo, 10.0 ** dhi)
        _name_in_place(ax, r"Gain in iteration $\mathit{i}$", p["gmax"], above=True, x_bound=bounds[0], ylim=ylim)
        _name_in_place(ax, "Largest gap between programs", p["env"], above=False, x_bound=bounds[0], ylim=ylim)

    style.row_xlabel(fig, axes, r"EM iteration, $\mathit{i}$")
    # The key's psmc-rs line is LW_DATA, not the 0.5 pt drawn: ten coinciding 0.5 pt lines read as one line
    # at least that wide (and it matches Fig 1's psmc-rs key).
    style.top_legend(fig, [style.handle(style.C_PSMC, kind="underlay"), style.handle(style.RS02)])
    style.panel_letters(fig, axes)
    fig.set_dpi(dpi0)
    return fig
