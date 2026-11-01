"""Supplement Fig S1 (E1): C psmc and psmc-rs climb the same likelihood at every EM iteration.

One column per simulated model of E1, in config.MODEL_ORDER (Constant,
Bottleneck, Expansion, Zigzag), every genome (replicate) of the model in
each, and two rows of panels sharing x, the EM iteration i:

* above, the log-likelihood gained in iteration i, LL(i) - LL(i - 1), from
  i = 1, one line per genome and program, drawn by the underlay rule: C
  psmc a wide pale-orange underlay, psmc-rs a thin blue line on top (the
  psmc-rs lines read as one blue ribbon with an orange rim). The axis is
  logarithmic and can show only positive gains, so a gain that is not
  positive raises :class:`ValueError` instead of vanishing from it: the
  lines drawn show that the log-likelihood never fell;
* below, psmc-rs's log-likelihood after iteration i minus C psmc's,
  LL_rs(i) - LL_c(i), from i = 0 (the same starting model in both), one
  thin line per genome on a linear axis about a grey zero line (REF), in
  INK2, as Fig 1e draws the programs' differences. Fig 1's strips draw
  theirs in psmc-rs's blue, one line per strip; here ten blue lines would
  read as more psmc-rs gains, and in greyscale blue stands only ΔL* 7.8
  from the zero line (INK2 20.4). Panel a names the sign in place:
  "psmc-rs higher" in its upper left corner, "C psmc higher" in its lower
  left one. The row's label, "Difference", leaves the quantity to the
  caption, which must name it: "Log-likelihood difference" (26.4 mm at
  7 pt) is longer than the 17 mm row, and centred on it would end 0.9 mm
  under the gains' label, reading as one line with it.

Each row shares one y axis (tick labels and its label on panel a only, the
two labels aligned) and both rows share x, with one label under the lower
row; the key (C psmc, psmc-rs) is one line at the top, and the letters and
titles sit above the upper row. The gains' axis runs between round limits
around every gain (:func:`.data.decades`: 0.2 to 10,000 for E1), labelled
at its decades below the top (1 to 1,000: a '10,000' would push both y
labels across the 1 mm left margin), without minor ticks or grid. The
differences' axis is symmetric about 0: its limit is the largest difference
and a fifth more, rounded up to one significant figure (0.08 for E1's
0.065; the 1-2-5 value 0.1 would flatten the lines by a fifth), labelled at
0 and at plus and minus the largest 1, 2 or 5 times a power of ten below it
(0.05). x runs from 0 to one past the last iteration.

Iteration i counts completed EM iterations: C psmc's LK of round i + 1 and
psmc-rs's ``loglike[i]`` are both the log-likelihood after i iterations. C
psmc prints none after its last iteration, so both rows are drawn up to the
last iteration both give. Left to the caption: how closely the two agreed
at i = 0, the largest difference, theta and rho, and how many times each
iteration's smallest gain exceeded the largest difference after that
iteration. That ratio is taken in each panel: in iteration i, the smallest
gain of any genome and either program over the largest |LL_rs(i) - LL_c(i)|
of any genome. In E1 it is at least 84 (Constant, i = 24). Taken across
panels (Expansion's smallest gain in iteration 24 over Constant's largest
difference after it), or over all of a panel's iterations at once
(Expansion's smallest gain, in iteration 24, over its largest difference,
after iteration 1), it would be only 23.

:func:`series` computes what the upper row draws and :func:`differences`
what the lower one draws, over the same genomes and iterations; the
manuscript's numbers (make_numbers.py) count the caption's gains with
:func:`series`, so they cannot drift from the lines drawn.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from . import style
from .data import Source, decades
from .fig2 import MODEL_LABEL, model_order

WIDTH_MM, HEIGHT_MM = style.WIDTH_MM["double"], 62.0
# Between columns: the y axes are shared (tick labels on the first panel
# only), but each panel's letter sits in the gutter left of its title, so the
# gutter is wider than style.GUTTER_SHARED_MM (3 mm would leave a letter
# touching the panel to its left) and narrower than style.GUTTER_MM.
GUTTER_MM = 5.0
# The two rows: the differences' row is DIFF_MM high, STACK_GAP_MM under the gains' row (no x tick labels between
# them, the rows share x), and the gains' row takes the rest.
DIFF_MM = 17.0
STACK_GAP_MM = 2.5
GAIN_MM = HEIGHT_MM - style.TOP_LEGEND_MM - STACK_GAP_MM - DIFF_MM - style.BOTTOM_MM  # 26.65
# The differences' limit: the largest difference and this share more (0.065 and a fifth: 0.078, rounded up to 0.08).
DIFF_ROOM = 1.2
DIFF_RESOLUTION = 1e-6  # C psmc prints its log-likelihoods to six decimals: a smaller difference is none
# Panel a's notes on the sign: in from the corners of its lower axes by this fraction of them (1.2 mm from the
# spine, 0.5 mm from the axes' top and bottom).
NOTE_INSET = 0.03
LINE = {"solid_joinstyle": "round"}  # round joins: no miter spikes where a line bends sharply (the 2.4 pt underlay)
# Round caps on the underlays only: ten 2.4 pt butt ends cut across lines of different slopes leave a notched
# edge and pale tabs at i = 1; round ones close the rim around the blue ribbon's ends (0.42 mm past the first and
# last iteration).
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
    """:func:`ratio_floor` of ``r`` as a fold: 84.5 is '84×', 143.6 '140×', 299.1 '290×'."""
    return style.fmt_num(ratio_floor(r)) + style.TIMES


def _genomes(df) -> Dict[str, Dict]:
    """Each model's genomes fitted by both programs: ``{model: {replicate: (ll_c, ll_rs)}}``, models sorted.

    Both log-likelihoods are taken after the iterations both programs give
    (C psmc prints none after its last), at least two of them. A genome
    lacking either program is left out; a model without any genome with both
    is absent.
    """
    df = df.sort_values(["model", "replicate", "tool", "iteration"])
    out: Dict[str, Dict] = {}
    for m, sub in df.groupby("model", sort=False):
        for rep, rsub in sub.groupby("replicate"):
            by = {t: s.set_index("iteration")["loglike"].astype(float) for t, s in rsub.groupby("tool")}
            if "c" not in by or "rs" not in by:
                continue
            common = by["c"].dropna().index.intersection(by["rs"].dropna().index).sort_values()
            if len(common) >= 2:
                out.setdefault(m, {})[rep] = (by["c"].loc[common], by["rs"].loc[common])
    return out


def series(df) -> Dict[str, Dict]:
    """What each model's upper panel draws, the gains, from E1_traces (models in no particular order).

    Per model: ``it`` (iterations 1..N, an index), ``N``, ``gains`` (one
    ``(gain_c, gain_rs)`` pair of Series per genome), ``gmin`` and ``gmax``
    (the smallest and largest gain of any genome and either program, per
    iteration), ``env`` (the largest |LL_rs - LL_c| over the genomes per
    iteration, NaN where it is 0) and ``arrow``, where the programs come
    closest for what an iteration gained (``k``, the iteration where
    gmin / env is lowest, that ``ratio`` and its ``label``, :func:`ratio_label`;
    None when no difference is positive). The genomes and iterations are
    those of :func:`_genomes`.
    """
    import numpy as np
    import pandas as pd

    out: Dict[str, Dict] = {}
    for m, genomes in _genomes(df).items():
        gc_cols, gr_cols, gap_cols = {}, {}, {}
        for rep, (ll_c, ll_rs) in genomes.items():
            for tool, ll, cols in (("c", ll_c, gc_cols), ("rs", ll_rs, gr_cols)):
                g = ll.diff().iloc[1:]  # the gain of each iteration after the first given
                bad = g[~(g > 0)]
                if len(bad):  # a log axis would silently drop it
                    raise ValueError(f"log-likelihood fell: {m} replicate {rep} {tool} iteration {int(bad.index[0])}")
                cols[rep] = g
            gap_cols[rep] = (ll_rs - ll_c).abs().iloc[1:]
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


def differences(df) -> Dict[str, List]:
    """What each model's lower panel draws, from E1_traces: per genome, LL_rs(i) - LL_c(i), a Series over i.

    From i = 0, the same starting model in both programs, to the last
    iteration both give, over the genomes :func:`series` draws.
    """
    return {m: [ll_rs - ll_c for ll_c, ll_rs in genomes.values()] for m, genomes in _genomes(df).items()}


def _gain_ticks(lo: float, hi: float) -> List[float]:
    """The gains' labelled ticks from ``lo`` up to, not at, ``hi``: decades, or 1, 2 and 5 times them on a span
    :func:`.style.narrow_log_span` calls narrow (0.2-10,000: 1 to 1,000, as a '10,000' would push both y labels
    across the 1 mm left margin, to 0.22 mm)."""
    subs = (1.0, 2.0, 5.0) if style.narrow_log_span(lo, hi) else (1.0,)
    return [s * 10.0 ** e for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))) for s in subs
            if lo * (1.0 - 1e-9) <= s * 10.0 ** e < hi * (1.0 - 1e-9)]


def _diff_axis(largest: float) -> Tuple[float, float]:
    """The differences' limit (plus and minus) and their outer tick, for differences up to ``largest`` in size.

    The limit is ``largest`` and :data:`DIFF_ROOM` more, rounded up to one
    significant figure (0.065: 0.08); the tick, the largest 1, 2 or 5 times
    a power of ten below the limit (0.05).
    """
    v = DIFF_ROOM * max(largest, DIFF_RESOLUTION)
    mag = 10.0 ** math.floor(math.log10(v))
    lim = round(math.ceil(v / mag * (1.0 - 1e-12)) * mag, 12)
    e = math.floor(math.log10(lim))
    tick = max(s * 10.0 ** k for k in (e - 1, e) for s in (1, 2, 5) if s * 10.0 ** k < lim * (1.0 - 1e-9))
    return lim, round(tick, 12)


def draw(src: Source):
    from matplotlib.ticker import FixedLocator, NullLocator

    traces = src.table("E1_traces")
    per, diffs = series(traces), differences(traces)
    models = model_order(per.keys())
    n = max(len(models), 1)
    # Laid out at the PNG's dpi: the x label and the letters are placed by measured text (style, Layout).
    fig = style.figure(WIDTH_MM, HEIGHT_MM, dpi=style.PNG_DPI)
    # A grid, not axes placed by hand: fig.align_ylabels aligns the labels of grid cells only.
    gs = style.grid(fig, 2, n, top_mm=style.TOP_LEGEND_MM, wspace_mm=GUTTER_MM, hspace_mm=STACK_GAP_MM,
                    height_ratios=(GAIN_MM, DIFF_MM))
    above = [fig.add_subplot(gs[0, j]) for j in range(n)]
    below = [fig.add_subplot(gs[1, j]) for j in range(n)]

    for ax, bx, m in zip(above, below, models):
        p = per[m]
        # Above, the underlay rule: every C psmc line pale and wide beneath, every psmc-rs line thin on top.
        for gc, _ in p["gains"]:
            ax.plot(gc.index, gc.values, **style.underlay_kw(style.C_PSMC, **UNDER))
        for _, gr in p["gains"]:
            ax.plot(gr.index, gr.values, **style.overlay_kw(style.RS02, linewidth=style.LW_THIN, **LINE))
        style.panel_title(ax, MODEL_LABEL.get(m, m))
        # Below, the programs' differences in INK2, as Fig 1e's (not blue: the module's docstring says why).
        for d in diffs[m]:
            bx.plot(d.index, d.values, color=style.INK2, linewidth=style.LW_THIN, zorder=style.Z_DATA, **LINE)

    # One y axis per row and one x axis for both.
    gains = [v for p in per.values() for v in (float(p["gmin"].min()), float(p["gmax"].max()))]
    lo, hi = decades(min(gains), max(gains)) if gains else (1.0, 10.0)
    lim, tick = _diff_axis(max((float(d.abs().max()) for ds in diffs.values() for d in ds), default=0.0))
    n_it = max((p["N"] for p in per.values()), default=1)
    step = 6 if n_it >= 12 else max(1, math.ceil(n_it / 4))
    for j, (ax, bx) in enumerate(zip(above, below)):
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator(_gain_ticks(lo, hi)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelbottom=False)
        style.ref_line(bx, 0.0)
        bx.set_ylim(-lim, lim)
        bx.yaxis.set_major_locator(FixedLocator([-tick, 0.0, tick]))
        for a in (ax, bx):
            a.yaxis.set_major_formatter(style.plain_formatter())
            a.set_xlim(0.0, n_it + 1.0)
            a.xaxis.set_major_locator(FixedLocator(list(range(0, n_it + 1, step))))
            a.xaxis.set_minor_locator(NullLocator())
            if j:
                a.tick_params(axis="y", which="both", labelleft=False)
    above[0].set_ylabel("Log-likelihood gain")
    below[0].set_ylabel("Difference")
    fig.align_ylabels([above[0], below[0]])
    # The sign of the difference, named in place in panel a's lower axes.
    style.note(below[0], NOTE_INSET, 1.0 - NOTE_INSET, "psmc-rs higher", va="top")
    style.note(below[0], NOTE_INSET, NOTE_INSET, "C psmc higher", va="bottom")

    style.row_xlabel(fig, below, r"EM iteration, $\mathit{i}$")
    # The key's psmc-rs line is LW_DATA, not the 0.5 pt drawn: ten coinciding 0.5 pt lines read as one line
    # at least that wide (and it matches Fig 1's psmc-rs key).
    style.top_legend(fig, [style.handle(style.C_PSMC, kind="underlay"), style.handle(style.RS02)])
    style.panel_letters(fig, above)
    return fig
