"""Supplement Fig S2 (E4): taking partly callable bins as full biases PSMC; psmc-rs's weighting removes most of it.

One row of two panels, 183 x 60 mm, in the paper's two colours. There is
no legend: every series is named in place, in its own text colour.

Blue is psmc-rs's default: psmc-rs on the VCF with its callable mask,
each called bin weighted by the callable bases it holds (``--bin-callable
auto``, route ``mask_auto``). Orange is C psmc's input and model: the bins
fq2psmcfa makes from the genome's consensus FASTA, kept unless more than
90% of their bases are uncallable, every called bin taken as full (route
``fq2psmcfa``, fitted by psmc-rs with C psmc's model). These bins equal
the VCF's bins read with ``--bin-callable ignore`` (``mask_ignore``), bin
for bin and in every fitted value, so only one of the two is drawn:
``fq2psmcfa``, or ``mask_ignore`` when a table has no ``fq2psmcfa`` rows
(:func:`full_route` checks that they agree where both exist). multihetsep
and the VCF read without its mask are not drawn; the caption gives their
theta (E4_routes' medians).

When missing data are dispersed in short runs, almost every bin stays
called, and many hold only part of their bases (about half of the called
bins with 10% missing, nearly all with 30%). Taken as full, those bins
make theta per bin low, and every size and time with it, since both
scale with theta. Weighted by their callable bases, they leave theta
only slightly low. Clustered runs empty whole bins, which both models
drop.

* a: one genome with the example mask (``config.FIGURES.e4_example_mask``,
  dispersed 30%), over the fit of the same genome without missing data
  (wide grey underlay, ``nomissing``/``mask_auto``). Blue (1 pt) is
  weighted bins and orange (0.75 pt) is bins taken as full. The genome is
  the replicate whose weighted fit has the median RMSE of log10 Ne against
  its complete-data fit; the title counts the genomes. Every curve is a
  PSMC step curve clipped to the model's informative window, on log axes
  of N_e and years ago, both in thousands; a last interval that the
  window's end cuts to a sliver (under :data:`MIN_LAST_MM`) is not drawn,
  and all three curves end together. The arrow (one head: a scaling,
  :data:`style.SCALE_ARROW_PROPS`) starts on the complete-data curve at
  :data:`SHIFT_AT_YEARS` and points along the multiplication of both its
  time and its N_e by r = theta with full bins / theta without missing
  data, for this genome; r is printed beside it, number first (0.65×),
  and it stops where it meets the orange curve (the scaling is close, not
  exact).
* b: theta relative to the fit of the same genome without missing data
  (theta / that fit's theta, the unit of a's r), against the share of
  positions uncallable. Two facets share one y axis, from 0.6 to 1.04
  (:data:`YLIM_B`, widened to a round tenth if a replicate needs it),
  ticked every 0.1 and labelled with the step's decimals (0.6 to 1.0,
  :func:`style.plain_formatter`): the dispersed runs on the left, beside
  a's example, and the clustered runs on the right. Each is titled by
  what its masks do to the bins, as the weighted fits measure it
  (:func:`facet_title`: "Dispersed: bins partly callable", "Clustered:
  whole bins missing"); the run lengths are the caption's. A grey line
  marks 1, named "No missing data" in the clustered facet. Large
  white-edged circles are the medians over the replicates, joined by
  lines, each route's :data:`DODGE` beside its x (blue left, orange
  right; 0.73 mm, under a median's width) so that both show in the
  clustered facet, where the two routes agree. The small pale dots are
  the replicates, all of them, :data:`JITTER` apart around their median
  in replicate order, beneath the lines and medians. In the clustered
  facet the orange line would run under the blue one, and a segment that
  lies within a blue line's width of it all along is not drawn: only its
  fringe showed. The control (0%) is the complete-data fit itself, 1 for
  every replicate, so both lines start from one point there, undodged.
  Every mark stays :data:`X_PAD_MM` inside the x limits. In the dispersed
  facet each route is named beside its line, and its median at the
  largest missing fraction is printed number first: "0.65×" left of the
  orange median, "Weighted bins" and "0.98×" on one baseline under the
  blue line's end, and "Bins taken as full" above the orange line from
  about its middle. The labels at b's right ("0.98×", "Bins taken as
  full", "No missing data") end on one edge, 1 percentage point right of
  the last x tick: a name that would end within :data:`ALIGN_MM` of that
  edge, or past it, ends on it.

Each label sits :data:`CLEAR_MM` above or below the marks it names (the
arrow included), over its own width as it prints and that clearance past
each end, and stays inside its axes (:func:`_place`); a median's number
sits that clearance left of the median, its replicates and its line
(:func:`_beside`).

Everything comes from ``E4_routes`` (theta; the RMSE that picks the
example; the shares of bins missing and partly callable that title b's
facets) and ``E4_curves``; mask structures and missing fractions from the
profile's ``E4.masks``. A mask, route or replicate missing from the tables
is left out.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config
from . import style
from .data import Source, curves, decades, informative_window

WEIGHTED = "mask_auto"  # psmc-rs's default: each called bin weighted by its callable bases
FULL = "fq2psmcfa"  # every called bin taken as full: C psmc's input and model
FULL_FALLBACK = "mask_ignore"  # the same bins and fit, from the VCF (drawn when FULL has no rows)
DRAWN = (WEIGHTED, FULL)
COMPLETE = config.E4_CONTROL_MASK  # the no-missing control; its WEIGHTED fit is the complete-data fit
STRUCTURES = ("dispersed", "clustered")  # b's facets, left to right: the dispersed runs beside a's example
# b's facet titles: what a structure's masks do to the bins (facet_title).
PARTLY_CALLABLE, WHOLE_MISSING = "bins partly callable", "whole bins missing"

# Layout in mm from the figure's top-left corner, with the house margins.
W_MM, H_MM = style.WIDTH_MM["double"], 60.0
AXES_H_MM = H_MM - style.TOP_MM - style.BOTTOM_MM
AX_A = (style.LEFT_MM, style.TOP_MM, 76.0, AXES_H_MM)  # aspect 1.58 (style.NE_ASPECT)
# The 16 mm between a and b holds b's y tick labels, its y label and the letter b.
B_X_MM = AX_A[0] + AX_A[2] + 16.0
FACET_W_MM = (W_MM - style.RIGHT_MM - B_X_MM - style.GUTTER_SHARED_MM) / 2.0
AX_B = tuple((B_X_MM + k * (FACET_W_MM + style.GUTTER_SHARED_MM), style.TOP_MM, FACET_W_MM, AXES_H_MM)
             for k in range(len(STRUCTURES)))

YLIM_A = (500.0, 1e5)  # a's N_e axis spans at least this (more if a curve needs it)
SHIFT_AT_YEARS = 4.0e5  # the arrow's tail, on the curves' flat stretch from 200 to 800 ky
# a's label anchors, times a's left x limit ("No missing data" starts CLEAR_MM inside the left spine).
LABEL_X_A = {"complete": 1.0, "weighted": 2.6, "full": 10.0}
MIN_LAST_MM = 0.5  # a: a last interval the window cuts narrower than this is not drawn (it reads as a hook)
CLEAR_MM = 1.0  # a label's clearance from the edge of the marks it names
ARROW_CLEAR_MM = 1.5  # and from a's arrow, so that the arrow points at the curve, not at the label below it
EDGE_MM = 1.0  # a label's least distance from its axes' right and top edges
NAME_GAP_MM = 1.0  # between a series' name and its number set on one baseline (b's "Weighted bins 0.98×")
# b's labels at the right end on one edge. A name placed from its line's middle that would end within this of
# that edge, or past it, ends on it: a near miss of under 2 mm reads as an accident, not as a choice.
ALIGN_MM = 2.0

# b's theta ratio spans at least this: 1.04 leaves room above the line at 1 for its name.
YLIM_B = (0.6, 1.04)
# b's x, in percentage points (1 pp is about 1.05 mm across a facet). Each
# route's median sits DODGE beside its x, blue left and orange right, so
# that both show in the clustered facet, where the routes agree; 0.7 pp is
# 0.73 mm, less than a median's diameter, so every median reads at its tick.
DODGE = 0.7
# The replicates sit in replicate order around their median, JITTER apart
# (x + dx + (i - (n - 1) / 2) JITTER), under it and the lines: those nearest
# the median tuck partly under its white-edged circle.
JITTER = 0.35
X_PAD_MM = 0.5  # b's least room between a mark's outer edge and the facet's x limits
# The white edge of b's medians, a written exception to the house's edgeless markers (style, Markers): it keeps
# each median apart from its line, its replicates and the other route's median, which the dodge sets 0.73 mm away.
MEDIAN_EDGE_PT = 0.5
# b's replicate dots are style.REPLICATE_MS (1.6 pt), beside a 3 pt median.
# At style.REPLICATE_ALPHA (0.35) a dot that small fades into the white, so
# they take their colour at 55%, still clearly paler than the median (also
# a written exception in style, Markers).
REPLICATE_SHARE = 0.55
# b's routes: (line width, z of the line, of the replicates, of the medians). The pale replicates lie beneath
# (orange under blue), then the lines, then the medians, orange first and blue over it. The orange medians lie
# over the blue line, which runs past them where the routes agree; their white edge keeps them apart from it.
B_LOOK = {FULL: (style.LW_DATA, style.Z_DATA, style.Z_UNDERLAY, style.Z_DATA + 0.6),
          WEIGHTED: (style.LW_EMPH, style.Z_DATA + 0.5, style.Z_UNDERLAY + 0.1, style.Z_DATA + 0.7)}

# Series names are set in the series' text colour (the house exception for
# a panel without a legend), the same darker step for both routes as in Fig 2.
TEXT_COLOR = {WEIGHTED: style.BLUE_TEXT, FULL: style.ORANGE_TEXT}
B_NAMES = {WEIGHTED: "Weighted bins", FULL: "Bins taken as full"}  # b's names: a's labels without the parentheses
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


# ---------------------------------------------------------------------------
# Choices from the tables
# ---------------------------------------------------------------------------


def full_route(routes) -> str:
    """The route drawn as bins taken as full: ``fq2psmcfa``, else ``mask_ignore``.

    The two fit the same bins, so where both have rows their theta must
    agree for every mask and replicate; if they do not, raises ValueError.
    """
    import numpy as np

    have = set(routes["route"])
    if FULL not in have:
        return FULL_FALLBACK
    if FULL_FALLBACK in have:
        cols = ["mask", "replicate", "theta_rel_complete"]
        both = routes[routes.route == FULL][cols].merge(routes[routes.route == FULL_FALLBACK][cols],
                                                        on=["mask", "replicate"], suffixes=("_fq", "_ignore"))
        if not np.allclose(both["theta_rel_complete_fq"].to_numpy(float),
                           both["theta_rel_complete_ignore"].to_numpy(float), rtol=0.0, atol=1e-12, equal_nan=True):
            raise ValueError(f"{FULL} and {FULL_FALLBACK} fit the same bins, but their theta differ")
    return FULL


def example_mask(cfg: config.E4, routes) -> Optional[str]:
    """a's mask: ``config.FIGURES.e4_example_mask``, else the most-missing dispersed mask, else the last mask."""
    present = set(routes["mask"])
    masks = [m for m in cfg.masks if m.name in present]
    if not masks:
        return None
    if config.FIGURES.e4_example_mask in {m.name for m in masks}:
        return config.FIGURES.e4_example_mask
    dispersed = [m for m in masks if m.structure == "dispersed"]
    if dispersed:
        return max(dispersed, key=lambda m: m.missing_fraction).name
    return masks[-1].name


def example_replicate(routes, mask: str) -> Optional[int]:
    """The replicate whose weighted fit with ``mask`` has the median RMSE against its complete-data fit.

    The fits are sorted by (RMSE, replicate); an even count takes the lower median.
    """
    sub = routes[(routes["mask"] == mask) & (routes.route == WEIGHTED)]
    sub = sub.sort_values(["rmse_log10_ne_vs_complete", "replicate"])
    if sub.empty:
        return None
    return int(sub["replicate"].iloc[(len(sub) - 1) // 2])


def facet_masks(cfg: config.E4, structure: str) -> List[Tuple[str, float]]:
    """``(mask, percent uncallable)`` in one facet of b: the control, then the structure's masks by fraction."""
    own = sorted(((m.name, 100.0 * m.missing_fraction) for m in cfg.masks if m.structure == structure),
                 key=lambda t: (t[1], t[0]))
    return [(COMPLETE, 0.0)] + own if own else []


def facet_title(routes, cfg: config.E4, structure: str) -> str:
    """The title of b's facet of ``structure``: the structure, and what its masks do to the bins.

    The weighted fits (which count each bin's callable bases) of the
    structure's mask with the most missing tell it: "bins partly callable"
    where the median share of called bins holding only part of their bases
    is above the median share of bins missing, "whole bins missing"
    otherwise. Without those shares the title is the structure alone.
    """
    import numpy as np

    present = set(routes["mask"])
    own = [m for m in cfg.masks if m.structure == structure and m.name in present]
    if not own:
        return structure.capitalize()
    mask = max(own, key=lambda m: (m.missing_fraction, m.name)).name
    sub = routes[(routes["mask"] == mask) & (routes.route == WEIGHTED)]
    partly, missing = (sub[c].to_numpy(dtype=float) for c in ("partly_callable_fraction", "missing_bin_fraction"))
    partly, missing = partly[np.isfinite(partly)], missing[np.isfinite(missing)]
    if not (partly.size and missing.size):
        return structure.capitalize()
    phrase = PARTLY_CALLABLE if np.median(partly) > np.median(missing) else WHOLE_MISSING
    return f"{structure.capitalize()}: {phrase}"


def _theta_ratio(routes, mask: str, route: str):
    """theta over the complete-data fit's (1 + theta_rel_complete) for ``mask`` and ``route``, by replicate.

    Finite values only.
    """
    import numpy as np

    sub = routes[(routes["mask"] == mask) & (routes.route == route)].sort_values("replicate")
    v = 1.0 + sub["theta_rel_complete"].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _count(n: int) -> str:
    """``n`` in words up to nine, in figures above."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else f"{n:,}"


def _fold(r: float) -> str:
    """A factor as printed on the data: the number first, to two decimals, then the times sign (0.65×)."""
    return f"{r:.2f}{style.TIMES}"


# ---------------------------------------------------------------------------
# Labels placed from the data
# ---------------------------------------------------------------------------


def _px(fig, mm: float) -> float:
    """``mm`` in display pixels."""
    return mm * style.MM * fig.dpi


def _extent(fig, text: str, size: float):
    """The extent of ``text`` at ``size`` as it prints, in display pixels."""
    probe = fig.text(0.0, 0.0, text, fontsize=size)
    ext = probe.get_window_extent(fig.canvas.get_renderer())
    probe.remove()
    return ext


def _envelope(ax, marks, x0: float, x1: float, upper: bool) -> Optional[float]:
    """The top (``upper``) or bottom edge, in display y, of ``marks`` between data x ``x0`` and ``x1``.

    ``marks`` are ``(kind, xs, ys, width_pt)``. The kind is ``"steps"`` (a
    steps-post curve: ``ys[k]`` from ``xs[k]`` to ``xs[k + 1]``),
    ``"line"`` (a polyline) or ``"points"``. ``width_pt`` is the line
    width, or the marker diameter for points.
    """
    import numpy as np

    edge = None
    for kind, xs, ys, width in marks:
        xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        if not xs.size:
            continue
        if kind == "steps":
            ends = np.append(xs[1:], xs[-1])
            ys = ys[(ends >= x0) & (xs <= x1)]
        elif kind == "line" and x1 >= xs.min() and x0 <= xs.max():
            t = np.linspace(max(x0, xs.min()), min(x1, xs.max()), 64)
            ys = np.concatenate([np.interp(t, xs, ys), ys[(xs >= x0) & (xs <= x1)]])
        else:
            ys = ys[(xs >= x0) & (xs <= x1)]
        ys = ys[np.isfinite(ys)]
        if not ys.size:
            continue
        disp = ax.transData.transform(np.column_stack([np.full(ys.size, x0), ys]))[:, 1]
        half = width / 2.0 * ax.figure.dpi / 72.0
        e = float(disp.max() + half) if upper else float(disp.min() - half)
        edge = e if edge is None else (max(edge, e) if upper else min(edge, e))
    return edge


def _place(ax, text: str, x: float, marks, *, above: bool, ha: str, color: str, size: float = style.LABEL_PT,
           avoid: Sequence = (), lead: Optional[Tuple[str, str]] = None) -> List:
    """A label anchored at data x ``x``, :data:`CLEAR_MM` above (or below) ``marks`` over its width; its texts.

    ``lead`` is ``(text, colour)``, set before the label on its baseline,
    :data:`NAME_GAP_MM` from it (a series' name before its number): the
    two are placed as one. The width counts :data:`CLEAR_MM` past each
    end, so a mark just beyond an end stays clear as well. A label that
    would cross the axes' right edge moves left until it ends
    :data:`EDGE_MM` inside it, and one that would overlap a text of
    ``avoid`` moves on past it; one that would leave the axes at the top or
    bottom is held inside, even if it then touches a mark. Extents are
    measured at the figure's resolution, the PNG's (:func:`draw`).
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    parts = ([lead] if lead else []) + [(text, color)]
    exts = [_extent(fig, s, size) for s, _ in parts]
    width = sum(e.width for e in exts) + _px(fig, NAME_GAP_MM) * (len(parts) - 1)
    height = max(e.height for e in exts)
    X = float(ax.transData.transform((x, ax.get_ylim()[0]))[0])
    left = {"left": 0.0, "center": 0.5, "right": 1.0}[ha] * width
    X -= max(0.0, X - left + width - (box.x1 - _px(fig, EDGE_MM)))  # inside the right edge
    X += max(0.0, box.x0 + _px(fig, CLEAR_MM) - (X - left))  # clear of the left spine
    inv = ax.transData.inverted()
    # Over its width and its clearance beyond each end, so that a mark just past an end stays clear too.
    x0, x1 = (float(inv.transform((p, 0.0))[0]) for p in (X - left - _px(fig, CLEAR_MM),
                                                           X - left + width + _px(fig, CLEAR_MM)))
    edge = _envelope(ax, marks, x0, x1, upper=above)
    if edge is None:
        Y = (box.y0 + box.y1) / 2.0
    else:
        Y = edge + _px(fig, CLEAR_MM) if above else edge - _px(fig, CLEAR_MM)
    for other in avoid:
        bb = other.get_window_extent(renderer)
        lo_y, hi_y = (Y, Y + height) if above else (Y - height, Y)
        if bb.x0 < X - left + width and X - left < bb.x1 and bb.y0 < hi_y and lo_y < bb.y1:
            Y = bb.y1 + _px(fig, 0.5) if above else bb.y0 - _px(fig, 0.5)
    if above:
        Y = min(max(Y, box.y0 + _px(fig, 0.5)), box.y1 - _px(fig, EDGE_MM) - height)
    else:
        Y = max(min(Y, box.y1 - _px(fig, EDGE_MM)), box.y0 + _px(fig, 0.5) + height)
    # Each part from its left edge, on one baseline (same size and alignment: the same box top or bottom).
    texts, at = [], X - left
    for (s, c), ext in zip(parts, exts):
        xd, yd = inv.transform((at, Y))
        texts.append(style.direct_label(ax, float(xd), float(yd), s, ha="left", va="bottom" if above else "top",
                                        color=c, size=size, zorder=style.Z_TEXT))
        at += ext.width + _px(fig, NAME_GAP_MM)
    return texts


def _pieces(ax, marks):
    """Display boxes ``(x0, x1, y0, y1)`` covering ``marks`` (``"points"`` and ``"line"``, as for :func:`_envelope`).

    A box per point, and per point of each segment sampled every tenth of a
    millimetre, as wide as the mark.
    """
    import numpy as np

    out = [np.empty((0, 4))]
    step = _px(ax.figure, 0.1)
    for kind, xs, ys, width in marks:
        p = ax.transData.transform(np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)]))
        if kind == "line" and len(p) > 1:
            p = np.concatenate([a + np.linspace(0.0, 1.0, 2 + int(np.hypot(*(b - a)) / step))[:, None] * (b - a)
                                for a, b in zip(p[:-1], p[1:])])
        p = p[np.isfinite(p).all(axis=1)]
        half = width / 2.0 * ax.figure.dpi / 72.0
        out.append(np.column_stack([p[:, 0] - half, p[:, 0] + half, p[:, 1] - half, p[:, 1] + half]))
    return np.concatenate(out)


def _beside(ax, text: str, x: float, y: float, marks, *, color: str = style.INK, size: float = style.LABEL_PT):
    """A label centred on data height ``y``, ending :data:`CLEAR_MM` left of data x ``x`` and of the marks there.

    From ``x`` it moves left past every mark of ``marks`` (``"points"`` and
    ``"line"``) that reaches within that clearance of its box, until none
    does, but stays :data:`CLEAR_MM` inside the axes' left edge.
    """
    import numpy as np

    fig = ax.figure
    box = ax.get_window_extent(fig.canvas.get_renderer())
    ext = _extent(fig, text, size)
    clear = _px(fig, CLEAR_MM)
    X, Y = (float(v) for v in ax.transData.transform((x, y)))
    pieces = _pieces(ax, marks)
    pieces = pieces[(pieces[:, 2] < Y + ext.height / 2.0) & (pieces[:, 3] > Y - ext.height / 2.0)]  # at its height
    right = X - clear
    while True:
        hit = pieces[(pieces[:, 0] < right + clear) & (pieces[:, 1] > right - ext.width - clear)]
        if not hit.size:
            break
        right = float(hit[:, 0].min()) - clear
    right = max(right, box.x0 + clear + ext.width)
    xd, yd = ax.transData.inverted().transform((right, Y))
    return style.direct_label(ax, float(xd), float(yd), text, ha="right", va="center", color=color, size=size,
                              zorder=style.Z_TEXT)


def _snap_right(ax, text: str, x: float, edge: float, size: float = style.LABEL_PT) -> Tuple[float, str]:
    """The anchor ``(x, ha)`` for :func:`_place` of a label meant to start at data x ``x``.

    That is ``(x, "left")``, unless the label would then end within
    :data:`ALIGN_MM` of data x ``edge``, or past it: then it ends on
    ``edge``, ``(edge, "right")``.
    """
    fig = ax.figure
    y = ax.get_ylim()[0]
    end = float(ax.transData.transform((x, y))[0]) + _extent(fig, text, size).width
    if end > float(ax.transData.transform((edge, y))[0]) - _px(fig, ALIGN_MM):
        return edge, "right"
    return x, "left"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _trim_sliver(x, y, mm_per_decade: float):
    """A clipped step curve (:func:`style.step_xy`) without a last interval narrower than :data:`MIN_LAST_MM`.

    The window's end can cut the last PSMC interval to a sliver, which
    draws as a riser with a stub: a hook, not a level. The curve then ends
    where that interval starts.
    """
    import numpy as np

    if len(x) < 3 or math.log10(x[-1] / x[-2]) * mm_per_decade >= MIN_LAST_MM:
        return x, y
    return x[:-1], np.append(y[:-2], y[-3])


def _common_steps(curves: Dict[str, object], window: Tuple[float, float], mm_per_decade: float) -> Dict[str, Tuple]:
    """Each curve's step coordinates, clipped to ``window`` and ended together.

    Every curve ends where the first of them ends once its last sliver is
    trimmed (:func:`_trim_sliver`), so that no curve runs on alone past the
    others; the cut is repeated until no curve is left with a sliver.
    """
    if not curves:
        return {}
    end = window[1]
    while True:
        steps = {role: _trim_sliver(*style.step_xy(c.years, c.ne, (window[0], end)), mm_per_decade)
                 for role, c in curves.items()}
        first = min(float(x[-1]) for x, _ in steps.values())
        if not first < end:
            return steps
        end = first


def _arrow_tip(ax, p0, p1, curve, curve_lw: float, head_overhang_pt: float):
    """Where an arrow from data ``p0`` towards ``p1`` should end: its tip's ink on the top edge of ``curve``.

    ``curve`` is a steps-post ``(x, y)``. The tip ends where the arrow first
    meets the curve's top edge, moved back by ``head_overhang_pt`` (how far
    the head's ink passes its path's end); ``p1`` if the arrow never meets
    the curve before it.
    """
    import numpy as np

    to = ax.transData
    a, b = to.transform(p0), to.transform(p1)
    s = np.linspace(0.0, 1.0, 801)
    pts = a + s[:, None] * (b - a)
    xd = to.inverted().transform(pts)[:, 0]
    cx, cy = (np.asarray(v, dtype=float) for v in curve)
    k = np.clip(np.searchsorted(cx, xd, side="right") - 1, 0, len(cy) - 1)
    top = to.transform(np.column_stack([xd, cy[k]]))[:, 1] + curve_lw / 2.0 * ax.figure.dpi / 72.0
    hit = np.flatnonzero(pts[:, 1] <= top)
    if not hit.size:
        return tuple(p1)
    back = head_overhang_pt * ax.figure.dpi / 72.0 / float(np.hypot(*(b - a)))
    end = a + max(0.0, s[hit[0]] - back) * (b - a)
    return tuple(float(v) for v in to.inverted().transform(end))


def curve_panel(ax, routes, fits, cfg: config.E4, full: str) -> None:
    """a: one genome's fits with the example mask, over its fit without missing data."""
    import numpy as np

    window = informative_window(cfg.model, cfg.grid, cfg.scaling.gen_years)
    xlim = decades(*window)
    mask = example_mask(cfg, routes)
    rep = example_replicate(routes, mask) if mask is not None else None
    mm_per_decade = AX_A[2] / math.log10(xlim[1] / xlim[0])

    # role -> the step curve's x and y, clipped to the window; all three end together.
    keys = (("complete", (rep, COMPLETE, WEIGHTED)), ("full", (rep, mask, full)), ("weighted", (rep, mask, WEIGHTED)))
    steps = _common_steps({role: fits[key] for role, key in keys if key in fits}, window, mm_per_decade)
    ys = np.concatenate([y for _, y in steps.values()]) if steps else np.array(YLIM_A)
    lo, hi = decades(float(ys.min()) / 1.3, float(ys.max()) * 1.3)
    style.years_axis(ax, xlim)
    style.ne_axis(ax, (min(YLIM_A[0], lo), max(YLIM_A[1], hi)))
    style.on_baseline(ax.yaxis.label)  # its mathtext N_e where the PNG has it, in the PDF too (as b's theta)

    kw = {"complete": style.underlay_kw(style.UNDERLAY_GREY, share=1.0),  # the reference, wide and pale
          "full": style.overlay_kw(style.ROUTES[FULL].color),
          "weighted": style.overlay_kw(style.ROUTES[WEIGHTED].color, linewidth=style.LW_EMPH,
                                       zorder=style.Z_DATA + 0.5)}
    for role in ("complete", "full", "weighted"):
        if role in steps:
            style.step_line(ax, *steps[role], **kw[role])
    if not steps:
        return

    spec = next(m for m in cfg.masks if m.name == mask)
    n = int(((routes["mask"] == mask) & (routes.route == WEIGHTED)).sum())
    count = f"one of {_count(n)} genomes" if n > 1 else "one genome"
    style.panel_title(ax, f"{spec.structure.capitalize()}, {100 * spec.missing_fraction:.0f}% uncallable ({count})")

    marks = [("steps", x, y, kw[role]["linewidth"]) for role, (x, y) in steps.items()]

    # The shift: theta with full bins over theta without missing data scales N_e and time alike.
    row = routes[(routes["mask"] == mask) & (routes.route == full) & (routes.replicate == rep)]
    r = 1.0 + float(row["theta_rel_complete"].iloc[0]) if len(row) else math.nan
    t0 = SHIFT_AT_YEARS
    shift = None
    if "complete" in steps and "full" in steps and 0.0 < r < 1.0 and xlim[0] < r * t0 and t0 < window[1]:
        c0 = float(fits[(rep, COMPLETE, WEIGHTED)].at(np.array([t0]))[0])
        # The house scaling arrow, one-headed. It points along (t0, c0) -> (r t0, r c0) and stops where it meets
        # the orange curve, its tip's ink on the line's top edge (the scaling is not exact). The head's edge
        # has matplotlib's round join, so its ink passes the tip by half the line's width.
        arrow = dict(style.SCALE_ARROW_PROPS, shrinkA=1.5)
        tip = _arrow_tip(ax, (t0, c0), (r * t0, r * c0), steps["full"], kw["full"]["linewidth"],
                         arrow["linewidth"] / 2.0)
        ax.annotate("", xy=tip, xytext=(t0, c0), arrowprops=arrow, zorder=style.Z_TEXT)
        # The mark's width keeps labels ARROW_CLEAR_MM from the arrow.
        shift = ("line", [tip[0], t0], [tip[1], c0], 2.0 * (ARROW_CLEAR_MM - CLEAR_MM) * style.PT_PER_MM)
        marks.append(shift)

    # Bins taken as full are named for what they are: fq2psmcfa's, or (with no such rows) the VCF's.
    for role, text, color, size, above in (
            ("complete", "No missing data", style.INK2, style.NOTE_PT, True),
            ("weighted", style.ROUTES[WEIGHTED].label, TEXT_COLOR[WEIGHTED], style.LABEL_PT, True),
            ("full", style.ROUTES[full].label, TEXT_COLOR[FULL], style.LABEL_PT, False)):
        if role in steps:
            _place(ax, text, LABEL_X_A[role] * xlim[0], marks, above=above, ha="left", color=color, size=size)
    if shift is not None:
        _place(ax, _fold(r), 1.04 * t0, marks, above=True, ha="left", color=style.INK)


def _under(ax, xs, ys, xb, yb, width_pt: float) -> List[bool]:
    """For each segment of the polyline ``(xs, ys)``: whether it runs within ``width_pt`` of ``(xb, yb)``.

    Distances are vertical, in display units, at points along the segment;
    beyond its ends ``(xb, yb)`` is taken as level.
    """
    import numpy as np

    if len(xb) < 2:
        return [False] * max(0, len(xs) - 1)
    tol = width_pt * ax.figure.dpi / 72.0
    out = []
    for i in range(len(xs) - 1):
        t = np.linspace(xs[i], xs[i + 1], 33)
        a = ax.transData.transform(np.column_stack([t, np.interp(t, (xs[i], xs[i + 1]), (ys[i], ys[i + 1]))]))
        b = ax.transData.transform(np.column_stack([t, np.interp(t, xb, yb)]))
        out.append(bool(np.all(np.abs(a[:, 1] - b[:, 1]) < tol)))
    return out


def _jitter(n: int):
    """Offsets of ``n`` replicates from their median's x, in percentage points: :data:`JITTER` apart, centred."""
    import numpy as np

    return (np.arange(n) - (n - 1) / 2.0) * JITTER


def _x_limits(facets: Sequence, xmax: float) -> Tuple[float, float]:
    """b's x limits: (-0.1, 1.1) x ``xmax``, widened if a mark would come within :data:`X_PAD_MM` of either.

    Marks are the dodged medians (the control's at x itself) with their white
    edge, and the replicate dots around them; ``facets`` holds, per facet,
    ``(mask, x, number of replicates)``.
    """
    lo, hi = -0.1 * xmax, 1.1 * xmax
    r_med = (style.MARKER_MS + MEDIAN_EDGE_PT) / 2.0 / style.PT_PER_MM  # mm
    r_dot = style.REPLICATE_MS / 2.0 / style.PT_PER_MM
    for _ in range(3):  # the room in pp depends on the limits; this settles at once
        pp = (hi - lo) / FACET_W_MM  # pp per mm
        left, right = lo, hi
        for points in facets:
            for mask, x, n in points:
                if mask == COMPLETE:
                    left, right = min(left, x - (r_med + X_PAD_MM) * pp), max(right, x + (r_med + X_PAD_MM) * pp)
                    continue
                reach = max(r_med * pp, float(_jitter(n).max()) + r_dot * pp) + X_PAD_MM * pp
                left, right = min(left, x - DODGE - reach), max(right, x + DODGE + reach)
        lo, hi = min(-0.1 * xmax, left), max(1.1 * xmax, right)
    return lo, hi


def _y_limits(values: Sequence[float]) -> Tuple[float, float]:
    """b's y limits: :data:`YLIM_B`, widened to a round tenth with 0.03 to spare if a value needs it."""
    lo = min(YLIM_B[0], round(0.1 * math.floor((min(values) - 0.03) / 0.1), 10))
    hi = YLIM_B[1] if max(values) <= 1.01 else round(0.1 * math.ceil((max(values) + 0.03) / 0.1), 10)
    return lo, hi


def _facet_axes(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], xmax: float, title: str,
                ylabel: bool) -> None:
    """One facet of b: x ticks every 10 points, y every 0.1 (0.2 on a tall axis), the y grid and the line at 1."""
    import numpy as np
    from matplotlib.ticker import FixedLocator, MultipleLocator

    lo, hi = ylim
    ax.set_xlim(*xlim)
    ax.set_ylim(lo, hi)
    step = 0.1 if hi - lo <= 0.6 else 0.2
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0.0, xmax + 1e-9, 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(style.plain_formatter())
    ax.yaxis.set_major_formatter(style.plain_formatter(decimals=1))  # both steps are tenths: 0.6 to 1.0 (Notation)
    style.value_grid(ax, "y")
    style.ref_line(ax, y=1.0, zorder=style.Z_BAND)
    if ylabel:
        ax.set_ylabel(f"{style.THETA} relative to no missing data")
        style.on_baseline(ax.yaxis.label)  # the letter b sits at its left edge in the PDF too
    else:
        ax.tick_params(labelleft=False)
    style.panel_title(ax, title)


def _draw_route(ax, role: str, xs, med, pts, hidden: Sequence[bool]) -> List:
    """One route of a facet of b: the line of its medians less its ``hidden`` segments, its replicates, its medians.

    ``pts`` holds ``(mask, x, theta ratio by replicate)`` per median;
    returns the marks drawn, for :func:`_place` and :func:`_beside`.
    """
    lw, z_line, z_dots, z_med = B_LOOK[role]
    color = style.ROUTES[role].color
    px, py, marks = [xs[0]], [med[0]], []
    for i, gone in enumerate(hidden):
        if gone:
            px.append(math.nan)
            py.append(math.nan)
        else:
            marks.append(("line", xs[i:i + 2], med[i:i + 2], lw))
        px.append(xs[i + 1])
        py.append(med[i + 1])
    ax.plot(px, py, color=color, linewidth=lw, solid_capstyle="round", zorder=z_line)
    for x, (mask, _, v) in zip(xs, pts):
        if mask == COMPLETE:  # the complete-data fit itself: 1 for every replicate, all under the median
            continue
        jit = x + _jitter(v.size)
        ax.plot(jit, v, linestyle="none", marker="o", markersize=style.REPLICATE_MS, markeredgewidth=0.0,
                color=style.tint(color, REPLICATE_SHARE), zorder=z_dots, clip_on=False)
        marks.append(("points", jit, v, style.REPLICATE_MS))
    ax.plot(xs, med, linestyle="none", marker="o", markersize=style.MARKER_MS, markeredgewidth=MEDIAN_EDGE_PT,
            markeredgecolor=style.SURFACE, color=color, zorder=z_med, clip_on=False)
    marks.append(("points", xs, med, style.MARKER_MS))
    return marks


def dose_panel(axes: Sequence, routes, cfg: config.E4, full: str) -> None:
    """b: theta relative to the complete-data fit against the share uncallable, one facet per structure."""
    import numpy as np

    facets = [facet_masks(cfg, s) for s in STRUCTURES]
    route_of = {FULL: full, WEIGHTED: WEIGHTED}  # the route drawn in each role
    series: Dict[Tuple[int, str], List] = {}  # (facet, role) -> [(mask, x, theta ratio by replicate)]
    values = [1.0]
    for k, masks in enumerate(facets):
        for role, route in route_of.items():
            pts = [(mask, x, _theta_ratio(routes, mask, route)) for mask, x in masks]
            series[(k, role)] = [(mask, x, v) for mask, x, v in pts if v.size]
            values += [float(a) for _, _, v in series[(k, role)] for a in v]
    ylim = _y_limits(values)
    xmax = 10.0 * math.ceil(max([x for masks in facets for _, x in masks] + [30.0]) / 10.0 - 1e-9)
    right = xmax + 1.0  # where b's right-aligned labels end
    counts = [[(mask, x, max([v.size for role in route_of for m, _, v in series[(k, role)] if m == mask] + [0]))
               for mask, x in masks] for k, masks in enumerate(facets)]
    xlim = _x_limits([[c for c in points if c[2]] for points in counts], xmax)
    shown = [k for k, masks in enumerate(facets) if masks]
    first, last = (shown[0], shown[-1]) if shown else (None, None)  # the y axis; the line at 1's name
    named_yet = False  # the routes are named in the first facet that draws them as lines

    for k, (ax, masks) in enumerate(zip(axes, facets)):
        if not masks:
            ax.set_axis_off()
            continue
        _facet_axes(ax, xlim, ylim, xmax, facet_title(routes, cfg, STRUCTURES[k]), k == first)
        # Each route's medians, dodged; the control is the complete-data fit itself, where both lines start.
        lines: Dict[str, Tuple] = {}
        for role, dx in ((FULL, DODGE), (WEIGHTED, -DODGE)):
            pts = series[(k, role)]
            if pts:
                lines[role] = ([x + (0.0 if mask == COMPLETE else dx) for mask, x, _ in pts],
                               [float(np.median(v)) for _, _, v in pts], pts)
        marks: Dict[str, List] = {}
        for role, (xs, med, pts) in lines.items():
            # The orange line where it would run under the blue one is left out: its fringe is all that showed.
            hidden = (_under(ax, xs, med, *lines[WEIGHTED][:2], style.LW_EMPH)
                      if role == FULL and WEIGHTED in lines else [False] * (len(xs) - 1))
            marks[role] = _draw_route(ax, role, xs, med, pts, hidden)

        # Each route named beside its line, and its median with the most missing printed, number first.
        placed: List = []
        named = {role: line for role, line in lines.items()  # the routes with a line in sight
                 if any(kind == "line" for kind, *_ in marks[role])}
        if named and not named_yet:
            named_yet = True
            if FULL in named:  # the number left of the orange median
                xs, med, _ = named[FULL]
                placed.append(_beside(ax, _fold(med[-1]), xs[-1], med[-1], marks[FULL]))
            if WEIGHTED in named:  # the name, then the number, as one under the blue line's end
                xs, med, _ = named[WEIGHTED]
                placed += _place(ax, _fold(med[-1]), right, marks[WEIGHTED], above=False, ha="right",
                                 color=style.INK, avoid=placed, lead=(B_NAMES[WEIGHTED], TEXT_COLOR[WEIGHTED]))
            if FULL in named:  # the name above the orange line from its middle, on the open side of its slope
                xs = named[FULL][0]
                at, ha = _snap_right(ax, B_NAMES[FULL], (xs[0] + xs[-1]) / 2.0, right)
                placed += _place(ax, B_NAMES[FULL], at, marks[FULL], above=True, ha=ha, color=TEXT_COLOR[FULL],
                                 avoid=placed)
        if k == last:  # the line at 1's name, above it and every mark at the right of the last facet
            ref = [("line", list(xlim), [1.0, 1.0], style.LW_THIN)]
            _place(ax, "No missing data", right, ref + [m for ms in marks.values() for m in ms], above=True,
                   ha="right", color=style.INK2, size=style.NOTE_PT, avoid=placed)


def draw(src: Source):
    cfg = src.profile.e4
    routes = src.table("E4_routes")
    fits = curves(src.table("E4_curves"), ["replicate", "mask", "route"])
    full = full_route(routes)

    # Laid out at the PNG's dpi, so that labels are placed by their extents as they print (style, Layout).
    fig = style.figure(W_MM, H_MM, dpi=style.PNG_DPI)
    ax_a = style.axes_mm(fig, *AX_A)
    axes_b = [style.axes_mm(fig, *box) for box in AX_B]
    curve_panel(ax_a, routes, fits, cfg, full)
    dose_panel(axes_b, routes, cfg, full)
    shown = [ax for ax in axes_b if ax.axison]
    if shown:
        style.row_xlabel(fig, shown, "Positions uncallable (%)")
    # a at the figure's left edge, b at the left edge of its ink (its y label), both on the titles' baseline.
    style.panel_letters(fig, [ax_a] + shown[:1])
    return fig


__all__ = ["DRAWN", "draw", "example_mask", "example_replicate", "facet_masks", "facet_title", "full_route"]
