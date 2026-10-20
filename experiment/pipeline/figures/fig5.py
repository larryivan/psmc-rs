"""Supplement Fig S2 (E4): partly callable bins bias PSMC; psmc-rs's per-bin weighting removes most of the bias.

One row of two panels, 183 x 60 mm, in the paper's two colours. There is
no legend: every series is named in place, in its own text colour.

Blue is psmc-rs's default: psmc-rs on the VCF with its callable mask,
each called bin weighted by the callable bases it holds (``--bin-callable
auto``, route ``mask_auto``). Orange is C psmc's input and model: the bins
fq2psmcfa makes from the genome's consensus FASTA, kept unless more than
90% of their bases are uncallable, every called bin taken as full (route
``fq2psmcfa``, fitted by psmc-rs, which fits such input as C psmc does).
These bins equal the VCF's bins read with ``--bin-callable ignore``
(``mask_ignore``), bin for bin and in every fitted value, so only one of
the two is drawn: ``fq2psmcfa``, or ``mask_ignore`` when a table has no
``fq2psmcfa`` rows (:func:`full_route` checks that they agree where both
exist). multihetsep and the VCF read without its mask are not drawn; the
caption gives their numbers.

When missing data are dispersed in short runs, almost every bin stays
called but holds only part of its bases. Taken as full, those bins make
theta per bin low, and every size and time with it, since both scale
with theta. Weighted by their callable bases, they do not. Clustered runs
empty whole bins, which both models drop.

* a: one genome with the example mask (``config.FIGURES.e4_example_mask``,
  dispersed 30%), over the fit of the same genome without missing data
  (wide grey underlay, ``nomissing``/``mask_auto``). Blue (1 pt) is
  weighted bins and orange (0.75 pt) is bins taken as full. The genome is
  the replicate whose weighted fit has the median RMSE of log10 Ne against
  its complete-data fit; the title counts the genomes. Every curve is a
  PSMC step curve clipped to the model's informative window, on log axes
  of N_e and years ago, both in thousands; a last interval that the
  window's end cuts to a sliver (under :data:`MIN_LAST_MM`) is not drawn,
  and all three curves end together. The arrow (the house annotation
  arrow, one head) starts on the complete-data curve at
  :data:`SHIFT_AT_YEARS` and points along the multiplication of both its
  time and its N_e by r = theta with full bins / theta without missing
  data, for this genome; r is printed beside it, and it stops where it
  meets the orange curve (the scaling is close, not exact).
* b: theta relative to the fit of the same genome without missing data,
  in percent, against the share of positions uncallable. Two facets share
  one y axis: clustered runs on the left and dispersed runs on the right,
  each headed by its mean run length. Large white-edged circles are the
  medians over the replicates, joined by lines, each route's
  :data:`DODGE` beside its x (blue left, orange right; 0.73 mm, under a
  median's width) so that both show in the clustered facet, where the two
  routes agree. The small pale dots are the replicates, all of them,
  :data:`JITTER` apart around their median in replicate order, beneath the
  lines and medians. In the clustered facet the orange line would run under
  the blue one, and a segment that lies within a blue line's width of it
  all along is not drawn: only its fringe showed. The control (0%) is the
  complete-data fit itself, 0 for every replicate, so both lines start
  from one point there, undodged. Every mark stays :data:`X_PAD_MM` inside
  the x limits.

Each label sits :data:`CLEAR_MM` above or below the marks it names (the
arrow included), over its own width as it prints and that clearance past
each end, and stays inside its axes (:func:`_place`).

Everything comes from ``E4_routes`` (theta, and the RMSE that picks the
example) and ``E4_curves``; mask structures, missing fractions and run
lengths from the profile's ``E4.masks``. A mask, route or replicate
missing from the tables is left out.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config
from . import style
from .data import Source, curves, decades, fmt_bp, informative_window

WEIGHTED = "mask_auto"  # psmc-rs's default: each called bin weighted by its callable bases
FULL = "fq2psmcfa"  # every called bin taken as full: C psmc's input and model
FULL_FALLBACK = "mask_ignore"  # the same bins and fit, from the VCF (drawn when FULL has no rows)
DRAWN = (WEIGHTED, FULL)
COMPLETE = config.E4_CONTROL_MASK  # the no-missing control; its WEIGHTED fit is the complete-data fit
STRUCTURES = ("clustered", "dispersed")  # b's facets, left to right

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

# b, in percentage points (1 pp is about 1.05 mm across a facet). Each
# route's median sits DODGE beside its x, blue left and orange right, so
# that both show in the clustered facet, where the routes agree; 0.7 pp is
# 0.73 mm, less than a median's diameter, so every median reads at its tick.
DODGE = 0.7
# The replicates sit in replicate order around their median, JITTER apart
# (x + dx + (i - (n - 1) / 2) JITTER), under it and the lines: those nearest
# the median tuck partly under its white-edged circle.
JITTER = 0.35
X_PAD_MM = 0.5  # b's least room between a mark's outer edge and the facet's x limits
MEDIAN_EDGE_PT = 0.5  # the white edge of b's medians, which keeps them apart from lines, dots and each other
# b's replicate dots are style.REPLICATE_MS (1.6 pt), beside a 3 pt median.
# At style.REPLICATE_ALPHA (0.35) a dot that small fades into the white, so
# they take their colour at 55%, still clearly paler than the median.
REPLICATE_SHARE = 0.55
# b's routes: (line width, z of the line, of the replicates, of the medians). The pale replicates lie beneath
# (orange under blue), then the lines, then the medians, orange first and blue over it. The orange medians lie
# over the blue line, which runs past them where the routes agree; their white edge keeps them apart from it.
B_LOOK = {FULL: (style.LW_DATA, style.Z_DATA, style.Z_UNDERLAY, style.Z_DATA + 0.6),
          WEIGHTED: (style.LW_EMPH, style.Z_DATA + 0.5, style.Z_UNDERLAY + 0.1, style.Z_DATA + 0.7)}

# Series names are set in the series' text colour (the house exception for
# figures without a legend), the same darker step for both routes as in Fig 2.
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


def _theta_change(routes, mask: str, route: str):
    """100 x theta_rel_complete for ``mask`` and ``route``, by replicate (finite values only)."""
    import numpy as np

    sub = routes[(routes["mask"] == mask) & (routes.route == route)].sort_values("replicate")
    v = 100.0 * sub["theta_rel_complete"].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def _count(n: int) -> str:
    """``n`` in words up to nine, in figures above."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else f"{n:,}"


# ---------------------------------------------------------------------------
# Labels placed from the data
# ---------------------------------------------------------------------------


def _px(fig, mm: float) -> float:
    """``mm`` in display pixels."""
    return mm * style.MM * fig.dpi


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
           avoid: Sequence = ()):
    """A label anchored at data x ``x``, :data:`CLEAR_MM` above (or below) ``marks`` over its width.

    The width counts :data:`CLEAR_MM` past each end, so a mark just beyond
    an end stays clear as well. A label that would cross the axes' right
    edge moves left until it ends :data:`EDGE_MM` inside it, and one that
    would overlap a label of ``avoid`` moves on past it; one that would
    leave the axes at the top or bottom is held inside, even if it then
    touches a mark. Extents are measured at the figure's resolution, which
    :func:`draw` raises to the PNG's while it places labels.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    probe = fig.text(0.0, 0.0, text, fontsize=size)
    ext = probe.get_window_extent(renderer)
    probe.remove()
    X = float(ax.transData.transform((x, ax.get_ylim()[0]))[0])
    left = {"left": 0.0, "center": 0.5, "right": 1.0}[ha] * ext.width
    X -= max(0.0, X - left + ext.width - (box.x1 - _px(fig, EDGE_MM)))  # inside the right edge
    X += max(0.0, box.x0 + _px(fig, CLEAR_MM) - (X - left))  # clear of the left spine
    inv = ax.transData.inverted()
    # Over its width and its clearance beyond each end, so that a mark just past an end stays clear too.
    x0, x1 = (float(inv.transform((p, 0.0))[0]) for p in (X - left - _px(fig, CLEAR_MM),
                                                           X - left + ext.width + _px(fig, CLEAR_MM)))
    edge = _envelope(ax, marks, x0, x1, upper=above)
    if edge is None:
        Y = (box.y0 + box.y1) / 2.0
    else:
        Y = edge + _px(fig, CLEAR_MM) if above else edge - _px(fig, CLEAR_MM)
    for other in avoid:
        bb = other.get_window_extent(renderer)
        lo_y, hi_y = (Y, Y + ext.height) if above else (Y - ext.height, Y)
        if bb.x0 < X - left + ext.width and X - left < bb.x1 and bb.y0 < hi_y and lo_y < bb.y1:
            Y = bb.y1 + _px(fig, 0.5) if above else bb.y0 - _px(fig, 0.5)
    if above:
        Y = min(max(Y, box.y0 + _px(fig, 0.5)), box.y1 - _px(fig, EDGE_MM) - ext.height)
    else:
        Y = max(min(Y, box.y1 - _px(fig, EDGE_MM)), box.y0 + _px(fig, 0.5) + ext.height)
    xd, yd = inv.transform((X, Y))
    return style.direct_label(ax, float(xd), float(yd), text, ha=ha, va="bottom" if above else "top",
                              color=color, size=size, zorder=style.Z_TEXT)


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
        # The house annotation arrow, one-headed. It points along (t0, c0) -> (r t0, r c0) and stops where
        # it meets the orange curve, its tip's ink on the line's top edge (the scaling is not exact).
        # The head's edge has matplotlib's round join, so its ink passes the tip by half the line's width.
        arrow = dict(style.ARROW_PROPS, arrowstyle="-|>", shrinkA=1.5)
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
        _place(ax, f"{style.TIMES}{r:.2f}", 1.04 * t0, marks, above=True, ha="left", color=style.INK)


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


def _facet_axes(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], xmax: float, title: str,
                ylabel: bool) -> None:
    """One facet of b: ticks every 10 points (y every 20 on a tall axis), the y grid and the zero line."""
    import numpy as np
    from matplotlib.ticker import FixedLocator, MultipleLocator

    lo, hi = ylim
    ax.set_xlim(*xlim)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0.0, xmax + 1e-9, 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator(10.0 if hi - lo <= 60.0 else 20.0))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(style.plain_formatter())
    style.value_grid(ax, "y")
    style.ref_line(ax, y=0.0, zorder=style.Z_BAND)
    if ylabel:
        ax.set_ylabel(f"Change in {style.THETA} (%)")
    else:
        ax.tick_params(labelleft=False)
    style.panel_title(ax, title)


def _draw_route(ax, role: str, xs, med, pts, hidden: Sequence[bool]) -> List:
    """One route of a facet of b: the line of its medians less its ``hidden`` segments, its replicates, its medians.

    ``pts`` holds ``(mask, x, theta change by replicate)`` per median;
    returns the marks drawn, for :func:`_place`.
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
        if mask == COMPLETE:  # the complete-data fit itself: 0 for every replicate, all under the median
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
    """b: the change in theta against the share uncallable, one facet per structure, on one y axis."""
    import numpy as np

    facets = [facet_masks(cfg, s) for s in STRUCTURES]
    route_of = {FULL: full, WEIGHTED: WEIGHTED}  # the route drawn in each role
    series: Dict[Tuple[int, str], List] = {}  # (facet, role) -> [(mask, x, theta change by replicate)]
    values = [0.0]
    for k, masks in enumerate(facets):
        for role, route in route_of.items():
            pts = [(mask, x, _theta_change(routes, mask, route)) for mask, x in masks]
            series[(k, role)] = [(mask, x, v) for mask, x, v in pts if v.size]
            values += [float(a) for _, _, v in series[(k, role)] for a in v]
    ylim = (min(-40.0, 10.0 * math.floor((min(values) - 3.0) / 10.0)),
            4.0 if max(values) <= 1.0 else 10.0 * math.ceil((max(values) + 3.0) / 10.0))
    xmax = 10.0 * math.ceil(max([x for masks in facets for _, x in masks] + [30.0]) / 10.0 - 1e-9)
    right = xmax + 1.0  # where b's labels end
    counts = [[(mask, x, max([v.size for role in route_of for m, _, v in series[(k, role)] if m == mask] + [0]))
               for mask, x in masks] for k, masks in enumerate(facets)]
    xlim = _x_limits([[c for c in points if c[2]] for points in counts], xmax)
    shown = [k for k, masks in enumerate(facets) if masks]
    first, last = (shown[0], shown[-1]) if shown else (None, None)  # y axis and zero line's name; routes' names

    for k, (ax, masks) in enumerate(zip(axes, facets)):
        if not masks:
            ax.set_axis_off()
            continue
        run = next(m.mean_run_bp for m in cfg.masks if m.structure == STRUCTURES[k])
        _facet_axes(ax, xlim, ylim, xmax, f"{STRUCTURES[k].capitalize()} ({fmt_bp(run)} runs)", k == first)
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

        if k == first:  # the zero line's name, above it and the data at the right of the first facet
            zero = [("line", list(xlim), [0.0, 0.0], style.LW_THIN)]
            _place(ax, "No missing data", right, zero + [m for ms in marks.values() for m in ms], above=True,
                   ha="right", color=style.INK2, size=style.NOTE_PT)
        if k == last:  # each route's name under its last median and replicates, both right-aligned
            names: List = []
            for role in (WEIGHTED, FULL):
                if role in marks:
                    names.append(_place(ax, B_NAMES[role], right, marks[role], above=False, ha="right",
                                        color=TEXT_COLOR[role], avoid=names))


def draw(src: Source):
    cfg = src.profile.e4
    routes = src.table("E4_routes")
    fits = curves(src.table("E4_curves"), ["replicate", "mask", "route"])
    full = full_route(routes)

    fig = style.figure(W_MM, H_MM)
    dpi = fig.dpi
    # Labels are placed by their extents as they print: at the default 100 dpi, Agg's hinted metrics make a
    # 6 pt label about 7% narrower.
    fig.set_dpi(style.PNG_DPI)
    ax_a = style.axes_mm(fig, *AX_A)
    axes_b = [style.axes_mm(fig, *box) for box in AX_B]
    curve_panel(ax_a, routes, fits, cfg, full)
    dose_panel(axes_b, routes, cfg, full)
    shown = [ax for ax in axes_b if ax.axison]
    if shown:
        style.row_xlabel(fig, shown, "Positions uncallable (%)")
    # a at the figure's left edge, b at the left edge of its ink (its y label), both on the titles' baseline.
    style.panel_letters(fig, [ax_a] + shown[:1])
    fig.set_dpi(dpi)
    return fig


__all__ = ["DRAWN", "draw", "example_mask", "example_replicate", "facet_masks", "full_route"]
