"""Supplement Fig S3 (E5): the bootstrap measures the sampling variation of a PSMC fit, not PSMC's bias.

The bottleneck model is simulated as many independent genomes, each fitted
once: their spread at each time is the true sampling distribution of the
estimate. The first genomes are bootstrapped at two block sizes (splitfa's
trunks), named by their number of blocks per genome, which is what drives
the bootstrap: the most blocks in blue, the fewest in light blue
(``style.TRUNK_COLORS``) where both are drawn (b and d); a and c draw the
most blocks only, as their titles say, so that no colour changes meaning
between panels. Two rows of two panels, 183 x 107 mm, each panel titled on
its letter's baseline, with no figure legend: a over c on one time axis,
the model's informative window (its label under c); b over d. c and d
share a coverage axis from -3 to 103%, ticked at 0, 50 and 100: padded
alike at both ends, so that coverages of 0 and 100 lie clear of the axes'
edges, and none reaches up to a title.

* a, "One genome, 60 blocks": one bootstrapped genome
  (``config.E5.example_dataset``, else the first): the truth (dashed ink),
  the genome's fit (blue) and its bootstrap 95% band (psmc-rs's band,
  ``style.RS_BAND``, the panel's only fill), and the 95% range of the
  independent fits (two thin grey lines), all as PSMC step curves, each
  named in place in its text colour: "Truth" (ink), "Its fit and bootstrap
  95% band" (blue), "Independent fits, 95% range" (grey). Band and range
  are about as wide as each other, and both miss the truth where PSMC is
  biased. The N_e axis spans half the truth's smallest size to five times
  its largest, widened to round values that hold the band and the range
  whole: nothing runs off the panel.
* b, "Spread at each time": calibration of the bootstrap's spread: at each
  time of the window, the bootstrap SD of log10 Ne (median over the
  genomes) against the true SD (over the independent fits), one dot per
  distinct pair of values over the window's times (the times within one
  PSMC interval share theirs), on log axes with the identity line. The
  grey band around the identity line is where a bootstrap SD equal to the
  true SD would fall: the true SD is itself estimated, from the
  independent fits, and the band is that estimate's chi-square 95%
  interval, whose bounds are fixed multiples of the SD (the band takes
  their median over the window). A dot under the band is a bootstrap
  narrower than the true spread. The key gives each block count's median
  ratio over the window (``E5_summary``), its counts right-aligned so that
  the ratios form one column. Inside the panel and with numbers, it is the
  house's one written exception to its key rule (the reasons are beside
  its call in ``_panel_b``).
* c, "Coverage at each time, 60 blocks": at each time of a's axis, the
  share of the bootstrapped genomes whose band holds the fits' mean (the
  geometric mean of the independent fits; a blue line, the colour of the
  most blocks) or the truth (dashed ink, the truth's style as in a), named
  in place, "Fits' mean" in blue and "Truth" in ink; the grey line is the
  nominal coverage. Where the two are equal (mostly at 100%) the truth's
  dash lies over the blue, the truth's written exception to the house's
  underlay rule: there both hold the same share. It shows where the bands
  miss the truth: where PSMC smooths the truth's sudden changes in size,
  while they still hold the fits' mean.
* d, "Coverage of each genome": for each genome (a dot), the share of the
  window's times at which its band holds the fits' mean or the truth, for
  each block count (named under its strip); bars are the means
  (``E5_summary``), the grey line the nominal coverage. Each strip's dots
  are the house's deterministic swarm, every genome a dot of its own that
  overlaps no other, within the house's strip width. The gap between the
  two groups is PSMC's bias, not the bootstrap's.

Text is placed from the fonts' own metrics, not from what a renderer
measures (the PDF backend hints widths at 72 dpi, the PNG at its own dpi),
so the figure is the same in every output. The axis labels hang at fixed
distances from their spines: the y labels of a column at one distance (so
that they line up), each row's x labels and d's group names on one
baseline (their mathtext on it in every output, ``style.on_baseline``).
The letters a and c sit at ``style.LETTER_X_MM``, b and d at one x at the
ink of their column's y labels, and the left column ends ``GAP_MM`` of
white before them; the top row starts the house's ``style.TOP_MM`` below
the figure's top. The names in a and c are placed from the data,
``LABEL_CLEAR_MM`` clear of every curve under them: in a the truth's under
its lowest step, the fit's and band's over the fit's highest step after
its lowest, the range's under the truth's longest step; in c one over the
other in the widest stretch where the bands hold the truth least, the
fits' mean's below its line and the nominal line where it sits closest to
them, the truth's above its own.

Every number comes from E5's tables: ``E5_summary`` (block counts, median
SD ratios, window coverages), ``E5_sampling`` (truth, the fits' range and
mean), ``E5_bands`` (the genomes' fits and bands, coverage of the truth),
``E5_sd_ratio`` (the SDs) and ``E5_coverage`` (c's shares). The per-genome
coverages are recomputed from ``E5_bands`` and checked against
``E5_summary``, and ``E5_coverage``'s pointwise shares against both: a
disagreement raises ``ValueError`` rather than drawing numbers the text
does not cite.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import style
from .data import Source, decades

# Layout (mm). Two rows whose axes share their tops and heights, the top row style.TOP_MM below the figure's top
# and the rows style.ROW_GAP_TITLED_MM apart (b's x label above, letters and titles below): a and b TOP_ROW_MM
# high, c and d the rest. The right column (b over d) ends style.RIGHT_MM from the right edge and is as wide as b
# is high (b is square: a calibration plot). The left column (a over c) starts at style.LEFT_MM; its width follows
# from the text (_layout): GAP_MM of white between its ink and the right column's y labels. 107 mm keeps the
# figure a whole number of millimetres, and a within style.NE_ASPECT.
W_MM, H_MM = style.WIDTH_MM["double"], 107.0
TOP_ROW_MM = 56.0
LOWER_MM = style.TOP_MM + TOP_ROW_MM + style.ROW_GAP_TITLED_MM  # the bottom row's top
BOTTOM_ROW_MM = H_MM - LOWER_MM - style.BOTTOM_MM  # 27.15
RIGHT_W_MM = TOP_ROW_MM
RIGHT_X_MM = W_MM - style.RIGHT_MM - RIGHT_W_MM  # the right column's left spines
GAP_MM = 6.5

# a's and c's curves are E5's fits and truth read at the grid's times (``curve.at(grid)``), not PSMC's interval
# starts, so they are drawn steps-mid, not steps-post as the set's other N_e(t) curves: a size change happened
# somewhere between two grid times, and the midpoint puts each riser within half a grid step of it (0.5% of a
# decade in production; steps-post would put it up to a whole step late), e.g. the truth's at 20, 30, 100 and
# 1,000 thousand years.
STEP = "mid"
# Names drawn in place (a and c; d's group names), in the words of the caption.
TRUTH_NAME = style.TRUTH.label
RANGE_NAME = "Independent fits, 95% range"
BAND_NAME = "Its fit and bootstrap 95% band"
MEAN_NAME = "Fits’ mean"
LABEL_CLEAR_MM = 1.0  # a name's clearance from the curves under (or over) it, and from the axes' edges
LABEL_END_MM = 2.0  # and past each of its ends, where a riser may stand beside it
# b's key: counts padded with figure spaces (as wide as a digit) so that they end, and the ratios
# start, at one x; no-break spaces before the ratio, which an SVG keeps.
FIGURE_SPACE = " "
RATIO_SEP = "  "
# d: strips in data units. The block counts of a group are ARM_STEP apart (the strip spacing) and the groups
# GROUP_GAP further; the mean bars span the house's +-style.STRIP_BAR_HALF of the strip spacing. A strip's dots are
# the house's deterministic swarm for a few clustered points (_swarm), not a random jitter, so that each genome is
# a dot of its own (jittered at random, a strip's 10 genomes printed as about 7 marks): no dot overlaps another,
# each as near its strip's centre as that allows. On E5's tables every swarm stays within the house's
# +-style.STRIP_JITTER of the strip spacing (+-1.6 mm); one that could not would be drawn wider, never overlapping.
ARM_STEP = 0.42
GROUP_GAP = 0.80
X_PAD = 0.30
SWARM_STEP_MM = 0.05  # the swarm's grid of offsets
DOT = {"linestyle": "none", "marker": "o", "markersize": style.STRIP_MS, "markeredgewidth": 0.0}  # b's and d's
# c's and d's axis (%), padded alike at both ends: 0 and 100 drawn clear of the axes' edges, so that no coverage
# lies on the bottom spine or reaches up to the titles.
COVERAGE_LIM = (-3.0, 103.0)
COVERAGE_TICKS = (0, 50, 100)
NOMINAL_DROP_MM = 0.8  # d's 'Nominal 95%' hangs this far below its line
TOL = 1e-9  # recomputed coverages against E5_summary


def _arm_colors(arms: Sequence[int]) -> Dict[int, str]:
    """The most blocks blue, the fewest light blue (``style.TRUNK_COLORS``)."""
    last = len(style.TRUNK_COLORS) - 1
    return {t: style.TRUNK_COLORS[min(i, last)] for i, t in enumerate(arms)}


def _text_color(color: str) -> str:
    """The colour a series' name is set in: psmc-rs's blue as its text colour (``style.BLUE_TEXT``)."""
    return style.BLUE_TEXT if color == style.BLUE else color


def genome_coverage(bands, sampling):
    """Per genome and block size, the share of times its band holds the fits' mean and the truth.

    ``DataFrame`` indexed by ``(trunk_bins, dataset)`` with columns ``mean``
    (the band holds the geometric mean of the independent fits,
    ``10 ** mean_log10_ne``, bounds included, as E5 counts it) and
    ``truth`` (``covers_truth``).
    """
    gm = 10.0 ** sampling.set_index("point")["mean_log10_ne"]
    bb = bands.assign(gm=bands["point"].map(gm))
    bb["cov_mean"] = ((bb["boot_lo_ne"] <= bb["gm"]) & (bb["gm"] <= bb["boot_hi_ne"])).astype(float)
    bb["cov_truth"] = bb["covers_truth"].astype(float)
    return bb.groupby(["trunk_bins", "dataset"]).agg(mean=("cov_mean", "mean"), truth=("cov_truth", "mean"))


def _check_coverage(per, cov, rows: Dict, arms: Sequence[int]) -> None:
    """Raise ``ValueError`` if the per-genome coverages or ``E5_coverage`` disagree with ``E5_summary``."""
    for t in arms:
        want = {"mean": float(rows[t].coverage_of_fit_mean_mean), "truth": float(rows[t].coverage_mean)}
        got = per.xs(t, level="trunk_bins")
        pointwise = cov[cov["trunk_bins"] == t]
        found = {("E5_bands", "mean"): got["mean"].mean(), ("E5_bands", "truth"): got["truth"].mean(),
                 ("E5_coverage", "mean"): pointwise["coverage_fit_mean"].mean(),
                 ("E5_coverage", "truth"): pointwise["coverage"].mean()}
        for (table, what), v in found.items():
            if math.isnan(float(v)) and math.isnan(want[what]):
                continue
            if not abs(float(v) - want[what]) <= TOL:
                raise ValueError(f"E5 coverage of arm {t} ({what}) from {table} is {float(v):.6g}, "
                                 f"but E5_summary gives {want[what]:.6g}")


# ---------------------------------------------------------------------------
# Text metrics (points), from the font files: the same in every backend
# ---------------------------------------------------------------------------


def _prop(size: float, weight: str = "normal"):
    from matplotlib.font_manager import FontProperties

    return FontProperties(size=size, weight=weight)


def _box(text: str, size: float) -> Tuple[float, float, float]:
    """(width, ascent, descent) of ``text``'s line box as matplotlib lays it out, unhinted.

    A line is at least as tall as the font's typographic ascender and
    descender (``Text._get_layout``); mathtext's box is its own.
    """
    from matplotlib.cbook import is_math_text
    from matplotlib.font_manager import findfont
    from matplotlib.ft2font import FT2Font
    from matplotlib.textpath import TextToPath

    prop = _prop(size)
    w, h, d = TextToPath().get_text_width_height_descent(text, prop, ismath=is_math_text(text))
    font = FT2Font(findfont(prop))
    os2, em = font.get_sfnt_table("OS/2"), font.get_sfnt_table("head")["unitsPerEm"]
    return w, max(h - d, os2["sTypoAscender"] * size / em), max(d, -os2["sTypoDescender"] * size / em)


def _ink(text: str, size: float, weight: str = "normal"):
    """The ink ``Bbox`` of ``text`` set at the origin on its baseline, in points."""
    from matplotlib.textpath import TextPath

    return TextPath((0.0, 0.0), text, size=size, prop=_prop(size, weight)).get_extents()


def _tick_texts(axis) -> List[Tuple[float, str]]:
    """(axes fraction, label) of each major tick ``axis`` draws (inside its view limits)."""
    import numpy as np

    lo, hi = sorted(axis.get_view_interval())
    tr = axis.get_transform()
    t_lo, t_hi = np.ravel(tr.transform(np.array([lo, hi], dtype=float)))
    locs = [float(v) for v in axis.get_majorticklocs()]
    span = t_hi - t_lo
    keep, fracs = [], []
    for v, tv in zip(locs, np.ravel(tr.transform(np.array(locs, dtype=float))) if locs else []):
        f = (tv - t_lo) / span
        if -1e-9 <= f <= 1.0 + 1e-9:
            keep.append(v)
            fracs.append(float(f))
    labels = axis.get_major_formatter().format_ticks(keep) if keep else []
    return [(f, s) for f, s in zip(fracs, labels) if s]


def _ylabel_pt(ax) -> Tuple[float, float]:
    """(baseline, ink) of ``ax``'s y label: how far left of the spine it sits, in points.

    matplotlib's own rule, from unhinted metrics: the label's box starts
    ``axes.labelpad`` left of the widest tick label.
    """
    import matplotlib as mpl

    widths = [_box(s, style.TICK_PT)[0] for _, s in _tick_texts(ax.yaxis)]
    ticks = mpl.rcParams["ytick.major.size"] + mpl.rcParams["ytick.major.pad"] + max(widths, default=0.0)
    label = ax.yaxis.label.get_text()
    base = ticks + ax.yaxis.labelpad + _box(label, style.TEXT_PT)[2]
    return base, base + _ink(label, style.TEXT_PT).y1


def _xlabel_drop_pt() -> float:
    """How far below the bottom spine every x label's baseline sits (matplotlib's rule, unhinted)."""
    import matplotlib as mpl

    _, ta, td = _box("0", style.TICK_PT)
    _, la, _ = _box("x", style.TEXT_PT)
    return mpl.rcParams["xtick.major.size"] + mpl.rcParams["xtick.major.pad"] + ta + td + mpl.rcParams[
        "axes.labelpad"] + la


def _right_ink_mm(ax, width_mm: float) -> float:
    """How far right of ``ax``'s right edge its ink reaches: the bottom spine's square cap or a tick label, in mm."""
    import matplotlib as mpl

    out = mpl.rcParams["axes.linewidth"] / 2.0 / style.PT_PER_MM
    for f, s in _tick_texts(ax.xaxis):
        w = _box(s, style.TICK_PT)[0]
        out = max(out, f * width_mm - w / 2.0 / style.PT_PER_MM + _ink(s, style.TICK_PT).x1 / style.PT_PER_MM
                  - width_mm)
    return out


# ---------------------------------------------------------------------------
# Names in place (a and c)
# ---------------------------------------------------------------------------


def _runs(x, y) -> List[Tuple[float, float, float]]:
    """(value, start, end) of each stretch over which a steps-mid line holds one value, left to right.

    Drawn steps-mid, ``y[k]`` holds from halfway to ``x[k - 1]`` to halfway
    to ``x[k + 1]`` (from ``x[0]``, to ``x[-1]``); NaN values hold nowhere.
    """
    import numpy as np

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not len(x):
        return []
    mid = (x[1:] + x[:-1]) / 2.0
    starts, ends = np.concatenate(([x[0]], mid)), np.concatenate((mid, [x[-1]]))
    runs: List[Tuple[float, float, float]] = []
    for k in range(len(x)):
        if not math.isfinite(y[k]):
            continue
        if runs and runs[-1][0] == y[k] and runs[-1][2] == starts[k]:
            runs[-1] = (runs[-1][0], runs[-1][1], float(ends[k]))
        else:
            runs.append((float(y[k]), float(starts[k]), float(ends[k])))
    return runs


def _widest(runs) -> Tuple[float, float, float]:
    """The run that spans the most of a log time axis."""
    return max(runs, key=lambda r: math.log(r[2] / r[1]) if r[1] > 0 else 0.0)


def _centre(run: Tuple[float, float, float]) -> float:
    """A run's middle on a log time axis."""
    return math.sqrt(run[1] * run[2])


def _envelope(marks, x0: float, x1: float, upper: bool) -> Optional[float]:
    """The highest (``upper``) or lowest value the steps-mid ``marks`` ((x, y) pairs) take between ``x0`` and ``x1``."""
    vals = [v for x, y in marks for v, a, b in _runs(x, y) if b >= x0 and a <= x1]
    return (max(vals) if upper else min(vals)) if vals else None


def _name(ax, x: float, text: str, marks, *, above: bool, color: str, within: Optional[Tuple[float, float]] = None):
    """``text`` (6 pt) centred at data x ``x``, :data:`LABEL_CLEAR_MM` above or below every mark under it.

    Under its width and :data:`LABEL_END_MM` past each end, the name's ink
    clears the highest mark (``above``) or the lowest by
    :data:`LABEL_CLEAR_MM` and half a data line's width; it is held inside
    the axes, even if it then touches a mark. With ``within`` (data x), the
    name moves inside that stretch to where it sits closest to the marks it
    clears, the nearest such place to ``x``. Placed on the axes as laid out,
    from the font's metrics, so the drawn name is where it was placed in
    every output.
    """
    import numpy as np

    fig = ax.figure
    px = fig.dpi / 72.0 * style.PT_PER_MM  # pixels per millimetre
    half = _box(text, style.LABEL_PT)[0] / style.PT_PER_MM * px / 2.0
    ink = _ink(text, style.LABEL_PT)
    rise, depth = ink.y1 / style.PT_PER_MM * px, -ink.y0 / style.PT_PER_MM * px  # the ink about its baseline
    clear = LABEL_CLEAR_MM * px + style.LW_DATA / 2.0 / style.PT_PER_MM * px
    box = ax.bbox
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform

    def baseline(X: float) -> float:
        x0, x1 = (float(to_data((X + s * (half + LABEL_END_MM * px), box.y0))[0]) for s in (-1.0, 1.0))
        edge = _envelope(marks, x0, x1, upper=above)
        if edge is None:
            return (box.y0 + box.y1) / 2.0
        e = float(to_px((x0, edge))[1])
        return e + clear + depth if above else e - clear - rise

    lo, hi = box.x0 + LABEL_END_MM * px + half, box.x1 - LABEL_END_MM * px - half  # its centre, inside the axes
    X = min(max(float(to_px((x, 1.0))[0]), lo), hi)
    if within is not None:
        a, b = (min(max(float(to_px((v, 1.0))[0]), lo), hi) for v in within)
        # Its whole width inside the stretch, if it fits; one candidate per tenth of a millimetre.
        a, b = (a + half, b - half) if b - a > 2.0 * half else ((a + b) / 2.0,) * 2
        options = sorted(np.linspace(a, b, max(int((b - a) / (0.1 * px)), 1) + 1), key=lambda v: abs(v - X))
        heights = [baseline(v) for v in options]
        best = min(heights) if above else max(heights)
        X = next(v for v, h in zip(options, heights) if abs(h - best) <= 0.01 * px)
    base = min(max(baseline(X), box.y0 + LABEL_CLEAR_MM * px + depth), box.y1 - LABEL_CLEAR_MM * px - rise)
    xd, yd = to_data((X, base))
    return style.direct_label(ax, float(xd), float(yd), text, ha="center", va="baseline", color=color,
                              zorder=style.Z_TEXT)


def _name_a(ax, sampling, example, marks, color: str) -> None:
    """Name a's series in place: the truth under its lowest step, the fit and band over the fit's highest step after
    its lowest, the independent fits' range under the truth's longest step."""
    truth = _runs(sampling["years"], sampling["true_ne"])
    if truth:
        lowest = min(r[0] for r in truth)
        _name(ax, _centre(_widest([r for r in truth if r[0] == lowest])), TRUTH_NAME, marks, above=False,
              color=style.INK)
        _name(ax, _centre(_widest(truth)), RANGE_NAME, marks, above=False, color=style.INK2)
    fit = _runs(example["years"], example["fit_ne"]) if example is not None else []
    if fit:
        k = min(range(len(fit)), key=lambda i: fit[i][0])
        after = fit[k + 1:] or fit
        _name(ax, _centre(max(after, key=lambda r: r[0])), BAND_NAME, marks, above=True, color=_text_color(color))


def _name_c(ax, lines: Dict, color: str, nominal: float) -> None:
    """Name c's lines, one name over the other, in the widest stretch where the bands hold the truth least.

    The fits' mean's name below its line and the nominal line (``nominal``,
    %), where in that stretch it sits closest to them (nearest the
    stretch's middle); the truth's above its own line, at the same x.
    """
    runs = _runs(*lines["truth"])
    if not runs:
        return
    lowest = min(r[0] for r in runs)
    trough = _widest([r for r in runs if r[0] == lowest])
    x = lines["mean"][0]
    ref = ([x[0], x[-1]], [nominal, nominal])  # the nominal line: under a fits' mean at 100%, the name would sit on it
    mean = _name(ax, _centre(trough), MEAN_NAME, [lines["mean"], ref], above=False, color=_text_color(color),
                 within=trough[1:])
    _name(ax, mean.get_position()[0], TRUTH_NAME, [lines["truth"]], above=True, color=style.INK)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _time_lim(years) -> Tuple[float, float]:
    """a's and c's time axis: the window, from the power of ten below its start when that is within 10% of it."""
    first, last = float(years[0]), float(years[-1])
    lo = 10.0 ** math.floor(math.log10(first) + 1e-9)
    return (lo if first / lo < 1.1 else first, last)


def _coverage_axis(ax) -> None:
    """c's and d's coverage axis, in %."""
    from matplotlib.ticker import FixedLocator

    ax.set_ylim(*COVERAGE_LIM)
    ax.yaxis.set_major_locator(FixedLocator(list(COVERAGE_TICKS)))
    ax.set_ylabel("Coverage (%)")


def _panel_a(ax, sampling, example, color: str, xlim: Tuple[float, float]) -> List:
    """a: the truth, the independent fits' range and (``example``) one genome's fit and band; return its marks.

    The marks are the (x, y) of every curve and band edge drawn, which
    the names keep clear of; the N_e axis holds all of them.
    """
    import numpy as np

    years = sampling["years"].to_numpy(dtype=float)
    true_ne = sampling["true_ne"].to_numpy(dtype=float)
    step = {"drawstyle": f"steps-{STEP}"}  # see STEP: grid times, not interval starts
    marks = [(years, true_ne)] + [(years, sampling[e].to_numpy(dtype=float)) for e in ("lo_ne", "hi_ne")]
    if example is not None and len(example) > 0:
        x = example["years"].to_numpy(dtype=float)
        lo, fit, hi = (example[c].to_numpy(dtype=float) for c in ("boot_lo_ne", "fit_ne", "boot_hi_ne"))
        ax.fill_between(x, lo, hi, step=STEP, **style.band_kw(color))
        ax.plot(x, fit, **style.overlay_kw(color), **step)
        marks += [(x, lo), (x, fit), (x, hi)]
    for edge in ("lo_ne", "hi_ne"):  # the independent fits' range: two lines, never a second fill
        ax.plot(years, sampling[edge], color=style.REF, linewidth=style.LW_THIN, zorder=style.Z_CONTEXT, **step)
    ax.plot(years, true_ne, **style.truth_kw(), **step)

    values = np.concatenate([np.asarray(y, dtype=float) for _, y in marks])
    values = values[np.isfinite(values) & (values > 0)]
    floor, top = float(np.nanmin(true_ne)) / 2.0, float(np.nanmax(true_ne)) * 5.0
    if values.size:  # round limits that hold every curve and band whole
        floor, top = min(floor, float(values.min())), max(top, float(values.max()))
    style.years_axis(ax, xlim, label=False)  # c, below, carries the time axis' label
    style.ne_axis(ax, decades(floor, top))
    return marks


def _key_labels_b(arms: Sequence[int], rows: Dict) -> List[str]:
    """'60 blocks  0.93×' and ' 6 blocks  0.84×': counts right-aligned, ratios in one column."""
    counts = {t: f"{int(rows[t].trunks)}" for t in arms}
    n = max((len(c) for c in counts.values()), default=0)
    labels = []
    for t in arms:
        label = FIGURE_SPACE * (n - len(counts[t])) + f"{counts[t]} blocks"
        ratio = float(rows[t].sd_ratio_median)
        if math.isfinite(ratio):
            label += f"{RATIO_SEP}{ratio:.2f}{style.TIMES}"
        labels.append(label)
    return labels


def _panel_b(ax, sd, arms: Sequence[int], color: Dict[int, str], rows: Dict) -> None:
    import numpy as np
    from matplotlib.lines import Line2D

    sd = sd[sd["sd_true_log10"] > 0]
    pts = {}
    for t in arms:
        s = sd[sd["trunk_bins"] == t][["sd_true_log10", "sd_boot_log10_median"]].dropna()
        # Times within one PSMC interval share their values: one dot per distinct pair.
        pts[t] = s[(s > 0).all(axis=1)].drop_duplicates()
    values = np.concatenate([p.to_numpy(dtype=float).ravel() for p in pts.values()] + [np.array([])])
    lo = min(0.005, 0.8 * float(values.min())) if values.size else 0.005
    hi = max(1.0, 1.25 * float(values.max())) if values.size else 1.0
    xs = np.array([lo / 3.0, hi * 3.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r_lo = float(np.nanmedian(sd["sd_true_ci_lo"] / sd["sd_true_log10"])) if len(sd) else math.nan
        r_hi = float(np.nanmedian(sd["sd_true_ci_hi"] / sd["sd_true_log10"])) if len(sd) else math.nan
    if math.isfinite(r_lo) and math.isfinite(r_hi):
        ax.fill_between(xs, xs * r_lo, xs * r_hi, color=style.FILL, linewidth=0.0, zorder=style.Z_BAND)
    ax.plot(xs, xs, color=style.REF, linewidth=style.LW_THIN, zorder=style.Z_REF)
    for t in reversed(arms):  # the fewest blocks first, under the most
        p = pts[t]
        ax.plot(p["sd_true_log10"], p["sd_boot_log10_median"], color=color[t],
                zorder=style.Z_DATA if t == arms[0] else style.Z_CONTEXT, **DOT)
    for which in ("x", "y"):  # plain decimals at the decades: 0.01, 0.1, 1
        style.log_axis(ax, which, fmt="plain", lim=(lo, hi))
    ax.set_xlabel(f"True SD of {style.LOGNE}")
    ax.set_ylabel(f"Bootstrap SD of {style.LOGNE}")
    # b keeps a key, the house's one written exception to its key rule (one line at the figure's top, no numbers).
    # The 6-block light blue (2.1:1 on white) cannot be text, so the block counts cannot be named in place. The key
    # serves b alone (d names its block counts under its strips; a and c draw the most blocks only), so it sits in
    # b's empty upper-left corner. Each entry ends in its block count's median SD ratio, b's result, set at the
    # key's 7 pt as part of the entry: the key is the one place where a colour, its count and its ratio meet.
    if arms:
        style.legend(ax, [Line2D([], [], color=color[t], **DOT) for t in arms], _key_labels_b(arms, rows),
                     loc="upper left", handlelength=0.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)


def _panel_c(ax, cov, arm: Optional[int], color: str, xlim: Tuple[float, float], nominal: float) -> Dict:
    """c: at each time, the share of the bootstrapped genomes whose band (``arm``) holds the fits' mean or the truth.

    Returns the two lines' (x, y), in %, by name ('mean', 'truth'), for
    their names. The fits' mean in ``color``, the blue of the most blocks;
    the truth in its own style, dashed ink on top, as in a.
    """
    lines: Dict = {}
    style.ref_line(ax, y=nominal)
    if arm is not None:
        rows = cov[cov["trunk_bins"] == arm].sort_values("point")
        step = {"drawstyle": f"steps-{STEP}"}
        x = rows["years"].to_numpy(dtype=float)
        for name, col, kw in (("mean", "coverage_fit_mean", style.overlay_kw(color)),
                              ("truth", "coverage", style.truth_kw())):
            y = 100.0 * rows[col].to_numpy(dtype=float)
            ax.plot(x, y, **kw, **step)
            lines[name] = (x, y)
    style.years_axis(ax, xlim)
    _coverage_axis(ax)
    return lines


def _swarm(y_mm):
    """Offsets (mm) of a strip's dots from its centre, given their heights ``y_mm`` (mm): a deterministic swarm.

    From the highest dot down (ties in table order), each dot takes the
    offset nearest the centre, on a :data:`SWARM_STEP_MM` grid (the left one
    first at equal distance), at which it overlaps no dot placed before it:
    their centres at least a dot's diameter (:data:`style.STRIP_MS`) apart.
    A dot that is not drawn (NaN) stays at the centre.
    """
    import numpy as np

    y = np.asarray(y_mm, dtype=float)
    size = style.STRIP_MS / style.PT_PER_MM
    out = np.zeros(len(y))
    placed: List[Tuple[float, float]] = []
    for k in (int(i) for i in np.argsort(-y, kind="stable") if math.isfinite(y[i])):
        h = float(y[k])
        near = [(px, py) for px, py in placed if abs(py - h) < size]
        step = 0
        while True:  # outwards from the centre until a place is free; one always is, past the dots placed
            offsets = (0.0,) if step == 0 else (-step * SWARM_STEP_MM, step * SWARM_STEP_MM)
            free = [c for c in offsets if all(math.hypot(c - px, h - py) >= size - 1e-9 for px, py in near)]
            if free:
                break
            step += 1
        out[k] = free[0]
        placed.append((free[0], h))
    return out


def _panel_d(ax, per, arms: Sequence[int], color: Dict[int, str], rows: Dict, nominal: float) -> List[float]:
    """Draw d's strips; return the centre of each group (fits' mean, truth) in data x."""
    import matplotlib as mpl
    import numpy as np
    from matplotlib.transforms import ScaledTranslation

    groups = (("mean", "coverage_of_fit_mean_mean"), ("truth", "coverage_mean"))
    _coverage_axis(ax)
    style.ref_line(ax, y=nominal)
    drop = ScaledTranslation(0.0, -NOMINAL_DROP_MM * style.MM, ax.figure.dpi_scale_trans)
    style.note(ax, 1.0, nominal, f"Nominal {nominal:.0f}%", ha="right", va="top",
               transform=ax.get_yaxis_transform() + drop)
    if not arms:
        return []
    width = (len(arms) - 1) * ARM_STEP + GROUP_GAP
    pos = {(g, t): gi * width + i * ARM_STEP for gi, (g, _) in enumerate(groups) for i, t in enumerate(arms)}
    xmax = max(pos.values())
    ax.set_xlim(-X_PAD, xmax + X_PAD)
    # Millimetres per data unit, from the axes as placed: d keeps its size (the right column's) through _layout.
    box, fig_mm = ax.get_position(), (ax.figure.get_figwidth() / style.MM, ax.figure.get_figheight() / style.MM)
    x_mm = box.width * fig_mm[0] / (xmax + 2.0 * X_PAD)
    y_mm = box.height * fig_mm[1] / (COVERAGE_LIM[1] - COVERAGE_LIM[0])
    bar = style.STRIP_BAR_HALF * ARM_STEP
    for g, summary_col in groups:
        for t in arms:
            x = pos[(g, t)]
            v = 100.0 * per.xs(t, level="trunk_bins").sort_index()[g].to_numpy(dtype=float)
            ax.plot(x + _swarm(v * y_mm) / x_mm, v, color=color[t], zorder=style.Z_UNDERLAY, **DOT)
            m = 100.0 * float(getattr(rows[t], summary_col))
            ax.plot([x - bar, x + bar], [m, m], color=style.INK, linewidth=style.LW_DATA, solid_capstyle="butt",
                    zorder=style.Z_DATA)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"{int(rows[t].trunks)}" for (_, t) in pos])
    # No tick marks under the strips; the pad puts the counts on the baseline of c's tick labels.
    ax.tick_params(axis="x", length=0.0, pad=mpl.rcParams["xtick.major.size"] + mpl.rcParams["xtick.major.pad"])
    return [float(np.mean([pos[(g, t)] for t in arms])) for g, _ in groups]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _layout(ax_a, ax_b, ax_c, ax_d) -> Tuple[Dict, float]:
    """Place the four axes; return each one's y-label baseline (points left of its spine) and the right column's
    y-label ink (mm left of its spine).

    The y labels of a column hang at one distance from its spines, the larger
    of matplotlib's two, so that they and the column's letters line up. The
    right column ends ``style.RIGHT_MM`` from the right edge; the left one
    starts at ``style.LEFT_MM`` and leaves ``GAP_MM`` of white between its ink
    (its spines or tick labels) and the right column's y labels.
    """
    base: Dict = {}
    ink: List[float] = []
    for column in ((ax_a, ax_c), (ax_b, ax_d)):
        pt = {ax: _ylabel_pt(ax) for ax in column}
        b = max(p[0] for p in pt.values())
        base.update({ax: b for ax in column})
        ink.append(max(b + p[1] - p[0] for p in pt.values()) / style.PT_PER_MM)
    w0 = ax_a.get_position().width * W_MM
    left_w = RIGHT_X_MM - ink[1] - GAP_MM - max(_right_ink_mm(ax, w0) for ax in (ax_a, ax_c)) - style.LEFT_MM
    for ax, x, y, w, h in ((ax_a, style.LEFT_MM, style.TOP_MM, left_w, TOP_ROW_MM),
                           (ax_b, RIGHT_X_MM, style.TOP_MM, RIGHT_W_MM, TOP_ROW_MM),
                           (ax_c, style.LEFT_MM, LOWER_MM, left_w, BOTTOM_ROW_MM),
                           (ax_d, RIGHT_X_MM, LOWER_MM, RIGHT_W_MM, BOTTOM_ROW_MM)):
        style.place_mm(ax, x, y, w, h)
    return base, ink[1]


def _pin_labels(fig, base: Dict) -> float:
    """Hang every axis label at a fixed distance from its spine; return the x labels' drop in points.

    y labels on their baseline ``base[ax]`` points left of the spine; x
    labels on one baseline under the bottom spines. ``style.on_baseline``
    keeps the mathtext labels (a's y label, b's two) there in the PDF and
    SVG, which would set them 0.9 pt (0.32 mm) higher: b's x label off its
    row's baseline, a's and b's y labels off their column's.
    """
    from matplotlib.transforms import ScaledTranslation

    drop = _xlabel_drop_pt()
    for ax, b in base.items():
        ax.yaxis.set_label_coords(0.0, 0.5, transform=ax.transAxes + ScaledTranslation(
            -b / 72.0, 0.0, fig.dpi_scale_trans))
        ax.xaxis.set_label_coords(0.5, 0.0, transform=ax.transAxes + ScaledTranslation(
            0.0, -drop / 72.0, fig.dpi_scale_trans))
        for label in (ax.yaxis.label, ax.xaxis.label):
            label.set_verticalalignment("baseline")
            style.on_baseline(label)
    return drop


def draw(src: Source):
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    cfg = src.profile.e5
    summary = src.table("E5_summary")
    sampling = src.table("E5_sampling")
    bands = src.table("E5_bands")
    sd = src.table("E5_sd_ratio")
    cov = src.table("E5_coverage")

    informative = summary[summary["window"] == "informative"]
    rows = {int(r.trunk_bins): r for r in informative.itertuples()}
    bootstrapped = {int(t) for t in bands["trunk_bins"]}
    # The block sizes E5 bootstrapped, the most blocks first.
    arms: List[int] = sorted((t for t in rows if t in bootstrapped), key=lambda t: (-int(rows[t].trunks), t))
    color = _arm_colors(arms)
    sampling = sampling.sort_values("point").reset_index(drop=True)
    bands = bands[bands["trunk_bins"].isin(arms)]
    per = genome_coverage(bands, sampling)
    _check_coverage(per, cov, rows, arms)
    many = arms[0] if arms else None  # a and c: the most blocks
    blue = color[many] if arms else style.BLUE
    blocks = f", {int(rows[many].trunks)} blocks" if arms else ""

    fig = style.figure(W_MM, H_MM)
    # The left column's width is provisional; _layout sets it once the tick labels are known.
    ax_a = style.axes_mm(fig, style.LEFT_MM, style.TOP_MM, 100.0, TOP_ROW_MM)
    ax_b = style.axes_mm(fig, RIGHT_X_MM, style.TOP_MM, RIGHT_W_MM, TOP_ROW_MM)
    ax_c = style.axes_mm(fig, style.LEFT_MM, LOWER_MM, 100.0, BOTTOM_ROW_MM)
    ax_d = style.axes_mm(fig, RIGHT_X_MM, LOWER_MM, RIGHT_W_MM, BOTTOM_ROW_MM)

    datasets = {int(d) for d in bands["dataset"]}
    ds = cfg.example_dataset if cfg.example_dataset in datasets else (min(datasets) if datasets else None)
    example = None
    if ds is not None and arms:
        example = bands[(bands["dataset"] == ds) & (bands["trunk_bins"] == many)].sort_values("point")
    xlim = _time_lim(sampling["years"].to_numpy(dtype=float))
    marks = _panel_a(ax_a, sampling, example, blue, xlim)
    _panel_b(ax_b, sd, arms, color, rows)
    nominal = 100.0 * (cfg.band[1] - cfg.band[0])
    lines = _panel_c(ax_c, cov, many, blue, xlim, nominal)
    centres = _panel_d(ax_d, per, arms, color, rows, nominal)
    for ax, title in ((ax_a, f"One genome{blocks}"), (ax_b, "Spread at each time"),
                      (ax_c, f"Coverage at each time{blocks}"), (ax_d, "Coverage of each genome")):
        style.panel_title(ax, title)

    axes = (ax_a, ax_b, ax_c, ax_d)
    base, ink = _layout(*axes)
    drop = _pin_labels(fig, base)
    # Names in place, on the axes as laid out.
    _name_a(ax_a, sampling, example, marks, blue)
    if lines:
        _name_c(ax_c, lines, blue, nominal)
    # d's group names on the x labels' baseline, under its block counts.
    tr = blended_transform_factory(ax_d.transData, ax_d.transAxes) + ScaledTranslation(
        0.0, -drop / 72.0, fig.dpi_scale_trans)
    for xc, name in zip(centres, (MEAN_NAME, TRUTH_NAME)):
        ax_d.text(xc, 0.0, name, transform=tr, ha="center", va="baseline", fontsize=style.TEXT_PT, color=style.INK)

    # Letters on their row's title baseline, style.TITLE_RISE_MM above the axes tops: a and c at
    # style.LETTER_X_MM as in every figure; b and d at one x, the leftmost of their ink at the ink of their
    # column's y labels.
    bearing = min(_ink(letter, style.PANEL_PT, "bold").x0 for letter in "bd") / style.PT_PER_MM
    for (left, right), top in (((ax_a, ax_b), style.TOP_MM), ((ax_c, ax_d), LOWER_MM)):
        baseline = top - style.TITLE_RISE_MM
        style.panel_letter(fig, style.LETTER_X_MM, baseline, "abcd"[axes.index(left)])
        style.panel_letter(fig, RIGHT_X_MM - ink - bearing, baseline, "abcd"[axes.index(right)])
    return fig


__all__ = ["draw", "genome_coverage"]
