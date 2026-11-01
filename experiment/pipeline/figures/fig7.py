"""The article's Fig 3 (E7): population size histories of the 1000 Genomes individuals, each genome fitted by psmc-rs.

Six equal small multiples, 2 rows by 3 columns (183 x 99 mm), sharing one
log time axis (thousands of years ago) and one log Ne axis (thousands), so
that every curve lines up across rows and columns. Only the first column
(a, d) shows the Ne axis, its spine set out left of its label column; one
x label serves the figure.

a: each superpopulation's curve, the median of all its genomes' fits (the
fits of their psmcfa, ``fit-rs-psmcfa``), the five on one axis; the title
gives the number of genomes in all.
b-f: one panel per superpopulation (``config.E7.superpopulations``, in
that order): each of its genomes' fits as a faint hairline (the halo), the
median curve of each of its populations in the superpopulation's colour,
and the other four superpopulations' medians from a in light grey
(``style.CONTEXT``); each title gives the continent's number of genomes.
A median is pointwise: at each time of ``E7_pop_bands``' grid, the median
of its genomes' Ne.

Weights rank summary, message and context: a's medians 1 pt
(``style.LW_EMPH``), b-f's populations 0.75 pt (``style.LW_DATA``), the
grey context 0.5 pt (``style.LW_THIN``), each genome 0.35 pt
(``style.LW_HAIR``). a has five emphasised summaries, not the house's one
per panel: it is the summary panel whose curves b-f repeat in grey, and
its curves are never drawn beside thinner ones.

PSMC's first window is the first free lambda of the pattern fitted (the
first term of ``E7_fits.pattern``; in 4+25*2+4+6, one parameter for the
four most recent intervals): a single parameter, which one genome
constrains poorly. A genome's first window ends where the next interval
of its curve starts (19-41 kya here). Each median is drawn in a 50% tint
of its colour (``style.FIRST_WINDOW_SHARE``), at its own weight and just
beneath, until its group has left that window: up to, and including its
step at, the first time of the grid after :data:`SPLIT_QUANTILE` (90%) of
its genomes' first windows have ended (25-41 kya). Its step out of the
window, outside Africa its largest (a drop), is pale too. Each genome's
hairline starts where its own first window ends, and each grey median at
the time it turns full in a.

The halo (``style.cloud_kw``) is the one alpha of the article's figures:
each genome a hairline in a 45% tint of its superpopulation's colour at
alpha 0.07. It darkens where many genomes run together and fades where few
stray: where a continent's genomes agree, its hairlines merge into a soft
band around the medians; where they part, they fan out. It lies above the
grey context and beneath the medians, and is rasterized at 600 dpi in the
PDF and SVG; a has none. Its genomes are those the medians and titles
count (each group's number is checked against ``E7_pop_bands``). It has no
key: the caption names it.

Every curve is a PSMC step curve over the cohort's comparison window
(``E7_pop_bands``, group ``all``: the medians over all 2,504 genomes of
their 1% and 99% pairwise coalescence times, 10.6 kya to 4.6 Mya), the
same window for every curve, so that all medians start at one x, where
their names stand in one column, and end at one x. A group's own window is
not marked: some start later than the cohort's (BEB's, the latest, at 29
kya), each inside its group's pale first window.

There is no legend: every median is named directly, in a label column left
of where the curves start, in its superpopulation's text colour, each name
tied to its curve's start by its own thin grey leader
(``style.LABEL_LEADER_KW``); the names in a are the key to the five
colours, and each title of b-f names the continent whose colour it uses.
A leader is horizontal where its name sits within :data:`STRAIGHT_MM` of
its curve, and otherwise an elbow (short flats at both ends, a long
diagonal), so that leaders of curves that start together converge there
and never cross. Admixed populations (``config.E7.admixed``) carry an
asterisk, hung right of the code so that the codes of a column stay
right-aligned.

Letters and titles: the first column's letters stand at
``style.LETTER_X_MM``; the others at the left edge of their label column,
or at the leftmost ink of the column's codes where a code passes that edge
(b's GWD, by 0.1 mm), aligned by their ink down a column. Each title
stands :data:`TITLE_GAP_MM` of clear space right of its letter's ink, on
the letter's baseline, ``style.TITLE_RISE_MM`` above the axes: Fig 3's
deliberate exception to the house rule of titles at the left spine, since
b, c, e and f have no spine. The time axes carry unlabelled minor ticks at
2-9 under their decades; the Ne axis, labelled at 1, 2 and 5, carries none
(between labelled 1-2-5 ticks they are a comb of clutter), and keeps 2-9
only where widened data label it at decades.

A psmcfa records no callable bases per bin, so these fits take every called
bin as full, which makes theta, and every size and time, a little low: the
VCF fits of the 52 default genomes, each bin weighted by its callable
bases, give theta 8.5% higher (the median, ``E7_route_shift``), a shift
that leaves the populations' order and shapes as they are. Curves are
scaled with ``config.E7.scaling`` (mu = 1.25e-8 per bp per generation,
g = 29 years).

Colours: :data:`style.SUPERPOP_COLORS` (AFR amber, AMR green, EAS violet,
EUR pink, SAS sienna), hues apart from the programs' blue and orange.
The five pass an external palette validator's all-pairs check on white
(not in this repository): worst CVD dE 8.4 (protan, EUR against AMR),
normal vision 19.6 (EUR against AFR). Amber (2.17:1) and pink (2.69:1)
are below 3:1 on white, which the direct labels relieve: they are set in
the darker :data:`style.SUPERPOP_TEXT` colours, at least 4.38:1 on white.
The grey context (``style.CONTEXT``, #d9d9d9) parts from all five: in
greyscale by at least Delta L* 14.8 (AFR amber), under colour-vision
deficiency by at least Delta E 16.3 (deuteranopia, EUR pink).

Everything comes from E7's tables: ``E7_pop_bands`` (the medians, the
window and the counts), ``E7_curves`` (each genome's fit) and ``E7_fits``
(each genome's population, superpopulation and pattern).
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from ..results import pattern_groups
from . import style

ROUTE = "rs-psmcfa"  # every genome's psmcfa fit
SUP_NAMES = {"AFR": "Africa", "AMR": "Americas", "EAS": "East Asia", "EUR": "Europe", "SAS": "South Asia"}
Z_A = {"AMR": 4.0, "EUR": 4.1, "SAS": 4.2, "EAS": 4.3, "AFR": 4.4}  # a: the smooth Americas underneath, Africa on top
LABEL_LINE = 1.1  # a label's height, in font sizes
ADMIXED_MARK = "*"
# A median leaves PSMC's first window at the first time of its grid after this share of its genomes' first windows
# have ended: by then nearly all its genomes are past theirs, and its drop out of the window lies before.
SPLIT_QUANTILE = 0.9

# Shared limits (production: window 10.6 kya-4.6 Mya, medians 1.9k-52k); widened to 1-2-5 values if data exceed them
XLIM, YLIM = (1e4, 5e6), (1.5e3, 6e4)

# Layout, mm from the figure's top-left corner
W_MM, H_MM = style.WIDTH_MM["double"], 99.0
TOP_MM, ROW_H_MM, ROW_GAP_MM = 5.0, 37.0, 11.5
RULER_MM = 8.5  # the first column's y spine, set out left of its label column
LABEL_COL1_MM, LABEL_COL_MM = 16.0, 9.0  # label columns left of the data: first column, the others
PANEL_GAP_MM, RIGHT_MM = 3.0, 1.5  # a label column's clearance after the previous panel's data; right margin
DW_MM = (W_MM - RULER_MM - LABEL_COL1_MM - 2 * LABEL_COL_MM - 2 * PANEL_GAP_MM - RIGHT_MM) / 3.0  # 44.33
COL_X_MM = (RULER_MM + LABEL_COL1_MM,
            RULER_MM + LABEL_COL1_MM + DW_MM + PANEL_GAP_MM + LABEL_COL_MM,
            RULER_MM + LABEL_COL1_MM + 2 * (DW_MM + PANEL_GAP_MM + LABEL_COL_MM))  # 24.5, 80.83, 137.17
ROW_Y_MM = (TOP_MM, TOP_MM + ROW_H_MM + ROW_GAP_MM)  # 5.0, 53.5
CELLS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2))  # a b c / d e f
# Leaders: from LEADER_MM left of the curves' start to LEADER_END_MM left of it; flats of LEADER_FLAT_MM at both
# ends of an elbow, the diagonal between them (2.05 mm of run); a label within STRAIGHT_MM of its curve's height
# gets one horizontal leader at that height.
LEADER_MM, LEADER_FLAT_MM, LEADER_END_MM, TEXT_GAP_MM, STRAIGHT_MM = 3.1, 0.4, 0.25, 0.5, 0.1
TITLE_GAP_MM = 1.6  # clear space from a letter's ink to its title's ink
XLABEL_BOTTOM_MM = 1.2  # the x label's bottom above the figure's bottom edge


class GenomeFit(NamedTuple):
    """One genome's psmcfa fit: its groups, its step curve and the length of its first window."""

    population: str
    superpopulation: str
    years: Sequence[float]  # each interval's start, years ago
    ne: Sequence[float]  # each interval's Ne
    first: int  # intervals in PSMC's first window: the width of the pattern's first free lambda

    @property
    def first_end(self) -> float:
        """Where the first window ends: the start of the interval after it, in years."""
        return float(self.years[self.first])


def _isotonic(z: Sequence[float]) -> List[float]:
    """The non-decreasing sequence closest to ``z`` in least squares (pool adjacent violators)."""
    blocks: List[List[float]] = []
    for v in z:
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out: List[float] = []
    for s, n in blocks:
        out += [s / n] * int(n)
    return out


def place_labels(targets: Sequence[float], heights: Sequence[float], lo: float, hi: float) -> List[float]:
    """Centres of labels stacked in the order of ``targets`` (ascending), as close to them as possible.

    Neighbouring labels do not overlap (centres at least half their heights
    apart), the stack lies within ``[lo, hi]``, and the centres are the
    least-squares closest to the targets under those constraints.
    """
    if not targets:
        return []
    offsets = [0.0]
    for a, b in zip(heights[:-1], heights[1:]):
        offsets.append(offsets[-1] + (a + b) / 2.0)
    z = _isotonic([t - c for t, c in zip(targets, offsets)])
    z_lo, z_hi = lo + heights[0] / 2.0, hi - heights[-1] / 2.0 - offsets[-1]
    return [min(max(v, z_lo), max(z_lo, z_hi)) + c for v, c in zip(z, offsets)]


def _group(bands, kind: str, route: str) -> Dict[str, object]:
    """``{group: rows sorted by years}`` of one kind and route of ``E7_pop_bands``."""
    sub = bands[(bands.group_kind == kind) & (bands.route == route)]
    return {g: d.sort_values("years") for g, d in sub.groupby("group", sort=True) if len(d)}


def genome_fits(fits, curves) -> Dict[str, GenomeFit]:
    """``{sample: its fit}`` of every genome's psmcfa fit (:data:`ROUTE`), from ``E7_fits`` and ``E7_curves``."""
    who = fits[fits.route == ROUTE].set_index("sample_id")
    mine = curves[curves.route == ROUTE].sort_values(["sample_id", "interval"])
    out: Dict[str, GenomeFit] = {}
    for sample, d in mine.groupby("sample_id", sort=True):
        f = who.loc[sample]
        out[str(sample)] = GenomeFit(str(f.population), str(f.superpopulation), d["years"].to_numpy(float),
                                     d["ne"].to_numpy(float), pattern_groups(str(f.pattern))[0][1])
    return out


def first_window_splits(fits: Mapping[str, GenomeFit], grid: Sequence[float]) -> Dict[Tuple[str, str], float]:
    """``{(kind, group): time}`` where each median leaves PSMC's first window (kind ``population`` or
    ``superpopulation``): the first time of ``grid`` after :data:`SPLIT_QUANTILE` of the group's genomes' first
    windows have ended (that quantile itself where no time of the grid follows it)."""
    import numpy as np

    ends: Dict[Tuple[str, str], List[float]] = {}
    for g in fits.values():
        ends.setdefault(("population", g.population), []).append(g.first_end)
        ends.setdefault(("superpopulation", g.superpopulation), []).append(g.first_end)
    grid = np.sort(np.asarray(grid, dtype=float))
    out: Dict[Tuple[str, str], float] = {}
    for key, v in ends.items():
        q = float(np.quantile(v, SPLIT_QUANTILE))
        i = int(np.searchsorted(grid, q, side="right"))
        out[key] = float(grid[i]) if i < len(grid) else q
    return out


def _split(xy, at: float):
    """Steps-post coordinates ``xy`` cut at ``at``: ``(before, after)``, either None where empty.

    A step that starts at ``at`` goes after it, the drop into it before.
    """
    import numpy as np

    x, y = (np.asarray(v, dtype=float) for v in xy)
    if at <= x[0]:
        return None, (x, y)
    if at >= x[-1]:
        return (x, y), None
    k = int(np.searchsorted(x, at, side="right")) - 1  # the step that holds ``at``
    if x[k] == at:
        return (x[:k + 1], y[:k + 1]), (x[k:], y[k:])
    before = (np.append(x[:k + 1], at), np.append(y[:k + 1], y[k]))
    return before, (np.append(at, x[k + 1:]), np.append(y[k], y[k + 1:]))


def _halo(fits: Mapping[str, GenomeFit], window: Tuple[float, float]) -> Dict[str, list]:
    """``{superpopulation: vertices}``: each genome's step curve from the end of its first window to the window's
    end, as the vertices of the steps (one array per genome, for a ``LineCollection``)."""
    import numpy as np

    out: Dict[str, list] = {}
    for g in fits.values():
        start = max(window[0], g.first_end)
        if start >= window[1]:
            continue
        x, y = style.step_xy(g.years[g.first:], g.ne[g.first:], (start, window[1]))
        out.setdefault(g.superpopulation, []).append(np.column_stack([np.repeat(x, 2)[1:], np.repeat(y, 2)[:-1]]))
    return out


def _genomes(n: int) -> str:
    return f"{n:,} genome" + ("" if n == 1 else "s")


def _widen(lim: Tuple[float, float], values: Sequence[float]) -> Tuple[float, float]:
    """``lim``, each end widened to the enclosing 1-2-5 value where ``values`` pass it (tiny data only)."""
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    if not vals:
        return lim
    lo, hi = lim
    if min(vals) < lo:
        e = math.floor(math.log10(min(vals)))
        lo = max(m * 10.0**e for m in (1.0, 2.0, 5.0) if m * 10.0**e <= min(vals) * (1 + 1e-9))
    if max(vals) > hi:
        e = math.floor(math.log10(max(vals)))
        hi = min(m * 10.0**e for m in (1.0, 2.0, 5.0, 10.0) if m * 10.0**e >= max(vals) * (1 - 1e-9))
    return lo, hi


def _ticks(lim: Tuple[float, float], subs: Sequence[float]) -> List[float]:
    """``m x 10^e`` for ``m`` in ``subs`` within ``lim`` (ends included)."""
    lo, hi = lim
    out = [m * 10.0**e for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1) for m in subs]
    return sorted(v for v in out if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9))


def _minor(lim: Tuple[float, float], major: Sequence[float]) -> List[float]:
    """Unlabelled minor ticks at 2-9 x 10^e strictly within ``lim``, where no major tick is."""
    lo, hi = lim
    taken = {round(math.log10(v), 9) for v in major}
    return [v for v in _ticks(lim, tuple(float(m) for m in range(2, 10)))
            if lo * (1 + 1e-9) < v < hi * (1 - 1e-9) and round(math.log10(v), 9) not in taken]


def _thousands_ticks(axis, lim: Tuple[float, float], subs: Sequence[float], minor: bool) -> None:
    """Ticks at ``m x 10^e`` (``m`` in ``subs``) labelled in thousands, and optionally unlabelled minor ones.

    A labelled tick at the top of the axis is dropped (only widened, tiny
    limits end on one): at the right it would cross the figure's edge (c, f)
    or the next label column, at the top push the y label off.
    """
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    major = [v for v in _ticks(lim, subs) if v < lim[1] * (1 - 1e-9)]
    axis.set_major_locator(FixedLocator(major))
    axis.set_major_formatter(style.thousands_formatter())
    axis.set_minor_locator(FixedLocator(_minor(lim, major)) if minor else NullLocator())
    axis.set_minor_formatter(NullFormatter())


def _frame(ax, first_col: bool, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """The shared axes: log time in thousands (every panel), log Ne in thousands (first column only).

    The time axis is labelled at decades, with unlabelled minor ticks at
    2-9; the Ne axis at 1, 2 and 5 (``style.narrow_log_span``), with no
    minor ticks, or at decades with minor ticks at 2-9 where tiny data
    widen it.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _thousands_ticks(ax.xaxis, xlim, (1.0,), minor=True)
    if first_col:
        ax.spines["left"].set_position(("outward", LABEL_COL1_MM * style.PT_PER_MM))
        narrow = style.narrow_log_span(*ylim)
        _thousands_ticks(ax.yaxis, ylim, (1.0, 2.0, 5.0) if narrow else (1.0,), minor=not narrow)
        ax.set_ylabel(style.NE_LABEL)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def _xy(d, window: Tuple[float, float]):
    """A group's median curve over the cohort's ``window``, as steps-post coordinates (``style.step_xy``)."""
    return style.step_xy(d.years.values, d.ne_median.values, window)


def _median(ax, xy, split: float, color: str, linewidth: float, zorder: float, dex_mm: float) -> float:
    """Draw one median: its first window, up to ``split``, pale and just beneath; the rest in full ``color``.

    The step at ``split`` is pale too: the colour changes straight across
    the line where that step's ink ends, half the line's width later
    (``dex_mm``: log10 time per mm). The full part is clipped there, over
    the pale part, which runs on beneath it for a line's width; so the line
    keeps its corners, and neither colour pokes into the other, whatever
    the size of the steps around the split. Returns the Ne where the median
    starts (its label's target).
    """
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    pale, full = _split(xy, split)
    half = 10 ** (linewidth / style.PT_PER_MM / 2.0 * dex_mm)  # half the line's width, as a factor of time
    if pale is not None:
        under = pale if full is None else _split(xy, split * half * half)[0]
        style.step_line(ax, *under, color=style.tint(color, style.FIRST_WINDOW_SHARE), linewidth=linewidth,
                        zorder=zorder - 0.5)
    if full is not None:
        line, = style.step_line(ax, *full, color=color, linewidth=linewidth, zorder=zorder)
        if pale is not None:  # not passed to ax.plot, where the axes' own clip box would replace it
            cut = split * half
            line.set_clip_path(Rectangle((cut, 0.0), full[0][-1] * half - cut, 1.0,
                                         transform=blended_transform_factory(ax.transData, ax.transAxes)))
    return float(xy[1][0])


def _ink(text: str, size_pt: float, bold: bool = False) -> Tuple[float, float, float]:
    """``(advance, ink left, ink right)`` of ``text`` set from x = 0, in mm (from the font, not a renderer).

    matplotlib sets a line of text by its advance width: a left-aligned
    text's ink starts ``ink left`` right of its x, a right-aligned one's
    ``advance - ink left`` left of it.
    """
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath, TextToPath

    prop = FontProperties(family=mpl.rcParams["font.family"], size=size_pt, weight="bold" if bold else "normal")
    advance, _, _ = TextToPath().get_text_width_height_descent(text, prop, ismath=False)
    ext = TextPath((0.0, 0.0), text, prop=prop).get_extents()
    return advance / style.PT_PER_MM, ext.x0 / style.PT_PER_MM, ext.x1 / style.PT_PER_MM


def _labels(ax, starts: Mapping[str, float], size_pt: float, colors: Mapping[str, str], window_lo: float,
            xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Optional[float]:
    """Name each curve left of the data, with its own leader; ``starts``: each name's Ne where its curve starts.

    The labels are stacked without overlap, as close to their curves'
    starts as they can be. A label within ``STRAIGHT_MM`` of its curve gets
    one horizontal leader at its curve's height; any other runs flat from
    its label, bends, crosses on a long diagonal and runs flat into its
    curve's start, so leaders of curves that start together converge there.
    All elbows share their bends, which keeps leaders from crossing. An
    admixed mark hangs right of its code, in the gap before the leaders,
    and every code of a panel with such a mark moves left by its width, so
    the codes stay right-aligned. Returns the leftmost ink of the labels, in
    mm right of the axes' left edge (negative: left of it), or None.
    """
    from matplotlib.transforms import blended_transform_factory

    starts = {k: v for k, v in starts.items() if v > 0 and math.isfinite(v)}
    if not starts:
        return None
    names = sorted(starts, key=starts.get)  # low to high
    targets = [math.log10(starts[k]) for k in names]
    dex_mm = math.log10(ylim[1] / ylim[0]) / ROW_H_MM
    line = size_pt * LABEL_LINE / style.PT_PER_MM * dex_mm
    centres = place_labels(targets, [line] * len(names), math.log10(ylim[0]), math.log10(ylim[1]))
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    xc = math.log10(window_lo / xlim[0]) / math.log10(xlim[1] / xlim[0]) * DW_MM  # where the curves start, mm
    x3 = xc - LEADER_END_MM
    x0 = xc - LEADER_MM
    x1, x2 = x0 + LEADER_FLAT_MM, x3 - LEADER_FLAT_MM
    marked = any(k.endswith(ADMIXED_MARK) for k in names)
    x_text = x0 - TEXT_GAP_MM - (_ink(ADMIXED_MARK, size_pt)[0] if marked else 0.0)
    text_kw = {"transform": tr, "va": "center", "fontsize": size_pt, "clip_on": False, "zorder": style.Z_TEXT}
    leader = {**style.LABEL_LEADER_KW, "transform": tr, "clip_on": False, "zorder": style.Z_TEXT,
              "solid_joinstyle": "miter", "solid_capstyle": "butt"}
    left = math.inf
    for text, y, c in zip(names, targets, centres):
        code = text[:-len(ADMIXED_MARK)] if text.endswith(ADMIXED_MARK) else text
        ax.text(x_text / DW_MM, 10**c, code, ha="right", color=colors[text], **text_kw)
        if code != text:
            ax.text(x_text / DW_MM, 10**c, ADMIXED_MARK, ha="left", color=colors[text], **text_kw)
        advance, ink_left, _ = _ink(code, size_pt)
        left = min(left, x_text - advance + ink_left)
        if abs(c - y) < STRAIGHT_MM * dex_mm:  # the label sits by its curve: one horizontal leader
            xs, ys = [x0, x3], [10**y, 10**y]
        else:
            xs, ys = [x0, x1, x2, x3], [10**c, 10**c, 10**y, 10**y]
        ax.plot([v / DW_MM for v in xs], ys, **leader)
    return left


def draw(src):
    from matplotlib.collections import LineCollection

    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    pops = _group(bands, "population", ROUTE)
    sups = _group(bands, "superpopulation", ROUTE)
    every = _group(bands, "all", ROUTE).get("all")
    pop_sup = {g: str(d.superpopulation.iloc[0]) for g, d in pops.items()}
    slots = list(cfg.superpopulations)[:5]  # b-f, in config order; a slot without data keeps its frame
    present = [s for s in slots if s in sups]
    if every is not None:
        window = (float(every.window_lo_years.iloc[0]), float(every.window_hi_years.iloc[0]))
    else:
        window = tuple(float(v) for v in cfg.figure_years)

    # Each genome's fit: where its first window ends, for its hairline and its groups' medians. The halo must hold
    # the genomes the medians and the titles count.
    fits = genome_fits(src.table("E7_fits"), src.table("E7_curves"))
    for kind, groups in (("population", pops), ("superpopulation", sups)):
        for name, d in groups.items():
            n, want = sum(getattr(g, kind) == name for g in fits.values()), int(d.n.iloc[0])
            if n != want:
                raise ValueError(f"E7_curves holds {n} psmcfa fits of {name}, E7_pop_bands {want}: the tables are "
                                 "of different runs")
    split = first_window_splits(fits, bands[bands.route == ROUTE].years.unique())
    halo = _halo(fits, window)

    # Every curve, cut to the cohort's window, before any is drawn: the limits must hold them all.
    xlim = _widen(XLIM, window)
    dex_mm = math.log10(xlim[1] / xlim[0]) / DW_MM  # log10 time per mm
    sup_xy = {s: _xy(sups[s], window) for s in present}
    pop_xy = {g: _xy(d, window) for g, d in pops.items() if pop_sup[g] in present}
    ylim = _widen(YLIM, [float(v) for _, y in list(sup_xy.values()) + list(pop_xy.values()) for v in y])

    fig = style.figure(W_MM, H_MM)
    axes = [style.axes_mm(fig, COL_X_MM[c], ROW_Y_MM[r], DW_MM, ROW_H_MM) for r, c in CELLS]
    label_left: Dict[int, float] = {}  # column: leftmost label ink, mm from the figure's left edge

    def name_curves(ax, col: int, starts, size_pt: float, colors) -> None:
        left = _labels(ax, starts, size_pt, colors, window[0], xlim, ylim)
        if left is not None:
            label_left[col] = min(label_left.get(col, math.inf), COL_X_MM[col] + left)

    # a: the superpopulations on one axis, named in their own hues.
    starts, colors = {}, {}
    for s in present:
        name = SUP_NAMES.get(s, s)
        starts[name] = _median(axes[0], sup_xy[s], split[("superpopulation", s)],
                               style.SUPERPOP_COLORS.get(s, style.INK2), style.LW_EMPH, Z_A.get(s, style.Z_DATA),
                               dex_mm)
        colors[name] = style.SUPERPOP_TEXT.get(s, style.INK)
    _frame(axes[0], True, xlim, ylim)
    name_curves(axes[0], 0, starts, style.TEXT_PT, colors)
    titles = [f"All {_genomes(int(every.n.iloc[0]))}" if every is not None else "All genomes"]

    # b-f: each superpopulation's genomes and populations, over the other four superpopulations' medians from
    # where they turn full in a.
    for i, (ax, (_, col)) in enumerate(zip(axes[1:], CELLS[1:])):
        s = slots[i] if i < len(slots) else None
        name = SUP_NAMES.get(s, s) if s is not None else ""
        if s in sups:
            color = style.SUPERPOP_COLORS.get(s, style.INK2)
            if halo.get(s):
                ax.add_collection(LineCollection(halo[s], **style.cloud_kw(color)))
            for o in present:
                if o != s:
                    _, full = _split(sup_xy[o], split[("superpopulation", o)])
                    if full is not None:
                        style.step_line(ax, *full, color=style.CONTEXT, linewidth=style.LW_THIN,
                                        zorder=style.Z_CONTEXT)
            starts = {}
            for g in sorted(g for g in pop_xy if pop_sup[g] == s):
                code = g + (ADMIXED_MARK if g in cfg.admixed else "")
                starts[code] = _median(ax, pop_xy[g], split[("population", g)], color, style.LW_DATA, style.Z_DATA,
                                       dex_mm)
            name_curves(ax, col, starts, style.LABEL_PT, dict.fromkeys(starts, style.SUPERPOP_TEXT.get(s, style.INK)))
            name = f"{name}, {_genomes(int(sups[s].n.iloc[0]))}"
        _frame(ax, col == 0, xlim, ylim)
        titles.append(name)

    # Letters: the first column's at LETTER_X_MM; the others' ink at their label column's left edge, or at the
    # column's leftmost label ink where a code passes it. Titles a constant clear space right of their letter's
    # ink, on its baseline. One x label.
    for letter, (row, col), title in zip("abcdef", CELLS, titles):
        _, l_left, l_right = _ink(letter, style.PANEL_PT, bold=True)
        if col == 0:
            x = style.LETTER_X_MM
        else:
            x = min(COL_X_MM[col] - LABEL_COL_MM, label_left.get(col, math.inf)) - l_left
        baseline = ROW_Y_MM[row] - style.TITLE_RISE_MM
        style.panel_letter(fig, x, baseline, letter)
        if title:
            style.fig_text(fig, x + l_right + TITLE_GAP_MM - _ink(title, style.TEXT_PT)[1], baseline, title,
                           ha="left", va="baseline")
    style.fig_text(fig, COL_X_MM[1] + DW_MM / 2.0, H_MM - XLABEL_BOTTOM_MM, style.YEARS_LABEL, ha="center",
                   va="bottom")
    return fig


__all__ = ["ROUTE", "GenomeFit", "draw", "first_window_splits", "genome_fits", "place_labels"]
