"""Fig 7 (E7; the article's Fig 3): the 1000 Genomes genomes, five superpopulations and their 26 populations.

Six equal small multiples, 2 rows by 3 columns (183 x 99 mm), that share
one time axis (thousands of years ago) and one Ne axis (thousands), both
log, so that every curve lines up across rows and columns. Only the
panels of the first column (a, d) show the Ne axis, its spine set out to
the left of their label column; one x label serves the figure.

a: each superpopulation's curve, the median of all its genomes' fits (the
fits of their psmcfa, ``fit-rs-psmcfa``), the five on one axis, so that
they can be compared directly. The title gives the number of genomes in
all.
b-f: one panel per superpopulation (``config.E7.superpopulations``, in
that order): the median curve of each of its populations, all in the
superpopulation's colour, over the other four superpopulations' medians
from a in light grey (``style.CONTEXT``). The titles give each
continent's number of genomes.

Weights rank summary, message and context: a's medians 1 pt
(``style.LW_EMPH``), b-f's populations 0.75 pt (``style.LW_DATA``), the
grey context 0.5 pt (``style.LW_THIN``). a has five emphasised summaries,
not the house's one per panel: a is the summary panel, whose five curves
the other five panels refer to, and its own curves are never drawn
beside thinner ones.

Every curve is a PSMC step curve (``steps-post``, vertical risers) over
the cohort's comparison window (``E7_pop_bands``, group ``all``: the
medians over all 2,504 genomes of their 1% and 99% pairwise coalescence
times, 10.6 kya to 4.6 Mya), the same window for every curve, so that all
curves start at one x, where their names stand in one column, and end at
one x; every curve is drawn in its full colour across it. A group's own
window (columns ``window_lo_years`` and ``window_hi_years``) is not
marked: some start later than the cohort's (East Asia at 17 kya, CHB and
CHS at 23 kya, BEB at 29 kya), but only inside the first flat stretch of
every curve, which is PSMC's first time interval (it ends 21-41 kya here),
a single parameter that one genome constrains poorly; the caption says so,
and the populations' spread there is not to be read as demography. (A
tint before each group's own window was tried and dropped: at print size
its seams, at a different x on every curve, read as rendering faults.)

There is no legend: every curve is named directly, in a label column left
of where the curves start, in its own hue (its superpopulation's text
colour), each name tied to its curve's start by its own thin grey leader
(``style.REF``, ``style.LW_HAIR``); the names in a are the key to the five
colours, and each title of b-f names the continent whose colour it uses.
A leader is horizontal, at its curve's height, where its name sits within
:data:`STRAIGHT_MM` of its curve, and otherwise an elbow (short flats at
both ends, a long diagonal), so that leaders of curves that start together
converge there and never cross. Admixed populations (``config.E7.admixed``)
carry an asterisk, hung right of the code so that the codes of a column
stay right-aligned.

Letters and titles: the first column's letters stand at
``style.LETTER_X_MM``; the others at the left edge of their label column,
or at the leftmost ink of the column's codes where a code passes that edge
(b's GWD, by 0.25 mm), the letters of a column aligned by their ink, as
``style.panel_letters`` aligns letters with their panel's ink. Each title
stands :data:`TITLE_GAP_MM` of clear space right of its letter's ink
(the letters' ink ends at different x: f's 0.64 mm before b's), on the
letter's baseline, 1.5 mm above the axes (``style.TITLE_RISE_MM``). This
is Fig 3's deliberate exception to the house rule of titles at the left
spine: b, c, e and f have no spine, and a and d follow the same rule so
that the six titles share one rhythm. The time axes carry unlabelled minor ticks at 2-9 under
their decades, as ``style.log_axis`` does; the Ne axis, labelled at 1, 2
and 5, carries none (between labelled 1-2-5 ticks they are a comb of
clutter), and keeps 2-9 only where widened data label it at decades.

A psmcfa records no callable bases per bin, so these fits take every called
bin as full, which makes theta, and every size and time, a little low: the
VCF route on the default set gives theta 8.4% higher, a shift that leaves
the populations' order and shapes as they are. Curves are scaled with
``config.E7.scaling`` (mu = 1.25e-8 per bp per generation, g = 29 years).

Colours: :data:`style.SUPERPOP_COLORS` (AFR amber, AMR green, EAS violet,
EUR pink, SAS sienna), hues apart from the psmc-rs blue and C psmc orange
of the other figures. ``validate_palette.js
"#eda100,#12966b,#4a3aa7,#e87ba4,#a0522d" --mode light --surface "#ffffff"
--pairs all`` passes: worst all-pairs CVD dE 8.4 (protan, EUR against
AMR), normal vision 19.6 (EUR against AFR). Amber (2.17:1) and pink
(2.69:1) are below 3:1 on white, which the direct labels relieve: they are
set in the darker :data:`style.SUPERPOP_TEXT` colours, at least 4.38:1 on
white.

Everything comes from E7's table ``E7_pop_bands``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import style

ROUTE = "rs-psmcfa"  # every genome's psmcfa fit
SUP_COLORS = style.SUPERPOP_COLORS  # the curves
SUP_TEXT = style.SUPERPOP_TEXT  # their names and codes (at least 4.38:1 on white)
SUP_NAMES = {"AFR": "Africa", "AMR": "Americas", "EAS": "East Asia", "EUR": "Europe", "SAS": "South Asia"}
# a: the smooth Americas ramp underneath, Africa on top
Z_A = {"AMR": 4.0, "EUR": 4.1, "SAS": 4.2, "EAS": 4.3, "AFR": 4.4}

# Chrome, set on every axes (not left to rcParams)
INK, AXIS, CONTEXT, LEADER = style.INK, style.AXIS, style.CONTEXT, style.REF
# a's five superpopulation medians, 1 pt: five emphasised summaries in one panel, Fig 3's exception to the
# house's one, because a is the summary panel whose curves b-f repeat in grey (module docstring).
LW_SUP = style.LW_EMPH
LW_POP = style.LW_DATA  # b-f's population medians, 0.75 pt: the message of b-f
LW_CONTEXT = style.LW_THIN  # b-f's grey superpopulation medians, 0.5 pt: context
LW_LEADER = style.LW_HAIR  # 0.35 pt
LW_SPINE = style.LW_THIN
NAME_PT, CODE_PT, LABEL_LINE = style.TEXT_PT, style.LABEL_PT, 1.1  # LABEL_LINE: a label's height, in font sizes
ADMIXED_MARK = "*"

# Shared limits (production: window 10.6 kya-4.6 Mya, medians 1.9k-52k); widened to 1-2-5 values if data exceed them
XLIM, YLIM = (1e4, 5e6), (1.5e3, 6e4)

# Layout, mm from the figure's top-left corner
W_MM, H_MM = 183.0, 99.0
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
LETTER_X1_MM = style.LETTER_X_MM  # the first column's letters
TITLE_GAP_MM = 1.6  # clear space from a letter's ink to its title's ink
XLABEL_BOTTOM_MM = 1.2  # the x label's bottom above the figure's bottom edge


def label_blocks(ys: Dict[str, float], merge_dex: float) -> List[Tuple[List[str], float]]:
    """Labels for curves at ``ys`` (Ne): ``[(codes, log10 Ne)]``, low to high.

    Curves whose log10 Ne lies within ``merge_dex`` of the next one's form
    one stroke and share one label, its codes top down; the label points at
    their mean.
    """
    groups: List[List[Tuple[str, float]]] = []
    for k in sorted(ys, key=lambda k: ys[k]):
        y = math.log10(ys[k])
        if groups and y - groups[-1][-1][1] < merge_dex:
            groups[-1].append((k, y))
        else:
            groups.append([(k, y)])
    return [([k for k, _ in reversed(g)], sum(y for _, y in g) / len(g)) for g in groups]


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


def _frame(ax, first_col: bool, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    """The shared axes: log time in thousands (every panel), log Ne in thousands (first column only).

    The time axis is labelled at decades, with unlabelled minor ticks at
    2-9; the Ne axis at 1, 2 and 5 (``style.narrow_log_span``), with no
    minor ticks, or at decades with minor ticks at 2-9 where tiny data
    widen it.
    """
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)
    # A labelled tick at the far end of an axis is dropped (only widened, tiny limits end on one): at the
    # right it would cross the figure's edge (c, f) or the next label column.
    x_major = [v for v in _ticks(xlim, (1.0,)) if v < xlim[1] * (1 - 1e-9)]
    ax.xaxis.set_major_locator(FixedLocator(x_major))
    ax.xaxis.set_major_formatter(style.thousands_formatter())
    ax.xaxis.set_minor_locator(FixedLocator(_minor(xlim, x_major)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(LW_SPINE)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(which="major", direction="out", length=2.0, width=LW_SPINE, color=AXIS, pad=1.5,
                   labelsize=style.TICK_PT, labelcolor=INK)
    ax.tick_params(which="minor", direction="out", length=1.2, width=style.LW_HAIR, color=AXIS)
    if first_col:
        left = ax.spines["left"]
        left.set_position(("outward", LABEL_COL1_MM * style.PT_PER_MM))
        left.set_linewidth(LW_SPINE)
        left.set_color(AXIS)
        narrow = style.narrow_log_span(*ylim)
        y_major = [v for v in _ticks(ylim, (1.0, 2.0, 5.0) if narrow else (1.0,))
                   if v < ylim[1] * (1 - 1e-9)]  # as on x: a widened top's '1,000' would push the y label off
        ax.yaxis.set_major_locator(FixedLocator(y_major))
        ax.yaxis.set_major_formatter(style.thousands_formatter())
        ax.yaxis.set_minor_locator(NullLocator() if narrow else FixedLocator(_minor(ylim, y_major)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(style.NE_LABEL, fontsize=style.TEXT_PT, color=INK, labelpad=2.0)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def _xy(d, window: Tuple[float, float]):
    """A group's median curve over the cohort's ``window``, as steps-post coordinates (``style.step_xy``)."""
    return style.step_xy(d.years.values, d.ne_median.values, window)


def _curve(ax, xy, **kw) -> float:
    """Draw one step curve; the Ne where it starts (its label's target)."""
    x, y = xy
    style.step_line(ax, x, y, **kw)
    return float(y[0])


def _font(size_pt: float, bold: bool = False):
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties

    return FontProperties(family=mpl.rcParams["font.family"], size=size_pt, weight="bold" if bold else "normal")


def _ink(text: str, size_pt: float, bold: bool = False) -> Tuple[float, float, float]:
    """``(advance, ink left, ink right)`` of ``text`` set from x = 0, in mm (from the font, not a renderer).

    matplotlib sets a line of text by its advance width: a left-aligned
    text's ink starts ``ink left`` right of its x, a right-aligned one's
    ``advance - ink left`` left of it.
    """
    from matplotlib.textpath import TextPath, TextToPath

    prop = _font(size_pt, bold)
    advance, _, _ = TextToPath().get_text_width_height_descent(text, prop, ismath=False)
    ext = TextPath((0.0, 0.0), text, prop=prop).get_extents()
    return advance / style.PT_PER_MM, ext.x0 / style.PT_PER_MM, ext.x1 / style.PT_PER_MM


def _labels(ax, items: Mapping[str, float], size_pt: float, color: Union[str, Callable[[str], str]],
            window_lo: float, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> Optional[float]:
    """Each of ``items`` ({text: Ne where its curve starts}) left of the data, with its own leader.

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

    items = {k: v for k, v in items.items() if v > 0 and math.isfinite(v)}
    if not items:
        return None
    colour = color if callable(color) else (lambda _text: color)
    dex_mm = math.log10(ylim[1] / ylim[0]) / ROW_H_MM
    blocks = label_blocks(items, 0.0)  # every label its own block: no merging
    line = size_pt * LABEL_LINE / style.PT_PER_MM * dex_mm
    centres = place_labels([y for _, y in blocks], [line] * len(blocks), math.log10(ylim[0]), math.log10(ylim[1]))
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    xc = math.log10(window_lo / xlim[0]) / math.log10(xlim[1] / xlim[0]) * DW_MM  # where the curves start, mm
    x3 = xc - LEADER_END_MM
    x0 = xc - LEADER_MM
    x1, x2 = x0 + LEADER_FLAT_MM, x3 - LEADER_FLAT_MM
    marked = any(codes[0].endswith(ADMIXED_MARK) for codes, _ in blocks)
    x_text = x0 - TEXT_GAP_MM - (_ink(ADMIXED_MARK, size_pt)[0] if marked else 0.0)
    text_kw = {"transform": tr, "va": "center", "fontsize": size_pt, "clip_on": False, "zorder": style.Z_TEXT}
    leader = {"transform": tr, "color": LEADER, "linewidth": LW_LEADER, "clip_on": False, "zorder": style.Z_TEXT,
              "solid_joinstyle": "miter", "solid_capstyle": "butt"}
    left = math.inf
    for (codes, y), c in zip(blocks, centres):
        text = codes[0]
        code = text[:-len(ADMIXED_MARK)] if text.endswith(ADMIXED_MARK) else text
        ax.text(x_text / DW_MM, 10**c, code, ha="right", color=colour(text), **text_kw)
        if code != text:
            ax.text(x_text / DW_MM, 10**c, ADMIXED_MARK, ha="left", color=colour(text), **text_kw)
        advance, ink_left, _ = _ink(code, size_pt)
        left = min(left, x_text - advance + ink_left)
        if abs(c - y) < STRAIGHT_MM * dex_mm:  # the label sits by its curve: one horizontal leader
            xs, ys = [x0, x3], [10**y, 10**y]
        else:
            xs, ys = [x0, x1, x2, x3], [10**c, 10**c, 10**y, 10**y]
        ax.plot([v / DW_MM for v in xs], ys, **leader)
    return left


def draw(src):
    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    pops = _group(bands, "population", ROUTE)
    sups = _group(bands, "superpopulation", ROUTE)
    every = _group(bands, "all", ROUTE).get("all")
    pop_sup = {g: str(d.superpopulation.iloc[0]) for g, d in pops.items()}
    admixed = set(cfg.admixed)
    slots = list(cfg.superpopulations)[:5]  # b-f, in config order; a slot without data keeps its frame
    present = [s for s in slots if s in sups]
    if every is not None:
        window = (float(every.window_lo_years.iloc[0]), float(every.window_hi_years.iloc[0]))
    else:
        window = tuple(float(v) for v in cfg.figure_years)

    # Every curve, cut to the cohort's window, before any is drawn: the limits must hold them all.
    xlim = _widen(XLIM, window)
    sup_xy = {s: _xy(sups[s], window) for s in present}
    pop_xy = {g: _xy(d, window) for g, d in pops.items() if pop_sup[g] in present}
    ylim = _widen(YLIM, [float(v) for _, y in list(sup_xy.values()) + list(pop_xy.values()) for v in y])

    fig = style.figure("double", H_MM)
    axes = [style.axes_mm(fig, COL_X_MM[c], ROW_Y_MM[r], DW_MM, ROW_H_MM) for r, c in CELLS]
    label_left: Dict[int, float] = {}  # column: leftmost label ink, mm from the figure's left edge

    def note_left(col: int, left: Optional[float]) -> None:
        if left is not None:
            label_left[col] = min(label_left.get(col, math.inf), COL_X_MM[col] + left)

    # a: the superpopulations on one axis, named in their own hues.
    ax_a = axes[0]
    starts = {}
    for s in present:
        starts[SUP_NAMES.get(s, s)] = _curve(ax_a, sup_xy[s], color=SUP_COLORS.get(s, style.INK2),
                                             linewidth=LW_SUP, zorder=Z_A.get(s, style.Z_DATA))
    _frame(ax_a, True, xlim, ylim)
    name_sup = {SUP_NAMES.get(s, s): s for s in present}
    note_left(0, _labels(ax_a, starts, NAME_PT, lambda name: SUP_TEXT.get(name_sup[name], INK), window[0],
                         xlim, ylim))
    titles = [f"All {_genomes(int(every.n.iloc[0]))}" if every is not None else "All genomes"]

    # b-f: each superpopulation's populations over the other four superpopulations' medians.
    for i in range(5):
        ax, (_, col) = axes[i + 1], CELLS[i + 1]
        s = slots[i] if i < len(slots) else None
        name = SUP_NAMES.get(s, s) if s is not None else ""
        if s in sups:
            for o in present:
                if o != s:
                    _curve(ax, sup_xy[o], color=CONTEXT, linewidth=LW_CONTEXT, zorder=style.Z_CONTEXT)
            hue = SUP_COLORS.get(s, style.INK2)
            starts = {}
            for g in sorted(g for g in pop_xy if pop_sup[g] == s):
                code = g + (ADMIXED_MARK if g in admixed else "")
                starts[code] = _curve(ax, pop_xy[g], color=hue, linewidth=LW_POP, zorder=style.Z_DATA)
            note_left(col, _labels(ax, starts, CODE_PT, SUP_TEXT.get(s, INK), window[0], xlim, ylim))
            name = f"{name}, {_genomes(int(sups[s].n.iloc[0]))}"
        _frame(ax, col == 0, xlim, ylim)
        titles.append(name)

    # Letters: the first column's at LETTER_X1_MM; the others' ink at their label column's left edge, or at the
    # column's leftmost label ink where a code passes it, aligned by ink down the column. Titles a constant clear
    # space right of their letter's ink, on its baseline. One x label.
    for k, ((row, col), title) in enumerate(zip(CELLS, titles)):
        letter = "abcdef"[k]
        _, l_left, l_right = _ink(letter, style.PANEL_PT, bold=True)
        if col == 0:
            x = LETTER_X1_MM
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


__all__ = ["ROUTE", "draw", "label_blocks", "place_labels"]
