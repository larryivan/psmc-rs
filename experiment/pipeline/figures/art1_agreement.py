"""The article's Fig 1: psmc-rs reproduces C psmc, on simulated and on real genomes.

Five panels at 183 x 88 mm. Top row, a-d: four histories fitted by both
programs to the same genome, C psmc's as a wide pale-orange underlay
(:data:`.style.C_HALO`) with psmc-rs's as a thin blue line on top, each
over the genome's informative window (curves and band clipped to it, so all
end flush with it; never resampled, so the two programs' risers stay at their
exact times). Grey (:data:`.style.FILL`, opaque, square steps) is the
spread of estimates. The note at the bottom right of each, Delta, is the
RMSE of log10 Ne between the two programs' fits, interval by interval over
the window (``rmse_log10_ne_rs_vs_c_per_interval``). Times are in thousands of years
ago, one label for the row. Letters and titles sit on one baseline, the
house rise (``style.TITLE_RISE_MM``) above the axes.

* a, b (E1): two simulated models, ``bottleneck`` and ``zigzag`` first
  (:data:`SIM_PREF`), then the profile's other models in order; the first
  replicate both programs fitted (replicate 1 in production). Grey: the
  range (pointwise min to max) of psmc-rs's fits to every replicate of the
  model; dashed ink: the true Ne(t) on top (:func:`.data.truth_xy`).
  Window from ``E1_table1``, Delta from ``E1_rs_vs_c``.
* c (E2): the human genome, the first sample of ``config.E2.panel_order``
  whose species is Homo sapiens and that both programs fitted (NA12878).
  Grey: the pointwise 95% band of psmc-rs's bootstrap at the default block
  size (``E2_bands`` at ``bootstrap_trunk_bins``), drawn as steps. No truth.
* d (E2): the non-human genome on which the programs differ most (the
  largest Delta in ``E2_rs_vs_c``, chosen from the data: Pelusios castaneus
  in production), drawn as c.

a, b and c share both scales (5,000 to 5,000,000 years; Ne 500 to
200,000); d has its own (5,000 to 20,000,000 years; Ne 5,000 to
2,000,000), with the same span of Ne, 2.6 decades, so its Ne scale (mm per
decade) and the heights of its labelled ticks are theirs, the top one
('100', '1,000') 0.3 decade below the axes top; its time axis spans 3.6
decades instead of 3. Limits that would not hold every drawn Ne at least a
factor :data:`PAD` from either end, or every window (the tiny test
profile), widen to round values of the data (:func:`.data.decades`). A
column with nothing to draw is left empty, without letter or title, and the
letters of the drawn panels run on. The four panels are equally wide, and
their gutters are set by the clear white between a panel and the next one's
y tick labels (:data:`CLEAR_MM`), so d's wider labels ('1,000') get a wider
gutter.

* e: a strip plot on one log axis of RMSE of log10 Ne. Upper left, grey,
  psmc-rs against C psmc (Delta, as in a-d) for every genome both programs
  fitted, each counted once: ``Human`` (the 1000 Genomes genomes of
  ``E7_rs_vs_c``, NA12878 among them, plus any human of E2 that E7 lacks),
  the non-human genomes of ``E2_rs_vs_c`` (named by their shared common
  name, pluralised: ``Turtles``; ``Other genomes`` when their common names
  differ) and ``Simulated`` (``E1_rs_vs_c``). Lower right, each program's
  error against the truth on the simulated genomes (``E1_replicates``;
  blue circles psmc-rs, orange squares C psmc, one shared jitter so each
  genome sits at the same height in both rows). Bars (ink, cased in white)
  are medians of rows of at least :data:`MEDIAN_MIN` genomes. A double
  arrow joins the median of ``Simulated`` to that of psmc-rs's error,
  labelled with their ratio to one significant digit (:func:`fold_median`, :func:`round_fold`: the
  rounding of ``\\EoneFoldMedian``), '7,000x smaller' in production; dotted
  leaders join its ends to the two bars. Laid out as a forest plot: each
  block of rows has a header (7 pt: ``psmc-rs against C psmc (Delta)``,
  ``Error against the truth (simulated)``) and each row its name with its n
  (6 pt, every n from the tables), in one label column at the left of the
  figure. The headers start where a-d's titles do (a's spine), the
  letter 'e' at :data:`.style.LETTER_X_MM` on the first header's baseline;
  the label column starts at the left edge of a's ink (its y label);
  the row names are right-aligned :data:`E_LABEL_GAP_MM` left of e's axes,
  as tick labels, the column as wide as the widest name; the axes run from
  there to d's right edge.

The key (Truth, C psmc, psmc-rs, Spread of estimates) serves a-d and draws
each entry as plotted; e is labelled directly.

Everything comes from the tables ``E1_curves``, ``E1_table1``,
``E1_replicates``, ``E1_rs_vs_c``, ``E2_curves``, ``E2_bands``,
``E2_rs_vs_c`` and ``E7_rs_vs_c``. :func:`agreement_intervals` and
:func:`paired_intervals` (every interval of every genome both programs
fitted) serve the text's numbers (``manuscript/make_numbers.py``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import style
from .data import Source, curves, decades, time_grid, truth_xy

H_MM = 88.0  # the figure's height (README: 183 x 88 mm)

# Horizontal grid, shared by both rows (mm from the figure's left edge).
L_MM = style.LEFT_MM  # a's left spine
R_MM = 2.8  # right margin: e's centred '10^0' ends the house margin (1 mm) from the edge
RIGHT_EDGE_MM = style.WIDTH_MM["double"] - R_MM  # 180.2: d's right edge, and e's
GUT_MM = 7.5  # nominal gutter of a-d: the first layout, and the share of a column left empty
CLEAR_MM = 2.5  # clear white between a panel's ink and the next panel's y tick labels, in every gutter
PW_MM = (RIGHT_EDGE_MM - L_MM - 3 * GUT_MM) / 4.0  # nominal width; :func:`_columns` sets the drawn one
COLS = tuple(L_MM + i * (PW_MM + GUT_MM) for i in range(4))  # nominal left spines

# Vertical (mm from the figure's top edge).
LEG_TOP_MM = style.MARGIN_MM  # the key's top
ROW_TOP_MM, ROW_H_MM = 10.5, 28.0
TITLE_BASE_MM = ROW_TOP_MM - style.TITLE_RISE_MM  # letters and titles of a-d, the house rise above the axes
XLAB_TOP_MM = 43.1  # top of row 1's x label (lower if the tick labels reach it)
E_TOP_MM, E_H_MM = 50.5, 29.0
E_LABEL_GAP_MM = 1.5  # e's row names end this far left of e's axes, as tick labels do

# The fixed limits of a-c and of d; used when every drawn value lies inside them with room to spare.
# All four span 2.6 decades of Ne (the same mm per decade), d's labelled ticks (10, 100, 1,000) at the
# heights of a-c's (1, 10, 100), the top one 0.3 decade below the axes top, clear of the letters.
XL_SIM, YL_SIM = (5e3, 5e6), (5e2, 2e5)
XL_REAL2, YL_REAL2 = (5e3, 2e7), (5e3, 2e6)
PAD = 1.15  # Ne keeps this factor clear of the fixed limits; the fallback widens the data's range by it

SIM_PREF = ("bottleneck", "zigzag")
SIM_TITLES = {"bottleneck": "Simulated bottleneck", "zigzag": "Simulated zigzag",
              "constant": "Simulated constant size", "expansion": "Simulated expansion"}
HUMAN = "Homo sapiens"
COMMON = {HUMAN: "Human", "Emydura subglobosa": "Turtle", "Hydromedusa tectifera": "Turtle",
          "Pelusios castaneus": "Turtle"}

# e: eight slots of the inverted y axis, one strip each.
SLOT_DIFF_HEAD, SLOT_HUMAN, SLOT_TURTLE, SLOT_SIM = 0, 1, 2, 3
SLOT_ARROW = 4
SLOT_ERR_HEAD, SLOT_ERR_RS, SLOT_ERR_C = 5, 6, 7
E_XLIM = (1e-6, 1.0)
DOT_S = style.STRIP_MS ** 2  # scatter size of a strip's circle (2.4 pt across)
SQUARE_S = DOT_S * math.pi / 4.0  # a square of the circle's area
EDGE_LW = 0.25  # white rim that separates overlapping points of a dense strip
CASING_LW = 2.0  # white casing under a median bar, so it shows inside a dense strip
MEDIAN_MIN = 10  # a row of fewer genomes has no median bar
SEED = 7
CAP_EM = 0.716  # cap height of Arial: e's labels centre their capitals on their strip


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
    """The text's ``(Ne_C, Ne_rs)`` pairs by experiment (Table 1), and the number of genomes they come from.

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


def _positive(values):
    """``values`` as a float array without the non-finite and non-positive ones (a log axis)."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v) & (v > 0)]


def fold_median(replicates, rs_vs_c) -> float:
    """Median RMSE of psmc-rs against the truth over the median Delta, on the simulated genomes.

    ``replicates`` is ``E1_replicates``, ``rs_vs_c`` is ``E1_rs_vs_c``; nan
    when either is empty. ``\\EoneFoldMedian`` is this, rounded by
    :func:`round_fold`; e's arrow is labelled with the same number.
    """
    import numpy as np

    err = _positive(replicates[replicates.tool == "rs"].rmse_log10_ne_vs_truth)
    dif = _positive(rs_vs_c.rmse_log10_ne_rs_vs_c_per_interval)
    return float(np.median(err) / np.median(dif)) if err.size and dif.size else math.nan


def round_fold(fold: float) -> float:
    """``fold`` to one significant digit (7,044 is 7,000), as ``make_numbers`` rounds it."""
    mag = 10 ** math.floor(math.log10(fold))
    return round(fold / mag) * mag


def _limits(pref: Tuple[float, float], values: Sequence[float], room: float = 1.0) -> Tuple[float, float]:
    """``pref`` if every value lies at least a factor ``room`` inside both its ends, else the values' range rounded out.

    Ne takes ``room`` :data:`PAD`, so no band or curve grazes the axes' ends;
    time takes 1, as the curves end at their window's ends by design.
    """
    v = [float(x) for x in values if x > 0 and math.isfinite(x)]
    if not v or (min(v) / room > pref[0] and max(v) * room < pref[1]):
        return pref
    return decades(min(v) / PAD, max(v) * PAD)


def _rep_of(e1c, m: str) -> Optional[int]:
    """Replicate 1 of model ``m`` if both programs fitted it, else the lowest both fitted."""
    def fitted(tool):
        return {r for (mm, t, r) in e1c if mm == m and t == tool}

    both = sorted(fitted("c") & fitted("rs"))
    if not both:
        return None
    return 1 if 1 in both else both[0]


def _clip_to(artist, ax, window) -> None:
    """Clip ``artist`` to ``window`` in time, so that its ends are flush with the window's.

    A riser just inside the window's end would otherwise let the 2.4 pt
    underlay run past the end of the 0.75 pt line on top of it.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory

    lo, hi = window
    artist.set_clip_path(Rectangle((lo, 0.0), hi - lo, 1.0,
                                   transform=blended_transform_factory(ax.transData, ax.transAxes)))


def _fit(ax, curve, window, kw) -> List[float]:
    """Draw a fitted step curve on ``window``, clipped to it; its drawn values."""
    x, y = style.step_xy(curve.years, curve.ne, window)
    (line,) = ax.plot(x, y, drawstyle="steps-post", **kw)
    _clip_to(line, ax, window)
    return list(y)


def _others_name(names: Sequence[Optional[str]], n: int) -> str:
    """The label of e's non-human row: the genomes' shared common name, plural unless ``n`` is 1 ('Turtles')."""
    common = {COMMON.get(s) if s else None for s in names}
    if len(common) == 1 and None not in common:
        name = common.pop()
        return name if n == 1 else name + "s"
    return "Other genome" if n == 1 else "Other genomes"


def _columns(labels: Sequence[float], overhang: Sequence[float]) -> Tuple[List[float], float]:
    """Left spines of a-d and their one width, so that every gutter has :data:`CLEAR_MM` of clear white.

    ``labels[i]`` is how far panel ``i``'s ink (its y tick labels) reaches
    left of its spine, ``overhang[i]`` how far it reaches right of its right
    edge (both mm). a's spine stays at :data:`L_MM` and d's right edge at
    :data:`RIGHT_EDGE_MM`.
    """
    w = (RIGHT_EDGE_MM - L_MM - 3 * CLEAR_MM - sum(labels[1:]) - sum(overhang[:3])) / 4.0
    cols = [L_MM]
    for i in range(1, 4):
        cols.append(cols[-1] + w + overhang[i - 1] + CLEAR_MM + labels[i])
    return cols, w


def _place(ax, x: float, y: float, w: float, h: float) -> None:
    """Move ``ax`` to ``x`` mm right and ``y`` mm down from the figure's top-left corner, ``w`` x ``h`` mm."""
    fw, fh = ax.figure.get_figwidth() / style.MM, ax.figure.get_figheight() / style.MM
    ax.set_position((x / fw, 1.0 - (y + h) / fh, w / fw, h / fh))


def _band(ax, x, lo, hi, window) -> None:
    band = ax.fill_between(x, lo, hi, step="post", color=style.FILL, linewidth=0.0, zorder=style.Z_BAND)
    _clip_to(band, ax, window)


def _title_of(sid: str, species: Optional[str]) -> str:
    """'Human NA12878' for a human, else the common name and the italic binomial."""
    if species is None:
        return sid
    if species == HUMAN:
        return f"{COMMON[HUMAN]} {sid}"
    return (COMMON.get(species, "") + " " + r"$\mathit{%s}$" % species.replace(" ", r"\ ")).strip()


def _centred_dash(dash, lw: float = style.LW_DATA):
    """``dash`` started at the phase that centres it on a legend handle, so both ends are equal part-dashes.

    matplotlib scales a pattern and its offset by the line's width; the
    handle is ``LEGEND_KW['handlelength']`` ems of 7 pt long.
    """
    on, off = dash[1][0] * lw, dash[1][1] * lw
    length = style.LEGEND_KW["handlelength"] * style.TEXT_PT
    return (((on - length) / 2.0) % (on + off) / lw, dash[1])


def _delta_note(ax, delta: Optional[float]) -> None:
    if delta is not None and math.isfinite(delta):
        style.note(ax, 0.97, 0.05, r"$\Delta$ " + _sci(delta), ha="right", va="bottom", zorder=style.Z_TEXT)


def draw(src: Source):
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator
    from matplotlib.transforms import blended_transform_factory, offset_copy

    prof = src.profile
    g1 = prof.e1.scaling.gen_years
    e1c = curves(src.table("E1_curves"), ["model", "tool", "replicate"])
    win1 = {r.model: (float(r.window_lo_years), float(r.window_hi_years)) for r in src.table("E1_table1").itertuples()}
    rep = src.table("E1_replicates")
    e1d = src.table("E1_rs_vs_c")
    e2c = curves(src.table("E2_curves"), ["sample_id", "tool"])
    bands = src.table("E2_bands")
    rsc2 = src.table("E2_rs_vs_c")
    rsc7 = src.table("E7_rs_vs_c")
    win2 = {r.sample_id: (float(r.window_lo_years), float(r.window_hi_years)) for r in rsc2.itertuples()}
    delta2 = {r.sample_id: float(r.rmse_log10_ne_rs_vs_c_per_interval) for r in rsc2.itertuples()}
    species = {g.sample_id: g.species for g in prof.e2.genomes}
    order2 = [s for s in prof.e2.panel_order if s in win2] + sorted(s for s in win2 if s not in prof.e2.panel_order)

    # ------------------------------------------------------------------ the panels of the top row
    fitted = {m for m in win1 if _rep_of(e1c, m) is not None}
    sims = [m for m in SIM_PREF if m in fitted] + [m for m in prof.e1.models if m in fitted and m not in SIM_PREF]
    both2 = [s for s in order2 if (s, "c") in e2c and (s, "rs") in e2c]
    human = next((s for s in both2 if species.get(s) == HUMAN), None)
    others = [s for s in both2 if species.get(s) != HUMAN and math.isfinite(delta2.get(s, math.nan))]
    worst = max(others, key=lambda s: delta2[s]) if others else None
    panels: List[Optional[Tuple[str, str]]] = [("sim", m) for m in sims[:2]]
    panels += [None] * (2 - len(panels)) + [("real", human) if human else None, ("real", worst) if worst else None]

    fig = style.figure("double", H_MM)
    axs = [axes_mm(fig, COLS[i], ROW_TOP_MM, PW_MM, ROW_H_MM) for i in range(4)]  # moved by _columns below

    c_kw = style.underlay_kw(style.C_PSMC)
    rs_kw = style.overlay_kw(style.RS02)
    shared_x: List[float] = []  # a-c: one pair of limits from everything they draw
    shared_y: List[float] = []
    own: Dict[int, Tuple[List[float], List[float]]] = {}
    titles: Dict[int, str] = {}
    for i, panel in enumerate(panels):
        ax = axs[i]
        if panel is None:
            ax.set_axis_off()
            continue
        kind, key = panel
        xs: List[float] = []
        ys: List[float] = []
        if kind == "sim":
            w = win1[key]
            r = _rep_of(e1c, key)
            grid = time_grid(w, 800)
            reps = sorted(rr for (mm, t, rr) in e1c if mm == key and t == "rs")
            on = np.vstack([e1c[(key, "rs", rr)].at(grid) for rr in reps])
            _band(ax, grid, on.min(axis=0), on.max(axis=0), w)
            ys += list(on.min(axis=0)) + list(on.max(axis=0))
            ys += _fit(ax, e1c[(key, "c", r)], w, c_kw)
            ys += _fit(ax, e1c[(key, "rs", r)], w, rs_kw)
            tx, ty = truth_xy(key, g1, w)
            ax.plot(tx, ty, **style.truth_kw(dash_joinstyle="miter"))  # square corners, as the fits'
            ys += list(ty)
            d = e1d[(e1d.model == key) & (e1d.replicate == r)].rmse_log10_ne_rs_vs_c_per_interval
            _delta_note(ax, float(d.iloc[0]) if len(d) else None)
            titles[i] = SIM_TITLES.get(key, f"Simulated {key}")
        else:
            w = win2[key]
            b = bands[(bands.sample_id == key) & (bands.trunk_bins == prof.e2.bootstrap_trunk_bins)]
            b = b.sort_values("years")
            b = b[b.years < w[1]]
            if not b.empty:  # the band's steps, from the window's start to its end, as the fits are drawn
                bx, blo = style.step_xy(b.years.to_numpy(), b.ne_band_lo.to_numpy(), w)
                _, bhi = style.step_xy(b.years.to_numpy(), b.ne_band_hi.to_numpy(), w)
                _band(ax, bx, blo, bhi, w)
                ys += list(blo) + list(bhi)
            ys += _fit(ax, e2c[(key, "c")], w, c_kw)
            ys += _fit(ax, e2c[(key, "rs")], w, rs_kw)
            _delta_note(ax, delta2.get(key))
            titles[i] = _title_of(key, species.get(key))
        xs += list(w)
        if i < 3:
            shared_x += xs
            shared_y += ys
        else:
            own[i] = (xs, ys)
    xl, yl = _limits(XL_SIM, shared_x), _limits(YL_SIM, shared_y, PAD)
    for i, panel in enumerate(panels):
        if panel is None:
            continue
        ax = axs[i]
        lims = (xl, yl) if i < 3 else (_limits(XL_REAL2, own[i][0]), _limits(YL_REAL2, own[i][1], PAD))
        style.thousands_axis(ax, "x", lims[0])
        style.thousands_axis(ax, "y", lims[1])
        ax.set_xlabel("")
        ax.set_ylabel(style.NE_LABEL if i == 0 else "")
    drawn = [axs[i] for i, p in enumerate(panels) if p is not None]

    # Equal widths, equal clear white: the gutters follow the drawn y tick labels (d's '1,000' is wider).
    fw = fig.get_figwidth() / style.MM
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels, overhang = [], []
    for i, ax in enumerate(axs):
        if panels[i] is None:  # an empty column keeps the nominal share
            labels.append(GUT_MM - CLEAR_MM)
            overhang.append(0.0)
            continue
        ink = ax.get_tightbbox(renderer)
        pos = ax.get_position()
        labels.append(pos.x0 * fw - ink.x0 / fig.dpi / style.MM)
        overhang.append(max(0.0, ink.x1 / fig.dpi / style.MM - pos.x1 * fw))
    cols, pw = _columns(labels, overhang)
    for i, ax in enumerate(axs):
        _place(ax, cols[i], ROW_TOP_MM, pw, ROW_H_MM)
        if panels[i] is not None:
            style.fig_text(fig, cols[i], TITLE_BASE_MM, titles[i], ha="left", va="baseline")
    head_x = cols[0] - labels[0]  # e's label column starts at a's ink (its y label), the figure's left edge of ink
    title_x = cols[0]  # e's group headers start where a-d's titles do, at a's spine
    ax_e = axes_mm(fig, cols[1], E_TOP_MM, RIGHT_EDGE_MM - cols[1], E_H_MM)  # moved once its row names are measured

    # ------------------------------------------------------------------ e: the difference against the error
    hum_ids = set(rsc7.sample_id)
    human_d = list(rsc7.rmse_log10_ne_rs_vs_c_per_interval) + [
        delta2[s] for s in sorted(delta2) if species.get(s) == HUMAN and s not in hum_ids]
    other_ids = [s for s in sorted(delta2) if species.get(s) != HUMAN]
    turtle_d = [delta2[s] for s in other_ids]
    err = {}
    for row in rep[rep.tool.isin(["rs", "c"])].itertuples():
        err.setdefault((row.model, row.replicate), {})[row.tool] = float(row.rmse_log10_ne_vs_truth)
    genomes = sorted(err)  # one height per simulated genome, shared by its two rows
    truth_jit = np.random.default_rng(SEED).uniform(-style.STRIP_JITTER, style.STRIP_JITTER, len(genomes))

    def truth_row(tool):
        v = np.array([err[k].get(tool, math.nan) for k in genomes], dtype=float)
        keep = np.isfinite(v) & (v > 0)
        return v[keep], truth_jit[keep]

    def own_jitter(v):
        return np.random.default_rng(SEED).uniform(-style.STRIP_JITTER, style.STRIP_JITTER, len(v))

    diff_rows = []
    for slot, name, values in ((SLOT_HUMAN, "Human", human_d), (SLOT_TURTLE, None, turtle_d),
                               (SLOT_SIM, "Simulated", e1d.rmse_log10_ne_rs_vs_c_per_interval)):
        v = _positive(values)
        if name is None:  # 'Turtles' in production
            name = _others_name([species.get(s) for s in other_ids], v.size)
        diff_rows.append((slot, name, v, own_jitter(v), style.INK2, "o", DOT_S))
    err_rows = []
    for slot, ent in ((SLOT_ERR_RS, style.RS02), (SLOT_ERR_C, style.C_PSMC)):
        v, jit = truth_row("rs" if ent is style.RS02 else "c")
        err_rows.append((slot, ent.label, v, jit, ent.color, ent.marker, DOT_S if ent.marker == "o" else SQUARE_S))

    medians: Dict[int, float] = {}
    for slot, _name, v, jit, color, marker, size in diff_rows + err_rows:
        if not v.size:
            continue
        ax_e.scatter(v, slot + jit, s=size, marker=marker, color=color, edgecolors=style.SURFACE,
                     linewidths=EDGE_LW, zorder=style.Z_CONTEXT + 1)
        if v.size >= MEDIAN_MIN:
            m = float(np.median(v))
            medians[slot] = m
            seg = ([m, m], [slot - style.STRIP_BAR_HALF, slot + style.STRIP_BAR_HALF])
            ax_e.plot(*seg, color=style.SURFACE, linewidth=CASING_LW, solid_capstyle="butt", zorder=style.Z_DATA)
            ax_e.plot(*seg, color=style.INK, linewidth=style.LW_DATA, solid_capstyle="butt", zorder=style.Z_TRUTH)

    at_left = blended_transform_factory(fig.transFigure, ax_e.transData)

    def on_strip(transform, size):
        """``transform`` moved down so that text of ``size`` set on its baseline has its capitals centred."""
        return offset_copy(transform, fig=fig, y=-CAP_EM * size / 2.0, units="points")

    names = []  # e's row names, right-aligned on the label column's right edge once it is known

    def block(rows, head_slot: int, head: str) -> None:
        """``head`` (7 pt), left-aligned with a-d's titles, and each row's name with its n (6 pt)."""
        if not any(row[2].size for row in rows):
            return
        ax_e.text(title_x / fw, head_slot, head, transform=on_strip(at_left, style.TEXT_PT), ha="left",
                  va="baseline", fontsize=style.TEXT_PT, color=style.INK, zorder=style.Z_TEXT)
        for slot, name, vals, *_ in rows:
            if vals.size:
                names.append(ax_e.text(head_x / fw, slot, f"{name} ({vals.size})",
                                       transform=on_strip(at_left, style.LABEL_PT), ha="right", va="baseline",
                                       fontsize=style.LABEL_PT, color=style.INK, zorder=style.Z_TEXT))

    block(diff_rows, SLOT_DIFF_HEAD, r"psmc-rs against C psmc ($\Delta$)")  # the Delta of a-d
    block(err_rows, SLOT_ERR_HEAD, "Error against the truth (simulated)")

    # A forest plot: the label column is as wide as the widest row name, the axes take the rest of the width.
    col_w = max((t.get_window_extent(renderer).width / fig.dpi / style.MM for t in names), default=0.0)
    e_x = head_x + col_w + E_LABEL_GAP_MM
    _place(ax_e, e_x, E_TOP_MM, RIGHT_EDGE_MM - e_x, E_H_MM)
    for t in names:
        t.set_x((e_x - E_LABEL_GAP_MM) / fw)

    # The arrow from psmc-rs's difference from C psmc to its error, on the simulated genomes (medians).
    if SLOT_SIM in medians and SLOT_ERR_RS in medians:
        dif, e_rs = medians[SLOT_SIM], medians[SLOT_ERR_RS]
        # Dotted, so lighter than the arrow, in the spines' grey: in REF's lighter grey they vanished at print size.
        leader = {"color": style.AXIS, "linewidth": style.LW_THIN, "linestyle": style.DOT, "zorder": style.Z_CONTEXT}
        ax_e.plot([dif, dif], [SLOT_SIM + style.STRIP_BAR_HALF, SLOT_ARROW], **leader)
        ax_e.plot([e_rs, e_rs], [SLOT_ERR_RS - style.STRIP_BAR_HALF, SLOT_ARROW], **leader)
        ax_e.annotate("", xy=(dif, SLOT_ARROW), xytext=(e_rs, SLOT_ARROW), arrowprops=dict(style.ARROW_PROPS),
                      zorder=style.Z_DATA)
        fold = fold_median(rep, e1d)  # the ratio of the medians the arrow joins, as \EoneFoldMedian
        if fold > 0 and math.isfinite(fold):
            ax_e.text(math.sqrt(dif * e_rs), SLOT_ARROW - 0.15, f"{round_fold(fold):,.0f}{style.TIMES} smaller",
                      ha="center", va="bottom", fontsize=style.TEXT_PT, fontweight="bold", color=style.INK,
                      zorder=style.Z_TEXT)

    style.log_axis(ax_e, "x", fmt="pow10", lim=E_XLIM)
    ax_e.xaxis.set_minor_locator(NullLocator())
    ax_e.set_ylim(SLOT_ERR_C + 0.5, SLOT_DIFF_HEAD - 0.5)
    ax_e.set_yticks([])
    ax_e.spines["left"].set_visible(False)
    ax_e.set_xlabel("RMSE of " + style.LOGNE)

    # ------------------------------------------------------------------ key, x label, letters
    handles = [style.handle(style.TRUTH, linestyle=_centred_dash(style.DASH)), style.handle(style.C_PSMC, "underlay"),
               style.handle(style.RS02), Patch(facecolor=style.FILL, linewidth=0.0, label="Spread of estimates")]
    leg = style.top_legend(fig, handles, x_mm=L_MM, top_mm=LEG_TOP_MM)
    if drawn:
        xlab = style.row_xlabel(fig, drawn)
        if (1.0 - xlab.get_position()[1]) * H_MM < XLAB_TOP_MM:  # a little clear of the tick labels
            xlab.set_y(1.0 - XLAB_TOP_MM / H_MM)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    off = leg.get_window_extent(renderer).x0 / fig.dpi / style.MM - L_MM
    if abs(off) > 0.05:  # the first handle starts right above a's left spine
        (x, y), _ = leg.get_bbox_to_anchor().get_points()  # pixels
        leg.set_bbox_to_anchor((x / fig.bbox.width - off / fw, y / fig.bbox.height), transform=fig.transFigure)
    letters = iter("abcde")
    for i, ax in enumerate(axs):
        if panels[i] is None:
            continue
        x_mm = style.LETTER_X_MM if i == 0 else style.ink_left_mm(ax, renderer)
        style.panel_letter(fig, x_mm, TITLE_BASE_MM, next(letters))
    fig.text(style.LETTER_X_MM / fw, SLOT_DIFF_HEAD, next(letters), transform=on_strip(at_left, style.TEXT_PT),
             ha="left", va="baseline", fontsize=style.PANEL_PT, fontweight="bold", color=style.INK)  # header's baseline
    return fig


__all__ = ["agreement_intervals", "draw", "fold_median", "paired_intervals", "round_fold"]
