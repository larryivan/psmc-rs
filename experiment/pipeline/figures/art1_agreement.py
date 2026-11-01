"""The article's Fig 1: psmc-rs reproduces C psmc, on simulated and on real genomes.

Five panels at 183 x 99 mm. Top row, a-d: four histories fitted by both
programs to the same genome, C psmc's as a wide pale-orange underlay
(:data:`.style.C_HALO`) under psmc-rs's thin blue line, over the genome's
informative window: 1% to 99% of its pairwise coalescence times, under the
true history for a simulated genome (``E1_table1``) and under C psmc's fit
for a real one (``E2_rs_vs_c``). Curves and band are clipped to it, so all
end flush with it, and never resampled, so the two programs' risers stay
at their exact times. The note at the bottom right of each, Delta, is the
RMSE of log10 Ne between the two programs' fits, interval by interval over
the window, each interval weighted by its log-time width inside it
(``rmse_log10_ne_rs_vs_c_per_interval``): the root mean square, over log
time, of the strip below.

Under each panel, a strip of the differences: psmc-rs's log10 Ne minus C
psmc's, interval by interval on C psmc's intervals
(:func:`..metrics.interval_log10_ratio`), drawn as psmc-rs's thin blue
steps over the same window, about a grey zero line across the strip
(:func:`.style.ref_line`), in units of 1e-4 (:data:`STRIP_UNIT`; a's strip
has the unit for its y label, '×10⁻⁴', in one column with a's N_e label:
:func:`_pin_ylabels`). a-c's strips share one scale, -1 to 1, and d's has
its own, -5 to 5 (:data:`STRIP_LIM`), each ticked at its ends and 0; a
scale widens to the next 1-2-5 value while a difference exceeds
:data:`STRIP_FILL` of it. The strips carry the time tick labels and the
row's one x label (thousands of years ago); the panels above keep their
ticks, unlabelled.

* a, b (E1): two simulated models, ``bottleneck`` and ``zigzag`` first
  (:data:`SIM_PREF`), then the profile's other models in order; replicate
  1 where both programs fitted it, else the first both fitted. Dashed ink:
  the true Ne(t) on top (:func:`_truth_line`). Delta from ``E1_rs_vs_c``.
* c (E2): the human genome, the first sample of ``config.E2.panel_order``
  whose species is Homo sapiens and that both programs fitted (NA12878).
  Pale blue (:data:`.style.RS_BAND`, opaque, square steps, under both
  fits): the pointwise 95% band of psmc-rs's bootstrap at the default block
  size (``E2_bands`` at ``bootstrap_trunk_bins``). No truth.
* d (E2): the non-human genome on which the programs differ most (the
  largest Delta in ``E2_rs_vs_c``: Pelusios castaneus in production),
  drawn as c.

a, b and c share both scales (5,000 to 5,000,000 years; Ne 500 to
200,000); d has its own (5,000 to 20,000,000 years; Ne 5,000 to
2,000,000), with the same span of Ne, 2.6 decades, so its mm per decade
and the heights of its labelled ticks are theirs. The Ne limits widen to
round values of the data (:func:`.data.decades`) when they would not hold,
at least a factor :data:`PAD` from either end, every Ne drawn and, in a
and b, the range (pointwise minimum to maximum) of psmc-rs's fits to every
replicate of the model, which is not drawn; the time limits, when they
would not hold every window (the tiny test profile). A column with
nothing to draw is left empty, without letter or title, and the letters
of the drawn panels run on. The four columns are equally wide, with the
same clear white (:data:`CLEAR_MM`) between a column's ink (its panel's
and its strip's) and the next one's y tick labels, so d's wider labels
('1,000') get a wider gutter. Letters and titles share one baseline, and
a's y labels one column, in every output: matplotlib's PDF and SVG raise
mathtext by up to 1 pt, and d's title (its binomial in italics) and a's y
labels are put back on their baselines there (:func:`.style.on_baseline`).

* e: a strip plot on one log axis of RMSE of log10 Ne, laid out as a
  forest plot. Upper block, in dark grey (:data:`.style.INK2`): psmc-rs
  against C psmc (Delta, as in a-d) for every genome both programs fitted,
  each counted once: ``Human`` (the 1000 Genomes genomes of
  ``E7_rs_vs_c``, NA12878 among them, plus any human of E2 that E7 lacks),
  the non-human genomes of ``E2_rs_vs_c`` (named by their shared common
  name, pluralised: ``Turtles``; ``Other genomes`` when their common names
  differ) and ``Simulated`` (``E1_rs_vs_c``). Under Simulated, in light
  grey (:data:`.style.REF`; its name in INK2), the noise floor of Delta:
  the same metric between C psmc's own fits of the same simulated genomes
  on two platforms (``C psmc on two platforms``; :func:`noise_floor`), each
  genome at its height in Simulated. Rings (:func:`.style.ring_kw`, with a
  white rim, :data:`RIM_LW`, so each lies visibly on top of the dots, bar
  or ring it crosses) mark the genomes of a-d, each named by its panel's
  letter (6 pt ink), clear of the rings and of the strip's median bar: the
  human's below, the non-human's to its right, the simulated pair's above,
  with one label for rings of a strip within :data:`RING_JOIN` decade of
  each other ('a, b', the pair's name, in the letters' order).
  Lower block: each program's error against the truth on the simulated
  genomes (``E1_replicates``; blue circles psmc-rs, orange squares C psmc,
  one shared jitter so each genome sits at the same height in both rows).
  Bars (ink, cased in white as the dots are rimmed, :data:`CASING_LW`) are
  the medians of rows of at least :data:`MEDIAN_MIN` genomes. A double
  arrow joins the median of ``Simulated`` to that of psmc-rs's error,
  labelled with their ratio to one significant digit, as
  ``\\EoneFoldMedian`` rounds it (:func:`fold_median`, :func:`round_fold`),
  in the headline style (:data:`.style.FOLD_HEADLINE`): '7,000× smaller' in
  production; dotted leaders (:data:`.style.LEADER_KW`) join its ends to
  the two bars, over any point they cross. Each block has a 7 pt header
  (:data:`DIFF_HEAD`, 'Δ between two fits of each genome', which holds for
  the noise floor too; :data:`ERR_HEAD`, 'Error against the truth
  (simulated)') starting where a-d's titles do, at a's spine, the letter
  'e' on the first header's baseline; each row its name with its n (6 pt),
  in one label column that starts at the left edge of a's ink (its y
  labels), is as wide as the widest name and ends :data:`E_LABEL_GAP_MM`
  left of e's axes, which run from there to d's right edge. A row with
  nothing to draw (the noise floor in the tiny test profile) leaves its
  slot empty.

The noise floor is drawn only from the fits it describes. Its file,
committed outside the runs directory, describes the production's fits: a
row belongs to a genome of the figure's ``E1_rs_vs_c`` when it holds the
genome's Delta (``rmse_window_rs_vs_c_ibex``, to :data:`FLOOR_TOL`). All
40 do in production, none in the tiny test profile; a file that holds the
Delta of only some of the genomes is stale, and the figure is refused
(ValueError).

The key (Truth, C psmc, psmc-rs, Bootstrap 95% band; the band's entry
only when a band is drawn), one line at the top (:func:`.style.top_legend`)
with a-d at the house's top under a key (:data:`.style.TOP_LEGEND_MM`),
serves a-d and draws each entry as plotted; e is labelled directly.

The tables: ``E1_curves``, ``E1_table1``, ``E1_replicates``,
``E1_rs_vs_c``, ``E2_curves``, ``E2_bands``, ``E2_rs_vs_c`` and
``E7_rs_vs_c``; and ``crossplatform/side_by_side_E1.csv``
(``experiment/crossplatform/``, :data:`.schemas.COMMITTED`), its
``rmse_window_c_laptop_vs_ibex``. :func:`agreement_intervals` and
:func:`paired_intervals` (every interval of every genome both programs
fitted) serve the text's numbers (``manuscript/make_numbers.py``).
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .. import metrics
from . import style
from .data import NICE_STEPS, Source, curves, decades, table_windows, time_grid, truth_xy
from .schemas import COMMITTED

H_MM = 99.0  # the figure's height (README: 183 x 99 mm)

# Horizontal grid, shared by both rows (mm from the figure's left edge).
L_MM = style.LEFT_MM  # a's left spine
R_MM = 2.8  # right margin: e's centred '10^0' ends the house margin (1 mm) from the edge
RIGHT_EDGE_MM = style.WIDTH_MM["double"] - R_MM  # 180.2: d's right edge, and e's
GUT_MM = 7.5  # nominal gutter of a-d: the first layout, and the share of a column left empty
CLEAR_MM = 2.5  # clear white between a panel's ink and the next panel's y tick labels, in every gutter
PW_MM = (RIGHT_EDGE_MM - L_MM - 3 * GUT_MM) / 4.0  # nominal width; :func:`_columns` sets the drawn one
COLS = tuple(L_MM + i * (PW_MM + GUT_MM) for i in range(4))  # nominal left spines

# Vertical (mm from the figure's top edge): a-d, a difference strip under each, then e.
ROW_TOP_MM = style.TOP_LEGEND_MM  # 8.6: a-d under the key, as the first row of every keyed figure (Fig S1)
ROW_H_MM = 27.0
STRIP_GAP_MM, STRIP_H_MM = 1.3, 8.5  # a strip's gap under its panel (clear of the panel's x ticks), its height
STRIP_TOP_MM = ROW_TOP_MM + ROW_H_MM + STRIP_GAP_MM  # 36.9
STRIP_BOTTOM_MM = STRIP_TOP_MM + STRIP_H_MM  # 45.4
TITLE_BASE_MM = ROW_TOP_MM - style.TITLE_RISE_MM  # letters and titles of a-d, the house rise above the axes
XLAB_TOP_MM = STRIP_BOTTOM_MM + 4.6  # top of the row's x label, under the strips (lower if the tick labels reach it)
E_TOP_MM = STRIP_BOTTOM_MM + style.ROW_GAP_TITLED_MM  # 57.4: the house gap under an x label
# Under e, the house's bottom under power-of-ten tick labels, which push its x label down.
E_H_MM = H_MM - E_TOP_MM - style.BOTTOM_POW10_MM  # 33.75: nine slots of 3.75 mm
E_LABEL_GAP_MM = 1.5  # e's row names end this far left of e's axes, as tick labels do

# The fixed limits of a-c and of d; used when every drawn value lies inside them with room to spare.
# All four span 2.6 decades of Ne (the same mm per decade), d's labelled ticks (10, 100, 1,000) at the
# heights of a-c's (1, 10, 100), the top one 0.3 decade below the axes top, clear of the letters.
XL_SIM, YL_SIM = (5e3, 5e6), (5e2, 2e5)
XL_REAL2, YL_REAL2 = (5e3, 2e7), (5e3, 2e6)
PAD = 1.15  # Ne keeps this factor clear of the fixed limits; the fallback widens the data's range by it

# The difference strips: log10 Ne in units of 1e-4. a-c share +-1, d has its own +-5, as it has its own Ne and
# time axes (one shared +-5 would flatten c's differences to a sixth of a millimetre).
STRIP_UNIT = 1e-4
STRIP_LIM = (1.0, 5.0)  # a-c's and d's, in STRIP_UNIT
STRIP_FILL = 0.9  # a scale holds its differences within this share of its ends, else widens to the next 1-2-5

SIM_PREF = ("bottleneck", "zigzag")
SIM_TITLES = {"bottleneck": "Simulated bottleneck", "zigzag": "Simulated zigzag",
              "constant": "Simulated constant size", "expansion": "Simulated expansion"}
HUMAN = "Homo sapiens"
COMMON = {HUMAN: "Human", "Emydura subglobosa": "Turtle", "Hydromedusa tectifera": "Turtle",
          "Pelusios castaneus": "Turtle"}
C_KW = style.underlay_kw(style.C_PSMC)
RS_KW = style.overlay_kw(style.RS02)

# e: nine slots of the inverted y axis, one strip each. The noise floor sits under Simulated, its genomes'.
SLOT_DIFF_HEAD, SLOT_HUMAN, SLOT_TURTLE, SLOT_SIM, SLOT_FLOOR = 0, 1, 2, 3, 4
SLOT_ARROW = 5
SLOT_ERR_HEAD, SLOT_ERR_RS, SLOT_ERR_C = 6, 7, 8
E_XLIM = (1e-6, 1.0)
DOT_S = style.STRIP_MS ** 2  # scatter size of a strip's circle (2.4 pt across)
SQUARE_S = DOT_S * math.pi / 4.0  # a square of the circle's area
# The white casing of a median bar, and the white rim of a ring: the dots' own rim (style.STRIP_EDGE_LW) on either
# side of the line, so a bar or ring shows inside a dense strip and a dot it crosses reads as a dot behind it (a
# wider casing cut the dots beside a bar into crescents).
CASING_LW = style.LW_DATA + 2.0 * style.STRIP_EDGE_LW
RIM_LW = style.LW_THIN + 2.0 * style.STRIP_EDGE_LW
MEDIAN_MIN = 10  # a row of fewer genomes has no median bar
SEED = 7
CAP_EM = 0.716  # cap height of Arial: e's labels centre their capitals on their strip
# e's block headers. The first names Delta and nothing else, as its rows compare two fits of each genome either
# by the two programs or (the noise floor) by C psmc on two platforms. Plain text (U+0394, the glyph mathtext's
# \Delta takes from Arial), so that it sits on the letter's baseline in the PDF too.
DIFF_HEAD = "Δ between two fits of each genome"
ERR_HEAD = "Error against the truth (simulated)"

# The noise floor of Delta: C psmc's fits of E1's genomes on two platforms (experiment/crossplatform), a row of
# which belongs to a genome of E1_rs_vs_c when its Delta of the production's fits is the genome's.
FLOOR_FILE = "crossplatform/side_by_side_E1.csv"
FLOOR_NAME = "C psmc on two platforms"
FLOOR_TOL = 1e-12  # relative; the file copies E1_rs_vs_c's value (equal in production)

# Rings: each label goes where e's strips leave room, RING_GAP_PT clear of its rings and of its strip's median
# bar: the human's below (between the Human and the Turtles strips), the non-human's to its right (its strip is
# sparse), the simulated pair's above (where the Turtles strip is empty).
RING_SIDE = {SLOT_HUMAN: "below", SLOT_TURTLE: "right", SLOT_SIM: "above"}
RING_JOIN = 0.15  # decades: rings of one strip this close share a label ('a, b')
RING_R_PT = (math.sqrt(style.RING_S) + style.LW_THIN) / 2.0  # a ring's outer radius (2.4 pt)
RING_GAP_PT = 1.0
SLOT_PT = E_H_MM * style.PT_PER_MM / (SLOT_ERR_C + 1)  # one slot of e (10.6 pt)


def paired_intervals(c, rs, window: Tuple[float, float]) -> List[Tuple[float, float]]:
    """``(Ne_C, Ne_rs)`` of every interval of C psmc's fit ``c`` overlapping ``window``, and psmc-rs's ``rs``."""
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


def noise_floor(rs_vs_c) -> Dict[Tuple[str, int], float]:
    """C psmc's Delta between its fits on two platforms, by ``(model, replicate)`` of ``rs_vs_c`` (``E1_rs_vs_c``).

    Read from :data:`FLOOR_FILE`, whose rows describe the production's fits:
    a row belongs to a genome when it holds the genome's Delta. Empty when
    no row does (the tiny profile's genomes); a ValueError when only some
    genomes have one, as the file then describes other fits.
    """
    side = COMMITTED[FLOOR_FILE].read()
    rows = {(str(r.model), int(r.replicate)): r for r in side.itertuples()}
    ours = {(str(r.model), int(r.replicate)): float(r.rmse_log10_ne_rs_vs_c_per_interval)
            for r in rs_vs_c.itertuples() if math.isfinite(r.rmse_log10_ne_rs_vs_c_per_interval)}
    floor = {k: float(rows[k].rmse_window_c_laptop_vs_ibex) for k, d in ours.items()
             if k in rows and math.isclose(float(rows[k].rmse_window_rs_vs_c_ibex), d, rel_tol=FLOOR_TOL)}
    if floor and len(floor) < len(ours):
        raise ValueError(f"{COMMITTED[FLOOR_FILE].path()} holds the Delta of {len(floor)} of E1's {len(ours)} "
                         f"genomes: it describes other fits (stale)")
    return floor


# ---------------------------------------------------------------------------
# a-d and their difference strips
# ---------------------------------------------------------------------------


def _limits(pref: Tuple[float, float], values: Sequence[float], room: float = 1.0) -> Tuple[float, float]:
    """``pref`` if every value lies at least a factor ``room`` inside both its ends, else the values' range rounded out.

    Ne takes ``room`` :data:`PAD`, so no band or curve grazes the axes' ends;
    time takes 1, as the curves end at their window's ends by design.
    """
    v = [float(x) for x in values if x > 0 and math.isfinite(x)]
    if not v or (min(v) / room > pref[0] and max(v) * room < pref[1]):
        return pref
    return decades(min(v) / PAD, max(v) * PAD)


def _strip_limit(pref: float, diffs: Sequence[float]) -> float:
    """``pref`` if every difference lies within :data:`STRIP_FILL` of it, else the next 1-2-5 value that holds them."""
    top = max((abs(d) for d in diffs if math.isfinite(d)), default=0.0)
    e = math.floor(math.log10(pref))
    while True:
        for s in NICE_STEPS:
            lim = s * 10.0 ** e
            if lim >= pref and top <= STRIP_FILL * lim:
                return lim
        e += 1


def _rep_of(e1c, m: str) -> Optional[int]:
    """Replicate 1 of model ``m`` if both programs fitted it, else the lowest both fitted."""
    def fitted(tool):
        return {r for (mm, t, r) in e1c if mm == m and t == tool}

    both = sorted(fitted("c") & fitted("rs"))
    if not both:
        return None
    return 1 if 1 in both else both[0]


def _top_panels(prof, e1c, win1, e2c, win2, delta2, species) -> List[Optional[Tuple[str, str]]]:
    """a-d as ``("sim", model)`` or ``("real", sample)``; None for a column with nothing to draw."""
    fitted = {m for m in win1 if _rep_of(e1c, m) is not None}
    sims = [m for m in SIM_PREF if m in fitted] + [m for m in prof.e1.models if m in fitted and m not in SIM_PREF]
    order2 = [s for s in prof.e2.panel_order if s in win2] + sorted(s for s in win2 if s not in prof.e2.panel_order)
    both2 = [s for s in order2 if (s, "c") in e2c and (s, "rs") in e2c]
    human = next((s for s in both2 if species.get(s) == HUMAN), None)
    others = [s for s in both2 if species.get(s) != HUMAN and math.isfinite(delta2.get(s, math.nan))]
    worst = max(others, key=lambda s: delta2[s]) if others else None
    panels: List[Optional[Tuple[str, str]]] = [("sim", m) for m in sims[:2]]
    return panels + [None] * (2 - len(panels)) + [("real", human) if human else None,
                                                   ("real", worst) if worst else None]


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


def _delta_note(ax, delta: Optional[float]) -> None:
    if delta is not None and math.isfinite(delta):
        style.note(ax, 0.97, 0.05, r"$\Delta$ " + _sci(delta), ha="right", va="bottom", zorder=style.Z_TEXT)


def _truth_line(model: str, gen_years: float, window: Tuple[float, float]):
    """The true Ne(t) of ``model`` on ``window`` (:func:`.data.truth_xy`), without the epoch joins that change nothing.

    Such a join is a point at its predecessor's time and, to a sub-visible
    1e-4, its Ne. At the zigzag's last join the exponential epoch ends a
    relative 6e-6 below the constant one, and the truth's miter join turned
    that hairpin into a 0.5 mm spike below the line. The bottleneck's steps
    (2- to 20-fold) are kept.
    """
    import numpy as np

    tx, ty = truth_xy(model, gen_years, window)
    rep = np.r_[False, np.isclose(tx[1:], tx[:-1], rtol=1e-9, atol=0.0) &
                np.isclose(ty[1:], ty[:-1], rtol=1e-4, atol=0.0)]
    return tx[~rep], ty[~rep]


def _sim_panel(ax, model: str, r: int, e1c, window, e1d, gen_years: float) -> List[float]:
    """a, b: both programs' fits to replicate ``r`` and the truth; the Ne its limits hold.

    Those include the range of psmc-rs's fits to every replicate of the
    model, which is not drawn (the underlay hides most of it), so that the
    limits hold any replicate's fit.
    """
    import numpy as np

    grid = time_grid(window, 800)
    reps = sorted(rr for (mm, t, rr) in e1c if mm == model and t == "rs")
    on = np.vstack([e1c[(model, "rs", rr)].at(grid) for rr in reps])
    ys = list(on.min(axis=0)) + list(on.max(axis=0))
    ys += _fit(ax, e1c[(model, "c", r)], window, C_KW)
    ys += _fit(ax, e1c[(model, "rs", r)], window, RS_KW)
    tx, ty = _truth_line(model, gen_years, window)
    ax.plot(tx, ty, **style.truth_kw(dash_joinstyle="miter"))  # square corners, as the fits'
    ys += list(ty)
    d = e1d[(e1d.model == model) & (e1d.replicate == r)].rmse_log10_ne_rs_vs_c_per_interval
    _delta_note(ax, float(d.iloc[0]) if len(d) else None)
    return ys


def _real_panel(ax, rs, c, window, band, delta: Optional[float]) -> Tuple[List[float], bool]:
    """c, d: psmc-rs's bootstrap band (``band``, its rows of E2_bands) under both programs' fits.

    Returns the Ne drawn and whether a band is.
    """
    ys: List[float] = []
    b = band.sort_values("years")
    b = b[b.years < window[1]]
    if not b.empty:  # the band's steps, from the window's start to its end, as the fits are drawn
        bx, blo = style.step_xy(b.years.to_numpy(), b.ne_band_lo.to_numpy(), window)
        _, bhi = style.step_xy(b.years.to_numpy(), b.ne_band_hi.to_numpy(), window)
        _clip_to(ax.fill_between(bx, blo, bhi, step="post", **style.band_kw()), ax, window)
        ys += list(blo) + list(bhi)
    ys += _fit(ax, c, window, C_KW)
    ys += _fit(ax, rs, window, RS_KW)
    _delta_note(ax, delta)
    return ys, not b.empty


def _diff_strip(ax, rs, c, window) -> List[float]:
    """Under a panel: psmc-rs's log10 Ne minus C psmc's, interval by interval on C psmc's, in :data:`STRIP_UNIT`.

    Steps over ``window``, clipped to it as the fits above, about the zero
    line; returns the differences drawn.
    """
    x, y = style.step_xy(c.years, metrics.interval_log10_ratio(rs, c) / STRIP_UNIT, window)
    (line,) = ax.plot(x, y, drawstyle="steps-post", **RS_KW)
    _clip_to(line, ax, window)
    style.ref_line(ax, 0.0)
    return list(y)


def _strip_axes(ax, xlim: Tuple[float, float], lim: float) -> None:
    """A strip's axes: the panel's times, labelled; differences from ``-lim`` to ``lim``, ticked at both and 0."""
    style.thousands_axis(ax, "x", xlim)
    ax.set_ylim(-lim, lim)
    ax.set_yticks([-lim, 0.0, lim])
    ax.yaxis.set_major_formatter(style.plain_formatter())


def _title_of(sid: str, species: Optional[str]) -> str:
    """'Human NA12878' for a human, else the common name and the italic binomial."""
    if species is None:
        return sid
    if species == HUMAN:
        return f"{COMMON[HUMAN]} {sid}"
    return (COMMON.get(species, "") + " " + r"$\mathit{%s}$" % species.replace(" ", r"\ ")).strip()


def _pin_ylabels(fig, axes: Sequence) -> None:
    """Hang the y labels of ``axes`` (a and its strip, under one left spine) on one baseline, in one column.

    The baseline is the farther left of matplotlib's two (each label's box
    ``axes.labelpad`` left of its own tick labels, measured at the output's
    dpi), so a's N_e label stays where matplotlib sets it and the strip's
    unit joins it; both are mathtext, kept on it in the PDF and SVG by
    :func:`.style.on_baseline`. ``fig.align_ylabels`` cannot do this: it
    aligns only axes with a subplotspec, and :func:`.style.axes_mm` makes
    none.
    """
    from matplotlib.cbook import is_math_text
    from matplotlib.transforms import ScaledTranslation

    fig.draw_without_rendering()  # matplotlib's own placement: each label's box bottom at its x (display pixels)
    renderer = fig.canvas.get_renderer()
    base = 0.0  # points left of the spine
    for ax in axes:
        label = ax.yaxis.label
        if label.get_text():
            prop, s = label.get_fontproperties(), label.get_text()
            # The line's descent as matplotlib lays it out: its own, or at least that of 'lp'.
            descent = max(renderer.get_text_width_height_descent(s, prop, ismath=is_math_text(s))[2],
                          renderer.get_text_width_height_descent("lp", prop, ismath=False)[2])
            spine = ax.transAxes.transform((0.0, 0.0))[0]
            base = max(base, (spine - label.get_position()[0] + descent) * 72.0 / fig.dpi)
    for ax in axes:
        ax.yaxis.set_label_coords(0.0, 0.5, transform=ax.transAxes + ScaledTranslation(-base / 72.0, 0.0,
                                                                                          fig.dpi_scale_trans))
        ax.yaxis.label.set_verticalalignment("baseline")
        style.on_baseline(ax.yaxis.label)


def _columns(labels: Sequence[float], overhang: Sequence[float]) -> Tuple[List[float], float]:
    """Left spines of a-d and their one width, so that every gutter has :data:`CLEAR_MM` of clear white.

    ``labels[i]`` is how far column ``i``'s ink (its y tick labels) reaches
    left of its spine, ``overhang[i]`` how far it reaches right of its right
    edge (both mm). a's spine stays at :data:`L_MM` and d's right edge at
    :data:`RIGHT_EDGE_MM`.
    """
    w = (RIGHT_EDGE_MM - L_MM - 3 * CLEAR_MM - sum(labels[1:]) - sum(overhang[:3])) / 4.0
    cols = [L_MM]
    for i in range(1, 4):
        cols.append(cols[-1] + w + overhang[i - 1] + CLEAR_MM + labels[i])
    return cols, w


def _layout_top(fig, axs, strips, drawn: Sequence[bool]) -> Tuple[List[float], float]:
    """Place a-d and their strips with equal widths and equal clear white; return the left spines and a's ink.

    The gutters follow the drawn y tick labels of each panel and its strip
    (d's '1,000' is wider); an empty column keeps the nominal share. a's ink
    is how far it reaches left of a's spine (its y labels).
    """
    fw = fig.get_figwidth() / style.MM
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    labels, overhang = [], []
    for ax, st, shown in zip(axs, strips, drawn):
        if not shown:
            labels.append(GUT_MM - CLEAR_MM)
            overhang.append(0.0)
            continue
        left = right = 0.0
        for a in (ax, st):
            ink, pos = a.get_tightbbox(renderer), a.get_position()
            left = max(left, pos.x0 * fw - ink.x0 / fig.dpi / style.MM)
            right = max(right, ink.x1 / fig.dpi / style.MM - pos.x1 * fw)
        labels.append(left)
        overhang.append(right)
    cols, pw = _columns(labels, overhang)
    for ax, st, x in zip(axs, strips, cols):
        style.place_mm(ax, x, ROW_TOP_MM, pw, ROW_H_MM)
        style.place_mm(st, x, STRIP_TOP_MM, pw, STRIP_H_MM)
    return cols, labels[0]


# ---------------------------------------------------------------------------
# e
# ---------------------------------------------------------------------------


class Row(NamedTuple):
    """One strip of e: its genomes (``keys``), their values and jitter, and how it is drawn and named."""

    slot: int
    name: str
    keys: Tuple
    values: object  # numpy array, positive and finite (a log axis)
    jitter: object  # numpy array, one height per genome
    color: str
    marker: str = "o"
    size: float = DOT_S
    text: str = style.INK  # its name's colour


def _row(slot: int, name: str, keys: Sequence, values, jitter, color: str, **kw) -> Row:
    """The row of the genomes ``keys`` whose value is positive and finite, each keeping its jitter."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v) & (v > 0)
    return Row(slot, name, tuple(k for k, ok in zip(keys, keep) if ok), v[keep], np.asarray(jitter)[keep], color,
               **kw)


def _others_name(names: Sequence[Optional[str]], n: int) -> str:
    """The label of e's non-human row: the genomes' shared common name, plural unless ``n`` is 1 ('Turtles')."""
    common = {COMMON.get(s) if s else None for s in names}
    if len(common) == 1 and None not in common:
        name = common.pop()
        return name if n == 1 else name + "s"
    return "Other genome" if n == 1 else "Other genomes"


def _jitter(n: int):
    import numpy as np

    return np.random.default_rng(SEED).uniform(-style.STRIP_JITTER, style.STRIP_JITTER, n)


def _e_rows(rep, e1d, rsc7, delta2, species, floor) -> Tuple[List[Row], List[Row]]:
    """e's rows: Delta's, with the noise floor ``floor`` (:func:`noise_floor`) if any; the errors against the truth."""
    import numpy as np

    hum7 = list(rsc7.sample_id)
    in7 = set(hum7)
    humans = hum7 + [s for s in sorted(delta2) if species.get(s) == HUMAN and s not in in7]
    human_d = list(rsc7.rmse_log10_ne_rs_vs_c_per_interval) + [delta2[s] for s in humans[len(hum7):]]
    others = [s for s in sorted(delta2) if species.get(s) != HUMAN]
    sims = [(str(m), int(r)) for m, r in zip(e1d.model, e1d.replicate)]
    sim_jit = _jitter(len(sims))  # one height per simulated genome, in Simulated and in the noise floor
    turtles = _row(SLOT_TURTLE, "", others, [delta2[s] for s in others], _jitter(len(others)), style.INK2)
    diff_rows = [_row(SLOT_HUMAN, "Human", humans, human_d, _jitter(len(humans)), style.INK2),
                 turtles._replace(name=_others_name([species.get(s) for s in others], turtles.values.size)),
                 _row(SLOT_SIM, "Simulated", sims, e1d.rmse_log10_ne_rs_vs_c_per_interval, sim_jit, style.INK2)]
    if floor:
        diff_rows.append(_row(SLOT_FLOOR, FLOOR_NAME, sims, [floor.get(k, math.nan) for k in sims], sim_jit,
                              style.REF, text=style.INK2))

    err = {}
    for row in rep[rep.tool.isin(["rs", "c"])].itertuples():
        err.setdefault((row.model, row.replicate), {})[row.tool] = float(row.rmse_log10_ne_vs_truth)
    genomes = sorted(err)
    jit = _jitter(len(genomes))  # one height per simulated genome, shared by its two rows
    err_rows = [_row(slot, ent.label, genomes, np.array([err[k].get(tool, math.nan) for k in genomes]), jit,
                     ent.color, marker=ent.marker, size=DOT_S if ent.marker == "o" else SQUARE_S)
                for slot, ent, tool in ((SLOT_ERR_RS, style.RS02, "rs"), (SLOT_ERR_C, style.C_PSMC, "c"))]
    return diff_rows, err_rows


def _strips(ax, rows: Sequence[Row]) -> Dict[int, float]:
    """Draw e's strips, with a cased median bar on each row of at least :data:`MEDIAN_MIN`; the medians by slot."""
    import numpy as np

    medians: Dict[int, float] = {}
    for row in rows:
        if not row.values.size:
            continue
        ax.scatter(row.values, row.slot + row.jitter, s=row.size, marker=row.marker, color=row.color,
                   edgecolors=style.SURFACE, linewidths=style.STRIP_EDGE_LW, zorder=style.Z_CONTEXT + 1)
        if row.values.size >= MEDIAN_MIN:
            m = float(np.median(row.values))
            medians[row.slot] = m
            seg = ([m, m], [row.slot - style.STRIP_BAR_HALF, row.slot + style.STRIP_BAR_HALF])
            ax.plot(*seg, color=style.SURFACE, linewidth=CASING_LW, solid_capstyle="butt", zorder=style.Z_DATA)
            ax.plot(*seg, color=style.INK, linewidth=style.LW_DATA, solid_capstyle="butt", zorder=style.Z_TRUTH)
    return medians


def _rings(ax, rows: Sequence[Row], genomes: Dict[str, object]) -> None:
    """Ring each genome of a-d (``genomes``: its key by its panel's letter) where e draws it, named by the letter.

    Rings of one strip within :data:`RING_JOIN` decade of each other share
    one label, on the side :data:`RING_SIDE` gives the strip. Each ring has
    a white rim (:data:`RIM_LW`), so that it lies visibly on top of the
    dots, median bar and other ring it crosses.
    """
    from matplotlib.patheffects import withStroke

    rim = [withStroke(linewidth=RIM_LW, foreground=style.SURFACE)]
    found = []  # (row, letter, x, y), in the letters' order
    for letter, key in sorted(genomes.items()):
        hit = next(((row, i) for row in rows for i, k in enumerate(row.keys) if k == key), None)
        if hit is not None:  # found by key, not by position: a row drops non-positive values
            row, i = hit
            x, y = float(row.values[i]), row.slot + float(row.jitter[i])
            ax.scatter([x], [y], **style.ring_kw(path_effects=rim))
            found.append((row, letter, x, y))
    groups: List[list] = []  # [row, label, xs, ys]
    for row, letter, x, y in found:
        if groups and groups[-1][0] is row and abs(math.log10(x / groups[-1][2][-1])) < RING_JOIN:
            groups[-1][1] += ", " + letter
            groups[-1][2].append(x)
            groups[-1][3].append(y)
        else:
            groups.append([row, letter, [x], [y]])
    for row, text, xs, ys in groups:
        side = RING_SIDE.get(row.slot, "above")
        if side == "right":  # right of the ring furthest right, its capitals centred on the rings' mean height
            xy, dx, dy = (max(xs), sum(ys) / len(ys)), RING_R_PT + RING_GAP_PT, -CAP_EM * style.LABEL_PT / 2.0
            ha, va = "left", "baseline"
        else:  # centred on the rings, clear of them and of the strip's median bar (the y axis runs down)
            up = side == "above"
            reach = max(((row.slot - y) if up else (y - row.slot)) * SLOT_PT + RING_R_PT for y in ys)
            if row.values.size >= MEDIAN_MIN:
                reach = max(reach, style.STRIP_BAR_HALF * SLOT_PT)
            xy, dx, ha = (math.prod(xs) ** (1.0 / len(xs)), row.slot), 0.0, "center"
            # above: the descent (a comma's tail) clears them; below: an ascender's top does
            dy, va = ((reach + RING_GAP_PT, "bottom") if up else
                      (-(reach + RING_GAP_PT + CAP_EM * style.LABEL_PT), "baseline"))
        ax.annotate(text, xy=xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=style.LABEL_PT, color=style.INK, zorder=style.Z_TEXT)


def _on_strip(fig, ax, size: float):
    """x in figure and y in ``ax``'s data coordinates, moved down so that ``size`` text set on its baseline has
    its capitals centred on the strip."""
    from matplotlib.transforms import blended_transform_factory, offset_copy

    return offset_copy(blended_transform_factory(fig.transFigure, ax.transData), fig=fig, y=-CAP_EM * size / 2.0,
                       units="points")


def _forest_labels(fig, ax, blocks, head_x: float, title_x: float) -> List:
    """Each block's header (7 pt, from ``title_x``) and its rows' names with their n (6 pt, right-aligned).

    ``blocks`` holds ``(rows, header slot, header)``. Returns the row names,
    set at ``head_x`` until the label column's width is known.
    """
    fw = fig.get_figwidth() / style.MM
    names = []
    for rows, head_slot, head in blocks:
        if not any(row.values.size for row in rows):
            continue
        ax.text(title_x / fw, head_slot, head, transform=_on_strip(fig, ax, style.TEXT_PT), ha="left",
                va="baseline", fontsize=style.TEXT_PT, color=style.INK, zorder=style.Z_TEXT)
        for row in rows:
            if row.values.size:
                names.append(ax.text(head_x / fw, row.slot, f"{row.name} ({row.values.size})",
                                     transform=_on_strip(fig, ax, style.LABEL_PT), ha="right", va="baseline",
                                     fontsize=style.LABEL_PT, color=row.text, zorder=style.Z_TEXT))
    return names


def _fold_arrow(ax, dif: float, err: float, fold: float) -> None:
    """The arrow from Simulated's median Delta (``dif``) to psmc-rs's median error, labelled with ``fold``."""
    # The leaders (style.LEADER_KW) above the points, so that one they cross (the noise floor's, under Simulated) does
    # not cut them.
    leader = {**style.LEADER_KW, "zorder": style.Z_DATA}
    ax.plot([dif, dif], [SLOT_SIM + style.STRIP_BAR_HALF, SLOT_ARROW], **leader)
    ax.plot([err, err], [SLOT_ERR_RS - style.STRIP_BAR_HALF, SLOT_ARROW], **leader)
    ax.annotate("", xy=(dif, SLOT_ARROW), xytext=(err, SLOT_ARROW), arrowprops=dict(style.ARROW_PROPS),
                zorder=style.Z_DATA)
    if fold > 0 and math.isfinite(fold):
        ax.text(math.sqrt(dif * err), SLOT_ARROW - 0.15, f"{round_fold(fold):,.0f}{style.TIMES} smaller",
                ha="center", va="bottom", zorder=style.Z_TEXT, **style.FOLD_HEADLINE)


def draw(src: Source):
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullFormatter, NullLocator

    prof = src.profile
    e1c = curves(src.table("E1_curves"), ["model", "tool", "replicate"])
    win1 = table_windows(src.table("E1_table1"), "model")
    rep = src.table("E1_replicates")
    e1d = src.table("E1_rs_vs_c")
    e2c = curves(src.table("E2_curves"), ["sample_id", "tool"])
    bands = src.table("E2_bands")
    rsc2 = src.table("E2_rs_vs_c")
    rsc7 = src.table("E7_rs_vs_c")
    win2 = table_windows(rsc2, "sample_id")
    delta2 = {r.sample_id: float(r.rmse_log10_ne_rs_vs_c_per_interval) for r in rsc2.itertuples()}
    species = {g.sample_id: g.species for g in prof.e2.genomes}
    floor = noise_floor(e1d)

    # ------------------------------------------------------------------ a-d and their strips
    panels = _top_panels(prof, e1c, win1, e2c, win2, delta2, species)
    fig = style.figure("double", H_MM, dpi=style.PNG_DPI)  # laid out at the output's dpi: text measures as printed
    axs = [style.axes_mm(fig, COLS[i], ROW_TOP_MM, PW_MM, ROW_H_MM) for i in range(4)]  # moved by _layout_top
    strips = [style.axes_mm(fig, COLS[i], STRIP_TOP_MM, PW_MM, STRIP_H_MM) for i in range(4)]
    shown = [p is not None for p in panels]
    letters = dict(zip([i for i in range(4) if shown[i]], "abcd"))  # the letters of the drawn panels run on
    shared: Tuple[List[float], List[float], List[float]] = ([], [], [])  # a-c: one set of limits from all they draw
    own: Dict[int, Tuple[List[float], List[float], List[float]]] = {}
    titles: Dict[int, str] = {}
    genomes: Dict[str, object] = {}  # each drawn panel's genome, by its letter: e rings them
    banded = False
    for i, (ax, st, panel) in enumerate(zip(axs, strips, panels)):
        if panel is None:
            ax.set_axis_off()
            st.set_axis_off()
            continue
        kind, key = panel
        if kind == "sim":
            w, r = win1[key], _rep_of(e1c, key)
            rs, c = e1c[(key, "rs", r)], e1c[(key, "c", r)]
            ys = _sim_panel(ax, key, r, e1c, w, e1d, prof.e1.scaling.gen_years)
            titles[i] = SIM_TITLES.get(key, f"Simulated {key}")
            genomes[letters[i]] = (str(key), int(r))
        else:
            w = win2[key]
            rs, c = e2c[(key, "rs")], e2c[(key, "c")]
            band = bands[(bands.sample_id == key) & (bands.trunk_bins == prof.e2.bootstrap_trunk_bins)]
            ys, has_band = _real_panel(ax, rs, c, w, band, delta2.get(key))
            banded |= has_band
            titles[i] = _title_of(key, species.get(key))
            genomes[letters[i]] = key
        held = (list(w), ys, _diff_strip(st, rs, c, w))  # what the limits hold: times, Ne, differences
        if i < 3:
            for into, values in zip(shared, held):
                into += values
        else:
            own[i] = held
    lims = {i: (_limits(XL_SIM, shared[0]), _limits(YL_SIM, shared[1], PAD), _strip_limit(STRIP_LIM[0], shared[2]))
            for i in range(3)}
    lims.update({i: (_limits(XL_REAL2, w), _limits(YL_REAL2, ys, PAD), _strip_limit(STRIP_LIM[1], ds))
                 for i, (w, ys, ds) in own.items()})
    for i, (ax, st) in enumerate(zip(axs, strips)):
        if not shown[i]:
            continue
        style.thousands_axis(ax, "x", lims[i][0])
        ax.xaxis.set_major_formatter(NullFormatter())  # the strip below carries the times
        style.thousands_axis(ax, "y", lims[i][1])
        _strip_axes(st, lims[i][0], lims[i][2])
    if shown[0]:  # a's y labels, one column: Ne, and the strips' unit (before the layout measures a's ink)
        axs[0].set_ylabel(style.NE_LABEL)
        strips[0].set_ylabel(r"$\times10^{%d}$" % round(math.log10(STRIP_UNIT)))
        _pin_ylabels(fig, (axs[0], strips[0]))
    cols, a_ink = _layout_top(fig, axs, strips, shown)
    for i in sorted(titles):  # on the letters' baseline, d's italic binomial (mathtext) too
        style.on_baseline(style.fig_text(fig, cols[i], TITLE_BASE_MM, titles[i], ha="left", va="baseline"))

    # ------------------------------------------------------------------ e: the difference against the error
    head_x = cols[0] - a_ink  # e's label column starts at a's ink (its y labels), the figure's left edge of ink
    ax_e = style.axes_mm(fig, cols[1], E_TOP_MM, RIGHT_EDGE_MM - cols[1], E_H_MM)  # moved once names are measured
    diff_rows, err_rows = _e_rows(rep, e1d, rsc7, delta2, species, floor)
    medians = _strips(ax_e, diff_rows + err_rows)
    # The headers start where a-d's titles do, at a's spine; the first is the Delta of a-d.
    names = _forest_labels(fig, ax_e, ((diff_rows, SLOT_DIFF_HEAD, DIFF_HEAD), (err_rows, SLOT_ERR_HEAD, ERR_HEAD)),
                           head_x, cols[0])
    # A forest plot: the label column is as wide as the widest row name, the axes take the rest of the width.
    fw = fig.get_figwidth() / style.MM
    renderer = fig.canvas.get_renderer()
    col_w = max((t.get_window_extent(renderer).width / fig.dpi / style.MM for t in names), default=0.0)
    e_x = head_x + col_w + E_LABEL_GAP_MM
    style.place_mm(ax_e, e_x, E_TOP_MM, RIGHT_EDGE_MM - e_x, E_H_MM)
    for t in names:
        t.set_x((e_x - E_LABEL_GAP_MM) / fw)
    if SLOT_SIM in medians and SLOT_ERR_RS in medians:  # the ratio of the medians the arrow joins
        _fold_arrow(ax_e, medians[SLOT_SIM], medians[SLOT_ERR_RS], fold_median(rep, e1d))
    _rings(ax_e, [row for row in diff_rows if row.slot != SLOT_FLOOR], genomes)
    style.log_axis(ax_e, "x", fmt="pow10", lim=E_XLIM)
    ax_e.xaxis.set_minor_locator(NullLocator())
    ax_e.set_ylim(SLOT_ERR_C + 0.5, SLOT_DIFF_HEAD - 0.5)
    ax_e.set_yticks([])
    ax_e.spines["left"].set_visible(False)
    ax_e.set_xlabel("RMSE of " + style.LOGNE)

    # ------------------------------------------------------------------ key, x label, letters
    handles = [style.handle(style.TRUTH, linestyle=style.centred_dash(style.DASH)),
               style.handle(style.C_PSMC, "underlay"), style.handle(style.RS02)]
    if banded:
        handles.append(Patch(facecolor=style.RS_BAND, linewidth=0.0, label="Bootstrap 95% band"))
    leg = style.top_legend(fig, handles, x_mm=L_MM)
    if any(shown):
        xlab = style.row_xlabel(fig, [st for st, s in zip(strips, shown) if s])
        if (1.0 - xlab.get_position()[1]) * H_MM < XLAB_TOP_MM:  # a little clear of the tick labels
            xlab.set_y(1.0 - XLAB_TOP_MM / H_MM)
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    off = leg.get_window_extent(renderer).x0 / fig.dpi / style.MM - L_MM
    if abs(off) > 0.05:  # the first handle starts right above a's left spine
        (x, y), _ = leg.get_bbox_to_anchor().get_points()  # pixels
        leg.set_bbox_to_anchor((x / fig.bbox.width - off / fw, y / fig.bbox.height), transform=fig.transFigure)
    for i, letter in letters.items():  # a at the house's x, the others at their column's ink
        x_mm = style.LETTER_X_MM if i == 0 else min(style.ink_left_mm(a, renderer) for a in (axs[i], strips[i]))
        style.panel_letter(fig, x_mm, TITLE_BASE_MM, letter)
    fig.text(style.LETTER_X_MM / fw, SLOT_DIFF_HEAD, "abcde"[len(letters)],
             transform=_on_strip(fig, ax_e, style.TEXT_PT), ha="left", va="baseline", fontsize=style.PANEL_PT,
             fontweight="bold", color=style.INK)  # on the first header's baseline
    return fig


__all__ = ["agreement_intervals", "draw", "fold_median", "noise_floor", "paired_intervals", "round_fold"]
