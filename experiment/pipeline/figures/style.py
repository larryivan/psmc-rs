"""The house style of every manuscript figure: type, sizes, colours, marks, layout and export.

Every figure module draws through this module, so the six figures of the
article and its supplement read as one set. The style is a **closed
vocabulary**: anything a figure needs that is not named here needs a
written reason beside its constant here.

The figures are designed and checked at their print size, 183 mm (the
manuscript places them at 100%, never scaled).

* **Type.** Arial, else Helvetica, else DejaVu Sans, in **three type
  sizes**: 8 pt bold lowercase panel letters (:data:`PANEL_PT`); 7 pt
  regular (:data:`TEXT_PT`) for axis labels, panel titles, legend text and
  group headers; and 6 pt regular for tick labels (:data:`TICK_PT`), for
  every number or name printed on the data (:data:`LABEL_PT`: bar values,
  line-end labels, fold changes, population codes), in :data:`INK`, and
  for context notes and the names of reference lines (:data:`NOTE_PT`), in
  :data:`INK2`. Tick labels are dark (:data:`INK`), never grey.
  Superscripts and subscripts are set at :data:`MATH_SHRINK` (0.84) of
  their text, not mathtext's 0.7, so a 6 pt label's exponent is 5.04 pt and
  a 7 pt N_e's subscript 5.9 pt: **no text is below :data:`MIN_TEXT_PT`
  (5 pt)**, the journal's minimum (:func:`small_text` finds any in a
  written SVG). PDF and SVG keep text as text (TrueType, fonttype 42; SVG
  ``svg.fonttype = none``).
* **Notation.** One spelling per quantity, matching the article: italic
  variables through :data:`NE`, :data:`LOGNE` and :data:`THETA`; U+2212
  for every minus (:data:`MINUS`), U+00D7 for times and fold changes
  written ``8.6×`` (:data:`TIMES`), U+2013 for ranges (:data:`ENDASH`);
  units in the axis label, never in tick labels.
* **Log axes.** Sizes and times are plain numbers in thousands
  (:func:`ne_axis`, :func:`years_axis`, :func:`thousands_axis`): ticks 1,
  10, 100, 1,000 under the labels :data:`NE_LABEL` and :data:`YEARS_LABEL`;
  an axis spanning less than :data:`SPAN_125` decades, or holding fewer than
  two powers of ten, is labelled at 1, 2 and 5 (2, 5, 10, 20;
  :func:`narrow_log_span`). Minor ticks 2-9 are unlabelled. ``10`` with a
  superscript (:func:`log_axis`'s default) is kept for dimensionless
  quantities spanning more than four decades only.
* **Neutral greys by role** (R = G = B; colour belongs to the data):
  :data:`INK` text, :data:`AXIS` spines and ticks, :data:`INK2` secondary
  notes, :data:`REF` reference lines (zero, y = x, nominal 95%, perfect
  scaling), :data:`CONTEXT` context curves, :data:`FILL` the one neutral
  band, :data:`GRID` hairlines. :data:`UNDERLAY_GREY` is the underlay of a
  neutral reference (see the underlay rule).
* **Line weights.** :data:`LW_HAIR` 0.35 pt (minor ticks, leaders, grid,
  the edges of a second interval); :data:`LW_THIN` 0.5 pt (axes, reference
  lines, replicate and context curves); :data:`LW_DATA` 0.75 pt (every data
  line, the truth included); :data:`LW_EMPH` 1.0 pt (at most one emphasised
  summary per panel); :data:`UNDERLAY_LW` 2.4 pt (a pale underlay only).
  Nothing else. Hierarchy, heaviest first: the message series, the
  comparison or reference series, annotations, axes, grid.
* **The underlay rule.** Where two curves coincide, the reference is a
  wide pale underlay (:func:`underlay_kw`: :data:`UNDERLAY_LW` in a tint of
  its own colour, :data:`C_HALO` for C psmc, :data:`UNDERLAY_SHARE` of the
  colour otherwise) and the tested curve a thin saturated line on top
  (:func:`overlay_kw`). Never a dashed line over a solid one of similar
  weight. Legend keys use exactly the drawn widths and colours
  (:func:`handle` with ``kind="underlay"``).
* **Dashes.** Two patterns: :data:`DASH` for the truth (:func:`truth_kw`:
  0.75 pt :data:`INK`, drawn on top) and the admixed populations of the
  1000 Genomes figure, and :data:`DOT` for leaders and secondary reference
  lines. matplotlib scales a pattern by the line's width.
* **Bands.** Opaque tints only, never alpha (alpha blends to violet under
  the orange underlay and makes legend keys that differ from the fill):
  :data:`RS_BAND` for psmc-rs's band (:func:`band_kw`), :data:`FILL` for a
  neutral one, at most one filled band per panel; a second interval is
  drawn as :data:`LW_HAIR` edges in its line's colour (:func:`edge_kw`).
  The legend key of a band is the plotted fill itself.
* **Grids.** None on N_e(t) or other time-series panels (every helper's
  default). :data:`GRID` hairlines on the value axis only where values are
  read off (bars and timings).
* **Markers.** psmc-rs a filled circle, C psmc a filled square, anything
  else a filled circle; no edge and no alpha. :data:`MARKER_MS` (3.0 pt) on
  line-and-marker plots, :data:`STRIP_MS` (2.4 pt) in strip plots.
  Continuous lines never carry markers. Strip plots: jitter
  ±:data:`STRIP_JITTER` of a category, a :data:`LW_DATA` :data:`INK`
  summary bar ±:data:`STRIP_BAR_HALF`, differences in :data:`INK2`.
* **Layout** (millimetres, shared by all six figures): outer ink margin
  :data:`MARGIN_MM`; first left spine at :data:`LEFT_MM`; right margin
  :data:`RIGHT_MM`; column gutters :data:`GUTTER_MM` (each panel has y tick
  labels) or :data:`GUTTER_SHARED_MM` (y shared and hidden); row gaps
  :data:`ROW_GAP_MM` under x tick labels, :data:`ROW_GAP_TITLED_MM` under an
  x label (letters and titles sit above every row); N_e(t) panels between
  :data:`NE_ASPECT` w:h. Within a row the x and y limits are identical, on
  round 1-2-5 values, with one axis label per row (:func:`row_xlabel`).
* **Letters, titles, legends.** Every first-column letter sits at
  :data:`LETTER_X_MM` from the figure's edge, the others at the left edge of
  their panel's ink, aligned in columns (:func:`panel_letters`). A letter
  shares its baseline with its panel's title, :data:`TITLE_RISE_MM` above
  the axes (:func:`panel_title`: 7 pt :data:`INK`, left-aligned with the
  left spine). A key, when the data cannot be labelled directly, is one
  line at the figure's top, left-aligned with the first left spine
  (:func:`top_legend`), at most four words per entry, no numbers and no
  panel references (the caption holds those).
* **Colour follows the entity in every figure** (validated with
  ``dataviz/scripts/validate_palette.js``; contrast on white in brackets):

  ===============  =========  =================================================
  psmc-rs          #2a78d6    blue (4.4:1); its band :data:`RS_BAND` #dde9f8
  C psmc           #eb6834    orange (3.2:1); its underlay :data:`C_HALO`
                              #f6bba4
  truth            #1a1a1a    :data:`INK`, dashed
  ===============  =========  =================================================

  ``validate_palette.js "#2a78d6,#eb6834" --mode light --pairs all
  --surface "#ffffff"`` passes. Blue and orange are reserved for the two
  programs. A psmc-rs variant takes the next step of the blue ramp, 650
  #104281 (:data:`BLUE_650`; ``"#2a78d6,#104281" --ordinal`` passes);
  magnitudes use the sequential ramp :data:`BLUE_RAMP`. Supplement Fig S2's
  routes (:data:`ROUTES`): psmc-rs's default, weighted bins, is psmc-rs's
  blue; every called bin taken as full by fq2psmcfa is C psmc's orange,
  because it is C psmc's input and model; the routes not drawn in the
  article keep colours of at least 3:1 (:data:`OCHRE` #b07800 3.8:1,
  :data:`RASPBERRY` #c94f80 4.3:1), so colour carries them without
  markers. The 1000 Genomes superpopulations (:data:`SUPERPOP_COLORS`) take
  five hues apart from blue and orange (South Asia sienna #a0522d, not a
  second green); :data:`SUPERPOP_TEXT` are their text colours.
  **One exception to ink-coloured text:** a figure without a legend may
  name a series in the series' text colour (:data:`ORANGE_TEXT`,
  :data:`BLUE_TEXT`, :data:`SUPERPOP_TEXT`: the mark colour, or a darker
  step of it where the mark is below 4.3:1 on white), because there the
  colour is the key.

  psmc-rs 0.1 (:data:`RS01`, :data:`AQUA`), band alpha
  (:data:`BAND_ALPHA`), outside-window shading (:func:`shade_outside`) and
  the former names kept beside the house constants serve the pipeline-only
  modules (fig1-fig4, figs2-figs6 and figs_turtle, which are not in the
  paper); no article figure uses them.

Export (:func:`save`) writes ``<name>.pdf``, ``.svg`` and a 600-dpi ``.png``
without creation dates, so drawing the same tables twice gives the same
bytes.
"""

from __future__ import annotations

import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

MM = 1.0 / 25.4  # inches per millimetre
PT_PER_MM = 72.0 / 25.4
WIDTH_MM = {"single": 89.0, "onehalf": 120.0, "double": 183.0}

# ---------------------------------------------------------------------------
# Type: three sizes
# ---------------------------------------------------------------------------

FONT_FAMILY = ("Arial", "Helvetica", "DejaVu Sans")
PANEL_PT = 8.0  # bold lowercase panel letters
TEXT_PT = 7.0  # axis labels, panel titles, legend text, group headers
TICK_PT = 6.0  # tick labels (and category labels set as tick labels)
LABEL_PT = 6.0  # a number or name printed on the data, in INK
NOTE_PT = 6.0  # a context note (condition, n) or a reference line's name, in INK2
MIN_TEXT_PT = 5.0  # the smallest text of any figure (the journal's minimum)
# Size of a superscript or subscript relative to its text (mathtext's own is
# 0.7: a 6 pt tick label's exponent would be 4.2 pt). matplotlib is pinned
# (experiment/pipeline/requirements.lock), and :func:`apply` checks that the
# private constant it sets exists.
MATH_SHRINK = 0.84

# Notation: one spelling per quantity, italic variables as in the article.
NE = r"$\mathit{N}_\mathrm{e}$"
LOGNE = r"$\log_{10}\,\mathit{N}_\mathrm{e}$"
THETA = r"$\mathit{\theta}$"
MINUS = "−"
TIMES = "×"
ENDASH = "–"
NE_LABEL = NE + " (thousands)"  # every N_e(t) size axis (ne_axis)
YEARS_LABEL = "Thousands of years ago"  # every N_e(t) time axis (years_axis)
NE_LABEL_SHORT = NE_LABEL  # former short label; an N_e axis is always labelled in thousands

# ---------------------------------------------------------------------------
# Line weights: four weights and one underlay width
# ---------------------------------------------------------------------------

LW_HAIR = 0.35  # minor ticks, leaders, grid, the edges of a second interval
LW_THIN = 0.5  # axes, reference lines, replicate and context curves
LW_DATA = 0.75  # every data line, the truth included
LW_EMPH = 1.0  # at most one emphasised summary per panel
UNDERLAY_LW = 2.4  # the pale underlay of a reference curve (the underlay rule)
LW_UNDERLAY = UNDERLAY_LW

# Former names, kept for the pipeline-only figures (they now take the house weights).
LW = LW_DATA
LW_BOLD = LW_EMPH
LW_AXIS = LW_THIN
LW_GRID = LW_HAIR
LW_REF = LW_THIN

# Drawing order: bands under references under context under the underlay
# under the data under the truth under text.
Z_BAND = 1.0
Z_REF = 1.5
Z_CONTEXT = 2.0
Z_UNDERLAY = 3.0
Z_DATA = 4.0
Z_TRUTH = 5.0
Z_TEXT = 6.0

# ---------------------------------------------------------------------------
# Neutral greys by role (R = G = B)
# ---------------------------------------------------------------------------

SURFACE = "#ffffff"
INK = "#1a1a1a"  # all text, tick labels included; the truth
AXIS = "#404040"  # spines and ticks
INK2 = "#595959"  # secondary notes; differences in strip plots
REF = "#8c8c8c"  # reference lines (y = 0, y = x, nominal 95%, perfect scaling), leaders
CONTEXT = "#c4c4c4"  # context curves (e.g. the other populations)
FILL = "#e8e8e8"  # the one neutral band
GRID = "#e6e6e6"  # hairline grid, where values are read off
# The underlay of a neutral reference curve (e.g. complete data): black at
# 15% (tint("#000000", 0.15)). A line, unlike a band, needs one step more than
# FILL to stay visible at 2.4 pt beside coloured lines.
UNDERLAY_GREY = "#d9d9d9"

# Former names (pipeline-only figures): the reference grey and the neutral band.
MUTED = REF
NEUTRAL = FILL

# ---------------------------------------------------------------------------
# Dashes, bands, markers
# ---------------------------------------------------------------------------

DASH = (0, (2.5, 1.5))  # the truth (and admixed populations)
DOT = (0, (0.8, 1.2))  # leaders and secondary reference lines

BAND_SHARE = 0.16  # an entity's band: its colour at 16% over white, opaque
UNDERLAY_SHARE = 0.30  # the underlay of a reference variant: its colour at 30%
C_HALO_SHARE = 0.45  # C psmc's underlay: orange at 30% no longer reads as orange
REPLICATE_ALPHA = 0.35  # thin replicate lines in an entity's colour (see tint), opaque
BAND_ALPHA = 0.18  # former alpha of bands, pipeline-only figures; article figures use band_kw

MARKER_MS = 3.0  # markers of a line-and-marker plot (e.g. timings)
STRIP_MS = 2.4  # points of a strip plot
MEDIAN_MS = MARKER_MS  # a median marker over its replicates
# Replicates behind a median marker: below the closed set (3.0, 2.4) because
# replicates sit behind a median marker and must read as subordinate.
REPLICATE_MS = 1.6
STRIP_JITTER = 0.15  # vertical (or horizontal) jitter of a strip, in categories
STRIP_BAR_HALF = 0.3  # half-length of a strip's summary bar, in categories

# A double-headed arrow marking a gap (e.g. the fold change between two medians).
ARROW_PROPS = {"arrowstyle": "<|-|>", "linewidth": LW_THIN, "color": AXIS, "mutation_scale": 5.0,
               "shrinkA": 0.0, "shrinkB": 0.0}

PNG_DPI = 600

# ---------------------------------------------------------------------------
# Layout (millimetres)
# ---------------------------------------------------------------------------

MARGIN_MM = 1.0  # outer ink margin on every side
LEFT_MM = 11.0  # the first left spine (fits a 6 pt '1,000' and a 7 pt y label)
RIGHT_MM = 1.5  # right edge of the last axes to the figure's edge
# Measured at these rc settings: a letter (and a title) reaches 3.55 mm above
# its axes, x tick labels 3.6 mm below them and an x label 6.25 mm below them.
TOP_MM = 4.5  # figure top to the first axes top: 1 mm margin, letter, title rise
TOP_LEGEND_MM = 8.5  # the same with a one-line key (1.0-3.3 mm) above the letters
BOTTOM_MM = 7.25  # last axes bottom to the figure bottom: x ticks, x label, 1 mm margin
GUTTER_MM = 6.0  # between columns whose panels each show y tick labels
GUTTER_SHARED_MM = 3.0  # between columns that share a hidden y axis
ROW_GAP_MM = 9.0  # between rows: x tick labels above, letters (and titles) below
ROW_GAP_TITLED_MM = 12.0  # between rows: x tick labels and an x label above, letters and titles below
LETTER_X_MM = 1.0  # every first-column panel letter, from the figure's left edge
TITLE_RISE_MM = 1.5  # letter and title baseline above the axes top
LETTER_TITLE_GAP_MM = 1.0  # least space between a letter and its panel's title
NE_ASPECT = (1.3, 1.8)  # width : height of an N_e(t) panel
SPAN_125 = 1.5  # decades below which a plain log axis is labelled at 1, 2 and 5


@dataclass(frozen=True)
class Entity:
    """How one thing that recurs across figures (a tool, the truth, a route) is drawn.

    ``halo`` is the share of its colour its underlay takes (the underlay rule).
    """

    label: str
    color: str
    linestyle: Union[str, Tuple] = "-"
    marker: str = "o"
    filled: bool = True
    halo: float = UNDERLAY_SHARE

    def line(self, **kw) -> Dict:
        """Keyword arguments of ``ax.plot`` for this entity's line (:data:`LW_DATA`)."""
        out = {"color": self.color, "linestyle": self.linestyle, "linewidth": LW_DATA, "label": self.label}
        out.update(kw)
        return out

    def points(self, **kw) -> Dict:
        """Keyword arguments of ``ax.plot`` for this entity's markers alone (no edge when filled)."""
        out = {
            "color": self.color,
            "linestyle": "none",
            "marker": self.marker,
            "markersize": MARKER_MS,
            "markeredgewidth": 0.0 if self.filled else LW_THIN,
            "markerfacecolor": self.color if self.filled else SURFACE,
            "markeredgecolor": self.color,
            "label": self.label,
        }
        out.update(kw)
        return out


# Palette (dataviz reference palette, light mode).
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_650 = "#104281"
# Sequential blue (magnitude, e.g. counts of a 2-D histogram), steps 100-700 of the ramp.
BLUE_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b")
OCHRE = "#b07800"  # multihetsep (3.8:1 on white)
RASPBERRY = "#c94f80"  # a VCF read without its mask (4.3:1)
# Former slots, pipeline-only: psmc-rs 0.1's aqua and the pale yellow and pink.
AQUA, YELLOW, MAGENTA = "#1baf7a", "#eda100", "#e87ba4"

# Text colours of entities named in place (darker steps of the mark colours).
ORANGE_TEXT = "#c4501f"  # C psmc (4.65:1 on white; the mark is 3.2:1)
BLUE_TEXT = BLUE_RAMP[9]  # psmc-rs, #1c5cab (6.6:1)

RS02 = Entity("psmc-rs", BLUE, "-", "o")
C_PSMC = Entity("C psmc", ORANGE, "-", "s", halo=C_HALO_SHARE)
TRUTH = Entity("Truth", INK, DASH, "none")
# psmc-rs 0.1: pipeline-only (fig4.py and figs4.py); no article figure may use it.
RS01 = Entity("psmc-rs 0.1", AQUA, "-", "D")

# Tools by the keys the tables use.
TOOLS: Dict[str, Entity] = {"rs": RS02, "rs02": RS02, "c": C_PSMC, "rs01": RS01}

# Supplement Fig S2 (fig5.py, E4): routes from a VCF and a callable mask to
# psmc-rs. psmc-rs's default, its callable-length model on the VCF with its
# mask, is psmc-rs's blue; the consensus FASTA through fq2psmcfa, every
# called bin taken as full (C psmc's input and model), is C psmc's orange;
# the same bins from the VCF with --bin-callable ignore are psmc-rs's darker
# variant. Labels are the direct labels' wording; the identities of the
# routes go in the caption.
ROUTES: Dict[str, Entity] = {
    "mask_auto": Entity("Weighted bins (psmc-rs default)", BLUE, "-", "o"),
    "mask_ignore": Entity("Bins taken as full", BLUE_650, "-", "o"),
    "fq2psmcfa": Entity("Bins taken as full (fq2psmcfa, C psmc)", ORANGE, "-", "o", halo=C_HALO_SHARE),
    "multihetsep": Entity("multihetsep", OCHRE, "-", "o"),
    "nomask": Entity("VCF, no mask", RASPBERRY, "-", "o"),
}
ROUTE_ORDER = ("mask_auto", "mask_ignore", "fq2psmcfa", "multihetsep", "nomask")

# The same genome in three formats (figs2.py, pipeline-only).
FORMATS: Dict[str, Entity] = {
    "psmcfa": Entity("psmcfa", BLUE, "-", "o"),
    "vcf": Entity("VCF", BLUE_650, DOT, "o", filled=False),
    "multihetsep": Entity("multihetsep", OCHRE, "-", "o"),
}
FORMAT_ORDER = ("psmcfa", "vcf", "multihetsep")

# Fig 3 (fig7.py, E7): the 1000 Genomes superpopulations, five hues apart
# from the programs' blue and orange (South Asia sienna, not a second green),
# and the text colours their names are set in (at least 4.3:1 on white).
SUPERPOP_COLORS: Dict[str, str] = {"AFR": "#eda100", "AMR": "#12966b", "EAS": "#4a3aa7", "EUR": "#e87ba4",
                                   "SAS": "#a0522d"}
SUPERPOP_TEXT: Dict[str, str] = {"AFR": "#a36e09", "AMR": "#00865d", "EAS": "#4a3aa7", "EUR": "#bb537d",
                                 "SAS": "#a0522d"}

# Supplement Fig S3 (fig6.py, E5): bootstrap block sizes, two steps of the
# blue ramp (the figure orders its arms). Variants are never dashed:
# TRUNK_STYLES, both solid, is kept only while fig6.py still reads it.
TRUNK_COLORS = (BLUE, BLUE_RAMP[3])
TRUNK_STYLES = ("-", "-")


def tint(color: str, share: float = REPLICATE_ALPHA) -> str:
    """``color`` at opacity ``share`` over the white surface, as an opaque colour.

    Lines and bands drawn in such a tint do not blend where two entities
    coincide (half-transparent orange and blue make a grey violet that is
    neither), and their legend key shows the colour drawn.
    """
    from matplotlib.colors import to_hex, to_rgb

    c, s = to_rgb(color), to_rgb(SURFACE)
    return to_hex(tuple(share * a + (1.0 - share) * b for a, b in zip(c, s)))


C_HALO = tint(ORANGE, C_HALO_SHARE)  # #f6bba4: C psmc's underlay
RS_BAND = tint(BLUE, BAND_SHARE)  # #dde9f8: psmc-rs's band


def blue_cmap():
    """The sequential blue ramp as a colormap, light (few) to dark (many)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("psmc_blue", BLUE_RAMP)


# ---------------------------------------------------------------------------
# Marks: the underlay rule, the truth, bands
# ---------------------------------------------------------------------------


def _color_share(what: Union[Entity, str], share: Optional[float]) -> Tuple[str, float]:
    if isinstance(what, Entity):
        return what.color, what.halo if share is None else share
    return what, UNDERLAY_SHARE if share is None else share


def underlay_kw(what: Union[Entity, str], share: Optional[float] = None, **kw) -> Dict:
    """``ax.plot`` arguments of the reference of two coinciding curves: wide, pale, beneath.

    ``what`` is an entity (its ``halo`` share: :data:`C_HALO` for C psmc) or
    a colour (:data:`UNDERLAY_SHARE`); ``share`` overrides either.
    """
    color, s = _color_share(what, share)
    out = {"color": tint(color, s), "linewidth": UNDERLAY_LW, "linestyle": "-", "solid_capstyle": "butt",
           "solid_joinstyle": "miter", "zorder": Z_UNDERLAY}
    if isinstance(what, Entity):
        out["label"] = what.label
    out.update(kw)
    return out


def overlay_kw(what: Union[Entity, str], **kw) -> Dict:
    """``ax.plot`` arguments of the tested one of two coinciding curves: thin, saturated, on top."""
    color = what.color if isinstance(what, Entity) else what
    out = {"color": color, "linewidth": LW_DATA, "linestyle": "-", "solid_capstyle": "butt",
           "solid_joinstyle": "miter", "zorder": Z_DATA}
    if isinstance(what, Entity):
        out["label"] = what.label
    out.update(kw)
    return out


def truth_kw(**kw) -> Dict:
    """``ax.plot`` arguments of the truth: 0.75 pt ink, :data:`DASH`, on top of the fits."""
    out = {"color": INK, "linewidth": LW_DATA, "linestyle": DASH, "dash_capstyle": "butt", "zorder": Z_TRUTH,
           "label": TRUTH.label}
    out.update(kw)
    return out


def band_kw(color: str = BLUE, share: float = BAND_SHARE, **kw) -> Dict:
    """``fill_between`` arguments of an entity's band: an opaque tint (:data:`RS_BAND` for psmc-rs)."""
    out = {"color": tint(color, share), "linewidth": 0.0, "zorder": Z_BAND}
    out.update(kw)
    return out


def edge_kw(color: str, **kw) -> Dict:
    """``ax.plot`` arguments of the edges of a second interval in one panel (hairlines in its line colour)."""
    out = {"color": color, "linewidth": LW_HAIR, "linestyle": "-", "zorder": Z_BAND + 0.1}
    out.update(kw)
    return out


def step_line(ax, years: Sequence[float], ne: Sequence[float], window: Optional[Tuple[float, float]] = None, **kw):
    """Draw a PSMC step curve (every N_e(t) curve, summaries included), clipped to ``window`` if given."""
    if window is not None:
        years, ne = step_xy(years, ne, window)
    style = {"drawstyle": "steps-post", "solid_capstyle": "butt", "solid_joinstyle": "miter"}
    style.update(kw)
    return ax.plot(years, ne, **style)


# ---------------------------------------------------------------------------
# Settings, figures and layout
# ---------------------------------------------------------------------------


def rc_params() -> Dict:
    """The matplotlib settings of every figure."""
    from cycler import cycler

    family = list(FONT_FAMILY)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "font.size": TEXT_PT,
        "axes.labelsize": TEXT_PT,
        "axes.titlesize": TEXT_PT,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": TITLE_RISE_MM * PT_PER_MM,
        "axes.titley": 1.0,  # the title's baseline stays TITLE_RISE_MM above the axes
        "legend.fontsize": TEXT_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": LW_THIN,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.labelpad": 2.0,
        "axes.facecolor": SURFACE,
        "axes.unicode_minus": True,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color": GRID,
        "grid.linewidth": LW_HAIR,
        "grid.linestyle": "-",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": LW_THIN,
        "ytick.major.width": LW_THIN,
        "xtick.minor.width": LW_HAIR,
        "ytick.minor.width": LW_HAIR,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.minor.pad": 1.5,
        "ytick.minor.pad": 1.5,
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE, BLUE_650, INK2]),  # an unstyled line is never off-palette
        "lines.linewidth": LW_DATA,
        "lines.markersize": MARKER_MS,
        "lines.markeredgewidth": 0.0,
        "lines.solid_joinstyle": "miter",
        "lines.solid_capstyle": "butt",
        "lines.dash_capstyle": "butt",
        "patch.linewidth": LW_HAIR,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.2,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": family[0],
        "mathtext.it": f"{family[0]}:italic",
        "mathtext.bf": f"{family[0]}:bold",
        "mathtext.sf": family[0],
        "mathtext.cal": family[0],
        "mathtext.tt": family[0],
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "psmc-rs-manuscript",
        "savefig.dpi": PNG_DPI,
        "path.simplify": False,
    }


def apply() -> None:
    """Use the Agg backend and this module's settings (idempotent)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcdefaults()
    plt.rcParams.update(rc_params())
    _set_math_shrink(MATH_SHRINK)


def _set_math_shrink(factor: float) -> None:
    """Set mathtext's superscript and subscript size to ``factor`` of their text.

    mathtext reads the module constant each time it lays out a script; the
    parses it has cached at another factor are dropped.
    """
    import matplotlib._mathtext as mt
    import matplotlib.mathtext
    import matplotlib.text

    if not isinstance(getattr(mt, "SHRINK_FACTOR", None), float):
        raise RuntimeError("matplotlib._mathtext.SHRINK_FACTOR is gone: the figures' superscripts would be "
                           "below 5 pt (check the pinned matplotlib, experiment/pipeline/requirements.lock)")
    if mt.SHRINK_FACTOR != factor:
        mt.SHRINK_FACTOR = factor
        matplotlib.mathtext.MathTextParser._parse_cached.cache_clear()
        cached = getattr(matplotlib.text, "_get_text_metrics_with_cache_impl", None)
        if cached is not None:
            cached.cache_clear()


def figure(width: Union[str, float], height_mm: float):
    """A new figure ``width`` (``single``, ``onehalf``, ``double`` or mm) by ``height_mm``."""
    import matplotlib.pyplot as plt

    apply()
    w = WIDTH_MM[width] if isinstance(width, str) else float(width)
    return plt.figure(figsize=(w * MM, height_mm * MM))


def _fig_mm(fig) -> Tuple[float, float]:
    return fig.get_figwidth() / MM, fig.get_figheight() / MM


def grid(
    fig,
    nrows: int,
    ncols: int,
    *,
    left_mm: float = LEFT_MM,
    right_mm: float = RIGHT_MM,
    top_mm: float = TOP_MM,
    bottom_mm: float = BOTTOM_MM,
    wspace_mm: float = GUTTER_MM,
    hspace_mm: float = ROW_GAP_MM,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
):
    """A GridSpec whose margins and gaps are given in millimetres (the house layout by default)."""
    w_mm, h_mm = _fig_mm(fig)
    plot_w = w_mm - left_mm - right_mm
    plot_h = h_mm - top_mm - bottom_mm
    wr = list(width_ratios or [1.0] * ncols)
    hr = list(height_ratios or [1.0] * nrows)
    mean_w = (plot_w - wspace_mm * (ncols - 1)) / ncols
    mean_h = (plot_h - hspace_mm * (nrows - 1)) / nrows
    return fig.add_gridspec(
        nrows,
        ncols,
        left=left_mm / w_mm,
        right=1.0 - right_mm / w_mm,
        bottom=bottom_mm / h_mm,
        top=1.0 - top_mm / h_mm,
        wspace=wspace_mm / mean_w if ncols > 1 else 0.0,
        hspace=hspace_mm / mean_h if nrows > 1 else 0.0,
        width_ratios=wr,
        height_ratios=hr,
    )


def axes_mm(fig, x: float, y: float, w: float, h: float):
    """Axes whose top-left corner is ``x`` mm right of and ``y`` mm down from the figure's, ``w`` x ``h`` mm."""
    fw, fh = _fig_mm(fig)
    return fig.add_axes((x / fw, 1.0 - (y + h) / fh, w / fw, h / fh))


def fig_text(fig, x_mm: float, y_mm: float, text: str, **kw):
    """Text at ``x_mm`` right of and ``y_mm`` down from the figure's top-left corner (7 pt ink by default)."""
    fw, fh = _fig_mm(fig)
    opts = {"fontsize": TEXT_PT, "color": INK}
    opts.update(kw)
    return fig.text(x_mm / fw, 1.0 - y_mm / fh, text, **opts)


def ink_left_mm(ax, renderer=None) -> float:
    """The left edge of everything ``ax`` draws (tick labels, axis label), in mm from the figure's left edge."""
    fig = ax.figure
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    return ax.get_tightbbox(renderer).x0 / fig.dpi / MM


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def fmt_num(v: float) -> str:
    """``v`` with comma thousands separators, the fewest decimals and a true minus (1,000; 20; 0.5)."""
    if v == 0:
        return "0"
    s = f"{v:,.10g}"
    if "e" in s:  # beyond ten significant digits' fixed range: not a number a reader reads off an axis
        mant, exp = f"{v:.1e}".split("e")
        s = rf"${mant.rstrip('0').rstrip('.')}\times10^{{{int(exp)}}}$"
    return s.replace("-", MINUS)


def thousands_formatter():
    """Tick labels of a quantity in thousands: 1e4 is '10', 1e6 '1,000', 500 '0.5'."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fmt_num(v / 1000.0))


def plain_formatter():
    """Tick labels as plain numbers with thousands separators (10, 30, 100, 1,000)."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fmt_num(v))


def narrow_log_span(lo: float, hi: float) -> bool:
    """Whether a log axis from ``lo`` to ``hi`` is labelled at 1, 2 and 5 rather than at decades.

    It is when it spans under :data:`SPAN_125` decades or holds fewer than
    two powers of ten (1,500-60,000: 2, 5, 10, 20, 50 thousand, not 10 alone).
    """
    lo, hi = sorted((float(lo), float(hi)))
    if not (lo > 0 and math.isfinite(hi)):
        return False
    powers = math.floor(math.log10(hi) + 1e-9) - math.ceil(math.log10(lo) - 1e-9) + 1
    return math.log10(hi / lo) < SPAN_125 or powers < 2


def log_ticks(numticks: int = 12):
    """A major locator of a plain-number log axis: decades, or 1, 2 and 5 where :func:`narrow_log_span`."""
    import numpy as np
    from matplotlib.ticker import LogLocator

    class _HouseLogLocator(LogLocator):
        def tick_values(self, vmin, vmax):
            subs = [1.0, 2.0, 5.0] if narrow_log_span(vmin, vmax) else [1.0]
            self.set_params(subs=np.array(subs))
            return super().tick_values(vmin, vmax)

    return _HouseLogLocator(base=10.0, numticks=numticks)


def log_axis(ax, which: str = "x", grid: bool = False, *, fmt: str = "pow10",
             lim: Optional[Tuple[float, float]] = None) -> None:
    """A log axis with unlabelled minor ticks 2-9 and optionally a hairline major grid.

    ``fmt`` sets the major labels: ``"pow10"`` (10 with a superscript, for a
    dimensionless quantity spanning more than four decades), ``"thousands"``
    (the quantity in thousands: :func:`thousands_axis`) or ``"plain"`` (plain
    numbers); the last two are labelled at 1, 2 and 5 on spans under
    :data:`SPAN_125` decades.
    """
    from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    if fmt == "pow10":
        axis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.set_major_formatter(LogFormatterMathtext(base=10.0))
    elif fmt in ("thousands", "plain"):
        axis.set_major_locator(log_ticks())
        axis.set_major_formatter(thousands_formatter() if fmt == "thousands" else plain_formatter())
    else:
        raise ValueError(f"unknown log-axis format {fmt!r}")
    axis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    axis.set_minor_formatter(NullFormatter())
    if lim is not None:
        (ax.set_xlim if which == "x" else ax.set_ylim)(*lim)
    if grid:
        ax.grid(True, which="major", axis=which, color=GRID, linewidth=LW_HAIR)


def thousands_axis(ax, which: str = "x", lim: Optional[Tuple[float, float]] = None, grid: bool = False) -> None:
    """A log axis of a quantity in thousands (N_e, years ago): ticks 1, 10, 100, 1,000."""
    log_axis(ax, which, grid=grid, fmt="thousands", lim=lim)


thousands_log_axis = thousands_axis


def years_axis(ax, lim: Tuple[float, float], label: Union[bool, str] = True, grid: bool = False) -> None:
    """The time axis of an N_e(t) panel: years ago, log scale, labelled in thousands (:data:`YEARS_LABEL`)."""
    thousands_axis(ax, "x", lim, grid=grid)
    if label:
        ax.set_xlabel(label if isinstance(label, str) else YEARS_LABEL)


def ne_axis(ax, lim: Tuple[float, float], label: Optional[str] = NE_LABEL, grid: bool = False) -> None:
    """The size axis of an N_e(t) panel: log scale, labelled in thousands (:data:`NE_LABEL`)."""
    thousands_axis(ax, "y", lim, grid=grid)
    if label:
        ax.set_ylabel(label)


def value_grid(ax, which: str = "y") -> None:
    """Hairlines at the major ticks of the value axis, where values are read off (bars, timings)."""
    ax.grid(True, which="major", axis=which, color=GRID, linewidth=LW_HAIR)
    ax.set_axisbelow(True)


def row_xlabel(fig, axes: Sequence, text: str = YEARS_LABEL, **kw):
    """One x label centred under a row of panels, where a panel's own x label would sit."""
    import matplotlib as mpl

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bottoms = [ax.get_position().y0 for ax in axes]
    for ax in axes:
        for t in ax.xaxis.get_ticklabels():
            if t.get_visible() and t.get_text():
                bottoms.append(inv.transform(t.get_window_extent(renderer).get_points())[0][1])
    pad = mpl.rcParams["axes.labelpad"] / 72.0 / fig.get_figheight()
    x0, x1 = min(ax.get_position().x0 for ax in axes), max(ax.get_position().x1 for ax in axes)
    opts = {"fontsize": TEXT_PT, "color": INK, "ha": "center", "va": "top"}
    opts.update(kw)
    return fig.text((x0 + x1) / 2.0, min(bottoms) - pad, text, **opts)


def shade_outside(ax, window: Tuple[float, float], lim: Tuple[float, float]) -> None:
    """Shade the times outside the informative window (pipeline-only figures; the article clips curves instead)."""
    lo, hi = window
    if lim[0] < lo:
        ax.axvspan(lim[0], lo, color=FILL, linewidth=0, zorder=0)
    if hi < lim[1]:
        ax.axvspan(hi, lim[1], color=FILL, linewidth=0, zorder=0)


def ref_line(ax, y: Optional[float] = None, x: Optional[float] = None, **kw) -> None:
    """A thin reference line (zero difference, ratio one, nominal coverage): 0.5 pt :data:`REF`."""
    style = {"color": REF, "linewidth": LW_THIN, "linestyle": "-", "zorder": Z_REF}
    style.update(kw)
    if y is not None:
        ax.axhline(y, **style)
    if x is not None:
        ax.axvline(x, **style)


def note(ax, x: float, y: float, text: str, *, ha: str = "left", va: str = "top", color: str = INK2,
         size: float = NOTE_PT, transform=None, **kw):
    """A context note: 6 pt secondary ink (never a series colour), in axes coordinates by default."""
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=size,
                   transform=transform if transform is not None else ax.transAxes, **kw)


def direct_label(ax, x: float, y: float, text: str, *, ha: str = "left", va: str = "center", color: str = INK,
                 size: float = LABEL_PT, transform=None, **kw):
    """A number or name printed on the data: 6 pt ink, in data coordinates by default."""
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=size,
                   transform=transform if transform is not None else ax.transData, **kw)


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------

LEGEND_KW = {"frameon": False, "fontsize": TEXT_PT, "handlelength": 1.8, "handletextpad": 0.5,
             "columnspacing": 1.5}


def legend(ax_or_fig, handles: Sequence, labels: Optional[Sequence[str]] = None, **kw):
    """A frameless legend in 7 pt ink."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    opts = dict(LEGEND_KW)
    opts.update(kw)
    return ax_or_fig.legend(list(handles), list(labels), **opts)


def top_legend(fig, handles: Sequence, labels: Optional[Sequence[str]] = None, *, ax=None,
               x_mm: Optional[float] = None, top_mm: float = MARGIN_MM, **kw):
    """The figure's key: one line at its top, left-aligned with the first left spine.

    The key's left edge is ``ax``'s left spine (the leftmost axes by
    default) or ``x_mm``; its top is ``top_mm`` below the figure's top.
    """
    if labels is None:
        labels = [h.get_label() for h in handles]
    fw, fh = _fig_mm(fig)
    if x_mm is None:
        ref = ax if ax is not None else min(fig.axes, key=lambda a: a.get_position().x0)
        x = ref.get_position().x0
    else:
        x = x_mm / fw
    opts = dict(LEGEND_KW)
    opts.update(loc="upper left", bbox_to_anchor=(x, 1.0 - top_mm / fh), bbox_transform=fig.transFigure,
                ncol=len(handles), borderaxespad=0.0, borderpad=0.0)
    opts.update(kw)
    return fig.legend(list(handles), list(labels), **opts)


def handle(entity: Entity, kind: str = "line", **kw):
    """A legend handle drawn as the entity is: its line, its marker, both, or its underlay."""
    from matplotlib.lines import Line2D

    if kind == "line":
        style = entity.line()
    elif kind == "points":
        style = entity.points()
    elif kind == "underlay":
        style = underlay_kw(entity)
        style.pop("zorder")
    elif kind == "both":
        style = entity.line()
        style.update(marker=entity.marker, markersize=MARKER_MS,
                     markeredgewidth=0.0 if entity.filled else LW_THIN,
                     markerfacecolor=entity.color if entity.filled else SURFACE, markeredgecolor=entity.color)
    else:
        raise ValueError(f"unknown handle kind {kind!r}")
    style.update(kw)
    return Line2D([], [], **style)


# ---------------------------------------------------------------------------
# Panel letters and titles
# ---------------------------------------------------------------------------


def panel_title(ax, text: str, **kw):
    """The panel's title: above the axes, left-aligned with the left spine, on its letter's baseline.

    7 pt :data:`INK`, its baseline :data:`TITLE_RISE_MM` above the axes top
    (the same rise :func:`panel_letters` gives the letters).
    """
    opts = {"loc": "left", "pad": TITLE_RISE_MM * PT_PER_MM, "y": 1.0, "fontsize": TEXT_PT, "color": INK,
            "fontweight": "normal"}
    opts.update(kw)
    return ax.set_title(text, **opts)


def panel_letter(fig, x_mm: float, baseline_mm: float, letter: str, **kw):
    """One panel letter placed by hand: 8 pt bold ink, its baseline ``baseline_mm`` below the figure's top."""
    opts = {"fontsize": PANEL_PT, "fontweight": "bold", "color": INK, "ha": "left", "va": "baseline"}
    opts.update(kw)
    return fig_text(fig, x_mm, baseline_mm, letter, **opts)


def panel_letters(fig, axes: Sequence, letters: Optional[Sequence[str]] = None, *,
                  first_x_mm: Optional[float] = LETTER_X_MM, rise_mm: float = TITLE_RISE_MM,
                  align_rows: bool = True, align_cols: bool = True) -> List:
    """Bold lowercase letters at the top left of each panel, on its title's baseline.

    Each letter's baseline is ``rise_mm`` above its axes' top (the highest
    top of its row: axes sharing a top edge share a baseline), the baseline
    :func:`panel_title` gives titles. The leftmost panel of each row has its
    letter at ``first_x_mm`` from the figure's left edge, so every figure's
    first letters line up; any other letter sits at the left edge of its
    panel's ink (tick labels, axis label) and, where the panel has a title,
    at least :data:`LETTER_TITLE_GAP_MM` left of it. With ``align_cols``,
    letters of panels sharing a left edge (of their grid cells, before a
    fixed aspect shrinks them) share the leftmost of their positions.
    ``first_x_mm=None`` places first-column letters at their ink too.
    """
    letters = list(letters or string.ascii_lowercase[: len(axes)])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fw, fh = _fig_mm(fig)
    rise = rise_mm / fh
    gap = LETTER_TITLE_GAP_MM / fw

    def left(ax):
        return ax.get_position(original=True).x0

    row_of = {id(ax): round(ax.get_position().y1, 3) if align_rows else id(ax) for ax in axes}
    rows: Dict = {}
    for ax in axes:
        rows.setdefault(row_of[id(ax)], []).append(ax)
    first = set()
    if first_x_mm is not None:
        for row in rows.values():
            x_min = min(left(ax) for ax in row)
            first |= {id(ax) for ax in row if (left(ax) - x_min) * fw < 0.5}

    texts = []
    xs: Dict[int, float] = {}
    for ax, letter in zip(axes, letters):
        top = max(a.get_position().y1 for a in rows[row_of[id(ax)]])
        t = fig.text(0.0, top + rise, letter, fontsize=PANEL_PT, fontweight="bold", ha="left", va="baseline",
                     color=INK)
        texts.append(t)
        if id(ax) in first:
            xs[id(ax)] = first_x_mm / fw
            continue
        (x0, _), _ = inv.transform(ax.get_tightbbox(renderer).get_points())
        title = getattr(ax, "_left_title", None) if ax.get_title("left") else None  # set by panel_title
        if title is not None:
            (a0, _), (a1, _) = inv.transform(t.get_window_extent(renderer).get_points())
            (title_x0, _), _ = inv.transform(title.get_window_extent(renderer).get_points())
            x0 = min(x0, title_x0 - gap - (a1 - a0))
        xs[id(ax)] = x0
    if align_cols:
        cols: Dict = {}
        for ax in axes:
            if id(ax) not in first:
                key = round(left(ax), 3)
                cols[key] = min(cols.get(key, xs[id(ax)]), xs[id(ax)])
        for ax in axes:
            if id(ax) not in first:
                xs[id(ax)] = cols[round(left(ax), 3)]
    for ax, t in zip(axes, texts):
        t.set_x(xs[id(ax)])
    return texts


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FORMATS_OUT = ("pdf", "svg", "png")


def save(fig, out_dir: Path, name: str, dpi: int = PNG_DPI) -> List[Path]:
    """Write ``name.pdf``, ``name.svg`` and ``name.png`` (``dpi``) to ``out_dir``, reproducibly."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    metadata = {
        "pdf": {"Creator": "psmc-rs manuscript figures", "Producer": None, "CreationDate": None, "ModDate": None},
        "svg": {"Creator": "psmc-rs manuscript figures", "Date": None},
        "png": {"Software": None},
    }
    written = []
    for ext in FORMATS_OUT:
        path = out_dir / f"{name}.{ext}"
        tmp = path.with_name(path.name + ".partial")
        fig.savefig(tmp, format=ext, dpi=dpi if ext == "png" else None, metadata=metadata[ext])
        tmp.replace(path)
        written.append(path)
    plt.close(fig)
    return written


_SVG_SIZE = re.compile(r"font-size:\s*([0-9.]+)px|font:\s*(?:[a-z-]+\s+)*([0-9.]+)px")
_SVG_TEXT = re.compile(r"<(?:text|tspan)\b[^>]*?style=\"([^\"]*)\"[^>]*>([^<]*)")


def small_text(svg: Union[str, Path], min_pt: float = MIN_TEXT_PT) -> List[Tuple[float, str]]:
    """``(size, text)`` of every piece of text in a written SVG smaller than ``min_pt``.

    The SVG's user units are points (matplotlib writes its width in pt), and
    with ``svg.fonttype = none`` every text element keeps its size.
    """
    out = []
    for style_attr, text in _SVG_TEXT.findall(Path(svg).read_text()):
        m = _SVG_SIZE.search(style_attr)
        if m:
            size = float(m.group(1) or m.group(2))
            if size < min_pt - 1e-6:
                out.append((size, text))
    return out


def step_xy(years: Sequence[float], ne: Sequence[float], xlim: Tuple[float, float]):
    """Coordinates of a PSMC step curve for ``drawstyle='steps-post'`` within ``xlim``.

    ``ne[k]`` holds from ``years[k]`` to ``years[k + 1]``; the first interval
    starts at the left edge of the axis and the last one runs to its right
    edge.
    """
    import numpy as np

    years = np.asarray(years, dtype=float)
    ne = np.asarray(ne, dtype=float)
    lo, hi = xlim
    starts = years.copy()
    starts[0] = lo
    keep = starts < hi
    keep[0] = True
    x = np.append(np.maximum(starts[keep], lo), hi)
    y = np.append(ne[keep], ne[keep][-1])
    # Intervals that end before the left edge collapse onto it; keep the last of them.
    first = np.searchsorted(x, lo, side="right") - 1
    return x[first:], y[first:]
