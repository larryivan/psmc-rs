"""Supplement Fig S3 (E5): psmc-rs's bootstrap measures the sampling spread of a PSMC fit, not PSMC's bias.

The bottleneck model is simulated as many independent genomes, each fitted
once: their spread at each time is the true sampling distribution of the
estimate. The first genomes are bootstrapped at two block sizes (splitfa's
trunks), named by their number of blocks per genome, which is what drives
the bootstrap: the most blocks in blue, the fewest in light blue
(``style.TRUNK_COLORS``), in every panel. Every panel spans the model's
informative window. One row of three panels, 183 x 56 mm, with no figure
legend:

* a: one bootstrapped genome (``config.E5.example_dataset``, else the
  first): the truth (dashed ink), the genome's fit (blue) and its bootstrap
  95% band with the most blocks (pale blue, the panel's only fill), and the
  95% range of the independent fits (two thin grey lines), all as PSMC step
  curves; band and range are as wide as each other, and both miss the
  truth where PSMC is biased. The y axis spans half the truth's smallest
  size to five times its largest, so the first interval's band and range
  run off the top (the caption says so).
* b: calibration of the bootstrap's spread: at each time of the window,
  the bootstrap SD of log10 Ne (median over the genomes) against the true
  SD (over the independent fits), one dot per distinct pair, on log axes
  with the identity line and, in grey, where a bootstrap with the true SD
  would fall given the true SD's own chi-square 95% interval (the median
  of its bounds over the window, relative to the SD). The key gives each
  block count's median ratio over the window (``E5_summary``), its counts
  right-aligned so that the ratios form one column.
* c: what the right width buys: for each genome (a dot), the share of the
  window's times at which its band holds the mean fit (the geometric mean
  of the independent fits) or the truth, for each block count; bars are
  the means (``E5_summary``), the grey line the nominal coverage. The gap
  between the two groups is PSMC's bias, not the bootstrap.

Text is placed from the fonts' own metrics, not from what a renderer
measures (the PDF backend hints widths at 72 dpi, the PNG at its own dpi),
so in every output the axis labels hang at fixed distances from their
spines (a's and b's x labels and c's group names on one baseline) and the
letters b and c start at the ink of their panel's y label. The panels
leave the same white between one panel's ink and the next one's.

Every number comes from E5's tables: ``E5_summary`` (block counts, median
SD ratios, window coverages), ``E5_sampling`` (truth, the fits' range and
mean), ``E5_bands`` (the genomes' fits and bands, coverage of the truth)
and ``E5_sd_ratio`` (the SDs). The per-genome coverages are recomputed from
``E5_bands`` and checked against ``E5_summary``, and ``E5_coverage``'s
pointwise shares against both: a disagreement raises ``ValueError`` rather
than drawing numbers the text does not cite.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import style
from .data import Source

# Layout (mm). Every axes top at style.TOP_MM, its bottom spine 7.5 mm above the figure's bottom; a starts at
# style.LEFT_MM, b is square (a calibration plot) and c ends style.RIGHT_MM from the right edge. a's width and
# b's x follow from the text (_layout): GAP_MM of white between a's ink and b's, and between b's and c's.
W_MM, H_MM = style.WIDTH_MM["double"], 56.0
AX_H_MM = 44.0
B_W_MM = AX_H_MM
C_W_MM = 27.5
GAP_MM = 6.5

# a's band is the one fill of the panel and must read against two 0.5 pt grey
# lines at print size: psmc-rs's band share (style.BAND_SHARE, 0.16) is too
# faint beside them, so it is the same blue at 25% over white (#caddf5).
BAND_SHARE_A = 0.25
# a's curves are E5's fits and truth read at the grid's times (``curve.at(grid)``),
# not PSMC's interval starts, so they are drawn steps-mid, not steps-post as the
# set's other N_e(t) curves: a size change happened somewhere between two grid
# times, and the midpoint puts each riser within half a grid step of it (0.5% of
# a decade in production; steps-post would put it up to a whole step late), e.g.
# the truth's at 20, 30, 100 and 1,000 thousand years.
STEP = "mid"
# b's key: counts padded with figure spaces (as wide as a digit) so that they end, and the ratios
# start, at one x; no-break spaces before the ratio, which an SVG keeps.
FIGURE_SPACE = " "
RATIO_SEP = "  "
# c: strips in data units. The block counts of a group are ARM_STEP apart and
# the groups GROUP_GAP further; dots jitter within +-JITTER (0.29 of the strip
# spacing) and the mean bars span +-BAR_HALF (0.40 of it).
ARM_STEP = 0.42
GROUP_GAP = 0.80
X_PAD = 0.30
JITTER = 0.12
BAR_HALF = 0.17
JITTER_SEED = 3
DOT = {"linestyle": "none", "marker": "o", "markersize": style.STRIP_MS, "markeredgewidth": 0.0}  # b's and c's
NOMINAL_DROP = 2.0  # c's 'Nominal 95%' hangs this many points of coverage below its line
TOL = 1e-9  # recomputed coverages against E5_summary


def _arm_colors(arms: Sequence[int]) -> Dict[int, str]:
    """The most blocks blue, the fewest light blue (``style.TRUNK_COLORS``)."""
    last = len(style.TRUNK_COLORS) - 1
    return {t: style.TRUNK_COLORS[min(i, last)] for i, t in enumerate(arms)}


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
# Panels
# ---------------------------------------------------------------------------


def _panel_a(ax, sampling, example, color: str) -> None:
    """a: the truth, the independent fits' range and (``example``) one genome's fit and band, with its key."""
    import numpy as np
    from matplotlib.lines import Line2D

    years = sampling["years"].to_numpy(dtype=float)
    true_ne = sampling["true_ne"].to_numpy(dtype=float)
    step = {"drawstyle": f"steps-{STEP}"}  # see STEP: grid times, not interval starts
    has_fit = example is not None and len(example) > 0
    if has_fit:
        band = ax.fill_between(example["years"], example["boot_lo_ne"], example["boot_hi_ne"], step=STEP,
                               **style.band_kw(color, BAND_SHARE_A))
    for edge in ("lo_ne", "hi_ne"):  # the independent fits' range: two lines, never a second fill
        ax.plot(years, sampling[edge], color=style.REF, linewidth=style.LW_THIN, zorder=style.Z_CONTEXT, **step)
    if has_fit:
        ax.plot(example["years"], example["fit_ne"], **style.overlay_kw(color), **step)
    ax.plot(years, true_ne, **style.truth_kw(), **step)

    lo = 10.0 ** math.floor(math.log10(years[0]) + 1e-9)
    xlim = (lo if years[0] / lo < 1.1 else float(years[0]), float(years[-1]))
    floor, top = float(np.nanmin(true_ne)) / 2.0, float(np.nanmax(true_ne)) * 5.0
    if has_fit:  # a fit far off the truth (small profiles) stays in view
        fit = example["fit_ne"].to_numpy(dtype=float)
        floor, top = min(floor, float(np.nanmin(fit)) / 1.5), max(top, float(np.nanmax(fit)) * 1.5)
    style.years_axis(ax, xlim)
    style.ne_axis(ax, (floor, top))

    fits = int(sampling["fits"].iloc[0])
    keys: List = [style.handle(style.TRUTH, linestyle=style.centred_dash(style.DASH))]
    labels = ["Truth"]
    if has_fit:
        fit_key = style.overlay_kw(color)
        fit_key.pop("zorder")
        keys += [Line2D([], [], **fit_key), band]
        labels += ["One genome’s fit", "Its bootstrap 95% band"]
    keys.append(Line2D([], [], color=style.REF, linewidth=style.LW_THIN))
    labels.append(f"95% range of {fits} independent fits")
    style.legend(ax, keys, labels, loc="upper right", labelspacing=0.25, borderaxespad=0.1)


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
    if arms:
        style.legend(ax, [Line2D([], [], color=color[t], **DOT) for t in arms], _key_labels_b(arms, rows),
                     loc="upper left", handlelength=0.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.1)


def _panel_c(ax, per, arms: Sequence[int], color: Dict[int, str], rows: Dict, nominal: float) -> List[float]:
    """Draw c's strips; return the centre of each group (mean fit, truth) in data x."""
    import matplotlib as mpl
    import numpy as np
    from matplotlib.ticker import FixedLocator

    groups = (("mean", "coverage_of_fit_mean_mean"), ("truth", "coverage_mean"))
    ax.set_ylim(0.0, 100.0)
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    ax.set_ylabel("Coverage (%)")
    if not arms:
        return []
    width = (len(arms) - 1) * ARM_STEP + GROUP_GAP
    pos = {(g, t): gi * width + i * ARM_STEP for gi, (g, _) in enumerate(groups) for i, t in enumerate(arms)}
    xmax = max(pos.values())
    style.ref_line(ax, y=nominal, zorder=style.Z_BAND)
    rng = np.random.default_rng(JITTER_SEED)
    for g, summary_col in groups:
        for t in arms:
            x = pos[(g, t)]
            v = 100.0 * per.xs(t, level="trunk_bins").sort_index()[g].to_numpy(dtype=float)
            ax.plot(x + rng.uniform(-JITTER, JITTER, len(v)), v, color=color[t], zorder=style.Z_UNDERLAY,
                    clip_on=False, **DOT)  # whole dots at 100%
            m = 100.0 * float(getattr(rows[t], summary_col))
            ax.plot([x - BAR_HALF, x + BAR_HALF], [m, m], color=style.INK, linewidth=style.LW_DATA,
                    solid_capstyle="butt", zorder=style.Z_DATA)
    ax.set_xlim(-X_PAD, xmax + X_PAD)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([f"{int(rows[t].trunks)}" for (_, t) in pos])
    # No tick marks under the strips; the pad puts the counts on the baseline of a's and b's tick labels.
    ax.tick_params(axis="x", length=0.0, pad=mpl.rcParams["xtick.major.size"] + mpl.rcParams["xtick.major.pad"])
    style.note(ax, xmax + X_PAD, nominal - NOMINAL_DROP, f"Nominal {nominal:.0f}%", ha="right", va="top",
               transform=ax.transData)
    return [float(np.mean([pos[(g, t)] for t in arms])) for g, _ in groups]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _layout(ax_a, ax_b, ax_c) -> List[float]:
    """Place the three axes; return each one's y-label ink, in mm left of its spine.

    a starts at ``style.LEFT_MM`` and c ends ``style.RIGHT_MM`` from the
    right edge; b's x and a's width leave ``GAP_MM`` of white between a's ink
    (its spine or tick labels) and b's y label, and between b's ink and c's.
    """
    ink = [_ylabel_pt(ax)[1] / style.PT_PER_MM for ax in (ax_a, ax_b, ax_c)]
    c_x = W_MM - style.RIGHT_MM - C_W_MM
    b_x = c_x - ink[2] - GAP_MM - _right_ink_mm(ax_b, B_W_MM) - B_W_MM
    a_w0 = ax_a.get_position().width * W_MM
    a_w = b_x - ink[1] - GAP_MM - _right_ink_mm(ax_a, a_w0) - style.LEFT_MM
    bottom = (H_MM - style.TOP_MM - AX_H_MM) / H_MM
    for ax, x, w in ((ax_a, style.LEFT_MM, a_w), (ax_b, b_x, B_W_MM), (ax_c, c_x, C_W_MM)):
        ax.set_position((x / W_MM, bottom, w / W_MM, AX_H_MM / H_MM))
    return ink


def _on_baseline(text) -> None:
    """Draw ``text``'s mathtext on its baseline in the PDF and SVG, as Agg draws it in the PNG.

    matplotlib's vector layout of mathtext (``_mathtext.Output.to_vector``)
    rounds the math box's height up to a whole point and so raises every
    glyph by the difference: 0.9 pt (0.32 mm) for b's labels, enough to lift
    b's x label off the shared baseline and a's and b's y labels off their
    letters. The raise is the height of the first glyph, which sits on the
    baseline; a renderer other than Agg draws ``text`` lowered by it.
    """
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.cbook import is_math_text
    from matplotlib.mathtext import MathTextParser
    from matplotlib.transforms import ScaledTranslation

    s = text.get_text()
    if not is_math_text(s):
        return
    glyphs = MathTextParser("path").parse(s, 72, text.get_fontproperties()).glyphs
    lift = float(glyphs[0][-1]) if glyphs else 0.0
    if not lift > 0.0:
        return
    a = math.radians(text.get_rotation())
    down = ScaledTranslation(math.sin(a) * lift / 72.0, -math.cos(a) * lift / 72.0, text.figure.dpi_scale_trans)
    draw = text.draw

    def draw_on_baseline(renderer):
        if isinstance(renderer, RendererAgg):
            return draw(renderer)
        tr = text.get_transform()
        text.set_transform(tr + down)
        try:
            return draw(renderer)
        finally:
            text.set_transform(tr)

    text.draw = draw_on_baseline


def _pin_labels(fig, axes) -> float:
    """Hang every axis label at a fixed distance from its spine; return the x labels' drop in points.

    y labels on their baseline, where matplotlib's rule would put them with
    unhinted metrics; x labels on one baseline under the bottom spines.
    """
    from matplotlib.transforms import ScaledTranslation

    drop = _xlabel_drop_pt()
    for ax in axes:
        base = _ylabel_pt(ax)[0]
        ax.yaxis.set_label_coords(0.0, 0.5, transform=ax.transAxes + ScaledTranslation(
            -base / 72.0, 0.0, fig.dpi_scale_trans))
        ax.xaxis.set_label_coords(0.5, 0.0, transform=ax.transAxes + ScaledTranslation(
            0.0, -drop / 72.0, fig.dpi_scale_trans))
        for label in (ax.yaxis.label, ax.xaxis.label):
            label.set_verticalalignment("baseline")
            _on_baseline(label)
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

    fig = style.figure(W_MM, H_MM)
    # Provisional widths; _layout places the axes once their tick labels are known.
    ax_a = style.axes_mm(fig, style.LEFT_MM, style.TOP_MM, 70.0, AX_H_MM)
    ax_b = style.axes_mm(fig, 96.0, style.TOP_MM, B_W_MM, AX_H_MM)
    ax_c = style.axes_mm(fig, W_MM - style.RIGHT_MM - C_W_MM, style.TOP_MM, C_W_MM, AX_H_MM)

    datasets = {int(d) for d in bands["dataset"]}
    ds = cfg.example_dataset if cfg.example_dataset in datasets else (min(datasets) if datasets else None)
    example = None
    if ds is not None and arms:
        example = bands[(bands["dataset"] == ds) & (bands["trunk_bins"] == arms[0])].sort_values("point")
    _panel_a(ax_a, sampling, example, color[arms[0]] if arms else style.BLUE)
    _panel_b(ax_b, sd, arms, color, rows)
    nominal = 100.0 * (cfg.band[1] - cfg.band[0])
    centres = _panel_c(ax_c, per, arms, color, rows, nominal)

    axes = (ax_a, ax_b, ax_c)
    ink = _layout(*axes)
    drop = _pin_labels(fig, axes)
    # c's group names on the x labels' baseline, under its block counts.
    tr = blended_transform_factory(ax_c.transData, ax_c.transAxes) + ScaledTranslation(
        0.0, -drop / 72.0, fig.dpi_scale_trans)
    for xc, name in zip(centres, ("Mean fit", "Truth")):
        ax_c.text(xc, 0.0, name, transform=tr, ha="center", va="baseline", fontsize=style.TEXT_PT, color=style.INK)

    # Letters on one baseline style.TITLE_RISE_MM above the axes tops: a at style.LETTER_X_MM as in every
    # figure; b and c with their ink at the ink of their panel's y label.
    baseline = style.TOP_MM - style.TITLE_RISE_MM
    style.panel_letter(fig, style.LETTER_X_MM, baseline, "a")
    for ax, letter, ink_mm in zip(axes[1:], "bc", ink[1:]):
        x = ax.get_position().x0 * W_MM - ink_mm
        style.panel_letter(fig, x - _ink(letter, style.PANEL_PT, "bold").x0 / style.PT_PER_MM, baseline, letter)
    return fig


__all__ = ["draw", "genome_coverage"]
