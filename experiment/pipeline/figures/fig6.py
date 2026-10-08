"""Fig 6 (E5): how well psmc-rs's bootstrap measures the uncertainty of a PSMC curve.

The bottleneck model is simulated as many independent genomes of 6 x 50 Mb,
each fitted once: their spread at each time is the true sampling
distribution of the estimate. The first genomes are bootstrapped at two
block sizes (splitfa's trunks), the default 50 Mb (blue) and 5 Mb (dark
blue, dashed). Every panel spans the model's informative window, where the
calibration is computed.

* a: one genome (``config.E5.example_dataset``): the truth (dashed), the
  95% range of the independent fits (grey, the panel's only fill), the
  genome's fit (blue line) and the edges of its bootstrap 95% bands (50 Mb
  blocks light blue, 5 Mb blocks dark blue, dashed);
* b: SD of log10 Ne: the true sampling SD over the independent fits (black,
  with its chi-square 95% interval, light grey) and the bootstrap's (median
  over the bootstrapped genomes);
* c: bootstrap SD / true SD: median over the genomes and their
  interquartile range (50 Mb blocks filled, 5 Mb blocks as dashed edges);
  the light grey band is where a bootstrap whose SD equalled the true one
  would fall, given the true SD's own sampling error. The axis ends at 2;
  a quartile above it is marked at the top with its value;
* d: share of the bootstrapped genomes whose 95% band covers the true Ne
  (thick; with the pointwise Clopper-Pearson 95% interval of the 50 Mb
  blocks, the 5 Mb blocks' is left out so the panel stays legible) and the
  fits' geometric mean (thin), against the nominal 95%; with 10 genomes the
  share moves in steps of 10%;
* e: the same share for each genome's pattern-projected truth (the truth
  averaged over each free parameter's span, which a PSMC curve can
  represent), with its Clopper-Pearson interval for 50 Mb blocks. Panel a
  spans the figure's width, so that d and e sit side by side.

The one-point spikes in b-e are times at which the fits' interval
boundaries, which move with each fit's theta and t_max, straddle the grid
time, so that the fits read different intervals there: the SD there is
genuinely larger, not an error.

The legend gives each block size's coverage averaged over the window (an
unweighted mean over the window's log-spaced times, with its interval from
resampling the bootstrapped genomes, ``E5_summary``), of the true Ne, of the
fits' mean and, when given, of the projected truth. Coverage of the true Ne
is low because PSMC's own bias, not the bootstrap's spread, dominates the
error at the bottleneck and at the size changes: the bands cover the fits'
mean far more often.

Everything comes from E5's tables: ``E5_bands``, ``E5_sampling``,
``E5_sd_ratio``, ``E5_coverage`` and ``E5_summary``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from . import style
from .data import Source

GREY_BAND = "#dbdad4"  # the independent fits' range (a), opaque: the only fill of a
BOOT_EDGE = style.BLUE_RAMP[4]  # the 50 Mb bootstrap band's edges in a: lighter than the fit's line
C_TOP = 2.0  # c's y-axis ends here (or just above the highest median); higher quartiles are marked at the top
REFERENCE_BAND = "#ebeae4"  # the true SD's sampling interval (b, c): lighter, and outlined in the legend
CI_ALPHA = 0.12  # Clopper-Pearson interval (d)


def trunk_styles(trunks: List[int]) -> Dict[int, tuple]:
    """``{trunk: (colour, line style)}``, the default (largest) trunk first."""
    return {t: (style.TRUNK_COLORS[min(i, 1)], style.TRUNK_STYLES[min(i, 1)])
            for i, t in enumerate(sorted(trunks, reverse=True))}


def trunk_label(row) -> str:
    """``50 Mb blocks (6 per genome)``: splitfa's trunks, named blocks as in Fig 3."""
    return f"{float(row.trunk_mb):g} Mb blocks ({int(row.trunks)} per genome)"


def _plain(columns, prefix: str = "coverage") -> List[str]:
    return [c for c in columns if c.startswith(prefix) and "proj" in c
            and not any(t in c for t in ("_ci_", "_lo", "_hi", "_sd", "_min", "_max"))]


def projected_columns(cov, summary) -> Optional[Dict[str, Optional[str]]]:
    """The columns of the coverage of the pattern-projected truth, if E5's tables have them.

    ``{"point", "lo", "hi", "mean", "mean_lo", "mean_hi"}``: the pointwise
    share in ``E5_coverage`` (``coverage_*proj*``) and its interval, and its
    window average in ``E5_summary`` with its interval; any but ``point``
    may be None. None when ``E5_coverage`` has no such column.
    """
    points = _plain(cov.columns)
    if not points:
        return None
    point = points[0]
    means = [c for c in _plain(summary.columns) if c.endswith("mean")] if summary is not None else []
    mean = means[0] if means else None

    def bound(table, col, side):
        if table is None or col is None:
            return None
        for name in (f"{col}_ci_{side}", f"{col}_{side}"):
            if name in table.columns:
                return name
        return None

    return {"point": point, "lo": bound(cov, point, "lo"), "hi": bound(cov, point, "hi"), "mean": mean,
            "mean_lo": bound(summary, mean, "lo"), "mean_hi": bound(summary, mean, "hi")}


def _pct(x) -> str:
    return f"{100 * float(x):.0f}%"


def _range(lo, hi) -> str:
    return f"{100 * float(lo):.0f}–{100 * float(hi):.0f}%"


def coverage_label(row, proj: Optional[Dict[str, Optional[str]]]) -> str:
    """A block size's legend label with its window coverages."""
    ci = _range(row.coverage_mean_ci_lo, row.coverage_mean_ci_hi)
    parts = [f"true $N_\\mathrm{{e}}$ {_pct(row.coverage_mean)} ({ci})"]
    if proj is not None and proj["mean"] is not None:
        v = getattr(row, proj["mean"])
        ci = ""
        if proj["mean_lo"] is not None and proj["mean_hi"] is not None:
            ci = f" ({_range(getattr(row, proj['mean_lo']), getattr(row, proj['mean_hi']))})"
        parts.append(f"projected truth {_pct(v)}{ci}")
    if hasattr(row, "coverage_of_fit_mean_mean"):
        parts.append(f"fits' mean {_pct(row.coverage_of_fit_mean_mean)}")
    return f"Bootstrap, {trunk_label(row)}; window coverage of the " + ", ".join(parts)


def draw(src: Source):
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = src.profile.e5
    summary = src.table("E5_summary")
    informative = summary[summary.window == "informative"]
    sampling = src.table("E5_sampling").sort_values("point")
    bands = src.table("E5_bands")
    sd = src.table("E5_sd_ratio")
    cov = src.table("E5_coverage")
    proj = projected_columns(cov, informative)
    trunks = sorted((int(t) for t in informative["trunk_bins"]), reverse=True)
    sty = trunk_styles(trunks)
    rows = {int(r.trunk_bins): r for r in informative.itertuples()}
    years = sampling["years"].to_numpy()
    xlim = (float(years[0]) / 1.12, float(years[-1]) * 1.12)

    fig = style.figure("double", 178)
    gs = style.grid(fig, 3, 2, left_mm=14.0, right_mm=3.0, top_mm=29.0, bottom_mm=9.0, wspace_mm=16.0,
                    hspace_mm=11.0, height_ratios=(0.8, 1.0, 1.0))
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c, ax_d, ax_e = (fig.add_subplot(gs[1 + i // 2, i % 2]) for i in range(4))

    # a: one genome.
    ds = cfg.example_dataset if cfg.example_dataset in set(bands["dataset"]) else int(bands["dataset"].min())
    ax_a.fill_between(years, sampling["lo_ne"], sampling["hi_ne"], color=GREY_BAND, linewidth=0, zorder=1)
    vals = [float(sampling["lo_ne"].min()), float(sampling["hi_ne"].max())]
    fit = None
    for t in trunks:
        b = bands[(bands.dataset == ds) & (bands.trunk_bins == t)].sort_values("point")
        color, ls = sty[t]
        # The bootstrap bands as edges only, so that no fill overlaps the grey range of the independent fits.
        for edge in ("boot_lo_ne", "boot_hi_ne"):
            if t == trunks[0]:
                ax_a.plot(b["years"], b[edge], color=BOOT_EDGE, linewidth=0.6, zorder=3)
            else:
                ax_a.plot(b["years"], b[edge], color=color, linestyle=ls, linewidth=0.7, zorder=3)
        vals += [float(b["boot_lo_ne"].min()), float(b["boot_hi_ne"].max())]
        fit = b
    if fit is not None:
        ax_a.plot(fit["years"], fit["fit_ne"], color=style.BLUE, linewidth=style.LW, zorder=4)
    ax_a.plot(years, sampling["true_ne"], color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=5)
    style.years_axis(ax_a, xlim)
    style.ne_axis(ax_a, (min(vals) / 1.3, max(vals) * 1.3))
    style.note(ax_a, 0.97, 0.03, f"genome {ds}", ha="right", va="bottom")

    # b: the SDs.
    first = sd[sd.trunk_bins == trunks[0]].sort_values("point")
    ax_b.fill_between(first["years"], first["sd_true_ci_lo"], first["sd_true_ci_hi"], color=REFERENCE_BAND,
                      linewidth=0, zorder=1)
    ax_b.plot(first["years"], first["sd_true_log10"], color=style.INK, linewidth=style.LW_BOLD, zorder=3)
    top = float(first["sd_true_ci_hi"].max())
    for t in trunks:
        s = sd[sd.trunk_bins == t].sort_values("point")
        color, ls = sty[t]
        ax_b.plot(s["years"], s["sd_boot_log10_median"], color=color, linestyle=ls, linewidth=style.LW, zorder=4)
        top = max(top, float(s["sd_boot_log10_median"].max()))
    ax_b.set_ylim(0.0, top * 1.1)
    ax_b.set_ylabel("SD of log$_{10}$ $N_\\mathrm{e}$")

    # c: SD ratio.
    ref_lo = float(np.median(first["sd_true_ci_lo"] / first["sd_true_log10"]))
    ref_hi = float(np.median(first["sd_true_ci_hi"] / first["sd_true_log10"]))
    # A bootstrap SD equal to the true SD gives sigma / sigma_hat, inside [ci_lo, ci_hi] / sigma_hat.
    ax_c.axhspan(ref_lo, ref_hi, color=REFERENCE_BAND, linewidth=0, zorder=0)
    style.ref_line(ax_c, y=1.0)
    top_c = max(C_TOP, 1.1 * float(np.nanmax(sd["ratio_median"])))
    over = []  # (years, upper quartile, colour) of the times whose upper quartile is off the top
    for t in trunks:
        s = sd[sd.trunk_bins == t].sort_values("point")
        color, ls = sty[t]
        if t == trunks[0]:  # one fill only: the other block size's quartiles are drawn as edges
            ax_c.fill_between(s["years"], s["ratio_q25"], s["ratio_q75"], color=color, alpha=style.BAND_ALPHA,
                              linewidth=0, zorder=1)
        else:
            for q in ("ratio_q25", "ratio_q75"):
                ax_c.plot(s["years"], s[q], color=color, linestyle=ls, linewidth=0.6, zorder=2)
        ax_c.plot(s["years"], s["ratio_median"], color=color, linestyle=ls, linewidth=style.LW_BOLD, zorder=3)
        over += [(r.years, r.ratio_q75, color) for r in s[s["ratio_q75"] > top_c].itertuples()]
    # The quartiles off the top (grid times where the fits' interval boundaries straddle the time): one mark at
    # the top edge per cluster of neighbouring times, both block sizes together, with the highest value.
    over.sort()
    clusters: List[List] = []
    for p in over:
        if clusters and p[0] / clusters[-1][-1][0] < 1.1:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    for cl in clusters:
        x, v, color = max(cl, key=lambda p: p[1])
        ax_c.plot(x, top_c, marker="^", markersize=3.0, color=color, clip_on=False, zorder=4)
        ax_c.annotate(f"{v:.1f}", (x, top_c), xytext=(3, -1), textcoords="offset points", ha="left", va="top",
                      fontsize=style.TICK_PT, color=style.INK2, annotation_clip=False)
    ax_c.set_ylim(0.0, top_c)
    ax_c.set_ylabel("Bootstrap SD / true SD")

    # d: coverage of the true Ne and of the fits' mean; e: of the projected
    # truth. The Clopper-Pearson interval of the default trunks only.
    nominal = cfg.band[1] - cfg.band[0]
    panels = [(ax_d, "coverage", "coverage_ci_lo", "coverage_ci_hi", "true $N_\\mathrm{e}$")]
    if proj is not None:
        panels.append((ax_e, proj["point"], proj["lo"], proj["hi"], "projected truth"))
    else:
        ax_e.set_axis_off()
    for ax, col, lo, hi, what in panels:
        style.ref_line(ax, y=nominal, linestyle=style.DOT)
        for t in trunks:
            c = cov[cov.trunk_bins == t].sort_values("point")
            color, ls = sty[t]
            if t == trunks[0] and lo is not None and hi is not None:
                ax.fill_between(c["years"], c[lo], c[hi], step="mid", color=color, alpha=CI_ALPHA, linewidth=0,
                                zorder=1)
            if ax is ax_d:
                ax.plot(c["years"], c["coverage_fit_mean"], drawstyle="steps-mid", color=color, linestyle=ls,
                        linewidth=style.LW_THIN, zorder=2)
            ax.plot(c["years"], c[col], drawstyle="steps-mid", color=color, linestyle=ls,
                    linewidth=style.LW_BOLD + 0.35, zorder=4)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(_percent())
        ax.set_ylabel(f"Coverage, {what}")
    for ax in (ax_b, ax_c, ax_d, ax_e):
        style.years_axis(ax, xlim)
        ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    fits = int(sampling["fits"].iloc[0])

    # Legends, top down: the shared entries, the trunk sizes, then the keys of c and d.
    ref_patch = Patch(facecolor=REFERENCE_BAND, edgecolor=style.MUTED, linewidth=0.4)
    shared: List = [
        style.handle(style.TRUTH, linewidth=style.LW),
        Patch(facecolor=GREY_BAND, linewidth=0, label=f"95% range of {fits} independent fits (a)"),
        Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Fit of one genome (a)"),
        (ref_patch, Line2D([], [], color=style.INK, linewidth=style.LW_BOLD)),
    ]
    shared_labels = [shared[0].get_label(), shared[1].get_label(), shared[2].get_label(),
                     "True sampling SD and its 95% interval (b, c)"]
    trunk_handles, trunk_labels = [], []
    for t in trunks:
        color, ls = sty[t]
        line = Line2D([], [], color=color, linestyle=ls, linewidth=style.LW_BOLD)
        trunk_handles.append(line)
        trunk_labels.append(coverage_label(rows[t], proj))
    key_ink = style.INK2
    keys: List = [Line2D([], [], color=key_ink, linewidth=style.LW_BOLD + 0.35),
                  Line2D([], [], color=key_ink, linewidth=style.LW_THIN)]
    n = int(cov["datasets"].max())
    covered = f"Share of the {n} genomes whose band covers the true $N_\\mathrm{{e}}$ (d)"
    key_labels = [covered + (" or the projected truth (e)" if proj is not None else ""),
                  "Share whose band covers the fits' mean (d)"]
    keys += [Patch(facecolor=sty[trunks[0]][0], alpha=CI_ALPHA * 2, linewidth=0),
             Line2D([], [], color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT),
             (Patch(facecolor=sty[trunks[0]][0], alpha=style.BAND_ALPHA, linewidth=0),
              Line2D([], [], color=sty[trunks[-1]][0], linestyle=sty[trunks[-1]][1], linewidth=0.6)),
             (Line2D([], [], color=BOOT_EDGE, linewidth=0.6),
              Line2D([], [], color=sty[trunks[-1]][0], linestyle=sty[trunks[-1]][1], linewidth=0.7))]
    mb = float(rows[trunks[0]].trunk_mb)
    key_labels += [f"Clopper-Pearson 95% interval of the thick {mb:g} Mb line (d, e)", "Nominal 95% (d, e)",
                   "Interquartile range over the genomes (c)", "The genome's bootstrap 95% band, its edges (a)"]

    from matplotlib.legend_handler import HandlerTuple

    height_mm = fig.get_figheight() / style.MM

    def place(handles, labels, y_mm: float, ncol: int, side_by_side: bool = False):
        # A tuple of handles is drawn overlaid (a band with its line), or side by side (one per trunk size).
        handler = {tuple: HandlerTuple(ndivide=None if side_by_side else 1, pad=0.0)}
        return style.legend(fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0 - y_mm / height_mm),
                            ncol=ncol, handler_map=handler)

    place(shared, shared_labels, 0.0, 4)
    place(trunk_handles, trunk_labels, 4.6, 1)
    place(keys, key_labels, 5.2 + 4.0 * len(trunks), 2, side_by_side=True)
    fig.align_ylabels([ax_a, ax_b, ax_d])
    fig.align_ylabels([ax_c, ax_e])
    style.panel_letters(fig, [ax_a, ax_b, ax_c, ax_d, ax_e], align_cols=True)
    return fig


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")


__all__ = ["coverage_label", "draw", "projected_columns", "trunk_label", "trunk_styles"]
