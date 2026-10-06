"""Fig 6 (E5): how well psmc-rs's bootstrap measures the uncertainty of a PSMC curve.

The bottleneck model is simulated as many independent genomes of 6 x 50 Mb,
each fitted once: their spread at each time is the true sampling
distribution of the estimate. The first genomes are bootstrapped at two
trunk sizes, the default 50 Mb (blue) and 5 Mb (dark blue, dashed). Every
panel spans the model's informative window, where the calibration is
computed.

* a: one genome (``config.E5.example_dataset``): the truth (dashed), the
  95% range of the independent fits (grey), the genome's fit (blue line)
  and its bootstrap 95% bands (50 Mb trunks filled, 5 Mb trunks as dashed
  edges);
* b: SD of log10 Ne: the true sampling SD over the independent fits (black,
  with its chi-square 95% interval, light grey) and the bootstrap's (median
  over the bootstrapped genomes);
* c: bootstrap SD / true SD: median over the genomes and their
  interquartile range (filled in each trunk size's colour); the light grey
  band is where a bootstrap whose SD equalled the true one would fall,
  given the true SD's own sampling error;
* d: share of the bootstrapped genomes whose 95% band covers the true Ne
  (thick; with the pointwise Clopper-Pearson 95% interval of the 50 Mb
  trunks, the 5 Mb trunks' is left out so the panel stays legible) and the
  fits' geometric mean (thin), against the nominal 95%; with 10 genomes the
  share moves in steps of 10%. When E5's tables also give the coverage of
  the pattern-projected truth (the truth averaged over each free
  parameter's span, which a PSMC curve can represent), it is drawn too
  (medium, with markers).

The legend gives each trunk size's coverage averaged over the window (an
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

from typing import Dict, List, Optional, Tuple

from . import style
from .data import Source

GREY_BAND = "#c3c2b7"  # the independent fits' range (a)
REFERENCE_BAND = "#ebeae4"  # the true SD's sampling interval (b, c): lighter, and outlined in the legend
CI_ALPHA = 0.12  # Clopper-Pearson interval (d)
PROJECTED_LW = 0.9


def trunk_styles(trunks: List[int]) -> Dict[int, tuple]:
    """``{trunk: (colour, line style, marker)}``, the default (largest) trunk first."""
    return {t: (style.TRUNK_COLORS[min(i, 1)], style.TRUNK_STYLES[min(i, 1)], style.TRUNK_MARKERS[min(i, 1)])
            for i, t in enumerate(sorted(trunks, reverse=True))}


def trunk_label(row) -> str:
    return f"{float(row.trunk_mb):g} Mb trunks ({int(row.trunks)} per genome)"


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
    """A trunk size's legend label with its window coverages."""
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

    fig = style.figure("double", 142)
    gs = style.grid(fig, 2, 2, left_mm=14.0, right_mm=3.0, top_mm=26.0, bottom_mm=9.0, wspace_mm=16.0,
                    hspace_mm=11.0)
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(gs[i // 2, i % 2]) for i in range(4))

    # a: one genome.
    ds = cfg.example_dataset if cfg.example_dataset in set(bands["dataset"]) else int(bands["dataset"].min())
    ax_a.fill_between(years, sampling["lo_ne"], sampling["hi_ne"], color=GREY_BAND, alpha=0.6, linewidth=0, zorder=1)
    vals = [float(sampling["lo_ne"].min()), float(sampling["hi_ne"].max())]
    fit = None
    for t in trunks:
        b = bands[(bands.dataset == ds) & (bands.trunk_bins == t)].sort_values("point")
        color, ls, _ = sty[t]
        if t == trunks[0]:
            ax_a.fill_between(b["years"], b["boot_lo_ne"], b["boot_hi_ne"], color=color, alpha=style.BAND_ALPHA + 0.07,
                              linewidth=0, zorder=2)
        else:
            for edge in ("boot_lo_ne", "boot_hi_ne"):
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
        color, ls, _ = sty[t]
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
    hi = 1.5
    for t in trunks:
        s = sd[sd.trunk_bins == t].sort_values("point")
        color, ls, _ = sty[t]
        ax_c.fill_between(s["years"], s["ratio_q25"], s["ratio_q75"], color=color, alpha=style.BAND_ALPHA,
                          linewidth=0, zorder=1)
        ax_c.plot(s["years"], s["ratio_median"], color=color, linestyle=ls, linewidth=style.LW_BOLD, zorder=3)
        hi = max(hi, float(np.nanmax(s["ratio_q75"])))
    ax_c.set_ylim(0.0, min(hi * 1.05, 4.0))
    ax_c.set_ylabel("Bootstrap SD / true SD")

    # d: coverage; the Clopper-Pearson interval of the default trunks only.
    nominal = cfg.band[1] - cfg.band[0]
    style.ref_line(ax_d, y=nominal, linestyle=style.DOT)
    for t in trunks:
        c = cov[cov.trunk_bins == t].sort_values("point")
        color, ls, marker = sty[t]
        if t == trunks[0]:
            ax_d.fill_between(c["years"], c["coverage_ci_lo"], c["coverage_ci_hi"], step="mid", color=color,
                              alpha=CI_ALPHA, linewidth=0, zorder=1)
        ax_d.plot(c["years"], c["coverage_fit_mean"], drawstyle="steps-mid", color=color, linestyle=ls,
                  linewidth=style.LW_THIN, zorder=2)
        if proj is not None:
            ax_d.plot(c["years"], c[proj["point"]], drawstyle="steps-mid", color=color, linestyle=ls,
                      linewidth=PROJECTED_LW, marker=marker, markevery=_markevery(len(c), trunks.index(t)),
                      markersize=2.6, markeredgewidth=0.6, markerfacecolor=style.SURFACE, zorder=3)
        ax_d.plot(c["years"], c["coverage"], drawstyle="steps-mid", color=color, linestyle=ls,
                  linewidth=style.LW_BOLD + 0.35, zorder=4)
    ax_d.set_ylim(-0.03, 1.03)
    ax_d.yaxis.set_major_formatter(_percent())
    n = int(cov["datasets"].max())
    ax_d.set_ylabel(f"Coverage of the 95% band ({n} genomes)")
    for ax in (ax_b, ax_c, ax_d):
        style.years_axis(ax, xlim)
        ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    fits = int(sampling["fits"].iloc[0])

    # Legends, top down: the shared entries, the trunk sizes, then the keys of c and d.
    ref_patch = Patch(facecolor=REFERENCE_BAND, edgecolor=style.MUTED, linewidth=0.4)
    shared: List = [
        style.handle(style.TRUTH, linewidth=style.LW),
        Patch(facecolor=GREY_BAND, alpha=0.6, linewidth=0, label=f"95% range of {fits} independent fits (a)"),
        Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Fit of one genome (a)"),
        (ref_patch, Line2D([], [], color=style.INK, linewidth=style.LW_BOLD)),
    ]
    shared_labels = [shared[0].get_label(), shared[1].get_label(), shared[2].get_label(),
                     "True sampling SD and its 95% interval (b, c)"]
    trunk_handles, trunk_labels = [], []
    for t in trunks:
        color, ls, _ = sty[t]
        line = Line2D([], [], color=color, linestyle=ls, linewidth=style.LW_BOLD)
        trunk_handles.append((Patch(facecolor=color, alpha=style.BAND_ALPHA + 0.07, linewidth=0), line)
                             if t == trunks[0] else line)
        trunk_labels.append(coverage_label(rows[t], proj))
    key_ink = style.INK2
    keys: List = [Line2D([], [], color=key_ink, linewidth=style.LW_BOLD + 0.35),
                  Line2D([], [], color=key_ink, linewidth=style.LW_THIN)]
    key_labels = ["Covers the true $N_\\mathrm{e}$ (d)", "Covers the fits' mean (d)"]
    if proj is not None:
        keys.append(Line2D([], [], color=key_ink, linewidth=PROJECTED_LW, marker="o", markersize=2.6,
                           markeredgewidth=0.6, markerfacecolor=style.SURFACE))
        key_labels.append("Covers the projected truth (d)")
    keys += [Patch(facecolor=sty[trunks[0]][0], alpha=CI_ALPHA * 2, linewidth=0),
             Line2D([], [], color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT),
             (Patch(facecolor=sty[trunks[0]][0], alpha=style.BAND_ALPHA, linewidth=0),
              Patch(facecolor=sty[trunks[-1]][0], alpha=style.BAND_ALPHA, linewidth=0))]
    key_labels += [f"Its Clopper-Pearson 95% interval, {float(rows[trunks[0]].trunk_mb):g} Mb (d)", "Nominal 95% (d)",
                   "Interquartile range over the genomes (c)"]

    from matplotlib.legend_handler import HandlerTuple

    height_mm = fig.get_figheight() / style.MM

    def place(handles, labels, y_mm: float, ncol: int, side_by_side: bool = False):
        # A tuple of handles is drawn overlaid (a band with its line), or side by side (one per trunk size).
        handler = {tuple: HandlerTuple(ndivide=None if side_by_side else 1, pad=0.0)}
        return style.legend(fig, handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0 - y_mm / height_mm),
                            ncol=ncol, handler_map=handler)

    place(shared, shared_labels, 0.0, 4)
    place(trunk_handles, trunk_labels, 4.6, 1)
    place(keys, key_labels, 5.2 + 4.0 * len(trunks), 3, side_by_side=True)
    fig.align_ylabels([ax_a, ax_c])
    fig.align_ylabels([ax_b, ax_d])
    style.panel_letters(fig, [ax_a, ax_b, ax_c, ax_d], align_cols=True)
    return fig


def _markevery(points: int, i: int) -> Tuple[int, int]:
    """Markers every 20th time, offset by trunk size so the two sizes' markers do not overlap."""
    return (5 + 10 * (i % 2), 20) if points > 40 else (0, 1)


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")


__all__ = ["coverage_label", "draw", "projected_columns", "trunk_label", "trunk_styles"]
