"""Fig 6 (E5): how well psmc-rs's bootstrap measures the uncertainty of a PSMC curve.

The bottleneck model is simulated as many independent genomes of 6 x 50 Mb,
each fitted once: their spread at each time is the true sampling
distribution of the estimate. The first genomes are bootstrapped at two
trunk sizes, the default 50 Mb (blue) and 5 Mb (dark blue, dashed). Every
panel spans the model's informative window, where the calibration is
computed.

* a: one genome (``config.E5.example_dataset``): the truth (dashed), the
  95% range of the independent fits (grey), the genome's fit (blue line)
  and its bootstrap 95% bands;
* b: SD of log10 Ne: the true sampling SD over the independent fits (black,
  with its chi-square 95% interval) and the bootstrap's (median over the
  bootstrapped genomes);
* c: bootstrap SD / true SD: median over the genomes and their
  interquartile range; the grey band is where a bootstrap whose SD equalled
  the true one would fall, given the true SD's own sampling error;
* d: share of the bootstrapped genomes whose 95% band covers the true Ne
  (thick, with its pointwise Clopper-Pearson 95% interval) and the fits'
  geometric mean (thin), against the nominal 95%; with 10 genomes the
  share moves in steps of 10%. The window-averaged coverage is given with
  its interval from resampling the genomes (``E5_summary``).

Everything comes from E5's tables: ``E5_bands``, ``E5_sampling``,
``E5_sd_ratio``, ``E5_coverage`` and ``E5_summary``.
"""

from __future__ import annotations

from typing import Dict, List

from . import style
from .data import Source

GREY_BAND = "#c3c2b7"
REFERENCE_BAND = "#e1e0d9"


def trunk_styles(trunks: List[int]) -> Dict[int, tuple]:
    """``{trunk: (colour, line style, marker)}``, the default (largest) trunk first."""
    return {t: (style.TRUNK_COLORS[min(i, 1)], style.TRUNK_STYLES[min(i, 1)], style.TRUNK_MARKERS[min(i, 1)])
            for i, t in enumerate(sorted(trunks, reverse=True))}


def trunk_label(row) -> str:
    return f"{float(row.trunk_mb):g} Mb trunks ({int(row.trunks)} per genome)"


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
    trunks = sorted((int(t) for t in informative["trunk_bins"]), reverse=True)
    sty = trunk_styles(trunks)
    rows = {int(r.trunk_bins): r for r in informative.itertuples()}
    years = sampling["years"].to_numpy()
    xlim = (float(years[0]) / 1.12, float(years[-1]) * 1.12)

    fig = style.figure("double", 132)
    gs = style.grid(fig, 2, 2, left_mm=14.0, right_mm=3.0, top_mm=19.0, bottom_mm=9.0, wspace_mm=16.0,
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
    ax_b.set_ylim(0.0, top * 1.15)
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
    ax_c.set_ylim(0.0, min(hi * 1.1, 4.0))
    ax_c.set_ylabel("Bootstrap SD / true SD")

    # d: coverage.
    nominal = cfg.band[1] - cfg.band[0]
    style.ref_line(ax_d, y=nominal, linestyle=style.DOT)
    for t in trunks:
        c = cov[cov.trunk_bins == t].sort_values("point")
        color, ls, _ = sty[t]
        ax_d.fill_between(c["years"], c["coverage_ci_lo"], c["coverage_ci_hi"], step="mid", color=color,
                          alpha=0.09, linewidth=0, zorder=1)
        ax_d.plot(c["years"], c["coverage_fit_mean"], drawstyle="steps-mid", color=color, linestyle=ls,
                  linewidth=style.LW_THIN, zorder=2)
        ax_d.plot(c["years"], c["coverage"], drawstyle="steps-mid", color=color, linestyle=ls,
                  linewidth=style.LW_BOLD, zorder=3)
    ax_d.set_ylim(-0.03, 1.03)
    ax_d.yaxis.set_major_formatter(_percent())
    n = int(cov["datasets"].max())
    ax_d.set_ylabel(f"Coverage of the true $N_\\mathrm{{e}}$ ({n} genomes)")
    for ax in (ax_b, ax_c, ax_d):
        style.years_axis(ax, xlim)
        ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    fits = int(sampling["fits"].iloc[0])

    handles: List = [
        style.handle(style.TRUTH, linewidth=style.LW),
        Patch(facecolor=GREY_BAND, alpha=0.6, linewidth=0, label=f"95% range of {fits} independent fits"),
        Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Fit of one genome"),
        Line2D([], [], color=style.INK, linewidth=style.LW_BOLD, label="True sampling SD"),
        Patch(facecolor=REFERENCE_BAND, linewidth=0, label="Its sampling error (b, c)"),
    ]
    for t in trunks:
        color, ls, _ = sty[t]
        r = rows[t]
        handles.append(Line2D([], [], color=color, linestyle=ls, linewidth=style.LW_BOLD,
                              label=f"Bootstrap, {trunk_label(r)}: coverage over the window "
                                    f"{100 * float(r.coverage_mean):.0f}% ({100 * float(r.coverage_mean_ci_lo):.0f}"
                                    f"-{100 * float(r.coverage_mean_ci_hi):.0f}%)"))
    style.legend(fig, handles[:5], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5)
    fig.legend(handles[5:], [h.get_label() for h in handles[5:]], loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - 4.2 / (fig.get_figheight() * 25.4)), ncol=1, frameon=False,
               fontsize=style.TEXT_PT, handlelength=2.4)
    style.panel_letters(fig, [ax_a, ax_b, ax_c, ax_d])
    return fig


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")


__all__ = ["draw", "trunk_label", "trunk_styles"]
