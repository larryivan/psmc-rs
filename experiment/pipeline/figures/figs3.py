"""Fig S3 (E6 S3): posterior TMRCA decoding against the true pairwise TMRCA.

One E1 replicate (``config.E6.tmrca_model``, ``tmrca_replicate``) decoded
by psmc-rs 0.2 with E1's fitted model (``--tmrca-out``); the true TMRCA of
every 100 bp bin comes from the tree sequence, re-simulated from the
replicate's seeds.

* a: an example window (the first 5 Mb of chr1): the true TMRCA (black),
  the posterior mean (blue) and its 95% interval (band); ticks mark the
  heterozygous bins;
* b: the posterior mean against the true TMRCA over every bin, as a 2-D
  histogram (sequential blue, log counts), with the identity line and the
  correlation of the logs. The grid runs from the youngest true TMRCA
  (1.5 ky) to the 99.95% quantile of both variables (2.8 My); the bins
  beyond it are counted in the right edge cells (true TMRCA, 0.10% of the
  bins) and the top ones (posterior mean, under 0.01%). Each vertical
  streak is a stretch of many bins that share one true TMRCA (one long
  genealogical segment, or a few): young coalescences are rare and their
  segments long, and the leftmost column, 0.4% of the bins, is the
  genome's youngest coalescence; along a streak the posterior moves across
  the segment's ends;
* c: share of bins whose 95% interval holds the true TMRCA, by decile of
  the true TMRCA (blue) and by decile of the posterior mean (yellow, when
  ``E6_S3_calibration`` has those rows): the interval of PSMC states from
  the 2.5% to the 97.5% posterior quantile (solid, filled circles), and the
  quantiles' state average times (dashed, open squares), against the
  nominal 95%. Grouped by the true TMRCA, the coverage of the states'
  interval is 95% to 98% in most deciles but 92% in decile 3 (33 to
  100 ky, just older than the bottleneck, whose times the posterior mean
  puts too old by 0.15 log10 on average), 90% in the oldest decile (too
  young by 0.13 log10) and 94% in the youngest; grouped by what the
  decoding sees, the posterior mean, it stays at 95% to 97%. One
  replicate's numbers (``E6_S3_calibration``).

Everything comes from ``E6_S3_track``, ``E6_S3_density``,
``E6_S3_calibration`` and ``E6_S3_summary``.
"""

from __future__ import annotations

from . import style
from .data import Source, decades

GROUPS = (("true", style.BLUE, "decile of the true TMRCA"),
          ("posterior_mean", style.YELLOW, "decile of the posterior mean"))  # grey is for keys and references
POINT_DASH = (0, (3.0, 1.5))  # c: the interval of the quantiles' state average times


def draw(src: Source):
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    track = src.table("E6_S3_track").sort_values(["sequence", "start_bp"])
    dens = src.table("E6_S3_density")
    calib_all = src.table("E6_S3_calibration")
    calib_all["by"] = calib_all["by"].astype(str).str.lower()  # a column of "true" alone reads as booleans
    groups = [g for g in GROUPS if (calib_all["by"] == g[0]).any()]
    summary = {r.bins: r for r in src.table("E6_S3_summary").itertuples()}
    overall = summary.get("all")
    bin_bp = float(np.median(np.diff(track["start_bp"]))) if len(track) > 1 else 100.0

    fig = style.figure("double", 118)
    gs = style.grid(fig, 2, 2, left_mm=14.0, right_mm=3.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=30.0,
                    hspace_mm=12.0, height_ratios=[0.75, 1.0], width_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # a: the example window.
    mb = np.append(track["start_bp"].to_numpy(), track["start_bp"].iloc[-1] + bin_bp) / 1e6

    def stepped(col):
        v = track[col].to_numpy(dtype=float)
        return np.append(v, v[-1])

    ax_a.fill_between(mb, stepped("q025_years"), stepped("q975_years"), step="post", color=style.BLUE,
                      alpha=style.BAND_ALPHA, linewidth=0, zorder=1)
    ax_a.plot(mb, stepped("posterior_mean_years"), drawstyle="steps-post", color=style.BLUE,
              linewidth=style.LW_THIN + 0.2, zorder=2)
    ax_a.plot(mb, stepped("true_years"), drawstyle="steps-post", color=style.INK, linewidth=style.LW_THIN + 0.2,
              zorder=3)
    lo = float(np.nanmin(track[["true_years", "q025_years"]].to_numpy()))
    hi = float(np.nanmax(track[["true_years", "q975_years"]].to_numpy()))
    ylim = decades(lo, hi)
    het = track[track["obs"] == "K"]
    ax_a.plot(het["start_bp"] / 1e6, np.full(len(het), ylim[0] * 1.3), linestyle="none", marker="|", markersize=3,
              markeredgewidth=0.4, color=style.REF, zorder=1)
    style.log_axis(ax_a, "y", grid=True, lim=ylim)
    ax_a.set_xlim(mb[0], mb[-1])
    ax_a.set_xlabel(f"Position on {track['sequence'].iloc[0]} (Mb)")
    ax_a.set_ylabel("TMRCA (years)")
    handles = [Line2D([], [], color=style.INK, linewidth=style.LW_DATA, label="True TMRCA"),
               Line2D([], [], color=style.BLUE, linewidth=style.LW_DATA, label="Posterior mean"),
               Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, label="95% posterior interval"),
               Line2D([], [], color=style.REF, linestyle="none", marker="|", markersize=4, markeredgewidth=0.6,
                      label="Heterozygous bin")]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)

    # b: posterior mean against truth.
    edges_x = np.unique(np.concatenate([dens["log10_true_lo"], dens["log10_true_hi"]]))
    edges_y = np.unique(np.concatenate([dens["log10_mean_lo"], dens["log10_mean_hi"]]))
    counts = np.full((len(edges_y) - 1, len(edges_x) - 1), np.nan)
    ix = np.searchsorted(edges_x, dens["log10_true_lo"].to_numpy())
    iy = np.searchsorted(edges_y, dens["log10_mean_lo"].to_numpy())
    counts[iy, ix] = dens["n_bins"].to_numpy(dtype=float)
    mesh = ax_b.pcolormesh(10**edges_x, 10**edges_y, counts, cmap=style.blue_cmap(),
                           norm=LogNorm(vmin=1.0, vmax=float(np.nanmax(counts))), shading="flat", rasterized=True)
    lims = (10 ** min(edges_x[0], edges_y[0]), 10 ** max(edges_x[-1], edges_y[-1]))
    ax_b.plot(lims, lims, color=style.INK, linewidth=style.LW_THIN, linestyle=style.DASH, zorder=3)
    style.log_axis(ax_b, "x", lim=lims)
    style.log_axis(ax_b, "y", lim=lims)
    ax_b.set_aspect("equal", anchor="W")  # square, flush left under a
    ax_b.set_xlabel("True TMRCA (years)")
    ax_b.set_ylabel("Posterior mean TMRCA (years)")
    if overall is not None:
        style.note(ax_b, 0.03, 0.97, f"r(log$_{{10}}$) = {float(overall.pearson_log10_mean):.2f}\n"
                                     f"{int(overall.n_bins):,} bins")
    style.note(ax_b, 0.97, 0.03, "right and top edge cells also\nhold the bins beyond the axes", ha="right",
               va="bottom")
    cax = ax_b.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(style.LW_HAIR)
    cb.ax.tick_params(labelsize=style.TICK_PT, width=style.LW_HAIR, length=2.0)
    cb.set_label("Bins per cell", fontsize=style.TEXT_PT, labelpad=2.0)

    # c: calibration of the 95% interval by decile of the true TMRCA and, if given, of the posterior mean.
    # Interval: solid line, filled circles (states) or dashed line, open squares (point); grouping: colour.
    style.ref_line(ax_c, y=0.95, linestyle=style.DOT)
    low = 1.0
    k = np.arange(1, 11)
    for i, (by, color, _) in enumerate(groups):
        calib = calib_all[calib_all["by"] == by].sort_values("class")
        k = calib["class"].to_numpy()
        z = 4 - i  # the true deciles on top
        ax_c.plot(k, calib["coverage95_states"], color=color, linewidth=style.LW_DATA, zorder=z, marker="o",
                  markersize=3.0, markeredgewidth=0.6)
        ax_c.plot(k, calib["coverage95_point"], color=color, linewidth=style.LW_DATA, linestyle=POINT_DASH, zorder=z,
                  marker="s", markersize=3.0, markeredgewidth=0.6, markerfacecolor=style.SURFACE)
        low = min(low, float(calib["coverage95_point"].min()), float(calib["coverage95_states"].min()))
    ax_c.set_xticks(k)
    ax_c.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    ax_c.set_ylim(min(0.75, np.floor((low - 0.02) * 20) / 20 - 0.05), 1.0)  # room below for the legend
    ax_c.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{100 * v:.0f}%"))
    style.value_grid(ax_c)
    ax_c.set_xlabel("Decile of the TMRCA (1 = youngest)")
    ax_c.set_ylabel("Coverage of the 95% interval")
    states, point = "States from the 2.5% to the 97.5% quantile", "Quantiles' state average times"
    if overall is not None:
        states += f" (all bins: {100 * float(overall.coverage95_states):.1f}%)"
        point += f" (all bins: {100 * float(overall.coverage95_point):.1f}%)"
    handles = [Line2D([], [], color=style.INK2, linewidth=style.LW_DATA, marker="o", markersize=3.0),
               Line2D([], [], color=style.INK2, linewidth=style.LW_DATA, linestyle=POINT_DASH, marker="s",
                      markersize=3.0, markerfacecolor=style.SURFACE)]
    labels = [states, point]
    for _, color, what in groups:
        handles.append(Line2D([], [], color=color, linewidth=style.LW_EMPH + 0.5))
        labels.append(f"By {what}")
    handles.append(Line2D([], [], color=style.REF, linewidth=style.LW_THIN, linestyle=style.DOT))
    labels.append("Nominal 95%")
    style.legend(ax_c, handles, labels, loc="lower left")
    style.panel_letters(fig, [ax_a, ax_b, ax_c])
    return fig


__all__ = ["draw"]
