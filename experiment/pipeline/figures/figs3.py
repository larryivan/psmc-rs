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
  correlation of the logs;
* c: share of bins whose 95% interval holds the true TMRCA, by decile of
  the true TMRCA: the interval of PSMC states from the 2.5% to the 97.5%
  posterior quantile (solid), and the quantiles' state average times
  (dashed), against the nominal 95%.

Everything comes from ``E6_S3_track``, ``E6_S3_density``,
``E6_S3_calibration`` and ``E6_S3_summary``.
"""

from __future__ import annotations

from . import style
from .data import Source, decades


def draw(src: Source):
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    track = src.table("E6_S3_track").sort_values(["sequence", "start_bp"])
    dens = src.table("E6_S3_density")
    calib = src.table("E6_S3_calibration")
    calib = calib[calib["by"] == "true"].sort_values("class")
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
              markeredgewidth=0.4, color=style.MUTED, zorder=1)
    style.log_axis(ax_a, "y", grid=True)
    ax_a.set_ylim(*ylim)
    ax_a.set_xlim(mb[0], mb[-1])
    ax_a.set_xlabel(f"Position on {track['sequence'].iloc[0]} (Mb)")
    ax_a.set_ylabel("TMRCA (years)")
    handles = [Line2D([], [], color=style.INK, linewidth=style.LW, label="True TMRCA"),
               Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Posterior mean"),
               Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, label="95% posterior interval"),
               Line2D([], [], color=style.MUTED, linestyle="none", marker="|", markersize=4, markeredgewidth=0.6,
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
    ax_b.plot(lims, lims, color=style.INK, linewidth=style.LW_REF, linestyle=style.DASH, zorder=3)
    style.log_axis(ax_b, "x")
    style.log_axis(ax_b, "y")
    ax_b.set_xlim(*lims)
    ax_b.set_ylim(*lims)
    ax_b.set_aspect("equal")
    ax_b.set_xlabel("True TMRCA (years)")
    ax_b.set_ylabel("Posterior mean TMRCA (years)")
    if overall is not None:
        style.note(ax_b, 0.03, 0.97, f"r(log$_{{10}}$) = {float(overall.pearson_log10_mean):.2f}\n"
                                     f"{int(overall.n_bins):,} bins", ha="left", va="top")
    cax = ax_b.inset_axes([1.04, 0.0, 0.05, 1.0])
    cb = fig.colorbar(mesh, cax=cax)
    cb.outline.set_linewidth(style.LW_GRID)
    cb.ax.tick_params(labelsize=style.TICK_PT, width=style.LW_GRID, length=2.0)
    cb.set_label("Bins per cell", fontsize=style.TICK_PT, labelpad=2.0)

    # c: calibration of the 95% interval by decile of the true TMRCA.
    k = calib["class"].to_numpy()
    style.ref_line(ax_c, y=0.95, linestyle=style.DOT)
    ax_c.plot(k, calib["coverage95_states"], color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_c.plot(k, calib["coverage95_states"], **style.RS02.points(markersize=3.0, zorder=4))
    ax_c.plot(k, calib["coverage95_point"], color=style.BLUE_650, linewidth=style.LW, linestyle=(0, (3.0, 1.5)),
              zorder=3)
    ax_c.plot(k, calib["coverage95_point"], **style.Entity("", style.BLUE_650, "-", "s", filled=False).points(
        markersize=3.0, zorder=4, label=None))
    ax_c.set_xticks(k)
    ax_c.set_xticklabels([f"{y / 1e3:,.0f}" for y in calib["median_true_years"]], fontsize=style.TICK_PT)
    ax_c.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    low = float(min(calib["coverage95_point"].min(), calib["coverage95_states"].min()))
    ax_c.set_ylim(min(0.8, np.floor((low - 0.02) * 20) / 20), 1.0)
    from matplotlib.ticker import MultipleLocator

    ax_c.yaxis.set_major_locator(MultipleLocator(0.05))
    ax_c.yaxis.set_major_formatter(_percent())
    ax_c.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    ax_c.set_xlabel("Decile of the true TMRCA (median, thousand years)")
    ax_c.set_ylabel("Coverage of the 95% interval")
    states, point = "States from the 2.5% to the 97.5% quantile", "Quantiles' state average times"
    if overall is not None:
        states += f" (all bins: {100 * float(overall.coverage95_states):.1f}%)"
        point += f" (all bins: {100 * float(overall.coverage95_point):.1f}%)"
    ax_c.legend([Line2D([], [], color=style.BLUE, linewidth=style.LW, marker="o", markersize=3.0),
                 Line2D([], [], color=style.BLUE_650, linewidth=style.LW, linestyle=(0, (3.0, 1.5)), marker="s",
                        markersize=3.0, markerfacecolor=style.SURFACE),
                 Line2D([], [], color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT)],
                [states, point, "Nominal 95%"], loc="lower left", frameon=False, fontsize=style.TICK_PT,
                handlelength=2.4)
    style.panel_letters(fig, [ax_a, ax_b, ax_c])
    return fig


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")


__all__ = ["draw"]
