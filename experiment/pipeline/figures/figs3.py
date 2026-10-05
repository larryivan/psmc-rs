"""Fig S3 (E6): posterior TMRCA decoding against the true pairwise TMRCA.

One E1 replicate (config.E6.tmrca_model, tmrca_replicate) fitted by psmc-rs
0.2 with ``--tmrca-out``; the true TMRCA comes from the tree sequence.

* a: an example window: the true TMRCA at each 100 bp bin (black), the
  posterior mean (blue) and the 95% posterior interval (band);
* b, c: posterior mean and MAP against the true TMRCA over every bin, as a
  2-D histogram (sequential blue, log counts), with the identity line and
  the correlation of the logs;
* d: share of bins whose 95% posterior interval contains the true TMRCA, by
  class of true TMRCA, with the nominal 95% and the overall share.
"""

from __future__ import annotations

from . import style
from .data import Source, decades


def draw(src: Source):
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    track = src.table("E6_tmrca_track").sort_values("start_bp")
    hist = src.table("E6_tmrca_hist2d")
    calib = src.table("E6_tmrca_calibration").sort_values("true_lo_years")
    summary = {r.estimator: r for r in src.table("E6_tmrca_summary").itertuples()}
    bin_bp = float(np.median(np.diff(track["start_bp"]))) if len(track) > 1 else 100.0

    fig = style.figure("double", 112)
    gs = style.grid(fig, 2, 3, left_mm=13.0, right_mm=12.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=14.0,
                    hspace_mm=12.0, height_ratios=[0.8, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b, ax_c, ax_d = (fig.add_subplot(gs[1, i]) for i in range(3))

    # a: the example window.
    mb = np.append(track["start_bp"].to_numpy(), track["start_bp"].iloc[-1] + bin_bp) / 1e6

    def stepped(col):
        v = track[col].to_numpy(dtype=float)
        return np.append(v, v[-1])

    ax_a.fill_between(mb, stepped("q025_years"), stepped("q975_years"), step="post", color=style.BLUE,
                      alpha=style.BAND_ALPHA, linewidth=0, zorder=1)
    ax_a.plot(mb, stepped("post_mean_years"), drawstyle="steps-post", color=style.BLUE, linewidth=style.LW_THIN + 0.2,
              zorder=2)
    ax_a.plot(mb, stepped("true_tmrca_years"), drawstyle="steps-post", color=style.INK, linewidth=style.LW_THIN + 0.2,
              zorder=3)
    lo = float(np.nanmin(track[["true_tmrca_years", "q025_years"]].to_numpy()))
    hi = float(np.nanmax(track[["true_tmrca_years", "q975_years"]].to_numpy()))
    style.log_axis(ax_a, "y", grid=True)
    ax_a.set_ylim(*decades(lo, hi))
    ax_a.set_xlim(mb[0], mb[-1])
    ax_a.set_xlabel("Position (Mb)")
    ax_a.set_ylabel("TMRCA (years)")
    handles = [Line2D([], [], color=style.INK, linewidth=style.LW, label="True TMRCA"),
               Line2D([], [], color=style.BLUE, linewidth=style.LW, label="Posterior mean"),
               Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0, label="95% posterior interval")]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)

    # b, c: estimate against truth.
    norm = None
    cmap = style.blue_cmap()
    lims = decades(float(min(hist["true_lo_years"].min(), hist["est_lo_years"].min())),
                   float(max(hist["true_hi_years"].max(), hist["est_hi_years"].max())))
    mesh = None
    for ax, est, label in ((ax_b, "mean", "Posterior mean"), (ax_c, "map", "Posterior MAP")):
        sub = hist[hist.estimator == est]
        if sub.empty:
            ax.set_axis_off()
            continue
        tx = np.unique(np.concatenate([sub["true_lo_years"], sub["true_hi_years"]]))
        ey = np.unique(np.concatenate([sub["est_lo_years"], sub["est_hi_years"]]))
        counts = np.full((len(ey) - 1, len(tx) - 1), np.nan)
        ti = np.searchsorted(tx, sub["true_lo_years"].to_numpy())
        ei = np.searchsorted(ey, sub["est_lo_years"].to_numpy())
        counts[ei, ti] = sub["count"].to_numpy(dtype=float)
        if norm is None:
            norm = LogNorm(vmin=1.0, vmax=float(hist["count"].max()))
        mesh = ax.pcolormesh(tx, ey, counts, cmap=cmap, norm=norm, shading="flat", rasterized=True)
        ax.plot(lims, lims, color=style.INK, linewidth=style.LW_REF, linestyle=style.DASH, zorder=3)
        style.log_axis(ax, "x")
        style.log_axis(ax, "y")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal")
        ax.set_xlabel("True TMRCA (years)")
        ax.set_ylabel(f"{label} (years)")
        s = summary.get(est)
        if s is not None:
            style.note(ax, 0.03, 0.97, f"r(log$_{{10}}$) = {float(s.pearson_log10):.2f}\n{int(s.bins):,} bins",
                       ha="left", va="top")
    if mesh is not None:
        cax = ax_c.inset_axes([0.56, 0.12, 0.38, 0.045])  # lower right: estimates far below the truth are rare
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(style.LW_GRID)
        cb.ax.tick_params(labelsize=style.TICK_PT, width=style.LW_GRID, length=2.0)
        cb.set_label("Bins per cell", fontsize=style.TICK_PT, labelpad=1.5)
        cb.ax.xaxis.set_label_position("top")

    # d: calibration of the 95% interval.
    mid = np.sqrt(calib["true_lo_years"].to_numpy() * calib["true_hi_years"].to_numpy())
    style.ref_line(ax_d, y=0.95, linestyle=style.DOT)
    ax_d.plot(mid, calib["coverage"], color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_d.plot(mid, calib["coverage"], **style.RS02.points(markersize=3.0, zorder=4))
    style.log_axis(ax_d, "x", grid=False)
    ax_d.set_xlim(*decades(float(calib["true_lo_years"].min()), float(calib["true_hi_years"].max())))
    ax_d.set_ylim(0.0, 1.02)
    ax_d.yaxis.set_major_formatter(_percent())
    ax_d.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    ax_d.set_xlabel("True TMRCA (years)")
    ax_d.set_ylabel("Coverage of the 95% interval")
    overall = summary.get("mean")
    text = "dotted: nominal 95%"
    if overall is not None:
        text = f"all bins: {100 * float(overall.coverage):.1f}%\n" + text
    style.note(ax_d, 0.97, 0.03, text, ha="right", va="bottom")
    style.panel_letters(fig, [ax_a, ax_b, ax_c, ax_d])
    return fig


def _percent():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _p: f"{100 * v:.0f}%")
