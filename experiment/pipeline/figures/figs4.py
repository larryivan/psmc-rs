"""Fig S4 (E3): thread scaling and determinism of psmc-rs 0.2.

On the thread-scaling genome (3,000 Mb, 25 EM iterations, E3's clean runs):

* a: wall time against threads: every repeat (dots) and the median (line),
  with ideal scaling from one thread (dotted); psmc-rs 0.2 with capped
  M-steps (dark blue) at its thread counts when E3 timed it (not by
  default). The M-step runs on one thread but converges in a few ms per EM
  iteration, so the E-step's sequences set the speed-up.
* b: parallel efficiency, speed-up / threads, of the medians.
* c: peak memory of the process tree against threads.
* d: determinism: for every tool and genome size, how many runs (thread
  counts and repeats) were made and how many distinct outputs they wrote.

Everything comes from ``E3_thread_scaling``, ``E3_table2``, ``E3_runs``
and ``E3_determinism``.
"""

from __future__ import annotations

from . import style
from .data import Source
from .fig4 import MEM_TICKS, TIME_TICKS, TOOL_ENTITY, _formatter, decade_span, fmt_mib, fmt_seconds, log_ticks


def thread_axis(ax, t) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(list(t)))
    ax.xaxis.set_major_formatter(_formatter(lambda v: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(t) / 1.3, max(t) * 1.3)
    ax.set_xlabel("Threads")


def draw(src: Source):
    scaling = src.table("E3_thread_scaling").sort_values("threads")
    table2 = src.table("E3_table2")
    runs = src.table("E3_runs")
    det = src.table("E3_determinism")
    size = int(scaling["size_mb"].iloc[0])
    t = scaling["threads"].to_numpy(dtype=float)
    wall = scaling["wall_median_s"].to_numpy(dtype=float)
    reps = runs[(runs.tool == "rs02") & (runs.size_mb == size) & (runs["clean"].astype(str) == "True")]
    default = table2[(table2.tool == "rs02-default") & (table2.size_mb == size)].sort_values("threads")
    default = default[default["wall_median_s"].notna()]
    dent = TOOL_ENTITY["rs02-default"]

    fig = style.figure("double", 62)
    gs = style.grid(fig, 1, 4, left_mm=12.0, right_mm=2.0, top_mm=9.0, bottom_mm=9.0, wspace_mm=13.0,
                    width_ratios=[1.0, 1.0, 1.0, 1.3])
    ax_t, ax_e, ax_m, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))

    ax_t.plot(t, wall[0] * t[0] / t, color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=1)
    ax_t.annotate("ideal", (t[-1], wall[0] * t[0] / t[-1]), xytext=(0, -4), textcoords="offset points",
                  ha="center", va="top", fontsize=style.TICK_PT, color=style.MUTED)
    ax_t.plot(t, wall, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_t.plot(reps["threads"], reps["wall_seconds"], linestyle="none", marker=".", markersize=2.0, color=style.BLUE,
              zorder=4)
    ax_t.plot(t, wall, **style.RS02.points(markersize=3.2, zorder=3))
    walls = list(reps["wall_seconds"]) + list(wall) + list(wall[0] * t[0] / t)
    if not default.empty:
        ax_t.plot(default["threads"], default["wall_median_s"], color=dent.color, linewidth=style.LW, zorder=3)
        ax_t.plot(default["threads"], default["wall_median_s"], **dent.points(markersize=3.2, zorder=4))
        walls += list(default["wall_median_s"])
    thread_axis(ax_t, t)
    log_ticks(ax_t, *decade_span(walls), TIME_TICKS, fmt_seconds)
    ax_t.set_ylabel(f"Wall time, {size:,} Mb")
    ax_t.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    eff = scaling["efficiency"].to_numpy(dtype=float)
    style.ref_line(ax_e, y=1.0, linestyle=style.DOT)
    ax_e.plot(t, eff, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_e.plot(t, eff, **style.RS02.points(markersize=3.2, zorder=4))
    for x, e in zip(t, eff):
        ax_e.annotate(f"{100 * e:.0f}%", (x, e), xytext=(0, -5), textcoords="offset points", ha="center", va="top",
                      fontsize=style.TICK_PT, color=style.INK2)
    if len(default) > 1:
        d = default["wall_median_s"].to_numpy(dtype=float)
        dt = default["threads"].to_numpy(dtype=float)
        deff = d[0] * dt[0] / d / dt
        ax_e.plot(dt, deff, color=dent.color, linewidth=style.LW, zorder=3)
        ax_e.plot(dt, deff, **dent.points(markersize=3.2, zorder=4))
    thread_axis(ax_e, t)
    ax_e.set_ylim(0.0, 1.1)
    ax_e.set_ylabel("Parallel efficiency")
    ax_e.yaxis.set_major_formatter(_formatter(lambda v: f"{100 * v:.0f}%"))
    ax_e.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    rss = scaling["peak_rss_median_mb"].to_numpy(dtype=float)
    ax_m.plot(t, rss, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_m.plot(reps["threads"], reps["peak_rss_mb"], linestyle="none", marker=".", markersize=2.0, color=style.BLUE,
              zorder=4)
    ax_m.plot(t, rss, **style.RS02.points(markersize=3.2, zorder=3))
    thread_axis(ax_m, t)
    log_ticks(ax_m, *decade_span(list(reps["peak_rss_mb"]) + list(rss)), MEM_TICKS, fmt_mib)
    ax_m.set_ylabel("Peak memory (RSS)")
    ax_m.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    determinism_panel(ax_d, det)
    handles = [style.handle(style.RS02, kind="both", linewidth=style.LW, label="psmc-rs 0.2"),
               style.handle(dent, kind="both", linewidth=style.LW)]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style.panel_letters(fig, [ax_t, ax_e, ax_m, ax_d])
    return fig


def determinism_panel(ax, det) -> None:
    """Per tool: runs (thread counts and repeats at every size) and the distinct outputs per size."""
    ax.set_axis_off()
    style.note(ax, 0.0, 0.99, "Outputs of repeated runs", color=style.INK, size=style.TEXT_PT)
    y = 0.86
    cols = (0.0, 0.68, 0.86)
    for x, head in zip(cols, ("Tool (threads)", "Runs", "Outputs\nper size")):
        style.note(ax, x, y, head, color=style.INK2, linespacing=1.1)
    y -= 0.1
    ax.plot([0.0, 1.0], [y + 0.015, y + 0.015], color=style.GRID, linewidth=style.LW_GRID, transform=ax.transAxes,
            clip_on=False)
    labels = {"c": "C psmc", "rs01": "psmc-rs 0.1", "rs02": "psmc-rs 0.2", "rs02-default": "0.2, capped M-step"}
    for tool in [k for k in labels if k in set(det["tool"])]:
        sub = det[det.tool == tool]
        threads = sorted(int(x) for x in sub["threads"].unique())
        span = f"{threads[0]}-{threads[-1]}" if len(threads) > 1 else str(threads[0])
        per_size = sub.groupby("size_mb")["sha256"].nunique()
        outputs = str(int(per_size.max())) if per_size.min() == per_size.max() else f"{per_size.min()}-{per_size.max()}"
        y -= 0.1
        for x, text in zip(cols, (f"{labels[tool]} ({span})", str(len(sub)), outputs)):
            style.note(ax, x, y, text, color=style.INK)
    style.note(ax, 0.0, y - 0.13, "1: every run at a size wrote\nthe same bytes", color=style.INK2, linespacing=1.1)


__all__ = ["determinism_panel", "draw"]
