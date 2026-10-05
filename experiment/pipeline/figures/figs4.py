"""Fig S4 (E3): thread scaling and determinism of psmc-rs 0.2.

On the thread-scaling genome (3,000 Mb, 25 EM iterations):

* a: wall time against threads: every repeat (dots) and the median (line),
  with ideal scaling from the median with the fewest threads (dotted).
* b: parallel efficiency, speedup / threads, of the medians.
* c: peak memory of the process tree against threads.
* d: the SHA-256 of the output JSON of every run: one digest for every
  thread count and repeat means the results do not depend on the threads.
"""

from __future__ import annotations

from . import style
from .data import Source
from .fig4 import MEM_TICKS, TIME_TICKS, _formatter, decade_span, fmt_mib, fmt_seconds, log_ticks


def thread_axis(ax, t) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(list(t)))
    ax.xaxis.set_major_formatter(_formatter(lambda v: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(t) / 1.3, max(t) * 1.3)
    ax.set_xlabel("Threads")


def draw(src: Source):
    import numpy as np

    df = src.table("E3_threads").sort_values(["threads", "repeat"])
    t = np.array(sorted(df["threads"].unique()), dtype=float)
    med = df.groupby("threads")[["wall_seconds", "peak_rss_mb"]].median().sort_index()
    size = int(df["genome_mb"].iloc[0])

    fig = style.figure("double", 60)
    gs = style.grid(fig, 1, 4, left_mm=12.0, right_mm=2.0, top_mm=5.0, bottom_mm=9.0, wspace_mm=13.0,
                    width_ratios=[1.0, 1.0, 1.0, 1.25])
    ax_t, ax_e, ax_m, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))

    wall = med["wall_seconds"].to_numpy()
    ax_t.plot(t, wall[0] * t[0] / t, color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=1)
    ax_t.annotate("ideal", (t[-1], wall[0] * t[0] / t[-1]), xytext=(0, -4), textcoords="offset points",
                  ha="center", va="top", fontsize=style.TICK_PT, color=style.MUTED)
    ax_t.plot(t, wall, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_t.plot(df["threads"], df["wall_seconds"], linestyle="none", marker=".", markersize=2.0, color=style.BLUE,
              zorder=4)
    ax_t.plot(t, wall, **style.RS02.points(markersize=3.2, markerfacecolor=style.SURFACE, zorder=3))
    thread_axis(ax_t, t)
    log_ticks(ax_t, *decade_span(list(df["wall_seconds"]) + list(wall[0] * t[0] / t)), TIME_TICKS, fmt_seconds)
    ax_t.set_ylabel(f"Wall time, {size:,} Mb")
    ax_t.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    eff = wall[0] * t[0] / wall / t
    style.ref_line(ax_e, y=1.0, linestyle=style.DOT)
    ax_e.plot(t, eff, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_e.plot(t, eff, **style.RS02.points(markersize=3.2, zorder=4))
    for x, e in zip(t, eff):
        ax_e.annotate(f"{100 * e:.0f}%", (x, e), xytext=(0, -5), textcoords="offset points", ha="center",
                      va="top", fontsize=style.TICK_PT, color=style.INK2)
    thread_axis(ax_e, t)
    ax_e.set_ylim(0.0, 1.1)
    ax_e.set_ylabel("Parallel efficiency")
    ax_e.yaxis.set_major_formatter(_formatter(lambda v: f"{100 * v:.0f}%"))
    ax_e.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    ax_m.plot(t, med["peak_rss_mb"].to_numpy(), color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_m.plot(df["threads"], df["peak_rss_mb"], linestyle="none", marker=".", markersize=2.0, color=style.BLUE,
              zorder=4)
    ax_m.plot(t, med["peak_rss_mb"].to_numpy(), **style.RS02.points(markersize=3.2, markerfacecolor=style.SURFACE,
                                                                     zorder=3))
    thread_axis(ax_m, t)
    log_ticks(ax_m, *decade_span(df["peak_rss_mb"]), MEM_TICKS, fmt_mib)
    ax_m.set_ylabel("Peak memory (RSS)")
    ax_m.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    determinism_panel(ax_d, df)
    style.panel_letters(fig, [ax_t, ax_e, ax_m, ax_d])
    return fig


def determinism_panel(ax, df) -> None:
    """The output digests by thread count, as a small table."""
    ax.set_axis_off()
    digests = sorted(df["json_sha256"].astype(str).unique())
    letter = {d: chr(ord("A") + i) for i, d in enumerate(digests)}
    rows = []
    for t, sub in df.groupby("threads", sort=True):
        ds = sorted(sub["json_sha256"].astype(str).unique())
        rows.append((int(t), len(sub), ", ".join(letter[d] for d in ds)))
    y = 0.98
    style.note(ax, 0.0, y, "Output JSON (SHA-256)", color=style.INK, size=style.TEXT_PT)
    y -= 0.11
    style.note(ax, 0.0, y, "Threads", color=style.INK2)
    style.note(ax, 0.32, y, "Runs", color=style.INK2)
    style.note(ax, 0.56, y, "Digest", color=style.INK2)
    ax.plot([0.0, 1.0], [y - 0.085, y - 0.085], color=style.GRID, linewidth=style.LW_GRID, transform=ax.transAxes,
            clip_on=False)
    for t, n, ds in rows:
        y -= 0.1
        style.note(ax, 0.0, y, f"{t}", color=style.INK)
        style.note(ax, 0.32, y, f"{n}", color=style.INK)
        style.note(ax, 0.56, y, ds, color=style.INK)
    y -= 0.14
    for d in digests:
        style.note(ax, 0.0, y, f"{letter[d]}: {d[:16]}…", color=style.INK2)
        y -= 0.1
    verdict = "identical for every thread count" if len(digests) == 1 else f"{len(digests)} different outputs"
    style.note(ax, 0.0, y - 0.02, verdict, color=style.INK, size=style.TEXT_PT)
