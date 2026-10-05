"""Fig 4 (E3): run time and peak memory against genome size, and thread scaling.

* a: wall time of 25 EM iterations against genome size (100 Mb chromosomes),
  for C psmc, psmc-rs 0.1 and psmc-rs 0.2 with 1 and 8 threads: the median
  of the repeats (line and marker) and each repeat (small dots).
* b: peak resident memory of the whole process tree, same runs.
* c: speedup of psmc-rs 0.2 on the thread-scaling genome against threads
  (median wall time with 1 thread / with n threads; dots: each repeat), with
  ideal linear scaling for reference.

Sizes at which a tool was not run (C psmc above its memory limit) are named
in the panel, with the reason the table gives.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from . import style
from .data import Source

DASHED = (0, (3.0, 1.5))


def series(runs) -> List[Tuple[str, int, style.Entity]]:
    """``(tool, threads, entity)`` of every series in the table, in the legend's order.

    C psmc and psmc-rs 0.1 run with one thread; psmc-rs 0.2's thread counts
    are those of the table, the first drawn solid with filled markers, the
    others dashed with open markers.
    """
    out = []
    for tool, ent in (("c", style.C_PSMC), ("rs01", style.RS01)):
        if (runs.tool == tool).any():
            out.append((tool, int(runs[runs.tool == tool]["threads"].min()), ent))
    for i, t in enumerate(sorted(int(t) for t in runs[runs.tool == "rs02"]["threads"].unique())):
        label = f"psmc-rs 0.2, {t} thread{'s' if t > 1 else ''}"
        out.append(("rs02", t, style.Entity(label, style.BLUE, "-" if i == 0 else DASHED, "o", filled=i == 0)))
    return out


def _formatter(fn):
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fn(v) if v > 0 else "")


def fmt_seconds(v: float) -> str:
    for unit, size in (("h", 3600.0), ("min", 60.0)):
        if v >= size * 0.999:
            return f"{v / size:g} {unit}"
    return f"{v:g} s"


def fmt_mib(v: float) -> str:
    return f"{v / 1024:g} GB" if v >= 1024 * 0.999 else f"{v:g} MB"


def log_ticks(ax, lo: float, hi: float, ticks: List[float], fmt) -> None:
    """A log y axis from ``lo`` to ``hi`` labelled at the ``ticks`` inside it."""
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(_formatter(fmt))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


TIME_TICKS = [1, 10, 60, 600, 3600, 36000]
MEM_TICKS = [1, 10, 100, 1024, 10240, 102400]


def size_axis(ax, sizes) -> None:
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sorted(sizes)))
    ax.xaxis.set_major_formatter(_formatter(lambda v: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(sizes) / 1.35, max(sizes) * 1.35)
    ax.set_xlabel("Genome size (Mb)")


def decade_span(values, pad: float = 1.3) -> Tuple[float, float]:
    v = [float(x) for x in values if x == x and x > 0]
    return 10 ** math.floor(math.log10(min(v) / pad)), 10 ** math.ceil(math.log10(max(v) * pad))


def draw_metric(ax, runs, metric: str) -> None:
    """One metric against genome size: medians (lines, markers), repeats (dots), sizes not run named."""
    ok = runs[runs.status == "ok"]
    notes = []
    for tool, threads, ent in series(runs):
        sub = ok[(ok.tool == tool) & (ok.threads == threads)]
        if sub.empty:
            continue
        med = sub.groupby("genome_mb")[metric].median().sort_index()
        ax.plot(med.index, med.values, color=ent.color, linestyle=ent.linestyle, linewidth=style.LW, zorder=3)
        # Repeats under the markers: visible only where they stray from the median.
        ax.plot(sub["genome_mb"], sub[metric], linestyle="none", marker=".", markersize=1.5, color=ent.color,
                zorder=2)
        ax.plot(med.index, med.values, **ent.points(markersize=3.4, zorder=4))
        skipped = runs[(runs.tool == tool) & (runs.threads == threads) & (runs.status != "ok")]
        if not skipped.empty:
            sizes = ", ".join(f"{s:,}" for s in sorted(set(int(x) for x in skipped["genome_mb"])))
            why = skipped["note"].dropna()
            notes.append(f"{ent.label} not run at {sizes} Mb" + (f"\n({why.iloc[0]})" if len(why) else ""))
    if notes:
        style.note(ax, 0.02, 0.98, "\n".join(notes), ha="left", va="top")
    ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)


def draw(src: Source):
    runs = src.table("E3_runs")
    threads = src.table("E3_threads")
    fig = style.figure("double", 64)
    gs = style.grid(fig, 1, 3, left_mm=12.5, right_mm=5.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=15.0,
                    width_ratios=[1.25, 1.25, 0.85])
    ax_t, ax_m, ax_s = (fig.add_subplot(gs[0, i]) for i in range(3))
    sizes = sorted(int(s) for s in runs["genome_mb"].unique())
    ok = runs[runs.status == "ok"]

    draw_metric(ax_t, runs, "wall_seconds")
    size_axis(ax_t, sizes)
    log_ticks(ax_t, *decade_span(ok["wall_seconds"]), TIME_TICKS, fmt_seconds)
    ax_t.set_ylabel("Wall time (25 EM iterations)")

    draw_metric(ax_m, runs, "peak_rss_mb")
    size_axis(ax_m, sizes)
    log_ticks(ax_m, *decade_span(ok["peak_rss_mb"]), MEM_TICKS, fmt_mib)
    ax_m.set_ylabel("Peak memory (RSS)")

    speedup_panel(ax_s, threads)

    handles = [style.handle(ent, kind="both") for _, _, ent in series(runs)]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, [ax_t, ax_m, ax_s])
    return fig


def speedup_panel(ax, threads) -> None:
    """Speedup of psmc-rs 0.2 against threads, with the ideal line."""
    from matplotlib.ticker import FixedLocator, NullLocator

    med = threads.groupby("threads")["wall_seconds"].median().sort_index()
    t = med.index.to_numpy(dtype=float)
    base = float(med.iloc[0])  # the fewest threads (1 in the design)
    ax.plot(t, t / t[0], color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=1)
    ax.annotate("ideal", (t[-1], t[-1] / t[0]), xytext=(-3, 0), textcoords="offset points", ha="right",
                va="bottom", fontsize=style.TICK_PT, color=style.MUTED)
    speed = base / med.to_numpy()
    ax.plot(t, speed, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax.plot(threads["threads"], base / threads["wall_seconds"].to_numpy(), linestyle="none", marker=".",
            markersize=1.5, color=style.BLUE, zorder=2)
    ax.plot(t, speed, **style.RS02.points(markersize=3.4, zorder=4))
    ax.annotate(f"{speed[-1]:.1f}×", (t[-1], speed[-1]), xytext=(3, -3), textcoords="offset points",
                ha="left", va="top", fontsize=style.TICK_PT, color=style.INK2)
    for set_scale, axis in ((ax.set_xscale, ax.xaxis), (ax.set_yscale, ax.yaxis)):
        set_scale("log", base=2)
        axis.set_major_locator(FixedLocator(list(t)))
        axis.set_major_formatter(_formatter(lambda v: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
    ax.set_xlim(t[0] / 1.3, t[-1] * 1.3)
    ax.set_ylim(t[0] / 1.3, t[-1] / t[0] * 1.3)
    ax.set_xlabel("Threads")
    ax.set_ylabel(f"Speedup, {int(threads['genome_mb'].iloc[0]):,} Mb")
    ax.grid(True, which="major", color=style.GRID, linewidth=style.LW_GRID)
