"""Fig 4 (E3): run time and peak memory against genome size, and the speed-up over C psmc.

Every run fits 25 EM iterations of the recipe to a constant-model genome of
100 Mb chromosomes, alone on the machine (E3's clean runs only): 1, 3, 10
and 30 chromosomes at 100, 300, 1,000 and 3,000 Mb, the second line of each
size's tick label.

psmc-rs (0.1 and 0.2) runs the E-step of each sequence on one thread, in
parallel across sequences, so on t threads a genome of n equal chromosomes
takes about ceil(n / t) times one chromosome's time: 17.2 to 18.3 s for
every psmc-rs 0.2 setup in production (Supplementary Note 1, section 10).
Threads beyond the chromosome count add nothing: at 100 Mb (one
chromosome) the 1-, 8- and 30-thread points coincide, the 30-thread lines
are flat, and the 30-thread speed-up at 3,000 Mb (243x in production)
holds because its 30 equal chromosomes give each thread one. On a real
genome the longest sequence sets the floor (Fig S4).

* a: wall time against genome size: C psmc (one thread), psmc-rs 0.1 and
  psmc-rs 0.2 at each of the thread counts E3 times every size at (1, 8
  and 30 in production: solid, dashed and dash-dotted; the other counts of
  the thread scaling are Fig S4's), and, when E3 timed it
  (``E3.default_mstep_iters``, not by default), psmc-rs 0.2 with capped
  M-steps (dark blue). The y label gives the profile's EM iterations (25 in
  production). Lines and markers are the medians of the repeats, small dots
  each repeat.
* b: peak resident memory of the tool's process tree, same runs.
* c: speed-up over C psmc at each size, the ratio of the medians.

"psmc-rs 0.2" is the setting of every other experiment, ``--mstep-iters
1000`` (``config.Inference``), its default: its M-steps converge, as C
psmc's do, in a few ms per EM iteration. A setup that was not run (its
estimated memory above the limit) is named in panel a with the reason
Table 2 gives. Everything comes from ``E3_table2`` and ``E3_runs``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import style
from .data import Source

DASHED = (0, (3.0, 1.5))
DASH_DOT = (0, (3.0, 1.2, 1.0, 1.2))
DOTTED = (0, (1.0, 1.2))
# A tool's thread counts, fewest first: one line style each (more counts than
# styles reuse the last).
THREAD_LINESTYLES = ("-", DASHED, DASH_DOT, DOTTED)
TOOL_ENTITY = {"c": style.C_PSMC, "rs01": style.RS01, "rs02": style.RS02,
               "rs02-default": style.Entity("psmc-rs 0.2, capped M-step", style.BLUE_650, "-", "^")}
TOOL_ORDER = ("c", "rs01", "rs02", "rs02-default")


def tool_threads(cfg) -> Dict[str, Tuple[int, ...]]:
    """The thread counts E3 times every genome size at, per tool (not the thread-scaling ones: Fig S4)."""
    return {"c": (1,), "rs01": tuple(cfg.rs01_threads), "rs02": tuple(cfg.rs02_threads),
            "rs02-default": tuple(cfg.rs02_default_threads)}


def series(table2, threads: Optional[Dict[str, Sequence[int]]] = None) -> List[Tuple[str, int, style.Entity]]:
    """``(tool, threads, entity)`` of every series of Table 2, in the legend's order.

    With ``threads`` (:func:`tool_threads`), only a tool's thread counts
    there. The fewest threads of a tool are drawn solid with filled markers,
    more threads with open markers and the next styles of
    :data:`THREAD_LINESTYLES` (dashed, then dash-dotted, then dotted).
    """
    out = []
    for tool in TOOL_ORDER:
        sub = table2[table2.tool == tool]
        if sub.empty:
            continue
        base = TOOL_ENTITY[tool]
        counts = sorted(int(t) for t in sub["threads"].unique()
                        if threads is None or int(t) in threads.get(tool, ()))
        for i, t in enumerate(counts):
            label = base.label if tool == "c" else f"{base.label}, {t} thread{'s' if t > 1 else ''}"
            linestyle = THREAD_LINESTYLES[min(i, len(THREAD_LINESTYLES) - 1)]
            out.append((tool, t, style.Entity(label, base.color, linestyle, base.marker, filled=i == 0)))
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


TIME_TICKS = [1, 10, 60, 600, 3600, 36000]
MEM_TICKS = [1, 10, 100, 1024, 10240, 102400]


def log_ticks(ax, lo: float, hi: float, ticks: List[float], fmt) -> None:
    """A log y axis from ``lo`` to ``hi`` labelled at the ``ticks`` inside it."""
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(_formatter(fmt))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)


def size_axis(ax, sizes, chromosomes: Optional[Dict[int, int]] = None) -> None:
    """The genome-size axis; with ``chromosomes`` (size -> count), each size's count on a second line."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sorted(sizes)))
    if chromosomes:
        ax.xaxis.set_major_formatter(_formatter(lambda v: f"{v:,.0f}\n{chromosomes[int(round(v))]} chr."))
    else:
        ax.xaxis.set_major_formatter(_formatter(lambda v: f"{v:,.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(sizes) / 1.35, max(sizes) * 1.35)
    ax.set_xlabel("Genome size (Mb)")


def decade_span(values, pad: float = 1.3) -> Tuple[float, float]:
    v = [float(x) for x in values if x == x and x > 0]
    return 10 ** math.floor(math.log10(min(v) / pad)), 10 ** math.ceil(math.log10(max(v) * pad))


def draw_metric(ax, table2, runs, median_col: str, run_col: str,
                threads: Optional[Dict[str, Sequence[int]]] = None) -> List[str]:
    """One metric against genome size: medians (lines, markers), clean repeats (dots); returns notes on setups not run."""
    notes = []
    for tool, t, ent in series(table2, threads):
        rows = table2[(table2.tool == tool) & (table2.threads == t)].sort_values("size_mb")
        ok = rows[rows[median_col].notna()]
        if not ok.empty:
            ax.plot(ok["size_mb"], ok[median_col], color=ent.color, linestyle=ent.linestyle, linewidth=style.LW_DATA,
                    zorder=3)
            ax.plot(ok["size_mb"], ok[median_col], **ent.points(markersize=3.4, zorder=4))
        reps = runs[(runs.tool == tool) & (runs.threads == t) & (runs["clean"].astype(str) == "True")]
        ax.plot(reps["size_mb"], reps[run_col], linestyle="none", marker=".", markersize=1.5, color=ent.color,
                zorder=2)
        skipped = rows[rows.status == "skipped"]
        if not skipped.empty:
            sizes = ", ".join(f"{int(s):,}" for s in skipped["size_mb"])
            notes.append(f"{ent.label}: not run at {sizes} Mb ({str(skipped['note'].iloc[0]).split(';')[0]})")
    ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_HAIR)
    return notes


def draw(src: Source):
    table2 = src.table("E3_table2")
    runs = src.table("E3_runs")
    fig = style.figure("double", 70)
    # bottom_mm: a second tick line under each size, its chromosome count.
    gs = style.grid(fig, 1, 3, left_mm=12.5, right_mm=3.0, top_mm=12.0, bottom_mm=11.5, wspace_mm=14.0)
    ax_t, ax_m, ax_s = (fig.add_subplot(gs[0, i]) for i in range(3))
    sizes = sorted(int(s) for s in table2["size_mb"].unique())
    chromosomes = chromosome_counts(table2)
    threads = tool_threads(src.profile.e3)

    notes = draw_metric(ax_t, table2, runs, "wall_median_s", "wall_seconds", threads)
    size_axis(ax_t, sizes, chromosomes)
    log_ticks(ax_t, *decade_span(list(table2["wall_median_s"].dropna()) + list(runs["wall_seconds"])), TIME_TICKS,
              fmt_seconds)
    ax_t.set_ylabel(f"Wall time, {src.profile.e3.inference.n_iter} EM iterations")
    if notes:
        style.note(ax_t, 0.02, 0.98, "\n".join(notes), ha="left", va="top")

    draw_metric(ax_m, table2, runs, "peak_rss_median_mb", "peak_rss_mb", threads)
    size_axis(ax_m, sizes, chromosomes)
    log_ticks(ax_m, *decade_span(list(table2["peak_rss_median_mb"].dropna()) + list(runs["peak_rss_mb"])), MEM_TICKS,
              fmt_mib)
    ax_m.set_ylabel("Peak memory (RSS)")

    speedup_panel(ax_s, table2, sizes, threads, chromosomes)

    handles = [style.handle(ent, kind="both") for _, _, ent in series(table2, threads)]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    style.panel_letters(fig, [ax_t, ax_m, ax_s])
    return fig


def chromosome_counts(table2) -> Dict[int, int]:
    """Genome size (Mb) -> its number of chromosomes, from Table 2."""
    counts = table2.groupby("size_mb")["chromosomes"].unique()
    out = {}
    for size, n in counts.items():
        if len(n) != 1:
            raise ValueError(f"E3_table2 gives {size} Mb more than one chromosome count: {sorted(n)}")
        out[int(size)] = int(n[0])
    return out


def speedup_panel(ax, table2, sizes, threads: Optional[Dict[str, Sequence[int]]] = None,
                  chromosomes: Optional[Dict[int, int]] = None) -> None:
    """Median wall time of C psmc over each tool's, by genome size."""
    import numpy as np

    hi, lo = 1.0, 1.0
    for tool, t, ent in series(table2, threads):
        if tool == "c":
            continue
        rows = table2[(table2.tool == tool) & (table2.threads == t)].sort_values("size_mb")
        rows = rows[rows["speedup_vs_c"].notna()]
        if rows.empty:
            continue
        ax.plot(rows["size_mb"], rows["speedup_vs_c"], color=ent.color, linestyle=ent.linestyle,
                linewidth=style.LW_DATA, zorder=3)
        ax.plot(rows["size_mb"], rows["speedup_vs_c"], **ent.points(markersize=3.4, zorder=4))
        hi = max(hi, float(np.nanmax(rows["speedup_vs_c"])))
        lo = min(lo, float(np.nanmin(rows["speedup_vs_c"])))
    style.ref_line(ax, y=1.0)
    size_axis(ax, sizes, chromosomes)
    ax.set_yscale("log", base=2)
    from matplotlib.ticker import FixedLocator, NullLocator

    top = 2 ** math.ceil(math.log2(hi * 1.2))
    bottom = min(0.5, 2 ** math.floor(math.log2(lo / 1.2)))
    ax.yaxis.set_major_locator(FixedLocator([2.0**k for k in range(int(math.log2(bottom)), int(math.log2(top)) + 1)]))
    ax.yaxis.set_major_formatter(_formatter(lambda v: f"{v:g}×"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Speed-up over C psmc")
    ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_HAIR)


__all__ = ["chromosome_counts", "draw", "fmt_mib", "fmt_seconds", "log_ticks", "series", "size_axis",
           "tool_threads"]
