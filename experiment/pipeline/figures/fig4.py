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

# A tool's thread counts, fewest first: solid, dashed, dash-dotted, dotted (more counts reuse the last).
THREAD_LINESTYLES = ("-", (0, (3.0, 1.5)), (0, (3.0, 1.2, 1.0, 1.2)), (0, (1.0, 1.2)))
TOOL_ENTITY = {"c": style.C_PSMC, "rs01": style.RS01, "rs02": style.RS02,
               "rs02-default": style.Entity("psmc-rs 0.2, capped M-step", style.BLUE_650, "-", "^")}  # legend order
TIME_TICKS = [1, 10, 60, 600, 3600, 36000]
MEM_TICKS = [1, 10, 100, 1024, 10240, 102400]


def tool_threads(cfg) -> Dict[str, Tuple[int, ...]]:
    """The thread counts E3 times every genome size at, per tool (not the thread-scaling ones: Fig S4)."""
    return {"c": (1,), "rs01": tuple(cfg.rs01_threads), "rs02": tuple(cfg.rs02_threads),
            "rs02-default": tuple(cfg.rs02_default_threads)}


def series(table2, threads: Optional[Dict[str, Sequence[int]]] = None) -> List[Tuple[str, int, style.Entity]]:
    """``(tool, threads, entity)`` of every series of Table 2, in the legend's order.

    With ``threads`` (:func:`tool_threads`), only a tool's thread counts
    there. The fewest threads of a tool are drawn solid with filled markers,
    more threads with open markers and the next :data:`THREAD_LINESTYLES`.
    """
    out = []
    for tool, base in TOOL_ENTITY.items():
        sub = table2[table2.tool == tool]
        counts = sorted(int(t) for t in sub["threads"].unique() if threads is None or int(t) in threads.get(tool, ()))
        for i, t in enumerate(counts):
            label = base.label if tool == "c" else f"{base.label}, {t} thread{'s' if t > 1 else ''}"
            linestyle = THREAD_LINESTYLES[min(i, len(THREAD_LINESTYLES) - 1)]
            out.append((tool, t, style.Entity(label, base.color, linestyle, base.marker, filled=i == 0)))
    return out


def formatter(fn):
    """Tick labels ``fn(v)``, none at zero or below."""
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
    """A log y axis from ``lo`` to ``hi`` labelled ``fmt`` at the ``ticks`` inside it."""
    from matplotlib.ticker import FixedLocator

    style.log_axis(ax, "y", fmt="plain", lim=(lo, hi))
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(formatter(fmt))


def decade_span(values) -> Tuple[float, float]:
    """The powers of ten around the positive ``values`` padded by 1.3."""
    v = [float(x) for x in values if x == x and x > 0]
    return 10 ** math.floor(math.log10(min(v) / 1.3)), 10 ** math.ceil(math.log10(max(v) * 1.3))


def fixed_log_xaxis(ax, ticks: Sequence[float], fmt, pad: float, label: str, base: int = 10) -> None:
    """A log x axis labelled ``fmt`` at ``ticks`` alone (no minor ticks), ``pad`` times wider on each side."""
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.set_xscale("log", base=base)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(formatter(fmt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(ticks) / pad, max(ticks) * pad)
    ax.set_xlabel(label)


def size_axis(ax, sizes, chromosomes: Dict[int, int]) -> None:
    """The genome-size axis, each size's chromosome count on a second line."""
    fixed_log_xaxis(ax, sorted(sizes), lambda v: f"{v:,.0f}\n{chromosomes[int(round(v))]} chr.", 1.35,
                    "Genome size (Mb)")


def chromosome_counts(table2) -> Dict[int, int]:
    """Genome size (Mb) -> its number of chromosomes, from Table 2."""
    out = {}
    for size, n in table2.groupby("size_mb")["chromosomes"].unique().items():
        if len(n) != 1:
            raise ValueError(f"E3_table2 gives {size} Mb more than one chromosome count: {sorted(n)}")
        out[int(size)] = int(n[0])
    return out


def medians(ax, x, y, ent: style.Entity, markersize: float = 3.4, points_z: float = 4) -> None:
    """A series' medians: its line (zorder 3) and its markers."""
    ax.plot(x, y, color=ent.color, linestyle=ent.linestyle, linewidth=style.LW_DATA, zorder=3)
    ax.plot(x, y, **ent.points(markersize=markersize, zorder=points_z))


def draw_metric(ax, table2, runs, median_col: str, run_col: str, threads: Dict[str, Sequence[int]]) -> List[str]:
    """One metric against genome size: medians (lines, markers), clean repeats (dots); returns notes on setups not run."""
    notes = []
    for tool, t, ent in series(table2, threads):
        rows = table2[(table2.tool == tool) & (table2.threads == t)].sort_values("size_mb")
        ok = rows[rows[median_col].notna()]
        if not ok.empty:
            medians(ax, ok["size_mb"], ok[median_col], ent)
        reps = runs[(runs.tool == tool) & (runs.threads == t) & (runs["clean"].astype(str) == "True")]
        ax.plot(reps["size_mb"], reps[run_col], linestyle="none", marker=".", markersize=1.5, color=ent.color,
                zorder=2)
        skipped = rows[rows.status == "skipped"]
        if not skipped.empty:
            sizes = ", ".join(f"{int(s):,}" for s in skipped["size_mb"])
            notes.append(f"{ent.label}: not run at {sizes} Mb ({str(skipped['note'].iloc[0]).split(';')[0]})")
    style.value_grid(ax)
    return notes


def speedup_panel(ax, table2, sizes, threads: Dict[str, Sequence[int]], chromosomes: Dict[int, int]) -> None:
    """Median wall time of C psmc over each tool's, by genome size."""
    from matplotlib.ticker import FixedLocator, NullLocator

    hi, lo = 1.0, 1.0
    for tool, t, ent in series(table2, threads):
        if tool == "c":
            continue
        rows = table2[(table2.tool == tool) & (table2.threads == t)].sort_values("size_mb")
        rows = rows[rows["speedup_vs_c"].notna()]
        if rows.empty:
            continue
        medians(ax, rows["size_mb"], rows["speedup_vs_c"], ent)
        hi = max(hi, float(rows["speedup_vs_c"].max()))
        lo = min(lo, float(rows["speedup_vs_c"].min()))
    style.ref_line(ax, y=1.0)
    size_axis(ax, sizes, chromosomes)
    ax.set_yscale("log", base=2)
    top = 2 ** math.ceil(math.log2(hi * 1.2))
    bottom = min(0.5, 2 ** math.floor(math.log2(lo / 1.2)))
    ax.yaxis.set_major_locator(FixedLocator([2.0**k for k in range(int(math.log2(bottom)), int(math.log2(top)) + 1)]))
    ax.yaxis.set_major_formatter(formatter(lambda v: f"{v:g}×"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom, top)
    ax.set_ylabel("Speed-up over C psmc")
    style.value_grid(ax)


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
    for ax, median_col, run_col, ticks, fmt, label in (
            (ax_t, "wall_median_s", "wall_seconds", TIME_TICKS, fmt_seconds,
             f"Wall time, {src.profile.e3.inference.n_iter} EM iterations"),
            (ax_m, "peak_rss_median_mb", "peak_rss_mb", MEM_TICKS, fmt_mib, "Peak memory (RSS)")):
        notes = draw_metric(ax, table2, runs, median_col, run_col, threads)
        size_axis(ax, sizes, chromosomes)
        log_ticks(ax, *decade_span(list(table2[median_col].dropna()) + list(runs[run_col])), ticks, fmt)
        ax.set_ylabel(label)
        if notes and ax is ax_t:  # the same setups in both panels: named once
            style.note(ax, 0.02, 0.98, "\n".join(notes), ha="left", va="top")
    speedup_panel(ax_s, table2, sizes, threads, chromosomes)

    handles = [style.handle(ent, kind="both") for _, _, ent in series(table2, threads)]
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    style.panel_letters(fig, [ax_t, ax_m, ax_s])
    return fig


__all__ = ["chromosome_counts", "draw", "fixed_log_xaxis", "fmt_mib", "fmt_seconds", "formatter", "log_ticks",
           "medians", "series", "size_axis", "tool_threads"]
