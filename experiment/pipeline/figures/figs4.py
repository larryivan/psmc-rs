"""Fig S4 (E3): thread scaling and determinism of psmc-rs 0.2.

On the thread-scaling genome (3,000 Mb as 30 chromosomes of 100 Mb, 25 EM
iterations, E3's clean runs), on 1 to 30 threads:

* a: wall time against threads: every repeat (dots) and the median (line),
  with ideal scaling from the fewest threads (dotted). The M-step runs on
  one thread but converges in a few ms per EM iteration, so the E-step's
  sequences set the speed-up. psmc-rs works on
  one sequence per thread, so the ideal is that of whole sequences per
  thread (:func:`ceiling_speedup`: 30 sequences on 8 threads take 4 rounds,
  a speed-up of 7.5), and its floor, marked, is the time of one chromosome:
  threads beyond the 30 sequences add nothing.
* b: parallel efficiency, speed-up / threads, of the medians, and the
  ideal's (dotted): 1 where the threads divide the sequences, below where
  they do not (94% on 4, 8 and 16 threads). Every point is labelled but
  the fewest threads', 100% by construction.
* c: peak memory of the process tree against threads.
* d: determinism, per tool, with its thread counts in brackets: Runs, the
  runs summed over E3's four genome sizes (100 to 3,000 Mb; thread counts
  x repeats x sizes), and Outputs, how many distinct output files the runs
  at one genome size wrote (a range where the sizes differ). A footnote
  names the runs behind any count above 1: in production, psmc-rs 0.1 on
  one thread wrote different bytes than on 8 and 30 threads at 300 Mb and
  above (parameters within 2.5 x 10^-5 relative), while its repeats at a
  thread count were identical.

Only when E3 times psmc-rs 0.2 with capped M-steps
(``E3.default_mstep_iters``, which production does not set) do a and b
draw it too (dark blue) and the legend name it. Everything comes from
``E3_thread_scaling``, ``E3_table2``, ``E3_runs`` and ``E3_determinism``.
"""

from __future__ import annotations

from typing import List, Tuple

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


def ceiling_speedup(threads, sequences: int):
    """The most speed-up over one thread that ``threads`` give on ``sequences`` equal sequences, one per thread.

    Each thread works on whole sequences, so the run takes
    ceil(sequences / threads) rounds: sequences / ceil(sequences / threads).
    It stops growing at ``sequences`` threads.
    """
    import numpy as np

    return sequences / np.ceil(sequences / np.asarray(threads, dtype=float))


def draw(src: Source):
    scaling = src.table("E3_thread_scaling").sort_values("threads")
    table2 = src.table("E3_table2")
    runs = src.table("E3_runs")
    det = src.table("E3_determinism")
    size = int(scaling["size_mb"].iloc[0])
    chrom_mb = src.profile.e3.chromosome_length / 1e6
    n_seq = max(1, round(size / chrom_mb))
    t = scaling["threads"].to_numpy(dtype=float)
    wall = scaling["wall_median_s"].to_numpy(dtype=float)
    # Ideal times and efficiencies relative to the fewest threads, and the floor: one chromosome.
    ceiling = ceiling_speedup(t, n_seq) / ceiling_speedup(t[0], n_seq)
    ideal = wall[0] / ceiling
    floor = wall[0] * ceiling_speedup(t[0], n_seq) / n_seq
    reps = runs[(runs.tool == "rs02") & (runs.size_mb == size) & (runs["clean"].astype(str) == "True")]
    default = table2[(table2.tool == "rs02-default") & (table2.size_mb == size)].sort_values("threads")
    default = default[default["wall_median_s"].notna()]
    dent = TOOL_ENTITY["rs02-default"]

    fig = style.figure("double", 62)
    gs = style.grid(fig, 1, 4, left_mm=12.0, right_mm=2.0, top_mm=9.0, bottom_mm=9.0, wspace_mm=13.0,
                    width_ratios=[1.0, 1.0, 1.0, 1.3])
    ax_t, ax_e, ax_m, ax_d = (fig.add_subplot(gs[0, i]) for i in range(4))

    ax_t.plot(t, ideal, color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=1)
    at = len(t) - 2 if len(t) > 2 else len(t) - 1  # clear of the floor's label
    ax_t.annotate("ideal", (t[at], ideal[at]), xytext=(0, -4), textcoords="offset points",
                  ha="center", va="top", fontsize=style.TICK_PT, color=style.MUTED)
    style.ref_line(ax_t, y=floor, linestyle=style.DASH)
    ax_t.annotate(f"one {chrom_mb:g} Mb chromosome:\nno gain beyond {n_seq} threads", (t[0], floor), xytext=(0, -3),
                  textcoords="offset points", ha="left", va="top", fontsize=style.TICK_PT, color=style.MUTED,
                  linespacing=1.1)
    ax_t.plot(t, wall, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_t.plot(reps["threads"], reps["wall_seconds"], linestyle="none", marker=".", markersize=2.0, color=style.BLUE,
              zorder=4)
    ax_t.plot(t, wall, **style.RS02.points(markersize=3.2, zorder=3))
    walls = list(reps["wall_seconds"]) + list(wall) + list(ideal) + [floor / 2.5]  # room for the floor's label
    if not default.empty:
        ax_t.plot(default["threads"], default["wall_median_s"], color=dent.color, linewidth=style.LW, zorder=3)
        ax_t.plot(default["threads"], default["wall_median_s"], **dent.points(markersize=3.2, zorder=4))
        walls += list(default["wall_median_s"])
    thread_axis(ax_t, t)
    log_ticks(ax_t, *decade_span(walls), TIME_TICKS, fmt_seconds)
    ax_t.set_ylabel(f"Wall time, {size:,} Mb")
    ax_t.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)

    eff = scaling["efficiency"].to_numpy(dtype=float)
    ax_e.plot(t, ceiling * t[0] / t, color=style.MUTED, linewidth=style.LW_REF, linestyle=style.DOT, zorder=1)
    ax_e.annotate("ideal", (t[-1], ceiling[-1] * t[0] / t[-1]), xytext=(0, 1), textcoords="offset points",
                  ha="center", va="bottom", fontsize=style.TICK_PT, color=style.MUTED)
    ax_e.plot(t, eff, color=style.BLUE, linewidth=style.LW, zorder=3)
    ax_e.plot(t, eff, **style.RS02.points(markersize=3.2, zorder=4))
    # The fewest threads are 100% by construction (the gridline and tick say so); a
    # label there would sit on the y axis.
    for x, e in zip(t[1:], eff[1:]):
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

    # Panel d has no y axis: it takes most of the gap before it, which a-c keep for their y labels.
    pos = ax_d.get_position()
    shift = TABLE_SHIFT_MM / (fig.get_figwidth() / style.MM)
    ax_d.set_position((pos.x0 - shift, pos.y0, pos.width + shift, pos.height))
    determinism_panel(ax_d, det)
    handles = [style.handle(style.RS02, kind="both", linewidth=style.LW, label="psmc-rs 0.2")]
    if not default.empty:  # timed only when E3.default_mstep_iters sets a cap
        handles.append(style.handle(dent, kind="both", linewidth=style.LW))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, [ax_t, ax_e, ax_m, ax_d])
    return fig


TABLE_SHIFT_MM = 9.5  # panel d starts this much left of its grid cell
TOOL_LABELS = {"c": "C psmc", "rs01": "psmc-rs 0.1", "rs02": "psmc-rs 0.2", "rs02-default": "0.2, capped M-step"}


NBSP = "\u00a0"  # keeps a number on the line of its unit when a footnote wraps


def _threads_text(threads) -> str:
    threads = sorted(int(t) for t in threads)
    if len(threads) == 1:
        return f"{threads[0]}{NBSP}thread" + ("s" if threads[0] > 1 else "")
    return ", ".join(str(t) for t in threads[:-1]) + f" and {threads[-1]}{NBSP}threads"


def _sizes_text(sizes, all_sizes) -> str:
    sizes, all_sizes = sorted(int(x) for x in sizes), sorted(int(x) for x in all_sizes)
    if 1 < len(sizes) < len(all_sizes) and sizes == [x for x in all_sizes if x >= sizes[0]]:
        return f"{sizes[0]:,}{NBSP}Mb and above"
    names = [f"{x:,}" for x in sizes]
    return (", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]) + f"{NBSP}Mb"


def output_differences(det) -> List[Tuple[int, str]]:
    """``(outputs, sentence)`` for each tool whose runs at a genome size wrote more than one output.

    Within a size, runs are grouped by the bytes they wrote. When each
    thread count's repeats agree and the thread counts split the same way
    at every such size, the sentence names the thread counts on each side;
    ``outputs`` is the tool's most distinct outputs at one size.
    """
    from .data import fmt_sig

    out = []
    all_sizes = det["size_mb"].unique()
    for tool in [k for k in TOOL_LABELS if k in set(det["tool"])]:
        sub = det[det.tool == tool]
        per_size = sub.groupby("size_mb")["sha256"].nunique()
        diff_sizes = sorted(int(x) for x in per_size[per_size > 1].index)
        if not diff_sizes:
            continue
        rows = sub[sub.size_mb.isin(diff_sizes)]
        repeats_agree = bool((rows.groupby(["size_mb", "threads"])["sha256"].nunique() == 1).all())
        splits = {tuple(sorted(tuple(sorted(int(t) for t in g["threads"].unique()))
                               for _, g in rows[rows.size_mb == size].groupby("sha256")))
                  for size in diff_sizes}
        rel = fmt_sig(float(rows["max_rel_diff_params"].max()))
        where = f"{TOOL_LABELS[tool]} at {_sizes_text(diff_sizes, all_sizes)}"
        if repeats_agree and len(splits) == 1 and len(next(iter(splits))) == 2:
            fewer, more = sorted(next(iter(splits)), key=lambda g: (len(g), g))
            text = (f"{where}: {_threads_text(fewer)} wrote different bytes from {_threads_text(more)} "
                    f"(parameters within {rel} relative); repeats at a thread count are identical")
        else:
            text = f"{where}: runs differ (parameters within {rel} relative)"
        out.append((int(per_size.max()), text))
    return out


def _wrap(ax, text: str, width: float = 1.0, **kw) -> str:
    """``text`` broken at spaces into lines no wider than ``width`` of ``ax`` (measured as drawn)."""
    renderer = ax.figure.canvas.get_renderer()
    limit = ax.get_window_extent(renderer).width * width
    probe = ax.text(0.0, 0.0, "", transform=ax.transAxes, **kw)
    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        probe.set_text(trial)
        if line and probe.get_window_extent(renderer).width > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    probe.remove()
    return "\n".join(lines + [line])


def determinism_panel(ax, det) -> None:
    """Per tool: runs (thread counts and repeats, summed over the sizes) and the distinct outputs per size."""
    ax.set_axis_off()
    style.note(ax, 0.0, 0.99, "Outputs of repeated runs", color=style.INK, size=style.TEXT_PT)
    # The tool left-aligned, the two counts right-aligned under one-line headers, all on baselines.
    cols = ((0.0, "left"), (0.68, "right"), (1.0, "right"))
    y = 0.80
    for (x, ha), head in zip(cols, ("Tool (threads)", "Runs", "Outputs")):
        style.note(ax, x, y, head, ha=ha, va="baseline", color=style.INK2)
    ax.plot([0.0, cols[-1][0]], [y - 0.045, y - 0.045], color=style.GRID, linewidth=style.LW_GRID,
            transform=ax.transAxes, clip_on=False)
    for tool in [k for k in TOOL_LABELS if k in set(det["tool"])]:
        sub = det[det.tool == tool]
        threads = sorted(int(x) for x in sub["threads"].unique())
        span = f"{threads[0]}-{threads[-1]}" if len(threads) > 1 else str(threads[0])
        per_size = sub.groupby("size_mb")["sha256"].nunique()
        lo, hi = int(per_size.min()), int(per_size.max())
        outputs = str(hi) if lo == hi else f"{lo}-{hi}"
        y -= 0.1
        for (x, ha), text in zip(cols, (f"{TOOL_LABELS[tool]} ({span})", str(len(sub)), outputs)):
            style.note(ax, x, y, text, ha=ha, va="baseline", color=style.INK)
    notes = ["1: all runs at a size wrote the same bytes"]
    notes += [f"{n}: {text}" for n, text in output_differences(det)]
    text = "\n".join(_wrap(ax, n, cols[-1][0], fontsize=style.TEXT_PT) for n in notes)
    style.note(ax, 0.0, y - 0.09, text, color=style.INK2, linespacing=1.15)


__all__ = ["ceiling_speedup", "determinism_panel", "draw", "output_differences"]
