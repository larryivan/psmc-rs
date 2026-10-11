"""The article's Fig 2: psmc-rs is fast and lean (E3).

Every run fits the recipe's EM iterations (25 in production) to a
constant-model genome made of 100 Mb chromosomes (1, 3, 10 and 30 of them at
100, 300, 1,000 and 3,000 Mb), alone on the machine. Points are the medians
of the repeats (five in production, within 6% of one another:
``wall_min_s`` and ``wall_max_s``), from ``E3_table2``; psmc-rs is 0.2
(``rs02``) with its default settings, C psmc is single-threaded. E3's runs
of psmc-rs 0.1 (``rs01``) are not drawn (``fig4_performance`` draws them).

* a: wall time against genome size: C psmc and psmc-rs on each of the
  thread counts E3 times every size at (``config.E3.rs02_threads``: 1, 8
  and 30), darker with more threads, each line's time on the largest genome
  at its end. psmc-rs runs the forward-backward pass of each sequence on one
  thread, in parallel across sequences, so threads beyond the number of
  chromosomes add nothing: on the 100 Mb genome, one chromosome, the
  psmc-rs points coincide.
* b: speed-up over C psmc on the thread-scaling genome (3,000 Mb, 30
  chromosomes) against threads (``config.E3.thread_scaling_threads``), the
  ratio of the medians; the dotted line is perfect scaling, the one-thread
  speed-up times the threads.
* c: peak memory (resident set size of the process tree) on the largest
  genome: C psmc, and psmc-rs on the threads of a.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from . import style
from .data import Source

H_MM = 62.0
TOP_MM, BOTTOM_MM = 7.0, 11.5
A_X_MM, A_W_MM = 13.0, 60.0  # a's axes; each line's time at its end, to the right
B_X_MM, B_W_MM = 101.0, 36.0
C_X_MM, C_W_MM = 151.0, 29.0
THREAD_RAMP = (style.BLUE, "#1c5cab", "#0d366b")  # steps 500, 700 and 900 of the blue ramp: more threads, darker
TIME_TICKS = (1.0, 10.0, 60.0, 600.0, 3600.0, 36000.0)


def thread_colors(threads: Sequence[int]) -> Dict[int, str]:
    """psmc-rs's colour at each thread count: its blue at one thread, darker with more (by log threads)."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    cmap = LinearSegmentedColormap.from_list("threads", THREAD_RAMP)
    hi = max(threads)
    return {t: to_hex(cmap(0.0 if hi <= 1 else math.log(t) / math.log(hi))) for t in threads}


def fmt_time(seconds: float) -> str:
    """A run time as people say it: 18 s, 8.7 min, 74 min, 2.5 h."""
    if seconds < 100:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 100:
        return f"{minutes:.1f} min" if minutes < 10 else f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


def fmt_tick(seconds: float) -> str:
    for unit, size in (("h", 3600.0), ("min", 60.0)):
        if seconds >= size * 0.999:
            return f"{seconds / size:g} {unit}"
    return f"{seconds:g} s"


def fmt_fold(x: float) -> str:
    return f"{x:.1f}×" if x < 10 else f"{x:,.0f}×"


def axes_mm(fig, x: float, w: float):
    fw, fh = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    return fig.add_axes((x / fw, BOTTOM_MM / fh, w / fw, (fh - TOP_MM - BOTTOM_MM) / fh))


def _plural(t: int) -> str:
    return f"{t} thread" + ("s" if t != 1 else "")


def draw(src: Source):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    cfg = src.profile.e3
    t2 = src.table("E3_table2")
    c = t2[(t2.tool == "c") & t2.wall_median_s.notna()].sort_values("size_mb")
    rs = t2[(t2.tool == "rs02") & t2.wall_median_s.notna()]
    threads_a = [t for t in cfg.rs02_threads if not rs[rs.threads == t].empty]
    scaling = rs[(rs.size_mb == cfg.thread_scaling_size_mb) & rs.speedup_vs_c.notna()].sort_values("threads")
    colors = thread_colors(sorted(set(threads_a) | {int(t) for t in scaling.threads}) or [1])
    sizes = sorted(int(s) for s in t2.size_mb.unique())
    big = max(sizes)

    fig = style.figure("double", H_MM)
    ax_a = axes_mm(fig, A_X_MM, A_W_MM)
    ax_b = axes_mm(fig, B_X_MM, B_W_MM)
    ax_c = axes_mm(fig, C_X_MM, C_W_MM)

    # a: wall time against genome size.
    lines = [("C psmc", c, style.ORANGE, "s")]
    lines += [(f"psmc-rs, {_plural(t)}", rs[rs.threads == t].sort_values("size_mb"), colors[t], "o")
              for t in threads_a]
    handles = []
    for name, d, color, marker in lines:
        if d.empty:
            continue
        ax_a.plot(d.size_mb, d.wall_median_s, color=color, linewidth=style.LW, marker=marker, markersize=3.6,
                  markeredgewidth=0, zorder=3)
        last = d.iloc[-1]
        ax_a.annotate(fmt_time(float(last.wall_median_s)), xy=(float(last.size_mb), float(last.wall_median_s)),
                      xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=style.TEXT_PT,
                      color=style.INK, annotation_clip=False)
        handles.append(Line2D([], [], color=color, linewidth=style.LW, marker=marker, markersize=3.6,
                              markeredgewidth=0, label=name))
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_locator(FixedLocator(sizes))
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.set_xlim(min(sizes) / 1.4, max(sizes) * 1.4)
    ax_a.set_xlabel(f"Genome size (Mb), in chromosomes of {cfg.chromosome_length / 1e6:g} Mb")
    walls = list(t2.loc[t2.tool.isin(["c", "rs02"]), "wall_median_s"].dropna())
    lo = 10 ** math.floor(math.log10(min(walls) / 1.3))
    hi = max(walls) * 3.0  # room for the legend above the lines
    ax_a.set_yscale("log")
    ax_a.yaxis.set_major_locator(FixedLocator([t for t in TIME_TICKS if lo <= t <= hi]))
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_tick(v)))
    ax_a.yaxis.set_minor_locator(NullLocator())
    ax_a.set_ylim(lo, hi)
    ax_a.set_ylabel(f"Wall time, {cfg.inference.n_iter} EM iterations")
    ax_a.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    style.legend(ax_a, handles, loc="upper left", bbox_to_anchor=(0.0, 1.02), handlelength=1.6,
                 labelspacing=0.35, borderaxespad=0.0)

    # b: speed-up over C psmc against threads, on the thread-scaling genome.
    if not scaling.empty:
        th = [int(t) for t in scaling.threads]
        sp = [float(v) for v in scaling.speedup_vs_c]
        x_lo, x_hi = 0.75, max(th) * 1.35
        s1 = sp[th.index(1)] if 1 in th else sp[0] / th[0]
        ax_b.plot([x_lo, x_hi], [s1 * x_lo, s1 * x_hi], color=style.MUTED, linewidth=style.LW_REF,
                  linestyle=style.DOT, zorder=1)
        ax_b.plot(th, sp, color=style.BLUE, linewidth=style.LW, zorder=2)
        ax_b.scatter(th, sp, s=16, c=[colors[t] for t in th], linewidths=0, zorder=3)
        for t, v in zip(th, sp):
            ax_b.annotate(fmt_fold(v), xy=(t, v), xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                          fontsize=style.TICK_PT, color=style.INK)
        ax_b.set_xscale("log")
        ax_b.xaxis.set_major_locator(FixedLocator(th))
        ax_b.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax_b.xaxis.set_minor_locator(NullLocator())
        ax_b.set_xlim(x_lo, x_hi)
        y_lo = min(sp) / 2.0
        y_hi = max(s1 * x_hi, max(sp)) * 1.4
        ax_b.set_yscale("log")
        ax_b.set_ylim(y_lo, y_hi)
        ax_b.yaxis.set_major_locator(FixedLocator([v for v in (1, 3, 10, 30, 100, 300, 1000, 3000)
                                                   if y_lo <= v <= y_hi]))
        ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}×"))
        ax_b.yaxis.set_minor_locator(NullLocator())
        ax_b.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
        # the perfect-scaling line's name, along it
        xa, xb = 1.6, 3.2
        p0, p1 = ax_b.transData.transform([(xa, s1 * xa), (xb, s1 * xb)])
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        xm = math.sqrt(xa * xb)
        ax_b.annotate("perfect scaling", xy=(xm, s1 * xm), xytext=(-2, 3), textcoords="offset points",
                      rotation=angle, rotation_mode="anchor", ha="center", va="bottom", fontsize=style.TICK_PT,
                      color=style.INK2)
        size = float(scaling.size_mb.iloc[0])
        n_chrom = int(scaling.chromosomes.iloc[0])
        style.note(ax_b, 0.03, 0.98, f"{size:,.0f} Mb, {n_chrom} chromosome{'s' if n_chrom != 1 else ''}",
                   size=style.TICK_PT)
    ax_b.set_xlabel("Threads")
    ax_b.set_ylabel("Speed-up over C psmc")

    # c: peak memory on the largest genome.
    bars: List = []
    c_big = c[c.size_mb == big]
    if not c_big.empty:
        bars.append(("C psmc", float(c_big.peak_rss_median_mb.iloc[0]), style.ORANGE))
    for t in threads_a:
        r = rs[(rs.threads == t) & (rs.size_mb == big)]
        if not r.empty:
            bars.append((f"{t}", float(r.peak_rss_median_mb.iloc[0]), colors[t]))
    xs = [0.0] + [1.25 + i for i in range(len(bars) - 1)]
    for x, (_, v, color) in zip(xs, bars):
        ax_c.bar(x, v, width=0.72, color=color, linewidth=0, zorder=3)
        ax_c.annotate(f"{v:,.0f}", xy=(x, v), xytext=(0, 2), textcoords="offset points", ha="center",
                      va="bottom", fontsize=style.TICK_PT, color=style.INK)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([b[0] for b in bars])
    ax_c.tick_params(axis="x", length=0)
    ax_c.set_xlim(-0.6, xs[-1] + 0.6 if xs else 1.0)
    top = max([b[1] for b in bars] or [1.0])
    ax_c.set_ylim(0, top * 1.15)
    step = 10 ** math.floor(math.log10(top))
    step *= 5 if top / step >= 5 else (0.5 if top / step < 2 else 1)  # two to four labelled ticks
    ax_c.yaxis.set_major_locator(FixedLocator([k * step for k in range(int(top * 1.15 // step) + 1)]))
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax_c.set_ylabel("Peak memory (MB)")
    style.note(ax_c, 0.97, 0.98, f"{big:,} Mb", ha="right", size=style.TICK_PT)
    ax_c.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)
    if len(xs) > 1:  # "psmc-rs, threads" under psmc-rs's bars only
        x_mid = (xs[1] + xs[-1] - ax_c.get_xlim()[0] * 2) / 2.0 / (ax_c.get_xlim()[1] - ax_c.get_xlim()[0])
        below_pt = style.TICK_PT + 2 * 1.8 + 1.0
        ax_c.set_xlabel("psmc-rs, threads")
        ax_c.xaxis.set_label_coords(x_mid, -below_pt / 72.0 / (ax_c.get_position().height * fig.get_figheight()))
    style.panel_letters(fig, [ax_a, ax_b, ax_c])
    return fig


__all__ = ["draw", "fmt_time", "thread_colors"]
