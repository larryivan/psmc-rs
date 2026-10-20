"""The article's Fig 2: psmc-rs is fast and lean (E3, E7).

One row of three panels on one rule: cost on the y axis, C psmc in orange
high in the panel, psmc-rs in blue low. a and c show the advantage as a
double arrow with a bold fold label ("8.6× faster"; 7 pt, 3 pt from its
arrow, on two lines in c's narrow gap between the columns); b shows it as
the empty height on a linear axis from zero. No legend and no grid: each
program is named once per panel beside its data, in a text colour darker
than its mark (:data:`.style.ORANGE_TEXT`, :data:`.style.BLUE_TEXT`), as
one line of name and value ("C psmc  74 min"), and every number the text
cites is printed on the data, to the precision that reproduces the printed
folds (74 min over 8.65 min and over 18.3 s give 8.6× and 243×; 104.3 min
over 12.1 min, the text's medians, give 8.6×).

* a, b (E3, ``E3_table2``): the thread-scaling genome
  (``config.E3.thread_scaling_size_mb``: 3,000 Mb, 30 chromosomes of 100
  Mb in production), fitted for the recipe's EM iterations, each run alone
  on its machine; points are the medians of the repeats. The x axis is
  psmc-rs's threads (``config.E3.thread_scaling_threads``: 1, 2, 4, 8, 16
  and 30); a and b have the same width, so that their thread ticks line
  up. psmc-rs 0.2 (``rs02``, default settings) is circles on a 1 pt line;
  C psmc is single-threaded, so its one measurement (a square at one
  thread) is drawn flat across the thread counts.

  * a: wall time (log scale). Two arrows give the speed-up over C psmc
    (``speedup_vs_c``) on one thread (per-core efficiency, 8.6×) and on
    the most threads (243× on 30; drawn when they are at least four times
    the fewest, so that the two labels have room), with C psmc's time and
    psmc-rs's on one thread and on the most threads printed beside them.
    The upper limit puts C psmc's line at the same height of the axes as in
    b, so that the two orange lines of these small multiples align.
    psmc-rs fits each sequence on one thread, in parallel across
    sequences, so the speed-up on many threads needs as many sequences of
    similar length (for the caption to say).
  * b: peak memory, the resident set size of the process tree (linear
    scale from zero): C psmc's, and psmc-rs's at one thread and at the most
    threads printed above their points; psmc-rs's rises with threads
    because each thread holds its own sequence. No arrow.

* c (E7, ``E7_fits``): the 1000 Genomes cohort, each genome fitted on one
  core of a node shared with other fits: wall time per genome of C psmc
  (route ``c``, the default genomes; squares) and of psmc-rs (route
  ``rs-psmcfa``, every genome; circles), jittered within their column in a
  fixed order (sorted by sample, a seeded generator), with ink bars at the
  medians (as wide as the jitter, so that the narrow psmc-rs band shows
  around its bar), n under each name (on the baseline of a's and b's x
  labels), each median printed by its cloud and an arrow between the
  medians giving their ratio. The absolute times are not comparable with
  a's (a shared node), the ratio is. Without ``E7_fits`` a and b take the
  row alone; the figure's spec must list ``E7_fits`` for the pipeline to
  redraw it when the cohort changes.

C psmc's or psmc-rs's marks, labels and arrows are left out where their
rows are missing (the small profiles time fewer threads and genomes).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from . import style
from .data import Source

H_MM = 55.0
C_W_MM = 32.0  # panel c, two categories; a and b share the rest of the row equally
TIME_TICKS = (0.01, 0.1, 1.0, 10.0, 60.0, 600.0, 3600.0, 36000.0)  # 10 s, 1 min, 10 min, 1 h
ROUTE_C, ROUTE_RS = "c", "rs-psmcfa"  # E7_fits routes: C psmc, and psmc-rs on the psmcfa (every genome)
COHORT_MS = 1.7  # psmc-rs's 2,504 cohort dots: a band about 1 mm tall, which shows around its median bar
COHORT_EDGE_LW = 0.25  # white rim between c's overlapping squares, as in Fig 1 e (EDGE_LW)
COHORT_BAR_HALF = style.STRIP_JITTER  # c's median bars span the clouds' centres, not past them (house: STRIP_BAR_HALF)
COHORT_HEAD = 1.1  # c's y axis runs this far past its top tick, clear of the title row
A_HI_FALLBACK = 1.62  # a's upper limit over C psmc's time when b has no C psmc line to match
A_ROOM = 1.3  # ...and at least this much, room above C psmc's line for its name
SHRINK_PT = 2.6  # a's arrow heads stop this far from the line or the marker they point at
FOLD_DX_PT = 3.0  # a fold label's edge from its arrow, in a and c
THREAD_ROOM = 1.6  # the thread axis ends this far past the most threads: room for the last value in a
SEED = 7  # the jitter of c's points


def fmt_time(seconds: float) -> str:
    """A run time as people say it: 0.35 s, 18.3 s, 8.65 min, 74 min, 2.5 h.

    Seconds keep one decimal (as the text's macros do) and minutes below ten keep three figures, so
    that C psmc's time over psmc-rs's printed beside it gives the printed fold (74 min / 18.3 s = 243×,
    where 18 s would give 247×; 74 min / 8.65 min = 8.6×, where 8.7 min would give 8.5×).
    """
    if seconds < 1:
        return f"{seconds:.2g} s"
    if seconds < 100:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 100:
        return f"{minutes:.3g} min" if minutes < 10 else f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


def fmt_tick(seconds: float) -> str:
    """A time tick: 10 s, 1 min, 10 min, 1 h."""
    for unit, size in (("h", 3600.0), ("min", 60.0)):
        if seconds >= size * 0.999:
            return f"{seconds / size:g} {unit}"
    return f"{seconds:g} s"


def fmt_fold(x: float) -> str:
    """A fold change: 8.6×, 17×, 243×, 7,000×."""
    return f"{x:.1f}{style.TIMES}" if x < 10 else f"{x:,.0f}{style.TIMES}"


def _fmt_value(v: float) -> str:
    """A value printed on the data: 1,062, 36, 0.29."""
    return f"{v:,.0f}" if v >= 10 else f"{v:.2g}"


def _fmt_median(v: float) -> str:
    """A median of c, with one decimal as the text's macros print them (104.3, 12.1); three figures below 10."""
    return f"{v:,.1f}" if v >= 10 else f"{v:.3g}"


def _lab(ax, s: str, xy, dx: float, dy: float, *, ha: str = "left", va: str = "bottom", color: str = style.INK,
         size: float = style.LABEL_PT, **kw):
    """Text ``dx``, ``dy`` points from the data point ``xy``."""
    return ax.annotate(s, xy=xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                       color=color, annotation_clip=False, zorder=style.Z_TEXT, **kw)


def _fold_label(ax, s: str, xy, side: int, **kw):
    """A fold change, the figure's message: 7 pt bold ink FOLD_DX_PT right (``side`` +1) or left (-1) of its
    arrow."""
    return _lab(ax, s, xy, FOLD_DX_PT * side, 0, ha="left" if side > 0 else "right", va="center",
                size=style.TEXT_PT, fontweight="bold", **kw)


def _gap_arrow(ax, p_from, p_to, shrink: float = 0.0) -> None:
    """A double arrow between two levels, its heads ``shrink`` points short of them."""
    props = dict(style.ARROW_PROPS, shrinkA=shrink, shrinkB=shrink)
    ax.annotate("", xy=p_to, xytext=p_from, arrowprops=props, annotation_clip=False, zorder=style.Z_TEXT)


def _programs(ax, th: Sequence[float], ys: Sequence[float], c_y: Optional[float]) -> None:
    """C psmc flat at ``c_y`` with its one measurement (a square at the fewest threads); psmc-rs's line of circles."""
    if c_y is not None:
        x0, x1 = (th[0], th[-1]) if len(th) else (1.0, 1.0)
        ax.plot([x0, x1], [c_y, c_y], color=style.ORANGE, linewidth=style.LW_DATA, zorder=style.Z_UNDERLAY)
        ax.plot([x0], [c_y], **style.C_PSMC.points(label=None, zorder=style.Z_TRUTH, clip_on=False))
    if len(th):
        ax.plot(th, ys, color=style.BLUE, linewidth=style.LW_EMPH, zorder=style.Z_DATA)
        ax.plot(th, ys, **style.RS02.points(label=None, zorder=style.Z_TRUTH, clip_on=False))


def _thread_axis(ax, th: Sequence[float]) -> None:
    """psmc-rs's threads on a log axis, ticked at the measured counts only."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(th)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    lo, hi = (min(th), max(th)) if len(th) else (1.0, 1.0)
    ax.set_xlim(lo / 1.19, hi * THREAD_ROOM)  # room right of the last point for a's value
    ax.set_xlabel("psmc-rs threads")


def _linear_ticks(top: float, most: int = 3) -> Tuple[float, List[float]]:
    """The fewest-digit 1-2-5 step giving at most ``most`` ticks from zero to ``top``, and those ticks."""
    k = math.floor(math.log10(top)) - 1 if top > 0 else 0
    while True:
        for m in (1.0, 2.0, 5.0):
            step = m * 10.0 ** k
            n = int(math.floor(top / step + 1e-9)) + 1
            if n <= most:
                return step, [i * step for i in range(n)]
        k += 1


def _panel_time(ax, th, wall, sp, c_wall: Optional[float], c_frac: Optional[float] = None) -> None:
    """a: wall time against threads, with the speed-up over C psmc on the fewest and the most threads.

    ``c_frac`` is the height of b's C psmc line as a fraction of b's axes: a's upper limit puts C psmc's
    time at the same height, so that the two orange lines align across the row.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    _thread_axis(ax, th)
    ax.set_yscale("log")
    values = [v for v in list(wall) + ([c_wall] if c_wall is not None else []) if math.isfinite(v) and v > 0]
    if values:
        floor = min(values) / 1.6
        lows = [t for t in TIME_TICKS if t <= floor]
        lo = lows[-1] if lows else 10.0 ** math.floor(math.log10(floor))
        top = max(values)
        hi = top * A_HI_FALLBACK  # room above C psmc's line for its name
        if c_wall is not None and c_wall >= top and c_frac is not None and 0.5 < c_frac < 0.97:
            hi = max(lo * (c_wall / lo) ** (1.0 / c_frac), c_wall * A_ROOM)
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator([t for t in TIME_TICKS if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: fmt_tick(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("Wall time")
    _programs(ax, th, wall, c_wall)
    if c_wall is not None and len(th):
        # The speed-up on one thread and on the most; the second only where the two labels have room.
        ends = [(0, 1)] + ([(len(th) - 1, -1)] if len(th) > 1 and th[-1] / th[0] >= 4 else [])
        for i, side in ends:
            _gap_arrow(ax, (th[i], c_wall), (th[i], wall[i]), SHRINK_PT)
            _fold_label(ax, f"{fmt_fold(sp[i])} faster", (th[i], math.sqrt(c_wall * wall[i])), side)
    if c_wall is not None:
        _lab(ax, f"C psmc  {fmt_time(c_wall)}", (th[0] if len(th) else 1.0, c_wall), -1.5, 2.5,
             color=style.ORANGE_TEXT)
    if len(th):
        _lab(ax, f"psmc-rs  {fmt_time(wall[0])}", (th[0], wall[0]), 3.5, 1.2, color=style.BLUE_TEXT)
    if len(th) > 1:
        _lab(ax, fmt_time(wall[-1]), (th[-1], wall[-1]), 3.5, 0, va="center", color=style.BLUE_TEXT)


def _memory_axis(mem, c_mem: Optional[float]) -> Tuple[float, List[float], float]:
    """b's y axis: its tick step, its ticks and its upper limit (the lower is zero)."""
    values = [v for v in list(mem) + ([c_mem] if c_mem is not None else []) if math.isfinite(v) and v > 0]
    top = max(values) * 1.13 if values else 1.0  # room above the highest line for its name
    step, ticks = _linear_ticks(top)
    q = step / 5.0
    return step, ticks, max(q * round(top / q), max(values or [0]) * 1.08)


def _panel_memory(ax, th, mem, c_mem: Optional[float], step: float, ticks: List[float], top: float) -> None:
    """b: peak memory against threads, on a linear axis from zero to ``top`` (:func:`_memory_axis`)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    _thread_axis(ax, th)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}" if step >= 1 else f"{v:g}"))
    ax.set_ylabel("Peak memory (MB)")
    _programs(ax, th, mem, c_mem)
    if c_mem is not None:
        _lab(ax, f"C psmc  {_fmt_value(c_mem)} MB", (th[0] if len(th) else 1.0, c_mem), -1.5, 2.5,
             color=style.ORANGE_TEXT)
    if len(th):  # left-aligned under C psmc's name, as in a
        _lab(ax, f"psmc-rs  {_fmt_value(mem[0])} MB", (th[0], mem[0]), -1.5, 3.5, color=style.BLUE_TEXT)
    if len(th) > 1:
        _lab(ax, f"{_fmt_value(mem[-1])} MB", (th[-1], mem[-1]), 0, 3.5, ha="center", color=style.BLUE_TEXT)


def _cohort(src: Source):
    """Panel c's wall times in seconds, sorted by sample: (C psmc's, psmc-rs's), or None without them.

    Only one-thread fits count (all of them in E7): the panel compares the programs core for core.
    """
    if not src.has("E7_fits"):
        return None
    f = src.table("E7_fits")
    f = f[f.wall_seconds.notna() & (f.wall_seconds > 0) & (f.threads == 1)]
    out = tuple(f[f.route == r].sort_values("sample_id").wall_seconds.to_numpy(dtype=float)
                for r in (ROUTE_C, ROUTE_RS))
    return out if any(len(v) for v in out) else None


def _column_names(ax, columns: Sequence[Tuple[float, str, int]]) -> None:
    """Each ``(x, name, n)`` of c: the name on the baseline of a's and b's tick labels, n under it on their x labels'.

    So the row ends on one line (two-line tick labels would set "C psmc"
    lower than "psmc-rs", whose first line has no capital). matplotlib sets
    a line of text at least as tall as "lp": a tick label's top is the
    tick's length and pad below the spine, an x label's top labelpad below
    the tick labels' bottom.
    """
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties

    renderer = ax.figure.canvas.get_renderer()

    def lp(size: float) -> Tuple[float, float]:
        _w, h, d = renderer.get_text_width_height_descent("lp", FontProperties(size=size), ismath=False)
        return h * 72.0 / ax.figure.dpi, d * 72.0 / ax.figure.dpi

    (h_tick, d_tick), (h_lab, d_lab) = lp(style.TICK_PT), lp(ax.xaxis.label.get_fontsize())
    tick_top = mpl.rcParams["xtick.major.size"] + mpl.rcParams["xtick.major.pad"]
    drops = (tick_top + h_tick - d_tick, tick_top + h_tick + mpl.rcParams["axes.labelpad"] + h_lab - d_lab)
    for x, name, n in columns:
        for s, drop in zip((name, f"({n:,})"), drops):
            ax.annotate(s, xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -drop),
                        textcoords="offset points", ha="center", va="baseline", fontsize=style.TICK_PT,
                        color=style.INK, annotation_clip=False)


def _panel_cohort(ax, c_s, rs_s) -> None:
    """c: wall time per genome over the 1000 Genomes cohort, C psmc against psmc-rs, with the ratio of medians."""
    import numpy as np
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    longest = max(float(np.max(v)) for v in (c_s, rs_s) if len(v))
    scale, unit = (60.0, "min") if longest >= 120 else (1.0, "s")
    cm, rm = c_s / scale, rs_s / scale
    rng = np.random.default_rng(SEED)
    jr = rng.uniform(-style.STRIP_JITTER, style.STRIP_JITTER, rm.size)  # psmc-rs first, then C psmc
    jc = rng.uniform(-style.STRIP_JITTER, style.STRIP_JITTER, cm.size)
    xc, xr = 0.0, 1.0
    if rm.size:
        ax.scatter(xr + jr, rm, s=COHORT_MS ** 2, marker=style.RS02.marker, c=style.BLUE, linewidths=0,
                   zorder=style.Z_UNDERLAY)
    if cm.size:  # white hairline edges keep the overlapping squares apart
        ax.scatter(xc + jc, cm, s=style.STRIP_MS ** 2, marker=style.C_PSMC.marker, c=style.ORANGE,
                   edgecolors=style.SURFACE, linewidths=COHORT_EDGE_LW, zorder=style.Z_UNDERLAY)
    medians = {}
    for x, v in ((xc, cm), (xr, rm)):
        if v.size:
            medians[x] = float(np.median(v))
            ax.plot([x - COHORT_BAR_HALF, x + COHORT_BAR_HALF], [medians[x]] * 2, color=style.INK,
                    linewidth=style.LW_DATA, solid_capstyle="butt", zorder=style.Z_DATA)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([])
    _column_names(ax, [(x, name, v.size) for x, name, v in ((xc, "C psmc", cm), (xr, "psmc-rs", rm)) if v.size])
    ticks = MaxNLocator(nbins=4, steps=[1, 1.5, 3, 5, 10]).tick_values(0, longest / scale)
    ticks = [t for t in ticks if t >= 0]
    ax.set_ylim(0, ticks[-1] * COHORT_HEAD)  # the top tick's label clear of the title, as in a and b
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_ylabel(f"Wall time per genome ({unit})")
    if cm.size:
        _lab(ax, f"{_fmt_median(medians[xc])} {unit}", (xc, float(cm.min())), 0, -3.0, ha="center", va="top",
             color=style.ORANGE_TEXT)
    if rm.size:
        _lab(ax, f"{_fmt_median(medians[xr])} {unit}", (xr, float(rm.max())), 0, 3.0, ha="center", va="bottom",
             color=style.BLUE_TEXT)
    if cm.size and rm.size:
        mc, mr = medians[xc], medians[xr]
        _gap_arrow(ax, (0.5, mc), (0.5, mr))
        # Set as in a, on two lines so that it stays in the gap between the arrow and psmc-rs's column.
        _fold_label(ax, f"{fmt_fold(mc / mr)}\nfaster", (0.5, (mc + mr) / 2.0), +1, linespacing=1.0)


def _place(fig, axes: Sequence, widths: Sequence[Optional[float]]) -> None:
    """Lay the row out: first spine at LEFT_MM (or further right, so that the first panel's ink starts no
    further left than the letters, LETTER_X_MM), GUTTER_MM of white between one panel's ink and the next's,
    the last axes ending RIGHT_MM from the edge; panels of width None share what is left equally."""
    fw, fh = fig.get_figwidth() / style.MM, fig.get_figheight() / style.MM
    y0 = style.BOTTOM_MM / fh
    h = (fh - style.TOP_MM - style.BOTTOM_MM) / fh
    for _ in range(3):  # labels anchored to the data move a little with the widths
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ext = []  # ink left of the left spine, right of the right one (mm)
        for ax in axes:
            box, ink = ax.get_position(), ax.get_tightbbox(renderer)
            ext.append((box.x0 * fw - ink.x0 / fig.dpi / style.MM,
                        max(0.0, ink.x1 / fig.dpi / style.MM - box.x1 * fw)))
        x = max(style.LEFT_MM, style.LETTER_X_MM + ext[0][0])
        fixed = sum(w for w in widths if w is not None)
        free = (fw - x - style.RIGHT_MM - fixed - ext[-1][1] - sum(r for _, r in ext[:-1])
                - sum(left for left, _ in ext[1:]) - style.GUTTER_MM * (len(axes) - 1))
        share = free / max(1, sum(w is None for w in widths))
        for i, (ax, w) in enumerate(zip(axes, widths)):
            if i:
                x += ext[i - 1][1] + style.GUTTER_MM + ext[i][0]
            w = share if w is None else w
            ax.set_position((x / fw, y0, w / fw, h))
            x += w


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e3
    size = cfg.thread_scaling_size_mb
    t2 = src.table("E3_table2")
    t2 = t2[(t2.size_mb == size) & t2.wall_median_s.notna()]
    c_rows = t2[t2.tool == "c"]
    rs = t2[t2.tool == "rs02"].sort_values("threads")  # its default M-step, not a capped arm (rs02-default)
    c_row = c_rows.iloc[0] if not c_rows.empty else None
    c_wall = float(c_row.wall_median_s) if c_row is not None else None
    c_mem = float(c_row.peak_rss_median_mb) if c_row is not None else None
    th = rs.threads.to_numpy(dtype=float)
    wall = rs.wall_median_s.to_numpy(dtype=float)
    mem = rs.peak_rss_median_mb.to_numpy(dtype=float)
    sp = rs.speedup_vs_c.to_numpy(dtype=float)
    if c_wall is not None:
        sp = np.where(np.isfinite(sp), sp, c_wall / wall)
    rows = rs if c_rows.empty else c_rows
    n_chrom = int(rows.chromosomes.iloc[0]) if not rows.empty else 0
    title = f"Simulated genome, {n_chrom} {style.TIMES} {size / n_chrom:g} Mb" if n_chrom else (
        f"Simulated genome, {size:,} Mb")
    cohort = _cohort(src)

    fig = style.figure("double", H_MM)
    ax_a = fig.add_axes((0.1, 0.2, 0.2, 0.6))
    ax_b = fig.add_axes((0.4, 0.2, 0.2, 0.6))
    axes, widths = [ax_a, ax_b], [None, None]
    mem_axis = _memory_axis(mem, c_mem)
    c_frac = c_mem / mem_axis[2] if c_mem is not None and math.isfinite(c_mem) else None
    _panel_time(ax_a, th, wall, sp, c_wall, c_frac)
    _panel_memory(ax_b, th, mem, c_mem, *mem_axis)
    style.panel_title(ax_a, title)
    style.panel_title(ax_b, title)  # the same genome: small multiples
    if cohort is not None:
        ax_c = fig.add_axes((0.8, 0.2, 0.15, 0.6))
        _panel_cohort(ax_c, *cohort)
        style.panel_title(ax_c, "1000 Genomes")
        axes.append(ax_c)
        widths.append(C_W_MM)
    _place(fig, axes, widths)
    style.panel_letters(fig, axes)
    return fig


__all__ = ["draw", "fmt_fold", "fmt_time"]
