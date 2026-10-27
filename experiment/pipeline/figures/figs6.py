"""Fig S6 (E7): the 1000 Genomes inputs and checks.

a: heterozygous SNVs per callable kb of every genome of the cohort, by
superpopulation (points jittered sideways; the line is the median); under
each superpopulation, Spearman's rho between het/kb and raw sequencing
depth among its genomes: near zero, so depth does not drive the
differences between superpopulations. b: the two routes on the default set
(52 genomes): theta of the VCF fit (the strict mask, callable bases per
bin, ``--bin-callable auto``) over theta of the psmcfa fit of the same
genome, minus one, by superpopulation, with zero for reference. The VCF
fit's theta is 8.4% higher on average (7.9-8.8%), as the psmcfa fit takes
every called bin as full, and the bases missing from the called bins (the
panel gives their share, the same in every genome: 7.6%) make theta low
(Fig 5); AFR's shift is a little smaller (8.1% against 8.5%). c: the
split-first-window check (Hilgers et al. 2025): the first window's Ne over
the next window's, under the default pattern (x) and with the first window
split into its four intervals (y; ``config.E7.split_pattern``; their
log10 Ne averaged by duration, the vertical line their range). Both axes
share their scale, so a point on the diagonal did not move. A range that
runs past the top ends in an arrow. Filled markers: genomes whose split
fit put one first-window interval at psmc-rs's upper bound on a free
lambda (10^4 N0; in production 3 of 26: HG04134, NA12878 and NA18633),
where one genome barely constrains such a short interval; the cross is
the duration-weighted mean of their other intervals, which lies near the
diagonal, so their rise comes from that one interval. d: time per fit of
every route, on a linear axis: single-thread fits by their median wall
time; psmc-rs's bootstrap (a main fit and ``config.E7.bootstrap_replicates``
replicates per run, on ``config.E7.bootstrap_threads`` threads; 3 runs) by
its median wall time times its threads over its fits, the core time of one
of its fits. C psmc ran on the default set; on one thread psmc-rs fits a
psmcfa about 14.1 times faster (the bar gives the ratio, of C psmc's
median over the default set to psmc-rs's over every genome). The note
under the legend gives NA12878's heterozygous SNVs against GIAB HG001
v4.2.1 inside the strict mask and GIAB's confident regions.

From ``E7_inputs``, ``E7_route_shift``, ``E7_split``, ``E7_fits`` (N0 of
the split fits), ``E7_curves`` (their intervals), ``E7_giab`` and
``E7_throughput``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from . import style

SUP_MARKERS = ("o", "s", "^", "D", "v")  # by config.E7.superpopulations (c)
SUP_MARK = {"linestyle": "none", "markersize": 3.4, "markeredgewidth": 0.6, "markeredgecolor": style.INK2}  # c
CROSS = {"linestyle": "none", "marker": "x", "markersize": 3.4, "markeredgewidth": 0.7, "color": style.INK}  # c
FIRST_WINDOW = 4  # intervals of the default pattern's first free lambda, split by the check (as e7.split_row)
LAMBDA_HI = 1e4  # psmc-rs's upper bound on a free lambda (src/opt.rs), Ne = 1e4 N0
AT_BOUND = 1e-6  # within this share of the bound, a lambda is at it (as psmc-rs's run report)
ROUTE_LABELS = {"c": "C psmc (psmcfa)", "rs-psmcfa": "psmc-rs (psmcfa)", "rs-vcf": "psmc-rs (VCF + mask)",
                "rs-split": "psmc-rs (split first window)", "boot-rs": "psmc-rs bootstrap, per fit"}
ROUTE_ORDER = ("c", "rs-psmcfa", "rs-vcf", "rs-split", "boot-rs")  # top down
STRIP_WIDTH = 0.36  # half the width of a superpopulation's strip (category units)
EDGE_MM = 2.0  # the least distance between d's route labels and the figure's left edge


def jitter(n: int, width: float = STRIP_WIDTH) -> List[float]:
    """``n`` sideways offsets spread evenly over ``[-width, width]`` (low-discrepancy: no seed, no clumps)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    return [width * (2.0 * ((k * phi) % 1.0) - 1.0) for k in range(n)]


def split_first_window(split, fits, curves) -> Dict[str, Tuple[bool, float]]:
    """``{sample: (at_bound, log10 first / next window Ne without the intervals at the bound)}`` of the split fits.

    ``at_bound``: one of the first window's intervals has its lambda at
    psmc-rs's upper bound. The ratio averages the other intervals' log10 Ne
    by their durations, as ``E7_split.log10_first_vs_next_split`` averages
    all four (NaN when every interval is at the bound).
    """
    n0 = {r.sample_id: float(r.n0) for r in fits[fits.route == "rs-split"].itertuples()}
    mine = curves[curves.route == "rs-split"]
    out: Dict[str, Tuple[bool, float]] = {}
    for r in split.itertuples():
        c = mine[mine.sample_id == r.sample_id].sort_values("interval")
        if len(c) <= FIRST_WINDOW or r.sample_id not in n0:
            continue
        years, ne = list(c["years"]), list(c["ne"])
        spans = [years[k + 1] - years[k] for k in range(FIRST_WINDOW)]
        bound = [ne[k] / n0[r.sample_id] >= LAMBDA_HI * (1.0 - AT_BOUND) for k in range(FIRST_WINDOW)]
        keep = [(w, math.log10(ne[k] / float(r.ne_next_split))) for k, w in enumerate(spans) if not bound[k]]
        mean = sum(w * v for w, v in keep) / sum(w for w, _ in keep) if keep else float("nan")
        out[r.sample_id] = (any(bound), mean)
    return out


def _strip(ax, groups: Sequence[str], values: Dict[str, Sequence[float]], size: float, alpha: float) -> None:
    """Every value of each group as a point jittered sideways, and the group's median as a line."""
    import numpy as np

    for i, g in enumerate(groups):
        v = np.asarray(values.get(g, []), dtype=float)
        if not len(v):
            continue
        ax.scatter(i + np.asarray(jitter(len(v))), v, s=size, color=style.INK2, alpha=alpha, linewidths=0,
                   zorder=3, rasterized=len(v) > 200)
        ax.plot([i - STRIP_WIDTH - 0.06, i + STRIP_WIDTH + 0.06], [np.median(v)] * 2, color=style.INK,
                linewidth=style.LW_EMPH, solid_capstyle="butt", zorder=4)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xticks(range(len(groups)))
    ax.tick_params(axis="x", length=0)
    style.value_grid(ax)


def _het_panel(ax, inputs, sups: List[str]) -> None:
    """a: het/kb by superpopulation, and its rank correlation with depth."""
    _strip(ax, sups, {s: inputs[inputs.superpopulation == s].het_per_callable_kb for s in sups}, size=1.2,
           alpha=0.35)
    ticks = []
    for s in sups:
        g = inputs.loc[inputs.superpopulation == s, ["het_per_callable_kb", "raw_depth"]]
        # A DataFrame's Spearman is pandas' own; a Series' imports scipy, most of a second of this figure's run.
        rho = g.corr(method="spearman").iloc[0, 1] if len(g) > 2 else float("nan")
        ticks.append(s if math.isnan(rho) else f"{s}\nρ = {rho:.2f}".replace("-", "−"))
    ax.set_xticklabels(ticks)
    ax.set_ylabel("Heterozygous SNVs per callable kb")
    ax.set_xlabel("ρ: Spearman's, with raw depth")


def _shift_panel(ax, shift, sups: List[str]) -> None:
    """b: the VCF route's theta over the psmcfa route's, by superpopulation."""
    rel = 100.0 * shift.theta_rel_vcf_vs_psmcfa
    _strip(ax, sups, {s: rel[shift.superpopulation == s] for s in sups}, size=7.0, alpha=0.8)
    ax.set_xticklabels(sups)
    style.ref_line(ax, y=0.0)
    top = float(rel.max()) if not shift.empty else 1.0
    ax.set_ylim(min(0.0, float(rel.min()) * 1.1) if not shift.empty else 0.0, max(top, 0.0) * 1.12 + 0.2)
    ax.set_ylabel(r"$\theta_\mathrm{VCF}$ / $\theta_\mathrm{psmcfa}$ $-$ 1 (%)")
    ax.set_xlabel(f"Superpopulation ({len(shift)} genomes)")
    if not shift.empty:
        lo, hi = 100.0 * shift.missing_share_of_called_bins.min(), 100.0 * shift.missing_share_of_called_bins.max()
        share = f"{lo:.1f}%" if round(lo, 1) == round(hi, 1) else f"{lo:.1f}-{hi:.1f}%"
        style.note(ax, 0.5, 0.06, f"Bases missing in the called bins:\n{share} of each genome's", ha="center",
                   va="bottom", linespacing=1.25)


def _split_panel(ax, split, robust: Dict[str, Tuple[bool, float]], sups: List[str]) -> None:
    """c: the split-first-window check, on a square of shared limits."""
    if not split.empty:
        x, y = split.log10_first_vs_next_default, split.log10_first_vs_next_split
        rob = [v for _, v in robust.values() if not math.isnan(v)]
        lo = float(min(x.min(), y.min(), split.log10_first_vs_next_split_lo.min(), *rob))
        hi = float(max(x.max(), y.max(), *rob))
        pad = 0.08 * max(hi - lo, 0.1)
        lim = (lo - pad, hi + 2.0 * pad)
        ax.plot(lim, lim, color=style.REF, linewidth=style.LW_THIN, linestyle=style.DOT, zorder=1)
        for r in split.itertuples():
            x0 = r.log10_first_vs_next_default
            ax.vlines(x0, r.log10_first_vs_next_split_lo, min(float(r.log10_first_vs_next_split_hi), lim[1]),
                      color=style.REF, linewidth=style.LW_THIN, zorder=2)
            if r.log10_first_vs_next_split_hi > lim[1]:  # the range runs past the top
                ax.annotate("", xy=(x0, lim[1]), xytext=(x0, lim[1] - 0.06 * (lim[1] - lim[0])),
                            arrowprops={"arrowstyle": "-|>,head_length=0.35,head_width=0.18", "color": style.REF,
                                        "linewidth": style.LW_THIN, "shrinkA": 0.0, "shrinkB": 0.0}, zorder=2)
        for sup, m in zip(sups, SUP_MARKERS):
            for r in split[split.superpopulation == sup].itertuples():
                at_bound, mean = robust.get(r.sample_id, (False, float("nan")))
                ax.plot(r.log10_first_vs_next_default, r.log10_first_vs_next_split, **SUP_MARK, marker=m,
                        markerfacecolor=style.INK2 if at_bound else style.SURFACE, zorder=4)
                if at_bound and not math.isnan(mean):
                    ax.plot(r.log10_first_vs_next_default, mean, **CROSS, zorder=5)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box", anchor="W")
    ax.set_xlabel(r"log$_{10}$ first / next window $N_\mathrm{e}$, default")
    ax.set_ylabel(r"log$_{10}$ first / next window $N_\mathrm{e}$, split")


def _time_panel(fig, ax, thr, replicates: int) -> None:
    """d: time per fit, linear from zero, in minutes."""
    routes = [r for r in ROUTE_ORDER if r in set(thr.route)] + sorted(set(thr.route) - set(ROUTE_ORDER))
    by = {r.route: r for r in thr.itertuples()}
    minutes, notes = [], []
    reps = replicates + 1  # a run fits the data once, then each replicate
    for route in routes:
        r = by[route]
        if route == "boot-rs":
            per_fit = float(r.wall_median_s) * int(r.threads) / reps
            minutes.append(per_fit / 60.0)
            notes.append(f"{per_fit / 60.0:.1f} min ({int(r.jobs)} runs of {reps} fits, {int(r.threads)} threads: "
                         f"{float(r.wall_median_s) / 60.0:.1f} min each)")
        else:
            minutes.append(float(r.wall_median_s) / 60.0)
            notes.append(f"{minutes[-1]:.1f} min (n = {int(r.jobs):,})")
    if "c" in by and "rs-psmcfa" in by:
        k = routes.index("c")
        notes[k] += f"; {float(by['c'].wall_median_s) / float(by['rs-psmcfa'].wall_median_s):.1f}x psmc-rs (psmcfa)"
    ypos = list(range(len(routes)))[::-1]
    colors = [style.ORANGE if by[r].tool == "c" else style.BLUE for r in routes]
    ax.barh(ypos, minutes, color=colors, height=0.62, zorder=3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([ROUTE_LABELS.get(r, r) for r in routes])
    ax.tick_params(axis="y", length=0)
    # The route labels are wider than the figure's left margin: move the panel right until they clear the edge.
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    labels_x0 = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels()) / fig.bbox.width
    need = EDGE_MM / style.WIDTH_MM["double"] - labels_x0
    if need > 0:
        pos = ax.get_position()
        ax.set_position([pos.x0 + need, pos.y0, pos.width - need, pos.height])
    ax.set_xlim(0.0, max(minutes or [1.0]) * 1.02)
    ax.set_xlabel("Time per fit (min): median wall time on one thread; bootstrap: core time per fit")
    style.value_grid(ax, "x")
    span = ax.get_xlim()[1]
    for yy, m, text in zip(ypos, minutes, notes):
        inside = m > 0.6 * span
        ax.text(m - 0.01 * span if inside else m + 0.01 * span, yy, text, va="center",
                ha="right" if inside else "left", fontsize=style.TEXT_PT, color=style.SURFACE if inside else style.INK2,
                zorder=4)


def _key_panel(ax, giab, sups: List[str]) -> None:
    """The key to c, and the GIAB comparison."""
    from matplotlib.lines import Line2D

    ax.axis("off")
    handles: List = [Line2D([], [], **SUP_MARK, marker=m, markerfacecolor=style.SURFACE, label=s)
                     for s, m in zip(sups, SUP_MARKERS)]
    handles += [
        Line2D([], [], **SUP_MARK, marker="o", markerfacecolor=style.INK2,
               label=r"An interval at psmc-rs's bound ($\lambda$ = 10$^4$)"),
        Line2D([], [], **CROSS, label="Mean of the other intervals"),
        Line2D([], [], color=style.REF, linewidth=style.LW_THIN, label="Range of the four split intervals"),
        Line2D([], [], color=style.REF, linewidth=style.LW_THIN, linestyle=style.DOT, label="No change (diagonal)"),
    ]
    style.legend(ax, handles, loc="upper left", bbox_to_anchor=(0.0, 1.04), title="c: superpopulation and keys",
                 handlelength=1.6, alignment="left")
    lines = []
    for r in giab.itertuples():
        mb = int(r.eval_bases) / 1e6
        size = f"{mb:,.0f} Mb" if mb >= 10 else f"{mb:.2g} Mb"
        lines += [f"{r.sample_id} against GIAB HG001 v4.2.1:", f"{int(r.truth_het_snvs):,} het SNVs in {size};",
                  f"precision {float(r.precision):.4f}, recall {float(r.recall):.4f}"]
    style.note(ax, 0.0, 0.0, "\n".join(lines) if lines else "No GIAB comparison", va="bottom", color=style.INK,
               linespacing=1.3)


def draw(src):
    cfg = src.profile.e7
    sups = list(cfg.superpopulations)
    split = src.table("E7_split")
    fig = style.figure("double", 110.0)
    gs = style.grid(fig, 2, 3, left_mm=13.0, right_mm=3.0, top_mm=5.0, bottom_mm=9.0, wspace_mm=12.0, hspace_mm=15.0)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))
    d = fig.add_subplot(gs[1, 0:2])
    f = fig.add_subplot(gs[1, 2])
    _het_panel(a, src.table("E7_inputs"), sups)
    _shift_panel(b, src.table("E7_route_shift"), sups)
    _split_panel(c, split, split_first_window(split, src.table("E7_fits"), src.table("E7_curves")), sups)
    _time_panel(fig, d, src.table("E7_throughput"), int(cfg.bootstrap_replicates))
    _key_panel(f, src.table("E7_giab"), sups)
    style.panel_letters(fig, [a, b, c, d])
    return fig


__all__ = ["draw", "jitter", "split_first_window"]
