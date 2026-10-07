"""Fig 5 (E4): partly callable bins bias PSMC, and psmc-rs's per-bin model removes most of the bias.

Routes (colour, marker): psmc-rs on the VCF with its callable mask, each
called bin weighed by the callable bases it holds (``--bin-callable auto``,
psmc-rs's default; blue); the same bins with every called bin taken as full
(``--bin-callable ignore``, dark blue, dashed), which is what fq2psmcfa and
C psmc do: the classic consensus FASTA -> fq2psmcfa route has the same bins
(asserted bin for bin) and the same fit (asserted, every fitted value) and
is not drawn separately; multihetsep (``psmc-rs convert --multihetsep``,
read back with its uncalled sites placed at the start of each interval and
every called bin taken as full, ``--bin-callable auto``; yellow) and the
VCF read without its mask (pink). Masks: no missing data, and 10% or 30% of positions
uncallable in clustered (mean 20 kb) or dispersed (mean 20 bp) runs.

fq2psmcfa's rule calls a bin missing only when more than 90% of its bases
are uncallable, so a dispersed mask leaves almost every bin called with
only about (1 - f) of its bases. Taken as full, they make theta per bin,
and every time and size with it, low by somewhat more than the share of
bases missing from the called bins (about 1.2 times it in E4: theta -12%
with 10% and -35% with 30% dispersed missing); weighed by their callable
bases they are not. Clustered missing data empty whole bins, which
both models drop. multihetsep does not say which sites of an interval are
uncalled; psmc-rs places them at the start of each interval, which turns
dispersed missing data into runs of N bins after heterozygous sites and
biases theta low, less so with clustered missing data. Without its mask a VCF
counts every uncallable site as homozygous.

* a: theta per bin against the complete-data fit of the same genome, as a
  change in percent (every replicate, and the mean);
* b: RMSE of log10 Ne against the truth on the informative window;
* c: RMSE of log10 Ne against the complete-data fit of the same genome,
  exact over log time on the informative window;
* d: the curves of one replicate with the dispersed 30% mask, its
  complete-data fit (grey) and the truth;
* e, f: bias over time, mean over replicates of log10(Ne / true Ne), with
  the largest clustered and dispersed masks; grey: the complete data.

In d-f every called bin taken as full (dark blue, dashed) is drawn above
the other routes: with dispersed missing data its curve lies on the VCF
read without its mask (pink), and with clustered missing data on psmc-rs's
default (blue), which would otherwise hide it. The limits of e and f are
those of the data, shared by both panels.

Everything comes from ``E4_routes``, ``E4_curves`` and ``E4_bias``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .. import config
from . import style
from .data import Source, curves, decades, informative_window, truth_xy

DRAWN = ("mask_auto", "mask_ignore", "multihetsep", "nomask")  # fq2psmcfa fits as mask_ignore (asserted)
COMPLETE = config.E4_CONTROL_MASK  # the no-missing control
COMPLETE_ROUTE = "mask_auto"  # the complete-data fit
BOLD = "mask_auto"  # the route the figure is about
MARKED = ("multihetsep", "nomask")  # light colours carry their marker along the line too
REFERENCE_LW = 2.0  # the complete data, grey, under the routes
# Stacking of the routes' lines in d-f: every called bin taken as full (dashed) on top, as it coincides
# with the VCF without its mask (dispersed masks) or with the default (clustered masks).
LINE_Z = {"nomask": 3.0, "multihetsep": 3.1, "fq2psmcfa": 3.2, "mask_auto": 3.3, "mask_ignore": 3.4}
MASK_TITLE = "Uncallable positions in the mask"


def masks_of(routes, cfg: config.E4) -> List[Tuple[str, str, float]]:
    """``(mask, structure, missing fraction)``: the control, then clustered and dispersed by fraction."""
    specs = {m.name: (m.structure, m.missing_fraction) for m in cfg.masks}
    present = list(dict.fromkeys(routes["mask"]))
    items = [(m, *specs.get(m, ("none", 0.0))) for m in present]
    order = {"none": 0, "clustered": 1, "dispersed": 2}
    return sorted(items, key=lambda x: (order.get(x[1], 3), x[2], x[0]))


def routes_of(routes) -> List[str]:
    present = set(routes["route"])
    return [r for r in DRAWN if r in present]


def mask_label(structure: str, fraction: float) -> str:
    return "no missing data" if structure == "none" else f"{structure} {100 * fraction:.0f}%"


def mask_axis(ax, masks) -> None:
    """Masks on x: the missing fraction under each group, the structure under its group."""
    from matplotlib.transforms import blended_transform_factory

    ax.set_xlim(-0.55, len(masks) - 0.45)
    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels(["none" if s == "none" else f"{100 * f:.0f}%" for _, s, f in masks], fontsize=style.TICK_PT)
    ax.tick_params(axis="x", length=0)
    ax.text(0.5, -0.235, MASK_TITLE, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK,
            transform=ax.transAxes)
    for i in range(1, len(masks)):
        ax.axvline(i - 0.5, color=style.GRID, linewidth=style.LW_GRID, zorder=0)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    groups: Dict[str, List[int]] = {}
    for i, (_, s, _) in enumerate(masks):
        if s != "none":
            groups.setdefault(s, []).append(i)
    for s, idx in groups.items():
        lo, hi = min(idx) - 0.4, max(idx) + 0.4
        ax.plot([lo, hi], [-0.13, -0.13], color=style.MUTED, linewidth=style.LW_GRID, transform=trans, clip_on=False)
        ax.text((lo + hi) / 2, -0.15, s, ha="center", va="top", fontsize=style.TEXT_PT, color=style.INK2,
                transform=trans)


def dot_panel(ax, routes, column: str, scale: float, masks, route_names) -> None:
    """One value per replicate by mask (groups) and route (offsets), with the means."""
    import numpy as np

    width = 0.8 / len(route_names)
    for gi, route in enumerate(route_names):
        ent = style.ROUTES[route]
        for i, (mask, _, _) in enumerate(masks):
            v = routes[(routes["mask"] == mask) & (routes.route == route)][column].to_numpy(dtype=float) * scale
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            x0 = i + (gi - (len(route_names) - 1) / 2) * width
            jitter = (np.arange(v.size) - (v.size - 1) / 2) * (0.5 * width / max(v.size, 2))
            ax.plot(x0 + jitter, v, **ent.points(markersize=2.4, zorder=4))
            m = float(np.mean(v))
            ax.plot([x0 - 0.42 * width, x0 + 0.42 * width], [m, m], color=style.INK, linewidth=style.LW, zorder=5)
    mask_axis(ax, masks)
    ax.grid(True, which="major", axis="y", color=style.GRID, linewidth=style.LW_GRID)


def route_line(ax, x, y, route: str, xlim: Tuple[float, float], n_marks: int = 6, **kw) -> None:
    """A route's line; the routes in light colours also get their marker at a few log-spaced times."""
    import numpy as np

    ent = style.ROUTES[route]
    style_kw = {"color": ent.color, "linestyle": ent.linestyle, "linewidth": style.LW, "zorder": LINE_Z[route]}
    style_kw.update(kw)
    ax.plot(x, y, **style_kw)
    if route not in MARKED:
        return
    shift = (MARKED.index(route) + 0.5) / (len(MARKED) * (n_marks + 1))
    span = np.log10(xlim[1] / xlim[0])
    xm = xlim[0] * 10 ** (span * (np.arange(1, n_marks + 1) / (n_marks + 1) + shift))
    idx = np.clip(np.searchsorted(np.asarray(x), xm, side="right") - 1, 0, len(y) - 1)
    ax.plot(xm, np.asarray(y)[idx], **ent.points(markersize=2.8, zorder=5))


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e4
    figs = config.FIGURES
    g = cfg.scaling.gen_years
    routes = src.table("E4_routes")
    fits = curves(src.table("E4_curves"), ["replicate", "mask", "route"])
    bias = src.table("E4_bias")
    masks = masks_of(routes, cfg)
    route_names = routes_of(routes)
    xlim = figs.sim_years
    window = informative_window(cfg.model, cfg.grid, g)

    fig = style.figure("double", 126)
    gs = style.grid(fig, 2, 3, left_mm=13.0, right_mm=2.5, top_mm=12.0, bottom_mm=9.0, wspace_mm=13.0,
                    hspace_mm=18.0, height_ratios=[1.0, 1.1])
    ax_t, ax_r, ax_b = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_c, ax_e, ax_f = (fig.add_subplot(gs[1, i]) for i in range(3))

    dot_panel(ax_t, routes, "theta_rel_complete", 100.0, masks, route_names)
    style.ref_line(ax_t, y=0.0)
    ax_t.set_ylabel("θ, change from the complete data (%)")
    dot_panel(ax_r, routes, "rmse_log10_ne_vs_truth", 1.0, masks, route_names)
    ax_r.set_ylim(bottom=0.0)
    ax_r.set_ylabel("RMSE of log$_{10}$ $N_\\mathrm{e}$ vs the truth")
    dot_panel(ax_b, routes, "rmse_log10_ne_vs_complete", 1.0, masks, route_names)
    ax_b.set_ylim(bottom=0.0)
    ax_b.set_ylabel("RMSE of log$_{10}$ $N_\\mathrm{e}$ vs complete data")

    # d: one replicate, one mask, with its complete-data fit.
    rep, mask = figs.e4_example_replicate, figs.e4_example_mask
    if not any(k[1] == mask for k in fits):
        mask = masks[-1][0]
    specs = {m: (s, f) for m, s, f in masks}
    style.shade_outside(ax_c, window, xlim)
    vals: List[float] = []
    grid = np.logspace(np.log10(window[0]), np.log10(window[1]), 200)
    complete = fits.get((rep, COMPLETE, COMPLETE_ROUTE))
    if complete is not None:
        x, y = style.step_xy(complete.years, complete.ne, xlim)
        ax_c.plot(x, y, color=style.MUTED, linewidth=REFERENCE_LW, drawstyle="steps-post", zorder=2,
                  solid_capstyle="butt")
        vals += list(complete.at(grid))
    for route in route_names:
        c = fits.get((rep, mask, route))
        if c is None:
            continue
        x, y = style.step_xy(c.years, c.ne, xlim)
        route_line(ax_c, x, y, route, xlim, drawstyle="steps-post",
                   linewidth=style.LW_BOLD if route == BOLD else style.LW)
        vals += list(c.at(grid))
    tx, ty = truth_xy(cfg.model, g, xlim)
    ax_c.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=6)
    vals += [float(ty.min()), float(ty.max())]
    style.years_axis(ax_c, xlim)
    style.ne_axis(ax_c, decades(min(vals) / 1.3, max(vals) * 1.3))
    style.note(ax_c, 0.97, 0.97, f"{mask_label(*specs.get(mask, ('none', 0.0)))},\nreplicate {rep}",
               ha="right", va="top")

    # e, f: mean bias against the truth over time, with the complete data for reference.
    bias_masks = [m for m in figs.e4_bias_masks if m in specs][:2]
    reference = bias[(bias["mask"] == COMPLETE) & (bias.route == COMPLETE_ROUTE)].sort_values("years")
    drawn = [bias[(bias["mask"] == m) & bias.route.isin(route_names)]["log10_bias_mean"] for m in bias_masks]
    values = np.concatenate([v.to_numpy(dtype=float) for v in drawn + [reference["log10_bias_mean"]]] + [[0.0]])
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    pad = 0.08 * max(hi - lo, 0.1)
    ylim = (lo - pad, hi + pad)
    for ax, m in zip((ax_e, ax_f), bias_masks):
        if not reference.empty:
            ax.plot(reference["years"], reference["log10_bias_mean"], color=style.MUTED, linewidth=REFERENCE_LW,
                    zorder=2, solid_capstyle="butt", drawstyle="steps-post")
        for route in route_names:
            sub = bias[(bias["mask"] == m) & (bias.route == route)].sort_values("years")
            if sub.empty:
                continue
            route_line(ax, sub["years"].to_numpy(), sub["log10_bias_mean"].to_numpy(), route,
                       (float(sub["years"].min()), float(sub["years"].max())), drawstyle="steps-post",
                       linewidth=style.LW_BOLD if route == BOLD else style.LW)
        style.ref_line(ax, y=0.0)
        style.years_axis(ax, (window[0] / 1.15, window[1] * 1.15))
        ax.set_ylim(*ylim)
        ax.set_ylabel("Bias, mean log$_{10}$($N_\\mathrm{e}$ / true $N_\\mathrm{e}$)")
        n = int(bias[bias["mask"] == m]["replicates"].max())
        style.note(ax, 0.97, 0.03, f"{mask_label(*specs[m])},\nmean of {n} replicate{'s' if n > 1 else ''}",
                   ha="right", va="bottom")
    for ax in (ax_e, ax_f)[len(bias_masks):]:
        ax.set_axis_off()

    from matplotlib.lines import Line2D

    handles = [style.handle(style.ROUTES[r], kind="both", linewidth=style.LW) for r in route_names]
    handles.append(style.handle(style.TRUTH, linewidth=style.LW))
    handles.append(Line2D([], [], color=style.MUTED, linewidth=REFERENCE_LW, label="Complete-data fit (d-f)"))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style.panel_letters(fig, [ax_t, ax_r, ax_b, ax_c, ax_e, ax_f], align_cols=True)
    return fig


__all__ = ["DRAWN", "draw", "mask_label", "masks_of", "routes_of"]
