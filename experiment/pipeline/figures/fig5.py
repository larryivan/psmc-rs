"""Fig 5 (E4): how the route from a VCF and a callable mask to PSMC biases the inference.

Routes (colour, marker): the exact route (psmc-rs reading the VCF with
``--vcf-mask``), the classic consensus FASTA -> fq2psmcfa route (identical
bins by construction: drawn as open markers and a dotted line over the exact
route), multihetsep (missing sites placed at interval starts), and the VCF
read without its mask. Masks: no missing data, and 10% or 30% of positions
uncallable in clustered (mean 20 kb) or dispersed (mean 20 bp) runs.

* a: relative bias of theta per bin against the exact route on the same
  replicate without missing data (every replicate, and the mean);
* b: RMSE of log10 Ne against the truth on the informative window;
* c: share of bins whose state (called, heterozygous, missing) differs from
  the exact route's;
* d: the curves of one replicate with one mask, and the truth;
* e, f: bias over time, log10(Ne / true Ne), median over replicates, for two
  masks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .. import config
from . import style
from .data import Source, curves, decades, informative_window, log10_ratio, time_grid, truth_xy

MASK_LABEL = {"none": "None"}


def mask_label(name: str, structure: str, fraction: float, two_lines: bool = True) -> str:
    if name == "none" or structure == "none":
        return "No missing\ndata" if two_lines else "no missing data"
    sep = "\n" if two_lines else " "
    return f"{structure.capitalize()}{sep}{100 * fraction:.0f}%"


def masks_of(routes) -> List[Tuple[str, str, float]]:
    """``(mask, structure, missing fraction)`` in the figure's order: none, then clustered, then dispersed."""
    seen = {}
    for r in routes.itertuples():
        seen[str(r.mask)] = (str(r.structure), float(r.missing_fraction))
    order = {"none": 0, "clustered": 1, "dispersed": 2}
    items = sorted(seen.items(), key=lambda kv: (order.get(kv[1][0], 3), kv[1][1], kv[0]))
    return [(m, s, f) for m, (s, f) in items]


def routes_of(routes) -> List[str]:
    present = set(routes["route"])
    return [r for r in style.ROUTE_ORDER if r in present]


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


def mask_axis(ax, masks) -> None:
    """Masks on x: the missing fraction under each group, the structure under its group."""
    from matplotlib.transforms import blended_transform_factory

    ax.set_xlim(-0.55, len(masks) - 0.45)
    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels(["none" if s == "none" else f"{100 * f:.0f}%" for _, s, f in masks], fontsize=style.TICK_PT)
    ax.tick_params(axis="x", length=0)
    for i in range(1, len(masks)):
        ax.axvline(i - 0.5, color=style.GRID, linewidth=style.LW_GRID, zorder=0)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    groups: Dict[str, List[int]] = {}
    for i, (_, s, _) in enumerate(masks):
        if s != "none":
            groups.setdefault(s, []).append(i)
    for s, idx in groups.items():
        lo, hi = min(idx) - 0.4, max(idx) + 0.4
        ax.plot([lo, hi], [-0.13, -0.13], color=style.MUTED, linewidth=style.LW_GRID, transform=trans,
                clip_on=False)
        ax.text((lo + hi) / 2, -0.15, s, ha="center", va="top", fontsize=style.TICK_PT, color=style.INK2,
                transform=trans)


MARKED_ROUTES = ("multihetsep", "nomask")  # the light colours carry their marker along the line too


def route_line(ax, x, y, route: str, lw: float, xlim: Tuple[float, float], n_marks: int = 6) -> None:
    """A route's step line; the routes in light colours also get their marker at a few log-spaced times."""
    import numpy as np

    ent = style.ROUTES[route]
    ax.plot(x, y, drawstyle="steps-post", color=ent.color, linestyle=ent.linestyle, linewidth=lw,
            zorder=4 if route == "fq2psmcfa" else 3)
    if route not in MARKED_ROUTES:
        return
    shift = (MARKED_ROUTES.index(route) + 0.5) / (len(MARKED_ROUTES) * (n_marks + 1))
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
    cdf = src.table("E4_curves")
    masks = masks_of(routes)
    route_names = routes_of(routes)
    fits = curves(cdf, ["replicate", "mask", "route"])
    xlim = figs.sim_years
    window = informative_window(cfg.model, cfg.grid, g)
    grid = time_grid(xlim, figs.curve_grid_points)

    fig = style.figure("double", 120)
    gs = style.grid(fig, 2, 3, left_mm=13.0, right_mm=2.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=13.0,
                    hspace_mm=15.0, height_ratios=[1.0, 1.1])
    ax_t, ax_r, ax_b = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_c, ax_e, ax_f = (fig.add_subplot(gs[1, i]) for i in range(3))

    dot_panel(ax_t, routes, "theta_rel_bias", 100.0, masks, route_names)
    style.ref_line(ax_t, y=0.0)
    ax_t.set_ylabel("Bias in θ (%)")
    dot_panel(ax_r, routes, "rmse_log10_ne_vs_truth", 1.0, masks, route_names)
    ax_r.set_ylim(bottom=0.0)
    ax_r.set_ylabel("RMSE of log$_{10}$ $N_\\mathrm{e}$")
    dot_panel(ax_b, routes, "bins_differ_fraction", 100.0, masks, route_names)
    ax_b.set_ylim(bottom=0.0)
    ax_b.set_ylabel("Bins differing from exact (%)")

    # d: one replicate, one mask.
    rep, mask = figs.e4_example_replicate, figs.e4_example_mask
    if not any(k[1] == mask for k in fits):
        mask = masks[-1][0]
    style.shade_outside(ax_c, window, xlim)
    vals: List[float] = []
    for route in route_names:
        c = fits.get((rep, mask, route))
        if c is None:
            continue
        ent = style.ROUTES[route]
        x, y = style.step_xy(c.years, c.ne, xlim)
        lw = style.LW_BOLD if route == "exact" else style.LW
        route_line(ax_c, x, y, route, lw, xlim)
        inside = (grid >= window[0]) & (grid <= window[1])
        vals += list(c.at(grid[inside]))
    tx, ty = truth_xy(cfg.model, g, xlim)
    ax_c.plot(tx, ty, color=style.INK, linestyle=style.DASH, linewidth=style.LW, zorder=5)
    vals += [float(ty.min()), float(ty.max())]
    style.years_axis(ax_c, xlim)
    style.ne_axis(ax_c, decades(min(vals) / 1.3, max(vals) * 1.3))
    m_struct = {m: (s, f) for m, s, f in masks}
    style.note(ax_c, 0.97, 0.97, f"{mask_label(mask, *m_struct.get(mask, ('', 0.0)), two_lines=False)},\n"
                                 f"replicate {rep}", ha="right", va="top")

    # e, f: the bias each route adds, over time: log10(Ne_route / Ne_exact) on
    # the same replicate and mask, median over replicates. (PSMC's own error
    # against the truth, shared by every route, is in b and d.)
    bias_axes = [ax_e, ax_f]
    bias_masks = [m for m in figs.e4_bias_masks if m in m_struct][: len(bias_axes)]
    inside = (grid >= window[0]) & (grid <= window[1])
    blim = 0.02
    series: Dict[Tuple[str, str], np.ndarray] = {}
    n_reps: Dict[str, int] = {}
    for mask_b in bias_masks:
        for route in route_names:
            reps = sorted(k[0] for k in fits if k[1] == mask_b and k[2] == route and (k[0], mask_b, "exact") in fits)
            if not reps:
                continue
            n_reps[mask_b] = len(reps)
            b = np.median(np.vstack([log10_ratio(fits[(r, mask_b, route)], fits[(r, mask_b, "exact")], grid)
                                     for r in reps]), axis=0)
            series[(mask_b, route)] = b
            blim = max(blim, float(np.max(np.abs(b[inside]))))
    blim = blim * 1.2
    for ax, mask_b in zip(bias_axes, bias_masks):
        style.shade_outside(ax, window, xlim)
        for route in route_names:
            b = series.get((mask_b, route))
            if b is None:
                continue
            ent = style.ROUTES[route]
            lw = style.LW_BOLD if route == "exact" else style.LW
            route_line(ax, grid, b, route, lw, xlim)
        style.years_axis(ax, xlim)
        ax.set_ylim(-blim, blim)
        ax.set_ylabel("log$_{10}$($N_\\mathrm{e}$ / $N_\\mathrm{e}$ of exact route)")
        style.note(ax, 0.03, 0.97, f"{mask_label(mask_b, *m_struct[mask_b], two_lines=False)},\n"
                                   f"median of {n_reps.get(mask_b, 0)} replicates", ha="left", va="top")
    for ax in bias_axes[len(bias_masks):]:
        ax.set_axis_off()

    handles = [style.handle(style.ROUTES[r], kind="both", linewidth=style.LW) for r in route_names]
    handles.append(style.handle(style.TRUTH, linewidth=style.LW))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, [ax_t, ax_r, ax_b, ax_c, ax_e, ax_f])
    return fig
