"""Fig 3 (E2): four real genomes fitted by C psmc and psmc-rs 0.2, with psmc-rs's bootstrap.

One panel per genome, in ``config.E2.panel_order``: C psmc's fit (orange,
wider, underneath) and psmc-rs's (blue), the pointwise 95% band of
psmc-rs's bootstrap at the default trunks (blue, light) and, for a genome
whose default trunks are few (Papuan chr1: 5 trunks of 50 Mb), the band at
5 Mb trunks too (dark blue outline). Each genome is scaled with its own mu
and g (given in the panel; the turtle values are unverified, see
``config._TURTLE_SCALING_SOURCE``). Times outside the window on which the
tools are compared (1% to 99% of pairwise coalescence times under C psmc's
fitted model) are shaded; the RMSE of log10 Ne between the tools on it is
given in the panel, with the bootstrap's replicates and trunks. The human
sample is chromosome 1 only, and the panel says so.

Everything comes from E2's tables: ``E2_curves`` (the fits), ``E2_bands``
(the bands, on a dense grid), ``E2_bootstrap`` (replicates, trunks),
``E2_rs_vs_c`` (window, RMSE) and ``E2_fit_summary`` (scaling).
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .. import config
from . import style
from .data import Source, curves, fmt_sig

C_LW, RS_LW = 2.0, 0.8
MARGIN_DECADES = 0.5  # the x range: the comparison window widened by this much on each side
FINE_STYLE = (0, (2.4, 1.2))


def genome_order(cfg: config.E2, present) -> List[config.RealGenome]:
    """The genomes of the tables in the panel order of the configuration (then any others)."""
    by_id = {g.sample_id: g for g in cfg.genomes}
    ids = [s for s in cfg.panel_order if s in present] + sorted(s for s in present if s not in cfg.panel_order)
    return [by_id[s] for s in ids if s in by_id]


def where(gnm: config.RealGenome) -> str:
    """The sample and what of it the input holds (from the label after the species)."""
    rest = gnm.label[len(gnm.species):].strip() if gnm.label.startswith(gnm.species) else gnm.label
    return rest.strip("()") or gnm.sample_id


def sci(x: float) -> str:
    mant, exp = f"{x:.2e}".split("e")
    return rf"{float(mant):g}$\times$10$^{{{int(exp)}}}$"


def trunk_text(row) -> str:
    """``5 trunks of 50 Mb`` (and a warning when they are few)."""
    mb = float(row.trunk_mb)
    text = f"{int(row.trunks)} trunks of {mb:g} Mb"
    return text + " (few: coarse)" if str(row.coarse_band) == "True" else text


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e2
    fits = curves(src.table("E2_curves"), ["sample_id", "tool"])
    bands = src.table("E2_bands")
    boots = src.table("E2_bootstrap")
    pairs = {r.sample_id: r for r in src.table("E2_rs_vs_c").itertuples()}
    scaling = {r.sample_id: r for r in src.table("E2_fit_summary").itertuples() if r.tool == "c"}
    genomes = genome_order(cfg, {s for s, _ in fits})

    n = len(genomes)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig = style.figure("double", 12.0 + 50.0 * nrow)
    gs = style.grid(fig, nrow, ncol, left_mm=13.0, right_mm=3.0, top_mm=8.5, bottom_mm=8.5, hspace_mm=9.0,
                    wspace_mm=10.0)
    axes = []
    for i, gnm in enumerate(genomes):
        sid = gnm.sample_id
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        axes.append(ax)
        pair = pairs[sid]
        window = (float(pair.window_lo_years), float(pair.window_hi_years))
        xlim = (window[0] / 10**MARGIN_DECADES, window[1] * 10**MARGIN_DECADES)
        style.shade_outside(ax, window, xlim)
        mine = boots[boots.sample_id == sid].sort_values("trunk_bins", ascending=False)
        vals: List[float] = []
        for row in mine.itertuples():
            b = bands[(bands.sample_id == sid) & (bands.trunk_bins == row.trunk_bins)].sort_values("years")
            x = b["years"].to_numpy()
            lo, hi = b["ne_band_lo"].to_numpy(), b["ne_band_hi"].to_numpy()
            keep = (x >= xlim[0]) & (x <= xlim[1])
            if str(row.default_trunks) == "True":
                ax.fill_between(x[keep], lo[keep], hi[keep], step="post", color=style.BLUE, alpha=style.BAND_ALPHA,
                                linewidth=0, zorder=2)
            else:
                for edge in (lo, hi):
                    ax.plot(x[keep], edge[keep], drawstyle="steps-post", color=style.BLUE_650, linewidth=0.6,
                            linestyle=FINE_STYLE, zorder=3)
            inside = (x >= window[0]) & (x <= window[1])
            vals += [float(lo[inside].min()), float(hi[inside].max())]
        for tool, ent, lw, z in (("c", style.C_PSMC, C_LW, 4), ("rs", style.RS02, RS_LW, 5)):
            c = fits.get((sid, tool))
            if c is None:
                continue
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linewidth=lw, zorder=z)
            grid = np.logspace(math.log10(window[0]), math.log10(window[1]), 200)
            vals += [float(c.at(grid).min()), float(c.at(grid).max())]
        lo, hi = min(vals) / 1.6, max(vals) * 1.6
        style.years_axis(ax, xlim, label=i // ncol == nrow - 1)
        style.ne_axis(ax, (lo, hi), label=style.NE_LABEL if i % ncol == 0 else None)
        sc = scaling.get(sid)
        mu = float(sc.mu) if sc is not None else gnm.scaling.mu
        g = float(sc.gen_years) if sc is not None else gnm.scaling.gen_years
        style.note(ax, 0.03, 0.97, gnm.species, color=style.INK, size=style.TEXT_PT, fontstyle="italic")
        style.note(ax, 0.03, 0.885, f"{where(gnm)}\n" + rf"$\mu$ = {sci(mu)}, g = {g:g} y", linespacing=1.25)
        lines = [f"RMSE psmc-rs vs C: {fmt_sig(float(pair.rmse_log10_ne_rs_vs_c))}"]
        for row in mine.itertuples():
            lines.append(f"{int(row.replicates)} replicates, {trunk_text(row)}")
        style.note(ax, 0.97, 0.03, "\n".join(lines), ha="right", va="bottom", linespacing=1.25)
    for j in range(n, nrow * ncol):
        fig.add_subplot(gs[j // ncol, j % ncol]).set_axis_off()

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    default_mb = cfg.bootstrap_trunk_bins * cfg.inference.bin_size / 1e6
    handles = [
        style.handle(style.C_PSMC, linewidth=C_LW),
        style.handle(style.RS02, linewidth=RS_LW),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"Bootstrap 95% band, {default_mb:g} Mb trunks"),
    ]
    if (boots["default_trunks"].astype(str) == "False").any():
        fine_mb = float(boots[boots["default_trunks"].astype(str) == "False"]["trunk_mb"].iloc[0])
        handles.append(Line2D([], [], color=style.BLUE_650, linewidth=0.6, linestyle=FINE_STYLE,
                              label=f"Bootstrap 95% band, {fine_mb:g} Mb trunks"))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles))
    style.panel_letters(fig, axes)
    return fig


def window_of(src: Source, sample_id: str) -> Tuple[float, float]:
    """The comparison window of a genome (years), from ``E2_rs_vs_c``."""
    r = next(r for r in src.table("E2_rs_vs_c").itertuples() if r.sample_id == sample_id)
    return float(r.window_lo_years), float(r.window_hi_years)


__all__ = ["draw", "genome_order", "trunk_text", "where", "window_of"]
