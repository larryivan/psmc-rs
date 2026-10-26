"""Fig S5 (E7): every 1000 Genomes population, every genome's spread.

One panel per population, ordered by superpopulation
(``config.E7.superpopulations``) and code: the median of the population's
genomes (fits of their psmcfa, ``fit-rs-psmcfa``; blue; the population's
curve in fig7_1000_genomes b-f) with the spread between them
(``config.E7.spread``: a band of the individuals, not a confidence
interval; n, the genomes, in the panel), and the median of the VCF fits
(the strict mask, callable bases per bin, ``fit-rs-vcf``) of its two
default genomes (dark blue, thin).
A psmcfa records no callable bases per bin, so its fit takes every called
bin as full (7.6% of the called bins' bases are missing here), which makes
theta, and every size and time, low (fig5_input_preparation). On the same genome the VCF
fit's theta is 8.4% higher (7.9-8.8% over the 52 default genomes) and its
Ne 0.045 log10 (11%) higher on average over the window (Fig S6b). The dark
line therefore runs about 0.04 log10 above, and slightly later than, the
blue one from about 60 kya on, mostly outside the narrow spread; the VCF
route is the less biased one (fig5_input_preparation: 0.7% low in theta with 10% missing),
so the blue curves are the ones a little low. In the first window two
genomes are a small sample of a wide spread, and the dark line strays to
either side of the blue one. All panels share their axes. Times outside
the population's median comparison window are shaded grey. Admixed
populations (``config.E7.admixed``) are marked. Curves are scaled with
``config.E7.scaling``.

From ``E7_pop_bands`` (the groups) and ``E7_samples`` (the populations).
"""

from __future__ import annotations

import math

from . import style

NCOLS = 6


def vcf_label(counts) -> str:
    """The legend label of the VCF route's median, with its genomes per population when they are the same."""
    counts = set(counts)
    if len(counts) == 1:
        k = counts.pop()
        what = f"its {k} default genome's" if k == 1 else f"its {k} default genomes'"
    else:
        what = "its default genomes'"
    return f"Median of {what} VCF fits (callable bases per bin)"


def draw(src):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = src.profile.e7
    bands = src.table("E7_pop_bands")
    samples = src.table("E7_samples")
    xlim = tuple(cfg.figure_years)
    pops = bands[bands.group_kind == "population"]
    sup_of = dict(zip(samples.population, samples.superpopulation))
    sups = list(cfg.superpopulations)
    order = sorted(set(pops.group), key=lambda p: (sups.index(sup_of[p]) if sup_of.get(p, "") in sups else 99, p))
    inx = pops[(pops.years >= xlim[0]) & (pops.years <= xlim[1])]
    shown = inx[inx.route == "rs-psmcfa"]
    lo = float(shown.ne_lo.min()) if not shown.empty else 1e3
    hi = float(shown.ne_hi.max()) if not shown.empty else 1e5
    ylim = (lo / 1.5, hi * 3.0)
    # Only the inner decades are labelled: the edge labels of neighbouring panels would touch.
    inner = [10.0**e for e in range(math.ceil(math.log10(xlim[0]) + 1e-9), math.floor(math.log10(xlim[1]) - 1e-9) + 1)
             if xlim[0] < 10.0**e < xlim[1]]
    nrows = max(math.ceil(len(order) / NCOLS), 1)
    fig = style.figure("double", 30.0 * nrows + 22.0)
    gs = style.grid(fig, nrows, NCOLS, left_mm=13.0, right_mm=3.0, top_mm=14.0, bottom_mm=8.5, wspace_mm=3.0,
                    hspace_mm=4.0)
    axes = []
    for k, pop in enumerate(order):
        ax = fig.add_subplot(gs[k // NCOLS, k % NCOLS])
        axes.append(ax)
        psmcfa = pops[(pops.group == pop) & (pops.route == "rs-psmcfa")].sort_values("years")
        vcf = pops[(pops.group == pop) & (pops.route == "rs-vcf")].sort_values("years")
        if not psmcfa.empty:
            window = (float(psmcfa.window_lo_years.iloc[0]), float(psmcfa.window_hi_years.iloc[0]))
            style.shade_outside(ax, window, xlim)
            ax.fill_between(psmcfa.years, psmcfa.ne_lo, psmcfa.ne_hi, color=style.BLUE, alpha=style.BAND_ALPHA,
                            linewidth=0, zorder=2)
            ax.plot(psmcfa.years, psmcfa.ne_median, color=style.BLUE, linewidth=style.LW_DATA, zorder=4)
        if not vcf.empty:
            ax.plot(vcf.years, vcf.ne_median, color=style.BLUE_650, linewidth=style.LW_THIN, zorder=5)
        first_col, last_row = k % NCOLS == 0, k + NCOLS >= len(order)
        style.years_axis(ax, xlim, label=last_row)
        ax.set_xticks(inner)
        style.ne_axis(ax, ylim, label=style.NE_LABEL if first_col else None)
        if not first_col:
            ax.tick_params(labelleft=False)
        if not last_row:
            ax.tick_params(labelbottom=False)
        n = int(psmcfa.n.iloc[0]) if not psmcfa.empty else 0
        mark = ", admixed" if pop in cfg.admixed else ""
        style.note(ax, 0.96, 0.95, f"{pop} ({sup_of.get(pop, '?')}{mark})", ha="right", color=style.INK,
                   size=style.TEXT_PT, zorder=7)
        style.note(ax, 0.96, 0.95 - 0.11, f"n = {n}", ha="right", zorder=7)
    lo_q, hi_q = cfg.spread
    handles = [
        Line2D([], [], color=style.BLUE, linewidth=style.LW_DATA,
               label="Median of the population's genomes (psmcfa fits)"),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"{lo_q:.0%}-{hi_q:.0%} of its genomes"),
        Line2D([], [], color=style.BLUE_650, linewidth=style.LW_THIN,
               label=vcf_label({int(n) for n in pops[pops.route == "rs-vcf"].n})),
        style.outside_handle(),
    ]
    style.key_above(fig, handles, ncol=2, handlelength=2.2)
    style.panel_letters(fig, axes)
    return fig


__all__ = ["draw", "vcf_label"]
