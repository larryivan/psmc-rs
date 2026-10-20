"""Fig 3 (E2): four real genomes fitted by C psmc and psmc-rs 0.2, with psmc-rs's bootstrap.

One panel per genome, in ``config.E2.panel_order``, on a common time axis:
C psmc's fit (orange, wider, underneath) and psmc-rs's (blue), the
pointwise 95% band of psmc-rs's bootstrap at the default block size (blue,
light) and, for a genome whose default blocks are few (fewer than
psmc-rs's 20), the band at 5 Mb blocks too (dark blue, dotted outline); no
genome of the production run has so few (NA12878: 59), only the smoke and
tiny profiles' prefixes. The bootstrap resamples blocks: the genome's
sequences split at the block size, 50 Mb (splitfa's trunks; a remainder
under 1.5 blocks stays in the last block), so a sequence shorter than that
is one whole block, and each panel gives its genome's blocks and sequences
(``config.RealGenome.sequence_noun`` names them). Whole sequences shorter
than 50 Mb hold 97% of *P. castaneus*'s bins, 50% of *E. subglobosa*'s,
23% of *H. tectifera*'s and only chr21's 1.6% of NA12878's (production,
``E2_bootstrap.bins_below_trunk_fraction``). E5 calibrates equal 50 Mb and
5 Mb blocks of 50 Mb chromosomes, the scheme closest to NA12878's; the
turtles' bands are not calibrated. The legend gives no block size (few
blocks are 50 Mb long), unless a 5 Mb band is drawn too. Each genome is
scaled with its own mu and g, given in the panel with their source in a
few words (``config.RealGenome.scaling_cite``;
in full in ``scaling_source``): the turtles with mu = 4.6e-9 per bp per generation,
the mean germline rate of nine *Chrysemys picta* (painted turtle) trios
(Bergeron et al. 2023; no pleurodire rate has been measured), and g = 15
years for all three species; the human with mu = 1.25e-8 (Scally & Durbin
2012) and g = 29 years (Fenner 2005). Ne and time scale as 1/mu, and time
also as g (the caption's sensitivity sentence: figures/README.md). Times
outside the window on which the tools are compared (1% to 99% of pairwise
coalescence times under C psmc's fitted model) are shaded grey; the panel
gives the tools' difference on it, interval by interval (the RMSE of
log10(Ne_rs,k / Ne_c,k) over C psmc's intervals in the window, weighted by
their log-time width, and the largest), and the bootstrap's replicates and
blocks. The Ne axis covers the fits and the bands inside the window, except
a coarse band (fewer blocks than psmc-rs's 20; none in the production
run), which is clipped where it reaches beyond (the panel then says so).
The notes have no box: room is kept for them above and below the data, and
a box would cover the grey of the times outside the window. The human
genome is NA12878's chr1-22, the psmcfa E7 fits for that sample, and the
panel names the sample (``config.RealGenome.label``).

Everything comes from E2's tables: ``E2_curves`` (the fits), ``E2_bands``
(the bands, on a dense grid), ``E2_bootstrap`` (replicates, blocks),
``E2_rs_vs_c`` (window, differences) and ``E2_fit_summary`` (scaling and
its source, sequences).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .. import config
from . import style
from .data import Source, curves, fmt_sig

C_LW, RS_LW = 2.0, 0.8
MARGIN_DECADES = 0.5  # the x range: the comparison windows widened by this much on each side
FINE_STYLE = style.DOT  # the fine band's outline (not the truth's dashes)
FINE_LW = 0.8
NOTE_LINE_SHARE = 0.072  # share of a panel's height one line of a note takes (7 pt, line spacing 1.25)
SCALING_NOTE_Y = 0.865  # the top of the note under the species name (axes fraction)
SCALING_NOTE_LINES = 3  # sample; mu and g; their source


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


def trunk_text(row, sequences: Optional[int] = None, noun: str = "scaffolds") -> str:
    """``59 blocks: 22 chromosomes split at 50 Mb``, or ``4 blocks of 0.5 Mb`` of one sequence (and a warning when few).

    psmc-rs's bootstrap, as splitfa, resamples blocks of the sequences
    split at the block size; a sequence shorter than that is one block.
    ``noun`` says what the sequences are (``config.RealGenome.sequence_noun``).
    """
    mb = float(row.trunk_mb)
    blocks = int(row.trunks)
    if sequences == 1:
        text = f"{blocks:,} blocks of {mb:g} Mb"
    else:
        what = "sequences" if sequences is None else f"{sequences:,} {noun}"
        text = f"{blocks:,} blocks: {what} split at {mb:g} Mb"
    return text + " (few: coarse)" if str(row.coarse_band) == "True" else text


def _log_ticks_1_2_5(ax) -> None:
    """Label 1, 2 and 5 times each power of ten on a y axis spanning less than two decades."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    def label(y, _pos):
        exp = int(math.floor(math.log10(y) + 1e-9))
        mant = round(y / 10**exp)
        return rf"$10^{{{exp}}}$" if mant == 1 else rf"${mant}\times10^{{{exp}}}$"

    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(label))


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e2
    fits = curves(src.table("E2_curves"), ["sample_id", "tool"])
    bands = src.table("E2_bands")
    boots = src.table("E2_bootstrap")
    pairs = {r.sample_id: r for r in src.table("E2_rs_vs_c").itertuples()}
    summary = {r.sample_id: r for r in src.table("E2_fit_summary").itertuples() if r.tool == "c"}
    genomes = genome_order(cfg, {s for s, _ in fits})
    windows = {s: (float(p.window_lo_years), float(p.window_hi_years)) for s, p in pairs.items()}
    shown = [windows[g.sample_id] for g in genomes]
    xlim = (min(w[0] for w in shown) / 10**MARGIN_DECADES, max(w[1] for w in shown) * 10**MARGIN_DECADES)

    n = len(genomes)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig = style.figure("double", 12.0 + 50.0 * nrow)
    gs = style.grid(fig, nrow, ncol, left_mm=13.0, right_mm=3.0, top_mm=8.5, bottom_mm=8.5, hspace_mm=9.0,
                    wspace_mm=10.0)
    axes = []
    panels: List = []  # (index, axes, genome, Ne values to show, bottom note lines, coarse band top, mu, g, cite)
    for i, gnm in enumerate(genomes):
        sid = gnm.sample_id
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        axes.append(ax)
        pair = pairs[sid]
        window = windows[sid]
        style.shade_outside(ax, window, xlim)
        mine = boots[boots.sample_id == sid].sort_values("trunk_bins", ascending=False)
        vals: List[float] = []
        coarse_hi = 0.0
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
                    ax.plot(x[keep], edge[keep], drawstyle="steps-post", color=style.BLUE_650, linewidth=FINE_LW,
                            linestyle=FINE_STYLE, zorder=3)
            inside = (x >= window[0]) & (x <= window[1])
            if str(row.coarse_band) == "True":  # too few blocks to be a band: clipped, not fitted to the axis
                coarse_hi = max(coarse_hi, float(hi[inside].max()))
                continue
            vals += [float(lo[inside].min()), float(hi[inside].max())]
        for tool, ent, lw, z in (("c", style.C_PSMC, C_LW, 4), ("rs", style.RS02, RS_LW, 5)):
            c = fits.get((sid, tool))
            if c is None:
                continue
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linewidth=lw, zorder=z)
            grid = np.logspace(math.log10(window[0]), math.log10(window[1]), 200)
            vals += [float(c.at(grid).min()), float(c.at(grid).max())]
        sc = summary.get(sid)
        mu = float(sc.mu) if sc is not None else gnm.scaling.mu
        g = float(sc.gen_years) if sc is not None else gnm.scaling.gen_years
        cite = str(sc.scaling_cite) if sc is not None else gnm.scaling_cite
        sequences = int(sc.sequences) if sc is not None else None
        lines = [r"psmc-rs vs C, log$_{10}$ $N_\mathrm{e}$ per interval:",
                 f"RMSE {fmt_sig(float(pair.rmse_log10_ne_rs_vs_c_per_interval))}, "
                 f"max {fmt_sig(float(pair.max_abs_log10_ne_rs_vs_c_per_interval))}"]
        for row in mine.itertuples():
            lines.append(f"{int(row.replicates)} replicates, {trunk_text(row, sequences, gnm.sequence_noun)}")
        panels.append((i, ax, gnm, vals, lines, coarse_hi, mu, g, cite))
    # One Ne range per scaling (the turtles share theirs) with room for the notes, which are not to cover the
    # curves or the bands (they have no box, which would cover the grey of the times outside the window): the top
    # notes above the data, the bottom ones below it. Every panel then spans the same number of decades, widened
    # evenly about its data, so that a ten-fold change is drawn at the same height in all of them.
    room_top = 1.0 - SCALING_NOTE_Y + SCALING_NOTE_LINES * NOTE_LINE_SHARE
    ranges: Dict = {}
    for key in {(p[6], p[7]) for p in panels}:
        group = [p for p in panels if (p[6], p[7]) == key]
        lo = min(v for p in group for v in p[3]) / 1.6
        hi = max(v for p in group for v in p[3]) * 1.6
        room_bottom = 0.03 + NOTE_LINE_SHARE * max(len(p[4]) for p in group)
        span = math.log10(hi / lo) / (1.0 - room_top - room_bottom)
        ranges[key] = (lo / 10 ** (room_bottom * span), hi * 10 ** (room_top * span))
    decades = max(math.log10(hi / lo) for lo, hi in ranges.values())
    for key, (lo, hi) in ranges.items():
        extra = (decades - math.log10(hi / lo)) / 2.0
        ranges[key] = (lo / 10**extra, hi * 10**extra)
    for i, ax, gnm, vals, lines, coarse_hi, mu, g, cite in panels:
        lo, hi = ranges[(mu, g)]
        if coarse_hi > hi:
            lines = [t.replace("(few: coarse)", "(few: coarse, clipped)") for t in lines]
        style.years_axis(ax, xlim, label=i // ncol == nrow - 1)
        style.ne_axis(ax, (lo, hi), label=style.NE_LABEL if i % ncol == 0 else None)
        if math.log10(hi / lo) < 2.0:
            _log_ticks_1_2_5(ax)
        style.note(ax, 0.03, 0.97, gnm.species, color=style.INK, size=style.TEXT_PT, fontstyle="italic", zorder=7)
        style.note(ax, 0.03, SCALING_NOTE_Y, f"{where(gnm)}\n" + rf"$\mu$ = {sci(mu)}, g = {g:g} y" + "\n"
                   + rf"$\mu$: {cite}", linespacing=1.25, zorder=7)
        style.note(ax, 0.97, 0.03, "\n".join(lines), ha="right", va="bottom", linespacing=1.25, zorder=7)
    for j in range(n, nrow * ncol):
        fig.add_subplot(gs[j // ncol, j % ncol]).set_axis_off()

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # The block size names a band only beside the other band: a genome's blocks are
    # its sequences split at that size, so few of them are that long (the panels say
    # how many blocks of how many sequences).
    default_mb = cfg.bootstrap_trunk_bins * cfg.inference.bin_size / 1e6
    fine = (boots["default_trunks"].astype(str) == "False").any()
    handles = [
        style.handle(style.C_PSMC, linewidth=C_LW),
        style.handle(style.RS02, linewidth=RS_LW),
        Patch(facecolor=style.BLUE, alpha=style.BAND_ALPHA, linewidth=0,
              label=f"Bootstrap 95% band, {default_mb:g} Mb blocks" if fine else "Bootstrap 95% band"),
    ]
    if fine:
        fine_mb = float(boots[boots["default_trunks"].astype(str) == "False"]["trunk_mb"].iloc[0])
        handles.append(Line2D([], [], color=style.BLUE_650, linewidth=FINE_LW, linestyle=FINE_STYLE,
                              label=f"Bootstrap 95% band, {fine_mb:g} Mb blocks"))
    handles.append(Patch(facecolor=style.FILL, linewidth=0, label="Outside the comparison window"))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                 columnspacing=1.0, handlelength=2.0)
    style.panel_letters(fig, axes)
    return fig


def window_of(src: Source, sample_id: str) -> Tuple[float, float]:
    """The comparison window of a genome (years), from ``E2_rs_vs_c``."""
    r = next(r for r in src.table("E2_rs_vs_c").itertuples() if r.sample_id == sample_id)
    return float(r.window_lo_years), float(r.window_hi_years)


__all__ = ["draw", "genome_order", "trunk_text", "where", "window_of"]
