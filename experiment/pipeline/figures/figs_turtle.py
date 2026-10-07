"""Fig S (E2, turtle patterns): the split-first-window check of the three turtles.

Hilgers et al. 2025 (Curr Biol 35:927) found false recent peaks of Ne in
PSMC curves across many studies, in all their turtles among them, under the
pattern 4+25*2+4+6 that Fig 3 uses: its first free lambda spans the first
four of the 64 intervals, and splitting that window (2+2+25*2+4+6 or
1+1+1+1+25*2+4+6) removed the peaks. One panel per turtle, in
``config.E2.panel_order``: psmc-rs's fit of Fig 3 (4+25*2+4+6, blue, wider,
underneath) and its fits of the same input with the first window split in
two (dark blue, dashed) and in four (dark blue, dotted), every other setting
the same. The curves are scaled as in Fig 3 (mu = 4.6e-9 per bp per
generation, g = 15 years; ``config._TURTLE_SCALING_SOURCE``). Times outside
the comparison window of Fig 3 (1% to 99% of pairwise coalescence times
under C psmc's fit) are shaded grey. The note gives, for each pattern, the
largest Ne of the recent intervals (the first 8: the default's first window
and the two after it) relative to the plateau (the median Ne of the next 8
intervals), and "peak" where the largest lies after the first interval and
is at least 1.25 times both the first interval's Ne and the plateau
(``config.E2.split_recent_intervals``, ``split_peak_ratio``).

Everything comes from E2's tables: ``E2_split_curves`` (the fits) and
``E2_split_patterns`` (scaling, window, recent intervals).
"""

from __future__ import annotations

import math
from typing import Dict, List

from .. import config
from . import style
from .data import Source, curves
from .fig3 import MARGIN_DECADES, NOTE_LINE_SHARE, sci, where

DEFAULT_LW, SPLIT_LW = 1.6, 0.8
# The default pattern is psmc-rs's (Fig 3); a split pattern is a psmc-rs variant
# of the default: blue ramp 650, dashed (first window in two) or dotted (in four).
SPLIT_STYLES = ((0, (3.0, 1.5)), style.DOT)
TOP_NOTE_LINES = 2  # species; sample, mu and g


def pattern_entities(cfg: config.E2) -> Dict[str, style.Entity]:
    """How each pattern is drawn, by its name in the tables: ``default`` first, then the split patterns."""
    out = {"default": style.Entity(f"psmc-rs 0.2, {cfg.inference.pattern} (Fig 3)", style.BLUE, "-", "o")}
    for (name, pattern), ls in zip(cfg.split_patterns, SPLIT_STYLES * len(cfg.split_patterns)):
        out[name] = style.Entity(pattern, style.BLUE_650, ls, "o", filled=False)
    return out


def note_lines(rows) -> List[str]:
    """``Recent max / plateau:`` and one line per pattern: its ratio, and whether it is a peak."""
    lines = ["Recent max / plateau:"]
    for r in rows:
        peak = " (peak)" if str(r.recent_peak) == "True" else ""
        lines.append(f"{r.pattern}  {float(r.recent_max_vs_plateau):.2f}{peak}")
    return lines


def draw(src: Source):
    import numpy as np

    cfg = src.profile.e2
    fits = curves(src.table("E2_split_curves"), ["sample_id", "pattern_name"])
    table = src.table("E2_split_patterns")
    ents = pattern_entities(cfg)
    order = list(ents)
    by_id = {g.sample_id: g for g in cfg.genomes}
    present = set(table["sample_id"])
    ids = [s for s in cfg.panel_order if s in present] + sorted(s for s in present if s not in cfg.panel_order)
    genomes = [by_id[s] for s in ids if s in by_id]
    rows = {s: sorted(table[table.sample_id == s].itertuples(),
                      key=lambda r: order.index(r.pattern_name) if r.pattern_name in order else len(order))
            for s in ids}
    windows = {s: (float(rows[s][0].window_lo_years), float(rows[s][0].window_hi_years)) for s in ids}
    shown = [windows[g.sample_id] for g in genomes]
    xlim = (min(w[0] for w in shown) / 10**MARGIN_DECADES, max(w[1] for w in shown) * 10**MARGIN_DECADES)

    n = max(len(genomes), 1)
    fig = style.figure("double", 72.0)
    gs = style.grid(fig, 1, n, left_mm=13.0, right_mm=3.0, top_mm=8.5, bottom_mm=8.5, wspace_mm=8.0)
    axes = []
    for i, gnm in enumerate(genomes):
        sid = gnm.sample_id
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        window = windows[sid]
        style.shade_outside(ax, window, xlim)
        grid = np.logspace(math.log10(window[0]), math.log10(window[1]), 200)
        vals: List[float] = []
        for name in order:
            c = fits.get((sid, name))
            if c is None:
                continue
            ent = ents[name]
            default = name == "default"
            x, y = style.step_xy(c.years, c.ne, xlim)
            ax.plot(x, y, drawstyle="steps-post", color=ent.color, linestyle=ent.linestyle,
                    linewidth=DEFAULT_LW if default else SPLIT_LW, zorder=3 if default else 4)
            vals += [float(c.at(grid).min()), float(c.at(grid).max())]
        lines = note_lines(rows[sid])
        # Room for the notes, above and below the data (no box: it would cover the grey outside the window).
        lo, hi = min(vals) / 1.4, max(vals) * 1.4
        room_top = 0.03 + (TOP_NOTE_LINES + 0.6) * NOTE_LINE_SHARE
        room_bottom = 0.03 + NOTE_LINE_SHARE * len(lines)
        span = math.log10(hi / lo) / (1.0 - room_top - room_bottom)
        lo, hi = lo / 10 ** (room_bottom * span), hi * 10 ** (room_top * span)
        style.years_axis(ax, xlim)
        style.ne_axis(ax, (lo, hi), label=style.NE_LABEL if i == 0 else None)
        first = rows[sid][0]
        style.note(ax, 0.03, 0.97, gnm.species, color=style.INK, size=style.TEXT_PT, fontstyle="italic", zorder=7)
        style.note(ax, 0.03, 0.97 - NOTE_LINE_SHARE,
                   f"{where(gnm)}, " + rf"$\mu$ = {sci(float(first.mu))}, g = {float(first.gen_years):g} y",
                   zorder=7)
        style.note(ax, 0.97, 0.03, "\n".join(lines), ha="right", va="bottom", linespacing=1.25, zorder=7)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=e.color, linestyle=e.linestyle,
                      linewidth=DEFAULT_LW if k == "default" else SPLIT_LW, label=e.label)
               for k, e in ents.items()]
    handles.append(Patch(facecolor=style.NEUTRAL, linewidth=0, label="Outside the comparison window"))
    style.legend(fig, handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                 columnspacing=1.0, handlelength=2.6)
    style.panel_letters(fig, axes)
    return fig


__all__ = ["draw", "note_lines", "pattern_entities"]
