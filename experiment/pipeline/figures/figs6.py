"""Fig S6 (E7): the 1000 Genomes inputs and checks.

a: heterozygous SNVs per callable kb against raw sequencing depth, every
genome of the cohort, by superpopulation (marker). b: the two routes on the
default set: theta of the VCF fit (callable bases per bin,
``--bin-callable auto``) relative to the psmcfa fit, against the share of
called bins with some bases missing. c: the split-first-window check
(Hilgers et al. 2025): the first window's Ne over the next window's, under
the default pattern and with the first window split
(``config.E7.split_pattern``); points on the diagonal did not move. d:
NA12878's heterozygous SNVs against GIAB HG001 v4.2.1 inside the strict mask
and GIAB's confident regions. e: wall time per fit (median) of every route,
one thread; C psmc on the default set.

From ``E7_inputs``, ``E7_route_shift``, ``E7_split``, ``E7_giab`` and
``E7_throughput``.
"""

from __future__ import annotations

from . import style

SUP_MARKERS = ("o", "s", "^", "D", "v")  # by config.E7.superpopulations
ROUTE_LABELS = {"rs-psmcfa": "psmc-rs, psmcfa", "rs-vcf": "psmc-rs, VCF + mask", "rs-split": "psmc-rs, split window",
                "c": "C psmc, psmcfa", "boot-rs": "psmc-rs bootstrap (whole run)"}


def _by_sup(ax, df, x: str, y: str, sups, size: float = 3.0, alpha: float = 0.6) -> None:
    for sup, m in zip(sups, SUP_MARKERS):
        d = df[df.superpopulation == sup]
        if not d.empty:
            ax.scatter(d[x], d[y], s=size, marker=m, facecolor="none", edgecolor=style.INK2, linewidths=0.4,
                       alpha=alpha, label=sup, zorder=3)


def draw(src):
    cfg = src.profile.e7
    sups = list(cfg.superpopulations)
    inputs = src.table("E7_inputs")
    shift = src.table("E7_route_shift")
    split = src.table("E7_split")
    giab = src.table("E7_giab")
    thr = src.table("E7_throughput")

    fig = style.figure("double", 110.0)
    gs = style.grid(fig, 2, 3, left_mm=13.0, right_mm=3.0, top_mm=8.0, bottom_mm=9.0, wspace_mm=12.0, hspace_mm=13.0)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    a, b, c, d, e, f = axes

    _by_sup(a, inputs, "raw_depth", "het_per_callable_kb", sups)
    a.set_xlabel("Raw sequencing depth")
    a.set_ylabel("Heterozygous SNVs per callable kb")

    _by_sup(b, shift, "partly_callable_bin_fraction", "theta_rel_vcf_vs_psmcfa", sups, size=8.0, alpha=0.9)
    style.ref_line(b, y=0.0)
    b.set_xlabel("Called bins with bases missing")
    b.set_ylabel(r"$\theta$, VCF fit vs psmcfa fit (relative)")

    _by_sup(c, split, "log10_first_vs_next_default", "log10_first_vs_next_split", sups, size=8.0, alpha=0.9)
    if not split.empty:
        lo = float(min(split.log10_first_vs_next_default.min(), split.log10_first_vs_next_split.min()))
        hi = float(max(split.log10_first_vs_next_default.max(), split.log10_first_vs_next_split.max()))
        pad = 0.05 * max(hi - lo, 0.1)
        c.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=style.MUTED, linewidth=style.LW_REF, zorder=2)
    c.set_xlabel(r"log$_{10}$ first / next window $N_\mathrm{e}$, default")
    c.set_ylabel(r"log$_{10}$ first / next window $N_\mathrm{e}$, split")

    d.axis("off")
    lines = ["GIAB HG001 v4.2.1 (NA12878)"]
    for r in giab.itertuples():
        mb = int(r.eval_bases) / 1e6
        lines += [f"{r.sample_id}: {int(r.truth_het_snvs):,} GIAB het SNVs in {mb:,.0f} Mb" if mb >= 10 else
                  f"{r.sample_id}: {int(r.truth_het_snvs):,} GIAB het SNVs in {mb:.2g} Mb",
                  f"precision {float(r.precision):.4f}", f"recall {float(r.recall):.4f}"]
    style.note(d, 0.0, 0.95, "\n".join(lines) if len(lines) > 1 else "no GIAB comparison", color=style.INK,
               linespacing=1.4)

    t = thr.sort_values("wall_median_s")
    labels = [ROUTE_LABELS.get(r, r) for r in t.route]
    colors = [style.ORANGE if tool == "c" else style.BLUE for tool in t.tool]
    e.barh(range(len(t)), t.wall_median_s, color=colors, height=0.6, zorder=3)
    e.set_yticks(range(len(t)))
    e.set_yticklabels(labels)
    style.log_axis(e, "x", grid=True)
    e.set_xlabel("Wall time per fit, median (s)")
    for i, r in enumerate(t.itertuples()):
        e.text(float(r.wall_median_s) * 1.15, i, f"n = {int(r.jobs):,}", va="center", fontsize=style.TICK_PT,
               color=style.INK2)

    f.axis("off")
    handles, labels = a.get_legend_handles_labels()
    if handles:
        style.legend(f, handles, labels, loc="center left", title="Superpopulation", markerscale=2.0)
    style.panel_letters(fig, axes[:5], align_cols=True)
    return fig


__all__ = ["draw"]
