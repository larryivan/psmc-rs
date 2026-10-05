"""Fig 5: input preparation (E4), drawn from E4's tables.

Panels:

a, b. Ne(t) of every replicate for the masks with the largest missing
      fraction, clustered (a) and dispersed (b), by route, with the truth;
      times outside the informative window (where c and d are measured) shaded.
c.    Mean log10(fitted / true Ne) over replicates along the informative
      window, for the same two masks: solid clustered, dashed dispersed.
d.    RMSE of log10 Ne against the truth, by mask and route (small marks:
      replicates; large marks: their mean).
e.    theta per bin relative to the exact route on the complete data of the
      same genome, in percent.
f.    Bins that differ from the exact route's, in percent.

The fq2psmcfa route is not drawn separately: its bins equal the exact
route's bin for bin (asserted by the ``bins`` jobs), so its fits do too; the
legend says so and ``E4_summary.csv`` counts the identical fits.

Run as ``python -m experiment.pipeline.figures.fig5 --tables DIR --out-dir DIR``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

NAMES = ("E4_fig5.pdf", "E4_fig5.png")
WIDTH_IN, HEIGHT_IN = 7.09, 4.9  # 180 mm, a double-column figure

# Identity of the routes: the first three slots of the validated reference
# palette (all-pairs CVD separation >= 9 in light mode) plus a distinct marker
# per route, so that no route is told apart by colour alone.
ROUTE_STYLE = {
    "exact": {"color": "#2a78d6", "marker": "o", "label": "exact: VCF + mask (= fq2psmcfa route)"},
    "multihetsep": {"color": "#eb6834", "marker": "s", "label": "multihetsep"},
    "nomask": {"color": "#1baf7a", "marker": "^", "label": "VCF without mask"},
}
DRAWN_ROUTES = tuple(ROUTE_STYLE)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
WINDOW_SHADE = "#f1f0ed"  # times outside the informative window (panels a, b)


def read_table(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x: str) -> Optional[float]:
    return float(x) if x not in ("", None, "nan") else None


def load(tables: Path) -> Dict[str, List[Dict[str, str]]]:
    return {k: read_table(tables / f"E4_{k}.csv") for k in ("summary", "routes", "bias", "bins", "curves", "truth")}


def mask_order(summary: Sequence[Dict[str, str]]) -> List[str]:
    """Masks in table order (the control first), each once."""
    seen: List[str] = []
    for row in summary:
        if row["mask"] not in seen:
            seen.append(row["mask"])
    return seen


def mask_info(summary: Sequence[Dict[str, str]]) -> Dict[str, Tuple[str, float]]:
    """``{mask: (structure, target missing fraction)}``."""
    return {r["mask"]: (r["structure"], float(r["missing_fraction_target"])) for r in summary}


def mask_label(structure: str, fraction: float) -> str:
    """The tick of a mask: its missing fraction (the structure is a group label below)."""
    return "none" if structure == "none" else f"{fraction:.0%}"


def group_labels(ax, masks: Sequence[str], info: Dict[str, Tuple[str, float]]) -> None:
    """``clustered`` / ``dispersed`` under the ticks of their masks."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(masks):
        if info[m][0] != "none":
            groups[info[m][0]].append(i)
    for structure, idx in groups.items():
        lo, hi = min(idx), max(idx)
        ax.plot([lo - 0.3, hi + 0.3], [-0.13, -0.13], transform=trans, color=INK_2, lw=0.6, clip_on=False)
        ax.text((lo + hi) / 2, -0.15, structure, transform=trans, ha="center", va="top", fontsize=6.5, color=INK)


def worst_masks(info: Dict[str, Tuple[str, float]]) -> Dict[str, Optional[str]]:
    """The mask with the largest missing fraction of each structure."""
    out: Dict[str, Optional[str]] = {}
    for structure in ("clustered", "dispersed"):
        cands = [(f, m) for m, (s, f) in info.items() if s == structure]
        out[structure] = max(cands)[1] if cands else None
    return out


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(INK_2)
    ax.tick_params(width=0.6, length=2.5, color=INK_2, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def informative_window(data) -> Optional[Tuple[float, float]]:
    """The times of the bias grid: the model's informative window, where the metrics are taken."""
    years = [num(r["years"]) for r in data["bias"]]
    return (min(years), max(years)) if years else None


def draw_curves(ax, data, mask: Optional[str], title: str) -> None:
    window = informative_window(data)
    if window is not None:
        for lo, hi in ((1.0, window[0]), (window[1], 1e9)):
            ax.axvspan(lo, hi, color=WINDOW_SHADE, lw=0, zorder=0)
    truth = data["truth"]
    ax.step([num(r["years"]) for r in truth], [num(r["ne"]) for r in truth], where="post", color=INK, lw=1.4,
            label="truth", zorder=5)
    if mask is not None:
        by: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        for r in data["curves"]:
            if r["mask"] == mask and r["route"] in DRAWN_ROUTES:
                by[(r["route"], r["replicate"])].append((num(r["years"]), num(r["ne"])))
        for (route, _), pts in sorted(by.items()):
            pts.sort()
            xs = [p[0] for p in pts if p[0] > 0]
            ys = [p[1] for p in pts if p[0] > 0]
            xs.append(xs[-1] * 1.5)
            ys.append(ys[-1])
            ax.step(xs, ys, where="post", color=ROUTE_STYLE[route]["color"], lw=0.9, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e3, 3e6)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    ax.set_xlabel("years ago (g = 25, μ = 2.5e-8)")
    ax.set_ylabel("Ne (diploid individuals)")
    style_axes(ax)


def draw_bias(ax, data, masks: Dict[str, Optional[str]]) -> None:
    dashes = {"clustered": "-", "dispersed": (0, (4, 2))}
    for structure, mask in masks.items():
        if mask is None:
            continue
        for route in DRAWN_ROUTES:
            rows = [r for r in data["bias"] if r["mask"] == mask and r["route"] == route]
            if not rows:
                continue
            ax.plot([num(r["years"]) for r in rows], [num(r["log10_bias_mean"]) for r in rows],
                    color=ROUTE_STYLE[route]["color"], lw=1.3, linestyle=dashes[structure])
    ax.axhline(0.0, color=INK_2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("years ago")
    ax.set_ylabel("log10(fitted / true Ne)")
    ax.set_title("mean bias, largest missing fractions", fontsize=7.5, color=INK, loc="left")
    style_axes(ax)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK_2, lw=1.2, linestyle=dashes[s], label=s) for s, m in masks.items() if m]
    ax.legend(handles=handles, frameon=False, fontsize=6.5, loc="best", handlelength=2.6)


def draw_by_mask(ax, data, masks: Sequence[str], info, value, ylabel: str, title: str, routes=DRAWN_ROUTES,
                 percent: bool = False) -> None:
    """Replicates (small) and their mean (large) of ``value(row)`` for every mask and route."""
    width = 0.22
    offsets = {route: (i - (len(routes) - 1) / 2) * width for i, route in enumerate(routes)}
    scale = 100.0 if percent else 1.0
    for i, mask in enumerate(masks):
        for route in routes:
            st = ROUTE_STYLE[route]
            ys = [value(r) for r in data["routes"] if r["mask"] == mask and r["route"] == route]
            ys = [y * scale for y in ys if y is not None]
            if not ys:
                continue
            x = i + offsets[route]
            ax.scatter([x] * len(ys), ys, s=7, color=st["color"], alpha=0.45, marker=st["marker"], linewidths=0,
                       zorder=3)
            ax.scatter([x], [sum(ys) / len(ys)], s=30, color=st["color"], marker=st["marker"],
                       edgecolors=SURFACE, linewidths=0.8, zorder=4)
    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels([mask_label(*info[m]) for m in masks], fontsize=6.5)
    ax.set_xlim(-0.6, len(masks) - 0.4)
    group_labels(ax, masks, info)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    style_axes(ax)
    ax.grid(False, axis="x")


def figure(data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    summary = data["summary"]
    masks = mask_order(summary)
    info = mask_info(summary)
    worst = worst_masks(info)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, HEIGHT_IN), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.12)
    for ax, structure in zip(axes[0][:2], ("clustered", "dispersed")):
        m = worst[structure]
        title = f"{structure} gaps, {info[m][1]:.0%} missing" if m else f"{structure}: no mask"
        draw_curves(ax, data, m, title)
    draw_bias(axes[0][2], data, worst)
    draw_by_mask(axes[1][0], data, masks, info, lambda r: num(r["rmse_log10_ne_vs_truth"]),
                 "RMSE of log10 Ne", "error against the truth")
    draw_by_mask(axes[1][1], data, masks, info, lambda r: num(r["theta_rel_complete"]),
                 "θ per bin vs complete data (%)", "θ against the complete genome", percent=True)
    axes[1][1].axhline(0.0, color=INK_2, lw=0.6, zorder=2)
    draw_by_mask(axes[1][2], data, masks, info, lambda r: num(r["differing_bin_fraction"]),
                 "bins differing (%)", "bins unlike the exact route's", routes=("multihetsep", "nomask"),
                 percent=True)
    for ax, letter in zip([a for row in axes for a in row], "abcdef"):
        panel_letter(ax, letter)
    handles = [Line2D([], [], color=INK, lw=1.4, label="truth")] + [
        Line2D([], [], color=st["color"], lw=1.2, marker=st["marker"], markersize=4, label=st["label"])
        for st in ROUTE_STYLE.values()
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=7)
    return fig


def save(fig, out_dir: Path, names: Sequence[str] = NAMES) -> List[Path]:
    """Write the figure; PDF and PNG metadata carry no date, so equal tables give equal files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        path = out_dir / name
        tmp = path.with_name(path.name + ".partial")
        fmt = path.suffix.lstrip(".")
        if fmt == "pdf":
            fig.savefig(tmp, format=fmt, metadata={"CreationDate": None, "Creator": None, "Producer": None})
        else:
            fig.savefig(tmp, format=fmt, dpi=300, metadata={"Software": None})
        tmp.replace(path)
        written.append(path)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.figures.fig5", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", type=Path, required=True, help="E4's tables directory")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)
    for path in save(figure(load(args.tables)), args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
