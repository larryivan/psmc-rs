"""Figures S1-S4 of the supplement, drawn from E6's tables (and nothing else).

Each figure reads only the CSV tables its part writes (:mod:`..experiments.e6`),
so it can be redrawn from the committed tables without any run. Every figure
is saved as PDF (vector, fonts embedded as TrueType) and PNG (300 dpi), at the
two-column width of a journal page (180 mm).

Colours follow one fixed categorical order, assigned by entity and never by
rank: psmc-rs blue, C psmc orange, the third series (multihetsep) aqua; the
truth is drawn in ink. Line style repeats the identity (psmc-rs solid, C psmc
dashed, multihetsep dotted) so that no figure depends on colour alone.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter  # noqa: E402

# Ink, chrome and the categorical slots (validated for colour-vision
# deficiencies: the first three slots pass all-pairs; four lines pass adjacent).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
BLUE, ORANGE, AQUA = SLOTS[0], SLOTS[1], SLOTS[2]
TOOL_STYLE = {
    "rs": {"color": BLUE, "ls": "-", "label": "psmc-rs 0.2"},
    "c": {"color": ORANGE, "ls": (0, (4, 2)), "label": "C psmc"},
}
FORMAT_STYLE = {
    "psmcfa": {"color": BLUE, "ls": "-", "label": "psmcfa"},
    "vcf": {"color": ORANGE, "ls": (0, (4, 2)), "label": "VCF"},
    "multihetsep": {"color": AQUA, "ls": (0, (1, 1.5)), "label": "multihetsep"},
}
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
WIDTH_IN = 180 / 25.4  # two-column width
MODEL_TITLES = {"constant": "Constant", "bottleneck": "Bottleneck", "expansion": "Expansion", "zigzag": "Zigzag"}


def style() -> None:
    """Quiet, print-ready defaults: sans text in ink, hairline axes, no box clutter."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "axes.titleweight": "regular",
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def panel_label(ax, letter: str, y: float = 1.04) -> None:
    """The panel's letter, above its top left corner (``y`` in axes fractions)."""
    ax.text(-0.02, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="right", va="bottom")


def ygrid(ax) -> None:
    ax.yaxis.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def log_ticks(axis) -> None:
    """Label only the decades of a log axis (minor ticks stay, unlabelled)."""
    axis.set_major_locator(LogLocator(base=10, numticks=12))
    axis.set_minor_formatter(NullFormatter())


def at_least_a_decade(ax) -> None:
    """Widen a log y-axis to span at least one decade around its data, so that it carries a labelled tick."""
    lo, hi = ax.get_ylim()
    if hi / lo < 10:
        mid = math.sqrt(lo * hi)
        ax.set_ylim(mid / 10**0.55, mid * 10**0.55)


def top_legend(fig, handles, ncol: int) -> None:
    """One legend for the whole figure, centred above the panels."""
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, handlelength=2.6,
               columnspacing=1.6)


def save(fig, outputs: Sequence[Path]) -> Dict[str, Path]:
    """Save to every output (format from the extension) under a temporary name, then rename."""
    written = {}
    for out in outputs:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".partial")
        ext = out.suffix.lstrip(".")
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(tmp, format=ext, bbox_inches="tight", pad_inches=0.02, metadata=meta)
        os.replace(tmp, out)
        written[f"figure/{out.name}"] = out
    plt.close(fig)
    return written


def model_order(models: Sequence[str]) -> List[str]:
    known = [m for m in MODEL_TITLES if m in set(models)]
    return known + [m for m in dict.fromkeys(models) if m not in MODEL_TITLES]


def years_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlabel("Years ago")


# ---------------------------------------------------------------------------
# Fig S1: EM convergence
# ---------------------------------------------------------------------------


def fig_s1(tables: Mapping[str, Path], outputs: Sequence[Path]) -> Dict[str, Path]:
    """Log-likelihood gap to the last common iteration (a) and the tools' relative difference (b), per model."""
    tr = pd.read_csv(tables["traces"])
    models = model_order(tr["model"].unique())
    n_iter = int(tr["iteration"].max())
    fig, axes = plt.subplots(2, len(models), figsize=(WIDTH_IN, 3.9), sharex=True, squeeze=False)
    for j, model in enumerate(models):
        m = tr[tr["model"] == model]
        ax = axes[0, j]
        for tool in ("rs", "c"):
            st = TOOL_STYLE[tool]
            for _, rep_rows in m.groupby("replicate"):
                g = rep_rows[(rep_rows["iteration"] < n_iter - 1) & rep_rows[f"gap_{tool}"].notna()]
                ax.plot(g["iteration"], g[f"gap_{tool}"].clip(lower=1e-6), color=st["color"], ls=st["ls"], lw=0.8, alpha=0.75)
        ax.set_yscale("log")
        log_ticks(ax.yaxis)
        ax.set_title(MODEL_TITLES.get(model, model))
        ygrid(ax)
        ax = axes[1, j]
        for _, rep_rows in m.groupby("replicate"):
            d = rep_rows[rep_rows["loglike_rel_diff"].notna()]
            vals = d["loglike_rel_diff"].abs()
            ax.plot(d["iteration"][vals > 0], vals[vals > 0], color=INK_2, lw=0.6, alpha=0.55, marker="o", ms=1.6)
        ax.set_yscale("log")
        log_ticks(ax.yaxis)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("EM iteration")
        ygrid(ax)
    axes[0, 0].set_ylabel(f"Log-likelihood below\niteration {n_iter - 1}")
    axes[1, 0].set_ylabel("|psmc-rs − C psmc|\n/ |C psmc| log-likelihood")
    handles = [plt.Line2D([], [], color=TOOL_STYLE[t]["color"], ls=TOOL_STYLE[t]["ls"], lw=1.2, label=TOOL_STYLE[t]["label"])
               for t in ("rs", "c")]
    handles.append(plt.Line2D([], [], color=INK_2, lw=0.6, marker="o", ms=1.6, label="one replicate (b)"))
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    top_legend(fig, handles, 3)
    return save(fig, outputs)


# ---------------------------------------------------------------------------
# Fig S2: format consistency
# ---------------------------------------------------------------------------


def _step(ax, years, ne, **kw):
    x = np.append(np.asarray(years, dtype=float), np.asarray(years, dtype=float)[-1] * 4)
    y = np.append(np.asarray(ne, dtype=float), np.asarray(ne, dtype=float)[-1])
    ax.step(x, y, where="post", **kw)


def _curve_at(years, ne, t):
    idx = np.clip(np.searchsorted(np.asarray(years), t, side="right") - 1, 0, len(ne) - 1)
    return np.asarray(ne)[idx]


def fig_s2(tables: Mapping[str, Path], outputs: Sequence[Path]) -> Dict[str, Path]:
    """Ne(t) fitted from each format (a) and its log10 difference from the psmcfa fit (b), per model."""
    curves = pd.read_csv(tables["curves"])
    bins = pd.read_csv(tables["bins"])
    fits = pd.read_csv(tables["fits"])
    models = model_order(curves["model"].unique())
    fig, axes = plt.subplots(2, len(models), figsize=(WIDTH_IN, 3.9), squeeze=False,
                             gridspec_kw={"height_ratios": [1.6, 1.0]})
    for j, model in enumerate(models):
        c = curves[curves["model"] == model]
        f = fits[(fits["model"] == model)].set_index("format")
        lo, hi = float(f["window_lo_years"].iloc[0]), float(f["window_hi_years"].iloc[0])
        ax = axes[0, j]
        t = c[c["format"] == "truth"]
        ax.plot(t["years"], t["ne"], color=INK, lw=1.6, alpha=0.35, label="truth", zorder=1)
        for fmt in ("psmcfa", "vcf", "multihetsep"):
            s = c[c["format"] == fmt]
            st = FORMAT_STYLE[fmt]
            _step(ax, s["years"], s["ne"], color=st["color"], ls=st["ls"], lw=1.1, label=st["label"], zorder=2)
        ax.axvspan(lo, hi, color=GRID, alpha=0.35, lw=0, zorder=0)
        years_axis(ax)
        ax.set_yscale("log")
        log_ticks(ax.yaxis)
        at_least_a_decade(ax)
        ax.set_xlim(1e3, 1e7)
        ax.set_xlabel("")
        ax.set_title(MODEL_TITLES.get(model, model))
        ygrid(ax)
        ax = axes[1, j]
        grid = np.logspace(math.log10(lo), math.log10(hi), 400)
        ref = c[c["format"] == "psmcfa"]
        base = np.log10(_curve_at(ref["years"].to_numpy(), ref["ne"].to_numpy(), grid))
        for fmt in ("vcf", "multihetsep"):
            s = c[c["format"] == fmt]
            st = FORMAT_STYLE[fmt]
            ax.plot(grid, np.log10(_curve_at(s["years"].to_numpy(), s["ne"].to_numpy(), grid)) - base,
                    color=st["color"], ls=st["ls"], lw=1.1)
        ax.axhline(0.0, color=AXIS, lw=0.6, zorder=0)
        years_axis(ax)
        ax.set_xlim(1e3, 1e7)
        tot = bins[(bins["model"] == model) & (bins["sequence"] == "all")].set_index("format")
        vcf_diff = int(tot.loc["vcf", "bins_differing"]) if "vcf" in tot.index else -1
        tail = int(tot.loc["multihetsep", "bins_missing_at_end"]) if "multihetsep" in tot.index else -1
        rmse = float(f.loc["multihetsep", "rmse_log10_ne_vs_psmcfa"]) if "multihetsep" in f.index else float("nan")
        ax.set_title(f"VCF: {vcf_diff:,} bins differ\nmultihetsep: {tail:,} tail bins missing\n"
                     f"RMSE of log10 Ne {rmse:.1e}", loc="left", fontsize=6, color=INK_2)
        span = max(np.nanmax(np.abs(ax.get_ylim())), 1e-4)
        ax.set_ylim(-span, span)
        ygrid(ax)
    axes[0, 0].set_ylabel("Ne")
    axes[1, 0].set_ylabel("log10 Ne − log10 Ne(psmcfa)")
    handles = [plt.Line2D([], [], color=INK, lw=1.6, alpha=0.35, label="truth")]
    handles += [plt.Line2D([], [], color=st["color"], ls=st["ls"], lw=1.1, label=st["label"]) for st in FORMAT_STYLE.values()]
    handles.append(Patch(color=GRID, alpha=0.6, lw=0, label="informative window"))
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b", y=1.36)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    top_legend(fig, handles, 5)
    return save(fig, outputs)


# ---------------------------------------------------------------------------
# Fig S3: TMRCA decoding
# ---------------------------------------------------------------------------


def fig_s3(tables: Mapping[str, Path], outputs: Sequence[Path]) -> Dict[str, Path]:
    """An example track (a), the density of posterior mean against truth (b) and coverage by true TMRCA (c)."""
    track = pd.read_csv(tables["track"])
    dens = pd.read_csv(tables["density"])
    cal = pd.read_csv(tables["calibration"])
    summ = pd.read_csv(tables["summary"]).set_index("bins")
    fig = plt.figure(figsize=(WIDTH_IN, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.0, 1.15], hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    x = track["start_bp"].to_numpy() / 1e6
    step = (x[1] - x[0]) if len(x) > 1 else 1e-4
    xe = np.append(x, x[-1] + step)
    ax.fill_between(xe, np.append(track["q025_years"], track["q025_years"].iloc[-1]),
                    np.append(track["q975_years"], track["q975_years"].iloc[-1]),
                    step="post", color=BLUE, alpha=0.16, lw=0, label="95% posterior interval")
    ax.step(xe, np.append(track["posterior_mean_years"], track["posterior_mean_years"].iloc[-1]), where="post",
            color=BLUE, lw=0.9, label="posterior mean")
    ax.step(xe, np.append(track["true_years"], track["true_years"].iloc[-1]), where="post", color=INK, lw=0.9,
            label="true TMRCA")
    het = track[track["obs"] == "K"]
    ymin = float(min(track["q025_years"].min(), track["true_years"].min())) / 2.5
    ax.plot(het["start_bp"] / 1e6, np.full(len(het), ymin * 1.25), ls="none", marker="|", ms=3, mew=0.4, color=MUTED,
            label="heterozygous bin")
    ax.set_yscale("log")
    log_ticks(ax.yaxis)
    ax.set_ylim(bottom=ymin)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_xlabel(f"Position on {track['sequence'].iloc[0]} (Mb)")
    ax.set_ylabel("TMRCA (years)")
    ygrid(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, handlelength=2.0)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    edges_x = np.unique(np.concatenate([dens["log10_true_lo"], dens["log10_true_hi"]]))
    edges_y = np.unique(np.concatenate([dens["log10_mean_lo"], dens["log10_mean_hi"]]))
    grid = np.full((len(edges_y) - 1, len(edges_x) - 1), np.nan)
    ix = np.searchsorted(edges_x, dens["log10_true_lo"].to_numpy())
    iy = np.searchsorted(edges_y, dens["log10_mean_lo"].to_numpy())
    grid[iy, ix] = dens["n_bins"].to_numpy()
    mesh = ax.pcolormesh(10**edges_x, 10**edges_y, grid, cmap=cmap, norm=LogNorm(vmin=1, vmax=np.nanmax(grid)),
                         rasterized=True)
    lim = (10 ** edges_x[0], 10 ** edges_x[-1])
    ax.plot(lim, lim, color=INK_2, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_ticks(ax.xaxis)
    log_ticks(ax.yaxis)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True TMRCA (years)")
    ax.set_ylabel("Posterior mean TMRCA (years)")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    cb.ax.set_title("Bins", fontsize=6.5, color=INK_2, loc="left")
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(colors=AXIS, labelcolor=INK_2, labelsize=6)
    a = summ.loc["all"]
    ax.text(0.03, 0.97,
            f"Pearson r (log10) {a['pearson_log10_mean']:.3f}\nSpearman ρ {a['spearman_mean']:.3f}\n"
            f"{int(a['n_bins']):,} bins",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK_2)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    t = cal[cal["by"] == "true"].sort_values("class")
    k = t["class"].to_numpy()
    ax.axhline(0.95, color=AXIS, lw=0.8, zorder=0)
    ax.text(k[0] - 0.4, 0.951, "nominal 0.95", fontsize=6, color=MUTED, va="bottom", ha="left")
    ax.plot(k, t["coverage95_states"], color=BLUE, lw=1.1, marker="o", ms=4, mec="white", mew=0.8,
            label="state intervals")
    ax.plot(k, t["coverage95_point"], color=ORANGE, lw=1.1, ls=(0, (4, 2)), marker="s", ms=3.6,
            mec="white", mew=0.8, label="state average times")
    ax.set_xticks(k)
    ax.set_xticklabels([f"{y / 1e3:.0f}" for y in t["median_true_years"]], fontsize=6)
    ax.set_xlim(k[0] - 0.5, k[-1] + 0.5)
    low = float(min(t["coverage95_point"].min(), t["coverage95_states"].min()))
    ax.set_ylim(min(0.8, math.floor((low - 0.02) * 20) / 20), 1.0)
    ax.set_xlabel("Decile of the true TMRCA (median, thousand years)")
    ax.set_ylabel("Coverage of the 95% interval")
    ax.set_title(f"overall {a['coverage95_states']:.3f} (state intervals), {a['coverage95_point']:.3f} (average times)",
                 loc="left", fontsize=6, color=INK_2)
    ygrid(ax)
    ax.legend(loc="lower center")
    panel_label(ax, "c")
    return save(fig, outputs)


# ---------------------------------------------------------------------------
# Fig S4: thread scaling and determinism
# ---------------------------------------------------------------------------


def _size_label(mb: float) -> str:
    return f"{mb / 1000:g} Gb" if mb >= 1000 else f"{mb:g} Mb"


def fig_s4(tables: Mapping[str, Path], outputs: Sequence[Path]) -> Dict[str, Path]:
    """Wall time (a), speedup (b) and peak memory (c) of psmc-rs 0.2 against threads, per genome size."""
    sc = pd.read_csv(tables["scaling"])
    runs = pd.read_csv(tables["runs"])
    det = pd.read_csv(tables["determinism"])
    inputs = list(sc.sort_values("genome_mb")["input"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.4))
    handles = []
    for k, inp in enumerate(inputs):
        s = sc[sc["input"] == inp].sort_values("threads")
        r = runs[runs["input"] == inp]
        color = SLOTS[k % len(SLOTS)]
        label = _size_label(float(s["genome_mb"].iloc[0]))
        th = s["threads"].to_numpy()
        axes[0].plot(r["threads"], r["wall_seconds"], ls="none", marker="o", ms=2.2, color=color, alpha=0.45)
        handles += axes[0].plot(th, s["wall_median"], color=color, lw=1.1, marker="o", ms=4, mec="white", mew=0.8,
                                label=f"{label} (median of runs)")
        axes[0].plot(th, s["wall_median"].iloc[0] * th[0] / th, color=color, lw=0.6, ls=(0, (2, 2)), alpha=0.7)
        axes[1].plot(th, s["speedup"], color=color, lw=1.1, marker="o", ms=4, mec="white", mew=0.8, label=label)
        axes[2].plot(th, s["peak_rss_mb_median"], color=color, lw=1.1, marker="o", ms=4, mec="white", mew=0.8, label=label)
    all_threads = sorted(sc["threads"].unique())
    axes[1].plot([min(all_threads), max(all_threads)], [1, max(all_threads) / min(all_threads)], color=AXIS, lw=0.8,
                 zorder=0)
    axes[1].text(max(all_threads), max(all_threads) / min(all_threads), " ideal", fontsize=6, color=MUTED, va="center")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_threads)
        ax.set_xticklabels([str(t) for t in all_threads])
        ax.set_xlabel("Threads")
        ygrid(ax)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall time (s)")
    axes[1].set_ylabel(f"Speedup over {min(all_threads)} thread{'s' if min(all_threads) > 1 else ''}")
    axes[2].set_ylabel("Peak RSS (MB)")
    axes[2].set_ylim(bottom=0)
    handles.append(plt.Line2D([], [], color=MUTED, lw=0.6, ls=(0, (2, 2)), label="ideal scaling"))
    n_runs, identical = int(det["runs"].sum()), bool(det["json_identical"].all())
    axes[2].text(0.03, 0.04, f"JSON {'byte-identical' if identical else 'NOT identical'}\nacross threads ({n_runs} runs)",
                 transform=axes[2].transAxes, fontsize=6, color=INK_2)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    fig.tight_layout(w_pad=1.2)
    top_legend(fig, handles, min(len(handles), 5))
    return save(fig, outputs)


FIGURE_FUNCTIONS = {"s1": fig_s1, "s2": fig_s2, "s3": fig_s3, "s4": fig_s4}


def draw(part: str, tables: Mapping[str, Path], outputs: Sequence[Path]) -> Dict[str, Path]:
    """Draw the figure of ``part`` (``s1``..``s4``) from its tables into ``outputs`` (PDF and PNG)."""
    style()
    return FIGURE_FUNCTIONS[part](tables, outputs)


__all__ = ["draw", "fig_s1", "fig_s2", "fig_s3", "fig_s4", "style"]
