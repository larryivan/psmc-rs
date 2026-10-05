"""Fig 6 (E5): how well the bootstrap measures the uncertainty of a PSMC curve.

Drawn from E5's tables (``experiments/e5.py``) and the true model of its
truth record; every panel shares the time axis, the model's informative
window in years.

* **A, B**: one genome (``E5.example_dataset``) bootstrapped with trunks of
  each size: its fit and the 95% band of its bootstrap replicates, over the
  true Ne(t) and the 95% range of the fits of all the independent genomes
  (the true sampling distribution the band should match).
* **C**: bootstrap SD of log10 Ne over the true sampling SD at each time:
  median over the bootstrapped genomes (line) and their interquartile range
  (shade). The grey band around 1 is where a bootstrap that measured the SD
  exactly would fall, given that the true SD is itself estimated from a
  finite number of fits (chi-square 95% interval).
* **D**: pointwise coverage of the true Ne(t) by the 95% bands (share of the
  bootstrapped genomes; thick), and of the fits' geometric mean (thin): the
  estimator's expectation, which a well-calibrated band covers even where
  PSMC's bias keeps it from the truth.

Run ``python -m experiment.pipeline.figures.e5 [--profile P] [--runs-dir D]``;
it writes ``<runs>/E5/figures/E5_fig6.pdf`` (vector, fonts embedded) and
``E5_fig6.png`` (600 dpi). Both are byte-for-byte reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, demography, paths
from ..experiments import e5 as exp

MM = 1 / 25.4  # inches per mm
WIDTH_MM, HEIGHT_MM = 180.0, 128.0  # a two-column figure

# Colours: the two trunk sizes take the first two categorical slots (blue,
# orange; validated for colour-vision deficiency); the truth is primary ink
# and the true sampling distribution a neutral grey.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SAMPLING = "#d6d4cc"
REFERENCE_BAND = "#ebeae4"
TRUNK_COLOURS = ("#2a78d6", "#eb6834")
BAND_ALPHA = 0.18
# Legends inside a panel sit on a translucent surface so that lines behind them stay visible.
INNER_LEGEND = {"frameon": True, "framealpha": 0.85, "facecolor": "white", "edgecolor": "none", "fancybox": False}


def style() -> Dict:
    """matplotlib settings of the figure: 7 pt sans-serif, thin axes, TrueType fonts in the PDF."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.default": "regular",
        "font.size": 7.0,
        "axes.titlesize": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_table(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows: Sequence[Dict[str, str]], **where) -> List[Dict[str, str]]:
    """The rows whose columns equal ``where`` (compared as text), in ``point`` order when there is one."""
    out = [r for r in rows if all(r[k] == str(v) for k, v in where.items())]
    if out and "point" in out[0]:
        out.sort(key=lambda r: int(r["point"]))
    return out


def col(rows: Sequence[Dict[str, str]], key: str):
    import numpy as np

    return np.array([float(r[key]) for r in rows])


def truth_steps(record: Dict, lo: float, hi: float, gen_years: float, points: int = 2000):
    """The true Ne(t) from ``lo`` to ``hi`` years, with every change of the model at its exact time."""
    import numpy as np

    epochs = demography.epochs_from_record(record)
    changes = [e.start * gen_years for e in epochs if lo < e.start * gen_years < hi]
    years = np.unique(np.concatenate([np.logspace(math.log10(lo), math.log10(hi), points), changes]))
    ne = np.array([demography.ne_at(epochs, t / gen_years) for t in years])
    return years, ne


def trunk_label(summary_row: Dict[str, str]) -> str:
    mb = float(summary_row["trunk_mb"])
    return f"{mb:g} Mb trunks ({int(summary_row['trunks'])} per genome)"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def time_limits(lo: float, hi: float) -> Tuple[float, float]:
    """The x range: the grid's, widened to a power of ten within 5% of either end so that it gets a tick."""
    lo_decade = 10.0 ** math.floor(math.log10(lo))
    hi_decade = 10.0 ** math.ceil(math.log10(hi))
    return (lo_decade if lo / lo_decade < 1.05 else lo, hi_decade if hi_decade / hi < 1.05 else hi)


def _time_axis(ax, lo: float, hi: float) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*time_limits(lo, hi))
    ax.set_xlabel("Years ago")
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)


def _panel_letter(ax, letter: str) -> None:
    ax.text(-0.14, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left",
            color=INK)


def _title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, pad=4)


def draw_band_panel(ax, sampling, bands, truth_xy, colour: str) -> None:
    """Panel A/B: one genome's fit and bootstrap band over the truth and the sampling distribution."""
    years = col(sampling, "years")
    ax.fill_between(years, col(sampling, "lo_ne"), col(sampling, "hi_ne"), color=SAMPLING, linewidth=0, zorder=1)
    ax.fill_between(years, col(bands, "boot_lo_ne"), col(bands, "boot_hi_ne"), facecolor=colour, alpha=BAND_ALPHA,
                    edgecolor="none", zorder=2)
    for key in ("boot_lo_ne", "boot_hi_ne"):
        ax.plot(years, col(bands, key), color=colour, linewidth=0.6, zorder=3)
    ax.plot(*truth_xy, color=INK, linewidth=1.1, drawstyle="steps-post", zorder=4)
    ax.plot(years, col(bands, "fit_ne"), color=colour, linewidth=1.4, zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("Effective population size")


def ratio_reference(rows: List[Dict[str, str]]) -> Tuple[float, float]:
    """Where SD ratios fall when the bootstrap SD equals the true SD, from the true SD's chi-square interval."""
    import numpy as np

    sd = col(rows, "sd_true_log10")
    return float(np.median(col(rows, "sd_true_ci_lo") / sd)), float(np.median(col(rows, "sd_true_ci_hi") / sd))


def draw_ratio_panel(ax, sd_rows: Dict[int, List[Dict[str, str]]], colours: Dict[int, str], n_fits: int) -> None:
    """Panel C: bootstrap SD / true sampling SD by time (median and IQR over the bootstrapped genomes)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    lo_f, hi_f = ratio_reference(next(iter(sd_rows.values())))
    ax.axhspan(lo_f, hi_f, color=REFERENCE_BAND, linewidth=0, zorder=0)
    ax.axhline(1.0, color=INK_2, linewidth=0.7, zorder=1)
    for trunk, rows in sd_rows.items():
        years = col(rows, "years")
        ax.fill_between(years, col(rows, "ratio_q25"), col(rows, "ratio_q75"), facecolor=colours[trunk],
                        alpha=BAND_ALPHA, edgecolor="none", zorder=2)
        ax.plot(years, col(rows, "ratio_median"), color=colours[trunk], linewidth=1.4, zorder=3)
    ax.set_yscale("log")
    lo = min(min(col(r, "ratio_q25").min() for r in sd_rows.values()), lo_f)
    hi = max(max(col(r, "ratio_q75").max() for r in sd_rows.values()), hi_f)
    k_lo, k_hi = math.floor(math.log2(lo)), math.ceil(math.log2(hi))
    ax.set_ylim(2.0**k_lo, 2.0**k_hi)
    ax.yaxis.set_major_locator(FixedLocator([2.0**k for k in range(k_lo, k_hi + 1)]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Bootstrap SD / true SD of $log_{10}$ Ne")
    ax.legend(
        [Line2D([], [], color=INK_2, linewidth=0.7), Patch(facecolor=REFERENCE_BAND)],
        ["Equal SDs", f"Sampling error of the true SD ({n_fits} fits)"],
        loc="best", handlelength=1.6, borderaxespad=0.3, **INNER_LEGEND,
    )


def draw_coverage_panel(ax, cov_rows: Dict[int, List[Dict[str, str]]], colours: Dict[int, str], nominal: float) -> None:
    """Panel D: pointwise coverage of the true Ne(t) (thick) and of the fits' mean (thin)."""
    from matplotlib.lines import Line2D

    ax.axhline(nominal, color=INK_2, linewidth=0.7, zorder=1)
    for trunk, rows in cov_rows.items():
        years = col(rows, "years")
        ax.plot(years, col(rows, "coverage_fit_mean"), color=colours[trunk], linewidth=0.6, alpha=0.8,
                drawstyle="steps-mid", zorder=2)
        ax.plot(years, col(rows, "coverage"), color=colours[trunk], linewidth=1.4, drawstyle="steps-mid", zorder=3)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Share of genomes whose band covers")
    ax.legend(
        [Line2D([], [], color=INK, linewidth=1.4), Line2D([], [], color=INK, linewidth=0.6),
         Line2D([], [], color=INK_2, linewidth=0.7)],
        ["True Ne", "Mean of the fits", f"Nominal {nominal:.0%}"],
        loc="best", handlelength=1.6, borderaxespad=0.3, **INNER_LEGEND,
    )


def figure_legend(fig, labels: Dict[int, str], colours: Dict[int, str], n_fits: int) -> None:
    """The legend of all panels, above them: the truth, the sampling distribution and one entry per trunk size."""
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color=INK, linewidth=1.1), Patch(facecolor=SAMPLING)]
    names = ["True Ne", f"95% range of {n_fits} independent fits"]
    for trunk, label in labels.items():
        c = colours[trunk]
        handles.append((Patch(facecolor=c, alpha=BAND_ALPHA, edgecolor=c, linewidth=0.6),
                        Line2D([], [], color=c, linewidth=1.4)))
        names.append(label)
    fig.legend(handles, names, loc="outside upper center", ncol=len(names), handlelength=2.0,
               handler_map={tuple: HandlerTuple(ndivide=1)}, columnspacing=1.6)


def draw(tables: Dict[str, Path], truth_record: Dict, cfg: config.E5):
    """The figure (a matplotlib Figure) from E5's tables and the true model; call it under :func:`style`."""
    import matplotlib.pyplot as plt

    summary = read_table(tables["summary"])
    sampling = select(read_table(tables["sampling"]))
    sd_ratio = read_table(tables["sd_ratio"])
    coverage = read_table(tables["coverage"])
    bands = read_table(tables["bands"])
    trunks = list(cfg.trunk_bins)
    if len(trunks) > len(TRUNK_COLOURS):
        raise ValueError(f"the figure draws at most {len(TRUNK_COLOURS)} trunk sizes")
    rows_by_trunk = {t: select(summary, trunk_bins=t, window="informative")[0] for t in trunks}
    labels = {t: trunk_label(rows_by_trunk[t]) for t in trunks}
    colours = dict(zip(trunks, TRUNK_COLOURS))
    lo, hi = float(sampling[0]["years"]), float(sampling[-1]["years"])
    truth_xy = truth_steps(truth_record, *time_limits(lo, hi), cfg.scaling.gen_years)
    n_fits = int(sampling[0]["fits"])
    n_sets = int(rows_by_trunk[trunks[0]]["datasets"])
    d = cfg.example_dataset

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM * MM, HEIGHT_MM * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=2 * MM, h_pad=2 * MM, wspace=0.06, hspace=0.08)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for ax, trunk in zip((ax_a, ax_b), trunks):
        draw_band_panel(ax, sampling, select(bands, dataset=d, trunk_bins=trunk), truth_xy, colours[trunk])
        _title(ax, f"Genome {d}: fit and bootstrap 95% band, {labels[trunk].split(' (')[0]}")
    y_lo = min(ax.get_ylim()[0] for ax in (ax_a, ax_b))
    y_hi = max(ax.get_ylim()[1] for ax in (ax_a, ax_b))
    for ax in (ax_a, ax_b):
        ax.set_ylim(y_lo, y_hi)
    ax_b.set_ylabel("")
    draw_ratio_panel(ax_c, {t: select(sd_ratio, trunk_bins=t) for t in trunks}, colours, n_fits)
    _title(ax_c, f"Bootstrap spread vs true sampling spread (median, IQR of {n_sets} genomes)")
    draw_coverage_panel(ax_d, {t: select(coverage, trunk_bins=t) for t in trunks}, colours, cfg.band[1] - cfg.band[0])
    _title(ax_d, f"Pointwise coverage by the 95% band ({n_sets} genomes)")
    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        _time_axis(ax, lo, hi)
        _panel_letter(ax, letter)
    figure_legend(fig, labels, colours, n_fits)
    return fig


def save(fig, out_dir: Path) -> List[Path]:
    """Write the figure as PDF and PNG, without timestamps, so that a redraw is byte-identical."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in exp.FIGURES:
        path = out_dir / name
        tmp = path.with_name(path.name + ".partial")
        fmt = path.suffix.lstrip(".")
        meta = {"CreationDate": None} if fmt == "pdf" else {"Software": None}
        fig.savefig(tmp, format=fmt, metadata=meta, facecolor="white")
        tmp.replace(path)
        written.append(path)
    return written


def make(profile: config.Profile, runs_dir: Path) -> List[Path]:
    """Draw the figure of ``profile`` from the tables in ``runs_dir``; returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = profile.e5
    lay = exp.Layout(Path(runs_dir) / exp.NAME, cfg.model)
    tables = {k: lay.table(k) for k in exp.TABLES}
    record = json.loads(lay.truth(1).read_text())["truth"]
    # Fonts and the PDF font type are read when the figure is drawn and saved.
    with plt.rc_context(style()):
        fig = draw(tables, record, cfg)
        try:
            return save(fig, lay.root / "figures")
        finally:
            plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.figures.e5", description="Draw Fig 6 (E5) from E5's tables.")
    p.add_argument("--profile", default="production", choices=sorted(config.PROFILES))
    p.add_argument("--runs-dir", type=Path, help="runs directory (default: the profile's)")
    args = p.parse_args(argv)
    runs = (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()
    for path in make(config.get_profile(args.profile), runs):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
