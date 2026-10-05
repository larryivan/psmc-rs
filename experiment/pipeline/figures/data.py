"""Reading the tables of :mod:`.schemas`, and the curve arithmetic the figures share.

A fitted PSMC curve is a step function (:class:`..results.Curve`); several
replicates are summarised pointwise (median, percentile band) on a common
log-spaced grid of times. The truth comes from the demographic model in
:mod:`..config` through :mod:`..demography`, exactly as the truth records
hold it, and is drawn with its exact change times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config, demography, metrics
from ..results import Curve
from .schemas import SCHEMAS


class MissingTable(FileNotFoundError):
    """A table a figure needs does not exist (its experiment has not been run)."""


class SchemaError(ValueError):
    """A table lacks columns a figure needs."""


def read(runs_dir: Path, key: str):
    """The table ``key`` of :data:`.schemas.SCHEMAS` as a DataFrame, its columns checked."""
    import pandas as pd

    schema = SCHEMAS[key]
    path = schema.path(runs_dir)
    if not path.exists():
        raise MissingTable(f"{path} does not exist (run {schema.experiment} first)")
    df = pd.read_csv(path)
    missing = [c for c in schema.names if c not in df.columns]
    if missing:
        raise SchemaError(f"{path}: missing column(s) {', '.join(missing)}; expected {', '.join(schema.names)}")
    return df


@dataclass
class Source:
    """What a figure draws from: the runs directory and the profile of its experiments."""

    runs_dir: Path
    profile: config.Profile

    def table(self, key: str):
        return read(self.runs_dir, key)

    def has(self, key: str) -> bool:
        return SCHEMAS[key].path(self.runs_dir).exists()


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def curves(df, keys: Sequence[str]) -> Dict[Tuple, Curve]:
    """One step curve per group of ``keys`` in a curve table (rows sorted by interval)."""
    out: Dict[Tuple, Curve] = {}
    for k, sub in df.groupby(list(keys), sort=True):
        sub = sub.sort_values("interval")
        out[k if isinstance(k, tuple) else (k,)] = Curve(
            tuple(float(x) for x in sub["years"]), tuple(float(x) for x in sub["ne"])
        )
    return out


def time_grid(lim: Tuple[float, float], points: int = 600):
    """``points`` log-spaced times spanning ``lim``."""
    return metrics.log_grid(lim[0], lim[1], points)


def on_grid(cs: Iterable[Curve], grid):
    """Ne of each curve at each time of ``grid``: an array of curves x times."""
    import numpy as np

    rows = [c.at(grid) for c in cs]
    return np.vstack(rows) if rows else np.empty((0, len(grid)))


def quantiles(cs: Iterable[Curve], grid, qs: Sequence[float]):
    """Pointwise quantiles of Ne over the curves (geometric, i.e. of log10 Ne): array qs x times."""
    import numpy as np

    m = on_grid(cs, grid)
    return 10 ** np.quantile(np.log10(m), qs, axis=0)


def median(cs: Iterable[Curve], grid):
    return quantiles(cs, grid, [0.5])[0]


def log10_ratio(a: Curve, b: Curve, grid):
    """log10(Ne_a / Ne_b) at each time of ``grid``."""
    import numpy as np

    return np.log10(a.at(grid)) - np.log10(b.at(grid))


def curve_span(cs: Iterable[Curve]) -> Tuple[float, float]:
    """The first positive and the last interval start over the curves (the fitted time range)."""
    firsts, lasts = [], []
    for c in cs:
        pos = [y for y in c.years if y > 0]
        if pos:
            firsts.append(pos[0])
            lasts.append(pos[-1])
    return (min(firsts), max(lasts)) if firsts else (1e3, 1e7)


def decades(lo: float, hi: float) -> Tuple[float, float]:
    """``[lo, hi]`` widened to whole powers of ten."""
    return 10.0 ** math.floor(math.log10(lo)), 10.0 ** math.ceil(math.log10(hi))


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


def model_epochs(model: str) -> List[demography.Epoch]:
    return demography.epochs(config.MODELS[model])


def truth_at(model: str, years, gen_years: float):
    """The true Ne of ``model`` at each time of ``years``."""
    return metrics.truth_on_grid(model_epochs(model), years, gen_years)


def truth_xy(model: str, gen_years: float, lim: Tuple[float, float], per_decade: int = 120):
    """Coordinates of the true Ne(t) within ``lim`` (years) for a plain line plot.

    Constant epochs are drawn from their exact start to their exact end (the
    joins are vertical steps); epochs of exponential change are sampled
    ``per_decade`` times per decade of time.
    """
    import numpy as np

    lo, hi = lim
    xs: List[float] = []
    ys: List[float] = []
    for e in model_epochs(model):
        a = max(e.start * gen_years, lo)
        b = min(e.end * gen_years, hi)
        if not b > a:
            continue
        if e.growth_rate == 0:
            t = np.array([a, b])
        else:
            n = max(int(per_decade * math.log10(b / a)), 2)
            t = np.logspace(math.log10(a), math.log10(b), n)
        xs.extend(t)
        ys.extend(e.size_at(float(x) / gen_years) for x in t)
    return np.asarray(xs), np.asarray(ys)


def informative_window(model: str, grid: config.TimeGrid, gen_years: float) -> Tuple[float, float]:
    """The model's informative window in years (the quantiles of the true coalescence time)."""
    return metrics.informative_window(model_epochs(model), grid.quantiles, gen_years)


def check_truth_table(df, gen_years: float, rtol: float = 1e-9) -> None:
    """Check that a truth table (model, years, ne) agrees with :func:`truth_at`."""
    import numpy as np

    for model, sub in df.groupby("model"):
        want = truth_at(str(model), sub["years"].to_numpy(), gen_years)
        got = sub["ne"].to_numpy(dtype=float)
        if not np.allclose(got, want, rtol=rtol, atol=0.0):
            raise ValueError(f"truth table of {model} disagrees with config.MODELS (max rel diff "
                             f"{float(np.max(np.abs(got / want - 1))):.3g})")


def fmt_bp(bp: float) -> str:
    """A length in bp as ``50 Mb``, ``200 kb`` or ``800 bp``."""
    for unit, size in (("Gb", 1e9), ("Mb", 1e6), ("kb", 1e3)):
        if bp >= size:
            v = bp / size
            return f"{v:.0f} {unit}" if abs(v - round(v)) < 1e-9 else f"{v:.1f} {unit}"
    return f"{bp:.0f} bp"


def fmt_sig(x: Optional[float], digits: int = 2) -> str:
    """``x`` with ``digits`` significant figures, as 1.2e-3 written 1.2 × 10⁻³."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    if -2 <= exp <= 3:
        return f"{x:.{max(digits - 1 - exp, 0)}f}"
    mant = x / 10**exp
    return rf"{mant:.{digits - 1}f}$\times$10$^{{{exp}}}$"


__all__ = [
    "MissingTable", "SchemaError", "Source", "check_truth_table", "curve_span", "curves", "decades", "fmt_bp",
    "fmt_sig", "informative_window", "log10_ratio", "median", "model_epochs", "on_grid", "quantiles", "read",
    "time_grid", "truth_at", "truth_xy",
]
