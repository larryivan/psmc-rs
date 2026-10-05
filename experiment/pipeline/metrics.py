"""Time grids and RMSE of log10 Ne.

All metrics compare curves at the points of a common, log-spaced grid of
times in years. A fitted PSMC curve is a step function
(:class:`.results.Curve`); the truth is evaluated exactly at each point from
the model's epochs (:mod:`.demography`), with exponential growth where the
model has it. For simulations the informative window of a model is the
range between two quantiles of its true pairwise coalescence time (see
:class:`.config.TimeGrid` for why), the same for both tools and all
replicates.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from . import config, demography
from .results import Curve


def log_grid(lo: float, hi: float, points: int):
    """``points`` log-spaced times from ``lo`` to ``hi`` inclusive."""
    import numpy as np

    if not (0 < lo < hi) or points < 2:
        raise ValueError(f"bad grid [{lo}, {hi}] x {points}")
    return np.logspace(math.log10(lo), math.log10(hi), points)


def informative_window(
    epochs: Sequence[demography.Epoch], quantiles: Tuple[float, float], gen_years: float
) -> Tuple[float, float]:
    """The years between which the given share of true pairwise coalescence times fall."""
    lo, hi = quantiles
    return (
        demography.coalescence_quantile(epochs, lo) * gen_years,
        demography.coalescence_quantile(epochs, hi) * gen_years,
    )


def model_grid(epochs: Sequence[demography.Epoch], grid: config.TimeGrid, gen_years: float):
    """The informative-window grid of a model (years)."""
    lo, hi = informative_window(epochs, grid.quantiles, gen_years)
    return log_grid(lo, hi, grid.points)


def fixed_grid(grid: config.TimeGrid):
    """The model-independent grid of ``grid.fixed_years``."""
    return log_grid(grid.fixed_years[0], grid.fixed_years[1], grid.points)


def truth_on_grid(epochs: Sequence[demography.Epoch], years, gen_years: float):
    """The true Ne at each time of ``years``."""
    import numpy as np

    return np.array([demography.ne_at(epochs, float(t) / gen_years) for t in years])


def rmse_log10(a, b) -> float:
    """Root mean squared difference of log10 of two arrays of positive sizes."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("arrays must be non-empty and of equal shape")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("sizes must be positive")
    return float(np.sqrt(np.mean((np.log10(a) - np.log10(b)) ** 2)))


def rmse_vs_truth(curve: Curve, epochs: Sequence[demography.Epoch], years, gen_years: float) -> float:
    """RMSE of log10 Ne of a fitted curve against the truth on ``years``."""
    return rmse_log10(curve.at(years), truth_on_grid(epochs, years, gen_years))


def rmse_between(a: Curve, b: Curve, years) -> float:
    """RMSE of log10 Ne between two fitted curves on ``years``."""
    return rmse_log10(a.at(years), b.at(years))


def log10_bias(curve: Curve, epochs: Sequence[demography.Epoch], years, gen_years: float):
    """log10(fitted Ne / true Ne) at each time of ``years``."""
    import numpy as np

    return np.log10(curve.at(years)) - np.log10(truth_on_grid(epochs, years, gen_years))
