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
from typing import List, Sequence, Tuple

from . import config, demography
from .results import Curve, Fit, pattern_groups


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


# ---------------------------------------------------------------------------
# The truth as the fitted model's pattern can show it
# ---------------------------------------------------------------------------


def free_spans(fit: Fit) -> List[Tuple[int, int]]:
    """``(first, last + 1)`` intervals of each free lambda of a fit, from its pattern (one each without one)."""
    n_intervals = fit.n + 1
    if not fit.pattern:
        return [(k, k + 1) for k in range(n_intervals)]
    out: List[Tuple[int, int]] = []
    k = 0
    for count, width in pattern_groups(fit.pattern):
        for _ in range(count):
            out.append((k, k + width))
            k += width
    if k != n_intervals:
        raise ValueError(f"pattern {fit.pattern} covers {k} intervals, the fit has {n_intervals}")
    return out


def _epoch_hazard(e: demography.Epoch, t):
    """The hazard accumulated in epoch ``e`` from its start to each time of ``t`` (inside the epoch)."""
    import numpy as np

    x = np.asarray(t, dtype=float) - e.start
    if e.growth_rate == 0:
        return x / (2.0 * e.start_size)
    return np.expm1(e.growth_rate * x) / (2.0 * e.start_size * e.growth_rate)


def _survival_integral(epochs: Sequence[demography.Epoch], a: float, b: float, points: int) -> float:
    """``integral of exp(-(H(t) - H(a))) dt`` over ``[a, b)`` generations (H: the cumulative coalescence hazard).

    Exact within epochs of constant size; trapezoids on ``points``
    log-spaced times within epochs of exponential change. An open span
    (``b`` infinite) needs a constant oldest epoch, as every model here has.
    """
    import numpy as np

    h_a = demography.cumulative_hazard(epochs, a)
    tail = 0.0
    if math.isinf(b):
        last = epochs[-1]
        if last.growth_rate != 0:
            raise ValueError("an open span needs a constant oldest epoch")
        cut = max(a, last.start)
        tail = math.exp(-(demography.cumulative_hazard(epochs, cut) - h_a)) * 2.0 * last.start_size
        b = cut
    knots = sorted({a, b, *(e.start for e in epochs if a < e.start < b)})
    total = 0.0
    for lo, hi in zip(knots[:-1], knots[1:]):
        e = epochs[demography._epoch_index(epochs, lo)]
        s_lo = math.exp(-(demography.cumulative_hazard(epochs, lo) - h_a))
        if e.growth_rate == 0:  # exactly: S falls exponentially at rate 1 / (2 N)
            total += s_lo * 2.0 * e.start_size * -math.expm1(-(hi - lo) / (2.0 * e.start_size))
            continue
        first = lo if lo > 0 else min(hi * 1e-6, 1e-3)
        t = np.logspace(math.log10(first), math.log10(hi), points)
        if lo == 0:
            t = np.concatenate([[0.0], t])
        h = demography.cumulative_hazard(epochs, e.start) + _epoch_hazard(e, t) - h_a
        s = np.exp(-h)
        total += float(np.sum(np.diff(t) * (s[1:] + s[:-1]) / 2.0))
    return total + tail


def projected_ne(epochs: Sequence[demography.Epoch], a: float, b: float, points: int = 400) -> float:
    """The true Ne over ``[a, b)`` generations as one constant size, in coalescence-rate terms.

    A constant size over the span would be estimated from the coalescences
    that fall in it over the time the two lineages spend in it uncoalesced:
    rate = integral of S(t) / (2 N(t)) over integral of S(t), S the true
    pairwise survival. The numerator is S(a) - S(b), so the projected size is
    integral of S / (2 (S(a) - S(b))), with S taken relative to S(a).
    """
    h_a = demography.cumulative_hazard(epochs, a)
    gone = 1.0 if math.isinf(b) else -math.expm1(-(demography.cumulative_hazard(epochs, b) - h_a))
    if gone <= 0.0:
        return demography.ne_at(epochs, a)
    return _survival_integral(epochs, a, b, points) / (2.0 * gone)


def projected_truth(fit: Fit, epochs: Sequence[demography.Epoch], mu: float, gen_years: float, bin_size: float) -> Curve:
    """The truth projected onto the fit's free-parameter spans: one size per free lambda, in years.

    Each free lambda of the fit holds over a span of its intervals (its
    pattern; under ``4+25*2+4+6`` the first spans intervals 0-3). A curve
    with this time grid can at best show the true Ne averaged over each span
    in coalescence-rate terms (:func:`projected_ne`). RMSE against this
    curve separates the estimation error from the error of approximating
    the truth by the pattern's steps, which no estimator with the pattern
    can remove.
    """
    curve = fit.curve(mu, gen_years, bin_size)
    starts = list(curve.years)
    years: List[float] = []
    ne: List[float] = []
    for first, end in free_spans(fit):
        a = starts[first] / gen_years
        b = starts[end] / gen_years if end < len(starts) else math.inf
        years.append(starts[first])
        ne.append(projected_ne(epochs, a, b))
    return Curve(tuple(years), tuple(ne))


def rmse_vs_projected_truth(curve: Curve, projected: Curve, years) -> float:
    """RMSE of log10 Ne of a fitted curve against its pattern-projected truth (:func:`projected_truth`)."""
    return rmse_log10(curve.at(years), projected.at(years))
