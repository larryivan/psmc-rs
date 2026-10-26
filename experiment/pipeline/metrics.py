"""Time grids and RMSE of log10 Ne.

Metrics against the truth compare curves at the points of a common,
log-spaced grid of times in years. A fitted PSMC curve is a step function
(:class:`.results.Curve`); the truth is evaluated exactly at each point from
the model's epochs (:mod:`.demography`), with exponential growth where the
model has it. Two fits of the same time pattern are compared interval by
interval instead (:func:`interval_diff`), and two fits whose time scales
differ exactly over log time (:func:`log_time_diff`). For simulations the
informative window of a model is the range between two quantiles of its
true pairwise coalescence time (see :class:`.config.TimeGrid` for why), the
same for both tools and all replicates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    """RMSE of log10 Ne between two fitted curves on ``years``.

    Noisy for two close fits: a time between two nearly equal boundaries
    picks up a whole step of one curve. Compare fits of the same pattern with
    :func:`interval_rmse`, and fits whose time scales differ with
    :func:`log_time_diff`.
    """
    return rmse_log10(a.at(years), b.at(years))


def log10_bias(curve: Curve, epochs: Sequence[demography.Epoch], years, gen_years: float):
    """log10(fitted Ne / true Ne) at each time of ``years``."""
    import numpy as np

    return np.log10(curve.at(years)) - np.log10(truth_on_grid(epochs, years, gen_years))


# ---------------------------------------------------------------------------
# Two fitted curves compared over log time on a window
# ---------------------------------------------------------------------------


class _WeightedDiff:
    """log10(Ne_a / Ne_b) on pieces of a window, each weighted by the log time it spans."""

    log10_ratio: Tuple[float, ...]
    weight: Tuple[float, ...]

    @property
    def rmse(self) -> float:
        """Root of the log-time-weighted mean of the squared differences over the window."""
        total = sum(self.weight)
        return math.sqrt(sum(w * d * d for w, d in zip(self.weight, self.log10_ratio)) / total)

    @property
    def max_abs(self) -> float:
        """Largest |difference| on the window, a sliver between two near boundaries included."""
        return max(abs(d) for d in self.log10_ratio)


@dataclass(frozen=True)
class LogTimeDiff(_WeightedDiff):
    """The pieces between both curves' boundaries in a window (see :func:`log_time_diff`)."""

    log10_ratio: Tuple[float, ...]  # one per piece, where both curves are constant
    weight: Tuple[float, ...]  # ln(end / start) of the piece


@dataclass(frozen=True)
class IntervalDiff(_WeightedDiff):
    """The PSMC intervals of two fits of the same pattern (see :func:`interval_diff`)."""

    interval: Tuple[int, ...]  # the intervals overlapping the window
    log10_ratio: Tuple[float, ...]  # log10(Ne_a,k / Ne_b,k)
    weight: Tuple[float, ...]  # ln of the part of b's interval k inside the window (ln(end / start))
    all_log10_ratio: Tuple[float, ...]  # every interval, inside the window or not

    @property
    def max_abs_any(self) -> float:
        """Largest |difference| of any interval, outside the window too (a runaway first lambda)."""
        return max(abs(d) for d in self.all_log10_ratio)


def _window(window: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = float(window[0]), float(window[1])
    if not 0 < lo < hi:
        raise ValueError(f"bad window [{lo}, {hi}]")
    return lo, hi


def log_time_diff(a: Curve, b: Curve, window: Tuple[float, float]) -> LogTimeDiff:
    """Two step curves compared exactly over log time on ``window`` (years).

    Between consecutive boundaries of either curve both are constant, so
    the mean over log time of the squared difference of log10 Ne is a sum
    over those pieces, each weighted by ln(end / start): the limit of
    :func:`rmse_between` on an ever denser log grid, without its sampling
    noise. Suits fits whose time scales differ (by percents in theta), where
    matching intervals (:func:`interval_diff`) would compare different times.
    """
    import numpy as np

    lo, hi = _window(window)
    knots = sorted({lo, hi, *(float(t) for t in (*a.years, *b.years) if lo < t < hi)})
    starts = knots[:-1]
    diffs = np.log10(a.at(starts)) - np.log10(b.at(starts))
    weights = [math.log(e / s) for s, e in zip(knots[:-1], knots[1:])]
    return LogTimeDiff(tuple(float(d) for d in diffs), tuple(weights))


def interval_log10_ratio(a: Curve, b: Curve):
    """log10(Ne_a,k / Ne_b,k) for every interval k of two curves with the same number of intervals."""
    import numpy as np

    if len(a.ne) != len(b.ne):
        raise ValueError(f"the curves have {len(a.ne)} and {len(b.ne)} intervals")
    return np.log10(np.asarray(a.ne, dtype=float)) - np.log10(np.asarray(b.ne, dtype=float))


def interval_diff(a: Curve, b: Curve, window: Tuple[float, float]) -> IntervalDiff:
    """Two fits of the same time pattern compared interval by interval on ``window`` (years).

    Interval k of ``a`` is compared with interval k of ``b``, the
    reference, whose boundaries (in years) decide which intervals overlap
    the window and how much each weighs: the log of the share of the
    interval inside the window, ``ln(min(end, hi) / max(start, lo))``.

    Why intervals and not times: the two fits' boundaries are the same
    function of t_max in units of 2 N0, and differ in years only through
    their N0 (theta) and t_max, which differ by up to about 1e-3 (relative)
    between psmc-rs and C psmc (E1_rs_vs_c, E2_rs_vs_c).
    A fixed time between two such nearly equal boundaries reads a different
    step of each curve, a difference of up to 2 in log10 Ne that says nothing
    about how well the fits agree; integrating over log time
    (:func:`log_time_diff`) weighs such a sliver by its width, but still
    mixes the shift of the time scale into the difference of the sizes.
    Matching the intervals compares what each fit estimated, lambda_k * N0,
    directly; the time scales are reported separately (theta, t_max).

    Why weighted by log time: the RMSE is then the RMSE over log time on the
    window, which the RMSE against the truth on a log-spaced grid also
    measures, so the two read side by side; and an interval that only
    reaches into the window (a recent first lambda) counts by its part inside.
    """
    lo, hi = _window(window)
    ratio = interval_log10_ratio(a, b)
    starts = list(b.years)
    idx, diffs, weights = [], [], []
    for k, d in enumerate(ratio):
        end = starts[k + 1] if k + 1 < len(starts) else math.inf
        s, e = max(starts[k], lo), min(end, hi)
        if e > s:
            idx.append(k)
            diffs.append(float(d))
            weights.append(math.log(e / s))
    if not idx:
        raise ValueError(f"no interval overlaps the window [{lo}, {hi}]")
    return IntervalDiff(tuple(idx), tuple(diffs), tuple(weights), tuple(float(d) for d in ratio))


def interval_rmse(a: Curve, b: Curve, window: Tuple[float, float]) -> float:
    """The RMSE of :func:`interval_diff`."""
    return interval_diff(a, b, window).rmse


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
        e = epochs[demography.epoch_index(epochs, lo)]
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
    starts = list(fit.curve(mu, gen_years, bin_size).years)
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
