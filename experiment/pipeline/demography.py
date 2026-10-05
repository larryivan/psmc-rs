"""The demographic models as ``msprime.Demography``, and their true Ne(t).

A model (:class:`.config.DemographicModel`) is a present size and a list of
size changes back in time. :func:`epochs` turns it into epochs of constant
or exponentially changing size, exactly as msprime does; from them the true
Ne(t) and the distribution of the pairwise coalescence time follow in
closed form (two lineages in a diploid population of size N coalesce at
rate 1 / (2N) per generation). Everything here is in generations and diploid
individuals.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import config

POPULATION = "pop_0"


@dataclass(frozen=True)
class Epoch:
    """``[start, end)`` generations ago, size ``start_size * exp(-growth_rate * (t - start))``."""

    start: float
    end: float
    start_size: float
    growth_rate: float

    def size_at(self, t: float) -> float:
        return self.start_size * math.exp(-self.growth_rate * (t - self.start))

    @property
    def end_size(self) -> float:
        return self.start_size if math.isinf(self.end) and self.growth_rate == 0 else self.size_at(self.end)

    def hazard(self, t: float) -> float:
        """Integral of 1 / (2 N(s)) over ``[start, min(t, end)]``."""
        x = min(t, self.end) - self.start
        if x <= 0:
            return 0.0
        r = self.growth_rate
        if r == 0:
            return x / (2.0 * self.start_size)
        return math.expm1(r * x) / (2.0 * self.start_size * r)

    def time_for_hazard(self, h: float) -> float:
        """The time in this epoch at which the hazard accumulated from ``start`` reaches ``h`` (inf if never)."""
        r = self.growth_rate
        if r == 0:
            x = 2.0 * self.start_size * h
        else:
            arg = 1.0 + 2.0 * self.start_size * r * h
            if arg <= 0:
                return math.inf
            x = math.log(arg) / r
        t = self.start + x
        return t if t < self.end else math.inf


def epochs(model: config.DemographicModel) -> List[Epoch]:
    """The epochs of ``model``, from the present back, as msprime builds them.

    A change with ``size=None`` keeps the size the previous epoch reached at
    its time; the present epoch has no growth.
    """
    out: List[Epoch] = []
    start, size, rate = 0.0, float(model.initial_size), 0.0
    for change in sorted(model.changes, key=lambda c: c.time):
        if change.time <= start:
            raise ValueError(f"{model.name}: size changes must be at increasing times > 0")
        out.append(Epoch(start, float(change.time), size, rate))
        reached = size * math.exp(-rate * (change.time - start))
        size = float(change.size) if change.size is not None else reached
        rate = float(change.growth_rate)
        start = float(change.time)
    out.append(Epoch(start, math.inf, size, rate))
    return out


def to_msprime(model: config.DemographicModel):
    """``model`` as an ``msprime.Demography`` with one population, ``pop_0``."""
    import msprime

    demography = msprime.Demography()
    demography.add_population(name=POPULATION, initial_size=model.initial_size)
    for change in sorted(model.changes, key=lambda c: c.time):
        demography.add_population_parameters_change(
            time=change.time,
            initial_size=change.size,
            growth_rate=change.growth_rate,
            population=POPULATION,
        )
    return demography


def _epoch_index(eps: Sequence[Epoch], t: float) -> int:
    for i, e in enumerate(eps):
        if t < e.end:
            return i
    return len(eps) - 1


def ne_at(eps: Sequence[Epoch], t: float) -> float:
    """The true (diploid) size ``t`` generations ago."""
    if t < 0:
        raise ValueError("time must be >= 0")
    return eps[_epoch_index(eps, t)].size_at(t)


def cumulative_hazard(eps: Sequence[Epoch], t: float) -> float:
    """Integral of the pairwise coalescence rate 1 / (2 N(s)) from 0 to ``t``."""
    return sum(e.hazard(t) for e in eps if e.start < t)


def coalescence_cdf(eps: Sequence[Epoch], t: float) -> float:
    """P(the two haplotypes have coalesced by ``t`` generations ago)."""
    return -math.expm1(-cumulative_hazard(eps, t))


def coalescence_quantile(eps: Sequence[Epoch], q: float) -> float:
    """The time by which the two haplotypes have coalesced with probability ``q``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    target = -math.log1p(-q)
    acc = 0.0
    for e in eps:
        t = e.time_for_hazard(target - acc)
        if not math.isinf(t):
            return t
        acc += e.hazard(e.end)
    raise ValueError("the coalescence time distribution does not reach this quantile")


def truth_record(model: config.DemographicModel, quantiles: Sequence[float] = (0.01, 0.025, 0.5, 0.975, 0.99)) -> Dict:
    """The true Ne(t) of ``model`` as a JSON-ready record (generations, diploid sizes)."""
    eps = epochs(model)
    rec: Dict = {
        "model": model.name,
        "source": model.source,
        "units": {"time": "generations ago", "size": "diploid individuals"},
        "initial_size": model.initial_size,
        "changes": [
            {"time": c.time, "size": c.size, "growth_rate": c.growth_rate} for c in model.changes
        ],
        "epochs": [
            {
                "start": e.start,
                "end": None if math.isinf(e.end) else e.end,
                "start_size": e.start_size,
                "end_size": e.end_size,
                "growth_rate": e.growth_rate,
            }
            for e in eps
        ],
        "coalescence_quantiles": {f"{q:g}": coalescence_quantile(eps, q) for q in quantiles},
    }
    if model.ms_n0 is not None:
        rec["ms"] = {
            "n0": model.ms_n0,
            "events": [{"time_4n0": t, "size_ratio": x} for t, x in model.ms_events],
        }
    return rec


def epochs_from_record(record: Dict) -> List[Epoch]:
    """The epochs saved by :func:`truth_record`."""
    return [
        Epoch(
            float(e["start"]),
            math.inf if e["end"] is None else float(e["end"]),
            float(e["start_size"]),
            float(e["growth_rate"]),
        )
        for e in record["epochs"]
    ]


def compare_with_debugger(model: config.DemographicModel, reference=None, rel_tol: float = 1e-9) -> Dict:
    """Compare :func:`epochs` of ``model`` with an msprime ``DemographyDebugger``.

    ``reference`` is a ``msprime.Demography`` to compare with (default: the
    model's own :func:`to_msprime`). Returns the number of epochs and the
    largest relative difference in start/end times, sizes and growth rates,
    and raises ``AssertionError`` when they differ by more than ``rel_tol``.
    """
    ref = (reference if reference is not None else to_msprime(model)).debug()
    mine = epochs(model)
    if len(ref.epochs) != len(mine):
        raise AssertionError(f"{model.name}: {len(mine)} epochs, msprime has {len(ref.epochs)}")
    worst = 0.0
    for e, r in zip(mine, ref.epochs):
        pop = r.populations[0]
        pairs = [
            (e.start, r.start_time),
            (e.end, r.end_time),
            (e.start_size, pop.start_size),
            (e.end_size, pop.end_size),
            (e.growth_rate, pop.growth_rate),
        ]
        for a, b in pairs:
            if math.isinf(a) or math.isinf(b):
                if a != b:
                    raise AssertionError(f"{model.name}: {a} != {b}")
                continue
            diff = abs(a - b) / max(abs(a), abs(b), 1e-300) if (a or b) else 0.0
            worst = max(worst, diff)
    if worst > rel_tol:
        raise AssertionError(f"{model.name}: epochs differ from msprime's by a relative {worst:.3g}")
    return {"epochs": len(mine), "max_rel_diff": worst}


def check_zigzag_against_stdpopsim(rel_tol: float = 1e-9) -> Dict:
    """Check :data:`.config.ZIGZAG` against stdpopsim's Zigzag_1S14, epoch by epoch.

    Both demographies go through msprime's DemographyDebugger; the epochs
    (times, start and end sizes, growth rates) must agree within ``rel_tol``,
    and so must the pairwise coalescence probabilities on a grid of times.
    """
    import numpy as np
    import stdpopsim

    species, model_id = config.ZIGZAG_STDPOPSIM
    reference = stdpopsim.get_species(species).get_demographic_model(model_id)
    result = compare_with_debugger(config.ZIGZAG, reference=reference.model, rel_tol=rel_tol)
    ref_pop = reference.model.populations[0].name
    times = np.linspace(0.0, 2e5, 2001)
    with warnings.catch_warnings():
        # msprime integrates numerically and may warn about its step size;
        # both sides use the same steps, so they are compared like for like.
        warnings.simplefilter("ignore", UserWarning)
        _, p_ref = reference.model.debug().coalescence_rate_trajectory(times, lineages={ref_pop: 2})
        _, p_mine = to_msprime(config.ZIGZAG).debug().coalescence_rate_trajectory(times, lineages={POPULATION: 2})
    worst = float(np.max(np.abs(p_ref - p_mine)))
    if worst > rel_tol:
        raise AssertionError(f"zigzag: coalescence probabilities differ from stdpopsim's by {worst:.3g}")
    result.update(
        stdpopsim_version=stdpopsim.__version__,
        stdpopsim_model=f"{species}/{model_id}",
        max_abs_diff_coalescence_probability=worst,
    )
    return result
