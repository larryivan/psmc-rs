"""C psmc ``.psmc`` and psmc-rs JSON outputs, read as PSMC models and Ne(t) curves.

A :class:`Fit` is a PSMC model in C psmc's parametrisation: theta and rho
per bin, ``t_max`` (in units of 2*N0 generations), C's ``n`` (the number of
intervals minus one) and one lambda per interval. Its time boundaries are
C psmc's (``psmc_update_intv`` with alpha 0.1)::

    t_k = alpha * (exp(beta * k) - 1),  beta = ln(1 + t_max / alpha) / n,  k < n;  t_n = t_max

and it scales as ``psmc_plot.pl`` does: N0 = theta / (4 * mu * bin_size),
Ne = lambda * N0 and years = 2 * N0 * t * g. Both tools' outputs go through
the same code: C psmc's from the ``PA`` line of a round (9 decimals), psmc-rs's
from its JSON.

Iterations are counted as completed EM iterations. C psmc's round ``k``
prints the parameters after ``k`` iterations (``TR``, ``MT``, ``RS``, ``PA``)
and the log-likelihood of the parameters after ``k - 1`` (``LK``: the
E-step of round ``k`` runs on them; round 0 prints a placeholder 0).
psmc-rs's ``loglike[i]`` is the log-likelihood after ``i`` iterations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ALPHA = 0.1  # C psmc's pp->alpha


def pattern_groups(pattern: str) -> List[Tuple[int, int]]:
    """``(count, width)`` groups of a pattern: ``a*b`` is ``a`` free lambdas of ``b`` intervals, ``b`` one of ``b``."""
    out = []
    for part in pattern.split("+"):
        part = part.strip()
        if not part:
            continue
        if "*" in part:
            a, b = part.split("*", 1)
            out.append((int(a), int(b)))
        else:
            out.append((1, int(part)))
    if not out:
        raise ValueError(f"empty pattern {pattern!r}")
    return out


def pattern_intervals(pattern: str) -> int:
    return sum(a * b for a, b in pattern_groups(pattern))


def pattern_free(pattern: str) -> int:
    return sum(a for a, _ in pattern_groups(pattern))


def expand_lambdas(free: Sequence[float], pattern: Optional[str], n_intervals: int) -> List[float]:
    """One lambda per interval from one per free parameter (identity without a pattern)."""
    free = [float(x) for x in free]
    if pattern is None:
        if len(free) != n_intervals:
            raise ValueError(f"{len(free)} lambdas for {n_intervals} intervals")
        return free
    groups = pattern_groups(pattern)
    if len(free) != sum(a for a, _ in groups):
        raise ValueError(f"{len(free)} lambdas for pattern {pattern} ({pattern_free(pattern)} free)")
    out: List[float] = []
    i = 0
    for a, b in groups:
        for _ in range(a):
            out.extend([free[i]] * b)
            i += 1
    if len(out) != n_intervals:
        raise ValueError(f"pattern {pattern} gives {len(out)} intervals, expected {n_intervals}")
    return out


def boundaries(t_max: float, n: int, alpha: float = ALPHA) -> List[float]:
    """C psmc's interval boundaries ``t_0 .. t_n`` (``t_0 = 0``, ``t_n = t_max``)."""
    beta = math.log(1.0 + t_max / alpha) / n
    return [alpha * (math.exp(beta * k) - 1.0) for k in range(n)] + [float(t_max)]


@dataclass(frozen=True)
class Curve:
    """A step function Ne(t): ``ne[k]`` from ``years[k]`` to ``years[k + 1]`` (the last to infinity)."""

    years: Tuple[float, ...]
    ne: Tuple[float, ...]

    def at(self, t_years):
        """Ne at each time of ``t_years`` (numpy array or sequence)."""
        import numpy as np

        idx = np.searchsorted(np.asarray(self.years), np.asarray(t_years, dtype=float), side="right") - 1
        return np.asarray(self.ne)[np.clip(idx, 0, len(self.ne) - 1)]


@dataclass(frozen=True)
class Fit:
    theta: float
    rho: float
    t_max: float
    n: int  # C's n: intervals - 1
    lambdas: Tuple[float, ...]  # one per interval (n + 1)
    pattern: Optional[str] = None
    free_lambdas: Tuple[float, ...] = ()

    def boundaries(self) -> List[float]:
        return boundaries(self.t_max, self.n)

    def n0(self, mu: float, bin_size: int) -> float:
        return self.theta / (4.0 * mu * bin_size)

    def curve(self, mu: float, gen_years: float, bin_size: int) -> Curve:
        n0 = self.n0(mu, bin_size)
        years = tuple(2.0 * n0 * t * gen_years for t in self.boundaries())
        ne = tuple(lam * n0 for lam in self.lambdas)
        return Curve(years, ne)


# ---------------------------------------------------------------------------
# psmc-rs JSON
# ---------------------------------------------------------------------------


def read_rs_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def fit_from_rs(params: Dict) -> Fit:
    """The model of a psmc-rs (0.1 or 0.2) parameter JSON."""
    n = int(params["n_steps"])
    pattern = params.get("pattern")
    lam = expand_lambdas(params["lam"], pattern, n + 1)
    return Fit(
        theta=float(params["theta"]),
        rho=float(params["rho"]),
        t_max=float(params["t_max"]),
        n=n,
        lambdas=tuple(lam),
        pattern=pattern,
        free_lambdas=tuple(float(x) for x in params["lam"]),
    )


def rs_loglikes(params: Dict) -> List[float]:
    """``loglike[i]``: the log-likelihood after ``i`` EM iterations (psmc-rs 0.2)."""
    return [float(x) for x in params.get("loglike", [])]


# ---------------------------------------------------------------------------
# C psmc .psmc (also psmc-rs --psmc-out, which follows the format)
# ---------------------------------------------------------------------------


@dataclass
class Round:
    """One ``RD`` block of a ``.psmc`` file."""

    round: int
    lk: Optional[float] = None
    qd: Optional[Tuple[float, float]] = None
    ri: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    t_max: Optional[float] = None
    c_pi: Optional[float] = None
    n_recomb: Optional[float] = None
    rs: List[Tuple[int, float, float, float, float, float]] = field(default_factory=list)
    pa: Optional[Dict] = None
    complete: bool = False  # closed by '//'

    def fit(self) -> Fit:
        """The model of this round, from its ``PA`` line."""
        if self.pa is None:
            raise ValueError(f"round {self.round} has no PA line")
        pattern = self.pa["pattern"]
        n_intervals = pattern_intervals(pattern) if pattern else len(self.rs)
        free = self.pa["lambdas"]
        lam = expand_lambdas(free, pattern, n_intervals)
        return Fit(
            theta=self.pa["theta"],
            rho=self.pa["rho"],
            t_max=self.pa["t_max"],
            n=n_intervals - 1,
            lambdas=tuple(lam),
            pattern=pattern,
            free_lambdas=tuple(free),
        )


@dataclass
class PsmcFile:
    header: Dict
    rounds: List[Round]

    def complete_rounds(self) -> List[Round]:
        return [r for r in self.rounds if r.complete]

    def last(self) -> Round:
        rounds = self.complete_rounds()
        if not rounds:
            raise ValueError("no complete round")
        return rounds[-1]

    def loglikes(self) -> Dict[int, float]:
        """``{i: log-likelihood after i iterations}`` from the LK lines of rounds 1 and later."""
        return {r.round - 1: r.lk for r in self.complete_rounds() if r.round >= 1 and r.lk is not None}


_HEADER_PATTERNS = {
    "version": r"^Version:\s*(.+)$",
    "pattern": r"^pattern:([^,]*), n:(\d+), n_free_lambdas:(\d+)",
    "iterations": r"^n_iterations:(\d+), skip:(\d+), max_t:([^,]+), theta/rho:(\S+)",
    "seqs": r"^n_seqs:(\d+), sum_L:(\d+), sum_n:(\d+)",
}


def _parse_pa(fields: List[str]) -> Dict:
    # "PA\t<pattern> theta rho max_t lambda... [divergence]"
    parts = fields[1].split() if len(fields) == 2 else " ".join(fields[1:]).split()
    pattern = parts[0]
    numbers = [float(x) for x in parts[1:]]
    n_free = pattern_free(pattern)
    return {
        "pattern": pattern,
        "theta": numbers[0],
        "rho": numbers[1],
        "t_max": numbers[2],
        "lambdas": numbers[3 : 3 + n_free],
    }


def parse_psmc(path: Path) -> PsmcFile:
    """Every round of a ``.psmc`` file: LK, QD, RI, TR, MT, C_pi/n_recomb, RS and PA lines."""
    import re

    header: Dict = {}
    rounds: List[Round] = []
    cur: Optional[Round] = None
    for line in Path(path).read_text().splitlines():
        tag, _, rest = line.partition("\t")
        if tag == "MM" and cur is None:
            for key, pat in _HEADER_PATTERNS.items():
                m = re.match(pat, rest)
                if not m:
                    continue
                if key == "version":
                    header["version"] = m.group(1).strip()
                elif key == "pattern":
                    header.update(pattern=m.group(1), n=int(m.group(2)), n_free=int(m.group(3)))
                elif key == "iterations":
                    header.update(
                        n_iterations=int(m.group(1)), max_t=float(m.group(3)), tr_ratio=float(m.group(4))
                    )
                elif key == "seqs":
                    header.update(n_seqs=int(m.group(1)), sum_L=int(m.group(2)), sum_n=int(m.group(3)))
        elif tag == "RD":
            cur = Round(round=int(rest))
            rounds.append(cur)
        elif cur is None:
            continue
        elif tag == "LK":
            cur.lk = float(rest)
        elif tag == "QD":
            a, _, b = rest.partition(" -> ")
            cur.qd = (float(a), float(b))
        elif tag == "RI":
            cur.ri = float(rest)
        elif tag == "TR":
            th, rh = rest.split("\t")[:2]
            cur.theta, cur.rho = float(th), float(rh)
        elif tag == "MT":
            cur.t_max = float(rest)
        elif tag == "MM":
            m = re.match(r"C_pi:\s*([^,]+),\s*n_recomb:\s*(\S+)", rest)
            if m:
                cur.c_pi, cur.n_recomb = float(m.group(1)), float(m.group(2))
        elif tag == "RS":
            f = rest.split("\t")
            cur.rs.append((int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
        elif tag == "PA":
            cur.pa = _parse_pa(line.split("\t"))
        elif line.startswith("//"):
            cur.complete = True
    return PsmcFile(header=header, rounds=rounds)


def rounds_table(psmc: PsmcFile) -> List[Dict]:
    """One row per complete round: iteration, theta, rho, t_max, n_recomb, and the LK it prints."""
    rows = []
    for r in psmc.complete_rounds():
        rows.append(
            {
                "round": r.round,
                "theta": r.theta,
                "rho": r.rho,
                "t_max": r.t_max,
                "n_recomb": r.n_recomb,
                "lk": r.lk if r.round >= 1 else None,
            }
        )
    return rows
