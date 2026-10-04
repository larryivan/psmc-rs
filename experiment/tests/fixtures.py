"""Small synthetic psmc-rs / C psmc outputs for the experiment tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

PATTERN = "4+25*2+4+6"
N_STEPS = 63  # 4 + 25*2 + 4 + 6 = 64 intervals
N_FREE = 28  # 1 + 25 + 1 + 1 free lambdas


def _lams(scale: float):
    return [scale * (1.0 + 0.5 * math.sin(i / 3.0)) for i in range(N_FREE)]


def write_rust_json(path: Path, scale: float = 1.0, theta: float = 0.002) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "theta": theta,
        "rho": theta / 4,
        "lam": _lams(scale),
        "t_max": 15.0,
        "n_steps": N_STEPS,
        "mu": 2.5e-8,
        "pattern": PATTERN,
    }
    path.write_text(json.dumps(params, indent=2))


def write_c_psmc(path: Path, scale: float = 1.0, theta: float = 0.002) -> None:
    """A two-round C psmc output (RD 0 unfitted, RD 20 final) with the same curve shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha, t_max = 0.1, 15.0
    beta = math.log(1 + t_max / alpha) / N_STEPS
    t = [alpha * (math.exp(beta * k) - 1.0) for k in range(N_STEPS)] + [t_max]
    groups = [(1, 4), (25, 2), (1, 4), (1, 6)]
    lam = []
    for (nr, gl), v in zip([g for g in groups for _ in range(g[0])], _lams(scale)):
        lam.extend([v] * gl)
    lines = ["CC\tsynthetic", "MM\tVersion: test"]
    for rd, lk, sc in ((0, 0.0, 1.0), (20, -12345.6, 1.0)):
        lines += [f"RD\t{rd}", f"LK\t{lk}", f"TR\t{theta}\t{theta / 4}"]
        lines += [f"RS\t{k}\t{t[k]:.6f}\t{lam[k] * sc:.6f}\t0\t0\t0" for k in range(N_STEPS + 1)]
        lines += ["PA\t" + PATTERN, "//"]
    path.write_text("\n".join(lines) + "\n")
