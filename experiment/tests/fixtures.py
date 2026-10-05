"""Small synthetic psmc-rs / C psmc outputs and stand-in tools for the experiment tests."""

from __future__ import annotations

import json
import math
import stat
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

PATTERN = "4+25*2+4+6"
N_STEPS = 63  # 4 + 25*2 + 4 + 6 = 64 intervals
N_FREE = 28  # 1 + 25 + 1 + 1 free lambdas


def _lams(scale: float):
    return [scale * (1.0 + 0.5 * math.sin(i / 3.0)) for i in range(N_FREE)]


def rust_loglike(i: int) -> float:
    """The log-likelihood the stand-in psmc-rs reports after ``i`` iterations."""
    return -1000.0 - i


def write_rust_json(
    path: Path, scale: float = 1.0, theta: float = 0.002, rho_ratio: float = 4.0, n_iter: Optional[int] = None
) -> None:
    """A psmc-rs parameter JSON: the 0.1 fields, plus 0.2's run metadata when ``n_iter`` is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "theta": theta,
        "rho": theta / rho_ratio,
        "lam": _lams(scale),
        "t_max": 15.0,
        "n_steps": N_STEPS,
        "mu": 2.5e-8,
        "pattern": PATTERN,
    }
    if n_iter is not None:
        params.update(
            version="0.2.0",
            n_iter=n_iter,
            loglike=[rust_loglike(i) for i in range(n_iter + 1)],
            tr_ratio=rho_ratio,
        )
    path.write_text(json.dumps(params, indent=2))


def write_c_psmc(
    path: Path, scale: float = 1.0, theta: float = 0.002, rounds: Sequence[int] = (0, 20)
) -> None:
    """A C psmc output with one block per round in ``rounds``, all with the same curve.

    The default is a two-round output (RD 0 unfitted, RD 20 final).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha, t_max = 0.1, 15.0
    beta = math.log(1 + t_max / alpha) / N_STEPS
    t = [alpha * (math.exp(beta * k) - 1.0) for k in range(N_STEPS)] + [t_max]
    groups = [(1, 4), (25, 2), (1, 4), (1, 6)]
    lam = []
    for (nr, gl), v in zip([g for g in groups for _ in range(g[0])], _lams(scale)):
        lam.extend([v] * gl)
    lines = ["CC\tsynthetic", "MM\tVersion: test"]
    for rd in rounds:
        lk = 0.0 if rd == 0 else -12345.6
        lines += [f"RD\t{rd}", f"LK\t{lk}", f"TR\t{theta}\t{theta / 4}"]
        lines += [f"RS\t{k}\t{t[k]:.6f}\t{lam[k]:.6f}\t0\t0\t0" for k in range(N_STEPS + 1)]
        lines += ["PA\t" + PATTERN, "//"]
    path.write_text("\n".join(lines) + "\n")


# Stand-ins for the two tools, so the runners can be exercised without a
# psmc-rs build or lh3/psmc. Environment variables steer them:
#   FAKE_TOOL_LOG   append one line per call ("<tool> <output name> <n_iter>")
#   FAKE_RHO_RATIO  psmc-rs: theta/rho of the written parameters, whatever
#                   --tr-ratio says [--tr-ratio, else 4]
#   FAKE_PRE_0_2    psmc-rs: reject --tr-ratio, as psmc-rs 0.1 does [unset]
#   FAKE_C_ROUNDS   C psmc: number of rounds written [N_ITER + 1]
#   FAKE_C_EXIT     C psmc: exit status [0]
_FAKE_PSMC_RS = '''#!{python}
import os
import sys
from pathlib import Path

sys.path.insert(0, {fixtures_dir!r})
import fixtures

args = sys.argv[1:]
out, n_iter = Path(args[1]), int(args[2])
if os.environ.get("FAKE_TOOL_LOG"):
    with open(os.environ["FAKE_TOOL_LOG"], "a") as f:
        f.write(f"psmc-rs {{out.name}} {{n_iter}}\\n")
if os.environ.get("FAKE_PRE_0_2") and "--tr-ratio" in args:
    sys.stderr.write("error: unexpected argument '--tr-ratio' found\\n")
    sys.exit(2)
ratio = args[args.index("--tr-ratio") + 1] if "--tr-ratio" in args else "4"
fixtures.write_rust_json(out, rho_ratio=float(os.environ.get("FAKE_RHO_RATIO", ratio)), n_iter=n_iter)
if n_iter > 0:
    print(f"Last EM loglike: {{fixtures.rust_loglike(n_iter)}}")
'''

_FAKE_C_PSMC = '''#!{python}
import os
import sys
from pathlib import Path

sys.path.insert(0, {fixtures_dir!r})
import fixtures

args = sys.argv[1:]
n_iter = next(int(a[2:]) for a in args if a.startswith("-N"))
out = Path(args[args.index("-o") + 1])
if os.environ.get("FAKE_TOOL_LOG"):
    with open(os.environ["FAKE_TOOL_LOG"], "a") as f:
        f.write(f"c {{out.name}} {{n_iter}}\\n")
fixtures.write_c_psmc(out, rounds=range(int(os.environ.get("FAKE_C_ROUNDS", n_iter + 1))))
sys.exit(int(os.environ.get("FAKE_C_EXIT", "0")))
'''


def make_fake_tools(dest: Path) -> Tuple[Path, Path]:
    """Write executable stand-ins for psmc-rs and C psmc into ``dest``; return their paths."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, template in (("psmc-rs", _FAKE_PSMC_RS), ("psmc", _FAKE_C_PSMC)):
        p = dest / name
        p.write_text(template.format(python=sys.executable, fixtures_dir=str(Path(__file__).resolve().parent)))
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        paths.append(p)
    return paths[0], paths[1]
