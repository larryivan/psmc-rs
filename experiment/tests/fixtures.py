"""What the experiment tests share: the repository's paths, stand-in tools, and small psmc-rs / C psmc outputs.

Importing it puts the repository root on ``sys.path`` (for ``experiment.pipeline``). It stays light to
import, as the stand-in tools below import it at every call: the pipeline is imported only where needed.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import importlib.util
import io
import json
import math
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"  # C psmc's output, 5 iterations


def has_modules(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def quiet() -> contextlib.ExitStack:
    """Silence what the code under test prints, on stdout and stderr."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


# Stand-ins for the tools a Context identifies (Context._tools): their identities enter the jobs, but no job runs.
_FAKE_TOOLS = {
    "c": ("c-psmc", "/x/psmc", {"version": "0.6.5", "sha256": "0" * 64}),
    "rs01": ("psmc-rs-0.1", "/x/psmc-rs-0.1", {"version": "psmc-rs 0.1.0", "commit": "e" * 40}),
    "rs02": ("psmc-rs-0.2", "/x/psmc-rs", {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40}),
    "fq2psmcfa": ("fq2psmcfa", "/x/fq2psmcfa", {"sha256": "1" * 64}),
    "generate_multihetsep.py": ("generate_multihetsep.py", "/x/gm.py", {"sha256": "2" * 64}),
}


def fake_tools(*keys: str) -> dict:
    """The stand-in tools of ``keys`` (default: all of them), as a new dict for Context._tools."""
    from experiment.pipeline import tools

    return {k: tools.Tool(name, Path(path), dict(info)) for k, (name, path, info) in _FAKE_TOOLS.items()
            if k in (keys or _FAKE_TOOLS)}


def fake_ctx(profile, runs: Path, *keys: str, **kw):
    """A Context of ``profile`` in ``runs`` with the stand-in tools of ``keys`` (default: all)."""
    from experiment.pipeline.experiments import Context

    return Context(profile, runs, _tools=fake_tools(*keys), **kw)


def option(command: Sequence, name: str) -> Optional[str]:
    """The value ``command`` gives its option ``name`` (None without the option)."""
    command = [str(c) for c in command]
    return command[command.index(name) + 1] if name in command else None


@functools.lru_cache(maxsize=None)
def _c_ref():
    from experiment.pipeline import results

    return results.parse_psmc(C_REF)


def c_ref_json(lam_scale: float = 1.0, **fields) -> dict:
    """psmc-rs's JSON of C psmc's reference fit (its last round), the lambdas times ``lam_scale``, and ``fields``."""
    pa = _c_ref().last().pa
    return {"theta": pa["theta"], "rho": pa["rho"], "t_max": pa["t_max"], "n_steps": 63, "pattern": "4+25*2+4+6",
            "lam": [x * lam_scale for x in pa["lambdas"]], **fields}


def c_ref_loglikes(final: float = -25000.0) -> List[float]:
    """psmc-rs's trace of the reference's 5 iterations: C psmc's log-likelihoods, then ``final`` (of the last model,
    which C psmc never evaluates)."""
    ll = _c_ref().loglikes()
    return [ll[i] for i in range(5)] + [final]


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def _repo_path(path) -> str:
    return Path(path).relative_to(ROOT).as_posix() if Path(path).is_absolute() else str(path)


def git_ignored(paths: Iterable[str]) -> Set[str]:
    """Those of ``paths`` (repository-relative) that git's ignore rules ignore.

    ``--no-index``: by the rules alone, whether or not a file exists or is tracked.
    """
    res = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"], input="\0".join(paths),
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode not in (0, 1):  # 1: none is ignored
        raise RuntimeError(f"git check-ignore failed: {res.stderr}")
    return set(res.stdout.split("\0")) - {""}


def assert_ignore_rules(test, kept: Iterable = (), ignored: Iterable = ()) -> None:
    """Git keeps every path of ``kept`` and ignores every one of ``ignored`` (repository-relative or absolute)."""
    kept, ignored = [_repo_path(p) for p in kept], [_repo_path(p) for p in ignored]
    hit = git_ignored(kept + ignored)
    with test.subTest("kept"):
        test.assertEqual([p for p in kept if p in hit], [], "ignored, but git must keep them")
    with test.subTest("ignored"):
        test.assertEqual([p for p in ignored if p not in hit], [], "kept, but git must ignore them")


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
    path: Path,
    scale: float = 1.0,
    theta: float = 0.002,
    rounds: Sequence[int] = (0, 20),
    *,
    n_iter: Optional[int] = None,
    t_max: float = 15.0,
    tr_ratio: float = 4.0,
    pattern: str = PATTERN,
) -> None:
    """A C psmc output with one block per round in ``rounds``, all with the same curve.

    The default is a two-round output (RD 0 unfitted, RD 20 final). The
    header records ``-N`` (``n_iter``, default the last round), ``-t``,
    ``-r`` and ``-p`` as C psmc does; the curve is always that of ``PATTERN``
    and t_max 15.
    """
    rounds = list(rounds)
    if n_iter is None:
        n_iter = rounds[-1] if rounds else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha, t_end = 0.1, 15.0
    beta = math.log(1 + t_end / alpha) / N_STEPS
    t = [alpha * (math.exp(beta * k) - 1.0) for k in range(N_STEPS)] + [t_end]
    groups = [(1, 4), (25, 2), (1, 4), (1, 6)]
    lam = []
    for (nr, gl), v in zip([g for g in groups for _ in range(g[0])], _lams(scale)):
        lam.extend([v] * gl)
    lines = [
        "CC\tsynthetic",
        "MM\tVersion: test",
        f"MM\tpattern:{pattern}, n:{N_STEPS}, n_free_lambdas:{N_FREE}",
        f"MM\tn_iterations:{n_iter}, skip:1, max_t:{t_max:g}, theta/rho:{tr_ratio:g}",
    ]
    for rd in rounds:
        lk = 0.0 if rd == 0 else -12345.6
        lines += [f"RD\t{rd}", f"LK\t{lk}", f"TR\t{theta}\t{theta / tr_ratio}"]
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
opt = {{a[:2]: a[2:] for a in args if a[:2] in ("-N", "-t", "-r")}}
n_iter = int(opt["-N"])
out = Path(args[args.index("-o") + 1])
if os.environ.get("FAKE_TOOL_LOG"):
    with open(os.environ["FAKE_TOOL_LOG"], "a") as f:
        f.write(f"c {{out.name}} {{n_iter}}\\n")
fixtures.write_c_psmc(
    out,
    rounds=range(int(os.environ.get("FAKE_C_ROUNDS", n_iter + 1))),
    n_iter=n_iter,
    t_max=float(opt.get("-t", 15)),
    tr_ratio=float(opt.get("-r", 4)),
    pattern=args[args.index("-p") + 1] if "-p" in args else "4+5*3+4",
)
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
