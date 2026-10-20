"""What the experiment tests share: the repository's paths, stand-in tools, and small psmc-rs / C psmc outputs.

Importing it puts the repository root on ``sys.path`` (for ``experiment.pipeline``). It stays light to
import: the pipeline is imported only where needed.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import importlib.util
import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

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
    ref = _c_ref()
    pa = ref.last().pa
    return {"theta": pa["theta"], "rho": pa["rho"], "t_max": pa["t_max"], "n_steps": ref.header["n"],
            "pattern": ref.header["pattern"], "lam": [x * lam_scale for x in pa["lambdas"]], **fields}


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


def assert_ignore_rules(test, kept: Iterable = (), ignored: Iterable = ()) -> None:
    """Git keeps every path of ``kept`` and ignores every one of ``ignored`` (repository-relative or absolute).

    One ``git check-ignore --no-index`` for all: by the rules alone, whether or not a file exists or is tracked.
    """
    kept, ignored = [_repo_path(p) for p in kept], [_repo_path(p) for p in ignored]
    res = subprocess.run(["git", "check-ignore", "--no-index", "--stdin", "-z"], input="\0".join(kept + ignored),
                         cwd=ROOT, capture_output=True, text=True)
    test.assertIn(res.returncode, (0, 1), res.stderr)  # 1: none is ignored
    hit = set(res.stdout.split("\0"))
    with test.subTest("kept"):
        test.assertEqual([p for p in kept if p in hit], [], "ignored, but git must keep them")
    with test.subTest("ignored"):
        test.assertEqual([p for p in ignored if p not in hit], [], "kept, but git must ignore them")
