"""Where the repository, the tools and the runs are.

Everything is located from this file, so the pipeline behaves the same from
any working directory. Two locations can be overridden from the environment:

* ``PSMC_EXP_TOOLS``: the tools directory (C psmc and its utilities,
  vcfutils.pl, msmc-tools, and the psmc-rs 0.1 build). Default
  ``experiment/.tools`` of this checkout, or, in a git worktree that has none,
  of the main checkout, so that worktrees share one set of tools.
* ``PSMC_EXP_RUNS``: the runs directory of the production profile (default
  ``experiment/runs``). The smoke profile writes to ``experiment/.runs-smoke``.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiment"
REAL_DATA = EXPERIMENT / "real_data"
RUNS = EXPERIMENT / "runs"
SMOKE_RUNS = EXPERIMENT / ".runs-smoke"


@functools.lru_cache(maxsize=None)
def main_checkout() -> Optional[Path]:
    """The main working tree of this repository when ROOT is a linked worktree, else None."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    main = Path(res.stdout.strip()).parent
    return None if main.resolve() == ROOT.resolve() else main


def tools_dir() -> Path:
    """The tools directory: ``$PSMC_EXP_TOOLS``, else ``experiment/.tools`` (see module doc)."""
    env = os.environ.get("PSMC_EXP_TOOLS")
    if env:
        return Path(env).expanduser().resolve()
    own = EXPERIMENT / ".tools"
    if own.exists():
        return own
    main = main_checkout()
    if main is not None and (main / "experiment" / ".tools").exists():
        return main / "experiment" / ".tools"
    return own


def default_runs_dir(profile: str) -> Path:
    """The runs directory of a profile: ``experiment/runs`` for production."""
    if profile == "production":
        env = os.environ.get("PSMC_EXP_RUNS")
        return Path(env).expanduser().resolve() if env else RUNS
    return SMOKE_RUNS if profile == "smoke" else SMOKE_RUNS / profile


def python() -> str:
    """The interpreter the jobs run with: the one running the pipeline."""
    return sys.executable


def placeholders(runs_dir: Path) -> Dict[str, str]:
    """Machine-specific path prefixes and the names that replace them in provenance records.

    Longest prefixes first, so that a runs directory inside the repository is
    written as ``<runs>``, not ``<repo>/experiment/runs``.
    """
    pairs = {
        str(Path(runs_dir).resolve()): "<runs>",
        str(tools_dir()): "<tools>",
        str(ROOT): "<repo>",
        python(): "<python>",
    }
    return dict(sorted(pairs.items(), key=lambda kv: -len(kv[0])))


def portable(text: str, runs_dir: Path) -> str:
    """``text`` with the machine-specific prefixes of :func:`placeholders` replaced."""
    for prefix, name in placeholders(runs_dir).items():
        text = text.replace(prefix, name)
    return text


def repo_relative(path: Path) -> str:
    """``path`` relative to the repository root when inside it, else absolute."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)
