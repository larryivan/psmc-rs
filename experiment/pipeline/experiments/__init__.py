"""The experiments, as runner jobs.

Each experiment is a module with ``jobs(ctx) -> list[Job]`` and, when it
writes tables, ``report(ctx)``. :class:`Context` gives them the profile, the
runs directory and the tools.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from .. import config, paths, tools
from ..runner import Job

# Experiment name -> module (in this package).
REGISTRY: Dict[str, str] = {"E1": "e1", "E2": "e2", "E3": "e3", "E4": "e4", "E5": "e5", "E6": "e6", "E7": "e7", "FIG": "fig"}


@dataclass
class Context:
    profile: config.Profile
    runs_dir: Path
    build_tools: bool = False  # build psmc-rs 0.2 first (the daemon does; status only identifies)
    log: Callable[[str], None] = print
    _tools: Dict[str, tools.Tool] = field(default_factory=dict)  # identified once

    def exp_dir(self, experiment: str) -> Path:
        return Path(self.runs_dir) / experiment

    def _tool(self, key: str, identify: Callable[[], tools.Tool]) -> tools.Tool:
        if key not in self._tools:
            self._tools[key] = identify()
        return self._tools[key]

    def rs02(self) -> tools.Tool:
        return self._tool("rs02", lambda: tools.psmc_rs_02(build=self.build_tools, log=self.log))

    def c_psmc(self) -> tools.Tool:
        return self._tool("c", tools.c_psmc)

    def utility(self, name: str) -> tools.Tool:
        """A utility of the tools directory (``fq2psmcfa``, ``generate_multihetsep.py``)."""
        return self._tool(name, lambda: tools.utility(name))

    def pipeline_cmd(self, *args: str) -> List[str]:
        """``python -m experiment.pipeline ARGS``, with the interpreter that runs the pipeline."""
        return [paths.python(), "-m", "experiment.pipeline", *args]

    def profile_args(self) -> List[str]:
        return ["--profile", self.profile.name, "--runs-dir", str(self.runs_dir)]


def available() -> List[str]:
    return list(REGISTRY)


def module(name: str):
    try:
        return importlib.import_module(f".{REGISTRY[name]}", __name__)
    except KeyError:
        raise KeyError(f"unknown experiment {name!r}; implemented: {', '.join(REGISTRY)}") from None


def jobs_for(ctx: Context, experiments: Sequence[str]) -> List[Job]:
    return [job for name in experiments for job in module(name).jobs(ctx)]
