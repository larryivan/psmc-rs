"""Every experiment at the tiny profile, run once through the runner and shared by the tests.

E7's data stage (kg1000, outside the runner) runs first, on its synthetic
cohort. The end-to-end tests of E1-E6 and the figure tests read the tables these
runs write, so the figures are drawn from what the experiments really
write (not from stand-ins), and each experiment runs once per test process
instead of once per test module. The runs directory is temporary and is
removed when the process exits.
"""

from __future__ import annotations

import atexit
import functools
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, jobs_for  # noqa: E402

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")
MODULES = ("msprime", "psutil", "numpy", "pandas", "scipy", "matplotlib")


def missing() -> str:
    """What the tiny runs lack on this machine ('' when everything is there)."""
    absent = [m for m in MODULES if importlib.util.find_spec(m) is None]
    try:
        tools.c_psmc()
        tools.psmc_rs_01(build=False)
        for name in ("fq2psmcfa", "generate_multihetsep.py"):
            tools.utility(name)
    except (tools.ToolError, OSError) as exc:
        absent.append(str(exc))
    if not tools.rs02_binary().exists():
        absent.append("target/release/psmc-rs (cargo build --release)")
    for prog in ("bgzip", "tabix", "bcftools"):
        if shutil.which(prog) is None:
            absent.append(prog)
    if shutil.which("bcftools") is not None:
        from experiment.pipeline import kg1000

        if not kg1000.split_plugin_ok():
            absent.append("bcftools +split (E7)")
    return "; ".join(absent)


def e7_data(runs: Path) -> None:
    """E7's data stage, which runs outside the runner: the synthetic cohort, then every step (the pilot on chr21)."""
    from experiment.pipeline import kg1000

    cfg = config.TINY.e7
    lay = kg1000.layout(runs)
    kg1000.make_fixture(cfg, kg1000.local_root(cfg, lay), log=lambda msg: None)
    kg1000.run_all(cfg, lay, kg1000.temp_dirs(lay), jobs=4, pilot_chroms=cfg.chromosomes[:1], log=lambda msg: None)


@functools.lru_cache(maxsize=None)
def runs_dir() -> Path:
    """The runs directory of the tiny profile with every experiment done (run on first use)."""
    tmp = Path(tempfile.mkdtemp(prefix="psmc-exp-tiny-")).resolve()
    atexit.register(shutil.rmtree, tmp, True)
    e7_data(tmp)
    ctx = Context(config.TINY, tmp, log=lambda msg: None)
    graph = runner.Graph(jobs_for(ctx, EXPERIMENTS), tmp)
    daemon = runner.Daemon(graph, tmp, config.Budget(cores=4, mem_gb=8), runner.Logger(tmp), {"profile": "tiny"})
    if daemon.run() != 0:
        bad = [(jid, daemon.reason.get(jid, "")) for jid, s in daemon.state.items() if s in (runner.FAILED, runner.BLOCKED)]
        logs = []
        for jid, why in bad:
            p = graph.jobs[jid].log_path(tmp)
            logs.append(f"{jid}: {why}\n" + (p.read_text()[-2000:] if p.exists() else ""))
        raise RuntimeError("the tiny runs failed:\n" + "\n".join(logs))
    return tmp


def graph() -> runner.Graph:
    """The job graph of :func:`runs_dir` (for status checks)."""
    runs = runs_dir()
    return runner.Graph(jobs_for(Context(config.TINY, runs, log=lambda msg: None), EXPERIMENTS), runs)


REASON = missing()
SKIP = f"the tiny runs need {REASON}" if REASON else ""

__all__ = ["EXPERIMENTS", "REASON", "SKIP", "e7_data", "graph", "missing", "runs_dir"]
