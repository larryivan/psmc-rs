"""Tests for experiment FIG: the figures as runner jobs, and what git keeps of them."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, runner, tools  # noqa: E402
from experiment.pipeline.experiments import REGISTRY, Context, e1, fig, jobs_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

HAS_PLOTTING = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "matplotlib"))
FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})


def fake_ctx(runs: Path) -> Context:
    return Context(config.SMOKE, runs, _tools={"c": FAKE_C, "rs02": FAKE_RS})


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        self.ctx = fake_ctx(self.runs)

    def test_registered(self):
        self.assertEqual(REGISTRY["FIG"], "fig")

    def test_figures_of_implemented_experiments_only(self):
        from experiment.pipeline import figures

        ids = {j.id for j in fig.jobs(self.ctx)}
        want = {f"FIG/{fs.name}" for fs in figures.FIGURES if all(e in REGISTRY for e in fs.sources)}
        self.assertEqual(ids, want)
        self.assertIn("FIG/fig1_workflow", ids)
        self.assertIn("FIG/fig2_simulated_accuracy", ids)

    def test_a_figure_runs_after_the_tables_it_reads(self):
        jobs = e1.jobs(self.ctx) + fig.jobs(self.ctx)
        g = runner.Graph(jobs, self.runs)
        self.assertEqual(g.deps["FIG/fig2_simulated_accuracy"], ["E1/report"])
        self.assertEqual(g.deps["FIG/figS1_em_convergence"], ["E1/report"])
        self.assertEqual(g.deps["FIG/fig1_workflow"], [])

    def test_job_declaration(self):
        job = {j.id: j for j in fig.jobs(self.ctx)}["FIG/fig2_simulated_accuracy"]
        self.assertEqual(job.command[1:4], ["-m", "experiment.pipeline.figures", "fig2_simulated_accuracy"])
        self.assertEqual(job.command[-4:], ["--profile", "smoke", "--runs-dir", str(self.runs)])
        self.assertEqual([p.suffix for p in job.outputs], [".pdf", ".svg", ".png"])
        self.assertTrue(all(p.parent == self.runs / "E1" / "figures" for p in job.outputs))
        self.assertIn("E1", job.settings["experiments"])
        self.assertIn("code", job.tools["pipeline"])
        self.assertIn("matplotlib", job.tools["pipeline"])
        self.assertIn("sha256", job.tools["font"])
        self.assertFalse(job.exclusive)

    @unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
    def test_report_draws_what_it_can(self):
        shutil.copytree(tiny_runs.runs_dir() / "E3" / "tables", self.runs / "E3" / "tables")
        ctx = Context(config.TINY, self.runs, log=lambda _msg: None)
        written = fig.report(ctx)
        self.assertEqual(set(written), {"fig1_workflow", "fig4_performance", "figS4_thread_scaling"})
        self.assertTrue(all(p.exists() for p in written.values()))

    def test_every_figure_has_one_owner(self):
        # No experiment job writes a figure; FIG's outputs do not collide with any other job's.
        ctx = Context(config.SMOKE, self.runs, _tools={"c": FAKE_C, "rs02": FAKE_RS, "rs01": FAKE_RS,
                                                     "fq2psmcfa": FAKE_C, "generate_multihetsep.py": FAKE_C},
                      log=lambda _msg: None)
        jobs = jobs_for(ctx, [e for e in REGISTRY if e != "FIG"])
        self.assertFalse([j.id for j in jobs if any("figures" in p.parts for p in j.outputs)])
        runner.Graph(jobs + fig.jobs(ctx), self.runs)  # no output is declared twice


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class Layout(unittest.TestCase):
    def ignored(self, rel: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 0

    def test_figures_and_their_records_are_tracked(self):
        for p in ("experiment/runs/E1/figures/fig2_simulated_accuracy.pdf",
                  "experiment/runs/E1/figures/fig2_simulated_accuracy.svg",
                  "experiment/runs/E1/figures/fig2_simulated_accuracy.png",
                  "experiment/runs/FIG/figures/fig1_workflow.pdf",
                  "experiment/runs/FIG/provenance/fig2_simulated_accuracy.json"):
            self.assertFalse(self.ignored(p), p)

    def test_partial_figures_are_ignored(self):
        self.assertTrue(self.ignored("experiment/runs/E1/figures/fig2_simulated_accuracy.pdf.partial"))


if __name__ == "__main__":
    unittest.main()
