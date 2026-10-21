"""Tests for experiment FIG: the figures as runner jobs, and what git keeps of them."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import assert_ignore_rules, fake_ctx, has_modules, in_git_work_tree  # noqa: E402

from experiment.pipeline import config, runner  # noqa: E402
from experiment.pipeline.experiments import REGISTRY, Context, e1, fig, jobs_for  # noqa: E402


@unittest.skipUnless(has_modules("numpy", "pandas", "matplotlib"), "numpy, pandas and matplotlib are needed")
class Jobs(unittest.TestCase):
    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ctx = fake_ctx(config.SMOKE, self.runs, log=lambda _msg: None)

    def ids(self) -> set:
        return {j.id for j in fig.jobs(self.ctx)}

    def test_registered(self):
        self.assertEqual(REGISTRY["FIG"], "fig")

    def test_figures_of_implemented_experiments_only(self):
        from experiment.pipeline import figures

        want = {f"FIG/{fs.name}" for fs in figures.FIGURES
                if all(e in REGISTRY for e in fs.sources) and "E7" not in fs.sources}
        self.assertEqual(self.ids(), want)
        self.assertIn("FIG/fig1_workflow", want)
        self.assertIn("FIG/fig2_simulated_accuracy", want)

    def test_e7s_figures_once_its_data_stage_has_finished(self):
        from experiment.pipeline import kg1000

        e7_figs = {"FIG/fig7_1000_genomes", "FIG/figS5_1000g_populations", "FIG/figS6_1000g_checks"}
        self.assertFalse(e7_figs & self.ids())
        kg1000.write_json(kg1000.layout(self.runs).manifest, {})
        self.assertTrue(e7_figs <= self.ids())

    def test_e7s_figures_from_its_tables_without_its_inputs(self):
        # Committed runs hold E7's tables but not the data stage's inputs (MANIFEST.json among them).
        from experiment.pipeline import figures
        from experiment.pipeline.figures.schemas import SCHEMAS

        self.assertNotIn("FIG/fig7_1000_genomes", self.ids())
        for key in figures.spec("fig7").tables:
            p = SCHEMAS[key].path(self.runs)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n")
        self.assertIn("FIG/fig7_1000_genomes", self.ids())

    def test_a_figure_runs_after_the_tables_it_reads(self):
        g = runner.Graph(e1.jobs(self.ctx) + fig.jobs(self.ctx), self.runs)
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

    def test_a_figure_is_redrawn_when_a_figure_module_it_imports_changes(self):
        from experiment.pipeline import figures

        self.assertIn("figures/fig2.py", figures.spec("figS1").code_files())  # Fig 2's model names
        self.assertIn("figures/fig3.py", figures.spec("figS_turtle_patterns").code_files())
        self.assertNotIn("figures/fig2.py", figures.spec("fig3").code_files())
        self.assertIn("figures/data.py", figures.spec("fig1").code_files())  # render() reads through it

    @unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
    def test_report_draws_what_it_can(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures.schemas import SCHEMAS

        shutil.copytree(tiny_runs.runs_dir() / "E3" / "tables", self.runs / "E3" / "tables")
        # The article's Fig 2 also reads E7's fits (its panel c): with E3's tables alone it waits for them.
        fits = SCHEMAS["E7_fits"]
        self.assertEqual(figures.missing_tables(figures.spec("art2_performance"), self.runs), [fits.path(self.runs)])
        fits.path(self.runs).parent.mkdir(parents=True)
        shutil.copy(fits.path(tiny_runs.runs_dir()), fits.path(self.runs))
        written = fig.report(Context(config.TINY, self.runs, log=lambda _msg: None))
        self.assertEqual(set(written), {"art2_performance", "fig1_workflow", "fig4_performance", "figS4_thread_scaling"})
        self.assertTrue(all(p.exists() for p in written.values()))

    def test_every_figure_has_one_owner(self):
        # No experiment job writes a figure; FIG's outputs do not collide with any other job's.
        jobs = jobs_for(self.ctx, [e for e in REGISTRY if e != "FIG"])
        self.assertFalse([j.id for j in jobs if any("figures" in p.parts for p in j.outputs)])
        runner.Graph(jobs + fig.jobs(self.ctx), self.runs)  # no output is declared twice


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class Layout(unittest.TestCase):
    def test_figures_and_their_records_are_tracked_and_partial_figures_ignored(self):
        assert_ignore_rules(self, kept=(
            "experiment/runs/E1/figures/fig2_simulated_accuracy.pdf",
            "experiment/runs/E1/figures/fig2_simulated_accuracy.svg",
            "experiment/runs/E1/figures/fig2_simulated_accuracy.png",
            "experiment/runs/FIG/figures/fig1_workflow.pdf",
            "experiment/runs/FIG/provenance/fig2_simulated_accuracy.json",
        ), ignored=("experiment/runs/E1/figures/fig2_simulated_accuracy.pdf.partial",))


if __name__ == "__main__":
    unittest.main()
