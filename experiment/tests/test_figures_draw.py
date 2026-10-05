"""Every manuscript figure, drawn from the tables the experiments write (their tiny profile, run once)."""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tiny_runs  # noqa: E402

from experiment.pipeline import config  # noqa: E402

DPI = 40  # enough to exercise the PNG writer; the figures are drawn at 600 dpi


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class Draw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = tiny_runs.runs_dir()
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_schema_is_a_table_its_experiment_writes(self):
        from experiment.pipeline.figures import data
        from experiment.pipeline.figures.schemas import SCHEMAS

        for key in SCHEMAS:
            with self.subTest(table=key):
                df = data.read(self.runs, key)  # raises SchemaError naming any missing column
                self.assertGreater(len(df), 0, key)

    def test_every_figure_draws_from_the_experiments_tables(self):
        from experiment.pipeline import figures

        for fs in figures.FIGURES:
            with self.subTest(figure=fs.name):
                self.assertEqual(figures.missing_tables(fs, self.runs), [])
                written = figures.render(fs, self.runs, config.TINY, dpi=DPI, out_dir=self.out / fs.experiment)
                self.assertEqual([p.name for p in written], [p.name for p in fs.outputs(self.runs)])
                for p in written:
                    self.assertGreater(p.stat().st_size, 1000, p)

    def test_figures_are_written_where_their_experiment_is(self):
        from experiment.pipeline import figures

        fs = figures.spec("fig2")
        self.assertEqual(fs.name, "fig2_simulated_accuracy")
        self.assertEqual(fs.outputs(self.runs)[0], self.runs / "E1" / "figures" / "fig2_simulated_accuracy.pdf")
        self.assertEqual(figures.spec("fig1").sources, ())
        self.assertEqual(figures.spec("figS4").sources, ("E3",))
        self.assertEqual(figures.spec("fig5").sources, ("E4",))
        with self.assertRaises(KeyError):
            figures.spec("fig9")

    def test_every_table_a_figure_reads_has_a_schema(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures.schemas import SCHEMAS

        used = {k for fs in figures.FIGURES for k in fs.tables}
        self.assertEqual(used, set(SCHEMAS))  # and no schema is unused

    def test_drawing_twice_gives_the_same_bytes(self):
        from experiment.pipeline import figures

        fs = figures.spec("fig5")
        a = [p.read_bytes() for p in figures.render(fs, self.runs, config.TINY, dpi=DPI, out_dir=self.out / "a")]
        b = [p.read_bytes() for p in figures.render(fs, self.runs, config.TINY, dpi=DPI, out_dir=self.out / "b")]
        self.assertEqual(a, b)

    def test_cli_lists_and_skips_figures_without_tables(self):
        from experiment.pipeline.figures.__main__ import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--list", "--runs-dir", str(self.runs)]), 0)
        self.assertIn("fig6_bootstrap_calibration", out.getvalue())
        with tempfile.TemporaryDirectory() as empty:
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertEqual(main(["fig3", "--profile", "tiny", "--runs-dir", empty, "--dpi", str(DPI)]), 1)
            self.assertIn("skip fig3_real_genomes", err.getvalue())

    def test_a_truth_table_that_disagrees_with_the_models_is_refused(self):
        import pandas as pd

        from experiment.pipeline import figures

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            shutil.copytree(self.runs / "E1" / "tables", runs / "E1" / "tables")
            truth = pd.read_csv(figures.SCHEMAS["E1_truth"].path(runs))
            truth["ne"] *= 2.0
            truth.to_csv(figures.SCHEMAS["E1_truth"].path(runs), index=False)
            with self.assertRaises(ValueError):
                figures.render(figures.spec("fig2"), runs, config.TINY, dpi=DPI)


if __name__ == "__main__":
    unittest.main()
