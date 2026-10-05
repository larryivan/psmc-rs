"""Tests that every manuscript figure draws from tables in the documented schemas."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config  # noqa: E402

HAS_PLOTTING = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "matplotlib"))
DPI = 40  # enough to exercise the PNG writer; the figures are drawn at 600 dpi


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Draw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiment.pipeline.figures import fake

        cls._tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls._tmp.name)
        fake.write_all(cls.runs, config.SMOKE)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_figure_draws(self):
        from experiment.pipeline import figures

        for fs in figures.FIGURES:
            with self.subTest(figure=fs.name):
                self.assertEqual(figures.missing_tables(fs, self.runs), [])
                written = figures.render(fs, self.runs, config.SMOKE, dpi=DPI)
                self.assertEqual(written, fs.outputs(self.runs))
                for p in written:
                    self.assertGreater(p.stat().st_size, 1000, p)

    def test_figures_are_written_where_their_experiment_is(self):
        from experiment.pipeline import figures

        fs = figures.spec("fig2")
        self.assertEqual(fs.name, "fig2_simulated_accuracy")
        self.assertEqual(fs.outputs(self.runs)[0], self.runs / "E1" / "figures" / "fig2_simulated_accuracy.pdf")
        self.assertEqual(figures.spec("fig1").sources, ())
        self.assertEqual(figures.spec("figS4").sources, ("E3",))
        with self.assertRaises(KeyError):
            figures.spec("fig9")

    def test_every_table_a_figure_reads_has_a_schema(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures.schemas import SCHEMAS

        used = {k for fs in figures.FIGURES for k in fs.tables}
        self.assertTrue(used <= set(SCHEMAS))
        self.assertEqual(used, set(SCHEMAS))  # and no schema is unused

    def test_drawing_twice_gives_the_same_bytes(self):
        from experiment.pipeline import figures

        fs = figures.spec("fig5")
        with tempfile.TemporaryDirectory() as d:
            a = [p.read_bytes() for p in figures.render(fs, self.runs, config.SMOKE, dpi=DPI, out_dir=Path(d) / "a")]
            b = [p.read_bytes() for p in figures.render(fs, self.runs, config.SMOKE, dpi=DPI, out_dir=Path(d) / "b")]
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
                self.assertEqual(main(["fig3", "--profile", "smoke", "--runs-dir", empty, "--dpi", str(DPI)]), 1)
            self.assertIn("skip fig3_real_genomes", err.getvalue())

    def test_a_truth_table_that_disagrees_with_the_models_is_refused(self):
        import pandas as pd

        from experiment.pipeline import figures

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for key in ("E1_curves", "E1_table1", "E1_replicates", "E1_rs_vs_c"):
                src = figures.SCHEMAS[key].path(self.runs)
                dst = figures.SCHEMAS[key].path(runs)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
            truth = pd.read_csv(figures.SCHEMAS["E1_truth"].path(self.runs))
            truth["ne"] *= 2.0
            truth.to_csv(figures.SCHEMAS["E1_truth"].path(runs), index=False)
            with self.assertRaises(ValueError):
                figures.render(figures.spec("fig2"), runs, config.SMOKE, dpi=DPI)


if __name__ == "__main__":
    unittest.main()
