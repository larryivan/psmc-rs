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

    def _draw(self, module: str, runs: Path = None):
        import importlib

        import matplotlib.pyplot as plt

        from experiment.pipeline.figures import data

        fig = importlib.import_module(f"experiment.pipeline.figures.{module}").draw(
            data.Source(runs or self.runs, config.TINY))
        self.addCleanup(plt.close, fig)
        return fig

    @staticmethod
    def _labels(fig) -> list:
        return [t.get_text() for leg in fig.legends + [ax.get_legend() for ax in fig.axes if ax.get_legend()]
                for t in leg.get_texts()]

    def test_fig5_draws_every_called_bin_taken_as_full_above_the_vcf_without_its_mask(self):
        from experiment.pipeline.figures import style

        fig = self._draw("fig5")
        for ax in fig.axes[3:6]:  # d, e, f
            z = {}
            for ln in ax.lines:
                if ln.get_linestyle() != "None" and ln.get_color() in (style.BLUE_650, style.MAGENTA):
                    z.setdefault(ln.get_color(), set()).add(ln.get_zorder())
            if len(z) == 2:
                self.assertGreater(min(z[style.BLUE_650]), max(z[style.MAGENTA]))

    def test_fig6_legend_names_every_element_and_the_projected_truth_when_given(self):
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        fig = self._draw("fig6")
        labels = self._labels(fig)
        for want in ("True sampling SD and its 95% interval (b, c)", "Covers the fits' mean (d)",
                     "Clopper-Pearson 95% interval", "Interquartile range over the genomes (c)", "Nominal 95% (d)",
                     "window coverage of the true"):
            self.assertTrue(any(want in label for label in labels), want)
        self.assertFalse(any("projected" in label for label in labels))
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            shutil.copytree(self.runs / "E5" / "tables", runs / "E5" / "tables")
            path = SCHEMAS["E5_coverage"].path(runs)
            cov = pd.read_csv(path)
            cov["coverage_projected"] = cov["coverage_fit_mean"]
            cov.to_csv(path, index=False)
            path = SCHEMAS["E5_summary"].path(runs)
            summary = pd.read_csv(path)
            summary["coverage_projected_mean"] = 0.5
            summary.to_csv(path, index=False)
            fig = self._draw("fig6", runs)
        labels = self._labels(fig)
        self.assertIn("Covers the projected truth (d)", labels)
        self.assertTrue(any("projected truth 50%" in label for label in labels))
        trunks = cov["trunk_bins"].nunique()
        self.assertEqual(sum(1 for ln in fig.axes[3].lines if ln.get_marker() not in ("None", "", None)), trunks)

    def test_figs1_ends_both_tools_where_c_psmc_stops_and_shows_its_rounding(self):
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        traces = pd.read_csv(SCHEMAS["E1_traces"].path(self.runs))
        last_c = int(traces[(traces.tool == "c") & traces.loglike.notna()]["iteration"].max())
        fig = self._draw("figs1")
        n = traces["model"].nunique()
        for ax in fig.axes[:n]:
            for ln in ax.lines:
                self.assertLessEqual(max(ln.get_xdata()), last_c)
        self.assertTrue(any("Rounding of C psmc's printed" in label for label in self._labels(fig)))
        for ax in fig.axes[2 * n:3 * n]:
            self.assertEqual(len(ax.collections), 1)  # the rounding band

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
