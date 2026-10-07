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

    def _assert_notes_inside_axes(self, fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i, ax in enumerate(fig.axes):
            box = ax.get_window_extent(renderer)
            for t in ax.texts:
                ext = t.get_window_extent(renderer)
                with self.subTest(axes=i, text=t.get_text()[:40]):
                    self.assertGreaterEqual(ext.x0, box.x0 - 0.5)
                    self.assertLessEqual(ext.x1, box.x1 + 0.5)
                    self.assertGreaterEqual(ext.y0, box.y0 - 0.5)
                    self.assertLessEqual(ext.y1, box.y1 + 0.5)

    def test_the_readme_gives_each_figures_size(self):
        import math
        import re

        from experiment.pipeline import figures

        text = (Path(figures.__file__).parent / "README.md").read_text()
        for fs in figures.FIGURES:
            with self.subTest(figure=fs.name):
                row = next(line for line in text.splitlines() if line.startswith(f"| `{fs.name}`"))
                w, h = (float(x) for x in re.search(r"\| (\d+) x (\d+) mm \|$", row).groups())
                fig = self._draw(fs.module)
                if fs.module == "fig3":  # a row of panels per two genomes: the README gives the four of E2
                    n = len(fig.axes)
                    h += 50.0 * (math.ceil(n / 2) - 2)
                if fs.module == "figs5":  # a row of panels per six populations: the README gives the 26 of E7
                    rows = math.ceil(len([ax for ax in fig.axes if ax.axison]) / 6)
                    h += 30.0 * (rows - 5)
                self.assertAlmostEqual(fig.get_figwidth() * 25.4, w, places=1)
                self.assertAlmostEqual(fig.get_figheight() * 25.4, h, places=1)

    def test_panel_letters_line_up_in_columns(self):
        from experiment.pipeline.figures import style

        for module in ("fig2", "fig5", "fig6", "figs1", "figs2", "figs3", "figs_turtle"):
            with self.subTest(figure=module):
                fig = self._draw(module)
                letters = [t for t in fig.texts if t.get_fontsize() == style.PANEL_PT and t.get_fontweight() == "bold"]
                columns = {round(ax.get_position(original=True).x0, 3) for ax in fig.axes if ax.axison}
                self.assertEqual(len({round(t.get_position()[0], 4) for t in letters}), len(columns))

    def test_fig3_notes_lie_inside_their_panels_and_have_no_box(self):
        fig = self._draw("fig3")
        self._assert_notes_inside_axes(fig)
        for ax in fig.axes:
            for t in ax.texts:  # a box would cover the grey of the times outside the window
                self.assertIsNone(t.get_bbox_patch())

    def test_fig3_gives_each_genomes_scaling_and_its_source(self):
        from experiment.pipeline.figures import fig3

        fig = self._draw("fig3")
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        for gid in ("Papuan", "HLhydTec1"):  # the tiny profile's genomes
            g = next(g for g in config.REAL_GENOMES if g.sample_id == gid)
            note = next(t for t in texts if t.startswith(fig3.where(g)))
            self.assertIn(f"g = {g.scaling.gen_years:g} y", note)
            self.assertIn(g.scaling_cite, note)
        self.assertTrue(any("Fenner 2005" in t for t in texts))

    def test_turtle_patterns_figure(self):
        import pandas as pd

        from experiment.pipeline.figures import figs_turtle, style
        from experiment.pipeline.figures.schemas import SCHEMAS

        fig = self._draw("figs_turtle")
        self._assert_notes_inside_axes(fig)
        table = pd.read_csv(SCHEMAS["E2_split_patterns"].path(self.runs))
        self.assertEqual(len([ax for ax in fig.axes if ax.axison]), table["sample_id"].nunique())
        labels = self._labels(fig)
        for pattern in ("4+25*2+4+6 (Fig 3)", "2+2+25*2+4+6", "1+1+1+1+25*2+4+6", "Outside the comparison window"):
            self.assertTrue(any(pattern in label for label in labels), pattern)
        ax = fig.axes[0]
        # The default in psmc-rs's blue, the split patterns as its dark variant, dashed and dotted.
        drawn = [(ln.get_color(), ln.get_linestyle()) for ln in ax.lines]
        self.assertEqual(sorted(c for c, _ in drawn), sorted([style.BLUE, style.BLUE_650, style.BLUE_650]))
        self.assertTrue(all((ls == "-") == (c == style.BLUE) for c, ls in drawn))
        self.assertEqual([e.linestyle for k, e in figs_turtle.pattern_entities(config.TINY.e2).items() if k != "default"],
                         list(figs_turtle.SPLIT_STYLES))
        note = "\n".join(t.get_text() for t in ax.texts)
        self.assertIn("Recent max / plateau:", note)
        for row in table[table.sample_id == table.sample_id.iloc[0]].itertuples():
            self.assertIn(f"{row.pattern}  {row.recent_max_vs_plateau:.2f}", note)
        cfg = config.PRODUCTION.e2
        self.assertEqual(list(figs_turtle.pattern_entities(cfg)), ["default", "split2", "split1"])

    def test_fig3_names_the_blocks_of_one_sequence_by_their_size(self):
        from types import SimpleNamespace

        from experiment.pipeline.figures import fig3

        row = SimpleNamespace(trunk_mb=50.0, trunks=5, coarse_band="True")
        self.assertEqual(fig3.trunk_text(row, 1), "5 blocks of 50 Mb (few: coarse)")
        row = SimpleNamespace(trunk_mb=50.0, trunks=897, coarse_band="False")
        self.assertEqual(fig3.trunk_text(row, 889), "897 blocks: 889 scaffolds split at 50 Mb")

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

    def test_fig6_legend_names_every_element_and_the_projected_truth(self):
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        # E5's report writes the coverage of the pattern-projected truth, which e draws.
        fig = self._draw("fig6")
        labels = self._labels(fig)
        for want in ("True sampling SD and its 95% interval (b, c)", "Share whose band covers the fits' mean (d)",
                     "Clopper-Pearson 95% interval of the thick",
                     "Interquartile range over the genomes (c)", "Nominal 95% (d, e)", "window coverage of the true",
                     "or the projected truth (e)", " Mb blocks ("):
            self.assertTrue(any(want in label for label in labels), want)
        self.assertFalse(any("trunk" in label or label.startswith("Its ") for label in labels))
        summary = pd.read_csv(SCHEMAS["E5_summary"].path(self.runs))
        for row in summary[summary.window == "informative"].itertuples():
            pct = f"projected truth {100 * row.coverage_projected_mean:.0f}%"
            self.assertTrue(any(pct in label for label in labels), pct)
        # d: per block size the true Ne and the fits' mean; e: the projected truth; both the nominal line.
        trunks = pd.read_csv(SCHEMAS["E5_coverage"].path(self.runs))["trunk_bins"].nunique()
        ax_d, ax_e = fig.axes[3], fig.axes[4]
        self.assertEqual((len(ax_d.lines), len(ax_e.lines)), (2 * trunks + 1, trunks + 1))
        for ax in (ax_d, ax_e):
            self.assertTrue(all(ln.get_marker() in ("None", "", None) for ln in ax.lines))
            self.assertEqual(len(ax.collections), 1)  # the Clopper-Pearson interval of the default blocks

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

    def test_figs3_draws_coverage_by_posterior_mean_decile_when_given(self):
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        from experiment.pipeline.figures import style

        calib = pd.read_csv(SCHEMAS["E6_S3_calibration"].path(self.runs))
        fig = self._draw("figs3")
        labels = self._labels(fig)
        # Every series of c in a palette colour: grey is for the style keys and the nominal line.
        series = {ln.get_color() for ln in fig.axes[2].lines if ln.get_marker() not in ("None", "", None)}
        self.assertIn(style.BLUE, series)
        self.assertLessEqual(series, {style.BLUE, style.YELLOW})
        self.assertEqual("By decile of the posterior mean" in labels, (calib["by"] == "posterior_mean").any())
        self.assertIn("By decile of the true TMRCA", labels)
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            shutil.copytree(self.runs / "E6" / "tables", runs / "E6" / "tables")
            path = SCHEMAS["E6_S3_calibration"].path(runs)
            calib[calib["by"] == "true"].to_csv(path, index=False)
            labels = self._labels(self._draw("figs3", runs))
        self.assertNotIn("By decile of the posterior mean", labels)
        self.assertIn("By decile of the true TMRCA", labels)

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
