"""Every manuscript figure, drawn from the tables the experiments write (their tiny profile, run once)."""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

from experiment.pipeline import config  # noqa: E402

DPI = 40  # enough to exercise the PNG writer; the figures are drawn at 600 dpi
# The figures of the paper: the article's Figs 1-3 and the supplement's Figs S1-S3, in that order.
ARTICLE_FIGURES = ("art1_agreement", "art2_performance", "fig7_1000_genomes", "figS1_em_convergence",
                   "fig5_input_preparation", "fig6_bootstrap_calibration")


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class Draw(unittest.TestCase):
    """The figures of the tiny runs' tables are drawn once and written once, and the tests share them: none
    changes a figure it is given."""

    @classmethod
    def setUpClass(cls):
        import matplotlib.pyplot as plt

        cls.runs = tiny_runs.runs_dir()
        cls.out = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        cls.drawn, cls.written = {}, {}
        cls.addClassCleanup(lambda: [plt.close(fig) for fig in cls.drawn.values()])

    def _draw(self, module: str, runs: Path = None):
        """The figure ``module`` draws from the tables of ``runs`` (default: the tiny runs', drawn once)."""
        import importlib

        import matplotlib.pyplot as plt

        from experiment.pipeline.figures import data

        if runs is None and module in self.drawn:
            return self.drawn[module]
        fig = importlib.import_module(f"experiment.pipeline.figures.{module}").draw(
            data.Source(runs or self.runs, config.TINY))
        if runs is None:
            self.drawn[module] = fig
        else:
            self.addCleanup(plt.close, fig)
        return fig

    def _render(self, fs) -> list:
        """The files of figure ``fs``, written once from the tiny runs' tables."""
        from experiment.pipeline import figures

        if fs.name not in self.written:
            self.written[fs.name] = figures.render(fs, self.runs, config.TINY, dpi=DPI,
                                                   out_dir=self.out / fs.experiment)
        return self.written[fs.name]

    def _table(self, key: str):
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        return pd.read_csv(SCHEMAS[key].path(self.runs))

    def _runs_with(self, experiment: str, **edits) -> Path:
        """A runs directory with a copy of ``experiment``'s tables; ``edits`` maps a table to a function of it."""
        import pandas as pd

        from experiment.pipeline.figures.schemas import SCHEMAS

        runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copytree(self.runs / experiment / "tables", runs / experiment / "tables")
        for key, edit in edits.items():
            path = SCHEMAS[key].path(runs)
            edit(pd.read_csv(path)).to_csv(path, index=False)
        return runs

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

    def test_every_schema_is_a_table_its_experiment_writes(self):
        from experiment.pipeline.figures import data
        from experiment.pipeline.figures.schemas import SCHEMAS

        for key in SCHEMAS:
            with self.subTest(table=key):
                df = data.read(self.runs, key)  # raises SchemaError naming any missing column
                self.assertGreater(len(df), 0, key)

    def test_every_figure_draws_from_the_experiments_tables(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures import style

        for fs in figures.FIGURES:
            with self.subTest(figure=fs.name):
                self.assertEqual(figures.missing_tables(fs, self.runs), [])
                written = self._render(fs)
                self.assertEqual([p.name for p in written], [p.name for p in fs.outputs(self.runs)])
                for p in written:
                    self.assertGreater(p.stat().st_size, 1000, p)
                svg = next(p for p in written if p.suffix == ".svg")
                self.assertEqual(style.small_text(svg), [], "text below 5 pt")

    def test_figures_are_written_where_their_experiment_is(self):
        from experiment.pipeline import figures

        fs = figures.spec("fig2")
        self.assertEqual(fs.name, "fig2_simulated_accuracy")
        self.assertEqual(fs.outputs(self.runs)[0], self.runs / "E1" / "figures" / "fig2_simulated_accuracy.pdf")
        self.assertEqual(figures.spec("fig1").sources, ())
        self.assertEqual(figures.spec("figS4").sources, ("E3",))
        self.assertEqual(figures.spec("fig5").sources, ("E4",))
        # The article's Fig 2 times the 1000 Genomes fits too (its panel c), so E7's tables redraw it.
        self.assertEqual(figures.spec("art2").sources, ("E3", "E7"))
        with self.assertRaises(KeyError):
            figures.spec("fig9")

    def test_every_table_a_figure_reads_has_a_schema(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures.schemas import SCHEMAS

        used = {k for fs in figures.FIGURES for k in fs.tables}
        self.assertEqual(used, set(SCHEMAS))  # and no schema is unused

    def test_drawing_twice_gives_the_same_bytes(self):
        from experiment.pipeline import figures

        for name in ARTICLE_FIGURES:
            with self.subTest(figure=name):
                fs = figures.spec(name)
                again = figures.render(fs, self.runs, config.TINY, dpi=DPI, out_dir=self.out / "again" / name)
                self.assertEqual([p.read_bytes() for p in self._render(fs)], [p.read_bytes() for p in again])

    def test_the_article_figures_name_no_psmc_rs_version(self):
        from matplotlib.text import Text

        from experiment.pipeline import figures

        for name in ARTICLE_FIGURES:
            fs = figures.spec(name)
            with self.subTest(figure=name):
                self.assertNotIn("psmc-rs 0.", fs.title)
                for t in self._draw(fs.module).findobj(Text):
                    self.assertNotIn("psmc-rs 0.", t.get_text())

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
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(["fig9"])
        self.assertEqual(cm.exception.code, 2)  # a usage error naming the figures there are, not a traceback
        self.assertIn("unknown figure 'fig9'; known: art1_agreement", err.getvalue())

    def test_cli_draws_figures_at_once_to_the_same_bytes(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures.__main__ import main

        drawn = [p for name in ("fig1", "fig4") for p in self._render(figures.spec(name))]
        out = Path(self.enterContext(tempfile.TemporaryDirectory()))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(main(["fig1", "fig4", "-j", "2", "--profile", "tiny", "--runs-dir", str(self.runs),
                                   "--dpi", str(DPI), "--out-dir", str(out)]), 0)
        self.assertEqual(printed.getvalue().split(), [str(out / p.name) for p in drawn])  # in order
        for p in drawn:
            self.assertEqual((out / p.name).read_bytes(), p.read_bytes(), p.name)

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

        # Letters anchored at the same x within this many mm. Fig 7 (the article's Fig 3) aligns the letters of a
        # column by their ink, so b's and e's anchors differ by 0.10 mm and c's and f's by 0.06 mm.
        tolerance = {"fig7": 0.2}
        for module in ("fig2", "fig5", "fig6", "fig7", "figs1", "figs2", "figs3", "figs_turtle"):
            with self.subTest(figure=module):
                fig = self._draw(module)
                fig.canvas.draw()
                mm = fig.dpi / 25.4  # pixels per mm
                tol = tolerance.get(module, 0.005) * mm
                letters = [t for t in fig.texts if t.get_fontsize() == style.PANEL_PT and t.get_fontweight() == "bold"]
                boxes = [ax.get_window_extent() for ax in fig.axes if ax.axison]
                columns = {}  # the left edge of the panel each letter names: its letters' x
                for t in letters:
                    x, y = t.get_transform().transform(t.get_position())
                    # A letter names the first panel right of it in the row of panels just below it (a panel
                    # whose letter it shares, as Fig 5 b's second facet, names none).
                    top = max(b.y1 for b in boxes if b.y1 <= y + 0.5)
                    panel = min((b for b in boxes if abs(b.y1 - top) < 0.5 and b.x0 >= x - 0.5), key=lambda b: b.x0)
                    columns.setdefault(round(panel.x0 / mm, 2), []).append(x)
                for xs in columns.values():  # the letters of a column line up
                    self.assertLessEqual(max(xs) - min(xs), tol, [round(v / mm, 2) for v in xs])
                firsts = sorted(xs[0] for xs in columns.values())  # and each column has its own
                self.assertTrue(all(b - a > tol for a, b in zip(firsts, firsts[1:])), firsts)

    def test_fig3_notes_lie_inside_their_panels_and_have_no_box(self):
        fig = self._draw("fig3")
        self._assert_notes_inside_axes(fig)
        for ax in fig.axes:
            for t in ax.texts:  # a box would cover the grey of the times outside the window
                self.assertIsNone(t.get_bbox_patch())

    def test_fig3_gives_each_genomes_scaling_and_its_source(self):
        from experiment.pipeline.figures import fig3

        texts = [t.get_text() for ax in self._draw("fig3").axes for t in ax.texts]
        for gid in ("NA12878", "HLhydTec1"):  # the tiny profile's genomes
            g = next(g for g in config.REAL_GENOMES if g.sample_id == gid)
            note = next(t for t in texts if t.startswith(fig3.where(g)))
            self.assertIn(f"g = {g.scaling.gen_years:g} y", note)
            self.assertIn(g.scaling_cite, note)
        self.assertTrue(any("Fenner 2005" in t for t in texts))

    def test_turtle_patterns_figure(self):
        from experiment.pipeline.figures import figs_turtle, style

        fig = self._draw("figs_turtle")
        self._assert_notes_inside_axes(fig)
        table = self._table("E2_split_patterns")
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
        # The human's sequences are chromosomes (config.RealGenome.sequence_noun).
        human = next(g for g in config.REAL_GENOMES if g.sample_id == "NA12878")
        row = SimpleNamespace(trunk_mb=50.0, trunks=59, coarse_band="False")
        self.assertEqual(fig3.trunk_text(row, 22, human.sequence_noun), "59 blocks: 22 chromosomes split at 50 Mb")

    def test_fig1_gives_the_fit_its_invocation(self):
        from experiment.pipeline.figures import fig1

        texts = [t.get_text() for t in self._draw("fig1").axes[0].texts]
        self.assertIn("fit", texts)
        self.assertIn("psmc-rs INPUT OUTPUT N_ITER", texts)
        self.assertEqual(fig1.FIT_USAGE, "psmc-rs INPUT OUTPUT N_ITER")

    def test_fig4_names_each_sizes_chromosomes_under_it(self):
        table2 = self._table("E3_table2")
        fig = self._draw("fig4")
        fig.canvas.draw()
        for ax in fig.axes:
            labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            want = [f"{int(s):,}\n{int(n)} chr." for s, n in
                    sorted(table2.groupby("size_mb")["chromosomes"].first().items())]
            self.assertEqual(labels, want)

    def test_figs4_legend_draws_what_is_drawn_and_d_fits_the_figure(self):
        def check(runs=None):
            fig = self._draw("figs4", runs)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            right = fig.bbox.x1
            for t in fig.axes[3].texts:  # d's table and its notes lie inside the figure
                self.assertLessEqual(t.get_window_extent(renderer).x1, right, t.get_text())
            heads = [t.get_text() for t in fig.axes[3].texts]
            self.assertIn("Outputs", heads)
            self.assertFalse(any("\n" in h for h in heads[:4]))  # one-line title and headers
            # b: no label at the fewest threads, whose efficiency is 100% by construction.
            ax_e = fig.axes[1]
            first = min(float(x) for x in ax_e.lines[0].get_xdata())
            self.assertFalse(any(abs(a.xy[0] - first) < 1e-9 for a in ax_e.texts if hasattr(a, "xy")
                                 and a.get_text().endswith("%")))
            return self._labels(fig)

        table2 = self._table("E3_table2")
        labels = check()
        self.assertEqual(any("capped" in label for label in labels), (table2.tool == "rs02-default").any())
        uncapped = self._runs_with("E3", **{key: (lambda t: t[t.tool != "rs02-default"])
                                            for key in ("E3_table2", "E3_runs", "E3_determinism")})
        self.assertEqual(check(uncapped), ["psmc-rs 0.2"])

    def test_fig3_legend_names_a_block_size_only_beside_another(self):
        boots = self._table("E2_bootstrap")
        labels = self._labels(self._draw("fig3"))
        fine = (boots["default_trunks"].astype(str) == "False").any()
        self.assertEqual("Bootstrap 95% band" in labels, not fine)
        self.assertEqual(any(label.endswith(" Mb blocks") for label in labels), fine)

    def test_fig5_draws_the_weighted_bins_above_the_bins_taken_as_full(self):
        from matplotlib.colors import to_hex

        from experiment.pipeline.figures import style

        fig = self._draw("fig5")
        self.assertEqual(len(fig.axes), 3)  # a, and b's two facets
        # Only the two routes of the paper's colours are drawn: never the dark blue of the same bins read from
        # the VCF, multihetsep's ochre or the raspberry of a VCF without its mask.
        absent = {style.ROUTES[r].color for r in ("mask_ignore", "multihetsep", "nomask")}
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for i, ax in enumerate(fig.axes):
            with self.subTest(axes=i):
                z = {}
                for ln in ax.lines:
                    self.assertIn(ln.get_linestyle(), ("-", "None"))  # colour alone carries a route: no dashes
                    colour = to_hex(ln.get_color())
                    self.assertNotIn(colour, absent)
                    z.setdefault(colour, []).append(ln.get_zorder())
                    if i and ln.get_marker() not in ("None", "", None):  # b's marks lie inside their facet
                        box, ext = ax.get_window_extent(renderer), ln.get_window_extent(renderer)
                        self.assertGreaterEqual(ext.x0, box.x0 - 0.5)
                        self.assertLessEqual(ext.x1, box.x1 + 0.5)
                # psmc-rs's default (blue) above the bins taken as full (orange): lines and medians alike.
                self.assertGreater(max(z[style.BLUE]), max(z[style.ORANGE]))

    def test_fig5_names_the_bins_taken_as_full_for_what_they_are(self):
        from experiment.pipeline.figures import fig5, style

        def texts(fig):
            return [t.get_text() for ax in fig.axes for t in ax.texts]

        # fq2psmcfa's bins, C psmc's input and model; without its rows, the same bins read from the VCF.
        self.assertIn(fig5.FULL, set(self._table("E4_routes").route))
        a = texts(self._draw("fig5"))
        self.assertIn(style.ROUTES["fq2psmcfa"].label, a)
        self.assertIn(style.ROUTES["mask_auto"].label, a)
        without_full = self._runs_with("E4", **{key: (lambda t: t[t.route != fig5.FULL])
                                                for key in ("E4_routes", "E4_curves")})
        b = texts(self._draw("fig5", without_full))
        self.assertIn(style.ROUTES["mask_ignore"].label, b)
        self.assertFalse(any("fq2psmcfa" in t for t in b))

    def test_fig6_draws_one_row_of_three_panels_from_e5s_numbers(self):
        from matplotlib.colors import to_hex
        from matplotlib.text import Text

        from experiment.pipeline.figures import style

        summary, sampling, bands = self._table("E5_summary"), self._table("E5_sampling"), self._table("E5_bands")
        rows = summary[(summary.window == "informative") & summary.trunk_bins.isin(set(bands.trunk_bins))]
        arms = len(rows)
        self.assertGreater(arms, 0)
        fig = self._draw("fig6")
        texts = [t.get_text() for t in fig.findobj(Text)]
        self.assertEqual((len(fig.axes), fig.legends), (3, []))  # no figure legend
        ax_a, ax_b, ax_c = fig.axes
        # a: the truth, one genome's fit, its band and the independent fits' range, as PSMC steps at the grid's
        # times; the band is the panel's only fill.
        labels = [t.get_text() for t in ax_a.get_legend().get_texts()]
        self.assertEqual(len(labels), 4)
        fits = int(sampling.fits.iloc[0])
        self.assertEqual((labels[0], labels[-1]), ("Truth", f"95% range of {fits} independent fits"))
        self.assertEqual(len(ax_a.collections), 1)
        self.assertTrue(all(ln.get_drawstyle() == "steps-mid" for ln in ax_a.lines))
        # b: each block count's median SD ratio, in the key.
        keys = [t.get_text() for t in ax_b.get_legend().get_texts()]
        self.assertEqual(len(keys), arms)
        for row in rows.itertuples():
            self.assertTrue(any(f"{int(row.trunks)} blocks" in k and f"{row.sd_ratio_median:.2f}×" in k for k in keys),
                            (row.trunks, keys))
        # c: per block count a strip of genomes and a mean bar, for the fits' mean and for the truth.
        bars = [ln for ln in ax_c.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
                and ln.get_marker() in ("None", "", None) and to_hex(ln.get_color()) == style.INK]
        want = sorted(100.0 * v for v in list(rows.coverage_of_fit_mean_mean) + list(rows.coverage_mean))
        got = sorted(float(ln.get_ydata()[0]) for ln in bars)
        self.assertEqual(len(got), 2 * arms)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=1e-9)
        dots = [ln for ln in ax_c.lines if ln.get_marker() == "o"]
        self.assertEqual(len(dots), 2 * arms)
        per_arm = bands.groupby("trunk_bins").dataset.nunique()
        self.assertEqual(sorted(len(ln.get_xdata()) for ln in dots),
                         sorted(int(per_arm[t]) for t in rows.trunk_bins for _ in range(2)))
        self.assertIn("Nominal 95%", [t.get_text() for t in ax_c.texts])
        self.assertFalse(any(t == "Median ratio" or "↑" in t for t in texts))

    def test_figs1_ends_both_tools_where_c_psmc_stops(self):
        from matplotlib.collections import LineCollection
        from matplotlib.text import Text

        from experiment.pipeline.figures import figs1
        from experiment.pipeline.figures.fig2 import model_order

        traces = self._table("E1_traces")
        last_c = int(traces[(traces.tool == "c") & traces.loglike.notna()]["iteration"].max())
        fig = self._draw("figs1")
        shown = [ax for ax in fig.axes if ax.axison]
        per = figs1.series(traces)
        models = model_order(per.keys())
        self.assertEqual(len(shown), traces["model"].nunique())  # one panel per model
        for ax, m in zip(shown, models):
            for ln in ax.lines:
                self.assertLessEqual(max(ln.get_xdata()), last_c)
                self.assertIn(ln.get_marker(), ("None", "", None))  # continuous lines carry no markers
            # No band: the y hairlines are the only collection.
            self.assertLessEqual(len(ax.collections), 1)
            self.assertTrue(all(type(c) is LineCollection for c in ax.collections))
            if per[m]["arrow"] is not None:  # the arrow's ratio, floored: a bound
                self.assertIn(per[m]["arrow"]["label"], [t.get_text() for t in ax.texts])
        self.assertEqual(self._labels(fig), ["C psmc", "psmc-rs"])
        self.assertFalse(any("Rounding" in t.get_text() or "psmc-rs 0." in t.get_text() for t in fig.findobj(Text)))

    def test_figs3_draws_coverage_by_posterior_mean_decile_when_given(self):
        from experiment.pipeline.figures import style

        calib = self._table("E6_S3_calibration")
        fig = self._draw("figs3")
        labels = self._labels(fig)
        # Every series of c in a palette colour: grey is for the style keys and the nominal line.
        series = {ln.get_color() for ln in fig.axes[2].lines if ln.get_marker() not in ("None", "", None)}
        self.assertIn(style.BLUE, series)
        self.assertLessEqual(series, {style.BLUE, style.YELLOW})
        self.assertEqual("By decile of the posterior mean" in labels, (calib["by"] == "posterior_mean").any())
        self.assertIn("By decile of the true TMRCA", labels)
        by_truth = self._runs_with("E6", E6_S3_calibration=lambda t: t[t["by"] == "true"])
        labels = self._labels(self._draw("figs3", by_truth))
        self.assertNotIn("By decile of the posterior mean", labels)
        self.assertIn("By decile of the true TMRCA", labels)

    def test_a_truth_table_that_disagrees_with_the_models_is_refused(self):
        from experiment.pipeline import figures

        runs = self._runs_with("E1", E1_truth=lambda t: t.assign(ne=t["ne"] * 2.0))
        with self.assertRaises(ValueError):
            figures.render(figures.spec("fig2"), runs, config.TINY, dpi=DPI)


if __name__ == "__main__":
    unittest.main()
