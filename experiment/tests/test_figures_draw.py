"""Every manuscript figure, drawn from the tables the experiments write (their tiny profile, run once)."""

from __future__ import annotations

import contextlib
import io
import math
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
            self.assertEqual(main(["fig1", "fig4", "fig1_workflow", "-j", "2", "--profile", "tiny", "--runs-dir",
                                   str(self.runs), "--dpi", str(DPI), "--out-dir", str(out)]), 0)
        self.assertEqual(printed.getvalue().split(), [str(out / p.name) for p in drawn])  # in order, once each
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
        self.assertEqual(any("capped" in label for label in labels), (table2.tool == "rs02-capped").any())
        uncapped = self._runs_with("E3", **{key: (lambda t: t[t.tool != "rs02-capped"])
                                            for key in ("E3_table2", "E3_runs", "E3_determinism")})
        self.assertEqual(check(uncapped), ["psmc-rs 0.2"])

    def test_fig3_legend_names_a_block_size_only_beside_another(self):
        boots = self._table("E2_bootstrap")
        labels = self._labels(self._draw("fig3"))
        fine = (boots["default_trunks"].astype(str) == "False").any()
        self.assertEqual("Bootstrap 95% band" in labels, not fine)
        self.assertEqual(any(label.endswith(" Mb blocks") for label in labels), fine)

    def test_keyed_article_figures_put_their_first_row_under_the_key(self):
        from experiment.pipeline import figures
        from experiment.pipeline.figures import style

        keyed = []
        for name in ARTICLE_FIGURES:
            fig = self._draw(figures.spec(name).module)
            if not fig.legends:  # fig7 keeps its own top, without a key
                continue
            keyed.append(name)
            with self.subTest(figure=name):
                h = fig.get_figheight() * 25.4
                self.assertAlmostEqual(min((1 - ax.get_position().y1) * h for ax in fig.axes if ax.axison),
                                       style.TOP_LEGEND_MM, places=2)
        self.assertEqual(keyed, ["art1_agreement", "figS1_em_convergence"])

    def test_art1_noise_floor_is_tied_to_e1s_fits(self):
        import numpy as np

        from experiment.pipeline.figures import art1_agreement as A
        from experiment.pipeline.figures.schemas import COMMITTED

        side = COMMITTED[A.FLOOR_FILE].read()
        # The production's fits, as the committed file holds them: every genome has its noise floor.
        e1d = side[["model", "replicate"]].assign(rmse_log10_ne_rs_vs_c_per_interval=side.rmse_window_rs_vs_c_ibex)
        floor = A.noise_floor(e1d)
        self.assertEqual(len(floor), len(side))
        self.assertAlmostEqual(float(np.median(list(floor.values()))), float(side.rmse_window_c_laptop_vs_ibex.median()))
        # The tiny profile's fits are not the production's: no row is tied to them, and e draws no such row.
        self.assertEqual(A.noise_floor(self._table("E1_rs_vs_c")), {})
        self.assertFalse([t for t in self._draw("art1_agreement").axes[8].texts
                          if t.get_text().startswith(A.FLOOR_NAME)])
        # A file that holds the Delta of only some of the genomes describes other fits: refused.
        stale = e1d.copy()
        stale.loc[0, "rmse_log10_ne_rs_vs_c_per_interval"] *= 1.001
        with self.assertRaises(ValueError):
            A.noise_floor(stale)

    def test_art1_draws_each_panels_differences_under_it(self):
        import re

        import numpy as np

        from experiment.pipeline.figures import art1_agreement as A

        fig = self._draw("art1_agreement")
        drawn = []
        for i, (ax, st) in enumerate(zip(fig.axes[:4], fig.axes[4:8])):
            self.assertEqual(ax.axison, st.axison)  # a strip under each drawn panel, none under an empty column
            if not st.axison:
                continue
            drawn.append(i)
            with self.subTest(column=i):
                ln = next(ln for ln in st.lines if ln.get_drawstyle() == "steps-post")
                lo, hi = st.get_ylim()
                self.assertEqual(lo, -hi)  # symmetric about the zero line
                self.assertGreaterEqual(hi, A.STRIP_LIM[0 if i < 3 else 1])  # a-c's scale and d's, or wider
                self.assertLessEqual(np.nanmax(np.abs(np.asarray(ln.get_ydata(), dtype=float))), A.STRIP_FILL * hi)
                self.assertIn(hi / 10 ** math.floor(math.log10(hi)), (1.0, 2.0, 5.0))
        self.assertTrue(drawn)
        self.assertEqual(fig.axes[4 + drawn[0]].get_ylabel(), r"$\times10^{-4}$")  # the unit, on the first strip
        # e's rings name the genomes of the drawn panels by their letters, which run on past an empty column.
        rings = [t.get_text() for t in fig.axes[8].texts if re.fullmatch(r"[a-d](, [a-d])*", t.get_text())]
        self.assertEqual(sorted(x for t in rings for x in t.split(", ")), list("abcd"[:len(drawn)]))

    def test_art1_texts_that_share_a_baseline_stay_on_it(self):
        from experiment.pipeline.figures import style

        fig = self._draw("art1_agreement")
        a, sa = fig.axes[0], fig.axes[4]
        labels = (a.yaxis.label, sa.yaxis.label)
        xs = [label.get_transform().transform(label.get_position())[0] for label in labels]
        self.assertAlmostEqual(*xs)  # one column: N_e and the strips' unit
        for label in labels:  # on their baselines in every output (style.on_baseline marks what it puts back)
            self.assertEqual(label.get_verticalalignment(), "baseline")
            self.assertTrue(getattr(label, "_house_on_baseline", False))
        titles = [t for t in fig.texts if t.get_fontsize() == style.TEXT_PT and "$" in t.get_text()]
        self.assertTrue(titles)  # d's binomial, in italics
        for t in titles:
            self.assertTrue(getattr(t, "_house_on_baseline", False), t.get_text())
        # e's headers are plain text (its Delta is U+0394), which every output sets on its baseline.
        self.assertFalse([t.get_text() for t in fig.axes[8].texts
                          if t.get_fontsize() == style.TEXT_PT and "$" in t.get_text()])

    def test_art1_draws_the_truth_without_joins_that_change_nothing(self):
        import numpy as np

        from experiment.pipeline.figures import art1_agreement as A
        from experiment.pipeline.figures.data import truth_xy

        # A vertex that repeats its neighbour makes a miter join of zero length, which strokes as a spike.
        for model in config.MODELS:
            x, y = A._truth_line(model, 25.0, (5e3, 5e6))
            same = np.isclose(x[1:], x[:-1], rtol=1e-9, atol=0) & np.isclose(y[1:], y[:-1], rtol=1e-4, atol=0)
            self.assertFalse(same.any(), model)
        # The zigzag loses its four epoch joins; the bottleneck keeps its four steps.
        self.assertEqual(len(A._truth_line("zigzag", 25.0, (5e3, 5e6))[0]), len(truth_xy("zigzag", 25.0, (5e3, 5e6))[0]) - 4)
        self.assertEqual(int((np.diff(A._truth_line("bottleneck", 25.0, (1e4, 1.3e6))[0]) == 0).sum()), 4)

    def test_art2_titles_every_panel(self):
        titles = [ax.get_title(loc="left") for ax in self._draw("art2_performance").axes]
        self.assertEqual(len(titles), 3)  # the tiny profile writes E7_fits, so c is drawn
        self.assertTrue(titles[0].endswith(", alone on a node"), titles[0])
        self.assertEqual(titles[1:], ["Same genome", "1000 Genomes, one thread each"])

    def test_fig7_draws_each_genome_as_a_halo(self):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_hex

        from experiment.pipeline.figures import style

        fig = self._draw("fig7")
        fits = self._table("E7_fits")
        rs = fits[fits.route == "rs-psmcfa"]
        self.assertFalse([c for c in fig.axes[0].collections if isinstance(c, LineCollection)])  # a has none
        drawn = 0
        for ax, s in zip(fig.axes[1:], config.TINY.e7.superpopulations):
            if not (rs.superpopulation == s).any():
                continue
            with self.subTest(superpopulation=s):
                lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
                self.assertEqual(len(lcs), 1)
                halo = lcs[0]
                # The one alpha of the article figures: rasterized hairlines in a tint of the continent's colour.
                self.assertTrue(halo.get_rasterized())
                self.assertEqual(halo.get_alpha(), style.CLOUD_ALPHA)
                self.assertEqual(halo.get_linewidth()[0], style.LW_HAIR)
                self.assertEqual(to_hex(halo.get_colors()[0]), style.tint(style.SUPERPOP_COLORS[s], style.CLOUD_SHARE))
                # Every genome at most once: one whose first window ends past the cohort's window has no line.
                self.assertGreater(len(halo.get_segments()), 0)
                self.assertLessEqual(len(halo.get_segments()), int((rs.superpopulation == s).sum()))
                drawn += 1
        self.assertGreater(drawn, 0)

    def test_fig7_turns_each_median_full_straight_after_its_split(self):
        import numpy as np
        from matplotlib.colors import to_hex

        from experiment.pipeline.figures import fig7, style

        # The tiny cohort's window starts after every group has left its first window; the production's (10.6
        # kya) starts inside it. Started earlier, as there, it puts every median's split inside the window.
        window_lo = 2e4
        runs = self._runs_with("E7", E7_pop_bands=lambda t: t.assign(
            window_lo_years=t.window_lo_years.where(t.group_kind != "all", window_lo)))
        fig = self._draw("fig7", runs)
        fig.canvas.draw()
        bands = self._table("E7_pop_bands")
        grid = bands[bands.route == fig7.ROUTE].years.unique()
        split_of = fig7.first_window_splits(fig7.genome_fits(self._table("E7_fits"), self._table("E7_curves")), grid)
        self.assertGreater(min(split_of.values()), window_lo)
        splits = set(split_of.values())

        def same_steps(pale, full, split) -> bool:
            """Whether ``pale`` runs on beneath ``full`` past the split as the same steps (steps-post)."""
            fx, fy = np.asarray(full.get_xdata(), dtype=float), np.asarray(full.get_ydata(), dtype=float)
            return all(fy[np.searchsorted(fx, x, side="right") - 1] == y
                       for x, y in zip(pale.get_xdata(), pale.get_ydata()) if x >= split)

        cut = 0
        for i, ax in enumerate(fig.axes):
            xlim = ax.get_xlim()
            dex_mm = math.log10(xlim[1] / xlim[0]) / fig7.DW_MM
            for ln in ax.lines:
                box = ln.get_clip_box()
                if box is None or abs(box.x0 - ax.bbox.x0) < 1e-6:  # clipped by its axes (the leaders too)
                    continue
                cut += 1
                with self.subTest(axes=i, color=ln.get_color()):
                    half = 10 ** (ln.get_linewidth() / style.PT_PER_MM / 2.0 * dex_mm)
                    split = float(ln.get_xdata()[0])
                    self.assertIn(split, splits)
                    # The colour changes half a line's width after the split, straight across the line...
                    x_cut = ax.transData.inverted().transform((box.x0, box.y0))[0]
                    self.assertAlmostEqual(x_cut / (split * half), 1.0, places=6)
                    # ...over its pale first window, from the window's start, just beneath at the same width,
                    # its step at the split included, a line's width further.
                    pale = [p for p in ax.lines if to_hex(p.get_color()) == style.tint(to_hex(ln.get_color()),
                                                                                      style.FIRST_WINDOW_SHARE)
                            and p.get_linewidth() == ln.get_linewidth() and p.get_zorder() == ln.get_zorder() - 0.5
                            and math.isclose(float(p.get_xdata()[-1]), split * half ** 2, rel_tol=1e-6)
                            and same_steps(p, ln, split)]
                    self.assertEqual(len(pale), 1)
                    self.assertAlmostEqual(float(pale[0].get_xdata()[0]), window_lo)
            # The grey context medians start where they turn full in a, or at the window's start.
            for ln in ax.lines:
                if to_hex(ln.get_color()) == style.CONTEXT:
                    x0 = float(ln.get_xdata()[0])
                    self.assertTrue(x0 in splits or np.isclose(x0, window_lo), x0)
        self.assertEqual(cut, len(split_of))  # every median, once: a's superpopulations, b-f's populations

    def test_fig7_refuses_tables_of_different_runs(self):
        # E7_curves without one genome's fits counts one genome fewer than E7_pop_bands.
        runs = self._runs_with("E7", E7_curves=lambda t: t[t.sample_id != sorted(t.sample_id)[0]])
        with self.assertRaises(ValueError):
            self._draw("fig7", runs)

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

    def test_fig5_draws_b_as_a_ratio_dispersed_first_with_its_numbers_first(self):
        from matplotlib.colors import to_hex

        from experiment.pipeline.figures import fig5, style

        fig = self._draw("fig5")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        a, left, right = fig.axes
        # b's facets: the dispersed runs beside a's example, then the clustered ones, titled by what they do.
        self.assertEqual([ax.get_title(loc="left").split(":")[0] for ax in (left, right)], ["Dispersed", "Clustered"])
        # theta as a ratio to the complete-data fit: a grey line at 1, ticks with the step's one decimal.
        ref = [ln for ln in left.lines if to_hex(ln.get_color()) == style.REF]
        self.assertEqual(list(ref[0].get_ydata()), [1.0, 1.0])
        lo, hi = left.get_ylim()
        ticks = [t.get_text() for t in left.get_yticklabels() if lo <= t.get_position()[1] <= hi]
        self.assertTrue(ticks)
        self.assertTrue(all(len(t.split(".")[-1]) == 1 for t in ticks), ticks)
        # Folds number first: the medians of the largest dispersed fraction, of the two routes drawn.
        folds = [t for t in left.texts if t.get_text().endswith(style.TIMES)]
        self.assertFalse([t for ax in fig.axes for t in ax.texts if t.get_text().startswith(style.TIMES)])
        routes = self._table("E4_routes")
        cfg = config.TINY.e4
        top = max((m for m in cfg.masks if m.structure == "dispersed"), key=lambda m: m.missing_fraction)
        want = {r: f"{1 + routes[(routes['mask'] == top.name) & (routes.route == r)].theta_rel_complete.median():.2f}"
                   f"{style.TIMES}" for r in (fig5.full_route(routes), fig5.WEIGHTED)}
        self.assertEqual(sorted(t.get_text() for t in folds), sorted(want.values()))
        # 'Bins taken as full' ends where the weighted fold does, on b's right edge of labels.
        full = next(t for t in left.texts if t.get_text() == fig5.B_NAMES[fig5.FULL])
        blue = next(t for t in folds if t.get_text() == want[fig5.WEIGHTED])
        mm = fig.dpi / 25.4
        self.assertAlmostEqual(full.get_window_extent(renderer).x1 / mm, blue.get_window_extent(renderer).x1 / mm,
                               delta=0.01)
        # The y labels sit on their baselines in the PDF too (style.on_baseline).
        self.assertTrue(a.yaxis.label._house_on_baseline and left.yaxis.label._house_on_baseline)

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

    def test_fig6_draws_two_rows_of_two_panels_from_e5s_numbers(self):
        import numpy as np
        from matplotlib.colors import to_hex
        from matplotlib.text import Text

        from experiment.pipeline.figures import fig6, style

        summary, sampling, bands = self._table("E5_summary"), self._table("E5_sampling"), self._table("E5_bands")
        sd, cov = self._table("E5_sd_ratio"), self._table("E5_coverage")
        rows = summary[(summary.window == "informative") & summary.trunk_bins.isin(set(bands.trunk_bins))]
        arms = len(rows)
        self.assertGreater(arms, 0)
        many = rows.sort_values("trunks").iloc[-1]  # a and c draw the most blocks only
        n = int(many.trunks)
        fig = self._draw("fig6")
        texts = [t.get_text() for t in fig.findobj(Text)]
        self.assertEqual((len(fig.axes), fig.legends), (4, []))  # a over c, b over d; no figure legend
        ax_a, ax_b, ax_c, ax_d = fig.axes
        self.assertEqual([ax.get_title(loc="left") for ax in fig.axes],
                         [f"One genome, {n} blocks", "Spread at each time", f"Coverage at each time, {n} blocks",
                          "Coverage of each genome"])
        # a: the truth, one genome's fit, its band and the independent fits' range, as PSMC steps at the grid's
        # times, each named in place; the band is the panel's only fill, and nothing runs off the panel.
        self.assertIsNone(ax_a.get_legend())
        self.assertEqual(sorted(t.get_text() for t in ax_a.texts),
                         sorted([fig6.TRUTH_NAME, fig6.RANGE_NAME, fig6.BAND_NAME]))
        self.assertEqual(len(ax_a.collections), 1)
        self.assertTrue(all(ln.get_drawstyle() == "steps-mid" for ln in ax_a.lines))
        lo, hi = ax_a.get_ylim()
        drawn = [np.asarray(ln.get_ydata(), dtype=float) for ln in ax_a.lines]
        drawn.append(ax_a.collections[0].get_paths()[0].vertices[:, 1])
        for y in drawn:
            y = y[np.isfinite(y)]
            self.assertTrue(((y >= lo) & (y <= hi)).all())
        # b: each block count's median SD ratio, in its key; the grey band is the true SD's chi-square interval,
        # its bounds' median ratios to the true SD over the window.
        keys = [t.get_text() for t in ax_b.get_legend().get_texts()]
        self.assertEqual(len(keys), arms)
        for row in rows.itertuples():
            self.assertTrue(any(f"{int(row.trunks)} blocks" in k and f"{row.sd_ratio_median:.2f}×" in k for k in keys),
                            (row.trunks, keys))
        band = ax_b.collections[0]
        self.assertEqual(to_hex(band.get_facecolor()[0]), style.FILL)
        v = band.get_paths()[0].vertices
        ratio = v[:, 1] / v[:, 0]
        known = sd[sd.sd_true_log10 > 0]
        self.assertAlmostEqual(ratio.min(), float(np.median(known.sd_true_ci_lo / known.sd_true_log10)))
        self.assertAlmostEqual(ratio.max(), float(np.median(known.sd_true_ci_hi / known.sd_true_log10)))
        # c: at each time, the share of the genomes whose band holds the fits' mean (blue) or the truth (dashed
        # ink, over the blue), from E5_coverage; c and d share one coverage axis.
        c = cov[cov.trunk_bins == many.trunk_bins].sort_values("point")
        lines = {(to_hex(ln.get_color()), ln.get_linestyle()): ln for ln in ax_c.lines if len(ln.get_xdata()) > 2}
        mean, truth = lines[(style.BLUE, "-")], lines[(style.INK, "--")]
        np.testing.assert_allclose(mean.get_ydata(), 100.0 * c.coverage_fit_mean)
        np.testing.assert_allclose(truth.get_ydata(), 100.0 * c.coverage)
        self.assertGreater(truth.get_zorder(), mean.get_zorder())
        self.assertEqual(sorted(t.get_text() for t in ax_c.texts), sorted([fig6.MEAN_NAME, fig6.TRUTH_NAME]))
        self.assertEqual((ax_c.get_ylim(), ax_d.get_ylim()), (fig6.COVERAGE_LIM, fig6.COVERAGE_LIM))
        # d: per block count a strip of genomes and a mean bar, for the fits' mean and for the truth.
        bars = [ln for ln in ax_d.lines if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1]
                and ln.get_marker() in ("None", "", None) and to_hex(ln.get_color()) == style.INK]
        want = sorted(100.0 * v for v in list(rows.coverage_of_fit_mean_mean) + list(rows.coverage_mean))
        got = sorted(float(ln.get_ydata()[0]) for ln in bars)
        self.assertEqual(len(got), 2 * arms)
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=1e-9)
        for ln in bars:
            x0, x1 = ln.get_xdata()
            self.assertAlmostEqual((x1 - x0) / 2.0, style.STRIP_BAR_HALF * fig6.ARM_STEP)
        dots = [ln for ln in ax_d.lines if ln.get_marker() == "o"]
        self.assertEqual(len(dots), 2 * arms)
        per_arm = bands.groupby("trunk_bins").dataset.nunique()
        self.assertEqual(sorted(len(ln.get_xdata()) for ln in dots),
                         sorted(int(per_arm[t]) for t in rows.trunk_bins for _ in range(2)))
        self.assertIn("Nominal 95%", [t.get_text() for t in ax_d.texts])
        # Each strip is a swarm in which no genome hides another: dot centres a dot's diameter apart, in mm.
        fig.canvas.draw()
        mm = fig.dpi / 25.4
        for ln in dots:
            xy = ax_d.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()])) / mm
            gaps = [math.dist(p, q) for k, p in enumerate(xy) for q in xy[k + 1:]]
            self.assertGreaterEqual(min(gaps, default=math.inf), style.STRIP_MS / style.PT_PER_MM - 1e-6)
        self.assertFalse(any(t == "Median ratio" or "↑" in t for t in texts))

    def test_figs1_ends_both_tools_where_c_psmc_stops(self):
        from matplotlib.text import Text

        from experiment.pipeline.figures import figs1, style
        from experiment.pipeline.figures.fig2 import model_order

        traces = self._table("E1_traces")
        last_c = int(traces[(traces.tool == "c") & traces.loglike.notna()]["iteration"].max())
        fig = self._draw("figs1")
        shown = [ax for ax in fig.axes if ax.axison]
        per, diffs = figs1.series(traces), figs1.differences(traces)
        models = model_order(per.keys())
        n = traces["model"].nunique()
        self.assertEqual(len(shown), 2 * n)  # per model, the gains above and the differences below
        for ax, bx, m in zip(shown[:n], shown[n:], models):
            for ln in ax.lines:  # the gains, from iteration 1
                self.assertGreaterEqual(min(ln.get_xdata()), 1)
                self.assertLessEqual(max(ln.get_xdata()), last_c)
            lines = [ln for ln in bx.lines if ln.get_color() == style.INK2]  # the differences, from iteration 0
            self.assertEqual(len(lines), len(diffs[m]))
            for ln in ax.lines + lines:
                self.assertIn(ln.get_marker(), ("None", "", None))  # continuous lines carry no markers
            for ln in lines:
                self.assertEqual((min(ln.get_xdata()), max(ln.get_xdata())), (0, last_c))
            self.assertEqual(ax.collections + bx.collections, [])  # no band, no hairlines
            self.assertFalse([t for t in ax.texts + bx.texts if style.TIMES in t.get_text()])  # no arrows' folds
        self.assertEqual([t.get_text() for t in shown[n].texts], ["psmc-rs higher", "C psmc higher"])
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
