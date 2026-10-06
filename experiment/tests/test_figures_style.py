"""Tests for the figures' style, table schemas, readers and stand-in tables (experiment.pipeline.figures)."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, demography  # noqa: E402
from experiment.pipeline.results import Curve  # noqa: E402

HAS_PLOTTING = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "matplotlib"))


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Style(unittest.TestCase):
    def test_text_sizes_and_fonts(self):
        from experiment.pipeline.figures import style

        rc = style.rc_params()
        self.assertEqual(rc["font.size"], 7.0)
        self.assertEqual(rc["xtick.labelsize"], 6.0)
        self.assertEqual(rc["font.sans-serif"][:2], ["Arial", "Helvetica"])
        self.assertEqual(rc["pdf.fonttype"], 42)
        self.assertEqual(rc["svg.fonttype"], "none")
        self.assertFalse(rc["axes.spines.top"])
        self.assertFalse(rc["axes.spines.right"])

    def test_entity_colours_are_fixed(self):
        from experiment.pipeline.figures import style

        self.assertEqual(style.RS02.color, "#2a78d6")
        self.assertEqual(style.C_PSMC.color, "#eb6834")
        self.assertEqual(style.RS01.color, "#1baf7a")
        self.assertEqual(style.TRUTH.color, "#0b0b0b")
        self.assertIs(style.TOOLS["rs"], style.RS02)
        self.assertIs(style.TOOLS["c"], style.C_PSMC)
        # Routes and formats reuse the same colour for the same thing.
        self.assertEqual(style.ROUTES["multihetsep"].color, style.FORMATS["multihetsep"].color)
        # psmc-rs's default (the callable-length model) is psmc-rs's blue; every called
        # bin taken as full, and the fq2psmcfa route it equals, are its darker variant.
        self.assertEqual(style.ROUTES["mask_auto"].color, style.RS02.color)
        self.assertEqual(style.ROUTES["mask_ignore"].color, style.BLUE_650)
        self.assertEqual(style.ROUTES["fq2psmcfa"].color, style.BLUE_650)

    def test_every_e4_route_has_a_style(self):
        from experiment.pipeline import config
        from experiment.pipeline.experiments import e4
        from experiment.pipeline.figures import fig5, style

        self.assertEqual(set(style.ROUTES), set(e4.ROUTES))
        self.assertEqual(set(style.ROUTE_ORDER), set(e4.ROUTES))
        self.assertEqual(set(config.E4_BIN_CALLABLE), set(e4.ROUTES))
        self.assertTrue(set(fig5.DRAWN) <= set(e4.ROUTES))
        self.assertNotIn("fq2psmcfa", fig5.DRAWN)  # it fits as mask_ignore does (asserted by the report)

    def test_figure_sizes(self):
        from experiment.pipeline.figures import style

        fig = style.figure("double", 100)
        self.assertAlmostEqual(fig.get_figwidth() * 25.4, 183.0)
        self.assertAlmostEqual(fig.get_figheight() * 25.4, 100.0)
        fig = style.figure("single", 50)
        self.assertAlmostEqual(fig.get_figwidth() * 25.4, 89.0)

    def test_step_coordinates(self):
        from experiment.pipeline.figures import style

        x, y = style.step_xy([0.0, 10.0, 100.0, 1000.0], [1.0, 2.0, 3.0, 4.0], (50.0, 500.0))
        self.assertEqual(list(x), [50.0, 100.0, 500.0])
        self.assertEqual(list(y), [2.0, 3.0, 3.0])
        x, y = style.step_xy([0.0, 10.0], [1.0, 2.0], (1.0, 5.0))
        self.assertEqual(list(x), [1.0, 5.0])
        self.assertEqual(list(y), [1.0, 1.0])

    def test_save_writes_three_formats_reproducibly(self):
        from experiment.pipeline.figures import style

        def draw():
            fig = style.figure("single", 40)
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 4, 9], color=style.BLUE)
            ax.set_xlabel("x")
            style.panel_letters(fig, [ax])
            return fig

        with tempfile.TemporaryDirectory() as d:
            first = [p.read_bytes() for p in style.save(draw(), Path(d) / "a", "f", dpi=72)]
            second = [p.read_bytes() for p in style.save(draw(), Path(d) / "b", "f", dpi=72)]
            self.assertEqual([p.suffix for p in sorted((Path(d) / "a").iterdir())], [".pdf", ".png", ".svg"])
            self.assertFalse(list(Path(d).rglob("*.partial")))
        self.assertEqual(first, second)
        self.assertNotIn(b"CreationDate", first[0])
        self.assertIn(b"<text", first[1])  # text stays text in the SVG

    def test_notes_and_legends_are_7pt_ticks_the_documented_exception(self):
        from experiment.pipeline.figures import style

        fig = style.figure("single", 40)
        ax = fig.add_subplot(111)
        self.assertEqual(style.note(ax, 0.5, 0.5, "a note").get_fontsize(), 7.0)
        leg = style.legend(ax, [ax.plot([0, 1], [0, 1], label="a line")[0]])
        self.assertEqual(leg.get_texts()[0].get_fontsize(), 7.0)
        self.assertEqual(style.TICK_PT, 6.0)
        self.assertIn("exception to 7 pt: tick labels are 6 pt", " ".join(style.__doc__.split()))

    def test_panel_letters_align_columns(self):
        from experiment.pipeline.figures import style

        fig = style.figure("single", 80)
        gs = style.grid(fig, 2, 1, left_mm=20.0)
        top, bottom = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        top.plot([0, 1], [0, 100000])  # wide tick labels
        bottom.plot([0, 1], [0, 1])
        top.set_ylabel("y")
        bottom.set_ylabel("y")
        a, b = style.panel_letters(fig, [top, bottom])
        self.assertNotAlmostEqual(a.get_position()[0], b.get_position()[0], places=3)
        for t in fig.texts[:]:
            t.remove()
        a, b = style.panel_letters(fig, [top, bottom], align_cols=True)
        self.assertAlmostEqual(a.get_position()[0], b.get_position()[0])
        self.assertLess(a.get_position()[0], top.get_position().x0)


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Data(unittest.TestCase):
    def test_truth_line_has_exact_change_times(self):
        from experiment.pipeline.figures import data

        x, y = data.truth_xy("bottleneck", 25.0, (1e3, 1e7))
        changes = [t * 4 * 20_000 * 25.0 for t, _ in config.BOTTLENECK.ms_events]
        for t in changes:
            self.assertIn(t, list(x))
        eps = demography.epochs(config.BOTTLENECK)
        for xi, yi in zip(x, y):
            if xi not in changes:
                self.assertAlmostEqual(yi, demography.ne_at(eps, xi / 25.0))

    def test_truth_line_samples_growth(self):
        from experiment.pipeline.figures import data

        x, y = data.truth_xy("zigzag", 25.0, (1e3, 1e7))
        self.assertGreater(len(x), 100)
        self.assertTrue(all(b >= a for a, b in zip(x, x[1:])))

    def test_truth_table_check(self):
        import numpy as np
        import pandas as pd

        from experiment.pipeline.figures import data

        years = np.logspace(3, 7, 20)
        good = pd.DataFrame({"model": "expansion", "years": years, "ne": data.truth_at("expansion", years, 25.0)})
        data.check_truth_table(good, 25.0)
        bad = good.assign(ne=good["ne"] * 1.01)
        with self.assertRaises(ValueError):
            data.check_truth_table(bad, 25.0)

    def test_quantiles_of_curves(self):
        import numpy as np

        from experiment.pipeline.figures import data

        cs = [Curve((0.0, 10.0), (float(k), float(k))) for k in (1, 10, 100)]
        grid = np.array([1.0, 20.0])
        med = data.median(cs, grid)
        self.assertTrue(np.allclose(med, [10.0, 10.0]))
        lo, hi = data.quantiles(cs, grid, (0.0, 1.0))
        self.assertTrue(np.allclose(lo, 1.0) and np.allclose(hi, 100.0))
        self.assertTrue(np.allclose(data.log10_ratio(cs[1], cs[0], grid), 1.0))

    def test_curves_from_a_table(self):
        import pandas as pd

        from experiment.pipeline.figures import data

        df = pd.DataFrame({"model": ["a", "a", "b", "b"], "interval": [1, 0, 0, 1],
                           "years": [5.0, 0.0, 0.0, 7.0], "ne": [2.0, 1.0, 3.0, 4.0]})
        cs = data.curves(df, ["model"])
        self.assertEqual(cs[("a",)].years, (0.0, 5.0))
        self.assertEqual(cs[("a",)].ne, (1.0, 2.0))

    def test_formatting(self):
        from experiment.pipeline.figures import data

        self.assertEqual(data.fmt_bp(50_000_000), "50 Mb")
        self.assertEqual(data.fmt_bp(500_000), "500 kb")
        self.assertEqual(data.fmt_bp(2_500_000), "2.5 Mb")
        self.assertEqual(data.fmt_sig(0.0381), "0.038")
        self.assertIn("10$^{-4}$", data.fmt_sig(9.0e-4))
        self.assertEqual(data.decades(1.5e3, 6e5), (1e3, 1e6))

    def test_decades_never_add_an_empty_decade(self):
        from experiment.pipeline.figures import data

        # More than a third of a decade from a power of ten: the next 1, 2 or 5.
        self.assertEqual(data.decades(3e3, 2e5), (2e3, 2e5))
        self.assertEqual(data.decades(769.2, 6.3e4), (500.0, 1e5))  # Fig 5d: not down to 10^2
        self.assertEqual(data.decades(521.5, 1.073e5), (500.0, 2e5))
        self.assertEqual(data.decades(1e-6, 2.032), (1e-6, 5.0))
        self.assertEqual(data.decades(2.449e-6, 0.4902), (2e-6, 1.0))
        # Close to a power of ten, or on one: that power.
        self.assertEqual(data.decades(1.561e4, 3.0e6), (1e4, 5e6))
        self.assertEqual(data.decades(1e3, 1e5), (1e3, 1e5))
        self.assertEqual(data.decades(1e3 * (1 + 1e-12), 1e5 * (1 - 1e-12)), (1e3, 1e5))
        for lo, hi in ((1.1, 9.9), (3.3e-5, 4.4e-2), (7.7e2, 1.2e7)):
            a, b = data.decades(lo, hi)
            self.assertLessEqual(a, lo)
            self.assertGreaterEqual(b, hi)
            self.assertLess(math.log10(lo / a), 0.5)  # under half a decade added at either end
            self.assertLess(math.log10(b / hi), 0.5)


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Schemas(unittest.TestCase):
    """The schemas against real tables are checked in test_figures_draw (the experiments' tiny runs)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)

    def test_missing_table_and_columns_are_named(self):
        from experiment.pipeline.figures import data
        from experiment.pipeline.figures.schemas import SCHEMAS

        with self.assertRaises(data.MissingTable):
            data.read(self.runs, "E2_rs_vs_c")
        path = SCHEMAS["E2_rs_vs_c"].path(self.runs)
        path.parent.mkdir(parents=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["sample_id", "window_lo_years"], ["Papuan", "1000"]])
        with self.assertRaises(data.SchemaError) as cm:
            data.read(self.runs, "E2_rs_vs_c")
        self.assertIn("rmse_log10_ne_rs_vs_c", str(cm.exception))

    def test_every_schema_names_an_experiments_table(self):
        from experiment.pipeline.experiments import e1, e2, e3, e4, e5, e6
        from experiment.pipeline.figures.schemas import SCHEMAS

        written = {f"{m.NAME}/{name}" for m in (e1, e2, e3, e4, e5) for name in m.TABLES.values()}
        written |= {f"E6/{name}" for part in e6.TABLES.values() for name in part.values()}
        for key, s in SCHEMAS.items():
            self.assertIn(f"{s.experiment}/{s.file}", written, key)


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class FigureContent(unittest.TestCase):
    """What Figs 1, 5 and 6 draw, without the experiments' tables."""

    def test_fig1_shows_the_callable_length_model_and_the_fit_feeds_the_outputs(self):
        from matplotlib.colors import to_hex
        from matplotlib.patches import FancyBboxPatch

        from experiment.pipeline.figures import fig1

        fig = fig1.draw()
        ax = fig.axes[0]
        text = "\n".join(t.get_text() for t in ax.texts)
        self.assertIn("--bin-callable", text)
        self.assertIn(r"e_T(k)^{\,c/s}", text)
        self.assertIn("callable", text.split("Route B")[1].split("Inputs made")[0])
        self.assertIn("C psmc's model and EM", text)
        bullets = [t for t in ax.texts if t.get_text() == "•"]
        self.assertEqual(len(bullets), len(fig1.FIT_ITEMS))
        # One line leaves the fit box (the lowest of psmc-rs's boxes) for a bus beside the five outputs.
        boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
        rs_boxes = [p for p in boxes if to_hex(p.get_facecolor()) == fig1.RS_FILL]
        fit = min(rs_boxes, key=lambda p: p.get_y())
        outputs = [p for p in boxes if p.get_x() > fit.get_x() + fit.get_width() + 1.0]
        self.assertEqual(len(outputs), 5)
        lines = [ln for ln in ax.lines if ln.get_color() == fig1.style.BLUE]
        self.assertEqual(len(lines), 2)
        out, bus = lines
        self.assertAlmostEqual(out.get_xdata()[0], fit.get_x() + fit.get_width())
        self.assertTrue(fit.get_y() < out.get_ydata()[0] < fit.get_y() + fit.get_height())
        self.assertEqual(out.get_xdata()[1], bus.get_xdata()[0])
        centres = [p.get_y() + p.get_height() / 2 for p in outputs]
        self.assertLessEqual(min(bus.get_ydata()), min(centres))
        self.assertGreaterEqual(max(bus.get_ydata()), max(centres))

    def test_fig5_draws_every_called_bin_taken_as_full_on_top(self):
        from experiment.pipeline.figures import fig5, style

        self.assertEqual(set(fig5.LINE_Z), set(style.ROUTES))
        self.assertEqual(max(fig5.LINE_Z, key=fig5.LINE_Z.get), "mask_ignore")
        self.assertGreater(fig5.LINE_Z["mask_ignore"], fig5.LINE_Z["nomask"])
        self.assertGreater(fig5.LINE_Z["mask_ignore"], fig5.LINE_Z["mask_auto"])

    def test_fig6_finds_projected_truth_coverage_when_given(self):
        import pandas as pd

        from experiment.pipeline.figures import fig6

        cov = pd.DataFrame(columns=["trunk_bins", "coverage", "coverage_ci_lo", "coverage_ci_hi", "coverage_fit_mean"])
        summary = pd.DataFrame(columns=["coverage_mean", "coverage_mean_ci_lo", "coverage_mean_ci_hi",
                                        "coverage_of_fit_mean_mean"])
        self.assertIsNone(fig6.projected_columns(cov, summary))
        cov["coverage_projected"] = []
        cov["coverage_projected_ci_lo"] = []
        cov["coverage_projected_ci_hi"] = []
        cols = fig6.projected_columns(cov, summary)
        self.assertEqual((cols["point"], cols["lo"], cols["hi"], cols["mean"]),
                         ("coverage_projected", "coverage_projected_ci_lo", "coverage_projected_ci_hi", None))
        summary["coverage_of_projected_truth_mean"] = []
        summary["coverage_of_projected_truth_mean_ci_lo"] = []
        summary["coverage_of_projected_truth_mean_ci_hi"] = []
        cols = fig6.projected_columns(cov, summary)
        self.assertEqual((cols["mean"], cols["mean_lo"], cols["mean_hi"]),
                         ("coverage_of_projected_truth_mean", "coverage_of_projected_truth_mean_ci_lo",
                          "coverage_of_projected_truth_mean_ci_hi"))

        row = pd.DataFrame([{"trunk_mb": 50.0, "trunks": 6, "coverage_mean": 0.3715, "coverage_mean_ci_lo": 0.3145,
                             "coverage_mean_ci_hi": 0.429, "coverage_of_fit_mean_mean": 0.827,
                             "coverage_of_projected_truth_mean": 0.6, "coverage_of_projected_truth_mean_ci_lo": 0.5,
                             "coverage_of_projected_truth_mean_ci_hi": 0.7}]).itertuples().__next__()
        label = fig6.coverage_label(row, cols)
        self.assertIn("50 Mb trunks (6 per genome)", label)
        self.assertIn("37% (31–43%)", label)
        self.assertIn("projected truth 60% (50–70%)", label)
        self.assertIn("fits' mean 83%", label)
        self.assertNotIn("projected", fig6.coverage_label(row, None))


class Readme(unittest.TestCase):
    def test_lists_every_table_and_column(self):
        from experiment.pipeline.figures.schemas import SCHEMAS

        text = (ROOT / "experiment" / "pipeline" / "figures" / "README.md").read_text()
        for s in SCHEMAS.values():
            rows = [line for line in text.splitlines() if line.startswith(f"| `{s.experiment}/tables/{s.file}`")]
            self.assertEqual(len(rows), 1, s.file)
            for col in s.names:
                self.assertIn(f"`{col}`", rows[0], f"{s.file}: {col}")


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class Performance(unittest.TestCase):
    """Figs 4 and S4 (E3): thread counts and the ceiling of the thread scaling."""

    def test_fig4_draws_each_thread_count_of_a_tool_in_its_own_style(self):
        import pandas as pd

        from experiment.pipeline.figures import fig4

        cfg = config.PRODUCTION.e3
        # Table 2's rows: every size at 1, 8 and 30 threads, the thread-scaling size at 2, 4 and 16 too.
        rows = [("c", 1)] + [(tool, t) for tool in ("rs01", "rs02") for t in (1, 8, 30)] + [("rs02", t) for t in (2, 4, 16)]
        table2 = pd.DataFrame(rows, columns=["tool", "threads"])
        series = fig4.series(table2, fig4.tool_threads(cfg))
        self.assertEqual([(tool, t) for tool, t, _ in series], rows[:7])  # not the thread-scaling counts
        for tool in ("rs01", "rs02"):
            ents = [e for name, _, e in series if name == tool]
            self.assertEqual(len({str(e.linestyle) for e in ents}), 3, tool)
            self.assertEqual([e.filled for e in ents], [True, False, False])
            self.assertEqual(ents[0].linestyle, "-")
        self.assertEqual(len(fig4.series(table2)), 10)  # without the profile's counts: every one in the table

    def test_figs4_ceiling_of_whole_sequences_per_thread(self):
        from experiment.pipeline.figures import figs4

        speedup = figs4.ceiling_speedup([1, 2, 4, 8, 16, 30, 60], 30)
        self.assertEqual([round(float(x), 6) for x in speedup], [1.0, 2.0, 3.75, 7.5, 15.0, 30.0, 30.0])
        self.assertEqual(float(figs4.ceiling_speedup(1, 30)), 1.0)


class FiguresConfig(unittest.TestCase):
    def test_settings(self):
        f = config.FIGURES
        self.assertLess(f.sim_years[0], f.sim_years[1])
        self.assertIn(f.e4_example_mask, {m.name for m in config.PRODUCTION.e4.masks})
        self.assertTrue(set(f.e4_bias_masks) <= {m.name for m in config.PRODUCTION.e4.masks})
        self.assertLessEqual(config.PRODUCTION.e5.example_dataset, config.PRODUCTION.e5.bootstrap_datasets)
        for name in config.MODEL_ORDER:
            lo = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.01) * 25.0
            hi = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.99) * 25.0
            self.assertTrue(f.sim_years[0] < lo and hi < f.sim_years[1], name)
            self.assertTrue(math.isfinite(lo))


if __name__ == "__main__":
    unittest.main()
