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

from experiment.pipeline import config, demography, paths  # noqa: E402
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
        self.assertEqual(style.ROUTES["exact"].color, style.RS02.color)

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
        self.assertEqual(data.decades(3e3, 2e5), (1e3, 1e6))


@unittest.skipUnless(HAS_PLOTTING, "numpy, pandas and matplotlib are needed")
class SchemasAndStandIns(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)

    def test_every_schema_is_written_and_read_back(self):
        from experiment.pipeline.figures import data, fake
        from experiment.pipeline.figures.schemas import SCHEMAS

        written = fake.write_all(self.runs, config.SMOKE)
        self.assertEqual({p.name for p in written}, {s.file for s in SCHEMAS.values()})
        for key in SCHEMAS:
            df = data.read(self.runs, key)
            self.assertGreater(len(df), 0, key)

    def test_missing_table_and_columns_are_named(self):
        from experiment.pipeline.figures import data
        from experiment.pipeline.figures.schemas import SCHEMAS

        with self.assertRaises(data.MissingTable):
            data.read(self.runs, "E2_summary")
        path = SCHEMAS["E2_summary"].path(self.runs)
        path.parent.mkdir(parents=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["sample", "mu"], ["Papuan", "1.25e-8"]])
        with self.assertRaises(data.SchemaError) as cm:
            data.read(self.runs, "E2_summary")
        self.assertIn("gen_years", str(cm.exception))

    def test_stand_ins_follow_the_profile(self):
        import pandas as pd

        from experiment.pipeline.figures import fake

        fake.write_all(self.runs, config.SMOKE, ["E3", "E5"])
        runs = pd.read_csv(self.runs / "E3" / "tables" / "E3_runs.csv")
        self.assertEqual(sorted(runs["genome_mb"].unique()), list(config.SMOKE.e3.genome_sizes_mb))
        cal = pd.read_csv(self.runs / "E5" / "tables" / "E5_calibration.csv")
        self.assertEqual(sorted(cal["trunk_bins"].unique()), sorted(config.SMOKE.e5.trunk_bins))
        self.assertTrue(((cal["coverage"] >= 0) & (cal["coverage"] <= 1)).all())

    def test_calibration_definition(self):
        import numpy as np

        from experiment.pipeline.figures import fake

        truth = np.zeros(3)
        fits = {d: np.full(3, v) for d, v in ((1, -0.1), (2, 0.1))}
        boots = {(1, 10): np.array([[-0.2] * 3, [0.2] * 3]), (2, 10): np.array([[0.2] * 3, [0.4] * 3])}
        rows = fake.calibration(fits, boots, truth, (0.025, 0.975))
        self.assertEqual(len(rows), 3)
        sd_true = float(np.std([-0.1, 0.1], ddof=1))
        sds = [float(np.std([-0.2, 0.2], ddof=1)), float(np.std([0.2, 0.4], ddof=1))]
        self.assertAlmostEqual(rows[0]["sd_true"], sd_true)
        self.assertAlmostEqual(rows[0]["sd_boot_median"], float(np.median(sds)))
        self.assertAlmostEqual(rows[0]["sd_ratio_median"], float(np.median(sds)) / sd_true)
        self.assertEqual(rows[0]["coverage"], 0.5)  # only dataset 1's band contains the truth, 0
        self.assertEqual(rows[0]["n_datasets"], 2)

    def test_refuses_the_production_runs_directory(self):
        from experiment.pipeline.figures import fake

        with self.assertRaises(ValueError):
            fake.write_all(paths.RUNS, config.SMOKE, ["E3"])


class Readme(unittest.TestCase):
    def test_lists_every_table_and_column(self):
        from experiment.pipeline.figures.schemas import SCHEMAS

        text = (ROOT / "experiment" / "pipeline" / "figures" / "README.md").read_text()
        for s in SCHEMAS.values():
            rows = [line for line in text.splitlines() if line.startswith(f"| `{s.experiment}/tables/{s.file}`")]
            self.assertEqual(len(rows), 1, s.file)
            for col in s.names:
                self.assertIn(f"`{col}`", rows[0], f"{s.file}: {col}")


class FiguresConfig(unittest.TestCase):
    def test_settings(self):
        f = config.FIGURES
        self.assertLess(f.sim_years[0], f.sim_years[1])
        self.assertIn(f.e4_example_mask, {m.name for m in config.PRODUCTION.e4.masks})
        self.assertTrue(set(f.e4_bias_masks) <= {m.name for m in config.PRODUCTION.e4.masks})
        self.assertLessEqual(f.e5_example_dataset, config.PRODUCTION.e5.bootstrap_datasets)
        for name in config.MODEL_ORDER:
            lo = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.01) * 25.0
            hi = demography.coalescence_quantile(demography.epochs(config.MODELS[name]), 0.99) * 25.0
            self.assertTrue(f.sim_years[0] < lo and hi < f.sim_years[1], name)
            self.assertTrue(math.isfinite(lo))


if __name__ == "__main__":
    unittest.main()
