"""Tests for experiment.pipeline.metrics."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, demography, metrics  # noqa: E402
from experiment.pipeline.results import Curve  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy not installed")
class Metrics(unittest.TestCase):
    def test_log_grid(self):
        g = metrics.log_grid(1e3, 1e6, 4)
        self.assertEqual([round(math.log10(x), 12) for x in g], [3.0, 4.0, 5.0, 6.0])
        with self.assertRaises(ValueError):
            metrics.log_grid(0, 1, 10)

    def test_rmse(self):
        self.assertEqual(metrics.rmse_log10([1, 10], [1, 10]), 0.0)
        self.assertAlmostEqual(metrics.rmse_log10([10, 10], [1, 100]), 1.0)
        with self.assertRaises(ValueError):
            metrics.rmse_log10([1, -1], [1, 1])

    def test_window_of_constant_model(self):
        eps = demography.epochs(config.CONSTANT)
        lo, hi = metrics.informative_window(eps, (0.01, 0.99), 25.0)
        self.assertAlmostEqual(lo, -2 * 10_000 * math.log(0.99) * 25, places=3)
        self.assertAlmostEqual(hi, -2 * 10_000 * math.log(0.01) * 25, places=3)
        grid = metrics.model_grid(eps, config.TimeGrid(points=50), 25.0)
        self.assertEqual(len(grid), 50)
        self.assertAlmostEqual(grid[0], lo)

    def test_rmse_against_truth(self):
        eps = demography.epochs(config.BOTTLENECK)
        g = 25.0
        # The truth itself as a step curve (in years) has zero error.
        years = tuple(e.start * g for e in eps)
        sizes = tuple(e.start_size for e in eps)
        exact = Curve(years, sizes)
        grid = metrics.model_grid(eps, config.TimeGrid(), g)
        self.assertEqual(metrics.rmse_vs_truth(exact, eps, grid, g), 0.0)
        # Twice the truth everywhere: log10(2) at every point.
        double = Curve(years, tuple(2 * s for s in sizes))
        self.assertAlmostEqual(metrics.rmse_vs_truth(double, eps, grid, g), math.log10(2))
        self.assertAlmostEqual(metrics.rmse_between(double, exact, grid), math.log10(2))
        self.assertTrue(all(abs(b - math.log10(2)) < 1e-12 for b in metrics.log10_bias(double, eps, grid, g)))

    def test_fixed_grid(self):
        g = metrics.fixed_grid(config.TimeGrid())
        self.assertAlmostEqual(g[0], 1e4)
        self.assertAlmostEqual(g[-1], 1e6)


if __name__ == "__main__":
    unittest.main()
