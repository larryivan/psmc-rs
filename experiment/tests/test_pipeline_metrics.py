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


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy not installed")
class ProjectedTruth(unittest.TestCase):
    def test_a_constant_size_projects_onto_itself(self):
        eps = demography.epochs(config.CONSTANT)
        for a, b in ((0.0, 500.0), (1000.0, 3000.0), (5000.0, math.inf)):
            self.assertAlmostEqual(metrics.projected_ne(eps, a, b) / 10_000, 1.0, places=6)

    def test_two_epochs_weight_by_the_time_spent_uncoalesced(self):
        # N1 on [0, T), N2 after: from a < T the projection is N1 (1 - q) + N2 q, q = exp(-(T - a) / (2 N1)).
        n1, n2, t = 1_000.0, 20_000.0, 1_500.0
        model = config.DemographicModel("two", n1, (config.SizeChange(t, n2),), "test")
        eps = demography.epochs(model)
        a = 300.0
        q = math.exp(-(t - a) / (2 * n1))
        self.assertAlmostEqual(metrics.projected_ne(eps, a, math.inf), n1 * (1 - q) + n2 * q, delta=1e-4 * n2)
        self.assertAlmostEqual(metrics.projected_ne(eps, 0.0, t), n1, delta=1e-6 * n1)

    def test_free_spans_follow_the_pattern(self):
        from experiment.pipeline import results

        fit = results.Fit(theta=0.01, rho=0.002, t_max=15.0, n=63, lambdas=(1.0,) * 64, pattern="4+25*2+4+6")
        spans = metrics.free_spans(fit)
        self.assertEqual(len(spans), 28)
        self.assertEqual((spans[0], spans[1], spans[-1]), ((0, 4), (4, 6), (58, 64)))
        no_pattern = results.Fit(theta=0.01, rho=0.002, t_max=15.0, n=3, lambdas=(1.0,) * 4)
        self.assertEqual(metrics.free_spans(no_pattern), [(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_projected_truth_of_a_fit(self):
        from experiment.pipeline import results

        eps = demography.epochs(config.BOTTLENECK)
        mu, g = config.SIMULATION_SCALING.mu, config.SIMULATION_SCALING.gen_years
        theta = 4 * 20_000 * mu * 100  # N0 of the fit equals the model's N0
        fit = results.Fit(theta=theta, rho=theta / 5, t_max=15.0, n=63, lambdas=(1.0,) * 64, pattern="4+25*2+4+6")
        proj = metrics.projected_truth(fit, eps, mu, g, 100)
        self.assertEqual(len(proj.ne), 28)
        self.assertEqual(proj.years[0], 0.0)
        self.assertEqual(proj.years[1], fit.curve(mu, g, 100).years[4])
        # The present-day span covers the bottleneck's start: within the sizes it spans.
        sizes = [e.start_size for e in eps]
        self.assertTrue(min(sizes) <= proj.ne[0] <= max(sizes))
        grid = metrics.model_grid(eps, config.TimeGrid(), g)
        self.assertEqual(metrics.rmse_vs_projected_truth(proj, proj, grid), 0.0)


if __name__ == "__main__":
    unittest.main()
