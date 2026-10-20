"""Tests for experiment.pipeline.demography (msprime and stdpopsim parts skip when absent)."""

from __future__ import annotations

import math
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import has_modules  # noqa: E402

from experiment.pipeline import config, demography  # noqa: E402

HAS_MSPRIME = has_modules("msprime")
if HAS_MSPRIME:
    import numpy as np  # msprime's own dependency


class ClosedForm(unittest.TestCase):
    def test_constant_size(self):
        eps = demography.epochs(config.CONSTANT)
        self.assertEqual(len(eps), 1)
        self.assertEqual(demography.ne_at(eps, 123456.0), 10_000)
        # P(T > t) = exp(-t / 2N)
        self.assertAlmostEqual(demography.coalescence_cdf(eps, 20_000), 1 - math.exp(-1), places=12)
        self.assertAlmostEqual(demography.coalescence_quantile(eps, 0.5), 20_000 * math.log(2), places=6)

    def test_bottleneck_sizes(self):
        eps = demography.epochs(config.BOTTLENECK)
        for t, n in ((0, 20_000), (799.9, 20_000), (800, 1000), (1199, 1000), (1200, 10_000), (4000, 5000), (1e6, 10_000)):
            self.assertEqual(demography.ne_at(eps, t), n, t)

    def test_quantiles_invert_the_cdf(self):
        for name in config.MODEL_ORDER:
            eps = demography.epochs(config.MODELS[name])
            for q in (0.01, 0.2, 0.5, 0.99):
                t = demography.coalescence_quantile(eps, q)
                self.assertAlmostEqual(demography.coalescence_cdf(eps, t), q, places=9, msg=name)

    def test_zigzag_sizes_follow_growth(self):
        eps = demography.epochs(config.ZIGZAG)
        self.assertEqual(len(eps), 7)
        # Each growth epoch ends near N0/10 or N0 (the zigzag) and the last is N0/10.
        for e in eps[1:6]:
            self.assertTrue(math.isclose(e.end_size, 7156, rel_tol=1e-3) or math.isclose(e.end_size, 71560, rel_tol=1e-3))
        self.assertEqual(eps[-1].start_size, 7156.0)

    def test_truth_record_roundtrip(self):
        rec = demography.truth_record(config.ZIGZAG)
        back = demography.epochs_from_record(rec)
        self.assertEqual(back, demography.epochs(config.ZIGZAG))
        self.assertIn("0.01", rec["coalescence_quantiles"])
        self.assertNotIn("ms", rec)
        self.assertEqual(demography.truth_record(config.BOTTLENECK)["ms"]["n0"], 20_000)

    def test_changes_must_increase(self):
        bad = config.DemographicModel("bad", 1.0, (config.SizeChange(10, 1.0), config.SizeChange(5, 2.0)), "")
        demography.epochs(bad)  # sorted: fine
        dup = config.DemographicModel("dup", 1.0, (config.SizeChange(10, 1.0), config.SizeChange(10, 2.0)), "")
        with self.assertRaises(ValueError):
            demography.epochs(dup)


@unittest.skipUnless(HAS_MSPRIME, "msprime not installed")
class AgainstMsprime(unittest.TestCase):
    def test_epochs_match_the_debugger(self):
        for name in config.MODEL_ORDER:
            res = demography.compare_with_debugger(config.MODELS[name])
            self.assertLess(res["max_rel_diff"], 1e-9)

    def test_hazard_integrates_msprimes_sizes(self):
        for name in config.MODEL_ORDER:
            model = config.MODELS[name]
            eps = demography.epochs(model)
            dd = demography.to_msprime(model).debug()
            for horizon in (500.0, 5000.0, 50000.0):
                grid = np.linspace(0.0, horizon, 20001)
                sizes = dd.population_size_trajectory(grid)[:, 0]
                numeric = np.trapezoid(1.0 / (2.0 * sizes), grid)
                exact = demography.cumulative_hazard(eps, horizon)
                # Step changes fall between grid points: allow the trapezoid's error there.
                self.assertAlmostEqual(numeric / exact, 1.0, delta=1e-3, msg=f"{name} {horizon}")

    def test_coalescence_probability_close_to_msprime(self):
        times = np.linspace(0.0, 2e5, 2001)
        for name in ("bottleneck", "zigzag"):
            model = config.MODELS[name]
            with warnings.catch_warnings():
                # msprime integrates the rate numerically and warns about its steps.
                warnings.simplefilter("ignore", UserWarning)
                _, p_not = demography.to_msprime(model).debug().coalescence_rate_trajectory(
                    times, lineages={demography.POPULATION: 2}
                )
            eps = demography.epochs(model)
            idx = slice(0, None, 50)
            mine = np.array([1 - demography.coalescence_cdf(eps, t) for t in times[idx]])
            self.assertLess(float(np.max(np.abs(mine - p_not[idx]))), 5e-3, name)

    @unittest.skipUnless(has_modules("stdpopsim"), "stdpopsim not installed")
    def test_zigzag_equals_stdpopsim(self):
        res = demography.check_zigzag_against_stdpopsim()
        self.assertEqual(res["epochs"], 7)
        self.assertLess(res["max_rel_diff"], 1e-9)


if __name__ == "__main__":
    unittest.main()
