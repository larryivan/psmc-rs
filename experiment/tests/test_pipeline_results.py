"""Tests for experiment.pipeline.results, against C psmc's reference outputs in tests/data."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import results  # noqa: E402

C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class Patterns(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(results.pattern_intervals("4+25*2+4+6"), 64)
        self.assertEqual(results.pattern_free("4+25*2+4+6"), 28)
        self.assertEqual(results.pattern_intervals("4+5*3+4"), 23)
        self.assertEqual(results.pattern_free("4+5*3+4"), 7)

    def test_expand(self):
        self.assertEqual(results.expand_lambdas([1, 2, 3, 4], "1+2*2+1", 6), [1, 2, 2, 3, 3, 4])
        self.assertEqual(results.expand_lambdas([5, 6], None, 2), [5.0, 6.0])
        with self.assertRaises(ValueError):
            results.expand_lambdas([1, 2], "1+2*2+1", 6)

    def test_boundaries_are_c_psmcs(self):
        b = results.boundaries(15.0, 63)
        self.assertEqual(len(b), 64)
        self.assertEqual(b[0], 0.0)
        self.assertEqual(b[-1], 15.0)
        beta = math.log(1 + 15 / 0.1) / 63
        self.assertAlmostEqual(b[10], 0.1 * (math.exp(10 * beta) - 1), places=15)


class CReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.psmc = results.parse_psmc(C_REF)

    def test_header(self):
        h = self.psmc.header
        self.assertEqual(h["pattern"], "4+25*2+4+6")
        self.assertEqual((h["n"], h["n_free"], h["n_iterations"], h["max_t"], h["tr_ratio"]), (63, 28, 5, 15.0, 4.0))
        self.assertEqual((h["n_seqs"], h["sum_L"], h["sum_n"]), (3, 150947, 7373))
        self.assertTrue(h["version"].startswith("0.6.5"))

    def test_rounds(self):
        rounds = self.psmc.complete_rounds()
        self.assertEqual([r.round for r in rounds], [0, 1, 2, 3, 4, 5])
        self.assertEqual(rounds[0].lk, 0.0)  # placeholder: no E-step yet
        self.assertEqual(len(rounds[3].rs), 64)
        self.assertIsNotNone(rounds[3].n_recomb)
        ll = self.psmc.loglikes()
        self.assertEqual(sorted(ll), [0, 1, 2, 3, 4])
        self.assertEqual(ll[0], rounds[1].lk)

    def test_fit_from_pa_line_reproduces_rs_lines(self):
        for rd in self.psmc.complete_rounds():
            fit = rd.fit()
            self.assertEqual(fit.n, 63)
            self.assertAlmostEqual(fit.theta, rd.theta, places=6)
            for (k, t_k, lam_k, *_), t, lam in zip(rd.rs, fit.boundaries(), fit.lambdas):
                self.assertAlmostEqual(t, t_k, delta=5e-7)
                self.assertAlmostEqual(lam / lam_k, 1.0, delta=1e-5)

    def test_round_zero_is_the_start(self):
        first = self.psmc.complete_rounds()[0]
        self.assertEqual((first.round, first.t_max), (0, 15.0))
        self.assertNotIn(-1, self.psmc.loglikes())  # its LK is a placeholder, not a log-likelihood


class RsJson(unittest.TestCase):
    def params(self):
        free = [1.0 + 0.1 * i for i in range(28)]
        return {"theta": 0.002, "rho": 0.0004, "lam": free, "t_max": 15.0, "n_steps": 63, "pattern": "4+25*2+4+6",
                "n_iter": 3, "loglike": [-10.0, -9.0, -8.5, -8.4]}

    def test_fit_and_loglikes(self):
        fit = results.fit_from_rs(self.params())
        self.assertEqual(len(fit.lambdas), 64)
        self.assertEqual(fit.lambdas[:5], (1.0, 1.0, 1.0, 1.0, 1.1))
        self.assertEqual(results.rs_loglikes(self.params()), [-10.0, -9.0, -8.5, -8.4])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text(json.dumps(self.params()))
            self.assertEqual(results.fit_from_rs(results.read_rs_json(p)), fit)

    @unittest.skipUnless(HAS_NUMPY, "numpy not installed")
    def test_curve_scaling(self):
        fit = results.fit_from_rs(self.params())
        curve = fit.curve(mu=2.5e-8, gen_years=25, bin_size=100)
        n0 = 0.002 / (4 * 2.5e-8 * 100)
        self.assertAlmostEqual(fit.n0(2.5e-8, 100), n0)
        self.assertAlmostEqual(curve.years[-1], 2 * n0 * 15.0 * 25)
        self.assertAlmostEqual(curve.ne[0], n0)
        # Step lookup: left-closed intervals, the last one open-ended.
        got = curve.at([0.0, curve.years[4] - 1e-9, curve.years[4], 1e12])
        self.assertEqual(list(got), [curve.ne[0], curve.ne[3], curve.ne[4], curve.ne[-1]])


if __name__ == "__main__":
    unittest.main()
