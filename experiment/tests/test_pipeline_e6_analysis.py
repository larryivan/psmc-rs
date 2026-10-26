"""Tests for experiment.pipeline.experiments.e6_analysis: the analyses of E6 (S1-S3) on small known inputs."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import C_REF, has_modules  # noqa: E402

from experiment.pipeline import config, demography, results, simulate  # noqa: E402

HAS_NUMERIC = has_modules("numpy", "pandas", "scipy")

if HAS_NUMERIC:
    import numpy as np
    import pandas as pd

    from experiment.pipeline.experiments import e6_analysis as A


@unittest.skipUnless(HAS_NUMERIC, "numpy, pandas and scipy needed")
class Convergence(unittest.TestCase):
    """C psmc's reference output read as both tools: the traces must agree exactly."""

    def setUp(self):
        self.c = results.parse_psmc(C_REF)
        ll = self.c.loglikes()
        self.params = {"loglike": [ll[i] for i in range(5)] + [ll[4] + 0.5]}
        self.rows = A.trace_rows("bottleneck", 1, self.c, self.params, self.c, 5)

    def test_identical_traces_agree_exactly(self):
        self.assertEqual([r["iteration"] for r in self.rows], list(range(6)))
        self.assertIsNone(self.rows[-1]["loglike_c"])  # C psmc never evaluates its final model
        self.assertEqual(self.rows[-1]["loglike_rs"], self.params["loglike"][-1])
        self.assertTrue(all(r["loglike_rel_diff"] == 0.0 for r in self.rows[:-1]))
        self.assertEqual(self.rows[4]["gap_c"], 0.0)
        self.assertEqual(self.rows[0]["theta_c"], 0.050078)
        agree = A.agreement_summary("bottleneck", self.rows, 5)
        self.assertEqual(agree["loglike_rel_diff_max"], 0.0)
        self.assertEqual(agree["theta_rel_diff_max"], 0.0)
        self.assertEqual(agree["iterations_compared"], 5)

    def test_convergence_summary(self):
        s = A.convergence_summary("bottleneck", "c", self.rows, 5, within=1.0, tolerance=1e-3)
        self.assertEqual(s["replicates"], 1)
        self.assertEqual(s["last_evaluated_iteration"], 4)
        ll = self.c.loglikes()
        self.assertAlmostEqual(s["gain_last_common_median"], ll[4] - ll[3])
        self.assertEqual(s["replicates_monotone"], int(all(ll[i + 1] >= ll[i] - 1e-3 for i in range(4))))
        first = next(i for i in range(5) if ll[4] - ll[i] <= 1.0)
        self.assertEqual(s["iterations_to_within_median"], first)
        rs = A.convergence_summary("bottleneck", "rs", self.rows, 5, within=1.0, tolerance=1e-3)
        self.assertEqual(rs["last_evaluated_iteration"], 5)

    def test_a_drop_in_log_likelihood_is_reported(self):
        params = {"loglike": [-10.0, -8.0, -8.5, -8.4, -8.3, -8.2]}
        rows = A.trace_rows("m", 1, self.c, params, self.c, 5)
        s = A.convergence_summary("m", "rs", rows, 5, within=1.0, tolerance=1e-3)
        self.assertEqual((s["replicates_monotone"], s["max_decrease"]), (0, 0.5))


@unittest.skipUnless(HAS_NUMERIC, "numpy, pandas and scipy needed")
class Formats(unittest.TestCase):
    def test_read_psmcfa_reads_what_the_simulation_writes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.psmcfa"
            bins = "T" * 70 + "K" + "T" * 9
            p.write_text(simulate.psmcfa_record("chr1", bins) + simulate.psmcfa_record("chr2 extra", "KT"))
            self.assertEqual(A.read_psmcfa(p), {"chr1": bins, "chr2": "KT"})

    def test_compare_bins_counts_the_tail_after_the_last_het(self):
        ref = {"a": "TTKTTT", "b": "KTTT"}
        rows = A.compare_bins("m", "multihetsep", ref, {"a": "TTK", "b": "K"})
        total = rows[-1]
        self.assertEqual(total["sequence"], "all")
        self.assertEqual((total["bins_differing"], total["bins_missing_at_end"], total["bins_after_last_het"]), (0, 6, 6))
        self.assertTrue(total["tail_is_after_last_het"])
        self.assertFalse(total["identical"])
        bad = A.compare_bins("m", "vcf", ref, {"a": "TTKTKT", "b": "KTTT"})[-1]
        self.assertEqual(bad["bins_differing"], 1)
        self.assertFalse(bad["identical"])
        short = A.compare_bins("m", "multihetsep", {"a": "TKTT"}, {"a": "T"})[-1]
        self.assertFalse(short["tail_is_after_last_het"])

    def test_fit_rows_compare_with_the_psmcfa_fit(self):
        last = results.parse_psmc(C_REF).last()
        base = {"theta": last.pa["theta"], "rho": last.pa["rho"], "t_max": last.pa["t_max"], "n_steps": 63,
                "pattern": "4+25*2+4+6", "lam": list(last.pa["lambdas"]), "loglike": [-10.0, -9.0],
                "input": {"format": "psmcfa", "bins": 100}, "qc": {"warnings": ["TOO_SHORT"], "notes": []}}
        doubled = dict(base, lam=[2 * x for x in base["lam"]], loglike=[-10.0, -8.5])
        fits = {"psmcfa": A.FormatFit("psmcfa", base), "vcf": A.FormatFit("vcf", dict(base)),
                "multihetsep": A.FormatFit("multihetsep", doubled)}
        epochs = demography.epochs(config.BOTTLENECK)
        window = np.logspace(4, 6, 50)
        rows = {r["format"]: r for r in A.fit_rows("bottleneck", fits, epochs, window, 2.5e-8, 25.0, 100)}
        self.assertTrue(rows["vcf"]["parameters_identical"] and rows["vcf"]["loglikes_identical"])
        self.assertEqual(rows["vcf"]["rmse_log10_ne_vs_psmcfa_per_interval"], 0.0)
        self.assertAlmostEqual(rows["multihetsep"]["rmse_log10_ne_vs_psmcfa_per_interval"], math.log10(2), places=12)
        self.assertAlmostEqual(rows["multihetsep"]["max_abs_log10_ne_vs_psmcfa_per_interval"], math.log10(2),
                               places=12)
        self.assertAlmostEqual(rows["multihetsep"]["lambda_max_abs_rel_diff"], 1.0)
        self.assertEqual(rows["multihetsep"]["loglike_final_diff"], 0.5)
        self.assertEqual(rows["vcf"]["qc_warnings"], "TOO_SHORT")
        curve = A.curve_rows("bottleneck", "vcf", fits["vcf"].fit, 2.5e-8, 25.0, 100)
        self.assertEqual(len(curve), 64)
        truth = A.truth_rows("bottleneck", epochs, [1e3, 1e5], 25.0)
        self.assertEqual([r["ne"] for r in truth], [20000.0, 5000.0])


@unittest.skipUnless(HAS_NUMERIC, "numpy, pandas and scipy needed")
class Tmrca(unittest.TestCase):
    def test_bin_tmrca_is_the_length_weighted_mean(self):
        mean, n = A.bin_tmrca([(0, 150, 10.0), (150, 300, 20.0)], 300, 100)
        np.testing.assert_allclose(mean, [10.0, 15.0, 20.0])
        np.testing.assert_array_equal(n, [1, 2, 1])
        mean, n = A.bin_tmrca([(0, 250, 7.0)], 250, 100)  # a last partial bin
        np.testing.assert_allclose(mean, [7.0, 7.0, 7.0])
        with self.assertRaises(ValueError):
            A.bin_tmrca([(0, 100, 1.0), (120, 300, 2.0)], 300, 100)

    def decoded(self, true, mean, lo, hi, edges=(0.0, 10.0, 100.0, 1000.0)):
        k = len(true)
        post = pd.DataFrame({
            "seq_id": np.zeros(k, dtype=int), "seq_bin": np.arange(k), "obs": ["T"] * k,
            "map_state": A.state_of(mean, edges), "tmrca_map_years": mean, "tmrca_mean_years": mean,
            "tmrca_q025_years": lo, "tmrca_q975_years": hi,
        })
        return A.align(post, [(np.asarray(true, dtype=float), np.ones(k, dtype=int))], edges, gen_years=1.0)

    def test_coverage_by_points_and_by_state_intervals(self):
        # Truth 50: inside the intervals [10, 1000) of the states of 20 and 40, not between the points 20 and 40.
        d = self.decoded([5.0, 50.0, 500.0], [5.0, 30.0, 500.0], [2.0, 20.0, 200.0], [8.0, 40.0, 800.0])
        np.testing.assert_array_equal(d.covered_point(), [True, False, True])
        np.testing.assert_array_equal(d.covered_states(), [True, True, True])
        self.assertEqual(A.state_mismatches(d), 0)
        s = A.tmrca_summary(d, "all")
        self.assertAlmostEqual(s["coverage95_point"], 2 / 3)
        self.assertEqual(s["coverage95_states"], 1.0)
        self.assertEqual(s["map_state_is_true_state"], 1.0)

    def test_a_map_time_outside_its_state_is_a_mismatch(self):
        d = self.decoded([5.0], [5.0], [2.0], [8.0])
        d.map_state = np.array([2])
        self.assertEqual(A.state_mismatches(d), 1)

    def test_perfect_posterior(self):
        rng = np.random.default_rng(1)
        true = 10 ** rng.uniform(0.5, 2.9, 500)
        d = self.decoded(true, true, true / 2, true * 2)
        s = A.tmrca_summary(d, "all")
        self.assertAlmostEqual(s["pearson_log10_mean"], 1.0)
        self.assertAlmostEqual(s["spearman_mean"], 1.0)
        self.assertEqual(s["rmse_log10_mean"], 0.0)
        cal = A.calibration_rows(d, "true", 10)
        self.assertEqual(sum(r["n_bins"] for r in cal), 500)
        self.assertTrue(all(r["coverage95_point"] == 1.0 for r in cal))
        dens = A.density_rows(d, 20)
        self.assertEqual(sum(r["n_bins"] for r in dens), 500)
        track = A.track_rows(d, 0, "chr1", 10, 5, 100)
        self.assertEqual([r["start_bp"] for r in track], [1000, 1100, 1200, 1300, 1400])

    def test_alignment_refuses_mismatched_lengths(self):
        post = pd.DataFrame({c: [0] for c in A.POSTERIOR_COLUMNS})
        with self.assertRaises(ValueError):
            A.align(post, [(np.ones(2), np.ones(2, dtype=int))], [0.0, 1.0], 1.0)


if __name__ == "__main__":
    unittest.main()
