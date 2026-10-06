"""Tests for the tables of experiment/validation/bin_callable.py (not the simulations it runs)."""

from __future__ import annotations

import unittest

from experiment.pipeline import config
from experiment.validation import bin_callable as bc


def row(rep, mask, route, mode, theta, model="callable-length"):
    return {
        "replicate": rep, "mask": mask, "route": route, "mode": mode, "model": model,
        "mean_callable": 0.7, "theta": theta, "theta_vs_complete": theta, "rmse_vs_truth": 0.3 + theta / 10,
        "rmse_vs_complete": 0.1, "log_ratio_vs_truth": -0.1, "final_loglike": -1.0, "seconds": 5.0,
    }


class Tables(unittest.TestCase):
    def test_summary_gives_the_range_of_theta_over_replicates(self):
        rows = [row(r, "dispersed30", "vcf", "auto", t) for r, t in ((1, 0.98), (2, 0.96), (3, 0.97))]
        rows += [row(r, "dispersed30", "multihetsep", "auto", 0.68, "full-bins") for r in (1, 2, 3)]
        summary = bc.summarize(rows)
        self.assertEqual([(s["route"], s["mode"]) for s in summary], [("vcf", "auto"), ("multihetsep", "auto")])
        vcf = summary[0]
        self.assertAlmostEqual(vcf["theta_vs_complete"], 0.97)
        self.assertEqual((vcf["theta_vs_complete_min"], vcf["theta_vs_complete_max"]), (0.96, 0.98))
        self.assertEqual(summary[1]["model"], "full-bins")
        table = bc.markdown(summary)
        self.assertIn("| dispersed30 | vcf | auto | callable-length | 0.700 | 0.970 (0.960-0.980) |", table)
        self.assertIn("| multihetsep | auto | full-bins |", table)

    def test_multihetsep_is_fitted_with_and_without_its_counts(self):
        # auto fits C psmc's model on multihetsep (exact counts only); approximate fits its counts.
        self.assertEqual(bc.MHS_MODES, ("auto", "approximate"))
        self.assertEqual(bc.VCF_MODES, ("auto", "ignore"))
        self.assertIs(bc.MODEL, config.BOTTLENECK)
        self.assertIn("expansion", config.MODELS)


if __name__ == "__main__":
    unittest.main()
