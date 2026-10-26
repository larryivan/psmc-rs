"""Tests for experiment E6: its jobs, its checks and (with the tools present) a tiny run end to end.

The analyses have their own tests (test_pipeline_e6_analysis.py); the figures are FIG's (test_figures_*.py).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import ROOT, assert_ignore_rules, fake_ctx, in_git_work_tree, read_csv  # noqa: E402

from experiment.pipeline import config, runner  # noqa: E402
from experiment.pipeline.experiments import e1, e6  # noqa: E402


# ---------------------------------------------------------------------------
# Jobs (no numerical packages needed)
# ---------------------------------------------------------------------------


class Jobs(unittest.TestCase):
    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.ctx = fake_ctx(config.SMOKE, self.runs, log=lambda msg: None)
        self.e1_jobs = e1.jobs(self.ctx)
        self.jobs = e6.jobs(self.ctx)
        self.by_id = {j.id: j for j in self.jobs}
        self.graph = runner.Graph(self.e1_jobs + self.jobs, self.runs)

    def test_parts_and_their_dependencies_on_e1(self):
        ids = set(self.by_id)
        models = config.SMOKE.e1.models
        for part in ("s1", "s2", "s3"):
            self.assertIn(f"E6/{part}/report", ids)
        self.assertFalse(any(j.startswith("E6/s4") for j in ids))  # S4 is E3's own tables
        self.assertFalse(any(j.startswith("E6/s5") for j in ids))  # E2's qc jobs check E2's genomes
        self.assertFalse(any(j.endswith("/figure") for j in ids))  # the figures are FIG's
        s1 = set(self.graph.deps["E6/s1/report"])
        self.assertEqual(s1, {j.id for j in self.e1_jobs if j.id.startswith(("E1/fit-c/", "E1/fit-rs/", "E1/sim/"))})
        self.assertEqual(self.graph.deps["E6/s2/convert-vcf/bottleneck/rep01"], ["E1/sim/bottleneck/rep01"])
        self.assertEqual(sorted(self.graph.deps["E6/s3/decode/bottleneck/rep01"]),
                         ["E1/fit-rs/bottleneck/rep01", "E1/sim/bottleneck/rep01"])
        self.assertEqual(len([j for j in ids if j.startswith("E6/s2/fit-")]), 2 * len(models))

    def test_s2_reads_the_vcf_in_mask_mode_and_fits_with_e1s_settings(self):
        conv = self.by_id["E6/s2/convert-vcf/constant/rep01"]
        self.assertEqual(conv.command[1], "convert")
        self.assertIn("--multihetsep", conv.command)
        self.assertEqual(conv.command[conv.command.index("--vcf-mode") + 1], "mask")
        fit = self.by_id["E6/s2/fit-multihetsep/constant/rep01"]
        e1_fit = self.graph.jobs["E1/fit-rs/constant/rep01"]
        # E1's options after INPUT OUTPUT N_ITER, up to --psmc-out FILE; then the input format.
        self.assertEqual(fit.command[3:-3], e1_fit.command[3:-1])
        self.assertEqual(fit.command[-2:], ["--input-format", "mhs"])
        self.assertEqual(fit.settings["inference"], e1_fit.settings["inference"])

    def test_s3_decodes_e1s_fit_without_iterating(self):
        dec = self.by_id["E6/s3/decode/bottleneck/rep01"]
        e1lay = e1.Layout(self.runs / "E1")
        self.assertEqual(dec.command[3], "0")
        self.assertEqual(dec.command[dec.command.index("--init-params") + 1], str(e1lay.rs_json("bottleneck", 1)))
        self.assertNotIn("--t-max", dec.command)  # psmc-rs refuses it with --init-params
        self.assertIn("--tmrca-out", dec.command)
        truth = self.by_id["E6/s3/truth/bottleneck/rep01"]
        self.assertIn("--tmrca-only", truth.command)
        self.assertEqual(truth.seeds["chr1"]["ancestry"], config.seed("E1", "bottleneck", 1, "ancestry/chr1"))

    def test_production_plan(self):
        jobs = e6.jobs(fake_ctx(config.PRODUCTION, self.runs, log=lambda msg: None))
        self.assertEqual(len([j for j in jobs if j.id.startswith("E6/s2/fit-")]), 8)
        graph = runner.Graph(e1.jobs(fake_ctx(config.PRODUCTION, self.runs)) + jobs, self.runs)
        plan = runner.plan(graph, config.Budget())
        self.assertLess(plan["E6"]["cpu_hours"], 1.0)
        self.assertTrue(all(j.mem_gb <= config.Budget().mem_gb for j in jobs))


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    """What git keeps of experiment/runs/E6: results and records, not the regenerable inputs."""

    def test_layout_paths_fall_on_the_right_side_of_the_ignore_rules(self):
        lay = e6.Layout(ROOT / "experiment" / "runs" / "E6")
        assert_ignore_rules(self, kept=(
            lay.s2_fit("constant", 1, "vcf"), lay.s2_fit_psmc("constant", 1, "multihetsep"),
            lay.s3_decode("bottleneck", 1), lay.table("s3", "track"),
            lay.root / "figures" / "figS3_tmrca_decoding.pdf", lay.root / "provenance" / "s3" / "report.json",
        ), ignored=(
            lay.s2_vcf_psmcfa("constant", 1), lay.s2_multihetsep("constant", 1), lay.s2_mhs_psmcfa("zigzag", 1),
            lay.s3_true_track("bottleneck", 1), lay.s3_posterior("bottleneck", 1),
            lay.s2_fit("constant", 1, "vcf").with_suffix(".html"), lay.root / "logs" / "runner.log",
        ))


class Checks(unittest.TestCase):
    def test_s2_problems_flag_a_vcf_that_differs(self):
        ok_bins = [{"model": "m", "format": "vcf", "sequence": "all", "identical": True, "bins_differing": 0,
                    "bins_format": 5, "bins_psmcfa": 5},
                   {"model": "m", "format": "multihetsep", "sequence": "all", "identical": False, "bins_differing": 0,
                    "bins_extra_at_end": 0, "tail_is_after_last_het": True}]
        ok_fits = [{"model": "m", "format": "vcf", "parameters_identical": True, "loglikes_identical": True}]
        self.assertEqual(e6.s2_problems(ok_bins, ok_fits), [])
        bad = [dict(ok_bins[0], identical=False, bins_differing=3), ok_bins[1]]
        self.assertEqual(len(e6.s2_problems(bad, [dict(ok_fits[0], parameters_identical=False)])), 2)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E6 in the tiny runs of every experiment (tiny_runs): every part's tables and checks."""

    def test_tiny_profile(self):
        runs = tiny_runs.runs_dir()
        lay = e6.Layout(runs / "E6")
        for part in e6.PARTS:
            for p in lay.tables(part):
                self.assertTrue(p.exists(), p)
        bins = {(r["format"], r["sequence"]): r for r in read_csv(lay.table("s2", "bins"))}
        self.assertEqual(bins[("vcf", "all")]["identical"], "True")
        self.assertEqual(bins[("multihetsep", "all")]["tail_is_after_last_het"], "True")
        fits = {r["format"]: r for r in read_csv(lay.table("s2", "fits"))}
        self.assertEqual(fits["vcf"]["parameters_identical"], "True")
        s3 = {r["bins"]: r for r in read_csv(lay.table("s3", "summary"))}
        self.assertEqual(s3["all"]["state_index_mismatches"], "0")
        self.assertEqual(s3["all"]["decoded_model_is_e1_fit"], "True")
        self.assertGreater(float(s3["all"]["pearson_log10_mean"]), 0.3)
        self.assertTrue(all("/Users/" not in p.read_text() for part in e6.PARTS for p in lay.tables(part)))


if __name__ == "__main__":
    unittest.main()
