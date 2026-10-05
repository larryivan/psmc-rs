"""Tests for experiment E4 (input preparation): its jobs, its tables, git's view of its runs, a tiny run end to end."""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, paths, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e4  # noqa: E402

HAS_RUN_DEPS = all(importlib.util.find_spec(m) is not None for m in ("msprime", "psutil", "numpy", "matplotlib"))
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})
FAKE_TOOLS = {
    "rs02": FAKE_RS,
    "c": tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64}),
    "fq2psmcfa": tools.Tool("fq2psmcfa", Path("/x/fq2psmcfa"), {"sha256": "1" * 64}),
    "generate_multihetsep.py": tools.Tool("generate_multihetsep.py", Path("/x/gm.py"), {"sha256": "2" * 64}),
}

def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        self.ctx = Context(config.SMOKE, self.runs, _tools=dict(FAKE_TOOLS))
        self.jobs = e4.jobs(self.ctx)
        self.by_id = {j.id: j for j in self.jobs}
        self.graph = runner.Graph(self.jobs, self.runs)

    def test_job_counts(self):
        cfg = config.SMOKE.e4
        masks = len(e4.mask_names(cfg))
        per_mask = 6 + 2 * len(cfg.routes)  # mask, consensus, 3 converts, bins; a fit and a qc per route
        crosschecks = masks * min(cfg.multihetsep_crosscheck_replicates, cfg.replicates)
        self.assertEqual(len(self.jobs), cfg.replicates * (1 + masks * per_mask) + crosschecks + 2)
        self.assertEqual(e4.mask_names(cfg)[0], config.E4_CONTROL_MASK)

    def test_routes_read_what_they_should(self):
        exact = self.by_id["E4/fit/dispersed30/exact/rep01"]
        nomask = self.by_id["E4/fit/dispersed30/nomask/rep01"]
        self.assertEqual(exact.inputs[0], nomask.inputs[0])  # the same masked VCF
        self.assertIn("--vcf-mask", exact.command)
        self.assertNotIn("--vcf-mask", nomask.command)
        self.assertEqual(self.graph.deps["E4/fit/dispersed30/multihetsep/rep01"], ["E4/convert/dispersed30/rep01"])
        self.assertEqual(self.graph.deps["E4/fit/dispersed30/fq2psmcfa/rep01"], ["E4/consensus/dispersed30/rep01"])
        self.assertEqual(self.graph.deps["E4/mask/dispersed30/rep01"], ["E4/sim/rep01"])
        for j in (exact, nomask):
            self.assertEqual(j.command[3], str(config.SMOKE.e4.inference.n_iter))

    def test_bins_job_asserts_the_fq2psmcfa_route(self):
        bins = self.by_id["E4/bins/clustered10/rep01"]
        self.assertIn("--assert-identical", bins.command)
        self.assertEqual(bins.command[bins.command.index("--assert-identical") + 1], "fq2psmcfa")
        self.assertEqual(set(self.graph.deps["E4/bins/clustered10/rep01"]), {
            "E4/convert/clustered10/rep01", "E4/consensus/clustered10/rep01",
            "E4/convert/clustered10/multihetsep/rep01", "E4/convert/clustered10/nomask/rep01"})

    def test_seeds_are_recorded(self):
        mask = self.by_id["E4/mask/clustered30/rep01"]
        self.assertEqual(mask.seeds["chr2"], config.seed("E4", "bottleneck", 1, "mask/clustered30/chr2"))
        self.assertIsNone(self.by_id["E4/mask/nomissing/rep01"].seeds["chr1"])
        self.assertNotIn("--missing-fraction", self.by_id["E4/mask/nomissing/rep01"].command)
        cons = self.by_id["E4/consensus/clustered30/rep01"]
        self.assertEqual(cons.seeds["chr1"], config.seed("E4", "bottleneck", 1, "reference/chr1"))
        sim = self.by_id["E4/sim/rep01"]
        self.assertEqual(sim.seeds["chr1"]["ancestry"], config.seed("E4", "bottleneck", 1, "ancestry/chr1"))

    def test_report_and_figure_wait_for_everything(self):
        report_deps = set(self.graph.deps["E4/report"])
        self.assertTrue({j.id for j in self.jobs if j.id.startswith(("E4/fit/", "E4/qc/", "E4/crosscheck/", "E4/bins/"))}
                        <= report_deps)
        self.assertEqual(self.graph.deps["E4/figure"], ["E4/report"])

    def test_production_plan(self):
        jobs = e4.jobs(Context(config.PRODUCTION, self.runs, _tools=dict(FAKE_TOOLS)))
        self.assertEqual(sum(1 for j in jobs if j.id.startswith("E4/fit/")), 5 * 5 * 4)
        self.assertEqual(sum(1 for j in jobs if j.id.startswith("E4/crosscheck/")), 5)
        plan = runner.plan(runner.Graph(jobs, self.runs), config.Budget())
        self.assertGreater(plan["E4"]["cpu_hours"], 1.0)
        self.assertLess(plan["E4"]["finish_hours"], 3.0)
        self.assertTrue(all(j.mem_gb <= 1.5 for j in jobs))


class Summary(unittest.TestCase):
    def test_means_and_qc_codes_over_replicates(self):
        cfg = replace(config.E4(), replicates=2, routes=("exact",), masks=())
        rows = [
            {"mask": "nomissing", "route": "exact", "missing_fraction": 0.0, "differing_bin_fraction": 0.0,
             "missing_bin_fraction": 0.0, "theta_rel_exact": 0.0, "theta_rel_complete": 0.0,
             "rmse_log10_ne_vs_truth": rmse, "rmse_log10_ne_vs_truth_fixed": rmse, "rmse_log10_ne_vs_exact": 0.0,
             "identical_to_exact": True, "qc_warnings": warn, "qc_info": ""}
            for rmse, warn in ((0.1, "A;B"), (0.3, "B"))
        ]
        (row,) = e4.summary_rows(cfg, rows, [])
        self.assertAlmostEqual(row["rmse_truth_mean"], 0.2)
        self.assertEqual((row["qc_warnings_all"], row["qc_warnings_any"]), ("B", "A;B"))
        self.assertEqual((row["fits_identical_to_exact"], row["structure"]), (2, "none"))

    def test_bases_per_called_bin_from_qc(self):
        qc = {"totals": {"bins": 1000, "missing_bins": 10, "called_bases": 69300}}
        self.assertAlmostEqual(e4.bases_per_called_bin(qc), 70.0)
        self.assertIsNone(e4.bases_per_called_bin({"totals": {"bins": 1000, "missing_bins": 0, "called_bases": None}}))
        self.assertIsNone(e4.bases_per_called_bin({"totals": {"bins": 5, "missing_bins": 5, "called_bases": 0}}))

    def test_rescaled_routes_multiply_times_and_sizes_by_bin_size_over_callable_bases(self):
        from experiment.pipeline import results

        cfg = config.E4()
        fit = results.Fit(theta=0.03, rho=0.006, t_max=15.0, n=3, lambdas=(1.0, 2.0, 0.5, 1.0))
        qc70 = {"totals": {"bins": 100, "missing_bins": 0, "called_bases": 7000}}
        qc_psmcfa = {"totals": {"bins": 100, "missing_bins": 0, "called_bases": None}}

        def route_fit(route, qc):
            sc = cfg.scaling
            return e4.RouteFit("dispersed30", route, 1, {}, fit, fit.curve(sc.mu, sc.gen_years, 100), qc, {})

        fits = {"exact": route_fit("exact", qc70), "fq2psmcfa": route_fit("fq2psmcfa", qc_psmcfa),
                "multihetsep": route_fit("multihetsep", qc_psmcfa)}
        variants = {v.route: v for v in e4.route_variants(cfg, fits)}
        self.assertEqual(set(variants), {"exact", "fq2psmcfa", "multihetsep", "exact_rescaled"})
        v = variants["exact_rescaled"]
        self.assertAlmostEqual(v.scale, 100 / 70)
        raw = fits["exact"].curve
        for a, b in zip(v.curve.years, raw.years):
            self.assertAlmostEqual(a, b * 100 / 70, delta=1e-9 * max(b, 1.0))
        for a, b in zip(v.curve.ne, raw.ne):
            self.assertAlmostEqual(a, b * 100 / 70, delta=1e-9 * b)
        self.assertEqual(e4.table_routes(cfg), ("exact", "fq2psmcfa", "multihetsep", "nomask",
                                                "exact_rescaled", "multihetsep_rescaled"))

    def test_same_fit(self):
        a = {"theta": 1.0, "rho": 0.2, "t_max": 15.0, "lam": [1, 2], "loglike": [-3.0], "n_steps": 63, "pattern": "p"}
        self.assertTrue(e4.same_fit(a, dict(a)))
        self.assertFalse(e4.same_fit(a, {**a, "lam": [1, 2.0000001]}))


def git_ignores(rel_path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel_path], cwd=ROOT).returncode == 0


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "not a git checkout")
class GitLayout(unittest.TestCase):
    """Records, fits, QC reports, tables and figures are kept; genomes, masks and route inputs are not."""

    def test_small_results_are_tracked(self):
        for p in ("truth/rep01.json", "masks/dispersed30/rep01.json", "consensus/dispersed30/rep01.json",
                  "bins/dispersed30/rep01.json", "crosscheck/dispersed30/rep01.json",
                  "fits/dispersed30/multihetsep/rep01.json", "qc/dispersed30/nomask/rep01.json",
                  "provenance/fit/dispersed30/exact/rep01.json", "tables/E4_summary.csv", "figures/E4_fig5.pdf"):
            self.assertFalse(git_ignores(f"experiment/runs/E4/{p}"), p)

    def test_regenerable_inputs_are_ignored(self):
        for p in ("sims/rep01/genome.vcf.gz", "sims/rep01/dispersed30/callable.bed.gz",
                  "sims/rep01/dispersed30/variants.vcf.gz", "sims/rep01/dispersed30/fq2psmcfa.psmcfa",
                  "sims/rep01/dispersed30/input.multihetsep", "sims/rep01/dispersed30/bins/exact.psmcfa",
                  "fits/dispersed30/exact/rep01.html", "logs/attempts.jsonl", "figures/E4_fig5.pdf.partial"):
            self.assertTrue(git_ignores(f"experiment/runs/E4/{p}"), p)


def run_tools_present() -> bool:
    try:
        for name in ("fq2psmcfa", "generate_multihetsep.py"):
            tools.utility(name)
    except tools.ToolError:
        return False
    return (paths.ROOT / "target" / "release" / "psmc-rs").exists() and shutil.which("bgzip") is not None


@unittest.skipUnless(HAS_RUN_DEPS and run_tools_present(), "needs msprime, psutil, matplotlib, fq2psmcfa, "
                     "generate_multihetsep.py, psmc-rs and bgzip")
class EndToEnd(unittest.TestCase):
    """A tiny E4 through the runner: one replicate of 2 x 300 kb, two masks plus the control."""

    def test_tiny_e4(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            graph = runner.Graph(e4.jobs(Context(config.TINY, runs)), runs)
            daemon = runner.Daemon(graph, runs, config.Budget(cores=4, mem_gb=8), runner.Logger(runs), {})
            self.assertEqual(daemon.run(), 0, (runs / "_runner" / "runner.log").read_text()[-3000:])
            lay = e4.Layout(runs / "E4")
            summary = {(r["mask"], r["route"]): r for r in read_csv(lay.table("summary"))}
            self.assertEqual(len(summary), 3 * (4 + 2))  # 4 routes and 2 rescaled ones per mask
            for mask in ("nomissing", "clustered30", "dispersed30"):
                self.assertEqual(summary[(mask, "fq2psmcfa")]["fits_identical_to_exact"], "1")
                self.assertIn("VARIANTS_ONLY_NO_MASK", summary[(mask, "nomask")]["qc_warnings_all"])
            self.assertEqual(summary[("nomissing", "nomask")]["fits_identical_to_exact"], "1")
            # The dispersed mask leaves bins called with fewer callable bases; the rescaled route knows how many.
            self.assertLess(float(summary[("dispersed30", "exact_rescaled")]["bases_per_called_bin_mean"]), 90.0)
            self.assertGreater(float(summary[("dispersed30", "exact_rescaled")]["scale_mean"]), 1.1)
            self.assertAlmostEqual(float(summary[("nomissing", "exact_rescaled")]["scale_mean"]), 1.0)
            bins = {(r["mask"], r["route"]): r for r in read_csv(lay.table("bins"))}
            self.assertEqual(bins[("clustered30", "fq2psmcfa")]["bytes_identical"], "True")
            self.assertGreater(int(bins[("clustered30", "nomask")]["N_T"]), 0)
            self.assertTrue(all(r["identical"] == "True" for r in read_csv(lay.table("crosscheck"))))
            self.assertTrue(lay.figure("E4_fig5.pdf").exists())


if __name__ == "__main__":
    unittest.main()
