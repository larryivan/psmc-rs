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

from experiment.pipeline import config, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e4  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

HAS_RUN_DEPS = all(importlib.util.find_spec(m) is not None for m in ("msprime", "psutil", "numpy", "matplotlib"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
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


def flag(command, name):
    """The value of option ``name`` in ``command`` (None when absent)."""
    command = [str(c) for c in command]
    return command[command.index(name) + 1] if name in command else None


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
        per_mask = 6 + len(cfg.routes)  # mask, consensus, 3 converts, bins; a fit per route
        crosschecks = masks * min(cfg.multihetsep_crosscheck_replicates, cfg.replicates)
        self.assertEqual(len(self.jobs), cfg.replicates * (1 + masks * per_mask) + crosschecks + 1)
        self.assertEqual(e4.mask_names(cfg)[0], config.E4_CONTROL_MASK)
        self.assertEqual(cfg.routes, e4.ROUTES)
        self.assertFalse(any(j.id.startswith("E4/qc/") for j in self.jobs))  # QC comes with each fit

    def test_routes_read_what_they_should(self):
        auto = self.by_id["E4/fit/dispersed30/mask_auto/rep01"]
        ignore = self.by_id["E4/fit/dispersed30/mask_ignore/rep01"]
        nomask = self.by_id["E4/fit/dispersed30/nomask/rep01"]
        self.assertEqual(auto.inputs, ignore.inputs)  # the masked VCF and its mask
        self.assertEqual(auto.inputs[0], nomask.inputs[0])  # the same masked VCF
        self.assertEqual(flag(auto.command, "--vcf-mask"), flag(ignore.command, "--vcf-mask"))
        self.assertIsNotNone(flag(auto.command, "--vcf-mask"))
        self.assertNotIn("--vcf-mask", nomask.command)
        self.assertEqual(self.graph.deps["E4/fit/dispersed30/multihetsep/rep01"], ["E4/convert/dispersed30/rep01"])
        self.assertEqual(self.graph.deps["E4/fit/dispersed30/fq2psmcfa/rep01"], ["E4/consensus/dispersed30/rep01"])
        self.assertEqual(self.graph.deps["E4/mask/dispersed30/rep01"], ["E4/sim/rep01"])
        for j in (auto, ignore, nomask):
            self.assertEqual(j.command[3], str(config.SMOKE.e4.inference.n_iter))

    def test_every_fit_passes_its_bin_callable(self):
        want = {"mask_auto": "auto", "mask_ignore": "ignore", "fq2psmcfa": "auto", "multihetsep": "auto",
                "nomask": "auto"}
        self.assertEqual(config.E4_BIN_CALLABLE, want)
        for route, mode in want.items():
            job = self.by_id[f"E4/fit/clustered10/{route}/rep01"]
            self.assertEqual(flag(job.command, "--bin-callable"), mode, route)
            self.assertEqual(job.settings["bin_callable"], mode, route)
            self.assertEqual(job.command.count("--bin-callable"), 1, route)

    def test_mask_routes_fit_the_exact_bins(self):
        lay = e4.Layout(self.runs / "E4")
        for route in e4.MASK_ROUTES:
            self.assertEqual(lay.route_bins("dispersed30", route, 1), lay.exact_bins("dispersed30", 1))
        convert = self.by_id["E4/convert/dispersed30/rep01"]
        self.assertIn(lay.exact_bins("dispersed30", 1), convert.outputs)

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

    def test_report_waits_for_everything(self):
        report_deps = set(self.graph.deps["E4/report"])
        self.assertTrue({j.id for j in self.jobs if j.id.startswith(("E4/fit/", "E4/crosscheck/", "E4/bins/"))}
                        <= report_deps)

    def test_routes_must_include_both_mask_routes(self):
        for routes in (("mask_auto", "fq2psmcfa"), ("mask_ignore",), ("mask_auto", "mask_ignore", "exact")):
            cfg = replace(config.SMOKE, e4=replace(config.SMOKE.e4, routes=routes))
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                e4.jobs(Context(cfg, self.runs, _tools=dict(FAKE_TOOLS)))

    def test_production_plan(self):
        jobs = e4.jobs(Context(config.PRODUCTION, self.runs, _tools=dict(FAKE_TOOLS)))
        self.assertEqual(sum(1 for j in jobs if j.id.startswith("E4/fit/")), 5 * 5 * 5)
        self.assertEqual(sum(1 for j in jobs if j.id.startswith("E4/crosscheck/")), 5)
        plan = runner.plan(runner.Graph(jobs, self.runs), config.Budget())
        self.assertGreater(plan["E4"]["cpu_hours"], 1.0)
        self.assertLess(plan["E4"]["finish_hours"], 3.0)
        self.assertTrue(all(j.mem_gb <= 1.5 for j in jobs))


def fit_json(theta=0.03, lam=(1.0, 2.0), mode="auto", model="callable-length", counts="exact", mean_fraction=0.7,
             qc=None, bins=100):
    """A psmc-rs JSON as E4 reads it."""
    callable_ = None if counts is None else {"bin_size": 100, "called_bins": bins, "partial_bins": bins // 2,
                                             "mean_fraction": mean_fraction, "histogram": [0] * 10}
    bc = {"mode": mode, "model": model}
    if counts is not None:
        bc.update(counts=counts, callable=callable_)
    return {"theta": theta, "rho": theta / 5, "t_max": 15.0, "lam": list(lam), "loglike": [-3.0, -2.0],
            "final_loglike": -1.5, "n_steps": len(lam) - 1, "pattern": f"{len(lam)}*1", "n_iter": 2,
            "input": {"format": "vcf", "bins": bins, "het_bins": 5, "missing_bins": 0},
            "qc": qc or {"warnings": [], "notes": []}, "bin_callable": bc}


class Summary(unittest.TestCase):
    def test_means_and_qc_codes_over_replicates(self):
        cfg = replace(config.E4(), replicates=2, routes=("mask_auto",), masks=())
        rows = [
            {"mask": "nomissing", "route": "mask_auto", "bin_callable": "auto", "bin_model": "callable-length",
             "counts": "exact", "missing_fraction": 0.0, "mean_callable_fraction": 1.0,
             "partly_callable_fraction": 0.0, "differing_bin_fraction": 0.0, "missing_bin_fraction": 0.0,
             "theta_rel_complete": 0.0, "rmse_log10_ne_vs_truth": rmse, "rmse_log10_ne_vs_truth_fixed": rmse,
             "rmse_log10_ne_vs_complete": 0.0, "mean_log10_bias": -0.1, "identical_to_mask_auto": True,
             "identical_to_mask_ignore": True, "qc_warnings": warn, "qc_notes": "C"}
            for rmse, warn in ((0.1, "A;B"), (0.3, "B"))
        ]
        (row,) = e4.summary_rows(cfg, rows)
        self.assertAlmostEqual(row["rmse_truth_mean"], 0.2)
        self.assertEqual((row["qc_warnings_all"], row["qc_warnings_any"], row["qc_notes_any"]), ("B", "A;B", "C"))
        self.assertEqual((row["fits_identical_to_mask_ignore"], row["structure"]), (2, "none"))
        self.assertEqual((row["bin_model"], row["counts"], row["bin_callable"]), ("callable-length", "exact", "auto"))

    def test_same_fit_ignores_only_the_input_and_the_kernels(self):
        a = fit_json(mode="auto")
        psmcfa = fit_json(mode="auto", model="full-bins", counts=None, qc={"warnings": ["X"], "notes": []})
        psmcfa["input"]["format"] = "psmcfa"
        self.assertTrue(e4.same_fit(a, psmcfa))  # input, qc and bin_callable differ: the fit does not
        self.assertEqual(e4.fit_differences(a, fit_json(lam=(1.0, 2.0000001))), ["lam"])
        self.assertEqual(e4.fit_differences(a, {**a, "loglike": [-3.0, -2.5], "theta": 0.031}), ["loglike", "theta"])
        self.assertFalse(e4.same_fit(a, {**a, "t_max": 14.0}))
        self.assertTrue(e4.same_fit({**a, "simd": "avx2+fma"}, {**a, "simd": "portable"}))  # the same values

    def test_callable_bases_of_a_fit(self):
        self.assertAlmostEqual(e4.mean_callable_fraction(fit_json(mean_fraction=0.71)), 0.71)
        self.assertAlmostEqual(e4.partly_callable_fraction(fit_json(bins=100)), 0.5)
        self.assertIsNone(e4.mean_callable_fraction(fit_json(counts=None, model="full-bins")))
        self.assertIsNone(e4.partly_callable_fraction(fit_json(counts=None, model="full-bins")))

    def test_qc_codes_come_from_the_fit(self):
        params = fit_json(qc={"warnings": ["PARTLY_CALLABLE_BINS", "FEW_HETS"], "notes": ["HIGH_MISSING"]})
        self.assertEqual(e4.qc_findings(params), [
            {"code": "PARTLY_CALLABLE_BINS", "severity": "warning"}, {"code": "FEW_HETS", "severity": "warning"},
            {"code": "HIGH_MISSING", "severity": "note"}])
        self.assertEqual(e4.qc_findings({}), [])

    def test_fits_are_checked_for_their_bin_callable(self):
        e4.check_fit_mode(fit_json(mode="ignore"), "mask_ignore", "x")
        e4.check_fit_mode(fit_json(mode="auto"), "fq2psmcfa", "x")
        with self.assertRaises(ValueError):
            e4.check_fit_mode(fit_json(mode="auto"), "mask_ignore", "x")
        with self.assertRaises(ValueError):  # a psmc-rs without the callable-length model records none
            e4.check_fit_mode({k: v for k, v in fit_json().items() if k != "bin_callable"}, "mask_auto", "x")

    def _fits(self, mask, **params):
        from experiment.pipeline import results

        out = {}
        for route in e4.ROUTES:
            p = params.get(route) or fit_json()
            fit = results.fit_from_rs(p)
            out[route] = e4.RouteFit(mask, route, 1, p, fit, fit.curve(2.5e-8, 25.0, 100), {})
        return out

    def test_the_fits_that_must_agree_are_checked(self):
        e4.check_identities("dispersed30", 1, self._fits("dispersed30", mask_auto=fit_json(theta=0.04)))
        with self.assertRaisesRegex(ValueError, "fq2psmcfa fit differs from the mask_ignore fit in rho, theta"):
            e4.check_identities("dispersed30", 1, self._fits("dispersed30", fq2psmcfa=fit_json(theta=0.02)))
        # Every bin of the control is full: both models must fit identically there.
        e4.check_identities(config.E4_CONTROL_MASK, 1, self._fits(config.E4_CONTROL_MASK))
        with self.assertRaisesRegex(ValueError, "mask_auto fit differs from the mask_ignore fit"):
            e4.check_identities(config.E4_CONTROL_MASK, 1,
                                self._fits(config.E4_CONTROL_MASK, mask_auto=fit_json(theta=0.04)))

    def test_rescaling_multiplies_times_and_sizes_by_bin_size_over_callable_bases(self):
        from experiment.pipeline import results

        cfg = config.E4()
        fit = results.Fit(theta=0.03, rho=0.006, t_max=15.0, n=3, lambdas=(1.0, 2.0, 0.5, 1.0))
        sc = cfg.scaling
        rf = e4.RouteFit("dispersed30", "mask_ignore", 1, fit_json(), fit, fit.curve(sc.mu, sc.gen_years, 100), {})
        rescaled = e4.rescaled_curve(cfg, rf, 0.7)
        for a, b in zip(rescaled.years, rf.curve.years):
            self.assertAlmostEqual(a, b / 0.7, delta=1e-9 * max(b, 1.0))
        for a, b in zip(rescaled.ne, rf.curve.ne):
            self.assertAlmostEqual(a, b / 0.7, delta=1e-9 * b)

    @unittest.skipUnless(HAS_NUMPY, "needs numpy")
    def test_rescaling_row_sets_the_global_correction_beside_the_per_bin_model(self):
        import numpy as np

        from experiment.pipeline import demography, metrics

        cfg = config.E4()
        epochs = demography.epochs(config.MODELS[cfg.model])
        window = metrics.model_grid(epochs, cfg.grid, cfg.scaling.gen_years)
        fits = self._fits("dispersed30", mask_auto=fit_json(theta=0.03), mask_ignore=fit_json(theta=0.021,
                                                                                               mean_fraction=0.7))
        complete = self._fits(config.E4_CONTROL_MASK, mask_auto=fit_json(theta=0.03))["mask_auto"]

        def vs_truth(curve):
            return {"rmse_log10_ne_vs_truth": 0.0, "mean_log10_bias": 0.0}

        def vs_complete(curve, c):
            return {"rmse_log10_ne_vs_complete": metrics.log_time_diff(curve, c.curve, (window[0], window[-1])).rmse}

        row = e4.rescaling_row(cfg, "dispersed30", 1, 0.3, fits, complete, vs_truth, vs_complete, window)
        self.assertAlmostEqual(row["scale"], 1 / 0.7)
        self.assertAlmostEqual(row["theta_rel_complete_ignore"], -0.3)
        self.assertAlmostEqual(row["theta_rel_complete_rescaled"], 0.0)
        self.assertAlmostEqual(row["theta_rel_complete_auto"], 0.0)
        # theta / 0.7 and times and sizes x 1 / 0.7: the rescaled curve is the complete one here.
        self.assertLess(row["rmse_complete_rescaled"], 1e-9)
        self.assertLess(row["rmse_rescaled_vs_auto_per_interval"], 1e-9)
        self.assertLess(row["max_abs_rescaled_vs_auto_per_interval"], 1e-9)
        self.assertGreater(row["rmse_complete_ignore"], 0.1)
        self.assertTrue(np.isfinite(row["rmse_complete_auto"]))
        self.assertEqual(set(row), set(e4.FIELDS["rescaling"]))


def git_ignores(rel_path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel_path], cwd=ROOT).returncode == 0


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "not a git checkout")
class GitLayout(unittest.TestCase):
    """Records, fits, tables and figures are kept; genomes, masks and route inputs are not."""

    def test_small_results_are_tracked(self):
        for p in ("truth/rep01.json", "masks/dispersed30/rep01.json", "consensus/dispersed30/rep01.json",
                  "bins/dispersed30/rep01.json", "crosscheck/dispersed30/rep01.json",
                  "fits/dispersed30/multihetsep/rep01.json", "fits/dispersed30/mask_ignore/rep01.json",
                  "provenance/fit/dispersed30/mask_auto/rep01.json", "tables/E4_summary.csv",
                  "tables/E4_rescaling.csv", "figures/fig5_input_preparation.pdf"):
            self.assertFalse(git_ignores(f"experiment/runs/E4/{p}"), p)

    def test_regenerable_inputs_are_ignored(self):
        for p in ("sims/rep01/genome.vcf.gz", "sims/rep01/dispersed30/callable.bed.gz",
                  "sims/rep01/dispersed30/variants.vcf.gz", "sims/rep01/dispersed30/fq2psmcfa.psmcfa",
                  "sims/rep01/dispersed30/input.multihetsep", "sims/rep01/dispersed30/bins/exact.psmcfa",
                  "fits/dispersed30/mask_auto/rep01.html", "logs/attempts.jsonl",
                  "figures/fig5_input_preparation.pdf.partial"):
            self.assertTrue(git_ignores(f"experiment/runs/E4/{p}"), p)


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E4 in the tiny runs of every experiment (tiny_runs): one replicate of 2 x 300 kb, two masks plus the control."""

    def test_tiny_e4(self):
        import json

        lay = e4.Layout(tiny_runs.runs_dir() / "E4")
        summary = {(r["mask"], r["route"]): r for r in read_csv(lay.table("summary"))}
        masks = ("nomissing", "clustered30", "dispersed30")
        self.assertEqual(set(summary), {(m, r) for m in masks for r in e4.ROUTES})
        for mask in masks:
            # The report asserted it: fq2psmcfa fits as the VCF + mask with every bin full does.
            self.assertEqual(summary[(mask, "fq2psmcfa")]["fits_identical_to_mask_ignore"], "1")
            self.assertEqual(summary[(mask, "fq2psmcfa")]["bin_model"], "full-bins")
            self.assertEqual(summary[(mask, "mask_ignore")]["bin_model"], "full-bins")
            self.assertEqual(summary[(mask, "mask_auto")]["bin_model"], "callable-length")
            self.assertEqual(summary[(mask, "mask_auto")]["counts"], "exact")
            self.assertEqual(summary[(mask, "multihetsep")]["counts"], "approximate")
            self.assertIn("VARIANTS_ONLY_NO_MASK", summary[(mask, "nomask")]["qc_warnings_all"])
        # Complete data: every bin full, so both models, and the VCF without a mask, fit identically.
        for route in ("mask_ignore", "fq2psmcfa", "nomask"):
            self.assertEqual(summary[("nomissing", route)]["fits_identical_to_mask_auto"], "1", route)
            self.assertEqual(float(summary[("nomissing", route)]["rmse_complete_mean"]), 0.0, route)
        self.assertEqual(float(summary[("nomissing", "mask_auto")]["mean_callable_fraction_mean"]), 1.0)
        # The dispersed mask leaves bins called with fewer callable bases: the full-bins fit's theta is low
        # by about the missing share, and QC warns about it only where the fit ignores the counts.
        d_auto, d_ignore = summary[("dispersed30", "mask_auto")], summary[("dispersed30", "mask_ignore")]
        self.assertLess(float(d_auto["mean_callable_fraction_mean"]), 0.9)
        self.assertGreater(float(d_auto["partly_callable_fraction_mean"]), 0.5)
        self.assertEqual(d_auto["fits_identical_to_mask_ignore"], "0")
        self.assertLess(float(d_ignore["theta_rel_complete_mean"]), -0.1)
        self.assertGreater(float(d_auto["theta_rel_complete_mean"]), float(d_ignore["theta_rel_complete_mean"]) + 0.1)
        self.assertIn("PARTLY_CALLABLE_BINS", d_ignore["qc_warnings_all"])
        self.assertNotIn("PARTLY_CALLABLE_BINS", d_auto["qc_warnings_any"])
        self.assertIn("PARTLY_CALLABLE_BINS", d_auto["qc_notes_any"])
        fit = json.loads(lay.fit("dispersed30", "mask_auto", 1).read_text())
        self.assertEqual(fit["bin_callable"]["mode"], "auto")
        self.assertEqual(fit["bin_callable"]["model"], "callable-length")
        # The rescaling cross-check reads the callable bases of the full-bins fit.
        rescaling = {r["mask"]: r for r in read_csv(lay.table("rescaling"))}
        self.assertEqual(set(rescaling), set(masks))
        self.assertAlmostEqual(float(rescaling["nomissing"]["scale"]), 1.0)
        self.assertGreater(float(rescaling["dispersed30"]["scale"]), 1.1)
        self.assertAlmostEqual(float(rescaling["dispersed30"]["mean_callable_fraction"]),
                               float(d_ignore["mean_callable_fraction_mean"]))
        bins = {(r["mask"], r["route"]): r for r in read_csv(lay.table("bins"))}
        self.assertEqual(bins[("clustered30", "fq2psmcfa")]["bytes_identical"], "True")
        self.assertGreater(int(bins[("clustered30", "nomask")]["N_T"]), 0)
        self.assertTrue(all(r["identical"] == "True" for r in read_csv(lay.table("crosscheck"))))
        qc = read_csv(lay.table("qc"))
        self.assertEqual({r["severity"] for r in qc}, {"warning", "note"})


if __name__ == "__main__":
    unittest.main()
