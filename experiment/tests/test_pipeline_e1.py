"""Tests for experiment E1: its jobs, its tables, and (with the tools present) a tiny run end to end."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, demography, results, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e1  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_MSPRIME = importlib.util.find_spec("msprime") is not None and importlib.util.find_spec("psutil") is not None


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fake_ctx(profile: config.Profile, runs: Path) -> Context:
    return Context(profile, runs, _tools={"c": FAKE_C, "rs02": FAKE_RS})


class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        self.jobs = e1.jobs(fake_ctx(config.SMOKE, self.runs))
        self.by_id = {j.id: j for j in self.jobs}

    def test_one_simulation_and_three_fits_per_replicate(self):
        cfg = config.SMOKE.e1
        self.assertEqual(len(self.jobs), len(cfg.models) * cfg.replicates * 4 + 1)
        default = self.by_id["E1/fit-rs-default/bottleneck/rep02"]
        self.assertEqual(default.command[default.command.index("--mstep-iters") + 1], "100")
        main = self.by_id["E1/fit-rs/bottleneck/rep02"]
        self.assertEqual(main.command[main.command.index("--mstep-iters") + 1], "1000")
        g = runner.Graph(self.jobs, self.runs)
        self.assertEqual(g.deps["E1/fit-c/bottleneck/rep02"], ["E1/sim/bottleneck/rep02"])
        self.assertEqual(g.deps["E1/fit-rs/bottleneck/rep02"], ["E1/sim/bottleneck/rep02"])
        report = set(g.deps["E1/report"])
        self.assertEqual(report, {j.id for j in self.jobs if j.id.startswith(("E1/fit-", "E1/sim"))})

    def test_fits_read_the_identical_psmcfa_with_identical_settings(self):
        c = self.by_id["E1/fit-c/constant/rep01"]
        rs = self.by_id["E1/fit-rs/constant/rep01"]
        self.assertEqual(c.inputs, rs.inputs)
        self.assertEqual(c.command[1:6], ["-N3", "-t15", "-r5", "-p", "4+25*2+4+6"])
        self.assertEqual(rs.command[3], "3")
        self.assertIn("--psmc-out", rs.command)
        self.assertEqual(c.settings["inference"], rs.settings["inference"])
        self.assertEqual(c.tools, {"c-psmc": FAKE_C.identity})

    def test_simulation_seeds_are_recorded(self):
        sim = self.by_id["E1/sim/zigzag/rep01"] if "E1/sim/zigzag/rep01" in self.by_id else self.by_id["E1/sim/bottleneck/rep01"]
        model = sim.id.split("/")[2]
        self.assertEqual(sim.seeds["chr2"]["ancestry"], config.seed("E1", model, 1, "ancestry/chr2"))
        self.assertEqual(sim.settings["genome"]["chromosome_length"], config.SMOKE.e1.genome.chromosome_length)
        self.assertIn("--profile", sim.command)
        self.assertEqual(sim.outputs[-1], self.runs / "E1" / "truth" / model / "rep01.json")

    def test_production_estimates(self):
        jobs = e1.jobs(fake_ctx(config.PRODUCTION, self.runs))
        self.assertEqual(len(jobs), 4 * 10 * 4 + 1)
        c = next(j for j in jobs if j.id == "E1/fit-c/zigzag/rep10")
        self.assertLess(c.mem_gb, 1.0)  # one 50 Mb chromosome at a time
        plan = runner.plan(runner.Graph(jobs, self.runs), config.Budget())
        self.assertGreater(plan["E1"]["cpu_hours"], 1.0)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class Tables(unittest.TestCase):
    """The report on stand-in fits: C psmc's reference output for both tools."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        inference = replace(config.INFERENCE, n_iter=5, tr_ratio=4.0)
        self.profile = replace(
            config.TINY, e1=replace(config.TINY.e1, models=("bottleneck",), replicates=2, inference=inference)
        )
        lay = e1.Layout(self.runs / "E1")
        ref = results.parse_psmc(C_REF)
        last = ref.last()
        for r in (1, 2):
            truth = {
                "chromosomes": [{"name": "chr1", "bins": 150947, "het_bins": 7373, "het_sites": 7400,
                                 "seeds": {"ancestry": 1, "mutations": 2}}],
                "truth": demography.truth_record(config.BOTTLENECK),
            }
            lay.truth("bottleneck", r).parent.mkdir(parents=True, exist_ok=True)
            lay.truth("bottleneck", r).write_text(json.dumps(truth))
            lay.c_psmc("bottleneck", r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(C_REF, lay.c_psmc("bottleneck", r))
            shutil.copy(C_REF, lay.rs_psmc("bottleneck", r))
            scale = 1.0 if r == 1 else 2.0
            params = {
                "theta": last.pa["theta"], "rho": last.pa["rho"], "t_max": last.pa["t_max"], "n_steps": 63,
                "pattern": "4+25*2+4+6", "lam": [x * scale for x in last.pa["lambdas"]], "n_iter": 5,
                "loglike": [ref.loglikes()[i] for i in range(5)] + [-25000.0],
                "input": {"bins": 150947, "het_bins": 7373},
            }
            lay.rs_json("bottleneck", r).write_text(json.dumps(params))
            # psmc-rs's default M-step budget: 1.5 times the sizes of psmc-rs's fit
            default = dict(params, lam=[x * 1.5 for x in params["lam"]])
            lay.rs_default_json("bottleneck", r).write_text(json.dumps(default))
        self.written = e1.report(Context(self.profile, self.runs))
        self.lay = lay

    def test_identical_fits_agree_and_scaled_ones_differ_by_the_scale(self):
        pairs = read_csv(self.lay.table("pairs"))
        self.assertEqual(float(pairs[0]["rmse_log10_ne_rs_vs_c"]), 0.0)
        self.assertAlmostEqual(float(pairs[1]["rmse_log10_ne_rs_vs_c"]), math.log10(2), places=12)
        self.assertEqual(float(pairs[0]["loglike_max_rel_diff"]), 0.0)
        t1 = read_csv(self.lay.table("table1"))
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1[0]["replicates"], "2")
        self.assertAlmostEqual(float(t1[0]["rmse_rs_vs_c_max"]), math.log10(2), places=12)

    def test_default_mstep_budget_against_c_psmc(self):
        rows = read_csv(self.lay.table("mstep"))
        self.assertEqual([r["replicate"] for r in rows], ["1", "2"])
        self.assertAlmostEqual(float(rows[0]["rmse_log10_ne_default_vs_c"]), math.log10(1.5), places=9)
        self.assertAlmostEqual(float(rows[0]["rmse_log10_ne_default_vs_rs"]), math.log10(1.5), places=9)
        self.assertEqual(float(rows[0]["theta_default_rel_c"]), 0.0)
        self.assertEqual(rows[0]["mstep_iters_default"], "100")
        t1 = read_csv(self.lay.table("table1"))[0]
        self.assertGreater(float(t1["rmse_rs_default_vs_c_max"]), 0)

    def test_traces_count_iterations(self):
        rows = read_csv(self.lay.table("traces"))
        c = [r for r in rows if r["tool"] == "c" and r["replicate"] == "1"]
        rs = [r for r in rows if r["tool"] == "rs" and r["replicate"] == "1"]
        self.assertEqual([int(r["iteration"]) for r in c], list(range(6)))
        self.assertEqual(c[-1]["loglike"], "")  # C psmc never evaluates the final model
        self.assertEqual(float(rs[-1]["loglike"]), -25000.0)
        self.assertEqual(float(c[0]["t_max"]), 15.0)

    def test_every_table_is_written(self):
        self.assertEqual(set(self.written), set(e1.TABLES))
        curves = read_csv(self.lay.table("curves"))
        self.assertEqual(len(curves), 2 * 2 * 64)
        reps = read_csv(self.lay.table("replicates"))
        self.assertEqual({r["tool"] for r in reps}, {"c", "rs"})
        self.assertTrue(all(float(r["rmse_log10_ne_vs_truth"]) > 0 for r in reps))
        self.assertEqual(read_csv(self.lay.table("seeds"))[0]["ancestry_seed"], "1")


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E1 in the tiny runs of every experiment (tiny_runs): simulate, fit with both tools, write the tables."""

    def test_tiny_profile(self):
        lay = e1.Layout(tiny_runs.runs_dir() / "E1")
        pairs = read_csv(lay.table("pairs"))
        self.assertEqual(len(pairs), 1)
        # The first iteration's log-likelihood equals C psmc's to its printed precision.
        self.assertLess(float(pairs[0]["loglike_iteration0_rel_diff"]), 1e-8)
        self.assertLess(float(pairs[0]["rmse_log10_ne_rs_vs_c"]), 0.5)
        self.assertEqual(len(read_csv(lay.table("mstep"))), 1)
        truth = json.loads(lay.truth("bottleneck", 1).read_text())
        self.assertEqual(truth["chromosomes"][0]["seeds"]["ancestry"], config.seed("E1", "bottleneck", 1, "ancestry/chr1"))


if __name__ == "__main__":
    unittest.main()
