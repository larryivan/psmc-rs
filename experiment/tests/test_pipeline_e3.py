"""Tests for experiment E3: its runs and estimates, its tables, and (with every tool) a tiny run end to end."""

from __future__ import annotations

import csv
import importlib.util
import json
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

from experiment.pipeline import config, provenance, results, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS01 = tools.Tool("psmc-rs-0.1", Path("/x/psmc-rs-0.1"), {"version": "psmc-rs 0.1.0", "commit": "e" * 40})
FAKE_RS02 = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})
HAS_TOOLCHAIN = all(importlib.util.find_spec(m) is not None for m in ("msprime", "psutil", "numpy"))


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fake_ctx(profile: config.Profile, runs: Path) -> Context:
    return Context(profile, runs, _tools={"c": FAKE_C, "rs01": FAKE_RS01, "rs02": FAKE_RS02})


def setup_of(cfg, tool, size, threads):
    return next(p for p in e3.plan(cfg) if p.setup == e3.Setup(tool, size, threads))


class Plan(unittest.TestCase):
    def test_production_setups(self):
        cfg = config.PRODUCTION.e3
        keys = [u.key() for u in e3.setups(cfg)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(sum(k.startswith("c/") for k in keys), 4)
        self.assertEqual(sum(k.startswith("rs01/") for k in keys), 8)
        # rs02: 4 sizes x (1, 8) threads, plus 3000 Mb on 2 and 4 threads
        self.assertEqual(sorted(u.threads for u in e3.setups(cfg) if u.tool == "rs02" and u.size_mb == 3000), [1, 2, 4, 8])
        self.assertEqual(sum(k.startswith("rs02/") for k in keys), 10)
        self.assertEqual(e3.sizes(cfg), [100, 300, 1000, 3000])

    def test_memory_estimates_follow_each_tool(self):
        cfg = config.PRODUCTION.e3
        c = [setup_of(cfg, "c", s, 1).est_gb for s in cfg.genome_sizes_mb]
        self.assertEqual(len(set(c)), 1)  # the longest sequence, 100 Mb, at every size
        self.assertAlmostEqual(c[0], (6.0 + 1005.0) / 1024, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 1000, 1).est_gb * 1024, 5.0 + 9.5 + 8 * 248 + 496, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 3000, 8).est_gb * 1024, 5.0 + 28.5 + 8 * 496, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 100, 8).est_gb, setup_of(cfg, "rs01", 100, 1).est_gb)
        self.assertLess(setup_of(cfg, "rs02", 3000, 8).est_gb, 0.1)
        self.assertTrue(all(p.skip is None for p in e3.plan(cfg)))

    def test_runs_over_the_memory_limit_are_skipped_and_say_why(self):
        cfg = replace(config.PRODUCTION.e3, genome_sizes_mb=(1500, 3000), chromosome_length=1_500_000_000,
                      thread_scaling_size_mb=3000)
        plan = {p.setup: p for p in e3.plan(cfg)}
        c = plan[e3.Setup("c", 1500, 1)]
        self.assertIsNotNone(c.skip)
        self.assertIn("14.7 GB exceeds the 12 GB limit", c.skip)
        self.assertIn("longest sequence (1500 Mb)", c.skip)
        self.assertIsNone(plan[e3.Setup("rs01", 1500, 1)].skip)  # nothing cached, one 7.3 GB buffer
        self.assertIsNotNone(plan[e3.Setup("rs01", 3000, 8)].skip)  # two buffers at once
        self.assertIsNone(plan[e3.Setup("rs02", 3000, 8)].skip)
        self.assertFalse(any(r.setup.tool == "c" for r in e3.runs(cfg)))

    def test_time_estimates(self):
        cfg = config.PRODUCTION.e3
        c = setup_of(cfg, "c", 3000, 1).est_seconds
        self.assertGreater(c, 3000)
        one, eight = setup_of(cfg, "rs02", 3000, 1).est_seconds, setup_of(cfg, "rs02", 3000, 8).est_seconds
        self.assertGreater(one / eight, 4)  # 30 sequences in parallel; the M-step takes ms
        # psmc-rs 0.1's M-step, 1000 forward-mode iterations, costs seconds per EM iteration.
        self.assertGreater(setup_of(cfg, "rs01", 100, 1).est_seconds,
                           setup_of(cfg, "rs02", 100, 1).est_seconds + 100)
        capped = replace(cfg, default_mstep_iters=5)
        self.assertLessEqual(setup_of(capped, "rs02-default", 3000, 8).est_seconds, eight)
        # one 100 Mb chromosome: no parallel speed-up
        self.assertAlmostEqual(setup_of(cfg, "rs02", 100, 1).est_seconds, setup_of(cfg, "rs02", 100, 8).est_seconds)


class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)

    def jobs(self, profile):
        return {j.id: j for j in e3.jobs(fake_ctx(profile, self.runs))}

    def test_production_job_graph(self):
        jobs = self.jobs(config.PRODUCTION)
        cfg = config.PRODUCTION.e3
        runs = [j for j in jobs.values() if j.name.startswith("run/")]
        # No run with capped M-steps: psmc-rs 0.2's default M-step converges.
        self.assertEqual(len(runs), 22 * cfg.repeats)
        self.assertFalse(any("rs02-default" in j.id for j in runs))
        self.assertEqual(len(jobs), len(runs) + 4 + 1)
        self.assertTrue(all(j.exclusive for j in runs))
        self.assertFalse(any(j.exclusive for j in jobs.values() if not j.name.startswith("run/")))
        g = runner.Graph(list(jobs.values()), self.runs)
        self.assertEqual(g.deps["E3/run/rs02/3000Mb/t4/rep2"], ["E3/sim/3000Mb"])
        self.assertEqual(set(g.deps["E3/report"]), {j.id for j in jobs.values() if j.id != "E3/report"})
        plan = runner.plan(g, config.Budget())
        self.assertGreater(plan["E3"]["wall_hours_alone"], 5)

    def test_commands_follow_the_recipe(self):
        jobs = self.jobs(config.PRODUCTION)
        c = jobs["E3/run/c/300Mb/t1/rep1"].command
        tool = c[c.index("--") + 1 :]
        self.assertEqual(tool[:6], ["/x/psmc", "-N25", "-t15", "-r5", "-p", "4+25*2+4+6"])
        self.assertIn("experiment.pipeline.experiments.e3_run", c)
        rs01 = jobs["E3/run/rs01/1000Mb/t8/rep3"]
        cmd = rs01.command[rs01.command.index("--") + 1 :]
        self.assertEqual(cmd[3], "25")
        self.assertNotIn("--tr-ratio", cmd)
        for opt, val in (("--pattern", "4+25*2+4+6"), ("--n-steps", "63"), ("--t-max", "15"),
                         ("--smooth-lambda", "0"), ("--mstep-iters", "1000"), ("--threads", "8")):
            self.assertEqual(cmd[cmd.index(opt) + 1], val)
        self.assertEqual(rs01.env, {"RAYON_NUM_THREADS": "8"})
        self.assertEqual(rs01.cores, config.Budget().cores)
        rs02 = jobs["E3/run/rs02/3000Mb/t2/rep1"]
        cmd = rs02.command[rs02.command.index("--") + 1 :]
        self.assertEqual(cmd[cmd.index("--tr-ratio") + 1], "5")
        self.assertEqual(cmd[cmd.index("--threads") + 1], "2")
        self.assertEqual(rs02.settings["repeat"], 1)
        self.assertEqual(set(rs02.tools), {"psmc-rs-0.2", "timing"})
        self.assertEqual(cmd[cmd.index("--mstep-iters") + 1], "1000")
        capped = replace(config.PRODUCTION, e3=replace(config.PRODUCTION.e3, default_mstep_iters=5))
        default = self.jobs(capped)["E3/run/rs02-default/3000Mb/t8/rep1"]
        cmd = default.command[default.command.index("--") + 1 :]
        self.assertEqual(cmd[cmd.index("--mstep-iters") + 1], "5")
        self.assertEqual(default.settings["inference"]["mstep_iters"], 5)

    def test_simulation_jobs(self):
        jobs = self.jobs(config.PRODUCTION)
        sim = jobs["E3/sim/300Mb"]
        self.assertEqual(sim.settings["genome"]["n_chromosomes"], 3)
        self.assertEqual(sim.seeds["chr3"]["mutations"], config.seed("E3", "constant", 1, "mutations/chr3"))
        self.assertEqual(sim.seeds["chr1"], jobs["E3/sim/3000Mb"].seeds["chr1"])  # one set of chromosomes
        self.assertIn("--size-mb", sim.command)

    def test_missing_rs01_build_does_not_stop_planning(self):
        ctx = Context(config.SMOKE, self.runs, _tools={"c": FAKE_C, "rs02": FAKE_RS02})
        tool = e3.rs01_tool(ctx)
        self.assertEqual(tool.name, "psmc-rs-0.1")


class Report(unittest.TestCase):
    """The tables from stand-in runs: C psmc's reference output, and psmc-rs JSON made from it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        cfg = replace(config.TINY.e3, inference=replace(config.TINY.e3.inference, n_iter=5), repeats=2)
        self.profile = replace(config.TINY, e3=cfg)
        self.cfg = cfg
        lay = self.lay = e3.Layout(self.runs / "E3")
        ref = results.parse_psmc(C_REF)
        last = ref.last()
        truth = {"chromosomes": [
            {"name": "chr1", "bins": 100000, "het_bins": 5000, "het_sites": 5100, "seeds": {"ancestry": 1, "mutations": 2}},
            {"name": "chr2", "bins": 50947, "het_bins": 2373, "het_sites": 2400, "seeds": {"ancestry": 3, "mutations": 4}},
        ]}
        lay.truth(1).parent.mkdir(parents=True)
        lay.truth(1).write_text(json.dumps(truth))
        walls = {"c": 10.0, "rs01": 4.0, "rs02": 1.0, "rs02-default": 0.5}
        for run in e3.runs(cfg):
            u = run.setup
            fit = lay.fit(run)
            fit.parent.mkdir(parents=True, exist_ok=True)
            if u.tool == "c":
                shutil.copy(C_REF, fit)
            else:
                # psmc-rs 0.1 on 2 threads would differ; here rs02 on 2 threads repeat 2 does
                scale = 1.001 if (u.tool == "rs02" and u.threads == 2 and run.repeat == 2) else 1.0
                params = {"theta": last.pa["theta"], "rho": last.pa["rho"], "t_max": last.pa["t_max"], "n_steps": 63,
                          "pattern": "4+25*2+4+6", "lam": [x * scale for x in last.pa["lambdas"]]}
                if u.tool in ("rs02", "rs02-default"):
                    params.update(n_iter=5, final_loglike=-25000.0, input={"bins": 150947, "het_bins": 7373})
                fit.write_text(json.dumps(params))
            wall = walls[u.tool] / (1.6 if u.threads == 2 else 1.0) + 0.1 * run.repeat
            timing = {
                "wall_seconds": wall, "user_seconds": wall * u.threads * 0.9, "system_seconds": 0.1,
                "cpu_seconds": wall * u.threads * 0.9 + 0.1, "peak_rss_mb": 100.0 if u.tool == "c" else 10.0,
                "maxrss_mb": 100.0, "peak_rss_tree_sampled_mb": 99.0, "loadavg_before": [0.5, 0.5, 0.5],
                "loadavg_after": [1.0, 0.5, 0.5], "start": "2026-10-05T00:00:00", "end": "2026-10-05T00:00:10",
                "machine_before": {"quiet": True, "busy_percent": 3.0, "waited_seconds": 0.0},
                "during": {"other_cpu_share": 0.01}, "machine": {"cpu": "Test CPU", "cpu_count": 8},
            }
            lay.timing(run).parent.mkdir(parents=True, exist_ok=True)
            lay.timing(run).write_text(json.dumps(timing))
            lay.record(run).parent.mkdir(parents=True, exist_ok=True)
            tool = {"c": FAKE_C, "rs01": FAKE_RS01, "rs02": FAKE_RS02, "rs02-default": FAKE_RS02}[u.tool]
            lay.record(run).write_text(json.dumps({"key": {"tools": {tool.name: tool.identity}}}))
        self.written = e3.report(Context(self.profile, self.runs))

    def test_every_table_is_written(self):
        self.assertEqual(set(self.written), set(e3.TABLES))
        self.assertEqual(len(read_csv(self.lay.table("runs"))), len(e3.runs(self.cfg)))
        seeds = read_csv(self.lay.table("seeds"))
        self.assertEqual([r["chromosome"] for r in seeds], ["chr1", "chr2"])

    def test_table2_medians_and_speedups(self):
        rows = {(r["tool"], r["threads"]): r for r in read_csv(self.lay.table("table2"))}
        c = rows[("c", "1")]
        self.assertEqual(c["status"], "measured")
        self.assertAlmostEqual(float(c["wall_median_s"]), 10.15)
        self.assertAlmostEqual(float(c["wall_min_s"]), 10.1)
        self.assertAlmostEqual(float(c["seconds_per_iteration_median"]), 10.15 / 5)
        rs02 = rows[("rs02", "1")]
        self.assertAlmostEqual(float(rs02["speedup_vs_c"]), 10.15 / 1.15)
        self.assertAlmostEqual(float(rs02["speedup_vs_rs01_same_threads"]), 4.15 / 1.15)
        self.assertAlmostEqual(float(rs02["c_rss_over_rss"]), 10.0)
        self.assertEqual(rs02["all_quiet"], "True")
        self.assertEqual(rs02["outputs_identical"], "True")
        self.assertEqual(rows[("rs02", "2")]["outputs_identical"], "False")
        self.assertEqual(c["speedup_vs_c"], "")

    def test_thread_scaling(self):
        rows = {int(r["threads"]): r for r in read_csv(self.lay.table("scaling"))}
        self.assertEqual(sorted(rows), [1, 2])
        one = (1.0 + 0.1 + 1.0 + 0.2) / 2
        two = (1.0 / 1.6 + 0.1 + 1.0 / 1.6 + 0.2) / 2
        self.assertAlmostEqual(float(rows[2]["speedup"]), one / two)
        self.assertAlmostEqual(float(rows[2]["efficiency"]), one / two / 2)
        self.assertEqual(rows[1]["json_identical_to_reference"], "True")
        self.assertEqual(rows[2]["json_identical_to_reference"], "False")

    def test_determinism_against_the_first_run(self):
        rows = read_csv(self.lay.table("determinism"))
        rs02 = {(r["threads"], r["repeat"]): r for r in rows if r["tool"] == "rs02"}
        self.assertEqual(rs02[("1", "1")]["byte_identical"], "True")
        self.assertEqual(rs02[("2", "1")]["byte_identical"], "True")
        self.assertEqual(rs02[("2", "2")]["byte_identical"], "False")
        self.assertAlmostEqual(float(rs02[("2", "2")]["max_rel_diff_params"]), 0.001, places=9)
        self.assertEqual(float(rs02[("2", "2")]["loglike_rel_diff"]), 0.0)
        c = [r for r in rows if r["tool"] == "c"]
        self.assertTrue(all(r["byte_identical"] == "True" for r in c))

    def test_environment(self):
        env = {r["key"]: r["value"] for r in read_csv(self.lay.table("environment"))}
        self.assertEqual(env["machine.cpu"], "Test CPU")
        self.assertEqual(env["tool.rs01.commit"], "e" * 40)
        self.assertEqual((env["runs_not_quiet"], env["runs_not_clean"]), ("0", "0"))

    def test_a_fit_of_another_genome_is_refused(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs02")
        params = json.loads(self.lay.fit(run).read_text())
        params["input"]["bins"] += 1
        self.lay.fit(run).write_text(json.dumps(params))
        with self.assertRaises(ValueError):
            e3.report(Context(self.profile, self.runs))

    def test_a_run_on_a_busy_machine_stays_out_of_the_medians(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "c" and r.repeat == 2)
        timing = json.loads(self.lay.timing(run).read_text())
        timing["machine_before"]["quiet"] = False
        self.lay.timing(run).write_text(json.dumps(timing))
        e3.report(Context(self.profile, self.runs))
        c = next(r for r in read_csv(self.lay.table("table2")) if r["tool"] == "c")
        self.assertEqual((c["status"], c["repeats"], c["repeats_measured"]), ("incomplete", "1", "2"))
        self.assertAlmostEqual(float(c["wall_median_s"]), 10.1)  # repeat 1 only
        runs = [r for r in read_csv(self.lay.table("runs")) if r["tool"] == "c"]
        self.assertEqual(sorted(r["clean"] for r in runs), ["False", "True"])

    def test_timing_runs_are_ranked_by_repeat(self):
        jobs = e3.jobs(Context(self.profile, self.runs, _tools={"c": FAKE_C, "rs01": FAKE_RS01, "rs02": FAKE_RS02}))
        runs = [j for j in jobs if j.exclusive]
        self.assertTrue(runs)
        self.assertTrue(all(j.rank == int(j.id.rsplit("rep", 1)[1]) for j in runs))

    def test_partial_results_are_reported_as_incomplete(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs01")
        self.lay.timing(run).unlink()
        e3.report(Context(self.profile, self.runs))
        row = next(r for r in read_csv(self.lay.table("table2")) if r["tool"] == "rs01")
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["repeats"], "1")


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def ignored(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=ROOT).returncode == 0


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    """Git keeps E3's fits, timings, truth, provenance and tables, not its genomes or HTML reports."""

    def test_what_git_keeps(self):
        lay = e3.Layout(ROOT / "experiment" / "runs" / "E3")
        c = e3.Run(e3.Setup("c", 3000, 1), 3)
        rs = e3.Run(e3.Setup("rs02", 3000, 8), 1)
        for p in (lay.fit(c), lay.fit(rs), lay.timing(rs), lay.record(rs), lay.truth(3000), lay.table("table2"),
                  lay.root / "provenance" / "sim" / "3000Mb.json"):
            self.assertFalse(ignored(p), p)
        for p in (lay.psmcfa(3000), lay.fit(rs).with_suffix(".html"), lay.root / "logs" / "attempts.jsonl"):
            self.assertTrue(ignored(p), p)


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E3 in the tiny runs of every experiment (tiny_runs): simulate, time every tool twice, write the tables."""

    def test_tiny_profile(self):
        lay = e3.Layout(tiny_runs.runs_dir() / "E3")
        table2 = read_csv(lay.table("table2"))
        self.assertEqual(len(table2), len(e3.setups(config.TINY.e3)))
        self.assertTrue(all(r["status"] == "measured" for r in table2))
        self.assertTrue(all(float(r["wall_median_s"]) > 0 and float(r["peak_rss_median_mb"]) > 0 for r in table2))
        det = [r for r in read_csv(lay.table("determinism")) if r["tool"] in ("rs02", "rs02-default", "c")]
        self.assertTrue(det and all(r["byte_identical"] == "True" for r in det))
        run = e3.Run(e3.Setup("rs02", 1, 2), 2)
        self.assertEqual(json.loads(lay.timing(run).read_text())["exit_status"], 0)
        rec = json.loads(lay.record(run).read_text())
        self.assertTrue(rec["run"]["exclusive"])
        self.assertEqual(provenance.sha256_file(lay.fit(run)), provenance.sha256_file(lay.fit(e3.Run(e3.Setup("rs02", 1, 1), 1))))


if __name__ == "__main__":
    unittest.main()
