"""Tests for experiment E3: its runs and estimates, its tables, and (with every tool) a tiny run end to end."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import (C_REF, ROOT, assert_ignore_rules, c_ref_json, fake_ctx, fake_tools, in_git_work_tree,  # noqa: E402
                      option, read_csv)

from experiment.pipeline import config, provenance, runner  # noqa: E402
from experiment.pipeline.experiments import Context, e3  # noqa: E402


def tool_command(job) -> list:
    """The timed tool's own command: what follows ``--`` in the timing wrapper's."""
    return job.command[job.command.index("--") + 1:]


def setup_of(cfg, tool, size, threads):
    return next(p for p in e3.plan(cfg) if p.setup == e3.Setup(tool, size, threads))


class Plan(unittest.TestCase):
    def test_production_setups(self):
        cfg = config.PRODUCTION.e3
        keys = [u.key() for u in e3.setups(cfg)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(sum(k.startswith("c/") for k in keys), 4)
        self.assertEqual(sum(k.startswith("rs01/") for k in keys), 12)
        # rs02: 4 sizes x (1, 8, 30) threads, plus 3000 Mb on 2, 4 and 16 threads
        self.assertEqual(sorted(u.threads for u in e3.setups(cfg) if u.tool == "rs02" and u.size_mb == 3000),
                         [1, 2, 4, 8, 16, 30])
        self.assertEqual(sum(k.startswith("rs02/") for k in keys), 15)
        self.assertEqual(e3.sizes(cfg), [100, 300, 1000, 3000])

    def test_memory_estimates_follow_each_tool(self):
        cfg = config.PRODUCTION.e3
        c = [setup_of(cfg, "c", s, 1).est_gb for s in cfg.genome_sizes_mb]
        self.assertEqual(len(set(c)), 1)  # the longest sequence, 100 Mb, at every size
        self.assertAlmostEqual(c[0], (6.0 + 1005.0) / 1024, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 1000, 1).est_gb * 1024, 5.0 + 9.5 + 8 * 248 + 496, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 3000, 8).est_gb * 1024, 5.0 + 28.5 + 8 * 496, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 3000, 30).est_gb * 1024, 5.0 + 28.5 + 30 * 496, places=6)
        self.assertAlmostEqual(setup_of(cfg, "rs01", 100, 8).est_gb, setup_of(cfg, "rs01", 100, 1).est_gb)
        self.assertLess(setup_of(cfg, "rs02", 3000, 8).est_gb, 0.1)
        self.assertLess(setup_of(cfg, "rs02", 3000, 30).est_gb, 0.2)
        self.assertTrue(all(p.skip is None for p in e3.plan(cfg)))  # the largest, rs01 on 30 threads, 14.6 GB

    def test_runs_over_the_memory_limit_are_skipped_and_say_why(self):
        cfg = replace(config.PRODUCTION.e3, genome_sizes_mb=(1500, 3000), chromosome_length=1_500_000_000,
                      thread_scaling_size_mb=3000, memory_limit_gb=12.0)
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
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def jobs(self, profile):
        return {j.id: j for j in e3.jobs(fake_ctx(profile, self.runs))}

    def test_production_job_graph(self):
        jobs = self.jobs(config.PRODUCTION)
        cfg = config.PRODUCTION.e3
        runs = [j for j in jobs.values() if j.name.startswith("run/")]
        # No run with capped M-steps: psmc-rs 0.2's default M-step converges.
        self.assertEqual(len(runs), 31 * cfg.repeats)
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
        c = jobs["E3/run/c/300Mb/t1/rep1"]
        self.assertEqual(tool_command(c)[:6], ["/x/psmc", "-N25", "-t15", "-r5", "-p", "4+25*2+4+6"])
        self.assertIn("experiment.pipeline.experiments.e3_run", c.command)
        rs01 = jobs["E3/run/rs01/1000Mb/t8/rep3"]
        cmd = tool_command(rs01)
        self.assertEqual(cmd[3], "25")
        self.assertNotIn("--tr-ratio", cmd)
        for opt, val in (("--pattern", "4+25*2+4+6"), ("--n-steps", "63"), ("--t-max", "15"),
                         ("--smooth-lambda", "0"), ("--mstep-iters", "1000"), ("--threads", "8")):
            self.assertEqual(option(cmd, opt), val)
        self.assertEqual(rs01.env, {"RAYON_NUM_THREADS": "8"})
        self.assertEqual(rs01.cores, min(8, e3.machine_cores()))
        rs02 = jobs["E3/run/rs02/3000Mb/t2/rep1"]
        cmd = tool_command(rs02)
        self.assertEqual([option(cmd, o) for o in ("--tr-ratio", "--threads", "--mstep-iters")], ["5", "2", "1000"])
        self.assertEqual(rs02.settings["repeat"], 1)
        self.assertEqual(set(rs02.tools), {"psmc-rs-0.2", "timing"})
        capped = replace(config.PRODUCTION, e3=replace(config.PRODUCTION.e3, default_mstep_iters=5))
        default = self.jobs(capped)["E3/run/rs02-default/3000Mb/t8/rep1"]
        self.assertEqual(option(tool_command(default), "--mstep-iters"), "5")
        self.assertEqual(default.settings["inference"]["mstep_iters"], 5)

    def test_timing_jobs_get_their_threads_as_cores_up_to_the_machine(self):
        self.assertEqual([e3.run_cores(t, 128) for t in (1, 8, 30)], [1, 8, 30])
        self.assertEqual([e3.run_cores(t, 7) for t in (1, 8, 30)], [1, 7, 7])
        self.assertGreaterEqual(e3.machine_cores(), 1)
        for machine in (192, 128, 40, 8):
            with self.subTest(machine=machine), mock.patch.object(e3, "machine_cores", return_value=machine):
                jobs = self.jobs(config.PRODUCTION)
                big = jobs["E3/run/rs02/3000Mb/t30/rep5"]
                self.assertEqual(big.cores, min(30, machine))  # not capped at the laptop's budget of 7
                self.assertEqual((option(tool_command(big), "--threads"), big.env), ("30", {"RAYON_NUM_THREADS": "30"}))
                # measure is told the threads, and refuses to time 30 threads on fewer cores
                self.assertEqual(option(big.command[: big.command.index("--")], "--threads"), "30")
                # The runs are exclusive: the default budget's 7 cores start them all. Its 12 GB do not
                # start psmc-rs 0.1 on 30 threads (14.6 GB); E3 is started with --mem-gb 15.
                runs = [j for j in jobs.values() if j.exclusive]
                blocked = {j.id: runner.fits_budget(j, config.Budget()) for j in runs}
                self.assertEqual({k for k, why in blocked.items() if why},
                                 {f"E3/run/rs01/3000Mb/t30/rep{r}" for r in range(1, 6)})
                self.assertTrue(all("GB" in why for why in blocked.values() if why))
                self.assertEqual([j.id for j in runs if runner.fits_budget(j, config.Budget(mem_gb=15.0))], [])

    def test_the_report_reads_c_psmcs_build_record_when_there_is_one(self):
        c_dir = self.runs / "tools" / "psmc_c"
        c_dir.mkdir(parents=True)
        with mock.patch.object(e3.tools, "c_psmc_dir", return_value=c_dir):
            self.assertNotIn(c_dir / "BUILD.json", self.jobs(config.TINY)["E3/report"].inputs)
            (c_dir / "BUILD.json").write_text("{}")
            self.assertIn(c_dir / "BUILD.json", self.jobs(config.TINY)["E3/report"].inputs)

    def test_simulation_jobs(self):
        jobs = self.jobs(config.PRODUCTION)
        sim = jobs["E3/sim/300Mb"]
        self.assertEqual(sim.settings["genome"]["n_chromosomes"], 3)
        self.assertEqual(sim.seeds["chr3"]["mutations"], config.seed("E3", "constant", 1, "mutations/chr3"))
        self.assertEqual(sim.seeds["chr1"], jobs["E3/sim/3000Mb"].seeds["chr1"])  # one set of chromosomes
        self.assertIn("--size-mb", sim.command)

    def test_missing_rs01_build_does_not_stop_planning(self):
        tool = e3.rs01_tool(fake_ctx(config.SMOKE, self.runs, "c", "rs02"))
        self.assertEqual(tool.name, "psmc-rs-0.1")


class Report(unittest.TestCase):
    """The tables from stand-in runs: C psmc's reference output, and psmc-rs JSON made from it."""

    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cfg = self.cfg = replace(config.TINY.e3, inference=replace(config.TINY.e3.inference, n_iter=5), repeats=2)
        self.profile = replace(config.TINY, e3=cfg)
        lay = self.lay = e3.Layout(self.runs / "E3")
        tools = fake_tools()
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
                params = c_ref_json(1.001 if (u.tool == "rs02" and u.threads == 2 and run.repeat == 2) else 1.0)
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
            tool = tools["rs02" if u.tool == "rs02-default" else u.tool]
            lay.record(run).write_text(json.dumps({"key": {"tools": {tool.name: tool.identity}}}))
        self.written = self.report()

    def report(self, **kw):
        return e3.report(Context(self.profile, self.runs, **kw))

    def timing(self, run) -> dict:
        return json.loads(self.lay.timing(run).read_text())

    def set_timing(self, run, timing: dict) -> None:
        self.lay.timing(run).write_text(json.dumps(timing))

    def environment(self) -> dict:
        return {r["key"]: r["value"] for r in read_csv(self.lay.table("environment"))}

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
        env = self.environment()
        self.assertEqual((env["machine.cpu"], env["machine.cpu_count"]), ("Test CPU", "8"))
        self.assertEqual(env["tool.rs01.commit"], "e" * 40)
        self.assertEqual((env["runs_not_quiet"], env["runs_not_clean"]), ("0", "0"))

    def test_environment_lists_every_host_of_merged_shards(self):
        for run in e3.runs(self.cfg):
            timing = self.timing(run)
            timing["machine"]["hostname"] = f"cn{run.repeat}"  # repeat 2 ran on another node
            self.set_timing(run, timing)
        self.report()
        env = self.environment()
        self.assertEqual((env["machine.hostname"], env["machine.cpu"]), ("cn1;cn2", "Test CPU"))

    def test_runs_on_two_kinds_of_cpu_are_flagged(self):
        env = self.environment()
        self.assertEqual((env["clean_runs_on_mixed_machines"], env["clean_runs_machines"]),
                         ("False", "Test CPU (governor ?, boost ?)"))
        for run in e3.runs(self.cfg):
            timing = self.timing(run)
            rome = {"cpu": "AMD EPYC 7702 64-Core Processor", "cpu_governor": "performance", "cpu_boost": "1",
                    "cpus_allowed": 8, "hostname": "cn509-02-l", "slurm_job_id": "41000001"}
            turin = {**rome, "cpu": "AMD EPYC 9655 96-Core Processor", "hostname": "cn610-11-l",
                     "slurm_job_id": "41000002"}
            timing["machine"] = turin if run.repeat == 2 else rome  # repeat 2 ran on another kind of node
            self.set_timing(run, timing)
        logged = []
        self.report(log=logged.append)
        env = self.environment()
        self.assertEqual(env["clean_runs_on_mixed_machines"], "True")
        self.assertEqual(env["clean_runs_machines"],
                         "AMD EPYC 7702 64-Core Processor (governor performance, boost 1);"
                         "AMD EPYC 9655 96-Core Processor (governor performance, boost 1)")
        self.assertEqual(env["machine.hostname"], "cn509-02-l;cn610-11-l")
        self.assertTrue(any("2 kinds of machine" in m for m in logged))
        runs = read_csv(self.lay.table("runs"))
        self.assertEqual({(r["repeat"], r["cpu"], r["hostname"], r["slurm_job_id"]) for r in runs if r["tool"] == "c"},
                         {("1", "AMD EPYC 7702 64-Core Processor", "cn509-02-l", "41000001"),
                          ("2", "AMD EPYC 9655 96-Core Processor", "cn610-11-l", "41000002")})
        # Only clean runs count: with repeat 2 not clean, the medians are of one kind of machine.
        for run in e3.runs(self.cfg):
            if run.repeat == 2:
                timing = self.timing(run)
                timing["machine_before"]["quiet"] = False
                self.set_timing(run, timing)
        self.report(log=logged.append)
        env = self.environment()
        self.assertEqual(env["clean_runs_on_mixed_machines"], "False")

    def test_environment_says_how_c_psmc_was_built(self):
        build = {"compiler": "/usr/bin/gcc", "compiler_version": "gcc (GCC) 11.4.1 20231218 (Red Hat 11.4.1-3)",
                 "cflags": "-g -Wall -O2 -Wno-unused-function", "ldflags": "", "libs": "-lm -lz",
                 "platform": "Linux-5.14.0-x86_64-with-glibc2.34", "built": "2026-10-06T12:00:00"}
        with mock.patch.object(e3.tools, "read_c_build", return_value=build):
            self.report()
        env = self.environment()
        self.assertIn("'-g -Wall -O2 -Wno-unused-function'", env["tool.c.build"])
        self.assertNotIn("-O1", env["tool.c.build"])
        self.assertEqual(env["cc"], build["compiler_version"])  # BUILD.json's, not today's cc
        self.assertEqual(env["tool.c.build_platform"], build["platform"])
        with mock.patch.object(e3.tools, "read_c_build", return_value=None), \
                mock.patch.object(e3, "c_compiler", return_value="Apple clang version 17.0.0"):
            self.report()
        env = self.environment()
        self.assertEqual((env["tool.c.build"], env["cc"]), (e3.C_PSMC_BUILD, "Apple clang version 17.0.0"))
        self.assertNotIn("tool.c.build_platform", env)

    def test_a_run_with_other_work_beside_it_stays_out_of_the_medians(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "c" and r.repeat == 2)
        timing = self.timing(run)
        protocol = config.TimingProtocol()
        self.assertTrue(e3.is_clean(timing, protocol))
        # 1% of 192 cores is 1.9 cores of other work: within the share, not within the cores.
        timing["during"] = {"other_cpu_share": 0.01, "other_cores": 1.9}
        self.assertFalse(e3.is_clean(timing, protocol))
        # a record without other_cores: from the other processes' CPU seconds over the wall time
        timing["during"] = {"other_cpu_share": 0.01, "other_cpu_seconds": 0.6 * timing["wall_seconds"]}
        self.assertFalse(e3.is_clean(timing, protocol))
        timing["during"] = {"other_cpu_share": 0.01, "other_cpu_seconds": 0.4 * timing["wall_seconds"]}
        self.assertTrue(e3.is_clean(timing, protocol))
        timing["machine_before"]["busy_cores"] = 1.5
        self.assertFalse(e3.is_clean(timing, protocol))

    def test_a_run_is_judged_by_its_own_measure_of_other_work(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs02" and r.repeat == 1)
        timing = self.timing(run)
        protocol = config.TimingProtocol()
        # timing/3: the machine's busy CPU but the tool's, interrupts included (a 30-thread run on Ibex)
        timing["during"] = {"other_cpu_share": 0.0044, "other_cores": 0.56, "other_cpu_seconds": 56.0}
        self.assertFalse(e3.is_clean(timing, protocol))
        # timing/4 on Linux: other user-space processes; kernel threads and interrupts recorded apart
        timing["record"] = "psmc-rs-manuscript/E3/timing/4"
        timing["machine_before"].update(process_cores=0.2, kernel_thread_cores=0.1, interrupt_cores=0.1)
        timing["during"] = {
            "other_method": "processes", "other_process_cpu_seconds": 20.0, "other_process_cpu_share": 0.0016,
            "other_process_cores": 0.2, "other_processes_exited": 2, "processes_unreadable": 0,
            "kernel_thread_cores": 0.15, "interrupt_cores": 0.2, "steal_cores": 0.0, "unaccounted_cores": 0.01,
            "other_cpu_seconds": 56.0, "other_cpu_share": 0.0044, "other_cores": 0.56, "machine_busy_percent": 23.9,
        }
        self.assertTrue(e3.is_clean(timing, protocol))
        self.assertFalse(e3.is_clean({**timing, "during": {**timing["during"], "other_process_cores": 0.6}}, protocol))
        self.assertFalse(e3.is_clean({**timing, "during": {**timing["during"], "other_process_cpu_share": 0.06}},
                                     protocol))
        # timing/4 where not every process could be read: the machine-wide measure
        fallback = {**timing["during"], "other_method": "machine", "other_process_cores": None,
                    "other_process_cpu_share": None, "processes_unreadable": 155}
        self.assertFalse(e3.is_clean({**timing, "during": fallback}, protocol))
        self.assertTrue(e3.is_clean({**timing, "during": {**fallback, "other_cores": 0.4}}, protocol))
        self.set_timing(run, timing)
        self.report()
        row = next(r for r in read_csv(self.lay.table("runs"))
                   if (r["tool"], r["threads"], r["repeat"]) == ("rs02", str(run.setup.threads), "1"))
        self.assertEqual((row["clean"], row["other_method"]), ("True", "processes"))
        self.assertEqual((row["other_cores_during"], row["machine_other_cores_during"]), ("0.2", "0.56"))
        self.assertEqual((row["kernel_thread_cores_during"], row["interrupt_cores_during"]), ("0.15", "0.2"))
        self.assertEqual((row["process_cores_before"], row["other_processes_exited"]), ("0.2", "2"))
        old = next(r for r in read_csv(self.lay.table("runs")) if r["tool"] == "c")
        self.assertEqual((old["other_method"], old["kernel_thread_cores_during"]), ("machine", ""))
        env = self.environment()
        self.assertEqual(env["other_methods"], "machine;processes")
        # other_cores_during is a different quantity by method: summarised apart
        self.assertEqual(env["cores.other_cores_during.processes"], "median 0.200, max 0.200 over 1 runs")
        self.assertTrue(env["cores.other_cores_during.machine"].endswith(f" over {len(e3.runs(self.cfg)) - 1} runs"))
        self.assertNotIn("cores.other_cores_during", env)
        self.assertEqual(env["cores.interrupt_cores_during"], "median 0.200, max 0.200 over 1 runs")
        self.assertEqual(env["cores.machine_other_cores_during"].split(", max")[1], " 0.560 over "
                         f"{len(e3.runs(self.cfg))} runs")
        self.assertEqual(env["other_processes_exited"], "2 over 1 runs")

    def test_a_run_with_more_threads_than_usable_cpus_is_not_clean(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs02" and r.setup.threads == 2)
        timing = self.timing(run)
        protocol = config.TimingProtocol()
        self.assertTrue(e3.is_clean(timing, protocol))
        self.assertTrue(e3.is_clean({**timing, "threads": 30, "cpus_usable": 30}, protocol))
        self.assertFalse(e3.is_clean({**timing, "threads": 30, "cpus_usable": 16}, protocol))
        # a record without cpus_usable: the cpus_allowed of a Linux machine
        self.assertFalse(e3.is_clean({**timing, "threads": 30, "machine": {"cpus_allowed": 16}}, protocol))
        self.set_timing(run, {**timing, "threads": 30, "cpus_usable": 16})
        self.report()
        row = next(r for r in read_csv(self.lay.table("runs"))
                   if (r["tool"], r["threads"], r["repeat"]) == ("rs02", "2", str(run.repeat)))
        self.assertEqual((row["clean"], row["cpus_usable"]), ("False", "16"))

    def test_a_fit_of_another_genome_is_refused(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs02")
        params = json.loads(self.lay.fit(run).read_text())
        params["input"]["bins"] += 1
        self.lay.fit(run).write_text(json.dumps(params))
        with self.assertRaises(ValueError):
            self.report()

    def test_a_run_on_a_busy_machine_stays_out_of_the_medians(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "c" and r.repeat == 2)
        timing = self.timing(run)
        timing["machine_before"]["quiet"] = False
        self.set_timing(run, timing)
        self.report()
        c = next(r for r in read_csv(self.lay.table("table2")) if r["tool"] == "c")
        self.assertEqual((c["status"], c["repeats"], c["repeats_measured"]), ("incomplete", "1", "2"))
        self.assertAlmostEqual(float(c["wall_median_s"]), 10.1)  # repeat 1 only
        runs = [r for r in read_csv(self.lay.table("runs")) if r["tool"] == "c"]
        self.assertEqual(sorted(r["clean"] for r in runs), ["False", "True"])

    def test_timing_runs_are_ranked_by_repeat(self):
        jobs = e3.jobs(fake_ctx(self.profile, self.runs))
        runs = [j for j in jobs if j.exclusive]
        self.assertTrue(runs)
        self.assertTrue(all(j.rank == int(j.id.rsplit("rep", 1)[1]) for j in runs))

    def test_partial_results_are_reported_as_incomplete(self):
        run = next(r for r in e3.runs(self.cfg) if r.setup.tool == "rs01")
        self.lay.timing(run).unlink()
        self.report()
        row = next(r for r in read_csv(self.lay.table("table2")) if r["tool"] == "rs01")
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["repeats"], "1")


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    """Git keeps E3's fits, timings, truth, provenance and tables, not its genomes or HTML reports."""

    def test_what_git_keeps(self):
        lay = e3.Layout(ROOT / "experiment" / "runs" / "E3")
        c = e3.Run(e3.Setup("c", 3000, 1), 3)
        rs = e3.Run(e3.Setup("rs02", 3000, 8), 1)
        assert_ignore_rules(self, kept=(lay.fit(c), lay.fit(rs), lay.timing(rs), lay.record(rs), lay.truth(3000),
                                        lay.table("table2"), lay.root / "provenance" / "sim" / "3000Mb.json"),
                            ignored=(lay.psmcfa(3000), lay.fit(rs).with_suffix(".html"),
                                     lay.root / "logs" / "attempts.jsonl"))


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
