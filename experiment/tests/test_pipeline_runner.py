"""Tests for experiment.pipeline.runner and provenance, with small stand-in jobs."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, has_modules, quick_runner  # noqa: E402

from experiment.pipeline import config, provenance, runner  # noqa: E402
from experiment.pipeline.__main__ import main  # noqa: E402
from experiment.pipeline.runner import Graph, Job  # noqa: E402

HAS_PSUTIL = has_modules("psutil")

FAKE_JOB = textwrap.dedent(
    """
    import json, os, sys, time
    a = json.loads(sys.argv[1])
    start = time.time()
    if a.get("fail_if") and os.path.exists(a["fail_if"]):
        sys.exit(3)
    buf = bytearray(a.get("alloc_mb", 0) * 2**20)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    time.sleep(a.get("sleep", 0))
    data = "".join(open(p).read() for p in a.get("inputs", []))
    for out in a["outputs"]:
        with open(out, "w") as f:
            json.dump({"start": start, "end": time.time(), "in": data, "tag": a.get("tag", "")}, f)
    with open(a["count"], "a") as f:
        f.write(a["id"] + "\\n")
    """
)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.runs = self.tmp / "runs"
        self.script = self.tmp / "fake_job.py"
        self.script.write_text(FAKE_JOB)
        self.count = self.tmp / "count.txt"

    def job(self, jid: str, outputs, inputs=(), sleep=0.0, tag="", fail_if=None, alloc_mb=0, **kw) -> Job:
        outs = [self.tmp / o for o in outputs]
        ins = [self.tmp / i for i in inputs]
        args = {"id": jid, "outputs": [str(p) for p in outs], "inputs": [str(p) for p in ins], "sleep": sleep,
                "tag": tag, "count": str(self.count), "alloc_mb": alloc_mb}
        if fail_if:
            args["fail_if"] = str(self.tmp / fail_if)
        return Job(id=jid, command=[sys.executable, str(self.script), json.dumps(args)], inputs=ins, outputs=outs,
                   est_seconds=max(sleep, 0.1), **kw)

    def runs_of(self, jid: str) -> int:
        if not self.count.exists():
            return 0
        return self.count.read_text().split().count(jid)

    def run_daemon(self, jobs, budget=config.Budget()):
        d = runner.Daemon(Graph(jobs, self.runs), self.runs, budget, runner.Logger(self.runs),
                          {"budget": {"cores": budget.cores, "mem_gb": budget.mem_gb}})
        with quick_runner():  # the jobs take less than a second
            return d, d.run()

    def span(self, name: str):
        d = json.loads((self.tmp / name).read_text())
        return d["start"], d["end"]


class GraphTests(Base):
    def test_dependencies_from_files(self):
        a = self.job("T/a", ["a.txt"])
        b = self.job("T/b", ["b.txt"], inputs=["a.txt"])
        c = self.job("T/c", ["c.txt"], inputs=["a.txt", "b.txt", "static.txt"])
        g = Graph([c, b, a], self.runs)
        self.assertEqual(g.deps["T/c"], ["T/a", "T/b"])
        self.assertEqual(g.order.index("T/a") < g.order.index("T/b") < g.order.index("T/c"), True)
        self.assertEqual(g.static_inputs["T/c"], [self.tmp / "static.txt"])
        self.assertGreater(g.priority["T/a"], g.priority["T/c"])

    def test_errors(self):
        with self.assertRaises(runner.GraphError):
            Graph([self.job("T/a", ["x"]), self.job("T/b", ["x"])], self.runs)
        with self.assertRaises(runner.GraphError):
            Graph([self.job("T/a", ["x"], inputs=["y"]), self.job("T/b", ["y"], inputs=["x"])], self.runs)
        with self.assertRaises(ValueError):
            Job(id="no-experiment", command=["true"])
        with self.assertRaisesRegex(ValueError, "at least one core"):
            Job(id="T/a", command=["true"], cores=0)

    def test_a_file_reached_through_a_symbolic_link_is_one_file(self):
        (self.tmp / "real").mkdir()
        (self.tmp / "linked").symlink_to(self.tmp / "real", target_is_directory=True)
        with self.assertRaises(runner.GraphError):
            Graph([self.job("T/a", ["real/x"]), self.job("T/b", ["linked/x"])], self.runs)
        (self.tmp / "x-link").symlink_to(self.tmp / "real" / "x")
        g = Graph([self.job("T/a", ["real/x"]), self.job("T/b", ["b"], inputs=["linked/x"]),
                   self.job("T/c", ["c"], inputs=["x-link"])], self.runs)
        self.assertEqual((g.deps["T/b"], g.deps["T/c"]), (["T/a"], ["T/a"]))

    @unittest.skipUnless(sys.platform == "darwin", "case-insensitive file systems")
    def test_outputs_that_differ_only_in_case_collide(self):
        # FIG's figS2_format_consistency.pdf and E6's FigS2_format_consistency.pdf were one file.
        with self.assertRaises(runner.GraphError):
            Graph([self.job("T/a", ["figS2.pdf"]), self.job("T/b", ["FigS2.pdf"])], self.runs)

    def test_record_paths(self):
        j = self.job("E9/fit-c/model/rep01", ["o"])
        self.assertEqual(j.record_path(self.runs), self.runs / "E9" / "provenance" / "fit-c" / "model" / "rep01.json")
        self.assertEqual(j.log_path(self.runs), self.runs / "E9" / "logs" / "jobs" / "fit-c" / "model" / "rep01.log")
        self.assertEqual(Job.from_dict(j.to_dict()), j)


class Scheduling(unittest.TestCase):
    def mk(self, jid, cores=1, mem=1.0, est=10.0, exclusive=False):
        return Job(id=jid, command=["true"], cores=cores, mem_gb=mem, est_seconds=est, exclusive=exclusive)

    def test_budget(self):
        jobs = [self.mk(f"T/{i}") for i in range(5)]
        got = runner.pick(jobs, [], config.Budget(cores=3, mem_gb=100), {}, 0.0)
        self.assertEqual(len(got), 3)
        got = runner.pick(jobs, [], config.Budget(cores=8, mem_gb=2.5), {}, 0.0)
        self.assertEqual(len(got), 2)

    def test_exclusive_runs_alone_and_last(self):
        ex = self.mk("T/x", exclusive=True)
        normal = self.mk("T/n")
        self.assertEqual(runner.pick([ex, normal], [], config.Budget(), {}, 0.0), [normal])
        self.assertEqual(runner.pick([ex], [runner.Slot(normal, 0.0)], config.Budget(), {}, 0.0), [])
        self.assertEqual(runner.pick([ex], [], config.Budget(), {}, 0.0), [ex])
        self.assertEqual(runner.pick([normal], [runner.Slot(ex, 0.0)], config.Budget(), {}, 0.0), [])

    def test_exclusive_jobs_start_in_rank_order(self):
        # E3's repeats: every setup's first repeat, then the second, whatever the chain lengths.
        jobs = [Job(id=f"T/{s}-rep{r}", command=["true"], est_seconds=10.0, exclusive=True, rank=r)
                for s in ("a", "b") for r in (1, 2)]
        prio = {"T/a-rep1": 50.0, "T/a-rep2": 50.0, "T/b-rep1": 10.0, "T/b-rep2": 10.0}
        order = []
        ready = list(jobs)
        while ready:
            (got,) = runner.pick(ready, [], config.Budget(), prio, 0.0)
            order.append(got.id)
            ready.remove(got)
        self.assertEqual(order, ["T/a-rep1", "T/b-rep1", "T/a-rep2", "T/b-rep2"])

    def test_an_exclusive_job_needs_the_memory_budget_not_the_cores(self):
        budget = config.Budget(cores=7, mem_gb=12.0)
        self.assertIn("30 cores", runner.fits_budget(self.mk("T/n", cores=30), budget))
        ex = self.mk("T/x", cores=30, exclusive=True)
        self.assertIsNone(runner.fits_budget(ex, budget))  # it runs alone
        self.assertEqual(runner.pick([ex], [], budget, {}, 0.0), [ex])
        self.assertIn("14.6 GB", runner.fits_budget(self.mk("T/y", cores=30, mem=14.6, exclusive=True), budget))

    def test_machine_lock_is_shared_by_ordinary_jobs_and_held_alone_by_exclusive_ones(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "machine.lock"
            a, b = runner.MachineLock(path), runner.MachineLock(path)
            try:
                self.assertTrue(a.hold("shared"))
                self.assertTrue(b.hold("shared"))
                self.assertFalse(b.hold("exclusive"))  # a runs ordinary jobs
                b.release()
                a.release()
                self.assertTrue(a.hold("exclusive"))
                self.assertFalse(b.hold("shared"))  # a runs an exclusive job
                self.assertFalse(b.hold("exclusive"))
                a.release()
                self.assertTrue(b.hold("exclusive"))
            finally:
                a.release()
                b.release()

    def test_priority_and_backfill(self):
        big = self.mk("T/big", mem=8.0, est=100.0)
        small_short = self.mk("T/s1", est=5.0)
        small_long = self.mk("T/s2", est=500.0)
        running = [runner.Slot(self.mk("T/r", mem=6.0, est=20.0), 0.0)]
        prio = {"T/big": 1000.0, "T/s1": 1.0, "T/s2": 2.0}
        got = runner.pick([small_long, big, small_short], running, config.Budget(cores=7, mem_gb=12), prio, 0.0)
        # big does not fit; only the job that ends before the running one may jump ahead
        self.assertEqual(got, [small_short])

    def test_schedule_estimate(self):
        jobs = [self.mk("T/a", est=10), self.mk("T/b", est=10), Job(id="T/c", command=["true"], est_seconds=5)]
        g = Graph(jobs, Path("/tmp/none"))
        fin = runner.simulate_schedule(g, ["T/a", "T/b", "T/c"], config.Budget(cores=1, mem_gb=10), {})
        self.assertEqual(sorted(fin.values()), [10, 20, 25])
        fin = runner.simulate_schedule(g, ["T/a", "T/b", "T/c"], config.Budget(cores=3, mem_gb=10), {})
        self.assertEqual(max(fin.values()), 10)
        p = runner.plan(g, config.Budget(cores=3, mem_gb=10))
        self.assertAlmostEqual(p["T"]["cpu_hours"], 25 / 3600)


class Status(Base):
    def statuses(self, jobs, **kw):
        return {s.job.id: (s.state, s.reason) for s in runner.job_statuses(Graph(jobs, self.runs), self.runs, **kw)}

    def test_a_job_the_budget_cannot_hold_is_blocked_and_so_is_what_needs_it(self):
        jobs = [self.job("T/big", ["big.txt"], cores=8), self.job("T/after", ["after.txt"], inputs=["big.txt"]),
                self.job("T/other", ["other.txt"])]
        got = self.statuses(jobs, daemon_pid=None, budget=config.Budget(cores=7))
        self.assertEqual((got["T/big"], got["T/after"]),
                         ((runner.BLOCKED, "needs 8 cores, budget 7"), (runner.BLOCKED, "needs T/big")))
        self.assertEqual(got["T/other"], (runner.PENDING, "not run yet"))
        g, budget = Graph(jobs, self.runs), config.Budget(cores=7)
        report = runner.format_status(g, runner.job_statuses(g, self.runs, daemon_pid=None, budget=budget), budget,
                                      self.runs, None)
        self.assertRegex(report, r"\nT\s+0\s+0\s+0\s+2\s+1\s+never\s")  # T never finishes, though T/other can run
        got = self.statuses(jobs, daemon_pid=None, budget=config.Budget(cores=8))
        self.assertEqual({state for state, _ in got.values()}, {runner.PENDING})

    def test_a_few_jobs_of_each_state_are_listed_unless_all_are_asked_for(self):
        n = runner.STATUS_LINES + 3
        g = Graph([self.job(f"T/j{i:02d}", [f"j{i}.txt"], inputs=[f"absent{i}.txt"]) for i in range(n)], self.runs)
        statuses = runner.job_statuses(g, self.runs, daemon_pid=None)
        short = runner.format_status(g, statuses, config.Budget(), self.runs, None)
        self.assertRegex(short, rf"\nT\s+0\s+0\s+0\s+{n}\s+0\s+never\s")  # every job blocked: not "nothing left"
        self.assertEqual(short.count("\n  blocked  T/"), runner.STATUS_LINES)
        self.assertIn("\n  ... and 3 more blocked jobs of T (--jobs lists them)", short)
        full = runner.format_status(g, statuses, config.Budget(), self.runs, None, verbose=True)
        self.assertEqual(full.count("\n  blocked  T/"), n)


class CommandLine(Base):
    """What a user sees: errors without tracebacks, experiments by name, the plan of what cannot run."""

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main([*args, "--runs-dir", str(self.runs)])
            except SystemExit as exc:  # argparse's status 2, or a message that Python prints with status 1
                rc = exc.code if isinstance(exc.code, int) else 1
                err.write("" if isinstance(exc.code, int) else exc.code)
        return rc, out.getvalue(), err.getvalue()

    def jobs_file(self, *jobs) -> str:
        spec = self.tmp / "jobs.json"
        spec.write_text(json.dumps([j.to_dict() for j in jobs]))
        return str(spec)

    def test_the_plan_names_what_never_starts_on_the_budget(self):
        spec = self.jobs_file(self.job("T/big", ["big.txt"], cores=8), self.job("T/small", ["small.txt"]))
        rc, out, _ = self.cli("plan", "--jobs-file", spec)
        self.assertEqual(rc, 0)
        self.assertRegex(out, r"\nT\s+2\s+[\d.]+\s+never\s+never\n")
        self.assertIn("\n  T/big (1 job): needs 8 cores, budget 7\n", out)
        rc, out, _ = self.cli("plan", "--jobs-file", spec, "--cores", "8")
        self.assertNotIn("never", out)

    def test_experiments_are_checked_by_name_in_any_case(self):
        from experiment.pipeline.__main__ import build_parser

        self.assertEqual(build_parser().parse_args(["plan", "e1", "fig"]).experiments, ["E1", "FIG"])
        rc, _, err = self.cli("list", "E9")
        self.assertEqual(rc, 2)
        self.assertIn("invalid choice: 'E9'", err)

    def test_errors_are_messages_not_tracebacks(self):
        rc, _, err = self.cli("list", "--jobs-file", str(self.tmp / "absent.json"))
        self.assertEqual((rc, err), (1, f"--jobs-file {self.tmp / 'absent.json'}: [Errno 2] No such file or "
                                        f"directory: '{self.tmp / 'absent.json'}'"))
        from experiment.pipeline import __main__ as cli, tools

        with mock.patch.object(cli, "make_graph", side_effect=tools.ToolError("C psmc not found at /x/psmc")):
            self.assertEqual(self.cli("list"), (1, "", "C psmc not found at /x/psmc\n"))


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class DaemonTests(Base):
    def test_runs_records_and_resumes(self):
        (self.tmp / "static.txt").write_text("v1")
        a = self.job("T/a", ["a.txt"], inputs=["static.txt"])
        b = self.job("T/b", ["b.txt"], inputs=["a.txt"])
        c = self.job("T/c", ["c.txt"])
        d, rc = self.run_daemon([a, b, c])
        self.assertEqual(rc, 0)
        self.assertEqual(set(d.state.values()), {runner.DONE})
        rec = json.loads(a.record_path(self.runs).read_text())
        self.assertEqual(rec["run"]["exit_status"], 0)
        self.assertEqual(rec["key"]["command"][0], "<python>")  # machine-specific prefixes are replaced
        self.assertEqual(list(rec["key"]["inputs"].values()), [provenance.sha256_file(self.tmp / "static.txt")])
        for k in ("wall_seconds", "peak_rss_mb", "loadavg_before", "loadavg_after", "user_seconds"):
            self.assertIn(k, rec["run"])
        # Nothing runs again.
        self.run_daemon([a, b, c])
        self.assertEqual([self.runs_of(j) for j in ("T/a", "T/b", "T/c")], [1, 1, 1])
        # A changed input re-runs its job and, through the new output, the next one.
        (self.tmp / "static.txt").write_text("v2")
        self.run_daemon([a, b, c])
        self.assertEqual([self.runs_of(j) for j in ("T/a", "T/b", "T/c")], [2, 2, 1])
        # A changed setting re-runs only that job.
        c2 = self.job("T/c", ["c.txt"], settings={"x": 1})
        self.run_daemon([a, b, c2])
        self.assertEqual([self.runs_of(j) for j in ("T/a", "T/b", "T/c")], [2, 2, 2])
        # An output edited after the run is not trusted.
        (self.tmp / "c.txt").write_text("edited")
        ok, why = provenance.check_done(c2, self.runs, provenance.ChecksumCache(None))
        self.assertFalse(ok)
        self.assertIn("output changed", why)

    def test_failure_is_recorded_blocks_dependents_and_is_retried(self):
        (self.tmp / "flag").write_text("")
        f = self.job("T/f", ["f.txt"], fail_if="flag")
        g = self.job("T/g", ["g.txt"], inputs=["f.txt"])
        d, rc = self.run_daemon([f, g])
        self.assertEqual(rc, 1)
        self.assertEqual((d.state["T/f"], d.state["T/g"]), (runner.FAILED, runner.BLOCKED))
        attempts = runner.last_attempts(self.runs, ["T"])
        self.assertEqual(attempts["T/f"]["status"], runner.FAILED)
        self.assertIn("exit status 3", attempts["T/f"]["error"])
        states = {s.job.id: s.state for s in runner.job_statuses(Graph([f, g], self.runs), self.runs)}
        self.assertEqual(states["T/f"], runner.FAILED)
        (self.tmp / "flag").unlink()
        d, rc = self.run_daemon([f, g])
        self.assertEqual(rc, 0)
        self.assertEqual(self.runs_of("T/g"), 1)

    def test_budget_and_exclusive_jobs(self):
        jobs = [self.job(f"T/n{i}", [f"n{i}.txt"], sleep=0.4) for i in range(4)]
        jobs.append(self.job("T/x", ["x.txt"], sleep=0.3, exclusive=True))
        self.run_daemon(jobs, config.Budget(cores=2, mem_gb=10))
        spans = [self.span(f"n{i}.txt") for i in range(4)]
        events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
        level = peak = 0
        for _, step in events:
            level += step
            peak = max(peak, level)
        self.assertLessEqual(peak, 2)
        xs, xe = self.span("x.txt")
        self.assertTrue(all(e <= xs or s >= xe for s, e in spans))
        rec = json.loads(jobs[-1].record_path(self.runs).read_text())
        self.assertIn("busiest_processes_before", rec["run"])

    def test_exit_status_75_defers_the_job_instead_of_failing_it(self):
        flag = self.tmp / "busy"
        flag.write_text("")
        script = self.tmp / "busy.py"
        script.write_text(
            "import os, sys\n"
            f"if os.path.exists({str(flag)!r}):\n"
            f"    os.remove({str(flag)!r}); print('machine busy'); print('not quiet'); sys.exit(75)\n"
            f"open({str(self.tmp / 'd.txt')!r}, 'w').write('done')\n"
        )
        j = Job(id="T/d", command=[sys.executable, str(script)], outputs=[self.tmp / "d.txt"], exclusive=True)
        with mock.patch.object(runner, "DEFER_SECONDS", 0.2):
            d, rc = self.run_daemon([j])
        self.assertEqual((rc, d.state["T/d"], d.deferrals["T/d"]), (0, runner.DONE, 1))
        statuses = [json.loads(line)["status"] for line in runner.attempts_path(self.runs, "T").read_text().splitlines()]
        self.assertEqual(statuses, [runner.DEFERRED, runner.DONE])
        rec = json.loads(j.record_path(self.runs).read_text())
        self.assertIsNone(rec["run"]["peak_rss_tree_sampled_mb"])  # exclusive jobs are not sampled

    def test_peak_rss_of_the_job(self):
        j = self.job("T/mem", ["m.txt"], alloc_mb=80, sleep=0.3)
        self.run_daemon([j])
        rec = json.loads(j.record_path(self.runs).read_text())
        self.assertGreater(rec["run"]["peak_rss_mb"], 70)

    def test_a_job_whose_static_input_is_missing_is_blocked_not_pending(self):
        j = self.job("T/s", ["s.txt"], inputs=["never_made.txt"])
        statuses = runner.job_statuses(Graph([j], self.runs), self.runs, daemon_pid=None)
        self.assertEqual(statuses[0].state, runner.BLOCKED)
        self.assertIn("no job makes it", statuses[0].reason)
        self.assertFalse((self.runs / "_runner").exists())  # status creates nothing

    def test_key_differences_compare_as_json(self):
        old = json.loads(json.dumps({"settings": {"a": [1, 2]}, "inputs": {}}))
        self.assertEqual(provenance.key_differences(old, {"settings": {"a": (1, 2)}, "inputs": {}}), [])
        self.assertEqual(provenance.key_differences(old, {"settings": {"a": (1, 3)}, "inputs": {}}), ["settings"])

    def test_unschedulable_job_is_blocked(self):
        j = self.job("T/huge", ["h.txt"], mem_gb=64.0)
        d, rc = self.run_daemon([j], config.Budget(cores=2, mem_gb=12))
        self.assertEqual(d.state["T/huge"], runner.BLOCKED)
        self.assertIn("64 GB", d.reason["T/huge"])

    def test_executables_are_recorded_and_a_changed_one_stops_new_jobs(self):
        tool = self.tmp / "tool.py"
        tool.write_text("#!" + sys.executable + "\n" + FAKE_JOB)
        tool.chmod(0o755)
        a = self.job("T/a", ["a.txt"])
        a.command = [str(tool), *a.command[2:]]
        # b rewrites the tool (a rebuild during the run); c, after it, must not start.
        b = Job(id="T/b", command=[sys.executable, "-c", f"open({str(tool)!r}, 'a').write('# rebuilt\\\\n'); "
                                   f"open({str(self.tmp / 'b.txt')!r}, 'w').write('b')"],
                inputs=[self.tmp / "a.txt"], outputs=[self.tmp / "b.txt"])
        c = self.job("T/c", ["c.txt"], inputs=["b.txt"])
        c.command = [str(tool), *c.command[2:]]
        d, rc = self.run_daemon([a, b, c])
        self.assertEqual(rc, 1)
        self.assertEqual((d.state["T/a"], d.state["T/b"]), (runner.DONE, runner.DONE))
        self.assertNotEqual(d.state["T/c"], runner.DONE)
        self.assertEqual(self.runs_of("T/c"), 0)
        self.assertIn("tool.py changed since the runner started", d.halted)
        rec = json.loads(a.record_path(self.runs).read_text())
        self.assertEqual(list(rec["run"]["executables"].values()), [d.watch.snapshot[tool.resolve()]])


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class DetachedRunner(Base):
    def cli(self, *args, **extra_env) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(ROOT), PSMC_EXP_MACHINE_LOCK=str(self.tmp / "machine.lock"),
                   **extra_env)
        return subprocess.run([sys.executable, "-m", "experiment.pipeline", *args, "--runs-dir", str(self.runs)],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)

    def wait_for(self, pred, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(0.1)
        return False

    def test_start_status_stop_resume(self):
        jobs = [self.job(f"T/s{i}", [f"s{i}.txt"], sleep=2.0 if i < 2 else 0.1) for i in range(3)]
        spec = self.tmp / "jobs.json"
        spec.write_text(json.dumps([j.to_dict() for j in jobs]))
        res = self.cli("start", "--jobs-file", str(spec), "--cores", "2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("runner started", res.stdout)
        self.assertTrue(self.wait_for(lambda: (self.runs / "_runner" / "state.json").exists()
                                      and json.loads((self.runs / "_runner" / "state.json").read_text()).get("running")))
        res = self.cli("start", "--jobs-file", str(spec))
        self.assertEqual(res.returncode, 1)  # one runner per runs directory
        st = self.cli("status")
        self.assertIn("runner: running", st.stdout)
        self.assertRegex(st.stdout, r"T\s+\d+\s+[12]\s+0")
        res = self.cli("stop")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIsNone(runner.daemon_alive(self.runs))
        attempts = [json.loads(x) for x in runner.attempts_path(self.runs, "T").read_text().splitlines()]
        self.assertIn("interrupted", {a["status"] for a in attempts})
        st = self.cli("status", "--jobs-file", str(spec))
        self.assertIn("runner: not running", st.stdout)
        self.assertNotIn("failed  T/", st.stdout)
        # Resume: the interrupted jobs run again, the finished one does not.
        done_before = {j.id for j in jobs if runner.provenance.check_done(j, self.runs, provenance.ChecksumCache(None))[0]}
        res = self.cli("start", "--jobs-file", str(spec), "--cores", "2")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertTrue(self.wait_for(lambda: runner.daemon_alive(self.runs) is None, timeout=60))
        for j in jobs:
            self.assertTrue(provenance.check_done(j, self.runs, provenance.ChecksumCache(None))[0], j.id)
            self.assertEqual(self.runs_of(j.id), 1)
        self.assertLessEqual(len(done_before), 1)
        self.assertIn("runner finished: 3 done", (self.runs / "_runner" / "runner.log").read_text())
        # With the runner gone, status without arguments reports the jobs the last runner ran.
        st = self.cli("status")
        self.assertIn("runner: not running", st.stdout)
        self.assertRegex(st.stdout, r"\nT\s+3\s+0\s+0\s+0\s+0")

    def test_only_runs_and_shows_the_matching_jobs(self):
        # A shard: a and the rep2 jobs; b2 reads b.txt, which only the dropped b makes.
        jobs = [self.job("T/a", ["a.txt"]), self.job("T/b", ["b.txt"]),
                self.job("T/run/x/rep1", ["x1.txt"], inputs=["a.txt"]),
                self.job("T/run/x/rep2", ["x2.txt"], inputs=["a.txt"]),
                self.job("T/run/y/rep2", ["y2.txt"], inputs=["b.txt"])]
        spec = self.tmp / "jobs.json"
        spec.write_text(json.dumps([j.to_dict() for j in jobs]))
        only = "T/(a$|run/.*/rep2$)"
        listed = self.cli("list", "--jobs-file", str(spec), "--only", only)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertEqual(re.findall(r"^(T/\S+)  \[", listed.stdout, re.M), ["T/a", "T/run/x/rep2", "T/run/y/rep2"])
        planned = self.cli("plan", "--jobs-file", str(spec), "--only", only)
        self.assertRegex(planned.stdout, r"\nT\s+3\s")
        res = self.cli("start", "--foreground", "--jobs-file", str(spec), "--only", only)
        self.assertEqual(res.returncode, 1, res.stderr)  # T/run/y/rep2 is blocked
        self.assertEqual([self.runs_of(j.id) for j in jobs], [1, 0, 0, 1, 0])
        state = json.loads((self.runs / "_runner" / "state.json").read_text())
        self.assertEqual(state["only"], only)
        self.assertIn("input missing", state["blocked"]["T/run/y/rep2"])
        # status without arguments: the last runner's jobs and filter.
        st = self.cli("status", "--jobs")
        self.assertRegex(st.stdout, r"\nT\s+2\s+0\s+0\s+1\s+0")
        self.assertIn("input missing, and no job makes it", st.stdout)
        self.assertNotIn("T/b ", st.stdout)
        self.assertIn(f"(only the 3 jobs that --only {only!r} of the last runner keeps; --only '' shows all)",
                      st.stdout)
        st = self.cli("status", "--only", "")  # all 5 jobs: a and x/rep2 done, the others pending
        self.assertRegex(st.stdout, r"\nT\s+2\s+0\s+0\s+0\s+3\s")
        self.assertNotIn("--only", st.stdout)
        # Once the file exists (merged in from another shard), the kept job runs.
        (self.tmp / "b.txt").write_text("from another runs directory")
        res = self.cli("start", "--foreground", "--jobs-file", str(spec), "--only", only)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual([self.runs_of(j.id) for j in jobs], [1, 0, 0, 1, 1])
        # Without the filter, the jobs it kept are done and the others run; y/rep2
        # again, because b now makes the b.txt it reads, with another content.
        res = self.cli("start", "--foreground", "--jobs-file", str(spec))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual([self.runs_of(j.id) for j in jobs], [1, 1, 1, 1, 2])

    def test_a_bad_sample_interval_stops_only_start(self):
        spec = self.tmp / "jobs.json"
        spec.write_text(json.dumps([self.job("T/a", ["a.txt"]).to_dict()]))
        bad = {runner.SAMPLE_SECONDS_ENV: "fast"}
        for args in (("status", "--jobs-file", str(spec)), ("stop",)):
            res = self.cli(*args, **bad)
            self.assertEqual(res.returncode, 0, res.stderr)
        res = self.cli("start", "--foreground", "--jobs-file", str(spec), **bad)
        self.assertEqual((res.returncode, self.runs_of("T/a")), (1, 0))
        self.assertEqual(res.stderr.strip(), f"{runner.SAMPLE_SECONDS_ENV}='fast': give a number of seconds, "
                         f"at least {runner.MIN_SAMPLE_SECONDS:g}")

    def test_only_rejects_a_bad_or_unmatched_pattern(self):
        spec = self.tmp / "jobs.json"
        spec.write_text(json.dumps([self.job("T/a", ["a.txt"]).to_dict()]))
        res = self.cli("list", "--jobs-file", str(spec), "--only", "T/(")
        self.assertEqual(res.returncode, 2)
        self.assertIn("not a regular expression", res.stderr)
        res = self.cli("start", "--jobs-file", str(spec), "--only", "T/rep02$")
        self.assertEqual(res.returncode, 1)
        self.assertIn("matches none of the 1 jobs", res.stderr)
        self.assertIsNone(runner.daemon_alive(self.runs))


class Selection(unittest.TestCase):
    def test_select_by_regex_search(self):
        jobs = [Job(id=i, command=["true"]) for i in ("E3/sim/500Mb", "E3/run/c/500Mb/t1/rep1",
                                                       "E3/run/c/500Mb/t1/rep2", "E3/run/rs02/50Mb/t8/rep12", "E3/report")]
        ids = [j.id for j in runner.select(jobs, r"E3/(sim/|run/.*/rep2$)")]
        self.assertEqual(ids, ["E3/sim/500Mb", "E3/run/c/500Mb/t1/rep2"])
        self.assertEqual(runner.select(jobs, None), jobs)
        self.assertEqual([j.id for j in runner.select(jobs, "report")], ["E3/report"])

    def test_a_kept_job_whose_input_a_dropped_job_makes_reads_it_as_a_static_input(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d).resolve()
            make = Job(id="T/make", command=["true"], outputs=[d / "x.txt"])
            use = Job(id="T/use", command=["true"], inputs=[d / "x.txt"], outputs=[d / "y.txt"])
            g = Graph(runner.select([make, use], "use"), d / "runs")
            self.assertEqual((list(g.jobs), g.deps["T/use"], g.static_inputs["T/use"]), (["T/use"], [], [d / "x.txt"]))
            (status,) = runner.job_statuses(g, d / "runs", daemon_pid=None)
            self.assertEqual(status.state, runner.BLOCKED)


class MachineInfo(unittest.TestCase):
    """What the runner records about the machine and the SLURM job."""

    def test_linux_machine_reads_the_cpu_model_and_memory(self):
        with tempfile.TemporaryDirectory() as d:
            proc = Path(d)
            (proc / "cpuinfo").write_text("processor\t: 0\nvendor_id\t: AuthenticAMD\n"
                                          "model name\t: AMD EPYC 7702 64-Core Processor\n\nprocessor\t: 1\n"
                                          "model name\t: AMD EPYC 7702 64-Core Processor\n")
            (proc / "meminfo").write_text("MemTotal:       527424588 kB\nMemFree:        1 kB\n")
            self.assertEqual(provenance.linux_machine(proc), {"cpu": "AMD EPYC 7702 64-Core Processor", "memory_gb": 503.0})
            self.assertEqual(provenance.linux_machine(proc / "absent"), {})

    def test_slurm_job_from_the_environment(self):
        self.assertEqual(provenance.slurm({}), {})
        env = {"SLURM_JOB_ID": "3412", "SLURM_JOB_NODELIST": "cn605-26-l", "SLURMD_NODENAME": "cn605-26-l",
               "SLURM_CPUS_ON_NODE": "128"}
        self.assertEqual(provenance.slurm(env),
                         {"slurm_job_id": "3412", "slurm_nodelist": "cn605-26-l", "slurm_cpus_on_node": "128"})
        self.assertEqual(provenance.slurm({"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "cn1"}),
                         {"slurm_job_id": "1", "slurm_nodelist": "cn1"})

    def test_machine_records_the_host_and_on_linux_the_slurm_job(self):
        info = provenance.machine()
        self.assertEqual(info["hostname"], socket.gethostname())
        # Without the variables of a SLURM job the tests may themselves run in.
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM")}
        env.update({"SLURM_JOB_ID": "77", "SLURMD_NODENAME": "cn9"})
        with mock.patch.object(provenance.platform, "system", return_value="Linux"), \
                mock.patch.dict(os.environ, env, clear=True):
            info = provenance.machine()
        self.assertEqual((info["slurm_job_id"], info["slurm_nodelist"]), ("77", "cn9"))
        if hasattr(os, "sched_getaffinity"):
            self.assertGreater(info["cpus_usable"], 0)

    def test_machine_and_host_enter_no_job_key(self):
        j = Job(id="T/a", command=["true"], settings={"x": 1})
        key = provenance.job_key(j, Path("/tmp/none"), provenance.ChecksumCache(None))
        self.assertEqual(set(key), {"job", "command", "env", "settings", "tools", "seeds", "inputs"})
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "77"}), \
                mock.patch.object(socket, "gethostname", return_value="cn1"):
            self.assertEqual(provenance.job_key(j, Path("/tmp/none"), provenance.ChecksumCache(None)), key)


class RunnerSettings(unittest.TestCase):
    def test_sample_seconds_from_the_environment(self):
        self.assertEqual(runner.sample_seconds({}), 0.05)
        self.assertEqual(runner.sample_seconds({runner.SAMPLE_SECONDS_ENV: "1.5"}), 1.5)
        self.assertEqual(runner.sample_seconds({runner.SAMPLE_SECONDS_ENV: "0.01"}), 0.01)
        for bad in ("0", "-1", "inf", "nan", "1e-9", "0.009", "fast"):
            with self.assertRaisesRegex(ValueError, runner.SAMPLE_SECONDS_ENV, msg=bad):
                runner.sample_seconds({runner.SAMPLE_SECONDS_ENV: bad})
        runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with mock.patch.dict(os.environ, {runner.SAMPLE_SECONDS_ENV: "0.5"}):  # read when a runner starts
            d = runner.Daemon(Graph([], runs), runs, config.Budget(), runner.Logger(runs), {})
        self.assertEqual(d.sample_every, 0.5)

    def test_default_machine_lock_is_per_host_except_on_macos(self):
        base = Path.home() / ".cache" / "psmc-exp"
        self.assertEqual(runner.GUARD_HOSTS, sys.platform != "darwin")  # likewise the guards on other hosts
        self.assertEqual(runner.default_machine_lock("darwin", host="mac"), base / "machine.lock")
        self.assertEqual(runner.default_machine_lock("linux", host="cn605-26-l"), base / "machine-cn605-26-l.lock")
        with mock.patch.dict(os.environ, {runner.MACHINE_LOCK_ENV: "/node/local/m.lock"}):
            self.assertEqual(runner.MachineLock().path, Path("/node/local/m.lock"))


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class AcrossHosts(Base):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(runner, "GUARD_HOSTS", True))  # as on Linux, also on macOS

    def sleeper(self) -> subprocess.Popen:
        p = subprocess.Popen(["sleep", "30"], start_new_session=True)
        self.addCleanup(lambda: (p.kill(), p.wait()))
        return p

    def write_state(self, host, pid) -> None:
        state = {"running": {"T/a": {"pid": pid, "start": time.time()}}}
        if host is not None:
            state["host"] = host
        provenance.write_json_atomic(self.runs / "_runner" / "state.json", state)

    def test_kill_orphans_leaves_the_jobs_of_a_runner_on_another_host_alone(self):
        p = self.sleeper()
        logged = []
        self.write_state("another-host", p.pid)
        runner.kill_orphans(self.runs, logged.append)
        time.sleep(0.2)
        self.assertIsNone(p.poll())
        self.assertIn("another-host", logged[0])
        self.write_state(runner.this_host(), p.pid)
        runner.kill_orphans(self.runs, logged.append)
        self.assertEqual(p.wait(timeout=10), -15)

    def test_macos_does_not_guard_hosts(self):
        # Its host name follows the network: the laptop's own runner may have recorded another one.
        p = self.sleeper()
        self.write_state("before-the-network-changed", p.pid)
        self.hold_lock(424242, "before-the-network-changed")
        with mock.patch.object(runner, "GUARD_HOSTS", False):
            self.assertFalse(runner.on_another_host("before-the-network-changed"))
            with mock.patch.object(runner.os, "kill") as kill:
                self.assertEqual(runner.stop_daemon(self.runs, drain=True), 424242)
            kill.assert_called_once_with(424242, signal.SIGUSR1)
            runner.kill_orphans(self.runs, lambda msg: None)
        self.assertEqual(p.wait(timeout=10), -15)
        self.assertTrue(runner.on_another_host("before-the-network-changed"))
        self.assertFalse(runner.on_another_host(None))
        self.assertFalse(runner.on_another_host(runner.this_host()))

    def test_kill_orphans_survives_a_process_it_may_not_signal(self):
        p = self.sleeper()
        self.write_state(None, p.pid)  # a state from before hosts were recorded: this host's
        with mock.patch.object(runner.os, "killpg", side_effect=PermissionError):
            runner.kill_orphans(self.runs, lambda msg: None)
        self.assertIsNone(p.poll())

    def hold_lock(self, pid: int, host: str) -> runner.Lock:
        lock = runner.Lock(self.runs)
        self.assertTrue(lock.acquire())
        self.addCleanup(lock.release)
        (self.runs / "_runner" / "runner.pid").write_text(f"{pid}\n")
        (self.runs / "_runner" / "runner.host").write_text(f"{host}\n")
        return lock

    def test_stop_refuses_a_runner_on_another_host(self):
        self.hold_lock(os.getpid(), "another-host")
        with mock.patch.object(runner.os, "kill") as kill:
            with self.assertRaises(runner.ElsewhereError):
                runner.stop_daemon(self.runs)
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = main(["stop", "--runs-dir", str(self.runs)])
        kill.assert_not_called()
        self.assertEqual(rc, 1)
        self.assertIn("runs on another-host", err.getvalue())

    def test_stop_handles_a_runner_that_is_gone(self):
        lock = self.hold_lock(424242, runner.this_host())

        def gone(pid, sig):
            lock.release()  # it exited between the lookup and the signal
            raise ProcessLookupError

        with mock.patch.object(runner.os, "kill", side_effect=gone):
            self.assertIsNone(runner.stop_daemon(self.runs))
        self.hold_lock(424242, runner.this_host())
        with mock.patch.object(runner.os, "kill", side_effect=ProcessLookupError):
            with self.assertRaisesRegex(runner.ElsewhereError, "no process 424242"):
                runner.stop_daemon(self.runs)

    def test_attempts_record_the_host_and_slurm_job_outside_the_key(self):
        a = self.job("T/a", ["a.txt"])
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4242"}):
            self.run_daemon([a])
        rec = json.loads(a.record_path(self.runs).read_text())
        self.assertEqual((rec["run"]["host"], rec["run"]["slurm_job_id"]), (socket.gethostname(), "4242"))
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "4243"}), \
                mock.patch.object(socket, "gethostname", return_value="cn1"):
            self.run_daemon([a])  # another SLURM job on another host: still done
        self.assertEqual(self.runs_of("T/a"), 1)


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class Signals(Base):
    def test_jobs_killed_by_the_signal_that_stops_the_runner_are_interrupted_not_failed(self):
        # SLURM sends SIGTERM to the runner and its jobs at once: the jobs die before the runner terminates them.
        j = self.job("T/long", ["l.txt"], sleep=30)
        d = runner.Daemon(Graph([j], self.runs), self.runs, config.Budget(), runner.Logger(self.runs), {})

        def slurm():
            while not (d.running.get("T/long") and d.running["T/long"].proc):
                time.sleep(0.01)
            os.killpg(d.running["T/long"].proc.pid, signal.SIGTERM)
            d.stop_now.set()

        threading.Thread(target=slurm, daemon=True).start()
        with mock.patch.object(runner, "POLL_SECONDS", 1.0):  # the job's thread sees the death first
            rc = d.run()
        self.assertEqual((rc, d.state["T/long"]), (0, runner.PENDING))
        (attempt,) = runner.last_attempts(self.runs, ["T"]).values()
        self.assertEqual((attempt["status"], attempt["run"]["exit_status"]), ("interrupted", -signal.SIGTERM))

    def test_a_stop_or_drain_while_the_graph_is_made_is_not_lost(self):
        j = self.job("T/a", ["a.txt"])
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGHUP)}
        self.addCleanup(lambda: [signal.signal(s, h) for s, h in saved.items()])
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            def factory():
                os.kill(os.getpid(), sig)  # e.g. scancel while the tools are built
                return Graph([j], self.runs)

            with mock.patch.dict(os.environ, {runner.MACHINE_LOCK_ENV: str(self.tmp / "machine.lock")}):
                rc = runner.daemon_main(factory, self.runs, config.Budget(), {})
            self.assertEqual((rc, self.runs_of("T/a")), (0, 0), sig)
            self.assertFalse((self.runs / "_runner" / "runner.host").exists())
        state = json.loads((self.runs / "_runner" / "state.json").read_text())
        self.assertEqual(state["host"], runner.this_host())
        self.assertIn("hostname", state["machine"])


if __name__ == "__main__":
    unittest.main()
