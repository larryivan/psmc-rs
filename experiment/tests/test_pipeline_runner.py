"""Tests for experiment.pipeline.runner and provenance, with small stand-in jobs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, provenance, runner  # noqa: E402
from experiment.pipeline.runner import Graph, Job  # noqa: E402

HAS_PSUTIL = importlib.util.find_spec("psutil") is not None

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
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
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
        graph = Graph(jobs, self.runs)
        d = runner.Daemon(graph, self.runs, budget, runner.Logger(self.runs), {"budget": {"cores": budget.cores, "mem_gb": budget.mem_gb}})
        rc = d.run()
        return d, rc

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
        from unittest import mock

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
    def cli(self, *args) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONPATH=str(ROOT), PSMC_EXP_MACHINE_LOCK=str(self.tmp / "machine.lock"))
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


if __name__ == "__main__":
    unittest.main()
