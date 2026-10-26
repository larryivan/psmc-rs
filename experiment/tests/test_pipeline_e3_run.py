"""Tests for experiment.pipeline.experiments.e3_run: E3's genomes by size and the timing of one run."""

from __future__ import annotations

import collections
import contextlib
import functools
import io
import json
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import has_modules, option  # noqa: E402

from experiment.pipeline import config, simulate  # noqa: E402
from experiment.pipeline.experiments import e3_run  # noqa: E402

HAS_PSUTIL = has_modules("psutil")
NO_WAIT = replace(config.QUICK_TIMING, quiet_window_seconds=0.05, rss_sample_seconds=0.01)

# A child that allocates and touches MB megabytes, optionally in a grandchild,
# sleeps, and exits with STATUS.
CHILD = textwrap.dedent(
    """
    import subprocess, sys, time
    mb, status, nested = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "nested"
    if nested:
        sys.exit(subprocess.call([sys.executable, __file__, str(mb), str(status), "flat"]))
    buf = bytearray(mb * 2**20)
    for i in range(0, len(buf), 4096):
        buf[i] = 1
    time.sleep(0.3)
    sys.exit(status)
    """
)


class Genomes(unittest.TestCase):
    def test_sizes_are_whole_chromosomes(self):
        cfg = config.PRODUCTION.e3
        self.assertEqual([e3_run.genome_for(cfg, s).n_chromosomes for s in cfg.genome_sizes_mb], [1, 3, 10, 30])
        self.assertEqual(e3_run.genome_for(cfg, 300).length, 300_000_000)
        with self.assertRaises(ValueError):
            e3_run.genome_for(cfg, 150)
        self.assertEqual(e3_run.genome_for(config.TINY.e3, 1).n_chromosomes, 2)

    def test_measure_args_spell_out_the_protocol(self):
        args = e3_run.measure_args(config.TimingProtocol())
        for opt, value in (("--settle", "10"), ("--quiet-cpu", "10"), ("--max-wait", "180"), ("--quiet-max-cores", "1"),
                           ("--max-other-cores", "0.5")):
            self.assertEqual(option(args, opt), value, opt)
        parsed = e3_run.build_parser().parse_args(["measure", "--out", "x.json", *args])
        self.assertEqual(parsed.max_wait, config.TimingProtocol().max_wait_seconds)
        self.assertEqual((parsed.quiet_max_cores, parsed.max_other_cores), (1.0, 0.5))

    def test_limits_are_shares_of_a_laptop_and_core_counts_of_a_cluster_node(self):
        p = config.TimingProtocol()
        for cores, quiet, other in ((8, 0.8, 0.4), (128, 1.0, 0.5), (192, 1.0, 0.5)):
            with self.subTest(cores=cores):
                self.assertAlmostEqual(min(p.quiet_cpu_percent / 100 * cores, p.quiet_max_cores), quiet)
                self.assertAlmostEqual(min(p.max_other_cpu_share * cores, p.max_other_cores), other)
        res = {"during": {"other_method": "machine", "other_cpu_share": 0.004, "other_cores": 0.5},
               "wall_seconds": 10.0, "clock_span_seconds": 10.0}
        self.assertIsNone(e3_run.unclean(res, p))
        res["during"]["other_cores"] = 0.77  # 0.6% of 128 cores
        self.assertIn("0.77 cores", e3_run.unclean(res, p))

    def test_linux_machine(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cpu = root / "sys/devices/system/cpu"
            for i in range(2):
                (cpu / f"cpu{i}/cpufreq").mkdir(parents=True)
                (cpu / f"cpu{i}/cpufreq/scaling_governor").write_text("performance\n")
            (cpu / "cpu0/cpufreq/scaling_driver").write_text("acpi-cpufreq\n")
            (cpu / "cpufreq").mkdir()
            (cpu / "cpufreq/boost").write_text("1\n")
            (root / "proc").mkdir()
            (root / "proc/cpuinfo").write_text("processor\t: 0\nmodel name\t: AMD EPYC 7702 64-Core Processor\n")
            (root / "proc/meminfo").write_text("MemTotal:       527622968 kB\n")
            env = {"SLURM_JOB_ID": "12345678", "SLURM_JOB_NODELIST": "cn509-02-l", "HOME": "/h"}
            info = e3_run.linux_machine(root, env)
            self.assertEqual(info["cpu"], "AMD EPYC 7702 64-Core Processor")
            self.assertEqual((info["cpu_governor"], info["cpu_scaling_driver"], info["cpu_boost"]),
                             ("performance", "acpi-cpufreq", "1"))
            self.assertAlmostEqual(info["memory_gb"], 503.2, places=1)
            self.assertEqual((info["slurm_job_id"], info["slurm_job_nodelist"]), ("12345678", "cn509-02-l"))
            self.assertNotIn("slurm_cpus_on_node", info)
            self.assertTrue(info["hostname"])
            # Intel: no boost switch, but intel_pstate's no_turbo; governors that differ are all named
            (cpu / "cpufreq/boost").unlink()
            (cpu / "intel_pstate").mkdir()
            (cpu / "intel_pstate/no_turbo").write_text("1\n")
            (cpu / "cpu1/cpufreq/scaling_governor").write_text("powersave\n")
            info = e3_run.linux_machine(root, {})
            self.assertEqual((info["cpu_boost"], info["cpu_governor"]), ("0", "performance,powersave"))
        with tempfile.TemporaryDirectory() as d:  # nothing readable: nothing claimed
            info = e3_run.linux_machine(Path(d), {})
            self.assertEqual(set(info), {"hostname"})


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class Quiet(unittest.TestCase):
    """The quiet check of a machine of many cores and of a laptop, with psutil's counters made up."""

    def quiet(self, cpus: int, busy_cores: float) -> dict:
        import psutil

        times = collections.namedtuple("scputimes", "user system idle")
        window = 2.0
        before, after = times(0.0, 0.0, 0.0), times(busy_cores * window, 0.0, (cpus - busy_cores) * window)
        protocol = replace(config.TimingProtocol(), settle_seconds=0.0, max_wait_seconds=0.0)  # the default limits
        with mock.patch.object(psutil, "cpu_times", side_effect=[before, after]), \
                mock.patch.object(psutil, "cpu_count", return_value=cpus), \
                mock.patch.object(psutil, "process_iter", return_value=[]), \
                mock.patch.object(e3_run.time, "sleep"):
            return e3_run.wait_until_quiet(protocol, log=lambda m: None)

    def test_a_node_of_many_cores_is_quiet_by_the_cores_and_a_laptop_by_the_share(self):
        seen = self.quiet(192, 1.5)  # 0.8% of the cores: within the share, not the 1 core
        self.assertAlmostEqual(seen["busy_cores"], 1.5)
        self.assertLess(seen["busy_percent"], 1.0)
        self.assertFalse(seen["quiet"])
        self.assertTrue(self.quiet(192, 0.9)["quiet"])
        self.assertTrue(self.quiet(8, 0.5)["quiet"])  # 6.25% of a laptop
        seen = self.quiet(8, 0.9)  # 11%: within the core, not the share
        self.assertAlmostEqual(seen["busy_cores"], 0.9)
        self.assertFalse(seen["quiet"])


CPU = collections.namedtuple("scputimes", "user nice system idle iowait irq softirq steal")
PROC_TIMES = collections.namedtuple("pcputimes", "user system children_user children_system")
TOOL, LEFTOVER, WRAPPER = 4001, 4002, 4000  # the tool, a process it started that outlives it, this wrapper


def proc(pid: int, ppid: int, cpu: float, created: float = 1.0, name: str = "p", reaped: float = 0.0):
    """A process as psutil.process_iter(attrs) gives it; ``reaped``: its waited-for children's CPU."""
    return types.SimpleNamespace(pid=pid, info={"ppid": ppid, "name": name, "create_time": created,
                                                "cpu_times": PROC_TIMES(cpu * 0.75, cpu * 0.25, reaped / 2, reaped / 2)})


LINUX_METHOD = functools.partial(e3_run.process_method, platform="linux")


# An idle 128-core node as Ibex's: init, kthreadd and a kworker, the WekaFS
# client and the timing wrapper, before and after a run of 100 s (the
# wrapper has reaped the tool, of 3000 s, by then).
IDLE_BEFORE = [proc(1, 0, 100.0, name="systemd", reaped=40.0), proc(2, 0, 0.0, name="kthreadd"),
               proc(50, 2, 10.0, name="kworker/0:1"), proc(900, 1, 500.0, name="wekanode"),
               proc(WRAPPER, 3999, 2.0, name="python")]
IDLE_AFTER = [proc(1, 0, 100.0, name="systemd", reaped=40.0), proc(2, 0, 0.0, name="kthreadd"),
              proc(50, 2, 25.0, name="kworker/0:1"), proc(900, 1, 520.0, name="wekanode"),
              proc(WRAPPER, 3999, 3.0, name="python", reaped=3000.0)]


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class OtherProcesses(unittest.TestCase):
    """What counts as other work during a run, with psutil's counters made up (Linux, 128 cores, 100 s)."""

    CPU0 = CPU(user=0.0, nice=0.0, system=0.0, idle=0.0, iowait=0.0, irq=0.0, softirq=0.0, steal=0.0)

    def during(self, before, after, extra_user: float = 0.0, extra_system: float = 0.0, hidden: bool = False,
               steal: float = 0.0, platform: str = "linux"):
        """A 30-thread run (3000 s of CPU) and what the machine-wide counters add to it: wekanode 20 s,
        the kworker 15 s, the wrapper 1 s, 20 s of interrupts, plus ``extra_*`` and ``steal``."""
        import psutil

        user, system, irq, softirq = 2950.0 + 15.0 + extra_user, 50.0 + 5.0 + 15.0 + 1.0 + extra_system, 5.0, 15.0
        cpu1 = CPU(user=user, nice=0.0, system=system, idle=128 * 100.0 - user - system - irq - softirq - steal,
                   iowait=0.0, irq=irq, softirq=softirq, steal=steal)
        with mock.patch.object(psutil, "process_iter", side_effect=[before, after]):
            procs0, unread0 = e3_run.process_times("linux")
            procs1, unread1 = e3_run.process_times("linux")
        with mock.patch.object(e3_run, "processes_hidden", return_value=hidden):
            method = e3_run.process_method(unread0, unread1, platform=platform)
        during = e3_run.during_run(self.CPU0, cpu1, procs0, procs1, 100.0, 3000.0, exclude={TOOL, LEFTOVER, WRAPPER},
                                   method=method, unreadable=max(unread0, unread1))
        return {"during": during, "wall_seconds": 100.0, "clock_span_seconds": 100.0}

    def test_a_run_on_many_cores_is_clean_though_the_kernel_works_for_it(self):
        res = self.during(IDLE_BEFORE, IDLE_AFTER + [proc(LEFTOVER, TOOL, 2.0, created=10.0, name="helper")],
                          extra_system=2.0)
        d = res["during"]
        self.assertEqual(d["other_method"], "processes")
        self.assertAlmostEqual(d["other_process_cores"], 0.2)  # wekanode only: not the tool, not the wrapper
        self.assertAlmostEqual(d["other_process_cpu_share"], 20.0 / 12800, places=4)
        self.assertAlmostEqual(d["kernel_thread_cores"], 0.15)
        self.assertAlmostEqual(d["interrupt_cores"], 0.2)
        self.assertEqual(d["steal_cores"], 0.0)
        # the wrapper and the leftover, their own CPU: not other work (the tool the wrapper reaped is the tool's)
        self.assertAlmostEqual(d["excluded_cores"], 0.03)
        self.assertAlmostEqual(d["unaccounted_cores"], 0.0)
        self.assertEqual([b["name"] for b in d["busiest_others"]], ["wekanode", "systemd"])
        protocol = config.TimingProtocol()
        # the machine-wide measure is recorded, and would have rejected the run
        self.assertAlmostEqual(d["other_cores"], 0.58)
        self.assertGreater(e3_run.machine_other_cores(res), protocol.max_other_cores)
        self.assertIsNone(e3_run.unclean(res, protocol))
        self.assertAlmostEqual(e3_run.other_cores(res), 0.2)
        # a kernel thread busy for a whole core is recorded, not limited
        busy_kworker = [proc(50, 2, 125.0, name="kworker/0:1") if p.pid == 50 else p for p in IDLE_AFTER]
        res = self.during(IDLE_BEFORE, busy_kworker, extra_system=100.0)
        self.assertAlmostEqual(res["during"]["kernel_thread_cores"], 1.15)
        self.assertIsNone(e3_run.unclean(res, protocol))

    def test_another_process_using_more_than_half_a_core_makes_the_run_unclean(self):
        res = self.during(IDLE_BEFORE + [proc(950, 1, 0.0, name="job")],
                          IDLE_AFTER + [proc(950, 1, 60.0, name="job")], extra_user=60.0)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.8)
        self.assertEqual(res["during"]["busiest_others"][0], {"pid": 950, "name": "job", "cores": 0.6})
        self.assertIn("other user-space processes and steal time kept 0.80 cores",
                      e3_run.unclean(res, config.TimingProtocol()))

    def test_a_process_started_during_the_run_counts_whole(self):
        res = self.during(IDLE_BEFORE, IDLE_AFTER + [proc(960, 1, 40.0, created=50.0)], extra_user=40.0)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.6)
        self.assertIsNotNone(e3_run.unclean(res, config.TimingProtocol()))
        # a pid used again is another process: matched by pid and creation time
        res = self.during(IDLE_BEFORE + [proc(970, 1, 300.0, created=1.0)],
                          IDLE_AFTER + [proc(970, 1, 5.0, created=60.0)], extra_user=5.0)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.25)
        self.assertEqual(res["during"]["other_processes_exited"], 1)

    def test_processes_that_ended_during_the_run_are_counted(self):
        res = self.during(IDLE_BEFORE + [proc(980, 1, 7.0), proc(51, 2, 1.0, name="kworker/1:2")], IDLE_AFTER)
        self.assertEqual(res["during"]["other_processes_exited"], 1)  # kernel threads not among them
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.2)
        protocol = config.TimingProtocol()
        # a job of 20 cores for 95 s that ended before the tool: its CPU is in its parent's reaped children's
        shell = proc(990, 1, 1.0, name="bash")
        job_before = [shell, proc(991, 990, 0.0, name="job")]
        res = self.during(IDLE_BEFORE + job_before, IDLE_AFTER + [proc(990, 1, 1.0, name="bash", reaped=1900.0)],
                          extra_user=1900.0)
        d = res["during"]
        self.assertEqual(d["other_processes_exited"], 1)
        self.assertAlmostEqual(d["other_process_cores"], 19.2)
        self.assertAlmostEqual(d["unaccounted_cores"], 0.0)
        self.assertEqual(d["busiest_others"][0], {"pid": 990, "name": "bash", "cores": 19.0})
        self.assertIn("used 15.0% of all cores", e3_run.unclean(res, protocol))
        # short-lived processes (a make -j's compilers), never seen: counted through make, which waited for them
        make = proc(995, 1, 0.5, name="make")
        res = self.during(IDLE_BEFORE + [make], IDLE_AFTER + [proc(995, 1, 0.6, name="make", reaped=400.0)],
                          extra_user=400.1)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 4.201)
        self.assertEqual(res["during"]["other_processes_exited"], 0)
        self.assertIsNotNone(e3_run.unclean(res, protocol))
        # one orphaned and reaped by init counts through init; a process nothing waits for is only unaccounted
        reaped_by_init = [proc(1, 0, 100.0, name="systemd", reaped=70.0)] + IDLE_AFTER[1:]
        res = self.during(IDLE_BEFORE, reaped_by_init, extra_user=30.0)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.5)
        res = self.during(IDLE_BEFORE, IDLE_AFTER, extra_user=30.0)
        self.assertAlmostEqual(res["during"]["other_process_cores"], 0.2)
        self.assertAlmostEqual(res["during"]["unaccounted_cores"], 0.3)

    def test_steal_time_is_other_work(self):
        res = self.during(IDLE_BEFORE, IDLE_AFTER, steal=40.0)
        d = res["during"]
        self.assertEqual((d["steal_cores"], d["steal_cpu_share"]), (0.4, round(40.0 / 12800, 4)))
        self.assertAlmostEqual(e3_run.other_cores(res), 0.6)
        self.assertAlmostEqual(e3_run.other_share(res), (20.0 + 40.0) / 12800, places=4)
        self.assertIn("kept 0.60 cores", e3_run.unclean(res, config.TimingProtocol()))
        self.assertIsNone(e3_run.unclean(self.during(IDLE_BEFORE, IDLE_AFTER, steal=20.0), config.TimingProtocol()))

    def test_unreadable_processes_fall_back_to_the_whole_machine(self):
        denied = proc(990, 1, 0.0, name="other user's")
        denied.info["cpu_times"] = e3_run._DENIED
        res = self.during(IDLE_BEFORE, IDLE_AFTER + [denied])
        d = res["during"]
        self.assertEqual((d["other_method"], d["processes_unreadable"]), ("machine", 1))
        self.assertIsNone(d["other_process_cores"])  # a partial sum would under-count
        self.assertIsNone(d["kernel_thread_cores"])
        self.assertAlmostEqual(e3_run.other_cores(res), 0.56)
        self.assertIn("the machine's busy CPU but the tool's", e3_run.unclean(res, config.TimingProtocol()))
        # a /proc that hides other users' processes: the same
        res = self.during(IDLE_BEFORE, IDLE_AFTER, hidden=True)
        self.assertEqual(res["during"]["other_method"], "machine")
        self.assertIsNotNone(e3_run.unclean(res, config.TimingProtocol()))
        # and every process read but not on Linux (macOS as root), which gives no reaped children's CPU
        res = self.during(IDLE_BEFORE, IDLE_AFTER, platform="darwin")
        self.assertEqual(res["during"]["other_method"], "machine")
        self.assertIsNone(res["during"]["other_process_cores"])

    def test_proc_mounted_hidepid_hides_processes(self):
        with tempfile.TemporaryDirectory() as d:
            mounts = Path(d) / "mounts"
            with mock.patch.object(e3_run.os, "geteuid", return_value=1000, create=True):
                for opts, hidden in (("rw,nosuid,nodev,noexec,relatime", False), ("rw,relatime,hidepid=2", True),
                                     ("rw,hidepid=invisible,gid=10", True), ("rw,hidepid=0", False)):
                    mounts.write_text(f"sysfs /sys sysfs rw 0 0\nproc /proc proc {opts} 0 0\n")
                    self.assertEqual(e3_run.processes_hidden(mounts, "linux"), hidden, opts)
                self.assertFalse(e3_run.processes_hidden(mounts, "darwin"))
            with mock.patch.object(e3_run.os, "geteuid", return_value=0, create=True):
                self.assertFalse(e3_run.processes_hidden(mounts, "linux"))  # root sees every process

    def test_kernel_threads(self):
        self.assertTrue(e3_run.is_kernel_thread(2, 0, "linux"))
        self.assertTrue(e3_run.is_kernel_thread(50, 2, "linux"))
        self.assertFalse(e3_run.is_kernel_thread(1, 0, "linux"))
        self.assertTrue(e3_run.is_kernel_thread(0, 0, "darwin"))
        self.assertFalse(e3_run.is_kernel_thread(50, 2, "darwin"))

    def test_the_quiet_check_records_what_kept_the_machine_busy(self):
        import psutil

        window = 2.0  # wekanode 0.2 core, the kworker 0.15, the wrapper 0.05, interrupts 0.25
        cpu1 = CPU(user=0.4, nice=0.0, system=0.4, idle=128 * window - 1.3, iowait=0.0, irq=0.1, softirq=0.4, steal=0.0)
        before = [proc(50, 2, 10.0, name="kworker/0:1"), proc(900, 1, 500.0, name="wekanode"), proc(WRAPPER, 1, 2.0)]
        after = [proc(50, 2, 10.3, name="kworker/0:1"), proc(900, 1, 500.4, name="wekanode"), proc(WRAPPER, 1, 2.1)]
        linux_times = functools.partial(e3_run.process_times, "linux")
        with mock.patch.object(psutil, "cpu_times", side_effect=[self.CPU0, cpu1]), \
                mock.patch.object(psutil, "cpu_count", return_value=128), \
                mock.patch.object(psutil, "process_iter", side_effect=[before, after]), \
                mock.patch.object(e3_run, "process_times", linux_times), \
                mock.patch.object(e3_run, "process_method", LINUX_METHOD), \
                mock.patch.object(e3_run, "processes_hidden", return_value=False), \
                mock.patch.object(e3_run.os, "getpid", return_value=WRAPPER), \
                mock.patch.object(e3_run.time, "sleep"), \
                mock.patch.object(e3_run.time, "perf_counter", side_effect=[0.0, window, window]):
            act = e3_run.machine_activity(window)
        self.assertAlmostEqual(act["busy_cores"], 0.65)  # the whole machine: what the quiet check limits
        self.assertAlmostEqual(act["process_cores"], 0.2)  # this process (the wrapper) left out
        self.assertAlmostEqual(act["kernel_thread_cores"], 0.15)
        self.assertAlmostEqual(act["interrupt_cores"], 0.25)
        self.assertEqual((act["steal_cores"], act["processes_unreadable"]), (0.0, 0))
        self.assertEqual(act["busiest"][0], {"pid": 900, "name": "wekanode", "cpu_percent": 20.0})


@unittest.skipUnless(has_modules("msprime"), "msprime not installed")
class SimulateSize(unittest.TestCase):
    def test_tiny_genome_uses_the_e3_seeds(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rec = e3_run.simulate_size(config.TINY, 1, d / "sim", d / "truth.json", log=lambda m: None)
            text = (d / "sim" / simulate.PSMCFA_NAME).read_text()
            self.assertEqual(text.count(">"), 2)
            self.assertFalse((d / "sim" / simulate.VCF_NAME).exists())
            self.assertEqual(rec["experiment"], "E3")
            self.assertEqual(rec["chromosomes"][1]["seeds"]["ancestry"], config.seed("E3", "constant", 1, "ancestry/chr2"))
            self.assertEqual(json.loads((d / "truth.json").read_text())["totals"]["length"], 1_000_000)


@unittest.skipUnless(HAS_PSUTIL, "psutil not installed")
class Measure(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.child = self.tmp / "child.py"
        self.child.write_text(CHILD)
        self.out = self.tmp / "runs" / "timing.json"

    def measure(self, mb: int, status: int = 0, nested: bool = False, protocol=NO_WAIT) -> int:
        cmd = [sys.executable, str(self.child), str(mb), str(status), "nested" if nested else "flat"]
        return e3_run.measure(cmd, self.out, protocol, runs_dir=self.tmp / "runs", log=lambda m: None)

    def test_records_time_memory_and_the_machine(self):
        self.assertEqual(self.measure(64), 0)
        rec = json.loads(self.out.read_text())
        self.assertGreaterEqual(rec["wall_seconds"], 0.3)
        self.assertGreaterEqual(rec["peak_rss_mb"], 64)
        self.assertGreaterEqual(rec["maxrss_mb"], 64)
        self.assertGreater(rec["rss_samples"], 0)
        self.assertGreater(rec["cpu_seconds"], 0)
        self.assertEqual(rec["exit_status"], 0)
        self.assertEqual(len(rec["loadavg_before"]), 3)
        self.assertIn(rec["machine_before"]["quiet"], (True, False))
        self.assertLessEqual(rec["machine_before"]["busy_percent"], 100)
        self.assertGreaterEqual(rec["during"]["other_cpu_share"], 0)
        self.assertLessEqual(rec["during"]["other_cpu_share"], 1)
        self.assertEqual(rec["protocol"]["settle_seconds"], 0.0)
        self.assertIn("power", rec["machine_before"])
        self.assertIn("removed", rec["environment"])
        self.assertGreaterEqual(rec["clock_span_seconds"], rec["wall_seconds"] - 0.01)
        self.assertIn("cpu_count", rec["machine"])
        if sys.platform == "darwin" and rec["machine"].get("machine") == "arm64":
            self.assertGreater(rec["machine"]["performance_cores"], 0)
        self.assertEqual(rec["command"][1], str(self.child))  # outside the runs directory: as it was

    def test_the_process_tree_is_measured(self):
        self.assertEqual(self.measure(96, nested=True), 0)
        rec = json.loads(self.out.read_text())
        self.assertGreaterEqual(rec["peak_rss_tree_sampled_mb"], 96)

    def test_every_process_is_read_before_and_after_the_run(self):
        import psutil

        me = os.getpid()
        before = [proc(900, 1, 500.0, name="wekanode"), proc(me, 1, 2.0, name="python")]
        after = [proc(900, 1, 500.5, name="wekanode"), proc(me, 1, 50.0, name="python")]
        with mock.patch.object(psutil, "process_iter", side_effect=[before, after]), \
                mock.patch.object(e3_run, "process_method", LINUX_METHOD), \
                mock.patch.object(e3_run, "processes_hidden", return_value=False):
            res = e3_run.run_measured([sys.executable, "-c", "pass"], 0.01)
        d = res["during"]
        self.assertEqual(d["other_method"], "processes")
        self.assertAlmostEqual(d["other_process_cpu_seconds"], 0.5)  # this wrapper's 48 s are not other work
        self.assertAlmostEqual(d["other_process_cores"] * res["wall_seconds"], 0.5, places=2)
        self.assertEqual([b["name"] for b in d["busiest_others"]], ["wekanode"])
        self.assertGreaterEqual(d["other_cores"], 0.0)  # the machine-wide measure, beside it

    def test_a_failed_run_writes_nothing_and_passes_its_status_on(self):
        self.assertEqual(self.measure(1, status=3), 3)
        self.assertFalse(self.out.exists())

    def test_more_threads_than_usable_cpus_are_not_timed(self):
        cmd = [sys.executable, str(self.child), "1", "0", "flat"]
        with mock.patch.object(e3_run, "usable_cpus", return_value=16):
            rc = e3_run.measure(cmd, self.out, NO_WAIT, log=lambda m: None, threads=30)
            self.assertEqual(rc, e3_run.EX_UNAVAILABLE)  # a failure, not "try later": it never fits here
            self.assertFalse(self.out.exists())
            self.assertEqual(e3_run.measure(cmd, self.out, NO_WAIT, log=lambda m: None, threads=16), 0)
        rec = json.loads(self.out.read_text())
        self.assertEqual((rec["threads"], rec["cpus_usable"]), (16, 16))
        self.assertGreaterEqual(e3_run.usable_cpus(), 1)

    def test_paths_in_the_runs_directory_are_portable(self):
        target = self.tmp / "runs" / "child.py"
        target.parent.mkdir(parents=True)
        target.write_text(CHILD)
        e3_run.measure([sys.executable, str(target), "1", "0", "flat"], self.out, NO_WAIT,
                       runs_dir=self.tmp / "runs", log=lambda m: None)
        self.assertEqual(json.loads(self.out.read_text())["command"][1], "<runs>/child.py")

    def test_waits_for_a_quiet_machine_at_most_max_wait(self):
        never = replace(NO_WAIT, quiet_cpu_percent=-1.0, max_wait_seconds=0.2)
        t0 = time.monotonic()
        seen = e3_run.wait_until_quiet(never, log=lambda m: None)
        self.assertGreaterEqual(time.monotonic() - t0, 0.2)
        self.assertFalse(seen["quiet"])
        always = replace(NO_WAIT, quiet_cpu_percent=100.0, max_wait_seconds=60.0)
        seen = e3_run.wait_until_quiet(always, log=lambda m: None)
        self.assertTrue(seen["quiet"])
        self.assertLess(seen["waited_seconds"], 5.0)
        self.assertGreaterEqual(seen["busy_cores"], 0.0)
        # a share that any machine meets, but no cores to spare
        no_cores = replace(NO_WAIT, quiet_cpu_percent=100.0, quiet_max_cores=-1.0, max_wait_seconds=0.0)
        seen = e3_run.wait_until_quiet(no_cores, log=lambda m: None)
        self.assertFalse(seen["quiet"])
        self.assertEqual(seen["threshold_cores"], -1.0)

    def test_an_unclean_run_writes_nothing_and_asks_to_be_run_later(self):
        busy = replace(NO_WAIT, quiet_cpu_percent=-1.0, max_wait_seconds=0.0)
        self.assertEqual(self.measure(1, protocol=busy), e3_run.EX_TEMPFAIL)  # never quiet: not run
        self.assertFalse(self.out.exists())
        crowded = replace(NO_WAIT, max_other_cpu_share=-1.0)
        self.assertEqual(self.measure(1, protocol=crowded), e3_run.EX_TEMPFAIL)  # contaminated: not kept
        self.assertFalse(self.out.exists())
        crowded = replace(NO_WAIT, max_other_cores=-1.0)
        self.assertEqual(self.measure(1, protocol=crowded), e3_run.EX_TEMPFAIL)  # by the cores, too
        self.assertFalse(self.out.exists())
        res = {"during": {"other_method": "machine", "other_cpu_share": 0.01, "other_cores": 0.08},
               "wall_seconds": 10.0, "clock_span_seconds": 70.0}
        protocol = config.TimingProtocol()
        self.assertIn("slept", e3_run.unclean(res, protocol))
        self.assertIsNone(e3_run.unclean({**res, "clock_span_seconds": 10.5}, protocol))

    def test_the_tool_runs_without_psmc_variables(self):
        env, removed = e3_run.clean_environment({"PSMC_ALPHA_CACHE_MB": "64", "RUSTFLAGS": "-C x", "HOME": "/h",
                                                 "RAYON_NUM_THREADS": "8"})
        self.assertEqual(removed, ["PSMC_ALPHA_CACHE_MB", "RUSTFLAGS"])
        self.assertEqual(env, {"HOME": "/h", "RAYON_NUM_THREADS": "8"})

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = e3_run.main(["measure", "--out", str(self.out), "--settle", "0", "--max-wait", "0",
                              "--quiet-cpu", "100", "--quiet-max-cores", "1e9", "--max-other-cpu", "1",
                              "--max-other-cores", "1e9", "--no-require-ac", "--threads", "1",
                              "--quiet-window", "0.05", "--", sys.executable, "-c", "pass"])
        self.assertEqual(rc, 0)
        rec = json.loads(self.out.read_text())
        self.assertEqual(rec["threads"], 1)
        self.assertEqual((rec["protocol"]["quiet_max_cores"], rec["protocol"]["max_other_cores"]), (1e9, 1e9))
        self.assertGreaterEqual(rec["during"]["other_cores"], 0.0)
        with self.assertRaises(SystemExit):
            e3_run.main(["measure", "--out", str(self.out)])


if __name__ == "__main__":
    unittest.main()
