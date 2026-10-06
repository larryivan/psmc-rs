"""Tests for experiment.pipeline.experiments.e3_run: E3's genomes by size and the timing of one run."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import textwrap
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, simulate  # noqa: E402
from experiment.pipeline.experiments import e3_run  # noqa: E402

HAS_PSUTIL = importlib.util.find_spec("psutil") is not None
HAS_MSPRIME = importlib.util.find_spec("msprime") is not None
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
        self.assertEqual(args[args.index("--settle") + 1], "10")
        self.assertEqual(args[args.index("--quiet-cpu") + 1], "10")
        self.assertEqual(args[args.index("--max-wait") + 1], "180")
        self.assertEqual(args[args.index("--quiet-max-cores") + 1], "1")
        self.assertEqual(args[args.index("--max-other-cores") + 1], "0.5")
        parsed = e3_run.build_parser().parse_args(["measure", "--out", "x.json", *args])
        self.assertEqual(parsed.max_wait, config.TimingProtocol().max_wait_seconds)
        self.assertEqual((parsed.quiet_max_cores, parsed.max_other_cores), (1.0, 0.5))

    def test_limits_are_shares_of_a_laptop_and_core_counts_of_a_cluster_node(self):
        p = config.TimingProtocol()
        for cores, quiet, other in ((8, 0.8, 0.4), (128, 1.0, 0.5), (192, 1.0, 0.5)):
            with self.subTest(cores=cores):
                self.assertAlmostEqual(min(p.quiet_cpu_percent / 100 * cores, p.quiet_max_cores), quiet)
                self.assertAlmostEqual(min(p.max_other_cpu_share * cores, p.max_other_cores), other)
        res = {"during": {"other_cpu_share": 0.004, "other_cores": 0.5}, "wall_seconds": 10.0, "clock_span_seconds": 10.0}
        self.assertIsNone(e3_run.unclean(res, p))
        res["during"]["other_cores"] = 0.77  # 0.6% of 128 cores
        self.assertIn("0.77 cores", e3_run.unclean(res, p))
        del res["during"]["other_cores"]  # an older record: from the CPU seconds of the others
        res["during"]["other_cpu_seconds"] = 7.7
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
        import collections

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


@unittest.skipUnless(HAS_MSPRIME, "msprime not installed")
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
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
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
        res = {"during": {"other_cpu_share": 0.01}, "wall_seconds": 10.0, "clock_span_seconds": 70.0}
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
