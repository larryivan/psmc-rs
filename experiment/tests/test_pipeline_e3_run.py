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
        parsed = e3_run.build_parser().parse_args(["measure", "--out", "x.json", *args])
        self.assertEqual(parsed.max_wait, config.TimingProtocol().max_wait_seconds)


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

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = e3_run.main(["measure", "--out", str(self.out), "--settle", "0", "--max-wait", "0",
                              "--quiet-window", "0.05", "--", sys.executable, "-c", "pass"])
        self.assertEqual(rc, 0)
        self.assertTrue(self.out.exists())
        with self.assertRaises(SystemExit):
            e3_run.main(["measure", "--out", str(self.out)])


if __name__ == "__main__":
    unittest.main()
