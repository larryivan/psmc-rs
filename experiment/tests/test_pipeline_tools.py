"""Tests for experiment.pipeline.tools: identities and exact command lines."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, tools  # noqa: E402

FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5"})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0"})


def in_git() -> bool:
    if shutil.which("git") is None:
        return False
    return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True).returncode == 0


class Commands(unittest.TestCase):
    def test_c_psmc(self):
        cmd = tools.c_psmc_fit(FAKE_C, config.INFERENCE, Path("in.psmcfa"), Path("out.psmc"))
        self.assertEqual(cmd, ["/x/psmc", "-N25", "-t15", "-r5", "-p", "4+25*2+4+6", "-o", "out.psmc", "in.psmcfa"])

    def test_psmc_rs(self):
        cmd = tools.rs02_fit(FAKE_RS, config.INFERENCE, config.SIMULATION_SCALING, Path("in"), Path("o.json"),
                             threads=3, psmc_out=Path("o.psmc"), extra=["--qc-json", "q.json"])
        self.assertEqual(cmd[:4], ["/x/psmc-rs", "in", "o.json", "25"])
        opts = dict(zip(cmd[4::2], cmd[5::2]))
        self.assertEqual(opts["--pattern"], "4+25*2+4+6")
        self.assertEqual(opts["--t-max"], "15")
        self.assertEqual(opts["--tr-ratio"], "5")
        self.assertEqual(opts["--smooth-lambda"], "0")
        self.assertEqual(opts["--mstep-iters"], "100")
        self.assertEqual(opts["--bin-size"], "100")
        self.assertEqual(opts["--mu"], "2.5e-08")
        self.assertEqual(opts["--gen-years"], "25")
        self.assertEqual(opts["--threads"], "3")
        self.assertIn("--no-progress", cmd)
        self.assertEqual(cmd[-4:], ["--psmc-out", "o.psmc", "--qc-json", "q.json"])

    def test_qc(self):
        self.assertEqual(tools.rs02_qc(FAKE_RS, Path("a.vcf.gz"), Path("q.json")), ["/x/psmc-rs", "qc", "a.vcf.gz", "--json", "q.json"])

    def test_fmt(self):
        self.assertEqual([tools.fmt(x) for x in (15.0, 5.0, 2.5e-8, 0.0, 1e-3)], ["15", "5", "2.5e-08", "0", "0.001"])


@unittest.skipUnless(in_git(), "not a git checkout")
class Identities(unittest.TestCase):
    def test_rs02_source_state(self):
        state = tools.rs02_source_state()
        self.assertRegex(state["source_commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(state["source_dirty"] == "" or re.match(r"^[0-9a-f]{64}$", state["source_dirty"]))

    def test_psmc_rs_outside_a_checkout_is_identified_by_its_binary(self):
        binary = ROOT / "target" / "release" / "psmc-rs"
        if not binary.exists():
            self.skipTest("target/release/psmc-rs not built")
        with mock.patch.object(tools, "rs02_source_state", side_effect=tools.ToolError("not a repository")):
            t = tools.psmc_rs_02(build=False)
        self.assertEqual(t.identity["binary_sha256"], tools.sha256_file(binary))
        self.assertIn("unknown", t.identity["source_commit"])

    def test_missing_tools_are_reported(self):
        with self.assertRaises(tools.ToolError):
            tools.c_psmc(Path("/nonexistent/tools"))
        with self.assertRaises(tools.ToolError):
            tools.utility("fq2psmcfa", Path("/nonexistent/tools"))


if __name__ == "__main__":
    unittest.main()
