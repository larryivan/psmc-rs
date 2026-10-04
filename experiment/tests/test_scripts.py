"""Tests for the experiment scripts (stdlib unittest; no package installs needed).

Run from anywhere with:

    python3 -m unittest discover -s experiment/tests -v

Tests that need optional packages (msprime, numpy, ...) skip when those are absent.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiment" / "scripts"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_script(script: Path, *args: str, cwd: Path, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


class ScriptsAreLocationIndependent(unittest.TestCase):
    def test_scripts_contain_no_machine_specific_paths(self):
        for script in sorted(SCRIPTS.glob("*.py")):
            text = script.read_text(encoding="utf-8")
            for needle in ("/Users/", "/home/", "miniforge", "psmc-master"):
                self.assertFalse(needle in text, f"{script.name} contains {needle!r}")

    def test_help_works_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            for script in sorted(SCRIPTS.glob("*.py")):
                if script.name == "exp_common.py":
                    continue
                res = run_script(script, "--help", cwd=Path(tmp))
                self.assertEqual(res.returncode, 0, f"{script.name} --help failed:\n{res.stderr}")
                self.assertIn("usage:", res.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [], "--help must not create files")

    def test_psmcfa_writer_creates_missing_parent_directories(self):
        sim = load_module(SCRIPTS / "simulate_msprime_to_psmcfa.py", "sim_psmcfa")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "dir" / "x.psmcfa"
            sim.write_psmcfa(out, "TTK" * 30)
            self.assertEqual(out.read_text().splitlines()[0], "> chr1")
            self.assertEqual(len("".join(out.read_text().splitlines()[1:])), 90)


@unittest.skipUnless(has_module("msprime"), "msprime not installed")
class MhsSimulationSmoke(unittest.TestCase):
    def test_tiny_simulation_writes_valid_multihetsep(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "tiny.mhs"
            res = run_script(
                SCRIPTS / "simulate_msprime_to_mhs.py",
                "--out",
                str(out),
                "--length",
                "200000",
                "--seed",
                "7",
                cwd=Path(tmp),
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            rows = [ln.split("\t") for ln in out.read_text().splitlines()]
            self.assertGreater(len(rows), 0)
            prev = 0
            for chrom, pos, nr_called, alleles in rows:
                self.assertEqual(chrom, "chr1")
                self.assertEqual(int(pos) - prev, int(nr_called))
                self.assertEqual(len(alleles), 2)
                self.assertNotEqual(alleles[0], alleles[1])
                prev = int(pos)


if __name__ == "__main__":
    unittest.main()
