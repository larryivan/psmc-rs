"""What git keeps of the pipeline's runs directory (experiment/runs/<experiment>/)."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def ignored(rel_path: str) -> bool:
    # --no-index: judge by the ignore rules alone, whether or not the file exists.
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel_path], cwd=ROOT).returncode == 0


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    def test_small_results_are_tracked(self):
        for p in (
            "experiment/runs/E1/fits/bottleneck/rep01.c.psmc",
            "experiment/runs/E1/fits/bottleneck/rep01.rs.json",
            "experiment/runs/E1/fits/bottleneck/rep01.rs.psmc",
            "experiment/runs/E1/truth/bottleneck/rep01.json",
            "experiment/runs/E1/provenance/fit-c/bottleneck/rep01.json",
            "experiment/runs/E1/provenance/report.json",
            "experiment/runs/E1/tables/E1_table1.csv",
        ):
            self.assertFalse(ignored(p), p)

    def test_regenerable_and_local_files_are_ignored(self):
        for p in (
            "experiment/runs/E1/sims/bottleneck/rep01/genome.psmcfa",
            "experiment/runs/E1/sims/bottleneck/rep01/genome.vcf.gz",
            "experiment/runs/E1/sims/bottleneck/rep01/genome.vcf.gz.tbi",
            "experiment/runs/E1/fits/bottleneck/rep01.rs.html",
            "experiment/runs/E1/fits/bottleneck/rep01.c.psmc.partial",
            "experiment/runs/E1/logs/attempts.jsonl",
            "experiment/runs/E1/logs/jobs/fit-c/bottleneck/rep01.log",
            "experiment/runs/_runner/runner.pid",
            "experiment/runs/_runner/checksums.json",
            "experiment/.runs-smoke/E1/tables/E1_table1.csv",
            "experiment/.tools/psmc-rs-0.1/target/release/psmc-rs",
        ):
            self.assertTrue(ignored(p), p)


if __name__ == "__main__":
    unittest.main()
