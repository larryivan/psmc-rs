"""What git keeps of the pipeline's runs directory (experiment/runs/<experiment>/)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import assert_ignore_rules, in_git_work_tree  # noqa: E402


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    def test_small_results_are_tracked_and_regenerable_and_local_files_ignored(self):
        assert_ignore_rules(self, kept=(
            "experiment/runs/E1/fits/bottleneck/rep01.c.psmc",
            "experiment/runs/E1/fits/bottleneck/rep01.rs.json",
            "experiment/runs/E1/fits/bottleneck/rep01.rs.psmc",
            "experiment/runs/E1/truth/bottleneck/rep01.json",
            "experiment/runs/E1/provenance/fit-c/bottleneck/rep01.json",
            "experiment/runs/E1/provenance/report.json",
            "experiment/runs/E1/tables/E1_table1.csv",
        ), ignored=(
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
        ))


if __name__ == "__main__":
    unittest.main()
