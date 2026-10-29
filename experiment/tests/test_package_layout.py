"""Tests for what the reproduction package tracks in git and the data it ships."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, assert_ignore_rules, in_git_work_tree  # noqa: E402

RUNS = ROOT / "experiment" / "runs"


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class IgnoreRules(unittest.TestCase):
    def test_e7s_data_stage_keeps_its_tables_manifest_and_records(self):
        assert_ignore_rules(self, kept=(
            "experiment/runs/E7/inputs/samples.tsv",
            "experiment/runs/E7/inputs/chunks.tsv",
            "experiment/runs/E7/inputs/MANIFEST.json",
            "experiment/runs/E7/inputs/records/samples.json",
            "experiment/runs/E7/inputs/records/chromosomes/chr21.json",
        ), ignored=(
            "experiment/runs/E7/inputs/psmcfa/NA12878.psmcfa.gz",
            "experiment/runs/E7/inputs/mask/callable.bed",
        ))

    def test_a_crossplatform_rerun_leaves_only_its_results(self):
        assert_ignore_rules(self, kept=(
            "experiment/crossplatform/runs/E1/fits/constant/rep01.rs.json",
        ), ignored=(
            "experiment/crossplatform/bin/psmc-rs",
            "experiment/crossplatform/logs/a.log",
            "experiment/crossplatform/logs-records-repeat/a.log",
            "experiment/crossplatform/runs/E1/fits/constant/rep01.rs.html",
        ))


class PackagedData(unittest.TestCase):
    def test_no_table_contains_machine_specific_paths(self):
        for table in RUNS.glob("*/tables/*.csv"):
            text = table.read_text(encoding="utf-8")
            for needle in ("/Users/", "/home/", "C:\\"):
                self.assertFalse(needle in text, f"{table.relative_to(ROOT)} contains {needle!r}")

    def test_papuan_sample_is_chromosome_1_only(self):
        # The psmcfa the Rust tests and docs/performance.md's benchmarks read holds one sequence, '>chr1'
        # (2,487,383 bins).
        psmcfa = ROOT / "experiment" / "real_data" / "Papuan_highlands_diploid.psmcfa"
        with psmcfa.open(encoding="utf-8") as f:
            headers = [ln.strip() for ln in f if ln.startswith(">")]
        self.assertEqual(headers, [">chr1"])


class Documentation(unittest.TestCase):
    def test_no_machine_specific_paths_in_docs(self):
        files = []
        for d, subdirs, names in os.walk(ROOT / "experiment"):
            # Hidden directories (.venv, .tools, smoke runs) are local and ignored by git.
            subdirs[:] = [s for s in subdirs if not s.startswith(".")]
            files += [Path(d) / n for n in names if n.endswith(".md") and not n.startswith(".")]
        self.assertTrue(files)
        for f in files:
            text = f.read_text(encoding="utf-8")
            for needle in ("/Users/", "/home/", "miniforge"):
                self.assertFalse(needle in text, f"{f.relative_to(ROOT)} contains {needle!r}")

    def test_readme_points_to_the_pipeline_and_every_experiment(self):
        text = (ROOT / "experiment" / "README.md").read_text(encoding="utf-8")
        for needle in (
            "python -m experiment.pipeline",
            "pipeline/README.md",
            "Supplementary_Note_1_Reproducibility.md",
            "slurm/README.md",
            "pipeline/requirements.lock",
            "tools --build-c",
            *(f"| E{i} |" for i in range(1, 8)),
            "| FIG |",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
