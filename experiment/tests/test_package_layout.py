"""Tests for what the reproduction package tracks in git and the data it ships."""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiment" / "runs"


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True
    )
    return res.returncode == 0 and res.stdout.strip() == "true"


def ignored(rel_path: str) -> bool:
    # --no-index: judge by the ignore rules alone, whether or not the file exists.
    res = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel_path], cwd=ROOT)
    return res.returncode == 0


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class IgnoreRules(unittest.TestCase):
    def test_small_inference_outputs_are_tracked(self):
        for p in (
            "experiment/runs/main_text/outputs/constant.rust.main.json",
            "experiment/runs/main_text/outputs/constant.c.main.psmc",
            "experiment/runs/real_data/outputs/HomoSapiens.rust.json",
            "experiment/runs/real_data/outputs/HomoSapiens.c.psmc",
            "experiment/runs/supplementary/s2_em/outputs/zigzag.rust.iter01.json",
            "experiment/runs/supplementary/s2_em/outputs/zigzag.rust.iter01.loglike.txt",
            "experiment/runs/main_text/outputs/.gitkeep",
            "experiment/runs/supplementary/tables/S2_em_convergence_trace.csv",
        ):
            self.assertFalse(ignored(p), f"{p} must not be ignored")

    def test_large_regenerable_artefacts_stay_ignored(self):
        for p in (
            "experiment/runs/main_text/inputs/constant.psmcfa",
            "experiment/runs/supplementary/shared_inputs/constant.500000000bp.psmcfa",
            "experiment/runs/supplementary/s5_format/inputs/zigzag.500000000bp.vcf",
            "experiment/runs/main_text/outputs/constant.rust.main.html",
            "experiment/runs/main_text/outputs/constant.rust.main.bootstrap/replicate_001.json",
            "experiment/runs/main_text/perf/constant.rust.rep01.json",
            "experiment/runs/main_text/logs/commands.jsonl",
            "experiment/runs/supplementary/diagnostic_figures/S2_em_convergence.png",
            "experiment/runs/supplementary/tables/S2_em_convergence_trace.tsv",
        ):
            self.assertTrue(ignored(p), f"{p} should be ignored")

    def test_output_directories_exist_in_a_clone(self):
        for d in ("main_text", "real_data"):
            self.assertTrue((RUNS / d / "outputs" / ".gitkeep").exists(), d)


if __name__ == "__main__":
    unittest.main()
