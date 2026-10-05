"""Tests for what the reproduction package tracks in git and the data it ships."""

from __future__ import annotations

import csv
import json
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
            "experiment/runs/real_data/outputs/HomoSapiens.rust.settings.json",
            "experiment/runs/real_data/outputs/HomoSapiens.c.psmc",
            "experiment/runs/supplementary/s2_em/outputs/zigzag.rust.trace.json",
            "experiment/runs/supplementary/s2_em/outputs/zigzag.rust.trace.settings.json",
            "experiment/runs/supplementary/s2_em/outputs/zigzag.c.trace.psmc",
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
            "experiment/runs/supplementary/s3_memory/outputs/constant.100mb.rust.rep01.json",
            "experiment/runs/supplementary/s3_memory/outputs/constant.100mb.c.rep01.psmc",
            "experiment/runs/supplementary/s4_threads/outputs/zigzag.threads4.rep01.json",
            "experiment/runs/main_text/logs/commands.jsonl",
            "experiment/runs/supplementary/diagnostic_figures/S2_em_convergence.png",
            "experiment/runs/supplementary/tables/S2_em_convergence_trace.tsv",
        ):
            self.assertTrue(ignored(p), f"{p} should be ignored")

    def test_output_directories_exist_in_a_clone(self):
        for d in ("main_text", "real_data"):
            self.assertTrue((RUNS / d / "outputs" / ".gitkeep").exists(), d)


class PackagedTables(unittest.TestCase):
    FIT = RUNS / "real_data" / "tables" / "real_data_rust_vs_c_fit_summary.csv"

    def rows(self, path: Path):
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_fit_summary_paths_are_repository_relative(self):
        for r in self.rows(self.FIT):
            for col in ("rust_json", "c_psmc"):
                self.assertFalse(Path(r[col]).is_absolute(), f"{r['sample_id']} {col}: {r[col]}")
                self.assertTrue(r[col].startswith("experiment/runs/real_data/outputs/"), r[col])

    def test_no_table_contains_machine_specific_paths(self):
        for table in RUNS.glob("*/tables/*.csv"):
            text = table.read_text(encoding="utf-8")
            for needle in ("/Users/", "/home/", "C:\\"):
                self.assertFalse(needle in text, f"{table.relative_to(ROOT)} contains {needle!r}")

    def test_papuan_sample_is_labelled_as_chromosome_1_only(self):
        # The packaged psmcfa holds one sequence, '>chr1' (2,487,383 bins).
        psmcfa = ROOT / "experiment" / "real_data" / "Papuan_highlands_diploid.psmcfa"
        with psmcfa.open(encoding="utf-8") as f:
            headers = [ln.strip() for ln in f if ln.startswith(">")]
        self.assertEqual(headers, [">chr1"])
        for name in ("real_data_rust_vs_c_fit_summary.csv", "figure_3_empirical_rmse_rs_vs_c.csv"):
            species = {r["sample_id"]: r["species"] for r in self.rows(RUNS / "real_data" / "tables" / name)}
            self.assertIn("chr1", species["HomoSapiens"], name)
        nb = json.loads((ROOT / "experiment/notebooks/all_plots_precomputed.ipynb").read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"])
        self.assertIn("'HomoSapiens': 'Homo sapiens (Papuan Highlands, chr1)'", src)


class Documentation(unittest.TestCase):
    def test_no_machine_specific_paths_in_docs_or_entry_points(self):
        exp = ROOT / "experiment"
        # Hidden directories (.venv, .tools with its psmc-rs 0.1 checkout,
        # smoke runs) are local and ignored by git.
        files = [
            f
            for f in exp.rglob("*.md")
            if not any(part.startswith(".") for part in f.relative_to(exp).parts)
        ] + [exp / "run.py", exp / "requirements.txt"]
        for f in files:
            text = f.read_text(encoding="utf-8")
            for needle in ("/Users/", "/home/", "miniforge"):
                self.assertFalse(needle in text, f"{f.relative_to(ROOT)} contains {needle!r}")

    def test_readme_documents_every_pipeline_stage(self):
        text = (ROOT / "experiment" / "README.md").read_text(encoding="utf-8")
        for needle in (
            "### 1. Simulate",
            "### 2. Run psmc-rs and C psmc",
            "### 3. Tables",
            "### 4. Figures",
            "cc -O1 -o psmc cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz",
            "must be regenerated after the 0.2 fixes",
            "S5_format_consistency.csv",
            "chr1",
        ):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
