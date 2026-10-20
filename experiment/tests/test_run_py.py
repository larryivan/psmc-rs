"""Tests for experiment/run.py and the plotting notebook it executes.

Each test builds a throw-away copy of the reproduction package (run.py, the
notebook and the committed tables) in a temporary directory, so nothing in the
working tree is touched. Rendering tests skip without numpy/pandas/matplotlib.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures  # noqa: E402
from fixtures import ROOT, has_modules, quiet  # noqa: E402

EXP = ROOT / "experiment"
sys.path.insert(0, str(EXP / "scripts"))
import run_real_data  # noqa: E402

MODELS = ("constant", "bottleneck", "expansion", "zigzag")


def make_package(dest: Path, *, with_outputs: bool) -> Path:
    """Copy run.py, the notebook and the committed tables into a fake repository."""
    (dest / "src").mkdir(parents=True)
    (dest / "Cargo.toml").write_text("")
    (dest / "experiment" / "notebooks").mkdir(parents=True)
    shutil.copy2(EXP / "run.py", dest / "experiment" / "run.py")
    shutil.copy2(EXP / "notebooks" / "all_plots_precomputed.ipynb", dest / "experiment" / "notebooks")
    for table in (EXP / "runs").glob("*/tables/*.csv"):
        target = dest / table.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(table, target)
    if with_outputs:
        out = dest / "experiment" / "runs" / "main_text" / "outputs"
        for i, m in enumerate(MODELS):
            fixtures.write_rust_json(out / f"{m}.rust.main.json", scale=1.0 + 0.1 * i)
            fixtures.write_c_psmc(out / f"{m}.c.main.psmc", scale=1.0 + 0.1 * i)
        real = dest / "experiment" / "runs" / "real_data"
        for s in run_real_data.SAMPLES:
            fixtures.write_rust_json(real / "outputs" / f"{s.sid}.rust.json", theta=0.001)
            fixtures.write_c_psmc(real / "outputs" / f"{s.sid}.c.psmc", theta=0.001)
        with quiet():
            run_real_data.write_fit_summary(real, root=dest)
    return dest / "experiment" / "run.py"


class PackageCase(unittest.TestCase):
    """A package in ``self.repo``, run from a directory outside it."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.repo = self.tmp / "repo"
        self.elsewhere = self.tmp / "elsewhere"
        self.elsewhere.mkdir()

    def run_package(self, *args: str, with_outputs: bool, stubbed=()) -> subprocess.CompletedProcess:
        """Build the package and run its run.py with ``args``; the modules ``stubbed`` fail to import."""
        run_py = make_package(self.repo, with_outputs=with_outputs)
        return self.execute(run_py, *args, stubbed=stubbed)

    def execute(self, run_py: Path, *args: str, stubbed=()) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("PSMC_RS_ROOT", None)
        if stubbed:
            stubs = self.tmp / "stubs"
            stubs.mkdir()
            for name in stubbed:
                (stubs / f"{name}.py").write_text("raise ImportError('stubbed out by the test')\n")
            env["PYTHONPATH"] = str(stubs)
        return subprocess.run([sys.executable, str(run_py), *args], cwd=str(self.elsewhere), capture_output=True,
                              text=True, env=env, timeout=900)


class Checks(PackageCase):
    def test_missing_outputs_name_the_command_that_regenerates_them(self):
        res = self.run_package("--skip-plot", with_outputs=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("python3 experiment/scripts/run_main_text.py --steps simulate,main", res.stderr)
        self.assertIn("missing: experiment/runs/main_text/outputs/constant.rust.main.json", res.stderr)
        self.assertIn("python3 experiment/scripts/run_real_data.py", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_check_needs_no_plotting_packages(self):
        res = self.run_package("--skip-plot", with_outputs=True, stubbed=("numpy", "pandas", "matplotlib"))
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("[check] required files and key tables are valid.", res.stdout)

    def test_timing_tables_without_values_are_reported(self):
        # What an S3 re-run that reused its outputs used to write.
        run_py = make_package(self.repo, with_outputs=True)
        s3 = self.repo / "experiment/runs/supplementary/tables/S3_memory_scaling_summary.csv"
        header = s3.read_text().splitlines()[0]
        s3.write_text(header + "\n1,c,,,,\n1,rust,,,,\n")
        res = self.execute(run_py, "--skip-plot")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("S3_memory_scaling_summary.csv contains no finite", res.stderr)
        self.assertIn("run_supplementary.py --steps S3", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_missing_plotting_packages_are_named_for_this_interpreter(self):
        res = self.run_package("--skip-check", with_outputs=True, stubbed=("pandas", "matplotlib"))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn(f"{sys.executable} cannot import: ", res.stderr)
        self.assertIn("pandas, matplotlib", res.stderr)
        self.assertIn(f"{sys.executable} -m pip install numpy pandas matplotlib", res.stderr)
        self.assertNotIn("Traceback", res.stderr)


@unittest.skipUnless(has_modules("numpy", "pandas", "matplotlib"), "numpy/pandas/matplotlib not installed")
class Rendering(PackageCase):
    STEMS = {
        "main_text": ("figure_2_simulated_equivalence", "figure_4_performance_scalability"),
        "real_data": ("figure_3_empirical_genomes_rust_vs_c",),
        "supplementary": (
            "supplementary_figure_S1_em_convergence",
            "supplementary_figure_S2_format_consistency",
            "supplementary_figure_S3_interfaces",
        ),
    }

    def figure(self, group: str, stem: str) -> Path:
        return self.repo / "experiment" / "runs" / group / "figures" / f"{stem}.png"

    def test_all_figures_render_when_started_outside_the_repository(self):
        res = self.run_package(with_outputs=True)
        self.assertEqual(res.returncode, 0, res.stderr[-3000:])
        for group, stems in self.STEMS.items():
            for stem in stems:
                self.assertTrue(self.figure(group, stem).exists(), stem)
        rmse = (self.repo / "experiment/runs/real_data/tables/figure_3_empirical_rmse_rs_vs_c.csv").read_text()
        self.assertEqual(len(rmse.strip().splitlines()), 5)
        self.assertIn("Homo sapiens (Papuan Highlands, chr1)", rmse)

    def test_keep_going_draws_the_figures_whose_inputs_exist(self):
        res = self.run_package("--keep-going", with_outputs=False)
        self.assertEqual(res.returncode, 1, res.stderr[-3000:])
        self.assertIn("run_main_text.py --steps simulate,main", res.stderr)
        self.assertFalse(self.figure("main_text", "figure_2_simulated_equivalence").exists())
        self.assertFalse(self.figure("real_data", "figure_3_empirical_genomes_rust_vs_c").exists())
        self.assertTrue(self.figure("main_text", "figure_4_performance_scalability").exists())
        for stem in self.STEMS["supplementary"]:
            self.assertTrue(self.figure("supplementary", stem).exists(), stem)

    def test_keep_going_survives_a_fit_summary_missing_a_sample(self):
        # run_real_data.py --samples <subset> can produce such a table; the
        # Figure 3 cell then fails with ValueError, not FileNotFoundError.
        run_py = make_package(self.repo, with_outputs=True)
        fit = self.repo / "experiment/runs/real_data/tables/real_data_rust_vs_c_fit_summary.csv"
        lines = fit.read_text().splitlines()
        fit.write_text("\n".join(ln for ln in lines if not ln.startswith("HLpelCas1,")) + "\n")
        res = self.execute(run_py, "--keep-going")
        self.assertEqual(res.returncode, 1, res.stderr[-3000:])
        self.assertIn("missing sample_id: HLpelCas1", res.stderr)
        self.assertIn("missing sample in fit summary: HLpelCas1", res.stderr)
        self.assertNotIn("Traceback", res.stderr)
        self.assertFalse(self.figure("real_data", "figure_3_empirical_genomes_rust_vs_c").exists())
        for group, stems in self.STEMS.items():
            for stem in stems:
                if group != "real_data":
                    self.assertTrue(self.figure(group, stem).exists(), stem)


if __name__ == "__main__":
    unittest.main()
