"""Tests for the experiment scripts (stdlib unittest; no package installs needed).

Run from anywhere with:

    python3 -m unittest discover -s experiment/tests -v

Tests that need optional packages (msprime, numpy, ...) skip when those are absent.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiment" / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixtures  # noqa: E402


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_PLOT_DEPS = all(has_module(m) for m in ("numpy", "pandas", "matplotlib"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # dataclasses look their module up in sys.modules
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


class SharedHelpers(unittest.TestCase):
    def setUp(self):
        self.ec = load_module(SCRIPTS / "exp_common.py", "exp_common_t")

    def test_repo_relative_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inside = root / "experiment" / "runs" / "x.json"
            self.assertEqual(self.ec.repo_relative(inside, root), "experiment/runs/x.json")
            self.assertEqual(self.ec.resolve_repo_path("experiment/runs/x.json", root), inside)
            outside = Path(tempfile.gettempdir()).resolve() / "elsewhere.json"
            self.assertEqual(self.ec.repo_relative(outside, root), str(outside))

    def test_matched_settings_use_the_same_ratio_and_no_smoothing(self):
        s = self.ec.MatchedSettings()
        c = s.c_cmd(Path("psmc"), Path("in.psmcfa"), Path("out.psmc"))
        self.assertIn("-r4", c)
        self.assertIn("-N20", c)
        rust = s.rust_cmd(Path("psmc-rs"), Path("in.psmcfa"), Path("out.json"), mu=2.5e-8)
        self.assertEqual(rust[rust.index("--smooth-lambda") + 1], "0.0")
        self.assertEqual(rust[rust.index("--pattern") + 1], c[c.index("-p") + 1])

    def test_curves_from_both_tools_agree_on_identical_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            rj, cp = Path(tmp) / "a.json", Path(tmp) / "a.psmc"
            fixtures.write_rust_json(rj)
            fixtures.write_c_psmc(cp)
            rust = self.ec.curve_from_json(rj, 2.5e-8, 25.0)
            c = self.ec.curve_from_c_psmc(cp, 2.5e-8, 25.0)
            self.assertEqual(len(rust[0]), fixtures.N_STEPS + 2)
            self.assertLess(self.ec.rmse_log10(c, rust), 1e-5)
            fixtures.write_rust_json(rj, scale=10.0)
            shifted = self.ec.curve_from_json(rj, 2.5e-8, 25.0)
            self.assertAlmostEqual(self.ec.rmse_log10(c, shifted), 1.0, places=5)


class RealDataRunner(unittest.TestCase):
    def test_fit_summary_is_repo_relative_and_labels_papuan_as_chr1(self):
        rd = load_module(SCRIPTS / "run_real_data.py", "run_real_data_t")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            out_dir = root / "experiment" / "runs" / "real_data"
            for s in rd.SAMPLES:
                fixtures.write_rust_json(out_dir / "outputs" / f"{s.sid}.rust.json")
                fixtures.write_c_psmc(out_dir / "outputs" / f"{s.sid}.c.psmc")
            with contextlib.redirect_stdout(io.StringIO()):
                path = rd.write_fit_summary(out_dir, root=root)
            with path.open(newline="") as f:
                rows = {r["sample_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(set(rows), {"HLemySub1", "HLhydTec1", "HLpelCas1", "HomoSapiens"})
        for sid, r in rows.items():
            self.assertEqual(r["rust_json"], f"experiment/runs/real_data/outputs/{sid}.rust.json")
            self.assertEqual(r["c_psmc"], f"experiment/runs/real_data/outputs/{sid}.c.psmc")
            self.assertLess(float(r["rmse_log10_ne_rs_vs_c"]), 1e-5)
        self.assertIn("chr1", rows["HomoSapiens"]["species"])

    def test_fit_summary_update_keeps_the_rows_of_other_samples(self):
        # As after `run_real_data.py --samples HomoSapiens` in a clone that
        # has the committed summary but not the other samples' outputs.
        rd = load_module(SCRIPTS / "run_real_data.py", "run_real_data_merge")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            out_dir = root / "experiment" / "runs" / "real_data"
            for s in rd.SAMPLES:
                fixtures.write_rust_json(out_dir / "outputs" / f"{s.sid}.rust.json")
                fixtures.write_c_psmc(out_dir / "outputs" / f"{s.sid}.c.psmc")
            with contextlib.redirect_stdout(io.StringIO()):
                path = rd.write_fit_summary(out_dir, root=root)
            before = {r["sample_id"]: r for r in read_csv(path)}
            for s in rd.SAMPLES[:-1]:
                (out_dir / "outputs" / f"{s.sid}.rust.json").unlink()
                (out_dir / "outputs" / f"{s.sid}.c.psmc").unlink()
            fixtures.write_rust_json(out_dir / "outputs" / "HomoSapiens.rust.json", scale=10.0)
            with contextlib.redirect_stdout(io.StringIO()):
                rd.write_fit_summary(out_dir, root=root)
            after = {r["sample_id"]: r for r in read_csv(path)}
        self.assertEqual(set(after), set(before))
        for sid in ("HLemySub1", "HLhydTec1", "HLpelCas1"):
            self.assertEqual(after[sid], before[sid])
        self.assertAlmostEqual(float(after["HomoSapiens"]["rmse_log10_ne_rs_vs_c"]), 1.0, places=5)


class MainTextRunner(unittest.TestCase):
    def test_perf_summary_uses_sample_sd_in_figure_order(self):
        mt = load_module(SCRIPTS / "run_main_text.py", "run_main_text_t")
        rows = [
            {"model": m, "tool": t, "rep": rep, "wall_sec": w, "peak_rss_mb": 100.0}
            for m in ("zigzag", "constant")
            for t in ("rust", "c")
            for rep, w in ((1, 1.0), (2, 3.0))
        ]
        out = mt.summarise_perf(rows)
        self.assertEqual([(r["model"], r["tool"]) for r in out], [
            ("constant", "c"), ("constant", "rust"), ("zigzag", "c"), ("zigzag", "rust")
        ])
        self.assertAlmostEqual(out[0]["wall_sec_mean"], 2.0)
        self.assertAlmostEqual(out[0]["wall_sec_std"], 2 ** 0.5)
        self.assertEqual(out[0]["peak_rss_mb_std"], 0.0)


@unittest.skipUnless(HAS_PLOT_DEPS, "numpy/pandas/matplotlib not installed")
class SupplementaryRunner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.psmc_rs, self.c_psmc = fixtures.make_fake_tools(self.tmp / "bin")
        self.out = self.tmp / "supp"
        self.env = dict(os.environ, N_ITER="2", MPLCONFIGDIR=str(self.tmp / "mpl"))

    def tearDown(self):
        self._tmp.cleanup()

    def run_steps(self, steps: str, env) -> subprocess.CompletedProcess:
        return run_script(
            SCRIPTS / "run_supplementary.py",
            "--steps",
            steps,
            "--out-dir",
            str(self.out),
            "--psmc-rs",
            str(self.psmc_rs),
            "--c-psmc",
            str(self.c_psmc),
            cwd=self.tmp,
            env=env,
        )

    def test_s3_rerun_keeps_its_timings(self):
        # A second run into the same directory used to find the outputs of the
        # first, skip the timed runs and write NaN timings.
        inp = self.out / "shared_inputs" / "constant.1000000bp.psmcfa"
        inp.parent.mkdir(parents=True)
        inp.write_text(">chr1\nTTTTKTTTTK\n")
        env = dict(self.env, S3_LENGTHS_MB="1", S3_REPEATS="2")
        for attempt in (1, 2):
            res = self.run_steps("S3", env)
            self.assertEqual(res.returncode, 0, f"run {attempt}:\n{res.stderr[-3000:]}")
            rows = read_csv(self.out / "tables" / "S3_memory_scaling_summary.csv")
            self.assertEqual(sorted(r["tool"] for r in rows), ["c", "rust"])
            for r in rows:
                self.assertTrue(is_finite_number(r["wall_sec_mean"]), f"run {attempt}: {r}")


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
