"""Tests for the experiment scripts (stdlib unittest; no package installs needed).

Run from anywhere with:

    python3 -m unittest discover -s experiment/tests -v

Tests that need optional packages (msprime, numpy, ...) skip when those are absent.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


def c_rounds(path: Path) -> int:
    return sum(1 for ln in path.read_text().splitlines() if ln == "//")


class FakeToolsCase(unittest.TestCase):
    """Runs the runners in-process against the stand-in tools of fixtures.py."""

    N_ITER = 5

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self.psmc_rs, self.c_psmc = fixtures.make_fake_tools(self.tmp / "bin")
        self.calls_log = self.tmp / "calls.log"
        env = mock.patch.dict(os.environ, {"N_ITER": str(self.N_ITER), "FAKE_TOOL_LOG": str(self.calls_log)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self._tmp.cleanup)

    def calls(self, tool: str, output_name: str):
        """Logged calls of ``tool`` whose output starts with ``output_name`` (staged names too)."""
        if not self.calls_log.exists():
            return []
        out = []
        for ln in self.calls_log.read_text().splitlines():
            t, name, _ = ln.split()
            if t == tool and name.startswith(output_name):
                out.append(ln)
        return out


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
        self.assertEqual(rust[3], "20")
        self.assertEqual(rust[rust.index("--smooth-lambda") + 1], "0.0")
        self.assertEqual(rust[rust.index("--pattern") + 1], c[c.index("-p") + 1])
        self.assertEqual(f"-t{rust[rust.index('--t-max') + 1]}", "-t15.0")
        self.assertIn("-t15.0", c)
        self.assertEqual(f"-r{rust[rust.index('--tr-ratio') + 1]}", "-r4")
        # psmc-rs 0.2's default, a cap its converged M-steps stay below.
        self.assertEqual(rust[rust.index("--mstep-iters") + 1], "1000")
        # The pattern sets the interval count, as in C psmc; batching changes nothing in 0.2.
        self.assertNotIn("--n-steps", rust)
        self.assertNotIn("--batch-size", rust)

    def test_matched_settings_from_the_environment(self):
        env = {"PATTERN": "4+5*3+4", "RHO_T_RATIO": "5", "MSTEP_ITERS": "50", "BATCH_SIZE": "300000", "N_ITER": "3"}
        with mock.patch.dict(os.environ, env):
            s = self.ec.MatchedSettings.from_env()
        rust = s.rust_cmd(Path("psmc-rs"), Path("in.psmcfa"), Path("out.json"), mu=2.5e-8, extra=["-q"])
        c = s.c_cmd(Path("psmc"), Path("in.psmcfa"), Path("out.psmc"))
        self.assertEqual(rust[rust.index("--pattern") + 1], "4+5*3+4")
        self.assertEqual(rust[rust.index("--tr-ratio") + 1], "5")
        self.assertEqual(rust[rust.index("--mstep-iters") + 1], "50")
        self.assertEqual(rust[rust.index("--batch-size") + 1], "300000")
        self.assertEqual(rust[-1], "-q")
        self.assertEqual(c[1:4], ["-N3", "-t15.0", "-r5"])

    def test_supplementary_runs_use_the_matched_settings(self):
        rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_cmd")
        rs.PSMC_RS_BIN = Path("psmc-rs")
        ec_cmd = rs.SETTINGS.rust_cmd(Path("psmc-rs"), Path("in"), Path("out.json"), mu=rs.MU)
        for fmt in ("psmcfa", "mhs", "vcf"):
            cmd = rs.rust_cmd(Path("in"), Path("out.json"), 3, input_format=fmt, bin_size=50, threads=2)
            self.assertEqual(cmd[3], "3")
            for flag in ("--pattern", "--t-max", "--tr-ratio", "--mstep-iters", "--mu"):
                self.assertEqual(cmd[cmd.index(flag) + 1], ec_cmd[ec_cmd.index(flag) + 1], flag)
            self.assertEqual(cmd[cmd.index("--smooth-lambda") + 1], "0.0")
            self.assertEqual(cmd[cmd.index("--threads") + 1], "2")
            self.assertEqual(cmd[cmd.index("--input-format") + 1], fmt)
            self.assertEqual(cmd[cmd.index("--bin-size") + 1], "50")
            self.assertEqual("--vcf-mode" in cmd, fmt == "vcf", fmt)
        c = []
        rs.C_PSMC_BIN = Path("psmc")
        with mock.patch.object(rs.exp_common, "run_staged", lambda cmd_for, out, **kw: c.extend(cmd_for(out))):
            rs.run_c(Path("in.psmcfa"), Path("out.psmc"), n_iter=3, bootstrap=True)
        self.assertEqual(c[:4], ["psmc", "-N3", f"-t{rs.T_MAX}", f"-r{rs.RHO_T_RATIO}"])
        self.assertEqual(c[c.index("-b") + 1 :], ["-o", "out.psmc", "in.psmcfa"])

    def test_c_loglikes_leave_out_round_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.psmc"
            fixtures.write_c_psmc(p, rounds=range(4))  # round 0 prints LK 0.0
            self.assertEqual(self.ec.c_psmc_loglikes(p), [(1, -12345.6), (2, -12345.6), (3, -12345.6)])

    def test_c_header_is_read_as_c_psmc_writes_it(self):
        real = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
        self.assertEqual(
            self.ec.c_psmc_header(real), {"pattern": "4+25*2+4+6", "n_iter": 5, "t_max": 15.0, "tr_ratio": 4.0}
        )
        settings = self.ec.MatchedSettings(n_iter=5)
        self.assertIsNone(self.ec.c_psmc_stale(real, settings))
        other = self.ec.MatchedSettings(n_iter=5, pattern="4+5*3+4", t_max=20.0, rho_t_ratio=5)
        self.assertEqual(
            self.ec.c_psmc_stale(real, other),
            "was made with -p 4+25*2+4+6 (PATTERN is 4+5*3+4), -t15 (T_MAX is 20), -r4 (RHO_T_RATIO is 5)",
        )
        self.assertIn("not a finished 4-iteration run (6 rounds)", self.ec.c_psmc_stale(real, self.ec.MatchedSettings(n_iter=4)))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.ec.c_psmc_stale(Path(tmp) / "none.psmc", settings), "does not exist")

    def test_psmc_rs_fit_args_leave_out_what_does_not_change_the_fit(self):
        s = self.ec.MatchedSettings(threads=8, batch_size=300000)
        cmd = s.rust_cmd(Path("/a/psmc-rs"), Path("/b/in.psmcfa"), Path("/c/out.json"), mu=1e-8, extra=["--bin-size", "50"])
        args = self.ec.psmc_rs_fit_args(cmd)
        self.assertEqual(args[0], "20")
        for gone in ("--threads", "8", "--batch-size", "300000", "--no-progress", "/b/in.psmcfa", "/c/out.json"):
            self.assertNotIn(gone, args)
        for kept in ("--pattern", "--t-max", "--tr-ratio", "--smooth-lambda", "--mstep-iters", "--mu", "--bin-size"):
            self.assertIn(kept, args)
        self.assertEqual(self.ec.psmc_rs_fit_args([*cmd, "--threads=2"]), args)
        self.assertEqual(self.ec.psmc_rs_fit_args(self.ec.MatchedSettings().rust_cmd(
            Path("psmc-rs"), Path("x"), Path("y"), mu=1e-8, extra=["--bin-size", "50"])), args)

    def test_psmc_rs_stale_names_the_changed_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fit.json"
            s = self.ec.MatchedSettings(n_iter=3)
            cmd = s.rust_cmd(Path("psmc-rs"), Path("in"), out, mu=1e-8)
            self.assertEqual(self.ec.psmc_rs_stale(out, cmd), "does not exist")
            fixtures.write_rust_json(out, n_iter=3)
            self.assertIn("has no record of its settings (fit.settings.json)", self.ec.psmc_rs_stale(out, cmd))
            self.ec.record_psmc_rs_run(out, cmd)
            self.assertIsNone(self.ec.psmc_rs_stale(out, cmd))
            other = self.ec.MatchedSettings(n_iter=3, t_max=20.0, mstep_iters=50).rust_cmd(
                Path("psmc-rs"), Path("in"), out, mu=1e-8
            )
            self.assertEqual(
                self.ec.psmc_rs_stale(out, other),
                "was made with --t-max 15.0, --mstep-iters 1000 (now --t-max 20.0, --mstep-iters 50)",
            )
            more = self.ec.MatchedSettings(n_iter=4).rust_cmd(Path("psmc-rs"), Path("in"), out, mu=1e-8)
            self.assertIn("finished 4-iteration", self.ec.psmc_rs_stale(out, more))

    def test_psmc_rs_loglikes_are_numbered_like_c_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fit.json"
            fixtures.write_rust_json(p, n_iter=3)
            # loglike[i], after i iterations, is C's round i + 1; loglike[3] has no C round.
            want = [(k, fixtures.rust_loglike(k - 1)) for k in (1, 2, 3)]
            self.assertEqual(self.ec.psmc_rs_loglikes(p, 3), want)
            self.assertIsNone(self.ec.psmc_rs_loglikes(p, 4), "another N_ITER")
            fixtures.write_rust_json(p)  # a 0.1 file: no trace
            self.assertIsNone(self.ec.psmc_rs_loglikes(p, 3))
            p.write_text("{")
            self.assertIsNone(self.ec.psmc_rs_loglikes(p, 3))
            self.assertIsNone(self.ec.psmc_rs_loglikes(Path(tmp) / "missing.json", 3))

    def test_bins_after_last_het(self):
        rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_bins")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.psmcfa"
            for seq, want in (("TTKTT", 2), ("TTTTK", 0), ("TTTT", 4), ("K" + "T" * 70, 70)):
                p.write_text("> chr1\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n")
                self.assertEqual(rs.bins_after_last_het(p), want, seq)

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

    @staticmethod
    def notebook_lam_helpers():
        """The notebook's own pattern helpers (pure Python), without running its plotting setup."""
        nb = json.loads((ROOT / "experiment/notebooks/all_plots_precomputed.ipynb").read_text(encoding="utf-8"))
        tree = ast.parse("".join(nb["cells"][0]["source"]))
        wanted = {"parse_pattern_spec", "parse_pattern_spec_legacy", "expand_lam"}
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
        scope: dict = {}
        exec(compile(tree, "all_plots_precomputed.ipynb:cell0", "exec"), scope)
        return scope

    def test_expand_lam_matches_the_notebook_for_every_lam_length(self):
        # Legacy psmc-rs read a plain pattern part `a` as a states of width 1:
        # for 4+25*2+4+6 that is 4+25+4+6 = 39 values plus a trailing one.
        nb = self.notebook_lam_helpers()

        def outcome(fn, *args):
            try:
                return fn(*args)
            except ValueError:
                return "rejected"

        pattern = fixtures.PATTERN
        for n_steps in (63, 64):
            for n in range(20, 70):
                lam = [float(i + 1) for i in range(n)]
                want = outcome(nb["expand_lam"], lam, n_steps, nb["parse_pattern_spec"](pattern), pattern)
                got = outcome(self.ec.expand_lam, lam, n_steps, pattern)
                self.assertEqual(got, want, f"n_steps={n_steps}, {n} lam values")
        legacy = self.ec.expand_lam([float(i) for i in range(40)], 64, pattern)
        self.assertEqual(legacy[:6], [0.0, 1.0, 2.0, 3.0, 4.0, 4.0])
        self.assertEqual(len(legacy), 65)


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


class InterruptedRuns(FakeToolsCase):
    """C psmc writes every round as it goes; a partial output must never be reused."""

    def setUp(self):
        super().setUp()
        self.rd = load_module(SCRIPTS / "run_real_data.py", "run_real_data_int")
        self.ec = self.rd.exp_common
        self.sample = self.rd.SAMPLES[-1]
        self.inputs = self.tmp / "in"
        self.inputs.mkdir()
        (self.inputs / self.sample.file).write_text(">chr1\nTTTTKTTTTK\n")
        self.out = self.tmp / "real_data"
        self.r_out = self.rd.rust_json_path(self.out, self.sample)
        self.c_out = self.rd.c_psmc_path(self.out, self.sample)

    def run_real_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.rd.run_samples([self.sample], self.out, str(self.psmc_rs), str(self.c_psmc), False, self.inputs)

    def test_failed_c_run_leaves_no_output_behind(self):
        with mock.patch.dict(os.environ, {"FAKE_C_ROUNDS": "3", "FAKE_C_EXIT": "130"}):
            with self.assertRaises(RuntimeError):
                self.run_real_data()
        self.assertFalse(self.c_out.exists(), "a partial C output was left at the final path")

    def test_incomplete_cached_c_output_is_run_again(self):
        self.run_real_data()
        fixtures.write_c_psmc(self.c_out, rounds=range(3), n_iter=self.N_ITER)  # interrupted after round 2 of 5
        self.run_real_data()
        self.assertEqual(c_rounds(self.c_out), self.N_ITER + 1)
        self.assertEqual(len(self.calls("c", self.c_out.name)), 2)
        self.assertEqual(len(self.calls("psmc-rs", self.r_out.name)), 1, "a finished psmc-rs output is reused")
        self.run_real_data()
        self.assertEqual(len(self.calls("c", self.c_out.name)), 2, "a finished C output is reused")

    def test_truncated_psmc_rs_output_is_run_again(self):
        fixtures.write_rust_json(self.r_out)
        self.r_out.write_text(self.r_out.read_text()[:40])
        fixtures.write_c_psmc(self.c_out, rounds=range(self.N_ITER + 1))
        self.run_real_data()
        self.assertEqual(len(self.calls("psmc-rs", self.r_out.name)), 1)
        self.assertEqual(self.calls("c", self.c_out.name), [])

    def test_main_text_reruns_an_incomplete_c_output(self):
        mt = load_module(SCRIPTS / "run_main_text.py", "run_main_text_int")
        out = self.tmp / "main_text"
        runner = mt.Runner(out, str(self.psmc_rs), str(self.c_psmc), self.tmp / "unused_sim.py")
        runner.input_path("constant").write_text(">chr1\nTTTTKTTTTK\n")
        c_out = out / "outputs" / "constant.c.main.psmc"
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main_runs(["constant"], force=False)
        fixtures.write_c_psmc(c_out, rounds=range(2), n_iter=self.N_ITER)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.main_runs(["constant"], force=False)
        self.assertEqual(c_rounds(c_out), self.N_ITER + 1)
        self.assertEqual(len(self.calls("c", c_out.name)), 2)
        self.assertEqual(len(self.calls("psmc-rs", "constant.rust.main")), 1)

    def test_psmc_rs_output_of_another_n_iter_is_run_again(self):
        # As C outputs with another round count are: N_ITER changed, or the
        # file came from psmc-rs 0.1, which recorded no n_iter.
        fixtures.write_c_psmc(self.c_out, rounds=range(self.N_ITER + 1))
        for stale in (dict(n_iter=self.N_ITER - 1), dict(n_iter=None)):
            fixtures.write_rust_json(self.r_out, **stale)
            self.run_real_data()
        self.assertEqual(len(self.calls("psmc-rs", self.r_out.name)), 2)
        self.run_real_data()
        self.assertEqual(len(self.calls("psmc-rs", self.r_out.name)), 2, "a matching output is reused")
        self.assertEqual(self.calls("c", self.c_out.name), [])

    def test_outputs_of_other_settings_are_run_again(self):
        # Regenerating over 0.1-era outputs: C psmc ran with -r5 and the
        # psmc-rs JSON records no settings. Both must be made again at -r4.
        fixtures.write_c_psmc(self.c_out, rounds=range(self.N_ITER + 1), tr_ratio=5)
        fixtures.write_rust_json(self.r_out, rho_ratio=5)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.rd.run_samples([self.sample], self.out, str(self.psmc_rs), str(self.c_psmc), False, self.inputs)
        self.assertIn(f"{self.c_out.name} was made with -r5 (RHO_T_RATIO is 4); re-running", out.getvalue())
        self.assertEqual(len(self.calls("c", self.c_out.name)), 1)
        self.assertEqual(len(self.calls("psmc-rs", self.r_out.name)), 1)
        self.assertEqual(self.ec.c_psmc_header(self.c_out)["tr_ratio"], 4.0)
        self.assertEqual(json.loads(self.r_out.read_text())["tr_ratio"], 4.0)

    def test_changed_settings_rerun_both_tools_and_threads_do_not(self):
        self.run_real_data()
        for env, rust, c in (
            ({"RUST_THREADS": "4", "BATCH_SIZE": "1000"}, 0, 0),  # results do not depend on these
            ({"T_MAX": "20"}, 1, 1),  # psmc-rs's JSON does not record its starting t_max
            ({"T_MAX": "20", "PATTERN": "4+5*3+4"}, 1, 1),
            ({"T_MAX": "20", "PATTERN": "4+5*3+4", "MSTEP_ITERS": "50"}, 1, 0),  # psmc-rs's alone
            ({"T_MAX": "20", "PATTERN": "4+5*3+4", "MSTEP_ITERS": "50", "N_ITER": "3"}, 1, 1),
        ):
            before = (len(self.calls("psmc-rs", self.r_out.name)), len(self.calls("c", self.c_out.name)))
            with mock.patch.dict(os.environ, env):
                self.run_real_data()
            after = (len(self.calls("psmc-rs", self.r_out.name)), len(self.calls("c", self.c_out.name)))
            self.assertEqual(after, (before[0] + rust, before[1] + c), env)

    def test_a_failed_psmc_rs_run_leaves_no_settings_record(self):
        self.run_real_data()
        record = self.ec.psmc_rs_settings_path(self.r_out)
        self.assertEqual(record.name, "HomoSapiens.rust.settings.json")
        self.assertTrue(record.exists())
        cmd = self.ec.MatchedSettings.from_env().rust_cmd(self.psmc_rs, self.inputs / self.sample.file, self.r_out, mu=1e-8)
        with mock.patch.dict(os.environ, {"FAKE_PRE_0_2": "1"}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                self.ec.run_psmc_rs(cmd, self.r_out, log_path=self.tmp / "log.jsonl")
        self.assertFalse(record.exists())
        self.assertIn("has no record of its settings", self.ec.psmc_rs_stale(self.r_out, cmd))

    def test_supplementary_c_runs_are_staged(self):
        rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_int")
        rs.configure_paths(self.tmp / "supp")
        rs.C_PSMC_BIN = self.c_psmc
        out = self.tmp / "supp" / "s2_em" / "outputs" / "constant.c.trace.psmc"
        with mock.patch.dict(os.environ, {"FAKE_C_ROUNDS": "2", "FAKE_C_EXIT": "1"}):
            with self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                rs.run_c(self.inputs / self.sample.file, out, n_iter=self.N_ITER)
        self.assertFalse(out.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            rs.run_c(self.inputs / self.sample.file, out, n_iter=self.N_ITER)
        self.assertEqual(c_rounds(out), self.N_ITER + 1)
        self.assertEqual(list(out.parent.glob("*.partial")), [])


class SupplementaryConvergence(FakeToolsCase):
    """S2 pairs the two tools by what their log-likelihoods measure, from one run each."""

    def setUp(self):
        super().setUp()
        self.rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_s2")
        self.rs.configure_paths(self.tmp / "supp")
        self.rs.configure_tools(str(self.psmc_rs), str(self.c_psmc), None)
        for key in self.rs.MODEL_ORDER:
            self.rs.shared_input_path(key, self.rs.SIM_LENGTH_BP).write_text(">chr1\nTTTTKTTTTK\n")

    def rows(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.rs.s2_trace_rows()

    def test_rows_pair_the_rounds_of_both_tools(self):
        rows = self.rows()
        for key in self.rs.MODEL_ORDER:
            for tool in ("c", "rust"):
                iters = [r["iter"] for r in rows if r["model"] == key and r["tool"] == tool]
                self.assertEqual(iters, list(range(1, self.N_ITER + 1)), (key, tool))
            rust = {r["iter"]: r["loglike"] for r in rows if r["model"] == key and r["tool"] == "rust"}
            # Row k: the log-likelihood after k - 1 iterations, not the line printed after k.
            self.assertEqual(rust, {k: fixtures.rust_loglike(k - 1) for k in range(1, self.N_ITER + 1)})
            self.assertEqual(
                [ln.split()[2] for ln in self.calls("psmc-rs", f"{key}.rust")], [str(self.N_ITER)], key
            )

    def test_a_finished_trace_is_reused_and_another_n_iter_is_run_again(self):
        self.rows()
        self.rows()
        self.assertEqual(len(self.calls("psmc-rs", "constant.rust")), 1)
        out = self.rs.S2_DIR / "outputs" / "constant.rust.trace.json"
        fixtures.write_rust_json(out, n_iter=self.N_ITER + 1)
        self.rows()
        self.assertEqual(len(self.calls("psmc-rs", "constant.rust")), 2)


class RhoRatioGuard(FakeToolsCase):
    """C psmc runs with -r RHO_T_RATIO; psmc-rs must default to the same rho0 = theta0/ratio."""

    def setUp(self):
        super().setUp()
        self.inputs = self.tmp / "in"
        self.inputs.mkdir()

    def run_real_data(self, **env):
        rd = load_module(SCRIPTS / "run_real_data.py", "run_real_data_ratio")
        sample = rd.SAMPLES[-1]
        (self.inputs / sample.file).write_text(">chr1\nTTTTKTTTTK\n")
        with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            rd.run_samples([sample], self.tmp / "out", str(self.psmc_rs), str(self.c_psmc), False, self.inputs)

    def comparison_runs(self):
        return [ln for ln in self.calls_log.read_text().splitlines() if not ln.startswith("psmc-rs probe")]

    def test_mismatched_ratio_stops_before_any_run(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_real_data(FAKE_RHO_RATIO="5")
        self.assertIn("theta0/5", str(cm.exception))
        self.assertIn("RHO_T_RATIO", str(cm.exception))
        self.assertEqual(self.comparison_runs(), [])

    def test_matching_ratio_runs_both_tools(self):
        self.run_real_data(FAKE_RHO_RATIO="5", RHO_T_RATIO="5")
        self.assertEqual(sorted(ln.split()[0] for ln in self.comparison_runs()), ["c", "psmc-rs"])

    def test_the_probe_passes_the_ratio_given_to_c(self):
        # The stand-in starts from theta0/--tr-ratio, as psmc-rs 0.2 does, so the
        # check passes only if the probe gets the --tr-ratio of the real runs.
        self.run_real_data(RHO_T_RATIO="7")
        self.assertEqual(sorted(ln.split()[0] for ln in self.comparison_runs()), ["c", "psmc-rs"])

    def test_a_build_without_tr_ratio_is_refused_with_advice(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_real_data(FAKE_PRE_0_2="1")
        self.assertIn("unexpected argument '--tr-ratio'", str(cm.exception))
        self.assertIn("psmc-rs 0.2", str(cm.exception))
        self.assertEqual(self.comparison_runs(), [])

    def test_main_text_checks_before_running_c(self):
        mt = load_module(SCRIPTS / "run_main_text.py", "run_main_text_ratio")
        runner = mt.Runner(self.tmp / "main", str(self.psmc_rs), str(self.c_psmc), self.tmp / "unused_sim.py")
        runner.input_path("constant").write_text(">chr1\nTTTTKTTTTK\n")
        fixtures.write_rust_json(runner.outputs / "constant.rust.main.json")
        with mock.patch.dict(os.environ, {"FAKE_RHO_RATIO": "5"}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main_runs(["constant"], force=False)
        self.assertEqual(self.comparison_runs(), [])

    def test_supplementary_checks_only_the_steps_that_compare_with_c(self):
        rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_ratio")
        rs.configure_paths(self.tmp / "supp")
        rs.configure_tools(str(self.psmc_rs), str(self.c_psmc), None)
        with mock.patch.dict(os.environ, {"FAKE_RHO_RATIO": "5"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(rs.ensure_tools(require_c=False))  # S4, S5: psmc-rs only
            with self.assertRaises(SystemExit):
                rs.ensure_tools(require_c=True)  # S1-S3


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


class SimulationScriptLookup(unittest.TestCase):
    """--sim-script > $SIM_SCRIPT > the packaged script, in both runners that simulate."""

    FAKE_SIM = (
        "import sys\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "open(out, 'w').write('>fake\\nTTTK\\n')\n"
    )

    def test_main_text_honours_the_environment_variable(self):
        mt = load_module(SCRIPTS / "run_main_text.py", "run_main_text_sim")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp).resolve()
            fake = tmp / "fake_sim.py"
            fake.write_text(self.FAKE_SIM)
            stubs = {m: type(sys)(m) for m in ("stdpopsim", "msprime") if not has_module(m)}
            with mock.patch.dict(os.environ, {"SIM_SCRIPT": str(fake)}), mock.patch.dict(sys.modules, stubs):
                with contextlib.redirect_stdout(io.StringIO()):
                    mt.main(["--steps", "simulate", "--models", "constant", "--out-dir", str(tmp / "out")])
            self.assertEqual((tmp / "out" / "inputs" / "constant.psmcfa").read_text(), ">fake\nTTTK\n")

    def test_flag_beats_environment_beats_default(self):
        ec = load_module(SCRIPTS / "exp_common.py", "exp_common_sim")
        with mock.patch.dict(os.environ, {"SIM_SCRIPT": "/env/sim.py"}):
            self.assertEqual(ec.resolve_sim_script("/flag/sim.py"), Path("/flag/sim.py").resolve())
            self.assertEqual(ec.resolve_sim_script(None), Path("/env/sim.py").resolve())
        with mock.patch.dict(os.environ):
            os.environ.pop("SIM_SCRIPT", None)
            self.assertEqual(ec.resolve_sim_script(None), ec.SIM_SCRIPT_DEFAULT)

    def test_supplementary_honours_the_environment_variable(self):
        rs = load_module(SCRIPTS / "run_supplementary.py", "run_supplementary_sim")
        with mock.patch.dict(os.environ, {"SIM_SCRIPT": "/env/sim.py"}):
            rs.configure_tools(None, None, None)
            self.assertEqual(rs.SIM_SCRIPT, Path("/env/sim.py").resolve())
            rs.configure_tools(None, None, "/flag/sim.py")
            self.assertEqual(rs.SIM_SCRIPT, Path("/flag/sim.py").resolve())


class MainTextPerf(FakeToolsCase):
    def test_perf_for_a_subset_of_models_keeps_the_other_rows_of_table_2(self):
        mt = load_module(SCRIPTS / "run_main_text.py", "run_main_text_perf")
        runner = mt.Runner(self.tmp / "main", str(self.psmc_rs), str(self.c_psmc), self.tmp / "unused_sim.py")
        runner.input_path("constant").write_text(">chr1\nTTTTKTTTTK\n")
        committed = ROOT / "experiment" / "runs" / "main_text" / "tables" / "table_2_runtime_memory.csv"
        table_2 = runner.tables / committed.name
        table_2.write_text(committed.read_text())
        before = read_csv(table_2)
        # The perf step needs psutil only for peak RSS; a stand-in is enough here.
        stub = {} if has_module("psutil") else {"psutil": type(sys)("psutil")}
        with mock.patch.object(mt, "PERF_REPEATS", 1), mock.patch.dict(sys.modules, stub):
            with contextlib.redirect_stdout(io.StringIO()):
                runner.perf_runs(["constant"])
        after = read_csv(table_2)
        self.assertEqual([(r["model"], r["tool"]) for r in after], [(r["model"], r["tool"]) for r in before])
        self.assertEqual([r for r in after if r["model"] != "constant"], [r for r in before if r["model"] != "constant"])
        new = [r for r in after if r["model"] == "constant"]
        self.assertTrue(all(is_finite_number(r["wall_sec_mean"]) for r in new))
        self.assertNotEqual(new, [r for r in before if r["model"] == "constant"])
        raw = read_csv(runner.tables / "perf_raw_repeats.csv")
        self.assertEqual(sorted((r["model"], r["tool"]) for r in raw), [("constant", "c"), ("constant", "rust")])


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

    @unittest.skipUnless(has_module("msprime"), "msprime not installed")
    def test_s5_records_the_bins_only_the_psmcfa_has(self):
        env = dict(self.env, SIM_LENGTH_BP="200000")
        res = self.run_steps("S5", env)
        self.assertEqual(res.returncode, 0, res.stderr[-3000:])
        rows = {r["model"]: r for r in read_csv(self.out / "tables" / "S5_format_consistency.csv")}
        self.assertEqual(sorted(rows), sorted(["constant", "bottleneck", "expansion", "zigzag"]))
        for model, r in rows.items():
            psmcfa = self.out / "s5_format" / "inputs" / f"{model}.200000bp.psmcfa"
            seq = "".join(ln for ln in psmcfa.read_text().splitlines() if not ln.startswith(">"))
            self.assertEqual(int(r["psmcfa_bins_after_last_het"]), len(seq) - 1 - seq.rindex("K"), model)

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
