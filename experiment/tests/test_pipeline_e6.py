"""Tests for experiment E6: its jobs, its checks and (with the tools present) a tiny run end to end.

The analyses have their own tests (test_pipeline_e6_analysis.py); the figures are FIG's (test_figures_*.py).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, paths, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e1, e6  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fake_ctx(profile: config.Profile, runs: Path) -> Context:
    return Context(profile, runs, _tools={"c": FAKE_C, "rs02": FAKE_RS}, log=lambda msg: None)


# ---------------------------------------------------------------------------
# Jobs (no numerical packages needed)
# ---------------------------------------------------------------------------


class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name).resolve()
        self.ctx = fake_ctx(config.SMOKE, self.runs)
        self.e1_jobs = e1.jobs(self.ctx)
        self.jobs = e6.jobs(self.ctx)
        self.by_id = {j.id: j for j in self.jobs}
        self.graph = runner.Graph(self.e1_jobs + self.jobs, self.runs)

    def test_parts_and_their_dependencies_on_e1(self):
        ids = set(self.by_id)
        models = config.SMOKE.e1.models
        for part in ("s1", "s2", "s3", "s5"):
            self.assertIn(f"E6/{part}/report", ids)
        self.assertFalse(any(j.startswith("E6/s4") for j in ids))  # S4 is E3's own tables
        self.assertFalse(any(j.endswith("/figure") for j in ids))  # the figures are FIG's
        s1 = set(self.graph.deps["E6/s1/report"])
        self.assertEqual(s1, {j.id for j in self.e1_jobs if j.id.startswith(("E1/fit-c/", "E1/fit-rs/", "E1/sim/"))})
        self.assertEqual(self.graph.deps["E6/s2/convert-vcf/bottleneck/rep01"], ["E1/sim/bottleneck/rep01"])
        self.assertEqual(sorted(self.graph.deps["E6/s3/decode/bottleneck/rep01"]),
                         ["E1/fit-rs/bottleneck/rep01", "E1/sim/bottleneck/rep01"])
        self.assertEqual(len([j for j in ids if j.startswith("E6/s2/fit-")]), 2 * len(models))

    def test_s2_reads_the_vcf_in_mask_mode_and_fits_with_e1s_settings(self):
        conv = self.by_id["E6/s2/convert-vcf/constant/rep01"]
        self.assertEqual(conv.command[1], "convert")
        self.assertIn("--multihetsep", conv.command)
        self.assertEqual(conv.command[conv.command.index("--vcf-mode") + 1], "mask")
        fit = self.by_id["E6/s2/fit-multihetsep/constant/rep01"]
        e1_fit = self.graph.jobs["E1/fit-rs/constant/rep01"]
        # E1's options after INPUT OUTPUT N_ITER, up to --psmc-out FILE; then the input format.
        self.assertEqual(fit.command[3:-3], e1_fit.command[3:-1])
        self.assertEqual(fit.command[-2:], ["--input-format", "mhs"])
        self.assertEqual(fit.settings["inference"], e1_fit.settings["inference"])

    def test_s3_decodes_e1s_fit_without_iterating(self):
        dec = self.by_id["E6/s3/decode/bottleneck/rep01"]
        e1lay = e1.Layout(self.runs / "E1")
        self.assertEqual(dec.command[3], "0")
        self.assertEqual(dec.command[dec.command.index("--init-params") + 1], str(e1lay.rs_json("bottleneck", 1)))
        self.assertNotIn("--t-max", dec.command)  # psmc-rs refuses it with --init-params
        self.assertIn("--tmrca-out", dec.command)
        truth = self.by_id["E6/s3/truth/bottleneck/rep01"]
        self.assertIn("--tmrca-only", truth.command)
        self.assertEqual(truth.seeds["chr1"]["ancestry"], config.seed("E1", "bottleneck", 1, "ancestry/chr1"))

    def test_s5_checks_e2s_genomes_with_e2s_trunk(self):
        qc = [j for j in self.jobs if j.id.startswith("E6/s5/qc/")]
        self.assertEqual([j.id.rsplit("/", 1)[1] for j in qc], [g.sample_id for g in config.SMOKE.e2.genomes])
        cmd = qc[0].command
        self.assertEqual(cmd[1], "qc")
        self.assertEqual(cmd[cmd.index("--bootstrap-block-size") + 1], str(config.SMOKE.e2.bootstrap_trunk_bins))
        self.assertEqual(qc[0].inputs, [paths.REAL_DATA / config.SMOKE.e2.genomes[0].file])

    def test_production_plan(self):
        jobs = e6.jobs(fake_ctx(config.PRODUCTION, self.runs))
        self.assertEqual(len([j for j in jobs if j.id.startswith("E6/s2/fit-")]), 8)
        graph = runner.Graph(e1.jobs(fake_ctx(config.PRODUCTION, self.runs)) + jobs, self.runs)
        plan = runner.plan(graph, config.Budget())
        self.assertLess(plan["E6"]["cpu_hours"], 1.0)
        self.assertTrue(all(j.mem_gb <= config.Budget().mem_gb for j in jobs))


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
    """What git keeps of experiment/runs/E6: results and records, not the regenerable inputs."""

    def test_layout_paths_fall_on_the_right_side_of_the_ignore_rules(self):
        lay = e6.Layout(ROOT / "experiment" / "runs" / "E6")
        rel = lambda p: Path(p).relative_to(ROOT).as_posix()  # noqa: E731
        kept = [lay.s2_fit("constant", 1, "vcf"), lay.s2_fit_psmc("constant", 1, "multihetsep"),
                lay.s3_decode("bottleneck", 1), lay.s5_qc("Papuan"), lay.table("s3", "track"),
                lay.root / "figures" / "figS3_tmrca_decoding.pdf", lay.root / "provenance" / "s3" / "report.json"]
        dropped = [lay.s2_vcf_psmcfa("constant", 1), lay.s2_multihetsep("constant", 1), lay.s2_mhs_psmcfa("zigzag", 1),
                   lay.s3_true_track("bottleneck", 1), lay.s3_posterior("bottleneck", 1),
                   lay.s2_fit("constant", 1, "vcf").with_suffix(".html"), lay.root / "logs" / "runner.log"]
        for p in kept:
            self.assertFalse(ignored(rel(p)), rel(p))
        for p in dropped:
            self.assertTrue(ignored(rel(p)), rel(p))


class Checks(unittest.TestCase):
    def test_s2_problems_flag_a_vcf_that_differs(self):
        ok_bins = [{"model": "m", "format": "vcf", "sequence": "all", "identical": True, "bins_differing": 0,
                    "bins_format": 5, "bins_psmcfa": 5},
                   {"model": "m", "format": "multihetsep", "sequence": "all", "identical": False, "bins_differing": 0,
                    "bins_extra_at_end": 0, "tail_is_after_last_het": True}]
        ok_fits = [{"model": "m", "format": "vcf", "parameters_identical": True, "loglikes_identical": True}]
        self.assertEqual(e6.s2_problems(ok_bins, ok_fits), [])
        bad = [dict(ok_bins[0], identical=False, bins_differing=3), ok_bins[1]]
        self.assertEqual(len(e6.s2_problems(bad, [dict(ok_fits[0], parameters_identical=False)])), 2)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E6 in the tiny runs of every experiment (tiny_runs): every part's tables and checks."""

    def test_tiny_profile(self):
        runs = tiny_runs.runs_dir()
        lay = e6.Layout(runs / "E6")
        for part in e6.PARTS:
            for p in lay.tables(part):
                self.assertTrue(p.exists(), p)
        bins = {(r["format"], r["sequence"]): r for r in read_csv(lay.table("s2", "bins"))}
        self.assertEqual(bins[("vcf", "all")]["identical"], "True")
        self.assertEqual(bins[("multihetsep", "all")]["tail_is_after_last_het"], "True")
        fits = {r["format"]: r for r in read_csv(lay.table("s2", "fits"))}
        self.assertEqual(fits["vcf"]["parameters_identical"], "True")
        s3 = {r["bins"]: r for r in read_csv(lay.table("s3", "summary"))}
        self.assertEqual(s3["all"]["state_index_mismatches"], "0")
        self.assertEqual(s3["all"]["decoded_model_is_e1_fit"], "True")
        self.assertGreater(float(s3["all"]["pearson_log10_mean"]), 0.3)
        qc = read_csv(lay.table("s5", "qc"))
        self.assertIn("chr1 only", qc[-1]["label"])
        self.assertTrue(all("/Users/" not in p.read_text() for part in e6.PARTS for p in lay.tables(part)))


if __name__ == "__main__":
    unittest.main()
