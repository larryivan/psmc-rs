"""Tests for experiment E6: its jobs, its checks and (with the tools present) a tiny run end to end.

The analyses and the figures have their own tests (test_pipeline_e6_analysis.py, test_pipeline_e6_figures.py).
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, paths, runner, tools  # noqa: E402
from experiment.pipeline.experiments import REGISTRY, Context, e1, e6  # noqa: E402
from experiment.pipeline.runner import Job  # noqa: E402

FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})
HAS = {m: importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "scipy", "matplotlib", "msprime", "psutil")}
HAS_NUMERIC = HAS["numpy"] and HAS["pandas"] and HAS["scipy"]


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fake_ctx(profile: config.Profile, runs: Path) -> Context:
    return Context(profile, runs, _tools={"c": FAKE_C, "rs02": FAKE_RS}, log=lambda msg: None)


def fake_e3_jobs(ctx: Context, psmcfa: Path, n_iter: int = 2) -> list:
    """Stand-ins for E3's runs: psmc-rs 0.2 on one input with 1 thread (twice) and 2 threads, plus a qc run."""
    inference = replace(config.INFERENCE, n_iter=n_iter)
    out = []
    for t, k in ((1, 1), (1, 2), (2, 1)):
        js = Path(ctx.runs_dir) / "E3" / "fits" / f"t{t}-r{k}.json"
        out.append(Job(
            id=f"E3/fit/t{t}-r{k}",
            command=tools.rs02_fit(ctx.rs02(), inference, config.SIMULATION_SCALING, psmcfa, js, threads=t,
                                   psmc_out=js.with_suffix(".psmc")),
            inputs=[psmcfa], outputs=[js, js.with_suffix(".psmc")], cores=t, exclusive=True,
        ))
    qc = Path(ctx.runs_dir) / "E3" / "qc.json"
    out.append(Job(id="E3/qc", command=tools.rs02_qc(ctx.rs02(), psmcfa, qc), inputs=[psmcfa], outputs=[qc]))
    return out


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
        # Without E3 registered, so that S4 is left out whatever the tree registers.
        with mock.patch.dict(REGISTRY, {"E1": "e1", "E6": "e6"}, clear=True):
            self.jobs = e6.jobs(self.ctx)
        self.by_id = {j.id: j for j in self.jobs}
        self.graph = runner.Graph(self.e1_jobs + self.jobs, self.runs)

    def test_parts_and_their_dependencies_on_e1(self):
        ids = set(self.by_id)
        models = config.SMOKE.e1.models
        for part in ("s1", "s2", "s3", "s5"):
            self.assertIn(f"E6/{part}/report", ids)
        self.assertFalse(any(j.startswith("E6/s4") for j in ids))  # E3 is not registered in setUp
        self.assertEqual({f"E6/{p}/figure" for p in ("s1", "s2", "s3")}, {j for j in ids if j.endswith("/figure")})
        s1 = set(self.graph.deps["E6/s1/report"])
        self.assertEqual(s1, {j.id for j in self.e1_jobs if j.id.startswith(("E1/fit-", "E1/sim/"))})
        self.assertEqual(self.graph.deps["E6/s2/convert-vcf/bottleneck/rep01"], ["E1/sim/bottleneck/rep01"])
        self.assertEqual(sorted(self.graph.deps["E6/s3/decode/bottleneck/rep01"]),
                         ["E1/fit-rs/bottleneck/rep01", "E1/sim/bottleneck/rep01"])
        self.assertEqual(self.graph.deps["E6/s1/figure"], ["E6/s1/report"])
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

    def test_s4_reads_e3s_runs_through_their_records(self):
        psmcfa = self.runs / "E3" / "genome.psmcfa"
        with mock.patch.object(e6, "source_jobs", lambda ctx, exp: fake_e3_jobs(ctx, psmcfa)):
            jobs = {j.id: j for j in e6.jobs(self.ctx)}
        s4 = jobs["E6/s4/report"]
        self.assertIn("E6/s4/figure", jobs)
        records = [Path(s4.command[i + 1]) for i, a in enumerate(s4.command) if a == "--record"]
        self.assertEqual(len(records), 3)  # the three fits, not the qc run
        self.assertTrue(all(r.parent == self.runs / "E3" / "provenance" / "fit" for r in records))
        self.assertEqual(set(records) - set(s4.inputs), set())

    def test_production_plan(self):
        jobs = e6.jobs(fake_ctx(config.PRODUCTION, self.runs))
        self.assertEqual(len([j for j in jobs if j.id.startswith("E6/s2/fit-")]), 8)
        graph = runner.Graph(e1.jobs(fake_ctx(config.PRODUCTION, self.runs)) + jobs, self.runs)
        plan = runner.plan(graph, config.Budget())
        self.assertLess(plan["E6"]["cpu_hours"], 1.0)
        self.assertTrue(all(j.mem_gb <= config.Budget().mem_gb for j in jobs))


class CommandParsing(unittest.TestCase):
    def test_fits_are_recognised_and_other_commands_are_not(self):
        rs = Path("/r/target/release/psmc-rs")
        fit = tools.rs02_fit(tools.Tool("rs", rs), config.INFERENCE, config.SIMULATION_SCALING, Path("/in.psmcfa"),
                             Path("/out.json"), threads=8, psmc_out=Path("/out.psmc"))
        got = e6.parse_rs02_fit(["/usr/bin/time", "-l", *fit], str(rs))
        self.assertEqual(got, {"input": "/in.psmcfa", "output": "/out.json", "threads": 8, "psmc_out": "/out.psmc"})
        self.assertIsNone(e6.parse_rs02_fit(tools.rs02_qc(tools.Tool("rs", rs), Path("/in"), Path("/q.json")), str(rs)))
        self.assertIsNone(e6.parse_rs02_fit(["/other/psmc-rs", "/in", "/out.json", "25", "--threads", "1"], str(rs)))

    def test_portable_paths_round_trip(self):
        runs = Path("/tmp/somewhere/runs")
        text = f"{runs}/E3/fits/a.json"
        self.assertEqual(e6.local_path(paths.portable(text, runs), runs), text)

    def test_runs_are_grouped_by_input_with_several_thread_counts(self):
        mk = lambda jid, inp, t: e6.RsRun(jid, Path(inp), Path(f"/{jid}.json"), t, Path(f"/{jid}.rec"))  # noqa: E731
        groups = e6.scaling_groups([mk("a", "/x", 8), mk("b", "/x", 1), mk("c", "/y", 1), mk("d", "/y", 1)])
        self.assertEqual(list(groups), ["/x"])
        self.assertEqual([r.threads for r in groups["/x"]], [1, 8])


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
                *lay.figures("s3"), lay.root / "provenance" / "s3" / "report.json"]
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


def tools_present() -> bool:
    try:
        tools.c_psmc()
    except tools.ToolError:
        return False
    return (paths.ROOT / "target" / "release" / "psmc-rs").exists() and shutil.which("bgzip") is not None


@unittest.skipUnless(HAS_NUMERIC and HAS["matplotlib"] and HAS["msprime"] and HAS["psutil"] and tools_present(),
                     "needs numpy, pandas, scipy, matplotlib, msprime, psutil, C psmc, psmc-rs and bgzip")
class EndToEnd(unittest.TestCase):
    """The tiny profile through the runner: E1, stand-ins for E3's runs, and every part of E6."""

    def test_tiny_profile(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d).resolve()
            ctx = Context(config.TINY, runs, log=lambda msg: None)
            psmcfa = e1.Layout(runs / "E1").psmcfa("bottleneck", 1)
            with mock.patch.object(e6, "source_jobs", lambda c, exp: fake_e3_jobs(c, psmcfa)):
                graph = runner.Graph(e1.jobs(ctx) + fake_e3_jobs(ctx, psmcfa) + e6.jobs(ctx), runs)
                daemon = runner.Daemon(graph, runs, config.Budget(cores=4, mem_gb=8), runner.Logger(runs), {})
                self.assertEqual(daemon.run(), 0, (runs / "_runner" / "runner.log").read_text()[-3000:])
            lay = e6.Layout(runs / "E6")
            for part in e6.PARTS:
                for p in lay.tables(part):
                    self.assertTrue(p.exists(), p)
            for part in e6.FIGURES:
                for p in lay.figures(part):
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
            det = read_csv(lay.table("s4", "determinism"))
            self.assertEqual((det[0]["runs"], det[0]["json_identical"], det[0]["psmc_identical"]), ("3", "True", "True"))
            qc = read_csv(lay.table("s5", "qc"))
            self.assertIn("chr1 only", qc[-1]["label"])
            self.assertTrue(all("/Users/" not in p.read_text() for part in e6.PARTS for p in lay.tables(part)))


if __name__ == "__main__":
    unittest.main()
