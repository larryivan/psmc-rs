"""Tests for experiment E2 (real genomes): jobs, inputs, windows, bands, tables, Figure 3, and a tiny run."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, demography, metrics, paths, results, runner, tools  # noqa: E402
from experiment.pipeline.experiments import Context, available, e2  # noqa: E402

C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
FAKE_C = tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64})
FAKE_RS = tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40})
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
HAS_PSUTIL = importlib.util.find_spec("psutil") is not None


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fake_ctx(profile: config.Profile, runs: Path) -> Context:
    return Context(profile, runs, _tools={"c": FAKE_C, "rs02": FAKE_RS})


def genome(sid: str) -> config.RealGenome:
    return next(g for g in config.REAL_GENOMES if g.sample_id == sid)


class Config(unittest.TestCase):
    def test_e2_is_registered(self):
        self.assertIn("E2", available())

    def test_production_settings(self):
        cfg = config.PRODUCTION.e2
        self.assertIsNone(cfg.max_bins)
        self.assertEqual((cfg.bootstrap_replicates, cfg.bootstrap_trunk_bins), (100, 500_000))
        self.assertEqual(cfg.inference, config.INFERENCE)
        self.assertEqual(sorted(cfg.panel_order), sorted(g.sample_id for g in cfg.genomes))
        papuan = genome("Papuan")
        self.assertIn("chr1 only", papuan.label)
        self.assertEqual((papuan.scaling.mu, papuan.scaling.gen_years), (1.25e-8, 25.0))
        for sid in ("HLemySub1", "HLhydTec1", "HLpelCas1"):
            self.assertEqual((genome(sid).scaling.mu, genome(sid).scaling.gen_years), (7.9e-9, 15.0))

    def test_smoke_and_tiny_fit_prefixes(self):
        self.assertEqual(config.get_profile("smoke").e2.max_bins, 200_000)
        tiny = config.get_profile("tiny").e2
        self.assertEqual({g.sample_id for g in tiny.genomes}, {"Papuan", "HLhydTec1"})
        self.assertLess(tiny.bootstrap_trunk_bins, tiny.max_bins)
        self.assertEqual(config.get_profile("smoke").e1, config.SMOKE.e1)


class Jobs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name)
        self.jobs = e2.jobs(fake_ctx(config.PRODUCTION, self.runs))
        self.by_id = {j.id: j for j in self.jobs}

    def test_four_jobs_per_genome_then_report_and_figure(self):
        # Papuan chr1 forms 5 default trunks (< 20): it gets the second bootstrap, on 5 Mb trunks.
        self.assertEqual(len(self.jobs), 4 * 4 + 1 + 2)
        self.assertIn("E2/boot-rs/trunk50000/Papuan", self.by_id)
        self.assertNotIn("E2/boot-rs/trunk50000/HLhydTec1", self.by_id)  # 51 default trunks
        g = runner.Graph(self.jobs, self.runs)
        self.assertEqual(g.deps["E2/fit-c/Papuan"], [])  # reads the packaged file
        self.assertEqual(g.deps["E2/figure"], ["E2/report"])
        report = set(g.deps["E2/report"])
        self.assertEqual(report, {j.id for j in self.jobs if j.id.split("/")[1] in ("fit-c", "fit-rs", "boot-rs", "qc")})

    def test_tools_read_the_identical_input_with_identical_settings(self):
        c, rs = self.by_id["E2/fit-c/HLhydTec1"], self.by_id["E2/fit-rs/HLhydTec1"]
        self.assertEqual(c.inputs, [paths.REAL_DATA / "HLhydTec1_diploid.psmcfa"])
        self.assertEqual(c.inputs, rs.inputs)
        self.assertEqual(c.command[1:6], ["-N25", "-t15", "-r5", "-p", "4+25*2+4+6"])
        self.assertEqual(rs.command[3], "25")
        self.assertEqual(rs.command[rs.command.index("--mu") + 1], "7.9e-09")
        self.assertEqual(rs.command[rs.command.index("--gen-years") + 1], "15")
        self.assertEqual(c.settings["inference"], rs.settings["inference"])

    def test_bootstrap_job(self):
        boot = self.by_id["E2/boot-rs/trunk500000/Papuan"]
        cmd = boot.command
        seed = config.seed("E2", "Papuan", 0, "bootstrap")
        self.assertEqual(boot.seeds, {"bootstrap": seed})
        self.assertEqual(cmd[cmd.index("--bootstrap-seed") + 1], str(seed))
        self.assertEqual(cmd[cmd.index("--bootstrap") + 1], "100")
        self.assertEqual(cmd[cmd.index("--bootstrap-block-size") + 1], "500000")
        self.assertEqual(cmd[cmd.index("--threads") + 1], "7")
        self.assertEqual(boot.cores, 7)
        lay = e2.Layout(self.runs / "E2")
        # Replicates are not declared: psmc-rs may skip one, and the report reads those the summary names.
        self.assertEqual(boot.outputs, [lay.boot_main("Papuan", 500_000), lay.boot_summary("Papuan", 500_000),
                                        lay.boot_summary("Papuan", 500_000, "tsv")])
        self.assertNotEqual(seed, config.seed("E2", "HLhydTec1", 0, "bootstrap"))
        fine = self.by_id["E2/boot-rs/trunk50000/Papuan"]
        self.assertEqual(fine.command[fine.command.index("--bootstrap-block-size") + 1], "50000")
        self.assertEqual(fine.seeds, {"bootstrap": config.seed("E2", "Papuan", 0, "bootstrap/trunk50000")})
        self.assertEqual(e2.trunk_sizes(config.PRODUCTION.e2, [2_487_383]), [500_000, 50_000])
        self.assertEqual(e2.trunk_sizes(config.PRODUCTION.e2, [600_000] * 30), [500_000])

    def test_qc_job_checks_the_bootstrap_trunks(self):
        qc = self.by_id["E2/qc/HLpelCas1"]
        self.assertEqual(qc.command[1:3], ["qc", str(paths.REAL_DATA / "HLpelCas1_diploid.psmcfa")])
        self.assertEqual(qc.command[qc.command.index("--bootstrap-block-size") + 1], "500000")
        self.assertEqual(qc.stdout, self.runs / "E2" / "qc" / "HLpelCas1.qc.txt")

    def test_estimates_fit_the_budget(self):
        c = self.by_id["E2/fit-c/HLhydTec1"]  # longest sequence 3,152,835 bins
        self.assertAlmostEqual(c.mem_gb, 0.1 + 0.0105 * 315.2835, places=1)
        self.assertGreater(c.est_seconds, 30 * 60)
        budget = config.Budget()
        for j in self.jobs:
            self.assertIsNone(runner.fits_budget(j, budget), j.id)
        plan = runner.plan(runner.Graph(self.jobs, self.runs), budget)
        self.assertGreater(plan["E2"]["cpu_hours"], 20)
        self.assertLess(plan["E2"]["finish_hours"], 8)

    def test_smoke_profile_fits_prefixes_written_by_a_job(self):
        jobs = {j.id: j for j in e2.jobs(fake_ctx(config.SMOKE, self.runs))}
        subset = jobs["E2/subset/Papuan"]
        self.assertEqual(subset.command[-2:], ["--max-bins", "200000"])
        self.assertEqual(jobs["E2/fit-c/Papuan"].inputs, subset.outputs)
        self.assertEqual(jobs["E2/fit-c/Papuan"].mem_gb, round(0.1 + 0.0105 * 20, 2))


class Psmcfa(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.src = self.tmp / "in.psmcfa"
        seqs = [("a desc", "T" * 70 + "K" * 5), ("empty", ""), ("b", "NK" * 40), ("c", "T" * 10)]
        self.src.write_text("".join(f">{n}\n" + "".join(s[i:i + 50] + "\n" for i in range(0, len(s), 50)) for n, s in seqs))

    def test_lengths(self):
        self.assertEqual(e2.psmcfa_lengths(self.src), (75, 0, 80, 10))
        self.assertEqual(e2.prefix_lengths((75, 0, 80, 10), 100), (75, 25))
        self.assertEqual(e2.input_shape((75, 0, 80, 10)), e2.Shape(4, 165, 80))
        self.assertEqual(e2.input_shape((75, 0, 80, 10), 100), e2.Shape(2, 100, 75))
        with self.assertRaises(ValueError):
            e2.prefix_lengths((5,), 0)

    def test_prefix_is_laid_out_as_fq2psmcfa_writes(self):
        dst = self.tmp / "out" / "prefix.psmcfa"
        shape = e2.write_psmcfa_prefix(self.src, dst, 100)
        self.assertEqual(shape, e2.Shape(2, 100, 75))
        lines = dst.read_text().splitlines()
        self.assertEqual(lines[0], ">a desc")
        self.assertEqual(lines[1], "T" * 60)
        self.assertEqual(lines[2], "T" * 10 + "K" * 5)
        self.assertEqual(lines[3:], [">b", ("NK" * 40)[:25]])
        self.assertEqual(list(e2.read_psmcfa(dst)), [("a desc", "T" * 70 + "K" * 5), ("b", ("NK" * 40)[:25])])
        self.assertEqual(e2.psmcfa_lengths(dst), (75, 25))

    def test_command_line(self):
        dst = self.tmp / "cli.psmcfa"
        res = subprocess.run([sys.executable, "-m", e2.MODULE, "subset", str(self.src), str(dst), "--max-bins", "80"],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(e2.psmcfa_lengths(dst), (75, 5))


def flat_fit(theta: float, t_max: float = 15.0, lam: float = 1.0) -> results.Fit:
    return results.Fit(theta=theta, rho=theta / 5, t_max=t_max, n=63, lambdas=tuple([lam] * 64))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class WindowAndBand(unittest.TestCase):
    def test_window_of_a_constant_model(self):
        sc = config.Scaling(mu=1.25e-8, gen_years=25.0)
        fit = flat_fit(0.01, lam=2.0)
        n0 = fit.n0(sc.mu, 100)
        lo, hi = e2.fitted_window(fit, sc, 100, (0.01, 0.99))
        # Two lineages coalesce at rate 1 / (2 Ne) per generation, Ne = 2 N0.
        self.assertAlmostEqual(lo, -math.log(0.99) * 4 * n0 * 25, delta=1e-6 * lo)
        self.assertAlmostEqual(hi, -math.log(0.01) * 4 * n0 * 25, delta=1e-6 * hi)

    def test_epochs_follow_the_curve(self):
        fit = results.Fit(theta=0.02, rho=0.004, t_max=10.0, n=3, lambdas=(1.0, 0.5, 2.0, 4.0))
        eps = e2.fit_epochs(fit, 2.5e-8, 100)
        curve = fit.curve(2.5e-8, 1.0, 100)  # years = generations
        self.assertEqual([e.start for e in eps], list(curve.years))
        for t in (0.0, 5.0, curve.years[2] * 1.01, 1e9):
            self.assertAlmostEqual(demography.ne_at(eps, t), float(curve.at([t])[0]))
        self.assertTrue(math.isinf(eps[-1].end))

    def test_band_and_psmc_rs_summary(self):
        import numpy as np

        base = flat_fit(0.01).curve(1e-8, 25, 100)
        curves = [results.Curve(base.years, tuple(x * s for x in base.ne)) for s in (1.0, 2.0, 3.0, 4.0)]
        years = np.array([1e3, 1e5])
        lo, med, hi = e2.band(curves, years, (0.025, 0.5, 0.975))
        self.assertTrue(np.allclose(med, 2.5 * base.at(years)))
        self.assertTrue(np.allclose(lo, (1 + 0.075) * base.at(years)))
        xs = list(base.years[1:])
        q = e2.band(curves, xs, e2.SUMMARY_PROBS)
        summary = {"points": [{"x_years": x, "ne_main": float(base.at([x])[0]), "ne_q025": a, "ne_q500": b, "ne_q975": c}
                              for x, a, b, c in zip(xs, *q)]}
        self.assertEqual(e2.summary_difference(base, curves, summary), 0.0)
        summary["points"][3]["ne_q975"] *= 1.001
        self.assertAlmostEqual(e2.summary_difference(base, curves, summary), 0.001 / 1.001, places=9)


def write_stand_in_outputs(runs: Path, cfg: config.E2) -> e2.Layout:
    """Outputs of every E2 job for ``cfg``, made from C psmc's reference output (5 iterations).

    psmc-rs's fit is C's last round; replicate r scales its lambdas by
    1 + r/10; HLhydTec1's psmc-rs fit has lambdas twice C's.
    """
    lay = e2.Layout(runs / "E2")
    ref = results.parse_psmc(C_REF)
    last = ref.last()
    for g in cfg.genomes:
        sid = g.sample_id
        scale = 2.0 if sid == "HLhydTec1" else 1.0
        lay.subset(sid).parent.mkdir(parents=True, exist_ok=True)
        lay.subset(sid).write_text(">chr1\nTTKT\n")
        lay.c_psmc(sid).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(C_REF, lay.c_psmc(sid))
        shutil.copy(C_REF, lay.rs_psmc(sid))

        def params(lam_scale: float, extra: bool = True):
            p = {"theta": last.pa["theta"], "rho": last.pa["rho"], "t_max": last.pa["t_max"], "n_steps": 63,
                 "mu": g.scaling.mu, "pattern": "4+25*2+4+6", "lam": [x * lam_scale for x in last.pa["lambdas"]]}
            if extra:
                p.update(n_iter=5, loglike=[ref.loglikes()[i] for i in range(5)] + [-25000.0],
                         input={"sequences": 3, "bins": 150947, "missing_bins": 0, "het_bins": 7373},
                         qc={"warnings": ["TOO_SHORT"], "notes": []})
            return p

        main = json.dumps(params(scale), indent=2)
        lay.rs_json(sid).write_text(main)
        sc = g.scaling
        main_curve = results.fit_from_rs(params(scale)).curve(sc.mu, sc.gen_years, 100)
        for trunk in e2.trunk_sizes(cfg, e2.input_lengths(cfg, g)):
            lay.boot_dir(sid, trunk).mkdir(parents=True, exist_ok=True)
            lay.boot_main(sid, trunk).write_text(main)
            reps = []
            for r in range(1, cfg.bootstrap_replicates + 1):
                p = params(scale * (1 + r / 10), extra=False)
                lay.boot_replicate(sid, trunk, r).write_text(json.dumps(p))
                reps.append(results.fit_from_rs(p).curve(sc.mu, sc.gen_years, 100))
            xs = list(main_curve.years[1:])
            q = e2.band(reps, xs, e2.SUMMARY_PROBS)
            points = [{"x_years": x, "ne_main": float(main_curve.at([x])[0]), "ne_q025": float(a),
                       "ne_q500": float(b), "ne_q975": float(c)} for x, a, b, c in zip(xs, *q)]
            summary = {"n_replicates": cfg.bootstrap_replicates, "skipped_replicates": [],
                       "block_size": trunk, "seed": e2.bootstrap_seed(cfg, g, trunk), "points": points}
            lay.boot_summary(sid, trunk).write_text(json.dumps(summary))
            lay.boot_summary(sid, trunk, "tsv").write_text("x_years\n")
        qc = {"n_sequences": 1, "bin_size": 100, "bootstrap_trunk_bins": cfg.bootstrap_trunk_bins,
              "totals": {"bins": 4, "k_fraction": 0.25, "called_bases_from_bins": 400},
              "lengths": {"bootstrap_trunks": 12, "below_trunk": {"fraction": 0.0}, "n50_bases": 400},
              "saturation": {"multi_het_share": 0.01}, "longest_missing_run": None,
              "findings": [{"code": "TOO_SHORT", "severity": "warning", "message": "short, really",
                            "recommendation": "use more", "sequences": []},
                           {"code": "HET_OUTLIER", "severity": "info", "message": "m", "recommendation": "r",
                            "sequences": ["chr1"]}],
              "sequences": [{"name": "chr1"}]}
        lay.qc_json(sid).parent.mkdir(parents=True, exist_ok=True)
        lay.qc_json(sid).write_text(json.dumps(qc))
    return lay


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class Tables(unittest.TestCase):
    """The report on stand-in outputs (see :func:`write_stand_in_outputs`)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls._tmp.name)
        inference = replace(config.INFERENCE, n_iter=5, tr_ratio=4.0)
        cls.profile = replace(config.TINY, e2=replace(config.TINY.e2, inference=inference, bootstrap_replicates=3))
        cls.lay = write_stand_in_outputs(cls.runs, cls.profile.e2)
        cls.written = e2.report(Context(cls.profile, cls.runs))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_table_is_written(self):
        self.assertEqual(set(self.written), set(e2.TABLES))
        for key in e2.TABLES:
            self.assertTrue(read_csv(self.lay.table(key)), key)

    def test_tool_comparison(self):
        pairs = {r["sample_id"]: r for r in read_csv(self.lay.table("pairs"))}
        self.assertEqual(float(pairs["Papuan"]["rmse_log10_ne_rs_vs_c"]), 0.0)
        self.assertAlmostEqual(float(pairs["HLhydTec1"]["rmse_log10_ne_rs_vs_c"]), math.log10(2), places=12)
        self.assertEqual(float(pairs["Papuan"]["loglike_iteration0_rel_diff"]), 0.0)
        self.assertEqual(pairs["Papuan"]["bootstrap_main_identical"], "True")
        lo, hi = float(pairs["Papuan"]["window_lo_years"]), float(pairs["Papuan"]["window_hi_years"])
        fit = results.parse_psmc(C_REF).last().fit()
        self.assertEqual((lo, hi), e2.fitted_window(fit, genome("Papuan").scaling, 100, (0.01, 0.99)))
        self.assertLess(lo, hi)

    def test_bootstrap_summary(self):
        rows = read_csv(self.lay.table("bootstrap"))
        # 20,000 bins of each genome: 4 trunks of 5,000 bins (< 20), so a second bootstrap on 1,000-bin trunks.
        self.assertEqual({(r["sample_id"], r["trunk_bins"]) for r in rows},
                         {(s, t) for s in ("Papuan", "HLhydTec1") for t in ("5000", "1000")})
        fine = {r["sample_id"]: r for r in rows if r["default_trunks"] == "False"}
        self.assertEqual((fine["Papuan"]["trunks"], fine["Papuan"]["coarse_band"]), ("20", "False"))
        self.assertEqual(fine["Papuan"]["seed"], str(config.seed("E2", "Papuan", 0, "bootstrap/trunk1000")))
        boot = {r["sample_id"]: r for r in rows if r["default_trunks"] == "True"}
        self.assertEqual(boot["Papuan"]["replicates"], "3")
        self.assertEqual((boot["Papuan"]["trunks"], boot["Papuan"]["coarse_band"]), ("4", "True"))  # 4 < 20
        self.assertEqual(boot["Papuan"]["seed"], str(config.seed("E2", "Papuan", 0, "bootstrap")))
        self.assertLess(float(boot["Papuan"]["summary_max_rel_diff"]), 1e-12)
        # The replicates are 1.1 to 1.3 times the main curve, which lies below the band.
        self.assertEqual(float(boot["Papuan"]["rs_in_band_fraction"]), 0.0)
        width = math.log10(1.3 - 0.005) - math.log10(1.1 + 0.005)
        self.assertAlmostEqual(float(boot["Papuan"]["band_width_log10_median"]), width, places=9)
        reps = [r for r in read_csv(self.lay.table("replicates")) if r["sample_id"] == "Papuan" and r["trunk_bins"] == "5000"]
        self.assertEqual([r["replicate"] for r in reps], ["1", "2", "3"])
        self.assertEqual(len(reps[0]["lam"].split()), 28)

    def test_fit_summary_scaling_and_qc(self):
        rows = read_csv(self.lay.table("fit_summary"))
        self.assertEqual({(r["sample_id"], r["tool"]) for r in rows},
                         {(s, t) for s in ("Papuan", "HLhydTec1") for t in ("c", "rs")})
        papuan = next(r for r in rows if r["sample_id"] == "Papuan" and r["tool"] == "c")
        self.assertEqual((papuan["mu"], papuan["gen_years"]), ("1.25e-08", "25.0"))
        self.assertIn("chr1", papuan["label"])
        self.assertEqual(papuan["loglike_final"], "")  # C psmc never evaluates its final model
        qc = {r["sample_id"]: r for r in read_csv(self.lay.table("qc"))}
        self.assertEqual(qc["Papuan"]["warnings"], "TOO_SHORT")
        self.assertEqual(qc["Papuan"]["notes"], "HET_OUTLIER")
        self.assertEqual(qc["Papuan"]["scope"], "1 sequence (chr1 only)")
        findings = read_csv(self.lay.table("qc_findings"))
        self.assertEqual(len(findings), 4)

    def test_bands_and_curves(self):
        bands = [r for r in read_csv(self.lay.table("bands")) if r["sample_id"] == "Papuan"]
        self.assertEqual(len(bands), 2 * self.profile.e2.display_years[2])  # two trunk sizes
        self.assertTrue(any(r["in_window"] == "True" for r in bands))
        self.assertTrue(all(float(r["ne_band_lo"]) <= float(r["ne_band_hi"]) for r in bands))
        curves = read_csv(self.lay.table("curves"))
        self.assertEqual(len(curves), 2 * 2 * 64)

    @unittest.skipUnless(HAS_MPL, "matplotlib not installed")
    def test_figure_is_drawn_from_the_tables(self):
        from experiment.pipeline.figures import fig3_real_genomes as fig3

        panels = fig3.load_panels(self.lay.root / "tables")
        self.assertEqual(set(panels), {"Papuan", "HLhydTec1"})
        self.assertIn("chr1 only", fig3.title(panels["Papuan"]))
        self.assertIn("4 trunks of 0.5 Mb (few trunks: a coarse band)", fig3.subtitle(panels["Papuan"]))
        out = self.runs / "fig" / "fig3"
        written = fig3.render(panels, ["Papuan", "HLhydTec1"], out, ["pdf", "png"])
        self.assertEqual([p.name for p in written], ["fig3.pdf", "fig3.png"])
        self.assertTrue(all(p.stat().st_size > 1000 for p in written))
        first = written[0].read_bytes()
        fig3.render(panels, ["Papuan", "HLhydTec1"], out, ["pdf"])
        self.assertEqual(written[0].read_bytes(), first)  # no creation date: same bytes

    def test_step_curve_is_cut_to_the_range(self):
        from experiment.pipeline.figures import fig3_real_genomes as fig3

        xs, ys = fig3.step_xy([0.0, 10.0, 100.0, 1000.0], [1.0, 2.0, 3.0, 4.0], 50.0, 500.0)
        self.assertEqual((xs, ys), ([50.0, 100.0, 500.0], [2.0, 3.0, 3.0]))


def in_git_work_tree() -> bool:
    if shutil.which("git") is None:
        return False
    res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == "true"


def ignored(rel_path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", rel_path], cwd=ROOT).returncode == 0


@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    def test_small_results_are_tracked(self):
        for p in (
            "experiment/runs/E2/fits/HLemySub1.c.psmc",
            "experiment/runs/E2/fits/HLemySub1.rs.json",
            "experiment/runs/E2/fits/HLemySub1.rs.psmc",
            "experiment/runs/E2/qc/HLemySub1.qc.json",
            "experiment/runs/E2/qc/HLemySub1.qc.txt",
            "experiment/runs/E2/provenance/boot-rs/trunk500000/HLemySub1.json",
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/main.json",
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/replicate_001.json",
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/summary.json",
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/summary.tsv",
            "experiment/runs/E2/provenance/fit-c/HLemySub1.json",
            "experiment/runs/E2/provenance/report.json",
            "experiment/runs/E2/tables/E2_bootstrap_replicates.csv",
            "experiment/runs/E2/figures/E2_fig3_real_genomes.pdf",
            "experiment/runs/E2/figures/E2_fig3_real_genomes.png",
        ):
            self.assertFalse(ignored(p), p)

    def test_replicate_rounds_and_local_files_are_ignored(self):
        for p in (
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/replicate_001.psmc",
            "experiment/runs/E2/bootstrap/HLemySub1/trunk500000/replicate_001.html",
            "experiment/runs/E2/inputs/HLemySub1.psmcfa",
            "experiment/runs/E2/fits/HLemySub1.rs.html",
            "experiment/runs/E2/logs/jobs/boot-rs/HLemySub1.log",
        ):
            self.assertTrue(ignored(p), p)


def tools_present() -> bool:
    try:
        tools.c_psmc()
    except tools.ToolError:
        return False
    return tools.rs02_binary().exists()


@unittest.skipUnless(HAS_NUMPY and HAS_MPL and HAS_PSUTIL and tools_present(), "needs numpy, matplotlib, psutil, C psmc and psmc-rs")
class EndToEnd(unittest.TestCase):
    """The tiny profile through the runner: prefixes, both tools, bootstrap, qc, tables and Figure 3."""

    def test_tiny_profile(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            graph = runner.Graph(e2.jobs(Context(config.TINY, runs)), runs)
            daemon = runner.Daemon(graph, runs, config.Budget(cores=3, mem_gb=4), runner.Logger(runs), {})
            self.assertEqual(daemon.run(), 0)
            lay = e2.Layout(runs / "E2")
            pairs = {r["sample_id"]: r for r in read_csv(lay.table("pairs"))}
            self.assertEqual(set(pairs), {"Papuan", "HLhydTec1"})
            for sid, row in pairs.items():
                # The first iteration's log-likelihood equals C psmc's to its printed precision.
                self.assertLess(float(row["loglike_iteration0_rel_diff"]), 1e-8, sid)
                self.assertEqual(row["bootstrap_main_identical"], "True", sid)
            boot = {r["sample_id"]: r for r in read_csv(lay.table("bootstrap"))}
            for sid, row in boot.items():
                self.assertLess(float(row["summary_max_rel_diff"]), 1e-9, sid)
                self.assertEqual(row["main_identical_to_fit"], "True", sid)
            trunks = {(r["sample_id"], r["trunk_bins"]): int(r["trunks"]) for r in read_csv(lay.table("bootstrap"))}
            self.assertEqual(trunks[("Papuan", "5000")], 4)  # 20,000 bins in trunks of 5,000
            self.assertEqual(trunks[("Papuan", "1000")], 20)
            self.assertTrue(all(lay.figure(ext).exists() for ext in e2.FIGURE_FORMATS))
            again = runner.job_statuses(graph, runs, daemon_pid=None)
            self.assertTrue(all(s.state == runner.DONE for s in again))


if __name__ == "__main__":
    unittest.main()
