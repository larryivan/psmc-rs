"""Tests for experiment E5 (bootstrap calibration): its jobs, its statistics, its tables, and a tiny run end to end."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import C_REF, assert_ignore_rules, fake_ctx, has_modules, in_git_work_tree, option, read_csv  # noqa: E402

from experiment.pipeline import config, demography, metrics, results, runner  # noqa: E402
from experiment.pipeline.experiments import Context, e5  # noqa: E402

HAS_SCIPY = has_modules("numpy", "scipy")


class Jobs(unittest.TestCase):
    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = config.SMOKE.e5
        self.jobs = e5.jobs(fake_ctx(config.SMOKE, self.runs))
        self.by_id = {j.id: j for j in self.jobs}
        self.graph = runner.Graph(self.jobs, self.runs)

    def test_job_counts_and_dependencies(self):
        cfg = self.cfg
        boots = cfg.bootstrap_datasets * len(cfg.trunk_bins)
        self.assertEqual(len(self.jobs), 2 * cfg.truth_replicates + boots + 1)
        g = self.graph
        self.assertEqual(g.deps["E5/fit/bottleneck/rep05"], ["E5/sim/bottleneck/rep05"])
        self.assertEqual(g.deps["E5/boot-trunk2000/bottleneck/rep03"], ["E5/sim/bottleneck/rep03"])
        self.assertNotIn("E5/boot-trunk2000/bottleneck/rep04", self.by_id)  # only the first genomes
        report = set(g.deps["E5/report"])
        expected = {j.id for j in self.jobs if j.id.startswith(("E5/sim/", "E5/fit/", "E5/boot-"))}
        self.assertEqual(report, expected)

    def test_bootstrap_command(self):
        job = self.by_id["E5/boot-trunk5000/bottleneck/rep02"]
        cmd = job.command
        self.assertEqual(cmd[3], str(self.cfg.inference.n_iter))
        self.assertEqual(option(cmd, "--bootstrap"), str(self.cfg.bootstrap_replicates))
        self.assertEqual(option(cmd, "--bootstrap-block-size"), "5000")
        seed = config.seed("E5", "bottleneck", 2, "bootstrap/trunk5000")
        self.assertEqual(option(cmd, "--bootstrap-seed"), str(seed))
        self.assertEqual(job.seeds, {"bootstrap": seed})
        lay = e5.Layout(self.runs / "E5", "bottleneck")
        self.assertEqual(option(cmd, "--bootstrap-dir"), str(lay.boot_dir(2, 5000)))
        self.assertEqual(job.outputs, [lay.boot_json(2, 5000), lay.boot_summary(2, 5000), lay.boot_summary_tsv(2, 5000)])
        self.assertEqual(option(cmd, "--threads"), str(self.cfg.bootstrap_threads))

    def test_fits_and_bootstraps_share_the_inference_settings(self):
        fit = self.by_id["E5/fit/bottleneck/rep01"]
        boot = self.by_id["E5/boot-trunk2000/bottleneck/rep01"]
        self.assertEqual(fit.inputs, boot.inputs)
        end = fit.command.index("--threads")
        self.assertEqual(fit.command[3:end], boot.command[3:end])
        self.assertEqual(fit.settings["inference"], boot.settings["inference"])
        self.assertNotIn("--psmc-out", fit.command)

    def test_simulation_records_its_seeds_and_writes_no_vcf(self):
        sim = self.by_id["E5/sim/bottleneck/rep02"]
        self.assertIn("--no-vcf", sim.command)
        self.assertEqual(option(sim.command, "--experiment"), "E5")
        self.assertEqual(sim.seeds["chr2"]["mutations"], config.seed("E5", "bottleneck", 2, "mutations/chr2"))
        self.assertEqual(len(sim.outputs), 2)

    def test_seeds_are_distinct(self):
        seeds = [j.seeds["bootstrap"] for j in self.jobs if j.id.startswith("E5/boot-")]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_production(self):
        jobs = e5.jobs(fake_ctx(config.PRODUCTION, self.runs))
        cfg = config.PRODUCTION.e5
        self.assertEqual(len(jobs), 50 * 2 + 10 * 2 + 1)
        boot = next(j for j in jobs if j.id == "E5/boot-trunk50000/bottleneck/rep10")
        one_fit = config.Costs().rs02_seconds(300, cfg.inference)
        self.assertAlmostEqual(boot.est_seconds, one_fit * (1 + cfg.bootstrap_replicates))
        self.assertLessEqual(boot.mem_gb, 1.0)
        plan = runner.plan(runner.Graph(jobs, self.runs), config.Budget())
        self.assertGreater(plan["E5"]["cpu_hours"], 10.0)
        # The E-steps of the bootstrap fits; the converged M-steps take ms.
        self.assertLess(plan["E5"]["wall_hours_alone"], 4.0)


class TrunkCount(unittest.TestCase):
    def test_splitfa_rule(self):
        self.assertEqual(e5.trunk_count([500_000] * 6, 500_000), 6)
        self.assertEqual(e5.trunk_count([500_000] * 6, 50_000), 60)
        self.assertEqual(e5.trunk_count([749_999], 500_000), 1)  # a remainder < 1.5 trunks stays
        self.assertEqual(e5.trunk_count([750_000], 500_000), 2)
        self.assertEqual(e5.trunk_count([20_000, 20_000], 5_000), 8)
        self.assertEqual(e5.trunk_count([3_000], 1_000), 3)
        with self.assertRaises(ValueError):
            e5.trunk_count([10], 0)


@unittest.skipUnless(HAS_SCIPY, "numpy and scipy not installed")
class Statistics(unittest.TestCase):
    def test_quantile_is_psmc_rs_linear_rule(self):
        # psmc-rs: pos = q (n - 1), linear between the order statistics around it.
        vals = [4.0, 1.0, 3.0, 2.0, 10.0]
        self.assertAlmostEqual(float(e5.quantile(vals, 0.975)), 4.0 + 0.9 * 6.0)
        self.assertAlmostEqual(float(e5.quantile(vals, 0.025)), 1.0 + 0.1 * 1.0)

    def test_intervals(self):
        lo, hi = e5.clopper_pearson([0, 10, 5], 10)
        self.assertAlmostEqual(float(lo[0]), 0.0)
        self.assertAlmostEqual(float(hi[0]), 1 - 0.025 ** (1 / 10), places=12)
        self.assertAlmostEqual(float(lo[1]), 0.025 ** (1 / 10), places=12)
        self.assertAlmostEqual(float(hi[1]), 1.0)
        self.assertLess(float(lo[2]), 0.5)
        self.assertGreater(float(hi[2]), 0.5)
        lo, hi = e5.sd_interval(1.0, 50)
        self.assertAlmostEqual(float(lo), 0.835, places=3)
        self.assertAlmostEqual(float(hi), 1.246, places=3)

    def test_ratio_and_coverage(self):
        import numpy as np

        truth = np.array([100.0, 100.0, 100.0])
        offsets = np.array([-0.1, 0.0, 0.1, 0.2])
        fits = 100.0 * 10.0 ** offsets[:, None] * np.ones((1, 3))
        sampling = e5.sampling_stats(fits, truth, (0.025, 0.975))
        self.assertAlmostEqual(float(sampling["sd_log10"][0]), float(np.std(offsets, ddof=1)))
        self.assertAlmostEqual(float(sampling["bias_log10"][0]), 0.05)
        # Bootstrap 1 spreads twice as wide around the truth; bootstrap 2 is shifted away from it.
        b1 = e5.band_stats(100.0 * 10.0 ** (2 * offsets)[:, None] * np.ones((1, 3)), (0.025, 0.975))
        b2 = e5.band_stats(1000.0 * 10.0 ** offsets[:, None] * np.ones((1, 3)), (0.025, 0.975))
        cal = e5.calibration([b1, b2], sampling, truth)
        np.testing.assert_allclose(cal["ratio"][0], 2.0)
        np.testing.assert_allclose(cal["ratio"][1], 1.0)
        np.testing.assert_allclose(cal["ratio_median"], 1.5)
        np.testing.assert_array_equal(cal["covered"], [1, 1, 1])
        np.testing.assert_allclose(cal["coverage"], 0.5)
        np.testing.assert_allclose(cal["coverage_mean"], 0.5)  # the fits' geometric mean, 10**2.05, is in b1's band only
        self.assertNotIn("coverage_projected", cal)
        # Each dataset's own projected truth: the truth for b1 (in its band), 1000 for b2 (in its band too).
        cal = e5.calibration([b1, b2], sampling, truth, np.array([truth, [1000.0] * 3]))
        np.testing.assert_array_equal(cal["covered_projected"], [2, 2, 2])
        np.testing.assert_allclose(cal["coverage_projected"], 1.0)
        np.testing.assert_allclose(cal["coverage"], 0.5)
        res = e5.dataset_resampling(cal, 200, seed=1)
        self.assertEqual((res["coverage_projected_mean_ci_lo"], res["coverage_projected_mean_ci_hi"]), (1.0, 1.0))

    def test_dataset_resampling_intervals(self):
        import numpy as np

        # Four datasets covering 100%, 100%, 50% and 0% of three points; SD ratios 1, 1, 2, 2.
        covers = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=bool)
        ratio = np.array([[1.0] * 3, [1.0] * 3, [2.0] * 3, [2.0] * 3])
        res = e5.dataset_resampling({"covers": covers, "ratio": ratio}, 4000, seed=1)
        self.assertAlmostEqual(res["coverage_dataset_min"], 0.0)
        self.assertAlmostEqual(res["coverage_dataset_max"], 1.0)
        self.assertLess(res["coverage_mean_ci_lo"], covers.mean())
        self.assertGreater(res["coverage_mean_ci_hi"], covers.mean())
        self.assertGreaterEqual(res["coverage_mean_ci_lo"], 0.0)
        self.assertLessEqual(res["coverage_mean_ci_hi"], 1.0)
        self.assertEqual((res["sd_ratio_median_ci_lo"], res["sd_ratio_median_ci_hi"]), (1.0, 2.0))
        again = e5.dataset_resampling({"covers": covers, "ratio": ratio}, 4000, seed=1)
        self.assertEqual(res, again)  # seeded


# ---------------------------------------------------------------------------
# The report on synthetic outputs with a known answer
# ---------------------------------------------------------------------------

FIT_OFFSETS = (-0.1, 0.0, 0.1, 0.2)  # log10 Ne of the genomes' fits, around a base curve
BOOT_OFFSETS = (-0.2, 0.0, 0.2, 0.4)  # of the bootstrap replicates, around their genome's fit: twice the SD
BASE = results.parse_psmc(C_REF).last().pa
BINS, HET_BINS = 3000, 150


def params(offset: float, n_iter: int, theta: float = None) -> dict:
    """A psmc-rs JSON whose curve is the base curve shifted by ``offset`` in log10 Ne (same times)."""
    return {
        "theta": BASE["theta"] if theta is None else theta, "rho": BASE["rho"], "t_max": BASE["t_max"],
        "n_steps": 63, "pattern": "4+25*2+4+6", "lam": [x * 10.0**offset for x in BASE["lambdas"]],
        "n_iter": n_iter, "loglike": [-1000.0 - i for i in range(n_iter + 1)],
        "input": {"bins": BINS, "het_bins": HET_BINS},
    }


def psmc_rs_summary(main: dict, replicates, cfg: config.E5, trunk: int, seed: int, skipped=()) -> dict:
    """summary.json as psmc-rs writes it: the replicates' quantiles at the main curve's positive times."""
    import numpy as np

    sc = cfg.scaling
    main_curve = results.fit_from_rs(main).curve(sc.mu, sc.gen_years, cfg.inference.bin_size)
    curves = [results.fit_from_rs(p).curve(sc.mu, sc.gen_years, cfg.inference.bin_size) for p in replicates]
    points = []
    for x, ne in zip(main_curve.years[1:], main_curve.ne[1:]):
        vals = sorted(float(c.at([x])[0]) for c in curves)
        q = {k: float(np.quantile(vals, p)) for k, p in (("ne_q025", 0.025), ("ne_q500", 0.5), ("ne_q975", 0.975))}
        points.append({"x_years": x, "ne_main": ne, **q})
    return {"n_replicates": len(replicates), "skipped_replicates": list(skipped), "block_size": trunk, "seed": seed,
            "points": points}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def synthetic_profile() -> config.Profile:
    e5_cfg = replace(
        config.TINY.e5, truth_replicates=4, bootstrap_datasets=2, bootstrap_replicates=4, trunk_bins=(1000, 500),
        inference=replace(config.INFERENCE, n_iter=5),
    )
    return replace(config.TINY, e5=e5_cfg)


def write_synthetic_runs(runs: Path, profile: config.Profile) -> e5.Layout:
    cfg = profile.e5
    lay = e5.Layout(runs / "E5", cfg.model)
    n = cfg.inference.n_iter
    truth = {
        "chromosomes": [{"name": "chr1", "bins": BINS, "het_bins": HET_BINS, "het_sites": 160,
                         "seeds": {"ancestry": 11, "mutations": 12}}],
        "totals": {"het_sites": 160},
        "truth": demography.truth_record(config.BOTTLENECK),
    }
    for r, off in enumerate(FIT_OFFSETS, start=1):
        write_json(lay.truth(r), truth)
        write_json(lay.fit_json(r), params(off, n))
    for d in range(1, cfg.bootstrap_datasets + 1):
        for trunk in cfg.trunk_bins:
            main = params(FIT_OFFSETS[d - 1], n)
            if d == 2 and trunk == 500:
                main["theta"] *= 1.0 + 1e-12  # a main fit that differs from the genome's own fit
            write_json(lay.boot_json(d, trunk), main)
            reps = [params(FIT_OFFSETS[d - 1] + b, n) for b in BOOT_OFFSETS]
            for i, p in enumerate(reps, start=1):
                write_json(lay.replicate_json(d, trunk, i), {k: p[k] for k in ("theta", "rho", "lam", "t_max", "n_steps", "pattern")})
            summary = psmc_rs_summary(main, reps, cfg, trunk, e5.bootstrap_seed(cfg, d, trunk))
            write_json(lay.boot_summary(d, trunk), summary)
    return lay


@unittest.skipUnless(HAS_SCIPY, "numpy and scipy not installed")
class Tables(unittest.TestCase):
    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.profile = synthetic_profile()
        self.cfg = self.profile.e5
        self.lay = write_synthetic_runs(self.runs, self.profile)
        self.written = e5.report(Context(self.profile, self.runs, log=lambda msg: None))

    def test_every_table_is_written(self):
        self.assertEqual(set(self.written), set(e5.TABLES))
        n = self.cfg.grid.points
        self.assertEqual(len(read_csv(self.lay.table("sampling"))), n)
        self.assertEqual(len(read_csv(self.lay.table("sd_ratio"))), 2 * n)
        self.assertEqual(len(read_csv(self.lay.table("bands"))), 2 * 2 * n)
        self.assertEqual(len(read_csv(self.lay.table("summary"))), 2 * len(e5.WINDOWS))
        self.assertEqual(len(read_csv(self.lay.table("fits"))), 4)
        seeds = read_csv(self.lay.table("seeds"))
        self.assertEqual(len(seeds), 4 * 2 + 2 * 2 + len(e5.WINDOWS) * 2)  # genomes, bootstraps, summaries
        boot_seed = [s for s in seeds if s["replicate"] == "2" and s["stage"] == "bootstrap/trunk500"]
        self.assertEqual(int(boot_seed[0]["seed"]), config.seed("E5", "bottleneck", 2, "bootstrap/trunk500"))

    def test_sd_ratio_is_the_known_factor(self):
        rows = read_csv(self.lay.table("sd_ratio"))
        sd = float(__import__("statistics").stdev(FIT_OFFSETS))
        for r in rows:
            self.assertAlmostEqual(float(r["ratio_median"]), 2.0, places=9)
            self.assertAlmostEqual(float(r["sd_true_log10"]), sd, places=9)
        summary = read_csv(self.lay.table("summary"))
        for r in summary:
            self.assertAlmostEqual(float(r["sd_ratio_median"]), 2.0, places=9)
            self.assertEqual(float(r["share_points_sd_ratio_below_1"]), 0.0)

    def test_coverage_against_an_independent_computation(self):
        import numpy as np

        cfg = self.cfg
        epochs = demography.epochs_from_record(demography.truth_record(config.BOTTLENECK))
        grid = metrics.model_grid(epochs, cfg.grid, cfg.scaling.gen_years)
        truth = metrics.truth_on_grid(epochs, grid, cfg.scaling.gen_years)
        base = results.fit_from_rs(params(0.0, 5)).curve(cfg.scaling.mu, cfg.scaling.gen_years, 100).at(grid)
        q_lo, q_hi = (np.quantile(10.0 ** np.array(BOOT_OFFSETS), q) for q in cfg.band)
        covered = sum(
            (base * 10.0 ** FIT_OFFSETS[d] * q_lo <= truth) & (truth <= base * 10.0 ** FIT_OFFSETS[d] * q_hi)
            for d in range(cfg.bootstrap_datasets)
        )
        rows = [r for r in read_csv(self.lay.table("coverage")) if r["trunk_bins"] == "1000"]
        self.assertEqual([int(r["covered"]) for r in rows], covered.tolist())
        self.assertTrue(0 < sum(covered) < 2 * len(grid))  # the check is not vacuous
        bands = [r for r in read_csv(self.lay.table("bands")) if r["trunk_bins"] == "1000" and r["dataset"] == "1"]
        np.testing.assert_allclose([float(r["boot_lo_ne"]) for r in bands], base * 10.0 ** FIT_OFFSETS[0] * q_lo)
        # Coverage of each dataset's pattern-projected truth (its own fit's spans).
        sc = cfg.scaling
        projected = [metrics.projected_truth(results.fit_from_rs(params(FIT_OFFSETS[d], 5)), epochs, sc.mu,
                                             sc.gen_years, 100).at(grid) for d in range(cfg.bootstrap_datasets)]
        covered_proj = sum(
            (base * 10.0 ** FIT_OFFSETS[d] * q_lo <= projected[d]) & (projected[d] <= base * 10.0 ** FIT_OFFSETS[d] * q_hi)
            for d in range(cfg.bootstrap_datasets)
        )
        self.assertEqual([int(r["covered_projected"]) for r in rows], covered_proj.tolist())
        np.testing.assert_allclose([float(r["projected_ne"]) for r in bands], projected[0])
        summary = next(r for r in read_csv(self.lay.table("summary")) if r["trunk_bins"] == "1000"
                       and r["window"] == "informative")
        self.assertAlmostEqual(float(summary["coverage_projected_mean"]), covered_proj.mean() / cfg.bootstrap_datasets)
        fits = read_csv(self.lay.table("fits"))
        self.assertTrue(all(float(r["rmse_log10_ne_vs_projected_truth"]) > 0 for r in fits))

    def test_dataset_checks(self):
        rows = {(r["dataset"], r["trunk_bins"]): r for r in read_csv(self.lay.table("datasets"))}
        self.assertEqual(rows[("1", "1000")]["main_fit_equals_fit"], "True")
        self.assertEqual(rows[("2", "500")]["main_fit_equals_fit"], "False")
        self.assertGreater(float(rows[("2", "500")]["main_fit_max_abs_diff"]), 0.0)
        for r in rows.values():
            self.assertLess(float(r["summary_band_max_rel_diff"]), 1e-12)
            self.assertEqual(r["replicates"], "4")
        self.assertEqual(rows[("1", "1000")]["trunks"], "3")
        self.assertEqual(rows[("1", "500")]["trunks"], "6")

    def test_skipped_replicates_are_left_out(self):
        cfg = self.cfg
        n = cfg.inference.n_iter
        reps = [params(FIT_OFFSETS[0] + b, n) for b in BOOT_OFFSETS[:3]]
        self.lay.replicate_json(1, 1000, 4).unlink()
        write_json(self.lay.boot_summary(1, 1000),
                   psmc_rs_summary(params(FIT_OFFSETS[0], n), reps, cfg, 1000, e5.bootstrap_seed(cfg, 1, 1000), [4]))
        e5.report(Context(self.profile, self.runs, log=lambda msg: None))
        rows = {(r["dataset"], r["trunk_bins"]): r for r in read_csv(self.lay.table("datasets"))}
        self.assertEqual(rows[("1", "1000")]["replicates"], "3")
        self.assertEqual(rows[("1", "1000")]["skipped_replicates"], "4")

    def test_a_summary_that_disagrees_with_the_replicates_is_refused(self):
        path = self.lay.boot_summary(2, 1000)
        summary = json.loads(path.read_text())
        summary["points"][10]["ne_q975"] *= 1.001
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "summary.json"):
            e5.report(Context(self.profile, self.runs, log=lambda msg: None))

    def test_a_bootstrap_with_another_seed_is_refused(self):
        path = self.lay.boot_summary(1, 500)
        summary = json.loads(path.read_text())
        summary["seed"] = 42
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "seed"):
            e5.report(Context(self.profile, self.runs, log=lambda msg: None))

@unittest.skipUnless(in_git_work_tree(), "not a git checkout")
class RunsLayout(unittest.TestCase):
    def test_what_git_keeps(self):
        assert_ignore_rules(self, kept=(
            "experiment/runs/E5/truth/bottleneck/rep01.json",
            "experiment/runs/E5/fits/bottleneck/rep01.rs.json",
            "experiment/runs/E5/boot/trunk500000/rep01.rs.json",
            "experiment/runs/E5/boot/trunk500000/rep01/replicate_001.json",
            "experiment/runs/E5/boot/trunk50000/rep10/summary.json",
            "experiment/runs/E5/boot/trunk50000/rep10/summary.tsv",
            "experiment/runs/E5/provenance/boot-trunk50000/bottleneck/rep10.json",
            "experiment/runs/E5/tables/E5_summary.csv",
            "experiment/runs/E5/figures/fig6_bootstrap_calibration.pdf",
            "experiment/runs/E5/figures/fig6_bootstrap_calibration.png",
        ), ignored=(
            "experiment/runs/E5/sims/bottleneck/rep01/genome.psmcfa",
            "experiment/runs/E5/boot/trunk50000/rep10/replicate_050.psmc",
            "experiment/runs/E5/boot/trunk50000/rep10.rs.html",
            "experiment/runs/E5/fits/bottleneck/rep01.rs.html",
            "experiment/runs/E5/logs/jobs/boot-trunk50000/bottleneck/rep10.log",
        ))


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """E5 in the tiny runs of every experiment (tiny_runs): simulate, fit, bootstrap and tables; then nothing is left to do."""

    def test_tiny_profile(self):
        runs = tiny_runs.runs_dir()
        lay = e5.Layout(runs / e5.NAME, config.TINY.e5.model)
        datasets = read_csv(lay.table("datasets"))
        self.assertEqual(len(datasets), len(config.TINY.e5.trunk_bins))
        for row in datasets:
            self.assertEqual(row["main_fit_equals_fit"], "True")  # psmc-rs's results do not depend on the run
            self.assertLess(float(row["summary_band_max_rel_diff"]), 1e-9)
            self.assertEqual(int(row["replicates"]), config.TINY.e5.bootstrap_replicates)
        statuses = runner.job_statuses(tiny_runs.graph(), runs, daemon_pid=None)
        self.assertEqual({s.state for s in statuses if s.job.experiment == "E5"}, {runner.DONE})


if __name__ == "__main__":
    unittest.main()
