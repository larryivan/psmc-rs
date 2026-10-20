"""Every psmc-rs fit of the pipeline: the converged M-step, and the callable-length model only where it can act.

psmc-rs 0.2's M-step runs to convergence, and ``--mstep-iters`` is only its
cap; no profile of the manuscript fits or times an arm with capped M-steps.
psmc-rs's callable-length model (``--bin-callable auto``, its default) needs
the callable bases of each bin, which only VCF and multihetsep input give:
E1, E2, E3 and E5 fit psmcfa files, so their fits are C psmc's model by
construction, and only E4 (and E6's S2, on complete data) reads VCF or
multihetsep.
"""

from __future__ import annotations

import functools
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import fake_ctx, option  # noqa: E402

from experiment.pipeline import config, paths  # noqa: E402
from experiment.pipeline.experiments import e1, e2, e5, e6, jobs_for  # noqa: E402

COUNTED_SUFFIXES = (".vcf", ".vcf.gz", ".multihetsep", ".mhs")  # inputs whose bins carry callable bases
EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")  # E7 has no jobs before its data stage


def default_mstep_iters() -> int:
    """psmc-rs's ``DEFAULT_MSTEP_ITERS`` (src/opt.rs)."""
    m = re.search(r"pub const DEFAULT_MSTEP_ITERS: usize = (\d+);", (paths.ROOT / "src" / "opt.rs").read_text())
    assert m, "DEFAULT_MSTEP_ITERS not found in src/opt.rs"
    return int(m.group(1))


class Fit:
    """A job that fits with psmc-rs 0.2: its id, inputs and psmc-rs's own command line (E3 wraps it)."""

    def __init__(self, job, command):
        self.id, self.inputs, self.command = job.id, job.inputs, command
        self.experiment = job.id.split("/")[0]


@functools.lru_cache(maxsize=None)
def _all_fits(profile_name: str) -> tuple:
    out = []
    with tempfile.TemporaryDirectory() as runs:
        ctx = fake_ctx(config.get_profile(profile_name), Path(runs), log=lambda msg: None)
        rs = str(ctx.rs02().path)
        for j in jobs_for(ctx, EXPERIMENTS):
            cmd = [str(c) for c in j.command]
            if rs in cmd:
                own = cmd[cmd.index(rs):]
                if own[1] not in ("convert", "qc"):
                    out.append(Fit(j, own))
    return tuple(out)


def rs02_fits(profile: config.Profile, experiments) -> list:
    """The fits of ``experiments`` with psmc-rs 0.2 (not its convert or qc subcommands); E1-E6's made once."""
    return [f for f in _all_fits(profile.name) if f.experiment in experiments]


def by_experiment(fits):
    return {f.experiment for f in fits}


class MStep(unittest.TestCase):
    def test_every_inference_uses_psmc_rs_default_cap(self):
        default = default_mstep_iters()
        self.assertEqual(config.INFERENCE.mstep_iters, default)
        for p in (config.PRODUCTION, config.SMOKE, config.TINY):
            for name in ("e1", "e2", "e3", "e4", "e5"):
                with self.subTest(profile=p.name, experiment=name):
                    self.assertEqual(getattr(p, name).inference.mstep_iters, default)

    def test_no_capped_arm_in_the_manuscript_profiles(self):
        # The tiny profile caps one arm at 5 iterations to test the machinery; the others fit none.
        for p in (config.PRODUCTION, config.SMOKE):
            with self.subTest(profile=p.name):
                self.assertIsNone(p.e1.default_mstep_iters)
                self.assertIsNone(p.e3.default_mstep_iters)

    def test_every_fit_passes_the_cap(self):
        default = str(default_mstep_iters())
        for p in (config.PRODUCTION, config.SMOKE):
            jobs = rs02_fits(p, EXPERIMENTS)
            self.assertEqual(by_experiment(jobs), set(EXPERIMENTS))
            for j in jobs:
                with self.subTest(profile=p.name, job=j.id):
                    self.assertEqual(option(j.command, "--mstep-iters"), default)


class CallableBases(unittest.TestCase):
    def test_psmcfa_experiments_fit_psmcfa_only(self):
        for p in (config.PRODUCTION, config.SMOKE):
            jobs = rs02_fits(p, ("E1", "E2", "E3", "E5"))
            self.assertEqual(by_experiment(jobs), {"E1", "E2", "E3", "E5"})
            for j in jobs:
                with self.subTest(profile=p.name, job=j.id):
                    counted = [str(x) for x in j.inputs if str(x).endswith(COUNTED_SUFFIXES)]
                    self.assertEqual(counted, [])
                    # E2's NA12878 is the packaged gzip psmcfa, read as it is.
                    self.assertTrue(j.command[1].endswith((".psmcfa", ".psmcfa.gz")), j.command[1])
                    self.assertNotIn("--bin-callable", j.command)
                    self.assertIn(option(j.command, "--input-format"), (None, "psmcfa"))

    def test_only_e4_and_e6_read_counted_input(self):
        jobs = rs02_fits(config.PRODUCTION, ("E4", "E6"))
        self.assertEqual(by_experiment(j for j in jobs if j.command[1].endswith(COUNTED_SUFFIXES)), {"E4", "E6"})
        e4 = [j for j in jobs if j.id.startswith("E4/fit/")]
        self.assertTrue(all(option(j.command, "--bin-callable") in ("auto", "ignore") for j in e4))


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """What the fits of the tiny runs record about the callable bases of their bins."""

    @staticmethod
    def bin_callable(path: Path):
        return json.loads(Path(path).read_text()).get("bin_callable")

    def test_psmcfa_fits_are_c_psmcs_model(self):
        runs = tiny_runs.runs_dir()
        tiny = config.TINY
        paths_ = [e1.Layout(runs / "E1").rs_json(m, r) for m in tiny.e1.models for r in range(1, tiny.e1.replicates + 1)]
        paths_ += [e2.Layout(runs / "E2").rs_json(g.sample_id) for g in tiny.e2.genomes]
        paths_ += [e5.Layout(runs / "E5", tiny.e5.model).fit_json(r) for r in range(1, tiny.e5.truth_replicates + 1)]
        for p in paths_:
            with self.subTest(fit=p.name):
                self.assertEqual(self.bin_callable(p), {"mode": "auto", "model": "full-bins"})

    def test_e6_formats_of_complete_data_hold_full_bins(self):
        # S2 reads E1's genome as VCF and multihetsep: no missing data, so every called bin is full and the
        # callable-length model of the VCF is C psmc's (the S2 tables also find the VCF fit identical to the
        # psmcfa's). --bin-callable auto fits exact counts only, so the multihetsep, whose counts are
        # approximate, is fitted with C psmc's model itself.
        runs = tiny_runs.runs_dir()
        lay = e6.Layout(runs / "E6")
        expected = {"vcf": ("callable-length", "exact"), "multihetsep": ("full-bins", "approximate")}
        for model in config.TINY.e1.models:
            for fmt, (bin_model, counts) in expected.items():
                with self.subTest(model=model, format=fmt):
                    bc = self.bin_callable(lay.s2_fit(model, config.TINY.e6.format_replicate, fmt))
                    self.assertEqual((bc["mode"], bc["model"], bc["counts"]), ("auto", bin_model, counts))
                    self.assertEqual(bc["callable"]["partial_bins"], 0)
                    self.assertEqual(bc["callable"]["mean_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
