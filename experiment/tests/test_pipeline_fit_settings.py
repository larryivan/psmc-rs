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

import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, paths, tools  # noqa: E402
from experiment.pipeline.experiments import Context, e1, e2, e5, e6, jobs_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

FAKE = {
    "c": tools.Tool("c-psmc", Path("/x/psmc"), {"version": "0.6.5", "sha256": "0" * 64}),
    "rs01": tools.Tool("psmc-rs-0.1", Path("/x/psmc-rs-0.1"), {"version": "psmc-rs 0.1.0", "commit": "e" * 40}),
    "rs02": tools.Tool("psmc-rs-0.2", Path("/x/psmc-rs"), {"version": "psmc-rs 0.2.0", "source_commit": "a" * 40}),
    "fq2psmcfa": tools.Tool("fq2psmcfa", Path("/x/fq2psmcfa"), {"sha256": "1" * 64}),
    "generate_multihetsep.py": tools.Tool("generate_multihetsep.py", Path("/x/gm.py"), {"sha256": "2" * 64}),
}
COUNTED_SUFFIXES = (".vcf", ".vcf.gz", ".multihetsep", ".mhs")  # inputs whose bins carry callable bases


def default_mstep_iters() -> int:
    """psmc-rs's ``DEFAULT_MSTEP_ITERS`` (src/opt.rs)."""
    m = re.search(r"pub const DEFAULT_MSTEP_ITERS: usize = (\d+);", (paths.ROOT / "src" / "opt.rs").read_text())
    assert m, "DEFAULT_MSTEP_ITERS not found in src/opt.rs"
    return int(m.group(1))


class Fit:
    """A job that fits with psmc-rs 0.2: its id, inputs and psmc-rs's own command line (E3 wraps it)."""

    def __init__(self, job, command):
        self.id, self.inputs, self.command = job.id, job.inputs, command


def rs02_fits(profile: config.Profile, experiments, runs: Path):
    """The fits of ``experiments`` with psmc-rs 0.2 (not its convert or qc subcommands)."""
    ctx = Context(profile, runs, _tools=dict(FAKE), log=lambda msg: None)
    rs = str(FAKE["rs02"].path)
    out = []
    for j in jobs_for(ctx, experiments):
        cmd = [str(c) for c in j.command]
        if rs in cmd:
            own = cmd[cmd.index(rs):]
            if own[1] not in ("convert", "qc"):
                out.append(Fit(j, own))
    return out


def by_experiment(fits):
    return {f.id.split("/")[0] for f in fits}


def flag(command, name):
    command = [str(c) for c in command]
    return command[command.index(name) + 1] if name in command else None


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
        with tempfile.TemporaryDirectory() as d:
            for p in (config.PRODUCTION, config.SMOKE):
                jobs = rs02_fits(p, ("E1", "E2", "E3", "E4", "E5", "E6"), Path(d))
                self.assertEqual(by_experiment(jobs), {"E1", "E2", "E3", "E4", "E5", "E6"})
                for j in jobs:
                    with self.subTest(profile=p.name, job=j.id):
                        self.assertEqual(flag(j.command, "--mstep-iters"), default)


class CallableBases(unittest.TestCase):
    def test_psmcfa_experiments_fit_psmcfa_only(self):
        with tempfile.TemporaryDirectory() as d:
            for p in (config.PRODUCTION, config.SMOKE):
                jobs = rs02_fits(p, ("E1", "E2", "E3", "E5"), Path(d))
                self.assertEqual(by_experiment(jobs), {"E1", "E2", "E3", "E5"})
                for j in jobs:
                    with self.subTest(profile=p.name, job=j.id):
                        counted = [str(x) for x in j.inputs if str(x).endswith(COUNTED_SUFFIXES)]
                        self.assertEqual(counted, [])
                        self.assertTrue(str(j.command[1]).endswith(".psmcfa"), j.command[1])
                        self.assertNotIn("--bin-callable", [str(c) for c in j.command])
                        self.assertIn(flag(j.command, "--input-format"), (None, "psmcfa"))

    def test_only_e4_and_e6_read_counted_input(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = rs02_fits(config.PRODUCTION, ("E4", "E6"), Path(d))
        self.assertEqual(by_experiment(j for j in jobs if str(j.command[1]).endswith(COUNTED_SUFFIXES)),
                         {"E4", "E6"})
        e4 = [j for j in jobs if j.id.startswith("E4/fit/")]
        self.assertTrue(all(flag(j.command, "--bin-callable") in ("auto", "ignore") for j in e4))


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class EndToEnd(unittest.TestCase):
    """What the fits of the tiny runs record about the callable bases of their bins."""

    @staticmethod
    def bin_callable(path: Path):
        import json

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
        # callable-length model is C psmc's (the S2 tables also find the VCF fit identical to the psmcfa's).
        runs = tiny_runs.runs_dir()
        lay = e6.Layout(runs / "E6")
        for model in config.TINY.e1.models:
            for fmt in ("vcf", "multihetsep"):
                with self.subTest(model=model, format=fmt):
                    bc = self.bin_callable(lay.s2_fit(model, config.TINY.e6.format_replicate, fmt))
                    self.assertEqual((bc["mode"], bc["model"]), ("auto", "callable-length"))
                    self.assertEqual(bc["callable"]["partial_bins"], 0)
                    self.assertEqual(bc["callable"]["mean_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
