"""Tests for experiment.pipeline.figures.e6_supplementary: every figure draws from tables shaped as E6 writes them."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment.pipeline import config, demography, results  # noqa: E402
from experiment.pipeline.experiments import e1  # noqa: E402

C_REF = ROOT / "tests" / "data" / "c_reference" / "c_4+25x2+4+6_N5.psmc"
HAS_ALL = all(importlib.util.find_spec(m) is not None for m in ("numpy", "pandas", "scipy", "matplotlib"))

if HAS_ALL:
    import numpy as np
    import pandas as pd

    from experiment.pipeline.experiments import e6_analysis as A
    from experiment.pipeline.figures import e6_supplementary as F
    from test_pipeline_e6_analysis import thread_rows


@unittest.skipUnless(HAS_ALL, "numpy, pandas, scipy and matplotlib needed")
class Figures(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)

    def table(self, name: str, rows) -> Path:
        e1.write_csv(self.d / name, rows)
        return self.d / name

    def check(self, part, tables):
        outs = [self.d / f"fig_{part}.pdf", self.d / f"fig_{part}.png"]
        written = F.draw(part, tables, outs)
        self.assertEqual(set(written.values()), set(outs))
        for o in outs:
            self.assertGreater(o.stat().st_size, 1000)
        self.assertFalse(list(self.d.glob("*.partial")))

    def test_s1(self):
        c = results.parse_psmc(C_REF)
        ll = c.loglikes()
        rows = []
        for model in ("constant", "zigzag"):
            for r in (1, 2):
                params = {"loglike": [ll[i] * (1 + 1e-9 * r) for i in range(5)] + [ll[4] + 0.5]}
                rows += A.trace_rows(model, r, c, params, c, 5)
        self.check("s1", {"traces": self.table("traces.csv", rows)})

    def test_s2(self):
        last = results.parse_psmc(C_REF).last()
        base = {"theta": last.pa["theta"], "rho": last.pa["rho"], "t_max": last.pa["t_max"], "n_steps": 63,
                "pattern": "4+25*2+4+6", "lam": list(last.pa["lambdas"]), "loglike": [-10.0], "input": {}}
        mhs = dict(base, lam=[x * 1.01 for x in base["lam"]])
        fits = {"psmcfa": A.FormatFit("psmcfa", base), "vcf": A.FormatFit("vcf", base), "multihetsep": A.FormatFit("multihetsep", mhs)}
        epochs = demography.epochs(config.CONSTANT)
        window = np.logspace(4, 6, 50)
        fit_rows = A.fit_rows("constant", fits, epochs, window, 2.5e-8, 25.0, 100)
        curves = [r for f, ff in fits.items() for r in A.curve_rows("constant", f, ff.fit, 2.5e-8, 25.0, 100)]
        curves += A.truth_rows("constant", epochs, np.logspace(2, 8, 50), 25.0)
        ref = {"chr1": "TTKT" * 10}
        bins = A.compare_bins("constant", "vcf", ref, ref) + A.compare_bins("constant", "multihetsep", ref, {"chr1": "TTKT" * 9 + "TTK"})
        self.check("s2", {"bins": self.table("bins.csv", bins), "fits": self.table("fits.csv", fit_rows),
                          "curves": self.table("curves.csv", curves)})

    def test_s3(self):
        rng = np.random.default_rng(3)
        k = 2000
        true = np.repeat(10 ** rng.uniform(4, 6, k // 50), 50)
        mean = true * 10 ** rng.normal(0, 0.2, k)
        edges = list(np.logspace(3, 7, 40))
        edges[0] = 0.0
        post = pd.DataFrame({
            "seq_id": np.zeros(k, dtype=int), "seq_bin": np.arange(k), "obs": np.where(rng.random(k) < 0.1, "K", "T"),
            "map_state": A.state_of(mean, edges), "tmrca_map_years": mean, "tmrca_mean_years": mean,
            "tmrca_q025_years": mean / 3, "tmrca_q975_years": mean * 3,
        })
        d = A.align(post, [(true, np.ones(k, dtype=int))], edges, 1.0)
        self.check("s3", {
            "summary": self.table("summary.csv", [A.tmrca_summary(d, "all")]),
            "calibration": self.table("cal.csv", A.calibration_rows(d, "true", 10) + A.calibration_rows(d, "posterior_mean", 10)),
            "density": self.table("dens.csv", A.density_rows(d, 30)),
            "track": self.table("track.csv", A.track_rows(d, 0, "chr1", 0, 1000, 100)),
        })

    def test_s4(self):
        rows = thread_rows()
        small = [dict(r, input="<runs>/E3/small.psmcfa", genome_mb=100.0, wall_seconds=r["wall_seconds"] / 30)
                 for r in rows if r["threads"] in (1, 8)]
        rows += small
        self.check("s4", {
            "runs": self.table("runs.csv", rows),
            "scaling": self.table("scaling.csv", A.scaling_rows(rows)),
            "determinism": self.table("det.csv", A.determinism_rows(rows)),
        })


if __name__ == "__main__":
    unittest.main()
