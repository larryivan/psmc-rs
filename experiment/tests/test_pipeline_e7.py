"""Tests for experiment E7: its jobs, the GIAB comparison and (with the tools present) the tiny run's tables.

The data stage has its own tests (test_pipeline_kg1000.py); the figures are FIG's (test_figures_*.py).
"""

from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402
from fixtures import fake_ctx, read_csv  # noqa: E402

from experiment.pipeline import config, kg1000, runner  # noqa: E402
from experiment.pipeline.experiments import REGISTRY, e7  # noqa: E402


def sample(sid, pop="YRI", sup="AFR", default=False, split=False, boot=False, giab=False):
    return kg1000.Sample(sample_id=sid, population=pop, superpopulation=sup, sex="female", is_parent=False,
                         admixed=False, raw_bases=93_000_000_000, raw_depth=30.0, default=default,
                         selection="drawn" if default else "cohort", split_check=split, bootstrap=boot, giab=giab)


class Jobs(unittest.TestCase):
    def setUp(self):
        self.runs = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.ctx = fake_ctx(config.PRODUCTION, self.runs, log=lambda msg: None)
        self.lay = e7.Layout(self.ctx.exp_dir("E7"))

    def stage(self, samples):
        data = self.lay.data
        kg1000.write_tsv(data.samples, [s.row() for s in samples], kg1000.SAMPLE_COLUMNS)
        kg1000.write_tsv(data.chunks, [{"chrom": "chr1", "chunk": 0, "start": 1, "end": 248_956_422},
                                       {"chrom": "chr2", "chunk": 0, "start": 1, "end": 242_193_529}],
                         ("chrom", "chunk", "start", "end"))
        kg1000.write_json(data.manifest, {"manifest": kg1000.MANIFEST_ID})

    def test_registered(self):
        self.assertEqual(REGISTRY["E7"], "e7")

    def test_no_jobs_before_the_data_stage_has_finished(self):
        self.assertEqual(e7.jobs(self.ctx), [])
        self.stage([sample("NA00002")])
        self.lay.data.manifest.unlink()  # setup has written the samples, the chromosomes are not done
        self.assertEqual(e7.jobs(self.ctx), [])

    def test_jobs_follow_each_samples_roles(self):
        self.stage([sample("NA12878", "CEU", "EUR", default=True, split=True, boot=True, giab=True),
                    sample("NA00002", default=True), sample("NA00003")])
        ids = {j.id for j in e7.jobs(self.ctx)}
        self.assertEqual(ids, {
            "E7/fit-rs-psmcfa/NA12878", "E7/fit-rs-vcf/NA12878", "E7/fit-c/NA12878", "E7/boot-rs/NA12878",
            "E7/fit-rs-split/NA12878", "E7/giab/NA12878",
            "E7/fit-rs-psmcfa/NA00002", "E7/fit-rs-vcf/NA00002", "E7/fit-c/NA00002",
            "E7/fit-rs-psmcfa/NA00003",
            "E7/report",
        })

    def test_the_routes_read_what_the_stage_wrote(self):
        self.stage([sample("NA00002", default=True, split=True)])
        by_id = {j.id: j for j in e7.jobs(self.ctx)}
        data = self.lay.data
        psmcfa = data.psmcfa("NA00002")
        self.assertEqual(by_id["E7/fit-rs-psmcfa/NA00002"].inputs, [psmcfa])
        self.assertEqual(by_id["E7/fit-c/NA00002"].inputs, [psmcfa])  # both tools read the identical psmcfa
        self.assertEqual(by_id["E7/fit-rs-split/NA00002"].inputs, [psmcfa])
        vcf = by_id["E7/fit-rs-vcf/NA00002"]
        self.assertEqual(vcf.inputs, [data.vcf("NA00002"), data.mask_all, data.nonpass_all])
        cmd = " ".join(map(str, vcf.command))
        for opt in ("--vcf-mode mask", f"--vcf-mask {data.mask_all}", f"--vcf-negative-mask {data.nonpass_all}",
                    "--bin-callable auto"):
            self.assertIn(opt, cmd)
        split = " ".join(map(str, by_id["E7/fit-rs-split/NA00002"].command))
        self.assertIn(config.PRODUCTION.e7.split_pattern, split)
        self.assertNotIn(config.PRODUCTION.e7.split_pattern, " ".join(map(str, by_id["E7/fit-rs-psmcfa/NA00002"].command)))

    def test_the_bootstrap_seed_and_settings(self):
        self.stage([sample("NA18507", default=True, boot=True)])
        job = {j.id: j for j in e7.jobs(self.ctx)}["E7/boot-rs/NA18507"]
        cfg = config.PRODUCTION.e7
        self.assertEqual(job.seeds, {"bootstrap": config.seed("E7", "NA18507", 0, "bootstrap")})
        cmd = [str(x) for x in job.command]
        self.assertEqual(cmd[cmd.index("--bootstrap") + 1], str(cfg.bootstrap_replicates))
        self.assertEqual(cmd[cmd.index("--bootstrap-block-size") + 1], str(cfg.bootstrap_trunk_bins))
        self.assertEqual(job.cores, cfg.bootstrap_threads)

    def test_the_report_reads_every_fit_and_writes_every_table(self):
        self.stage([sample("NA00002", default=True, split=True), sample("NA00003")])
        jobs = e7.jobs(self.ctx)
        report = jobs[-1]
        self.assertEqual(report.id, "E7/report")
        for j in jobs[:-1]:
            for p in j.outputs:
                self.assertIn(p, report.inputs, j.id)
        self.assertEqual(set(report.outputs), {self.lay.table(k) for k in e7.TABLES})


class Giab(unittest.TestCase):
    HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"

    def vcf(self, path: Path, rows):
        with gzip.open(path, "wt") as f:
            f.write(self.HEADER)
            for chrom, pos, ref, alt, filt, gt in rows:
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t{filt}\t.\tGT\t{gt}\n")

    def test_intervals(self):
        iv = e7.Intervals({"chr1": [(10, 20), (15, 30), (40, 50)]})
        self.assertEqual(iv.bases(), 30)
        self.assertTrue(iv.contains("chr1", 11))  # 1-based 11 is 0-based 10
        self.assertFalse(iv.contains("chr1", 10))
        self.assertTrue(iv.contains("chr1", 30))
        self.assertFalse(iv.contains("chr1", 31))
        both = iv.intersect(e7.Intervals({"chr1": [(25, 45)]}))
        self.assertEqual(both.bases(), 10)

    def test_classify(self):
        self.assertEqual(e7.classify("A", ["G"], ["0", "1"]), ("het-snv", frozenset("AG")))
        self.assertEqual(e7.classify("A", ["G", "T"], ["1", "2"]), ("het-snv", frozenset("GT")))
        self.assertEqual(e7.classify("A", ["AT"], ["0", "1"]), ("het-other", None))
        self.assertEqual(e7.classify("A", ["G"], ["1", "1"]), ("hom", None))
        self.assertEqual(e7.classify("A", ["G"], [".", "1"]), ("missing", None))

    def test_precision_recall_and_why_a_site_was_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "mask.bed").write_text("chr21\t0\t1000\n")
            (tmp / "giab.bed").write_text("chr21\t0\t500\nchr21\t600\t2000\n")
            self.vcf(tmp / "s.vcf.gz", [
                ("chr21", 10, "A", "G", "PASS", "0|1"),    # tp
                ("chr21", 20, "C", "T", "PASS", "0|1"),    # fp: not in GIAB
                ("chr21", 30, "A", "G", "LowQual", "0|1"),  # GIAB het, ours non-PASS
                ("chr21", 40, "A", "G", "PASS", ".|."),    # GIAB het, ours missing
                ("chr21", 50, "A", "AT", "PASS", "0|1"),   # GIAB het SNV here, ours an indel
                ("chr21", 70, "A", "C", "PASS", "0|1"),    # GIAB has A/G: other alleles
                ("chr21", 550, "A", "G", "PASS", "0|1"),   # outside GIAB's regions: not counted
                ("chr21", 1500, "A", "G", "PASS", "0|1"),  # outside the mask: not counted
            ])
            self.vcf(tmp / "giab.vcf.gz", [
                ("chr21", 10, "A", "G", "PASS", "0|1"),
                ("chr21", 30, "A", "G", "PASS", "1|0"),
                ("chr21", 40, "A", "G", "PASS", "0/1"),
                ("chr21", 50, "A", "G", "PASS", "0/1"),
                ("chr21", 60, "A", "G", "PASS", "0/1"),    # absent from ours
                ("chr21", 70, "A", "G", "PASS", "0/1"),
                ("chr21", 80, "A", "G", "PASS", "1/1"),    # hom in GIAB: not a truth het
            ])
            got = e7.giab_compare(tmp / "s.vcf.gz", tmp / "giab.vcf.gz", tmp / "giab.bed", tmp / "mask.bed",
                                  ("chr21",), "S")
        self.assertEqual(got["eval_bases"], 500 + 400)
        self.assertEqual((got["truth_het_snvs"], got["called_het_snvs"], got["tp"], got["fp"], got["fn"]), (6, 3, 1, 2, 5))
        self.assertAlmostEqual(got["precision"], 1 / 3)
        self.assertAlmostEqual(got["recall"], 1 / 6)
        self.assertEqual((got["fn_nonpass"], got["fn_missing_gt"], got["fn_het_indel"], got["fn_hom_or_absent"],
                          got["fn_other_alleles"]), (1, 1, 1, 1, 1))


class Outputs(unittest.TestCase):
    def test_lengths_of_concatenated_gzip_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.psmcfa.gz"
            p.write_bytes(gzip.compress(b">chr1\nTTKN\nT\n") + gzip.compress(b">chr2\nNN\n"))
            self.assertEqual(e7.psmcfa_lengths(p), (5, 2))  # e2's reader


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class TinyRun(unittest.TestCase):
    """E7 at the tiny profile on the synthetic cohort (tiny_runs.py runs it once for every test module)."""

    @classmethod
    def setUpClass(cls):
        cls.runs = tiny_runs.runs_dir()
        cls.cfg = config.TINY.e7
        cls.lay = e7.Layout(cls.runs / "E7")
        cls.samples = kg1000.read_samples(cls.lay.data.samples)

    def table(self, key):
        return read_csv(self.lay.table(key))

    def test_every_job_is_done(self):
        statuses = [s for s in runner.job_statuses(tiny_runs.graph(), self.runs, daemon_pid=None)
                    if s.job.experiment == "E7"]
        self.assertGreater(len(statuses), len(self.samples))
        for s in statuses:
            self.assertEqual(s.state, runner.DONE, s.job.id)

    def test_every_table_is_written(self):
        for key in e7.TABLES:
            self.assertTrue(self.lay.table(key).exists(), key)

    def test_one_psmcfa_fit_per_sample_and_the_default_sets_other_routes(self):
        fits = self.table("fits")
        default = {s.sample_id for s in self.samples if s.default}
        by_route = {}
        for r in fits:
            by_route.setdefault(r["route"], set()).add(r["sample_id"])
        self.assertEqual(by_route["rs-psmcfa"], {s.sample_id for s in self.samples})
        self.assertEqual(by_route["rs-vcf"], default)
        self.assertEqual(by_route["c"], default)
        self.assertEqual(by_route["rs-split"], {s.sample_id for s in self.samples if s.split_check})

    def test_psmc_rs_and_c_psmc_agree_on_the_same_psmcfa(self):
        rows = self.table("rs_vs_c")
        self.assertEqual({r["sample_id"] for r in rows}, {s.sample_id for s in self.samples if s.default})
        for r in rows:
            self.assertLess(float(r["rmse_log10_ne_rs_vs_c_per_interval"]), 0.05, r["sample_id"])

    def test_bands_of_every_population_and_superpopulation(self):
        bands = self.table("pop_bands")
        kinds = {(r["group_kind"], r["group"]) for r in bands if r["route"] == "rs-psmcfa"}
        self.assertIn(("all", "all"), kinds)
        for s in self.samples:
            self.assertIn(("population", s.population), kinds)
            self.assertIn(("superpopulation", s.superpopulation), kinds)
        for r in bands:
            self.assertLessEqual(float(r["ne_lo"]), float(r["ne_median"]) * (1 + 1e-9))
            self.assertLessEqual(float(r["ne_median"]), float(r["ne_hi"]) * (1 + 1e-9))

    def test_the_split_first_window_lies_within_its_intervals(self):
        rows = self.table("split")
        self.assertEqual({r["sample_id"] for r in rows}, {s.sample_id for s in self.samples if s.split_check})
        for r in rows:
            lo, mid, hi = (float(r[k]) for k in ("log10_first_vs_next_split_lo", "log10_first_vs_next_split",
                                                  "log10_first_vs_next_split_hi"))
            self.assertLessEqual(lo, mid + 1e-12, r["sample_id"])
            self.assertLessEqual(mid, hi + 1e-12, r["sample_id"])

    def test_giab_on_the_fixture(self):
        rows = self.table("giab")
        self.assertEqual([r["sample_id"] for r in rows], [s.sample_id for s in self.samples if s.giab])


if __name__ == "__main__":
    unittest.main()

