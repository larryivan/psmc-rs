"""Tests for E7's data stage (kg1000): its parsers, the samples, the chunks, the download and (with the tools) a run.

The run is the tiny profile's (tiny_runs.py), on the synthetic cohort of ``kg1000 fixture``.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import shutil
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, kg1000  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiny_runs  # noqa: E402

PED = """FamilyID SampleID FatherID MotherID Sex Population Superpopulation
F1 A1 0 0 1 YRI AFR
F1 A2 0 0 2 YRI AFR
F1 A3 A1 A2 1 YRI AFR
F2 B1 0 0 2 CEU EUR
F2 B2 0 0 1 CEU EUR
F3 C1 0 0 2 ASW AFR
"""


def index_text(rows):
    """A sequence index with the columns the stage reads (10: sample, 11: population, 21: BASE_COUNT)."""
    head = ["ENA_FILE_PATH", "MD5SUM", "RUN_ID", "STUDY_ID", "STUDY_NAME", "CENTER_NAME", "SUBMISSION_ID",
            "SUBMISSION_DATE", "SAMPLE_ID", "SAMPLE_NAME", "POPULATION", "EXPERIMENT_ID", "INSTRUMENT_PLATFORM",
            "INSTRUMENT_MODEL", "LIBRARY_NAME", "RUN_NAME", "INSERT_SIZE", "LIBRARY_LAYOUT", "PAIRED_FASTQ",
            "READ_COUNT", "BASE_COUNT"]
    lines = ["##FileDate=20200101", "#" + "\t".join(head)]
    for name, pop, bases in rows:
        f = ["x"] * len(head)
        f[9], f[10], f[20] = name, pop, str(bases)
        lines.append("\t".join(f))
    return "\n".join(lines) + "\n"


class Parsers(unittest.TestCase):
    def test_manifest_columns_in_any_order(self):
        md5a, md5b = "a" * 32, "b" * 32
        text = f"./x/chr1.vcf.gz\t123\t{md5a}\n{md5b}  chr2.vcf.gz.tbi 45\nheader line\n"
        self.assertEqual(kg1000.parse_manifest(text), {"chr1.vcf.gz": {"bytes": 123, "md5": md5a},
                                                       "chr2.vcf.gz.tbi": {"bytes": 45, "md5": md5b}})

    def test_index_strips_names_and_sums_a_samples_runs(self):
        text = index_text([("A1", "YRI", 10), ("A1 ", "YRI", 5), ("B1", "CEU", 7)])
        self.assertEqual(kg1000.read_index(text), {"A1": {"population": "YRI", "raw_bases": 15},
                                                   "B1": {"population": "CEU", "raw_bases": 7}})

    def test_index_refuses_a_sample_in_two_populations(self):
        with self.assertRaises(kg1000.StageError):
            kg1000.read_index(index_text([("A1", "YRI", 10), ("A1", "CEU", 5)]))

    def test_pedigree_people_and_parents(self):
        people, parents = kg1000.read_ped(PED)
        self.assertEqual(people["A3"], {"population": "YRI", "superpopulation": "AFR", "sex": "male"})
        self.assertEqual(people["B1"]["sex"], "female")
        self.assertEqual(parents, {"A1", "A2"})

    def test_contigs_and_samples_of_a_header(self):
        header = ("##fileformat=VCFv4.2\n##contig=<ID=chr21,length=46709983>\n##contig=<ID=chrX,length=156040895,"
                  "assembly=hg38>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n")
        self.assertEqual(kg1000.contig_lengths(header), {"chr21": 46709983, "chrX": 156040895})
        self.assertEqual(kg1000.header_samples(header), ["S1", "S2"])


class Samples(unittest.TestCase):
    def setUp(self):
        self.people, self.parents = kg1000.read_ped(PED)
        self.index = kg1000.read_index(index_text([("A1", "YRI", 3_100_000_000), ("A2", "YRI", 1), ("A3", "YRI", 1),
                                                   ("B1", "CEU", 1), ("B2", "CEU", 1), ("C1", "ASW", 1)]))
        self.cfg = replace(config.E7(), pinned=("A2",), per_population=2, bootstrap_samples=("A2",))

    def select(self, **kw):
        return {s.sample_id: s for s in kg1000.select_samples(replace(self.cfg, **kw), self.index, self.people,
                                                               self.parents)}

    def test_pinned_first_then_drawn_per_population(self):
        out = self.select()
        self.assertEqual(sorted(out), ["A1", "A2", "A3", "B1", "B2", "C1"])  # the cohort: every unrelated sample
        default = {s for s, r in out.items() if r.default}
        self.assertIn("A2", default)
        self.assertEqual(out["A2"].selection, "pinned")
        self.assertEqual(len([s for s in default if out[s].population == "YRI"]), 2)
        self.assertEqual({s for s in default if out[s].population == "CEU"}, {"B1", "B2"})
        self.assertEqual(out["C1"].selection, "drawn")
        # The first default sample of a population is its split check: the pinned one in YRI.
        self.assertEqual({s for s, r in out.items() if r.split_check and r.population == "YRI"}, {"A2"})
        self.assertEqual(sum(r.split_check for r in out.values()), 3)

    def test_roles(self):
        out = self.select()
        self.assertTrue(out["C1"].admixed)
        self.assertFalse(out["A1"].admixed)
        self.assertTrue(out["A1"].is_parent)
        self.assertTrue(out["A2"].bootstrap)
        self.assertFalse(out["A1"].bootstrap)
        self.assertAlmostEqual(out["A1"].raw_depth, round(3_100_000_000 / self.cfg.raw_depth_genome_bp, 3))
        self.assertEqual({r.selection for s, r in out.items() if not r.default}, {"cohort"})

    def test_the_draw_is_the_same_every_time(self):
        self.assertEqual(self.select(), self.select())

    def test_without_the_cohort_only_the_default_set(self):
        out = self.select(cohort=None)
        self.assertTrue(all(r.default for r in out.values()))
        self.assertEqual(len(out), 5)  # two per population, one in ASW

    def test_refusals(self):
        with self.assertRaises(kg1000.StageError):
            self.select(pinned=("Z9",))
        with self.assertRaises(kg1000.StageError):
            self.select(cohort="all3202")
        index = dict(self.index, Z9={"population": "YRI", "raw_bases": 1})
        with self.assertRaises(kg1000.StageError):  # not in the pedigree
            kg1000.select_samples(self.cfg, index, self.people, self.parents)

    def test_the_table_reads_back(self):
        rows = list(self.select().values())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.tsv"
            kg1000.write_tsv(path, [r.row() for r in rows], kg1000.SAMPLE_COLUMNS)
            self.assertEqual(kg1000.read_samples(path), rows)


class Chunks(unittest.TestCase):
    def test_chunks_tile_each_chromosome(self):
        cfg = replace(config.E7(), chromosomes=("chr21", "chr22"), chunk_bp=1_000)
        plan = kg1000.plan_chunks(cfg, {"chr21": 2_500, "chr22": 1_000, "chrX": 9})
        self.assertEqual([(c.chrom, c.index, c.start, c.end) for c in plan],
                         [("chr21", 0, 1, 1_000), ("chr21", 1, 1_001, 2_000), ("chr21", 2, 2_001, 2_500),
                          ("chr22", 0, 1, 1_000)])
        self.assertEqual(plan[1].region, "chr21:1001-2000")

    def test_a_window_cuts_them(self):
        cfg = replace(config.E7(), chromosomes=("chr21",), chunk_bp=400, window=(1_001, 2_000))
        plan = kg1000.plan_chunks(cfg, {"chr21": 5_000})
        self.assertEqual([(c.start, c.end) for c in plan], [(1_001, 1_400), (1_401, 1_800), (1_801, 2_000)])

    def test_a_chromosome_without_a_length_is_refused(self):
        with self.assertRaises(kg1000.StageError):
            kg1000.plan_chunks(replace(config.E7(), chromosomes=("chr1",)), {"chr21": 5})


class _Ranges(http.server.BaseHTTPRequestHandler):
    """Serves ``data`` with byte ranges; the first ``cuts`` answers stop after ``cut`` bytes (a broken transfer)."""

    data = b""
    cut = 0
    cuts = 0
    lock = threading.Lock()
    ignore_range = False

    def do_GET(self):
        rng = self.headers.get("Range")
        if self.ignore_range or rng is None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.data)))
            self.end_headers()
            self.wfile.write(self.data)
            return
        a, b = (int(x) for x in rng.split("=")[1].split("-"))
        body = self.data[a:b + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        self.end_headers()
        with self.lock:
            broken = type(self).cuts > 0 and len(body) > self.cut
            if broken:
                type(self).cuts -= 1
        if broken:
            self.wfile.write(body[: self.cut])
            self.wfile.flush()
            self.connection.shutdown(2)  # the client sees a short read
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Fetch(unittest.TestCase):
    def serve(self, data: bytes, cut: int = 0, cuts: int = 0, ignore_range: bool = False) -> str:
        handler = type("H", (_Ranges,), {"data": data, "cut": cut, "cuts": cuts, "lock": threading.Lock(),
                                         "ignore_range": ignore_range})
        server_cls = type("S", (http.server.ThreadingHTTPServer,), {"request_queue_size": 64})  # every range at once
        server = server_cls(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/chr21.vcf.gz"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.data = bytes(range(256)) * 1_000 + b"end"

    def test_ranges_join_to_the_file(self):
        url = self.serve(self.data)
        dest = self.tmp / "local" / "chr21.vcf.gz"
        got = kg1000.fetch_ranges(url, dest, len(self.data), parts=7, attempts=1, wait=0.0)
        self.assertEqual(dest.read_bytes(), self.data)
        self.assertEqual(got["md5"], hashlib.md5(self.data).hexdigest())
        self.assertEqual(got["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertEqual(sorted(p.name for p in dest.parent.iterdir()), ["chr21.vcf.gz"])  # no parts left

    def test_a_broken_range_resumes_where_it_stopped(self):
        url = self.serve(self.data, cut=1_000, cuts=3)
        dest = self.tmp / "chr21.vcf.gz"
        kg1000.fetch_ranges(url, dest, len(self.data), parts=3, attempts=4, wait=0.0)
        self.assertEqual(dest.read_bytes(), self.data)

    def test_a_part_longer_than_its_range_is_fetched_again(self):
        url = self.serve(self.data)
        dest = self.tmp / "chr21.vcf.gz"
        (self.tmp / "chr21.vcf.gz.part000").write_bytes(b"x" * (len(self.data) + 5))
        kg1000.fetch_ranges(url, dest, len(self.data), parts=1, attempts=2, wait=0.0)
        self.assertEqual(dest.read_bytes(), self.data)

    def test_a_server_without_ranges_is_refused(self):
        url = self.serve(self.data, ignore_range=True)
        with self.assertRaisesRegex(kg1000.StageError, "ignored the range"):
            kg1000.fetch_ranges(url, self.tmp / "chr21.vcf.gz", len(self.data), parts=2, attempts=1, wait=0.0)

    def test_too_little_local_space_is_refused_before_downloading(self):
        from unittest import mock

        url = self.serve(self.data)
        lay = kg1000.layout(self.tmp / "runs")
        dirs = kg1000.temp_dirs(lay, local_tmp_dir=self.tmp / "node")
        kg1000.write_json(lay.record_setup, {"vcfs": {"chr21": {"read_from": url,
                                                                 "head": {"content_length": len(self.data)}}}})
        full = shutil.disk_usage(self.tmp)._replace(free=len(self.data))
        with mock.patch.object(kg1000.shutil, "disk_usage", return_value=full):
            with self.assertRaisesRegex(kg1000.StageError, "GB free"):
                kg1000.fetch_chrom(config.TINY.e7, lay, dirs, "chr21", log=lambda msg: None)
        self.assertFalse(kg1000.local_vcf(config.TINY.e7, dirs, "chr21").exists())

    def test_a_local_source_is_not_fetched(self):
        lay = kg1000.layout(self.tmp / "runs")
        kg1000.write_json(lay.record_setup, {"vcfs": {"chr21": {"read_from": "/data/chr21.vcf.gz"}}})
        rec = kg1000.fetch_chrom(config.TINY.e7, lay, kg1000.temp_dirs(lay), "chr21", log=lambda msg: None)
        self.assertFalse(rec["fetched"])
        self.assertFalse(lay.record_fetch("chr21").exists())

    def test_the_download_is_checked_against_the_manifest(self):
        url = self.serve(self.data)
        lay = kg1000.layout(self.tmp / "runs")
        dirs = kg1000.temp_dirs(lay, local_tmp_dir=self.tmp / "node")
        cfg = replace(config.TINY.e7, fetch_parts=4)
        entry = {"read_from": url, "head": {"content_length": len(self.data)},
                 "manifest": {"md5": hashlib.md5(self.data).hexdigest()}}
        kg1000.write_json(lay.record_setup, {"vcfs": {"chr21": entry}})
        rec = kg1000.fetch_chrom(cfg, lay, dirs, "chr21", log=lambda msg: None)
        self.assertTrue(rec["fetched"])
        self.assertEqual(Path(rec["path"]), kg1000.local_vcf(cfg, dirs, "chr21"))
        self.assertEqual(Path(rec["path"]).read_bytes(), self.data)
        self.assertEqual(kg1000.read_json(lay.record_fetch("chr21"))["md5"], entry["manifest"]["md5"])
        self.assertEqual(kg1000.stream_path(cfg, lay, "chr21", dirs).split("##idx##")[0], rec["path"])  # the copy
        # Present already: checked, not fetched again; a wrong MD5 deletes it.
        self.assertFalse(kg1000.fetch_chrom(cfg, lay, dirs, "chr21", log=lambda msg: None)["fetched"])
        entry["manifest"]["md5"] = "0" * 32
        kg1000.write_json(lay.record_setup, {"vcfs": {"chr21": entry}})
        with self.assertRaisesRegex(kg1000.StageError, "MD5"):
            kg1000.fetch_chrom(cfg, lay, dirs, "chr21", log=lambda msg: None)
        self.assertFalse(kg1000.local_vcf(cfg, dirs, "chr21").exists())


@unittest.skipIf(tiny_runs.SKIP, tiny_runs.SKIP)
class TinyStage(unittest.TestCase):
    """What the tiny profile's data stage wrote (tiny_runs.e7_data: the fixture, every step, the pilot on chr21)."""

    @classmethod
    def setUpClass(cls):
        cls.runs = tiny_runs.runs_dir()
        cls.cfg = config.TINY.e7
        cls.lay = kg1000.layout(cls.runs)

    def test_the_pilot_matches_the_split_route_byte_for_byte(self):
        rec = kg1000.read_json(self.lay.record_pilot("chr21"))
        self.assertTrue(rec["identical"])
        self.assertEqual({r["sample_id"] for r in rec["samples"]}, set(self.cfg.pilot_samples))

    def test_every_sample_has_its_psmcfa_and_the_manifest_lists_it(self):
        samples = kg1000.read_samples(self.lay.samples)
        manifest = kg1000.read_json(self.lay.manifest)
        for s in samples:
            p = self.lay.psmcfa(s.sample_id)
            self.assertTrue(p.exists(), p)
            self.assertIn(p.relative_to(self.lay.inputs).as_posix(), manifest["outputs"])
            text = gzip.decompress(p.read_bytes()).decode()
            self.assertEqual([line[1:] for line in text.splitlines() if line.startswith(">")],
                             list(self.cfg.chromosomes))
        for s in samples:
            self.assertEqual(self.lay.vcf(s.sample_id).exists(), s.default, s.sample_id)

    def test_setup_again_with_the_same_settings_does_nothing(self):
        before = self.lay.record_setup.read_bytes()
        rec = kg1000.setup(self.cfg, self.lay, log=lambda msg: None)
        self.assertEqual(self.lay.record_setup.read_bytes(), before)
        self.assertEqual(rec["step"], "setup")

    def test_the_manifest_says_how_each_chromosome_was_read(self):
        manifest = kg1000.read_json(self.lay.manifest)
        for chrom in self.cfg.chromosomes:  # the fixture's files are local copies
            self.assertEqual(manifest["chromosomes"][chrom]["read"], {"how": "read by region from a local copy"})

    def test_the_chunks_records_add_up_to_each_chromosome(self):
        for chrom in self.cfg.chromosomes:
            rec = kg1000.read_json(self.lay.record_chrom(chrom))
            self.assertEqual(rec["records_check"], "equal")


if __name__ == "__main__":
    unittest.main()

