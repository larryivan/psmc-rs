"""Tests for experiment.pipeline.inputprep: callable masks, masked VCF, consensus FASTA, bins."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import has_modules, quiet  # noqa: E402

from experiment.pipeline import config  # noqa: E402

HAS_NUMPY = has_modules("numpy")

if HAS_NUMPY:
    import numpy as np

    from experiment.pipeline import inputprep as ip

VCF_TEXT = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000>\n"
    "##contig=<ID=chr2,length=250>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdiploid\n"
    "chr1\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0|1\n"
    "chr1\t150\t.\tG\tT\t.\tPASS\t.\tGT\t1|0\n"
    "chr1\t420\t.\tC\tG\t.\tPASS\t.\tGT\t1|1\n"
    "chr1\t901\t.\tT\tA,G\t.\tPASS\t.\tGT\t1|2\n"
    "chr2\t17\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class Masks(unittest.TestCase):
    def test_runs_are_disjoint_inside_the_chromosome_and_hit_the_missing_fraction(self):
        rng = np.random.default_rng(1)
        runs = ip.uncallable_runs(2_000_000, 0.3, 20.0, rng)
        self.assertTrue(np.all(runs.ends > runs.starts))
        self.assertTrue(np.all(runs.starts[1:] > runs.ends[:-1]))  # disjoint and not adjacent
        self.assertGreater(runs.starts[0], 0)  # a callable stretch comes first
        self.assertLessEqual(int(runs.ends[-1]), 2_000_000)
        self.assertAlmostEqual(runs.total() / 2_000_000, 0.3, delta=0.01)
        self.assertAlmostEqual(runs.total() / len(runs), 20.0, delta=0.5)

    def test_masks_follow_their_seeds(self):
        spec = config.MaskSpec("dispersed10", 0.1, "dispersed", 20.0)
        a = ip.callable_mask(spec, [("chr1", 100_000), ("chr2", 50_000)], "bottleneck", 1)
        b = ip.callable_mask(spec, [("chr1", 100_000), ("chr2", 50_000)], "bottleneck", 1)
        c = ip.callable_mask(spec, [("chr1", 100_000)], "bottleneck", 2)
        self.assertTrue(np.array_equal(a["chr1"].starts, b["chr1"].starts))
        self.assertFalse(np.array_equal(a["chr1"].starts[:10], c["chr1"].starts[:10]))
        self.assertEqual(ip.mask_seed("bottleneck", 1, "dispersed10", "chr2"),
                         config.seed("E4", "bottleneck", 1, "mask/dispersed10/chr2"))

    def test_control_is_callable_everywhere(self):
        m = ip.callable_mask(None, [("chr1", 1000)], "bottleneck", 1)
        self.assertEqual((m["chr1"].starts.tolist(), m["chr1"].ends.tolist()), ([0], [1000]))
        s = ip.mask_summary(m, {"chr1": 1000})
        self.assertEqual((s["totals"]["missing_fraction"], s["totals"]["uncallable_runs"]), (0.0, 0))

    def test_complement_contains_and_indicator(self):
        runs = ip.Intervals(np.array([2, 7]), np.array([4, 10]))
        callable_ = ip.complement(runs, 10)
        self.assertEqual((callable_.starts.tolist(), callable_.ends.tolist()), ([0, 4], [2, 7]))
        self.assertEqual(callable_.contains([0, 1, 2, 3, 4, 6, 7, 9]).tolist(),
                         [True, True, False, False, True, True, False, False])
        self.assertEqual(callable_.indicator(10).tolist(), [True, True, False, False, True, True, True, False, False, False])

    def test_merge_joins_overlapping_and_adjacent_rows(self):
        m = ip.merge_intervals([10, 0, 5, 30, 40], [20, 5, 8, 30, 45])
        self.assertEqual((m.starts.tolist(), m.ends.tolist()), ([0, 10, 40], [8, 20, 45]))

    def test_bed_round_trip_is_byte_reproducible(self):
        mask = {"chr1": ip.Intervals(np.array([0, 50]), np.array([10, 100])), "chr2": ip.Intervals(np.array([5]), np.array([9]))}
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.bed.gz", Path(d) / "b.bed.gz"
            ip.write_bed(a, mask)
            ip.write_bed(b, mask)
            self.assertEqual(a.read_bytes(), b.read_bytes())  # no time or name in the gzip header
            self.assertEqual(gzip.decompress(a.read_bytes()).decode(), "chr1\t0\t10\nchr1\t50\t100\nchr2\t5\t9\n")
            back = ip.read_bed(a)
            self.assertEqual(back["chr1"].starts.tolist(), [0, 50])
            self.assertEqual(back["chr2"].ends.tolist(), [9])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class VcfAndConsensus(unittest.TestCase):
    def setUp(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "x.vcf"
        path.write_text(VCF_TEXT)
        self.vcf = ip.read_vcf(path)
        # chr1: callable [0, 100) and [400, 1000); chr2 entirely callable
        self.mask = {"chr1": ip.Intervals(np.array([0, 400]), np.array([100, 1000])),
                     "chr2": ip.Intervals(np.array([0]), np.array([250]))}

    def test_reads_records_and_contigs(self):
        self.assertEqual(self.vcf.contigs(), [("chr1", 1000), ("chr2", 250)])
        self.assertEqual([r.pos for r in self.vcf.records], [4, 149, 419, 900, 16])
        self.assertEqual(self.vcf.records[3].alleles(), ("A", "G"))

    def test_records_at_uncallable_sites_are_removed(self):
        masked, counts = ip.mask_vcf(self.vcf, self.mask, "test")
        self.assertEqual([r.pos for r in masked.records], [4, 419, 900, 16])
        self.assertEqual((counts["het_sites_removed"], counts["hom_sites_removed"], counts["het_sites_kept"]), (1, 0, 3))
        self.assertTrue(masked.header[-1].startswith("#CHROM"))
        self.assertIn("##callable_mask=test\n", masked.header)

    def test_consensus_codes(self):
        seq = ip.consensus_sequence(1000, [r for r in self.vcf.records if r.chrom == "chr1"], self.mask["chr1"],
                                    np.random.default_rng(0))
        text = seq.tobytes().decode()
        self.assertEqual(text[4], "M")  # A/C
        self.assertEqual(text[149], "N")  # a record at an uncallable site stays N
        self.assertEqual(text[419], "G")  # homozygous for the derived allele
        self.assertEqual(text[900], "R")  # A/G, both derived
        self.assertEqual(set(text[100:400]), {"N"})
        self.assertTrue(set(text[400:419]) <= set("ACGT"))

    def test_fasta_layout(self):
        rec = ip.fasta_record("c", np.frombuffer(b"A" * 125, dtype=np.uint8))
        self.assertEqual(rec, b">c\n" + b"A" * 60 + b"\n" + b"A" * 60 + b"\n" + b"A" * 5 + b"\n")
        self.assertEqual(ip.fasta_record("c", np.frombuffer(b"A" * 60, dtype=np.uint8)), b">c\n" + b"A" * 60 + b"\n")

    def test_iupac_refuses_indels(self):
        self.assertEqual(ip.iupac("g", "T"), "K")
        with self.assertRaises(ValueError):
            ip.iupac("A", "AT")


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class Bins(unittest.TestCase):
    def test_confusion_and_differing_bins(self):
        exact = {"chr1": b"TTKNNT", "chr2": b"KT"}
        other = {"chr1": b"TTKTNT", "chr2": b"K"}
        comp = ip.compare_bins(exact, other)
        c1 = comp["chromosomes"][0]
        self.assertEqual((c1["confusion"]["N_T"], c1["confusion"]["T_T"], c1["differing"]), (1, 3, 1))
        self.assertEqual(comp["chromosomes"][1]["confusion"]["exact_only"], 1)
        self.assertEqual(comp["totals"]["differing"], 2)
        self.assertAlmostEqual(comp["totals"]["differing_fraction"], 2 / 8)
        self.assertFalse(comp["totals"]["identical"])
        self.assertTrue(ip.compare_bins(exact, dict(exact))["totals"]["identical"])

    def test_missing_and_extra_sequences_count(self):
        comp = ip.compare_bins({"a": b"TT"}, {"b": b"KKK"})
        self.assertEqual(comp["totals"]["differing"], 5)
        self.assertEqual(comp["extra_sequences"], ["b"])

    def test_read_psmcfa(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.psmcfa"
            p.write_text(">chr1 desc\nTTK\nNT\n>chr2\nK\n")
            self.assertEqual(ip.read_psmcfa(p), {"chr1": b"TTKNT", "chr2": b"K"})

    def test_first_difference(self):
        self.assertIsNone(ip.first_difference(["a", "b"], ["a", "b"]))
        self.assertEqual(ip.first_difference(["a", "b"], ["a", "c"])["row"], 2)
        self.assertEqual(ip.first_difference(["a"], ["a", "c"]), {"row": 2, "a": None, "b": "c"})

    def test_bins_step_fails_when_an_asserted_route_differs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.psmcfa").write_text(">c\nTTKN\n")
            (d / "b.psmcfa").write_text(">c\nTTKT\n")
            out = d / "bins.json"
            argv = ["bins", "--exact", str(d / "a.psmcfa"), "--route", f"fq2psmcfa={d / 'b.psmcfa'}", "--out", str(out)]
            with quiet():
                self.assertEqual(ip.main(argv), 0)
                self.assertEqual(ip.main(argv + ["--assert-identical", "fq2psmcfa"]), 1)
            rec = json.loads(out.read_text())
            self.assertFalse(rec["assertions_passed"])
            self.assertEqual(rec["routes"]["fq2psmcfa"]["totals"]["confusion"]["N_T"], 1)


if __name__ == "__main__":
    unittest.main()
