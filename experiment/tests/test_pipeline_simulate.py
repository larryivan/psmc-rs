"""Tests for experiment.pipeline.simulate (skip without msprime; VCF parts without bgzip)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, simulate  # noqa: E402

HAS_MSPRIME = importlib.util.find_spec("msprime") is not None
HAS_HTSLIB = shutil.which("bgzip") is not None and shutil.which("tabix") is not None
PSMC_RS = ROOT / "target" / "release" / "psmc-rs"


class Layout(unittest.TestCase):
    def test_psmcfa_record_is_laid_out_as_fq2psmcfa(self):
        bins = "T" * 59 + "K" + "T" * 61
        self.assertEqual(simulate.psmcfa_record("chr1", bins), ">chr1\n" + "T" * 59 + "K\n" + "T" * 60 + "\nT\n")

    def test_bins_count_a_partial_last_bin(self):
        self.assertEqual(simulate.n_bins(200_000, 100), 2000)
        self.assertEqual(simulate.n_bins(200_001, 100), 2001)


@unittest.skipUnless(HAS_MSPRIME, "msprime not installed")
class SimulatedGenome(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.genome = config.Genome(n_chromosomes=2, chromosome_length=200_000)
        cls.rec = simulate.simulate_genome(
            "TEST", config.BOTTLENECK, 1, cls.genome, config.SIMULATION, cls.tmp / "sim", cls.tmp / "truth.json",
            write_vcf=HAS_HTSLIB, tmrca_path=cls.tmp / "tmrca.tsv.gz", log=lambda m: None,
        )
        cls.psmcfa = (cls.tmp / "sim" / simulate.PSMCFA_NAME).read_text()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def records(self):
        out, name = {}, None
        for line in self.psmcfa.splitlines():
            if line.startswith(">"):
                name = line[1:]
                out[name] = []
            else:
                out[name].append(line)
        return out

    def test_psmcfa(self):
        recs = self.records()
        self.assertEqual(list(recs), ["chr1", "chr2"])
        for lines in recs.values():
            self.assertTrue(all(len(x) == 60 for x in lines[:-1]))
            seq = "".join(lines)
            self.assertEqual(len(seq), 2000)
            self.assertLessEqual(set(seq), {"T", "K"})
        self.assertGreater(self.psmcfa.count("K"), 20)

    def test_truth_record(self):
        rec = json.loads((self.tmp / "truth.json").read_text())
        self.assertEqual(rec, json.loads(json.dumps(self.rec)))
        for ch in rec["chromosomes"]:
            self.assertEqual(ch["seeds"], simulate.seeds_for("TEST", "bottleneck", 1, ch["name"]))
            self.assertEqual(ch["bins"], 2000)
        het_bins = sum(ch["het_bins"] for ch in rec["chromosomes"])
        self.assertEqual(het_bins, self.psmcfa.count("K"))
        self.assertEqual(rec["truth"]["model"], "bottleneck")
        self.assertEqual(rec["simulation"]["recombination_rate"], 1e-8)
        self.assertIn("msprime", rec["software"])

    def test_deterministic(self):
        ch = self.rec["chromosomes"][0]
        again = simulate.simulate_chromosome(config.BOTTLENECK, config.SIMULATION, "chr1", 200_000, ch["seeds"])
        self.assertEqual(simulate.positions_digest(again.het_positions), ch["het_positions_sha256"])

    @unittest.skipUnless(HAS_HTSLIB, "bgzip/tabix not on PATH")
    def test_vcf_matches_psmcfa(self):
        vcf = self.tmp / "sim" / simulate.VCF_NAME
        self.assertTrue(Path(f"{vcf}.tbi").exists())
        het = {"chr1": set(), "chr2": set()}
        contigs = []
        with gzip.open(vcf, "rt") as f:
            for line in f:
                if line.startswith("##contig"):
                    contigs.append(line.strip())
                if line.startswith("#"):
                    continue
                chrom, pos, _, ref, alt, _, filt, _, fmt, gt = line.rstrip("\n").split("\t")
                self.assertEqual((filt, fmt), ("PASS", "GT"))
                a, b = gt.split("|")
                if a != b:
                    het[chrom].add((int(pos) - 1) // 100)
        self.assertEqual(contigs, ["##contig=<ID=chr1,length=200000>", "##contig=<ID=chr2,length=200000>"])
        for name, lines in self.records().items():
            seq = "".join(lines)
            self.assertEqual({i for i, c in enumerate(seq) if c == "K"}, het[name])

    @unittest.skipUnless(HAS_HTSLIB and PSMC_RS.exists(), "needs bgzip and target/release/psmc-rs")
    def test_psmc_rs_reads_the_vcf_as_the_psmcfa(self):
        out = self.tmp / "from_vcf.psmcfa"
        res = subprocess.run(
            [str(PSMC_RS), "convert", str(self.tmp / "sim" / simulate.VCF_NAME), "--psmcfa", str(out), "-q"],
            capture_output=True, text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(out.read_text(), self.psmcfa)

    def test_tmrca_track(self):
        track = simulate.read_tmrca_track(self.tmp / "tmrca.tsv.gz")
        self.assertEqual(set(track), {"chr1", "chr2"})
        for intervals in track.values():
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], 200_000)
            for (s0, e0, t0), (s1, _, t1) in zip(intervals, intervals[1:]):
                self.assertEqual(e0, s1)
                self.assertNotEqual(t0, t1)
            self.assertTrue(all(t > 0 for _, _, t in intervals))

    def test_tmrca_on_demand_reproduces_the_genome(self):
        out = self.tmp / "again.tsv.gz"
        simulate.write_tmrca(self.tmp / "truth.json", out, log=lambda m: None)
        self.assertEqual(simulate.read_tmrca_track(out), simulate.read_tmrca_track(self.tmp / "tmrca.tsv.gz"))

    def test_tmrca_on_demand_refuses_another_genome(self):
        rec = json.loads((self.tmp / "truth.json").read_text())
        rec["chromosomes"][0]["het_positions_sha256"] = "0" * 64
        bad = self.tmp / "bad_truth.json"
        bad.write_text(json.dumps(rec))
        with self.assertRaises(RuntimeError):
            simulate.write_tmrca(bad, self.tmp / "bad.tsv.gz", log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
