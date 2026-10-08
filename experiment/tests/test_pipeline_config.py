"""Tests for experiment.pipeline.config and experiment.pipeline.paths."""

from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.pipeline import config, paths  # noqa: E402


class Seeds(unittest.TestCase):
    def test_follows_the_documented_scheme(self):
        key = "psmc-rs-manuscript/E1/bottleneck/3/mutations/chr7"
        expected = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1) + 1
        self.assertEqual(config.seed("E1", "bottleneck", 3, "mutations/chr7"), expected)

    def test_pinned(self):
        # Changing the scheme changes every simulation: this value must not move.
        self.assertEqual(config.seed("E1", "constant", 1, "ancestry/chr1"), 2135881609)

    def test_in_range_and_distinct(self):
        seeds = {
            config.seed(e, m, r, f"{s}/chr{c}")
            for e in ("E1", "E4")
            for m in config.MODEL_ORDER
            for r in range(1, 6)
            for s in ("ancestry", "mutations")
            for c in range(1, 11)
        }
        self.assertEqual(len(seeds), 2 * 4 * 5 * 2 * 10)
        self.assertTrue(all(1 <= s <= 2**31 - 1 for s in seeds))

    def test_negative_replicate_is_refused(self):
        with self.assertRaises(ValueError):
            config.seed("E1", "constant", -1, "ancestry/chr1")


class Models(unittest.TestCase):
    def test_ms_events_convert_to_generations_and_sizes(self):
        got = [(c.time, c.size, c.growth_rate) for c in config.BOTTLENECK.changes]
        self.assertEqual(got, [(800.0, 1000.0, 0.0), (1200.0, 10000.0, 0.0), (4000.0, 5000.0, 0.0), (40000.0, 10000.0, 0.0)])
        self.assertEqual(config.BOTTLENECK.initial_size, 20000.0)
        got = [(c.time, c.size) for c in config.EXPANSION.changes]
        self.assertEqual(got, [(400.0, 1000.0), (2400.0, 10000.0), (8000.0, 5000.0), (40000.0, 10000.0), (80000.0, 20000.0)])
        self.assertEqual(config.CONSTANT.changes, ())

    def test_zigzag_is_stdpopsims_definition(self):
        z = config.ZIGZAG
        self.assertEqual(z.initial_size, 5 * 14312)
        self.assertEqual(len(z.changes), 6)
        self.assertTrue(all(c.size is None for c in z.changes[:5]))
        self.assertEqual(z.changes[-1].size, z.initial_size / 10)
        self.assertEqual(z.changes[-1].growth_rate, 0.0)

    def test_registry(self):
        self.assertEqual(config.MODEL_ORDER, ("constant", "bottleneck", "expansion", "zigzag"))
        self.assertEqual(set(config.MODELS), set(config.MODEL_ORDER))


class Design(unittest.TestCase):
    def test_inference_is_the_psmc_readme_human_recipe(self):
        i = config.INFERENCE
        self.assertEqual((i.n_iter, i.t_max, i.tr_ratio, i.pattern), (25, 15.0, 5.0, "4+25*2+4+6"))
        self.assertEqual((i.smooth_lambda, i.bin_size), (0.0, 100))

    def test_simulation(self):
        s = config.SIMULATION
        self.assertEqual((s.mutation_rate, s.recombination_rate), (2.5e-8, 1.0e-8))
        self.assertEqual((s.ancestry_model, s.mutation_model, s.discrete_genome, s.ploidy), ("hudson", "jc69", True, 2))
        self.assertEqual((config.SIMULATION_SCALING.mu, config.SIMULATION_SCALING.gen_years), (2.5e-8, 25.0))

    def test_e1(self):
        e1 = config.PRODUCTION.e1
        self.assertEqual(e1.models, config.MODEL_ORDER)
        self.assertEqual(e1.replicates, 10)
        self.assertEqual((e1.genome.n_chromosomes, e1.genome.chromosome_length, e1.genome.length), (10, 50_000_000, 500_000_000))
        self.assertEqual(e1.genome.names()[0], "chr1")
        self.assertEqual(e1.inference, config.INFERENCE)

    def test_later_experiments(self):
        p = config.PRODUCTION
        self.assertEqual([g.sample_id for g in p.e2.genomes], ["HLemySub1", "HLhydTec1", "HLpelCas1", "NA12878"])
        human = p.e2.genomes[-1]
        self.assertEqual((human.file, human.label, human.sequence_noun),
                         ("NA12878_1kGP30x_diploid.psmcfa.gz", "Homo sapiens (NA12878, CEU)", "chromosomes"))
        # E7's psmcfa of NA12878, committed byte for byte: the pin is its SHA-256 in E7's MANIFEST.json.
        self.assertEqual(hashlib.sha256((paths.REAL_DATA / human.file).read_bytes()).hexdigest(), human.sha256)
        self.assertEqual((human.scaling.mu, human.scaling.gen_years), (1.25e-8, 29.0))
        self.assertIn("Fenner 2005", human.scaling_source)
        self.assertEqual((human.scaling, human.scaling_source), (p.e7.scaling, p.e7.scaling_source))  # as E7 scales it
        for g in p.e2.genomes:
            self.assertTrue((paths.REAL_DATA / g.file).exists(), g.file)

    def test_na12878_is_e7s_psmcfa(self):
        # The pin is E7's record of the file, not only the committed file's own hash: once E7's tables are
        # committed, E7_inputs.csv gives the SHA-256 of the psmcfa its data stage wrote for NA12878.
        import csv

        table = paths.RUNS / "E7" / "tables" / "E7_inputs.csv"
        if not table.exists():
            self.skipTest("E7's tables are not committed yet")
        with open(table, newline="") as f:
            row = next(r for r in csv.DictReader(f) if r["sample_id"] == "NA12878")
        human = next(g for g in config.PRODUCTION.e2.genomes if g.sample_id == "NA12878")
        self.assertEqual(row["psmcfa_sha256"], human.sha256)
        for g in p.e2.genomes[:3]:
            # The mean of nine Chrysemys picta trios (Bergeron et al. 2023), not the crocodilian 7.9e-9.
            self.assertEqual((g.scaling.mu, g.scaling.gen_years), (4.6e-9, 15.0))
            self.assertIn("Bergeron et al. 2023", g.scaling_source)
            self.assertIn("no pleurodire rate has been measured", g.scaling_source)
            self.assertIn("crocodilian rate (Green et al. 2014", g.scaling_source)
            self.assertNotIn("UNVERIFIED", g.scaling_source)
            self.assertTrue(g.scaling_cite)
        self.assertEqual([g for g in p.e2.genomes if g.sample_id in p.e2.split_genomes], list(p.e2.genomes[:3]))
        self.assertEqual(dict(p.e2.split_patterns), {"split2": "2+2+25*2+4+6", "split1": "1+1+1+1+25*2+4+6"})
        self.assertEqual(p.e2.bootstrap_replicates, 100)
        self.assertEqual(p.e3.genome_sizes_mb, (100, 300, 1000, 3000))
        self.assertEqual(p.e2.rs_threads, 50)
        self.assertEqual(p.e3.thread_scaling_threads, (1, 2, 4, 8, 16, 30))
        # 30 chromosomes of 100 Mb: psmc-rs, one sequence per thread, can use no more threads
        self.assertEqual(max(p.e3.thread_scaling_threads), p.e3.thread_scaling_size_mb * 1_000_000 // p.e3.chromosome_length)
        self.assertEqual((p.e3.rs01_threads, p.e3.rs02_threads, p.e3.rs02_default_threads), ((1, 8, 30),) * 3)
        self.assertEqual((p.e3.repeats, p.e3.memory_limit_gb), (5, 64.0))
        self.assertEqual(p.e3.rs01_commit, "e50d69d")
        self.assertEqual(sorted((m.missing_fraction, m.structure) for m in p.e4.masks),
                         [(0.1, "clustered"), (0.1, "dispersed"), (0.3, "clustered"), (0.3, "dispersed")])
        self.assertEqual((p.e5.truth_replicates, p.e5.bootstrap_datasets, p.e5.bootstrap_replicates), (50, 10, 50))
        self.assertEqual(p.e5.trunk_bins, (500_000, 50_000))
        self.assertEqual(p.e5.genome.length, 300_000_000)

    def test_profiles(self):
        self.assertEqual(config.get_profile("production"), config.PRODUCTION)
        smoke, tiny = config.get_profile("smoke"), config.get_profile("tiny")
        self.assertLess(smoke.e1.genome.length, config.PRODUCTION.e1.genome.length)
        self.assertLess(tiny.e1.genome.length, smoke.e1.genome.length)
        self.assertEqual(smoke.e1.inference.pattern, config.INFERENCE.pattern)
        with self.assertRaises(KeyError):
            config.get_profile("nope")
        self.assertEqual((config.Budget().cores, config.Budget().mem_gb), (7, 12.0))


class Paths(unittest.TestCase):
    def test_portable_replaces_longest_prefix_first(self):
        runs = paths.ROOT / "experiment" / "runs"
        text = f"{runs}/E1/x {paths.ROOT}/target/release/psmc-rs {paths.python()}"
        self.assertEqual(paths.portable(text, runs), "<runs>/E1/x <repo>/target/release/psmc-rs <python>")

    def test_default_runs_dirs(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PSMC_EXP_RUNS", None)
            self.assertEqual(paths.default_runs_dir("production"), paths.RUNS)
        self.assertEqual(paths.default_runs_dir("smoke"), paths.SMOKE_RUNS)
        self.assertEqual(paths.default_runs_dir("tiny"), paths.SMOKE_RUNS / "tiny")

    def test_tools_dir_override(self):
        with mock.patch.dict(os.environ, {"PSMC_EXP_TOOLS": "/some/where"}):
            self.assertEqual(paths.tools_dir(), Path("/some/where").resolve())


if __name__ == "__main__":
    unittest.main()
