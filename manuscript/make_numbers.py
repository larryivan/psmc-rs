"""The manuscript's numbers, written as LaTeX macros from the committed results.

    experiment/.venv/bin/python manuscript/make_numbers.py [--check]

reads ``experiment/runs/<EXP>/tables/*.csv``, E7's ``inputs/MANIFEST.json``
and chr21 pilot (``experiment/e7_pilot``), the two runs of NA12878's
bootstrap, NA12878's ``psmcfa`` in ``experiment/real_data`` (E7's input),
``experiment/crossplatform``, ``experiment/c_equivalence`` and
``experiment.pipeline.config`` (the production profile), and writes
``numbers.tex`` (one ``\\newcommand`` per number the text quotes, with its
source) and Tables 1, 2 and S1, each only when its text changed. The text
quotes no result that is not one of these, and the assertions check the
claims it makes in words; ``--check`` writes nothing and exits 1 when the
files on disk differ from what the results give.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiment.pipeline import config  # noqa: E402

RUNS = ROOT / "experiment" / "runs"
OUT = Path(__file__).resolve().parent
P = config.PRODUCTION
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


@functools.lru_cache(maxsize=None)
def table(exp: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / exp / "tables" / f"{name}.csv")


# Formats: every macro expands in text and in math mode.

def sci(x: float, sig: int = 2) -> str:
    """2.8\\times10^{-4}."""
    x = float(x)
    if x == 0:
        return r"\ensuremath{0}"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e, sig - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return rf"\ensuremath{{{m:.{sig - 1}f}\times10^{{{e}}}}}"


def fix(x: float, d: int = 2) -> str:
    return rf"\ensuremath{{{float(x):.{d}f}}}"


def sig2(x: float) -> str:
    """Two significant figures, plain digits: 0.43, 5.9, 84."""
    return rf"\ensuremath{{{float(f'{float(x):.2g}'):g}}}"


def pct(x: float, d: int = 0) -> str:
    """A share (0.37) as a percentage (37\\%)."""
    return rf"{100.0 * float(x):.{d}f}\%"


def thousands(x: float) -> str:
    return f"{int(round(float(x))):,}"


def minutes(seconds: float) -> str:
    """A time in minutes to three significant figures, as the figures print it (74.3, 5.28, 88.0), so that the
    text's minutes give the figures' folds."""
    return f"{float(seconds) / 60.0:#.3g}"


def floor2(x: float) -> float:
    """``x`` floored, never rounded, to two significant figures (261.3 is 260, 84.5 is 84), so that a smallest
    ratio stays a bound ("at least 260 times")."""
    mag = 10.0 ** (math.floor(math.log10(x)) - 1)
    return round(math.floor(x / mag * (1.0 + 1e-12)) * mag, 12)


MACROS: dict = {}


def put(name: str, value: str, source: str) -> None:
    assert name.isalpha() and name not in MACROS, name
    MACROS[name] = (value, source)


def put_range(name: str, values, fmt, source: str, ends=("Min", "Max")) -> None:
    """``name+Min`` and ``name+Max`` (or the given ``ends``): the smallest and the largest of ``values``."""
    put(name + ends[0], fmt(min(values)), source)
    put(name + ends[1], fmt(max(values)), source)


# E1: simulated accuracy and agreement

def e1() -> None:
    c = P.e1
    put("EoneModels", str(len(c.models)), "config.E1.models")
    put("EoneReps", str(c.replicates), "config.E1.replicates")
    put("EoneGenomes", str(len(c.models) * c.replicates), "models x replicates")
    put("EoneChroms", str(c.genome.n_chromosomes), "config.E1.genome")
    put("EoneChromMb", str(c.genome.chromosome_length // 10**6), "config.E1.genome")
    put("EoneGenomeMb", str(c.genome.n_chromosomes * c.genome.chromosome_length // 10**6), "config.E1.genome")
    r = table("E1", "E1_rs_vs_c")
    put_range("EoneRsCRmse", r.rmse_log10_ne_rs_vs_c_per_interval, sci, "E1_rs_vs_c rmse")
    worst = r.loc[r.max_abs_log10_ne_rs_vs_c_any_interval.idxmax()]
    cur = table("E1", "E1_curves")
    first = cur[(cur.model == worst.model) & (cur.replicate == worst.replicate) & (cur.tool == "c") & (cur.interval == 0)]
    put("EoneRunawayC", sci(first.iloc[0]["lambda"]), "E1_curves worst replicate, C psmc's first lambda")
    # The text: "In one of the EoneGenomes simulated genomes, this parameter [the first free lambda] ran away in both
    # programs, outside the informative window: to psmc-rs's upper bound of 10^4 and, in C psmc, ... to EoneRunawayC".
    # One genome only, the one of the largest difference in any interval; every other stays below 10^3 in both.
    lam0 = cur[cur.interval == 0].pivot_table(index=["model", "replicate"], columns="tool", values="lambda")
    away = lam0[(lam0.rs >= 1e4 * (1 - 1e-9)) | (lam0.c >= 1e4)]
    assert list(away.index) == [(worst.model, worst.replicate)] and away.rs.iloc[0] <= 1e4 * (1 + 1e-9) and \
        away.c.iloc[0] > 1e4 and (lam0.drop(away.index) < 1e3).all().all(), \
        "the text says the first free lambda ran away in one genome only, to psmc-rs's bound and beyond it in C psmc"
    ends = cur[(cur.model == worst.model) & (cur.replicate == worst.replicate) & (cur.interval == 4)].years
    assert len(ends) == 2 and (ends < worst.window_lo_years).all(), \
        "the text says it ran away outside the informative window (the first free lambda ends before the window starts)"
    put("EoneLoglikeFirstRelMax", sci(r.loglike_iteration0_rel_diff.max()), "E1_rs_vs_c loglike_iteration0_rel_diff")
    # how much smaller the programs' difference is than their error against the truth (Fig 1e)
    err = table("E1", "E1_replicates").pivot_table(index=["model", "replicate"], columns="tool",
                                                    values="rmse_log10_ne_vs_truth")
    low = (err.min(axis=1) / r.set_index(["model", "replicate"]).rmse_log10_ne_rs_vs_c_per_interval).min()
    mag = 10 ** math.floor(math.log10(low)) / 10
    put("EoneFoldMin", thousands(math.floor(low / mag) * mag),
        "E1: min over replicates of (smaller error vs truth) / rmse rs vs C, floored to 2 digits")
    med = float(err["rs"].median() / r.rmse_log10_ne_rs_vs_c_per_interval.median())
    mag = 10 ** math.floor(math.log10(med))
    put("EoneFoldMedian", thousands(round(med / mag) * mag), "E1: median rs error vs truth / median rmse rs vs C (Fig 1e)")
    t = table("E1", "E1_table1").set_index("model")
    for m in ("constant", "bottleneck"):
        put(f"Eone{m.capitalize()}TruthRs", fix(t.loc[m, "rmse_truth_rs_mean"], 3), "E1_table1 rmse_truth_rs_mean")
    assert all(f"{t.loc[m, f'rmse_truth_c_{s}']:.3f}" == f"{t.loc[m, f'rmse_truth_rs_{s}']:.3f}"
               for m in c.models for s in ("mean", "sd")), \
        "the text says the two programs' errors against the truth were equal to three decimals (Table S1)"
    s1 = table("E6", "E6_S1_summary")
    assert s1.replicates_monotone.sum() == s1.replicates.sum(), "the text says the log-likelihood never decreased"


# E2: real genomes

def e2() -> None:
    r = table("E2", "E2_rs_vs_c")
    put("EtwoRsCRmseMax", sci(r.rmse_log10_ne_rs_vs_c_per_interval.max()), "E2_rs_vs_c")
    f = table("E2", "E2_fit_summary")
    rs = f[f.tool == "rs"].set_index("sample_id")
    put("EtwoHumanSequences", str(int(rs.loc["NA12878", "sequences"])), "E2_fit_summary NA12878 sequences")
    turtles = rs.drop("NA12878")
    put("EtwoTurtleSeqMax", thousands(turtles.sequences.max()), "E2_fit_summary turtle sequences")
    put("EtwoTurtles", str(len(turtles)), "E2_fit_summary turtle genomes")
    b = table("E2", "E2_bootstrap")
    put("EtwoBootReps", str(int(b.replicates.iloc[0])), "E2_bootstrap replicates")
    put("EtwoBootThreads", str(int(b.threads.iloc[0])), "E2_bootstrap threads")
    assert b.main_identical_to_fit.all(), "the text says every bootstrap refitted its main fit exactly"
    assert (b.c_in_band_fraction == 1).all() and (b.rs_in_band_fraction == 1).all(), \
        "Fig 1c-d draw both tools' curves inside psmc-rs's band everywhere"
    # Delta against the sampling uncertainty of a fit: the median width over the window of psmc-rs's bootstrap 95%
    # band (the default blocks, Fig 1c-d's band), in log10 Ne, over Delta, the smallest over the genomes
    delta = r.set_index("sample_id").rmse_log10_ne_rs_vs_c_per_interval
    band = b[b.default_trunks].set_index("sample_id")
    assert set(band.index) == set(delta.index), "a band for every genome"
    put("EtwoBandOverDeltaMin", thousands(floor2((band.band_width_log10_median / delta).min())),
        "E2_bootstrap band_width_log10_median / E2_rs_vs_c rmse, smallest over the genomes, floored to 2 digits")
    # The text and Fig 1's caption: Delta grows with the number of sequences, smallest for NA12878 and largest for
    # P. castaneus, the turtle of EtwoTurtleSeqMax sequences, whose bootstrap blocks are its whole scaffolds (one
    # block per sequence) and whose Delta is the largest of every genome both programs fitted (so is EtwoRsCRmseMax)
    seq = rs.sequences.sort_values()
    assert seq.is_unique and (delta[seq.index].diff().iloc[1:] > 0).all() and seq.index[0] == "NA12878", \
        "the text says Delta grows with the number of sequences, from NA12878"
    most = seq.index[-1]
    assert rs.loc[most, "species"] == "Pelusios castaneus" and seq.iloc[-1] == turtles.sequences.max(), \
        "the text names Pelusios castaneus as the turtle of EtwoTurtleSeqMax sequences"
    assert band.loc[most, "trunks"] == seq.iloc[-1], "Fig 1's caption says the turtle's blocks are its whole scaffolds"
    others = pd.concat([table("E1", "E1_rs_vs_c"), table("E7", "E7_rs_vs_c")]).rmse_log10_ne_rs_vs_c_per_interval
    assert delta.idxmax() == most and delta.max() > others.max(), \
        "the text says P. castaneus is the genome of the largest Delta of all (EtwoRsCRmseMax, the largest of all)"
    sp = table("E2", "E2_split_patterns")
    split = sp[sp.pattern_name != "default"]
    assert sp[sp.pattern_name == "default"].recent_peak.all() and not split.recent_peak.any(), \
        "the text says every turtle has a recent peak, and splitting the first window removed it in every fit"
    put("EtwoSplitFits", str(len(split)), "E2_split_patterns split rows")
    put_range("EtwoSplitGain", split.loglike_gain_vs_default, lambda v: fix(v, 0), "E2_split_patterns loglike_gain_vs_default")


# E3: performance

def e3() -> None:
    c = P.e3
    put("EthreeRepeats", str(c.repeats), "config.E3.repeats")
    t = table("E3", "E3_table2")
    t = t[t.tool.isin(["c", "rs02"])]  # not psmc-rs with capped M-steps (rs02-capped), when E3 times it
    big = t[t.size_mb == max(c.genome_sizes_mb)].set_index(["tool", "threads"])
    put("EthreeBigMb", thousands(max(c.genome_sizes_mb)), "config.E3.genome_sizes_mb")
    put("EthreeCWall", thousands(big.loc[("c", 1), "wall_median_s"]), "E3_table2 c 1 thread")
    put("EthreeCMin", minutes(big.loc[("c", 1), "wall_median_s"]), "E3_table2 c 1 thread (min)")
    assert big.loc[("c", 1), "wall_median_s"] > 3600, \
        "the Introduction says 25 EM iterations of C psmc on a 3 Gb genome take more than an hour"
    assert big.loc["rs02"].peak_rss_median_mb.idxmax() == 30, \
        "the abstract says psmc-rs used at most EthreeRsMemThirty MB (its 30-thread peak) on the 3,000 Mb genome"
    put("EthreeRsOneWall", thousands(big.loc[("rs02", 1), "wall_median_s"]), "E3_table2 rs02 1 thread")
    put("EthreeRsOneMin", minutes(big.loc[("rs02", 1), "wall_median_s"]), "E3_table2 rs02 1 thread (min)")
    for th, name in ((1, "One"), (8, "Eight"), (16, "Sixteen"), (30, "Thirty")):
        put(f"EthreeSpeed{name}", fix(big.loc[("rs02", th), "speedup_vs_c"], 1 if th == 1 else 0),
            f"E3_table2 rs02 {th} speedup")
    put("EthreeRsWallThirty", fix(big.loc[("rs02", 30), "wall_median_s"], 1), "E3_table2 rs02 30 wall")
    put("EthreeCMem", thousands(big.loc[("c", 1), "peak_rss_median_mb"]), "E3_table2 c peak MB")
    put("EthreeRsMemOne", fix(big.loc[("rs02", 1), "peak_rss_median_mb"], 0), "E3_table2 rs02 1 peak MB")
    put("EthreeRsMemThirty", fix(big.loc[("rs02", 30), "peak_rss_median_mb"], 0), "E3_table2 rs02 30 peak MB")
    rs = t[t.tool == "rs02"]
    put_range("EthreeSpeedOne", rs[rs.threads == 1].speedup_vs_c, lambda v: fix(v, 1),
              "E3_table2 rs02 1 thread speedup, every size")
    put_range("EthreeChromWall", rs[rs.size_mb == min(c.genome_sizes_mb)].wall_median_s, lambda v: fix(v, 2),
              "E3_table2 rs02 one 100 Mb chromosome, any threads")
    one = rs[rs.threads == 1].speedup_vs_c
    many = rs[(rs.size_mb == max(c.genome_sizes_mb)) & (rs.threads >= 8)].speedup_vs_c
    assert one.between(10, 100, inclusive="left").all() and many.between(100, 1000, inclusive="left").all(), \
        "the text says an order of magnitude faster on one thread (every size) and two orders on 8 to 30 threads"
    c_mem = t[t.tool == "c"].set_index("size_mb").peak_rss_median_mb
    assert (rs.peak_rss_median_mb <= rs.size_mb.map(c_mem) / 7).all(), \
        "the text says psmc-rs needs a small fraction (a seventh or less) of C psmc's memory, in every setup"
    ts = table("E3", "E3_thread_scaling")
    one_thread = ts.set_index("threads").wall_median_s[1]
    assert ((one_thread / ts.wall_median_s / ts.threads - ts.efficiency).abs() < 1e-9).all(), \
        "the text defines parallel efficiency as the speed-up over the one-thread time divided by the threads"
    put_range("EthreeEff", ts[ts.threads > 1].efficiency, lambda v: fix(v, 2), "E3_thread_scaling efficiency, 2+ threads")
    runs = table("E3", "E3_runs")
    put("EthreeNodes", str(runs[runs.tool.isin(["c", "rs02"])].hostname.nunique()), "E3_runs hostname")
    put("EthreeSetups", str(len(t)), "E3_table2 rows, C psmc and psmc-rs")
    put("EthreeSpreadMax", pct(((big.wall_max_s - big.wall_min_s) / big.wall_median_s).max(), 1),
        "E3_table2 (max-min)/median wall time, the 3000 Mb setups of Fig 2")
    det = table("E3", "E3_determinism")
    for tool, name in (("rs02", "Rs"), ("c", "C")):
        d = det[det.tool == tool]
        assert d.byte_identical.all(), "the text says every output was byte-identical"
        put(f"EthreeDet{name}Total", str(len(d)), f"E3_determinism {tool}")
    env = table("E3", "E3_environment").set_index("key")["value"]
    put("EthreeCpu", str(env["machine.cpu"]).replace(" 64-Core Processor", ""), "E3_environment machine.cpu")


# E4: partly callable bins

def e4() -> None:
    r = table("E4", "E4_routes")
    theta = r.groupby(["mask", "route"]).theta_rel_complete.median()
    for mask, mk in (("dispersed30", "DispThirty"), ("dispersed10", "DispTen")):
        for route, rk in (("fq2psmcfa", "Fq"), ("mask_auto", "Auto")):
            assert theta[(mask, route)] < 0, "the text says both routes underestimated theta"
            put(f"Efour{mk}{rk}Theta", pct(-theta[(mask, route)], 1), f"E4_routes {mask} {route} -theta_rel (median)")
    # The routes Fig S2 does not draw: multihetsep, whose bins psmc-rs takes as full, and the VCF without its mask
    assert (r[r.route == "multihetsep"].bin_model == "full-bins").all(), \
        "Fig S2's caption says multihetsep bins are taken as full"
    assert all(theta[(m.name, "multihetsep")] < -abs(theta[(m.name, "mask_auto")]) for m in P.e4.masks), \
        "the text says multihetsep's fits are biased as well: with every mask they underestimate theta more than " \
        "the weighted fits miss it"
    for mask, mk in (("dispersed30", "DispThirty"), ("clustered30", "ClusThirty")):
        for route, rk in (("multihetsep", "Mhs"), ("nomask", "Nomask")):
            assert theta[(mask, route)] < 0, "the text says multihetsep and the VCF without its mask underestimated theta"
            put(f"Efour{mk}{rk}Theta", pct(-theta[(mask, route)], 1), f"E4_routes {mask} {route} -theta_rel (median)")
    put("EfourReps", str(P.e4.replicates), "config.E4.replicates")
    su = table("E4", "E4_summary")
    su = su[su.route == "mask_auto"].set_index("mask").partly_callable_fraction_mean
    put("EfourDispTenPartly", pct(su["dispersed10"]), "E4_summary dispersed10 mask_auto")
    put("EfourDispThirtyPartly", pct(su["dispersed30"]), "E4_summary dispersed30 mask_auto")
    runs = {m.name: m.mean_run_bp for m in P.e4.masks}
    put("EfourClusRunKb", fix(runs["clustered30"] / 1e3, 0), "config.E4.masks mean uncallable run")
    put("EfourDispRunBp", fix(runs["dispersed30"], 0), "config.E4.masks mean uncallable run")
    put("EfourGenomeMb", str(P.e4.genome.n_chromosomes * P.e4.genome.chromosome_length // 10**6), "config.E4.genome")


# E5: bootstrap calibration

def e5() -> None:
    s = table("E5", "E5_summary")
    s = s[s.window == "informative"].set_index("trunk_mb")
    put("EfiveFits", str(int(s.fits.iloc[0])), "E5_summary fits")
    put("EfiveDatasets", str(int(s.datasets.iloc[0])), "E5_summary datasets")
    put("EfiveReps", str(int(s.replicates_per_dataset.iloc[0])), "E5_summary replicates_per_dataset")
    put("EfiveChroms", str(P.e5.genome.n_chromosomes), "config.E5.genome")
    for mb, k in ((50.0, "Fifty"), (5.0, "Five")):
        row = s.loc[mb]
        put(f"Efive{k}Sd", fix(row.sd_ratio_median, 2), "E5_summary sd_ratio_median")
        put(f"Efive{k}Cov", pct(row.coverage_mean), "E5_summary coverage_mean")
        put(f"Efive{k}CovMean", pct(row.coverage_of_fit_mean_mean), "E5_summary coverage_of_fit_mean_mean")
        put(f"Efive{k}Trunks", str(int(row.trunks)), "E5_summary trunks")
    put("EfiveBias", fix(s.abs_bias_log10_median.iloc[0], 3), "E5_summary abs_bias_log10_median")
    put("EfiveSdTrue", fix(s.sd_true_log10_median.iloc[0], 3), "E5_summary sd_true_log10_median")
    assert 1.5 <= s.abs_bias_log10_median.iloc[0] / s.sd_true_log10_median.iloc[0] < 2.5, \
        "Fig S3's caption says the fits' median bias was about twice their median SD"


# E6: formats and decoding

def e6() -> None:
    f = table("E6", "E6_S2_fits")
    vcf = f[f.format == "vcf"]
    bins = table("E6", "E6_S2_bins")
    assert vcf.parameters_identical.all() and vcf.loglikes_identical.all() and bins[bins.format == "vcf"].identical.all(), \
        "the text says psmc-rs fits a genome read as psmcfa and as VCF identically"
    assert (vcf.missing_bins == 0).all(), "the text says E6's VCFs have no uncallable positions"
    mhs = f[f.format == "multihetsep"]
    assert len(mhs) and not mhs.parameters_identical.any(), "the text says the multihetsep fits differ from the psmcfa fits"
    put("EsixMhsRmseMax", sci(mhs.rmse_log10_ne_vs_psmcfa_per_interval.max()),
        "E6_S2_fits multihetsep rmse_log10_ne_vs_psmcfa_per_interval, max over E1's first replicates")
    s = table("E6", "E6_S3_summary").set_index("bins")
    put("EsixDecodeCov", pct(s.loc["all", "coverage95_states"], 1), "E6_S3_summary coverage95_states")


# E7: 1000 Genomes

def e7() -> None:
    c = P.e7
    smp = table("E7", "E7_samples")
    put("EsevenGenomes", thousands(len(smp)), "E7_samples rows")
    put("EsevenPops", str(smp.population.nunique()), "E7_samples populations")
    put("EsevenDefault", str(int(smp["default"].sum())), "E7_samples default")
    subset = smp[smp["default"].astype(bool)]
    assert (subset.groupby("population").size() == 2).all() and \
        subset.population.nunique() == smp.population.nunique(), \
        "the text says the comparison subset has two genomes per population"
    put_range("EsevenPerPop", smp.groupby("population").size(), lambda v: str(int(v)), "E7_samples per population")
    put("EsevenDepth", fix(table("E7", "E7_inputs").raw_depth.median(), 1), "E7_inputs raw_depth median")
    shift = table("E7", "E7_route_shift")
    missing = float(shift.missing_share_of_called_bins.median())
    put("EsevenMissingShare", pct(missing, 1), "E7_route_shift missing share, median")
    put("EsevenThetaExpected", pct(1.0 / (1.0 - missing) - 1.0, 1), "1 / (1 - missing share) - 1")
    theta = float(shift.theta_rel_vcf_vs_psmcfa.median())  # the missing share's statistic
    put("EsevenThetaShift", pct(theta, 1), "E7_route_shift theta_rel median")
    assert math.isclose(theta, 1.0 / (1.0 - missing) - 1.0, rel_tol=0.1), \
        "the text says the theta shift is close to the expected"
    # The VCF fits against the psmcfa fits, interval by interval over the window and weighted as Delta: each curve
    # moves by the weighted mean of the differences (its shift) and changes shape by their weighted SD, the root of
    # rmse^2 - mean^2
    move = shift.mean_log10_ne_vcf_vs_psmcfa
    shape = (shift.rmse_log10_ne_vcf_vs_psmcfa_per_interval ** 2 - move ** 2).clip(lower=0) ** 0.5
    put("EsevenRouteShiftMedian", fix(move.median(), 3), "E7_route_shift mean_log10_ne_vcf_vs_psmcfa, median")
    put("EsevenRouteShapeMedian", fix(shape.median(), 3),
        "E7_route_shift sqrt(rmse^2 - mean^2) of log10 Ne vcf vs psmcfa (the change of shape), median")
    assert (move > 0).all() and (shape / move).max() < 1 / 3, \
        f"the text says the VCF fits moved every curve almost uniformly (largest change of shape {shape.max():.3f})"
    sp = table("E7", "E7_split")
    assert (sp.pattern_split == c.split_pattern).all() and sp.population.is_unique and \
        set(sp.population) == set(smp.population) and set(sp.sample_id) <= set(subset.sample_id), \
        "the text says one genome of the comparison subset per population, its first free lambda split into four"
    # Methods: that genome is the first of the subset in its population, its pinned genome where it has one (NA18507 in
    # YRI, NA12878 in CEU), elsewhere the first drawn (the order of kg1000.select_samples)
    first = {}
    for pop, g in subset.groupby("population"):
        pinned = [s for s in c.pinned if s in set(g.sample_id)]
        drawn = sorted((s for s in g.sample_id if s not in pinned), key=lambda s: (config.seed("E7", pop, 0, f"select/{s}"), s))
        first[pop] = (pinned + drawn)[0]
    assert dict(zip(sp.population, sp.sample_id)) == first and \
        {p: s for p, s in first.items() if s in c.pinned} == {"YRI": "NA18507", "CEU": "NA12878"}, \
        "Methods: the split genome is the first of the subset in each population: NA18507, NA12878, else the first drawn"
    span = (sp.ne_first_window_split_max / sp.ne_first_window_split_min).apply(math.log10)
    put("EsevenSplitSpanMedian", fix(span.median(), 1), "E7_split log10(max / min) of the four split lambdas' Ne, median")
    assert span.median() > 1, "the text says the four split lambdas' sizes differ more than tenfold (median)"
    rc = table("E7", "E7_rs_vs_c")
    put_range("EsevenRsCRmse", rc.rmse_log10_ne_rs_vs_c_per_interval, sci, "E7_rs_vs_c")
    real = set(table("E2", "E2_rs_vs_c").sample_id) | set(rc.sample_id)
    put("RealGenomes", str(len(real)), "E2_rs_vs_c and E7_rs_vs_c samples (NA12878 in both)")
    g = table("E7", "E7_giab").iloc[0]
    put("EsevenGiabPrecision", fix(g.precision, 4), "E7_giab precision")
    put("EsevenGiabRecall", fix(g.recall, 4), "E7_giab recall")
    put("EsevenGiabMb", thousands(g.eval_bases / 1e6), "E7_giab eval_bases (Mb)")
    tp = table("E7", "E7_throughput").set_index("route")
    put("EsevenRsCoreHours", thousands(tp.loc["rs-psmcfa", "core_hours"]), "E7_throughput rs-psmcfa core_hours")
    put("EsevenRsFitMin", minutes(tp.loc["rs-psmcfa", "wall_median_s"]), "E7_throughput rs-psmcfa wall median (min)")
    put("EsevenCFitMin", minutes(tp.loc["c", "wall_median_s"]), "E7_throughput c wall median (min)")
    put("EsevenCCoreHoursAll", thousands(round(tp.loc["c", "wall_median_s"] * len(smp) / 3600.0, -2)),
        "E7_throughput c wall median x genomes (estimate)")
    f7 = table("E7", "E7_fits")
    wall = f7.pivot_table(index="sample_id", columns="route", values="wall_seconds")[["c", "rs-psmcfa"]].dropna()
    assert len(wall) == int(smp["default"].sum()), "both programs timed on every genome of the comparison subset"
    put("EsevenPairedSpeedMedian", fix((wall["c"] / wall["rs-psmcfa"]).median(), 1),
        "E7_fits: C psmc's wall time over psmc-rs's on the same genome, median over the comparison subset")
    # Fig 2 prints the same fold in a (the one-thread speed-up) and c (the ratio of the medians): 14.1x
    med = f7.groupby("route").wall_seconds.median()
    t2 = table("E3", "E3_table2")
    a1 = t2[(t2.tool == "rs02") & (t2.threads == 1) & (t2.size_mb == max(P.e3.genome_sizes_mb))].speedup_vs_c.iloc[0]
    assert f"{med['c'] / med['rs-psmcfa']:.1f}" == f"{a1:.1f}", \
        "Fig 2 prints the same fold (14.1x) in a, on one thread, and in c, the ratio of the medians"
    bs = table("E7", "E7_bootstrap")
    put("EsevenBootReps", str(int(bs.replicates.iloc[0])), "E7_bootstrap replicates")
    put("EsevenBootThreads", str(int(bs.threads.iloc[0])), "E7_bootstrap threads")
    put("EsevenMu", sci(c.scaling.mu, 3), "config.E7.scaling")
    put("EsevenG", str(int(c.scaling.gen_years)), "config.E7.scaling")
    # NA12878's bootstrap ran twice with one seed: E2's on 50 threads, E7's on 32
    a, b = RUNS / "E2" / "bootstrap" / "NA12878" / "trunk500000", RUNS / "E7" / "boot" / "NA12878"
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir()), "the two runs wrote different files"
    put("EsevenBootFiles", str(len(names)), "E2/bootstrap and E7/boot NA12878 files")
    put("EsevenBootFilesIdentical", str(sum((a / n).read_bytes() == (b / n).read_bytes() for n in names)),
        "E2/bootstrap and E7/boot NA12878 files byte-identical")
    e7_histories()
    e7_data_stage()


def e7_histories() -> None:
    """The population histories (Fig 3): superpopulation and population medians of the psmcfa fits."""
    pb = table("E7", "E7_pop_bands")
    pb = pb[pb.route == "rs-psmcfa"]
    al = pb[pb.group_kind == "all"]
    lo, hi = float(al.window_lo_years.iloc[0]), float(al.window_hi_years.iloc[0])
    put("EsevenWindowLoKy", fix(lo / 1e3, 1), "E7_pop_bands all window_lo_years (ky)")
    put("EsevenWindowHiMy", fix(hi / 1e6, 1), "E7_pop_bands all window_hi_years (My)")
    fits = table("E7", "E7_fits")
    fits = fits[fits.route == "rs-psmcfa"]
    assert math.isclose(lo, fits.window_lo_years.median(), rel_tol=1e-9) and \
        math.isclose(hi, fits.window_hi_years.median(), rel_tol=1e-9), \
        "Fig 3's caption: the cohort's window is the medians over the genomes of each genome's window"
    wide = pb[pb.group_kind == "superpopulation"].pivot(index="years", columns="group", values="ne_median")
    wide = wide[(wide.index >= lo) & (wide.index <= hi)]
    trough = wide[(wide.index >= 1e4) & (wide.index <= 3e5)]
    non_afr = trough.drop(columns="AFR")
    put_range("EsevenNonAfrMinNe", non_afr.min(), thousands, "E7_pop_bands non-AFR superpopulation minima", ("Lo", "Hi"))
    put_range("EsevenNonAfrMinKy", non_afr.idxmin(), lambda v: fix(v / 1e3, 0), "E7_pop_bands non-AFR minima (ky)",
              ("Lo", "Hi"))
    put("EsevenAfrMinNe", thousands(trough["AFR"].min()), "E7_pop_bands AFR minimum")
    spread = (wide.max(axis=1) / wide.min(axis=1)).apply(math.log10)
    t_conv = min(t for t in wide.index if t >= 1e5 and (spread[spread.index >= t] <= math.log10(1.2)).all())
    put("EsevenConvergeKy", fix(t_conv / 1e3, 0), "E7_pop_bands: superpopulation medians within 20% from here on")
    # The populations of each superpopulation: their spread in PSMC's first time window (intervals 0-3, one
    # free lambda), whose end is the latest of its populations' (the median start of interval 4), then up to
    # 150 ky, and before.
    cut = 1.5e5
    put("EsevenPopCutKy", fix(cut / 1e3, 0), "split of recent and older times for the populations' spread")
    cur = table("E7", "E7_curves")
    c4 = cur[(cur.route == "rs-psmcfa") & (cur.interval == 4)][["sample_id", "years"]]
    first_end = c4.merge(table("E7", "E7_samples")[["sample_id", "population"]]).groupby("population").years.median()
    first, mid, older, mid_med, old_med, end, brief = [], [], [], [], [], {}, set()
    for sup, gp in pb[pb.group_kind == "population"].groupby("superpopulation"):
        w = gp.pivot(index="years", columns="group", values="ne_median")
        w = w[(w.index >= lo) & (w.index <= hi)]
        sp = (w.max(axis=1) / w.min(axis=1)).apply(math.log10)
        t1 = end[sup] = float(first_end[list(w.columns)].max())
        assert t1 < cut
        first.append(sp[sp.index <= t1].max())
        mid.append(sp[(sp.index > t1) & (sp.index < cut)].max())
        older.append(sp[sp.index >= cut].max())
        # the same spreads' medians over the times, which steps at slightly different times do not set
        mid_med.append(sp[(sp.index > t1) & (sp.index < cut)].median())
        old_med.append(sp[sp.index >= cut].median())
        # Brief: after the first window, every time at which the spread exceeds half its largest value is a lone peak
        # (at least three times the spread at each neighbouring time of the grid) across which every population median
        # steps by more than half the peak, so that the populations agree before and after it: their steps fall at
        # slightly different times.
        times, logw = list(sp.index), w.apply(lambda col: col.apply(math.log10))
        after = sp[sp.index > t1]
        peaks = [times.index(t) for t in after.index[after > after.max() / 2]]
        if all(0 < i < len(times) - 1 and sp.iloc[i] >= 3 * max(sp.iloc[i - 1], sp.iloc[i + 1]) and
               (logw.iloc[i + 1] - logw.iloc[i - 1]).abs().min() > sp.iloc[i] / 2 for i in peaks):
            brief.add(sup)
    assert all(f > m for f, m in zip(first, mid)), "the text says the populations differ most in the first window"
    assert brief == {"EAS", "EUR", "SAS"}, \
        "the text says the largest differences after the first window were brief in East Asia, Europe and South Asia, " \
        f"where the populations' steps fall at slightly different times (brief: {sorted(brief)})"
    two = lambda v: fix(v, 2)  # noqa: E731
    three = lambda v: fix(v, 3)  # noqa: E731
    put_range("EsevenPopSpreadRecent", first, two, "E7_pop_bands populations' max log10 spread, first window", ("Lo", "Hi"))
    put_range("EsevenPopSpreadMid", mid, two, "E7_pop_bands populations' max log10 spread, after it to 150 ky", ("Lo", "Hi"))
    put("EsevenPopSpreadOldMax", two(max(older)), "E7_pop_bands populations' max log10 spread, >= 150 ky")
    put_range("EsevenPopSpreadMidMed", mid_med, three,
              "E7_pop_bands populations' log10 spread after the first window to 150 ky, median over times",
              ("Lo", "Hi"))
    put("EsevenPopSpreadOldMed", three(max(old_med)),
        "E7_pop_bands populations' log10 spread >= 150 ky, median over times, largest over the superpopulations")
    # The genomes of each superpopulation outside the admixed Americas (Fig 3b-f's faint lines): the width of their
    # middle 80% in log10 Ne, the median over the times after the first window
    sup_rows = pb[pb.group_kind == "superpopulation"]
    assert (sup_rows.spread_lo_quantile == 0.1).all() and (sup_rows.spread_hi_quantile == 0.9).all(), "the middle 80%"
    smp = table("E7", "E7_samples")
    admixed = {s for s, g in smp.groupby("superpopulation") if g.admixed.all()}
    assert admixed == {"AMR"}, \
        "the text leaves out the admixed Americas, the one superpopulation of admixed populations"
    width = [(g.ne_hi / g.ne_lo).apply(math.log10)[(g.years > end[s]) & (g.years <= hi)].median()
             for s, g in sup_rows.groupby("group") if s not in admixed]
    put_range("EsevenIndivSpread", width, three, "E7_pop_bands superpopulations but AMR: log10(ne_hi / ne_lo), 10% to "
              "90% of genomes, median over the times after the first window", ("Lo", "Hi"))


def e7_data_stage() -> None:
    """E7's data stage, from its MANIFEST.json, and its chr21 pilot run with the production's binary."""
    man = json.loads((RUNS / "E7" / "inputs" / "MANIFEST.json").read_text())
    ch = man["chromosomes"].values()
    assert all(v["records_check"] == "equal" for v in ch)
    put("EsevenRecords", thousands(sum(v["records"] for v in ch)), "MANIFEST chromosomes records")
    put("EsevenNonpass", thousands(sum(v["nonpass"] for v in ch)), "MANIFEST chromosomes nonpass")
    # The chr21 pilot redone with the production's binary (MANIFEST's pilot ran the data stage's build, src 4c0d777).
    pilot = json.loads((ROOT / "experiment" / "e7_pilot" / "pilot_chr21.json").read_text())
    prod = json.loads((RUNS / "E1" / "provenance" / "fit-rs" / "bottleneck" / "rep01.json").read_text())
    # What the binary is built from (Cargo.toml's package metadata, which the pilot's checkout predates, is not).
    built = lambda tree: {k: v for k, v in (kv.split("=") for kv in tree.split(";")) if k != "Cargo.toml"}  # noqa: E731
    assert built(pilot["psmc-rs"]["source_tree"]) == built(prod["key"]["tools"]["psmc-rs-0.2"]["source_tree"]), \
        "the pilot ran a build of the production's src/ and Cargo.lock"
    assert pilot["psmc-rs"]["binary_sha256"] in prod["run"]["executables"].values(), "the pilot ran the production's binary"
    assert pilot["identical"] and all(s["psmcfa_identical"] and not s["qc_fields_differing"] for s in pilot["samples"]), \
        "the text says the pilot's direct conversion gave byte-identical files"
    put("EsevenPilotChrom", pilot["chrom"].replace("chr", ""), "experiment/e7_pilot/pilot_chr21.json")
    # A human genome's longest sequence, which bounds its speed-up on many threads: NA12878's psmcfa, E7's input
    # (E7_inputs' sha256), committed for E2; every input of the cohort has its bins, so its sequences
    na12878 = next(g for g in P.e2.genomes if g.sample_id == "NA12878")
    raw = (ROOT / "experiment" / "real_data" / na12878.file).read_bytes()
    inputs = table("E7", "E7_inputs").set_index("sample_id")
    assert hashlib.sha256(raw).hexdigest() == inputs.loc["NA12878", "psmcfa_sha256"] and inputs.bins.nunique() == 1
    bins = {}
    for record in gzip.decompress(raw).decode().split(">")[1:]:
        name, _, body = record.partition("\n")
        bins[name.split()[0]] = len(body) - body.count("\n")
    assert sum(bins.values()) == inputs.bins.iloc[0] and max(bins, key=bins.get) == "chr1", "the text: its chromosome 1"
    put("EsevenLongestMb", thousands(max(bins.values()) * P.e7.inference.bin_size / 1e6),
        "experiment/real_data NA12878 psmcfa (E7's input): longest sequence, bins x bin size (Mb)")


# Numbers computed by the figures' own functions

def agreement() -> None:
    """Every interval in its window of every genome both programs fitted (Fig 1's agreement_intervals)."""
    from experiment.pipeline.figures import art1_agreement as art1
    from experiment.pipeline.figures.data import curves, table_windows as windows

    pairs, genomes = art1.agreement_intervals(
        curves(table("E1", "E1_curves"), ["model", "tool", "replicate"]), windows(table("E1", "E1_table1"), "model"),
        curves(table("E2", "E2_curves"), ["sample_id", "tool"]), windows(table("E2", "E2_rs_vs_c"), "sample_id"),
        curves(table("E7", "E7_curves"), ["sample_id", "route"]), windows(table("E7", "E7_rs_vs_c"), "sample_id"))
    d = pd.Series([abs(math.log10(rs / c)) for k in pairs for c, rs in pairs[k]])
    assert genomes == len(P.e1.models) * P.e1.replicates + int(MACROS["RealGenomes"][0]), "the agreement's genomes"
    src = "every interval in the window of every genome both tools fitted"
    put("AgreeIntervals", thousands(len(d)), src)
    put("AgreeGenomes", str(genomes), "genomes both tools fitted (NA12878 once)")
    put("AgreeMedianAbs", sci(d.median()), f"{src}: median |log10 Ne difference|")
    put("AgreeMaxAbs", sci(d.max()), f"{src}: largest |log10 Ne difference|")
    put("AgreeMaxPct", pct(10 ** d.max() - 1, 2), f"{src}: largest relative Ne difference")


def em_traces() -> None:
    """Fig S1: the EM iterations of both programs (figs1.series; the programs' differences from E1_traces)."""
    from experiment.pipeline.figures import figs1

    tr = table("E1", "E1_traces")
    s = figs1.series(tr)
    gains = [g for d in s.values() for pair in d["gains"] for g in pair]
    put("EoneSoneGains", thousands(sum(len(g) for g in gains)), "Fig S1: gains drawn (genomes x programs x iterations)")
    put_range("EoneSoneGainLast", [float(g.iloc[-1]) for g in gains], sig2, "Fig S1: gains in the last iteration drawn")
    assert min(float(g.iloc[-1]) for g in gains) > 0.1, \
        "the Discussion says 25 EM iterations do not reach convergence: every log-likelihood still rose in the last"
    put("EoneSoneLastIter", str(max(d["N"] for d in s.values())), "Fig S1: last iteration both programs report")
    w = tr.pivot_table(index=["model", "replicate", "iteration"], columns="tool", values=["loglike", "theta", "rho"])
    first = w.xs(0, level="iteration")["loglike"]
    put("EoneSoneFirstAbsMax", sci((first["rs"] - first["c"]).abs().max()), "E1_traces: largest |LL_rs - LL_c| at the start")
    assert (first["rs"] - first["c"]).abs().max() <= 0.5e-6, \
        "Fig S1's caption says the starting model's log-likelihoods agreed within C psmc's six printed decimals"
    last = w["loglike"].dropna().groupby(level=["model", "replicate"]).tail(1)
    put("EoneSoneRsHigher", str(int((last["rs"] > last["c"]).sum())),
        "E1_traces: genomes whose psmc-rs log-likelihood is higher at the last common iteration")
    # Fig S1's lower row, LL_rs(i) - LL_c(i) after each iteration i >= 1 both programs report, against the gains of
    # the upper one, LL(i) - LL(i - 1), as figs1.series takes them (computed here from the table)
    ll = w["loglike"].dropna()
    gain = ll.groupby(level=["model", "replicate"]).diff().dropna()
    diff = (ll["rs"] - ll["c"]).abs().drop(0, level="iteration")
    assert (gain > 0).all().all(), "the text says the log-likelihood never decreased"
    put("EoneSoneDiffMax", sig2(diff.max()), "E1_traces: largest |LL_rs - LL_c| after any iteration")
    by = ["model", "iteration"]
    ratio = gain.min(axis=1).groupby(level=by).min() / diff.groupby(level=by).max()
    put("EoneSoneRatioMin", thousands(floor2(ratio.min())),
        "E1_traces: a model's smallest gain in iteration i over its largest |LL_rs - LL_c| after i, smallest over the "
        "models and iterations (Fig S1's panels), floored to 2 digits")
    # theta and rho: the traces, at C psmc's six printed decimals, and the final fits at full precision
    r = table("E1", "E1_rs_vs_c")
    for col in ("theta", "rho"):
        x = w[col].dropna()
        rel = max((x["rs"] / x["c"] - 1).abs().max(), r[f"{col}_rel_diff"].abs().max())
        put(f"EoneSone{col.capitalize()}Max", sci(rel),
            f"E1_traces and E1_rs_vs_c {col}_rel_diff: largest relative {col} difference, any iteration")


# Outside the tables

def c_equivalence() -> None:
    """The diagnostic of the residual difference between the tools on E1's genomes: experiment/c_equivalence."""
    s = json.loads((ROOT / "experiment" / "c_equivalence" / "summary.json").read_text())
    v, src = s["variants"], "c_equivalence/summary.json"
    rmse = table("E1", "E1_rs_vs_c").rmse_log10_ne_rs_vs_c_per_interval
    assert s["genomes"] == len(rmse) and math.isclose(v["default"]["rmse_vs_c"]["median"], rmse.median(), rel_tol=1e-9), \
        "the default is E1's fits"
    side = pd.read_csv(ROOT / "experiment" / "crossplatform" / "side_by_side_E1.csv")
    cb = s["c_builds_rmse"]["median"]
    assert math.isclose(cb, side.rmse_window_c_laptop_vs_ibex.median(), rel_tol=1e-12), \
        "two builds of C psmc as in REPRODUCIBILITY.md, section 9"
    lb, hj = v["lastbin"], v["hj"]
    noise = v["default"]["rmse_vs_c"]["median"] / cb
    assert 1 < noise < 10, "the text says the median Delta is a few times C psmc's own noise, its fits on two platforms"
    put("DiagOverCBuilds", str(math.floor(noise + 0.5)),
        f"{src} default rmse_vs_c median / c_builds_rmse median, rounded: the median Delta in units of C psmc's own noise")
    assert lb["rmse_vs_c"]["median"] < cb, "the text says the median with the last bin left out is below the C builds'"
    assert all(g["default"] <= g["c_builds"] for g in s["lastbin_not_closer"]), \
        "the text says the genomes the last bin does not bring closer are as close as the two builds of C psmc"
    assert hj["rmse_vs_c"]["median"] > 0.9 * v["default"]["rmse_vs_c"]["median"], "the text says the optimizer does not matter"
    put("DiagMedianDefault", sci(v["default"]["rmse_vs_c"]["median"]), f"{src} default rmse_vs_c median")
    put("DiagMedianLastBin", sci(lb["rmse_vs_c"]["median"]), f"{src} lastbin rmse_vs_c median")
    put("DiagMedianHookeJeeves", sci(hj["rmse_vs_c"]["median"]), f"{src} hj rmse_vs_c median")
    put("DiagCBuildsMedian", sci(cb), f"{src} c_builds_rmse median")
    put("DiagLastBinCloser", str(lb["closer_to_c_than_default"]), f"{src} lastbin closer_to_c_than_default")
    put("DiagLastBinOther", str(s["genomes"] - lb["closer_to_c_than_default"]), f"{src} lastbin genomes not closer")
    put("DiagPrecisionMedian", sci(v["prec"]["rmse_vs_default"]["median"]), f"{src} prec rmse_vs_default median")
    put("DiagPrecisionMax", sci(v["prec"]["rmse_vs_default"]["max"]), f"{src} prec rmse_vs_default max")


def crossplatform() -> None:
    """The laptop (arm64) against the production (x86-64): experiment/crossplatform, REPRODUCIBILITY.md section 9."""
    xp = ROOT / "experiment" / "crossplatform"
    rs = pd.read_csv(xp / "rs_laptop_vs_ibex.csv")
    toolchain = (xp / "toolchain.txt").read_text()
    laptop = dict(re.findall(r"(src|Cargo\.toml|Cargo\.lock)=([0-9a-f]+)", toolchain))  # the rebuild's sources
    ibex = {json.loads((RUNS / j["ibex_record"].removeprefix("experiment/runs/")).read_text())["key"]["tools"]["psmc-rs-0.2"]["source_tree"]
            for j in json.loads((xp / "jobs.json").read_text())}
    assert ibex == {";".join(f"{k}={v}" for k, v in laptop.items())} and "dirty_files=0" in toolchain, \
        "the laptop's rebuild is of the production's sources"
    # The builds that ran the laptop's fits: what the binary is built from (src/, Cargo.lock) is the production's, and
    # each one's digest without Mach-O UUID and code signature is the rebuild's.
    fits = re.findall(r"from src ([0-9a-f]+) Cargo\.toml [0-9a-f]+ Cargo\.lock ([0-9a-f]+)", toolchain)
    same = re.findall(r"sha256 without LC_UUID and code signature: [0-9a-f]+, this build's", toolchain)
    assert fits and len(same) == len(fits) and all(f"src={s};" in t and t.endswith(f"Cargo.lock={lk}") for s, lk in fits
                                                   for t in ibex), "the text says the laptop refitted from the production's sources"
    assert len(rs) == P.e1.replicates * len(P.e1.models) + 1 and set(rs.exp) == {"E1", "E2"}, "E1's genomes and NA12878"
    side = pd.read_csv(xp / "side_by_side_E1.csv")
    assert (side.rmse_window_rs_laptop_vs_ibex < side.rmse_window_c_laptop_vs_ibex).all(), \
        "the text says psmc-rs's curves agree across the platforms more closely than two builds of C psmc"
    put("XpRsRmseMax", sci(rs[rs.exp == "E1"].dlog10ne_rmse_window.max()), "crossplatform rs RMSE window max (E1)")
    # the largest relative difference of any parameter (theta, rho, t_max, every lambda), as REPRODUCIBILITY.md,
    # section 9; NA12878's refit is within it too
    rel = rs[rs.exp == "E1"].param_max_rel.max()
    assert (rs.param_max_rel <= rel).all() and \
        math.isclose(rel, side.param_max_rel_rs_laptop_vs_ibex.max(), rel_tol=1e-12), "side_by_side_E1's; NA12878's below"
    put("XpParamRelMax", sci(rel),
        "crossplatform rs param_max_rel max (E1): largest relative difference of any parameter")
    put("XpRatioCMedian", fix(side.ratio_rmse_c_xplat_over_rs_xplat.median(), 0), "side_by_side ratio C cross-build / rs cross-platform, median")


# Tables 1, 2 and S1

def tabular(columns: str, head: list, rows: list) -> str:
    return "\n".join([rf"\begin{{tabular}}{{{columns}}}", r"\toprule", *head, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"]) + "\n"


def table1() -> str:
    """Agreement between psmc-rs and C psmc on every genome both fitted (E1, E2, E7's comparison subset)."""
    e1, e2, e7 = table("E1", "E1_rs_vs_c"), table("E2", "E2_rs_vs_c"), table("E7", "E7_rs_vs_c")

    def params_rel(df):
        return df[["theta_rel_diff", "rho_rel_diff", "t_max_rel_diff"]].abs().max().max()

    def loglike_rel(wide, rs, c):
        return ((wide[rs] - wide[c]).abs() / wide[c].abs()).max()

    f2 = table("E2", "E2_fit_summary").pivot_table(index="sample_id", columns="tool",
                                                     values=["theta", "rho", "t_max", "loglike_common"])
    rel2 = max(((f2[k]["rs"] - f2[k]["c"]).abs() / f2[k]["c"].abs()).max() for k in ("theta", "rho", "t_max"))
    f7 = table("E7", "E7_fits")
    w7 = f7[f7.route.isin(["c", "rs-psmcfa"])].pivot_table(index="sample_id", columns="route", values="loglike_common")
    w1 = table("E1", "E1_replicates").pivot_table(index=["model", "replicate"], columns="tool", values="loglike_common")
    rows = []
    for name, d, prel, ll in ((r"Simulated (E1)", e1, params_rel(e1), loglike_rel(w1, "rs", "c")),
                              (r"Turtles and NA12878 (E2)", e2, rel2, loglike_rel(f2["loglike_common"], "rs", "c")),
                              (r"1000 Genomes (E7)", e7, params_rel(e7), loglike_rel(w7, "rs-psmcfa", "c"))):
        r = d.rmse_log10_ne_rs_vs_c_per_interval
        rows.append(f"{name} & {len(d)} & {sci(r.min())} to {sci(r.max())} & "
                    f"{sci(d.max_abs_log10_ne_rs_vs_c_per_interval.max())} & {sci(prel)} & {sci(ll)} & "
                    f"{sci(d.loglike_iteration0_rel_diff.max())} \\\\")
    head = [r"& & \multicolumn{2}{c}{$\log_{10}N_\mathrm{e}$ in the window} & \multicolumn{3}{c}{largest relative difference} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
            r"Genomes & $n$ & $\Delta$ & largest $|d_k|$ & $\theta,\rho,t_{\max}$ & log-likelihood & first E-step \\"]
    return tabular("lrrrrrr", head, rows)


def table2() -> str:
    t = table("E3", "E3_table2")
    sizes = sorted(P.e3.genome_sizes_mb)
    n = len(sizes)
    rows = []
    for tool, th in [("c", 1)] + [("rs02", th) for th in (1, 2, 4, 8, 16, 30)]:
        sub = t[(t.tool == tool) & (t.threads == th)].set_index("size_mb")
        if sub.empty:
            continue
        cells = [f"{sub.loc[s, 'wall_median_s']:,.1f}" if s in sub.index else "--" for s in sizes]
        big = sub.loc[max(sizes)] if max(sizes) in sub.index else None
        sp = big.speedup_vs_c if big is not None else float("nan")
        mem = big.peak_rss_median_mb if big is not None else float("nan")
        speedup = "1" if tool == "c" else (f"{sp:.1f}" if sp < 100 else f"{sp:.0f}")  # the text's precision
        label = "C psmc" if tool == "c" else "psmc-rs"
        rows.append(f"{label} & {th} & " + " & ".join(cells) + f" & {speedup} & {mem:,.0f} \\\\")
    head = [rf"& & \multicolumn{{{n}}}{{c}}{{Wall time, median (s)}} & \multicolumn{{2}}{{c}}{{{thousands(max(sizes))} Mb}} \\",
            rf"\cmidrule(lr){{3-{2 + n}}}\cmidrule(lr){{{3 + n}-{4 + n}}}",
            "Tool & Threads & " + " & ".join(f"{thousands(s)} Mb" for s in sizes) + r" & Speed-up & Memory (MB) \\"]
    return tabular("lr" + "r" * n + "rr", head, rows)


def table_s1() -> str:
    t = table("E1", "E1_table1").set_index("model")
    r = table("E1", "E1_rs_vs_c")
    rows = []
    for m in P.e1.models:
        row, rr = t.loc[m], r[r.model == m]
        rows.append(f"{m.capitalize()} & {thousands(row.window_lo_years)} & {thousands(row.window_hi_years)} & "
                    f"{row.rmse_truth_c_mean:.3f} $\\pm$ {row.rmse_truth_c_sd:.3f} & "
                    f"{row.rmse_truth_rs_mean:.3f} $\\pm$ {row.rmse_truth_rs_sd:.3f} & "
                    f"{sci(rr.rmse_log10_ne_rs_vs_c_per_interval.median())} & "
                    f"{sci(rr.max_abs_log10_ne_rs_vs_c_per_interval.max())} \\\\")
    head = [r"& \multicolumn{2}{c}{Window (years)} & \multicolumn{2}{c}{RMSE vs truth} & \multicolumn{2}{c}{psmc-rs vs C psmc} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Model & from & to & C psmc & psmc-rs & $\Delta$ (median) & max $|d_k|$ \\"]
    return tabular("lrrrrrr", head, rows)


def render() -> dict:
    MACROS.clear()
    for f in (e1, e2, e3, e4, e5, e6, e7, agreement, em_traces, c_equivalence, crossplatform):
        f()
    for name, (value, src) in list(MACROS.items()):
        if value.isdigit() and int(value) < 10:
            put(name + "W", WORDS[int(value)], f"{src} (in words)")
    body = [rf"\newcommand{{\{k}}}{{{v}}}  % {src}" for k, (v, src) in MACROS.items()]
    return {"numbers.tex": "\n".join(["% Generated by manuscript/make_numbers.py from experiment/runs; do not edit.", "",
                                      *body]) + "\n",
            "table1.tex": table1(), "table2.tex": table2(), "tableS1.tex": table_s1()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the files on disk are not what the tables give")
    check = ap.parse_args().check
    files = render()
    stale = [n for n, text in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != text]
    if not stale:
        print(f"{len(MACROS)} numbers and {len(files) - 1} tables up to date")
    elif check:
        print("stale: " + ", ".join(stale))
    else:
        for n in stale:  # only these, so that make rebuilds only the PDFs whose numbers or tables changed
            (OUT / n).write_text(files[n])
        print("wrote " + ", ".join(stale))
    return 1 if stale and check else 0


if __name__ == "__main__":
    sys.exit(main())
