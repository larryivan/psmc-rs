"""The manuscript's numbers, written as LaTeX macros from the committed results.

    experiment/.venv/bin/python manuscript/make_numbers.py [--check]

reads ``experiment/runs/<EXP>/tables/*.csv``, E7's ``inputs/MANIFEST.json``,
the two runs of NA12878's bootstrap, ``experiment/crossplatform``, the
README's diagnostic and ``experiment.pipeline.config`` (the production
profile), and writes ``numbers.tex`` (one ``\\newcommand`` per number the text
quotes, with its source) and Tables 1, 2 and S1. The text quotes no result
that is not one of these, and the assertions check the claims it makes in
words; ``--check`` exits 1 when the files on disk differ from what the
results give.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiment.pipeline import config, metrics, results  # noqa: E402

RUNS = ROOT / "experiment" / "runs"
OUT = Path(__file__).resolve().parent
P = config.PRODUCTION
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


@functools.lru_cache(maxsize=None)
def table(exp: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / exp / "tables" / f"{name}.csv")


def curve(df) -> results.Curve:
    df = df.sort_values("interval")
    return results.Curve(tuple(df["years"]), tuple(df["ne"]))


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
    return f"{float(seconds) / 60.0:.1f}"


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
        "the text says both tools' curves lie inside every band"
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
    t = t[t.tool.isin(["c", "rs02"])]  # E3 also times psmc-rs 0.1, which the article does not report
    big = t[t.size_mb == max(c.genome_sizes_mb)].set_index(["tool", "threads"])
    put("EthreeBigMb", thousands(max(c.genome_sizes_mb)), "config.E3.genome_sizes_mb")
    put("EthreeCWall", thousands(big.loc[("c", 1), "wall_median_s"]), "E3_table2 c 1 thread")
    put("EthreeCMin", fix(big.loc[("c", 1), "wall_median_s"] / 60.0, 0), "E3_table2 c 1 thread (min)")
    put("EthreeRsOneWall", thousands(big.loc[("rs02", 1), "wall_median_s"]), "E3_table2 rs02 1 thread")
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
    ts = table("E3", "E3_thread_scaling")
    put_range("EthreeEff", ts[ts.threads > 1].efficiency, lambda v: fix(v, 2), "E3_thread_scaling efficiency, 2+ threads")
    runs = table("E3", "E3_runs")
    put("EthreeNodes", str(runs[runs.tool.isin(["c", "rs02"])].hostname.nunique()), "E3_runs hostname")
    put("EthreeSetups", str(len(t)), "E3_table2 rows, C psmc and psmc-rs")
    put("EthreeSpreadMax", pct(((t.wall_max_s - t.wall_min_s) / t.wall_median_s).max(), 1),
        "E3_table2 (max-min)/median wall time")
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


# E6: formats and decoding

def e6() -> None:
    f = table("E6", "E6_S2_fits")
    vcf = f[f.format == "vcf"]
    bins = table("E6", "E6_S2_bins")
    assert vcf.parameters_identical.all() and vcf.loglikes_identical.all() and bins[bins.format == "vcf"].identical.all(), \
        "the text says psmc-rs fits a genome read as psmcfa and as VCF identically"
    s = table("E6", "E6_S3_summary").set_index("bins")
    put("EsixDecodeCov", pct(s.loc["all", "coverage95_states"], 1), "E6_S3_summary coverage95_states")


# E7: 1000 Genomes

def e7() -> None:
    c = P.e7
    smp = table("E7", "E7_samples")
    put("EsevenGenomes", thousands(len(smp)), "E7_samples rows")
    put("EsevenPops", str(smp.population.nunique()), "E7_samples populations")
    put("EsevenDefault", str(int(smp["default"].sum())), "E7_samples default")
    put_range("EsevenPerPop", smp.groupby("population").size(), lambda v: str(int(v)), "E7_samples per population")
    put("EsevenDepth", fix(table("E7", "E7_inputs").raw_depth.median(), 1), "E7_inputs raw_depth median")
    shift = table("E7", "E7_route_shift")
    missing = float(shift.missing_share_of_called_bins.median())
    put("EsevenMissingShare", pct(missing, 1), "E7_route_shift missing share")
    put("EsevenThetaExpected", pct(1.0 / (1.0 - missing) - 1.0, 1), "1 / (1 - missing share) - 1")
    put("EsevenThetaShift", pct(shift.theta_rel_vcf_vs_psmcfa.mean(), 1), "E7_route_shift theta_rel mean")
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
    put("EsevenWindowHiMy", fix(hi / 1e6, 1), "E7_pop_bands all window_hi_years (My)")
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
    first, mid, older = [], [], []
    for _, gp in pb[pb.group_kind == "population"].groupby("superpopulation"):
        w = gp.pivot(index="years", columns="group", values="ne_median")
        w = w[(w.index >= lo) & (w.index <= hi)]
        sp = (w.max(axis=1) / w.min(axis=1)).apply(math.log10)
        t1 = float(first_end[list(w.columns)].max())
        assert t1 < cut
        first.append(sp[sp.index <= t1].max())
        mid.append(sp[(sp.index > t1) & (sp.index < cut)].max())
        older.append(sp[sp.index >= cut].max())
    assert all(f > m for f, m in zip(first, mid)), "the text says the populations differ most in the first window"
    two = lambda v: fix(v, 2)  # noqa: E731
    put_range("EsevenPopSpreadRecent", first, two, "E7_pop_bands populations' max log10 spread, first window", ("Lo", "Hi"))
    put_range("EsevenPopSpreadMid", mid, two, "E7_pop_bands populations' max log10 spread, after it to 150 ky", ("Lo", "Hi"))
    put("EsevenPopSpreadOldMax", two(max(older)), "E7_pop_bands populations' max log10 spread, >= 150 ky")


def e7_data_stage() -> None:
    """E7's data stage, from its MANIFEST.json."""
    man = json.loads((RUNS / "E7" / "inputs" / "MANIFEST.json").read_text())
    ch = man["chromosomes"].values()
    assert all(v["records_check"] == "equal" for v in ch)
    put("EsevenRecords", thousands(sum(v["records"] for v in ch)), "MANIFEST chromosomes records")
    put("EsevenNonpass", thousands(sum(v["nonpass"] for v in ch)), "MANIFEST chromosomes nonpass")
    (chrom, pilot), = man["pilots"].items()
    assert pilot["identical"] and all(s["psmcfa_identical"] and not s["qc_fields_differing"] for s in pilot["samples"]), \
        "the text says the pilot's direct conversion gave byte-identical files"
    put("EsevenPilotChrom", chrom.replace("chr", ""), "MANIFEST pilots")


# Numbers computed by the figures' own functions

def agreement() -> None:
    """Every interval in its window of every genome both programs fitted (Fig 1's agreement_intervals)."""
    from experiment.pipeline.figures import art1_agreement as art1
    from experiment.pipeline.figures.data import curves

    def windows(df, key):
        return {getattr(r, key): (float(r.window_lo_years), float(r.window_hi_years)) for r in df.itertuples()}

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
    """Fig S1: the EM iterations of both programs (figs1.series)."""
    from experiment.pipeline.figures import figs1

    tr = table("E1", "E1_traces")
    s = figs1.series(tr)
    gains = [g for d in s.values() for pair in d["gains"] for g in pair]
    put("EoneSoneGains", thousands(sum(len(g) for g in gains)), "Fig S1: gains drawn (genomes x programs x iterations)")
    put_range("EoneSoneGainLast", [float(g.iloc[-1]) for g in gains], sig2, "Fig S1: gains in the last iteration drawn")
    put("EoneSoneLastIter", str(max(d["N"] for d in s.values())), "Fig S1: last iteration both programs report")
    w = tr.pivot_table(index=["model", "replicate", "iteration"], columns="tool", values=["loglike", "theta", "rho"])
    first = w.xs(0, level="iteration")["loglike"]
    put("EoneSoneFirstAbsMax", sci((first["rs"] - first["c"]).abs().max()), "E1_traces: largest |LL_rs - LL_c| at the start")
    last = w["loglike"].dropna().groupby(level=["model", "replicate"]).tail(1)
    put("EoneSoneRsHigher", str(int((last["rs"] > last["c"]).sum())),
        "E1_traces: genomes whose psmc-rs log-likelihood is higher at the last common iteration")
    for col in ("theta", "rho"):
        x = w[col].dropna()
        put(f"EoneSone{col.capitalize()}Max", sci((x["rs"] / x["c"] - 1).abs().max()),
            f"E1_traces: largest relative {col} difference")


# Outside the tables

def readme() -> None:
    """The diagnostic of the residual difference between the tools (README.md, Testing and C equivalence)."""
    text = " ".join((ROOT / "README.md").read_text().split())
    pats = {
        "DiagCurve": r"bottleneck genome \(msprime, Hudson, μ = 2\.5e-8, r = 1e-8\), the curves are ([0-9.e-]+) apart",
        "DiagHookeJeeves": r"Hooke-Jeeves search in place of its L-BFGS gives a curve ([0-9.e-]+) from C's",
        "DiagLastBin": r"left out of psmc-rs's counts too, its curve is within ([0-9.e-]+) of C's",
        "DiagPrecision": r"default fit is within ([0-9.e-]+) of one whose M-steps run to the precision limit",
    }
    for name, pat in pats.items():
        m = re.search(pat, text)
        assert m, f"README.md no longer says {name}"
        put(name, sci(float(m.group(1))), f"README.md Testing and C equivalence: {name}")
    m = re.search(r"simulated (\d+) × (\d+) Mb bottleneck genome", text)
    put("DiagChroms", m.group(1), "README.md Testing and C equivalence")
    put("DiagChromMb", m.group(2), "README.md Testing and C equivalence")


def crossplatform() -> None:
    """The laptop (arm64) against the production (x86-64): experiment/crossplatform, Supplementary Note 1 section 9."""
    xp = ROOT / "experiment" / "crossplatform"
    rs = pd.read_csv(xp / "rs_laptop_vs_ibex.csv")
    put("XpRsRmseMax", sci(rs[rs.exp == "E1"].dlog10ne_rmse_window.max()), "crossplatform rs RMSE window max (E1)")
    ratio = pd.read_csv(xp / "side_by_side_E1.csv").ratio_rmse_c_xplat_over_rs_xplat
    put("XpRatioCMedian", fix(ratio.median(), 0), "side_by_side ratio C cross-build / rs cross-platform, median")


# Tables 1, 2 and S1

def tabular(columns: str, head: list, rows: list) -> str:
    return "\n".join([rf"\begin{{tabular}}{{{columns}}}", r"\toprule", *head, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"]) + "\n"


def table1() -> str:
    """Agreement between psmc-rs and C psmc on every genome both fitted (E1, E2, E7's default set)."""
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
        rows.append(f"{name} & {len(d)} & {sci(r.min())}--{sci(r.max())} & "
                    f"{sci(d.max_abs_log10_ne_rs_vs_c_per_interval.max())} & {sci(prel)} & {sci(ll)} & "
                    f"{sci(d.loglike_iteration0_rel_diff.max())} \\\\")
    head = [r"& & \multicolumn{2}{c}{$\log_{10}N_\mathrm{e}$ in the window} & \multicolumn{3}{c}{largest relative difference} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
            r"Genomes & $n$ & RMSE & largest $|\Delta|$ & $\theta,\rho,t_{\max}$ & log-likelihood & first E-step \\"]
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
        speedup = "1" if tool == "c" else (f"{sp:.1f}" if sp < 10 else f"{sp:.0f}")
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
        rows.append(f"{m} & {thousands(row.window_lo_years)} & {thousands(row.window_hi_years)} & "
                    f"{row.rmse_truth_c_mean:.3f} $\\pm$ {row.rmse_truth_c_sd:.3f} & "
                    f"{row.rmse_truth_rs_mean:.3f} $\\pm$ {row.rmse_truth_rs_sd:.3f} & "
                    f"{sci(rr.rmse_log10_ne_rs_vs_c_per_interval.median())} & "
                    f"{sci(rr.max_abs_log10_ne_rs_vs_c_per_interval.max())} \\\\")
    head = [r"& \multicolumn{2}{c}{Window (years)} & \multicolumn{2}{c}{RMSE vs truth} & \multicolumn{2}{c}{psmc-rs vs C psmc} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
            r"Model & from & to & C psmc & psmc-rs & RMSE (median) & max $|\Delta|$ \\"]
    return tabular("lrrrrrr", head, rows)


def render() -> dict:
    MACROS.clear()
    for f in (e1, e2, e3, e4, e5, e6, e7, agreement, em_traces, readme, crossplatform):
        f()
    for name, (value, src) in list(MACROS.items()):
        if value.isdigit() and int(value) < 10:
            put(name + "W", WORDS[int(value)], f"{src} (in words)")
    body = [rf"\newcommand{{\{k}}}{{{v}}}  % {src}" for k, (v, src) in MACROS.items()]
    return {"numbers.tex": "\n".join(["% Generated by manuscript/make_numbers.py from experiment/runs; do not edit.", "",
                                      *body]) + "\n",
            "table1.tex": table1(), "table2.tex": table2(), "tableS1.tex": table_s1()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the files on disk are not what the tables give")
    files = render()
    if ap.parse_args().check:
        stale = [n for n, text in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != text]
        print("stale: " + ", ".join(stale) if stale else f"{len(MACROS)} numbers and {len(files) - 1} tables up to date")
        return 1 if stale else 0
    for n, text in files.items():
        (OUT / n).write_text(text)
    print(f"wrote {len(MACROS)} numbers and " + ", ".join(n for n in files if n != "numbers.tex"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
