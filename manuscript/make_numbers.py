"""The manuscript's numbers, written as LaTeX macros from the committed results.

    experiment/.venv/bin/python manuscript/make_numbers.py

reads ``experiment/runs/<EXP>/tables/*.csv``, E7's ``inputs/MANIFEST.json``,
the two runs of NA12878's bootstrap and ``experiment.pipeline.config`` (the
production profile), and writes ``manuscript/numbers.tex`` (one
``\\newcommand`` per number the text quotes, with its source), and Tables 1,
and S1 (``table1.tex``, ``table2.tex``, ``tableS1.tex``). The text quotes no
result that is not one of these, so it cannot drift from the runs;
``--check`` exits 1 when the files on disk differ from what the results give.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from experiment.pipeline import config  # noqa: E402

RUNS = ROOT / "experiment" / "runs"
OUT = Path(__file__).resolve().parent
P = config.PRODUCTION


def table(exp: str, name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS / exp / "tables" / f"{name}.csv")


# ---------------------------------------------------------------------------
# Formats (every macro expands in text and in math mode)
# ---------------------------------------------------------------------------


def sci(x: float, sig: int = 2) -> str:
    """2.8\\times10^{-4}; plain digits between 0.01 and 1000."""
    x = float(x)
    if x == 0:
        return r"\ensuremath{0}"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e, sig - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    ms = f"{m:.{sig - 1}f}"
    return rf"\ensuremath{{{ms}\times10^{{{e}}}}}"


def fix(x: float, d: int = 2) -> str:
    return rf"\ensuremath{{{float(x):.{d}f}}}"


def pct(x: float, d: int = 0) -> str:
    """A share (0.37) as a percentage (37\\%)."""
    return rf"{100.0 * float(x):.{d}f}\%"


def thousands(x: float) -> str:
    return f"{int(round(float(x))):,}"


def mmss(seconds: float) -> str:
    return f"{float(seconds) / 60.0:.1f}"


MACROS: dict = {}


def put(name: str, value: str, source: str) -> None:
    assert name.isalpha(), name
    assert name not in MACROS, name
    MACROS[name] = (value, source)


# ---------------------------------------------------------------------------
# E1: simulated accuracy and equivalence
# ---------------------------------------------------------------------------


def e1() -> None:
    c = P.e1
    put("EoneModels", str(len(c.models)), "config.E1.models")
    put("EoneReps", str(c.replicates), "config.E1.replicates")
    put("EoneGenomes", str(len(c.models) * c.replicates), "models x replicates")
    put("EoneChroms", str(c.genome.n_chromosomes), "config.E1.genome")
    put("EoneChromMb", str(c.genome.chromosome_length // 10**6), "config.E1.genome")
    put("EoneGenomeMb", str(c.genome.n_chromosomes * c.genome.chromosome_length // 10**6), "config.E1.genome")
    r = table("E1", "E1_rs_vs_c")
    put("EoneRsCRmseMin", sci(r.rmse_log10_ne_rs_vs_c_per_interval.min()), "E1_rs_vs_c min rmse")
    put("EoneRsCRmseMax", sci(r.rmse_log10_ne_rs_vs_c_per_interval.max()), "E1_rs_vs_c max rmse")
    put("EoneRsCRmseMedian", sci(r.rmse_log10_ne_rs_vs_c_per_interval.median()), "E1_rs_vs_c median rmse")
    put("EoneRsCMaxAbs", sci(r.max_abs_log10_ne_rs_vs_c_per_interval.max()), "E1_rs_vs_c max abs in window")
    worst = r.loc[r.max_abs_log10_ne_rs_vs_c_any_interval.idxmax()]
    put("EoneAnyMax", fix(worst.max_abs_log10_ne_rs_vs_c_any_interval, 1), "E1_rs_vs_c max abs any interval")
    put("EoneAnyModel", str(worst.model), "E1_rs_vs_c worst any interval")
    put("EoneAnyReplicate", str(int(worst.replicate)), "E1_rs_vs_c worst any interval")
    second = r.max_abs_log10_ne_rs_vs_c_any_interval.drop(worst.name).max()
    put("EoneAnySecond", sci(second), "E1_rs_vs_c second largest any interval")
    cur = table("E1", "E1_curves")
    cur = cur[(cur.model == worst.model) & (cur.replicate == worst.replicate)]
    put("EoneRunawayC", sci(cur[(cur.tool == "c") & (cur.interval == 0)].iloc[0]["lambda"]),
        "E1_curves worst replicate, C psmc's first lambda")
    put("EoneRunawayCNe", sci(cur[(cur.tool == "c") & (cur.interval == 0)].iloc[0]["ne"]),
        "E1_curves worst replicate, C psmc's Ne in the first interval")
    put("EoneRunawayKy", fix(cur[(cur.tool == "c") & (cur.interval == 4)].iloc[0].years / 1e3, 1),
        "E1_curves worst replicate, end of the first free lambda (ky)")
    params = r[["theta_rel_diff", "rho_rel_diff", "t_max_rel_diff"]].abs().max().max()
    put("EoneParamRelMax", sci(params), "E1_rs_vs_c max |rel diff| theta, rho, t_max")
    put("EoneLoglikeRelMax", sci(r.loglike_max_rel_diff.max()), "E1_rs_vs_c loglike_max_rel_diff (any iteration)")
    reps = table("E1", "E1_replicates").pivot_table(index=["model", "replicate"], columns="tool", values="loglike_common")
    put("EoneLoglikeLastRelMax", sci(((reps["rs"] - reps["c"]).abs() / reps["c"].abs()).max()),
        "E1_replicates loglike_common (the last iteration both report)")
    put("EoneLoglikeFirstRelMax", sci(r.loglike_iteration0_rel_diff.max()), "E1_rs_vs_c loglike_iteration0_rel_diff")
    rr = table("E1", "E1_replicates").pivot_table(index=["model", "replicate"], columns="tool",
                                                   values="rmse_log10_ne_vs_truth")
    fold = rr.min(axis=1) / r.set_index(["model", "replicate"]).rmse_log10_ne_rs_vs_c_per_interval
    low = fold.min()
    mag = 10 ** math.floor(math.log10(low)) / 10
    put("EoneFoldMin", thousands(math.floor(low / mag) * mag),
        "E1: min over replicates of (smaller of the two errors vs truth) / rmse rs vs C, floored to 2 digits")
    med = float(rr["rs"].median() / r.rmse_log10_ne_rs_vs_c_per_interval.median())
    mag = 10 ** math.floor(math.log10(med))
    put("EoneFoldMedian", thousands(round(med / mag) * mag),
        "E1: median rs error vs truth / median rmse rs vs C, to 1 digit (Fig 1j's arrow)")
    t = table("E1", "E1_table1").set_index("model")
    for m in c.models:
        k = m.capitalize()
        put(f"Eone{k}TruthC", fix(t.loc[m, "rmse_truth_c_mean"], 3), "E1_table1 rmse_truth_c_mean")
        put(f"Eone{k}TruthRs", fix(t.loc[m, "rmse_truth_rs_mean"], 3), "E1_table1 rmse_truth_rs_mean")
    s1 = table("E6", "E6_S1_summary")
    put("EoneMonotoneRuns", str(int(s1.replicates_monotone.sum())), "E6_S1_summary replicates_monotone")
    put("EoneRuns", str(int(s1.replicates.sum())), "E6_S1_summary replicates")
    assert int(s1.replicates_monotone.sum()) == int(s1.replicates.sum()), "the text says the log-likelihood never decreased"


# ---------------------------------------------------------------------------
# E2: real genomes
# ---------------------------------------------------------------------------


def e2() -> None:
    r = table("E2", "E2_rs_vs_c").set_index("sample_id")
    put("EtwoRsCRmseMin", sci(r.rmse_log10_ne_rs_vs_c_per_interval.min()), "E2_rs_vs_c")
    put("EtwoRsCRmseMax", sci(r.rmse_log10_ne_rs_vs_c_per_interval.max()), "E2_rs_vs_c")
    put("EtwoRsCMaxAbs", sci(r.max_abs_log10_ne_rs_vs_c_per_interval.max()), "E2_rs_vs_c")
    put("EtwoHumanRmse", sci(r.loc["NA12878", "rmse_log10_ne_rs_vs_c_per_interval"]), "E2_rs_vs_c NA12878")
    f = table("E2", "E2_fit_summary")
    rel, ll = 0.0, 0.0
    for _, g in f.groupby("sample_id"):
        cc, rs = g[g.tool == "c"].iloc[0], g[g.tool == "rs"].iloc[0]
        rel = max(rel, *(abs(rs[k] - cc[k]) / abs(cc[k]) for k in ("theta", "rho", "t_max")))
        ll = max(ll, abs(rs.loglike_common - cc.loglike_common) / abs(cc.loglike_common))
    put("EtwoParamRelMax", sci(rel), "E2_fit_summary theta, rho, t_max")
    put("EtwoLoglikeRelMax", sci(ll), "E2_fit_summary loglike_common")
    put("EtwoLoglikeFirstRelMax", sci(r.loglike_iteration0_rel_diff.max()), "E2_rs_vs_c loglike_iteration0_rel_diff")
    seq = f[(f.sample_id == "NA12878") & (f.tool == "rs")].iloc[0]
    put("EtwoHumanSequences", str(int(seq.sequences)), "E2_fit_summary NA12878 sequences")
    turtles = f[(f.tool == "rs") & (f.sample_id != "NA12878")].sequences
    put("EtwoTurtleSeqMin", thousands(turtles.min()), "E2_fit_summary turtle sequences")
    put("EtwoTurtleSeqMax", thousands(turtles.max()), "E2_fit_summary turtle sequences")
    b = table("E2", "E2_bootstrap").set_index("sample_id")
    put("EtwoHumanTrunks", str(int(b.loc["NA12878", "trunks"])), "E2_bootstrap NA12878 trunks")
    put("EtwoBootReps", str(int(b.replicates.iloc[0])), "E2_bootstrap replicates")
    put("EtwoBootIdentical", "all" if bool(b.main_identical_to_fit.all()) else "not all", "E2_bootstrap main_identical_to_fit")
    put("EtwoBootThreads", str(int(b.threads.iloc[0])), "E2_bootstrap threads")
    both_in = bool((b.c_in_band_fraction == 1.0).all() and (b.rs_in_band_fraction == 1.0).all())
    assert both_in, "the text says both tools' curves lie inside every band"
    put("EtwoBandWidthMin", fix(b.band_width_log10_median.min(), 2), "E2_bootstrap band_width_log10_median")
    put("EtwoBandWidthMax", fix(b.band_width_log10_median.max(), 2), "E2_bootstrap band_width_log10_median")
    sp = table("E2", "E2_split_patterns")
    d = sp[sp.pattern_name == "default"]
    put("EtwoPeakMin", fix(d.recent_max_vs_plateau.min(), 1), "E2_split_patterns default recent_max_vs_plateau")
    put("EtwoPeakMax", fix(d.recent_max_vs_plateau.max(), 1), "E2_split_patterns default recent_max_vs_plateau")
    put("EtwoPeaksDefault", str(int(d.recent_peak.sum())), "E2_split_patterns default recent_peak")
    s = sp[sp.pattern_name != "default"]
    put("EtwoPeaksSplit", str(int(s.recent_peak.sum())), "E2_split_patterns split recent_peak")
    put("EtwoSplitFits", str(len(s)), "E2_split_patterns split rows")
    put("EtwoSplitGainMin", fix(s.loglike_gain_vs_default.min(), 0), "E2_split_patterns loglike_gain_vs_default")
    put("EtwoSplitGainMax", fix(s.loglike_gain_vs_default.max(), 0), "E2_split_patterns loglike_gain_vs_default")
    # the older history compared at the same times (metrics.log_time_diff): the split fits' theta differs from
    # the default's by percents, so matching intervals would compare different times
    from experiment.pipeline import metrics, results
    from experiment.pipeline.experiments import e2 as e2mod
    cur = table("E2", "E2_split_curves")

    def curve(df):
        df = df.sort_values("interval")
        return results.Curve(tuple(df["years"]), tuple(df["ne"]))

    older, theta_shift = [], []
    for sid, g in cur.groupby("sample_id"):
        d = g[g.pattern_name == "default"]
        row = sp[(sp.sample_id == sid) & (sp.pattern_name == "default")].iloc[0]
        start = float(d.sort_values("interval")["years"].iloc[P.e2.split_recent_intervals])
        for pn in sorted(set(g.pattern_name) - {"default"}):
            older.append(metrics.log_time_diff(curve(g[g.pattern_name == pn]), curve(d),
                                               (max(start, row.window_lo_years), row.window_hi_years)).rmse)
            theta_shift.append(abs(sp[(sp.sample_id == sid) & (sp.pattern_name == pn)].theta.iloc[0] / row.theta - 1))
    put("EtwoSplitOlderMax", sci(max(older)), "E2_split_curves: split vs default after the recent intervals, same times")
    put("EtwoSplitThetaMax", pct(max(theta_shift), 1), "E2_split_patterns theta of split vs default")

    def expand(pattern, free):
        spans = []
        for tok in pattern.split("+"):
            a, _, b2 = tok.partition("*")
            spans += [int(b2)] * int(a) if b2 else [int(a)]
        assert len(spans) == len(free)
        return [v for s2, v in zip(spans, free) for _ in range(s2)]

    br = table("E2", "E2_bootstrap_replicates")
    peaks = {}
    for sid, g in br.groupby("sample_id"):
        n = 0
        for _, rr in g.iterrows():
            lam = expand(rr.pattern, [float(x) for x in rr.lam.split()])
            c = results.Curve(tuple(range(len(lam))), tuple(lam))  # Ne is proportional to lambda: the ratios are Ne's
            n += bool(e2mod.recent_shape(c, P.e2.split_recent_intervals, P.e2.split_peak_ratio)["recent_peak"])
        peaks[sid] = n
    put("EtwoBootPeakEmy", str(peaks["HLemySub1"]), "E2_bootstrap_replicates recent_peak, E. subglobosa")
    put("EtwoBootPeakHyd", str(peaks["HLhydTec1"]), "E2_bootstrap_replicates recent_peak, H. tectifera")
    put("EtwoBootPeakPel", str(peaks["HLpelCas1"]), "E2_bootstrap_replicates recent_peak, P. castaneus")
    assert peaks["HLhydTec1"] == peaks["HLpelCas1"] == int(b.replicates.iloc[0]), \
        "the Discussion says every replicate of two turtles has the peak"
    put("EtwoTurtles", str(f[f.sample_id != "NA12878"].sample_id.nunique()), "E2_fit_summary turtle genomes")


# ---------------------------------------------------------------------------
# E3: performance
# ---------------------------------------------------------------------------


def e3() -> None:
    c = P.e3
    put("EthreeRepeats", str(c.repeats), "config.E3.repeats")
    t = table("E3", "E3_table2")
    big = t[t.size_mb == max(c.genome_sizes_mb)]

    def row(tool, threads):
        return big[(big.tool == tool) & (big.threads == threads)].iloc[0]

    put("EthreeBigMb", thousands(max(c.genome_sizes_mb)), "config.E3.genome_sizes_mb")
    put("EthreeBigChroms", str(int(row("c", 1).chromosomes)), "E3_table2 chromosomes")
    put("EthreeCWall", thousands(row("c", 1).wall_median_s), "E3_table2 c 1 thread")
    put("EthreeCMin", fix(row("c", 1).wall_median_s / 60.0, 0), "E3_table2 c 1 thread (min)")
    put("EthreeRsOneWall", thousands(row("rs02", 1).wall_median_s), "E3_table2 rs02 1 thread")
    for th, name in ((1, "One"), (8, "Eight"), (16, "Sixteen"), (30, "Thirty")):
        put(f"EthreeSpeed{name}", fix(row("rs02", th).speedup_vs_c, 1 if th == 1 else 0), f"E3_table2 rs02 {th} speedup")
        put(f"EthreeRsWall{name}", fix(row("rs02", th).wall_median_s, 1 if th > 1 else 0), f"E3_table2 rs02 {th} wall")
    put("EthreeCMem", thousands(row("c", 1).peak_rss_median_mb), "E3_table2 c peak MB")
    put("EthreeRsMemOne", fix(row("rs02", 1).peak_rss_median_mb, 0), "E3_table2 rs02 1 peak MB")
    put("EthreeRsMemThirty", fix(row("rs02", 30).peak_rss_median_mb, 0), "E3_table2 rs02 30 peak MB")
    one = t[(t.tool == "rs02") & (t.threads == 1)]
    put("EthreeSpeedOneMin", fix(one.speedup_vs_c.min(), 1), "E3_table2 rs02 1 thread speedup, every size")
    put("EthreeSpeedOneMax", fix(one.speedup_vs_c.max(), 1), "E3_table2 rs02 1 thread speedup, every size")
    small = t[(t.tool == "rs02") & (t.size_mb == min(c.genome_sizes_mb))]
    put("EthreeChromWall", fix(small[small.threads == 1].wall_median_s.iloc[0], 1), "E3_table2 rs02 one 100 Mb chromosome, 1 thread")
    put("EthreeChromWallMin", fix(small.wall_median_s.min(), 2), "E3_table2 rs02 100 Mb, any threads")
    put("EthreeChromWallMax", fix(small.wall_median_s.max(), 2), "E3_table2 rs02 100 Mb, any threads")
    rs = t[t.tool == "rs02"]
    rounds = (rs.chromosomes / rs.threads).apply(math.ceil)
    per_round = rs.wall_median_s / rounds
    put("EthreeRoundMin", fix(per_round.min(), 1), "E3_table2 rs02 wall / ceil(chromosomes / threads)")
    put("EthreeRoundMax", fix(per_round.max(), 1), "E3_table2 rs02 wall / ceil(chromosomes / threads)")
    ts = table("E3", "E3_thread_scaling")
    put("EthreeEffMin", fix(ts.efficiency.min(), 2), "E3_thread_scaling efficiency")
    put("EthreeEffMax", fix(ts[ts.threads > 1].efficiency.max(), 2), "E3_thread_scaling efficiency")
    n = int(row("rs02", 1).chromosomes)
    cap = min(n / (th * math.ceil(n / th)) for th in ts.threads if th > 1)
    put("EthreeEffCap", fix(cap, 2), "whole chromosomes per thread: min over t of n / (t ceil(n / t))")
    runs = table("E3", "E3_runs")
    runs = runs[runs.tool.isin(["c", "rs02"])]
    put("EthreeRuns", str(len(runs)), "E3_runs rows, C psmc and psmc-rs 0.2")
    put("EthreeCleanRuns", str(int(runs.clean.sum())), "E3_runs clean, C psmc and psmc-rs 0.2")
    put("EthreeNodes", str(runs.hostname.nunique()), "E3_runs hostname, C psmc and psmc-rs 0.2")
    put("EthreeSetups", str(len(t[t.tool.isin(["c", "rs02"])])), "E3_table2 rows, C psmc and psmc-rs 0.2")
    tt = t[t.tool.isin(["c", "rs02"])]
    spread = ((tt.wall_max_s - tt.wall_min_s) / tt.wall_median_s)
    put("EthreeSpreadMax", pct(spread.max(), 1), "E3_table2 (max-min)/median, C psmc and psmc-rs 0.2")
    det = table("E3", "E3_determinism")
    for tool, name in (("rs02", "Rs"), ("c", "C")):
        d = det[det.tool == tool]
        put(f"EthreeDet{name}Identical", str(int(d.byte_identical.sum())), f"E3_determinism {tool}")
        put(f"EthreeDet{name}Total", str(len(d)), f"E3_determinism {tool}")
    env = table("E3", "E3_environment").set_index("key")["value"]
    put("EthreeCpu", str(env["machine.cpu"]).replace(" 64-Core Processor", ""), "E3_environment machine.cpu")
    put("EthreeCores", str(env["machine.cpu_count"]), "E3_environment machine.cpu_count")


# ---------------------------------------------------------------------------
# E4: partly callable bins
# ---------------------------------------------------------------------------


def e4() -> None:
    r = table("E4", "E4_routes")
    g = r.groupby(["mask", "route"]).agg(theta=("theta_rel_complete", "median"),
                                         rmse=("rmse_log10_ne_vs_complete", "median")).reset_index()

    def get(mask, route, col):
        return float(g[(g["mask"] == mask) & (g.route == route)][col].iloc[0])

    for mask, mk in (("dispersed30", "DispThirty"), ("dispersed10", "DispTen"), ("clustered30", "ClusThirty")):
        for route, rk in (("fq2psmcfa", "Fq"), ("mask_auto", "Auto"), ("nomask", "Nomask"), ("multihetsep", "Mhs")):
            put(f"Efour{mk}{rk}Theta", pct(-get(mask, route, "theta"), 1), f"E4_routes {mask} {route} -theta_rel (median)")
            put(f"Efour{mk}{rk}Rmse", fix(get(mask, route, "rmse"), 2), f"E4_routes {mask} {route} rmse vs complete")
    put("EfourReps", str(P.e4.replicates), "config.E4.replicates")
    su = table("E4", "E4_summary")
    su = su[su.route == "mask_auto"].set_index("mask")
    put("EfourDispTenPartly", pct(su.loc["dispersed10", "partly_callable_fraction_mean"]), "E4_summary dispersed10 mask_auto")
    put("EfourDispThirtyPartly", pct(su.loc["dispersed30", "partly_callable_fraction_mean"]), "E4_summary dispersed30 mask_auto")
    import dataclasses
    ms = {m[0]: m for m in (dataclasses.astuple(x) for x in P.e4.masks)}
    put("EfourClusRunKb", fix(ms["clustered30"][3] / 1e3, 0), "config.E4.masks mean uncallable run")
    put("EfourDispRunBp", fix(ms["dispersed30"][3], 0), "config.E4.masks mean uncallable run")
    put("EfourGenomeMb", str(P.e4.genome.n_chromosomes * P.e4.genome.chromosome_length // 10**6), "config.E4.genome")


# ---------------------------------------------------------------------------
# E5: bootstrap calibration
# ---------------------------------------------------------------------------


def e5() -> None:
    c = P.e5
    s = table("E5", "E5_summary")
    s = s[s.window == "informative"].set_index("trunk_mb")
    put("EfiveFits", str(int(s.fits.iloc[0])), "E5_summary fits")
    put("EfiveDatasets", str(int(s.datasets.iloc[0])), "E5_summary datasets")
    put("EfiveReps", str(int(s.replicates_per_dataset.iloc[0])), "E5_summary replicates_per_dataset")
    put("EfiveGenomeMb", str(c.genome.n_chromosomes * c.genome.chromosome_length // 10**6), "config.E5.genome")
    put("EfiveChroms", str(c.genome.n_chromosomes), "config.E5.genome")
    for mb, k in ((50.0, "Fifty"), (5.0, "Five")):
        row = s.loc[mb]
        put(f"Efive{k}Sd", fix(row.sd_ratio_median, 2), "E5_summary sd_ratio_median")
        put(f"Efive{k}SdLo", fix(row.sd_ratio_median_ci_lo, 2), "E5_summary sd_ratio_median_ci_lo")
        put(f"Efive{k}SdHi", fix(row.sd_ratio_median_ci_hi, 2), "E5_summary sd_ratio_median_ci_hi")
        put(f"Efive{k}Cov", pct(row.coverage_mean), "E5_summary coverage_mean")
        put(f"Efive{k}CovLo", pct(row.coverage_mean_ci_lo), "E5_summary coverage_mean_ci_lo")
        put(f"Efive{k}CovHi", pct(row.coverage_mean_ci_hi), "E5_summary coverage_mean_ci_hi")
        put(f"Efive{k}CovProj", pct(row.coverage_projected_mean), "E5_summary coverage_projected_mean")
        put(f"Efive{k}CovMean", pct(row.coverage_of_fit_mean_mean), "E5_summary coverage_of_fit_mean_mean")
        put(f"Efive{k}Trunks", str(int(row.trunks)), "E5_summary trunks")
    put("EfiveBias", fix(s.abs_bias_log10_median.iloc[0], 3), "E5_summary abs_bias_log10_median")
    put("EfiveSdTrue", fix(s.sd_true_log10_median.iloc[0], 3), "E5_summary sd_true_log10_median")


# ---------------------------------------------------------------------------
# E6: formats and decoding
# ---------------------------------------------------------------------------


def e6() -> None:
    f = table("E6", "E6_S2_fits")
    v = f[f.format == "vcf"]
    put("EsixVcfIdentical", "all" if bool(v.parameters_identical.all() and v.loglikes_identical.all()) else "not all",
        "E6_S2_fits vcf identical")
    m = f[f.format == "multihetsep"]
    put("EsixMhsMaxAbs", fix(m.max_abs_log10_ne_vs_psmcfa_per_interval.max(), 3), "E6_S2_fits multihetsep max abs")
    b = table("E6", "E6_S2_bins")
    mb = b[b.format == "multihetsep"]
    put("EsixMhsDiffering", str(int(mb.bins_differing.sum())), "E6_S2_bins multihetsep bins_differing")
    tail = mb[mb.sequence == "all"].bins_missing_at_end
    assert (mb[mb.sequence == "all"].bins_missing_at_end == mb[mb.sequence == "all"].bins_after_last_het).all()
    put("EsixMhsTailMin", thousands(tail.min()), "E6_S2_bins multihetsep bins_missing_at_end (whole genome)")
    put("EsixMhsTailMax", thousands(tail.max()), "E6_S2_bins multihetsep bins_missing_at_end (whole genome)")
    vb = b[b.format == "vcf"]
    assert int(vb.bins_differing.sum()) == 0 and bool(vb.identical.all())
    s = table("E6", "E6_S3_summary").set_index("bins")
    put("EsixDecodeR", fix(s.loc["all", "pearson_log10_mean"], 2), "E6_S3_summary pearson_log10_mean")
    put("EsixDecodeCov", pct(s.loc["all", "coverage95_states"], 1), "E6_S3_summary coverage95_states")
    put("EsixDecodeBins", thousands(s.loc["all", "n_bins"]), "E6_S3_summary n_bins")
    cal = table("E6", "E6_S3_calibration")
    tr, pm = cal[cal.by == "true"], cal[cal.by == "posterior_mean"]
    put("EsixTrueDecileMin", pct(tr.coverage95_states.min(), 1), "E6_S3_calibration by true TMRCA")
    put("EsixTrueDecileMax", pct(tr.coverage95_states.max(), 1), "E6_S3_calibration by true TMRCA")
    put("EsixPostDecileMin", pct(pm.coverage95_states.min(), 1), "E6_S3_calibration by posterior mean")
    put("EsixPostDecileMax", pct(pm.coverage95_states.max(), 1), "E6_S3_calibration by posterior mean")
    put("EsixYoungBias", fix(tr[tr["class"] == 1].mean_log10_error_mean.iloc[0], 2), "E6_S3_calibration true decile 1 bias")
    put("EsixOldBias", fix(-tr[tr["class"] == 10].mean_log10_error_mean.iloc[0], 2), "E6_S3_calibration true decile 10 -bias")
    q = table("E6", "E6_S5_qc")
    sat = q[q.warnings.fillna("").str.contains("BIN_SATURATION")]
    put("EsixSaturated", str(len(sat)), "E6_S5_qc BIN_SATURATION")
    put("EsixSatShareLo", pct(sat.multi_het_share.min()), "E6_S5_qc multi_het_share")
    put("EsixSatShareHi", pct(sat.multi_het_share.max()), "E6_S5_qc multi_het_share")
    put("EsixSatBinLo", str(int(sat.max_bin_size_below_target.min())), "E6_S5_qc max_bin_size_below_target")
    put("EsixSatBinHi", str(int(sat.max_bin_size_below_target.max())), "E6_S5_qc max_bin_size_below_target")
    put("EsixShortSeq", str(int(q.warnings.fillna("").str.contains("SHORT_SEQUENCES").sum())), "E6_S5_qc SHORT_SEQUENCES")


# ---------------------------------------------------------------------------
# E7: 1000 Genomes
# ---------------------------------------------------------------------------


def e7() -> None:
    c = P.e7
    smp = table("E7", "E7_samples")
    put("EsevenGenomes", thousands(len(smp)), "E7_samples rows")
    put("EsevenPops", str(smp.population.nunique()), "E7_samples populations")
    put("EsevenSups", str(smp.superpopulation.nunique()), "E7_samples superpopulations")
    put("EsevenDefault", str(int(smp["default"].sum())), "E7_samples default")
    per = smp.groupby("population").size()
    put("EsevenPerPopMin", str(int(per.min())), "E7_samples per population")
    put("EsevenPerPopMax", str(int(per.max())), "E7_samples per population")
    inp = table("E7", "E7_inputs")
    put("EsevenDepth", fix(inp.raw_depth.median(), 1), "E7_inputs raw_depth median")
    put("EsevenCallableGb", fix(inp.callable_bases.median() / 1e9, 2), "E7_inputs callable_bases median")
    put("EsevenPartly", pct(inp.partly_callable_bin_fraction.median()), "E7_inputs partly_callable_bin_fraction")
    het = inp.groupby("superpopulation").het_per_callable_kb.median()
    put("EsevenHetAfr", fix(het["AFR"], 2), "E7_inputs het_per_callable_kb AFR")
    put("EsevenHetEas", fix(het["EAS"], 2), "E7_inputs het_per_callable_kb EAS")
    rs = table("E7", "E7_route_shift")
    put("EsevenMissingShare", pct(rs.missing_share_of_called_bins.median(), 1), "E7_route_shift missing share")
    put("EsevenThetaShift", pct(rs.theta_rel_vcf_vs_psmcfa.mean(), 1), "E7_route_shift theta_rel mean")
    put("EsevenThetaShiftLo", pct(rs.theta_rel_vcf_vs_psmcfa.min(), 1), "E7_route_shift theta_rel min")
    put("EsevenThetaShiftHi", pct(rs.theta_rel_vcf_vs_psmcfa.max(), 1), "E7_route_shift theta_rel max")
    from experiment.pipeline import metrics, results
    c7 = table("E7", "E7_curves")
    c7 = c7[c7.route.isin(["rs-vcf", "rs-psmcfa"])]

    def curve7(df):
        df = df.sort_values("interval")
        return results.Curve(tuple(df["years"]), tuple(df["ne"]))

    shifts = []
    for _, rr in rs.iterrows():  # at the same times: the routes' theta differ by 8%, and so do their time scales
        g7 = c7[c7.sample_id == rr.sample_id]
        d7 = metrics.log_time_diff(curve7(g7[g7.route == "rs-vcf"]), curve7(g7[g7.route == "rs-psmcfa"]),
                                   (rr.window_lo_years, rr.window_hi_years))
        shifts.append(sum(w * x for w, x in zip(d7.weight, d7.log10_ratio)) / sum(d7.weight))
    put("EsevenNeShift", fix(sum(shifts) / len(shifts), 3), "E7_curves rs-vcf vs rs-psmcfa, mean log10 over the window at the same times")
    rc = table("E7", "E7_rs_vs_c")
    put("EsevenRsCRmseMin", sci(rc.rmse_log10_ne_rs_vs_c_per_interval.min()), "E7_rs_vs_c")
    put("EsevenRsCRmseMax", sci(rc.rmse_log10_ne_rs_vs_c_per_interval.max()), "E7_rs_vs_c")
    put("EsevenRsCMaxAbs", sci(rc.max_abs_log10_ne_rs_vs_c_per_interval.max()), "E7_rs_vs_c")
    g = table("E7", "E7_giab").iloc[0]
    put("EsevenGiabPrecision", fix(g.precision, 4), "E7_giab precision")
    put("EsevenGiabRecall", fix(g.recall, 4), "E7_giab recall")
    put("EsevenGiabTruth", thousands(g.truth_het_snvs), "E7_giab truth_het_snvs")
    put("EsevenGiabMb", thousands(g.eval_bases / 1e6), "E7_giab eval_bases (Mb)")
    tp = table("E7", "E7_throughput").set_index("route")
    put("EsevenRsCoreHours", thousands(tp.loc["rs-psmcfa", "core_hours"]), "E7_throughput rs-psmcfa core_hours")
    put("EsevenRsFitMin", mmss(tp.loc["rs-psmcfa", "wall_median_s"]), "E7_throughput rs-psmcfa wall median (min)")
    put("EsevenCFitMin", mmss(tp.loc["c", "wall_median_s"]), "E7_throughput c wall median (min)")
    c_all = tp.loc["c", "wall_median_s"] * len(smp) / 3600.0
    put("EsevenCCoreHoursAll", thousands(round(c_all, -2)), "E7_throughput c wall median x genomes (estimate)")
    put("EsevenSpeed", fix(tp.loc["c", "wall_median_s"] / tp.loc["rs-psmcfa", "wall_median_s"], 1),
        "E7_throughput c / rs-psmcfa wall median")
    bs = table("E7", "E7_bootstrap")
    put("EsevenBootIdentical", "all" if bool(bs.main_identical_to_fit.all()) else "not all", "E7_bootstrap main_identical_to_fit")
    put("EsevenBootReps", str(int(bs.replicates.iloc[0])), "E7_bootstrap replicates")
    put("EsevenBootThreads", str(int(bs.threads.iloc[0])), "E7_bootstrap threads")
    put("EsevenBootMin", mmss(bs.wall_seconds.median()), "E7_bootstrap wall median (min)")
    sp = table("E7", "E7_split")
    put("EsevenSplitRmse", fix(sp.rmse_log10_ne_split_vs_default_per_interval.median(), 3), "E7_split rmse median")
    put("EsevenSplitN", str(len(sp)), "E7_split rows")
    put("EsevenMu", sci(c.scaling.mu, 3), "config.E7.scaling")
    put("EsevenG", str(int(c.scaling.gen_years)), "config.E7.scaling")
    m = float(rs.missing_share_of_called_bins.median())
    put("EsevenThetaExpected", pct(1.0 / (1.0 - m) - 1.0, 1), "1 / (1 - missing share) - 1")
    put("EsevenRsCParamRelMax", sci(rc[["theta_rel_diff", "rho_rel_diff", "t_max_rel_diff"]].abs().max().max()),
        "E7_rs_vs_c theta, rho, t_max")
    put("EsevenRsCLoglikeFirst", sci(rc.loglike_iteration0_rel_diff.max()), "E7_rs_vs_c loglike_iteration0_rel_diff")
    real = set(table("E2", "E2_rs_vs_c").sample_id) | set(rc.sample_id)
    put("RealGenomes", str(len(real)), "E2_rs_vs_c and E7_rs_vs_c samples (NA12878 in both)")
    put("EsevenGiabCalled", thousands(g.called_het_snvs), "E7_giab called_het_snvs")
    put("EsevenGiabFp", thousands(g.fp), "E7_giab fp")
    put("EsevenGiabFn", thousands(g.fn), "E7_giab fn")
    put("EsevenGiabFnIndel", thousands(g.fn_het_indel), "E7_giab fn_het_indel")
    put("EsevenGiabFnNonpass", thousands(g.fn_nonpass), "E7_giab fn_nonpass")
    put("EsevenRsMem", fix(tp.loc["rs-psmcfa", "peak_rss_median_mb"], 0), "E7_throughput rs-psmcfa peak MB")
    put("EsevenCMemFit", thousands(tp.loc["c", "peak_rss_median_mb"]), "E7_throughput c peak MB")
    put("EsevenVcfFitMin", mmss(tp.loc["rs-vcf", "wall_median_s"]), "E7_throughput rs-vcf wall median (min)")
    put("EsevenBootCoreHours", thousands(tp.loc["boot-rs", "core_hours"]), "E7_throughput boot-rs core_hours")
    put("EsevenCCoreHours", thousands(tp.loc["c", "core_hours"]), "E7_throughput c core_hours")
    put("EsevenBootSamples", str(len(bs)), "E7_bootstrap rows")
    assert bool(bs.main_identical_to_fit.all()) and (bs.c_in_band_fraction == 1).all() and (bs.rs_in_band_fraction == 1).all(), \
        "the text says every bootstrap refitted its main fit exactly and its band held both tools' curves"
    # NA12878's bootstrap ran twice with one seed: E2's on 50 threads, E7's on 32
    a, b = RUNS / "E2" / "bootstrap" / "NA12878" / "trunk500000", RUNS / "E7" / "boot" / "NA12878"
    fa, fb = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
    assert fa == fb, "the two runs wrote different files"
    same = sum((a / n).read_bytes() == (b / n).read_bytes() for n in fa)
    put("EsevenBootFiles", str(len(fa)), "E2/bootstrap and E7/boot NA12878 files")
    put("EsevenBootFilesIdentical", str(same), "E2/bootstrap and E7/boot NA12878 files byte-identical")
    # the fits whose first free lambda stopped at psmc-rs's upper bound (1e4)
    fits = table("E7", "E7_fits").set_index(["sample_id", "route"])
    cur = table("E7", "E7_curves")
    lam = cur["ne"].to_numpy() / fits["n0"].reindex(list(zip(cur.sample_id, cur.route))).to_numpy()
    first = cur[(cur.interval < 4) & (lam >= 1e4 * (1 - 1e-9))]
    put("EsevenBoundDefault", str(first[first.route == "rs-psmcfa"].sample_id.nunique()), "E7_curves rs-psmcfa first window at 1e4")
    assert first[first.route == "rs-psmcfa"].empty, "the text says no fit of the cohort reached the bound"
    put("EsevenSplitBound", str(first[first.route == "rs-split"].sample_id.nunique()), "E7_curves rs-split first window at 1e4")
    put("EsevenSplitRmseMax", fix(sp.rmse_log10_ne_split_vs_default_per_interval.max(), 2), "E7_split rmse max")
    rise = sp[sp.log10_first_vs_next_default > 0.1]
    assert (rise.log10_first_vs_next_split > 0.5 * rise.log10_first_vs_next_default).all(), \
        "the text says splitting the first window left the recent rise in place"
    # the population histories (Fig 7): superpopulation medians of the psmcfa fits
    pb = table("E7", "E7_pop_bands")
    pb = pb[pb.route == "rs-psmcfa"]
    al = pb[pb.group_kind == "all"]
    lo, hi = float(al.window_lo_years.iloc[0]), float(al.window_hi_years.iloc[0])
    wide = pb[pb.group_kind == "superpopulation"].pivot(index="years", columns="group", values="ne_median")
    wide = wide[(wide.index >= lo) & (wide.index <= hi)]
    trough = wide[(wide.index >= 1e4) & (wide.index <= 3e5)]
    non = [s for s in wide.columns if s != "AFR"]
    put("EsevenNonAfrMinNeLo", thousands(trough[non].min().min()), "E7_pop_bands non-AFR superpopulation minima")
    put("EsevenNonAfrMinNeHi", thousands(trough[non].min().max()), "E7_pop_bands non-AFR superpopulation minima")
    put("EsevenNonAfrMinKyLo", fix(trough[non].idxmin().min() / 1e3, 0), "E7_pop_bands non-AFR minima (ky)")
    put("EsevenNonAfrMinKyHi", fix(trough[non].idxmin().max() / 1e3, 0), "E7_pop_bands non-AFR minima (ky)")
    put("EsevenAfrMinNe", thousands(trough["AFR"].min()), "E7_pop_bands AFR minimum")
    put("EsevenAfrMinKy", fix(trough["AFR"].idxmin() / 1e3, 0), "E7_pop_bands AFR minimum (ky)")
    hump = wide[(wide.index >= 1e5) & (wide.index <= 2e6)]
    put("EsevenHumpNeLo", thousands(hump.max().min()), "E7_pop_bands local maxima 100 ky-2 My")
    put("EsevenHumpNeHi", thousands(hump.max().max()), "E7_pop_bands local maxima 100 ky-2 My")
    put("EsevenHumpKyLo", fix(hump.idxmax().min() / 1e3, 0), "E7_pop_bands local maxima (ky)")
    put("EsevenHumpKyHi", fix(hump.idxmax().max() / 1e3, 0), "E7_pop_bands local maxima (ky)")
    spread = (wide.max(axis=1) / wide.min(axis=1)).apply(math.log10)
    tol = math.log10(1.2)
    t_conv = min(t for t in wide.index if t >= 1e5 and (spread[spread.index >= t] <= tol).all())
    put("EsevenConvergeKy", fix(t_conv / 1e3, 0), "E7_pop_bands: superpopulation medians within 20% from here to the window's end")
    put("EsevenWindowHiMy", fix(hi / 1e6, 1), "E7_pop_bands all window_hi_years (My)")
    # the populations within each superpopulation: their spread before and after 150 ky
    cut = 1.5e5
    put("EsevenPopCutKy", fix(cut / 1e3, 0), "split of recent and older times for the populations' spread")
    pop = pb[pb.group_kind == "population"]
    recent, older = [], []
    for _, gp in pop.groupby("superpopulation"):
        w = gp.pivot(index="years", columns="group", values="ne_median")
        w = w[(w.index >= lo) & (w.index <= hi)]
        spread_p = (w.max(axis=1) / w.min(axis=1)).apply(math.log10)
        recent.append(spread_p[spread_p.index < cut].max())
        older.append(spread_p[spread_p.index >= cut].max())
    put("EsevenPopSpreadRecentLo", fix(min(recent), 2), "E7_pop_bands populations of a superpopulation, max log10 spread < 150 ky (lowest)")
    put("EsevenPopSpreadRecentHi", fix(max(recent), 2), "E7_pop_bands populations of a superpopulation, max log10 spread < 150 ky (highest)")
    put("EsevenPopSpreadOldMax", fix(max(older), 2), "E7_pop_bands populations of a superpopulation, max log10 spread >= 150 ky")
    # the data stage (MANIFEST.json)
    import json
    man = json.loads((RUNS / "E7" / "inputs" / "MANIFEST.json").read_text())
    ch = man["chromosomes"]
    assert all(v["records_check"] == "equal" for v in ch.values())
    put("EsevenRecords", thousands(sum(v["records"] for v in ch.values())), "MANIFEST chromosomes records")
    put("EsevenNonpass", thousands(sum(v["nonpass"] for v in ch.values())), "MANIFEST chromosomes nonpass")
    put("EsevenVcfTb", fix(sum(v["read"]["bytes"] for v in ch.values()) / 1e12, 2), "MANIFEST chromosomes read bytes (TB)")
    (pchrom, pil), = man["pilots"].items()
    assert pil["identical"] and all(s["psmcfa_identical"] and not s["qc_fields_differing"] for s in pil["samples"])
    put("EsevenPilotChrom", pchrom.replace("chr", ""), "MANIFEST pilots")
    put("EsevenPilotSamples", str(len(pil["samples"])), "MANIFEST pilots samples")


# ---------------------------------------------------------------------------
# Tables 1, 2 and S1
# ---------------------------------------------------------------------------


def table_s1() -> str:
    t = table("E1", "E1_table1").set_index("model")
    r = table("E1", "E1_rs_vs_c")
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Window (years)} & \multicolumn{2}{c}{RMSE vs truth} & \multicolumn{2}{c}{psmc-rs vs C psmc} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & from & to & C psmc & psmc-rs & RMSE (median) & max $|\Delta|$ \\", r"\midrule"]
    for m in P.e1.models:
        row = t.loc[m]
        rr = r[r.model == m]
        lines.append(
            f"{m} & {thousands(row.window_lo_years)} & {thousands(row.window_hi_years)} & "
            f"{row.rmse_truth_c_mean:.3f} $\\pm$ {row.rmse_truth_c_sd:.3f} & "
            f"{row.rmse_truth_rs_mean:.3f} $\\pm$ {row.rmse_truth_rs_sd:.3f} & "
            f"{sci(rr.rmse_log10_ne_rs_vs_c_per_interval.median())} & {sci(rr.max_abs_log10_ne_rs_vs_c_per_interval.max())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table1() -> str:
    """Agreement between psmc-rs and C psmc on every genome both fitted (E1, E2, E7's default set)."""
    e1 = table("E1", "E1_rs_vs_c")
    reps = table("E1", "E1_replicates").pivot_table(index=["model", "replicate"], columns="tool", values="loglike_common")
    e2 = table("E2", "E2_rs_vs_c")
    f2 = table("E2", "E2_fit_summary")
    e7 = table("E7", "E7_rs_vs_c")
    f7 = table("E7", "E7_fits")
    w7 = f7[f7.route.isin(["c", "rs-psmcfa"])].pivot_table(index="sample_id", columns="route", values="loglike_common")

    def params_rel(df):
        return df[["theta_rel_diff", "rho_rel_diff", "t_max_rel_diff"]].abs().max().max()

    rel2, ll2 = 0.0, 0.0
    for _, g in f2.groupby("sample_id"):
        cc, rs = g[g.tool == "c"].iloc[0], g[g.tool == "rs"].iloc[0]
        rel2 = max(rel2, *(abs(rs[k] - cc[k]) / abs(cc[k]) for k in ("theta", "rho", "t_max")))
        ll2 = max(ll2, abs(rs.loglike_common - cc.loglike_common) / abs(cc.loglike_common))
    rows = [
        (r"Simulated (E1)", len(e1), e1, params_rel(e1),
         ((reps["rs"] - reps["c"]).abs() / reps["c"].abs()).max()),
        (r"Turtles and NA12878 (E2)", len(e2), e2, rel2, ll2),
        (r"1000 Genomes (E7)", len(e7), e7, params_rel(e7),
         ((w7["rs-psmcfa"] - w7["c"]).abs() / w7["c"].abs()).max()),
    ]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& & \multicolumn{2}{c}{$\log_{10}N_\mathrm{e}$ in the window} & \multicolumn{3}{c}{largest relative difference} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
             r"Genomes & $n$ & RMSE & largest $|\Delta|$ & $\theta,\rho,t_{\max}$ & log-likelihood & first E-step \\", r"\midrule"]
    for name, n, d, prel, ll in rows:
        r = d.rmse_log10_ne_rs_vs_c_per_interval
        lines.append(f"{name} & {n} & {sci(r.min())}--{sci(r.max())} & {sci(d.max_abs_log10_ne_rs_vs_c_per_interval.max())} & "
                     f"{sci(prel)} & {sci(ll)} & {sci(d.loglike_iteration0_rel_diff.max())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table2() -> str:
    t = table("E3", "E3_table2")
    sizes = sorted(P.e3.genome_sizes_mb)
    label = {"c": "C psmc", "rs02": "psmc-rs"}
    rows = t[t.tool.isin(label)].sort_values(["tool", "threads"])
    lines = [r"\begin{tabular}{lr" + "r" * len(sizes) + "rr}", r"\toprule",
             r"& & \multicolumn{" + str(len(sizes)) + r"}{c}{Wall time, median (s)} & \multicolumn{2}{c}{" + thousands(max(sizes)) + r" Mb} \\",
             r"\cmidrule(lr){3-" + str(2 + len(sizes)) + r"}\cmidrule(lr){" + str(3 + len(sizes)) + "-" + str(4 + len(sizes)) + "}",
             "Tool & Threads & " + " & ".join(f"{thousands(s)} Mb" for s in sizes) + r" & Speed-up & Memory (MB) \\", r"\midrule"]
    order = [("c", 1)] + [("rs02", th) for th in (1, 2, 4, 8, 16, 30)]
    for tool, th in order:
        sub = rows[(rows.tool == tool) & (rows.threads == th)]
        if sub.empty:
            continue
        cells = []
        for s in sizes:
            x = sub[sub.size_mb == s]
            cells.append(f"{x.wall_median_s.iloc[0]:,.1f}" if not x.empty else "--")
        big = sub[sub.size_mb == max(sizes)]
        sp = big.speedup_vs_c.iloc[0] if not big.empty else float("nan")
        mem = big.peak_rss_median_mb.iloc[0] if not big.empty else float("nan")
        sps = "1" if tool == "c" else (f"{sp:.1f}" if sp < 10 else f"{sp:.0f}")
        lines.append(f"{label[tool]} & {th} & " + " & ".join(cells) + f" & {sps} & {mem:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def crossplatform() -> None:
    """The laptop (arm64) against the production (x86-64): experiment/crossplatform, Supplementary Note 1 section 9."""
    xp = ROOT / "experiment" / "crossplatform"
    rs = pd.read_csv(xp / "rs_laptop_vs_ibex.csv")
    c = pd.read_csv(xp / "c_laptop_vs_ibex.csv")
    sbs = pd.read_csv(xp / "side_by_side_E1.csv")
    gen = pd.read_csv(xp / "inputs_E1_genomes.csv")
    e1 = rs[rs.exp == "E1"]
    put("XpGenomes", str(len(gen)), "crossplatform inputs_E1_genomes rows")
    put("XpGenomesIdentical", str(int((gen.psmcfa_eq_ibex_truth & gen.vcf_gz_eq_ibex_truth).sum())), "crossplatform inputs identical")
    put("XpRsFits", str(len(rs)), "crossplatform rs fits")
    put("XpRsBytesIdentical", str(int(rs.json_bytes_identical.sum())), "crossplatform rs JSON byte-identical")
    put("XpCBytesIdentical", str(int(c.psmc_bytes_identical.sum())), "crossplatform C .psmc byte-identical")
    put("XpRsRmseMax", sci(e1.dlog10ne_rmse_window.max()), "crossplatform rs RMSE window max (E1)")
    put("XpRsRmseMedian", sci(e1.dlog10ne_rmse_window.median()), "crossplatform rs RMSE window median (E1)")
    put("XpRsParamMax", sci(e1.param_max_rel.max()), "crossplatform rs largest relative parameter difference (E1)")
    put("XpCRmseMedian", sci(c.dlog10ne_rmse_window.median()), "crossplatform C RMSE window median")
    put("XpCRmseMax", sci(c.dlog10ne_rmse_window.max()), "crossplatform C RMSE window max")
    r = sbs.ratio_rmse_c_xplat_over_rs_xplat
    assert (r > 1).all(), "the text says psmc-rs agreed more closely across platforms in every replicate"
    put("XpRatioCMedian", fix(r.median(), 0), "side_by_side ratio C cross-build / rs cross-platform, median")
    put("XpRatioCMin", fix(r.min(), 1), "side_by_side ratio C cross-build / rs cross-platform, min")
    q = sbs.ratio_rmse_rsvsc_over_rs_xplat
    put("XpRatioRsCMedian", fix(q.median(), 0), "side_by_side ratio rs-vs-C / rs cross-platform, median")
    put("XpRatioRsCMin", fix(q.min(), 0), "side_by_side ratio rs-vs-C / rs cross-platform, min")
    put("XpIterZero", str(int((rs.loglike_iter0_rel == 0).sum())), "crossplatform rs iteration-0 log-likelihood identical")
    tiers = e1.tier.value_counts()
    for k, name in (("B", "XpTierB"), ("C", "XpTierC"), ("fail", "XpTierFail")):
        put(name, str(int(tiers.get(k, 0))), f"crossplatform rs E1 tier {k}")
    put("XpCTierFail", str(int((c.tier == "fail").sum())), "crossplatform C tier fail")
    na = rs[rs.exp == "E2"].iloc[0]
    put("XpNaRmse", sci(na.dlog10ne_rmse_window), "crossplatform NA12878 RMSE window")
    put("XpNaTier", str(na.tier), "crossplatform NA12878 tier")


def readme() -> None:
    """The diagnostic of the residual difference between the tools (README.md, Testing and C equivalence), which has no table."""
    import re
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


def agreement() -> None:
    """Fig 1i: every interval in its window of every genome both programs fitted, by the figure's own function."""
    from experiment.pipeline.figures import art1_agreement as art1
    from experiment.pipeline.figures.data import curves

    def windows(df, key):
        return {getattr(r, key): (float(r.window_lo_years), float(r.window_hi_years)) for r in df.itertuples()}

    e1c = curves(table("E1", "E1_curves"), ["model", "tool", "replicate"])
    e2c = curves(table("E2", "E2_curves"), ["sample_id", "tool"])
    e7c = curves(table("E7", "E7_curves"), ["sample_id", "route"])
    pairs, genomes = art1.agreement_intervals(e1c, windows(table("E1", "E1_table1"), "model"), e2c,
                                              windows(table("E2", "E2_rs_vs_c"), "sample_id"), e7c,
                                              windows(table("E7", "E7_rs_vs_c"), "sample_id"))
    d = pd.Series([abs(math.log10(rs / c)) for k in pairs for c, rs in pairs[k]])
    assert genomes == len(P.e1.models) * P.e1.replicates + int(MACROS["RealGenomes"][0]), "Fig 1i's genomes"
    put("AgreeIntervals", thousands(len(d)), "Fig 1i: intervals in the window, every genome both tools fitted")
    put("AgreeGenomes", str(genomes), "Fig 1i: genomes (NA12878 once)")
    put("AgreeMedianAbs", sci(d.median()), "Fig 1i: median |log10 Ne difference| per interval")
    put("AgreeMaxAbs", sci(d.max()), "Fig 1i: largest |log10 Ne difference| per interval")
    put("AgreeMaxPct", pct(10 ** d.max() - 1, 2), "Fig 1i: largest relative Ne difference per interval")


def render() -> dict:
    MACROS.clear()
    for f in (e1, e2, e3, e4, e5, e6, e7, agreement, readme, crossplatform):
        f()
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    for name, (value, src) in list(MACROS.items()):
        if value.isdigit() and int(value) < 10:
            put(name + "W", words[int(value)], f"{src} (in words)")
    head = ["% Generated by manuscript/make_numbers.py from experiment/runs/*/tables; do not edit.", ""]
    body = [rf"\newcommand{{\{k}}}{{{v}}}  % {src}" for k, (v, src) in MACROS.items()]
    return {"numbers.tex": "\n".join(head + body) + "\n", "table1.tex": table1(), "table2.tex": table2(),
            "tableS1.tex": table_s1()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the files on disk are not what the tables give")
    a = ap.parse_args()
    files = render()
    stale = [n for n, text in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != text]
    if a.check:
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        print(f"{len(MACROS)} numbers and {len(files) - 1} tables up to date")
        return 0
    for n, text in files.items():
        (OUT / n).write_text(text)
    print(f"wrote {len(MACROS)} numbers and " + ", ".join(n for n in files if n != "numbers.tex"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
