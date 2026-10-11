"""Print the numeric tables for summary.md from summary.json and the CSVs (markdown)."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

X = Path(__file__).resolve().parent.parent
S = json.loads((X / "summary.json").read_text())
MODELS = ("constant", "bottleneck", "expansion", "zigzag")


def rows(name):
    return list(csv.DictReader(open(X / name)))


def f(x, d=2):
    if x is None or x == "":
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    return f"{x:.{d - 1}e}"


def rng(st):
    if st is None:
        return "–"
    return f"{f(st['min'])} / {f(st['median'])} / {f(st['max'])}"


rs = rows("rs_laptop_vs_ibex.csv")
c = rows("c_laptop_vs_ibex.csv")
rsc = rows("rs_vs_c_ibex.csv")
na = next(r for r in rs if r["exp"] == "E2")
rs1 = [r for r in rs if r["exp"] == "E1"]

metrics = [
    ("theta_rel", "θ, relative"),
    ("rho_rel", "ρ, relative"),
    ("t_max_rel", "t_max, relative"),
    ("lambda_max_rel", "largest free λ, relative"),
    ("lambda_median_rel", "median free λ, relative"),
    ("param_max_rel", "largest of θ, ρ, t_max, λ (pre-registered p)"),
    ("dlog10ne_max_all_intervals", "max |Δlog10 Ne|, all 64 intervals (pre-registered d)"),
    ("dlog10ne_max_window", "max |Δlog10 Ne|, intervals in the window"),
    ("dlog10ne_rmse_window", "RMSE Δlog10 Ne over the window (interval_diff)"),
    ("dlog10_t_boundaries_max", "max |Δlog10 t_k| (time boundaries, years)"),
    ("loglike_rel_final", "final log-likelihood, relative (pre-registered)"),
    ("loglike_rel_max_trace", "largest relative LL difference, any iteration"),
    ("loglike_iter0_rel", "iteration-0 log-likelihood, relative"),
]
print("### Agreement (min / median / max over the 40 E1 fits)\n")
print("| quantity | psmc-rs laptop vs Ibex | C psmc laptop vs Ibex | psmc-rs vs C psmc, Ibex | NA12878 psmc-rs laptop vs Ibex |")
print("|---|---|---|---|---|")
for k, label in metrics:
    print(f"| {label} | {rng(S['rs']['E1'].get(k))} | {rng(S['c']['E1'].get(k))} | {rng(S['rs_vs_c_ibex']['E1'].get(k))} | {f(na[k])} |")
print()
print("C psmc's final log-likelihood is that of iteration 24 (C prints none for the final parameters); psmc-rs vs C psmc likewise.\n")

print("### Per model (median over 10 replicates; [max])\n")
print("| model | rs xplat RMSE | C xplat RMSE | rs vs C RMSE (Ibex) | rs xplat max window | C xplat max window | rs vs C max window | rs xplat d (all) | C xplat d (all) | rs vs C d (all) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    a = S["rs"]["E1_per_model"][m]
    b = S["c"]["E1_per_model"][m]
    r = S["rs_vs_c_ibex"]["E1_per_model"][m]

    def mm(st):
        return f"{f(st['median'])} [{f(st['max'])}]"
    print(f"| {m} | {mm(a['dlog10ne_rmse_window'])} | {mm(b['dlog10ne_rmse_window'])} | {mm(r['dlog10ne_rmse_window'])} | "
          f"{mm(a['dlog10ne_max_window'])} | {mm(b['dlog10ne_max_window'])} | {mm(r['dlog10ne_max_window'])} | "
          f"{mm(a['dlog10ne_max_all_intervals'])} | {mm(b['dlog10ne_max_all_intervals'])} | {mm(r['dlog10ne_max_all_intervals'])} |")
print()

print("### Tiers\n")
keys = [("tier", "primary: per-model mean rs-vs-C RMSE (E1_table1, matched intervals); E2: mean of all 40"),
        ("tier_vs_pair_rmse", "same replicate's rs-vs-C RMSE"),
        ("tier_vs_all40_mean", "mean over all 40 E1 replicates"),
        ("tier_vs_min_model_mean", "smallest per-model mean (constant)"),
        ("tier_vs_preB1_ref", "superseded pre-B1 references of the review (sampled curves)"),
        ("tier_window_d_DEVIATION", "DEVIATION: d over the window's intervals only"),
        ("tier_bytes_excl_simd_label_DEVIATION", "DEVIATION: tier A test ignores the JSON's simd label")]
print("| reference for tier C | psmc-rs E1 (40) | psmc-rs NA12878 | C psmc E1 (40) |")
print("|---|---|---|---|")
for k, label in keys:
    def cnt(rr):
        if not rr or k not in rr[0]:
            return "–"
        cc = Counter(r[k] for r in rr)
        return ", ".join(f"{t} {cc[t]}" for t in ("A", "B", "C", "fail") if cc.get(t))
    print(f"| {label} | {cnt(rs1)} | {cnt([na])} | {cnt(c)} |")
print()

print("### What fails which test (counts over the 40 E1 fits)\n")
for name, rr in (("psmc-rs", rs1), ("C psmc", c)):
    p_ok = sum(float(r["param_max_rel"]) <= 1e-6 for r in rr)
    d_ok = sum(float(r["dlog10ne_max_all_intervals"]) <= 1e-6 for r in rr)
    l_ok = sum(float(r["loglike_rel_final"]) <= 1e-8 for r in rr)
    c_ok = sum(float(r["dlog10ne_max_all_intervals"]) <= 1e-3 and 50 * float(r["dlog10ne_max_all_intervals"]) <= float(r["ref_rmse_primary"]) for r in rr)
    print(f"- {name}: p <= 1e-6 in {p_ok}/40, d <= 1e-6 in {d_ok}/40, LL <= 1e-8 in {l_ok}/40; "
          f"tier C test (d <= 1e-3 and 50 d <= reference) passes in {c_ok}/40; "
          f"ratio reference/d: min {f(min(float(r['ratio_ref_over_d']) for r in rr))}, "
          f"median {f(statistics.median(float(r['ratio_ref_over_d']) for r in rr))} (50 needed)")
print(f"- NA12878: p = {f(na['param_max_rel'])} ({na['param_max_rel_name']}), d = {f(na['dlog10ne_max_all_intervals'])}, "
      f"LL = {f(na['loglike_rel_final'])}, reference {f(na['ref_rmse_primary'])}, ratio {f(na['ratio_ref_over_d'])}")
print()
side = rows("side_by_side_E1.csv")
print("### Side by side (per replicate ratios, min / median / max)\n")
for k, label in (("ratio_rmse_rsvsc_over_rs_xplat", "rs-vs-C RMSE / psmc-rs cross-platform RMSE"),
                 ("ratio_rmse_rsvsc_over_c_xplat", "rs-vs-C RMSE / C psmc cross-platform RMSE"),
                 ("ratio_rmse_c_xplat_over_rs_xplat", "C cross-platform RMSE / psmc-rs cross-platform RMSE")):
    v = [float(s[k]) for s in side]
    print(f"- {label}: {f(min(v))} / {f(statistics.median(v))} / {f(max(v))}")
print(f"- counts: {S['side_by_side_counts']}")
