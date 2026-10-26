"""summary.json from e1_variants.csv: each variant over E1's 40 genomes and on bottleneck replicate 1, next to
the difference between two builds of C psmc (the laptop's and Ibex's, experiment/crossplatform/side_by_side_E1.csv,
on the same windows with the same metric). manuscript/make_numbers.py reads summary.json.

Usage: python3 -I -B scripts/summary.py
"""
import csv
import json
import statistics
from pathlib import Path

C = Path(__file__).resolve().parent.parent
REPO = C.parents[1]
VARIANTS = ["default", "lastbin", "hj", "both", "prec"]
N_ITER = 25


def num(x):
    try:
        return float(x)
    except ValueError:
        return x


def spread(xs):
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs)}


rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(open(C / "e1_variants.csv"))]
key = lambda r: (r["model"], int(r["replicate"]))  # noqa: E731
by = {v: {key(r): r for r in rows if r["variant"] == v} for v in VARIANTS}
side = {(r["model"], int(r["replicate"])): r for r in csv.DictReader(open(REPO / "experiment/crossplatform/side_by_side_E1.csv"))}
assert set(side) == set(by["default"]), "the same 40 genomes"
cx = {k: float(r["rmse_window_c_laptop_vs_ibex"]) for k, r in side.items()}
for k, r in by["default"].items():  # the side-by-side table's psmc-rs against C psmc is E1's
    assert abs(r["rmse_log10_ne_rs_vs_c_per_interval"] / float(side[k]["rmse_window_rs_vs_c_ibex"]) - 1) < 1e-9, k
RMSE = "rmse_log10_ne_rs_vs_c_per_interval"
out = {"genomes": len(cx), "c_builds_rmse": spread(list(cx.values())), "variants": {}}
for v in VARIANTS:
    rs = by[v]
    out["variants"][v] = {
        "rmse_vs_c": spread([r[RMSE] for r in rs.values()]),
        "max_abs_vs_c_window": max(r["max_abs_log10_ne_rs_vs_c_per_interval"] for r in rs.values()),
        "params_rel_vs_c_max": max(max(abs(r["theta_rel_diff"]), abs(r["rho_rel_diff"]), abs(r["t_max_rel_diff"]))
                                   for r in rs.values()),
        "loglike_max_rel_vs_c": spread([r["loglike_max_rel_diff"] for r in rs.values()]),
        "rmse_vs_default": spread([r["rmse_log10_ne_vs_default_per_interval"] for r in rs.values()]),
        "ratio_rmse_to_default": spread([r[RMSE] / by["default"][k][RMSE] for k, r in rs.items()]),
        "closer_to_c_than_default": sum(r[RMSE] < by["default"][k][RMSE] for k, r in rs.items()),
        "at_or_below_c_builds": sum(r[RMSE] <= cx[k] for k, r in rs.items()),
        "ratio_to_c_builds": spread([r[RMSE] / cx[k] for k, r in rs.items()]),
        "evaluations_per_m_step": {"min": min(r["it_min"] for r in rs.values()), "max": max(r["it_max"] for r in rs.values()),
                                   "mean": sum(r["it_total"] for r in rs.values()) / (N_ITER * len(rs))},
    }
dflt = by["default"].values()
out["c_evaluations_per_m_step"] = {"min": min(r["c_it_min"] for r in dflt), "max": max(r["c_it_max"] for r in dflt),
                                   "mean": sum(r["c_it_total"] for r in dflt) / (N_ITER * len(dflt))}
out["lastbin_not_closer"] = [
    {"model": k[0], "replicate": k[1], "default": by["default"][k][RMSE], "lastbin": r[RMSE], "c_builds": cx[k]}
    for k, r in by["lastbin"].items() if not r[RMSE] < by["default"][k][RMSE]]
rep01 = ("bottleneck", 1)
out["bottleneck_rep01"] = {"c_builds_rmse": cx[rep01], **{v: {k: by[v][rep01][k] for k in (
    RMSE, "max_abs_log10_ne_rs_vs_c_per_interval", "max_abs_log10_ne_rs_vs_c_any_interval", "theta_rel_diff",
    "rho_rel_diff", "t_max_rel_diff", "loglike_max_rel_diff", "loglike_common_diff", "rmse_log10_ne_vs_default_per_interval",
    "max_abs_log10_ne_vs_default_per_interval", "it_min", "it_max", "lam0", "c_lam0")} for v in VARIANTS}}
(C / "summary.json").write_text(json.dumps(out, indent=1) + "\n")

f = lambda x: f"{x:.2e}"  # noqa: E731
print(f"E1, {out['genomes']} genomes: RMSE of log10 Ne against C psmc's fit (C psmc's two builds: median "
      f"{f(out['c_builds_rmse']['median'])})")
print(f"{'variant':8s} {'min':>9s} {'median':>9s} {'max':>9s} {'vs default':>10s} {'closer':>6s} {'<= C builds':>11s}")
for v in VARIANTS:
    o = out["variants"][v]
    print(f"{v:8s} {f(o['rmse_vs_c']['min']):>9s} {f(o['rmse_vs_c']['median']):>9s} {f(o['rmse_vs_c']['max']):>9s} "
          f"{f(o['rmse_vs_default']['median']):>10s} {o['closer_to_c_than_default']:6d} {o['at_or_below_c_builds']:11d}")
print(f"bottleneck rep01 (C psmc's two builds {f(cx[rep01])}): " +
      ", ".join(f"{v} {f(out['bottleneck_rep01'][v][RMSE])}" for v in VARIANTS[:4]) +
      f"; prec vs default {f(out['bottleneck_rep01']['prec']['rmse_log10_ne_vs_default_per_interval'])}")
