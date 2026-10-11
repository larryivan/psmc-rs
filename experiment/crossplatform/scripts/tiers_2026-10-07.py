"""Cross-platform comparison: laptop (frozen .production) vs Ibex (experiment/runs).

Read-only on the repository. Writes JSON/CSV results next to this script.
"""
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/larryivanhan/Desktop/psmc-rs")
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from experiment.pipeline import config, demography, metrics, results  # noqa: E402

LAP = REPO / "experiment/.production/experiment/runs"
IBX = REPO / "experiment/runs"
OUT = Path(__file__).resolve().parent

MODELS = config.MODEL_ORDER
REPS = range(1, 11)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outputs(rec: Path):
    return json.loads(rec.read_text())["outputs"]


# --------------------------------------------------------------------------
# E1 reference: psmc-rs vs C RMSE (Ibex E1 table 1)
# --------------------------------------------------------------------------
t1 = list(csv.DictReader(open(IBX / "E1/tables/E1_table1.csv")))
pairs = list(csv.DictReader(open(IBX / "E1/tables/E1_rs_vs_c.csv")))
RMSE_MODEL = {r["model"]: float(r["rmse_rs_vs_c_mean"]) for r in t1}
RMSE_ALL = statistics.fmean(float(p["rmse_log10_ne_rs_vs_c"]) for p in pairs)
RMSE_MIN_MODEL = min(RMSE_MODEL.values())
RMSE_PAIR = {(p["model"], int(p["replicate"])): float(p["rmse_log10_ne_rs_vs_c"]) for p in pairs}

report = {"reference": {"rmse_rs_vs_c_per_model_mean": RMSE_MODEL, "rmse_rs_vs_c_mean_all40": RMSE_ALL}}

# --------------------------------------------------------------------------
# 1. simulated files: laptop disk SHA vs Ibex provenance outputs
# --------------------------------------------------------------------------
sim_rows = []


def sim_compare(exp, rel_rec, files):
    lap_rec = LAP / exp / "provenance" / rel_rec
    ibx_rec = IBX / exp / "provenance" / rel_rec
    lo, io = outputs(lap_rec), outputs(ibx_rec)
    for key in files:
        lap_file = LAP / key.replace("<runs>/", "")
        disk = sha(lap_file) if lap_file.exists() else None
        sim_rows.append({
            "exp": exp, "record": rel_rec, "file": key,
            "lap_disk": disk, "lap_prov": lo.get(key), "ibex_prov": io.get(key),
            "identical": disk is not None and disk == io.get(key),
            "lap_disk_matches_lap_prov": disk == lo.get(key),
        })


for m in MODELS:
    for r in REPS:
        base = f"<runs>/E1/sims/{m}/rep{r:02d}/genome"
        sim_compare("E1", f"sim/{m}/rep{r:02d}.json", [base + ".psmcfa", base + ".vcf.gz", base + ".vcf.gz.tbi"])
for r in range(1, 6):
    base = f"<runs>/E4/sims/rep{r:02d}/genome"
    sim_compare("E4", f"sim/rep{r:02d}.json", [base + ".psmcfa", base + ".vcf.gz", base + ".vcf.gz.tbi"])
for r in range(1, 51):
    sim_compare("E5", f"sim/bottleneck/rep{r:02d}.json", [f"<runs>/E5/sims/bottleneck/rep{r:02d}/genome.psmcfa"])

report["sims"] = {
    "n_files": len(sim_rows),
    "n_psmcfa": sum(1 for s in sim_rows if s["file"].endswith(".psmcfa")),
    "n_identical": sum(s["identical"] for s in sim_rows),
    "n_psmcfa_identical": sum(s["identical"] for s in sim_rows if s["file"].endswith(".psmcfa")),
    "lap_integrity_ok": sum(s["lap_disk_matches_lap_prov"] for s in sim_rows),
    "mismatches": [s for s in sim_rows if not s["identical"]],
}
# also the seeds/settings key of the sim records must agree
key_diffs = []
for exp, recs in (("E1", [f"sim/{m}/rep{r:02d}.json" for m in MODELS for r in REPS]),
                  ("E4", [f"sim/rep{r:02d}.json" for r in range(1, 6)]),
                  ("E5", [f"sim/bottleneck/rep{r:02d}.json" for r in range(1, 51)])):
    for rec in recs:
        a = json.loads((LAP / exp / "provenance" / rec).read_text())["key"]
        b = json.loads((IBX / exp / "provenance" / rec).read_text())["key"]
        for k in ("command", "seeds", "settings", "tools"):
            if a.get(k) != b.get(k):
                key_diffs.append((exp, rec, k))
report["sims"]["record_key_diffs"] = key_diffs

# --------------------------------------------------------------------------
# 2. truth JSONs
# --------------------------------------------------------------------------


def flatten(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{pre}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{pre}[{i}]")
    else:
        yield pre, x


truth_rows = []
truth_paths = ([f"E1/truth/{m}/rep{r:02d}.json" for m in MODELS for r in REPS]
               + [f"E4/truth/rep{r:02d}.json" for r in range(1, 6)]
               + [f"E5/truth/bottleneck/rep{r:02d}.json" for r in range(1, 51)])
for rel in truth_paths:
    a_b, b_b = (LAP / rel).read_bytes(), (IBX / rel).read_bytes()
    a, b = json.loads(a_b), json.loads(b_b)
    fa, fb = dict(flatten(a)), dict(flatten(b))
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            if isinstance(va, float) and isinstance(vb, float):
                diffs.append((k, va, vb, abs(va - vb) / abs(vb), abs(va - vb) / np.spacing(abs(vb))))
            else:
                diffs.append((k, va, vb, None, None))
    het_a = [c["het_positions_sha256"] for c in a["chromosomes"]]
    het_b = [c["het_positions_sha256"] for c in b["chromosomes"]]
    truth_rows.append({
        "path": rel, "bytes_identical": a_b == b_b, "n_diff_fields": len(diffs),
        "diffs": diffs, "het_sha_equal": het_a == het_b, "n_chrom": len(het_a),
        "files_equal": a.get("files") == b.get("files"),
    })
non_numeric = [d for t in truth_rows for d in t["diffs"] if d[3] is None]
num = [d for t in truth_rows for d in t["diffs"] if d[3] is not None]
report["truth"] = {
    "n": len(truth_rows),
    "n_bytes_identical": sum(t["bytes_identical"] for t in truth_rows),
    "n_het_sha_equal": sum(t["het_sha_equal"] for t in truth_rows),
    "n_chromosomes_checked": sum(t["n_chrom"] for t in truth_rows),
    "n_files_block_equal": sum(t["files_equal"] for t in truth_rows),
    "non_numeric_diffs": non_numeric,
    "numeric_diff_fields": sorted({d[0] for d in num}),
    "max_rel_numeric_diff": max((d[3] for d in num), default=0.0),
    "max_ulps": max((d[4] for d in num), default=0.0),
    "files_with_diffs": [t["path"] for t in truth_rows if not t["bytes_identical"]],
    "example_diffs": [(t["path"], t["diffs"]) for t in truth_rows if t["diffs"]][:3],
}

# --------------------------------------------------------------------------
# 3. E4 masks
# --------------------------------------------------------------------------
mask_rows = []
for mask in ("nomissing", "dispersed10", "dispersed30", "clustered10", "clustered30"):
    for r in range(1, 6):
        rel = f"mask/{mask}/rep{r:02d}.json"
        lp, ip = LAP / "E4/provenance" / rel, IBX / "E4/provenance" / rel
        if not lp.exists() or not ip.exists():
            continue
        lo, io = outputs(lp), outputs(ip)
        for key, ih in io.items():
            f = LAP / key.replace("<runs>/", "")
            disk = sha(f) if f.exists() else None
            row = {"mask": mask, "rep": r, "file": key, "identical": disk == ih, "lap_integrity": disk == lo.get(key)}
            if key.endswith(".json"):  # also present on Ibex disk
                row["ibex_disk_matches_prov"] = sha(IBX / key.replace("<runs>/", "")) == ih
            mask_rows.append(row)
report["masks"] = {
    "n_jobs": len({(m["mask"], m["rep"]) for m in mask_rows}),
    "n_files": len(mask_rows),
    "n_identical": sum(m["identical"] for m in mask_rows),
    "lap_integrity_ok": sum(m["lap_integrity"] for m in mask_rows),
    "mismatches": [m for m in mask_rows if not m["identical"]],
}

# --------------------------------------------------------------------------
# Fit comparison helpers
# --------------------------------------------------------------------------


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


def curve_maxdiff(ca, cb, window=None):
    """max |log10 Ne_a(t) - log10 Ne_b(t)| over all t (union of breakpoints), or over a grid."""
    if window is not None:
        return float(np.max(np.abs(np.log10(ca.at(window)) - np.log10(cb.at(window)))))
    pts = np.unique(np.concatenate([np.asarray(ca.years), np.asarray(cb.years)]))
    return float(np.max(np.abs(np.log10(ca.at(pts)) - np.log10(cb.at(pts)))))


def interval_maxdiff(fa, fb, lo=0):
    """max over intervals k >= lo of |log10(lambda_k N0)| differences (N0 proportional to theta): time-aligned by interval."""
    a = np.log10(np.asarray(fa.lambdas) * fa.theta)
    b = np.log10(np.asarray(fb.lambdas) * fb.theta)
    return float(np.max(np.abs(a - b)[lo:]))


def boundary_maxdiff(fa, fb):
    a = np.log10(np.asarray(fa.boundaries()[1:]) * fa.theta)
    b = np.log10(np.asarray(fb.boundaries()[1:]) * fb.theta)
    return float(np.max(np.abs(a - b)))


def param_maxrel(fa, fb):
    pa = [fa.theta, fa.rho, fa.t_max, *fa.free_lambdas]
    pb = [fb.theta, fb.rho, fb.t_max, *fb.free_lambdas]
    assert len(pa) == len(pb)
    names = ["theta", "rho", "t_max"] + [f"lam{i}" for i in range(len(fa.free_lambdas))]
    rels = [rel(x, y) for x, y in zip(pa, pb)]
    i = int(np.argmax(rels))
    return rels[i], names[i]


def tier(bytes_eq, p_rel, d_ne, ll_rel, rmse_ref):
    if bytes_eq:
        return "A"
    if p_rel <= 1e-6 and d_ne <= 1e-6 and ll_rel <= 1e-8:
        return "B"
    if d_ne <= 1e-3 and d_ne * 50 <= rmse_ref:
        return "C"
    return "fail"


# --------------------------------------------------------------------------
# 4. C psmc fits (E1: 40, E2: 4)
# --------------------------------------------------------------------------
cfits = []
sc = config.SIMULATION_SCALING
grid = config.TimeGrid()
for m in MODELS:
    eps = demography.epochs(config.MODELS[m]) if isinstance(config.MODELS, dict) else None
    window = metrics.model_grid(eps, grid, sc.gen_years) if eps is not None else None
    for r in REPS:
        relp = f"E1/fits/{m}/rep{r:02d}.c.psmc"
        cfits.append(("E1", m, r, relp, sc, window, RMSE_MODEL[m], RMSE_PAIR[(m, r)]))
real = {g.sample_id: g for g in config.REAL_GENOMES}
for sid in ("HLemySub1", "HLhydTec1", "HLpelCas1", "Papuan"):
    cfits.append(("E2", sid, 0, f"E2/fits/{sid}.c.psmc", real[sid].scaling, None, RMSE_ALL, None))

c_rows = []
for exp, m, r, relp, scal, window, rmse_model, rmse_pair in cfits:
    a_p, b_p = LAP / relp, IBX / relp
    a, b = results.parse_psmc(a_p), results.parse_psmc(b_p)
    ra, rb = a.complete_rounds(), b.complete_rounds()
    assert [x.round for x in ra] == [x.round for x in rb]
    assert a.header == b.header, (a.header, b.header)
    fa, fb = ra[-1].fit(), rb[-1].fit()
    p_rel, p_name = param_maxrel(fa, fb)
    ca, cb = fa.curve(scal.mu, scal.gen_years, 100), fb.curve(scal.mu, scal.gen_years, 100)
    d_jump = curve_maxdiff(ca, cb)
    d_all = interval_maxdiff(fa, fb)
    d_excl0 = interval_maxdiff(fa, fb, 1)
    d_bound = boundary_maxdiff(fa, fb)
    d_win = curve_maxdiff(ca, cb, window) if window is not None else None
    rmse_win = metrics.rmse_between(ca, cb, window) if window is not None else None
    # every round
    p_rel_rounds = max(param_maxrel(x.fit(), y.fit())[0] for x, y in zip(ra, rb) if x.pa is not None)
    d_rounds = max(interval_maxdiff(x.fit(), y.fit()) for x, y in zip(ra, rb) if x.pa is not None)
    lka, lkb = a.loglikes(), b.loglikes()
    ll_rels = [rel(lka[i], lkb[i]) for i in lkb if i >= 0 and lkb[i] != 0]
    ll_final = rel(lka[max(lkb)], lkb[max(lkb)])
    # first differing line
    la, lb = a_p.read_text().splitlines(), b_p.read_text().splitlines()
    first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
    first_info = None
    if first is not None:
        rd = max((int(l.split("\t")[1]) for l in la[:first + 1] if l.startswith("RD\t")), default=None)
        first_info = {"line": first + 1, "round": rd, "lap": la[first][:80], "ibex": lb[first][:80]}
    bytes_eq = a_p.read_bytes() == b_p.read_bytes()
    ref = rmse_model
    c_rows.append({
        "exp": exp, "model": m, "rep": r, "bytes_identical": bytes_eq,
        "param_max_rel": p_rel, "param_max_rel_name": p_name, "param_max_rel_all_rounds": p_rel_rounds,
        "max_abs_dlog10ne_all_t": d_all, "max_abs_dlog10ne_excl_lam0": d_excl0, "max_abs_dlog10_t_boundaries": d_bound,
        "sup_steps_union_breakpoints": d_jump, "max_abs_dlog10ne_window": d_win, "rmse_window_c_vs_c": rmse_win, "max_abs_dlog10ne_all_rounds": d_rounds,
        "loglike_rel_final": ll_final, "loglike_rel_max_rounds": max(ll_rels),
        "rmse_ref_model": ref, "rmse_ref_pair": rmse_pair,
        "ratio_ref_over_d": ref / d_all if d_all > 0 else math.inf,
        "tier": tier(bytes_eq, p_rel, d_all, ll_final, ref),
        "tier_vs_pair_rmse": tier(bytes_eq, p_rel, d_all, ll_final, rmse_pair) if rmse_pair else None,
        "tier_vs_all40": tier(bytes_eq, p_rel, d_all, ll_final, RMSE_ALL),
        "first_diff": first_info,
    })


def summarize(rows, key="tier"):
    from collections import Counter
    return dict(Counter(r[key] for r in rows))


report["c_fits"] = {
    "n": len(c_rows),
    "tiers": summarize(c_rows),
    "tiers_vs_pair_rmse": summarize([r for r in c_rows if r["tier_vs_pair_rmse"]], "tier_vs_pair_rmse"),
    "tiers_vs_all40": summarize(c_rows, "tier_vs_all40"),
    "max_param_rel": max(r["param_max_rel"] for r in c_rows),
    "max_param_rel_all_rounds": max(r["param_max_rel_all_rounds"] for r in c_rows),
    "max_dlog10ne": max(r["max_abs_dlog10ne_all_t"] for r in c_rows),
    "max_dlog10ne_excl_lam0": max(r["max_abs_dlog10ne_excl_lam0"] for r in c_rows),
    "median_dlog10ne": float(np.median([r["max_abs_dlog10ne_all_t"] for r in c_rows])),
    "max_dlog10_t_boundaries": max(r["max_abs_dlog10_t_boundaries"] for r in c_rows),
    "rmse_window_c_vs_c_mean_E1": statistics.fmean(r["rmse_window_c_vs_c"] for r in c_rows if r["rmse_window_c_vs_c"] is not None),
    "n_E1_c_vs_c_rmse_gt_rs_vs_c_rmse": sum(1 for r in c_rows if r["rmse_window_c_vs_c"] is not None and r["rmse_window_c_vs_c"] > r["rmse_ref_pair"]),
    "tier_counts_if_d_excludes_lam0": None,
    "max_dlog10ne_window": max(r["max_abs_dlog10ne_window"] for r in c_rows if r["max_abs_dlog10ne_window"] is not None),
    "max_dlog10ne_all_rounds": max(r["max_abs_dlog10ne_all_rounds"] for r in c_rows),
    "max_loglike_rel_final": max(r["loglike_rel_final"] for r in c_rows),
    "max_loglike_rel_rounds": max(r["loglike_rel_max_rounds"] for r in c_rows),
    "min_ratio_ref_over_d": min(r["ratio_ref_over_d"] for r in c_rows),
    "first_diff_rounds": sorted({r["first_diff"]["round"] for r in c_rows if r["first_diff"]}),
    "per_model": {},
}
for m in list(MODELS) + ["E2"]:
    rows = [r for r in c_rows if (r["model"] == m if m != "E2" else r["exp"] == "E2")]
    report["c_fits"]["per_model"][m] = {
        "n": len(rows), "tiers": summarize(rows),
        "max_param_rel": max(r["param_max_rel"] for r in rows),
        "max_dlog10ne": max(r["max_abs_dlog10ne_all_t"] for r in rows),
        "max_dlog10ne_window": max((r["max_abs_dlog10ne_window"] or 0) for r in rows),
        "rmse_c_vs_c_mean": statistics.fmean(r["rmse_window_c_vs_c"] for r in rows) if rows[0]["rmse_window_c_vs_c"] is not None else None,
        "rmse_rs_vs_c_mean": statistics.fmean(r["rmse_ref_pair"] for r in rows) if rows[0]["rmse_ref_pair"] is not None else None,
        "max_loglike_rel": max(r["loglike_rel_final"] for r in rows),
        "min_ratio": min(r["ratio_ref_over_d"] for r in rows),
    }

# --------------------------------------------------------------------------
# 5. E2 HLpelCas1 trunk500000 bootstrap
# --------------------------------------------------------------------------
B = "E2/bootstrap/HLpelCas1/trunk500000"
scal = real["HLpelCas1"].scaling
boot_rows = []


def rs_fit(p):
    return results.fit_from_rs(json.loads(p.read_text()))


ma, mb = json.loads((LAP / B / "main.json").read_text()), json.loads((IBX / B / "main.json").read_text())
fa, fb = results.fit_from_rs(ma), results.fit_from_rs(mb)
p_rel, p_name = param_maxrel(fa, fb)
d = interval_maxdiff(fa, fb)
d_b = boundary_maxdiff(fa, fb)
lla, llb = ma["loglike"], mb["loglike"]
ll = rel(ma["final_loglike"], mb["final_loglike"])
ll_trace = max(rel(x, y) for x, y in zip(lla, llb))
ll_first_diff = next((i for i, (x, y) in enumerate(zip(lla, llb)) if x != y), None)
boot_rows.append({"name": "main", "bytes_identical": (LAP / B / "main.json").read_bytes() == (IBX / B / "main.json").read_bytes(),
                  "param_max_rel": p_rel, "param_name": p_name, "dlog10ne": d, "dlog10_t": d_b, "ll_rel": ll, "ll_rel_trace": ll_trace,
                  "ll_first_diff_iter": ll_first_diff, "n_iter_equal": len(lla) == len(llb),
                  "tier": tier(False, p_rel, d, ll, RMSE_ALL), "tier_strict": tier(False, p_rel, d, ll, RMSE_MIN_MODEL)})
for i in range(1, 101):
    ja, jb = LAP / B / f"replicate_{i:03d}.json", IBX / B / f"replicate_{i:03d}.json"
    pa_, pb_ = LAP / B / f"replicate_{i:03d}.psmc", IBX / B / f"replicate_{i:03d}.psmc"
    fa, fb = rs_fit(ja), rs_fit(jb)
    p_rel, p_name = param_maxrel(fa, fb)
    d = interval_maxdiff(fa, fb)
    xa, xb = results.parse_psmc(pa_), results.parse_psmc(pb_)
    assert xa.header == xb.header, (i, xa.header, xb.header)
    lka, lkb = xa.loglikes(), xb.loglikes()
    k = max(lkb)
    ll = rel(lka[k], lkb[k])
    lk0_equal = [l for l in pa_.read_text().splitlines() if l.startswith("LK")][1] == [l for l in pb_.read_text().splitlines() if l.startswith("LK")][1]
    boot_rows.append({"name": f"rep{i:03d}", "bytes_identical": ja.read_bytes() == jb.read_bytes(),
                      "psmc_bytes_identical": pa_.read_bytes() == pb_.read_bytes(),
                      "param_max_rel": p_rel, "param_name": p_name, "dlog10ne": d, "ll_rel": ll,
                      "same_input": xa.header == xb.header, "lk_iter0_text_equal": lk0_equal,
                      "tier": tier(ja.read_bytes() == jb.read_bytes(), p_rel, d, ll, RMSE_ALL),
                      "tier_strict": tier(ja.read_bytes() == jb.read_bytes(), p_rel, d, ll, RMSE_MIN_MODEL)})
reps = [r for r in boot_rows if r["name"] != "main"]
# summary quantiles
sa = json.loads((LAP / B / "summary.json").read_text())
sb = json.loads((IBX / B / "summary.json").read_text())
qdiff = {}
for key in ("ne_main", "ne_q025", "ne_q500", "ne_q975"):
    qdiff[key] = max(abs(math.log10(x[key]) - math.log10(y[key])) for x, y in zip(sa["points"], sb["points"]))
xdiff = max(rel(x["x_years"], y["x_years"]) for x, y in zip(sa["points"], sb["points"]))
report["bootstrap"] = {
    "n": len(boot_rows),
    "tiers": summarize(boot_rows),
    "tiers_strict_vs_min_model_rmse": summarize(boot_rows, "tier_strict"),
    "main": boot_rows[0],
    "replicates_max_param_rel": max(r["param_max_rel"] for r in reps),
    "replicates_median_param_rel": float(np.median([r["param_max_rel"] for r in reps])),
    "replicates_max_dlog10ne": max(r["dlog10ne"] for r in reps),
    "replicates_median_dlog10ne": float(np.median([r["dlog10ne"] for r in reps])),
    "replicates_max_ll_rel": max(r["ll_rel"] for r in reps),
    "replicates_same_input_headers": sum(r["same_input"] for r in reps),
    "replicates_bytes_identical": sum(r["bytes_identical"] for r in reps),
    "replicates_lk_iter0_text_equal": sum(r["lk_iter0_text_equal"] for r in reps),
    "worst_replicates": sorted(reps, key=lambda r: -r["dlog10ne"])[:3],
    "param_name_counts": dict(__import__("collections").Counter(r["param_name"] for r in reps)),
    "summary_seed_equal": sa["seed"] == sb["seed"],
    "summary_n_reps": (sa["n_replicates"], sb["n_replicates"]),
    "summary_max_abs_dlog10": qdiff,
    "summary_x_years_max_rel": xdiff,
}

# --------------------------------------------------------------------------
# extra: laptop partial HLemySub1 bootstrap (job did not complete on the laptop)
# --------------------------------------------------------------------------
B2 = "E2/bootstrap/HLemySub1/trunk500000"
extra = []
for ja in sorted((LAP / B2).glob("replicate_*.json")):
    jb = IBX / B2 / ja.name
    if not jb.exists():
        continue
    fa, fb = rs_fit(ja), rs_fit(jb)
    p_rel, _ = param_maxrel(fa, fb)
    sc2 = real["HLemySub1"].scaling
    d = interval_maxdiff(fa, fb)
    extra.append((ja.name, p_rel, d))
report["extra_HLemySub1_partial"] = {
    "n": len(extra), "max_param_rel": max(e[1] for e in extra) if extra else None,
    "max_dlog10ne": max(e[2] for e in extra) if extra else None,
}

(OUT / "xplatform.json").write_text(json.dumps(report, indent=1, default=str))
with open(OUT / "xplatform_cfits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in c_rows[0] if k != "first_diff"] + ["first_diff_round"])
    w.writeheader()
    for r in c_rows:
        rr = {k: v for k, v in r.items() if k != "first_diff"}
        rr["first_diff_round"] = r["first_diff"]["round"] if r["first_diff"] else None
        w.writerow(rr)
with open(OUT / "xplatform_bootstrap.csv", "w", newline="") as f:
    keys = sorted({k for r in boot_rows for k in r})
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(boot_rows)
print(json.dumps({k: v for k, v in report.items()}, indent=1, default=str)[:20000])
