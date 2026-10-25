"""Steps 4-5: laptop vs Ibex, psmc-rs and C psmc, against the pre-registered tiers.

Comparisons (a = laptop, b = Ibex, the reference):
  rs: the laptop psmc-rs fits (runs/) vs the committed Ibex fits of the production (experiment/runs),
      40 E1 replicates and E2's NA12878 whole-genome fit, both from the same sources;
  c:  the laptop C psmc fits of the frozen .production run (copied in laptop_c/) vs the committed
      Ibex C fits, 40 E1 replicates;
  and, side by side, the committed Ibex psmc-rs vs C psmc agreement (E1_rs_vs_c.csv).

Pre-registered tiers (review/xplatform.py, unchanged):
  A  output bytes identical
  B  max relative parameter difference <= 1e-6 and max |dlog10 Ne| over all intervals <= 1e-6
     and relative final log-likelihood difference <= 1e-8
  C  max |dlog10 Ne| over all intervals <= 1e-3 and 50 * that <= the reference psmc-rs-vs-C RMSE
  fail otherwise
Primary reference for C (B1/B2 of the review): the per-model mean psmc-rs-vs-C RMSE of the committed
E1_table1.csv (rmse_rs_vs_c_per_interval_mean, matched intervals), for E2 the mean over all 40 E1 pairs.

Run with: experiment/.venv/bin/python -I -B compare.py
"""
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from experiment.pipeline import metrics, results  # noqa: E402

X = Path(__file__).resolve().parent.parent
LAP = REPO / "experiment/.production/experiment/runs"
IBX = REPO / "experiment/runs"
MODELS = ("constant", "bottleneck", "expansion", "zigzag")
REPS = range(1, 11)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a - b)


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------
t1 = {r["model"]: r for r in csv.DictReader(open(IBX / "E1/tables/E1_table1.csv"))}
pairs = {(r["model"], int(r["replicate"])): r for r in csv.DictReader(open(IBX / "E1/tables/E1_rs_vs_c.csv"))}
e2pairs = {r["sample_id"]: r for r in csv.DictReader(open(IBX / "E2/tables/E2_rs_vs_c.csv"))}
REF_MODEL = {m: float(t1[m]["rmse_rs_vs_c_per_interval_mean"]) for m in MODELS}
REF_ALL40 = statistics.fmean(float(p["rmse_log10_ne_rs_vs_c_per_interval"]) for p in pairs.values())
REF_MIN = min(REF_MODEL.values())
REF_PAIR = {k: float(p["rmse_log10_ne_rs_vs_c_per_interval"]) for k, p in pairs.items()}
REF_PAIR[("NA12878", 0)] = float(e2pairs["NA12878"]["rmse_log10_ne_rs_vs_c_per_interval"])
# the reference the review's run used, before the B1 fix (sampled step curves); superseded, for sensitivity only
REF_PRE_B1_MODEL = {"constant": 1.0471840522603617e-05, "bottleneck": 7.036926314623781e-05,
                    "expansion": 0.0006401140897108174, "zigzag": 0.0014215631947421246}
REF_PRE_B1_ALL40 = 0.0005356295970304458
WINDOW = {m: (float(pairs[(m, 1)]["window_lo_years"]), float(pairs[(m, 1)]["window_hi_years"])) for m in MODELS}
WINDOW["NA12878"] = (float(e2pairs["NA12878"]["window_lo_years"]), float(e2pairs["NA12878"]["window_hi_years"]))
for m in MODELS:
    assert all((float(pairs[(m, r)]["window_lo_years"]), float(pairs[(m, r)]["window_hi_years"])) == WINDOW[m] for r in REPS)


def tier(bytes_eq, p_rel, d_ne, ll_rel, rmse_ref):
    """The pre-registered tiers, verbatim from review/xplatform.py."""
    if bytes_eq:
        return "A"
    if p_rel <= 1e-6 and d_ne <= 1e-6 and ll_rel <= 1e-8:
        return "B"
    if d_ne <= 1e-3 and d_ne * 50 <= rmse_ref:
        return "C"
    return "fail"


# --------------------------------------------------------------------------
# helpers (the pre-registered ones from review/xplatform.py, and the pipeline's interval_diff)
# --------------------------------------------------------------------------


def interval_maxdiff(fa, fb, lo=0):
    """max over intervals k >= lo of |log10(lambda_k theta)| differences (N0 proportional to theta)."""
    a = np.log10(np.asarray(fa.lambdas) * fa.theta)
    b = np.log10(np.asarray(fb.lambdas) * fb.theta)
    return float(np.max(np.abs(a - b)[lo:]))


def boundary_maxdiff(fa, fb):
    a = np.log10(np.asarray(fa.boundaries()[1:]) * fa.theta)
    b = np.log10(np.asarray(fb.boundaries()[1:]) * fb.theta)
    return float(np.max(np.abs(a - b)))


def param_rels(fa, fb):
    out = {"theta": rel(fa.theta, fb.theta), "rho": rel(fa.rho, fb.rho), "t_max": rel(fa.t_max, fb.t_max)}
    lam = [rel(x, y) for x, y in zip(fa.free_lambdas, fb.free_lambdas)]
    assert len(fa.free_lambdas) == len(fb.free_lambdas)
    k = int(np.argmax(lam))
    out["lambda_max"] = lam[k]
    out["lambda_argmax"] = k
    out["lambda_median"] = float(np.median(lam))
    allp = {"theta": out["theta"], "rho": out["rho"], "t_max": out["t_max"], **{f"lam{i}": v for i, v in enumerate(lam)}}
    name = max(allp, key=allp.get)
    out["param_max"] = allp[name]
    out["param_max_name"] = name
    return out


def first_diff(pa: Path, pb: Path):
    """The first differing line of two .psmc files: its number, tag and EM iteration.

    The round is the RD block the line belongs to; an IT line (the M-step's
    evaluation count) is printed just before the RD block of the iteration
    whose M-step it counts, so it is attributed to that next block.
    """
    la, lb = pa.read_text().splitlines(), pb.read_text().splitlines()
    i = next((k for k, (x, y) in enumerate(zip(la, lb)) if x != y), None)
    if i is None:
        return None if len(la) == len(lb) else {"line": min(len(la), len(lb)) + 1, "round": None, "tag": "length"}
    tag = la[i].split("\t")[0]
    if tag == "IT":
        rd = next((int(l.split("\t")[1]) for l in la[i:] if l.startswith("RD\t")), None)
    else:
        rd = max((int(l.split("\t")[1]) for l in la[: i + 1] if l.startswith("RD\t")), default=None)
    return {"line": i + 1, "round": rd, "tag": tag, "a": la[i][:60], "b": lb[i][:60]}


def it_counts(p: Path):
    return [int(l.split("\t")[1]) for l in p.read_text().splitlines() if l.startswith("IT\t")]


def interval_lambda(fit, k):
    """The free lambda (its index) that interval k belongs to."""
    widths = [b for a, b in results.pattern_groups(fit.pattern) for _ in range(a)] if fit.pattern else [1] * (fit.n + 1)
    return int(np.searchsorted(np.cumsum(widths), k, side="right"))


def curve_metrics(fa, fb, scal, window):
    mu, g, bin_size = scal
    ca, cb = fa.curve(mu, g, bin_size), fb.curve(mu, g, bin_size)
    d = metrics.interval_diff(ca, cb, window)
    diffs = np.abs(np.log10(np.asarray(fa.lambdas) * fa.theta) - np.log10(np.asarray(fb.lambdas) * fb.theta))
    k = int(np.argmax(diffs))
    return {
        "dlog10ne_max_all_intervals": interval_maxdiff(fa, fb),  # the pre-registered d
        "dlog10ne_argmax_interval": k,
        "dlog10ne_argmax_lambda": interval_lambda(fb, k),
        "dlog10ne_argmax_in_window": k in d.interval,
        "dlog10ne_max_any_interval_pipeline": d.max_abs_any,
        "dlog10ne_max_window": d.max_abs,
        "dlog10ne_rmse_window": d.rmse,
        "intervals_in_window": len(d.interval),
        "dlog10_t_boundaries_max": boundary_maxdiff(fa, fb),
    }


def psmc_rounds_param_max(pa: results.PsmcFile, pb: results.PsmcFile):
    ra, rb = pa.complete_rounds(), pb.complete_rounds()
    assert [x.round for x in ra] == [x.round for x in rb]
    return max(param_rels(x.fit(), y.fit())["param_max"] for x, y in zip(ra, rb) if x.pa is not None)


def tiers_for(row, bytes_eq, key_model, pair_key):
    p, d, ll = row["param_max_rel"], row["dlog10ne_max_all_intervals"], row["loglike_rel_final"]
    if key_model in REF_MODEL:
        ref, pre = REF_MODEL[key_model], REF_PRE_B1_MODEL[key_model]
    else:
        ref, pre = REF_ALL40, REF_PRE_B1_ALL40
    row["ref_rmse_primary"] = ref
    row["tier"] = tier(bytes_eq, p, d, ll, ref)
    row["tier_vs_pair_rmse"] = tier(bytes_eq, p, d, ll, REF_PAIR[pair_key])
    row["tier_vs_all40_mean"] = tier(bytes_eq, p, d, ll, REF_ALL40)
    row["tier_vs_min_model_mean"] = tier(bytes_eq, p, d, ll, REF_MIN)
    row["tier_vs_preB1_ref"] = tier(bytes_eq, p, d, ll, pre)
    # deviation (not pre-registered): d over the informative window only (excludes an out-of-window first lambda)
    row["tier_window_d_DEVIATION"] = tier(bytes_eq, p, row["dlog10ne_max_window"], ll, ref)
    row["ratio_ref_over_d"] = ref / d if d > 0 else math.inf


# --------------------------------------------------------------------------
# psmc-rs: laptop vs Ibex (and the laptop controls: another rustc, a repeat)
# --------------------------------------------------------------------------
RS_JOBS = [("E1", m, r) for m in MODELS for r in REPS] + [("E2", "NA12878", 0)]
LAMBDA_ROWS = []


def lambda_rows(comparison, exp, m, r, fa, fb):
    """One row per fit: the relative difference of every parameter (theta, rho, t_max, each free lambda)."""
    row = {"comparison": comparison, "exp": exp, "model": m, "replicate": r,
           "theta": rel(fa.theta, fb.theta), "rho": rel(fa.rho, fb.rho), "t_max": rel(fa.t_max, fb.t_max)}
    for i, (x, y) in enumerate(zip(fa.free_lambdas, fb.free_lambdas)):
        row[f"lam{i}"] = rel(x, y)
    return row


def rs_compare(runs_a: Path, records_a: Path, runs_b: Path = IBX, ibex_b: bool = True):
    """psmc-rs fits under runs_a (laptop) against runs_b: the committed Ibex fits, or another laptop run."""
    out = []
    for exp, m, r in RS_JOBS:
        row = rs_one(exp, m, r, runs_a, records_a, runs_b, ibex_b)
        if row is not None:
            out.append(row)
    return out


def rs_one(exp, m, r, runs_a, records_a, runs_b, ibex_b):
    if exp == "E1":
        stem = f"E1/fits/{m}/rep{r:02d}"
        prov = IBX / f"E1/provenance/fit-rs/{m}/rep{r:02d}.json"
        rec = records_a / f"E1__fit-rs__{m}__rep{r:02d}.json"
    else:
        stem = "E2/fits/NA12878"
        prov = IBX / "E2/provenance/fit-rs/NA12878.json"
        rec = records_a / "E2__fit-rs__NA12878.json"
    ja, jb = runs_a / f"{stem}.rs.json", runs_b / f"{stem}.rs.json"
    pa_, pb_ = runs_a / f"{stem}.rs.psmc", runs_b / f"{stem}.rs.psmc"
    if not ja.exists() or not jb.exists():
        return None
    ibx_prov = json.loads(prov.read_text())
    ibx_out_ok = (sha(jb) == ibx_prov["outputs"][f"<runs>/{stem}.rs.json"]
                  and sha(pb_) == ibx_prov["outputs"][f"<runs>/{stem}.rs.psmc"]) if ibex_b else None
    lap_rec = json.loads(rec.read_text())
    ma, mb = json.loads(ja.read_text()), json.loads(jb.read_text())
    fa, fb = results.fit_from_rs(ma), results.fit_from_rs(mb)
    scal = (float(mb["mu"]), float(mb["gen_years"]), int(mb["bin_size"]))
    assert scal == (float(ma["mu"]), float(ma["gen_years"]), int(ma["bin_size"]))
    sa = {k: v for k, v in ma.items() if k != "simd"}
    sb = {k: v for k, v in mb.items() if k != "simd"}
    ta = [l for l in ja.read_text().splitlines() if not l.strip().startswith('"simd"')]
    tb = [l for l in jb.read_text().splitlines() if not l.strip().startswith('"simd"')]
    pr = param_rels(fa, fb)
    if ibex_b and runs_a == X / "runs":
        LAMBDA_ROWS.append(lambda_rows("psmc-rs laptop vs Ibex", exp, m, r, fa, fb))
    lla, llb = ma["loglike"], mb["loglike"]
    ll_rels = [rel(x, y) for x, y in zip(lla, llb)]
    ll_first = next((i for i, (x, y) in enumerate(zip(lla, llb)) if x != y), None)
    ita, itb = it_counts(pa_), it_counts(pb_)
    fd = first_diff(pa_, pb_)
    b = "ibex" if ibex_b else "ref"
    row = {
        "exp": exp, "model": m, "replicate": r,
        "simd_laptop": ma.get("simd"), f"simd_{b}": mb.get("simd"),
        "ibex_files_match_provenance": ibx_out_ok,
        "laptop_exit_status": lap_rec["exit_status"],
        "laptop_threads": lap_rec["command"][lap_rec["command"].index("--threads") + 1],
        "ibex_threads": ibx_prov["key"]["command"][ibx_prov["key"]["command"].index("--threads") + 1],
        "json_bytes_identical": ja.read_bytes() == jb.read_bytes(),
        "json_identical_except_simd_label": ta == tb and sa == sb,
        "psmc_bytes_identical": pa_.read_bytes() == pb_.read_bytes(),
        "theta_rel": pr["theta"], "rho_rel": pr["rho"], "t_max_rel": pr["t_max"],
        "lambda_max_rel": pr["lambda_max"], "lambda_argmax": pr["lambda_argmax"], "lambda_median_rel": pr["lambda_median"],
        "param_max_rel": pr["param_max"], "param_max_rel_name": pr["param_max_name"],
        "param_max_rel_all_rounds_psmc": psmc_rounds_param_max(results.parse_psmc(pa_), results.parse_psmc(pb_)),
        **curve_metrics(fa, fb, scal, WINDOW[m] if exp == "E1" else WINDOW["NA12878"]),
        "loglike_rel_final": rel(ma["final_loglike"], mb["final_loglike"]),
        "loglike_abs_final": abs(ma["final_loglike"] - mb["final_loglike"]),
        "loglike_rel_max_trace": max(ll_rels),
        "loglike_iter0_rel": ll_rels[0],
        "loglike_iter0_ulps": round(abs(lla[0] - llb[0]) / math.ulp(llb[0])),
        "loglike_first_diff_iter": ll_first,
        "loglike_iters_identical": sum(x == y for x, y in zip(lla, llb)),
        "n_iter": len(lla) - 1,
        "mstep_evals_laptop": sum(ita), f"mstep_evals_{b}": sum(itb),
        "mstep_counts_differ": sum(x != y for x, y in zip(ita, itb)),
        "psmc_first_diff_line": fd["line"] if fd else None,
        "psmc_first_diff_round": fd["round"] if fd else None,
        "psmc_first_diff_tag": fd["tag"] if fd else None,
        "laptop_wall_seconds": lap_rec["wall_seconds"], "ibex_wall_seconds": ibx_prov["run"]["wall_seconds"],
        "laptop_binary_sha256": lap_rec["binary_sha256"],
    }
    if ibex_b:
        tiers_for(row, row["json_bytes_identical"], m, (m, r))
        row["tier_bytes_excl_simd_label_DEVIATION"] = tier(row["json_identical_except_simd_label"], row["param_max_rel"],
                                                           row["dlog10ne_max_all_intervals"], row["loglike_rel_final"],
                                                           row["ref_rmse_primary"])
    return row


rs_rows = rs_compare(X / "runs", X / "records")
# controls on the laptop: the same sources built by rustc 1.99.0 (Ibex's rustc), and a repeat of two fits
rs199_rows = rs_compare(X / "runs-rustc1.99", X / "records-rustc1.99")
lap199_vs_197 = rs_compare(X / "runs-rustc1.99", X / "records-rustc1.99", X / "runs", ibex_b=False)
repeat_rows = rs_compare(X / "runs-repeat", X / "records-repeat", X / "runs", ibex_b=False)
assert len(rs_rows) == len(rs199_rows) == len(lap199_vs_197) == len(RS_JOBS), "every fit and both controls' fits"
assert len(repeat_rows) == len(json.loads((X / "jobs_repeat.json").read_text())), "the repeat control's fits"

# --------------------------------------------------------------------------
# C psmc: laptop (.production) vs Ibex
# --------------------------------------------------------------------------
c_rows = []
for m in MODELS:
    for r in REPS:
        rp = f"E1/fits/{m}/rep{r:02d}.c.psmc"
        pa_, pb_ = LAP / rp, IBX / rp
        lap_prov = json.loads((LAP / f"E1/provenance/fit-c/{m}/rep{r:02d}.json").read_text())
        ibx_prov = json.loads((IBX / f"E1/provenance/fit-c/{m}/rep{r:02d}.json").read_text())
        inkey = f"<runs>/E1/sims/{m}/rep{r:02d}/genome.psmcfa"
        a, b = results.parse_psmc(pa_), results.parse_psmc(pb_)
        ra, rb = a.complete_rounds(), b.complete_rounds()
        assert [x.round for x in ra] == [x.round for x in rb] and ra[-1].round == 25
        fa, fb = ra[-1].fit(), rb[-1].fit()
        pr = param_rels(fa, fb)
        LAMBDA_ROWS.append(lambda_rows("C psmc laptop vs Ibex", "E1", m, r, fa, fb))
        lka, lkb = a.loglikes(), b.loglikes()
        common = sorted(i for i in lkb if i in lka and lkb[i] != 0)
        ll_rels = {i: rel(lka[i], lkb[i]) for i in common}
        last = max(common)
        ita, itb = it_counts(pa_), it_counts(pb_)
        fd = first_diff(pa_, pb_)
        rd1a, rd1b = next(x for x in ra if x.round == 1), next(x for x in rb if x.round == 1)
        rs_ibx = json.loads((IBX / f"E1/fits/{m}/rep{r:02d}.rs.json").read_text())
        scal = (float(rs_ibx["mu"]), float(rs_ibx["gen_years"]), int(rs_ibx["bin_size"]))
        row = {
            "exp": "E1", "model": m, "replicate": r,
            "laptop_input_sha_eq_ibex": lap_prov["key"]["inputs"][inkey] == ibx_prov["key"]["inputs"][inkey],
            "headers_equal": a.header.get("sum_L") == b.header.get("sum_L") and a.header.get("sum_n") == b.header.get("sum_n")
                             and a.header.get("n_seqs") == b.header.get("n_seqs"),
            "laptop_file_matches_its_record": sha(pa_) == lap_prov["outputs"][f"<runs>/{rp}"],
            "laptop_file_eq_laptop_c_copy": pa_.read_bytes() == (X / "laptop_c" / rp).read_bytes(),
            "ibex_file_matches_its_record": sha(pb_) == ibx_prov["outputs"][f"<runs>/{rp}"],
            "laptop_c_sha256": lap_prov["key"]["tools"]["c-psmc"]["sha256"],
            "ibex_c_sha256": ibx_prov["key"]["tools"]["c-psmc"]["sha256"],
            "psmc_bytes_identical": pa_.read_bytes() == pb_.read_bytes(),
            "theta_rel": pr["theta"], "rho_rel": pr["rho"], "t_max_rel": pr["t_max"],
            "lambda_max_rel": pr["lambda_max"], "lambda_argmax": pr["lambda_argmax"], "lambda_median_rel": pr["lambda_median"],
            "param_max_rel": pr["param_max"], "param_max_rel_name": pr["param_max_name"],
            "param_max_rel_all_rounds_psmc": psmc_rounds_param_max(a, b),
            **curve_metrics(fa, fb, scal, WINDOW[m]),
            "loglike_rel_final": ll_rels[last],  # C prints no LK for the final parameters: iteration 24
            "loglike_final_iter": last,
            "loglike_abs_final": abs(lka[last] - lkb[last]),
            "loglike_rel_max_trace": max(ll_rels.values()),
            "loglike_iter0_rel": ll_rels.get(0),
            "loglike_first_diff_iter": next((i for i in common if lka[i] != lkb[i]), None),
            "loglike_iters_identical": sum(lka[i] == lkb[i] for i in common),
            "round1_lk_and_qd_before_identical": rd1a.lk == rd1b.lk and rd1a.qd[0] == rd1b.qd[0],
            "mstep_evals_laptop": sum(ita), "mstep_evals_ibex": sum(itb),
            "mstep_counts_differ": sum(x != y for x, y in zip(ita, itb)),
            "psmc_first_diff_line": fd["line"] if fd else None,
            "psmc_first_diff_round": fd["round"] if fd else None,
            "psmc_first_diff_tag": fd["tag"] if fd else None,
            "laptop_wall_seconds": lap_prov["run"]["wall_seconds"], "ibex_wall_seconds": ibx_prov["run"]["wall_seconds"],
        }
        tiers_for(row, row["psmc_bytes_identical"], m, (m, r))
        c_rows.append(row)

# --------------------------------------------------------------------------
# check: this code reproduces the committed Ibex psmc-rs vs C numbers
# --------------------------------------------------------------------------
repro = []
rsc_rows = []  # Ibex psmc-rs vs Ibex C psmc, with the same metrics as above (C psmc the reference)
for m in MODELS:
    for r in REPS:
        rs_ibx = json.loads((IBX / f"E1/fits/{m}/rep{r:02d}.rs.json").read_text())
        frs = results.fit_from_rs(rs_ibx)
        cfile = results.parse_psmc(IBX / f"E1/fits/{m}/rep{r:02d}.c.psmc")
        fc = cfile.last().fit()
        scal = (float(rs_ibx["mu"]), float(rs_ibx["gen_years"]), int(rs_ibx["bin_size"]))
        d = metrics.interval_diff(frs.curve(*scal), fc.curve(*scal), WINDOW[m])
        p = pairs[(m, r)]
        repro.append(max(abs(d.rmse - float(p["rmse_log10_ne_rs_vs_c_per_interval"])),
                         abs(d.max_abs - float(p["max_abs_log10_ne_rs_vs_c_per_interval"])),
                         abs(d.max_abs_any - float(p["max_abs_log10_ne_rs_vs_c_any_interval"]))))
        pr = param_rels(frs, fc)
        lrs = {i: v for i, v in enumerate(rs_ibx["loglike"])}
        lc = cfile.loglikes()
        common = sorted(i for i in lc if i in lrs and lc[i] != 0)
        llr = {i: rel(lrs[i], lc[i]) for i in common}
        rsc_rows.append({
            "exp": "E1", "model": m, "replicate": r,
            "theta_rel": pr["theta"], "rho_rel": pr["rho"], "t_max_rel": pr["t_max"],
            "lambda_max_rel": pr["lambda_max"], "lambda_argmax": pr["lambda_argmax"], "lambda_median_rel": pr["lambda_median"],
            "param_max_rel": pr["param_max"], "param_max_rel_name": pr["param_max_name"],
            "param_max_rel_all_rounds_psmc": psmc_rounds_param_max(results.parse_psmc(IBX / f"E1/fits/{m}/rep{r:02d}.rs.psmc"), cfile),
            **curve_metrics(frs, fc, scal, WINDOW[m]),
            "loglike_rel_final": llr[max(common)],
            "loglike_final_iter": max(common),
            "loglike_abs_final": abs(lrs[max(common)] - lc[max(common)]),
            "loglike_rel_max_trace": max(llr.values()),
            "loglike_iter0_rel": llr.get(0),
        })
repro_max = max(repro)
rsc_by = {(x["model"], x["replicate"]): x for x in rsc_rows}

# --------------------------------------------------------------------------
# side by side (E1): the three agreements per replicate
# --------------------------------------------------------------------------
rs_by = {(x["model"], x["replicate"]): x for x in rs_rows if x["exp"] == "E1"}
c_by = {(x["model"], x["replicate"]): x for x in c_rows}
side = []
for m in MODELS:
    for r in REPS:
        p, a, c = pairs[(m, r)], rs_by.get((m, r)), c_by[(m, r)]
        if a is None:
            continue
        rs_c_rmse = float(p["rmse_log10_ne_rs_vs_c_per_interval"])
        side.append({
            "model": m, "replicate": r,
            "window_lo_years": WINDOW[m][0], "window_hi_years": WINDOW[m][1],
            "rmse_window_rs_laptop_vs_ibex": a["dlog10ne_rmse_window"],
            "rmse_window_c_laptop_vs_ibex": c["dlog10ne_rmse_window"],
            "rmse_window_rs_vs_c_ibex": rs_c_rmse,
            "max_window_rs_laptop_vs_ibex": a["dlog10ne_max_window"],
            "max_window_c_laptop_vs_ibex": c["dlog10ne_max_window"],
            "max_window_rs_vs_c_ibex": float(p["max_abs_log10_ne_rs_vs_c_per_interval"]),
            "max_any_rs_laptop_vs_ibex": a["dlog10ne_max_all_intervals"],
            "max_any_c_laptop_vs_ibex": c["dlog10ne_max_all_intervals"],
            "max_any_rs_vs_c_ibex": float(p["max_abs_log10_ne_rs_vs_c_any_interval"]),
            "theta_rel_rs_laptop_vs_ibex": a["theta_rel"],
            "theta_rel_c_laptop_vs_ibex": c["theta_rel"],
            "theta_rel_rs_vs_c_ibex": rsc_by[(m, r)]["theta_rel"],
            "param_max_rel_rs_laptop_vs_ibex": a["param_max_rel"],
            "param_max_rel_c_laptop_vs_ibex": c["param_max_rel"],
            "param_max_rel_rs_vs_c_ibex": rsc_by[(m, r)]["param_max_rel"],
            "loglike_rel_max_rs_laptop_vs_ibex": a["loglike_rel_max_trace"],
            "loglike_rel_max_c_laptop_vs_ibex": c["loglike_rel_max_trace"],
            "loglike_rel_max_rs_vs_c_ibex": float(p["loglike_max_rel_diff"]),
            "ratio_rmse_rsvsc_over_rs_xplat": rs_c_rmse / a["dlog10ne_rmse_window"] if a["dlog10ne_rmse_window"] > 0 else math.inf,
            "ratio_rmse_rsvsc_over_c_xplat": rs_c_rmse / c["dlog10ne_rmse_window"] if c["dlog10ne_rmse_window"] > 0 else math.inf,
            "ratio_rmse_c_xplat_over_rs_xplat": c["dlog10ne_rmse_window"] / a["dlog10ne_rmse_window"] if a["dlog10ne_rmse_window"] > 0 else math.inf,
            "tier_rs": a["tier"], "tier_c": c["tier"],
        })


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


write(X / "rs_laptop_vs_ibex.csv", rs_rows)
write(X / "c_laptop_vs_ibex.csv", c_rows)
write(X / "rs_vs_c_ibex.csv", rsc_rows)
write(X / "params_rel_per_fit.csv", LAMBDA_ROWS)
for name, rr in (("rs_laptop_rustc1.99_vs_ibex.csv", rs199_rows),
                 ("control_rustc1.99_vs_rustc1.97_laptop.csv", lap199_vs_197),
                 ("control_repeat_laptop.csv", repeat_rows)):
    if rr:
        write(X / name, rr)
if side:
    write(X / "side_by_side_E1.csv", side)


def stats(rows, key):
    v = [float(x[key]) for x in rows if x[key] is not None]
    return {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v)} if v else None


def block(rows):
    if not rows:
        return None
    keys = ["theta_rel", "rho_rel", "t_max_rel", "lambda_max_rel", "lambda_median_rel", "param_max_rel",
            "param_max_rel_all_rounds_psmc", "dlog10ne_max_all_intervals", "dlog10ne_max_window",
            "dlog10ne_rmse_window", "dlog10_t_boundaries_max", "loglike_rel_final", "loglike_rel_max_trace",
            "loglike_iter0_rel"]
    out = {k: stats(rows, k) for k in keys}
    for t in ("tier", "tier_vs_pair_rmse", "tier_vs_all40_mean", "tier_vs_min_model_mean", "tier_vs_preB1_ref",
              "tier_window_d_DEVIATION", "tier_bytes_excl_simd_label_DEVIATION"):
        if t in rows[0]:
            out[t] = dict(Counter(x[t] for x in rows))
    return out


e1rs = [x for x in rs_rows if x["exp"] == "E1"]


def runaway():
    """C psmc's fit with the largest parameter difference, and psmc-rs's same lambda on that replicate, round by round."""
    c_top = max(c_rows, key=lambda x: x["param_max_rel"])
    m, r, name = c_top["model"], c_top["replicate"], c_top["param_max_rel_name"]
    if not name.startswith("lam"):
        return {"model": m, "replicate": r, "c_param": name}
    i = int(name[3:])
    stem = f"E1/fits/{m}/rep{r:02d}"
    ja, jb = (json.loads((root / f"{stem}.rs.json").read_text()) for root in (X / "runs", IBX))
    fa, fb = results.fit_from_rs(ja), results.fit_from_rs(jb)
    ra, rb = (results.parse_psmc(root / f"{stem}.rs.psmc").complete_rounds() for root in (X / "runs", IBX))
    per_round = [(x.round, rel(x.fit().free_lambdas[i], y.fit().free_lambdas[i])) for x, y in zip(ra, rb) if x.pa is not None]
    worst = max(per_round, key=lambda t: t[1])
    ca, cb = results.parse_psmc(LAP / f"{stem}.c.psmc").last().fit(), results.parse_psmc(IBX / f"{stem}.c.psmc").last().fit()
    return {
        "model": m, "replicate": r, "c_param": name, "lambda_index": i,
        "c_lambda_laptop": ca.free_lambdas[i], "c_lambda_ibex": cb.free_lambdas[i],
        "c_param_max_rel": c_top["param_max_rel"], "c_d": c_top["dlog10ne_max_all_intervals"],
        "c_max_window": c_top["dlog10ne_max_window"],
        "rs_lambda_laptop": fa.free_lambdas[i], "rs_lambda_ibex": fb.free_lambdas[i],
        "rs_rel_final": rel(fa.free_lambdas[i], fb.free_lambdas[i]),
        "rs_rel_max_over_rounds": worst[1], "rs_rel_max_round": worst[0],
    }


summary = {
    "references": {"model_mean_primary": REF_MODEL, "all40_mean": REF_ALL40, "min_model_mean": REF_MIN,
                   "pre_B1_model_mean_superseded": REF_PRE_B1_MODEL, "pre_B1_all40_superseded": REF_PRE_B1_ALL40,
                   "NA12878_pair": REF_PAIR[("NA12878", 0)], "windows": WINDOW},
    "repro_committed_rs_vs_c_max_abs_error": repro_max,
    "runaway": runaway(),
    "rs": {
        "n": len(rs_rows), "n_E1": len(e1rs),
        "json_bytes_identical": sum(x["json_bytes_identical"] for x in rs_rows),
        "json_identical_except_simd_label": sum(x["json_identical_except_simd_label"] for x in rs_rows),
        "psmc_bytes_identical": sum(x["psmc_bytes_identical"] for x in rs_rows),
        "simd": {f"{k[0]} vs {k[1]}": v for k, v in Counter((x["simd_laptop"], x["simd_ibex"]) for x in rs_rows).items()},
        "ibex_files_match_provenance": sum(x["ibex_files_match_provenance"] for x in rs_rows),
        "threads": {f"{k[0]} vs {k[1]}": v for k, v in Counter((x["laptop_threads"], x["ibex_threads"]) for x in rs_rows).items()},
        "loglike_iter0_identical": sum(x["loglike_iter0_rel"] == 0 for x in rs_rows),
        "loglike_first_diff_iter": dict(Counter(x["loglike_first_diff_iter"] for x in rs_rows)),
        "psmc_first_diff": dict(Counter(f"round {x['psmc_first_diff_round']} {x['psmc_first_diff_tag']}" for x in rs_rows)),
        "E1": block(e1rs),
        "E1_per_model": {m: block([x for x in e1rs if x["model"] == m]) for m in MODELS},
        "NA12878": next((x for x in rs_rows if x["exp"] == "E2"), None),
    },
    "c": {
        "n": len(c_rows),
        "inputs_identical": sum(x["laptop_input_sha_eq_ibex"] for x in c_rows),
        "headers_equal": sum(x["headers_equal"] for x in c_rows),
        "laptop_files_match_record": sum(x["laptop_file_matches_its_record"] for x in c_rows),
        "laptop_files_eq_laptop_c_copy": sum(x["laptop_file_eq_laptop_c_copy"] for x in c_rows),
        "ibex_files_match_record": sum(x["ibex_file_matches_its_record"] for x in c_rows),
        "binaries": {f"laptop {k[0][:12]} / ibex {k[1][:12]}": v for k, v in Counter((x["laptop_c_sha256"], x["ibex_c_sha256"]) for x in c_rows).items()},
        "psmc_bytes_identical": sum(x["psmc_bytes_identical"] for x in c_rows),
        "round1_lk_and_qd_before_identical": sum(x["round1_lk_and_qd_before_identical"] for x in c_rows),
        "loglike_iter0_identical": sum(x["loglike_iter0_rel"] == 0 for x in c_rows),
        "psmc_first_diff": dict(Counter(f"round {x['psmc_first_diff_round']} {x['psmc_first_diff_tag']}" for x in c_rows)),
        "E1": block(c_rows),
        "E1_per_model": {m: block([x for x in c_rows if x["model"] == m]) for m in MODELS},
    },
    "rs_rustc1.99": ({
        "n": len(rs199_rows),
        "json_bytes_identical": sum(x["json_bytes_identical"] for x in rs199_rows),
        "json_identical_except_simd_label": sum(x["json_identical_except_simd_label"] for x in rs199_rows),
        "psmc_bytes_identical": sum(x["psmc_bytes_identical"] for x in rs199_rows),
        "loglike_iter0_identical": sum(x["loglike_iter0_rel"] == 0 for x in rs199_rows),
        "E1": block([x for x in rs199_rows if x["exp"] == "E1"]),
        "E1_per_model": {m: block([x for x in rs199_rows if x["exp"] == "E1" and x["model"] == m]) for m in MODELS},
        "NA12878": next((x for x in rs199_rows if x["exp"] == "E2"), None),
    } if rs199_rows else None),
    "control_rustc1.99_vs_rustc1.97_laptop": ({
        "n": len(lap199_vs_197),
        "json_bytes_identical": sum(x["json_bytes_identical"] for x in lap199_vs_197),
        "psmc_bytes_identical": sum(x["psmc_bytes_identical"] for x in lap199_vs_197),
        "binaries": sorted({x["laptop_binary_sha256"] for x in lap199_vs_197}),
        "E1": block([x for x in lap199_vs_197 if x["exp"] == "E1"]),
    } if lap199_vs_197 else None),
    "control_repeat_laptop": ({
        "n": len(repeat_rows),
        "fits": [f"{x['model']} rep{x['replicate']:02d}" if isinstance(x['replicate'], int) else x['model'] for x in repeat_rows],
        "json_bytes_identical": sum(x["json_bytes_identical"] for x in repeat_rows),
        "psmc_bytes_identical": sum(x["psmc_bytes_identical"] for x in repeat_rows),
    } if repeat_rows else None),
    "rs_vs_c_ibex": {"E1": block(rsc_rows), "E1_per_model": {m: block([x for x in rsc_rows if x["model"] == m]) for m in MODELS}},
    "side_by_side": {k: stats(side, k) for k in (side[0] if side else {}) if k.startswith(("rmse_", "max_", "ratio_", "theta_", "param_", "loglike_"))},
    "side_by_side_counts": {
        "rs_xplat_rmse_below_rs_vs_c": sum(s["rmse_window_rs_laptop_vs_ibex"] < s["rmse_window_rs_vs_c_ibex"] for s in side),
        "c_xplat_rmse_below_rs_vs_c": sum(s["rmse_window_c_laptop_vs_ibex"] < s["rmse_window_rs_vs_c_ibex"] for s in side),
        "rs_xplat_rmse_below_c_xplat": sum(s["rmse_window_rs_laptop_vs_ibex"] < s["rmse_window_c_laptop_vs_ibex"] for s in side),
        "n": len(side),
    },
}
(X / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("references", "repro_committed_rs_vs_c_max_abs_error")}, indent=1, default=str))
for tool in ("rs", "c"):
    s = summary[tool]
    print(f"== {tool}: n={s['n']}")
    for k, v in s.items():
        if k in ("E1", "E1_per_model", "NA12878"):
            continue
        print(f"  {k}: {v}")
    for k, v in s["E1"].items():
        print(f"  E1 {k}: {v}")
print("== side by side")
for k, v in summary["side_by_side"].items():
    print(f"  {k}: {v}")
print(summary["side_by_side_counts"])
