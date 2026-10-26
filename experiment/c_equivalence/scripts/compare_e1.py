"""Each variant's fit of E1's 40 genomes against C psmc's committed fit, as E1_rs_vs_c compares them: e1_variants.csv.

Usage: experiment/.venv/bin/python -I -B scripts/compare_e1.py

One row per genome and variant: default is psmc-rs's committed fit (experiment/runs/E1/fits/*/repNN.rs.*),
whose row must reproduce E1_rs_vs_c.csv; lastbin, hj, both and prec are the fits of run_e1.py (fits/). The
columns are E1's comparison with C psmc (the pipeline's e1.compare_tools and e1.loglike_diffs on the genome's
window), the first lambda of both programs, the M-step's evaluations per EM iteration of both (it_*, c_it_*, from
the .psmc files), and the same comparison against the default fit (*_vs_default).
"""
import csv
import json
import math
import sys
from pathlib import Path

C = Path(__file__).resolve().parent.parent
REPO = C.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment.pipeline import config, demography, metrics, results  # noqa: E402
from experiment.pipeline.experiments import e1  # noqa: E402
from psmcfile import rounds  # noqa: E402

VARIANTS = ["default", "lastbin", "hj", "both", "prec"]
cfg = config.get_profile("production").e1
lay = e1.Layout(REPO / "experiment" / "runs" / "E1")
n_iter, sc, bin_size, g = cfg.inference.n_iter, cfg.scaling, cfg.inference.bin_size, cfg.scaling.gen_years
committed = {(row["model"], int(row["replicate"])): row for row in csv.DictReader(open(lay.table("pairs")))}


def evaluations(path):
    it = [x["it"] for x in rounds(path)[1:]]
    return {"it_min": min(it), "it_max": max(it), "it_total": sum(it)}


rows = []
for model in cfg.models:
    for r in range(1, cfg.replicates + 1):
        truth = json.loads(lay.truth(model, r).read_text())
        window = metrics.model_grid(demography.epochs_from_record(truth["truth"]), cfg.grid, g)
        w = (float(window[0]), float(window[-1]))
        c = e1.read_c_fit(lay.c_psmc(model, r), n_iter)
        fc = c.last().fit()
        cc = fc.curve(sc.mu, g, bin_size)
        fits = {}
        for v in VARIANTS:
            if v == "default":
                js, ps = lay.rs_json(model, r), lay.rs_psmc(model, r)
            else:
                js, ps = C / "fits" / model / f"{e1.rep(r)}.{v}.json", C / "fits" / model / f"{e1.rep(r)}.{v}.psmc"
            _, _, params = e1.read_fits(lay.c_psmc(model, r), ps, js, n_iter)
            e1.check_same_input(c, params, f"{model} {r} {v}")
            f = results.fit_from_rs(params)
            fits[v] = (params, f, f.curve(sc.mu, g, bin_size), ps)
        dp, df, dcu, _ = fits["default"]
        for v, (params, f, cu, ps) in fits.items():
            row = {"model": model, "replicate": r, "variant": v,
                   **e1.compare_tools({"c": fc, "rs": f}, {"c": cc, "rs": cu}, w),
                   **e1.loglike_diffs(e1.loglikes(c, params), n_iter),
                   "loglike_final": results.rs_loglikes(params)[-1],
                   "lam0": params["lam"][0], "c_lam0": fc.lambdas[0], **evaluations(ps),
                   **{f"c_{k}": x for k, x in evaluations(lay.c_psmc(model, r)).items()}}
            dvd = metrics.interval_diff(cu, dcu, w)
            row.update({"rmse_log10_ne_vs_default_per_interval": dvd.rmse,
                        "max_abs_log10_ne_vs_default_per_interval": dvd.max_abs,
                        "max_abs_log10_ne_vs_default_any_interval": dvd.max_abs_any,
                        "theta_rel_vs_default": f.theta / df.theta - 1, "rho_rel_vs_default": f.rho / df.rho - 1,
                        "t_max_rel_vs_default": f.t_max / df.t_max - 1,
                        "loglike_final_minus_default": results.rs_loglikes(params)[-1] - results.rs_loglikes(dp)[-1]})
            if v == "default":
                ref = committed[(model, r)]
                for k in ("rmse_log10_ne_rs_vs_c_per_interval", "max_abs_log10_ne_rs_vs_c_per_interval",
                          "max_abs_log10_ne_rs_vs_c_any_interval", "theta_rel_diff", "rho_rel_diff", "t_max_rel_diff",
                          "loglike_max_rel_diff", "loglike_common_diff"):
                    assert math.isclose(float(ref[k]), row[k], rel_tol=1e-9, abs_tol=0), (model, r, k, ref[k], row[k])
            rows.append(row)
with open(C / "e1_variants.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print(f"e1_variants.csv: {len(rows)} rows; the default rows reproduce E1_rs_vs_c.csv (relative 1e-9)")
