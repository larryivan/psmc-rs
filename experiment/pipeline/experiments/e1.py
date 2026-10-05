"""E1: simulated accuracy and equivalence (Fig 2, Table 1, Fig S1).

For every model and replicate (:class:`.config.E1`) the runner simulates one
diploid genome, fits it with C psmc and with psmc-rs 0.2 (identical psmcfa,
identical settings), and finally writes the tables. Layout under
``<runs>/E1/``::

    sims/<model>/repNN/genome.psmcfa, genome.vcf.gz(.tbi)   simulated genome (not committed)
    truth/<model>/repNN.json                               seeds, counts, checksums, true Ne(t)
    fits/<model>/repNN.c.psmc                              C psmc, every round
    fits/<model>/repNN.rs.json, repNN.rs.psmc              psmc-rs 0.2 (JSON and .psmc)
    provenance/<job>.json                                  one record per job
    tables/E1_*.csv                                        written by ``report E1``
    logs/                                                  runner and job logs (not committed)

Tables (``tables/``):

* ``E1_table1.csv``: Table 1, one row per model: RMSE of log10 Ne against
  the truth for each tool (mean and SD over replicates) on the model's
  informative window and on the fixed window, RMSE between the tools, and
  the largest differences in log-likelihood and theta.
* ``E1_replicates.csv``: one row per model, replicate and tool: final theta,
  rho, t_max and N0, RMSE against the truth, log-likelihoods, and the run's
  wall time, peak RSS and load (recorded, not used for timing claims).
* ``E1_rs_vs_c.csv``: one row per model and replicate: the two tools compared.
* ``E1_traces.csv``: EM traces (Fig S1): log-likelihood, theta, rho and
  t_max after each iteration, for both tools.
* ``E1_curves.csv``: the final Ne(t) step curves of every fit (Fig 2).
* ``E1_truth.csv``: the true Ne(t) of every model on a dense grid (Fig 2).
* ``E1_seeds.csv``: every simulation seed, with each chromosome's
  heterozygous sites, trees and longest stretch without a heterozygous
  site (a long recent-coalescence segment makes a replicate an outlier).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .. import config, demography, metrics, provenance, results, simulate, tools
from ..runner import Job
from . import Context

NAME = "E1"
TOOLS = ("c", "rs")
TABLES = {
    "table1": "E1_table1.csv",
    "replicates": "E1_replicates.csv",
    "pairs": "E1_rs_vs_c.csv",
    "traces": "E1_traces.csv",
    "curves": "E1_curves.csv",
    "truth": "E1_truth.csv",
    "seeds": "E1_seeds.csv",
}
TRUTH_GRID_YEARS = (1e2, 1e8, 400)  # dense grid of E1_truth.csv


def rep(r: int) -> str:
    return f"rep{r:02d}"


@dataclass(frozen=True)
class Layout:
    """Where E1's files are."""

    root: Path

    def sim_dir(self, model: str, r: int) -> Path:
        return self.root / "sims" / model / rep(r)

    def psmcfa(self, model: str, r: int) -> Path:
        return self.sim_dir(model, r) / simulate.PSMCFA_NAME

    def vcf(self, model: str, r: int) -> Path:
        return self.sim_dir(model, r) / simulate.VCF_NAME

    def truth(self, model: str, r: int) -> Path:
        return self.root / "truth" / model / f"{rep(r)}.json"

    def c_psmc(self, model: str, r: int) -> Path:
        return self.root / "fits" / model / f"{rep(r)}.c.psmc"

    def rs_json(self, model: str, r: int) -> Path:
        return self.root / "fits" / model / f"{rep(r)}.rs.json"

    def rs_psmc(self, model: str, r: int) -> Path:
        return self.root / "fits" / model / f"{rep(r)}.rs.psmc"

    def record(self, kind: str, model: str, r: int) -> Path:
        return self.root / "provenance" / kind / model / f"{rep(r)}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_id(kind: str, model: str, r: int) -> str:
    return f"{NAME}/{kind}/{model}/{rep(r)}"


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e1
    costs = ctx.profile.costs
    lay = Layout(ctx.exp_dir(NAME))
    genome_mb = cfg.genome.length / 1e6
    chrom_mb = cfg.genome.chromosome_length / 1e6
    inference = asdict(cfg.inference)
    scaling = asdict(cfg.scaling)
    sim_tools = {
        "pipeline": provenance.python_identity(
            ["simulate.py", "demography.py"], ["msprime", "tskit", "numpy", "stdpopsim"]
        ),
        "htslib": tools.htslib_versions(),
    }
    c_tool, rs_tool = ctx.c_psmc(), ctx.rs02()
    out: List[Job] = []
    for model in cfg.models:
        for r in range(1, cfg.replicates + 1):
            psmcfa = lay.psmcfa(model, r)
            out.append(
                Job(
                    id=job_id("sim", model, r),
                    command=ctx.pipeline_cmd(
                        "simulate", "--profile", ctx.profile.name, "--experiment", NAME, "--model", model,
                        "--replicate", str(r), "--out-dir", str(lay.sim_dir(model, r)), "--truth", str(lay.truth(model, r)),
                    ),
                    outputs=[psmcfa, lay.vcf(model, r), Path(f"{lay.vcf(model, r)}.tbi"), lay.truth(model, r)],
                    mem_gb=costs.simulate_gb,
                    est_seconds=costs.simulate_s_base + costs.simulate_s_per_mb * genome_mb,
                    settings={
                        "model": asdict(config.MODELS[model]),
                        "genome": asdict(cfg.genome),
                        "simulation": asdict(cfg.simulation),
                    },
                    tools=sim_tools,
                    seeds={c: simulate.seeds_for(NAME, model, r, c) for c in cfg.genome.names()},
                    description=f"simulate {model} replicate {r}",
                )
            )
            out.append(
                Job(
                    id=job_id("fit-c", model, r),
                    command=tools.c_psmc_fit(c_tool, cfg.inference, psmcfa, lay.c_psmc(model, r)),
                    inputs=[psmcfa],
                    outputs=[lay.c_psmc(model, r)],
                    mem_gb=round(costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * chrom_mb, 2),
                    est_seconds=costs.c_psmc_s_per_mb_iter * genome_mb * cfg.inference.n_iter,
                    settings={"inference": inference},
                    tools={c_tool.name: c_tool.identity},
                    description=f"C psmc on {model} replicate {r}",
                )
            )
            out.append(
                Job(
                    id=job_id("fit-rs", model, r),
                    command=tools.rs02_fit(
                        rs_tool, cfg.inference, cfg.scaling, psmcfa, lay.rs_json(model, r),
                        threads=cfg.rs_threads, psmc_out=lay.rs_psmc(model, r),
                    ),
                    inputs=[psmcfa],
                    outputs=[lay.rs_json(model, r), lay.rs_psmc(model, r)],
                    cores=cfg.rs_threads,
                    mem_gb=costs.rs02_gb,
                    est_seconds=costs.rs02_s_per_mb_iter * genome_mb * cfg.inference.n_iter / cfg.rs_threads,
                    settings={"inference": inference, "scaling": scaling, "threads": cfg.rs_threads},
                    tools={rs_tool.name: rs_tool.identity},
                    description=f"psmc-rs 0.2 on {model} replicate {r}",
                )
            )
    report_inputs: List[Path] = []
    for model in cfg.models:
        for r in range(1, cfg.replicates + 1):
            report_inputs += [
                lay.truth(model, r), lay.c_psmc(model, r), lay.rs_json(model, r), lay.rs_psmc(model, r),
                lay.record("fit-c", model, r), lay.record("fit-rs", model, r),
            ]
    out.append(
        Job(
            id=f"{NAME}/report",
            command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
            inputs=report_inputs,
            outputs=[lay.table(k) for k in TABLES],
            mem_gb=costs.report_gb,
            est_seconds=costs.report_s,
            settings={"e1": asdict(cfg)},
            tools={"pipeline": provenance.python_identity(
                ["experiments/e1.py", "results.py", "metrics.py", "demography.py"], ["numpy"]
            )},
            description="E1 tables",
        )
    )
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return "nan" if math.isnan(x) else repr(x)
    return str(x)


def write_csv(path: Path, rows: Sequence[Dict], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or (rows[0].keys() if rows else []))
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _num(row.get(k)) for k in fields})
    tmp.replace(path)


@dataclass
class Replicate:
    """The fits of one simulated genome, read back."""

    model: str
    replicate: int
    truth: Dict
    epochs: List[demography.Epoch]
    c: results.PsmcFile
    rs_params: Dict
    rs: results.PsmcFile
    fits: Dict[str, results.Fit]
    curves: Dict[str, results.Curve]
    runs: Dict[str, Dict]


def load_replicate(lay: Layout, cfg: config.E1, model: str, r: int) -> Replicate:
    truth = json.loads(lay.truth(model, r).read_text())
    c = results.parse_psmc(lay.c_psmc(model, r))
    rs = results.parse_psmc(lay.rs_psmc(model, r))
    rs_params = results.read_rs_json(lay.rs_json(model, r))
    n_iter = cfg.inference.n_iter
    if c.last().round != n_iter:
        raise ValueError(f"{lay.c_psmc(model, r)}: last complete round {c.last().round}, expected {n_iter}")
    if int(rs_params.get("n_iter", -1)) != n_iter or rs.last().round != n_iter:
        raise ValueError(f"{lay.rs_json(model, r)}: not a fit of {n_iter} iterations")
    het_bins = sum(ch["het_bins"] for ch in truth["chromosomes"])
    bins = sum(ch["bins"] for ch in truth["chromosomes"])
    inp = rs_params.get("input", {})
    if not (c.header.get("sum_n") == het_bins == inp.get("het_bins") and c.header.get("sum_L") == bins == inp.get("bins")):
        raise ValueError(
            f"{model} {rep(r)}: the fits did not read the simulated genome "
            f"(truth {bins} bins/{het_bins} het, C {c.header.get('sum_L')}/{c.header.get('sum_n')}, "
            f"psmc-rs {inp.get('bins')}/{inp.get('het_bins')})"
        )
    fits = {"c": c.last().fit(), "rs": results.fit_from_rs(rs_params)}
    sc = cfg.scaling
    curves = {k: f.curve(sc.mu, sc.gen_years, cfg.inference.bin_size) for k, f in fits.items()}
    runs = {}
    for tool, kind in (("c", "fit-c"), ("rs", "fit-rs")):
        try:
            runs[tool] = json.loads(lay.record(kind, model, r).read_text()).get("run", {})
        except (OSError, ValueError):
            runs[tool] = {}
    return Replicate(
        model, r, truth, demography.epochs_from_record(truth["truth"]), c, rs_params, rs, fits, curves, runs
    )


def loglikes(rp: Replicate) -> Dict[str, Dict[int, float]]:
    """``{tool: {iterations: log-likelihood}}``."""
    return {
        "c": rp.c.loglikes(),
        "rs": {i: v for i, v in enumerate(results.rs_loglikes(rp.rs_params))},
    }


def replicate_rows(rp: Replicate, cfg: config.E1, window, fixed) -> Dict[str, List[Dict]]:
    import numpy as np

    g = cfg.scaling.gen_years
    n = cfg.inference.n_iter
    ll = loglikes(rp)
    rows: Dict[str, List[Dict]] = {"replicates": [], "pairs": [], "traces": [], "curves": []}
    for tool in TOOLS:
        fit, curve, run = rp.fits[tool], rp.curves[tool], rp.runs[tool]
        rows["replicates"].append(
            {
                "model": rp.model,
                "replicate": rp.replicate,
                "tool": tool,
                "theta": fit.theta,
                "rho": fit.rho,
                "t_max": fit.t_max,
                "n0": fit.n0(cfg.scaling.mu, cfg.inference.bin_size),
                "rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(curve, rp.epochs, window, g),
                "rmse_log10_ne_vs_truth_fixed": metrics.rmse_vs_truth(curve, rp.epochs, fixed, g),
                "loglike_common": ll[tool].get(n - 1),
                "loglike_final": ll[tool].get(n),
                "wall_seconds": run.get("wall_seconds"),
                "peak_rss_mb": run.get("peak_rss_mb"),
                "user_seconds": run.get("user_seconds"),
                "loadavg_before_1m": (run.get("loadavg_before") or [None])[0],
                "run_start": run.get("start"),
            }
        )
        boundaries = fit.boundaries()
        for k, (t, y, lam, ne) in enumerate(zip(boundaries, curve.years, fit.lambdas, curve.ne)):
            rows["curves"].append(
                {"model": rp.model, "replicate": rp.replicate, "tool": tool, "interval": k,
                 "t_2n0": t, "years": y, "lambda": lam, "ne": ne}
            )
    psmc_files = {"c": rp.c, "rs": rp.rs}
    for tool in TOOLS:
        by_round = {r.round: r for r in psmc_files[tool].complete_rounds()}
        for i in range(0, n + 1):
            rd = by_round.get(i)
            rows["traces"].append(
                {
                    "model": rp.model, "replicate": rp.replicate, "tool": tool, "iteration": i,
                    "loglike": ll[tool].get(i),
                    "theta": rd.theta if rd else None, "rho": rd.rho if rd else None,
                    "t_max": rd.t_max if rd else None,
                }
            )
    common = sorted(i for i in set(ll["c"]) & set(ll["rs"]) if ll["c"][i])
    rel = {i: abs(ll["rs"][i] - ll["c"][i]) / abs(ll["c"][i]) for i in common}
    fc, fr = rp.fits["c"], rp.fits["rs"]
    diff_w = metrics.rmse_between(rp.curves["rs"], rp.curves["c"], window)
    rows["pairs"].append(
        {
            "model": rp.model,
            "replicate": rp.replicate,
            "window_lo_years": float(window[0]),
            "window_hi_years": float(window[-1]),
            "rmse_log10_ne_rs_vs_c": diff_w,
            "rmse_log10_ne_rs_vs_c_fixed": metrics.rmse_between(rp.curves["rs"], rp.curves["c"], fixed),
            "max_abs_log10_ne_diff": float(
                np.max(np.abs(np.log10(rp.curves["rs"].at(window)) - np.log10(rp.curves["c"].at(window))))
            ),
            "theta_rel_diff": fr.theta / fc.theta - 1.0,
            "rho_rel_diff": fr.rho / fc.rho - 1.0,
            "t_max_rel_diff": fr.t_max / fc.t_max - 1.0,
            "loglike_common_diff": (ll["rs"].get(n - 1) - ll["c"].get(n - 1)) if (n - 1) in common else None,
            "loglike_max_rel_diff": max(rel.values()) if rel else None,
            "loglike_iteration0_rel_diff": rel.get(0),
        }
    )
    return rows


def _mean_sd(xs: Sequence[float]):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def table1_row(model: str, window, reps: List[Dict], pairs: List[Dict]) -> Dict:
    def col(tool: str, key: str) -> List[float]:
        return [r[key] for r in reps if r["tool"] == tool]

    c_mean, c_sd = _mean_sd(col("c", "rmse_log10_ne_vs_truth"))
    r_mean, r_sd = _mean_sd(col("rs", "rmse_log10_ne_vs_truth"))
    cf_mean, cf_sd = _mean_sd(col("c", "rmse_log10_ne_vs_truth_fixed"))
    rf_mean, rf_sd = _mean_sd(col("rs", "rmse_log10_ne_vs_truth_fixed"))
    d = [p["rmse_log10_ne_rs_vs_c"] for p in pairs]
    return {
        "model": model,
        "replicates": len(pairs),
        "window_lo_years": float(window[0]),
        "window_hi_years": float(window[-1]),
        "rmse_truth_c_mean": c_mean,
        "rmse_truth_c_sd": c_sd,
        "rmse_truth_rs_mean": r_mean,
        "rmse_truth_rs_sd": r_sd,
        "rmse_truth_fixed_c_mean": cf_mean,
        "rmse_truth_fixed_c_sd": cf_sd,
        "rmse_truth_fixed_rs_mean": rf_mean,
        "rmse_truth_fixed_rs_sd": rf_sd,
        "rmse_rs_vs_c_mean": statistics.fmean(d) if d else None,
        "rmse_rs_vs_c_max": max(d) if d else None,
        "loglike_max_rel_diff": max((p["loglike_max_rel_diff"] for p in pairs if p["loglike_max_rel_diff"] is not None), default=None),
        "theta_max_abs_rel_diff": max((abs(p["theta_rel_diff"]) for p in pairs), default=None),
    }


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E1 table from the fits; returns the paths written."""
    import numpy as np

    cfg = ctx.profile.e1
    lay = Layout(ctx.exp_dir(NAME))
    fixed = metrics.fixed_grid(cfg.grid)
    all_rows: Dict[str, List[Dict]] = {k: [] for k in ("replicates", "pairs", "traces", "curves")}
    table1: List[Dict] = []
    truth_rows: List[Dict] = []
    seed_rows: List[Dict] = []
    lo, hi, npts = TRUTH_GRID_YEARS
    dense = np.logspace(math.log10(lo), math.log10(hi), npts)
    g = cfg.scaling.gen_years
    for model in cfg.models:
        reps_rows: List[Dict] = []
        pair_rows: List[Dict] = []
        window = None
        for r in range(1, cfg.replicates + 1):
            rp = load_replicate(lay, cfg, model, r)
            if window is None:
                window = metrics.model_grid(rp.epochs, cfg.grid, g)
                for t, ne in zip(dense, metrics.truth_on_grid(rp.epochs, dense, g)):
                    truth_rows.append({"model": model, "years": float(t), "ne": float(ne)})
            rows = replicate_rows(rp, cfg, window, fixed)
            for k, v in rows.items():
                all_rows[k].extend(v)
            reps_rows += rows["replicates"]
            pair_rows += rows["pairs"]
            for ch in rp.truth["chromosomes"]:
                seed_rows.append(
                    {"model": model, "replicate": r, "chromosome": ch["name"],
                     "ancestry_seed": ch["seeds"]["ancestry"], "mutation_seed": ch["seeds"]["mutations"],
                     "het_sites": ch["het_sites"], "het_bins": ch["het_bins"], "trees": ch.get("trees"),
                     "longest_het_gap_bp": ch.get("longest_het_gap_bp"),
                     "longest_het_gap_start": ch.get("longest_het_gap_start"),
                     "longest_hetfree_bins": ch.get("longest_hetfree_bins")}
                )
        table1.append(table1_row(model, window, reps_rows, pair_rows))
    written = {}
    for key, rows in (("table1", table1), ("truth", truth_rows), ("seeds", seed_rows), *all_rows.items()):
        write_csv(lay.table(key), rows)
        written[key] = lay.table(key)
    return written


__all__ = ["Layout", "NAME", "TABLES", "jobs", "report"]
