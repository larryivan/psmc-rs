"""E1: simulated accuracy and equivalence (Fig 2, Table 1, Fig S1).

For every model and replicate (:class:`.config.E1`) the runner simulates one
diploid genome, fits it with C psmc and with psmc-rs 0.2 (identical psmcfa,
identical settings), and finally writes the tables. Layout under
``<runs>/E1/``::

    sims/<model>/repNN/genome.psmcfa, genome.vcf.gz(.tbi)   simulated genome (not committed)
    truth/<model>/repNN.json                               seeds, counts, checksums, true Ne(t)
    fits/<model>/repNN.c.psmc                              C psmc, every round
    fits/<model>/repNN.rs.json, repNN.rs.psmc              psmc-rs 0.2 (JSON and .psmc)
    fits/<model>/repNN.rs-default.json                     psmc-rs 0.2 with capped M-steps (only with E1.default_mstep_iters)
    provenance/<job>.json                                  one record per job
    tables/E1_*.csv                                        written by ``report E1``
    logs/                                                  runner and job logs (not committed)

Tables (``tables/``):

* ``E1_table1.csv``: Table 1, one row per model: RMSE of log10 Ne against
  the truth for each tool (mean and SD over replicates, and the median) on
  the model's informative window and on the fixed window, RMSE against the
  pattern-projected truth (below), the agreement of the tools interval by
  interval (below: mean, median and largest RMSE over replicates, and the
  largest difference of one interval inside the windows and anywhere), and
  the largest differences in log-likelihood and theta.
* ``E1_replicates.csv``: one row per model, replicate and tool: final theta,
  rho, t_max and N0, RMSE against the truth, log-likelihoods, and the run's
  wall time, peak RSS and load (recorded, not used for timing claims).
* ``E1_rs_vs_c.csv``: one row per model and replicate: the two tools compared
  interval by interval (below).
* ``E1_mstep.csv`` (header only unless ``E1.default_mstep_iters`` sets a
  cap; by default it does not, since psmc-rs 0.2's default M-step converges
  like the main fit's, see :class:`.config.Inference`): one row per model
  and replicate: psmc-rs with its M-steps capped at that many iterations
  against C psmc and against the main fit (``--mstep-iters 1000``), its RMSE
  against the truth and its wall time.
* ``E1_traces.csv``: EM traces (Fig S1): log-likelihood, theta, rho and
  t_max after each iteration, for both tools.
* ``E1_curves.csv``: the final Ne(t) step curves of every fit (Fig 2).
* ``E1_truth.csv``: the true Ne(t) of every model on a dense grid (Fig 2).
* ``E1_seeds.csv``: every simulation seed, with each chromosome's
  heterozygous sites, trees and longest stretch without a heterozygous
  site (a long recent-coalescence segment makes a replicate an outlier).

**Pattern-projected truth.** The first free lambda of ``4+25*2+4+6`` spans
intervals 0-3 (about 0-37 ky for the zigzag model, whose window starts at
7 ky), so part of the RMSE against the truth is the error of approximating
the truth by the pattern's steps, which no estimator with this pattern can
remove. Each fit is therefore also compared with the truth projected onto
its own free-parameter spans: the true Ne averaged over each span in
coalescence-rate terms (:func:`.metrics.projected_truth`), on the same
window. ``rmse_log10_ne_vs_projected_truth`` is then the estimation error,
and ``rmse_log10_projected_truth_vs_truth`` the pattern's own error.

**The tools compared.** Both fits have the same intervals (pattern
``4+25*2+4+6``), whose boundaries in years differ only through each fit's
theta and t_max. Interval k of psmc-rs is compared with interval k of C
psmc (:func:`.metrics.interval_diff`): ``*_per_interval`` columns hold
log10(Ne_rs,k / Ne_c,k) over C psmc's intervals that overlap the window,
the RMSE weighted by each interval's log-time width inside the window and
the largest absolute value; ``max_abs_log10_ne_rs_vs_c_any_interval`` is the
largest over every interval, also outside the window, where a first lambda
the data barely constrain can run away (in expansion replicate 5 psmc-rs's
stopped at its bound on every free lambda, 1e4, C psmc's reached about
1.1e6, so this column gives the gap the bound sets). Sampling both step
curves at fixed times would instead read whole steps wherever a time falls
between two nearly equal boundaries.

The helpers of the tables that E2 shares (both tools on one input) are here
too: :func:`read_fits`, :func:`compare_tools`, :func:`step_rows`,
:func:`trace_rows` and :func:`run_fields`.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, provenance, results, simulate, tools
from ..runner import Job
from . import Context

NAME = "E1"
TABLES = {
    "table1": "E1_table1.csv",
    "replicates": "E1_replicates.csv",
    "pairs": "E1_rs_vs_c.csv",
    "mstep": "E1_mstep.csv",
    "traces": "E1_traces.csv",
    "curves": "E1_curves.csv",
    "truth": "E1_truth.csv",
    "seeds": "E1_seeds.csv",
}


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

    def rs_default_json(self, model: str, r: int) -> Path:
        return self.root / "fits" / model / f"{rep(r)}.rs-default.json"

    def record(self, kind: str, model: str, r: int) -> Path:
        return self.root / "provenance" / kind / model / f"{rep(r)}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_id(kind: str, model: str, r: int) -> str:
    return f"{NAME}/{kind}/{model}/{rep(r)}"


def default_inference(cfg: config.E1) -> Optional[config.Inference]:
    """The inference settings of the fit with capped M-steps (``E1.default_mstep_iters``), or None when there is none."""
    m = cfg.default_mstep_iters
    if m is None or m == cfg.inference.mstep_iters:
        return None
    return replace(cfg.inference, mstep_iters=m)


def jobs(ctx: Context) -> List[Job]:
    cfg, costs = ctx.profile.e1, ctx.profile.costs
    lay = Layout(ctx.exp_dir(NAME))
    genome_mb, chrom_mb = cfg.genome.length / 1e6, cfg.genome.chromosome_length / 1e6
    inference, scaling = asdict(cfg.inference), asdict(cfg.scaling)
    default = default_inference(cfg)
    sim_tools = {
        "pipeline": provenance.python_identity(
            ["simulate.py", "demography.py"], ["msprime", "tskit", "numpy", "stdpopsim"]
        ),
        "htslib": tools.htslib_versions(),
    }
    c_tool, rs_tool = ctx.c_psmc(), ctx.rs02()
    rs_ident = {rs_tool.name: rs_tool.identity}
    out: List[Job] = []
    report_inputs: List[Path] = []
    for model in cfg.models:
        for r in range(1, cfg.replicates + 1):
            psmcfa, vcf, truth = lay.psmcfa(model, r), lay.vcf(model, r), lay.truth(model, r)
            out.append(Job(
                id=job_id("sim", model, r),
                command=ctx.pipeline_cmd(
                    "simulate", "--profile", ctx.profile.name, "--experiment", NAME, "--model", model,
                    "--replicate", str(r), "--out-dir", str(lay.sim_dir(model, r)), "--truth", str(truth),
                ),
                outputs=[psmcfa, vcf, Path(f"{vcf}.tbi"), truth],
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
            ))
            out.append(Job(
                id=job_id("fit-c", model, r),
                command=tools.c_psmc_fit(c_tool, cfg.inference, psmcfa, lay.c_psmc(model, r)),
                inputs=[psmcfa],
                outputs=[lay.c_psmc(model, r)],
                mem_gb=round(costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * chrom_mb, 2),
                est_seconds=costs.c_psmc_s_per_mb_iter * genome_mb * cfg.inference.n_iter,
                settings={"inference": inference},
                tools={c_tool.name: c_tool.identity},
                description=f"C psmc on {model} replicate {r}",
            ))
            out.append(Job(
                id=job_id("fit-rs", model, r),
                command=tools.rs02_fit(
                    rs_tool, cfg.inference, cfg.scaling, psmcfa, lay.rs_json(model, r),
                    threads=cfg.rs_threads, psmc_out=lay.rs_psmc(model, r),
                ),
                inputs=[psmcfa],
                outputs=[lay.rs_json(model, r), lay.rs_psmc(model, r)],
                cores=cfg.rs_threads,
                mem_gb=costs.rs02_gb,
                est_seconds=costs.rs02_seconds(genome_mb, cfg.inference, cfg.rs_threads),
                settings={"inference": inference, "scaling": scaling, "threads": cfg.rs_threads},
                tools=rs_ident,
                description=f"psmc-rs 0.2 on {model} replicate {r}",
            ))
            report_inputs += [
                truth, lay.c_psmc(model, r), lay.rs_json(model, r), lay.rs_psmc(model, r),
                lay.record("fit-c", model, r), lay.record("fit-rs", model, r),
            ]
            if default is not None:
                out.append(Job(
                    id=job_id("fit-rs-default", model, r),
                    command=tools.rs02_fit(rs_tool, default, cfg.scaling, psmcfa, lay.rs_default_json(model, r),
                                           threads=cfg.rs_threads),
                    inputs=[psmcfa],
                    outputs=[lay.rs_default_json(model, r)],
                    cores=cfg.rs_threads,
                    mem_gb=costs.rs02_gb,
                    est_seconds=costs.rs02_seconds(genome_mb, default, cfg.rs_threads),
                    settings={"inference": asdict(default), "scaling": scaling, "threads": cfg.rs_threads},
                    tools=rs_ident,
                    description=f"psmc-rs 0.2, M-steps capped at {default.mstep_iters}, on {model} replicate {r}",
                ))
                report_inputs += [lay.rs_default_json(model, r), lay.record("fit-rs-default", model, r)]
    out.append(Job(
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
    ))
    return out


# ---------------------------------------------------------------------------
# Tables (the helpers from write_csv to run_fields serve E2 and the others too)
# ---------------------------------------------------------------------------


def _num(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return "nan" if math.isnan(x) else repr(x)
    return str(x)


def write_csv(path: Path, rows: Sequence[Dict], fields: Optional[Sequence[str]] = None) -> None:
    """Write ``rows`` atomically, with the columns ``fields`` (default: the first row's keys); None is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or (rows[0].keys() if rows else []))
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _num(row.get(k)) for k in fields})
    tmp.replace(path)


def read_fits(c_path: Path, rs_psmc: Path, rs_json: Path, n_iter: int) -> Tuple[results.PsmcFile, results.PsmcFile, Dict]:
    """C psmc's ``.psmc``, psmc-rs's ``.psmc`` and its JSON, checked to be fits of ``n_iter`` iterations."""
    c, rs, rs_params = results.parse_psmc(c_path), results.parse_psmc(rs_psmc), results.read_rs_json(rs_json)
    if c.last().round != n_iter:
        raise ValueError(f"{c_path}: last complete round {c.last().round}, expected {n_iter}")
    if int(rs_params.get("n_iter", -1)) != n_iter or rs.last().round != n_iter:
        raise ValueError(f"{rs_json}: not a fit of {n_iter} iterations")
    return c, rs, rs_params


def loglikes(c: results.PsmcFile, rs_params: Dict) -> Dict[str, Dict[int, float]]:
    """``{tool: {iterations: log-likelihood}}``."""
    return {"c": c.loglikes(), "rs": dict(enumerate(results.rs_loglikes(rs_params)))}


def loglike_diffs(ll: Dict[str, Dict[int, float]], n_iter: int) -> Dict:
    """psmc-rs's log-likelihoods against C psmc's: after ``n_iter - 1`` iterations, the largest relative, the first."""
    common = sorted(i for i in set(ll["c"]) & set(ll["rs"]) if ll["c"][i])
    rel = {i: abs(ll["rs"][i] - ll["c"][i]) / abs(ll["c"][i]) for i in common}
    return {
        "loglike_common_diff": ll["rs"][n_iter - 1] - ll["c"][n_iter - 1] if (n_iter - 1) in common else None,
        "loglike_max_rel_diff": max(rel.values()) if rel else None,
        "loglike_iteration0_rel_diff": rel.get(0),
    }


def compare_tools(fits: Dict[str, results.Fit], curves: Dict[str, results.Curve], window: Tuple[float, float],
                  fixed: Tuple[float, float]) -> Dict:
    """psmc-rs's fit against C psmc's: interval by interval on ``window`` and ``fixed`` (years), and the parameters."""
    fc, fr = fits["c"], fits["rs"]
    diff = metrics.interval_diff(curves["rs"], curves["c"], window)
    return {
        "window_lo_years": window[0],
        "window_hi_years": window[1],
        "intervals_in_window": len(diff.interval),
        "rmse_log10_ne_rs_vs_c_per_interval": diff.rmse,
        "rmse_log10_ne_rs_vs_c_per_interval_fixed": metrics.interval_rmse(curves["rs"], curves["c"], fixed),
        "max_abs_log10_ne_rs_vs_c_per_interval": diff.max_abs,
        "max_abs_log10_ne_rs_vs_c_any_interval": diff.max_abs_any,
        "theta_rel_diff": fr.theta / fc.theta - 1.0,
        "rho_rel_diff": fr.rho / fc.rho - 1.0,
        "t_max_rel_diff": fr.t_max / fc.t_max - 1.0,
    }


def step_rows(fit: results.Fit, curve: results.Curve, **ids) -> List[Dict]:
    """A fit's step curve, one row per interval (``ids`` first): its start in 2N0 units and in years, lambda and Ne."""
    return [{**ids, "interval": k, "t_2n0": t, "years": y, "lambda": lam, "ne": ne}
            for k, (t, y, lam, ne) in enumerate(zip(fit.boundaries(), curve.years, fit.lambdas, curve.ne))]


def trace_rows(psmc: results.PsmcFile, ll: Dict[int, float], n_iter: int, **ids) -> List[Dict]:
    """A fit's log-likelihood, theta, rho and t_max after 0 to ``n_iter`` EM iterations (empty where it has none)."""
    by_round = {r.round: r for r in psmc.complete_rounds()}
    rows = []
    for i in range(n_iter + 1):
        rd = by_round.get(i)
        rows.append({**ids, "iteration": i, "loglike": ll.get(i), "theta": rd.theta if rd else None,
                     "rho": rd.rho if rd else None, "t_max": rd.t_max if rd else None})
    return rows


def run_record(path: Path) -> Dict:
    """How a job ran (its provenance record's ``run``); empty when there is no record."""
    try:
        return json.loads(Path(path).read_text()).get("run", {})
    except (OSError, ValueError):
        return {}


def run_fields(run: Dict) -> Dict:
    """The columns of how a fit ran (recorded, not used for timing claims)."""
    return {
        "wall_seconds": run.get("wall_seconds"),
        "peak_rss_mb": run.get("peak_rss_mb"),
        "user_seconds": run.get("user_seconds"),
        "loadavg_before_1m": (run.get("loadavg_before") or [None])[0],
        "run_start": run.get("start"),
    }


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
    default_params: Optional[Dict] = None  # psmc-rs with capped M-steps
    default_curve: Optional[results.Curve] = None


def load_replicate(lay: Layout, cfg: config.E1, model: str, r: int, with_default: bool = True) -> Replicate:
    """The fits of one genome, read back and checked; ``with_default``: also psmc-rs's capped M-step fit."""
    truth = json.loads(lay.truth(model, r).read_text())
    n_iter = cfg.inference.n_iter
    c, rs, rs_params = read_fits(lay.c_psmc(model, r), lay.rs_psmc(model, r), lay.rs_json(model, r), n_iter)
    het_bins = sum(ch["het_bins"] for ch in truth["chromosomes"])
    bins = sum(ch["bins"] for ch in truth["chromosomes"])
    inp = rs_params.get("input", {})
    if not (c.header.get("sum_n") == het_bins == inp.get("het_bins") and c.header.get("sum_L") == bins == inp.get("bins")):
        raise ValueError(
            f"{model} {rep(r)}: the fits did not read the simulated genome "
            f"(truth {bins} bins/{het_bins} het, C {c.header.get('sum_L')}/{c.header.get('sum_n')}, "
            f"psmc-rs {inp.get('bins')}/{inp.get('het_bins')})"
        )
    sc, bin_size = cfg.scaling, cfg.inference.bin_size
    fits = {"c": c.last().fit(), "rs": results.fit_from_rs(rs_params)}
    curves = {k: f.curve(sc.mu, sc.gen_years, bin_size) for k, f in fits.items()}
    runs = {tool: run_record(lay.record(kind, model, r)) for tool, kind in (("c", "fit-c"), ("rs", "fit-rs"))}
    default_params = default_curve = None
    if with_default and default_inference(cfg) is not None:
        default_params = results.read_rs_json(lay.rs_default_json(model, r))
        if int(default_params.get("n_iter", -1)) != n_iter:
            raise ValueError(f"{lay.rs_default_json(model, r)}: not a fit of {n_iter} iterations")
        default_curve = results.fit_from_rs(default_params).curve(sc.mu, sc.gen_years, bin_size)
        runs["rs-default"] = run_record(lay.record("fit-rs-default", model, r))
    return Replicate(
        model, r, truth, demography.epochs_from_record(truth["truth"]), c, rs_params, rs, fits, curves, runs,
        default_params, default_curve,
    )


def replicate_rows(rp: Replicate, cfg: config.E1, window, fixed) -> Dict[str, List[Dict]]:
    """One replicate's rows of E1_replicates, _rs_vs_c, _traces, _curves and _mstep."""
    g, n = cfg.scaling.gen_years, cfg.inference.n_iter
    sc, bin_size = cfg.scaling, cfg.inference.bin_size
    ll = loglikes(rp.c, rp.rs_params)
    ids = {"model": rp.model, "replicate": rp.replicate}
    rows: Dict[str, List[Dict]] = {"replicates": [], "pairs": [], "traces": [], "curves": [], "mstep": []}
    for tool, psmc in (("c", rp.c), ("rs", rp.rs)):
        fit, curve = rp.fits[tool], rp.curves[tool]
        projected = metrics.projected_truth(fit, rp.epochs, sc.mu, g, bin_size)
        rows["replicates"].append({
            **ids,
            "tool": tool,
            "theta": fit.theta,
            "rho": fit.rho,
            "t_max": fit.t_max,
            "n0": fit.n0(sc.mu, bin_size),
            "rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(curve, rp.epochs, window, g),
            "rmse_log10_ne_vs_truth_fixed": metrics.rmse_vs_truth(curve, rp.epochs, fixed, g),
            "rmse_log10_ne_vs_projected_truth": metrics.rmse_vs_projected_truth(curve, projected, window),
            "rmse_log10_projected_truth_vs_truth": metrics.rmse_vs_truth(projected, rp.epochs, window, g),
            "loglike_common": ll[tool].get(n - 1),
            "loglike_final": ll[tool].get(n),
            **run_fields(rp.runs[tool]),
        })
        rows["curves"] += step_rows(fit, curve, **ids, tool=tool)
        rows["traces"] += trace_rows(psmc, ll[tool], n, **ids, tool=tool)
    w = (float(window[0]), float(window[-1]))
    rows["pairs"].append({**ids, **compare_tools(rp.fits, rp.curves, w, (fixed[0], fixed[-1])), **loglike_diffs(ll, n)})
    if rp.default_params is not None:
        rows["mstep"].append(mstep_row(rp, cfg, window))
    return rows


# The columns of E1_mstep.csv, written also when there is no capped fit
# (the default), so that the table keeps its header.
MSTEP_FIELDS = (
    "model", "replicate", "mstep_iters_default", "mstep_iters", "rmse_log10_ne_default_vs_c_per_interval",
    "max_abs_log10_ne_default_vs_c_per_interval", "rmse_log10_ne_rs_vs_c_per_interval",
    "rmse_log10_ne_default_vs_rs_per_interval", "rmse_log10_ne_default_vs_truth", "rmse_log10_ne_rs_vs_truth",
    "theta_default_rel_c", "loglike_final_default", "loglike_final_rs", "wall_seconds_default", "wall_seconds_rs",
)


def mstep_row(rp: Replicate, cfg: config.E1, window) -> Dict:
    """psmc-rs with capped M-steps against C psmc and against the main psmc-rs fit (interval by interval)."""
    g = cfg.scaling.gen_years
    d, c, rs = rp.default_curve, rp.curves["c"], rp.curves["rs"]
    w = (window[0], window[-1])
    d_c = metrics.interval_diff(d, c, w)
    ll_d, ll_rs = results.rs_loglikes(rp.default_params), results.rs_loglikes(rp.rs_params)
    return {
        "model": rp.model,
        "replicate": rp.replicate,
        "mstep_iters_default": cfg.default_mstep_iters,
        "mstep_iters": cfg.inference.mstep_iters,
        "rmse_log10_ne_default_vs_c_per_interval": d_c.rmse,
        "max_abs_log10_ne_default_vs_c_per_interval": d_c.max_abs,
        "rmse_log10_ne_rs_vs_c_per_interval": metrics.interval_rmse(rs, c, w),
        "rmse_log10_ne_default_vs_rs_per_interval": metrics.interval_rmse(d, rs, w),
        "rmse_log10_ne_default_vs_truth": metrics.rmse_vs_truth(d, rp.epochs, window, g),
        "rmse_log10_ne_rs_vs_truth": metrics.rmse_vs_truth(rs, rp.epochs, window, g),
        "theta_default_rel_c": results.fit_from_rs(rp.default_params).theta / rp.fits["c"].theta - 1.0,
        "loglike_final_default": ll_d[-1] if ll_d else None,
        "loglike_final_rs": ll_rs[-1] if ll_rs else None,
        "wall_seconds_default": rp.runs["rs-default"].get("wall_seconds"),
        "wall_seconds_rs": rp.runs["rs"].get("wall_seconds"),
    }


def seed_rows(rp: Replicate) -> List[Dict]:
    return [
        {"model": rp.model, "replicate": rp.replicate, "chromosome": ch["name"],
         "ancestry_seed": ch["seeds"]["ancestry"], "mutation_seed": ch["seeds"]["mutations"],
         "het_sites": ch["het_sites"], "het_bins": ch["het_bins"], "trees": ch.get("trees"),
         "longest_het_gap_bp": ch.get("longest_het_gap_bp"), "longest_het_gap_start": ch.get("longest_het_gap_start"),
         "longest_hetfree_bins": ch.get("longest_hetfree_bins")}
        for ch in rp.truth["chromosomes"]
    ]


def _mean_sd(xs: Sequence[Optional[float]]):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def _median(xs: Sequence[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def table1_row(model: str, window, reps: List[Dict], pairs: List[Dict], msteps: Sequence[Dict]) -> Dict:
    def col(tool: str, key: str) -> List[float]:
        return [r[key] for r in reps if r["tool"] == tool]

    def mean_sd(prefix: str, key: str) -> Dict:
        (c_mean, c_sd), (rs_mean, rs_sd) = _mean_sd(col("c", key)), _mean_sd(col("rs", key))
        return {f"{prefix}_c_mean": c_mean, f"{prefix}_c_sd": c_sd, f"{prefix}_rs_mean": rs_mean, f"{prefix}_rs_sd": rs_sd}

    d = [p["rmse_log10_ne_rs_vs_c_per_interval"] for p in pairs]
    capped = [m["rmse_log10_ne_default_vs_c_per_interval"] for m in msteps]
    return {
        "model": model,
        "replicates": len(pairs),
        "window_lo_years": float(window[0]),
        "window_hi_years": float(window[-1]),
        **mean_sd("rmse_truth", "rmse_log10_ne_vs_truth"),
        "rmse_truth_c_median": _median(col("c", "rmse_log10_ne_vs_truth")),
        "rmse_truth_rs_median": _median(col("rs", "rmse_log10_ne_vs_truth")),
        **mean_sd("rmse_truth_fixed", "rmse_log10_ne_vs_truth_fixed"),
        **mean_sd("rmse_projected", "rmse_log10_ne_vs_projected_truth"),
        "rmse_pattern_vs_truth_mean": _mean_sd(col("c", "rmse_log10_projected_truth_vs_truth"))[0],
        "rmse_rs_vs_c_per_interval_mean": statistics.fmean(d) if d else None,
        "rmse_rs_vs_c_per_interval_median": _median(d),
        "rmse_rs_vs_c_per_interval_max": max(d) if d else None,
        "max_abs_rs_vs_c_per_interval": max((p["max_abs_log10_ne_rs_vs_c_per_interval"] for p in pairs), default=None),
        "max_abs_rs_vs_c_any_interval": max((p["max_abs_log10_ne_rs_vs_c_any_interval"] for p in pairs), default=None),
        "loglike_max_rel_diff": max((p["loglike_max_rel_diff"] for p in pairs if p["loglike_max_rel_diff"] is not None), default=None),
        "theta_max_abs_rel_diff": max((abs(p["theta_rel_diff"]) for p in pairs), default=None),
        "rmse_rs_default_vs_c_per_interval_mean": _mean_sd(capped)[0],
        "rmse_rs_default_vs_c_per_interval_max": max(capped, default=None),
        "rmse_truth_rs_default_mean": _mean_sd([m["rmse_log10_ne_default_vs_truth"] for m in msteps])[0],
    }


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E1 table from the fits; returns the paths written."""
    cfg = ctx.profile.e1
    lay = Layout(ctx.exp_dir(NAME))
    fixed = metrics.fixed_grid(cfg.grid)
    dense = metrics.log_grid(*config.TRUTH_TABLE_YEARS)
    g = cfg.scaling.gen_years
    tables: Dict[str, List[Dict]] = {k: [] for k in ("table1", "truth", "seeds", "replicates", "pairs", "traces",
                                                     "curves", "mstep")}
    for model in cfg.models:
        reps = [load_replicate(lay, cfg, model, r) for r in range(1, cfg.replicates + 1)]
        epochs = reps[0].epochs
        window = metrics.model_grid(epochs, cfg.grid, g)
        tables["truth"] += [{"model": model, "years": float(t), "ne": float(ne)}
                            for t, ne in zip(dense, metrics.truth_on_grid(epochs, dense, g))]
        rows: Dict[str, List[Dict]] = {k: [] for k in ("replicates", "pairs", "traces", "curves", "mstep")}
        for rp in reps:
            for k, v in replicate_rows(rp, cfg, window, fixed).items():
                rows[k] += v
            tables["seeds"] += seed_rows(rp)
        tables["table1"].append(table1_row(model, window, rows["replicates"], rows["pairs"], rows["mstep"]))
        for k, v in rows.items():
            tables[k] += v
    for key, rows in tables.items():
        write_csv(lay.table(key), rows, MSTEP_FIELDS if key == "mstep" else None)
    return {key: lay.table(key) for key in tables}


__all__ = [
    "Layout", "NAME", "TABLES", "compare_tools", "jobs", "load_replicate", "loglike_diffs", "loglikes", "read_fits",
    "rep", "report", "run_fields", "run_record", "step_rows", "trace_rows", "write_csv",
]
