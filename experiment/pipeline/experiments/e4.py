"""E4: input preparation (Fig 5).

How the path from a VCF and a callable mask to PSMC's input changes the
result. For every replicate (:class:`.config.E4`: bottleneck model, 500 Mb
as 10 chromosomes of 50 Mb) the runner simulates one diploid genome, draws
callable masks (missing fraction 0.1 or 0.3, clustered in runs of 20 kb or
dispersed in runs of 20 bp, plus the no-missing control), removes the
records at uncallable sites from the VCF, and takes the masked VCF to PSMC
by four routes (see section 8 of config.py and :mod:`.inputprep`):

* ``exact``: psmc-rs on the VCF with ``--vcf-mask``;
* ``fq2psmcfa``: consensus FASTA -> fq2psmcfa -> psmc-rs on the psmcfa. Its
  bins must equal the exact route's: the ``bins`` job fails otherwise;
* ``multihetsep``: ``psmc-rs convert --multihetsep`` read back by psmc-rs
  (cross-checked against generate_multihetsep.py on replicate 1);
* ``nomask``: psmc-rs on the VCF without the mask.

Every route is fitted with psmc-rs 0.2 (the human recipe of the psmc README)
and checked with ``psmc-rs qc``. Layout under ``<runs>/E4/``::

    sims/repNN/genome.psmcfa, genome.vcf.gz(.tbi)        simulated genome (not committed)
    sims/repNN/<mask>/callable.bed.gz                    callable sites
    sims/repNN/<mask>/variants.vcf.gz(.tbi)              the VCF of callable records
    sims/repNN/<mask>/fq2psmcfa.psmcfa                   route fq2psmcfa's input
    sims/repNN/<mask>/input.multihetsep                  route multihetsep's input
    sims/repNN/<mask>/bins/{exact,multihetsep,nomask}.psmcfa  each route's bins as psmc-rs reads them
    truth/repNN.json                                     seeds, counts, checksums, true Ne(t)
    masks/<mask>/repNN.json                              the mask drawn: missing fraction, runs, seeds
    consensus/<mask>/repNN.json                          the consensus FASTA's counts and seeds
    bins/<mask>/repNN.json                               every route's bins against the exact route's
    crosscheck/<mask>/rep01.json                         psmc-rs vs generate_multihetsep.py
    fits/<mask>/<route>/repNN.json                       psmc-rs 0.2
    qc/<mask>/<route>/repNN.json                         psmc-rs qc
    provenance/, tables/E4_*.csv, figures/               (logs/ and sims/ are not committed)

Tables (``tables/``):

* ``E4_summary.csv``: one row per mask and route (Fig 5): replicates, the
  realised missing fraction, the share of bins that differ from the exact
  route, theta relative to the exact route and to the complete data, RMSE of
  log10 Ne against the truth, RMSE against the exact route, how many fits
  equal the exact route's, and the QC codes that fired.
* ``E4_routes.csv``: one row per mask, route and replicate: the fit, its
  input as read, the metrics and the QC codes.
* ``E4_bias.csv``: log10(fitted / true Ne) at every time of the model's
  informative window, mean, SD, min and max over replicates (Fig 5).
* ``E4_bins.csv``: one row per mask, replicate and route: how its bins differ
  from the exact route's (T/K/N against T/K/N, and bins only one route has).
* ``E4_masks.csv``: one row per mask and replicate: target and realised
  missing fraction, uncallable runs, heterozygous sites removed.
* ``E4_qc.csv``: one row per QC finding (mask, route, replicate, code, severity).
* ``E4_crosscheck.csv``: one row per chromosome checked against
  generate_multihetsep.py.
* ``E4_curves.csv``, ``E4_truth.csv``: every fitted Ne(t) step curve and the
  true Ne(t) on a dense grid.
* ``E4_seeds.csv``: every seed (simulation, masks, consensus bases).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import TRUTH_GRID_YEARS, rep, write_csv

NAME = "E4"
ROUTES = ("exact", "fq2psmcfa", "multihetsep", "nomask")
BIN_ROUTES = ("fq2psmcfa", "multihetsep", "nomask")  # compared with the exact route's bins
TABLES = {
    "summary": "E4_summary.csv",
    "routes": "E4_routes.csv",
    "bias": "E4_bias.csv",
    "bins": "E4_bins.csv",
    "masks": "E4_masks.csv",
    "qc": "E4_qc.csv",
    "crosscheck": "E4_crosscheck.csv",
    "curves": "E4_curves.csv",
    "truth": "E4_truth.csv",
    "seeds": "E4_seeds.csv",
}
FIGURES = ("E4_fig5.pdf", "E4_fig5.png")


def mask_names(cfg: config.E4) -> Tuple[str, ...]:
    """The masks of ``cfg``: the no-missing control first, then the simulated masks."""
    names = tuple(m.name for m in cfg.masks)
    if config.E4_CONTROL_MASK in names:
        raise ValueError(f"mask name {config.E4_CONTROL_MASK!r} is reserved for the control")
    return ((config.E4_CONTROL_MASK,) if cfg.no_missing_control else ()) + names


def mask_spec(cfg: config.E4, name: str) -> Optional[config.MaskSpec]:
    """The spec of mask ``name`` (None for the control)."""
    if name == config.E4_CONTROL_MASK:
        return None
    for m in cfg.masks:
        if m.name == name:
            return m
    raise KeyError(f"unknown mask {name!r}")


@dataclass(frozen=True)
class Layout:
    """Where E4's files are."""

    root: Path

    def sim_dir(self, r: int) -> Path:
        return self.root / "sims" / rep(r)

    def psmcfa(self, r: int) -> Path:
        return self.sim_dir(r) / simulate.PSMCFA_NAME

    def vcf(self, r: int) -> Path:
        return self.sim_dir(r) / simulate.VCF_NAME

    def truth(self, r: int) -> Path:
        return self.root / "truth" / f"{rep(r)}.json"

    def mask_dir(self, mask: str, r: int) -> Path:
        return self.sim_dir(r) / mask

    def bed(self, mask: str, r: int) -> Path:
        return self.mask_dir(mask, r) / "callable.bed.gz"

    def variants(self, mask: str, r: int) -> Path:
        return self.mask_dir(mask, r) / "variants.vcf.gz"

    def fq2psmcfa(self, mask: str, r: int) -> Path:
        return self.mask_dir(mask, r) / "fq2psmcfa.psmcfa"

    def multihetsep(self, mask: str, r: int) -> Path:
        return self.mask_dir(mask, r) / "input.multihetsep"

    def route_bins(self, mask: str, route: str, r: int) -> Path:
        """The bins of a route as psmc-rs reads them (fq2psmcfa: its own output)."""
        if route == "fq2psmcfa":
            return self.fq2psmcfa(mask, r)
        return self.mask_dir(mask, r) / "bins" / f"{route}.psmcfa"

    def mask_record(self, mask: str, r: int) -> Path:
        return self.root / "masks" / mask / f"{rep(r)}.json"

    def consensus_record(self, mask: str, r: int) -> Path:
        return self.root / "consensus" / mask / f"{rep(r)}.json"

    def bins_record(self, mask: str, r: int) -> Path:
        return self.root / "bins" / mask / f"{rep(r)}.json"

    def crosscheck(self, mask: str, r: int) -> Path:
        return self.root / "crosscheck" / mask / f"{rep(r)}.json"

    def fit(self, mask: str, route: str, r: int) -> Path:
        return self.root / "fits" / mask / route / f"{rep(r)}.json"

    def qc(self, mask: str, route: str, r: int) -> Path:
        return self.root / "qc" / mask / route / f"{rep(r)}.json"

    def record(self, job_name: str) -> Path:
        return self.root / "provenance" / f"{job_name}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]

    def figure(self, name: str) -> Path:
        return self.root / "figures" / name


def job_id(*parts: str) -> str:
    return "/".join((NAME, *parts))


def route_input(lay: Layout, mask: str, route: str, r: int) -> Tuple[Path, List[str]]:
    """The file a route gives psmc-rs and the input options it reads it with."""
    if route == "exact":
        return lay.variants(mask, r), ["--input-format", "vcf", "--vcf-mode", "mask", "--vcf-mask", str(lay.bed(mask, r))]
    if route == "fq2psmcfa":
        return lay.fq2psmcfa(mask, r), ["--input-format", "psmcfa"]
    if route == "multihetsep":
        return lay.multihetsep(mask, r), ["--input-format", "mhs"]
    if route == "nomask":
        return lay.variants(mask, r), ["--input-format", "vcf", "--vcf-mode", "mask"]
    raise KeyError(f"unknown route {route!r}")


def route_inputs(lay: Layout, mask: str, route: str, r: int) -> List[Path]:
    """Every file psmc-rs reads for a route (the exact route's mask too)."""
    inp, _ = route_input(lay, mask, route, r)
    return [inp, lay.bed(mask, r)] if route == "exact" else [inp]


def step_cmd(*args: str) -> List[str]:
    """``python -m experiment.pipeline.inputprep ARGS``."""
    return [paths.python(), "-m", "experiment.pipeline.inputprep", *args]


def utility(ctx: Context, name: str) -> tools.Tool:
    """A utility of the tools directory, or the stand-in a test put in the context."""
    if name not in ctx._tools:
        ctx._tools[name] = tools.utility(name)
    return ctx._tools[name]


@dataclass(frozen=True)
class Shared:
    """What the jobs of every replicate and mask share: tools, identities, sizes."""

    rs_tool: tools.Tool
    fq2psmcfa: tools.Tool
    generate_multihetsep: tools.Tool
    steps: Dict  # identity of the Python steps (inputprep.py)
    htslib: Dict
    bin_size: str
    mb: float  # genome size, for the cost estimates

    @property
    def rs(self) -> Dict:
        return {self.rs_tool.name: self.rs_tool.identity}


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e4
    costs = ctx.profile.costs
    lay = Layout(ctx.exp_dir(NAME))
    model = cfg.model
    htslib = tools.htslib_versions()
    sh = Shared(
        rs_tool=ctx.rs02(),
        fq2psmcfa=utility(ctx, "fq2psmcfa"),
        generate_multihetsep=utility(ctx, "generate_multihetsep.py"),
        steps={"pipeline": provenance.python_identity(["inputprep.py"], ["numpy"])},
        htslib=htslib,
        bin_size=str(cfg.inference.bin_size),
        mb=cfg.genome.length / 1e6,
    )
    sim_tools = {
        "pipeline": provenance.python_identity(["simulate.py", "demography.py"], ["msprime", "tskit", "numpy", "stdpopsim"]),
        "htslib": htslib,
    }
    out: List[Job] = []
    for r in range(1, cfg.replicates + 1):
        out.append(
            Job(
                id=job_id("sim", rep(r)),
                command=ctx.pipeline_cmd(
                    "simulate", "--profile", ctx.profile.name, "--experiment", NAME, "--model", model,
                    "--replicate", str(r), "--out-dir", str(lay.sim_dir(r)), "--truth", str(lay.truth(r)),
                ),
                outputs=[lay.psmcfa(r), lay.vcf(r), Path(f"{lay.vcf(r)}.tbi"), lay.truth(r)],
                mem_gb=costs.simulate_gb,
                est_seconds=costs.simulate_s_base + costs.simulate_s_per_mb * sh.mb,
                settings={"model": asdict(config.MODELS[model]), "genome": asdict(cfg.genome),
                          "simulation": asdict(cfg.simulation)},
                tools=sim_tools,
                seeds={c: simulate.seeds_for(NAME, model, r, c) for c in cfg.genome.names()},
                description=f"simulate {model} replicate {r}",
            )
        )
        for mask in mask_names(cfg):
            out.extend(mask_jobs(ctx, lay, sh, mask, r))
    out.extend(_report_jobs(ctx, cfg, lay))
    return out


def mask_jobs(ctx: Context, lay: Layout, sh: Shared, mask: str, r: int) -> List[Job]:
    """The jobs of one replicate and mask: mask, routes' inputs and bins, fits and QC."""
    cfg = ctx.profile.e4
    ec, costs = config.E4_COSTS, ctx.profile.costs
    model, mb, bin_size, rs = cfg.model, sh.mb, sh.bin_size, sh.rs
    rs_tool, fq2psmcfa, gms, step_tools = sh.rs_tool, sh.fq2psmcfa, sh.generate_multihetsep, sh.steps
    spec = mask_spec(cfg, mask)
    spec_settings = asdict(spec) if spec else {"name": mask, "missing_fraction": 0.0}
    seed_args = ["--experiment", NAME, "--model", model, "--replicate", str(r)]
    variants, bed, mhs = lay.variants(mask, r), lay.bed(mask, r), lay.multihetsep(mask, r)
    tbi = Path(f"{variants}.tbi")
    jobs: List[Job] = []
    spec_args = [] if spec is None else [
        "--missing-fraction", repr(spec.missing_fraction), "--structure", spec.structure,
        "--mean-run-bp", repr(spec.mean_run_bp),
    ]
    jobs.append(
        Job(
            id=job_id("mask", mask, rep(r)),
            command=step_cmd("mask", *seed_args, "--vcf", str(lay.vcf(r)), "--mask-name", mask, *spec_args,
                             "--bed-out", str(bed), "--vcf-out", str(variants), "--record", str(lay.mask_record(mask, r))),
            inputs=[lay.vcf(r)],
            outputs=[bed, variants, tbi, lay.mask_record(mask, r)],
            mem_gb=ec.mask_gb,
            est_seconds=5 + ec.mask_s_per_mb * mb,
            settings={"mask": spec_settings, "genome": asdict(cfg.genome)},
            tools={**step_tools, "htslib": sh.htslib},
            seeds={c: (config.seed(NAME, model, r, f"mask/{mask}/{c}") if spec else None) for c in cfg.genome.names()},
            description=f"mask {mask} on replicate {r}",
        )
    )
    jobs.append(
        Job(
            id=job_id("consensus", mask, rep(r)),
            command=step_cmd("consensus", *seed_args, "--vcf", str(variants), "--mask", str(bed),
                             "--fq2psmcfa", str(fq2psmcfa.path), "--bin-size", bin_size,
                             "--out", str(lay.fq2psmcfa(mask, r)), "--record", str(lay.consensus_record(mask, r))),
            inputs=[variants, bed],
            outputs=[lay.fq2psmcfa(mask, r), lay.consensus_record(mask, r)],
            mem_gb=ec.consensus_gb,
            est_seconds=5 + ec.consensus_s_per_mb * mb,
            settings={"bin_size": cfg.inference.bin_size},
            tools={**step_tools, fq2psmcfa.name: fq2psmcfa.identity},
            seeds={c: config.seed(NAME, model, r, f"reference/{c}") for c in cfg.genome.names()},
            description=f"consensus FASTA -> fq2psmcfa, {mask} replicate {r}",
        )
    )
    # The exact route's bins and the multihetsep route's input, in one pass.
    exact_in, exact_args = route_input(lay, mask, "exact", r)
    jobs.append(
        Job(
            id=job_id("convert", mask, rep(r)),
            command=[str(rs_tool.path), "convert", str(exact_in), *exact_args, "--bin-size", bin_size,
                     "--psmcfa", str(lay.route_bins(mask, "exact", r)), "--multihetsep", str(mhs), "--quiet"],
            inputs=route_inputs(lay, mask, "exact", r),
            outputs=[lay.route_bins(mask, "exact", r), mhs],
            mem_gb=ec.convert_gb,
            est_seconds=2 + ec.convert_s_per_mb * mb,
            tools=rs,
            description=f"psmc-rs convert (exact bins, multihetsep), {mask} replicate {r}",
        )
    )
    for route in ("multihetsep", "nomask"):
        inp, args = route_input(lay, mask, route, r)
        jobs.append(
            Job(
                id=job_id("convert", mask, route, rep(r)),
                command=[str(rs_tool.path), "convert", str(inp), *args, "--bin-size", bin_size,
                         "--psmcfa", str(lay.route_bins(mask, route, r)), "--quiet"],
                inputs=route_inputs(lay, mask, route, r),
                outputs=[lay.route_bins(mask, route, r)],
                mem_gb=ec.convert_gb,
                est_seconds=2 + ec.convert_s_per_mb * mb,
                tools=rs,
                description=f"psmc-rs convert ({route} bins), {mask} replicate {r}",
            )
        )
    jobs.append(
        Job(
            id=job_id("bins", mask, rep(r)),
            command=step_cmd("bins", "--exact", str(lay.route_bins(mask, "exact", r)),
                             *[a for route in BIN_ROUTES for a in ("--route", f"{route}={lay.route_bins(mask, route, r)}")],
                             "--assert-identical", "fq2psmcfa", "--out", str(lay.bins_record(mask, r))),
            inputs=[lay.route_bins(mask, route, r) for route in ("exact", *BIN_ROUTES)],
            outputs=[lay.bins_record(mask, r)],
            mem_gb=ec.bins_gb,
            est_seconds=2 + ec.bins_s_per_mb * mb,
            tools=step_tools,
            description=f"bins of every route against the exact route, {mask} replicate {r}",
        )
    )
    if r <= cfg.multihetsep_crosscheck_replicates:
        jobs.append(
            Job(
                id=job_id("crosscheck", mask, rep(r)),
                command=step_cmd("crosscheck", "--vcf", str(variants), "--mask", str(bed), "--multihetsep", str(mhs),
                                 "--script", str(gms.path), "--work-dir", str(lay.mask_dir(mask, r)),
                                 "--out", str(lay.crosscheck(mask, r))),
                inputs=[variants, bed, mhs],
                outputs=[lay.crosscheck(mask, r)],
                mem_gb=ec.crosscheck_gb,
                est_seconds=5 + ec.crosscheck_s_per_mb * mb,
                tools={**step_tools, gms.name: gms.identity},
                description=f"multihetsep against generate_multihetsep.py, {mask} replicate {r}",
            )
        )
    for route in cfg.routes:
        inp, args = route_input(lay, mask, route, r)
        jobs.append(
            Job(
                id=job_id("fit", mask, route, rep(r)),
                command=tools.rs02_fit(rs_tool, cfg.inference, cfg.scaling, inp, lay.fit(mask, route, r),
                                       threads=cfg.rs_threads, extra=args),
                inputs=route_inputs(lay, mask, route, r),
                outputs=[lay.fit(mask, route, r)],
                cores=cfg.rs_threads,
                mem_gb=ec.fit_gb,
                est_seconds=costs.rs02_s_per_mb_iter * mb * cfg.inference.n_iter / cfg.rs_threads,
                settings={"inference": asdict(cfg.inference), "scaling": asdict(cfg.scaling),
                          "threads": cfg.rs_threads, "route": route, "mask": spec_settings},
                tools=rs,
                description=f"psmc-rs 0.2, route {route}, {mask} replicate {r}",
            )
        )
        jobs.append(
            Job(
                id=job_id("qc", mask, route, rep(r)),
                command=tools.rs02_qc(rs_tool, inp, lay.qc(mask, route, r), extra=[*args, "--bin-size", bin_size]),
                inputs=route_inputs(lay, mask, route, r),
                outputs=[lay.qc(mask, route, r)],
                mem_gb=ec.convert_gb,
                est_seconds=2 + ec.convert_s_per_mb * mb,
                settings={"route": route, "mask": spec_settings},
                tools=rs,
                description=f"psmc-rs qc, route {route}, {mask} replicate {r}",
            )
        )
    return jobs


def report_inputs(cfg: config.E4, lay: Layout) -> List[Path]:
    """Every file the tables are made from."""
    files: List[Path] = []
    for r in range(1, cfg.replicates + 1):
        files.append(lay.truth(r))
        for mask in mask_names(cfg):
            files += [lay.mask_record(mask, r), lay.consensus_record(mask, r), lay.bins_record(mask, r)]
            if r <= cfg.multihetsep_crosscheck_replicates:
                files.append(lay.crosscheck(mask, r))
            for route in cfg.routes:
                files += [lay.fit(mask, route, r), lay.qc(mask, route, r), lay.record(f"fit/{mask}/{route}/{rep(r)}")]
    return files


def _report_jobs(ctx: Context, cfg: config.E4, lay: Layout) -> List[Job]:
    costs, ec = ctx.profile.costs, config.E4_COSTS
    report = Job(
        id=job_id("report"),
        command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
        inputs=report_inputs(cfg, lay),
        outputs=[lay.table(k) for k in TABLES],
        mem_gb=costs.report_gb,
        est_seconds=costs.report_s,
        settings={"e4": asdict(cfg)},
        tools={"pipeline": provenance.python_identity(
            ["experiments/e4.py", "experiments/e1.py", "results.py", "metrics.py", "demography.py"], ["numpy"]
        )},
        description="E4 tables",
    )
    figure = Job(
        id=job_id("figure"),
        command=[paths.python(), "-m", "experiment.pipeline.figures.fig5", "--tables", str(lay.root / "tables"),
                 "--out-dir", str(lay.root / "figures")],
        inputs=[lay.table(k) for k in ("summary", "bias", "bins", "curves", "truth", "routes")],
        outputs=[lay.figure(f) for f in FIGURES],
        mem_gb=costs.report_gb,
        est_seconds=ec.figure_s,
        tools={"pipeline": provenance.python_identity(["figures/fig5.py"], ["matplotlib", "numpy"])},
        description="Fig 5",
    )
    return [report, figure]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _read(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.fmean(xs) if xs else None


def _sd(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def same_fit(a: Dict, b: Dict) -> bool:
    """Whether two psmc-rs fits have identical parameters and log-likelihoods."""
    keys = ("theta", "rho", "t_max", "lam", "loglike", "n_steps", "pattern")
    return all(a.get(k) == b.get(k) for k in keys)


def qc_findings(qc: Dict) -> List[Dict]:
    return [{"code": f["code"], "severity": f["severity"]} for f in qc.get("findings", [])]


def _codes(findings: Sequence[Dict], severity: str) -> str:
    return ";".join(f["code"] for f in findings if f["severity"] == severity)


def check_fit_input(fit: Dict, counts: Dict[str, int], where: str) -> None:
    """The fit read the bins its route's psmcfa holds (raises otherwise)."""
    inp = fit.get("input", {})
    want = {"bins": sum(counts.values()), "het_bins": counts["K"], "missing_bins": counts["N"]}
    got = {k: inp.get(k) for k in want}
    if got != want:
        raise ValueError(f"{where}: the fit read {got}, its route's bins are {want}")


@dataclass
class RouteFit:
    mask: str
    route: str
    replicate: int
    params: Dict
    fit: results.Fit
    curve: results.Curve
    qc: Dict
    run: Dict


def load_fits(cfg: config.E4, lay: Layout, r: int, mask: str, bins: Dict) -> Dict[str, RouteFit]:
    sc = cfg.scaling
    out: Dict[str, RouteFit] = {}
    for route in cfg.routes:
        params = results.read_rs_json(lay.fit(mask, route, r))
        if int(params.get("n_iter", -1)) != cfg.inference.n_iter:
            raise ValueError(f"{lay.fit(mask, route, r)}: not a fit of {cfg.inference.n_iter} iterations")
        counts = bins["exact"]["counts"] if route in ("exact",) else bins["routes"][route]["counts"]
        check_fit_input(params, counts, f"{mask} {route} {rep(r)}")
        fit = results.fit_from_rs(params)
        try:
            run = _read(lay.record(f"fit/{mask}/{route}/{rep(r)}")).get("run", {})
        except (OSError, ValueError):
            run = {}
        out[route] = RouteFit(mask, route, r, params, fit, fit.curve(sc.mu, sc.gen_years, cfg.inference.bin_size),
                              _read(lay.qc(mask, route, r)), run)
    return out


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E4 table from the jobs' outputs; returns the paths written."""
    import numpy as np

    cfg = ctx.profile.e4
    lay = Layout(ctx.exp_dir(NAME))
    g = cfg.scaling.gen_years
    epochs = demography.epochs(config.MODELS[cfg.model])
    window = metrics.model_grid(epochs, cfg.grid, g)
    fixed = metrics.fixed_grid(cfg.grid)
    truth_window = metrics.truth_on_grid(epochs, window, g)
    rows: Dict[str, List[Dict]] = {k: [] for k in TABLES}
    bias: Dict[Tuple[str, str], List] = {}
    dense = metrics.log_grid(*TRUTH_GRID_YEARS)
    for t, ne in zip(dense, metrics.truth_on_grid(epochs, dense, g)):
        rows["truth"].append({"model": cfg.model, "years": float(t), "ne": float(ne)})
    for r in range(1, cfg.replicates + 1):
        truth = _read(lay.truth(r))
        if demography.epochs_from_record(truth["truth"]) != epochs:
            raise ValueError(f"{lay.truth(r)}: not the {cfg.model} model")
        for ch in truth["chromosomes"]:
            rows["seeds"] += [
                {"replicate": r, "chromosome": ch["name"], "stage": "ancestry", "seed": ch["seeds"]["ancestry"]},
                {"replicate": r, "chromosome": ch["name"], "stage": "mutations", "seed": ch["seeds"]["mutations"]},
            ]
        het_simulated = truth["totals"]["het_sites"]
        complete_theta: Optional[float] = None
        for mask in mask_names(cfg):
            spec = mask_spec(cfg, mask)
            mrec = _read(lay.mask_record(mask, r))
            crec = _read(lay.consensus_record(mask, r))
            brec = _read(lay.bins_record(mask, r))
            if not brec.get("assertions_passed") or not brec["routes"]["fq2psmcfa"]["totals"]["identical"]:
                raise ValueError(f"{lay.bins_record(mask, r)}: fq2psmcfa's bins differ from the exact route's")
            tot = mrec["summary"]["totals"]
            rows["masks"].append(
                {
                    "mask": mask, "replicate": r,
                    "structure": spec.structure if spec else "none",
                    "missing_fraction_target": spec.missing_fraction if spec else 0.0,
                    "mean_run_bp_target": spec.mean_run_bp if spec else 0.0,
                    "missing_fraction": tot["missing_fraction"],
                    "uncallable_runs": tot["uncallable_runs"],
                    "mean_uncallable_run_bp": tot["mean_uncallable_run_bp"],
                    "het_sites_simulated": het_simulated,
                    "het_sites_kept": mrec["vcf"]["het_sites_kept"],
                    "het_sites_removed": mrec["vcf"]["het_sites_removed"],
                    "consensus_n_sites": sum(c["n_sites"] for c in crec["chromosomes"]),
                    "consensus_iupac_sites": sum(c["het_sites"] for c in crec["chromosomes"]),
                }
            )
            for chrom, s in mrec["seeds"].items():
                if s is not None:
                    rows["seeds"].append({"replicate": r, "chromosome": chrom, "stage": f"mask/{mask}", "seed": s})
            if mask == mask_names(cfg)[0]:
                for c in crec["chromosomes"]:
                    rows["seeds"].append({"replicate": r, "chromosome": c["name"], "stage": "reference",
                                          "seed": c["reference_seed"]})
            for route in BIN_ROUTES:
                comp = brec["routes"][route]
                t = comp["totals"]
                rows["bins"].append(
                    {"mask": mask, "replicate": r, "route": route, "bins_exact": t["bins_exact"],
                     "bins_route": t["bins_route"], "differing": t["differing"],
                     "differing_fraction": t["differing_fraction"], "identical": t["identical"],
                     "bytes_identical": comp["bytes_identical"], **t["confusion"]}
                )
            if r <= cfg.multihetsep_crosscheck_replicates:
                xrec = _read(lay.crosscheck(mask, r))
                for c in xrec["chromosomes"]:
                    rows["crosscheck"].append(
                        {"mask": mask, "replicate": r, "chromosome": c["name"],
                         "rows_generate_multihetsep": c["rows_generate_multihetsep"], "rows_psmc_rs": c["rows_psmc_rs"],
                         "identical": c["identical"],
                         "first_difference_row": (c["first_difference"] or {}).get("row"),
                         "generate_multihetsep_seconds": c["seconds"]}
                    )
            fits = load_fits(cfg, lay, r, mask, brec)
            exact = fits["exact"]
            if mask == config.E4_CONTROL_MASK:
                complete_theta = exact.fit.theta
            for route, rf in fits.items():
                findings = qc_findings(rf.qc)
                for f in findings:
                    rows["qc"].append({"mask": mask, "route": route, "replicate": r, **f})
                inp = rf.params.get("input", {})
                bias_r = metrics.log10_bias(rf.curve, epochs, window, g)
                bias.setdefault((mask, route), []).append(bias_r)
                diff_bins = brec["routes"][route]["totals"] if route in BIN_ROUTES else None
                qct = rf.qc.get("totals", {})
                rows["routes"].append(
                    {
                        "mask": mask, "route": route, "replicate": r,
                        "missing_fraction": tot["missing_fraction"],
                        "theta": rf.fit.theta, "rho": rf.fit.rho, "t_max": rf.fit.t_max,
                        "n0": rf.fit.n0(cfg.scaling.mu, cfg.inference.bin_size),
                        "theta_rel_exact": rf.fit.theta / exact.fit.theta - 1.0,
                        "theta_rel_complete": (rf.fit.theta / complete_theta - 1.0) if complete_theta else None,
                        "theta0": qct.get("theta0_per_bin"),
                        "bins": inp.get("bins"), "het_bins": inp.get("het_bins"), "missing_bins": inp.get("missing_bins"),
                        "missing_bin_fraction": inp.get("missing_bins", 0) / inp["bins"] if inp.get("bins") else None,
                        "differing_bin_fraction": diff_bins["differing_fraction"] if diff_bins else 0.0,
                        "rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(rf.curve, epochs, window, g),
                        "rmse_log10_ne_vs_truth_fixed": metrics.rmse_vs_truth(rf.curve, epochs, fixed, g),
                        "rmse_log10_ne_vs_exact": metrics.rmse_between(rf.curve, exact.curve, window),
                        "max_abs_log10_ne_vs_exact": float(np.max(np.abs(
                            np.log10(rf.curve.at(window)) - np.log10(exact.curve.at(window))))),
                        "identical_to_exact": same_fit(rf.params, exact.params),
                        "loglike_final": rf.params.get("final_loglike"),
                        "qc_warnings": _codes(findings, "warning"),
                        "qc_info": _codes(findings, "info"),
                        "qc_other": ";".join(f"{f['code']}:{f['severity']}" for f in findings
                                             if f["severity"] not in ("warning", "info")),
                        "wall_seconds": rf.run.get("wall_seconds"),
                        "peak_rss_mb": rf.run.get("peak_rss_mb"),
                    }
                )
                for k, (t, y, lam, ne) in enumerate(zip(rf.fit.boundaries(), rf.curve.years, rf.fit.lambdas, rf.curve.ne)):
                    rows["curves"].append({"mask": mask, "route": route, "replicate": r, "interval": k,
                                           "t_2n0": t, "years": y, "lambda": lam, "ne": ne})
    rows["summary"] = summary_rows(cfg, rows["routes"], rows["bins"])
    for (mask, route), series in bias.items():
        arr = np.vstack(series)
        for i, t in enumerate(window):
            col = arr[:, i]
            rows["bias"].append(
                {"mask": mask, "route": route, "years": float(t), "true_ne": float(truth_window[i]),
                 "replicates": int(len(col)), "log10_bias_mean": float(col.mean()),
                 "log10_bias_sd": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                 "log10_bias_min": float(col.min()), "log10_bias_max": float(col.max())}
            )
    written = {}
    for key in TABLES:
        write_csv(lay.table(key), rows[key], fields=FIELDS.get(key))
        written[key] = lay.table(key)
    return written


def summary_rows(cfg: config.E4, routes: List[Dict], bins: List[Dict]) -> List[Dict]:
    """One row per mask and route (Fig 5): means over replicates."""
    out = []
    for mask in mask_names(cfg):
        for route in cfg.routes:
            rs = [x for x in routes if x["mask"] == mask and x["route"] == route]
            n = len(rs)
            warn_sets = [set(filter(None, x["qc_warnings"].split(";"))) for x in rs]
            info_sets = [set(filter(None, x["qc_info"].split(";"))) for x in rs]
            spec = mask_spec(cfg, mask)
            out.append(
                {
                    "mask": mask, "route": route, "replicates": n,
                    "structure": spec.structure if spec else "none",
                    "missing_fraction_target": spec.missing_fraction if spec else 0.0,
                    "missing_fraction_mean": _mean([x["missing_fraction"] for x in rs]),
                    "differing_bin_fraction_mean": _mean([x["differing_bin_fraction"] for x in rs]),
                    "missing_bin_fraction_mean": _mean([x["missing_bin_fraction"] for x in rs]),
                    "theta_rel_exact_mean": _mean([x["theta_rel_exact"] for x in rs]),
                    "theta_rel_exact_sd": _sd([x["theta_rel_exact"] for x in rs]),
                    "theta_rel_complete_mean": _mean([x["theta_rel_complete"] for x in rs]),
                    "theta_rel_complete_sd": _sd([x["theta_rel_complete"] for x in rs]),
                    "rmse_truth_mean": _mean([x["rmse_log10_ne_vs_truth"] for x in rs]),
                    "rmse_truth_sd": _sd([x["rmse_log10_ne_vs_truth"] for x in rs]),
                    "rmse_truth_fixed_mean": _mean([x["rmse_log10_ne_vs_truth_fixed"] for x in rs]),
                    "rmse_exact_mean": _mean([x["rmse_log10_ne_vs_exact"] for x in rs]),
                    "rmse_exact_max": max((x["rmse_log10_ne_vs_exact"] for x in rs), default=None),
                    "fits_identical_to_exact": sum(1 for x in rs if x["identical_to_exact"]),
                    "qc_warnings_all": ";".join(sorted(set.intersection(*warn_sets))) if warn_sets else "",
                    "qc_warnings_any": ";".join(sorted(set.union(*warn_sets))) if warn_sets else "",
                    "qc_info_any": ";".join(sorted(set.union(*info_sets))) if info_sets else "",
                }
            )
    return out


FIELDS: Dict[str, Sequence[str]] = {
    "bins": ["mask", "replicate", "route", "bins_exact", "bins_route", "differing", "differing_fraction", "identical",
             "bytes_identical", "T_T", "T_K", "T_N", "K_T", "K_K", "K_N", "N_T", "N_K", "N_N", "other",
             "exact_only", "route_only"],
    "qc": ["mask", "route", "replicate", "code", "severity"],
    "crosscheck": ["mask", "replicate", "chromosome", "rows_generate_multihetsep", "rows_psmc_rs", "identical",
                   "first_difference_row", "generate_multihetsep_seconds"],
}


__all__ = ["BIN_ROUTES", "Layout", "NAME", "ROUTES", "TABLES", "jobs", "mask_names", "report", "route_input"]
