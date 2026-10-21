"""E4: input preparation (Fig 5).

How the path from a VCF and a callable mask to PSMC's input changes the
result, and what psmc-rs's per-bin callable-length model does about it. For
every replicate (:class:`.config.E4`: bottleneck model, 500 Mb as 10
chromosomes of 50 Mb) the runner simulates one diploid genome, draws
callable masks (missing fraction 0.1 or 0.3, clustered in runs of 20 kb or
dispersed in runs of 20 bp, plus the no-missing control), removes the
records at uncallable sites from the VCF, and takes the masked VCF to PSMC
by five routes, each fitted by psmc-rs 0.2 with its ``--bin-callable``
passed explicitly (``config.E4.routes`` and ``config.E4_BIN_CALLABLE``,
described in section 5 of config.py; their steps are in :mod:`.inputprep`):

* ``mask_auto``: the VCF with ``--vcf-mask``, ``--bin-callable auto``
  (psmc-rs's default): each called bin weighted by its callable bases,
  counted exactly from the mask;
* ``mask_ignore``: the same bins, ``--bin-callable ignore``: every called bin
  taken as ``bin_size`` bases, as fq2psmcfa and C psmc take it;
* ``fq2psmcfa``: consensus FASTA -> fq2psmcfa -> psmc-rs on the psmcfa, which
  records no callable bases (C psmc's model). Its bins must equal the exact
  bins, psmc-rs's bins of the VCF with its mask (the ``bins`` job fails
  otherwise), and its fit ``mask_ignore``'s, every fitted value of the JSON
  (the report fails otherwise);
* ``multihetsep``: ``psmc-rs convert --multihetsep`` read back with
  ``--bin-callable auto`` (cross-checked against generate_multihetsep.py on
  replicate 1). multihetsep says how many sites between two rows are
  callable, not which, so the reader places the uncalled ones at the start
  of each interval; ``auto`` fits exact counts only, so every called bin is
  taken as full: the route measures the placement alone (its summary rows:
  model ``full-bins``, counts ``approximate``);
* ``nomask``: the VCF without the mask (variants only): every bin full.

**Why.** fq2psmcfa's rule calls a bin missing only when more than 90% of its
bases are uncallable, so a dispersed mask leaves almost every bin called
while each holds only about (1 - f) of its 100 bases. Taking them as full
(``mask_ignore``, ``fq2psmcfa``) makes theta per bin, and with it N0 =
theta / (4 mu s), every time and every size, low by about (1 - f) or
more. ``mask_auto`` fits P(T | k) = exp(-theta (c / s) avg_t[k]) for a bin
of c callable bases (docs/design/bin-callable-model.md), which removes most
of the bias (experiment/validation/: theta 2-4% low on average at f = 0.3).

**Complete data.** On the ``nomissing`` control every bin is full, so the
callable-length model is C psmc's: the report fails unless ``mask_auto`` and
``mask_ignore`` fit identically there, and that fit is the complete-data
fit every route of the replicate is compared with.

**Rescaling cross-check** (``E4_rescaling.csv``, not a route). Multiplying
``mask_ignore``'s times and sizes by s / s_eff, with s_eff the mean callable
bases of its called bins (``bin_callable.callable.mean_fraction * s`` of its
JSON), corrects the bias on average, after the fit. The table sets that
global correction beside ``mask_auto``'s per-bin model: the two should agree
on the scale, and the per-bin model should be at least as close to the
complete-data fit. The rescaled fit has the pattern and the bins of
``mask_auto``'s, so the two are compared interval by interval
(:func:`.metrics.interval_diff`, ``*_per_interval``); their scales are the
theta columns.

**Against the complete-data fit.** A route's fit and the complete-data fit
have the same pattern but time scales that differ by percents (theta), so
the RMSE of log10 Ne between them (``rmse_log10_ne_vs_complete``) is the
exact mean over log time on the informative window
(:func:`.metrics.log_time_diff`), not a mean at the window's grid times:
on 200 times, a time that falls between two nearby boundaries of the two
fits reads a whole step, and such hits made the RMSE of close fits noisy
by tens of percent. ``max_abs_log10_ne_vs_complete`` is the largest
difference anywhere in the window, a whole step where two boundaries are
close.

QC codes are those of each fit's own QC (the ``qc`` record of its JSON,
which psmc-rs computes with the fit's settings, ``--bin-callable`` included).

Layout under ``<runs>/E4/``::

    sims/repNN/genome.psmcfa, genome.vcf.gz(.tbi)        simulated genome (not committed)
    sims/repNN/<mask>/callable.bed.gz                    callable sites
    sims/repNN/<mask>/variants.vcf.gz(.tbi)              the VCF of callable records
    sims/repNN/<mask>/fq2psmcfa.psmcfa                   route fq2psmcfa's input
    sims/repNN/<mask>/input.multihetsep                  route multihetsep's input
    sims/repNN/<mask>/bins/{exact,multihetsep,nomask}.psmcfa  the bins as psmc-rs reads them
    truth/repNN.json                                     seeds, counts, checksums, true Ne(t)
    masks/<mask>/repNN.json                              the mask drawn: missing fraction, runs, seeds
    consensus/<mask>/repNN.json                          the consensus FASTA's counts and seeds
    bins/<mask>/repNN.json                               every route's bins against the exact bins
    crosscheck/<mask>/rep01.json                         psmc-rs vs generate_multihetsep.py
    fits/<mask>/<route>/repNN.json                       psmc-rs 0.2
    provenance/, tables/E4_*.csv                         (logs/ and sims/ are not committed)
    figures/fig5_input_preparation.{pdf,svg,png}         Fig 5, drawn from the tables (experiment FIG)

Tables (``tables/``):

* ``E4_summary.csv``: one row per mask and route (Fig 5): replicates, the
  emission model fitted and where its callable bases came from, the
  realised missing fraction, the mean share of callable bases of the called
  bins and the share of them partly callable, the share of bins unlike the
  exact bins, theta relative to the complete-data fit, RMSE of log10 Ne
  against the truth and against the complete-data fit of the same genome,
  the mean log10 bias, how many fits equal ``mask_auto``'s and
  ``mask_ignore``'s, and the QC codes.
* ``E4_routes.csv``: one row per mask, route and replicate: the fit, its input
  as read, its callable bases, the metrics and the QC codes.
* ``E4_bias.csv``: log10(fitted / true Ne) at every time of the model's
  informative window, mean, SD, min and max over replicates (Fig 5).
* ``E4_rescaling.csv``: the rescaling cross-check, one row per mask and
  replicate.
* ``E4_bins.csv``: one row per mask, replicate and route: how its bins differ
  from the exact bins (T/K/N against T/K/N, and bins only one route has).
* ``E4_masks.csv``: one row per mask and replicate: target and realised
  missing fraction, uncallable runs, heterozygous sites removed.
* ``E4_qc.csv``: one row per QC finding of a fit (mask, route, replicate,
  code, severity: warning or note).
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import read_rs_fit, rep, run_record, step_rows, write_tables

NAME = "E4"
ROUTES = ("mask_auto", "mask_ignore", "fq2psmcfa", "multihetsep", "nomask")
MASK_ROUTES = ("mask_auto", "mask_ignore")  # the VCF with its mask: the exact bins, both models
BIN_ROUTES = ("fq2psmcfa", "multihetsep", "nomask")  # routes with bins of their own, compared with the exact bins
COMPLETE_ROUTE = "mask_auto"  # on the control: the complete-data fit
# Keys of a psmc-rs JSON that are not the fit: its input and how it was read
# (two routes fit identically when everything else is equal), and the HMM
# kernels that computed it (the portable and the AVX2+FMA kernels differ in
# the last digits, so a fit on a node without AVX2, or with
# PSMC_RS_SIMD=portable, shows in the values).
NOT_FIT_KEYS = ("input", "qc", "bin_callable", "simd")
TABLES = {
    "summary": "E4_summary.csv",
    "routes": "E4_routes.csv",
    "bias": "E4_bias.csv",
    "rescaling": "E4_rescaling.csv",
    "bins": "E4_bins.csv",
    "masks": "E4_masks.csv",
    "qc": "E4_qc.csv",
    "crosscheck": "E4_crosscheck.csv",
    "curves": "E4_curves.csv",
    "truth": "E4_truth.csv",
    "seeds": "E4_seeds.csv",
}
# The columns of these tables are fixed (a table without rows keeps its header); the others take their first row's.
FIELDS: Dict[str, Sequence[str]] = {
    "bins": ["mask", "replicate", "route", "bins_exact", "bins_route", "differing", "differing_fraction", "identical",
             "bytes_identical", "T_T", "T_K", "T_N", "K_T", "K_K", "K_N", "N_T", "N_K", "N_N", "other",
             "exact_only", "route_only"],
    "qc": ["mask", "route", "replicate", "code", "severity"],
    "crosscheck": ["mask", "replicate", "chromosome", "rows_generate_multihetsep", "rows_psmc_rs", "identical",
                   "first_difference_row", "generate_multihetsep_seconds"],
    "rescaling": ["mask", "replicate", "structure", "missing_fraction", "mean_callable_fraction", "scale",
                  *(f"{m}_{v}" for m in ("theta_rel_complete", "rmse_truth", "rmse_complete", "mean_log10_bias")
                    for v in ("auto", "ignore", "rescaled")),
                  "rmse_rescaled_vs_auto_per_interval", "max_abs_rescaled_vs_auto_per_interval"],
}


def mask_names(cfg: config.E4) -> Tuple[str, ...]:
    """The masks of ``cfg``: the no-missing control first, then the simulated masks."""
    names = tuple(m.name for m in cfg.masks)
    if config.E4_CONTROL_MASK in names:
        raise ValueError(f"mask name {config.E4_CONTROL_MASK!r} is reserved for the control")
    return ((config.E4_CONTROL_MASK,) if cfg.no_missing_control else ()) + names


def check_routes(cfg: config.E4) -> None:
    """The routes of ``cfg`` are known, and include the two mask routes every comparison needs."""
    unknown = [r for r in cfg.routes if r not in ROUTES]
    if unknown:
        raise ValueError(f"unknown E4 route(s) {', '.join(unknown)}; known: {', '.join(ROUTES)}")
    missing = [r for r in MASK_ROUTES if r not in cfg.routes]
    if missing:
        raise ValueError(f"E4 needs the routes {', '.join(MASK_ROUTES)}; missing: {', '.join(missing)}")


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

    def exact_bins(self, mask: str, r: int) -> Path:
        """psmc-rs's bins of the VCF with its mask: the bins both mask routes fit."""
        return self.mask_dir(mask, r) / "bins" / "exact.psmcfa"

    def route_bins(self, mask: str, route: str, r: int) -> Path:
        """The bins of a route as psmc-rs reads them (fq2psmcfa: its own output)."""
        if route in MASK_ROUTES:
            return self.exact_bins(mask, r)
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

    def fit_record(self, mask: str, route: str, r: int) -> Path:
        """The provenance record of a fit's job."""
        return self.root / "provenance" / "fit" / mask / route / f"{rep(r)}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_id(*parts: str) -> str:
    return "/".join((NAME, *parts))


def route_input(lay: Layout, mask: str, route: str, r: int) -> Tuple[Path, List[str]]:
    """The file a route gives psmc-rs and the input options it reads it with."""
    if route in MASK_ROUTES:
        return lay.variants(mask, r), ["--input-format", "vcf", "--vcf-mode", "mask", "--vcf-mask", str(lay.bed(mask, r))]
    if route == "fq2psmcfa":
        return lay.fq2psmcfa(mask, r), ["--input-format", "psmcfa"]
    if route == "multihetsep":
        return lay.multihetsep(mask, r), ["--input-format", "mhs"]
    if route == "nomask":
        return lay.variants(mask, r), ["--input-format", "vcf", "--vcf-mode", "mask"]
    raise KeyError(f"unknown route {route!r}")


def route_inputs(lay: Layout, mask: str, route: str, r: int) -> List[Path]:
    """Every file psmc-rs reads for a route (the mask routes' mask too)."""
    inp, _ = route_input(lay, mask, route, r)
    return [inp, lay.bed(mask, r)] if route in MASK_ROUTES else [inp]


def step_cmd(*args: str) -> List[str]:
    """``python -m experiment.pipeline.inputprep ARGS``."""
    return [paths.python(), "-m", "experiment.pipeline.inputprep", *args]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shared:
    """What the jobs of every replicate and mask share: tools, identities, the genome's size."""

    rs: tools.Tool
    fq2psmcfa: tools.Tool
    generate_multihetsep: tools.Tool
    steps: Dict  # identity of the Python steps (inputprep.py)
    htslib: Dict
    mb: float  # genome size, for the cost estimates

    @property
    def rs_identity(self) -> Dict:
        return {self.rs.name: self.rs.identity}


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e4
    check_routes(cfg)
    costs = ctx.profile.costs
    lay = Layout(ctx.exp_dir(NAME))
    model = cfg.model
    htslib = tools.htslib_versions()
    sh = Shared(
        rs=ctx.rs02(),
        fq2psmcfa=ctx.utility("fq2psmcfa"),
        generate_multihetsep=ctx.utility("generate_multihetsep.py"),
        steps={"pipeline": provenance.python_identity(["inputprep.py"], ["numpy"])},
        htslib=htslib,
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
            out += mask_jobs(ctx, lay, sh, mask, r)
    out.append(report_job(ctx, cfg, lay))
    return out


def mask_jobs(ctx: Context, lay: Layout, sh: Shared, mask: str, r: int) -> List[Job]:
    """The jobs of one replicate and mask: the mask, the routes' inputs and bins, the fits."""
    cfg, ec = ctx.profile.e4, config.E4_COSTS
    model, bin_size = cfg.model, str(cfg.inference.bin_size)
    spec = mask_spec(cfg, mask)
    spec_settings = asdict(spec) if spec else {"name": mask, "missing_fraction": 0.0}
    spec_args = [] if spec is None else [
        "--missing-fraction", repr(spec.missing_fraction), "--structure", spec.structure,
        "--mean-run-bp", repr(spec.mean_run_bp),
    ]
    seed_args = ["--experiment", NAME, "--model", model, "--replicate", str(r)]
    variants, bed, mhs = lay.variants(mask, r), lay.bed(mask, r), lay.multihetsep(mask, r)
    out = [
        Job(
            id=job_id("mask", mask, rep(r)),
            command=step_cmd("mask", *seed_args, "--vcf", str(lay.vcf(r)), "--mask-name", mask, *spec_args,
                             "--bed-out", str(bed), "--vcf-out", str(variants), "--record", str(lay.mask_record(mask, r))),
            inputs=[lay.vcf(r)],
            outputs=[bed, variants, Path(f"{variants}.tbi"), lay.mask_record(mask, r)],
            mem_gb=ec.mask_gb,
            est_seconds=5 + ec.mask_s_per_mb * sh.mb,
            settings={"mask": spec_settings, "genome": asdict(cfg.genome)},
            tools={**sh.steps, "htslib": sh.htslib},
            seeds={c: (config.seed(NAME, model, r, f"mask/{mask}/{c}") if spec else None) for c in cfg.genome.names()},
            description=f"mask {mask} on replicate {r}",
        ),
        Job(
            id=job_id("consensus", mask, rep(r)),
            command=step_cmd("consensus", *seed_args, "--vcf", str(variants), "--mask", str(bed),
                             "--fq2psmcfa", str(sh.fq2psmcfa.path), "--bin-size", bin_size,
                             "--out", str(lay.fq2psmcfa(mask, r)), "--record", str(lay.consensus_record(mask, r))),
            inputs=[variants, bed],
            outputs=[lay.fq2psmcfa(mask, r), lay.consensus_record(mask, r)],
            mem_gb=ec.consensus_gb,
            est_seconds=5 + ec.consensus_s_per_mb * sh.mb,
            settings={"bin_size": cfg.inference.bin_size},
            tools={**sh.steps, sh.fq2psmcfa.name: sh.fq2psmcfa.identity},
            seeds={c: config.seed(NAME, model, r, f"reference/{c}") for c in cfg.genome.names()},
            description=f"consensus FASTA -> fq2psmcfa, {mask} replicate {r}",
        ),
    ]

    def convert(route: str, *id_route: str, what: str, multihetsep: bool = False) -> Job:
        inp, args = route_input(lay, mask, route, r)
        outputs = [lay.route_bins(mask, route, r), *([mhs] if multihetsep else [])]
        return Job(
            id=job_id("convert", mask, *id_route, rep(r)),
            command=[str(sh.rs.path), "convert", str(inp), *args, "--bin-size", bin_size, "--psmcfa", str(outputs[0]),
                     *(["--multihetsep", str(mhs)] if multihetsep else []), "--quiet"],
            inputs=route_inputs(lay, mask, route, r),
            outputs=outputs,
            mem_gb=ec.convert_gb,
            est_seconds=2 + ec.convert_s_per_mb * sh.mb,
            tools=sh.rs_identity,
            description=f"psmc-rs convert ({what}), {mask} replicate {r}",
        )

    # The exact bins (the mask routes') and the multihetsep route's input, in one pass.
    out.append(convert(MASK_ROUTES[0], what="exact bins, multihetsep", multihetsep=True))
    out += [convert(route, route, what=f"{route} bins") for route in ("multihetsep", "nomask")]
    out.append(
        Job(
            id=job_id("bins", mask, rep(r)),
            command=step_cmd("bins", "--exact", str(lay.exact_bins(mask, r)),
                             *[a for route in BIN_ROUTES for a in ("--route", f"{route}={lay.route_bins(mask, route, r)}")],
                             "--assert-identical", "fq2psmcfa", "--out", str(lay.bins_record(mask, r))),
            inputs=[lay.exact_bins(mask, r), *(lay.route_bins(mask, route, r) for route in BIN_ROUTES)],
            outputs=[lay.bins_record(mask, r)],
            mem_gb=ec.bins_gb,
            est_seconds=2 + ec.bins_s_per_mb * sh.mb,
            tools=sh.steps,
            description=f"bins of every route against the exact bins, {mask} replicate {r}",
        )
    )
    if r <= cfg.multihetsep_crosscheck_replicates:
        gms = sh.generate_multihetsep
        out.append(
            Job(
                id=job_id("crosscheck", mask, rep(r)),
                command=step_cmd("crosscheck", "--vcf", str(variants), "--mask", str(bed), "--multihetsep", str(mhs),
                                 "--script", str(gms.path), "--work-dir", str(lay.mask_dir(mask, r)),
                                 "--out", str(lay.crosscheck(mask, r))),
                inputs=[variants, bed, mhs],
                outputs=[lay.crosscheck(mask, r)],
                mem_gb=ec.crosscheck_gb,
                est_seconds=5 + ec.crosscheck_s_per_mb * sh.mb,
                tools={**sh.steps, gms.name: gms.identity},
                description=f"multihetsep against generate_multihetsep.py, {mask} replicate {r}",
            )
        )
    for route in cfg.routes:
        inp, args = route_input(lay, mask, route, r)
        mode = config.E4_BIN_CALLABLE[route]
        out.append(
            Job(
                id=job_id("fit", mask, route, rep(r)),
                command=tools.rs02_fit(sh.rs, cfg.inference, cfg.scaling, inp, lay.fit(mask, route, r),
                                       threads=cfg.rs_threads, extra=[*args, "--bin-callable", mode]),
                inputs=route_inputs(lay, mask, route, r),
                outputs=[lay.fit(mask, route, r)],
                cores=cfg.rs_threads,
                mem_gb=ec.fit_gb,
                est_seconds=ctx.profile.costs.rs02_seconds(sh.mb, cfg.inference, cfg.rs_threads),
                settings={"inference": asdict(cfg.inference), "scaling": asdict(cfg.scaling),
                          "threads": cfg.rs_threads, "route": route, "mask": spec_settings,
                          "bin_callable": mode},
                tools=sh.rs_identity,
                description=f"psmc-rs 0.2, route {route} (--bin-callable {mode}), {mask} replicate {r}",
            )
        )
    return out


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
                files += [lay.fit(mask, route, r), lay.fit_record(mask, route, r)]
    return files


def report_job(ctx: Context, cfg: config.E4, lay: Layout) -> Job:
    costs = ctx.profile.costs
    return Job(
        id=job_id("report"),
        command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
        inputs=report_inputs(cfg, lay),
        outputs=[lay.table(k) for k in TABLES],
        mem_gb=costs.report_gb,
        est_seconds=costs.report_s,
        settings={"e4": asdict(cfg), "bin_callable": {r: config.E4_BIN_CALLABLE[r] for r in cfg.routes}},
        tools={"pipeline": provenance.python_identity(
            ["experiments/e4.py", "experiments/e1.py", "results.py", "metrics.py", "demography.py"], ["numpy"]
        )},
        description="E4 tables",
    )


# ---------------------------------------------------------------------------
# Reading the outputs back
# ---------------------------------------------------------------------------


def _read(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def fit_differences(a: Dict, b: Dict) -> List[str]:
    """The keys of two psmc-rs JSONs whose fits differ (empty when they fit identically)."""
    keys = (set(a) | set(b)).difference(NOT_FIT_KEYS)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def same_fit(a: Dict, b: Dict) -> bool:
    """Whether two psmc-rs JSONs record the same fit: every value but the input's and the kernels', to the bit."""
    return not fit_differences(a, b)


def qc_findings(params: Dict) -> List[Dict]:
    """The QC codes of a fit, from its JSON: warnings, then notes."""
    qc = params.get("qc") or {}
    return ([{"code": c, "severity": "warning"} for c in qc.get("warnings", [])]
            + [{"code": c, "severity": "note"} for c in qc.get("notes", [])])


def _codes(findings: Sequence[Dict], severity: str) -> str:
    return ";".join(f["code"] for f in findings if f["severity"] == severity)


def callable_summary(params: Dict) -> Optional[Dict]:
    """The callable bases of a fit's called bins (``bin_callable.callable`` of its JSON); None without counts."""
    return (params.get("bin_callable") or {}).get("callable")


def mean_callable_fraction(params: Dict) -> Optional[float]:
    """The mean share of callable bases over the called bins; None when the input does not count them (psmcfa)."""
    c = callable_summary(params)
    return None if c is None else float(c["mean_fraction"])


def partly_callable_fraction(params: Dict) -> Optional[float]:
    """The share of called bins with fewer callable bases than the bin size; None without counts."""
    c = callable_summary(params)
    if c is None:
        return None
    return c["partial_bins"] / c["called_bins"] if c["called_bins"] else 0.0


def check_fit_input(fit: Dict, counts: Dict[str, int], where: str) -> None:
    """The fit read the bins its route's psmcfa holds (raises otherwise)."""
    inp = fit.get("input", {})
    want = {"bins": sum(counts.values()), "het_bins": counts["K"], "missing_bins": counts["N"]}
    got = {k: inp.get(k) for k in want}
    if got != want:
        raise ValueError(f"{where}: the fit read {got}, its route's bins are {want}")


def check_fit_mode(fit: Dict, route: str, where: str) -> None:
    """The fit ran with its route's ``--bin-callable`` (raises otherwise)."""
    mode, want = (fit.get("bin_callable") or {}).get("mode"), config.E4_BIN_CALLABLE[route]
    if mode != want:
        raise ValueError(f"{where}: the fit records --bin-callable {mode!r}, route {route} passes "
                         f"{want!r} (a psmc-rs without the callable-length model?)")


@dataclass
class RouteFit:
    mask: str
    route: str
    replicate: int
    params: Dict
    fit: results.Fit
    curve: results.Curve
    run: Dict


def load_fits(cfg: config.E4, lay: Layout, r: int, mask: str, bins: Dict) -> Dict[str, RouteFit]:
    """The fits of one replicate and mask, checked against the bins record ``bins`` and their routes' modes."""
    sc = cfg.scaling
    out: Dict[str, RouteFit] = {}
    for route in cfg.routes:
        where = f"{mask} {route} {rep(r)}"
        params = read_rs_fit(lay.fit(mask, route, r), cfg.inference.n_iter)
        counts = bins["exact"]["counts"] if route in MASK_ROUTES else bins["routes"][route]["counts"]
        check_fit_input(params, counts, where)
        check_fit_mode(params, route, where)
        fit = results.fit_from_rs(params)
        out[route] = RouteFit(mask, route, r, params, fit, fit.curve(sc.mu, sc.gen_years, cfg.inference.bin_size),
                              run_record(lay.fit_record(mask, route, r)))
    return out


def check_identities(mask: str, r: int, fits: Dict[str, RouteFit]) -> None:
    """The fits that must be identical are (raises otherwise).

    fq2psmcfa's bins are the exact bins, and a psmcfa records no callable
    bases, so it must fit as ``mask_ignore`` does; on the control every bin
    is full, so ``mask_auto`` must fit as ``mask_ignore`` does.
    """
    pairs = [("fq2psmcfa", "mask_ignore")]
    if mask == config.E4_CONTROL_MASK:
        pairs.append(("mask_auto", "mask_ignore"))
    for a, b in pairs:
        if a in fits and b in fits:
            diff = fit_differences(fits[a].params, fits[b].params)
            if diff:
                raise ValueError(f"{mask} {rep(r)}: the {a} fit differs from the {b} fit in {', '.join(diff)}; "
                                 f"they must be identical")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _values(xs: Sequence[Optional[float]]) -> List[float]:
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = _values(xs)
    return statistics.fmean(xs) if xs else None


def _sd(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = _values(xs)
    return statistics.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def _max(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = _values(xs)
    return max(xs) if xs else None


def mask_row(mask: str, spec: Optional[config.MaskSpec], r: int, mrec: Dict, crec: Dict, het_simulated: int) -> Dict:
    """The masks table's row of one mask and replicate, from its mask and consensus records."""
    tot = mrec["summary"]["totals"]
    return {
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


def bins_rows(mask: str, r: int, brec: Dict) -> List[Dict]:
    """How the bins of each route with bins of its own differ from the exact bins (the bins record ``brec``)."""
    out = []
    for route in BIN_ROUTES:
        comp = brec["routes"][route]
        t = comp["totals"]
        out.append({"mask": mask, "replicate": r, "route": route, "bins_exact": t["bins_exact"],
                    "bins_route": t["bins_route"], "differing": t["differing"],
                    "differing_fraction": t["differing_fraction"], "identical": t["identical"],
                    "bytes_identical": comp["bytes_identical"], **t["confusion"]})
    return out


def crosscheck_rows(mask: str, r: int, xrec: Dict) -> List[Dict]:
    """One row per chromosome of a crosscheck record: psmc-rs's multihetsep against generate_multihetsep.py's."""
    return [{"mask": mask, "replicate": r, "chromosome": c["name"],
             "rows_generate_multihetsep": c["rows_generate_multihetsep"], "rows_psmc_rs": c["rows_psmc_rs"],
             "identical": c["identical"], "first_difference_row": (c["first_difference"] or {}).get("row"),
             "generate_multihetsep_seconds": c["seconds"]}
            for c in xrec["chromosomes"]]


@dataclass(frozen=True)
class TrueModel:
    """The simulated model, and the windows on which a fitted curve is compared with it and with the complete-data fit."""

    epochs: List[demography.Epoch]
    window: Any  # the informative window's times (years)
    fixed: Any  # the fixed window's times
    gen_years: float

    def vs_truth(self, curve: results.Curve) -> Dict:
        import numpy as np

        e, w, g = self.epochs, self.window, self.gen_years
        return {"rmse_log10_ne_vs_truth": metrics.rmse_vs_truth(curve, e, w, g),
                "rmse_log10_ne_vs_truth_fixed": metrics.rmse_vs_truth(curve, e, self.fixed, g),
                "mean_log10_bias": float(np.mean(metrics.log10_bias(curve, e, w, g)))}

    def vs_complete(self, curve: results.Curve, complete: Optional[RouteFit]) -> Dict:
        if complete is None:
            return {"rmse_log10_ne_vs_complete": None, "max_abs_log10_ne_vs_complete": None}
        diff = metrics.log_time_diff(curve, complete.curve, (self.window[0], self.window[-1]))
        return {"rmse_log10_ne_vs_complete": diff.rmse, "max_abs_log10_ne_vs_complete": diff.max_abs}


def route_row(cfg: config.E4, rf: RouteFit, fits: Dict[str, RouteFit], complete: Optional[RouteFit], brec: Dict,
              missing_fraction: float, findings: Sequence[Dict], truth: TrueModel) -> Dict:
    """The routes table's row of one fit: its input as read, its callable bases, the metrics and the QC codes."""
    p, fit = rf.params, rf.fit
    inp, qc, bc = p.get("input", {}), p.get("qc") or {}, p.get("bin_callable") or {}
    return {
        "mask": rf.mask, "route": rf.route, "replicate": rf.replicate,
        "bin_callable": bc.get("mode"), "bin_model": bc.get("model"), "counts": bc.get("counts"),
        "missing_fraction": missing_fraction,
        "theta": fit.theta, "rho": fit.rho, "t_max": fit.t_max, "n0": fit.n0(cfg.scaling.mu, cfg.inference.bin_size),
        "theta_rel_complete": (fit.theta / complete.fit.theta - 1.0) if complete else None,
        "theta0_per_bin": qc.get("theta0_per_bin"),
        "bins": inp.get("bins"), "het_bins": inp.get("het_bins"), "missing_bins": inp.get("missing_bins"),
        "missing_bin_fraction": inp.get("missing_bins", 0) / inp["bins"] if inp.get("bins") else None,
        "differing_bin_fraction": (brec["routes"][rf.route]["totals"]["differing_fraction"]
                                   if rf.route in BIN_ROUTES else 0.0),
        "called_bases": qc.get("called_bases"),
        "mean_callable_fraction": mean_callable_fraction(p),
        "partly_callable_fraction": partly_callable_fraction(p),
        **truth.vs_truth(rf.curve),
        **truth.vs_complete(rf.curve, complete),
        "identical_to_mask_auto": same_fit(p, fits["mask_auto"].params),
        "identical_to_mask_ignore": same_fit(p, fits["mask_ignore"].params),
        "loglike_final": p.get("final_loglike"),
        "qc_warnings": _codes(findings, "warning"),
        "qc_notes": _codes(findings, "note"),
        "wall_seconds": rf.run.get("wall_seconds"),
        "peak_rss_mb": rf.run.get("peak_rss_mb"),
    }


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E4 table from the jobs' outputs; returns the paths written."""
    cfg = ctx.profile.e4
    check_routes(cfg)
    lay = Layout(ctx.exp_dir(NAME))
    g = cfg.scaling.gen_years
    epochs = demography.epochs(config.MODELS[cfg.model])
    truth = TrueModel(epochs, metrics.model_grid(epochs, cfg.grid, g), metrics.fixed_grid(cfg.grid), g)
    masks = mask_names(cfg)
    rows: Dict[str, List[Dict]] = {k: [] for k in TABLES}
    bias: Dict[Tuple[str, str], List] = {}  # (mask, route): log10 bias on the window, one array per replicate
    dense = metrics.log_grid(*config.TRUTH_TABLE_YEARS)
    rows["truth"] = [{"model": cfg.model, "years": float(t), "ne": float(ne)}
                     for t, ne in zip(dense, metrics.truth_on_grid(epochs, dense, g))]
    for r in range(1, cfg.replicates + 1):
        record = _read(lay.truth(r))
        if demography.epochs_from_record(record["truth"]) != epochs:
            raise ValueError(f"{lay.truth(r)}: not the {cfg.model} model")
        rows["seeds"] += [{"replicate": r, "chromosome": ch["name"], "stage": stage, "seed": ch["seeds"][stage]}
                          for ch in record["chromosomes"] for stage in ("ancestry", "mutations")]
        complete: Optional[RouteFit] = None  # the control's mask_auto fit
        for mask in masks:
            mrec = _read(lay.mask_record(mask, r))
            crec = _read(lay.consensus_record(mask, r))
            brec = _read(lay.bins_record(mask, r))
            if not brec.get("assertions_passed") or not brec["routes"]["fq2psmcfa"]["totals"]["identical"]:
                raise ValueError(f"{lay.bins_record(mask, r)}: fq2psmcfa's bins differ from the exact bins")
            missing_fraction = mrec["summary"]["totals"]["missing_fraction"]
            rows["masks"].append(mask_row(mask, mask_spec(cfg, mask), r, mrec, crec, record["totals"]["het_sites"]))
            rows["seeds"] += [{"replicate": r, "chromosome": chrom, "stage": f"mask/{mask}", "seed": seed}
                              for chrom, seed in mrec["seeds"].items() if seed is not None]
            if mask == masks[0]:
                rows["seeds"] += [{"replicate": r, "chromosome": c["name"], "stage": "reference",
                                   "seed": c["reference_seed"]} for c in crec["chromosomes"]]
            rows["bins"] += bins_rows(mask, r, brec)
            if r <= cfg.multihetsep_crosscheck_replicates:
                rows["crosscheck"] += crosscheck_rows(mask, r, _read(lay.crosscheck(mask, r)))
            fits = load_fits(cfg, lay, r, mask, brec)
            check_identities(mask, r, fits)
            if mask == config.E4_CONTROL_MASK:
                complete = fits[COMPLETE_ROUTE]
            for route, rf in fits.items():
                findings = qc_findings(rf.params)
                rows["qc"] += [{"mask": mask, "route": route, "replicate": r, **f} for f in findings]
                bias.setdefault((mask, route), []).append(metrics.log10_bias(rf.curve, epochs, truth.window, g))
                rows["routes"].append(route_row(cfg, rf, fits, complete, brec, missing_fraction, findings, truth))
                rows["curves"] += step_rows(rf.fit, rf.curve, mask=mask, route=route, replicate=r)
            rows["rescaling"].append(rescaling_row(cfg, mask, r, missing_fraction, fits, complete, truth))
    rows["summary"] = summary_rows(cfg, rows["routes"])
    rows["bias"] = bias_rows(bias, truth.window, metrics.truth_on_grid(epochs, truth.window, g))
    return write_tables(lay.table, rows, FIELDS)


def rescaled_curve(cfg: config.E4, rf: RouteFit, mean_fraction: float) -> results.Curve:
    """``rf``'s curve read with s_eff = ``mean_fraction`` x bin size bases per bin: times and sizes x s / s_eff."""
    sc = cfg.scaling
    return rf.fit.curve(sc.mu, sc.gen_years, cfg.inference.bin_size * mean_fraction)


def rescaling_row(cfg: config.E4, mask: str, r: int, missing_fraction: float, fits: Dict[str, RouteFit],
                  complete: Optional[RouteFit], truth: TrueModel) -> Dict:
    """The rescaling cross-check of one mask and replicate: mask_ignore rescaled after the fit, beside mask_auto."""
    auto, ignore = fits["mask_auto"], fits["mask_ignore"]
    f = mean_callable_fraction(ignore.params)
    spec = mask_spec(cfg, mask)
    row: Dict = {"mask": mask, "replicate": r, "structure": spec.structure if spec else "none",
                 "missing_fraction": missing_fraction, "mean_callable_fraction": f,
                 "scale": (1.0 / f) if f else None}
    rescaled = rescaled_curve(cfg, ignore, f) if f else None
    theta_c = complete.fit.theta if complete else None
    for name, curve, theta in (("auto", auto.curve, auto.fit.theta), ("ignore", ignore.curve, ignore.fit.theta),
                               ("rescaled", rescaled, ignore.fit.theta / f if f else None)):
        t, c = (truth.vs_truth(curve), truth.vs_complete(curve, complete)) if curve is not None else ({}, {})
        row[f"theta_rel_complete_{name}"] = (theta / theta_c - 1.0) if (theta and theta_c) else None
        row[f"rmse_truth_{name}"] = t.get("rmse_log10_ne_vs_truth")
        row[f"rmse_complete_{name}"] = c.get("rmse_log10_ne_vs_complete")
        row[f"mean_log10_bias_{name}"] = t.get("mean_log10_bias")
    diff = metrics.interval_diff(rescaled, auto.curve, (truth.window[0], truth.window[-1])) if rescaled else None
    row["rmse_rescaled_vs_auto_per_interval"] = diff.rmse if diff else None
    row["max_abs_rescaled_vs_auto_per_interval"] = diff.max_abs if diff else None
    return row


def bias_rows(bias: Dict[Tuple[str, str], List], window, truth_window) -> List[Dict]:
    """log10(fitted / true Ne) at each time of the window, over the replicates of each mask and route."""
    import numpy as np

    out = []
    for (mask, route), series in bias.items():
        arr = np.vstack(series)
        for i, t in enumerate(window):
            col = arr[:, i]
            out.append(
                {"mask": mask, "route": route, "years": float(t), "true_ne": float(truth_window[i]),
                 "replicates": int(len(col)), "log10_bias_mean": float(col.mean()),
                 "log10_bias_sd": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                 "log10_bias_min": float(col.min()), "log10_bias_max": float(col.max())}
            )
    return out


def summary_rows(cfg: config.E4, routes: List[Dict]) -> List[Dict]:
    """One row per mask and route (Fig 5): means over replicates (``routes``: the routes table's rows)."""
    by_key: Dict[Tuple[str, str], List[Dict]] = {}
    for x in routes:
        by_key.setdefault((x["mask"], x["route"]), []).append(x)
    out = []
    for mask in mask_names(cfg):
        spec = mask_spec(cfg, mask)
        for route in cfg.routes:
            rs = by_key.get((mask, route), [])
            warn_sets = [set(filter(None, x["qc_warnings"].split(";"))) for x in rs]
            note_sets = [set(filter(None, x["qc_notes"].split(";"))) for x in rs]

            def col(key: str) -> List:
                return [x.get(key) for x in rs]

            def one(key: str) -> str:
                return ";".join(sorted({str(v) for v in col(key) if v is not None}))

            out.append(
                {
                    "mask": mask, "route": route, "replicates": len(rs),
                    "structure": spec.structure if spec else "none",
                    "missing_fraction_target": spec.missing_fraction if spec else 0.0,
                    "bin_callable": one("bin_callable"), "bin_model": one("bin_model"), "counts": one("counts"),
                    "missing_fraction_mean": _mean(col("missing_fraction")),
                    "mean_callable_fraction_mean": _mean(col("mean_callable_fraction")),
                    "partly_callable_fraction_mean": _mean(col("partly_callable_fraction")),
                    "missing_bin_fraction_mean": _mean(col("missing_bin_fraction")),
                    "differing_bin_fraction_mean": _mean(col("differing_bin_fraction")),
                    "theta_rel_complete_mean": _mean(col("theta_rel_complete")),
                    "theta_rel_complete_sd": _sd(col("theta_rel_complete")),
                    "rmse_truth_mean": _mean(col("rmse_log10_ne_vs_truth")),
                    "rmse_truth_sd": _sd(col("rmse_log10_ne_vs_truth")),
                    "rmse_truth_fixed_mean": _mean(col("rmse_log10_ne_vs_truth_fixed")),
                    "rmse_complete_mean": _mean(col("rmse_log10_ne_vs_complete")),
                    "rmse_complete_sd": _sd(col("rmse_log10_ne_vs_complete")),
                    "rmse_complete_max": _max(col("rmse_log10_ne_vs_complete")),
                    "mean_log10_bias_mean": _mean(col("mean_log10_bias")),
                    "fits_identical_to_mask_auto": sum(1 for x in rs if x["identical_to_mask_auto"]),
                    "fits_identical_to_mask_ignore": sum(1 for x in rs if x["identical_to_mask_ignore"]),
                    "qc_warnings_all": ";".join(sorted(set.intersection(*warn_sets))) if warn_sets else "",
                    "qc_warnings_any": ";".join(sorted(set.union(*warn_sets))) if warn_sets else "",
                    "qc_notes_any": ";".join(sorted(set.union(*note_sets))) if note_sets else "",
                }
            )
    return out


__all__ = [
    "BIN_ROUTES", "COMPLETE_ROUTE", "Layout", "MASK_ROUTES", "NAME", "ROUTES", "TABLES", "check_identities",
    "check_routes", "fit_differences", "jobs", "mask_names", "mean_callable_fraction", "report", "rescaled_curve",
    "route_input", "same_fit",
]
