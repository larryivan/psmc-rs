"""E6: supplementary analyses S1-S5 (Figs S1-S4, Table S5).

E6 runs little of its own: it reads what E1, E2 and E3 make, where the design
says so, and adds the runs that only it needs. Its settings are
:class:`.config.E6`; the models, inference settings, scaling, threads and
time grid of S1-S3 are E1's (``profile.e1``).

* **S1, EM convergence.** E1's fits of every model and replicate: the
  log-likelihood and the parameters after every iteration, for both tools.
* **S2, format consistency.** Replicate ``format_replicate`` of every E1
  model, read by psmc-rs as psmcfa (E1's fit), as the simulated VCF (no
  missing data, so no mask) and as multihetsep (``psmc-rs convert
  --multihetsep`` of that VCF). The VCF must give the psmcfa's bins exactly;
  the multihetsep ends at its last row, so it lacks the bins after the last
  heterozygous site, which are counted. Each format is fitted with E1's
  settings and the fits are compared.
* **S3, TMRCA decoding.** E1's psmc-rs fit of ``tmrca_model`` replicate
  ``tmrca_replicate`` is decoded on its psmcfa (``--init-params`` of the
  fit, 0 iterations, ``--tmrca-out``), so the posterior is that of the
  model E1 reports; the true pairwise TMRCA comes from re-simulating the
  replicate from its truth record (``simulate --tmrca-only``, which checks
  the heterozygous sites against the record). Per bin: correlation of the
  posterior mean and MAP with the truth, coverage of the 95% interval, and a
  5 Mb example track.
* **S4, thread scaling and determinism.** E3's psmc-rs 0.2 runs: E6 finds
  them among E3's jobs and names their provenance records on the report's
  command line (``--record``); the report reads the runner's measurements
  from them (wall time, peak RSS of the process tree, user and system time,
  load before and after, the busiest other processes) and compares the
  JSON (and ``.psmc``) the runs wrote byte for byte across thread counts.
  The contract with E3: one runner job per run, whose command contains
  ``<psmc-rs 0.2> INPUT OUTPUT.json N_ITER ... --threads T`` (as
  :func:`.tools.rs02_fit` writes it). Every input fitted with two or more
  thread counts is tabulated (the thread-scaling genome, and every size run
  with 1 and 8 threads). Without E3 (not registered, or its jobs cannot be
  listed) S4 is left out and the log says why.
* **S5, QC table.** ``psmc-rs qc --json`` of E2's four real genomes, with
  E2's bin size and bootstrap trunk (``--bootstrap-block-size``). E6 runs it
  itself: a QC report depends only on the file and these options and takes
  a fraction of a second, so S5 does not depend on how E2 lays out its files.

Checks: the S2 VCF route must reproduce the psmcfa bins and fit, the
multihetsep bins may differ only by the tail after the last heterozygous
bin, the S3 decoding must use E1's parameters exactly, and the S4 outputs
must be byte-identical across thread counts. A report job writes its tables
and then fails when a check does not hold, so the tables can be inspected
and nothing downstream (figures) runs on them.

Layout under ``<runs>/E6/``::

    sims/s2/<model>/repNN/{from_vcf.psmcfa, genome.multihetsep, from_multihetsep.psmcfa}   (not committed)
    sims/s3/<model>/repNN/{tmrca_true.tsv.gz, posterior.tsv.gz}                             (not committed)
    fits/s2/<model>/repNN.{vcf,multihetsep}.{json,psmc}     psmc-rs fits of the other formats
    fits/s3/<model>/repNN.decode.json                       the decoding run (E1's parameters)
    qc/s5/<sample>.json                                     psmc-rs qc of the real genomes
    tables/E6_S*.csv, figures/FigS*.{pdf,png}, provenance/, logs/ (not committed)

The Python steps run as ``python -m experiment.pipeline.experiments.e6
{report,figure} --part sN``; ``python -m experiment.pipeline report E6``
writes the tables of every part whose inputs exist.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .. import config, demography, metrics, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context, e1

NAME = "E6"
PARTS = ("s1", "s2", "s3", "s4", "s5")
TABLES: Dict[str, Dict[str, str]] = {
    "s1": {"traces": "E6_S1_traces.csv", "summary": "E6_S1_summary.csv", "agreement": "E6_S1_agreement.csv"},
    "s2": {"bins": "E6_S2_bins.csv", "fits": "E6_S2_fits.csv", "curves": "E6_S2_curves.csv"},
    "s3": {
        "summary": "E6_S3_summary.csv",
        "calibration": "E6_S3_calibration.csv",
        "density": "E6_S3_density.csv",
        "track": "E6_S3_track.csv",
    },
    "s4": {"runs": "E6_S4_runs.csv", "scaling": "E6_S4_scaling.csv", "determinism": "E6_S4_determinism.csv"},
    "s5": {"qc": "E6_S5_qc.csv", "findings": "E6_S5_findings.csv"},
}
FIGURES = {
    "s1": "FigS1_em_convergence",
    "s2": "FigS2_format_consistency",
    "s3": "FigS3_tmrca_decoding",
    "s4": "FigS4_thread_scaling",
}
FIGURE_FORMATS = ("pdf", "png")
S2_FORMATS = {"vcf": ["--input-format", "vcf", "--vcf-mode", "mask"], "multihetsep": ["--input-format", "mhs"]}
REPORT_MODULES = ["experiments/e6.py", "experiments/e6_analysis.py", "experiments/e1.py", "results.py",
                  "metrics.py", "demography.py", "simulate.py"]
FIGURE_MODULES = ["experiments/e6.py", "figures/e6_supplementary.py"]


def rep(r: int) -> str:
    return f"rep{r:02d}"


class CheckFailed(RuntimeError):
    """A consistency check of E6 did not hold (the tables were written)."""


class MissingInput(FileNotFoundError):
    """An input of a part does not exist yet."""


@dataclass(frozen=True)
class Layout:
    """Where E6's files are."""

    root: Path

    # S2
    def s2_dir(self, model: str, r: int) -> Path:
        return self.root / "sims" / "s2" / model / rep(r)

    def s2_vcf_psmcfa(self, model: str, r: int) -> Path:
        return self.s2_dir(model, r) / "from_vcf.psmcfa"

    def s2_multihetsep(self, model: str, r: int) -> Path:
        return self.s2_dir(model, r) / "genome.multihetsep"

    def s2_mhs_psmcfa(self, model: str, r: int) -> Path:
        return self.s2_dir(model, r) / "from_multihetsep.psmcfa"

    def s2_fit(self, model: str, r: int, fmt: str) -> Path:
        return self.root / "fits" / "s2" / model / f"{rep(r)}.{fmt}.json"

    def s2_fit_psmc(self, model: str, r: int, fmt: str) -> Path:
        return self.root / "fits" / "s2" / model / f"{rep(r)}.{fmt}.psmc"

    # S3
    def s3_true_track(self, model: str, r: int) -> Path:
        return self.root / "sims" / "s3" / model / rep(r) / "tmrca_true.tsv.gz"

    def s3_posterior(self, model: str, r: int) -> Path:
        return self.root / "sims" / "s3" / model / rep(r) / "posterior.tsv.gz"

    def s3_decode(self, model: str, r: int) -> Path:
        return self.root / "fits" / "s3" / model / f"{rep(r)}.decode.json"

    # S5
    def s5_qc(self, sample_id: str) -> Path:
        return self.root / "qc" / "s5" / f"{sample_id}.json"

    # Tables and figures
    def table(self, part: str, key: str) -> Path:
        return self.root / "tables" / TABLES[part][key]

    def tables(self, part: str) -> List[Path]:
        return [self.table(part, k) for k in TABLES[part]]

    def figures(self, part: str) -> List[Path]:
        return [self.root / "figures" / f"{FIGURES[part]}.{ext}" for ext in FIGURE_FORMATS]


def job_id(*parts) -> str:
    return "/".join([NAME, *[str(p) for p in parts]])


def e6_command(ctx: Context, action: str, part: str) -> List[str]:
    """The command of one of E6's Python steps (``report`` or ``figure`` of a part)."""
    return [paths.python(), "-m", "experiment.pipeline.experiments.e6", action, "--part", part, *ctx.profile_args()]


def _report_job(ctx: Context, part: str, inputs: Sequence[Path], settings: Dict, est: float, mem: float, what: str) -> Job:
    lay = Layout(ctx.exp_dir(NAME))
    return Job(
        id=job_id(part, "report"),
        command=e6_command(ctx, "report", part),
        inputs=list(inputs),
        outputs=lay.tables(part),
        mem_gb=mem,
        est_seconds=est,
        settings=settings,
        tools={"pipeline": provenance.python_identity(REPORT_MODULES, ["numpy", "pandas", "scipy"])},
        description=what,
    )


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def s1_inputs(ctx: Context) -> List[Path]:
    """What the S1 report reads: E1's truth records and fits."""
    e1cfg, e1lay = ctx.profile.e1, e1.Layout(ctx.exp_dir(e1.NAME))
    inputs: List[Path] = []
    for model in e1cfg.models:
        for r in range(1, e1cfg.replicates + 1):
            inputs += [e1lay.truth(model, r), e1lay.c_psmc(model, r), e1lay.rs_json(model, r), e1lay.rs_psmc(model, r)]
    return inputs


def s1_jobs(ctx: Context) -> List[Job]:
    """S1: one report job over E1's fits."""
    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    inputs = s1_inputs(ctx)
    settings = {
        "e1": {"models": list(e1cfg.models), "replicates": e1cfg.replicates, "inference": asdict(e1cfg.inference)},
        "e6": {"convergence_within_loglike": cfg.convergence_within_loglike,
               "loglike_decrease_tolerance": cfg.loglike_decrease_tolerance},
    }
    return [_report_job(ctx, "s1", inputs, settings, cfg.costs.table_s, cfg.costs.table_gb, "S1 tables: EM convergence")]


def s2_inputs(ctx: Context) -> List[Path]:
    """What the S2 report reads: E1's genome, truth and fit, the converted formats and their fits."""
    e1cfg, r = ctx.profile.e1, ctx.profile.e6.format_replicate
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    inputs: List[Path] = []
    for model in e1cfg.models:
        inputs += [
            e1lay.psmcfa(model, r), e1lay.truth(model, r), e1lay.rs_json(model, r),
            lay.s2_vcf_psmcfa(model, r), lay.s2_mhs_psmcfa(model, r),
            lay.s2_fit(model, r, "vcf"), lay.s2_fit(model, r, "multihetsep"),
        ]
    return inputs


def s2_jobs(ctx: Context) -> List[Job]:
    """S2: per E1 model, convert the VCF, convert the multihetsep, fit both formats; then the report."""
    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    costs, gcosts = cfg.costs, ctx.profile.costs
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    r = cfg.format_replicate
    if not 1 <= r <= e1cfg.replicates:
        raise ValueError(f"E6 format_replicate {r} is not an E1 replicate (1..{e1cfg.replicates})")
    rs = ctx.rs02()
    genome_mb = e1cfg.genome.length / 1e6
    bin_size = str(e1cfg.inference.bin_size)
    convert_s = costs.convert_s_base + costs.convert_s_per_mb * genome_mb
    fit_s = gcosts.rs02_s_per_mb_iter * genome_mb * e1cfg.inference.n_iter / e1cfg.rs_threads + convert_s
    out: List[Job] = []
    for model in e1cfg.models:
        vcf, mhs = e1lay.vcf(model, r), lay.s2_multihetsep(model, r)
        out.append(Job(
            id=job_id("s2", "convert-vcf", model, rep(r)),
            command=[rs.path, "convert", vcf, *S2_FORMATS["vcf"], "--bin-size", bin_size,
                     "--psmcfa", lay.s2_vcf_psmcfa(model, r), "--multihetsep", mhs, "-q"],
            inputs=[vcf],
            outputs=[lay.s2_vcf_psmcfa(model, r), mhs],
            mem_gb=costs.convert_gb,
            est_seconds=convert_s,
            settings={"bin_size": e1cfg.inference.bin_size, "formats": S2_FORMATS["vcf"]},
            tools={rs.name: rs.identity},
            description=f"S2: {model} {rep(r)} VCF as psmcfa and multihetsep",
        ))
        out.append(Job(
            id=job_id("s2", "convert-multihetsep", model, rep(r)),
            command=[rs.path, "convert", mhs, *S2_FORMATS["multihetsep"], "--bin-size", bin_size,
                     "--psmcfa", lay.s2_mhs_psmcfa(model, r), "-q"],
            inputs=[mhs],
            outputs=[lay.s2_mhs_psmcfa(model, r)],
            mem_gb=costs.convert_gb,
            est_seconds=convert_s,
            settings={"bin_size": e1cfg.inference.bin_size, "formats": S2_FORMATS["multihetsep"]},
            tools={rs.name: rs.identity},
            description=f"S2: {model} {rep(r)} multihetsep as psmcfa",
        ))
        for fmt, inp in (("vcf", vcf), ("multihetsep", mhs)):
            out.append(Job(
                id=job_id("s2", f"fit-{fmt}", model, rep(r)),
                command=tools.rs02_fit(
                    rs, e1cfg.inference, e1cfg.scaling, inp, lay.s2_fit(model, r, fmt), threads=e1cfg.rs_threads,
                    psmc_out=lay.s2_fit_psmc(model, r, fmt), extra=S2_FORMATS[fmt],
                ),
                inputs=[inp],
                outputs=[lay.s2_fit(model, r, fmt), lay.s2_fit_psmc(model, r, fmt)],
                cores=e1cfg.rs_threads,
                mem_gb=gcosts.rs02_gb + costs.convert_gb,
                est_seconds=fit_s,
                settings={"inference": asdict(e1cfg.inference), "scaling": asdict(e1cfg.scaling),
                          "threads": e1cfg.rs_threads, "formats": S2_FORMATS[fmt]},
                tools={rs.name: rs.identity},
                description=f"S2: psmc-rs 0.2 on {model} {rep(r)} as {fmt}",
            ))
    settings = {"e1": {"models": list(e1cfg.models), "inference": asdict(e1cfg.inference),
                       "scaling": asdict(e1cfg.scaling), "grid": asdict(e1cfg.grid)},
                "e6": {"format_replicate": r}}
    out.append(_report_job(ctx, "s2", s2_inputs(ctx), settings, costs.table_s, costs.table_gb, "S2 tables: formats"))
    return out


def s3_inputs(ctx: Context) -> List[Path]:
    """What the S3 report reads: E1's truth record and fit, the true track, the decoding run and its posterior."""
    cfg = ctx.profile.e6
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    model, r = cfg.tmrca_model, cfg.tmrca_replicate
    return [e1lay.truth(model, r), lay.s3_true_track(model, r), lay.s3_posterior(model, r), lay.s3_decode(model, r),
            e1lay.rs_json(model, r)]


def s3_jobs(ctx: Context) -> List[Job]:
    """S3: the true TMRCA track, the decoding of E1's fit, and the report."""
    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    costs = cfg.costs
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    model, r = cfg.tmrca_model, cfg.tmrca_replicate
    if model not in e1cfg.models or not 1 <= r <= e1cfg.replicates:
        raise ValueError(f"E6 decodes {model} {rep(r)}, which E1 ({', '.join(e1cfg.models)} x {e1cfg.replicates}) does not fit")
    if abs(cfg.tmrca_interval - 0.95) > 1e-12:
        raise ValueError("psmc-rs --tmrca-out gives the 2.5% and 97.5% quantiles: tmrca_interval must be 0.95")
    rs = ctx.rs02()
    genome_mb = e1cfg.genome.length / 1e6
    inf, sc = e1cfg.inference, e1cfg.scaling
    truth, track, post, decode = e1lay.truth(model, r), lay.s3_true_track(model, r), lay.s3_posterior(model, r), lay.s3_decode(model, r)
    sim_tools = provenance.python_identity(["simulate.py", "demography.py"], ["msprime", "tskit", "numpy"])
    jobs = [
        Job(
            id=job_id("s3", "truth", model, rep(r)),
            command=ctx.pipeline_cmd(
                "simulate", "--profile", ctx.profile.name, "--experiment", e1.NAME, "--model", model,
                "--replicate", str(r), "--truth", str(truth), "--tmrca-only", "--tmrca-out", str(track),
            ),
            inputs=[truth],
            outputs=[track],
            mem_gb=costs.tmrca_truth_gb,
            est_seconds=costs.tmrca_truth_s_base + costs.tmrca_truth_s_per_mb * genome_mb,
            settings={"model": asdict(config.MODELS[model]), "genome": asdict(e1cfg.genome),
                      "simulation": asdict(e1cfg.simulation)},
            tools={"pipeline": sim_tools},
            seeds={c: simulate.seeds_for(e1.NAME, model, r, c) for c in e1cfg.genome.names()},
            description=f"S3: true TMRCA of E1 {model} {rep(r)}",
        ),
        Job(
            id=job_id("s3", "decode", model, rep(r)),
            command=[
                rs.path, e1lay.psmcfa(model, r), decode, "0",
                "--init-params", e1lay.rs_json(model, r),
                "--pattern", inf.pattern,
                "--smooth-lambda", tools.fmt(inf.smooth_lambda),
                "--mstep-iters", str(inf.mstep_iters),
                "--bin-size", str(inf.bin_size),
                "--mu", tools.fmt(sc.mu),
                "--gen-years", tools.fmt(sc.gen_years),
                "--threads", str(cfg.tmrca_decode_threads),
                "--no-progress",
                "--tmrca-out", post,
            ],
            inputs=[e1lay.psmcfa(model, r), e1lay.rs_json(model, r)],
            outputs=[decode, post],
            cores=cfg.tmrca_decode_threads,
            mem_gb=costs.decode_gb,
            est_seconds=costs.decode_s_per_mb * genome_mb,
            settings={"inference": asdict(inf), "scaling": asdict(sc), "threads": cfg.tmrca_decode_threads},
            tools={rs.name: rs.identity},
            description=f"S3: decode E1's psmc-rs fit of {model} {rep(r)}",
        ),
    ]
    settings = {
        "e1": {"inference": asdict(inf), "scaling": asdict(sc), "genome": asdict(e1cfg.genome)},
        "e6": {k: getattr(cfg, k) for k in (
            "tmrca_model", "tmrca_replicate", "tmrca_interval", "tmrca_example_track_bp", "tmrca_example_chrom",
            "tmrca_example_start_bp", "tmrca_calibration_classes", "tmrca_density_cells")},
    }
    jobs.append(_report_job(
        ctx, "s3", s3_inputs(ctx), settings,
        costs.tmrca_report_s_base + costs.tmrca_report_s_per_mb * genome_mb, costs.tmrca_report_gb,
        "S3 tables: TMRCA decoding",
    ))
    return jobs


# --- S4: E3's psmc-rs 0.2 runs ------------------------------------------------


@dataclass(frozen=True)
class RsRun:
    """One psmc-rs 0.2 fit that another experiment's job runs."""

    job_id: str
    input: Path
    output: Path  # the parameter JSON
    threads: int
    record: Path  # the job's provenance record: the runner's measurements
    psmc_out: Optional[Path] = None


def parse_rs02_fit(command: Sequence[str], binary: str) -> Optional[Dict]:
    """``input``, ``output``, ``threads`` and ``psmc_out`` of a psmc-rs 0.2 fit command, or None.

    The command must hold ``<binary> INPUT OUTPUT.json N_ITER ... --threads T``
    (:func:`.tools.rs02_fit`), possibly after a wrapper; ``qc`` and
    ``convert`` are not fits.
    """
    cmd = [str(c) for c in command]
    if binary not in cmd:
        return None
    args = cmd[cmd.index(binary) + 1 :]
    if len(args) < 3 or args[0] in ("qc", "convert", "help") or not args[1].endswith(".json") or "--threads" not in args:
        return None
    try:
        threads = int(args[args.index("--threads") + 1])
    except (IndexError, ValueError):
        return None
    psmc = args[args.index("--psmc-out") + 1] if "--psmc-out" in args[:-1] else None
    return {"input": args[0], "output": args[1], "threads": threads, "psmc_out": psmc}


def rs02_run(job: Job, rs02_path: Path, runs_dir: Path) -> Optional[RsRun]:
    """The psmc-rs 0.2 fit that ``job`` runs (:func:`parse_rs02_fit`), or None when it runs something else."""
    fit = parse_rs02_fit(job.command, str(rs02_path))
    if fit is None:
        return None
    psmc = Path(fit["psmc_out"]) if fit["psmc_out"] else None
    return RsRun(job.id, Path(fit["input"]), Path(fit["output"]), fit["threads"], job.record_path(runs_dir), psmc)


def local_path(text: str, runs_dir: Path) -> str:
    """The inverse of :func:`.paths.portable`: placeholders (``<runs>``, ``<repo>``, ...) back to this machine's paths."""
    for prefix, name in paths.placeholders(runs_dir).items():
        text = text.replace(name, prefix)
    return text


def run_from_record(record: Path, runs_dir: Path) -> RsRun:
    """The psmc-rs 0.2 fit recorded in a provenance record (the command of its key)."""
    rec = json.loads(Path(record).read_text())
    key = rec.get("key", {})
    fit = parse_rs02_fit(key.get("command", []), paths.portable(str(tools.rs02_binary()), runs_dir))
    if fit is None:
        raise ValueError(f"{record}: not the record of a psmc-rs 0.2 fit with --threads")
    loc = lambda p: Path(local_path(p, runs_dir))  # noqa: E731
    return RsRun(key.get("job", str(record)), loc(fit["input"]), loc(fit["output"]), fit["threads"], Path(record),
                 loc(fit["psmc_out"]) if fit["psmc_out"] else None)


def scaling_groups(runs: Sequence[RsRun]) -> "OrderedDict[str, List[RsRun]]":
    """Runs grouped by input, for the inputs fitted with at least two thread counts."""
    groups: "OrderedDict[str, List[RsRun]]" = OrderedDict()
    for r in runs:
        groups.setdefault(str(r.input), []).append(r)
    return OrderedDict(
        (k, sorted(v, key=lambda r: (r.threads, r.job_id)))
        for k, v in groups.items()
        if len({r.threads for r in v}) >= 2
    )


def source_jobs(ctx: Context, experiment: str) -> Optional[List[Job]]:
    """The jobs of another experiment, or None when it is not registered or its tools are missing."""
    from . import REGISTRY, module

    if experiment not in REGISTRY:
        ctx.log(f"[E6] S4 is left out: experiment {experiment} is not registered")
        return None
    try:
        return module(experiment).jobs(ctx)
    except (tools.ToolError, OSError) as exc:
        ctx.log(f"[E6] S4 is left out: the jobs of {experiment} cannot be listed ({exc})")
        return None


def thread_scaling(ctx: Context) -> "OrderedDict[str, List[RsRun]]":
    """E3's psmc-rs 0.2 runs, grouped by input (inputs fitted with two or more thread counts)."""
    src = source_jobs(ctx, ctx.profile.e6.thread_scaling_experiment)
    if not src:
        return OrderedDict()
    binary = ctx.rs02().path
    runs = [run for run in (rs02_run(j, binary, ctx.runs_dir) for j in src) if run is not None]
    return scaling_groups(runs)


def s4_jobs(ctx: Context) -> List[Job]:
    """S4: one report job over E3's runs, when there are any.

    The report reads the runs' provenance records, named on its command line
    (``--record``), and their outputs; it does not need E3's module.
    """
    cfg = ctx.profile.e6
    groups = thread_scaling(ctx)
    if not groups:
        return []
    inputs: List[Path] = []
    records: List[str] = []
    for runs in groups.values():
        for run in runs:
            inputs += [run.output, run.record] + ([run.psmc_out] if run.psmc_out else [])
            records += ["--record", str(run.record)]
    settings = {"source": cfg.thread_scaling_experiment,
                "runs": {paths.portable(k, ctx.runs_dir): [r.job_id for r in v] for k, v in groups.items()}}
    job = _report_job(ctx, "s4", inputs, settings, cfg.costs.table_s, cfg.costs.table_gb, "S4 tables: threads")
    job.command = job.command + records
    return [job]


def s5_jobs(ctx: Context) -> List[Job]:
    """S5: psmc-rs qc of E2's genomes, and the report."""
    cfg = ctx.profile.e6
    src = getattr(ctx.profile, cfg.qc_experiment.lower())
    lay = Layout(ctx.exp_dir(NAME))
    rs = ctx.rs02()
    extra = ["--bin-size", str(src.inference.bin_size), "--bootstrap-block-size", str(src.bootstrap_trunk_bins)]
    out: List[Job] = []
    for g in src.genomes:
        out.append(Job(
            id=job_id("s5", "qc", g.sample_id),
            command=tools.rs02_qc(rs, paths.REAL_DATA / g.file, lay.s5_qc(g.sample_id), extra=extra),
            inputs=[paths.REAL_DATA / g.file],
            outputs=[lay.s5_qc(g.sample_id)],
            mem_gb=cfg.costs.qc_gb,
            est_seconds=cfg.costs.qc_s,
            settings={"options": extra},
            tools={rs.name: rs.identity},
            description=f"S5: psmc-rs qc of {g.label}",
        ))
    settings = {"genomes": [asdict(g) for g in src.genomes]}
    out.append(_report_job(ctx, "s5", [lay.s5_qc(g.sample_id) for g in src.genomes], settings,
                           cfg.costs.table_s, cfg.costs.table_gb, "S5 table: QC of the real genomes"))
    return out


def figure_job(ctx: Context, part: str) -> Job:
    cfg = ctx.profile.e6
    lay = Layout(ctx.exp_dir(NAME))
    return Job(
        id=job_id(part, "figure"),
        command=e6_command(ctx, "figure", part),
        inputs=lay.tables(part),
        outputs=lay.figures(part),
        mem_gb=cfg.costs.figure_gb,
        est_seconds=cfg.costs.figure_s,
        settings={"figure": FIGURES[part], "formats": list(FIGURE_FORMATS)},
        tools={"pipeline": provenance.python_identity(FIGURE_MODULES, ["matplotlib", "numpy", "pandas"])},
        description=f"{FIGURES[part]}",
    )


def jobs(ctx: Context) -> List[Job]:
    out: List[Job] = []
    for part, make in (("s1", s1_jobs), ("s2", s2_jobs), ("s3", s3_jobs), ("s4", s4_jobs), ("s5", s5_jobs)):
        mine = make(ctx)
        out += mine
        if mine and part in FIGURES:
            out.append(figure_job(ctx, part))
    return out


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _require(paths_: Sequence[Path]) -> None:
    for p in paths_:
        if not Path(p).exists():
            raise MissingInput(f"{p} does not exist")


def significant(rows: List[Dict], digits: int = 6) -> List[Dict]:
    """``rows`` with every float rounded to ``digits`` significant digits (for the large S3 tables)."""
    def r(x):
        return float(f"{x:.{digits}g}") if isinstance(x, float) and math.isfinite(x) else x

    return [{k: r(v) for k, v in row.items()} for row in rows]


def _write(lay: Layout, part: str, tables: Dict[str, List[Dict]]) -> Dict[str, Path]:
    written = {}
    for key, rows in tables.items():
        e1.write_csv(lay.table(part, key), rows)
        written[f"{part}/{key}"] = lay.table(part, key)
    return written


def report_s1(ctx: Context) -> Dict[str, Path]:
    """S1 tables from E1's fits."""
    from . import e6_analysis as A

    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    e1lay = e1.Layout(ctx.exp_dir(e1.NAME))
    n = e1cfg.inference.n_iter
    _require(s1_inputs(ctx))
    traces: List[Dict] = []
    for model in e1cfg.models:
        for r in range(1, e1cfg.replicates + 1):
            rp = e1.load_replicate(e1lay, e1cfg, model, r)
            traces += A.trace_rows(model, r, rp.c, rp.rs_params, rp.rs, n)
    summary, agreement = [], []
    for model in e1cfg.models:
        mine = [t for t in traces if t["model"] == model]
        for tool in A.TOOLS:
            summary.append(A.convergence_summary(model, tool, mine, n, cfg.convergence_within_loglike,
                                                 cfg.loglike_decrease_tolerance))
        agreement.append(A.agreement_summary(model, mine, n))
    return _write(Layout(ctx.exp_dir(NAME)), "s1", {"traces": traces, "summary": summary, "agreement": agreement})


def report_s2(ctx: Context) -> Dict[str, Path]:
    """S2 tables; fails after writing them when the formats disagree beyond the multihetsep tail."""
    from . import e6_analysis as A

    import numpy as np

    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    sc, inf = e1cfg.scaling, e1cfg.inference
    r = cfg.format_replicate
    _require(s2_inputs(ctx))
    bins_rows, fit_rows, curves = [], [], []
    lo, hi, npts = e1.TRUTH_GRID_YEARS
    dense = np.logspace(np.log10(lo), np.log10(hi), npts)
    for model in e1cfg.models:
        ref = A.read_psmcfa(e1lay.psmcfa(model, r))
        bins_rows += A.compare_bins(model, "vcf", ref, A.read_psmcfa(lay.s2_vcf_psmcfa(model, r)))
        bins_rows += A.compare_bins(model, "multihetsep", ref, A.read_psmcfa(lay.s2_mhs_psmcfa(model, r)))
        fits = OrderedDict([("psmcfa", A.FormatFit("psmcfa", results.read_rs_json(e1lay.rs_json(model, r))))])
        for fmt in ("vcf", "multihetsep"):
            fits[fmt] = A.FormatFit(fmt, results.read_rs_json(lay.s2_fit(model, r, fmt)))
        truth = json.loads(e1lay.truth(model, r).read_text())
        epochs = demography.epochs_from_record(truth["truth"])
        window = metrics.model_grid(epochs, e1cfg.grid, sc.gen_years)
        rows = A.fit_rows(model, fits, epochs, window, sc.mu, sc.gen_years, inf.bin_size)
        for row in rows:
            row["replicate"] = r
        fit_rows += rows
        for fmt, ff in fits.items():
            curves += A.curve_rows(model, fmt, ff.fit, sc.mu, sc.gen_years, inf.bin_size)
        curves += A.truth_rows(model, epochs, dense, sc.gen_years)
    written = _write(lay, "s2", {"bins": bins_rows, "fits": fit_rows, "curves": curves})
    problems = s2_problems(bins_rows, fit_rows)
    if problems:
        raise CheckFailed("S2: " + "; ".join(problems))
    return written


def s2_problems(bins_rows: Sequence[Dict], fit_rows: Sequence[Dict]) -> List[str]:
    """What contradicts the expected S2 result (empty when everything holds)."""
    out = []
    for b in bins_rows:
        if b["sequence"] != "all":
            continue
        if b["format"] == "vcf" and not b["identical"]:
            out.append(f"{b['model']}: the VCF gives {b['bins_differing']} differing bins and {b['bins_format']} "
                       f"bins instead of {b['bins_psmcfa']}")
        if b["format"] == "multihetsep" and (b["bins_differing"] or b["bins_extra_at_end"] or not b["tail_is_after_last_het"]):
            out.append(f"{b['model']}: the multihetsep differs beyond the tail after the last heterozygous bin")
    for f in fit_rows:
        if f["format"] == "vcf" and not (f["parameters_identical"] and f["loglikes_identical"]):
            out.append(f"{f['model']}: the fit of the VCF differs from the fit of the psmcfa")
    return out


def report_s3(ctx: Context) -> Dict[str, Path]:
    """S3 tables; fails after writing them when the decoding is not of E1's model or the states do not match."""
    from . import e6_analysis as A

    import pandas as pd

    cfg, e1cfg = ctx.profile.e6, ctx.profile.e1
    lay, e1lay = Layout(ctx.exp_dir(NAME)), e1.Layout(ctx.exp_dir(e1.NAME))
    sc, bin_size = e1cfg.scaling, e1cfg.inference.bin_size
    model, r = cfg.tmrca_model, cfg.tmrca_replicate
    _require(s3_inputs(ctx))
    truth = json.loads(e1lay.truth(model, r).read_text())
    decoded, fitted = results.read_rs_json(lay.s3_decode(model, r)), results.read_rs_json(e1lay.rs_json(model, r))
    same_model = all(decoded.get(k) == fitted.get(k) for k in ("theta", "rho", "t_max", "lam", "pattern", "n_steps"))
    same_ll = decoded.get("final_loglike") == fitted.get("final_loglike")
    left_edges = results.fit_from_rs(decoded).curve(sc.mu, sc.gen_years, bin_size).years
    track = simulate.read_tmrca_track(lay.s3_true_track(model, r))
    names = [ch["name"] for ch in truth["chromosomes"]]
    truths = [A.bin_tmrca(track[ch["name"]], int(ch["length"]), bin_size) for ch in truth["chromosomes"]]
    posterior = pd.read_csv(
        lay.s3_posterior(model, r), sep="\t", usecols=list(A.POSTERIOR_COLUMNS),
        dtype={"seq_id": "int32", "seq_bin": "int64", "obs": "str", "map_state": "int32"},
    )
    d = A.align(posterior, truths, left_edges, sc.gen_years)
    del posterior
    mismatches = A.state_mismatches(d)
    summary = []
    for label, mask in (("all", None), ("heterozygous (K)", d.obs == "K"), ("homozygous (T)", d.obs == "T")):
        row = {"model": model, "replicate": r, **A.tmrca_summary(d, label, mask)}
        row.update(state_index_mismatches=mismatches, decoded_model_is_e1_fit=same_model,
                   decoded_loglike=decoded.get("final_loglike"), decoded_loglike_is_e1_final=same_ll)
        summary.append(row)
    calibration = [{"model": model, "replicate": r, **row}
                   for by in ("true", "posterior_mean") for row in A.calibration_rows(d, by, cfg.tmrca_calibration_classes)]
    density = A.density_rows(d, cfg.tmrca_density_cells)
    chrom = cfg.tmrca_example_chrom if cfg.tmrca_example_chrom in names else names[0]
    k = names.index(chrom)
    length = int(truth["chromosomes"][k]["length"])
    start = min(cfg.tmrca_example_start_bp, max(length - cfg.tmrca_example_track_bp, 0)) // bin_size
    n_bins = (min(cfg.tmrca_example_track_bp, length) + bin_size - 1) // bin_size
    track_rows = A.track_rows(d, k, chrom, start, n_bins, bin_size)
    written = _write(lay, "s3", {"summary": summary, "calibration": calibration, "density": significant(density),
                                 "track": significant(track_rows)})
    problems = []
    if not (same_model and same_ll):
        problems.append("the decoding run did not use E1's fitted model")
    if mismatches:
        problems.append(f"{mismatches} bins have a MAP time outside their MAP state's interval")
    if problems:
        raise CheckFailed("S3: " + "; ".join(problems))
    return written


def report_s4(ctx: Context, records: Optional[Sequence[Path]] = None) -> Dict[str, Path]:
    """S4 tables from E3's runs; fails after writing them when outputs differ across thread counts.

    The runs are those of ``records`` (provenance records), else those found
    among E3's jobs.
    """
    from . import e6_analysis as A

    if records:
        groups = scaling_groups([run_from_record(r, ctx.runs_dir) for r in records])
    else:
        groups = thread_scaling(ctx)
    if not groups:
        raise MissingInput(f"no psmc-rs 0.2 runs of {ctx.profile.e6.thread_scaling_experiment} with several thread counts")
    _require([p for runs in groups.values() for r in runs for p in [r.output, r.record] + ([r.psmc_out] if r.psmc_out else [])])
    rows = []
    for inp, runs in groups.items():
        label = paths.portable(inp, ctx.runs_dir)
        seen: Dict[int, int] = {}
        for run in runs:
            seen[run.threads] = seen.get(run.threads, 0) + 1
            rows.append(A.run_row(
                run.threads, run.job_id, seen[run.threads], json.loads(run.record.read_text()),
                json.loads(run.output.read_text()),
                provenance.sha256_file(run.output), provenance.sha256_file(run.psmc_out) if run.psmc_out else None, label,
            ))
    determinism = A.determinism_rows(rows)
    written = _write(Layout(ctx.exp_dir(NAME)), "s4",
                     {"runs": rows, "scaling": A.scaling_rows(rows), "determinism": determinism})
    bad = [d["input"] for d in determinism if not d["json_identical"] or d["psmc_identical"] is False]
    if bad:
        raise CheckFailed(f"S4: outputs differ across thread counts for {', '.join(bad)}")
    return written


def report_s5(ctx: Context) -> Dict[str, Path]:
    """S5 tables from the QC reports of E2's genomes."""
    from . import e6_analysis as A

    cfg = ctx.profile.e6
    src = getattr(ctx.profile, cfg.qc_experiment.lower())
    lay = Layout(ctx.exp_dir(NAME))
    _require([lay.s5_qc(g.sample_id) for g in src.genomes])
    qc_rows, findings = [], []
    for g in src.genomes:
        qc = json.loads(lay.s5_qc(g.sample_id).read_text())
        qc_rows.append({**A.qc_row(g.sample_id, g.label, qc), "species": g.species, "file": f"experiment/real_data/{g.file}"})
        findings += A.finding_rows(g.sample_id, qc)
    return _write(lay, "s5", {"qc": qc_rows, "findings": findings})


REPORTS: Dict[str, Callable[[Context], Dict[str, Path]]] = {
    "s1": report_s1, "s2": report_s2, "s3": report_s3, "s4": report_s4, "s5": report_s5,
}


def report(ctx: Context, parts: Sequence[str] = PARTS) -> Dict[str, Path]:
    """Write the tables of every part in ``parts`` whose inputs exist; returns the paths written."""
    written: Dict[str, Path] = {}
    for part in parts:
        try:
            written.update(REPORTS[part](ctx))
        except MissingInput as exc:
            ctx.log(f"[E6] {part.upper()} not written: {exc}")
    return written


def figure(ctx: Context, part: str) -> Dict[str, Path]:
    """Draw the figure of ``part`` from its tables."""
    from ..figures import e6_supplementary

    lay = Layout(ctx.exp_dir(NAME))
    _require(lay.tables(part))
    return e6_supplementary.draw(part, {k: lay.table(part, k) for k in TABLES[part]}, lay.figures(part))


# ---------------------------------------------------------------------------
# Command line of the Python steps
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.experiments.e6",
                                description="E6's Python steps: the tables and figures of one part (S1-S5).")
    p.add_argument("action", choices=("report", "figure"))
    p.add_argument("--part", required=True, choices=PARTS)
    p.add_argument("--profile", default="production", choices=sorted(config.PROFILES))
    p.add_argument("--runs-dir", type=Path)
    p.add_argument("--record", type=Path, action="append", default=[],
                   help="S4: a provenance record of a psmc-rs 0.2 run to read (repeatable)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    runs_dir = (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()
    ctx = Context(config.get_profile(args.profile), runs_dir)
    if args.action == "figure":
        if args.part not in FIGURES:
            raise SystemExit(f"{args.part} has no figure")
        written = figure(ctx, args.part)
    elif args.part == "s4":
        written = report_s4(ctx, args.record)
    else:
        written = REPORTS[args.part](ctx)
    for key, path in written.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["FIGURES", "Layout", "NAME", "PARTS", "RsRun", "TABLES", "figure", "jobs", "parse_rs02_fit", "report",
           "rs02_run", "run_from_record", "scaling_groups", "thread_scaling"]
