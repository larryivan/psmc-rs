"""E3: performance (Fig 4, Table 2, Fig S4).

How fast and how lean are C psmc, psmc-rs 0.1 and psmc-rs 0.2 on the same
inputs, and does psmc-rs 0.2 give the same result whatever its thread count?
:class:`.config.E3` holds every setting. The runner

1. simulates one constant-model genome per size (100, 300, 1000 and
   3000 Mb as 1, 3, 10 and 30 chromosomes of 100 Mb; the chromosomes are the
   same for every size, so each genome is the first chromosomes of the next
   larger one);
2. fits every size with every tool, ``repeats`` times, with the inference
   recipe (``psmc -N25 -t15 -r5 -p "4+25*2+4+6"``): C psmc on its one
   thread, psmc-rs 0.1 (commit e50d69d, built out of tree) and psmc-rs 0.2
   at each of their thread counts; psmc-rs 0.2 also fits the 3000 Mb genome
   on 1, 2, 4 and 8 threads (thread scaling). Each run is an exclusive job
   (nothing else runs, on this runner or any other: see the runner's
   machine lock) timed by ``e3_run measure`` (wall time, CPU time, peak RSS
   of the tool's process tree, machine load before and after, CPU used by
   other processes during the run, power source; see :mod:`.e3_run`). A run
   that is not clean (the machine not quiet before it, more than
   ``timing.max_other_cpu_share`` of the cores used by others during it, a
   sleep, battery power) is not kept: the runner starts it again later.
   Jobs are ranked by repeat, so every setup's first repeat runs before any
   second one and drift over the night does not fall on one tool;
3. writes the tables, from clean runs only.

A run whose estimated peak memory (:class:`.config.ResourceModel`) exceeds
``memory_limit_gb`` is not run; Table 2 lists it as skipped, with the
estimate and the reason. With 100 Mb chromosomes no run is skipped: C psmc
holds the forward-backward matrices of one sequence at a time (about 1 GB
for 100 Mb), psmc-rs 0.1 at most about 4 GB (8 threads), psmc-rs 0.2 well
under 0.1 GB.

psmc-rs 0.1 has no ``--tr-ratio`` or ``--bin-size``: it gets the pattern,
``--n-steps``, ``--t-max``, ``--smooth-lambda 0``, ``--mstep-iters`` and the
number of iterations, and its default start (rho0 = theta0 / 5, the
recipe's ``-r5``; psmcfa bins of 100 bp). Both psmc-rs versions write their
HTML report next to the JSON, as in normal use; C psmc writes its ``.psmc``.

Layout under ``<runs>/E3/``::

    sims/<size>Mb/genome.psmcfa                 simulated genome (not committed)
    truth/<size>Mb.json                         seeds, counts, checksums, true Ne(t)
    fits/<tool>/<size>Mb/t<threads>/rep<r>.psmc|.json   each run's output (c: .psmc; rs01, rs02: .json)
    timings/<tool>/<size>Mb/t<threads>/rep<r>.json      each run's timing record (e3_run measure)
    provenance/...                              one record per job
    tables/E3_*.csv                             written by ``report E3``

Tables (``tables/``):

* ``E3_table2.csv`` (Table 2): one row per tool, thread count and size:
  status (measured, skipped with the reason, or incomplete), estimated peak
  memory, median, minimum and maximum wall time over the repeats, median
  CPU time and peak RSS, seconds per EM iteration, speed-up over C psmc and
  over psmc-rs 0.1 at the same thread count, C psmc's peak RSS over the
  tool's, whether every repeat ran on a quiet machine, and whether the
  repeats' outputs are byte-identical.
* ``E3_runs.csv``: one row per run: every measurement, the machine's state
  before and during it, the fitted theta, rho and t_max, and the SHA-256 of
  the output.
* ``E3_thread_scaling.csv`` (Fig S4): psmc-rs 0.2 on the thread-scaling
  genome: median wall time per thread count, speed-up over one thread,
  parallel efficiency, peak RSS, and whether the JSON is byte-identical to
  the one-thread JSON.
* ``E3_determinism.csv`` (Fig S4): every output compared with its tool's
  reference at the same size (fewest threads, first repeat): byte-identical
  or not, and the largest relative difference in theta, rho, t_max and the
  lambdas (and psmc-rs 0.2's final log-likelihood).
* ``E3_environment.csv``: the machine, the tools' identities, the timing
  protocol and the memory limit.
* ``E3_seeds.csv``: every simulation seed.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import config, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import write_csv
from .e3_run import genome_for, measure_args

NAME = "E3"
TOOL_ORDER = ("c", "rs01", "rs02", "rs02-default")
TOOL_LABELS = {"c": "C psmc", "rs01": "psmc-rs 0.1", "rs02": "psmc-rs 0.2",
               "rs02-default": "psmc-rs 0.2, default M-step budget"}
TOOL_NAMES = {"c": "c-psmc", "rs01": "psmc-rs-0.1", "rs02": "psmc-rs-0.2", "rs02-default": "psmc-rs-0.2"}  # tools.Tool.name
RS02 = ("rs02", "rs02-default")
TABLES = {
    "table2": "E3_table2.csv",
    "runs": "E3_runs.csv",
    "scaling": "E3_thread_scaling.csv",
    "determinism": "E3_determinism.csv",
    "environment": "E3_environment.csv",
    "seeds": "E3_seeds.csv",
}
C_PSMC_BUILD = "lh3/psmc 0.6.5-r74 built with -O1 (an -O2 build segfaults on macOS arm64)"


# ---------------------------------------------------------------------------
# What runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """One tool on one genome size with one thread count (repeated ``E3.repeats`` times)."""

    tool: str  # "c", "rs01", "rs02" or "rs02-default"
    size_mb: int
    threads: int

    def key(self) -> str:
        return f"{self.tool}/{self.size_mb}Mb/t{self.threads}"


@dataclass(frozen=True)
class Run:
    """One timed run: a setup and its repeat (from 1)."""

    setup: Setup
    repeat: int

    def key(self) -> str:
        return f"{self.setup.key()}/rep{self.repeat}"


@dataclass(frozen=True)
class Planned:
    """A setup with its estimated peak memory and run time; ``skip`` says why it does not run."""

    setup: Setup
    est_gb: float
    est_seconds: float
    skip: Optional[str] = None


def sizes(cfg: config.E3) -> List[int]:
    """Every genome size to simulate: the sizes and the thread-scaling size."""
    return sorted(set(cfg.genome_sizes_mb) | {cfg.thread_scaling_size_mb})


def setups(cfg: config.E3) -> List[Setup]:
    """Every setup of the experiment, in table order (tool, threads, size)."""
    out = {Setup("c", s, 1) for s in cfg.genome_sizes_mb}
    out |= {Setup("rs01", s, t) for s in cfg.genome_sizes_mb for t in cfg.rs01_threads}
    out |= {Setup("rs02", s, t) for s in cfg.genome_sizes_mb for t in cfg.rs02_threads}
    out |= {Setup("rs02", cfg.thread_scaling_size_mb, t) for t in cfg.thread_scaling_threads}
    if cfg.default_mstep_iters != cfg.inference.mstep_iters:
        out |= {Setup("rs02-default", s, t) for s in cfg.genome_sizes_mb for t in cfg.rs02_default_threads}
    return sorted(out, key=lambda u: (TOOL_ORDER.index(u.tool), u.threads, u.size_mb))


def inference_of(cfg: config.E3, tool: str) -> config.Inference:
    """The inference settings a tool runs with: the experiments' recipe, or psmc-rs's default M-step budget."""
    return replace(cfg.inference, mstep_iters=cfg.default_mstep_iters) if tool == "rs02-default" else cfg.inference


def in_flight(cfg: config.E3, size_mb: int, threads: int) -> int:
    """How many sequences psmc-rs works on at once: one per thread, at most every sequence."""
    return max(1, min(threads, genome_for(cfg, size_mb).n_chromosomes))


def rs01_caches(cfg: config.E3, setup: Setup) -> Optional[int]:
    """How many sequences psmc-rs 0.1 caches: None when it runs them in parallel (no cache)."""
    genome = genome_for(cfg, setup.size_mb)
    if setup.threads > 1 and genome.n_chromosomes > 1:
        return None
    row = cfg.resources.rs01_mb_per_mb_cache * genome.chromosome_length / 1e6
    return min(genome.n_chromosomes, int(cfg.resources.rs01_cache_mb // row)) if row > 0 else genome.n_chromosomes


def estimate_mb(cfg: config.E3, setup: Setup) -> float:
    """Estimated peak RSS of a setup in MB (2**20 bytes), from :class:`.config.ResourceModel`."""
    m = cfg.resources
    genome = genome_for(cfg, setup.size_mb)
    chrom_mb = genome.chromosome_length / 1e6
    if setup.tool == "c":
        return m.c_mb_base + m.c_mb_per_mb_longest * chrom_mb
    if setup.tool == "rs01":
        base = m.rs01_mb_base + m.rs01_mb_per_mb_input * setup.size_mb
        cached = rs01_caches(cfg, setup)
        if cached is not None:
            return base + cached * m.rs01_mb_per_mb_cache * chrom_mb + m.rs01_mb_per_mb_buffer * chrom_mb
        return base + in_flight(cfg, setup.size_mb, setup.threads) * m.rs01_mb_per_mb_buffer * chrom_mb
    if setup.tool in RS02:
        extra = in_flight(cfg, setup.size_mb, setup.threads) - 1
        return m.rs02_mb_base + m.rs02_mb_per_mb_input * setup.size_mb + m.rs02_mb_per_extra_thread * extra
    raise ValueError(f"unknown tool {setup.tool!r}")


def estimate_seconds(cfg: config.E3, setup: Setup) -> float:
    """Estimated wall time of one run (scheduling and ETA only), waiting before it included."""
    m = cfg.resources
    work = setup.size_mb * cfg.inference.n_iter
    if setup.tool == "c":
        run = m.c_s_per_mb_iter * work
    else:
        per = m.rs01_s_per_mb_iter if setup.tool == "rs01" else m.rs02_s_per_mb_iter
        speedup = 1.0 + (in_flight(cfg, setup.size_mb, setup.threads) - 1) * m.parallel_efficiency
        run = per * work / speedup
        run += m.mstep_s_per_iter_1000 * cfg.inference.n_iter * inference_of(cfg, setup.tool).mstep_iters / 1000.0
        cached = rs01_caches(cfg, setup) if setup.tool == "rs01" else None
        if cached is not None:
            n = genome_for(cfg, setup.size_mb).n_chromosomes
            run *= 1.0 + m.rs01_uncached_extra * (n - cached) / n
    return run + cfg.timing.settle_seconds + cfg.timing.quiet_window_seconds + 1.0


SKIP_WHY = {
    "c": "C psmc holds the forward and backward matrices of its longest sequence ({chrom:g} Mb)",
    "rs01": "psmc-rs 0.1 caches forward variables and holds one forward buffer per sequence in work ({chrom:g} Mb each)",
    "rs02": "psmc-rs 0.2 holds the input and a buffer per thread",
    "rs02-default": "psmc-rs 0.2 holds the input and a buffer per thread",
}


def plan(cfg: config.E3) -> List[Planned]:
    """Every setup with its estimates; those above the memory limit are skipped, with the reason."""
    out = []
    for u in setups(cfg):
        gb = estimate_mb(cfg, u) / 1024.0
        skip = None
        if gb > cfg.memory_limit_gb:
            chrom = cfg.chromosome_length / 1e6
            skip = (f"not run: estimated peak memory {gb:.1f} GB exceeds the {cfg.memory_limit_gb:g} GB limit; "
                    + SKIP_WHY[u.tool].format(chrom=chrom))
        out.append(Planned(u, gb, estimate_seconds(cfg, u), skip))
    return out


def runs(cfg: config.E3) -> List[Run]:
    """Every run that is not skipped."""
    return [Run(p.setup, r) for p in plan(cfg) if p.skip is None for r in range(1, cfg.repeats + 1)]


# ---------------------------------------------------------------------------
# Layout and jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where E3's files are."""

    root: Path

    def sim_dir(self, size_mb: int) -> Path:
        return self.root / "sims" / f"{size_mb}Mb"

    def psmcfa(self, size_mb: int) -> Path:
        return self.sim_dir(size_mb) / simulate.PSMCFA_NAME

    def truth(self, size_mb: int) -> Path:
        return self.root / "truth" / f"{size_mb}Mb.json"

    def _run_path(self, top: str, run: Run, ext: str) -> Path:
        u = run.setup
        return self.root / top / u.tool / f"{u.size_mb}Mb" / f"t{u.threads}" / f"rep{run.repeat}{ext}"

    def fit(self, run: Run) -> Path:
        return self._run_path("fits", run, ".psmc" if run.setup.tool == "c" else ".json")

    def timing(self, run: Run) -> Path:
        return self._run_path("timings", run, ".json")

    def record(self, run: Run) -> Path:
        return self._run_path("provenance/run", run, ".json")

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def run_job_id(run: Run) -> str:
    return f"{NAME}/run/{run.key()}"


def rs01_tool(ctx: Context) -> tools.Tool:
    """psmc-rs 0.1, built (when the context builds tools) from ``E3.rs01_commit`` into the tools directory.

    Without a build, a missing binary gives a stand-in identity, so that
    ``plan``, ``list`` and ``status`` work before the first ``start``.
    """
    if "rs01" not in ctx._tools:
        try:
            ctx._tools["rs01"] = tools.psmc_rs_01(build=ctx.build_tools, commit=ctx.profile.e3.rs01_commit, log=ctx.log)
        except tools.ToolError as exc:
            if ctx.build_tools:
                raise
            binary = tools.rs01_dirs()["target"] / "release" / "psmc-rs"
            ctx._tools["rs01"] = tools.Tool("psmc-rs-0.1", binary, {"version": "not built", "error": str(exc)})
    return ctx._tools["rs01"]


def rs01_fit(tool: tools.Tool, inference: config.Inference, scaling: config.Scaling, inp: Path, out_json: Path,
             *, threads: int) -> List[str]:
    """``psmc-rs IN OUT.json N_ITER`` for psmc-rs 0.1 with the recipe as far as its options go."""
    return [
        str(tool.path), str(inp), str(out_json), str(inference.n_iter),
        "--pattern", inference.pattern,
        "--n-steps", str(results.pattern_intervals(inference.pattern) - 1),
        "--t-max", tools.fmt(inference.t_max),
        "--smooth-lambda", tools.fmt(inference.smooth_lambda),
        "--mstep-iters", str(inference.mstep_iters),
        "--mu", tools.fmt(scaling.mu),
        "--threads", str(threads),
        "--no-progress",
    ]


def tool_command(ctx: Context, cfg: config.E3, lay: Layout, run: Run) -> Tuple[tools.Tool, List[str]]:
    """The tool and its exact command line for one run."""
    u = run.setup
    inp, out = lay.psmcfa(u.size_mb), lay.fit(run)
    if u.tool == "c":
        tool = ctx.c_psmc()
        return tool, tools.c_psmc_fit(tool, cfg.inference, inp, out)
    if u.tool == "rs01":
        tool = rs01_tool(ctx)
        return tool, rs01_fit(tool, cfg.inference, cfg.scaling, inp, out, threads=u.threads)
    tool = ctx.rs02()
    return tool, tools.rs02_fit(tool, inference_of(cfg, u.tool), cfg.scaling, inp, out, threads=u.threads)


def measure_command(ctx: Context, cfg: config.E3, lay: Layout, run: Run, tool_cmd: Sequence[str]) -> List[str]:
    return [
        paths.python(), "-m", "experiment.pipeline.experiments.e3_run", "measure",
        "--out", str(lay.timing(run)), "--runs-dir", str(ctx.runs_dir), *measure_args(cfg.timing),
        "--", *tool_cmd,
    ]


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e3
    costs = ctx.profile.costs
    budget = config.Budget()
    lay = Layout(ctx.exp_dir(NAME))
    out: List[Job] = []
    sim_identity = {
        "pipeline": provenance.python_identity(
            ["simulate.py", "demography.py", "experiments/e3_run.py"], ["msprime", "tskit", "numpy"]
        ),
    }
    for size in sizes(cfg):
        genome = genome_for(cfg, size)
        out.append(
            Job(
                id=f"{NAME}/sim/{size}Mb",
                command=[
                    paths.python(), "-m", "experiment.pipeline.experiments.e3_run", "simulate",
                    "--profile", ctx.profile.name, "--size-mb", str(size),
                    "--out-dir", str(lay.sim_dir(size)), "--truth", str(lay.truth(size)),
                ],
                outputs=[lay.psmcfa(size), lay.truth(size)],
                mem_gb=costs.simulate_gb,
                est_seconds=costs.simulate_s_base + costs.simulate_s_per_mb * size,
                settings={
                    "model": asdict(config.MODELS[cfg.model]),
                    "genome": asdict(genome),
                    "simulation": asdict(cfg.simulation),
                    "replicate": cfg.replicate,
                },
                tools=sim_identity,
                seeds={c: simulate.seeds_for(NAME, cfg.model, cfg.replicate, c) for c in genome.names()},
                description=f"simulate the {size} Mb genome",
            )
        )
    timing_identity = provenance.python_identity(["experiments/e3_run.py"], ["psutil"])
    estimates = {p.setup: p for p in plan(cfg)}
    report_inputs: List[Path] = [lay.truth(s) for s in sizes(cfg)]
    for run in runs(cfg):
        u = run.setup
        tool, cmd = tool_command(ctx, cfg, lay, run)
        settings = {
            "tool": u.tool, "size_mb": u.size_mb, "threads": u.threads, "repeat": run.repeat,
            "inference": asdict(inference_of(cfg, u.tool)), "timing": asdict(cfg.timing),
        }
        if u.tool != "c":
            settings["scaling"] = asdict(cfg.scaling)
        est = estimates[u]
        out.append(
            Job(
                id=run_job_id(run),
                command=measure_command(ctx, cfg, lay, run, cmd),
                inputs=[lay.psmcfa(u.size_mb)],
                outputs=[lay.fit(run), lay.timing(run)],
                # Exclusive jobs run alone whatever their cores; capping them at
                # the default budget keeps 8-thread runs startable.
                cores=min(u.threads, budget.cores),
                mem_gb=round(max(est.est_gb, 0.05) + 0.005, 2),
                exclusive=True,
                est_seconds=est.est_seconds,
                settings=settings,
                tools={tool.name: tool.identity, "timing": timing_identity},
                env={"RAYON_NUM_THREADS": str(u.threads)},
                description=f"{TOOL_LABELS[u.tool]} on {u.size_mb} Mb, {u.threads} thread(s), repeat {run.repeat}",
                rank=run.repeat,
            )
        )
        report_inputs += [lay.fit(run), lay.timing(run), lay.record(run)]
    out.append(
        Job(
            id=f"{NAME}/report",
            command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
            inputs=report_inputs,
            outputs=[lay.table(k) for k in TABLES],
            mem_gb=costs.report_gb,
            est_seconds=costs.report_s,
            settings={"e3": asdict(cfg)},
            tools={"pipeline": provenance.python_identity(
                ["experiments/e3.py", "experiments/e3_run.py", "experiments/e1.py", "results.py"], []
            )},
            description="E3 tables",
        )
    )
    return out


# ---------------------------------------------------------------------------
# Reading the runs back
# ---------------------------------------------------------------------------


@dataclass
class Output:
    """The fitted model of one run's output, and its checksum."""

    sha256: str
    theta: float
    rho: float
    t_max: float
    lambdas: Tuple[float, ...]
    loglike: Optional[float]  # psmc-rs 0.2: after the last iteration; C psmc: LK of its last round
    bins: Optional[int]
    het_bins: Optional[int]


def read_output(path: Path, tool: str, n_iter: int) -> Output:
    """The model a run wrote, checked to be a complete fit of ``n_iter`` iterations where the format says."""
    digest = provenance.sha256_file(path)
    if tool == "c":
        pf = results.parse_psmc(path)
        last = pf.last()
        if last.round != n_iter:
            raise ValueError(f"{path}: last complete round {last.round}, expected {n_iter}")
        fit = last.fit()
        return Output(digest, fit.theta, fit.rho, fit.t_max, fit.free_lambdas, last.lk,
                      pf.header.get("sum_L"), pf.header.get("sum_n"))
    params = results.read_rs_json(path)
    fit = results.fit_from_rs(params)
    if tool in RS02 and int(params.get("n_iter", -1)) != n_iter:
        raise ValueError(f"{path}: n_iter {params.get('n_iter')}, expected {n_iter}")
    inp = params.get("input", {})
    loglike = params.get("final_loglike")
    return Output(digest, fit.theta, fit.rho, fit.t_max, fit.free_lambdas,
                  float(loglike) if loglike is not None else None, inp.get("bins"), inp.get("het_bins"))


def max_rel_diff(a: Output, b: Output) -> float:
    """The largest relative difference between two fits' theta, rho, t_max and lambdas."""
    if len(a.lambdas) != len(b.lambdas):
        return math.inf
    pairs = [(a.theta, b.theta), (a.rho, b.rho), (a.t_max, b.t_max), *zip(a.lambdas, b.lambdas)]
    return max(abs(x - y) / abs(y) if y else abs(x - y) for x, y in pairs)


def _median(xs: Iterable[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return a / b if a is not None and b else None


def run_row(lay: Layout, cfg: config.E3, run: Run, truth: Dict, out: Output, timing: Dict) -> Dict:
    u = run.setup
    bins = sum(c["bins"] for c in truth["chromosomes"])
    het_bins = sum(c["het_bins"] for c in truth["chromosomes"])
    if out.bins is not None and (out.bins, out.het_bins) != (bins, het_bins):
        raise ValueError(
            f"{lay.fit(run)}: read {out.bins} bins ({out.het_bins} heterozygous), "
            f"the simulated genome has {bins} ({het_bins})"
        )
    before = timing.get("machine_before", {})
    during = timing.get("during", {})
    power = before.get("power") or {}
    return {
        "tool": u.tool,
        "tool_label": TOOL_LABELS[u.tool],
        "size_mb": u.size_mb,
        "chromosomes": len(truth["chromosomes"]),
        "bins": bins,
        "het_bins": het_bins,
        "threads": u.threads,
        "repeat": run.repeat,
        "wall_seconds": timing["wall_seconds"],
        "user_seconds": timing["user_seconds"],
        "system_seconds": timing["system_seconds"],
        "cpu_seconds": timing["cpu_seconds"],
        "cpu_per_wall": timing["cpu_seconds"] / timing["wall_seconds"] if timing["wall_seconds"] else None,
        "seconds_per_iteration": timing["wall_seconds"] / cfg.inference.n_iter,
        "peak_rss_mb": timing["peak_rss_mb"],
        "maxrss_mb": timing["maxrss_mb"],
        "peak_rss_tree_sampled_mb": timing["peak_rss_tree_sampled_mb"],
        "clean": is_clean(timing, cfg.timing),
        "quiet_before": before.get("quiet"),
        "busy_before_percent": before.get("busy_percent"),
        "waited_seconds": before.get("waited_seconds"),
        "other_cpu_share_during": during.get("other_cpu_share"),
        "clock_span_seconds": timing.get("clock_span_seconds"),
        "power_source": power.get("source"),
        "low_power_mode": power.get("low_power_mode"),
        "loadavg_before_1m": (timing.get("loadavg_before") or [None])[0],
        "loadavg_after_1m": (timing.get("loadavg_after") or [None])[0],
        "start": timing.get("start"),
        "end": timing.get("end"),
        "theta": out.theta,
        "rho": out.rho,
        "t_max": out.t_max,
        "loglike": out.loglike,
        "output_sha256": out.sha256,
    }


def is_clean(timing: Dict, protocol: config.TimingProtocol) -> bool:
    """Whether a timing record is a clean measurement by ``protocol`` (``measure`` keeps only those).

    Checked again here, so that a record written under an older or laxer
    protocol never enters the medians.
    """
    before, during = timing.get("machine_before", {}), timing.get("during", {})
    power = before.get("power") or {}
    span = timing.get("clock_span_seconds")
    return bool(
        before.get("quiet")
        and (during.get("other_cpu_share") or 0.0) <= protocol.max_other_cpu_share
        and (span is None or span - timing["wall_seconds"] <= protocol.max_sleep_seconds)
        and not (protocol.require_ac_power and (power.get("source") not in (None, "AC Power") or power.get("low_power_mode")))
    )


def table2_rows(cfg: config.E3, planned: Sequence[Planned], rows: Dict[Run, Dict]) -> List[Dict]:
    by_setup: Dict[Setup, List[Dict]] = {}
    measured: Dict[Setup, int] = {}
    for run, row in rows.items():
        measured[run.setup] = measured.get(run.setup, 0) + 1
        if row["clean"]:
            by_setup.setdefault(run.setup, []).append(row)
    medians = {u: _median(r["wall_seconds"] for r in rs) for u, rs in by_setup.items()}
    rss = {u: _median(r["peak_rss_mb"] for r in rs) for u, rs in by_setup.items()}
    out = []
    for p in planned:
        u = p.setup
        rs = sorted(by_setup.get(u, []), key=lambda r: r["repeat"])
        walls = [r["wall_seconds"] for r in rs]
        if p.skip:
            status, note = "skipped", p.skip
        elif len(rs) < cfg.repeats:
            status, note = "incomplete", f"{len(rs)} of {cfg.repeats} repeats measured cleanly"
        else:
            status, note = "measured", ""
        c_setup = Setup("c", u.size_mb, 1)
        rs01_setup = Setup("rs01", u.size_mb, u.threads)
        med = medians.get(u)
        out.append(
            {
                "tool": u.tool,
                "tool_label": TOOL_LABELS[u.tool],
                "threads": u.threads,
                "size_mb": u.size_mb,
                "chromosomes": genome_for(cfg, u.size_mb).n_chromosomes,
                "status": status,
                "note": note,
                "est_peak_gb": round(p.est_gb, 3),
                "repeats": len(rs),
                "repeats_measured": measured.get(u, 0),
                "wall_median_s": med,
                "wall_min_s": min(walls) if walls else None,
                "wall_max_s": max(walls) if walls else None,
                "cpu_median_s": _median(r["cpu_seconds"] for r in rs),
                "seconds_per_iteration_median": med / cfg.inference.n_iter if med is not None else None,
                "peak_rss_median_mb": rss.get(u),
                "peak_rss_max_mb": max((r["peak_rss_mb"] for r in rs), default=None),
                "speedup_vs_c": _ratio(medians.get(c_setup), med) if u.tool != "c" else None,
                "speedup_vs_rs01_same_threads": _ratio(medians.get(rs01_setup), med) if u.tool in RS02 else None,
                "mstep_iters": inference_of(cfg, u.tool).mstep_iters if u.tool != "c" else None,
                "c_rss_over_rss": _ratio(rss.get(c_setup), rss.get(u)) if u.tool != "c" else None,
                "all_quiet": all(bool(r["quiet_before"]) for r in rs) if rs else None,
                "max_other_cpu_share": max((r["other_cpu_share_during"] or 0.0 for r in rs), default=None),
                "outputs_identical": len({r["output_sha256"] for r in rs}) == 1 if rs else None,
            }
        )
    return out


def scaling_rows(cfg: config.E3, rows: Dict[Run, Dict], outputs: Dict[Run, Output]) -> List[Dict]:
    size = cfg.thread_scaling_size_mb
    threads = sorted(cfg.thread_scaling_threads)
    by_t = {t: sorted((r for run, r in rows.items() if run.setup == Setup("rs02", size, t) and r["clean"]),
                      key=lambda r: r["repeat"])
            for t in threads}
    base_t = threads[0]
    base = _median(r["wall_seconds"] for r in by_t[base_t])
    ref = outputs.get(Run(Setup("rs02", size, base_t), 1))
    out = []
    for t in threads:
        rs = by_t[t]
        med = _median(r["wall_seconds"] for r in rs)
        speedup = _ratio(base, med)
        mine = [o for run, o in outputs.items() if run.setup == Setup("rs02", size, t)]
        out.append(
            {
                "tool": "rs02",
                "size_mb": size,
                "threads": t,
                "repeats": len(rs),
                "wall_median_s": med,
                "wall_min_s": min((r["wall_seconds"] for r in rs), default=None),
                "wall_max_s": max((r["wall_seconds"] for r in rs), default=None),
                "speedup": speedup,
                "efficiency": speedup * base_t / t if speedup is not None else None,
                "cpu_median_s": _median(r["cpu_seconds"] for r in rs),
                "peak_rss_median_mb": _median(r["peak_rss_mb"] for r in rs),
                "all_quiet": all(bool(r["quiet_before"]) for r in rs) if rs else None,
                "json_identical_to_reference": (all(o.sha256 == ref.sha256 for o in mine) if (mine and ref) else None),
                "reference": f"{base_t} thread(s), repeat 1",
            }
        )
    return out


def determinism_rows(outputs: Dict[Run, Output]) -> List[Dict]:
    """Every output against its tool's reference at the same size: fewest threads, first repeat."""
    out = []
    groups: Dict[Tuple[str, int], List[Run]] = {}
    for run in outputs:
        groups.setdefault((run.setup.tool, run.setup.size_mb), []).append(run)
    for (tool, size), group in sorted(groups.items(), key=lambda kv: (TOOL_ORDER.index(kv[0][0]), kv[0][1])):
        group.sort(key=lambda r: (r.setup.threads, r.repeat))
        ref_run = group[0]
        ref = outputs[ref_run]
        for run in group:
            o = outputs[run]
            ll = (abs(o.loglike - ref.loglike) / abs(ref.loglike)
                  if o.loglike is not None and ref.loglike else None)
            out.append(
                {
                    "tool": tool,
                    "size_mb": size,
                    "threads": run.setup.threads,
                    "repeat": run.repeat,
                    "reference_threads": ref_run.setup.threads,
                    "reference_repeat": ref_run.repeat,
                    "byte_identical": o.sha256 == ref.sha256,
                    "max_rel_diff_params": max_rel_diff(o, ref),
                    "loglike_rel_diff": ll,
                    "sha256": o.sha256,
                }
            )
    return out


def environment_rows(cfg: config.E3, lay: Layout, rows: Dict[Run, Dict], timings: Dict[Run, Dict]) -> List[Dict]:
    out: List[Dict] = []

    def add(key: str, value) -> None:
        out.append({"key": key, "value": value if isinstance(value, str) else json.dumps(value, sort_keys=True)})

    first = next(iter(timings.values()), None)
    if first:
        for k, v in sorted(first.get("machine", {}).items()):
            add(f"machine.{k}", str(v))
    seen = set()
    for run in sorted(rows, key=lambda r: (TOOL_ORDER.index(r.setup.tool), r.setup.threads, r.setup.size_mb, r.repeat)):
        if run.setup.tool in seen:
            continue
        try:
            rec = json.loads(lay.record(run).read_text())
        except (OSError, ValueError):
            continue
        seen.add(run.setup.tool)
        ident = rec.get("key", {}).get("tools", {}).get(TOOL_NAMES[run.setup.tool], {})
        for k, v in sorted(ident.items()):
            add(f"tool.{run.setup.tool}.{k}", str(v))
        command = rec.get("key", {}).get("command", [])
        tool_path = command[command.index("--") + 1] if "--" in command else None
        exes = rec.get("run", {}).get("executables", {})
        if tool_path in exes:
            add(f"tool.{run.setup.tool}.binary_sha256", exes[tool_path])
    add("tool.c.build", C_PSMC_BUILD)
    add("cc", c_compiler())
    add("tool.rs01.settings", "pattern, --n-steps, --t-max, --smooth-lambda 0, --mstep-iters and N_ITER of the recipe; "
        "its default start rho0 = theta0/5 (= -r5) and 100 bp bins (it has no --tr-ratio or --bin-size)")
    add("inference", asdict(cfg.inference))
    add("timing_protocol", asdict(cfg.timing))
    add("memory_limit_gb", f"{cfg.memory_limit_gb:g}")
    add("repeats", str(cfg.repeats))
    add("runs_not_quiet", str(sum(1 for r in rows.values() if not r["quiet_before"])))
    add("runs_not_clean", str(sum(1 for r in rows.values() if not r["clean"])))
    add("power_sources", ";".join(sorted({str(r["power_source"]) for r in rows.values()})))
    add("low_power_mode_seen", str(any(r["low_power_mode"] for r in rows.values())))
    return out


def c_compiler() -> str:
    """``cc --version`` (first line): the compiler C psmc was built with, if it is still this machine's."""
    import subprocess

    try:
        res = subprocess.run(["cc", "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return res.stdout.splitlines()[0].strip() if res.returncode == 0 and res.stdout else "unavailable"


def seed_rows(cfg: config.E3, truths: Dict[int, Dict]) -> List[Dict]:
    out = []
    for size in sorted(truths):
        for ch in truths[size]["chromosomes"]:
            out.append(
                {"size_mb": size, "chromosome": ch["name"], "ancestry_seed": ch["seeds"]["ancestry"],
                 "mutation_seed": ch["seeds"]["mutations"], "het_sites": ch["het_sites"], "het_bins": ch["het_bins"],
                 "trees": ch.get("trees"), "longest_het_gap_bp": ch.get("longest_het_gap_bp")}
            )
    return out


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E3 table from the runs measured so far; returns the paths written.

    Setups with fewer than ``repeats`` measured runs are reported as
    incomplete, so the tables can be looked at while the runs go on; the
    report job of the runner runs only when every run is done.
    """
    cfg = ctx.profile.e3
    lay = Layout(ctx.exp_dir(NAME))
    planned = plan(cfg)
    truths = {s: json.loads(lay.truth(s).read_text()) for s in sizes(cfg) if lay.truth(s).exists()}
    rows: Dict[Run, Dict] = {}
    outputs: Dict[Run, Output] = {}
    timings: Dict[Run, Dict] = {}
    for run in runs(cfg):
        if not (lay.timing(run).exists() and lay.fit(run).exists() and run.setup.size_mb in truths):
            continue
        timing = json.loads(lay.timing(run).read_text())
        out = read_output(lay.fit(run), run.setup.tool, cfg.inference.n_iter)
        rows[run] = run_row(lay, cfg, run, truths[run.setup.size_mb], out, timing)
        outputs[run] = out
        timings[run] = timing
    ordered = [rows[r] for r in sorted(rows, key=lambda r: (TOOL_ORDER.index(r.setup.tool), r.setup.threads,
                                                             r.setup.size_mb, r.repeat))]
    tables = {
        "table2": table2_rows(cfg, planned, rows),
        "runs": ordered,
        "scaling": scaling_rows(cfg, rows, outputs),
        "determinism": determinism_rows(outputs),
        "environment": environment_rows(cfg, lay, rows, timings),
        "seeds": seed_rows(cfg, truths),
    }
    written = {}
    for key, table in tables.items():
        write_csv(lay.table(key), table)
        written[key] = lay.table(key)
    return written


__all__ = [
    "Layout", "NAME", "Planned", "Run", "Setup", "TABLES", "estimate_mb", "estimate_seconds", "jobs", "plan",
    "report", "runs", "setups", "sizes",
]
