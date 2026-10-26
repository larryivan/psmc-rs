"""E3: performance (Fig 4, Table 2, Fig S4).

How fast and how lean are C psmc and psmc-rs 0.2 on the same inputs, and
does psmc-rs 0.2 give the same result whatever its thread count?
:class:`.config.E3` holds every setting. The runner

1. simulates one constant-model genome per size (100, 300, 1000 and
   3000 Mb as 1, 3, 10 and 30 chromosomes of 100 Mb; the chromosomes are the
   same for every size, so each genome is the first chromosomes of the next
   larger one);
2. fits every size with every tool, ``repeats`` times, with the inference
   recipe (``psmc -N25 -t15 -r5 -p "4+25*2+4+6"``): C psmc on its one
   thread and psmc-rs 0.2 at each of its thread counts (also with its
   M-steps capped at ``E3.capped_mstep_iters`` iterations,
   ``rs02-capped``, when that is set; by default it is not, since psmc-rs
   0.2's default M-step converges like every other run's); psmc-rs 0.2
   also fits the 3000 Mb genome on 1, 2, 4, 8, 16 and 30 threads (thread
   scaling; with one sequence per thread, its 30 chromosomes are the most
   threads can use). Each run is an
   exclusive job (nothing else runs, on this runner or any other: see the
   runner's machine lock) with as many cores as threads, at most the
   machine's (:func:`.e3_run.usable_cpus`), timed by ``e3_run measure`` (wall
   time, CPU time, peak RSS of the tool's process tree, machine load before
   and after, CPU used by other work during the run, power source). A run
   that is not clean (the whole machine not quiet before it; other work
   during it, as :mod:`.e3_run` defines it and each record's
   ``other_method`` says, above ``timing.max_other_cpu_share`` of the cores
   or ``timing.max_other_cores`` cores; a sleep; battery power) is not
   kept: the runner starts it again later. Jobs are ranked by repeat, so
   every setup's first repeat runs before any second one and drift over the
   night does not fall on one tool;
3. writes the tables, from clean runs only.

A run whose estimated peak memory (:class:`.config.ResourceModel`) exceeds
``memory_limit_gb`` is not run; Table 2 lists it as skipped, with the
estimate and the reason. With 100 Mb chromosomes no run is skipped: C psmc
holds the forward-backward matrices of one sequence at a time (about 1 GB
for 100 Mb), psmc-rs 0.2 well under 1 GB. psmc-rs writes its HTML report
next to the JSON, as in normal use; C psmc writes its ``.psmc``.

Layout under ``<runs>/E3/``::

    sims/<size>Mb/genome.psmcfa                 simulated genome (not committed)
    truth/<size>Mb.json                         seeds, counts, checksums, true Ne(t)
    fits/<tool>/<size>Mb/t<threads>/rep<r>.psmc|.json   each run's output (c: .psmc; rs02: .json)
    timings/<tool>/<size>Mb/t<threads>/rep<r>.json      each run's timing record (e3_run measure)
    provenance/...                              one record per job
    tables/E3_*.csv                             written by ``report E3``

Tables (``tables/``):

* ``E3_table2.csv`` (Table 2): one row per tool, thread count and size:
  status (measured, skipped with the reason, or incomplete), estimated peak
  memory, the clean repeats and every repeat measured, median, minimum and
  maximum wall time over the clean repeats, median CPU time and peak RSS,
  seconds per EM iteration, speed-up over C psmc, C psmc's peak RSS over
  the tool's, the M-step budget, and whether the repeats' outputs are
  byte-identical.
* ``E3_runs.csv``: one row per run: every measurement, the machine's state
  before and during it (before: busy cores, and of them user-space
  processes, kernel threads and interrupts; during: the other work the
  clean check limited, ``other_cores_during`` and ``other_cpu_share_during``
  by ``other_method``, and beside it the machine-wide measure, kernel
  threads, interrupts, steal, what none explains and the processes that
  ended during the run), the machine it ran on (CPU model, governor, boost,
  usable cores, host, SLURM job), the fitted theta, rho and t_max, and the
  SHA-256 of the output.
* ``E3_thread_scaling.csv`` (Fig S4): psmc-rs 0.2 on the thread-scaling
  genome: median wall time per thread count, speed-up over one thread,
  parallel efficiency, peak RSS, and whether the JSON is byte-identical to
  the one-thread JSON.
* ``E3_determinism.csv`` (Fig S4): every output compared with its tool's
  reference at the same size (fewest threads, first repeat): byte-identical
  or not, and the largest relative difference in theta, rho, t_max and the
  lambdas (and psmc-rs 0.2's final log-likelihood).
* ``E3_environment.csv``: the machine (each distinct value over the runs,
  ';'-joined: hosts, SLURM jobs), the tools' identities in the runs'
  records (C psmc's compiler and CFLAGS among them), the timing
  protocol, the memory limit, the measures of other work used and the
  median and maximum of each of E3_runs' CPU-use columns over the runs
  (``other_cores_during`` for each measure apart), and
  the kinds of machine (CPU model, governor, boost) the clean runs were
  timed on: when there are several (``clean_runs_on_mixed_machines``), the
  medians and speed-ups mix them, and the report warns.
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
from .e1 import read_c_fit, read_rs_fit, write_tables
from .e3_run import genome_for, machine_other_cores, measure_args, other_cores, other_method, other_share, usable_cpus

NAME = "E3"
TOOL_ORDER = ("c", "rs02", "rs02-capped")
TOOL_LABELS = {"c": "C psmc", "rs02": "psmc-rs 0.2", "rs02-capped": "psmc-rs 0.2, capped M-step"}
TOOL_NAMES = {"c": "c-psmc", "rs02": "psmc-rs-0.2", "rs02-capped": "psmc-rs-0.2"}  # tools.Tool.name
RS02 = ("rs02", "rs02-capped")
TABLES = {
    "table2": "E3_table2.csv",
    "runs": "E3_runs.csv",
    "scaling": "E3_thread_scaling.csv",
    "determinism": "E3_determinism.csv",
    "environment": "E3_environment.csv",
    "seeds": "E3_seeds.csv",
}


# ---------------------------------------------------------------------------
# What runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """One tool on one genome size with one thread count (repeated ``E3.repeats`` times)."""

    tool: str  # "c", "rs02" or "rs02-capped"
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
    out |= {Setup("rs02", s, t) for s in cfg.genome_sizes_mb for t in cfg.rs02_threads}
    out |= {Setup("rs02", cfg.thread_scaling_size_mb, t) for t in cfg.thread_scaling_threads}
    if cfg.capped_mstep_iters is not None and cfg.capped_mstep_iters != cfg.inference.mstep_iters:
        out |= {Setup("rs02-capped", s, t) for s in cfg.genome_sizes_mb for t in cfg.rs02_capped_threads}
    return sorted(out, key=lambda u: (TOOL_ORDER.index(u.tool), u.threads, u.size_mb))


def inference_of(cfg: config.E3, tool: str) -> config.Inference:
    """The inference settings a tool runs with: the experiments' recipe, or the capped M-steps of ``rs02-capped``."""
    return replace(cfg.inference, mstep_iters=cfg.capped_mstep_iters) if tool == "rs02-capped" else cfg.inference


def in_flight(cfg: config.E3, size_mb: int, threads: int) -> int:
    """How many sequences psmc-rs works on at once: one per thread, at most every sequence."""
    return max(1, min(threads, genome_for(cfg, size_mb).n_chromosomes))


def estimate_mb(cfg: config.E3, setup: Setup) -> float:
    """Estimated peak RSS of a setup in MB (2**20 bytes), from :class:`.config.ResourceModel`."""
    m = cfg.resources
    genome = genome_for(cfg, setup.size_mb)
    chrom_mb = genome.chromosome_length / 1e6
    if setup.tool == "c":
        return m.c_mb_base + m.c_mb_per_mb_longest * chrom_mb
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
        speedup = 1.0 + (in_flight(cfg, setup.size_mb, setup.threads) - 1) * m.parallel_efficiency
        run = m.rs02_s_per_mb_iter * work / speedup
        run += m.rs02_mstep_s_per_iter * cfg.inference.n_iter
    return run + cfg.timing.settle_seconds + cfg.timing.quiet_window_seconds + 1.0


SKIP_WHY = {
    "c": "C psmc holds the forward and backward matrices of its longest sequence ({chrom:g} Mb)",
    "rs02": "psmc-rs 0.2 holds the input and a buffer per thread",
    "rs02-capped": "psmc-rs 0.2 holds the input and a buffer per thread",
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


def table_order(run: Run) -> Tuple[int, int, int, int]:
    """The order of runs in the tables: tool, threads, size, repeat."""
    u = run.setup
    return TOOL_ORDER.index(u.tool), u.threads, u.size_mb, run.repeat


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


def tool_command(ctx: Context, cfg: config.E3, lay: Layout, run: Run) -> Tuple[tools.Tool, List[str]]:
    """The tool and its exact command line for one run."""
    u = run.setup
    inp, out = lay.psmcfa(u.size_mb), lay.fit(run)
    if u.tool == "c":
        tool = ctx.c_psmc()
        return tool, tools.c_psmc_fit(tool, cfg.inference, inp, out)
    tool = ctx.rs02()
    return tool, tools.rs02_fit(tool, inference_of(cfg, u.tool), cfg.scaling, inp, out, threads=u.threads)


def measure_command(ctx: Context, cfg: config.E3, lay: Layout, run: Run, tool_cmd: Sequence[str]) -> List[str]:
    return [
        paths.python(), "-m", "experiment.pipeline.experiments.e3_run", "measure",
        "--out", str(lay.timing(run)), "--runs-dir", str(ctx.runs_dir), "--threads", str(run.setup.threads),
        *measure_args(cfg.timing), "--", *tool_cmd,
    ]


def run_cores(threads: int, machine: int) -> int:
    """The cores of a timing job: its threads, at most the machine's.

    A timing job is exclusive: it runs alone, so it needs only fit the
    machine, not a share of it, and the runner starts it whatever its
    ``--cores`` budget (:func:`.runner.fits_budget`). Its memory must fit
    ``--mem-gb``: the default budget fits every run (C psmc's, the largest,
    need about 1 GB). On fewer cores than threads, ``measure
    --threads`` refuses to time the run (it would be oversubscribed), and
    the job fails until a runner runs it on a machine or allocation of
    enough cores.
    """
    return max(1, min(threads, machine))


def jobs(ctx: Context) -> List[Job]:
    cfg, costs = ctx.profile.e3, ctx.profile.costs
    cores = usable_cpus()
    lay = Layout(ctx.exp_dir(NAME))
    out: List[Job] = []
    sim_identity = {
        "pipeline": provenance.python_identity(
            ["simulate.py", "demography.py", "experiments/e3_run.py"], ["msprime", "tskit", "numpy"]
        ),
    }
    for size in sizes(cfg):
        genome = genome_for(cfg, size)
        out.append(Job(
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
        ))
    timing_identity = provenance.python_identity(["experiments/e3_run.py"], ["psutil"])
    estimates = {p.setup: p for p in plan(cfg)}
    report_inputs: List[Path] = [lay.truth(s) for s in sizes(cfg)]
    for run in runs(cfg):
        u, est = run.setup, estimates[run.setup]
        tool, cmd = tool_command(ctx, cfg, lay, run)
        settings = {
            "tool": u.tool, "size_mb": u.size_mb, "threads": u.threads, "repeat": run.repeat,
            "inference": asdict(inference_of(cfg, u.tool)), "timing": asdict(cfg.timing),
        }
        if u.tool != "c":
            settings["scaling"] = asdict(cfg.scaling)
        out.append(Job(
            id=run_job_id(run),
            command=measure_command(ctx, cfg, lay, run, cmd),
            inputs=[lay.psmcfa(u.size_mb)],
            outputs=[lay.fit(run), lay.timing(run)],
            cores=run_cores(u.threads, cores),
            mem_gb=round(max(est.est_gb, 0.05) + 0.005, 2),
            exclusive=True,
            est_seconds=est.est_seconds,
            settings=settings,
            tools={tool.name: tool.identity, "timing": timing_identity},
            env={"RAYON_NUM_THREADS": str(u.threads)},
            description=f"{TOOL_LABELS[u.tool]} on {u.size_mb} Mb, {u.threads} thread(s), repeat {run.repeat}",
            rank=run.repeat,
        ))
        report_inputs += [lay.fit(run), lay.timing(run), lay.record(run)]
    out.append(Job(
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
    ))
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
        pf = read_c_fit(path, n_iter)
        last = pf.last()
        fit = last.fit()
        return Output(digest, fit.theta, fit.rho, fit.t_max, fit.free_lambdas, last.lk,
                      pf.header.get("sum_L"), pf.header.get("sum_n"))
    params = read_rs_fit(path, n_iter)
    fit = results.fit_from_rs(params)
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
    machine = timing.get("machine", {})
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
        "busy_before_cores": before.get("busy_cores"),
        "waited_seconds": before.get("waited_seconds"),
        "process_cores_before": before.get("process_cores"),
        "kernel_thread_cores_before": before.get("kernel_thread_cores"),
        "interrupt_cores_before": before.get("interrupt_cores"),
        # the other work the clean check limits: other user-space processes
        # (other_method "processes") or the machine's busy CPU but the tool's ("machine")
        "other_method": other_method(timing),
        "other_cpu_share_during": other_share(timing),
        "other_cores_during": other_cores(timing),
        "machine_other_cores_during": machine_other_cores(timing),
        "machine_other_cpu_share_during": during.get("other_cpu_share"),
        "kernel_thread_cores_during": during.get("kernel_thread_cores"),
        "interrupt_cores_during": during.get("interrupt_cores"),
        "steal_cores_during": during.get("steal_cores"),
        "unaccounted_cores_during": during.get("unaccounted_cores"),
        "other_processes_exited": during.get("other_processes_exited"),
        "clock_span_seconds": timing.get("clock_span_seconds"),
        "power_source": power.get("source"),
        "low_power_mode": power.get("low_power_mode"),
        "cpu": machine.get("cpu"),
        "cpu_governor": machine.get("cpu_governor"),
        "cpu_boost": machine.get("cpu_boost"),
        "cpus_usable": timing.get("cpus_usable", machine.get("cpus_allowed")),
        "hostname": machine.get("hostname"),
        "slurm_job_id": machine.get("slurm_job_id"),
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
    protocol never enters the medians; a run with more threads than usable
    CPUs (:func:`oversubscribed`) is not clean either. The other work during
    the run is the record's own measure (:func:`.e3_run.other_cores`, by
    ``during.other_method``): other user-space processes in records that
    have it (with steal time), else the machine's busy CPU but the tool's
    (``timing/3``).
    """
    before = timing.get("machine_before", {})
    power = before.get("power") or {}
    span = timing.get("clock_span_seconds")
    return bool(
        before.get("quiet")
        and (before.get("busy_cores") or 0.0) <= protocol.quiet_max_cores
        and other_share(timing) <= protocol.max_other_cpu_share
        and other_cores(timing) <= protocol.max_other_cores
        and not oversubscribed(timing)
        and (span is None or span - timing["wall_seconds"] <= protocol.max_sleep_seconds)
        and not (protocol.require_ac_power and (power.get("source") not in (None, "AC Power") or power.get("low_power_mode")))
    )


def oversubscribed(timing: Dict) -> bool:
    """Whether a run had more threads than the CPUs it could use (``measure --threads`` refuses such runs)."""
    threads = timing.get("threads")
    cpus = timing.get("cpus_usable", timing.get("machine", {}).get("cpus_allowed"))
    return threads is not None and cpus is not None and int(threads) > int(cpus)


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
        med = medians.get(u)
        out.append({
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
            "mstep_iters": inference_of(cfg, u.tool).mstep_iters if u.tool != "c" else None,
            "c_rss_over_rss": _ratio(rss.get(c_setup), rss.get(u)) if u.tool != "c" else None,
            "all_quiet": all(bool(r["quiet_before"]) for r in rs) if rs else None,
            "max_other_cpu_share": max((r["other_cpu_share_during"] or 0.0 for r in rs), default=None),
            "outputs_identical": len({r["output_sha256"] for r in rs}) == 1 if rs else None,
        })
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
        out.append({
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
        })
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
            out.append({
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
            })
    return out


def environment_rows(cfg: config.E3, lay: Layout, rows: Dict[Run, Dict], timings: Dict[Run, Dict]) -> List[Dict]:
    out: List[Dict] = []

    def add(key: str, value) -> None:
        out.append({"key": key, "value": value if isinstance(value, str) else json.dumps(value, sort_keys=True)})

    # Every distinct value, ';'-joined: the runs of a runs directory merged from
    # shards ran on several hosts (and SLURM jobs), and possibly CPU models.
    values: Dict[str, set] = {}
    for timing in timings.values():
        for k, v in timing.get("machine", {}).items():
            values.setdefault(k, set()).add(str(v))
    for k, vs in sorted(values.items()):
        add(f"machine.{k}", ";".join(sorted(vs)))
    seen = set()
    for run in sorted(rows, key=table_order):
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
    add("inference", asdict(cfg.inference))
    add("timing_protocol", asdict(cfg.timing))
    add("memory_limit_gb", f"{cfg.memory_limit_gb:g}")
    add("repeats", str(cfg.repeats))
    add("runs_not_quiet", str(sum(1 for r in rows.values() if not r["quiet_before"])))
    add("runs_not_clean", str(sum(1 for r in rows.values() if not r["clean"])))
    for key, value in cpu_use_rows(rows):
        add(key, value)
    kinds = clean_machine_kinds(rows, timings)
    add("clean_runs_machines", ";".join(kinds))
    add("clean_runs_on_mixed_machines", str(len(kinds) > 1))
    add("power_sources", ";".join(sorted({str(r["power_source"]) for r in rows.values()})))
    add("low_power_mode_seen", str(any(r["low_power_mode"] for r in rows.values())))
    return out


# What kept the machine busy before and during the runs, in cores (E3_runs.csv's columns).
CPU_USE_COLUMNS = (
    "busy_before_cores", "process_cores_before", "kernel_thread_cores_before", "interrupt_cores_before",
    "other_cores_during", "machine_other_cores_during", "kernel_thread_cores_during", "interrupt_cores_during",
    "steal_cores_during", "unaccounted_cores_during",
)
OTHER_METHODS = ("processes", "machine")  # what other_cores_during measures, by run (e3_run.other_method)


def cpu_use_rows(rows: Dict[Run, Dict]) -> List[Tuple[str, str]]:
    """What kept the machine busy over every run, for E3_environment.csv.

    The measures of other work the clean checks used (``other_methods``),
    the median and maximum of each of :data:`CPU_USE_COLUMNS` over the runs
    that recorded it (``other_cores_during`` apart for each measure, as
    ``cores.other_cores_during.processes`` and ``.machine``, which are not
    the same quantity), and how many other processes ended during runs.
    """
    def summary(xs: List[float]) -> str:
        return f"median {statistics.median(xs):.3f}, max {max(xs):.3f} over {len(xs)} runs" if xs else "not recorded"

    out = [("other_methods", ";".join(sorted({str(r["other_method"]) for r in rows.values()})))]
    for col in CPU_USE_COLUMNS:
        if col == "other_cores_during":
            for method in OTHER_METHODS:
                out.append((f"cores.{col}.{method}", summary([float(r[col]) for r in rows.values()
                                                              if r["other_method"] == method and r.get(col) is not None])))
            continue
        out.append((f"cores.{col}", summary([float(r[col]) for r in rows.values() if r.get(col) is not None])))
    exited = [r["other_processes_exited"] for r in rows.values() if r.get("other_processes_exited") is not None]
    out.append(("other_processes_exited", f"{sum(exited)} over {len(exited)} runs" if exited else "not recorded"))
    return out


def machine_kind(timing: Dict) -> str:
    """The CPU model, frequency governor and boost a run was timed on: what its time depends on beyond the tool."""
    m = timing.get("machine", {})
    return f"{m.get('cpu', 'unknown CPU')} (governor {m.get('cpu_governor', '?')}, boost {m.get('cpu_boost', '?')})"


def clean_machine_kinds(rows: Dict[Run, Dict], timings: Dict[Run, Dict]) -> List[str]:
    """Every :func:`machine_kind` of the clean runs; more than one makes the medians and speed-ups mix them."""
    return sorted({machine_kind(timings[run]) for run, row in rows.items() if row["clean"] and run in timings})


def seed_rows(truths: Dict[int, Dict]) -> List[Dict]:
    return [
        {"size_mb": size, "chromosome": ch["name"], "ancestry_seed": ch["seeds"]["ancestry"],
         "mutation_seed": ch["seeds"]["mutations"], "het_sites": ch["het_sites"], "het_bins": ch["het_bins"],
         "trees": ch.get("trees"), "longest_het_gap_bp": ch.get("longest_het_gap_bp")}
        for size in sorted(truths) for ch in truths[size]["chromosomes"]
    ]


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
    ordered = [rows[r] for r in sorted(rows, key=table_order)]
    kinds = clean_machine_kinds(rows, timings)
    if len(kinds) > 1:
        ctx.log(f"[E3] warning: the clean runs were timed on {len(kinds)} kinds of machine ({'; '.join(kinds)}); "
                "their medians and speed-ups mix them (E3_environment.csv: clean_runs_on_mixed_machines)")
    return write_tables(lay.table, {
        "table2": table2_rows(cfg, planned, rows),
        "runs": ordered,
        "scaling": scaling_rows(cfg, rows, outputs),
        "determinism": determinism_rows(outputs),
        "environment": environment_rows(cfg, lay, rows, timings),
        "seeds": seed_rows(truths),
    })


__all__ = [
    "Layout", "NAME", "Planned", "Run", "Setup", "TABLES", "estimate_mb", "estimate_seconds", "jobs", "plan",
    "report", "runs", "setups", "sizes",
]
