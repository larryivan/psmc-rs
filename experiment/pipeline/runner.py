"""The resumable, detached job runner.

Experiments declare :class:`Job` objects: a command, its input and output
files, the cores and memory it needs, an estimated duration, whether it must
run alone (``exclusive``), and the settings, tool identities and seeds that
determine its outputs. A job that reads a file another job writes runs after
it (make-like). A job is *done* when its outputs exist with a matching
provenance record (:mod:`.provenance`).

``start`` runs every job that is not done, as a daemon in a session of its
own, which outlives the one that started it: it writes
``<runs>/_runner/runner.pid``, holds ``runner.lock`` while it lives and logs
to ``<runs>/_runner/runner.log`` and ``<runs>/<experiment>/logs/``. The
running jobs stay within a budget of cores and memory (default 7 cores,
12 GB). Exclusive jobs run only when nothing else runs, in order of their
``rank`` (E3 interleaves the repeats of its timing runs), and must fit the
memory budget but not the cores. Runners of other runs directories (or
worktrees) on the same machine coordinate through a machine-wide lock
(:class:`MachineLock`), held shared for ordinary jobs and exclusively for an
exclusive one, so that an exclusive job never overlaps another runner's jobs.

``stop`` sends SIGTERM (SIGINT does the same): the running jobs are
terminated and stay pending, also when they die of the signal first, as
under SLURM, which signals every process of a job. ``stop --drain`` sends
SIGUSR1: no new job starts, and the runner exits when the running ones
finish. ``stop`` refuses a runner that ``runner.host`` places on another
host. A failed job is recorded (``logs/attempts.jsonl`` and its log file)
and blocks the jobs that need its outputs; the next ``start`` retries it. A
job that exits with status 75 (``EX_TEMPFAIL``: E3's timing wrapper when the
machine is not quiet) is deferred instead: it runs again after
:data:`DEFER_SECONDS`. ``status`` prints done/running/failed/pending per
experiment with an ETA.

Every attempt records its wall time, the peak RSS of its process tree
(psutil, sampled every 50 ms or ``$PSMC_EXP_SAMPLE_SECONDS``, plus the main
process's ``ru_maxrss`` from ``wait4``), its exit status, the host (and
SLURM job) it ran on and the machine's load average before and after.
Exclusive jobs also record the busiest other processes before they start,
and are not sampled (their own wrapper measures them; a sampler would take
CPU from the run).

The jobs run this checkout's executables (``target/release/psmc-rs``, the
tools directory's binaries) and pipeline code, which development may change
during a long run. The runner records the SHA-256 of the pipeline's Python
files and of every executable a job names when it starts (:class:`Watch`),
and the executables' checksums with every attempt. Once one of them changes
it starts no further job: it waits for the running ones and exits, so that
no record claims an identity its job did not run with. The next ``start``
identifies the tools afresh.
"""

from __future__ import annotations

import bisect
import errno
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config, paths, provenance

DONE, READY, PENDING, RUNNING, FAILED, BLOCKED = "done", "ready", "pending", "running", "failed", "blocked"
DEFERRED, INTERRUPTED = "deferred", "interrupted"  # attempts after which the job runs again
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+=-]*(/[A-Za-z0-9_.+=-]+)+$")
SAMPLE_SECONDS_ENV = "PSMC_EXP_SAMPLE_SECONDS"
MIN_SAMPLE_SECONDS = 0.01
EXIT_POLL_SECONDS = 0.05  # how often a job's exit is checked: the resolution of its wall time
POLL_SECONDS = 0.25
EX_TEMPFAIL = 75  # a job's "not now": deferred, not failed
DEFER_SECONDS = 60.0
MACHINE_LOCK_ENV = "PSMC_EXP_MACHINE_LOCK"


def sample_seconds(environ: Optional[Dict[str, str]] = None) -> float:
    """Seconds between samples of a job's process tree: ``$PSMC_EXP_SAMPLE_SECONDS``, default 0.05.

    Every sample lists the machine's processes (psutil reads ``/proc`` on
    Linux) while holding the GIL, so with ~190 jobs at once on a 192-core
    node, 20 samples a second per job keep the runner's threads busy; a
    longer interval misses short peaks of memory but not the job's end.
    Read when a runner starts (not on import: a bad value must not break
    ``stop`` or ``status``); below 0.01 s the job threads would busy-poll.
    """
    raw = (os.environ if environ is None else environ).get(SAMPLE_SECONDS_ENV, "")
    if not raw.strip():
        return 0.05
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not MIN_SAMPLE_SECONDS <= value < math.inf:
        raise ValueError(f"{SAMPLE_SECONDS_ENV}={raw!r}: give a number of seconds, at least {MIN_SAMPLE_SECONDS:g}")
    return value


def this_host() -> str:
    """The name of this machine, as runners record it (a PID means nothing on another host)."""
    return socket.gethostname()


# macOS derives the host name from the network when none is set (``scutil
# --get HostName``), so the laptop's own runner could look like another
# host's; and no other machine shares a runs directory there.
GUARD_HOSTS = sys.platform != "darwin"


def on_another_host(host: Optional[str]) -> bool:
    """Whether ``host``, recorded by a runner, is another machine than this one (never on macOS)."""
    return GUARD_HOSTS and host is not None and host != this_host()


@dataclass
class Job:
    """One command of an experiment. ``id`` is ``<experiment>/<path>`` (``E1/fit-c/constant/rep01``)."""

    id: str
    command: List[str]
    inputs: List[Path] = field(default_factory=list)
    outputs: List[Path] = field(default_factory=list)
    cores: int = 1
    mem_gb: float = 1.0
    exclusive: bool = False
    est_seconds: float = 60.0
    settings: Dict = field(default_factory=dict)
    tools: Dict[str, Dict] = field(default_factory=dict)
    seeds: Dict = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    stdout: Optional[Path] = None  # a file for the command's stdout (default: its log)
    description: str = ""
    rank: int = 0  # exclusive jobs start in order of rank, then of their chain of dependent work

    def __post_init__(self):
        if not _ID_RE.match(self.id):
            raise ValueError(f"bad job id {self.id!r}: use <experiment>/<name>[/...] of [A-Za-z0-9_.+=-]")
        if self.cores < 1:
            raise ValueError(f"{self.id}: a job needs at least one core, not {self.cores}")
        self.command = [str(c) for c in self.command]
        self.inputs = [Path(p) for p in self.inputs]
        self.outputs = [Path(p) for p in self.outputs]
        self.stdout = Path(self.stdout) if self.stdout is not None else None
        if self.stdout is not None and self.stdout not in self.outputs:
            self.outputs.append(self.stdout)

    @property
    def experiment(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.id.split("/", 1)[1]

    def record_path(self, runs_dir: Path) -> Path:
        return Path(runs_dir) / self.experiment / "provenance" / f"{self.name}.json"

    def log_path(self, runs_dir: Path) -> Path:
        return Path(runs_dir) / self.experiment / "logs" / "jobs" / f"{self.name}.log"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["inputs"] = [str(p) for p in self.inputs]
        d["outputs"] = [str(p) for p in self.outputs if p != self.stdout]
        d["stdout"] = str(self.stdout) if self.stdout else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Job":
        return cls(**d)


# ---------------------------------------------------------------------------
# The job graph
# ---------------------------------------------------------------------------


class GraphError(ValueError):
    pass


def path_key(p: Path, real_dirs: Dict[str, str]) -> str:
    """A path as the file system tells files apart: resolved, and without case on macOS and Windows.

    macOS's default file systems ignore case, so ``figS2.pdf`` and
    ``FigS2.pdf`` are one file there, and two jobs writing them would
    overwrite each other's output. ``real_dirs`` holds the directories
    resolved so far: resolving costs an ``lstat`` per component, and a
    graph's files lie in far fewer directories than there are files (a file
    that is itself a symbolic link is resolved whole).
    """
    s = os.fspath(p)
    head, name = os.path.split(s)
    if os.path.isabs(s) and name not in ("", ".", "..") and not os.path.islink(s):
        if head not in real_dirs:
            real_dirs[head] = os.path.realpath(head)
        s = os.path.join(real_dirs[head], name)
    else:
        s = os.path.realpath(s)
    return s.lower() if sys.platform in ("darwin", "win32") else s


def select(jobs: Sequence[Job], pattern: Optional[str]) -> List[Job]:
    """The jobs whose id matches ``pattern`` (:func:`re.search`), all of them when it is None.

    For sharding a run over runs directories (E3's repeats over several
    nodes, merged afterwards). A graph of the kept jobs knows nothing of the
    others: a kept job's input that a dropped job would make is a file no
    job makes, so the job runs if the file exists (with its content in the
    key, like any input) and is blocked while it does not.
    """
    if pattern is None:
        return list(jobs)
    rx = re.compile(pattern)
    return [j for j in jobs if rx.search(j.id)]


class Graph:
    """Jobs and their file dependencies."""

    def __init__(self, jobs: Sequence[Job], runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.jobs: Dict[str, Job] = {}
        for j in jobs:
            if j.id in self.jobs:
                raise GraphError(f"duplicate job id {j.id}")
            self.jobs[j.id] = j
        producer: Dict[str, str] = {}
        real_dirs: Dict[str, str] = {}
        for j in jobs:
            for p in [*j.outputs, j.record_path(self.runs_dir)]:
                key = path_key(p, real_dirs)
                if key in producer:
                    raise GraphError(f"{p} is an output of both {producer[key]} and {j.id}")
                producer[key] = j.id
        self.deps: Dict[str, List[str]] = {}
        self.static_inputs: Dict[str, List[Path]] = {}  # the inputs no job makes
        for j in jobs:
            deps, static = [], []
            for p in j.inputs:
                pid = producer.get(path_key(p, real_dirs))
                if pid is None:
                    static.append(p)
                elif pid == j.id:
                    raise GraphError(f"{j.id} reads its own output {p}")
                elif pid not in deps:
                    deps.append(pid)
            self.deps[j.id] = deps
            self.static_inputs[j.id] = static
        self.order = self._topological_order()
        self.dependents: Dict[str, List[str]] = {jid: [] for jid in self.jobs}
        for jid, deps in self.deps.items():
            for d in deps:
                self.dependents[d].append(jid)
        self.priority = self._critical_paths()

    def _topological_order(self) -> List[str]:
        order: List[str] = []
        mark: Dict[str, int] = {}

        def visit(jid: str, stack: Tuple[str, ...]) -> None:
            if mark.get(jid) == 2:
                return
            if mark.get(jid) == 1:
                raise GraphError("dependency cycle: " + " -> ".join(stack + (jid,)))
            mark[jid] = 1
            for d in self.deps[jid]:
                visit(d, stack + (jid,))
            mark[jid] = 2
            order.append(jid)

        for jid in self.jobs:
            visit(jid, ())
        return order

    def _critical_paths(self) -> Dict[str, float]:
        """Estimated seconds from a job's start to the end of its longest chain of dependents."""
        out: Dict[str, float] = {}
        for jid in reversed(self.order):
            out[jid] = self.jobs[jid].est_seconds + max((out[d] for d in self.dependents[jid]), default=0.0)
        return out

    def experiments(self) -> List[str]:
        """The experiments of the jobs, in the order of their first jobs."""
        return list(dict.fromkeys(j.experiment for j in self.jobs.values()))


# ---------------------------------------------------------------------------
# Running one job
# ---------------------------------------------------------------------------


def tree_rss(proc) -> int:
    """Resident set size of a psutil process and all its descendants, in bytes."""
    import psutil

    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def busiest_processes(limit: int = 5, interval: float = 0.5) -> List[Dict]:
    """The processes using the most CPU over ``interval`` seconds (for exclusive runs)."""
    import psutil

    procs = list(psutil.process_iter(["pid", "name"]))
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(interval)
    usage = []
    me = os.getpid()
    for p in procs:
        try:
            if p.pid != me:
                usage.append((p.cpu_percent(None), p.info["pid"], p.info["name"]))
        except psutil.Error:
            pass
    usage.sort(reverse=True)
    return [{"cpu_percent": round(c, 1), "pid": pid, "name": name} for c, pid, name in usage[:limit]]


def _repo_on_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    """``env`` with the repository first on PYTHONPATH (``python -m experiment.pipeline`` from anywhere)."""
    env["PYTHONPATH"] = str(paths.ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def job_environment(job: Job) -> Dict[str, str]:
    """The job's environment: ours, the repository on PYTHONPATH, thread pools capped at its cores."""
    env = _repo_on_pythonpath(dict(os.environ))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
        env[var] = str(job.cores)
    env.update(job.env)
    return env


class Execution(threading.Thread):
    """Runs one job in its own process group and measures it."""

    def __init__(self, job: Job, runs_dir: Path, executables: Optional[Dict[str, str]] = None,
                 sample_every: float = 0.05):
        super().__init__(name=f"job:{job.id}", daemon=True)
        self.job = job
        self.runs_dir = Path(runs_dir)
        self.executables = dict(executables or {})  # {portable path: SHA-256} of what the command runs
        self.sample_every = sample_every  # seconds between samples of the process tree's RSS
        self.proc: Optional[subprocess.Popen] = None
        self.result: Optional[Dict] = None
        self.started = time.time()
        self._kill = threading.Event()

    def terminate(self) -> None:
        # Not proc.poll(): it would reap the process that _run waits for.
        self._kill.set()
        if self.proc is not None and self.proc.returncode is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def run(self) -> None:
        try:
            self.result = self._run()
        except Exception as exc:  # noqa: BLE001 - reported as a failed attempt
            self.result = {"status": FAILED, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

    def _run(self) -> Dict:
        job = self.job
        for p in job.outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                p.unlink()
        record = job.record_path(self.runs_dir)
        if record.exists():
            record.unlink()
        log = job.log_path(self.runs_dir)
        log.parent.mkdir(parents=True, exist_ok=True)
        busiest = busiest_processes() if job.exclusive else None
        with open(log, "w", encoding="utf-8") as logf:
            logf.write(f"# {job.id}\n# {' '.join(job.command)}\n# started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        load_before = os.getloadavg()
        start_wall = time.time()
        # Appending: the command's output and the closing line share the file.
        with open(log, "ab") as logf:
            wall, status, rusage, peak = self._execute(logf)
            exit_code = os.waitstatus_to_exitcode(status)
            self.proc.returncode = exit_code
            logf.write(f"\n# exit {exit_code} after {wall:.1f} s\n".encode())
        maxrss = rusage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)  # MiB
        run = {
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_wall)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": round(wall, 3),
            "peak_rss_mb": round(max(peak / 2**20, maxrss), 1),
            "peak_rss_tree_sampled_mb": round(peak / 2**20, 1) if not job.exclusive else None,
            "maxrss_main_mb": round(maxrss, 1),
            "user_seconds": round(rusage.ru_utime, 3),
            "system_seconds": round(rusage.ru_stime, 3),
            "exit_status": exit_code,
            "loadavg_before": [round(x, 2) for x in load_before],
            "loadavg_after": [round(x, 2) for x in os.getloadavg()],
            "cores": job.cores,
            "mem_gb": job.mem_gb,
            "exclusive": job.exclusive,
            "executables": self.executables,
            "host": this_host(),  # information only, like the rest of the run: not in the job's key
        }
        if os.environ.get("SLURM_JOB_ID"):
            run["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        if busiest is not None:
            run["busiest_processes_before"] = busiest
        if self._kill.is_set():
            return {"status": INTERRUPTED, "run": run}
        if exit_code == EX_TEMPFAIL:
            return {"status": DEFERRED, "run": run, "error": f"exit status {exit_code}: not now", "log_tail": tail(log)}
        if exit_code != 0:
            return {"status": FAILED, "run": run, "error": f"exit status {exit_code}", "log_tail": tail(log)}
        missing = [str(p) for p in job.outputs if not p.exists()]
        if missing:
            return {"status": FAILED, "run": run, "error": f"declared outputs missing: {missing}", "log_tail": tail(log)}
        return {"status": DONE, "run": run}

    def _execute(self, logf) -> Tuple[float, int, object, int]:
        """Run the command to its end: its wall seconds, wait status, rusage and tree's peak RSS (bytes)."""
        import psutil

        job = self.job
        out = open(job.stdout, "wb") if job.stdout else logf
        try:
            t0 = time.perf_counter()
            self.proc = subprocess.Popen(job.command, cwd=paths.ROOT, env=job_environment(job),
                                         stdin=subprocess.DEVNULL, stdout=out, stderr=logf, start_new_session=True)
            try:
                ps = psutil.Process(self.proc.pid)
            except psutil.Error:
                ps = None
            sample = ps is not None and not job.exclusive
            peak, next_sample = 0, t0
            while True:
                pid, status, rusage = os.wait4(self.proc.pid, os.WNOHANG)
                if pid != 0:
                    return time.perf_counter() - t0, status, rusage, peak
                if sample and time.perf_counter() >= next_sample:
                    peak = max(peak, tree_rss(ps))
                    next_sample = time.perf_counter() + self.sample_every
                time.sleep(min(self.sample_every, EXIT_POLL_SECONDS))
        finally:
            if job.stdout:
                out.close()


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Records of attempts
# ---------------------------------------------------------------------------


def attempts_path(runs_dir: Path, experiment: str) -> Path:
    return Path(runs_dir) / experiment / "logs" / "attempts.jsonl"


def append_attempt(runs_dir: Path, job: Job, result: Dict) -> None:
    p = attempts_path(runs_dir, job.experiment)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"job": job.id, "status": result.get("status"), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for k in ("error", "run", "log_tail"):
        if k in result:
            rec[k] = result[k]
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def last_attempts(runs_dir: Path, experiments: Iterable[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for exp in experiments:
        p = attempts_path(runs_dir, exp)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out[rec["job"]] = rec
    return out


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    job: Job
    started: float  # time.time() or simulated seconds

    def remaining(self, now: float) -> float:
        return max(self.job.est_seconds - (now - self.started), 0.0)


def fits_budget(job: Job, budget: config.Budget) -> Optional[str]:
    """Why ``job`` can never start within ``budget``, or None.

    An exclusive job runs alone, so its cores (its threads: E3 gives a timing
    run as many cores as threads, at most the machine's) need not fit the
    budget, which is the share of the machine that ordinary jobs may use at
    once. Its memory must: the budget is also what the machine (or a SLURM
    allocation) has to give.
    """
    if job.cores > budget.cores and not job.exclusive:
        return f"needs {job.cores} cores, budget {budget.cores}"
    if job.mem_gb > budget.mem_gb:
        return f"needs {job.mem_gb:g} GB, budget {budget.mem_gb:g} GB"
    return None


def priority_order(priority: Dict[str, float]) -> Callable[[Job], Tuple[float, str]]:
    """The sort key of :func:`pick`'s order: the longest chain of dependent work first, then the id."""
    return lambda j: (-priority.get(j.id, 0.0), j.id)


def pick(ready: Sequence[Job], running: Sequence[Slot], budget: config.Budget, priority: Dict[str, float],
         now: float) -> List[Job]:
    """The ready jobs to start now.

    Non-exclusive jobs start in priority order (longest chain of dependent
    work first) while they fit the free cores and memory. When the first one
    does not fit, later ones may start only if they are expected to finish
    before enough running work ends for it ("backfilling"), so a large job is
    not starved by small ones. An exclusive job starts only when nothing
    runs and no non-exclusive job is ready, and then alone.
    """
    return pick_in_order(sorted(ready, key=priority_order(priority)), running, budget, priority, now)


def pick_in_order(ready: Sequence[Job], running: Sequence[Slot], budget: config.Budget,
                  priority: Dict[str, float], now: float) -> List[Job]:
    """:func:`pick` of ready jobs already sorted by :func:`priority_order` (the schedule keeps them so)."""
    if any(s.job.exclusive for s in running):
        return []
    free_cores = budget.cores - sum(s.job.cores for s in running)
    free_mem = budget.mem_gb - sum(s.job.mem_gb for s in running)
    chosen: List[Job] = []
    head_blocked_until: Optional[float] = None
    any_normal = False
    for j in ready:
        if j.exclusive:
            continue
        any_normal = True
        if free_cores < 1:  # every job needs a core: no other one fits
            break
        fits = j.cores <= free_cores and j.mem_gb <= free_mem + 1e-9
        if head_blocked_until is None and not fits:
            head_blocked_until = min((s.remaining(now) for s in running), default=0.0)
        elif fits and (head_blocked_until is None or j.est_seconds <= head_blocked_until):
            chosen.append(j)
            free_cores -= j.cores
            free_mem -= j.mem_gb
    if not chosen and not running and not any_normal and ready:  # all the ready jobs are exclusive
        chosen.append(min(ready, key=lambda j: (j.rank, -priority.get(j.id, 0.0), j.id)))
    return chosen


def simulate_schedule(
    graph: Graph, todo: Sequence[str], budget: config.Budget, running: Dict[str, float]
) -> Dict[str, float]:
    """Expected finish time (seconds from now) of each job in ``todo``.

    ``running`` maps running job ids to their expected remaining seconds.
    Jobs not in ``todo`` or ``running`` count as done. The ready jobs stay
    in :func:`pick`'s order, which thousands of them at a time (E7's fits)
    would otherwise cost a sort at every step.
    """
    todo_set = set(todo)
    now = 0.0
    slots = [Slot(graph.jobs[jid], now - (graph.jobs[jid].est_seconds - rem)) for jid, rem in running.items()]
    pending = dict.fromkeys(jid for jid in graph.order if jid in todo_set and jid not in running)
    finished = {jid for jid in graph.jobs if jid not in todo_set and jid not in running}
    waiting = {jid: sum(d not in finished for d in graph.deps[jid]) for jid in pending}  # on unfinished dependencies

    def runnable(jid: str) -> bool:
        return waiting[jid] == 0 and fits_budget(graph.jobs[jid], budget) is None

    order = priority_order(graph.priority)
    ready = sorted((graph.jobs[jid] for jid in pending if runnable(jid)), key=order)
    finish: Dict[str, float] = {}
    guard = 0
    while pending or slots:
        guard += 1
        if guard > 100000:
            break
        for j in pick_in_order(ready, slots, budget, graph.priority, now):
            slots.append(Slot(j, now))
            del pending[j.id], ready[bisect.bisect_left(ready, order(j), key=order)]
        if not slots:
            finish.update(dict.fromkeys(pending, math.inf))
            break
        now = max(now, min(s.started + s.job.est_seconds for s in slots))
        for s in [s for s in slots if s.started + s.job.est_seconds <= now + 1e-9]:
            slots.remove(s)
            finish[s.job.id] = now
            for d in graph.dependents[s.job.id]:
                if d in pending:
                    waiting[d] -= 1
                    if runnable(d):
                        bisect.insort(ready, graph.jobs[d], key=order)
    return finish


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


def runner_dir(runs_dir: Path) -> Path:
    return Path(runs_dir) / "_runner"


def read_state(runs_dir: Path) -> Dict:
    """The ``state.json`` that the running (or else the last) runner of ``runs_dir`` wrote, or {}."""
    return provenance.read_json(runner_dir(runs_dir) / "state.json")


def try_flock(fd: int, flag: int) -> bool:
    """``flock(fd, flag)`` without waiting; False if another process holds a lock in the way."""
    try:
        fcntl.flock(fd, flag | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise
    return True


class Lock:
    """An exclusive ``flock`` on ``<runs>/_runner/runner.lock``, held by the daemon while it lives."""

    def __init__(self, runs_dir: Path):
        self.path = runner_dir(runs_dir) / "runner.lock"
        self.fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            locked = try_flock(fd, fcntl.LOCK_EX)
        except OSError:
            os.close(fd)
            raise
        if not locked:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def default_machine_lock(platform: str = sys.platform, host: Optional[str] = None) -> Path:
    """``~/.cache/psmc-exp/machine.lock`` on macOS; elsewhere ``machine-<host>.lock`` there.

    A cluster's nodes share the home directory over NFS, which passes
    ``flock`` on to the server: one file for all of them would keep an
    exclusive job on one node from running beside any job on another.
    """
    base = Path.home() / ".cache" / "psmc-exp"
    if platform == "darwin":
        return base / "machine.lock"
    return base / f"machine-{host or this_host()}.lock"


class MachineLock:
    """A machine-wide ``flock`` that every runner holds while it runs jobs.

    Shared for ordinary jobs, exclusive for exclusive ones. The file is
    ``$PSMC_EXP_MACHINE_LOCK`` or :func:`default_machine_lock`.
    Locks are taken without blocking: a runner that cannot take the mode it
    needs starts nothing and tries again at its next poll.
    """

    def __init__(self, path: Optional[Path] = None):
        env = os.environ.get(MACHINE_LOCK_ENV)
        self.path = Path(path or env or default_machine_lock()).expanduser()
        self.fd: Optional[int] = None
        self.mode: Optional[str] = None  # None, "shared" or "exclusive"

    def hold(self, mode: str) -> bool:
        """Hold the lock in ``mode`` ("shared" or "exclusive"); False if another runner's lock is in the way."""
        if self.mode == mode:
            return True
        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        if not try_flock(self.fd, fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH):
            return False
        self.mode = mode
        return True

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        self.fd, self.mode = None, None


class NoMachineLock(MachineLock):
    """No coordination with other runners (a Daemon made directly, as the tests make them)."""

    def __init__(self):
        super().__init__(Path(os.devnull))

    def hold(self, mode: str) -> bool:
        self.mode = mode
        return True

    def release(self) -> None:
        self.mode = None


def daemon_alive(runs_dir: Path) -> Optional[int]:
    """The PID of the running daemon of ``runs_dir``, or None (without creating anything in ``runs_dir``)."""
    lock = Lock(runs_dir)
    if not lock.path.exists():
        return None
    if lock.acquire():
        lock.release()
        return None
    try:
        return int((runner_dir(runs_dir) / "runner.pid").read_text().strip())
    except (OSError, ValueError):
        return -1


def runner_host(runs_dir: Path) -> Optional[str]:
    """The host the running (or last) daemon of ``runs_dir`` runs on, or None if it did not say."""
    try:
        return (runner_dir(runs_dir) / "runner.host").read_text().strip() or None
    except OSError:
        return None


class Logger:
    """Timestamped lines in ``<runs>/_runner/runner.log`` and, about an experiment's jobs, in its ``logs/``."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.main = runner_dir(runs_dir) / "runner.log"
        self.main.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str, experiment: Optional[str] = None) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
        with open(self.main, "a", encoding="utf-8") as f:
            f.write(line)
        if experiment:
            p = self.runs_dir / experiment / "logs" / "runner.log"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)


def executables(command: Sequence[str]) -> List[Path]:
    """The words of a command that are executable files (the tool, an interpreter, a wrapper)."""
    out = []
    for word in command:
        p = Path(word)
        if p.is_absolute() and p.is_file() and os.access(p, os.X_OK) and p not in out:
            out.append(p)
    return out


def pipeline_files() -> List[Path]:
    """The pipeline's Python files, which every Python step and job key depend on."""
    return sorted((paths.EXPERIMENT / "pipeline").rglob("*.py"))


class Watch:
    """Files whose content must not change while a runner lives: its code and the executables its jobs run.

    Checksums go through the runner's cache (by size and modification
    time), so checking costs a ``stat`` per file unless a file was touched.
    """

    def __init__(self, files: Iterable[Path], cache: provenance.ChecksumCache):
        self.cache = cache
        self.snapshot: Dict[Path, str] = {}
        for p in files:
            p = Path(p).resolve()
            if p not in self.snapshot and p.is_file():
                self.snapshot[p] = cache.sha256(p)

    def changed(self) -> List[Path]:
        """The watched files that changed (or vanished) since the snapshot."""
        out = []
        for p, digest in self.snapshot.items():
            try:
                if self.cache.sha256(p) != digest:
                    out.append(p)
            except OSError:
                out.append(p)
        return out

    def digests(self, files: Iterable[Path], runs_dir: Path) -> Dict[str, str]:
        """``{portable path: SHA-256 at the snapshot}`` of ``files`` (the executables of one job)."""
        return {paths.portable(str(p), runs_dir): self.snapshot[Path(p).resolve()]
                for p in files if Path(p).resolve() in self.snapshot}


class Daemon:
    """The scheduling loop. Runs in the foreground of whatever process calls :meth:`run`."""

    def __init__(self, graph: Graph, runs_dir: Path, budget: config.Budget, log: Logger, meta: Dict,
                 machine_lock: Optional[MachineLock] = None):
        self.graph = graph
        self.runs_dir = Path(runs_dir)
        self.budget = budget
        self.log = log
        self.meta = meta
        self.cache = provenance.ChecksumCache(runner_dir(runs_dir) / "checksums.json")
        self.state: Dict[str, str] = {jid: PENDING for jid in graph.jobs}
        self.reason: Dict[str, str] = {}
        self.running: Dict[str, Execution] = {}
        self.keys: Dict[str, Dict] = {}
        self.stop_now = threading.Event()
        self.drain = threading.Event()
        self.exes: Dict[str, List[Path]] = {jid: executables(j.command) for jid, j in graph.jobs.items()}
        self.watch = Watch([*pipeline_files(), *(p for ps in self.exes.values() for p in ps)], self.cache)
        self.halted: Optional[str] = None  # why no further job starts
        # daemon_main passes the machine-wide lock; a Daemon made directly (tests) coordinates with nothing.
        self.machine = machine_lock if machine_lock is not None else NoMachineLock()
        self.sample_every = sample_seconds()
        self.not_before: Dict[str, float] = {}  # deferred jobs: when they may start again
        self.deferrals: Dict[str, int] = {}
        self._waiting_logged = 0.0

    # -- state -------------------------------------------------------------
    def _deps_state(self, jid: str) -> str:
        states = [self.state[d] for d in self.graph.deps[jid]]
        if any(s in (FAILED, BLOCKED) for s in states):
            return BLOCKED
        return DONE if all(s == DONE for s in states) else PENDING

    def refresh(self) -> None:
        """Move pending jobs whose dependencies are done to done, ready or blocked."""
        for jid in self.graph.order:
            if self.state[jid] != PENDING:
                continue
            deps = self._deps_state(jid)
            job = self.graph.jobs[jid]
            if deps == BLOCKED:
                self.state[jid] = BLOCKED
                self.reason[jid] = "needs " + next(d for d in self.graph.deps[jid] if self.state[d] in (FAILED, BLOCKED))
                continue
            if deps != DONE:
                continue
            missing = [p for p in self.graph.static_inputs[jid] if not p.exists()]
            if missing:
                self.state[jid] = BLOCKED
                self.reason[jid] = f"input missing: {missing[0]}"
                continue
            done, why = provenance.check_done(job, self.runs_dir, self.cache)
            if done:
                self.state[jid] = DONE
                continue
            over = fits_budget(job, self.budget)
            if over:
                self.state[jid] = BLOCKED
                self.reason[jid] = over
                self.log(f"blocked {jid}: {over}", job.experiment)
                continue
            self.state[jid] = READY
            self.reason[jid] = why

    def write_state(self) -> None:
        """``<runs>/_runner/state.json``: how the runner was started, and its jobs that are not pending."""
        running = {jid: {"pid": ex.proc.pid if ex.proc else None, "start": ex.started, "cores": ex.job.cores,
                         "mem_gb": ex.job.mem_gb, "est_seconds": ex.job.est_seconds}
                   for jid, ex in self.running.items()}
        provenance.write_json_atomic(runner_dir(self.runs_dir) / "state.json", {
            **self.meta,
            "pid": os.getpid(),
            "updated": time.time(),
            "draining": self.drain.is_set(),
            "halted": self.halted,
            "running": running,
            "failed": {j: self.reason.get(j, "") for j, s in self.state.items() if s == FAILED},
            "deferred": {j: {"until": t, "times": self.deferrals.get(j, 0), "why": self.reason.get(j, "")}
                         for j, t in self.not_before.items()},
            "blocked": {j: self.reason.get(j, "") for j, s in self.state.items() if s == BLOCKED},
        })

    # -- execution -----------------------------------------------------------
    def launch(self, job: Job) -> None:
        self.keys[job.id] = provenance.job_key(job, self.runs_dir, self.cache)
        ex = Execution(job, self.runs_dir, executables=self.watch.digests(self.exes[job.id], self.runs_dir),
                       sample_every=self.sample_every)
        self.running[job.id] = ex
        self.state[job.id] = RUNNING
        self.log(f"start {job.id} ({self.reason.get(job.id, '')}): {' '.join(job.command)}", job.experiment)
        ex.start()

    def reap(self) -> bool:
        """Settle the jobs whose threads have ended; whether there were any."""
        changed = False
        for jid, ex in list(self.running.items()):
            if not ex.is_alive():
                del self.running[jid]
                self.settle(ex.job, ex.result or {"status": FAILED, "error": "no result"})
                changed = True
        return changed

    def settle(self, job: Job, res: Dict) -> None:
        """Record an attempt of ``job`` and move the job on: done, pending again, deferred or failed."""
        jid = job.id
        if res.get("status") == FAILED and self.stop_now.is_set() and res.get("run", {}).get("exit_status", 0) != 0:
            # SLURM signals every process of a job (at its time limit, on scancel), so the
            # jobs often die of the runner's SIGTERM before it terminates them itself.
            res = {**res, "status": INTERRUPTED}
        if res.get("status") == DONE:
            try:
                provenance.write_record(job, self.runs_dir, self.cache, self.keys[jid], res["run"])
            except Exception as exc:  # noqa: BLE001
                res = {**res, "status": FAILED, "error": f"recording provenance failed: {exc}"}
        append_attempt(self.runs_dir, job, res)
        status, run = res.get("status"), res.get("run", {})
        if status == DONE:
            self.state[jid] = DONE
            self.log(f"done {jid} in {run.get('wall_seconds', 0):.1f} s, peak RSS {run.get('peak_rss_mb', 0):.0f} MB", job.experiment)
        elif status == INTERRUPTED:
            self.state[jid] = PENDING
            self.log(f"interrupted {jid}", job.experiment)
        elif status == DEFERRED:
            self.state[jid] = READY
            self.not_before[jid] = time.time() + DEFER_SECONDS
            self.deferrals[jid] = self.deferrals.get(jid, 0) + 1
            why = (res.get("log_tail") or "").strip().splitlines()
            self.reason[jid] = f"deferred: {why[-2] if len(why) > 1 else 'not now'}"
            self.log(f"deferred {jid} ({self.reason[jid]}); again in {DEFER_SECONDS:.0f} s", job.experiment)
        else:
            self.state[jid] = FAILED
            self.reason[jid] = res.get("error", "failed")
            self.log(f"FAILED {jid}: {self.reason[jid]} (log: {job.log_path(self.runs_dir)})", job.experiment)

    def terminate_all(self, timeout: float = 20.0) -> None:
        for ex in self.running.values():
            ex.terminate()
        deadline = time.time() + timeout
        while any(ex.is_alive() for ex in self.running.values()) and time.time() < deadline:
            time.sleep(0.1)
        for ex in self.running.values():
            if ex.is_alive() and ex.proc is not None:
                try:
                    os.killpg(ex.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for ex in self.running.values():
            ex.join(timeout=10)
        self.reap()

    def run(self) -> int:
        """Run until every job is done, failed or blocked, or until stopped. Returns 1 if a job failed."""
        self.log(f"runner {os.getpid()}: {len(self.graph.jobs)} jobs of {', '.join(self.graph.experiments())}; "
                 f"budget {self.budget.cores} cores, {self.budget.mem_gb:g} GB")
        self.refresh()
        self.write_state()
        recorded_pids: Dict[str, Optional[int]] = {}
        last_write = last_save = time.time()
        while True:
            changed = self.reap()
            if self.stop_now.is_set():
                self.log("stop requested: terminating running jobs")
                self.terminate_all()
                break
            if changed:
                self.refresh()
            if not self.running:
                self.machine.release()
            started = not self.drain.is_set() and self.start_ready()
            # Rewrite the state when it changed, once the processes of new jobs
            # exist (their PIDs let a later runner stop them if this one is
            # killed), and every 10 s.
            pids = {jid: (ex.proc.pid if ex.proc else None) for jid, ex in self.running.items()}
            if changed or started or pids != recorded_pids or time.time() - last_write > 10:
                self.write_state()
                recorded_pids, last_write = pids, time.time()
            # The checksum cache every 10 s, not after every job: rewriting it takes ~20 ms once it holds the
            # production's 12,000 files, and an entry lost in a crash only costs a hash again.
            if time.time() - last_save > 10:
                self.cache.save()
                last_save = time.time()
            if not self.running and (self.drain.is_set() or READY not in self.state.values()):
                break
            time.sleep(POLL_SECONDS)
        self.machine.release()
        self.cache.save()
        self.write_state()
        counts = Counter(self.state.values())
        self.log("runner finished: " + ", ".join(f"{counts[s]} {s}" for s in (DONE, FAILED, BLOCKED, PENDING, READY)
                                                 if counts[s]))
        if self.halted:
            self.log(f"runner halted: {self.halted}")
        return 1 if counts[FAILED] or counts[BLOCKED] or self.halted else 0

    def start_ready(self) -> bool:
        """Start the ready jobs that :func:`pick` chooses; whether any started.

        None starts once a watched file has changed, or while another
        runner's machine lock is in the way.
        """
        now = time.time()
        ready = [self.graph.jobs[j] for j in self.graph.order
                 if self.state[j] == READY and self.not_before.get(j, 0.0) <= now]
        slots = [Slot(ex.job, ex.started) for ex in self.running.values()]
        chosen = pick(ready, slots, self.budget, self.graph.priority, now)
        if not chosen or self.code_or_tools_changed():
            return False
        if not self.machine.hold("exclusive" if any(j.exclusive for j in chosen) else "shared"):
            if now - self._waiting_logged > 600:
                self.log(f"waiting: another runner on this machine holds {self.machine.path}")
                self._waiting_logged = now
            return False
        for job in chosen:
            self.not_before.pop(job.id, None)
            self.launch(job)
        return True

    def code_or_tools_changed(self) -> bool:
        """Whether a watched file changed; if so, start nothing more (drain) and say why."""
        changed = self.watch.changed()
        if not changed:
            return False
        names = ", ".join(provenance.portable(changed[:3], self.runs_dir)) + ("..." if len(changed) > 3 else "")
        self.halted = (f"{names} changed since the runner started; no further job starts (running ones finish). "
                       "Start the runner again to rerun what the change affects")
        self.log(self.halted)
        self.drain.set()
        self.write_state()
        return True


def daemon_main(graph_factory: Callable[[], Graph], runs_dir: Path, budget: config.Budget, meta: Dict) -> int:
    """The daemon process: lock, PID file, signals, then :meth:`Daemon.run`."""
    runs_dir = Path(runs_dir)
    lock = Lock(runs_dir)
    if not lock.acquire():
        print(f"another runner holds {lock.path}", file=sys.stderr)
        return 3
    pid_file = runner_dir(runs_dir) / "runner.pid"
    host_file = runner_dir(runs_dir) / "runner.host"
    host_file.write_text(f"{this_host()}\n")  # before the PID: `stop` must not signal it on another host
    pid_file.write_text(f"{os.getpid()}\n")
    log = Logger(runs_dir)
    # Set by the signals even before the Daemon exists (while the graph is made),
    # so that a stop then is not lost; the Daemon takes them over.
    stop_now, drain = threading.Event(), threading.Event()

    def on_term(signum, frame):
        stop_now.set()

    def on_drain(signum, frame):
        drain.set()
        log("drain requested: no new jobs start")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    signal.signal(signal.SIGUSR1, on_drain)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        log(f"runner {os.getpid()} starting on {this_host()} ({meta})")
        kill_orphans(runs_dir, log)
        graph = graph_factory()
        meta = {**meta, "experiments": graph.experiments(), "host": this_host(), "machine": provenance.machine()}
        daemon = Daemon(graph, runs_dir, budget, log, meta, machine_lock=MachineLock())
        daemon.stop_now, daemon.drain = stop_now, drain
        return daemon.run()
    except Exception:
        log("runner crashed:\n" + traceback.format_exc())
        return 2
    finally:
        for p in (pid_file, host_file):
            try:
                p.unlink()
            except OSError:
                pass
        lock.release()


def kill_orphans(runs_dir: Path, log: Callable[[str], None]) -> None:
    """Stop the jobs of an earlier runner that died without stopping them (e.g. SIGKILL).

    They are the process groups recorded as running in ``state.json`` whose
    leader is still the process started then (same PID, started no later
    than 5 s after the job, leading its own process group). A state written
    on another host (a runs directory on a cluster's shared file system) is
    left alone: its PIDs are not this machine's processes.
    """
    import psutil

    state = read_state(runs_dir)
    host = state.get("host")
    if on_another_host(host):
        if state.get("running"):
            log(f"not stopping the {len(state['running'])} job(s) the last runner, on {host}, recorded as running: "
                f"they are not processes of {this_host()}")
        return
    for jid, info in state.get("running", {}).items():
        pid = info.get("pid")
        if not pid:
            continue
        try:
            proc = psutil.Process(pid)
            if proc.create_time() > info.get("start", 0) + 5 or os.getpgid(pid) != pid:
                continue  # a different process now has this PID
        except (psutil.Error, OSError):
            continue
        log(f"stopping job {jid} (pid {pid}) left running by an earlier runner")
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def spawn_daemon(argv: Sequence[str], runs_dir: Path, wait: float = 60.0) -> int:
    """Start ``argv`` (the daemon command) detached; return its PID once it holds the lock."""
    runs_dir = Path(runs_dir)
    rdir = runner_dir(runs_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "daemon.out", "ab") as out:
        proc = subprocess.Popen(list(argv), cwd=paths.ROOT, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env=_repo_on_pythonpath(dict(os.environ)), close_fds=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the runner exited at once (status {proc.returncode}); see {rdir / 'daemon.out'}")
        if daemon_alive(runs_dir) == proc.pid:
            return proc.pid
        time.sleep(0.1)
    return proc.pid


class ElsewhereError(RuntimeError):
    """The runner of a runs directory is not a process of this host."""


def stop_daemon(runs_dir: Path, drain: bool = False, wait: float = 60.0) -> Optional[int]:
    """Ask the daemon to stop (SIGTERM) or drain (SIGUSR1); wait for it to exit unless draining.

    Raises :class:`ElsewhereError` when the runner runs on another host (its
    PID would name an unrelated process here).
    """
    pid = daemon_alive(runs_dir)
    if pid is None or pid <= 0:
        return None
    host = runner_host(runs_dir)
    if on_another_host(host):
        raise ElsewhereError(f"the runner of {runs_dir} (pid {pid}) runs on {host}, not on {this_host()}: "
                             "stop it from there, or cancel its SLURM job")
    try:
        os.kill(pid, signal.SIGUSR1 if drain else signal.SIGTERM)
    except ProcessLookupError:
        if daemon_alive(runs_dir) is None:
            return None  # it exited meanwhile
        raise ElsewhereError(f"the runner of {runs_dir} holds its lock, but no process {pid} exists on "
                             f"{this_host()}: it runs on another host") from None
    if drain:
        return pid
    deadline = time.time() + wait
    while time.time() < deadline and daemon_alive(runs_dir) is not None:
        time.sleep(0.2)
    return pid


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@dataclass
class JobStatus:
    job: Job
    state: str
    reason: str = ""
    elapsed: float = 0.0


def job_statuses(graph: Graph, runs_dir: Path, daemon_pid: Optional[int] = None) -> List[JobStatus]:
    """The state of every job now, from its files, the daemon's state and the attempt records.

    ``daemon_pid`` is the result of :func:`daemon_alive` (looked up when not given).
    """
    runs_dir = Path(runs_dir)
    cache = provenance.ChecksumCache(runner_dir(runs_dir) / "checksums.json")
    alive = (daemon_pid if daemon_pid is not None else daemon_alive(runs_dir)) is not None
    state = read_state(runs_dir) if alive else {}
    running, deferred, blocked = (state.get(k, {}) for k in ("running", "deferred", "blocked"))
    attempts = last_attempts(runs_dir, graph.experiments())
    out: List[JobStatus] = []
    for jid in graph.order:
        job = graph.jobs[jid]
        if jid in running:
            out.append(JobStatus(job, RUNNING, elapsed=time.time() - running[jid]["start"]))
            continue
        absent = [p for p in graph.static_inputs[jid] if not p.exists()]
        if absent:  # no job of this graph makes it: waiting would be forever
            out.append(JobStatus(job, BLOCKED, f"input missing, and no job makes it: {absent[0]}"))
            continue
        missing = [p for p in job.inputs if not p.exists()]
        if missing:
            done, why = False, f"waiting for {missing[0].name}"
        else:
            done, why = provenance.check_done(job, runs_dir, cache)
        if done:
            out.append(JobStatus(job, DONE))
            continue
        last = attempts.get(jid)
        if last and last.get("status") == FAILED:
            out.append(JobStatus(job, FAILED, last.get("error", "")))
        elif jid in deferred:
            d = deferred[jid]
            out.append(JobStatus(job, PENDING, f"{d.get('why') or 'deferred'} ({d.get('times', 1)} time(s))"))
        elif jid in blocked:
            out.append(JobStatus(job, BLOCKED, blocked[jid]))
        else:
            over = fits_budget(job, config.Budget(**state["budget"])) if state.get("budget") else None
            out.append(JobStatus(job, BLOCKED if over else PENDING, over or why))
    cache.save()
    return out


def fmt_duration(seconds: float) -> str:
    if math.isinf(seconds):
        return "never (blocked)"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def format_status(
    graph: Graph, statuses: List[JobStatus], budget: config.Budget, runs_dir: Path, pid: Optional[int], verbose: bool = False
) -> str:
    """The status report; ``pid`` is the daemon's (None when it is not running)."""
    lines = [f"runs: {runs_dir}", f"runner: {'running (pid %s)' % pid if pid else 'not running'}"]
    todo = [s.job.id for s in statuses if s.state in (PENDING, FAILED, READY)]
    running = {s.job.id: max(s.job.est_seconds - s.elapsed, 0.0) for s in statuses if s.state == RUNNING}
    finish = simulate_schedule(graph, todo, budget, running)
    lines.append(f"{'experiment':<11}{'done':>6}{'running':>9}{'failed':>8}{'blocked':>9}{'pending':>9}  {'ETA':>10}  CPU left")
    for exp in graph.experiments():
        mine = [s for s in statuses if s.job.experiment == exp]
        c = {k: sum(1 for s in mine if s.state == k) for k in (DONE, RUNNING, FAILED, BLOCKED, PENDING)}
        ids = [s.job.id for s in mine if s.job.id in finish]
        eta = max((finish[i] for i in ids), default=0.0)
        cpu = sum(s.job.est_seconds * s.job.cores for s in mine if s.state in (PENDING, FAILED)) + sum(
            running[s.job.id] * s.job.cores for s in mine if s.state == RUNNING
        )
        lines.append(
            f"{exp:<11}{c[DONE]:>6}{c[RUNNING]:>9}{c[FAILED]:>8}{c[BLOCKED]:>9}{c[PENDING]:>9}  "
            f"{fmt_duration(eta) if (ids or c[RUNNING]) else '-':>10}  {cpu / 3600:.2f} h"
        )
    if pid is None and todo:
        lines.append("(ETA assumes the runner is started now; failed jobs are retried)")
    for s in statuses:
        if s.state == RUNNING:
            lines.append(f"  running  {s.job.id}  {fmt_duration(s.elapsed)} of ~{fmt_duration(s.job.est_seconds)}")
        elif s.state in (FAILED, BLOCKED):
            lines.append(f"  {s.state:<8} {s.job.id}  {s.reason}")
        elif verbose and s.state == PENDING:
            lines.append(f"  pending  {s.job.id}  {s.reason}")
    return "\n".join(lines)


def plan(graph: Graph, budget: config.Budget) -> Dict[str, Dict[str, float]]:
    """Estimated CPU hours and wall time of every experiment, run from scratch on ``budget``."""
    finish = simulate_schedule(graph, list(graph.jobs), budget, {})
    out: Dict[str, Dict[str, float]] = {}
    for exp in graph.experiments():
        jobs = [j for j in graph.jobs.values() if j.experiment == exp]
        sub = Graph(jobs, graph.runs_dir)
        alone = simulate_schedule(sub, list(sub.jobs), budget, {})
        out[exp] = {
            "jobs": len(jobs),
            "cpu_hours": sum(j.est_seconds * j.cores for j in jobs) / 3600,
            "wall_hours_alone": max(alone.values()) / 3600,
            "finish_hours": max(finish[j.id] for j in jobs) / 3600,
        }
    return out
