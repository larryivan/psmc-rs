"""The resumable, detached job runner.

Experiments declare :class:`Job` objects: a command, its input and output
files, the cores and memory it needs, an estimated duration, whether it must
run alone (``exclusive``), and the settings, tool identities and seeds that
determine its outputs. Dependencies follow from the files: a job that reads
a file another job writes runs after it (make-like).

A job is *done* when its outputs exist with a matching provenance record
(:mod:`.provenance`). ``start`` runs every job that is not done, as a daemon
that survives the session that started it: it detaches into its own session
(``start_new_session``), writes ``<runs>/_runner/runner.pid``, holds
``runner.lock`` while it lives and logs to ``<runs>/_runner/runner.log`` and
``<runs>/<experiment>/logs/``. It keeps the running jobs within a budget of
cores and memory (default 7 cores, 12 GB) and runs exclusive jobs only when
nothing else runs. ``stop`` ends it cleanly (``--drain``: after the running
jobs finish). A failed job is recorded (``logs/attempts.jsonl`` and its log
file) and blocks the jobs that need its outputs; the next ``start`` retries
it. ``status`` prints done/running/failed/pending per experiment with an
ETA.

Every attempt records its wall time, the peak RSS of its process tree
(psutil, sampled every 50 ms, plus the main process's ``ru_maxrss`` from
``wait4``), its exit status and the machine's load average before and after;
exclusive jobs also record the busiest other processes before they start.
"""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config, paths, provenance

DONE, READY, PENDING, RUNNING, FAILED, BLOCKED = "done", "ready", "pending", "running", "failed", "blocked"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+=-]*(/[A-Za-z0-9_.+=-]+)+$")
SAMPLE_SECONDS = 0.05
POLL_SECONDS = 0.25


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

    def __post_init__(self):
        if not _ID_RE.match(self.id):
            raise ValueError(f"bad job id {self.id!r}: use <experiment>/<name>[/...] of [A-Za-z0-9_.+=-]")
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
        for j in jobs:
            for p in [*j.outputs, j.record_path(self.runs_dir)]:
                key = str(Path(p).resolve())
                if key in producer:
                    raise GraphError(f"{p} is an output of both {producer[key]} and {j.id}")
                producer[key] = j.id
        self.producer = producer
        self.deps: Dict[str, List[str]] = {}
        self.static_inputs: Dict[str, List[Path]] = {}
        for j in jobs:
            deps, static = [], []
            for p in j.inputs:
                pid = producer.get(str(Path(p).resolve()))
                if pid is None:
                    static.append(Path(p))
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
            tail = max((out[d] for d in self.dependents[jid]), default=0.0)
            out[jid] = self.jobs[jid].est_seconds + tail
        return out

    def experiments(self) -> List[str]:
        seen: List[str] = []
        for j in self.jobs.values():
            if j.experiment not in seen:
                seen.append(j.experiment)
        return seen


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


def job_environment(job: Job) -> Dict[str, str]:
    """The job's environment: ours, the repository on PYTHONPATH, thread pools capped at its cores."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
        env[var] = str(job.cores)
    env.update(job.env)
    return env


class Execution(threading.Thread):
    """Runs one job in its own process group and measures it."""

    def __init__(self, job: Job, runs_dir: Path):
        super().__init__(name=f"job:{job.id}", daemon=True)
        self.job = job
        self.runs_dir = Path(runs_dir)
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
        import psutil

        job = self.job
        for p in job.outputs:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            if Path(p).exists():
                Path(p).unlink()
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
            out = open(job.stdout, "wb") if job.stdout else logf
            try:
                t0 = time.perf_counter()
                self.proc = subprocess.Popen(
                    job.command,
                    cwd=paths.ROOT,
                    env=job_environment(job),
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=logf,
                    start_new_session=True,
                )
                try:
                    ps = psutil.Process(self.proc.pid)
                except psutil.Error:
                    ps = None
                peak = 0
                while True:
                    pid, status, rusage = os.wait4(self.proc.pid, os.WNOHANG)
                    if pid != 0:
                        break
                    if ps is not None:
                        peak = max(peak, tree_rss(ps))
                    time.sleep(SAMPLE_SECONDS)
                wall = time.perf_counter() - t0
            finally:
                if job.stdout:
                    out.close()
            exit_code = os.waitstatus_to_exitcode(status)
            self.proc.returncode = exit_code
            logf.write(f"\n# exit {exit_code} after {wall:.1f} s\n".encode())
        maxrss = rusage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)  # MiB
        run = {
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_wall)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": round(wall, 3),
            "peak_rss_mb": round(max(peak / 2**20, maxrss), 1),
            "peak_rss_tree_sampled_mb": round(peak / 2**20, 1),
            "maxrss_main_mb": round(maxrss, 1),
            "user_seconds": round(rusage.ru_utime, 3),
            "system_seconds": round(rusage.ru_stime, 3),
            "exit_status": exit_code,
            "loadavg_before": [round(x, 2) for x in load_before],
            "loadavg_after": [round(x, 2) for x in os.getloadavg()],
            "cores": job.cores,
            "mem_gb": job.mem_gb,
            "exclusive": job.exclusive,
        }
        if busiest is not None:
            run["busiest_processes_before"] = busiest
        if self._kill.is_set():
            return {"status": "interrupted", "run": run}
        if exit_code != 0:
            return {"status": FAILED, "run": run, "error": f"exit status {exit_code}", "log_tail": tail(log)}
        missing = [str(p) for p in job.outputs if not Path(p).exists()]
        if missing:
            return {"status": FAILED, "run": run, "error": f"declared outputs missing: {missing}", "log_tail": tail(log)}
        return {"status": DONE, "run": run}


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
    if job.cores > budget.cores:
        return f"needs {job.cores} cores, budget {budget.cores}"
    if job.mem_gb > budget.mem_gb:
        return f"needs {job.mem_gb:g} GB, budget {budget.mem_gb:g} GB"
    return None


def pick(ready: Sequence[Job], running: Sequence[Slot], budget: config.Budget, priority: Dict[str, float], now: float) -> List[Job]:
    """The ready jobs to start now.

    Non-exclusive jobs start in priority order (longest chain of dependent
    work first) while they fit the free cores and memory. When the first one
    does not fit, later ones may start only if they are expected to finish
    before enough running work ends for it ("backfilling"), so a large job is
    not starved by small ones. An exclusive job starts only when nothing
    runs and no non-exclusive job is ready, and then alone.
    """
    if any(s.job.exclusive for s in running):
        return []
    order = sorted(ready, key=lambda j: (-priority.get(j.id, 0.0), j.id))
    normal = [j for j in order if not j.exclusive]
    exclusive = [j for j in order if j.exclusive]
    free_cores = budget.cores - sum(s.job.cores for s in running)
    free_mem = budget.mem_gb - sum(s.job.mem_gb for s in running)
    chosen: List[Job] = []
    head_blocked_until: Optional[float] = None
    for j in normal:
        fits = j.cores <= free_cores and j.mem_gb <= free_mem + 1e-9
        if head_blocked_until is None:
            if fits:
                chosen.append(j)
                free_cores -= j.cores
                free_mem -= j.mem_gb
                continue
            ends = sorted(s.remaining(now) for s in running)
            head_blocked_until = ends[0] if ends else 0.0
            continue
        if fits and j.est_seconds <= head_blocked_until:
            chosen.append(j)
            free_cores -= j.cores
            free_mem -= j.mem_gb
    if not chosen and not running and not normal and exclusive:
        chosen.append(exclusive[0])
    return chosen


def simulate_schedule(
    graph: Graph, todo: Sequence[str], budget: config.Budget, running: Dict[str, float]
) -> Dict[str, float]:
    """Expected finish time (seconds from now) of each job in ``todo``.

    ``running`` maps running job ids to their expected remaining seconds.
    Jobs not in ``todo`` or ``running`` count as done. Uses :func:`pick`.
    """
    todo_set = set(todo)
    now = 0.0
    slots: List[Slot] = []
    finish: Dict[str, float] = {}
    for jid, rem in running.items():
        job = graph.jobs[jid]
        slots.append(Slot(job, now - (job.est_seconds - rem)))
    pending = [jid for jid in graph.order if jid in todo_set and jid not in running]
    finished = {jid for jid in graph.jobs if jid not in todo_set and jid not in running}
    guard = 0
    while pending or slots:
        guard += 1
        if guard > 100000:
            break
        ready = [graph.jobs[j] for j in pending if all(d in finished for d in graph.deps[j]) and fits_budget(graph.jobs[j], budget) is None]
        for j in pick(ready, slots, budget, graph.priority, now):
            slots.append(Slot(j, now))
            pending.remove(j.id)
        if not slots:
            for j in pending:
                finish[j] = math.inf
            break
        nxt = min(slots, key=lambda s: s.started + s.job.est_seconds)
        now = max(now, nxt.started + nxt.job.est_seconds)
        for s in [s for s in slots if s.started + s.job.est_seconds <= now + 1e-9]:
            slots.remove(s)
            finished.add(s.job.id)
            finish[s.job.id] = now
    return finish


# ---------------------------------------------------------------------------
# The daemon
# ---------------------------------------------------------------------------


def runner_dir(runs_dir: Path) -> Path:
    return Path(runs_dir) / "_runner"


class Lock:
    """An exclusive ``flock`` on ``<runs>/_runner/runner.lock``, held by the daemon while it lives."""

    def __init__(self, runs_dir: Path):
        self.path = runner_dir(runs_dir) / "runner.lock"
        self.fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            raise
        self.fd = fd
        return True

    def release(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


def daemon_alive(runs_dir: Path) -> Optional[int]:
    """The PID of the running daemon of ``runs_dir``, or None."""
    lock = Lock(runs_dir)
    if lock.acquire():
        lock.release()
        return None
    try:
        return int((runner_dir(runs_dir) / "runner.pid").read_text().strip())
    except (OSError, ValueError):
        return -1


class Logger:
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


class Daemon:
    """The scheduling loop. Runs in the foreground of whatever process calls :meth:`run`."""

    def __init__(self, graph: Graph, runs_dir: Path, budget: config.Budget, log: Logger, meta: Dict):
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
                bad = [d for d in self.graph.deps[jid] if self.state[d] in (FAILED, BLOCKED)]
                self.reason[jid] = f"needs {bad[0]}"
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
        running = {
            jid: {
                "pid": ex.proc.pid if ex.proc else None,
                "start": ex.started,
                "cores": ex.job.cores,
                "mem_gb": ex.job.mem_gb,
                "est_seconds": ex.job.est_seconds,
            }
            for jid, ex in self.running.items()
        }
        provenance.write_json_atomic(
            runner_dir(self.runs_dir) / "state.json",
            {
                **self.meta,
                "pid": os.getpid(),
                "updated": time.time(),
                "draining": self.drain.is_set(),
                "running": running,
                "failed": {j: self.reason.get(j, "") for j, s in self.state.items() if s == FAILED},
                "blocked": {j: self.reason.get(j, "") for j, s in self.state.items() if s == BLOCKED},
            },
        )

    # -- execution -----------------------------------------------------------
    def launch(self, job: Job) -> None:
        self.keys[job.id] = provenance.job_key(job, self.runs_dir, self.cache)
        ex = Execution(job, self.runs_dir)
        self.running[job.id] = ex
        self.state[job.id] = RUNNING
        self.log(f"start {job.id} ({self.reason.get(job.id, '')}): {' '.join(job.command)}", job.experiment)
        ex.start()

    def reap(self) -> bool:
        changed = False
        for jid, ex in list(self.running.items()):
            if ex.is_alive():
                continue
            changed = True
            del self.running[jid]
            job = ex.job
            res = ex.result or {"status": FAILED, "error": "no result"}
            status = res.get("status")
            if status == DONE:
                try:
                    provenance.write_record(job, self.runs_dir, self.cache, self.keys[jid], res["run"])
                    self.cache.save()
                except Exception as exc:  # noqa: BLE001
                    res = {**res, "status": FAILED, "error": f"recording provenance failed: {exc}"}
                    status = FAILED
            append_attempt(self.runs_dir, job, res)
            run = res.get("run", {})
            if status == DONE:
                self.state[jid] = DONE
                self.log(f"done {jid} in {run.get('wall_seconds', 0):.1f} s, peak RSS {run.get('peak_rss_mb', 0):.0f} MB", job.experiment)
            elif status == "interrupted":
                self.state[jid] = PENDING
                self.log(f"interrupted {jid}", job.experiment)
            else:
                self.state[jid] = FAILED
                self.reason[jid] = res.get("error", "failed")
                self.log(f"FAILED {jid}: {self.reason[jid]} (log: {job.log_path(self.runs_dir)})", job.experiment)
        return changed

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
        last_write = time.time()
        while True:
            changed = self.reap()
            if self.stop_now.is_set():
                self.log("stop requested: terminating running jobs")
                self.terminate_all()
                break
            if changed:
                self.refresh()
            started = False
            if not self.drain.is_set():
                ready = [self.graph.jobs[j] for j in self.graph.order if self.state[j] == READY]
                slots = [Slot(ex.job, ex.started) for ex in self.running.values()]
                for job in pick(ready, slots, self.budget, self.graph.priority, time.time()):
                    self.launch(job)
                    started = True
            # Rewrite the state when it changed, once the processes of new jobs
            # exist (their PIDs let a later runner stop them if this one is
            # killed), and every 10 s.
            pids = {jid: (ex.proc.pid if ex.proc else None) for jid, ex in self.running.items()}
            if changed or started or pids != recorded_pids or time.time() - last_write > 10:
                self.write_state()
                recorded_pids, last_write = pids, time.time()
            if not self.running:
                if self.drain.is_set() or not any(s == READY for s in self.state.values()):
                    break
            time.sleep(POLL_SECONDS)
        self.cache.save()
        self.write_state()
        counts = {s: sum(1 for v in self.state.values() if v == s) for s in (DONE, FAILED, BLOCKED, PENDING, READY)}
        self.log("runner finished: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
        return 1 if counts[FAILED] or counts[BLOCKED] else 0


def daemon_main(graph_factory: Callable[[], Graph], runs_dir: Path, budget: config.Budget, meta: Dict) -> int:
    """The daemon process: lock, PID file, signals, then :meth:`Daemon.run`."""
    runs_dir = Path(runs_dir)
    lock = Lock(runs_dir)
    if not lock.acquire():
        print(f"another runner holds {lock.path}", file=sys.stderr)
        return 3
    pid_file = runner_dir(runs_dir) / "runner.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    log = Logger(runs_dir)
    daemon: Optional[Daemon] = None

    def on_term(signum, frame):
        if daemon is not None:
            daemon.stop_now.set()

    def on_drain(signum, frame):
        if daemon is not None:
            daemon.drain.set()
            log("drain requested: no new jobs start")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    signal.signal(signal.SIGUSR1, on_drain)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        log(f"runner {os.getpid()} starting ({meta})")
        kill_orphans(runs_dir, log)
        graph = graph_factory()
        meta = {**meta, "experiments": graph.experiments(), "machine": provenance.machine()}
        daemon = Daemon(graph, runs_dir, budget, log, meta)
        return daemon.run()
    except Exception:
        log("runner crashed:\n" + traceback.format_exc())
        return 2
    finally:
        try:
            pid_file.unlink()
        except OSError:
            pass
        lock.release()


def kill_orphans(runs_dir: Path, log: Callable[[str], None]) -> None:
    """Stop the jobs of an earlier runner that died without stopping them (e.g. SIGKILL).

    They are the process groups recorded as running in ``state.json`` whose
    leader is still the process started then (same PID, started no later
    than 5 s after the job, leading its own process group).
    """
    import psutil

    try:
        state = json.loads((runner_dir(runs_dir) / "state.json").read_text())
    except (OSError, ValueError):
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
        except ProcessLookupError:
            pass


def spawn_daemon(argv: Sequence[str], runs_dir: Path, wait: float = 60.0) -> int:
    """Start ``argv`` (the daemon command) detached; return its PID once it holds the lock."""
    runs_dir = Path(runs_dir)
    rdir = runner_dir(runs_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    out = open(rdir / "daemon.out", "ab")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.Popen(
        list(argv),
        cwd=paths.ROOT,
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        close_fds=True,
    )
    out.close()
    deadline = time.time() + wait
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the runner exited at once (status {proc.returncode}); see {rdir / 'daemon.out'}")
        if daemon_alive(runs_dir) == proc.pid:
            return proc.pid
        time.sleep(0.1)
    return proc.pid


def stop_daemon(runs_dir: Path, drain: bool = False, wait: float = 60.0) -> Optional[int]:
    """Ask the daemon to stop (SIGTERM) or drain (SIGUSR1); wait for it to exit unless draining."""
    pid = daemon_alive(runs_dir)
    if pid is None or pid <= 0:
        return None
    os.kill(pid, signal.SIGUSR1 if drain else signal.SIGTERM)
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
    state: Dict = {}
    if alive:
        try:
            state = json.loads((runner_dir(runs_dir) / "state.json").read_text())
        except (OSError, ValueError):
            state = {}
    running = state.get("running", {}) if alive else {}
    attempts = last_attempts(runs_dir, graph.experiments())
    out: List[JobStatus] = []
    for jid in graph.order:
        job = graph.jobs[jid]
        if jid in running:
            out.append(JobStatus(job, RUNNING, elapsed=time.time() - running[jid]["start"]))
            continue
        missing = [p for p in job.inputs if not Path(p).exists()]
        if missing:
            done, why = False, f"waiting for {Path(missing[0]).name}"
        else:
            done, why = provenance.check_done(job, runs_dir, cache)
        if done:
            out.append(JobStatus(job, DONE))
            continue
        last = attempts.get(jid)
        if last and last.get("status") == FAILED:
            out.append(JobStatus(job, FAILED, last.get("error", "")))
        elif alive and jid in state.get("blocked", {}):
            out.append(JobStatus(job, BLOCKED, state["blocked"][jid]))
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
        out[exp] = {
            "jobs": len(jobs),
            "cpu_hours": sum(j.est_seconds * j.cores for j in jobs) / 3600,
            "wall_hours_alone": 0.0,
            "finish_hours": max(finish[j.id] for j in jobs) / 3600,
        }
        sub = Graph(jobs, graph.runs_dir)
        alone = simulate_schedule(sub, list(sub.jobs), budget, {})
        out[exp]["wall_hours_alone"] = max(alone.values()) / 3600
    return out
