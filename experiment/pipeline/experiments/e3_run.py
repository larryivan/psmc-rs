"""The commands that E3's jobs run: simulate the genome of one size, and time one run of a tool.

``python -m experiment.pipeline.experiments.e3_run simulate --profile P --size-mb S --out-dir D --truth T``
    simulates the E3 genome of S Mb (``config.E3``: chromosomes of
    ``chromosome_length`` under ``model``, seeds
    ``seed("E3", model, replicate, "ancestry/chrN")``) and writes its psmcfa
    and truth record, as :func:`.simulate.simulate_genome` does (no VCF).

``python -m experiment.pipeline.experiments.e3_run measure --out TIMING.json [options] -- TOOL ARGS...``
    runs ``TOOL ARGS...`` and writes how it ran to TIMING.json. With
    ``--threads N`` it first refuses (exit status 69) to time a tool of N
    threads on fewer CPUs than N, which would time it oversubscribed: the
    CPUs this process may run on (its affinity: a SLURM allocation's).
    Before the run it waits ``--settle`` seconds, then until the machine is
    quiet: the CPU use of the whole machine (everything but idle and iowait
    time, so kernel threads, interrupts and steal too) over
    ``--quiet-window`` seconds is at most ``--quiet-cpu`` percent of all
    cores and at most ``--quiet-max-cores`` cores; nothing of ours runs
    then (it reads the processes outside that window), so all of it is
    other work. It then starts the tool, with the environment cleaned of
    variables that change psmc-rs's behaviour or build (``PSMC_*``,
    ``RUSTFLAGS``, ...), and samples the RSS of the tool's process tree
    every ``--sample`` seconds; the peak RSS is the larger of the sampled
    peak and the tool's own ``ru_maxrss``. The record holds the wall time,
    the tool's user and system CPU time, the peak RSS, the load average
    before and after, what kept the machine busy before the run, the CPU
    time other processes used while it ran (see below), the power source,
    the environment and the machine (on Linux also its CPU model, frequency
    governor and boost, and the SLURM job). When the tool fails, nothing is
    written and the command exits with the tool's status. When the run is
    not clean (the machine did not become quiet within ``--max-wait``
    seconds, is on battery or in Low Power Mode with ``--require-ac``, slept
    during the run, or other processes used more than ``--max-other-cpu`` of
    all cores or more than ``--max-other-cores`` cores on average), nothing
    is written and the command exits with status 75, which the runner takes
    as "not now": it starts the run again later.

What "other processes" means during a run (``during.other_method``):

``processes`` (Linux, where every process's ``/proc/<pid>/stat`` is
    readable): the CPU time (user + system) of the user-space processes
    other than the tool's process tree and this wrapper, read for every
    process just before the tool starts and right after it ends
    (:func:`process_times`) and matched by pid and creation time: the
    increase of each process seen both times, the whole CPU time of one
    that started during the run, each with the CPU time of the children it
    reaped meanwhile (Linux's cutime and cstime). So a process that ended
    during the run (``other_processes_exited`` counts those seen before it)
    or started and ended in it (a ``make -j``'s compilers, a script's
    commands) is counted through the process that waited for it, its
    parent or, orphaned, init; one that had started before the run counts
    whole, its CPU before the run too, which errs towards rejecting a run.
    Only a process that no running process waits for (whose parent ignores
    SIGCHLD) is missed: its CPU shows only in ``unaccounted_cores``, the
    machine's busy CPU that nothing here explains, which is recorded but
    not limited, since it also holds the differences between how the
    kernel counts the machine's CPU time and the processes'. Time that a
    hypervisor gave to other machines (``steal_cores``, 0 on bare metal) is
    other work too, and is limited with the processes. Kernel threads
    (kthreadd and its children) and the kernel's interrupt time (irq and
    softirq) are not other processes. They are the kernel's work for the
    run itself (a tool that keeps 30 cores busy makes the kernel handle its
    page faults, timers, scheduling and I/O: about 0.1 core more than on
    the idle 128-core node) and the machine's own housekeeping, which the
    quiet check before the run bounds; neither is another workload, and the
    two cannot be told apart. Counted as other work, the run's own share
    would reject the runs that keep the most cores busy. They are recorded
    apart (``kernel_thread_cores``, ``interrupt_cores``).
``machine`` (where some process cannot be read, as other users' processes
    on macOS, or ``/proc`` hides them, ``hidepid``; and wherever reaped
    children's CPU time cannot be read: every system but Linux): all busy
    CPU time of the machine but the tool's, as before the run. Counting
    only the processes it could read would under-count, so the record says
    which measure the limits were checked against.

The machine-wide measure is recorded in either case (``other_cpu_seconds``,
``other_cpu_share``, ``other_cores``, ``machine_busy_percent``), so records
of either kind stay comparable with older ones (``timing/3``, which have
only it); :func:`other_cores` and :func:`other_share` give the measure the
limits apply to.

Only the tool's processes are measured, not this wrapper (the runner's own
record of the job includes the wrapper's Python interpreter). The module is
kept apart from :mod:`.e3` (jobs and tables) because its SHA-256 is part of
the provenance key of every timing job: editing the tables must not
invalidate hours of timing runs, while changing how runs are timed must.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, paths, provenance, simulate

RECORD = "psmc-rs-manuscript/E3/timing/4"
EX_TEMPFAIL = 75  # not a clean run: the runner defers the job
EX_UNAVAILABLE = 69  # too few CPUs for the tool's threads: a failure, which the next start retries
# Variables removed from the tool's environment: psmc-rs reads PSMC_*
# (e.g. psmc-rs 0.1's PSMC_ALPHA_CACHE_MB, which changes its memory and
# time), and the others change a build or what a tool prints.
CLEANED_PREFIXES = ("PSMC_", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUST_LOG", "RUST_BACKTRACE")


# ---------------------------------------------------------------------------
# Genomes
# ---------------------------------------------------------------------------


def genome_for(cfg: config.E3, size_mb: int) -> config.Genome:
    """The genome of ``size_mb`` Mb: whole chromosomes of ``cfg.chromosome_length``."""
    total = int(size_mb) * 1_000_000
    if size_mb <= 0 or total % cfg.chromosome_length:
        raise ValueError(f"{size_mb} Mb is not a whole number of {cfg.chromosome_length} bp chromosomes")
    return config.Genome(n_chromosomes=total // cfg.chromosome_length, chromosome_length=cfg.chromosome_length)


def simulate_size(profile: config.Profile, size_mb: int, out_dir: Path, truth: Path, log=print) -> Dict:
    """Simulate the E3 genome of ``size_mb`` Mb; returns its truth record."""
    cfg = profile.e3
    return simulate.simulate_genome(
        "E3",
        config.MODELS[cfg.model],
        cfg.replicate,
        genome_for(cfg, size_mb),
        cfg.simulation,
        Path(out_dir),
        Path(truth),
        write_vcf=False,
        log=log,
    )


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


def usable_cpus() -> int:
    """The CPUs this process may run on: its affinity (a SLURM allocation's, taskset's) where there is one, else all."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def busy_seconds(times) -> float:
    """CPU seconds not spent idle in a ``psutil.cpu_times()`` result (all cores together)."""
    idle = times.idle + getattr(times, "iowait", 0.0)
    return sum(times) - idle


_DENIED = object()  # psutil's value for what this process may not read


def is_kernel_thread(pid: int, ppid: Optional[int], platform: str = sys.platform) -> bool:
    """Whether a process is the kernel's: kernel_task (pid 0) on macOS; on Linux kthreadd (pid 2) and its children.

    Every Linux kernel thread is started by kthreadd. (A process without a
    command line is not necessarily one: a zombie has none either.)
    """
    if platform.startswith("linux"):
        return pid == 2 or ppid == 2
    return pid == 0


def process_times(platform: str = sys.platform) -> Tuple[Dict[Tuple[int, float], Tuple[float, float, bool, str]], int]:
    """Every process's CPU seconds so far, by pid and creation time, and how many processes could not be read.

    Each value is (user + system CPU seconds, those of the children it has
    waited for, kernel thread, name). The children's (Linux's cutime and
    cstime; 0 elsewhere) hold every process that ended and was reaped by
    this one, with the children they had reaped in turn. A process this one
    may not read (AccessDenied: other users' processes on macOS) is counted,
    not listed; one that ends while it is being read is neither.
    """
    import psutil

    procs: Dict[Tuple[int, float], Tuple[float, float, bool, str]] = {}
    unreadable = 0
    for p in psutil.process_iter(["ppid", "name", "create_time", "cpu_times"], ad_value=_DENIED):
        info = p.info
        if any(info[k] is _DENIED or info[k] is None for k in ("ppid", "create_time", "cpu_times")):
            unreadable += 1
            continue
        t = info["cpu_times"]
        name = "?" if info["name"] is _DENIED else str(info["name"])
        reaped = getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)
        procs[(p.pid, info["create_time"])] = (t.user + t.system, reaped, is_kernel_thread(p.pid, info["ppid"], platform),
                                               name)
    return procs, unreadable


def processes_hidden(mounts: Path = Path("/proc/mounts"), platform: str = sys.platform) -> bool:
    """Whether Linux's /proc hides other users' processes from this one (mounted ``hidepid``, and not root)."""
    if not platform.startswith("linux") or os.geteuid() == 0:
        return False
    for line in (_read_text(mounts) or "").splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "/proc":
            opts = dict(o.partition("=")[::2] for o in fields[3].split(","))
            return opts.get("hidepid", "0") not in ("0", "off")
    return False


def process_method(*unreadable: int, platform: str = sys.platform) -> str:
    """``processes`` when every process could be read (no :func:`process_times` count above 0, none hidden), else ``machine``.

    Only on Linux: elsewhere (macOS) a process's reaped children's CPU time
    is not readable, and the processes that ended during a run would be
    missed.
    """
    if not platform.startswith("linux") or any(unreadable) or processes_hidden(platform=platform):
        return "machine"
    return "processes"


def _delta(cpu0, cpu1, *fields: str) -> Optional[float]:
    """The increase of these ``psutil.cpu_times()`` fields together; None where it has none of them (macOS)."""
    have = [f for f in fields if hasattr(cpu1, f)]
    return sum(getattr(cpu1, f) - getattr(cpu0, f) for f in have) if have else None


def cpu_breakdown(cpu0, cpu1, procs0: Dict, procs1: Dict, exclude=()) -> Dict:
    """What kept the CPUs busy between two looks at the machine (``psutil.cpu_times()`` and :func:`process_times`).

    CPU seconds of: ``processes``, the user-space processes (the increase of
    each one seen both times, the same pid and creation time, and of its
    reaped children's; the whole of both for one that started in between);
    ``kernel_threads``, the same for the kernel's threads; ``excluded``, the
    own CPU of the pids in ``exclude``, which neither of those counts (not
    their reaped children's: the timed tool is this wrapper's);
    ``interrupts``, the machine's irq and softirq time, and ``steal``, time
    a hypervisor gave to other machines (None where psutil has neither).
    A process that ended in between is counted through the one that reaped
    it, its parent (or init, if it was orphaned), with its whole CPU time:
    for one that had started before, also what it used before, which errs
    towards counting too much. ``exited``: how many user-space processes
    seen the first time were gone the second. ``busiest``: (CPU seconds,
    pid, name, kernel thread) of every process seen the second time, most
    first.
    """
    out: Dict = {"processes": 0.0, "kernel_threads": 0.0, "excluded": 0.0, "busiest": []}
    for key, (cpu, reaped, kernel, name) in procs1.items():
        prev = procs0.get(key) or (0.0, 0.0)
        own = max(cpu - prev[0], 0.0)
        if key[0] in exclude:
            out["excluded"] += own
            continue
        used = own + max(reaped - prev[1], 0.0)
        out["kernel_threads" if kernel else "processes"] += used
        out["busiest"].append((used, key[0], name, kernel))
    out["busiest"].sort(key=lambda u: -u[0])
    out["exited"] = sum(1 for key, (_, _, kernel, _) in procs0.items()
                        if key not in procs1 and key[0] not in exclude and not kernel)
    out["interrupts"] = _delta(cpu0, cpu1, "irq", "softirq")
    out["steal"] = _delta(cpu0, cpu1, "steal")
    return out


def _per(seconds: Optional[float], span: float, digits: int = 3) -> Optional[float]:
    """CPU seconds over ``span`` seconds: the cores they kept busy on average (None stays None)."""
    if seconds is None:
        return None
    return round(seconds / span, digits) if span > 0 else 0.0


def machine_activity(window: float, top: int = 5) -> Dict:
    """The CPU use of the whole machine over ``window`` seconds, what it was, and the busiest processes.

    ``busy_percent`` is the share of all cores in use (everything but idle
    and iowait time; 100 = every core busy) and ``busy_cores`` the same as a
    number of cores: what the quiet check limits. Of it, as cores
    (:func:`cpu_breakdown`): ``process_cores`` (user-space processes, this
    one excluded), ``kernel_thread_cores``, ``interrupt_cores`` and
    ``steal_cores``; the first two are None when some process could not be
    read (``processes_unreadable`` of them, or /proc hides them, or not
    Linux), since they would under-count. ``busiest`` lists the processes
    with the most CPU in the window (``cpu_percent``: 100 = one core),
    kernel threads included, this process excluded.

    The two reads of every process fall outside the window of the machine's
    counters, so that their CPU (this process's) is not in ``busy_*``; each
    process's two reads are about the window and one read of them all apart.
    """
    import psutil

    me = os.getpid()
    procs0, unread0 = process_times()
    t0 = time.perf_counter()
    before = psutil.cpu_times()
    time.sleep(max(window, 0.0))
    after = psutil.cpu_times()
    elapsed = time.perf_counter() - t0
    procs1, unread1 = process_times()
    span = time.perf_counter() - t0
    total = sum(after) - sum(before)
    busy = busy_seconds(after) - busy_seconds(before)
    share = busy / total if total > 0 else 0.0
    parts = cpu_breakdown(before, after, procs0, procs1, exclude={me})
    complete = process_method(unread0, unread1) == "processes"
    return {
        "window_seconds": round(elapsed, 3),
        "busy_percent": round(100.0 * share, 2),
        "busy_cores": round(share * (psutil.cpu_count() or 1), 3),
        "process_cores": _per(parts["processes"], span) if complete else None,
        "kernel_thread_cores": _per(parts["kernel_threads"], span) if complete else None,
        "interrupt_cores": _per(parts["interrupts"], elapsed),
        "steal_cores": _per(parts["steal"], elapsed),
        "processes_unreadable": max(unread0, unread1),
        "busiest": [{"pid": pid, "name": name, "cpu_percent": round(100.0 * used / span, 1) if span > 0 else 0.0}
                    for used, pid, name, _ in parts["busiest"][:top]],
    }


def power_state() -> Dict:
    """The power source (``AC Power`` or ``Battery Power``) and Low Power Mode on macOS; Nones elsewhere."""
    out: Dict = {"source": None, "low_power_mode": None}
    if sys.platform != "darwin":
        return out
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    m = re.search(r"drawing from '([^']+)'", batt)
    out["source"] = m.group(1) if m else None
    m = re.search(r"^\s*lowpowermode\s+(\d)", settings, re.M)
    out["low_power_mode"] = (m.group(1) == "1") if m else None
    return out


def clean_environment(env: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], List[str]]:
    """The tool's environment (:data:`CLEANED_PREFIXES` removed) and the names removed."""
    env = dict(os.environ if env is None else env)
    removed = sorted(k for k in env if k.startswith(CLEANED_PREFIXES))
    for k in removed:
        del env[k]
    return env, removed


def wait_until_quiet(protocol: config.TimingProtocol, log=print) -> Dict:
    """Settle, then wait for a quiet machine (see the module documentation); returns what was seen.

    Quiet is judged on the whole machine's busy CPU (``busy_percent``,
    ``busy_cores``): before the run nothing of ours runs, so kernel threads
    and interrupts are other work too. What it was made of is recorded with
    it (:func:`machine_activity`).
    """
    time.sleep(max(protocol.settle_seconds, 0.0))
    t0 = time.monotonic()
    while True:
        act = machine_activity(protocol.quiet_window_seconds)
        waited = time.monotonic() - t0
        quiet = act["busy_percent"] <= protocol.quiet_cpu_percent and act["busy_cores"] <= protocol.quiet_max_cores
        if quiet or waited >= protocol.max_wait_seconds:
            break
        log(f"[measure] machine busy ({act['busy_percent']:.0f}% of all cores, {act['busy_cores']:.1f} cores); waiting")
    if not quiet:
        log(f"[measure] machine not quiet after {waited:.0f} s ({act['busy_percent']:.0f}% busy, "
            f"{act['busy_cores']:.1f} cores)")
    return {
        "quiet": quiet,
        "busy_percent": act["busy_percent"],
        "busy_cores": act["busy_cores"],
        "threshold_percent": protocol.quiet_cpu_percent,
        "threshold_cores": protocol.quiet_max_cores,
        "window_seconds": act["window_seconds"],
        "settle_seconds": protocol.settle_seconds,
        "waited_seconds": round(waited, 3),
        # of busy_cores, for the report (the check is on the whole machine)
        **{k: act[k] for k in ("process_cores", "kernel_thread_cores", "interrupt_cores", "steal_cores",
                               "processes_unreadable")},
        "busiest": act["busiest"],
    }


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def linux_machine(root: Path = Path("/"), env: Optional[Dict[str, str]] = None) -> Dict:
    """What Linux tells of the CPU and the job, where it is readable (keys left out otherwise).

    ``cpu`` (``model name`` of /proc/cpuinfo), ``memory_gb``, the cpufreq
    ``cpu_governor`` (every core's, joined by "," when they differ) and
    ``cpu_scaling_driver`` of cpu0, ``cpu_boost`` ("1" on, "0" off: the
    ``cpufreq/boost`` switch, else the inverse of intel_pstate's
    ``no_turbo``), the cores this process may run on (``cpus_allowed``, a
    SLURM allocation's), the ``hostname`` and the SLURM job
    (``SLURM_JOB_ID``, ``SLURM_JOB_NODELIST``, ``SLURM_CPUS_ON_NODE``).
    """
    env = dict(os.environ if env is None else env)
    info: Dict = {}
    m = re.search(r"^model name\s*:\s*(.+)$", _read_text(root / "proc/cpuinfo") or "", re.M)
    if m:
        info["cpu"] = m.group(1).strip()
    m = re.search(r"^MemTotal:\s*(\d+)\s*kB", _read_text(root / "proc/meminfo") or "", re.M)
    if m:
        info["memory_gb"] = round(int(m.group(1)) / 2**20, 1)
    cpus = root / "sys/devices/system/cpu"
    governors = {g for g in (_read_text(p) for p in cpus.glob("cpu[0-9]*/cpufreq/scaling_governor")) if g}
    if governors:
        info["cpu_governor"] = ",".join(sorted(governors))
    driver = _read_text(cpus / "cpu0/cpufreq/scaling_driver")
    if driver:
        info["cpu_scaling_driver"] = driver
    boost, no_turbo = _read_text(cpus / "cpufreq/boost"), _read_text(cpus / "intel_pstate/no_turbo")
    if boost in ("0", "1"):
        info["cpu_boost"] = boost
    elif no_turbo in ("0", "1"):
        info["cpu_boost"] = "0" if no_turbo == "1" else "1"
    if hasattr(os, "sched_getaffinity") and root == Path("/"):
        info["cpus_allowed"] = len(os.sched_getaffinity(0))
    info["hostname"] = platform.node()
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_ON_NODE"):
        if env.get(key):
            info[key.lower()] = env[key]
    return info


def machine() -> Dict:
    """:func:`.provenance.machine`, with the performance and efficiency cores of Apple silicon, or :func:`linux_machine`."""
    info = provenance.machine()
    if sys.platform.startswith("linux"):
        info.update(linux_machine())
    if sys.platform == "darwin":
        for key, name in (("hw.perflevel0.physicalcpu", "performance_cores"), ("hw.perflevel1.physicalcpu", "efficiency_cores")):
            try:
                res = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if res.returncode == 0 and res.stdout.strip().isdigit():
                info[name] = int(res.stdout.strip())
    return info


def tree_rss(proc, pids: Optional[set] = None) -> int:
    """Resident set size of a psutil process and all its descendants, in bytes; their pids are added to ``pids``."""
    import psutil

    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    if pids is not None:
        pids.update(p.pid for p in procs)
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def maxrss_mb(rusage) -> float:
    """``ru_maxrss`` in MB (2**20 bytes): bytes on macOS, KiB on Linux."""
    return rusage.ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


def during_run(cpu0, cpu1, procs0: Dict, procs1: Dict, wall: float, tool_cpu: float, exclude, method: str,
               unreadable: int = 0, top: int = 5) -> Dict:
    """What else used the CPUs during a run of ``wall`` seconds whose tool used ``tool_cpu`` CPU seconds.

    From ``psutil.cpu_times()`` and :func:`process_times` just before and
    right after the run, ``exclude`` the pids of the tool's process tree
    and this wrapper; ``method`` (:func:`process_method`) says whether every
    process could be read. ``other_process_*`` (the CPU of other user-space
    processes: what the limits check with ``processes``, with the steal
    time) and ``kernel_thread_cores`` and ``unaccounted_cores`` are None
    with ``machine``, since they would under-count; ``other_cpu_*`` and
    ``other_cores`` (the machine's busy CPU but the tool's: what the limits
    check with ``machine``) are always given. See the module documentation.
    """
    machine_busy = busy_seconds(cpu1) - busy_seconds(cpu0)
    machine_total = sum(cpu1) - sum(cpu0)
    other = max(machine_busy - tool_cpu, 0.0)
    parts = cpu_breakdown(cpu0, cpu1, procs0, procs1, exclude)
    measured = method == "processes"
    explained = sum(parts[k] or 0.0 for k in ("processes", "kernel_threads", "excluded", "interrupts", "steal"))

    def share(seconds: float) -> float:
        return round(seconds / machine_total, 4) if machine_total > 0 else 0.0

    return {
        # the measure the limits check: "processes" (other_process_* and steal_*) or
        # "machine" (other_cpu_share, other_cores)
        "other_method": method,
        # CPU of the user-space processes other than the tool's tree and this wrapper
        "other_process_cpu_seconds": round(parts["processes"], 3) if measured else None,
        "other_process_cpu_share": share(parts["processes"]) if measured else None,
        "other_process_cores": _per(parts["processes"], wall, 4) if measured else None,
        "other_processes_exited": parts["exited"],  # ended during the run: counted through their reaper
        "processes_unreadable": unreadable,
        # not other processes: the kernel's work, mostly the run's own
        "kernel_thread_cores": _per(parts["kernel_threads"], wall, 4) if measured else None,
        "interrupt_cores": _per(parts["interrupts"], wall, 4),
        # other machines' (a hypervisor's): other work, limited with other_process_*
        "steal_cores": _per(parts["steal"], wall, 4),
        "steal_cpu_share": None if parts["steal"] is None else share(parts["steal"]),
        # this wrapper, and any process of the tool's tree still running after it
        "excluded_cores": _per(parts["excluded"], wall, 4) if measured else None,
        # the machine's busy CPU that none of these nor the tool explains (processes no
        # running process reaped, differences in how the kernel counts each)
        "unaccounted_cores": _per(machine_busy - tool_cpu - explained, wall, 4) if measured else None,
        "busiest_others": [{"pid": pid, "name": name, "cores": _per(used, wall, 4)}
                           for used, pid, name, kernel in parts["busiest"] if not kernel][:top],
        # machine-wide: all busy CPU time (everything but idle and iowait, so kernel
        # threads and interrupts too) but the tool's, as a share of every core over
        # the run and as the cores it kept busy on average; recorded with either method
        "other_cpu_seconds": round(other, 3),
        "other_cpu_share": share(other),
        "other_cores": round(other / wall, 4) if wall > 0 else 0.0,
        "machine_busy_percent": round(100.0 * machine_busy / machine_total, 2) if machine_total > 0 else 0.0,
    }


def run_measured(cmd: Sequence[str], sample_seconds: float, env: Optional[Dict[str, str]] = None) -> Dict:
    """Run ``cmd`` (its output goes to ours) and measure it; see the module documentation."""
    import psutil

    cpu_before = psutil.cpu_times()
    procs_before, unread_before = process_times()
    load_before = os.getloadavg()
    start = time.time()
    t0 = time.perf_counter()
    proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.DEVNULL, env=env)
    tree = {proc.pid}
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    peak, samples = 0, 0
    while True:
        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        if pid != 0:
            break
        if ps is not None:
            peak = max(peak, tree_rss(ps, tree))
            samples += 1
        time.sleep(sample_seconds)
    wall = time.perf_counter() - t0
    clock_span = time.time() - start  # runs on while the machine sleeps; perf_counter does not
    proc.returncode = os.waitstatus_to_exitcode(status)
    procs_after, unread_after = process_times()
    cpu_after = psutil.cpu_times()
    load_after = os.getloadavg()
    tool_cpu = rusage.ru_utime + rusage.ru_stime
    maxrss = maxrss_mb(rusage)
    return {
        "exit_status": proc.returncode,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(wall, 4),
        "clock_span_seconds": round(clock_span, 4),
        "user_seconds": round(rusage.ru_utime, 4),
        "system_seconds": round(rusage.ru_stime, 4),
        "cpu_seconds": round(tool_cpu, 4),
        "peak_rss_mb": round(max(peak / 2**20, maxrss), 2),
        "peak_rss_tree_sampled_mb": round(peak / 2**20, 2),
        "maxrss_mb": round(maxrss, 2),
        "rss_samples": samples,
        "loadavg_before": [round(x, 2) for x in load_before],
        "loadavg_after": [round(x, 2) for x in load_after],
        "during": during_run(cpu_before, cpu_after, procs_before, procs_after, wall, tool_cpu,
                             exclude=tree | {os.getpid()}, method=process_method(unread_before, unread_after),
                             unreadable=max(unread_before, unread_after)),
    }


def _log(msg: str) -> None:
    print(msg, flush=True)


def measure(
    cmd: Sequence[str],
    out: Path,
    protocol: config.TimingProtocol,
    runs_dir: Optional[Path] = None,
    log=_log,
    threads: Optional[int] = None,
) -> int:
    """Time one run of ``cmd`` and write the record to ``out``; returns the exit status for the shell.

    ``threads``: the tool's threads; with more than :func:`usable_cpus` the
    run is not timed (:data:`EX_UNAVAILABLE`).
    """
    if not cmd:
        raise ValueError("no command to measure")
    cpus = usable_cpus()
    if threads is not None and threads > cpus:
        log(f"[measure] {threads} threads on {cpus} usable CPUs would be timed oversubscribed; not run "
            "(give the runner a machine or allocation of more CPUs)")
        return EX_UNAVAILABLE
    power = power_state()
    if protocol.require_ac_power and (power["source"] not in (None, "AC Power") or power["low_power_mode"]):
        log(f"[measure] not on AC power or in Low Power Mode ({power}); not run, try later")
        return EX_TEMPFAIL
    before = wait_until_quiet(protocol, log=log)
    if not before["quiet"]:
        log("[measure] the machine is not quiet; not run, try later")
        return EX_TEMPFAIL
    env, removed = clean_environment()
    res = run_measured(cmd, protocol.rss_sample_seconds, env=env)
    status = res["exit_status"]
    if status != 0:
        log(f"[measure] {Path(cmd[0]).name} exited with status {status}; no timing record written")
        return status if 0 < status < 256 else 128 + abs(status)
    why = unclean(res, protocol)
    if why:
        log(f"[measure] {why}; no timing record written, try later")
        return EX_TEMPFAIL
    words = [str(c) for c in cmd]
    if runs_dir is not None:
        words = [paths.portable(w, Path(runs_dir)) for w in words]
    record = {
        "record": RECORD,
        "command": words,
        "threads": threads,
        "cpus_usable": cpus,
        **res,
        "machine_before": {**before, "power": power},
        "environment": {"removed": removed, **{k: env[k] for k in sorted(env) if k.endswith("_NUM_THREADS")}},
        "protocol": asdict(protocol),
        "machine": machine(),
    }
    provenance.write_json_atomic(Path(out), record)
    log(
        f"[measure] {Path(cmd[0]).name}: {res['wall_seconds']:.2f} s wall, {res['cpu_seconds']:.2f} s CPU, "
        f"peak RSS {res['peak_rss_mb']:.1f} MB"
    )
    return 0


def other_method(res: Dict) -> str:
    """Which measure of other work a run's limits check: ``processes`` or ``machine`` (records before ``timing/4``)."""
    return res.get("during", {}).get("other_method", "machine")


def machine_other_cores(res: Dict) -> float:
    """The cores the machine's busy CPU but the tool's kept busy on average (from their CPU seconds in older records)."""
    during = res.get("during", {})
    if "other_cores" in during:
        return float(during["other_cores"])
    wall = res.get("wall_seconds") or 0.0
    return float(during.get("other_cpu_seconds") or 0.0) / wall if wall > 0 else 0.0


def other_cores(res: Dict) -> float:
    """The cores other work kept busy on average during a run, by its :func:`other_method`: what ``max_other_cores`` limits.

    With ``processes``: other user-space processes and steal time.
    """
    if other_method(res) == "processes":
        during = res["during"]
        return float(during["other_process_cores"]) + float(during.get("steal_cores") or 0.0)
    return machine_other_cores(res)


def other_share(res: Dict) -> float:
    """The share of all cores other work used during a run, by its :func:`other_method`: what ``max_other_cpu_share`` limits."""
    during = res.get("during", {})
    if other_method(res) == "processes":
        return float(during["other_process_cpu_share"]) + float(during.get("steal_cpu_share") or 0.0)
    return float(during.get("other_cpu_share") or 0.0)


def unclean(res: Dict, protocol: config.TimingProtocol) -> Optional[str]:
    """Why a finished run is not a clean measurement, or None."""
    who = ("other user-space processes and steal time" if other_method(res) == "processes"
           else "other work (the machine's busy CPU but the tool's)")
    share = other_share(res)
    if share > protocol.max_other_cpu_share:
        return (f"{who} used {100 * share:.1f}% of all cores during the run "
                f"(at most {100 * protocol.max_other_cpu_share:g}%)")
    cores = other_cores(res)
    if cores > protocol.max_other_cores:
        return f"{who} kept {cores:.2f} cores busy during the run (at most {protocol.max_other_cores:g})"
    slept = res["clock_span_seconds"] - res["wall_seconds"]
    if slept > protocol.max_sleep_seconds:
        return f"the machine slept for about {slept:.0f} s during the run"
    return None


def measure_args(protocol: config.TimingProtocol) -> List[str]:
    """The ``measure`` options of ``protocol``, each given explicitly."""
    return [
        "--settle", f"{protocol.settle_seconds:g}",
        "--quiet-cpu", f"{protocol.quiet_cpu_percent:g}",
        "--quiet-max-cores", f"{protocol.quiet_max_cores:g}",
        "--quiet-window", f"{protocol.quiet_window_seconds:g}",
        "--max-wait", f"{protocol.max_wait_seconds:g}",
        "--sample", f"{protocol.rss_sample_seconds:g}",
        "--max-other-cpu", f"{protocol.max_other_cpu_share:g}",
        "--max-other-cores", f"{protocol.max_other_cores:g}",
        "--max-sleep", f"{protocol.max_sleep_seconds:g}",
        "--require-ac" if protocol.require_ac_power else "--no-require-ac",
    ]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.experiments.e3_run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("simulate", help="simulate the E3 genome of one size (psmcfa and truth record)")
    s.add_argument("--profile", default="production", choices=sorted(config.PROFILES))
    s.add_argument("--size-mb", type=int, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--truth", type=Path, required=True)
    d = config.TimingProtocol()
    m = sub.add_parser("measure", help="time one run of a tool: measure [options] -- TOOL ARGS...")
    m.add_argument("--out", type=Path, required=True, help="the timing record (JSON) to write")
    m.add_argument("--runs-dir", type=Path, help="runs directory, written as <runs> in the record")
    m.add_argument("--threads", type=int, help="the tool's threads: not run on fewer usable CPUs")
    m.add_argument("--settle", type=float, default=d.settle_seconds, help="seconds to wait first")
    m.add_argument("--quiet-cpu", type=float, default=d.quiet_cpu_percent, help="quiet: CPU use at most this %% of all cores")
    m.add_argument("--quiet-max-cores", type=float, default=d.quiet_max_cores, help="quiet: and at most this many cores")
    m.add_argument("--quiet-window", type=float, default=d.quiet_window_seconds, help="seconds over which CPU use is measured")
    m.add_argument("--max-wait", type=float, default=d.max_wait_seconds, help="run anyway after waiting this long")
    m.add_argument("--sample", type=float, default=d.rss_sample_seconds, help="seconds between RSS samples")
    m.add_argument("--max-other-cpu", type=float, default=d.max_other_cpu_share,
                   help="clean: other processes used at most this share of all cores during the run")
    m.add_argument("--max-other-cores", type=float, default=d.max_other_cores,
                   help="clean: and kept at most this many cores busy on average")
    m.add_argument("--max-sleep", type=float, default=d.max_sleep_seconds,
                   help="clean: the machine slept at most this many seconds during the run")
    m.add_argument("--require-ac", action=argparse.BooleanOptionalAction, default=d.require_ac_power,
                   help="clean: on AC power and not in Low Power Mode (macOS)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1 :]
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        if cmd:
            raise SystemExit("simulate takes no command after --")
        simulate_size(config.get_profile(args.profile), args.size_mb, args.out_dir, args.truth)
        return 0
    if not cmd:
        raise SystemExit("measure needs the command to run after --")
    protocol = config.TimingProtocol(
        settle_seconds=args.settle,
        quiet_cpu_percent=args.quiet_cpu,
        quiet_max_cores=args.quiet_max_cores,
        quiet_window_seconds=args.quiet_window,
        max_wait_seconds=args.max_wait,
        rss_sample_seconds=args.sample,
        max_other_cpu_share=args.max_other_cpu,
        max_other_cores=args.max_other_cores,
        max_sleep_seconds=args.max_sleep,
        require_ac_power=args.require_ac,
    )
    return measure(cmd, args.out, protocol, runs_dir=args.runs_dir, threads=args.threads)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())


__all__ = [
    "EX_TEMPFAIL", "EX_UNAVAILABLE", "clean_environment", "cpu_breakdown", "during_run", "genome_for", "linux_machine",
    "machine_activity", "machine_other_cores", "measure", "measure_args", "other_cores", "other_method", "other_share",
    "power_state", "process_method", "process_times", "processes_hidden", "run_measured", "simulate_size", "unclean",
    "usable_cpus", "wait_until_quiet",
]
