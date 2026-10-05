"""The commands that E3's jobs run: simulate the genome of one size, and time one run of a tool.

``python -m experiment.pipeline.experiments.e3_run simulate --profile P --size-mb S --out-dir D --truth T``
    simulates the E3 genome of S Mb (``config.E3``: chromosomes of
    ``chromosome_length`` under ``model``, seeds
    ``seed("E3", model, replicate, "ancestry/chrN")``) and writes its psmcfa
    and truth record, as :func:`.simulate.simulate_genome` does (no VCF).

``python -m experiment.pipeline.experiments.e3_run measure --out TIMING.json [options] -- TOOL ARGS...``
    runs ``TOOL ARGS...`` and writes how it ran to TIMING.json. Before the
    run it waits ``--settle`` seconds, then until the machine is quiet: the
    CPU use of all processes together over ``--quiet-window`` seconds is at
    most ``--quiet-cpu`` percent of all cores. It then starts the tool, with
    the environment cleaned of variables that change psmc-rs's behaviour or
    build (``PSMC_*``, ``RUSTFLAGS``, ...), and samples the RSS of the tool's
    process tree every ``--sample`` seconds; the peak RSS is the larger of
    the sampled peak and the tool's own ``ru_maxrss``. The record holds the
    wall time, the tool's user and system CPU time, the peak RSS, the load
    average before and after, the busiest processes before the run, the CPU
    time that other processes used while it ran, the power source and the
    environment. When the tool fails, nothing is written and the command
    exits with the tool's status. When the run is not clean (the machine
    did not become quiet within ``--max-wait`` seconds, is on battery or in
    Low Power Mode with ``--require-ac``, slept during the run, or other
    processes used more than ``--max-other-cpu`` of all cores), nothing is
    written and the command exits with status 75, which the runner takes as
    "not now": it starts the run again later.

Only the tool's processes are measured, not this wrapper (the runner's own
record of the job includes the wrapper's Python interpreter). The module is
kept apart from :mod:`.e3` (jobs and tables) because its SHA-256 is part of
the provenance key of every timing job: editing the tables must not
invalidate hours of timing runs, while changing how runs are timed must.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config, paths, provenance, simulate

RECORD = "psmc-rs-manuscript/E3/timing/2"
EX_TEMPFAIL = 75  # not a clean run: the runner defers the job
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


def busy_seconds(times) -> float:
    """CPU seconds not spent idle in a ``psutil.cpu_times()`` result (all cores together)."""
    idle = times.idle + getattr(times, "iowait", 0.0)
    return sum(times) - idle


def machine_activity(window: float, top: int = 5) -> Dict:
    """The CPU use of the whole machine over ``window`` seconds, and the busiest processes.

    ``busy_percent`` is the share of all cores in use (100 = every core
    busy); ``busiest`` lists the processes with the most CPU in the window
    (``cpu_percent``: 100 = one core), this process excluded.
    """
    import psutil

    me = os.getpid()
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        if p.pid == me:
            continue
        try:
            p.cpu_percent(None)
            procs.append(p)
        except psutil.Error:
            pass
    before = psutil.cpu_times()
    t0 = time.perf_counter()
    time.sleep(max(window, 0.0))
    after = psutil.cpu_times()
    elapsed = time.perf_counter() - t0
    total = sum(after) - sum(before)
    busy = busy_seconds(after) - busy_seconds(before)
    usage = []
    for p in procs:
        try:
            usage.append((p.cpu_percent(None), p.info["pid"], p.info["name"]))
        except psutil.Error:
            pass
    usage.sort(key=lambda u: -u[0])
    return {
        "window_seconds": round(elapsed, 3),
        "busy_percent": round(100.0 * busy / total, 2) if total > 0 else 0.0,
        "busiest": [{"pid": pid, "name": name, "cpu_percent": round(c, 1)} for c, pid, name in usage[:top]],
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
    """Settle, then wait for a quiet machine (see the module documentation); returns what was seen."""
    time.sleep(max(protocol.settle_seconds, 0.0))
    t0 = time.monotonic()
    while True:
        act = machine_activity(protocol.quiet_window_seconds)
        waited = time.monotonic() - t0
        quiet = act["busy_percent"] <= protocol.quiet_cpu_percent
        if quiet or waited >= protocol.max_wait_seconds:
            break
        log(f"[measure] machine busy ({act['busy_percent']:.0f}% of all cores); waiting")
    if not quiet:
        log(f"[measure] machine not quiet after {waited:.0f} s ({act['busy_percent']:.0f}% busy)")
    return {
        "quiet": quiet,
        "busy_percent": act["busy_percent"],
        "threshold_percent": protocol.quiet_cpu_percent,
        "window_seconds": act["window_seconds"],
        "settle_seconds": protocol.settle_seconds,
        "waited_seconds": round(waited, 3),
        "busiest": act["busiest"],
    }


def machine() -> Dict:
    """:func:`.provenance.machine`, with the performance and efficiency cores of Apple silicon."""
    info = provenance.machine()
    if sys.platform == "darwin":
        for key, name in (("hw.perflevel0.physicalcpu", "performance_cores"), ("hw.perflevel1.physicalcpu", "efficiency_cores")):
            try:
                res = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if res.returncode == 0 and res.stdout.strip().isdigit():
                info[name] = int(res.stdout.strip())
    return info


def tree_rss(proc) -> int:
    """Resident set size of a psutil process and all its descendants, in bytes."""
    import psutil

    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
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


def run_measured(cmd: Sequence[str], sample_seconds: float, env: Optional[Dict[str, str]] = None) -> Dict:
    """Run ``cmd`` (its output goes to ours) and measure it; see the module documentation."""
    import psutil

    cpu_before = psutil.cpu_times()
    load_before = os.getloadavg()
    start = time.time()
    t0 = time.perf_counter()
    proc = subprocess.Popen([str(c) for c in cmd], stdin=subprocess.DEVNULL, env=env)
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
            peak = max(peak, tree_rss(ps))
            samples += 1
        time.sleep(sample_seconds)
    wall = time.perf_counter() - t0
    clock_span = time.time() - start  # runs on while the machine sleeps; perf_counter does not
    proc.returncode = os.waitstatus_to_exitcode(status)
    cpu_after = psutil.cpu_times()
    load_after = os.getloadavg()
    tool_cpu = rusage.ru_utime + rusage.ru_stime
    machine_busy = busy_seconds(cpu_after) - busy_seconds(cpu_before)
    machine_total = sum(cpu_after) - sum(cpu_before)
    other = max(machine_busy - tool_cpu, 0.0)
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
        "during": {
            "other_cpu_seconds": round(other, 3),
            # CPU time of other processes as a share of every core over the run
            "other_cpu_share": round(other / machine_total, 4) if machine_total > 0 else 0.0,
            "machine_busy_percent": round(100.0 * machine_busy / machine_total, 2) if machine_total > 0 else 0.0,
        },
    }


def _log(msg: str) -> None:
    print(msg, flush=True)


def measure(
    cmd: Sequence[str],
    out: Path,
    protocol: config.TimingProtocol,
    runs_dir: Optional[Path] = None,
    log=_log,
) -> int:
    """Time one run of ``cmd`` and write the record to ``out``; returns the exit status for the shell."""
    if not cmd:
        raise ValueError("no command to measure")
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


def unclean(res: Dict, protocol: config.TimingProtocol) -> Optional[str]:
    """Why a finished run is not a clean measurement, or None."""
    share = res["during"]["other_cpu_share"]
    if share > protocol.max_other_cpu_share:
        return (f"other processes used {100 * share:.1f}% of all cores during the run "
                f"(at most {100 * protocol.max_other_cpu_share:g}%)")
    slept = res["clock_span_seconds"] - res["wall_seconds"]
    if slept > protocol.max_sleep_seconds:
        return f"the machine slept for about {slept:.0f} s during the run"
    return None


def measure_args(protocol: config.TimingProtocol) -> List[str]:
    """The ``measure`` options of ``protocol``, each given explicitly."""
    return [
        "--settle", f"{protocol.settle_seconds:g}",
        "--quiet-cpu", f"{protocol.quiet_cpu_percent:g}",
        "--quiet-window", f"{protocol.quiet_window_seconds:g}",
        "--max-wait", f"{protocol.max_wait_seconds:g}",
        "--sample", f"{protocol.rss_sample_seconds:g}",
        "--max-other-cpu", f"{protocol.max_other_cpu_share:g}",
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
    m.add_argument("--settle", type=float, default=d.settle_seconds, help="seconds to wait first")
    m.add_argument("--quiet-cpu", type=float, default=d.quiet_cpu_percent, help="quiet: CPU use at most this %% of all cores")
    m.add_argument("--quiet-window", type=float, default=d.quiet_window_seconds, help="seconds over which CPU use is measured")
    m.add_argument("--max-wait", type=float, default=d.max_wait_seconds, help="run anyway after waiting this long")
    m.add_argument("--sample", type=float, default=d.rss_sample_seconds, help="seconds between RSS samples")
    m.add_argument("--max-other-cpu", type=float, default=d.max_other_cpu_share,
                   help="clean: other processes used at most this share of all cores during the run")
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
        quiet_window_seconds=args.quiet_window,
        max_wait_seconds=args.max_wait,
        rss_sample_seconds=args.sample,
        max_other_cpu_share=args.max_other_cpu,
        max_sleep_seconds=args.max_sleep,
        require_ac_power=args.require_ac,
    )
    return measure(cmd, args.out, protocol, runs_dir=args.runs_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())


__all__ = [
    "EX_TEMPFAIL", "clean_environment", "genome_for", "machine_activity", "measure", "measure_args", "power_state",
    "run_measured", "simulate_size", "unclean", "wait_until_quiet",
]
