"""Run a job of jobs.json (by index) and write records/<job>.json, or run them all.

Usage: run_job.py <index> [<jobs file> <records dir>]   (default: jobs.json, records/)
       run_job.py all [<jobs file> <records dir>] [-P 4]
'all' runs every job of the file, each in a process of its own (so that
its CPU time and memory are its own), at most -P at once, under nice 19.
Skips a job whose record says it finished with exit status 0 and whose
outputs still match. The record's paths are written as <repo> (this
repository) and <xplat> (this directory), as in the jobs file.
"""
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent


def expand(arg: str) -> str:
    if arg.startswith("<xplat>/"):
        return str(X / arg[len("<xplat>/"):])
    if arg.startswith("<repo>/"):
        return str(REPO / arg[len("<repo>/"):])
    return arg


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outputs_of(cmd):
    # positional: binary, input, output json, n_iter; plus the --psmc-out value
    out = [cmd[2]]
    if "--psmc-out" in cmd:
        out.append(cmd[cmd.index("--psmc-out") + 1])
    return out


def run_all(jobs_file: Path, records: Path, parallel: int):
    os.nice(19 - os.nice(0))
    n = len(json.loads(jobs_file.read_text()))

    def one(i):
        return subprocess.run([sys.executable, "-I", "-B", __file__, str(i), str(jobs_file), str(records)]).returncode

    with ThreadPoolExecutor(parallel) as pool:
        codes = list(pool.map(one, range(n)))
    print(f"{jobs_file.name}: {n} jobs, {sum(c != 0 for c in codes)} failed")
    return max(codes, default=0)


def main():
    args = [a for a in sys.argv[1:]]
    parallel = 4
    if "-P" in args:
        k = args.index("-P")
        parallel = int(args[k + 1])
        del args[k:k + 2]
    jobs_file = Path(args[1]) if len(args) > 1 else X / "jobs.json"
    records = Path(args[2]) if len(args) > 2 else X / "records"
    if args[0] == "all":
        sys.exit(run_all(jobs_file.resolve(), records.resolve(), parallel))
    idx = int(args[0])
    jobs = json.loads(jobs_file.read_text())
    job = jobs[idx]
    name = job["job"]
    rec_path = records / (name.replace("/", "__") + ".json")
    log_path = records.parent / ("logs" if records.name == "records" else f"logs-{records.name}") / (name.replace("/", "__") + ".log")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = job["command"]
    outs = outputs_of(cmd)
    if rec_path.exists():
        old = json.loads(rec_path.read_text())
        if old.get("exit_status") == 0 and all(Path(expand(p)).exists() and sha(Path(expand(p))) == old["outputs"].get(p) for p in outs):
            print(f"[skip] {name}")
            return
    for p in outs:
        Path(expand(p)).parent.mkdir(parents=True, exist_ok=True)
    binary = Path(expand(cmd[0]))
    load_before = os.getloadavg()
    start = time.time()
    with open(log_path, "w") as log:
        res = subprocess.run([expand(a) for a in cmd], stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - start
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "job": name,
        "command": cmd,
        "ibex_record": job["ibex_record"],
        "inputs": {k: v["laptop_sha256"] for k, v in job["inputs"].items()},
        "outputs": {p: (sha(Path(expand(p))) if Path(expand(p)).exists() else None) for p in outs},
        "binary_sha256": sha(binary),
        "exit_status": res.returncode,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "wall_seconds": round(wall, 3),
        "user_seconds": round(ru.ru_utime, 3),
        "system_seconds": round(ru.ru_stime, 3),
        "maxrss_mb": round(ru.ru_maxrss / (1024 * 1024), 1),  # bytes on macOS
        "loadavg_before": [round(x, 2) for x in load_before],
        "loadavg_after": [round(x, 2) for x in os.getloadavg()],
        "nice": os.nice(0),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    rec_path.write_text(json.dumps(record, indent=1))
    print(f"[done] {name} exit {res.returncode} wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
