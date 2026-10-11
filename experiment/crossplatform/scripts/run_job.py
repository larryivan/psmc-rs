"""Run one job of xplat/jobs.json (by index) and write xplat/records/<job>.json.

Usage: run_job.py <index> [<jobs file> <records dir>]   (default: xplat/jobs.json, xplat/records)
Skips a job whose record says it finished with exit status 0 and whose outputs still match.
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
from pathlib import Path

X = Path(__file__).resolve().parent.parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def outputs_of(cmd):
    # positional: binary, input, output json, n_iter; plus the --psmc-out value
    out = [Path(cmd[2])]
    if "--psmc-out" in cmd:
        out.append(Path(cmd[cmd.index("--psmc-out") + 1]))
    return out


def main():
    idx = int(sys.argv[1])
    jobs_file = Path(sys.argv[2]) if len(sys.argv) > 2 else X / "jobs.json"
    records = Path(sys.argv[3]) if len(sys.argv) > 3 else X / "records"
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
        if old.get("exit_status") == 0 and all(p.exists() and sha(p) == old["outputs"].get(str(p)) for p in outs):
            print(f"[skip] {name}")
            return
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
    binary = Path(cmd[0])
    load_before = os.getloadavg()
    start = time.time()
    with open(log_path, "w") as log:
        res = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    wall = time.time() - start
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = {
        "job": name,
        "command": cmd,
        "ibex_record": job["ibex_record"],
        "inputs": {k: v["laptop_sha256"] for k, v in job["inputs"].items()},
        "outputs": {str(p): (sha(p) if p.exists() else None) for p in outs},
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
    print(f"[done] {name} exit {res.returncode} wall {wall:.0f}s")


if __name__ == "__main__":
    main()
