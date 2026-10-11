"""Build the laptop job list from the committed Ibex provenance records.

Each command is taken verbatim from the Ibex record's key.command; only
the placeholders are substituted:
  <repo>/target/release/psmc-rs  -> the frozen copy of this checkout's build (xplat/bin/psmc-rs)
  <runs>/E1/sims/...             -> the laptop's identical simulated genome (experiment/.production)
  <runs>/E1/fits/..., <runs>/E2/fits/... -> xplat/runs/E1/fits/..., xplat/runs/E2/fits/...
  <repo>/experiment/real_data/... -> this checkout's real_data file
Every other argument (including --threads) is kept as recorded.
Writes xplat/jobs.json.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent
LAP_RUNS = REPO / "experiment/.production/experiment/runs"
IBX_RUNS = REPO / "experiment/runs"
BIN = X / "bin/psmc-rs"
MODELS = ("constant", "bottleneck", "expansion", "zigzag")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def substitute(arg: str) -> str:
    if arg == "<repo>/target/release/psmc-rs":
        return str(BIN)
    if arg.startswith("<runs>/E1/sims/"):
        return str(LAP_RUNS / arg[len("<runs>/"):])
    if arg.startswith("<runs>/E1/fits/") or arg.startswith("<runs>/E2/fits/"):
        return str(X / "runs" / arg[len("<runs>/"):])
    if arg.startswith("<repo>/experiment/real_data/"):
        return str(REPO / arg[len("<repo>/"):])
    if "<" in arg:
        raise ValueError(f"unhandled placeholder in {arg!r}")
    return arg


def job_from(record: Path, name: str):
    rec = json.loads(record.read_text())
    cmd = rec["key"]["command"]
    new = [substitute(a) for a in cmd]
    # inputs: the substituted input must have the checksum Ibex recorded
    inputs = {}
    for k, v in rec["key"]["inputs"].items():
        p = Path(substitute(k))
        got = sha(p)
        inputs[str(p)] = {"ibex_placeholder": k, "ibex_sha256": v, "laptop_sha256": got, "identical": got == v}
        if got != v:
            raise SystemExit(f"{name}: input {p} has sha256 {got}, Ibex recorded {v}")
    changed = [(a, b) for a, b in zip(cmd, new) if a != b]
    return {
        "job": name,
        "ibex_record": str(record.relative_to(REPO)),
        "ibex_command": cmd,
        "command": new,
        "substituted": changed,
        "inputs": inputs,
        "ibex_outputs": rec["outputs"],
        "ibex_tool": rec["key"]["tools"],
        "ibex_settings": rec["key"]["settings"],
        "ibex_run": {k: rec["run"].get(k) for k in ("host", "cores", "wall_seconds", "user_seconds", "start")},
    }


jobs = []
# NA12878 first: it is the longest job
jobs.append(job_from(IBX_RUNS / "E2/provenance/fit-rs/NA12878.json", "E2/fit-rs/NA12878"))
for m in MODELS:
    for r in range(1, 11):
        jobs.append(job_from(IBX_RUNS / f"E1/provenance/fit-rs/{m}/rep{r:02d}.json", f"E1/fit-rs/{m}/rep{r:02d}"))

# sanity: every non-path argument is unchanged
for j in jobs:
    for a, b in j["substituted"]:
        assert a.startswith("<"), (j["job"], a, b)
    assert "--threads" in j["command"]
    t = j["command"][j["command"].index("--threads") + 1]
    j["threads"] = t

(X / "jobs.json").write_text(json.dumps(jobs, indent=1))
print(f"{len(jobs)} jobs; threads values: {sorted({j['threads'] for j in jobs})}")
print("all inputs identical to the Ibex records:", all(i["identical"] for j in jobs for i in j["inputs"].values()))
print("example:", " ".join(jobs[1]["command"]))
