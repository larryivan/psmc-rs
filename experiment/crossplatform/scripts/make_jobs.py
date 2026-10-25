"""Build the laptop job list from the committed Ibex provenance records.

Each command is taken verbatim from the Ibex record's key.command; only
the placeholders are substituted:
  <repo>/target/release/psmc-rs  -> the laptop build that runs (<xplat>/bin/psmc-rs by default)
  <runs>/E1/sims/...             -> the laptop's identical simulated genome (experiment/.production)
  <runs>/E1/fits/..., <runs>/E2/fits/... -> <xplat>/runs/E1/fits/..., <xplat>/runs/E2/fits/...
  <repo>/experiment/real_data/... -> this checkout's real_data file
Every other argument (including --threads) is kept as recorded. Paths are
written as <repo> (this repository) and <xplat> (this directory), which
run_job.py expands.

Usage:
  make_jobs.py                       # jobs.json: E2's NA12878 and E1's 40 fits, with bin/psmc-rs into runs/
  make_jobs.py --bin bin/psmc-rs-rustc1.99 --runs runs-rustc1.99 --out jobs_rustc1.99.json
  make_jobs.py --runs runs-repeat --out jobs_repeat.json --only E1/fit-rs/constant/rep01 E1/fit-rs/bottleneck/rep06
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent
LAP_RUNS = REPO / "experiment/.production/experiment/runs"
IBX_RUNS = REPO / "experiment/runs"
MODELS = ("constant", "bottleneck", "expansion", "zigzag")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def placeholder(p: Path) -> str:
    """<xplat>/... or <repo>/... for a path under this directory or the repository."""
    p = Path(p)
    for root, name in ((X, "<xplat>"), (REPO, "<repo>")):
        try:
            return f"{name}/{p.relative_to(root)}"
        except ValueError:
            pass
    raise ValueError(f"{p} is outside the repository")


def expand(arg: str) -> str:
    if arg.startswith("<xplat>/"):
        return str(X / arg[len("<xplat>/"):])
    if arg.startswith("<repo>/"):
        return str(REPO / arg[len("<repo>/"):])
    return arg


def substitute(arg: str, binary: str, runs: str) -> str:
    if arg == "<repo>/target/release/psmc-rs":
        return f"<xplat>/{binary}"
    if arg.startswith("<runs>/E1/sims/"):
        return placeholder(LAP_RUNS / arg[len("<runs>/"):])
    if arg.startswith("<runs>/E1/fits/") or arg.startswith("<runs>/E2/fits/"):
        return f"<xplat>/{runs}/{arg[len('<runs>/'):]}"
    if arg.startswith("<repo>/experiment/real_data/"):
        return arg
    if "<" in arg:
        raise ValueError(f"unhandled placeholder in {arg!r}")
    return arg


def job_from(record: Path, name: str, binary: str, runs: str):
    rec = json.loads(record.read_text())
    cmd = rec["key"]["command"]
    new = [substitute(a, binary, runs) for a in cmd]
    # inputs: the substituted input must have the checksum Ibex recorded
    inputs = {}
    for k, v in rec["key"]["inputs"].items():
        p = substitute(k, binary, runs)
        got = sha(Path(expand(p)))
        inputs[p] = {"ibex_placeholder": k, "ibex_sha256": v, "laptop_sha256": got, "identical": got == v}
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
        "ibex_run": {k: rec["run"].get(k) for k in ("host", "slurm_job_id", "cores", "wall_seconds", "user_seconds", "start")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="bin/psmc-rs", help="the binary, relative to this directory")
    ap.add_argument("--runs", default="runs", help="where the fits go, relative to this directory")
    ap.add_argument("--out", default="jobs.json")
    ap.add_argument("--only", nargs="*", help="these jobs only")
    a = ap.parse_args()
    names = [("E2/provenance/fit-rs/NA12878.json", "E2/fit-rs/NA12878")]  # NA12878 first: it is the longest job
    names += [(f"E1/provenance/fit-rs/{m}/rep{r:02d}.json", f"E1/fit-rs/{m}/rep{r:02d}") for m in MODELS for r in range(1, 11)]
    if a.only:
        names = [n for n in names if n[1] in a.only]
        assert len(names) == len(a.only), a.only
    jobs = [job_from(IBX_RUNS / rec, name, a.bin, a.runs) for rec, name in names]
    # sanity: every non-path argument is unchanged
    for j in jobs:
        for x, y in j["substituted"]:
            assert x.startswith("<"), (j["job"], x, y)
        assert "--threads" in j["command"]
        j["threads"] = j["command"][j["command"].index("--threads") + 1]
    (X / a.out).write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{a.out}: {len(jobs)} jobs; threads values: {sorted({j['threads'] for j in jobs})}")
    print("all inputs identical to the Ibex records:", all(i["identical"] for j in jobs for i in j["inputs"].values()))
    print("Ibex sources:", sorted({t["source_tree"] for j in jobs for t in j["ibex_tool"].values()}))
    print("example:", " ".join(jobs[-1]["command"]))


if __name__ == "__main__":
    main()
