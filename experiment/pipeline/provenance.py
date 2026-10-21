"""Checksums and provenance records: when is a job done?

A job is done when its outputs exist and its provenance record matches what
the job would do now. The record's *key* holds everything that determines
the outputs: the command, extra environment, settings, tool identities,
seeds and the SHA-256 of every input. Its digest is compared with the
digest of the key computed afresh, and every output must still have the
checksum recorded when the job finished. Paths are written with
machine-independent prefixes (``<repo>``, ``<runs>``, ``<tools>``,
``<python>``), so moving the repository or running from a worktree keeps
the records valid.

Because inputs enter the key by checksum, not by the record of the job that
made them, re-running an upstream job that reproduces its outputs byte for
byte (a deterministic simulation, say) leaves everything downstream done.

Checksums are cached by path, size and modification time in
``<runs>/_runner/checksums.json``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import socket
import subprocess
import threading
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import paths

RECORD_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict:
    """The JSON in ``path``, or {} when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ChecksumCache:
    """SHA-256 of files by their resolved paths, cached by (size, mtime_ns)."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._data: Dict[str, List] = read_json(self.path) if self.path is not None else {}
        self._dirty = False
        self._real_dirs: Dict[str, str] = {}  # see paths.resolve

    def sha256(self, path: Path) -> str:
        p = paths.resolve(path, self._real_dirs)
        st = os.stat(p)
        with self._lock:
            hit = self._data.get(p)
            if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
                return hit[2]
        digest = sha256_file(Path(p))
        with self._lock:
            self._data[p] = [st.st_size, st.st_mtime_ns, digest]
            self._dirty = True
        return digest

    def save(self) -> None:
        """Merge with what is on disk (other processes) and write atomically."""
        if self.path is None or not self._dirty:
            return
        with self._lock:
            merged = read_json(self.path)
            merged.update(self._data)
            self._data = merged
            write_json_atomic(self.path, merged)
            self._dirty = False


def portable(value, runs_dir: Path):
    """``value`` (str, path, or a list or dict of them) with machine-specific prefixes replaced."""
    prefixes = paths.placeholders(runs_dir)

    def convert(v):
        if isinstance(v, (str, Path)):
            return paths.replace_prefixes(str(v), prefixes)
        if isinstance(v, dict):
            return {convert(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [convert(x) for x in v]
        return v

    return convert(value)


def digest(key: Dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def job_key(job, runs_dir: Path, cache: ChecksumCache) -> Dict:
    """The key of ``job`` now: what its outputs depend on. Inputs must exist."""
    command, env, inputs = portable([[str(c) for c in job.command], dict(job.env), job.inputs], runs_dir)
    return {
        "job": job.id,
        "command": command,
        "env": env,
        "settings": job.settings,
        "tools": job.tools,
        "seeds": job.seeds,
        "inputs": {name: cache.sha256(p) for name, p in zip(inputs, job.inputs)},
    }


def machine() -> Dict:
    """What the measurements were taken on (information only: it enters no job key)."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
    }
    if platform.system() == "Linux":
        info.update(linux_machine(Path("/proc")))
        lib, version = platform.libc_ver()
        if lib:
            info["libc"] = f"{lib} {version}"
        if hasattr(os, "sched_getaffinity"):
            info["cpus_usable"] = len(os.sched_getaffinity(0))  # a SLURM allocation's cgroup, taskset
        info.update(slurm(os.environ))
    if platform.system() == "Darwin":
        info["macos"] = platform.mac_ver()[0]
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
                                 capture_output=True, text=True, timeout=5)
            lines = res.stdout.split("\n")
            info["cpu"] = lines[0].strip()
            info["memory_gb"] = round(int(lines[1]) / 2**30, 1)
        except Exception:
            pass
    return info


def linux_machine(proc: Path) -> Dict:
    """The CPU model and memory of a Linux machine, from ``proc`` (``/proc``)."""
    info: Dict = {}
    try:
        for line in (proc / "cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("model name") and ":" in line:
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        for line in (proc / "meminfo").read_text(errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                info["memory_gb"] = round(int(line.split()[1]) / 2**20, 1)  # kB
                break
    except (OSError, ValueError, IndexError):
        pass
    return info


def slurm(environ: Mapping[str, str]) -> Dict:
    """The SLURM job and node a process runs in, from its environment (empty outside SLURM)."""
    info: Dict = {}
    for name, keys in (
        ("slurm_job_id", ("SLURM_JOB_ID",)),
        ("slurm_nodelist", ("SLURM_JOB_NODELIST", "SLURMD_NODENAME")),
        ("slurm_cpus_on_node", ("SLURM_CPUS_ON_NODE",)),
    ):
        value = next((environ[k] for k in keys if environ.get(k)), None)
        if value is not None:
            info[name] = value
    return info


def key_differences(old: Dict, new: Dict) -> List[str]:
    """Human-readable names of what changed between two keys.

    Both are compared as JSON (a fresh key's tuples are a stored key's lists).
    """
    old, new = json.loads(json.dumps(old)), json.loads(json.dumps(new))
    out = []
    for field in ("command", "env", "settings", "tools", "seeds"):
        if old.get(field) != new.get(field):
            out.append(field)
    old_in, new_in = old.get("inputs", {}), new.get("inputs", {})
    if old_in != new_in:
        changed = sorted(k for k in set(old_in) | set(new_in) if old_in.get(k) != new_in.get(k))
        out.append("inputs (" + ", ".join(Path(k).name for k in changed[:3]) + ("..." if len(changed) > 3 else "") + ")")
    return out


def check_done(job, runs_dir: Path, cache: ChecksumCache) -> Tuple[bool, str]:
    """``(done, reason)``: whether ``job``'s outputs exist with a matching record."""
    rec_path = job.record_path(runs_dir)
    if not rec_path.exists():
        return False, "not run yet"
    try:
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError):
        return False, "unreadable provenance record"
    missing = [p for p in job.inputs if not Path(p).exists()]
    if missing:
        return False, f"input missing: {Path(missing[0]).name}"
    key = job_key(job, runs_dir, cache)
    if rec.get("digest") != digest(key):
        what = key_differences(rec.get("key", {}), key)
        return False, "changed: " + (", ".join(what) if what else "record format")
    recorded = rec.get("outputs", {})
    for p, name in zip(job.outputs, portable(job.outputs, runs_dir)):
        if not Path(p).exists():
            return False, f"output missing: {Path(p).name}"
        if recorded.get(name) != cache.sha256(p):
            return False, f"output changed since the run: {Path(p).name}"
    return True, ""


def write_record(job, runs_dir: Path, cache: ChecksumCache, key: Dict, run: Dict) -> None:
    """Write the provenance record of a successful run of ``job``."""
    write_json_atomic(job.record_path(runs_dir), {
        "record_version": RECORD_VERSION,
        "digest": digest(key),
        "key": key,
        "outputs": {name: cache.sha256(p) for p, name in zip(job.outputs, portable(job.outputs, runs_dir))},
        "run": run,
    })


def python_identity(modules: Sequence[str], packages: Sequence[str] = ()) -> Dict[str, str]:
    """Identity of a pipeline step run in Python: its source files and package versions.

    ``modules`` are file names in experiment/pipeline (``simulate.py``,
    ``experiments/e1.py``); their SHA-256 together form ``code``. A change
    to them re-runs the step; when it reproduces its outputs, nothing
    downstream re-runs (see the module documentation).
    """
    base = paths.EXPERIMENT / "pipeline"
    h = hashlib.sha256()
    for m in sorted(modules):
        h.update(m.encode())
        h.update((base / m).read_bytes())
    return {"code": h.hexdigest(), "python": platform.python_version(), **{p: package_version(p) for p in packages}}


@functools.lru_cache(maxsize=None)
def package_version(name: str) -> str:
    """The installed version of a Python package, or "missing" (cached: reading it scans the installed packages)."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"
