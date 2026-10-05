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

import hashlib
import json
import os
import platform
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import paths

RECORD_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class ChecksumCache:
    """SHA-256 of files, cached by (size, mtime_ns)."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._data: Dict[str, List] = {}
        self._dirty = False
        if self.path is not None and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._data = {}

    def sha256(self, path: Path) -> str:
        p = str(Path(path).resolve())
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
            merged: Dict[str, List] = {}
            if self.path.exists():
                try:
                    merged = json.loads(self.path.read_text())
                except (OSError, ValueError):
                    merged = {}
            merged.update(self._data)
            self._data = merged
            write_json_atomic(self.path, merged)
            self._dirty = False


def portable(value, runs_dir: Path):
    """``value`` (str, path, list or dict) with machine-specific prefixes replaced."""
    if isinstance(value, (str, Path)):
        return paths.portable(str(value), runs_dir)
    if isinstance(value, dict):
        return {portable(k, runs_dir): portable(v, runs_dir) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v, runs_dir) for v in value]
    return value


def digest(key: Dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def job_key(job, runs_dir: Path, cache: ChecksumCache) -> Dict:
    """The key of ``job`` now: what its outputs depend on. Inputs must exist."""
    return {
        "job": job.id,
        "command": portable([str(c) for c in job.command], runs_dir),
        "env": portable(dict(job.env), runs_dir),
        "settings": job.settings,
        "tools": job.tools,
        "seeds": job.seeds,
        "inputs": {portable(str(p), runs_dir): cache.sha256(p) for p in job.inputs},
    }


def machine() -> Dict:
    """What the measurements were taken on."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if platform.system() == "Darwin":
        info["macos"] = platform.mac_ver()[0]
        try:
            import subprocess

            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            lines = res.stdout.split("\n")
            info["cpu"] = lines[0].strip()
            info["memory_gb"] = round(int(lines[1]) / 2**30, 1)
        except Exception:
            pass
    return info


def key_differences(old: Dict, new: Dict) -> List[str]:
    """Human-readable names of what changed between two keys."""
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
    for p in job.outputs:
        name = portable(str(p), runs_dir)
        if not Path(p).exists():
            return False, f"output missing: {Path(p).name}"
        if recorded.get(name) != cache.sha256(p):
            return False, f"output changed since the run: {Path(p).name}"
    return True, ""


def write_record(job, runs_dir: Path, cache: ChecksumCache, key: Dict, run: Dict) -> Dict:
    """Write the provenance record of a successful run of ``job``."""
    rec = {
        "record_version": RECORD_VERSION,
        "digest": digest(key),
        "key": key,
        "outputs": {portable(str(p), runs_dir): cache.sha256(p) for p in job.outputs},
        "run": run,
    }
    write_json_atomic(job.record_path(runs_dir), rec)
    return rec


def read_record(job, runs_dir: Path) -> Optional[Dict]:
    p = job.record_path(runs_dir)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def python_identity(modules: Sequence[str], packages: Sequence[str] = ()) -> Dict[str, str]:
    """Identity of a pipeline step run in Python: its source files and package versions.

    ``modules`` are file names in experiment/pipeline (``simulate.py``,
    ``experiments/e1.py``); their SHA-256 together form ``code``. A change
    to them re-runs the step; when it reproduces its outputs, nothing
    downstream re-runs (see the module documentation).
    """
    from importlib import metadata

    base = paths.EXPERIMENT / "pipeline"
    h = hashlib.sha256()
    for m in sorted(modules):
        h.update(m.encode())
        h.update((base / m).read_bytes())
    ident = {"code": h.hexdigest(), "python": platform.python_version()}
    for pkg in packages:
        try:
            ident[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            ident[pkg] = "missing"
    return ident
