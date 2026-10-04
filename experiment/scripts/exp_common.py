"""Shared helpers for the experiment scripts.

Everything here is located relative to this file, so the scripts behave the
same whatever the current working directory is and contain no machine-specific
paths. Tool locations follow one rule everywhere:

    command-line flag  >  environment variable  >  default

* psmc-rs: ``--psmc-rs`` > ``$PSMC_RS_BIN`` > ``<repo>/target/release/psmc-rs``
  (built with ``cargo build --release`` when missing).
* C psmc:  ``--c-psmc``  > ``$C_PSMC_BIN``  > ``psmc`` found on ``$PATH``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

try:
    import psutil

    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Cargo.toml").exists() and (p / "src").exists():
            return p
    raise RuntimeError(f"Cannot locate psmc-rs root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
EXPERIMENT = ROOT / "experiment"
SCRIPTS = EXPERIMENT / "scripts"
RUNS = EXPERIMENT / "runs"
SIM_SCRIPT_DEFAULT = SCRIPTS / "simulate_msprime_to_psmcfa.py"
PSMC_RS_DEFAULT = ROOT / "target" / "release" / "psmc-rs"

C_PSMC_BUILD_HELP = """\
The original C psmc (lh3/psmc 0.6.5-r74) is needed for the Rust-vs-C comparison.
Build it and pass its path with --c-psmc (or set C_PSMC_BIN, or put `psmc` on PATH):

    git clone https://github.com/lh3/psmc.git
    cd psmc
    cc -O1 -o psmc cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz

Use -O1 (or -O0): on macOS arm64 with Apple clang an -O2 build segfaults after
the first EM round, while -O0 and -O1 builds agree byte for byte."""


def resolve_psmc_rs(cli_value: Optional[str], *, build: bool = True) -> Path:
    """Locate the psmc-rs binary, building ``target/release/psmc-rs`` if needed.

    An explicitly requested binary (flag or ``$PSMC_RS_BIN``) must exist; the
    default is always the release build of this checkout, never a ``psmc-rs``
    found on ``$PATH``, so results cannot silently come from another version.
    """
    explicit = cli_value or os.environ.get("PSMC_RS_BIN")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"psmc-rs binary not found: {p}")
        return p
    if not PSMC_RS_DEFAULT.exists() and build:
        print("[build] cargo build --release --locked", flush=True)
        subprocess.run(["cargo", "build", "--release", "--locked"], cwd=ROOT, check=True)
    if not PSMC_RS_DEFAULT.exists():
        raise FileNotFoundError(
            f"psmc-rs binary not found: {PSMC_RS_DEFAULT}\n"
            "Build it with `cargo build --release --locked` or pass --psmc-rs."
        )
    return PSMC_RS_DEFAULT


def resolve_c_psmc(cli_value: Optional[str], *, required: bool) -> Optional[Path]:
    """Locate the C psmc binary; raise with build instructions if it is required."""
    explicit = cli_value or os.environ.get("C_PSMC_BIN")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"C psmc binary not found: {p}\n\n{C_PSMC_BUILD_HELP}")
        return p
    found = shutil.which("psmc")
    if found:
        return Path(found).resolve()
    if required:
        raise FileNotFoundError(f"C psmc binary not found.\n\n{C_PSMC_BUILD_HELP}")
    return None


def repo_relative(path: Path, root: Path = ROOT) -> str:
    """``path`` relative to ``root`` with ``/`` separators, or absolute if outside it.

    Tables store paths in this form so they stay valid in any clone.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(p)


def resolve_repo_path(value: str, root: Path = ROOT) -> Path:
    """Inverse of :func:`repo_relative`: relative paths are taken from ``root``."""
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def run_command(
    cmd: Sequence[str],
    *,
    log_path: Path,
    cwd: Optional[Path] = None,
    check: bool = True,
    env: Optional[dict] = None,
    stdout_path: Optional[Path] = None,
) -> Dict:
    """Run ``cmd``, sampling the peak RSS of its process tree when psutil is present.

    The record (command, return code, output, wall time, peak RSS in MiB) is
    appended to ``log_path`` as one JSON line. Output goes to temporary files
    rather than pipes: polling a process whose pipe buffer is full would hang.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryFile("w+", encoding="utf-8") as err_fh, tempfile.TemporaryFile(
        "w+", encoding="utf-8"
    ) as tmp_out:
        out_fh = open(stdout_path, "w", encoding="utf-8") if stdout_path is not None else tmp_out
        try:
            t0 = time.perf_counter()
            proc = subprocess.Popen(
                [str(x) for x in cmd],
                cwd=str(cwd) if cwd is not None else None,
                stdout=out_fh,
                stderr=err_fh,
                text=True,
                env=env,
            )

            ps_proc = None
            peak_bytes = 0
            if HAS_PSUTIL:
                try:
                    ps_proc = psutil.Process(proc.pid)
                except Exception:
                    ps_proc = None

            while proc.poll() is None:
                if ps_proc is not None:
                    try:
                        rss = ps_proc.memory_info().rss
                        for ch in ps_proc.children(recursive=True):
                            try:
                                rss += ch.memory_info().rss
                            except Exception:
                                pass
                        peak_bytes = max(peak_bytes, rss)
                    except Exception:
                        pass
                time.sleep(0.02)
            dt = time.perf_counter() - t0
        finally:
            if stdout_path is not None:
                out_fh.close()

        tmp_out.seek(0)
        err_fh.seek(0)
        out = tmp_out.read() if stdout_path is None else ""
        err = err_fh.read()

    peak_rss_mb = peak_bytes / (1024**2) if ps_proc is not None else float("nan")
    rec = {
        "cmd": " ".join(shlex.quote(str(x)) for x in cmd),
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
        "wall_sec": dt,
        "peak_rss_mb": peak_rss_mb,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if check and proc.returncode != 0:
        print(rec["cmd"])
        print("--- stdout ---")
        print(out)
        print("--- stderr ---")
        print(err)
        raise RuntimeError(f"command failed: rc={proc.returncode}")
    return rec


def require_modules(names: Sequence[str], purpose: str) -> None:
    """Exit with a clear message naming every module in ``names`` that cannot be imported."""
    import importlib
    import sys

    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"error: {purpose} needs Python packages that {sys.executable} cannot import: "
            f"{', '.join(missing)}\n"
            f"Install them with:\n    {sys.executable} -m pip install {' '.join(missing)}\n"
            "(see experiment/requirements.txt for the full list)."
        )
