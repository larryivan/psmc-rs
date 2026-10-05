"""The tools: psmc-rs 0.2 (this checkout), psmc-rs 0.1 (commit e50d69d), C psmc and utilities.

Each tool is a :class:`Tool`: a path, an identity and information. The
identity enters the provenance key of every job that runs the tool, and is
chosen so that it changes exactly when the tool's results could; the
information is recorded next to it and changes nothing:

* psmc-rs 0.2: its version, the content of the Rust sources (the git tree
  and blob ids of ``src/``, ``Cargo.toml`` and ``Cargo.lock`` at HEAD, so
  that a rebase or squash that keeps the sources keeps the identity), a
  digest of their uncommitted changes, and the toolchain (``rustc -vV`` and
  ``RUSTFLAGS``). Information: the last commit that touched the sources and
  the binary's SHA-256 (it embeds the checkout's path, so it differs between
  clones of the same sources). Outside a git checkout the binary's SHA-256
  is the identity. The binary is ``target/release/psmc-rs`` of this
  checkout, built with ``cargo build --release --locked --offline`` (a no-op
  when up to date).
* psmc-rs 0.1: the commit it is built from (in a git worktree under the
  tools directory with ``CARGO_TARGET_DIR`` next to it, see
  :func:`build_rs01`) and the toolchain; information: the binary's SHA-256.
* C psmc and the utilities: their version where they print one, and the
  SHA-256 of the file.

The runner records the SHA-256 of every executable a job runs, and refuses
to start jobs once one of them changes (see :mod:`.runner`).

The command builders give the exact command lines of the experiments.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import config, paths

RS_SOURCES = ("src", "Cargo.toml", "Cargo.lock")


@dataclass(frozen=True)
class Tool:
    name: str
    path: Path
    identity: Dict[str, str] = field(default_factory=dict)  # enters provenance keys
    info: Dict[str, str] = field(default_factory=dict)  # recorded, not part of any key

    def __str__(self) -> str:
        return str(self.path)


class ToolError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: Sequence[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def _git(*args: str, cwd: Path = paths.ROOT) -> str:
    res = _run(["git", *args], cwd=cwd)
    if res.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


# ---------------------------------------------------------------------------
# psmc-rs 0.2
# ---------------------------------------------------------------------------


def rs02_binary() -> Path:
    return paths.ROOT / "target" / "release" / "psmc-rs"


def source_tree(cwd: Path = paths.ROOT) -> str:
    """The content of the Rust sources at HEAD: ``src=<tree id>;Cargo.toml=<blob id>;Cargo.lock=<blob id>``."""
    return ";".join(f"{p}={_git('rev-parse', f'HEAD:{p}', cwd=cwd)}" for p in RS_SOURCES)


def toolchain() -> Dict[str, str]:
    """The compiler a build uses: ``rustc -vV``'s release, commit and host, LLVM, and ``RUSTFLAGS``."""
    out = {"rustflags": os.environ.get("RUSTFLAGS", ""), "cargo_encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", "")}
    try:
        res = _run(["rustc", "-vV"])
    except OSError:
        res = None
    if res is None or res.returncode != 0:
        out["rustc"] = "unavailable"
        return out
    fields = dict(line.split(": ", 1) for line in res.stdout.splitlines()[1:] if ": " in line)
    out["rustc"] = f"{fields.get('release', '?')} ({fields.get('commit-hash', '?')[:12]}) {fields.get('host', '?')}"
    out["llvm"] = fields.get("LLVM version", "")
    return out


def rs02_source_state() -> Dict[str, str]:
    """The content of the Rust sources (:func:`source_tree`) and a digest of their uncommitted changes ('' if clean)."""
    tree = source_tree()
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *RS_SOURCES)
    dirty = ""
    if status:
        h = hashlib.sha256(_git("diff", "HEAD", "--", *RS_SOURCES).encode())
        for line in sorted(status.splitlines()):
            if line.startswith("??"):
                p = paths.ROOT / line[3:]
                h.update(line.encode())
                if p.is_file():
                    h.update(p.read_bytes())
        dirty = h.hexdigest()
    return {"source_tree": tree, "source_dirty": dirty}


def build_rs02(log: Callable[[str], None] = print) -> Path:
    """``cargo build --release --locked --offline`` in this checkout (a no-op when up to date)."""
    res = _run(["cargo", "build", "--release", "--locked", "--offline"], cwd=paths.ROOT, timeout=1800)
    if res.returncode != 0:
        raise ToolError(f"building psmc-rs 0.2 failed:\n{res.stderr[-4000:]}")
    log("[tools] psmc-rs 0.2 is built (target/release/psmc-rs)")
    return rs02_binary()


def _version(binary: Path) -> str:
    res = _run([binary, "--version"])
    if res.returncode != 0:
        raise ToolError(f"{binary} --version failed: {res.stderr.strip()}")
    return res.stdout.strip()


def psmc_rs_02(build: bool = True, log: Callable[[str], None] = print) -> Tool:
    """psmc-rs 0.2 from this checkout, built first when ``build``.

    Outside a git checkout (an unpacked archive of the repository) the
    sources have no git ids, and the binary's SHA-256 identifies it instead.
    """
    binary = build_rs02(log) if build else rs02_binary()
    if not binary.exists():
        raise ToolError(f"{binary} does not exist; build it with cargo build --release")
    version = _version(binary)
    if not version.startswith("psmc-rs 0.2"):
        raise ToolError(f"{binary} is {version!r}, not psmc-rs 0.2")
    digest = sha256_file(binary)
    try:
        source = rs02_source_state()
        commit = _git("log", "-1", "--format=%H", "--", *RS_SOURCES)
    except (ToolError, OSError):
        source = {"source_tree": "unknown: not a git checkout", "binary_sha256": digest}
        commit = "unknown: not a git checkout"
    tc = toolchain()
    identity = {"version": version, **source, "rustc": tc["rustc"], "rustflags": tc["rustflags"],
                "cargo_encoded_rustflags": tc["cargo_encoded_rustflags"]}
    return Tool("psmc-rs-0.2", binary, identity, {"source_commit": commit, "binary_sha256": digest,
                                                  "llvm": tc.get("llvm", "")})


# ---------------------------------------------------------------------------
# psmc-rs 0.1
# ---------------------------------------------------------------------------


def rs01_dirs(tools: Optional[Path] = None) -> Dict[str, Path]:
    base = (tools or paths.tools_dir()) / "psmc-rs-0.1"
    return {"base": base, "src": base / "src", "target": base / "target"}


def build_rs01(commit: str = config.E3().rs01_commit, tools: Optional[Path] = None, log: Callable[[str], None] = print) -> Path:
    """Build psmc-rs 0.1 from ``commit`` into the tools directory.

    The source is a detached git worktree at ``<tools>/psmc-rs-0.1/src`` and
    the build goes to ``CARGO_TARGET_DIR=<tools>/psmc-rs-0.1/target``, so
    nothing of this checkout changes. Offline: the dependencies must be in
    cargo's cache.
    """
    d = rs01_dirs(tools)
    full = _git("rev-parse", f"{commit}^{{commit}}")
    if not d["src"].exists():
        d["base"].mkdir(parents=True, exist_ok=True)
        _git("worktree", "add", "--detach", str(d["src"]), full)
        log(f"[tools] git worktree for psmc-rs 0.1 at {d['src']}")
    head = _git("rev-parse", "HEAD", cwd=d["src"])
    if head != full:
        raise ToolError(f"{d['src']} is at {head}, expected {full}; remove it (git worktree remove) and rebuild")
    env = dict(os.environ, CARGO_TARGET_DIR=str(d["target"]))
    res = _run(["cargo", "build", "--release", "--locked", "--offline"], cwd=d["src"], env=env, timeout=3600)
    if res.returncode != 0:
        raise ToolError(f"building psmc-rs 0.1 failed:\n{res.stderr[-4000:]}")
    binary = d["target"] / "release" / "psmc-rs"
    log(f"[tools] psmc-rs 0.1 is built ({binary})")
    return binary


def psmc_rs_01(build: bool = False, commit: str = config.E3().rs01_commit, log: Callable[[str], None] = print) -> Tool:
    """psmc-rs 0.1, built from ``commit`` when ``build``.

    0.1 has no ``--version``; its version comes from the Cargo.toml it was
    built from, and the commit and the binary's SHA-256 identify it.
    """
    d = rs01_dirs()
    binary = build_rs01(commit, log=log) if build else d["target"] / "release" / "psmc-rs"
    if not binary.exists():
        raise ToolError(f"{binary} does not exist; build it with: python -m experiment.pipeline tools --build-rs01")
    head = _git("rev-parse", "HEAD", cwd=d["src"]) if d["src"].exists() else ""
    m = re.search(r'^version\s*=\s*"([^"]+)"', (d["src"] / "Cargo.toml").read_text(), re.M) if head else None
    tc = toolchain()
    digest = sha256_file(binary)
    identity = {"version": f"psmc-rs {m.group(1)}" if m else "unknown", "commit": head, "rustc": tc["rustc"],
                "rustflags": tc["rustflags"]}
    if not head:
        identity["binary_sha256"] = digest  # no sources to identify it by
    return Tool("psmc-rs-0.1", binary, identity, {"binary_sha256": digest})


# ---------------------------------------------------------------------------
# C psmc and utilities
# ---------------------------------------------------------------------------

UTILITIES = {
    "fq2psmcfa": ("psmc_c", "utils", "fq2psmcfa"),
    "splitfa": ("psmc_c", "utils", "splitfa"),
    "psmc_plot.pl": ("psmc_c", "psmc_plot.pl"),
    "vcfutils.pl": ("vcfutils.pl",),
    "generate_multihetsep.py": ("msmc-tools", "generate_multihetsep.py"),
}


def c_psmc_version(binary: Path) -> str:
    """The ``Version:`` that C psmc prints when run without arguments."""
    res = _run([binary])
    m = re.search(r"Version:\s*(\S+)", res.stdout + res.stderr)
    if not m:
        raise ToolError(f"{binary} printed no version; is it lh3/psmc?")
    return m.group(1)


def c_psmc(tools: Optional[Path] = None) -> Tool:
    """lh3/psmc, built with -O1 (an -O2 build segfaults on macOS arm64), at ``<tools>/psmc_c/psmc``."""
    binary = (tools or paths.tools_dir()) / "psmc_c" / "psmc"
    if not binary.exists():
        raise ToolError(f"C psmc not found at {binary} (set PSMC_EXP_TOOLS)")
    return Tool("c-psmc", binary, {"version": c_psmc_version(binary), "sha256": sha256_file(binary)})


def utility(name: str, tools: Optional[Path] = None) -> Tool:
    """One of :data:`UTILITIES`, identified by the SHA-256 of its file."""
    p = (tools or paths.tools_dir()).joinpath(*UTILITIES[name])
    if not p.exists():
        raise ToolError(f"{name} not found at {p} (set PSMC_EXP_TOOLS)")
    return Tool(name, p, {"sha256": sha256_file(p)})


def htslib_versions() -> Dict[str, str]:
    out = {}
    for name in ("bgzip", "tabix", "bcftools"):
        try:
            res = _run([name, "--version"])
            out[name] = res.stdout.splitlines()[0].strip() if res.returncode == 0 and res.stdout else "unavailable"
        except OSError:
            out[name] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# Command lines
# ---------------------------------------------------------------------------


def fmt(x: float) -> str:
    """A number as C's %g would print it (15.0 -> "15", 2.5e-08 -> "2.5e-08")."""
    return f"{x:g}"


def c_psmc_fit(tool: Tool, inference: config.Inference, inp: Path, out: Path) -> List[str]:
    """``psmc -N<n> -t<t> -r<r> -p <pattern> -o OUT IN``."""
    return [
        str(tool.path),
        f"-N{inference.n_iter}",
        f"-t{fmt(inference.t_max)}",
        f"-r{fmt(inference.tr_ratio)}",
        "-p",
        inference.pattern,
        "-o",
        str(out),
        str(inp),
    ]


def rs02_model_args(inference: config.Inference) -> List[str]:
    """The psmc-rs 0.2 options of the model, each passed explicitly."""
    return [
        "--pattern", inference.pattern,
        "--t-max", fmt(inference.t_max),
        "--tr-ratio", fmt(inference.tr_ratio),
        "--smooth-lambda", fmt(inference.smooth_lambda),
        "--mstep-iters", str(inference.mstep_iters),
        "--bin-size", str(inference.bin_size),
    ]


def rs02_fit(
    tool: Tool,
    inference: config.Inference,
    scaling: config.Scaling,
    inp: Path,
    out_json: Path,
    *,
    threads: int,
    psmc_out: Optional[Path] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """``psmc-rs IN OUT.json N_ITER`` with the model, scaling and run options."""
    cmd = [
        str(tool.path),
        str(inp),
        str(out_json),
        str(inference.n_iter),
        *rs02_model_args(inference),
        "--mu", fmt(scaling.mu),
        "--gen-years", fmt(scaling.gen_years),
        "--threads", str(threads),
        "--no-progress",
    ]
    if psmc_out is not None:
        cmd += ["--psmc-out", str(psmc_out)]
    return cmd + [str(x) for x in extra]


def rs02_qc(tool: Tool, inp: Path, json_out: Path, extra: Sequence[str] = ()) -> List[str]:
    """``psmc-rs qc IN --json OUT``."""
    return [str(tool.path), "qc", str(inp), "--json", str(json_out), *[str(x) for x in extra]]


def describe(tools: Sequence[Tool]) -> str:
    lines = []
    for t in tools:
        ident = ", ".join(f"{k}={v}" for k, v in {**t.identity, **t.info}.items() if v)
        lines.append(f"{t.name:24s} {t.path}\n{'':24s} {ident}")
    return "\n".join(lines)
