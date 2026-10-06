"""The tools: psmc-rs 0.2 (this checkout), psmc-rs 0.1 (commit e50d69d), C psmc and utilities.

Each tool is a :class:`Tool`: a path, an identity and information. The
identity enters the provenance key of every job that runs the tool, and is
chosen so that it changes exactly when the tool's results could; the
information is recorded next to it and changes nothing:

* psmc-rs 0.2: its version, the content of the Rust sources (the git tree
  and blob ids of ``src/``, ``Cargo.toml`` and ``Cargo.lock`` at HEAD, so
  that a rebase or squash that keeps the sources keeps the identity), a
  digest of their uncommitted changes, and the build (:func:`toolchain`:
  ``rustc -vV``, the rustflags and ``CARGO_PROFILE_RELEASE_*`` variables
  cargo reads, and the target features the flags enable). Information: the
  last commit that touched the sources and the binary's SHA-256 (it embeds
  the checkout's path, so it differs between clones of the same sources).
  Outside a git checkout the binary's SHA-256 is the identity. The binary
  is ``target/release/psmc-rs`` of this checkout, built with ``cargo build
  --release --locked --offline`` (a no-op when up to date).
* psmc-rs 0.1: the commit it is built from (in a git worktree under the
  tools directory with ``CARGO_TARGET_DIR`` next to it, see
  :func:`build_rs01`) and the build, as 0.2's; information: the binary's
  SHA-256.
* C psmc and the utilities: their version where they print one, and the
  SHA-256 of the file. C psmc built by :func:`build_c` (``tools
  --build-c``) also has the compiler and CFLAGS its ``BUILD.json`` records.

The runner records the SHA-256 of every executable a job runs, and refuses
to start jobs once one of them changes (see :mod:`.runner`).

The command builders give the exact command lines of the experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
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


# The target features that change psmc-rs's code: with ``fma`` an x86-64
# build fuses its multiply-adds (fmadd in src/hmm/kernel.rs), which changes
# its floating-point results; the others select or widen vector kernels.
TARGET_FEATURES = ("avx2", "fma", "avx512f", "neon")
PROFILE_ENV_PREFIX = "CARGO_PROFILE_RELEASE_"
BUILD_FIELDS = ("rustc", "rustflags", "cargo_encoded_rustflags", "cargo_build_rustflags", "cargo_target_rustflags",
                "cargo_profile_release", "target_features")


def target_rustflags_var(host: str) -> str:
    """The variable of ``target.<host>.rustflags``, e.g. ``CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS``."""
    return "CARGO_TARGET_" + re.sub(r"[^A-Za-z0-9]", "_", host).upper() + "_RUSTFLAGS"


def effective_rustflags(env: Dict[str, str], host: str) -> List[str]:
    """The flags cargo gives rustc: those of the first of these variables that is set (cargo's precedence).

    ``CARGO_ENCODED_RUSTFLAGS`` (separated by 0x1f), ``RUSTFLAGS``,
    ``CARGO_TARGET_<HOST>_RUSTFLAGS``, ``CARGO_BUILD_RUSTFLAGS``. Cargo's
    configuration files are not read (this checkout's ``.cargo/config.toml``
    sets no flags).
    """
    if "CARGO_ENCODED_RUSTFLAGS" in env:
        return [f for f in env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if f]
    for var in ("RUSTFLAGS", target_rustflags_var(host), "CARGO_BUILD_RUSTFLAGS"):
        if var in env:
            return env[var].split()
    return []


def target_features(flags: Sequence[str]) -> str:
    """Which :data:`TARGET_FEATURES` ``rustc --print cfg FLAGS`` enables: "avx2,fma", "none" or "unavailable"."""
    try:
        res = _run(["rustc", "--print", "cfg", *flags])
    except OSError:
        return "unavailable"
    if res.returncode != 0:
        return "unavailable"
    enabled = set(re.findall(r'^target_feature="([^"]+)"$', res.stdout, re.M))
    return ",".join(f for f in TARGET_FEATURES if f in enabled) or "none"


def toolchain() -> Dict[str, str]:
    """The build cargo makes here: ``rustc -vV``'s release, commit and host, LLVM, and what changes the code.

    That is the rustflags of every variable cargo reads them from, every
    ``CARGO_PROFILE_RELEASE_*`` variable (``KEY=VALUE``, joined by ";"), and
    which :data:`TARGET_FEATURES` the effective flags enable.
    """
    env = dict(os.environ)
    out = {
        "rustflags": env.get("RUSTFLAGS", ""),
        "cargo_encoded_rustflags": env.get("CARGO_ENCODED_RUSTFLAGS", ""),
        "cargo_build_rustflags": env.get("CARGO_BUILD_RUSTFLAGS", ""),
        "cargo_target_rustflags": "",
        "cargo_profile_release": ";".join(f"{k}={env[k]}" for k in sorted(env) if k.startswith(PROFILE_ENV_PREFIX)),
    }
    try:
        res = _run(["rustc", "-vV"])
    except OSError:
        res = None
    if res is None or res.returncode != 0:
        out["rustc"] = out["target_features"] = "unavailable"
        return out
    fields = dict(line.split(": ", 1) for line in res.stdout.splitlines()[1:] if ": " in line)
    host = fields.get("host", "?")
    out["rustc"] = f"{fields.get('release', '?')} ({fields.get('commit-hash', '?')[:12]}) {host}"
    out["llvm"] = fields.get("LLVM version", "")
    out["cargo_target_rustflags"] = env.get(target_rustflags_var(host), "")
    out["target_features"] = target_features(effective_rustflags(env, host))
    return out


def build_identity(tc: Dict[str, str]) -> Dict[str, str]:
    """The :data:`BUILD_FIELDS` of a :func:`toolchain`: the part of a psmc-rs identity that its build sets."""
    return {k: tc.get(k, "") for k in BUILD_FIELDS}


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
    identity = {"version": version, **source, **build_identity(tc)}
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
    identity = {"version": f"psmc-rs {m.group(1)}" if m else "unknown", "commit": head, **build_identity(tc)}
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


C_PSMC_DIR = "psmc_c"
C_BUILD_RECORD = "BUILD.json"
C_BUILD_RECORD_ID = "psmc-rs-manuscript/tools/c-psmc-build/1"
# What lh3/psmc's Makefile builds psmc from (OBJS and main.o), and its CFLAGS
# and LIBS. The utilities are built in utils/ with -I.. (they include
# ../kseq.h), as utils/Makefile does.
C_PSMC_SOURCES = ("khmm.c", "kmin.c", "cli.c", "core.c", "em.c", "aux.c", "main.c")
C_UTILITY_SOURCES = {"fq2psmcfa": "fq2psmcfa.c", "splitfa": "splitfa.c"}
C_DEFAULT_CFLAGS = "-g -Wall -O2 -Wno-unused-function"
# On macOS arm64 an -O2 build of C psmc segfaults; the laptop's is -O1.
C_MACOS_ARM64_CFLAGS = "-g -Wall -O1 -Wno-unused-function"
C_LIBS = "-lm -lz"


def c_psmc_dir(tools: Optional[Path] = None) -> Path:
    """``<tools>/psmc_c``: lh3/psmc's sources, its binary and :data:`C_BUILD_RECORD`."""
    return (tools or paths.tools_dir()) / C_PSMC_DIR


def default_cc() -> str:
    """The compiler :func:`build_c` uses by default: ``$CC``, else ``gcc`` (the Makefile's) if on PATH, else ``cc``."""
    return os.environ.get("CC") or ("gcc" if shutil.which("gcc") else "cc")


def default_cflags() -> str:
    """The CFLAGS :func:`build_c` uses by default: ``$PSMC_C_CFLAGS``, else the Makefile's (:data:`C_DEFAULT_CFLAGS`).

    On macOS arm64, where an ``-O2`` build segfaults, the default is the
    laptop's ``-O1`` (:data:`C_MACOS_ARM64_CFLAGS`).
    """
    macos_arm64 = sys.platform == "darwin" and platform.machine() == "arm64"
    return os.environ.get("PSMC_C_CFLAGS", C_MACOS_ARM64_CFLAGS if macos_arm64 else C_DEFAULT_CFLAGS)


def c_sources(src: Path) -> Dict[str, str]:
    """``{path relative to src: SHA-256}`` of lh3/psmc's sources: ``*.c``, ``*.h`` and ``Makefile``, and utils' own."""
    found = {p for pattern in ("*.c", "*.h", "Makefile", "utils/*.c", "utils/*.h", "utils/Makefile")
             for p in Path(src).glob(pattern) if p.is_file()}
    return {p.relative_to(src).as_posix(): sha256_file(p) for p in sorted(found)}


def build_c(src: Optional[Path] = None, cc: Optional[str] = None, cflags: Optional[str] = None,
            ldflags: Optional[str] = None, force: bool = False, log: Callable[[str], None] = print) -> Dict:
    """Build C psmc and its utilities fq2psmcfa and splitfa in ``src`` and write ``src/BUILD.json``.

    The compiler and flags are explicit (defaults: :func:`default_cc`,
    :func:`default_cflags`, ``$LDFLAGS``), and each program is compiled and
    linked by one command: ``CC CFLAGS [-I..] LDFLAGS -o PROG SOURCES -lm
    -lz``. The programs are built in a temporary directory and replace the
    old ones only when all of them built. ``BUILD.json`` records the
    compiler (path and the first line of its ``--version``), the flags, the
    commands, the SHA-256 of every source file and of every program built,
    and the platform; :func:`c_psmc` reads it. Returns the record.

    A program that the old ``BUILD.json`` does not describe (one built by
    hand, such as the laptop's) is not replaced unless ``force``: a build
    with ``-g`` is not byte-reproducible on macOS, so it could not be made
    again. Whatever is replaced is kept as ``<program>.prev`` and
    ``BUILD.json.prev``.
    """
    src = Path(src) if src is not None else c_psmc_dir()
    cc = cc or default_cc()
    compiler = shutil.which(cc)
    if compiler is None:
        raise ToolError(f"compiler {cc!r} not found (set CC)")
    cflags = default_cflags() if cflags is None else cflags
    ldflags = os.environ.get("LDFLAGS", "") if ldflags is None else ldflags
    targets = {"psmc": (src, C_PSMC_SOURCES, [])}
    targets.update({f"utils/{name}": (src / "utils", (source,), ["-I.."]) for name, source in C_UTILITY_SOURCES.items()})
    missing = [str(cwd / s) for cwd, sources, _ in targets.values() for s in sources if not (cwd / s).exists()]
    if missing:
        raise ToolError(f"lh3/psmc sources missing: {', '.join(missing)}")
    old = read_c_build(src)
    undescribed = [name for name in targets if (src / name).exists()
                   and (old or {}).get("binaries", {}).get(name) != sha256_file(src / name)]
    if undescribed and not force:
        raise ToolError(f"{', '.join(str(src / n) for n in undescribed)} not described by "
                        f"{src / C_BUILD_RECORD}: built otherwise, and not replaced (move them away, or pass --force "
                        "to keep them as <program>.prev)")
    sources = c_sources(src)
    try:
        res = _run([compiler, "--version"])
    except subprocess.TimeoutExpired:
        raise ToolError(f"{compiler} --version timed out") from None
    version = res.stdout.splitlines()[0].strip() if res.returncode == 0 and res.stdout.strip() else "unknown"
    commands: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(dir=src, prefix=".build-") as tmp:
        for name, (cwd, files, extra) in targets.items():
            def command(out: str) -> List[str]:
                return [compiler, *shlex.split(cflags), *extra, *shlex.split(ldflags), "-o", out, *files,
                        *shlex.split(C_LIBS)]

            commands[name] = shlex.join(command(Path(name).name))  # as run, but for where the program goes
            try:
                res = _run(command(str(Path(tmp) / Path(name).name)), cwd=cwd, timeout=600)
            except subprocess.TimeoutExpired:
                raise ToolError(f"building {name} timed out after 600 s: {commands[name]}") from None
            if res.returncode != 0:
                raise ToolError(f"building {name} failed: {commands[name]}\n{res.stderr[-4000:]}")
        for name in targets:
            if (src / name).exists():
                os.replace(src / name, src / f"{name}.prev")
            os.replace(Path(tmp) / Path(name).name, src / name)
    prev_record = src / f"{C_BUILD_RECORD}.prev"
    if (src / C_BUILD_RECORD).exists():
        shutil.copy2(src / C_BUILD_RECORD, prev_record)
    elif prev_record.exists():
        prev_record.unlink()  # it describes no <program>.prev now
    record = {
        "record": C_BUILD_RECORD_ID,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compiler": compiler,
        "compiler_version": version,
        "cflags": cflags,
        "ldflags": ldflags,
        "libs": C_LIBS,
        "commands": commands,
        "sources": sources,
        "binaries": {name: sha256_file(src / name) for name in targets},
        "platform": platform.platform(),
    }
    tmp_record = src / f".{C_BUILD_RECORD}.tmp"
    tmp_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_record, src / C_BUILD_RECORD)
    log(f"[tools] C psmc, fq2psmcfa and splitfa are built in {src} ({version}; CFLAGS {cflags})")
    return record


def read_c_build(src: Optional[Path] = None) -> Optional[Dict]:
    """The :data:`C_BUILD_RECORD` of ``src`` (default ``<tools>/psmc_c``), or None when there is none."""
    path = (Path(src) if src is not None else c_psmc_dir()) / C_BUILD_RECORD
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise ToolError(f"{path} is not valid JSON: {exc}") from None


def c_psmc(tools: Optional[Path] = None) -> Tool:
    """lh3/psmc at ``<tools>/psmc_c/psmc``.

    Its Makefile builds it with ``-O2``; on macOS arm64 an ``-O2`` build
    segfaults, so the one there is built with ``-O1``. Built by
    :func:`build_c`, its ``BUILD.json`` adds the compiler (``--version``)
    and the CFLAGS to the identity, and the binary must be the one it
    describes. Built otherwise (no ``BUILD.json``), its version and SHA-256
    identify it, as before ``BUILD.json`` existed.
    """
    binary = c_psmc_dir(tools) / "psmc"
    if not binary.exists():
        raise ToolError(f"C psmc not found at {binary} (set PSMC_EXP_TOOLS)")
    digest = sha256_file(binary)
    identity = {"version": c_psmc_version(binary), "sha256": digest}
    build = read_c_build(binary.parent)
    if build is None:
        return Tool("c-psmc", binary, identity)
    if build.get("binaries", {}).get("psmc") != digest:
        raise ToolError(f"{binary} is not the binary {binary.parent / C_BUILD_RECORD} describes; rebuild it with: "
                        "python -m experiment.pipeline tools --build-c")
    identity.update(cc=build["compiler_version"], cflags=build["cflags"])
    info = {"cc_path": build["compiler"], "ldflags": build.get("ldflags", ""), "libs": build.get("libs", ""),
            "built_on": build.get("platform", "")}
    return Tool("c-psmc", binary, identity, info)


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


LOCK = paths.EXPERIMENT / "pipeline" / "requirements.lock"


def read_lock(path: Path = LOCK) -> Dict[str, str]:
    """``{package: version}`` of the requirements lock (``python`` from its ``# python==`` line)."""
    out: Dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("# python=="):
            out["python"] = line.split("==", 1)[1]
        elif line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            out[name.strip().lower()] = version.strip()
    return out


def lock_differences(path: Path = LOCK) -> List[str]:
    """How the running interpreter differs from the requirements lock (empty when it matches)."""
    import platform
    from importlib import metadata

    out = []
    for name, want in read_lock(path).items():
        if name == "python":
            have = platform.python_version()
        else:
            try:
                have = metadata.version(name)
            except metadata.PackageNotFoundError:
                have = "missing"
        if have != want:
            out.append(f"{name} {have} (the lock has {want})")
    return out


def describe(tools: Sequence[Tool]) -> str:
    lines = []
    for t in tools:
        ident = ", ".join(f"{k}={v}" for k, v in {**t.identity, **t.info}.items() if v)
        lines.append(f"{t.name:24s} {t.path}\n{'':24s} {ident}")
    return "\n".join(lines)
