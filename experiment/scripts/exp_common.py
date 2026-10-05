"""Shared helpers for the experiment scripts.

Everything here is located relative to this file, so the scripts behave the
same whatever the current working directory is and contain no machine-specific
paths. Tool locations follow one rule everywhere:

    command-line flag  >  environment variable  >  default

* psmc-rs: ``--psmc-rs`` > ``$PSMC_RS_BIN`` > ``<repo>/target/release/psmc-rs``
  (built with ``cargo build --release`` when missing).
* C psmc:  ``--c-psmc``  > ``$C_PSMC_BIN``  > ``psmc`` found on ``$PATH``.
* simulation script: ``--sim-script`` > ``$SIM_SCRIPT`` >
  ``experiment/scripts/simulate_msprime_to_psmcfa.py``.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def resolve_sim_script(cli_value: Optional[str]) -> Path:
    """Locate the simulation script: flag, else ``$SIM_SCRIPT``, else the packaged one."""
    explicit = cli_value or os.environ.get("SIM_SCRIPT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return SIM_SCRIPT_DEFAULT


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


# Simulated demographic models shared by run_main_text.py and run_supplementary.py.
# Times are in units of 4*N0 generations and sizes are ratios to N0 (ms convention);
# simulate_msprime_to_psmcfa.py converts them to generations.
MODELS: Dict[str, Dict] = {
    "constant": {
        "title": "Constant",
        "sim_model": "constant",
        "sim_ne": 10_000.0,
        "sim_seed": 42,
        "true_kind": "constant",
        "true_params": {"ne": 10_000.0},
    },
    "bottleneck": {
        "title": "Bottleneck",
        "sim_model": "bottleneck",
        "sim_ne": 20_000.0,
        "sim_seed": 43,
        "true_kind": "ms_piecewise",
        "true_params": {
            "ne0": 20_000.0,
            "events": [(0.01, 0.05), (0.015, 0.5), (0.05, 0.25), (0.5, 0.5)],
        },
    },
    "expansion": {
        "title": "Expansion",
        "sim_model": "expansion",
        "sim_ne": 10_000.0,
        "sim_seed": 44,
        "true_kind": "ms_piecewise",
        "true_params": {
            "ne0": 10_000.0,
            "events": [(0.01, 0.1), (0.06, 1.0), (0.2, 0.5), (1.0, 1.0), (2.0, 2.0)],
        },
    },
    "zigzag": {
        "title": "Zigzag",
        "sim_model": "sim2_zigzag",
        "sim_ne": 1_000.0,
        "sim_seed": 45,
        "true_kind": "ms_piecewise",
        "true_params": {
            "ne0": 1_000.0,
            "events": [(0.1, 5.0), (0.6, 20.0), (2.0, 5.0), (10.0, 10.0), (20.0, 5.0)],
        },
    },
}
MODEL_ORDER = ["constant", "bottleneck", "expansion", "zigzag"]


@dataclass(frozen=True)
class MatchedSettings:
    """EM settings applied identically to psmc-rs and C psmc.

    Defaults are the manuscript settings; each field can be overridden through
    the environment variable named in :meth:`from_env`. Every setting that
    shapes the fit is passed to psmc-rs explicitly, so the comparison does not
    depend on its defaults: the pattern (which also fixes the number of
    intervals, as in C psmc), ``t_max``, the initial ``theta0/rho0`` ratio
    (C's ``-r``, psmc-rs ``--tr-ratio``), no smoothing (C has none) and the
    M-step iteration cap. psmc-rs runs single-threaded for a fair runtime
    comparison; ``batch_size`` (``--batch-size``) is passed only when set,
    since from psmc-rs 0.2 on it changes neither results nor memory.
    """

    n_iter: int = 20
    t_max: float = 15.0
    pattern: str = "4+25*2+4+6"
    rho_t_ratio: int = 4
    smooth_lambda: float = 0.0
    mstep_iters: int = 100
    threads: int = 1
    batch_size: Optional[int] = None

    @classmethod
    def from_env(cls) -> "MatchedSettings":
        env = os.environ
        d = cls()
        batch = env.get("BATCH_SIZE", "").strip()
        return cls(
            n_iter=int(env.get("N_ITER", d.n_iter)),
            t_max=float(env.get("T_MAX", d.t_max)),
            pattern=env.get("PATTERN", d.pattern),
            rho_t_ratio=int(env.get("RHO_T_RATIO", d.rho_t_ratio)),
            smooth_lambda=float(env.get("SMOOTH_LAMBDA", d.smooth_lambda)),
            mstep_iters=int(env.get("MSTEP_ITERS", d.mstep_iters)),
            threads=int(env.get("RUST_THREADS", d.threads)),
            batch_size=int(batch) if batch else d.batch_size,
        )

    def rust_model_args(self) -> List[str]:
        """The psmc-rs options that set the starting model, as C psmc's ``-p -t -r``."""
        return [
            "--pattern",
            self.pattern,
            "--t-max",
            str(self.t_max),
            "--tr-ratio",
            str(self.rho_t_ratio),
        ]

    def rust_cmd(
        self, psmc_rs: Path, inp: Path, out_json: Path, *, mu: float, extra: Sequence[str] = ()
    ) -> List[str]:
        batch = ["--batch-size", str(self.batch_size)] if self.batch_size else []
        return [
            str(psmc_rs),
            str(inp),
            str(out_json),
            str(self.n_iter),
            *self.rust_model_args(),
            "--smooth-lambda",
            str(self.smooth_lambda),
            "--mstep-iters",
            str(self.mstep_iters),
            "--mu",
            str(mu),
            "--threads",
            str(self.threads),
            *batch,
            "--no-progress",
            *extra,
        ]

    def c_cmd(self, c_psmc: Path, inp: Path, out_psmc: Path) -> List[str]:
        return [
            str(c_psmc),
            f"-N{self.n_iter}",
            f"-t{self.t_max}",
            f"-r{self.rho_t_ratio}",
            "-p",
            self.pattern,
            "-o",
            str(out_psmc),
            str(inp),
        ]

    def check_psmc_rs(self, psmc_rs: Path) -> None:
        """Exit unless ``psmc_rs``, run with these settings, starts from C psmc's theta/rho ratio."""
        check_rho_ratio(psmc_rs, self.rho_t_ratio, self.rust_model_args())


# 2,000 bins, one in 20 heterozygous: enough for psmc-rs to set theta0.
_RATIO_PROBE_SEQ = ("T" * 19 + "K") * 100
_RATIOS_CHECKED: set = set()


def psmc_rs_start_ratio(psmc_rs: Path, args: Sequence[str] = ()) -> float:
    """``theta0 / rho0`` that ``psmc_rs`` starts EM from when run with ``args``.

    Read from a zero-iteration run on a small synthetic input, which writes
    the starting parameters. Exits with an explanation when the run fails,
    for example because a psmc-rs build older than 0.2 has no ``--tr-ratio``.
    """
    with tempfile.TemporaryDirectory(prefix="psmc-rs-ratio-") as tmp:
        inp = Path(tmp) / "probe.psmcfa"
        seq = _RATIO_PROBE_SEQ
        inp.write_text(">probe\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n")
        out = Path(tmp) / "probe.json"
        res = subprocess.run(
            [str(psmc_rs), str(inp), str(out), "0", "--no-progress", *args], capture_output=True, text=True
        )
        if res.returncode != 0 or not out.exists():
            hint = ""
            if "--tr-ratio" in res.stderr:
                hint = (
                    "\nThis psmc-rs build has no --tr-ratio, which psmc-rs 0.2 added: build this "
                    "checkout (cargo build --release --locked) or pass a 0.2 binary with --psmc-rs."
                )
            raise SystemExit(
                f"error: cannot read the starting theta/rho ratio of {psmc_rs}: a zero-iteration run "
                f"with {' '.join(args)} failed (rc={res.returncode}).\n{res.stderr.strip()}{hint}"
            )
        params = json.loads(out.read_text())
    return float(params["theta"]) / float(params["rho"])


def check_rho_ratio(psmc_rs: Path, ratio: float, args: Sequence[str] = ()) -> None:
    """Exit unless ``psmc_rs`` run with ``args`` starts from ``rho0 = theta0 / ratio``.

    ``ratio`` is what C psmc gets as ``-r``; otherwise the two tools start EM
    from different parameters and the comparison is not matched. ``args`` are
    the model options of the real runs (:meth:`MatchedSettings.rust_model_args`,
    which pass ``--tr-ratio``), so this checks that the binary honours them.
    Each binary and set of options is probed once per process.
    """
    key = (str(psmc_rs), float(ratio), tuple(args))
    if key in _RATIOS_CHECKED:
        return
    found = psmc_rs_start_ratio(psmc_rs, args)
    if not math.isclose(found, ratio, rel_tol=1e-6):
        raise SystemExit(
            f"error: {psmc_rs} starts EM from rho0 = theta0/{found:.6g}, but C psmc is run with "
            f"-r{ratio:g} (RHO_T_RATIO), so the two tools would not start from the same parameters.\n"
            "psmc-rs 0.2 starts from rho0 = theta0/--tr-ratio (default 4, as lh3/psmc); psmc-rs 0.1 "
            "always used theta0/5. Use a psmc-rs 0.2 build (the packaged 0.1 results were made with "
            "RHO_T_RATIO=5)."
        )
    _RATIOS_CHECKED.add(key)


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


def run_staged(cmd_for: Callable[[Path], Sequence[str]], out: Path, **kwargs) -> Dict:
    """Run ``cmd_for(tmp)`` and move ``tmp`` to ``out`` only if the command succeeds.

    ``tmp`` is ``out`` plus a ``.partial`` suffix, in the same directory, so
    ``out`` never holds the output of an interrupted or failed run. Used for
    C psmc, which writes every EM round to its output as it goes. Keyword
    arguments go to :func:`run_command`.
    """
    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    rec = run_command(cmd_for(tmp), **kwargs)
    if rec["returncode"] == 0 and tmp.exists():
        os.replace(tmp, out)
    return rec


def c_psmc_rounds(path: Path) -> int:
    """Number of complete EM rounds (``RD`` blocks closed by ``//``) in a C psmc output."""
    try:
        text = Path(path).read_text()
    except OSError:
        return 0
    rounds = 0
    in_round = False
    for ln in text.splitlines():
        if ln.startswith("RD\t"):
            in_round = True
        elif ln.startswith("//") and in_round:
            rounds += 1
            in_round = False
    return rounds


def c_psmc_complete(path: Path, n_iter: int) -> bool:
    """Whether ``path`` holds a finished C psmc run of ``n_iter`` iterations.

    A finished run has ``n_iter + 1`` rounds (round 0 is the starting point).
    Fewer rounds mean an interrupted run, whose last round must not be taken
    as the result; a different count means other settings.
    """
    return c_psmc_rounds(path) == n_iter + 1


def c_psmc_loglikes(path: Path) -> List[Tuple[int, float]]:
    """``(round, LK)`` of every round of a C psmc output but round 0.

    C psmc's round k (k >= 1) prints the log-likelihood of the parameters
    after k - 1 EM iterations, computed by that round's E-step. Round 0 runs
    no E-step and prints a placeholder ``LK 0.000000``, which is left out.
    """
    out: List[Tuple[int, float]] = []
    rd: Optional[int] = None
    for ln in Path(path).read_text().splitlines():
        f = ln.split("\t")
        if f[0] == "RD" and len(f) > 1:
            rd = int(f[1])
        elif f[0] == "LK" and len(f) > 1 and rd:
            out.append((rd, float(f[1])))
    return out


def psmc_rs_loglikes(path: Path, n_iter: int) -> Optional[List[Tuple[int, float]]]:
    """psmc-rs's log-likelihoods of a run of ``n_iter`` iterations, numbered like C psmc's rounds.

    The output JSON's ``loglike[i]`` (psmc-rs >= 0.2) is the log-likelihood
    after i iterations, which C psmc prints as the LK of round i + 1, so this
    returns ``(i + 1, loglike[i])`` for i < ``n_iter``, the rounds an
    ``n_iter``-iteration C run prints (the last value, of the written model,
    has no C counterpart). None when the file is missing, unreadable, from
    another ``n_iter``, or from a version that wrote no trace (0.1).
    """
    try:
        params = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(params, dict) or params.get("n_iter") != n_iter:
        return None
    trace = params.get("loglike")
    if not isinstance(trace, list) or len(trace) != n_iter + 1:
        return None
    return [(i + 1, float(trace[i])) for i in range(n_iter)]


def psmc_rs_output_ok(path: Path) -> bool:
    """Whether ``path`` is a readable psmc-rs parameter JSON.

    psmc-rs writes the JSON once, after its last EM iteration, so a file that
    parses is a finished run; an interrupted write leaves invalid JSON.
    """
    try:
        params = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(params, dict) and all(k in params for k in ("theta", "rho", "lam", "n_steps", "t_max"))


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


# ---------------------------------------------------------------------------
# Ne(t) curves. Same definitions as the plotting notebook, without numpy.
# ---------------------------------------------------------------------------

Curve = Tuple[List[float], List[float]]


def _pattern_groups(pattern: Optional[str], *, legacy: bool = False) -> Optional[List[Tuple[int, int]]]:
    """``(count, width)`` groups of a pattern such as ``4+25*2+4+6``.

    C psmc reads a plain part ``a`` as one free parameter spanning ``a``
    intervals. psmc-rs development builds before v0.1.1 read it as ``a``
    parameters of width 1 (``legacy``), and their JSON files hold one value
    per such parameter plus a trailing one.
    """
    if pattern is None:
        return None
    out = []
    for part in str(pattern).split("+"):
        part = part.strip()
        if not part:
            continue
        if "*" in part:
            a, b = part.split("*", 1)
            out.append((int(a), int(b)))
        elif legacy:
            out.append((int(part), 1))
        else:
            out.append((1, int(part)))
    return out or None


def expand_lam(lam_grouped: Sequence[float], n_steps: int, pattern: Optional[str]) -> List[float]:
    """Expand one lambda per free parameter to one per time interval (n_steps + 1)."""
    lam_grouped = [float(x) for x in lam_grouped]
    groups = _pattern_groups(pattern)
    if groups is None:
        if len(lam_grouped) != n_steps + 1:
            raise ValueError(f"lam length {len(lam_grouped)} != n_steps+1 ({n_steps + 1})")
        return lam_grouped
    if len(lam_grouped) == sum(nr for nr, _ in groups):
        lam = []
        idx = 0
        for nr, gl in groups:
            for _ in range(nr):
                lam.extend([lam_grouped[idx]] * gl)
                idx += 1
        if len(lam) != n_steps + 1:
            raise ValueError(f"expanded lam length {len(lam)} != n_steps+1 ({n_steps + 1})")
        return lam
    # Legacy files: `a*b` read as a states of width b, a plain `a` as a states
    # of width 1, plus one trailing value.
    legacy = _pattern_groups(pattern, legacy=True)
    if len(lam_grouped) == sum(a for a, _ in legacy) + 1:
        lam = []
        idx = 0
        for a, b in legacy:
            for _ in range(a):
                lam.extend([lam_grouped[idx]] * b)
                idx += 1
        lam.append(lam_grouped[-1])
        if len(lam) != n_steps + 1:
            raise ValueError(f"expanded legacy lam length {len(lam)} != n_steps+1 ({n_steps + 1})")
        return lam
    raise ValueError("grouped lam length mismatch with pattern")


def curve_from_json(path: Path, mu: float, gen_years: float, bin_size: int = 100) -> Curve:
    """Ne(years) step curve from a psmc-rs parameter JSON."""
    params = json.loads(Path(path).read_text())
    theta = float(params["theta"])
    n_steps = int(params["n_steps"])
    t_max = float(params["t_max"])
    lam = expand_lam(params["lam"], n_steps, params.get("pattern"))
    alpha = 0.1
    beta = math.log(1 + t_max / alpha) / n_steps
    t = [alpha * (math.exp(beta * k) - 1.0) for k in range(n_steps)] + [t_max]
    n0 = theta / (4.0 * mu * bin_size)
    xs = [tk * 2.0 * gen_years * n0 for tk in t] + [1e8]
    ys = [lk * n0 for lk in lam] + [lam[-1] * n0]
    return xs, ys


def curve_from_c_psmc(path: Path, mu: float, gen_years: float, bin_size: int = 100) -> Curve:
    """Ne(years) step curve from the last complete round of a C psmc output."""
    blocks = []
    cur = None
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("RD\t"):
            if cur is not None:
                blocks.append(cur)
            cur = {"tr": None, "pa": None, "rs": []}
        elif cur is not None and ln.startswith("TR\t"):
            _, th, rh = ln.split("\t")[:3]
            cur["tr"] = (float(th), float(rh))
        elif cur is not None and ln.startswith("PA\t"):
            cur["pa"] = ln
        elif cur is not None and ln.startswith("RS\t"):
            f = ln.split("\t")
            cur["rs"].append((int(f[1]), float(f[2]), float(f[3])))
    if cur is not None:
        blocks.append(cur)
    best = next((b for b in reversed(blocks) if b["pa"] and b["tr"] is not None and b["rs"]), None)
    if best is None:
        raise ValueError(f"cannot parse valid block in {path}")
    n0 = best["tr"][0] / (4.0 * mu * bin_size)
    xs = [2.0 * n0 * tk * gen_years for _, tk, _ in best["rs"]] + [1e8]
    ys = [n0 * lk for _, _, lk in best["rs"]]
    ys.append(ys[-1])
    return xs, ys


def step_value(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    idx = bisect.bisect_right(xs, xq) - 1
    return float(ys[max(0, min(idx, len(ys) - 1))])


def rmse_log10(a: Curve, b: Curve, x_min: float = 1e3, x_max: float = 1e8, n: int = 500) -> float:
    """RMSE of log10 Ne between two step curves on a log-spaced grid of years."""
    lo, hi = math.log10(x_min), math.log10(x_max)
    total = 0.0
    for i in range(n):
        x = 10 ** (lo + (hi - lo) * i / (n - 1))
        va = max(step_value(a[0], a[1], x), 1e-12)
        vb = max(step_value(b[0], b[1], x), 1e-12)
        total += (math.log10(va) - math.log10(vb)) ** 2
    return math.sqrt(total / n)
