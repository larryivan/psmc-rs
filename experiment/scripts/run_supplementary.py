#!/usr/bin/env python3
"""Run supplementary experiments S1-S5 and export publication figures/tables.

S1: smoothness ablation (A/B/C)
S2: EM convergence (log-likelihood vs iteration)
S3: memory scaling by sequence length
S4: Rust thread scaling (with ideal speedup line)
S5: format consistency (psmcfa/mhs/vcf)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
    import numpy as np
    import pandas as pd
    HAS_PY_DEPS = True
    PY_DEPS_ERROR = None
except Exception as e:
    plt = None
    np = None
    pd = None
    HAS_PY_DEPS = False
    PY_DEPS_ERROR = e

try:
    import psutil

    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Cargo.toml").exists() and (p / "src").exists():
            return p
    raise RuntimeError(f"Cannot locate psmc-rs root from {start}")


ROOT = find_repo_root(Path.cwd().resolve())
RUN_DIR = ROOT / "experiment" / "runs" / "supplementary"
FIG_DIR = RUN_DIR / "figures"
TABLE_DIR = RUN_DIR / "tables"
LOG_DIR = RUN_DIR / "logs"
SHARED_INPUT_DIR = RUN_DIR / "shared_inputs"
S1_DIR = RUN_DIR / "s1_smooth"
S2_DIR = RUN_DIR / "s2_em"
S3_DIR = RUN_DIR / "s3_memory"
S4_DIR = RUN_DIR / "s4_threads"
S5_DIR = RUN_DIR / "s5_format"

for d in (
    RUN_DIR,
    FIG_DIR,
    TABLE_DIR,
    LOG_DIR,
    SHARED_INPUT_DIR,
    S1_DIR,
    S2_DIR,
    S3_DIR,
    S4_DIR,
    S5_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

for d in (
    S1_DIR / "outputs",
    S2_DIR / "outputs",
    S3_DIR / "outputs",
    S4_DIR / "inputs",
    S4_DIR / "outputs",
    S5_DIR / "inputs",
    S5_DIR / "outputs",
):
    d.mkdir(parents=True, exist_ok=True)


PSMC_RS_BIN = Path(os.environ.get("PSMC_RS_BIN", str(ROOT / "target" / "release" / "psmc-rs")))
C_PSMC_BIN = Path(os.environ.get("C_PSMC_BIN", str(ROOT.parent / "psmc-master" / "psmc")))
C_UTILS_DIR = Path(os.environ.get("C_UTILS_DIR", str(ROOT.parent / "psmc-master" / "utils")))
SPLITFA_BIN = Path(os.environ.get("SPLITFA_BIN", str(C_UTILS_DIR / "splitfa")))
SIM_SCRIPT = Path(
    os.environ.get(
        "SIM_SCRIPT",
        str(ROOT / "experiment" / "scripts" / "simulate_msprime_to_psmcfa.py"),
    )
)

MU = float(os.environ.get("MU", "2.5e-8"))
GEN_YEARS = float(os.environ.get("GEN_YEARS", "25"))
BIN_SIZE = int(os.environ.get("BIN_SIZE", "100"))
N_ITER = int(os.environ.get("N_ITER", "20"))
T_MAX = float(os.environ.get("T_MAX", "15"))
N_STEPS = int(os.environ.get("N_STEPS", "64"))
PATTERN = os.environ.get("PATTERN", "4+25*2+4+6")
RHO_T_RATIO = int(os.environ.get("RHO_T_RATIO", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "300000"))
RUST_THREADS_BASE = int(os.environ.get("RUST_THREADS_BASE", "1"))
ALPHA_CACHE_MB = int(os.environ.get("PSMC_ALPHA_CACHE_MB", "2048"))

SIM_LENGTH_BP = int(os.environ.get("SIM_LENGTH_BP", "500000000"))
SIM_WINDOW_BP = int(os.environ.get("SIM_WINDOW_BP", "100"))
SIM_MUTATION = float(os.environ.get("SIM_MUTATION", str(MU)))
SIM_RECOMB = os.environ.get("SIM_RECOMB", "").strip() or None

PERF_REPEATS = int(os.environ.get("PERF_REPEATS", "10"))
S3_REPEATS = int(os.environ.get("S3_REPEATS", "3"))
THREAD_REPEATS = int(os.environ.get("THREAD_REPEATS", "3"))
THREAD_LIST = [int(x.strip()) for x in os.environ.get("THREAD_LIST", "1,2,4,8").split(",") if x.strip()]
S4_CONTIGS = int(os.environ.get("S4_CONTIGS", "8"))
S4_CONTIG_MB = int(os.environ.get("S4_CONTIG_MB", "75"))
S4_SEED_BASE = int(os.environ.get("S4_SEED_BASE", "7400"))
S3_LENGTHS_MB = [
    int(x.strip()) for x in os.environ.get("S3_LENGTHS_MB", "100,250,500,1000").split(",") if x.strip()
]
if os.environ.get("S3_INCLUDE_2000", "0") == "1" and 2000 not in S3_LENGTHS_MB:
    S3_LENGTHS_MB.append(2000)
S3_LENGTHS_MB = sorted(set(S3_LENGTHS_MB))

BOOTSTRAP_REPS = int(os.environ.get("BOOTSTRAP_REPS", "100"))
BOOTSTRAP_ITERS = int(os.environ.get("BOOTSTRAP_ITERS", str(N_ITER)))
BOOTSTRAP_BLOCK_SIZE = int(os.environ.get("BOOTSTRAP_BLOCK_SIZE", "50000"))
BOOTSTRAP_SEED = int(os.environ.get("BOOTSTRAP_SEED", "42"))
S5_FORMAT_VERSION = 2


def _parse_float_list(env_name: str, default_csv: str) -> List[float]:
    raw = os.environ.get(env_name, default_csv)
    vals: List[float] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError(f"{env_name} cannot be empty")
    return vals


S1_SMOOTH_LAMBDAS = sorted(set(_parse_float_list("S1_SMOOTH_LAMBDAS", "0,1e-5,1e-4,1e-3,1e-2")))

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

COLORS = {
    "true": "#3B82F6",
    "rust": "#E15759",
    "rust_alt": "#F28E2B",
    "c": "#59A14F",
    "mhs": "#4E79A7",
    "vcf": "#B6992D",
    "grid": "#DCE3EC",
    "axis": "#425466",
    "text": "#1F2A37",
}


def setup_publication_style():
    """Consistent publication-style theme for all supplementary figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F9FBFD",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#B8C4D6",
            "axes.linewidth": 1.0,
            "axes.labelcolor": COLORS["axis"],
            "axes.titlecolor": COLORS["text"],
            "axes.titleweight": "semibold",
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.9,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "font.size": 10.5,
            "font.family": "DejaVu Sans",
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_labels(axes, labels="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    for i, ax in enumerate(axes):
        ax.text(
            0.01,
            0.99,
            labels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS["text"],
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#D0D8E5", "lw": 0.8},
        )


def stylize_axis(ax, *, xlog: bool = False, yfmt: Optional[str] = "int"):
    if xlog:
        ax.set_xscale("log")
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.35, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA7B6")
    ax.spines["bottom"].set_color("#9AA7B6")
    ax.tick_params(axis="both", which="major", labelsize=10)
    if yfmt == "int":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    elif yfmt == "float3":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))



def step_with_outline(ax, x, y, *, color: str, lw: float = 2.0, ls="-", label: Optional[str] = None, zorder: int = 3, alpha: float = 1.0):
    """Draw step curves with white halo so overlaps remain visible."""
    line = ax.step(x, y, where="post", color=color, lw=lw, ls=ls, label=label, zorder=zorder, alpha=alpha)[0]
    line.set_path_effects(
        [
            pe.Stroke(linewidth=lw + 1.4, foreground="white", alpha=0.92),
            pe.Normal(),
        ]
    )
    return line


def curve_log10_values(curve: Tuple[np.ndarray, np.ndarray], x_grid: np.ndarray) -> np.ndarray:
    x, y = curve
    vals = np.asarray([max(step_value(x, y, xv), 1e-12) for xv in x_grid], dtype=float)
    return np.log10(vals)


def ensure_pydeps():
    if HAS_PY_DEPS:
        setup_publication_style()
        return
    raise RuntimeError(
        "Missing Python plotting dependencies. Install with: "
        "pip install matplotlib numpy pandas msprime"
    ) from PY_DEPS_ERROR


def save_figure_multi(fig, stem: str):
    out = []
    for ext in ("png", "svg", "pdf"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            p,
            dpi=320 if ext == "png" else None,
            bbox_inches="tight",
            metadata={"Creator": "psmc-rs experiment/scripts/run_supplementary.py"},
        )
        out.append(p)
    print("saved figure:", ", ".join(str(x) for x in out))


def save_table_multi(df: pd.DataFrame, stem: str):
    p_csv = TABLE_DIR / f"{stem}.csv"
    p_tsv = TABLE_DIR / f"{stem}.tsv"
    p_md = TABLE_DIR / f"{stem}.md"
    df.to_csv(p_csv, index=False)
    df.to_csv(p_tsv, index=False, sep="\t")
    p_md.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"saved table: {p_csv}, {p_tsv}, {p_md}")


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    env: Optional[dict] = None,
    stdout_path: Optional[Path] = None,
):
    t0 = time.perf_counter()

    stdout_fh = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_fh = open(stdout_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        stdout=stdout_fh if stdout_fh is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    peak_rss_mb = float("nan")
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
                if rss > peak_bytes:
                    peak_bytes = rss
            except Exception:
                pass
        time.sleep(0.02)

    out, err = proc.communicate() if stdout_fh is None else ("", proc.stderr.read())
    if stdout_fh is not None:
        stdout_fh.close()

    dt = time.perf_counter() - t0
    if HAS_PSUTIL and ps_proc is not None:
        peak_rss_mb = peak_bytes / (1024**2)

    rec = {
        "cmd": " ".join(shlex.quote(x) for x in cmd),
        "returncode": proc.returncode,
        "stdout": out,
        "stderr": err,
        "wall_sec": dt,
        "peak_rss_mb": peak_rss_mb,
    }

    with (LOG_DIR / "commands.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if check and proc.returncode != 0:
        print(rec["cmd"])
        print("--- stdout ---")
        print(out)
        print("--- stderr ---")
        print(err)
        raise RuntimeError(f"command failed: rc={proc.returncode}")

    return rec


def ensure_tools(require_c: bool = True) -> bool:
    if not SIM_SCRIPT.exists():
        raise FileNotFoundError(f"simulation script missing: {SIM_SCRIPT}")

    if not PSMC_RS_BIN.exists():
        print("[build] cargo build --release")
        run_cmd(["cargo", "build", "--release"], cwd=ROOT)

    if not PSMC_RS_BIN.exists():
        raise FileNotFoundError(f"psmc-rs binary not found: {PSMC_RS_BIN}")

    has_c = C_PSMC_BIN.exists()
    if require_c and not has_c:
        raise FileNotFoundError(f"C psmc binary not found: {C_PSMC_BIN}")
    if not has_c:
        print(f"[warn] C binary not found: {C_PSMC_BIN}")
    return has_c


def ensure_splitfa() -> Path:
    if SPLITFA_BIN.exists():
        return SPLITFA_BIN
    splitfa_c = C_UTILS_DIR / "splitfa.c"
    if not splitfa_c.exists():
        raise FileNotFoundError(f"splitfa.c not found: {splitfa_c}")
    print(f"[build] cc -O3 -I.. -o {SPLITFA_BIN} {splitfa_c} -lm -lz")
    run_cmd(
        ["cc", "-O3", "-I..", "-o", str(SPLITFA_BIN), str(splitfa_c), "-lm", "-lz"],
        cwd=C_UTILS_DIR,
    )
    if not SPLITFA_BIN.exists():
        raise RuntimeError("failed to build splitfa")
    return SPLITFA_BIN


def shared_input_path(model_key: str, length_bp: int) -> Path:
    return SHARED_INPUT_DIR / f"{model_key}.{length_bp}bp.psmcfa"


def ensure_shared_inputs(length_bp: int, force: bool = False):
    for key in MODEL_ORDER:
        spec = MODELS[key]
        out_path = shared_input_path(key, length_bp)
        if out_path.exists() and not force:
            continue
        cmd = [
            sys.executable,
            str(SIM_SCRIPT),
            "--model",
            spec["sim_model"],
            "--out",
            str(out_path),
            "--length",
            str(length_bp),
            "--window",
            str(SIM_WINDOW_BP),
            "--mutation",
            str(SIM_MUTATION),
            "--seed",
            str(spec["sim_seed"]),
        ]
        if SIM_RECOMB:
            cmd += ["--recomb", str(SIM_RECOMB)]
        if spec.get("sim_ne"):
            cmd += ["--ne", str(spec["sim_ne"])]
        print("[simulate]", key, f"length={length_bp:,}")
        run_cmd(cmd, cwd=ROOT)


def s4_thread_input_path() -> Path:
    return S4_DIR / "inputs" / f"zigzag.{S4_CONTIGS}x{S4_CONTIG_MB}mb.psmcfa"


def _read_psmcfa_seq(path: Path) -> str:
    seq_parts: List[str] = []
    for ln in path.read_text().splitlines():
        if not ln:
            continue
        if ln.startswith(">"):
            continue
        seq_parts.append(ln.strip())
    return "".join(seq_parts)


def ensure_s4_parallel_input(force: bool = False) -> Path:
    """Build multi-sequence zigzag input so E-step sequence parallelism can scale."""
    out_path = s4_thread_input_path()
    if out_path.exists() and not force:
        return out_path

    spec = MODELS["zigzag"]
    contig_bp = int(S4_CONTIG_MB * 1_000_000)
    tmp_dir = S4_DIR / "inputs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(S4_CONTIGS):
            tmp_psmcfa = tmp_dir / f"zigzag.contig_{i+1:02d}.psmcfa"
            seed = int(S4_SEED_BASE + i)
            cmd = [
                sys.executable,
                str(SIM_SCRIPT),
                "--model",
                spec["sim_model"],
                "--out",
                str(tmp_psmcfa),
                "--length",
                str(contig_bp),
                "--window",
                str(SIM_WINDOW_BP),
                "--mutation",
                str(SIM_MUTATION),
                "--seed",
                str(seed),
            ]
            if SIM_RECOMB:
                cmd += ["--recomb", str(SIM_RECOMB)]
            if spec.get("sim_ne"):
                cmd += ["--ne", str(spec["sim_ne"])]
            print("[simulate]", f"S4 contig {i+1}/{S4_CONTIGS}", f"length={contig_bp:,}", f"seed={seed}")
            run_cmd(cmd, cwd=ROOT)

            seq = _read_psmcfa_seq(tmp_psmcfa)
            out.write(f">contig_{i+1}\n")
            for j in range(0, len(seq), 60):
                out.write(seq[j : j + 60] + "\n")

    return out_path


def run_rust(
    input_file: Path,
    output_json: Path,
    n_iter: int,
    *,
    input_format: str = "psmcfa",
    mhs_bin_size: int = BIN_SIZE,
    threads: int = 1,
    smooth_lambda: Optional[float] = None,
    extra: Optional[List[str]] = None,
):
    if smooth_lambda is None:
        # Fair C-vs-Rust baseline: disable smoothness unless explicitly requested.
        smooth_lambda = 0.0
    cmd = [
        str(PSMC_RS_BIN),
        str(input_file),
        str(output_json),
        str(n_iter),
        "--input-format",
        input_format,
        "--t-max",
        str(T_MAX),
        "--n-steps",
        str(N_STEPS),
        "--pattern",
        PATTERN,
        "--mu",
        str(MU),
        "--smooth-lambda",
        str(float(smooth_lambda)),
        "--batch-size",
        str(BATCH_SIZE),
        "--threads",
        str(max(1, threads)),
        "--no-progress",
    ]
    if input_format in ("mhs", "vcf"):
        cmd += ["--mhs-bin-size", str(mhs_bin_size)]
    if extra:
        cmd += extra

    env = os.environ.copy()
    env["PSMC_ALPHA_CACHE_MB"] = str(ALPHA_CACHE_MB)
    return run_cmd(cmd, cwd=ROOT, env=env)


def run_c(input_psmcfa: Path, output_psmc: Path, n_iter: int, bootstrap: bool = False):
    cmd = [
        str(C_PSMC_BIN),
        f"-N{n_iter}",
        f"-t{T_MAX}",
        f"-r{RHO_T_RATIO}",
        "-p",
        PATTERN,
    ]
    if bootstrap:
        cmd.append("-b")
    cmd += ["-o", str(output_psmc), str(input_psmcfa)]
    return run_cmd(cmd, cwd=ROOT)


def parse_pattern_spec(pattern):
    if pattern is None:
        return None
    out = []
    for part in str(pattern).split("+"):
        part = part.strip()
        if not part:
            continue
        if "*" in part:
            a, b = part.split("*", 1)
            nr = int(a.strip())
            gl = int(b.strip())
        else:
            nr = 1
            gl = int(part)
        if nr <= 0 or gl <= 0:
            raise ValueError(f"invalid pattern token: {part}")
        out.append((nr, gl))
    return out if out else None


def parse_pattern_spec_legacy(pattern):
    if pattern is None:
        return None
    out = []
    for part in str(pattern).split("+"):
        part = part.strip()
        if not part:
            continue
        if "*" in part:
            a, b = part.split("*", 1)
            ts = int(a.strip())
            gs = int(b.strip())
        else:
            ts = int(part)
            gs = 1
        out.append((ts, gs))
    return out if out else None


def expand_lam(lam_grouped, n_steps, pattern_spec, pattern_raw=None):
    lam_grouped = list(map(float, lam_grouped))
    if pattern_spec is None:
        if len(lam_grouped) != n_steps + 1:
            raise ValueError(f"lam length {len(lam_grouped)} != n_steps+1 ({n_steps+1})")
        return lam_grouped

    expected_c = sum(nr for nr, _ in pattern_spec)
    if len(lam_grouped) == expected_c:
        lam = []
        idx = 0
        for nr, gl in pattern_spec:
            for _ in range(nr):
                lam.extend([lam_grouped[idx]] * gl)
                idx += 1
        if len(lam) != n_steps + 1:
            raise ValueError(f"expanded lam length {len(lam)} != n_steps+1 ({n_steps+1})")
        return lam

    legacy = parse_pattern_spec_legacy(pattern_raw)
    expected_legacy = sum(ts for ts, _ in legacy) + 1 if legacy is not None else None
    if expected_legacy is not None and len(lam_grouped) == expected_legacy:
        lam = []
        idx = 0
        for ts, gs in legacy:
            for _ in range(ts):
                lam.extend([lam_grouped[idx]] * gs)
                idx += 1
        lam.append(lam_grouped[-1])
        if len(lam) != n_steps + 1:
            raise ValueError(f"expanded legacy lam length {len(lam)} != n_steps+1 ({n_steps+1})")
        return lam

    raise ValueError("grouped lam length mismatch with pattern")


def compute_t_grid(n_steps: int, t_max: float, alpha: float = 0.1):
    beta = math.log(1 + t_max / alpha) / n_steps
    t = [alpha * (math.exp(beta * k) - 1.0) for k in range(n_steps)]
    t.append(float(t_max))
    return np.asarray(t, dtype=float)


def curve_from_json(json_path: Path):
    params = json.loads(json_path.read_text())
    theta = float(params["theta"])
    mu = float(params.get("mu", MU))
    n_steps = int(params["n_steps"])
    t_max = float(params["t_max"])
    pattern_raw = params.get("pattern")
    pattern_spec = parse_pattern_spec(pattern_raw)
    lam = np.asarray(expand_lam(params["lam"], n_steps, pattern_spec, pattern_raw), dtype=float)

    t = compute_t_grid(n_steps, t_max)
    n0 = theta / (4.0 * mu * float(BIN_SIZE))
    x = t * 2.0 * float(GEN_YEARS) * n0
    y = lam * n0
    x = np.append(x, 1e8)
    y = np.append(y, y[-1])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), params


def load_c_curve(psmc_path: Path):
    if not psmc_path.exists():
        return None

    lines = psmc_path.read_text().splitlines()
    blocks = []
    cur = None
    for ln in lines:
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
            t = ln.split("\t")
            cur["rs"].append((int(t[1]), float(t[2]), float(t[3])))
    if cur is not None:
        blocks.append(cur)

    best = None
    for b in blocks[::-1]:
        if b["pa"] and b["tr"] is not None and b["rs"]:
            best = b
            break
    if best is None:
        return None

    theta = best["tr"][0]
    n0 = theta / (4.0 * float(MU) * float(BIN_SIZE))
    xs = []
    ys = []
    for _, tk, lk in best["rs"]:
        xs.append(2.0 * n0 * tk * float(GEN_YEARS))
        ys.append(n0 * lk)
    xs.append(1e8)
    ys.append(ys[-1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def true_curve_constant(ne: float):
    return np.asarray([1e3, 1e8], dtype=float), np.asarray([ne, ne], dtype=float)


def true_curve_ms_piecewise(ne0: float, events: List[Tuple[float, float]]):
    xs = [1e3]
    ys = [ne0]
    for t_4n0, ratio in sorted(events, key=lambda x: x[0]):
        t_gen = t_4n0 * 4.0 * ne0
        xs.append(max(1e3, t_gen * GEN_YEARS))
        ys.append(ratio * ne0)
    xs.append(1e8)
    ys.append(ys[-1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def true_curve_for_model(model_key: str):
    spec = MODELS[model_key]
    if spec["true_kind"] == "constant":
        return true_curve_constant(**spec["true_params"])
    if spec["true_kind"] == "ms_piecewise":
        return true_curve_ms_piecewise(**spec["true_params"])
    raise ValueError("unknown true_kind")


def step_value(xs: np.ndarray, ys: np.ndarray, xq: float) -> float:
    idx = int(np.searchsorted(xs, xq, side="right") - 1)
    idx = max(0, min(idx, len(ys) - 1))
    return float(ys[idx])


def rmse_log10(curve_a, curve_b, x_min=1e3, x_max=1e8, n=400):
    xa, ya = curve_a
    xb, yb = curve_b
    grid = np.geomspace(x_min, x_max, n)
    va = np.asarray([max(step_value(xa, ya, x), 1e-12) for x in grid], dtype=float)
    vb = np.asarray([max(step_value(xb, yb, x), 1e-12) for x in grid], dtype=float)
    return float(np.sqrt(np.mean((np.log10(va) - np.log10(vb)) ** 2)))


def parse_c_lk_trace(psmc_path: Path) -> pd.DataFrame:
    rd = None
    rows = []
    for ln in psmc_path.read_text().splitlines():
        if ln.startswith("RD\t"):
            rd = int(ln.split("\t")[1])
        elif ln.startswith("LK\t") and rd is not None:
            try:
                lk = float(ln.split("\t")[1])
                rows.append({"iter": rd, "loglike": lk})
            except Exception:
                pass
    return pd.DataFrame(rows).sort_values("iter")


def parse_rust_last_loglike(text: str) -> Optional[float]:
    m = re.search(r"Last EM loglike:\s*([-+0-9.eE]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def simulate_three_formats(model_key: str, length_bp: int, window_bp: int, force: bool = False):
    import msprime

    spec = MODELS[model_key]
    p_psmcfa = S5_DIR / "inputs" / f"{model_key}.{length_bp}bp.psmcfa"
    p_mhs = S5_DIR / "inputs" / f"{model_key}.{length_bp}bp.multihetsep"
    p_vcf = S5_DIR / "inputs" / f"{model_key}.{length_bp}bp.vcf"
    p_meta = S5_DIR / "inputs" / f"{model_key}.{length_bp}bp.meta.json"
    if (not force) and p_psmcfa.exists() and p_mhs.exists() and p_vcf.exists() and p_meta.exists():
        try:
            meta = json.loads(p_meta.read_text())
            if (
                int(meta.get("format_version", 0)) == S5_FORMAT_VERSION
                and int(meta.get("length_bp", 0)) == int(length_bp)
                and int(meta.get("window_bp", 0)) == int(window_bp)
            ):
                return p_psmcfa, p_mhs, p_vcf
        except Exception:
            pass

    n0 = float(spec["sim_ne"])

    def events_to_gen(ne0: float, events_4n0: List[Tuple[float, float]]):
        return [(t4 * 4.0 * ne0, ratio * ne0) for t4, ratio in events_4n0]

    events = []
    if spec["sim_model"] in ("bottleneck", "expansion", "sim2_zigzag"):
        events = events_to_gen(spec["true_params"]["ne0"], spec["true_params"]["events"])

    dem = msprime.Demography()
    dem.add_population(name="pop0", initial_size=n0)
    for t_gen, ne in sorted(events, key=lambda x: x[0]):
        dem.add_population_parameters_change(time=float(t_gen), population="pop0", initial_size=float(ne))

    ts = msprime.sim_ancestry(
        samples={"pop0": 1},
        ploidy=2,
        demography=dem,
        sequence_length=float(length_bp),
        recombination_rate=float(SIM_RECOMB) if SIM_RECOMB else 1.25e-8,
        random_seed=int(spec["sim_seed"]),
    )
    mts = msprime.sim_mutations(ts, rate=float(SIM_MUTATION), random_seed=int(spec["sim_seed"]) + 1)

    p_psmcfa.parent.mkdir(parents=True, exist_ok=True)
    with p_vcf.open("w", encoding="utf-8") as f:
        mts.write_vcf(f, contig_id="chr1")

    # Important: derive heterozygous sites from the actual VCF POS/GT encoding
    # so psmcfa, mhs, and vcf share the same coordinate system.
    het_sites: List[Tuple[int, str]] = []
    with p_vcf.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            pos = int(cols[1])
            fmt_keys = cols[8].split(":")
            sample_vals = cols[9].split(":")
            if "GT" not in fmt_keys:
                continue
            gt = sample_vals[fmt_keys.index("GT")]
            if gt in (".", "./.", ".|."):
                continue
            parts = gt.replace("|", "/").split("/")
            if len(parts) != 2 or "." in parts:
                continue
            try:
                i0 = int(parts[0])
                i1 = int(parts[1])
            except ValueError:
                continue
            ref = cols[3]
            alts = cols[4].split(",") if cols[4] else []

            def allele(idx: int) -> Optional[str]:
                if idx == 0:
                    return ref
                j = idx - 1
                if 0 <= j < len(alts):
                    return alts[j]
                return None

            a0 = allele(i0)
            a1 = allele(i1)
            if a0 is None or a1 is None:
                continue
            if a0 == a1:
                continue
            if 1 <= pos <= length_bp:
                het_sites.append((pos, f"{a0}{a1}"))

    n_bins = math.ceil(length_bp / window_bp)
    has_het = np.zeros(n_bins, dtype=bool)
    mhs_rows: List[Tuple[int, int, str]] = []
    prev_emit_pos = 0
    for pos, alleles in het_sites:
        has_het[(pos - 1) // window_bp] = True
        nr_called = pos - prev_emit_pos
        if nr_called <= 0:
            continue
        mhs_rows.append((pos, nr_called, alleles))
        prev_emit_pos = pos

    with p_psmcfa.open("w", encoding="utf-8") as f:
        f.write("> chr1\n")
        seq = "".join("K" if x else "T" for x in has_het)
        for i in range(0, len(seq), 60):
            f.write(seq[i : i + 60] + "\n")

    with p_mhs.open("w", encoding="utf-8") as f:
        for pos, nr_called, alleles in mhs_rows:
            f.write(f"chr1\t{pos}\t{nr_called}\t{alleles}\n")

    p_meta.write_text(
        json.dumps(
            {
                "format_version": S5_FORMAT_VERSION,
                "model": model_key,
                "length_bp": int(length_bp),
                "window_bp": int(window_bp),
                "seed": int(spec["sim_seed"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return p_psmcfa, p_mhs, p_vcf


def run_s1_smooth_ablation(force: bool = False):
    ensure_pydeps()
    print("[S1] smoothness ablation")
    print("[S1] lambda grid =", ", ".join(f"{x:g}" for x in S1_SMOOTH_LAMBDAS))
    ensure_tools(require_c=True)
    ensure_shared_inputs(SIM_LENGTH_BP, force=force)

    rows_long = []
    fig = plt.figure(figsize=(15.0, 12.6), dpi=220, constrained_layout=True)
    outer = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.15)
    main_axes = []
    diff_axes = []

    cmap = plt.get_cmap("magma")
    lam_colors = {}
    for i, lam in enumerate(S1_SMOOTH_LAMBDAS):
        if lam == 0.0:
            lam_colors[lam] = COLORS["rust"]
        else:
            frac = (i + 1) / (len(S1_SMOOTH_LAMBDAS) + 1)
            lam_colors[lam] = cmap(0.2 + 0.7 * frac)

    lam_min = min(S1_SMOOTH_LAMBDAS)
    lam_max = max(S1_SMOOTH_LAMBDAS)

    for i, key in enumerate(MODEL_ORDER):
        inp = shared_input_path(key, SIM_LENGTH_BP)
        c_psmc = S1_DIR / "outputs" / f"{key}.c.C.psmc"

        if force or not c_psmc.exists():
            run_c(inp, c_psmc, n_iter=N_ITER)

        true_curve = true_curve_for_model(key)
        c_curve = load_c_curve(c_psmc)
        rmse_c = rmse_log10(true_curve, c_curve)

        rust_curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        rmse_by_lam: Dict[float, float] = {}
        for lam in S1_SMOOTH_LAMBDAS:
            lam_tag = f"{lam:g}".replace("+", "").replace("/", "_")
            rust_json = S1_DIR / "outputs" / f"{key}.rust.lambda_{lam_tag}.json"
            if force or not rust_json.exists():
                run_rust(inp, rust_json, n_iter=N_ITER, threads=RUST_THREADS_BASE, smooth_lambda=lam)
            xr, yr, _ = curve_from_json(rust_json)
            rust_curves[lam] = (xr, yr)
            rmse_r = rmse_log10(true_curve, (xr, yr))
            rmse_by_lam[lam] = rmse_r
            rows_long.append(
                {
                    "model": key,
                    "smooth_lambda": lam,
                    "rmse_log10_ne_rust": rmse_r,
                    "rmse_log10_ne_c": rmse_c,
                }
            )

        rr, cc = divmod(i, 2)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[4.0, 1.35], hspace=0.05)
        ax = fig.add_subplot(inner[0])
        axd = fig.add_subplot(inner[1], sharex=ax)
        main_axes.append(ax)
        diff_axes.append(axd)

        tx, ty = true_curve
        step_with_outline(ax, tx, ty, ls=(0, (5, 3)), lw=2.25, color=COLORS["true"], label="True", zorder=1)
        step_with_outline(ax, c_curve[0], c_curve[1], lw=2.15, color=COLORS["c"], label="C", zorder=2)

        ymax = max(np.max(ty), np.max(c_curve[1]))
        for lam in S1_SMOOTH_LAMBDAS:
            xr, yr = rust_curves[lam]
            lw = 2.05 if lam == 0.0 else 1.65
            alpha = 0.96 if lam == 0.0 else 0.85
            label = f"Rust lambda={lam:g}"
            step_with_outline(ax, xr, yr, lw=lw, color=lam_colors[lam], alpha=alpha, label=label, zorder=3)
            ymax = max(ymax, np.max(yr))

        ax.set_xlim(1e3, 1e8)
        ax.set_ylim(0, ymax * 1.18)
        stylize_axis(ax, xlog=True)
        ax.set_title(MODELS[key]["title"], fontsize=12.5, pad=8)
        ax.set_ylabel("Effective population size (Ne)")
        ax.tick_params(labelbottom=False)

        # Delta strip: Rust(lambda) - C in log10 space.
        xg = np.geomspace(1e3, 1e8, 900)
        lc = curve_log10_values(c_curve, xg)
        max_abs = 0.0
        for lam in S1_SMOOTH_LAMBDAS:
            d = curve_log10_values(rust_curves[lam], xg) - lc
            max_abs = max(max_abs, float(np.max(np.abs(d))))
            lw = 1.8 if lam == 0.0 else 1.35
            alpha = 0.92 if (lam == lam_min or lam == lam_max or lam == 0.0) else 0.68
            lbl = None
            if lam == 0.0:
                lbl = "Delta (lambda=0 - C)"
            elif lam == lam_max and lam != 0.0:
                lbl = f"Delta (lambda={lam:g} - C)"
            axd.plot(xg, d, lw=lw, color=lam_colors[lam], alpha=alpha, label=lbl, zorder=3)

        axd.axhline(0.0, color="#6B7280", lw=1.0, ls=(0, (4, 2)), zorder=1)
        stylize_axis(axd, xlog=True, yfmt=None)
        axd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.2f}"))
        lim = max(0.03, min(0.60, max_abs * 1.25))
        axd.set_ylim(-lim, lim)
        axd.set_xlabel(f"Years (g={GEN_YEARS}, mu={MU:.1e})")
        axd.set_ylabel("Delta log10(Ne)", fontsize=9)
        axd.tick_params(axis="both", labelsize=8.2)

        best_lam = min(rmse_by_lam, key=rmse_by_lam.get)
        txt = (
            f"RMSE C={rmse_c:.3f}\n"
            f"RMSE Rust lambda=0: {rmse_by_lam.get(0.0, float('nan')):.3f}\n"
            f"Best Rust: {rmse_by_lam[best_lam]:.3f} @ lambda={best_lam:g}"
        )
        ax.text(
            0.985,
            0.975,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color=COLORS["axis"],
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#D2DAE6", "lw": 0.8, "alpha": 0.93},
        )

    panel_labels(main_axes)
    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, 1.03))
    h2, l2 = diff_axes[0].get_legend_handles_labels()
    if h2:
        diff_axes[0].legend(h2, l2, loc="lower left", fontsize=8.0, frameon=False)

    fig.suptitle("S1. Smoothness Lambda Impact (Curves + Delta Strips)", fontsize=15.5, fontweight="bold", y=1.05)
    save_figure_multi(fig, "S1_smooth_ablation")
    plt.close(fig)

    table_long = pd.DataFrame(rows_long).sort_values(["model", "smooth_lambda"])
    save_table_multi(table_long, "S1_rmse_ablation_long")

    # Keep historical filename for compatibility with previous notebooks/reports.
    table_wide = (
        table_long.pivot(index="model", columns="smooth_lambda", values="rmse_log10_ne_rust")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    c_rmse = (
        table_long.groupby("model", as_index=False)["rmse_log10_ne_c"]
        .first()
        .rename(columns={"rmse_log10_ne_c": "rmse_log10_ne_c"})
    )
    table_wide = table_wide.merge(c_rmse, on="model", how="left")
    rename_map = {}
    for col in table_wide.columns:
        if isinstance(col, float):
            rename_map[col] = f"rmse_log10_ne_rust_lambda_{col:g}"
    table_wide = table_wide.rename(columns=rename_map).sort_values("model")
    save_table_multi(table_wide, "S1_rmse_ablation")

    # Additional sensitivity summary figure: RMSE vs lambda (categorical scale).
    fig2, axes2 = plt.subplots(2, 2, figsize=(14.2, 10.2), dpi=220, constrained_layout=True)
    axes2 = axes2.ravel()
    x_idx = np.arange(len(S1_SMOOTH_LAMBDAS))
    x_labels = [f"{x:g}" for x in S1_SMOOTH_LAMBDAS]
    for i, key in enumerate(MODEL_ORDER):
        ax = axes2[i]
        sub = table_long[table_long["model"] == key]
        ys = [float(sub[sub["smooth_lambda"] == lam]["rmse_log10_ne_rust"].iloc[0]) for lam in S1_SMOOTH_LAMBDAS]
        yc = float(sub["rmse_log10_ne_c"].iloc[0])
        ax.plot(x_idx, ys, marker="o", ms=5.2, lw=2.1, color=COLORS["rust"], label="Rust")
        ax.axhline(yc, ls=(0, (5, 3)), lw=2.0, color=COLORS["c"], label="C baseline")
        stylize_axis(ax, xlog=False, yfmt=None)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, rotation=20)
        ax.set_xlabel("smooth lambda")
        ax.set_ylabel("RMSE (log10 Ne)")
        ax.set_title(MODELS[key]["title"], fontsize=12.5, pad=8)
    panel_labels(axes2)
    h2, l2 = axes2[0].get_legend_handles_labels()
    fig2.legend(h2, l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig2.suptitle("S1. Smoothness Lambda Impact (RMSE)", fontsize=15.5, fontweight="bold", y=1.05)
    save_figure_multi(fig2, "S1_smooth_lambda_rmse")
    plt.close(fig2)

    return table_wide


def run_s2_em_convergence(force: bool = False):
    ensure_pydeps()
    print("[S2] EM convergence")
    ensure_tools(require_c=True)
    ensure_shared_inputs(SIM_LENGTH_BP, force=force)

    trace_rows = []

    for key in MODEL_ORDER:
        inp = shared_input_path(key, SIM_LENGTH_BP)

        c_psmc = S2_DIR / "outputs" / f"{key}.c.trace.psmc"
        if force or not c_psmc.exists():
            run_c(inp, c_psmc, n_iter=N_ITER)
        c_df = parse_c_lk_trace(c_psmc)
        for _, r in c_df.iterrows():
            trace_rows.append({"model": key, "tool": "c", "iter": int(r["iter"]), "loglike": float(r["loglike"])})

        for it in range(1, N_ITER + 1):
            rust_json = S2_DIR / "outputs" / f"{key}.rust.iter{it:02d}.json"
            stamp = S2_DIR / "outputs" / f"{key}.rust.iter{it:02d}.loglike.txt"
            if force or not rust_json.exists() or not stamp.exists():
                rec = run_rust(inp, rust_json, n_iter=it, threads=RUST_THREADS_BASE, smooth_lambda=None)
                lk = parse_rust_last_loglike((rec.get("stdout") or "") + "\n" + (rec.get("stderr") or ""))
                if lk is None:
                    raise RuntimeError(f"failed to parse Rust loglike for {key} iter={it}")
                stamp.write_text(f"{lk}\n", encoding="utf-8")
            else:
                lk = float(stamp.read_text().strip())
            trace_rows.append({"model": key, "tool": "rust", "iter": it, "loglike": lk})

    trace_df = pd.DataFrame(trace_rows).sort_values(["model", "tool", "iter"])
    save_table_multi(trace_df, "S2_em_convergence_trace")

    fig = plt.figure(figsize=(14.8, 11.8), dpi=220, constrained_layout=True)
    outer = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.15)
    main_axes = []
    diff_axes = []

    for i, key in enumerate(MODEL_ORDER):
        rr, cc = divmod(i, 2)
        inner = outer[rr, cc].subgridspec(2, 1, height_ratios=[3.8, 1.35], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axd = fig.add_subplot(inner[1], sharex=ax)
        main_axes.append(ax)
        diff_axes.append(axd)

        sub = trace_df[trace_df.model == key]
        tt_r = sub[sub.tool == "rust"].sort_values("iter")
        tt_c = sub[sub.tool == "c"].sort_values("iter")

        ax.plot(
            tt_r["iter"],
            tt_r["loglike"],
            marker="o",
            ms=4.1,
            lw=2.1,
            color=COLORS["rust"],
            label="Rust",
        )
        ax.plot(
            tt_c["iter"],
            tt_c["loglike"],
            marker="o",
            ms=4.1,
            lw=2.1,
            color=COLORS["c"],
            label="C",
        )

        stylize_axis(ax, xlog=False, yfmt=None)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3e}"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(MODELS[key]["title"], fontsize=12.5, pad=8)
        ax.set_ylabel("log-likelihood")
        ax.tick_params(labelbottom=False)

        # Delta strip to resolve heavy overlap.
        common = sorted(set(tt_r["iter"].tolist()) & set(tt_c["iter"].tolist()))
        rr_vals = np.asarray([float(tt_r[tt_r["iter"] == k]["loglike"].iloc[0]) for k in common], dtype=float)
        cc_vals = np.asarray([float(tt_c[tt_c["iter"] == k]["loglike"].iloc[0]) for k in common], dtype=float)
        dvals = rr_vals - cc_vals

        axd.axhline(0.0, color="#6B7280", lw=1.0, ls=(0, (4, 2)))
        axd.plot(common, dvals, marker="o", ms=3.4, lw=1.8, color="#374151", label="Rust - C")
        stylize_axis(axd, xlog=False, yfmt=None)
        axd.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:+.2e}"))
        axd.xaxis.set_major_locator(MaxNLocator(integer=True))
        max_abs = float(np.max(np.abs(dvals))) if len(dvals) else 0.0
        lim = max(1e-7, min(1.0, max_abs * 1.35))
        axd.set_ylim(-lim, lim)
        axd.set_xlabel("EM iteration")
        axd.set_ylabel("Delta LL", fontsize=9)
        axd.tick_params(axis="both", labelsize=8.3)

    panel_labels(main_axes)
    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    h2, l2 = diff_axes[0].get_legend_handles_labels()
    if h2:
        diff_axes[0].legend(h2, l2, loc="lower left", fontsize=8.1, frameon=False)

    fig.suptitle("S2. EM Convergence: Rust vs C (with Delta Strip)", fontsize=15.5, fontweight="bold", y=1.05)
    save_figure_multi(fig, "S2_em_convergence")
    plt.close(fig)

    return trace_df


def run_s3_memory_scaling(force: bool = False):
    ensure_pydeps()
    print("[S3] memory scaling")
    ensure_tools(require_c=True)

    rows = []
    for length_mb in S3_LENGTHS_MB:
        length_bp = int(length_mb * 1_000_000)
        ensure_shared_inputs(length_bp, force=force)

        inp = shared_input_path("constant", length_bp)
        for rep in range(1, S3_REPEATS + 1):
            rust_out = S3_DIR / "outputs" / f"constant.{length_mb}mb.rust.rep{rep:02d}.json"
            if force or not rust_out.exists():
                r = run_rust(inp, rust_out, n_iter=N_ITER, threads=1, smooth_lambda=None)
            else:
                r = {"wall_sec": np.nan, "peak_rss_mb": np.nan}
            rows.append(
                {
                    "length_mb": length_mb,
                    "tool": "rust",
                    "rep": rep,
                    "wall_sec": r["wall_sec"],
                    "peak_rss_mb": r["peak_rss_mb"],
                }
            )

            c_out = S3_DIR / "outputs" / f"constant.{length_mb}mb.c.rep{rep:02d}.psmc"
            if force or not c_out.exists():
                c = run_c(inp, c_out, n_iter=N_ITER)
            else:
                c = {"wall_sec": np.nan, "peak_rss_mb": np.nan}
            rows.append(
                {
                    "length_mb": length_mb,
                    "tool": "c",
                    "rep": rep,
                    "wall_sec": c["wall_sec"],
                    "peak_rss_mb": c["peak_rss_mb"],
                }
            )

    raw = pd.DataFrame(rows)
    save_table_multi(raw, "S3_memory_scaling_raw")

    summ = (
        raw.groupby(["length_mb", "tool"], as_index=False)
        .agg(
            wall_sec_mean=("wall_sec", "mean"),
            wall_sec_std=("wall_sec", "std"),
            peak_rss_mb_mean=("peak_rss_mb", "mean"),
            peak_rss_mb_std=("peak_rss_mb", "std"),
        )
        .sort_values(["length_mb", "tool"])
    )
    save_table_multi(summ, "S3_memory_scaling_summary")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), dpi=220, constrained_layout=True)

    for ax, metric, metric_std, title in [
        (axes[0], "peak_rss_mb_mean", "peak_rss_mb_std", "Peak RSS vs sequence length"),
        (axes[1], "wall_sec_mean", "wall_sec_std", "Runtime vs sequence length"),
    ]:
        for tool, color in [("rust", COLORS["rust"]), ("c", COLORS["c"])]:
            sub = summ[summ.tool == tool].sort_values("length_mb")
            xvals = sub["length_mb"].to_numpy(dtype=float)
            yvals = sub[metric].to_numpy(dtype=float)
            ystd = sub[metric_std].fillna(0.0).to_numpy(dtype=float)
            ax.plot(xvals, yvals, marker="o", ms=5.5, lw=2.2, color=color, label=tool.upper())
            ax.fill_between(xvals, yvals - ystd, yvals + ystd, color=color, alpha=0.16, linewidth=0)
        stylize_axis(ax, xlog=True)
        ax.set_xticks(S3_LENGTHS_MB)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Sequence length (Mb)")
        ax.set_title(title, fontsize=12.5, pad=8)

    axes[0].set_ylabel("Peak RSS (MB)")
    axes[1].set_ylabel("Runtime (s)")
    axes[0].legend(frameon=False)
    panel_labels(axes)
    fig.suptitle("S3. Scaling with sequence length (constant model)", fontsize=15.5, fontweight="bold", y=1.05)
    save_figure_multi(fig, "S3_memory_scaling")
    plt.close(fig)

    return summ


def run_s4_thread_scaling(force: bool = False):
    ensure_pydeps()
    print("[S4] Rust thread scaling")
    ensure_tools(require_c=False)
    inp = ensure_s4_parallel_input(force=force)
    model = "zigzag"
    rows = []

    for t in sorted(set(THREAD_LIST)):
        for rep in range(1, THREAD_REPEATS + 1):
            out = S4_DIR / "outputs" / f"{model}.threads{t}.rep{rep:02d}.json"
            # Always re-run S4 timing so summary never becomes NaN from cached artifacts.
            r = run_rust(inp, out, n_iter=N_ITER, threads=t, smooth_lambda=None)
            rows.append({"threads": t, "rep": rep, "wall_sec": r["wall_sec"]})

    raw = pd.DataFrame(rows)
    save_table_multi(raw, "S4_thread_scaling_raw")

    summ = raw.groupby("threads", as_index=False).agg(wall_sec_mean=("wall_sec", "mean"), wall_sec_std=("wall_sec", "std"))
    base_thread = 1 if (summ["threads"] == 1).any() else int(summ["threads"].min())
    t_base = float(summ[summ.threads == base_thread]["wall_sec_mean"].iloc[0])
    summ["speedup"] = t_base / summ["wall_sec_mean"]
    save_table_multi(summ, "S4_thread_scaling_summary")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=220, constrained_layout=True)

    axes[0].errorbar(
        summ["threads"],
        summ["wall_sec_mean"],
        yerr=summ["wall_sec_std"],
        marker="o",
        lw=2.2,
        ms=5.5,
        capsize=3,
        color=COLORS["rust"],
        label="Rust",
    )
    stylize_axis(axes[0], xlog=False)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlabel("Threads")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Runtime", fontsize=12.5, pad=8)

    th = summ["threads"].to_numpy(dtype=float)
    axes[1].plot(
        th,
        summ["speedup"],
        marker="o",
        ms=5.5,
        lw=2.2,
        color=COLORS["rust"],
        label="Observed speedup",
    )
    axes[1].plot(
        th,
        th,
        ls=(0, (5, 3)),
        lw=2.0,
        color=COLORS["true"],
        label="Ideal linear",
    )
    stylize_axis(axes[1], xlog=False)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_xlabel("Threads")
    axes[1].set_ylabel("Speedup (x)")
    axes[1].set_title("Speedup vs ideal", fontsize=12.5, pad=8)
    axes[1].legend(frameon=False)

    panel_labels(axes)
    fig.suptitle(
        f"S4. Rust thread scaling (zigzag, {S4_CONTIGS}x{S4_CONTIG_MB}Mb; total {S4_CONTIGS*S4_CONTIG_MB}Mb)",
        fontsize=15.5,
        fontweight="bold",
        y=1.05,
    )
    save_figure_multi(fig, "S4_thread_scaling")
    plt.close(fig)

    return summ


def run_s5_format_consistency(force: bool = False):
    ensure_pydeps()
    print("[S5] format consistency")
    ensure_tools(require_c=False)

    rows = []

    for key in MODEL_ORDER:
        psmcfa_in, mhs_in, vcf_in = simulate_three_formats(key, SIM_LENGTH_BP, SIM_WINDOW_BP, force=force)

        p_psmcfa = S5_DIR / "outputs" / f"{key}.psmcfa.rust.json"
        p_mhs = S5_DIR / "outputs" / f"{key}.mhs.rust.json"
        p_vcf = S5_DIR / "outputs" / f"{key}.vcf.rust.json"

        if force or not p_psmcfa.exists():
            run_rust(psmcfa_in, p_psmcfa, n_iter=N_ITER, input_format="psmcfa", threads=RUST_THREADS_BASE, smooth_lambda=None)
        if force or not p_mhs.exists():
            run_rust(mhs_in, p_mhs, n_iter=N_ITER, input_format="mhs", mhs_bin_size=SIM_WINDOW_BP, threads=RUST_THREADS_BASE, smooth_lambda=None)
        if force or not p_vcf.exists():
            run_rust(vcf_in, p_vcf, n_iter=N_ITER, input_format="vcf", mhs_bin_size=SIM_WINDOW_BP, threads=RUST_THREADS_BASE, smooth_lambda=None)

        x_psmcfa, y_psmcfa, params_psmcfa = curve_from_json(p_psmcfa)
        x_mhs, y_mhs, params_mhs = curve_from_json(p_mhs)
        x_vcf, y_vcf, params_vcf = curve_from_json(p_vcf)

        rmse_mhs = rmse_log10((x_psmcfa, y_psmcfa), (x_mhs, y_mhs))
        rmse_vcf = rmse_log10((x_psmcfa, y_psmcfa), (x_vcf, y_vcf))
        rmse_vcf_mhs = rmse_log10((x_vcf, y_vcf), (x_mhs, y_mhs))

        lam0 = np.asarray(params_psmcfa["lam"], dtype=float)
        lam1 = np.asarray(params_mhs["lam"], dtype=float)
        lam2 = np.asarray(params_vcf["lam"], dtype=float)

        rows.append(
            {
                "model": key,
                "delta_theta_rel_mhs": abs(params_mhs["theta"] - params_psmcfa["theta"]) / max(abs(params_psmcfa["theta"]), 1e-12),
                "delta_theta_rel_vcf": abs(params_vcf["theta"] - params_psmcfa["theta"]) / max(abs(params_psmcfa["theta"]), 1e-12),
                "delta_rho_rel_mhs": abs(params_mhs["rho"] - params_psmcfa["rho"]) / max(abs(params_psmcfa["rho"]), 1e-12),
                "delta_rho_rel_vcf": abs(params_vcf["rho"] - params_psmcfa["rho"]) / max(abs(params_psmcfa["rho"]), 1e-12),
                "lam_l1_mhs_vs_psmcfa": float(np.mean(np.abs(lam1 - lam0))),
                "lam_l1_vcf_vs_psmcfa": float(np.mean(np.abs(lam2 - lam0))),
                "rmse_curve_log10_mhs_vs_psmcfa": rmse_mhs,
                "rmse_curve_log10_vcf_vs_psmcfa": rmse_vcf,
                "rmse_curve_log10_vcf_vs_mhs": rmse_vcf_mhs,
            }
        )

    tab = pd.DataFrame(rows).sort_values("model")
    save_table_multi(tab, "S5_format_consistency")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=220, constrained_layout=True)

    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    mhs_vals = [tab[tab.model == m]["rmse_curve_log10_mhs_vs_psmcfa"].iloc[0] for m in MODEL_ORDER]
    vcf_vals = [tab[tab.model == m]["rmse_curve_log10_vcf_vs_psmcfa"].iloc[0] for m in MODEL_ORDER]

    axes[0].bar(
        x - width / 2,
        mhs_vals,
        width=width,
        color=COLORS["mhs"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.9,
        label="MHS vs PSMCFA",
    )
    axes[0].bar(
        x + width / 2,
        vcf_vals,
        width=width,
        color=COLORS["vcf"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.9,
        label="VCF vs PSMCFA",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODEL_ORDER, rotation=15)
    axes[0].set_ylabel("Curve RMSE (log10 Ne)")
    axes[0].set_title("Curve-level consistency", fontsize=12.5, pad=8)
    stylize_axis(axes[0], xlog=False, yfmt=None)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    ymax = max(mhs_vals + vcf_vals) if (mhs_vals or vcf_vals) else 0.0
    axes[0].set_ylim(0.0, max(0.003, ymax * 1.18))
    axes[0].grid(axis="x", alpha=0.0)
    axes[0].legend(frameon=False)

    rel_theta_mhs = [tab[tab.model == m]["delta_theta_rel_mhs"].iloc[0] for m in MODEL_ORDER]
    rel_theta_vcf = [tab[tab.model == m]["delta_theta_rel_vcf"].iloc[0] for m in MODEL_ORDER]
    axes[1].plot(
        MODEL_ORDER,
        rel_theta_mhs,
        marker="o",
        ms=5.5,
        lw=2.2,
        color=COLORS["mhs"],
        label="theta rel diff (MHS)",
    )
    axes[1].plot(
        MODEL_ORDER,
        rel_theta_vcf,
        marker="o",
        ms=5.5,
        lw=2.2,
        color=COLORS["vcf"],
        label="theta rel diff (VCF)",
    )
    axes[1].set_ylabel("Relative difference")
    axes[1].set_title("Parameter-level consistency", fontsize=12.5, pad=8)
    stylize_axis(axes[1], xlog=False)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2e}"))
    axes[1].legend(frameon=False)

    panel_labels(axes)
    fig.suptitle("S5. Format consistency: psmcfa / mhs / vcf", fontsize=15.5, fontweight="bold", y=1.05)
    save_figure_multi(fig, "S5_format_consistency")
    plt.close(fig)

    return tab


def run_all(force: bool = False):
    ensure_pydeps()
    s1 = run_s1_smooth_ablation(force=force)
    s2 = run_s2_em_convergence(force=force)
    s3 = run_s3_memory_scaling(force=force)
    s4 = run_s4_thread_scaling(force=force)
    s5 = run_s5_format_consistency(force=force)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}


def parse_steps(s: str) -> List[str]:
    xs = [x.strip().upper() for x in s.split(",") if x.strip()]
    valid = {"S1", "S2", "S3", "S4", "S5"}
    for x in xs:
        if x not in valid:
            raise ValueError(f"invalid step: {x}")
    return xs


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Run supplementary experiments S1-S5")
    p.add_argument("--steps", default="S1,S2,S3,S4,S5", help="Comma-separated steps (S1..S5)")
    p.add_argument("--force", action="store_true", help="Force rerun even if outputs already exist")
    args = p.parse_args(argv)

    steps = parse_steps(args.steps)

    print("ROOT=", ROOT)
    print("RUN_DIR=", RUN_DIR)
    print("PSMC_RS_BIN=", PSMC_RS_BIN)
    print("C_PSMC_BIN=", C_PSMC_BIN)
    print("SIM_LENGTH_BP=", SIM_LENGTH_BP)
    print("S4_INPUT=", f"{S4_CONTIGS}x{S4_CONTIG_MB}Mb (seed_base={S4_SEED_BASE})")
    print("S5_FORMAT_VERSION=", S5_FORMAT_VERSION)
    print("PERF_REPEATS=", PERF_REPEATS)
    print("BOOTSTRAP_REPS=", BOOTSTRAP_REPS)
    print("steps=", steps)

    out = {}
    for s in steps:
        if s == "S1":
            out[s] = run_s1_smooth_ablation(force=args.force)
        elif s == "S2":
            out[s] = run_s2_em_convergence(force=args.force)
        elif s == "S3":
            out[s] = run_s3_memory_scaling(force=args.force)
        elif s == "S4":
            out[s] = run_s4_thread_scaling(force=args.force)
        elif s == "S5":
            out[s] = run_s5_format_consistency(force=args.force)

    print("done. figures:")
    for pth in sorted(FIG_DIR.glob("*")):
        print(" -", pth)
    print("tables:")
    for pth in sorted(TABLE_DIR.glob("*")):
        print(" -", pth)
    print("logs:", LOG_DIR / "commands.jsonl")
    return out


if __name__ == "__main__":
    main()
