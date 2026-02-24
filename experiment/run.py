#!/usr/bin/env python3
"""One-command experiment check + plotting entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Cargo.toml").exists() and (p / "src").exists():
            return p
    raise RuntimeError(f"Cannot locate psmc-rs root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
EXP = ROOT / "experiment"
NB = EXP / "notebooks" / "all_plots_precomputed.ipynb"
FALLBACK_PY = Path("/Users/larryivanhan/miniforge3/bin/python")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiment sanity checks and render all figures.")
    p.add_argument("--skip-check", action="store_true", help="Skip sanity checks.")
    p.add_argument("--skip-plot", action="store_true", help="Skip plotting.")
    p.add_argument(
        "--notebook",
        default=str(NB),
        help="Notebook path (default: experiment/notebooks/all_plots_precomputed.ipynb).",
    )
    return p.parse_args()


def ensure_runtime_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        return
    except Exception:
        pass

    exe = Path(sys.executable).resolve()
    if FALLBACK_PY.exists() and exe != FALLBACK_PY.resolve():
        os.execv(str(FALLBACK_PY), [str(FALLBACK_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

    raise RuntimeError(
        "Missing plotting dependencies (matplotlib/numpy/pandas). "
        "Install them or run with /Users/larryivanhan/miniforge3/bin/python."
    )


def required_files() -> List[Path]:
    return [
        # Main text outputs
        EXP / "runs" / "main_text" / "outputs" / "constant.rust.main.json",
        EXP / "runs" / "main_text" / "outputs" / "constant.c.main.psmc",
        EXP / "runs" / "main_text" / "outputs" / "bottleneck.rust.main.json",
        EXP / "runs" / "main_text" / "outputs" / "bottleneck.c.main.psmc",
        EXP / "runs" / "main_text" / "outputs" / "expansion.rust.main.json",
        EXP / "runs" / "main_text" / "outputs" / "expansion.c.main.psmc",
        EXP / "runs" / "main_text" / "outputs" / "zigzag.rust.main.json",
        EXP / "runs" / "main_text" / "outputs" / "zigzag.c.main.psmc",
        # Supplementary tables
        EXP / "runs" / "supplementary" / "tables" / "S2_em_convergence_trace.csv",
        EXP / "runs" / "supplementary" / "tables" / "S3_memory_scaling_summary.csv",
        EXP / "runs" / "supplementary" / "tables" / "S4_thread_scaling_summary.csv",
        EXP / "runs" / "supplementary" / "tables" / "S5_format_consistency.csv",
        # Real-data fit summary
        EXP / "runs" / "real_data" / "tables" / "real_data_rust_vs_c_fit_summary.csv",
    ]


def _as_float(v: str) -> float:
    if v is None:
        return float("nan")
    s = str(v).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _csv_has_non_nan(path: Path, columns: Iterable[str]) -> bool:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        found = False
        for r in rows:
            found = True
            vals = [_as_float(r.get(c, "")) for c in columns]
            if any(math.isfinite(x) for x in vals):
                return True
        return not found


def run_checks() -> None:
    missing = [p for p in required_files() if not p.exists()]
    if missing:
        msg = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required experiment files:\n{msg}")

    t2 = EXP / "runs" / "main_text" / "tables" / "table_2_runtime_memory.csv"
    if t2.exists():
        ok = _csv_has_non_nan(t2, ["wall_sec_mean", "peak_rss_mb_mean"])
        if not ok:
            raise ValueError(f"{t2} contains no finite runtime/memory values.")

    fit = EXP / "runs" / "real_data" / "tables" / "real_data_rust_vs_c_fit_summary.csv"
    expected = {"HLemySub1", "HLhydTec1", "HLpelCas1", "HomoSapiens"}
    with fit.open("r", encoding="utf-8", newline="") as f:
        got = {row.get("sample_id", "") for row in csv.DictReader(f)}
    if expected - got:
        miss = ", ".join(sorted(expected - got))
        raise ValueError(f"real_data fit summary missing sample_id: {miss}")

    print("[check] required files and key tables are valid.")


def execute_notebook(notebook_path: Path) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"notebook not found: {notebook_path}")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    import matplotlib

    matplotlib.use("Agg")

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {}
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        print(f"[plot] execute cell {i}")
        exec(compile(src, f"{notebook_path.name}:cell{i}", "exec"), scope, scope)


def main() -> int:
    args = parse_args()
    nb = Path(args.notebook).resolve()
    ensure_runtime_dependencies()

    if not args.skip_check:
        run_checks()
    if not args.skip_plot:
        execute_notebook(nb)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
