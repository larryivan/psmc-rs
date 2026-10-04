#!/usr/bin/env python3
"""One-command experiment check + plotting entrypoint.

Checks that the packaged results are complete, then executes
experiment/notebooks/all_plots_precomputed.ipynb headlessly to redraw every
manuscript figure. Runs with the interpreter it is started with: the check
needs only the standard library, plotting needs numpy, pandas and matplotlib
(experiment/requirements.txt).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT_ENV = "PSMC_RS_ROOT"
PLOT_PACKAGES = ("numpy", "pandas", "matplotlib")


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "Cargo.toml").exists() and (p / "src").exists():
            return p
    raise RuntimeError(f"Cannot locate psmc-rs root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
EXP = ROOT / "experiment"
NB = EXP / "notebooks" / "all_plots_precomputed.ipynb"
MAIN = EXP / "runs" / "main_text"
REAL = EXP / "runs" / "real_data"
SUPP = EXP / "runs" / "supplementary"
FIT_SUMMARY = REAL / "tables" / "real_data_rust_vs_c_fit_summary.csv"
REAL_SAMPLES = ("HLemySub1", "HLhydTec1", "HLpelCas1", "HomoSapiens")

# What each group of inputs feeds, and the command that regenerates it.
GROUPS: Dict[str, Tuple[str, str]] = {
    "main_text": (
        "Figure 2 and Table 1",
        "python3 experiment/scripts/run_main_text.py --steps simulate,main",
    ),
    "main_perf": (
        "Figure 4 A/B and Table 2",
        "python3 experiment/scripts/run_main_text.py --steps perf",
    ),
    "real_data": ("Figure 3", "python3 experiment/scripts/run_real_data.py"),
    "supplementary": (
        "Figure 4 C/D and Supplementary Figures S1-S2",
        "python3 experiment/scripts/run_supplementary.py --steps S2,S3,S4,S5",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run experiment sanity checks and render all figures.")
    p.add_argument("--skip-check", action="store_true", help="Skip sanity checks.")
    p.add_argument("--skip-plot", action="store_true", help="Skip plotting.")
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Report missing results but still draw every figure whose inputs are present "
        "(exit status 1 if any figure was skipped).",
    )
    p.add_argument(
        "--notebook",
        default=str(NB),
        help="Notebook path (default: experiment/notebooks/all_plots_precomputed.ipynb).",
    )
    return p.parse_args()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def resolve_table_path(value: str) -> Path:
    """Paths in result tables are repository-relative (absolute ones are accepted)."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def missing_packages(names: Iterable[str]) -> List[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    return missing


def ensure_runtime_dependencies() -> None:
    missing = missing_packages(PLOT_PACKAGES)
    if missing:
        raise SystemExit(
            f"error: plotting needs Python packages that {sys.executable} cannot import: "
            f"{', '.join(missing)}\n"
            f"Install them into this interpreter:\n"
            f"    {sys.executable} -m pip install {' '.join(PLOT_PACKAGES)}\n"
            "or start run.py with an interpreter that has them, e.g.\n"
            "    /path/to/python experiment/run.py\n"
            "(--skip-plot runs the checks without these packages)."
        )


def required_files() -> List[Tuple[Path, str]]:
    files = []
    for model in ("constant", "bottleneck", "expansion", "zigzag"):
        files.append((MAIN / "outputs" / f"{model}.rust.main.json", "main_text"))
        files.append((MAIN / "outputs" / f"{model}.c.main.psmc", "main_text"))
    files.append((MAIN / "tables" / "table_2_runtime_memory.csv", "main_perf"))
    for name in (
        "S2_em_convergence_trace.csv",
        "S3_memory_scaling_summary.csv",
        "S4_thread_scaling_summary.csv",
        "S5_format_consistency.csv",
    ):
        files.append((SUPP / "tables" / name, "supplementary"))
    files.append((FIT_SUMMARY, "real_data"))
    if FIT_SUMMARY.exists():
        with FIT_SUMMARY.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for col in ("rust_json", "c_psmc"):
                    if row.get(col):
                        files.append((resolve_table_path(row[col]), "real_data"))
    return files


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


def missing_report(missing: List[Tuple[Path, str]]) -> str:
    lines = [f"Missing experiment results ({len(missing)} files):", ""]
    for group, (feeds, command) in GROUPS.items():
        paths = [p for p, g in missing if g == group]
        if not paths:
            continue
        lines.append(f"  {feeds}: regenerate with")
        lines.append(f"      {command}")
        lines += [f"    missing: {rel(p)}" for p in paths]
        lines.append("")
    lines += [
        "The outputs are small text files meant to be committed with the package; a",
        "fresh clone lacks them only if they were never committed. Regenerating needs",
        "the psmc-rs release build and C psmc (lh3/psmc built with -O1); simulation also",
        "needs stdpopsim and msprime. See experiment/README.md, section 'Pipeline'.",
        "Use --keep-going to draw the figures whose inputs are present.",
    ]
    return "\n".join(lines)


def run_checks(keep_going: bool) -> bool:
    """Return True if every check passed; raise (or warn with keep_going) otherwise."""
    problems = []
    missing = [(p, g) for p, g in required_files() if not p.exists()]
    if missing:
        problems.append(missing_report(missing))

    # Timing tables can exist and still be empty (NaN timings).
    timing_tables = (
        (MAIN / "tables" / "table_2_runtime_memory.csv", ["wall_sec_mean", "peak_rss_mb_mean"], GROUPS["main_perf"][1]),
        (
            SUPP / "tables" / "S3_memory_scaling_summary.csv",
            ["wall_sec_mean", "peak_rss_mb_mean"],
            "python3 experiment/scripts/run_supplementary.py --steps S3",
        ),
        (
            SUPP / "tables" / "S4_thread_scaling_summary.csv",
            ["wall_sec_mean"],
            "python3 experiment/scripts/run_supplementary.py --steps S4",
        ),
    )
    for table, columns, command in timing_tables:
        if table.exists() and not _csv_has_non_nan(table, columns):
            problems.append(
                f"{rel(table)} contains no finite runtime/memory values; regenerate with\n    {command}"
            )

    if FIT_SUMMARY.exists():
        with FIT_SUMMARY.open("r", encoding="utf-8", newline="") as f:
            got = {row.get("sample_id", "") for row in csv.DictReader(f)}
        absent = set(REAL_SAMPLES) - got
        if absent:
            problems.append(
                f"{rel(FIT_SUMMARY)} is missing sample_id: {', '.join(sorted(absent))}; regenerate with\n"
                f"    {GROUPS['real_data'][1]}"
            )

    if not problems:
        print("[check] required files and key tables are valid.")
        return True
    report = "\n\n".join(problems)
    if not keep_going:
        raise SystemExit(f"error: {report}")
    print(f"[check] warning: {report}\n", file=sys.stderr)
    return False


def execute_notebook(notebook_path: Path, keep_going: bool) -> List[str]:
    """Execute the code cells; return the cells that failed and were skipped (keep_going).

    With keep_going any error in a figure cell skips that figure: missing or
    incomplete inputs surface in many forms (FileNotFoundError, a sample
    missing from a table, an unparsable output), and the checks have already
    reported what is missing.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"notebook not found: {notebook_path}")

    os.environ[ROOT_ENV] = str(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "psmc-rs-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope: Dict = {}
    skipped = []
    first_code_cell = True
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        title = src.strip().splitlines()[0].lstrip("# ").strip()
        print(f"[plot] execute cell {i}: {title}")
        try:
            exec(compile(src, f"{notebook_path.name}:cell{i}", "exec"), scope, scope)
        except Exception as e:
            # The first cell defines the helpers every other cell needs.
            if not keep_going or first_code_cell:
                raise
            print(f"[plot] skipped cell {i} ({title}): {type(e).__name__}: {e}", file=sys.stderr)
            skipped.append(title)
        finally:
            first_code_cell = False
            plt.close("all")
    return skipped


def main() -> int:
    args = parse_args()
    nb = Path(args.notebook).resolve()
    if not args.skip_plot:
        ensure_runtime_dependencies()

    ok = True
    if not args.skip_check:
        ok = run_checks(args.keep_going)
    if not args.skip_plot:
        skipped = execute_notebook(nb, args.keep_going)
        if skipped:
            print("[plot] figures not drawn (see the errors above): " + "; ".join(skipped), file=sys.stderr)
            ok = False

    print("Done." if ok else "Done, with missing results (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
