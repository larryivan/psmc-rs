"""The analysis code on both platforms: E1, E2, E4 and E5's tables written again on the laptop.

Each experiment's report (python -m experiment.pipeline report EXP) reads
the committed fits, bootstrap files, truth and QC records and writes the
experiment's tables; the committed tables are those the production's
reports wrote on Ibex (x86-64, Linux). Here the same reports run on the
laptop (arm64, macOS) on a copy of the committed runs directory, its
tables removed first, and every table is compared cell by cell with the
committed one:
  * whether the files are byte-identical, and the cells that differ;
  * the largest relative difference of the numbers of magnitude 1e-3 or more;
  * the differences between psmc-rs and C psmc (the columns named *rs_vs_c* or
    *rs_default_vs_c*, of order 1e-5): their largest relative and absolute difference;
  * that the report job's identity on the laptop (its code digest and package
    versions, which key its provenance record) equals that of Ibex's record.
Writes reports_laptop_vs_ibex.csv (one row per table) and reports.json.

Run with: experiment/.venv/bin/python -B reports.py
"""
import csv
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # the repository root
sys.path.insert(0, str(REPO))
from experiment.pipeline import config, tools  # noqa: E402
from experiment.pipeline.experiments import Context  # noqa: E402

X = Path(__file__).resolve().parent.parent
IBX = REPO / "experiment/runs"
EXPS = ("E1", "E2", "E4", "E5")


class _Identities(Context):
    """A context that only builds the job graph: no tool is looked for (a report's identity names none)."""

    def _tool(self, key, identify):
        return tools.Tool(key, Path("/nonexistent") / key)


def report_tools(exp: str, runs: Path):
    """The report job's tools as this checkout and this machine give them (its code digest, package versions)."""
    mod = importlib.import_module(f"experiment.pipeline.experiments.{exp.lower()}")
    return next(j for j in mod.jobs(_Identities(config.get_profile("production"), runs)) if j.id == f"{exp}/report").tools


def num(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def rs_vs_c(col):
    return "rs_vs_c" in col or "rs_default_vs_c" in col


def compare(a: Path, b: Path):
    """Table a (laptop) against b (Ibex, the reference)."""
    ra, rb = list(csv.reader(open(a))), list(csv.reader(open(b)))
    row = {"table": b.name, "bytes_identical": a.read_bytes() == b.read_bytes(),
           "same_shape": len(ra) == len(rb) and all(len(x) == len(y) for x, y in zip(ra, rb)) and ra[:1] == rb[:1],
           "cells": 0, "numeric_cells": 0, "cells_differing": 0, "nonnumeric_differing": 0,
           "max_rel_ge_1e-3": 0.0, "max_abs_lt_1e-3": 0.0,
           "rs_vs_c_cells": 0, "rs_vs_c_max_rel": 0.0, "rs_vs_c_max_abs": 0.0, "rs_vs_c_median_magnitude": None}
    if not row["same_shape"]:
        return row
    head = rb[0]
    mags = []
    for x, y in zip(ra[1:], rb[1:]):
        for col, s, t in zip(head, x, y):
            row["cells"] += 1
            u, v = num(s), num(t)
            if u is None or v is None:
                if s != t:
                    row["cells_differing"] += 1
                    row["nonnumeric_differing"] += 1
                continue
            row["numeric_cells"] += 1
            if s != t:
                row["cells_differing"] += 1
            d = abs(u - v)
            if abs(v) >= 1e-3:
                row["max_rel_ge_1e-3"] = max(row["max_rel_ge_1e-3"], d / abs(v))
            else:
                row["max_abs_lt_1e-3"] = max(row["max_abs_lt_1e-3"], d)
            if rs_vs_c(col) and v != 0:
                row["rs_vs_c_cells"] += 1
                row["rs_vs_c_max_rel"] = max(row["rs_vs_c_max_rel"], d / abs(v))
                row["rs_vs_c_max_abs"] = max(row["rs_vs_c_max_abs"], d)
                mags.append(abs(v))
    if mags:
        mags.sort()
        row["rs_vs_c_median_magnitude"] = mags[len(mags) // 2]
    return row


def main():
    os.nice(19 - os.nice(0))
    rows, code = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        for exp in EXPS:
            shutil.copytree(IBX / exp, runs / exp, symlinks=True)
            shutil.rmtree(runs / exp / "tables")
        for exp in EXPS:
            res = subprocess.run([sys.executable, "-B", "-m", "experiment.pipeline", "report", exp,
                                  "--profile", "production", "--runs-dir", str(runs)],
                                 cwd=REPO, capture_output=True, text=True)
            if res.returncode != 0:
                sys.exit(f"report {exp} failed:\n{res.stdout}\n{res.stderr}")
            ibx = json.loads((IBX / exp / "provenance/report.json").read_text())
            lap = report_tools(exp, runs)
            code[exp] = {"laptop": lap, "ibex": ibx["key"]["tools"], "identical": lap == ibx["key"]["tools"],
                         "ibex_host": ibx["run"].get("host"), "ibex_slurm_job_id": ibx["run"].get("slurm_job_id")}
            names = sorted(p.name for p in (IBX / exp / "tables").glob("*.csv"))
            assert names == sorted(p.name for p in (runs / exp / "tables").glob("*.csv")), exp
            for name in names:
                rows.append({"exp": exp, **compare(runs / exp / "tables" / name, IBX / exp / "tables" / name)})
    with open(X / "reports_laptop_vs_ibex.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rc = [r for r in rows if r["rs_vs_c_cells"]]
    summary = {
        "experiments": list(EXPS),
        "tables": len(rows),
        "tables_bytes_identical": sum(r["bytes_identical"] for r in rows),
        "tables_same_shape": sum(r["same_shape"] for r in rows),
        "cells": sum(r["cells"] for r in rows),
        "numeric_cells": sum(r["numeric_cells"] for r in rows),
        "cells_differing": sum(r["cells_differing"] for r in rows),
        "nonnumeric_differing": sum(r["nonnumeric_differing"] for r in rows),
        "max_rel_ge_1e-3": max(r["max_rel_ge_1e-3"] for r in rows),
        "max_abs_lt_1e-3": max(r["max_abs_lt_1e-3"] for r in rows),
        "rs_vs_c_cells": sum(r["rs_vs_c_cells"] for r in rc),
        "rs_vs_c_max_rel": max(r["rs_vs_c_max_rel"] for r in rc),
        "rs_vs_c_max_abs": max(r["rs_vs_c_max_abs"] for r in rc),
        "rs_vs_c_tables": [f"{r['exp']}/{r['table']}" for r in rc],
        "code": code,
        "code_identical": all(c["identical"] for c in code.values()),
    }
    (X / "reports.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "code"}, indent=1))


if __name__ == "__main__":
    main()
