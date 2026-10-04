#!/usr/bin/env python3
"""Regenerate the simulated main-text results (Figure 2, Table 1, Figure 4 A/B, Table 2).

Steps (``--steps``, default ``simulate,main,perf``):

simulate  ``inputs/{model}.psmcfa``: 500 Mb diploid genome per model, simulated
          with ``simulate_msprime_to_psmcfa.py`` (stdpopsim, msprime engine).
main      ``outputs/{model}.rust.main.json`` and ``outputs/{model}.c.main.psmc``:
          psmc-rs and C psmc with matched settings (Figure 2 and Table 1).
perf      ``PERF_REPEATS`` single-thread runs per model and tool under ``perf/``,
          summarised in ``tables/perf_raw_repeats.csv`` and
          ``tables/table_2_runtime_memory.csv`` (Figure 4 A/B and Table 2).
          Only the rows of the timed models are replaced.

The figures and Table 1 are drawn from these files by ``experiment/run.py``.

This is the run logic of ``experiment/notebooks/main_text_6items.ipynb``
(deleted in commit 79ac364) as a script. Paths are resolved from this file;
run parameters come from environment variables (``N_ITER``, ``SIM_LENGTH_BP``,
``PERF_REPEATS``, ... see ``experiment/README.md``).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common  # noqa: E402
from exp_common import MODEL_ORDER, MODELS  # noqa: E402

MU = float(os.environ.get("MU", "2.5e-8"))
SIM_LENGTH_BP = int(os.environ.get("SIM_LENGTH_BP", "500000000"))
SIM_WINDOW_BP = int(os.environ.get("SIM_WINDOW_BP", "100"))
SIM_MUTATION = float(os.environ.get("SIM_MUTATION", str(MU)))
SIM_RECOMB = os.environ.get("SIM_RECOMB", "").strip() or None
PERF_REPEATS = int(os.environ.get("PERF_REPEATS", "10"))
ALPHA_CACHE_MB = int(os.environ.get("PSMC_ALPHA_CACHE_MB", "2048"))

STEPS = ("simulate", "main", "perf")


class Runner:
    def __init__(self, out_dir: Path, psmc_rs: Optional[str], c_psmc: Optional[str], sim_script: Path):
        self.out_dir = out_dir.resolve()
        self.inputs = self.out_dir / "inputs"
        self.outputs = self.out_dir / "outputs"
        self.perf = self.out_dir / "perf"
        self.tables = self.out_dir / "tables"
        self.log = self.out_dir / "logs" / "commands.jsonl"
        for d in (self.inputs, self.outputs, self.perf, self.tables, self.log.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.sim_script = sim_script
        self._psmc_rs_arg = psmc_rs
        self._c_psmc_arg = c_psmc
        self._psmc_rs: Optional[Path] = None
        self._c_psmc: Optional[Path] = None
        self.settings = exp_common.MatchedSettings.from_env()

    @property
    def psmc_rs(self) -> Path:
        if self._psmc_rs is None:
            self._psmc_rs = exp_common.resolve_psmc_rs(self._psmc_rs_arg)
            print(f"[tool] psmc-rs: {self._psmc_rs}")
        return self._psmc_rs

    @property
    def c_psmc(self) -> Path:
        if self._c_psmc is None:
            self._c_psmc = exp_common.resolve_c_psmc(self._c_psmc_arg, required=True)
            print(f"[tool] C psmc: {self._c_psmc}")
        return self._c_psmc

    def run(self, cmd: List[str], env: Optional[dict] = None) -> Dict:
        return exp_common.run_command(cmd, log_path=self.log, cwd=exp_common.ROOT, env=env)

    def input_path(self, model: str) -> Path:
        return self.inputs / f"{model}.psmcfa"

    def simulate(self, models: List[str], force: bool) -> None:
        exp_common.require_modules(["stdpopsim", "msprime"], "the simulate step")
        for key in models:
            spec = MODELS[key]
            out_path = self.input_path(key)
            if out_path.exists() and not force:
                print(f"[simulate] {key}: exists, skipped ({exp_common.repo_relative(out_path)})")
                continue
            cmd = [
                sys.executable,
                str(self.sim_script),
                "--model",
                spec["sim_model"],
                "--out",
                str(out_path),
                "--length",
                str(SIM_LENGTH_BP),
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
            print(f"[simulate] {key} length={SIM_LENGTH_BP:,}", flush=True)
            self.run(cmd)

    def check_matched(self) -> None:
        """Exit before any comparison run unless both tools start from the same theta/rho ratio."""
        self.settings.check_psmc_rs(self.psmc_rs)

    def rust(self, inp: Path, out_json: Path) -> Dict:
        self.check_matched()
        env = os.environ.copy()
        env["PSMC_ALPHA_CACHE_MB"] = str(ALPHA_CACHE_MB)
        return self.run(self.settings.rust_cmd(self.psmc_rs, inp, out_json, mu=MU), env=env)

    def c(self, inp: Path, out_psmc: Path) -> Dict:
        self.check_matched()
        # C psmc writes each round as it goes; stage it so that an interrupted
        # run never leaves a partial file at out_psmc.
        return exp_common.run_staged(
            lambda tmp: self.settings.c_cmd(self.c_psmc, inp, tmp), out_psmc, log_path=self.log, cwd=exp_common.ROOT
        )

    def require_inputs(self, models: List[str]) -> None:
        missing = [self.input_path(m) for m in models if not self.input_path(m).exists()]
        if missing:
            raise SystemExit(
                "error: simulated inputs missing:\n"
                + "\n".join(f"  {exp_common.repo_relative(p)}" for p in missing)
                + "\nRun the simulate step first (--steps simulate)."
            )

    def main_runs(self, models: List[str], force: bool) -> None:
        self.require_inputs(models)
        for key in models:
            inp = self.input_path(key)
            rust_out = self.outputs / f"{key}.rust.main.json"
            c_out = self.outputs / f"{key}.c.main.psmc"
            # Reuse only finished runs: an interrupted C psmc leaves fewer rounds.
            if force or not exp_common.psmc_rs_output_ok(rust_out):
                print(f"[main] psmc-rs {key}", flush=True)
                self.rust(inp, rust_out)
            if force or not exp_common.c_psmc_complete(c_out, self.settings.n_iter):
                if c_out.exists() and not force:
                    print(f"[main] {c_out.name} is not a finished {self.settings.n_iter}-iteration run; re-running")
                print(f"[main] C psmc {key}", flush=True)
                self.c(inp, c_out)

    def perf_runs(self, models: List[str]) -> None:
        # Timings cannot be cached, so every requested repeat is run again.
        exp_common.require_modules(["psutil"], "the perf step (peak RSS sampling)")
        self.require_inputs(models)
        rows = []
        for rep in range(1, PERF_REPEATS + 1):
            for key in models:
                inp = self.input_path(key)
                print(f"[perf] rep {rep}/{PERF_REPEATS} {key}", flush=True)
                r = self.rust(inp, self.perf / f"{key}.rust.rep{rep:02d}.json")
                rows.append({"model": key, "tool": "rust", "rep": rep, "wall_sec": r["wall_sec"], "peak_rss_mb": r["peak_rss_mb"]})
                c = self.c(inp, self.perf / f"{key}.c.rep{rep:02d}.psmc")
                rows.append({"model": key, "tool": "c", "rep": rep, "wall_sec": c["wall_sec"], "peak_rss_mb": c["peak_rss_mb"]})

        # Replace only the timed models: a run with --models <subset> must not
        # drop the other models' rows from Table 2.
        raw_path = self.tables / "perf_raw_repeats.csv"
        sum_path = self.tables / "table_2_runtime_memory.csv"
        kept = sorted({r["model"] for r in read_csv_rows(sum_path)} - set(models))
        write_csv(raw_path, RAW_FIELDS, merge_by_model(read_csv_rows(raw_path), rows, models))
        write_csv(sum_path, SUMMARY_FIELDS, merge_by_model(read_csv_rows(sum_path), summarise_perf(rows), models))
        if kept:
            print(f"[perf] kept the earlier timings of: {', '.join(kept)}")
        print(f"saved table: {raw_path}, {sum_path}")


RAW_FIELDS = ["model", "tool", "rep", "wall_sec", "peak_rss_mb"]
SUMMARY_FIELDS = ["model", "tool", "wall_sec_mean", "wall_sec_std", "peak_rss_mb_mean", "peak_rss_mb_std"]


def read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge_by_model(existing: List[Dict], new: List[Dict], models: List[str]) -> List[Dict]:
    """``new`` plus the ``existing`` rows of models not in ``models``, in figure order (c before rust)."""
    replaced = set(models)
    merged = [r for r in existing if r.get("model") not in replaced] + list(new)

    def key(r: Dict):
        m = r["model"]
        return (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m, r["tool"], int(r.get("rep") or 0))

    return sorted(merged, key=key)


def summarise_perf(rows: List[Dict]) -> List[Dict]:
    """Mean and sample SD (ddof=1, NaN for one repeat) per model and tool, in figure order."""

    def mean_sd(vals: List[float]):
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return float("nan"), float("nan")
        return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else float("nan")

    out = []
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    for model in models:
        for tool in ("c", "rust"):
            sel = [r for r in rows if r["model"] == model and r["tool"] == tool]
            if not sel:
                continue
            w_mean, w_sd = mean_sd([float(r["wall_sec"]) for r in sel])
            m_mean, m_sd = mean_sd([float(r["peak_rss_mb"]) for r in sel])
            out.append(
                {
                    "model": model,
                    "tool": tool,
                    "wall_sec_mean": w_mean,
                    "wall_sec_std": w_sd,
                    "peak_rss_mb_mean": m_mean,
                    "peak_rss_mb_std": m_sd,
                }
            )
    return out


def write_csv(path: Path, fields: List[str], rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_list(value: str, valid, what: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in valid]
    if bad:
        raise SystemExit(f"error: unknown {what}: {', '.join(bad)} (valid: {', '.join(valid)})")
    return items


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Simulate the four demographic models and run psmc-rs and C psmc on them.",
        epilog="Run parameters (N_ITER, SIM_LENGTH_BP, PERF_REPEATS, RHO_T_RATIO, ...) are read "
        "from environment variables; see experiment/README.md.",
    )
    p.add_argument("--steps", default=",".join(STEPS), help="Comma-separated subset of: simulate,main,perf")
    p.add_argument("--models", default=",".join(MODEL_ORDER), help="Comma-separated subset of the models")
    p.add_argument(
        "--out-dir",
        default=str(exp_common.RUNS / "main_text"),
        help="Output root (inputs/, outputs/, perf/, tables/, logs/). Default: experiment/runs/main_text",
    )
    p.add_argument("--psmc-rs", help="psmc-rs binary (default: $PSMC_RS_BIN or target/release/psmc-rs)")
    p.add_argument("--c-psmc", help="C psmc binary (default: $C_PSMC_BIN or `psmc` on PATH)")
    p.add_argument(
        "--sim-script",
        help="Simulation script (default: $SIM_SCRIPT or experiment/scripts/simulate_msprime_to_psmcfa.py)",
    )
    p.add_argument("--force-sim", action="store_true", help="Re-simulate inputs that already exist")
    p.add_argument("--force-run", action="store_true", help="Re-run main outputs that already exist")
    args = p.parse_args(argv)

    steps = parse_list(args.steps, STEPS, "step")
    models = parse_list(args.models, MODEL_ORDER, "model")
    runner = Runner(Path(args.out_dir), args.psmc_rs, args.c_psmc, exp_common.resolve_sim_script(args.sim_script))
    print(f"out_dir={runner.out_dir}")
    print(f"settings={runner.settings}")

    if "simulate" in steps:
        runner.simulate(models, force=args.force_sim)
    if "main" in steps:
        runner.main_runs(models, force=args.force_run)
    if "perf" in steps:
        runner.perf_runs(models)
    print(f"log: {runner.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
