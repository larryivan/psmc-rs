#!/usr/bin/env python3
"""Regenerate the empirical results (Figure 3): psmc-rs and C psmc on experiment/real_data.

For every sample this writes ``outputs/{sample}.rust.json`` and
``outputs/{sample}.c.psmc`` under ``--out-dir`` (default
``experiment/runs/real_data``) and then
``tables/real_data_rust_vs_c_fit_summary.csv``, whose ``rust_json`` and
``c_psmc`` columns are repository-relative so the table works in any clone.
Figure 3 is drawn from these files by ``experiment/run.py``.

Both tools use the matched settings of ``exp_common.MatchedSettings``
(``-N20 -t15 -r4 -p 4+25*2+4+6``, passed to psmc-rs as the corresponding
options, with no smoothing and one psmc-rs thread); ``mu`` and the
generation time only scale the curves.

This is the run logic of ``experiment/notebooks/real_data_rust_vs_c.ipynb``
(deleted in commit 79ac364), updated to the four samples of the current figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common  # noqa: E402

REAL_DATA = exp_common.EXPERIMENT / "real_data"
SUMMARY_NAME = "real_data_rust_vs_c_fit_summary.csv"
SUMMARY_FIELDS = ["sample_id", "species", "mu", "gen_years", "rmse_log10_ne_rs_vs_c", "rust_json", "c_psmc"]


@dataclass(frozen=True)
class Sample:
    sid: str
    file: str
    species: str
    mu: float
    gen_years: float


SAMPLES: List[Sample] = [
    Sample("HLemySub1", "HLemySub1_diploid.psmcfa", "Emydura subglobosa (HLemySub1)", 7.9e-9, 15.0),
    Sample("HLhydTec1", "HLhydTec1_diploid.psmcfa", "Hydromedusa tectifera (HLhydTec1)", 7.9e-9, 15.0),
    Sample("HLpelCas1", "HLpelCas1_diploid.psmcfa", "Pelusios castaneus (HLpelCas1)", 7.9e-9, 15.0),
    # The packaged human psmcfa holds chromosome 1 only (one sequence, 2,487,383
    # bins of 100 bp), not a whole genome.
    Sample(
        "HomoSapiens",
        "Papuan_highlands_diploid.psmcfa",
        "Homo sapiens (Papuan Highlands, chr1)",
        1.25e-8,
        25.0,
    ),
]


def rust_json_path(out_dir: Path, s: Sample) -> Path:
    return out_dir / "outputs" / f"{s.sid}.rust.json"


def c_psmc_path(out_dir: Path, s: Sample) -> Path:
    return out_dir / "outputs" / f"{s.sid}.c.psmc"


def run_samples(
    samples: List[Sample],
    out_dir: Path,
    psmc_rs: Optional[str],
    c_psmc: Optional[str],
    force: bool,
    input_dir: Path = REAL_DATA,
) -> None:
    settings = exp_common.MatchedSettings.from_env()
    print(f"settings={settings}")
    log = out_dir / "logs" / "commands.jsonl"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    rust_bin = exp_common.resolve_psmc_rs(psmc_rs)
    c_bin = exp_common.resolve_c_psmc(c_psmc, required=True)
    print(f"[tool] psmc-rs: {rust_bin}\n[tool] C psmc: {c_bin}")
    settings.check_psmc_rs(rust_bin)
    for s in samples:
        inp = input_dir / s.file
        if not inp.exists():
            raise FileNotFoundError(f"input missing: {inp}")
        r_out, c_out = rust_json_path(out_dir, s), c_psmc_path(out_dir, s)
        # Reuse only finished runs: an interrupted C psmc leaves fewer rounds.
        if force or not exp_common.psmc_rs_output_ok(r_out):
            print(f"[run] psmc-rs {s.sid}", flush=True)
            exp_common.run_command(settings.rust_cmd(rust_bin, inp, r_out, mu=s.mu), log_path=log, cwd=exp_common.ROOT)
        if force or not exp_common.c_psmc_complete(c_out, settings.n_iter):
            if c_out.exists() and not force:
                print(f"[run] {c_out.name} is not a finished {settings.n_iter}-iteration run; re-running")
            print(f"[run] C psmc {s.sid}", flush=True)
            exp_common.run_staged(
                lambda tmp: settings.c_cmd(c_bin, inp, tmp), c_out, log_path=log, cwd=exp_common.ROOT
            )


def read_fit_summary(path: Path) -> Dict[str, Dict[str, str]]:
    """Rows of an existing fit summary by sample id (empty if there is none)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f) if r.get("sample_id")}


def write_fit_summary(out_dir: Path, root: Path = exp_common.ROOT, samples: List[Sample] = SAMPLES) -> Path:
    """Update the fit summary from the existing outputs of ``samples``; return its path.

    The row of every sample in ``samples`` with both outputs is recomputed.
    All other rows of an existing summary are kept unchanged, so running a
    subset of samples (or a clone that lacks some outputs) does not drop the
    others; experiment/run.py names the outputs that a kept row points to but
    that are missing. The RMSE is that of the plotting notebook: log10 Ne of
    psmc-rs vs C on 500 log-spaced points between 1e3 and 1e8 years.
    """
    path = out_dir / "tables" / SUMMARY_NAME
    rows = read_fit_summary(path)
    updated = []
    missing = []
    for s in samples:
        r_out, c_out = rust_json_path(out_dir, s), c_psmc_path(out_dir, s)
        absent = [x for x in (r_out, c_out) if not x.exists()]
        if absent:
            missing += absent
            continue
        rust = exp_common.curve_from_json(r_out, s.mu, s.gen_years)
        c = exp_common.curve_from_c_psmc(c_out, s.mu, s.gen_years)
        rows[s.sid] = {
            "sample_id": s.sid,
            "species": s.species,
            "mu": s.mu,
            "gen_years": s.gen_years,
            "rmse_log10_ne_rs_vs_c": exp_common.rmse_log10(c, rust),
            "rust_json": exp_common.repo_relative(r_out, root),
            "c_psmc": exp_common.repo_relative(c_out, root),
        }
        updated.append(s.sid)
    if missing:
        print(
            "[warn] outputs missing; these samples keep their previous row, if any:\n"
            + "\n".join(f"  {exp_common.repo_relative(x, root)}" for x in missing)
        )
    if not updated:
        raise SystemExit("error: no sample has both outputs; run without --summary-only first")
    kept = sorted(set(rows) - set(updated))
    if kept:
        print(f"[info] rows kept from the previous summary: {', '.join(kept)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows[sid] for sid in sorted(rows))
    print(f"saved table: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run psmc-rs and C psmc on the packaged real-data psmcfa files.",
        epilog="EM settings (N_ITER, T_MAX, PATTERN, RHO_T_RATIO, ...) are read from environment "
        "variables; see experiment/README.md.",
    )
    p.add_argument(
        "--out-dir",
        default=str(exp_common.RUNS / "real_data"),
        help="Output root (outputs/, tables/, logs/). Default: experiment/runs/real_data",
    )
    p.add_argument(
        "--samples",
        default=",".join(s.sid for s in SAMPLES),
        help="Comma-separated sample ids (default: all four)",
    )
    p.add_argument("--psmc-rs", help="psmc-rs binary (default: $PSMC_RS_BIN or target/release/psmc-rs)")
    p.add_argument("--c-psmc", help="C psmc binary (default: $C_PSMC_BIN or `psmc` on PATH)")
    p.add_argument(
        "--input-dir",
        default=str(REAL_DATA),
        help="Directory holding the sample psmcfa files (default: experiment/real_data)",
    )
    p.add_argument("--force", action="store_true", help="Re-run outputs that already exist")
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Only update the fit summary from existing outputs (of all samples; rows of "
        "samples without outputs are kept)",
    )
    args = p.parse_args(argv)

    by_id = {s.sid: s for s in SAMPLES}
    wanted = [x.strip() for x in args.samples.split(",") if x.strip()]
    unknown = [x for x in wanted if x not in by_id]
    if unknown:
        raise SystemExit(f"error: unknown sample(s): {', '.join(unknown)} (valid: {', '.join(by_id)})")
    samples = [by_id[x] for x in wanted]
    out_dir = Path(args.out_dir).resolve()

    if not args.summary_only:
        run_samples(samples, out_dir, args.psmc_rs, args.c_psmc, args.force, Path(args.input_dir).resolve())
    write_fit_summary(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
