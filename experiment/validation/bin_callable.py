"""Validation of ``--bin-callable`` on simulated genomes with callable masks.

Does fitting the callable bases of each bin (``--bin-callable auto``)
remove the bias that missing data scattered in short runs cause, which
treating every called bin as full (``--bin-callable ignore``, fq2psmcfa and
C psmc's model) leaves? For each replicate this script

1. simulates one diploid genome of a demographic model of the experiment
   pipeline with msprime (:mod:`experiment.pipeline.simulate`; the
   bottleneck model and the seeds of experiment ``BC`` by default,
   ``--model`` and ``--experiment`` choose others);
2. draws callable masks (:mod:`experiment.pipeline.inputprep`): none,
   dispersed 10% and 30% (mean uncallable run 20 bp) and clustered 30%
   (mean run 20 kb), and removes the VCF records at uncallable sites, as a
   caller would not call them;
3. fits psmc-rs to each masked VCF with its mask (mask mode) with
   ``--bin-callable auto`` and ``ignore``, and to the multihetsep of each
   mask (``psmc-rs convert --multihetsep``) with ``--bin-callable auto``
   (which fits multihetsep with C psmc's model, every called bin full) and
   ``approximate`` (which fits its approximate counts);
4. compares every fit with the truth (RMSE of log10 Ne on the model's
   informative window, :mod:`experiment.pipeline.metrics`) and with the fit
   of the same genome without missing data (θ ratio, RMSE between curves).

It is not part of CI: it takes a few minutes per replicate. Rerun it from
the repository root with the experiment's environment, after ``cargo build
--release``::

    experiment/.venv/bin/python -m experiment.validation.bin_callable \\
        --psmc-rs target/release/psmc-rs --out /tmp/bin_callable

It writes ``results.tsv`` (one row per replicate, mask, route and mode) and
``summary.tsv`` (means over replicates, with the range of θ over them) to
``--out`` and prints the summary as a Markdown table.
``experiment/validation/README.md`` has the numbers of the runs reported
with the change.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from experiment.pipeline import config, demography, inputprep, metrics, results, simulate

EXPERIMENT = "BC"  # the default seed namespace of this validation (config.seed)
MODEL = config.BOTTLENECK  # the default demographic model
SIMULATION = config.SIMULATION
INFERENCE = config.INFERENCE  # -N25 -t15 -r5 -p 4+25*2+4+6, bins of 100
SCALING = config.SIMULATION_SCALING
MASKS: Sequence[Optional[config.MaskSpec]] = (
    None,
    config.MaskSpec("dispersed10", 0.1, "dispersed", 20.0),
    config.MaskSpec("dispersed30", 0.3, "dispersed", 20.0),
    config.MaskSpec("clustered30", 0.3, "clustered", 20_000.0),
)
COMPLETE = "none"  # the mask without missing data
VCF_MODES = ("auto", "ignore")  # the VCF with its mask: exact counts, or every called bin full
# Multihetsep: auto fits C psmc's model (it uses exact counts only), and
# approximate the counts that follow the reader's placement of uncalled sites.
MHS_MODES = ("auto", "approximate")


def mask_name(spec: Optional[config.MaskSpec]) -> str:
    return COMPLETE if spec is None else spec.name


@dataclass
class FitRow:
    replicate: int
    mask: str
    route: str  # "vcf" (exact counts) or "multihetsep" (approximate counts)
    mode: str  # --bin-callable
    json_path: Path
    seconds: float


def run(cmd: List[str]) -> float:
    """Run ``cmd``; returns its wall time in seconds."""
    start = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stderr}")
    return time.monotonic() - start


def fit_args(threads: int) -> List[str]:
    """The inference settings of every fit (the psmc README's human recipe)."""
    return [
        "--pattern", INFERENCE.pattern,
        "--t-max", str(INFERENCE.t_max),
        "--tr-ratio", str(INFERENCE.tr_ratio),
        "--bin-size", str(INFERENCE.bin_size),
        "--mu", str(SCALING.mu),
        "--gen-years", str(SCALING.gen_years),
        "--threads", str(threads),
        "--no-progress",
        "--quiet",
    ]


def simulate_replicate(
    rep: int, genome: config.Genome, out: Path, model: config.DemographicModel, experiment: str
) -> inputprep.Vcf:
    """Simulate one genome, write its VCF (all records) and return it."""
    vcf_path = out / "genome.vcf"
    chroms = []
    for name in genome.names():
        seeds = simulate.seeds_for(experiment, model.name, rep, name)
        chroms.append(simulate.simulate_chromosome(model, SIMULATION, name, genome.chromosome_length, seeds))
        print(f"[simulate] rep{rep} {name}: {len(chroms[-1].het_positions)} heterozygous sites", flush=True)
    with open(vcf_path, "w", encoding="ascii", newline="\n") as f:
        f.write(simulate.vcf_header(chroms, "psmc-rs bin-callable validation"))
        for c in chroms:
            f.writelines(simulate.vcf_lines(c))
    return inputprep.read_vcf(vcf_path)


def replicate(
    rep: int,
    genome: config.Genome,
    psmc_rs: Path,
    out: Path,
    threads: int,
    n_iter: int,
    model: config.DemographicModel = MODEL,
    experiment: str = EXPERIMENT,
) -> List[FitRow]:
    """Simulate, mask and fit replicate ``rep``; returns its fits."""
    rep_dir = out / f"rep{rep:02d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    vcf = simulate_replicate(rep, genome, rep_dir, model, experiment)
    chroms = vcf.contigs()
    rows: List[FitRow] = []
    common = fit_args(threads)
    for spec in MASKS:
        name = mask_name(spec)
        mask = inputprep.callable_mask(spec, chroms, model.name, rep, experiment)
        bed = rep_dir / f"{name}.bed"
        inputprep.write_bed(bed, mask)
        masked, _ = inputprep.mask_vcf(vcf, mask, name)
        masked_vcf = rep_dir / f"{name}.vcf"
        inputprep.write_vcf(masked_vcf, masked)
        summary = inputprep.mask_summary(mask, dict(chroms))["totals"]
        print(f"[mask] rep{rep} {name}: missing fraction {summary['missing_fraction']:.4f}", flush=True)
        for mode in VCF_MODES:
            out_json = rep_dir / f"{name}.{mode}.json"
            secs = run(
                [str(psmc_rs), str(masked_vcf), str(out_json), str(n_iter), "--vcf-mode", "mask",
                 "--vcf-mask", str(bed), "--bin-callable", mode, *common]
            )
            rows.append(FitRow(rep, name, "vcf", mode, out_json, secs))
            print(f"[fit] rep{rep} {name} {mode}: {secs:.1f} s", flush=True)
        if spec is None:
            continue  # without missing data multihetsep bins exactly
        mhs = rep_dir / f"{name}.multihetsep"
        run([str(psmc_rs), "convert", str(masked_vcf), "--vcf-mode", "mask", "--vcf-mask", str(bed),
             "--multihetsep", str(mhs), "--quiet"])
        for mode in MHS_MODES:
            out_json = rep_dir / f"{name}.mhs.{mode}.json"
            secs = run([str(psmc_rs), str(mhs), str(out_json), str(n_iter), "--bin-callable", mode, *common])
            rows.append(FitRow(rep, name, "multihetsep", mode, out_json, secs))
            print(f"[fit] rep{rep} {name} multihetsep {mode}: {secs:.1f} s", flush=True)
    return rows


def evaluate(rows: List[FitRow], model: config.DemographicModel = MODEL) -> List[Dict]:
    """Every fit against the truth and the complete-data fit of its replicate."""
    epochs = demography.epochs(model)
    years = metrics.model_grid(epochs, config.TimeGrid(), SCALING.gen_years)
    params = {id(r): results.read_rs_json(r.json_path) for r in rows}
    fits = {id(r): results.fit_from_rs(params[id(r)]) for r in rows}

    def curve(r: FitRow):
        return fits[id(r)].curve(SCALING.mu, SCALING.gen_years, INFERENCE.bin_size)

    out = []
    for r in rows:
        complete = next(c for c in rows if c.replicate == r.replicate and c.mask == COMPLETE
                        and c.route == "vcf" and c.mode == "auto")
        p = params[id(r)]
        bc = p.get("bin_callable", {})
        callable_ = bc.get("callable") or {}
        out.append({
            "replicate": r.replicate,
            "mask": r.mask,
            "route": r.route,
            "mode": r.mode,
            "model": bc.get("model"),
            "mean_callable": callable_.get("mean_fraction"),
            "theta": fits[id(r)].theta,
            "theta_vs_complete": fits[id(r)].theta / fits[id(complete)].theta,
            "rmse_vs_truth": metrics.rmse_vs_truth(curve(r), epochs, years, SCALING.gen_years),
            "rmse_vs_complete": metrics.rmse_between(curve(r), curve(complete), years),
            "log_ratio_vs_truth": float(metrics.log10_bias(curve(r), epochs, years, SCALING.gen_years).mean()),
            "final_loglike": p.get("final_loglike"),
            "seconds": r.seconds,
        })
    return out


def summarize(rows: List[Dict]) -> List[Dict]:
    """Means over replicates, with the standard deviation of the RMSE vs
    truth and the range of θ relative to the complete-data fit."""
    keys = []
    for r in rows:
        k = (r["mask"], r["route"], r["mode"])
        if k not in keys:
            keys.append(k)
    out = []
    for mask, route, mode in keys:
        sel = [r for r in rows if (r["mask"], r["route"], r["mode"]) == (mask, route, mode)]

        def mean(col):
            vals = [r[col] for r in sel if r[col] is not None]
            return sum(vals) / len(vals) if vals else None

        rm = [r["rmse_vs_truth"] for r in sel]
        sd = math.sqrt(sum((x - mean("rmse_vs_truth")) ** 2 for x in rm) / (len(rm) - 1)) if len(rm) > 1 else 0.0
        thetas = [r["theta_vs_complete"] for r in sel]
        out.append({
            "mask": mask, "route": route, "mode": mode, "replicates": len(sel),
            "model": sel[0]["model"],
            "mean_callable": mean("mean_callable"),
            "theta_vs_complete": mean("theta_vs_complete"),
            "theta_vs_complete_min": min(thetas), "theta_vs_complete_max": max(thetas),
            "rmse_vs_truth": mean("rmse_vs_truth"), "rmse_vs_truth_sd": sd,
            "rmse_vs_complete": mean("rmse_vs_complete"),
            "log_ratio_vs_truth": mean("log_ratio_vs_truth"),
            "seconds": mean("seconds"),
        })
    return out


def write_tsv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def markdown(summary: List[Dict]) -> str:
    def fmt(v, d=3):
        return "-" if v is None else f"{v:.{d}f}"

    lines = [
        "| mask | route | --bin-callable | model | mean c/s | theta / complete (range) | RMSE vs truth (sd) | "
        "RMSE vs complete | mean log10 ratio vs truth | s per fit |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        lines.append(
            f"| {s['mask']} | {s['route']} | {s['mode']} | {s['model']} | {fmt(s['mean_callable'])} | "
            f"{fmt(s['theta_vs_complete'])} ({fmt(s['theta_vs_complete_min'])}-{fmt(s['theta_vs_complete_max'])}) | "
            f"{fmt(s['rmse_vs_truth'])} ({fmt(s['rmse_vs_truth_sd'])}) | "
            f"{fmt(s['rmse_vs_complete'])} | {fmt(s['log_ratio_vs_truth'])} | {fmt(s['seconds'], 1)} |"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experiment.validation.bin_callable",
        description="Validate --bin-callable on simulated genomes with callable masks.",
    )
    p.add_argument("--psmc-rs", required=True, type=Path, help="the psmc-rs binary to validate")
    p.add_argument("--out", required=True, type=Path, help="working and output directory")
    p.add_argument("--model", default=MODEL.name, choices=sorted(config.MODELS),
                   help="demographic model of the experiment pipeline (default: %(default)s)")
    p.add_argument("--experiment", default=EXPERIMENT,
                   help="seed namespace (config.seed's experiment; default: %(default)s)")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--chromosomes", type=int, default=10)
    p.add_argument("--chromosome-length", type=int, default=20_000_000)
    p.add_argument("--n-iter", type=int, default=INFERENCE.n_iter)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--evaluate-only", action="store_true", help="recompute the tables from the fits in --out")
    args = p.parse_args(argv)

    genome = config.Genome(n_chromosomes=args.chromosomes, chromosome_length=args.chromosome_length)
    model = config.MODELS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    fits_record = args.out / "fits.json"
    if args.evaluate_only:
        rows = [FitRow(**{**r, "json_path": Path(r["json_path"])}) for r in json.loads(fits_record.read_text())]
    else:
        rows = []
        for rep in range(1, args.replicates + 1):
            rows.extend(replicate(rep, genome, args.psmc_rs, args.out, args.threads, args.n_iter,
                                  model, args.experiment))
        fits_record.write_text(json.dumps([{**r.__dict__, "json_path": str(r.json_path)} for r in rows], indent=1))
    per_fit = evaluate(rows, model)
    summary = summarize(per_fit)
    write_tsv(args.out / "results.tsv", per_fit)
    write_tsv(args.out / "summary.tsv", summary)
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
