"""E7: 1000 Genomes (Fig 7, Figs S5 and S6).

The data stage (:mod:`..kg1000`, outside the runner: it streams 1.68 TB)
writes ``<runs>/E7/inputs/``: the sample table, one psmcfa per sample of the
cohort (chr1-22, the strict mask, non-PASS positions missing), the default
set's het-or-missing VCFs, the masks, the non-PASS positions and
``MANIFEST.json``. The runner's jobs fit from them; their provenance records
hold the checksums of every input, so a data stage that writes other bytes
re-runs what read them. Without ``inputs/MANIFEST.json`` (the stage has not
finished) E7 has no jobs. Layout under ``<runs>/E7/``::

    fits/rs-psmcfa/<sample>.json            psmc-rs 0.2 on the psmcfa (every sample; .psmc too for the default set)
    fits/rs-vcf/<sample>.json, .psmc        psmc-rs 0.2 on the VCF with the mask (default set)
    fits/c/<sample>.psmc                    C psmc on the psmcfa (default set)
    fits/rs-split/<sample>.json, .psmc      psmc-rs 0.2 on the psmcfa, the first window split (one per population)
    boot/<sample>/main.json, replicate_NNN.json, summary.{json,tsv}   psmc-rs's bootstrap (replicate .psmc not committed)
    qc/giab_<sample>.json                   heterozygous SNVs against GIAB (NA12878)
    provenance/<kind>/<sample>.json, tables/E7_*.csv, figures/ (Fig 7, S5, S6: experiment FIG)

Jobs: ``fit-rs-psmcfa`` (every sample: ``psmc-rs IN OUT N_ITER`` with the
recipe, ``--mu 1.25e-8 --gen-years 29 --threads 1``), ``fit-rs-vcf`` (the
default set: the same on its VCF with ``--vcf-mode mask --vcf-mask
mask/callable.bed --vcf-negative-mask nonpass/nonpass.bed.gz
--bin-callable auto``, psmc-rs's callable bases per bin), ``fit-c`` (the
default set: C psmc on the psmcfa, which reads gzip), ``boot-rs`` (the
bootstrap samples: ``--bootstrap 100``, trunks of 500,000 bins, seed
``seed("E7", sample, 0, "bootstrap")``; its main fit must be
``fit-rs-psmcfa``'s, field for field), ``fit-rs-split`` (the first default
sample of every population: ``--pattern 1+1+1+1+25*2+4+6``, the first
window of the recipe split into its four intervals, which removes the false
recent peaks Hilgers et al. 2025 found with ``4+...``), ``giab`` (NA12878:
``python -m experiment.pipeline.experiments.e7 giab``) and ``report``.

**Routes.** A psmcfa records no callable bases per bin, so its fit takes
every called bin as 100 bases, as C psmc does (Fig 5's "every called bin
full"); the strict mask leaves about a fifth of the called bins partly
callable. The VCF route weighs each bin by its callable bases. Fig 7's
population curves, the cohort spread and Fig S5 use the psmcfa fits of
every genome (61 to 113 per population); Fig S5 adds the median of each
population's default genomes' VCF fits, and ``E7_route_shift`` measures
the difference between the routes on the default set. psmc-rs and C psmc are compared on the identical psmcfa
(as E2: interval by interval on the window of 1% to 99% of coalescences
under C psmc's fit, :func:`..metrics.interval_diff`).

Tables (``tables/``):

* ``E7_samples.csv``: every sample: population, superpopulation, sex,
  whether a parent of a sample of the 3,202, admixed, raw depth, its roles.
* ``E7_inputs.csv``: per sample: bins, called and heterozygous bins, called
  bases (callable), heterozygous sites, the share of called bins partly
  callable, non-PASS positions, file sizes and the psmcfa's SHA-256.
* ``E7_fits.csv``: per sample and route: theta, rho, t_max, N0, the
  log-likelihoods, its own window, wall time and peak RSS.
* ``E7_curves.csv``: the step curves of every fit.
* ``E7_pop_bands.csv``: per population, superpopulation and all, per route:
  the median and the ``spread`` quantiles of Ne across individuals on a
  dense grid (geometric: quantiles of log10 Ne), and the median window.
* ``E7_rs_vs_c.csv``: the default set's psmc-rs and C psmc fits of the
  psmcfa compared (as ``E2_rs_vs_c``).
* ``E7_bootstrap.csv``, ``E7_bands.csv``: the bootstraps (as E2's).
* ``E7_route_shift.csv``: the VCF route against the psmcfa route.
* ``E7_split.csv``: the split-first-window fits against the recipe's: per
  interval, and the first window's Ne over the next one's (for the split fit
  its four intervals' duration-weighted log mean, with their range).
* ``E7_giab.csv``: NA12878's heterozygous SNVs against GIAB HG001 v4.2.1
  inside the strict mask and GIAB's confident regions.
* ``E7_throughput.csv``: per kind of fit, the measured wall times, peak RSS
  and core hours (C psmc only as measured on the default set).
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import config, kg1000, metrics, paths, provenance, results, tools
from ..runner import Job
from . import Context
from .e1 import write_csv
from .e2 import band, fitted_window, main_fit_mismatch, psmcfa_lengths, summary_difference
from .e5 import main_fit_difference, trunk_count

NAME = "E7"
MODULE = "experiment.pipeline.experiments.e7"
RS_ROUTES = ("rs-psmcfa", "rs-vcf", "rs-split")
ROUTES = ("rs-psmcfa", "rs-vcf", "c", "rs-split")
TABLES = {
    "samples": "E7_samples.csv",
    "inputs": "E7_inputs.csv",
    "fits": "E7_fits.csv",
    "curves": "E7_curves.csv",
    "pop_bands": "E7_pop_bands.csv",
    "rs_vs_c": "E7_rs_vs_c.csv",
    "bootstrap": "E7_bootstrap.csv",
    "bands": "E7_bands.csv",
    "route_shift": "E7_route_shift.csv",
    "split": "E7_split.csv",
    "giab": "E7_giab.csv",
    "throughput": "E7_throughput.csv",
}
KIND_OF = {"rs-psmcfa": "fit-rs-psmcfa", "rs-vcf": "fit-rs-vcf", "rs-split": "fit-rs-split", "c": "fit-c"}


# ---------------------------------------------------------------------------
# Layout and jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where E7's fits and tables are (the inputs: :class:`..kg1000.Layout`)."""

    root: Path

    @property
    def data(self) -> kg1000.Layout:
        return kg1000.Layout(self.root)

    def fit_json(self, route: str, sid: str) -> Path:
        return self.root / "fits" / route / f"{sid}.json"

    def fit_psmc(self, route: str, sid: str) -> Path:
        return self.root / "fits" / route / f"{sid}.psmc"

    def boot_dir(self, sid: str) -> Path:
        return self.root / "boot" / sid

    def boot_main(self, sid: str) -> Path:
        return self.boot_dir(sid) / "main.json"

    def boot_summary(self, sid: str, ext: str = "json") -> Path:
        return self.boot_dir(sid) / f"summary.{ext}"

    def boot_replicate(self, sid: str, i: int) -> Path:
        return self.boot_dir(sid) / f"replicate_{i:03d}.json"

    def giab(self, sid: str) -> Path:
        return self.root / "qc" / f"giab_{sid}.json"

    def record(self, kind: str, sid: str) -> Path:
        return self.root / "provenance" / kind / f"{sid}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_id(kind: str, sid: str) -> str:
    return f"{NAME}/{kind}/{sid}"


def bootstrap_seed(sid: str) -> int:
    return config.seed(NAME, sid, 0, "bootstrap")


def genome_mb(data: kg1000.Layout) -> Tuple[float, float]:
    """Mb of the stage's chromosomes (its chunks' span) and of the longest one, for the cost estimates."""
    spans: Dict[str, int] = {}
    for c in kg1000.read_chunks(data):
        spans[c.chrom] = spans.get(c.chrom, 0) + c.end - c.start + 1
    return sum(spans.values()) / 1e6, max(spans.values(), default=0) / 1e6


def vcf_fit_args(cfg: config.E7, data: kg1000.Layout) -> List[str]:
    """The options of a fit of a sample's VCF: mask mode, the strict mask, the non-PASS positions, callable bases."""
    out = ["--vcf-mode", "mask", "--vcf-mask", str(data.mask_all), "--vcf-negative-mask", str(data.nonpass_all),
           "--bin-callable", "auto"]
    if cfg.min_good_fraction is not None:
        out += ["--min-good-fraction", f"{cfg.min_good_fraction:g}"]
    return out


def bootstrap_args(cfg: config.E7, sid: str, boot_dir: Path) -> List[str]:
    return [
        "--bootstrap", str(cfg.bootstrap_replicates),
        "--bootstrap-block-size", str(cfg.bootstrap_trunk_bins),
        "--bootstrap-seed", str(bootstrap_seed(sid)),
        "--bootstrap-iters", str(cfg.inference.n_iter),
        "--bootstrap-dir", str(boot_dir),
    ]


def ready(ctx: Context) -> bool:
    """Whether the data stage has finished: ``assemble-samples``, its last step, wrote ``MANIFEST.json``."""
    return Layout(ctx.exp_dir(NAME)).data.manifest.exists()


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e7
    costs, e7c = ctx.profile.costs, cfg.costs
    lay = Layout(ctx.exp_dir(NAME))
    data = lay.data
    if not ready(ctx):
        ctx.log(f"E7: no {data.manifest}: run its data stage first (python -m experiment.pipeline.kg1000, or "
                "experiment/slurm/submit.sh e7-prep); E7 has no jobs until then")
        return []
    samples = kg1000.read_samples(data.samples)
    mb, longest_mb = genome_mb(data)
    inference = asdict(cfg.inference)
    scaling = asdict(cfg.scaling)
    c_tool, rs_tool = ctx.c_psmc(), ctx.rs02()
    rs_ident = {rs_tool.name: rs_tool.identity}
    fit_s = costs.rs02_seconds(mb, cfg.inference)
    out: List[Job] = []
    report_inputs: List[Path] = [data.samples, data.qc_samples, data.manifest]
    for s in samples:
        sid = s.sample_id
        psmcfa = data.psmcfa(sid)
        route = "rs-psmcfa"
        psmc_out = lay.fit_psmc(route, sid) if s.default else None
        outputs = [lay.fit_json(route, sid)] + ([psmc_out] if psmc_out else [])
        out.append(Job(
            id=job_id(KIND_OF[route], sid),
            command=tools.rs02_fit(rs_tool, cfg.inference, cfg.scaling, psmcfa, lay.fit_json(route, sid),
                                   threads=cfg.fit_threads, psmc_out=psmc_out),
            inputs=[psmcfa], outputs=outputs, cores=cfg.fit_threads, mem_gb=costs.rs02_gb, est_seconds=fit_s,
            settings={"inference": inference, "scaling": scaling, "threads": cfg.fit_threads},
            tools=rs_ident, description=f"psmc-rs 0.2 on {sid}'s psmcfa ({s.population})",
        ))
        report_inputs += outputs + [lay.record(KIND_OF[route], sid)]
        if s.default:
            vcf_inputs = [data.vcf(sid), data.mask_all, data.nonpass_all]
            route = "rs-vcf"
            outputs = [lay.fit_json(route, sid), lay.fit_psmc(route, sid)]
            out.append(Job(
                id=job_id(KIND_OF[route], sid),
                command=tools.rs02_fit(rs_tool, cfg.inference, cfg.scaling, data.vcf(sid), lay.fit_json(route, sid),
                                       threads=cfg.fit_threads, psmc_out=lay.fit_psmc(route, sid),
                                       extra=vcf_fit_args(cfg, data)),
                inputs=vcf_inputs, outputs=outputs, cores=cfg.fit_threads, mem_gb=e7c.vcf_fit_gb,
                est_seconds=fit_s + 30.0,
                settings={"inference": inference, "scaling": scaling, "threads": cfg.fit_threads,
                          "vcf": {"min_good_fraction": cfg.min_good_fraction}},
                tools=rs_ident, description=f"psmc-rs 0.2 on {sid}'s VCF with the strict mask ({s.population})",
            ))
            report_inputs += outputs + [lay.record(KIND_OF[route], sid)]
            if cfg.c_psmc:
                out.append(Job(
                    id=job_id("fit-c", sid),
                    command=tools.c_psmc_fit(c_tool, cfg.inference, psmcfa, lay.fit_psmc("c", sid)),
                    inputs=[psmcfa], outputs=[lay.fit_psmc("c", sid)],
                    mem_gb=round(costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * longest_mb, 2),
                    est_seconds=costs.c_psmc_s_per_mb_iter * mb * cfg.inference.n_iter,
                    settings={"inference": inference}, tools={c_tool.name: c_tool.identity},
                    description=f"C psmc on {sid}'s psmcfa ({s.population})",
                ))
                report_inputs += [lay.fit_psmc("c", sid), lay.record("fit-c", sid)]
        if s.bootstrap:
            outputs = [lay.boot_main(sid), lay.boot_summary(sid, "json"), lay.boot_summary(sid, "tsv")]
            out.append(Job(
                id=job_id("boot-rs", sid),
                command=tools.rs02_fit(rs_tool, cfg.inference, cfg.scaling, psmcfa, lay.boot_main(sid),
                                       threads=cfg.bootstrap_threads,
                                       extra=bootstrap_args(cfg, sid, lay.boot_dir(sid))),
                inputs=[psmcfa], outputs=outputs, cores=cfg.bootstrap_threads,
                mem_gb=round(costs.rs02_gb * cfg.bootstrap_threads, 2),
                est_seconds=fit_s + cfg.bootstrap_replicates * fit_s / max(cfg.bootstrap_threads, 1),
                settings={"inference": inference, "scaling": scaling, "threads": cfg.bootstrap_threads,
                          "bootstrap": {"replicates": cfg.bootstrap_replicates, "trunk_bins": cfg.bootstrap_trunk_bins,
                                        "iterations": cfg.inference.n_iter}},
                tools=rs_ident, seeds={"bootstrap": bootstrap_seed(sid)},
                description=f"psmc-rs 0.2 bootstrap ({cfg.bootstrap_replicates} replicates) on {sid}",
            ))
            report_inputs += outputs + [lay.record("boot-rs", sid)]
        if s.split_check:
            route = "rs-split"
            split = config.replace(cfg.inference, pattern=cfg.split_pattern)
            outputs = [lay.fit_json(route, sid), lay.fit_psmc(route, sid)]
            out.append(Job(
                id=job_id(KIND_OF[route], sid),
                command=tools.rs02_fit(rs_tool, split, cfg.scaling, psmcfa, lay.fit_json(route, sid),
                                       threads=cfg.fit_threads, psmc_out=lay.fit_psmc(route, sid)),
                inputs=[psmcfa], outputs=outputs, cores=cfg.fit_threads, mem_gb=costs.rs02_gb, est_seconds=fit_s,
                settings={"inference": asdict(split), "scaling": scaling, "threads": cfg.fit_threads},
                tools=rs_ident, description=f"psmc-rs 0.2 on {sid}'s psmcfa, first window split ({s.population})",
            ))
            report_inputs += outputs + [lay.record(KIND_OF[route], sid)]
        if s.giab:
            giab_inputs = [data.vcf(sid), data.giab_vcf, data.giab_bed, data.mask_all]
            out.append(Job(
                id=job_id("giab", sid),
                command=[paths.python(), "-m", MODULE, "giab", "--vcf", str(data.vcf(sid)), "--truth",
                         str(data.giab_vcf), "--truth-bed", str(data.giab_bed), "--mask", str(data.mask_all),
                         "--chromosomes", ",".join(cfg.chromosomes), "--sample", sid, "--out", str(lay.giab(sid))],
                inputs=giab_inputs, outputs=[lay.giab(sid)], mem_gb=e7c.giab_gb, est_seconds=e7c.giab_s,
                settings={"chromosomes": list(cfg.chromosomes)},
                tools={"pipeline": provenance.python_identity(["experiments/e7.py"])},
                description=f"{sid}'s heterozygous SNVs against GIAB",
            ))
            report_inputs.append(lay.giab(sid))
    n_fits = sum(1 for j in out if "/fit-" in j.id)
    out.append(Job(
        id=f"{NAME}/report",
        command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
        inputs=report_inputs,
        outputs=[lay.table(k) for k in TABLES],
        mem_gb=e7c.report_gb,
        est_seconds=e7c.report_s_base + e7c.report_s_per_fit * n_fits,
        settings={"e7": asdict(cfg)},
        tools={"pipeline": provenance.python_identity(
            ["experiments/e7.py", "experiments/e2.py", "experiments/e5.py", "experiments/e1.py", "kg1000.py",
             "results.py", "metrics.py", "demography.py"], ["numpy"])},
        description="E7 tables",
    ))
    return out


# ---------------------------------------------------------------------------
# GIAB: heterozygous SNVs of one sample against GIAB's truth
# ---------------------------------------------------------------------------


class Intervals:
    """Half-open intervals per chromosome, merged; membership by bisection."""

    def __init__(self, rows: Dict[str, List[Tuple[int, int]]]):
        self.starts: Dict[str, List[int]] = {}
        self.ends: Dict[str, List[int]] = {}
        for chrom, iv in rows.items():
            merged: List[List[int]] = []
            for a, b in sorted(iv):
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            self.starts[chrom] = [a for a, _ in merged]
            self.ends[chrom] = [b for _, b in merged]

    @classmethod
    def read_bed(cls, path: Path, chromosomes: Sequence[str]) -> "Intervals":
        rows: Dict[str, List[Tuple[int, int]]] = {c: [] for c in chromosomes}
        with _open_text(path) as f:
            for line in f:
                p = line.split("\t")
                if len(p) >= 3 and p[0] in rows:
                    rows[p[0]].append((int(p[1]), int(p[2])))
        return cls(rows)

    def intersect(self, other: "Intervals") -> "Intervals":
        rows: Dict[str, List[Tuple[int, int]]] = {}
        for chrom in self.starts:
            a_s, a_e = self.starts[chrom], self.ends[chrom]
            b_s, b_e = other.starts.get(chrom, []), other.ends.get(chrom, [])
            out, i, j = [], 0, 0
            while i < len(a_s) and j < len(b_s):
                lo, hi = max(a_s[i], b_s[j]), min(a_e[i], b_e[j])
                if hi > lo:
                    out.append((lo, hi))
                if a_e[i] < b_e[j]:
                    i += 1
                else:
                    j += 1
            rows[chrom] = out
        return Intervals(rows)

    def contains(self, chrom: str, pos1: int) -> bool:
        """Whether 1-based position ``pos1`` is inside."""
        starts = self.starts.get(chrom)
        if not starts:
            return False
        k = bisect.bisect_right(starts, pos1 - 1) - 1
        return k >= 0 and pos1 - 1 < self.ends[chrom][k]

    def bases(self) -> int:
        return sum(e - s for c in self.starts for s, e in zip(self.starts[c], self.ends[c]))


def _open_text(path: Path):
    p = Path(path)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def vcf_records(path: Path, chromosomes: Sequence[str]) -> Iterator[Tuple[str, int, str, List[str], str, List[str]]]:
    """``(chrom, pos, ref, alts, filter, gt alleles)`` of a one-sample VCF's records on ``chromosomes``."""
    keep = set(chromosomes)
    with _open_text(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] not in keep:
                continue
            gt = p[9].split(":", 1)[0] if len(p) > 9 else "."
            yield p[0], int(p[1]), p[3], p[4].split(","), p[6], [a for a in gt.replace("|", "/").split("/")]


def classify(ref: str, alts: List[str], gt: List[str]) -> Tuple[str, Optional[frozenset]]:
    """("het-snv", alleles), ("het-other", None), ("missing", None) or ("hom", None), as psmc-rs reads a genotype."""
    if not gt or any(a in (".", "") for a in gt):
        return "missing", None
    alleles = [ref] + alts
    try:
        called = [alleles[int(a)] for a in gt]
    except (ValueError, IndexError):
        return "missing", None
    if len(set(called)) < 2:
        return "hom", None
    if all(len(a) == 1 and a.upper() in "ACGT" for a in called):
        return "het-snv", frozenset(a.upper() for a in called)
    return "het-other", None


def giab_compare(vcf: Path, truth: Path, truth_bed: Path, mask: Path, chromosomes: Sequence[str], sample: str) -> Dict:
    """Heterozygous SNVs of ``vcf`` (one sample's het-or-missing VCF) against GIAB's, inside mask and GIAB's BED.

    A call is a PASS (or ".") record whose genotype is heterozygous with
    single-base alleles (what psmc-rs reads as a heterozygous site); a truth
    site a GIAB record with such a genotype. They match when position and
    alleles agree; a truth site the sample lacks is a false negative, by
    what the sample's VCF has there (a non-PASS record, a missing genotype,
    a heterozygous indel, or nothing: called homozygous).
    """
    region = Intervals.read_bed(mask, chromosomes).intersect(Intervals.read_bed(truth_bed, chromosomes))
    ours: Dict[Tuple[str, int], Tuple[str, Optional[frozenset]]] = {}
    for chrom, pos, ref, alts, filt, gt in vcf_records(vcf, chromosomes):
        if not region.contains(chrom, pos):
            continue
        kind, alleles = classify(ref, alts, gt)
        if filt not in ("PASS", "."):
            kind = "nonpass"
        ours[(chrom, pos)] = (kind, alleles)
    truth_hets: Dict[Tuple[str, int], frozenset] = {}
    for chrom, pos, ref, alts, _filt, gt in vcf_records(truth, chromosomes):
        if not region.contains(chrom, pos):
            continue
        kind, alleles = classify(ref, alts, gt)
        if kind == "het-snv":
            truth_hets[(chrom, pos)] = alleles
    called = {k: a for k, (kind, a) in ours.items() if kind == "het-snv"}
    tp = sum(1 for k, a in called.items() if truth_hets.get(k) == a)
    fn_kinds = {"nonpass": 0, "missing": 0, "het-other": 0, "hom": 0, "absent": 0, "het-snv": 0}
    for k, a in truth_hets.items():
        if called.get(k) == a:
            continue
        kind = ours.get(k, ("absent", None))[0]
        fn_kinds[kind] += 1
    fp, fn = len(called) - tp, len(truth_hets) - tp
    return {
        "sample_id": sample,
        "eval_bases": region.bases(),
        "truth_het_snvs": len(truth_hets),
        "called_het_snvs": len(called),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(called) if called else None,
        "recall": tp / len(truth_hets) if truth_hets else None,
        "fn_nonpass": fn_kinds["nonpass"],
        "fn_missing_gt": fn_kinds["missing"],
        "fn_het_indel": fn_kinds["het-other"],
        "fn_hom_or_absent": fn_kinds["hom"] + fn_kinds["absent"],
        "fn_other_alleles": fn_kinds["het-snv"],
    }


# ---------------------------------------------------------------------------
# Reading the outputs
# ---------------------------------------------------------------------------


def run_record(path: Path) -> Dict:
    try:
        return json.loads(Path(path).read_text()).get("run", {})
    except (OSError, ValueError):
        return {}


@dataclass
class SampleFits:
    """The fits of one sample, read back."""

    sample: kg1000.Sample
    params: Dict[str, Dict]  # psmc-rs JSON by route
    fits: Dict[str, results.Fit]  # by route (c included)
    curves: Dict[str, results.Curve]
    windows: Dict[str, Tuple[float, float]]
    c: Optional[results.PsmcFile]
    runs: Dict[str, Dict]


def load_sample(lay: Layout, cfg: config.E7, s: kg1000.Sample) -> SampleFits:
    sid, n_iter, bin_size, sc = s.sample_id, cfg.inference.n_iter, cfg.inference.bin_size, cfg.scaling
    routes = ["rs-psmcfa"] + (["rs-vcf"] if s.default else []) + (["rs-split"] if s.split_check else [])
    params = {r: results.read_rs_json(lay.fit_json(r, sid)) for r in routes}
    for r, p in params.items():
        if int(p.get("n_iter", -1)) != n_iter:
            raise ValueError(f"{lay.fit_json(r, sid)}: not a fit of {n_iter} iterations")
    fits = {r: results.fit_from_rs(p) for r, p in params.items()}
    c = None
    if s.default and cfg.c_psmc:
        c = results.parse_psmc(lay.fit_psmc("c", sid))
        if c.last().round != n_iter:
            raise ValueError(f"{lay.fit_psmc('c', sid)}: last complete round {c.last().round}, expected {n_iter}")
        inp = params["rs-psmcfa"].get("input", {})
        called = inp.get("bins", -1) - inp.get("missing_bins", 0)
        if not (c.header.get("sum_L") == called and c.header.get("sum_n") == inp.get("het_bins")):
            raise ValueError(f"{sid}: the tools read different inputs (C sum_L {c.header.get('sum_L')}, sum_n "
                             f"{c.header.get('sum_n')}; psmc-rs {called} called bins, {inp.get('het_bins')} heterozygous)")
        fits["c"] = c.last().fit()
    curves = {r: f.curve(sc.mu, sc.gen_years, bin_size) for r, f in fits.items()}
    windows = {r: fitted_window(f, sc, bin_size, cfg.grid.quantiles) for r, f in fits.items()}
    runs = {r: run_record(lay.record(KIND_OF[r], sid)) for r in fits}
    return SampleFits(s, params, fits, curves, windows, c, runs)


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------


def _run_fields(run: Dict) -> Dict:
    return {"wall_seconds": run.get("wall_seconds"), "peak_rss_mb": run.get("peak_rss_mb"),
            "user_seconds": run.get("user_seconds"), "run_start": run.get("start"), "host": run.get("host")}


def _who(s: kg1000.Sample) -> Dict:
    return {"sample_id": s.sample_id, "population": s.population, "superpopulation": s.superpopulation,
            "admixed": s.admixed}


def fit_rows(sf: SampleFits, cfg: config.E7) -> List[Dict]:
    n, bin_size = cfg.inference.n_iter, cfg.inference.bin_size
    rows = []
    for route in ROUTES:
        if route not in sf.fits:
            continue
        fit = sf.fits[route]
        n0 = fit.n0(cfg.scaling.mu, bin_size)
        if route == "c":
            ll = sf.c.loglikes()
            called, het = sf.c.header.get("sum_L"), sf.c.header.get("sum_n")
        else:
            ll = dict(enumerate(results.rs_loglikes(sf.params[route])))
            inp = sf.params[route].get("input", {})
            called, het = inp.get("bins", 0) - inp.get("missing_bins", 0), inp.get("het_bins")
        w = sf.windows[route]
        rows.append({
            **_who(sf.sample), "default": sf.sample.default, "route": route, "pattern": fit.pattern,
            "called_bins": called, "het_bins": het, "theta": fit.theta, "rho": fit.rho,
            "theta_per_bp": fit.theta / bin_size, "t_max": fit.t_max, "n0": n0,
            "ne_recent": fit.lambdas[0] * n0, "ne_oldest": fit.lambdas[-1] * n0,
            "loglike_common": ll.get(n - 1), "loglike_final": ll.get(n), "window_lo_years": w[0],
            "window_hi_years": w[1], "threads": 1 if route == "c" else cfg.fit_threads, **_run_fields(sf.runs[route]),
        })
    return rows


def curve_rows(sf: SampleFits) -> List[Dict]:
    rows = []
    for route in ROUTES:
        if route not in sf.curves:
            continue
        c = sf.curves[route]
        for k, (y, ne) in enumerate(zip(c.years, c.ne)):
            rows.append({"sample_id": sf.sample.sample_id, "route": route, "interval": k, "years": y, "ne": ne})
    return rows


def pop_band_rows(fits: Sequence[SampleFits], cfg: config.E7) -> List[Dict]:
    """Per group (population, superpopulation, all) and rs route: median and spread across individuals."""
    import numpy as np

    lo_y, hi_y, npts = cfg.display_years
    years = np.logspace(math.log10(lo_y), math.log10(hi_y), int(npts))
    qs = (cfg.spread[0], 0.5, cfg.spread[1])
    groups: Dict[Tuple[str, str, str], List[SampleFits]] = {}
    for sf in fits:
        s = sf.sample
        for route in ("rs-psmcfa", "rs-vcf"):
            if route not in sf.curves:
                continue
            for kind, name in (("population", s.population), ("superpopulation", s.superpopulation), ("all", "all")):
                groups.setdefault((kind, name, route), []).append(sf)
    rows = []
    for (kind, name, route), members in sorted(groups.items()):
        logs = np.vstack([np.log10(m.curves[route].at(years)) for m in members])
        lo, med, hi = 10 ** np.quantile(logs, qs, axis=0)
        w_lo = float(np.median([m.windows[route][0] for m in members]))
        w_hi = float(np.median([m.windows[route][1] for m in members]))
        sup = members[0].sample.superpopulation if kind == "population" else (name if kind == "superpopulation" else "")
        for t, a, b, c in zip(years, lo, med, hi):
            rows.append({"group_kind": kind, "group": name, "superpopulation": sup,
                         "admixed": kind == "population" and name in cfg.admixed, "route": route, "n": len(members),
                         "years": float(t), "ne_median": float(b), "ne_lo": float(a), "ne_hi": float(c),
                         "spread_lo_quantile": cfg.spread[0], "spread_hi_quantile": cfg.spread[1],
                         "window_lo_years": w_lo, "window_hi_years": w_hi, "in_window": bool(w_lo <= t <= w_hi)})
    return rows


def rs_vs_c_row(sf: SampleFits, cfg: config.E7) -> Dict:
    n = cfg.inference.n_iter
    ll_c, ll_rs = sf.c.loglikes(), dict(enumerate(results.rs_loglikes(sf.params["rs-psmcfa"])))
    common = sorted(i for i in set(ll_c) & set(ll_rs) if ll_c[i])
    rel = {i: abs(ll_rs[i] - ll_c[i]) / abs(ll_c[i]) for i in common}
    fc, fr = sf.fits["c"], sf.fits["rs-psmcfa"]
    window = sf.windows["c"]
    diff = metrics.interval_diff(sf.curves["rs-psmcfa"], sf.curves["c"], window)
    return {
        **_who(sf.sample), "window_lo_years": window[0], "window_hi_years": window[1],
        "intervals_in_window": len(diff.interval),
        "rmse_log10_ne_rs_vs_c_per_interval": diff.rmse,
        "max_abs_log10_ne_rs_vs_c_per_interval": diff.max_abs,
        "max_abs_log10_ne_rs_vs_c_any_interval": diff.max_abs_any,
        "theta_rel_diff": fr.theta / fc.theta - 1.0, "rho_rel_diff": fr.rho / fc.rho - 1.0,
        "t_max_rel_diff": fr.t_max / fc.t_max - 1.0,
        "loglike_iteration0_rel_diff": rel.get(0), "loglike_max_rel_diff": max(rel.values()) if rel else None,
        "loglike_common_diff": (ll_rs[n - 1] - ll_c[n - 1]) if (n - 1) in common else None,
    }


def route_shift_row(sf: SampleFits, qc: Dict[str, str], cfg: config.E7) -> Dict:
    fv, fp = sf.fits["rs-vcf"], sf.fits["rs-psmcfa"]
    window = sf.windows["rs-psmcfa"]
    diff = metrics.interval_diff(sf.curves["rs-vcf"], sf.curves["rs-psmcfa"], window)
    w = sum(diff.weight)
    called_bins, called_bases = int(qc["called_bins"]), int(qc["called_bases"])
    return {
        **_who(sf.sample), "theta_psmcfa": fp.theta, "theta_vcf": fv.theta,
        "theta_rel_vcf_vs_psmcfa": fv.theta / fp.theta - 1.0,
        "window_lo_years": window[0], "window_hi_years": window[1],
        "mean_log10_ne_vcf_vs_psmcfa": sum(x * d for x, d in zip(diff.weight, diff.log10_ratio)) / w,
        "rmse_log10_ne_vcf_vs_psmcfa_per_interval": diff.rmse,
        "partly_callable_bin_fraction": int(qc["partial_bins"]) / called_bins if called_bins else None,
        "missing_share_of_called_bins": 1.0 - called_bases / (called_bins * cfg.inference.bin_size) if called_bins else None,
    }


def split_row(sf: SampleFits) -> Dict:
    fd, fs = sf.fits["rs-psmcfa"], sf.fits["rs-split"]
    window = sf.windows["rs-psmcfa"]
    diff = metrics.interval_diff(sf.curves["rs-split"], sf.curves["rs-psmcfa"], window)
    cd, cs = sf.curves["rs-psmcfa"].ne, sf.curves["rs-split"].ne
    first_split = cs[:4]
    # The split fit's first window: its four intervals' log10 Ne averaged by their durations (comparable with the
    # default pattern's single first lambda over the same span); the four intervals' range is reported too, since
    # one genome barely constrains such short intervals and one of them can run away.
    years = sf.curves["rs-split"].years
    spans = [years[k + 1] - years[k] for k in range(4)]
    first_split_mean = 10 ** (sum(w * math.log10(n) for w, n in zip(spans, first_split)) / sum(spans))
    return {
        **_who(sf.sample), "pattern_default": fd.pattern, "pattern_split": fs.pattern,
        "window_lo_years": window[0], "window_hi_years": window[1],
        "rmse_log10_ne_split_vs_default_per_interval": diff.rmse,
        "max_abs_log10_ne_split_vs_default_per_interval": diff.max_abs,
        "ne_first_window_default": cd[0], "ne_first_window_split_min": min(first_split),
        "ne_first_window_split_max": max(first_split), "ne_next_default": cd[4], "ne_next_split": cs[4],
        "log10_first_vs_next_default": math.log10(cd[0] / cd[4]),
        "log10_first_vs_next_split": math.log10(first_split_mean / cs[4]),
        "log10_first_vs_next_split_lo": math.log10(min(first_split) / cs[4]),
        "log10_first_vs_next_split_hi": math.log10(max(first_split) / cs[4]),
        "theta_rel_split_vs_default": fs.theta / fd.theta - 1.0,
    }


def bootstrap_rows(lay: Layout, cfg: config.E7, sf: SampleFits) -> Tuple[Dict, List[Dict]]:
    """The bootstrap's row (as ``E2_bootstrap``) and its band on the dense grid."""
    import numpy as np

    sid, sc, bin_size = sf.sample.sample_id, cfg.scaling, cfg.inference.bin_size
    summary = json.loads(lay.boot_summary(sid).read_text())
    expect = {"block_size": cfg.bootstrap_trunk_bins, "seed": bootstrap_seed(sid)}
    for key, value in expect.items():
        if summary.get(key) != value:
            raise ValueError(f"{lay.boot_summary(sid)}: {key} {summary.get(key)}, expected {value}")
    skipped = set(summary.get("skipped_replicates", []))
    reps = [(i, results.read_rs_json(lay.boot_replicate(sid, i)))
            for i in range(1, cfg.bootstrap_replicates + 1) if i not in skipped]
    if len(reps) != summary.get("n_replicates"):
        raise ValueError(f"{sid}: {len(reps)} replicate files, summary.json counts {summary.get('n_replicates')}")
    curves = [results.fit_from_rs(p).curve(sc.mu, sc.gen_years, bin_size) for _, p in reps]
    main = results.read_rs_json(lay.boot_main(sid))
    differing = main_fit_mismatch(main, sf.params["rs-psmcfa"])
    window = sf.windows["c"] if "c" in sf.windows else sf.windows["rs-psmcfa"]
    grid = metrics.log_grid(window[0], window[1], cfg.grid.points)
    lo, med, hi = band(curves, grid, (cfg.band[0], 0.5, cfg.band[1]))
    rs_ne = sf.curves["rs-psmcfa"].at(grid)
    c_ne = sf.curves["c"].at(grid) if "c" in sf.curves else None
    lengths = psmcfa_lengths(lay.data.psmcfa(sid))  # gzip, one member per chromosome
    row = {
        **_who(sf.sample), "replicates": len(reps), "skipped_replicates": len(skipped), "seed": summary.get("seed"),
        "trunk_bins": cfg.bootstrap_trunk_bins, "trunk_mb": cfg.bootstrap_trunk_bins * bin_size / 1e6,
        "trunks": trunk_count(lengths, cfg.bootstrap_trunk_bins), "band_lo_quantile": cfg.band[0],
        "band_hi_quantile": cfg.band[1], "window_lo_years": window[0], "window_hi_years": window[1],
        "band_width_log10_median": float(np.median(np.log10(hi) - np.log10(lo))),
        "c_in_band_fraction": float(np.mean((lo <= c_ne) & (c_ne <= hi))) if c_ne is not None else None,
        "rs_in_band_fraction": float(np.mean((lo <= rs_ne) & (rs_ne <= hi))),
        "summary_max_rel_diff": summary_difference(sf.curves["rs-psmcfa"], curves, summary),
        "main_identical_to_fit": not differing,
        "main_fit_max_abs_diff": main_fit_difference(main, sf.params["rs-psmcfa"]),
        "main_fit_fields_differing": ";".join(differing), "threads": cfg.bootstrap_threads,
        **_run_fields(run_record(lay.record("boot-rs", sid))),
    }
    lo_y, hi_y, npts = cfg.display_years
    years = np.logspace(math.log10(lo_y), math.log10(hi_y), int(npts))
    blo, bmed, bhi = band(curves, years, (cfg.band[0], 0.5, cfg.band[1]))
    rs_y = sf.curves["rs-psmcfa"].at(years)
    c_y = sf.curves["c"].at(years) if "c" in sf.curves else [None] * len(years)
    bands = [{"sample_id": sid, "years": float(t), "in_window": bool(window[0] <= t <= window[1]),
              "ne_c": None if c is None else float(c), "ne_rs": float(r), "ne_band_lo": float(a),
              "ne_band_median": float(m), "ne_band_hi": float(b)}
             for t, c, r, a, m, b in zip(years, c_y, rs_y, blo, bmed, bhi)]
    return row, bands


def throughput_rows(fits: Sequence[SampleFits], lay: Layout, cfg: config.E7, boots: Sequence[str]) -> List[Dict]:
    import numpy as np

    rows = []
    per_kind: Dict[str, List[Tuple[Dict, int]]] = {}
    for sf in fits:
        for route in sf.fits:
            per_kind.setdefault(route, []).append((sf.runs[route], 1 if route == "c" else cfg.fit_threads))
    for sid in boots:
        per_kind.setdefault("boot-rs", []).append((run_record(lay.record("boot-rs", sid)), cfg.bootstrap_threads))
    for route, runs in per_kind.items():
        walls = [r["wall_seconds"] for r, _ in runs if r.get("wall_seconds") is not None]
        rss = [r["peak_rss_mb"] for r, _ in runs if r.get("peak_rss_mb") is not None]
        rows.append({
            "route": route, "tool": "c" if route == "c" else "rs", "jobs": len(runs), "measured": len(walls),
            "threads": runs[0][1], "wall_median_s": float(np.median(walls)) if walls else None,
            "wall_min_s": min(walls) if walls else None, "wall_max_s": max(walls) if walls else None,
            "core_hours": sum(r["wall_seconds"] * t for r, t in runs if r.get("wall_seconds") is not None) / 3600,
            "peak_rss_median_mb": float(np.median(rss)) if rss else None, "peak_rss_max_mb": max(rss) if rss else None,
        })
    return rows


def input_row(s: kg1000.Sample, qc: Dict[str, str], nonpass: int, cfg: config.E7) -> Dict:
    called_bins, called = int(qc["called_bins"]), int(qc["called_bases"])
    return {
        **_who(s), "default": s.default, "raw_depth": s.raw_depth, "input": cfg.input,
        "bins": int(qc["bins"]), "called_bins_T_K": int(qc["t_bins"]) + int(qc["k_bins"]),
        "het_bins": int(qc["k_bins"]), "missing_bins": int(qc["missing_bins"]), "callable_bases": called,
        "het_sites": int(qc["het_sites"]), "het_per_callable_kb": 1e3 * int(qc["het_sites"]) / called if called else None,
        "partly_callable_bin_fraction": int(qc["partial_bins"]) / called_bins if called_bins else None,
        "nonpass_positions": nonpass, "vcf_bytes": int(qc["vcf_bytes"]), "psmcfa_bytes": int(qc["psmcfa_bytes"]),
        "psmcfa_sha256": qc["psmcfa_sha256"],
    }


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E7 table from the fits; returns the paths written."""
    cfg = ctx.profile.e7
    lay = Layout(ctx.exp_dir(NAME))
    data = lay.data
    samples = kg1000.read_samples(data.samples)
    qc = {r["sample_id"]: r for r in kg1000.read_tsv(data.qc_samples)}
    manifest = json.loads(data.manifest.read_text())
    nonpass = sum(int(c.get("nonpass", 0)) for c in manifest.get("chromosomes", {}).values())
    rows: Dict[str, List[Dict]] = {k: [] for k in TABLES}
    fits: List[SampleFits] = []
    for s in samples:
        sf = load_sample(lay, cfg, s)
        fits.append(sf)
        rows["samples"].append({**s.row(), "fit_routes": ";".join(r for r in ROUTES if r in sf.fits)})
        rows["inputs"].append(input_row(s, qc[s.sample_id], nonpass, cfg))
        rows["fits"] += fit_rows(sf, cfg)
        rows["curves"] += curve_rows(sf)
        if "c" in sf.fits:
            rows["rs_vs_c"].append(rs_vs_c_row(sf, cfg))
        if "rs-vcf" in sf.fits:
            rows["route_shift"].append(route_shift_row(sf, qc[s.sample_id], cfg))
        if "rs-split" in sf.fits:
            rows["split"].append(split_row(sf))
        if s.bootstrap:
            row, bands = bootstrap_rows(lay, cfg, sf)
            rows["bootstrap"].append(row)
            rows["bands"] += bands
        if s.giab:
            rows["giab"].append(json.loads(lay.giab(s.sample_id).read_text()))
    rows["pop_bands"] = pop_band_rows(fits, cfg)
    rows["throughput"] = throughput_rows(fits, lay, cfg, [s.sample_id for s in samples if s.bootstrap])
    written = {}
    for key, table in rows.items():
        write_csv(lay.table(key), table)
        written[key] = lay.table(key)
    differ = [r for r in rows["bootstrap"] if not r["main_identical_to_fit"]]
    if differ:
        raise ValueError("the main fit of a bootstrap run differs from fit-rs-psmcfa's fit (psmc-rs's results must not "
                         "depend on --threads): " + "; ".join(f"{r['sample_id']}: {r['main_fit_fields_differing']}"
                                                             for r in differ))
    return written


# ---------------------------------------------------------------------------
# Command line (the giab job)
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="Helpers of experiment E7's jobs.")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("giab", help="a sample's heterozygous SNVs against GIAB's truth")
    g.add_argument("--vcf", type=Path, required=True)
    g.add_argument("--truth", type=Path, required=True)
    g.add_argument("--truth-bed", type=Path, required=True)
    g.add_argument("--mask", type=Path, required=True)
    g.add_argument("--chromosomes", required=True)
    g.add_argument("--sample", required=True)
    g.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    res = giab_compare(args.vcf, args.truth, args.truth_bed, args.mask, args.chromosomes.split(","), args.sample)
    provenance.write_json_atomic(args.out, res)
    print(f"{args.sample}: precision {res['precision']}, recall {res['recall']} ({res['truth_het_snvs']} GIAB "
          f"heterozygous SNVs in {res['eval_bases']:,} bp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Intervals", "Layout", "NAME", "ROUTES", "SampleFits", "TABLES", "bootstrap_seed", "classify", "giab_compare",
    "jobs", "load_sample", "pop_band_rows", "ready", "report", "vcf_fit_args",
]
