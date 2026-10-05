"""E2: real genomes (Fig 3; Table S5).

For every genome of :class:`.config.E2` (three turtle assemblies and the
chromosome 1 of a Papuan Highlands human, which is not a whole genome) the
runner fits the packaged psmcfa (``experiment/real_data``) with C psmc and
with psmc-rs 0.2 (the identical file and settings), fits psmc-rs's
C-compatible bootstrap, runs ``psmc-rs qc``, and then writes the tables and
Figure 3. Layout under ``<runs>/E2/``::

    fits/<sample>.c.psmc                          C psmc, every round
    fits/<sample>.rs.json, <sample>.rs.psmc       psmc-rs 0.2, one thread (JSON and .psmc)
    bootstrap/<sample>/trunk<B>/main.json         psmc-rs 0.2 with --bootstrap: the main fit, again
    bootstrap/<sample>/trunk<B>/replicate_NNN.json  the replicates' parameters, and summary.{json,tsv}
    bootstrap/<sample>/trunk<B>/replicate_NNN.psmc  the replicates' rounds (not committed)
    qc/<sample>.qc.json, <sample>.qc.txt          psmc-rs qc: JSON and the text report
    inputs/<sample>.psmcfa                        smoke and tiny profiles only: the first bins (not committed)
    provenance/<kind>/<sample>.json               one record per job
    tables/E2_*.csv                               written by ``report E2``
    figures/E2_fig3_real_genomes.{pdf,png}        Figure 3, drawn from the tables
    logs/                                         runner and job logs (not committed)

Jobs per genome: ``fit-c`` (C psmc), ``fit-rs`` (psmc-rs 0.2),
``boot-rs/trunk<B>`` (psmc-rs 0.2 with ``--bootstrap``, at the default
trunks and, where they are few, at the fine trunks) and ``qc`` (``psmc-rs qc
--json``, with the default trunk size), plus ``subset`` when the profile
fits only the first bins of each genome; then ``report`` and ``figure``.

**Settings.** Both tools fit with the psmc README's human recipe
(``-N25 -t15 -r5 -p "4+25*2+4+6"``; :class:`.config.Inference`). Curves are
scaled per genome (:class:`.config.RealGenome`): the turtles with mu 7.9e-9
and g 15, whose source the repository does not record (the config flags
them as unverified), the human with mu 1.25e-8 and g 25.

**Bootstrap.** ``boot-rs`` runs ``psmc-rs IN main.json N_ITER --bootstrap
B --bootstrap-block-size T --bootstrap-seed S``: psmc-rs fits the main model
again, then B replicates, each drawing splitfa trunks of T bins (default
500,000, splitfa's 50 Mb) with replacement and fitted from a fresh start, as
``splitfa | psmc -b`` does. S is ``config.seed("E2", sample_id, 0,
"bootstrap")``. psmc-rs's output does not depend on ``--threads``, so
``main.json`` must equal ``fit-rs``'s JSON byte for byte; the report checks
it. A genome whose default trunks are fewer than ``few_trunks`` (Papuan
chr1: 5 trunks of 50 Mb) is bootstrapped a second time with trunks of
``fine_trunk_bins`` (5 Mb, a size E5 calibrates), seed stage
``bootstrap/trunk<B>``. E5 calibrates bootstraps of chromosome-scale
sequences; the turtles' default bootstraps resample hundreds of whole
scaffolds of very different lengths, which E5 does not calibrate. The job
declares ``main.json`` and the summary only: psmc-rs skips a replicate
whose fresh start is invalid, and the report reads exactly the replicates
the summary names. The replicates' parameters (``replicate_NNN.json``) are
committed and tabulated in ``tables/E2_bootstrap_replicates.csv``; their
``.psmc`` files (every EM round, about 100 kB each) are not.

**Comparison window.** A real genome has no true Ne(t), so the informative
window of :class:`.config.TimeGrid` is taken from the fitted model: the 1%
and 99% quantiles of the pairwise coalescence time under C psmc's final
model, in the genome's years. Outside it less than 1% of the genome's
coalescences fall, and both tools only extend their first and last lambda.
The tools and the bootstrap band are compared at 200 log-spaced times in
it; the fixed window (10 ky to 1 My) is reported next to it.

**Band.** At each time, the 2.5% and 97.5% quantiles (linear interpolation,
as psmc-rs computes them) of the replicates' Ne. Each replicate is read as
a step function in its own years (its own N0). The report recomputes
psmc-rs's ``summary.json`` from the replicate JSONs as a check of both.

Tables (``tables/``):

* ``E2_fit_summary.csv``: per genome and tool: scaling (mu, g and its
  source), sequences, called and heterozygous bins, theta, rho, t_max, N0,
  log-likelihoods, wall time and peak RSS.
* ``E2_rs_vs_c.csv``: per genome: RMSE of log10 Ne between the tools (window
  and fixed window), the largest difference, relative differences of theta,
  rho and t_max and of the log-likelihoods, and whether the bootstrap run's
  main fit equals ``fit-rs``'s output.
* ``E2_bootstrap.csv``: per genome and trunk size: replicates, trunks
  (``coarse_band`` when fewer than psmc-rs's 20), seed, the band's width,
  how much of C psmc's and psmc-rs's curves it holds, and the check against
  psmc-rs's own summary.
* ``E2_bootstrap_replicates.csv``: the parameters of every replicate.
* ``E2_bands.csv``: per genome and trunk size, both curves and the band on
  a dense grid (Fig 3).
* ``E2_curves.csv``: the fitted step curves of both tools (Fig 3).
* ``E2_traces.csv``: log-likelihood, theta, rho and t_max per EM iteration.
* ``E2_qc.csv`` and ``E2_qc_findings.csv``: Table S5, the metrics and the
  findings of ``psmc-rs qc`` for each genome.

Run ``python -m experiment.pipeline.experiments.e2 subset SRC DST
--max-bins N`` to write the first N bins of a psmcfa (the ``subset`` job).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import write_csv
from .e5 import trunk_count

NAME = "E2"
TOOLS = ("c", "rs")
MODULE = "experiment.pipeline.experiments.e2"
FIGURE_MODULE = "experiment.pipeline.figures.fig3_real_genomes"
FIGURE_STEM = "E2_fig3_real_genomes"
FIGURE_FORMATS = ("pdf", "png")
TABLES = {
    "fit_summary": "E2_fit_summary.csv",
    "pairs": "E2_rs_vs_c.csv",
    "bootstrap": "E2_bootstrap.csv",
    "replicates": "E2_bootstrap_replicates.csv",
    "bands": "E2_bands.csv",
    "curves": "E2_curves.csv",
    "traces": "E2_traces.csv",
    "qc": "E2_qc.csv",
    "qc_findings": "E2_qc_findings.csv",
}
FIGURE_TABLES = ("fit_summary", "pairs", "bootstrap", "bands", "curves")
SUMMARY_PROBS = (0.025, 0.5, 0.975)  # the quantiles of psmc-rs's summary.json


# ---------------------------------------------------------------------------
# psmcfa inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """What the cost estimates need of an input: sequences, bins and the longest sequence."""

    sequences: int
    bins: int
    longest: int


def read_psmcfa(path: Path) -> Iterator[Tuple[str, str]]:
    """``(header, bins)`` of each record of a psmcfa file, in order (header without '>')."""
    name: Optional[str] = None
    chunks: List[str] = []
    with open(path, encoding="ascii") as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].rstrip("\r\n"), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


@functools.lru_cache(maxsize=None)
def _record_lengths(path: str, size: int, mtime_ns: int) -> Tuple[int, ...]:
    data = Path(path).read_bytes()
    out = []
    for chunk in data.split(b">")[1:]:
        _, _, body = chunk.partition(b"\n")
        out.append(len(body.translate(None, b" \t\r\n")))
    return tuple(out)


def psmcfa_lengths(path: Path) -> Tuple[int, ...]:
    """The bins of each record of a psmcfa file (cached by size and modification time)."""
    st = Path(path).stat()
    return _record_lengths(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def prefix_lengths(lengths: Sequence[int], max_bins: int) -> Tuple[int, ...]:
    """The record lengths of the first ``max_bins`` bins: records in order, the last one cut, empty ones left out."""
    if max_bins < 1:
        raise ValueError(f"max_bins must be >= 1, got {max_bins}")
    out: List[int] = []
    left = max_bins
    for n in lengths:
        if left <= 0:
            break
        take = min(n, left)
        if take > 0:
            out.append(take)
        left -= take
    return tuple(out)


def input_shape(lengths: Sequence[int], max_bins: Optional[int] = None) -> Shape:
    if max_bins is not None:
        lengths = prefix_lengths(lengths, max_bins)
    return Shape(len(lengths), sum(lengths), max(lengths, default=0))


def write_psmcfa_prefix(src: Path, dst: Path, max_bins: int) -> Shape:
    """Write the first ``max_bins`` bins of ``src`` to ``dst``, laid out as fq2psmcfa writes them.

    Records are kept in order with their headers; the last one is cut and
    empty ones are left out (as :func:`prefix_lengths`). Written atomically.
    """
    if max_bins < 1:
        raise ValueError(f"max_bins must be >= 1, got {max_bins}")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    lengths: List[int] = []
    left = max_bins
    with open(tmp, "w", encoding="ascii") as out:
        for name, bins in read_psmcfa(src):
            if left <= 0:
                break
            bins = bins[:left]
            if not bins:
                continue
            out.write(simulate.psmcfa_record(name, bins))
            lengths.append(len(bins))
            left -= len(bins)
    tmp.replace(dst)
    return input_shape(lengths)


# ---------------------------------------------------------------------------
# Layout and jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Where E2's files are."""

    root: Path

    def subset(self, sid: str) -> Path:
        return self.root / "inputs" / f"{sid}.psmcfa"

    def c_psmc(self, sid: str) -> Path:
        return self.root / "fits" / f"{sid}.c.psmc"

    def rs_json(self, sid: str) -> Path:
        return self.root / "fits" / f"{sid}.rs.json"

    def rs_psmc(self, sid: str) -> Path:
        return self.root / "fits" / f"{sid}.rs.psmc"

    def boot_dir(self, sid: str, trunk: int) -> Path:
        return self.root / "bootstrap" / sid / f"trunk{trunk}"

    def boot_main(self, sid: str, trunk: int) -> Path:
        return self.boot_dir(sid, trunk) / "main.json"

    def boot_summary(self, sid: str, trunk: int, ext: str = "json") -> Path:
        return self.boot_dir(sid, trunk) / f"summary.{ext}"

    def boot_replicate(self, sid: str, trunk: int, i: int) -> Path:
        return self.boot_dir(sid, trunk) / f"replicate_{i:03d}.json"

    def qc_json(self, sid: str) -> Path:
        return self.root / "qc" / f"{sid}.qc.json"

    def qc_txt(self, sid: str) -> Path:
        return self.root / "qc" / f"{sid}.qc.txt"

    def record(self, kind: str, sid: str) -> Path:
        return self.root / "provenance" / kind / f"{sid}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]

    def figure(self, ext: str) -> Path:
        return self.root / "figures" / f"{FIGURE_STEM}.{ext}"


def job_id(kind: str, sid: str) -> str:
    return f"{NAME}/{kind}/{sid}"


def source(genome: config.RealGenome) -> Path:
    """The packaged psmcfa of a genome."""
    return paths.REAL_DATA / genome.file


def fit_input(lay: Layout, cfg: config.E2, genome: config.RealGenome) -> Path:
    """The psmcfa the fits read: the packaged file, or its first bins when the profile says so."""
    return source(genome) if cfg.max_bins is None else lay.subset(genome.sample_id)


def boot_kind(trunk: int) -> str:
    """The job kind (and provenance directory) of a bootstrap: ``boot-rs/trunk<B>``."""
    return f"boot-rs/trunk{trunk}"


def bootstrap_stage(cfg: config.E2, trunk: int) -> str:
    """The seed stage of a bootstrap: ``bootstrap`` at the default trunks, ``bootstrap/trunk<B>`` otherwise."""
    return "bootstrap" if trunk == cfg.bootstrap_trunk_bins else f"bootstrap/trunk{trunk}"


def bootstrap_seed(cfg: config.E2, genome: config.RealGenome, trunk: int) -> int:
    return config.seed(NAME, genome.sample_id, 0, bootstrap_stage(cfg, trunk))


def input_lengths(cfg: config.E2, genome: config.RealGenome) -> Tuple[int, ...]:
    """The bins of each sequence the fits read (the packaged file, or its first bins)."""
    lengths = psmcfa_lengths(source(genome))
    return prefix_lengths(lengths, cfg.max_bins) if cfg.max_bins is not None else lengths


def trunk_sizes(cfg: config.E2, lengths: Sequence[int]) -> List[int]:
    """The trunk sizes a genome is bootstrapped with: the default, and the fine one where the default leaves few trunks."""
    out = [cfg.bootstrap_trunk_bins]
    fine = cfg.fine_trunk_bins
    if fine and fine != cfg.bootstrap_trunk_bins and trunk_count(lengths, cfg.bootstrap_trunk_bins) < cfg.few_trunks:
        out.append(fine)
    return out


def below_trunk_fraction(lengths: Sequence[int], trunk: int) -> float:
    """The share of bins in sequences shorter than one trunk (each one resampled whole)."""
    total = sum(lengths)
    return sum(n for n in lengths if n < trunk) / total if total else 0.0


def rs_seconds(costs: config.Costs, mb: float, inference: config.Inference, parallel: int) -> float:
    """Estimated seconds of one psmc-rs fit, or of one of ``parallel`` replicates fitted at once."""
    return costs.rs02_seconds(mb, inference) / max(parallel, 1)


def bootstrap_args(cfg: config.E2, genome: config.RealGenome, trunk: int, boot_dir: Path) -> List[str]:
    return [
        "--bootstrap", str(cfg.bootstrap_replicates),
        "--bootstrap-block-size", str(trunk),
        "--bootstrap-seed", str(bootstrap_seed(cfg, genome, trunk)),
        "--bootstrap-iters", str(cfg.inference.n_iter),
        "--bootstrap-dir", str(boot_dir),
    ]


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e2
    costs = ctx.profile.costs
    lay = Layout(ctx.exp_dir(NAME))
    inference = asdict(cfg.inference)
    n_iter = cfg.inference.n_iter
    bin_mb = cfg.inference.bin_size / 1e6
    c_tool, rs_tool = ctx.c_psmc(), ctx.rs02()
    rs_ident = {rs_tool.name: rs_tool.identity}
    out: List[Job] = []
    report_inputs: List[Path] = []
    for g in cfg.genomes:
        sid = g.sample_id
        src = source(g)
        shape = input_shape(psmcfa_lengths(src), cfg.max_bins)
        mb, longest_mb = shape.bins * bin_mb, shape.longest * bin_mb
        inp = fit_input(lay, cfg, g)
        scaling = asdict(g.scaling)
        if cfg.max_bins is not None:
            out.append(
                Job(
                    id=job_id("subset", sid),
                    command=[paths.python(), "-m", MODULE, "subset", str(src), str(inp), "--max-bins", str(cfg.max_bins)],
                    inputs=[src],
                    outputs=[inp],
                    mem_gb=0.5,
                    est_seconds=2.0 + 0.5 * mb / 100,
                    settings={"max_bins": cfg.max_bins},
                    tools={"pipeline": provenance.python_identity(["experiments/e2.py", "simulate.py"])},
                    description=f"first {cfg.max_bins} bins of {g.label}",
                )
            )
        out.append(
            Job(
                id=job_id("fit-c", sid),
                command=tools.c_psmc_fit(c_tool, cfg.inference, inp, lay.c_psmc(sid)),
                inputs=[inp],
                outputs=[lay.c_psmc(sid)],
                mem_gb=round(costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * longest_mb, 2),
                est_seconds=costs.c_psmc_s_per_mb_iter * mb * n_iter,
                settings={"inference": inference},
                tools={c_tool.name: c_tool.identity},
                description=f"C psmc on {g.label}",
            )
        )
        out.append(
            Job(
                id=job_id("fit-rs", sid),
                command=tools.rs02_fit(
                    rs_tool, cfg.inference, g.scaling, inp, lay.rs_json(sid),
                    threads=cfg.fit_threads, psmc_out=lay.rs_psmc(sid),
                ),
                inputs=[inp],
                outputs=[lay.rs_json(sid), lay.rs_psmc(sid)],
                cores=cfg.fit_threads,
                mem_gb=costs.rs02_gb,
                est_seconds=costs.rs02_seconds(mb, cfg.inference, min(cfg.fit_threads, shape.sequences)),
                settings={"inference": inference, "scaling": scaling, "threads": cfg.fit_threads},
                tools=rs_ident,
                description=f"psmc-rs 0.2 on {g.label}",
            )
        )
        boot_outputs: List[Path] = []
        for trunk in trunk_sizes(cfg, input_lengths(cfg, g)):
            # psmc-rs skips a replicate whose fresh start is invalid (it gets no
            # files); the report reads exactly the replicates summary.json names.
            outputs = [lay.boot_main(sid, trunk), lay.boot_summary(sid, trunk, "json"),
                       lay.boot_summary(sid, trunk, "tsv")]
            boot_outputs += outputs
            out.append(
                Job(
                    id=job_id(boot_kind(trunk), sid),
                    command=tools.rs02_fit(
                        rs_tool, cfg.inference, g.scaling, inp, lay.boot_main(sid, trunk),
                        threads=cfg.rs_threads, extra=bootstrap_args(cfg, g, trunk, lay.boot_dir(sid, trunk)),
                    ),
                    inputs=[inp],
                    outputs=outputs,
                    cores=cfg.rs_threads,
                    mem_gb=round(costs.rs02_gb * cfg.rs_threads, 2),  # one replicate per thread at once
                    est_seconds=costs.rs02_seconds(mb, cfg.inference, min(cfg.rs_threads, shape.sequences))
                    + cfg.bootstrap_replicates * rs_seconds(costs, mb, cfg.inference, cfg.rs_threads),
                    settings={
                        "inference": inference,
                        "scaling": scaling,
                        "threads": cfg.rs_threads,
                        "bootstrap": {"replicates": cfg.bootstrap_replicates, "trunk_bins": trunk, "iterations": n_iter},
                    },
                    tools=rs_ident,
                    seeds={"bootstrap": bootstrap_seed(cfg, g, trunk)},
                    description=f"psmc-rs 0.2 bootstrap ({cfg.bootstrap_replicates} replicates, trunks of {trunk} "
                                f"bins) on {g.label}",
                )
            )
            report_inputs.append(lay.record(boot_kind(trunk), sid))
        out.append(
            Job(
                id=job_id("qc", sid),
                command=tools.rs02_qc(
                    rs_tool, inp, lay.qc_json(sid),
                    extra=["--bin-size", str(cfg.inference.bin_size), "--bootstrap-block-size", str(cfg.bootstrap_trunk_bins)],
                ),
                inputs=[inp],
                outputs=[lay.qc_json(sid)],
                stdout=lay.qc_txt(sid),
                mem_gb=costs.rs02_gb,
                est_seconds=2.0 + 0.002 * mb,
                settings={"bin_size": cfg.inference.bin_size, "bootstrap_trunk_bins": cfg.bootstrap_trunk_bins},
                tools=rs_ident,
                description=f"psmc-rs qc on {g.label}",
            )
        )
        report_inputs += [
            inp, lay.c_psmc(sid), lay.rs_json(sid), lay.rs_psmc(sid), *boot_outputs, lay.qc_json(sid),
            lay.record("fit-c", sid), lay.record("fit-rs", sid),
        ]
    out.append(
        Job(
            id=f"{NAME}/report",
            command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
            inputs=report_inputs,
            outputs=[lay.table(k) for k in TABLES],
            mem_gb=costs.report_gb,
            est_seconds=costs.report_s,
            settings={"e2": asdict(cfg)},
            tools={"pipeline": provenance.python_identity(
                ["experiments/e2.py", "experiments/e1.py", "results.py", "metrics.py", "demography.py"], ["numpy"]
            )},
            description="E2 tables",
        )
    )
    order = [sid for sid in cfg.panel_order if sid in {g.sample_id for g in cfg.genomes}]
    order += [g.sample_id for g in cfg.genomes if g.sample_id not in order]
    out.append(
        Job(
            id=f"{NAME}/figure",
            command=[
                paths.python(), "-m", FIGURE_MODULE, "--tables", str(lay.root / "tables"),
                "--out", str(lay.root / "figures" / FIGURE_STEM), "--order", ",".join(order),
                "--formats", ",".join(FIGURE_FORMATS),
            ],
            inputs=[lay.table(k) for k in FIGURE_TABLES],
            outputs=[lay.figure(ext) for ext in FIGURE_FORMATS],
            mem_gb=costs.report_gb,
            est_seconds=costs.report_s,
            settings={"order": order, "formats": list(FIGURE_FORMATS)},
            tools={"pipeline": provenance.python_identity(["figures/fig3_real_genomes.py"], ["matplotlib", "numpy"])},
            description="Figure 3",
        )
    )
    return out


# ---------------------------------------------------------------------------
# Curves, windows and bands
# ---------------------------------------------------------------------------


def fit_epochs(fit: results.Fit, mu: float, bin_size: int) -> List[demography.Epoch]:
    """A fitted PSMC model as epochs of constant size (generations, diploid individuals).

    Interval k runs from ``2 N0 t_k`` to ``2 N0 t_{k+1}`` generations ago
    (the last one without end) with size ``lambda_k N0``, so the model's
    pairwise coalescence rate is 1 / (2 Ne), as for the true models.
    """
    n0 = fit.n0(mu, bin_size)
    starts = [2.0 * n0 * t for t in fit.boundaries()]
    eps = []
    for k, lam in enumerate(fit.lambdas):
        end = starts[k + 1] if k + 1 < len(starts) else math.inf
        eps.append(demography.Epoch(starts[k], end, lam * n0, 0.0))
    return eps


def fitted_window(fit: results.Fit, scaling: config.Scaling, bin_size: int, quantiles: Tuple[float, float]) -> Tuple[float, float]:
    """The years between which ``quantiles`` of pairwise coalescence times fall under ``fit``."""
    return metrics.informative_window(fit_epochs(fit, scaling.mu, bin_size), quantiles, scaling.gen_years)


def band(curves: Sequence[results.Curve], years, probs: Sequence[float]):
    """Quantiles ``probs`` (rows) of the curves' Ne at each time of ``years`` (columns), linearly interpolated."""
    import numpy as np

    if not curves:
        raise ValueError("no curves")
    ne = np.vstack([c.at(years) for c in curves])
    return np.quantile(ne, list(probs), axis=0, method="linear")


def summary_difference(main: results.Curve, replicates: Sequence[results.Curve], summary: Dict) -> float:
    """Largest relative difference between psmc-rs's ``summary.json`` and the same summary computed here."""
    import numpy as np

    pts = summary["points"]
    xs = np.array([p["x_years"] for p in pts], dtype=float)
    mine = np.vstack([main.at(xs), band(replicates, xs, SUMMARY_PROBS)])
    theirs = np.array([[p[k] for p in pts] for k in ("ne_main", "ne_q025", "ne_q500", "ne_q975")], dtype=float)
    return float(np.max(np.abs(mine / theirs - 1.0)))


# ---------------------------------------------------------------------------
# Reading a genome's outputs
# ---------------------------------------------------------------------------


@dataclass
class Bootstrap:
    """One bootstrap run of a genome, read back and checked."""

    trunk: int
    identical: bool  # its main.json equals fit-rs's JSON byte for byte
    summary: Dict
    replicates: List[Tuple[int, Dict]]
    curves: List[results.Curve]
    run: Dict


@dataclass
class GenomeResult:
    """The outputs of one genome, read back and checked."""

    genome: config.RealGenome
    input: Path
    lengths: Tuple[int, ...]
    c: results.PsmcFile
    rs: results.PsmcFile
    rs_params: Dict
    fits: Dict[str, results.Fit]
    curves: Dict[str, results.Curve]
    boots: Dict[int, Bootstrap]  # by trunk size, the default first
    qc: Dict
    runs: Dict[str, Dict]
    window: Tuple[float, float]


def _run_record(path: Path) -> Dict:
    try:
        return json.loads(Path(path).read_text()).get("run", {})
    except (OSError, ValueError):
        return {}


def load_genome(lay: Layout, cfg: config.E2, g: config.RealGenome) -> GenomeResult:
    sid, n_iter, bin_size = g.sample_id, cfg.inference.n_iter, cfg.inference.bin_size
    c = results.parse_psmc(lay.c_psmc(sid))
    rs = results.parse_psmc(lay.rs_psmc(sid))
    rs_params = results.read_rs_json(lay.rs_json(sid))
    if c.last().round != n_iter:
        raise ValueError(f"{lay.c_psmc(sid)}: last complete round {c.last().round}, expected {n_iter}")
    if int(rs_params.get("n_iter", -1)) != n_iter or rs.last().round != n_iter:
        raise ValueError(f"{lay.rs_json(sid)}: not a fit of {n_iter} iterations")
    inp = rs_params.get("input", {})
    called = inp.get("bins", -1) - inp.get("missing_bins", 0)
    if not (c.header.get("sum_L") == called and c.header.get("sum_n") == inp.get("het_bins")):
        raise ValueError(
            f"{sid}: the tools read different inputs (C sum_L {c.header.get('sum_L')}, sum_n "
            f"{c.header.get('sum_n')}; psmc-rs {called} called bins, {inp.get('het_bins')} heterozygous)"
        )
    sc = g.scaling
    lengths = input_lengths(cfg, g)
    boots = {trunk: load_bootstrap(lay, cfg, g, trunk) for trunk in trunk_sizes(cfg, lengths)}
    fits = {"c": c.last().fit(), "rs": results.fit_from_rs(rs_params)}
    curves = {k: f.curve(sc.mu, sc.gen_years, bin_size) for k, f in fits.items()}
    runs = {kind: _run_record(lay.record(kind, sid)) for kind in ("fit-c", "fit-rs")}
    return GenomeResult(
        genome=g,
        input=fit_input(lay, cfg, g),
        lengths=lengths,
        c=c,
        rs=rs,
        rs_params=rs_params,
        fits=fits,
        curves=curves,
        boots=boots,
        qc=json.loads(lay.qc_json(sid).read_text()),
        runs=runs,
        window=fitted_window(fits["c"], sc, bin_size, cfg.grid.quantiles),
    )


def load_bootstrap(lay: Layout, cfg: config.E2, g: config.RealGenome, trunk: int) -> Bootstrap:
    """One bootstrap run: checked to be the run the configuration asks for, its replicates as the summary names them."""
    sid, sc = g.sample_id, g.scaling
    summary = json.loads(lay.boot_summary(sid, trunk).read_text())
    expect = {"block_size": trunk, "seed": bootstrap_seed(cfg, g, trunk)}
    for key, value in expect.items():
        if summary.get(key) != value:
            raise ValueError(f"{lay.boot_summary(sid, trunk)}: {key} {summary.get(key)}, expected {value}")
    skipped = set(summary.get("skipped_replicates", []))
    replicates = [
        (i, results.read_rs_json(lay.boot_replicate(sid, trunk, i)))
        for i in range(1, cfg.bootstrap_replicates + 1)
        if i not in skipped
    ]
    if len(replicates) != summary.get("n_replicates"):
        raise ValueError(f"{sid} trunk{trunk}: {len(replicates)} replicate files, summary.json counts "
                         f"{summary.get('n_replicates')}")
    return Bootstrap(
        trunk=trunk,
        identical=lay.boot_main(sid, trunk).read_bytes() == lay.rs_json(sid).read_bytes(),
        summary=summary,
        replicates=replicates,
        curves=[results.fit_from_rs(p).curve(sc.mu, sc.gen_years, cfg.inference.bin_size) for _, p in replicates],
        run=_run_record(lay.record(boot_kind(trunk), sid)),
    )


# ---------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------


def scope(qc: Dict) -> str:
    """What the input holds, in words: one named sequence, or a number of them."""
    seqs = qc.get("sequences", [])
    if len(seqs) == 1:
        return f"1 sequence ({seqs[0].get('name')} only)"
    return f"{qc.get('n_sequences', len(seqs))} sequences"


def loglikes(gr: GenomeResult) -> Dict[str, Dict[int, float]]:
    """``{tool: {iterations: log-likelihood}}``."""
    return {"c": gr.c.loglikes(), "rs": dict(enumerate(results.rs_loglikes(gr.rs_params)))}


def _run_fields(run: Dict) -> Dict:
    return {
        "wall_seconds": run.get("wall_seconds"),
        "peak_rss_mb": run.get("peak_rss_mb"),
        "user_seconds": run.get("user_seconds"),
        "loadavg_before_1m": (run.get("loadavg_before") or [None])[0],
        "run_start": run.get("start"),
    }


def fit_summary_rows(gr: GenomeResult, cfg: config.E2) -> List[Dict]:
    g, n = gr.genome, cfg.inference.n_iter
    ll = loglikes(gr)
    inp = gr.rs_params.get("input", {})
    rows = []
    for tool, kind, threads in (("c", "fit-c", 1), ("rs", "fit-rs", cfg.fit_threads)):
        fit = gr.fits[tool]
        n0 = fit.n0(g.scaling.mu, cfg.inference.bin_size)
        rows.append(
            {
                "sample_id": g.sample_id,
                "species": g.species,
                "label": g.label,
                "scope": scope(gr.qc),
                "mu": g.scaling.mu,
                "gen_years": g.scaling.gen_years,
                "scaling_source": g.scaling_source,
                "tool": tool,
                "sequences": gr.c.header.get("n_seqs") if tool == "c" else inp.get("sequences"),
                "called_bins": gr.c.header.get("sum_L") if tool == "c" else inp.get("bins", 0) - inp.get("missing_bins", 0),
                "het_bins": gr.c.header.get("sum_n") if tool == "c" else inp.get("het_bins"),
                "n_iter": n,
                "theta": fit.theta,
                "rho": fit.rho,
                "theta_rho_ratio": fit.theta / fit.rho,
                "t_max": fit.t_max,
                "n0": n0,
                "ne_recent": fit.lambdas[0] * n0,
                "ne_oldest": fit.lambdas[-1] * n0,
                "loglike_common": ll[tool].get(n - 1),
                "loglike_final": ll[tool].get(n),
                "threads": threads,
                **_run_fields(gr.runs[kind]),
            }
        )
    return rows


def pair_row(gr: GenomeResult, cfg: config.E2) -> Dict:
    import numpy as np

    n = cfg.inference.n_iter
    grid = metrics.log_grid(gr.window[0], gr.window[1], cfg.grid.points)
    fixed = metrics.fixed_grid(cfg.grid)
    ll = loglikes(gr)
    common = sorted(i for i in set(ll["c"]) & set(ll["rs"]) if ll["c"][i])
    rel = {i: abs(ll["rs"][i] - ll["c"][i]) / abs(ll["c"][i]) for i in common}
    fc, fr = gr.fits["c"], gr.fits["rs"]
    rs_c, c_c = gr.curves["rs"], gr.curves["c"]
    return {
        "sample_id": gr.genome.sample_id,
        "label": gr.genome.label,
        "window_lo_years": gr.window[0],
        "window_hi_years": gr.window[1],
        "window_points": len(grid),
        "rmse_log10_ne_rs_vs_c": metrics.rmse_between(rs_c, c_c, grid),
        "rmse_log10_ne_rs_vs_c_fixed": metrics.rmse_between(rs_c, c_c, fixed),
        "max_abs_log10_ne_diff": float(np.max(np.abs(np.log10(rs_c.at(grid)) - np.log10(c_c.at(grid))))),
        "theta_rel_diff": fr.theta / fc.theta - 1.0,
        "rho_rel_diff": fr.rho / fc.rho - 1.0,
        "t_max_rel_diff": fr.t_max / fc.t_max - 1.0,
        "loglike_iteration0_rel_diff": rel.get(0),
        "loglike_max_rel_diff": max(rel.values()) if rel else None,
        "loglike_common_diff": (ll["rs"][n - 1] - ll["c"][n - 1]) if (n - 1) in common else None,
        "bootstrap_main_identical": all(b.identical for b in gr.boots.values()),
    }


def bootstrap_row(gr: GenomeResult, cfg: config.E2, boot: Bootstrap) -> Dict:
    import numpy as np

    grid = metrics.log_grid(gr.window[0], gr.window[1], cfg.grid.points)
    lo, med, hi = band(boot.curves, grid, (cfg.band[0], 0.5, cfg.band[1]))
    width = np.log10(hi) - np.log10(lo)
    c_ne, rs_ne = gr.curves["c"].at(grid), gr.curves["rs"].at(grid)
    trunks = trunk_count(gr.lengths, boot.trunk)
    thetas = [float(p["theta"]) for _, p in boot.replicates]
    return {
        "sample_id": gr.genome.sample_id,
        "label": gr.genome.label,
        "default_trunks": boot.trunk == cfg.bootstrap_trunk_bins,
        "replicates": len(boot.replicates),
        "skipped_replicates": len(boot.summary.get("skipped_replicates", [])),
        "seed": boot.summary.get("seed"),
        "trunk_bins": boot.trunk,
        "trunk_mb": boot.trunk * cfg.inference.bin_size / 1e6,
        "trunks": trunks,
        "coarse_band": trunks < cfg.few_trunks,
        "bins_below_trunk_fraction": below_trunk_fraction(gr.lengths, boot.trunk),
        "iterations": cfg.inference.n_iter,
        "band_lo_quantile": cfg.band[0],
        "band_hi_quantile": cfg.band[1],
        "window_lo_years": gr.window[0],
        "window_hi_years": gr.window[1],
        "band_width_log10_median": float(np.median(width)),
        "band_width_log10_max": float(np.max(width)),
        "c_in_band_fraction": float(np.mean((lo <= c_ne) & (c_ne <= hi))),
        "rs_in_band_fraction": float(np.mean((lo <= rs_ne) & (rs_ne <= hi))),
        "rmse_log10_rs_vs_band_median": metrics.rmse_log10(rs_ne, med),
        "replicate_theta_min": min(thetas),
        "replicate_theta_max": max(thetas),
        "summary_max_rel_diff": summary_difference(gr.curves["rs"], boot.curves, boot.summary),
        "main_identical_to_fit": boot.identical,
        "threads": cfg.rs_threads,
        **_run_fields(boot.run),
    }


def replicate_rows(gr: GenomeResult, boot: Bootstrap) -> List[Dict]:
    return [
        {
            "sample_id": gr.genome.sample_id,
            "trunk_bins": boot.trunk,
            "replicate": i,
            "theta": float(p["theta"]),
            "rho": float(p["rho"]),
            "t_max": float(p["t_max"]),
            "n_steps": int(p["n_steps"]),
            "pattern": p.get("pattern") or "",
            "lam": " ".join(repr(float(x)) for x in p["lam"]),
        }
        for i, p in boot.replicates
    ]


def band_rows(gr: GenomeResult, cfg: config.E2, boot: Bootstrap) -> List[Dict]:
    import numpy as np

    lo_y, hi_y, npts = cfg.display_years
    years = np.logspace(math.log10(lo_y), math.log10(hi_y), int(npts))
    lo, med, hi = band(boot.curves, years, (cfg.band[0], 0.5, cfg.band[1]))
    c_ne, rs_ne = gr.curves["c"].at(years), gr.curves["rs"].at(years)
    w_lo, w_hi = gr.window
    return [
        {
            "sample_id": gr.genome.sample_id,
            "trunk_bins": boot.trunk,
            "years": float(t),
            "in_window": bool(w_lo <= t <= w_hi),
            "ne_c": float(a),
            "ne_rs": float(b),
            "ne_band_lo": float(x),
            "ne_band_median": float(m),
            "ne_band_hi": float(y),
        }
        for t, a, b, x, m, y in zip(years, c_ne, rs_ne, lo, med, hi)
    ]


def curve_rows(gr: GenomeResult) -> List[Dict]:
    rows = []
    for tool in TOOLS:
        fit, curve = gr.fits[tool], gr.curves[tool]
        for k, (t, y, lam, ne) in enumerate(zip(fit.boundaries(), curve.years, fit.lambdas, curve.ne)):
            rows.append(
                {"sample_id": gr.genome.sample_id, "tool": tool, "interval": k,
                 "t_2n0": t, "years": y, "lambda": lam, "ne": ne}
            )
    return rows


def trace_rows(gr: GenomeResult, cfg: config.E2) -> List[Dict]:
    ll = loglikes(gr)
    rows = []
    for tool, psmc in (("c", gr.c), ("rs", gr.rs)):
        by_round = {r.round: r for r in psmc.complete_rounds()}
        for i in range(cfg.inference.n_iter + 1):
            rd = by_round.get(i)
            rows.append(
                {
                    "sample_id": gr.genome.sample_id, "tool": tool, "iteration": i,
                    "loglike": ll[tool].get(i),
                    "theta": rd.theta if rd else None, "rho": rd.rho if rd else None,
                    "t_max": rd.t_max if rd else None,
                }
            )
    return rows


def _codes(findings: Sequence[Dict], warnings: bool) -> str:
    """The codes of the warnings, or of the other findings (notes; qc's JSON calls their severity 'info')."""
    return ";".join(f["code"] for f in findings if (f.get("severity") == "warning") == warnings)


def qc_row(gr: GenomeResult) -> Dict:
    q = gr.qc
    tot, lengths = q.get("totals", {}), q.get("lengths", {})
    sat, miss = q.get("saturation") or {}, q.get("longest_missing_run") or {}
    findings = q.get("findings", [])
    fit_qc = gr.rs_params.get("qc", {})
    bases = lambda x: None if x is None else x / 1e6  # noqa: E731 - Mb
    return {
        "sample_id": gr.genome.sample_id,
        "label": gr.genome.label,
        "scope": scope(q),
        "input": paths.repo_relative(gr.input),
        "input_sha256": provenance.sha256_file(gr.input),
        "sequences": q.get("n_sequences"),
        "bins": tot.get("bins"),
        "bin_size": q.get("bin_size"),
        "called_mb": bases(tot.get("called_bases_from_bins")),
        "k_fraction": tot.get("k_fraction"),
        "missing_fraction": tot.get("missing_fraction"),
        "het_bin_fraction": tot.get("het_bin_fraction"),
        "theta0_per_bin": tot.get("theta0_per_bin"),
        "het_per_kb": tot.get("het_per_kb"),
        "n50_mb": bases(lengths.get("n50_bases")),
        "longest_mb": bases(lengths.get("longest_bases")),
        "bins_below_1mb_fraction": (lengths.get("below_1mb") or {}).get("fraction"),
        "bootstrap_trunk_bins": q.get("bootstrap_trunk_bins"),
        "bootstrap_trunks": lengths.get("bootstrap_trunks"),
        "bins_below_trunk_fraction": (lengths.get("below_trunk") or {}).get("fraction"),
        "longest_missing_run_kb": None if miss.get("bases") is None else miss["bases"] / 1e3,
        "multi_het_share": sat.get("multi_het_share"),
        "max_bin_size_below_target": sat.get("max_bin_size_below_target"),
        "expected_recombinations": q.get("expected_recombinations"),
        "warnings": _codes(findings, warnings=True),
        "notes": _codes(findings, warnings=False),
        "fit_warnings": ";".join(fit_qc.get("warnings", [])),
        "fit_notes": ";".join(fit_qc.get("notes", [])),
    }


def qc_finding_rows(gr: GenomeResult) -> List[Dict]:
    return [
        {
            "sample_id": gr.genome.sample_id,
            "code": f.get("code"),
            "severity": f.get("severity"),
            "message": f.get("message"),
            "recommendation": f.get("recommendation"),
            "sequences": len(f.get("sequences") or []),
        }
        for f in gr.qc.get("findings", [])
    ]


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E2 table from the outputs; returns the paths written."""
    cfg = ctx.profile.e2
    lay = Layout(ctx.exp_dir(NAME))
    rows: Dict[str, List[Dict]] = {k: [] for k in TABLES}
    for g in cfg.genomes:
        gr = load_genome(lay, cfg, g)
        rows["fit_summary"] += fit_summary_rows(gr, cfg)
        rows["pairs"].append(pair_row(gr, cfg))
        for boot in gr.boots.values():
            rows["bootstrap"].append(bootstrap_row(gr, cfg, boot))
            rows["replicates"] += replicate_rows(gr, boot)
            rows["bands"] += band_rows(gr, cfg, boot)
        rows["curves"] += curve_rows(gr)
        rows["traces"] += trace_rows(gr, cfg)
        rows["qc"].append(qc_row(gr))
        rows["qc_findings"] += qc_finding_rows(gr)
    written = {}
    for key, table in rows.items():
        write_csv(lay.table(key), table)
        written[key] = lay.table(key)
    return written


# ---------------------------------------------------------------------------
# Command line (the subset job)
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="Helpers of experiment E2's jobs.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("subset", help="write the first bins of a psmcfa (smoke and tiny profiles)")
    s.add_argument("src", type=Path)
    s.add_argument("dst", type=Path)
    s.add_argument("--max-bins", type=int, required=True)
    args = p.parse_args(argv)
    shape = write_psmcfa_prefix(args.src, args.dst, args.max_bins)
    print(f"{args.dst}: {shape.sequences} sequences, {shape.bins} bins (longest {shape.longest})")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Bootstrap", "GenomeResult", "Layout", "NAME", "Shape", "TABLES", "band", "below_trunk_fraction", "boot_kind",
    "bootstrap_seed", "fit_epochs", "fitted_window", "input_lengths", "input_shape", "jobs", "load_genome",
    "prefix_lengths", "psmcfa_lengths", "read_psmcfa", "report", "summary_difference", "trunk_sizes",
    "write_psmcfa_prefix",
]
