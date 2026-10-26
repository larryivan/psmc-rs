"""E2: real genomes (Fig 3).

For every genome of :class:`.config.E2` (three turtle assemblies and the
human NA12878's chr1-22, E7's psmcfa of that sample) the runner fits the
packaged psmcfa (``experiment/real_data``, plain or gzip-compressed; both
tools read either) with C psmc and with psmc-rs 0.2 (the identical file
and settings), fits psmc-rs's C-compatible bootstrap, runs ``psmc-rs qc``,
and then writes the tables (Figure 3 is drawn from them by
``figures/fig3.py``, job ``FIG/fig3_real_genomes``). Layout under
``<runs>/E2/``::

    fits/<sample>.c.psmc                          C psmc, every round
    fits/<sample>.rs.json, <sample>.rs.psmc       psmc-rs 0.2, one thread (JSON and .psmc)
    fits/<sample>.<split>.rs.{json,psmc}          psmc-rs 0.2 with the first window split (the turtles)
    bootstrap/<sample>/trunk<B>/main.json         psmc-rs 0.2 with --bootstrap: the main fit, again
    bootstrap/<sample>/trunk<B>/replicate_NNN.json  the replicates' parameters, and summary.{json,tsv}
    bootstrap/<sample>/trunk<B>/replicate_NNN.psmc  the replicates' rounds (not committed)
    qc/<sample>.qc.json, <sample>.qc.txt          psmc-rs qc: JSON and the text report
    inputs/<sample>.psmcfa                        smoke and tiny profiles only: the first bins (not committed)
    provenance/<kind>/<sample>.json               one record per job
    tables/E2_*.csv                               written by ``report E2``
    figures/fig3_real_genomes.{pdf,svg,png}       Figure 3 (experiment FIG)
    logs/                                         runner and job logs (not committed)

Jobs per genome: ``subset`` when the profile fits only the first bins of
each genome, ``fit-c`` (C psmc), ``fit-rs`` (psmc-rs 0.2), for the turtles
``fit-rs-split2`` and ``fit-rs-split1`` (the split-first-window check),
``boot-rs/trunk<B>`` (psmc-rs 0.2 with ``--bootstrap``, at the default
trunks and, where they are few, at the fine trunks) and ``qc`` (``psmc-rs
qc --json``, with the default trunk size); then ``report``.

**Settings.** Both tools fit with the psmc README's human recipe
(``-N25 -t15 -r5 -p "4+25*2+4+6"``; :class:`.config.Inference`). Curves are
scaled per genome (:class:`.config.RealGenome`, whose ``scaling_source``
gives the citations): the turtles with mu 4.6e-9 per bp per generation, the
mean germline rate of nine *Chrysemys picta* trios (Bergeron et al. 2023; a
cryptodire's rate, since no pleurodire rate has been measured), and g 15
years for all three; the human with mu 1.25e-8 (Scally & Durbin 2012) and g 29
years (Fenner 2005). psmc-rs is given mu and g (``--mu``, ``--gen-years``)
for its own report; its fitted parameters do not depend on them, and C
psmc is given neither.

**Split-first-window check.** Hilgers et al. 2025 (Curr Biol 35:927) found
false recent peaks of Ne across many PSMC studies, in all their turtles
among them, under the pattern ``4+25*2+4+6``, whose first free lambda spans
the first four intervals; splitting that window removed them. psmc-rs fits
each genome of ``split_genomes`` (the turtles) again with each pattern of
``split_patterns`` (``fit-rs-split2``: ``2+2+25*2+4+6``; ``fit-rs-split1``:
``1+1+1+1+25*2+4+6``; every other setting fit-rs's), and
``E2_split_patterns.csv`` compares each with fit-rs's fit: the Ne of the
recent intervals (the first ``split_recent_intervals``) against the plateau
(the median Ne of the next as many), whether the curve has a recent peak
(:func:`recent_shape`), and the difference from fit-rs's fit interval by
interval in the recent intervals and after them.

**Bootstrap.** ``boot-rs`` runs ``psmc-rs IN main.json N_ITER --bootstrap
B --bootstrap-block-size T --bootstrap-seed S``: psmc-rs fits the main model
again, then B replicates, each drawing splitfa trunks of T bins (default
500,000, splitfa's 50 Mb) with replacement and fitted from a fresh start, as
``splitfa | psmc -b`` does. S is ``config.seed("E2", sample_id, 0,
"bootstrap")``, but NA12878's bootstrap is E7's run of the same psmcfa
(``RealGenome.bootstrap_shared_with``): E7's seed, ``seed("E7", "NA12878",
0, "bootstrap")``, with E7's replicates, trunks and iterations (E2 refuses
to plan the whole file with other settings): one seed and one set of
settings, so both bootstraps write the same files. Its band is drawn from
E2's tables (``fig3_real_genomes`` a, the article's Fig 1c); no figure draws
E7's.
psmc-rs's output does not depend on ``--threads``, so
``main.json`` must hold ``fit-rs``'s fit exactly: every field of
:data:`MAIN_FIT_FIELDS` equal. The report writes its tables and then fails
when a main fit differs. A genome whose default trunks are fewer than
``few_trunks`` is bootstrapped a second time with trunks of
``fine_trunk_bins`` (5 Mb, a size E5 calibrates), seed stage
``bootstrap/trunk<B>``. No genome of the production run is (NA12878 forms
59 default trunks, the turtles 51 to 1,220); the smoke and tiny profiles'
prefixes are. E5 calibrates bootstraps of chromosome-scale sequences cut
into trunks, as NA12878's are (its 22 chromosomes cut at 50 Mb; only chr21,
1.6% of its bins, is shorter than a trunk); the turtles' default bootstraps
resample whole scaffolds of very different lengths
(``bins_below_trunk_fraction`` of ``E2_bootstrap.csv``: 50%, 23% and 97% of
their bins lie in scaffolds shorter than a trunk), which E5 does not
calibrate. The job declares ``main.json`` and the summary only: psmc-rs
skips a replicate whose fresh start is invalid, and the report reads
exactly the replicates the summary names. The replicates' parameters
(``replicate_NNN.json``) are committed and tabulated in
``tables/E2_bootstrap_replicates.csv``; their ``.psmc`` files (every EM
round, about 100 kB each) are not.

**Comparison window.** A real genome has no true Ne(t), so the informative
window of :class:`.config.TimeGrid` is taken from the fitted model: the 1%
and 99% quantiles of the pairwise coalescence time under C psmc's final
model, in the genome's years. Outside it less than 1% of the genome's
coalescences fall, and both tools only extend their first and last lambda.
The tools are compared interval by interval on it
(:func:`.metrics.interval_diff`: log10(Ne_rs,k / Ne_c,k) over C psmc's
intervals that overlap the window, the RMSE weighted by each interval's
log-time width inside it), since their boundaries differ in years only by
a hair (theta, t_max) and fixed times between two such boundaries would read
whole steps; the fixed window (10 ky to 1 My) is reported next to it. The
bootstrap band is compared with the curves at 200 log-spaced times in the
window.

**Band.** At each time, the 2.5% and 97.5% quantiles (linear interpolation,
as psmc-rs computes them) of the replicates' Ne. Each replicate is read as
a step function in its own years (its own N0). The report recomputes
psmc-rs's ``summary.json`` from the replicate JSONs as a check of both.

Tables (``tables/``):

* ``E2_fit_summary.csv``: per genome and tool: scaling (mu, g and its
  source), sequences, called and heterozygous bins, theta, rho, t_max, N0,
  log-likelihoods, wall time and peak RSS.
* ``E2_rs_vs_c.csv``: per genome: the tools compared interval by interval
  (``*_per_interval``: RMSE of log10 Ne on the window and on the fixed
  window, the largest difference of an interval overlapping the window;
  ``*_any_interval``: of any interval), relative differences of theta, rho
  and t_max and of the log-likelihoods, and whether the bootstrap runs'
  main fits equal ``fit-rs``'s fit.
* ``E2_bootstrap.csv``: per genome and trunk size: replicates, trunks
  (``coarse_band`` when fewer than psmc-rs's 20), seed, the band's width,
  how much of C psmc's and psmc-rs's curves it holds, the check against
  psmc-rs's own summary, and whether the run's main fit is ``fit-rs``'s
  (the fields that differ, and by how much).
* ``E2_bootstrap_replicates.csv``: the parameters of every replicate.
* ``E2_bands.csv``: per genome and trunk size, both curves and the band on
  a dense grid (Fig 3).
* ``E2_curves.csv``: the fitted step curves of both tools (Fig 3).
* ``E2_traces.csv``: log-likelihood, theta, rho and t_max per EM iteration.
* ``E2_qc.csv`` and ``E2_qc_findings.csv``: the metrics and the findings
  of ``psmc-rs qc`` for each input E2 fitted (in production the packaged
  files, whose reports Table S5 shows: E6's ``E6_S5_*``).
* ``E2_split_patterns.csv``: per turtle and pattern (``default`` is
  fit-rs's fit): theta, rho, t_max, N0, the final log-likelihood and its
  gain over the default (the split patterns have 1 and 3 more free
  lambdas), the recent intervals' Ne against the plateau
  (``recent_max_vs_plateau``, ``first_vs_plateau``, ``recent_peak``), and
  the largest difference from fit-rs's fit in the recent intervals and the
  RMSE after them, on the comparison window (log10 Ne, interval by
  interval).
* ``E2_split_curves.csv``: the step curves of those fits (figure
  ``figS_turtle_patterns``).

Run ``python -m experiment.pipeline.experiments.e2 subset SRC DST
--max-bins N`` to write the first N bins of a psmcfa, plain or gzip, to a
plain psmcfa (the ``subset`` job).
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import config, demography, metrics, paths, provenance, results, simulate, tools
from ..runner import Job
from . import Context
from .e1 import (check_same_input, compare_tools, loglike_diffs, loglikes, read_fits, read_rs_fit, run_fields,
                 run_record, step_rows, trace_rows, write_tables)
from .e5 import main_fit_difference, trunk_count

NAME = "E2"
MODULE = "experiment.pipeline.experiments.e2"
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
    "split": "E2_split_patterns.csv",
    "split_curves": "E2_split_curves.csv",
}
SUMMARY_PROBS = config.PSMC_RS_SUMMARY_QUANTILES  # the quantiles of psmc-rs's summary.json
# The fields of a psmc-rs JSON that are the fit. Its ``qc`` record describes
# the input as the run cut it: a bootstrap run adds SHORT_SEQUENCES where
# sequences are shorter than its trunks, so it is not compared.
MAIN_FIT_FIELDS = ("theta", "rho", "t_max", "lam", "loglike", "final_loglike", "n_iter", "n_steps", "pattern")


# ---------------------------------------------------------------------------
# psmcfa inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """What the cost estimates need of an input: sequences, bins and the longest sequence."""

    sequences: int
    bins: int
    longest: int


GZIP_MAGIC = b"\x1f\x8b"


def is_gzip(path: Path) -> bool:
    """Whether a file is gzip-compressed, told by its first two bytes as psmc-rs tells it (not by its name)."""
    with open(path, "rb") as f:
        return f.read(2) == GZIP_MAGIC


def read_psmcfa(path: Path) -> Iterator[Tuple[str, str]]:
    """``(header, bins)`` of each record of a psmcfa file, plain or gzip, in order (header without '>')."""
    name: Optional[str] = None
    chunks: List[str] = []
    opened = gzip.open(path, "rt", encoding="ascii") if is_gzip(path) else open(path, encoding="ascii")
    with opened as f:
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
    if data[:2] == GZIP_MAGIC:
        data = gzip.decompress(data)  # every member: E7 writes one per chromosome
    out = []
    for chunk in data.split(b">")[1:]:
        _, _, body = chunk.partition(b"\n")
        out.append(len(body.translate(None, b" \t\r\n")))
    return tuple(out)


def psmcfa_lengths(path: Path) -> Tuple[int, ...]:
    """The bins of each record of a psmcfa file, plain or gzip (cached by size and modification time)."""
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
    """Write the first ``max_bins`` bins of ``src`` (plain or gzip) to the plain ``dst``, as fq2psmcfa lays them out.

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
class BootFiles:
    """The files of one ``psmc-rs --bootstrap`` run (E7's too): ``main.json`` (its OUT) and the ``--bootstrap-dir``."""

    dir: Path

    @property
    def main(self) -> Path:
        return self.dir / "main.json"

    def summary(self, ext: str = "json") -> Path:
        return self.dir / f"summary.{ext}"

    def replicate(self, i: int) -> Path:
        return self.dir / f"replicate_{i:03d}.json"


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

    def split_json(self, sid: str, name: str) -> Path:
        return self.root / "fits" / f"{sid}.{name}.rs.json"

    def split_psmc(self, sid: str, name: str) -> Path:
        return self.root / "fits" / f"{sid}.{name}.rs.psmc"

    def boot(self, sid: str, trunk: int) -> BootFiles:
        return BootFiles(self.root / "bootstrap" / sid / f"trunk{trunk}")

    def qc_json(self, sid: str) -> Path:
        return self.root / "qc" / f"{sid}.qc.json"

    def qc_txt(self, sid: str) -> Path:
        return self.root / "qc" / f"{sid}.qc.txt"

    def record(self, kind: str, sid: str) -> Path:
        return self.root / "provenance" / kind / f"{sid}.json"

    def table(self, key: str) -> Path:
        return self.root / "tables" / TABLES[key]


def job_id(kind: str, sid: str) -> str:
    return f"{NAME}/{kind}/{sid}"


def source(genome: config.RealGenome) -> Path:
    """The packaged psmcfa of a genome."""
    return paths.REAL_DATA / genome.file


def fit_input(lay: Layout, cfg: config.E2, genome: config.RealGenome) -> Path:
    """The psmcfa the fits read: the packaged file, or its first bins when the profile says so."""
    return source(genome) if cfg.max_bins is None else lay.subset(genome.sample_id)


def split_kind(name: str) -> str:
    """The job kind (and provenance directory) of a split-first-window fit: ``fit-rs-<name>``."""
    return f"fit-rs-{name}"


def boot_kind(trunk: int) -> str:
    """The job kind (and provenance directory) of a bootstrap: ``boot-rs/trunk<B>``."""
    return f"boot-rs/trunk{trunk}"


def bootstrap_seed(cfg: config.E2, genome: config.RealGenome, trunk: int) -> int:
    """The seed of a bootstrap: stage ``bootstrap`` at the default trunks, ``bootstrap/trunk<B>`` otherwise.

    At the default trunks it is the seed of the experiment the genome shares
    its bootstrap with, if any.
    """
    if trunk == cfg.bootstrap_trunk_bins:
        return config.seed(genome.bootstrap_shared_with or NAME, genome.sample_id, 0, "bootstrap")
    return config.seed(NAME, genome.sample_id, 0, f"bootstrap/trunk{trunk}")


def shared_bootstrap_differences(profile: config.Profile, genome: config.RealGenome) -> Tuple[str, ...]:
    """The settings in which E2's default bootstrap of a genome and that of the experiment it shares it with differ.

    Replicates, trunks and inference (the replicates' iterations among it);
    not threads, on which psmc-rs's output does not depend. Empty when the
    genome shares no bootstrap.
    """
    other = genome.bootstrap_shared_with
    if other is None:
        return ()
    mine, theirs = profile.e2, getattr(profile, other.lower())
    pairs = {
        "replicates": (mine.bootstrap_replicates, theirs.bootstrap_replicates),
        "trunk_bins": (mine.bootstrap_trunk_bins, theirs.bootstrap_trunk_bins),
        "inference": (mine.inference, theirs.inference),
    }
    return tuple(name for name, (a, b) in pairs.items() if a != b)


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


def bootstrap_args(cfg, trunk: int, seed: int, boot_dir: Path) -> List[str]:
    """psmc-rs's options of a bootstrap run of ``cfg``'s replicates and iterations (E2's or E7's: their shared
    bootstrap of NA12878 is one command)."""
    return [
        "--bootstrap", str(cfg.bootstrap_replicates),
        "--bootstrap-block-size", str(trunk),
        "--bootstrap-seed", str(seed),
        "--bootstrap-iters", str(cfg.inference.n_iter),
        "--bootstrap-dir", str(boot_dir),
    ]


def genome_jobs(ctx: Context, lay: Layout, g: config.RealGenome) -> Tuple[List[Job], List[Path]]:
    """The jobs of one genome, and the files of theirs the report reads."""
    cfg, costs = ctx.profile.e2, ctx.profile.costs
    sid = g.sample_id
    differing = shared_bootstrap_differences(ctx.profile, g) if cfg.max_bins is None else ()
    if differing:
        raise ValueError(f"E2 bootstraps {sid} with {g.bootstrap_shared_with}'s seed, but with other "
                         f"{', '.join(differing)}: the two bands would differ")
    src, inp = source(g), fit_input(lay, cfg, g)
    lengths = input_lengths(cfg, g)
    shape = input_shape(lengths)
    bin_mb = cfg.inference.bin_size / 1e6
    mb, longest_mb = shape.bins * bin_mb, shape.longest * bin_mb
    n_iter = cfg.inference.n_iter
    c_tool, rs_tool = ctx.c_psmc(), ctx.rs02()
    rs_ident = {rs_tool.name: rs_tool.identity}
    scaling = asdict(g.scaling)

    def rs_fit(kind: str, inference: config.Inference, out_json: Path, psmc_out: Path, description: str) -> Job:
        return Job(
            id=job_id(kind, sid),
            command=tools.rs02_fit(rs_tool, inference, g.scaling, inp, out_json, threads=cfg.fit_threads,
                                   psmc_out=psmc_out),
            inputs=[inp],
            outputs=[out_json, psmc_out],
            cores=cfg.fit_threads,
            mem_gb=costs.rs02_gb,
            est_seconds=costs.rs02_seconds(mb, inference, min(cfg.fit_threads, shape.sequences)),
            settings={"inference": asdict(inference), "scaling": scaling, "threads": cfg.fit_threads},
            tools=rs_ident,
            description=description,
        )

    out: List[Job] = []
    if cfg.max_bins is not None:
        out.append(Job(
            id=job_id("subset", sid),
            command=[paths.python(), "-m", MODULE, "subset", str(src), str(inp), "--max-bins", str(cfg.max_bins)],
            inputs=[src],
            outputs=[inp],
            mem_gb=0.5,
            est_seconds=2.0 + 0.5 * mb / 100,
            settings={"max_bins": cfg.max_bins},
            tools={"pipeline": provenance.python_identity(["experiments/e2.py", "simulate.py"])},
            description=f"first {cfg.max_bins} bins of {g.label}",
        ))
    out.append(Job(
        id=job_id("fit-c", sid),
        command=tools.c_psmc_fit(c_tool, cfg.inference, inp, lay.c_psmc(sid)),
        inputs=[inp],
        outputs=[lay.c_psmc(sid)],
        mem_gb=round(costs.c_psmc_gb_base + costs.c_psmc_gb_per_mb_longest * longest_mb, 2),
        est_seconds=costs.c_psmc_s_per_mb_iter * mb * n_iter,
        settings={"inference": asdict(cfg.inference)},
        tools={c_tool.name: c_tool.identity},
        description=f"C psmc on {g.label}",
    ))
    out.append(rs_fit("fit-rs", cfg.inference, lay.rs_json(sid), lay.rs_psmc(sid), f"psmc-rs 0.2 on {g.label}"))
    report_inputs: List[Path] = []
    if sid in cfg.split_genomes:
        for name, pattern in cfg.split_patterns:
            out.append(rs_fit(split_kind(name), replace(cfg.inference, pattern=pattern), lay.split_json(sid, name),
                              lay.split_psmc(sid, name), f"psmc-rs 0.2 on {g.label}, pattern {pattern} (first window split)"))
            report_inputs += [lay.split_json(sid, name), lay.split_psmc(sid, name), lay.record(split_kind(name), sid)]
    boot_outputs: List[Path] = []
    for trunk in trunk_sizes(cfg, lengths):
        # psmc-rs skips a replicate whose fresh start is invalid (it gets no
        # files); the report reads exactly the replicates summary.json names.
        boot, seed = lay.boot(sid, trunk), bootstrap_seed(cfg, g, trunk)
        outputs = [boot.main, boot.summary("json"), boot.summary("tsv")]
        boot_outputs += outputs
        out.append(Job(
            id=job_id(boot_kind(trunk), sid),
            command=tools.rs02_fit(rs_tool, cfg.inference, g.scaling, inp, boot.main, threads=cfg.rs_threads,
                                   extra=bootstrap_args(cfg, trunk, seed, boot.dir)),
            inputs=[inp],
            outputs=outputs,
            cores=cfg.rs_threads,
            mem_gb=round(costs.rs02_gb * cfg.rs_threads, 2),  # one replicate per thread at once
            # the main fit, then the replicates, rs_threads of them at once
            est_seconds=costs.rs02_seconds(mb, cfg.inference, min(cfg.rs_threads, shape.sequences))
            + cfg.bootstrap_replicates * (costs.rs02_seconds(mb, cfg.inference) / max(cfg.rs_threads, 1)),
            settings={
                "inference": asdict(cfg.inference),
                "scaling": scaling,
                "threads": cfg.rs_threads,
                "bootstrap": {"replicates": cfg.bootstrap_replicates, "trunk_bins": trunk, "iterations": n_iter},
            },
            tools=rs_ident,
            seeds={"bootstrap": seed},
            description=f"psmc-rs 0.2 bootstrap ({cfg.bootstrap_replicates} replicates, trunks of {trunk} "
                        f"bins) on {g.label}",
        ))
        report_inputs.append(lay.record(boot_kind(trunk), sid))
    out.append(Job(
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
    ))
    report_inputs += [
        inp, lay.c_psmc(sid), lay.rs_json(sid), lay.rs_psmc(sid), *boot_outputs, lay.qc_json(sid),
        lay.record("fit-c", sid), lay.record("fit-rs", sid),
    ]
    return out, report_inputs


def jobs(ctx: Context) -> List[Job]:
    cfg = ctx.profile.e2
    lay = Layout(ctx.exp_dir(NAME))
    out: List[Job] = []
    report_inputs: List[Path] = []
    for g in cfg.genomes:
        genome_out, genome_inputs = genome_jobs(ctx, lay, g)
        out += genome_out
        report_inputs += genome_inputs
    out.append(Job(
        id=f"{NAME}/report",
        command=ctx.pipeline_cmd("report", NAME, *ctx.profile_args()),
        inputs=report_inputs,
        outputs=[lay.table(k) for k in TABLES],
        mem_gb=ctx.profile.costs.report_gb,
        est_seconds=ctx.profile.costs.report_s,
        settings={"e2": asdict(cfg)},
        tools={"pipeline": provenance.python_identity(
            ["experiments/e2.py", "experiments/e1.py", "experiments/e5.py", "results.py", "metrics.py",
             "demography.py"], ["numpy"],
        )},
        description="E2 tables",
    ))
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
    ends = starts[1:] + [math.inf]
    return [demography.Epoch(s, e, lam * n0, 0.0) for s, e, lam in zip(starts, ends, fit.lambdas)]


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


def band_rows(ids: Dict, years, window: Tuple[float, float], boot: Bootstrap, band_quantiles: Tuple[float, float],
              c: Optional[results.Curve], rs: results.Curve) -> List[Dict]:
    """Both curves and a bootstrap's band at each time of ``years``, ``ids`` first (``ne_c`` empty without ``c``)."""
    lo, med, hi = band(boot.curves, years, (band_quantiles[0], 0.5, band_quantiles[1]))
    c_ne = [None] * len(years) if c is None else c.at(years)
    return [
        {
            **ids,
            "years": float(t),
            "in_window": bool(window[0] <= t <= window[1]),
            "ne_c": None if a is None else float(a),
            "ne_rs": float(b),
            "ne_band_lo": float(x),
            "ne_band_median": float(m),
            "ne_band_hi": float(y),
        }
        for t, a, b, x, m, y in zip(years, c_ne, rs.at(years), lo, med, hi)
    ]


# psmc-rs's summary times are the main curve's interval starts, computed by
# psmc-rs; the same starts computed here can differ in the last place (the
# platforms' exp), which puts a start a hair after its time and reads the
# previous step. The main curve is read this far (relative) after each time.
SUMMARY_MAIN_NUDGE = 1e-9


def summary_difference(main: results.Curve, replicates: Sequence[results.Curve], summary: Dict) -> float:
    """Largest relative difference between psmc-rs's ``summary.json`` and the same summary computed here."""
    import numpy as np

    pts = summary["points"]
    xs = np.array([p["x_years"] for p in pts], dtype=float)
    mine = np.vstack([main.at(xs * (1.0 + SUMMARY_MAIN_NUDGE)), band(replicates, xs, SUMMARY_PROBS)])
    theirs = np.array([[p[k] for p in pts] for k in ("ne_main", "ne_q025", "ne_q500", "ne_q975")], dtype=float)
    return float(np.max(np.abs(mine / theirs - 1.0)))


# ---------------------------------------------------------------------------
# Reading a genome's outputs
# ---------------------------------------------------------------------------


@dataclass
class Bootstrap:
    """One bootstrap run of a genome (E2's or E7's), read back and checked."""

    trunk: int
    differing: Tuple[str, ...]  # the MAIN_FIT_FIELDS in which its main.json differs from the fit's JSON
    max_abs_diff: float  # the largest difference of a number of those fields (0: the same fit)
    summary: Dict
    replicates: List[Tuple[int, Dict]]
    curves: List[results.Curve]
    run: Dict

    @property
    def identical(self) -> bool:
        """Its main fit is the fit, field for field."""
        return not self.differing

    @property
    def skipped(self) -> List[int]:
        """The replicates psmc-rs skipped (their fresh start was invalid)."""
        return self.summary.get("skipped_replicates", [])


def main_fit_mismatch(main: Dict, fit: Dict) -> Tuple[str, ...]:
    """The :data:`MAIN_FIT_FIELDS` in which two psmc-rs JSONs differ (empty: the same fit)."""
    return tuple(k for k in MAIN_FIT_FIELDS if main.get(k) != fit.get(k))


@dataclass
class GenomeResult:
    """The outputs of one genome, read back and checked."""

    genome: config.RealGenome
    input: Path
    lengths: Tuple[int, ...]
    c: results.PsmcFile
    rs: results.PsmcFile
    rs_params: Dict
    loglikes: Dict[str, Dict[int, float]]  # by tool, by iterations
    fits: Dict[str, results.Fit]
    curves: Dict[str, results.Curve]
    boots: Dict[int, Bootstrap]  # by trunk size, the default first
    qc: Dict
    runs: Dict[str, Dict]
    window: Tuple[float, float]


def load_genome(lay: Layout, cfg: config.E2, g: config.RealGenome) -> GenomeResult:
    sid, bin_size = g.sample_id, cfg.inference.bin_size
    c, rs, rs_params = read_fits(lay.c_psmc(sid), lay.rs_psmc(sid), lay.rs_json(sid), cfg.inference.n_iter)
    check_same_input(c, rs_params, sid)
    sc = g.scaling
    lengths = input_lengths(cfg, g)
    boots = {trunk: load_bootstrap(lay.boot(sid, trunk), cfg, trunk, bootstrap_seed(cfg, g, trunk), sc, rs_params,
                                   lay.record(boot_kind(trunk), sid))
             for trunk in trunk_sizes(cfg, lengths)}
    fits = {"c": c.last().fit(), "rs": results.fit_from_rs(rs_params)}
    return GenomeResult(
        genome=g,
        input=fit_input(lay, cfg, g),
        lengths=lengths,
        c=c,
        rs=rs,
        rs_params=rs_params,
        loglikes=loglikes(c, rs_params),
        fits=fits,
        curves={k: f.curve(sc.mu, sc.gen_years, bin_size) for k, f in fits.items()},
        boots=boots,
        qc=json.loads(lay.qc_json(sid).read_text()),
        runs={kind: run_record(lay.record(kind, sid)) for kind in ("fit-c", "fit-rs")},
        window=fitted_window(fits["c"], sc, bin_size, cfg.grid.quantiles),
    )


def load_bootstrap(files: BootFiles, cfg, trunk: int, seed: int, scaling: config.Scaling, fit: Dict,
                   record: Path) -> Bootstrap:
    """A bootstrap run, checked to be of these trunks and seed, with the replicates its summary names.

    ``cfg`` (:class:`.config.E2` or :class:`.config.E7`) gives the replicates asked for and the bin size; ``fit`` is
    the JSON of the fit the run's main fit must equal, and ``record`` the run's provenance record.
    """
    summary = json.loads(files.summary().read_text())
    for key, value in {"block_size": trunk, "seed": seed}.items():
        if summary.get(key) != value:
            raise ValueError(f"{files.summary()}: {key} {summary.get(key)}, expected {value}")
    skipped = set(summary.get("skipped_replicates", []))
    replicates = [(i, results.read_rs_json(files.replicate(i)))
                  for i in range(1, cfg.bootstrap_replicates + 1) if i not in skipped]
    if len(replicates) != summary.get("n_replicates"):
        raise ValueError(f"{files.dir}: {len(replicates)} replicate files, summary.json counts "
                         f"{summary.get('n_replicates')}")
    main = results.read_rs_json(files.main)
    return Bootstrap(
        trunk=trunk,
        differing=main_fit_mismatch(main, fit),
        max_abs_diff=main_fit_difference(main, fit),
        summary=summary,
        replicates=replicates,
        curves=[results.fit_from_rs(p).curve(scaling.mu, scaling.gen_years, cfg.inference.bin_size)
                for _, p in replicates],
        run=run_record(record),
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


def fit_summary_rows(gr: GenomeResult, cfg: config.E2) -> List[Dict]:
    g, n = gr.genome, cfg.inference.n_iter
    hdr, inp = gr.c.header, gr.rs_params.get("input", {})
    # what each tool read: sequences, called bins, heterozygous bins
    counts = {"c": (hdr.get("n_seqs"), hdr.get("sum_L"), hdr.get("sum_n")),
              "rs": (inp.get("sequences"), inp.get("bins", 0) - inp.get("missing_bins", 0), inp.get("het_bins"))}
    rows = []
    for tool, kind, threads in (("c", "fit-c", 1), ("rs", "fit-rs", cfg.fit_threads)):
        fit = gr.fits[tool]
        n0 = fit.n0(g.scaling.mu, cfg.inference.bin_size)
        sequences, called, het = counts[tool]
        rows.append({
            "sample_id": g.sample_id,
            "species": g.species,
            "label": g.label,
            "scope": scope(gr.qc),
            "mu": g.scaling.mu,
            "gen_years": g.scaling.gen_years,
            "scaling_source": g.scaling_source,
            "scaling_cite": g.scaling_cite,
            "tool": tool,
            "sequences": sequences,
            "called_bins": called,
            "het_bins": het,
            "n_iter": n,
            "theta": fit.theta,
            "rho": fit.rho,
            "theta_rho_ratio": fit.theta / fit.rho,
            "t_max": fit.t_max,
            "n0": n0,
            "ne_recent": fit.lambdas[0] * n0,
            "ne_oldest": fit.lambdas[-1] * n0,
            "loglike_common": gr.loglikes[tool].get(n - 1),
            "loglike_final": gr.loglikes[tool].get(n),
            "threads": threads,
            **run_fields(gr.runs[kind]),
        })
    return rows


def pair_row(gr: GenomeResult, cfg: config.E2) -> Dict:
    ll = loglike_diffs(gr.loglikes, cfg.inference.n_iter)
    return {
        "sample_id": gr.genome.sample_id,
        "label": gr.genome.label,
        **compare_tools(gr.fits, gr.curves, gr.window, cfg.grid.fixed_years),
        "loglike_iteration0_rel_diff": ll["loglike_iteration0_rel_diff"],
        "loglike_max_rel_diff": ll["loglike_max_rel_diff"],
        "loglike_common_diff": ll["loglike_common_diff"],
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
        "skipped_replicates": len(boot.skipped),
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
        "main_fit_max_abs_diff": boot.max_abs_diff,
        "main_fit_fields_differing": ";".join(boot.differing),
        "threads": cfg.rs_threads,
        **run_fields(boot.run),
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


# ---------------------------------------------------------------------------
# The split-first-window check
# ---------------------------------------------------------------------------


@dataclass
class SplitFit:
    """One psmc-rs fit of a genome with a pattern: fit-rs's (``default``) or a split-first-window one."""

    name: str
    pattern: str
    params: Dict
    fit: results.Fit
    curve: results.Curve
    run: Dict


def load_split(lay: Layout, cfg: config.E2, gr: GenomeResult, name: str, pattern: str) -> SplitFit:
    """A split-first-window fit, checked to be of its pattern and of fit-rs's input."""
    g, n_iter = gr.genome, cfg.inference.n_iter
    path = lay.split_json(g.sample_id, name)
    params = read_rs_fit(path, n_iter)
    if params.get("pattern") != pattern:
        raise ValueError(f"{path}: pattern {params.get('pattern')}, expected {pattern}")
    keys = ("sequences", "bins", "missing_bins", "het_bins")
    mine, main = params.get("input", {}), gr.rs_params.get("input", {})
    if any(mine.get(k) != main.get(k) for k in keys):
        raise ValueError(f"{path}: another input than fit-rs's ({', '.join(f'{k} {mine.get(k)}' for k in keys)})")
    fit = results.fit_from_rs(params)
    sc = g.scaling
    return SplitFit(name, pattern, params, fit, fit.curve(sc.mu, sc.gen_years, cfg.inference.bin_size),
                    run_record(lay.record(split_kind(name), g.sample_id)))


def split_fits(lay: Layout, cfg: config.E2, gr: GenomeResult) -> List[SplitFit]:
    """fit-rs's fit (``default``), then the genome's split-first-window fits in the configuration's order."""
    default = SplitFit("default", cfg.inference.pattern, gr.rs_params, gr.fits["rs"], gr.curves["rs"],
                       gr.runs["fit-rs"])
    return [default] + [load_split(lay, cfg, gr, name, pattern) for name, pattern in cfg.split_patterns]


def recent_shape(curve: results.Curve, recent: int, peak_ratio: float) -> Dict:
    """The recent intervals of a curve against the plateau before them (``config.E2.split_recent_intervals``).

    The plateau is the median Ne of the ``recent`` intervals after the
    recent ones. A recent peak: the largest Ne of the recent intervals lies
    after the first interval and is at least ``peak_ratio`` times both the
    first interval's Ne and the plateau.
    """
    ne = curve.ne
    if len(ne) < 2 * recent:
        raise ValueError(f"{len(ne)} intervals, fewer than twice the {recent} recent ones")
    k = max(range(recent), key=lambda i: ne[i])  # the first of equal maxima
    plateau = statistics.median(ne[recent:2 * recent])
    return {
        "recent_end_years": curve.years[recent],
        "ne_first": ne[0],
        "recent_max_interval": k,
        "recent_max_years": curve.years[k],
        "ne_recent_max": ne[k],
        "ne_plateau": plateau,
        "first_vs_plateau": ne[0] / plateau,
        "recent_max_vs_plateau": ne[k] / plateau,
        "recent_max_vs_first": ne[k] / ne[0],
        "recent_peak": k > 0 and ne[k] >= peak_ratio * ne[0] and ne[k] >= peak_ratio * plateau,
    }


def split_rows(gr: GenomeResult, cfg: config.E2, fits: Sequence[SplitFit]) -> List[Dict]:
    """One row per fit of the genome: its parameters, its recent intervals, and how it differs from fit-rs's fit."""
    g, n, recent = gr.genome, cfg.inference.n_iter, cfg.split_recent_intervals
    default = fits[0]
    ll_default = results.rs_loglikes(default.params)
    older = (default.curve.years[recent], gr.window[1])  # fit-rs's intervals after the recent ones, in the window
    rows = []
    for sf in fits:
        ratio = metrics.interval_log10_ratio(sf.curve, default.curve)
        ll = results.rs_loglikes(sf.params)
        rows.append({
            "sample_id": g.sample_id,
            "label": g.label,
            "mu": g.scaling.mu,
            "gen_years": g.scaling.gen_years,
            "pattern_name": sf.name,
            "pattern": sf.pattern,
            "free_lambdas": results.pattern_free(sf.pattern),
            "n_iter": n,
            "theta": sf.fit.theta,
            "rho": sf.fit.rho,
            "t_max": sf.fit.t_max,
            "n0": sf.fit.n0(g.scaling.mu, cfg.inference.bin_size),
            "loglike_final": ll[n] if len(ll) > n else None,
            "loglike_gain_vs_default": ll[n] - ll_default[n] if len(ll) > n and len(ll_default) > n else None,
            "window_lo_years": gr.window[0],
            "window_hi_years": gr.window[1],
            "recent_intervals": recent,
            **recent_shape(sf.curve, recent, cfg.split_peak_ratio),
            "max_abs_log10_ne_vs_default_recent": float(max(abs(x) for x in ratio[:recent])),
            "rmse_log10_ne_vs_default_older": (
                metrics.interval_diff(sf.curve, default.curve, older).rmse if older[0] < older[1] else None
            ),
            **run_fields(sf.run),
        })
    return rows


def report(ctx: Context) -> Dict[str, Path]:
    """Write every E2 table from the outputs; returns the paths written."""
    cfg = ctx.profile.e2
    n = cfg.inference.n_iter
    lay = Layout(ctx.exp_dir(NAME))
    years = metrics.log_grid(*cfg.display_years)
    rows: Dict[str, List[Dict]] = {k: [] for k in TABLES}
    for g in cfg.genomes:
        gr = load_genome(lay, cfg, g)
        sid = g.sample_id
        rows["fit_summary"] += fit_summary_rows(gr, cfg)
        rows["pairs"].append(pair_row(gr, cfg))
        for boot in gr.boots.values():
            rows["bootstrap"].append(bootstrap_row(gr, cfg, boot))
            rows["replicates"] += replicate_rows(gr, boot)
            rows["bands"] += band_rows({"sample_id": sid, "trunk_bins": boot.trunk}, years, gr.window, boot, cfg.band,
                                       gr.curves["c"], gr.curves["rs"])
        for tool, psmc in (("c", gr.c), ("rs", gr.rs)):
            rows["curves"] += step_rows(gr.fits[tool], gr.curves[tool], sample_id=sid, tool=tool)
            rows["traces"] += trace_rows(psmc, gr.loglikes[tool], n, sample_id=sid, tool=tool)
        rows["qc"].append(qc_row(gr))
        rows["qc_findings"] += qc_finding_rows(gr)
        if sid in cfg.split_genomes:
            fits = split_fits(lay, cfg, gr)
            rows["split"] += split_rows(gr, cfg, fits)
            for sf in fits:
                rows["split_curves"] += step_rows(sf.fit, sf.curve, sample_id=sid, pattern_name=sf.name,
                                                  pattern=sf.pattern)
    written = write_tables(lay.table, rows)
    differ = [r for r in rows["bootstrap"] if not r["main_identical_to_fit"]]
    if differ:
        raise ValueError(
            "the main fit of a bootstrap run differs from fit-rs's fit (psmc-rs's results must not depend on "
            "--threads): " + "; ".join(f"{r['sample_id']} trunk{r['trunk_bins']}: {r['main_fit_fields_differing']} "
                                       f"(largest difference {r['main_fit_max_abs_diff']:.3g})" for r in differ)
        )
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
    "BootFiles", "Bootstrap", "GenomeResult", "Layout", "NAME", "Shape", "SplitFit", "TABLES", "band", "band_rows",
    "below_trunk_fraction", "boot_kind", "bootstrap_args", "bootstrap_seed", "fit_epochs", "fitted_window",
    "input_lengths", "input_shape", "is_gzip", "jobs", "load_bootstrap", "load_genome", "load_split",
    "main_fit_mismatch", "prefix_lengths", "psmcfa_lengths", "read_psmcfa", "recent_shape", "report",
    "shared_bootstrap_differences", "split_fits", "split_kind", "split_rows", "summary_difference", "trunk_sizes",
    "write_psmcfa_prefix",
]
