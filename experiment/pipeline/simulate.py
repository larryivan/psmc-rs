"""One simulated diploid genome: psmcfa, VCF, truth record and TMRCA track.

Each chromosome is simulated on its own (chromosomes are unlinked) with
msprime: ``sim_ancestry`` (Hudson coalescent, discrete genome) for one
diploid individual, then ``sim_mutations`` (JC69, discrete genome). The
seeds come from :func:`.config.seed` with stages ``ancestry/<chrom>`` and
``mutations/<chrom>``. Outputs of :func:`simulate_genome`:

* ``genome.psmcfa``: one record per chromosome, as fq2psmcfa writes it
  (``>chr1``, then 60 bins per line): ``ceil(length / bin_size)`` bins, ``K``
  when the bin holds a heterozygous site, else ``T``; nothing is missing.
* ``genome.vcf.gz`` (bgzip) and its tabix index: ``##contig`` lines with the
  chromosome lengths and one record per site where the diploid carries a
  derived allele (heterozygous, or homozygous for a derived allele), with
  the phased GT of the one sample; POS is msprime's 0-based position + 1.
* the truth record (JSON): settings, seeds, software versions, per-
  chromosome counts and checksums, and the true Ne(t) of the model.

:func:`write_tmrca` gives the true pairwise TMRCA along the genome on
demand: it repeats the simulation from the recorded seeds, checks that the
heterozygous sites are those of the truth record, and writes run-length
intervals ``chrom start end tmrca_generations`` (0-based, end exclusive).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from . import config, demography, provenance

PSMCFA_NAME = "genome.psmcfa"
VCF_NAME = "genome.vcf.gz"
SAMPLE_NAME = "diploid"
LINE_WIDTH = 60  # fq2psmcfa's psmcfa line length


@dataclass
class Chromosome:
    """The diploid's variation on one simulated chromosome."""

    name: str
    length: int
    seeds: Dict[str, int]
    het_positions: "object"  # numpy int64 array, 0-based, sorted
    records: List[Tuple[int, str, str, str]]  # (0-based pos, REF, ALT, GT) of every site with a derived allele
    trees: int
    ts: "object" = None  # the tree sequence, kept only when asked for


def seeds_for(experiment: str, model: str, replicate: int, chrom: str) -> Dict[str, int]:
    """The ancestry and mutation seeds of one chromosome (see :func:`.config.seed`)."""
    return {
        "ancestry": config.seed(experiment, model, replicate, f"ancestry/{chrom}"),
        "mutations": config.seed(experiment, model, replicate, f"mutations/{chrom}"),
    }


def simulate_chromosome(
    model: config.DemographicModel,
    sim: config.Simulation,
    name: str,
    length: int,
    seeds: Dict[str, int],
    keep_ts: bool = False,
) -> Chromosome:
    """Simulate one chromosome of one diploid individual."""
    import msprime
    import numpy as np

    if sim.ploidy != 2:
        raise ValueError("PSMC needs one diploid individual (ploidy 2)")
    if sim.mutation_model.lower() != "jc69":
        raise ValueError(f"unsupported mutation model {sim.mutation_model!r}")
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(1, population=demography.POPULATION, ploidy=sim.ploidy)],
        demography=demography.to_msprime(model),
        sequence_length=length,
        recombination_rate=sim.recombination_rate,
        discrete_genome=sim.discrete_genome,
        model=sim.ancestry_model,
        random_seed=seeds["ancestry"],
    )
    mts = msprime.sim_mutations(
        ts,
        rate=sim.mutation_rate,
        model=msprime.JC69(),
        discrete_genome=sim.discrete_genome,
        random_seed=seeds["mutations"],
    )
    het: List[int] = []
    records: List[Tuple[int, str, str, str]] = []
    for var in mts.variants():
        ref = var.alleles[0]
        pair = (var.alleles[int(var.genotypes[0])], var.alleles[int(var.genotypes[1])])
        alts = list(dict.fromkeys(al for al in pair if al != ref))
        if not alts:
            continue  # the diploid carries the ancestral state (also after a back mutation)
        index = {al: i for i, al in enumerate([ref, *alts])}
        ga, gb = index[pair[0]], index[pair[1]]
        pos = int(var.site.position)
        if ga != gb:
            het.append(pos)
        records.append((pos, ref, ",".join(alts), f"{ga}|{gb}"))
    return Chromosome(
        name=name,
        length=int(length),
        seeds=dict(seeds),
        het_positions=np.asarray(het, dtype=np.int64),
        records=records,
        trees=mts.num_trees,
        ts=mts if keep_ts else None,
    )


def n_bins(length: int, bin_size: int) -> int:
    """fq2psmcfa's number of bins: a last partial bin counts."""
    return (int(length) + bin_size - 1) // bin_size


def psmcfa_bins(chrom: Chromosome, bin_size: int) -> str:
    """The chromosome's bins: ``K`` where a heterozygous site falls, else ``T``."""
    import numpy as np

    seq = np.full(n_bins(chrom.length, bin_size), ord("T"), dtype=np.uint8)
    if len(chrom.het_positions):
        seq[chrom.het_positions // bin_size] = ord("K")
    return seq.tobytes().decode("ascii")


def psmcfa_record(name: str, bins: str) -> str:
    """One psmcfa record laid out as fq2psmcfa writes it."""
    lines = [f">{name}"]
    lines.extend(bins[i : i + LINE_WIDTH] for i in range(0, len(bins), LINE_WIDTH))
    return "\n".join(lines) + "\n"


def vcf_header(chroms: Sequence[Chromosome], source: str) -> str:
    lines = [
        "##fileformat=VCFv4.2",
        f"##source={source}",
        *(f"##contig=<ID={c.name},length={c.length}>" for c in chroms),
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype of the simulated diploid">',
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", SAMPLE_NAME]),
    ]
    return "\n".join(lines) + "\n"


def vcf_lines(chrom: Chromosome) -> Iterator[str]:
    for pos, ref, alt, gt in chrom.records:
        yield f"{chrom.name}\t{pos + 1}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{gt}\n"


def bgzip_and_index(vcf: Path) -> Path:
    """``bgzip`` ``vcf`` in place (to ``vcf.gz``) and index it with ``tabix -p vcf``."""
    gz = vcf.with_name(vcf.name + ".gz")
    subprocess.run(["bgzip", "-f", str(vcf)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", str(gz)], check=True)
    return gz


def positions_digest(positions) -> str:
    """SHA-256 of the heterozygous positions (0-based, little-endian int64)."""
    import numpy as np

    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def tool_version(cmd: Sequence[str]) -> str:
    try:
        res = subprocess.run(list(cmd), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    out = res.stdout or res.stderr
    return out.splitlines()[0].strip() if out else "unknown"


def software_versions() -> Dict[str, str]:
    """Versions of everything that shapes a simulation."""
    import msprime
    import numpy
    import tskit

    return {
        "python": platform.python_version(),
        "msprime": msprime.__version__,
        "tskit": tskit.__version__,
        "numpy": numpy.__version__,
        "bgzip": tool_version(["bgzip", "--version"]),
        "tabix": tool_version(["tabix", "--version"]),
    }


def longest_het_gap(positions, length: int) -> Tuple[int, int]:
    """``(bases, start)`` of the longest stretch of the chromosome without a heterozygous site.

    Stretches run between consecutive heterozygous sites (exclusive) and
    from either end of the chromosome to its nearest one; with no
    heterozygous site the whole chromosome is one stretch.
    """
    import numpy as np

    p = np.asarray(positions, dtype=np.int64)
    bounds = np.concatenate([[-1], p, [int(length)]])
    gaps = np.diff(bounds) - 1
    k = int(np.argmax(gaps))
    return int(gaps[k]), int(bounds[k] + 1)


def longest_hetfree_bins(positions, length: int, bin_size: int) -> int:
    """The longest run of consecutive bins without a heterozygous site (``T`` bins of the psmcfa)."""
    import numpy as np

    het = np.unique(np.asarray(positions, dtype=np.int64) // bin_size)
    bounds = np.concatenate([[-1], het, [n_bins(length, bin_size)]])
    return int(np.max(np.diff(bounds) - 1))


def chromosome_summary(chrom: Chromosome, bin_size: int) -> Dict:
    """Counts and checksums of one simulated chromosome, for its truth record.

    The longest stretch without a heterozygous site flags a long segment of
    very recent coalescence (a genuine, if rare, coalescent outcome) that
    can make a replicate an outlier; it is recorded so that such replicates
    can be explained without dropping seeds.
    """
    import numpy as np

    het = chrom.het_positions
    gap, gap_start = longest_het_gap(het, chrom.length)
    return {
        "name": chrom.name,
        "length": chrom.length,
        "bins": n_bins(chrom.length, bin_size),
        "het_sites": len(het),
        "het_bins": len(np.unique(het // bin_size)),
        "derived_hom_sites": len(chrom.records) - len(het),
        "trees": chrom.trees,
        "last_het_position": int(het[-1]) if len(het) else None,
        "longest_het_gap_bp": gap,
        "longest_het_gap_start": gap_start,
        "longest_hetfree_bins": longest_hetfree_bins(het, chrom.length, bin_size),
        "het_positions_sha256": positions_digest(het),
        "seeds": chrom.seeds,
    }


def simulate_genome(
    experiment: str,
    model: config.DemographicModel,
    replicate: int,
    genome: config.Genome,
    sim: config.Simulation,
    out_dir: Path,
    truth_path: Path,
    *,
    write_vcf: bool = True,
    tmrca_path: Optional[Path] = None,
    log=print,
) -> Dict:
    """Simulate one diploid genome and write its psmcfa, VCF and truth record.

    Returns the truth record. Outputs are written under temporary names and
    renamed when complete.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psmcfa = out_dir / PSMCFA_NAME
    vcf_plain = out_dir / "genome.vcf"
    vcf_tmp = out_dir / "genome.vcf.partial"
    psmcfa_tmp = out_dir / (PSMCFA_NAME + ".partial")

    chroms: List[Chromosome] = []
    with open(psmcfa_tmp, "w", encoding="ascii", newline="\n") as fa:
        for name in genome.names():
            seeds = seeds_for(experiment, model.name, replicate, name)
            chrom = simulate_chromosome(
                model, sim, name, genome.chromosome_length, seeds, keep_ts=tmrca_path is not None
            )
            fa.write(psmcfa_record(name, psmcfa_bins(chrom, sim.bin_size)))
            log(
                f"[simulate] {experiment} {model.name} rep{replicate} {name}: "
                f"{len(chrom.het_positions)} heterozygous sites, {chrom.trees} trees"
            )
            chroms.append(chrom)
    os.replace(psmcfa_tmp, psmcfa)

    versions = software_versions()
    files = {"psmcfa": {"name": PSMCFA_NAME, "sha256": provenance.sha256_file(psmcfa)}}
    if write_vcf:
        source = f"psmc-rs experiment pipeline; msprime {versions['msprime']}, tskit {versions['tskit']}"
        with open(vcf_tmp, "w", encoding="ascii", newline="\n") as f:
            f.write(vcf_header(chroms, source))
            for chrom in chroms:
                f.writelines(vcf_lines(chrom))
        os.replace(vcf_tmp, vcf_plain)
        gz = bgzip_and_index(vcf_plain)
        files["vcf"] = {"name": gz.name, "sha256": provenance.sha256_file(gz)}

    if tmrca_path is not None:
        write_tmrca_track(chroms, Path(tmrca_path))

    record = {
        "experiment": experiment,
        "model": model.name,
        "replicate": replicate,
        "genome": asdict(genome),
        "simulation": asdict(sim),
        "seed_scheme": "config.seed(experiment, model, replicate, stage); stages ancestry/<chrom>, mutations/<chrom>",
        "software": versions,
        "chromosomes": [chromosome_summary(c, sim.bin_size) for c in chroms],
        "totals": {
            "length": sum(c.length for c in chroms),
            "het_sites": int(sum(len(c.het_positions) for c in chroms)),
        },
        "files": files,
        "truth": demography.truth_record(model),
    }
    if model.name == config.ZIGZAG.name:
        record["stdpopsim_check"] = _zigzag_check()
    write_json(Path(truth_path), record)
    return record


def _zigzag_check() -> Dict:
    try:
        return {"passed": True, **demography.check_zigzag_against_stdpopsim()}
    except ImportError as exc:
        return {"passed": None, "reason": f"stdpopsim not importable: {exc}"}


def write_json(path: Path, obj: Dict) -> None:
    """Write ``obj`` as indented JSON, keys in their order, through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# True TMRCA
# ---------------------------------------------------------------------------


def tmrca_intervals(ts) -> List[Tuple[int, int, float]]:
    """Run-length intervals ``(start, end, tmrca)`` of the two samples' TMRCA (generations).

    Adjacent trees with the same TMRCA are merged; coordinates are 0-based
    and the end is exclusive.
    """
    samples = list(ts.samples())
    if len(samples) != 2:
        raise ValueError(f"expected two sampled haplotypes, got {len(samples)}")
    a, b = samples
    out: List[Tuple[int, int, float]] = []
    for tree in ts.trees():
        left, right = int(tree.interval.left), int(tree.interval.right)
        t = float(tree.tmrca(a, b))
        if out and out[-1][2] == t and out[-1][1] == left:
            out[-1] = (out[-1][0], right, t)
        else:
            out.append((left, right, t))
    return out


def write_tmrca_track(chroms: Sequence[Chromosome], path: Path) -> None:
    """Write the TMRCA intervals of ``chroms`` (which kept their tree sequences) as TSV.gz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with gzip.open(tmp, "wt", encoding="ascii", newline="\n") as f:
        f.write("chrom\tstart\tend\ttmrca_generations\n")
        for chrom in chroms:
            if chrom.ts is None:
                raise ValueError(f"{chrom.name}: the tree sequence was not kept")
            for start, end, t in tmrca_intervals(chrom.ts):
                f.write(f"{chrom.name}\t{start}\t{end}\t{t!r}\n")
    os.replace(tmp, path)


def write_tmrca(truth_path: Path, out_path: Path, log=print) -> Dict:
    """Recompute a simulated genome from its truth record and write its TMRCA track.

    Every chromosome is simulated again from the recorded seeds; its
    heterozygous positions must have the recorded checksum, so the track
    belongs to the genome that was analysed.
    """
    record = json.loads(Path(truth_path).read_text())
    model = config.MODELS[record["model"]]
    sim = config.Simulation(**record["simulation"])
    chroms: List[Chromosome] = []
    for summary in record["chromosomes"]:
        chrom = simulate_chromosome(model, sim, summary["name"], summary["length"], summary["seeds"], keep_ts=True)
        digest = positions_digest(chrom.het_positions)
        if digest != summary["het_positions_sha256"]:
            raise RuntimeError(
                f"{summary['name']}: re-simulated heterozygous sites differ from the truth record "
                f"(software now {software_versions()}, recorded {record['software']})"
            )
        log(f"[tmrca] {record['model']} rep{record['replicate']} {summary['name']}: {chrom.trees} trees")
        chroms.append(chrom)
    write_tmrca_track(chroms, Path(out_path))
    return {"chromosomes": len(chroms), "path": str(out_path)}


def read_tmrca_track(path: Path) -> Dict[str, List[Tuple[int, int, float]]]:
    """The intervals written by :func:`write_tmrca_track`, per chromosome."""
    out: Dict[str, List[Tuple[int, int, float]]] = {}
    with gzip.open(path, "rt", encoding="ascii") as f:
        next(f)
        for line in f:
            chrom, start, end, t = line.rstrip("\n").split("\t")
            out.setdefault(chrom, []).append((int(start), int(end), float(t)))
    return out


def which_or_fail(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is not on PATH (it comes with htslib)")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m experiment.pipeline.simulate``: see :func:`build_parser`."""
    args = build_parser().parse_args(argv)
    profile = config.get_profile(args.profile)
    exp_cfg = getattr(profile, args.experiment.lower())
    model = config.MODELS[args.model]
    if args.tmrca_only:
        if not args.tmrca_out:
            raise SystemExit("--tmrca-only needs --tmrca-out")
        write_tmrca(Path(args.truth), Path(args.tmrca_out))
        return 0
    if not args.out_dir:
        raise SystemExit("--out-dir is required")
    if not args.no_vcf:
        which_or_fail("bgzip")
        which_or_fail("tabix")
    simulate_genome(
        args.experiment,
        model,
        args.replicate,
        exp_cfg.genome,
        exp_cfg.simulation,
        Path(args.out_dir),
        Path(args.truth),
        write_vcf=not args.no_vcf,
        tmrca_path=Path(args.tmrca_out) if args.tmrca_out else None,
    )
    return 0


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m experiment.pipeline.simulate",
        description="Simulate one diploid genome of an experiment (settings from the profile in config.py).",
    )
    p.add_argument("--profile", default="production")
    p.add_argument("--experiment", required=True, help="experiment whose genome and simulation settings apply (E1, E4, E5, ...)")
    p.add_argument("--model", required=True, choices=sorted(config.MODELS))
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--out-dir", help="directory of genome.psmcfa and genome.vcf.gz")
    p.add_argument("--truth", required=True, help="the truth record (JSON) to write, or to read with --tmrca-only")
    p.add_argument("--no-vcf", action="store_true", help="write the psmcfa only")
    p.add_argument("--tmrca-out", help="also write the true TMRCA track (TSV.gz)")
    p.add_argument("--tmrca-only", action="store_true", help="recompute the TMRCA track from --truth only")
    return p


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Chromosome", "PSMCFA_NAME", "VCF_NAME", "bgzip_and_index", "longest_het_gap", "longest_hetfree_bins", "n_bins",
    "positions_digest", "psmcfa_bins", "psmcfa_record", "read_tmrca_track", "seeds_for", "simulate_chromosome",
    "simulate_genome", "tmrca_intervals", "vcf_header", "vcf_lines", "write_json", "write_tmrca",
]
