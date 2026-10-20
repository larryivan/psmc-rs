"""Input preparation for E4: callable masks, masked VCFs, consensus FASTA and bins.

E4 asks how the path from a VCF and a callable mask to PSMC's binned input
changes the result. This module holds the data side of it; the jobs and
tables are in :mod:`.experiments.e4`.

**Callable masks.** A mask is an alternating renewal process along each
chromosome: a callable stretch, an uncallable run, a callable stretch, and so
on, starting with a callable stretch at position 0. Uncallable runs have a
geometric length (the discrete exponential, at least 1 bp) with mean
``mean_run_bp``; callable stretches have a geometric length with mean
``mean_run_bp * (1 - f) / f``, so that the expected missing fraction is ``f``.
The callable sites are written as BED (``chrom start end``, 0-based,
end-exclusive), the complement of the runs. The no-missing control is one
interval per chromosome.

**Masked VCF.** A caller does not call a site it cannot call, so a record at
an uncallable position is removed from the simulated VCF; everything else is
kept byte for byte.

**Consensus FASTA.** The classic route writes one base per site: the base of a
called homozygous site, the IUPAC code of a heterozygous one, and ``N`` where
the site is not callable (what ``vcfutils.pl vcf2fq`` writes for a called
genome, without qualities). Sites without a VCF record carry an ancestral base
drawn uniformly from ACGT (JC69's stationary distribution) with a seeded
generator; which base it is does not change any bin. The FASTA is streamed
into ``fq2psmcfa`` and never stored (it would take 500 MB per genome and
mask).

**Bins.** :func:`read_psmcfa` reads a psmcfa as fq2psmcfa writes it, and
:func:`compare_bins` tabulates, chromosome by chromosome, how the bins of a
route differ from those of the exact route (``T``, ``K``, ``N``, and bins a
route does not have at all, such as a multihetsep's tail after its last
heterozygous site).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import config, provenance, simulate

FASTA_LINE_WIDTH = 60
SYMBOLS = ("T", "K", "N")  # the bins fq2psmcfa writes: called, heterozygous, missing
IUPAC = {
    frozenset("AC"): "M", frozenset("AG"): "R", frozenset("AT"): "W",
    frozenset("CG"): "S", frozenset("CT"): "Y", frozenset("GT"): "K",
}


# ---------------------------------------------------------------------------
# Callable masks
# ---------------------------------------------------------------------------


@dataclass
class Intervals:
    """Sorted, disjoint, non-adjacent half-open intervals ``[starts[i], ends[i])`` of one chromosome."""

    starts: "object"  # numpy int64 array
    ends: "object"

    @classmethod
    def empty(cls) -> "Intervals":
        import numpy as np

        return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.starts)

    def total(self) -> int:
        return int((self.ends - self.starts).sum()) if len(self) else 0

    def contains(self, positions):
        """Whether each 0-based position of ``positions`` is inside an interval."""
        import numpy as np

        positions = np.asarray(positions, dtype=np.int64)
        if not len(self):
            return np.zeros(len(positions), dtype=bool)
        idx = np.searchsorted(self.starts, positions, side="right") - 1
        ok = idx >= 0
        ok[ok] = positions[ok] < self.ends[idx[ok]]
        return ok

    def indicator(self, length: int):
        """A boolean array of ``length``: True inside an interval."""
        import numpy as np

        diff = np.zeros(length + 1, dtype=np.int8)
        np.add.at(diff, np.minimum(self.starts, length), 1)
        np.add.at(diff, np.minimum(self.ends, length), -1)
        return np.cumsum(diff[:length], dtype=np.int8) > 0


def uncallable_runs(length: int, missing_fraction: float, mean_run_bp: float, rng) -> Intervals:
    """The uncallable runs of one chromosome (see the module documentation).

    ``rng`` is a ``numpy.random.Generator``. The process is drawn in blocks of
    cycles until it passes ``length``; the last run is cut at ``length``.
    """
    import numpy as np

    if not 0.0 < missing_fraction < 1.0:
        raise ValueError(f"missing fraction must be in (0, 1), got {missing_fraction}")
    if mean_run_bp < 1.0:
        raise ValueError(f"mean run length must be >= 1 bp, got {mean_run_bp}")
    mean_gap = mean_run_bp * (1.0 - missing_fraction) / missing_fraction
    p_run, p_gap = 1.0 / mean_run_bp, min(1.0 / mean_gap, 1.0)
    cycle = mean_run_bp + mean_gap
    starts: List = []
    ends: List = []
    pos = 0
    while pos < length:
        n = int((length - pos) / cycle * 1.1) + 16
        gaps = rng.geometric(p_gap, size=n).astype(np.int64)
        runs = rng.geometric(p_run, size=n).astype(np.int64)
        bounds = np.cumsum(np.column_stack([gaps, runs]).ravel()) + pos
        s, e = bounds[0::2], bounds[1::2]
        keep = s < length
        starts.append(s[keep])
        ends.append(np.minimum(e[keep], length))
        pos = int(bounds[-1])
    return Intervals(np.concatenate(starts), np.concatenate(ends)) if starts else Intervals.empty()


def complement(runs: Intervals, length: int) -> Intervals:
    """The intervals of ``[0, length)`` outside ``runs``."""
    import numpy as np

    starts = np.concatenate([[0], runs.ends]).astype(np.int64)
    ends = np.concatenate([runs.starts, [length]]).astype(np.int64)
    keep = ends > starts
    return Intervals(starts[keep], ends[keep])


def mask_seed(model: str, replicate: int, mask: str, chrom: str, experiment: str = "E4") -> int:
    """The seed of one chromosome's mask: stage ``mask/<mask>/<chrom>`` of :func:`.config.seed`."""
    return config.seed(experiment, model, replicate, f"mask/{mask}/{chrom}")


def reference_seed(model: str, replicate: int, chrom: str, experiment: str = "E4") -> int:
    """The seed of a chromosome's ancestral bases in the consensus FASTA: stage ``reference/<chrom>``."""
    return config.seed(experiment, model, replicate, f"reference/{chrom}")


def callable_mask(
    spec: Optional[config.MaskSpec], chromosomes: Sequence[Tuple[str, int]], model: str, replicate: int,
    experiment: str = "E4",
) -> Dict[str, Intervals]:
    """The callable intervals of every chromosome; ``spec`` None is the no-missing control."""
    import numpy as np

    out: Dict[str, Intervals] = {}
    for name, length in chromosomes:
        if spec is None:
            out[name] = Intervals(np.array([0], dtype=np.int64), np.array([length], dtype=np.int64))
            continue
        rng = np.random.default_rng(mask_seed(model, replicate, spec.name, name, experiment))
        runs = uncallable_runs(length, spec.missing_fraction, spec.mean_run_bp, rng)
        out[name] = complement(runs, length)
    return out


def gzip_writer(raw: IO[bytes]) -> gzip.GzipFile:
    """A gzip stream whose bytes depend on the content only (no time or file name in its header)."""
    return gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0)


def write_bed(path: Path, mask: Dict[str, Intervals]) -> None:
    """Write ``mask`` as BED (gzip when ``path`` ends in .gz), chromosomes in the given order."""
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as raw:
        f = gzip_writer(raw) if path.suffix == ".gz" else raw
        for name, iv in mask.items():
            if not len(iv):
                continue
            rows = (f"{name}\t{s}\t{e}\n" for s, e in zip(iv.starts.tolist(), iv.ends.tolist()))
            f.write("".join(rows).encode("ascii"))
        if f is not raw:
            f.close()
    os.replace(tmp, path)


def merge_intervals(starts, ends) -> Intervals:
    """Sorted, merged intervals from any half-open ones (overlapping or adjacent ones join)."""
    import numpy as np

    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    keep = e > s
    s, e = s[keep], e[keep]
    if not len(s):
        return Intervals(s, e)
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    reach = np.maximum.accumulate(e)
    new = np.ones(len(s), dtype=bool)
    new[1:] = s[1:] > reach[:-1]
    first = np.flatnonzero(new)
    last = np.append(first[1:] - 1, len(s) - 1)
    return Intervals(s[first], reach[last])


def read_bed(path: Path) -> Dict[str, Intervals]:
    """A BED of 3+ columns as merged intervals per chromosome (in file order).

    Rows go into typed arrays as they are read, so a mask of millions of
    rows takes 16 bytes per row.
    """
    from array import array

    rows: Dict[str, Tuple[array, array]] = {}
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            chrom, start, end = line.split(None, 3)[:3]
            if chrom not in rows:
                rows[chrom] = (array("q"), array("q"))
            rows[chrom][0].append(int(start))
            rows[chrom][1].append(int(end))
    return {chrom: merge_intervals(s, e) for chrom, (s, e) in rows.items()}


def mask_summary(mask: Dict[str, Intervals], lengths: Dict[str, int]) -> Dict:
    """Callable and missing sites and the uncallable runs of a mask, per chromosome and in total."""
    chroms = []
    for name, iv in mask.items():
        length = lengths[name]
        called = iv.total()
        runs = complement(iv, length)
        chroms.append(
            {
                "name": name,
                "length": length,
                "callable_sites": called,
                "missing_sites": length - called,
                "missing_fraction": (length - called) / length,
                "uncallable_runs": len(runs),
                "mean_uncallable_run_bp": (runs.total() / len(runs)) if len(runs) else 0.0,
                "callable_intervals": len(iv),
            }
        )
    total = sum(c["length"] for c in chroms)
    missing = sum(c["missing_sites"] for c in chroms)
    n_runs = sum(c["uncallable_runs"] for c in chroms)
    return {
        "chromosomes": chroms,
        "totals": {
            "length": total,
            "missing_sites": missing,
            "missing_fraction": missing / total if total else 0.0,
            "uncallable_runs": n_runs,
            "mean_uncallable_run_bp": missing / n_runs if n_runs else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------


@dataclass
class VcfRecord:
    chrom: str
    pos: int  # 0-based
    ref: str
    alts: Tuple[str, ...]
    gt: Tuple[int, int]
    line: str  # the record as written, with its newline

    @property
    def is_het(self) -> bool:
        return self.gt[0] != self.gt[1]

    def alleles(self) -> Tuple[str, str]:
        alleles = (self.ref, *self.alts)
        return alleles[self.gt[0]], alleles[self.gt[1]]


@dataclass
class Vcf:
    header: List[str]  # every header line, with its newline
    records: List[VcfRecord] = field(default_factory=list)

    def contigs(self) -> List[Tuple[str, int]]:
        """``(name, length)`` of the ``##contig`` lines, in order."""
        matches = (re.match(r"##contig=<ID=([^,>]+),length=(\d+)", line) for line in self.header)
        return [(m.group(1), int(m.group(2))) for m in matches if m]

    def by_chrom(self) -> Dict[str, List[VcfRecord]]:
        """The records of each chromosome, in file order."""
        out: Dict[str, List[VcfRecord]] = {}
        for rec in self.records:
            out.setdefault(rec.chrom, []).append(rec)
        return out


def read_vcf(path: Path) -> Vcf:
    """A one-sample VCF of SNPs with a GT field first (as :mod:`.simulate` writes it)."""
    vcf = Vcf(header=[])
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="ascii", newline="") as f:
        for line in f:
            if line.startswith("#"):
                vcf.header.append(line)
                continue
            fields = line.rstrip("\n").split("\t")
            gt = fields[9].split(":", 1)[0].replace("|", "/").split("/")
            if len(gt) != 2 or "." in gt:
                raise ValueError(f"{path}: expected a diploid called genotype, got {fields[9]!r}")
            vcf.records.append(
                VcfRecord(fields[0], int(fields[1]) - 1, fields[3], tuple(fields[4].split(",")),
                          (int(gt[0]), int(gt[1])), line)
            )
    return vcf


def mask_vcf(vcf: Vcf, mask: Dict[str, Intervals], note: str) -> Tuple[Vcf, Dict]:
    """The records of ``vcf`` at callable positions, a header line ``##callable_mask=<note>`` added."""
    import numpy as np

    kept: List[VcfRecord] = []
    removed = {"het": 0, "hom": 0}
    for chrom, recs in vcf.by_chrom().items():
        iv = mask.get(chrom)
        ok = iv.contains(np.array([r.pos for r in recs])) if iv is not None else np.zeros(len(recs), dtype=bool)
        for rec, good in zip(recs, ok.tolist()):
            if good:
                kept.append(rec)
            else:
                removed["het" if rec.is_het else "hom"] += 1
    header = list(vcf.header)
    header.insert(len(header) - 1, f"##callable_mask={note}\n")
    counts = {
        "records_in": len(vcf.records),
        "records_kept": len(kept),
        "het_sites_kept": sum(1 for r in kept if r.is_het),
        "het_sites_removed": removed["het"],
        "hom_sites_removed": removed["hom"],
    }
    return Vcf(header, kept), counts


def write_vcf(path: Path, vcf: Vcf) -> None:
    """Write ``vcf`` as plain text (``path`` must not end in .gz; bgzip it afterwards)."""
    tmp = Path(path).with_name(Path(path).name + ".partial")
    with open(tmp, "w", encoding="ascii", newline="") as f:
        f.writelines(vcf.header)
        f.writelines(r.line for r in vcf.records)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Consensus FASTA -> fq2psmcfa
# ---------------------------------------------------------------------------


def iupac(a: str, b: str) -> str:
    """The IUPAC code of two different bases."""
    try:
        return IUPAC[frozenset((a.upper(), b.upper()))]
    except KeyError:
        raise ValueError(f"no IUPAC code for {a}/{b} (SNPs of single bases only)") from None


def consensus_sequence(length: int, records: Sequence[VcfRecord], callable_iv: Intervals, rng):
    """One chromosome of the consensus (numpy uint8 array of ASCII letters).

    Uncallable sites are ``N``; a callable site with a record is its base
    (homozygous) or IUPAC code (heterozygous); every other callable site has
    an ancestral base drawn from ``rng`` uniformly from ACGT.
    """
    import numpy as np

    seq = np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, size=length, dtype=np.uint8)]
    seq[~callable_iv.indicator(length)] = ord("N")
    called = callable_iv.contains([r.pos for r in records]).tolist() if records else []
    for rec, ok in zip(records, called):
        a, b = rec.alleles()
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f"{rec.chrom}:{rec.pos + 1}: not a SNP ({a}/{b})")
        if ok:  # the mask decides: a record at an uncallable site stays N
            seq[rec.pos] = ord(iupac(a, b) if a != b else a.upper())
    return seq


def fasta_record(name: str, seq, width: int = FASTA_LINE_WIDTH) -> bytes:
    """``>name`` and ``seq`` in lines of ``width``."""
    import numpy as np

    n = len(seq)
    full = n // width
    body = np.empty((full, width + 1), dtype=np.uint8)
    body[:, :width] = seq[: full * width].reshape(full, width)
    body[:, width] = ord("\n")
    tail = seq[full * width :].tobytes() + (b"\n" if n % width else b"")
    return b">" + name.encode("ascii") + b"\n" + body.tobytes() + tail


def consensus_fasta(
    vcf: Vcf, mask: Dict[str, Intervals], chromosomes: Sequence[Tuple[str, int]], model: str, replicate: int,
    experiment: str = "E4",
) -> Iterator[Tuple[str, bytes, Dict]]:
    """``(chrom, FASTA record, counts)`` for every chromosome, in order."""
    import numpy as np

    by_chrom = vcf.by_chrom()
    for name, length in chromosomes:
        seed = reference_seed(model, replicate, name, experiment)
        seq = consensus_sequence(length, by_chrom.get(name, []), mask[name], np.random.default_rng(seed))
        counts = {
            "name": name,
            "length": length,
            "n_sites": int(np.count_nonzero(seq == ord("N"))),
            "het_sites": int(np.isin(seq, np.frombuffer(b"MRWSYK", dtype=np.uint8)).sum()),
            "reference_seed": seed,
        }
        yield name, fasta_record(name, seq), counts


def run_fq2psmcfa(fq2psmcfa: Path, records: Iterable[bytes], out: Path, bin_size: int) -> None:
    """Stream FASTA ``records`` into ``fq2psmcfa -s BIN -`` and write its output to ``out``."""
    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "wb") as fo:
        proc = subprocess.Popen([str(fq2psmcfa), "-s", str(bin_size), "-"], stdin=subprocess.PIPE, stdout=fo)
        try:
            for rec in records:
                proc.stdin.write(rec)
            proc.stdin.close()
        except BrokenPipeError:
            pass
        status = proc.wait()
    if status != 0:
        raise RuntimeError(f"fq2psmcfa exited with status {status}")
    os.replace(tmp, out)


# ---------------------------------------------------------------------------
# psmcfa bins
# ---------------------------------------------------------------------------


def read_psmcfa(path: Path) -> Dict[str, bytes]:
    """The records of a psmcfa (gzip allowed) as ``{name: bins}``, in file order."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    out: Dict[str, bytes] = {}
    for chunk in data.split(b">")[1:]:
        head, _, body = chunk.partition(b"\n")
        name = head.split()[0].decode("ascii") if head.strip() else ""
        if name in out:
            raise ValueError(f"{path}: sequence {name} appears twice")
        out[name] = body.replace(b"\n", b"").replace(b"\r", b"")
    return out


def confusion(exact: bytes, other: bytes) -> Dict[str, int]:
    """``{"<exact>_<other>": count}`` over the bins both have, plus the bins only one has."""
    import numpy as np

    n = min(len(exact), len(other))
    a = np.frombuffer(exact[:n], dtype=np.uint8)
    b = np.frombuffer(other[:n], dtype=np.uint8)
    out = {}
    for x in SYMBOLS:
        ax = a == ord(x)
        for y in SYMBOLS:
            out[f"{x}_{y}"] = int(np.count_nonzero(ax & (b == ord(y))))
    out["other"] = n - sum(out.values())
    out["exact_only"] = len(exact) - n  # e.g. a multihetsep's tail after its last row
    out["route_only"] = len(other) - n
    return out


def differing(conf: Dict[str, int]) -> int:
    """Bins that differ from the exact route, including bins only one of them has."""
    same = sum(conf[f"{x}_{x}"] for x in SYMBOLS)
    compared = sum(conf[f"{x}_{y}"] for x in SYMBOLS for y in SYMBOLS) + conf["other"]
    return compared - same + conf["exact_only"] + conf["route_only"]


def compare_bins(exact: Dict[str, bytes], other: Dict[str, bytes]) -> Dict:
    """How the bins of a route differ from the exact route's, per chromosome and in total."""
    chroms = []
    for name, bins in exact.items():
        ob = other.get(name, b"")
        conf = confusion(bins, ob)
        chroms.append(
            {"name": name, "bins_exact": len(bins), "bins_route": len(ob), "present": name in other,
             "identical": bins == ob, "differing": differing(conf), "confusion": conf}
        )
    extra = [n for n in other if n not in exact]
    total_conf: Dict[str, int] = {}
    for c in chroms:
        for k, v in c["confusion"].items():
            total_conf[k] = total_conf.get(k, 0) + v
    for name in extra:
        total_conf["route_only"] = total_conf.get("route_only", 0) + len(other[name])
    n_exact = sum(c["bins_exact"] for c in chroms)
    n_diff = sum(c["differing"] for c in chroms) + sum(len(other[n]) for n in extra)
    return {
        "chromosomes": chroms,
        "extra_sequences": extra,
        "totals": {
            "bins_exact": n_exact,
            "bins_route": sum(len(b) for b in other.values()),
            "differing": n_diff,
            "differing_fraction": n_diff / n_exact if n_exact else 0.0,
            "identical": n_diff == 0 and not extra and all(c["present"] for c in chroms),
            "confusion": total_conf,
        },
    }


# ---------------------------------------------------------------------------
# generate_multihetsep.py cross-check
# ---------------------------------------------------------------------------


def split_vcf_by_chrom(vcf: Vcf, chrom: str, out: Path) -> int:
    """Write the header and ``chrom``'s records as a gzip VCF (the input of generate_multihetsep.py)."""
    recs = [r for r in vcf.records if r.chrom == chrom]
    with gzip.open(out, "wt", encoding="ascii", newline="") as f:
        f.writelines(vcf.header)
        f.writelines(r.line for r in recs)
    return len(recs)


def multihetsep_rows(path: Path) -> Dict[str, List[str]]:
    """The rows of a multihetsep file per chromosome (lines without their newline)."""
    out: Dict[str, List[str]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="ascii") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.setdefault(line.split("\t", 1)[0], []).append(line)
    return out


def first_difference(a: Sequence[str], b: Sequence[str]) -> Optional[Dict]:
    """The first row where two lists of lines differ (None when equal)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"row": i + 1, "a": x, "b": y}
    if len(a) != len(b):
        i = min(len(a), len(b))
        return {"row": i + 1, "a": a[i] if i < len(a) else None, "b": b[i] if i < len(b) else None}
    return None


# ---------------------------------------------------------------------------
# Steps: ``python -m experiment.pipeline.inputprep STEP`` (E4's runner jobs)
# ---------------------------------------------------------------------------


def step_mask(args) -> int:
    """Draw a callable mask, write it as BED, and write the VCF of the callable records."""
    if not str(args.vcf_out).endswith(".vcf.gz"):
        raise SystemExit("--vcf-out must end in .vcf.gz")
    vcf = read_vcf(Path(args.vcf))
    chroms = vcf.contigs()
    if not chroms:
        raise SystemExit(f"{args.vcf}: no ##contig lengths")
    spec = None
    if args.missing_fraction is not None:
        spec = config.MaskSpec(args.mask_name, args.missing_fraction, args.structure, args.mean_run_bp)
    mask = callable_mask(spec, chroms, args.model, args.replicate, args.experiment)
    Path(args.bed_out).parent.mkdir(parents=True, exist_ok=True)
    write_bed(Path(args.bed_out), mask)
    note = (f"{args.mask_name} (none missing)" if spec is None else
            f"{spec.name} ({spec.structure}: missing fraction {spec.missing_fraction:g}, "
            f"mean uncallable run {spec.mean_run_bp:g} bp); records at uncallable sites removed")
    masked, counts = mask_vcf(vcf, mask, note)
    summary = mask_summary(mask, dict(chroms))
    plain = Path(str(args.vcf_out)[: -len(".gz")])
    write_vcf(plain, masked)
    gz = simulate.bgzip_and_index(plain)
    simulate.write_json(
        Path(args.record),
        {
            "experiment": args.experiment,
            "model": args.model,
            "replicate": args.replicate,
            "mask": {"name": args.mask_name, "missing_fraction": spec.missing_fraction if spec else 0.0,
                     "structure": spec.structure if spec else "none", "mean_run_bp": spec.mean_run_bp if spec else 0.0},
            "process": "alternating renewal: geometric callable stretches (mean mean_run_bp*(1-f)/f) and "
                       "uncallable runs (mean mean_run_bp), starting with a callable stretch at position 0",
            "seeds": {name: (mask_seed(args.model, args.replicate, args.mask_name, name, args.experiment)
                             if spec else None) for name, _ in chroms},
            "summary": summary,
            "vcf": counts,
            "files": {"bed": {"name": Path(args.bed_out).name, "sha256": provenance.sha256_file(Path(args.bed_out))},
                      "vcf": {"name": gz.name, "sha256": provenance.sha256_file(gz)}},
        },
    )
    print(f"[mask] {args.mask_name} rep{args.replicate}: missing fraction "
          f"{summary['totals']['missing_fraction']:.4f}, "
          f"{counts['het_sites_removed']} heterozygous sites removed, {counts['records_kept']} records kept")
    return 0


def step_consensus(args) -> int:
    """Write the consensus FASTA of a masked VCF through fq2psmcfa (route fq2psmcfa)."""
    vcf = read_vcf(Path(args.vcf))
    mask = read_bed(Path(args.mask))
    chroms = vcf.contigs()
    for name, _ in chroms:
        mask.setdefault(name, Intervals.empty())  # no row: nothing callable
    counts: List[Dict] = []

    def records() -> Iterator[bytes]:
        for _, rec, c in consensus_fasta(vcf, mask, chroms, args.model, args.replicate, args.experiment):
            counts.append(c)
            yield rec

    run_fq2psmcfa(Path(args.fq2psmcfa), records(), Path(args.out), args.bin_size)
    simulate.write_json(
        Path(args.record),
        {
            "experiment": args.experiment,
            "model": args.model,
            "replicate": args.replicate,
            "fasta": "hom: base, het: IUPAC code, uncallable: N; other sites: uniform ACGT (seed reference/<chrom>)",
            "fq2psmcfa": [Path(args.fq2psmcfa).name, "-s", str(args.bin_size), "-"],
            "chromosomes": counts,
            "files": {"psmcfa": {"name": Path(args.out).name, "sha256": provenance.sha256_file(Path(args.out))}},
        },
    )
    print(f"[consensus] rep{args.replicate}: {sum(c['n_sites'] for c in counts)} N sites, "
          f"{sum(c['het_sites'] for c in counts)} IUPAC sites -> {args.out}")
    return 0


def step_bins(args) -> int:
    """Compare the bins of every route with the exact route's; fail when an asserted route differs."""
    exact_bytes = Path(args.exact).read_bytes()
    exact = read_psmcfa(Path(args.exact))
    out: Dict = {
        "exact": {"file": Path(args.exact).name, "counts": _bin_counts(exact), "bins": sum(len(b) for b in exact.values())},
        "routes": {},
        "asserted_identical": list(args.assert_identical or []),
    }
    failed = []
    for spec in args.route:
        name, _, path = spec.partition("=")
        other = read_psmcfa(Path(path))
        comp = compare_bins(exact, other)
        comp["counts"] = _bin_counts(other)
        comp["file"] = Path(path).name
        comp["bytes_identical"] = Path(path).read_bytes() == exact_bytes
        out["routes"][name] = comp
        if name in out["asserted_identical"] and not comp["totals"]["identical"]:
            failed.append(name)
    out["assertions_passed"] = not failed
    simulate.write_json(Path(args.out), out)
    for name, comp in out["routes"].items():
        t = comp["totals"]
        print(f"[bins] {name}: {t['differing']} of {t['bins_exact']} bins differ from the exact route "
              f"({t['differing_fraction']:.4%}){'; bytes identical' if comp['bytes_identical'] else ''}")
    if failed:
        print(f"[bins] FAILED: {', '.join(failed)} must equal the exact route bin for bin", file=sys.stderr)
        return 1
    return 0


def _bin_counts(seqs: Dict[str, bytes]) -> Dict[str, int]:
    """The bins of each symbol in all the sequences."""
    return {s: sum(bins.count(s.encode()) for bins in seqs.values()) for s in SYMBOLS}


def _rows_sha256(rows: Sequence[str]) -> str:
    """SHA-256 of rows written as lines."""
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


def step_crosscheck(args) -> int:
    """Compare psmc-rs's multihetsep with generate_multihetsep.py's, chromosome by chromosome."""
    vcf = read_vcf(Path(args.vcf))
    mask = read_bed(Path(args.mask))
    ours = multihetsep_rows(Path(args.multihetsep))
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    chroms = []
    with tempfile.TemporaryDirectory(dir=args.work_dir) as tmp:
        tmp = Path(tmp)
        for name, _ in vcf.contigs():
            vcf_path, bed_path = tmp / f"{name}.vcf.gz", tmp / f"{name}.bed.gz"
            n_records = split_vcf_by_chrom(vcf, name, vcf_path)
            write_bed(bed_path, {name: mask[name]} if name in mask else {})
            t0 = time.perf_counter()
            res = subprocess.run(
                [sys.executable, str(args.script), "--mask", str(bed_path), str(vcf_path)],
                capture_output=True, text=True,
            )
            seconds = time.perf_counter() - t0
            if res.returncode != 0:
                raise RuntimeError(f"generate_multihetsep.py failed on {name}:\n{res.stderr[-2000:]}")
            theirs = [line for line in res.stdout.split("\n") if line]
            mine = ours.get(name, [])
            diff = first_difference(theirs, mine)
            chroms.append(
                {
                    "name": name,
                    "vcf_records": n_records,
                    "rows_generate_multihetsep": len(theirs),
                    "rows_psmc_rs": len(mine),
                    "identical": diff is None,
                    "first_difference": diff,
                    "sha256_generate_multihetsep": _rows_sha256(theirs),
                    "sha256_psmc_rs": _rows_sha256(mine),
                    "seconds": round(seconds, 2),
                }
            )
            print(f"[crosscheck] {name}: {len(theirs)} rows from generate_multihetsep.py, {len(mine)} from "
                  f"psmc-rs, {'identical' if diff is None else 'DIFFERENT at row %d' % diff['row']}")
    extra = sorted(set(ours) - {c["name"] for c in chroms})
    ok = all(c["identical"] for c in chroms) and not extra
    simulate.write_json(
        Path(args.out),
        {
            "script": Path(args.script).name,
            "script_sha256": provenance.sha256_file(Path(args.script)),
            "python": platform.python_version(),
            "command": ["python", Path(args.script).name, "--mask", "<chrom>.bed.gz", "<chrom>.vcf.gz"],
            "chromosomes": chroms,
            "psmc_rs_rows_on_other_chromosomes": extra,
            "identical": ok,
        },
    )
    if not ok:
        print("[crosscheck] FAILED: the multihetsep files differ", file=sys.stderr)
        return 1
    return 0


def build_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m experiment.pipeline.inputprep",
        description="E4's input-preparation steps (runner jobs; see experiments/e4.py).",
    )
    sub = p.add_subparsers(dest="step", required=True)

    def seeds(q):
        q.add_argument("--experiment", default="E4")
        q.add_argument("--model", required=True)
        q.add_argument("--replicate", type=int, required=True)

    q = sub.add_parser("mask", help="draw a callable mask; write the BED and the VCF of callable records")
    seeds(q)
    q.add_argument("--vcf", required=True, help="the simulated VCF (with ##contig lengths)")
    q.add_argument("--mask-name", required=True)
    q.add_argument("--missing-fraction", type=float, help="expected missing fraction (omit: nothing missing)")
    q.add_argument("--structure", default="", help="clustered or dispersed (recorded)")
    q.add_argument("--mean-run-bp", type=float, default=0.0, help="mean length of an uncallable run")
    q.add_argument("--bed-out", required=True, help="callable sites, BED (.gz)")
    q.add_argument("--vcf-out", required=True, help="the masked VCF (.vcf.gz; bgzipped and indexed)")
    q.add_argument("--record", required=True, help="the mask's summary (JSON)")

    q = sub.add_parser("consensus", help="consensus FASTA of a masked VCF, streamed through fq2psmcfa")
    seeds(q)
    q.add_argument("--vcf", required=True)
    q.add_argument("--mask", required=True, help="callable sites (BED)")
    q.add_argument("--fq2psmcfa", required=True)
    q.add_argument("--bin-size", type=int, required=True)
    q.add_argument("--out", required=True, help="fq2psmcfa's psmcfa")
    q.add_argument("--record", required=True, help="counts and seeds (JSON)")

    q = sub.add_parser("bins", help="compare the bins of routes with the exact route's")
    q.add_argument("--exact", required=True, help="psmcfa of the exact route (psmc-rs convert)")
    q.add_argument("--route", action="append", required=True, metavar="NAME=PSMCFA")
    q.add_argument("--assert-identical", action="append", metavar="NAME", help="fail unless this route equals the exact one")
    q.add_argument("--out", required=True)

    q = sub.add_parser("crosscheck", help="psmc-rs's multihetsep against generate_multihetsep.py's")
    q.add_argument("--vcf", required=True)
    q.add_argument("--mask", required=True)
    q.add_argument("--multihetsep", required=True, help="psmc-rs convert --multihetsep of VCF and mask")
    q.add_argument("--script", required=True, help="generate_multihetsep.py")
    q.add_argument("--work-dir", required=True, help="for the per-chromosome VCF and mask (removed afterwards)")
    q.add_argument("--out", required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    steps = {"mask": step_mask, "consensus": step_consensus, "bins": step_bins, "crosscheck": step_crosscheck}
    return steps[args.step](args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Intervals", "SYMBOLS", "Vcf", "VcfRecord", "callable_mask", "compare_bins",
    "complement", "confusion", "consensus_fasta", "consensus_sequence", "differing", "fasta_record",
    "first_difference", "iupac", "mask_seed", "mask_summary", "mask_vcf", "multihetsep_rows", "read_bed",
    "read_psmcfa", "read_vcf", "reference_seed", "run_fq2psmcfa", "split_vcf_by_chrom", "uncallable_runs",
    "write_bed", "write_vcf",
]
