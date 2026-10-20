"""The data stage of E7: the 1000 Genomes joint-called VCFs to one psmcfa per sample.

The joint-called NYGC VCFs (3,202 samples, chr1-22: 1.68 TB) are not
stored: a chromosome is either streamed by region over HTTPS (``chunk``) or
downloaded to node-local disk, processed there and deleted (``chrom``, see
``fetch-chrom``). This stage runs outside the runner, since it is
network-bound: on a cluster its steps are SLURM jobs
(``experiment/slurm/e7_prep.sbatch``, submitted by ``submit.sh e7-prep``:
setup, a task per chromosome, the samples), and ``run`` makes everything in
one process (the tiny and smoke profiles, or one big node).
The runner's E7 jobs (``experiments/e7.py``) then fit from what it writes,
with the checksums of their inputs in their provenance records.

Steps (``python -m experiment.pipeline.kg1000 STEP --profile P --runs-dir R``):

``setup``
    The sources: HEAD of every VCF (Content-Length, ETag, Last-Modified, and
    S3's MD5 attribute) on both mirrors, checked against the 1000 Genomes
    manifest (sizes, MD5s); the indexes (.tbi, MD5 checked), the sequence
    index, the pedigree, the strict mask and GIAB's HG001 truth downloaded
    (SHA-256 recorded). The samples: the 2,504 unrelated samples are the
    sample column (10) of the sequence index, with surrounding blanks
    removed (one name there ends in a space); populations and
    superpopulations from the pedigree; the default set is
    ``per_population`` samples of every population, ``pinned`` ones first,
    the others the population's candidates with the smallest
    ``seed("E7", population, 0, "select/<sample>")``. The mask: the strict
    mask's rows of the chromosomes (cut to ``window`` if any), one BED per
    chromosome and their union. The chunks: every chromosome (its length from
    the VCF's ``##contig``) cut into regions of ``chunk_bp``. Skipped when
    done already with the same settings.
``chunk CHROM I``
    One region, streamed once::

        bcftools view --no-version --threads T -r C:A-E --regions-overlap pos -S stream_samples.txt -Ou URL##idx##TBI \\
          | bcftools annotate --no-version -x INFO,FORMAT -Ob -o chunk.bcf      # GT and FILTER stay
        bcftools index -f --csi chunk.bcf; bcftools index -n chunk.bcf          # the records
        bcftools query -e 'FILTER="PASS" || FILTER="."' -f '%CHROM\\t%POS0\\t%POS\\n' chunk.bcf | gzip   # non-PASS
        bcftools +split -S batch_K.txt -i 'GT="het" || GT="mis"' -Oz -o batch_K chunk.bcf   # each batch at once

    ``--regions-overlap pos`` puts every record in exactly one chunk.
    psmc-rs reads a non-PASS record (FILTER other than PASS or ".") as
    missing at its POS whatever the genotypes, so one BED of those positions,
    applied to every sample as a negative mask, is exactly the records kept
    (and the per-sample files hold only het or missing genotypes: in mask
    mode a hom genotype is the same as no record). ``FILTER!="PASS"`` would
    also take "." and is not used. A failed stream is retried
    (``chunk_attempts``, ``retry_wait_seconds``). The chunk BCF goes to the
    local temporary directory (node-local disk), the split files to the
    shared one (the chromosome's assembly reads them).
``fetch-chrom CHROM``, ``chrom CHROM``
    Where one connection to the mirrors is slow: the chromosome's VCF
    downloaded to the local temporary directory in ``fetch_parts`` byte
    ranges at once, its size and MD5 checked against the manifest; ``chrom``
    then runs every chunk on that copy and the assembly, and deletes it.
``assemble-chrom CHROM``
    Once every chunk of the chromosome is done: the chunks' records must sum
    to the index's count of the chromosome (``bcftools index -s``; not
    checkable for a window); the chunks' non-PASS BEDs are joined; then for
    every sample, in parallel::

        bcftools concat --no-version --naive -Oz -o S.C.vcf.gz batch_K/S.vcf.gz (every chunk, in order)
        psmc-rs convert S.C.vcf.gz --vcf-mode mask --vcf-mask mask/C.bed --vcf-negative-mask nonpass/C.bed.gz \\
          --bin-size 100 --psmcfa psmcfa/C/S.psmcfa.gz --qc-json S.json -q

    The default set keeps its per-sample VCF (``inputs/vcf/C/S.vcf.gz``),
    the others are deleted, and so are the chromosome's chunks.
``pilot CHROM``
    For ``pilot_samples``: the chromosome streamed once more, whole (all
    3,202 columns, every FORMAT field), into ``psmc-rs convert /dev/stdin
    --vcf-sample S`` with the same mask: its psmcfa must equal the split
    route's byte for byte and its QC totals and callable counts must be
    equal. It also reports the chromosome's bytes per sample and wall times,
    extrapolated to the genome, to fix the storage plan.
``assemble-samples``
    Per sample: ``cat psmcfa/chr1/S.psmcfa.gz ... chr22/S.psmcfa.gz >
    psmcfa/S.psmcfa.gz`` (psmc-rs and C psmc read concatenated gzip
    members); the default set's VCFs concatenated likewise (``bcftools
    concat --naive``); the non-PASS BEDs and the QC summed per sample; then
    ``MANIFEST.json``: the sources, tool versions, exact commands, every
    chunk's record count and the chromosome checks, how each chromosome's VCF
    was read (downloaded first, streamed, or a local copy), and the SHA-256 of
    every output. ``--prune`` then deletes the per-chromosome psmcfa and VCFs.

Layout under ``<runs>/E7/inputs/`` (not committed; the psmcfa files and the
manifest are the Zenodo deliverable)::

    samples.tsv, chunks.tsv              the sample table and the chunk plan
    meta/                                the downloaded small files, the stream's sample list and batches
    mask/<chrom>.bed, mask/callable.bed  the strict mask of each chromosome, and of all of them
    nonpass/<chrom>.bed.gz, nonpass.bed.gz   positions of non-PASS records
    psmcfa/<chrom>/<sample>.psmcfa.gz    per chromosome (until --prune)
    psmcfa/<sample>.psmcfa.gz            chr1-22: what the fits read
    vcf/<chrom>/<sample>.vcf.gz, vcf/<sample>.vcf.gz   het-or-missing VCFs of the default set
    qc/<chrom>.tsv, qc/samples.tsv       psmc-rs convert's QC per sample
    records/                             one JSON per step (setup, chunks, chromosomes, pilots)
    MANIFEST.json

Temporary files go to ``--tmp-dir`` (``$PSMC_E7_TMPDIR``, default
``<runs>/E7/tmp``; it must be shared by the chunk jobs and the chromosome's
assembly) and the chunk BCF to ``--local-tmp-dir`` (``$PSMC_E7_LOCAL_TMPDIR``,
node-local disk; default the shared one). One chromosome's split files for
2,504 samples take up to ~8 GB (chr2) and ~40,000 files; ``submit.sh
e7-prep`` keeps at most two chromosomes in flight.

``fixture`` writes the tiny profile's synthetic cohort (msprime) as local
files named as the URLs; ``check`` reports what the stage needs of a node
(the +split plugin, the open-file limit, free space, bandwidth).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import dataclasses
import gzip
import hashlib
import http.client
import io
import json
import math
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

from . import config, paths, provenance, tools

T = TypeVar("T")
NAME = "E7"
MANIFEST_ID = "psmc-rs-manuscript/E7/inputs/1"
TMP_ENV = "PSMC_E7_TMPDIR"
LOCAL_TMP_ENV = "PSMC_E7_LOCAL_TMPDIR"
NONPASS_EXPR = 'FILTER="PASS" || FILTER="."'  # excluded: what psmc-rs reads as PASS
SPLIT_EXPR = 'GT="het" || GT="mis"'
FETCH_ROOM = 1.25  # node-local space a downloaded VCF needs, per byte: the copy, its chunks and their split files
QC_COLUMNS = ("sample_id", "chrom", "bins", "t_bins", "k_bins", "missing_bins", "called_bins", "partial_bins",
              "called_bases", "het_sites", "length_bases", "vcf_bytes", "psmcfa_bytes", "psmcfa_sha256")
# Summed over the chromosomes of a sample in qc/samples.tsv; psmcfa_bytes is then the size of its psmcfa.gz
# (the members concatenated) and vcf_bytes that of its het-or-missing VCFs before any was deleted.
QC_SUMMED = ("bins", "t_bins", "k_bins", "missing_bins", "called_bins", "partial_bins", "called_bases", "het_sites",
             "length_bases", "vcf_bytes", "psmcfa_bytes")
SAMPLE_QC_COLUMNS = ("sample_id", *QC_SUMMED, "psmcfa_sha256", "vcf_bytes_kept", "vcf_sha256")


class StageError(RuntimeError):
    """A step cannot run or its check failed."""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _under_inputs(*parts: str) -> property:
    return property(lambda self: self.inputs.joinpath(*parts))


@dataclass(frozen=True)
class Layout:
    """Where E7's data stage reads and writes (``root`` is ``<runs>/E7``; see the module documentation)."""

    root: Path

    samples = _under_inputs("samples.tsv")
    chunks = _under_inputs("chunks.tsv")
    stream_samples = _under_inputs("meta", "stream_samples.txt")
    giab_vcf = _under_inputs("meta", "giab.vcf.gz")
    giab_bed = _under_inputs("meta", "giab.bed")
    mask_all = _under_inputs("mask", "callable.bed")
    nonpass_all = _under_inputs("nonpass", "nonpass.bed.gz")
    qc_samples = _under_inputs("qc", "samples.tsv")
    records = _under_inputs("records")
    record_setup = _under_inputs("records", "setup.json")
    record_samples = _under_inputs("records", "samples.json")
    manifest = _under_inputs("MANIFEST.json")

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    def meta(self, name: str) -> Path:
        return self.inputs / "meta" / name

    def batch(self, k: int) -> Path:
        return self.meta("batches") / f"batch_{k:02d}.txt"

    def tbi(self, chrom: str) -> Path:
        return self.meta("tbi") / f"{chrom}.vcf.gz.tbi"

    def mask(self, chrom: str) -> Path:
        return self.inputs / "mask" / f"{chrom}.bed"

    def nonpass(self, chrom: str) -> Path:
        return self.inputs / "nonpass" / f"{chrom}.bed.gz"

    def psmcfa_chrom(self, chrom: str, sample: str) -> Path:
        return self.inputs / "psmcfa" / chrom / f"{sample}.psmcfa.gz"

    def psmcfa(self, sample: str) -> Path:
        return self.inputs / "psmcfa" / f"{sample}.psmcfa.gz"

    def vcf_chrom(self, chrom: str, sample: str) -> Path:
        return self.inputs / "vcf" / chrom / f"{sample}.vcf.gz"

    def vcf(self, sample: str) -> Path:
        return self.inputs / "vcf" / f"{sample}.vcf.gz"

    def qc_chrom(self, chrom: str) -> Path:
        return self.inputs / "qc" / f"{chrom}.tsv"

    def record_chunk(self, chrom: str, i: int) -> Path:
        return self.records / "chunks" / f"{chrom}_{i:03d}.json"

    def record_chrom(self, chrom: str) -> Path:
        return self.records / "chromosomes" / f"{chrom}.json"

    def record_fetch(self, chrom: str) -> Path:
        return self.records / "fetches" / f"{chrom}.json"

    def record_pilot(self, chrom: str) -> Path:
        return self.records / f"pilot_{chrom}.json"


def layout(runs_dir: Path) -> Layout:
    return Layout(Path(runs_dir) / NAME)


@dataclass(frozen=True)
class Dirs:
    """The temporary directories of a step: ``shared`` (split files) and ``local`` (the chunk BCF)."""

    shared: Path
    local: Path

    def chrom(self, chrom: str) -> Path:
        return self.shared / chrom

    def chunk(self, chrom: str, i: int) -> Path:
        return self.chrom(chrom) / f"chunk{i:03d}"

    def samples(self, chrom: str) -> Path:
        """Where the chromosome's assembly concatenates and converts each sample's split files."""
        return self.chrom(chrom) / "samples"


def temp_dirs(lay: Layout, tmp_dir: Optional[Path] = None, local_tmp_dir: Optional[Path] = None) -> Dirs:
    """``--tmp-dir``, else ``$PSMC_E7_TMPDIR``, else ``<runs>/E7/tmp``; the local one defaults to it."""
    shared = Path(tmp_dir or os.environ.get(TMP_ENV) or lay.root / "tmp")
    local = Path(local_tmp_dir or os.environ.get(LOCAL_TMP_ENV) or shared)
    return Dirs(shared.expanduser().resolve(), local.expanduser().resolve())


# ---------------------------------------------------------------------------
# Files: every output is written to a .partial file and renamed when complete
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def replacing(path: Path) -> Iterator[Path]:
    """``path.partial`` (its directory made), renamed to ``path`` when the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    yield tmp
    os.replace(tmp, path)


def write_atomic(path: Path, data: bytes) -> None:
    with replacing(path) as tmp:
        tmp.write_bytes(data)


def concat_files(parts: Sequence[Path], dest: Path) -> None:
    """``cat parts > dest``, atomically."""
    with replacing(dest) as tmp, open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)


def gzip_bytes(data: bytes) -> bytes:
    """``data`` gzip-compressed with no name or time in the header (the same bytes every time)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as g:
        g.write(data)
    return buf.getvalue()


def write_json(path: Path, obj) -> None:
    provenance.write_json_atomic(Path(path), obj)


def read_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def write_tsv(path: Path, rows: Sequence[Dict], columns: Sequence[str]) -> None:
    with replacing(path) as tmp, open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def vcf_template(cfg: config.E7) -> str:
    """The URL template of the streamed VCFs: the joint VCF on the chosen mirror, or the phased panel (EBI)."""
    if cfg.input == "joint":
        return cfg.mirror_url if cfg.use_mirror else cfg.vcf_url
    if cfg.input == "panel":
        return cfg.panel_url
    raise StageError(f"E7.input must be 'joint' or 'panel', not {cfg.input!r}")


def vcf_url(cfg: config.E7, chrom: str) -> str:
    return vcf_template(cfg).format(chrom=chrom)


def manifest_url(cfg: config.E7) -> str:
    return cfg.manifest_url if cfg.input == "joint" else cfg.panel_manifest_url


def basename(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def local_root(cfg: config.E7, lay: Layout) -> Optional[Path]:
    """The directory of local files read instead of the URLs, or None."""
    if cfg.local_source is None:
        return None
    p = Path(cfg.local_source)
    return p if p.is_absolute() else lay.root / p


def resolve(cfg: config.E7, lay: Layout, url: str) -> str:
    """Where a source is read from: the URL, or the local file of its name."""
    root = local_root(cfg, lay)
    return url if root is None else str(root / basename(url))


def local_vcf(cfg: config.E7, dirs: Dirs, chrom: str) -> Path:
    """Where ``fetch-chrom`` puts a chromosome's VCF: the node-local temporary directory."""
    return dirs.local / "sources" / basename(vcf_url(cfg, chrom))


def stream_path(cfg: config.E7, lay: Layout, chrom: str, dirs: Optional[Dirs] = None) -> str:
    """What bcftools reads for a chromosome, with the index setup downloaded (``SOURCE##idx##TBI``).

    The copy ``fetch-chrom`` made in ``dirs``, when there is one; else the
    URL (or the local source's file).
    """
    src = resolve(cfg, lay, vcf_url(cfg, chrom))
    if dirs is not None and local_vcf(cfg, dirs, chrom).exists():
        src = str(local_vcf(cfg, dirs, chrom))
    return f"{src}##idx##{lay.tbi(chrom)}"


def _request(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, timeout: float = 60.0):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "psmc-rs-manuscript/E7", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def _retry(what: str, call: Callable[[], T], attempts: int, wait: float) -> T:
    """``call()``, tried ``attempts`` times, ``wait`` seconds after the first failure, 2 ``wait`` after the second..."""
    last: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return call()
        except OSError as exc:  # urllib's URLError too
            last = exc
            time.sleep(wait * (attempt + 1))
    raise StageError(f"{what} failed: {last}")


S3_MD5_RE = re.compile(r"(?:^|/)md5:([0-9a-f]{32})(?:/|$)")
MD5_RE = re.compile(r"[0-9a-f]{32}")


def head(url: str, attempts: int = 3) -> Dict:
    """Content-Length, ETag, Last-Modified (and S3's recorded MD5) of a URL, or of a local file."""
    if "://" not in url:
        st = Path(url).stat()
        return {"source": url, "content_length": st.st_size, "etag": None,
                "last_modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)), "s3_md5": None}

    def once() -> Dict:
        with _request(url, "HEAD") as r:
            h = r.headers
        m = S3_MD5_RE.search(h.get("x-amz-meta-s3cmd-attrs", "") or "")
        return {"source": url, "content_length": int(h.get("Content-Length", -1)), "etag": h.get("ETag"),
                "last_modified": h.get("Last-Modified"), "s3_md5": m.group(1) if m else None}

    return _retry(f"HEAD {url}", once, attempts, 2.0)


def file_digests(path: Path) -> Dict[str, object]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            md5.update(block)
            sha.update(block)
    return {"bytes": Path(path).stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def fetch(source: str, dest: Path, attempts: int = 3) -> Dict:
    """Download (or copy) ``source`` to ``dest`` atomically; returns its size, MD5 and SHA-256."""
    dest = Path(dest)
    with replacing(dest) as tmp:
        if "://" not in source:
            shutil.copyfile(source, tmp)
        else:
            def once() -> None:
                with _request(source, timeout=300.0) as r, open(tmp, "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 20)

            _retry(f"downloading {source}", once, attempts, 5.0)
    return {"source": source, "path": str(dest), **file_digests(dest)}


def parse_manifest(text: str) -> Dict[str, Dict]:
    """``{file name: {"bytes", "md5"}}`` of a 1000 Genomes manifest (columns in any order: name, size, MD5)."""
    out: Dict[str, Dict] = {}
    for line in text.splitlines():
        fields = line.split()
        md5 = next((f for f in fields if MD5_RE.fullmatch(f)), None)
        name = next((basename(f) for f in fields if "." in f and not MD5_RE.fullmatch(f)), None)
        if md5 is not None and name is not None:
            out[name] = {"bytes": next((int(f) for f in fields if f.isdigit()), None), "md5": md5}
    return out


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    sample_id: str
    population: str
    superpopulation: str
    sex: str
    is_parent: bool
    admixed: bool
    raw_bases: int
    raw_depth: float
    default: bool
    selection: str  # pinned, drawn or cohort
    split_check: bool
    bootstrap: bool
    giab: bool

    def row(self) -> Dict[str, object]:
        return asdict(self)


SAMPLE_COLUMNS = tuple(f.name for f in dataclasses.fields(Sample))
_PARSE = {"str": str, "int": int, "float": float, "bool": lambda v: v == "True"}


def read_samples(path: Path) -> List[Sample]:
    """The sample table setup wrote (each column parsed by its field's type)."""
    return [Sample(**{f.name: _PARSE[f.type](r[f.name]) for f in dataclasses.fields(Sample)}) for r in read_tsv(path)]


def read_index(text: str) -> Dict[str, Dict]:
    """The unrelated samples of the sequence index: ``{name: {"population", "raw_bases"}}``.

    Column 10 is the sample (its blanks removed: one name ends in a space),
    11 its population and 21 BASE_COUNT, summed over a sample's runs.
    """
    col: Dict[str, int] = {}
    out: Dict[str, Dict] = {}
    for line in text.splitlines():
        if line.startswith("##") or not line.strip():
            continue
        fields = line.rstrip("\r\n").split("\t")
        if line.startswith("#"):
            col = {f.lstrip("#").strip(): i for i, f in enumerate(fields)}
            continue
        name = fields[col.get("SAMPLE_NAME", 9)].strip()
        pop = fields[col.get("POPULATION", 10)].strip()
        try:
            bases = int(fields[col.get("BASE_COUNT", 20)].strip() or 0)
        except (ValueError, IndexError):
            bases = 0
        row = out.setdefault(name, {"population": pop, "raw_bases": 0})
        if row["population"] != pop:
            raise StageError(f"sequence index: {name} is in {row['population']} and {pop}")
        row["raw_bases"] += bases
    return out


def read_ped(text: str) -> Tuple[Dict[str, Dict], set]:
    """The pedigree: ``{sample: {population, superpopulation, sex}}`` and the samples that are a parent."""
    people: Dict[str, Dict] = {}
    parents = set()
    for line in text.splitlines():
        f = line.split()
        if len(f) < 7 or f[1] == "SampleID":
            continue
        people[f[1]] = {"population": f[5], "superpopulation": f[6], "sex": {"1": "male", "2": "female"}.get(f[4], f[4])}
        parents.update(p for p in (f[2], f[3]) if p != "0")
    return people, parents


def select_samples(cfg: config.E7, index: Dict[str, Dict], ped: Dict[str, Dict], parents: set) -> List[Sample]:
    """The samples the stage makes a psmcfa for (the cohort, or the default set), with their roles.

    Default set: per population, the ``pinned`` samples in it first, then
    the candidates with the smallest ``seed("E7", population, 0,
    "select/<sample>")`` up to ``per_population``. The first default sample
    of a population is its split-first-window check.
    """
    missing = sorted(s for s in index if s not in ped)
    if missing:
        raise StageError(f"{len(missing)} samples of the sequence index are not in the pedigree: {missing[:5]}")
    for s, row in index.items():
        if ped[s]["population"] != row["population"]:
            raise StageError(f"{s}: population {row['population']} in the index, {ped[s]['population']} in the pedigree")
    absent = [s for s in cfg.pinned if s not in index]
    if absent:
        raise StageError(f"pinned samples not among the unrelated samples: {absent}")
    by_pop: Dict[str, List[str]] = {}
    for s in sorted(index):
        by_pop.setdefault(index[s]["population"], []).append(s)
    default: Dict[str, str] = {}
    split: set = set()
    for pop, members in sorted(by_pop.items()):
        chosen = [s for s in cfg.pinned if s in members]
        rest = sorted((s for s in members if s not in chosen),
                      key=lambda s: (config.seed(NAME, pop, 0, f"select/{s}"), s))
        chosen += rest[: max(cfg.per_population - len(chosen), 0)]
        for s in chosen:
            default[s] = "pinned" if s in cfg.pinned else "drawn"
        if chosen:
            split.add(chosen[0])
    if cfg.cohort not in (None, "unrelated2504"):
        raise StageError(f"E7.cohort must be 'unrelated2504' or None, not {cfg.cohort!r}")
    out = []
    for s in sorted(index) if cfg.cohort == "unrelated2504" else sorted(default):
        p = ped[s]
        bases = int(index[s]["raw_bases"])
        out.append(Sample(
            sample_id=s, population=p["population"], superpopulation=p["superpopulation"], sex=p["sex"],
            is_parent=s in parents, admixed=p["population"] in cfg.admixed, raw_bases=bases,
            raw_depth=round(bases / cfg.raw_depth_genome_bp, 3), default=s in default,
            selection=default.get(s, "cohort"), split_check=s in split,
            bootstrap=s in default and s in cfg.bootstrap_samples, giab=s == "NA12878" and s in default and cfg.giab,
        ))
    return out


# ---------------------------------------------------------------------------
# Mask and chunks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    chrom: str
    index: int  # 0-based: the SLURM array task id
    start: int  # 1-based, inclusive
    end: int

    @property
    def region(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


def contig_lengths(header: str) -> Dict[str, int]:
    out = {}
    for m in re.finditer(r"^##contig=<([^>]*)>", header, re.M):
        fields = dict(kv.split("=", 1) for kv in m.group(1).split(",") if "=" in kv)
        if "ID" in fields and "length" in fields:
            out[fields["ID"]] = int(fields["length"])
    return out


def header_samples(header: str) -> List[str]:
    return next((line.split("\t")[9:] for line in header.splitlines() if line.startswith("#CHROM")), [])


def plan_chunks(cfg: config.E7, lengths: Dict[str, int]) -> List[Chunk]:
    """Every chromosome (or its window) cut into regions of ``chunk_bp``."""
    out = []
    for chrom in cfg.chromosomes:
        if chrom not in lengths:
            raise StageError(f"{chrom} has no ##contig length in the VCF header")
        lo, hi = 1, lengths[chrom]
        if cfg.window is not None:
            lo, hi = max(lo, cfg.window[0]), min(hi, cfg.window[1])
        n = max(math.ceil((hi - lo + 1) / cfg.chunk_bp), 1)
        for i in range(n):
            a = lo + i * cfg.chunk_bp
            out.append(Chunk(chrom, i, a, min(a + cfg.chunk_bp - 1, hi)))
    return out


def read_chunks(lay: Layout) -> List[Chunk]:
    return [Chunk(r["chrom"], int(r["chunk"]), int(r["start"]), int(r["end"])) for r in read_tsv(lay.chunks)]


def chunks_of(lay: Layout, chrom: str) -> List[Chunk]:
    out = [c for c in read_chunks(lay) if c.chrom == chrom]
    if not out:
        raise StageError(f"{chrom} has no chunks in {lay.chunks}")
    return out


def write_masks(cfg: config.E7, lay: Layout, bed: Path) -> Dict[str, int]:
    """The strict mask's rows of each chromosome (cut to the window): ``mask/<chrom>.bed`` and their union.

    Rows are kept as they are (3 columns, 0-based half-open); returns the
    callable bases of each chromosome.
    """
    rows: Dict[str, List[Tuple[int, int]]] = {c: [] for c in cfg.chromosomes}
    with open(bed) as f:
        for line in f:
            p = line.split("\t")
            if len(p) < 3 or p[0] not in rows:
                continue
            a, b = int(p[1]), int(p[2])
            if cfg.window is not None:
                a, b = max(a, cfg.window[0] - 1), min(b, cfg.window[1])
            if b > a:
                rows[p[0]].append((a, b))
    bases = {}
    with replacing(lay.mask_all) as tmp_all, open(tmp_all, "w") as out_all:
        for chrom in cfg.chromosomes:
            text = "".join(f"{chrom}\t{a}\t{b}\n" for a, b in rows[chrom])
            if not text:
                raise StageError(f"the strict mask has no callable bases on {chrom}")
            write_atomic(lay.mask(chrom), text.encode())
            out_all.write(text)
            bases[chrom] = sum(b - a for a, b in rows[chrom])
    return bases


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    """Run ``cmd``; raise :class:`StageError` with its stderr when it fails."""
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        raise StageError(f"{' '.join(str(c) for c in cmd)} failed (exit {res.returncode}): {res.stderr.strip()[-2000:]}")
    return res


def bcftools_version() -> Dict[str, str]:
    res = run(["bcftools", "--version"])
    lines = res.stdout.splitlines()
    return {"bcftools": lines[0].strip() if lines else "?", "htslib": lines[1].strip() if len(lines) > 1 else "?"}


def split_plugin_ok() -> bool:
    try:
        res = subprocess.run(["bcftools", "plugin", "-l"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return res.returncode == 0 and re.search(r"^split$", res.stdout, re.M) is not None


def rs_binary() -> Path:
    b = tools.rs02_binary()
    if not b.exists():
        raise StageError(f"{b} does not exist: cargo build --release --locked")
    return b


def rs_identity() -> Dict[str, str]:
    try:
        t = tools.psmc_rs_02(build=False, log=lambda _msg: None)
        return {**t.identity, **t.info}
    except (tools.ToolError, OSError) as exc:
        return {"error": str(exc)}


def portable_cmd(cmd: Sequence[str], lay: Layout, dirs: Optional[Dirs] = None) -> str:
    """A command line with the machine's paths as placeholders (``<runs>``, ``<tmp>``, ``<repo>``)."""
    text = " ".join(_quote(str(c)) for c in cmd)
    if dirs is not None:
        for p, name in sorted(((str(dirs.local), "<local-tmp>"), (str(dirs.shared), "<tmp>")), key=lambda x: -len(x[0])):
            text = text.replace(p, name)
    return paths.portable(text, lay.root.parent)


def _quote(s: str) -> str:
    return s if re.fullmatch(r"[A-Za-z0-9_./:=,+@%<>#-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def where() -> Dict:
    return {"host": socket.gethostname(), **provenance.slurm(os.environ), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def setup(cfg: config.E7, lay: Layout, force: bool = False, log=print) -> Dict:
    """Sources, samples, mask and chunks (see the module documentation); writes ``records/setup.json``.

    Done already with the same settings: nothing to do (``force``: again).
    """
    if not force and lay.record_setup.exists() and lay.samples.exists() and lay.chunks.exists():
        rec = read_json(lay.record_setup)
        if rec.get("settings") == json.loads(json.dumps(asdict(cfg))):
            log(f"[setup] done ({lay.record_setup})")
            return rec
    t0 = time.time()
    if not split_plugin_ok():
        raise StageError("bcftools +split is not available (bcftools plugin -l; set BCFTOOLS_PLUGINS)")
    lay.inputs.mkdir(parents=True, exist_ok=True)
    sources = fetch_files(cfg, lay, log)
    manifest = parse_manifest(lay.meta(basename(manifest_url(cfg))).read_text())
    vcfs: Dict[str, Dict] = {}
    header = ""
    for chrom in cfg.chromosomes:
        entry, h = check_vcf(cfg, lay, chrom, manifest)
        if not header:
            header = h
        elif header_samples(h) != header_samples(header):
            raise StageError(f"{basename(entry['url'])}: its samples differ from those of {cfg.chromosomes[0]}'s VCF")
        vcfs[chrom] = entry
        log(f"[setup] {chrom}: {entry['head']['content_length']:,} bytes, {entry['index_records']:,} records")
    in_header = set(header_samples(header))
    index = read_index(lay.meta(basename(cfg.index_url)).read_text())
    ped, parents = read_ped(lay.meta(basename(cfg.ped_url)).read_text())
    samples = select_samples(cfg, index, ped, parents)
    absent = [s.sample_id for s in samples if s.sample_id not in in_header]
    if absent:
        raise StageError(f"{len(absent)} samples are not in the VCF: {absent[:5]}")
    batches = write_samples(cfg, lay, samples)
    mask_bases = write_masks(cfg, lay, lay.meta(basename(cfg.mask_url)))
    lengths = contig_lengths(header)
    chunks = plan_chunks(cfg, lengths)
    write_tsv(lay.chunks, [{"chrom": c.chrom, "chunk": c.index, "start": c.start, "end": c.end} for c in chunks],
              ("chrom", "chunk", "start", "end"))
    record = {
        "step": "setup",
        "settings": asdict(cfg),
        "sources": sources,
        "vcfs": vcfs,
        "contig_lengths": {c: lengths[c] for c in cfg.chromosomes},
        "mask_callable_bases": mask_bases,
        "samples": {"unrelated_in_index": len(index), "processed": len(samples),
                    "default": sum(s.default for s in samples), "batches": batches, "vcf_columns": len(in_header)},
        "chunks": len(chunks),
        "tools": {**bcftools_version(), "psmc-rs": rs_identity(),
                  "pipeline": provenance.python_identity(["kg1000.py", "config.py"])},
        "seconds": round(time.time() - t0, 3),
        "where": where(),
    }
    write_json(lay.record_setup, record)
    log(f"[setup] {len(samples)} samples ({record['samples']['default']} default) in {batches} batches, "
        f"{len(chunks)} chunks")
    return record


def fetch_files(cfg: config.E7, lay: Layout, log=print) -> Dict[str, Dict]:
    """The manifest, sequence index, pedigree, strict mask (and GIAB's truth) into ``meta/``: their digests."""
    files = [(key, url, lay.meta(basename(url))) for key, url in (
        ("manifest", manifest_url(cfg)), ("index", cfg.index_url), ("ped", cfg.ped_url), ("mask", cfg.mask_url))]
    if cfg.giab:
        files += [("giab_vcf", cfg.giab_vcf_url, lay.giab_vcf), ("giab_bed", cfg.giab_bed_url, lay.giab_bed)]
    sources = {}
    for key, url, dest in files:
        log(f"[setup] {key} {url}")
        sources[key] = fetch(resolve(cfg, lay, url), dest)
    return sources


def check_vcf(cfg: config.E7, lay: Layout, chrom: str, manifest: Dict[str, Dict]) -> Tuple[Dict, str]:
    """A chromosome's VCF checked against the manifest (its index downloaded): its ``setup.json`` entry, its header."""
    url = vcf_url(cfg, chrom)
    name, source = basename(url), resolve(cfg, lay, url)
    entry = {"url": url, "read_from": source, "manifest": manifest.get(name), "head": head(source)}
    if cfg.input == "joint" and cfg.local_source is None:
        other = cfg.vcf_url if cfg.use_mirror else cfg.mirror_url
        entry["other_mirror"] = head(other.format(chrom=chrom))
    entry["tbi"] = {**fetch(source + ".tbi", lay.tbi(chrom)), "manifest": manifest.get(name + ".tbi")}
    entry["problems"] = source_problems(entry, local=cfg.local_source is not None)
    if entry["problems"]:
        raise StageError(f"{name}: " + "; ".join(entry["problems"]))
    entry["index_records"] = index_records(lay, chrom)
    return entry, run(["bcftools", "view", "-h", stream_path(cfg, lay, chrom)]).stdout


def write_samples(cfg: config.E7, lay: Layout, samples: Sequence[Sample]) -> int:
    """``samples.tsv``, the stream's sample list and the ``+split`` batches of it; returns the number of batches."""
    write_tsv(lay.samples, [s.row() for s in samples], SAMPLE_COLUMNS)
    names = [s.sample_id for s in samples]
    write_atomic(lay.stream_samples, "".join(f"{s}\n" for s in names).encode())
    if lay.batch(0).parent.exists():
        shutil.rmtree(lay.batch(0).parent)
    batches = [names[i:i + cfg.split_batch_size] for i in range(0, len(names), cfg.split_batch_size)]
    for k, b in enumerate(batches):
        write_atomic(lay.batch(k), "".join(f"{s}\n" for s in b).encode())
    return len(batches)


def source_problems(entry: Dict, local: bool) -> List[str]:
    """How a VCF's HEAD and index disagree with the manifest (empty: they agree)."""
    out = []
    man, h = entry.get("manifest"), entry["head"]
    if man is None:
        out.append("not in the manifest")
    else:
        if man.get("bytes") is not None and h["content_length"] != man["bytes"]:
            out.append(f"Content-Length {h['content_length']} but the manifest says {man['bytes']}")
        if h.get("s3_md5") and h["s3_md5"] != man["md5"]:
            out.append(f"S3's MD5 {h['s3_md5']} but the manifest says {man['md5']}")
        if local and "md5" in man:
            got = file_digests(Path(h["source"]))["md5"]
            if got != man["md5"]:
                out.append(f"MD5 {got} but the manifest says {man['md5']}")
    other = entry.get("other_mirror")
    if other is not None and other["content_length"] != h["content_length"]:
        out.append(f"the mirrors differ: {h['content_length']} and {other['content_length']} bytes")
    tbi = entry["tbi"]
    if tbi.get("manifest") is None:
        out.append("its index is not in the manifest")
    elif tbi["manifest"]["md5"] != tbi["md5"]:
        out.append(f"index MD5 {tbi['md5']} but the manifest says {tbi['manifest']['md5']}")
    return out


def index_records(lay: Layout, chrom: str) -> int:
    """The records of a chromosome, as its index counts them (``bcftools index -s``)."""
    res = run(["bcftools", "index", "-s", str(lay.tbi(chrom))])
    for line in res.stdout.splitlines():
        f = line.split("\t")
        if f and f[0] == chrom:
            return int(f[-1])
    return 0


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------


def chunk_commands(cfg: config.E7, lay: Layout, dirs: Dirs, chunk: Chunk, n_batches: int) -> Dict[str, object]:
    """The exact commands of one chunk."""
    bcf = dirs.local / f"{chunk.chrom}_chunk{chunk.index:03d}.bcf"
    out = dirs.chunk(chunk.chrom, chunk.index)
    return {
        "bcf": bcf,
        "out": out,
        "stream": ["bcftools", "view", "--no-version", "--threads", str(cfg.stream_threads), "-r", chunk.region,
                   "--regions-overlap", "pos", "-S", str(lay.stream_samples), "-Ou",
                   stream_path(cfg, lay, chunk.chrom, dirs)],
        "strip": ["bcftools", "annotate", "--no-version", "-x", "INFO,FORMAT", "-Ob", "-o", str(bcf)],
        "index": ["bcftools", "index", "-f", "--csi", str(bcf)],
        "count": ["bcftools", "index", "-n", str(bcf)],
        "nonpass": ["bcftools", "query", "-e", NONPASS_EXPR, "-f", r"%CHROM\t%POS0\t%POS\n", str(bcf)],
        "split": [["bcftools", "+split", "-S", str(lay.batch(k)), "-i", SPLIT_EXPR, "-Oz", "-o",
                   str(out / f"batch_{k:02d}"), str(bcf)] for k in range(n_batches)],
    }


def n_batches(lay: Layout) -> int:
    return len(list(lay.batch(0).parent.glob("batch_*.txt")))


def stream_once(cmds: Dict, log_path: Path) -> None:
    """``bcftools view ... | bcftools annotate ...``: raise when either fails (a broken stream)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as err:
        view = subprocess.Popen(cmds["stream"], stdout=subprocess.PIPE, stderr=err)
        strip = subprocess.Popen(cmds["strip"], stdin=view.stdout, stderr=err)
        view.stdout.close()
        rc_strip = strip.wait()
        rc_view = view.wait()
    if rc_view != 0 or rc_strip != 0:
        raise StageError(f"the stream failed (bcftools view exit {rc_view}, annotate exit {rc_strip}; see {log_path})")


def stream_chunk(cfg: config.E7, cmds: Dict, region: str, log=print) -> int:
    """The chunk's stream into its BCF, tried up to ``chunk_attempts`` times; returns the attempts made."""
    for attempts in range(1, max(cfg.chunk_attempts, 1) + 1):
        try:
            stream_once(cmds, cmds["out"] / "stderr.log")
            return attempts
        except StageError as exc:
            cmds["bcf"].unlink(missing_ok=True)
            if attempts >= cfg.chunk_attempts:
                raise StageError(f"{region}: {exc} (after {attempts} attempts)") from None
            log(f"[chunk] {region}: {exc}; again in {cfg.retry_wait_seconds:g} s")
            time.sleep(cfg.retry_wait_seconds)


def split_chunk(cmds: Dict, region: str) -> str:
    """Every batch's ``+split`` at once, and meanwhile the non-PASS positions (returned, and gzipped to the chunk)."""
    out = cmds["out"]
    logs = [open(out / f"split_{k:02d}.log", "w") for k in range(len(cmds["split"]))]
    try:
        procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=f) for c, f in zip(cmds["split"], logs)]
        nonpass = run(cmds["nonpass"]).stdout
        write_atomic(out / "nonpass.bed.gz", gzip_bytes(nonpass.encode()))
        failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    finally:
        for f in logs:
            f.close()
    if failed:
        raise StageError(f"{region}: bcftools +split failed for batch(es) {failed} (see {out})")
    return nonpass


def chunk(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, i: int, force: bool = False, log=print) -> Dict:
    """Stream one chunk and split it (see the module documentation); returns its record."""
    plan = {c.index: c for c in chunks_of(lay, chrom)}
    if i not in plan:
        raise StageError(f"{chrom} has chunks 0..{max(plan)}, not {i}")
    ch = plan[i]
    rec_path = lay.record_chunk(chrom, i)
    done = (dirs.chunk(chrom, i) / "DONE").exists() or lay.record_chrom(chrom).exists()
    if rec_path.exists() and not force and done:
        log(f"[chunk] {ch.region}: done")
        return read_json(rec_path)
    cmds = chunk_commands(cfg, lay, dirs, ch, n_batches(lay))
    out, bcf = cmds["out"], cmds["bcf"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dirs.local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    attempts = stream_chunk(cfg, cmds, ch.region, log)
    t_stream = time.time() - t0
    run(cmds["index"])
    records = int(run(cmds["count"]).stdout.split()[0])
    t1 = time.time()
    nonpass = split_chunk(cmds, ch.region)
    t_split = time.time() - t1
    bcf_bytes = bcf.stat().st_size
    bcf.unlink()
    Path(str(bcf) + ".csi").unlink(missing_ok=True)
    files = list(out.glob("batch_*/*.vcf.gz"))
    rec = {
        "step": "chunk",
        "chrom": chrom,
        "chunk": i,
        "start": ch.start,
        "end": ch.end,
        "records": records,
        "nonpass": nonpass.count("\n"),
        "attempts": attempts,
        "seconds": {"stream": round(t_stream, 3), "split": round(t_split, 3), "total": round(time.time() - t0, 3)},
        "bcf_bytes": bcf_bytes,
        "split_files": len(files),
        "split_bytes": sum(f.stat().st_size for f in files),
        "commands": {**{k: portable_cmd(cmds[k], lay, dirs) for k in ("stream", "strip", "index", "count", "nonpass")},
                     "split": [portable_cmd(c, lay, dirs) for c in cmds["split"]]},
        "where": where(),
    }
    write_json(rec_path, rec)
    (out / "DONE").write_text(json.dumps({"records": records}) + "\n")
    log(f"[chunk] {ch.region}: {records:,} records ({rec['nonpass']:,} non-PASS) in {rec['seconds']['total']:.0f} s, "
        f"{attempts} attempt(s)")
    return rec


# ---------------------------------------------------------------------------
# assemble-chrom
# ---------------------------------------------------------------------------


def convert_command(cfg: config.E7, vcf: Path, mask: Path, nonpass: Optional[Path], psmcfa: Path, qc: Path,
                    sample: Optional[str] = None) -> List[str]:
    """``psmc-rs convert`` of one sample's VCF (or of the joint stream with ``--vcf-sample``)."""
    cmd = [str(rs_binary()), "convert", str(vcf), "--vcf-mode", "mask"]
    if sample is not None:
        cmd += ["--vcf-sample", sample]
    cmd += ["--vcf-mask", str(mask)]
    if nonpass is not None:
        cmd += ["--vcf-negative-mask", str(nonpass)]
    cmd += ["--bin-size", str(cfg.inference.bin_size)]
    if cfg.min_good_fraction is not None:
        cmd += ["--min-good-fraction", f"{cfg.min_good_fraction:g}"]
    return cmd + ["--psmcfa", str(psmcfa), "--qc-json", str(qc), "-q"]


def qc_numbers(qc: Dict) -> Dict[str, object]:
    """What the QC table keeps of a ``--qc-json`` report."""
    t, c = qc.get("totals", {}), qc.get("callable") or {}
    return {
        "bins": t.get("bins"), "t_bins": t.get("t_bins"), "k_bins": t.get("k_bins"),
        "missing_bins": t.get("missing_bins"), "called_bins": c.get("called_bins"),
        "partial_bins": c.get("partial_bins"), "called_bases": t.get("called_bases"),
        "het_sites": t.get("het_sites"), "length_bases": t.get("length_bases"),
    }


def concat_command(out: Path, parts: Sequence[Path]) -> List[str]:
    return ["bcftools", "concat", "--no-version", "--naive", "-Oz", "-o", str(out), *[str(p) for p in parts]]


def assemble_sample_chrom(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, sample: Sample, chunks: Sequence[Chunk],
                          batch: int) -> Dict:
    """One sample's split files of the chromosome concatenated and converted: its row of ``qc/<chrom>.tsv``."""
    sid = sample.sample_id
    vcf, qc_path = dirs.samples(chrom) / f"{sid}.vcf.gz", dirs.samples(chrom) / f"{sid}.qc.json"
    parts = [dirs.chunk(chrom, c.index) / f"batch_{batch:02d}" / f"{sid}.vcf.gz" for c in chunks]
    missing = [p for p in parts if not p.exists()]
    if missing:
        raise StageError(f"{sid}: no split file {missing[0]}")
    run(concat_command(vcf, parts))
    psmcfa = lay.psmcfa_chrom(chrom, sid)
    psmcfa.parent.mkdir(parents=True, exist_ok=True)
    tmp = psmcfa.with_name(psmcfa.name.replace(".psmcfa.gz", ".partial.psmcfa.gz"))
    run(convert_command(cfg, vcf, lay.mask(chrom), lay.nonpass(chrom), tmp, qc_path))
    os.replace(tmp, psmcfa)
    row = {"sample_id": sid, "chrom": chrom, **qc_numbers(read_json(qc_path)), "vcf_bytes": vcf.stat().st_size,
           "psmcfa_bytes": psmcfa.stat().st_size, "psmcfa_sha256": provenance.sha256_file(psmcfa)}
    if sample.default:
        dest = lay.vcf_chrom(chrom, sid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vcf), dest)
    else:
        vcf.unlink()
    qc_path.unlink()
    return row


def assemble_chrom(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, jobs: int = 1, force: bool = False,
                   keep_tmp: bool = False, log=print) -> Dict:
    """Join a chromosome's chunks into every sample's psmcfa (see the module documentation)."""
    if lay.record_chrom(chrom).exists() and not force:
        log(f"[assemble-chrom] {chrom}: done")
        return read_json(lay.record_chrom(chrom))
    t0 = time.time()
    chunks = chunks_of(lay, chrom)
    absent = [c.index for c in chunks if not lay.record_chunk(chrom, c.index).exists()
              or not (dirs.chunk(chrom, c.index) / "DONE").exists()]
    if absent:
        raise StageError(f"{chrom}: chunk(s) {absent} not done")
    recs = [read_json(lay.record_chunk(chrom, c.index)) for c in chunks]
    records = sum(r["records"] for r in recs)
    expected = read_json(lay.record_setup)["vcfs"][chrom]["index_records"] if cfg.window is None else None
    if expected is not None and records != expected:
        raise StageError(f"{chrom}: the chunks hold {records:,} records, the index {expected:,}: a stream ended early; "
                         f"run its chunks again (chunk --force)")
    concat_files([dirs.chunk(chrom, c.index) / "nonpass.bed.gz" for c in chunks], lay.nonpass(chrom))
    batch_of = {s: k for k in range(n_batches(lay)) for s in lay.batch(k).read_text().split()}
    dirs.samples(chrom).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        rows = list(pool.map(lambda s: assemble_sample_chrom(cfg, lay, dirs, chrom, s, chunks, batch_of[s.sample_id]),
                             read_samples(lay.samples)))
    write_tsv(lay.qc_chrom(chrom), rows, QC_COLUMNS)
    vcf_bytes = sorted(r["vcf_bytes"] for r in rows)
    rec = {
        "step": "assemble-chrom",
        "chrom": chrom,
        "chunks": len(chunks),
        "records": records,
        "index_records": expected,
        "records_check": "equal" if expected is not None else "not checkable: a window of the chromosome",
        "nonpass": sum(r["nonpass"] for r in recs),
        "nonpass_sha256": provenance.sha256_file(lay.nonpass(chrom)),
        "samples": len(rows),
        "vcf_bytes_total": sum(vcf_bytes),
        "vcf_bytes_median": vcf_bytes[len(vcf_bytes) // 2] if vcf_bytes else 0,
        "psmcfa_bytes_total": sum(r["psmcfa_bytes"] for r in rows),
        "split_bytes_total": sum(r["split_bytes"] for r in recs),
        "stream_seconds": sum(r["seconds"]["stream"] for r in recs),
        "seconds": round(time.time() - t0, 3),
        "commands": {
            "concat": portable_cmd(concat_command(dirs.samples(chrom) / "S.vcf.gz",
                                                  [dirs.chunk(chrom, c.index) / "batch_K" / "S.vcf.gz" for c in chunks]),
                                   lay, dirs),
            "convert": portable_cmd(convert_command(cfg, dirs.samples(chrom) / "S.vcf.gz", lay.mask(chrom),
                                                    lay.nonpass(chrom), lay.psmcfa_chrom(chrom, "S"),
                                                    dirs.samples(chrom) / "S.qc.json"), lay, dirs),
        },
        "psmc-rs": rs_identity(),
        "where": where(),
    }
    write_json(lay.record_chrom(chrom), rec)
    if not keep_tmp:
        shutil.rmtree(dirs.chrom(chrom), ignore_errors=True)
    log(f"[assemble-chrom] {chrom}: {len(rows)} samples, {records:,} records ({rec['records_check']}), "
        f"{rec['seconds']:.0f} s")
    return rec


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------


def pilot(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, samples: Optional[Sequence[str]] = None,
          log=print) -> Dict:
    """The split route's psmcfa against ``psmc-rs convert --vcf-sample`` on the joint stream (see the module doc)."""
    if not lay.record_chrom(chrom).exists():
        raise StageError(f"{chrom} is not assembled yet (assemble-chrom {chrom})")
    wanted = list(samples or cfg.pilot_samples)
    known = {s.sample_id for s in read_samples(lay.samples)}
    absent = [s for s in wanted if s not in known]
    if absent:
        raise StageError(f"pilot samples not processed by the stage: {absent}")
    work = dirs.shared / f"pilot_{chrom}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    chunks = chunks_of(lay, chrom)
    region = f"{chrom}:{chunks[0].start}-{chunks[-1].end}"
    stream = ["bcftools", "view", "--no-version", "--threads", str(cfg.stream_threads), "-r", region,
              "--regions-overlap", "pos", "-Ov", stream_path(cfg, lay, chrom)]
    converts = {s: convert_command(cfg, Path("/dev/stdin"), lay.mask(chrom), None, work / f"{s}.psmcfa.gz",
                                   work / f"{s}.qc.json", sample=s) for s in wanted}
    t0 = time.time()
    tee(stream, converts, work)
    stream_s = time.time() - t0
    qc_rows = {r["sample_id"]: r for r in read_tsv(lay.qc_chrom(chrom))}
    rows = []
    for s in wanted:
        mine = gzip.decompress(lay.psmcfa_chrom(chrom, s).read_bytes())
        ref = gzip.decompress((work / f"{s}.psmcfa.gz").read_bytes())
        ref_numbers = qc_numbers(read_json(work / f"{s}.qc.json"))
        differ = [k for k, v in ref_numbers.items() if str(v) != qc_rows[s][k]]
        rows.append({"sample_id": s, "psmcfa_identical": mine == ref, "psmcfa_bytes": len(mine),
                     "qc_fields_differing": differ, "qc": ref_numbers})
    chrom_rec = read_json(lay.record_chrom(chrom))
    st, scale = storage(cfg, lay, chrom_rec, chunks[-1].end - chunks[0].start + 1)
    rec = {
        "step": "pilot",
        "chrom": chrom,
        "region": region,
        "samples": rows,
        "identical": all(r["psmcfa_identical"] and not r["qc_fields_differing"] for r in rows),
        "records_check": chrom_rec["records_check"],
        "stream_seconds_pilot": round(stream_s, 3),
        "storage": st,
        "commands": {"stream": portable_cmd(stream, lay, dirs),
                     "convert": portable_cmd(converts[wanted[0]], lay, dirs)},
        "where": where(),
    }
    write_json(lay.record_pilot(chrom), rec)
    shutil.rmtree(work, ignore_errors=True)
    for r in rows:
        log(f"[pilot] {chrom} {r['sample_id']}: psmcfa {'identical' if r['psmcfa_identical'] else 'DIFFERS'}, QC "
            + ("equal" if not r["qc_fields_differing"] else f"differs in {', '.join(r['qc_fields_differing'])}"))
    log(f"[pilot] per sample: {st['vcf_bytes_per_sample'] / 1e6:.2f} MB of het/missing VCF on {chrom} "
        f"(genome {st['vcf_bytes_per_sample_genome'] / 1e6:.1f} MB), "
        f"psmcfa.gz {st['psmcfa_bytes_per_sample_genome'] / 1e6:.2f} MB "
        f"for the genome; stream {st['stream_seconds_chrom'] / 3600:.2f} h on {chrom} (x{scale:.1f} for the genome)")
    if not rec["identical"]:
        raise StageError(f"pilot {chrom}: the split route differs from psmc-rs convert --vcf-sample "
                         f"(see {lay.record_pilot(chrom)})")
    return rec


def tee(stream: Sequence[str], sinks: Dict[str, Sequence[str]], work: Path) -> None:
    """``stream``'s output fed to every command of ``sinks`` at once (stderr to ``work``); raise if any fails."""
    with open(work / "stderr.log", "w") as err:
        src = subprocess.Popen(stream, stdout=subprocess.PIPE, stderr=err)
        procs = {s: subprocess.Popen(c, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
                 for s, c in sinks.items()}
        broken = set()
        for block in iter(lambda: src.stdout.read(1 << 20), b""):
            for s, p in procs.items():
                if s not in broken:
                    try:
                        p.stdin.write(block)
                    except BrokenPipeError:
                        broken.add(s)
        for p in procs.values():
            with contextlib.suppress(BrokenPipeError):
                p.stdin.close()
        src.stdout.close()
        rc = {s: p.wait() for s, p in procs.items()}
        rc_src = src.wait()
    if rc_src != 0 or any(rc.values()):
        raise StageError(f"the pilot's stream or conversion failed (bcftools {rc_src}, psmc-rs {rc}; see {work})")


def storage(cfg: config.E7, lay: Layout, chrom_rec: Dict, span: int) -> Tuple[Dict[str, float], float]:
    """A chromosome's bytes per sample and stream time (``span`` bases of it) extrapolated to the genome; the factor."""
    lengths = read_json(lay.record_setup)["contig_lengths"]
    genome = sum(lengths.values()) if cfg.window is None else span * len(cfg.chromosomes)
    scale = genome / span
    n = max(chrom_rec["samples"], 1)
    return {
        "scale_to_genome": round(scale, 3),
        "vcf_bytes_per_sample": chrom_rec["vcf_bytes_total"] / n,
        "vcf_bytes_per_sample_genome": chrom_rec["vcf_bytes_total"] / n * scale,
        "psmcfa_bytes_per_sample_genome": chrom_rec["psmcfa_bytes_total"] / n * scale,
        "split_bytes_chrom": chrom_rec["split_bytes_total"],
        "stream_seconds_chrom": chrom_rec["stream_seconds"],
        "stream_seconds_genome": chrom_rec["stream_seconds"] * scale,
    }, scale


# ---------------------------------------------------------------------------
# assemble-samples and the manifest
# ---------------------------------------------------------------------------


def assemble_samples(cfg: config.E7, lay: Layout, jobs: int = 1, prune: bool = False, log=print) -> Dict:
    """Every sample's chromosomes in one psmcfa (and VCF), the summed QC and ``MANIFEST.json``."""
    t0 = time.time()
    absent = [c for c in cfg.chromosomes if not lay.record_chrom(c).exists()]
    if absent:
        raise StageError(f"chromosomes not assembled yet: {', '.join(absent)}")
    samples = read_samples(lay.samples)
    by_chrom = {c: {r["sample_id"]: r for r in read_tsv(lay.qc_chrom(c))} for c in cfg.chromosomes}

    def one(s: Sample) -> Dict:
        sid = s.sample_id
        parts = [lay.psmcfa_chrom(c, sid) for c in cfg.chromosomes]
        for c, p in zip(cfg.chromosomes, parts):
            if not p.exists() or provenance.sha256_file(p) != by_chrom[c][sid]["psmcfa_sha256"]:
                raise StageError(f"{p} is missing or not the file assemble-chrom made")
        concat_files(parts, lay.psmcfa(sid))
        out = {"sample_id": sid, "psmcfa_sha256": provenance.sha256_file(lay.psmcfa(sid)),
               **{col: sum(int(by_chrom[c][sid][col]) for c in cfg.chromosomes) for col in QC_SUMMED}}
        if s.default:
            vcf = lay.vcf(sid)
            tmp = vcf.with_name(vcf.name.replace(".vcf.gz", ".partial.vcf.gz"))
            run(concat_command(tmp, [lay.vcf_chrom(c, sid) for c in cfg.chromosomes]))
            os.replace(tmp, vcf)
            out.update(vcf_sha256=provenance.sha256_file(vcf), vcf_bytes_kept=vcf.stat().st_size)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        rows = list(pool.map(one, samples))
    write_tsv(lay.qc_samples, rows, SAMPLE_QC_COLUMNS)
    concat_files([lay.nonpass(c) for c in cfg.chromosomes], lay.nonpass_all)
    rec = {"step": "assemble-samples", "samples": len(rows), "seconds": round(time.time() - t0, 3), "where": where()}
    write_json(lay.record_samples, rec)
    manifest = write_manifest(cfg, lay, rows)
    if prune:
        for c in cfg.chromosomes:
            shutil.rmtree(lay.inputs / "psmcfa" / c, ignore_errors=True)
            shutil.rmtree(lay.inputs / "vcf" / c, ignore_errors=True)
    log(f"[assemble-samples] {len(rows)} samples; {lay.manifest} ({manifest['outputs_total_bytes'] / 1e9:.2f} GB "
        f"of outputs)" + (" ; per-chromosome files pruned" if prune else ""))
    return rec


def chrom_read(lay: Layout, setup_rec: Dict, chrom: str) -> Dict[str, object]:
    """How a chromosome's VCF was read: downloaded whole first (fetch-chrom's record), from a local copy, or streamed."""
    if lay.record_fetch(chrom).exists():
        r = read_json(lay.record_fetch(chrom))
        return {"how": "downloaded whole to node-local disk in byte ranges, size and MD5 checked against the "
                       "manifest, then read by region", "url": r.get("url"), "bytes": r.get("bytes"), "md5": r.get("md5"),
                "mb_per_s": r.get("mb_per_s")}
    source = setup_rec["vcfs"][chrom]["read_from"]
    return {"how": "streamed by region over HTTPS" if "://" in source else "read by region from a local copy"}


def write_manifest(cfg: config.E7, lay: Layout, rows: Sequence[Dict]) -> Dict:
    """``MANIFEST.json``: sources, tools, commands, chunk counts and checks, and every output's SHA-256."""
    setup_rec = read_json(lay.record_setup)
    chunks = [read_json(lay.record_chunk(c.chrom, c.index)) for c in read_chunks(lay)]
    chroms = {c: read_json(lay.record_chrom(c)) for c in cfg.chromosomes}
    pilots = {p.stem.split("_", 1)[1]: read_json(p) for p in sorted(lay.records.glob("pilot_*.json"))}
    outputs: Dict[str, Dict] = {}

    def add(path: Path, sha: Optional[str] = None) -> None:
        outputs[path.relative_to(lay.inputs).as_posix()] = {"bytes": path.stat().st_size,
                                                             "sha256": sha or provenance.sha256_file(path)}

    for p in (lay.samples, lay.chunks, lay.mask_all, lay.nonpass_all, lay.qc_samples,
              *[lay.mask(c) for c in cfg.chromosomes],
              *[lay.nonpass(c) for c in cfg.chromosomes], *[lay.qc_chrom(c) for c in cfg.chromosomes]):
        add(p)
    for r in rows:
        add(lay.psmcfa(r["sample_id"]), r["psmcfa_sha256"])
        if r.get("vcf_sha256"):
            add(lay.vcf(r["sample_id"]), r["vcf_sha256"])
    manifest = {
        "manifest": MANIFEST_ID,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": ("1000 Genomes 30x NYGC joint-called VCFs (20201028_3202_raw_GT_with_annot), read by region"
                 if cfg.input == "joint" else "1000 Genomes phased panel (20220422), read by region: MAC >= 2, "
                                              "no missing genotypes (a fallback; figures must say so)")
                + " (how each chromosome was read: chromosomes.<chrom>.read)",
        "settings": setup_rec["settings"],
        "sources": {
            "vcfs": {c: {k: v for k, v in e.items() if k != "read_from"} for c, e in setup_rec["vcfs"].items()},
            "files": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in setup_rec["sources"].items()},
        },
        "tools": setup_rec["tools"],
        "commands": {
            "chunk": chunks[0]["commands"] if chunks else {},
            "assemble_chrom": next(iter(chroms.values()))["commands"] if chroms else {},
            "assemble_samples": {
                "psmcfa": "cat " + " ".join(f"psmcfa/{c}/S.psmcfa.gz" for c in cfg.chromosomes) + " > psmcfa/S.psmcfa.gz",
                "vcf": portable_cmd(concat_command(Path("vcf/S.vcf.gz"),
                                                   [Path(f"vcf/{c}/S.vcf.gz") for c in cfg.chromosomes]), lay),
            },
        },
        "chunks": [{k: r[k] for k in ("chrom", "chunk", "start", "end", "records", "nonpass", "attempts", "seconds",
                                      "bcf_bytes", "split_bytes")} for r in chunks],
        "chromosomes": {c: {**{k: r[k] for k in ("records", "index_records", "records_check", "nonpass", "samples",
                                                 "vcf_bytes_total", "psmcfa_bytes_total", "stream_seconds", "seconds")},
                            "read": chrom_read(lay, setup_rec, c)}
                        for c, r in chroms.items()},
        "pilots": {c: {k: p[k] for k in ("region", "identical", "samples", "storage")} for c, p in pilots.items()},
        "outputs": outputs,
        "outputs_total_bytes": sum(o["bytes"] for o in outputs.values()),
    }
    write_json(lay.manifest, manifest)
    return manifest


def run_all(cfg: config.E7, lay: Layout, dirs: Dirs, jobs: int = 1, pilot_chroms: Sequence[str] = (),
            prune: bool = False, log=print) -> None:
    """Every step in this process: setup, every chunk, every chromosome, the pilots, the samples."""
    setup(cfg, lay, log=log)
    plan = read_chunks(lay)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as pool:
        list(pool.map(lambda c: chunk(cfg, lay, dirs, c.chrom, c.index, log=log), plan))
    for chrom in cfg.chromosomes:
        assemble_chrom(cfg, lay, dirs, chrom, jobs=jobs, log=log)
    for chrom in pilot_chroms:
        pilot(cfg, lay, dirs, chrom, log=log)
    assemble_samples(cfg, lay, jobs=jobs, prune=prune, log=log)


# ---------------------------------------------------------------------------
# fetch-chrom and chrom: a chromosome downloaded to node-local disk first
# ---------------------------------------------------------------------------
#
# Streaming regions over HTTPS reads every region through one connection, in
# many small range requests. Where a connection to the mirrors is slow (on
# KAUST's Ibex: 0.4 to 1.7 MB/s per connection, about 25 MB/s over 4 at once;
# one 1 Mb region of chr21 took 21 minutes), ``fetch-chrom`` downloads the
# whole VCF instead, in ``fetch_parts`` byte ranges at once, each resumed
# where a broken transfer stopped, and checks its size and MD5 against the
# manifest. The chunks then read that copy (:func:`stream_path`), and
# ``chrom`` does all of a chromosome in one job on one node: fetch, every
# chunk, assembly, and the copy deleted.


def _fetch_range(url: str, part: Path, start: int, end: int, attempts: int, wait: float) -> None:
    """Bytes ``start..end`` (inclusive) of ``url`` into ``part``, appending to what an earlier attempt left."""
    want = end - start + 1
    last: Optional[Exception] = None
    for _ in range(max(attempts, 1)):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return
        if have > want:
            part.unlink()
            have = 0
        try:
            with _request(url, headers={"Range": f"bytes={start + have}-{end}"}, timeout=300.0) as r, \
                    open(part, "ab") as out:
                if r.status != 206:
                    raise StageError(f"{url}: the server ignored the range (HTTP {r.status})")
                shutil.copyfileobj(r, out, 1 << 20)
        except (OSError, http.client.HTTPException) as exc:  # URLError and IncompleteRead too
            last = exc
            time.sleep(wait)
    have = part.stat().st_size if part.exists() else 0
    if have != want:
        raise StageError(f"{url}: bytes {start}-{end} incomplete ({have} of {want}) after {attempts} attempts: {last}")


def fetch_ranges(url: str, dest: Path, size: int, parts: int, attempts: int, wait: float) -> Dict[str, object]:
    """``url`` (``size`` bytes) into ``dest`` over ``parts`` ranges at once; returns its size, MD5 and SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    step = max(math.ceil(size / max(parts, 1)), 1)
    ranges = [(start, min(start + step, size) - 1) for start in range(0, size, step)]
    part_files = [dest.with_name(f"{dest.name}.part{k:03d}") for k in range(len(ranges))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        for f in [pool.submit(_fetch_range, url, p, s, e, attempts, wait) for (s, e), p in zip(ranges, part_files)]:
            f.result()
    concat_files(part_files, dest)
    for p in part_files:
        p.unlink()
    return file_digests(dest)


def fetch_chrom(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, parts: Optional[int] = None, log=print) -> Dict:
    """Download a chromosome's VCF to node-local disk and check it against the manifest (see above)."""
    entry = read_json(lay.record_setup)["vcfs"].get(chrom)
    if entry is None:
        raise StageError(f"{chrom} is not among the chromosomes of setup")
    source, man = entry["read_from"], entry.get("manifest") or {}
    if "://" not in source:
        log(f"[fetch] {chrom}: read from {source}, nothing to fetch")
        return {"step": "fetch", "chrom": chrom, "path": source, "fetched": False}
    size = int(entry["head"]["content_length"])
    dest = local_vcf(cfg, dirs, chrom)
    parts = parts or cfg.fetch_parts
    t0 = time.time()
    fetched = not (dest.exists() and dest.stat().st_size == size)
    if fetched:
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = sum(p.stat().st_size for p in dest.parent.glob(dest.name + ".part*"))  # a resumed download
        need = int(size * FETCH_ROOM) - have
        free = shutil.disk_usage(dest.parent).free
        if free < need:
            raise StageError(f"{chrom}: {free / 1e9:.1f} GB free in {dest.parent}, the download and its chunks "
                             f"need {need / 1e9:.1f} GB (--local-tmp-dir)")
        log(f"[fetch] {chrom}: {size / 1e9:.1f} GB from {source} in {parts} ranges")
        digests = fetch_ranges(source, dest, size, parts, cfg.chunk_attempts * 10, cfg.retry_wait_seconds)
    else:
        digests = file_digests(dest)
    problems = []
    if digests["bytes"] != size:
        problems.append(f"{digests['bytes']} bytes, not {size}")
    if man.get("md5") and digests["md5"] != man["md5"]:
        problems.append(f"MD5 {digests['md5']}, the manifest says {man['md5']}")
    if problems:
        dest.unlink(missing_ok=True)
        raise StageError(f"{chrom}: the download is wrong ({'; '.join(problems)}); deleted, fetch it again")
    seconds = time.time() - t0
    rec = {"step": "fetch", "chrom": chrom, "url": source, "path": str(dest), "fetched": fetched, **digests,
           "manifest_md5": man.get("md5"), "parts": parts, "seconds": round(seconds, 3),
           "mb_per_s": round(size / seconds / 1e6, 2) if fetched and seconds > 0 else None, "where": where()}
    write_json(lay.record_fetch(chrom), rec)
    speed = f", {rec['mb_per_s']} MB/s" if fetched else ""
    log(f"[fetch] {chrom}: ok ({digests['md5']}{speed})")
    return rec


def chrom_local(cfg: config.E7, lay: Layout, dirs: Dirs, chrom: str, jobs: int = 1, chunk_jobs: int = 1,
                parts: Optional[int] = None, keep_source: bool = False, log=print) -> Dict:
    """One chromosome in one job: fetch it, every chunk (``chunk_jobs`` at once), its assembly; the copy deleted."""
    if lay.record_chrom(chrom).exists():
        log(f"[chrom] {chrom}: done")
        return read_json(lay.record_chrom(chrom))
    fetch_chrom(cfg, lay, dirs, chrom, parts=parts, log=log)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(chunk_jobs, 1)) as pool:
        list(pool.map(lambda c: chunk(cfg, lay, dirs, chrom, c.index, log=log), chunks_of(lay, chrom)))
    rec = assemble_chrom(cfg, lay, dirs, chrom, jobs=jobs, log=log)
    if not keep_source:
        local_vcf(cfg, dirs, chrom).unlink(missing_ok=True)
    return rec


# ---------------------------------------------------------------------------
# check: what the stage needs of a node
# ---------------------------------------------------------------------------


def probe(url: str, nbytes: int, timeout: float = 600.0) -> Dict:
    """Download the first ``nbytes`` of ``url`` (a Range request) and time it."""
    t0 = time.time()
    got = 0
    try:
        with _request(url, headers={"Range": f"bytes=0-{nbytes - 1}"}, timeout=timeout) as r:
            for block in iter(lambda: r.read(1 << 20), b""):
                got += len(block)
    except OSError as exc:  # URLError too
        return {"url": url, "error": str(exc), "bytes": got}
    dt = time.time() - t0
    return {"url": url, "bytes": got, "seconds": round(dt, 3), "mb_per_s": round(got / dt / 1e6, 2) if dt else None}


def check(cfg: config.E7, lay: Layout, dirs: Dirs, probe_bytes: int = 0, log=print) -> Dict:
    """The +split plugin, the open-file limit, free space and files of the temporary directories, and bandwidth."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    out: Dict[str, object] = {"split_plugin": split_plugin_ok(), "open_files": {"soft": soft, "hard": hard},
                              "split_batch_size": cfg.split_batch_size, "where": where()}
    try:
        out["bcftools"] = bcftools_version()
    except (StageError, OSError) as exc:
        out["bcftools"] = str(exc)
    for name, d in (("tmp", dirs.shared), ("local_tmp", dirs.local), ("inputs", lay.inputs)):
        d.mkdir(parents=True, exist_ok=True)
        st = os.statvfs(d)
        out[name] = {"path": str(d), "free_gb": round(st.f_bavail * st.f_frsize / 2**30, 1),
                     "free_inodes": st.f_favail}
    if probe_bytes > 0:
        out["bandwidth"] = [probe(u.format(chrom="chr21"), probe_bytes) for u in (cfg.vcf_url, cfg.mirror_url)]
    for k, v in out.items():
        log(f"[check] {k}: {v}")
    if soft < cfg.split_batch_size + 64:
        log(f"[check] the open-file limit {soft} is below split_batch_size + 64: lower E7.split_batch_size or ulimit -n")
    return out


# ---------------------------------------------------------------------------
# fixture: the tiny profile's synthetic cohort
# ---------------------------------------------------------------------------

# (sample, population, superpopulation, sex, father, mother, unrelated)
FIXTURE_PEOPLE = (
    ("NA18507", "YRI", "AFR", "1", "0", "0", True),
    ("NA19238", "YRI", "AFR", "2", "0", "0", True),
    ("NA19239", "YRI", "AFR", "1", "0", "0", True),
    ("NA19240", "YRI", "AFR", "2", "NA19239", "NA19238", False),
    ("HG01879", "ACB", "AFR", "1", "0", "0", True),
    ("HG01880", "ACB", "AFR", "2", "0", "0", True),
    ("HG01882", "ACB", "AFR", "1", "0", "0", True),
    ("NA12878", "CEU", "EUR", "2", "NA12891", "NA12892", True),
    ("NA12889", "CEU", "EUR", "1", "0", "0", True),
    ("NA12890", "CEU", "EUR", "2", "0", "0", True),
    ("NA12891", "CEU", "EUR", "1", "0", "0", False),
    ("NA12892", "CEU", "EUR", "2", "0", "0", False),
    ("HG00096", "GBR", "EUR", "1", "0", "0", True),
    ("HG00097", "GBR", "EUR", "2", "0", "0", True),
    ("HG00099", "GBR", "EUR", "2", "0", "0", True),
    ("NA18525", "CHB", "EAS", "2", "0", "0", True),
    ("NA18526", "CHB", "EAS", "2", "0", "0", True),
    ("NA18528", "CHB", "EAS", "2", "0", "0", True),
    ("NA19648", "MXL", "AMR", "2", "0", "0", True),
    ("NA19649", "MXL", "AMR", "1", "0", "0", True),
    ("NA19651", "MXL", "AMR", "2", "0", "0", True),
    ("NA20845", "GIH", "SAS", "1", "0", "0", True),
    ("NA20846", "GIH", "SAS", "1", "0", "0", True),
    ("NA20847", "GIH", "SAS", "2", "0", "0", True),
)
FIXTURE_CONTIGS = (("chr21", 400_000), ("chr22", 300_000))
FIXTURE_EXTRA_CONTIG = ("chrX", 500_000)  # in the header and the mask, never streamed
FIXTURE_NONPASS = "VQSRTrancheSNP99.80to100.00"
FIXTURE_INDEX_COLUMNS = (
    "ENA_FILE_PATH", "MD5SUM", "RUN_ID", "STUDY_ID", "STUDY_NAME", "CENTER_NAME", "SUBMISSION_ID", "SUBMISSION_DATE",
    "SAMPLE_ID", "SAMPLE_NAME", "POPULATION", "EXPERIMENT_ID", "INSTRUMENT_PLATFORM", "INSTRUMENT_MODEL",
    "LIBRARY_NAME", "RUN_NAME", "INSERT_SIZE", "LIBRARY_LAYOUT", "PAIRED_FASTQ", "READ_COUNT", "BASE_COUNT",
    "ANALYSIS_GROUP",
)  # the 22 columns of the real sequence index


def _bgzip(text: str, dest: Path) -> None:
    with open(dest, "wb") as out:
        subprocess.run(["bgzip", "-c"], input=text.encode(), stdout=out, check=True)


def make_fixture(cfg: config.E7, dest: Path, log=print) -> Dict[str, Path]:
    """The tiny profile's cohort, as local files named as :class:`.config.E7`'s URLs.

    24 diploid genomes (21 unrelated, in 7 populations of every
    superpopulation; 3 relatives in the pedigree only) simulated by msprime
    on a 400 kb ``chr21`` and a 300 kb ``chr22`` (a constant population of
    10,000, mutation rate 4e-8, recombination 1e-8; seeds
    ``seed("E7", "fixture", 0, ...)``), then made to look like the joint
    calls: 1% of genotypes missing, 4% of records non-PASS and one with
    FILTER ".", 3% of records turned into indels, INFO and FORMAT fields to
    strip. Also the strict mask (with chrX rows), a sequence index (one name
    with a trailing blank, one sample with two runs), the pedigree, the
    manifest (sizes and MD5s) and a GIAB-like truth for NA12878 with two of
    its heterozygous sites left out and three added.
    """
    import numpy as np

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed(NAME, "fixture", 0, "perturb"))
    names = [p[0] for p in FIXTURE_PEOPLE]
    contig_lines = "".join(f"##contig=<ID={c},length={n}>\n" for c, n in (*FIXTURE_CONTIGS, FIXTURE_EXTRA_CONTIG))
    header = ("##fileformat=VCFv4.2\n"
              "##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
              f"##FILTER=<ID={FIXTURE_NONPASS},Description=\"Truth sensitivity tranche level for SNP model\">\n"
              "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count\">\n"
              "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total alleles\">\n"
              "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
              "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
              "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths\">\n"
              "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
              "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">\n"
              + contig_lines + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names) + "\n")
    files: Dict[str, Path] = {}
    truth_hets: Dict[str, List[Tuple[int, str, str]]] = {}
    homref: Dict[str, List[Tuple[int, str]]] = {}
    for chrom, length in FIXTURE_CONTIGS:
        lines, truth_hets[chrom], homref[chrom] = _fixture_records(chrom, length, rng)
        files[chrom] = dest / basename(vcf_url(cfg, chrom))
        _bgzip(header + "".join(lines), files[chrom])
        run(["tabix", "-f", "-p", "vcf", str(files[chrom])])
        log(f"[fixture] {chrom}: {len(lines)} records")
    write_atomic(dest / basename(cfg.mask_url), _fixture_mask(rng).encode())
    write_atomic(dest / basename(cfg.index_url), _fixture_index().encode())
    ped = ["FamilyID SampleID FatherID MotherID Sex Population Superpopulation\n"]
    ped += [f"{s} {s} {f} {m} {sex} {pop} {sup}\n" for s, pop, sup, sex, f, m, _u in FIXTURE_PEOPLE]
    write_atomic(dest / basename(cfg.ped_url), "".join(ped).encode())
    truth, bed = _fixture_truth(contig_lines, truth_hets, homref)
    _bgzip(truth, dest / basename(cfg.giab_vcf_url))
    write_atomic(dest / basename(cfg.giab_bed_url), bed.encode())
    man = []  # name, size and MD5 of every VCF and index
    for chrom, _n in FIXTURE_CONTIGS:
        for p in (files[chrom], Path(str(files[chrom]) + ".tbi")):
            d = file_digests(p)
            man.append(f"{p.name}\t{d['bytes']}\t{d['md5']}\n")
    write_atomic(dest / basename(manifest_url(cfg)), "".join(man).encode())
    log(f"[fixture] {dest}: {len(names)} genomes, {sum(n for _c, n in FIXTURE_CONTIGS):,} bp")
    return files


def _fixture_records(chrom: str, length: int, rng) -> Tuple[List[str], List[Tuple[int, str, str]],
                                                           List[Tuple[int, str]]]:
    """A chromosome's records of the fixture's joint VCF, and NA12878's heterozygous and hom-ref sites in them.

    The genomes are simulated, then perturbed with ``rng``: a reference base at
    every position, indels, FILTER, missing genotypes and the FORMAT fields.
    """
    import msprime

    names = [p[0] for p in FIXTURE_PEOPLE]
    i12878 = names.index("NA12878")
    ts = msprime.sim_ancestry(samples=len(names), population_size=10_000, sequence_length=length,
                              recombination_rate=1e-8, random_seed=config.seed(NAME, "fixture", 0, f"ancestry/{chrom}"))
    mts = msprime.sim_mutations(ts, rate=4e-8, random_seed=config.seed(NAME, "fixture", 0, f"mutations/{chrom}"))
    ref = rng.choice(list("ACGT"), size=length + 10)
    sites = []
    for var in mts.variants():
        pos = int(var.site.position) + 1
        if pos + 4 > length:
            continue
        ref[pos - 1] = var.alleles[0]
        sites.append((pos, var.alleles, var.genotypes.reshape(-1, 2)))
    lines: List[str] = []
    hets: List[Tuple[int, str, str]] = []
    homref: List[Tuple[int, str]] = []
    last_end = 0
    for j, (pos, alleles, gts) in enumerate(sites):
        if pos <= last_end:
            continue
        nxt = sites[j + 1][0] if j + 1 < len(sites) else length
        used = sorted({int(a) for g in gts for a in g})
        alts = [alleles[a] for a in used if a != 0]
        if not alts:
            continue
        code = {0: "0", **{a: str(k + 1) for k, a in enumerate(a for a in used if a != 0)}}
        g12878 = gts[i12878]
        if g12878[0] != g12878[1]:
            hets.append((pos, alleles[0], ",".join(alleles[int(a)] for a in g12878 if int(a) != 0)))
        elif g12878[0] == 0:
            homref.append((pos, alleles[0]))
        r = alleles[0]
        alt = ",".join(alts)
        if len(alts) == 1 and nxt > pos + 4 and rng.random() < 0.03:
            if rng.random() < 0.5:
                r = "".join(ref[pos - 1:pos + 2])
                alt = ref[pos - 1]
            else:
                alt = r + "AT"
        filt = FIXTURE_NONPASS if rng.random() < 0.04 else "PASS"
        if j == len(sites) // 2:
            filt = "."
        fields = []
        ac = 0
        for g in gts:
            a, b = code[int(g[0])], code[int(g[1])]
            ac += (a != "0") + (b != "0")
            gt = "./." if rng.random() < 0.01 else f"{a}/{b}"
            dp = int(rng.integers(20, 50))
            fields.append(f"{gt}:{dp // 2},{dp - dp // 2}:{dp}:{int(rng.integers(10, 99))}")
        qual = f"{rng.uniform(30, 5000):.2f}"
        lines.append(f"{chrom}\t{pos}\t.\t{r}\t{alt}\t{qual}\t{filt}\tAC={ac};AN={2 * len(names)};DP=900\t"
                     "GT:AD:DP:GQ\t" + "\t".join(fields) + "\n")
        last_end = pos + len(r) - 1
    return lines, hets, homref


def _fixture_mask(rng) -> str:
    """The strict mask: callable stretches of ~3 kb between ~1 kb gaps, short gaps inside (partly callable bins)."""
    lines = []
    for chrom, length in (*FIXTURE_CONTIGS, FIXTURE_EXTRA_CONTIG):
        pos = int(rng.integers(0, 2_000))
        while pos < length:
            end = min(pos + int(rng.geometric(1 / 3_000)), length)
            cut = pos
            while cut < end:
                piece = min(cut + int(rng.geometric(1 / 400)), end)
                lines.append(f"{chrom}\t{cut}\t{piece}\tstrict\n")
                cut = piece + int(rng.integers(1, 8))
            pos = end + int(rng.geometric(1 / 1_000))
    return "".join(lines)


def _fixture_index() -> str:
    """The sequence index: the unrelated samples, HG00097 with two runs, NA19648's name with a trailing blank."""
    lines = ["##FileDate=20190503\n", "##SAMPLE_NAME=Sample name\n", "#" + "\t".join(FIXTURE_INDEX_COLUMNS) + "\n"]
    for k, (s, pop, *_rest, unrelated) in enumerate(FIXTURE_PEOPLE):
        if not unrelated:
            continue
        runs = 2 if s == "HG00097" else 1
        for r in range(runs):
            row = [""] * len(FIXTURE_INDEX_COLUMNS)
            row[0] = f"ftp://ftp.sra.ebi.ac.uk/vol1/run/ERR000/ERR{k:04d}{r}/{s}.final.cram"
            row[9] = s + (" " if s == "NA19648" else "")  # as in the real index
            row[10] = pop
            row[20] = str(int(1.1e11 + 1e9 * k) // runs)
            row[21] = "high_cov"
            lines.append("\t".join(row) + "\n")
    return "".join(lines)


def _fixture_truth(contig_lines: str, hets: Dict[str, List[Tuple[int, str, str]]],
                   homref: Dict[str, List[Tuple[int, str]]]) -> Tuple[str, str]:
    """A GIAB-like truth for NA12878 (its heterozygous sites, two left out, three added) and its confident regions
    (everywhere but a gap): the VCF's text and the BED's."""
    truth = ["##fileformat=VCFv4.2\n", contig_lines, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG001\n"]
    bed = []
    for chrom, length in FIXTURE_CONTIGS:
        sites = hets[chrom]
        drop = {sites[5][0], sites[-5][0]} if len(sites) > 10 else set()
        recs = [(p, r, a) for p, r, a in sites if p not in drop]
        recs += [(p, r, "A" if r != "A" else "C") for p, r in homref[chrom][10:13]]
        for p, r, a in sorted(recs):
            gt = "1/2" if "," in a else "0|1"
            truth.append(f"{chrom}\t{p}\t.\t{r}\t{a}\t50\tPASS\t.\tGT\t{gt}\n")
        gap = length // 2
        bed += [f"{chrom}\t0\t{gap}\n", f"{chrom}\t{gap + 20_000}\t{length}\n"]
    return "".join(truth), "".join(bed)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m experiment.pipeline.kg1000", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)

    def common(sp, jobs: bool = False):
        sp.add_argument("--profile", default="production", choices=sorted(config.PROFILES))
        sp.add_argument("--runs-dir", type=Path, help="runs directory (default: the profile's)")
        sp.add_argument("--tmp-dir", type=Path, help=f"shared temporary directory (default ${TMP_ENV}, else <runs>/E7/tmp)")
        sp.add_argument("--local-tmp-dir", type=Path,
                        help=f"node-local temporary directory for the chunk BCF (default ${LOCAL_TMP_ENV}, else --tmp-dir)")
        sp.add_argument("--input", choices=("joint", "panel"),
                        help="the VCFs read, instead of the profile's E7.input (setup records it)")
        sp.add_argument("--local-source", type=Path,
                        help="read the sources from this directory of files named as the URLs, instead of the URLs")
        if jobs:
            sp.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="samples (or chunks) at once")

    sp = sub.add_parser("setup", help="sources, samples, mask and chunk plan")
    common(sp)
    sp.add_argument("--force", action="store_true", help="again, even if it is done with the same settings")
    sp = sub.add_parser("chunk", help="stream and split one chunk")
    common(sp)
    sp.add_argument("chrom")
    sp.add_argument("index", type=int, help="0-based chunk of the chromosome (the SLURM array task id)")
    sp.add_argument("--force", action="store_true", help="again, even if it is done")
    sp = sub.add_parser("assemble-chrom", help="a chromosome's chunks into every sample's psmcfa")
    common(sp, jobs=True)
    sp.add_argument("chrom")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--keep-tmp", action="store_true", help="keep the chunks' split files")
    sp = sub.add_parser("fetch-chrom", help="download a chromosome's VCF to node-local disk, MD5 checked")
    common(sp)
    sp.add_argument("chrom")
    sp.add_argument("--parts", type=int, help="byte ranges at once (default E7.fetch_parts)")
    sp = sub.add_parser("chrom", help="a chromosome in one job: fetch, every chunk, its assembly")
    common(sp, jobs=True)
    sp.add_argument("chrom")
    sp.add_argument("--chunk-jobs", type=int, default=4, help="chunks at once")
    sp.add_argument("--parts", type=int, help="byte ranges at once (default E7.fetch_parts)")
    sp.add_argument("--keep-source", action="store_true", help="keep the downloaded VCF")
    sp = sub.add_parser("pilot", help="the split route against psmc-rs convert --vcf-sample on the stream")
    common(sp)
    sp.add_argument("chrom")
    sp.add_argument("--samples", help="comma-separated (default: E7.pilot_samples)")
    sp = sub.add_parser("assemble-samples", help="every sample's chromosomes in one psmcfa; MANIFEST.json")
    common(sp, jobs=True)
    sp.add_argument("--prune", action="store_true", help="then delete the per-chromosome psmcfa and VCFs")
    sp = sub.add_parser("run", help="every step in this process")
    common(sp, jobs=True)
    sp.add_argument("--pilot", action="append", default=[], metavar="CHROM", help="also run the pilot on CHROM")
    sp.add_argument("--prune", action="store_true")
    sp = sub.add_parser("chunks", help="print the chunks of the plan (chrom, index, region), for submit.sh")
    common(sp)
    sp.add_argument("--chrom", action="append", default=[])
    common(sub.add_parser("fixture", help="the tiny profile's synthetic cohort, as local files"))
    sp = sub.add_parser("check", help="what the stage needs of this node")
    common(sp)
    sp.add_argument("--probe-bytes", type=int, default=0, help="time downloading this many bytes of chr21 from each mirror")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.get_profile(args.profile).e7
    if args.input:
        cfg = config.replace(cfg, input=args.input)
    if args.local_source:
        cfg = config.replace(cfg, local_source=str(args.local_source.expanduser().resolve()))
    runs = (args.runs_dir or paths.default_runs_dir(args.profile)).resolve()
    lay = layout(runs)
    dirs = temp_dirs(lay, args.tmp_dir, args.local_tmp_dir)
    try:
        if args.step == "setup":
            setup(cfg, lay, force=args.force)
        elif args.step == "chunk":
            chunk(cfg, lay, dirs, args.chrom, args.index, force=args.force)
        elif args.step == "assemble-chrom":
            assemble_chrom(cfg, lay, dirs, args.chrom, jobs=args.jobs, force=args.force, keep_tmp=args.keep_tmp)
        elif args.step == "fetch-chrom":
            fetch_chrom(cfg, lay, dirs, args.chrom, parts=args.parts)
        elif args.step == "chrom":
            chrom_local(cfg, lay, dirs, args.chrom, jobs=args.jobs, chunk_jobs=args.chunk_jobs, parts=args.parts,
                        keep_source=args.keep_source)
        elif args.step == "pilot":
            pilot(cfg, lay, dirs, args.chrom, args.samples.split(",") if args.samples else None)
        elif args.step == "assemble-samples":
            assemble_samples(cfg, lay, jobs=args.jobs, prune=args.prune)
        elif args.step == "run":
            run_all(cfg, lay, dirs, jobs=args.jobs, pilot_chroms=args.pilot, prune=args.prune)
        elif args.step == "chunks":
            for c in read_chunks(lay):
                if not args.chrom or c.chrom in args.chrom:
                    print(f"{c.chrom}\t{c.index}\t{c.region}")
        elif args.step == "fixture":
            root = local_root(cfg, lay)
            if root is None:
                raise StageError(f"the {args.profile} profile reads the network (E7.local_source is None)")
            make_fixture(cfg, root)
        elif args.step == "check":
            check(cfg, lay, dirs, probe_bytes=args.probe_bytes)
    except StageError as exc:
        print(f"kg1000 {args.step}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Chunk", "Dirs", "Layout", "Sample", "StageError", "assemble_chrom", "assemble_samples", "check", "chrom_local",
    "chunk", "contig_lengths", "fetch_chrom", "fetch_ranges", "head", "layout", "make_fixture", "parse_manifest",
    "pilot", "plan_chunks", "read_chunks", "read_index", "read_ped", "read_samples", "run_all", "select_samples",
    "setup", "source_problems", "stream_path", "temp_dirs", "vcf_url", "write_manifest",
]
