"""E7 stage 1: the current code's per-chromosome conversions of the default set against E7's inputs.

usage: e7_check.py ARCHIVE_ROOT INPUTS NEW_DIR SAMPLES_TXT OUT_DIR [NA12878_REPO_GZ]

For each sample S of SAMPLES_TXT and each chromosome C of chr1-22:
  NEW_DIR/C/S.psmcfa.gz, NEW_DIR/C/S.qc.json   written by the data stage's convert command (MANIFEST
                                               commands.assemble_chrom.convert) with the production's binary
  INPUTS/psmcfa/C/S.psmcfa.gz                  E7's per-chromosome psmcfa (src 4c0d777)
  INPUTS/qc/C.tsv                              E7's QC row (kg1000.qc_numbers of the stage's QC JSON, deleted since)
and per sample INPUTS/psmcfa/S.psmcfa.gz (what the fits read: the 22 members concatenated) and INPUTS/qc/samples.tsv.
Writes OUT_DIR/stage1_chrom.tsv (one row per sample and chromosome), OUT_DIR/stage1_samples.tsv, OUT_DIR/stage1.json
(the counts) and, for any difference, OUT_DIR/diff/<S>.<C>.json with the difference in full.
"""
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

ROOT, I, NEW, SAMPLES, OUT = map(Path, sys.argv[1:6])
REPO_NA12878 = Path(sys.argv[6]) if len(sys.argv) > 6 else None
sys.path.insert(0, str(ROOT))
from experiment.pipeline import kg1000  # noqa: E402  (the archive of 4b2c636f: qc_numbers, QC_SUMMED)

CHROMS = [f"chr{i}" for i in range(1, 23)]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_tsv(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


QC_CHROM = {c: {r["sample_id"]: r for r in read_tsv(I / "qc" / f"{c}.tsv")} for c in CHROMS}
QC_SAMPLES = {r["sample_id"]: r for r in read_tsv(I / "qc" / "samples.tsv")}
MANIFEST = json.loads((I / "MANIFEST.json").read_text())


def records(text: bytes):
    """[(name, sequence)] of a decompressed psmcfa."""
    out = []
    for line in text.decode().splitlines():
        if line.startswith(">"):
            out.append([line[1:], []])
        elif out:
            out[-1][1].append(line)
    return [(n, "".join(s)) for n, s in out]


def describe(new: bytes, old: bytes):
    """A difference between two decompressed psmcfa, in full."""
    d = {"bytes_new": len(new), "bytes_old": len(old)}
    off = next((i for i, (a, b) in enumerate(zip(new, old)) if a != b), min(len(new), len(old)))
    d["first_differing_byte"] = off
    d["first_differing_line"] = new[:off].count(b"\n") + 1
    rn, ro = records(new), records(old)
    d["records_new"] = [(n, len(s), {k: s.count(k) for k in "TKN"}) for n, s in rn]
    d["records_old"] = [(n, len(s), {k: s.count(k) for k in "TKN"}) for n, s in ro]
    bins = []
    for (n1, s1), (n2, s2) in zip(rn, ro):
        diffs = [(i, s2[i], s1[i]) for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]]
        bins.append({"record": n1, "record_old": n2, "bins_differing": len(diffs),
                     "first_1000_differences (bin, old, new)": diffs[:1000]})
    d["per_record"] = bins
    return d


def one(sample: str):
    rows, diffs = [], []
    whole_new = []
    for c in CHROMS:
        newp, qcp = NEW / c / f"{sample}.psmcfa.gz", NEW / c / f"{sample}.qc.json"
        oldp = I / "psmcfa" / c / f"{sample}.psmcfa.gz"
        row = {"sample_id": sample, "chrom": c}
        if not newp.exists() or not qcp.exists():
            row["status"] = "convert-missing"
            rows.append(row)
            continue
        ngz, ogz = newp.read_bytes(), oldp.read_bytes()
        nd, od = gzip.decompress(ngz), gzip.decompress(ogz)
        whole_new.append((ngz, nd))
        qrow = QC_CHROM[c][sample]
        q = kg1000.qc_numbers(json.loads(qcp.read_text()))
        differ = [k for k, v in q.items() if str(v) != qrow[k]]
        vcf = I / "vcf" / c / f"{sample}.vcf.gz"
        row.update({
            "psmcfa_identical": nd == od,
            "qc_fields_differing": ",".join(differ),
            "gz_identical": ngz == ogz,
            "gz_sha256_new": sha(ngz),
            "gz_sha256_recorded": qrow["psmcfa_sha256"],
            "gz_sha256_old_file": sha(ogz),
            "gz_bytes_new": len(ngz),
            "gz_bytes_recorded": qrow["psmcfa_bytes"],
            "psmcfa_bytes_decompressed": len(nd),
            "vcf_bytes_now": vcf.stat().st_size,
            "vcf_bytes_recorded": qrow["vcf_bytes"],
            **{f"qc_{k}": v for k, v in q.items()},
        })
        row["status"] = "same" if nd == od and not differ else "DIFF"
        rows.append(row)
        if row["status"] != "same":
            d = {"sample_id": sample, "chrom": c, "qc_new": q, "qc_recorded": {k: qrow[k] for k in q},
                 "qc_fields_differing": differ}
            if nd != od:
                d["psmcfa"] = describe(nd, od)
            diffs.append(((sample, c), d))
    # the whole genome: what the fits read
    wp = I / "psmcfa" / f"{sample}.psmcfa.gz"
    wgz = wp.read_bytes()
    cat_old = b"".join((I / "psmcfa" / c / f"{sample}.psmcfa.gz").read_bytes() for c in CHROMS)
    srow = {"sample_id": sample, "chromosomes_converted": len(whole_new)}
    if len(whole_new) == len(CHROMS):
        new_dec = b"".join(d for _, d in whole_new)
        new_gz = b"".join(g for g, _ in whole_new)
        srow.update({
            "whole_psmcfa_identical": new_dec == gzip.decompress(wgz),
            "whole_gz_identical": new_gz == wgz,
            "whole_gz_sha256_new": sha(new_gz),
        })
        sums = {}
        for k in kg1000.QC_SUMMED:
            if k in ("vcf_bytes", "psmcfa_bytes"):
                continue
            sums[k] = sum(int(r[f"qc_{k}"]) for r in rows if f"qc_{k}" in r)
        srow["qc_sums_differing"] = ",".join(k for k, v in sums.items() if str(v) != QC_SAMPLES[sample][k])
    srow.update({
        "whole_gz_sha256": sha(wgz),
        "whole_gz_sha256_samples_tsv": QC_SAMPLES[sample]["psmcfa_sha256"],
        "whole_gz_sha256_manifest": MANIFEST["outputs"][f"psmcfa/{sample}.psmcfa.gz"]["sha256"],
        "whole_is_cat_of_chrom_files": sha(cat_old) == sha(wgz),
    })
    return rows, srow, diffs


def main():
    samples = SAMPLES.read_text().split()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diff").mkdir(exist_ok=True)
    with multiprocessing.Pool(int(os.environ.get("CHECK_PROCS", "16"))) as pool:
        res = pool.map(one, samples)
    crow = [r for rows, _, _ in res for r in rows]
    srow = [s for _, s, _ in res]
    for _, _, diffs in res:
        for (s, c), d in diffs:
            (OUT / "diff" / f"{s}.{c}.json").write_text(json.dumps(d, indent=1) + "\n")
    for name, rows in (("stage1_chrom.tsv", crow), ("stage1_samples.tsv", srow)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, keys, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    summ = {
        "samples": len(samples),
        "chromosome_files": len(crow),
        "converted": sum(r["status"] != "convert-missing" for r in crow),
        "psmcfa_identical_decompressed": sum(bool(r.get("psmcfa_identical")) for r in crow),
        "qc_equal": sum(r.get("qc_fields_differing") == "" for r in crow),
        "status_same": sum(r["status"] == "same" for r in crow),
        "gz_bytes_identical": sum(bool(r.get("gz_identical")) for r in crow),
        "recorded_sha256_is_old_file": sum(r.get("gz_sha256_recorded") == r.get("gz_sha256_old_file") for r in crow),
        "vcf_is_the_stage_input_by_size": sum(str(r.get("vcf_bytes_now")) == r.get("vcf_bytes_recorded") for r in crow),
        "whole_psmcfa_identical_decompressed": sum(bool(s.get("whole_psmcfa_identical")) for s in srow),
        "whole_gz_identical": sum(bool(s.get("whole_gz_identical")) for s in srow),
        "whole_qc_sums_equal": sum(s.get("qc_sums_differing") == "" for s in srow),
        "whole_is_cat_of_chrom_files": sum(bool(s["whole_is_cat_of_chrom_files"]) for s in srow),
        "whole_sha256_matches_samples_tsv_and_manifest": sum(
            s["whole_gz_sha256"] == s["whole_gz_sha256_samples_tsv"] == s["whole_gz_sha256_manifest"] for s in srow),
        "differences": sorted(f"{s}.{c}" for _, _, diffs in res for (s, c), _ in diffs),
    }
    if REPO_NA12878 is not None:
        repo = REPO_NA12878.read_bytes()
        e7 = (I / "psmcfa" / "NA12878.psmcfa.gz").read_bytes()
        new = b"".join(gzip.decompress((NEW / c / "NA12878.psmcfa.gz").read_bytes()) for c in CHROMS)
        summ["NA12878"] = {
            "repo_file_sha256": sha(repo),
            "e7_file_sha256": sha(e7),
            "repo_equals_e7_gz_bytes": repo == e7,
            "repo_equals_e7_decompressed": gzip.decompress(repo) == gzip.decompress(e7),
            "repo_equals_current_code_decompressed": gzip.decompress(repo) == new,
            "decompressed_bytes": len(gzip.decompress(repo)),
        }
    (OUT / "stage1.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
