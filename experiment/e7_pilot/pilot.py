"""I4: E7's chr21 pilot redone with the production's binary (src dcf29ca), reusing kg1000's own code.

usage: pilot.py ARCHIVE_ROOT INPUTS LOCAL_TMP OUT_DIR SPLIT_DIR

ARCHIVE_ROOT  an archive of 4b2c636f with the production's binary copied to target/release/psmc-rs
INPUTS        E7's inputs (read only)
LOCAL_TMP     node-local scratch: the 26.1 GB download goes here and is deleted at the end
OUT_DIR       the record (pilot_chr21.json) and the direct route's psmcfa and QC JSON
SPLIT_DIR     the split route's chr21 psmcfa and QC JSON of the three samples, written by I1's stage 1 with the
              same binary (MANIFEST commands.assemble_chrom.convert); converted here if absent

Direct route as kg1000.pilot: the joint chr21 VCF (all 3,202 columns), downloaded whole by kg1000.fetch_ranges in
16 ranges and checked against the manifest's size and MD5, streamed once by bcftools view -r <region> and fed by
kg1000.tee to kg1000.convert_command(..., sample=S) for each pilot sample.
"""
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT, I, LOCAL, OUT, SPLIT = map(Path, sys.argv[1:6])
sys.path.insert(0, str(ROOT))
from experiment.pipeline import config, kg1000  # noqa: E402

CHROM = "chr21"
man = json.loads((I / "MANIFEST.json").read_text())
cfg = config.get_profile("production").e7
assert list(cfg.pilot_samples) == man["settings"]["pilot_samples"], (cfg.pilot_samples, man["settings"]["pilot_samples"])
assert cfg.min_good_fraction is None and man["settings"]["min_good_fraction"] is None
assert cfg.inference.bin_size == man["settings"]["inference"]["bin_size"] == 100
assert cfg.stream_threads == man["settings"]["stream_threads"]
samples = list(cfg.pilot_samples)
read = man["chromosomes"][CHROM]["read"]
url, size, md5 = read["url"], int(read["bytes"]), read["md5"]
region = man["pilots"][CHROM]["region"]
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / "pilot.log", "a")


def say(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def sha(b):
    return hashlib.sha256(b).hexdigest()


binary = kg1000.rs_binary()
ident = kg1000.rs_identity()
say(f"binary {binary} sha256 {ident.get('binary_sha256')}")
rec = {"step": "pilot", "chrom": CHROM, "region": region, "samples": [], "where": kg1000.where(),
       "psmc-rs": ident}

# the split route (the data stage's per-chromosome convert of the kept VCF), current code
split = {}
for s in samples:
    p, q = SPLIT / f"{s}.psmcfa.gz", SPLIT / f"{s}.qc.json"
    how = f"reused from I1 stage 1 ({SPLIT})"
    if not (p.exists() and q.exists()):
        SPLIT.mkdir(parents=True, exist_ok=True)
        cmd = kg1000.convert_command(cfg, I / "vcf" / CHROM / f"{s}.vcf.gz", I / "mask" / f"{CHROM}.bed",
                                     I / "nonpass" / f"{CHROM}.bed.gz", p, q)
        subprocess.run(cmd, check=True)
        how = "converted in this job: " + " ".join(cmd)
    split[s] = (gzip.decompress(p.read_bytes()), kg1000.qc_numbers(json.loads(q.read_text())), sha(p.read_bytes()), how)
rec["commands"] = {"split_convert": kg1000.convert_command(cfg, Path("<inputs>/vcf/chr21/S.vcf.gz"),
                                                          Path("<inputs>/mask/chr21.bed"),
                                                          Path("<inputs>/nonpass/chr21.bed.gz"),
                                                          Path("<split>/S.psmcfa.gz"), Path("<split>/S.qc.json"))}

# the download
free = shutil.disk_usage(LOCAL).free
say(f"{free / 1e9:.1f} GB free in {LOCAL}; download {size / 1e9:.2f} GB from {url}")
if free < 2.2 * size:
    raise SystemExit(f"not enough node-local space: {free / 1e9:.1f} GB (fetch_ranges needs the parts and the file)")
dest = LOCAL / "sources" / Path(url).name
t0 = time.time()
dig = kg1000.fetch_ranges(url, dest, size, cfg.fetch_parts, cfg.chunk_attempts * 10, cfg.retry_wait_seconds)
fetch_s = time.time() - t0
rec["fetch"] = {"url": url, "parts": cfg.fetch_parts, "seconds": round(fetch_s, 3),
                "mb_per_s": round(size / fetch_s / 1e6, 2), **dig, "manifest_bytes": size, "manifest_md5": md5}
say(f"fetched in {fetch_s:.0f} s: {dig}")
if dig["bytes"] != size or dig["md5"] != md5:
    dest.unlink(missing_ok=True)
    (OUT / "pilot_chr21.partial.json").write_text(json.dumps(rec, indent=1) + "\n")
    raise SystemExit("the download does not match the manifest; deleted")

# the direct route: one stream, three converts
work = LOCAL / "pilot_chr21"
work.mkdir(parents=True, exist_ok=True)
stream = ["bcftools", "view", "--no-version", "--threads", str(cfg.stream_threads), "-r", region,
          "--regions-overlap", "pos", "-Ov", f"{dest}##idx##{I / 'meta' / 'tbi' / f'{CHROM}.vcf.gz.tbi'}"]
converts = {s: kg1000.convert_command(cfg, Path("/dev/stdin"), I / "mask" / f"{CHROM}.bed", None,
                                      work / f"{s}.psmcfa.gz", work / f"{s}.qc.json", sample=s) for s in samples}
rec["commands"].update({"stream": " ".join(stream), "convert": " ".join(converts[samples[0]])})
say("stream: " + " ".join(stream))
t0 = time.time()
kg1000.tee(stream, converts, work)
stream_s = time.time() - t0
rec["stream_seconds_pilot"] = round(stream_s, 3)
say(f"streamed and converted in {stream_s:.0f} s")
dest.unlink()
say("download deleted")

qc_rows = {r["sample_id"]: r for r in kg1000.read_tsv(I / "qc" / f"{CHROM}.tsv")}
direct_dir = OUT / "direct"
direct_dir.mkdir(exist_ok=True)
for s in samples:
    dgz = (work / f"{s}.psmcfa.gz").read_bytes()
    d = gzip.decompress(dgz)
    dq = kg1000.qc_numbers(json.loads((work / f"{s}.qc.json").read_text()))
    sp, sq, ssha, how = split[s]
    e7gz = (I / "psmcfa" / CHROM / f"{s}.psmcfa.gz").read_bytes()
    e7 = gzip.decompress(e7gz)
    row = qc_rows[s]
    rec["samples"].append({
        "sample_id": s,
        # direct against split, both with the current code (the pilot's claim)
        "psmcfa_identical": d == sp,
        "psmcfa_bytes": len(d),
        "qc_fields_differing": [k for k, v in dq.items() if v != sq[k]],
        "qc": dq,
        # each against E7's chr21 file and QC row (written by src 4c0d777)
        "e7": {
            "direct_psmcfa_identical": d == e7,
            "split_psmcfa_identical": sp == e7,
            "direct_qc_fields_differing": [k for k, v in dq.items() if str(v) != row[k]],
            "split_qc_fields_differing": [k for k, v in sq.items() if str(v) != row[k]],
            "gz_sha256_e7": sha(e7gz),
            "gz_sha256_recorded": row["psmcfa_sha256"],
        },
        "gz_sha256_direct": sha(dgz),
        "gz_sha256_split": ssha,
        "split_route": how,
    })
    shutil.copy(work / f"{s}.psmcfa.gz", direct_dir / f"{s}.psmcfa.gz")
    shutil.copy(work / f"{s}.qc.json", direct_dir / f"{s}.qc.json")
rec["identical"] = all(r["psmcfa_identical"] and not r["qc_fields_differing"] for r in rec["samples"])
rec["identical_to_e7"] = all(r["e7"]["direct_psmcfa_identical"] and r["e7"]["split_psmcfa_identical"]
                             and not r["e7"]["direct_qc_fields_differing"] and not r["e7"]["split_qc_fields_differing"]
                             for r in rec["samples"])
rec["supersedes"] = ("experiment/runs/E7/inputs/records/pilot_chr21.json and MANIFEST.json pilots.chr21 "
                     "(SLURM 53449988, 2026-10-08, psmc-rs source_commit 0c84f059, src 4c0d777, binary 2085fe3a)")
shutil.rmtree(work, ignore_errors=True)
(OUT / "pilot_chr21.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
for r in rec["samples"]:
    say(f"{r['sample_id']}: direct {'=' if r['psmcfa_identical'] else '!='} split; QC "
        f"{'equal' if not r['qc_fields_differing'] else r['qc_fields_differing']}; E7 chr21: direct "
        f"{r['e7']['direct_psmcfa_identical']}, split {r['e7']['split_psmcfa_identical']}")
say(f"identical {rec['identical']}, identical_to_e7 {rec['identical_to_e7']}")
