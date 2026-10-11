"""Step 1: are the laptop's E1 simulated genomes byte-identical to Ibex's?

The Ibex genomes are not committed; their committed truth records
(experiment/runs/E1/truth/<model>/repNN.json) hold the SHA-256 of each
genome's psmcfa and VCF.gz ('files') and, per chromosome, the SHA-256 of the
heterozygous positions (0-based, little-endian int64) with the counts of
bins, heterozygous sites, heterozygous bins and derived homozygous sites.
The committed Ibex provenance records also hold the SHA-256 of every sim
output (psmcfa, vcf.gz, tbi) and the input checksum each fit job read.

For each laptop genome (experiment/.production/experiment/runs/E1/sims):
  * whole-file SHA-256 of genome.psmcfa / genome.vcf.gz / .tbi vs the Ibex truth
    record ('files') and the Ibex sim provenance outputs;
  * per chromosome, recomputed from the laptop files: heterozygous positions
    (from the VCF) -> SHA-256 vs the Ibex truth record, het sites, derived hom
    sites, last het position; bins and K bins (from the psmcfa) vs bins, het_bins;
  * the input checksum of each fit job: Ibex fit-rs, Ibex fit-c, laptop fit-c.
NA12878 (E2): the checkout's real_data file vs the Ibex fit-rs and fit-c records.

Writes xplat/inputs_E1_genomes.csv, xplat/inputs_E1_chromosomes.csv, xplat/inputs.json.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # the repository root
X = Path(__file__).resolve().parent.parent
LAP = REPO / "experiment/.production/experiment/runs"
IBX = REPO / "experiment/runs"
MODELS = ("constant", "bottleneck", "expansion", "zigzag")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def positions_digest(positions) -> str:
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest()


def vcf_chroms(path: Path):
    """{chrom: (het positions 0-based, derived hom count)} in file order."""
    het, hom, order = {}, {}, []
    with gzip.open(path, "rt", encoding="ascii") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c, pos, _id, _ref, _alt, _q, _f, _i, _fmt, gt = line.rstrip("\n").split("\t")
            if c not in het:
                het[c], hom[c] = [], 0
                order.append(c)
            a, b = gt.split("|")
            if a != b:
                het[c].append(int(pos) - 1)
            else:
                hom[c] += 1
    return {c: (het[c], hom[c]) for c in order}


def psmcfa_chroms(path: Path):
    out, name, parts = {}, None, []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if name is not None:
        out[name] = "".join(parts)
    return {c: (len(s), s.count("K"), len(s) - s.count("K") - s.count("T")) for c, s in out.items()}


genome_rows, chrom_rows = [], []
for m in MODELS:
    for r in range(1, 11):
        rep = f"rep{r:02d}"
        sim = LAP / f"E1/sims/{m}/{rep}"
        fa, gz, tbi = sim / "genome.psmcfa", sim / "genome.vcf.gz", sim / "genome.vcf.gz.tbi"
        truth = json.loads((IBX / f"E1/truth/{m}/{rep}.json").read_text())
        lap_truth = json.loads((LAP / f"E1/truth/{m}/{rep}.json").read_text())
        ibx_sim_out = json.loads((IBX / f"E1/provenance/sim/{m}/{rep}.json").read_text())["outputs"]
        lap_sim_rec = json.loads((LAP / f"E1/provenance/sim/{m}/{rep}.json").read_text())
        ibx_rs_in = json.loads((IBX / f"E1/provenance/fit-rs/{m}/{rep}.json").read_text())["key"]["inputs"]
        ibx_c_in = json.loads((IBX / f"E1/provenance/fit-c/{m}/{rep}.json").read_text())["key"]["inputs"]
        lap_c_rec = json.loads((LAP / f"E1/provenance/fit-c/{m}/{rep}.json").read_text())
        lap_c_in = lap_c_rec["key"]["inputs"]
        key = f"<runs>/E1/sims/{m}/{rep}/genome"
        h_fa, h_gz, h_tbi = sha(fa), sha(gz), sha(tbi)
        # the laptop C fit really read this genome: its record's input checksum, and the .psmc header counts
        lap_c_out = LAP / f"E1/fits/{m}/{rep}.c.psmc"
        lap_c_out_ok = sha(lap_c_out) == lap_c_rec["outputs"][f"<runs>/E1/fits/{m}/{rep}.c.psmc"]
        # per chromosome
        vc = vcf_chroms(gz)
        pc = psmcfa_chroms(fa)
        n_ok = 0
        for ch in truth["chromosomes"]:
            name = ch["name"]
            het, hom = vc.get(name, ([], 0))
            bins, kbins, other = pc.get(name, (0, 0, -1))
            row = {
                "model": m, "replicate": r, "chromosome": name,
                "het_positions_sha256_laptop": positions_digest(het),
                "het_positions_sha256_ibex_truth": ch["het_positions_sha256"],
                "het_sha_match": positions_digest(het) == ch["het_positions_sha256"],
                "het_sites_laptop": len(het), "het_sites_ibex": ch["het_sites"],
                "derived_hom_laptop": hom, "derived_hom_ibex": ch["derived_hom_sites"],
                "last_het_laptop": het[-1] if het else None, "last_het_ibex": ch["last_het_position"],
                "bins_laptop": bins, "bins_ibex": ch["bins"],
                "het_bins_laptop": kbins, "het_bins_ibex": ch["het_bins"],
                "psmcfa_other_symbols": other,
            }
            row["all_match"] = (row["het_sha_match"] and row["het_sites_laptop"] == row["het_sites_ibex"]
                                and row["derived_hom_laptop"] == row["derived_hom_ibex"]
                                and row["last_het_laptop"] == row["last_het_ibex"]
                                and row["bins_laptop"] == row["bins_ibex"]
                                and row["het_bins_laptop"] == row["het_bins_ibex"] and other == 0)
            n_ok += row["all_match"]
            chrom_rows.append(row)
        genome_rows.append({
            "model": m, "replicate": r,
            "psmcfa_sha256_laptop": h_fa,
            "psmcfa_eq_ibex_truth": h_fa == truth["files"]["psmcfa"]["sha256"],
            "psmcfa_eq_ibex_sim_prov": h_fa == ibx_sim_out.get(key + ".psmcfa"),
            "psmcfa_eq_ibex_fit_rs_input": h_fa == ibx_rs_in.get(key + ".psmcfa"),
            "psmcfa_eq_ibex_fit_c_input": h_fa == ibx_c_in.get(key + ".psmcfa"),
            "psmcfa_eq_laptop_fit_c_input": h_fa == lap_c_in.get(key + ".psmcfa"),
            "laptop_c_output_eq_its_record": lap_c_out_ok,
            "vcf_gz_eq_ibex_truth": h_gz == truth["files"]["vcf"]["sha256"],
            "vcf_gz_eq_ibex_sim_prov": h_gz == ibx_sim_out.get(key + ".vcf.gz"),
            "tbi_eq_ibex_sim_prov": h_tbi == ibx_sim_out.get(key + ".vcf.gz.tbi"),
            "sim_record_key_equal": all(lap_sim_rec["key"].get(k) == json.loads((IBX / f"E1/provenance/sim/{m}/{rep}.json").read_text())["key"].get(k)
                                        for k in ("command", "seeds", "settings")),
            "truth_chromosomes_block_equal": lap_truth["chromosomes"] == truth["chromosomes"],
            "chromosomes_matching": n_ok, "chromosomes": len(truth["chromosomes"]),
        })

# NA12878
na = REPO / "experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz"
na_key = "<repo>/experiment/real_data/NA12878_1kGP30x_diploid.psmcfa.gz"
h_na = sha(na)
na_rs = json.loads((IBX / "E2/provenance/fit-rs/NA12878.json").read_text())["key"]["inputs"][na_key]
na_c = json.loads((IBX / "E2/provenance/fit-c/NA12878.json").read_text())["key"]["inputs"][na_key]

X.mkdir(parents=True, exist_ok=True)
with open(X / "inputs_E1_genomes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(genome_rows[0]))
    w.writeheader()
    w.writerows(genome_rows)
with open(X / "inputs_E1_chromosomes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(chrom_rows[0]))
    w.writeheader()
    w.writerows(chrom_rows)

bool_cols = [k for k, v in genome_rows[0].items() if isinstance(v, bool)]
summary = {
    "genomes": len(genome_rows),
    **{f"n_{k}": sum(r[k] for r in genome_rows) for k in bool_cols},
    "chromosomes": len(chrom_rows),
    "n_chromosomes_het_sha_match": sum(r["het_sha_match"] for r in chrom_rows),
    "n_chromosomes_all_counts_match": sum(r["all_match"] for r in chrom_rows),
    "NA12878": {"path": str(na.relative_to(REPO)), "sha256": h_na, "eq_ibex_fit_rs_input": h_na == na_rs,
                "eq_ibex_fit_c_input": h_na == na_c},
}
(X / "inputs.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
