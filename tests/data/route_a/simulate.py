#!/usr/bin/env python3
"""Simulate a small diploid genome and its read alignments for the route A
fixtures (see README.md).

Writes, into the output directory:
  ref.fa     the reference, three chromosomes
  aln.sam    coordinate-sorted single-end alignments of reads drawn from the
             two haplotypes, in SAM text (bcftools mpileup reads it directly)
  truth.tsv  the simulated variants (chrom, 1-based pos, REF, ALT, het/hom)

Everything is drawn from one seeded random.Random, so the output depends only
on the seed and the parameters below. Standard library only.

The genome is laid out to exercise every filter of the classic pipeline:
coverage dips (low depth) and a collapsed-repeat-like pile-up (high depth), a
region without reads (no records), a region whose reads have mapping quality
25 (MQ filter at -Q 30), reads of mapping quality 5 (dropped by
mpileup -q20), low base qualities (dropped by mpileup -Q20), sequencing
errors, heterozygous and homozygous SNPs, and heterozygous and homozygous
insertions and deletions (INDEL windows).
"""

import argparse
import os
import random

SEED = 20261005
READ_LEN = 100
ERROR_RATE = 0.004
LOW_BASEQ_RATE = 0.05

# (name, length, coverage segments as (start, end, depth, mapq), 1-based).
# chr1 is just long enough to pass fq2psmcfa's default output filter (10,000
# called bases); the others are dropped by it.
CHROMS = [
    ("chr1", 11500, [
        (1, 2499, 30, 60),
        (2500, 3199, 4, 60),
        (3200, 4999, 30, 60),
        (5000, 5399, 0, 60),
        (5400, 6999, 30, 60),
        (7000, 7699, 90, 60),
        (7700, 8999, 30, 60),
        (9000, 9799, 30, 25),
        (9800, 11500, 30, 60),
    ]),
    ("chr2", 3000, [
        (1, 999, 30, 60),
        (1000, 1599, 12, 60),
        (1600, 3000, 30, 60),
    ]),
    ("chr3", 1800, [
        (1, 1800, 8, 60),
    ]),
]
HET_SNP_RATE = 1 / 250
HOM_SNP_RATE = 1 / 2000
INDELS_PER_KB = 0.4
# Extra reads of mapping quality 5 per position, dropped by mpileup -q20.
LOW_MAPQ_DEPTH = 2

BASES = "ACGT"


def other_base(rng, b):
    return rng.choice([x for x in BASES if x != b])


def simulate_variants(rng, ref):
    """Variants of each haplotype as {pos0: (ref_allele, alt_allele)}."""
    haps = [{}, {}]
    truth = []
    n = len(ref)
    pos = 10
    while pos < n - 10:
        u = rng.random()
        if u < HET_SNP_RATE:
            h = rng.randrange(2)
            alt = other_base(rng, ref[pos])
            haps[h][pos] = (ref[pos], alt)
            truth.append((pos, ref[pos], alt, "het"))
        elif u < HET_SNP_RATE + HOM_SNP_RATE:
            alt = other_base(rng, ref[pos])
            haps[0][pos] = haps[1][pos] = (ref[pos], alt)
            truth.append((pos, ref[pos], alt, "hom"))
        elif u < HET_SNP_RATE + HOM_SNP_RATE + INDELS_PER_KB / 1000:
            size = rng.randint(1, 4)
            if rng.random() < 0.5:
                ref_allele, alt = ref[pos:pos + size + 1], ref[pos]
            else:
                ref_allele = ref[pos]
                alt = ref[pos] + "".join(rng.choice(BASES) for _ in range(size))
            kind = "het" if rng.random() < 0.8 else "hom"
            for h in ([rng.randrange(2)] if kind == "het" else [0, 1]):
                haps[h][pos] = (ref_allele, alt)
            truth.append((pos, ref_allele, alt, kind))
            pos += size + 2
        pos += 1
    return haps, truth


def build_haplotype(ref, variants):
    """The haplotype sequence and, per base, its 0-based reference position
    (-1 for inserted bases)."""
    seq, refpos = [], []
    i = 0
    while i < len(ref):
        if i in variants:
            ref_allele, alt = variants[i]
            if len(ref_allele) == len(alt) == 1:
                seq.append(alt)
                refpos.append(i)
                i += 1
                continue
            seq.append(alt[0])
            refpos.append(i)
            for b in alt[1:]:
                seq.append(b)
                refpos.append(-1)
            i += len(ref_allele)
            continue
        seq.append(ref[i])
        refpos.append(i)
        i += 1
    # Haplotype index of each reference position; positions deleted on this
    # haplotype map to the next base.
    absent = len(seq)
    first_at = [absent] * len(ref)
    for j, rp in enumerate(refpos):
        if rp >= 0:
            first_at[rp] = j
    nxt = absent
    for p in range(len(ref) - 1, -1, -1):
        if first_at[p] == absent:
            first_at[p] = nxt
        else:
            nxt = first_at[p]
    return "".join(seq), refpos, first_at


def cigar_of(refpos):
    ops = []

    def push(op, n=1):
        if ops and ops[-1][0] == op:
            ops[-1][1] += n
        else:
            ops.append([op, n])

    prev = None
    for rp in refpos:
        if rp < 0:
            push("I")
            continue
        if prev is not None and rp > prev + 1:
            push("D", rp - prev - 1)
        push("M")
        prev = rp
    return "".join(f"{n}{op}" for op, n in ops)


def make_read(rng, hap, start0):
    seq, refpos, first_at = hap
    h = first_at[start0]
    if h + READ_LEN > len(seq):
        return None
    # Reads must start and end on a reference base.
    while h < len(seq) and refpos[h] < 0:
        h += 1
    if h + READ_LEN > len(seq) or refpos[h + READ_LEN - 1] < 0:
        return None
    bases = list(seq[h:h + READ_LEN])
    quals = []
    for k in range(READ_LEN):
        if rng.random() < ERROR_RATE:
            bases[k] = other_base(rng, bases[k])
        quals.append("+" if rng.random() < LOW_BASEQ_RATE else "I")
    return refpos[h], cigar_of(refpos[h:h + READ_LEN]), "".join(bases), "".join(quals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    fa, sam_header, reads, truth_rows = [], ["@HD\tVN:1.6\tSO:coordinate"], [], []
    for name, length, segments in CHROMS:
        ref = "".join(rng.choice(BASES) for _ in range(length))
        fa.append(f">{name}")
        fa.extend(ref[i:i + 60] for i in range(0, length, 60))
        sam_header.append(f"@SQ\tSN:{name}\tLN:{length}")
        variants, truth = simulate_variants(rng, ref)
        truth_rows.extend(f"{name}\t{p + 1}\t{r}\t{a}\t{k}" for p, r, a, k in truth)
        haps = [build_haplotype(ref, v) for v in variants]
        plan = [(s, e, d, q) for s, e, d, q in segments]
        plan.append((1, length, LOW_MAPQ_DEPTH, 5))
        for start, end, depth, mapq in plan:
            n = round(depth * (end - start + 1) / READ_LEN)
            for _ in range(n):
                s0 = rng.randint(start - 1, max(start - 1, end - READ_LEN))
                read = make_read(rng, haps[rng.randrange(2)], s0)
                if read is None:
                    continue
                pos0, cigar, bases, quals = read
                flag = 16 if rng.random() < 0.5 else 0
                reads.append((name, pos0, cigar, bases, quals, mapq, flag))

    order = {name: i for i, (name, _, _) in enumerate(CHROMS)}
    reads.sort(key=lambda r: (order[r[0]], r[1]))
    with open(os.path.join(args.outdir, "ref.fa"), "w") as f:
        f.write("\n".join(fa) + "\n")
    with open(os.path.join(args.outdir, "aln.sam"), "w") as f:
        f.write("\n".join(sam_header) + "\n")
        for i, (name, pos0, cigar, bases, quals, mapq, flag) in enumerate(reads):
            f.write(f"r{i}\t{flag}\t{name}\t{pos0 + 1}\t{mapq}\t{cigar}\t*\t0\t0\t{bases}\t{quals}\n")
    with open(os.path.join(args.outdir, "truth.tsv"), "w") as f:
        f.write("\n".join(truth_rows) + "\n")


if __name__ == "__main__":
    main()
