#!/usr/bin/env python3
"""Simulate a small diploid genome and its read alignments for the
end-to-end test of scripts/bam_to_psmc.sh (see README.md).

Writes, into the output directory:
  ref.fa         the reference: three autosomes, an unplaced scaffold, chrX
                 and chrM
  aln.sam        coordinate-sorted single-end alignments in SAM text, which
                 bcftools mpileup reads directly; one read group, sample sim1
  truth.tsv      the simulated variants: chrom, 1-based pos, REF, ALT,
                 het/hom, snp/indel
  design.tsv     the coverage layout: chrom, 1-based start, end, kind, depth,
                 mapping quality
  callable.bed   where the pipeline must call every base: ordinary coverage,
                 away from segment edges and indels (BED, 0-based)
  masked.bed     where it must call nothing: coverage below or above the
                 depth bounds, no reads, unmappable reads (BED, 0-based)

The sample is male: chrX is a single copy outside an 8 kb pseudoautosomal
region, at half the autosomal depth, so its only heterozygous calls are
errors; chrM is a single copy at high depth. Heterozygosity varies along the
genome in segments, as local genealogies do. Coverage is laid out so that
every filter of the classic pipeline acts:

  chr1  a dip to 2x (below a third of the mean), a stretch without reads, a
        collapsed repeat (100x of reads from a 1.5% divergent paralog piled
        on top; mpileup -C50 lowers the mapping quality of many of them,
        and what remains is still above twice the mean, with differences
        that look heterozygous), and a soft-masked repeat whose reads have
        mapping quality 0 (dropped by mpileup -q20)
  chr2  a dip to 12x (within the bounds) and an assembly gap (N in the
        reference)
  chr3  plain 25x
  chrUn_scaffold1  8 kb: fewer than fq2psmcfa's 10,000 called bases

Everywhere, 2x of extra reads have mapping quality 5 (dropped by -q20), 5%
of bases have quality 10 (dropped by -Q20) and 0.3% are sequencing errors.

--depth and --paralog-depth change the base depth (25) and the paralog
depth of the collapsed repeat (100), and --chroms writes only the listed
sequences; run_e2e.sh uses them for a high-depth run of chr1.

Everything is drawn from one seeded random.Random, so the output depends
only on the seed and the options; random.Random's sequence is stable across
Python 3 releases. Standard library only.
"""

import argparse
import bisect
import math
import os
import random

SEED = 20261005
READ_LEN = 100
DEPTH = 25
MAPQ = 60
ERROR_RATE = 0.003
LOW_BASEQ_RATE = 0.05
LOW_MAPQ_DEPTH = 2
# Mean heterozygosity per bp; each segment of mean length HET_SEGMENT_BP
# gets this times an exponential draw, as if from its own TMRCA.
HET_RATE = 0.0015
HET_SEGMENT_BP = 20_000
HOM_SNP_RATE = 1 / 4000
INDELS_PER_KB = 0.15
PARALOG_DIVERGENCE = 0.015
# Extra depth of paralog reads in the collapsed repeat.
PARALOG_DEPTH = 100
# Indels and segment edges are kept this far apart, so that reads never
# carry an indel across a segment boundary.
EDGE_GAP = 50
# Margin around indels left out of callable.bed: vcf2fq's window (-l 5) plus
# the length of the event plus some slack for BAQ.
INDEL_MARGIN = 15
PAR_END = 8_000

BASES = "ACGT"


# name, length, ploidy outside the PAR ("par" for chrX), base depth, special
# segments as (1-based start, end, kind, depth, mapq); `depth` is the base
# depth of the nuclear sequences.
def chromosomes(depth=DEPTH):
    return [
        ("chr1", 150_000, 2, depth, [
            (30_001, 33_000, "low_depth", 2, MAPQ),
            (60_001, 62_000, "no_reads", 0, MAPQ),
            (90_001, 92_500, "collapsed_repeat", depth, MAPQ),
            (120_001, 123_000, "unmappable", depth, 0),
        ]),
        ("chr2", 100_000, 2, depth, [
            (40_001, 45_000, "dip", 12, MAPQ),
            (70_001, 71_000, "assembly_gap", 0, MAPQ),
        ]),
        ("chr3", 70_000, 2, depth, []),
        ("chrUn_scaffold1", 8_000, 2, depth, []),
        ("chrX", 80_000, "par", depth, []),
        ("chrM", 3_000, 1, 150, []),
    ]


# Kinds whose positions must come out missing, and kinds that are callable.
MASKED_KINDS = {"low_depth", "no_reads", "collapsed_repeat", "unmappable", "assembly_gap"}
CALLABLE_KINDS = {"normal", "dip"}


def geometric_positions(rng, p, n):
    """0-based offsets below n, each included independently with
    probability p (drawn by geometric jumps, which is much faster than one
    draw per position)."""
    out = []
    if p <= 0:
        return out
    log_q = math.log1p(-p)
    i = -1
    while True:
        i += 1 + int(math.log(1.0 - rng.random()) / log_q)
        if i >= n:
            return out
        out.append(i)


def other_base(rng, b):
    return rng.choice([x for x in BASES if x != b.upper()])


def layout(length, depth, specials):
    """The chromosome as contiguous segments (start0, end0, kind, depth,
    mapq), 0-based half-open."""
    segs, pos = [], 0
    for start, end, kind, d, q in specials:
        if start - 1 > pos:
            segs.append((pos, start - 1, "normal", depth, MAPQ))
        segs.append((start - 1, end, kind, d, q))
        pos = end
    if pos < length:
        segs.append((pos, length, "normal", depth, MAPQ))
    return segs


def simulate_variants(rng, ref, segs, diploid_at):
    """Variants of the two haplotypes as {pos0: (ref_allele, alt)}, and the
    truth rows. Where diploid_at(pos0) is false the sample has one copy, so
    every variant is on both haplotypes (hemizygous)."""
    n = len(ref)
    boundaries = sorted({s for s, _, _, _, _ in segs} | {e for _, e, _, _, _ in segs})
    haps = [{}, {}]
    truth = []
    het_rate, seg_end = 0.0, 0
    pos = 10
    while pos < n - 10:
        if pos >= seg_end:
            seg_end = pos + 1 + int(rng.expovariate(1 / HET_SEGMENT_BP))
            het_rate = HET_RATE * rng.expovariate(1.0)
        base = ref[pos].upper()
        if base == "N":
            pos += 1
            continue
        diploid = diploid_at(pos)
        u = rng.random()
        if diploid and u < het_rate:
            h = rng.randrange(2)
            alt = other_base(rng, base)
            haps[h][pos] = (base, alt)
            truth.append((pos, base, alt, "het", "snp"))
        elif u < het_rate + HOM_SNP_RATE:
            alt = other_base(rng, base)
            haps[0][pos] = haps[1][pos] = (base, alt)
            truth.append((pos, base, alt, "hom", "snp"))
        elif u < het_rate + HOM_SNP_RATE + INDELS_PER_KB / 1000:
            size = rng.randint(1, 6)
            i = bisect.bisect_left(boundaries, pos - EDGE_GAP)
            near_edge = i < len(boundaries) and boundaries[i] <= pos + size + EDGE_GAP
            if near_edge or "N" in ref[pos:pos + size + 1].upper():
                pos += 1
                continue
            if rng.random() < 0.5:
                ref_allele, alt = ref[pos:pos + size + 1].upper(), base
            else:
                ref_allele = base
                alt = base + "".join(rng.choice(BASES) for _ in range(size))
            kind = "het" if diploid and rng.random() < 0.8 else "hom"
            for h in ([rng.randrange(2)] if kind == "het" else [0, 1]):
                haps[h][pos] = (ref_allele, alt)
            truth.append((pos, ref_allele, alt, kind, "indel"))
            pos += size + 2
        pos += 1
    return haps, truth


class Haplotype:
    """A haplotype sequence with, per base, its 0-based reference position
    (-1 for inserted bases)."""

    def __init__(self, ref, variants):
        seq, refpos = [], []
        i = 0
        upper = ref.upper()
        while i < len(ref):
            if i in variants:
                ref_allele, alt = variants[i]
                seq.append(alt[0])
                refpos.append(i)
                for b in alt[1:]:
                    seq.append(b)
                    refpos.append(-1)
                i += len(ref_allele)
                continue
            seq.append(upper[i])
            refpos.append(i)
            i += 1
        self.seq = "".join(seq)
        self.refpos = refpos
        # Haplotype index of each reference position; positions deleted on
        # this haplotype map to the next base.
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
        self.first_at = first_at
        # Indices where the alignment is not a plain match continuing the
        # previous base: inserted bases and the base after a deletion.
        self.irregular = [
            j for j in range(1, len(refpos))
            if refpos[j] < 0 or refpos[j - 1] < 0 or refpos[j] != refpos[j - 1] + 1
        ]

    def cigar(self, h):
        """CIGAR of the read at haplotype indices h .. h + READ_LEN."""
        k = bisect.bisect_right(self.irregular, h)
        if k == len(self.irregular) or self.irregular[k] >= h + READ_LEN:
            return f"{READ_LEN}M"
        ops = []

        def push(op, n=1):
            if ops and ops[-1][0] == op:
                ops[-1][1] += n
            else:
                ops.append([op, n])

        prev = None
        for rp in self.refpos[h:h + READ_LEN]:
            if rp < 0:
                push("I")
                continue
            if prev is not None and rp > prev + 1:
                push("D", rp - prev - 1)
            push("M")
            prev = rp
        return "".join(f"{n}{op}" for op, n in ops)

    def read(self, start0):
        """A read starting at reference position start0, as (pos0, cigar,
        bases), or None if it would run off the haplotype. Reads start and
        end on a reference base."""
        h = self.first_at[start0]
        while h < len(self.seq) and self.refpos[h] < 0:
            h += 1
        if h + READ_LEN > len(self.seq) or self.refpos[h + READ_LEN - 1] < 0:
            return None
        return self.refpos[h], self.cigar(h), self.seq[h:h + READ_LEN]


def sequence_read(rng, bases):
    """Add sequencing errors and base qualities to read bases."""
    bases = list(bases)
    for k in geometric_positions(rng, ERROR_RATE, READ_LEN):
        bases[k] = other_base(rng, bases[k])
    quals = ["I"] * READ_LEN
    for k in geometric_positions(rng, LOW_BASEQ_RATE, READ_LEN):
        quals[k] = "+"
    return "".join(bases), "".join(quals)


def paralog_of(rng, ref_segment):
    """A diverged copy of a reference segment (substitutions only)."""
    s = list(ref_segment.upper())
    for k in geometric_positions(rng, PARALOG_DIVERGENCE, len(s)):
        s[k] = other_base(rng, s[k])
    return "".join(s)


def reads_over(rng, start0, end0, depth, draw):
    """round(depth * length / READ_LEN) reads starting uniformly in
    [start0, end0 - READ_LEN]; draw(start0) gives one or None."""
    out = []
    n = round(depth * (end0 - start0) / READ_LEN)
    last = max(start0, end0 - READ_LEN)
    for _ in range(n):
        r = draw(rng.randint(start0, last))
        if r is not None:
            out.append(r)
    return out


def shrink(intervals, margin):
    """Intervals with margin bases removed at each end (dropped if empty)."""
    return [(s + margin, e - margin) for s, e in intervals if e - s > 2 * margin]


def subtract(intervals, holes):
    """intervals minus holes; both lists of sorted (start0, end0)."""
    out = []
    holes = sorted(holes)
    for s, e in intervals:
        cur = s
        for hs, he in holes:
            if he <= cur or hs >= e:
                continue
            if hs > cur:
                out.append((cur, hs))
            cur = max(cur, he)
        if cur < e:
            out.append((cur, e))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help=f"base depth of the nuclear sequences [{DEPTH}]")
    ap.add_argument("--paralog-depth", type=int, default=PARALOG_DEPTH,
                    help=f"depth of paralog reads in the collapsed repeat [{PARALOG_DEPTH}]")
    ap.add_argument("--chroms", help="comma-separated sequences to write [all]")
    args = ap.parse_args()
    chroms = chromosomes(args.depth)
    if args.chroms:
        wanted = args.chroms.split(",")
        unknown = set(wanted) - {c[0] for c in chroms}
        if unknown:
            ap.error(f"unknown --chroms: {sorted(unknown)}")
        chroms = [c for c in chroms if c[0] in wanted]
    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    fa = []
    sam_header = ["@HD\tVN:1.6\tSO:coordinate"]
    reads, truth_rows, design_rows, callable_rows, masked_rows = [], [], [], [], []
    for name, length, ploidy, depth, specials in chroms:
        segs = layout(length, depth, specials)
        chars = [rng.choice(BASES) for _ in range(length)]
        for s, e, kind, _, _ in segs:
            if kind == "assembly_gap":
                chars[s:e] = ["N"] * (e - s)
            elif kind == "unmappable":
                chars[s:e] = [c.lower() for c in chars[s:e]]
        ref = "".join(chars)
        fa.append(f">{name}")
        fa.extend(ref[i:i + 60] for i in range(0, length, 60))
        sam_header.append(f"@SQ\tSN:{name}\tLN:{length}")

        if ploidy == "par":
            def diploid_at(p0):
                return p0 < PAR_END
        else:
            def diploid_at(p0, diploid=(ploidy == 2)):
                return diploid
        variants, truth = simulate_variants(rng, ref, segs, diploid_at)
        truth_rows.extend(f"{name}\t{p + 1}\t{r}\t{a}\t{z}\t{t}" for p, r, a, z, t in truth)
        haps = [Haplotype(ref, v) for v in variants]

        def draw(start0, haps=haps, mapq=MAPQ):
            r = haps[rng.randrange(2)].read(start0)
            return None if r is None else r + (mapq,)

        # A haploid stretch has half the reads (the second haplotype is a
        # copy of the first there, so drawing from either is the same).
        cover = []
        for s, e, kind, d, q in segs:
            if ploidy == "par" and e > PAR_END:
                if s < PAR_END:
                    cover.append((s, PAR_END, kind, d, q))
                    s = PAR_END
                cover.append((s, e, kind, d / 2, q))
            else:
                cover.append((s, e, kind, d, q))
        for s, e, kind, d, q in cover:
            design_rows.append(f"{name}\t{s + 1}\t{e}\t{kind}\t{d:g}\t{q}")
            reads.extend((name,) + r for r in reads_over(
                rng, s, e, d, lambda p, q=q: draw(p, mapq=q)))
            if kind == "collapsed_repeat":
                para = Haplotype(paralog_of(rng, ref[s:e]), {})

                def draw_paralog(p, para=para, s=s):
                    r = para.read(p - s)
                    return None if r is None else (r[0] + s, r[1], r[2], MAPQ)
                reads.extend((name,) + r for r in reads_over(
                    rng, s, e, args.paralog_depth, draw_paralog))
        reads.extend((name,) + r for r in reads_over(
            rng, 0, length, LOW_MAPQ_DEPTH, lambda p: draw(p, mapq=5)))

        # Expected calls: segments (merged where neighbours share a class)
        # shrunk by a read length at their edges, where depth ramps.
        def runs(kinds):
            out = []
            for s, e, kind, _, _ in cover:
                if kind in kinds:
                    if out and out[-1][1] == s:
                        out[-1] = (out[-1][0], e)
                    else:
                        out.append((s, e))
            return out
        indels = [(p - INDEL_MARGIN, p + len(r) + INDEL_MARGIN)
                  for p, r, a, z, t in truth if t == "indel"]
        good = subtract(shrink(runs(CALLABLE_KINDS), READ_LEN), indels)
        callable_rows.extend(f"{name}\t{s}\t{e}" for s, e in good)
        masked_rows.extend(f"{name}\t{s}\t{e}" for s, e in shrink(runs(MASKED_KINDS), READ_LEN))

    order = {name: i for i, (name, *_) in enumerate(chroms)}
    reads.sort(key=lambda r: (order[r[0]], r[1]))

    def write(name, lines):
        with open(os.path.join(args.outdir, name), "w") as f:
            f.write("".join(line + "\n" for line in lines))

    write("ref.fa", fa)
    sam = sam_header + ["@RG\tID:run1\tSM:sim1", "@PG\tID:simulate\tPN:simulate.py"]
    for i, (name, pos0, cigar, bases, mapq) in enumerate(reads):
        seq, qual = sequence_read(rng, bases)
        flag = 16 if rng.random() < 0.5 else 0
        sam.append(f"r{i}\t{flag}\t{name}\t{pos0 + 1}\t{mapq}\t{cigar}\t*\t0\t0\t{seq}\t{qual}\tRG:Z:run1")
    write("aln.sam", sam)
    write("truth.tsv", ["chrom\tpos\tref\talt\tzygosity\ttype"] + truth_rows)
    write("design.tsv", ["chrom\tstart\tend\tkind\tdepth\tmapq"] + design_rows)
    write("callable.bed", callable_rows)
    write("masked.bed", masked_rows)


if __name__ == "__main__":
    main()
