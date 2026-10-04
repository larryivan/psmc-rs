#!/usr/bin/env python3
"""Small, independent helpers for make_fixtures.sh (standard library only).

  helpers.py auto-depth VCF
      Print "mean min max": the mean INFO/DP over the records that enter
      vcfutils.pl vcf2fq's SNP/reference branch (one-character REF, no INDEL
      in INFO, ALT matching ^([A-Za-z.])(,[A-Za-z])*$), and the depth bounds
      max(1, round(mean / 3)) and round(2 * mean). Rounding is half away from
      zero, as in psmc-rs.

  helpers.py callable-mask VCF OUTDIR
      Write OUTDIR/mask_<chrom>.bed: the positions bamCaller.py (msmc-tools)
      calls covered, i.e. one-base ACGT REF, ALT one of ACGTacgt., and
      mean/2 <= DP <= 2*mean, MQ >= 20, |FQ| >= 20, as BED intervals.

  helpers.py phase VCF
      Print a VCF with every heterozygous 0/1 genotype written as phased,
      0|1 and 1|0 in turn, and every other genotype unchanged.
"""

import gzip
import math
import re
import sys

SNP_OR_REF = re.compile(r"^([A-Za-z.])(,[A-Za-z])*$")


def records(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                yield line.rstrip("\n").split("\t")


def round_half_away(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def mean_branch_depth(path):
    total = n = 0
    for t in records(path):
        if len(t[3]) == 1 and "INDEL" not in t[7] and SNP_OR_REF.match(t[4]):
            m = re.search(r"DP=(\d+)", t[7])
            if m:
                total += int(m.group(1))
                n += 1
    return total / n


def auto_depth(path):
    mean = mean_branch_depth(path)
    print(f"{mean!r} {max(1, round_half_away(mean / 3))} {round_half_away(2 * mean)}")


def callable_mask(path, outdir):
    mean = mean_branch_depth(path)
    lo, hi = mean / 2, mean * 2
    intervals = {}
    order = []
    for t in records(path):
        chrom, pos, ref, alt, info = t[0], int(t[1]), t[3], t[4], t[7]
        if chrom not in intervals:
            intervals[chrom] = []
            order.append(chrom)
        if not (re.match(r"^[ACTGactg]$", ref) and re.match(r"^[ACTGactg\.]$", alt)):
            continue
        dp = re.search(r"DP=(\d+)", info)
        mq = re.search(r"MQ=(\d+)", info)
        fq = re.search(r"FQ=([\d-]+)", info)
        if not (dp and mq and fq):
            continue
        if lo <= int(dp.group(1)) <= hi and int(mq.group(1)) >= 20 and abs(int(fq.group(1))) >= 20:
            iv = intervals[chrom]
            if iv and iv[-1][1] == pos - 1:
                iv[-1][1] = pos
            elif not iv or iv[-1][1] < pos:
                iv.append([pos, pos])
    for chrom in order:
        with open(f"{outdir}/mask_{chrom}.bed", "w") as f:
            for start, end in intervals[chrom]:
                f.write(f"{chrom}\t{start - 1}\t{end}\n")


def phase(path):
    flip = False
    with gzip.open(path, "rt") as f:
        for line in f:
            if not line.startswith("#"):
                t = line.rstrip("\n").split("\t")
                gt, _, rest = t[9].partition(":")
                if gt == "0/1":
                    gt = "1|0" if flip else "0|1"
                    flip = not flip
                    t[9] = gt + (":" + rest if rest else "")
                line = "\t".join(t) + "\n"
            sys.stdout.write(line)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"auto-depth": auto_depth, "callable-mask": callable_mask, "phase": phase}[cmd](*args)
