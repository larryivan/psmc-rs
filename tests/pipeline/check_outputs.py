#!/usr/bin/env python3
"""Checks of scripts/bam_to_psmc.sh outputs against the simulation truth,
for run_e2e.sh (see README.md). Standard library only.

  check_outputs.py recovery SIM_DIR PSMCFA BIN_SIZE [--min-recovery R]
                   [--max-false-k F] [--min-masked-n M]
      Heterozygous SNPs of the truth that lie in callable.bed must fall in K
      bins (recovery), K bins must hold a simulated heterozygous variant
      (false K share), and bins inside masked.bed must be N. Prints the
      rates; exits 1 when one misses its threshold.

  check_outputs.py multihetsep SIM_DIR PSMCFA BIN_SIZE FILE... [--min-true T]
      Every multihetsep row lies in a K bin of the psmcfa, has a positive
      nr_called and two different alleles, and at least a share T of the
      rows are simulated heterozygous SNPs.

  check_outputs.py region SIM_DIR PSMCFA BIN_SIZE KIND [--max-k F]
      The bins lying wholly inside the design.tsv segments of one kind (for
      example collapsed_repeat): prints how many are K, T and N, and exits 1
      when the K share is above F.

  check_outputs.py qc QC_JSON [--sex-warning NAME] [--kept NAME,...]
                   [--dropped N] [--format TEXT] [--has CODE] [--lacks CODE]
      Without --sex-warning, no SEX_OR_ORGANELLE finding and no sequence
      named like a sex chromosome or organelle; with it, a SEX_OR_ORGANELLE
      warning naming exactly NAME as hemizygous. --kept lists the
      sequences the report must hold, in order; --dropped how many the
      output filter dropped; --format how the input was read (default
      "vcf, consensus mode"); --has and --lacks (repeatable) findings that
      must or must not be there.

  check_outputs.py fit JSON PSMC N_ITER SEQUENCES
      The inference JSON holds N_ITER iterations and N_ITER + 1 finite
      log-likelihoods, read SEQUENCES psmcfa records, and the .psmc file
      has rounds 0 to N_ITER.

  check_outputs.py kinds DRY_RUN_OUTPUT QC_JSON
      The names the script's dry run leaves out (and why) are those
      psmc-rs qc classifies as sex chromosomes or organelles.
"""

import argparse
import json
import math
import sys


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def read_psmcfa(path):
    seqs, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def read_bed(path):
    out = {}
    with open(path) as f:
        for line in f:
            chrom, s, e = line.split()[:3]
            out.setdefault(chrom, []).append((int(s), int(e)))
    return out


def inside(intervals, pos0):
    return any(s <= pos0 < e for s, e in intervals)


def read_truth(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            chrom, pos, ref, alt, zyg, typ = line.rstrip("\n").split("\t")
            rows.append((chrom, int(pos), ref, alt, zyg, typ))
    return rows


def recovery(args):
    seqs = read_psmcfa(args.psmcfa)
    truth = read_truth(f"{args.sim}/truth.tsv")
    callable_ = read_bed(f"{args.sim}/callable.bed")
    masked = read_bed(f"{args.sim}/masked.bed")
    s = args.bin_size
    hets = [(c, p) for c, p, _, _, z, t in truth if z == "het" and t == "snp" and c in seqs]
    het_bins = {}
    for c, p, ref, alt, z, t in truth:
        if z == "het" and c in seqs:
            # An indel can reach into the next bin.
            for q in range(p, p + max(len(ref), len(alt))):
                het_bins.setdefault(c, set()).add((q - 1) // s)
    target = [(c, p) for c, p in hets if inside(callable_.get(c, []), p - 1)]
    found = [(c, p) for c, p in target if seqs[c][(p - 1) // s] == "K"]
    k_bins = [(c, i) for c, seq in seqs.items() for i, b in enumerate(seq) if b == "K"]
    false_k = [(c, i) for c, i in k_bins if i not in het_bins.get(c, set())]
    masked_bins = [
        (c, i) for c, seq in seqs.items() for i in range(len(seq))
        if any(ms <= i * s and (i + 1) * s <= me for ms, me in masked.get(c, []))
    ]
    masked_n = [(c, i) for c, i in masked_bins if seqs[c][i] == "N"]
    rec = len(found) / len(target) if target else 0.0
    fk = len(false_k) / len(k_bins) if k_bins else 1.0
    mn = len(masked_n) / len(masked_bins) if masked_bins else 0.0
    print(f"heterozygous SNPs in callable regions: {len(target)} of {len(hets)}; "
          f"in K bins: {len(found)} (recovery {rec:.4f}, threshold {args.min_recovery})")
    print(f"K bins without a simulated heterozygous variant: {len(false_k)} of {len(k_bins)} "
          f"({fk:.4f}, threshold {args.max_false_k})")
    print(f"bins inside masked regions: {len(masked_bins)}, N: {len(masked_n)} "
          f"({mn:.4f}, threshold {args.min_masked_n})")
    missed = [f"{c}:{p}" for c, p in target if (c, p) not in set(found)]
    if missed:
        print(f"missed: {' '.join(missed[:20])}")
    if len(target) < 100:
        fail(f"only {len(target)} heterozygous SNPs to recover; the simulation is too small")
    if not masked_bins:
        fail("no bin lies inside the masked regions")
    if rec < args.min_recovery:
        fail(f"recovery {rec:.4f} < {args.min_recovery}")
    if fk > args.max_false_k:
        fail(f"false K share {fk:.4f} > {args.max_false_k}")
    if mn < args.min_masked_n:
        fail(f"masked N share {mn:.4f} < {args.min_masked_n}")


def region(args):
    seqs = read_psmcfa(args.psmcfa)
    s = args.bin_size
    counts = {"K": 0, "T": 0, "N": 0}
    with open(f"{args.sim}/design.tsv") as f:
        next(f)
        for line in f:
            chrom, start, end, kind = line.split("\t")[:4]
            if kind != args.kind or chrom not in seqs:
                continue
            first, last = (int(start) - 1 + s - 1) // s, int(end) // s
            for i in range(first, min(last, len(seqs[chrom]))):
                counts[seqs[chrom][i]] += 1
    total = sum(counts.values())
    if total == 0:
        fail(f"no bin lies inside a {args.kind} segment")
    share = counts["K"] / total
    print(f"{args.kind}: {total} bins, {counts['K']} K, {counts['T']} T, {counts['N']} N "
          f"(K share {share:.2f}, threshold {args.max_k})")
    if share > args.max_k:
        fail(f"{counts['K']} of the {total} bins in {args.kind} segments are K")


def multihetsep(args):
    seqs = read_psmcfa(args.psmcfa)
    truth = {(c, p) for c, p, _, _, z, t in read_truth(f"{args.sim}/truth.tsv")
             if z == "het" and t == "snp"}
    rows, false = 0, []
    for path in args.files:
        with open(path) as f:
            for line in f:
                chrom, pos, nr_called, alleles = line.split()
                pos, nr_called = int(pos), int(nr_called)
                rows += 1
                if (chrom, pos) not in truth:
                    false.append(f"{chrom}:{pos}")
                if nr_called < 1:
                    fail(f"{path}: {chrom}:{pos} has nr_called {nr_called}")
                if len(alleles) != 2 or alleles[0] == alleles[1]:
                    fail(f"{path}: {chrom}:{pos} has alleles {alleles}")
                if seqs[chrom][(pos - 1) // args.bin_size] != "K":
                    fail(f"{path}: the bin of {chrom}:{pos} is not K in {args.psmcfa}")
    if rows == 0:
        fail("no multihetsep row")
    true_share = 1 - len(false) / rows
    print(f"multihetsep: {rows} rows in K bins, {rows - len(false)} simulated heterozygous "
          f"SNPs ({true_share:.4f}, threshold {args.min_true})"
          + (f"; others: {' '.join(false[:20])}" if false else ""))
    if true_share < args.min_true:
        fail(f"only {true_share:.4f} of the multihetsep rows are simulated heterozygous SNPs")


def qc(args):
    with open(args.json) as f:
        report = json.load(f)
    findings = {x["code"]: x for x in report["findings"]}
    kinds = {r["name"]: r["kind"] for r in report["sequences"]}
    sex = findings.get("SEX_OR_ORGANELLE")
    if args.sex_warning is None:
        if sex is not None:
            fail(f"unexpected SEX_OR_ORGANELLE finding: {sex['message']}")
        flagged = [n for n, k in kinds.items() if k is not None]
        if flagged:
            fail(f"sequences named like sex chromosomes or organelles: {flagged}")
        print("qc: no sex chromosome or organelle in the input")
    else:
        if sex is None:
            fail("no SEX_OR_ORGANELLE finding")
        if sex["severity"] != "warning":
            fail(f"SEX_OR_ORGANELLE is {sex['severity']}, not a warning: {sex['message']}")
        if sex["sequences"] != [args.sex_warning]:
            fail(f"SEX_OR_ORGANELLE names {sex['sequences']}, not [{args.sex_warning}]")
        if "hemizygous" not in sex["message"]:
            fail(f"SEX_OR_ORGANELLE does not say hemizygous: {sex['message']}")
        if kinds.get(args.sex_warning) != "sex":
            fail(f"{args.sex_warning} has kind {kinds.get(args.sex_warning)}")
        print(f"qc: SEX_OR_ORGANELLE warning: {sex['message']}")
    if args.kept is not None:
        names = [r["name"] for r in report["sequences"]]
        if names != args.kept.split(","):
            fail(f"qc lists sequences {names}, not {args.kept}")
    if args.dropped is not None:
        dropped = sum(d["count"] for d in report["dropped"])
        if dropped != args.dropped:
            fail(f"{dropped} sequences dropped, not {args.dropped}")
    if report["input_format"] != args.format:
        fail(f"input read as {report['input_format']}, not {args.format}")
    for code in args.has:
        if code not in findings:
            fail(f"no {code} finding")
        print(f"qc: {code}: {findings[code]['message']}")
    for code in args.lacks:
        if code in findings:
            fail(f"unexpected {code} finding: {findings[code]['message']}")


def fit(args):
    with open(args.json) as f:
        params = json.load(f)
    loglike = params.get("loglike", [])
    if params.get("n_iter") != args.n_iter:
        fail(f"n_iter {params.get('n_iter')}, not {args.n_iter}")
    if len(loglike) != args.n_iter + 1 or not all(math.isfinite(x) for x in loglike):
        fail(f"loglike {loglike}")
    if params.get("final_loglike") != loglike[-1]:
        fail("final_loglike is not the last loglike")
    if params["input"]["format"] != "psmcfa" or params["input"]["sequences"] != args.sequences:
        fail(f"input {params['input']}")
    if not (params["theta"] > 0 and params["rho"] > 0 and all(x > 0 for x in params["lam"])):
        fail("non-positive parameters")
    with open(args.psmc) as f:
        rounds = [line.split()[1] for line in f if line.startswith("RD\t")]
    if rounds != [str(i) for i in range(args.n_iter + 1)]:
        fail(f"{args.psmc} has rounds {rounds}")
    print(f"fit: {args.n_iter} iterations, log-likelihood {loglike[0]:.3f} -> {loglike[-1]:.3f}")


def kinds(args):
    left_out = {}
    section = None
    with open(args.dry_run) as f:
        for line in f:
            if not line.startswith("  "):
                section = line.split(" (")[0].strip()
                continue
            if section == "Left out by name":
                name, _, why = line.strip().rpartition(" (")
                left_out[name] = why.rstrip(")")
    with open(args.json) as f:
        report = json.load(f)
    if report["omitted"] is not None:
        fail("qc omitted sequences from its table; use fewer names")
    label = {"sex": "sex chromosome", "organelle": "organelle"}
    expected = {r["name"]: label[r["kind"]] for r in report["sequences"] if r["kind"]}
    if left_out != expected:
        only_script = sorted(set(left_out.items()) - set(expected.items()))
        only_qc = sorted(set(expected.items()) - set(left_out.items()))
        fail(f"the script and psmc-rs qc disagree: script only {only_script}, qc only {only_qc}")
    print(f"kinds: {len(expected)} of {len(report['sequences'])} names left out, as psmc-rs qc classifies them")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("recovery")
    p.add_argument("sim")
    p.add_argument("psmcfa")
    p.add_argument("bin_size", type=int)
    p.add_argument("--min-recovery", type=float, default=0.95)
    p.add_argument("--max-false-k", type=float, default=0.02)
    p.add_argument("--min-masked-n", type=float, default=0.95)
    p = sub.add_parser("region")
    p.add_argument("sim")
    p.add_argument("psmcfa")
    p.add_argument("bin_size", type=int)
    p.add_argument("kind")
    p.add_argument("--max-k", type=float, default=0.5)
    p = sub.add_parser("multihetsep")
    p.add_argument("sim")
    p.add_argument("psmcfa")
    p.add_argument("bin_size", type=int)
    p.add_argument("files", nargs="+")
    p.add_argument("--min-true", type=float, default=0.98)
    p = sub.add_parser("qc")
    p.add_argument("json")
    p.add_argument("--sex-warning")
    p.add_argument("--kept")
    p.add_argument("--dropped", type=int)
    p.add_argument("--format", default="vcf, consensus mode")
    p.add_argument("--has", action="append", default=[])
    p.add_argument("--lacks", action="append", default=[])
    p = sub.add_parser("fit")
    p.add_argument("json")
    p.add_argument("psmc")
    p.add_argument("n_iter", type=int)
    p.add_argument("sequences", type=int)
    p = sub.add_parser("kinds")
    p.add_argument("dry_run")
    p.add_argument("json")
    args = ap.parse_args()
    {"recovery": recovery, "region": region, "multihetsep": multihetsep, "qc": qc,
     "fit": fit, "kinds": kinds}[args.cmd](args)


if __name__ == "__main__":
    main()
