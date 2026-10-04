#!/usr/bin/env bash
# Regenerate c_psmc_round0.tsv (and synthetic.psmcfa) from C psmc.
#
# Usage, from anywhere:
#   PSMC=/path/to/psmc tests/data/qc_c_reference/generate.sh
#
# For every input file (and, where listed, every record of it on its own) this
# runs `psmc -N 0`, which reads the data, prints the MM summary line
# `n_seqs:.., sum_L:.., sum_n:..` and the round-0 parameters, and stops before
# any EM iteration. The table keeps:
#   n_seqs, sum_L (non-missing bins), sum_n (heterozygous bins),
#   theta0   = first value of the round-0 PA line (%.9lf), -ln(1 - sum_n/sum_L),
#   n_recomb = the round-0 `MM C_pi: .., n_recomb: ..` value (%lf).
set -euo pipefail

: "${PSMC:?set PSMC to a psmc binary (lh3/psmc 0.6.5-r74)}"
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
cd "$root"

python3 "$here/make_synthetic.py" "$here/synthetic.psmcfa"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
out="$here/c_psmc_round0.tsv"

# run LABEL RECORD FILE: append one table row for `psmc -N 0 FILE`.
run() {
    "$PSMC" -N 0 -o "$tmp/out.psmc" "$3"
    awk -F'\t' -v input="$1" -v rec="$2" '
        $1 == "MM" && $2 ~ /^n_seqs:/ {
            split($2, a, /[:,] ?/); n_seqs = a[2]; sum_L = a[4]; sum_n = a[6]
        }
        $1 == "MM" && $2 ~ /^C_pi:/ && n_recomb == "" {
            split($2, a, /: |, /); n_recomb = a[4]
        }
        $1 == "PA" && theta == "" { split($2, a, " "); theta = a[2] }
        END { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", input, rec, n_seqs, sum_L, sum_n, theta, n_recomb }
    ' "$tmp/out.psmc" >>"$out"
}

# per_record FILE: one row per record, each record run as its own file.
per_record() {
    rm -f "$tmp"/rec_*
    awk -v dir="$tmp" '
        /^>/ { n++; f = sprintf("%s/rec_%05d", dir, n); split(substr($0, 2), w, /[ \t]/); print w[1] > (f ".name") }
        { print > (f ".psmcfa") }
    ' "$1"
    for fa in "$tmp"/rec_*.psmcfa; do
        run "$1" "$(cat "${fa%.psmcfa}.name")" "$fa"
    done
}

printf 'input\trecord\tn_seqs\tsum_L\tsum_n\ttheta0\tn_recomb\n' >"$out"
for f in \
    tests/data/qc_c_reference/synthetic.psmcfa \
    tests/data/c_reference/papuan_chr1_3seg.psmcfa \
    experiment/real_data/Papuan_highlands_diploid.psmcfa \
    experiment/real_data/HLhydTec1_diploid.psmcfa \
    experiment/real_data/HLemySub1_diploid.psmcfa \
    experiment/real_data/HLpelCas1_diploid.psmcfa; do
    run "$f" "*" "$f"
done
for f in \
    tests/data/qc_c_reference/synthetic.psmcfa \
    tests/data/c_reference/papuan_chr1_3seg.psmcfa \
    experiment/real_data/HLhydTec1_diploid.psmcfa; do
    per_record "$f"
done
