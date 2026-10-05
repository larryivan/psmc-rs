#!/usr/bin/env bash
# Regenerate the keep-alts fixtures (see README.md) with the real tools:
#
#   VCFUTILS=/path/to/vcfutils.pl \
#   FQ2PSMCFA=/path/to/fq2psmcfa \
#   tests/data/route_a_keep_alts/make_fixtures.sh
#
# bcftools must be on PATH. The reads and reference go to a temporary
# directory. The output is byte for byte reproducible: bcftools runs with
# --no-version on relative paths, and gzip with -n.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
: "${VCFUTILS:?set VCFUTILS to misc/vcfutils.pl of bcftools}"
: "${FQ2PSMCFA:?set FQ2PSMCFA to utils/fq2psmcfa of lh3/psmc}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$here"
rm -rf expected
mkdir -p expected

# 1. One chromosome of 3,000 bp at 250x.
python3 simulate_high_depth.py "$work"

# 2. All sites with `call -c -A` (keep alternative alleles), and variant
#    sites only with `call -cv`, from the same reads.
(
  cd "$work"
  bcftools mpileup --no-version -C50 -q20 -Q20 -f ref.fa aln.sam 2>/dev/null \
    | bcftools call --no-version -c -A -Oz -o keep_alts_raw.vcf.gz 2>/dev/null
  bcftools mpileup --no-version -C50 -q20 -Q20 -f ref.fa aln.sam 2>/dev/null \
    | bcftools call --no-version -cv -o variants_c.vcf 2>/dev/null
)
# As in ../route_a, keep only what vcf2fq and the readers use.
bcftools annotate --no-version \
  -x 'QUAL,^INFO/DP,INFO/MQ,INFO/FQ,INFO/AF1,INFO/INDEL,^FORMAT/GT' \
  "$work/keep_alts_raw.vcf.gz" 2>/dev/null | gzip -9n > keep_alts.vcf.gz
gzip -9n < "$work/variants_c.vcf" > variants_c.vcf.gz

# 3. vcf2fq | fq2psmcfa with the depth bounds the psmc README recommends,
#    computed independently of psmc-rs.
read -r mean dmin dmax < <(python3 ../route_a/helpers.py auto-depth keep_alts.vcf.gz)
printf '%s\t%s\t%s\n' "$mean" "$dmin" "$dmax" > expected/auto_depth.tsv
gzip -dc keep_alts.vcf.gz | perl "$VCFUTILS" vcf2fq -d "$dmin" -D "$dmax" > "$work/keep_alts.fq"
"$FQ2PSMCFA" -g 1000 -s 50 "$work/keep_alts.fq" > expected/keep_alts.psmcfa

# 4. Record counts, as the README quotes them.
count() { gzip -dc "$1" | grep -v '^#' | awk -v what="$2" '
  { n++; if ($5 == ".") dot++; if ($8 ~ /FQ=/) fq++ }
  END { printf "%s\t%d\t%d\t%d\n", what, n, dot, fq }'; }
{
  printf 'file\trecords\talt_dot\twith_fq\n'
  count keep_alts.vcf.gz keep_alts.vcf.gz
  count variants_c.vcf.gz variants_c.vcf.gz
} > expected/records.tsv

# 5. Provenance.
bcftools_version=$(bcftools --version | head -2 | tr '\n' ' ')
perl_version=$(perl -e 'print $^V')
{
  echo "generated: $(date -u +%Y-%m-%d)"
  echo "bcftools: $bcftools_version"
  echo "perl: $perl_version"
  echo "python3: $(python3 --version 2>&1)"
  for f in "$VCFUTILS" "$FQ2PSMCFA"; do
    sum=$(shasum -a 256 "$f")
    echo "sha256 $(basename "$f"): ${sum%% *}"
  done
} > expected/versions.txt
