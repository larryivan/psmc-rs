#!/usr/bin/env bash
# Regenerate the route A fixtures (see README.md) with the real tools:
#
#   VCFUTILS=/path/to/vcfutils.pl \
#   FQ2PSMCFA=/path/to/fq2psmcfa \
#   GENERATE_MULTIHETSEP=/path/to/generate_multihetsep.py \
#   tests/data/route_a/make_fixtures.sh
#
# bcftools must be on PATH. The simulated reads and reference go to a
# temporary directory; only the files the tests read are written here. The
# output is byte for byte reproducible: bcftools runs with --no-version
# (no command lines or dates in the headers) on relative paths, and gzip
# with -n (no names or times).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
: "${VCFUTILS:?set VCFUTILS to misc/vcfutils.pl of bcftools}"
: "${FQ2PSMCFA:?set FQ2PSMCFA to utils/fq2psmcfa of lh3/psmc}"
: "${GENERATE_MULTIHETSEP:?set GENERATE_MULTIHETSEP to generate_multihetsep.py of msmc-tools}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$here"
rm -rf expected masks
mkdir -p expected masks

# 1. Reads and genome.
python3 simulate.py "$work"

# 2. The classic calling step, all sites, as in the PSMC README.
(
  cd "$work"
  bcftools mpileup --no-version -C50 -q20 -Q20 -f ref.fa aln.sam 2>/dev/null \
    | bcftools call --no-version -c -Oz -o allsites_raw.vcf.gz 2>/dev/null
)

# The main fixture keeps only what vcf2fq and the readers use (CHROM, POS,
# REF, ALT, INFO DP/MQ/FQ/AF1/INDEL and GT), which halves its size; the
# smallest chromosome is also kept with every field bcftools wrote.
bcftools annotate --no-version \
  -x 'QUAL,^INFO/DP,INFO/MQ,INFO/FQ,INFO/AF1,INFO/INDEL,^FORMAT/GT' \
  "$work/allsites_raw.vcf.gz" 2>/dev/null | gzip -9n > allsites.vcf.gz
bcftools view --no-version -t chr3 "$work/allsites_raw.vcf.gz" 2>/dev/null \
  | gzip -9n > allsites_chr3_unmodified.vcf.gz

# 3. vcf2fq | fq2psmcfa with several option sets: name, vcf2fq options,
#    fq2psmcfa options, input.
run() {
  local name=$1 v=$2 f=$3 vcf=$4
  gzip -dc "$vcf" | perl "$VCFUTILS" vcf2fq $v > "$work/$name.fq"
  "$FQ2PSMCFA" $f "$work/$name.fq" > "expected/$name.psmcfa"
  printf '%s\tvcf2fq %s\tfq2psmcfa %s\t%s\n' "$name" "$v" "$f" "$vcf" >> expected/option_sets.tsv
}
printf 'name\tvcf2fq\tfq2psmcfa\tinput\n' > expected/option_sets.tsv
run default "" "" allsites.vcf.gz
run depth "-d 10 -D 60" "-g 1000 -s 50" allsites.vcf.gz
run mq_window "-Q 30 -l 10" "-q 20 -g 500 -s 10" allsites.vcf.gz
run per_base "-d 1 -D 1000 -Q 0 -l 1" "-q 0 -g 0 -s 1" allsites.vcf.gz
read -r mean dmin dmax < <(python3 helpers.py auto-depth allsites.vcf.gz)
printf '%s\t%s\t%s\n' "$mean" "$dmin" "$dmax" > expected/auto_depth.tsv
run auto "-d $dmin -D $dmax" "-g 2000" allsites.vcf.gz
run unmodified_chr3 "-d 3" "-g 100 -s 20" allsites_chr3_unmodified.vcf.gz
gzip -9n expected/per_base.psmcfa

# 4. Multihetsep: SNP calls (variants only, no indels), bamCaller-style
#    callable masks, and generate_multihetsep.py, one chromosome per run.
(
  cd "$work"
  bcftools mpileup --no-version -C50 -q20 -Q20 -f ref.fa aln.sam 2>/dev/null \
    | bcftools call --no-version -mv -V indels 2>/dev/null
) | gzip -9n > snps.vcf.gz
python3 helpers.py phase snps.vcf.gz | gzip -9n > snps_phased.vcf.gz
python3 helpers.py callable-mask allsites.vcf.gz masks
gzip -9n masks/*.bed
for chrom in chr1 chr2 chr3; do
  for set in snps snps_phased; do
    bcftools view --no-version -t "$chrom" "$set.vcf.gz" 2>/dev/null \
      | gzip > "$work/$set.$chrom.vcf.gz"
    python3 "$GENERATE_MULTIHETSEP" --mask "masks/mask_$chrom.bed.gz" \
      "$work/$set.$chrom.vcf.gz" > "expected/$set.$chrom.multihetsep" 2>/dev/null
  done
done

# 5. Provenance. (Command substitutions are kept out of double quotes where
# they contain quotes, which macOS's bash 3.2 cannot parse.)
bcftools_version=$(bcftools --version | head -2 | tr '\n' ' ')
perl_version=$(perl -e 'print $^V')
{
  echo "generated: $(date -u +%Y-%m-%d)"
  echo "bcftools: $bcftools_version"
  echo "perl: $perl_version"
  echo "python3: $(python3 --version 2>&1)"
  for f in "$VCFUTILS" "$FQ2PSMCFA" "$GENERATE_MULTIHETSEP"; do
    sum=$(shasum -a 256 "$f")
    echo "sha256 $(basename "$f"): ${sum%% *}"
  done
} > expected/versions.txt
