#!/usr/bin/env bash
# End-to-end test of scripts/bam_to_psmc.sh (see README.md): simulated reads
# of a small diploid genome go through the script, and its outputs are
# checked against the classic tools and the simulation truth.
#
#   VCFUTILS=/path/to/vcfutils.pl FQ2PSMCFA=/path/to/fq2psmcfa \
#   PSMC_RS=target/release/psmc-rs tests/pipeline/run_e2e.sh
#
# VCFUTILS and FQ2PSMCFA default to vcfutils.pl and fq2psmcfa on PATH, and
# PSMC_RS to target/release/psmc-rs. bcftools (>= 1.10), bgzip, tabix, perl
# and python3 must be on PATH. E2E_WORK=DIR keeps the files there instead of
# a temporary directory; E2E_BASH=PATH runs the script with that bash (for
# example /bin/bash, macOS's bash 3.2).

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
script="$root/scripts/bam_to_psmc.sh"
helpers="$root/tests/data/route_a/helpers.py"

step() { printf '\n== %s\n' "$*"; }
ok() { printf 'ok: %s\n' "$*"; }
fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}
check() { python3 "$here/check_outputs.py" "$@"; }
run_script() {
  if [ -n "${E2E_BASH:-}" ]; then
    "$E2E_BASH" "$script" "$@"
  else
    "$script" "$@"
  fi
}

PSMC_RS=${PSMC_RS:-$root/target/release/psmc-rs}
VCFUTILS=${VCFUTILS:-$(command -v vcfutils.pl || true)}
FQ2PSMCFA=${FQ2PSMCFA:-$(command -v fq2psmcfa || true)}
[ -x "$PSMC_RS" ] || fail "no psmc-rs at $PSMC_RS: build it (cargo build --release) or set PSMC_RS"
[ -n "$VCFUTILS" ] && [ -f "$VCFUTILS" ] ||
  fail "vcfutils.pl not found: set VCFUTILS (bcftools ships it as misc/vcfutils.pl)"
[ -n "$FQ2PSMCFA" ] && [ -x "$FQ2PSMCFA" ] ||
  fail "fq2psmcfa not found: set FQ2PSMCFA (build utils/fq2psmcfa of lh3/psmc)"
for tool in bcftools bgzip tabix perl python3; do
  command -v "$tool" > /dev/null || fail "$tool is not on PATH"
done
PSMC_RS=$(cd "$(dirname "$PSMC_RS")" && pwd)/$(basename "$PSMC_RS")
export PSMC_RS

if [ -n "${E2E_WORK:-}" ]; then
  work=$E2E_WORK
  rm -rf "$work"
  mkdir -p "$work"
else
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
fi
printf 'work directory: %s\n' "$work"
# $BASH_VERSION is that of the bash running the script.
# shellcheck disable=SC2016
printf 'tools: %s; %s; psmc-rs %s; vcfutils.pl %s; fq2psmcfa %s\n' \
  "$(bcftools --version | sed -n 1p)" "$("${E2E_BASH:-bash}" -c 'echo bash $BASH_VERSION')" \
  "$("$PSMC_RS" --version)" "$VCFUTILS" "$FQ2PSMCFA"

# ---------------------------------------------------------------------------
step "command line and dependency checks"

out=$work/cli
mkdir -p "$out/fakebin"
touch "$out/empty.bam"
printf '>c1\nACGT\n' > "$out/tiny.fa"

run_script --help > "$out/help.txt" || fail "--help"
grep -q '^Usage: ' "$out/help.txt" || fail "--help prints no usage"
ok "--help"

expect_error() { # expected status, expected message, arguments...
  local want=$1 msg=$2 s=0
  shift 2
  run_script "$@" 2> "$out/err.txt" > /dev/null || s=$?
  [ "$s" -eq "$want" ] || fail "exit status $s, not $want, for: $* ($(cat "$out/err.txt"))"
  grep -qF -- "$msg" "$out/err.txt" || fail "no '$msg' in: $(cat "$out/err.txt")"
  ok "$msg (exit $want)"
}
base_args=(--bam "$out/empty.bam" --ref "$out/tiny.fa" --out "$out/x")
expect_error 2 "--bam is required" --ref "$out/tiny.fa" --out "$out/x"
expect_error 2 "--depth must be auto or MIN:MAX" "${base_args[@]}" --depth 10
expect_error 2 "--depth must be auto or MIN:MAX" "${base_args[@]}" --depth 50:10
expect_error 2 "--threads must be a whole number" "${base_args[@]}" --threads two
expect_error 2 "needs --run N_ITER" "${base_args[@]}" -- --pattern 4+5*3+4
expect_error 2 "unknown argument '--frobnicate'" "${base_args[@]}" --frobnicate
expect_error 2 "give the bin size as this script's --bin-size" "${base_args[@]}" --run 2 \
  -- --bin-size=50
expect_error 1 "cannot read --bam" --bam "$out/missing.bam" --ref "$out/tiny.fa" --out "$out/x"

printf '#!/bin/sh\necho "bcftools 1.9"\necho "Using htslib 1.9"\n' > "$out/fakebin/bcftools"
chmod +x "$out/fakebin/bcftools"
BCFTOOLS=$out/fakebin/bcftools expect_error 1 "bcftools 1.9 is too old: 1.10 or later is needed" \
  "${base_args[@]}"
printf '#!/bin/sh\necho "psmc-rs 0.1.2"\n' > "$out/fakebin/psmc-rs"
chmod +x "$out/fakebin/psmc-rs"
expect_error 1 "psmc-rs 0.1.2 is too old" "${base_args[@]}" --psmc-rs "$out/fakebin/psmc-rs"
expect_error 1 "psmc-rs not found" "${base_args[@]}" --psmc-rs "$out/no-such-psmc-rs"
expect_error 1 "--chroms: 'c2' is not a sequence of" "${base_args[@]}" --chroms c1,c2
expect_error 1 "--chroms names c1 twice" "${base_args[@]}" --chroms c1,c1

# The names left out by default are those psmc-rs qc calls sex chromosomes
# or organelles.
names="chr1 1 2L chr2R chrX X chrY Y chrM M MT chrMT Mt mito mtDNA chrW chrZ Z W X1 chrX2
Y2 Z1 chrY_KI270740v1_random chrX_alt chrUn_KI270302v1 chr1_KI270706v1_random chrEBV
NC_000023.11 NC_000024.10 NC_012920.1 NC_001807.4 NC_000086.8 NC_000087.8 NC_005089.1
NC_000001.11 scaffold_12 mitochondrion Pt chrC C cp CP chloroplast NC_037304_chloroplast
dmel_mitochondrion_genome apicoplast plastid_genome plastome chromosome_X chromosomeX
chromosome1 chrom-Y chrom_Z chr.W chrX1A chrXY xenopus_scaffold w1118 Wx HLA-A*01:01:01:01
chrx chrm mt cX chr_M chr-mt NW_123.1 Super-Scaffold_1 Z_random 2W chr2W"
: > "$out/names.fa"
: > "$out/names.psmcfa"
set -f # no globbing of names like HLA-A*01:01:01:01
for n in $names; do
  printf '>%s\nACGT\n' "$n" >> "$out/names.fa"
  printf '>%s\nT\n' "$n" >> "$out/names.psmcfa"
done
set +f
awk 'BEGIN { off = 0 }
  /^>/ { name = substr($0, 2); off += length($0) + 1; next }
  { printf "%s\t%d\t%d\t%d\t%d\n", name, length($0), off, length($0), length($0) + 1
    off += length($0) + 1 }' "$out/names.fa" > "$out/names.fa.fai"
run_script --bam "$out/empty.bam" --ref "$out/names.fa" --out "$out/names" --dry-run \
  > "$out/dry_run.txt"
"$PSMC_RS" qc "$out/names.psmcfa" --json "$out/names.qc.json" > /dev/null 2>&1
check kinds "$out/dry_run.txt" "$out/names.qc.json"

# ---------------------------------------------------------------------------
step "simulation"

sim=$work/sim
python3 "$here/simulate.py" "$sim"
ls -l "$sim"

# ---------------------------------------------------------------------------
step "run 1: SAM without index, 3 jobs, multihetsep and a 3-iteration fit"

out=$work/run
run_script --bam "$sim/aln.sam" --ref "$sim/ref.fa" --out "$out/sim" --threads 3 \
  --multihetsep --run 3 2> "$work/run1.err"
p=$out/sim
for f in "$p.vcf.gz" "$p.vcf.gz.tbi" "$p.qc.txt" "$p.qc.json" "$p.psmcfa" "$p.json" \
  "$p.html" "$p.psmc" "$p.log" "$p.chr1.multihetsep"; do
  [ -s "$f" ] || fail "missing output $f"
done
[ -z "$(find "$out" -name '.*tmp*')" ] || fail "working directory left behind in $out"
grep -qF "left out by name: chrX (sex chromosome), chrM (organelle)" "$work/run1.err" ||
  fail "the log does not list chrX and chrM as left out"
contigs=$(bcftools index -s "$p.vcf.gz" 2> /dev/null | cut -f 1 | tr '\n' ' ')
[ "$contigs" = "chr1 chr2 chr3 chrUn_scaffold1 " ] || fail "VCF contigs with records: $contigs"
ok "outputs written; chrX and chrM left out, VCF holds $contigs"

# (a) The psmcfa is what vcf2fq | fq2psmcfa make of the same VCF with the
# same thresholds; the automatic depth bounds are computed independently.
expect_classic() { # prefix, label
  local p=$1 mean dmin dmax
  read -r mean dmin dmax < <(python3 "$helpers" auto-depth "$p.vcf.gz")
  grep -qF "bases kept at depth $dmin-$dmax (auto)" "$p.log" ||
    fail "psmc-rs did not report the bounds $dmin-$dmax (mean depth $mean)"
  gzip -dc "$p.vcf.gz" | perl "$VCFUTILS" vcf2fq -d "$dmin" -D "$dmax" > "$p.expected.fq"
  "$FQ2PSMCFA" -q20 -s100 "$p.expected.fq" > "$p.expected.psmcfa"
  cmp "$p.expected.psmcfa" "$p.psmcfa" ||
    fail "$p.psmcfa differs from vcf2fq -d $dmin -D $dmax | fq2psmcfa -q20 -s100"
  ok "$2: psmcfa equals vcfutils.pl vcf2fq -d $dmin -D $dmax | fq2psmcfa -q20 -s100 byte for byte (mean depth $mean)"
}
expect_classic "$p" "run 1"
if grep -q '^>chrUn_scaffold1' "$p.psmcfa"; then
  fail "chrUn_scaffold1 should be dropped"
fi
ok "chrUn_scaffold1 (fewer than 10,000 called bases) dropped by both"

# (b) Recovery of the simulated heterozygous sites.
check recovery "$sim" "$p.psmcfa" 100 --min-recovery 0.97 --max-false-k 0.01 --min-masked-n 0.97
check multihetsep "$sim" "$p.psmcfa" 100 "$p".chr*.multihetsep --min-true 0.98
[ ! -e "$p.chrUn_scaffold1.multihetsep" ] || fail "multihetsep written for a dropped sequence"

# (c) QC: no sex chromosome or organelle reached the input.
check qc "$p.qc.json" --kept chr1,chr2,chr3 --dropped 1
grep -q '^Input QC (vcf, consensus mode): 3 sequences' "$p.qc.txt" || fail "QC text: $(head -1 "$p.qc.txt")"

# (d) The fit.
check fit "$p.json" "$p.psmc" 3 3

# ---------------------------------------------------------------------------
step "run 2: the same reads as bgzipped SAM with an index, 1 job, same prefix"

mkdir -p "$out/saved"
outputs=(sim.vcf.gz sim.vcf.gz.tbi sim.qc.txt sim.qc.json sim.psmcfa sim.chr1.multihetsep
  sim.chr2.multihetsep sim.chr3.multihetsep sim.json sim.psmc sim.html)
for f in "${outputs[@]}"; do
  cp "$out/$f" "$out/saved/$f"
done
bgzip -c "$sim/aln.sam" > "$sim/aln.sam.gz"
tabix -C -p sam "$sim/aln.sam.gz"
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$out/sim" --threads 1 \
  --multihetsep --run 3 2> "$work/run2.err"
if grep -q 'no index' "$work/run2.err"; then
  fail "the index of aln.sam.gz was not used"
fi
grep -qF -- " -R " "$p.log" || fail "the calls did not read by region"
for f in "${outputs[@]}"; do
  cmp "$out/saved/$f" "$out/$f" || fail "$f changed when rerun"
done
ok "a rerun by region with 1 job writes the same bytes: ${outputs[*]}"

# ---------------------------------------------------------------------------
step "run 3: --chroms from a file, with chrX"

printf '# autosomes and the X\nchr1\nchr2\nchr3\nchrX\n' > "$work/chroms.txt"
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$out/withx" \
  --chroms "$work/chroms.txt" 2> "$work/run3.err"
grep -qF "warning: chrX is named like a sex chromosome" "$work/run3.err" ||
  fail "no warning about chrX"
q=$out/withx
contigs=$(bcftools index -s "$q.vcf.gz" 2> /dev/null | cut -f 1 | tr '\n' ' ')
[ "$contigs" = "chr1 chr2 chr3 chrX " ] || fail "VCF contigs with records: $contigs"
expect_classic "$q" "run 3"
check qc "$q.qc.json" --sex-warning chrX --kept chr1,chr2,chr3,chrX --dropped 0

# ---------------------------------------------------------------------------
step "run 4: a contig name with colons, read by region and streamed"

# bcftools -r would take the colons for a range ("the sequence "HLA-A*01"
# not found"); the script's regions files keep the name whole.
c=$work/colon
mkdir -p "$c"
name='HLA-A*01:01:01:01'
sed "s/^>chr3\$/>$name/" "$sim/ref.fa" > "$c/ref.fa"
awk -v name="$name" 'BEGIN { OFS = "\t" }
  /^@SQ/ && $2 == "SN:chr3" { $2 = "SN:" name }
  /^@/ { print; next }
  $3 == "chr3" { $3 = name }
  { print }' "$sim/aln.sam" > "$c/aln.sam"
bgzip -c "$c/aln.sam" > "$c/aln.sam.gz"
tabix -C -p sam "$c/aln.sam.gz"
for bam in aln.sam aln.sam.gz; do
  run_script --bam "$c/$bam" --ref "$c/ref.fa" --out "$c/$bam.out" --chroms "$name" \
    2> "$c/$bam.err"
  contigs=$(bcftools index -s "$c/$bam.out.vcf.gz" 2> /dev/null | cut -f 1)
  [ "$contigs" = "$name" ] || fail "$bam: VCF contigs with records: $contigs"
  [ "$(grep '^>' "$c/$bam.out.psmcfa")" = ">$name" ] || fail "$bam: no psmcfa record for $name"
done
cmp "$c/aln.sam.out.psmcfa" "$c/aln.sam.gz.out.psmcfa" || fail "-T and -R calls differ"
ok "$name called by region and streamed, with the same psmcfa"

# ---------------------------------------------------------------------------
step "route B of docs/preprocessing.md: variant calls and a callable mask from a gVCF"

b=$work/routeb
mkdir -p "$b"
read -r mean dmin dmax < <(python3 "$helpers" auto-depth "$p.vcf.gz")
printf 'chr1\t1\t150000\nchr2\t1\t100000\nchr3\t1\t70000\n' > "$b/autosomes.txt"
bcftools mpileup -C50 -q20 -Q20 -a FORMAT/DP -f "$sim/ref.fa" -T "$b/autosomes.txt" \
  "$sim/aln.sam" 2> /dev/null |
  bcftools call -m -g "$dmin,$((dmax + 1))" -Oz -o "$b/sample.g.vcf.gz" 2> /dev/null
bcftools query -i "FMT/DP>=$dmin && FMT/DP<=$dmax" -f '%CHROM\t%POS0\t%END\n' \
  "$b/sample.g.vcf.gz" > "$b/callable.bed" 2> /dev/null
bcftools view -v snps,indels -Oz -o "$b/variants.vcf.gz" "$b/sample.g.vcf.gz" 2> /dev/null
"$PSMC_RS" qc "$b/variants.vcf.gz" --json "$b/nomask.qc.json" > /dev/null 2>&1
check qc "$b/nomask.qc.json" --format "vcf, mask mode" --has VARIANTS_ONLY_NO_MASK
"$PSMC_RS" convert "$b/variants.vcf.gz" --vcf-mask "$b/callable.bed" --psmcfa "$b/b.psmcfa" \
  --multihetsep-per-chrom "$b/b" --qc-json "$b/b.qc.json" -q 2> /dev/null > /dev/null
check qc "$b/b.qc.json" --format "vcf, mask mode" --lacks VARIANTS_ONLY_NO_MASK \
  --kept chr1,chr2,chr3 --dropped 0
check recovery "$sim" "$b/b.psmcfa" 100 --min-recovery 0.97 --max-false-k 0.01 --min-masked-n 0.97
check multihetsep "$sim" "$b/b.psmcfa" 100 "$b"/b.chr*.multihetsep --min-true 0.98

# ---------------------------------------------------------------------------
step "failures leave no output"

# Files of the run with prefix $1 other than its log.
leftovers() {
  local f
  for f in "$out/$1".*; do
    [ -e "$f" ] && [ "$f" != "$out/$1.log" ] && echo "$f"
  done
  return 0
}

f=$out/fail
s=0
run_script --bam "$sim/aln.sam" --ref "$sim/ref.fa" --out "$f" --chroms chr3 --min-mapq 255 \
  2> "$work/fail.err" || s=$?
[ "$s" -eq 1 ] || fail "exit status $s with no read passing --min-mapq"
grep -qF "bcftools called no site" "$work/fail.err" || fail "$(cat "$work/fail.err")"
[ -z "$(leftovers fail)" ] || fail "outputs left: $(leftovers fail)"
[ -z "$(find "$out" -name '.*tmp*')" ] || fail "working directory left behind in $out"
ok "no site called: error, only fail.log written"

# Two samples in one file (read groups with different SM tags).
awk 'BEGIN { OFS = "\t" }
  /^@RG/ { print; print "@RG\tID:run2\tSM:sim2"; next }
  /^@/ { print; next }
  { if (NR % 2) sub(/RG:Z:run1$/, "RG:Z:run2"); print }' "$sim/aln.sam" > "$work/two.sam"
s=0
run_script --bam "$work/two.sam" --ref "$sim/ref.fa" --out "$out/two" --chroms chr3 \
  2> "$work/two.err" || s=$?
[ "$s" -eq 1 ] || fail "exit status $s with two samples"
grep -qF "the alignments hold 2 samples" "$work/two.err" || fail "$(cat "$work/two.err")"
ok "two samples: error"

# Interrupted runs. A stand-in for one tool never finishes (a `sleep` with
# a number of its own); the script is stopped while it waits for it, and
# must exit at once, stop the stand-in and remove its working directory.
#
# interrupt NAME SIGNAL STATUS [VAR=VALUE...] -- SCRIPT_ARGS...
# starts the script with prefix $out/NAME in its own process group, as an
# interactive shell would (so that it does not ignore SIGINT); waits until
# `sleep $sleeper` runs; sends SIGNAL to the script (or, for INT-group,
# SIGINT to the whole group, as Ctrl+C in a terminal does); and checks the
# exit status.
interrupt() {
  local name=$1 sig=$2 want=$3 spid s=0 tries=0 start envs=()
  shift 3
  while [ "$1" != -- ]; do
    envs+=("$1")
    shift
  done
  shift
  set -m
  env ${envs[@]+"${envs[@]}"} ${E2E_BASH:+"$E2E_BASH"} "$script" "$@" --out "$out/$name" \
    2> "$work/$name.err" &
  spid=$!
  set +m
  until pgrep -f "sleep $sleeper" > /dev/null; do
    tries=$((tries + 1))
    [ "$tries" -lt 300 ] || fail "$name: the run did not reach the stand-in ($(cat "$work/$name.err"))"
    sleep 0.1
  done
  case "$sig" in
    INT-group) kill -INT -- "-$spid" ;;
    *) kill -"$sig" "$spid" ;;
  esac
  start=$SECONDS
  while kill -0 "$spid" 2> /dev/null; do
    if [ "$((SECONDS - start))" -ge 10 ]; then
      pkill -f "sleep $sleeper" || true
      wait "$spid" || true
      fail "$name: the script was still running 10 s after $sig"
    fi
    sleep 0.1
  done
  wait "$spid" || s=$?
  [ "$s" -eq "$want" ] || fail "$name: exit status $s after $sig, not $want"
  sleep 0.5
  if pgrep -f "sleep $sleeper" > /dev/null; then
    pkill -f "sleep $sleeper" || true
    fail "$name: the stand-in outlived the script"
  fi
  [ -z "$(find "$out" -name '.*tmp*')" ] || fail "$name: working directory left behind in $out"
}

# mpileup never finishes. The script runs with a PATH that holds what it
# needs but no pkill, pgrep or ps, so that it cannot rely on them to stop
# mpileup and call.
real_bcftools=$(command -v bcftools)
sleeper=987
# The wrapper expands $1 and $@ itself.
# shellcheck disable=SC2016
printf '#!/bin/sh\nif [ "$1" = mpileup ]; then exec sleep %s; fi\nexec "%s" "$@"\n' \
  "$sleeper" "$real_bcftools" > "$work/slow-bcftools"
chmod +x "$work/slow-bcftools"
mkdir -p "$work/minbin"
for tool in bash env awk basename cat cmp date dirname getconf grep head mkdir mkfifo mktemp \
  mv nproc rm sed sleep sort tail tr wc; do
  p=$(command -v "$tool" || true)
  case "$p" in
    /*) ln -sf "$p" "$work/minbin/$tool" ;;
  esac
done
for tool in pkill pgrep ps; do
  [ ! -e "$work/minbin/$tool" ] || fail "minbin holds $tool"
done
interrupt int TERM 143 BCFTOOLS="$work/slow-bcftools" PATH="$work/minbin" -- \
  --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr2,chr3 --threads 2
[ -z "$(leftovers int)" ] || fail "outputs left: $(leftovers int)"
ok "SIGTERM while calling, without pkill on PATH: exit 143, mpileup and call stopped, only int.log written"

# The fit never finishes: SIGTERM to the script alone, and SIGINT to its
# process group (Ctrl+C). The outputs made before the fit stay; the fit
# writes none.
sleeper=988
# The wrapper expands $1 and $@ itself.
# shellcheck disable=SC2016
printf '#!/bin/sh\ncase "$1" in *.psmcfa) [ -f "$1" ] && exec sleep %s ;; esac\nexec "%s" "$@"\n' \
  "$sleeper" "$PSMC_RS" > "$work/slow-psmc-rs"
chmod +x "$work/slow-psmc-rs"
for sig in TERM INT-group; do
  name=fit-$sig
  want=143
  [ "$sig" = TERM ] || want=130
  interrupt "$name" "$sig" "$want" -- --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr3 \
    --run 2 --psmc-rs "$work/slow-psmc-rs"
  left=$(leftovers "$name" | sed "s|^$out/$name||" | LC_ALL=C sort | tr '\n' ' ')
  [ "$left" = ".psmcfa .qc.json .qc.txt .vcf.gz .vcf.gz.tbi " ] ||
    fail "$name: outputs left: $left"
done
ok "SIGTERM while fitting, and SIGINT to the process group (Ctrl+C): exit 143 and 130 at once, the fit stopped, no fit output"

printf '\nall end-to-end checks passed\n'
