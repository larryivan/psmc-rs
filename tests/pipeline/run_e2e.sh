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
# and python3 must be on PATH. E2E_WORK=DIR works in DIR instead of a
# temporary directory (whose name holds a space) and keeps the files there:
# DIR must be new or empty, or a directory that an earlier run made, which is
# emptied first.
# E2E_BASH=PATH runs the script with that bash (for example /bin/bash,
# macOS's bash 3.2).

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
run_script() { ${E2E_BASH:+"$E2E_BASH"} "$script" "$@"; }
# The sequences with records in the VCF $1, on one line.
vcf_contigs() { bcftools index -s "$1" 2> /dev/null | cut -f 1 | tr '\n' ' '; }
# Fails unless every file given is there and not empty.
expect_files() {
  local f
  for f in "$@"; do
    [ -s "$f" ] || fail "missing output $f"
  done
}

PSMC_RS=${PSMC_RS:-$root/target/release/psmc-rs}
VCFUTILS=${VCFUTILS:-$(command -v vcfutils.pl || true)}
FQ2PSMCFA=${FQ2PSMCFA:-$(command -v fq2psmcfa || true)}
[ -x "$PSMC_RS" ] || fail "no psmc-rs at $PSMC_RS: build it (cargo build --release) or set PSMC_RS"
# An empty name is no file, so these also catch tools that are not found.
[ -f "$VCFUTILS" ] ||
  fail "vcfutils.pl not found: set VCFUTILS (bcftools ships it as misc/vcfutils.pl)"
[ -x "$FQ2PSMCFA" ] ||
  fail "fq2psmcfa not found: set FQ2PSMCFA (build utils/fq2psmcfa of lh3/psmc)"
for tool in bcftools bgzip tabix perl python3; do
  command -v "$tool" > /dev/null || fail "$tool is not on PATH"
done
PSMC_RS=$(cd "$(dirname "$PSMC_RS")" && pwd)/$(basename "$PSMC_RS")
export PSMC_RS

if [ -n "${E2E_WORK:-}" ]; then
  work=$E2E_WORK
  # Only a directory that holds the marker of an earlier run is emptied.
  if [ -e "$work" ] && [ ! -e "$work/.run_e2e_work" ] && [ -n "$(ls -A "$work")" ]; then
    fail "E2E_WORK=$work is not empty and was not made by run_e2e.sh; give a new or empty directory"
  fi
  rm -rf "$work"
  mkdir -p "$work"
  : > "$work/.run_e2e_work"
else
  # A space in the name, so that every run checks that the script and this
  # test handle such paths.
  tmp_root=$(mktemp -d)
  trap 'rm -rf "$tmp_root"' EXIT
  work="$tmp_root/e2e work"
  mkdir "$work"
fi
printf 'work directory: %s\n' "$work"
# $BASH_VERSION is that of the bash running the script.
# shellcheck disable=SC2016
printf 'tools: %s; %s; %s; vcfutils.pl %s; fq2psmcfa %s\n' \
  "$(bcftools --version | sed -n 1p)" "$("${E2E_BASH:-bash}" -c 'echo bash $BASH_VERSION')" \
  "$("$PSMC_RS" --version)" "$VCFUTILS" "$FQ2PSMCFA"

# ---------------------------------------------------------------------------
step "command line and dependency checks"

out=$work/cli
mkdir -p "$out/fakebin"
touch "$out/empty.bam"
printf '>c1\nACGT\n' > "$out/tiny.fa"

# E2E_WORK leaves a directory alone that this script did not make.
mkdir -p "$out/notmine"
echo keep > "$out/notmine/data.txt"
s=0
E2E_WORK=$out/notmine "$here/run_e2e.sh" > "$out/notmine.out" 2>&1 || s=$?
if [ "$s" -ne 1 ] || [ ! -f "$out/notmine/data.txt" ] ||
  ! grep -qF "is not empty and was not made by run_e2e.sh" "$out/notmine.out"; then
  fail "E2E_WORK on a directory of other files: exit $s, $(cat "$out/notmine.out")"
fi
ok "E2E_WORK refuses a non-empty directory that run_e2e.sh did not make"

run_script --help > "$out/help.txt" || fail "--help"
grep -q '^Usage: ' "$out/help.txt" || fail "--help prints no usage"
ok "--help"
s=0
run_script > /dev/null 2> "$out/noargs.txt" || s=$?
if [ "$s" -ne 2 ] || ! cmp -s "$out/help.txt" "$out/noargs.txt"; then
  fail "without arguments: exit $s, $(cat "$out/noargs.txt")"
fi
ok "without arguments: the help on standard error (exit 2)"

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
expect_error 2 "--out needs a value" --bam "$out/empty.bam" --ref "$out/tiny.fa" --out
expect_error 2 "--chroms needs a value" "${base_args[@]}" --chroms ""
expect_error 2 "--depth must be auto or MIN:MAX" "${base_args[@]}" --depth 10
expect_error 2 "--depth must be auto or MIN:MAX" "${base_args[@]}" --depth 50:10
expect_error 2 "--threads must be a whole number" "${base_args[@]}" --threads two
expect_error 2 "--max-depth 60 is less than twice the upper depth bound 46 of --depth 8:46" \
  "${base_args[@]}" --depth 8:46 --max-depth 60
expect_error 2 "needs --run N_ITER" "${base_args[@]}" -- --pattern 4+5*3+4
expect_error 2 "unknown argument '--frobnicate'" "${base_args[@]}" --frobnicate
expect_error 2 "give the bin size as this script's --bin-size" "${base_args[@]}" --run 2 \
  -- --bin-size=50
expect_error 2 "--tui does not go with this script" "${base_args[@]}" --run 2 -- --tui
expect_error 1 "psmc-rs prints its help or version instead of fitting" "${base_args[@]}" \
  --run 2 -- --version
expect_error 1 "cannot read --bam" --bam "$out/missing.bam" --ref "$out/tiny.fa" --out "$out/x"
expect_error 1 "cannot read --ref" --bam "$out/empty.bam" --ref "$out" --out "$out/x"

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

# The versions come from the first line of standard output that has one.
# What the dynamic loader writes on standard error before the program runs,
# as glibc 2.34 to 2.38 do for a binary linked on 2.39, is not read.
cat > "$out/fakebin/psmc-rs-warn" << EOF
#!/bin/sh
echo "$PSMC_RS: /lib/x86_64-linux-gnu/libc.so.6: weak version \\\`GLIBC_2.39' not found (required by $PSMC_RS)" >&2
exec "$PSMC_RS" "\$@"
EOF
cat > "$out/fakebin/bcftools-warn" << EOF
#!/bin/sh
echo "bcftools: /lib/x86_64-linux-gnu/libcurl.so.4: no version information available (required by bcftools)" >&2
if [ "\$1" = --version ]; then echo "Loading bcftools from the site's modules"; fi
exec "$(command -v bcftools)" "\$@"
EOF
printf '#!/bin/sh\necho "psmc-rs, a test stand-in"\necho "psmc-rs 0.2.0" >&2\n' \
  > "$out/fakebin/psmc-rs-stderr"
chmod +x "$out/fakebin/psmc-rs-warn" "$out/fakebin/bcftools-warn" "$out/fakebin/psmc-rs-stderr"
BCFTOOLS=$out/fakebin/bcftools-warn run_script "${base_args[@]}" \
  --psmc-rs "$out/fakebin/psmc-rs-warn" --dry-run > /dev/null 2> "$out/err.txt" ||
  fail "a loader warning before the versions: $(cat "$out/err.txt")"
ok "versions read past a loader warning on standard error and a line before them"
expect_error 1 "cannot read the psmc-rs version from 'psmc-rs, a test stand-in" \
  "${base_args[@]}" --psmc-rs "$out/fakebin/psmc-rs-stderr"

# An argument as the dry run prints it: printf %q of the script's bash.
quoted() {
  # shellcheck disable=SC2016
  "${E2E_BASH:-bash}" -c 'printf %q "$1"' quoted "$1"
}

# The fit gets --threads only when it is given, and otherwise picks its own
# default. The script's default follows the CPU affinity (checked where
# taskset exists, on Linux).
run_script "${base_args[@]}" --run 2 --dry-run > "$out/dry_fit.txt"
fitline=$(grep -F -- "$(quoted "$out/x.psmcfa") $(quoted "$out/x.json") 2 " "$out/dry_fit.txt") ||
  fail "no fit in the dry run: $(cat "$out/dry_fit.txt")"
case "$fitline" in
  *--threads*) fail "the fit gets --threads although none was given: $fitline" ;;
esac
run_script "${base_args[@]}" --run 2 --threads 3 --dry-run > "$out/dry_fit.txt"
grep -qF -- "$(quoted "$out/x.json") 2 --bin-size 100 --threads 3 " "$out/dry_fit.txt" ||
  fail "the fit does not get --threads 3: $(cat "$out/dry_fit.txt")"
ok "the fit gets --threads only when it is given"
if command -v taskset > /dev/null; then
  taskset -c 0 ${E2E_BASH:+"$E2E_BASH"} "$script" "${base_args[@]}" --dry-run > "$out/dry_cpu.txt"
  grep -qF "1 at a time" "$out/dry_cpu.txt" || fail "taskset -c 0: $(cat "$out/dry_cpu.txt")"
  ok "the default number of jobs follows the CPU affinity (taskset -c 0: 1 at a time)"
else
  printf 'skipped: no taskset, so the CPU affinity default is not checked\n'
fi

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
  > "$out/dry_run.txt" 2> "$out/dry_run.err"
"$PSMC_RS" qc "$out/names.psmcfa" --json "$out/names.qc.json" > /dev/null 2>&1
check kinds "$out/dry_run.txt" "$out/names.qc.json"

# Roman numerals: in a stickleback-like reference chrX is chromosome 10 and
# is kept; in a C. elegans-like one (I to V, X) X is the sex chromosome.
fai_of() { printf '%s\t4\t0\t4\t5\n' "$@"; } # name... > .fai lines
called_in() { # dry-run output: the chromosomes to call, on one line
  awk '/^Chromosomes to call/ { on = 1; next } /^[^ ]/ { on = 0 } on { print $1 }' "$1" |
    tr '\n' ' '
}
touch "$out/roman.fa" "$out/elegans.fa"
fai_of chrI chrII chrIII chrIV chrV chrVI chrVII chrVIII chrIX chrX chrXI chrXII chrXIII chrXIV \
  chrXV chrXVI chrXVII chrXVIII chrXIX chrXX chrXXI chrUn chrM > "$out/roman.fa.fai"
run_script --bam "$out/empty.bam" --ref "$out/roman.fa" --out "$out/roman" --dry-run \
  > "$out/dry_roman.txt" 2> "$out/dry_roman.err"
case "$(called_in "$out/dry_roman.txt")" in
  *" chrIX chrX chrXI "*"chrXIX chrXX chrXXI chrUn ") ;;
  *) fail "Roman numerals: called $(called_in "$out/dry_roman.txt")" ;;
esac
grep -qF "Left out by name (1):" "$out/dry_roman.txt" || fail "$(cat "$out/dry_roman.txt")"
grep -qF "kept chrX: the reference numbers its chromosomes in Roman numerals" \
  "$out/dry_roman.err" || fail "no note about chrX: $(cat "$out/dry_roman.err")"
fai_of I II III IV V X MtDNA > "$out/elegans.fa.fai"
run_script --bam "$out/empty.bam" --ref "$out/elegans.fa" --out "$out/elegans" --dry-run \
  > "$out/dry_elegans.txt"
[ "$(called_in "$out/dry_elegans.txt")" = "I II III IV V " ] ||
  fail "C. elegans names: called $(called_in "$out/dry_elegans.txt")"
ok "Roman numerals: chrX kept as chromosome 10 next to chrIX and chrXI; X left out after I to V"

# A GRCh38-like reference: the default selection calls the unplaced, ALT,
# decoy, viral and HLA contigs, with a warning that names them.
touch "$out/grch38.fa"
fai_of chr1 chr2 chrX chrY chrM chr1_KI270706v1_random chrUn_KI270302v1 chr6_GL000250v2_alt \
  chrUn_JTFH01000001v1_decoy chrEBV 'HLA-A*01:01:01:01' > "$out/grch38.fa.fai"
run_script --bam "$out/empty.bam" --ref "$out/grch38.fa" --out "$out/grch38" --dry-run \
  > "$out/dry_grch38.txt" 2> "$out/dry_grch38.err"
grep -qF "warning: the contigs to call include 6 (chr1_KI270706v1_random, chrUn_KI270302v1, chr6_GL000250v2_alt, ...) that look like unplaced, ALT, decoy, patch, viral or HLA contigs" \
  "$out/dry_grch38.err" || fail "no warning about the GRCh38 contigs: $(cat "$out/dry_grch38.err")"
run_script --bam "$out/empty.bam" --ref "$out/grch38.fa" --out "$out/grch38" --dry-run \
  --chroms chr1,chr2 > /dev/null 2> "$out/dry_grch38.err"
if grep -qF "look like unplaced" "$out/dry_grch38.err"; then
  fail "a warning about other contigs although --chroms lists the autosomes"
fi
ok "GRCh38-like contigs (unplaced, ALT, decoy, EBV, HLA): called by default with a warning, none with --chroms"

# --chroms from a BED file with Windows line ends, a track and a browser
# line, and several rows for one chromosome: its names, each once.
printf 'track name=callable\r\nbrowser position chr1:1-4\r\nchr1\t0\t2\r\nchr1\t2\t4\r\nscaffold_12\t0\t4\r\n' \
  > "$out/callable.bed"
run_script --bam "$out/empty.bam" --ref "$out/names.fa" --out "$out/names" --dry-run \
  --chroms "$out/callable.bed" > "$out/dry_bed.txt"
called=$(called_in "$out/dry_bed.txt")
[ "$called" = "chr1 scaffold_12 " ] || fail "--chroms callable.bed calls: $called"
ok "--chroms from a BED file (CRLF, track and browser lines, several rows per chromosome): $called"

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
expect_files "$p.vcf.gz" "$p.vcf.gz.tbi" "$p.qc.txt" "$p.qc.json" "$p.psmcfa" "$p.json" \
  "$p.html" "$p.psmc" "$p.log" "$p.chr1.multihetsep"
[ -z "$(find "$out" -name '.*tmp*')" ] || fail "working directory left behind in $out"
grep -qF "left out by name: chrX (sex chromosome), chrM (organelle)" "$work/run1.err" ||
  fail "the log does not list chrX and chrM as left out"
contigs=$(vcf_contigs "$p.vcf.gz")
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
contigs=$(vcf_contigs "$q.vcf.gz")
[ "$contigs" = "chr1 chr2 chr3 chrX " ] || fail "VCF contigs with records: $contigs"
expect_classic "$q" "run 3"
check qc "$q.qc.json" --sex-warning chrX --kept chr1,chr2,chr3,chrX --dropped 0

# ---------------------------------------------------------------------------
step "reruns: the outputs of an earlier run that no longer match go"

r=$out/re
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$r" --chroms chr3 --multihetsep \
  --run 1 2> "$work/re1.err"
expect_files "$r.json" "$r.html" "$r.psmc" "$r.chr3.multihetsep"
# The same psmcfa: the fit and the multihetsep file still match and stay.
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$r" --chroms chr3 \
  2> "$work/re2.err"
for f in "$r.json" "$r.html" "$r.psmc" "$r.chr3.multihetsep"; do
  [ -s "$f" ] || fail "$f removed although the psmcfa did not change"
done
if grep -qF "removed the outputs of an earlier run" "$work/re2.err"; then
  fail "outputs reported stale although the psmcfa did not change"
fi
# Another psmcfa and a fit: the fit's files are written again, and the
# multihetsep file, which this run does not write, goes with a warning.
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$r" --chroms chr2,chr3 \
  --run 1 2> "$work/re3.err"
grep -qF "removed the outputs of an earlier run, made from another $r.psmcfa: 1 file(s) $r.<chrom>.multihetsep" \
  "$work/re3.err" || fail "no warning about the removed multihetsep file: $(cat "$work/re3.err")"
[ ! -e "$r.chr3.multihetsep" ] || fail "$r.chr3.multihetsep of the earlier run is still there"
check fit "$r.json" "$r.psmc" 1 2
# Another psmcfa and no fit: the fit's files go, with a warning; a
# bootstrap directory and the working directory of a killed run are
# reported.
mkdir "$r.bootstrap" "$out/.re.tmp.killed"
run_script --bam "$sim/aln.sam.gz" --ref "$sim/ref.fa" --out "$r" --chroms chr3 \
  2> "$work/re4.err"
for f in "$r.json" "$r.html" "$r.psmc"; do
  [ ! -e "$f" ] || fail "$f of the earlier run is still there"
done
grep -qF "removed the outputs of an earlier run, made from another $r.psmcfa: $r.json, $r.html, $r.psmc" \
  "$work/re4.err" || fail "no warning about the removed outputs: $(cat "$work/re4.err")"
grep -qF "$r.bootstrap holds the bootstrap of an earlier run" "$work/re4.err" ||
  fail "no warning about $r.bootstrap"
grep -qF "working directories of earlier runs that did not finish: $out/.re.tmp.killed;" \
  "$work/re4.err" || fail "no warning about $out/.re.tmp.killed"
rmdir "$r.bootstrap" "$out/.re.tmp.killed"
ok "a rerun with the same psmcfa keeps the fit and multihetsep files; with another, removes those it does not write again and says so"

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
  contigs=$(vcf_contigs "$c/$bam.out.vcf.gz")
  [ "$contigs" = "$name " ] || fail "$bam: VCF contigs with records: $contigs"
  [ "$(grep '^>' "$c/$bam.out.psmcfa")" = ">$name" ] || fail "$bam: no psmcfa record for $name"
done
cmp "$c/aln.sam.out.psmcfa" "$c/aln.sam.gz.out.psmcfa" || fail "-T and -R calls differ"
ok "$name called by region and streamed, with the same psmcfa"

# ---------------------------------------------------------------------------
step "route B of docs/preprocessing.md: variant calls and a callable mask from a gVCF"

# The guide's commands, on the indexed reads; the depth bounds are those of
# run 1, whose VCF holds the same chromosomes but chrUn_scaffold1.
b=$work/routeb
mkdir -p "$b"
read -r mean dmin dmax < <(python3 "$helpers" auto-depth "$p.vcf.gz")
# The guide's awk program, verbatim.
# shellcheck disable=SC2016
awk -v OFS='\t' '$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ { print $1, 1, $2 }' "$sim/ref.fa.fai" \
  > "$b/autosomes.txt"
[ "$(cut -f 1 "$b/autosomes.txt" | tr '\n' ' ')" = "chr1 chr2 chr3 " ] ||
  fail "autosomes.txt: $(cat "$b/autosomes.txt")"
bcftools mpileup -C50 -q20 -Q20 -d 10000 -a FORMAT/DP -R "$b/autosomes.txt" -f "$sim/ref.fa" \
  "$sim/aln.sam.gz" 2> /dev/null |
  bcftools call -m -g "$dmin,$((dmax + 1))" -Oz -o "$b/sample.g.vcf.gz" 2> /dev/null
contigs=$(bcftools query -f '%CHROM\n' "$b/sample.g.vcf.gz" 2> /dev/null | uniq | tr '\n' ' ')
[ "$contigs" = "chr1 chr2 chr3 " ] || fail "the gVCF holds $contigs"
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
step "high depth: the upper depth bound still applies above 250 reads"

# chr1 at 140x, with 200x of paralog reads in the collapsed repeat: about
# 340 reads there, and an automatic upper bound of about 280. With
# mpileup's default -d 250 the repeat's depth stayed below the bound and 20
# of its 25 bins came out K; the script's -d 10000 counts every read.
deep=$work/deep
python3 "$here/simulate.py" "$deep/sim" --depth 140 --paralog-depth 200 --chroms chr1
run_script --bam "$deep/sim/aln.sam" --ref "$deep/sim/ref.fa" --out "$deep/x" 2> "$deep/x.err"
grep -qF -- " -d 10000 " "$deep/x.log" || fail "mpileup did not run with -d 10000"
check region "$deep/sim" "$deep/x.psmcfa" 100 collapsed_repeat --max-k 0.5
expect_classic "$deep/x" "140x"

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
only_log_left() { [ -z "$(leftovers "$1")" ] || fail "outputs left: $(leftovers "$1")"; }

# The script, with prefix $out/NAME and the other arguments, must fail (exit
# 1) with MESSAGE on standard error, which it keeps in $work/NAME.err.
expect_failure() { # name, message, arguments...
  local name=$1 msg=$2 s=0
  shift 2
  run_script --out "$out/$name" "$@" 2> "$work/$name.err" || s=$?
  [ "$s" -eq 1 ] || fail "$name: exit status $s, not 1"
  grep -qF -- "$msg" "$work/$name.err" || fail "$(cat "$work/$name.err")"
}

expect_failure fail "bcftools called no site" \
  --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr3 --min-mapq 255
only_log_left fail
[ -z "$(find "$out" -name '.*tmp*')" ] || fail "working directory left behind in $out"
ok "no site called: error, only fail.log written"

# Two samples in one file (read groups with different SM tags).
awk 'BEGIN { OFS = "\t" }
  /^@RG/ { print; print "@RG\tID:run2\tSM:sim2"; next }
  /^@/ { print; next }
  { if (NR % 2) sub(/RG:Z:run1$/, "RG:Z:run2"); print }' "$sim/aln.sam" > "$work/two.sam"
expect_failure two "the alignments hold 2 samples" \
  --bam "$work/two.sam" --ref "$sim/ref.fa" --chroms chr3
only_log_left two
ok "two samples: error, only two.log written"

# With an index, the samples are checked before any chromosome is called.
bgzip -c "$work/two.sam" > "$work/two.sam.gz"
tabix -C -p sam "$work/two.sam.gz"
expect_failure two-indexed "the alignments hold 2 samples" \
  --bam "$work/two.sam.gz" --ref "$sim/ref.fa" --chroms chr1,chr3
if grep -qF '# N = 1..' "$out/two-indexed.log"; then
  fail "the calls started before the samples were checked"
fi
only_log_left two-indexed
ok "two samples, indexed: error before any chromosome is called"

# A mistake in the fit's options stops the run before the calls, with
# psmc-rs's own error line and tip.
expect_failure typo "error: unexpected argument '--patern' found" \
  --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr3 --run 1 -- --patern '4+5*3+4'
grep -qF "tip: a similar argument exists: '--pattern'" "$work/typo.err" || fail "$(cat "$work/typo.err")"
if grep -qF '# N = 1..' "$out/typo.log"; then
  fail "the calls started before the fit's options were checked"
fi
only_log_left typo
ok "a typo in the fit's options: error before any chromosome is called, with psmc-rs's error and tip"

# A fit that fails only once it has read the input shows psmc-rs's error
# line, which follows pages of QC text.
expect_failure fitfail "error: --bootstrap-block-size is too large for this input" \
  --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr3 --run 1 \
  -- --bootstrap 2 --bootstrap-block-size 100000000
ok "a fit that fails: its error line is shown"

# A --max-depth below twice the automatic upper bound (about 50 at 25x).
expect_failure capped "more than --max-depth 60" \
  --bam "$sim/aln.sam" --ref "$sim/ref.fa" --chroms chr3 --max-depth 60
only_log_left capped
ok "--max-depth below twice the automatic upper bound: error, only capped.log written"

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
only_log_left int
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
