#!/usr/bin/env bash
# bam_to_psmc.sh: from the aligned reads of one diploid genome to psmc-rs
# input through the classic PSMC consensus:
#
#   bcftools mpileup -C50 -q20 -Q20 -f ref.fa aln.bam | bcftools call -c
#
# per chromosome in parallel, then one bgzipped, indexed all-sites VCF,
# `psmc-rs qc` and `psmc-rs convert` (psmcfa, optionally multihetsep), and
# optionally an inference run. Run with --help for the options.
#
# Works with the bash 3.2 that macOS ships: no associative arrays, no
# `wait -n`, and empty arrays are never expanded under `set -u`.

set -euo pipefail

readonly SCRIPT_VERSION="1.0"
readonly MIN_BCFTOOLS="1.10"
readonly MIN_PSMC_RS="0.2.0"
prog=$(basename "$0")

usage() {
  cat <<EOF
Usage: $prog --bam FILE --ref FILE --out PREFIX [options] [-- PSMC_RS_OPTIONS]

Call the all-sites consensus of one diploid sample with bcftools (mpileup |
call -c, per chromosome in parallel), write it as a bgzipped, indexed VCF,
check it with \`psmc-rs qc\`, convert it to psmcfa with \`psmc-rs convert\`
(the same bytes as vcfutils.pl vcf2fq | fq2psmcfa) and optionally fit PSMC.

Required:
  --bam FILE          Alignments of one sample: BAM, CRAM or SAM, coordinate
                      sorted. With an index (.bai, .csi, .crai) each job reads
                      only its chromosome; without one, every job streams the
                      whole file.
  --ref FILE          The reference the reads were aligned to (FASTA, plain or
                      bgzip). Its .fai is created next to it if missing.
  --out PREFIX        Output prefix; the directory is created if needed.

Options:
  --chroms LIST|FILE  Chromosomes to use: a comma-separated list, or a file
                      with one name per line (the first column, so a .fai or
                      BED works too). Default: every contig of the reference
                      not named like a sex chromosome or organelle (X, Y, Z,
                      W, M, MT, chrX, chrM, ..., the rule of \`psmc-rs qc\`).
  --threads N         Chromosomes called at once, and psmc-rs threads
                      [number of CPUs: $(default_threads)]
  --min-mapq N        mpileup -q: minimum mapping quality of a read [20]
                      (vcf2fq's -Q, on the RMS mapping quality of a site,
                      stays at its default 10)
  --min-baseq N       mpileup -Q: minimum base quality [20]
  --depth auto|MIN:MAX
                      Depth bounds of a called base (vcf2fq -d, -D). auto: a
                      third and twice the mean depth (psmc README) [auto]
  --min-qual N        fq2psmcfa -q: minimum consensus quality [20, as in the
                      psmc README]
  --bin-size N        fq2psmcfa -s: bases per bin [100]
  --multihetsep       Also write PREFIX.<chrom>.multihetsep (for MSMC2)
  --run N_ITER        Also fit PSMC with N_ITER EM iterations: PREFIX.json,
                      PREFIX.html and PREFIX.psmc. Options after \`--\` go to
                      this psmc-rs run (e.g. -- --pattern '4+25*2+4+6')
  --psmc-rs PATH      The psmc-rs binary [\$PSMC_RS, else psmc-rs on PATH]
  --keep-tmp          Keep the working directory (per-chromosome calls, logs)
  --dry-run           Print the chromosomes and commands, run nothing
  -h, --help          Show this help
  --version           Show the script version

Outputs: PREFIX.vcf.gz and .tbi (all sites, bcftools call -c),
PREFIX.qc.txt and PREFIX.qc.json (psmc-rs qc), PREFIX.psmcfa, with
--multihetsep PREFIX.<chrom>.multihetsep, with --run PREFIX.json,
PREFIX.html and PREFIX.psmc, and PREFIX.log (every command run).

The VCF, QC reports, psmcfa and multihetsep files are written in a working
directory next to them and renamed when complete, so an interrupted run
leaves none of them half written; psmc-rs writes the fit's files itself. A
rerun with the same inputs writes the same bytes to every output but the
log, whatever --threads is. bcftools >= $MIN_BCFTOOLS and psmc-rs >=
$MIN_PSMC_RS are required.
EOF
}

default_threads() {
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

# ---------------------------------------------------------------------------
# Messages and logging

log_file=""

say() {
  printf '[%s] %s\n' "$prog" "$*" >&2
  if [ -n "$log_file" ]; then
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$log_file"
  fi
}

warn() {
  say "warning: $*"
}

die() {
  say "error: $*"
  exit 1
}

usage_error() {
  printf '%s: %s\nRun %s --help for the options.\n' "$prog" "$*" "$prog" >&2
  exit 2
}

# A command line as the shell would read it back.
quoted() {
  local out="" a
  for a in "$@"; do
    out="$out $(printf '%q' "$a")"
  done
  printf '%s' "${out# }"
}

# Log a command (or a pipeline, given as one string) before it runs.
log_cmd() {
  say "+ $*"
}

# ---------------------------------------------------------------------------
# Arguments

command_line=$(quoted "$0" "$@")
bam="" ref="" out="" chroms_arg="" threads="" min_mapq=20 min_baseq=20
depth="auto" min_qual=20 bin_size=100 multihetsep=0 run_iters="" psmc_rs=""
keep_tmp=0 dry_run=0
extra=()

need_value() {
  [ "$#" -ge 2 ] && [ -n "$2" ] || usage_error "$1 needs a value"
}

is_uint() {
  case "$1" in
    '' | *[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bam) need_value "$@"; bam=$2; shift 2 ;;
    --ref) need_value "$@"; ref=$2; shift 2 ;;
    --out) need_value "$@"; out=$2; shift 2 ;;
    --chroms) need_value "$@"; chroms_arg=$2; shift 2 ;;
    --threads) need_value "$@"; threads=$2; shift 2 ;;
    --min-mapq) need_value "$@"; min_mapq=$2; shift 2 ;;
    --min-baseq) need_value "$@"; min_baseq=$2; shift 2 ;;
    --depth) need_value "$@"; depth=$2; shift 2 ;;
    --min-qual) need_value "$@"; min_qual=$2; shift 2 ;;
    --bin-size) need_value "$@"; bin_size=$2; shift 2 ;;
    --multihetsep) multihetsep=1; shift ;;
    --run) need_value "$@"; run_iters=$2; shift 2 ;;
    --psmc-rs) need_value "$@"; psmc_rs=$2; shift 2 ;;
    --keep-tmp) keep_tmp=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --version) echo "$prog $SCRIPT_VERSION"; exit 0 ;;
    --) shift; extra=("$@"); break ;;
    *) usage_error "unknown argument '$1'" ;;
  esac
done

[ -n "$bam" ] || usage_error "--bam is required"
[ -n "$ref" ] || usage_error "--ref is required"
[ -n "$out" ] || usage_error "--out is required"
[ -n "$threads" ] || threads=$(default_threads)
for pair in "--threads:$threads" "--min-mapq:$min_mapq" "--min-baseq:$min_baseq" \
  "--min-qual:$min_qual" "--bin-size:$bin_size"; do
  is_uint "${pair#*:}" || usage_error "${pair%%:*} must be a whole number, not '${pair#*:}'"
done
[ "$threads" -ge 1 ] || usage_error "--threads must be at least 1"
[ "$bin_size" -ge 1 ] || usage_error "--bin-size must be at least 1"
if [ "$depth" != auto ]; then
  case "$depth" in
    *:*) dmin=${depth%%:*} dmax=${depth#*:} ;;
    *) dmin="" dmax="" ;;
  esac
  if ! is_uint "$dmin" || ! is_uint "$dmax" || [ "$dmin" -gt "$dmax" ]; then
    usage_error "--depth must be auto or MIN:MAX with MIN <= MAX, not '$depth'"
  fi
fi
# Whether the options after -- include $1.
has_extra() {
  local a
  for a in ${extra[@]+"${extra[@]}"}; do
    case "$a" in
      "$1" | "$1"=*) return 0 ;;
    esac
  done
  return 1
}
if [ -n "$run_iters" ]; then
  is_uint "$run_iters" || usage_error "--run must be a whole number of EM iterations, not '$run_iters'"
elif [ "${#extra[@]}" -gt 0 ]; then
  usage_error "options after -- go to the inference run, which needs --run N_ITER"
fi
! has_extra --bin-size ||
  usage_error "give the bin size as this script's --bin-size, which also sets the psmcfa bins"
case "$out" in
  */) usage_error "--out is a prefix for file names, not a directory: '$out'" ;;
esac

[ -f "$bam" ] && [ -r "$bam" ] || die "cannot read --bam $bam"
[ -f "$ref" ] && [ -r "$ref" ] || die "cannot read --ref $ref"

# ---------------------------------------------------------------------------
# Tools

# Whether version $1 (x.y or x.y.z) is at least $2.
version_at_least() {
  awk -v have="$1" -v need="$2" 'BEGIN {
    n = split(have, h, "."); m = split(need, w, ".")
    for (i = 1; i <= (n > m ? n : m); i++) {
      a = (i <= n ? h[i] + 0 : 0); b = (i <= m ? w[i] + 0 : 0)
      if (a != b) exit (a > b ? 0 : 1)
    }
    exit 0
  }'
}

bcftools=${BCFTOOLS:-bcftools}
command -v "$bcftools" > /dev/null 2>&1 ||
  die "bcftools not found (install bcftools >= $MIN_BCFTOOLS, or set BCFTOOLS to its path)"
bcftools_out=$("$bcftools" --version 2>&1) || die "'$bcftools --version' failed: $bcftools_out"
bcftools_line=${bcftools_out%%$'\n'*}
bcftools_version=$(printf '%s\n' "$bcftools_line" |
  sed -n 's/^bcftools \([0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)*\).*/\1/p')
[ -n "$bcftools_version" ] ||
  die "cannot read the bcftools version from '$bcftools_line'"
version_at_least "$bcftools_version" "$MIN_BCFTOOLS" ||
  die "bcftools $bcftools_version is too old: $MIN_BCFTOOLS or later is needed"

if [ -z "$psmc_rs" ]; then
  psmc_rs=${PSMC_RS:-psmc-rs}
fi
command -v "$psmc_rs" > /dev/null 2>&1 ||
  die "psmc-rs not found (build it with cargo build --release, then give --psmc-rs PATH or set PSMC_RS)"
psmc_rs_out=$("$psmc_rs" --version 2>&1) || die "'$psmc_rs --version' failed: $psmc_rs_out"
psmc_rs_version=$(printf '%s\n' "$psmc_rs_out" |
  sed -n '1s/^psmc-rs \([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
[ -n "$psmc_rs_version" ] || die "cannot read the psmc-rs version from '$psmc_rs_out'"
version_at_least "$psmc_rs_version" "$MIN_PSMC_RS" ||
  die "psmc-rs $psmc_rs_version is too old: $MIN_PSMC_RS or later is needed (for qc and convert)"

# ---------------------------------------------------------------------------
# Reference index and chromosomes

fai="$ref.fai"
make_fai() {
  # htslib builds the .fai when a command first opens the FASTA; an empty
  # VCF through `bcftools norm -f` does that without reading any reads.
  log_cmd "bcftools norm -f $(quoted "$ref") (creates $(quoted "$fai"))"
  printf '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n' |
    "$bcftools" norm --no-version -f "$ref" -o /dev/null > /dev/null 2>&1 || true
  [ -s "$fai" ] ||
    die "cannot index $ref (gzip-compressed FASTA must be bgzip-compressed; or create $fai with samtools faidx)"
}

if [ ! -s "$fai" ]; then
  [ "$dry_run" -eq 0 ] || die "$fai does not exist; a dry run does not create it"
  make_fai
fi

# The rule of psmc-rs's qc::classify_name: prints "sex chromosome",
# "organelle" or nothing for each name on stdin.
classify_names() {
  LC_ALL=C awk '
    BEGIN {
      refseq["nc_000023"] = "sex chromosome"; refseq["nc_000024"] = "sex chromosome"
      refseq["nc_012920"] = "organelle"; refseq["nc_001807"] = "organelle"
      refseq["nc_000086"] = "sex chromosome"; refseq["nc_000087"] = "sex chromosome"
      refseq["nc_005089"] = "organelle"
    }
    function kind(name,    lower, acc, rest, prefixed, tok) {
      lower = tolower(name)
      if (lower ~ /mitochondri|chloroplast|plastid|plastome|apicoplast/) return "organelle"
      acc = lower; sub(/\..*/, "", acc)
      if (acc in refseq) return refseq[acc]
      rest = lower; prefixed = 1
      if (substr(rest, 1, 10) == "chromosome") rest = substr(rest, 11)
      else if (substr(rest, 1, 5) == "chrom") rest = substr(rest, 6)
      else if (substr(rest, 1, 3) == "chr") rest = substr(rest, 4)
      else prefixed = 0
      if (prefixed && rest ~ /^[-_.]/) rest = substr(rest, 2)
      tok = rest
      if (match(rest, /[^a-z0-9]/)) tok = substr(rest, 1, RSTART - 1)
      if (tok ~ /^[xyzw][0-9]*$/) return "sex chromosome"
      if (tok ~ /^(m|mt|mito|mtdna|pt|cp)$/) return "organelle"
      if (tok == "c" && prefixed) return "organelle"
      return ""
    }
    { print kind($0) }
  '
}

# Contigs of the reference, in its order: names and lengths.
ref_names=()
ref_lengths=()
while IFS=$'\t' read -r name len _; do
  [ -n "$name" ] || continue
  ref_names+=("$name")
  ref_lengths+=("$len")
done < "$fai"
[ "${#ref_names[@]}" -gt 0 ] || die "$fai lists no sequence"

ref_kinds=()
while IFS= read -r k; do
  ref_kinds+=("$k")
done < <(printf '%s\n' "${ref_names[@]}" | classify_names)

# Names asked for with --chroms, one per line.
requested_names() {
  if [ -f "$chroms_arg" ]; then
    awk '!/^[ \t]*(#|$)/ { print $1 }' "$chroms_arg"
  else
    printf '%s\n' "$chroms_arg" | tr ',' '\n' | awk 'NF { print $1 }'
  fi
}

# Indices into ref_names of the chromosomes to call, in reference order.
selected=()
excluded=()
if [ -z "$chroms_arg" ]; then
  i=0
  while [ "$i" -lt "${#ref_names[@]}" ]; do
    if [ -n "${ref_kinds[$i]}" ]; then
      excluded+=("${ref_names[$i]} (${ref_kinds[$i]})")
    else
      selected+=("$i")
    fi
    i=$((i + 1))
  done
else
  wanted=$(requested_names)
  [ -n "$wanted" ] || die "--chroms $chroms_arg names no chromosome"
  dup=$(printf '%s\n' "$wanted" | awk 'seen[$0]++ { print; exit }')
  [ -z "$dup" ] || die "--chroms names $dup twice"
  unknown=$(awk -F '\t' 'NR == FNR { have[$1] = 1; next } !($1 in have) { print; exit }' \
    "$fai" <(printf '%s\n' "$wanted"))
  [ -z "$unknown" ] || die "--chroms: '$unknown' is not a sequence of $ref (see $fai)"
  while IFS= read -r i; do
    selected+=("$i")
    if [ -n "${ref_kinds[$i]}" ]; then
      warn "${ref_names[$i]} is named like a ${ref_kinds[$i]}; PSMC assumes two copies sharing one autosomal history, so most analyses leave it out"
    fi
  done < <(awk -F '\t' 'NR == FNR { want[$0] = 1; next }
      $1 != "" { if ($1 in want) print k + 0; k++ }' <(printf '%s\n' "$wanted") "$fai")
fi
[ "${#selected[@]}" -gt 0 ] || die "no chromosome is left to call (all are named like sex chromosomes or organelles; list them with --chroms)"

# ---------------------------------------------------------------------------
# Outputs

out_dir=$(dirname "$out")
base=$(basename "$out")
vcf="$out.vcf.gz"
psmcfa="$out.psmcfa"
qc_txt="$out.qc.txt"
qc_json="$out.qc.json"
log_path="$out.log"

# The alignments are read by region when they have an index.
indexed=0
for idx in "$bam.bai" "$bam.csi" "$bam.crai" "${bam%.*}.bai" "${bam%.*}.crai"; do
  if [ -f "$idx" ]; then
    indexed=1
    break
  fi
done

common_input_opts=(--input-format vcf --vcf-mode consensus --depth "$depth"
  --min-qual "$min_qual" --bin-size "$bin_size")

mpileup_cmd() { # $1: regions file
  local where=-T
  [ "$indexed" -eq 0 ] || where=-R
  printf '%s' "$(quoted "$bcftools" mpileup --no-version -C50 -q "$min_mapq" -Q "$min_baseq" \
    -f "$ref" "$where" "$1" "$bam") | $(quoted "$bcftools" call --no-version -c -Ob -o)"
}

if [ "$dry_run" -eq 1 ]; then
  echo "Chromosomes to call (${#selected[@]}):"
  for i in "${selected[@]}"; do
    echo "  ${ref_names[$i]}"
  done
  if [ "${#excluded[@]}" -gt 0 ]; then
    echo "Left out by name (${#excluded[@]}):"
    for e in "${excluded[@]}"; do
      echo "  $e"
    done
  fi
  echo "Commands:"
  echo "  $(mpileup_cmd "TMP/regions/N.txt") TMP/calls/N.bcf   # per chromosome, $threads at a time"
  echo "  $(quoted "$bcftools" concat --no-version --threads "$threads" -Oz -o "$vcf" -f TMP/calls.txt)"
  echo "  $(quoted "$bcftools" index -t "$vcf")"
  echo "  $(quoted "$psmc_rs" qc "$vcf" "${common_input_opts[@]}" --json "$qc_json") > $(quoted "$qc_txt")"
  conv=("$psmc_rs" convert "$vcf" "${common_input_opts[@]}" -q --psmcfa "$psmcfa")
  [ "$multihetsep" -eq 0 ] || conv+=(--multihetsep-per-chrom "$out")
  echo "  $(quoted "${conv[@]}")"
  if [ -n "$run_iters" ]; then
    echo "  $(quoted "$psmc_rs" "$psmcfa" "$out.json" "$run_iters" --bin-size "$bin_size" \
      --threads "$threads" --psmc-out "$out.psmc" --no-progress ${extra[@]+"${extra[@]}"})"
  fi
  exit 0
fi

mkdir -p "$out_dir" || die "cannot create the output directory $out_dir"
: > "$log_path" || die "cannot write $log_path"
log_file=$log_path

tmp=$(mktemp -d "$out_dir/.$base.tmp.XXXXXX") || die "cannot create a working directory in $out_dir"
pids=()
job_names=()

cleanup() {
  local status=$?
  local p
  for p in ${pids[@]+"${pids[@]}"}; do
    pkill -TERM -P "$p" 2> /dev/null || true
    kill -TERM "$p" 2> /dev/null || true
  done
  if [ "$keep_tmp" -eq 1 ]; then
    say "working directory kept: $tmp"
  else
    rm -rf "$tmp"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

say "bam_to_psmc.sh $SCRIPT_VERSION: $command_line"
say "bcftools $bcftools_version ($(command -v "$bcftools")), psmc-rs $psmc_rs_version ($(command -v "$psmc_rs"))"
at_once=$threads
[ "$at_once" -le "${#selected[@]}" ] || at_once=${#selected[@]}
say "calling ${#selected[@]} chromosome(s), $at_once at a time"
if [ "${#excluded[@]}" -gt 0 ]; then
  say "left out by name: $(printf '%s, ' "${excluded[@]}" | sed 's/, $//')"
fi
[ "$indexed" -eq 1 ] || [ "${#selected[@]}" -eq 1 ] ||
  warn "$bam has no index (.bai, .csi or .crai), so every chromosome job reads the whole file; index it (samtools index) to read each chromosome directly"

# ---------------------------------------------------------------------------
# 1. Per-chromosome calls

mkdir -p "$tmp/regions" "$tmp/calls"

call_one() { # $1: number of the job, $2: index into ref_names
  local n=$1 i=$2
  local where=-T
  [ "$indexed" -eq 0 ] || where=-R
  printf '%s\t1\t%s\n' "${ref_names[$i]}" "${ref_lengths[$i]}" > "$tmp/regions/$n.txt"
  "$bcftools" mpileup --no-version -C50 -q "$min_mapq" -Q "$min_baseq" -f "$ref" \
    "$where" "$tmp/regions/$n.txt" "$bam" 2>> "$tmp/calls/$n.log" |
    "$bcftools" call --no-version -c -Ob -o "$tmp/calls/$n.bcf" 2>> "$tmp/calls/$n.log"
}

samples_checked=0

# Wait for the oldest job; stop everything if it failed.
wait_oldest() {
  local p=${pids[0]} name=${job_names[0]} n=${job_numbers[0]}
  if ! wait "$p"; then
    pids=("${pids[@]:1}")
    cat "$tmp/calls/$n.log" >> "$log_file" 2> /dev/null || true
    die "calling $name failed: $(tail -n 3 "$tmp/calls/$n.log" 2> /dev/null | tr '\n' ' ')(full messages in $log_file)"
  fi
  pids=("${pids[@]:1}")
  job_names=("${job_names[@]:1}")
  job_numbers=("${job_numbers[@]:1}")
  cat "$tmp/calls/$n.log" >> "$log_file"
  if [ "$samples_checked" -eq 0 ]; then
    local samples count
    samples=$("$bcftools" query -l "$tmp/calls/$n.bcf" 2>> "$log_file") ||
      die "cannot read the calls of $name"
    count=$(printf '%s\n' "$samples" | awk 'NF' | wc -l | tr -d ' ')
    [ "$count" -eq 1 ] ||
      die "the alignments hold $count samples (read-group SM tags: $(printf '%s\n' "$samples" | tr '\n' ' ' | sed 's/ $//')); PSMC needs the reads of one diploid genome, so give one sample's reads or fix the SM tags"
    say "sample: $samples"
    samples_checked=1
  fi
  say "called $name"
}

log_cmd "$(mpileup_cmd "$tmp/regions/N.txt") $tmp/calls/N.bcf   # N = 1..${#selected[@]}"
job_numbers=()
n=0
for i in "${selected[@]}"; do
  n=$((n + 1))
  while [ "${#pids[@]}" -ge "$threads" ]; do
    wait_oldest
  done
  call_one "$n" "$i" &
  pids+=("$!")
  job_names+=("${ref_names[$i]}")
  job_numbers+=("$n")
done
while [ "${#pids[@]}" -gt 0 ]; do
  wait_oldest
done

# ---------------------------------------------------------------------------
# 2. One all-sites VCF, bgzipped and indexed

n=0
: > "$tmp/calls.txt"
for i in "${selected[@]}"; do
  n=$((n + 1))
  printf '%s\n' "$tmp/calls/$n.bcf" >> "$tmp/calls.txt"
done
log_cmd "$(quoted "$bcftools" concat --no-version --threads "$threads" -Oz -o "$tmp/all.vcf.gz" -f "$tmp/calls.txt")"
"$bcftools" concat --no-version --threads "$threads" -Oz -o "$tmp/all.vcf.gz" \
  -f "$tmp/calls.txt" 2>> "$log_file" || die "bcftools concat failed (see $log_file)"
log_cmd "$(quoted "$bcftools" index -t "$tmp/all.vcf.gz")"
"$bcftools" index -t "$tmp/all.vcf.gz" 2>> "$log_file" || die "bcftools index failed (see $log_file)"
records=$("$bcftools" index -n "$tmp/all.vcf.gz" 2>> "$log_file") ||
  die "cannot count the records of the VCF"
[ "$records" -gt 0 ] ||
  die "bcftools called no site: check that $bam and $ref use the same sequence names (chr1 or 1), and that --min-mapq and --min-baseq leave reads"
say "all-sites VCF: $records records"
rm -f "$vcf.tbi"
mv -f "$tmp/all.vcf.gz" "$vcf"
mv -f "$tmp/all.vcf.gz.tbi" "$vcf.tbi"
[ "$keep_tmp" -eq 1 ] || rm -f "$tmp"/calls/*.bcf

# ---------------------------------------------------------------------------
# 3. Input QC

qc=("$psmc_rs" qc "$vcf" "${common_input_opts[@]}" --json "$tmp/qc.json")
log_cmd "$(quoted "${qc[@]}") > $(quoted "$tmp/qc.txt")"
if ! "${qc[@]}" > "$tmp/qc.txt" 2> "$tmp/qc.err"; then
  cat "$tmp/qc.err" >> "$log_file"
  die "psmc-rs qc failed: $(cat "$tmp/qc.err")"
fi
cat "$tmp/qc.err" >> "$log_file"
cat "$tmp/qc.txt" >> "$log_file"
# The report goes to its final name below, not to the working directory
# that psmc-rs names.
grep -v '^QC report: ' "$tmp/qc.err" >&2 || true
cat "$tmp/qc.txt" >&2
mv -f "$tmp/qc.txt" "$qc_txt"
mv -f "$tmp/qc.json" "$qc_json"

# ---------------------------------------------------------------------------
# 4. psmcfa (and multihetsep)

mkdir -p "$tmp/convert"
conv=("$psmc_rs" convert "$vcf" "${common_input_opts[@]}" -q --psmcfa "$tmp/convert/out.psmcfa")
[ "$multihetsep" -eq 0 ] || conv+=(--multihetsep-per-chrom "$tmp/convert/$base")
log_cmd "$(quoted "${conv[@]}")"
if ! "${conv[@]}" > "$tmp/convert.out" 2> "$tmp/convert.err"; then
  cat "$tmp/convert.err" >> "$log_file"
  die "psmc-rs convert failed: $(cat "$tmp/convert.err")"
fi
# convert reads the input as qc did and repeats its warnings: they go to the
# log only.
cat "$tmp/convert.err" "$tmp/convert.out" >> "$log_file"
mv -f "$tmp/convert/out.psmcfa" "$psmcfa"
if [ "$multihetsep" -eq 1 ]; then
  # Files of an earlier run whose chromosome is no longer called would be
  # stale; every name this script could have written is removed.
  for name in "${ref_names[@]}"; do
    rm -f "$out.$name.multihetsep"
  done
  for f in "$tmp/convert/$base".*.multihetsep; do
    [ -e "$f" ] || continue
    mv -f "$f" "$out_dir/"
  done
fi
grep -q '^>' "$psmcfa" ||
  warn "$psmcfa is empty: every chromosome was dropped (see $qc_txt)"

# ---------------------------------------------------------------------------
# 5. Inference

if [ -n "$run_iters" ]; then
  grep -q '^>' "$psmcfa" || die "nothing to fit: $psmcfa is empty"
  fit=("$psmc_rs" "$psmcfa" "$out.json" "$run_iters" --bin-size "$bin_size")
  has_extra --threads || fit+=(--threads "$threads")
  has_extra --psmc-out || fit+=(--psmc-out "$out.psmc")
  has_extra --no-progress || fit+=(--no-progress)
  fit+=(${extra[@]+"${extra[@]}"})
  log_cmd "$(quoted "${fit[@]}")"
  if ! "${fit[@]}" > "$tmp/fit.out" 2> "$tmp/fit.err"; then
    cat "$tmp/fit.err" >> "$log_file"
    die "psmc-rs failed: $(tail -n 5 "$tmp/fit.err")"
  fi
  cat "$tmp/fit.err" "$tmp/fit.out" >> "$log_file"
  # The QC warnings were shown already; the fit's own warnings and the
  # psmc_plot.pl hint were not.
  grep -v '^Warning: QC ' "$tmp/fit.err" | grep -E '^(Warning: |To plot it)' >&2 || true
  cat "$tmp/fit.out" >&2
fi

say "done: $vcf, $qc_txt, $qc_json, $psmcfa$([ "$multihetsep" -eq 0 ] || printf ', %s' "$out.<chrom>.multihetsep")$([ -z "$run_iters" ] || printf ', %s' "$out.json, $out.html, $out.psmc")"
