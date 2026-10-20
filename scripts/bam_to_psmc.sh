#!/usr/bin/env bash
# bam_to_psmc.sh: from the aligned reads of one diploid genome to psmc-rs
# input through the classic PSMC consensus (route A of docs/preprocessing.md):
#
#   bcftools mpileup -C50 -q20 -Q20 -f ref.fa aln.bam | bcftools call -c
#
# per chromosome in parallel, then one bgzipped, indexed all-sites VCF,
# `psmc-rs qc` and `psmc-rs convert` (psmcfa, optionally multihetsep), and
# optionally an inference run. Run with --help for the options;
# tests/pipeline/run_e2e.sh tests it.
#
# Works with the bash 3.2 that macOS ships: no associative arrays, no
# `wait -n`, and empty arrays are never expanded under `set -u`.

set -euo pipefail

readonly SCRIPT_VERSION="1.0"
readonly MIN_BCFTOOLS="1.10"
readonly MIN_PSMC_RS="0.2.0"
readonly DEFAULT_MAX_DEPTH=10000
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
                      only its chromosome, and the read groups are checked for
                      a single sample before any call; without one, every job
                      streams the whole file, and the check waits until the
                      first chromosome is called.
  --ref FILE          The reference the reads were aligned to (FASTA, plain or
                      bgzip). Its .fai is created next to it if missing.
  --out PREFIX        Output prefix; the directory is created if needed.

Options:
  --chroms LIST|FILE  Chromosomes to use: a comma-separated list, or a file
                      whose first column gives the names: one per line, a
                      .fai, or a BED (only the names count: each chromosome
                      is called whole). Default: every contig of the reference
                      not named like a sex chromosome or organelle (X, Y, Z,
                      W, M, MT, chrX, chrM, ..., the rule of \`psmc-rs qc\`),
                      except that X is kept when the names are Roman
                      numerals past VIII (chrIX, chrXI: X is chromosome 10).
                      Unplaced, ALT, decoy, viral and HLA contigs are called
                      too, with a warning: with a reference such as a GRCh38
                      analysis set, list the primary autosomes
  --threads N         Chromosomes called at once, and bcftools concat
                      threads [the CPUs this process may use: $(default_threads)].
                      Passed to the fit only when given; otherwise psmc-rs
                      picks its own default, which also follows CPU quotas
  --min-mapq N        mpileup -q: minimum mapping quality of a read [20]
                      (vcf2fq's -Q, on the RMS mapping quality of a site,
                      stays at its default 10)
  --min-baseq N       mpileup -Q: minimum base quality [20]
  --depth auto|MIN:MAX
                      Depth bounds of a called base (vcf2fq -d, -D). auto: a
                      third and twice the mean depth (psmc README) [auto]
  --max-depth N       mpileup -d: the most reads used per position [10000,
                      or twice MAX of --depth MIN:MAX if that is more]. It
                      must be at least twice the upper depth bound: beyond
                      the cap mpileup stops counting, so deeper sites
                      (collapsed repeats) would pass the bound. mpileup's
                      own default, 250, is enough only up to a mean depth
                      of about 60x
  --min-qual N        fq2psmcfa -q: minimum consensus quality [20, as in the
                      psmc README]
  --bin-size N        fq2psmcfa -s: bases per bin [100]
  --multihetsep       Also write PREFIX.<chrom>.multihetsep (for MSMC2)
  --run N_ITER        Also fit PSMC with N_ITER EM iterations: PREFIX.json,
                      PREFIX.html and PREFIX.psmc. Options after \`--\` go to
                      this psmc-rs run (e.g. -- --pattern '4+25*2+4+6');
                      psmc-rs checks them before anything is called
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
leaves none of them half written; psmc-rs writes the fit's files itself.
Ctrl+C or SIGTERM stops whatever runs (the calls, concat, qc, convert or
the fit) at once and removes the working directory (exit 130 or 143). A
rerun with the same inputs writes the same bytes to every output but the
log, whatever --threads is. When a rerun makes a different psmcfa, the
earlier run's PREFIX.json, .html, .psmc and multihetsep files no longer
match it and are removed, with a warning unless this run writes them
again. A PREFIX.bootstrap directory is then only reported, and so are
working directories that a killed run left. bcftools >= $MIN_BCFTOOLS and
psmc-rs >= $MIN_PSMC_RS are required. See docs/preprocessing.md.
EOF
}

# The CPUs this process may use. nproc follows the affinity mask (taskset,
# cpusets, a batch scheduler's binding), where getconf counts every online
# CPU of the machine; macOS has no nproc, and no affinity masks either.
default_threads() {
  nproc 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 1
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
bam="" ref="" out="" chroms_arg="" threads="" threads_given=0 min_mapq=20 min_baseq=20
depth="auto" max_depth="" min_qual=20 bin_size=100 multihetsep=0 run_iters="" psmc_rs=""
keep_tmp=0 dry_run=0
extra=()

need_value() {
  [ -n "${2-}" ] || usage_error "$1 needs a value"
}

is_uint() {
  case "$1" in
    '' | *[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

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

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bam) need_value "$@"; bam=$2; shift 2 ;;
    --ref) need_value "$@"; ref=$2; shift 2 ;;
    --out) need_value "$@"; out=$2; shift 2 ;;
    --chroms) need_value "$@"; chroms_arg=$2; shift 2 ;;
    --threads) need_value "$@"; threads=$2; threads_given=1; shift 2 ;;
    --min-mapq) need_value "$@"; min_mapq=$2; shift 2 ;;
    --min-baseq) need_value "$@"; min_baseq=$2; shift 2 ;;
    --depth) need_value "$@"; depth=$2; shift 2 ;;
    --max-depth) need_value "$@"; max_depth=$2; shift 2 ;;
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
  "--min-qual:$min_qual" "--bin-size:$bin_size" "--max-depth:${max_depth:-1}"; do
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
# mpileup stops adding reads at a position once --max-depth are there, and
# the INFO/DP of deeper sites then no longer grows with their depth
# (bcftools 1.24 gave a median of 245 at -d 250 for 340 reads). The cap must
# lie well above the upper depth bound, or those sites pass it; twice the
# bound leaves room. With --depth auto the bound is known only once psmc-rs
# has read the VCF, and it is checked then.
if [ -n "$max_depth" ]; then
  [ "$max_depth" -ge 1 ] || usage_error "--max-depth must be at least 1"
  if [ "$depth" != auto ] && [ "$((2 * dmax))" -gt "$max_depth" ]; then
    usage_error "--max-depth $max_depth is less than twice the upper depth bound $dmax of --depth $depth: deeper sites would pass the bound; give --max-depth $((2 * dmax)) or more"
  fi
elif [ "$depth" != auto ] && [ "$((2 * dmax))" -gt "$DEFAULT_MAX_DEPTH" ]; then
  max_depth=$((2 * dmax))
else
  max_depth=$DEFAULT_MAX_DEPTH
fi
if [ -n "$run_iters" ]; then
  is_uint "$run_iters" || usage_error "--run must be a whole number of EM iterations, not '$run_iters'"
elif [ "${#extra[@]}" -gt 0 ]; then
  usage_error "options after -- go to the inference run, which needs --run N_ITER"
fi
! has_extra --bin-size ||
  usage_error "give the bin size as this script's --bin-size, which also sets the psmcfa bins"
! has_extra --tui || usage_error "--tui does not go with this script; run psmc-rs --tui on $out.psmcfa"
case "$out" in
  */) usage_error "--out is a prefix for file names, not a directory: '$out'" ;;
esac

readable_file() { [ -f "$1" ] && [ -r "$1" ]; }
readable_file "$bam" || die "cannot read --bam $bam"
readable_file "$ref" || die "cannot read --ref $ref"

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

# The version of the program $1, named $2, which must be at least $3: what
# `$1 --version` prints after "$2 " at the start of a line of its standard
# output, the first such line; $4 matches the version (a sed BRE), and $5
# (optional) says what needs the minimum. Standard error is not searched,
# as the dynamic loader or an LD_PRELOAD library may write a line there
# before the program runs (glibc's "weak version `GLIBC_2.39' not found"
# with a binary built on a newer glibc, for one); the error messages show it.
tool_version() { # program, name, minimum, version pattern[, what needs it]
  local out version
  out=$("$1" --version 2> /dev/null) || die "'$1 --version' failed: $("$1" --version 2>&1)"
  version=$(printf '%s\n' "$out" | sed -n "s/^$2 \\($4\\).*/\\1/p" | sed -n 1p)
  [ -n "$version" ] || die "cannot read the $2 version from '$("$1" --version 2>&1)'"
  version_at_least "$version" "$3" || die "$2 $version is too old: $3 or later is needed${5-}"
  printf '%s\n' "$version"
}

bcftools=${BCFTOOLS:-bcftools}
command -v "$bcftools" > /dev/null 2>&1 ||
  die "bcftools not found (install bcftools >= $MIN_BCFTOOLS, or set BCFTOOLS to its path)"
bcftools_version=$(tool_version "$bcftools" bcftools "$MIN_BCFTOOLS" \
  '[0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)*') || exit 1

psmc_rs=${psmc_rs:-${PSMC_RS:-psmc-rs}}
command -v "$psmc_rs" > /dev/null 2>&1 ||
  die "psmc-rs not found (build it with cargo build --release, then give --psmc-rs PATH or set PSMC_RS)"
psmc_rs_version=$(tool_version "$psmc_rs" psmc-rs "$MIN_PSMC_RS" \
  '[0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*' " (for qc and convert)") || exit 1

# ---------------------------------------------------------------------------
# Reference index and chromosomes

fai="$ref.fai"
if [ ! -s "$fai" ]; then
  [ "$dry_run" -eq 0 ] || die "$fai does not exist; a dry run does not create it"
  # htslib builds the .fai when a command first opens the FASTA; an empty
  # VCF through `bcftools norm -f` does that without reading any reads.
  log_cmd "bcftools norm -f $(quoted "$ref") (creates $(quoted "$fai"))"
  printf '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n' |
    "$bcftools" norm --no-version -f "$ref" -o /dev/null > /dev/null 2>&1 || true
  [ -s "$fai" ] ||
    die "cannot index $ref (gzip-compressed FASTA must be bgzip-compressed; or create $fai with samtools faidx)"
fi

# The rule of psmc-rs's qc::classify_name: prints "sex chromosome",
# "organelle" or nothing for each name on stdin. One exception: when the
# names number chromosomes in Roman numerals past VIII (chrIX or chrXI, as
# in stickleback or yeast assemblies), X is chromosome 10, not a sex
# chromosome, and it is printed as "roman" (C. elegans's I to V and X keep
# their sex chromosome).
classify_names() {
  LC_ALL=C awk '
    BEGIN {
      refseq["nc_000023"] = "sex chromosome"; refseq["nc_000024"] = "sex chromosome"
      refseq["nc_012920"] = "organelle"; refseq["nc_001807"] = "organelle"
      refseq["nc_000086"] = "sex chromosome"; refseq["nc_000087"] = "sex chromosome"
      refseq["nc_005089"] = "organelle"
    }
    # Sets tok to the lower-case name without its chr, chrom or chromosome
    # prefix, up to the first character other than a letter or digit, and
    # prefixed to whether there was such a prefix.
    function split_name(lower,    rest) {
      rest = lower; prefixed = 1
      if (substr(rest, 1, 10) == "chromosome") rest = substr(rest, 11)
      else if (substr(rest, 1, 5) == "chrom") rest = substr(rest, 6)
      else if (substr(rest, 1, 3) == "chr") rest = substr(rest, 4)
      else prefixed = 0
      if (prefixed && rest ~ /^[-_.]/) rest = substr(rest, 2)
      tok = rest
      if (match(rest, /[^a-z0-9]/)) tok = substr(rest, 1, RSTART - 1)
    }
    function kind(name,    lower, acc) {
      lower = tolower(name)
      if (lower ~ /mitochondri|chloroplast|plastid|plastome|apicoplast/) return "organelle"
      acc = lower; sub(/\..*/, "", acc)
      if (acc in refseq) return refseq[acc]
      split_name(lower)
      if (tok ~ /^[xyzw][0-9]*$/) return (tok == "x" && roman) ? "roman" : "sex chromosome"
      if (tok ~ /^(m|mt|mito|mtdna|pt|cp)$/) return "organelle"
      if (tok == "c" && prefixed) return "organelle"
      return ""
    }
    {
      names[NR] = $0
      split_name(tolower($0))
      if (tok == "ix" || tok == "xi") roman = 1
    }
    END { for (i = 1; i <= NR; i++) print kind(names[i]) }
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
roman_x=()
while IFS= read -r k; do
  if [ "$k" = roman ]; then
    roman_x+=("${ref_names[${#ref_kinds[@]}]}")
    k=""
  fi
  ref_kinds+=("$k")
done < <(printf '%s\n' "${ref_names[@]}" | classify_names)

# Names asked for with --chroms, one per line. From a file: the first
# column, without blank and comment lines or a BED file's track and browser
# lines, each name once (a BED file has a row per interval), and without
# the carriage returns of Windows line ends.
requested_names() {
  if [ -f "$chroms_arg" ]; then
    awk '{ sub(/\r$/, "") }
      /^[ \t]*(#|$)/ || /^(track|browser)([ \t]|$)/ { next }
      !seen[$1]++ { print $1 }' "$chroms_arg"
  else
    printf '%s\n' "$chroms_arg" | tr ',' '\n' | awk '{ sub(/\r$/, "") } NF { print $1 }'
  fi
}

# Indices into ref_names of the chromosomes to call, in reference order.
selected=()
excluded=()
if [ -z "$chroms_arg" ]; then
  if [ "${#roman_x[@]}" -gt 0 ]; then
    say "kept ${roman_x[*]}: the reference numbers its chromosomes in Roman numerals (it has IX or XI), so X is chromosome 10, not a sex chromosome; if it is one, list the chromosomes with --chroms"
  fi
  for i in "${!ref_names[@]}"; do
    if [ -n "${ref_kinds[$i]}" ]; then
      excluded+=("${ref_names[$i]} (${ref_kinds[$i]})")
    else
      selected+=("$i")
    fi
  done
  # Contigs that are not chromosomes of the primary assembly, in the naming
  # of GRCh37, GRCh38 and their analysis sets: unplaced and unlocalized
  # (chrUn_*, *_random, GL*/KI* accessions), ALT haplotypes and patches,
  # decoys, EBV and HLA alleles. All of them would be called.
  others=$(for i in ${selected[@]+"${selected[@]}"}; do printf '%s\n' "${ref_names[$i]}"; done |
    LC_ALL=C awk '{ l = tolower($0) }
      l ~ /_(alt|decoy|random|fix)$/ || l ~ /_hap[0-9]+$/ || l ~ /^(chr)?un[_.-]/ ||
      l ~ /^(chr)?ebv$/ || l ~ /^nc_007605/ || l ~ /^hs3[78]d/ || l ~ /^hla-/ ||
      l ~ /^(gl|ki)[0-9]+\.[0-9]+$/ {
        n++
        if (n <= 3) ex = ex (n > 1 ? ", " : "") $0
      }
      END { if (n) printf "%d (%s%s)", n, ex, (n > 3 ? ", ..." : "") }')
  if [ -n "$others" ]; then
    warn "the contigs to call include $others that look like unplaced, ALT, decoy, patch, viral or HLA contigs; with a reference such as a GRCh38 analysis set, list the primary autosomes with --chroms: ALT haplotypes repeat parts of the primary chromosomes, and every contig is one more calling job"
  fi
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

# The alignments are read by region when they have an index.
indexed=0
for idx in "$bam.bai" "$bam.csi" "$bam.crai" "${bam%.*}.bai" "${bam%.*}.crai"; do
  if [ -f "$idx" ]; then
    indexed=1
    break
  fi
done

# The calls of one chromosome: mpileup, with its regions file and the
# alignments after these options, piped to call, with the BCF file to write
# after its own. A regions file, unlike -r, keeps a name with colons whole.
where=-T
[ "$indexed" -eq 0 ] || where=-R
mpileup=("$bcftools" mpileup --no-version -C50 -q "$min_mapq" -Q "$min_baseq" -d "$max_depth"
  -f "$ref" "$where")
call=("$bcftools" call --no-version -c -Ob -o)

# The calls as one command line, for the log and the dry run.
mpileup_cmd() { # $1: regions file
  printf '%s' "$(quoted "${mpileup[@]}" "$1" "$bam") | $(quoted "${call[@]}")"
}

common_input_opts=(--input-format vcf --vcf-mode consensus --depth "$depth"
  --min-qual "$min_qual" --bin-size "$bin_size")

# Sets conv to the psmc-rs convert command writing the psmcfa $1 and, with
# --multihetsep, the multihetsep files of prefix $2.
convert_cmd() {
  conv=("$psmc_rs" convert "$vcf" "${common_input_opts[@]}" -q --psmcfa "$1")
  [ "$multihetsep" -eq 0 ] || conv+=(--multihetsep-per-chrom "$2")
}

# The inference run, with the options after --.
fit=()
if [ -n "$run_iters" ]; then
  fit=("$psmc_rs" "$psmcfa" "$out.json" "$run_iters" --bin-size "$bin_size")
  if [ "$threads_given" -eq 1 ] && ! has_extra --threads; then
    fit+=(--threads "$threads")
  fi
  has_extra --psmc-out || fit+=(--psmc-out "$out.psmc")
  has_extra --no-progress || fit+=(--no-progress)
  fit+=(${extra[@]+"${extra[@]}"})
fi

if [ "$dry_run" -eq 1 ]; then
  echo "Chromosomes to call (${#selected[@]}):"
  for i in "${selected[@]}"; do
    echo "  ${ref_names[$i]}"
  done
  if [ "${#excluded[@]}" -gt 0 ]; then
    echo "Left out by name (${#excluded[@]}):"
    printf '  %s\n' "${excluded[@]}"
  fi
  echo "Commands:"
  echo "  $(mpileup_cmd "TMP/regions/N.txt") TMP/calls/N.bcf   # per chromosome, $threads at a time"
  echo "  $(quoted "$bcftools" concat --no-version --threads "$threads" -Oz -o "$vcf" -f TMP/calls.txt)"
  echo "  $(quoted "$bcftools" index -t "$vcf")"
  echo "  $(quoted "$psmc_rs" qc "$vcf" "${common_input_opts[@]}" --json "$qc_json") > $(quoted "$qc_txt")"
  convert_cmd "$psmcfa" "$out"
  echo "  $(quoted "${conv[@]}")"
  [ -z "$run_iters" ] || echo "  $(quoted "${fit[@]}")"
  exit 0
fi

mkdir -p "$out_dir" || die "cannot create the output directory $out_dir"
: > "$out.log" || die "cannot write $out.log"
log_file=$out.log

# Working directories of earlier runs with this prefix that were killed
# before they could remove them (SIGKILL, a lost node). Another run with
# this prefix may be going, so they are only reported.
leftover_dirs=""
for d in "$out_dir/.$base.tmp."*; do
  if [ -d "$d" ]; then
    leftover_dirs="$leftover_dirs $d"
  fi
done
[ -z "$leftover_dirs" ] ||
  warn "working directories of earlier runs that did not finish:$leftover_dirs; remove them unless another run with this prefix is going"

tmp=$(mktemp -d "$out_dir/.$base.tmp.XXXXXX") || die "cannot create a working directory in $out_dir"
# The calling jobs running, and the step that run_step waits for.
pids=()
child=""

# On exit, and so on SIGINT and SIGTERM: stop what still runs, then remove
# the working directory. Background jobs ignore SIGINT, so a Ctrl+C reaches
# them only through this TERM; each calling job then stops its own mpileup
# and call.
cleanup() {
  local status=$?
  local p
  for p in ${pids[@]+"${pids[@]}"} ${child:+"$child"}; do
    kill -TERM "$p" 2> /dev/null || true
  done
  for p in ${pids[@]+"${pids[@]}"} ${child:+"$child"}; do
    wait "$p" 2> /dev/null || true
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

# Run a command in the background and wait for it. bash runs a trap only
# once the foreground command has ended, so a long step run in the
# foreground would hold off SIGTERM until it finished; a wait is cut short
# at once, and cleanup stops the step.
run_step() {
  local s=0
  "$@" &
  child=$!
  wait "$child" || s=$?
  child=""
  return "$s"
}

# Log a command, then run it with run_step, its messages going to the log.
run_logged() {
  log_cmd "$(quoted "$@")"
  run_step "$@" 2>> "$log_file"
}

# Run a command with run_step, its output in the file $1 and its messages
# in $2, both then appended to the log (only the messages if it failed).
run_captured() { # output file, messages file, command...
  local stdout=$1 stderr=$2 s=0
  shift 2
  run_step "$@" > "$stdout" 2> "$stderr" || s=$?
  cat "$stderr" >> "$log_file"
  [ "$s" -ne 0 ] || cat "$stdout" >> "$log_file"
  return "$s"
}

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

# What psmc-rs's stderr in $1 says went wrong: its error line and clap's
# tips, or else its last lines.
error_lines() {
  local lines
  lines=$(grep -E '^(error|Error):|^ *tip: ' "$1" || true)
  [ -n "$lines" ] || lines=$(tail -n 5 "$1")
  printf '%s\n' "$lines"
}

# Whether the calls in $1 hold exactly one sample.
check_samples() {
  local samples count
  samples=$("$bcftools" query -l "$1" 2>> "$log_file") || die "cannot read the samples of $1"
  count=$(printf '%s\n' "$samples" | awk 'NF { n++ } END { print n + 0 }')
  [ "$count" -eq 1 ] ||
    die "the alignments hold $count samples (read-group SM tags: $(printf '%s\n' "$samples" | tr '\n' ' ' | sed 's/ $//')); PSMC needs the reads of one diploid genome, so give one sample's reads or fix the SM tags"
  say "sample: $samples"
}

# ---------------------------------------------------------------------------
# 0. What would otherwise fail only after the calls

# The fit's options. psmc-rs checks its options and output paths before it
# reads its input, so on an input that does not exist it stops at the
# first problem; if that is the missing input, the rest is fine.
if [ -n "$run_iters" ]; then
  check_fit=("$psmc_rs" "$tmp/none.psmcfa" "${fit[@]:2}")
  log_cmd "$(quoted "${check_fit[@]}")   # checks the fit's options"
  if "${check_fit[@]}" > "$tmp/check.out" 2> "$tmp/check.err"; then
    die "with the options after --, psmc-rs prints its help or version instead of fitting"
  fi
  if ! grep -qF -- "$tmp/none.psmcfa" "$tmp/check.err"; then
    cat "$tmp/check.err" >> "$log_file"
    die "psmc-rs rejects the fit's options, so nothing was run:
$(error_lines "$tmp/check.err")"
  fi
fi

# The samples, from the header that mpileup writes for one base, when the
# alignments have an index; without one this would read the whole file, and
# the samples are checked once the first chromosome is called.
samples_checked=0
if [ "$indexed" -eq 1 ]; then
  printf '%s\t1\t1\n' "${ref_names[${selected[0]}]}" > "$tmp/samples.txt"
  run_logged "$bcftools" mpileup --no-version -f "$ref" -R "$tmp/samples.txt" -Ou \
    -o "$tmp/samples.bcf" "$bam" || die "bcftools mpileup cannot read $bam (see $log_file)"
  check_samples "$tmp/samples.bcf"
  samples_checked=1
fi

# ---------------------------------------------------------------------------
# 1. Per-chromosome calls

mkdir -p "$tmp/regions" "$tmp/calls"

# One calling job, run in the background. mpileup and call are joined by a
# FIFO rather than a pipe so that the job knows both process IDs: when
# cleanup stops the job, its trap stops them too, with no need for pkill.
call_one() { # $1: number of the job, $2: index into ref_names
  local n=$1 i=$2 mp="" cl="" s=0
  trap 'kill -TERM ${mp:+"$mp"} ${cl:+"$cl"} 2> /dev/null; exit 143' TERM
  printf '%s\t1\t%s\n' "${ref_names[$i]}" "${ref_lengths[$i]}" > "$tmp/regions/$n.txt"
  mkfifo "$tmp/calls/$n.fifo"
  "${mpileup[@]}" "$tmp/regions/$n.txt" "$bam" 2>> "$tmp/calls/$n.log" > "$tmp/calls/$n.fifo" &
  mp=$!
  "${call[@]}" "$tmp/calls/$n.bcf" < "$tmp/calls/$n.fifo" 2>> "$tmp/calls/$n.log" &
  cl=$!
  wait "$mp" || s=$?
  wait "$cl" || s=$?
  rm -f "$tmp/calls/$n.fifo"
  return "$s"
}

# Wait for the oldest job, number $called + 1 (the jobs are waited for in
# the order they started); stop everything if it failed.
called=0
wait_oldest() {
  local n=$((called + 1)) name=${ref_names[${selected[$called]}]} s=0
  wait "${pids[0]}" || s=$?
  pids=("${pids[@]:1}")
  if [ "$s" -ne 0 ]; then
    cat "$tmp/calls/$n.log" >> "$log_file" 2> /dev/null || true
    die "calling $name failed: $(tail -n 3 "$tmp/calls/$n.log" 2> /dev/null | tr '\n' ' ')(full messages in $log_file)"
  fi
  cat "$tmp/calls/$n.log" >> "$log_file"
  called=$n
  if [ "$samples_checked" -eq 0 ]; then
    check_samples "$tmp/calls/$n.bcf"
    samples_checked=1
  fi
  say "called $name"
}

log_cmd "$(mpileup_cmd "$tmp/regions/N.txt") $tmp/calls/N.bcf   # N = 1..${#selected[@]}"
n=0
for i in "${selected[@]}"; do
  n=$((n + 1))
  while [ "${#pids[@]}" -ge "$threads" ]; do
    wait_oldest
  done
  call_one "$n" "$i" &
  pids+=("$!")
done
while [ "${#pids[@]}" -gt 0 ]; do
  wait_oldest
done

# ---------------------------------------------------------------------------
# 2. One all-sites VCF, bgzipped and indexed

for ((n = 1; n <= ${#selected[@]}; n++)); do
  printf '%s\n' "$tmp/calls/$n.bcf"
done > "$tmp/calls.txt"
run_logged "$bcftools" concat --no-version --threads "$threads" -Oz -o "$tmp/all.vcf.gz" \
  -f "$tmp/calls.txt" || die "bcftools concat failed (see $log_file)"
run_logged "$bcftools" index -t "$tmp/all.vcf.gz" || die "bcftools index failed (see $log_file)"
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
run_captured "$tmp/qc.txt" "$tmp/qc.err" "${qc[@]}" || die "psmc-rs qc failed: $(cat "$tmp/qc.err")"
if [ "$depth" = auto ]; then
  bound=$(sed -n 's/.*bases kept at depth [0-9]*-\([0-9]*\) (auto).*/\1/p' "$tmp/qc.err" | head -n 1)
  if [ -z "$bound" ]; then
    warn "cannot read the depth bounds from psmc-rs qc; check in $log_file that twice the upper one is at most --max-depth $max_depth"
  elif [ "$((2 * bound))" -gt "$max_depth" ]; then
    # The VCF's depths are capped: it would mislead whoever reuses it.
    rm -f "$vcf" "$vcf.tbi"
    die "the upper depth bound is $bound (twice the mean depth), so mpileup must read at least $((2 * bound)) reads per position, more than --max-depth $max_depth: deeper sites, such as collapsed repeats, would pass the bound. Rerun with --max-depth $((2 * bound)) or more"
  fi
fi
# The report goes to its final name below, not to the working directory
# that psmc-rs names.
grep -v '^QC report: ' "$tmp/qc.err" >&2 || true
cat "$tmp/qc.txt" >&2
mv -f "$tmp/qc.txt" "$qc_txt"
mv -f "$tmp/qc.json" "$qc_json"

# ---------------------------------------------------------------------------
# 4. psmcfa (and multihetsep)

mkdir -p "$tmp/convert"
convert_cmd "$tmp/convert/out.psmcfa" "$tmp/convert/$base"
log_cmd "$(quoted "${conv[@]}")"
# convert reads the input as qc did and repeats its warnings: they go to the
# log only.
run_captured "$tmp/convert.out" "$tmp/convert.err" "${conv[@]}" ||
  die "psmc-rs convert failed: $(cat "$tmp/convert.err")"
# When the psmcfa differs from the one an earlier run left here, that run's
# fit and multihetsep files no longer match it, and they are removed: in
# silence when this run writes them again (so that a failure leaves none
# behind), with a warning otherwise. A bootstrap directory is reported.
changed=1
if [ -f "$psmcfa" ] && cmp -s "$tmp/convert/out.psmcfa" "$psmcfa"; then
  changed=0
fi
mv -f "$tmp/convert/out.psmcfa" "$psmcfa"
stale=()
if [ "$changed" -eq 1 ]; then
  for f in "$out.json" "$out.html" "$out.psmc"; do
    [ -e "$f" ] || continue
    rm -f "$f"
    if [ -z "$run_iters" ] || { [ "$f" = "$out.psmc" ] && has_extra --psmc-out; }; then
      stale+=("$f")
    fi
  done
fi
if [ "$multihetsep" -eq 1 ] || [ "$changed" -eq 1 ]; then
  # Every name this script could have written: a chromosome no longer
  # called leaves a file too.
  n=0
  for name in "${ref_names[@]}"; do
    if [ -e "$out.$name.multihetsep" ]; then
      rm -f "$out.$name.multihetsep"
      n=$((n + 1))
    fi
  done
  [ "$multihetsep" -eq 1 ] || [ "$n" -eq 0 ] || stale+=("$n file(s) $out.<chrom>.multihetsep")
fi
if [ "${#stale[@]}" -gt 0 ]; then
  warn "removed the outputs of an earlier run, made from another $psmcfa: $(printf '%s, ' "${stale[@]}" | sed 's/, $//')"
fi
if [ "$changed" -eq 1 ] && [ -d "$out.bootstrap" ]; then
  warn "$out.bootstrap holds the bootstrap of an earlier run, made from another $psmcfa"
fi
if [ "$multihetsep" -eq 1 ]; then
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
  log_cmd "$(quoted "${fit[@]}")"
  run_captured "$tmp/fit.out" "$tmp/fit.err" "${fit[@]}" ||
    die "psmc-rs failed (full messages in $log_file):
$(error_lines "$tmp/fit.err")"
  # The QC warnings were shown already; the fit's own warnings and the
  # psmc_plot.pl hint were not.
  grep -v '^Warning: QC ' "$tmp/fit.err" | grep -E '^(Warning: |To plot it)' >&2 || true
  cat "$tmp/fit.out" >&2
fi

written="$vcf, $qc_txt, $qc_json, $psmcfa"
[ "$multihetsep" -eq 0 ] || written="$written, $out.<chrom>.multihetsep"
[ -z "$run_iters" ] || written="$written, $out.json, $out.html, $out.psmc"
say "done: $written"
