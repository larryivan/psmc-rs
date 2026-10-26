#!/bin/bash
# Every variant of the diagnostic build on the inputs of C psmc's reference outputs (tests/data), with the options
# those outputs were made with: papuan_chr1_3seg.psmcfa for 5 iterations with -p 4+5*3+4 (p23), -p 4+25*2+4+6 (p64)
# and 23 free lambdas (f23: C's -p 23*1, psmc-rs's --n-steps 22), and the 1,010-record fragmented.psmcfa for one
# iteration with -p 4+5*3+4 (frag). Writes ref/<variant>_<case>.json and .psmc.
# Usage: scripts/run_ref.sh DIAG_BINARY
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$C" rev-parse --show-toplevel)
X=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
REF=$REPO/tests/data/c_reference/papuan_chr1_3seg.psmcfa
FRAG=$REPO/tests/data/psmc_out/fragmented.psmcfa
mkdir -p "$C/ref"; cd "$C/ref"
run() { # name variant args...
  local name=$1 v=$2; shift 2
  local envs=()
  case $v in
    none) ;;
    lastbin) envs=(PSMC_DIAG_LASTBIN=1) ;;
    hj) envs=(PSMC_DIAG_HJ=1) ;;
    both) envs=(PSMC_DIAG_LASTBIN=1 PSMC_DIAG_HJ=1) ;;
    prec) envs=(PSMC_DIAG_REL_TOL=0); set -- "$@" --mstep-iters 100000 ;;
  esac
  env -u PSMC_DIAG_LASTBIN -u PSMC_DIAG_HJ -u PSMC_DIAG_REL_TOL ${envs[@]+"${envs[@]}"} \
    "$X" "$@" --psmc-out "$name.psmc" -q --no-progress > /dev/null
  rm -f "$name.html"
}
for v in none lastbin hj both prec; do
  run ${v}_p23 $v "$REF" ${v}_p23.json 5 --pattern '4+5*3+4'
  run ${v}_p64 $v "$REF" ${v}_p64.json 5 --pattern '4+25*2+4+6'
  run ${v}_f23 $v "$REF" ${v}_f23.json 5 --n-steps 22
  run ${v}_frag $v "$FRAG" ${v}_frag.json 1 --pattern '4+5*3+4'
done
