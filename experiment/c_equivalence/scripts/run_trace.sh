#!/bin/bash
# The Hooke-Jeeves port against C psmc's own search: the first M-step on papuan_chr1_3seg.psmcfa with -p 4+5*3+4
# (p23), -p 4+25*2+4+6 (p64) and 23 free lambdas (f23), every evaluation of the objective traced by C psmc built with
# c_trace.patch and by the diagnostic build with both changes (the last bin left out, as C does, and the port),
# then compared by compare_trace.py into trace_<case>.txt. The traces themselves (megabytes) go to WORKDIR.
# Usage: scripts/run_trace.sh DIAG_BINARY WORKDIR
#   C psmc with the trace, built as tests/data/c_reference was built:
#   cp -R experiment/.tools/psmc_c WORKDIR/psmc_c && (cd WORKDIR/psmc_c && patch -p1 < <this directory>/c_trace.patch &&
#     cc -O1 cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz -o psmc_trace)
set -euo pipefail
C=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$C" rev-parse --show-toplevel)
X=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
T=$(cd "$2" && pwd)
REF=$REPO/tests/data/c_reference/papuan_chr1_3seg.psmcfa
for case in p23 p64 f23; do
  case $case in
    p23) c=(-p '4+5*3+4'); rs=(--pattern '4+5*3+4') ;;
    p64) c=(-p '4+25*2+4+6'); rs=(--pattern '4+25*2+4+6') ;;
    f23) c=(-p '23*1'); rs=(--n-steps 22) ;;
  esac
  PSMC_TRACE=1 "$T/psmc_c/psmc_trace" -N1 "${c[@]}" -o "$T/c_tr_$case.psmc" "$REF" 2> "$T/c_tr_$case.txt"
  PSMC_DIAG_LASTBIN=1 PSMC_DIAG_HJ=1 PSMC_DIAG_HJ_TRACE=1 "$X" "$REF" "$T/rs_tr_$case.json" 1 "${rs[@]}" \
    --psmc-out "$T/rs_tr_$case.psmc" -q --no-progress 2> "$T/rs_tr_$case.txt" > /dev/null
  python3 -I -B "$C/scripts/compare_trace.py" "$T/c_tr_$case.txt" "$T/rs_tr_$case.txt" > "$C/trace_$case.txt"
done
