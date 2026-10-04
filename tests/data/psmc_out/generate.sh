#!/bin/sh
# Regenerates the files in this directory (see README.md).
#
#   PSMC=/path/to/psmc ./generate.sh
#
# PSMC is C psmc 0.6.5-r74 (lh3/psmc). Requires cc.
set -eu
cd "$(dirname "$0")"
: "${PSMC:?set PSMC to the C psmc binary}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# How C's printf formats doubles (src/psmc_out.rs unit test).
cc -O1 -o "$tmp/printf_probe" printf_probe.c
"$tmp/printf_probe" > printf_golden.tsv

# C psmc with one free lambda per state, the pattern psmc-rs writes for a
# model without a pattern.
"$PSMC" -N5 -p "23*1" -o c_23x1_N5.psmc ../c_reference/papuan_chr1_3seg.psmcfa
