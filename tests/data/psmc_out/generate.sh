#!/bin/sh
# Regenerates the files in this directory (see README.md).
#
#   PSMC=/path/to/psmc PSMC_PLOT=/path/to/psmc_plot.pl ./generate.sh
#
# PSMC is C psmc 0.6.5-r74 (lh3/psmc) and PSMC_PLOT its utils/psmc_plot.pl
# (version 0.2.0). Requires cc, perl, python3 and cargo; gnuplot is not
# needed.
set -eu
cd "$(dirname "$0")"
: "${PSMC:?set PSMC to the C psmc binary}"
: "${PSMC_PLOT:?set PSMC_PLOT to psmc_plot.pl}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# How C's printf formats doubles (src/psmc_out.rs unit test).
cc -O1 -o "$tmp/printf_probe" printf_probe.c
"$tmp/printf_probe" > printf_golden.tsv

# C psmc with one free lambda per state, the pattern psmc-rs writes for a
# model without a pattern.
"$PSMC" -N5 -p "23*1" -o c_23x1_N5.psmc ../c_reference/papuan_chr1_3seg.psmcfa

# One round of C psmc on many short sequences.
python3 fragment.py
"$PSMC" -N1 -p "4+5*3+4" -o c_4+5x3+4_fragmented_N1.psmc fragmented.psmcfa

# psmc-rs output on the same data (rs_*.psmc).
cargo test --test psmc_out_c_reference -- --ignored regenerate_fixtures

# psmc_plot.pl on C's and psmc-rs's output; keep the data files it plots.
rm -f plot/*.txt plot/*.par
grep -v '^#' plot/cases.tsv | while IFS="$(printf '\t')" read -r name opts inputs; do
	files=""
	for f in $inputs; do files="$files ../$f"; done
	# shellcheck disable=SC2086 # options and file lists are word lists
	perl "$PSMC_PLOT" -R $opts "plot/$name" $files 2> "$tmp/stderr" || {
		cat "$tmp/stderr" >&2
		exit 1
	}
	rm -f "plot/$name.gp" "plot/$name.eps"
done
