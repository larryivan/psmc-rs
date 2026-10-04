# `.psmc` output references

Reference files for psmc-rs's C-compatible `.psmc` output (`src/psmc_out.rs`),
read by the unit test in `src/psmc_out.rs`, `tests/psmc_out_c_reference.rs`,
`tests/psmc_out_plot.rs` and `tests/em_c_reference.rs`. `generate.sh`
regenerates them:

    PSMC=/path/to/psmc PSMC_PLOT=/path/to/psmc_plot.pl ./generate.sh

Tools used for the committed files:

- C psmc 0.6.5-r74 ([lh3/psmc](https://github.com/lh3/psmc), MIT license, see
  `../c_reference/PSMC_LICENSE.txt`), built as described in
  `../c_reference/README.md` (`cc -O1 cli.c core.c em.c khmm.c kmin.c main.c
  aux.c -lm -lz`, Apple clang 21, macOS arm64). Its version line reads
  `0.6.5-r74-dirty`, as in the other reference files.
- `utils/psmc_plot.pl` of the same lh3/psmc checkout (its version 0.2.0),
  run with perl 5.34.1. gnuplot is not needed: psmc_plot.pl writes the data
  files it plots before it starts gnuplot, and `-R` keeps them.
- Apple clang 21 and the macOS libc for `printf_probe.c`.
- Python 3.14.6 for `fragment.py` (standard library only; its output does
  not depend on the Python version).

## Files

- `printf_probe.c`, `printf_golden.tsv`: how C's `printf` formats 2,054
  doubles with `%lf`, `%.9lf`, `%.10lf` and `%g`, the conversions of C psmc's
  `.psmc` output. Each line holds the IEEE-754 bits in hex, then the four
  conversions. The values are special cases (signed zeros, infinities, NaNs,
  exact ties, `%g` style boundaries, typical `.psmc` values) and 2,000
  pseudo-random doubles between 2^-70 and 2^70.
- `c_23x1_N5.psmc`: `psmc -N5 -p "23*1" -o c_23x1_N5.psmc
  ../c_reference/papuan_chr1_3seg.psmcfa`, C psmc with one free lambda per
  state. psmc-rs writes this pattern for a model without `--pattern`; the C
  run shows that C psmc reads it and gives the output the tests compare with.
- `fragment.py`, `fragmented.psmcfa`: the first 20,000 bins of `seg1` and of
  `seg2` of `../c_reference/papuan_chr1_3seg.psmcfa`, cut by `fragment.py`
  into 1,010 records of 1 to 80 bins (lengths from a fixed linear
  congruential generator), as in a highly fragmented assembly. 10 records
  have one bin, 40 only missing bins, 117 end in a missing bin; 35,465 bins
  are not missing. C's expectations leave out the last bin of each
  sequence, which here is 2.5% of the data (see `posterior_sigma` in
  `src/em.rs`).
- `c_4+5x3+4_fragmented_N1.psmc`: `psmc -N1 -p "4+5*3+4" -o
  c_4+5x3+4_fragmented_N1.psmc fragmented.psmcfa`, one round of C psmc on
  the short records.
- `rs_4+5x3+4_N5.psmc`, `rs_23x1_N5.psmc`: psmc-rs's output for the runs of
  `../c_reference/c_4+5x3+4_N5.psmc` and `c_23x1_N5.psmc` (C's starting
  point, 5 rounds, no smoothing penalty), written by
  `cargo test --test psmc_out_c_reference -- --ignored regenerate_fixtures`.
  They are inputs for psmc_plot.pl. The tests read them only through
  psmc_plot.pl's rules, so they need regenerating (with `plot/`) only when
  the output format changes, not when the fit does.
- `plot/`: psmc_plot.pl's data files. For each line `name`, `options`,
  `inputs` of `plot/cases.tsv`, `generate.sh` runs

      perl psmc_plot.pl -R <options> plot/<name> <inputs, relative to tests/data>

  and keeps `plot/<name>.<i>.txt`, the curve of data set `i` (one per input
  run): years, N_e / 10^4, then the 4th to 6th `RS` columns, one line per
  plotted state; and `plot/<name>.par`, the `PA` line of the plotted round of
  data set 0. It deletes the gnuplot script `plot/<name>.gp`.
