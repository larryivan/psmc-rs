# `.psmc` output references

Reference files for psmc-rs's C-compatible `.psmc` output (`src/psmc_out.rs`),
read by the unit test in `src/psmc_out.rs` and by
`tests/psmc_out_c_reference.rs`. `generate.sh` regenerates them:

    PSMC=/path/to/psmc ./generate.sh

Tools used for the committed files:

- C psmc 0.6.5-r74 ([lh3/psmc](https://github.com/lh3/psmc), MIT license, see
  `../c_reference/PSMC_LICENSE.txt`), built as described in
  `../c_reference/README.md` (`cc -O1 cli.c core.c em.c khmm.c kmin.c main.c
  aux.c -lm -lz`, Apple clang 21, macOS arm64). Its version line reads
  `0.6.5-r74-dirty`, as in the other reference files.
- Apple clang 21 and the macOS libc for `printf_probe.c`.

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
