# C reference outputs

Reference outputs from Heng Li's original C implementation
([lh3/psmc](https://github.com/lh3/psmc), version 0.6.5-r74, MIT license, see
`PSMC_LICENSE.txt`), used by `tests/c_equivalence.rs`.

- `papuan_chr1_3seg.psmcfa`: three segments of the Papuan chr1 psmcfa in
  `experiment/real_data/` (bins 400000-460000, 1200000-1250000, 2000000-2070000),
  180,000 bins total.
- `c_4+5x3+4_N5.psmc`: `psmc -N5 -p "4+5*3+4"` (C defaults otherwise: `-t 15 -r 4`).
- `c_4+25x2+4+6_N5.psmc`: `psmc -N5 -p "4+25*2+4+6"`.

Built with `cc -O1 cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz` on
macOS arm64 (Apple clang 21). Note: an `-O2` build of lh3/psmc segfaults after the
first round on this platform; `-O0` and `-O1` builds produce identical output.
