# C psmc reference values for the QC metrics

Used by `tests/qc_c_reference.rs`. `c_psmc_round0.tsv` records what Heng Li's
C psmc ([lh3/psmc](https://github.com/lh3/psmc) 0.6.5-r74, MIT license, see
`../c_reference/PSMC_LICENSE.txt`) reports about its input before any EM
iteration, for each input file and for single records:

| column     | C source                                                        |
|------------|-----------------------------------------------------------------|
| `n_seqs`   | `MM n_seqs:` line (records, empty ones included)                |
| `sum_L`    | `MM ... sum_L:` (non-missing bins, `L_e` summed)                |
| `sum_n`    | `MM ... sum_n:` (heterozygous bins, `n_e` summed)               |
| `theta0`   | first parameter of the round-0 `PA` line, `-ln(1 - sum_n/sum_L)`, `%.9lf` |
| `n_recomb` | round-0 `MM C_pi: .., n_recomb: ..` value, `%lf`                 |

`record` is `*` for a whole file, otherwise the name of the one record that
was written to its own file and run alone.

Inputs: `synthetic.psmcfa` (written by `make_synthetic.py`, deterministic: six
records with different lengths, heterozygosity, `N` runs, lowercase symbols
and header letters), `../c_reference/papuan_chr1_3seg.psmcfa`, and the four
files in `experiment/real_data/` (whole files; per record also for
`HLhydTec1_diploid.psmcfa`).

## Regenerating

```sh
PSMC=/path/to/psmc tests/data/qc_c_reference/generate.sh
```

`generate.sh` rewrites `synthetic.psmcfa` and `c_psmc_round0.tsv`; it runs
`psmc -N 0 -o out.psmc <input>` for every row. The committed table was made
with `psmc` 0.6.5-r74 built with
`cc -O1 cli.c core.c em.c khmm.c kmin.c main.c aux.c -lm -lz` (Apple clang 21,
macOS arm64; an `-O2` build segfaults on this platform), Python 3.14 and the
BSD `awk` of macOS.
