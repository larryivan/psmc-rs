# Checking an input: `psmc-rs qc`

```bash
psmc-rs qc INPUT [--json FILE] [--bootstrap-block-size BINS] [--strict]
```

`psmc-rs qc` reads an input as a fit would and reports on it without
fitting. Every fit and conversion prints the same report (`-q` reduces it
to its warnings), and a fit's JSON keeps the codes of its findings.

## The report

- **Metrics:** the bin counts, θ0 (C psmc's starting value), the
  heterozygosity per kb, the called length, the callable bases of the
  called bins (VCF and multihetsep: their mean share and distribution), the
  sequence lengths (N50), the longest missing run, the share of bins likely
  to hold several heterozygous sites, and the read depth (VCF).
- **A per-sequence table.** With callable bases, sequences are compared per
  callable base (the table's `het/kb*`, with a note under the table, and the
  sex-chromosome and outlier checks), so that a sequence with more missing
  data does not look less heterozygous. The totals' θ0 and heterozygosity
  stay per kb of called bins, as C psmc counts them, and the fit's JSON keeps
  both (`qc.het_per_kb`, `qc.het_per_callable_kb`).
- **Findings**, each a warning or a note with a code and what to do. The text
  report labels them `WARNING` and `INFO`, the QC JSON gives their
  `severity` as `warning` and `info`, and a fit's JSON lists their codes
  under `qc.warnings` and `qc.notes`.

Options:

- `--json FILE` (or `-` for stdout, the text then going to stderr) writes the
  report as JSON;
- `--bootstrap-block-size BINS` adds the checks of a bootstrap with trunks of
  BINS bins;
- `--strict` exits with status 2 when a finding is a warning.

## Findings

| code | severity | when | what to do |
| --- | --- | --- | --- |
| `INVALID_START` | warning | the fit's start (for `qc`, the default start) has negative transition probabilities, from which EM cannot move: ρ0 per bin is too large for the time grid | start from a smaller ρ0: the finding names the bin size, `--tr-ratio` or `--rho0` that fixes it, or for `--init-params` the rho to set in the saved file ([fitting.md](fitting.md#the-starting-point)); a fit stops |
| `VARIANTS_ONLY_NO_MASK` | warning | a VCF read in mask mode without a callable mask, so every position without a record counts as a called homozygous site | give a BED of callable sites (`--vcf-mask`), or call all sites with `bcftools call -c` ([preprocessing.md](preprocessing.md#route-b-variant-calls-and-a-callable-mask)) |
| `PARTLY_CALLABLE_BINS` | note; warning with `--bin-callable ignore` below a 95% mean | at least 1% of the called bins of a VCF hold fewer than `--bin-size` callable bases | nothing with the default `--bin-callable auto`, which weighs each bin by its callable bases; fit the VCF, not a psmcfa made from it. For psmcfa input, which cannot tell, the note counts the called bins next to missing runs and says to run `qc` on the VCF the psmcfa came from. Not given for multihetsep (see `APPROX_MISSING_PLACEMENT`) |
| `HIGH_MISSING` | warning above 50%, note above 25% | missing bins, or uncalled sites when they are counted | check what removes the data: depth and mapping-quality filters, the callable mask. Nadachowska-Brzyska et al. (2016) recommend at most 25% missing data |
| `BIN_SATURATION` | warning | more than 10% of the heterozygous bins likely hold two or more sites | re-bin at the largest bin size the finding names (below 5%): `--bin-size` for VCF and multihetsep, `fq2psmcfa -s` and then `--bin-size` for psmcfa |
| `FEW_HETS` | warning | fewer than 1,000 heterozygous bins | add data (more of the genome, a less strict mask) or fit a `--pattern` with fewer, wider groups; read both ends of the curve with caution |
| `TOO_SHORT` | warning | less than 100 Mb of called sequence | use more of the genome (all autosomes, a less strict mask); otherwise check the curve with the bootstrap and read its ends with caution |
| `SHORT_SEQUENCES` | warning | more than 20% of the bins in sequences shorter than 1 Mb or than the bootstrap trunk | short sequences: prefer a more contiguous assembly, or drop the sequences shorter than 1 Mb if enough data remain; for the bootstrap, a smaller `--bootstrap-block-size`, as large as the finding names |
| `SEX_OR_ORGANELLE` | note; warning when one looks hemizygous | sequences named like sex chromosomes or organelles | keep the autosomes only, for example by restricting the mask, the VCF or the psmcfa to them ([preprocessing.md](preprocessing.md#sex-chromosomes-and-organelles)) |
| `HET_OUTLIER` | note | sequences whose heterozygosity is far from the others' median | check them for contamination, collapsed duplications or artefacts (too high), or for missing heterozygotes or hemizygosity (too low) |
| `APPROX_MISSING_PLACEMENT` | note; warning when more than 5% of the sites are uncalled | multihetsep input, whose reader places the uncalled sites itself, whatever `--bin-callable` | for exact bins and callable bases, read the VCF with its callable mask ([inputs.md](inputs.md#multihetsep)) |
| `DEPTH` | warning | consensus mode: a mean depth below 18x, or one filter removing more than 30% of the bases | sequence deeper if possible, or read the curve as a lower bound on heterozygosity; check the filters' thresholds against the data ([preprocessing.md](preprocessing.md#low-coverage)) |
| `DROPPED_SEQUENCES` | note | the input's filters left out whole sequences | check that they were meant to be left out (for example, chromosome names in the mask or the VCF that do not match) |

The thresholds and their sources are documented on `psmc_rs::qc::QcCode`
(`cargo doc --open`).
