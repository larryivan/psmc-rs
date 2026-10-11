# Manuscript

The article (`psmc-rs.tex`) and its supplementary information
(`supplement.tex`: Figs S1–S7, Table S1 and Supplementary Note 1).

```bash
make            # numbers and tables from the results, then build/psmc-rs.pdf and build/supplement.pdf
make check      # exit 1 if numbers.tex or a table is not what the committed results give
```

The build needs the experiment environment (`experiment/.venv`, see
Supplementary Note 1, section 2), [tectonic](https://tectonic-typesetting.github.io)
and pandoc.

- `make_numbers.py` writes `numbers.tex`, one `\newcommand` per number the
  text quotes, each with its source in a comment, and Tables 1, 2 and S1
  (`table1.tex`, `table2.tex`, `tableS1.tex`), from the committed tables of
  `experiment/runs/*/tables`, E7's `MANIFEST.json`, the two runs of
  NA12878's bootstrap, `experiment/crossplatform` and the README's
  diagnostic of the residual difference from C psmc. The text quotes no
  result that is not one of these macros, and the generator checks with
  assertions the claims the text makes in words (for example, that no fit of
  the 1000 Genomes cohort reached the bound on lambda).
- `make_sn1.py` converts `experiment/Supplementary_Note_1_Reproducibility.md`
  into `sn1.tex`.
- `references.bib` was fetched from doi.org, its years checked against
  Crossref's print dates; the software entries were checked against their
  repositories.
- The figures are those of `experiment/runs/*/figures`, drawn by the
  pipeline (`python -m experiment.pipeline.figures`).

Red bracketed text in the PDFs marks what the author still has to supply:
names, affiliation, the source of the turtle data, Zenodo DOIs, funding,
contributions and the statement on AI tools.
