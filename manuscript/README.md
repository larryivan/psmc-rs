# Manuscript

The article (`psmc-rs.tex`: three figures, two tables), with its
supplementary information (Figs S1–S3 and Table S1) at the end: one PDF.

```bash
make            # numbers and tables from the results, then build/psmc-rs.pdf
make check      # exit 1 if numbers.tex or a table is not what the committed results give
```

`make` rewrites only the numbers and tables whose text changed, and rebuilds
the PDF only if its text, numbers, tables or included figures
(`experiment/runs/*/figures/*.pdf`) changed. The build needs the experiment
environment (`experiment/.venv`, see `experiment/REPRODUCIBILITY.md`,
section 2) and [tectonic](https://tectonic-typesetting.github.io) 0.17.0.

- `make_numbers.py` writes `numbers.tex`, one `\newcommand` per number the
  text quotes, each with its source in a comment, and Tables 1, 2 and S1
  (`table1.tex`, `table2.tex`, `tableS1.tex`), from the committed tables of
  `experiment/runs/*/tables`, E7's `MANIFEST.json` and chr21 pilot
  (`experiment/e7_pilot`), the two runs of NA12878's bootstrap,
  `experiment/crossplatform`, the diagnostic of the residual difference
  from C psmc (`experiment/c_equivalence`) and the production profile of
  `experiment.pipeline.config`. The text quotes no
  result that is not one of these macros, and the generator checks with
  assertions the claims the text makes in words (for example, that
  splitting the turtles' first time window removed their recent peaks in every fit).
- `references.bib` was fetched from doi.org, its years checked against
  Crossref's print dates; the software entries were checked against their
  repositories.
- The figures are those of `experiment/runs/*/figures`, drawn by the
  pipeline (`python -m experiment.pipeline.figures`);
  `experiment/REPRODUCIBILITY.md`, section 3, maps the article's figure
  numbers to their files.
