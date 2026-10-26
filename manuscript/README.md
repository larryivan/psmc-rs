# Manuscript

The article (`psmc-rs.tex`: three figures, two tables), its supplementary
information (`supplement.tex`: Figs S1–S3 and Table S1) and Supplementary
Note 1 (`supplementary_note_1.tex`, from the repository's Markdown).

```bash
make            # numbers and tables from the results, then the three PDFs in build/
make check      # exit 1 if numbers.tex or a table is not what the committed results give
```

`make` rewrites only the numbers and tables whose text changed, and rebuilds
only the PDFs whose text, numbers, tables or included figures
(`experiment/runs/*/figures/*.pdf`) changed. The build needs the experiment
environment (`experiment/.venv`, see Supplementary Note 1, section 2),
[tectonic](https://tectonic-typesetting.github.io) 0.17.0 and pandoc 3.8
or later (`make_sn1.py` uses its `--syntax-highlighting`; `sn1.tex` was
made with 3.9.0.2).

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
- `make_sn1.py` converts `experiment/Supplementary_Note_1_Reproducibility.md`
  into `sn1.tex`.
- `references.bib` was fetched from doi.org, its years checked against
  Crossref's print dates; the software entries were checked against their
  repositories.
- The figures are those of `experiment/runs/*/figures`, drawn by the
  pipeline (`python -m experiment.pipeline.figures`); Supplementary Note 1,
  section 3, maps the article's figure numbers to their files.

Red bracketed text in the PDFs marks what the author still has to supply:
names, affiliation, the source of the turtle data, Zenodo DOIs, the
computing allocation, funding, contributions and the statement on AI tools.
