# The manuscript: `make` in this directory writes build/psmc-rs.pdf, the article with its supplementary
# information (Figs S1-S3 and Table S1) at the end. The numbers and tables are generated from the committed results
# first (make_numbers.py). The PDF is rebuilt when its text, a number or table it quotes, or a figure it includes
# changed.
PY ?= ../experiment/.venv/bin/python
TECTONIC ?= tectonic
BUILD := build
# The figures a document includes: its \includegraphics files, under its \graphicspath.
figures = $(addprefix ../experiment/runs/,$(shell sed -n 's/.*includegraphics[^{]*{\([^}]*\)}.*/\1/p' $(1)))

all: $(BUILD)/psmc-rs.pdf

# make_numbers.py reads more of the results than a rule could list, so it runs every time; it rewrites only the
# files whose text changed, and make rebuilds the PDF only if one it quotes did.
numbers.tex: FORCE
	$(PY) make_numbers.py

table1.tex table2.tex tableS1.tex: numbers.tex ;

numbers: numbers.tex

$(BUILD)/psmc-rs.pdf: psmc-rs.tex supplement-floats.tex numbers.tex table1.tex table2.tex tableS1.tex references.bib $(call figures,psmc-rs.tex supplement-floats.tex)
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) psmc-rs.tex

# The GigaByte version: build/gigabyte/ holds what to upload to GigaByte's Overleaf template (gigabyte.tex,
# numbers.tex, the two tables, references.bib, the three figures), a preview of it (gigabyte.pdf, compiled
# without the template's class) and Additional file 1 (additional-file-1.pdf, Figs S1-S3 and Table S1).
GB := $(BUILD)/gigabyte
GB_FIGURES := FIG/figures/art1_agreement.pdf E3/figures/art2_performance.pdf E7/figures/fig7_1000_genomes.pdf
gigabyte: numbers.tex
	mkdir -p $(GB)
	cp gigabyte.tex numbers.tex table1.tex table2.tex references.bib $(GB)/
	cp $(addprefix ../experiment/runs/,$(GB_FIGURES)) $(GB)/
	cd $(GB) && $(TECTONIC) gigabyte.tex
	$(TECTONIC) --outdir $(GB) gigabyte-additional-file-1.tex
	mv $(GB)/gigabyte-additional-file-1.pdf $(GB)/additional-file-1.pdf

# exit 1 if numbers.tex or a table is not what the committed results give
check:
	$(PY) make_numbers.py --check

clean:
	rm -rf $(BUILD)

.PHONY: all numbers check clean gigabyte FORCE
