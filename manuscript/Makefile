# The manuscript and its supplement: `make` in this directory writes build/psmc-rs.pdf, build/supplement.pdf and
# build/supplementary_note_1.pdf. The numbers and tables are generated from the committed results first
# (make_numbers.py), and Supplementary Note 1 is converted from its Markdown in the repository. A PDF is rebuilt
# when its text, a number or table it quotes, or a figure it includes changed.
PY ?= ../experiment/.venv/bin/python
TECTONIC ?= tectonic
BUILD := build
SN1 := ../experiment/Supplementary_Note_1_Reproducibility.md
# The figures a document includes: its \includegraphics files, under its \graphicspath.
figures = $(addprefix ../experiment/runs/,$(shell sed -n 's/.*includegraphics[^{]*{\([^}]*\)}.*/\1/p' $(1)))

all: $(BUILD)/psmc-rs.pdf $(BUILD)/supplement.pdf $(BUILD)/supplementary_note_1.pdf

# make_numbers.py reads more of the results than a rule could list, so it runs every time; it rewrites only the
# files whose text changed, and make rebuilds only the PDFs that quote them.
numbers.tex: FORCE
	$(PY) make_numbers.py

table1.tex table2.tex tableS1.tex: numbers.tex ;

numbers: numbers.tex

sn1.tex: $(SN1) make_sn1.py
	python3 make_sn1.py

$(BUILD)/psmc-rs.pdf: psmc-rs.tex numbers.tex table1.tex table2.tex references.bib $(call figures,psmc-rs.tex)
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) psmc-rs.tex

$(BUILD)/supplement.pdf: supplement.tex numbers.tex tableS1.tex $(call figures,supplement.tex)
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) supplement.tex

$(BUILD)/supplementary_note_1.pdf: supplementary_note_1.tex sn1.tex
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) supplementary_note_1.tex

# exit 1 if numbers.tex or a table is not what the committed results give
check:
	$(PY) make_numbers.py --check

clean:
	rm -rf $(BUILD)

.PHONY: all numbers check clean FORCE
