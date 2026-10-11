# The manuscript and its supplement: `make` in this directory writes
# build/psmc-rs.pdf and build/supplement.pdf. The numbers and tables are
# generated from the committed results first (make_numbers.py), and
# Supplementary Note 1 is converted from its Markdown in the repository.
PY ?= ../experiment/.venv/bin/python
TECTONIC ?= tectonic
BUILD := build
SN1 := ../experiment/Supplementary_Note_1_Reproducibility.md

all: numbers sn1.tex $(BUILD)/psmc-rs.pdf $(BUILD)/supplement.pdf

numbers:
	$(PY) make_numbers.py

sn1.tex: $(SN1) make_sn1.py
	python3 make_sn1.py

$(BUILD)/psmc-rs.pdf: psmc-rs.tex numbers.tex table1.tex table2.tex references.bib
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) psmc-rs.tex

$(BUILD)/supplement.pdf: supplement.tex numbers.tex tableS1.tex tableS2.tex sn1.tex references.bib
	mkdir -p $(BUILD) && $(TECTONIC) --outdir $(BUILD) supplement.tex

# exit 1 if numbers.tex or a table is not what the committed results give
check:
	$(PY) make_numbers.py --check

clean:
	rm -rf $(BUILD)

.PHONY: all numbers check clean
