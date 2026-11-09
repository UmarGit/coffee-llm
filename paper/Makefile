# Rebuild the paper from the neg-v1 campaign (tools/analysis). Nothing is hand typed.
PY := uv run --extra pizza python
TEX := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all figures tables review clean distclean check
all: figures tables paper.pdf

figures:
	$(PY) make_figures.py
	$(PY) make_tokenization.py

tables:
	$(PY) ../tools/analysis/errors.py
	$(PY) ../tools/analysis/extra.py
	$(PY) make_tables.py

paper.pdf: paper.tex refs.bib $(wildcard generated/*.tex) $(wildcard figures/*.pdf)
	$(TEX) paper.tex

# Anonymous submission build: line numbers, no author block.
review: paper.tex refs.bib
	$(TEX) -jobname=paper-review -usepretex='\def\aclmode{review}' paper.tex

check: paper.pdf
	@echo "pages:     $$(grep -oE 'paper.pdf \([0-9]+ pages' paper.log | grep -oE '[0-9]+')"
	@echo "overfull:  $$(grep -c Overfull paper.log)"
	@echo "undefined: $$(grep -c 'undefined' paper.log)"
	@gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=/tmp/_p.txt paper.pdf 2>/dev/null; \
	 echo "em dashes: $$(python3 -c "print(open('/tmp/_p.txt',encoding='utf-8',errors='replace').read().count(chr(8212)))")"

# Only latexmk's own droppings. generated/ and figures/ are NOT removed: they are
# built from runs/, which is CC-BY-NC and never committed, so once deleted they
# cannot be rebuilt from a clone. `make distclean` removes them deliberately.
clean:
	latexmk -c

distclean: clean
	@echo "This deletes generated/ and figures/, which only a full runs/ can rebuild."
	@read -r -p "type 'yes' to continue: " a && [ "$$a" = yes ]
	latexmk -C
	rm -rf generated figures
