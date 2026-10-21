# Rebuild the paper end to end from runs/. Nothing is hand typed.
PY := uv run --extra pizza python

.PHONY: all figures tables clean distclean check
all: figures tables paper.pdf

figures:
	$(PY) make_figures.py

tables:
	$(PY) make_tables.py

paper.pdf: paper.tex generated/macros.tex generated/main_table.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

check: paper.pdf
	@echo "pages:    $$(grep -oE '\([0-9]+ pages' paper.fls 2>/dev/null || grep -oE '\([0-9]+ pages' *.log | tail -1)"
	@echo "overfull: $$(grep -c Overfull paper.log)"
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
