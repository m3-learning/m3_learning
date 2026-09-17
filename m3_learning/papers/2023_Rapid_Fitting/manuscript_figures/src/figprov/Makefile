# Build the Dataerai provenance-graph figure (SI).
#
#     make        # -> fig_provenance_graph.pdf, cropped to the ink bbox
#
# Output: ../../fig_provenance_graph.pdf (included by manuscript/si.tex).

PDF = fig_provenance_graph.pdf

$(PDF): build.tex figprov_preamble.tex figprov_body.tex crop.py
	pdflatex -interaction=nonstopmode build.tex
	python3 crop.py build.pdf $(PDF)
	cp $(PDF) ../../$(PDF)
	$(RM) build.pdf build.aux build.log

clean:
	$(RM) *.aux *.log build.pdf missfont.log

.PHONY: clean
