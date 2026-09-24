# Build Figure 1 (concept + physics-constrained autoencoder).
#
# Preferred (needs the `standalone` LaTeX class):
#     make standalone
#
# Fallback (no standalone.cls / pdfcrop; needs Python + PyMuPDF):
#     make           # -> fig1_concept.pdf, auto-cropped to the figure bbox
#
# Output: ../../fig1_concept.pdf (included by manuscript/main.tex).

PDF = fig1_concept.pdf

$(PDF): build.tex fig1_preamble.tex fig1_body.tex crop.py
	pdflatex -interaction=nonstopmode build.tex
	python3 crop.py build.pdf $(PDF)
	cp $(PDF) ../../$(PDF)
	$(RM) build.pdf build.aux build.log

standalone: fig1_concept.tex fig1_preamble.tex fig1_body.tex
	pdflatex -interaction=nonstopmode fig1_concept.tex
	cp $(PDF) ../../$(PDF)

clean:
	$(RM) *.aux *.log build.pdf missfont.log

.PHONY: standalone clean
