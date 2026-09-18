# Build Figure 1 panel (b): the physics-constrained autoencoder, rendered with
# the authors' PlotNeuralNet layers package (vendored in ./layers/).
#
# standalone.cls is unavailable on this machine, so this uses the article
# fallback: render onto a large page, then crop to the figure bbox with
# PyMuPDF (needs Python + PyMuPDF).
#
#     make            # -> fig1_architecture.pdf, auto-cropped to the bbox
#
# Output is copied to ../../fig1_architecture.pdf (included by the manuscript).

PDF = fig1_architecture.pdf
# Make the vendored layers/*.sty and init.tex reachable by kpathsea.
export TEXINPUTS := .//:./layers//:$(TEXINPUTS)

$(PDF): build.tex fig1arch_body.tex SHO_input.png crop.py $(wildcard layers/*)
	pdflatex -interaction=nonstopmode build.tex
	python3 crop.py build.pdf $(PDF)
	cp $(PDF) ../../$(PDF)
	$(RM) build.pdf build.aux build.log

clean:
	$(RM) *.aux *.log build.pdf missfont.log

.PHONY: clean
