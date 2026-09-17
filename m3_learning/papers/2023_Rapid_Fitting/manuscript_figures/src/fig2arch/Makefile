# Build Figure 2 panel (c): the SAME physics-constrained autoencoder as the SHO
# network (Multiscale1DFitter), reconfigured for hysteresis loops --- one input
# channel and nine latent outputs --- with the 9-parameter empirical loop
# function as the decoder. Rendered with the authors' PlotNeuralNet layers
# package (vendored in ./layers/).
#
# standalone.cls is unavailable on this machine, so this uses the article
# fallback: render onto a large page, then crop to the figure bbox with
# PyMuPDF (needs Python + PyMuPDF).
#
#     make            # -> fig_hysteresis_architecture.pdf, auto-cropped
#
# Output is copied to ../../fig_hysteresis_architecture.pdf (included by the
# manuscript).

PDF = fig_hysteresis_architecture.pdf
# Make the vendored layers/*.sty and init.tex reachable by kpathsea.
export TEXINPUTS := .//:./layers//:$(TEXINPUTS)

$(PDF): build.tex fig2arch_body.tex loop_input.png crop.py $(wildcard layers/*)
	pdflatex -interaction=nonstopmode build.tex
	python3 crop.py build.pdf $(PDF)
	cp $(PDF) ../../$(PDF)
	$(RM) build.pdf build.aux build.log

clean:
	$(RM) *.aux *.log build.pdf missfont.log

.PHONY: clean
