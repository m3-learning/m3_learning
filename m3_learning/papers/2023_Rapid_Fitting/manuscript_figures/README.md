# Manuscript figures — "When the decoder is the physics" (npj submission)

Final figure panels of the manuscript (main text + Supplementary
Information) and the sources that build them, kept next to the notebooks
that generate the underlying data (`../`).

- Top level: the exact PNG/PDF panels included by the manuscript.
- `src/`: TikZ/pgfplots sources, PlotNeuralNet architecture sources,
  and the provenance-graph builder (`src/figprov/`), including the
  archived platform API response (`src/figprov/graph_manifest.json`)
  that the Supplementary Information cites as the identifier manifest.

Every panel is additionally a registered *figure* record on the Dataerai
platform (collection `e00425d0-9b2b-4e5f-8299-b402301027bc`); per-panel
record identifiers are listed in Supplementary Tables S5–S6 of the
manuscript. Data-driven panels carry an `analysis_of` relationship to the
dataset record `6323cd64-933d-4df2-af80-6000d481698e`.
