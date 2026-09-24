# Manuscript figures — "When the decoder is the physics" (npj submission)

Final figure panels of the manuscript (main text + Supplementary
Information) and the sources that build them, kept next to the notebooks
that generate the underlying data (`../`).

- Top level: the exact PNG/PDF panels included by the manuscript.
- `src/`: the committed generators — TikZ/pgfplots sources, PlotNeuralNet
  architecture sources, Python plotting scripts, and the provenance-graph
  builder (`src/figprov/`), including the archived platform API response
  (`src/figprov/graph_manifest.json`) that the Supplementary Information
  cites as the identifier manifest.

Every panel is additionally a registered *figure* record on the Dataerai
platform (collection `e00425d0-9b2b-4e5f-8299-b402301027bc`); per-panel
record identifiers are listed in Supplementary Tables S5–S6 of the
manuscript. Data-driven panels carry an `analysis_of` relationship to the
dataset record `6323cd64-933d-4df2-af80-6000d481698e`.

## How each panel is produced

Generators come in three kinds:

- **script** — a committed generator in `src/`; run it directly (`make` for
  TikZ directories, `python <script>` otherwise). These regenerate the
  committed panel exactly.
- **notebook** — the panel is rendered by the repository's plotting
  utilities from the outputs of the numbered analysis notebooks in `../`
  (run on the Zenodo dataset `10.5281/zenodo.7774788`; see `../README.md`
  for the notebook workflow). Regenerated versions reproduce the content;
  cosmetic details may differ with library versions.
- **archived** — no generative source exists; the committed file is the
  canonical artifact (also registered on the platform).

### Main text

| Panel | File | Generator |
|---|---|---|
| Fig 1 | `fig_bepfm_schematic.pdf` | archived (legacy schematic collage; no source) |
| Fig 2 | `fig1_concept.pdf` | script — `src/fig1/` (`make`) |
| Fig 3a | `fig2_bmw.png` | notebook — NB 2/6 (best/median/worst SHO fits) |
| Fig 3b | `fig2_violin.png` | notebook — NB 2/6 (clean parameter histograms) |
| Fig 4a | `fig_noise_violin.png` | notebook — NB 4/6 (per-noise parameter histograms) |
| Fig 4b | `fig_accuracy_noise.png` | script — `src/fig_accuracy/make_fig_accuracy_noise.py` (reads `../analysis/phase0_E4_E5_noise_sweep.csv`) |
| Fig 5 | `fig_truth_error_noise.png` | script — `src/fig_truth/make_fig_truth_error.py` (reads `../analysis/e13_results.json`, produced by `../analysis/e13_pipeline.py`) |
| Fig 6a | `fig5_loops_bmw.png` | notebook — NB 5/6 (best/median/worst hysteresis fits) |
| Fig 6b | `fig5_hysteresis_violin.png` | notebook — NB 5/6 (loop-parameter histograms) |

### Supplementary Information

| Panel | File(s) | Generator |
|---|---|---|
| S1 | `fig_param_scatter.png` | script — `src/fig_si_clean/make_si_clean_figs.py` (reads `../Data/data_raw.h5` + the released clean Adam checkpoint; prints sanity anchors that must match the manuscript: medians 0.0342/0.0337, NN lower 69.7%, A Pearson 0.996) |
| S2 | `fig_mse_dist.png` | script — same run as S1 |
| S3a–g | `si_hist_noise{1,2,3,4,5,6,8}.png` | notebook — NB 4 (full noise series) |
| S4a–b | `fig3_hist_noise{4,7}.png` | notebook — NB 4 (three-row histograms) |
| S5 | `fig3_switchingmaps_noise7.png` | notebook — NB 4 (switching maps at n=7) |
| S6a–d | `si_nnval_noise{2,4,6,8}.png` | notebook — NB 4 (network validation fits) |
| S7 | `fig5_hysteresis_maps.png` | notebook — NB 5/6 (loop-parameter spatial maps) |
| S8 | `fig2_switchingmaps.png` | notebook — NB 1/2 (clean switching maps) |
| S9 | `fig4_adahessian_bmw.png` | notebook — NB 3 (AdaHessian best/median/worst) |
| S10 | `fig5_fpga_latency.png` | script — `src/fig5/make_fig5_fpga.py` (values from main-text Table 2) |
| S11a–b | `fig4_clean_loss.png`, `fig4_margin_vs_noise.png` | script — `src/fig4/make_fig4.py` (reads the optimizer-benchmark CSV emitted by NB 2.5/3) |
| S12a | `fig1_architecture.pdf` | script — `src/fig1arch/` (`make`; dimensions verified against the released model code) |
| S12b | `fig_hysteresis_architecture.pdf` | script — `src/fig2arch/` (`make`) |
| S13 | `fig_provenance_graph.pdf` | script — `src/figprov/` (`make`; nodes from the archived `graph_manifest.json`) |

Checkpoints referenced by the scripts (the clean Adam SHO fitter and the
TR-CG hysteresis fitter) are registered records in the paper's platform
collection, public at acceptance; each script accepts explicit paths as
arguments if your copies live elsewhere.

`src/pubstyle.py` is the shared plotting style (Okabe-Ito palette;
`METHOD_COLORS` fixes LSQF = blue, network = orange everywhere).
