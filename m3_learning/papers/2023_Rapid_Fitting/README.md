# Rapid Fitting of BE-PFM — Reproduction Guide

Code for *"Rapid Fitting of Band-Excitation Piezoresponse Force Microscopy Using
Physics Constrained Unsupervised Neural Networks"* (NeurIPS 2023, ML4PS workshop).

The notebooks run in order on Google Colab (GPU runtime recommended) or locally.
Each notebook's first cell clones this repo (`BRANCH = "shofit"`), pip-installs the
`m3_learning` SDK, and upgrades `numpy_groupies` (required: the BGlib dependency
pins a version that crashes on NumPy ≥ 2). Data downloads automatically from
[Zenodo record 7774788](https://zenodo.org/record/7774788) (1.74 GB raw; the h5
grows to ~10+ GB as noisy records and fits are added).

## Notebook pipeline

| # | Notebook | What it does | Needs |
|---|----------|--------------|-------|
| 0 | `0_Introduction` | Background (prose only) | — |
| 0.5 | `0_5_Noisy_Data_and_Fitting` | Generates noisy records (noise 1–8) inside the h5 and LSQF-fits all datasets | Zenodo data |
| 1 | `1_SHO_Fitting` | LSQF SHO baseline on raw data + fit visualization | Zenodo data |
| 2 | `2_Pytorch_SHO_Fitter` | Trains the physics-constrained NN (paper's core model) | NB 1 fits |
| 2.5 | `2_5_nn_fitting_all` | Benchmark sweep: optimizers × noise × batch × seeds | NB 0.5 fits (for noise 1–8) |
| 3 | `3_Second_Order_Optimizers` | Adam vs AdaHessian vs Trust-Region CG comparison | NB 1 fits |
| 4 | `4_Noisy_Data_Analysis` | Noise-robustness analysis of trained models | NB 0.5 + 2.5 outputs |
| 4.5 | `4_5_benchmark_analysis` | Benchmark figures from `record_from_datafed.csv` | tracked CSV (self-contained) |
| 5 | `5_Hysteresis_Fitter` | 9-parameter hysteresis loop fitting (LSQF + NN/TRCG) | NB 1 fits |
| 6 | `6_Figures` | Regenerates the paper's figures | NB 1/2.5/5 outputs |

All long computations are cached inside the h5 (BGlib prints *"already been
performed"* and returns instantly on re-runs) or under `Trained Models/`.

## QUICK_RUN

Set `QUICK_RUN = True` in any notebook's first cell for a fast reduced-scale pass
(subset of pixels/samples, 1–2 epochs, fewer noise levels) — used to verify every
notebook end-to-end on CPU. Cells whose full-scale artifacts are unavailable under
QUICK_RUN skip themselves with a printed notice. The committed default is
`QUICK_RUN = False` (the full paper workflow).

## Dataerai provenance logging

The Rapid Fitting notebooks that train neural networks can record each training
run in Dataerai's lineage graph. The training code uses the Dataerai SDK helper
`dataerai.ml.lineage.record_training_run`; users authenticate and select/upload
the source data with the normal CLI.

```bash
python -m pip install dataerai-cli 'dataerai-sdk[ml]'
dataerai auth login --device --server https://beta.dataerai.com
dataerai auth status
```

Upload the source h5 once, or reuse an existing Dataerai asset id:

```bash
cat > dataerai_source_asset.json <<'JSON'
{
  "title": "PZT 2080 raw BE-PFM data",
  "description": "Source data for the Rapid Fitting SHO neural-network notebooks.",
  "tags": ["be-pfm", "sho", "neural-network", "m3-learning"],
  "metadata": {"source": "Zenodo 7774788"}
}
JSON

dataerai upload \
  --project <project-uuid> \
  --collection <collection-uuid> \
  --metadata dataerai_source_asset.json \
  --file Data/data_raw.h5
```

Then enable logging before running the training cells:

```bash
export DATAERAI_DATASET_ASSET_ID=<asset-id-from-upload>
export DATAERAI_LOG_PROVENANCE=1
```

If you already know the provenance surrogate, set
`DATAERAI_DATASET_RECORD_SK=<record-sk>` instead of `DATAERAI_DATASET_ASSET_ID`.
The notebooks print the Dataerai lineage run id after each successful training
run.

## Local development

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ./m3_learning
.venv/bin/pip install --upgrade numpy_groupies   # required, see above
.venv/bin/pytest m3_learning/src/m3_learning/tests/test_smoke.py
```

The smoke tests (7 tests, seconds, CPU-only, no data download) cover the SHO
function, scalers, the Fitter1D model forward/backward in train/eval modes, the
9-parameter hysteresis loop function, and one optimization step of AdaHessian and
Trust-Region CG.
