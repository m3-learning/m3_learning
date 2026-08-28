# Rapid Fitting of BE-PFM — Reproduction Guide

Code for *"Rapid Fitting of Band-Excitation Piezoresponse Force Microscopy Using
Physics Constrained Unsupervised Neural Networks"* (NeurIPS 2023, ML4PS workshop).

The notebooks run in order on Google Colab (GPU runtime recommended) or locally.
Each notebook's first cell clones this feature branch, pip-installs the
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

## Dataerai notebook and neural-network provenance

Every authored notebook starts a Dataerai execution trace and artifact publisher
and finishes both in the final code cell. The trace captures cell execution. The
publisher reuses the raw HDF5 asset, exports each rich figure, versions changed
HDF5/CSV data, and publishes model checkpoints, loss histories, and manifests.
All products carry the notebook run ID and receive source-data relationships;
the execution log receives `records_telemetry` relationships from the SDK.

```bash
python -m pip install --pre dataerai-cli-beta==0.1.54 'dataerai-sdk[notebook,nn-pytorch]==0.2.0b52'
dataerai auth login --device --client-id dataerai-mobile --server https://beta.dataerai.com
dataerai auth status
```

The first notebook to see `Data/data_raw.h5` uploads it under a stable source
title. Later notebooks find and reuse that exact asset, including across fresh
cloud kernels. Set only the destination before launching Jupyter; artifact and
training provenance are enabled by the managed setup cell.

```bash
export DATAERAI_DESTINATION_COLLECTION_PATH='My Project / M3 Learning Runs'
```

If you already know a source asset, set
`DATAERAI_RAW_DATA_ASSET_ID=<asset-id>` (and optionally
`DATAERAI_DATASET_ASSET_ID` to the same value);
if you know the provenance surrogate, set `DATAERAI_DATASET_RECORD_SK=<record-sk>`.
Each notebook creates a provenance root containing `Executions`, `Notebooks`,
`Data/Raw`, `Data/Derived`, `Figures`, `Movies`, `Models`, and `Manifests`.
Sweep folders such as a noise-level output folder become nested collections.
The fitters print the Dataerai PyTorch checkpoint asset ID after each successful
training run. The artifact summary then prints before `%dataerai --finish`
publishes that run's distinct execution log.

A successful final cell ends with output in this form (counts vary by notebook):

```text
Dataerai artifacts: 1 source dataset, 8 analyses, 2 derived datasets, 3 model artifacts.
Published notebook execution trace <run-id> (<cell-count> cells, <product-count> products).
```

The four managed provenance cells in each source notebook can be regenerated
idempotently with `python tools/update_dataerai_notebook_provenance.py`.

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
