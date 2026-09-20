# Analysis scripts for the revision experiments

Standalone scripts behind two Methods sections; both read the public Zenodo
dataset (10.5281/zenodo.7774788, `data_raw.h5`) and the trained checkpoints,
and write JSON/CSV summaries. Work-directory paths at the top of each script
are configurable; both run on CPU only.

- `e13_pipeline.py` — ground-truth recovery study (Methods, "Ground-truth
  recovery study"): builds the synthetic/semi-synthetic truth sets from clean
  least-squares fits, trains noise-matched networks, and evaluates four
  estimators (heuristic LSQF, multistart, empirical-prior GMM MAP, network)
  with branch canonicalization.
- `e12_loop_refit.py` — initialization-controlled hysteresis-loop refit
  (Methods, "Hysteresis-loop fitting"): numpy port of the loop function
  (verified against the PyTorch implementation to ~1e-20), unbounded
  Levenberg-Marquardt refits from network and least-squares initializations,
  pixel-clustered bootstrap intervals.

Numerical outputs from the runs reported in the paper are archived on the
Dataerai platform (revision-experiment archive; see Methods, "Data and
provenance management").

## Reproduction details

- **Environment (these scripts)**: Python 3.11.4, PyTorch 2.12.0, NumPy 2.4.6,
  h5py; CPU-only (both scripts ran on an Apple-silicon Mac; no CUDA required).
- **Training stack**: the scripts import `SHO_fit_func_nn`,
  `Multiscale1DFitter`/`Model`/`ComplexPostProcessor`, `random_seed`,
  `BE_Dataset`, and `loop_fitting_function_torch` from
  `m3-learning/m3_learning` (`m3_learning/src/m3_learning`). They were run
  against a local tree (commit `18e6bdad`, not publicly reachable); the
  publicly reachable pin is **`245d6bf`** (on branches `shofit` and
  `codex/dataerai-provenance-logging`). Of the five imported modules, three
  are byte-identical between the two trees, and `be/nn.py` +
  `nn/Fitter1D/Fitter1D.py` differ only by the opt-in, default-off Dataerai
  provenance parameters (and an associated checkpoint-path variable
  extraction) introduced by PR #26 --- no fitting or training logic differs
  (verified by per-module diff). Reproduce against `245d6bf`.
- **Separate from the above**: the paper's twenty provenance-logged training
  runs (Methods, "Data and provenance management") were executed on a Lambda
  A10 instance at commit `245d6bf` under Python 3.10 with `dataerai-sdk`
  0.1.0 installed; that environment is distinct from the Mac environment used
  for these two analysis scripts. The SDK is a lazy, fail-safe dependency of
  the provenance hooks only (`m3_learning/provenance.py` imports
  `dataerai.ml.lineage` inside the logging call and skips cleanly when the
  package is absent), so reproducing the training or these analyses does not
  require it. The same entry point is still present and call-compatible in
  the current SDK beta (0.2.0b27).
- **Configurable paths**: each script defines its inputs as constants near the
  top — `H5` (path to `data_raw.h5`; download from DOI 10.5281/zenodo.7774788),
  `WORK`/`OUT` (scratch directories), and in `e12_loop_refit.py` the hysteresis
  checkpoint path. Edit these constants for your machine; the committed copies
  are otherwise byte-identical to the archived versions in the Dataerai
  revision-experiment record.
- **Checkpoints**: `e13_pipeline.py` trains its own networks from scratch
  (~14 CPU-trainings, checkpointed stage by stage so a crash never loses
  compute). `e12_loop_refit.py` needs the paper's trained hysteresis
  checkpoint, available in the Dataerai collection "Trained models —
  provenance rerun 2026-07-13" (each checkpoint asset's metadata records its
  training-run lineage id, optimizer, seed, and loss; server-side integrity
  roots serve as checksums).
- **Expected outputs**: `e13_results.json` (per-noise, per-estimator parameter
  errors; the numbers behind main-text Fig. 5) and the E12 refit JSON (median
  MSEs and pixel-clustered bootstrap intervals quoted in Results and the
  Fig. 6 caption).
