# Dataerai notebook artifact publishing

## Decision

M3 notebooks will keep one Dataerai JSON execution log per run and promote the
scientific artifacts produced by that run into first-class Dataerai assets.
Publishing is implemented by one reusable `m3_learning` helper started beside
the `%dataerai --trace` magic and flushed immediately before `%dataerai
--finish`.

The helper uploads the source notebook, reuses source datasets by a stable,
exact asset title; captures full
rich-image payloads into deterministic files before the execution log applies
its bounded-JSON policy; versions changed HDF5 and CSV data by uploading them
under stable titles; and publishes checkpoints, loss histories, and generated
manifests. PyTorch training uses `dataerai.nn.PyTorchProvenanceTracker`, which
preserves model and optimizer state in a safe `.pt` checkpoint. Routed uploads
are explicitly registered with the active notebook trace, allowing it to stamp
the notebook run identity and create `records_telemetry` relationships from the
execution log.

Each notebook owns a stable provenance root with `Executions`, `Notebooks`,
`Data/Raw`, `Data/Derived`, `Figures`, `Movies`, `Models`, and `Manifests`.
Local subfolders created by sweep loops are mirrored under the appropriate
category collection.

## Role in the software

`m3_learning.artifacts` is the notebook-to-Dataerai publication boundary. It
does not change research calculations or teach the Dataerai SDK how to trace a
cell. Instead, it observes the same display channel and the notebook's changed
files, assigns scientific record types, and authors domain-level relationships
to the reused source dataset. If the display publisher does not support hooks,
it falls back to the trace's captured display bundle.

## Preserved contracts

- Scientific cells, plotting functions, filenames, and local model saving are
  unchanged.
- A notebook run still produces exactly one JSON execution-log asset.
- Every run has a distinct execution-log identity; earlier runs are preserved.
- Files unrelated to the documented artifact extensions are never scanned or
  uploaded.
- An unchanged source dataset is not uploaded again by later notebooks.
- Stable derived titles deliberately create new content versions instead of
  duplicate assets.
- Derived files above the configurable inline limit produce a versioned
  dataset-manifest asset rather than failing every later notebook against a
  smaller storage allocation.

## Publication relationships

- figure/analysis → source dataset: `analysis_of`
- derived HDF5/CSV → source dataset: `derived_from`
- model checkpoint → source dataset: `trained_on`
- loss history → checkpoint: `training_output_of`
- manifest → checkpoint: `describes`
- execution log → every upload: `records_telemetry` (authored by the SDK trace)
- source notebook → source dataset: `uses_data`
- every generated product → source notebook: `generated_by_notebook`

## Recovery and safety

The artifact directory is local and regenerable. Uploads use deterministic
titles, so retrying versions existing assets rather than creating a second
logical artifact. Relationship-exists responses are treated as idempotent;
other publication errors are reported and make the final publication step
fail rather than presenting partial provenance as success.

## Verification matrix

| Requirement | Independent oracle | Test evidence |
|---|---|---|
| Upload raw once and reuse | exact-title lookup before upload | `test_raw_dataset_is_uploaded_once_then_reused_by_exact_stable_title` |
| Publish every captured figure | untruncated display hook with trace fallback | `test_each_rich_figure_gets_a_deterministic_analysis_asset`, `test_direct_display_capture_preserves_figures_larger_than_trace_limit` |
| Version HDF5/CSV | Dataerai same-title upload contract | `test_changed_hdf5_and_csv_use_stable_titles_for_content_versioning` |
| Publish source notebook and hierarchy | immutable notebook upload plus routed collections | `test_source_notebook_and_nested_output_folders_are_first_class_collections` |
| Publish model bundle | specialized PyTorch tracker plus linked M3 outputs | `test_specialized_pytorch_tracker_routes_complete_training_bundle` |
| Link products | SDK relationship contract | model/figure/derived assertions above |
| Keep one execution log | all products upload through traced session | `test_every_uploaded_product_uses_the_traced_session_for_execution_linking` |
