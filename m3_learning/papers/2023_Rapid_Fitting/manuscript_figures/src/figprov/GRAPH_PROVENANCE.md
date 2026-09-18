# What Supplementary Fig. S13 encodes, and how it was verified

The figure is a hand-drawn (TikZ) diagram of the paper's artifact graph on the
Dataerai platform. It is not rendered from an API response; every node and
edge it shows was verified against the live platform as follows.

## Verified via the REST API

- **2026-07-13/14** — all relationship edges on dataset asset
  `6323cd64-933d-4df2-af80-6000d481698e` confirmed from both endpoints of each
  edge (`GET /api/assets/{id}/relationships/`): 20 model-checkpoint assets
  `derived_from` the dataset (each cross-referencing its lineage run in asset
  metadata), the optimizer-benchmark table `analysis_of`, 8 data-driven
  main-text figure assets `analysis_of` (plus `derived_from` their checkpoint
  or benchmark-table source), and the `derived_from` edge to the metadata-only
  Zenodo-original asset. 30 edges total on the dataset at that date.
- **2026-07-16** — full lineage listing via
  `GET /api/lineage/trace/?subject_kind=1&subject_id=<asset>&direction=up`:
  41 closed runs, all with server-side integrity roots (20 paper training
  runs sks 298–317; 19 revision-experiment runs; 2 early smoke runs), plus the
  `analysis_of` edge for the revision-experiment archive added that day.
- The platform's collection-scope provenance-graph UI was visually confirmed
  to render the same star-shaped graph (2026-07-14).

## Re-verified 2026-07-27 (post terms-of-service update)

- CLI auth valid; dataset asset resolves and its metadata is readable
  (`dataerai metadata get`); the revision-archive asset content downloads
  intact. No writes have been made to the collection since 2026-07-16, so the
  figure reflects the last-written state.
- A fresh full API pull of the graph was not possible from this machine: the
  CLI stores only an opaque refresh credential in the keychain and holds its
  access token in process memory, and the CLI surface (`download`, `upload`,
  `metadata`) does not expose the relationship or lineage endpoints. A
  current-state confirmation can be made in the platform UI (collection
  "Rapid Fitting — npj paper" → provenance graph) and should show: the
  dataset hub, 20 checkpoint assets, the benchmark table, 8 figure assets,
  the Zenodo-original record, and the revision-experiment archive (the
  archive is inside the collection but intentionally omitted from Fig. S13,
  which diagrams only paper-cited artifacts).

## Current-state confirmation (2026-07-28)

The platform's provenance-graph UI for collection "Rapid Fitting --- npj
paper" was inspected by the first author (screenshots on file): **35 of 35
records, 66 of 66 relationships**, matching Fig. S13's content --- the 20
fitter records, 8 figure assets, benchmark CSV, both dataset records
(live + Zenodo-original), the two sub-collections, and the
revision-experiment archive (in the collection, intentionally not in the
figure). One stray record was also visible: a disconnected
`requirements.txt` asset from CLI testing, not referenced by the paper ---
candidate for deletion before the collection is made public.

## Machine verification (2026-07-29, via dataerai-sdk 0.2.0b27)

The SDK's beta channel (`pip install --pre dataerai-sdk`) ships a REST client
whose auth layer decodes the CLI's Go-keyring credential entry, closing the
earlier bootstrap gap. A live pull through `RestClient` returned, for the
dataset asset: **31 relationships** (21 `derived_from` = 20 checkpoints + the
Zenodo-original link; 10 `analysis_of` = 8 figures + benchmark CSV + revision
archive) and a lineage trace of **42 nodes / 41 edges** (41 runs + the dataset
record), matching the July-16 state exactly. The raw API response is committed
next to this file as `graph_manifest.json` (no credentials inside). The
`%dataerai` notebook magic also connects end-to-end to
"Rapid Fitting of BE-PFM / Rapid Fitting — npj paper"; the daemon-routed
`search.assets` call times out on beta (reported as a platform gap).

## Figure-vs-record mapping

| Figure element | Platform record |
|---|---|
| BE-PFM dataset asset | `6323cd64-933d-4df2-af80-6000d481698e` |
| Zenodo record | metadata-only asset linked by `derived_from`, DOI 10.5281/zenodo.7774788 |
| 20 training-run lineage records | lineage runs sks 298–317 (`trained_on` edges) |
| 20 model checkpoint assets | collection "Trained models — provenance rerun 2026-07-13" (`eb8f2274…`) |
| optimizer-benchmark table | asset `0c7e13ba…` (`analysis_of`) |
| 8 figure assets | collection "Paper figures — main text" (`825336c7…`) |
| collection boundary | "Rapid Fitting — npj paper" (`e00425d0…`) |
