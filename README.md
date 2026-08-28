# m3_learning
Home of Tutorials and Data Sets for M3-Learning Research Group

Make sure you install the m3_learning package `pip install m3-learning`

## Dataerai provenance

All 31 authored notebooks start a Dataerai execution trace and an M3 artifact
publisher. Each run keeps one JSON execution log while promoting source data,
rich figures, changed HDF5/CSV data, checkpoints, loss histories, and model
manifests to first-class assets. Stable source titles are reused across
notebooks; stable derived titles create content versions. Neural-network runs
inherit the same trace ID. Authenticate once before running a notebook:

```bash
python -m pip install --pre dataerai-cli-beta==0.1.54 'dataerai-sdk[notebook]==0.2.0b52'
dataerai auth login --device --client-id dataerai-mobile --server https://beta.dataerai.com
```

Set `DATAERAI_DESTINATION_COLLECTION_PATH` to override a notebook's default
`M3 Learning / Notebook Provenance / ...` destination. Source notebook
provenance cells are maintained by
`python tools/update_dataerai_notebook_provenance.py`; generated notebooks under
`m3_learning/_build` are refreshed by the Jupyter Book build instead.

The publisher treats shallow HDF5/CSV files present when tracing starts as
source data, recognizes newly downloaded source archives, and recognizes files
with `raw`, `source`, `standard`, or `data_file` in their names. Set
`DATAERAI_RAW_DATA_PATHS` (colon-separated on Linux/macOS) to declare additional
source paths. Publication is strict by default: a partial upload makes the final cell fail. Set
`DATAERAI_ARTIFACT_STRICT=0` only when best-effort publication is intentional.
Set `DATAERAI_RAW_DATA_ASSET_ID` to force reuse of an already-published source
asset instead of resolving it by the stable title.
