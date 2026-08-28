# m3_learning
Home of Tutorials and Data Sets for M3-Learning Research Group

Make sure you install the m3_learning package `pip install m3-learning`

## Dataerai provenance

All 31 authored notebooks start and finish a Dataerai execution trace with the
`%dataerai` IPython magic. Neural-network runs made through the instrumented
`m3_learning` fitters inherit the same trace ID, linking training lineage to the
cell-level execution record. Authenticate once before running a notebook:

```bash
python -m pip install --pre dataerai-cli-beta 'dataerai-sdk[ml,notebook]>=0.2.0b1,<0.3'
dataerai auth login --device --server https://beta.dataerai.com
```

Set `DATAERAI_DESTINATION_COLLECTION_PATH` to override a notebook's default
`M3 Learning / Notebook Provenance / ...` destination. Source notebook
provenance cells are maintained by
`python tools/update_dataerai_notebook_provenance.py`; generated notebooks under
`m3_learning/_build` are refreshed by the Jupyter Book build instead.
