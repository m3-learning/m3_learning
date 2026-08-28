# Dataerai notebook artifact publishing acceptance plan

Automated tests use a contract-faithful fake `NotebookSession` to verify exact
titles, record types, metadata, generated manifest content, and relationships
without uploading research data during CI.

Manual cloud acceptance is intentionally deferred until the branch is run with
the user's authenticated beta account and source dataset. Run the Rapid Fitting
notebooks in order with `QUICK_RUN=True` and verify:

1. the first notebook uploads one source HDF5 asset;
2. later notebooks reuse that asset ID;
3. every displayed figure appears as an analysis asset;
4. changed HDF5/CSV titles gain content versions;
5. the neural-network notebook publishes checkpoint, loss, and manifest model
   assets plus the documented relationships; and
6. each run has one UUID-qualified execution log whose product count matches
   its promoted artifacts; rerunning a notebook creates another execution
   asset instead of replacing the earlier one.
