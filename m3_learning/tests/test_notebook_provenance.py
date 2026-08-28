import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANAGED_IDS = (
    "dataerai-provenance-intro",
    "dataerai-provenance-start",
    "dataerai-provenance-finish-note",
    "dataerai-provenance-finish",
)


def _source_notebooks():
    notebooks = list((ROOT / "m3_learning").rglob("*.ipynb"))
    notebooks.extend((ROOT / "Testing").rglob("*.ipynb"))
    return sorted(path for path in notebooks if "_build" not in path.parts)


def test_every_source_notebook_has_one_managed_provenance_boundary():
    notebooks = _source_notebooks()

    assert len(notebooks) == 31
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        ids = [cell.get("id") for cell in cells]
        for managed_id in MANAGED_IDS:
            assert ids.count(managed_id) == 1, f"{path}: {managed_id}"

        start = cells[ids.index("dataerai-provenance-start")]
        start_source = "".join(start["source"])
        assert "dataerai-cli-beta" in start_source
        assert "dataerai-sdk[ml,notebook]>=0.2.0b1,<0.3" in start_source
        assert "%load_ext dataerai.magics" in start_source
        assert "%dataerai --request-timeout 120 --trace" in start_source
        assert "DATAERAI_NOTEBOOK_TRACE_RUN_ID" in start_source
        assert "dataerai_session.trace_run_id" in start_source

        finish_index = ids.index("dataerai-provenance-finish")
        finish = cells[finish_index]
        assert "".join(finish["source"]).strip() == "%dataerai --finish"
        assert not any(
            cell.get("cell_type") == "code" for cell in cells[finish_index + 1 :]
        ), f"{path}: code appears after the trace is finished"


def test_generated_notebooks_are_not_managed_by_source_rewriter():
    generated = sorted((ROOT / "m3_learning" / "_build").rglob("*.ipynb"))

    assert generated
    for path in generated:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        ids = {cell.get("id") for cell in notebook.get("cells", [])}
        assert ids.isdisjoint(MANAGED_IDS), path
