#!/usr/bin/env python3
"""Add the standard Dataerai provenance boundary to source notebooks.

The checked-in Jupyter Book output under ``m3_learning/_build`` is generated
from these notebooks and is intentionally not edited here. Re-running this
script is idempotent: existing managed cells are replaced in place.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGED_IDS = {
    "dataerai-provenance-intro",
    "dataerai-provenance-start",
    "dataerai-provenance-finish-note",
    "dataerai-provenance-finish",
}
IGNORED_NOTEBOOK_DIRECTORIES = {
    ".dataerai-artifacts",
    ".ipynb_checkpoints",
    "_build",
    "executed",
}


def source_notebooks() -> list[Path]:
    """Return authored notebooks, excluding generated Jupyter Book output."""

    notebooks = list((ROOT / "m3_learning").rglob("*.ipynb"))
    notebooks.extend((ROOT / "Testing").rglob("*.ipynb"))
    return sorted(
        path
        for path in notebooks
        if not IGNORED_NOTEBOOK_DIRECTORIES.intersection(path.parts)
    )


def _source_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def _cell(cell_type: str, cell_id: str, source: str) -> dict:
    cell = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": _source_lines(source),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _title(notebook: dict, path: Path) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        for line in text.splitlines():
            match = re.match(r"^#\s+(.+?)\s*$", line)
            if match:
                return match.group(1).replace("`", "")
    return path.stem.replace("_", " ")


def _collection_path(path: Path) -> str:
    relative_parent = path.relative_to(ROOT).parent
    parts = [part.replace("_", " ") for part in relative_parent.parts]
    if parts and parts[0] == "m3 learning":
        parts = parts[1:]
    return " / ".join(["M3 Learning", "Notebook Provenance", *parts])


def _insertion_index(cells: list[dict]) -> int:
    """Insert after environment/package setup, otherwise before first code."""

    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if (
            "Environment setup" in source
            or "pip install m3_learning" in source
            or "pip install m3-learning" in source
        ):
            return index + 1
        return index
    return min(1, len(cells))


def _matching_bracket(text: str, opening_index: int) -> int:
    opening = text[opening_index]
    closing = {"[": "]", "{": "}"}[opening]
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed JSON {opening!r} at offset {opening_index}")


def _cells_array_bounds(text: str) -> tuple[int, int]:
    key_index = text.index('"cells"')
    opening_index = text.index("[", key_index)
    return opening_index, _matching_bracket(text, opening_index)


def _cell_spans(text: str, opening_index: int, closing_index: int):
    """Return exact source spans for top-level cell objects in the array."""

    spans = []
    index = opening_index + 1
    while index < closing_index:
        char = text[index]
        if char.isspace() or char == ",":
            index += 1
            continue
        if char != "{":
            raise ValueError(f"expected notebook cell object at offset {index}")
        end = _matching_bracket(text, index) + 1
        spans.append((index, end))
        index = end
    return spans


def _serialized_cells(cells: list[dict]) -> str:
    """Serialize cells at the repository's one-space JSON indentation."""

    items = []
    for cell in cells:
        item = json.dumps(cell, indent=1, ensure_ascii=False)
        items.append(item.replace("\n", "\n  "))
    return ",\n  ".join(items)


def _insert_cells(
    text: str,
    insertion_index: int,
    opening_cells: list[dict],
    closing_cells: list[dict],
) -> str:
    """Splice managed cells without reserializing existing notebook content."""

    array_open, array_close = _cells_array_bounds(text)
    spans = _cell_spans(text, array_open, array_close)
    opening = _serialized_cells(opening_cells)
    closing = _serialized_cells(closing_cells)

    if not spans:
        return text[: array_open + 1] + "\n  " + opening + ",\n  " + closing + "\n " + text[array_close:]

    insertion_index = min(insertion_index, len(spans))
    if insertion_index == len(spans):
        opening_position = spans[-1][1]
        text = (
            text[:opening_position]
            + ",\n  "
            + opening
            + text[opening_position:]
        )
    else:
        opening_position = spans[insertion_index][0]
        text = (
            text[:opening_position]
            + opening
            + ",\n  "
            + text[opening_position:]
        )

    # Recalculate after inserting the opening boundary, then append the finish
    # boundary directly after the last authored cell.
    array_open, array_close = _cells_array_bounds(text)
    spans = _cell_spans(text, array_open, array_close)
    closing_position = spans[-1][1]
    return (
        text[:closing_position]
        + ",\n  "
        + closing
        + text[closing_position:]
    )


def _replace_managed_cells(text: str, replacements: dict[str, dict]) -> str:
    """Replace managed cell objects without reserializing authored cells."""

    array_open, array_close = _cells_array_bounds(text)
    spans = _cell_spans(text, array_open, array_close)
    seen = set()
    for start, end in reversed(spans):
        cell = json.loads(text[start:end])
        cell_id = cell.get("id")
        if cell_id not in replacements:
            continue
        text = text[:start] + _serialized_cells([replacements[cell_id]]) + text[end:]
        seen.add(cell_id)
    missing = set(replacements) - seen
    if missing:
        raise ValueError(f"managed cells missing from notebook: {sorted(missing)}")
    return text


def _managed_cells(path: Path, notebook: dict) -> tuple[list[dict], list[dict]]:
    title = _title(notebook, path)
    default_collection = _collection_path(path)
    relative_path = path.relative_to(ROOT).as_posix()

    intro = _cell(
        "markdown",
        "dataerai-provenance-intro",
        "## Dataerai notebook and neural-network provenance\n\n"
        "Authenticate once with `dataerai auth login --device --client-id "
        "dataerai-mobile --server https://beta.dataerai.com`. The next cell "
        "starts `%dataerai --trace` and the M3 artifact publisher. Together "
        "they preserve every execution; upload the source notebook; reuse raw "
        "datasets; publish figures, movies, changed HDF5/CSV data, PyTorch "
        "checkpoints, loss histories, and manifests; and link those products "
        "to the notebook run and source data. Output folders are mirrored as "
        "nested Dataerai collections. Set "
        "`DATAERAI_DESTINATION_COLLECTION_PATH` before launching Jupyter to "
        "override the default destination.\n",
    )
    start = _cell(
        "code",
        "dataerai-provenance-start",
        "import os as _dataerai_os\n"
        "import subprocess as _dataerai_subprocess\n"
        "import sys as _dataerai_sys\n\n"
        "_dataerai_subprocess.run(\n"
        "    [\n"
        "        _dataerai_sys.executable,\n"
        "        \"-m\",\n"
        "        \"pip\",\n"
        "        \"install\",\n"
        "        \"--quiet\",\n"
        "        \"--upgrade\",\n"
        "        \"--pre\",\n"
        "        \"dataerai-cli-beta==0.1.54\",\n"
        "        \"dataerai-sdk[notebook,nn-pytorch]==0.2.0b52\",\n"
        "    ],\n"
        "    check=True,\n"
        ")\n\n"
        "try:\n"
        "    from m3_learning.artifacts import "
        "start_dataerai_artifact_publishing\n"
        "except ImportError:\n"
        "    _dataerai_subprocess.run(\n"
        "        [\n"
        "            _dataerai_sys.executable,\n"
        "            \"-m\",\n"
        "            \"pip\",\n"
        "            \"install\",\n"
        "            \"--upgrade\",\n"
        "            \"--no-deps\",\n"
        "            \"git+https://github.com/m3-learning/\"\n"
        "            \"m3_learning.git@codex/dataerai-notebook-training-provenance\"\n"
        "            \"#subdirectory=m3_learning\",\n"
        "        ],\n"
        "        check=True,\n"
        "    )\n"
        "    from m3_learning.artifacts import "
        "start_dataerai_artifact_publishing\n\n"
        f"DATAERAI_NOTEBOOK = {path.name!r}\n"
        f"DATAERAI_NOTEBOOK_TITLE = {title!r}\n"
        "DATAERAI_DESTINATION_COLLECTION_PATH = _dataerai_os.environ.get(\n"
        "    \"DATAERAI_DESTINATION_COLLECTION_PATH\",\n"
        f"    {default_collection!r},\n"
        ")\n\n"
        "DATAERAI_PROVENANCE_ROOT_PATH = (\n"
        f"    f\"{{DATAERAI_DESTINATION_COLLECTION_PATH}} / {path.stem}\"\n"
        ")\n"
        "DATAERAI_EXECUTION_COLLECTION_PATH = (\n"
        "    f\"{DATAERAI_PROVENANCE_ROOT_PATH} / Executions\"\n"
        ")\n\n"
        "%load_ext dataerai.magics\n"
        "%dataerai --request-timeout 120 --trace "
        "--notebook \"$DATAERAI_NOTEBOOK\" "
        "--title \"$DATAERAI_NOTEBOOK_TITLE\" "
        "\"$DATAERAI_EXECUTION_COLLECTION_PATH\"\n\n"
        "_dataerai_trace = dataerai_session._trace\n"
        "_dataerai_trace.title = (\n"
        "    f\"{DATAERAI_NOTEBOOK_TITLE} · run {_dataerai_trace.run_id}\"\n"
        ")\n"
        "_dataerai_os.environ[\"DATAERAI_NOTEBOOK_TRACE_RUN_ID\"] = "
        "dataerai_session.trace_run_id\n"
        "_dataerai_os.environ[\"DATAERAI_NOTEBOOK_COLLECTION_PATH\"] = "
        "dataerai_session.collection_path\n"
        "dataerai_artifacts = start_dataerai_artifact_publishing(\n"
        "    dataerai_session,\n"
        "    DATAERAI_NOTEBOOK,\n"
        "    shell=get_ipython(),\n"
        "    provenance_root_path=DATAERAI_PROVENANCE_ROOT_PATH,\n"
        ")\n",
    )
    finish_note = _cell(
        "markdown",
        "dataerai-provenance-finish-note",
        "## Finish the Dataerai provenance record\n\n"
        "This cell first uploads captured figures, movies, changed HDF5/CSV "
        "files, and model artifacts. It then publishes this run's distinct "
        "notebook execution log and its `records_telemetry` relationships.\n",
    )
    finish = _cell(
        "code",
        "dataerai-provenance-finish",
        "dataerai_artifact_result = dataerai_artifacts.finish()\n"
        "%dataerai --finish\n",
    )
    for cell in (intro, start, finish_note, finish):
        cell["metadata"]["dataerai"] = {
            "managed": True,
            "source_notebook": relative_path,
        }
    return [intro, start], [finish_note, finish]


def _source_text(path: Path, base_ref: str | None) -> str:
    if base_ref is None:
        return path.read_text(encoding="utf-8")
    relative_path = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def update_notebook(path: Path, *, base_ref: str | None = None) -> bool:
    original = path.read_text(encoding="utf-8")
    source = _source_text(path, base_ref)
    if all(f'"id": "{cell_id}"' in source for cell_id in MANAGED_IDS):
        notebook = json.loads(source)
        opening, closing = _managed_cells(path, notebook)
        replacements = {cell["id"]: cell for cell in (*opening, *closing)}
        target = _replace_managed_cells(source, replacements)
    elif any(f'"id": "{cell_id}"' in source for cell_id in MANAGED_IDS):
        raise ValueError(f"{path} has a partial Dataerai managed-cell boundary")
    else:
        notebook = json.loads(source)
        cells = notebook.get("cells", [])
        opening, closing = _managed_cells(path, notebook)
        target = _insert_cells(
            source,
            _insertion_index(cells),
            opening,
            closing,
        )
        json.loads(target)  # Validate the textual splice before writing.

    changed = target != original
    if changed:
        path.write_text(target, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        help="rebuild managed cells over each notebook from this Git ref",
    )
    args = parser.parse_args()
    notebooks = source_notebooks()
    changed = [
        path for path in notebooks if update_notebook(path, base_ref=args.base_ref)
    ]
    print(f"Updated {len(changed)} of {len(notebooks)} source notebooks.")
    for path in changed:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
