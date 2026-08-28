import base64
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "src" / "m3_learning" / "artifacts.py"
_SPEC = importlib.util.spec_from_file_location("m3_learning_artifacts_test", _MODULE_PATH)
artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = artifacts
_SPEC.loader.exec_module(artifacts)
DataeraiArtifactPublisher = artifacts.DataeraiArtifactPublisher


@pytest.fixture(autouse=True)
def _isolated_dataerai_environment(monkeypatch):
    for name in (
        "DATAERAI_RAW_DATA_ASSET_ID",
        "DATAERAI_DATASET_ASSET_ID",
        "DATAERAI_RAW_DATA_PATHS",
        "DATAERAI_LOG_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)


@dataclass
class _Asset:
    asset_id: str
    title: str
    has_content: bool = True


class _Events:
    def __init__(self):
        self.registered = []

    def register(self, event, callback):
        self.registered.append((event, callback))

    def unregister(self, event, callback):
        self.registered.remove((event, callback))


class _DisplayPublisher:
    def __init__(self):
        self.hooks = []

    def register_hook(self, callback):
        self.hooks.append(callback)

    def unregister_hook(self, callback):
        self.hooks.remove(callback)

    def publish(self, data):
        message = {"content": {"data": data}}
        for hook in list(self.hooks):
            message = hook(message)


class _Session:
    def __init__(self, *, existing=None, run_id="trace-123"):
        self.existing = list(existing or [])
        self.uploads = []
        self.relationships = []
        self.collection_id = "collection-1"
        self.trace_run_id = run_id
        self._trace = SimpleNamespace(run_id=run_id, cells=[])

    def find_assets(self, query, *, title=None, predicate=None):
        matches = [asset for asset in self.existing if title in (None, asset.title)]
        return [asset for asset in matches if predicate is None or predicate(asset)]

    def upload(self, local_path, *, title=None, **kwargs):
        path = Path(local_path)
        upload = {
            "path": path,
            "content": path.read_bytes(),
            "title": title,
            **kwargs,
        }
        self.uploads.append(upload)
        return SimpleNamespace(asset_id=f"asset-{len(self.uploads)}")

    def create_relationship(self, source, target, relation, **kwargs):
        self.relationships.append((source, target, relation, kwargs))
        return SimpleNamespace(id=f"relationship-{len(self.relationships)}")


def _publisher(tmp_path, *, session=None):
    session = session or _Session()
    shell = SimpleNamespace(
        events=_Events(),
        display_pub=_DisplayPublisher(),
        execution_count=1,
    )
    return DataeraiArtifactPublisher(
        session=session,
        notebook_name="2_Pytorch_SHO_Fitter.ipynb",
        root=tmp_path,
        shell=shell,
    )


def test_raw_dataset_is_uploaded_once_then_reused_by_exact_stable_title(
    tmp_path, monkeypatch
):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw-hdf5")

    first_session = _Session()
    first = _publisher(tmp_path, session=first_session).start()
    first_result = first.finish()

    raw_upload = next(item for item in first_session.uploads if item["path"] == raw)
    assert raw_upload["record_type"] == "dataset"
    assert raw_upload["metadata"]["component"] == "raw-dataset"
    assert first_result.raw_dataset_asset_ids == ("asset-1",)

    existing = _Asset(asset_id="raw-existing", title=raw_upload["title"])
    monkeypatch.delenv("DATAERAI_RAW_DATA_ASSET_ID")
    monkeypatch.delenv("DATAERAI_DATASET_ASSET_ID")
    second_session = _Session(existing=[existing], run_id="trace-456")
    second = _publisher(tmp_path, session=second_session).start()
    second_result = second.finish()

    assert all(item["path"] != raw for item in second_session.uploads)
    assert second_result.raw_dataset_asset_ids == ("raw-existing",)


def test_raw_dataset_downloaded_after_trace_start_is_published_before_later_cells(tmp_path):
    session = _Session()
    publisher = _publisher(tmp_path, session=session).start()
    raw = tmp_path / "Data" / "PZT_2080_raw_data.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"downloaded-raw")

    publisher._post_run_cell(SimpleNamespace())

    upload = next(item for item in session.uploads if item["path"] == raw)
    assert upload["metadata"]["component"] == "raw-dataset"
    assert publisher.raw_dataset_asset_ids == ("asset-1",)


def test_contentless_raw_dataset_placeholder_is_retried(tmp_path):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw-hdf5")
    title = "M3 source dataset · Data/data_raw.h5"
    session = _Session(
        existing=[
            _Asset(
                asset_id="failed-placeholder",
                title=title,
                has_content=False,
            )
        ]
    )

    result = _publisher(tmp_path, session=session).start().finish()

    raw_upload = next(item for item in session.uploads if item["path"] == raw)
    assert raw_upload["title"] == title
    assert result.raw_dataset_asset_ids == ("asset-1",)


def test_transient_upload_failure_is_retried(tmp_path, monkeypatch):
    class _TransientUploadSession(_Session):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def upload(self, local_path, *, title=None, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("POST transfers: HTTP 504 Gateway Timeout")
            return super().upload(local_path, title=title, **kwargs)

    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    figure = tmp_path / "Figures" / "frame.png"
    publisher = _publisher(tmp_path, session=_TransientUploadSession()).start()
    figure.parent.mkdir()
    figure.write_bytes(b"png")

    result = publisher.finish()

    assert publisher.session.attempts == 2
    assert len(result.analysis_asset_ids) == 1
    assert result.errors == ()


def test_downloaded_source_archive_is_published_as_raw_data(tmp_path):
    publisher = _publisher(tmp_path).start()
    archive = tmp_path / "Datasets" / "AFM.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"source-archive")

    publisher._post_run_cell(SimpleNamespace())

    upload = next(item for item in publisher.session.uploads if item["path"] == archive)
    assert upload["record_type"] == "dataset"
    assert upload["metadata"]["component"] == "raw-dataset"


def test_unrelated_existing_csv_is_not_misclassified_as_raw_data(tmp_path):
    benchmark = tmp_path / "record_from_datafed.csv"
    benchmark.write_text("optimizer,loss\nAdam,0.1\n", encoding="utf-8")
    publisher = _publisher(tmp_path).start()

    result = publisher.finish()

    assert result.raw_dataset_asset_ids == ()
    assert all(item["path"] != benchmark for item in publisher.session.uploads)


def test_each_rich_figure_gets_a_deterministic_analysis_asset(tmp_path):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw")
    session = _Session()
    publisher = _publisher(tmp_path, session=session).start()
    png = base64.b64encode(b"png-bytes").decode("ascii")
    publisher.shell.execution_count = 7
    publisher.shell.display_pub.publish(
        {"image/png": png, "text/plain": "<Figure size>"}
    )
    publisher.shell.display_pub.publish(
        {"image/svg+xml": "<svg><circle/></svg>"}
    )

    publisher.capture_completed_cells()
    result = publisher.finish()

    figure_uploads = [u for u in session.uploads if u["record_type"] == "analysis"]
    assert [u["path"].name for u in figure_uploads] == [
        "cell-0007-display-01.png",
        "cell-0007-display-02.svg",
    ]
    assert all(u["metadata"]["notebook_run_id"] == "trace-123" for u in figure_uploads)
    assert len(result.analysis_asset_ids) == 2
    assert sum(
        relation == "analysis_of"
        for _, _, relation, _ in session.relationships
    ) == 2


def test_direct_display_capture_preserves_figures_larger_than_trace_limit(tmp_path):
    publisher = _publisher(tmp_path).start()
    content = b"large-png" * 150_000
    publisher.shell.execution_count = 12

    publisher.shell.display_pub.publish(
        {"image/png": base64.b64encode(content).decode("ascii")}
    )
    publisher.finish()

    upload = next(
        item
        for item in publisher.session.uploads
        if item["metadata"]["component"] == "notebook-figure"
    )
    assert upload["path"].name == "cell-0012-display-01.png"
    assert upload["content"] == content


def test_changed_hdf5_and_csv_use_stable_titles_for_content_versioning(tmp_path):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw")
    h5 = tmp_path / "derived.h5"
    csv = tmp_path / "metrics.csv"
    h5.write_bytes(b"before")
    csv.write_text("loss\n1\n", encoding="utf-8")
    publisher = _publisher(tmp_path).start()

    h5.write_bytes(b"after")
    csv.write_text("loss\n1\n0.5\n", encoding="utf-8")
    publisher.finish()

    uploads = [u for u in publisher.session.uploads if u["metadata"]["component"] == "derived-data"]
    assert {u["path"] for u in uploads} == {h5, csv}
    assert all(u["record_type"] == "dataset" for u in uploads)
    assert all("trace-123" not in u["title"] for u in uploads)
    assert sum(
        relation == "derived_from"
        for _, _, relation, _ in publisher.session.relationships
    ) >= 2


def test_model_checkpoint_loss_and_manifest_are_model_assets_with_lineage(tmp_path):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw")
    model = tmp_path / "models" / "sho.pth"
    loss = tmp_path / "models" / "sho-loss.txt"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")
    loss.write_text("1.0\n0.5\n", encoding="utf-8")
    publisher = _publisher(tmp_path).start()
    publisher.queue_model_artifacts(
        model_path=model,
        loss_path=loss,
        params={"optimizer": "Adam", "epochs": 2},
        metrics={"train_loss": 0.5},
        lineage_run_id="training-run-1",
    )

    result = publisher.finish()

    model_uploads = [u for u in publisher.session.uploads if u["record_type"] == "model"]
    assert {u["metadata"]["component"] for u in model_uploads} == {
        "model-checkpoint",
        "training-loss",
        "model-manifest",
    }
    manifest_upload = next(u for u in model_uploads if u["metadata"]["component"] == "model-manifest")
    manifest = json.loads(manifest_upload["content"])
    assert manifest["schema"] == "m3-learning.dataerai-model-manifest.v1"
    assert manifest["notebook_run_id"] == "trace-123"
    assert manifest["lineage_run_id"] == "training-run-1"
    assert manifest["params"]["optimizer"] == "Adam"
    assert manifest["metrics"]["train_loss"] == 0.5
    assert len(result.model_asset_ids) == 3
    relations = {(source, target, rel) for source, target, rel, _ in publisher.session.relationships}
    assert any(rel == "trained_on" for _, _, rel in relations)
    assert any(rel == "describes" for _, _, rel in relations)


def test_every_uploaded_product_uses_the_traced_session_for_execution_linking(tmp_path):
    raw = tmp_path / "Data" / "data_raw.h5"
    raw.parent.mkdir()
    raw.write_bytes(b"raw")
    session = _Session()
    publisher = _publisher(tmp_path, session=session).start()
    publisher.shell.execution_count = 3
    publisher.shell.display_pub.publish(
        {"image/png": base64.b64encode(b"figure").decode("ascii")}
    )

    publisher.finish()

    assert session.uploads
    assert all(upload["metadata"]["notebook_run_id"] == "trace-123" for upload in session.uploads)


def test_publisher_unregisters_its_notebook_hook_on_finish(tmp_path):
    publisher = _publisher(tmp_path).start()
    assert publisher.shell.events.registered
    assert publisher.shell.display_pub.hooks

    publisher.finish()

    assert publisher.shell.events.registered == []
    assert publisher.shell.display_pub.hooks == []


@pytest.mark.parametrize(
    "relative_path",
    (
        "nn/Fitter1D/Fitter1D.py",
        "be/nn.py",
    ),
)
def test_instrumented_fitters_queue_saved_model_outputs(relative_path):
    source = (
        Path(__file__).parents[1] / "src" / "m3_learning" / relative_path
    ).read_text(encoding="utf-8")

    lineage_position = source.index("lineage = log_dataerai_training_run(")
    queue_position = source.index("queue_dataerai_model_artifacts(")
    queue_block = source[queue_position : queue_position + 400]

    assert queue_position > lineage_position
    assert "model_path=final_model_path" in queue_block
    assert "loss_path=training_loss_path" in queue_block
    assert "params=params" in queue_block
    assert "metrics=metrics" in queue_block
    assert "lineage_run_id=lineage.run_id" in queue_block
