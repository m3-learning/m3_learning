"""Publish notebook data, figures, and model artifacts to Dataerai.

The Dataerai notebook trace intentionally records rich output without scanning
or uploading arbitrary files. This module is the explicit M3 publication
policy layered on top: it promotes scientific display outputs and selected
changed files through the active ``NotebookSession`` so the trace can link the
resulting assets to the one execution-log asset for the run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DATA_EXTENSIONS = {".h5", ".hdf5", ".csv"}
_RAW_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".xz"}
_MODEL_EXTENSIONS = {".pth", ".pt"}
_FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
_MOVIE_EXTENSIONS = {".gif", ".mp4", ".webm"}
_ANALYSIS_EXTENSIONS = _FIGURE_EXTENSIONS | _MOVIE_EXTENSIONS
_WATCHED_EXTENSIONS = (
    _DATA_EXTENSIONS
    | _RAW_ARCHIVE_EXTENSIONS
    | _MODEL_EXTENSIONS
    | _ANALYSIS_EXTENSIONS
)
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "_build",
    "node_modules",
}
_PROVENANCE_COLLECTIONS = (
    "Notebooks",
    "Executions",
    "Data / Raw",
    "Data / Derived",
    "Figures",
    "Movies",
    "Models",
    "Manifests",
)
_ACTIVE_PUBLISHER: DataeraiArtifactPublisher | None = None


@dataclass(frozen=True)
class _FileState:
    size: int
    modified_ns: int


@dataclass
class _PendingArtifact:
    path: Path
    title: str
    component: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingModel:
    model_path: Path
    loss_path: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    lineage_run_id: str | None = None


@dataclass
class ArtifactPublishResult:
    """Assets published or reused by one notebook execution."""

    notebook_asset_id: str | None = None
    raw_dataset_asset_ids: tuple[str, ...] = ()
    analysis_asset_ids: tuple[str, ...] = ()
    dataset_asset_ids: tuple[str, ...] = ()
    model_asset_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class DataeraiArtifactPublisher:
    """Promote one traced notebook run's scientific artifacts to Dataerai."""

    session: Any
    notebook_name: str
    root: Path | str = field(default_factory=Path.cwd)
    shell: Any = None
    strict: bool | None = None
    provenance_root_path: str | None = None

    _baseline: dict[Path, _FileState] = field(default_factory=dict, init=False)
    _raw_states: dict[Path, _FileState] = field(default_factory=dict, init=False)
    _raw_by_path: dict[Path, str] = field(default_factory=dict, init=False)
    _analysis_queue: dict[Path, _PendingArtifact] = field(
        default_factory=dict, init=False
    )
    _model_queue: dict[Path, _PendingModel] = field(default_factory=dict, init=False)
    _processed_cell_count: int = field(default=0, init=False)
    _registered: bool = field(default=False, init=False)
    _display_hook_registered: bool = field(default=False, init=False)
    _display_counts: dict[int, int] = field(default_factory=dict, init=False)
    _finished: bool = field(default=False, init=False)
    _result: ArtifactPublishResult | None = field(default=None, init=False)
    _analysis_asset_ids: list[str] = field(default_factory=list, init=False)
    _dataset_asset_ids: list[str] = field(default_factory=list, init=False)
    _model_asset_ids: list[str] = field(default_factory=list, init=False)
    _collection_ids: dict[str, str] = field(default_factory=dict, init=False)
    _published_paths: set[Path] = field(default_factory=set, init=False)
    _ignored_model_paths: set[Path] = field(default_factory=set, init=False)
    _notebook_asset_id: str | None = field(default=None, init=False)
    _errors: list[str] = field(default_factory=list, init=False)
    _raw_errors: dict[Path, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if self.provenance_root_path is None:
            session_path = str(getattr(self.session, "collection_path", "")).strip()
            suffix = " / Executions"
            self.provenance_root_path = (
                session_path[: -len(suffix)]
                if session_path.endswith(suffix)
                else session_path
            )
        if self.strict is None:
            self.strict = _env_bool("DATAERAI_ARTIFACT_STRICT", default=True)
        self.artifact_directory = (
            self.root
            / ".dataerai-artifacts"
            / Path(self.notebook_name).stem
            / self.run_id
        )

    @property
    def run_id(self) -> str:
        run_id = getattr(self.session, "trace_run_id", None)
        if not run_id:
            trace = getattr(self.session, "_trace", None)
            run_id = getattr(trace, "run_id", None)
        if not run_id:
            raise RuntimeError("artifact publishing requires an active notebook trace")
        return str(run_id)

    @property
    def raw_dataset_asset_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._raw_by_path.values()))

    @property
    def notebook_asset_id(self) -> str | None:
        return self._notebook_asset_id

    def start(self) -> DataeraiArtifactPublisher:
        """Snapshot inputs, reuse/upload source data, and observe later cells."""

        if self._registered or self._finished:
            return self
        self._install_trace_upload_retry()
        self._install_trace_relationship_retry()
        self._ensure_provenance_collections()
        self._publish_source_notebook()
        self._baseline = self._snapshot()
        for path in sorted(self._baseline, key=str):
            if (
                path.suffix.lower() in _DATA_EXTENSIONS | _RAW_ARCHIVE_EXTENSIONS
                and self._is_source_input(path)
            ):
                self._ensure_raw_dataset(path)
        events = getattr(self.shell, "events", None)
        if events is not None:
            events.register("post_run_cell", self._post_run_cell)
            self._registered = True
        display_pub = getattr(self.shell, "display_pub", None)
        if display_pub is not None and hasattr(display_pub, "register_hook"):
            display_pub.register_hook(self._capture_display)
            self._display_hook_registered = True
        os.environ.setdefault("DATAERAI_LOG_PROVENANCE", "1")
        return self

    def _ensure_provenance_collections(self) -> None:
        """Create the stable notebook provenance tree once per session."""

        client = getattr(self.session, "client", None)
        ensure = getattr(client, "ensure_collection_path", None)
        root = str(self.provenance_root_path or "").strip()
        if not root or not callable(ensure):
            return
        for relative in _PROVENANCE_COLLECTIONS:
            path = f"{root} / {relative}"
            destination = self._ensure_collection_path(path)
            self._collection_ids[relative] = str(destination.collection_id)

    def _install_trace_upload_retry(self) -> None:
        """Retry the SDK's final execution-log upload on transient failures.

        The notebook magic publishes its JSON trace after the finish cell has
        completed.  Wrapping the session's bound upload here keeps that final
        write inside the SDK's signed notebook-tracing flow while making a
        temporary gateway or transfer-verification failure recoverable.
        """

        original = getattr(self.session, "_upload_bound", None)
        if not callable(original) or getattr(original, "_m3_retrying_upload", False):
            return

        def retrying_upload(
            local_path: str, *, title: str | None = None, **kwargs: Any
        ) -> Any:
            attempts = max(1, int(os.environ.get("DATAERAI_UPLOAD_ATTEMPTS", "3")))
            for attempt in range(1, attempts + 1):
                try:
                    return original(local_path, title=title, **kwargs)
                except Exception as exc:  # noqa: BLE001 - SDK/network boundary
                    if attempt == attempts or not _is_transient_upload_error(exc):
                        raise
                    time.sleep(min(2 ** (attempt - 1), _max_retry_backoff_seconds()))
            raise AssertionError("unreachable")

        retrying_upload._m3_retrying_upload = True  # type: ignore[attr-defined]
        self.session._upload_bound = retrying_upload

    def _install_trace_relationship_retry(self) -> None:
        """Retry signed execution-to-product relationship writes."""

        client = getattr(self.session, "client", None)
        original = getattr(client, "create_relationship", None)
        if not callable(original) or getattr(
            original, "_m3_retrying_relationship", False
        ):
            return

        def retrying_relationship(*args: Any, **kwargs: Any) -> Any:
            attempts = max(1, int(os.environ.get("DATAERAI_UPLOAD_ATTEMPTS", "3")))
            for attempt in range(1, attempts + 1):
                try:
                    return original(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - SDK/network boundary
                    code = str(getattr(exc, "code", ""))
                    if "EXISTS" in code.upper() or "EXISTS" in str(exc).upper():
                        # Already in the desired state - _relationship() treats this
                        # as success, so the wrapper must not diverge and re-raise.
                        return None
                    if attempt == attempts or not _is_transient_upload_error(exc):
                        raise
                    time.sleep(min(2 ** (attempt - 1), _max_retry_backoff_seconds()))
            raise AssertionError("unreachable")

        retrying_relationship._m3_retrying_relationship = True  # type: ignore[attr-defined]
        client.create_relationship = retrying_relationship

    def _ensure_collection_path(self, path: str) -> Any:
        """Resolve/create one collection with bounded transient retries."""

        ensure = self.session.client.ensure_collection_path
        attempts = max(1, int(os.environ.get("DATAERAI_UPLOAD_ATTEMPTS", "3")))
        for attempt in range(1, attempts + 1):
            try:
                return ensure(path, create_project=False)
            except Exception as exc:  # noqa: BLE001 - SDK/network boundary
                if attempt == attempts or not _is_transient_upload_error(exc):
                    raise
                time.sleep(min(2 ** (attempt - 1), _max_retry_backoff_seconds()))
        raise AssertionError("unreachable")

    def _publish_source_notebook(self) -> None:
        """Upload the raw source notebook as a first-class run input."""

        path = (self.root / self.notebook_name).resolve()
        if not path.is_file():
            self._errors.append(f"source notebook does not exist: {path}")
            return
        checksum = _sha256(path)
        try:
            uploaded = self._upload(
                path,
                title=_stable_title(
                    "M3 source notebook",
                    f"{self.notebook_name} · {checksum[:12]}",
                ),
                record_type="jupyter_notebook",
                component="source-notebook",
                metadata={
                    "source_path": self.notebook_name,
                    "sha256": checksum,
                    "size_bytes": path.stat().st_size,
                    "immutable_run_snapshot": True,
                },
                tags=["m3-learning", "source-notebook", "provenance-input"],
            )
            self._notebook_asset_id = str(uploaded.asset_id)
            os.environ["DATAERAI_NOTEBOOK_ASSET_ID"] = self._notebook_asset_id
        except Exception as exc:  # noqa: BLE001 - SDK/network boundary
            self._errors.append(f"could not publish source notebook {path}: {exc}")

    def _collection_relative_path(self, path: Path, component: str) -> str:
        """Map an artifact and its local folder into the provenance tree."""

        relative_parent = Path(self._relative(path)).parent
        parts = [part for part in relative_parent.parts if part not in ("", ".")]
        lowered = [part.casefold() for part in parts]
        if component == "source-notebook":
            return "Notebooks"
        if component.startswith("raw-dataset"):
            category = "Data / Raw"
            if component == "raw-dataset-part":
                parts = ["Parts", path.parent.name]
            elif component == "raw-dataset-manifest":
                parts = []
            elif lowered and lowered[0] in {"data", "datasets"}:
                parts = parts[1:]
        elif component in {"derived-data", "derived-data-manifest"}:
            category = "Data / Derived"
            if "derived-data" in lowered:
                parts = parts[lowered.index("derived-data") + 1 :]
                lowered = [part.casefold() for part in parts]
            if lowered and lowered[0] in {"data", "datasets"}:
                parts = parts[1:]
        elif component in {"saved-movie"}:
            category = "Movies"
            if lowered and lowered[0] == "movies":
                parts = parts[1:]
        elif component.startswith("nn-provenance") or component == "model-manifest":
            category = "Manifests"
            parts = _strip_artifact_category(parts, {"models", "trained models"})
        elif component.startswith("nn-") or component in {
            "model-checkpoint",
            "training-loss",
        }:
            category = "Models"
            parts = _strip_artifact_category(parts, {"models", "trained models"})
        else:
            category = "Figures"
            if component == "notebook-figure":
                parts = ["Rich Outputs"]
            elif lowered and lowered[0] == "figures":
                parts = parts[1:]
        clean = [_collection_segment(part) for part in parts]
        return " / ".join([category, *clean])

    def _collection_id_for(self, path: Path, component: str) -> str | None:
        relative = self._collection_relative_path(path, component)
        if relative in self._collection_ids:
            return self._collection_ids[relative]
        client = getattr(self.session, "client", None)
        ensure = getattr(client, "ensure_collection_path", None)
        root = str(self.provenance_root_path or "").strip()
        if not root or not callable(ensure):
            return None
        destination = self._ensure_collection_path(f"{root} / {relative}")
        collection_id = str(destination.collection_id)
        self._collection_ids[relative] = collection_id
        return collection_id

    def capture_completed_cells(self) -> None:
        """Extract each newly captured rich figure into a deterministic file."""

        trace = getattr(self.session, "_trace", None)
        cells = list(getattr(trace, "cells", []) or [])
        if self._display_hook_registered:
            self._processed_cell_count = len(cells)
            return
        for ordinal, cell in enumerate(
            cells[self._processed_cell_count :],
            start=self._processed_cell_count + 1,
        ):
            execution_count = cell.get("execution_count") or ordinal
            for display_index, output in enumerate(
                cell.get("display_outputs") or [], start=1
            ):
                self._queue_display_output(
                    (output or {}).get("data") or {},
                    execution_count=int(execution_count),
                    display_index=display_index,
                )
        self._processed_cell_count = len(cells)

    def _capture_display(self, message: dict[str, Any]) -> dict[str, Any]:
        """Save the untruncated rich output before the trace bounds its JSON."""

        content = message.get("content", message)
        execution_count = int(getattr(self.shell, "execution_count", 0) or 0)
        display_index = self._display_counts.get(execution_count, 0) + 1
        self._display_counts[execution_count] = display_index
        self._queue_display_output(
            content.get("data", {}),
            execution_count=execution_count,
            display_index=display_index,
        )
        return message

    def _queue_display_output(
        self,
        data: dict[str, Any],
        *,
        execution_count: int,
        display_index: int,
    ) -> None:
        extracted = self._display_bytes(data)
        if extracted is None:
            return
        extension, content = extracted
        filename = f"cell-{execution_count:04d}-display-{display_index:02d}{extension}"
        path = self.artifact_directory / "figures" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self._analysis_queue[path] = _PendingArtifact(
            path=path,
            title=_stable_title(f"{Path(self.notebook_name).stem} figure", filename),
            component="notebook-figure",
            metadata={
                "cell_execution_count": execution_count,
                "display_index": display_index,
                "source_notebook": self.notebook_name,
            },
        )

    def queue_model_artifacts(
        self,
        *,
        model_path: str | os.PathLike,
        loss_path: str | os.PathLike | None = None,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        lineage_run_id: str | None = None,
    ) -> None:
        """Queue a checkpoint, loss history, and generated provenance manifest."""

        model = Path(model_path).expanduser().resolve()
        self._model_queue[model] = _PendingModel(
            model_path=model,
            loss_path=(Path(loss_path).expanduser().resolve() if loss_path else None),
            params=_json_safe(params or {}),
            metrics=_json_safe(metrics or {}),
            lineage_run_id=lineage_run_id,
        )

    def publish_pytorch_training(
        self,
        *,
        model: Any,
        optimizer: Any,
        model_path: str | os.PathLike,
        loss_path: str | os.PathLike | None,
        params: dict[str, Any],
        metrics: dict[str, Any],
        epoch: int | None,
        run_name: str,
        dataset_asset_id: str | None = None,
    ) -> Any:
        """Publish a complete checkpoint with Dataerai's PyTorch toolkit."""

        from dataerai.nn import PyTorchProvenanceTracker

        legacy_model = Path(model_path).expanduser().resolve()
        legacy_loss = (
            Path(loss_path).expanduser().resolve() if loss_path is not None else None
        )
        self._ignored_model_paths.add(legacy_model)
        dataset_references = [
            {
                "asset_id": asset_id,
                "filename": self._relative(path),
            }
            for path, asset_id in self._raw_by_path.items()
        ]
        if dataset_asset_id and dataset_asset_id not in {
            item["asset_id"] for item in dataset_references
        }:
            dataset_references.append({"asset_id": dataset_asset_id})

        tracker = PyTorchProvenanceTracker(
            model,
            optimizer,
            session=_RoutedNotebookSession(self),
            output_dir=legacy_model.parent,
            run_name=run_name,
            dataset_references=dataset_references,
            notebook_path=(self.root / self.notebook_name),
            notebook_asset_id=self.notebook_asset_id,
            tags=(
                "m3-learning",
                "model",
                "checkpoint",
                "nn-provenance",
                "pytorch",
                f"notebook-run:{self.run_id}",
            ),
        )
        outcomes: dict[str, Any] = {
            "legacy_state_dict": _file_manifest(
                legacy_model, self._relative(legacy_model)
            )
        }
        if legacy_loss is not None and legacy_loss.is_file():
            outcomes["loss_history"] = _file_manifest(
                legacy_loss, self._relative(legacy_loss)
            )
        result = tracker.save_checkpoint(
            epoch=epoch,
            metrics=_json_safe(metrics),
            training_parameters=_json_safe(params),
            outcomes=outcomes,
            label="final",
            record_name=legacy_model.stem,
        )
        self._publish_training_companions(
            legacy_model,
            legacy_loss,
            checkpoint_asset_id=result.checkpoint_asset_id,
        )
        return result

    def _publish_training_companions(
        self,
        legacy_model: Path,
        legacy_loss: Path | None,
        *,
        checkpoint_asset_id: str | None,
    ) -> None:
        """Publish M3's state dict and loss history beside the safe checkpoint."""

        legacy = self._upload(
            legacy_model,
            title=_stable_title("M3 PyTorch state dict", self._relative(legacy_model)),
            record_type="model",
            component="nn-legacy-state-dict",
            metadata={
                "source_path": self._relative(legacy_model),
                "safe_checkpoint_asset_id": checkpoint_asset_id,
            },
            tags=["m3-learning", "pytorch", "state-dict", "neural-network"],
        )
        legacy_id = str(legacy.asset_id)
        self._model_asset_ids.append(legacy_id)
        if checkpoint_asset_id:
            self._relationship(legacy_id, str(checkpoint_asset_id), "derived_from")
        self._link_to_raw(legacy_id, "trained_on")
        self._link_to_notebook(legacy_id, "generated_by_notebook")

        if legacy_loss is None or not legacy_loss.is_file():
            return
        loss = self._upload(
            legacy_loss,
            title=_stable_title("M3 training loss", self._relative(legacy_loss)),
            record_type="model",
            component="nn-training-loss",
            metadata={
                "source_path": self._relative(legacy_loss),
                "safe_checkpoint_asset_id": checkpoint_asset_id,
            },
            tags=["m3-learning", "pytorch", "training-loss", "neural-network"],
        )
        loss_id = str(loss.asset_id)
        self._model_asset_ids.append(loss_id)
        if checkpoint_asset_id:
            self._relationship(loss_id, str(checkpoint_asset_id), "training_output_of")
        self._link_to_raw(loss_id, "derived_from")
        self._link_to_notebook(loss_id, "generated_by_notebook")

    def finish(self) -> ArtifactPublishResult:
        """Publish queued and changed artifacts before the trace is finalized."""

        if self._finished:
            return self._result or ArtifactPublishResult(errors=tuple(self._errors))
        self.capture_completed_cells()
        self._unregister()
        current = self._snapshot()
        self._discover_new_raw_datasets(current)
        self._queue_saved_figures(current)
        self._queue_generic_models(current)

        for artifact in sorted(
            self._analysis_queue.values(), key=lambda item: str(item.path)
        ):
            self._attempt(lambda artifact=artifact: self._publish_analysis(artifact))
        self._publish_changed_data(current)
        for model in sorted(
            self._model_queue.values(), key=lambda item: str(item.model_path)
        ):
            self._attempt(lambda model=model: self._publish_model(model))

        self._errors.extend(
            message
            for message in self._raw_errors.values()
            if message not in self._errors
        )

        self._result = ArtifactPublishResult(
            notebook_asset_id=self.notebook_asset_id,
            raw_dataset_asset_ids=self.raw_dataset_asset_ids,
            analysis_asset_ids=tuple(self._analysis_asset_ids),
            dataset_asset_ids=tuple(self._dataset_asset_ids),
            model_asset_ids=tuple(self._model_asset_ids),
            errors=tuple(self._errors),
        )
        self._finished = True
        print(
            "Dataerai artifacts: "
            f"{_counted(len(self.raw_dataset_asset_ids), 'source dataset')}, "
            f"{_counted(len(self._analysis_asset_ids), 'analysis', 'analyses')}, "
            f"{_counted(len(self._dataset_asset_ids), 'derived dataset')}, "
            f"{_counted(len(self._model_asset_ids), 'model artifact')}."
        )
        if self._errors and self.strict:
            # A trace withdrawn mid-run by a transient server fault is not an
            # artifact-publishing defect: the notebook computed correctly and its
            # outputs are already on disk. Failing here discards that work (we
            # lost a 2h09m run this way), so warn and let the notebook exit 0.
            if all(_is_withdrawn_trace_error(err) for err in self._errors):
                warnings.warn(
                    "Dataerai artifact publishing skipped: the notebook trace was "
                    "withdrawn before artifacts could be published "
                    f"({len(self._errors)} artifact(s) affected). Computed outputs "
                    "are unaffected.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._result
            raise RuntimeError(
                "Dataerai artifact publishing failed: " + "; ".join(self._errors)
            )
        return self._result

    def _post_run_cell(self, result: Any) -> None:
        del result
        self.capture_completed_cells()
        self._discover_new_raw_datasets(self._snapshot())

    def _unregister(self) -> None:
        events = getattr(self.shell, "events", None)
        if self._registered and events is not None:
            try:
                events.unregister("post_run_cell", self._post_run_cell)
            except (KeyError, ValueError):
                pass
        self._registered = False
        display_pub = getattr(self.shell, "display_pub", None)
        if (
            self._display_hook_registered
            and display_pub is not None
            and hasattr(display_pub, "unregister_hook")
        ):
            try:
                display_pub.unregister_hook(self._capture_display)
            except (KeyError, ValueError):
                pass
        self._display_hook_registered = False

    def _snapshot(self) -> dict[Path, _FileState]:
        snapshot: dict[Path, _FileState] = {}
        if not self.root.exists():
            return snapshot
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _WATCHED_EXTENSIONS:
                continue
            if self.artifact_directory in path.parents:
                continue
            relative_parts = path.relative_to(self.root).parts
            if any(part in _IGNORED_DIRECTORY_NAMES for part in relative_parts):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.resolve()] = _FileState(stat.st_size, stat.st_mtime_ns)
        return snapshot

    def _discover_new_raw_datasets(self, current: dict[Path, _FileState]) -> None:
        for path, state in sorted(current.items(), key=lambda item: str(item[0])):
            suffix = path.suffix.lower()
            if (
                path in self._raw_states
                or suffix not in _DATA_EXTENSIONS | _RAW_ARCHIVE_EXTENSIONS
            ):
                continue
            if (
                suffix in _RAW_ARCHIVE_EXTENSIONS
                or _is_raw_candidate(path)
                or (path in self._baseline and self._is_source_input(path))
            ):
                self._ensure_raw_dataset(path, state=state)

    def _ensure_raw_dataset(
        self,
        path: Path,
        *,
        state: _FileState | None = None,
    ) -> str | None:
        path = path.resolve()
        if path in self._raw_by_path:
            return self._raw_by_path[path]
        title = _stable_title("M3 source dataset", self._relative(path))
        try:
            explicit_id = (
                os.environ.get("DATAERAI_RAW_DATA_ASSET_ID")
                if not self._raw_by_path
                else None
            )
            if explicit_id:
                asset_id = explicit_id
            else:
                matches = self.session.find_assets(
                    title,
                    title=title,
                    predicate=lambda asset: bool(getattr(asset, "has_content", True)),
                )
                if len(matches) > 1:
                    raise LookupError(
                        f"more than one readable source dataset has title {title!r}; "
                        "set DATAERAI_RAW_DATA_ASSET_ID explicitly"
                    )
                if matches:
                    asset_id = str(matches[0].asset_id)
                elif _exceeds_inline_asset_limit(path):
                    if not self.raw_dataset_asset_ids:
                        raise RuntimeError(
                            f"source dataset is {path.stat().st_size:,} bytes, "
                            "which exceeds DATAERAI_MAX_INLINE_ASSET_BYTES; "
                            "publish the immutable original source first or set "
                            "DATAERAI_RAW_DATA_ASSET_ID"
                        )
                    asset_id = self.raw_dataset_asset_ids[0]
                elif _exceeds_raw_single_asset_limit(path):
                    asset_id = self._publish_raw_bundle(path, title=title)
                else:
                    uploaded = self._upload(
                        path,
                        title=title,
                        record_type="dataset",
                        component="raw-dataset",
                        metadata={"source_path": self._relative(path)},
                        tags=["m3-learning", "raw-data", "notebook-source"],
                    )
                    asset_id = str(uploaded.asset_id)
            self._raw_by_path[path] = asset_id
            self._raw_states[path] = state or self._state(path)
            self._raw_errors.pop(path, None)
            self._record_reused_raw_product(path, asset_id)
            if self.notebook_asset_id and self.notebook_asset_id != asset_id:
                self._relationship(self.notebook_asset_id, asset_id, "uses_data")
            os.environ.setdefault("DATAERAI_RAW_DATA_ASSET_ID", asset_id)
            os.environ.setdefault("DATAERAI_DATASET_ASSET_ID", asset_id)
            return asset_id
        except Exception as exc:  # noqa: BLE001 - SDK/network boundary
            self._raw_errors[path] = (
                f"could not publish/reuse source dataset {path}: {exc}"
            )
            return None

    def _record_reused_raw_product(self, path: Path, asset_id: str) -> None:
        """Include a reused raw input in the execution's signed relationships."""

        trace = getattr(self.session, "_trace", None)
        transfers = getattr(trace, "transfers", None)
        if not isinstance(transfers, list) or any(
            str(item.get("asset_id")) == str(asset_id) for item in transfers
        ):
            return
        transfers.append(
            {
                "direction": "reuse",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "asset_id": str(asset_id),
                "title": _stable_title("M3 source dataset", self._relative(path)),
                "record_type": "dataset",
                "component": "raw-dataset",
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": None,
                "run_identity_stamped": False,
                "collection_routing": "reused-provenance-input",
            }
        )

    def _publish_raw_bundle(self, path: Path, *, title: str) -> str:
        """Publish a large immutable source as parts plus one canonical manifest."""

        chunk_bytes = _raw_chunk_bytes()
        bundle_directory = self.artifact_directory / "raw-bundles" / path.name
        bundle_directory.mkdir(parents=True, exist_ok=True)
        part_paths: list[Path] = []
        full_digest = hashlib.sha256()
        with path.open("rb") as source:
            part_number = 1
            while True:
                content = source.read(chunk_bytes)
                if not content:
                    break
                full_digest.update(content)
                part_path = bundle_directory / f"part-{part_number:04d}.bin"
                part_path.write_bytes(content)
                part_paths.append(part_path)
                part_number += 1

        total_parts = len(part_paths)
        part_assets: list[dict[str, Any]] = []
        for index, part_path in enumerate(part_paths, start=1):
            part_sha256 = _sha256(part_path)
            part_title = _stable_title(
                "M3 source dataset part",
                f"{self._relative(path)} · {index:04d}-of-{total_parts:04d}",
            )
            matches = self.session.find_assets(
                part_title,
                title=part_title,
                predicate=lambda asset: bool(getattr(asset, "has_content", True)),
            )
            if len(matches) > 1:
                raise LookupError(
                    f"more than one readable raw-data part has title {part_title!r}"
                )
            if matches:
                part_asset_id = str(matches[0].asset_id)
            else:
                uploaded = self._upload(
                    part_path,
                    title=part_title,
                    record_type="dataset",
                    component="raw-dataset-part",
                    metadata={
                        "source_path": self._relative(path),
                        "part_number": index,
                        "part_count": total_parts,
                        "part_sha256": part_sha256,
                        "part_size_bytes": part_path.stat().st_size,
                    },
                    tags=["m3-learning", "raw-data", "dataset-part"],
                )
                part_asset_id = str(uploaded.asset_id)
            part_assets.append(
                {
                    "asset_id": part_asset_id,
                    "filename": part_path.name,
                    "part_number": index,
                    "size_bytes": part_path.stat().st_size,
                    "sha256": part_sha256,
                }
            )

        manifest_path = bundle_directory / f"{path.name}.manifest.json"
        manifest_payload = {
            "schema": "m3-learning.raw-dataset-bundle.v1",
            "source_path": self._relative(path),
            "source_filename": path.name,
            "source_size_bytes": path.stat().st_size,
            "source_sha256": full_digest.hexdigest(),
            "part_count": total_parts,
            "chunk_bytes": chunk_bytes,
            "parts": part_assets,
            "reassemble": f"cat part-*.bin > {path.name}",
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = self._upload(
            manifest_path,
            title=title,
            record_type="dataset",
            component="raw-dataset-manifest",
            metadata={
                "source_path": self._relative(path),
                "source_size_bytes": path.stat().st_size,
                "source_sha256": full_digest.hexdigest(),
                "part_count": total_parts,
                "content_mode": "multipart-manifest",
            },
            tags=["m3-learning", "raw-data", "multipart-manifest"],
        )
        manifest_asset_id = str(manifest.asset_id)
        for part in part_assets:
            self._relationship(str(part["asset_id"]), manifest_asset_id, "part_of")
        return manifest_asset_id

    def _queue_saved_figures(self, current: dict[Path, _FileState]) -> None:
        for path, state in current.items():
            suffix = path.suffix.lower()
            if suffix not in _ANALYSIS_EXTENSIONS:
                continue
            if (
                path in self._analysis_queue
                or path in self._published_paths
                or self._baseline.get(path) == state
            ):
                continue
            self._analysis_queue[path] = _PendingArtifact(
                path=path,
                title=_stable_title("M3 figure", self._relative(path)),
                component=(
                    "saved-movie" if suffix in _MOVIE_EXTENSIONS else "saved-figure"
                ),
                metadata={"source_path": self._relative(path)},
            )

    def _queue_generic_models(self, current: dict[Path, _FileState]) -> None:
        for path, state in current.items():
            if path.suffix.lower() not in _MODEL_EXTENSIONS:
                continue
            if (
                path in self._model_queue
                or path in self._published_paths
                or path in self._ignored_model_paths
                or self._baseline.get(path) == state
            ):
                continue
            self._model_queue[path] = _PendingModel(model_path=path)

    def _publish_analysis(self, artifact: _PendingArtifact) -> None:
        is_movie = artifact.path.suffix.lower() in _MOVIE_EXTENSIONS
        uploaded = self._upload(
            artifact.path,
            title=artifact.title,
            record_type="analysis",
            component=artifact.component,
            metadata=artifact.metadata,
            tags=[
                "m3-learning",
                "notebook-movie" if is_movie else "notebook-figure",
                "analysis",
            ],
        )
        asset_id = str(uploaded.asset_id)
        self._analysis_asset_ids.append(asset_id)
        self._link_to_raw(asset_id, "analysis_of")
        self._link_to_notebook(asset_id, "generated_by_notebook")

    def _publish_changed_data(self, current: dict[Path, _FileState]) -> None:
        for path, state in sorted(current.items(), key=lambda item: str(item[0])):
            if path.suffix.lower() not in _DATA_EXTENSIONS:
                continue
            comparison = self._raw_states.get(path, self._baseline.get(path))
            if comparison == state:
                continue
            self._attempt(lambda path=path: self._publish_derived_data(path))

    def _publish_derived_data(self, path: Path) -> None:
        upload_path = path
        component = "derived-data"
        content_mode = "full"
        if _exceeds_inline_asset_limit(path):
            upload_path = self._write_oversized_data_manifest(path)
            component = "derived-data-manifest"
            content_mode = "manifest-only"
        uploaded = self._upload(
            upload_path,
            title=_stable_title("M3 derived data", self._relative(path)),
            record_type="dataset",
            component=component,
            metadata={
                "source_path": self._relative(path),
                "source_size_bytes": path.stat().st_size,
                "versioned": True,
                "content_mode": content_mode,
            },
            tags=["m3-learning", "derived-data", "versioned", content_mode],
        )
        asset_id = str(uploaded.asset_id)
        self._dataset_asset_ids.append(asset_id)
        self._link_to_raw(asset_id, "derived_from")
        self._link_to_notebook(asset_id, "generated_by_notebook")

    def _write_oversized_data_manifest(self, path: Path) -> Path:
        """Represent a derived file that cannot fit in the upload allocation."""

        relative = Path(self._relative(path))
        manifest_path = (
            self.artifact_directory
            / "derived-data"
            / relative.parent
            / f"{relative.name}.manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "m3-learning.oversized-derived-data.v1",
            "notebook": self.notebook_name,
            "notebook_run_id": self.run_id,
            "source_path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "source_dataset_asset_ids": list(self.raw_dataset_asset_ids),
            "content_mode": "manifest-only",
            "reason": "file exceeds DATAERAI_MAX_INLINE_ASSET_BYTES",
            "max_inline_asset_bytes": _max_inline_asset_bytes(),
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest_path

    def _publish_model(self, queued: _PendingModel) -> None:
        if not queued.model_path.is_file():
            raise FileNotFoundError(queued.model_path)
        checkpoint = self._upload(
            queued.model_path,
            title=_stable_title("M3 model", self._relative(queued.model_path)),
            record_type="model",
            component="model-checkpoint",
            metadata={
                "source_path": self._relative(queued.model_path),
                "lineage_run_id": queued.lineage_run_id,
            },
            tags=["m3-learning", "model-checkpoint", "neural-network"],
        )
        checkpoint_id = str(checkpoint.asset_id)
        self._model_asset_ids.append(checkpoint_id)
        self._link_to_raw(checkpoint_id, "trained_on")
        self._link_to_notebook(checkpoint_id, "generated_by_notebook")

        if queued.loss_path is not None and queued.loss_path.is_file():
            loss = self._upload(
                queued.loss_path,
                title=_stable_title(
                    "M3 training loss", self._relative(queued.loss_path)
                ),
                record_type="model",
                component="training-loss",
                metadata={
                    "source_path": self._relative(queued.loss_path),
                    "lineage_run_id": queued.lineage_run_id,
                },
                tags=["m3-learning", "training-loss", "neural-network"],
            )
            loss_id = str(loss.asset_id)
            self._model_asset_ids.append(loss_id)
            self._relationship(loss_id, checkpoint_id, "training_output_of")
            self._link_to_raw(loss_id, "derived_from")

        manifest_path = self._write_model_manifest(queued)
        manifest = self._upload(
            manifest_path,
            title=_stable_title("M3 model manifest", self._relative(queued.model_path)),
            record_type="model",
            component="model-manifest",
            metadata={
                "model_path": self._relative(queued.model_path),
                "lineage_run_id": queued.lineage_run_id,
            },
            tags=["m3-learning", "model-manifest", "neural-network"],
        )
        manifest_id = str(manifest.asset_id)
        self._model_asset_ids.append(manifest_id)
        self._relationship(manifest_id, checkpoint_id, "describes")
        self._link_to_raw(manifest_id, "derived_from")

    def _write_model_manifest(self, queued: _PendingModel) -> Path:
        path = (
            self.artifact_directory
            / "models"
            / f"{queued.model_path.stem}.manifest.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "m3-learning.dataerai-model-manifest.v1",
            "notebook": self.notebook_name,
            "notebook_run_id": self.run_id,
            "lineage_run_id": queued.lineage_run_id,
            "source_dataset_asset_ids": list(self.raw_dataset_asset_ids),
            "checkpoint": _file_manifest(
                queued.model_path, self._relative(queued.model_path)
            ),
            "loss_history": (
                _file_manifest(queued.loss_path, self._relative(queued.loss_path))
                if queued.loss_path is not None and queued.loss_path.is_file()
                else None
            ),
            "params": queued.params,
            "metrics": queued.metrics,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def _upload(
        self,
        path: Path,
        *,
        title: str,
        record_type: str,
        component: str,
        metadata: dict[str, Any],
        tags: list[str],
    ) -> Any:
        stamped_metadata = {
            **_json_safe(metadata),
            "component": component,
            "m3_notebook_run_id": self.run_id,
            "source_notebook": self.notebook_name,
        }
        stamped_tags = list(
            dict.fromkeys(
                [
                    *tags,
                    "m3-notebook-provenance",
                    f"m3-notebook-run:{self.run_id}",
                ]
            )
        )
        collection_id = self._collection_id_for(path, component)
        attempts = max(1, int(os.environ.get("DATAERAI_UPLOAD_ATTEMPTS", "3")))
        for attempt in range(1, attempts + 1):
            try:
                uploaded = self._upload_once(
                    path,
                    title=title,
                    record_type=record_type,
                    tags=stamped_tags,
                    metadata=stamped_metadata,
                    collection_id=collection_id,
                )
                self._published_paths.add(path.resolve())
                return uploaded
            except Exception as exc:  # noqa: BLE001 - SDK/network boundary
                if attempt == attempts or not _is_transient_upload_error(exc):
                    raise
                time.sleep(min(2 ** (attempt - 1), _max_retry_backoff_seconds()))
        raise AssertionError("unreachable")

    def _upload_once(
        self,
        path: Path,
        *,
        title: str,
        record_type: str,
        tags: list[str],
        metadata: dict[str, Any],
        collection_id: str | None,
    ) -> Any:
        """Upload through a routed collection while preserving trace capture."""

        if collection_id in (None, str(getattr(self.session, "collection_id", ""))):
            return self.session.upload(
                str(path),
                title=title,
                record_type=record_type,
                tags=tags,
                metadata=metadata,
            )
        client = getattr(self.session, "client", None)
        if client is None or not hasattr(client, "upload"):
            return self.session.upload(
                str(path),
                title=title,
                record_type=record_type,
                tags=tags,
                metadata=metadata,
                collection_id=collection_id,
            )
        trace = getattr(self.session, "_trace", None)
        kwargs = {
            "owner_type": "project",
            "owner_id": self.session.project_id,
            "collection_id": collection_id,
            "record_type": record_type,
            "tags": tags,
            "metadata": metadata,
        }
        if trace is not None and getattr(trace, "uses_unsigned_legacy_contract", False):
            kwargs = trace.decorate_upload(kwargs)
        uploaded = client.upload(str(path), title=title, **kwargs)
        if trace is not None and getattr(self.session, "_trace", None) is trace:
            self._record_routed_trace_product(
                trace,
                path,
                title=title,
                record_type=record_type,
                component=str(metadata.get("component") or ""),
                asset_id=str(uploaded.asset_id),
            )
        return uploaded

    @staticmethod
    def _record_routed_trace_product(
        trace: Any,
        path: Path,
        *,
        title: str,
        record_type: str,
        component: str,
        asset_id: str,
    ) -> None:
        """Add a sibling-collection product to the signed execution payload."""

        transfers = getattr(trace, "transfers", None)
        if not isinstance(transfers, list):
            return
        transfers.append(
            {
                "direction": "upload",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "asset_id": asset_id,
                "title": title,
                "record_type": record_type,
                "component": component,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "run_identity_stamped": False,
                "collection_routing": "sibling-provenance-collection",
            }
        )

    def _link_to_raw(self, asset_id: str, relation: str) -> None:
        for raw_id in self.raw_dataset_asset_ids:
            if raw_id != asset_id:
                self._relationship(asset_id, raw_id, relation)

    def _link_to_notebook(self, asset_id: str, relation: str) -> None:
        if self.notebook_asset_id and self.notebook_asset_id != asset_id:
            self._relationship(asset_id, self.notebook_asset_id, relation)

    def _relationship(self, source: str, target: str, relation: str) -> None:
        attempts = max(1, int(os.environ.get("DATAERAI_UPLOAD_ATTEMPTS", "3")))
        for attempt in range(1, attempts + 1):
            try:
                self.session.create_relationship(
                    source,
                    target,
                    relation,
                    qualifiers={"notebook_run_id": self.run_id},
                )
                return
            except Exception as exc:
                code = str(getattr(exc, "code", ""))
                if "EXISTS" in code.upper() or "EXISTS" in str(exc).upper():
                    return
                if attempt == attempts or not _is_transient_upload_error(exc):
                    raise
                time.sleep(min(2 ** (attempt - 1), _max_retry_backoff_seconds()))

    def _attempt(self, operation: Any) -> None:
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 - isolate artifact failures
            self._errors.append(f"{type(exc).__name__}: {exc}")

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.name

    def _is_source_input(self, path: Path) -> bool:
        return path.suffix.lower() in _RAW_ARCHIVE_EXTENSIONS or _is_raw_candidate(path)

    @staticmethod
    def _state(path: Path) -> _FileState:
        stat = path.stat()
        return _FileState(stat.st_size, stat.st_mtime_ns)

    @staticmethod
    def _display_bytes(data: dict[str, Any]) -> tuple[str, bytes] | None:
        for mime, extension, binary in (
            ("image/png", ".png", True),
            ("image/jpeg", ".jpg", True),
            ("image/svg+xml", ".svg", False),
            ("application/pdf", ".pdf", True),
        ):
            if mime not in data:
                continue
            value = data[mime]
            if isinstance(value, list):
                value = "".join(str(item) for item in value)
            if binary:
                return extension, base64.b64decode(str(value))
            return extension, str(value).encode("utf-8")
        plotly = data.get("application/vnd.plotly.v1+json")
        if plotly is not None:
            return ".plotly.json", (
                json.dumps(_json_safe(plotly), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        return None


def start_dataerai_artifact_publishing(
    session: Any,
    notebook_name: str,
    *,
    root: Path | str | None = None,
    shell: Any = None,
    provenance_root_path: str | None = None,
) -> DataeraiArtifactPublisher:
    """Start and globally register the publisher used by instrumented fitters."""

    global _ACTIVE_PUBLISHER
    publisher = DataeraiArtifactPublisher(
        session=session,
        notebook_name=notebook_name,
        root=root or os.environ.get("DATAERAI_ARTIFACT_ROOT") or Path.cwd(),
        shell=shell,
        provenance_root_path=provenance_root_path,
    ).start()
    _ACTIVE_PUBLISHER = publisher
    return publisher


def queue_dataerai_model_artifacts(
    *,
    model_path: str | os.PathLike,
    loss_path: str | os.PathLike | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    lineage_run_id: str | None = None,
) -> None:
    """Queue fitter outputs on the active publisher, if a trace enabled one."""

    if _ACTIVE_PUBLISHER is None:
        return
    _ACTIVE_PUBLISHER.queue_model_artifacts(
        model_path=model_path,
        loss_path=loss_path,
        params=params,
        metrics=metrics,
        lineage_run_id=lineage_run_id,
    )


def publish_dataerai_pytorch_training(
    *,
    model: Any,
    optimizer: Any,
    model_path: str | os.PathLike,
    loss_path: str | os.PathLike | None,
    params: dict[str, Any],
    metrics: dict[str, Any],
    epoch: int | None,
    run_name: str,
    enabled: bool | None = None,
    dataset_asset_id: str | None = None,
) -> Any | None:
    """Publish NN state through ``dataerai.nn.PyTorchProvenanceTracker``."""

    should_publish = (
        _env_bool("DATAERAI_LOG_PROVENANCE", default=False)
        if enabled is None
        else bool(enabled)
    )
    if not should_publish or _ACTIVE_PUBLISHER is None:
        return None
    return _ACTIVE_PUBLISHER.publish_pytorch_training(
        model=model,
        optimizer=optimizer,
        model_path=model_path,
        loss_path=loss_path,
        params=params,
        metrics=metrics,
        epoch=epoch,
        run_name=run_name,
        dataset_asset_id=dataset_asset_id,
    )


class _RoutedNotebookSession:
    """Notebook-session facade that routes toolkit products into collections."""

    def __init__(self, publisher: DataeraiArtifactPublisher) -> None:
        self._publisher = publisher
        self.client = publisher.session.client

    @property
    def trace_run_id(self) -> str:
        return self._publisher.run_id

    def upload(
        self, local_path: str, *, title: str | None = None, **kwargs: Any
    ) -> Any:
        path = Path(local_path).expanduser().resolve()
        metadata = dict(kwargs.pop("metadata", {}) or {})
        component = str(metadata.pop("component", "nn-training-artifact"))
        record_type = str(kwargs.pop("record_type", "analysis"))
        tags = list(kwargs.pop("tags", []) or [])
        if kwargs:
            raise TypeError(f"unsupported routed upload arguments: {sorted(kwargs)}")
        uploaded = self._publisher._upload(
            path,
            title=title or path.name,
            record_type=record_type,
            component=component,
            metadata=metadata,
            tags=tags,
        )
        asset_id = str(uploaded.asset_id)
        if record_type == "model":
            self._publisher._model_asset_ids.append(asset_id)
        else:
            self._publisher._analysis_asset_ids.append(asset_id)
        return uploaded

    def create_relationship(self, *args: Any, **kwargs: Any) -> Any:
        return self._publisher.session.create_relationship(*args, **kwargs)

    def get_metadata(self, asset_id: str) -> Any:
        return self._publisher.session.get_metadata(asset_id)


def _stable_title(prefix: str, relative_path: str) -> str:
    title = f"{prefix} · {relative_path}"
    if len(title) <= 220:
        return title
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    return f"{title[:204]} · {digest}"


def _collection_segment(value: str) -> str:
    """Return a safe, readable collection path segment."""

    segment = " ".join(str(value).replace("/", " ").split()).strip(" .")
    return segment or "Uncategorized"


def _strip_artifact_category(parts: list[str], names: set[str]) -> list[str]:
    """Drop local category wrappers while retaining loop-created subfolders."""

    lowered = [part.casefold() for part in parts]
    for index, value in enumerate(lowered):
        if value in names:
            return parts[index + 1 :]
    return parts


def _counted(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _file_manifest(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return value.item()
    except (AttributeError, ValueError):
        return str(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_withdrawn_trace_error(message: str) -> bool:
    """Return whether an artifact error is just "the trace is already gone"."""

    return "requires an active notebook trace" in str(message).casefold()


def _is_transient_upload_error(exc: Exception) -> bool:
    """Return whether retrying an SDK upload can reasonably succeed."""

    message = f"{getattr(exc, 'code', '')} {exc}".casefold()
    return any(
        token in message
        for token in (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "temporarily unavailable",
            "too many requests",
            "http 408",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "err_server_error",
            "err_transfer_failed",
            "upload was not verified",
            'server status "failed"',
        )
    )


def _max_retry_backoff_seconds() -> float:
    """Cap for exponential retry backoff.

    The previous hard-coded 8s cap gave ~23s of total sleep across 5 attempts,
    which could not outlast a 22s beta outage observed on 2026-08-30.
    """

    return float(os.environ.get("DATAERAI_MAX_RETRY_BACKOFF_S", "30"))


def _max_inline_asset_bytes() -> int:
    return int(os.environ.get("DATAERAI_MAX_INLINE_ASSET_BYTES", str(8 * 1024**3)))


def _exceeds_inline_asset_limit(path: Path) -> bool:
    return path.stat().st_size > _max_inline_asset_bytes()


def _raw_chunk_bytes() -> int:
    return int(os.environ.get("DATAERAI_RAW_CHUNK_BYTES", str(600 * 1024**2)))


def _exceeds_raw_single_asset_limit(path: Path) -> bool:
    return path.stat().st_size > _raw_chunk_bytes()


def _is_raw_candidate(path: Path) -> bool:
    explicit = {
        Path(item).expanduser().resolve()
        for item in os.environ.get("DATAERAI_RAW_DATA_PATHS", "").split(os.pathsep)
        if item.strip()
    }
    if path.resolve() in explicit:
        return True
    name = path.name.casefold()
    return any(token in name for token in ("raw", "source", "standard", "data_file"))
