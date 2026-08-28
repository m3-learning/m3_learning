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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DATA_EXTENSIONS = {".h5", ".hdf5", ".csv"}
_RAW_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".xz"}
_MODEL_EXTENSIONS = {".pth", ".pt"}
_FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
_WATCHED_EXTENSIONS = (
    _DATA_EXTENSIONS
    | _RAW_ARCHIVE_EXTENSIONS
    | _MODEL_EXTENSIONS
    | _FIGURE_EXTENSIONS
)
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "_build",
    "node_modules",
}
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
    _errors: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
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

    def start(self) -> DataeraiArtifactPublisher:
        """Snapshot inputs, reuse/upload source data, and observe later cells."""

        if self._registered or self._finished:
            return self
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
        filename = (
            f"cell-{execution_count:04d}-display-{display_index:02d}{extension}"
        )
        path = self.artifact_directory / "figures" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self._analysis_queue[path] = _PendingArtifact(
            path=path,
            title=_stable_title(
                f"{Path(self.notebook_name).stem} figure", filename
            ),
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

        self._result = ArtifactPublishResult(
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
            if suffix in _RAW_ARCHIVE_EXTENSIONS or _is_raw_candidate(path) or (
                path in self._baseline and self._is_source_input(path)
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
                matches = self.session.find_assets(title, title=title)
                if len(matches) > 1:
                    raise LookupError(
                        f"more than one readable source dataset has title {title!r}; "
                        "set DATAERAI_RAW_DATA_ASSET_ID explicitly"
                    )
                if matches:
                    asset_id = str(matches[0].asset_id)
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
            os.environ.setdefault("DATAERAI_RAW_DATA_ASSET_ID", asset_id)
            os.environ.setdefault("DATAERAI_DATASET_ASSET_ID", asset_id)
            return asset_id
        except Exception as exc:  # noqa: BLE001 - SDK/network boundary
            self._errors.append(
                f"could not publish/reuse source dataset {path}: {exc}"
            )
            return None

    def _queue_saved_figures(self, current: dict[Path, _FileState]) -> None:
        for path, state in current.items():
            if path.suffix.lower() not in _FIGURE_EXTENSIONS:
                continue
            if path in self._analysis_queue or self._baseline.get(path) == state:
                continue
            self._analysis_queue[path] = _PendingArtifact(
                path=path,
                title=_stable_title("M3 figure", self._relative(path)),
                component="saved-figure",
                metadata={"source_path": self._relative(path)},
            )

    def _queue_generic_models(self, current: dict[Path, _FileState]) -> None:
        for path, state in current.items():
            if path.suffix.lower() not in _MODEL_EXTENSIONS:
                continue
            if path in self._model_queue or self._baseline.get(path) == state:
                continue
            self._model_queue[path] = _PendingModel(model_path=path)

    def _publish_analysis(self, artifact: _PendingArtifact) -> None:
        uploaded = self._upload(
            artifact.path,
            title=artifact.title,
            record_type="analysis",
            component=artifact.component,
            metadata=artifact.metadata,
            tags=["m3-learning", "notebook-figure", "analysis"],
        )
        asset_id = str(uploaded.asset_id)
        self._analysis_asset_ids.append(asset_id)
        self._link_to_raw(asset_id, "analysis_of")

    def _publish_changed_data(self, current: dict[Path, _FileState]) -> None:
        for path, state in sorted(current.items(), key=lambda item: str(item[0])):
            if path.suffix.lower() not in _DATA_EXTENSIONS:
                continue
            comparison = self._raw_states.get(path, self._baseline.get(path))
            if comparison == state:
                continue
            self._attempt(lambda path=path: self._publish_derived_data(path))

    def _publish_derived_data(self, path: Path) -> None:
        uploaded = self._upload(
            path,
            title=_stable_title("M3 derived data", self._relative(path)),
            record_type="dataset",
            component="derived-data",
            metadata={"source_path": self._relative(path), "versioned": True},
            tags=["m3-learning", "derived-data", "versioned"],
        )
        asset_id = str(uploaded.asset_id)
        self._dataset_asset_ids.append(asset_id)
        self._link_to_raw(asset_id, "derived_from")

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
            title=_stable_title(
                "M3 model manifest", self._relative(queued.model_path)
            ),
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
            "notebook_run_id": self.run_id,
            "source_notebook": self.notebook_name,
        }
        return self.session.upload(
            str(path),
            title=title,
            record_type=record_type,
            tags=tags,
            metadata=stamped_metadata,
        )

    def _link_to_raw(self, asset_id: str, relation: str) -> None:
        for raw_id in self.raw_dataset_asset_ids:
            if raw_id != asset_id:
                self._relationship(asset_id, raw_id, relation)

    def _relationship(self, source: str, target: str, relation: str) -> None:
        try:
            self.session.create_relationship(
                source,
                target,
                relation,
                qualifiers={"notebook_run_id": self.run_id},
            )
        except Exception as exc:
            code = str(getattr(exc, "code", ""))
            if "EXISTS" in code.upper() or "EXISTS" in str(exc).upper():
                return
            raise

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
        if _is_raw_candidate(path):
            return True
        try:
            parts = path.resolve().relative_to(self.root).parts
        except ValueError:
            return False
        return len(parts) <= 2

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
) -> DataeraiArtifactPublisher:
    """Start and globally register the publisher used by instrumented fitters."""

    global _ACTIVE_PUBLISHER
    publisher = DataeraiArtifactPublisher(
        session=session,
        notebook_name=notebook_name,
        root=root or os.environ.get("DATAERAI_ARTIFACT_ROOT") or Path.cwd(),
        shell=shell,
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


def _stable_title(prefix: str, relative_path: str) -> str:
    title = f"{prefix} · {relative_path}"
    if len(title) <= 220:
        return title
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    return f"{title[:204]} · {digest}"


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


def _is_raw_candidate(path: Path) -> bool:
    explicit = {
        Path(item).expanduser().resolve()
        for item in os.environ.get("DATAERAI_RAW_DATA_PATHS", "").split(os.pathsep)
        if item.strip()
    }
    if path.resolve() in explicit:
        return True
    name = path.name.casefold()
    return any(
        token in name for token in ("raw", "source", "standard", "data_file")
    )
