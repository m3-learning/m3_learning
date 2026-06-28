"""Dataerai provenance helpers for neural-network training runs.

The actual lineage write is delegated to Dataerai's SDK
``dataerai.ml.lineage.record_training_run``. This module only handles the
user-facing glue that notebooks need: loading credentials created by the
``dataerai`` CLI, resolving a normal Dataerai asset id to the lineage
``record_sk`` surrogate, and packaging common training parameters/metrics.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ASSET_SUBJECT_KIND = 1
_TIMEOUT = 30
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


@dataclass
class DataeraiLineageResult:
    """Outcome of an attempted Dataerai lineage write."""

    enabled: bool
    run_id: str | None = None
    dataset_record_sk: int | None = None
    idempotent_replay: bool = False
    skipped_reason: str | None = None
    response: dict[str, Any] | None = None


def dataerai_provenance_enabled(explicit: bool | None = None) -> bool:
    """Return whether Dataerai provenance logging should run."""

    if explicit is not None:
        return bool(explicit)
    value = (
        os.environ.get("DATAERAI_LOG_PROVENANCE")
        or os.environ.get("DATAERAI_PROVENANCE")
        or ""
    ).strip().lower()
    if value in _FALSE_VALUES:
        return False
    return value in _TRUE_VALUES


def load_dataerai_cli_credentials(path: str | os.PathLike | None = None) -> dict:
    """Load credentials written by the ``dataerai`` CLI.

    Lookup order:
    1. ``DATAERAI_CREDENTIALS_JSON``.
    2. Explicit ``path`` or ``DATAERAI_CREDENTIALS_PATH``.
    3. CLI file fallbacks: ``$XDG_CONFIG_HOME/dataerai/credentials``,
       ``~/.config/dataerai/credentials``, and the older
       ``~/.dataerai/credentials.json``.
    4. Optional Python ``keyring`` package reading the CLI's ``dataerai/auth``
       keychain item.
    """

    raw = os.environ.get("DATAERAI_CREDENTIALS_JSON")
    if raw:
        return _normalize_credentials(json.loads(raw))

    candidates: list[Path] = []
    env_path = os.environ.get("DATAERAI_CREDENTIALS_PATH")
    for p in (path, env_path):
        if p:
            candidates.append(Path(p).expanduser())

    xdg_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_home:
        candidates.append(Path(xdg_home).expanduser() / "dataerai" / "credentials")
    candidates.extend(
        [
            Path.home() / ".config" / "dataerai" / "credentials",
            Path.home() / ".dataerai" / "credentials.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return _normalize_credentials(json.loads(candidate.read_text()))

    try:
        import keyring  # type: ignore
    except Exception:
        keyring = None
    if keyring is not None:
        try:
            keychain_raw = keyring.get_password("dataerai", "auth")
        except Exception:
            keychain_raw = None
        if keychain_raw:
            return _normalize_credentials(json.loads(keychain_raw))

    raise FileNotFoundError(
        "No Dataerai CLI credentials found. Run `dataerai auth login --device` "
        "or set DATAERAI_CREDENTIALS_JSON / DATAERAI_CREDENTIALS_PATH."
    )


def resolve_dataerai_record_sk(
    asset_id: str,
    *,
    credentials: dict | None = None,
    session=None,
    subject_kind: int = ASSET_SUBJECT_KIND,
) -> int:
    """Resolve a Dataerai asset id to its provenance ``record_sk``."""

    if not asset_id:
        raise ValueError("asset_id is required")
    creds = _normalize_credentials(credentials or load_dataerai_cli_credentials())
    url = f"{_server(creds)}/api/lineage/trace/"
    params = {
        "subject_kind": int(subject_kind),
        "subject_id": str(asset_id),
        "direction": "down",
        "depth": 1,
    }
    http = session or _requests()
    resp = http.get(url, params=params, headers=_headers(creds), timeout=_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    try:
        return int(payload["start"]["sk"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Dataerai lineage trace did not include start.sk: {payload}") from exc


def build_training_lineage_payload(
    *,
    model_name: str,
    optimizer_name: str,
    epochs: int,
    batch_size: int,
    seed: int,
    train_loss: float,
    training_time_s: float,
    model_updates: int,
    noise_level: Any = None,
    loss_func: Any = None,
    model_path: str | os.PathLike | None = None,
    training_loss_path: str | os.PathLike | None = None,
    device: Any = None,
    stopped_early: bool = False,
    extra_params: dict | None = None,
    extra_metrics: dict | None = None,
) -> tuple[dict, dict]:
    """Build JSON-safe params and metrics for Dataerai NN lineage logging."""

    params = {
        "model_name": model_name,
        "optimizer": optimizer_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "noise_level": noise_level,
        "loss_function": str(loss_func),
        "model_path": str(model_path) if model_path is not None else None,
        "training_loss_path": (
            str(training_loss_path) if training_loss_path is not None else None
        ),
        "device": str(device) if device is not None else None,
        "runtime": _runtime_metadata(),
    }
    if extra_params:
        params.update(extra_params)

    metrics = {
        "train_loss": float(train_loss),
        "training_time_s": float(training_time_s),
        "model_updates": int(model_updates),
        "stopped_early": bool(stopped_early),
    }
    if extra_metrics:
        metrics.update(extra_metrics)

    return _json_safe(params), _json_safe(metrics)


def log_dataerai_training_run(
    *,
    enabled: bool | None = None,
    dataset_asset_id: str | None = None,
    dataset_record_sk: int | str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
    idempotency_key: str | None = None,
    credentials: dict | None = None,
    session=None,
) -> DataeraiLineageResult:
    """Record a Dataerai ``training`` lineage run when configured.

    If ``enabled`` is false, this returns a skipped result without importing the
    Dataerai SDK or performing network I/O. When enabled, ``dataset_record_sk``
    may be supplied directly or resolved from ``dataset_asset_id``.
    """

    should_log = dataerai_provenance_enabled(enabled)
    if not should_log:
        return DataeraiLineageResult(enabled=False, skipped_reason="disabled")

    dataset_record_sk = (
        dataset_record_sk
        or os.environ.get("DATAERAI_DATASET_RECORD_SK")
        or os.environ.get("DATAERAI_RECORD_SK")
    )
    dataset_asset_id = (
        dataset_asset_id
        or os.environ.get("DATAERAI_DATASET_ASSET_ID")
        or os.environ.get("DATAERAI_ASSET_ID")
    )
    idempotency_key = idempotency_key or os.environ.get("DATAERAI_LINEAGE_IDEMPOTENCY_KEY")

    if dataset_record_sk is None and not dataset_asset_id:
        return DataeraiLineageResult(
            enabled=True,
            skipped_reason=(
                "set DATAERAI_DATASET_ASSET_ID or DATAERAI_DATASET_RECORD_SK "
                "to identify the training source asset"
            ),
        )

    try:
        from dataerai.ml.lineage import record_training_run
    except Exception as exc:
        return DataeraiLineageResult(
            enabled=True,
            skipped_reason=f"dataerai-sdk is not installed or importable: {exc}",
        )

    creds = _normalize_credentials(credentials or load_dataerai_cli_credentials())
    if dataset_record_sk is None:
        dataset_record_sk = resolve_dataerai_record_sk(
            str(dataset_asset_id), credentials=creds, session=session
        )
    dataset_record_sk = int(dataset_record_sk)

    response = record_training_run(
        creds,
        dataset_record_sk=dataset_record_sk,
        params=_json_safe(params or {}),
        metrics=_json_safe(metrics or {}),
        idempotency_key=idempotency_key,
        session=session,
    )
    return DataeraiLineageResult(
        enabled=True,
        run_id=response.get("run_id"),
        dataset_record_sk=dataset_record_sk,
        idempotent_replay=bool(response.get("idempotent_replay")),
        response=response,
    )


def _normalize_credentials(creds: dict) -> dict:
    out = dict(creds)
    if "server" not in out and out.get("server_url"):
        out["server"] = out["server_url"]
    if "access_token" not in out and out.get("token"):
        out["access_token"] = out["token"]
    if "client_id" not in out:
        out["client_id"] = os.environ.get("DATAERAI_CLIENT_ID", "dataerai-cli")
    return out


def _server(creds: dict) -> str:
    server = creds.get("server") or creds.get("server_url") or ""
    if not server:
        raise KeyError('credentials missing "server" or "server_url"')
    return str(server).rstrip("/")


def _headers(creds: dict) -> dict:
    token = creds.get("access_token") or creds.get("token")
    if not token:
        raise KeyError('credentials missing "access_token"')
    return {"Authorization": f"Bearer {token}"}


def _requests():
    import requests

    return requests


def _runtime_metadata() -> dict:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return str(value)
