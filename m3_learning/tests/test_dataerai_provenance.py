import importlib.util
import json
import sys
import types
from pathlib import Path


_MODULE_PATH = Path(__file__).parents[1] / "src" / "m3_learning" / "provenance.py"
_SPEC = importlib.util.spec_from_file_location("m3_learning_provenance_test", _MODULE_PATH)
provenance = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = provenance
_SPEC.loader.exec_module(provenance)

build_training_lineage_payload = provenance.build_training_lineage_payload
load_dataerai_cli_credentials = provenance.load_dataerai_cli_credentials
log_dataerai_training_run = provenance.log_dataerai_training_run
resolve_dataerai_record_sk = provenance.resolve_dataerai_record_sk


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.get_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return _Resp({"start": {"kind": "record", "sk": 42}})


def test_load_dataerai_cli_credentials_from_env(monkeypatch):
    monkeypatch.setenv(
        "DATAERAI_CREDENTIALS_JSON",
        json.dumps({"server_url": "https://dev.dataerai.com", "access_token": "AT"}),
    )

    creds = load_dataerai_cli_credentials()

    assert creds["server"] == "https://dev.dataerai.com"
    assert creds["access_token"] == "AT"
    assert creds["client_id"] == "dataerai-cli"


def test_resolve_dataerai_record_sk_from_lineage_trace(monkeypatch):
    monkeypatch.delenv("DATAERAI_CREDENTIALS_JSON", raising=False)
    session = _Session()

    record_sk = resolve_dataerai_record_sk(
        "asset-1",
        credentials={"server_url": "https://dev.dataerai.com", "access_token": "AT"},
        session=session,
    )

    assert record_sk == 42
    url, kwargs = session.get_calls[0]
    assert url == "https://dev.dataerai.com/api/lineage/trace/"
    assert kwargs["params"]["subject_kind"] == 1
    assert kwargs["params"]["subject_id"] == "asset-1"
    assert kwargs["headers"]["Authorization"] == "Bearer AT"


def test_log_dataerai_training_run_disabled_is_noop():
    result = log_dataerai_training_run(enabled=False)

    assert result.enabled is False
    assert result.run_id is None
    assert result.skipped_reason == "disabled"


def test_log_dataerai_training_run_uses_sdk_helper(monkeypatch):
    calls = []
    lineage_mod = types.ModuleType("dataerai.ml.lineage")

    def _record_training_run(creds, **kwargs):
        calls.append((creds, kwargs))
        return {"run_id": "run-1", "idempotent_replay": False}

    lineage_mod.record_training_run = _record_training_run
    monkeypatch.setitem(sys.modules, "dataerai", types.ModuleType("dataerai"))
    monkeypatch.setitem(sys.modules, "dataerai.ml", types.ModuleType("dataerai.ml"))
    monkeypatch.setitem(sys.modules, "dataerai.ml.lineage", lineage_mod)

    result = log_dataerai_training_run(
        enabled=True,
        dataset_record_sk=42,
        credentials={"server_url": "https://dev.dataerai.com", "access_token": "AT"},
        params={"epochs": 1},
        metrics={"train_loss": 0.1},
        idempotency_key="stable-run",
    )

    assert result.run_id == "run-1"
    assert result.dataset_record_sk == 42
    creds, kwargs = calls[0]
    assert creds["server"] == "https://dev.dataerai.com"
    assert kwargs["dataset_record_sk"] == 42
    assert kwargs["params"] == {"epochs": 1}
    assert kwargs["metrics"] == {"train_loss": 0.1}
    assert kwargs["idempotency_key"] == "stable-run"


def test_build_training_lineage_payload_json_safe():
    params, metrics = build_training_lineage_payload(
        model_name="model",
        optimizer_name="Adam",
        epochs=2,
        batch_size=128,
        seed=41,
        train_loss=0.01,
        training_time_s=3.5,
        model_updates=10,
        noise_level=0,
        loss_func=object(),
    )

    assert params["model_name"] == "model"
    assert params["optimizer"] == "Adam"
    assert params["epochs"] == 2
    assert metrics["train_loss"] == 0.01
    assert metrics["model_updates"] == 10
