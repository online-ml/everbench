from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from everbench import archive_store, artifacts
from everbench.api import create_app, format_duration


class ConstantModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVERBENCH_API_KEY", "test-api-key")
    monkeypatch.setenv("EVERBENCH_MODEL_SIGNING_KEY", "test-signing-key")


def post_backtest(client, payload: bytes, signature: str, archive_sha256: str):
    return client.post(
        "/api/tasks/dummy/backtest",
        data={"model": (io.BytesIO(payload), "model.pkl"), "archive_sha256": archive_sha256},
        headers={
            "X-API-Key": "test-api-key",
            "X-Everbench-Artifact-Signature": signature,
        },
    )


def test_duration_format_is_compact() -> None:
    assert format_duration(1.4) == "1s"
    assert format_duration(61) == "1m"
    assert format_duration(3_700) == "1h"


def test_backtest_uses_the_posted_model_without_a_registration(
    credentials: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = artifacts.dumps(ConstantModel())
    signature = artifacts.sign(payload)
    path = tmp_path / "events.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "one",
                    "event_sequence": 1,
                    "event_available_at": "2026-01-01T00:00:00+00:00",
                    "payload_json": '{"value":1}',
                    "label": 1,
                    "label_available_at": "2026-01-01T00:00:01+00:00",
                }
            ]
        ),
        path,
    )
    manifest = SimpleNamespace(path=str(path), content_sha256="archive", row_count=1, byte_size=path.stat().st_size)
    monkeypatch.setattr("everbench.api._session", lambda: SimpleNamespace())
    monkeypatch.setattr(archive_store, "task_archive", lambda *args: manifest)

    with create_app().test_client() as client:
        response = post_backtest(client, payload, signature, "archive")

    assert response.status_code == 200
    body = response.get_json()
    assert body is not None
    assert body["predictions"] == 1
    assert body["labels"] == 1
    assert "timing_seconds" in body


def test_missing_archive_remains_a_not_found_response(
    credentials: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = artifacts.dumps(ConstantModel())
    signature = artifacts.sign(payload)
    manifest = SimpleNamespace(
        path=str(tmp_path / "missing.parquet"),
        content_sha256="missing",
        row_count=1,
        byte_size=1,
    )
    monkeypatch.setattr("everbench.api._session", lambda: SimpleNamespace())
    monkeypatch.setattr(archive_store, "task_archive", lambda *args: manifest)

    with create_app().test_client() as client:
        response = post_backtest(client, payload, signature, "missing")

    assert response.status_code == 404
