from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from everbench import artifacts
from everbench.archive import replay_archive
from everbench.tasks import load_task


class DelayedRate:
    """A model whose prediction exposes whether labels arrived too early."""

    def __init__(self) -> None:
        self.seen = 0

    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return float(self.seen)

    def learn_one(self, event_id: str, event: dict[str, Any], label: int) -> None:
        del event_id, event, label
        self.seen += 1


def test_labels_only_affect_the_model_when_they_become_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVERBENCH_MODEL_SIGNING_KEY", "test-signing-key")
    task = load_task(Path(__file__).parents[1] / "tasks" / "dummy" / "task.py")
    model = DelayedRate()
    payload = artifacts.dumps(model)
    signature = artifacts.sign(payload)
    rows = [
        {
            "event_id": "one",
            "event_sequence": 1,
            "event_available_at": "2026-01-01T00:00:00+00:00",
            "payload_json": '{"value":1}',
            "label": 1,
            "label_available_at": "2026-01-01T00:00:03+00:00",
        },
        {
            "event_id": "two",
            "event_sequence": 2,
            "event_available_at": "2026-01-01T00:00:01+00:00",
            "payload_json": '{"value":2}',
            "label": 0,
            "label_available_at": "2026-01-01T00:00:04+00:00",
        },
    ]
    archive = tmp_path / "events.parquet"
    pq.write_table(pa.Table.from_pylist(rows), archive)

    result = replay_archive(task, artifacts.loads(payload, signature), archive)

    # Both predictions happen before the first label. A row-at-a-time replay
    # would learn the first label before predicting the second event.
    assert result["predictions"] == 2
    assert result["labels"] == 2
    assert result["metrics"]["Accuracy"] == 0.5
    assert result["timing_seconds"]["predict"] >= 0.0
    assert result["timing_seconds"]["learn"] >= 0.0
    assert result["timing_seconds"]["total"] == (
        result["timing_seconds"]["predict"] + result["timing_seconds"]["learn"]
    )
