from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from everbench import artifacts
from everbench.archive import replay_archive
from everbench.tasks import load_task


class DelayedRate:
    """A model whose prediction exposes whether labels arrived too early."""

    def __init__(self) -> None:
        self.seen = 0

    def predict_one(self, features: dict[str, float]) -> float:
        del features
        return float(self.seen)

    def learn_one(self, features: dict[str, float], y: int) -> None:
        del features, y
        self.seen += 1


class BacktestTimelineTest(unittest.TestCase):
    def test_labels_only_affect_the_model_when_they_become_available(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "test-signing-key"
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
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "events.parquet"
            pq.write_table(pa.Table.from_pylist(rows), archive)
            result = replay_archive(task, artifacts.loads(payload, signature), archive)

        # Both predictions happen before the first label. The obsolete
        # row-at-a-time replay would learn the first label before predicting
        # ``two`` and would therefore report a different value.
        self.assertEqual(result["predictions"], 2)
        self.assertEqual(result["labels"], 2)
        self.assertEqual(result["metrics"]["Accuracy"], 0.5)
        self.assertGreaterEqual(result["timing_seconds"]["predict"], 0.0)
        self.assertGreaterEqual(result["timing_seconds"]["learn"], 0.0)
        self.assertEqual(
            result["timing_seconds"]["total"],
            result["timing_seconds"]["predict"] + result["timing_seconds"]["learn"],
        )


if __name__ == "__main__":
    unittest.main()
