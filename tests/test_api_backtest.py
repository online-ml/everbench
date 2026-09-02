from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from everbench import artifacts
from everbench.api import create_app


class ConstantModel:
    def predict_one(self, features: dict[str, float]) -> float:
        del features
        return 0.5


class BacktestApiTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["EVERBENCH_API_KEY"] = "test-api-key"
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "test-signing-key"

    def test_backtest_uses_the_posted_model_without_a_registration(self) -> None:
        payload = artifacts.dumps(ConstantModel())
        signature = artifacts.sign(payload)
        rows = [
            {
                "event_id": "one",
                "event_sequence": 1,
                "event_available_at": "2026-01-01T00:00:00+00:00",
                "payload_json": '{"value":1}',
                "label": 1,
                "label_available_at": "2026-01-01T00:00:01+00:00",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            manifest = SimpleNamespace(
                path=str(path), content_sha256="archive", row_count=1, byte_size=path.stat().st_size
            )
            with (
                patch("everbench.api._session", return_value=SimpleNamespace()),
                patch("everbench.api.store.task_archive", return_value=manifest),
                patch("everbench.api.store.model_registration") as model_registration,
                create_app().test_client() as client,
            ):
                response = client.post(
                    "/api/tasks/dummy/backtest",
                    data={"model": (io.BytesIO(payload), "model.pkl"), "archive_sha256": "archive"},
                    headers={
                        "X-API-Key": "test-api-key",
                        "X-Everbench-Artifact-Signature": signature,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsNotNone(body)
        assert body is not None
        self.assertEqual(body["predictions"], 1)
        self.assertEqual(body["labels"], 1)
        self.assertIn("timing_seconds", body)
        model_registration.assert_not_called()


if __name__ == "__main__":
    unittest.main()
