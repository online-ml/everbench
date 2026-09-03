from __future__ import annotations

import os
import time
import unittest
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import patch

from sqlalchemy.orm import Session

from everbench import artifacts
from everbench.api import validation_examples
from everbench.batching import TimedBatch
from everbench.models import validate_uploaded_model
from everbench.workers import _load_model


class ConstantModel:
    def predict_one(self, features: dict[str, float]) -> float:
        del features
        return 0.5


class RuntimeHardeningTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "test-signing-key"

    def test_model_load_uses_the_task_name_for_snapshots(self) -> None:
        payload = artifacts.dumps(ConstantModel())
        artifact = SimpleNamespace(payload=payload, signature=artifacts.sign(payload))
        registration = SimpleNamespace(model_id="constant", artifact_id="artifact")
        task = SimpleNamespace(TASK_NAME="example")
        with (
            patch("everbench.workers.store.latest_snapshot", return_value=None),
            patch("everbench.workers.store.artifact", return_value=artifact),
        ):
            model, snapshot = _load_model(cast(Session, None), cast(ModuleType, task), registration)
        self.assertIsNone(snapshot)
        self.assertEqual(model.predict_one({}), 0.5)

    def test_idle_batch_flushes_without_another_source_item(self) -> None:
        flushed: list[list[str]] = []
        batch = TimedBatch(100, 0.01, lambda items: flushed.append(list(items)))
        batch.add("one")
        time.sleep(0.02)
        self.assertTrue(batch.flush_if_due())
        self.assertEqual(flushed, [["one"]])

    def test_upload_validation_uses_recent_postgres_labels_before_archives(self) -> None:
        recent = [("one", {"x": 1.0}, 1)] * 5
        with (
            patch("everbench.api.store.latest_labelled_examples", return_value=recent),
            patch("everbench.api.archive.latest_labelled_examples") as archived,
        ):
            self.assertEqual(validation_examples(cast(Session, SimpleNamespace()), "task"), recent)
        archived.assert_not_called()

    def test_upload_validation_combines_archived_and_recent_labels(self) -> None:
        recent = [("recent", {"x": 1.0}, 1)] * 2
        archived_rows = [("old", {"x": 0.0}, 0)] * 3
        session = cast(Session, SimpleNamespace())
        with (
            patch("everbench.api.store.latest_labelled_examples", return_value=recent),
            patch("everbench.api.store.task_archives", return_value=["manifest"]),
            patch("everbench.api.archive.latest_labelled_examples", return_value=archived_rows),
        ):
            self.assertEqual(validation_examples(session, "task"), archived_rows + recent)

    def test_upload_validation_accepts_a_task_without_labels(self) -> None:
        task = SimpleNamespace(PROBLEM_TYPE="binary_classification", METRICS=())
        payload = artifacts.dumps(ConstantModel())

        self.assertEqual(validate_uploaded_model(cast(ModuleType, task), payload, artifacts.sign(payload), []), 0)


if __name__ == "__main__":
    unittest.main()
