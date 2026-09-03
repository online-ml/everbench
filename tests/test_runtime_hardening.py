from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from sqlalchemy.orm import Session

from everbench import artifacts
from everbench.api import task_source_url, validation_examples
from everbench.batching import TimedBatch
from everbench.models import PickledModel, prediction_for, validate_model
from everbench.tasks import TaskDefinition
from everbench.workers import _load_model


class ConstantModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


class EventAwareModel:
    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event
        return {False: float(event_id != "event-1"), True: float(event_id == "event-1")}


class LegacyEventModel:
    def predict_event(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


class MulticlassModel:
    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[str, float]:
        del event_id, event
        return {"first": 0.2, "second": 0.8}


class ScoreOnlyModel:
    def score_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id
        return float(event["score"])


class BrokenProbabilityModel:
    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event_id, event
        raise AttributeError("internal typo")

    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.25


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
            model, snapshot = _load_model(cast(Session, None), cast(TaskDefinition, task), registration)
        self.assertIsNone(snapshot)
        self.assertEqual(model.predict_one("event", {}), 0.5)

    def test_task_source_url_links_to_the_checked_in_definition(self) -> None:
        task = SimpleNamespace(__file__=Path("tasks/dummy/task.py").resolve())

        self.assertEqual(task_source_url(task), "https://github.com/online-ml/everbench/blob/main/tasks/dummy/task.py")

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

        self.assertEqual(validate_model(cast(TaskDefinition, task), PickledModel("constant", ConstantModel()), []), 0)

    def test_prediction_passes_event_id_to_standard_prediction_method(self) -> None:
        task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="binary_classification"))
        model = PickledModel("event-aware", EventAwareModel())

        self.assertEqual(prediction_for(task, model, "event-1", {"x": 1.0}), 1.0)
        self.assertEqual(prediction_for(task, model, "event-2", {"x": 1.0}), 0.0)

    def test_multiclass_prediction_keeps_probability_mapping(self) -> None:
        task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="multiclass_classification"))
        model = PickledModel("multiclass", MulticlassModel())

        self.assertEqual(prediction_for(task, model, "event-1", {"x": 1.0}), {"first": 0.2, "second": 0.8})

    def test_legacy_predict_event_protocol_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "predict_one"):
            PickledModel("legacy", LegacyEventModel())

    def test_score_only_anomaly_model_is_accepted(self) -> None:
        task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="anomaly_detection"))
        model = PickledModel("anomaly", ScoreOnlyModel())

        self.assertEqual(prediction_for(task, model, "event-1", {"score": 0.3}), 0.3)

    def test_internal_attribute_error_does_not_trigger_prediction_fallback(self) -> None:
        task = cast(TaskDefinition, SimpleNamespace(PROBLEM_TYPE="binary_classification"))
        model = PickledModel("broken-probability", BrokenProbabilityModel())

        with self.assertRaisesRegex(AttributeError, "internal typo"):
            prediction_for(task, model, "event-1", {})


if __name__ == "__main__":
    unittest.main()
