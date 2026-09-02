from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from uuid import uuid4

from river import metrics
from sqlalchemy import select

from everbench import archive, artifacts, store
from everbench.config import CONFIG
from everbench.db import make_engine, make_session_factory
from everbench.schema import ArchiveManifest, BenchmarkEvent, BenchmarkLabel, ModelRegistration, Prediction
from everbench.workers import learn_once


class BrokenModel:
    def predict_one(self, features: dict[str, float]) -> float:
        del features
        raise RuntimeError("intentional test failure")


class WorkingModel:
    def predict_one(self, features: dict[str, float]) -> float:
        del features
        return 0.5


@unittest.skipUnless(os.getenv("EVERBENCH_RUN_POSTGRES_TESTS") == "1", "requires EVERBENCH_RUN_POSTGRES_TESTS=1")
class PostgresLifecycleTest(unittest.TestCase):
    """Exercise the transaction boundaries that unit tests cannot emulate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.url = os.getenv("EVERBENCH_TEST_DATABASE_URL") or os.environ["DATABASE_URL"]
        cls.sessions = make_session_factory(cls.url)

    @classmethod
    def tearDownClass(cls) -> None:
        make_engine(cls.url).dispose()

    def test_archive_removes_predictions_before_events(self) -> None:
        task_name = f"archive-test-{uuid4()}"
        event_id = "event"
        with self.sessions.begin() as session:
            store.add_events(
                session,
                task_name,
                [(event_id, (datetime.now(UTC) - timedelta(days=2)).timestamp(), {"value": 1.0})],
            )
            store.add_labels(session, task_name, [(event_id, 1, "test")], delay_seconds=None)
            session.add(Prediction(task_name=task_name, event_id=event_id, model_id="retired", prediction=0.5))

        with tempfile.TemporaryDirectory() as directory:
            original_config = archive.CONFIG
            archive.CONFIG = replace(
                CONFIG,
                archive_root=Path(directory),
                s3_bucket_name=None,
                archive_after_days=0,
                archive_batch_size=100,
            )
            try:
                self.assertEqual(
                    archive.archive_once(self.sessions, cast(ModuleType, SimpleNamespace(TASK_NAME=task_name))), 1
                )
            finally:
                archive.CONFIG = original_config

        with self.sessions() as session:
            self.assertIsNone(session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id}))
            self.assertIsNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id}))
            self.assertIsNone(
                session.get(Prediction, {"task_name": task_name, "event_id": event_id, "model_id": "retired"})
            )
            self.assertIsNotNone(session.scalar(select(ArchiveManifest).where(ArchiveManifest.task_name == task_name)))

    def test_failed_model_does_not_block_healthy_model(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "postgres-test-signing-key"
        task_name = f"model-test-{uuid4()}"
        task = SimpleNamespace(TASK_NAME=task_name, PROBLEM_TYPE="binary_classification", METRICS=(metrics.Accuracy(),))
        with self.sessions.begin() as session:
            for model_id, model in (("broken", BrokenModel()), ("working", WorkingModel())):
                payload = artifacts.dumps(model)
                artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
                store.register_model(session, task_name, model_id, "test", artifact.artifact_id)
            store.add_events(session, task_name, [("event", datetime.now(UTC).timestamp(), {"value": 1.0})])

        with self.sessions.begin() as session:
            learn_once(session, cast(ModuleType, task))

        with self.sessions() as session:
            broken = session.get(ModelRegistration, {"task_name": task_name, "model_id": "broken"})
            self.assertIsNotNone(broken)
            assert broken is not None
            self.assertEqual(broken.failure_count, 1)
            self.assertTrue(broken.active)
            self.assertIsNotNone(
                session.get(Prediction, {"task_name": task_name, "event_id": "event", "model_id": "working"})
            )
            leaderboard = {row["model_id"]: row for row in store.task_leaderboard(session, task_name)}
            self.assertGreater(leaderboard["working"]["model_bytes"], 0)
            self.assertIsNotNone(leaderboard["working"]["created_at"])

    def test_stream_cursor_is_updated_atomically(self) -> None:
        task_name = f"cursor-test-{uuid4()}"
        with self.sessions.begin() as session:
            store.save_stream_cursor(session, task_name, "events", "first")
            store.save_stream_cursor(session, task_name, "events", "second")

        with self.sessions() as session:
            self.assertEqual(store.stream_cursor(session, task_name, "events"), "second")


if __name__ == "__main__":
    unittest.main()
