from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from river import metrics
from sqlalchemy import select

from everbench import archive, artifacts, store
from everbench.config import CONFIG
from everbench.db import make_engine, make_session_factory
from everbench.schema import ArchiveManifest, BenchmarkEvent, BenchmarkLabel, ModelEventState, ModelRegistration
from everbench.tasks import TaskDefinition
from everbench.workers import learn_once


class BrokenModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        raise RuntimeError("intentional test failure")


class WorkingModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
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
            store.add_labels(session, task_name, [store.LabelInput(event_id, 1, "test")], delay_seconds=None)
            payload = artifacts.dumps(WorkingModel())
            artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
            registration, _ = store.register_model(session, task_name, "retired", "test", artifact.artifact_id)
            registration.active = False
            session.flush()
            session.add(
                ModelEventState(
                    task_name=task_name,
                    event_id=event_id,
                    model_id="retired",
                    prediction=0.5,
                    prediction_status="predicted",
                )
            )

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
                    archive.archive_once(self.sessions, cast(TaskDefinition, SimpleNamespace(TASK_NAME=task_name))), 1
                )
            finally:
                archive.CONFIG = original_config

        with self.sessions() as session:
            self.assertIsNone(session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id}))
            self.assertIsNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id}))
            self.assertIsNone(
                session.get(ModelEventState, {"task_name": task_name, "event_id": event_id, "model_id": "retired"})
            )
            self.assertIsNotNone(session.scalar(select(ArchiveManifest).where(ArchiveManifest.task_name == task_name)))

    def test_task_stats_include_archives_and_exclude_orphan_labels(self) -> None:
        task_name = f"stats-test-{uuid4()}"
        with self.sessions.begin() as session:
            store.add_events(session, task_name, [("live", datetime.now(UTC).timestamp(), {"value": 1.0})])
            store.add_labels(
                session,
                task_name,
                [store.LabelInput("live", 1, "test"), store.LabelInput("orphan", 1, "test")],
                delay_seconds=None,
            )
            session.add(
                ArchiveManifest(
                    content_sha256=uuid4().hex,
                    task_name=task_name,
                    event_date=datetime.now(UTC).date(),
                    path="test.parquet",
                    row_count=7,
                    byte_size=1,
                )
            )

        with self.sessions() as session:
            self.assertEqual(store.task_stats(session, task_name), {"events": 8, "labels": 8})

    def test_failed_model_does_not_block_healthy_model(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "postgres-test-signing-key"
        task_name = f"model-test-{uuid4()}"
        task = SimpleNamespace(
            TASK_NAME=task_name,
            PROBLEM_TYPE="binary_classification",
            METRICS=(metrics.Accuracy(),),
            metric_inputs_for=None,
        )
        with self.sessions.begin() as session:
            for model_id, model in (("broken", BrokenModel()), ("working", WorkingModel())):
                payload = artifacts.dumps(model)
                artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
                store.register_model(session, task_name, model_id, "test", artifact.artifact_id)
            store.add_events(session, task_name, [("event", datetime.now(UTC).timestamp(), {"value": 1.0})])

        with self.sessions.begin() as session:
            learn_once(session, cast(TaskDefinition, task))

        with self.sessions() as session:
            broken = session.get(ModelRegistration, {"task_name": task_name, "model_id": "broken"})
            self.assertIsNotNone(broken)
            assert broken is not None
            self.assertEqual(broken.failure_count, 1)
            self.assertTrue(broken.active)
            self.assertIsNotNone(broken.disabled_until)
            self.assertIsNotNone(
                session.get(ModelEventState, {"task_name": task_name, "event_id": "event", "model_id": "working"})
            )
            leaderboard = {row["model_id"]: row for row in store.task_leaderboard(session, task_name)}
            self.assertGreater(leaderboard["working"]["model_bytes"], 0)
            self.assertIsNotNone(leaderboard["working"]["created_at"])
            self.assertEqual(leaderboard["broken"]["error_count"], 1)
            self.assertEqual(leaderboard["broken"]["skipped"], 1)

        with self.sessions.begin() as session:
            store.add_labels(session, task_name, [store.LabelInput("event", 1, "test")], delay_seconds=None)
            learn_once(session, cast(TaskDefinition, task))

        with self.sessions() as session:
            leaderboard = {row["model_id"]: row for row in store.task_leaderboard(session, task_name)}
            self.assertEqual(leaderboard["broken"]["error_count"], 1)
            self.assertEqual(leaderboard["broken"]["skipped"], 2)
            checkpoint = store.latest_snapshot(session, task_name, "broken")
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertIsNotNone(checkpoint.checkpoint_label_available_at)
            self.assertEqual(store.completed_labelled_events(session, task_name, ["event"]), ["event"])

    def test_deleted_model_ids_start_fresh_registrations(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "postgres-test-signing-key"
        task_name = f"retired-model-test-{uuid4()}"
        with self.sessions.begin() as session:
            payload = artifacts.dumps(WorkingModel())
            artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
            store.register_model(session, task_name, "original", "test", artifact.artifact_id)
            self.assertTrue(store.delete_model(session, task_name, "original"))

        with self.sessions() as session:
            self.assertEqual(store.task_leaderboard(session, task_name), [])
            registration, created = store.register_model(session, task_name, "original", "test", artifact.artifact_id)
            self.assertTrue(created)
            self.assertEqual(registration.model_id, "original")

    def test_event_completion_requires_a_model_checkpoint(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "postgres-test-signing-key"
        task_name = f"checkpoint-test-{uuid4()}"
        event_id = "event"
        with self.sessions.begin() as session:
            payload = artifacts.dumps(WorkingModel())
            artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
            registration, _ = store.register_model(session, task_name, "model", "test", artifact.artifact_id)
            store.add_events(session, task_name, [(event_id, datetime.now(UTC).timestamp(), {"value": 1.0})])
            store.add_labels(session, task_name, [store.LabelInput(event_id, 1, "test")], delay_seconds=None)
            store.add_prediction_skips(session, task_name, "model", [event_id])
            store.add_trainings(session, task_name, "model", [event_id])

            self.assertEqual(store.completed_labelled_events(session, task_name, [event_id]), [])

            label_record = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id})
            event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id})
            assert label_record is not None and event is not None
            store.advance_model_checkpoint(session, task_name, registration, label_record.available_at, event.sequence)
            self.assertEqual(store.completed_labelled_events(session, task_name, [event_id]), [event_id])

    def test_stream_cursor_is_updated_atomically(self) -> None:
        task_name = f"cursor-test-{uuid4()}"
        with self.sessions.begin() as session:
            store.save_stream_cursor(session, task_name, "events", "first")
            store.save_stream_cursor(session, task_name, "events", "second")

        with self.sessions() as session:
            self.assertEqual(store.stream_cursor(session, task_name, "events"), "second")

    def test_duplicate_old_event_does_not_erase_an_existing_positive_label(self) -> None:
        task_name = f"duplicate-label-test-{uuid4()}"
        timestamp = (datetime.now(UTC) - timedelta(days=2)).timestamp()
        with self.sessions.begin() as session:
            store.add_events(session, task_name, [("event", timestamp, {"value": 1.0})])
            # This represents a positive that was accepted while it was timely.
            store.add_labels(session, task_name, [store.LabelInput("event", 1, "positive")], delay_seconds=None)

        with self.sessions.begin() as session:
            inserted = store.add_events(
                session,
                task_name,
                [("event", timestamp, {"value": 999.0})],
                delay_seconds=60,
            )
            self.assertEqual(inserted, [])

        with self.sessions() as session:
            label = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "event"})
            self.assertIsNotNone(label)
            assert label is not None
            self.assertEqual(label.y, 1)

    def test_positive_horizon_uses_source_availability_time(self) -> None:
        task_name = f"source-time-label-test-{uuid4()}"
        event_time = datetime.now(UTC) - timedelta(days=4)
        with self.sessions.begin() as session:
            store.add_events(
                session,
                task_name,
                [
                    ("timely", event_time.timestamp(), {"value": 1.0}),
                    ("late", event_time.timestamp(), {"value": 2.0}),
                ],
            )
            inserted = store.add_labels(
                session,
                task_name,
                [
                    store.LabelInput("timely", 1, "positive", event_time + timedelta(hours=47)),
                    store.LabelInput("late", 1, "positive", event_time + timedelta(hours=49)),
                ],
                delay_seconds=48 * 60 * 60,
            )
            self.assertEqual(inserted, ["timely"])

        with self.sessions() as session:
            self.assertIsNotNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}))
            self.assertIsNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}))

    def test_pending_positive_horizon_uses_source_availability_time(self) -> None:
        task_name = f"pending-source-time-label-test-{uuid4()}"
        event_time = datetime.now(UTC) - timedelta(days=4)
        with self.sessions.begin() as session:
            store.add_labels(
                session,
                task_name,
                [
                    store.LabelInput("timely", 1, "positive", event_time + timedelta(hours=47)),
                    store.LabelInput("late", 1, "positive", event_time + timedelta(hours=49)),
                ],
                delay_seconds=48 * 60 * 60,
            )
            store.add_events(
                session,
                task_name,
                [
                    ("timely", event_time.timestamp(), {"value": 1.0}),
                    ("late", event_time.timestamp(), {"value": 2.0}),
                ],
                delay_seconds=48 * 60 * 60,
            )

        with self.sessions() as session:
            self.assertIsNotNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}))
            self.assertIsNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}))

    def test_orphan_label_retention_does_not_remove_matched_labels(self) -> None:
        task_name = f"orphan-label-test-{uuid4()}"
        cutoff = datetime.now(UTC) + timedelta(seconds=1)
        with self.sessions.begin() as session:
            store.add_events(session, task_name, [("matched", datetime.now(UTC).timestamp(), {"value": 1.0})])
            store.add_labels(
                session,
                task_name,
                [store.LabelInput("matched", 1, "test"), store.LabelInput("orphan", 1, "test")],
                delay_seconds=None,
            )
            self.assertEqual(store.purge_orphan_labels(session, task_name, cutoff), 1)

        with self.sessions() as session:
            self.assertIsNotNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "matched"}))
            self.assertIsNone(session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "orphan"}))

    def test_deleting_registration_cascades_all_model_event_state(self) -> None:
        os.environ["EVERBENCH_MODEL_SIGNING_KEY"] = "postgres-test-signing-key"
        task_name = f"delete-cascade-test-{uuid4()}"
        with self.sessions.begin() as session:
            payload = artifacts.dumps(WorkingModel())
            artifact = store.store_artifact(session, payload, artifacts.sign(payload), {})
            store.register_model(session, task_name, "model", "test", artifact.artifact_id)
            session.flush()
            store.add_events(session, task_name, [("event", datetime.now(UTC).timestamp(), {"value": 1.0})])
            store.add_predictions(session, task_name, "model", [("event", 0.5)])

        with self.sessions.begin() as session:
            self.assertTrue(store.delete_model(session, task_name, "model"))

        with self.sessions() as session:
            self.assertIsNone(
                session.get(ModelEventState, {"task_name": task_name, "event_id": "event", "model_id": "model"})
            )


if __name__ == "__main__":
    unittest.main()
