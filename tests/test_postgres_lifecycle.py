from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from river import metrics
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive, archive_store, artifacts, event_store, model_store, reporting
from everbench.auto import store as auto_store
from everbench.auto.service import _reset_generation_metrics
from everbench.config import CONFIG
from everbench.db import make_session_factory
from everbench.learner import learn_once
from everbench.metrics import MetricTracker
from everbench.schema import (
    ArchiveManifest,
    BenchmarkEvent,
    BenchmarkLabel,
    MetricState,
    ModelEventState,
    ModelRegistration,
    ReadyLabel,
)
from everbench.tasks import TaskDefinition


class BrokenModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        raise RuntimeError("intentional test failure")


class WorkingModel:
    def predict_one(self, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


@pytest.fixture(scope="module")
def sessions() -> Iterator[sessionmaker[Session]]:
    if os.getenv("EVERBENCH_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("requires EVERBENCH_RUN_POSTGRES_TESTS=1")
    url = os.getenv("EVERBENCH_TEST_DATABASE_URL") or os.environ["DATABASE_URL"]
    factory = make_session_factory(url)
    yield factory
    bind = factory.kw.get("bind")
    if bind is not None:
        bind.dispose()


@pytest.fixture(autouse=True)
def signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVERBENCH_MODEL_SIGNING_KEY", "postgres-test-signing-key")


def test_archive_removes_predictions_before_events(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_name = f"archive-test-{uuid4()}"
    event_id = "event"
    inserted_at = datetime.now(UTC) - timedelta(days=15)
    week_start = inserted_at.date() - timedelta(days=inserted_at.weekday())
    with sessions.begin() as session:
        event_store.add_events(
            session,
            task_name,
            [(event_id, (datetime.now(UTC) - timedelta(days=2)).timestamp(), {"value": 1.0})],
        )
        event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id})
        assert event is not None
        event.inserted_at = inserted_at
        event_store.add_labels(session, task_name, [event_store.LabelInput(event_id, 1, "test")], delay_seconds=None)
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
        registration, _ = model_store.register_model(session, task_name, "retired", "test", artifact.artifact_id)
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

    monkeypatch.setattr(
        archive,
        "CONFIG",
        replace(
            CONFIG,
            archive_root=tmp_path,
            s3_bucket_name=None,
            archive_after_days=0,
        ),
    )

    assert archive.archive_once(sessions, cast(TaskDefinition, SimpleNamespace(TASK_NAME=task_name))) == 1
    assert archive.archive_once(sessions, cast(TaskDefinition, SimpleNamespace(TASK_NAME=task_name))) == 0

    with sessions() as session:
        assert session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id}) is None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id}) is None
        assert (
            session.get(
                ModelEventState,
                {"task_name": task_name, "event_id": event_id, "model_id": "retired"},
            )
            is None
        )
        manifests = list(session.scalars(select(ArchiveManifest).where(ArchiveManifest.task_name == task_name)))
        assert len(manifests) == 1
        assert manifests[0].row_count == 1
        assert archive_store.archive_for_week(session, task_name, manifests[0].event_date) == manifests[0]
        assert archive_store.latest_complete_archive_week(session, task_name, datetime.now(UTC)) == week_start
        assert not archive_store.record_archive(
            session,
            "f" * 64,
            task_name,
            manifests[0].event_date,
            "different.parquet",
            1,
            1,
        )


def test_task_stats_include_archives_and_exclude_orphan_labels(sessions: sessionmaker[Session]) -> None:
    task_name = f"stats-test-{uuid4()}"
    with sessions.begin() as session:
        event_store.add_events(session, task_name, [("live", datetime.now(UTC).timestamp(), {"value": 1.0})])
        event_store.add_labels(
            session,
            task_name,
            [event_store.LabelInput("live", 1, "test"), event_store.LabelInput("orphan", 1, "test")],
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

    with sessions() as session:
        assert reporting.task_stats(session, task_name) == {"events": 8, "labels": 8}


def test_task_names_are_registered_once_and_sorted(sessions: sessionmaker[Session]) -> None:
    prefix = f"task-names-{uuid4()}"
    first, second = (f"{prefix}-{suffix}" for suffix in ("a", "b"))
    with sessions.begin() as session:
        reporting.register_tasks(session, [second, first])
        reporting.register_tasks(session, [first])

    with sessions() as session:
        names = reporting.task_names(session)

    assert names == sorted(set(names))
    assert {first, second} <= set(names)


def test_failed_model_does_not_block_healthy_model(sessions: sessionmaker[Session]) -> None:
    task_name = f"model-test-{uuid4()}"
    task = SimpleNamespace(
        TASK_NAME=task_name,
        PROBLEM_TYPE="binary_classification",
        METRICS=(metrics.Accuracy(),),
        metric_inputs_for=None,
    )
    with sessions.begin() as session:
        for model_id, model in (("broken", BrokenModel()), ("working", WorkingModel())):
            payload = artifacts.dumps(model)
            artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
            model_store.register_model(session, task_name, model_id, "test", artifact.artifact_id)
        event_store.add_events(session, task_name, [("event", datetime.now(UTC).timestamp(), {"value": 1.0})])

    with sessions.begin() as session:
        learn_once(session, cast(TaskDefinition, task))

    with sessions() as session:
        broken = session.get(ModelRegistration, {"task_name": task_name, "model_id": "broken"})
        assert broken is not None
        assert broken.failure_count == 1
        assert broken.active
        assert broken.disabled_until is not None
        assert (
            session.get(
                ModelEventState,
                {"task_name": task_name, "event_id": "event", "model_id": "working"},
            )
            is not None
        )
        leaderboard = {row["model_id"]: row for row in reporting.task_leaderboard(session, task_name)}
        assert leaderboard["working"]["model_bytes"] > 0
        assert leaderboard["working"]["created_at"] is not None
        assert leaderboard["broken"]["error_count"] == 1
        assert leaderboard["broken"]["skipped"] == 1

    with sessions.begin() as session:
        event_store.add_labels(session, task_name, [event_store.LabelInput("event", 1, "test")], delay_seconds=None)
        learn_once(session, cast(TaskDefinition, task))

    with sessions() as session:
        leaderboard = {row["model_id"]: row for row in reporting.task_leaderboard(session, task_name)}
        assert leaderboard["broken"]["error_count"] == 1
        assert leaderboard["broken"]["skipped"] == 2
        checkpoint = model_store.latest_snapshot(session, task_name, "broken")
        assert checkpoint is not None
        assert checkpoint.checkpoint_label_available_at is not None
        assert event_store.completed_labelled_events(session, task_name, ["event"]) == ["event"]


def test_deleted_model_ids_start_fresh_registrations(sessions: sessionmaker[Session]) -> None:
    task_name = f"retired-model-test-{uuid4()}"
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
        model_store.register_model(session, task_name, "original", "test", artifact.artifact_id)
        assert model_store.delete_model(session, task_name, "original")

    with sessions() as session:
        assert reporting.task_leaderboard(session, task_name) == []
        registration, created = model_store.register_model(session, task_name, "original", "test", artifact.artifact_id)
        assert created
        assert registration.model_id == "original"


def test_model_detail_uses_autonomous_candidate_source(sessions: sessionmaker[Session]) -> None:
    task_name = f"auto-source-test-{uuid4()}"
    source = "def build_model():\n    return None\n"
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(
            session,
            payload,
            artifacts.sign(payload),
            {"source_code": source},
        )
        model_store.register_model(session, task_name, "auto", "test", artifact.artifact_id)

    with sessions() as session:
        detail = reporting.model_detail(session, task_name, "auto")

    assert detail is not None
    assert detail["class_definition"] == source


def test_auto_promotion_starts_fresh_generation_metrics(sessions: sessionmaker[Session]) -> None:
    task_name = f"auto-metric-generation-test-{uuid4()}"
    task = cast(
        TaskDefinition,
        SimpleNamespace(
            TASK_NAME=task_name,
            PROBLEM_TYPE="binary_classification",
            METRICS=(metrics.Accuracy(),),
        ),
    )
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
        model_store.register_model(session, task_name, "auto", "test", artifact.artifact_id)
        event_store.add_events(
            session,
            task_name,
            [
                ("old-pending", datetime.now(UTC).timestamp(), {"value": 1.0}),
                ("old-scored", datetime.now(UTC).timestamp(), {"value": 2.0}),
            ],
        )
        event_store.add_predictions(
            session,
            task_name,
            "auto",
            [("old-pending", 0.5), ("old-scored", 0.5)],
        )
        event_store.add_metric_updates(session, task_name, "auto", ["old-scored"])
        tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS, predictions=2)
        tracker.update(True, True)
        model_store.save_metric_state(
            session,
            task_name,
            "auto",
            tracker.definition,
            tracker.payload(),
            tracker.predictions,
            tracker.observations,
            tracker.values(),
        )

        _reset_generation_metrics(session, task, "auto")

    with sessions() as session:
        state = session.get(MetricState, {"task_name": task_name, "model_id": "auto"})
        assert state is not None
        assert state.predictions == 0
        assert state.observations == 0
        assert state.values == {"Accuracy": 0.0}
        rows = session.scalars(
            select(ModelEventState).where(
                ModelEventState.task_name == task_name,
                ModelEventState.model_id == "auto",
            )
        ).all()
        assert all(row.evaluated_at is not None for row in rows)


def test_auto_model_detail_updates_from_research_without_changing_champion(sessions: sessionmaker[Session]) -> None:
    task_name = f"auto-documentation-test-{uuid4()}"
    source = "def build_model():\n    return None\n"
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(
            session,
            payload,
            artifacts.sign(payload),
            {"source": "auto-bootstrap", "generation": 0, "source_code": source},
        )
        model_store.register_model(session, task_name, "auto", "everbench-auto", artifact.artifact_id)
        model_store.save_pickle_snapshot(session, task_name, "auto", payload, None, None, None)

    with sessions() as session:
        detail = reporting.model_detail(session, task_name, "auto")
        assert detail is not None
        assert "generation 0" in detail["class_definition"]
        assert "No research rounds have run yet" in detail["class_definition"]
        champion = model_store.artifact(session, artifact.artifact_id)
        assert champion is not None
        assert champion.metadata_["source"] == "auto-bootstrap"

    # Artifacts checkpointed before source metadata was preserved still carry
    # enough ownership and generation metadata to be recognized as autonomous.
    with sessions.begin() as session:
        champion = model_store.artifact(session, artifact.artifact_id)
        assert champion is not None
        champion.metadata_ = {**champion.metadata_, "source": "worker-snapshot"}

    with sessions.begin() as session:
        experiment = auto_store.begin_experiment(
            session,
            task_name,
            "auto",
            0,
            "researcher",
            {},
            1,
            10,
            artifact.artifact_id,
        )
        session.flush()
        experiment_id = experiment.experiment_id

    with sessions() as session:
        detail = reporting.model_detail(session, task_name, "auto")
        assert detail is not None
        assert "1 running" in detail["class_definition"]
        assert "exploring candidates against generation 0" in detail["class_definition"]

    with sessions.begin() as session:
        experiment = auto_store.latest_experiment(session, task_name, "auto")
        assert experiment is not None and experiment.experiment_id == experiment_id
        auto_store.finish_experiment(experiment, status="rejected", hypothesis="Try richer features.")

    with sessions() as session:
        detail = reporting.model_detail(session, task_name, "auto")
        assert detail is not None
        assert "1 rejected" in detail["class_definition"]
        assert "Try richer features" in detail["class_definition"]
        assert "No research round is currently running" in detail["class_definition"]
        assert detail["class_definition"].endswith(source)
        champion = model_store.artifact(session, artifact.artifact_id)
        assert champion is not None
        assert champion.metadata_["source"] == "worker-snapshot"
        assert champion.metadata_["source_code"] == source


def test_event_completion_requires_a_model_checkpoint(sessions: sessionmaker[Session]) -> None:
    task_name = f"checkpoint-test-{uuid4()}"
    event_id = "event"
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
        registration, _ = model_store.register_model(session, task_name, "model", "test", artifact.artifact_id)
        event_store.add_events(session, task_name, [(event_id, datetime.now(UTC).timestamp(), {"value": 1.0})])
        event_store.add_labels(session, task_name, [event_store.LabelInput(event_id, 1, "test")], delay_seconds=None)
        event_store.add_prediction_skips(session, task_name, "model", [event_id])
        event_store.add_trainings(session, task_name, "model", [event_id])

        assert event_store.completed_labelled_events(session, task_name, [event_id]) == []

        label_record = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id})
        event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id})
        ready = session.get(ReadyLabel, {"task_name": task_name, "event_id": event_id})
        assert label_record is not None and event is not None and ready is not None
        model_store.advance_model_checkpoint(
            session,
            task_name,
            registration,
            label_record.available_at,
            event.sequence,
            ready.sequence,
        )
        assert event_store.completed_labelled_events(session, task_name, [event_id]) == [event_id]


def test_archive_purge_accepts_more_than_postgres_parameter_limit(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        archive_store.purge_archived_events(
            session,
            f"large-archive-purge-test-{uuid4()}",
            [f"event-{index}" for index in range(65_536)],
        )


def test_stream_cursor_is_updated_atomically(sessions: sessionmaker[Session]) -> None:
    task_name = f"cursor-test-{uuid4()}"
    with sessions.begin() as session:
        event_store.save_stream_cursor(session, task_name, "events", "first")
        event_store.save_stream_cursor(session, task_name, "events", "second")

    with sessions() as session:
        assert event_store.stream_cursor(session, task_name, "events") == "second"


def test_duplicate_old_event_does_not_erase_an_existing_positive_label(sessions: sessionmaker[Session]) -> None:
    task_name = f"duplicate-label-test-{uuid4()}"
    timestamp = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    with sessions.begin() as session:
        event_store.add_events(session, task_name, [("event", timestamp, {"value": 1.0})])
        event_store.add_labels(session, task_name, [event_store.LabelInput("event", 1, "positive")], delay_seconds=None)

    with sessions.begin() as session:
        inserted = event_store.add_events(
            session,
            task_name,
            [("event", timestamp, {"value": 999.0})],
            delay_seconds=60,
        )
        assert inserted == []

    with sessions() as session:
        label = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "event"})
        assert label is not None
        assert label.y == 1


def test_positive_horizon_uses_source_availability_time(sessions: sessionmaker[Session]) -> None:
    task_name = f"source-time-label-test-{uuid4()}"
    event_time = datetime.now(UTC) - timedelta(days=4)
    with sessions.begin() as session:
        event_store.add_events(
            session,
            task_name,
            [
                ("timely", event_time.timestamp(), {"value": 1.0}),
                ("late", event_time.timestamp(), {"value": 2.0}),
            ],
        )
        inserted = event_store.add_labels(
            session,
            task_name,
            [
                event_store.LabelInput("timely", 1, "positive", event_time + timedelta(hours=47)),
                event_store.LabelInput("late", 1, "positive", event_time + timedelta(hours=49)),
            ],
            delay_seconds=48 * 60 * 60,
        )
        assert inserted == ["timely"]

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}) is None


def test_pending_positive_horizon_uses_source_availability_time(sessions: sessionmaker[Session]) -> None:
    task_name = f"pending-source-time-label-test-{uuid4()}"
    event_time = datetime.now(UTC) - timedelta(days=4)
    with sessions.begin() as session:
        event_store.add_labels(
            session,
            task_name,
            [
                event_store.LabelInput("timely", 1, "positive", event_time + timedelta(hours=47)),
                event_store.LabelInput("late", 1, "positive", event_time + timedelta(hours=49)),
            ],
            delay_seconds=48 * 60 * 60,
        )
        event_store.add_events(
            session,
            task_name,
            [
                ("timely", event_time.timestamp(), {"value": 1.0}),
                ("late", event_time.timestamp(), {"value": 2.0}),
            ],
            delay_seconds=48 * 60 * 60,
        )

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}) is None


def test_orphan_label_retention_does_not_remove_matched_labels(sessions: sessionmaker[Session]) -> None:
    task_name = f"orphan-label-test-{uuid4()}"
    cutoff = datetime.now(UTC) + timedelta(seconds=1)
    with sessions.begin() as session:
        event_store.add_events(session, task_name, [("matched", datetime.now(UTC).timestamp(), {"value": 1.0})])
        event_store.add_labels(
            session,
            task_name,
            [event_store.LabelInput("matched", 1, "test"), event_store.LabelInput("orphan", 1, "test")],
            delay_seconds=None,
        )
        assert event_store.purge_orphan_labels(session, task_name, cutoff) == 1

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "matched"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "orphan"}) is None


def test_deleting_registration_cascades_all_model_event_state(sessions: sessionmaker[Session]) -> None:
    task_name = f"delete-cascade-test-{uuid4()}"
    with sessions.begin() as session:
        payload = artifacts.dumps(WorkingModel())
        artifact = model_store.store_artifact(session, payload, artifacts.sign(payload), {})
        model_store.register_model(session, task_name, "model", "test", artifact.artifact_id)
        session.flush()
        event_store.add_events(session, task_name, [("event", datetime.now(UTC).timestamp(), {"value": 1.0})])
        event_store.add_predictions(session, task_name, "model", [("event", 0.5)])

    with sessions.begin() as session:
        assert model_store.delete_model(session, task_name, "model")

    with sessions() as session:
        assert (
            session.get(
                ModelEventState,
                {"task_name": task_name, "event_id": "event", "model_id": "model"},
            )
            is None
        )
