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
from everbench.records import Observation
from everbench.schema import (
    ArchiveManifest,
    BenchmarkEvent,
    BenchmarkLabel,
    MetricState,
    ModelEventState,
    ModelRegistration,
    ReadyLabel,
    TaskRegistration,
)
from everbench.tasks import LabelPolicy, TaskDefinition, load_task


class BrokenModel:
    def predict_one(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        raise RuntimeError("intentional test failure")


class WorkingModel:
    def predict_one(self, *, event_id: str, event: dict[str, Any]) -> float:
        del event_id, event
        return 0.5


class CountingModel:
    def __init__(self) -> None:
        self.learned = []

    def predict_one(self, *, event_id: str, event: dict) -> float:
        return 0.0

    def learn_one(self, *, event_id: str, event: dict, label: float) -> None:
        self.learned.append(event_id)


@pytest.fixture(scope="module")
def sessions() -> Iterator[sessionmaker[Session]]:
    if os.getenv("EVERBENCH_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("requires EVERBENCH_RUN_POSTGRES_TESTS=1")
    url = os.getenv("EVERBENCH_TEST_DATABASE_URL") or os.environ["DATABASE_URL"]
    factory = make_session_factory(url=url)
    yield factory
    bind = factory.kw.get("bind")
    if bind is not None:
        bind.dispose()


@pytest.fixture(autouse=True)
def signing_key(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVERBENCH_MODEL_SIGNING_KEY", "postgres-test-signing-key")


def test_archive_removes_predictions_before_events(
    *, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_name = f"archive-test-{uuid4()}"
    event_id = "event"
    inserted_at = datetime.now(UTC) - timedelta(days=15)
    week_start = inserted_at.date() - timedelta(days=inserted_at.weekday())
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(
                    event_id=event_id,
                    timestamp=(datetime.now(UTC) - timedelta(days=2)).timestamp(),
                    payload={"value": 1.0},
                )
            ],
        )
        event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id})
        assert event is not None
        event.inserted_at = inserted_at
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id=event_id, y=1, reason="test")],
            policy=None,
        )
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        registration, _ = model_store.register_model(
            session=session, task_name=task_name, model_id="retired", owner="test", artifact_id=artifact.artifact_id
        )
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

    task = cast(TaskDefinition, SimpleNamespace(TASK_NAME=task_name, label_policy=None))
    assert archive.archive_once(sessions=sessions, task=task) == 1
    assert archive.archive_once(sessions=sessions, task=task) == 0

    with sessions() as session:
        assert session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id}) is None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id}) is None
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 1, "labels": 1}
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
        assert (
            archive_store.archive_for_week(session=session, task_name=task_name, event_date=manifests[0].event_date)
            == manifests[0]
        )
        assert (
            archive_store.latest_complete_archive_week(session=session, task_name=task_name, cutoff=datetime.now(UTC))
            == week_start
        )
        assert not archive_store.record_archive(
            session=session,
            content_sha256="f" * 64,
            task_name=task_name,
            event_date=manifests[0].event_date,
            path="different.parquet",
            row_count=1,
            byte_size=1,
        )


def test_task_stats_include_archives_and_exclude_orphan_labels(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"stats-test-{uuid4()}"
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="live", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[
                event_store.LabelInput(event_id="live", y=1, reason="test"),
                event_store.LabelInput(event_id="orphan", y=1, reason="test"),
            ],
            policy=None,
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
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 8, "labels": 8}

    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="live", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="live", y=1, reason="test")],
            policy=None,
        )

    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 8, "labels": 8}


def test_task_names_are_registered_once_and_sorted(*, sessions: sessionmaker[Session]) -> None:
    prefix = f"task-names-{uuid4()}"
    first, second = (f"{prefix}-{suffix}" for suffix in ("a", "b"))
    with sessions.begin() as session:
        reporting.register_tasks(session=session, task_names=[second, first])
        reporting.register_tasks(session=session, task_names=[first])

    with sessions() as session:
        names = reporting.task_names(session=session)

    assert names == sorted(set(names))
    assert {first, second} <= set(names)


def test_task_counts_follow_committed_matched_events(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"counter-lifecycle-test-{uuid4()}"
    with sessions.begin() as session:
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="first", y=1, reason="early")],
            policy=None,
        )

    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 0, "labels": 0}

    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(event_id="first", timestamp=datetime.now(UTC).timestamp(), payload={}),
                Observation(event_id="second", timestamp=datetime.now(UTC).timestamp(), payload={}),
            ],
        )

    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 2, "labels": 1}

    with pytest.raises(RuntimeError, match="roll back"):
        with sessions.begin() as session:
            event_store.add_events(
                session=session,
                task_name=task_name,
                events=[Observation(event_id="third", timestamp=datetime.now(UTC).timestamp(), payload={})],
            )
            raise RuntimeError("roll back")

    with sessions.begin() as session:
        event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": "first"})
        assert event is not None
        session.delete(event)

    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 1, "labels": 0}
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "first"}) is not None


def test_task_stats_reflect_writes_in_the_same_session(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"counter-session-test-{uuid4()}"
    with sessions.begin() as session:
        reporting.register_tasks(session=session, task_names=[task_name])
        task = session.get(TaskRegistration, task_name)
        assert task is not None
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 0, "labels": 0}
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="event", timestamp=datetime.now(UTC).timestamp(), payload={})],
        )
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 1, "labels": 0}


def test_failed_model_does_not_block_healthy_model(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"model-test-{uuid4()}"
    task = SimpleNamespace(
        TASK_NAME=task_name,
        PROBLEM_TYPE="binary_classification",
        METRICS=(metrics.Accuracy(),),
        metric_inputs_for=None,
    )
    with sessions.begin() as session:
        for model_id, model in (("broken", BrokenModel()), ("working", WorkingModel())):
            payload = artifacts.dumps(model=model)
            artifact = model_store.store_artifact(
                session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
            )
            model_store.register_model(
                session=session, task_name=task_name, model_id=model_id, owner="test", artifact_id=artifact.artifact_id
            )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="event", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )

    with sessions.begin() as session:
        learn_once(session=session, task=cast(TaskDefinition, task))

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
        leaderboard = {row["model_id"]: row for row in reporting.task_leaderboard(session=session, task_name=task_name)}
        assert leaderboard["working"]["model_bytes"] > 0
        assert leaderboard["working"]["created_at"] is not None
        assert leaderboard["broken"]["error_count"] == 1
        assert leaderboard["broken"]["skipped"] == 1

    with sessions.begin() as session:
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="event", y=1, reason="test")],
            policy=None,
        )
        learn_once(session=session, task=cast(TaskDefinition, task))

    with sessions() as session:
        leaderboard = {row["model_id"]: row for row in reporting.task_leaderboard(session=session, task_name=task_name)}
        assert leaderboard["broken"]["error_count"] == 1
        assert leaderboard["broken"]["skipped"] == 2
        checkpoint = model_store.latest_snapshot(session=session, task_name=task_name, model_id="broken")
        assert checkpoint is not None
        assert checkpoint.checkpoint_ready_sequence is not None
        assert event_store.completed_labelled_events(session=session, task_name=task_name, event_ids=["event"]) == [
            "event"
        ]


def test_deleted_model_ids_start_fresh_registrations(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"retired-model-test-{uuid4()}"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        model_store.register_model(
            session=session, task_name=task_name, model_id="original", owner="test", artifact_id=artifact.artifact_id
        )
        assert model_store.delete_model(session=session, task_name=task_name, model_id="original")

    with sessions() as session:
        assert reporting.task_leaderboard(session=session, task_name=task_name) == []
        registration, created = model_store.register_model(
            session=session, task_name=task_name, model_id="original", owner="test", artifact_id=artifact.artifact_id
        )
        assert created
        assert registration.model_id == "original"


def test_model_detail_uses_autonomous_candidate_source(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"auto-source-test-{uuid4()}"
    source = "def build_model():\n    return None\n"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session,
            payload=payload,
            signature=artifacts.sign(payload=payload),
            metadata={"source_code": source},
        )
        model_store.register_model(
            session=session, task_name=task_name, model_id="auto", owner="test", artifact_id=artifact.artifact_id
        )

    with sessions() as session:
        detail = reporting.model_detail(session=session, task_name=task_name, model_id="auto")

    assert detail is not None
    assert detail["class_definition"] == source


def test_auto_promotion_starts_fresh_generation_metrics(*, sessions: sessionmaker[Session]) -> None:
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
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        model_store.register_model(
            session=session, task_name=task_name, model_id="auto", owner="test", artifact_id=artifact.artifact_id
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(event_id="old-pending", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0}),
                Observation(event_id="old-scored", timestamp=datetime.now(UTC).timestamp(), payload={"value": 2.0}),
            ],
        )
        event_store.add_predictions(
            session=session,
            task_name=task_name,
            model_id="auto",
            predictions={"old-pending": 0.5, "old-scored": 0.5},
        )
        event_store.add_metric_updates(session=session, task_name=task_name, model_id="auto", event_ids=["old-scored"])
        tracker = MetricTracker.fresh(problem_type=task.PROBLEM_TYPE, prototypes=task.METRICS, predictions=2)
        tracker.update(y_true=True, prediction=True)
        model_store.save_metric_state(
            session=session,
            task_name=task_name,
            model_id="auto",
            definition=tracker.definition,
            state=tracker.payload(),
            predictions=tracker.predictions,
            observations=tracker.observations,
            values=tracker.values(),
        )

        _reset_generation_metrics(session=session, task=task, model_id="auto")

    with sessions() as session:
        state = session.get(MetricState, {"task_name": task_name, "model_id": "auto"})
        assert state is not None
        assert state.predictions == 0
        assert state.observations == 0
        assert state.values == {"Accuracy": 0.0}


def test_auto_model_detail_updates_from_research_without_changing_champion(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"auto-documentation-test-{uuid4()}"
    source = "def build_model():\n    return None\n"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session,
            payload=payload,
            signature=artifacts.sign(payload=payload),
            metadata={"source": "auto-bootstrap", "generation": 0, "source_code": source},
        )
        model_store.register_model(
            session=session,
            task_name=task_name,
            model_id="auto",
            owner="everbench-auto",
            artifact_id=artifact.artifact_id,
        )
        model_store.save_pickle_snapshot(
            session=session, task_name=task_name, model_id="auto", payload=payload, checkpoint_ready_sequence=0
        )

    with sessions() as session:
        detail = reporting.model_detail(session=session, task_name=task_name, model_id="auto")
        assert detail is not None
        assert "generation 0" in detail["class_definition"]
        assert "No research rounds have run yet" in detail["class_definition"]
        champion = model_store.artifact(session=session, artifact_id=artifact.artifact_id)
        assert champion is not None
        assert champion.metadata_["source"] == "auto-bootstrap"

    # Artifacts checkpointed before source metadata was preserved still carry
    # enough ownership and generation metadata to be recognized as autonomous.
    with sessions.begin() as session:
        champion = model_store.artifact(session=session, artifact_id=artifact.artifact_id)
        assert champion is not None
        champion.metadata_ = {**champion.metadata_, "source": "worker-snapshot"}

    with sessions.begin() as session:
        experiment = auto_store.begin_experiment(
            session=session,
            task_name=task_name,
            model_id="auto",
            parent_generation=0,
            researcher="researcher",
            research_summary={},
            comparison_start_sequence=1,
            comparison_end_sequence=10,
            champion_artifact_id=artifact.artifact_id,
        )
        session.flush()
        experiment_id = experiment.experiment_id

    with sessions() as session:
        detail = reporting.model_detail(session=session, task_name=task_name, model_id="auto")
        assert detail is not None
        assert "1 running" in detail["class_definition"]
        assert "exploring candidates against generation 0" in detail["class_definition"]

    with sessions.begin() as session:
        experiment = auto_store.latest_experiment(session=session, task_name=task_name, model_id="auto")
        assert experiment is not None and experiment.experiment_id == experiment_id
        auto_store.finish_experiment(experiment=experiment, status="rejected", hypothesis="Try richer features.")

    with sessions() as session:
        detail = reporting.model_detail(session=session, task_name=task_name, model_id="auto")
        assert detail is not None
        assert "1 rejected" in detail["class_definition"]
        assert "Try richer features" in detail["class_definition"]
        assert "No research round is currently running" in detail["class_definition"]
        assert detail["class_definition"].endswith(source)
        champion = model_store.artifact(session=session, artifact_id=artifact.artifact_id)
        assert champion is not None
        assert champion.metadata_["source"] == "worker-snapshot"
        assert champion.metadata_["source_code"] == source


def test_event_completion_requires_a_model_checkpoint(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"checkpoint-test-{uuid4()}"
    event_id = "event"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        registration, _ = model_store.register_model(
            session=session, task_name=task_name, model_id="model", owner="test", artifact_id=artifact.artifact_id
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id=event_id, timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id=event_id, y=1, reason="test")],
            policy=None,
        )
        event_store.add_prediction_skips(session=session, task_name=task_name, model_id="model", event_ids=[event_id])
        event_store.add_trainings(session=session, task_name=task_name, model_id="model", event_ids=[event_id])

        assert event_store.completed_labelled_events(session=session, task_name=task_name, event_ids=[event_id]) == []

        label_record = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": event_id})
        event = session.get(BenchmarkEvent, {"task_name": task_name, "event_id": event_id})
        ready = session.get(ReadyLabel, {"task_name": task_name, "event_id": event_id})
        assert label_record is not None and event is not None and ready is not None
        model_store.advance_model_checkpoint(
            session=session,
            task_name=task_name,
            registration=registration,
            previous_sequence=0,
            ready_sequence=ready.sequence,
        )
        assert event_store.completed_labelled_events(session=session, task_name=task_name, event_ids=[event_id]) == [
            event_id
        ]


def test_archive_purge_accepts_more_than_postgres_parameter_limit(*, sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as session:
        archive_store.purge_archived_events(
            session=session,
            task_name=f"large-archive-purge-test-{uuid4()}",
            event_ids=[f"event-{index}" for index in range(65_536)],
        )


def test_stream_cursor_is_updated_atomically(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"cursor-test-{uuid4()}"
    with sessions.begin() as session:
        event_store.save_stream_cursor(session=session, task_name=task_name, stream_name="events", event_id="first")
        event_store.save_stream_cursor(session=session, task_name=task_name, stream_name="events", event_id="second")

    with sessions() as session:
        assert event_store.stream_cursor(session=session, task_name=task_name, stream_name="events") == "second"


def test_duplicate_old_event_does_not_erase_an_existing_positive_label(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"duplicate-label-test-{uuid4()}"
    timestamp = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="event", timestamp=timestamp, payload={"value": 1.0})],
        )
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="event", y=1, reason="positive")],
            policy=None,
        )

    with sessions.begin() as session:
        inserted = event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="event", timestamp=timestamp, payload={"value": 999.0})],
            policy=LabelPolicy(delay_seconds=60),
        )
        assert inserted == []

    with sessions() as session:
        label = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "event"})
        assert label is not None
        assert label.y == 1


def test_positive_horizon_uses_source_availability_time(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"source-time-label-test-{uuid4()}"
    event_time = datetime.now(UTC) - timedelta(days=4)
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(event_id="timely", timestamp=event_time.timestamp(), payload={"value": 1.0}),
                Observation(event_id="late", timestamp=event_time.timestamp(), payload={"value": 2.0}),
            ],
        )
        inserted = event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[
                event_store.LabelInput(
                    event_id="timely", y=1, reason="positive", available_at=event_time + timedelta(hours=47)
                ),
                event_store.LabelInput(
                    event_id="late", y=1, reason="positive", available_at=event_time + timedelta(hours=49)
                ),
            ],
            policy=LabelPolicy(delay_seconds=48 * 60 * 60),
        )
        assert inserted == ["timely"]

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}) is None


def test_pending_positive_horizon_uses_source_availability_time(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"pending-source-time-label-test-{uuid4()}"
    event_time = datetime.now(UTC) - timedelta(days=4)
    with sessions.begin() as session:
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[
                event_store.LabelInput(
                    event_id="timely", y=1, reason="positive", available_at=event_time + timedelta(hours=47)
                ),
                event_store.LabelInput(
                    event_id="late", y=1, reason="positive", available_at=event_time + timedelta(hours=49)
                ),
            ],
            policy=LabelPolicy(delay_seconds=48 * 60 * 60),
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(event_id="timely", timestamp=event_time.timestamp(), payload={"value": 1.0}),
                Observation(event_id="late", timestamp=event_time.timestamp(), payload={"value": 2.0}),
            ],
            policy=LabelPolicy(delay_seconds=48 * 60 * 60),
        )

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "timely"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "late"}) is None


def test_orphan_label_retention_does_not_remove_matched_labels(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"orphan-label-test-{uuid4()}"
    cutoff = datetime.now(UTC) + timedelta(seconds=1)
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="matched", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[
                event_store.LabelInput(event_id="matched", y=1, reason="test"),
                event_store.LabelInput(event_id="orphan", y=1, reason="test"),
            ],
            policy=None,
        )
        assert event_store.purge_orphan_labels(session=session, task_name=task_name, cutoff=cutoff) == 1

    with sessions() as session:
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "matched"}) is not None
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "orphan"}) is None


def test_deleting_registration_cascades_all_model_event_state(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"delete-cascade-test-{uuid4()}"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        model_store.register_model(
            session=session, task_name=task_name, model_id="model", owner="test", artifact_id=artifact.artifact_id
        )
        session.flush()
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="event", timestamp=datetime.now(UTC).timestamp(), payload={"value": 1.0})],
        )
        event_store.add_predictions(session=session, task_name=task_name, model_id="model", predictions={"event": 0.5})

    with sessions.begin() as session:
        assert model_store.delete_model(session=session, task_name=task_name, model_id="model")

    with sessions() as session:
        assert (
            session.get(
                ModelEventState,
                {"task_name": task_name, "event_id": "event", "model_id": "model"},
            )
            is None
        )


def test_forecast_snapshot_resolves_two_polls_ahead_and_missing_target_is_archived(
    *, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from everbench.tasks import load_task

    task_name = f"forecast-test-{uuid4()}"
    task = replace(load_task(path="tasks/citibike/task.py"), TASK_NAME=task_name)
    origin = datetime.now(UTC) - timedelta(days=15)
    policy = task.label_policy
    assert policy is not None
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(
                    event_id="origin", timestamp=origin.timestamp(), payload={}, entity_key="station", value=12
                ),
                Observation(event_id="missing", timestamp=origin.timestamp(), payload={}, entity_key="absent", value=5),
            ],
            policy=policy,
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(
                    event_id="early",
                    timestamp=(origin + timedelta(minutes=15)).timestamp(),
                    payload={},
                    entity_key="station",
                    value=99,
                ),
                Observation(
                    event_id="target",
                    timestamp=(origin + timedelta(minutes=30)).timestamp(),
                    payload={},
                    entity_key="station",
                    value=0,
                ),
            ],
            policy=policy,
        )
        resolved = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "origin"})
        assert resolved is not None and resolved.y == 0
        event_store.resolve_due(session=session, task_name=task_name, policy=policy)
        # Missing targets remain durable resolutions, never synthetic zeroes.
        missing = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "missing"})
        assert missing is not None and missing.y is None and missing.reason == "target-unavailable"
        for row in session.scalars(select(BenchmarkEvent).where(BenchmarkEvent.task_name == task_name)):
            row.inserted_at = origin
    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 4, "labels": 1}
    monkeypatch.setattr(
        archive, "CONFIG", replace(CONFIG, archive_root=tmp_path, s3_bucket_name=None, archive_after_days=0)
    )
    assert archive.archive_once(sessions=sessions, task=task) == 4
    with sessions() as session:
        assert reporting.task_stats(session=session, task_name=task_name) == {"events": 4, "labels": 1}
        manifest = session.scalar(select(ArchiveManifest).where(ArchiveManifest.task_name == task_name))
        assert manifest is not None and manifest.label_count == 1
        from everbench.replay import read_examples

        rows = list(read_examples(path=Path(manifest.path)))
        assert sum(row.target is None for row in rows) == 3


def test_new_generation_does_not_replay_old_live_training(*, sessions: sessionmaker[Session]) -> None:
    task_name = f"generation-boundary-{uuid4()}"
    with sessions.begin() as session:
        payload = artifacts.dumps(model=WorkingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        registration, _ = model_store.register_model(
            session=session, task_name=task_name, model_id="model", owner="test", artifact_id=artifact.artifact_id
        )
        session.flush()
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="old", timestamp=datetime.now(UTC).timestamp(), payload={})],
        )
        event_store.add_labels(
            session=session, task_name=task_name, labels=[event_store.LabelInput(event_id="old", y=1, reason="test")]
        )
        event_store.add_predictions(session=session, task_name=task_name, model_id="model", predictions={"old": 0.5})
        event_store.add_trainings(session=session, task_name=task_name, model_id="model", event_ids=["old"])
        frontier = model_store.start_live_generation(session=session, registration=registration)
        model_store.save_pickle_snapshot(
            session=session, task_name=task_name, model_id="model", payload=payload, checkpoint_ready_sequence=frontier
        )
        assert (
            list(
                model_store.trained_examples_since_checkpoint(
                    session=session, task_name=task_name, model_id="model", checkpoint_ready_sequence=0
                )
            )
            == []
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[Observation(event_id="new", timestamp=datetime.now(UTC).timestamp(), payload={})],
        )
        pending = event_store.events_after_cursor(
            session=session,
            task_name=task_name,
            model_id="model",
            cursor_sequence=registration.prediction_cursor_sequence,
            start_sequence=registration.start_sequence,
        )
        assert [row.event_id for row in pending] == ["new"]


def test_restart_preserves_learning_before_pause_and_skips_disabled_targets(
    *, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from everbench import learner

    task_name = f"pause-recovery-{uuid4()}"
    task = replace(load_task(path="tasks/citibike/task.py"), TASK_NAME=task_name)
    cache = {}
    monkeypatch.setattr(learner, "CONFIG", replace(CONFIG, model_checkpoint_seconds=3600))
    with sessions.begin() as session:
        payload = artifacts.dumps(model=CountingModel())
        artifact = model_store.store_artifact(
            session=session, payload=payload, signature=artifacts.sign(payload=payload), metadata={}
        )
        model_store.register_model(
            session=session, task_name=task_name, model_id="model", owner="test", artifact_id=artifact.artifact_id
        )
        model_store.save_pickle_snapshot(
            session=session, task_name=task_name, model_id="model", payload=payload, checkpoint_ready_sequence=0
        )
        event_store.add_events(
            session=session,
            task_name=task_name,
            events=[
                Observation(event_id=identifier, timestamp=datetime.now(UTC).timestamp(), payload={})
                for identifier in ("learned", "unavailable", "disabled")
            ],
        )
        learner.learn_once(session=session, task=task, cache=cache)
    with sessions.begin() as session:
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[
                event_store.LabelInput(event_id="learned", y=1, reason="test"),
                event_store.LabelInput(event_id="unavailable", y=None, reason="target-unavailable"),
            ],
        )
        learner.learn_once(session=session, task=task, cache=cache)
        assert cache["model"].model.model.learned == ["learned"]
        assert cache["model"].tracker.observations == 1
        registration = model_store.model_registration(session=session, task_name=task_name, model_id="model")
        assert registration is not None
        registration.disabled_until = datetime.now(UTC) + timedelta(hours=1)
    with sessions.begin() as session:
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="disabled", y=99, reason="test")],
        )
        learner.learn_once(session=session, task=task, cache=cache)
        snapshot = model_store.latest_snapshot(session=session, task_name=task_name, model_id="model")
        assert snapshot is not None and snapshot.checkpoint_ready_sequence == 0
        registration = model_store.model_registration(session=session, task_name=task_name, model_id="model")
        assert registration is not None
        registration.disabled_until = None
    # Restart with no new data. Recovery must still checkpoint its restored learning.
    monkeypatch.setattr(learner, "CONFIG", replace(CONFIG, model_checkpoint_seconds=0.000001))
    with sessions.begin() as session:
        learner.learn_once(session=session, task=task, cache={})
        snapshot = model_store.latest_snapshot(session=session, task_name=task_name, model_id="model")
        assert snapshot is not None and snapshot.checkpoint_ready_sequence > 0
        artifact = model_store.artifact(session=session, artifact_id=snapshot.artifact_id)
        assert artifact is not None
        restored = artifacts.loads(payload=artifact.payload, signature=artifact.signature)
        assert restored.learned == ["learned"]
        assert (
            len(
                event_store.completed_labelled_events(
                    session=session, task_name=task_name, event_ids=["learned", "unavailable", "disabled"]
                )
            )
            == 3
        )


@pytest.mark.parametrize("seconds,expected", [(1799, None), (1800, 0), (1920, 0), (1921, None)])
def test_forecast_matching_respects_station_and_horizon_after_restart(
    *, sessions: sessionmaker[Session], seconds: int, expected: int | None
) -> None:
    task_name = f"forecast-window-{uuid4()}"
    origin = datetime.now(UTC) - timedelta(hours=1)
    policy = LabelPolicy(delay_seconds=1800, tolerance_seconds=120)
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            policy=policy,
            events=[Observation(event_id="origin", timestamp=origin.timestamp(), payload={}, entity_key="A", value=5)],
        )
    with sessions.begin() as session:
        event_store.add_events(
            session=session,
            task_name=task_name,
            policy=policy,
            events=[
                Observation(
                    event_id="other-station", timestamp=origin.timestamp() + 1800, payload={}, entity_key="B", value=99
                )
            ],
        )
        assert session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "origin"}) is None
        event_store.add_events(
            session=session,
            task_name=task_name,
            policy=policy,
            events=[
                Observation(
                    event_id="target", timestamp=origin.timestamp() + seconds, payload={}, entity_key="A", value=0
                )
            ],
        )
        label = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "origin"})
        assert (label.y if label is not None else None) == expected
        event_store.resolve_due(session=session, task_name=task_name, policy=policy)
        label = session.get(BenchmarkLabel, {"task_name": task_name, "event_id": "origin"})
        assert label is not None and label.y == expected
        # Retries and later source updates cannot overwrite a finalized resolution.
        event_store.add_labels(
            session=session,
            task_name=task_name,
            labels=[event_store.LabelInput(event_id="origin", y=50, reason="retry")],
        )
        session.refresh(label)
        assert label.y == expected


def test_synthetic_labels_use_source_time_despite_poll_jitter(
    *, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from everbench.records import LabelInput
    from everbench.sources import PollingSource

    task = replace(load_task(path="tasks/dummy/task.py"), TASK_NAME=f"synthetic-deadline-{uuid4()}")
    source = task.sources[0]
    assert isinstance(source, PollingSource)
    monkeypatch.setattr(time, "time", lambda: 1_000.1)
    first = next(iter(source.poll(client=None)))
    assert isinstance(first, Observation)
    with sessions.begin() as session:
        event_store.add_events(session=session, task_name=task.TASK_NAME, events=[first], policy=task.label_policy)
    monkeypatch.setattr(time, "time", lambda: 1_003.3)
    labels = [record for record in source.poll(client=None) if isinstance(record, LabelInput)]
    with sessions.begin() as session:
        assert event_store.add_labels(
            session=session, task_name=task.TASK_NAME, labels=labels, policy=task.label_policy
        ) == [first.event_id]
