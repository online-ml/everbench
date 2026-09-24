"""Verify the one-time copy against the legacy PostgreSQL schema when available."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import MetaData, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from everbench.db import make_engine
from everbench.schema import (
    AutoExperiment,
    BenchmarkEvent,
    BenchmarkLabel,
    DatabaseMigration,
    DatabaseSequence,
    ModelArtifact,
    ModelRegistration,
    ModelSnapshot,
    ReadyLabel,
)
from everbench.sqlite_migration import copy_postgres_to_sqlite


def test_postgres_copy_preserves_wiki_model_and_excludes_citibike(*, tmp_path) -> None:  # noqa: PLR0917 -- pytest fixture
    source_url = os.getenv("EVERBENCH_TEST_DATABASE_URL")
    if not source_url:
        pytest.skip("legacy PostgreSQL test database is not configured")
    source = make_engine(url=source_url)
    target = make_engine(url=f"sqlite:///{tmp_path / 'copied.db'}")
    metadata = MetaData()
    metadata.reflect(bind=source)
    tables = metadata.tables
    suffix = uuid4().hex
    wiki_task = f"wiki-copy-{suffix}"
    wiki_event = f"wiki-event-{suffix}"
    bike_event = f"bike-event-{suffix}"
    wiki_model = f"wiki-model-{suffix}"
    bike_model = f"bike-model-{suffix}"
    wiki_artifact = f"wiki-artifact-{suffix}"
    bike_artifact = f"bike-artifact-{suffix}"
    wiki_experiment = f"wiki-experiment-{suffix}"
    sequence = uuid4().int % (2**60)
    now = datetime.now(UTC)
    try:
        with source.begin() as connection:
            connection.execute(
                pg_insert(tables["benchmark_tasks"])
                .values([{"task_name": wiki_task}, {"task_name": "citibike"}])
                .on_conflict_do_nothing(index_elements=["task_name"])
            )
            connection.execute(
                tables["benchmark_events"].insert(),
                [
                    {
                        "task_name": wiki_task,
                        "event_id": wiki_event,
                        "sequence": sequence,
                        "event_time": now,
                        "event": {"x": 1},
                    },
                    {
                        "task_name": "citibike",
                        "event_id": bike_event,
                        "sequence": sequence + 1,
                        "event_time": now,
                        "event": {"x": 2},
                    },
                ],
            )
            connection.execute(
                tables["benchmark_labels"].insert(),
                [
                    {"task_name": wiki_task, "event_id": wiki_event, "y": 1, "reason": "source", "available_at": now},
                    {"task_name": "citibike", "event_id": bike_event, "y": 2, "reason": "source", "available_at": now},
                ],
            )
            connection.execute(
                tables["benchmark_ready_labels"].insert(),
                [
                    {"task_name": wiki_task, "event_id": wiki_event, "sequence": sequence, "has_target": True},
                    {"task_name": "citibike", "event_id": bike_event, "sequence": sequence + 1, "has_target": True},
                ],
            )
            connection.execute(
                tables["model_artifacts"].insert(),
                [
                    {
                        "artifact_id": wiki_artifact,
                        "sha256": suffix * 2,
                        "payload": b"wiki",
                        "signature": suffix * 2,
                        "metadata": {},
                    },
                    {
                        "artifact_id": bike_artifact,
                        "sha256": suffix[::-1] * 2,
                        "payload": b"bike",
                        "signature": suffix[::-1] * 2,
                        "metadata": {},
                    },
                ],
            )
            connection.execute(
                tables["benchmark_models"].insert(),
                [
                    {
                        "task_name": wiki_task,
                        "model_id": wiki_model,
                        "owner": "test",
                        "artifact_id": wiki_artifact,
                        "start_sequence": sequence + 10,
                        "failure_count": 0,
                        "skipped_predictions": 0,
                        "skipped_labels": 0,
                        "error_count": 0,
                        "prediction_cursor_sequence": 0,
                        "label_cursor_sequence": 0,
                    },
                    {
                        "task_name": "citibike",
                        "model_id": bike_model,
                        "owner": "test",
                        "artifact_id": bike_artifact,
                        "start_sequence": sequence + 1,
                        "failure_count": 0,
                        "skipped_predictions": 0,
                        "skipped_labels": 0,
                        "error_count": 0,
                        "prediction_cursor_sequence": 0,
                        "label_cursor_sequence": 0,
                    },
                ],
            )
            connection.execute(
                tables["model_snapshots"].insert(),
                [
                    {
                        "task_name": wiki_task,
                        "model_id": wiki_model,
                        "artifact_id": wiki_artifact,
                        "checkpoint_ready_sequence": sequence,
                    },
                    {
                        "task_name": "citibike",
                        "model_id": bike_model,
                        "artifact_id": bike_artifact,
                        "checkpoint_ready_sequence": sequence + 1,
                    },
                ],
            )
            connection.execute(
                tables["auto_experiments"].insert(),
                [
                    {
                        "experiment_id": wiki_experiment,
                        "task_name": wiki_task,
                        "model_id": wiki_model,
                        "parent_generation": 1,
                        "researcher": "test",
                        "status": "rejected",
                        "comparison_start_sequence": sequence,
                        "comparison_end_sequence": sequence,
                        "champion_artifact_id": wiki_artifact,
                    }
                ],
            )

        counts = copy_postgres_to_sqlite(source=source, target=target)
        assert counts["benchmark_events"] >= 1
        with target.connect() as connection:
            assert (
                connection.scalar(select(BenchmarkEvent.event_id).where(BenchmarkEvent.event_id == wiki_event))
                == wiki_event
            )
            assert (
                connection.scalar(select(BenchmarkEvent.event_id).where(BenchmarkEvent.event_id == bike_event)) is None
            )
            assert (
                connection.scalar(select(BenchmarkLabel.event_id).where(BenchmarkLabel.event_id == wiki_event))
                == wiki_event
            )
            assert connection.scalar(select(ReadyLabel.event_id).where(ReadyLabel.event_id == wiki_event)) == wiki_event
            assert (
                connection.scalar(select(ModelRegistration.model_id).where(ModelRegistration.model_id == wiki_model))
                == wiki_model
            )
            assert (
                connection.scalar(select(ModelSnapshot.model_id).where(ModelSnapshot.model_id == wiki_model))
                == wiki_model
            )
            assert (
                connection.scalar(
                    select(AutoExperiment.experiment_id).where(AutoExperiment.experiment_id == wiki_experiment)
                )
                == wiki_experiment
            )
            assert (
                connection.scalar(select(ModelArtifact.artifact_id).where(ModelArtifact.artifact_id == wiki_artifact))
                == wiki_artifact
            )
            assert (
                connection.scalar(select(ModelArtifact.artifact_id).where(ModelArtifact.artifact_id == bike_artifact))
                is None
            )
            assert connection.scalar(select(DatabaseMigration.name)) == "postgres-import"
            assert (
                connection.scalar(select(DatabaseSequence.value).where(DatabaseSequence.name == "events")) or 0
            ) >= sequence + 10
            assert (
                connection.scalar(select(DatabaseSequence.value).where(DatabaseSequence.name == "ready_labels")) or 0
            ) >= sequence + 1
    finally:
        with source.begin() as connection:
            for name, id_column, ids in (
                ("auto_experiments", "experiment_id", [wiki_experiment]),
                ("model_snapshots", "model_id", [wiki_model, bike_model]),
                ("benchmark_models", "model_id", [wiki_model, bike_model]),
                ("benchmark_ready_labels", "event_id", [wiki_event, bike_event]),
                ("benchmark_labels", "event_id", [wiki_event, bike_event]),
                ("benchmark_events", "event_id", [wiki_event, bike_event]),
            ):
                table = tables[name]
                connection.execute(
                    delete(table).where(
                        table.c.task_name.in_([wiki_task, "citibike"]),
                        table.c[id_column].in_(ids),
                    )
                )
            connection.execute(
                delete(tables["model_artifacts"]).where(
                    tables["model_artifacts"].c.artifact_id.in_([wiki_artifact, bike_artifact])
                )
            )
            connection.execute(
                delete(tables["benchmark_tasks"]).where(tables["benchmark_tasks"].c.task_name == wiki_task)
            )
        source.dispose()
        target.dispose()
