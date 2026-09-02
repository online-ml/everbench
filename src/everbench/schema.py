"""Canonical SQLAlchemy schema. Alembic owns changes to these tables."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKeyConstraint,
    Identity,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


class BenchmarkEvent(Base):
    __tablename__ = "benchmark_events"
    __table_args__ = (UniqueConstraint("task_name", "sequence"),)

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    sequence: Mapped[int] = mapped_column(BigInteger, Identity(), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    features: Mapped[dict[str, float]] = mapped_column(JSON_TYPE)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BenchmarkLabel(Base):
    __tablename__ = "benchmark_labels"

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    y: Mapped[Any] = mapped_column(JSON_TYPE, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "benchmark_predictions"
    __table_args__ = (
        ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    prediction: Mapped[Any] = mapped_column(JSON_TYPE, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PredictionSkip(Base):
    """Receipt for an event whose outcome arrived before prediction was possible."""

    __tablename__ = "benchmark_prediction_skips"
    __table_args__ = (
        ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    skipped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Training(Base):
    """Receipt proving an online model has learned one labelled event."""

    __tablename__ = "benchmark_trainings"
    __table_args__ = (
        ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MetricState(Base):
    """One operational metric checkpoint per (task, model)."""

    __tablename__ = "benchmark_metric_state"

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    definition: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    state: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    predictions: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    observations: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    values: Mapped[dict[str, float | None]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class MetricUpdate(Base):
    """Receipt ensuring a prediction contributes to a metric exactly once."""

    __tablename__ = "benchmark_metric_updates"
    __table_args__ = (
        ForeignKeyConstraint(["task_name", "event_id"], ["benchmark_events.task_name", "benchmark_events.event_id"]),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelRegistration(Base):
    """The per-task model identity, ownership, and operational status."""

    __tablename__ = "benchmark_models"

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    artifact_id: Mapped[str | None] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    start_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelArtifact(Base):
    """A content-addressed signed pickle plus its display metadata."""

    __tablename__ = "model_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    signature: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON_TYPE, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelSnapshot(Base):
    """The single restart checkpoint for one model's learned state."""

    __tablename__ = "model_snapshots"

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    artifact_id: Mapped[str] = mapped_column(String, nullable=False)
    checkpoint_label_available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpoint_event_sequence: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ArchiveManifest(Base):
    """Index entry for an immutable Parquet archive stored outside Postgres."""

    __tablename__ = "archive_manifest"

    content_sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_name: Mapped[str] = mapped_column(String, nullable=False)
    event_date: Mapped[date] = mapped_column(Date, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_size: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String, primary_key=True)
    task_name: Mapped[str | None] = mapped_column(String)
    role: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class StreamCursor(Base):
    """Last durably accepted SSE event ID for a task input stream."""

    __tablename__ = "stream_cursors"

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    stream_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
