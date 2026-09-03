"""Canonical SQLAlchemy schema. Alembic owns changes to these tables."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKeyConstraint,
    Identity,
    Index,
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
    __table_args__ = (
        UniqueConstraint("task_name", "sequence"),
        Index("benchmark_events_inserted_idx", "task_name", "inserted_at"),
        Index("benchmark_events_time_idx", "task_name", "event_time"),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    sequence: Mapped[int] = mapped_column(BigInteger, Identity(), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE)
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class BenchmarkLabel(Base):
    __tablename__ = "benchmark_labels"
    __table_args__ = (
        Index("benchmark_labels_available_idx", "task_name", "available_at"),
        Index("benchmark_labels_inserted_idx", "task_name", "inserted_at"),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    y: Mapped[Any] = mapped_column(JSON_TYPE, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    inserted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelEventState(Base):
    """One model's durable progress for one benchmark event."""

    __tablename__ = "benchmark_model_events"
    __table_args__ = (
        CheckConstraint("prediction_status IN ('predicted', 'skipped')", name="model_event_prediction_status"),
        CheckConstraint(
            "(prediction_status = 'predicted' AND prediction IS NOT NULL AND prediction_reason IS NULL) "
            "OR (prediction_status = 'skipped' AND prediction IS NULL AND prediction_reason IS NOT NULL)",
            name="model_event_prediction_payload",
        ),
        ForeignKeyConstraint(
            ["task_name", "event_id"],
            ["benchmark_events.task_name", "benchmark_events.event_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["task_name", "model_id"],
            ["benchmark_models.task_name", "benchmark_models.model_id"],
            ondelete="CASCADE",
        ),
        Index("benchmark_model_events_model_idx", "task_name", "model_id", "event_id"),
    )

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    prediction: Mapped[Any | None] = mapped_column(JSON_TYPE)
    prediction_status: Mapped[str] = mapped_column(String, nullable=False)
    prediction_reason: Mapped[str | None] = mapped_column(Text)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MetricState(Base):
    """One operational metric checkpoint per (task, model)."""

    __tablename__ = "benchmark_metric_state"
    __table_args__ = (
        ForeignKeyConstraint(
            ["task_name", "model_id"],
            ["benchmark_models.task_name", "benchmark_models.model_id"],
            ondelete="CASCADE",
        ),
    )

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


class ModelRegistration(Base):
    """The per-task model identity, ownership, and operational status."""

    __tablename__ = "benchmark_models"
    __table_args__ = (Index("benchmark_models_active_idx", "task_name", "active"),)

    task_name: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner: Mapped[str] = mapped_column(String, nullable=False)
    artifact_id: Mapped[str | None] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    disabled_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    skipped_predictions: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    skipped_labels: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
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
    __table_args__ = (
        ForeignKeyConstraint(
            ["task_name", "model_id"],
            ["benchmark_models.task_name", "benchmark_models.model_id"],
            ondelete="CASCADE",
        ),
    )

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
