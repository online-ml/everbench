"""Validated, environment-driven runtime policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _int(*, name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(*, name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    ingest_batch_size: int = 200
    ingest_flush_seconds: float = 5.0
    ingest_max_pending_items: int = 2_000
    stream_cursor_checkpoint_seconds: float = 30.0
    learner_batch_size: int = 500
    learner_idle_seconds: float = 5.0
    heartbeat_seconds: float = 30.0
    hot_event_capacity: int = 250_000
    hot_event_max_bytes: int = 512 * 1024
    shutdown_flush_seconds: float = 20.0
    # Minimum hold after a UTC week ends. The task's delayed-label window plus
    # one day may require a longer hold.
    archive_after_days: int = 1
    archive_interval_seconds: float = 3_600.0
    archive_root: Path | None = None
    s3_bucket_name: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "auto"
    s3_ca_bundle: str | None = None
    max_model_bytes: int = 10 * 1024 * 1024
    max_class_definition_bytes: int = 100 * 1024
    model_checkpoint_seconds: float = 60.0
    # The runtime ceiling is deliberately higher than any auto-research
    # promotion ceiling. That gap lets a bounded online model grow between
    # research cycles without making its durable checkpoints fail.
    max_model_snapshot_bytes: int = 32 * 1024 * 1024
    max_active_models_per_task: int = 20
    model_retry_initial_seconds: float = 1.0
    model_retry_max_seconds: float = 1_800.0
    max_backtest_bytes: int = 25 * 1024 * 1024
    max_backtest_rows: int = 100_000
    label_inbox_retention_days: int = 7

    def __post_init__(self) -> None:
        positive = {
            "ingest_batch_size": self.ingest_batch_size,
            "ingest_flush_seconds": self.ingest_flush_seconds,
            "ingest_max_pending_items": self.ingest_max_pending_items,
            "stream_cursor_checkpoint_seconds": self.stream_cursor_checkpoint_seconds,
            "learner_batch_size": self.learner_batch_size,
            "learner_idle_seconds": self.learner_idle_seconds,
            "heartbeat_seconds": self.heartbeat_seconds,
            "hot_event_capacity": self.hot_event_capacity,
            "hot_event_max_bytes": self.hot_event_max_bytes,
            "shutdown_flush_seconds": self.shutdown_flush_seconds,
            "archive_interval_seconds": self.archive_interval_seconds,
            "max_model_bytes": self.max_model_bytes,
            "max_class_definition_bytes": self.max_class_definition_bytes,
            "model_checkpoint_seconds": self.model_checkpoint_seconds,
            "max_model_snapshot_bytes": self.max_model_snapshot_bytes,
            "max_active_models_per_task": self.max_active_models_per_task,
            "model_retry_max_seconds": self.model_retry_max_seconds,
            "max_backtest_bytes": self.max_backtest_bytes,
            "max_backtest_rows": self.max_backtest_rows,
            "label_inbox_retention_days": self.label_inbox_retention_days,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"runtime settings must be positive: {', '.join(invalid)}")
        if self.ingest_max_pending_items < self.ingest_batch_size:
            raise ValueError("ingest_max_pending_items must be at least ingest_batch_size")
        if self.archive_after_days < 0 or self.model_retry_initial_seconds < 0:
            raise ValueError("archive_after_days and model_retry_initial_seconds cannot be negative")

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        defaults = cls()
        default_ca = "/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").is_file() else None
        return replace(
            defaults,
            ingest_batch_size=_int(name="EVERBENCH_INGEST_BATCH_SIZE", default=defaults.ingest_batch_size),
            ingest_flush_seconds=_float(name="EVERBENCH_INGEST_FLUSH_SECONDS", default=defaults.ingest_flush_seconds),
            ingest_max_pending_items=_int(
                name="EVERBENCH_INGEST_MAX_PENDING_ITEMS", default=defaults.ingest_max_pending_items
            ),
            stream_cursor_checkpoint_seconds=_float(
                name="EVERBENCH_STREAM_CURSOR_CHECKPOINT_SECONDS", default=defaults.stream_cursor_checkpoint_seconds
            ),
            learner_batch_size=_int(name="EVERBENCH_LEARN_BATCH_SIZE", default=defaults.learner_batch_size),
            learner_idle_seconds=_float(name="EVERBENCH_LEARN_IDLE_SECONDS", default=defaults.learner_idle_seconds),
            heartbeat_seconds=_float(name="EVERBENCH_HEARTBEAT_SECONDS", default=defaults.heartbeat_seconds),
            hot_event_capacity=_int(name="EVERBENCH_HOT_EVENT_CAPACITY", default=defaults.hot_event_capacity),
            hot_event_max_bytes=_int(name="EVERBENCH_HOT_EVENT_MAX_BYTES", default=defaults.hot_event_max_bytes),
            shutdown_flush_seconds=_float(
                name="EVERBENCH_SHUTDOWN_FLUSH_SECONDS", default=defaults.shutdown_flush_seconds
            ),
            archive_after_days=_int(name="EVERBENCH_ARCHIVE_AFTER_DAYS", default=defaults.archive_after_days),
            archive_interval_seconds=_float(
                name="EVERBENCH_ARCHIVE_INTERVAL_SECONDS", default=defaults.archive_interval_seconds
            ),
            archive_root=Path(value) if (value := os.getenv("EVERBENCH_ARCHIVE_ROOT")) else None,
            s3_bucket_name=os.getenv("S3_BUCKET_NAME"),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            s3_region=os.getenv("S3_REGION", defaults.s3_region),
            s3_ca_bundle=os.getenv("S3_CA_BUNDLE") or default_ca,
            max_model_bytes=_int(name="EVERBENCH_MAX_MODEL_BYTES", default=defaults.max_model_bytes),
            max_class_definition_bytes=_int(
                name="EVERBENCH_MAX_CLASS_DEFINITION_BYTES", default=defaults.max_class_definition_bytes
            ),
            model_checkpoint_seconds=_float(
                name="EVERBENCH_MODEL_CHECKPOINT_SECONDS", default=defaults.model_checkpoint_seconds
            ),
            max_model_snapshot_bytes=_int(
                name="EVERBENCH_MAX_MODEL_SNAPSHOT_BYTES", default=defaults.max_model_snapshot_bytes
            ),
            max_active_models_per_task=_int(
                name="EVERBENCH_MAX_ACTIVE_MODELS_PER_TASK", default=defaults.max_active_models_per_task
            ),
            model_retry_initial_seconds=_float(
                name="EVERBENCH_MODEL_RETRY_INITIAL_SECONDS", default=defaults.model_retry_initial_seconds
            ),
            model_retry_max_seconds=_float(
                name="EVERBENCH_MODEL_RETRY_MAX_SECONDS", default=defaults.model_retry_max_seconds
            ),
            max_backtest_bytes=_int(name="EVERBENCH_MAX_BACKTEST_BYTES", default=defaults.max_backtest_bytes),
            max_backtest_rows=_int(name="EVERBENCH_MAX_BACKTEST_ROWS", default=defaults.max_backtest_rows),
            label_inbox_retention_days=_int(
                name="EVERBENCH_LABEL_INBOX_RETENTION_DAYS", default=defaults.label_inbox_retention_days
            ),
        )


CONFIG = RuntimeConfig.from_env()
