"""Small, environment-driven runtime policy knobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class RuntimeConfig:
    ingest_batch_size: int = _int("EVERBENCH_INGEST_BATCH_SIZE", 200)
    ingest_flush_seconds: float = float(os.getenv("EVERBENCH_INGEST_FLUSH_SECONDS", "1"))
    ingest_max_pending_items: int = _int("EVERBENCH_INGEST_MAX_PENDING_ITEMS", 2_000)
    learner_batch_size: int = _int("EVERBENCH_LEARN_BATCH_SIZE", 500)
    learner_idle_seconds: float = float(os.getenv("EVERBENCH_LEARN_IDLE_SECONDS", "5"))
    heartbeat_seconds: float = float(os.getenv("EVERBENCH_HEARTBEAT_SECONDS", "30"))
    hot_event_capacity: int = _int("EVERBENCH_HOT_EVENT_CAPACITY", 10_000)
    shutdown_flush_seconds: float = float(os.getenv("EVERBENCH_SHUTDOWN_FLUSH_SECONDS", "20"))
    archive_after_days: int = _int("EVERBENCH_ARCHIVE_AFTER_DAYS", 30)
    archive_batch_size: int = _int("EVERBENCH_ARCHIVE_BATCH_SIZE", 10_000)
    archive_interval_seconds: float = float(os.getenv("EVERBENCH_ARCHIVE_INTERVAL_SECONDS", "3600"))
    archive_root: Path | None = (
        Path(os.environ["EVERBENCH_ARCHIVE_ROOT"]) if "EVERBENCH_ARCHIVE_ROOT" in os.environ else None
    )
    s3_bucket_name: str | None = os.getenv("S3_BUCKET_NAME")
    s3_endpoint_url: str | None = os.getenv("S3_ENDPOINT_URL")
    s3_region: str = os.getenv("S3_REGION", "auto")
    s3_ca_bundle: str | None = os.getenv("S3_CA_BUNDLE") or (
        "/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").is_file() else None
    )
    max_model_bytes: int = _int("EVERBENCH_MAX_MODEL_BYTES", 10 * 1024 * 1024)
    max_class_definition_bytes: int = _int("EVERBENCH_MAX_CLASS_DEFINITION_BYTES", 100 * 1024)
    model_checkpoint_seconds: float = float(os.getenv("EVERBENCH_MODEL_CHECKPOINT_SECONDS", "60"))
    max_active_models_per_task: int = _int("EVERBENCH_MAX_ACTIVE_MODELS_PER_TASK", 20)


CONFIG = RuntimeConfig()
