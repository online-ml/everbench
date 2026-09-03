"""Archive completed benchmark rows to replayable Parquet files."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive_store
from everbench.config import CONFIG
from everbench.metrics import MetricTracker
from everbench.models import PickledModel, metric_inputs_for, prediction_for
from everbench.tasks import TaskDefinition


def storage_configured() -> bool:
    return CONFIG.s3_bucket_name is not None or CONFIG.archive_root is not None


@lru_cache
def _s3_client():
    if not CONFIG.s3_bucket_name or not CONFIG.s3_endpoint_url:
        raise RuntimeError("S3_BUCKET_NAME and S3_ENDPOINT_URL are required for R2 archive storage")
    return boto3.client(
        "s3",
        endpoint_url=CONFIG.s3_endpoint_url,
        region_name=CONFIG.s3_region,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        verify=CONFIG.s3_ca_bundle,
    )


def _publish(task_name: str, week_start: str, content_sha256: str, payload: bytes) -> tuple[str, int]:
    """Publish immutable archive bytes to R2, or a local development directory."""
    if CONFIG.s3_bucket_name:
        key = f"task={task_name}/week={week_start}/events-{content_sha256}.parquet"
        _s3_client().put_object(
            Bucket=CONFIG.s3_bucket_name, Key=key, Body=payload, ContentType="application/octet-stream"
        )
        return f"s3://{CONFIG.s3_bucket_name}/{key}", len(payload)
    if CONFIG.archive_root is None:
        raise RuntimeError("configure S3_BUCKET_NAME or EVERBENCH_ARCHIVE_ROOT for durable archives")
    directory = CONFIG.archive_root / f"task={task_name}" / f"week={week_start}"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"events-{content_sha256}.parquet"
    if not output.exists():
        temporary = output.with_suffix(".parquet.tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
    return str(output), output.stat().st_size


def _s3_location(location: str) -> tuple[str, str] | None:
    if not location.startswith("s3://"):
        return None
    bucket, key = location.removeprefix("s3://").split("/", 1)
    if bucket != CONFIG.s3_bucket_name:
        raise FileNotFoundError("archive is not in the configured R2 bucket")
    return bucket, key


def read_archive(location: str) -> bytes:
    if remote := _s3_location(location):
        bucket, key = remote
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
        try:
            return body.read()
        finally:
            body.close()
    return Path(location).read_bytes()


def stream_archive(location: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield an archive without buffering the entire object in web-process memory."""
    if remote := _s3_location(location):
        bucket, key = remote
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
        try:
            yield from body.iter_chunks(chunk_size=chunk_size)
        finally:
            body.close()
        return
    with Path(location).open("rb") as source:
        while chunk := source.read(chunk_size):
            yield chunk


def archive_size(location: str) -> int:
    if remote := _s3_location(location):
        bucket, key = remote
        return int(_s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"])
    return Path(location).stat().st_size


def replay_archive(task: TaskDefinition, uploaded_model: Any, path: Path | bytes) -> dict[str, Any]:
    """Backtest an uploaded model against an archive.

    An event creates a prediction at ``event_available_at``. Its label only
    affects metrics and learning at ``label_available_at``. This preserves the
    delayed-feedback semantics of the live benchmark rather than treating each
    archived row as an immediately labelled example.
    """
    model = PickledModel("backtest", uploaded_model)
    tracker = MetricTracker.fresh(task.PROBLEM_TYPE, task.METRICS)
    parquet = pq.ParquetFile(pa.BufferReader(path) if isinstance(path, bytes) else path)
    # Read archives made before the compact schema too.
    event_column = "features_json" if "features_json" in parquet.schema.names else "payload_json"
    has_sequence = "event_sequence" in parquet.schema.names
    columns = ["event_id", event_column, "label", "event_available_at", "label_available_at"]
    if has_sequence:
        columns.append("event_sequence")

    timeline: list[tuple[datetime, int, int, str, str, dict[str, float] | Any]] = []
    fallback_sequence = 0
    for batch in parquet.iter_batches(columns=columns):
        for row in batch.to_pylist():
            event = json.loads(row[event_column])
            sequence = int(row["event_sequence"]) if has_sequence else fallback_sequence
            fallback_sequence += 1
            event_id = row["event_id"]
            event_at = datetime.fromisoformat(row["event_available_at"])
            label_at = datetime.fromisoformat(row["label_available_at"])
            if label_at < event_at:
                raise ValueError(f"archive label for {event_id!r} became available before its event")
            # Event actions sort before labels at exactly the same time.
            timeline.append((event_at, 0, sequence, event_id, "event", event))
            timeline.append((label_at, 1, sequence, event_id, "label", row["label"]))

    predictions: dict[str, tuple[dict[str, float], Any]] = {}
    predict_seconds = 0.0
    learn_seconds = 0.0
    for _, _, _, event_id, action, value in sorted(timeline):
        if action == "event":
            event = value
            started_at = perf_counter()
            predictions[event_id] = (event, prediction_for(task, model, event_id, event))
            predict_seconds += perf_counter() - started_at
            tracker.predictions += 1
            continue
        event, prediction = predictions.pop(event_id)
        target = value
        tracker.update(target, prediction, lambda metric, y, prediction: metric_inputs_for(task, metric, y, prediction))
        if model.supports_learning:
            started_at = perf_counter()
            model.learn_one(event_id, event, target)
            learn_seconds += perf_counter() - started_at
    return {
        "predictions": tracker.predictions,
        "labels": tracker.observations,
        "metrics": tracker.values(),
        "timing_seconds": {
            "predict": predict_seconds,
            "learn": learn_seconds,
            "total": predict_seconds + learn_seconds,
        },
    }


def _record(row: dict) -> dict:
    """Use JSON strings for task-varying raw event payloads while keeping tabular columns."""
    return {
        "event_id": row["event_id"],
        # ``event_sequence`` makes same-timestamp replay deterministic. It is
        # the durable ordering assigned when Everbench accepted the event.
        "event_sequence": row["sequence"],
        "event_available_at": row["inserted_at"].isoformat(),
        "payload_json": json.dumps(row["event"], sort_keys=True, separators=(",", ":")),
        "label": row["y"],
        "label_reason": row["reason"],
        "label_available_at": row["available_at"].isoformat(),
    }


def archive_once(sessions: sessionmaker[Session], task: TaskDefinition) -> int:
    """Archive one eligible weekly partition batch, returning its event count.

    Files have a deterministic content-hash name. A crash after file creation
    and before committing the manifest can therefore be safely retried without
    creating a duplicate replay dataset.
    """
    if not storage_configured():
        raise RuntimeError("configure S3_BUCKET_NAME or EVERBENCH_ARCHIVE_ROOT for durable archives")
    cutoff = datetime.now(UTC) - timedelta(days=CONFIG.archive_after_days)
    with sessions() as session:
        week_start = archive_store.next_archive_week(session, task.TASK_NAME, cutoff)
        if week_start is None:
            return 0
        rows = archive_store.archive_rows(session, task.TASK_NAME, week_start, cutoff, CONFIG.archive_batch_size)
    if not rows:
        return 0
    records = [_record(row) for row in rows]
    identity = {"task_name": task.TASK_NAME, "records": records}
    content_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), buffer, compression="zstd")
    location, byte_size = _publish(task.TASK_NAME, week_start.isoformat(), content_sha256, buffer.getvalue())
    event_ids = [record["event_id"] for record in records]
    with sessions.begin() as session:
        archive_store.record_archive(
            session, content_sha256, task.TASK_NAME, week_start, location, len(records), byte_size
        )
        # The manifest commits with the delete, and only after the immutable
        # file was atomically published. A failed cycle leaves source rows for
        # the next periodic attempt.
        archive_store.purge_archived_events(session, task.TASK_NAME, event_ids)
    return len(records)


def latest_labelled_examples(manifests: list, limit: int = 5) -> list[tuple[str, dict[str, Any], object]]:
    """Read the last labelled records from Parquet manifests in event order."""
    examples: list[tuple[str, dict[str, Any], object]] = []
    for manifest in manifests:
        if len(examples) >= limit:
            break
        parquet = pq.ParquetFile(pa.BufferReader(read_archive(manifest.path)))
        # Archives written before the compact schema used ``features_json``;
        # retain read compatibility while new files use ``payload_json``.
        event_column = "features_json" if "features_json" in parquet.schema.names else "payload_json"
        for index in range(parquet.num_row_groups - 1, -1, -1):
            rows = parquet.read_row_group(index, columns=["event_id", event_column, "label"]).to_pylist()
            for row in reversed(rows):
                examples.append((row["event_id"], json.loads(row[event_column]), row["label"]))
                if len(examples) == limit:
                    break
            if len(examples) == limit:
                break
    return list(reversed(examples))
