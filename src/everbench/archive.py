"""Archive completed benchmark rows to replayable Parquet files."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session, sessionmaker

from everbench import archive_store
from everbench.config import CONFIG
from everbench.metrics import MetricTracker
from everbench.models import PickledModel, metric_inputs_for, prediction_for
from everbench.records import LabelledExample
from everbench.replay import read_examples, replay
from everbench.tasks import TaskDefinition


@dataclass(frozen=True, kw_only=True)
class PublishedArchive:
    content_sha256: str
    path: str
    row_count: int
    byte_size: int


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


def _publish(*, task_name: str, week_start: str, content_sha256: str, payload: bytes) -> tuple[str, int]:
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


def _s3_location(*, location: str) -> tuple[str, str] | None:
    if not location.startswith("s3://"):
        return None
    bucket, key = location.removeprefix("s3://").split("/", 1)
    if bucket != CONFIG.s3_bucket_name:
        raise FileNotFoundError("archive is not in the configured R2 bucket")
    return bucket, key


def read_archive(*, location: str) -> bytes:
    if remote := _s3_location(location=location):
        bucket, key = remote
        body = _s3_client().get_object(Bucket=bucket, Key=key)["Body"]
        try:
            return body.read()
        finally:
            body.close()
    return Path(location).read_bytes()


def stream_archive(*, location: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    """Yield an archive without buffering the entire object in web-process memory."""
    if remote := _s3_location(location=location):
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


def archive_size(*, location: str) -> int:
    if remote := _s3_location(location=location):
        bucket, key = remote
        return int(_s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"])
    return Path(location).stat().st_size


def delete_archive(*, location: str) -> None:
    """Delete a superseded archive object after its manifest has been replaced."""
    if remote := _s3_location(location=location):
        bucket, key = remote
        _s3_client().delete_object(Bucket=bucket, Key=key)
        return
    Path(location).unlink(missing_ok=True)


def replay_archive(*, task: TaskDefinition, uploaded_model: Any, path: Path | bytes) -> dict[str, Any]:
    """Backtest an uploaded model against an archive.

    An event creates a prediction at ``event_available_at``. Its label only
    affects metrics and learning at ``label_available_at``. This preserves the
    delayed-feedback semantics of the live benchmark rather than treating each
    archived row as an immediately labelled example.
    """
    model = PickledModel(model_id="backtest", model=uploaded_model)
    tracker = MetricTracker.fresh(problem_type=task.PROBLEM_TYPE, prototypes=task.METRICS)
    result = replay(
        observations=read_examples(path=path),
        predict=partial(prediction_for, task=task, model=model),
        score=lambda *, target, prediction: tracker.update(
            y_true=target,
            prediction=prediction,
            inputs_for=partial(metric_inputs_for, task=task),
        ),
        learn=model.learn_one if model.supports_learning else None,
    )
    return {
        "predictions": result.predictions,
        "labels": tracker.observations,
        "metrics": tracker.values(),
        "timing_seconds": {
            "predict": result.predict_seconds,
            "learn": result.learn_seconds,
            "total": result.predict_seconds + result.learn_seconds,
        },
    }


def _record(*, row: dict) -> dict:
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


def _publish_records(*, task_name: str, week_start: date, records: list[dict]) -> PublishedArchive:
    identity = {"task_name": task_name, "records": records}
    content_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(records), buffer, compression="zstd")
    location, byte_size = _publish(
        task_name=task_name, week_start=week_start.isoformat(), content_sha256=content_sha256, payload=buffer.getvalue()
    )
    return PublishedArchive(content_sha256=content_sha256, path=location, row_count=len(records), byte_size=byte_size)


def archive_week_closed(*, week_start: date, cutoff: datetime) -> bool:
    """Return whether an entire UTC availability week is past the cutoff."""
    week_end = datetime.combine(week_start + timedelta(days=7), time.min, UTC)
    return cutoff >= week_end


def archive_cutoff(*, task: TaskDefinition, now: datetime, minimum_days: int) -> datetime:
    """Wait for delayed labels and one extra day after a UTC week closes."""
    label_delay = timedelta(seconds=task.label_policy.close_after_seconds if task.label_policy else 0)
    retention = max(label_delay + timedelta(days=1), timedelta(days=minimum_days))
    return now - retention


def archive_once(*, sessions: sessionmaker[Session], task: TaskDefinition) -> int:
    """Archive one complete availability week into one Parquet file.

    Files have a deterministic content-hash name. A crash after file creation
    and before committing the manifest can therefore be safely retried without
    creating a duplicate replay dataset.
    """
    if not storage_configured():
        raise RuntimeError("configure S3_BUCKET_NAME or EVERBENCH_ARCHIVE_ROOT for durable archives")
    cutoff = archive_cutoff(task=task, now=datetime.now(UTC), minimum_days=CONFIG.archive_after_days)
    with sessions() as session:
        week_start = archive_store.next_archive_week(session=session, task_name=task.TASK_NAME, cutoff=cutoff)
        if week_start is None:
            return 0
        if not archive_week_closed(week_start=week_start, cutoff=cutoff):
            return 0
        if not archive_store.archive_week_ready(session=session, task_name=task.TASK_NAME, week_start=week_start):
            return 0
        rows = archive_store.archive_rows(session=session, task_name=task.TASK_NAME, week_start=week_start)
    if not rows:
        return 0
    records = [_record(row=row) for row in rows]
    published = _publish_records(task_name=task.TASK_NAME, week_start=week_start, records=records)
    event_ids = [record["event_id"] for record in records]
    with sessions.begin() as session:
        inserted = archive_store.record_archive(
            session=session,
            content_sha256=published.content_sha256,
            task_name=task.TASK_NAME,
            event_date=week_start,
            path=published.path,
            row_count=published.row_count,
            byte_size=published.byte_size,
            label_count=sum(row["y"] is not None for row in rows),
        )
        if not inserted:
            existing = archive_store.archive_for_week(session=session, task_name=task.TASK_NAME, event_date=week_start)
            if existing is None or existing.content_sha256 != published.content_sha256:
                raise RuntimeError(f"archive week {task.TASK_NAME}/{week_start} already has a different file")
        # The manifest commits with the delete, and only after the immutable
        # file was atomically published. A failed cycle leaves source rows for
        # the next periodic attempt.
        archive_store.purge_archived_events(session=session, task_name=task.TASK_NAME, event_ids=event_ids)
    return len(records)


def latest_labelled_examples(*, manifests: list, limit: int = 5) -> list[LabelledExample]:
    """Read recent examples from the newest weekly archive files."""
    examples: list[LabelledExample] = []
    for manifest in manifests:
        if len(examples) >= limit:
            break
        parquet = pq.ParquetFile(pa.BufferReader(read_archive(location=manifest.path)))
        for index in range(parquet.num_row_groups - 1, -1, -1):
            rows = parquet.read_row_group(index, columns=["event_id", "payload_json", "label"]).to_pylist()
            for row in reversed(rows):
                if row["label"] is None:
                    continue
                examples.append(
                    LabelledExample(
                        event_id=row["event_id"], payload=json.loads(row["payload_json"]), target=row["label"]
                    )
                )
                if len(examples) == limit:
                    break
            if len(examples) == limit:
                break
    return list(reversed(examples))
