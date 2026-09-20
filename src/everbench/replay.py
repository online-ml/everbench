"""Shared archive decoding and delayed replay, retaining only pending targets."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from river import stream


@dataclass(frozen=True, slots=True, kw_only=True)
class ArchiveExample:
    event_id: str
    sequence: int
    payload: dict[str, Any]
    target: Any
    available_at: datetime
    resolved_at: datetime


def read_examples(*, path: Path | bytes) -> Iterator[ArchiveExample]:
    parquet = pq.ParquetFile(pa.BufferReader(path) if isinstance(path, bytes) else path)
    columns = ["event_id", "event_sequence", "payload_json", "label", "event_available_at", "label_available_at"]
    for batch in parquet.iter_batches(columns=columns):
        for row in batch.to_pylist():
            event_at = datetime.fromisoformat(row["event_available_at"])
            resolved_at = max(datetime.fromisoformat(row["label_available_at"]), event_at)
            payload = json.loads(row["payload_json"])
            if not isinstance(payload, dict):
                raise TypeError(f"archived event {row['event_id']!r} is not a mapping")
            yield ArchiveExample(
                event_id=row["event_id"],
                sequence=int(row["event_sequence"]),
                payload=payload,
                target=row["label"],
                available_at=event_at,
                resolved_at=resolved_at,
            )


@dataclass(kw_only=True)
class ReplayResult:
    predictions: int = 0
    labels: int = 0
    predict_seconds: float = 0.0
    learn_seconds: float = 0.0


def replay(
    *,
    observations: Iterable[ArchiveExample],
    predict: Callable[..., Any],
    score: Callable[..., None],
    learn: Callable[..., None] | None,
) -> ReplayResult:
    """Use River's delayed validation order for both backtests and research.

    Only pending predictions are retained. A True answer marker distinguishes a
    resolution with an unavailable target from River's None question marker.
    River reveals due targets before the next observation, including time ties.
    """
    result = ReplayResult()
    predictions: dict[int, Any] = {}
    answers = stream.simulate_qa(
        dataset=(({"observation": observation}, True) for observation in observations),
        moment=lambda record: record["observation"].available_at,
        delay=lambda record, _: record["observation"].resolved_at - record["observation"].available_at,
        copy=False,
    )
    for index, record, resolved in answers:
        observation = record["observation"]
        if resolved is None:
            started = perf_counter()
            prediction = predict(event_id=observation.event_id, event=copy.deepcopy(observation.payload))
            result.predict_seconds += perf_counter() - started
            result.predictions += 1
            if observation.target is not None:
                predictions[index] = prediction
        elif observation.target is not None:
            score(target=observation.target, prediction=predictions.pop(index))
            result.labels += 1
            if learn is not None:
                started = perf_counter()
                learn(event_id=observation.event_id, event=copy.deepcopy(observation.payload), label=observation.target)
                result.learn_seconds += perf_counter() - started
    return result
