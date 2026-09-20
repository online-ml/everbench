"""Named records crossing task, storage, and learner boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    event_id: str
    timestamp: float
    payload: dict[str, Any]
    entity_key: str | None = None
    value: Any = None


@dataclass(frozen=True, kw_only=True)
class LabelInput:
    """A terminal resolution. A None target explicitly means unavailable."""

    event_id: str
    y: Any
    reason: str
    available_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelledExample:
    event_id: str
    payload: dict[str, Any]
    target: Any


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingPrediction:
    event_id: str
    sequence: int
    resolved: bool
    has_state: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadyObservation:
    sequence: int
    event_id: str
    target: Any
    available_at: datetime
    event_sequence: int
    prediction: Any
    prediction_status: str | None
    evaluated: bool
    trained: bool
