"""A local task for exercising the runtime and dashboard.

It creates one deterministic observation every half-second and emits its label
three seconds later. No network connection is required.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from threading import Event

from river import metrics

TASK_NAME = "dummy"
DESCRIPTION_HTML = """
<p>A deterministic local binary-classification task. Events arrive every half-second and labels follow three seconds later.</p>
"""
PROBLEM_TYPE = "binary_classification"
METRICS = (metrics.Accuracy(), metrics.F1(), metrics.ROCAUC(), metrics.LogLoss())
EVENT_STREAM_URL = "memory://dummy-events"
LABEL_STREAM_URL = "memory://dummy-labels"
NEGATIVE_LABEL_DELAY_SECONDS = None


def _tick() -> int:
    return int(time.time() * 2)


def event_stream(stop: Event) -> Iterator[dict]:
    while not stop.is_set():
        tick = _tick()
        yield {"id": f"dummy:{tick}", "timestamp": tick / 2, "value": (tick * 17) % 100}
        stop.wait(0.5)


def label_stream(stop: Event) -> Iterator[dict]:
    while not stop.is_set():
        tick = _tick() - 6
        yield {"id": f"dummy:{tick}", "y": int(tick % 5 == 0)}
        stop.wait(0.5)


def event_id(event: dict) -> str | None:
    return event.get("id")


def accepts_event(event: dict) -> bool:
    return event_id(event) is not None


def features_for(event: dict) -> dict[str, float]:
    return {"value": float(event["value"]), "is_even": float(event["value"] % 2 == 0)}


def metric_inputs_for(metric: object, y_true: int, prediction: float) -> tuple[bool, bool | float]:
    """Accuracy and F1 use a hard decision; ranking/loss metrics use probability."""
    target = bool(y_true)
    if isinstance(metric, (metrics.Accuracy, metrics.F1)):
        return target, prediction >= 0.5
    return target, prediction


def label_for(event: dict) -> tuple[str, int, str] | None:
    identifier = event.get("id")
    return (identifier, int(event["y"]), "synthetic") if identifier is not None else None
