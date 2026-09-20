"""A local task for exercising the runtime and dashboard.

It creates one deterministic observation every half-second and emits its label
three seconds later. No network connection is required.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from river import metrics

from everbench.records import LabelInput, Observation
from everbench.sources import PollingSource
from everbench.tasks import LabelPolicy, TaskDefinition

TASK_NAME = "dummy"
DESCRIPTION_HTML = """
<p>A deterministic local binary-classification task. Events arrive every half-second and labels follow three seconds later.</p>
"""
PROBLEM_TYPE = "binary_classification"
METRICS = (metrics.Accuracy(), metrics.F1(), metrics.ROCAUC(), metrics.LogLoss())


def poll(*, client):
    tick = int(time.time() * 2)
    identifier = f"dummy:{tick}"
    yield Observation(
        event_id=identifier,
        timestamp=tick / 2,
        payload={"id": identifier, "timestamp": tick / 2, "value": (tick * 17) % 100},
    )
    yield LabelInput(
        event_id=f"dummy:{tick - 6}",
        y=int((tick - 6) % 5 == 0),
        reason="synthetic",
        available_at=datetime.fromtimestamp(tick / 2, UTC),
    )


TASK = TaskDefinition(
    TASK_NAME=TASK_NAME,
    PROBLEM_TYPE=PROBLEM_TYPE,
    METRICS=METRICS,
    DESCRIPTION_HTML=DESCRIPTION_HTML,
    sources=(PollingSource(name="synthetic", interval_seconds=0.5, poll=poll),),
    label_policy=LabelPolicy(delay_seconds=3),
)
