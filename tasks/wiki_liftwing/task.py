"""Task definition for predicting whether English Wikipedia edits are reverted.

Run it through the generic harness:

    uv run everbench debug worker tasks/wiki_liftwing/task.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from river import metrics

from everbench.auto.config import AutoResearchConfig
from everbench.auto.research import MetricConstraint, Objective
from everbench.records import LabelInput, Observation
from everbench.sources import SSESource
from everbench.tasks import LabelPolicy, TaskDefinition

TASK_NAME = "wiki-liftwing"
DESCRIPTION_HTML = """
<p>Predict whether an English Wikipedia article edit receives MediaWiki’s <code>mw-reverted</code> tag within 48 hours. Edits without that tag by the deadline receive a negative label.</p>
"""
PROBLEM_TYPE = "binary_classification"
METRICS = (metrics.Accuracy(), metrics.F1(), metrics.ROCAUC(), metrics.LogLoss())
LEADERBOARD_PRIMARY_METRIC = "ROCAUC"
EVENT_STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
LABEL_STREAM_URL = "https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change"
WIKI = "enwiki"

AUTO_RESEARCH = AutoResearchConfig(
    model_id="auto-river",
    owner="everbench-auto",
    seed_path=Path(__file__).with_name("auto") / "candidate.py",
    objective=Objective(
        metric=metrics.ROCAUC(),
        min_improvement=0.01,
        min_observations=100_000,
        required_constraints=("prediction_time_ratio", "serialized_model_size"),
        metric_constraints=(
            MetricConstraint(name="log_loss_non_regression", metric=metrics.LogLoss(), max_regression=0.01),
        ),
    ),
    context={
        "problem_description": (
            "Predict whether an English Wikipedia main-namespace edit receives the mw-reverted tag within 48 hours."
        ),
        "primary_metric": "ROCAUC (higher is better)",
        "notes": [
            "Positive labels are delayed and negatives mature after 48 hours.",
            "The model receives the complete raw Wikimedia recent-change event mapping.",
            "Logged-out editors may use temporary account names beginning with a tilde.",
        ],
    },
    min_archive_observations=100_000,
    # Explore several complete model programs in the single weekly run.
    candidate_budget_per_week=6,
    # Leave 50% growth headroom below the 32 MiB operational checkpoint
    # ceiling. Stateful candidates must remain bounded after promotion.
    max_candidate_model_bytes=16 * 1024 * 1024,
)


def event_id(*, event: dict) -> str | None:
    """ID common to the edit and revision-tags-change event schemas."""
    wiki = event.get("wiki") or event.get("database")
    revision = (
        event.get("rev_id")
        or event.get("revid")
        or (event.get("revision") or {}).get("new")
        or (event.get("revision") or {}).get("rev_id")
    )
    return f"{wiki}:{revision}" if wiki is not None and revision is not None else None


def accepts_event(*, event: dict) -> bool:
    return (
        event.get("type") == "edit"
        and event.get("wiki") == WIKI
        and event.get("namespace") == 0
        and event_id(event=event) is not None
    )


def label_timestamp(*, event: dict) -> float:
    """Use source time so reconnect lag cannot turn a timely positive into a late one."""
    value = (event.get("meta") or {}).get("dt")
    if not isinstance(value, str):
        raise ValueError("revision tag events must contain meta.dt")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def reversions(*, event: dict):
    """Emit a resolution when an English edit gains the reverted tag."""
    current = event.get("tags") or []
    previous = (event.get("prior_state") or {}).get("tags") or []
    wiki = event.get("wiki") or event.get("database")
    if wiki != WIKI or not isinstance(current, list) or not isinstance(previous, list):
        return
    if "mw-reverted" not in set(current) - set(previous):
        return
    identifier = event_id(event=event)
    if identifier is not None:
        yield LabelInput(
            event_id=identifier,
            y=1,
            reason="mw-reverted",
            available_at=datetime.fromtimestamp(label_timestamp(event=event), UTC),
        )


def edits(*, event: dict):
    identifier = event_id(event=event)
    if accepts_event(event=event) and identifier is not None:
        yield Observation(event_id=identifier, timestamp=float(event["timestamp"]), payload=event)


TASK = TaskDefinition(
    TASK_NAME=TASK_NAME,
    PROBLEM_TYPE=PROBLEM_TYPE,
    METRICS=METRICS,
    LEADERBOARD_PRIMARY_METRIC=LEADERBOARD_PRIMARY_METRIC,
    DESCRIPTION_HTML=DESCRIPTION_HTML,
    sources=(
        SSESource(name="events", url=EVENT_STREAM_URL, decode=edits),
        SSESource(name="labels", url=LABEL_STREAM_URL, decode=reversions),
    ),
    label_policy=LabelPolicy(delay_seconds=48 * 60 * 60, default_label=0),
    AUTO_RESEARCH=AUTO_RESEARCH,
)
