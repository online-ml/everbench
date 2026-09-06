"""Task definition for predicting whether English Wikipedia edits are reverted.

Run it through the generic harness:

    uv run everbench debug worker tasks/wiki_liftwing/task.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from river import metrics

from everbench.auto.config import AutoResearchConfig
from everbench.auto.research import MetricConstraint, Objective

TASK_NAME = "wiki-liftwing"
DESCRIPTION_HTML = """
<p>Predict whether an English Wikipedia article edit receives MediaWiki’s <code>mw-reverted</code> tag within 48 hours. Edits without that tag by the deadline receive a negative label.</p>
"""
PROBLEM_TYPE = "binary_classification"
METRICS = (metrics.Accuracy(), metrics.F1(), metrics.ROCAUC(), metrics.LogLoss())
EVENT_STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
LABEL_STREAM_URL = "https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change"
WIKI = "enwiki"
NEGATIVE_LABEL_DELAY_SECONDS = 48 * 60 * 60

AUTO_RESEARCH = AutoResearchConfig(
    model_id="auto-river",
    owner="everbench-auto",
    candidate_path=Path(__file__).with_name("auto") / "candidate.py",
    objective=Objective(
        metrics.ROCAUC(),
        min_improvement=0.002,
        min_observations=10_000,
        required_constraints=("prediction_time_ratio",),
        metric_constraints=(MetricConstraint("log_loss_non_regression", metrics.LogLoss(), 0.01),),
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
    # The stream currently produces roughly 4,500 eligible edits/hour. A
    # 50,000-row default would cover only about 12 hours and could never expose
    # delayed feedback to a fresh model before promotion predictions begin.
    history_limit=300_000,
    min_research_span_seconds=48 * 60 * 60,
    retain_raw_examples=True,
)


def event_id(event: dict) -> str | None:
    """ID common to the edit and revision-tags-change event schemas."""
    wiki = event.get("wiki") or event.get("database")
    revision = (
        event.get("rev_id")
        or event.get("revid")
        or (event.get("revision") or {}).get("new")
        or (event.get("revision") or {}).get("rev_id")
    )
    return f"{wiki}:{revision}" if wiki is not None and revision is not None else None


def accepts_event(event: dict) -> bool:
    return (
        event.get("type") == "edit"
        and event.get("wiki") == WIKI
        and event.get("namespace") == 0
        and event_id(event) is not None
    )


def label_timestamp(event: dict) -> float:
    """Use source time so reconnect lag cannot turn a timely positive into a late one."""
    value = (event.get("meta") or {}).get("dt")
    if not isinstance(value, str):
        raise ValueError("revision tag events must contain meta.dt")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def metric_inputs_for(metric: object, y_true: int, prediction: float) -> tuple[bool, bool | float]:
    """Accuracy and F1 use a hard decision; ranking/loss metrics use probability."""
    target = bool(y_true)
    if isinstance(metric, (metrics.Accuracy, metrics.F1)):
        return target, prediction >= 0.5
    return target, prediction


def label_for(event: dict) -> tuple[str, int, str] | None:
    """Return a label emitted by the label stream, or ``None`` to ignore it."""
    current = event.get("tags") or []
    previous = (event.get("prior_state") or {}).get("tags") or []
    wiki = event.get("wiki") or event.get("database")
    if wiki != WIKI or not isinstance(current, list) or not isinstance(previous, list):
        return None
    if "mw-reverted" not in set(current) - set(previous):
        return None
    key = event_id(event)
    return (key, 1, "mw-reverted") if key is not None else None
