"""Task definition for predicting whether English Wikipedia edits are reverted.

Run it through the generic harness:

    uv run everbench debug worker tasks/wiki_leftwing/task.py
"""

from __future__ import annotations

from river import metrics

TASK_NAME = "wiki-leftwing"
DESCRIPTION_HTML = """
<p>Predict whether an English Wikipedia article edit will be reverted. A reversion tag is a positive label; edits without one after 24 hours are labelled valid.</p>
"""
PROBLEM_TYPE = "binary_classification"
METRICS = (metrics.Accuracy(), metrics.F1(), metrics.ROCAUC(), metrics.LogLoss())
EVENT_STREAM_URL = "https://stream.wikimedia.org/v2/stream/recentchange"
LABEL_STREAM_URL = "https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change"
WIKI = "enwiki"
NEGATIVE_LABEL_DELAY_SECONDS = 24 * 60 * 60


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


def features_for(event: dict) -> dict[str, float]:
    """Features are frozen by the harness before any label is available."""
    change = event.get("length") or {}
    return {
        "anonymous": float(bool(event.get("user_is_anon"))),
        "comment_length": float(len(event.get("comment") or "")),
        "title_length": float(len(event.get("title") or "")),
        "byte_change": float(change.get("new", 0)) - float(change.get("old", 0)),
    }


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
    if event.get("wiki") != WIKI or not isinstance(current, list) or not isinstance(previous, list):
        return None
    if "mw-reverted" not in set(current) - set(previous):
        return None
    key = event_id(event)
    return (key, 1, "mw-reverted") if key is not None else None
