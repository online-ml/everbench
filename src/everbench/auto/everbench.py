"""Thin adapter between portable AutoClassifier and Everbench's model protocol."""

from __future__ import annotations

from typing import Any

from river import base

from everbench.auto.classifier import AutoClassifier


class EverbenchAutoClassifier:
    """Expose an AutoClassifier through Everbench's event-ID-aware protocol."""

    def __init__(self, *, auto_classifier: AutoClassifier) -> None:
        self.auto_classifier = auto_classifier

    def predict_one(self, *, event_id: str, event: dict[str, Any]) -> Any:
        del event_id
        return self.auto_classifier.predict_one(event)

    def predict_proba_one(self, *, event_id: str, event: dict[str, Any]) -> dict[base.typing.ClfTarget, float]:
        del event_id
        return self.auto_classifier.predict_proba_one(event)

    def learn_one(self, *, event_id: str, event: dict[str, Any], label: Any) -> None:
        del event_id
        self.auto_classifier.learn_one(event, label)
