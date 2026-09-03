"""A feature-engineered online KNN classifier for the Wikipedia task."""

from __future__ import annotations

from math import log1p
from typing import Any

from river import neighbors


class WikiKNNClassifier:
    """Classify raw edit metadata from a bounded window of past edits."""

    def __init__(self) -> None:
        self.model = neighbors.KNNClassifier(
            n_neighbors=25,
            engine=neighbors.LazySearch(window_size=10_000),
        )

    @staticmethod
    def transform(event: dict[str, Any]) -> dict[str, float]:
        change = event.get("length") or {}
        anonymous = float(bool(event.get("user_is_anon")))
        comment_length = float(len(event.get("comment") or ""))
        title_length = float(len(event.get("title") or ""))
        byte_change = float(change.get("new", 0)) - float(change.get("old", 0))
        magnitude = abs(byte_change)
        return {
            "anonymous": anonymous,
            "log_comment_length": log1p(comment_length),
            "log_title_length": log1p(title_length),
            "signed_log_byte_change": (1 if byte_change >= 0 else -1) * log1p(magnitude),
            "log_abs_byte_change": log1p(magnitude),
            "is_large_change": float(magnitude >= 500),
            "anonymous_large_change": anonymous * float(magnitude >= 500),
            "anonymous_short_comment": anonymous * float(comment_length < 10),
        }

    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event_id
        probabilities = self.model.predict_proba_one(self.transform(event))
        positive = float(probabilities.get(True, 0.0))
        return {False: 1.0 - positive, True: positive}

    def learn_one(self, event_id: str, event: dict[str, Any], label: int) -> None:
        del event_id
        self.model.learn_one(self.transform(event), bool(label))
