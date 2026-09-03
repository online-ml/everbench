"""An online k-nearest-neighbours classifier for the dummy task."""

from __future__ import annotations

from typing import Any

from river import neighbors


class DummyKNNClassifier:
    """Extract two features and keep a bounded recent window of them."""

    def __init__(self) -> None:
        self.model = neighbors.KNNClassifier(n_neighbors=5, engine=neighbors.LazySearch(window_size=1_000))

    @staticmethod
    def transform(event: dict[str, Any]) -> dict[str, float]:
        value = float(event["value"])
        return {"value": value, "is_even": float(value % 2 == 0)}

    def predict_proba_one(self, event_id: str, event: dict[str, Any]) -> dict[bool, float]:
        del event_id
        return self.model.predict_proba_one(self.transform(event))

    def learn_one(self, event_id: str, event: dict[str, Any], label: int) -> None:
        del event_id
        self.model.learn_one(self.transform(event), bool(label))
