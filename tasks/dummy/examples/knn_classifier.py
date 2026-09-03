"""An online k-nearest-neighbours classifier for the dummy task."""

from __future__ import annotations

from river import neighbors


class DummyKNNClassifier:
    """Keep a bounded recent window of the dummy task's two features."""

    def __init__(self) -> None:
        self.model = neighbors.KNNClassifier(n_neighbors=5, engine=neighbors.LazySearch(window_size=1_000))

    def predict_proba_one(self, features: dict[str, float], *, event_id: str | None = None) -> dict[bool, float]:
        del event_id
        return self.model.predict_proba_one(features)

    def learn_one(self, features: dict[str, float], label: int) -> None:
        self.model.learn_one(features, bool(label))
