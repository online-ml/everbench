"""A small online logistic-regression model for the dummy task."""

from __future__ import annotations

from river import linear_model, optim, preprocessing


class DummyLogisticRegression:
    """Standardize the two supplied features and learn a binary probability."""

    def __init__(self) -> None:
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.05))

    def predict_proba_one(self, features: dict[str, float], *, event_id: str | None = None) -> dict[bool, float]:
        del event_id
        return self.model.predict_proba_one(features)

    def learn_one(self, features: dict[str, float], label: int) -> None:
        self.model.learn_one(features, bool(label))
