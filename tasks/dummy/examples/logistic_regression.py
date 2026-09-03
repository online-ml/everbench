"""A small online logistic-regression model for the dummy task."""

from __future__ import annotations

from typing import Any

from river import linear_model, optim, preprocessing


class DummyLogisticRegression:
    """Extract two features from each event and learn a binary probability."""

    def __init__(self) -> None:
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.05))

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
