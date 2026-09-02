"""A minimal online baseline for the dummy task."""

from __future__ import annotations


class RevertRate:
    """Predict a Laplace-smoothed positive-label rate."""

    def __init__(self, n: int = 0, positive: int = 0):
        self.n = n
        self.positive = positive

    def predict_one(self, features: dict[str, float]) -> float:
        del features
        return (self.positive + 1) / (self.n + 2)

    def learn_one(self, features: dict[str, float], y: int) -> None:
        del features
        self.n += 1
        self.positive += y
