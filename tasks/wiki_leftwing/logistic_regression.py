"""Feature-engineered online logistic regression for the Wikipedia task."""

from __future__ import annotations

from math import log1p

from river import linear_model, optim, preprocessing


class WikiFeatureLogisticRegression:
    """Learn from the task's frozen edit metadata with stable derived features."""

    def __init__(self) -> None:
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.03))

    @staticmethod
    def transform(features: dict[str, float]) -> dict[str, float]:
        anonymous = float(features.get("anonymous", 0.0))
        comment_length = max(float(features.get("comment_length", 0.0)), 0.0)
        title_length = max(float(features.get("title_length", 0.0)), 0.0)
        byte_change = float(features.get("byte_change", 0.0))
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

    def predict_proba_one(self, features: dict[str, float]) -> dict[bool, float]:
        return self.model.predict_proba_one(self.transform(features))

    def learn_one(self, features: dict[str, float], y: int) -> None:
        self.model.learn_one(self.transform(features), bool(y))
