from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from river import base, metrics

from everbench.auto import (
    Objective,
    TemporalObservation,
    evaluate_temporally,
    temporal_split,
)
from everbench.auto.code_researcher import summarize_research


class LabelCountClassifier(base.Classifier):
    def __init__(self) -> None:
        self.learned = 0

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        del x, kwargs
        positive = 0.1 if self.learned == 0 else 0.9
        return {0: 1 - positive, 1: positive}

    def learn_one(self, x: dict[Any, Any], y: Any) -> None:
        del x, y
        self.learned += 1


def observations(count: int = 6) -> tuple[TemporalObservation, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(
        TemporalObservation(
            observation_id=str(index),
            sequence=index,
            x={"value": index},
            y=1,
            available_at=start + timedelta(seconds=index),
            label_available_at=start + timedelta(seconds=index + 10),
        )
        for index in range(count)
    )


def test_temporal_split_reserves_the_latest_event_cohort() -> None:
    split = temporal_split(tuple(reversed(observations())), promotion_observations=2, min_research_observations=3)

    assert [row.observation_id for row in split.research] == ["0", "1", "2", "3"]
    assert [row.observation_id for row in split.promotion] == ["4", "5"]

    with pytest.raises(ValueError, match="at least 7 complete observations"):
        temporal_split(observations(), promotion_observations=4, min_research_observations=3)


def test_research_summary_requires_an_explicit_raw_example_opt_in() -> None:
    rows = observations()

    assert "raw_examples" not in summarize_research(rows)
    visible = summarize_research(rows, include_raw_examples=True, max_examples=2)
    assert [item["event"] for item in visible["raw_examples"]] == [{"value": 0}, {"value": 3}]


def test_evaluation_preserves_delayed_feedback() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = (
        TemporalObservation("research", 1, {}, 1, start, start + timedelta(seconds=3)),
        TemporalObservation(
            "promotion",
            2,
            {},
            1,
            start + timedelta(seconds=1),
            start + timedelta(seconds=4),
        ),
    )
    split = temporal_split(rows, promotion_observations=1)

    outcome = evaluate_temporally(
        LabelCountClassifier(),
        LabelCountClassifier(),
        split,
        Objective(metrics.LogLoss()),
    )

    assert outcome.evaluation.candidate_score == pytest.approx(-__import__("math").log(0.1))
    assert outcome.evaluation.champion_score == outcome.evaluation.candidate_score
    assert isinstance(outcome.trained_candidate, LabelCountClassifier)
    assert outcome.trained_candidate.learned == 2
