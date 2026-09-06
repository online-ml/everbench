from __future__ import annotations

from typing import Any

import pytest
from river import base, metrics

from everbench.auto import (
    AutoClassifier,
    Candidate,
    ConstraintResult,
    Evaluation,
    MetricConstraint,
    Objective,
)


class CountingClassifier(base.Classifier):
    def __init__(self, positive_probability: float = 0.5) -> None:
        self.positive_probability = positive_probability
        self.examples = 0

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, Any], **kwargs: Any
    ) -> dict[base.typing.ClfTarget, float]:
        del x, kwargs
        return {0: 1.0 - self.positive_probability, 1: self.positive_probability}

    def learn_one(self, x: dict[Any, Any], y: Any, w: float = 1.0) -> None:
        del x, y, w
        self.examples += 1


class FailingClassifier(CountingClassifier):
    def learn_one(self, x: dict[Any, Any], y: Any, w: float = 1.0) -> None:
        del x, y, w
        raise RuntimeError("learning failed")


def test_auto_classifier_delegates_and_only_retains_history_when_opted_in() -> None:
    without_history = AutoClassifier(
        model=CountingClassifier(positive_probability=0.75),
        objective=Objective(metrics.LogLoss()),
    )
    assert without_history.predict_proba_one({"value": 1})[1] == 0.75
    without_history.learn_one({"value": 1}, 1)
    assert isinstance(without_history.model, CountingClassifier)
    assert without_history.model.examples == 1
    assert without_history.research_snapshot().history == ()

    with_history = AutoClassifier(
        model=CountingClassifier(positive_probability=0.75),
        objective=Objective(metrics.LogLoss()),
        history_capacity=2,
    )
    x = {"value": [1]}
    with_history.learn_one(x, 1, w=2.0)
    x["value"].append(2)

    snapshot = with_history.research_snapshot()
    assert snapshot.observations_seen == 1
    assert snapshot.history[0].x == {"value": [1]}
    assert snapshot.history[0].y == 1
    assert snapshot.history[0].prediction[1] == 0.75
    assert snapshot.history[0].learn_kwargs == {"w": 2.0}

    snapshot.history[0].x["value"].append(3)
    assert with_history.research_snapshot().history[0].x == {"value": [1]}


def test_failed_learning_does_not_create_research_evidence() -> None:
    model = AutoClassifier(FailingClassifier(), Objective(metrics.LogLoss()), history_capacity=10_000)

    with pytest.raises(RuntimeError, match="learning failed"):
        model.learn_one({"value": 1}, 1)

    snapshot = model.research_snapshot()
    assert snapshot.observations_seen == 0
    assert snapshot.history == ()
    assert snapshot.current_score == 0.0


def test_research_snapshot_is_detached_from_live_model_and_objective() -> None:
    source_metric = metrics.LogLoss()
    source_context = {"problem_description": "Predict a delayed outcome", "labels": [0, 1]}
    model = AutoClassifier(
        CountingClassifier(), Objective(source_metric), history_capacity=10_000, context=source_context
    )
    source_context["labels"].append(2)
    model.learn_one({"value": 1}, 1)

    snapshot = model.research_snapshot()
    snapshot.champion.learn_one({"value": 2}, 0)
    snapshot.objective.metric.update(1, {0: 0.9, 1: 0.1})
    snapshot.context["labels"].append(3)
    source_metric.update(1, {0: 0.9, 1: 0.1})

    fresh = model.research_snapshot()
    assert isinstance(model.model, CountingClassifier)
    assert isinstance(fresh.champion, CountingClassifier)
    assert model.model.examples == 1
    assert fresh.champion.examples == 1
    assert fresh.objective.metric.get() == 0.0
    assert fresh.context == {"problem_description": "Predict a delayed outcome", "labels": [0, 1]}


def test_consider_promotes_only_fresh_candidates_with_sufficient_evidence() -> None:
    objective = Objective(
        metrics.Accuracy(),
        min_improvement=0.1,
        min_observations=10,
        required_constraints=("latency",),
    )
    model = AutoClassifier(CountingClassifier(positive_probability=0.25), objective)
    candidate_model = CountingClassifier(positive_probability=0.8)
    candidate_model.learn_one({}, 1)
    candidate = Candidate(candidate_model, parent_generation=0, hypothesis="favor the positive class")

    assert not model.consider(
        candidate,
        Evaluation(
            champion_score=0.5,
            candidate_score=0.7,
            observations=9,
            constraints=(ConstraintResult("latency", True),),
        ),
    )
    assert not model.consider(
        candidate,
        Evaluation(champion_score=0.5, candidate_score=0.7, observations=10),
    )
    assert model.generation == 0

    assert model.consider(
        candidate,
        Evaluation(
            champion_score=0.5,
            candidate_score=0.7,
            observations=10,
            constraints=(ConstraintResult("latency", True),),
        ),
    )
    assert model.generation == 1
    assert isinstance(model.model, CountingClassifier)
    assert model.model.positive_probability == 0.8
    assert model.model.examples == 1
    assert model.score == 0.0

    candidate_model.learn_one({}, 1)
    assert model.model.examples == 1

    with pytest.raises(ValueError, match="targets generation 0, current generation is 1"):
        model.consider(
            candidate,
            Evaluation(
                champion_score=0.5,
                candidate_score=0.7,
                observations=10,
                constraints=(ConstraintResult("latency", True),),
            ),
        )


def test_objective_uses_the_metric_direction_and_rejects_non_finite_scores() -> None:
    minimize = Objective(metrics.LogLoss(), min_improvement=0.05)
    assert minimize.accepts(Evaluation(champion_score=0.5, candidate_score=0.4, observations=1))
    assert not minimize.accepts(Evaluation(champion_score=0.5, candidate_score=0.46, observations=1))
    assert not Objective(metrics.LogLoss()).accepts(Evaluation(champion_score=0.5, candidate_score=0.5, observations=1))
    assert not minimize.accepts(Evaluation(champion_score=float("nan"), candidate_score=0.4, observations=1))

    maximize = Objective(metrics.Accuracy(), min_improvement=0.05)
    assert maximize.accepts(Evaluation(champion_score=0.5, candidate_score=0.6, observations=1))
    assert not maximize.accepts(Evaluation(champion_score=0.5, candidate_score=0.54, observations=1))


def test_objective_rejects_ambiguous_constraint_results() -> None:
    objective = Objective(metrics.Accuracy(), required_constraints=("latency",))
    evaluation = Evaluation(
        champion_score=0.5,
        candidate_score=0.6,
        observations=1,
        constraints=(ConstraintResult("latency", False), ConstraintResult("latency", True)),
    )

    assert not objective.accepts(evaluation)


def test_secondary_metric_constraint_is_immutable_and_required() -> None:
    objective = Objective(
        metrics.LogLoss(),
        metric_constraints=(MetricConstraint("auc", metrics.ROCAUC(), max_regression=0.01),),
    )
    assert objective.accepts(
        Evaluation(
            champion_score=0.5,
            candidate_score=0.4,
            observations=1,
            constraints=(ConstraintResult("auc", True),),
        )
    )
    assert not objective.accepts(Evaluation(champion_score=0.5, candidate_score=0.4, observations=1))


def test_in_memory_history_is_bounded() -> None:
    model = AutoClassifier(CountingClassifier(), Objective(metrics.LogLoss()), history_capacity=2)

    for value in range(3):
        model.learn_one({"value": value}, value % 2)

    assert [observation.sequence for observation in model.research_snapshot().history] == [2, 3]
